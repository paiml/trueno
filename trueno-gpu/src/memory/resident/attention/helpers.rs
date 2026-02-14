//! Helper functions for batched attention operations.
//!
//! Contains GPU kernel wrappers for layout conversion, transpose, GEMM,
//! scaling, softmax, and head extraction/copy operations.

#[cfg(feature = "cuda")]
use super::super::cache::get_or_compile_kernel;
#[cfg(feature = "cuda")]
use super::super::GpuResidentTensor;
#[cfg(feature = "cuda")]
use crate::driver::{CudaContext, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use crate::error::Result;
#[cfg(feature = "cuda")]
use crate::kernels::Kernel;

// ============================================================================
// Helper Functions for Batched Attention
// ============================================================================

/// Convert interleaved tensor to batched layout for all heads
#[cfg(feature = "cuda")]
pub(super) fn interleaved_to_batched_all(
    ctx: &CudaContext,
    input: &GpuResidentTensor<f32>,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::InterleavedToBatchedKernel;

    let total_size = (seq_len * n_heads * head_dim) as usize;
    let output = GpuBuffer::new(ctx, total_size)?;

    let kernel = InterleavedToBatchedKernel::new(seq_len, n_heads, head_dim);
    let ptx = kernel.emit_ptx();
    let cache_key = format!(
        "interleaved_to_batched:{}:{}:{}",
        seq_len, n_heads, head_dim
    );
    let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
    let stream = CudaStream::new(ctx)?;

    let threads = 256u32;
    let blocks = (total_size as u32 + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
    ];

    {
        let mut module = module_arc
            .lock()
            .map_err(|e| crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e)))?;
        // SAFETY: Kernel launch with validated buffer sizes and correct argument layout.
        // input_ptr and output_ptr point to valid GPU allocations of total_size elements.
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// Transpose all matrices in batch using grid.z
#[cfg(feature = "cuda")]
pub(super) fn batched_transpose_all(
    ctx: &CudaContext,
    input: &GpuResidentTensor<f32>,
    batch: u32,
    rows: u32,
    cols: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::BatchedTransposeKernel;

    let total_size = (batch * rows * cols) as usize;
    let output = GpuBuffer::new(ctx, total_size)?;

    let kernel = BatchedTransposeKernel::new(batch, rows, cols);
    let ptx = kernel.emit_ptx();
    let cache_key = format!("batched_transpose:{}:{}:{}", batch, rows, cols);
    let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
    let stream = CudaStream::new(ctx)?;

    let threads = 256u32;
    let elems_per_batch = rows * cols;
    let blocks_x = (elems_per_batch + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks_x, 1, batch), // z-dimension for batch/heads
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(batch) as *mut _,
        std::ptr::addr_of!(rows) as *mut _,
        std::ptr::addr_of!(cols) as *mut _,
    ];

    {
        let mut module = module_arc
            .lock()
            .map_err(|e| crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e)))?;
        // SAFETY: Kernel launch with validated buffer sizes. input and output are
        // batch*rows*cols elements. Args match the BatchedTransposeKernel parameter layout.
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// Batched GEMM: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
#[cfg(feature = "cuda")]
pub(super) fn batched_gemm(
    ctx: &CudaContext,
    a: &GpuResidentTensor<f32>,
    b: &GpuResidentTensor<f32>,
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::BatchedGemmKernel;

    let output_size = (batch * m * n) as usize;
    let output = GpuBuffer::new(ctx, output_size)?;

    // WAPR-PERF-011: Use WMMA Tensor Cores for batched GEMM when dimensions are suitable
    // WMMA 16x16x16 tiles work best when m, n, k are multiples of 16
    // For attention: typical dims are batch=6 heads, m=seq_len, n=64, k=64
    let tile_size = 16u32;
    // WAPR-PERF-014: Allow disabling WMMA for precision debugging
    let force_fp32 = std::env::var("TRUENO_FORCE_FP32_GEMM").is_ok();
    let use_wmma = !force_fp32 && k >= 64 && n >= 16 && m >= 16;

    let (kernel, cache_key, wmma_mode) = if use_wmma {
        let kernel = BatchedGemmKernel::wmma_fp16(batch, m, n, k);
        let key = format!("batched_gemm_wmma_fp16:{}:{}:{}:{}", batch, m, n, k);
        (kernel, key, true)
    } else {
        let kernel = BatchedGemmKernel::naive(batch, m, n, k);
        let key = format!("batched_gemm_naive:{}:{}:{}:{}", batch, m, n, k);
        (kernel, key, false)
    };

    let ptx = kernel.emit_ptx();
    let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
    let stream = CudaStream::new(ctx)?;

    // WAPR-PERF-011: WMMA uses warps (32 threads), naive uses tile blocks
    let (blocks_x, blocks_y, threads_x, threads_y, shared_mem) = if wmma_mode {
        // WMMA: one warp (32 threads) per 16x16 output tile
        let bx = (n + 15) / 16;
        let by = (m + 15) / 16;
        let smem = tile_size * tile_size * 2 * 2; // Two FP16 tiles (A and B)
        (bx, by, 32u32, 1u32, smem)
    } else {
        // Naive: one thread per output element
        let bx = (n + tile_size - 1) / tile_size;
        let by = (m + tile_size - 1) / tile_size;
        (bx, by, tile_size, tile_size, 0u32)
    };
    let config = LaunchConfig {
        grid: (blocks_x, blocks_y, batch),
        block: (threads_x, threads_y, 1),
        shared_mem,
    };

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(a_ptr) as *mut _,
        std::ptr::addr_of!(b_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(batch) as *mut _,
        std::ptr::addr_of!(m) as *mut _,
        std::ptr::addr_of!(n) as *mut _,
        std::ptr::addr_of!(k) as *mut _,
    ];

    {
        let mut module = module_arc
            .lock()
            .map_err(|e| crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e)))?;
        // SAFETY: Kernel launch with validated dimensions. a has batch*m*k elements,
        // b has batch*k*n elements, output has batch*m*n elements. Args match kernel ABI.
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// Scale all elements in a tensor
#[cfg(feature = "cuda")]
pub(super) fn batched_scale_all(
    ctx: &CudaContext,
    input: &GpuResidentTensor<f32>,
    scale: f32,
    n: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::BatchedScaleKernel;

    let output = GpuBuffer::new(ctx, n as usize)?;

    let kernel = BatchedScaleKernel::new(n);
    let ptx = kernel.emit_ptx();
    let cache_key = format!("batched_scale:{}", n);
    let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
    let stream = CudaStream::new(ctx)?;

    let threads = 256u32;
    let blocks = (n + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(scale) as *mut _,
        std::ptr::addr_of!(n) as *mut _,
    ];

    {
        let mut module = module_arc
            .lock()
            .map_err(|e| crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e)))?;
        // SAFETY: Kernel launch with n-element input and output buffers.
        // scale is passed by pointer to match kernel parameter layout.
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// Softmax for all rows in all batches
#[cfg(feature = "cuda")]
pub(super) fn batched_softmax_all(
    ctx: &CudaContext,
    input: &GpuResidentTensor<f32>,
    total_rows: u32,
    row_size: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::BatchedSoftmaxKernel;

    let output_size = (total_rows * row_size) as usize;
    let output = GpuBuffer::new(ctx, output_size)?;

    let kernel = BatchedSoftmaxKernel::new(total_rows, row_size);
    let ptx = kernel.emit_ptx();
    let cache_key = format!("batched_softmax:{}:{}", total_rows, row_size);
    let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
    let stream = CudaStream::new(ctx)?;

    // One warp (32 threads) per row
    let config = LaunchConfig {
        grid: (total_rows, 1, 1),
        block: (32, 1, 1),
        shared_mem: 72,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(total_rows) as *mut _,
        std::ptr::addr_of!(row_size) as *mut _,
    ];

    {
        let mut module = module_arc
            .lock()
            .map_err(|e| crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e)))?;
        // SAFETY: Kernel launch with total_rows*row_size element buffers.
        // One warp per row with 72 bytes shared memory for warp reductions.
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// Convert batched tensor back to interleaved layout
#[cfg(feature = "cuda")]
pub(super) fn batched_to_interleaved_all(
    ctx: &CudaContext,
    input: &GpuResidentTensor<f32>,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::BatchedToInterleavedKernel;

    let total_size = (seq_len * n_heads * head_dim) as usize;
    let output = GpuBuffer::new(ctx, total_size)?;

    let kernel = BatchedToInterleavedKernel::new(seq_len, n_heads, head_dim);
    let ptx = kernel.emit_ptx();
    let cache_key = format!(
        "batched_to_interleaved:{}:{}:{}",
        seq_len, n_heads, head_dim
    );
    let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
    let stream = CudaStream::new(ctx)?;

    let threads = 256u32;
    let blocks = (total_size as u32 + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
    ];

    {
        let mut module = module_arc
            .lock()
            .map_err(|e| crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e)))?;
        // SAFETY: Kernel launch converting batched [n_heads, seq_len, head_dim] to
        // interleaved [seq_len, n_heads * head_dim]. Both buffers are total_size elements.
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// Transpose a matrix on GPU: [rows, cols] -> [cols, rows]
#[cfg(feature = "cuda")]
pub(super) fn transpose_matrix(
    ctx: &CudaContext,
    input: &GpuBuffer<f32>,
    rows: u32,
    cols: u32,
) -> Result<GpuBuffer<f32>> {
    let output_size = (rows * cols) as usize;
    let output = GpuBuffer::new(ctx, output_size)?;

    use crate::kernels::TransposeKernel;
    let transpose = TransposeKernel::new(rows, cols);
    let ptx = transpose.emit_ptx();
    let cache_key = format!("transpose:{}x{}", rows, cols);
    let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
    let stream = CudaStream::new(ctx)?;

    let threads = 256u32;
    let total = rows * cols;
    let blocks = (total + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(rows) as *mut _,
        std::ptr::addr_of!(cols) as *mut _,
    ];

    {
        let mut module = module_arc
            .lock()
            .map_err(|e| crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e)))?;
        // SAFETY: Kernel launch for matrix transpose. input and output both have
        // rows*cols elements. Args pass dimensions for index computation.
        unsafe {
            stream.launch_kernel(&mut module, transpose.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(output)
}

/// Extract single head from interleaved tensor
#[cfg(feature = "cuda")]
pub(super) fn extract_single_head(
    ctx: &CudaContext,
    input: &GpuResidentTensor<f32>,
    head_idx: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
) -> Result<GpuResidentTensor<f32>> {
    let output_size = (seq_len * head_dim) as usize;
    let output_buffer = GpuBuffer::new(ctx, output_size)?;

    use crate::kernels::ExtractSingleHeadKernel;
    let kernel = ExtractSingleHeadKernel::new(seq_len, n_heads, head_dim);
    let ptx = kernel.emit_ptx();
    let cache_key = format!("extract_head:{}:{}:{}", seq_len, n_heads, head_dim);
    let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
    let stream = CudaStream::new(ctx)?;

    let threads = 256u32;
    let blocks = (output_size as u32 + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input.as_ptr();
    let output_ptr = output_buffer.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(head_idx) as *mut _,
    ];

    {
        let mut module = module_arc
            .lock()
            .map_err(|e| crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e)))?;
        // SAFETY: Kernel launch extracting head_idx from interleaved input
        // (seq_len * n_heads * head_dim elements) into output (seq_len * head_dim elements).
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
}

/// Copy single head output to interleaved output buffer
#[cfg(feature = "cuda")]
pub(super) fn copy_head_to_output(
    ctx: &CudaContext,
    output: &GpuBuffer<f32>,
    head_output: &GpuResidentTensor<f32>,
    head_idx: u32,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
) -> Result<()> {
    use crate::kernels::CopySingleHeadKernel;
    let kernel = CopySingleHeadKernel::new(seq_len, n_heads, head_dim);
    let ptx = kernel.emit_ptx();
    let cache_key = format!("copy_head:{}:{}:{}", seq_len, n_heads, head_dim);
    let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
    let stream = CudaStream::new(ctx)?;

    let input_size = (seq_len * head_dim) as usize;
    let threads = 256u32;
    let blocks = (input_size as u32 + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = head_output.as_ptr();
    let output_ptr = output.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(head_idx) as *mut _,
    ];

    {
        let mut module = module_arc
            .lock()
            .map_err(|e| crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e)))?;
        // SAFETY: Kernel launch copying head_output (seq_len * head_dim elements) into
        // the interleaved output buffer at head_idx position. Output buffer is
        // seq_len * n_heads * head_dim elements total.
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(())
}
