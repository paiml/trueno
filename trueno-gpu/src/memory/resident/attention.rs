//! GPU-resident attention operations for transformer architectures.
//!
//! This module contains batched and incremental multi-head attention implementations
//! that operate entirely on GPU with zero intermediate host transfers.
//!
//! # Implementations
//!
//! - `batched_multihead_attention` - Standard per-head attention processing
//! - `batched_multihead_attention_optimized` - Optimized batched attention (WAPR-PERF-008)
//! - `incremental_attention_gpu` - Autoregressive decoder attention (WAPR-PERF-013)
//! - `kv_cache_scatter_gpu` - KV cache scatter operation

#[cfg(feature = "cuda")]
use super::GpuResidentTensor;
#[cfg(feature = "cuda")]
use crate::driver::{CudaContext, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use crate::error::Result;
#[cfg(feature = "cuda")]
use crate::kernels::Kernel;
#[cfg(feature = "cuda")]
use super::cache::get_or_compile_kernel;

// Note: Batched4DGemmKernel available for optimized multi-head attention (future)
#[allow(unused_imports)]
#[cfg(feature = "cuda")]
use crate::kernels::Batched4DGemmKernel;

// ============================================================================
// Batched Multi-Head Attention (GPU-Resident)
// ============================================================================

/// Batched multi-head attention that stays on GPU
///
/// Computes: output = softmax(Q @ K^T / sqrt(d_k)) @ V
/// All operations happen on GPU with ZERO intermediate host transfers.
///
/// This simplified version uses standard matmul operations per head.
/// The key benefit is ZERO host↔device transfers during computation.
///
/// # Arguments
///
/// * `ctx` - CUDA context
/// * `q` - Query tensor [seq_len * d_model] flattened
/// * `k` - Key tensor [seq_len * d_model] flattened
/// * `v` - Value tensor [seq_len * d_model] flattened
/// * `n_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
/// * `seq_len` - Sequence length
///
/// # Returns
///
/// Output tensor [seq_len * d_model] still on GPU
///
/// # Citations
///
/// - [Vaswani2017] Attention Is All You Need - original multi-head attention
/// - [Dao2022] FlashAttention - fused attention for memory efficiency
#[cfg(feature = "cuda")]
pub fn batched_multihead_attention(
    ctx: &CudaContext,
    q: &GpuResidentTensor<f32>,
    k: &GpuResidentTensor<f32>,
    v: &GpuResidentTensor<f32>,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
) -> Result<GpuResidentTensor<f32>> {
    let d_model = (n_heads * head_dim) as usize;
    let expected_size = (seq_len as usize) * d_model;

    // Validate input dimensions
    if q.len() != expected_size {
        return Err(crate::GpuError::InvalidParameter(format!(
            "Q has {} elements, expected {} (seq_len={}, d_model={})",
            q.len(),
            expected_size,
            seq_len,
            d_model
        )));
    }
    if k.len() != expected_size || v.len() != expected_size {
        return Err(crate::GpuError::InvalidParameter(
            "K and V must have same size as Q".to_string(),
        ));
    }

    // Proper multi-head attention: process each head independently
    // This involves more kernel launches but produces correct results.
    // Batched optimization can be added later.
    //
    // For each head h:
    //   1. Extract Q_h, K_h, V_h from interleaved layout
    //   2. Transpose K_h: [seq_len, head_dim] -> [head_dim, seq_len]
    //   3. Scores = Q_h @ K_h^T: [seq_len, head_dim] @ [head_dim, seq_len] = [seq_len, seq_len]
    //   4. Scale and softmax
    //   5. Output_h = Attn @ V_h: [seq_len, seq_len] @ [seq_len, head_dim] = [seq_len, head_dim]
    //   6. Copy Output_h to output at head h position

    let scale = 1.0 / (head_dim as f32).sqrt();

    // Allocate output buffer
    let output_buffer = GpuBuffer::new(ctx, expected_size)?;

    let debug_attn = std::env::var("WHISPER_DEBUG_ATTN").is_ok();

    for h in 0..n_heads {
        // Extract head h from Q, K, V
        let q_h = extract_single_head(ctx, q, h, seq_len, n_heads, head_dim)?;
        let k_h = extract_single_head(ctx, k, h, seq_len, n_heads, head_dim)?;
        let v_h = extract_single_head(ctx, v, h, seq_len, n_heads, head_dim)?;

        if debug_attn && h == 0 {
            let q_host = q_h.peek_host()?;
            let k_host = k_h.peek_host()?;
            let v_host = v_h.peek_host()?;
            eprintln!("[DEBUG-ATTN] head 0: Q_h mean={:.6}, K_h mean={:.6}, V_h mean={:.6}",
                q_host.iter().sum::<f32>() / q_host.len() as f32,
                k_host.iter().sum::<f32>() / k_host.len() as f32,
                v_host.iter().sum::<f32>() / v_host.len() as f32);
        }

        // Transpose K_h: [seq_len, head_dim] -> [head_dim, seq_len]
        let kt_h = transpose_matrix(ctx, &k_h.buffer, seq_len, head_dim)?;
        let kt_tensor = GpuResidentTensor::from_buffer_internal(kt_h, 1);

        // Q_h @ K_h^T: [seq_len, head_dim] @ [head_dim, seq_len] = [seq_len, seq_len]
        let scores_h = q_h.matmul(ctx, &kt_tensor, seq_len, seq_len, head_dim)?;

        if debug_attn && h == 0 {
            let scores_host = scores_h.peek_host()?;
            eprintln!("[DEBUG-ATTN] head 0: scores mean={:.6}, max={:.6}",
                scores_host.iter().sum::<f32>() / scores_host.len() as f32,
                scores_host.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }

        // Scale and softmax
        let scaled_h = scores_h.scale(ctx, scale)?;
        let attn_h = scaled_h.softmax(ctx, seq_len)?;

        if debug_attn && h == 0 {
            let attn_host = attn_h.peek_host()?;
            // Check first row sums to ~1
            let first_row_sum: f32 = attn_host[..seq_len as usize].iter().sum();
            eprintln!("[DEBUG-ATTN] head 0: attn first_row_sum={:.6}, mean={:.6}",
                first_row_sum,
                attn_host.iter().sum::<f32>() / attn_host.len() as f32);
        }

        // Attn @ V_h: [seq_len, seq_len] @ [seq_len, head_dim] = [seq_len, head_dim]
        let out_h = attn_h.matmul(ctx, &v_h, seq_len, head_dim, seq_len)?;

        if debug_attn && h == 0 {
            let out_host = out_h.peek_host()?;
            eprintln!("[DEBUG-ATTN] head 0: out mean={:.6}, std={:.6}",
                out_host.iter().sum::<f32>() / out_host.len() as f32,
                (out_host.iter().map(|v| v.powi(2)).sum::<f32>() / out_host.len() as f32).sqrt());
        }

        // Copy out_h to output at head h position
        copy_head_to_output(ctx, &output_buffer, &out_h, h, seq_len, n_heads, head_dim)?;
    }

    Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
}

/// Batched multi-head attention optimized for all heads in parallel (WAPR-PERF-008)
///
/// Reduces kernel launches from 54 (6 heads × 9 ops) to 9 by batching all heads.
/// Uses grid.z = n_heads for parallel head processing.
///
/// # Memory Layout
/// - Input Q, K, V: [seq_len, d_model] interleaved (d_model = n_heads * head_dim)
/// - Internal: [n_heads, seq_len, head_dim] batched for parallel processing
/// - Output: [seq_len, d_model] interleaved
#[cfg(feature = "cuda")]
pub fn batched_multihead_attention_optimized(
    ctx: &CudaContext,
    q: &GpuResidentTensor<f32>,
    k: &GpuResidentTensor<f32>,
    v: &GpuResidentTensor<f32>,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
) -> Result<GpuResidentTensor<f32>> {
    let d_model = (n_heads * head_dim) as usize;
    let expected_size = (seq_len as usize) * d_model;

    // Validate input dimensions
    if q.len() != expected_size || k.len() != expected_size || v.len() != expected_size {
        return Err(crate::GpuError::InvalidParameter(format!(
            "Q/K/V size mismatch: expected {} (seq_len={}, d_model={})",
            expected_size, seq_len, d_model
        )));
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let batch = n_heads;  // Each head is a "batch" item

    // Step 1: Convert interleaved -> batched for Q, K, V
    // [seq_len, n_heads * head_dim] -> [n_heads, seq_len, head_dim]
    let q_batched = interleaved_to_batched_all(ctx, q, seq_len, n_heads, head_dim)?;
    let k_batched = interleaved_to_batched_all(ctx, k, seq_len, n_heads, head_dim)?;
    let v_batched = interleaved_to_batched_all(ctx, v, seq_len, n_heads, head_dim)?;

    // Step 2: Transpose K for all heads
    // [n_heads, seq_len, head_dim] -> [n_heads, head_dim, seq_len]
    let kt_batched = batched_transpose_all(ctx, &k_batched, batch, seq_len, head_dim)?;

    // Step 3: Q @ K^T for all heads using BatchedGemmKernel
    // [n_heads, seq_len, head_dim] @ [n_heads, head_dim, seq_len] -> [n_heads, seq_len, seq_len]
    let scores = batched_gemm(ctx, &q_batched, &kt_batched, batch, seq_len, seq_len, head_dim)?;

    // Step 4: Scale all scores
    let total_scores = batch * seq_len * seq_len;
    let scaled_scores = batched_scale_all(ctx, &scores, scale, total_scores)?;

    // Step 5: Softmax for all heads (n_heads * seq_len rows of seq_len elements each)
    let attn = batched_softmax_all(ctx, &scaled_scores, batch * seq_len, seq_len)?;

    // Step 6: Attn @ V for all heads
    // [n_heads, seq_len, seq_len] @ [n_heads, seq_len, head_dim] -> [n_heads, seq_len, head_dim]
    let out_batched = batched_gemm(ctx, &attn, &v_batched, batch, seq_len, head_dim, seq_len)?;

    // Step 7: Convert batched -> interleaved
    // [n_heads, seq_len, head_dim] -> [seq_len, n_heads * head_dim]
    let output = batched_to_interleaved_all(ctx, &out_batched, seq_len, n_heads, head_dim)?;

    Ok(output)
}

// ============================================================================
// Helper Functions for Batched Attention
// ============================================================================

/// Convert interleaved tensor to batched layout for all heads
#[cfg(feature = "cuda")]
fn interleaved_to_batched_all(
    ctx: &CudaContext,
    input: &GpuResidentTensor<f32>,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::{InterleavedToBatchedKernel, Kernel};

    let total_size = (seq_len * n_heads * head_dim) as usize;
    let output = GpuBuffer::new(ctx, total_size)?;

    let kernel = InterleavedToBatchedKernel::new(seq_len, n_heads, head_dim);
    let ptx = kernel.emit_ptx();
    let cache_key = format!("interleaved_to_batched:{}:{}:{}", seq_len, n_heads, head_dim);
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
        let mut module = module_arc.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
        })?;
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// Transpose all matrices in batch using grid.z
#[cfg(feature = "cuda")]
fn batched_transpose_all(
    ctx: &CudaContext,
    input: &GpuResidentTensor<f32>,
    batch: u32,
    rows: u32,
    cols: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::{BatchedTransposeKernel, Kernel};

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
        grid: (blocks_x, 1, batch),  // z-dimension for batch/heads
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
        let mut module = module_arc.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
        })?;
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// Batched GEMM: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
#[cfg(feature = "cuda")]
fn batched_gemm(
    ctx: &CudaContext,
    a: &GpuResidentTensor<f32>,
    b: &GpuResidentTensor<f32>,
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::{BatchedGemmKernel, Kernel};

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
        let mut module = module_arc.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
        })?;
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// Scale all elements in a tensor
#[cfg(feature = "cuda")]
fn batched_scale_all(
    ctx: &CudaContext,
    input: &GpuResidentTensor<f32>,
    scale: f32,
    n: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::{BatchedScaleKernel, Kernel};

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
        let mut module = module_arc.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
        })?;
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// Softmax for all rows in all batches
#[cfg(feature = "cuda")]
fn batched_softmax_all(
    ctx: &CudaContext,
    input: &GpuResidentTensor<f32>,
    total_rows: u32,
    row_size: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::{BatchedSoftmaxKernel, Kernel};

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
        let mut module = module_arc.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
        })?;
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// Convert batched tensor back to interleaved layout
#[cfg(feature = "cuda")]
fn batched_to_interleaved_all(
    ctx: &CudaContext,
    input: &GpuResidentTensor<f32>,
    seq_len: u32,
    n_heads: u32,
    head_dim: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::{BatchedToInterleavedKernel, Kernel};

    let total_size = (seq_len * n_heads * head_dim) as usize;
    let output = GpuBuffer::new(ctx, total_size)?;

    let kernel = BatchedToInterleavedKernel::new(seq_len, n_heads, head_dim);
    let ptx = kernel.emit_ptx();
    let cache_key = format!("batched_to_interleaved:{}:{}:{}", seq_len, n_heads, head_dim);
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
        let mut module = module_arc.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
        })?;
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// Transpose a matrix on GPU: [rows, cols] -> [cols, rows]
#[cfg(feature = "cuda")]
fn transpose_matrix(
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
        let mut module = module_arc.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
        })?;
        unsafe {
            stream.launch_kernel(&mut module, transpose.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(output)
}

/// Extract single head from interleaved tensor
#[cfg(feature = "cuda")]
fn extract_single_head(
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
        let mut module = module_arc.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
        })?;
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
}

/// Copy single head output to interleaved output buffer
#[cfg(feature = "cuda")]
fn copy_head_to_output(
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
        let mut module = module_arc.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
        })?;
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }
    stream.synchronize()?;

    Ok(())
}

// ============================================================================
// Incremental Attention (Autoregressive Decoder)
// ============================================================================

/// WAPR-PERF-013: Incremental attention for autoregressive decoding
///
/// Computes attention for a single query token against the full KV cache.
/// Designed for GPU-resident KV caches with zero D2H transfers.
///
/// # Memory Layout (Head-First)
///
/// - `q`: `[n_heads, head_dim]` - query for current position (1 token)
/// - `k_cache`: `[n_heads, max_seq_len, head_dim]` - cached keys (head-first)
/// - `v_cache`: `[n_heads, max_seq_len, head_dim]` - cached values (head-first)
/// - output: `[n_heads, head_dim]` - attention output
///
/// # Arguments
///
/// * `ctx` - CUDA context
/// * `q` - Query tensor `[n_heads * head_dim]`
/// * `k_cache` - Key cache `[n_heads * max_seq_len * head_dim]`
/// * `v_cache` - Value cache `[n_heads * max_seq_len * head_dim]`
/// * `n_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
/// * `seq_len` - Current sequence length (tokens in cache)
/// * `max_seq_len` - Maximum sequence length (cache capacity)
///
/// # Returns
///
/// Output tensor `[n_heads * head_dim]` (same shape as Q)
#[cfg(feature = "cuda")]
pub fn incremental_attention_gpu(
    ctx: &CudaContext,
    q: &GpuResidentTensor<f32>,
    k_cache: &GpuResidentTensor<f32>,
    v_cache: &GpuResidentTensor<f32>,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::{IncrementalAttentionKernel, Kernel};

    // Validate Q size: [n_heads, head_dim]
    let q_expected = (n_heads * head_dim) as usize;
    if q.len() != q_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "Q has {} elements, expected {} (n_heads={}, head_dim={})",
            q.len(),
            q_expected,
            n_heads,
            head_dim
        )));
    }

    // Validate K/V cache size: [n_heads, max_seq_len, head_dim]
    let cache_expected = (n_heads * max_seq_len * head_dim) as usize;
    if k_cache.len() != cache_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "K cache has {} elements, expected {} (n_heads={}, max_seq_len={}, head_dim={})",
            k_cache.len(),
            cache_expected,
            n_heads,
            max_seq_len,
            head_dim
        )));
    }
    if v_cache.len() != cache_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "V cache has {} elements, expected {}",
            v_cache.len(),
            cache_expected
        )));
    }

    // Validate seq_len <= max_seq_len
    if seq_len > max_seq_len {
        return Err(crate::GpuError::InvalidParameter(format!(
            "seq_len ({}) exceeds max_seq_len ({})",
            seq_len, max_seq_len
        )));
    }

    // Handle empty sequence (no attention needed)
    if seq_len == 0 {
        // Return zeros
        let zeros = vec![0.0f32; q_expected];
        return GpuResidentTensor::from_host(ctx, &zeros);
    }

    // Allocate output: [n_heads, head_dim]
    let output = GpuBuffer::new(ctx, q_expected)?;

    // Build and cache kernel
    let kernel = IncrementalAttentionKernel::new(max_seq_len, head_dim, n_heads);
    let ptx = kernel.emit_ptx();
    let cache_key = format!(
        "incremental_attention:{}:{}:{}",
        max_seq_len, head_dim, n_heads
    );
    let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
    let stream = CudaStream::new(ctx)?;

    // Launch config: one block per head, one warp per block
    let config = LaunchConfig {
        grid: (n_heads, 1, 1),
        block: (32, 1, 1), // One warp
        shared_mem: 0,
    };

    // Prepare kernel arguments
    let q_ptr = q.as_ptr();
    let k_ptr = k_cache.as_ptr();
    let v_ptr = v_cache.as_ptr();
    let out_ptr = output.as_ptr();
    let seq_len_val = seq_len;

    let mut args: [*mut std::ffi::c_void; 5] = [
        std::ptr::addr_of!(q_ptr) as *mut _,
        std::ptr::addr_of!(k_ptr) as *mut _,
        std::ptr::addr_of!(v_ptr) as *mut _,
        std::ptr::addr_of!(out_ptr) as *mut _,
        std::ptr::addr_of!(seq_len_val) as *mut _,
    ];

    // Launch kernel (lock the cached module)
    {
        let mut module = module_arc.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
        })?;
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }

    // WAPR-PERF-014: MUST sync before returning since stream goes out of scope
    // Without sync, kernel may not complete before output is used (UB!)
    stream.synchronize()?;

    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// WAPR-PERF-014: Incremental attention with external stream (no stream creation)
///
/// Same as `incremental_attention_gpu` but uses caller-provided stream instead of
/// creating a new one. This eliminates stream creation overhead (~5-10µs per call).
///
/// # Use Case
///
/// When running attention in a loop (autoregressive decoding), use a single shared
/// stream for all operations to avoid creating ~40 streams per token.
///
/// # Arguments
///
/// * `stream` - Caller-provided CUDA stream (reuse across operations)
/// * Other args same as `incremental_attention_gpu`
#[cfg(feature = "cuda")]
pub fn incremental_attention_gpu_with_stream(
    ctx: &CudaContext,
    q: &GpuResidentTensor<f32>,
    k_cache: &GpuResidentTensor<f32>,
    v_cache: &GpuResidentTensor<f32>,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    stream: &CudaStream,
) -> Result<GpuResidentTensor<f32>> {
    use crate::kernels::{IncrementalAttentionKernel, Kernel};

    // Validate Q size: [n_heads, head_dim]
    let q_expected = (n_heads * head_dim) as usize;
    if q.len() != q_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "Q has {} elements, expected {} (n_heads={}, head_dim={})",
            q.len(), q_expected, n_heads, head_dim
        )));
    }

    // Validate K/V cache size: [n_heads, max_seq_len, head_dim]
    let cache_expected = (n_heads * max_seq_len * head_dim) as usize;
    if k_cache.len() != cache_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "K cache has {} elements, expected {}",
            k_cache.len(), cache_expected
        )));
    }
    if v_cache.len() != cache_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "V cache has {} elements, expected {}",
            v_cache.len(), cache_expected
        )));
    }

    if seq_len > max_seq_len {
        return Err(crate::GpuError::InvalidParameter(format!(
            "seq_len ({}) exceeds max_seq_len ({})", seq_len, max_seq_len
        )));
    }

    // Handle empty sequence
    if seq_len == 0 {
        let zeros = vec![0.0f32; q_expected];
        return GpuResidentTensor::from_host(ctx, &zeros);
    }

    // Allocate output
    let output = GpuBuffer::new(ctx, q_expected)?;

    // Build and cache kernel
    let kernel = IncrementalAttentionKernel::new(max_seq_len, head_dim, n_heads);
    let ptx = kernel.emit_ptx();
    let cache_key = format!("incremental_attention:{}:{}:{}", max_seq_len, head_dim, n_heads);
    let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;

    let config = LaunchConfig {
        grid: (n_heads, 1, 1),
        block: (32, 1, 1),
        shared_mem: 0,
    };

    let q_ptr = q.as_ptr();
    let k_ptr = k_cache.as_ptr();
    let v_ptr = v_cache.as_ptr();
    let out_ptr = output.as_ptr();
    let seq_len_val = seq_len;

    let mut args: [*mut std::ffi::c_void; 5] = [
        std::ptr::addr_of!(q_ptr) as *mut _,
        std::ptr::addr_of!(k_ptr) as *mut _,
        std::ptr::addr_of!(v_ptr) as *mut _,
        std::ptr::addr_of!(out_ptr) as *mut _,
        std::ptr::addr_of!(seq_len_val) as *mut _,
    ];

    {
        let mut module = module_arc.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
        })?;
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }

    // NO SYNC - uses caller's stream for pipelining
    Ok(GpuResidentTensor::from_buffer_internal(output, 1))
}

/// WAPR-PERF-013: Async incremental attention with explicit stream return
///
/// Same as `incremental_attention_gpu` but returns the stream for caller-controlled
/// synchronization. Use this in autoregressive loops to avoid ghost syncs.
///
/// # Point 149 Compliance
///
/// This function launches the kernel without synchronizing. The caller MUST:
/// 1. Chain dependent operations on the same stream, OR
/// 2. Call `stream.synchronize()` before reading the output
///
/// # Returns
///
/// Tuple of (output tensor, stream) - stream must be synchronized before reading output
#[cfg(feature = "cuda")]
pub fn incremental_attention_gpu_async(
    ctx: &CudaContext,
    q: &GpuResidentTensor<f32>,
    k_cache: &GpuResidentTensor<f32>,
    v_cache: &GpuResidentTensor<f32>,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
) -> Result<(GpuResidentTensor<f32>, CudaStream)> {
    use crate::kernels::{IncrementalAttentionKernel, Kernel};

    // Validate Q size: [n_heads, head_dim]
    let q_expected = (n_heads * head_dim) as usize;
    if q.len() != q_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "Q has {} elements, expected {} (n_heads={}, head_dim={})",
            q.len(), q_expected, n_heads, head_dim
        )));
    }

    // Validate K/V cache size: [n_heads, max_seq_len, head_dim]
    let cache_expected = (n_heads * max_seq_len * head_dim) as usize;
    if k_cache.len() != cache_expected || v_cache.len() != cache_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "K/V cache size mismatch: expected {} (n_heads={}, max_seq_len={}, head_dim={})",
            cache_expected, n_heads, max_seq_len, head_dim
        )));
    }

    if seq_len > max_seq_len {
        return Err(crate::GpuError::InvalidParameter(format!(
            "seq_len ({}) exceeds max_seq_len ({})", seq_len, max_seq_len
        )));
    }

    // Handle empty sequence
    if seq_len == 0 {
        let zeros = vec![0.0f32; q_expected];
        let output = GpuResidentTensor::from_host(ctx, &zeros)?;
        let stream = CudaStream::new(ctx)?;
        return Ok((output, stream));
    }

    // Allocate output
    let output = GpuBuffer::new(ctx, q_expected)?;

    // Build and cache kernel
    let kernel = IncrementalAttentionKernel::new(max_seq_len, head_dim, n_heads);
    let ptx = kernel.emit_ptx();
    let cache_key = format!("incremental_attention:{}:{}:{}", max_seq_len, head_dim, n_heads);
    let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
    let stream = CudaStream::new(ctx)?;

    let config = LaunchConfig {
        grid: (n_heads, 1, 1),
        block: (32, 1, 1),
        shared_mem: 0,
    };

    let q_ptr = q.as_ptr();
    let k_ptr = k_cache.as_ptr();
    let v_ptr = v_cache.as_ptr();
    let out_ptr = output.as_ptr();
    let seq_len_val = seq_len;

    let mut args: [*mut std::ffi::c_void; 5] = [
        std::ptr::addr_of!(q_ptr) as *mut _,
        std::ptr::addr_of!(k_ptr) as *mut _,
        std::ptr::addr_of!(v_ptr) as *mut _,
        std::ptr::addr_of!(out_ptr) as *mut _,
        std::ptr::addr_of!(seq_len_val) as *mut _,
    ];

    {
        let mut module = module_arc.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
        })?;
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }

    // NO SYNC - caller controls synchronization (Point 149)
    Ok((GpuResidentTensor::from_buffer_internal(output, 1), stream))
}

/// WAPR-PERF-013: Scatter interleaved K/V to head-first cache slot
///
/// Writes a single position's K or V projection directly into the head-first
/// cache layout without intermediate conversion.
///
/// # Memory Layout
///
/// - Source: `[n_heads * head_dim]` (interleaved, from GEMV output)
/// - Cache: `[n_heads, max_seq_len, head_dim]` (head-first)
/// - Position `pos` is written to `cache[head, pos, :]` for all heads
///
/// # Performance
///
/// - Single kernel launch (no conversion overhead)
/// - Coalesced writes (threads write contiguous elements per head)
/// - Can be chained on same stream as GEMV (no sync needed)
///
/// # Arguments
///
/// * `ctx` - CUDA context
/// * `src` - Source tensor `[n_heads * head_dim]` (interleaved)
/// * `cache` - Target cache buffer `[n_heads * max_seq_len * head_dim]`
/// * `pos` - Sequence position to write
/// * `n_heads` - Number of attention heads
/// * `head_dim` - Dimension per head
/// * `max_seq_len` - Maximum sequence length (cache capacity)
/// * `stream` - CUDA stream for async execution
#[cfg(feature = "cuda")]
pub fn kv_cache_scatter_gpu(
    ctx: &CudaContext,
    src: &GpuResidentTensor<f32>,
    cache: &mut GpuResidentTensor<f32>,
    pos: u32,
    n_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    stream: &CudaStream,
) -> Result<()> {
    use crate::kernels::{KvCacheScatterKernel, Kernel};

    // Validate source size
    let src_expected = (n_heads * head_dim) as usize;
    if src.len() != src_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "Source has {} elements, expected {} (n_heads={}, head_dim={})",
            src.len(), src_expected, n_heads, head_dim
        )));
    }

    // Validate cache size
    let cache_expected = (n_heads * max_seq_len * head_dim) as usize;
    if cache.len() != cache_expected {
        return Err(crate::GpuError::InvalidParameter(format!(
            "Cache has {} elements, expected {} (n_heads={}, max_seq_len={}, head_dim={})",
            cache.len(), cache_expected, n_heads, max_seq_len, head_dim
        )));
    }

    // Validate position
    if pos >= max_seq_len {
        return Err(crate::GpuError::InvalidParameter(format!(
            "Position {} >= max_seq_len {}", pos, max_seq_len
        )));
    }

    // Build and cache kernel
    let kernel = KvCacheScatterKernel::new(n_heads, head_dim, max_seq_len);
    let ptx = kernel.emit_ptx();
    let cache_key = format!("kv_scatter:{}:{}:{}", n_heads, head_dim, max_seq_len);
    let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;

    // Launch config: one block per head, head_dim threads per block
    let config = LaunchConfig {
        grid: (n_heads, 1, 1),
        block: (head_dim.min(256), 1, 1), // Cap at 256 threads
        shared_mem: 0,
    };

    let src_ptr = src.as_ptr();
    let cache_ptr = cache.as_ptr();

    let mut args: [*mut std::ffi::c_void; 5] = [
        std::ptr::addr_of!(src_ptr) as *mut _,
        std::ptr::addr_of!(cache_ptr) as *mut _,
        std::ptr::addr_of!(pos) as *mut _,
        std::ptr::addr_of!(head_dim) as *mut _,
        std::ptr::addr_of!(max_seq_len) as *mut _,
    ];

    {
        let mut module = module_arc.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
        })?;
        unsafe {
            stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
        }
    }

    // NO SYNC - caller chains operations (Point 149)
    Ok(())
}
