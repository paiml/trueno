//! Layout conversion helpers for batched attention operations.
//!
//! Functions for converting between interleaved and batched tensor layouts,
//! and batched transpose operations.

#[cfg(feature = "cuda")]
use super::super::super::cache::get_or_compile_kernel;
#[cfg(feature = "cuda")]
use super::super::super::GpuResidentTensor;
#[cfg(feature = "cuda")]
use crate::driver::{CudaContext, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use crate::error::Result;
#[cfg(feature = "cuda")]
use crate::kernels::Kernel;

/// Default CUDA workgroup size for batched attention kernels.
#[cfg(feature = "cuda")]
const CUDA_WORKGROUP_SIZE: u32 = 256;

/// Convert interleaved tensor to batched layout for all heads
#[cfg(feature = "cuda")]
pub(in super::super) fn interleaved_to_batched_all(
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

    let threads = CUDA_WORKGROUP_SIZE;
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
pub(in super::super) fn batched_transpose_all(
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

    let threads = CUDA_WORKGROUP_SIZE;
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

/// Convert batched tensor back to interleaved layout
#[cfg(feature = "cuda")]
pub(in super::super) fn batched_to_interleaved_all(
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

    let threads = CUDA_WORKGROUP_SIZE;
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
