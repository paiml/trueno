//! GPU-Resident Tensor (WAPR-PERF-004)
//!
//! Tensors that stay on GPU with transfer tracking to minimize host↔device traffic.
//!
//! ## Problem
//!
//! Standard approach: Each operation transfers data back to host
//! - matmul: GPU → CPU (for softmax)
//! - softmax: CPU → GPU (for next matmul)
//! - Result: ~150 transfers per encoder pass
//!
//! ## Solution
//!
//! GpuResidentTensor keeps data on device, only transfers when explicitly requested.
//! - Operations return new GpuResidentTensors (still on device)
//! - Only `.to_host()` triggers device→host transfer
//! - Transfer counters enable debugging and verification
//!
//! ## Citations
//!
//! - [Dao2022] FlashAttention: Fast and Memory-Efficient Exact Attention
//! - [Kwon2023] PagedAttention for LLM Serving with vLLM

// Sub-modules
mod attention;
mod cache;
mod stats;
mod weights;

// Re-exports from submodules
#[cfg(feature = "cuda")]
pub use attention::{
    batched_multihead_attention, batched_multihead_attention_optimized,
    incremental_attention_gpu, incremental_attention_gpu_async,
    incremental_attention_gpu_with_stream, kv_cache_scatter_gpu,
};
pub use cache::{
    clear_kernel_cache, kernel_cache_hits, kernel_cache_misses, reset_kernel_cache_stats,
};
pub use stats::{
    reset_transfer_counters, total_d2h_bytes, total_d2h_transfers, total_h2d_bytes,
    total_h2d_transfers, TransferStats,
};
#[cfg(feature = "cuda")]
pub use weights::{
    forward_encoder_block_gpu, GpuConvFrontendWeights, GpuDecoderBlockWeights,
    GpuDecoderConfig, GpuEncoderBlockWeights, GpuEncoderConfig, GpuKvCache,
};

// Internal access to submodule functions
#[cfg(feature = "cuda")]
use cache::get_or_compile_kernel;
#[cfg(feature = "cuda")]
use stats::{record_d2h_transfer, record_h2d_transfer};

#[cfg(feature = "cuda")]
use crate::driver::{CudaContext, GpuBuffer};
#[cfg(feature = "cuda")]
use crate::error::Result;

// ============================================================================
// GpuResidentTensor (CUDA-only)
// ============================================================================

#[cfg(feature = "cuda")]
/// A tensor that resides on GPU with transfer tracking
///
/// Unlike regular GpuBuffer, this tracks all transfers for debugging
/// and verification of GPU-resident pipelines.
///
/// # Example
///
/// ```ignore
/// use trueno_gpu::memory::resident::GpuResidentTensor;
///
/// let ctx = CudaContext::new(0)?;
/// let data = vec![1.0f32, 2.0, 3.0, 4.0];
///
/// // Upload data (1 H2D transfer)
/// let tensor = GpuResidentTensor::from_host(&ctx, &data)?;
/// assert_eq!(tensor.h2d_transfers(), 1);
///
/// // Operations stay on GPU
/// let doubled = tensor.scale(2.0)?; // No transfer!
/// assert_eq!(doubled.d2h_transfers(), 0);
///
/// // Only explicit download triggers transfer
/// let result = doubled.to_host()?;
/// assert_eq!(doubled.d2h_transfers(), 1);
/// ```
pub struct GpuResidentTensor<T: Copy> {
    /// Underlying GPU buffer
    buffer: GpuBuffer<T>,
    /// Number of host-to-device transfers for this tensor
    h2d_count: u64,
    /// Number of device-to-host transfers for this tensor
    d2h_count: u64,
    /// Number of kernel launches involving this tensor
    kernel_launches: u64,
    /// Whether this tensor is currently on device
    is_resident: bool,
}

#[cfg(feature = "cuda")]
impl<T: Copy> GpuResidentTensor<T> {
    /// Create a GPU-resident tensor from host data
    ///
    /// This uploads the data to GPU (1 H2D transfer).
    pub fn from_host(ctx: &CudaContext, data: &[T]) -> Result<Self> {
        let buffer = GpuBuffer::from_host(ctx, data)?;
        let bytes = data.len() * std::mem::size_of::<T>();

        // Track transfer
        record_h2d_transfer(bytes as u64);
        

        Ok(Self {
            buffer,
            h2d_count: 1,
            d2h_count: 0,
            kernel_launches: 0,
            is_resident: true,
        })
    }

    /// Create an uninitialized tensor on GPU
    ///
    /// The tensor has allocated memory but uninitialized contents.
    /// Use this for output buffers.
    pub fn new_uninit(ctx: &CudaContext, len: usize) -> Result<Self> {
        let buffer = GpuBuffer::new(ctx, len)?;

        Ok(Self {
            buffer,
            h2d_count: 0,
            d2h_count: 0,
            kernel_launches: 0,
            is_resident: true,
        })
    }

    /// Create from existing GPU buffer (internal constructor)
    ///
    /// Used when creating result tensors from GPU operations.
    /// Does NOT count as a transfer since data never left GPU.
    pub(crate) fn from_buffer_internal(buffer: GpuBuffer<T>, kernel_launches: u64) -> Self {
        Self {
            buffer,
            h2d_count: 0,
            d2h_count: 0,
            kernel_launches,
            is_resident: true,
        }
    }

    /// Download tensor to host memory
    ///
    /// This triggers 1 D2H transfer.
    pub fn to_host(&mut self) -> Result<Vec<T>>
    where
        T: Default + Clone,
    {
        let mut result = vec![T::default(); self.buffer.len()];
        self.buffer.copy_to_host(&mut result)?;

        let bytes = result.len() * std::mem::size_of::<T>();

        // Track transfer
        self.d2h_count += 1;
        record_d2h_transfer(bytes as u64);
        

        Ok(result)
    }

    /// Peek at tensor data on host (debug only, no transfer tracking)
    ///
    /// This copies data to host without updating transfer counters.
    /// Use only for debugging to avoid affecting transfer statistics.
    pub fn peek_host(&self) -> Result<Vec<T>>
    where
        T: Default + Clone,
    {
        let mut result = vec![T::default(); self.buffer.len()];
        self.buffer.copy_to_host(&mut result)?;
        Ok(result)
    }

    /// Check if tensor is currently resident on device
    #[must_use]
    pub const fn is_device_resident(&self) -> bool {
        self.is_resident
    }

    /// Get number of host-to-device transfers for this tensor
    #[must_use]
    pub const fn h2d_transfers(&self) -> u64 {
        self.h2d_count
    }

    /// Alias for h2d_transfers
    #[must_use]
    pub const fn host_to_device_transfers(&self) -> u64 {
        self.h2d_count
    }

    /// Get number of device-to-host transfers for this tensor
    #[must_use]
    pub const fn d2h_transfers(&self) -> u64 {
        self.d2h_count
    }

    /// Alias for d2h_transfers
    #[must_use]
    pub const fn device_to_host_transfers(&self) -> u64 {
        self.d2h_count
    }

    /// Get number of kernel launches involving this tensor
    #[must_use]
    pub const fn kernel_launches(&self) -> u64 {
        self.kernel_launches
    }

    /// Get tensor length (number of elements)
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if tensor is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.buffer.size_bytes()
    }

    /// Get underlying GPU buffer (for kernel operations)
    #[must_use]
    pub fn buffer(&self) -> &GpuBuffer<T> {
        &self.buffer
    }

    /// Get mutable reference to underlying GPU buffer
    #[must_use]
    pub fn buffer_mut(&mut self) -> &mut GpuBuffer<T> {
        &mut self.buffer
    }

    /// Get device pointer
    #[must_use]
    pub fn as_ptr(&self) -> u64 {
        self.buffer.as_ptr()
    }

    /// Increment kernel launch counter (called by kernel executors)
    pub fn record_kernel_launch(&mut self) {
        self.kernel_launches += 1;
    }
}

// ============================================================================
// GPU-Resident Operations (f32 specialization)
// ============================================================================

#[cfg(feature = "cuda")]
use crate::driver::{CudaStream, LaunchConfig};
#[cfg(feature = "cuda")]
use crate::kernels::{GemmKernel, Kernel, LongRowSoftmaxKernel, ScaleKernel, SoftmaxKernel};

#[cfg(feature = "cuda")]
impl GpuResidentTensor<f32> {
    /// Matrix multiply: C = A @ B (stays on GPU)
    ///
    /// Both tensors must be f32. Result is a new GPU-resident tensor.
    /// Does NOT transfer data to host.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `other` - Right-hand matrix
    /// * `m` - Rows of A
    /// * `n` - Columns of B
    /// * `k` - Columns of A / Rows of B
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn matmul(
        &self,
        ctx: &CudaContext,
        other: &GpuResidentTensor<f32>,
        m: u32,
        n: u32,
        k: u32,
    ) -> Result<GpuResidentTensor<f32>> {
        // Validate dimensions
        let expected_a = (m * k) as usize;
        let expected_b = (k * n) as usize;
        let output_size = (m * n) as usize;

        if self.len() != expected_a {
            return Err(crate::GpuError::InvalidParameter(format!(
                "A has {} elements, expected {} ({}x{})",
                self.len(),
                expected_a,
                m,
                k
            )));
        }
        if other.len() != expected_b {
            return Err(crate::GpuError::InvalidParameter(format!(
                "B has {} elements, expected {} ({}x{})",
                other.len(),
                expected_b,
                k,
                n
            )));
        }

        // Allocate output buffer on GPU
        let output_buffer = GpuBuffer::new(ctx, output_size)?;

        // Build and compile GEMM kernel (cached)
        // WAPR-PERF-010: Use WMMA Tensor Cores for large matrices
        // Fixed: D → C accumulator copy for multi-tile K dimension
        let tile_size = 16u32;
        // WAPR-PERF-014: Allow disabling WMMA for precision debugging
        let force_fp32 = std::env::var("TRUENO_FORCE_FP32_GEMM").is_ok();
        let use_wmma = !force_fp32 && k >= 64 && m >= 64 && n >= 64;
        let use_tiled = !use_wmma && k >= 64;

        let (kernel, cache_key, config) = if use_wmma {
            let kernel = GemmKernel::wmma_fp16(m, n, k);
            let key = format!("gemm_wmma_fp16:{}x{}x{}", m, n, k);
            // WMMA: one warp (32 threads) per 16x16 output tile
            let grid_x = (n + 15) / 16;
            let grid_y = (m + 15) / 16;
            // Shared memory: 2 FP16 tiles = 16*16*2*2 = 1024 bytes
            let cfg = LaunchConfig {
                grid: (grid_x, grid_y, 1),
                block: (32, 1, 1), // One warp
                shared_mem: 1024,
            };
            (kernel, key, cfg)
        } else if use_tiled {
            let kernel = GemmKernel::tiled_unrolled(m, n, k, tile_size);
            let key = format!("gemm_tiled_unrolled:{}x{}x{}", m, n, k);
            let grid_x = (n + tile_size - 1) / tile_size;
            let grid_y = (m + tile_size - 1) / tile_size;
            let cfg = LaunchConfig {
                grid: (grid_x, grid_y, 1),
                block: (tile_size, tile_size, 1),
                shared_mem: tile_size * tile_size * 4 * 2,
            };
            (kernel, key, cfg)
        } else {
            let kernel = GemmKernel::naive(m, n, k);
            let key = format!("gemm_naive:{}x{}x{}", m, n, k);
            let block_size = 16u32;
            let grid_x = (n + block_size - 1) / block_size;
            let grid_y = (m + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid: (grid_x, grid_y, 1),
                block: (block_size, block_size, 1),
                shared_mem: 0,
            };
            (kernel, key, cfg)
        };

        let ptx = kernel.emit_ptx();
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
        let stream = CudaStream::new(ctx)?;

        // Prepare arguments
        let a_ptr = self.as_ptr();
        let b_ptr = other.as_ptr();
        let c_ptr = output_buffer.as_ptr();
        let m_val = m;
        let n_val = n;
        let k_val = k;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(a_ptr) as *mut _,
            std::ptr::addr_of!(b_ptr) as *mut _,
            std::ptr::addr_of!(c_ptr) as *mut _,
            std::ptr::addr_of!(m_val) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
            std::ptr::addr_of!(k_val) as *mut _,
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
        stream.synchronize()?;

        // Return result as GPU-resident tensor (no host transfer!)
        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// WAPR-PERF-014: Matrix multiply with external stream (no stream creation, no sync)
    ///
    /// Same as `matmul` but uses caller-provided stream and does NOT synchronize.
    /// Use this in tight loops to avoid 16+ stream creates/syncs per token.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `other` - Right-hand matrix
    /// * `m` - Rows of A
    /// * `n` - Columns of B
    /// * `k` - Columns of A / Rows of B
    /// * `stream` - Caller-provided CUDA stream (reuse across operations)
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn matmul_with_stream(
        &self,
        ctx: &CudaContext,
        other: &GpuResidentTensor<f32>,
        m: u32,
        n: u32,
        k: u32,
        stream: &CudaStream,
    ) -> Result<GpuResidentTensor<f32>> {
        // Validate dimensions
        let expected_a = (m * k) as usize;
        let expected_b = (k * n) as usize;
        let output_size = (m * n) as usize;

        if self.len() != expected_a {
            return Err(crate::GpuError::InvalidParameter(format!(
                "A has {} elements, expected {} ({}x{})",
                self.len(), expected_a, m, k
            )));
        }
        if other.len() != expected_b {
            return Err(crate::GpuError::InvalidParameter(format!(
                "B has {} elements, expected {} ({}x{})",
                other.len(), expected_b, k, n
            )));
        }

        // Allocate output buffer on GPU
        let output_buffer = GpuBuffer::new(ctx, output_size)?;

        // Build and compile GEMM kernel (cached)
        let tile_size = 16u32;
        // WAPR-PERF-014: Allow disabling WMMA for precision debugging
        let force_fp32 = std::env::var("TRUENO_FORCE_FP32_GEMM").is_ok();
        let use_wmma = !force_fp32 && k >= 64 && m >= 64 && n >= 64;
        let use_tiled = !use_wmma && k >= 64;

        let (kernel, cache_key, config) = if use_wmma {
            let kernel = GemmKernel::wmma_fp16(m, n, k);
            let key = format!("gemm_wmma_fp16:{}x{}x{}", m, n, k);
            let grid_x = (n + 15) / 16;
            let grid_y = (m + 15) / 16;
            let cfg = LaunchConfig {
                grid: (grid_x, grid_y, 1),
                block: (32, 1, 1),
                shared_mem: 1024,
            };
            (kernel, key, cfg)
        } else if use_tiled {
            let kernel = GemmKernel::tiled_unrolled(m, n, k, tile_size);
            let key = format!("gemm_tiled_unrolled:{}x{}x{}", m, n, k);
            let grid_x = (n + tile_size - 1) / tile_size;
            let grid_y = (m + tile_size - 1) / tile_size;
            let cfg = LaunchConfig {
                grid: (grid_x, grid_y, 1),
                block: (tile_size, tile_size, 1),
                shared_mem: tile_size * tile_size * 4 * 2,
            };
            (kernel, key, cfg)
        } else {
            let kernel = GemmKernel::naive(m, n, k);
            let key = format!("gemm_naive:{}x{}x{}", m, n, k);
            let block_size = 16u32;
            let grid_x = (n + block_size - 1) / block_size;
            let grid_y = (m + block_size - 1) / block_size;
            let cfg = LaunchConfig {
                grid: (grid_x, grid_y, 1),
                block: (block_size, block_size, 1),
                shared_mem: 0,
            };
            (kernel, key, cfg)
        };

        let ptx = kernel.emit_ptx();
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;

        // Prepare arguments
        let a_ptr = self.as_ptr();
        let b_ptr = other.as_ptr();
        let c_ptr = output_buffer.as_ptr();
        let m_val = m;
        let n_val = n;
        let k_val = k;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(a_ptr) as *mut _,
            std::ptr::addr_of!(b_ptr) as *mut _,
            std::ptr::addr_of!(c_ptr) as *mut _,
            std::ptr::addr_of!(m_val) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
            std::ptr::addr_of!(k_val) as *mut _,
        ];

        // Launch kernel using caller's stream (lock the cached module)
        {
            let mut module = module_arc.lock().map_err(|e| {
                crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
            })?;
            unsafe {
                stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
            }
        }

        // NO SYNC - caller controls synchronization for pipelining
        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Row-wise softmax (stays on GPU)
    ///
    /// Computes softmax along the last dimension.
    /// Result is a new GPU-resident tensor.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `seq_len` - Sequence length (number of rows)
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn softmax(&self, ctx: &CudaContext, seq_len: u32) -> Result<GpuResidentTensor<f32>> {
        let total_elements = self.len();
        let row_size = total_elements / (seq_len as usize);

        if total_elements % (seq_len as usize) != 0 {
            return Err(crate::GpuError::InvalidParameter(format!(
                "Tensor size {} not divisible by seq_len {}",
                total_elements, seq_len
            )));
        }

        // Allocate output buffer on GPU
        let output_buffer = GpuBuffer::new(ctx, total_elements)?;

        // Choose kernel based on row size:
        // - row_size <= 32: warp shuffle softmax (1 warp per row)
        // - row_size > 32: long row softmax (multi-warp with grid-stride loops)
        let stream = CudaStream::new(ctx)?;
        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();
        let row_size_val = row_size as u32;

        if row_size <= 32 {
            // Use warp shuffle softmax for short rows (cached)
            let kernel = SoftmaxKernel::new(row_size as u32);
            let ptx = kernel.emit_ptx();
            let cache_key = format!("softmax:{}", row_size);
            let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;

            let config = LaunchConfig {
                grid: (seq_len, 1, 1),
                block: (32, 1, 1), // One warp per row
                shared_mem: 0,
            };

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                std::ptr::addr_of!(input_ptr) as *mut _,
                std::ptr::addr_of!(output_ptr) as *mut _,
                std::ptr::addr_of!(row_size_val) as *mut _,
            ];

            {
                let mut module = module_arc.lock().map_err(|e| {
                    crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
                })?;
                unsafe {
                    stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
                }
            }
        } else {
            // Use long row softmax for rows > 32 elements (cached)
            let kernel = LongRowSoftmaxKernel::new(row_size as u32);
            let ptx = kernel.emit_ptx();
            let cache_key = format!("softmax_long_row:{}", row_size);
            let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;

            // 256 threads per block (8 warps), one block per row
            // Shared memory: 8 warp maxes + 8 warp sums + 2 global = 72 bytes
            let config = LaunchConfig {
                grid: (seq_len, 1, 1),
                block: (256, 1, 1),
                shared_mem: 72,
            };

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                std::ptr::addr_of!(input_ptr) as *mut _,
                std::ptr::addr_of!(output_ptr) as *mut _,
                std::ptr::addr_of!(row_size_val) as *mut _,
            ];

            {
                let mut module = module_arc.lock().map_err(|e| {
                    crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
                })?;
                unsafe {
                    stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
                }
            }
        }

        stream.synchronize()?;

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Row-wise softmax with external stream (WAPR-PERF-017: CUDA Graph capture)
    ///
    /// Same as `softmax` but accepts caller-provided stream for pipelining.
    /// Does NOT synchronize - caller controls when to sync.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `seq_len` - Sequence length (number of rows)
    /// * `stream` - Caller-provided CUDA stream
    pub fn softmax_with_stream(
        &self,
        ctx: &CudaContext,
        seq_len: u32,
        stream: &CudaStream,
    ) -> Result<GpuResidentTensor<f32>> {
        let total_elements = self.len();
        let row_size = total_elements / (seq_len as usize);

        if total_elements % (seq_len as usize) != 0 {
            return Err(crate::GpuError::InvalidParameter(format!(
                "Tensor size {} not divisible by seq_len {}",
                total_elements, seq_len
            )));
        }

        // Allocate output buffer on GPU
        let output_buffer = GpuBuffer::new(ctx, total_elements)?;
        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();
        let row_size_val = row_size as u32;

        if row_size <= 32 {
            // Use warp shuffle softmax for short rows (cached)
            let kernel = SoftmaxKernel::new(row_size as u32);
            let ptx = kernel.emit_ptx();
            let cache_key = format!("softmax:{}", row_size);
            let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;

            let config = LaunchConfig {
                grid: (seq_len, 1, 1),
                block: (32, 1, 1), // One warp per row
                shared_mem: 0,
            };

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                std::ptr::addr_of!(input_ptr) as *mut _,
                std::ptr::addr_of!(output_ptr) as *mut _,
                std::ptr::addr_of!(row_size_val) as *mut _,
            ];

            {
                let mut module = module_arc.lock().map_err(|e| {
                    crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
                })?;
                unsafe {
                    stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
                }
            }
        } else {
            // Use long row softmax for rows > 32 elements (cached)
            let kernel = LongRowSoftmaxKernel::new(row_size as u32);
            let ptx = kernel.emit_ptx();
            let cache_key = format!("softmax_long_row:{}", row_size);
            let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;

            // 256 threads per block (8 warps), one block per row
            let config = LaunchConfig {
                grid: (seq_len, 1, 1),
                block: (256, 1, 1),
                shared_mem: 72,
            };

            let mut args: Vec<*mut std::ffi::c_void> = vec![
                std::ptr::addr_of!(input_ptr) as *mut _,
                std::ptr::addr_of!(output_ptr) as *mut _,
                std::ptr::addr_of!(row_size_val) as *mut _,
            ];

            {
                let mut module = module_arc.lock().map_err(|e| {
                    crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
                })?;
                unsafe {
                    stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
                }
            }
        }
        // NO SYNC - caller controls synchronization for graph capture

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Element-wise add (stays on GPU)
    ///
    /// Computes C = A + B element-wise.
    /// Result is a new GPU-resident tensor.
    ///
    /// # Errors
    ///
    /// Returns error if sizes don't match or kernel fails.
    pub fn add(&self, ctx: &CudaContext, other: &GpuResidentTensor<f32>) -> Result<GpuResidentTensor<f32>> {
        if self.len() != other.len() {
            return Err(crate::GpuError::InvalidParameter(format!(
                "Size mismatch: {} vs {}",
                self.len(),
                other.len()
            )));
        }

        let n = self.len();

        // Allocate output buffer on GPU
        let output_buffer = GpuBuffer::new(ctx, n)?;

        // Use simple add kernel via ResidualAddKernel (cached)
        use crate::kernels::ResidualAddKernel;
        let kernel = ResidualAddKernel::new(n as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("residual_add:{}", n);
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
        let stream = CudaStream::new(ctx)?;

        // Configure launch
        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        // Prepare arguments
        let a_ptr = self.as_ptr();
        let b_ptr = other.as_ptr();
        let c_ptr = output_buffer.as_ptr();
        let n_val = n as u32;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(a_ptr) as *mut _,
            std::ptr::addr_of!(b_ptr) as *mut _,
            std::ptr::addr_of!(c_ptr) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
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
        stream.synchronize()?;

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Element-wise add with external stream (WAPR-PERF-017: CUDA Graph capture)
    ///
    /// Same as `add` but accepts caller-provided stream for pipelining.
    /// Does NOT synchronize - caller controls when to sync.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `other` - Tensor to add
    /// * `stream` - Caller-provided CUDA stream
    pub fn add_with_stream(
        &self,
        ctx: &CudaContext,
        other: &GpuResidentTensor<f32>,
        stream: &CudaStream,
    ) -> Result<GpuResidentTensor<f32>> {
        if self.len() != other.len() {
            return Err(crate::GpuError::InvalidParameter(format!(
                "Size mismatch: {} vs {}",
                self.len(),
                other.len()
            )));
        }

        let n = self.len();

        // Allocate output buffer on GPU
        let output_buffer = GpuBuffer::new(ctx, n)?;

        // Use simple add kernel via ResidualAddKernel (cached)
        use crate::kernels::ResidualAddKernel;
        let kernel = ResidualAddKernel::new(n as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("residual_add:{}", n);
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;

        // Configure launch
        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        // Prepare arguments
        let a_ptr = self.as_ptr();
        let b_ptr = other.as_ptr();
        let c_ptr = output_buffer.as_ptr();
        let n_val = n as u32;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(a_ptr) as *mut _,
            std::ptr::addr_of!(b_ptr) as *mut _,
            std::ptr::addr_of!(c_ptr) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
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
        // NO SYNC - caller controls synchronization for graph capture

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Transform interleaved layout to head-first layout (for attention KV caches)
    ///
    /// Converts: [seq_len, n_heads * head_dim] -> [n_heads, seq_len, head_dim]
    ///
    /// This is the inverse of batched-to-interleaved and is used for preparing
    /// cross-attention K/V caches from encoder output projections.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `seq_len` - Sequence length (first dimension)
    /// * `n_heads` - Number of attention heads
    /// * `head_dim` - Dimension per head (n_heads * head_dim = d_model)
    /// * `stream` - Caller-provided CUDA stream
    ///
    /// # Errors
    ///
    /// Returns error if dimensions don't match tensor size.
    pub fn interleaved_to_head_first(
        &self,
        ctx: &CudaContext,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        stream: &CudaStream,
    ) -> Result<GpuResidentTensor<f32>> {
        let d_model = n_heads * head_dim;
        let total_elems = (seq_len * d_model) as usize;

        if self.len() != total_elems {
            return Err(crate::GpuError::InvalidParameter(format!(
                "Tensor size {} doesn't match seq_len ({}) × d_model ({})",
                self.len(),
                seq_len,
                d_model
            )));
        }

        let output_buffer = GpuBuffer::new(ctx, total_elems)?;

        use crate::kernels::{InterleavedToBatchedKernel, Kernel};
        let kernel = InterleavedToBatchedKernel::new(seq_len, n_heads, head_dim);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("interleaved_to_batched:{}:{}:{}", seq_len, n_heads, head_dim);
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;

        let threads = 256u32;
        let blocks = (total_elems as u32 + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();

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
        // NO SYNC - caller controls synchronization for graph capture

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Scale tensor by constant (stays on GPU)
    ///
    /// Computes B = A * scale element-wise.
    pub fn scale(&self, ctx: &CudaContext, scale: f32) -> Result<GpuResidentTensor<f32>> {
        let n = self.len();

        // Allocate output buffer on GPU
        let output_buffer = GpuBuffer::new(ctx, n)?;

        // Use ScaleKernel (multiplies by scalar constant, cached)
        let kernel = ScaleKernel::new(n as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("scale:{}", n);
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
        let stream = CudaStream::new(ctx)?;

        // Configure launch
        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        // Prepare arguments (must match kernel params: input_ptr, output_ptr, scale, n)
        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();
        let n_val = n as u32;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(scale) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
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
        stream.synchronize()?;

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Layer normalization (stays on GPU)
    ///
    /// Computes: output = (x - mean) / sqrt(var + eps) * gamma + beta
    ///
    /// # Arguments
    /// * `ctx` - CUDA context
    /// * `gamma` - Scale parameters [hidden_size]
    /// * `beta` - Shift parameters [hidden_size]
    /// * `hidden_size` - Dimension being normalized
    /// * `batch_size` - Number of rows to normalize
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn layer_norm(
        &self,
        ctx: &CudaContext,
        gamma: &GpuResidentTensor<f32>,
        beta: &GpuResidentTensor<f32>,
        hidden_size: u32,
        batch_size: u32,
    ) -> Result<GpuResidentTensor<f32>> {
        let n = self.len();
        let output_buffer = GpuBuffer::new(ctx, n)?;

        use crate::kernels::LayerNormKernel;
        let kernel = LayerNormKernel::new(hidden_size);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("layer_norm:{}", hidden_size);
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
        let stream = CudaStream::new(ctx)?;

        // Launch one warp per row
        let threads = 32u32.min(hidden_size);
        let blocks = batch_size;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();
        let gamma_ptr = gamma.as_ptr();
        let beta_ptr = beta.as_ptr();

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(gamma_ptr) as *mut _,
            std::ptr::addr_of!(beta_ptr) as *mut _,
            std::ptr::addr_of!(hidden_size) as *mut _,
            std::ptr::addr_of!(batch_size) as *mut _,
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

    /// Layer normalization with external stream (WAPR-PERF-017: CUDA Graph capture)
    ///
    /// Same as `layer_norm` but accepts caller-provided stream for pipelining.
    /// Does NOT synchronize - caller controls when to sync.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `gamma` - Scale parameters (hidden_size)
    /// * `beta` - Bias parameters (hidden_size)
    /// * `hidden_size` - Size of hidden dimension
    /// * `batch_size` - Number of rows (batch or seq_len)
    /// * `stream` - Caller-provided CUDA stream
    pub fn layer_norm_with_stream(
        &self,
        ctx: &CudaContext,
        gamma: &GpuResidentTensor<f32>,
        beta: &GpuResidentTensor<f32>,
        hidden_size: u32,
        batch_size: u32,
        stream: &CudaStream,
    ) -> Result<GpuResidentTensor<f32>> {
        let n = self.len();
        let output_buffer = GpuBuffer::new(ctx, n)?;

        use crate::kernels::LayerNormKernel;
        let kernel = LayerNormKernel::new(hidden_size);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("layer_norm:{}", hidden_size);
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;

        // Launch one warp per row
        let threads = 32u32.min(hidden_size);
        let blocks = batch_size;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();
        let gamma_ptr = gamma.as_ptr();
        let beta_ptr = beta.as_ptr();

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(gamma_ptr) as *mut _,
            std::ptr::addr_of!(beta_ptr) as *mut _,
            std::ptr::addr_of!(hidden_size) as *mut _,
            std::ptr::addr_of!(batch_size) as *mut _,
        ];

        {
            let mut module = module_arc.lock().map_err(|e| {
                crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
            })?;
            unsafe {
                stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
            }
        }
        // NO SYNC - caller controls synchronization for graph capture

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// GELU activation (stays on GPU)
    ///
    /// Computes: output = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn gelu(&self, ctx: &CudaContext) -> Result<GpuResidentTensor<f32>> {
        let n = self.len();
        let output_buffer = GpuBuffer::new(ctx, n)?;

        use crate::kernels::GeluKernel;
        let kernel = GeluKernel::new(n as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("gelu:{}", n);
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
        let stream = CudaStream::new(ctx)?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();
        let n_val = n as u32;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
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

    /// GELU activation with external stream (WAPR-PERF-017: CUDA Graph capture)
    ///
    /// Same as `gelu` but accepts caller-provided stream for pipelining.
    /// Does NOT synchronize - caller controls when to sync.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `stream` - Caller-provided CUDA stream
    pub fn gelu_with_stream(
        &self,
        ctx: &CudaContext,
        stream: &CudaStream,
    ) -> Result<GpuResidentTensor<f32>> {
        let n = self.len();
        let output_buffer = GpuBuffer::new(ctx, n)?;

        use crate::kernels::GeluKernel;
        let kernel = GeluKernel::new(n as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("gelu:{}", n);
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        let input_ptr = self.as_ptr();
        let output_ptr = output_buffer.as_ptr();
        let n_val = n as u32;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
        ];

        {
            let mut module = module_arc.lock().map_err(|e| {
                crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
            })?;
            unsafe {
                stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
            }
        }
        // NO SYNC - caller controls synchronization for graph capture

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Bias add (stays on GPU)
    ///
    /// Computes: output[i] = input[i] + bias[i % bias_size]
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn bias_add(
        &self,
        ctx: &CudaContext,
        bias: &GpuResidentTensor<f32>,
    ) -> Result<GpuResidentTensor<f32>> {
        let n = self.len();
        let bias_size = bias.len();

        // WAPR-PERF-027 FIX: Create stream FIRST to ensure D2D copy and kernel use same stream
        // BiasActivationKernel is IN-PLACE: reads from output, adds bias, writes to output
        let stream = CudaStream::new(ctx)?;

        // Allocate output buffer and copy input data using SAME stream
        // Previously used clone() which ran on default stream - race condition with kernel!
        let mut output_buffer = GpuBuffer::new(ctx, n)?;
        // SAFETY: both buffers valid, stream will be synchronized before returning
        unsafe {
            output_buffer.copy_from_buffer_async(&self.buffer, &stream)?;
        }

        use crate::kernels::BiasActivationKernel;
        let kernel = BiasActivationKernel::new(n as u32, bias_size as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("bias_add:{}:{}", n, bias_size);
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        let output_ptr = output_buffer.as_ptr();
        let bias_ptr = bias.as_ptr();
        let n_val = n as u32;

        // Kernel params: (output, bias, n) - kernel is in-place on output
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(bias_ptr) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
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

    /// WAPR-PERF-014: Bias add with external stream (no stream creation, no sync)
    pub fn bias_add_with_stream(
        &self,
        ctx: &CudaContext,
        bias: &GpuResidentTensor<f32>,
        stream: &CudaStream,
    ) -> Result<GpuResidentTensor<f32>> {
        let n = self.len();
        let bias_size = bias.len();

        let output_buffer = self.buffer.clone(ctx)?;

        use crate::kernels::BiasActivationKernel;
        let kernel = BiasActivationKernel::new(n as u32, bias_size as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("bias_add:{}:{}", n, bias_size);
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;

        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        let output_ptr = output_buffer.as_ptr();
        let bias_ptr = bias.as_ptr();
        let n_val = n as u32;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(bias_ptr) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
        ];

        {
            let mut module = module_arc.lock().map_err(|e| {
                crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e))
            })?;
            unsafe {
                stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)?;
            }
        }
        // NO SYNC - caller controls synchronization

        Ok(GpuResidentTensor::from_buffer_internal(output_buffer, 1))
    }

    /// Linear projection: output = input @ weight + bias (stays on GPU)
    ///
    /// Weight is [in_features, out_features] row-major.
    /// Input is [batch_size * in_features] flattened.
    /// Output is [batch_size * out_features].
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn linear(
        &self,
        ctx: &CudaContext,
        weight: &GpuResidentTensor<f32>,
        bias: Option<&GpuResidentTensor<f32>>,
        batch_size: u32,
        in_features: u32,
        out_features: u32,
    ) -> Result<GpuResidentTensor<f32>> {
        let debug = std::env::var("WHISPER_DEBUG_LINEAR").is_ok();
        if debug {
            eprintln!("[DEBUG-LINEAR] input: len={}, batch={}, in_feat={}, out_feat={}",
                self.len(), batch_size, in_features, out_features);
            let inp = self.peek_host()?;
            eprintln!("[DEBUG-LINEAR] input stats: mean={:.6}, max={:.6}",
                inp.iter().sum::<f32>() / inp.len() as f32,
                inp.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }

        // matmul: [batch_size, in_features] @ [in_features, out_features] = [batch_size, out_features]
        let result = self.matmul(ctx, weight, batch_size, out_features, in_features)?;

        if debug {
            let res = result.peek_host()?;
            eprintln!("[DEBUG-LINEAR] matmul result: len={}, mean={:.6}, max={:.6}",
                res.len(),
                res.iter().sum::<f32>() / res.len() as f32,
                res.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        }

        // Add bias if provided
        if let Some(b) = bias {
            let output = result.bias_add(ctx, b)?;
            if debug {
                let out = output.peek_host()?;
                eprintln!("[DEBUG-LINEAR] after bias_add: len={}, mean={:.6}, max={:.6}",
                    out.len(),
                    out.iter().sum::<f32>() / out.len() as f32,
                    out.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
            }
            Ok(output)
        } else {
            Ok(result)
        }
    }

    /// Fused linear + GELU: output = GELU(input @ weight + bias) (WAPR-PERF-007)
    ///
    /// Combines GEMM + Bias + GELU into a single kernel launch, eliminating
    /// 2 kernel launches and associated memory traffic.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `weight` - Weight matrix [in_features, out_features]
    /// * `bias` - Bias vector [out_features]
    /// * `batch_size` - Number of rows in input
    /// * `in_features` - Input dimension (K)
    /// * `out_features` - Output dimension (N)
    ///
    /// # Errors
    ///
    /// Returns error if kernel launch fails.
    pub fn fused_linear_gelu(
        &self,
        ctx: &CudaContext,
        weight: &GpuResidentTensor<f32>,
        bias: &GpuResidentTensor<f32>,
        batch_size: u32,
        in_features: u32,
        out_features: u32,
    ) -> Result<GpuResidentTensor<f32>> {
        use crate::kernels::FusedGemmBiasGeluKernel;

        let output_size = (batch_size * out_features) as usize;
        let output_buffer = GpuBuffer::new(ctx, output_size)?;

        // Build and compile fused kernel (cached)
        let kernel = FusedGemmBiasGeluKernel::new(batch_size, out_features, in_features);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("fused_gemm_bias_gelu:{}x{}x{}", batch_size, out_features, in_features);
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
        let stream = CudaStream::new(ctx)?;

        // Configure launch: 16x16 block, grid covers output matrix
        let block_size = 16u32;
        let grid_x = (out_features + block_size - 1) / block_size;
        let grid_y = (batch_size + block_size - 1) / block_size;
        let config = LaunchConfig {
            grid: (grid_x, grid_y, 1),
            block: (block_size, block_size, 1),
            shared_mem: 0,
        };

        // Prepare arguments
        let a_ptr = self.as_ptr();
        let b_ptr = weight.as_ptr();
        let bias_ptr = bias.as_ptr();
        let c_ptr = output_buffer.as_ptr();
        let m_val = batch_size;
        let n_val = out_features;
        let k_val = in_features;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(a_ptr) as *mut _,
            std::ptr::addr_of!(b_ptr) as *mut _,
            std::ptr::addr_of!(bias_ptr) as *mut _,
            std::ptr::addr_of!(c_ptr) as *mut _,
            std::ptr::addr_of!(m_val) as *mut _,
            std::ptr::addr_of!(n_val) as *mut _,
            std::ptr::addr_of!(k_val) as *mut _,
        ];

        // Launch fused kernel
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

    /// WAPR-PERF-012: GPU Conv1d with GELU activation
    ///
    /// Computes 1D convolution for Whisper audio frontend.
    /// Target: Move 588ms CPU conv to GPU (<50ms).
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `weight` - Weight tensor [out_channels, in_channels, kernel_size]
    /// * `bias` - Bias tensor [out_channels] (optional)
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Convolution kernel size
    /// * `stride` - Stride
    /// * `padding` - Padding
    /// * `seq_len` - Input sequence length
    ///
    /// # Returns
    ///
    /// Output tensor [out_seq_len, out_channels] with GELU applied
    pub fn conv1d(
        &self,
        ctx: &CudaContext,
        weight: &GpuResidentTensor<f32>,
        bias: Option<&GpuResidentTensor<f32>>,
        in_channels: u32,
        out_channels: u32,
        kernel_size: u32,
        stride: u32,
        padding: u32,
        seq_len: u32,
    ) -> Result<GpuResidentTensor<f32>> {
        use crate::kernels::Conv1dKernel;

        // Calculate output sequence length
        let out_seq_len = (seq_len + 2 * padding - kernel_size) / stride + 1;
        let output_size = (out_seq_len * out_channels) as usize;

        // Validate input dimensions
        let expected_input = (seq_len * in_channels) as usize;
        if self.len() != expected_input {
            return Err(crate::GpuError::InvalidParameter(format!(
                "Input has {} elements, expected {} ({}x{})",
                self.len(), expected_input, seq_len, in_channels
            )));
        }

        let expected_weight = (out_channels * in_channels * kernel_size) as usize;
        if weight.len() != expected_weight {
            return Err(crate::GpuError::InvalidParameter(format!(
                "Weight has {} elements, expected {} ({}x{}x{})",
                weight.len(), expected_weight, out_channels, in_channels, kernel_size
            )));
        }

        // Allocate output buffer
        let output_buffer = GpuBuffer::new(ctx, output_size)?;

        // Build kernel
        let kernel = Conv1dKernel::new(in_channels, out_channels, kernel_size, stride, padding);
        let cache_key = format!(
            "conv1d:{}:{}:{}:{}:{}",
            in_channels, out_channels, kernel_size, stride, padding
        );
        let ptx = kernel.emit_ptx();
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
        let stream = CudaStream::new(ctx)?;

        // Launch configuration
        let block_x = 32u32;
        let block_y = 8u32;
        let grid_x = (out_seq_len + block_x - 1) / block_x;
        let grid_y = (out_channels + block_y - 1) / block_y;

        let config = LaunchConfig {
            grid: (grid_x, grid_y, 1),
            block: (block_x, block_y, 1),
            shared_mem: 0,
        };

        // Prepare arguments
        let input_ptr = self.as_ptr();
        let weight_ptr = weight.as_ptr();
        let bias_ptr = bias.map_or(0_u64, |b| b.as_ptr());
        let output_ptr = output_buffer.as_ptr();
        let seq_len_val = seq_len;

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(weight_ptr) as *mut _,
            std::ptr::addr_of!(bias_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
            std::ptr::addr_of!(seq_len_val) as *mut _,
        ];

        // Launch kernel
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
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::resident::stats::{record_d2h_transfer, record_h2d_transfer};

    // =========================================================================
    // GpuResidentTensor Lifecycle Tests (Titan Duel Strategy - PMAT-018)
    // =========================================================================

    /// Test GpuResidentTensor lifecycle: allocate, write, read, drop
    ///
    /// This test verifies the complete lifecycle path to ensure coverage
    /// of allocation, transfer tracking, and deallocation paths.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_resident_tensor_lifecycle() {
        use crate::driver::CudaContext;

        // Skip gracefully if no CUDA context available
        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping CUDA lifecycle test: {:?}", e);
                return;
            }
        };

        // Reset counters for clean test
        reset_transfer_counters();

        // 1. Create tensor from host data (1 H2D transfer)
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut tensor = GpuResidentTensor::from_host(&ctx, &data)
            .expect("Failed to create GpuResidentTensor");

        // Verify initial state
        assert!(tensor.is_device_resident());
        assert_eq!(tensor.len(), 8);
        assert_eq!(tensor.h2d_transfers(), 1);
        assert_eq!(tensor.d2h_transfers(), 0);
        assert_eq!(tensor.kernel_launches(), 0);

        // 2. Verify global transfer counters
        assert_eq!(total_h2d_transfers(), 1);
        assert_eq!(total_d2h_transfers(), 0);
        assert_eq!(total_h2d_bytes(), 32); // 8 * sizeof(f32) = 32

        // 3. Read data back (1 D2H transfer)
        let result = tensor.to_host().expect("Failed to read from GPU");
        assert_eq!(result, data);
        assert_eq!(tensor.d2h_transfers(), 1);
        assert_eq!(total_d2h_transfers(), 1);
        assert_eq!(total_d2h_bytes(), 32);

        // 4. Tensor drops automatically at end of scope
        // This tests the Drop implementation for GpuBuffer
    }

    /// Test new_uninit path for output buffers
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_resident_tensor_uninit() {
        use crate::driver::CudaContext;

        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping CUDA uninit test: {:?}", e);
                return;
            }
        };

        reset_transfer_counters();

        // Create uninitialized tensor (no transfer)
        let tensor: GpuResidentTensor<f32> = GpuResidentTensor::new_uninit(&ctx, 16)
            .expect("Failed to create uninit GpuResidentTensor");

        // No transfers for uninitialized buffer
        assert_eq!(tensor.h2d_transfers(), 0);
        assert_eq!(tensor.d2h_transfers(), 0);
        assert!(tensor.is_device_resident());
        assert_eq!(tensor.len(), 16);
        assert_eq!(tensor.size_bytes(), 64); // 16 * sizeof(f32)

        // Global counters unchanged
        assert_eq!(total_h2d_transfers(), 0);
        assert_eq!(total_d2h_transfers(), 0);
    }

    /// Test peek_host doesn't affect transfer counters
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_resident_tensor_peek() {
        use crate::driver::CudaContext;

        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping CUDA peek test: {:?}", e);
                return;
            }
        };

        reset_transfer_counters();

        let data = vec![42.0f32; 4];
        let tensor = GpuResidentTensor::from_host(&ctx, &data)
            .expect("Failed to create GpuResidentTensor");

        // Initial state: 1 H2D, 0 D2H
        let before_h2d = total_h2d_transfers();
        let before_d2h = total_d2h_transfers();

        // Peek doesn't update counters
        let peeked = tensor.peek_host().expect("Failed to peek");
        assert_eq!(peeked, data);

        // Counters unchanged after peek
        assert_eq!(total_h2d_transfers(), before_h2d);
        assert_eq!(total_d2h_transfers(), before_d2h);
        assert_eq!(tensor.d2h_transfers(), 0); // Instance counter also unchanged
    }

    /// Test buffer accessor methods
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_resident_tensor_buffer_access() {
        use crate::driver::CudaContext;

        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping CUDA buffer access test: {:?}", e);
                return;
            }
        };

        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut tensor = GpuResidentTensor::from_host(&ctx, &data)
            .expect("Failed to create GpuResidentTensor");

        // Test immutable buffer access
        let buf = tensor.buffer();
        assert_eq!(buf.len(), 4);

        // Test mutable buffer access
        let buf_mut = tensor.buffer_mut();
        assert_eq!(buf_mut.len(), 4);
    }

    // =========================================================================
    // Original Transfer Stats Tests
    // =========================================================================

    #[test]
    fn test_transfer_stats_capture_and_delta() {
        reset_transfer_counters();

        let before = TransferStats::capture();
        assert_eq!(before.total_transfers(), 0);

        // Simulate some transfers using the record functions
        record_h2d_transfer(1024);
        record_h2d_transfer(2048);
        record_h2d_transfer(512);
        record_d2h_transfer(512);

        let after = TransferStats::capture();
        let delta = after.delta_from(&before);

        assert_eq!(delta.h2d_transfers, 3);
        assert_eq!(delta.d2h_transfers, 1);
        assert_eq!(delta.h2d_bytes, 3584); // 1024 + 2048 + 512
        assert_eq!(delta.d2h_bytes, 512);
        assert_eq!(delta.total_transfers(), 4);
        assert_eq!(delta.total_bytes(), 4096);
    }

    #[test]
    fn test_transfer_stats_display() {
        let stats = TransferStats {
            h2d_transfers: 5,
            d2h_transfers: 2,
            h2d_bytes: 1024 * 1024 * 10, // 10 MB
            d2h_bytes: 1024 * 1024 * 5,  // 5 MB
        };

        let display = format!("{}", stats);
        assert!(display.contains("H2D: 5"));
        assert!(display.contains("D2H: 2"));
        assert!(display.contains("10.00 MB"));
        assert!(display.contains("5.00 MB"));
    }

    #[test]
    fn test_reset_counters() {
        record_h2d_transfer(100);
        record_d2h_transfer(50);

        reset_transfer_counters();

        assert_eq!(total_h2d_transfers(), 0);
        assert_eq!(total_d2h_transfers(), 0);
        assert_eq!(total_h2d_bytes(), 0);
        assert_eq!(total_d2h_bytes(), 0);
    }

    // =========================================================================
    // GPU Memory Pressure Test (PMAT-018: Coverage Killer Remediation)
    // =========================================================================

    /// Test GPU behavior under memory pressure
    ///
    /// This test exercises the allocation failure path by:
    /// 1. Allocating tensors until memory is exhausted
    /// 2. Verifying that allocation failures are graceful (no panic)
    /// 3. Demonstrating that memory is reclaimed after dropping tensors
    ///
    /// Note: This test does NOT assert automatic eviction since no eviction
    /// policy is currently implemented. It tests graceful degradation.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gpu_allocation_under_pressure() {
        use crate::driver::CudaContext;

        let ctx = match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping GPU pressure test: {:?}", e);
                return;
            }
        };

        reset_transfer_counters();

        // Allocate 64MB chunks until we hit an allocation failure
        const CHUNK_SIZE: usize = 64 * 1024 * 1024 / 4; // 64MB in f32s
        const MAX_CHUNKS: usize = 1024; // Safety limit (64GB max)

        let mut tensors: Vec<GpuResidentTensor<f32>> = Vec::new();
        let mut allocation_count = 0;
        let mut hit_limit = false;

        // Phase 1: Allocate until we hit memory limit
        for _ in 0..MAX_CHUNKS {
            let data = vec![0.0f32; CHUNK_SIZE];
            match GpuResidentTensor::from_host(&ctx, &data) {
                Ok(tensor) => {
                    tensors.push(tensor);
                    allocation_count += 1;
                }
                Err(_) => {
                    // Expected: CUDA_ERROR_OUT_OF_MEMORY or similar
                    hit_limit = true;
                    break;
                }
            }
        }

        // We should have allocated at least one tensor
        assert!(
            allocation_count > 0,
            "Should have allocated at least one tensor"
        );

        // Record how many we allocated before hitting the limit
        let tensors_at_limit = tensors.len();
        eprintln!(
            "GPU pressure test: Allocated {} tensors ({} MB) before limit",
            tensors_at_limit,
            tensors_at_limit * 64
        );

        // Phase 2: Free half the tensors
        let drop_count = tensors.len() / 2;
        for _ in 0..drop_count {
            tensors.pop();
        }

        // Phase 3: Verify we can allocate again after freeing
        let data = vec![0.0f32; CHUNK_SIZE];
        let recovery_result = GpuResidentTensor::from_host(&ctx, &data);

        // If we hit the limit, we should be able to recover after freeing
        if hit_limit {
            assert!(
                recovery_result.is_ok(),
                "Should be able to allocate after freeing tensors"
            );
        }

        // Verify transfer tracking still works under pressure
        let total_transfers = total_h2d_transfers();
        assert!(
            total_transfers >= allocation_count as u64,
            "Transfer counter should track all allocations"
        );
    }

    /// Test MemoryPool behavior under pressure (CPU-side simulation)
    ///
    /// This tests the MemoryPool allocator's behavior when full,
    /// verifying that allocation failures are properly reported.
    #[test]
    fn test_memory_pool_exhaustion() {
        use crate::memory::pool::{MemoryPool, PoolConfig};

        // Create a tiny pool (1MB with 64KB pages = 16 pages)
        let config = PoolConfig {
            total_bytes: 1024 * 1024,
            page_size: 64 * 1024,
        };
        let mut pool = MemoryPool::new(config);

        // Allocate all pages
        let mut allocations = Vec::new();
        for _ in 0..16 {
            if let Some(id) = pool.allocate(64 * 1024) {
                allocations.push(id);
            }
        }

        // Pool should now be full
        let stats = pool.stats();
        assert_eq!(stats.free_pages, 0, "Pool should be completely full");

        // Next allocation should fail
        let failed_alloc = pool.allocate(64 * 1024);
        assert!(
            failed_alloc.is_none(),
            "Allocation should fail when pool is exhausted"
        );

        // Free one allocation
        if let Some(id) = allocations.pop() {
            assert!(pool.free(id), "Free should succeed");
        }

        // Now allocation should succeed
        let recovered_alloc = pool.allocate(64 * 1024);
        assert!(
            recovered_alloc.is_some(),
            "Allocation should succeed after freeing"
        );
    }
}
