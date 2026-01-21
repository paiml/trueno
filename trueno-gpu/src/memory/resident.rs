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

use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "cuda")]
use crate::driver::{CudaContext, GpuBuffer};
#[cfg(feature = "cuda")]
use crate::error::Result;

// ============================================================================
// Transfer Statistics
// ============================================================================

/// Global transfer counters for debugging
static TOTAL_H2D_TRANSFERS: AtomicU64 = AtomicU64::new(0);
static TOTAL_D2H_TRANSFERS: AtomicU64 = AtomicU64::new(0);
static TOTAL_H2D_BYTES: AtomicU64 = AtomicU64::new(0);
static TOTAL_D2H_BYTES: AtomicU64 = AtomicU64::new(0);

/// Get total host-to-device transfers since last reset
#[must_use]
pub fn total_h2d_transfers() -> u64 {
    TOTAL_H2D_TRANSFERS.load(Ordering::Relaxed)
}

/// Get total device-to-host transfers since last reset
#[must_use]
pub fn total_d2h_transfers() -> u64 {
    TOTAL_D2H_TRANSFERS.load(Ordering::Relaxed)
}

/// Get total bytes transferred host-to-device since last reset
#[must_use]
pub fn total_h2d_bytes() -> u64 {
    TOTAL_H2D_BYTES.load(Ordering::Relaxed)
}

/// Get total bytes transferred device-to-host since last reset
#[must_use]
pub fn total_d2h_bytes() -> u64 {
    TOTAL_D2H_BYTES.load(Ordering::Relaxed)
}

/// Reset all transfer counters to zero
pub fn reset_transfer_counters() {
    TOTAL_H2D_TRANSFERS.store(0, Ordering::Relaxed);
    TOTAL_D2H_TRANSFERS.store(0, Ordering::Relaxed);
    TOTAL_H2D_BYTES.store(0, Ordering::Relaxed);
    TOTAL_D2H_BYTES.store(0, Ordering::Relaxed);
}

// ============================================================================
// Kernel Cache (WAPR-PERF-004)
// ============================================================================

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "cuda")]
use crate::driver::CudaModule;

/// Global kernel cache to eliminate PTX recompilation overhead.
///
/// Each unique kernel configuration (name + parameters) is compiled once
/// and cached for reuse. This eliminates the 24x recompilation per inference
/// that was previously observed.
///
/// ## Keying Strategy
///
/// Keys are strings of format: `"{kernel_name}:{config}"` where config
/// encodes all parameters that affect the PTX output.
///
/// ## Thread Safety
///
/// The cache uses double-locking:
/// - Outer Mutex guards the HashMap
/// - Inner Arc<Mutex<CudaModule>> allows concurrent kernel launches
///
/// ## Example Keys
///
/// - `"softmax:32"` - SoftmaxKernel for row_size=32
/// - `"softmax_long_row:1500"` - LongRowSoftmaxKernel for row_size=1500
/// - `"residual_add:1024"` - ResidualAddKernel for n=1024
#[cfg(feature = "cuda")]
static KERNEL_CACHE: OnceLock<Mutex<HashMap<String, Arc<Mutex<CudaModule>>>>> = OnceLock::new();

/// Statistics for kernel cache performance
#[cfg(feature = "cuda")]
static KERNEL_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "cuda")]
static KERNEL_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

/// Get the global kernel cache, initializing if needed
#[cfg(feature = "cuda")]
fn get_kernel_cache() -> &'static Mutex<HashMap<String, Arc<Mutex<CudaModule>>>> {
    KERNEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get a cached kernel module, compiling if not present.
///
/// # Arguments
///
/// * `ctx` - CUDA context for compilation
/// * `key` - Cache key (kernel_name:config)
/// * `ptx` - PTX source to compile on cache miss
///
/// # Returns
///
/// Arc to the cached module, wrapped in Mutex for mutable access.
#[cfg(feature = "cuda")]
fn get_or_compile_kernel(
    ctx: &CudaContext,
    key: &str,
    ptx: &str,
) -> Result<Arc<Mutex<CudaModule>>> {
    let cache = get_kernel_cache();

    // Fast path: check if already cached
    {
        let cache_guard = cache.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Cache lock poisoned: {}", e))
        })?;
        if let Some(module) = cache_guard.get(key) {
            KERNEL_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            return Ok(Arc::clone(module));
        }
    }

    // Slow path: compile and cache
    KERNEL_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
    eprintln!("[KERNEL-CACHE] Compiling: {}", key);

    let module = CudaModule::from_ptx(ctx, ptx)?;
    let module_arc = Arc::new(Mutex::new(module));

    // Insert into cache
    {
        let mut cache_guard = cache.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Cache lock poisoned: {}", e))
        })?;
        cache_guard.insert(key.to_string(), Arc::clone(&module_arc));
    }

    Ok(module_arc)
}

/// Get kernel cache statistics
#[cfg(feature = "cuda")]
#[must_use]
pub fn kernel_cache_hits() -> u64 {
    KERNEL_CACHE_HITS.load(Ordering::Relaxed)
}

/// Get kernel cache miss count
#[cfg(feature = "cuda")]
#[must_use]
pub fn kernel_cache_misses() -> u64 {
    KERNEL_CACHE_MISSES.load(Ordering::Relaxed)
}

/// Reset kernel cache statistics
#[cfg(feature = "cuda")]
pub fn reset_kernel_cache_stats() {
    KERNEL_CACHE_HITS.store(0, Ordering::Relaxed);
    KERNEL_CACHE_MISSES.store(0, Ordering::Relaxed);
}

/// Clear the kernel cache (useful for testing)
#[cfg(feature = "cuda")]
pub fn clear_kernel_cache() {
    if let Some(cache) = KERNEL_CACHE.get() {
        if let Ok(mut guard) = cache.lock() {
            guard.clear();
        }
    }
    reset_kernel_cache_stats();
}

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
        TOTAL_H2D_TRANSFERS.fetch_add(1, Ordering::Relaxed);
        TOTAL_H2D_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);

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
        TOTAL_D2H_TRANSFERS.fetch_add(1, Ordering::Relaxed);
        TOTAL_D2H_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);

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
        let use_wmma = k >= 64 && m >= 64 && n >= 64; // WAPR-PERF-010: Tensor Cores for large matrices
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

        // BiasActivationKernel is IN-PLACE: reads from output, adds bias, writes to output
        // So we first copy input to output buffer, then run kernel in-place
        let output_buffer = self.buffer.clone(ctx)?;

        use crate::kernels::BiasActivationKernel;
        let kernel = BiasActivationKernel::new(n as u32, bias_size as u32);
        let ptx = kernel.emit_ptx();
        let cache_key = format!("bias_add:{}:{}", n, bias_size);
        let module_arc = get_or_compile_kernel(ctx, &cache_key, &ptx)?;
        let stream = CudaStream::new(ctx)?;

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
// Batched Multi-Head Attention (GPU-Resident)
// ============================================================================

// Note: Batched4DGemmKernel available for optimized multi-head attention (future)
#[allow(unused_imports)]
#[cfg(feature = "cuda")]
use crate::kernels::Batched4DGemmKernel;

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
    let use_wmma = k >= 64 && n >= 16 && m >= 16; // WAPR-PERF-011: Tensor Cores for suitable dimensions

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
// GPU-Resident Encoder Block (Total Offload)
// ============================================================================

/// Weights for a single GPU-resident encoder block
///
/// Pre-upload all weights to GPU at model load time.
/// Then run forward passes with ZERO host transfers.
#[cfg(feature = "cuda")]
pub struct GpuEncoderBlockWeights {
    /// Layer norm 1: gamma [d_model]
    pub ln1_gamma: GpuResidentTensor<f32>,
    /// Layer norm 1: beta [d_model]
    pub ln1_beta: GpuResidentTensor<f32>,
    /// Query projection: weight [d_model, d_model]
    pub w_q: GpuResidentTensor<f32>,
    /// Query projection: bias [d_model]
    pub b_q: GpuResidentTensor<f32>,
    /// Key projection: weight [d_model, d_model]
    pub w_k: GpuResidentTensor<f32>,
    /// Key projection: bias [d_model]
    pub b_k: GpuResidentTensor<f32>,
    /// Value projection: weight [d_model, d_model]
    pub w_v: GpuResidentTensor<f32>,
    /// Value projection: bias [d_model]
    pub b_v: GpuResidentTensor<f32>,
    /// Output projection: weight [d_model, d_model]
    pub w_o: GpuResidentTensor<f32>,
    /// Output projection: bias [d_model]
    pub b_o: GpuResidentTensor<f32>,
    /// Layer norm 2: gamma [d_model]
    pub ln2_gamma: GpuResidentTensor<f32>,
    /// Layer norm 2: beta [d_model]
    pub ln2_beta: GpuResidentTensor<f32>,
    /// FFN up projection: weight [d_model, ffn_dim]
    pub ffn_up_w: GpuResidentTensor<f32>,
    /// FFN up projection: bias [ffn_dim]
    pub ffn_up_b: GpuResidentTensor<f32>,
    /// FFN down projection: weight [ffn_dim, d_model]
    pub ffn_down_w: GpuResidentTensor<f32>,
    /// FFN down projection: bias [d_model]
    pub ffn_down_b: GpuResidentTensor<f32>,
}

/// WAPR-PERF-012: GPU Conv Frontend Weights
#[cfg(feature = "cuda")]
pub struct GpuConvFrontendWeights {
    /// Conv1: weight [out_channels, in_channels, kernel_size] = [384, 80, 3]
    pub conv1_weight: GpuResidentTensor<f32>,
    /// Conv1: bias [out_channels] = [384]
    pub conv1_bias: GpuResidentTensor<f32>,
    /// Conv2: weight [out_channels, in_channels, kernel_size] = [384, 384, 3]
    pub conv2_weight: GpuResidentTensor<f32>,
    /// Conv2: bias [out_channels] = [384]
    pub conv2_bias: GpuResidentTensor<f32>,
}

/// WAPR-PERF-013: GPU Decoder Block Weights (similar to encoder but with cross-attention)
#[cfg(feature = "cuda")]
pub struct GpuDecoderBlockWeights {
    // Self-Attention weights
    /// Layer norm 1: gamma [d_model]
    pub ln1_gamma: GpuResidentTensor<f32>,
    /// Layer norm 1: beta [d_model]
    pub ln1_beta: GpuResidentTensor<f32>,
    /// Self-Attention Q: weight [d_model, d_model]
    pub self_w_q: GpuResidentTensor<f32>,
    /// Self-Attention Q: bias [d_model]
    pub self_b_q: GpuResidentTensor<f32>,
    /// Self-Attention K: weight [d_model, d_model]
    pub self_w_k: GpuResidentTensor<f32>,
    /// Self-Attention K: bias [d_model]
    pub self_b_k: GpuResidentTensor<f32>,
    /// Self-Attention V: weight [d_model, d_model]
    pub self_w_v: GpuResidentTensor<f32>,
    /// Self-Attention V: bias [d_model]
    pub self_b_v: GpuResidentTensor<f32>,
    /// Self-Attention O: weight [d_model, d_model]
    pub self_w_o: GpuResidentTensor<f32>,
    /// Self-Attention O: bias [d_model]
    pub self_b_o: GpuResidentTensor<f32>,

    // Cross-Attention weights
    /// Layer norm 2: gamma [d_model]
    pub ln2_gamma: GpuResidentTensor<f32>,
    /// Layer norm 2: beta [d_model]
    pub ln2_beta: GpuResidentTensor<f32>,
    /// Cross-Attention Q: weight [d_model, d_model]
    pub cross_w_q: GpuResidentTensor<f32>,
    /// Cross-Attention Q: bias [d_model]
    pub cross_b_q: GpuResidentTensor<f32>,
    /// Cross-Attention K: weight [d_model, d_model]
    pub cross_w_k: GpuResidentTensor<f32>,
    /// Cross-Attention K: bias [d_model]
    pub cross_b_k: GpuResidentTensor<f32>,
    /// Cross-Attention V: weight [d_model, d_model]
    pub cross_w_v: GpuResidentTensor<f32>,
    /// Cross-Attention V: bias [d_model]
    pub cross_b_v: GpuResidentTensor<f32>,
    /// Cross-Attention O: weight [d_model, d_model]
    pub cross_w_o: GpuResidentTensor<f32>,
    /// Cross-Attention O: bias [d_model]
    pub cross_b_o: GpuResidentTensor<f32>,

    // FFN weights
    /// Layer norm 3: gamma [d_model]
    pub ln3_gamma: GpuResidentTensor<f32>,
    /// Layer norm 3: beta [d_model]
    pub ln3_beta: GpuResidentTensor<f32>,
    /// FFN up projection: weight [d_model, ffn_dim]
    pub ffn_up_w: GpuResidentTensor<f32>,
    /// FFN up projection: bias [ffn_dim]
    pub ffn_up_b: GpuResidentTensor<f32>,
    /// FFN down projection: weight [ffn_dim, d_model]
    pub ffn_down_w: GpuResidentTensor<f32>,
    /// FFN down projection: bias [d_model]
    pub ffn_down_b: GpuResidentTensor<f32>,
}

/// WAPR-PERF-013: GPU-Resident KV Cache for decoder
///
/// Stores K/V tensors on GPU to avoid D2H/H2D transfers during decoding.
#[cfg(feature = "cuda")]
pub struct GpuKvCache {
    /// Key cache [max_seq_len, d_model] - grows incrementally
    pub key: GpuResidentTensor<f32>,
    /// Value cache [max_seq_len, d_model] - grows incrementally
    pub value: GpuResidentTensor<f32>,
    /// Current sequence length (number of tokens cached)
    pub seq_len: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Model dimension
    pub d_model: usize,
}

#[cfg(feature = "cuda")]
impl GpuKvCache {
    /// Create new GPU KV cache
    pub fn new(ctx: &CudaContext, max_seq_len: usize, d_model: usize) -> Result<Self> {
        let total_size = max_seq_len * d_model;
        let zeros = vec![0.0f32; total_size];

        let key = GpuResidentTensor::from_host(ctx, &zeros)?;
        let value = GpuResidentTensor::from_host(ctx, &zeros)?;

        Ok(Self {
            key,
            value,
            seq_len: 0,
            max_seq_len,
            d_model,
        })
    }

    /// Reset cache (for new sequence)
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Get current sequence length
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }
}

/// Configuration for GPU decoder
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct GpuDecoderConfig {
    /// Model dimension (d_model)
    pub d_model: u32,
    /// Number of attention heads
    pub n_heads: u32,
    /// FFN hidden dimension (typically 4 * d_model)
    pub ffn_dim: u32,
    /// Maximum sequence length
    pub max_seq_len: u32,
    /// Number of decoder layers
    pub n_layers: u32,
}

/// Configuration for GPU encoder
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct GpuEncoderConfig {
    /// Model dimension (d_model)
    pub d_model: u32,
    /// Number of attention heads
    pub n_heads: u32,
    /// FFN hidden dimension (typically 4 * d_model)
    pub ffn_dim: u32,
}

/// Forward pass through one encoder block (100% GPU-resident)
///
/// Architecture: Pre-norm with residual connections
/// x + Attention(LN(x)) then x + FFN(LN(x))
///
/// # Arguments
/// * `ctx` - CUDA context
/// * `x` - Input tensor [seq_len * d_model] on GPU
/// * `weights` - Pre-uploaded encoder block weights
/// * `config` - Encoder configuration
/// * `seq_len` - Sequence length
///
/// # Returns
/// Output tensor [seq_len * d_model] on GPU
#[cfg(feature = "cuda")]
pub fn forward_encoder_block_gpu(
    ctx: &CudaContext,
    x: &GpuResidentTensor<f32>,
    weights: &GpuEncoderBlockWeights,
    config: &GpuEncoderConfig,
) -> Result<GpuResidentTensor<f32>> {
    let d_model = config.d_model;
    let n_heads = config.n_heads;
    let head_dim = d_model / n_heads;
    let ffn_dim = config.ffn_dim;
    let seq_len = (x.len() / d_model as usize) as u32;

    // Debug flag for intermediate value inspection
    let debug = std::env::var("WHISPER_DEBUG_GPU_INTERNALS").is_ok();

    // ====== Self-Attention Block ======

    // Pre-norm: x_norm = LayerNorm(x)
    let x_norm = x.layer_norm(ctx, &weights.ln1_gamma, &weights.ln1_beta, d_model, seq_len)?;

    if debug {
        let ln1_host = x_norm.peek_host()?;
        let mean = ln1_host.iter().sum::<f32>() / ln1_host.len() as f32;
        let std = (ln1_host.iter().map(|v| v.powi(2)).sum::<f32>() / ln1_host.len() as f32).sqrt();
        eprintln!("[DEBUG-GPU-INTERNAL] LN1 output: mean={:.6}, std={:.6}", mean, std);

        // Check weight matrices
        let wq_host = weights.w_q.peek_host()?;
        let bq_host = weights.b_q.peek_host()?;
        eprintln!("[DEBUG-GPU-INTERNAL] w_q: len={}, mean={:.6}, max={:.6}",
            wq_host.len(),
            wq_host.iter().sum::<f32>() / wq_host.len() as f32,
            wq_host.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
        eprintln!("[DEBUG-GPU-INTERNAL] b_q: len={}, mean={:.6}",
            bq_host.len(),
            bq_host.iter().sum::<f32>() / bq_host.len() as f32);
    }

    // Q, K, V projections (all on GPU)
    let q = x_norm.linear(ctx, &weights.w_q, Some(&weights.b_q), seq_len, d_model, d_model)?;
    let k = x_norm.linear(ctx, &weights.w_k, Some(&weights.b_k), seq_len, d_model, d_model)?;
    let v = x_norm.linear(ctx, &weights.w_v, Some(&weights.b_v), seq_len, d_model, d_model)?;

    if debug {
        let q_host = q.peek_host()?;
        let k_host = k.peek_host()?;
        let v_host = v.peek_host()?;
        eprintln!("[DEBUG-GPU-INTERNAL] Q: mean={:.6}, K: mean={:.6}, V: mean={:.6}",
            q_host.iter().sum::<f32>() / q_host.len() as f32,
            k_host.iter().sum::<f32>() / k_host.len() as f32,
            v_host.iter().sum::<f32>() / v_host.len() as f32);
    }

    // Multi-head attention (on GPU)
    // WAPR-PERF-008: Batched attention (reduces 54 kernel launches to 9, correct output)
    let attn_out = batched_multihead_attention_optimized(ctx, &q, &k, &v, n_heads, head_dim, seq_len)?;

    if debug {
        let attn_host = attn_out.peek_host()?;
        eprintln!("[DEBUG-GPU-INTERNAL] attn_out: mean={:.6}, std={:.6}",
            attn_host.iter().sum::<f32>() / attn_host.len() as f32,
            (attn_host.iter().map(|v| v.powi(2)).sum::<f32>() / attn_host.len() as f32).sqrt());
    }

    // Output projection
    let attn_proj = attn_out.linear(ctx, &weights.w_o, Some(&weights.b_o), seq_len, d_model, d_model)?;

    if debug {
        let proj_host = attn_proj.peek_host()?;
        eprintln!("[DEBUG-GPU-INTERNAL] attn_proj: mean={:.6}, std={:.6}",
            proj_host.iter().sum::<f32>() / proj_host.len() as f32,
            (proj_host.iter().map(|v| v.powi(2)).sum::<f32>() / proj_host.len() as f32).sqrt());
    }

    // Residual connection: x + attn_proj
    let residual1 = x.add(ctx, &attn_proj)?;

    if debug {
        let res1_host = residual1.peek_host()?;
        eprintln!("[DEBUG-GPU-INTERNAL] residual1: mean={:.6}, std={:.6}",
            res1_host.iter().sum::<f32>() / res1_host.len() as f32,
            (res1_host.iter().map(|v| v.powi(2)).sum::<f32>() / res1_host.len() as f32).sqrt());
    }

    // ====== FFN Block ======

    // Pre-norm: x_norm2 = LayerNorm(residual1)
    let x_norm2 = residual1.layer_norm(ctx, &weights.ln2_gamma, &weights.ln2_beta, d_model, seq_len)?;

    if debug {
        let ln2_host = x_norm2.peek_host()?;
        eprintln!("[DEBUG-GPU-INTERNAL] LN2 output: mean={:.6}, std={:.6}",
            ln2_host.iter().sum::<f32>() / ln2_host.len() as f32,
            (ln2_host.iter().map(|v| v.powi(2)).sum::<f32>() / ln2_host.len() as f32).sqrt());
    }

    // FFN up projection + GELU (FUSED - WAPR-PERF-007)
    // Uses single kernel instead of 3 (GEMM + Bias + GELU)
    let ffn_gelu = x_norm2.fused_linear_gelu(
        ctx, &weights.ffn_up_w, &weights.ffn_up_b, seq_len, d_model, ffn_dim
    )?;

    if debug {
        let gelu_host = ffn_gelu.peek_host()?;
        eprintln!("[DEBUG-GPU-INTERNAL] ffn_gelu (fused): mean={:.6}, std={:.6}",
            gelu_host.iter().sum::<f32>() / gelu_host.len() as f32,
            (gelu_host.iter().map(|v| v.powi(2)).sum::<f32>() / gelu_host.len() as f32).sqrt());
    }

    // FFN down projection
    let ffn_down = ffn_gelu.linear(ctx, &weights.ffn_down_w, Some(&weights.ffn_down_b), seq_len, ffn_dim, d_model)?;

    if debug {
        let down_host = ffn_down.peek_host()?;
        eprintln!("[DEBUG-GPU-INTERNAL] ffn_down: mean={:.6}, std={:.6}",
            down_host.iter().sum::<f32>() / down_host.len() as f32,
            (down_host.iter().map(|v| v.powi(2)).sum::<f32>() / down_host.len() as f32).sqrt());
    }

    // Residual connection: residual1 + ffn_down
    let output = residual1.add(ctx, &ffn_down)?;

    if debug {
        let out_host = output.peek_host()?;
        eprintln!("[DEBUG-GPU-INTERNAL] block_output: mean={:.6}, std={:.6}",
            out_host.iter().sum::<f32>() / out_host.len() as f32,
            (out_host.iter().map(|v| v.powi(2)).sum::<f32>() / out_host.len() as f32).sqrt());
    }

    Ok(output)
}

// ============================================================================
// Transfer Statistics Summary
// ============================================================================

/// Summary of GPU transfer statistics
#[derive(Debug, Clone, Default)]
pub struct TransferStats {
    /// Total host-to-device transfers
    pub h2d_transfers: u64,
    /// Total device-to-host transfers
    pub d2h_transfers: u64,
    /// Total bytes transferred host-to-device
    pub h2d_bytes: u64,
    /// Total bytes transferred device-to-host
    pub d2h_bytes: u64,
}

impl TransferStats {
    /// Capture current transfer statistics
    #[must_use]
    pub fn capture() -> Self {
        Self {
            h2d_transfers: total_h2d_transfers(),
            d2h_transfers: total_d2h_transfers(),
            h2d_bytes: total_h2d_bytes(),
            d2h_bytes: total_d2h_bytes(),
        }
    }

    /// Calculate delta from a previous snapshot
    #[must_use]
    pub fn delta_from(&self, prev: &Self) -> Self {
        Self {
            h2d_transfers: self.h2d_transfers.saturating_sub(prev.h2d_transfers),
            d2h_transfers: self.d2h_transfers.saturating_sub(prev.d2h_transfers),
            h2d_bytes: self.h2d_bytes.saturating_sub(prev.h2d_bytes),
            d2h_bytes: self.d2h_bytes.saturating_sub(prev.d2h_bytes),
        }
    }

    /// Total transfers (H2D + D2H)
    #[must_use]
    pub const fn total_transfers(&self) -> u64 {
        self.h2d_transfers + self.d2h_transfers
    }

    /// Total bytes transferred
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.h2d_bytes + self.d2h_bytes
    }
}

impl std::fmt::Display for TransferStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "H2D: {} ({:.2} MB), D2H: {} ({:.2} MB)",
            self.h2d_transfers,
            self.h2d_bytes as f64 / (1024.0 * 1024.0),
            self.d2h_transfers,
            self.d2h_bytes as f64 / (1024.0 * 1024.0)
        )
    }
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

    // WAPR-PERF-013: Sync removed for async pipeline support
    // Caller is responsible for synchronization when needed
    // stream.synchronize()?;

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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_stats_capture_and_delta() {
        reset_transfer_counters();

        let before = TransferStats::capture();
        assert_eq!(before.total_transfers(), 0);

        // Simulate some transfers
        TOTAL_H2D_TRANSFERS.fetch_add(3, Ordering::Relaxed);
        TOTAL_D2H_TRANSFERS.fetch_add(1, Ordering::Relaxed);
        TOTAL_H2D_BYTES.fetch_add(1024, Ordering::Relaxed);
        TOTAL_D2H_BYTES.fetch_add(512, Ordering::Relaxed);

        let after = TransferStats::capture();
        let delta = after.delta_from(&before);

        assert_eq!(delta.h2d_transfers, 3);
        assert_eq!(delta.d2h_transfers, 1);
        assert_eq!(delta.h2d_bytes, 1024);
        assert_eq!(delta.d2h_bytes, 512);
        assert_eq!(delta.total_transfers(), 4);
        assert_eq!(delta.total_bytes(), 1536);
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
        TOTAL_H2D_TRANSFERS.fetch_add(100, Ordering::Relaxed);
        TOTAL_D2H_TRANSFERS.fetch_add(50, Ordering::Relaxed);

        reset_transfer_counters();

        assert_eq!(total_h2d_transfers(), 0);
        assert_eq!(total_d2h_transfers(), 0);
        assert_eq!(total_h2d_bytes(), 0);
        assert_eq!(total_d2h_bytes(), 0);
    }
}
