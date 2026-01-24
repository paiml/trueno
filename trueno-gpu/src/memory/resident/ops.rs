//! GPU-Resident Tensor Operations (f32 specialization)
//!
//! This module contains all f32-specialized operations for `GpuResidentTensor`.
//! Operations include GEMM, Softmax, LayerNorm, GELU, etc.
//!
//! ## Design
//!
//! All operations:
//! - Stay on GPU (no implicit host transfers)
//! - Use kernel caching for performance
//! - Support both synchronous and stream-based async variants
//!
//! ## Usage
//!
//! ```ignore
//! let result = tensor.matmul(&ctx, &other, m, n, k)?;
//! let activated = result.gelu(&ctx)?;
//! ```

#![allow(clippy::similar_names)]

#[cfg(feature = "cuda")]
use crate::driver::{CudaContext, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use crate::error::Result;
#[cfg(feature = "cuda")]
use crate::kernels::{GemmKernel, Kernel, LongRowSoftmaxKernel, ScaleKernel, SoftmaxKernel};

#[cfg(feature = "cuda")]
use super::cache::get_or_compile_kernel;
#[cfg(feature = "cuda")]
use super::GpuResidentTensor;

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
