//! GPU backend using wgpu (Vulkan/Metal/DX12/WebGPU)
//!
//! This backend provides GPU-accelerated compute for large-scale operations.
//! It uses wgpu for cross-platform GPU access and WGSL compute shaders.
//!
//! # Performance
//!
//! GPU backend is optimal for very large workloads (>100K elements for reductions,
//! >1000×1000 for matrix operations) where transfer overhead is amortized.
//!
//! Expected speedups vs SIMD:
//! - Matrix multiplication (large): 10-50x
//! - Reductions (large): 5-20x
//!
//! # Architecture
//!
//! - Device initialization is lazy (first GPU operation)
//! - Compute shaders written in WGSL
//! - Asynchronous execution with pollster for blocking
//! - Automatic fallback to CPU if GPU unavailable
//!
//! # Memory Hierarchy Abstractions
//!
//! - [`TensorView`] - Structured view into GPU memory with shape/stride metadata
//! - [`PartitionView`] - Tiling strategy for efficient GPU work distribution
//!
//! Based on cuda-tile-behavior.md Section 3.2.

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
mod batch;

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
mod device;

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
mod shaders;

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
pub mod runtime;

// Memory hierarchy abstractions (always available, no GPU feature required)
mod partition_view;
mod tensor_view;
mod tiled_reduction;

pub use partition_view::{PartitionView, TileInfo};
pub use tensor_view::{MemoryLayout, TensorView};
pub use tiled_reduction::{
    tiled_max_2d, tiled_min_2d, tiled_reduce_2d, tiled_reduce_partial, tiled_sum_2d, MaxOp, MinOp,
    ReduceOp, SumOp, TILE_SIZE,
};

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub use batch::{BufferId, GpuCommandBatch};

// Export GpuDevice for both native and WASM GPU features
#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
pub use device::GpuDevice;

/// GPU backend for compute operations (native only, uses sync wrappers)
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
#[derive(Clone)]
pub struct GpuBackend {
    device: Option<GpuDevice>,
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
impl GpuBackend {
    /// Create a new GPU backend
    pub fn new() -> Self {
        Self { device: None }
    }

    /// Initialize GPU device (lazy)
    fn ensure_device(&mut self) -> Result<&GpuDevice, String> {
        if self.device.is_none() {
            self.device = Some(GpuDevice::new()?);
        }
        Ok(self.device.as_ref().expect("device initialized above"))
    }

    /// Check if GPU is available
    pub fn is_available() -> bool {
        GpuDevice::is_available()
    }

    /// Vector addition on GPU: c = a + b
    ///
    /// # Arguments
    ///
    /// * `a` - Vector a
    /// * `b` - Vector b
    ///
    /// # Returns
    ///
    /// Vector c (element-wise sum)
    pub fn vec_add(&mut self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
        if a.len() != b.len() {
            return Err(format!(
                "Vector length mismatch: {} != {}",
                a.len(),
                b.len()
            ));
        }

        // wgpu doesn't allow zero-sized buffers
        if a.is_empty() {
            return Err("Cannot perform GPU operation on empty vectors".to_string());
        }

        let device = self.ensure_device()?;

        // Create output buffer
        let mut result = vec![0.0f32; a.len()];

        // Execute GPU compute
        device.vec_add(a, b, &mut result)?;

        Ok(result)
    }

    /// Dot product on GPU: result = sum(a[i] * b[i])
    ///
    /// # Arguments
    ///
    /// * `a` - Vector a
    /// * `b` - Vector b
    ///
    /// # Returns
    ///
    /// Scalar dot product result
    pub fn dot(&mut self, a: &[f32], b: &[f32]) -> Result<f32, String> {
        if a.len() != b.len() {
            return Err(format!(
                "Vector length mismatch: {} != {}",
                a.len(),
                b.len()
            ));
        }

        let device = self.ensure_device()?;

        // Execute GPU compute
        device.dot(a, b)
    }

    /// ReLU activation on GPU: result[i] = max(0, input[i])
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector
    ///
    /// # Returns
    ///
    /// Vector with ReLU applied element-wise
    pub fn relu(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let device = self.ensure_device()?;

        // Create output buffer
        let mut result = vec![0.0f32; input.len()];

        // Execute GPU compute
        device.relu(input, &mut result)?;

        Ok(result)
    }

    /// Leaky ReLU activation on GPU: result[i] = max(negative_slope * input[i], input[i])
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector
    /// * `negative_slope` - Slope for negative values (typically 0.01)
    ///
    /// # Returns
    ///
    /// Vector with leaky ReLU applied element-wise
    pub fn leaky_relu(&mut self, input: &[f32], negative_slope: f32) -> Result<Vec<f32>, String> {
        let device = self.ensure_device()?;

        // Create output buffer
        let mut result = vec![0.0f32; input.len()];

        // Execute GPU compute
        device.leaky_relu(input, &mut result, negative_slope)?;

        Ok(result)
    }

    /// ELU activation on GPU: result[i] = x if x > 0, else alpha * (exp(x) - 1)
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector
    /// * `alpha` - Scaling factor for negative values (typically 1.0)
    ///
    /// # Returns
    ///
    /// Vector with ELU applied element-wise
    pub fn elu(&mut self, input: &[f32], alpha: f32) -> Result<Vec<f32>, String> {
        let device = self.ensure_device()?;

        // Create output buffer
        let mut result = vec![0.0f32; input.len()];

        // Execute GPU compute
        device.elu(input, &mut result, alpha)?;

        Ok(result)
    }

    /// Clip (clamp) operation on GPU: result[i] = clamp(input[i], min_val, max_val)
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector
    /// * `min_val` - Minimum value
    /// * `max_val` - Maximum value
    ///
    /// # Returns
    ///
    /// Vector with clip applied element-wise
    pub fn clip(&mut self, input: &[f32], min_val: f32, max_val: f32) -> Result<Vec<f32>, String> {
        let device = self.ensure_device()?;

        // Create output buffer
        let mut result = vec![0.0f32; input.len()];

        // Execute GPU compute
        device.clip(input, &mut result, min_val, max_val)?;

        Ok(result)
    }

    /// Sigmoid activation on GPU: result[i] = 1 / (1 + exp(-input[i]))
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector
    ///
    /// # Returns
    ///
    /// Vector with sigmoid applied element-wise
    pub fn sigmoid(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let device = self.ensure_device()?;

        // Create output buffer
        let mut result = vec![0.0f32; input.len()];

        // Execute GPU compute
        device.sigmoid(input, &mut result)?;

        Ok(result)
    }

    /// Tanh activation on GPU: result[i] = tanh(input[i])
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector
    ///
    /// # Returns
    ///
    /// Vector with tanh applied element-wise
    pub fn tanh(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let device = self.ensure_device()?;

        // Create output buffer
        let mut result = vec![0.0f32; input.len()];

        // Execute GPU compute
        device.tanh(input, &mut result)?;

        Ok(result)
    }

    /// Swish activation on GPU: result[i] = input[i] / (1 + exp(-input[i]))
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector
    ///
    /// # Returns
    ///
    /// Vector with swish applied element-wise
    pub fn swish(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let device = self.ensure_device()?;

        // Create output buffer
        let mut result = vec![0.0f32; input.len()];

        // Execute GPU compute
        device.swish(input, &mut result)?;

        Ok(result)
    }

    /// GELU activation on GPU: result[i] = 0.5 * input[i] * (1 + tanh(...))
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector
    ///
    /// # Returns
    ///
    /// Vector with GELU applied element-wise
    pub fn gelu(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let device = self.ensure_device()?;

        // Create output buffer
        let mut result = vec![0.0f32; input.len()];

        // Execute GPU compute
        device.gelu(input, &mut result)?;

        Ok(result)
    }

    /// Softmax activation on GPU: result[i] = exp(input[i] - max) / sum(exp(input - max))
    ///
    /// Uses multi-pass reduction for numerical stability:
    /// - Pass 1: Max reduction (parallel)
    /// - Pass 2: Exp-subtract (element-wise)
    /// - Pass 3: Sum reduction (parallel)
    /// - Pass 4: Normalize (element-wise)
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector
    ///
    /// # Returns
    ///
    /// Vector with softmax applied element-wise
    pub fn softmax(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let device = self.ensure_device()?;

        // Create output buffer
        let mut result = vec![0.0f32; input.len()];

        // Execute GPU compute
        device.softmax(input, &mut result)?;

        Ok(result)
    }

    /// Log-softmax activation on GPU: result[i] = log(softmax(input)[i])
    ///
    /// Uses multi-pass reduction for numerical stability:
    /// - Pass 1: Max reduction (parallel)
    /// - Pass 2: Exp-subtract (element-wise)
    /// - Pass 3: Sum reduction (parallel)
    /// - Pass 4: Log-normalize (element-wise)
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector
    ///
    /// # Returns
    ///
    /// Vector with log-softmax applied element-wise
    pub fn log_softmax(&mut self, input: &[f32]) -> Result<Vec<f32>, String> {
        let device = self.ensure_device()?;

        // Create output buffer
        let mut result = vec![0.0f32; input.len()];

        // Execute GPU compute
        device.log_softmax(input, &mut result)?;

        Ok(result)
    }

    /// 2D Convolution on GPU: output = input ⊗ kernel
    ///
    /// # Arguments
    ///
    /// * `input` - Input matrix (flattened row-major)
    /// * `kernel` - Convolution kernel (flattened row-major)
    /// * `input_rows` - Number of rows in input
    /// * `input_cols` - Number of columns in input
    /// * `kernel_rows` - Number of rows in kernel
    /// * `kernel_cols` - Number of columns in kernel
    ///
    /// # Returns
    ///
    /// Output matrix (flattened row-major, "valid" convolution)
    /// - output_rows = input_rows - kernel_rows + 1
    /// - output_cols = input_cols - kernel_cols + 1
    pub fn convolve2d(
        &mut self,
        input: &[f32],
        kernel: &[f32],
        input_rows: usize,
        input_cols: usize,
        kernel_rows: usize,
        kernel_cols: usize,
    ) -> Result<Vec<f32>, String> {
        let device = self.ensure_device()?;

        // Calculate output dimensions
        let output_rows = input_rows.saturating_sub(kernel_rows).saturating_add(1);
        let output_cols = input_cols.saturating_sub(kernel_cols).saturating_add(1);

        // Create output buffer
        let mut result = vec![0.0f32; output_rows * output_cols];

        // Execute GPU compute
        device.convolve2d(
            input,
            kernel,
            &mut result,
            input_rows,
            input_cols,
            kernel_rows,
            kernel_cols,
        )?;

        Ok(result)
    }

    /// Matrix multiplication on GPU: C = A × B
    ///
    /// # Arguments
    ///
    /// * `a` - Matrix A (m×k) in row-major order
    /// * `b` - Matrix B (k×n) in row-major order
    /// * `m` - Rows of A and C
    /// * `k` - Cols of A, rows of B
    /// * `n` - Cols of B and C
    ///
    /// # Returns
    ///
    /// Matrix C (m×n) in row-major order
    pub fn matmul(
        &mut self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, String> {
        let device = self.ensure_device()?;

        // Create output buffer
        let mut result = vec![0.0f32; m * n];

        // Execute GPU compute
        device.matmul(a, b, &mut result, m, k, n)?;

        Ok(result)
    }

    /// Symmetric eigendecomposition on GPU
    ///
    /// Computes eigenvalues and eigenvectors using Jacobi algorithm with
    /// GPU-accelerated Givens rotations.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Symmetric matrix data (row-major, n×n)
    /// * `n` - Matrix dimension
    ///
    /// # Returns
    ///
    /// Tuple of (eigenvalues, eigenvector_data) where eigenvector_data is row-major
    pub fn symmetric_eigen(
        &mut self,
        matrix: &[f32],
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let device = self.ensure_device()?;
        device.symmetric_eigen(matrix, n)
    }

    /// 2D Tiled Sum Reduction on GPU
    ///
    /// Uses 16×16 workgroups for efficient parallel reduction with
    /// optimal memory coalescing.
    ///
    /// # Arguments
    ///
    /// * `data` - Input 2D data in row-major order
    /// * `width` - Number of columns
    /// * `height` - Number of rows
    ///
    /// # Returns
    ///
    /// Sum of all elements
    pub fn tiled_sum_2d_gpu(
        &mut self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Result<f32, String> {
        let device = self.ensure_device()?;
        device.tiled_sum_2d(data, width, height)
    }

    /// 2D Tiled Max Reduction on GPU
    ///
    /// Uses 16×16 workgroups for efficient parallel max reduction.
    pub fn tiled_max_2d_gpu(
        &mut self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Result<f32, String> {
        let device = self.ensure_device()?;
        device.tiled_max_2d(data, width, height)
    }

    /// 2D Tiled Min Reduction on GPU
    ///
    /// Uses 16×16 workgroups for efficient parallel min reduction.
    pub fn tiled_min_2d_gpu(
        &mut self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Result<f32, String> {
        let device = self.ensure_device()?;
        device.tiled_min_2d(data, width, height)
    }
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
impl Default for GpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

// Stub implementation when GPU feature is disabled or on WASM
#[cfg(any(not(feature = "gpu"), target_arch = "wasm32"))]
#[derive(Clone)]
pub struct GpuBackend;

#[cfg(any(not(feature = "gpu"), target_arch = "wasm32"))]
impl GpuBackend {
    pub fn new() -> Self {
        Self
    }

    pub fn is_available() -> bool {
        false
    }
}

#[cfg(any(not(feature = "gpu"), target_arch = "wasm32"))]
impl Default for GpuBackend {
    fn default() -> Self {
        Self
    }
}

// Tests for stub implementation (when GPU feature is NOT enabled)
#[cfg(test)]
#[cfg(not(feature = "gpu"))]
mod stub_tests {
    use super::*;

    #[test]
    fn test_gpu_backend_stub_new() {
        let _backend = GpuBackend::new();
    }

    #[test]
    fn test_gpu_backend_stub_is_available() {
        assert!(!GpuBackend::is_available());
    }

    #[test]
    fn test_gpu_backend_stub_default() {
        let _backend = GpuBackend;
    }

    #[test]
    fn test_gpu_backend_stub_clone() {
        let backend = GpuBackend::new();
        let _cloned = backend.clone();
    }
}

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests_gpu;
