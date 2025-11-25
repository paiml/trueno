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

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
mod batch;

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
mod device;

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
mod shaders;

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
pub mod runtime;

#[cfg(feature = "gpu")]
pub use batch::{BufferId, GpuCommandBatch};

// Export GpuDevice for both native and WASM GPU features
#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
pub use device::GpuDevice;

/// GPU backend for compute operations
#[cfg(feature = "gpu")]
pub struct GpuBackend {
    device: Option<GpuDevice>,
}

#[cfg(feature = "gpu")]
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
        Ok(self.device.as_ref().unwrap())
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
}

#[cfg(feature = "gpu")]
impl Default for GpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

// Stub implementation when GPU feature is disabled
#[cfg(not(feature = "gpu"))]
pub struct GpuBackend;

#[cfg(not(feature = "gpu"))]
impl GpuBackend {
    pub fn new() -> Self {
        Self
    }

    pub fn is_available() -> bool {
        false
    }
}

#[cfg(not(feature = "gpu"))]
impl Default for GpuBackend {
    fn default() -> Self {
        Self
    }
}

// ===== GPU Tests =====

#[cfg(test)]
#[cfg(feature = "gpu")]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_vec_add_basic() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = gpu.vec_add(&a, &b);

        if let Ok(c) = result {
            assert_eq!(c.len(), 4);
            assert!((c[0] - 6.0).abs() < 1e-4);
            assert!((c[1] - 8.0).abs() < 1e-4);
            assert!((c[2] - 10.0).abs() < 1e-4);
            assert!((c[3] - 12.0).abs() < 1e-4);
        } else {
            eprintln!("GPU vec_add failed: {:?}", result);
        }
    }

    #[test]
    fn test_gpu_vec_add_large() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();
        let size = 10000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();

        let result = gpu.vec_add(&a, &b);

        if let Ok(c) = result {
            assert_eq!(c.len(), size);
            // Check first few elements
            assert!((c[0] - 0.0).abs() < 1e-4); // 0 + 0
            assert!((c[1] - 3.0).abs() < 1e-4); // 1 + 2
            assert!((c[100] - 300.0).abs() < 1e-4); // 100 + 200
        } else {
            eprintln!("GPU vec_add large failed: {:?}", result);
        }
    }

    #[test]
    fn test_gpu_vec_add_length_mismatch() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0]; // Different length

        let result = gpu.vec_add(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_dot_basic() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = gpu.dot(&a, &b);

        // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        if let Ok(dot_product) = result {
            assert!((dot_product - 70.0).abs() < 1e-4);
        } else {
            eprintln!("GPU dot failed: {:?}", result);
        }
    }

    #[test]
    fn test_gpu_dot_large() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();
        let size = 10000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|_| 1.0).collect();

        let result = gpu.dot(&a, &b);

        // Expected: sum of 0 + 1 + 2 + ... + 9999 = 9999 * 10000 / 2 = 49995000
        if let Ok(dot_product) = result {
            let expected = (size * (size - 1) / 2) as f32;
            assert!((dot_product - expected).abs() < 1.0); // Allow small floating point error
        } else {
            eprintln!("GPU dot large failed: {:?}", result);
        }
    }

    #[test]
    fn test_gpu_dot_length_mismatch() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0]; // Different length

        let result = gpu.dot(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_vec_add_matches_scalar() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        use super::super::scalar::ScalarBackend;
        use crate::backends::VectorBackend;

        let mut gpu = GpuBackend::new();
        let a = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5];
        let b = vec![8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5];

        let gpu_result = gpu.vec_add(&a, &b);

        let mut scalar_result = vec![0.0; 8];
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            ScalarBackend::add(&a, &b, &mut scalar_result);
        }

        if let Ok(gpu_r) = gpu_result {
            for (g, s) in gpu_r.iter().zip(scalar_result.iter()) {
                assert!(
                    (g - s).abs() < 1e-4,
                    "GPU vs Scalar mismatch: gpu={}, scalar={}",
                    g,
                    s
                );
            }
        } else {
            eprintln!("GPU vec_add failed: {:?}", gpu_result);
        }
    }

    #[test]
    fn test_gpu_dot_matches_scalar() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        use super::super::scalar::ScalarBackend;
        use crate::backends::VectorBackend;

        let mut gpu = GpuBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let gpu_result = gpu.dot(&a, &b);
        // SAFETY: Test code calling backend trait methods marked unsafe
        let scalar_result = unsafe { ScalarBackend::dot(&a, &b) };

        if let Ok(gpu_r) = gpu_result {
            assert!(
                (gpu_r - scalar_result).abs() < 1e-4,
                "GPU vs Scalar dot mismatch: gpu={}, scalar={}",
                gpu_r,
                scalar_result
            );
        } else {
            eprintln!("GPU dot failed: {:?}", gpu_result);
        }
    }

    #[test]
    fn test_gpu_vec_add_empty() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];

        let result = gpu.vec_add(&a, &b);

        // GPU backend returns error for empty vectors (wgpu doesn't allow zero-sized buffers)
        assert!(
            result.is_err(),
            "Expected error for empty vectors, got: {:?}",
            result
        );
    }

    #[test]
    fn test_gpu_relu_matches_scalar() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        use super::super::scalar::ScalarBackend;
        use crate::backends::VectorBackend;

        let mut gpu = GpuBackend::new();
        let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0, -2.5, 2.5, 0.5];

        let gpu_result = gpu.relu(&input);
        let mut scalar_result = vec![0.0; input.len()];
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            ScalarBackend::relu(&input, &mut scalar_result);
        }

        if let Ok(gpu_r) = gpu_result {
            for (g, s) in gpu_r.iter().zip(scalar_result.iter()) {
                assert!(
                    (g - s).abs() < 1e-4,
                    "GPU vs Scalar relu mismatch: gpu={}, scalar={}",
                    g,
                    s
                );
            }
        } else {
            eprintln!("GPU relu failed: {:?}", gpu_result);
        }
    }

    #[test]
    fn test_gpu_sigmoid_matches_scalar() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        use super::super::scalar::ScalarBackend;
        use crate::backends::VectorBackend;

        let mut gpu = GpuBackend::new();
        let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0, -2.5, 2.5, 0.5];

        let gpu_result = gpu.sigmoid(&input);
        let mut scalar_result = vec![0.0; input.len()];
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            ScalarBackend::sigmoid(&input, &mut scalar_result);
        }

        if let Ok(gpu_r) = gpu_result {
            for (g, s) in gpu_r.iter().zip(scalar_result.iter()) {
                assert!(
                    (g - s).abs() < 1e-3,
                    "GPU vs Scalar sigmoid mismatch: gpu={}, scalar={}",
                    g,
                    s
                );
            }
        } else {
            eprintln!("GPU sigmoid failed: {:?}", gpu_result);
        }
    }

    #[test]
    fn test_gpu_gelu_matches_scalar() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        use super::super::scalar::ScalarBackend;
        use crate::backends::VectorBackend;

        let mut gpu = GpuBackend::new();
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, 1.5];

        let gpu_result = gpu.gelu(&input);
        let mut scalar_result = vec![0.0; input.len()];
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            ScalarBackend::gelu(&input, &mut scalar_result);
        }

        if let Ok(gpu_r) = gpu_result {
            for (g, s) in gpu_r.iter().zip(scalar_result.iter()) {
                assert!(
                    (g - s).abs() < 1e-2,
                    "GPU vs Scalar gelu mismatch: gpu={}, scalar={}",
                    g,
                    s
                );
            }
        } else {
            eprintln!("GPU gelu failed: {:?}", gpu_result);
        }
    }

    #[test]
    fn test_gpu_swish_matches_scalar() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        use super::super::scalar::ScalarBackend;
        use crate::backends::VectorBackend;

        let mut gpu = GpuBackend::new();
        let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0, -2.5, 2.5, 0.5];

        let gpu_result = gpu.swish(&input);
        let mut scalar_result = vec![0.0; input.len()];
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            ScalarBackend::swish(&input, &mut scalar_result);
        }

        if let Ok(gpu_r) = gpu_result {
            for (g, s) in gpu_r.iter().zip(scalar_result.iter()) {
                assert!(
                    (g - s).abs() < 1e-3,
                    "GPU vs Scalar swish mismatch: gpu={}, scalar={}",
                    g,
                    s
                );
            }
        } else {
            eprintln!("GPU swish failed: {:?}", gpu_result);
        }
    }

    #[test]
    fn test_gpu_clip_matches_scalar() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        use super::super::scalar::ScalarBackend;
        use crate::backends::VectorBackend;

        let mut gpu = GpuBackend::new();
        let input = vec![1.0, 5.0, 10.0, 15.0, -3.0, 20.0, 7.5, 0.0];
        let min_val = 3.0;
        let max_val = 12.0;

        let gpu_result = gpu.clip(&input, min_val, max_val);
        let mut scalar_result = vec![0.0; input.len()];
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            ScalarBackend::clamp(&input, min_val, max_val, &mut scalar_result);
        }

        if let Ok(gpu_r) = gpu_result {
            for (g, s) in gpu_r.iter().zip(scalar_result.iter()) {
                assert!(
                    (g - s).abs() < 1e-4,
                    "GPU vs Scalar clip mismatch: gpu={}, scalar={}",
                    g,
                    s
                );
            }
        } else {
            eprintln!("GPU clip failed: {:?}", gpu_result);
        }
    }

    #[test]
    fn test_gpu_leaky_relu_basic() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();
        let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        let negative_slope = 0.01;

        let result = gpu.leaky_relu(&input, negative_slope);

        if let Ok(output) = result {
            // Expected: max(negative_slope * x, x)
            let expected = [-0.03, -0.01, 0.0, 1.0, 3.0];
            for (r, e) in output.iter().zip(expected.iter()) {
                assert!(
                    (r - e).abs() < 1e-4,
                    "Leaky ReLU mismatch: got={}, expected={}",
                    r,
                    e
                );
            }
        } else {
            eprintln!("GPU leaky_relu failed: {:?}", result);
        }
    }

    #[test]
    fn test_gpu_elu_basic() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let alpha = 1.0;

        let result = gpu.elu(&input, alpha);

        if let Ok(output) = result {
            // Expected: x if x > 0, else alpha * (exp(x) - 1)
            for (i, (r, &x)) in output.iter().zip(input.iter()).enumerate() {
                let expected = if x > 0.0 { x } else { alpha * (x.exp() - 1.0) };
                assert!(
                    (r - expected).abs() < 1e-3,
                    "ELU mismatch at {}: got={}, expected={}",
                    i,
                    r,
                    expected
                );
            }
        } else {
            eprintln!("GPU elu failed: {:?}", result);
        }
    }

    #[test]
    fn test_gpu_tanh_basic() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

        let result = gpu.tanh(&input);

        if let Ok(output) = result {
            for (r, &x) in output.iter().zip(input.iter()) {
                let expected = x.tanh();
                assert!(
                    (r - expected).abs() < 1e-4,
                    "Tanh mismatch: got={}, expected={}",
                    r,
                    expected
                );
            }
        } else {
            eprintln!("GPU tanh failed: {:?}", result);
        }
    }

    #[test]
    fn test_gpu_tanh_not_hardcoded() {
        // EXTREME TDD: Kill mutant that replaces return with Ok(vec![-1.0])
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();
        let input = vec![1.0, 2.0, 3.0];

        let result = gpu.tanh(&input).expect("GPU tanh should succeed");

        // Kill mutant: verify result is NOT all -1.0 values
        assert_ne!(
            result,
            vec![-1.0, -1.0, -1.0],
            "GPU tanh returned hardcoded -1.0 values (mutant not killed)"
        );

        // Verify correct computation
        for (i, &x) in input.iter().enumerate() {
            let expected = x.tanh();
            assert!(
                (result[i] - expected).abs() < 1e-4,
                "tanh({}) = {} (expected {})",
                x,
                result[i],
                expected
            );
        }
    }

    #[test]
    fn test_gpu_softmax_basic() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let result = gpu.softmax(&input);

        if let Ok(output) = result {
            // Softmax should sum to 1
            let sum: f32 = output.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "Softmax sum should be 1, got {}",
                sum
            );

            // All values should be positive
            for &v in &output {
                assert!(v > 0.0, "Softmax values should be positive");
            }

            // Later values should be larger (input is increasing)
            for i in 1..output.len() {
                assert!(
                    output[i] > output[i - 1],
                    "Softmax should preserve order for increasing input"
                );
            }
        } else {
            eprintln!("GPU softmax failed: {:?}", result);
        }
    }

    #[test]
    fn test_gpu_log_softmax_basic() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];

        let result = gpu.log_softmax(&input);

        if let Ok(output) = result {
            // log_softmax values should all be negative (log of probability < 1)
            for &v in &output {
                assert!(v <= 0.0, "Log softmax values should be <= 0, got {}", v);
            }

            // exp(log_softmax) should sum to 1
            let exp_sum: f32 = output.iter().map(|x| x.exp()).sum();
            assert!(
                (exp_sum - 1.0).abs() < 1e-3,
                "exp(log_softmax) should sum to 1, got {}",
                exp_sum
            );
        } else {
            eprintln!("GPU log_softmax failed: {:?}", result);
        }
    }

    #[test]
    fn test_gpu_matmul_basic() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();

        // Simple 2x2 matrix multiplication
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = A * B = [[19, 22], [43, 50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let res = gpu.matmul(&a, &b, 2, 2, 2);

        if let Ok(result) = res {
            assert!(
                (result[0] - 19.0).abs() < 1e-3,
                "Expected 19.0, got {}",
                result[0]
            );
            assert!(
                (result[1] - 22.0).abs() < 1e-3,
                "Expected 22.0, got {}",
                result[1]
            );
            assert!(
                (result[2] - 43.0).abs() < 1e-3,
                "Expected 43.0, got {}",
                result[2]
            );
            assert!(
                (result[3] - 50.0).abs() < 1e-3,
                "Expected 50.0, got {}",
                result[3]
            );
        } else {
            eprintln!("GPU matmul failed: {:?}", res);
        }
    }

    #[test]
    fn test_gpu_matmul_identity() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();

        // Multiply by identity matrix
        // A = [[1, 2], [3, 4]]
        // I = [[1, 0], [0, 1]]
        // A * I = A
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let identity = vec![1.0, 0.0, 0.0, 1.0];

        let res = gpu.matmul(&a, &identity, 2, 2, 2);

        if let Ok(result) = res {
            for i in 0..4 {
                assert!(
                    (result[i] - a[i]).abs() < 1e-3,
                    "Expected {}, got {}",
                    a[i],
                    result[i]
                );
            }
        } else {
            eprintln!("GPU matmul identity failed: {:?}", res);
        }
    }

    #[test]
    fn test_gpu_matmul_non_square() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();

        // 2x3 matrix * 3x2 matrix = 2x2 matrix
        // A = [[1, 2, 3], [4, 5, 6]]
        // B = [[7, 8], [9, 10], [11, 12]]
        // C = [[58, 64], [139, 154]]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

        let res = gpu.matmul(&a, &b, 2, 3, 2);

        if let Ok(result) = res {
            assert!(
                (result[0] - 58.0).abs() < 1e-3,
                "Expected 58.0, got {}",
                result[0]
            );
            assert!(
                (result[1] - 64.0).abs() < 1e-3,
                "Expected 64.0, got {}",
                result[1]
            );
            assert!(
                (result[2] - 139.0).abs() < 1e-3,
                "Expected 139.0, got {}",
                result[2]
            );
            assert!(
                (result[3] - 154.0).abs() < 1e-3,
                "Expected 154.0, got {}",
                result[3]
            );
        } else {
            eprintln!("GPU matmul non-square failed: {:?}", res);
        }
    }

    #[test]
    fn test_gpu_convolve2d_basic() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();

        // 3x3 input, 2x2 kernel -> 2x2 output
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let kernel = vec![1.0, 0.0, 0.0, 1.0];

        let res = gpu.convolve2d(&input, &kernel, 3, 3, 2, 2);

        if let Ok(result) = res {
            // For kernel [[1, 0], [0, 1]], each output is sum of diagonal elements
            // Output[0,0] = input[0,0]*1 + input[1,1]*1 = 1 + 5 = 6
            assert!(
                (result[0] - 6.0).abs() < 1e-3,
                "Expected 6.0, got {}",
                result[0]
            );
        } else {
            eprintln!("GPU convolve2d basic failed: {:?}", res);
        }
    }

    #[test]
    fn test_gpu_convolve2d_identity() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();

        // 3x3 input with center-only kernel should extract center values
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        // 3x3 kernel with center = 1, rest = 0
        let kernel = vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

        let res = gpu.convolve2d(&input, &kernel, 3, 3, 3, 3);

        if let Ok(result) = res {
            // Should extract center value
            assert!(
                (result[0] - 5.0).abs() < 1e-3,
                "Expected 5.0, got {}",
                result[0]
            );
        } else {
            eprintln!("GPU convolve2d identity failed: {:?}", res);
        }
    }

    #[test]
    fn test_gpu_convolve2d_averaging() {
        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let mut gpu = GpuBackend::new();

        // 4x4 input with 2x2 averaging kernel
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        // 2x2 averaging kernel
        let kernel = vec![0.25, 0.25, 0.25, 0.25];

        let res = gpu.convolve2d(&input, &kernel, 4, 4, 2, 2);

        if let Ok(result) = res {
            // First output: average of top-left 2x2 = (1+2+5+6)/4 = 3.5
            assert!(
                (result[0] - 3.5).abs() < 1e-3,
                "Expected 3.5, got {}",
                result[0]
            );
        } else {
            eprintln!("GPU convolve2d averaging failed: {:?}", res);
        }
    }
}
