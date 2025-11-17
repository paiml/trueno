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

#[cfg(feature = "gpu")]
mod device;

#[cfg(feature = "gpu")]
mod shaders;

#[cfg(feature = "gpu")]
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
