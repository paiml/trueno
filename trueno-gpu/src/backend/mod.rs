//! Multi-Backend Abstraction
//!
//! Provides a unified interface for different GPU backends:
//! - CUDA (NVIDIA)
//! - Metal (Apple, future)
//! - Vulkan (cross-platform, future)

/// Backend trait for GPU operations
pub trait Backend: Send + Sync {
    /// Backend name
    fn name(&self) -> &str;

    /// Check if backend is available
    fn is_available(&self) -> bool;

    /// Get device count
    fn device_count(&self) -> usize;
}

/// CUDA backend (NVIDIA GPUs)
#[derive(Debug, Default)]
pub struct CudaBackend;

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "CUDA"
    }

    fn is_available(&self) -> bool {
        crate::driver::cuda_available()
    }

    fn device_count(&self) -> usize {
        if self.is_available() {
            // TODO: Query actual device count
            0
        } else {
            0
        }
    }
}

/// Metal backend (Apple GPUs) - placeholder
#[derive(Debug, Default)]
pub struct MetalBackend;

impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "Metal"
    }

    fn is_available(&self) -> bool {
        false // Not implemented yet
    }

    fn device_count(&self) -> usize {
        0
    }
}

/// Vulkan backend (cross-platform) - placeholder
#[derive(Debug, Default)]
pub struct VulkanBackend;

impl Backend for VulkanBackend {
    fn name(&self) -> &str {
        "Vulkan"
    }

    fn is_available(&self) -> bool {
        false // Not implemented yet
    }

    fn device_count(&self) -> usize {
        0
    }
}

/// Detect best available backend
#[must_use]
pub fn detect_backend() -> Box<dyn Backend> {
    let cuda = CudaBackend;
    if cuda.is_available() {
        return Box::new(cuda);
    }

    let metal = MetalBackend;
    if metal.is_available() {
        return Box::new(metal);
    }

    let vulkan = VulkanBackend;
    if vulkan.is_available() {
        return Box::new(vulkan);
    }

    // Return CUDA as default (even if unavailable) for PTX generation
    Box::new(CudaBackend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_backend_name() {
        let backend = CudaBackend;
        assert_eq!(backend.name(), "CUDA");
    }

    #[test]
    fn test_metal_backend_unavailable() {
        let backend = MetalBackend;
        assert!(!backend.is_available());
    }

    #[test]
    fn test_detect_backend() {
        let backend = detect_backend();
        // Should return something
        assert!(!backend.name().is_empty());
    }
}
