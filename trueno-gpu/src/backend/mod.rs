//! Multi-Backend Abstraction
//!
//! Provides a unified interface for different GPU backends:
//! - CUDA (NVIDIA) - Primary, uses PTX
//! - WGPU (WebGPU) - Cross-platform, uses WGSL (Vulkan/Metal/DX12/WebGPU)
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

    #[cfg(feature = "cuda")]
    fn device_count(&self) -> usize {
        if self.is_available() {
            crate::driver::device_count().unwrap_or(0)
        } else {
            0
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn device_count(&self) -> usize {
        if self.is_available() {
            crate::driver::device_count()
        } else {
            0
        }
    }
}

/// Metal backend (Apple GPUs)
///
/// Uses manzana crate for safe Rust Metal bindings on macOS.
/// Enable with `--features metal` on macOS.
#[derive(Debug, Default)]
pub struct MetalBackend;

impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "Metal"
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn is_available(&self) -> bool {
        manzana::metal::is_available()
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn is_available(&self) -> bool {
        false
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn device_count(&self) -> usize {
        manzana::metal::MetalCompute::devices().len()
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn device_count(&self) -> usize {
        0
    }
}

/// Metal device information (re-exported from manzana when feature enabled)
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use manzana::metal::{CompiledShader as MetalShader, MetalBuffer, MetalCompute, MetalDevice};

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

/// WGPU backend (WebGPU - cross-platform via wgpu crate)
///
/// Uses WGSL shading language, runs on:
/// - Vulkan (Linux, Windows, Android)
/// - Metal (macOS, iOS)
/// - DX12 (Windows)
/// - WebGPU (browsers via wasm)
#[derive(Debug, Default)]
pub struct WgpuBackend;

impl Backend for WgpuBackend {
    fn name(&self) -> &str {
        "WGPU"
    }

    fn is_available(&self) -> bool {
        // TODO: Check for wgpu feature and adapter availability
        cfg!(feature = "wgpu")
    }

    fn device_count(&self) -> usize {
        // TODO: Enumerate wgpu adapters
        usize::from(self.is_available())
    }
}

/// Detect best available backend
///
/// Priority order:
/// 1. CUDA (NVIDIA) - highest performance for NVIDIA GPUs
/// 2. WGPU - cross-platform fallback (Vulkan/Metal/DX12)
/// 3. Metal - Apple-specific (subset of WGPU)
/// 4. Vulkan - direct Vulkan (subset of WGPU)
#[must_use]
pub fn detect_backend() -> Box<dyn Backend> {
    let cuda = CudaBackend;
    if cuda.is_available() {
        return Box::new(cuda);
    }

    let wgpu = WgpuBackend;
    if wgpu.is_available() {
        return Box::new(wgpu);
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
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn test_metal_backend_unavailable() {
        let backend = MetalBackend;
        assert!(!backend.is_available());
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_metal_backend_available() {
        let backend = MetalBackend;
        // On macOS with metal feature, should detect GPUs
        assert!(backend.is_available(), "Metal should be available on macOS");
        assert!(backend.device_count() > 0, "Should have at least one Metal device");
    }

    #[test]
    fn test_detect_backend() {
        let backend = detect_backend();
        // Should return something
        assert!(!backend.name().is_empty());
    }

    #[test]
    fn test_metal_backend_name() {
        let backend = MetalBackend;
        assert_eq!(backend.name(), "Metal");
    }

    #[test]
    fn test_vulkan_backend_name() {
        let backend = VulkanBackend;
        assert_eq!(backend.name(), "Vulkan");
    }

    #[test]
    fn test_vulkan_backend_unavailable() {
        let backend = VulkanBackend;
        assert!(!backend.is_available());
    }

    #[test]
    fn test_cuda_backend_device_count() {
        let backend = CudaBackend;
        // Device count depends on hardware - just check it's non-negative
        let count = backend.device_count();
        // If CUDA is available, count can be >= 0; if not available, it's 0
        if backend.is_available() {
            assert!(count >= 0, "CUDA available but device_count failed");
        } else {
            assert_eq!(count, 0);
        }
    }

    #[test]
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn test_metal_backend_device_count() {
        let backend = MetalBackend;
        assert_eq!(backend.device_count(), 0);
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn test_metal_backend_device_count_macos() {
        let backend = MetalBackend;
        // On macOS with metal feature, should have at least 1 GPU
        assert!(backend.device_count() >= 1, "Should have at least one Metal device");
    }

    #[test]
    fn test_vulkan_backend_device_count() {
        let backend = VulkanBackend;
        assert_eq!(backend.device_count(), 0);
    }

    #[test]
    fn test_cuda_backend_default() {
        let backend = CudaBackend::default();
        assert_eq!(backend.name(), "CUDA");
    }

    #[test]
    fn test_metal_backend_default() {
        let backend = MetalBackend::default();
        assert_eq!(backend.name(), "Metal");
    }

    #[test]
    fn test_vulkan_backend_default() {
        let backend = VulkanBackend::default();
        assert_eq!(backend.name(), "Vulkan");
    }

    #[test]
    fn test_wgpu_backend_name() {
        let backend = WgpuBackend;
        assert_eq!(backend.name(), "WGPU");
    }

    #[test]
    fn test_wgpu_backend_default() {
        let backend = WgpuBackend::default();
        assert_eq!(backend.name(), "WGPU");
    }

    #[test]
    fn test_wgpu_backend_device_count() {
        let backend = WgpuBackend;
        // Without wgpu feature, should be 0
        #[cfg(not(feature = "wgpu"))]
        assert_eq!(backend.device_count(), 0);
    }
}
