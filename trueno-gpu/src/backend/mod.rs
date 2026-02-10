//! Multi-Backend Abstraction
//!
//! Provides a unified interface for different GPU backends:
//! - CUDA (NVIDIA) - Primary, uses PTX
//! - WGPU (WebGPU) - Cross-platform, uses WGSL (Vulkan/Metal/DX12/WebGPU)
//! - Metal (Apple) - Native Apple GPU compute via manzana crate
//! - Vulkan (cross-platform, future)

#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal_shaders;

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
        // Availability based on wgpu feature flag
        cfg!(feature = "wgpu")
    }

    fn device_count(&self) -> usize {
        // Returns 1 if wgpu is available, 0 otherwise (adapter enumeration not yet wired)
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
        assert!(
            backend.device_count() > 0,
            "Should have at least one Metal device"
        );
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
        // Count is 0 when CUDA unavailable, otherwise a positive number
        assert!(backend.is_available() || count == 0);
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
        assert!(
            backend.device_count() >= 1,
            "Should have at least one Metal device"
        );
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

    #[test]
    fn test_wgpu_backend_is_available() {
        let backend = WgpuBackend;
        // Availability depends on wgpu feature flag
        #[cfg(not(feature = "wgpu"))]
        assert!(!backend.is_available());
        #[cfg(feature = "wgpu")]
        {
            // When feature is enabled, should be available
            let _ = backend.is_available(); // Just exercise the path
        }
    }

    #[test]
    fn test_cuda_backend_is_available() {
        let backend = CudaBackend;
        // Without CUDA hardware, should return false
        // This exercises the is_available path
        let available = backend.is_available();
        // The result depends on hardware, but the call should succeed
        let _ = available;
    }

    #[test]
    fn test_cuda_backend_debug() {
        let backend = CudaBackend;
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("CudaBackend"));
    }

    #[test]
    fn test_metal_backend_debug() {
        let backend = MetalBackend;
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("MetalBackend"));
    }

    #[test]
    fn test_vulkan_backend_debug() {
        let backend = VulkanBackend;
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("VulkanBackend"));
    }

    #[test]
    fn test_wgpu_backend_debug() {
        let backend = WgpuBackend;
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("WgpuBackend"));
    }

    #[test]
    fn test_detect_backend_returns_valid_name() {
        let backend = detect_backend();
        // Should always return a backend with a non-empty name
        let name = backend.name();
        assert!(!name.is_empty());
        // Name should be one of the known backends
        let valid_names = ["CUDA", "Metal", "Vulkan", "WGPU"];
        assert!(valid_names.contains(&name), "Unknown backend name");
    }

    #[test]
    fn test_detect_backend_fallback_is_cuda() {
        // When no backends are available, detect_backend should return CUDA
        // as the fallback (for PTX generation)
        let backend = detect_backend();
        // In CI without GPU hardware, this should be CUDA
        // (it's the fallback at line 167)
        let any_available = CudaBackend.is_available()
            || WgpuBackend.is_available()
            || MetalBackend.is_available()
            || VulkanBackend.is_available();
        if !any_available {
            assert_eq!(backend.name(), "CUDA");
        }
    }

    #[test]
    fn test_backend_trait_send_sync() {
        // Verify that all backends implement Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CudaBackend>();
        assert_send_sync::<MetalBackend>();
        assert_send_sync::<VulkanBackend>();
        assert_send_sync::<WgpuBackend>();
    }

    #[test]
    fn test_all_backends_device_count_consistent() {
        // Device count should be 0 when backend is not available
        // Test each backend: if unavailable, count must be 0
        let cuda = CudaBackend;
        let cuda_count = cuda.device_count();
        assert!(cuda.is_available() || cuda_count == 0);

        let metal = MetalBackend;
        let metal_count = metal.device_count();
        assert!(metal.is_available() || metal_count == 0);

        let vulkan = VulkanBackend;
        let vulkan_count = vulkan.device_count();
        assert!(vulkan.is_available() || vulkan_count == 0);

        let wgpu = WgpuBackend;
        let wgpu_count = wgpu.device_count();
        assert!(wgpu.is_available() || wgpu_count == 0);
    }

    #[test]
    fn test_detect_backend_is_deterministic() {
        // Calling detect_backend multiple times should return the same backend
        let backend1 = detect_backend();
        let backend2 = detect_backend();
        assert_eq!(backend1.name(), backend2.name());
    }

    #[test]
    fn test_boxed_backend_trait_object() {
        // Test that backends work correctly as trait objects
        let backends: Vec<Box<dyn Backend>> = vec![
            Box::new(CudaBackend),
            Box::new(MetalBackend),
            Box::new(VulkanBackend),
            Box::new(WgpuBackend),
        ];

        for backend in &backends {
            assert!(!backend.name().is_empty());
            let _ = backend.is_available();
            let _ = backend.device_count();
        }
    }
}
