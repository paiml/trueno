//! Trueno: Multi-Target High-Performance Compute Library
//!
//! **Trueno** (Spanish: "thunder") provides unified, high-performance compute primitives
//! across three execution targets:
//!
//! 1. **CPU SIMD** - x86 (SSE2/AVX/AVX2/AVX-512), ARM (NEON), WASM (SIMD128)
//! 2. **GPU** - Vulkan/Metal/DX12/WebGPU via `wgpu`
//! 3. **WebAssembly** - Portable SIMD128 for browser/edge deployment
//!
//! # Design Principles
//!
//! - **Write once, optimize everywhere**: Single algorithm, multiple backends
//! - **Runtime dispatch**: Auto-select best implementation based on CPU features
//! - **Zero unsafe in public API**: Safety via type system, `unsafe` isolated in backends
//! - **Benchmarked performance**: Every optimization must prove ≥10% speedup
//! - **Extreme TDD**: >90% test coverage, mutation testing, property-based tests
//!
//! # Quick Start
//!
//! ```rust
//! use trueno::Vector;
//!
//! let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
//! let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);
//!
//! // Auto-selects best backend (AVX2/GPU/WASM)
//! let result = a.add(&b).unwrap();
//! assert_eq!(result.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
//! ```

pub mod backends;
pub mod chaos;
pub mod error;
pub mod matrix;
pub mod vector;

pub use error::{Result, TruenoError};
pub use matrix::Matrix;
pub use vector::Vector;

/// Backend execution target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Scalar fallback (no SIMD)
    Scalar,
    /// SSE2 (x86_64 baseline)
    SSE2,
    /// AVX (256-bit)
    AVX,
    /// AVX2 (256-bit with FMA)
    AVX2,
    /// AVX-512 (512-bit)
    AVX512,
    /// ARM NEON
    NEON,
    /// WebAssembly SIMD128
    WasmSIMD,
    /// GPU compute (wgpu)
    GPU,
    /// Auto-select best available
    Auto,
}

impl Backend {
    /// Select the best available backend for the current platform
    ///
    /// This is a convenience wrapper around `select_best_available_backend()`
    pub fn select_best() -> Self {
        select_best_available_backend()
    }
}

/// Operation complexity for GPU dispatch eligibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpComplexity {
    /// Simple operations (add, mul) - prefer SIMD unless very large
    Low = 0,
    /// Moderate operations (dot, reduce) - GPU beneficial at 100K+
    Medium = 1,
    /// Complex operations (matmul, convolution) - GPU beneficial at 10K+
    High = 2,
}

/// Detect best SIMD backend for x86/x86_64 platforms
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn detect_x86_backend() -> Backend {
    if is_x86_feature_detected!("avx512f") {
        return Backend::AVX512;
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return Backend::AVX2;
    }
    if is_x86_feature_detected!("avx") {
        return Backend::AVX;
    }
    if is_x86_feature_detected!("sse2") {
        return Backend::SSE2;
    }
    Backend::Scalar
}

/// Detect best SIMD backend for ARM platforms
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
fn detect_arm_backend() -> Backend {
    #[cfg(target_feature = "neon")]
    {
        Backend::NEON
    }
    #[cfg(not(target_feature = "neon"))]
    {
        Backend::Scalar
    }
}

/// Detect best SIMD backend for WebAssembly
#[cfg(target_arch = "wasm32")]
fn detect_wasm_backend() -> Backend {
    #[cfg(target_feature = "simd128")]
    {
        Backend::WasmSIMD
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        Backend::Scalar
    }
}

/// Select the best available backend for the current platform
///
/// This function performs runtime CPU feature detection and selects the most
/// optimized backend available. The selection follows this priority:
///
/// **x86/x86_64**:
/// 1. AVX-512 (if `avx512f` feature detected)
/// 2. AVX2 (if `avx2` and `fma` features detected)
/// 3. AVX (if `avx` feature detected)
/// 4. SSE2 (baseline for x86_64)
/// 5. Scalar (fallback)
///
/// **ARM**:
/// 1. NEON (if available)
/// 2. Scalar (fallback)
///
/// **WASM**: SIMD128 (if available), else Scalar
///
/// **Other platforms**: Scalar
///
/// # Returns
///
/// The most optimized backend available on this CPU/platform
///
/// # Examples
///
/// ```
/// use trueno::select_best_available_backend;
///
/// let backend = select_best_available_backend();
/// println!("Using backend: {:?}", backend);
/// ```
pub fn select_best_available_backend() -> Backend {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        detect_x86_backend()
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    {
        detect_arm_backend()
    }

    #[cfg(target_arch = "wasm32")]
    {
        detect_wasm_backend()
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "x86",
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "wasm32"
    )))]
    {
        Backend::Scalar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_enum() {
        assert_eq!(Backend::Scalar, Backend::Scalar);
        assert_ne!(Backend::Scalar, Backend::AVX2);
    }

    #[test]
    fn test_op_complexity_ordering() {
        assert!(OpComplexity::Low < OpComplexity::Medium);
        assert!(OpComplexity::Medium < OpComplexity::High);
    }

    #[test]
    fn test_select_best_available_backend() {
        let backend = select_best_available_backend();

        // On x86_64, we should get at least SSE2 (baseline for x86_64)
        // or a more advanced SIMD backend if available
        #[cfg(target_arch = "x86_64")]
        {
            // x86_64 baseline is SSE2, so we should never get Scalar on x86_64
            assert_ne!(backend, Backend::Scalar);
            // Verify it's one of the x86 SIMD backends
            assert!(matches!(
                backend,
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512
            ));
        }

        // On other platforms, we might get Scalar or platform-specific SIMD
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Just verify we got a valid backend
            assert!(matches!(
                backend,
                Backend::Scalar
                    | Backend::SSE2
                    | Backend::AVX
                    | Backend::AVX2
                    | Backend::AVX512
                    | Backend::NEON
                    | Backend::WasmSIMD
            ));
        }
    }

    #[test]
    fn test_backend_selection_is_deterministic() {
        // Backend selection should be deterministic (same result on multiple calls)
        let backend1 = select_best_available_backend();
        let backend2 = select_best_available_backend();
        assert_eq!(backend1, backend2);
    }
}
