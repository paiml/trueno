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

pub mod error;
pub mod vector;

pub use error::{Result, TruenoError};
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

/// Select the best available backend for the current platform
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
    // For now, return Scalar (will implement CPU detection later)
    Backend::Scalar
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
        // Currently returns Scalar (baseline implementation)
        assert_eq!(backend, Backend::Scalar);
    }
}
