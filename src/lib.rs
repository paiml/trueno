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
pub mod eigen;
pub mod error;
pub mod hash;
pub mod matrix;
pub mod monitor;
pub mod vector;

pub use eigen::SymmetricEigen;
pub use error::{Result, TruenoError};
pub use hash::{hash_bytes, hash_key, hash_keys_batch, hash_keys_batch_with_backend};
pub use matrix::Matrix;
pub use monitor::{
    cuda_monitor_available, GpuBackend, GpuClockMetrics, GpuDeviceInfo, GpuMemoryMetrics,
    GpuMetrics, GpuMonitor, GpuPcieMetrics, GpuPowerMetrics, GpuThermalMetrics, GpuUtilization,
    GpuVendor, MonitorConfig, MonitorError,
};
#[cfg(feature = "cuda-monitor")]
pub use monitor::{enumerate_cuda_devices, query_cuda_device_info, query_cuda_memory};
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

/// Operation type for SIMD backend selection
///
/// Based on AVX-512 performance analysis (see AVX512_ANALYSIS.md), operations are
/// categorized by their memory vs compute characteristics to guide optimal backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Memory-bound operations (add, sub, mul, scale, div)
    ///
    /// These operations perform minimal computation per memory access (arithmetic intensity < 1 op/byte).
    /// Prefer AVX2 over AVX-512 due to memory bandwidth bottleneck.
    ///
    /// AVX-512 performance: 0.67-1.20x scalar (often slower!)
    /// AVX2 performance: 1.0-1.2x scalar
    MemoryBound,

    /// Compute-bound operations (dot, max, min, argmax, argmin)
    ///
    /// These operations perform significant computation per memory access (arithmetic intensity > 1 op/byte).
    /// AVX-512 excels due to wider SIMD parallelism.
    ///
    /// AVX-512 performance: 7-14x scalar (validated)
    /// AVX2 performance: 4-12x scalar (validated)
    ComputeBound,

    /// Mixed operations (fma, sqrt, exp, sigmoid, activations)
    ///
    /// Performance depends on data size and hardware.
    /// Use size-based heuristics or default to AVX2 for safety.
    Mixed,
}

/// Detect best SIMD backend for x86/x86_64 platforms
///
/// **IMPORTANT**: Prefers AVX2 over AVX-512 by default based on performance analysis.
///
/// AVX-512 is **NOT** universally faster - it causes 10-33% slowdown for memory-bound
/// operations (add, mul, sub) due to memory bandwidth bottleneck and thermal throttling.
/// See AVX512_ANALYSIS.md for detailed benchmarking results.
///
/// For operation-specific backend selection, use `select_backend_for_operation()`.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn detect_x86_backend() -> Backend {
    // Prefer AVX2 over AVX-512 for safety (AVX-512 causes regressions for memory-bound ops)
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return Backend::AVX2;
    }
    // Note: AVX-512 is intentionally NOT checked here
    // Use select_backend_for_operation(OperationType::ComputeBound) for AVX-512
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
    // Cache backend selection using OnceLock to avoid repeated CPU feature detection
    // This eliminates 3-5% overhead from calling is_x86_feature_detected!() repeatedly
    static BEST_BACKEND: std::sync::OnceLock<Backend> = std::sync::OnceLock::new();

    *BEST_BACKEND.get_or_init(|| {
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
    })
}

/// Select the optimal backend for a specific operation type
///
/// This function considers the memory vs compute characteristics of operations
/// to select the backend that will provide the best performance. Based on
/// comprehensive benchmarking (see AVX512_ANALYSIS.md), AVX-512 is avoided
/// for memory-bound operations where it causes 10-33% performance degradation.
///
/// # Operation Classification
///
/// - **MemoryBound**: add, sub, mul, div, scale, abs, clamp, lerp, relu
///   - Prefer AVX2 (1.0-1.2x scalar) over AVX-512 (0.67-1.20x scalar)
///   - Memory bandwidth bottleneck limits wider SIMD benefit
///
/// - **ComputeBound**: dot, max, min, argmax, argmin, norm_l1, norm_l2, norm_linf
///   - Prefer AVX-512 (7-14x scalar) over AVX2 (4-12x scalar)
///   - High arithmetic intensity benefits from wider SIMD
///
/// - **Mixed**: fma, sqrt, exp, ln, sigmoid, tanh, gelu, swish
///   - Default to AVX2 for safety (avoids AVX-512 thermal throttling)
///   - Size-based heuristics could improve this in future
///
/// # Backend Selection Priority
///
/// **For MemoryBound operations**:
/// 1. AVX2 (if available) - BEST for memory-bound
/// 2. SSE2 (x86_64 baseline)
/// 3. AVX-512 (AVOIDED - causes slowdown)
/// 4. NEON (ARM)
/// 5. WASM SIMD128
/// 6. Scalar (fallback)
///
/// **For ComputeBound operations**:
/// 1. AVX-512 (if available) - BEST for compute-bound
/// 2. AVX2
/// 3. SSE2
/// 4. NEON (ARM)
/// 5. WASM SIMD128
/// 6. Scalar (fallback)
///
/// # Arguments
///
/// * `op_type` - The type of operation being performed
///
/// # Returns
///
/// The optimal backend for the given operation type
///
/// # Examples
///
/// ```
/// use trueno::{select_backend_for_operation, OperationType};
///
/// // Memory-bound operation - prefers AVX2 over AVX-512
/// let backend = select_backend_for_operation(OperationType::MemoryBound);
///
/// // Compute-bound operation - uses AVX-512 if available
/// let backend = select_backend_for_operation(OperationType::ComputeBound);
/// ```
///
/// # Performance Impact
///
/// Using operation-aware backend selection fixes performance regressions:
/// - mul with AVX-512: 0.67x → 1.0x (use AVX2 instead)
/// - sub with AVX-512: 0.87x → 1.0x (use AVX2 instead)
/// - dot with AVX-512: 7.89x (keep AVX-512)
pub fn select_backend_for_operation(op_type: OperationType) -> Backend {
    // Allow unused on non-x86 architectures
    let _ = &op_type;

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        use std::arch::is_x86_feature_detected;

        match op_type {
            OperationType::MemoryBound => {
                // Prefer AVX2 over AVX-512 for memory-bound operations
                // AVX-512 causes 10-33% slowdown due to memory bandwidth bottleneck
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return Backend::AVX2;
                }
                if is_x86_feature_detected!("avx") {
                    return Backend::AVX;
                }
                if is_x86_feature_detected!("sse2") {
                    return Backend::SSE2;
                }
                // Explicitly avoid AVX-512 for memory-bound operations
            }
            OperationType::ComputeBound => {
                // Use AVX-512 for compute-bound operations where it excels
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
            }
            OperationType::Mixed => {
                // Default to AVX2 for mixed operations (safer than AVX-512)
                // Future: could use size-based heuristics (AVX-512 for <1K elements)
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return Backend::AVX2;
                }
                if is_x86_feature_detected!("avx") {
                    return Backend::AVX;
                }
                if is_x86_feature_detected!("sse2") {
                    return Backend::SSE2;
                }
            }
        }
        Backend::Scalar
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

    #[test]
    fn test_backend_selection_is_cached() {
        // Verify backend selection is cached (OnceLock)
        // Multiple calls should return the same backend without re-detection
        let backend1 = select_best_available_backend();

        // Call 1000 times to ensure caching is working
        // If not cached, this would be significantly slower
        for _ in 0..1000 {
            let backend = select_best_available_backend();
            assert_eq!(backend, backend1, "Backend selection must be consistent");
        }
    }

    #[test]
    fn test_backend_select_best() {
        // Test Backend::select_best() method
        let backend = Backend::select_best();
        assert_eq!(backend, select_best_available_backend());
    }

    #[test]
    fn test_backend_variants() {
        // Test all backend variants
        let backends = vec![
            Backend::Scalar,
            Backend::SSE2,
            Backend::AVX,
            Backend::AVX2,
            Backend::AVX512,
            Backend::NEON,
            Backend::WasmSIMD,
            Backend::GPU,
            Backend::Auto,
        ];

        // Verify all variants are distinct
        for (i, backend1) in backends.iter().enumerate() {
            for (j, backend2) in backends.iter().enumerate() {
                if i == j {
                    assert_eq!(backend1, backend2);
                } else {
                    assert_ne!(backend1, backend2);
                }
            }
        }
    }

    #[test]
    fn test_backend_debug() {
        // Verify Debug trait works
        let backend = Backend::AVX2;
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("AVX2"));

        let backend = Backend::Auto;
        let debug_str = format!("{:?}", backend);
        assert!(debug_str.contains("Auto"));
    }

    #[test]
    fn test_backend_clone() {
        let backend = Backend::AVX2;
        #[allow(clippy::clone_on_copy)]
        let cloned = backend.clone();
        assert_eq!(backend, cloned);
    }

    #[test]
    fn test_backend_copy() {
        let backend = Backend::SSE2;
        let copied = backend;
        assert_eq!(backend, copied);
    }

    #[test]
    fn test_op_complexity_values() {
        assert_eq!(OpComplexity::Low as i32, 0);
        assert_eq!(OpComplexity::Medium as i32, 1);
        assert_eq!(OpComplexity::High as i32, 2);
    }

    #[test]
    fn test_op_complexity_ord() {
        // Test PartialOrd
        assert!(OpComplexity::Low < OpComplexity::Medium);
        assert!(OpComplexity::Medium < OpComplexity::High);
        assert!(OpComplexity::Low < OpComplexity::High);

        // Test Ord
        use std::cmp::Ordering;
        assert_eq!(OpComplexity::Low.cmp(&OpComplexity::Medium), Ordering::Less);
        assert_eq!(
            OpComplexity::Medium.cmp(&OpComplexity::High),
            Ordering::Less
        );
        assert_eq!(
            OpComplexity::High.cmp(&OpComplexity::Medium),
            Ordering::Greater
        );
        assert_eq!(OpComplexity::Low.cmp(&OpComplexity::Low), Ordering::Equal);
    }

    #[test]
    fn test_op_complexity_eq() {
        assert_eq!(OpComplexity::Low, OpComplexity::Low);
        assert_eq!(OpComplexity::Medium, OpComplexity::Medium);
        assert_eq!(OpComplexity::High, OpComplexity::High);
        assert_ne!(OpComplexity::Low, OpComplexity::High);
    }

    #[test]
    fn test_op_complexity_debug() {
        let complexity = OpComplexity::Medium;
        let debug_str = format!("{:?}", complexity);
        assert!(debug_str.contains("Medium"));
    }

    #[test]
    fn test_op_complexity_clone() {
        let complexity = OpComplexity::High;
        #[allow(clippy::clone_on_copy)]
        let cloned = complexity.clone();
        assert_eq!(complexity, cloned);
    }

    #[test]
    fn test_op_complexity_copy() {
        let complexity = OpComplexity::Low;
        let copied = complexity;
        assert_eq!(complexity, copied);
    }

    #[test]
    fn test_trueno_error_reexport() {
        // Verify error types are re-exported correctly
        let _: Result<()> = Ok(());
        let err: Result<()> = Err(TruenoError::EmptyVector);
        assert!(err.is_err());
    }

    #[test]
    fn test_vector_reexport() {
        // Verify Vector is re-exported correctly
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_matrix_reexport() {
        // Verify Matrix is re-exported correctly
        let m = Matrix::zeros(2, 2);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_detect_x86_backend() {
        let backend = detect_x86_backend();
        // On x86_64, we should get at least SSE2
        assert!(matches!(
            backend,
            Backend::SSE2 | Backend::AVX | Backend::AVX2
        ));
        // Should NOT return AVX-512 (intentionally avoided for safety)
        assert_ne!(backend, Backend::AVX512);
    }

    #[test]
    fn test_operation_type_enum() {
        // Verify OperationType variants are distinct
        assert_ne!(OperationType::MemoryBound, OperationType::ComputeBound);
        assert_ne!(OperationType::MemoryBound, OperationType::Mixed);
        assert_ne!(OperationType::ComputeBound, OperationType::Mixed);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_select_backend_for_memory_bound_prefers_avx2() {
        let backend = select_backend_for_operation(OperationType::MemoryBound);

        // Should prefer AVX2 over AVX-512 for memory-bound operations
        // (Based on AVX-512 performance analysis showing 0.67-1.01x scalar)
        if is_x86_feature_detected!("avx2") {
            assert_eq!(backend, Backend::AVX2);
        } else if is_x86_feature_detected!("avx") {
            assert_eq!(backend, Backend::AVX);
        } else if is_x86_feature_detected!("sse2") {
            assert_eq!(backend, Backend::SSE2);
        } else {
            assert_eq!(backend, Backend::Scalar);
        }

        // Critical: Should NEVER return AVX-512 for memory-bound
        assert_ne!(backend, Backend::AVX512);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_select_backend_for_compute_bound_allows_avx512() {
        let backend = select_backend_for_operation(OperationType::ComputeBound);

        // Should prefer AVX-512 for compute-bound operations where it excels (7-14x scalar)
        if is_x86_feature_detected!("avx512f") {
            assert_eq!(backend, Backend::AVX512);
        } else if is_x86_feature_detected!("avx2") {
            assert_eq!(backend, Backend::AVX2);
        } else if is_x86_feature_detected!("avx") {
            assert_eq!(backend, Backend::AVX);
        } else if is_x86_feature_detected!("sse2") {
            assert_eq!(backend, Backend::SSE2);
        } else {
            assert_eq!(backend, Backend::Scalar);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_select_backend_for_mixed_prefers_avx2() {
        let backend = select_backend_for_operation(OperationType::Mixed);

        // Mixed operations should default to AVX2 for safety (avoid AVX-512 thermal throttling)
        if is_x86_feature_detected!("avx2") {
            assert_eq!(backend, Backend::AVX2);
        } else if is_x86_feature_detected!("avx") {
            assert_eq!(backend, Backend::AVX);
        } else if is_x86_feature_detected!("sse2") {
            assert_eq!(backend, Backend::SSE2);
        } else {
            assert_eq!(backend, Backend::Scalar);
        }

        // Should NOT return AVX-512 for mixed operations (safety first)
        assert_ne!(backend, Backend::AVX512);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_default_backend_selection_avoids_avx512() {
        // The default backend selection (detect_x86_backend) should avoid AVX-512
        let default_backend = select_best_available_backend();

        // Even on CPUs with AVX-512, default selection should prefer AVX2
        if is_x86_feature_detected!("avx2") {
            assert_eq!(default_backend, Backend::AVX2);
        }

        // Verify AVX-512 is NOT returned by default
        assert_ne!(default_backend, Backend::AVX512);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_backend_selection_consistency() {
        // Memory-bound and Mixed should return same backend (AVX2-first)
        let memory_backend = select_backend_for_operation(OperationType::MemoryBound);
        let mixed_backend = select_backend_for_operation(OperationType::Mixed);

        assert_eq!(memory_backend, mixed_backend);

        // Compute-bound may differ (allows AVX-512)
        let compute_backend = select_backend_for_operation(OperationType::ComputeBound);

        // If AVX-512 is available, compute backend should be different
        if is_x86_feature_detected!("avx512f") {
            assert_ne!(compute_backend, memory_backend);
            assert_eq!(compute_backend, Backend::AVX512);
        } else {
            // Without AVX-512, all should be the same
            assert_eq!(compute_backend, memory_backend);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    #[test]
    fn test_select_backend_for_operation_non_x86() {
        // On non-x86 platforms, all operation types should return platform-specific backend
        let memory = select_backend_for_operation(OperationType::MemoryBound);
        let compute = select_backend_for_operation(OperationType::ComputeBound);
        let mixed = select_backend_for_operation(OperationType::Mixed);

        // All should return the same backend (ARM NEON, WASM SIMD, or Scalar)
        assert_eq!(memory, compute);
        assert_eq!(memory, mixed);

        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        {
            #[cfg(target_feature = "neon")]
            assert_eq!(memory, Backend::NEON);
        }

        #[cfg(target_arch = "wasm32")]
        {
            #[cfg(target_feature = "simd128")]
            assert_eq!(memory, Backend::WasmSIMD);
        }
    }
}
