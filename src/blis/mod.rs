//! BLIS-Style Matrix Multiplication
//!
//! High-performance GEMM implementation based on the BLIS framework.
//!
//! # References
//!
//! - Goto, K., & Van de Geijn, R. A. (2008). Anatomy of High-Performance Matrix Multiplication.
//!   ACM TOMS, 34(3). <https://doi.org/10.1145/1356052.1356053>
//! - Van Zee, F. G., & Van de Geijn, R. A. (2015). BLIS: A Framework for Rapidly Instantiating
//!   BLAS Functionality. ACM TOMS, 41(3). <https://doi.org/10.1145/2764454>
//! - Low, T. M., et al. (2016). Analytical Modeling Is Enough for High-Performance BLIS.
//!   ACM TOMS, 43(2). <https://doi.org/10.1145/2925987>
//!
//! # Toyota Production System Integration
//!
//! - **Jidoka**: Runtime guards that stop on numerical errors (see [`jidoka`] module)
//! - **Poka-Yoke**: Compile-time type safety for panel dimensions
//! - **Heijunka**: Load-balanced parallel execution
//! - **Kaizen**: Performance tracking for continuous improvement (see [`profiler`] module)
//!
//! # Module Structure
//!
//! - [`jidoka`]: Runtime validation guards (stop-on-defect)
//! - [`profiler`]: Performance tracking at all BLIS hierarchy levels
//! - [`microkernels`]: High-performance SIMD compute kernels
//! - [`backend_selection`]: Automatic CPU/GPU backend selection
//! - [`reference`]: Scalar reference GEMM for validation
//! - [`packing`]: Cache-optimized matrix packing routines
//! - [`compute`]: Core BLIS blocked GEMM computation
//! - [`parallel`]: Parallel GEMM with Heijunka scheduling
//! - [`transpose`]: Matrix transpose operations

pub mod backend_selection;
pub mod compute;
pub mod elementwise;
pub mod gemv;
pub mod jidoka;
pub mod microkernels;
pub mod norms;
pub mod packing;
pub mod parallel;
pub mod prepacked;
pub mod profiler;
pub mod reference;
pub mod softmax;
pub mod transpose;

// Re-export jidoka types for backwards compatibility
pub use jidoka::{JidokaError, JidokaGuard};

// Re-export profiler types for backwards compatibility
pub use profiler::{BlisLevelStats, BlisProfileLevel, BlisProfiler, KaizenMetrics};

// Re-export microkernel functions
#[cfg(target_arch = "aarch64")]
pub use microkernels::microkernel_8x8_neon;
pub use microkernels::microkernel_scalar;
#[cfg(target_arch = "x86_64")]
pub use microkernels::{microkernel_8x6_avx2, microkernel_8x6_avx2_asm, microkernel_8x6_true_asm};

// Re-export backend selection types
pub use backend_selection::{
    gemm_auto, BackendCostModel, BrickLevel, ComputeBackend, PtxMicrokernelSpec, RooflineResult,
    UnifiedBrickProfiler, WgslMicrokernelSpec,
};

// Re-export reference GEMM
pub use reference::{gemm_reference, gemm_reference_with_jidoka};

// Re-export packing functions
pub use packing::{pack_a, pack_b, packed_a_size, packed_b_size};

// Re-export compute
pub use compute::{gemm_blis, gemm_blis_with_prepacked_b};

// Re-export parallel
pub use parallel::{gemm_blis_parallel, gemm_blis_parallel_with_prepacked_b, HeijunkaScheduler};

// Re-export prepacked
pub use prepacked::PrepackedB;

// Re-export transpose
pub use transpose::transpose;

use crate::error::TruenoError;

// ============================================================================
// BLIS Configuration Constants
// ============================================================================

/// Microkernel row dimension (AVX2: 8 f32 per ymm register)
pub const MR: usize = 8;

/// Microkernel column dimension (6 columns fit in remaining registers)
pub const NR: usize = 6;

/// K-dimension blocking (packed_a panel = MR*KC*4 = 16KB fits L1)
pub const KC: usize = 512;

/// M-dimension blocking (packed_a = MC*KC*4 = 256KB fits L2)
pub const MC: usize = 128;

/// N-dimension blocking (packed_b = KC*NC*4 = 1.4MB, Zen 4 L2 tuned)
pub const NC: usize = 720;

// ============================================================================
// Public API
// ============================================================================

/// High-performance GEMM using BLIS algorithm
///
/// Computes C += A * B where:
/// - A is M x K (row-major)
/// - B is K x N (row-major)
/// - C is M x N (row-major)
///
/// Automatically selects single-threaded or parallel execution based on matrix size.
pub fn gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    // Contract: matmul-kernel-v1.yaml precondition (pv codegen)
    contract_pre_matmul!(a);

    let result = {
        #[cfg(feature = "parallel")]
        {
            gemm_blis_parallel(m, n, k, a, b, c)
        }
        #[cfg(not(feature = "parallel"))]
        {
            gemm_blis(m, n, k, a, b, c, None)
        }
    };
    if result.is_ok() {
        contract_post_matmul!(c);
    }
    result
}

/// GEMM with profiling enabled
pub fn gemm_profiled(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    profiler: &mut BlisProfiler,
) -> Result<(), TruenoError> {
    gemm_blis(m, n, k, a, b, c, Some(profiler))
}

#[cfg(test)]
mod tests;
