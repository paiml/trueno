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

pub mod backend_selection;
pub mod jidoka;
pub mod microkernels;
pub mod profiler;

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

use std::time::Instant;

use crate::error::TruenoError;

// ============================================================================
// BLIS Configuration Constants
// ============================================================================

/// Microkernel row dimension (AVX2: 8 f32 per ymm register)
pub const MR: usize = 8;

/// Microkernel column dimension (6 columns fit in remaining registers)
pub const NR: usize = 6;

/// K-dimension blocking for L1 cache (256 elements = 1KB)
pub const KC: usize = 256;

/// M-dimension blocking for L2 cache
pub const MC: usize = 72;

/// N-dimension blocking for L3 cache
pub const NC: usize = 4096;

// ============================================================================
// Phase 1: Scalar Reference Implementation
// ============================================================================

/// Scalar reference GEMM for Jidoka validation
///
/// Computes C += A * B where:
/// - A is M x K (row-major)
/// - B is K x N (row-major)
/// - C is M x N (row-major)
///
/// This is the "gold standard" implementation used to validate optimized versions.
///
/// # References
///
/// This implements the naive O(MNK) algorithm as described in
/// Golub & Van Loan (2013), Matrix Computations, 4th ed., Algorithm 1.1.1.
pub fn gemm_reference(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    // Poka-yoke: dimension validation
    if a.len() != m * k {
        return Err(TruenoError::InvalidInput(format!(
            "A size mismatch: expected {}x{}={}, got {}",
            m,
            k,
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(TruenoError::InvalidInput(format!(
            "B size mismatch: expected {}x{}={}, got {}",
            k,
            n,
            k * n,
            b.len()
        )));
    }
    if c.len() != m * n {
        return Err(TruenoError::InvalidInput(format!(
            "C size mismatch: expected {}x{}={}, got {}",
            m,
            n,
            m * n,
            c.len()
        )));
    }

    // Scalar triple-nested loop
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] += sum;
        }
    }

    Ok(())
}

/// Scalar reference GEMM with Jidoka validation
///
/// Same as `gemm_reference` but validates outputs against known-good computation.
/// Validate a sampled output value for NaN/Inf (Jidoka andon cord).
#[inline(always)]
fn jidoka_check_output(val: f32, idx: usize, sample_rate: usize) -> Result<(), JidokaError> {
    if idx % sample_rate == 0 {
        if val.is_nan() {
            return Err(JidokaError::NaNDetected { location: "output" });
        }
        if val.is_infinite() {
            return Err(JidokaError::InfDetected { location: "output" });
        }
    }
    Ok(())
}

/// Validate sampled input values for NaN/Inf.
fn jidoka_check_inputs(a: &[f32], b: &[f32], guard: &JidokaGuard) -> Result<(), JidokaError> {
    for (idx, &val) in a.iter().enumerate() {
        if idx % guard.sample_rate == 0 {
            guard.check_input(val, "matrix A")?;
        }
    }
    for (idx, &val) in b.iter().enumerate() {
        if idx % guard.sample_rate == 0 {
            guard.check_input(val, "matrix B")?;
        }
    }
    Ok(())
}

pub fn gemm_reference_with_jidoka(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    guard: &JidokaGuard,
) -> Result<(), JidokaError> {
    jidoka_check_inputs(a, b, guard)?;

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            let output = c[i * n + j] + sum;
            jidoka_check_output(output, i * n + j, guard.sample_rate)?;
            c[i * n + j] = output;
        }
    }

    Ok(())
}

// ============================================================================
// Phase 2: Microkernel (MR=8, NR=6)
// ============================================================================

// Phase 3: Cache-Optimized Packing
// ============================================================================

/// Pack A into MC x KC panel with MR-aligned micro-panels
///
/// Memory layout (Van Zee & Van de Geijn, 2015, Fig. 4):
/// Original A (row-major):     Packed A (column-major micro-panels):
/// [a00 a01 a02 ...]           [a00 a10 a20 ... a(MR-1)0 | a01 a11 ...]
/// [a10 a11 a12 ...]            \____ MR elements ____/
///
/// This layout ensures:
/// 1. Sequential access in the microkernel
/// 2. Optimal cache line utilization
/// 3. Aligned loads for SIMD
pub fn pack_a(
    a: &[f32],
    lda: usize, // Leading dimension of A (number of columns in original)
    mc: usize,  // Number of rows to pack
    kc: usize,  // Number of columns to pack
    packed: &mut [f32],
) {
    let mut pack_idx = 0;

    // Process MR rows at a time
    let full_panels = mc / MR;
    let remainder = mc % MR;

    for panel in 0..full_panels {
        let row_start = panel * MR;

        for col in 0..kc {
            for row in 0..MR {
                packed[pack_idx] = a[(row_start + row) * lda + col];
                pack_idx += 1;
            }
        }
    }

    // Handle remainder rows (pad with zeros)
    if remainder > 0 {
        let row_start = full_panels * MR;

        for col in 0..kc {
            for row in 0..MR {
                if row < remainder {
                    packed[pack_idx] = a[(row_start + row) * lda + col];
                } else {
                    packed[pack_idx] = 0.0; // Zero padding
                }
                pack_idx += 1;
            }
        }
    }
}

/// Pack B into KC x NC panel with NR-aligned micro-panels
///
/// Memory layout:
/// Original B (row-major):     Packed B (row-major micro-panels):
/// [b00 b01 b02 ...]           [b00 b01 ... b(NR-1) | b10 b11 ...]
/// [b10 b11 b12 ...]            \____ NR elements ____/
pub fn pack_b(
    b: &[f32],
    ldb: usize, // Leading dimension of B (number of columns in original)
    kc: usize,  // Number of rows to pack
    nc: usize,  // Number of columns to pack
    packed: &mut [f32],
) {
    let mut pack_idx = 0;

    let full_panels = nc / NR;
    let remainder = nc % NR;

    for panel in 0..full_panels {
        let col_start = panel * NR;

        for row in 0..kc {
            for col in 0..NR {
                packed[pack_idx] = b[row * ldb + col_start + col];
                pack_idx += 1;
            }
        }
    }

    // Handle remainder columns (pad with zeros)
    if remainder > 0 {
        let col_start = full_panels * NR;

        for row in 0..kc {
            for col in 0..NR {
                if col < remainder {
                    packed[pack_idx] = b[row * ldb + col_start + col];
                } else {
                    packed[pack_idx] = 0.0;
                }
                pack_idx += 1;
            }
        }
    }
}

/// Compute required packed A buffer size
#[inline]
pub fn packed_a_size(mc: usize, kc: usize) -> usize {
    let panels = (mc + MR - 1) / MR;
    panels * MR * kc
}

/// Compute required packed B buffer size
#[inline]
pub fn packed_b_size(kc: usize, nc: usize) -> usize {
    let panels = (nc + NR - 1) / NR;
    panels * NR * kc
}

// ============================================================================
// Phase 4: Cache-Blocked GEMM
// ============================================================================

/// Load a tile of C into the micro workspace for accumulation.
#[inline(always)]
fn load_c_tile(c: &[f32], c_micro: &mut [f32], row: usize, col: usize, mr: usize, nr: usize, n: usize) {
    c_micro.fill(0.0);
    for jj in 0..nr {
        for ii in 0..mr {
            c_micro[jj * MR + ii] = c[(row + ii) * n + (col + jj)];
        }
    }
}

/// Store a micro tile back into C.
#[inline(always)]
fn store_c_tile(c: &mut [f32], c_micro: &[f32], row: usize, col: usize, mr: usize, nr: usize, n: usize) {
    for jj in 0..nr {
        for ii in 0..mr {
            c[(row + ii) * n + (col + jj)] = c_micro[jj * MR + ii];
        }
    }
}

/// Dispatch to the best available microkernel (AVX2 ASM or scalar fallback).
#[inline(always)]
fn dispatch_microkernel(kc: usize, a_panel: &[f32], b_panel: &[f32], c_micro: &mut [f32], mr_block: usize, nr_block: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && mr_block == MR
            && nr_block == NR
        {
            // SAFETY: AVX2+FMA verified by is_x86_feature_detected!() above.
            unsafe {
                microkernel_8x6_true_asm(
                    kc,
                    a_panel.as_ptr(),
                    b_panel.as_ptr(),
                    c_micro.as_mut_ptr(),
                    MR,
                );
            }
            return;
        }
    }
    microkernel_scalar(kc, a_panel, b_panel, c_micro, MR);
}

/// Execute microkernel tile iterations over one MC x NC x KC macro-block.
#[allow(clippy::too_many_arguments)]
fn compute_macroblock(
    c: &mut [f32],
    packed_a: &[f32],
    packed_b: &[f32],
    c_micro: &mut [f32],
    ic: usize,
    jc: usize,
    mc_block: usize,
    nc_block: usize,
    kc_block: usize,
    n: usize,
    profiler: &mut Option<&mut BlisProfiler>,
) {
    let midi_start = Instant::now();

    for jr in (0..nc_block).step_by(NR) {
        let nr_block = NR.min(nc_block - jr);
        for ir in (0..mc_block).step_by(MR) {
            let mr_block = MR.min(mc_block - ir);
            let micro_start = Instant::now();

            let a_panel = &packed_a[(ir / MR) * MR * kc_block..];
            let b_panel = &packed_b[(jr / NR) * NR * kc_block..];

            load_c_tile(c, c_micro, ic + ir, jc + jr, mr_block, nr_block, n);
            dispatch_microkernel(kc_block, a_panel, b_panel, c_micro, mr_block, nr_block);
            store_c_tile(c, c_micro, ic + ir, jc + jr, mr_block, nr_block, n);

            if let Some(ref mut prof) = profiler.as_deref_mut() {
                prof.record(
                    BlisProfileLevel::Micro,
                    micro_start.elapsed().as_nanos() as u64,
                    (2 * mr_block * nr_block * kc_block) as u64,
                );
            }
        }
    }

    if let Some(ref mut prof) = profiler.as_deref_mut() {
        prof.record(
            BlisProfileLevel::Midi,
            midi_start.elapsed().as_nanos() as u64,
            (2 * mc_block * nc_block * kc_block) as u64,
        );
    }
}

/// BLIS-style blocked GEMM
///
/// Implements the 5-loop BLIS algorithm (Van Zee & Van de Geijn, 2015):
/// Loop 5 (jc): N dimension, L3 blocking
/// Loop 4 (pc): K dimension, L2 blocking
/// Loop 3 (ic): M dimension, L1 blocking
/// Loop 2 (jr): Microkernel columns
/// Loop 1 (ir): Microkernel rows
/// Validate GEMM dimension inputs (Poka-yoke).
fn validate_gemm_dims(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &[f32]) -> Result<(), TruenoError> {
    if a.len() != m * k {
        return Err(TruenoError::InvalidInput(format!("A size mismatch: expected {}, got {}", m * k, a.len())));
    }
    if b.len() != k * n {
        return Err(TruenoError::InvalidInput(format!("B size mismatch: expected {}, got {}", k * n, b.len())));
    }
    if c.len() != m * n {
        return Err(TruenoError::InvalidInput(format!("C size mismatch: expected {}, got {}", m * n, c.len())));
    }
    Ok(())
}

/// Record a profiler event if profiler is active.
#[inline(always)]
fn record_prof(profiler: &mut Option<&mut BlisProfiler>, level: BlisProfileLevel, start: Instant, flops: u64) {
    if let Some(ref mut prof) = profiler.as_deref_mut() {
        prof.record(level, start.elapsed().as_nanos() as u64, flops);
    }
}

pub fn gemm_blis(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    mut profiler: Option<&mut BlisProfiler>,
) -> Result<(), TruenoError> {
    validate_gemm_dims(m, n, k, a, b, c)?;

    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }
    if m * n * k < 4096 {
        return gemm_reference(m, n, k, a, b, c);
    }

    let start = Instant::now();

    let mc = MC.min(m);
    let nc = NC.min(n);
    let kc = KC.min(k);

    let mut packed_a = vec![0.0f32; packed_a_size(mc, kc)];
    let mut packed_b = vec![0.0f32; packed_b_size(kc, nc)];
    let mut c_micro = vec![0.0f32; MR * NR];

    for jc in (0..n).step_by(NC) {
        let nc_block = NC.min(n - jc);

        for pc in (0..k).step_by(KC) {
            let kc_block = KC.min(k - pc);

            let pack_start = Instant::now();
            pack_b_block(b, n, pc, jc, kc_block, nc_block, &mut packed_b);
            record_prof(&mut profiler, BlisProfileLevel::Pack, pack_start, 0);

            for ic in (0..m).step_by(MC) {
                let mc_block = MC.min(m - ic);

                let pack_start = Instant::now();
                pack_a_block(a, k, ic, pc, mc_block, kc_block, &mut packed_a);
                record_prof(&mut profiler, BlisProfileLevel::Pack, pack_start, 0);

                compute_macroblock(
                    c, &packed_a, &packed_b, &mut c_micro,
                    ic, jc, mc_block, nc_block, kc_block, n,
                    &mut profiler,
                );
            }
        }
    }

    if let Some(prof) = profiler {
        prof.record(BlisProfileLevel::Macro, start.elapsed().as_nanos() as u64, (2 * m * n * k) as u64);
    }

    Ok(())
}

/// Pack A block from row-major source
fn pack_a_block(
    a: &[f32],
    lda: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f32],
) {
    let mut pack_idx = 0;
    let panels = (rows + MR - 1) / MR;

    for panel in 0..panels {
        let ir = panel * MR;
        let mr_actual = MR.min(rows - ir);

        for col in 0..cols {
            for row in 0..MR {
                if row < mr_actual {
                    packed[pack_idx] = a[(row_start + ir + row) * lda + col_start + col];
                } else {
                    packed[pack_idx] = 0.0;
                }
                pack_idx += 1;
            }
        }
    }
}

/// Pack B block from row-major source
fn pack_b_block(
    b: &[f32],
    ldb: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f32],
) {
    let mut pack_idx = 0;
    let panels = (cols + NR - 1) / NR;

    for panel in 0..panels {
        let jr = panel * NR;
        let nr_actual = NR.min(cols - jr);

        for row in 0..rows {
            for col in 0..NR {
                if col < nr_actual {
                    packed[pack_idx] = b[(row_start + row) * ldb + col_start + jr + col];
                } else {
                    packed[pack_idx] = 0.0;
                }
                pack_idx += 1;
            }
        }
    }
}

// ============================================================================
// Phase 5: Parallel GEMM with Heijunka
// ============================================================================

/// Heijunka (load-leveling) scheduler for parallel GEMM
#[derive(Debug, Clone)]
pub struct HeijunkaScheduler {
    /// Number of threads
    pub num_threads: usize,
    /// Target load variance threshold
    pub variance_threshold: f32,
}

impl Default for HeijunkaScheduler {
    fn default() -> Self {
        #[cfg(feature = "parallel")]
        let threads = rayon::current_num_threads();
        #[cfg(not(feature = "parallel"))]
        let threads = 1;

        Self {
            num_threads: threads,
            variance_threshold: 0.05, // 5% variance target
        }
    }
}

impl HeijunkaScheduler {
    /// Partition M dimension into balanced chunks
    pub fn partition_m(&self, m: usize, mc: usize) -> Vec<std::ops::Range<usize>> {
        let num_blocks = (m + mc - 1) / mc;
        let blocks_per_thread = num_blocks / self.num_threads;
        let remainder = num_blocks % self.num_threads;

        let mut partitions = Vec::with_capacity(self.num_threads);
        let mut start_block = 0;

        for t in 0..self.num_threads {
            let extra = if t < remainder { 1 } else { 0 };
            let thread_blocks = blocks_per_thread + extra;

            let start_row = start_block * mc;
            let end_row = ((start_block + thread_blocks) * mc).min(m);

            if start_row < end_row {
                partitions.push(start_row..end_row);
            }

            start_block += thread_blocks;
        }

        partitions
    }
}

/// Parallel BLIS GEMM using Rayon
#[cfg(feature = "parallel")]
pub fn gemm_blis_parallel(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    use rayon::prelude::*;

    // Dimension validation
    if a.len() != m * k || b.len() != k * n || c.len() != m * n {
        return Err(TruenoError::InvalidInput("Dimension mismatch".to_string()));
    }

    // Small matrices: single-threaded
    if m * n * k < 1_000_000 {
        return gemm_blis(m, n, k, a, b, c, None);
    }

    let scheduler = HeijunkaScheduler::default();
    let partitions = scheduler.partition_m(m, MC);

    // Pack B once (shared across threads)
    let nc = NC.min(n);
    let kc = KC.min(k);
    let packed_b_total_size = ((n + NR - 1) / NR) * ((k + KC - 1) / KC) * packed_b_size(kc, nc);
    let _packed_b = std::sync::Arc::new(std::sync::RwLock::new(vec![0.0f32; packed_b_total_size]));

    // Parallel over M partitions
    let c_ptr = c.as_mut_ptr() as usize;
    let _c_len = c.len();

    partitions.into_par_iter().for_each(|m_range| {
        let m_local = m_range.len();
        let m_start = m_range.start;

        // Local A slice
        let a_local = &a[m_start * k..(m_start + m_local) * k];

        // Local C slice (unsafe but safe due to non-overlapping partitions)
        let c_local = unsafe {
            let ptr = c_ptr as *mut f32;
            std::slice::from_raw_parts_mut(ptr.add(m_start * n), m_local * n)
        };

        // Run local GEMM
        let _ = gemm_blis(m_local, n, k, a_local, b, c_local, None);
    });

    Ok(())
}

/// Non-parallel fallback
#[cfg(not(feature = "parallel"))]
pub fn gemm_blis_parallel(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    gemm_blis(m, n, k, a, b, c, None)
}

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
    #[cfg(feature = "parallel")]
    {
        gemm_blis_parallel(m, n, k, a, b, c)
    }
    #[cfg(not(feature = "parallel"))]
    {
        gemm_blis(m, n, k, a, b, c, None)
    }
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

// ============================================================================
// Matrix Transpose (SIMD-optimized)
// ============================================================================

/// Transpose a matrix: B = A^T
///
/// SIMD-optimized for large matrices (>=64 elements).
/// Uses cache-efficient 8x8 blocking with manual unrolling.
///
/// # Arguments
///
/// * `rows` - Number of rows in A (cols in B)
/// * `cols` - Number of cols in A (rows in B)
/// * `a` - Input matrix A (rows x cols, row-major)
/// * `b` - Output matrix B (cols x rows, row-major)
///
/// # Returns
///
/// `Ok(())` on success, `Err` if dimensions mismatch
/// Scalar transpose of a sub-region of a row-major matrix.
#[inline(always)]
fn transpose_region(a: &[f32], b: &mut [f32], rows: std::ops::Range<usize>, cols: std::ops::Range<usize>, src_cols: usize, dst_rows: usize) {
    for r in rows {
        for c in cols.clone() {
            b[c * dst_rows + r] = a[r * src_cols + c];
        }
    }
}

pub fn transpose(rows: usize, cols: usize, a: &[f32], b: &mut [f32]) -> Result<(), TruenoError> {
    let expected = rows * cols;
    if a.len() != expected || b.len() != expected {
        return Err(TruenoError::InvalidInput(format!(
            "transpose size mismatch: a[{}], b[{}], expected {}",
            a.len(), b.len(), expected
        )));
    }

    if expected < 64 {
        transpose_region(a, b, 0..rows, 0..cols, cols, rows);
        return Ok(());
    }

    const BLOCK: usize = 8;
    let row_blocks = rows / BLOCK;
    let col_blocks = cols / BLOCK;

    for rb in 0..row_blocks {
        for cb in 0..col_blocks {
            let rs = rb * BLOCK;
            let cs = cb * BLOCK;
            transpose_region(a, b, rs..rs + BLOCK, cs..cs + BLOCK, cols, rows);
        }
    }

    // Right edge remainder
    let col_rem = col_blocks * BLOCK;
    if col_rem < cols {
        transpose_region(a, b, 0..row_blocks * BLOCK, col_rem..cols, cols, rows);
    }

    // Bottom edge remainder
    let row_rem = row_blocks * BLOCK;
    if row_rem < rows {
        transpose_region(a, b, row_rem..rows, 0..cols, cols, rows);
    }

    Ok(())
}

#[cfg(test)]
mod tests;
