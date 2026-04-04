//! Parallel GEMM with Heijunka (load-leveling) scheduling.
//!
//! Uses Rayon for parallel execution when the `parallel` feature is enabled,
//! with balanced M-dimension partitioning via [`HeijunkaScheduler`].

use crate::error::TruenoError;

use super::compute::{gemm_blis, gemm_blis_with_prepacked_b};
use super::prepacked::PrepackedB;
#[cfg(feature = "parallel")]
use super::{MC, MR};

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

    // Single-threaded threshold: 8M FLOPs ≈ 200³.
    // Rayon dispatch costs ~3µs. For GEMM ≤128 (~4M FLOP, ~35µs compute),
    // rayon overhead dominates. GEMM 256+ (33M FLOP, ~300µs) benefits.
    let flops = m * n * k;
    if flops < 8_000_000 {
        return gemm_blis(m, n, k, a, b, c, None);
    }

    // Scale thread count to problem size and cache topology.
    // cgp profile scaling measurements (2026-04-04, Threadripper 7960X 24C/48T):
    //
    //   256x256: 1T=28.6, 2T=35.9 (peak), 4T=35.2, 8T=34.4 → cap at 2
    //   512x512: 1T=82.3, 4T=161.1 (peak), 8T=148.3 → cap at 4
    //   1024x1024: 1T=105, 8T=485, 16T=541 (peak), 24T=552 → use all cores
    //
    // Root cause for small-problem regression: L3 contention and thread spawn
    // overhead (~40µs per thread::scope) dominate when compute < 1ms.
    let phys_cores = num_cpus::get_physical();
    let max_threads = if flops < 64_000_000 {
        // 256³ and below: barely benefits from parallelism
        2.min(phys_cores)
    } else if flops < 512_000_000 {
        // 512³ range: 4T is peak, >4 regresses due to L3 contention
        4.min(phys_cores)
    } else {
        // Very large (1024³+): use all cores, working set exceeds single CCD L3
        phys_cores
    };

    let mut scheduler = HeijunkaScheduler::default();
    scheduler.num_threads = scheduler.num_threads.min(max_threads);
    let ps = if m <= MC { MR.max(m / scheduler.num_threads) } else { MC };
    let partitions = scheduler.partition_m(m, ps);

    // Each thread packs B independently via gemm_blis.
    // NOTE: Pre-packing B and using gemm_blis_with_prepacked_b was tested
    // (2026-04-04 via cgp profiling) but regressed performance from 548→256
    // GFLOPS at 1024x1024x8T. The unpacked inner loop in gemm_blis is more
    // optimized (uses the ASM microkernel path more effectively). The B packing
    // cost per thread is amortized across K iterations.

    let c_ptr = c.as_mut_ptr() as usize;

    partitions.into_par_iter().for_each(|m_range| {
        let m_local = m_range.len();
        let m_start = m_range.start;

        let a_local = &a[m_start * k..(m_start + m_local) * k];

        // SAFETY: Each thread accesses a disjoint row range of C.
        // Partitions are non-overlapping by construction in HeijunkaScheduler::partition_m.
        let c_local = unsafe {
            let ptr = c_ptr as *mut f32;
            std::slice::from_raw_parts_mut(ptr.add(m_start * n), m_local * n)
        };

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

/// Parallel BLIS GEMM with pre-packed B matrix.
///
/// Key optimization: the pre-packed B is shared immutably across all threads.
/// Each thread only packs A (which differs per M partition). This eliminates
/// N_threads × redundant B packings per GEMM call.
///
/// # WAPR-KAIZEN Cycle 12
///
/// For 16-thread encoder FFN: eliminates 15 redundant B packings per GEMM call
/// (128 total across 2 GEMMs × 4 layers).
#[cfg(feature = "parallel")]
pub fn gemm_blis_parallel_with_prepacked_b(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    prepacked_b: &PrepackedB,
    c: &mut [f32],
) -> Result<(), TruenoError> {
    use rayon::prelude::*;

    if a.len() != m * k || c.len() != m * n {
        return Err(TruenoError::InvalidInput("Dimension mismatch".to_string()));
    }
    if prepacked_b.k != k || prepacked_b.n != n {
        return Err(TruenoError::InvalidInput(format!(
            "PrepackedB dimension mismatch: expected ({}, {}), got ({}, {})",
            k, n, prepacked_b.k, prepacked_b.n
        )));
    }

    // Small matrices: single-threaded
    if m * n * k < 1_000_000 {
        return gemm_blis_with_prepacked_b(m, n, k, a, prepacked_b, c, None);
    }

    let scheduler = HeijunkaScheduler::default();
    let partitions = scheduler.partition_m(m, MC);

    let c_ptr = c.as_mut_ptr() as usize;

    // Key: prepacked_b is shared (immutable &) across all threads — zero redundant packing
    partitions.into_par_iter().for_each(|m_range| {
        let m_local = m_range.len();
        let m_start = m_range.start;

        let a_local = &a[m_start * k..(m_start + m_local) * k];

        // SAFETY: Each thread accesses a disjoint row range of C.
        // Partitions are non-overlapping by construction in HeijunkaScheduler::partition_m.
        let c_local = unsafe {
            let ptr = c_ptr as *mut f32;
            std::slice::from_raw_parts_mut(ptr.add(m_start * n), m_local * n)
        };

        let _ = gemm_blis_with_prepacked_b(m_local, n, k, a_local, prepacked_b, c_local, None);
    });

    Ok(())
}

/// Non-parallel fallback for pre-packed B
#[cfg(not(feature = "parallel"))]
pub fn gemm_blis_parallel_with_prepacked_b(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    prepacked_b: &PrepackedB,
    c: &mut [f32],
) -> Result<(), TruenoError> {
    gemm_blis_with_prepacked_b(m, n, k, a, prepacked_b, c, None)
}
