//! Core BLIS compute routines: microkernel dispatch, macroblock execution,
//! and the cache-blocked GEMM main loop.
//!
//! Implements the 5-loop BLIS algorithm (Van Zee & Van de Geijn, 2015):
//! - Loop 5 (jc): N dimension, L3 blocking
//! - Loop 4 (pc): K dimension, L2 blocking
//! - Loop 3 (ic): M dimension, L1 blocking
//! - Loop 2 (jr): Microkernel columns
//! - Loop 1 (ir): Microkernel rows

use std::cell::RefCell;
use std::time::Instant;

use crate::error::TruenoError;

#[cfg(target_arch = "x86_64")]
use super::microkernels::microkernel_8x6_true_asm;
use super::microkernels::microkernel_scalar;
use super::packing::{pack_a_block, pack_b_block, packed_a_size, packed_b_size};
use super::prepacked::PrepackedB;
use super::profiler::{BlisProfileLevel, BlisProfiler};
use super::reference::gemm_reference;
use super::{KC, MC, MR, NC, NR};

// Thread-local workspace buffers to eliminate allocation churn in gemm_blis.
// These grow to the high-water mark and are reused across calls, avoiding
// ~4.3 MB of allocation+deallocation per GEMM invocation.
thread_local! {
    static TL_PACKED_A: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_PACKED_B: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static TL_C_MICRO: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Load a tile of C into the micro workspace for accumulation.
#[inline(always)]
fn load_c_tile(
    c: &[f32],
    c_micro: &mut [f32],
    row: usize,
    col: usize,
    mr: usize,
    nr: usize,
    n: usize,
) {
    c_micro.fill(0.0);
    for jj in 0..nr {
        for ii in 0..mr {
            c_micro[jj * MR + ii] = c[(row + ii) * n + (col + jj)];
        }
    }
}

/// Store a micro tile back into C.
#[inline(always)]
fn store_c_tile(
    c: &mut [f32],
    c_micro: &[f32],
    row: usize,
    col: usize,
    mr: usize,
    nr: usize,
    n: usize,
) {
    for jj in 0..nr {
        for ii in 0..mr {
            c[(row + ii) * n + (col + jj)] = c_micro[jj * MR + ii];
        }
    }
}

/// Dispatch to the best available microkernel (AVX2 ASM or scalar fallback).
#[inline(always)]
fn dispatch_microkernel(
    kc: usize,
    a_panel: &[f32],
    b_panel: &[f32],
    c_micro: &mut [f32],
    mr_block: usize,
    nr_block: usize,
) {
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
    // KAIZEN-038: Avoid Instant::now() syscall (~20-40ns) when profiler is disabled.
    // For 1024x1024 GEMM, this eliminates thousands of syscalls per macroblock.
    let track_time = profiler.is_some();
    let midi_start = if track_time { Some(Instant::now()) } else { None };

    for jr in (0..nc_block).step_by(NR) {
        let nr_block = NR.min(nc_block - jr);
        for ir in (0..mc_block).step_by(MR) {
            let mr_block = MR.min(mc_block - ir);
            let micro_start = if track_time { Some(Instant::now()) } else { None };

            let a_panel = &packed_a[(ir / MR) * MR * kc_block..];
            let b_panel = &packed_b[(jr / NR) * NR * kc_block..];

            load_c_tile(c, c_micro, ic + ir, jc + jr, mr_block, nr_block, n);
            dispatch_microkernel(kc_block, a_panel, b_panel, c_micro, mr_block, nr_block);
            store_c_tile(c, c_micro, ic + ir, jc + jr, mr_block, nr_block, n);

            if let (Some(ref mut prof), Some(start)) = (profiler.as_deref_mut(), micro_start) {
                prof.record(
                    BlisProfileLevel::Micro,
                    start.elapsed().as_nanos() as u64,
                    (2 * mr_block * nr_block * kc_block) as u64,
                );
            }
        }
    }

    if let (Some(ref mut prof), Some(start)) = (profiler.as_deref_mut(), midi_start) {
        prof.record(
            BlisProfileLevel::Midi,
            start.elapsed().as_nanos() as u64,
            (2 * mc_block * nc_block * kc_block) as u64,
        );
    }
}

/// Validate GEMM dimension inputs (Poka-yoke).
fn validate_gemm_dims(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &[f32],
) -> Result<(), TruenoError> {
    if a.len() != m * k {
        return Err(TruenoError::InvalidInput(format!(
            "A size mismatch: expected {}, got {}",
            m * k,
            a.len()
        )));
    }
    if b.len() != k * n {
        return Err(TruenoError::InvalidInput(format!(
            "B size mismatch: expected {}, got {}",
            k * n,
            b.len()
        )));
    }
    if c.len() != m * n {
        return Err(TruenoError::InvalidInput(format!(
            "C size mismatch: expected {}, got {}",
            m * n,
            c.len()
        )));
    }
    Ok(())
}

/// Record a profiler event if profiler is active.
#[inline(always)]
fn record_prof(
    profiler: &mut Option<&mut BlisProfiler>,
    level: BlisProfileLevel,
    start: Option<Instant>,
    flops: u64,
) {
    if let (Some(ref mut prof), Some(s)) = (profiler.as_deref_mut(), start) {
        prof.record(level, s.elapsed().as_nanos() as u64, flops);
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

    // KAIZEN-038: Only call Instant::now() when profiler is active
    let track_time = profiler.is_some();
    let start = if track_time { Some(Instant::now()) } else { None };

    let mc = MC.min(m);
    let nc = NC.min(n);
    let kc = KC.min(k);

    let needed_a = packed_a_size(mc, kc);
    let needed_b = packed_b_size(kc, nc);
    let needed_c = MR * NR;

    // Borrow thread-local workspace buffers, growing if necessary.
    // This eliminates ~4.3 MB of allocation churn per gemm_blis call.
    TL_PACKED_A.with(|tl_a| {
        TL_PACKED_B.with(|tl_b| {
            TL_C_MICRO.with(|tl_c| {
                let mut packed_a = tl_a.borrow_mut();
                let mut packed_b = tl_b.borrow_mut();
                let mut c_micro = tl_c.borrow_mut();

                // Grow buffers to required size (high-water mark).
                // Zero-fill to match the semantics of the original vec![0.0; N].
                if packed_a.len() < needed_a {
                    packed_a.resize(needed_a, 0.0);
                } else {
                    packed_a[..needed_a].fill(0.0);
                }
                if packed_b.len() < needed_b {
                    packed_b.resize(needed_b, 0.0);
                } else {
                    packed_b[..needed_b].fill(0.0);
                }
                if c_micro.len() < needed_c {
                    c_micro.resize(needed_c, 0.0);
                } else {
                    c_micro[..needed_c].fill(0.0);
                }

                for jc in (0..n).step_by(NC) {
                    let nc_block = NC.min(n - jc);

                    for pc in (0..k).step_by(KC) {
                        let kc_block = KC.min(k - pc);

                        let pack_start = if track_time { Some(Instant::now()) } else { None };
                        pack_b_block(b, n, pc, jc, kc_block, nc_block, &mut packed_b);
                        record_prof(&mut profiler, BlisProfileLevel::Pack, pack_start, 0);

                        for ic in (0..m).step_by(MC) {
                            let mc_block = MC.min(m - ic);

                            let pack_start = if track_time { Some(Instant::now()) } else { None };
                            pack_a_block(a, k, ic, pc, mc_block, kc_block, &mut packed_a);
                            record_prof(&mut profiler, BlisProfileLevel::Pack, pack_start, 0);

                            compute_macroblock(
                                c,
                                &packed_a,
                                &packed_b,
                                &mut c_micro,
                                ic,
                                jc,
                                mc_block,
                                nc_block,
                                kc_block,
                                n,
                                &mut profiler,
                            );
                        }
                    }
                }

                if let (Some(prof), Some(s)) = (profiler, start) {
                    prof.record(
                        BlisProfileLevel::Macro,
                        s.elapsed().as_nanos() as u64,
                        (2 * m * n * k) as u64,
                    );
                }
            });
        });
    });

    Ok(())
}

/// BLIS-style blocked GEMM with pre-packed B matrix.
///
/// Identical to [`gemm_blis`] but skips B packing entirely, reading packed
/// tiles from `prepacked_b` instead. This eliminates redundant B packing
/// when the same weight matrix is reused across calls (e.g., in parallel GEMM
/// where each thread would otherwise pack B independently).
///
/// # WAPR-KAIZEN Cycle 12
///
/// For encoder FFN: 16 threads × 2 GEMMs × 4 layers = 128 B packings eliminated.
pub fn gemm_blis_with_prepacked_b(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    prepacked_b: &PrepackedB,
    c: &mut [f32],
    mut profiler: Option<&mut BlisProfiler>,
) -> Result<(), TruenoError> {
    if a.len() != m * k {
        return Err(TruenoError::InvalidInput(format!(
            "A size mismatch: expected {}, got {}",
            m * k,
            a.len()
        )));
    }
    if c.len() != m * n {
        return Err(TruenoError::InvalidInput(format!(
            "C size mismatch: expected {}, got {}",
            m * n,
            c.len()
        )));
    }
    if prepacked_b.k != k || prepacked_b.n != n {
        return Err(TruenoError::InvalidInput(format!(
            "PrepackedB dimension mismatch: expected ({}, {}), got ({}, {})",
            k, n, prepacked_b.k, prepacked_b.n
        )));
    }

    if m == 0 || n == 0 || k == 0 {
        return Ok(());
    }

    let track_time = profiler.is_some();
    let start = if track_time { Some(Instant::now()) } else { None };

    let mc = MC.min(m);
    let kc = KC.min(k);

    let needed_a = packed_a_size(mc, kc);
    let needed_c = MR * NR;

    // Only need A and C micro buffers — B is already packed
    TL_PACKED_A.with(|tl_a| {
        TL_C_MICRO.with(|tl_c| {
            let mut packed_a = tl_a.borrow_mut();
            let mut c_micro = tl_c.borrow_mut();

            if packed_a.len() < needed_a {
                packed_a.resize(needed_a, 0.0);
            } else {
                packed_a[..needed_a].fill(0.0);
            }
            if c_micro.len() < needed_c {
                c_micro.resize(needed_c, 0.0);
            } else {
                c_micro[..needed_c].fill(0.0);
            }

            for (jc_idx, jc) in (0..n).step_by(NC).enumerate() {
                let nc_block = NC.min(n - jc);

                for (pc_idx, pc) in (0..k).step_by(KC).enumerate() {
                    let kc_block = KC.min(k - pc);

                    // Use pre-packed B tile instead of runtime packing
                    let packed_b_tile = prepacked_b.tile(jc_idx, pc_idx);

                    for ic in (0..m).step_by(MC) {
                        let mc_block = MC.min(m - ic);

                        let pack_start = if track_time { Some(Instant::now()) } else { None };
                        pack_a_block(a, k, ic, pc, mc_block, kc_block, &mut packed_a);
                        record_prof(&mut profiler, BlisProfileLevel::Pack, pack_start, 0);

                        compute_macroblock(
                            c,
                            &packed_a,
                            packed_b_tile,
                            &mut c_micro,
                            ic,
                            jc,
                            mc_block,
                            nc_block,
                            kc_block,
                            n,
                            &mut profiler,
                        );
                    }
                }
            }

            if let (Some(prof), Some(s)) = (profiler, start) {
                prof.record(
                    BlisProfileLevel::Macro,
                    s.elapsed().as_nanos() as u64,
                    (2 * m * n * k) as u64,
                );
            }
        });
    });

    Ok(())
}
