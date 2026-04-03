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
    for jj in 0..nr {
        for ii in 0..mr {
            c_micro[jj * MR + ii] = c[(row + ii) * n + (col + jj)];
        }
        for ii in mr..MR {
            c_micro[jj * MR + ii] = 0.0;
        }
    }
    for jj in nr..NR {
        for ii in 0..MR {
            c_micro[jj * MR + ii] = 0.0;
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

    for ir in (0..mc_block).step_by(MR) {
        let mr_block = MR.min(mc_block - ir);
        for jr in (0..nc_block).step_by(NR) {
            let nr_block = NR.min(nc_block - jr);
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

/// Small-matrix stride-based GEMM — no packing, no c_micro buffer.
/// For m,n,k <= 96 where packing overhead > cache benefit.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gemm_small_strided_avx2(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    use std::arch::x86_64::*;
    unsafe {
        for jr in (0..n).step_by(NR) {
            let nr = NR.min(n - jr);
            for ir in (0..m).step_by(MR) {
                let mr = MR.min(m - ir);
                let mut cv = [_mm256_setzero_ps(); 6];
                for j in 0..nr {
                    if mr == MR {
                        cv[j] = _mm256_set_ps(
                            *c.get_unchecked((ir + 7) * n + jr + j),
                            *c.get_unchecked((ir + 6) * n + jr + j),
                            *c.get_unchecked((ir + 5) * n + jr + j),
                            *c.get_unchecked((ir + 4) * n + jr + j),
                            *c.get_unchecked((ir + 3) * n + jr + j),
                            *c.get_unchecked((ir + 2) * n + jr + j),
                            *c.get_unchecked((ir + 1) * n + jr + j),
                            *c.get_unchecked(ir * n + jr + j),
                        );
                    } else {
                        let mut t = [0.0f32; 8];
                        for i in 0..mr {
                            t[i] = *c.get_unchecked((ir + i) * n + jr + j);
                        }
                        cv[j] = _mm256_loadu_ps(t.as_ptr());
                    }
                }
                for p in 0..k {
                    let a_col = if mr == MR {
                        _mm256_set_ps(
                            *a.get_unchecked((ir + 7) * k + p),
                            *a.get_unchecked((ir + 6) * k + p),
                            *a.get_unchecked((ir + 5) * k + p),
                            *a.get_unchecked((ir + 4) * k + p),
                            *a.get_unchecked((ir + 3) * k + p),
                            *a.get_unchecked((ir + 2) * k + p),
                            *a.get_unchecked((ir + 1) * k + p),
                            *a.get_unchecked(ir * k + p),
                        )
                    } else {
                        let mut t = [0.0f32; 8];
                        for i in 0..mr {
                            t[i] = *a.get_unchecked((ir + i) * k + p);
                        }
                        _mm256_loadu_ps(t.as_ptr())
                    };
                    let bp = b.as_ptr().add(p * n + jr);
                    // Unrolled FMA for NR=6 common case
                    if nr == NR {
                        cv[0] = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bp), cv[0]);
                        cv[1] = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bp.add(1)), cv[1]);
                        cv[2] = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bp.add(2)), cv[2]);
                        cv[3] = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bp.add(3)), cv[3]);
                        cv[4] = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bp.add(4)), cv[4]);
                        cv[5] = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bp.add(5)), cv[5]);
                    } else {
                        for j in 0..nr {
                            cv[j] = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bp.add(j)), cv[j]);
                        }
                    }
                }
                for j in 0..nr {
                    let mut t = [0.0f32; 8];
                    _mm256_storeu_ps(t.as_mut_ptr(), cv[j]);
                    for i in 0..mr {
                        *c.get_unchecked_mut((ir + i) * n + jr + j) = t[i];
                    }
                }
            }
        }
    }
    Ok(())
}

/// No-alloc 8x8 GEMM: stack buffers, SIMD transpose packing, K-unrolled micro-kernel.
/// For m,n divisible by 8 and m,n,k ≤ 256.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gemm_small_nopack_8x8(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    use std::arch::x86_64::*;

    let panels_m = m / 8;
    let panels_n = n / 8;

    // Stack buffers — no heap allocation.
    // A panel: 8×K col-major. B panel: K×8 row-major.
    let mut packed_a = [0.0f32; 8 * 256];
    let mut packed_b = [0.0f32; 256 * 8];
    let mut c_micro = [0.0f32; 64];

    unsafe {
        for ir_panel in 0..panels_m {
            let ir = ir_panel * 8;

            // Pack A panel: SIMD 8×8 transpose blocks
            let k_blocks = k / 8;
            let k_rem = k_blocks * 8;
            for kb in 0..k_blocks {
                let p = kb * 8;
                let r0 = _mm256_loadu_ps(a.as_ptr().add(ir * k + p));
                let r1 = _mm256_loadu_ps(a.as_ptr().add((ir + 1) * k + p));
                let r2 = _mm256_loadu_ps(a.as_ptr().add((ir + 2) * k + p));
                let r3 = _mm256_loadu_ps(a.as_ptr().add((ir + 3) * k + p));
                let r4 = _mm256_loadu_ps(a.as_ptr().add((ir + 4) * k + p));
                let r5 = _mm256_loadu_ps(a.as_ptr().add((ir + 5) * k + p));
                let r6 = _mm256_loadu_ps(a.as_ptr().add((ir + 6) * k + p));
                let r7 = _mm256_loadu_ps(a.as_ptr().add((ir + 7) * k + p));

                let t0 = _mm256_unpacklo_ps(r0, r1);
                let t1 = _mm256_unpackhi_ps(r0, r1);
                let t2 = _mm256_unpacklo_ps(r2, r3);
                let t3 = _mm256_unpackhi_ps(r2, r3);
                let t4 = _mm256_unpacklo_ps(r4, r5);
                let t5 = _mm256_unpackhi_ps(r4, r5);
                let t6 = _mm256_unpacklo_ps(r6, r7);
                let t7 = _mm256_unpackhi_ps(r6, r7);

                let u0 = _mm256_shuffle_ps(t0, t2, 0x44);
                let u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
                let u2 = _mm256_shuffle_ps(t1, t3, 0x44);
                let u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
                let u4 = _mm256_shuffle_ps(t4, t6, 0x44);
                let u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
                let u6 = _mm256_shuffle_ps(t5, t7, 0x44);
                let u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

                let dst = packed_a.as_mut_ptr().add(p * 8);
                _mm256_storeu_ps(dst, _mm256_permute2f128_ps(u0, u4, 0x20));
                _mm256_storeu_ps(dst.add(8), _mm256_permute2f128_ps(u1, u5, 0x20));
                _mm256_storeu_ps(dst.add(16), _mm256_permute2f128_ps(u2, u6, 0x20));
                _mm256_storeu_ps(dst.add(24), _mm256_permute2f128_ps(u3, u7, 0x20));
                _mm256_storeu_ps(dst.add(32), _mm256_permute2f128_ps(u0, u4, 0x31));
                _mm256_storeu_ps(dst.add(40), _mm256_permute2f128_ps(u1, u5, 0x31));
                _mm256_storeu_ps(dst.add(48), _mm256_permute2f128_ps(u2, u6, 0x31));
                _mm256_storeu_ps(dst.add(56), _mm256_permute2f128_ps(u3, u7, 0x31));
            }
            for p in k_rem..k {
                for i in 0..8 {
                    *packed_a.get_unchecked_mut(p * 8 + i) = *a.get_unchecked((ir + i) * k + p);
                }
            }

            for jr_panel in 0..panels_n {
                let jr = jr_panel * 8;

                // Pack B panel: SIMD 8-wide contiguous copies (only once per M-iteration
                // but B changes per N-panel, so pack here — still only panels_n × K copies)
                for p in 0..k {
                    _mm256_storeu_ps(
                        packed_b.as_mut_ptr().add(p * 8),
                        _mm256_loadu_ps(b.as_ptr().add(p * n + jr)),
                    );
                }

                // Zero C tile accumulator
                for v in c_micro.iter_mut() {
                    *v = 0.0;
                }

                // 8×8 micro-kernel: K-unrolled by 4 for ILP
                let ap = packed_a.as_ptr();
                let bp = packed_b.as_ptr();
                let k4 = k / 4;
                let k4_rem = k4 * 4;

                let mut c0 = _mm256_setzero_ps();
                let mut c1 = _mm256_setzero_ps();
                let mut c2 = _mm256_setzero_ps();
                let mut c3 = _mm256_setzero_ps();
                let mut c4 = _mm256_setzero_ps();
                let mut c5 = _mm256_setzero_ps();
                let mut c6 = _mm256_setzero_ps();
                let mut c7 = _mm256_setzero_ps();

                for p4 in 0..k4 {
                    let p = p4 * 4;
                    // Unroll K by 4: 4 A loads, 4×8 B broadcasts, 4×8 FMAs
                    for dp in 0..4 {
                        let pp = p + dp;
                        let a_col = _mm256_loadu_ps(ap.add(pp * 8));
                        let bpp = bp.add(pp * 8);
                        c0 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp), c0);
                        c1 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp.add(1)), c1);
                        c2 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp.add(2)), c2);
                        c3 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp.add(3)), c3);
                        c4 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp.add(4)), c4);
                        c5 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp.add(5)), c5);
                        c6 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp.add(6)), c6);
                        c7 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp.add(7)), c7);
                    }
                }
                // K remainder
                for pp in k4_rem..k {
                    let a_col = _mm256_loadu_ps(ap.add(pp * 8));
                    let bpp = bp.add(pp * 8);
                    c0 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp), c0);
                    c1 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp.add(1)), c1);
                    c2 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp.add(2)), c2);
                    c3 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp.add(3)), c3);
                    c4 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp.add(4)), c4);
                    c5 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp.add(5)), c5);
                    c6 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp.add(6)), c6);
                    c7 = _mm256_fmadd_ps(a_col, _mm256_set1_ps(*bpp.add(7)), c7);
                }

                // Accumulate into C (row-major output)
                // c_micro is column-major (col j at offset j*8), but we write row-major
                for ii in 0..8 {
                    let row_base = (ir + ii) * n + jr;
                    let ci = [c0, c1, c2, c3, c4, c5, c6, c7];
                    for jj in 0..8 {
                        // Extract element ii from column jj accumulator
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), ci[jj]);
                        *c.get_unchecked_mut(row_base + jj) += tmp[ii];
                    }
                }
            }
        }
    }
    Ok(())
}

/// Small-matrix 8x8 GEMM — stack-packed A/B, striped 8x8 AVX2 kernel.
/// Fewer tiles than 8x6 (64 outputs vs 48 per tile = 33% fewer tiles).
/// For dimensions that are multiples of 8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gemm_small_8x8(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<(), TruenoError> {
    use crate::blis::microkernels::microkernel_8x8_avx2_fma;
    // Stack pack buffers (max 128*128 = 64KB each for 128x128)
    let mut packed_a = vec![0.0f32; m * k];
    let mut packed_b = vec![0.0f32; k * n];
    let mut c_micro = [0.0f32; 8 * 8]; // 8x8 tile

    // Pack A: row-major a[i*k+p] → column-major packed_a[p*8 + (i%8)] per 8-row panel
    let panels_m = (m + 7) / 8;
    for panel in 0..panels_m {
        let ir = panel * 8;
        let mr = 8.min(m - ir);
        for p in 0..k {
            for i in 0..8 {
                unsafe {
                    packed_a[panel * 8 * k + p * 8 + i] =
                        if i < mr { *a.get_unchecked((ir + i) * k + p) } else { 0.0 };
                }
            }
        }
    }
    // Pack B: row-major b[p*n+j] → row-major packed_b[panel*8*k + p*8 + (j%8)]
    let panels_n = (n + 7) / 8;
    for panel in 0..panels_n {
        let jr = panel * 8;
        let nr = 8.min(n - jr);
        for p in 0..k {
            for j in 0..8 {
                unsafe {
                    packed_b[panel * 8 * k + p * 8 + j] =
                        if j < nr { *b.get_unchecked(p * n + jr + j) } else { 0.0 };
                }
            }
        }
    }

    // Run 8x8 micro-tiles
    unsafe {
        for ir_panel in 0..panels_m {
            let ir = ir_panel * 8;
            let mr = 8.min(m - ir);
            for jr_panel in 0..panels_n {
                let jr = jr_panel * 8;
                let nr = 8.min(n - jr);
                // Load C tile (8x8 column-major)
                for jj in 0..8 {
                    for ii in 0..8 {
                        c_micro[jj * 8 + ii] = if ii < mr && jj < nr {
                            *c.get_unchecked((ir + ii) * n + jr + jj)
                        } else {
                            0.0
                        };
                    }
                }
                let ap = packed_a.as_ptr().add(ir_panel * 8 * k);
                let bp = packed_b.as_ptr().add(jr_panel * 8 * k);
                microkernel_8x8_avx2_fma(k, ap, bp, c_micro.as_mut_ptr(), 8);
                // Store C tile
                for jj in 0..nr {
                    for ii in 0..mr {
                        *c.get_unchecked_mut((ir + ii) * n + jr + jj) = c_micro[jj * 8 + ii];
                    }
                }
            }
        }
    }
    Ok(())
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

    // Small: optimized no-pack GEMM for ≤256 (skip when profiler active).
    #[cfg(target_arch = "x86_64")]
    if profiler.is_none()
        && m <= 256
        && n <= 256
        && k <= 256
        && m > MR
        && is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("fma")
    {
        unsafe {
            if m % 8 == 0 && n % 8 == 0 {
                return gemm_small_nopack_8x8(m, n, k, a, b, c);
            }
            return gemm_small_strided_avx2(m, n, k, a, b, c);
        }
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
                }
                if packed_b.len() < needed_b {
                    packed_b.resize(needed_b, 0.0);
                }
                if c_micro.len() < needed_c {
                    c_micro.resize(needed_c, 0.0);
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
