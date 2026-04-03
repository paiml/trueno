//! AVX-512 SIMD Microkernels
//!
//! 16×8 output tile using ZMM registers (512-bit = 16 f32).
//! Exploits the 2× vector width over AVX2 for compute-bound GEMM.
//!
//! Register allocation:
//! - zmm0-zmm7: 8 columns of C (16 f32 each) = 128 outputs in registers
//! - A column loaded per iteration, B broadcast from memory via vbroadcastss
//!
//! 4-way K-unrolled main loop hides 5-cycle FMA latency across 2 FMA ports.

/// 16×8 AVX-512 microkernel — 4-way K-unrolled.
/// A: 16×K packed column-major. B: K×8 packed row-major.
/// C: 16×8 column-major with stride ldc.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn microkernel_16x8_avx512(
    k: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    ldc: usize,
) {
    unsafe {
        use std::arch::x86_64::*;

        // Load C (8 columns of 16 elements)
        let mut c0 = _mm512_loadu_ps(c);
        let mut c1 = _mm512_loadu_ps(c.add(ldc));
        let mut c2 = _mm512_loadu_ps(c.add(2 * ldc));
        let mut c3 = _mm512_loadu_ps(c.add(3 * ldc));
        let mut c4 = _mm512_loadu_ps(c.add(4 * ldc));
        let mut c5 = _mm512_loadu_ps(c.add(5 * ldc));
        let mut c6 = _mm512_loadu_ps(c.add(6 * ldc));
        let mut c7 = _mm512_loadu_ps(c.add(7 * ldc));

        let k4 = k / 4;
        let k_rem = k % 4;

        for p4 in 0..k4 {
            let base = p4 * 4;

            // K+0
            let a0 = _mm512_loadu_ps(a.add(base * 16));
            let bp0 = b.add(base * 8);
            c0 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0), c0);
            c1 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0.add(1)), c1);
            c2 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0.add(2)), c2);
            c3 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0.add(3)), c3);
            c4 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0.add(4)), c4);
            c5 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0.add(5)), c5);
            c6 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0.add(6)), c6);
            c7 = _mm512_fmadd_ps(a0, _mm512_set1_ps(*bp0.add(7)), c7);

            // K+1
            let a1 = _mm512_loadu_ps(a.add((base + 1) * 16));
            let bp1 = b.add((base + 1) * 8);
            c0 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1), c0);
            c1 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1.add(1)), c1);
            c2 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1.add(2)), c2);
            c3 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1.add(3)), c3);
            c4 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1.add(4)), c4);
            c5 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1.add(5)), c5);
            c6 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1.add(6)), c6);
            c7 = _mm512_fmadd_ps(a1, _mm512_set1_ps(*bp1.add(7)), c7);

            // K+2
            let a2 = _mm512_loadu_ps(a.add((base + 2) * 16));
            let bp2 = b.add((base + 2) * 8);
            c0 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2), c0);
            c1 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2.add(1)), c1);
            c2 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2.add(2)), c2);
            c3 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2.add(3)), c3);
            c4 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2.add(4)), c4);
            c5 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2.add(5)), c5);
            c6 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2.add(6)), c6);
            c7 = _mm512_fmadd_ps(a2, _mm512_set1_ps(*bp2.add(7)), c7);

            // K+3
            let a3 = _mm512_loadu_ps(a.add((base + 3) * 16));
            let bp3 = b.add((base + 3) * 8);
            c0 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3), c0);
            c1 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3.add(1)), c1);
            c2 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3.add(2)), c2);
            c3 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3.add(3)), c3);
            c4 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3.add(4)), c4);
            c5 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3.add(5)), c5);
            c6 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3.add(6)), c6);
            c7 = _mm512_fmadd_ps(a3, _mm512_set1_ps(*bp3.add(7)), c7);
        }

        // Remainder
        let base_rem = k4 * 4;
        for p in 0..k_rem {
            let pp = base_rem + p;
            let a_col = _mm512_loadu_ps(a.add(pp * 16));
            let bp = b.add(pp * 8);
            c0 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp), c0);
            c1 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp.add(1)), c1);
            c2 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp.add(2)), c2);
            c3 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp.add(3)), c3);
            c4 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp.add(4)), c4);
            c5 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp.add(5)), c5);
            c6 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp.add(6)), c6);
            c7 = _mm512_fmadd_ps(a_col, _mm512_set1_ps(*bp.add(7)), c7);
        }

        // Store C
        _mm512_storeu_ps(c, c0);
        _mm512_storeu_ps(c.add(ldc), c1);
        _mm512_storeu_ps(c.add(2 * ldc), c2);
        _mm512_storeu_ps(c.add(3 * ldc), c3);
        _mm512_storeu_ps(c.add(4 * ldc), c4);
        _mm512_storeu_ps(c.add(5 * ldc), c5);
        _mm512_storeu_ps(c.add(6 * ldc), c6);
        _mm512_storeu_ps(c.add(7 * ldc), c7);
    }
}
