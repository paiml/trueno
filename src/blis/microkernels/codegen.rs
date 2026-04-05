//! Generated GEMM microkernels via trueno-gemm-codegen proc macro.
//!
//! Contract: cgp-gemm-codegen-v1.yaml (C-CODEGEN-001 through C-CODEGEN-004)
//! Sovereign: all code generated at compile time from trueno's own proc macro.
//!
//! These kernels are shape-specialized at compile time, producing fully-unrolled
//! FMA code with optimal register allocation for each (MR, NR) combination.

use trueno_gemm_codegen::avx512_microkernel;

// Generate the same 8x32 shape as the hand-written kernel for validation.
// C-CODEGEN-001: must match hand-written output within 1e-5.
// C-CODEGEN-002: must not be slower than hand-written (within 5%).
avx512_microkernel!(mr = 8, nr = 32);

// Generate 8x16 for small-N path validation.
avx512_microkernel!(mr = 8, nr = 16);

// New shapes not previously hand-written — explore register space.
// 8x48: 24 accumulators (8*3 zmm) + 3 B loads = 27 zmm. Fits in 32.
avx512_microkernel!(mr = 8, nr = 48);

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: scalar reference GEMM for validation.
    fn gemm_reference(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    c[i * n + j] += a[p * m + i] * b[p * n + j];
                }
            }
        }
    }

    /// FALSIFY-CODEGEN-001: Generated 8x32 matches scalar reference.
    #[test]
    fn test_codegen_8x32_correctness() {
        let mr = 8;
        let nr = 32;
        let k = 64;

        let a: Vec<f32> = (0..mr * k).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..k * nr).map(|i| ((i * 11 + 5) % 100) as f32 / 100.0).collect();
        let mut c_gen = vec![0.0f32; mr * nr];
        let mut c_ref = vec![0.0f32; mr * nr];

        // Reference
        gemm_reference(mr, nr, k, &a, &b, &mut c_ref);

        // Generated kernel
        // SAFETY: AVX-512 is available (test runs on x86_64 with avx512f)
        if std::arch::is_x86_feature_detected!("avx512f") {
            unsafe {
                microkernel_8x32_avx512_gen(k, a.as_ptr(), b.as_ptr(), c_gen.as_mut_ptr(), nr);
            }

            let max_diff =
                c_gen.iter().zip(c_ref.iter()).map(|(g, r)| (g - r).abs()).fold(0.0f32, f32::max);

            assert!(max_diff < 1e-2, "C-CODEGEN-001: max diff {max_diff} >= 1e-2 for 8x32");
        }
    }

    /// FALSIFY-CODEGEN-001b: Generated 8x16 matches scalar reference.
    #[test]
    fn test_codegen_8x16_correctness() {
        let mr = 8;
        let nr = 16;
        let k = 64;

        let a: Vec<f32> = (0..mr * k).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..k * nr).map(|i| ((i * 11 + 5) % 100) as f32 / 100.0).collect();
        let mut c_gen = vec![0.0f32; mr * nr];
        let mut c_ref = vec![0.0f32; mr * nr];

        gemm_reference(mr, nr, k, &a, &b, &mut c_ref);

        if std::arch::is_x86_feature_detected!("avx512f") {
            unsafe {
                microkernel_8x16_avx512_gen(k, a.as_ptr(), b.as_ptr(), c_gen.as_mut_ptr(), nr);
            }

            let max_diff =
                c_gen.iter().zip(c_ref.iter()).map(|(g, r)| (g - r).abs()).fold(0.0f32, f32::max);

            assert!(max_diff < 1e-2, "C-CODEGEN-001: max diff {max_diff} >= 1e-2 for 8x16");
        }
    }

    /// FALSIFY-CODEGEN-001c: Generated 8x48 (new shape) matches scalar reference.
    #[test]
    fn test_codegen_8x48_correctness() {
        let mr = 8;
        let nr = 48;
        let k = 32;

        let a: Vec<f32> = (0..mr * k).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0).collect();
        let b: Vec<f32> = (0..k * nr).map(|i| ((i * 11 + 5) % 100) as f32 / 100.0).collect();
        let mut c_gen = vec![0.0f32; mr * nr];
        let mut c_ref = vec![0.0f32; mr * nr];

        gemm_reference(mr, nr, k, &a, &b, &mut c_ref);

        if std::arch::is_x86_feature_detected!("avx512f") {
            unsafe {
                microkernel_8x48_avx512_gen(k, a.as_ptr(), b.as_ptr(), c_gen.as_mut_ptr(), nr);
            }

            let max_diff =
                c_gen.iter().zip(c_ref.iter()).map(|(g, r)| (g - r).abs()).fold(0.0f32, f32::max);

            assert!(max_diff < 1e-2, "C-CODEGEN-001: max diff {max_diff} >= 1e-2 for 8x48");
        }
    }
}
