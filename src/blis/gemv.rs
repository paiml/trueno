//! SIMD-accelerated GEMV (General Matrix-Vector Multiply)
//!
//! Specialized kernel for M=1 matrix-vector product: c = a × B
//! where a is 1×K and B is K×N, both row-major.
//!
//! This bypasses the BLIS 5-loop packing overhead which dominates for M=1.
//! Instead, uses direct AVX2 VFMADD on unpacked row-major data.
//!
//! # Algorithm
//!
//! Process N in chunks of 8 (AVX2 register width). For each chunk,
//! accumulate K scaled B-rows using FMA. 4-way K-unrolling hides
//! load latency and enables instruction-level parallelism.
//!
//! # References
//!
//! - GH-380: matvec (M=1) performance gap vs ndarray

/// AVX2 GEMV using axpy pattern: c += a[k] * B[k,:] for each k
///
/// Outer loop over K (4-way unrolled), inner loop over N with AVX2 VFMADD.
/// This matches row-major B access: B[k,:] is contiguous → sequential reads.
///
/// # Safety
///
/// Requires AVX2+FMA CPU features. Caller must ensure:
/// - `a` has length >= `k`
/// - `b` has length >= `k * n`
/// - `c` has length >= `n`
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn gemv_avx2(
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    unsafe {
        use std::arch::x86_64::*;

        let n8 = n / 8 * 8;

        // 4-way K-unrolled axpy with AVX2 VFMADD on inner N loop
        let k4 = k / 4 * 4;
        let mut ki = 0;
        while ki < k4 {
            let a0 = _mm256_set1_ps(*a.get_unchecked(ki));
            let a1 = _mm256_set1_ps(*a.get_unchecked(ki + 1));
            let a2 = _mm256_set1_ps(*a.get_unchecked(ki + 2));
            let a3 = _mm256_set1_ps(*a.get_unchecked(ki + 3));
            let b0_base = ki * n;
            let b1_base = b0_base + n;
            let b2_base = b1_base + n;
            let b3_base = b2_base + n;

            let mut j = 0;
            while j < n8 {
                let cv = _mm256_loadu_ps(c.get_unchecked(j));
                let bv0 = _mm256_loadu_ps(b.get_unchecked(b0_base + j));
                let bv1 = _mm256_loadu_ps(b.get_unchecked(b1_base + j));
                let bv2 = _mm256_loadu_ps(b.get_unchecked(b2_base + j));
                let bv3 = _mm256_loadu_ps(b.get_unchecked(b3_base + j));

                let r = _mm256_fmadd_ps(a0, bv0, cv);
                let r = _mm256_fmadd_ps(a1, bv1, r);
                let r = _mm256_fmadd_ps(a2, bv2, r);
                let r = _mm256_fmadd_ps(a3, bv3, r);

                _mm256_storeu_ps(c.get_unchecked_mut(j), r);
                j += 8;
            }

            // Scalar remainder for N % 8
            while j < n {
                *c.get_unchecked_mut(j) += *a.get_unchecked(ki) * *b.get_unchecked(b0_base + j)
                    + *a.get_unchecked(ki + 1) * *b.get_unchecked(b1_base + j)
                    + *a.get_unchecked(ki + 2) * *b.get_unchecked(b2_base + j)
                    + *a.get_unchecked(ki + 3) * *b.get_unchecked(b3_base + j);
                j += 1;
            }

            ki += 4;
        }

        // Remainder K (scalar axpy)
        while ki < k {
            let ak = *a.get_unchecked(ki);
            let bk_base = ki * n;
            let ak_v = _mm256_set1_ps(ak);

            let mut j = 0;
            while j < n8 {
                let cv = _mm256_loadu_ps(c.get_unchecked(j));
                let bv = _mm256_loadu_ps(b.get_unchecked(bk_base + j));
                let r = _mm256_fmadd_ps(ak_v, bv, cv);
                _mm256_storeu_ps(c.get_unchecked_mut(j), r);
                j += 8;
            }
            while j < n {
                *c.get_unchecked_mut(j) += ak * *b.get_unchecked(bk_base + j);
                j += 1;
            }
            ki += 1;
        }
    }
}

/// Scalar fallback GEMV for non-x86 or non-AVX2 platforms
pub fn gemv_scalar(
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    // 4-way K-unrolled axpy (auto-vectorizable)
    let k4 = k / 4 * 4;
    for ki in (0..k4).step_by(4) {
        let a0 = a[ki];
        let a1 = a[ki + 1];
        let a2 = a[ki + 2];
        let a3 = a[ki + 3];
        let b0 = ki * n;
        let b1 = b0 + n;
        let b2 = b1 + n;
        let b3 = b2 + n;
        for j in 0..n {
            c[j] += a0 * b[b0 + j] + a1 * b[b1 + j] + a2 * b[b2 + j] + a3 * b[b3 + j];
        }
    }

    // Remainder K
    for ki in k4..k {
        let a_k = a[ki];
        let b_start = ki * n;
        for j in 0..n {
            c[j] += a_k * b[b_start + j];
        }
    }
}

/// Dispatch GEMV to best available implementation
pub fn gemv(
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: AVX2+FMA verified by feature detection above.
            // Slice bounds are checked by the caller (matmul_vector_matrix).
            unsafe {
                gemv_avx2(k, n, a, b, c);
            }
            return;
        }
    }
    gemv_scalar(k, n, a, b, c);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemv_basic() {
        // 1×3 @ 3×4 → 1×4
        let a = [1.0, 2.0, 3.0];
        let b = [
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];
        let mut c = [0.0f32; 4];

        gemv(3, 4, &a, &b, &mut c);

        // c[j] = 1*B[0,j] + 2*B[1,j] + 3*B[2,j]
        assert!((c[0] - 38.0).abs() < 1e-5);
        assert!((c[1] - 44.0).abs() < 1e-5);
        assert!((c[2] - 50.0).abs() < 1e-5);
        assert!((c[3] - 56.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemv_identity_row_select() {
        // e_1 @ B should give B[1,:]
        let a = [0.0, 1.0, 0.0];
        let b = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let mut c = [0.0f32; 3];

        gemv(3, 3, &a, &b, &mut c);

        assert!((c[0] - 4.0).abs() < 1e-5);
        assert!((c[1] - 5.0).abs() < 1e-5);
        assert!((c[2] - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_gemv_large_n() {
        // K=2, N=17 (tests AVX2 8-element chunks + scalar remainder)
        let k = 2;
        let n = 17;
        let a = [1.0f32, 2.0];
        let b: Vec<f32> = (0..k * n).map(|i| i as f32).collect();
        let mut c = vec![0.0f32; n];

        gemv(k, n, &a, &b, &mut c);

        // Verify against scalar reference
        for j in 0..n {
            let expected = a[0] * b[j] + a[1] * b[n + j];
            assert!((c[j] - expected).abs() < 1e-4, "c[{j}] = {} expected {expected}", c[j]);
        }
    }

    #[test]
    fn test_gemv_zeros() {
        let a = [0.0f32; 4];
        let b = vec![1.0f32; 4 * 8];
        let mut c = vec![0.0f32; 8];

        gemv(4, 8, &a, &b, &mut c);

        for j in 0..8 {
            assert!((c[j]).abs() < 1e-10);
        }
    }
}
