//! AVX-512 backend implementation (x86_64 advanced SIMD)
//!
//! This backend uses AVX-512 intrinsics for 512-bit SIMD operations.
//! AVX-512 is available on Intel Skylake-X/Sapphire Rapids (2017+) and AMD Zen 4 (2022+) CPUs.
//!
//! # Performance
//!
//! Expected speedup: 16x for operations on f32 vectors (16 elements per register)
//! This provides 2x improvement over AVX2 (8 elements) and ~16x over scalar.
//!
//! # Safety
//!
//! All AVX-512 intrinsics are marked `unsafe` by Rust. This module carefully isolates
//! all unsafe code and verifies correctness through comprehensive testing.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::VectorBackend;

/// AVX-512 backend (512-bit SIMD for x86_64)
pub struct Avx512Backend;

impl VectorBackend for Avx512Backend {
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=16 for AVX-512)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used (_mm512_loadu_ps/_mm512_storeu_ps) - no alignment requirement
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 16 elements at a time using AVX-512 (512-bit = 16 x f32)
        while i + 16 <= len {
            // Load 16 floats from a and b
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));

            // Add them
            let vresult = _mm512_add_ps(va, vb);

            // Store result
            _mm512_storeu_ps(result.as_mut_ptr().add(i), vresult);

            i += 16;
        }

        // Handle remaining elements with scalar code
        for j in i..len {
            result[j] = a[j] + b[j];
        }
    }

    // Stub implementations for remaining methods - will implement in future phases
    #[target_feature(enable = "avx512f")]
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        // Scalar fallback (AVX-512 optimization pending)
        for i in 0..a.len() {
            result[i] = a[i] - b[i];
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        // Scalar fallback (AVX-512 optimization pending)
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        // Scalar fallback (AVX-512 optimization pending)
        for i in 0..a.len() {
            result[i] = a[i] / b[i];
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        // Scalar fallback (AVX-512 optimization pending)
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn sum(a: &[f32]) -> f32 {
        // Scalar fallback (AVX-512 optimization pending)
        a.iter().sum()
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn max(a: &[f32]) -> f32 {
        // Scalar fallback (AVX-512 optimization pending)
        a.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn min(a: &[f32]) -> f32 {
        // Scalar fallback (AVX-512 optimization pending)
        a.iter().copied().fold(f32::INFINITY, f32::min)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn argmax(a: &[f32]) -> usize {
        // Scalar fallback (AVX-512 optimization pending)
        a.iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn argmin(a: &[f32]) -> usize {
        // Scalar fallback (AVX-512 optimization pending)
        a.iter()
            .enumerate()
            .min_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn sum_kahan(a: &[f32]) -> f32 {
        // Scalar fallback (AVX-512 optimization pending)
        let mut sum = 0.0;
        let mut c = 0.0;
        for &x in a {
            let y = x - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn norm_l2(a: &[f32]) -> f32 {
        // Scalar fallback (AVX-512 optimization pending)
        a.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        // Scalar fallback (AVX-512 optimization pending)
        a.iter().map(|x| x.abs()).sum()
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn norm_linf(a: &[f32]) -> f32 {
        // Scalar fallback (AVX-512 optimization pending)
        a.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        // Scalar fallback (AVX-512 optimization pending)
        for i in 0..a.len() {
            result[i] = a[i] * scalar;
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn abs(a: &[f32], result: &mut [f32]) {
        // Scalar fallback (AVX-512 optimization pending)
        for i in 0..a.len() {
            result[i] = a[i].abs();
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        // Scalar fallback (AVX-512 optimization pending)
        for i in 0..a.len() {
            result[i] = a[i].clamp(min_val, max_val);
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        // Scalar fallback (AVX-512 optimization pending)
        for i in 0..a.len() {
            result[i] = a[i] + t * (b[i] - a[i]);
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        // Scalar fallback (AVX-512 optimization pending)
        for i in 0..a.len() {
            result[i] = a[i].mul_add(b[i], c[i]);
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        // Scalar fallback (AVX-512 optimization pending)
        for i in 0..a.len() {
            result[i] = a[i].max(0.0);
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn exp(a: &[f32], result: &mut [f32]) {
        // Scalar fallback (AVX-512 optimization pending)
        for i in 0..a.len() {
            result[i] = a[i].exp();
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        // Scalar fallback (AVX-512 optimization pending)
        for i in 0..a.len() {
            result[i] = 1.0 / (1.0 + (-a[i]).exp());
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        // Scalar fallback (AVX-512 optimization pending)
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        for i in 0..a.len() {
            let x = a[i];
            let cube = x * x * x;
            let inner = SQRT_2_OVER_PI * (x + 0.044715 * cube);
            result[i] = 0.5 * x * (1.0 + inner.tanh());
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        // Scalar fallback (AVX-512 optimization pending)
        for i in 0..a.len() {
            let sigmoid_val = 1.0 / (1.0 + (-a[i]).exp());
            result[i] = a[i] * sigmoid_val;
        }
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn tanh(a: &[f32], result: &mut [f32]) {
        // Scalar fallback (AVX-512 optimization pending)
        for i in 0..a.len() {
            result[i] = a[i].tanh();
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::backends::scalar::ScalarBackend;

    /// Helper to run AVX-512 test only on CPUs that support it
    fn avx512_test<F>(test_fn: F)
    where
        F: FnOnce(),
    {
        if is_x86_feature_detected!("avx512f") {
            test_fn();
        } else {
            // Skip test on CPUs without AVX-512 support
            println!("Skipping AVX-512 test (CPU does not support avx512f)");
        }
    }

    #[test]
    fn test_avx512_add_basic() {
        avx512_test(|| {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let b = vec![5.0, 6.0, 7.0, 8.0];
            let mut result = vec![0.0; 4];

            unsafe {
                Avx512Backend::add(&a, &b, &mut result);
            }

            assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
        });
    }

    #[test]
    fn test_avx512_add_aligned_16() {
        avx512_test(|| {
            // Test with exactly 16 elements (one AVX-512 register)
            let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..16).map(|i| (i + 10) as f32).collect();
            let mut result = vec![0.0; 16];
            let expected: Vec<f32> = (0..16).map(|i| (i + i + 10) as f32).collect();

            unsafe {
                Avx512Backend::add(&a, &b, &mut result);
            }

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_avx512_add_non_aligned() {
        avx512_test(|| {
            // Test with 18 elements (16 + 2 remainder)
            let a: Vec<f32> = (0..18).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..18).map(|i| (i * 2) as f32).collect();
            let mut result = vec![0.0; 18];
            let expected: Vec<f32> = (0..18).map(|i| (i + i * 2) as f32).collect();

            unsafe {
                Avx512Backend::add(&a, &b, &mut result);
            }

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_avx512_add_large() {
        avx512_test(|| {
            // Test with 1000 elements (many AVX-512 iterations)
            let a: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
            let b: Vec<f32> = (0..1000).map(|i| i as f32 * 0.3).collect();
            let mut result = vec![0.0; 1000];

            unsafe {
                Avx512Backend::add(&a, &b, &mut result);
            }

            for i in 0..1000 {
                let expected = i as f32 * 0.5 + i as f32 * 0.3;
                assert!((result[i] - expected).abs() < 1e-5,
                    "Mismatch at index {}: expected {}, got {}", i, expected, result[i]);
            }
        });
    }

    #[test]
    fn test_avx512_add_single_element() {
        avx512_test(|| {
            let a = vec![42.0];
            let b = vec![13.0];
            let mut result = vec![0.0];

            unsafe {
                Avx512Backend::add(&a, &b, &mut result);
            }

            assert_eq!(result, vec![55.0]);
        });
    }

    #[test]
    fn test_avx512_add_negative_values() {
        avx512_test(|| {
            let a = vec![-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0,
                         9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
            let b = vec![10.0, 20.0, 30.0, 40.0, -50.0, -60.0, -70.0, -80.0,
                         1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let mut result = vec![0.0; 16];
            let expected = vec![9.0, 18.0, 27.0, 36.0, -45.0, -54.0, -63.0, -72.0,
                               10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0];

            unsafe {
                Avx512Backend::add(&a, &b, &mut result);
            }

            for i in 0..16 {
                assert!((result[i] - expected[i]).abs() < 1e-5,
                    "Mismatch at index {}: expected {}, got {}", i, expected[i], result[i]);
            }
        });
    }

    #[test]
    fn test_avx512_add_equivalence_to_scalar() {
        avx512_test(|| {
            // Backend equivalence: AVX-512 should produce same results as Scalar
            let sizes = vec![1, 7, 15, 16, 17, 32, 63, 100, 1000];

            for size in sizes {
                let a: Vec<f32> = (0..size).map(|i| (i as f32 * 1.5) - 50.0).collect();
                let b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.7) + 20.0).collect();

                let mut result_avx512 = vec![0.0; size];
                let mut result_scalar = vec![0.0; size];

                unsafe {
                    Avx512Backend::add(&a, &b, &mut result_avx512);
                    ScalarBackend::add(&a, &b, &mut result_scalar);
                }

                for i in 0..size {
                    assert!((result_avx512[i] - result_scalar[i]).abs() < 1e-5,
                        "Backend mismatch at size {} index {}: AVX512={}, Scalar={}",
                        size, i, result_avx512[i], result_scalar[i]);
                }
            }
        });
    }

    #[test]
    fn test_avx512_add_special_values() {
        avx512_test(|| {
            // Test with infinity, zero, and very small/large values
            let a = vec![0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY,
                         1e-20, -1e-20, 1e20, -1e20,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            let b = vec![0.0, 0.0, 1.0, -1.0,
                         2e-20, -2e-20, 2e20, -2e20,
                         1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let mut result = vec![0.0; 16];

            unsafe {
                Avx512Backend::add(&a, &b, &mut result);
            }

            assert_eq!(result[0], 0.0);
            assert_eq!(result[1], 0.0);
            assert_eq!(result[2], f32::INFINITY);
            assert_eq!(result[3], f32::NEG_INFINITY);
            assert!((result[4] - 3e-20).abs() < 1e-25);
            assert!((result[5] + 3e-20).abs() < 1e-25);
        });
    }

    #[test]
    fn test_avx512_add_remainder_correctness() {
        avx512_test(|| {
            // Specifically test remainder handling for sizes 16+1 through 16+15
            for remainder in 1..=15 {
                let size = 16 + remainder;
                let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
                let b: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();
                let mut result = vec![0.0; size];

                unsafe {
                    Avx512Backend::add(&a, &b, &mut result);
                }

                // Verify all elements, especially the remainder portion
                for i in 0..size {
                    let expected = i as f32 + (size - i) as f32;
                    assert_eq!(result[i], expected,
                        "Remainder test failed at size {} (remainder {}), index {}",
                        size, remainder, i);
                }
            }
        });
    }
}
