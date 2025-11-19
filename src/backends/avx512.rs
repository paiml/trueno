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
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=16 for AVX-512)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads used (_mm512_loadu_ps) - no alignment requirement
    // 5. FMA intrinsic (_mm512_fmadd_ps) provides better performance and numerical accuracy
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Accumulator for 16-way parallel accumulation
        let mut acc = _mm512_setzero_ps();

        // Process 16 elements at a time with FMA
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));

            // Fused multiply-add: acc = acc + (va * vb)
            // This is a single instruction on AVX-512 hardware
            acc = _mm512_fmadd_ps(va, vb, acc);

            i += 16;
        }

        // Horizontal sum: reduce 16 lanes to single value
        // AVX-512 provides a convenient intrinsic for this
        let mut result = _mm512_reduce_add_ps(acc);

        // Handle remaining elements with scalar code
        result += a[i..].iter().zip(&b[i..]).map(|(x, y)| x * y).sum::<f32>();

        result
    }

    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=16 for AVX-512)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads used (_mm512_loadu_ps) - no alignment requirement
    unsafe fn sum(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Accumulator for 16-way parallel accumulation
        let mut acc = _mm512_setzero_ps();

        // Process 16 elements at a time
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            acc = _mm512_add_ps(acc, va);
            i += 16;
        }

        // Horizontal sum: reduce 16 lanes to single value
        // AVX-512 provides a convenient intrinsic for this
        let mut result = _mm512_reduce_add_ps(acc);

        // Handle remaining elements with scalar code
        result += a[i..].iter().sum::<f32>();

        result
    }

    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=16 for AVX-512)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads used (_mm512_loadu_ps) - no alignment requirement
    unsafe fn max(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Start with first element broadcast to all 16 lanes
        let mut vmax = _mm512_set1_ps(a[0]);

        // Process 16 elements at a time
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            vmax = _mm512_max_ps(vmax, va);
            i += 16;
        }

        // Horizontal max: find maximum across all 16 lanes
        // AVX-512 provides a convenient intrinsic for this
        let mut result = _mm512_reduce_max_ps(vmax);

        // Check remaining elements
        for &val in &a[i..] {
            if val > result {
                result = val;
            }
        }

        result
    }

    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=16 for AVX-512)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads used (_mm512_loadu_ps) - no alignment requirement
    unsafe fn min(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Start with first element broadcast to all 16 lanes
        let mut vmin = _mm512_set1_ps(a[0]);

        // Process 16 elements at a time
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            vmin = _mm512_min_ps(vmin, va);
            i += 16;
        }

        // Horizontal min: find minimum across all 16 lanes
        // AVX-512 provides a convenient intrinsic for this
        let mut result = _mm512_reduce_min_ps(vmin);

        // Check remaining elements
        for &val in &a[i..] {
            if val < result {
                result = val;
            }
        }

        result
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn argmax(a: &[f32]) -> usize {
        if a.is_empty() {
            return 0;
        }

        let len = a.len();
        let mut i = 0;

        // Start with first element broadcast to all 16 lanes
        let mut vmax = _mm512_set1_ps(a[0]);

        // Process 16 elements at a time to find max value
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            vmax = _mm512_max_ps(vmax, va);
            i += 16;
        }

        // Horizontal max: find maximum value across all 16 lanes
        let mut max_val = _mm512_reduce_max_ps(vmax);

        // Check remaining elements
        for &val in &a[i..] {
            if val > max_val {
                max_val = val;
            }
        }

        // Find the index of the first occurrence of max_val
        a.iter()
            .position(|&x| x == max_val)
            .unwrap_or(0)
    }

    #[target_feature(enable = "avx512f")]
    unsafe fn argmin(a: &[f32]) -> usize {
        if a.is_empty() {
            return 0;
        }

        let len = a.len();
        let mut i = 0;

        // Start with first element broadcast to all 16 lanes
        let mut vmin = _mm512_set1_ps(a[0]);

        // Process 16 elements at a time to find min value
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            vmin = _mm512_min_ps(vmin, va);
            i += 16;
        }

        // Horizontal min: find minimum value across all 16 lanes
        let mut min_val = _mm512_reduce_min_ps(vmin);

        // Check remaining elements
        for &val in &a[i..] {
            if val < min_val {
                min_val = val;
            }
        }

        // Find the index of the first occurrence of min_val
        a.iter()
            .position(|&x| x == min_val)
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

    // =====================
    // AVX-512 dot() tests
    // =====================

    #[test]
    fn test_avx512_dot_basic() {
        avx512_test(|| {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let b = vec![5.0, 6.0, 7.0, 8.0];
            // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
            let result = unsafe { Avx512Backend::dot(&a, &b) };
            assert!((result - 70.0).abs() < 1e-5, "Expected 70.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_dot_aligned_16() {
        avx512_test(|| {
            // Test with exactly 16 elements (one AVX-512 register)
            let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
            // Expected: sum of i * (i + 1) for i in 0..16
            // = 0*1 + 1*2 + 2*3 + ... + 15*16
            let expected: f32 = (0..16).map(|i| (i * (i + 1)) as f32).sum();

            let result = unsafe { Avx512Backend::dot(&a, &b) };
            assert!((result - expected).abs() < 1e-4, "Expected {}, got {}", expected, result);
        });
    }

    #[test]
    fn test_avx512_dot_non_aligned() {
        avx512_test(|| {
            // Test with 18 elements (16 + 2 remainder)
            let a: Vec<f32> = (0..18).map(|i| (i as f32) * 1.5).collect();
            let b: Vec<f32> = (0..18).map(|i| (i as f32) * 0.7).collect();
            // Expected: sum of (i * 1.5) * (i * 0.7) = sum of i^2 * 1.05
            let expected: f32 = (0..18).map(|i| ((i * i) as f32) * 1.05).sum();

            let result = unsafe { Avx512Backend::dot(&a, &b) };
            assert!((result - expected).abs() < 1e-3, "Expected {}, got {}", expected, result);
        });
    }

    #[test]
    fn test_avx512_dot_large() {
        avx512_test(|| {
            // Test with 1000 elements (62 full AVX-512 registers + 8 remainder)
            let size = 1000;
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.3).collect();
            // Expected: sum of (i * 0.5) * (i * 0.3) = sum of i^2 * 0.15
            let expected: f32 = (0..size).map(|i| ((i * i) as f32) * 0.15).sum();

            let result = unsafe { Avx512Backend::dot(&a, &b) };
            // Larger tolerance for accumulation of floating point errors
            assert!((result - expected).abs() / expected.abs() < 1e-4,
                "Expected {}, got {}, relative error: {}",
                expected, result, ((result - expected).abs() / expected.abs()));
        });
    }

    #[test]
    fn test_avx512_dot_equivalence_to_scalar() {
        avx512_test(|| {
            // Backend equivalence: AVX-512 should produce same results as Scalar
            let sizes = vec![1, 7, 15, 16, 17, 32, 63, 100, 1000];

            for size in sizes {
                let a: Vec<f32> = (0..size).map(|i| (i as f32 * 1.5) - 50.0).collect();
                let b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.7) + 20.0).collect();

                let result_avx512 = unsafe { Avx512Backend::dot(&a, &b) };
                let result_scalar = unsafe { ScalarBackend::dot(&a, &b) };

                // Use relative tolerance for larger values
                let tolerance = if result_scalar.abs() > 1.0 {
                    result_scalar.abs() * 1e-5
                } else {
                    1e-5
                };

                assert!((result_avx512 - result_scalar).abs() < tolerance,
                    "Backend mismatch at size {}: AVX512={}, Scalar={}, diff={}",
                    size, result_avx512, result_scalar, (result_avx512 - result_scalar).abs());
            }
        });
    }

    #[test]
    fn test_avx512_dot_special_values() {
        avx512_test(|| {
            // Test with zero, negative, small, and large values
            let a = vec![0.0, -1.0, 1.0, -5.0, 5.0, 1e-10, 1e10, -1e10,
                         2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
            let b = vec![10.0, 2.0, 3.0, -2.0, 4.0, 2e-10, 2e10, -2e10,
                         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

            // Expected: 0*10 + (-1)*2 + 1*3 + (-5)*(-2) + 5*4 + (1e-10)*(2e-10) + (1e10)*(2e10) + (-1e10)*(-2e10) + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
            //         = 0 - 2 + 3 + 10 + 20 + 2e-20 + 2e20 + 2e20 + 44
            //         = 75 + 2e-20 + 4e20
            // Note: 2e-20 is negligible compared to 4e20

            let result = unsafe { Avx512Backend::dot(&a, &b) };
            let expected = unsafe { ScalarBackend::dot(&a, &b) };

            // Use relative tolerance due to large values
            let rel_error = if expected.abs() > 1.0 {
                (result - expected).abs() / expected.abs()
            } else {
                (result - expected).abs()
            };

            assert!(rel_error < 1e-5,
                "Expected {}, got {}, relative error: {}",
                expected, result, rel_error);
        });
    }

    #[test]
    fn test_avx512_dot_remainder_sizes() {
        avx512_test(|| {
            // Test all remainder sizes from 16+1 to 16+15
            for remainder in 1..=15 {
                let size = 16 + remainder;
                let a: Vec<f32> = (0..size).map(|i| (i as f32) + 1.0).collect();
                let b: Vec<f32> = (0..size).map(|i| (i as f32) + 2.0).collect();

                let result_avx512 = unsafe { Avx512Backend::dot(&a, &b) };
                let result_scalar = unsafe { ScalarBackend::dot(&a, &b) };

                let tolerance = if result_scalar.abs() > 1.0 {
                    result_scalar.abs() * 1e-5
                } else {
                    1e-5
                };

                assert!((result_avx512 - result_scalar).abs() < tolerance,
                    "Remainder test failed at size {} (remainder {}): AVX512={}, Scalar={}",
                    size, remainder, result_avx512, result_scalar);
            }
        });
    }

    #[test]
    fn test_avx512_dot_zero_vector() {
        avx512_test(|| {
            let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                         9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
            let b = vec![0.0; 16];

            let result = unsafe { Avx512Backend::dot(&a, &b) };
            assert_eq!(result, 0.0, "Dot product with zero vector should be 0.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_dot_orthogonal() {
        avx512_test(|| {
            // Orthogonal vectors: [1, 0, 0, 0, ...] and [0, 1, 0, 0, ...]
            let mut a = vec![0.0; 16];
            let mut b = vec![0.0; 16];
            a[0] = 1.0;
            b[1] = 1.0;

            let result = unsafe { Avx512Backend::dot(&a, &b) };
            assert_eq!(result, 0.0, "Dot product of orthogonal vectors should be 0.0, got {}", result);
        });
    }

    // =====================
    // AVX-512 sum() tests
    // =====================

    #[test]
    fn test_avx512_sum_basic() {
        avx512_test(|| {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            // Expected: 1 + 2 + 3 + 4 = 10
            let result = unsafe { Avx512Backend::sum(&a) };
            assert!((result - 10.0).abs() < 1e-5, "Expected 10.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_sum_aligned_16() {
        avx512_test(|| {
            // Test with exactly 16 elements (one AVX-512 register)
            let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
            // Expected: sum of 0..16 = 0+1+2+...+15 = 120
            let expected: f32 = (0..16).map(|i| i as f32).sum();

            let result = unsafe { Avx512Backend::sum(&a) };
            assert!((result - expected).abs() < 1e-4, "Expected {}, got {}", expected, result);
        });
    }

    #[test]
    fn test_avx512_sum_non_aligned() {
        avx512_test(|| {
            // Test with 18 elements (16 + 2 remainder)
            let a: Vec<f32> = (0..18).map(|i| (i as f32) * 1.5).collect();
            // Expected: sum of (i * 1.5) for i in 0..18
            let expected: f32 = (0..18).map(|i| (i as f32) * 1.5).sum();

            let result = unsafe { Avx512Backend::sum(&a) };
            assert!((result - expected).abs() < 1e-3, "Expected {}, got {}", expected, result);
        });
    }

    #[test]
    fn test_avx512_sum_large() {
        avx512_test(|| {
            // Test with 1000 elements (62 full AVX-512 registers + 8 remainder)
            let size = 1000;
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();
            // Expected: sum of (i * 0.5) for i in 0..1000
            let expected: f32 = (0..size).map(|i| (i as f32) * 0.5).sum();

            let result = unsafe { Avx512Backend::sum(&a) };
            // Larger tolerance for accumulation of floating point errors
            let rel_error = if expected.abs() > 1.0 {
                (result - expected).abs() / expected.abs()
            } else {
                (result - expected).abs()
            };
            assert!(rel_error < 1e-4,
                "Expected {}, got {}, relative error: {}",
                expected, result, rel_error);
        });
    }

    #[test]
    fn test_avx512_sum_equivalence_to_scalar() {
        avx512_test(|| {
            // Backend equivalence: AVX-512 should produce same results as Scalar
            let sizes = vec![1, 7, 15, 16, 17, 32, 63, 100, 1000];

            for size in sizes {
                let a: Vec<f32> = (0..size).map(|i| (i as f32 * 1.5) - 50.0).collect();

                let result_avx512 = unsafe { Avx512Backend::sum(&a) };
                let result_scalar = unsafe { ScalarBackend::sum(&a) };

                // Use relative tolerance for larger values
                let tolerance = if result_scalar.abs() > 1.0 {
                    result_scalar.abs() * 1e-5
                } else {
                    1e-5
                };

                assert!((result_avx512 - result_scalar).abs() < tolerance,
                    "Backend mismatch at size {}: AVX512={}, Scalar={}, diff={}",
                    size, result_avx512, result_scalar, (result_avx512 - result_scalar).abs());
            }
        });
    }

    #[test]
    fn test_avx512_sum_negative_values() {
        avx512_test(|| {
            let a = vec![-1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0,
                         -9.0, -10.0, 11.0, 12.0, -13.0, 14.0, -15.0, 16.0];
            // Expected: -1 - 2 - 3 - 4 + 5 + 6 + 7 + 8 - 9 - 10 + 11 + 12 - 13 + 14 - 15 + 16 = 22
            let expected = 22.0;

            let result = unsafe { Avx512Backend::sum(&a) };
            assert!((result - expected).abs() < 1e-5,
                "Expected {}, got {}", expected, result);
        });
    }

    #[test]
    fn test_avx512_sum_zero_vector() {
        avx512_test(|| {
            let a = vec![0.0; 16];
            let result = unsafe { Avx512Backend::sum(&a) };
            assert_eq!(result, 0.0, "Sum of zeros should be 0.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_sum_single_element() {
        avx512_test(|| {
            let a = vec![42.0];
            let result = unsafe { Avx512Backend::sum(&a) };
            assert_eq!(result, 42.0, "Sum of single element should be that element, got {}", result);
        });
    }

    #[test]
    fn test_avx512_sum_remainder_sizes() {
        avx512_test(|| {
            // Test all remainder sizes from 16+1 to 16+15
            for remainder in 1..=15 {
                let size = 16 + remainder;
                let a: Vec<f32> = (0..size).map(|i| (i as f32) + 1.0).collect();

                let result_avx512 = unsafe { Avx512Backend::sum(&a) };
                let result_scalar = unsafe { ScalarBackend::sum(&a) };

                let tolerance = if result_scalar.abs() > 1.0 {
                    result_scalar.abs() * 1e-5
                } else {
                    1e-5
                };

                assert!((result_avx512 - result_scalar).abs() < tolerance,
                    "Remainder test failed at size {} (remainder {}): AVX512={}, Scalar={}",
                    size, remainder, result_avx512, result_scalar);
            }
        });
    }

    // =====================
    // AVX-512 max() tests
    // =====================

    #[test]
    fn test_avx512_max_basic() {
        avx512_test(|| {
            let a = vec![1.0, 5.0, 3.0, 9.0, 2.0];
            let result = unsafe { Avx512Backend::max(&a) };
            assert_eq!(result, 9.0, "Expected 9.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_max_aligned_16() {
        avx512_test(|| {
            let mut a: Vec<f32> = (0..16).map(|i| i as f32).collect();
            a[8] = 100.0; // Max is in the middle
            let result = unsafe { Avx512Backend::max(&a) };
            assert_eq!(result, 100.0, "Expected 100.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_max_non_aligned() {
        avx512_test(|| {
            let mut a: Vec<f32> = (0..18).map(|i| (i as f32) * 1.5).collect();
            a[17] = 200.0; // Max is in remainder
            let result = unsafe { Avx512Backend::max(&a) };
            assert_eq!(result, 200.0, "Expected 200.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_max_negative_values() {
        avx512_test(|| {
            let a = vec![-5.0, -2.0, -10.0, -1.0, -8.0];
            let result = unsafe { Avx512Backend::max(&a) };
            assert_eq!(result, -1.0, "Expected -1.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_max_equivalence_to_scalar() {
        avx512_test(|| {
            let sizes = vec![1, 7, 15, 16, 17, 32, 63, 100, 1000];
            for size in sizes {
                let a: Vec<f32> = (0..size).map(|i| ((i * 7) % 100) as f32 - 50.0).collect();
                let result_avx512 = unsafe { Avx512Backend::max(&a) };
                let result_scalar = unsafe { ScalarBackend::max(&a) };
                assert_eq!(result_avx512, result_scalar,
                    "Backend mismatch at size {}: AVX512={}, Scalar={}",
                    size, result_avx512, result_scalar);
            }
        });
    }

    // =====================
    // AVX-512 min() tests
    // =====================

    #[test]
    fn test_avx512_min_basic() {
        avx512_test(|| {
            let a = vec![5.0, 1.0, 9.0, 3.0, 2.0];
            let result = unsafe { Avx512Backend::min(&a) };
            assert_eq!(result, 1.0, "Expected 1.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_min_aligned_16() {
        avx512_test(|| {
            let mut a: Vec<f32> = (0..16).map(|i| (i + 10) as f32).collect();
            a[8] = -100.0; // Min is in the middle
            let result = unsafe { Avx512Backend::min(&a) };
            assert_eq!(result, -100.0, "Expected -100.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_min_non_aligned() {
        avx512_test(|| {
            let mut a: Vec<f32> = (0..18).map(|i| (i as f32) * 1.5 + 10.0).collect();
            a[17] = -200.0; // Min is in remainder
            let result = unsafe { Avx512Backend::min(&a) };
            assert_eq!(result, -200.0, "Expected -200.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_min_positive_values() {
        avx512_test(|| {
            let a = vec![5.0, 2.0, 10.0, 1.0, 8.0];
            let result = unsafe { Avx512Backend::min(&a) };
            assert_eq!(result, 1.0, "Expected 1.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_min_equivalence_to_scalar() {
        avx512_test(|| {
            let sizes = vec![1, 7, 15, 16, 17, 32, 63, 100, 1000];
            for size in sizes {
                let a: Vec<f32> = (0..size).map(|i| ((i * 7) % 100) as f32 - 50.0).collect();
                let result_avx512 = unsafe { Avx512Backend::min(&a) };
                let result_scalar = unsafe { ScalarBackend::min(&a) };
                assert_eq!(result_avx512, result_scalar,
                    "Backend mismatch at size {}: AVX512={}, Scalar={}",
                    size, result_avx512, result_scalar);
            }
        });
    }

    // ============================================================================
    // argmax() tests
    // ============================================================================

    #[test]
    fn test_avx512_argmax_basic() {
        avx512_test(|| {
            let a = vec![1.0, 5.0, 3.0, 9.0, 2.0];
            let result = unsafe { Avx512Backend::argmax(&a) };
            assert_eq!(result, 3); // Index of 9.0
        });
    }

    #[test]
    fn test_avx512_argmax_aligned_16() {
        avx512_test(|| {
            let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
            let result = unsafe { Avx512Backend::argmax(&a) };
            assert_eq!(result, 15); // Maximum is at index 15
        });
    }

    #[test]
    fn test_avx512_argmax_non_aligned_18() {
        avx512_test(|| {
            let a: Vec<f32> = (0..18).map(|i| i as f32).collect();
            let result = unsafe { Avx512Backend::argmax(&a) };
            assert_eq!(result, 17); // Maximum is at index 17
        });
    }

    #[test]
    fn test_avx512_argmax_negative_values() {
        avx512_test(|| {
            let a = vec![-5.0, -2.0, -8.0, -1.0, -10.0];
            let result = unsafe { Avx512Backend::argmax(&a) };
            assert_eq!(result, 3); // Index of -1.0
        });
    }

    #[test]
    fn test_avx512_argmax_max_at_start() {
        avx512_test(|| {
            let a = vec![100.0, 1.0, 2.0, 3.0, 4.0];
            let result = unsafe { Avx512Backend::argmax(&a) };
            assert_eq!(result, 0); // Maximum is at index 0
        });
    }

    #[test]
    fn test_avx512_argmax_backend_equivalence() {
        avx512_test(|| {
            let sizes = [16, 17, 100, 1000, 10000, 16384, 16385, 100000, 1000000];
            for &size in &sizes {
                let a: Vec<f32> = (0..size).map(|i| ((i * 13) % 100) as f32 - 50.0).collect();
                let result_avx512 = unsafe { Avx512Backend::argmax(&a) };
                let result_scalar = unsafe { ScalarBackend::argmax(&a) };
                assert_eq!(result_avx512, result_scalar,
                    "Backend mismatch at size {}: AVX512={}, Scalar={}",
                    size, result_avx512, result_scalar);
            }
        });
    }

    // ============================================================================
    // argmin() tests
    // ============================================================================

    #[test]
    fn test_avx512_argmin_basic() {
        avx512_test(|| {
            let a = vec![5.0, 1.0, 9.0, 3.0, 2.0];
            let result = unsafe { Avx512Backend::argmin(&a) };
            assert_eq!(result, 1); // Index of 1.0
        });
    }

    #[test]
    fn test_avx512_argmin_aligned_16() {
        avx512_test(|| {
            let a: Vec<f32> = (0..16).rev().map(|i| i as f32).collect();
            let result = unsafe { Avx512Backend::argmin(&a) };
            assert_eq!(result, 15); // Minimum is at index 15
        });
    }

    #[test]
    fn test_avx512_argmin_non_aligned_18() {
        avx512_test(|| {
            let a: Vec<f32> = (0..18).rev().map(|i| i as f32).collect();
            let result = unsafe { Avx512Backend::argmin(&a) };
            assert_eq!(result, 17); // Minimum is at index 17
        });
    }

    #[test]
    fn test_avx512_argmin_positive_values() {
        avx512_test(|| {
            let a = vec![10.0, 5.0, 8.0, 2.0, 15.0];
            let result = unsafe { Avx512Backend::argmin(&a) };
            assert_eq!(result, 3); // Index of 2.0
        });
    }

    #[test]
    fn test_avx512_argmin_min_at_start() {
        avx512_test(|| {
            let a = vec![1.0, 100.0, 200.0, 300.0, 400.0];
            let result = unsafe { Avx512Backend::argmin(&a) };
            assert_eq!(result, 0); // Minimum is at index 0
        });
    }

    #[test]
    fn test_avx512_argmin_backend_equivalence() {
        avx512_test(|| {
            let sizes = [16, 17, 100, 1000, 10000, 16384, 16385, 100000, 1000000];
            for &size in &sizes {
                let a: Vec<f32> = (0..size).map(|i| ((i * 13) % 100) as f32 - 50.0).collect();
                let result_avx512 = unsafe { Avx512Backend::argmin(&a) };
                let result_scalar = unsafe { ScalarBackend::argmin(&a) };
                assert_eq!(result_avx512, result_scalar,
                    "Backend mismatch at size {}: AVX512={}, Scalar={}",
                    size, result_avx512, result_scalar);
            }
        });
    }
}
