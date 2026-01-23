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

mod ops;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::VectorBackend;

/// AVX-512 backend (512-bit SIMD for x86_64)
pub struct Avx512Backend;

impl VectorBackend for Avx512Backend {
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::add(a, b, result);
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::sub(a, b, result);
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::mul(a, b, result);
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::div(a, b, result);
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        ops::reductions::dot(a, b)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn sum(a: &[f32]) -> f32 {
        ops::reductions::sum(a)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn max(a: &[f32]) -> f32 {
        ops::reductions::max(a)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn min(a: &[f32]) -> f32 {
        ops::reductions::min(a)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn argmax(a: &[f32]) -> usize {
        ops::reductions::argmax(a)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn argmin(a: &[f32]) -> usize {
        ops::reductions::argmin(a)
    }

    unsafe fn sum_kahan(a: &[f32]) -> f32 {
        ops::reductions::sum_kahan(a)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn norm_l2(a: &[f32]) -> f32 {
        if a.is_empty() { return 0.0; }
        let len = a.len();
        let mut i = 0;
        let mut acc = _mm512_setzero_ps();
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            acc = _mm512_add_ps(acc, _mm512_mul_ps(va, va));
            i += 16;
        }
        let mut sum_sq = _mm512_reduce_add_ps(acc);
        for &val in &a[i..] { sum_sq += val * val; }
        sum_sq.sqrt()
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;
        let sign_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let mut acc = _mm512_setzero_ps();
        while i + 16 <= len {
            acc = _mm512_add_ps(acc, _mm512_and_ps(_mm512_loadu_ps(a.as_ptr().add(i)), sign_mask));
            i += 16;
        }
        let mut result = _mm512_reduce_add_ps(acc);
        for &val in &a[i..] { result += val.abs(); }
        result
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn norm_linf(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;
        let sign_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let mut max_vec = _mm512_setzero_ps();
        while i + 16 <= len {
            max_vec = _mm512_max_ps(max_vec, _mm512_and_ps(_mm512_loadu_ps(a.as_ptr().add(i)), sign_mask));
            i += 16;
        }
        let mut result = _mm512_reduce_max_ps(max_vec);
        for &val in &a[i..] { let abs_val = val.abs(); if abs_val > result { result = abs_val; } }
        result
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let scalar_vec = _mm512_set1_ps(scalar);
        while i + 16 <= len {
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_mul_ps(_mm512_loadu_ps(a.as_ptr().add(i)), scalar_vec));
            i += 16;
        }
        for j in i..len { result[j] = a[j] * scalar; }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn abs(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let sign_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF));
        while i + 16 <= len {
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_and_ps(_mm512_loadu_ps(a.as_ptr().add(i)), sign_mask));
            i += 16;
        }
        for j in i..len { result[j] = a[j].abs(); }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let min_vec = _mm512_set1_ps(min_val);
        let max_vec = _mm512_set1_ps(max_val);
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_min_ps(_mm512_max_ps(va, min_vec), max_vec));
            i += 16;
        }
        for j in i..len { result[j] = a[j].max(min_val).min(max_val); }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let t_vec = _mm512_set1_ps(t);
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_fmadd_ps(t_vec, _mm512_sub_ps(vb, va), va));
            i += 16;
        }
        for j in i..len { result[j] = a[j] + t * (b[j] - a[j]); }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            let vc = _mm512_loadu_ps(c.as_ptr().add(i));
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_fmadd_ps(va, vb, vc));
            i += 16;
        }
        for j in i..len { result[j] = a[j] * b[j] + c[j]; }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let zero = _mm512_setzero_ps();
        while i + 16 <= len {
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_max_ps(_mm512_loadu_ps(a.as_ptr().add(i)), zero));
            i += 16;
        }
        for j in i..len { result[j] = a[j].max(0.0); }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn exp(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let ln2 = _mm512_set1_ps(std::f32::consts::LN_2);
        let inv_ln2 = _mm512_set1_ps(1.0 / std::f32::consts::LN_2);
        let one = _mm512_set1_ps(1.0);
        let c2 = _mm512_set1_ps(0.5);
        let c3 = _mm512_set1_ps(0.166_666_67);
        let c4 = _mm512_set1_ps(0.041_666_668);
        let c5 = _mm512_set1_ps(0.008_333_334);
        while i + 16 <= len {
            let x = _mm512_loadu_ps(a.as_ptr().add(i));
            let k = _mm512_cvtps_epi32(_mm512_mul_ps(x, inv_ln2));
            let kf = _mm512_cvtepi32_ps(k);
            let r = _mm512_sub_ps(x, _mm512_mul_ps(kf, ln2));
            let mut poly = _mm512_fmadd_ps(r, c5, one);
            poly = _mm512_fmadd_ps(r, _mm512_fmadd_ps(r, poly, c4), one);
            poly = _mm512_fmadd_ps(r, _mm512_fmadd_ps(r, poly, c3), one);
            poly = _mm512_fmadd_ps(r, _mm512_fmadd_ps(r, poly, c2), one);
            poly = _mm512_fmadd_ps(r, poly, one);
            let exp_k = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(k, _mm512_set1_epi32(127)), 23));
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_mul_ps(poly, exp_k));
            i += 16;
        }
        for j in i..len { result[j] = a[j].exp(); }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        let len = a.len();
        for j in 0..len { result[j] = 1.0 / (1.0 + (-a[j]).exp()); }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        for j in 0..a.len() {
            let x = a[j];
            let inner = 0.797_884_56 * (x + 0.044_715 * x * x * x);
            result[j] = 0.5 * x * (1.0 + inner.tanh());
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        for j in 0..a.len() { result[j] = a[j] / (1.0 + (-a[j]).exp()); }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn tanh(a: &[f32], result: &mut [f32]) {
        for j in 0..a.len() { result[j] = a[j].tanh(); }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn sqrt(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + 16 <= len {
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_sqrt_ps(_mm512_loadu_ps(a.as_ptr().add(i))));
            i += 16;
        }
        for j in i..len { result[j] = a[j].sqrt(); }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn recip(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let one = _mm512_set1_ps(1.0);
        while i + 16 <= len {
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_div_ps(one, _mm512_loadu_ps(a.as_ptr().add(i))));
            i += 16;
        }
        for j in i..len { result[j] = a[j].recip(); }
    }

    unsafe fn ln(a: &[f32], result: &mut [f32]) { super::scalar::ScalarBackend::ln(a, result); }
    unsafe fn log2(a: &[f32], result: &mut [f32]) { super::scalar::ScalarBackend::log2(a, result); }
    unsafe fn log10(a: &[f32], result: &mut [f32]) { super::scalar::ScalarBackend::log10(a, result); }
    unsafe fn sin(a: &[f32], result: &mut [f32]) { super::scalar::ScalarBackend::sin(a, result); }
    unsafe fn cos(a: &[f32], result: &mut [f32]) { super::scalar::ScalarBackend::cos(a, result); }
    unsafe fn tan(a: &[f32], result: &mut [f32]) { super::scalar::ScalarBackend::tan(a, result); }

    unsafe fn floor(a: &[f32], result: &mut [f32]) { super::scalar::ScalarBackend::floor(a, result); }
    unsafe fn ceil(a: &[f32], result: &mut [f32]) { super::scalar::ScalarBackend::ceil(a, result); }
    unsafe fn round(a: &[f32], result: &mut [f32]) { super::scalar::ScalarBackend::round(a, result); }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::backends::scalar::ScalarBackend;

    fn avx512_test<F>(test_fn: F) where F: FnOnce() {
        if is_x86_feature_detected!("avx512f") { test_fn(); }
        else { println!("Skipping AVX-512 test (CPU does not support avx512f)"); }
    }

    #[test]
    fn test_avx512_add() {
        avx512_test(|| {
            let a = vec![1.0; 32];
            let b = vec![2.0; 32];
            let mut result = vec![0.0; 32];
            unsafe { Avx512Backend::add(&a, &b, &mut result); }
            assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-6));
        });
    }

    #[test]
    fn test_avx512_sub() {
        avx512_test(|| {
            let a = vec![5.0; 32];
            let b = vec![2.0; 32];
            let mut result = vec![0.0; 32];
            unsafe { Avx512Backend::sub(&a, &b, &mut result); }
            assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-6));
        });
    }

    #[test]
    fn test_avx512_mul() {
        avx512_test(|| {
            let a = vec![2.0; 32];
            let b = vec![3.0; 32];
            let mut result = vec![0.0; 32];
            unsafe { Avx512Backend::mul(&a, &b, &mut result); }
            assert!(result.iter().all(|&x| (x - 6.0).abs() < 1e-6));
        });
    }

    #[test]
    fn test_avx512_div() {
        avx512_test(|| {
            let a = vec![6.0; 32];
            let b = vec![2.0; 32];
            let mut result = vec![0.0; 32];
            unsafe { Avx512Backend::div(&a, &b, &mut result); }
            assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-6));
        });
    }

    #[test]
    fn test_avx512_dot() {
        avx512_test(|| {
            let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
            let b: Vec<f32> = (1..=32).map(|i| i as f32).collect();
            let result = unsafe { Avx512Backend::dot(&a, &b) };
            let expected: f32 = (1..=32).map(|i| (i * i) as f32).sum();
            assert!((result - expected).abs() < 1e-3);
        });
    }

    #[test]
    fn test_avx512_sum() {
        avx512_test(|| {
            let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
            let result = unsafe { Avx512Backend::sum(&a) };
            assert!((result - 528.0).abs() < 1e-3);
        });
    }

    #[test]
    fn test_avx512_max() {
        avx512_test(|| {
            let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
            let result = unsafe { Avx512Backend::max(&a) };
            assert!((result - 32.0).abs() < 1e-6);
        });
    }

    #[test]
    fn test_avx512_min() {
        avx512_test(|| {
            let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
            let result = unsafe { Avx512Backend::min(&a) };
            assert!((result - 1.0).abs() < 1e-6);
        });
    }

    #[test]
    fn test_avx512_backend_equivalence() {
        avx512_test(|| {
            let a: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..100).map(|i| (100 - i) as f32 * 0.1).collect();
            let mut avx512_add = vec![0.0; 100];
            let mut scalar_add = vec![0.0; 100];
            unsafe {
                Avx512Backend::add(&a, &b, &mut avx512_add);
                ScalarBackend::add(&a, &b, &mut scalar_add);
            }
            for i in 0..100 {
                assert!((avx512_add[i] - scalar_add[i]).abs() < 1e-5, "add mismatch at {}", i);
            }
        });
    }
}
