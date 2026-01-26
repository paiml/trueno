//! SSE2 backend implementation (x86_64 baseline SIMD)
//!
//! This backend uses SSE2 intrinsics for 128-bit SIMD operations.
//! SSE2 is available on all x86_64 CPUs as a baseline requirement.
//!
//! # Performance
//!
//! Expected speedup: 4x for operations on aligned f32 vectors (4 elements per register)
//!
//! # Safety
//!
//! All SSE2 intrinsics are marked `unsafe` by Rust. This module carefully isolates
//! all unsafe code and verifies correctness through comprehensive testing.

mod ops;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::VectorBackend;

/// SSE2 backend (128-bit SIMD for x86_64)
pub struct Sse2Backend;

impl VectorBackend for Sse2Backend {
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::add(a, b, result);
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::sub(a, b, result);
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::mul(a, b, result);
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        ops::arithmetic::div(a, b, result);
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        ops::reductions::dot(a, b)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn sum(a: &[f32]) -> f32 {
        ops::reductions::sum(a)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn max(a: &[f32]) -> f32 {
        ops::reductions::max(a)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn min(a: &[f32]) -> f32 {
        ops::reductions::min(a)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn argmax(a: &[f32]) -> usize {
        ops::reductions::argmax(a)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn argmin(a: &[f32]) -> usize {
        ops::reductions::argmin(a)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn sum_kahan(a: &[f32]) -> f32 {
        ops::reductions::sum_kahan(a)
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn norm_l2(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }
        Self::dot(a, a).sqrt()
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }
        let len = a.len();
        let mut i = 0;
        let mut acc = _mm_setzero_ps();
        let sign_mask = _mm_set1_ps(f32::from_bits(0x7FFF_FFFF));
        while i + 4 <= len {
            acc = _mm_add_ps(acc, _mm_and_ps(_mm_loadu_ps(a.as_ptr().add(i)), sign_mask));
            i += 4;
        }
        let mut result = {
            let temp = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
            let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };
        for &val in &a[i..] {
            result += val.abs();
        }
        result
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn norm_linf(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }
        let len = a.len();
        let mut i = 0;
        let mut max_vec = _mm_setzero_ps();
        let sign_mask = _mm_set1_ps(f32::from_bits(0x7FFF_FFFF));
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            max_vec = _mm_max_ps(max_vec, _mm_and_ps(va, sign_mask));
            i += 4;
        }
        let mut result = {
            let temp = _mm_max_ps(max_vec, _mm_movehl_ps(max_vec, max_vec));
            let temp = _mm_max_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };
        for &val in &a[i..] {
            let abs_val = val.abs();
            if abs_val > result {
                result = abs_val;
            }
        }
        result
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let scalar_vec = _mm_set1_ps(scalar);
        while i + 4 <= len {
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_mul_ps(_mm_loadu_ps(a.as_ptr().add(i)), scalar_vec),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] * scalar;
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn abs(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let sign_mask = _mm_set1_ps(f32::from_bits(0x7FFF_FFFF));
        while i + 4 <= len {
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_and_ps(_mm_loadu_ps(a.as_ptr().add(i)), sign_mask),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].abs();
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let min_vec = _mm_set1_ps(min_val);
        let max_vec = _mm_set1_ps(max_val);
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_min_ps(_mm_max_ps(va, min_vec), max_vec),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].max(min_val).min(max_val);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let t_vec = _mm_set1_ps(t);
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_add_ps(va, _mm_mul_ps(t_vec, _mm_sub_ps(vb, va))),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] + t * (b[j] - a[j]);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let vc = _mm_loadu_ps(c.as_ptr().add(i));
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_add_ps(_mm_mul_ps(va, vb), vc),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] * b[j] + c[j];
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let zero = _mm_setzero_ps();
        while i + 4 <= len {
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_max_ps(_mm_loadu_ps(a.as_ptr().add(i)), zero),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].max(0.0);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn exp(a: &[f32], result: &mut [f32]) {
        // Polynomial approximation for exp - range reduction + polynomial
        let len = a.len();
        let mut i = 0;
        let ln2 = _mm_set1_ps(std::f32::consts::LN_2);
        let inv_ln2 = _mm_set1_ps(1.0 / std::f32::consts::LN_2);
        let c1 = _mm_set1_ps(1.0);
        let c2 = _mm_set1_ps(0.5);
        let c3 = _mm_set1_ps(0.166_666_67);
        let c4 = _mm_set1_ps(0.041_666_668);
        let c5 = _mm_set1_ps(0.008_333_334);
        while i + 4 <= len {
            let x = _mm_loadu_ps(a.as_ptr().add(i));
            let k = _mm_cvtps_epi32(_mm_mul_ps(x, inv_ln2));
            let kf = _mm_cvtepi32_ps(k);
            let r = _mm_sub_ps(x, _mm_mul_ps(kf, ln2));
            let mut poly = _mm_add_ps(c1, _mm_mul_ps(r, c5));
            poly = _mm_add_ps(c1, _mm_mul_ps(r, _mm_add_ps(c4, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(c1, _mm_mul_ps(r, _mm_add_ps(c3, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(c1, _mm_mul_ps(r, _mm_add_ps(c2, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(c1, _mm_mul_ps(r, poly));
            let exp_k = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(k, _mm_set1_epi32(127)), 23));
            _mm_storeu_ps(result.as_mut_ptr().add(i), _mm_mul_ps(poly, exp_k));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].exp();
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        let len = a.len();
        let mut i = 0;
        let one = _mm_set1_ps(1.0);
        let neg_one = _mm_set1_ps(-1.0);
        let ln2 = _mm_set1_ps(std::f32::consts::LN_2);
        let inv_ln2 = _mm_set1_ps(1.0 / std::f32::consts::LN_2);
        let c2 = _mm_set1_ps(0.5);
        let c3 = _mm_set1_ps(0.166_666_67);
        let c4 = _mm_set1_ps(0.041_666_668);
        let c5 = _mm_set1_ps(0.008_333_334);
        while i + 4 <= len {
            let x = _mm_loadu_ps(a.as_ptr().add(i));
            let neg_x = _mm_mul_ps(x, neg_one);
            let k = _mm_cvtps_epi32(_mm_mul_ps(neg_x, inv_ln2));
            let kf = _mm_cvtepi32_ps(k);
            let r = _mm_sub_ps(neg_x, _mm_mul_ps(kf, ln2));
            let mut poly = _mm_add_ps(one, _mm_mul_ps(r, c5));
            poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c4, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c3, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c2, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(one, _mm_mul_ps(r, poly));
            let exp_k = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(k, _mm_set1_epi32(127)), 23));
            let exp_neg_x = _mm_mul_ps(poly, exp_k);
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_div_ps(one, _mm_add_ps(one, exp_neg_x)),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = 1.0 / (1.0 + (-a[j]).exp());
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let len = a.len();
        let mut i = 0;
        let half = _mm_set1_ps(0.5);
        let one = _mm_set1_ps(1.0);
        let sqrt_2_pi = _mm_set1_ps(0.797_884_56);
        let coeff = _mm_set1_ps(0.044_715);
        while i + 4 <= len {
            let x = _mm_loadu_ps(a.as_ptr().add(i));
            let x3 = _mm_mul_ps(_mm_mul_ps(x, x), x);
            let inner = _mm_mul_ps(sqrt_2_pi, _mm_add_ps(x, _mm_mul_ps(coeff, x3)));
            // tanh approximation: (e^2x - 1) / (e^2x + 1)
            let two_inner = _mm_add_ps(inner, inner);
            let exp_2x = Self::exp_approx_sse2(two_inner);
            let tanh_val = _mm_div_ps(_mm_sub_ps(exp_2x, one), _mm_add_ps(exp_2x, one));
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_mul_ps(half, _mm_mul_ps(x, _mm_add_ps(one, tanh_val))),
            );
            i += 4;
        }
        for j in i..len {
            let x = a[j];
            result[j] = 0.5
                * x
                * (1.0 + ((0.797_884_56 * (x + 0.044_715 * x * x * x)) as f64).tanh() as f32);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        // swish(x) = x * sigmoid(x)
        let len = a.len();
        let mut i = 0;
        let one = _mm_set1_ps(1.0);
        let neg_one = _mm_set1_ps(-1.0);
        let ln2 = _mm_set1_ps(std::f32::consts::LN_2);
        let inv_ln2 = _mm_set1_ps(1.0 / std::f32::consts::LN_2);
        let c2 = _mm_set1_ps(0.5);
        let c3 = _mm_set1_ps(0.166_666_67);
        let c4 = _mm_set1_ps(0.041_666_668);
        let c5 = _mm_set1_ps(0.008_333_334);
        while i + 4 <= len {
            let x = _mm_loadu_ps(a.as_ptr().add(i));
            let neg_x = _mm_mul_ps(x, neg_one);
            let k = _mm_cvtps_epi32(_mm_mul_ps(neg_x, inv_ln2));
            let kf = _mm_cvtepi32_ps(k);
            let r = _mm_sub_ps(neg_x, _mm_mul_ps(kf, ln2));
            let mut poly = _mm_add_ps(one, _mm_mul_ps(r, c5));
            poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c4, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c3, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c2, _mm_mul_ps(r, poly))));
            poly = _mm_add_ps(one, _mm_mul_ps(r, poly));
            let exp_k = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(k, _mm_set1_epi32(127)), 23));
            let exp_neg_x = _mm_mul_ps(poly, exp_k);
            let sigmoid = _mm_div_ps(one, _mm_add_ps(one, exp_neg_x));
            _mm_storeu_ps(result.as_mut_ptr().add(i), _mm_mul_ps(x, sigmoid));
            i += 4;
        }
        for j in i..len {
            result[j] = a[j] / (1.0 + (-a[j]).exp());
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn tanh(a: &[f32], result: &mut [f32]) {
        // tanh(x) = (e^2x - 1) / (e^2x + 1)
        let len = a.len();
        let mut i = 0;
        let one = _mm_set1_ps(1.0);
        let two = _mm_set1_ps(2.0);
        while i + 4 <= len {
            let x = _mm_loadu_ps(a.as_ptr().add(i));
            let exp_2x = Self::exp_approx_sse2(_mm_mul_ps(two, x));
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_div_ps(_mm_sub_ps(exp_2x, one), _mm_add_ps(exp_2x, one)),
            );
            i += 4;
        }
        for j in i..len {
            let exp_2x = (2.0 * a[j]).exp();
            result[j] = (exp_2x - 1.0) / (exp_2x + 1.0);
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn sqrt(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        while i + 4 <= len {
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_sqrt_ps(_mm_loadu_ps(a.as_ptr().add(i))),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].sqrt();
        }
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn recip(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;
        let one = _mm_set1_ps(1.0);
        while i + 4 <= len {
            _mm_storeu_ps(
                result.as_mut_ptr().add(i),
                _mm_div_ps(one, _mm_loadu_ps(a.as_ptr().add(i))),
            );
            i += 4;
        }
        for j in i..len {
            result[j] = a[j].recip();
        }
    }

    unsafe fn ln(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::ln(a, result);
    }
    unsafe fn log2(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::log2(a, result);
    }
    unsafe fn log10(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::log10(a, result);
    }
    unsafe fn sin(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::sin(a, result);
    }
    unsafe fn cos(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::cos(a, result);
    }
    unsafe fn tan(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::tan(a, result);
    }
    unsafe fn floor(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::floor(a, result);
    }
    unsafe fn ceil(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::ceil(a, result);
    }
    unsafe fn round(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::round(a, result);
    }
}

impl Sse2Backend {
    /// SSE2 exp approximation helper
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn exp_approx_sse2(x: __m128) -> __m128 {
        let ln2 = _mm_set1_ps(std::f32::consts::LN_2);
        let inv_ln2 = _mm_set1_ps(1.0 / std::f32::consts::LN_2);
        let one = _mm_set1_ps(1.0);
        let c2 = _mm_set1_ps(0.5);
        let c3 = _mm_set1_ps(0.166_666_67);
        let c4 = _mm_set1_ps(0.041_666_668);
        let c5 = _mm_set1_ps(0.008_333_334);
        let k = _mm_cvtps_epi32(_mm_mul_ps(x, inv_ln2));
        let kf = _mm_cvtepi32_ps(k);
        let r = _mm_sub_ps(x, _mm_mul_ps(kf, ln2));
        let mut poly = _mm_add_ps(one, _mm_mul_ps(r, c5));
        poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c4, _mm_mul_ps(r, poly))));
        poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c3, _mm_mul_ps(r, poly))));
        poly = _mm_add_ps(one, _mm_mul_ps(r, _mm_add_ps(c2, _mm_mul_ps(r, poly))));
        poly = _mm_add_ps(one, _mm_mul_ps(r, poly));
        let exp_k = _mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(k, _mm_set1_epi32(127)), 23));
        _mm_mul_ps(poly, exp_k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse2_add() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0, 20.0, 30.0, 40.0, 50.0];
        let mut result = [0.0f32; 5];
        unsafe {
            Sse2Backend::add(&a, &b, &mut result);
        }
        assert_eq!(result, [11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn test_sse2_sub() {
        let a = [10.0, 20.0, 30.0, 40.0, 50.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut result = [0.0f32; 5];
        unsafe {
            Sse2Backend::sub(&a, &b, &mut result);
        }
        assert_eq!(result, [9.0, 18.0, 27.0, 36.0, 45.0]);
    }

    #[test]
    fn test_sse2_mul() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut result = [0.0f32; 5];
        unsafe {
            Sse2Backend::mul(&a, &b, &mut result);
        }
        assert_eq!(result, [2.0, 6.0, 12.0, 20.0, 30.0]);
    }

    #[test]
    fn test_sse2_div() {
        let a = [10.0, 20.0, 30.0, 40.0, 50.0];
        let b = [2.0, 4.0, 5.0, 8.0, 10.0];
        let mut result = [0.0f32; 5];
        unsafe {
            Sse2Backend::div(&a, &b, &mut result);
        }
        assert_eq!(result, [5.0, 5.0, 6.0, 5.0, 5.0]);
    }

    #[test]
    fn test_sse2_dot() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0];
        let result = unsafe { Sse2Backend::dot(&a, &b) };
        assert!((result - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_sse2_sum() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = unsafe { Sse2Backend::sum(&a) };
        assert!((result - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_sse2_max() {
        let a = [1.0, 5.0, 3.0, 2.0, 4.0];
        let result = unsafe { Sse2Backend::max(&a) };
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sse2_min() {
        let a = [5.0, 1.0, 3.0, 2.0, 4.0];
        let result = unsafe { Sse2Backend::min(&a) };
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sse2_argmax() {
        let a = [1.0, 5.0, 3.0, 2.0, 4.0];
        let result = unsafe { Sse2Backend::argmax(&a) };
        assert_eq!(result, 1);
    }

    #[test]
    fn test_sse2_argmin() {
        let a = [5.0, 1.0, 3.0, 2.0, 4.0];
        let result = unsafe { Sse2Backend::argmin(&a) };
        assert_eq!(result, 1);
    }

    #[test]
    fn test_sse2_norm_linf() {
        let a = [-5.0, 1.0, 3.0, 2.0, -4.0];
        let result = unsafe { Sse2Backend::norm_linf(&a) };
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sse2_scale() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut result = [0.0f32; 5];
        unsafe {
            Sse2Backend::scale(&a, 2.0, &mut result);
        }
        assert_eq!(result, [2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_sse2_abs() {
        let a = [-1.0, 2.0, -3.0, 4.0, -5.0];
        let mut result = [0.0f32; 5];
        unsafe {
            Sse2Backend::abs(&a, &mut result);
        }
        assert_eq!(result, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sse2_clamp() {
        let a = [-1.0, 0.5, 1.5, 2.0, 3.0];
        let mut result = [0.0f32; 5];
        unsafe {
            Sse2Backend::clamp(&a, 0.0, 1.0, &mut result);
        }
        assert_eq!(result, [0.0, 0.5, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sse2_relu() {
        let a = [-1.0, 0.0, 1.0, -2.0, 3.0];
        let mut result = [0.0f32; 5];
        unsafe {
            Sse2Backend::relu(&a, &mut result);
        }
        assert_eq!(result, [0.0, 0.0, 1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_sse2_exp() {
        let a = [0.0, 1.0, -1.0, 2.0];
        let mut result = [0.0f32; 4];
        unsafe {
            Sse2Backend::exp(&a, &mut result);
        }
        assert!((result[0] - 1.0).abs() < 0.05);
        assert!((result[1] - std::f32::consts::E).abs() < 0.1);
    }

    #[test]
    fn test_sse2_sigmoid() {
        let a = [0.0, 1.0, -1.0, 10.0];
        let mut result = [0.0f32; 4];
        unsafe {
            Sse2Backend::sigmoid(&a, &mut result);
        }
        assert!((result[0] - 0.5).abs() < 0.01);
        assert!(result[1] > 0.5);
        assert!(result[2] < 0.5);
    }

    #[test]
    fn test_sse2_sqrt() {
        let a = [1.0, 4.0, 9.0, 16.0, 25.0];
        let mut result = [0.0f32; 5];
        unsafe {
            Sse2Backend::sqrt(&a, &mut result);
        }
        assert_eq!(result, [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sse2_sum_kahan() {
        let a: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let result = unsafe { Sse2Backend::sum_kahan(&a) };
        assert!((result - 136.0).abs() < 1e-3);
    }

    #[test]
    fn test_sse2_norm_l2() {
        let a = vec![3.0, 4.0];
        let result = unsafe { Sse2Backend::norm_l2(&a) };
        assert!((result - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_sse2_norm_l1() {
        let a = vec![-1.0, 2.0, -3.0, 4.0];
        let result = unsafe { Sse2Backend::norm_l1(&a) };
        assert!((result - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_sse2_lerp() {
        let a = vec![0.0; 16];
        let b = vec![10.0; 16];
        let mut result = vec![0.0; 16];
        unsafe {
            Sse2Backend::lerp(&a, &b, 0.5, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 5.0).abs() < 1e-5));
    }

    #[test]
    fn test_sse2_fma() {
        let a = vec![2.0; 16];
        let b = vec![3.0; 16];
        let c = vec![1.0; 16];
        let mut result = vec![0.0; 16];
        unsafe {
            Sse2Backend::fma(&a, &b, &c, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 7.0).abs() < 1e-5));
    }

    #[test]
    fn test_sse2_gelu() {
        let a = vec![0.0, 1.0];
        let mut result = vec![0.0; 2];
        unsafe {
            Sse2Backend::gelu(&a, &mut result);
        }
        assert!((result[0]).abs() < 1e-5);
        assert!((result[1] - 0.841_192).abs() < 1e-2);
    }

    #[test]
    fn test_sse2_swish() {
        let a = vec![0.0, 1.0];
        let mut result = vec![0.0; 2];
        unsafe {
            Sse2Backend::swish(&a, &mut result);
        }
        assert!((result[0]).abs() < 1e-5);
        assert!((result[1] - 0.731_059).abs() < 1e-2);
    }

    #[test]
    fn test_sse2_tanh() {
        let a = vec![0.0, 1.0];
        let mut result = vec![0.0; 2];
        unsafe {
            Sse2Backend::tanh(&a, &mut result);
        }
        assert!((result[0]).abs() < 1e-5);
        assert!((result[1] - 0.761_594_2).abs() < 1e-2);
    }

    #[test]
    fn test_sse2_recip() {
        let a = vec![2.0, 4.0, 5.0];
        let mut result = vec![0.0; 3];
        unsafe {
            Sse2Backend::recip(&a, &mut result);
        }
        assert!((result[0] - 0.5).abs() < 1e-5);
        assert!((result[1] - 0.25).abs() < 1e-5);
        assert!((result[2] - 0.2).abs() < 1e-5);
    }

    #[test]
    fn test_sse2_transcendental() {
        let a = vec![1.0, std::f32::consts::E, 10.0];
        let mut ln_result = vec![0.0; 3];
        let mut log10_result = vec![0.0; 3];
        unsafe {
            Sse2Backend::ln(&a, &mut ln_result);
            Sse2Backend::log10(&a, &mut log10_result);
        }
        assert!((ln_result[0]).abs() < 1e-5);
        assert!((ln_result[1] - 1.0).abs() < 1e-4);
        assert!((log10_result[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sse2_trig() {
        let a = vec![0.0, std::f32::consts::FRAC_PI_2];
        let mut sin_result = vec![0.0; 2];
        let mut cos_result = vec![0.0; 2];
        unsafe {
            Sse2Backend::sin(&a, &mut sin_result);
            Sse2Backend::cos(&a, &mut cos_result);
        }
        assert!((sin_result[0]).abs() < 1e-5);
        assert!((sin_result[1] - 1.0).abs() < 1e-5);
        assert!((cos_result[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sse2_rounding() {
        let a = vec![1.3, 1.5, 1.7, -1.3, -1.5, -1.7];
        let mut floor_result = vec![0.0; 6];
        let mut ceil_result = vec![0.0; 6];
        unsafe {
            Sse2Backend::floor(&a, &mut floor_result);
            Sse2Backend::ceil(&a, &mut ceil_result);
        }
        assert_eq!(floor_result, vec![1.0, 1.0, 1.0, -2.0, -2.0, -2.0]);
        assert_eq!(ceil_result, vec![2.0, 2.0, 2.0, -1.0, -1.0, -1.0]);
    }
}
