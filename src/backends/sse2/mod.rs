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
mod tests;
