//! AVX-512 arithmetic operations (add, sub, mul, div).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX-512 vector addition.
#[inline]
#[target_feature(enable = "avx512f")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_add_ps(va, vb));
            i += 16;
        }
        for j in i..len {
            result[j] = a[j] + b[j];
        }
    }
}

/// AVX-512 vector subtraction.
#[inline]
#[target_feature(enable = "avx512f")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_sub_ps(va, vb));
            i += 16;
        }
        for j in i..len {
            result[j] = a[j] - b[j];
        }
    }
}

/// AVX-512 vector multiplication.
#[inline]
#[target_feature(enable = "avx512f")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_mul_ps(va, vb));
            i += 16;
        }
        for j in i..len {
            result[j] = a[j] * b[j];
        }
    }
}

/// AVX-512 vector division.
#[inline]
#[target_feature(enable = "avx512f")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub(crate) unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
    unsafe {
        let len = a.len();
        let mut i = 0;
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            _mm512_storeu_ps(result.as_mut_ptr().add(i), _mm512_div_ps(va, vb));
            i += 16;
        }
        for j in i..len {
            result[j] = a[j] / b[j];
        }
    }
}
