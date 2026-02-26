//! AVX2 arithmetic operations (add, sub, mul, div).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2 vector addition.
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vresult = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
        i += 8;
    }

    for j in i..len {
        result[j] = a[j] + b[j];
    }
}

/// AVX2 vector subtraction.
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vresult = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
        i += 8;
    }

    for j in i..len {
        result[j] = a[j] - b[j];
    }
}

/// AVX2 vector multiplication.
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vresult = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
        i += 8;
    }

    for j in i..len {
        result[j] = a[j] * b[j];
    }
}

/// AVX2 vector division.
#[inline]
#[target_feature(enable = "avx2")]
// SAFETY: caller ensures preconditions are met for this unsafe function
pub unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let mut i = 0;

    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vresult = _mm256_div_ps(va, vb);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
        i += 8;
    }

    for j in i..len {
        result[j] = a[j] / b[j];
    }
}
