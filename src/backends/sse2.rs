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

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::VectorBackend;

/// SSE2 backend (128-bit SIMD for x86_64)
pub struct Sse2Backend;

impl VectorBackend for Sse2Backend {
    #[target_feature(enable = "sse2")]
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time using SSE2 (128-bit = 4 x f32)
        while i + 4 <= len {
            // Load 4 floats from a and b
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));

            // Add them
            let vresult = _mm_add_ps(va, vb);

            // Store result
            _mm_storeu_ps(result.as_mut_ptr().add(i), vresult);

            i += 4;
        }

        // Handle remaining elements with scalar code
        for j in i..len {
            result[j] = a[j] + b[j];
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time using SSE2 (128-bit = 4 x f32)
        while i + 4 <= len {
            // Load 4 floats from a and b
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));

            // Subtract them
            let vresult = _mm_sub_ps(va, vb);

            // Store result
            _mm_storeu_ps(result.as_mut_ptr().add(i), vresult);

            i += 4;
        }

        // Handle remaining elements with scalar code
        for j in i..len {
            result[j] = a[j] - b[j];
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let vresult = _mm_mul_ps(va, vb);
            _mm_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Handle remaining elements
        for j in i..len {
            result[j] = a[j] * b[j];
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let vresult = _mm_div_ps(va, vb);
            _mm_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Handle remaining elements
        for j in i..len {
            result[j] = a[j] / b[j];
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Accumulator for SIMD portion
        let mut sum_vec = _mm_setzero_ps();

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let vmul = _mm_mul_ps(va, vb);
            sum_vec = _mm_add_ps(sum_vec, vmul);
            i += 4;
        }

        // Horizontal sum of the SIMD accumulator
        // sum_vec contains [a, b, c, d]
        // We need to compute a + b + c + d
        let mut sum_array = [0.0f32; 4];
        _mm_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
        let mut sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

        // Handle remaining elements with scalar code
        for j in i..len {
            sum += a[j] * b[j];
        }

        sum
    }

    #[target_feature(enable = "sse2")]
    unsafe fn sum(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;
        let mut sum_vec = _mm_setzero_ps();

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            sum_vec = _mm_add_ps(sum_vec, va);
            i += 4;
        }

        // Horizontal sum
        let mut sum_array = [0.0f32; 4];
        _mm_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
        let mut sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

        // Handle remaining elements
        sum += a[i..len].iter().sum::<f32>();

        sum
    }

    #[target_feature(enable = "sse2")]
    unsafe fn max(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Initialize with first element broadcast to all lanes
        let mut max_vec = _mm_set1_ps(a[0]);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            max_vec = _mm_max_ps(max_vec, va);
            i += 4;
        }

        // Extract maximum from SIMD register
        let mut max_array = [0.0f32; 4];
        _mm_storeu_ps(max_array.as_mut_ptr(), max_vec);
        let mut maximum = max_array[0]
            .max(max_array[1])
            .max(max_array[2])
            .max(max_array[3]);

        // Handle remaining elements
        for &val in &a[i..len] {
            if val > maximum {
                maximum = val;
            }
        }

        maximum
    }

    #[target_feature(enable = "sse2")]
    unsafe fn min(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Initialize with first element broadcast to all lanes
        let mut min_vec = _mm_set1_ps(a[0]);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            min_vec = _mm_min_ps(min_vec, va);
            i += 4;
        }

        // Extract minimum from SIMD register
        let mut min_array = [0.0f32; 4];
        _mm_storeu_ps(min_array.as_mut_ptr(), min_vec);
        let mut minimum = min_array[0]
            .min(min_array[1])
            .min(min_array[2])
            .min(min_array[3]);

        // Handle remaining elements
        for &val in &a[i..len] {
            if val < minimum {
                minimum = val;
            }
        }

        minimum
    }

    #[target_feature(enable = "sse2")]
    unsafe fn argmax(a: &[f32]) -> usize {
        let len = a.len();
        let mut i = 0;

        // Track maximum value and index
        let mut max_value = a[0];
        let mut max_index = 0;

        // Initialize with first element broadcast to all lanes
        let mut max_vec = _mm_set1_ps(a[0]);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            max_vec = _mm_max_ps(max_vec, va);
            i += 4;
        }

        // Extract maximum from SIMD register (for potential future optimization)
        let mut max_array = [0.0f32; 4];
        _mm_storeu_ps(max_array.as_mut_ptr(), max_vec);

        // Find the index by checking all elements processed by SIMD
        for (idx, &val) in a[..i].iter().enumerate() {
            if val > max_value {
                max_value = val;
                max_index = idx;
            }
        }

        // Handle remaining elements
        for (idx, &val) in a[i..].iter().enumerate() {
            if val > max_value {
                max_value = val;
                max_index = i + idx;
            }
        }

        max_index
    }

    #[target_feature(enable = "sse2")]
    unsafe fn argmin(a: &[f32]) -> usize {
        let len = a.len();
        let mut i = 0;

        // Track minimum value and index
        let mut min_value = a[0];
        let mut min_index = 0;

        // Initialize with first element broadcast to all lanes
        let mut min_vec = _mm_set1_ps(a[0]);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            min_vec = _mm_min_ps(min_vec, va);
            i += 4;
        }

        // Extract minimum from SIMD register (for potential future optimization)
        let mut min_array = [0.0f32; 4];
        _mm_storeu_ps(min_array.as_mut_ptr(), min_vec);

        // Find the index by checking all elements processed by SIMD
        for (idx, &val) in a[..i].iter().enumerate() {
            if val < min_value {
                min_value = val;
                min_index = idx;
            }
        }

        // Handle remaining elements
        for (idx, &val) in a[i..].iter().enumerate() {
            if val < min_value {
                min_value = val;
                min_index = i + idx;
            }
        }

        min_index
    }

    unsafe fn sum_kahan(a: &[f32]) -> f32 {
        // Kahan summation is inherently sequential, use scalar implementation
        super::scalar::ScalarBackend::sum_kahan(a)
    }

    #[target_feature(enable = "sse2")]
    unsafe fn norm_l2(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        // L2 norm is sqrt(dot(a, a))
        let sum_of_squares = Self::dot(a, a);
        sum_of_squares.sqrt()
    }

    #[target_feature(enable = "sse2")]
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        let len = a.len();
        let mut i = 0;

        // Accumulator for 4-way parallel accumulation
        let mut acc = _mm_setzero_ps();

        // SSE2 doesn't have abs for floats, use bitwise AND to clear sign bit
        let sign_mask = _mm_set1_ps(f32::from_bits(0x7FFF_FFFF));

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));

            // Compute absolute value by clearing sign bit
            let abs_va = _mm_and_ps(va, sign_mask);

            // Accumulate
            acc = _mm_add_ps(acc, abs_va);

            i += 4;
        }

        // Horizontal sum: extract all 4 lanes and sum them
        let mut result = {
            let temp = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
            let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };

        // Handle remaining elements with scalar code
        for &val in &a[i..] {
            result += val.abs();
        }

        result
    }

    #[target_feature(enable = "sse2")]
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast scalar to all 4 lanes
        let scalar_vec = _mm_set1_ps(scalar);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vresult = _mm_mul_ps(va, scalar_vec);
            _mm_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] * scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast min and max to all 4 lanes
        let min_vec = _mm_set1_ps(min_val);
        let max_vec = _mm_set1_ps(max_val);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let clamped = _mm_min_ps(_mm_max_ps(va, min_vec), max_vec);
            _mm_storeu_ps(result.as_mut_ptr().add(i), clamped);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i].max(min_val).min(max_val);
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse2_add() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0, 6.0, 7.0, 8.0, 9.0];
        let mut result = [0.0; 5];

        unsafe {
            Sse2Backend::add(&a, &b, &mut result);
        }

        assert_eq!(result, [6.0, 8.0, 10.0, 12.0, 14.0]);
    }

    #[test]
    fn test_sse2_mul() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut result = [0.0; 5];

        unsafe {
            Sse2Backend::mul(&a, &b, &mut result);
        }

        assert_eq!(result, [2.0, 6.0, 12.0, 20.0, 30.0]);
    }

    #[test]
    fn test_sse2_dot() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 5.0, 6.0, 7.0];

        let result = unsafe { Sse2Backend::dot(&a, &b) };

        assert_eq!(result, 60.0); // 1*4 + 2*5 + 3*6 + 4*7 = 60
    }

    #[test]
    fn test_sse2_sum() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = unsafe { Sse2Backend::sum(&a) };
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_sse2_max() {
        let a = [1.0, 5.0, 3.0, 2.0, 4.0];
        let result = unsafe { Sse2Backend::max(&a) };
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_sse2_min() {
        let a = [1.0, 5.0, 3.0, 2.0, 4.0];
        let result = unsafe { Sse2Backend::min(&a) };
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_sse2_matches_scalar() {
        // Verify SSE2 produces same results as scalar
        let a = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
        let b = [8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5];

        let mut scalar_result = [0.0; 7];
        let mut sse2_result = [0.0; 7];

        unsafe {
            super::super::scalar::ScalarBackend::add(&a, &b, &mut scalar_result);
            Sse2Backend::add(&a, &b, &mut sse2_result);
        }

        assert_eq!(scalar_result, sse2_result);
    }
}
