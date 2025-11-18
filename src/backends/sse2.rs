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

    #[target_feature(enable = "sse2")]
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast t to all 4 lanes
        let t_vec = _mm_set1_ps(t);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));

            // result = a + t * (b - a)
            let diff = _mm_sub_ps(vb, va);
            let scaled_diff = _mm_mul_ps(t_vec, diff);
            let vresult = _mm_add_ps(va, scaled_diff);

            _mm_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] + t * (b[i] - a[i]);
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let vc = _mm_loadu_ps(c.as_ptr().add(i));

            // result = a * b + c
            // SSE2 doesn't have FMA, so we use separate mul and add
            let product = _mm_mul_ps(va, vb);
            let vresult = _mm_add_ps(product, vc);

            _mm_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] * b[i] + c[i];
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Zero vector for max comparison
        let zero = _mm_setzero_ps();

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = _mm_loadu_ps(a.as_ptr().add(i));

            // ReLU: max(0, x)
            let vresult = _mm_max_ps(zero, va);

            _mm_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
            i += 1;
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        // SSE2 doesn't have native exp(), use scalar with numerical stability
        // Future optimization: implement fast exp approximation
        for (i, &val) in a.iter().enumerate() {
            result[i] = if val < -50.0 {
                0.0
            } else if val > 50.0 {
                1.0
            } else {
                1.0 / (1.0 + (-val).exp())
            };
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        // SSE2 doesn't have native tanh(), use scalar
        // Future optimization: implement fast tanh approximation
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        const COEFF: f32 = 0.044715;

        for (i, &x) in a.iter().enumerate() {
            let x3 = x * x * x;
            let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            result[i] = 0.5 * x * (1.0 + inner.tanh());
        }
    }

    #[target_feature(enable = "sse2")]
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        // SSE2 doesn't have native exp(), use scalar
        for (i, &x) in a.iter().enumerate() {
            if x < -50.0 {
                result[i] = 0.0;
            } else if x > 50.0 {
                result[i] = x;
            } else {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                result[i] = x * sigmoid;
            }
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

    #[test]
    fn test_sse2_relu() {
        let a = [-3.0, -1.0, 0.0, 1.0, 3.0, -2.0, 2.0, -0.5];
        let mut result = [0.0; 8];
        unsafe {
            Sse2Backend::relu(&a, &mut result);
        }
        assert_eq!(result, [0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_sse2_relu_matches_scalar() {
        // Verify SSE2 relu produces same results as scalar
        let a = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0];

        let mut scalar_result = [0.0; 7];
        let mut sse2_result = [0.0; 7];

        unsafe {
            super::super::scalar::ScalarBackend::relu(&a, &mut scalar_result);
            Sse2Backend::relu(&a, &mut sse2_result);
        }

        assert_eq!(scalar_result, sse2_result);
    }

    #[test]
    fn test_sse2_sigmoid_matches_scalar() {
        // Verify SSE2 sigmoid produces same results as scalar
        let a = [-10.0, -1.0, 0.0, 1.0, 10.0];

        let mut scalar_result = [0.0; 5];
        let mut sse2_result = [0.0; 5];

        unsafe {
            super::super::scalar::ScalarBackend::sigmoid(&a, &mut scalar_result);
            Sse2Backend::sigmoid(&a, &mut sse2_result);
        }

        for (s, e) in scalar_result.iter().zip(sse2_result.iter()) {
            assert!(
                (s - e).abs() < 1e-6,
                "sigmoid mismatch: scalar={}, sse2={}",
                s,
                e
            );
        }
    }

    #[test]
    fn test_sse2_gelu_matches_scalar() {
        // Verify SSE2 gelu produces same results as scalar
        let a = [-2.0, -1.0, 0.0, 1.0, 2.0];

        let mut scalar_result = [0.0; 5];
        let mut sse2_result = [0.0; 5];

        unsafe {
            super::super::scalar::ScalarBackend::gelu(&a, &mut scalar_result);
            Sse2Backend::gelu(&a, &mut sse2_result);
        }

        for (s, e) in scalar_result.iter().zip(sse2_result.iter()) {
            assert!(
                (s - e).abs() < 1e-5,
                "gelu mismatch: scalar={}, sse2={}",
                s,
                e
            );
        }
    }

    #[test]
    fn test_sse2_swish_matches_scalar() {
        // Verify SSE2 swish produces same results as scalar
        let a = [-10.0, -1.0, 0.0, 1.0, 10.0];

        let mut scalar_result = [0.0; 5];
        let mut sse2_result = [0.0; 5];

        unsafe {
            super::super::scalar::ScalarBackend::swish(&a, &mut scalar_result);
            Sse2Backend::swish(&a, &mut sse2_result);
        }

        for (s, e) in scalar_result.iter().zip(sse2_result.iter()) {
            assert!(
                (s - e).abs() < 1e-5,
                "swish mismatch: scalar={}, sse2={}",
                s,
                e
            );
        }
    }

    #[test]
    fn test_sse2_sub_matches_scalar() {
        let a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let mut scalar_result = [0.0; 7];
        let mut sse2_result = [0.0; 7];

        unsafe {
            super::super::scalar::ScalarBackend::sub(&a, &b, &mut scalar_result);
            Sse2Backend::sub(&a, &b, &mut sse2_result);
        }

        assert_eq!(scalar_result, sse2_result);
    }

    #[test]
    fn test_sse2_div_matches_scalar() {
        let a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
        let b = [2.0, 4.0, 5.0, 8.0, 10.0, 12.0, 14.0];

        let mut scalar_result = [0.0; 7];
        let mut sse2_result = [0.0; 7];

        unsafe {
            super::super::scalar::ScalarBackend::div(&a, &b, &mut scalar_result);
            Sse2Backend::div(&a, &b, &mut sse2_result);
        }

        assert_eq!(scalar_result, sse2_result);
    }

    #[test]
    fn test_sse2_scale_matches_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let scalar = 2.5;

        let mut scalar_result = [0.0; 7];
        let mut sse2_result = [0.0; 7];

        unsafe {
            super::super::scalar::ScalarBackend::scale(&a, scalar, &mut scalar_result);
            Sse2Backend::scale(&a, scalar, &mut sse2_result);
        }

        assert_eq!(scalar_result, sse2_result);
    }

    #[test]
    fn test_sse2_clamp_matches_scalar() {
        let a = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0];

        let mut scalar_result = [0.0; 7];
        let mut sse2_result = [0.0; 7];

        unsafe {
            super::super::scalar::ScalarBackend::clamp(&a, 5.0, 20.0, &mut scalar_result);
            Sse2Backend::clamp(&a, 5.0, 20.0, &mut sse2_result);
        }

        assert_eq!(scalar_result, sse2_result);
    }

    #[test]
    fn test_sse2_fma_matches_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let c = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];

        let mut scalar_result = [0.0; 7];
        let mut sse2_result = [0.0; 7];

        unsafe {
            super::super::scalar::ScalarBackend::fma(&a, &b, &c, &mut scalar_result);
            Sse2Backend::fma(&a, &b, &c, &mut sse2_result);
        }

        assert_eq!(scalar_result, sse2_result);
    }

    #[test]
    fn test_sse2_lerp_matches_scalar() {
        let a = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let b = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0];

        let mut scalar_result = [0.0; 7];
        let mut sse2_result = [0.0; 7];

        unsafe {
            super::super::scalar::ScalarBackend::lerp(&a, &b, 0.25, &mut scalar_result);
            Sse2Backend::lerp(&a, &b, 0.25, &mut sse2_result);
        }

        for (s, e) in scalar_result.iter().zip(sse2_result.iter()) {
            assert!(
                (s - e).abs() < 1e-5,
                "lerp mismatch: scalar={}, sse2={}",
                s,
                e
            );
        }
    }

    #[test]
    fn test_sse2_argmax_matches_scalar() {
        let a = [1.0, 5.0, 3.0, 10.0, 2.0, 8.0, 4.0];

        let scalar_result = unsafe { super::super::scalar::ScalarBackend::argmax(&a) };
        let sse2_result = unsafe { Sse2Backend::argmax(&a) };

        assert_eq!(scalar_result, sse2_result);
    }

    #[test]
    fn test_sse2_argmin_matches_scalar() {
        let a = [5.0, 1.0, 3.0, 10.0, 2.0, 8.0, 4.0];

        let scalar_result = unsafe { super::super::scalar::ScalarBackend::argmin(&a) };
        let sse2_result = unsafe { Sse2Backend::argmin(&a) };

        assert_eq!(scalar_result, sse2_result);
    }

    #[test]
    fn test_sse2_sum_kahan_matches_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let scalar_result = unsafe { super::super::scalar::ScalarBackend::sum_kahan(&a) };
        let sse2_result = unsafe { Sse2Backend::sum_kahan(&a) };

        assert!((scalar_result - sse2_result).abs() < 1e-5);
    }

    #[test]
    fn test_sse2_norm_l1_matches_scalar() {
        let a = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0];

        let scalar_result = unsafe { super::super::scalar::ScalarBackend::norm_l1(&a) };
        let sse2_result = unsafe { Sse2Backend::norm_l1(&a) };

        assert!((scalar_result - sse2_result).abs() < 1e-5);
    }

    #[test]
    fn test_sse2_norm_l2_matches_scalar() {
        let a = [3.0, 4.0, 0.0, 0.0, 5.0, 12.0, 0.0];

        let scalar_result = unsafe { super::super::scalar::ScalarBackend::norm_l2(&a) };
        let sse2_result = unsafe { Sse2Backend::norm_l2(&a) };

        assert!((scalar_result - sse2_result).abs() < 1e-5);
    }

    #[test]
    fn test_sse2_dot_matches_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let scalar_result = unsafe { super::super::scalar::ScalarBackend::dot(&a, &b) };
        let sse2_result = unsafe { Sse2Backend::dot(&a, &b) };

        assert!((scalar_result - sse2_result).abs() < 1e-5);
    }

    #[test]
    fn test_sse2_mul_matches_scalar() {
        let a = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut scalar_result = [0.0; 7];
        let mut sse2_result = [0.0; 7];

        unsafe {
            super::super::scalar::ScalarBackend::mul(&a, &b, &mut scalar_result);
            Sse2Backend::mul(&a, &b, &mut sse2_result);
        }

        assert_eq!(scalar_result, sse2_result);
    }

    #[test]
    fn test_sse2_add_matches_scalar() {
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

    #[test]
    fn test_sse2_sum_matches_scalar() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        let scalar_result = unsafe { super::super::scalar::ScalarBackend::sum(&a) };
        let sse2_result = unsafe { Sse2Backend::sum(&a) };

        assert!((scalar_result - sse2_result).abs() < 1e-5);
    }

    #[test]
    fn test_sse2_max_matches_scalar() {
        let a = [1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0];

        let scalar_result = unsafe { super::super::scalar::ScalarBackend::max(&a) };
        let sse2_result = unsafe { Sse2Backend::max(&a) };

        assert_eq!(scalar_result, sse2_result);
    }

    #[test]
    fn test_sse2_min_matches_scalar() {
        let a = [5.0, 1.0, 3.0, 7.0, 2.0, 8.0, 4.0];

        let scalar_result = unsafe { super::super::scalar::ScalarBackend::min(&a) };
        let sse2_result = unsafe { Sse2Backend::min(&a) };

        assert_eq!(scalar_result, sse2_result);
    }
}
