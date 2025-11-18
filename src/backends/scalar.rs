//! Scalar (non-SIMD) backend implementation
//!
//! This is the portable baseline implementation that works on all platforms.
//! It uses simple loops without any SIMD instructions.
//!
//! # Performance
//!
//! This backend provides correctness reference but no SIMD acceleration.
//! Expected to be 8-32x slower than SIMD backends on operations with 1K+ elements.

use super::VectorBackend;

/// Scalar backend (portable, no SIMD)
pub struct ScalarBackend;

impl VectorBackend for ScalarBackend {
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] - b[i];
        }
    }

    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }

    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] / b[i];
        }
    }

    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    unsafe fn sum(a: &[f32]) -> f32 {
        let mut total = 0.0;
        for &val in a {
            total += val;
        }
        total
    }

    unsafe fn max(a: &[f32]) -> f32 {
        let mut maximum = a[0];
        for &val in &a[1..] {
            if val > maximum {
                maximum = val;
            }
        }
        maximum
    }

    unsafe fn min(a: &[f32]) -> f32 {
        let mut minimum = a[0];
        for &val in &a[1..] {
            if val < minimum {
                minimum = val;
            }
        }
        minimum
    }

    unsafe fn argmax(a: &[f32]) -> usize {
        let mut max_value = a[0];
        let mut max_index = 0;
        for (i, &val) in a.iter().enumerate() {
            if val > max_value {
                max_value = val;
                max_index = i;
            }
        }
        max_index
    }

    unsafe fn argmin(a: &[f32]) -> usize {
        let mut min_value = a[0];
        let mut min_index = 0;
        for (i, &val) in a.iter().enumerate() {
            if val < min_value {
                min_value = val;
                min_index = i;
            }
        }
        min_index
    }

    unsafe fn sum_kahan(a: &[f32]) -> f32 {
        let mut sum = 0.0;
        let mut c = 0.0; // Compensation for lost low-order bits

        for &value in a {
            let y = value - c; // Subtract the compensation
            let t = sum + y; // Add to sum
            c = (t - sum) - y; // Update compensation
            sum = t; // Update sum
        }

        sum
    }

    unsafe fn norm_l2(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        let mut sum_of_squares = 0.0;
        for &val in a {
            sum_of_squares += val * val;
        }
        sum_of_squares.sqrt()
    }

    unsafe fn norm_l1(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        let mut sum = 0.0;
        for &val in a {
            sum += val.abs();
        }
        sum
    }

    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val * scalar;
        }
    }

    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.max(min_val).min(max_val);
        }
    }

    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        for (i, (&a_val, &b_val)) in a.iter().zip(b.iter()).enumerate() {
            // result = a + t * (b - a)
            result[i] = a_val + t * (b_val - a_val);
        }
    }

    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        for (i, ((&a_val, &b_val), &c_val)) in a.iter().zip(b.iter()).zip(c.iter()).enumerate() {
            // result = a * b + c
            result[i] = a_val * b_val + c_val;
        }
    }

    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = if val > 0.0 { val } else { 0.0 };
        }
    }

    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            // Handle extreme values for numerical stability
            result[i] = if val < -50.0 {
                0.0 // exp(-x) would overflow, but sigmoid approaches 0
            } else if val > 50.0 {
                1.0 // exp(-x) underflows to 0, sigmoid approaches 1
            } else {
                1.0 / (1.0 + (-val).exp())
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_add() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let mut result = [0.0; 4];
        unsafe {
            ScalarBackend::add(&a, &b, &mut result);
        }
        assert_eq!(result, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_scalar_mul() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 3.0, 4.0, 5.0];
        let mut result = [0.0; 4];
        unsafe {
            ScalarBackend::mul(&a, &b, &mut result);
        }
        assert_eq!(result, [2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_scalar_dot() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = unsafe { ScalarBackend::dot(&a, &b) };
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_scalar_sum() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let result = unsafe { ScalarBackend::sum(&a) };
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_scalar_max() {
        let a = [1.0, 5.0, 3.0, 2.0];
        let result = unsafe { ScalarBackend::max(&a) };
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_scalar_min() {
        let a = [1.0, 5.0, 3.0, 2.0];
        let result = unsafe { ScalarBackend::min(&a) };
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_scalar_sub() {
        let a = [5.0, 6.0, 7.0, 8.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut result = [0.0; 4];
        unsafe {
            ScalarBackend::sub(&a, &b, &mut result);
        }
        assert_eq!(result, [4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_scalar_div() {
        let a = [10.0, 20.0, 30.0, 40.0];
        let b = [2.0, 4.0, 5.0, 8.0];
        let mut result = [0.0; 4];
        unsafe {
            ScalarBackend::div(&a, &b, &mut result);
        }
        assert_eq!(result, [5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_scalar_argmax() {
        let a = [1.0, 5.0, 3.0, 2.0];
        let result = unsafe { ScalarBackend::argmax(&a) };
        assert_eq!(result, 1); // Index of 5.0
    }

    #[test]
    fn test_scalar_argmin() {
        let a = [5.0, 1.0, 3.0, 2.0];
        let result = unsafe { ScalarBackend::argmin(&a) };
        assert_eq!(result, 1); // Index of 1.0
    }

    #[test]
    fn test_scalar_sum_kahan() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let result = unsafe { ScalarBackend::sum_kahan(&a) };
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_scalar_norm_l1() {
        let a = [1.0, -2.0, 3.0, -4.0];
        let result = unsafe { ScalarBackend::norm_l1(&a) };
        assert_eq!(result, 10.0); // |1| + |-2| + |3| + |-4| = 10
    }

    #[test]
    fn test_scalar_norm_l2() {
        let a = [3.0, 4.0];
        let result = unsafe { ScalarBackend::norm_l2(&a) };
        assert_eq!(result, 5.0); // sqrt(3² + 4²) = 5
    }

    #[test]
    fn test_scalar_scale() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let mut result = [0.0; 4];
        unsafe {
            ScalarBackend::scale(&a, 2.0, &mut result);
        }
        assert_eq!(result, [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_scalar_clamp() {
        let a = [1.0, 5.0, 10.0, 15.0];
        let mut result = [0.0; 4];
        unsafe {
            ScalarBackend::clamp(&a, 3.0, 12.0, &mut result);
        }
        assert_eq!(result, [3.0, 5.0, 10.0, 12.0]);
    }

    #[test]
    fn test_scalar_lerp() {
        let a = [0.0, 10.0, 20.0];
        let b = [100.0, 110.0, 120.0];
        let mut result = [0.0; 3];
        unsafe {
            ScalarBackend::lerp(&a, &b, 0.5, &mut result);
        }
        assert_eq!(result, [50.0, 60.0, 70.0]); // Midpoint between a and b
    }

    #[test]
    fn test_scalar_fma() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 3.0, 4.0];
        let c = [5.0, 6.0, 7.0];
        let mut result = [0.0; 3];
        unsafe {
            ScalarBackend::fma(&a, &b, &c, &mut result);
        }
        // FMA: a*b + c
        assert_eq!(result, [7.0, 12.0, 19.0]); // [1*2+5, 2*3+6, 3*4+7]
    }

    #[test]
    fn test_scalar_relu() {
        let a = [-3.0, -1.0, 0.0, 1.0, 3.0];
        let mut result = [0.0; 5];
        unsafe {
            ScalarBackend::relu(&a, &mut result);
        }
        assert_eq!(result, [0.0, 0.0, 0.0, 1.0, 3.0]);
    }
}
