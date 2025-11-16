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
            let y = value - c;  // Subtract the compensation
            let t = sum + y;    // Add to sum
            c = (t - sum) - y;  // Update compensation
            sum = t;            // Update sum
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
}
