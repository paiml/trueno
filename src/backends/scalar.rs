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

    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
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
