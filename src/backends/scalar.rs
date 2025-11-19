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
    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/indexing
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/indexing
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] - b[i];
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/indexing
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/indexing
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] / b[i];
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/indexing
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = 0.0;
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn sum(a: &[f32]) -> f32 {
        let mut total = 0.0;
        for &val in a {
            total += val;
        }
        total
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust slicing/iteration
    // 2. Caller must ensure slice is non-empty (a[0] access)
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn max(a: &[f32]) -> f32 {
        let mut maximum = a[0];
        for &val in &a[1..] {
            if val > maximum {
                maximum = val;
            }
        }
        maximum
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust slicing/iteration
    // 2. Caller must ensure slice is non-empty (a[0] access)
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn min(a: &[f32]) -> f32 {
        let mut minimum = a[0];
        for &val in &a[1..] {
            if val < minimum {
                minimum = val;
            }
        }
        minimum
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator
    // 2. Caller must ensure slice is non-empty (a[0] access)
    // 3. Marked unsafe only to match VectorBackend trait interface
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

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator
    // 2. Caller must ensure slice is non-empty (a[0] access)
    // 3. Marked unsafe only to match VectorBackend trait interface
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

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator
    // 2. Kahan summation uses only safe floating-point arithmetic
    // 3. Marked unsafe only to match VectorBackend trait interface
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

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator
    // 2. Empty check prevents undefined behavior
    // 3. Marked unsafe only to match VectorBackend trait interface
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

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator
    // 2. Empty check prevents undefined behavior
    // 3. Marked unsafe only to match VectorBackend trait interface
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

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator
    // 2. Empty check prevents undefined behavior
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn norm_linf(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        let mut max_val = 0.0_f32;
        for &val in a {
            let abs_val = val.abs();
            if abs_val > max_val {
                max_val = abs_val;
            }
        }
        max_val
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val * scalar;
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn abs(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.abs();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.max(min_val).min(max_val);
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate/zip
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        for (i, (&a_val, &b_val)) in a.iter().zip(b.iter()).enumerate() {
            // result = a + t * (b - a)
            result[i] = a_val + t * (b_val - a_val);
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate/zip
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        for (i, ((&a_val, &b_val), &c_val)) in a.iter().zip(b.iter()).zip(c.iter()).enumerate() {
            // result = a * b + c
            result[i] = a_val * b_val + c_val;
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = if val > 0.0 { val } else { 0.0 };
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn exp(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.exp();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. Clamping prevents exp() overflow
    // 3. Marked unsafe only to match VectorBackend trait interface
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

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        const COEFF: f32 = 0.044715;

        for (i, &x) in a.iter().enumerate() {
            let x3 = x * x * x;
            let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            result[i] = 0.5 * x * (1.0 + inner.tanh());
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. Clamping prevents exp() overflow
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        // Swish: x * sigmoid(x) = x / (1 + exp(-x))
        for (i, &x) in a.iter().enumerate() {
            if x < -50.0 {
                result[i] = 0.0; // x * 0 = 0
            } else if x > 50.0 {
                result[i] = x; // x * 1 = x
            } else {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                result[i] = x * sigmoid;
            }
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn tanh(a: &[f32], result: &mut [f32]) {
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        for (i, &x) in a.iter().enumerate() {
            result[i] = x.tanh();
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
        // SAFETY: Test code calling backend trait methods marked unsafe
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
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            ScalarBackend::mul(&a, &b, &mut result);
        }
        assert_eq!(result, [2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    fn test_scalar_dot() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // SAFETY: Test code calling backend trait methods marked unsafe
        let result = unsafe { ScalarBackend::dot(&a, &b) };
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_scalar_sum() {
        let a = [1.0, 2.0, 3.0, 4.0];
        // SAFETY: Test code calling backend trait methods marked unsafe
        let result = unsafe { ScalarBackend::sum(&a) };
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_scalar_max() {
        let a = [1.0, 5.0, 3.0, 2.0];
        // SAFETY: Test code calling backend trait methods marked unsafe
        let result = unsafe { ScalarBackend::max(&a) };
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_scalar_min() {
        let a = [1.0, 5.0, 3.0, 2.0];
        // SAFETY: Test code calling backend trait methods marked unsafe
        let result = unsafe { ScalarBackend::min(&a) };
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_scalar_sub() {
        let a = [5.0, 6.0, 7.0, 8.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut result = [0.0; 4];
        // SAFETY: Test code calling backend trait methods marked unsafe
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
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            ScalarBackend::div(&a, &b, &mut result);
        }
        assert_eq!(result, [5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_scalar_argmax() {
        let a = [1.0, 5.0, 3.0, 2.0];
        // SAFETY: Test code calling backend trait methods marked unsafe
        let result = unsafe { ScalarBackend::argmax(&a) };
        assert_eq!(result, 1); // Index of 5.0
    }

    #[test]
    fn test_scalar_argmin() {
        let a = [5.0, 1.0, 3.0, 2.0];
        // SAFETY: Test code calling backend trait methods marked unsafe
        let result = unsafe { ScalarBackend::argmin(&a) };
        assert_eq!(result, 1); // Index of 1.0
    }

    #[test]
    fn test_scalar_sum_kahan() {
        let a = [1.0, 2.0, 3.0, 4.0];
        // SAFETY: Test code calling backend trait methods marked unsafe
        let result = unsafe { ScalarBackend::sum_kahan(&a) };
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_scalar_norm_l1() {
        let a = [1.0, -2.0, 3.0, -4.0];
        // SAFETY: Test code calling backend trait methods marked unsafe
        let result = unsafe { ScalarBackend::norm_l1(&a) };
        assert_eq!(result, 10.0); // |1| + |-2| + |3| + |-4| = 10
    }

    #[test]
    fn test_scalar_norm_l2() {
        let a = [3.0, 4.0];
        // SAFETY: Test code calling backend trait methods marked unsafe
        let result = unsafe { ScalarBackend::norm_l2(&a) };
        assert_eq!(result, 5.0); // sqrt(3² + 4²) = 5
    }

    #[test]
    fn test_scalar_scale() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let mut result = [0.0; 4];
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            ScalarBackend::scale(&a, 2.0, &mut result);
        }
        assert_eq!(result, [2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_scalar_clamp() {
        let a = [1.0, 5.0, 10.0, 15.0];
        let mut result = [0.0; 4];
        // SAFETY: Test code calling backend trait methods marked unsafe
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
        // SAFETY: Test code calling backend trait methods marked unsafe
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
        // SAFETY: Test code calling backend trait methods marked unsafe
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
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            ScalarBackend::relu(&a, &mut result);
        }
        assert_eq!(result, [0.0, 0.0, 0.0, 1.0, 3.0]);
    }

    #[test]
    fn test_scalar_sigmoid() {
        let a = [-51.0, -1.0, 0.0, 1.0, 51.0];
        let mut result = [0.0; 5];
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            ScalarBackend::sigmoid(&a, &mut result);
        }
        // sigmoid(-51) = 0, sigmoid(0) = 0.5, sigmoid(51) = 1
        assert_eq!(result[0], 0.0); // Clamped to 0 for numerical stability
        assert!((result[1] - 0.2689).abs() < 0.001); // sigmoid(-1)
        assert_eq!(result[2], 0.5); // sigmoid(0)
        assert!((result[3] - 0.7311).abs() < 0.001); // sigmoid(1)
        assert_eq!(result[4], 1.0); // Clamped to 1 for numerical stability
    }

    #[test]
    fn test_scalar_gelu() {
        let a = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut result = [0.0; 5];
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            ScalarBackend::gelu(&a, &mut result);
        }
        // GELU approximation values
        assert!((result[0] - (-0.0454)).abs() < 0.01); // gelu(-2)
        assert!((result[1] - (-0.1588)).abs() < 0.01); // gelu(-1)
        assert_eq!(result[2], 0.0); // gelu(0) = 0
        assert!((result[3] - 0.8413).abs() < 0.01); // gelu(1)
        assert!((result[4] - 1.9545).abs() < 0.01); // gelu(2)
    }

    #[test]
    fn test_scalar_swish() {
        let a = [-51.0, -1.0, 0.0, 1.0, 51.0];
        let mut result = [0.0; 5];
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            ScalarBackend::swish(&a, &mut result);
        }
        // swish(x) = x * sigmoid(x)
        assert_eq!(result[0], 0.0); // x * 0 = 0 (numerical stability)
        assert!((result[1] - (-0.2689)).abs() < 0.001); // -1 * sigmoid(-1)
        assert_eq!(result[2], 0.0); // 0 * sigmoid(0) = 0
        assert!((result[3] - 0.7311).abs() < 0.001); // 1 * sigmoid(1)
        assert_eq!(result[4], 51.0); // x * 1 = x (numerical stability)
    }
}
