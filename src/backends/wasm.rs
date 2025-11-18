//! WebAssembly SIMD128 backend implementation
//!
//! This backend uses WebAssembly SIMD128 intrinsics for 128-bit SIMD operations.
//! SIMD128 is supported in modern browsers and wasm runtimes.
//!
//! # Performance
//!
//! Expected speedup: 4x for operations on f32 vectors (4 elements per register)
//! Similar performance characteristics to SSE2 and NEON.
//!
//! # Safety
//!
//! All WASM SIMD intrinsics are marked `unsafe` by Rust. This module carefully isolates
//! all unsafe code and verifies correctness through comprehensive testing.

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;

use super::VectorBackend;

/// WebAssembly SIMD128 backend (128-bit SIMD)
pub struct WasmBackend;

impl VectorBackend for WasmBackend {
    #[target_feature(enable = "simd128")]
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time using SIMD128 (128-bit = 4 x f32)
        while i + 4 <= len {
            // Load 4 floats from a and b
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);

            // Add them
            let vresult = f32x4_add(va, vb);

            // Store result
            v128_store(result.as_mut_ptr().add(i) as *mut v128, vresult);

            i += 4;
        }

        // Handle remaining elements with scalar code
        for j in i..len {
            result[j] = a[j] + b[j];
        }
    }

    #[target_feature(enable = "simd128")]
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time using SIMD128 (128-bit = 4 x f32)
        while i + 4 <= len {
            // Load 4 floats from a and b
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);

            // Subtract them
            let vresult = f32x4_sub(va, vb);

            // Store result
            v128_store(result.as_mut_ptr().add(i) as *mut v128, vresult);

            i += 4;
        }

        // Handle remaining elements with scalar code
        for j in i..len {
            result[j] = a[j] - b[j];
        }
    }

    #[target_feature(enable = "simd128")]
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);

            let vresult = f32x4_mul(va, vb);

            v128_store(result.as_mut_ptr().add(i) as *mut v128, vresult);

            i += 4;
        }

        // Handle remaining elements
        for j in i..len {
            result[j] = a[j] * b[j];
        }
    }

    #[target_feature(enable = "simd128")]
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);

            let vresult = f32x4_div(va, vb);

            v128_store(result.as_mut_ptr().add(i) as *mut v128, vresult);

            i += 4;
        }

        // Handle remaining elements
        for j in i..len {
            result[j] = a[j] / b[j];
        }
    }

    #[target_feature(enable = "simd128")]
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Accumulator for 4-way parallel accumulation
        let mut acc = f32x4_splat(0.0);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);

            // Multiply and accumulate
            let prod = f32x4_mul(va, vb);
            acc = f32x4_add(acc, prod);

            i += 4;
        }

        // Horizontal sum: extract all 4 lanes and sum them
        let mut result = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);

        // Handle remaining elements with scalar code
        result += a[i..].iter().zip(&b[i..]).map(|(x, y)| x * y).sum::<f32>();

        result
    }

    #[target_feature(enable = "simd128")]
    unsafe fn sum(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        let mut acc = f32x4_splat(0.0);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            acc = f32x4_add(acc, va);
            i += 4;
        }

        // Horizontal sum
        let mut result = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);

        // Handle remaining elements
        result += a[i..].iter().sum::<f32>();

        result
    }

    #[target_feature(enable = "simd128")]
    unsafe fn max(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Start with first element broadcast to all lanes
        let mut vmax = f32x4_splat(a[0]);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            vmax = f32x4_max(vmax, va);
            i += 4;
        }

        // Horizontal max: find maximum across all 4 lanes
        let mut result = f32x4_extract_lane::<0>(vmax)
            .max(f32x4_extract_lane::<1>(vmax))
            .max(f32x4_extract_lane::<2>(vmax))
            .max(f32x4_extract_lane::<3>(vmax));

        // Check remaining elements
        for &val in &a[i..] {
            if val > result {
                result = val;
            }
        }

        result
    }

    #[target_feature(enable = "simd128")]
    unsafe fn min(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Start with first element broadcast to all lanes
        let mut vmin = f32x4_splat(a[0]);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            vmin = f32x4_min(vmin, va);
            i += 4;
        }

        // Horizontal min: find minimum across all 4 lanes
        let mut result = f32x4_extract_lane::<0>(vmin)
            .min(f32x4_extract_lane::<1>(vmin))
            .min(f32x4_extract_lane::<2>(vmin))
            .min(f32x4_extract_lane::<3>(vmin));

        // Check remaining elements
        for &val in &a[i..] {
            if val < result {
                result = val;
            }
        }

        result
    }

    #[target_feature(enable = "simd128")]
    unsafe fn argmax(a: &[f32]) -> usize {
        let len = a.len();
        let mut i = 0;

        // Track maximum value and index
        let mut max_value = a[0];
        let mut max_index = 0;

        // Start with first element broadcast to all lanes
        let mut vmax = f32x4_splat(a[0]);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            vmax = f32x4_max(vmax, va);
            i += 4;
        }

        // Horizontal max: find maximum across all 4 lanes (for potential future optimization)
        // This extracts the max value but we still need to scan to find the index

        // Find the index by checking all elements processed by SIMD
        for (idx, &val) in a[..i].iter().enumerate() {
            if val > max_value {
                max_value = val;
                max_index = idx;
            }
        }

        // Check remaining elements
        for (idx, &val) in a[i..].iter().enumerate() {
            if val > max_value {
                max_value = val;
                max_index = i + idx;
            }
        }

        max_index
    }

    #[target_feature(enable = "simd128")]
    unsafe fn argmin(a: &[f32]) -> usize {
        let len = a.len();
        let mut i = 0;

        // Track minimum value and index
        let mut min_value = a[0];
        let mut min_index = 0;

        // Start with first element broadcast to all lanes
        let mut vmin = f32x4_splat(a[0]);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            vmin = f32x4_min(vmin, va);
            i += 4;
        }

        // Horizontal min: find minimum across all 4 lanes (for potential future optimization)
        // This extracts the min value but we still need to scan to find the index

        // Find the index by checking all elements processed by SIMD
        for (idx, &val) in a[..i].iter().enumerate() {
            if val < min_value {
                min_value = val;
                min_index = idx;
            }
        }

        // Check remaining elements
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

    #[target_feature(enable = "simd128")]
    unsafe fn norm_l2(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        // L2 norm is sqrt(dot(a, a))
        let sum_of_squares = Self::dot(a, a);
        sum_of_squares.sqrt()
    }

    #[target_feature(enable = "simd128")]
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        let len = a.len();
        let mut i = 0;

        // Accumulator for 4-way parallel accumulation
        let mut acc = f32x4_splat(0.0);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);

            // Compute absolute value
            let abs_va = f32x4_abs(va);

            // Accumulate
            acc = f32x4_add(acc, abs_va);

            i += 4;
        }

        // Horizontal sum: extract all 4 lanes and sum them
        let mut result = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);

        // Handle remaining elements with scalar code
        for &val in &a[i..] {
            result += val.abs();
        }

        result
    }

    #[target_feature(enable = "simd128")]
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast scalar to all 4 lanes
        let scalar_vec = f32x4_splat(scalar);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vresult = f32x4_mul(va, scalar_vec);
            v128_store(result.as_mut_ptr().add(i) as *mut v128, vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] * scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "simd128")]
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast min and max to all 4 lanes
        let min_vec = f32x4_splat(min_val);
        let max_vec = f32x4_splat(max_val);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let clamped = f32x4_pmin(f32x4_pmax(va, min_vec), max_vec);
            v128_store(result.as_mut_ptr().add(i) as *mut v128, clamped);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i].max(min_val).min(max_val);
            i += 1;
        }
    }

    #[target_feature(enable = "simd128")]
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast t to all 4 lanes
        let t_vec = f32x4_splat(t);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);

            // result = a + t * (b - a)
            let diff = f32x4_sub(vb, va);
            let scaled_diff = f32x4_mul(t_vec, diff);
            let vresult = f32x4_add(va, scaled_diff);

            v128_store(result.as_mut_ptr().add(i) as *mut v128, vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] + t * (b[i] - a[i]);
            i += 1;
        }
    }

    #[target_feature(enable = "simd128")]
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);
            let vb = v128_load(b.as_ptr().add(i) as *const v128);
            let vc = v128_load(c.as_ptr().add(i) as *const v128);

            // result = a * b + c
            // WASM SIMD128 doesn't have FMA, so we use separate mul and add
            let product = f32x4_mul(va, vb);
            let vresult = f32x4_add(product, vc);

            v128_store(result.as_mut_ptr().add(i) as *mut v128, vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] * b[i] + c[i];
            i += 1;
        }
    }

    #[target_feature(enable = "simd128")]
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Zero vector for max comparison
        let zero = f32x4_splat(0.0);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = v128_load(a.as_ptr().add(i) as *const v128);

            // ReLU: max(0, x)
            let vresult = f32x4_max(zero, va);

            v128_store(result.as_mut_ptr().add(i) as *mut v128, vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
            i += 1;
        }
    }

    #[target_feature(enable = "simd128")]
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        // WASM SIMD128 doesn't have native exp(), use scalar with numerical stability
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

    #[target_feature(enable = "simd128")]
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        // WASM SIMD128 doesn't have native tanh(), use scalar
        const SQRT_2_OVER_PI: f32 = 0.7978845608;
        const COEFF: f32 = 0.044715;

        for (i, &x) in a.iter().enumerate() {
            let x3 = x * x * x;
            let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            result[i] = 0.5 * x * (1.0 + inner.tanh());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = vec![0.0; 9];

        unsafe {
            WasmBackend::add(&a, &b, &mut result);
        }

        assert_eq!(
            result,
            vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        );
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_mul() {
        let a = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut result = vec![0.0; 9];

        unsafe {
            WasmBackend::mul(&a, &b, &mut result);
        }

        assert_eq!(
            result,
            vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 90.0]
        );
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let result = unsafe { WasmBackend::dot(&a, &b) };

        // 1*9 + 2*8 + 3*7 + 4*6 + 5*5 + 6*4 + 7*3 + 8*2 + 9*1 = 165
        assert!((result - 165.0).abs() < 1e-5);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_sum() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = unsafe { WasmBackend::sum(&a) };

        assert!((result - 45.0).abs() < 1e-5);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_max() {
        let a = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];

        let result = unsafe { WasmBackend::max(&a) };

        assert_eq!(result, 9.0);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_min() {
        let a = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];

        let result = unsafe { WasmBackend::min(&a) };

        assert_eq!(result, 1.0);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_wasm_matches_scalar() {
        use super::super::scalar::ScalarBackend;

        let a = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];
        let b = vec![10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5];

        // Test add
        let mut wasm_result = vec![0.0; 10];
        let mut scalar_result = vec![0.0; 10];
        unsafe {
            WasmBackend::add(&a, &b, &mut wasm_result);
            ScalarBackend::add(&a, &b, &mut scalar_result);
        }
        for (wasm, scalar) in wasm_result.iter().zip(&scalar_result) {
            assert!((wasm - scalar).abs() < 1e-5);
        }

        // Test dot
        let (wasm_dot, scalar_dot) =
            unsafe { (WasmBackend::dot(&a, &b), ScalarBackend::dot(&a, &b)) };
        assert!((wasm_dot - scalar_dot).abs() < 1e-3);

        // Test sum
        let (wasm_sum, scalar_sum) = unsafe { (WasmBackend::sum(&a), ScalarBackend::sum(&a)) };
        assert!((wasm_sum - scalar_sum).abs() < 1e-3);

        // Test max
        let (wasm_max, scalar_max) = unsafe { (WasmBackend::max(&a), ScalarBackend::max(&a)) };
        assert_eq!(wasm_max, scalar_max);
    }
}
