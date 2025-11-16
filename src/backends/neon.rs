//! ARM NEON backend implementation (ARM 128-bit SIMD)
//!
//! This backend uses ARM NEON intrinsics for 128-bit SIMD operations.
//! NEON is available on ARMv7 (32-bit) and ARMv8/AArch64 (64-bit) CPUs.
//!
//! # Performance
//!
//! Expected speedup: 4x for operations on f32 vectors (4 elements per register)
//! Similar performance characteristics to SSE2 on x86_64.
//!
//! # Safety
//!
//! All NEON intrinsics are marked `unsafe` by Rust. This module carefully isolates
//! all unsafe code and verifies correctness through comprehensive testing.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "arm")]
use std::arch::arm::*;

use super::VectorBackend;

/// ARM NEON backend (128-bit SIMD)
pub struct NeonBackend;

impl VectorBackend for NeonBackend {
    #[target_feature(enable = "neon")]
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time using NEON (128-bit = 4 x f32)
        while i + 4 <= len {
            // Load 4 floats from a and b
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));

            // Add them
            let vresult = vaddq_f32(va, vb);

            // Store result
            vst1q_f32(result.as_mut_ptr().add(i), vresult);

            i += 4;
        }

        // Handle remaining elements with scalar code
        for j in i..len {
            result[j] = a[j] + b[j];
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time using NEON (128-bit = 4 x f32)
        while i + 4 <= len {
            // Load 4 floats from a and b
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));

            // Subtract them
            let vresult = vsubq_f32(va, vb);

            // Store result
            vst1q_f32(result.as_mut_ptr().add(i), vresult);

            i += 4;
        }

        // Handle remaining elements with scalar code
        for j in i..len {
            result[j] = a[j] - b[j];
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));

            let vresult = vmulq_f32(va, vb);

            vst1q_f32(result.as_mut_ptr().add(i), vresult);

            i += 4;
        }

        // Handle remaining elements
        for j in i..len {
            result[j] = a[j] * b[j];
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));

            let vresult = vdivq_f32(va, vb);

            vst1q_f32(result.as_mut_ptr().add(i), vresult);

            i += 4;
        }

        // Handle remaining elements
        for j in i..len {
            result[j] = a[j] / b[j];
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Accumulator for 4-way parallel accumulation
        let mut acc = vdupq_n_f32(0.0);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));

            // Multiply and accumulate
            acc = vmlaq_f32(acc, va, vb);

            i += 4;
        }

        // Horizontal sum: reduce 4 lanes to single value
        // Use pairwise addition to sum all lanes
        let sum2 = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        let sum1 = vpadd_f32(sum2, sum2);

        // Extract the final sum
        let mut result = vget_lane_f32(sum1, 0);

        // Handle remaining elements with scalar code
        result += a[i..].iter().zip(&b[i..]).map(|(x, y)| x * y).sum::<f32>();

        result
    }

    #[target_feature(enable = "neon")]
    unsafe fn sum(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        let mut acc = vdupq_n_f32(0.0);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            acc = vaddq_f32(acc, va);
            i += 4;
        }

        // Horizontal sum (same as dot product reduction)
        let sum2 = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        let sum1 = vpadd_f32(sum2, sum2);

        let mut result = vget_lane_f32(sum1, 0);

        // Handle remaining elements
        result += a[i..].iter().sum::<f32>();

        result
    }

    #[target_feature(enable = "neon")]
    unsafe fn max(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Start with first element broadcast to all lanes
        let mut vmax = vdupq_n_f32(a[0]);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            vmax = vmaxq_f32(vmax, va);
            i += 4;
        }

        // Horizontal max: find maximum across all 4 lanes
        // Use pairwise max to find the maximum
        let max2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        let max1 = vpmax_f32(max2, max2);

        let mut result = vget_lane_f32(max1, 0);

        // Check remaining elements
        for &val in &a[i..] {
            if val > result {
                result = val;
            }
        }

        result
    }

    #[target_feature(enable = "neon")]
    unsafe fn min(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Start with first element broadcast to all lanes
        let mut vmin = vdupq_n_f32(a[0]);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            vmin = vminq_f32(vmin, va);
            i += 4;
        }

        // Horizontal min: find minimum across all 4 lanes
        // Use pairwise min to find the minimum
        let min2 = vpmin_f32(vget_low_f32(vmin), vget_high_f32(vmin));
        let min1 = vpmin_f32(min2, min2);

        let mut result = vget_lane_f32(min1, 0);

        // Check remaining elements
        for &val in &a[i..] {
            if val < result {
                result = val;
            }
        }

        result
    }

    #[target_feature(enable = "neon")]
    unsafe fn argmax(a: &[f32]) -> usize {
        let len = a.len();
        let mut i = 0;

        // Track maximum value and index
        let mut max_value = a[0];
        let mut max_index = 0;

        // Start with first element broadcast to all lanes
        let mut vmax = vdupq_n_f32(a[0]);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            vmax = vmaxq_f32(vmax, va);
            i += 4;
        }

        // Horizontal max: find maximum across all 4 lanes (for potential future optimization)
        // Use pairwise max to find the maximum
        let max2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
        let max1 = vpmax_f32(max2, max2);

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

    #[target_feature(enable = "neon")]
    unsafe fn argmin(a: &[f32]) -> usize {
        let len = a.len();
        let mut i = 0;

        // Track minimum value and index
        let mut min_value = a[0];
        let mut min_index = 0;

        // Start with first element broadcast to all lanes
        let mut vmin = vdupq_n_f32(a[0]);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            vmin = vminq_f32(vmin, va);
            i += 4;
        }

        // Horizontal min: find minimum across all 4 lanes (for potential future optimization)
        // Use pairwise min to find the minimum
        let min2 = vpmin_f32(vget_low_f32(vmin), vget_high_f32(vmin));
        let min1 = vpmin_f32(min2, min2);

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

    #[target_feature(enable = "neon")]
    unsafe fn norm_l2(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        // L2 norm is sqrt(dot(a, a))
        let sum_of_squares = Self::dot(a, a);
        sum_of_squares.sqrt()
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        let len = a.len();
        let mut i = 0;

        // Accumulator for 4-way parallel accumulation
        let mut acc = vdupq_n_f32(0.0);

        // Process 4 elements at a time using NEON (128-bit = 4 x f32)
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));

            // Compute absolute value using NEON intrinsic
            let abs_va = vabsq_f32(va);

            // Accumulate
            acc = vaddq_f32(acc, abs_va);

            i += 4;
        }

        // Horizontal sum: sum all 4 lanes
        let mut result = vaddvq_f32(acc);

        // Handle remaining elements with scalar code
        for &val in &a[i..] {
            result += val.abs();
        }

        result
    }

    #[cfg(target_arch = "arm")]
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        // ARMv7 NEON implementation (32-bit ARM)
        if a.is_empty() {
            return 0.0;
        }

        let len = a.len();
        let mut i = 0;

        let mut acc = vdupq_n_f32(0.0);

        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let abs_va = vabsq_f32(va);
            acc = vaddq_f32(acc, abs_va);
            i += 4;
        }

        // Manual horizontal sum for ARMv7 (no vaddvq_f32)
        let mut result = {
            let sum_halves = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
            let sum_all = vpadd_f32(sum_halves, sum_halves);
            vget_lane_f32(sum_all, 0)
        };

        for &val in &a[i..] {
            result += val.abs();
        }

        result
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast scalar to all 4 lanes
        let scalar_vec = vdupq_n_f32(scalar);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vresult = vmulq_f32(va, scalar_vec);
            vst1q_f32(result.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] * scalar;
            i += 1;
        }
    }

    #[cfg(target_arch = "arm")]
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast scalar to all 4 lanes
        let scalar_vec = vdupq_n_f32(scalar);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vresult = vmulq_f32(va, scalar_vec);
            vst1q_f32(result.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] * scalar;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    #[test]
    fn test_neon_add() {
        // Skip test if NEON not available
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = vec![0.0; 9];

        unsafe {
            NeonBackend::add(&a, &b, &mut result);
        }

        assert_eq!(result, vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    #[test]
    fn test_neon_mul() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        let a = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut result = vec![0.0; 9];

        unsafe {
            NeonBackend::mul(&a, &b, &mut result);
        }

        assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 90.0]);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    #[test]
    fn test_neon_dot() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let result = unsafe { NeonBackend::dot(&a, &b) };

        // 1*9 + 2*8 + 3*7 + 4*6 + 5*5 + 6*4 + 7*3 + 8*2 + 9*1 = 165
        assert!((result - 165.0).abs() < 1e-5);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    #[test]
    fn test_neon_sum() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = unsafe { NeonBackend::sum(&a) };

        assert!((result - 45.0).abs() < 1e-5);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    #[test]
    fn test_neon_max() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        let a = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];

        let result = unsafe { NeonBackend::max(&a) };

        assert_eq!(result, 9.0);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    #[test]
    fn test_neon_min() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        let a = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];

        let result = unsafe { NeonBackend::min(&a) };

        assert_eq!(result, 1.0);
    }

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    #[test]
    fn test_neon_matches_scalar() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];
        let b = vec![10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5];

        // Test add
        let mut neon_result = vec![0.0; 10];
        let mut scalar_result = vec![0.0; 10];
        unsafe {
            NeonBackend::add(&a, &b, &mut neon_result);
            ScalarBackend::add(&a, &b, &mut scalar_result);
        }
        for (neon, scalar) in neon_result.iter().zip(&scalar_result) {
            assert!((neon - scalar).abs() < 1e-5);
        }

        // Test dot
        let (neon_dot, scalar_dot) = unsafe {
            (NeonBackend::dot(&a, &b), ScalarBackend::dot(&a, &b))
        };
        assert!((neon_dot - scalar_dot).abs() < 1e-3);

        // Test sum
        let (neon_sum, scalar_sum) = unsafe {
            (NeonBackend::sum(&a), ScalarBackend::sum(&a))
        };
        assert!((neon_sum - scalar_sum).abs() < 1e-3);

        // Test max
        let (neon_max, scalar_max) = unsafe {
            (NeonBackend::max(&a), ScalarBackend::max(&a))
        };
        assert_eq!(neon_max, scalar_max);
    }
}
