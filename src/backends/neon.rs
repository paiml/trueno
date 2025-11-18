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

    #[cfg(target_arch = "aarch64")]
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast min and max to all 4 lanes
        let min_vec = vdupq_n_f32(min_val);
        let max_vec = vdupq_n_f32(max_val);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let clamped = vminq_f32(vmaxq_f32(va, min_vec), max_vec);
            vst1q_f32(result.as_mut_ptr().add(i), clamped);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i].max(min_val).min(max_val);
            i += 1;
        }
    }

    #[cfg(target_arch = "arm")]
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast min and max to all 4 lanes
        let min_vec = vdupq_n_f32(min_val);
        let max_vec = vdupq_n_f32(max_val);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let clamped = vminq_f32(vmaxq_f32(va, min_vec), max_vec);
            vst1q_f32(result.as_mut_ptr().add(i), clamped);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i].max(min_val).min(max_val);
            i += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast t to all 4 lanes
        let t_vec = vdupq_n_f32(t);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));

            // result = a + t * (b - a)
            // Can use FMA: vfmaq_f32(a, t, b-a) = a + t * (b - a)
            let diff = vsubq_f32(vb, va);
            let vresult = vfmaq_f32(va, t_vec, diff);

            vst1q_f32(result.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] + t * (b[i] - a[i]);
            i += 1;
        }
    }

    #[cfg(target_arch = "arm")]
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast t to all 4 lanes
        let t_vec = vdupq_n_f32(t);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));

            // result = a + t * (b - a)
            // ARMv7 NEON also has FMA: vmlaq_f32(a, t, b-a) = a + t * (b - a)
            let diff = vsubq_f32(vb, va);
            let vresult = vmlaq_f32(va, t_vec, diff);

            vst1q_f32(result.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] + t * (b[i] - a[i]);
            i += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vc = vld1q_f32(c.as_ptr().add(i));

            // result = a * b + c
            // Using FMA: vfmaq_f32(c, a, b) = c + a * b = a * b + c
            let vresult = vfmaq_f32(vc, va, vb);

            vst1q_f32(result.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] * b[i] + c[i];
            i += 1;
        }
    }

    #[cfg(target_arch = "arm")]
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let vc = vld1q_f32(c.as_ptr().add(i));

            // result = a * b + c
            // Using FMA: vmlaq_f32(c, a, b) = c + a * b = a * b + c
            let vresult = vmlaq_f32(vc, va, vb);

            vst1q_f32(result.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] * b[i] + c[i];
            i += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Zero vector for max comparison
        let zero = vdupq_n_f32(0.0);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));

            // ReLU: max(0, x)
            let vresult = vmaxq_f32(zero, va);

            vst1q_f32(result.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
            i += 1;
        }
    }

    #[cfg(target_arch = "arm")]
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Zero vector for max comparison
        let zero = vdupq_n_f32(0.0);

        // Process 4 elements at a time
        while i + 4 <= len {
            let va = vld1q_f32(a.as_ptr().add(i));

            // ReLU: max(0, x)
            let vresult = vmaxq_f32(zero, va);

            vst1q_f32(result.as_mut_ptr().add(i), vresult);
            i += 4;
        }

        // Handle remaining elements
        while i < len {
            result[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
            i += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        // SIMD implementation using range reduction for exp
        let len = a.len();
        let mut i = 0;

        // Constants for exp(-x) computation
        let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
        let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
        let one = vdupq_n_f32(1.0);
        let half = vdupq_n_f32(0.5);
        let zero = vdupq_n_f32(0.0);

        // Taylor series coefficients
        let c1 = vdupq_n_f32(1.0);
        let c2 = vdupq_n_f32(0.5);
        let c3 = vdupq_n_f32(0.166_666_67);
        let c4 = vdupq_n_f32(0.041_666_668);
        let c5 = vdupq_n_f32(0.008_333_334);
        let c6 = vdupq_n_f32(0.001_388_889);

        while i + 4 <= len {
            let x = vld1q_f32(a.as_ptr().add(i));
            let neg_x = vsubq_f32(zero, x);

            // Range reduction: k = floor(x * log2(e) + 0.5)
            let kf = vrndmq_f32(vaddq_f32(vmulq_f32(neg_x, log2e), half));
            let k = vcvtq_s32_f32(kf);
            let r = vsubq_f32(neg_x, vmulq_f32(kf, ln2));

            // Polynomial approximation using Horner's method
            let mut poly = vaddq_f32(c5, vmulq_f32(r, c6));
            poly = vaddq_f32(c4, vmulq_f32(r, poly));
            poly = vaddq_f32(c3, vmulq_f32(r, poly));
            poly = vaddq_f32(c2, vmulq_f32(r, poly));
            poly = vaddq_f32(c1, vmulq_f32(r, poly));
            poly = vaddq_f32(one, vmulq_f32(r, poly));

            // Scale by 2^k
            let k_shifted = vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23);
            let exp_neg_x = vmulq_f32(poly, vreinterpretq_f32_s32(k_shifted));

            // sigmoid = 1 / (1 + exp(-x))
            let sigmoid_result = vdivq_f32(one, vaddq_f32(one, exp_neg_x));

            vst1q_f32(result.as_mut_ptr().add(i), sigmoid_result);
            i += 4;
        }

        // Scalar fallback
        while i < len {
            let val = a[i];
            result[i] = if val < -50.0 {
                0.0
            } else if val > 50.0 {
                1.0
            } else {
                1.0 / (1.0 + (-val).exp())
            };
            i += 1;
        }
    }

    #[cfg(target_arch = "arm")]
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        // ARM32: use scalar since NEON division requires Newton-Raphson
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

    #[cfg(target_arch = "aarch64")]
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        let len = a.len();
        let mut i = 0;

        // Constants
        let sqrt_2_over_pi = vdupq_n_f32(0.797_884_6);
        let coeff = vdupq_n_f32(0.044715);
        let half = vdupq_n_f32(0.5);
        let one = vdupq_n_f32(1.0);
        let two = vdupq_n_f32(2.0);

        // exp constants
        let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
        let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
        let c1 = vdupq_n_f32(1.0);
        let c2 = vdupq_n_f32(0.5);
        let c3 = vdupq_n_f32(0.166_666_67);
        let c4 = vdupq_n_f32(0.041_666_668);
        let c5 = vdupq_n_f32(0.008_333_334);
        let c6 = vdupq_n_f32(0.001_388_889);

        while i + 4 <= len {
            let x = vld1q_f32(a.as_ptr().add(i));

            // inner = sqrt(2/π) * (x + 0.044715 * x³)
            let x2 = vmulq_f32(x, x);
            let x3 = vmulq_f32(x2, x);
            let inner = vmulq_f32(sqrt_2_over_pi, vaddq_f32(x, vmulq_f32(coeff, x3)));

            // Compute exp(2 * inner) for tanh
            let z = vmulq_f32(two, inner);

            // Range reduction
            let kf = vrndmq_f32(vaddq_f32(vmulq_f32(z, log2e), half));
            let k = vcvtq_s32_f32(kf);
            let r = vsubq_f32(z, vmulq_f32(kf, ln2));

            // Polynomial
            let mut poly = vaddq_f32(c5, vmulq_f32(r, c6));
            poly = vaddq_f32(c4, vmulq_f32(r, poly));
            poly = vaddq_f32(c3, vmulq_f32(r, poly));
            poly = vaddq_f32(c2, vmulq_f32(r, poly));
            poly = vaddq_f32(c1, vmulq_f32(r, poly));
            poly = vaddq_f32(one, vmulq_f32(r, poly));

            // Scale by 2^k
            let k_shifted = vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23);
            let exp_2z = vmulq_f32(poly, vreinterpretq_f32_s32(k_shifted));

            // tanh = (exp(2z) - 1) / (exp(2z) + 1)
            let tanh_val = vdivq_f32(vsubq_f32(exp_2z, one), vaddq_f32(exp_2z, one));

            // gelu = 0.5 * x * (1 + tanh)
            let gelu_result = vmulq_f32(half, vmulq_f32(x, vaddq_f32(one, tanh_val)));

            vst1q_f32(result.as_mut_ptr().add(i), gelu_result);
            i += 4;
        }

        // Scalar fallback
        while i < len {
            let x = a[i];
            let x3 = x * x * x;
            let inner = 0.797_884_6 * (x + 0.044715 * x3);
            result[i] = 0.5 * x * (1.0 + inner.tanh());
            i += 1;
        }
    }

    #[cfg(target_arch = "arm")]
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        // ARM32: use scalar since NEON division requires Newton-Raphson
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        const COEFF: f32 = 0.044715;

        for (i, &x) in a.iter().enumerate() {
            let x3 = x * x * x;
            let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            result[i] = 0.5 * x * (1.0 + inner.tanh());
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        // swish(x) = x / (1 + exp(-x))
        let len = a.len();
        let mut i = 0;

        // Constants
        let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
        let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
        let one = vdupq_n_f32(1.0);
        let half = vdupq_n_f32(0.5);
        let zero = vdupq_n_f32(0.0);

        // Taylor series coefficients
        let c1 = vdupq_n_f32(1.0);
        let c2 = vdupq_n_f32(0.5);
        let c3 = vdupq_n_f32(0.166_666_67);
        let c4 = vdupq_n_f32(0.041_666_668);
        let c5 = vdupq_n_f32(0.008_333_334);
        let c6 = vdupq_n_f32(0.001_388_889);

        while i + 4 <= len {
            let x = vld1q_f32(a.as_ptr().add(i));
            let neg_x = vsubq_f32(zero, x);

            // Range reduction
            let kf = vrndmq_f32(vaddq_f32(vmulq_f32(neg_x, log2e), half));
            let k = vcvtq_s32_f32(kf);
            let r = vsubq_f32(neg_x, vmulq_f32(kf, ln2));

            // Polynomial
            let mut poly = vaddq_f32(c5, vmulq_f32(r, c6));
            poly = vaddq_f32(c4, vmulq_f32(r, poly));
            poly = vaddq_f32(c3, vmulq_f32(r, poly));
            poly = vaddq_f32(c2, vmulq_f32(r, poly));
            poly = vaddq_f32(c1, vmulq_f32(r, poly));
            poly = vaddq_f32(one, vmulq_f32(r, poly));

            // Scale by 2^k
            let k_shifted = vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23);
            let exp_neg_x = vmulq_f32(poly, vreinterpretq_f32_s32(k_shifted));

            // swish = x / (1 + exp(-x))
            let swish_result = vdivq_f32(x, vaddq_f32(one, exp_neg_x));

            vst1q_f32(result.as_mut_ptr().add(i), swish_result);
            i += 4;
        }

        // Scalar fallback
        while i < len {
            let x = a[i];
            if x < -50.0 {
                result[i] = 0.0;
            } else if x > 50.0 {
                result[i] = x;
            } else {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                result[i] = x * sigmoid;
            }
            i += 1;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn tanh(a: &[f32], result: &mut [f32]) {
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        let len = a.len();
        let mut i = 0;

        // Constants
        let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
        let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
        let one = vdupq_n_f32(1.0);
        let two = vdupq_n_f32(2.0);
        let half = vdupq_n_f32(0.5);

        // Taylor series coefficients for exp(y)
        let c1 = vdupq_n_f32(1.0);
        let c2 = vdupq_n_f32(0.5);
        let c3 = vdupq_n_f32(0.166_666_67);
        let c4 = vdupq_n_f32(0.041_666_668);
        let c5 = vdupq_n_f32(0.008_333_334);
        let c6 = vdupq_n_f32(0.001_388_889);

        while i + 4 <= len {
            let x = vld1q_f32(a.as_ptr().add(i));

            // Compute exp(2x) for tanh = (exp(2x) - 1) / (exp(2x) + 1)
            let z = vmulq_f32(two, x);

            // Range reduction: k = floor(z * log2(e) + 0.5)
            let kf = vrndmq_f32(vaddq_f32(vmulq_f32(z, log2e), half));
            let k = vcvtq_s32_f32(kf);
            let r = vsubq_f32(z, vmulq_f32(kf, ln2));

            // Polynomial approximation using Horner's method
            let mut poly = vaddq_f32(c5, vmulq_f32(r, c6));
            poly = vaddq_f32(c4, vmulq_f32(r, poly));
            poly = vaddq_f32(c3, vmulq_f32(r, poly));
            poly = vaddq_f32(c2, vmulq_f32(r, poly));
            poly = vaddq_f32(c1, vmulq_f32(r, poly));
            poly = vaddq_f32(one, vmulq_f32(r, poly));

            // Scale by 2^k using IEEE754 exponent manipulation
            let k_shifted = vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23);
            let exp_2x = vmulq_f32(poly, vreinterpretq_f32_s32(k_shifted));

            // tanh = (exp(2x) - 1) / (exp(2x) + 1)
            let tanh_result = vdivq_f32(vsubq_f32(exp_2x, one), vaddq_f32(exp_2x, one));

            vst1q_f32(result.as_mut_ptr().add(i), tanh_result);
            i += 4;
        }

        // Scalar fallback for remainder
        while i < len {
            let val = a[i];
            result[i] = val.tanh();
            i += 1;
        }
    }

    #[cfg(target_arch = "arm")]
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        // ARM32: use scalar since NEON division requires Newton-Raphson
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

    #[cfg(target_arch = "arm")]
    unsafe fn tanh(a: &[f32], result: &mut [f32]) {
        // ARM32: use scalar fallback since NEON division requires Newton-Raphson
        for (i, &x) in a.iter().enumerate() {
            result[i] = x.tanh();
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

        assert_eq!(
            result,
            vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        );
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

        assert_eq!(
            result,
            vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 90.0]
        );
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
        let (neon_dot, scalar_dot) =
            unsafe { (NeonBackend::dot(&a, &b), ScalarBackend::dot(&a, &b)) };
        assert!((neon_dot - scalar_dot).abs() < 1e-3);

        // Test sum
        let (neon_sum, scalar_sum) = unsafe { (NeonBackend::sum(&a), ScalarBackend::sum(&a)) };
        assert!((neon_sum - scalar_sum).abs() < 1e-3);

        // Test max
        let (neon_max, scalar_max) = unsafe { (NeonBackend::max(&a), ScalarBackend::max(&a)) };
        assert_eq!(neon_max, scalar_max);
    }

    #[test]
    fn test_neon_sub_matches_scalar() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [5.0, 6.0, 7.0, 8.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let mut neon_result = [0.0; 4];
        let mut scalar_result = [0.0; 4];
        unsafe {
            NeonBackend::sub(&a, &b, &mut neon_result);
            ScalarBackend::sub(&a, &b, &mut scalar_result);
        }
        assert_eq!(neon_result, scalar_result);
    }

    #[test]
    fn test_neon_mul_matches_scalar() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut neon_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];
        unsafe {
            NeonBackend::mul(&a, &b, &mut neon_result);
            ScalarBackend::mul(&a, &b, &mut scalar_result);
        }
        assert_eq!(neon_result, scalar_result);
    }

    #[test]
    fn test_neon_div_matches_scalar() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [10.0, 20.0, 30.0, 40.0];
        let b = [2.0, 4.0, 5.0, 8.0];
        let mut neon_result = [0.0; 4];
        let mut scalar_result = [0.0; 4];
        unsafe {
            NeonBackend::div(&a, &b, &mut neon_result);
            ScalarBackend::div(&a, &b, &mut scalar_result);
        }
        assert_eq!(neon_result, scalar_result);
    }

    #[test]
    fn test_neon_min_matches_scalar() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [5.0, 1.0, 3.0, 2.0];
        let neon_result = unsafe { NeonBackend::min(&a) };
        let scalar_result = unsafe { ScalarBackend::min(&a) };
        assert_eq!(neon_result, scalar_result);
    }

    #[test]
    fn test_neon_argmax_matches_scalar() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [1.0, 5.0, 3.0, 2.0];
        let neon_result = unsafe { NeonBackend::argmax(&a) };
        let scalar_result = unsafe { ScalarBackend::argmax(&a) };
        assert_eq!(neon_result, scalar_result);
    }

    #[test]
    fn test_neon_argmin_matches_scalar() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [5.0, 1.0, 3.0, 2.0];
        let neon_result = unsafe { NeonBackend::argmin(&a) };
        let scalar_result = unsafe { ScalarBackend::argmin(&a) };
        assert_eq!(neon_result, scalar_result);
    }

    #[test]
    fn test_neon_relu_matches_scalar() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [-3.0, -1.0, 0.0, 1.0, 3.0];
        let mut neon_result = [0.0; 5];
        let mut scalar_result = [0.0; 5];
        unsafe {
            NeonBackend::relu(&a, &mut neon_result);
            ScalarBackend::relu(&a, &mut scalar_result);
        }
        assert_eq!(neon_result, scalar_result);
    }

    #[test]
    fn test_neon_sigmoid_matches_scalar() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut neon_result = [0.0; 5];
        let mut scalar_result = [0.0; 5];
        unsafe {
            NeonBackend::sigmoid(&a, &mut neon_result);
            ScalarBackend::sigmoid(&a, &mut scalar_result);
        }
        for (n, s) in neon_result.iter().zip(scalar_result.iter()) {
            assert!(
                (n - s).abs() < 1e-5,
                "sigmoid mismatch: neon={}, scalar={}",
                n,
                s
            );
        }
    }

    #[test]
    fn test_neon_gelu_matches_scalar() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut neon_result = [0.0; 5];
        let mut scalar_result = [0.0; 5];
        unsafe {
            NeonBackend::gelu(&a, &mut neon_result);
            ScalarBackend::gelu(&a, &mut scalar_result);
        }
        for (n, s) in neon_result.iter().zip(scalar_result.iter()) {
            assert!(
                (n - s).abs() < 1e-5,
                "gelu mismatch: neon={}, scalar={}",
                n,
                s
            );
        }
    }

    #[test]
    fn test_neon_swish_matches_scalar() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut neon_result = [0.0; 5];
        let mut scalar_result = [0.0; 5];
        unsafe {
            NeonBackend::swish(&a, &mut neon_result);
            ScalarBackend::swish(&a, &mut scalar_result);
        }
        for (n, s) in neon_result.iter().zip(scalar_result.iter()) {
            assert!(
                (n - s).abs() < 1e-5,
                "swish mismatch: neon={}, scalar={}",
                n,
                s
            );
        }
    }

    #[test]
    fn test_neon_tanh_matches_scalar() {
        #[cfg(target_arch = "aarch64")]
        if !std::arch::is_aarch64_feature_detected!("neon") {
            eprintln!("Skipping NEON test: CPU does not support NEON");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut neon_result = [0.0; 5];
        let mut scalar_result = [0.0; 5];
        unsafe {
            NeonBackend::tanh(&a, &mut neon_result);
            ScalarBackend::tanh(&a, &mut scalar_result);
        }
        for (n, s) in neon_result.iter().zip(scalar_result.iter()) {
            assert!(
                (n - s).abs() < 1e-5,
                "tanh mismatch: neon={}, scalar={}",
                n,
                s
            );
        }
    }
}
