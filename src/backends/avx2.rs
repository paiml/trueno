//! AVX2 backend implementation (x86_64 advanced SIMD)
//!
//! This backend uses AVX2 intrinsics for 256-bit SIMD operations with FMA.
//! AVX2 is available on Intel Haswell (2013+) and AMD Excavator (2015+) CPUs.
//!
//! # Performance
//!
//! Expected speedup: 8x for operations on aligned f32 vectors (8 elements per register)
//! FMA provides additional speedup for dot product operations.
//!
//! # Safety
//!
//! All AVX2 intrinsics are marked `unsafe` by Rust. This module carefully isolates
//! all unsafe code and verifies correctness through comprehensive testing.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::VectorBackend;

/// AVX2 backend (256-bit SIMD for x86_64)
pub struct Avx2Backend;

impl VectorBackend for Avx2Backend {
    #[target_feature(enable = "avx2")]
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 8 elements at a time using AVX2 (256-bit = 8 x f32)
        while i + 8 <= len {
            // Load 8 floats from a and b
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));

            // Add them
            let vresult = _mm256_add_ps(va, vb);

            // Store result
            _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);

            i += 8;
        }

        // Handle remaining elements with scalar code
        for j in i..len {
            result[j] = a[j] + b[j];
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 8 elements at a time using AVX2 (256-bit = 8 x f32)
        while i + 8 <= len {
            // Load 8 floats from a and b
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));

            // Subtract them
            let vresult = _mm256_sub_ps(va, vb);

            // Store result
            _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);

            i += 8;
        }

        // Handle remaining elements with scalar code
        for j in i..len {
            result[j] = a[j] - b[j];
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 8 elements at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));

            let vresult = _mm256_mul_ps(va, vb);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);

            i += 8;
        }

        // Handle remaining elements
        for j in i..len {
            result[j] = a[j] * b[j];
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 8 elements at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));

            let vresult = _mm256_div_ps(va, vb);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);

            i += 8;
        }

        // Handle remaining elements
        for j in i..len {
            result[j] = a[j] / b[j];
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Accumulator for 8-way parallel accumulation
        let mut acc = _mm256_setzero_ps();

        // Process 8 elements at a time with FMA
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));

            // Fused multiply-add: acc = acc + (va * vb)
            acc = _mm256_fmadd_ps(va, vb, acc);

            i += 8;
        }

        // Horizontal sum: reduce 8 lanes to single value
        let mut result = {
            // Sum upper and lower 128-bit halves
            let sum_halves = _mm_add_ps(
                _mm256_castps256_ps128(acc),
                _mm256_extractf128_ps(acc, 1),
            );
            // Horizontal sum of 4 elements using faster movehl/shuffle
            let temp = _mm_add_ps(sum_halves, _mm_movehl_ps(sum_halves, sum_halves));
            let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };

        // Handle remaining elements with scalar code
        result += a[i..].iter().zip(&b[i..]).map(|(x, y)| x * y).sum::<f32>();

        result
    }

    #[target_feature(enable = "avx2")]
    unsafe fn sum(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        let mut acc = _mm256_setzero_ps();

        // Process 8 elements at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            acc = _mm256_add_ps(acc, va);
            i += 8;
        }

        // Horizontal sum: reduce 8 lanes to single value
        let mut result = {
            // Sum upper and lower 128-bit halves
            let sum_halves = _mm_add_ps(
                _mm256_castps256_ps128(acc),
                _mm256_extractf128_ps(acc, 1),
            );
            // Horizontal sum of 4 elements using faster movehl/shuffle
            let temp = _mm_add_ps(sum_halves, _mm_movehl_ps(sum_halves, sum_halves));
            let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };

        // Handle remaining elements
        result += a[i..].iter().sum::<f32>();

        result
    }

    #[target_feature(enable = "avx2")]
    unsafe fn max(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Start with first element broadcast to all lanes
        let mut vmax = _mm256_set1_ps(a[0]);

        // Process 8 elements at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            vmax = _mm256_max_ps(vmax, va);
            i += 8;
        }

        // Horizontal max: find maximum across all 8 lanes
        let mut result = {
            // Max of upper and lower 128-bit halves
            let max_halves = _mm_max_ps(
                _mm256_castps256_ps128(vmax),
                _mm256_extractf128_ps(vmax, 1),
            );
            // Horizontal max of 4 elements
            let temp = _mm_max_ps(max_halves, _mm_movehl_ps(max_halves, max_halves));
            let temp = _mm_max_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };

        // Check remaining elements
        for &val in &a[i..] {
            if val > result {
                result = val;
            }
        }

        result
    }

    #[target_feature(enable = "avx2")]
    unsafe fn min(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Start with first element broadcast to all lanes
        let mut vmin = _mm256_set1_ps(a[0]);

        // Process 8 elements at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            vmin = _mm256_min_ps(vmin, va);
            i += 8;
        }

        // Horizontal min: find minimum across all 8 lanes
        let mut result = {
            // Min of upper and lower 128-bit halves
            let min_halves = _mm_min_ps(
                _mm256_castps256_ps128(vmin),
                _mm256_extractf128_ps(vmin, 1),
            );
            // Horizontal min of 4 elements
            let temp = _mm_min_ps(min_halves, _mm_movehl_ps(min_halves, min_halves));
            let temp = _mm_min_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };

        // Check remaining elements
        for &val in &a[i..] {
            if val < result {
                result = val;
            }
        }

        result
    }

    #[target_feature(enable = "avx2")]
    unsafe fn argmax(a: &[f32]) -> usize {
        let len = a.len();
        let mut i = 0;

        // Track maximum value and index
        let mut max_value = a[0];
        let mut max_index = 0;

        // Initialize SIMD vectors with first element value and index 0
        let mut vmax = _mm256_set1_ps(a[0]);
        let mut vmax_idx = _mm256_set1_ps(0.0); // Track indices as floats

        // Initialize index vector [0, 1, 2, 3, 4, 5, 6, 7] and increment constant
        let mut vidx_current = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
        let vinc = _mm256_set1_ps(8.0);

        // Process 8 elements at a time with index tracking
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));

            // Compare: va > vmax (strict greater-than to preserve first occurrence)
            // _CMP_GT_OQ = 30 (ordered, quiet, greater-than)
            let mask = _mm256_cmp_ps::<30>(va, vmax);

            // Conditionally update max values and indices using blend
            vmax = _mm256_blendv_ps(vmax, va, mask);
            vmax_idx = _mm256_blendv_ps(vmax_idx, vidx_current, mask);

            // Increment index vector for next iteration
            vidx_current = _mm256_add_ps(vidx_current, vinc);
            i += 8;
        }

        // Horizontal reduction: find max value and its index across all 8 lanes
        let mut values = [0.0f32; 8];
        let mut indices = [0.0f32; 8];
        _mm256_storeu_ps(values.as_mut_ptr(), vmax);
        _mm256_storeu_ps(indices.as_mut_ptr(), vmax_idx);

        for lane in 0..8 {
            if values[lane] > max_value {
                max_value = values[lane];
                max_index = indices[lane] as usize;
            }
        }

        // Check remaining elements (scalar fallback)
        for (idx, &val) in a[i..].iter().enumerate() {
            if val > max_value {
                max_value = val;
                max_index = i + idx;
            }
        }

        max_index
    }

    #[target_feature(enable = "avx2")]
    unsafe fn argmin(a: &[f32]) -> usize {
        let len = a.len();
        let mut i = 0;

        // Track minimum value and index
        let mut min_value = a[0];
        let mut min_index = 0;

        // Initialize SIMD vectors with first element value and index 0
        let mut vmin = _mm256_set1_ps(a[0]);
        let mut vmin_idx = _mm256_set1_ps(0.0); // Track indices as floats

        // Initialize index vector [0, 1, 2, 3, 4, 5, 6, 7] and increment constant
        let mut vidx_current = _mm256_set_ps(7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
        let vinc = _mm256_set1_ps(8.0);

        // Process 8 elements at a time with index tracking
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));

            // Compare: va < vmin (strict less-than to preserve first occurrence)
            // _CMP_LT_OQ = 17 (ordered, quiet, less-than)
            let mask = _mm256_cmp_ps::<17>(va, vmin);

            // Conditionally update min values and indices using blend
            vmin = _mm256_blendv_ps(vmin, va, mask);
            vmin_idx = _mm256_blendv_ps(vmin_idx, vidx_current, mask);

            // Increment index vector for next iteration
            vidx_current = _mm256_add_ps(vidx_current, vinc);
            i += 8;
        }

        // Horizontal reduction: find min value and its index across all 8 lanes
        let mut values = [0.0f32; 8];
        let mut indices = [0.0f32; 8];
        _mm256_storeu_ps(values.as_mut_ptr(), vmin);
        _mm256_storeu_ps(indices.as_mut_ptr(), vmin_idx);

        for lane in 0..8 {
            if values[lane] < min_value {
                min_value = values[lane];
                min_index = indices[lane] as usize;
            }
        }

        // Check remaining elements (scalar fallback)
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

    #[target_feature(enable = "avx2")]
    unsafe fn norm_l2(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        // L2 norm is sqrt(dot(a, a))
        let sum_of_squares = Self::dot(a, a);
        sum_of_squares.sqrt()
    }

    #[target_feature(enable = "avx2")]
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        let len = a.len();
        let mut i = 0;

        // Accumulator for 8-way parallel accumulation
        let mut acc = _mm256_setzero_ps();

        // Create mask to clear sign bit (absolute value)
        let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));

        // Process 8 elements at a time using AVX2 (256-bit = 8 x f32)
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));

            // Compute absolute value by clearing sign bit
            let abs_va = _mm256_and_ps(va, sign_mask);

            // Accumulate
            acc = _mm256_add_ps(acc, abs_va);

            i += 8;
        }

        // Horizontal sum across all 8 lanes
        let mut result = {
            // Sum upper and lower 128-bit halves
            let sum_halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
            // Horizontal sum of 4 elements
            let temp = _mm_add_ps(sum_halves, _mm_movehl_ps(sum_halves, sum_halves));
            let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };

        // Handle remaining elements with scalar code
        for &val in &a[i..] {
            result += val.abs();
        }

        result
    }

    #[target_feature(enable = "avx2")]
    unsafe fn norm_linf(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        let len = a.len();
        let mut i = 0;

        // Accumulator for 8-way parallel max
        let mut max_vec = _mm256_setzero_ps();

        // Create mask to clear sign bit (absolute value)
        let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));

        // Process 8 elements at a time using AVX2 (256-bit = 8 x f32)
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));

            // Compute absolute value by clearing sign bit
            let abs_va = _mm256_and_ps(va, sign_mask);

            // Track maximum
            max_vec = _mm256_max_ps(max_vec, abs_va);

            i += 8;
        }

        // Horizontal max across all 8 lanes
        let mut result = {
            // Max of upper and lower 128-bit halves
            let max_halves = _mm_max_ps(
                _mm256_castps256_ps128(max_vec),
                _mm256_extractf128_ps(max_vec, 1),
            );
            // Horizontal max of 4 elements
            let temp = _mm_max_ps(max_halves, _mm_movehl_ps(max_halves, max_halves));
            let temp = _mm_max_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };

        // Handle remaining elements with scalar code
        for &val in &a[i..] {
            let abs_val = val.abs();
            if abs_val > result {
                result = abs_val;
            }
        }

        result
    }

    #[target_feature(enable = "avx2")]
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast scalar to all 8 lanes
        let scalar_vec = _mm256_set1_ps(scalar);

        // Process 8 elements at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vresult = _mm256_mul_ps(va, scalar_vec);
            _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 8;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] * scalar;
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn abs(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Create mask to clear sign bit (0x7FFFFFFF for all elements)
        let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));

        // Process 8 elements at a time using AVX2 (256-bit = 8 x f32)
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));

            // Compute absolute value by clearing sign bit
            let abs_va = _mm256_and_ps(va, sign_mask);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), abs_va);
            i += 8;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].abs();
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast min and max to all 8 lanes
        let min_vec = _mm256_set1_ps(min_val);
        let max_vec = _mm256_set1_ps(max_val);

        // Process 8 elements at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let clamped = _mm256_min_ps(_mm256_max_ps(va, min_vec), max_vec);
            _mm256_storeu_ps(result.as_mut_ptr().add(i), clamped);
            i += 8;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i].max(min_val).min(max_val);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast t to all 8 lanes
        let t_vec = _mm256_set1_ps(t);

        // Process 8 elements at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));

            // result = a + t * (b - a)
            // Using FMA: result = fma(t, (b - a), a) = t * (b - a) + a
            let diff = _mm256_sub_ps(vb, va);
            let vresult = _mm256_fmadd_ps(t_vec, diff, va);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 8;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] + t * (b[i] - a[i]);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 8 elements at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vc = _mm256_loadu_ps(c.as_ptr().add(i));

            // result = a * b + c
            // Using FMA: result = fma(a, b, c) = a * b + c
            let vresult = _mm256_fmadd_ps(va, vb, vc);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 8;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i] * b[i] + c[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Zero vector for max comparison
        let zero = _mm256_setzero_ps();

        // Process 8 elements at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));

            // ReLU: max(0, x)
            let vresult = _mm256_max_ps(zero, va);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 8;
        }

        // Handle remaining elements
        while i < len {
            result[i] = if a[i] > 0.0 { a[i] } else { 0.0 };
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn exp(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Constants for range reduction: exp(x) = 2^(x * log2(e)) = 2^k * 2^r
        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E); // 1.442695...
        let ln2 = _mm256_set1_ps(std::f32::consts::LN_2); // 0.693147...
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);

        // Polynomial coefficients for e^r approximation (Remez minimax on [-ln(2)/2, ln(2)/2])
        // e^r ≈ 1 + c1*r + c2*r^2 + c3*r^3 + c4*r^4 + c5*r^5 + c6*r^6
        // Coefficients from Cephes/SLEEF libraries optimized for f32
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(0.5);
        let c3 = _mm256_set1_ps(0.166_666_67); // 1/6
        let c4 = _mm256_set1_ps(0.041_666_668); // 1/24
        let c5 = _mm256_set1_ps(0.008_333_334); // 1/120
        let c6 = _mm256_set1_ps(0.001_388_889); // 1/720

        // Limits for overflow/underflow handling
        let exp_hi = _mm256_set1_ps(88.376_26); // ln(FLT_MAX)
        let exp_lo = _mm256_set1_ps(-87.336_55); // ln(FLT_MIN) approximately

        // Process 8 elements at a time
        while i + 8 <= len {
            let x = _mm256_loadu_ps(a.as_ptr().add(i));

            // Clamp x to avoid overflow/underflow
            let x = _mm256_max_ps(_mm256_min_ps(x, exp_hi), exp_lo);

            // Range reduction: x' = x * log2(e), then k = round(x'), r = x' - k
            let x_scaled = _mm256_mul_ps(x, log2e);

            // k = round(x_scaled) = floor(x_scaled + 0.5)
            let k = _mm256_floor_ps(_mm256_add_ps(x_scaled, half));

            // r = x - k * ln(2) (in original base e space)
            let r = _mm256_sub_ps(x, _mm256_mul_ps(k, ln2));

            // Polynomial approximation: e^r ≈ 1 + c1*r + c2*r^2 + c3*r^3 + c4*r^4 + c5*r^5 + c6*r^6
            // Use Horner's method: ((((((c6*r + c5)*r + c4)*r + c3)*r + c2)*r + c1)*r + 1)
            let mut p = c6;
            p = _mm256_fmadd_ps(p, r, c5);
            p = _mm256_fmadd_ps(p, r, c4);
            p = _mm256_fmadd_ps(p, r, c3);
            p = _mm256_fmadd_ps(p, r, c2);
            p = _mm256_fmadd_ps(p, r, c1);
            p = _mm256_fmadd_ps(p, r, one);

            // Scale by 2^k using IEEE754 exponent manipulation
            // 2^k is computed by adding k to the exponent bits
            let k_int = _mm256_cvtps_epi32(k);
            let k_shifted = _mm256_slli_epi32(k_int, 23); // shift to exponent position
            let scale = _mm256_castsi256_ps(_mm256_add_epi32(
                _mm256_castps_si256(one),
                k_shifted,
            ));

            // Final result: e^x = e^r * 2^k
            let vresult = _mm256_mul_ps(p, scale);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 8;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].exp();
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        // Use SIMD exp approximation with range reduction
        let len = a.len();
        let mut i = 0;

        // Constants for exp(-x) computation
        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);

        // Taylor series coefficients for e^r
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(0.5);
        let c3 = _mm256_set1_ps(0.166_666_67);
        let c4 = _mm256_set1_ps(0.041_666_668);
        let c5 = _mm256_set1_ps(0.008_333_334);
        let c6 = _mm256_set1_ps(0.001_388_889);

        // Limits for overflow/underflow
        let exp_hi = _mm256_set1_ps(88.376_26);
        let exp_lo = _mm256_set1_ps(-87.336_55);

        // Process 8 elements at a time
        while i + 8 <= len {
            let x = _mm256_loadu_ps(a.as_ptr().add(i));

            // Compute -x for exp(-x)
            let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);

            // Clamp to avoid overflow/underflow
            let neg_x = _mm256_max_ps(_mm256_min_ps(neg_x, exp_hi), exp_lo);

            // Range reduction: exp(-x) computation
            let x_scaled = _mm256_mul_ps(neg_x, log2e);
            let k = _mm256_floor_ps(_mm256_add_ps(x_scaled, half));
            let r = _mm256_sub_ps(neg_x, _mm256_mul_ps(k, ln2));

            // Polynomial approximation using Horner's method with FMA
            let mut p = c6;
            p = _mm256_fmadd_ps(p, r, c5);
            p = _mm256_fmadd_ps(p, r, c4);
            p = _mm256_fmadd_ps(p, r, c3);
            p = _mm256_fmadd_ps(p, r, c2);
            p = _mm256_fmadd_ps(p, r, c1);
            p = _mm256_fmadd_ps(p, r, one);

            // Scale by 2^k
            let k_int = _mm256_cvtps_epi32(k);
            let k_shifted = _mm256_slli_epi32(k_int, 23);
            let scale = _mm256_castsi256_ps(_mm256_add_epi32(
                _mm256_castps_si256(one),
                k_shifted,
            ));
            let exp_neg_x = _mm256_mul_ps(p, scale);

            // sigmoid = 1 / (1 + exp(-x))
            let denom = _mm256_add_ps(one, exp_neg_x);
            let sigmoid_result = _mm256_div_ps(one, denom);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), sigmoid_result);
            i += 8;
        }

        // Handle remaining elements with scalar code
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

    #[target_feature(enable = "avx2")]
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        // Use SIMD tanh via: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        let len = a.len();
        let mut i = 0;

        // GELU constants
        let sqrt_2_over_pi = _mm256_set1_ps(0.797_884_6);
        let coeff = _mm256_set1_ps(0.044715);
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);
        let two = _mm256_set1_ps(2.0);

        // Constants for exp computation
        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);

        // Taylor series coefficients for e^r
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(0.5);
        let c3 = _mm256_set1_ps(0.166_666_67);
        let c4 = _mm256_set1_ps(0.041_666_668);
        let c5 = _mm256_set1_ps(0.008_333_334);
        let c6 = _mm256_set1_ps(0.001_388_889);

        // Limits for overflow/underflow
        let exp_hi = _mm256_set1_ps(88.376_26);
        let exp_lo = _mm256_set1_ps(-87.336_55);

        // Process 8 elements at a time
        while i + 8 <= len {
            let x = _mm256_loadu_ps(a.as_ptr().add(i));

            // Compute inner = sqrt(2/π) * (x + 0.044715 * x³)
            let x2 = _mm256_mul_ps(x, x);
            let x3 = _mm256_mul_ps(x2, x);
            let inner_sum = _mm256_fmadd_ps(coeff, x3, x);
            let inner = _mm256_mul_ps(sqrt_2_over_pi, inner_sum);

            // Compute tanh(inner) = (exp(2*inner) - 1) / (exp(2*inner) + 1)
            let two_inner = _mm256_mul_ps(two, inner);

            // Clamp to avoid overflow/underflow
            let two_inner = _mm256_max_ps(_mm256_min_ps(two_inner, exp_hi), exp_lo);

            // Range reduction for exp(2*inner)
            let x_scaled = _mm256_mul_ps(two_inner, log2e);
            let k = _mm256_floor_ps(_mm256_add_ps(x_scaled, half));
            let r = _mm256_sub_ps(two_inner, _mm256_mul_ps(k, ln2));

            // Polynomial approximation using Horner's method with FMA
            let mut p = c6;
            p = _mm256_fmadd_ps(p, r, c5);
            p = _mm256_fmadd_ps(p, r, c4);
            p = _mm256_fmadd_ps(p, r, c3);
            p = _mm256_fmadd_ps(p, r, c2);
            p = _mm256_fmadd_ps(p, r, c1);
            p = _mm256_fmadd_ps(p, r, one);

            // Scale by 2^k
            let k_int = _mm256_cvtps_epi32(k);
            let k_shifted = _mm256_slli_epi32(k_int, 23);
            let scale = _mm256_castsi256_ps(_mm256_add_epi32(
                _mm256_castps_si256(one),
                k_shifted,
            ));
            let exp_2inner = _mm256_mul_ps(p, scale);

            // tanh = (exp(2x) - 1) / (exp(2x) + 1)
            let tanh_numer = _mm256_sub_ps(exp_2inner, one);
            let tanh_denom = _mm256_add_ps(exp_2inner, one);
            let tanh_result = _mm256_div_ps(tanh_numer, tanh_denom);

            // gelu = 0.5 * x * (1 + tanh)
            let one_plus_tanh = _mm256_add_ps(one, tanh_result);
            let gelu_result = _mm256_mul_ps(half, _mm256_mul_ps(x, one_plus_tanh));

            _mm256_storeu_ps(result.as_mut_ptr().add(i), gelu_result);
            i += 8;
        }

        // Handle remaining elements with scalar code
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        const COEFF: f32 = 0.044715;

        while i < len {
            let x = a[i];
            let x3 = x * x * x;
            let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            result[i] = 0.5 * x * (1.0 + inner.tanh());
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        // swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
        // Use SIMD exp approximation with range reduction
        let len = a.len();
        let mut i = 0;

        // Constants for exp(-x) computation
        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);

        // Taylor series coefficients for e^r
        let c1 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(0.5);
        let c3 = _mm256_set1_ps(0.166_666_67);
        let c4 = _mm256_set1_ps(0.041_666_668);
        let c5 = _mm256_set1_ps(0.008_333_334);
        let c6 = _mm256_set1_ps(0.001_388_889);

        // Limits for overflow/underflow
        let exp_hi = _mm256_set1_ps(88.376_26);
        let exp_lo = _mm256_set1_ps(-87.336_55);

        // Process 8 elements at a time
        while i + 8 <= len {
            let x = _mm256_loadu_ps(a.as_ptr().add(i));

            // Compute -x for exp(-x)
            let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);

            // Clamp to avoid overflow/underflow
            let neg_x = _mm256_max_ps(_mm256_min_ps(neg_x, exp_hi), exp_lo);

            // Range reduction: exp(-x) computation
            let x_scaled = _mm256_mul_ps(neg_x, log2e);
            let k = _mm256_floor_ps(_mm256_add_ps(x_scaled, half));
            let r = _mm256_sub_ps(neg_x, _mm256_mul_ps(k, ln2));

            // Polynomial approximation using Horner's method with FMA
            let mut p = c6;
            p = _mm256_fmadd_ps(p, r, c5);
            p = _mm256_fmadd_ps(p, r, c4);
            p = _mm256_fmadd_ps(p, r, c3);
            p = _mm256_fmadd_ps(p, r, c2);
            p = _mm256_fmadd_ps(p, r, c1);
            p = _mm256_fmadd_ps(p, r, one);

            // Scale by 2^k
            let k_int = _mm256_cvtps_epi32(k);
            let k_shifted = _mm256_slli_epi32(k_int, 23);
            let scale = _mm256_castsi256_ps(_mm256_add_epi32(
                _mm256_castps_si256(one),
                k_shifted,
            ));
            let exp_neg_x = _mm256_mul_ps(p, scale);

            // swish = x / (1 + exp(-x))
            let denom = _mm256_add_ps(one, exp_neg_x);
            let swish_result = _mm256_div_ps(x, denom);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), swish_result);
            i += 8;
        }

        // Handle remaining elements with scalar code
        while i < len {
            let x = a[i];
            result[i] = if x < -50.0 {
                0.0
            } else if x > 50.0 {
                x
            } else {
                x / (1.0 + (-x).exp())
            };
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_add() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2");
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = vec![0.0; 9];

        unsafe {
            Avx2Backend::add(&a, &b, &mut result);
        }

        assert_eq!(
            result,
            vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_mul() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2");
            return;
        }

        let a = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut result = vec![0.0; 9];

        unsafe {
            Avx2Backend::mul(&a, &b, &mut result);
        }

        assert_eq!(
            result,
            vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 90.0]
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_dot() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let result = unsafe { Avx2Backend::dot(&a, &b) };

        // 1*9 + 2*8 + 3*7 + 4*6 + 5*5 + 6*4 + 7*3 + 8*2 + 9*1
        // = 9 + 16 + 21 + 24 + 25 + 24 + 21 + 16 + 9 = 165
        assert!((result - 165.0).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_sum() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2");
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let result = unsafe { Avx2Backend::sum(&a) };

        assert!((result - 45.0).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_max() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2");
            return;
        }

        let a = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];

        let result = unsafe { Avx2Backend::max(&a) };

        assert_eq!(result, 9.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_min() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2");
            return;
        }

        let a = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];

        let result = unsafe { Avx2Backend::min(&a) };

        assert_eq!(result, 1.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];
        let b = vec![10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5];

        // Test add
        let mut avx2_result = vec![0.0; 10];
        let mut scalar_result = vec![0.0; 10];
        unsafe {
            Avx2Backend::add(&a, &b, &mut avx2_result);
            ScalarBackend::add(&a, &b, &mut scalar_result);
        }
        for (avx2, scalar) in avx2_result.iter().zip(&scalar_result) {
            assert!((avx2 - scalar).abs() < 1e-5);
        }

        // Test dot
        let (avx2_dot, scalar_dot) =
            unsafe { (Avx2Backend::dot(&a, &b), ScalarBackend::dot(&a, &b)) };
        assert!((avx2_dot - scalar_dot).abs() < 1e-3); // Relaxed tolerance for FMA

        // Test sum
        let (avx2_sum, scalar_sum) = unsafe { (Avx2Backend::sum(&a), ScalarBackend::sum(&a)) };
        assert!((avx2_sum - scalar_sum).abs() < 1e-3);

        // Test max
        let (avx2_max, scalar_max) = unsafe { (Avx2Backend::max(&a), ScalarBackend::max(&a)) };
        assert_eq!(avx2_max, scalar_max);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_relu() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        // Test with 16 elements (2 AVX2 registers of 8 f32s)
        let a = [
            -3.0, -1.0, 0.0, 1.0, 3.0, -2.0, 2.0, -0.5, -4.0, 4.0, -5.0, 5.0, 0.0, -0.1, 0.1, 10.0,
        ];
        let mut result = [0.0; 16];
        unsafe {
            Avx2Backend::relu(&a, &mut result);
        }
        let expected = [
            0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 5.0, 0.0, 0.0, 0.1, 10.0,
        ];
        assert_eq!(result, expected);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_relu_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, -2.0, 2.0, -4.0, 4.0];
        let mut avx2_result = [0.0; 11];
        let mut scalar_result = [0.0; 11];

        unsafe {
            Avx2Backend::relu(&a, &mut avx2_result);
            ScalarBackend::relu(&a, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_sigmoid_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [-10.0, -1.0, 0.0, 1.0, 10.0];
        let mut avx2_result = [0.0; 5];
        let mut scalar_result = [0.0; 5];

        unsafe {
            Avx2Backend::sigmoid(&a, &mut avx2_result);
            ScalarBackend::sigmoid(&a, &mut scalar_result);
        }

        for (avx2, scalar) in avx2_result.iter().zip(scalar_result.iter()) {
            assert!(
                (avx2 - scalar).abs() < 1e-6,
                "sigmoid mismatch: avx2={}, scalar={}",
                avx2,
                scalar
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_gelu_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut avx2_result = [0.0; 5];
        let mut scalar_result = [0.0; 5];

        unsafe {
            Avx2Backend::gelu(&a, &mut avx2_result);
            ScalarBackend::gelu(&a, &mut scalar_result);
        }

        for (avx2, scalar) in avx2_result.iter().zip(scalar_result.iter()) {
            assert!(
                (avx2 - scalar).abs() < 1e-5,
                "gelu mismatch: avx2={}, scalar={}",
                avx2,
                scalar
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_swish_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [-10.0, -1.0, 0.0, 1.0, 10.0];
        let mut avx2_result = [0.0; 5];
        let mut scalar_result = [0.0; 5];

        unsafe {
            Avx2Backend::swish(&a, &mut avx2_result);
            ScalarBackend::swish(&a, &mut scalar_result);
        }

        for (avx2, scalar) in avx2_result.iter().zip(scalar_result.iter()) {
            assert!(
                (avx2 - scalar).abs() < 1e-5,
                "swish mismatch: avx2={}, scalar={}",
                avx2,
                scalar
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_sub_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        unsafe {
            Avx2Backend::sub(&a, &b, &mut avx2_result);
            ScalarBackend::sub(&a, &b, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_div_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let b = [2.0, 4.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        unsafe {
            Avx2Backend::div(&a, &b, &mut avx2_result);
            ScalarBackend::div(&a, &b, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_scale_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let scalar = 2.5;
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        unsafe {
            Avx2Backend::scale(&a, scalar, &mut avx2_result);
            ScalarBackend::scale(&a, scalar, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_clamp_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0];
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        unsafe {
            Avx2Backend::clamp(&a, 5.0, 30.0, &mut avx2_result);
            ScalarBackend::clamp(&a, 5.0, 30.0, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_fma_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let c = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        unsafe {
            Avx2Backend::fma(&a, &b, &c, &mut avx2_result);
            ScalarBackend::fma(&a, &b, &c, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_lerp_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let b = [
            100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0,
        ];
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        unsafe {
            Avx2Backend::lerp(&a, &b, 0.25, &mut avx2_result);
            ScalarBackend::lerp(&a, &b, 0.25, &mut scalar_result);
        }

        for (avx2, scalar) in avx2_result.iter().zip(scalar_result.iter()) {
            assert!(
                (avx2 - scalar).abs() < 1e-5,
                "lerp mismatch: avx2={}, scalar={}",
                avx2,
                scalar
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_argmax_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [1.0, 5.0, 3.0, 10.0, 2.0, 8.0, 4.0, 9.0, 6.0];

        let avx2_result = unsafe { Avx2Backend::argmax(&a) };
        let scalar_result = unsafe { ScalarBackend::argmax(&a) };

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_argmin_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [5.0, 1.0, 3.0, 10.0, 2.0, 8.0, 4.0, 9.0, 6.0];

        let avx2_result = unsafe { Avx2Backend::argmin(&a) };
        let scalar_result = unsafe { ScalarBackend::argmin(&a) };

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_sum_kahan_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let avx2_result = unsafe { Avx2Backend::sum_kahan(&a) };
        let scalar_result = unsafe { ScalarBackend::sum_kahan(&a) };

        assert!((avx2_result - scalar_result).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_norm_l1_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0];

        let avx2_result = unsafe { Avx2Backend::norm_l1(&a) };
        let scalar_result = unsafe { ScalarBackend::norm_l1(&a) };

        assert!((avx2_result - scalar_result).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_norm_l2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [3.0, 4.0, 0.0, 0.0, 5.0, 12.0, 0.0, 8.0, 15.0];

        let avx2_result = unsafe { Avx2Backend::norm_l2(&a) };
        let scalar_result = unsafe { ScalarBackend::norm_l2(&a) };

        assert!((avx2_result - scalar_result).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_dot_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let avx2_result = unsafe { Avx2Backend::dot(&a, &b) };
        let scalar_result = unsafe { ScalarBackend::dot(&a, &b) };

        assert!((avx2_result - scalar_result).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_mul_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        unsafe {
            Avx2Backend::mul(&a, &b, &mut avx2_result);
            ScalarBackend::mul(&a, &b, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_add_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
        let b = [8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5];
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        unsafe {
            Avx2Backend::add(&a, &b, &mut avx2_result);
            ScalarBackend::add(&a, &b, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_sum_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let avx2_result = unsafe { Avx2Backend::sum(&a) };
        let scalar_result = unsafe { ScalarBackend::sum(&a) };

        assert!((avx2_result - scalar_result).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_max_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [1.0, 5.0, 3.0, 10.0, 2.0, 8.0, 4.0, 9.0, 6.0];

        let avx2_result = unsafe { Avx2Backend::max(&a) };
        let scalar_result = unsafe { ScalarBackend::max(&a) };

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_min_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        use super::super::scalar::ScalarBackend;

        let a = [5.0, 1.0, 3.0, 10.0, 2.0, 8.0, 4.0, 9.0, 6.0];

        let avx2_result = unsafe { Avx2Backend::min(&a) };
        let scalar_result = unsafe { ScalarBackend::min(&a) };

        assert_eq!(avx2_result, scalar_result);
    }
}
