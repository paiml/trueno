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
    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=32 for unrolled, 8 for remainder)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps) - no alignment requirement
    //
    // OPTIMIZATION: 4-accumulator unrolling for better ILP (Instruction Level Parallelism)
    // This matches llama.cpp's approach and provides 2.3x speedup over single accumulator.
    // FMA has ~4 cycle latency but can issue multiple per clock - 4 accumulators keep the
    // execution units fed while hiding the latency.
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // 4 independent accumulators for better ILP (llama.cpp style)
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        // Process 32 elements at a time (4 × 8) with 4 independent FMA chains
        while i + 32 <= len {
            let va0 = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb0 = _mm256_loadu_ps(b.as_ptr().add(i));
            let va1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
            let vb1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
            let va2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
            let vb2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
            let va3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
            let vb3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));

            // 4 independent FMA operations - no dependency chain between them
            acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
            acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
            acc2 = _mm256_fmadd_ps(va2, vb2, acc2);
            acc3 = _mm256_fmadd_ps(va3, vb3, acc3);

            i += 32;
        }

        // Handle 8-element chunks that don't fit in 32-element blocks
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc0 = _mm256_fmadd_ps(va, vb, acc0);
            i += 8;
        }

        // Combine all 4 accumulators
        let acc01 = _mm256_add_ps(acc0, acc1);
        let acc23 = _mm256_add_ps(acc2, acc3);
        let acc = _mm256_add_ps(acc01, acc23);

        // Horizontal sum: reduce 8 lanes to single value
        let mut result = {
            // Sum upper and lower 128-bit halves
            let sum_halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
            // Horizontal sum of 4 elements using faster movehl/shuffle
            let temp = _mm_add_ps(sum_halves, _mm_movehl_ps(sum_halves, sum_halves));
            let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };

        // Handle remaining elements with scalar code
        result += a[i..].iter().zip(&b[i..]).map(|(x, y)| x * y).sum::<f32>();

        result
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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
            let sum_halves = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
            // Horizontal sum of 4 elements using faster movehl/shuffle
            let temp = _mm_add_ps(sum_halves, _mm_movehl_ps(sum_halves, sum_halves));
            let temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
            _mm_cvtss_f32(temp)
        };

        // Handle remaining elements
        result += a[i..].iter().sum::<f32>();

        result
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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
            let max_halves =
                _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
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

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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
            let min_halves =
                _mm_min_ps(_mm256_castps256_ps128(vmin), _mm256_extractf128_ps(vmin, 1));
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

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Delegates to scalar implementation, no direct SIMD operations
    unsafe fn sum_kahan(a: &[f32]) -> f32 {
        // Kahan summation is inherently sequential, use scalar implementation
        super::scalar::ScalarBackend::sum_kahan(a)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
    unsafe fn norm_l2(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        // L2 norm is sqrt(dot(a, a))
        let sum_of_squares = Self::dot(a, a);
        sum_of_squares.sqrt()
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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
            let scale = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(one), k_shifted));

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

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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
            let scale = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(one), k_shifted));
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

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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
            let scale = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(one), k_shifted));
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

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
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
            let scale = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(one), k_shifted));
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

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=8 for AVX2)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used (_mm256_loadu_ps/_mm256_storeu_ps) - no alignment requirement
    unsafe fn tanh(a: &[f32], result: &mut [f32]) {
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        // Use SIMD exp approximation with range reduction
        let len = a.len();
        let mut i = 0;

        // Constants for exp(2x) computation
        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
        let half = _mm256_set1_ps(0.5);
        let one = _mm256_set1_ps(1.0);
        let two = _mm256_set1_ps(2.0);

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

            // Compute 2*x for exp(2*x)
            let two_x = _mm256_mul_ps(two, x);

            // Clamp to avoid overflow/underflow
            let two_x = _mm256_max_ps(_mm256_min_ps(two_x, exp_hi), exp_lo);

            // Range reduction: exp(2*x) computation
            let x_scaled = _mm256_mul_ps(two_x, log2e);
            let k = _mm256_floor_ps(_mm256_add_ps(x_scaled, half));
            let r = _mm256_sub_ps(two_x, _mm256_mul_ps(k, ln2));

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
            let scale = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(one), k_shifted));
            let exp_2x = _mm256_mul_ps(p, scale);

            // tanh = (exp(2x) - 1) / (exp(2x) + 1)
            let tanh_numer = _mm256_sub_ps(exp_2x, one);
            let tanh_denom = _mm256_add_ps(exp_2x, one);
            let tanh_result = _mm256_div_ps(tanh_numer, tanh_denom);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), tanh_result);
            i += 8;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].tanh();
            i += 1;
        }
    }

    #[inline]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure proper array access
    // 2. All pointers derived from valid slice references
    // 3. AVX2 intrinsics marked with #[target_feature]
    // 4. Unaligned loads/stores handle unaligned data correctly
    #[target_feature(enable = "avx2")]
    unsafe fn sqrt(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 8 elements at a time with AVX2
        while i + 8 <= len {
            let vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let sqrt_vec = _mm256_sqrt_ps(vec);
            _mm256_storeu_ps(result.as_mut_ptr().add(i), sqrt_vec);
            i += 8;
        }

        // Handle remaining elements
        while i < len {
            result[i] = a[i].sqrt();
            // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
            // 1. Loop bounds ensure proper array access
            // 2. All pointers derived from valid slice references
            // 3. AVX2 intrinsics marked with #[target_feature]
            // 4. Unaligned loads/stores handle unaligned data correctly
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 8 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores handle unaligned data correctly
    unsafe fn recip(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        let one = _mm256_set1_ps(1.0);
        while i + 8 <= len {
            let vec = _mm256_loadu_ps(a.as_ptr().add(i));
            let recip_vec = _mm256_div_ps(one, vec);
            _mm256_storeu_ps(result.as_mut_ptr().add(i), recip_vec);
            i += 8;
        }

        while i < len {
            result[i] = a[i].recip();
            i += 1;
        }
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 8 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used - no alignment requirement
    //
    // Natural logarithm implementation using range reduction:
    // For x = 2^k * m where m ∈ [1, 2):
    //   ln(x) = k*ln(2) + ln(m)
    //   ln(m) approximated using 7th-degree polynomial
    unsafe fn ln(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Constants for ln calculation
        let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
        let one = _mm256_set1_ps(1.0);

        // Use atanh transformation: ln(m) = 2 * atanh((m-1)/(m+1)) for m ∈ [1, 2)
        // Let u = (m-1)/(m+1), then u ∈ [0, 1/3]
        // ln(m) = 2 * u * (1 + u²/3 + u⁴/5 + u⁶/7 + u⁸/9 + u¹⁰/11)
        let two = _mm256_set1_ps(2.0);
        let c1 = _mm256_set1_ps(1.0);
        let c3 = _mm256_set1_ps(1.0 / 3.0);
        let c5 = _mm256_set1_ps(1.0 / 5.0);
        let c7 = _mm256_set1_ps(1.0 / 7.0);
        let c9 = _mm256_set1_ps(1.0 / 9.0);
        let c11 = _mm256_set1_ps(1.0 / 11.0);

        let mantissa_mask = _mm256_set1_epi32(0x007F_FFFF_u32 as i32);
        let exponent_127 = _mm256_set1_epi32(127 << 23);

        // Process 8 elements at a time
        while i + 8 <= len {
            let x = _mm256_loadu_ps(a.as_ptr().add(i));
            let x_int = _mm256_castps_si256(x);

            // Extract exponent k
            let exp_biased = _mm256_srli_epi32(x_int, 23);
            let exp_biased_masked = _mm256_and_si256(exp_biased, _mm256_set1_epi32(0xFF));
            let k_int = _mm256_sub_epi32(exp_biased_masked, _mm256_set1_epi32(127));
            let k = _mm256_cvtepi32_ps(k_int);

            // Extract mantissa m ∈ [1, 2)
            let mantissa_bits = _mm256_and_si256(x_int, mantissa_mask);
            let m_int = _mm256_or_si256(mantissa_bits, exponent_127);
            let m = _mm256_castsi256_ps(m_int);

            // Compute u = (m-1)/(m+1)
            let m_minus_1 = _mm256_sub_ps(m, one);
            let m_plus_1 = _mm256_add_ps(m, one);
            let u = _mm256_div_ps(m_minus_1, m_plus_1);
            let u2 = _mm256_mul_ps(u, u);

            // P(u²) = 1 + u²*(1/3 + u²*(1/5 + u²*(1/7 + u²*(1/9 + u²*1/11))))
            let p = _mm256_fmadd_ps(c11, u2, c9);
            let p = _mm256_fmadd_ps(p, u2, c7);
            let p = _mm256_fmadd_ps(p, u2, c5);
            let p = _mm256_fmadd_ps(p, u2, c3);
            let p = _mm256_fmadd_ps(p, u2, c1);

            // ln(m) = 2 * u * P(u²)
            let ln_m = _mm256_mul_ps(two, _mm256_mul_ps(u, p));

            // ln(x) = k*ln(2) + ln(m)
            let result_vec = _mm256_fmadd_ps(k, ln2, ln_m);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
            i += 8;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].ln();
            i += 1;
        }
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 8 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used - no alignment requirement
    //
    // Base-2 logarithm implementation using range reduction:
    // For x = 2^k * m where m ∈ [1, 2):
    //   log2(x) = k + log2(m)
    //   log2(m) = ln(m) / ln(2)
    unsafe fn log2(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        let inv_ln2 = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let one = _mm256_set1_ps(1.0);

        let two = _mm256_set1_ps(2.0);
        let c1 = _mm256_set1_ps(1.0);
        let c3 = _mm256_set1_ps(1.0 / 3.0);
        let c5 = _mm256_set1_ps(1.0 / 5.0);
        let c7 = _mm256_set1_ps(1.0 / 7.0);
        let c9 = _mm256_set1_ps(1.0 / 9.0);
        let c11 = _mm256_set1_ps(1.0 / 11.0);

        let mantissa_mask = _mm256_set1_epi32(0x007F_FFFF_u32 as i32);
        let exponent_127 = _mm256_set1_epi32(127 << 23);

        while i + 8 <= len {
            let x = _mm256_loadu_ps(a.as_ptr().add(i));
            let x_int = _mm256_castps_si256(x);

            let exp_biased = _mm256_srli_epi32(x_int, 23);
            let exp_biased_masked = _mm256_and_si256(exp_biased, _mm256_set1_epi32(0xFF));
            let k_int = _mm256_sub_epi32(exp_biased_masked, _mm256_set1_epi32(127));
            let k = _mm256_cvtepi32_ps(k_int);

            let mantissa_bits = _mm256_and_si256(x_int, mantissa_mask);
            let m_int = _mm256_or_si256(mantissa_bits, exponent_127);
            let m = _mm256_castsi256_ps(m_int);

            // atanh transformation
            let m_minus_1 = _mm256_sub_ps(m, one);
            let m_plus_1 = _mm256_add_ps(m, one);
            let u = _mm256_div_ps(m_minus_1, m_plus_1);
            let u2 = _mm256_mul_ps(u, u);

            let p = _mm256_fmadd_ps(c11, u2, c9);
            let p = _mm256_fmadd_ps(p, u2, c7);
            let p = _mm256_fmadd_ps(p, u2, c5);
            let p = _mm256_fmadd_ps(p, u2, c3);
            let p = _mm256_fmadd_ps(p, u2, c1);

            let ln_m = _mm256_mul_ps(two, _mm256_mul_ps(u, p));

            let log2_m = _mm256_mul_ps(ln_m, inv_ln2);
            let result_vec = _mm256_add_ps(k, log2_m);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
            i += 8;
        }

        while i < len {
            result[i] = a[i].log2();
            i += 1;
        }
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 8 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used - no alignment requirement
    //
    // Base-10 logarithm implementation using range reduction:
    // For x = 2^k * m where m ∈ [1, 2):
    //   log10(x) = k*log10(2) + log10(m)
    //   log10(m) = ln(m) / ln(10)
    unsafe fn log10(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        let log10_2 = _mm256_set1_ps(std::f32::consts::LOG10_2);
        let inv_ln10 = _mm256_set1_ps(1.0 / std::f32::consts::LN_10);
        let one = _mm256_set1_ps(1.0);

        let two = _mm256_set1_ps(2.0);
        let c1 = _mm256_set1_ps(1.0);
        let c3 = _mm256_set1_ps(1.0 / 3.0);
        let c5 = _mm256_set1_ps(1.0 / 5.0);
        let c7 = _mm256_set1_ps(1.0 / 7.0);
        let c9 = _mm256_set1_ps(1.0 / 9.0);
        let c11 = _mm256_set1_ps(1.0 / 11.0);

        let mantissa_mask = _mm256_set1_epi32(0x007F_FFFF_u32 as i32);
        let exponent_127 = _mm256_set1_epi32(127 << 23);

        while i + 8 <= len {
            let x = _mm256_loadu_ps(a.as_ptr().add(i));
            let x_int = _mm256_castps_si256(x);

            let exp_biased = _mm256_srli_epi32(x_int, 23);
            let exp_biased_masked = _mm256_and_si256(exp_biased, _mm256_set1_epi32(0xFF));
            let k_int = _mm256_sub_epi32(exp_biased_masked, _mm256_set1_epi32(127));
            let k = _mm256_cvtepi32_ps(k_int);

            let mantissa_bits = _mm256_and_si256(x_int, mantissa_mask);
            let m_int = _mm256_or_si256(mantissa_bits, exponent_127);
            let m = _mm256_castsi256_ps(m_int);

            // atanh transformation
            let m_minus_1 = _mm256_sub_ps(m, one);
            let m_plus_1 = _mm256_add_ps(m, one);
            let u = _mm256_div_ps(m_minus_1, m_plus_1);
            let u2 = _mm256_mul_ps(u, u);

            let p = _mm256_fmadd_ps(c11, u2, c9);
            let p = _mm256_fmadd_ps(p, u2, c7);
            let p = _mm256_fmadd_ps(p, u2, c5);
            let p = _mm256_fmadd_ps(p, u2, c3);
            let p = _mm256_fmadd_ps(p, u2, c1);

            let ln_m = _mm256_mul_ps(two, _mm256_mul_ps(u, p));

            let log10_m = _mm256_mul_ps(ln_m, inv_ln10);
            let result_vec = _mm256_fmadd_ps(k, log10_2, log10_m);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
            i += 8;
        }

        while i < len {
            result[i] = a[i].log10();
            i += 1;
        }
    }

    // Trigonometric functions currently use scalar implementations
    // Full SIMD trig functions require complex range reduction and are left for future work

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Delegates to scalar implementation, no direct SIMD operations
    unsafe fn sin(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::sin(a, result);
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Delegates to scalar implementation, no direct SIMD operations
    unsafe fn cos(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::cos(a, result);
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Delegates to scalar implementation, no direct SIMD operations
    unsafe fn tan(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::tan(a, result);
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 8 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn floor(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 8 elements at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vresult = _mm256_floor_ps(va);
            _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 8;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].floor();
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 8 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn ceil(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 8 elements at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vresult = _mm256_ceil_ps(va);
            _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 8;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].ceil();
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 8 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX2 intrinsics marked with #[target_feature(enable = "avx2")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn round(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Rust's .round() rounds ties away from zero, but SIMD round modes don't support this.
        // Implement manually: round(x) = sign(x) * floor(abs(x) + 0.5)
        let half = _mm256_set1_ps(0.5);
        let sign_mask = _mm256_set1_ps(f32::from_bits(0x8000_0000)); // Sign bit only
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF)); // All except sign bit

        // Process 8 elements at a time
        while i + 8 <= len {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));

            // Extract sign and absolute value
            let sign = _mm256_and_ps(va, sign_mask);
            let abs_val = _mm256_and_ps(va, abs_mask);

            // Round away from zero: floor(abs(x) + 0.5) * sign(x)
            let shifted = _mm256_add_ps(abs_val, half);
            let rounded_abs = _mm256_floor_ps(shifted);
            let vresult = _mm256_or_ps(rounded_abs, sign);

            _mm256_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 8;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].round();
            i += 1;
        }
    }
}
