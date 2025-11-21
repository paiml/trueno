//! AVX-512 backend implementation (x86_64 advanced SIMD)
//!
//! This backend uses AVX-512 intrinsics for 512-bit SIMD operations.
//! AVX-512 is available on Intel Skylake-X/Sapphire Rapids (2017+) and AMD Zen 4 (2022+) CPUs.
//!
//! # Performance
//!
//! Expected speedup: 16x for operations on f32 vectors (16 elements per register)
//! This provides 2x improvement over AVX2 (8 elements) and ~16x over scalar.
//!
//! # Safety
//!
//! All AVX-512 intrinsics are marked `unsafe` by Rust. This module carefully isolates
//! all unsafe code and verifies correctness through comprehensive testing.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::VectorBackend;

/// AVX-512 backend (512-bit SIMD for x86_64)
pub struct Avx512Backend;

impl VectorBackend for Avx512Backend {
    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=16 for AVX-512)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used (_mm512_loadu_ps/_mm512_storeu_ps) - no alignment requirement
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 16 elements at a time using AVX-512 (512-bit = 16 x f32)
        while i + 16 <= len {
            // Load 16 floats from a and b
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));

            // Add them
            let vresult = _mm512_add_ps(va, vb);

            // Store result
            _mm512_storeu_ps(result.as_mut_ptr().add(i), vresult);

            i += 16;
        }

        // Handle remaining elements with scalar code
        for j in i..len {
            result[j] = a[j] + b[j];
        }
    }

    // Stub implementations for remaining methods - will implement in future phases
    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 16 elements at a time (512-bit = 16 x f32)
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            let vresult = _mm512_sub_ps(va, vb);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 16;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i] - b[i];
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 16 elements at a time (512-bit = 16 x f32)
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            let vresult = _mm512_mul_ps(va, vb);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 16;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i] * b[i];
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 16 elements at a time (512-bit = 16 x f32)
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            let vresult = _mm512_div_ps(va, vb);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 16;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i] / b[i];
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=16 for AVX-512)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads used (_mm512_loadu_ps) - no alignment requirement
    // 5. FMA intrinsic (_mm512_fmadd_ps) provides better performance and numerical accuracy
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Accumulator for 16-way parallel accumulation
        let mut acc = _mm512_setzero_ps();

        // Process 16 elements at a time with FMA
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));

            // Fused multiply-add: acc = acc + (va * vb)
            // This is a single instruction on AVX-512 hardware
            acc = _mm512_fmadd_ps(va, vb, acc);

            i += 16;
        }

        // Horizontal sum: reduce 16 lanes to single value
        // AVX-512 provides a convenient intrinsic for this
        let mut result = _mm512_reduce_add_ps(acc);

        // Handle remaining elements with scalar code
        result += a[i..].iter().zip(&b[i..]).map(|(x, y)| x * y).sum::<f32>();

        result
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=16 for AVX-512)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads used (_mm512_loadu_ps) - no alignment requirement
    unsafe fn sum(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Accumulator for 16-way parallel accumulation
        let mut acc = _mm512_setzero_ps();

        // Process 16 elements at a time
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            acc = _mm512_add_ps(acc, va);
            i += 16;
        }

        // Horizontal sum: reduce 16 lanes to single value
        // AVX-512 provides a convenient intrinsic for this
        let mut result = _mm512_reduce_add_ps(acc);

        // Handle remaining elements with scalar code
        result += a[i..].iter().sum::<f32>();

        result
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=16 for AVX-512)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads used (_mm512_loadu_ps) - no alignment requirement
    unsafe fn max(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Start with first element broadcast to all 16 lanes
        let mut vmax = _mm512_set1_ps(a[0]);

        // Process 16 elements at a time
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            vmax = _mm512_max_ps(vmax, va);
            i += 16;
        }

        // Horizontal max: find maximum across all 16 lanes
        // AVX-512 provides a convenient intrinsic for this
        let mut result = _mm512_reduce_max_ps(vmax);

        // Check remaining elements
        for &val in &a[i..] {
            if val > result {
                result = val;
            }
        }

        result
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + N <= len` before calling `.add(i)` (N=16 for AVX-512)
    // 2. All pointers derived from valid slice references with sufficient backing storage
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads used (_mm512_loadu_ps) - no alignment requirement
    unsafe fn min(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Start with first element broadcast to all 16 lanes
        let mut vmin = _mm512_set1_ps(a[0]);

        // Process 16 elements at a time
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            vmin = _mm512_min_ps(vmin, va);
            i += 16;
        }

        // Horizontal min: find minimum across all 16 lanes
        // AVX-512 provides a convenient intrinsic for this
        let mut result = _mm512_reduce_min_ps(vmin);

        // Check remaining elements
        for &val in &a[i..] {
            if val < result {
                result = val;
            }
        }

        result
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure proper array access
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature]
    // 4. Unaligned loads/stores handle unaligned data correctly
    unsafe fn argmax(a: &[f32]) -> usize {
        if a.is_empty() {
            return 0;
        }

        let len = a.len();
        let mut i = 0;

        // Start with first element broadcast to all 16 lanes
        let mut vmax = _mm512_set1_ps(a[0]);

        // Process 16 elements at a time to find max value
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            vmax = _mm512_max_ps(vmax, va);
            i += 16;
        }

        // Horizontal max: find maximum value across all 16 lanes
        let mut max_val = _mm512_reduce_max_ps(vmax);

        // Check remaining elements
        for &val in &a[i..] {
            if val > max_val {
                max_val = val;
            }
        }

        // Find the index of the first occurrence of max_val
        // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
        // 1. Loop bounds ensure proper array access
        // 2. All pointers derived from valid slice references
        // 3. AVX-512 intrinsics marked with #[target_feature]
        // 4. Unaligned loads/stores handle unaligned data correctly
        a.iter().position(|&x| x == max_val).unwrap_or(0)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn argmin(a: &[f32]) -> usize {
        if a.is_empty() {
            return 0;
        }

        let len = a.len();
        let mut i = 0;

        // Start with first element broadcast to all 16 lanes
        let mut vmin = _mm512_set1_ps(a[0]);

        // Process 16 elements at a time to find min value
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            vmin = _mm512_min_ps(vmin, va);
            i += 16;
        }

        // Horizontal min: find minimum value across all 16 lanes
        let mut min_val = _mm512_reduce_min_ps(vmin);

        // Check remaining elements
        for &val in &a[i..] {
            if val < min_val {
                // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
                // 1. Loop bounds ensure proper array access
                // 2. All pointers derived from valid slice references
                // 3. AVX-512 intrinsics marked with #[target_feature]
                // 4. Unaligned loads/stores handle unaligned data correctly
                min_val = val;
            }
        }

        // Find the index of the first occurrence of min_val
        a.iter().position(|&x| x == min_val).unwrap_or(0)
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Uses scalar implementation, no unsafe operations
    unsafe fn sum_kahan(a: &[f32]) -> f32 {
        // Scalar fallback (AVX-512 optimization pending)
        let mut sum = 0.0;
        let mut c = 0.0;
        for &x in a {
            let y = x - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads used (_mm512_loadu_ps) - no alignment requirement
    unsafe fn norm_l2(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        let len = a.len();
        let mut i = 0;

        // Accumulator for sum of squares
        let mut acc = _mm512_setzero_ps();

        // Process 16 elements at a time: compute x^2 and accumulate
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let squared = _mm512_mul_ps(va, va);
            acc = _mm512_add_ps(acc, squared);
            i += 16;
        }

        // Horizontal sum: reduce 16 lanes to single value
        let mut sum_of_squares = _mm512_reduce_add_ps(acc);

        // Handle remaining elements with scalar code
        for &val in &a[i..] {
            sum_of_squares += val * val;
        }

        sum_of_squares.sqrt()
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads used - no alignment requirement
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Sign bit mask for abs
        let sign_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF));

        // Accumulator for sum
        let mut acc = _mm512_setzero_ps();

        // Process 16 elements at a time
        // norm_l1 = sum(|x|)
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vabs = _mm512_and_ps(va, sign_mask);
            acc = _mm512_add_ps(acc, vabs);
            i += 16;
        }

        // Horizontal sum: reduce 16 lanes to single value
        let mut result = _mm512_reduce_add_ps(acc);

        // Handle remaining elements with scalar code
        for &val in &a[i..] {
            result += val.abs();
        }

        result
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads used - no alignment requirement
    unsafe fn norm_linf(a: &[f32]) -> f32 {
        let len = a.len();
        let mut i = 0;

        // Sign bit mask for abs
        let sign_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF));

        // Accumulator for max
        let mut acc = _mm512_setzero_ps();

        // Process 16 elements at a time
        // norm_linf = max(|x|)
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vabs = _mm512_and_ps(va, sign_mask);
            acc = _mm512_max_ps(acc, vabs);
            i += 16;
        }

        // Horizontal max: reduce 16 lanes to single value
        let mut result = _mm512_reduce_max_ps(acc);

        // Handle remaining elements with scalar code
        for &val in &a[i..] {
            result = result.max(val.abs());
        }

        result
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used (_mm512_loadu_ps/_mm512_storeu_ps) - no alignment requirement
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast scalar to all 16 lanes
        let scalar_vec = _mm512_set1_ps(scalar);

        // Process 16 elements at a time (512-bit = 16 x f32)
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vresult = _mm512_mul_ps(va, scalar_vec);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 16;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i] * scalar;
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn abs(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Sign bit mask: 0x7FFFFFFF clears sign bit (keeps magnitude)
        let sign_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF));

        // Process 16 elements at a time
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vabs = _mm512_and_ps(va, sign_mask);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), vabs);
            i += 16;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].abs();
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast min/max to all 16 lanes
        let vmin = _mm512_set1_ps(min_val);
        let vmax = _mm512_set1_ps(max_val);

        // Process 16 elements at a time
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            // clamp(x, min, max) = min(max(x, min), max)
            let vclamped = _mm512_min_ps(_mm512_max_ps(va, vmin), vmax);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), vclamped);
            i += 16;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].clamp(min_val, max_val);
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Broadcast t to all 16 lanes
        let t_vec = _mm512_set1_ps(t);

        // Process 16 elements at a time
        // lerp(a, b, t) = a + t * (b - a)
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));

            // Compute: b - a
            let vdiff = _mm512_sub_ps(vb, va);

            // Compute: t * (b - a)
            let vscaled = _mm512_mul_ps(t_vec, vdiff);

            // Compute: a + t * (b - a)
            let vresult = _mm512_add_ps(va, vscaled);

            _mm512_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 16;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i] + t * (b[i] - a[i]);
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    // 5. FMA instruction is part of AVX-512F (foundation)
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 16 elements at a time
        // fma(a, b, c) = a * b + c
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            let vc = _mm512_loadu_ps(c.as_ptr().add(i));

            // Single fused multiply-add instruction (higher precision + faster)
            let vresult = _mm512_fmadd_ps(va, vb, vc);

            _mm512_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 16;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].mul_add(b[i], c[i]);
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Create zero vector for comparison
        let zero = _mm512_setzero_ps();

        // Process 16 elements at a time
        // relu(x) = max(x, 0)
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vresult = _mm512_max_ps(va, zero);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 16;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].max(0.0);
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn exp(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Constants for range reduction: exp(x) = 2^(x * log2(e)) = 2^k * 2^r
        let log2e = _mm512_set1_ps(std::f32::consts::LOG2_E); // 1.442695...
        let ln2 = _mm512_set1_ps(std::f32::consts::LN_2); // 0.693147...
        let half = _mm512_set1_ps(0.5);
        let one = _mm512_set1_ps(1.0);

        // Polynomial coefficients for e^r approximation (Remez minimax on [-ln(2)/2, ln(2)/2])
        // e^r ≈ 1 + c1*r + c2*r^2 + c3*r^3 + c4*r^4 + c5*r^5 + c6*r^6
        let c1 = _mm512_set1_ps(1.0);
        let c2 = _mm512_set1_ps(0.5);
        let c3 = _mm512_set1_ps(0.166_666_67); // 1/6
        let c4 = _mm512_set1_ps(0.041_666_668); // 1/24
        let c5 = _mm512_set1_ps(0.008_333_334); // 1/120
        let c6 = _mm512_set1_ps(0.001_388_889); // 1/720

        // Limits for overflow/underflow handling
        let exp_hi = _mm512_set1_ps(88.376_26); // ln(FLT_MAX)
        let exp_lo = _mm512_set1_ps(-87.336_55); // ln(FLT_MIN) approximately

        // Process 16 elements at a time
        while i + 16 <= len {
            let x = _mm512_loadu_ps(a.as_ptr().add(i));

            // Clamp x to avoid overflow/underflow
            let x = _mm512_max_ps(_mm512_min_ps(x, exp_hi), exp_lo);

            // Range reduction: x' = x * log2(e), then k = round(x'), r = x' - k
            let x_scaled = _mm512_mul_ps(x, log2e);

            // k = round(x_scaled) = floor(x_scaled + 0.5)
            // AVX512 uses roundscale instead of floor: mode 0x09 = floor
            let k = _mm512_roundscale_ps(_mm512_add_ps(x_scaled, half), 0x09);

            // r = x - k * ln(2) (in original base e space)
            let r = _mm512_sub_ps(x, _mm512_mul_ps(k, ln2));

            // Polynomial approximation: e^r ≈ 1 + c1*r + c2*r^2 + c3*r^3 + c4*r^4 + c5*r^5 + c6*r^6
            // Use Horner's method: ((((((c6*r + c5)*r + c4)*r + c3)*r + c2)*r + c1)*r + 1)
            let mut p = c6;
            p = _mm512_fmadd_ps(p, r, c5);
            p = _mm512_fmadd_ps(p, r, c4);
            p = _mm512_fmadd_ps(p, r, c3);
            p = _mm512_fmadd_ps(p, r, c2);
            p = _mm512_fmadd_ps(p, r, c1);
            p = _mm512_fmadd_ps(p, r, one);

            // Scale by 2^k using IEEE754 exponent manipulation
            // 2^k is computed by adding k to the exponent bits
            let k_int = _mm512_cvtps_epi32(k);
            let k_shifted = _mm512_slli_epi32(k_int, 23); // shift to exponent position
            let scale = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(one), k_shifted));

            // Final result: e^x = e^r * 2^k
            let vresult = _mm512_mul_ps(p, scale);

            _mm512_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 16;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].exp();
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        // Use SIMD exp approximation with range reduction
        let len = a.len();
        let mut i = 0;

        // Constants for exp(-x) computation
        let log2e = _mm512_set1_ps(std::f32::consts::LOG2_E);
        let ln2 = _mm512_set1_ps(std::f32::consts::LN_2);
        let half = _mm512_set1_ps(0.5);
        let one = _mm512_set1_ps(1.0);

        // Polynomial coefficients for e^r
        let c1 = _mm512_set1_ps(1.0);
        let c2 = _mm512_set1_ps(0.5);
        let c3 = _mm512_set1_ps(0.166_666_67);
        let c4 = _mm512_set1_ps(0.041_666_668);
        let c5 = _mm512_set1_ps(0.008_333_334);
        let c6 = _mm512_set1_ps(0.001_388_889);

        // Limits for overflow/underflow
        let exp_hi = _mm512_set1_ps(88.376_26);
        let exp_lo = _mm512_set1_ps(-87.336_55);

        // Process 16 elements at a time
        while i + 16 <= len {
            let x = _mm512_loadu_ps(a.as_ptr().add(i));

            // Compute -x for exp(-x)
            let neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);

            // Clamp to avoid overflow/underflow
            let neg_x = _mm512_max_ps(_mm512_min_ps(neg_x, exp_hi), exp_lo);

            // Range reduction: exp(-x) computation
            let x_scaled = _mm512_mul_ps(neg_x, log2e);
            let k = _mm512_roundscale_ps(_mm512_add_ps(x_scaled, half), 0x09);
            let r = _mm512_sub_ps(neg_x, _mm512_mul_ps(k, ln2));

            // Polynomial approximation using Horner's method with FMA
            let mut p = c6;
            p = _mm512_fmadd_ps(p, r, c5);
            p = _mm512_fmadd_ps(p, r, c4);
            p = _mm512_fmadd_ps(p, r, c3);
            p = _mm512_fmadd_ps(p, r, c2);
            p = _mm512_fmadd_ps(p, r, c1);
            p = _mm512_fmadd_ps(p, r, one);

            // Scale by 2^k
            let k_int = _mm512_cvtps_epi32(k);
            let k_shifted = _mm512_slli_epi32(k_int, 23);
            let scale = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(one), k_shifted));
            let exp_neg_x = _mm512_mul_ps(p, scale);

            // sigmoid = 1 / (1 + exp(-x))
            let denom = _mm512_add_ps(one, exp_neg_x);
            let sigmoid_result = _mm512_div_ps(one, denom);

            _mm512_storeu_ps(result.as_mut_ptr().add(i), sigmoid_result);
            i += 16;
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
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        let len = a.len();
        let mut i = 0;

        let sqrt_2_over_pi = _mm512_set1_ps(0.797_884_6);
        let coeff = _mm512_set1_ps(0.044715);
        let half = _mm512_set1_ps(0.5);
        let one = _mm512_set1_ps(1.0);
        let two = _mm512_set1_ps(2.0);

        let log2e = _mm512_set1_ps(std::f32::consts::LOG2_E);
        let ln2 = _mm512_set1_ps(std::f32::consts::LN_2);

        let c1 = _mm512_set1_ps(1.0);
        let c2 = _mm512_set1_ps(0.5);
        let c3 = _mm512_set1_ps(0.166_666_67);
        let c4 = _mm512_set1_ps(0.041_666_668);
        let c5 = _mm512_set1_ps(0.008_333_334);
        let c6 = _mm512_set1_ps(0.001_388_889);

        let exp_hi = _mm512_set1_ps(88.376_26);
        let exp_lo = _mm512_set1_ps(-87.336_55);

        while i + 16 <= len {
            let x = _mm512_loadu_ps(a.as_ptr().add(i));

            // Compute inner = sqrt(2/π) * (x + 0.044715 * x³)
            let x2 = _mm512_mul_ps(x, x);
            let x3 = _mm512_mul_ps(x2, x);
            let inner_sum = _mm512_fmadd_ps(coeff, x3, x);
            let inner = _mm512_mul_ps(sqrt_2_over_pi, inner_sum);

            // Compute tanh(inner) = (exp(2*inner) - 1) / (exp(2*inner) + 1)
            let two_inner = _mm512_mul_ps(two, inner);
            let two_inner = _mm512_max_ps(_mm512_min_ps(two_inner, exp_hi), exp_lo);

            let x_scaled = _mm512_mul_ps(two_inner, log2e);
            let k = _mm512_roundscale_ps(_mm512_add_ps(x_scaled, half), 0x09);
            let r = _mm512_sub_ps(two_inner, _mm512_mul_ps(k, ln2));

            let mut p = c6;
            p = _mm512_fmadd_ps(p, r, c5);
            p = _mm512_fmadd_ps(p, r, c4);
            p = _mm512_fmadd_ps(p, r, c3);
            p = _mm512_fmadd_ps(p, r, c2);
            p = _mm512_fmadd_ps(p, r, c1);
            p = _mm512_fmadd_ps(p, r, one);

            let k_int = _mm512_cvtps_epi32(k);
            let k_shifted = _mm512_slli_epi32(k_int, 23);
            let scale = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(one), k_shifted));
            let exp_2inner = _mm512_mul_ps(p, scale);

            // tanh = (exp(2x) - 1) / (exp(2x) + 1)
            let tanh_numer = _mm512_sub_ps(exp_2inner, one);
            let tanh_denom = _mm512_add_ps(exp_2inner, one);
            let tanh_result = _mm512_div_ps(tanh_numer, tanh_denom);

            // gelu = 0.5 * x * (1 + tanh)
            let one_plus_tanh = _mm512_add_ps(one, tanh_result);
            let gelu_result = _mm512_mul_ps(half, _mm512_mul_ps(x, one_plus_tanh));

            _mm512_storeu_ps(result.as_mut_ptr().add(i), gelu_result);
            i += 16;
        }

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
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        // swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
        let len = a.len();
        let mut i = 0;

        let log2e = _mm512_set1_ps(std::f32::consts::LOG2_E);
        let ln2 = _mm512_set1_ps(std::f32::consts::LN_2);
        let half = _mm512_set1_ps(0.5);
        let one = _mm512_set1_ps(1.0);

        let c1 = _mm512_set1_ps(1.0);
        let c2 = _mm512_set1_ps(0.5);
        let c3 = _mm512_set1_ps(0.166_666_67);
        let c4 = _mm512_set1_ps(0.041_666_668);
        let c5 = _mm512_set1_ps(0.008_333_334);
        let c6 = _mm512_set1_ps(0.001_388_889);

        let exp_hi = _mm512_set1_ps(88.376_26);
        let exp_lo = _mm512_set1_ps(-87.336_55);

        while i + 16 <= len {
            let x = _mm512_loadu_ps(a.as_ptr().add(i));
            let neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
            let neg_x = _mm512_max_ps(_mm512_min_ps(neg_x, exp_hi), exp_lo);

            let x_scaled = _mm512_mul_ps(neg_x, log2e);
            let k = _mm512_roundscale_ps(_mm512_add_ps(x_scaled, half), 0x09);
            let r = _mm512_sub_ps(neg_x, _mm512_mul_ps(k, ln2));

            let mut p = c6;
            p = _mm512_fmadd_ps(p, r, c5);
            p = _mm512_fmadd_ps(p, r, c4);
            p = _mm512_fmadd_ps(p, r, c3);
            p = _mm512_fmadd_ps(p, r, c2);
            p = _mm512_fmadd_ps(p, r, c1);
            p = _mm512_fmadd_ps(p, r, one);

            let k_int = _mm512_cvtps_epi32(k);
            let k_shifted = _mm512_slli_epi32(k_int, 23);
            let scale = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(one), k_shifted));
            let exp_neg_x = _mm512_mul_ps(p, scale);

            // swish = x / (1 + exp(-x))
            let denom = _mm512_add_ps(one, exp_neg_x);
            let swish_result = _mm512_div_ps(x, denom);

            _mm512_storeu_ps(result.as_mut_ptr().add(i), swish_result);
            i += 16;
        }

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
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn tanh(a: &[f32], result: &mut [f32]) {
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        let len = a.len();
        let mut i = 0;

        let log2e = _mm512_set1_ps(std::f32::consts::LOG2_E);
        let ln2 = _mm512_set1_ps(std::f32::consts::LN_2);
        let half = _mm512_set1_ps(0.5);
        let one = _mm512_set1_ps(1.0);
        let two = _mm512_set1_ps(2.0);

        let c1 = _mm512_set1_ps(1.0);
        let c2 = _mm512_set1_ps(0.5);
        let c3 = _mm512_set1_ps(0.166_666_67);
        let c4 = _mm512_set1_ps(0.041_666_668);
        let c5 = _mm512_set1_ps(0.008_333_334);
        let c6 = _mm512_set1_ps(0.001_388_889);

        let exp_hi = _mm512_set1_ps(88.376_26);
        let exp_lo = _mm512_set1_ps(-87.336_55);

        while i + 16 <= len {
            let x = _mm512_loadu_ps(a.as_ptr().add(i));
            let two_x = _mm512_mul_ps(two, x);
            let two_x = _mm512_max_ps(_mm512_min_ps(two_x, exp_hi), exp_lo);

            let x_scaled = _mm512_mul_ps(two_x, log2e);
            let k = _mm512_roundscale_ps(_mm512_add_ps(x_scaled, half), 0x09);
            let r = _mm512_sub_ps(two_x, _mm512_mul_ps(k, ln2));

            let mut p = c6;
            p = _mm512_fmadd_ps(p, r, c5);
            p = _mm512_fmadd_ps(p, r, c4);
            p = _mm512_fmadd_ps(p, r, c3);
            p = _mm512_fmadd_ps(p, r, c2);
            p = _mm512_fmadd_ps(p, r, c1);
            p = _mm512_fmadd_ps(p, r, one);

            let k_int = _mm512_cvtps_epi32(k);
            let k_shifted = _mm512_slli_epi32(k_int, 23);
            let scale = _mm512_castsi512_ps(_mm512_add_epi32(_mm512_castps_si512(one), k_shifted));
            let exp_2x = _mm512_mul_ps(p, scale);

            let tanh_numer = _mm512_sub_ps(exp_2x, one);
            // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
            // 1. Loop bounds ensure proper array access
            // 2. All pointers derived from valid slice references
            // 3. AVX-512 intrinsics marked with #[target_feature]
            // 4. Unaligned loads/stores handle unaligned data correctly
            let tanh_denom = _mm512_add_ps(exp_2x, one);
            let tanh_result = _mm512_div_ps(tanh_numer, tanh_denom);

            _mm512_storeu_ps(result.as_mut_ptr().add(i), tanh_result);
            i += 16;
        }

        while i < len {
            result[i] = a[i].tanh();
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure proper array access
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature]
    // 4. Unaligned loads/stores handle unaligned data correctly
    unsafe fn sqrt(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 16 elements at a time with AVX-512
        while i + 16 <= len {
            let vec = _mm512_loadu_ps(a.as_ptr().add(i));
            let sqrt_vec = _mm512_sqrt_ps(vec);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), sqrt_vec);
            i += 16;
        }

        while i < len {
            result[i] = a[i].sqrt();
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores handle unaligned data correctly
    unsafe fn recip(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        let one = _mm512_set1_ps(1.0);
        while i + 16 <= len {
            let vec = _mm512_loadu_ps(a.as_ptr().add(i));
            let recip_vec = _mm512_div_ps(one, vec);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), recip_vec);
            i += 16;
        }

        while i < len {
            result[i] = a[i].recip();
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    //
    // Natural logarithm implementation using range reduction:
    // For x = 2^k * m where m ∈ [1, 2):
    //   ln(x) = k*ln(2) + ln(m)
    //   ln(m) approximated using 7th-degree polynomial
    #[inline]
    #[target_feature(enable = "avx512f,fma")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 and FMA intrinsics marked with #[target_feature]
    // 4. Unaligned loads/stores handle unaligned data correctly
    unsafe fn ln(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Constants for ln calculation
        let ln2 = _mm512_set1_ps(std::f32::consts::LN_2); // 0.693147...
        let one = _mm512_set1_ps(1.0);

        // Use atanh transformation for better accuracy
        let two = _mm512_set1_ps(2.0);
        let c1 = _mm512_set1_ps(1.0);
        let c3 = _mm512_set1_ps(1.0 / 3.0);
        let c5 = _mm512_set1_ps(1.0 / 5.0);
        let c7 = _mm512_set1_ps(1.0 / 7.0);
        let c9 = _mm512_set1_ps(1.0 / 9.0);
        let c11 = _mm512_set1_ps(1.0 / 11.0);

        let mantissa_mask = _mm512_set1_epi32(0x007F_FFFF_u32 as i32);
        let exponent_127 = _mm512_set1_epi32(127 << 23);

        while i + 16 <= len {
            let x = _mm512_loadu_ps(a.as_ptr().add(i));
            let x_int = _mm512_castps_si512(x);

            let exp_biased = _mm512_srli_epi32(x_int, 23);
            let exp_biased_masked = _mm512_and_si512(exp_biased, _mm512_set1_epi32(0xFF));
            let k_int = _mm512_sub_epi32(exp_biased_masked, _mm512_set1_epi32(127));
            let k = _mm512_cvtepi32_ps(k_int);

            let mantissa_bits = _mm512_and_si512(x_int, mantissa_mask);
            let m_int = _mm512_or_si512(mantissa_bits, exponent_127);
            let m = _mm512_castsi512_ps(m_int);

            // atanh transformation
            let m_minus_1 = _mm512_sub_ps(m, one);
            let m_plus_1 = _mm512_add_ps(m, one);
            let u = _mm512_div_ps(m_minus_1, m_plus_1);
            let u2 = _mm512_mul_ps(u, u);

            let p = _mm512_fmadd_ps(c11, u2, c9);
            let p = _mm512_fmadd_ps(p, u2, c7);
            let p = _mm512_fmadd_ps(p, u2, c5);
            let p = _mm512_fmadd_ps(p, u2, c3);
            let p = _mm512_fmadd_ps(p, u2, c1);

            let ln_m = _mm512_mul_ps(two, _mm512_mul_ps(u, p));

            // ln(x) = k*ln(2) + ln(m)
            let result_vec = _mm512_fmadd_ps(k, ln2, ln_m);

            _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
            i += 16;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].ln();
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    //
    // Base-2 logarithm implementation using range reduction:
    // For x = 2^k * m where m ∈ [1, 2):
    //   log2(x) = k + log2(m)
    //   log2(m) = ln(m) / ln(2)
    #[inline]
    #[target_feature(enable = "avx512f,fma")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 and FMA intrinsics marked with #[target_feature]
    // 4. Unaligned loads/stores handle unaligned data correctly
    unsafe fn log2(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        let inv_ln2 = _mm512_set1_ps(std::f32::consts::LOG2_E);
        let one = _mm512_set1_ps(1.0);

        let two = _mm512_set1_ps(2.0);
        let c1 = _mm512_set1_ps(1.0);
        let c3 = _mm512_set1_ps(1.0 / 3.0);
        let c5 = _mm512_set1_ps(1.0 / 5.0);
        let c7 = _mm512_set1_ps(1.0 / 7.0);
        let c9 = _mm512_set1_ps(1.0 / 9.0);
        let c11 = _mm512_set1_ps(1.0 / 11.0);

        let mantissa_mask = _mm512_set1_epi32(0x007F_FFFF_u32 as i32);
        let exponent_127 = _mm512_set1_epi32(127 << 23);

        while i + 16 <= len {
            let x = _mm512_loadu_ps(a.as_ptr().add(i));
            let x_int = _mm512_castps_si512(x);

            // Extract exponent k
            let exp_biased = _mm512_srli_epi32(x_int, 23);
            let exp_biased_masked = _mm512_and_si512(exp_biased, _mm512_set1_epi32(0xFF));
            let k_int = _mm512_sub_epi32(exp_biased_masked, _mm512_set1_epi32(127));
            let k = _mm512_cvtepi32_ps(k_int);

            let mantissa_bits = _mm512_and_si512(x_int, mantissa_mask);
            let m_int = _mm512_or_si512(mantissa_bits, exponent_127);
            let m = _mm512_castsi512_ps(m_int);

            // atanh transformation
            let m_minus_1 = _mm512_sub_ps(m, one);
            let m_plus_1 = _mm512_add_ps(m, one);
            let u = _mm512_div_ps(m_minus_1, m_plus_1);
            let u2 = _mm512_mul_ps(u, u);

            let p = _mm512_fmadd_ps(c11, u2, c9);
            let p = _mm512_fmadd_ps(p, u2, c7);
            let p = _mm512_fmadd_ps(p, u2, c5);
            let p = _mm512_fmadd_ps(p, u2, c3);
            let p = _mm512_fmadd_ps(p, u2, c1);

            let ln_m = _mm512_mul_ps(two, _mm512_mul_ps(u, p));

            let log2_m = _mm512_mul_ps(ln_m, inv_ln2);

            // log2(x) = k + log2(m)
            let result_vec = _mm512_add_ps(k, log2_m);

            _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
            i += 16;
        }

        while i < len {
            result[i] = a[i].log2();
            i += 1;
        }
    }

    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    //
    // Base-10 logarithm implementation using range reduction:
    // For x = 2^k * m where m ∈ [1, 2):
    //   log10(x) = k*log10(2) + log10(m)
    //   log10(m) = ln(m) / ln(10)
    #[inline]
    #[target_feature(enable = "avx512f,fma")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 and FMA intrinsics marked with #[target_feature]
    // 4. Unaligned loads/stores handle unaligned data correctly
    unsafe fn log10(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        let log10_2 = _mm512_set1_ps(std::f32::consts::LOG10_2);
        let inv_ln10 = _mm512_set1_ps(1.0 / std::f32::consts::LN_10);
        let one = _mm512_set1_ps(1.0);

        let two = _mm512_set1_ps(2.0);
        let c1 = _mm512_set1_ps(1.0);
        let c3 = _mm512_set1_ps(1.0 / 3.0);
        let c5 = _mm512_set1_ps(1.0 / 5.0);
        let c7 = _mm512_set1_ps(1.0 / 7.0);
        let c9 = _mm512_set1_ps(1.0 / 9.0);
        let c11 = _mm512_set1_ps(1.0 / 11.0);

        let mantissa_mask = _mm512_set1_epi32(0x007F_FFFF_u32 as i32);
        let exponent_127 = _mm512_set1_epi32(127 << 23);

        while i + 16 <= len {
            let x = _mm512_loadu_ps(a.as_ptr().add(i));
            let x_int = _mm512_castps_si512(x);

            // Extract exponent k
            let exp_biased = _mm512_srli_epi32(x_int, 23);
            let exp_biased_masked = _mm512_and_si512(exp_biased, _mm512_set1_epi32(0xFF));
            let k_int = _mm512_sub_epi32(exp_biased_masked, _mm512_set1_epi32(127));
            let k = _mm512_cvtepi32_ps(k_int);

            // Extract mantissa m ∈ [1, 2)
            let mantissa_bits = _mm512_and_si512(x_int, mantissa_mask);
            let m_int = _mm512_or_si512(mantissa_bits, exponent_127);
            let m = _mm512_castsi512_ps(m_int);

            // atanh transformation
            let m_minus_1 = _mm512_sub_ps(m, one);
            let m_plus_1 = _mm512_add_ps(m, one);
            let u = _mm512_div_ps(m_minus_1, m_plus_1);
            let u2 = _mm512_mul_ps(u, u);

            let p = _mm512_fmadd_ps(c11, u2, c9);
            let p = _mm512_fmadd_ps(p, u2, c7);
            let p = _mm512_fmadd_ps(p, u2, c5);
            let p = _mm512_fmadd_ps(p, u2, c3);
            let p = _mm512_fmadd_ps(p, u2, c1);

            let ln_m = _mm512_mul_ps(two, _mm512_mul_ps(u, p));

            let log10_m = _mm512_mul_ps(ln_m, inv_ln10);

            // log10(x) = k*log10(2) + log10(m)
            let result_vec = _mm512_fmadd_ps(k, log10_2, log10_m);

            _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
            i += 16;
        }

        while i < len {
            result[i] = a[i].log10();
            i += 1;
        }
    }

    // Trigonometric functions currently use scalar implementations
    // Full SIMD trig functions require complex range reduction and are left for future work
    // TODO: Implement proper SIMD range reduction for sin/cos/tan

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Delegates to scalar implementation, no direct SIMD operations
    unsafe fn sin(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::sin(a, result);
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Delegates to scalar implementation, no direct SIMD operations
    unsafe fn cos(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::cos(a, result);
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Delegates to scalar implementation, no direct SIMD operations
    unsafe fn tan(a: &[f32], result: &mut [f32]) {
        super::scalar::ScalarBackend::tan(a, result);
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn floor(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 16 elements at a time
        // Rounding mode 0x09 = round down (floor)
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vresult = _mm512_roundscale_ps(va, 0x09);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 16;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].floor();
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn ceil(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Process 16 elements at a time
        // Rounding mode 0x0A = round up (ceil)
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vresult = _mm512_roundscale_ps(va, 0x0A);
            _mm512_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 16;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].ceil();
            i += 1;
        }
    }

    #[inline]
    #[target_feature(enable = "avx512f")]
    // SAFETY: Pointer arithmetic and SIMD intrinsics are safe because:
    // 1. Loop bounds ensure `i + 16 <= len` before calling `.add(i)`
    // 2. All pointers derived from valid slice references
    // 3. AVX-512 intrinsics marked with #[target_feature(enable = "avx512f")]
    // 4. Unaligned loads/stores used - no alignment requirement
    unsafe fn round(a: &[f32], result: &mut [f32]) {
        let len = a.len();
        let mut i = 0;

        // Rust's .round() rounds ties away from zero, but SIMD round modes don't support this.
        // Implement manually: round(x) = sign(x) * floor(abs(x) + 0.5)
        let half = _mm512_set1_ps(0.5);
        let sign_mask = _mm512_set1_ps(f32::from_bits(0x8000_0000)); // Sign bit only
        let abs_mask = _mm512_set1_ps(f32::from_bits(0x7FFF_FFFF)); // All except sign bit

        // Process 16 elements at a time
        while i + 16 <= len {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));

            // Extract sign and absolute value
            let sign = _mm512_and_ps(va, sign_mask);
            let abs_val = _mm512_and_ps(va, abs_mask);

            // Round away from zero: floor(abs(x) + 0.5) * sign(x)
            let shifted = _mm512_add_ps(abs_val, half);
            let rounded_abs = _mm512_roundscale_ps(shifted, 0x09); // floor
            let vresult = _mm512_or_ps(rounded_abs, sign);

            _mm512_storeu_ps(result.as_mut_ptr().add(i), vresult);
            i += 16;
        }

        // Handle remaining elements with scalar code
        while i < len {
            result[i] = a[i].round();
            i += 1;
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::backends::scalar::ScalarBackend;

    /// Helper to run AVX-512 test only on CPUs that support it
    fn avx512_test<F>(test_fn: F)
    where
        F: FnOnce(),
    {
        if is_x86_feature_detected!("avx512f") {
            test_fn();
        } else {
            // Skip test on CPUs without AVX-512 support
            println!("Skipping AVX-512 test (CPU does not support avx512f)");
        }
    }

    #[test]
    fn test_avx512_add_basic() {
        avx512_test(|| {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let b = vec![5.0, 6.0, 7.0, 8.0];
            let mut result = vec![0.0; 4];

            unsafe {
                Avx512Backend::add(&a, &b, &mut result);
            }

            assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
        });
    }

    #[test]
    fn test_avx512_add_aligned_16() {
        avx512_test(|| {
            // Test with exactly 16 elements (one AVX-512 register)
            let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..16).map(|i| (i + 10) as f32).collect();
            let mut result = vec![0.0; 16];
            let expected: Vec<f32> = (0..16).map(|i| (i + i + 10) as f32).collect();

            unsafe {
                Avx512Backend::add(&a, &b, &mut result);
            }

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_avx512_add_non_aligned() {
        avx512_test(|| {
            // Test with 18 elements (16 + 2 remainder)
            let a: Vec<f32> = (0..18).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..18).map(|i| (i * 2) as f32).collect();
            let mut result = vec![0.0; 18];
            let expected: Vec<f32> = (0..18).map(|i| (i + i * 2) as f32).collect();

            unsafe {
                Avx512Backend::add(&a, &b, &mut result);
            }

            assert_eq!(result, expected);
        });
    }

    #[test]
    fn test_avx512_add_large() {
        avx512_test(|| {
            // Test with 1000 elements (many AVX-512 iterations)
            let a: Vec<f32> = (0..1000).map(|i| i as f32 * 0.5).collect();
            let b: Vec<f32> = (0..1000).map(|i| i as f32 * 0.3).collect();
            let mut result = vec![0.0; 1000];

            unsafe {
                Avx512Backend::add(&a, &b, &mut result);
            }

            for (i, &value) in result.iter().enumerate().take(1000) {
                let expected = i as f32 * 0.5 + i as f32 * 0.3;
                assert!(
                    (value - expected).abs() < 1e-5,
                    "Mismatch at index {}: expected {}, got {}",
                    i,
                    expected,
                    value
                );
            }
        });
    }

    #[test]
    fn test_avx512_add_single_element() {
        avx512_test(|| {
            let a = vec![42.0];
            let b = vec![13.0];
            let mut result = vec![0.0];

            unsafe {
                Avx512Backend::add(&a, &b, &mut result);
            }

            assert_eq!(result, vec![55.0]);
        });
    }

    #[test]
    fn test_avx512_add_negative_values() {
        avx512_test(|| {
            let a = vec![
                -1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0,
            ];
            let b = vec![
                10.0, 20.0, 30.0, 40.0, -50.0, -60.0, -70.0, -80.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0,
            ];
            let mut result = vec![0.0; 16];
            let expected = vec![
                9.0, 18.0, 27.0, 36.0, -45.0, -54.0, -63.0, -72.0, 10.0, 12.0, 14.0, 16.0, 18.0,
                20.0, 22.0, 24.0,
            ];

            unsafe {
                Avx512Backend::add(&a, &b, &mut result);
            }

            for i in 0..16 {
                assert!(
                    (result[i] - expected[i]).abs() < 1e-5,
                    "Mismatch at index {}: expected {}, got {}",
                    i,
                    expected[i],
                    result[i]
                );
            }
        });
    }

    #[test]
    fn test_avx512_add_equivalence_to_scalar() {
        avx512_test(|| {
            // Backend equivalence: AVX-512 should produce same results as Scalar
            let sizes = vec![1, 7, 15, 16, 17, 32, 63, 100, 1000];

            for size in sizes {
                let a: Vec<f32> = (0..size).map(|i| (i as f32 * 1.5) - 50.0).collect();
                let b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.7) + 20.0).collect();

                let mut result_avx512 = vec![0.0; size];
                let mut result_scalar = vec![0.0; size];

                unsafe {
                    Avx512Backend::add(&a, &b, &mut result_avx512);
                    ScalarBackend::add(&a, &b, &mut result_scalar);
                }

                for i in 0..size {
                    assert!(
                        (result_avx512[i] - result_scalar[i]).abs() < 1e-5,
                        "Backend mismatch at size {} index {}: AVX512={}, Scalar={}",
                        size,
                        i,
                        result_avx512[i],
                        result_scalar[i]
                    );
                }
            }
        });
    }

    #[test]
    fn test_avx512_add_special_values() {
        avx512_test(|| {
            // Test with infinity, zero, and very small/large values
            let a = vec![
                0.0,
                -0.0,
                f32::INFINITY,
                f32::NEG_INFINITY,
                1e-20,
                -1e-20,
                1e20,
                -1e20,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ];
            let b = vec![
                0.0, 0.0, 1.0, -1.0, 2e-20, -2e-20, 2e20, -2e20, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                8.0,
            ];
            let mut result = vec![0.0; 16];

            unsafe {
                Avx512Backend::add(&a, &b, &mut result);
            }

            assert_eq!(result[0], 0.0);
            assert_eq!(result[1], 0.0);
            assert_eq!(result[2], f32::INFINITY);
            assert_eq!(result[3], f32::NEG_INFINITY);
            assert!((result[4] - 3e-20).abs() < 1e-25);
            assert!((result[5] + 3e-20).abs() < 1e-25);
        });
    }

    #[test]
    fn test_avx512_add_remainder_correctness() {
        avx512_test(|| {
            // Specifically test remainder handling for sizes 16+1 through 16+15
            for remainder in 1..=15 {
                let size = 16 + remainder;
                let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
                let b: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();
                let mut result = vec![0.0; size];

                unsafe {
                    Avx512Backend::add(&a, &b, &mut result);
                }

                // Verify all elements, especially the remainder portion
                for (i, &value) in result.iter().enumerate().take(size) {
                    let expected = i as f32 + (size - i) as f32;
                    assert_eq!(
                        value, expected,
                        "Remainder test failed at size {} (remainder {}), index {}",
                        size, remainder, i
                    );
                }
            }
        });
    }

    // =====================
    // AVX-512 dot() tests
    // =====================

    #[test]
    fn test_avx512_dot_basic() {
        avx512_test(|| {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let b = vec![5.0, 6.0, 7.0, 8.0];
            // Expected: 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
            let result = unsafe { Avx512Backend::dot(&a, &b) };
            assert!(
                (result - 70.0).abs() < 1e-5,
                "Expected 70.0, got {}",
                result
            );
        });
    }

    #[test]
    fn test_avx512_dot_aligned_16() {
        avx512_test(|| {
            // Test with exactly 16 elements (one AVX-512 register)
            let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();
            // Expected: sum of i * (i + 1) for i in 0..16
            // = 0*1 + 1*2 + 2*3 + ... + 15*16
            let expected: f32 = (0..16).map(|i| (i * (i + 1)) as f32).sum();

            let result = unsafe { Avx512Backend::dot(&a, &b) };
            assert!(
                (result - expected).abs() < 1e-4,
                "Expected {}, got {}",
                expected,
                result
            );
        });
    }

    #[test]
    fn test_avx512_dot_non_aligned() {
        avx512_test(|| {
            // Test with 18 elements (16 + 2 remainder)
            let a: Vec<f32> = (0..18).map(|i| (i as f32) * 1.5).collect();
            let b: Vec<f32> = (0..18).map(|i| (i as f32) * 0.7).collect();
            // Expected: sum of (i * 1.5) * (i * 0.7) = sum of i^2 * 1.05
            let expected: f32 = (0..18).map(|i| ((i * i) as f32) * 1.05).sum();

            let result = unsafe { Avx512Backend::dot(&a, &b) };
            assert!(
                (result - expected).abs() < 1e-3,
                "Expected {}, got {}",
                expected,
                result
            );
        });
    }

    #[test]
    fn test_avx512_dot_large() {
        avx512_test(|| {
            // Test with 1000 elements (62 full AVX-512 registers + 8 remainder)
            let size = 1000;
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();
            let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.3).collect();
            // Expected: sum of (i * 0.5) * (i * 0.3) = sum of i^2 * 0.15
            let expected: f32 = (0..size).map(|i| ((i * i) as f32) * 0.15).sum();

            let result = unsafe { Avx512Backend::dot(&a, &b) };
            // Larger tolerance for accumulation of floating point errors
            assert!(
                (result - expected).abs() / expected.abs() < 1e-4,
                "Expected {}, got {}, relative error: {}",
                expected,
                result,
                ((result - expected).abs() / expected.abs())
            );
        });
    }

    #[test]
    fn test_avx512_dot_equivalence_to_scalar() {
        avx512_test(|| {
            // Backend equivalence: AVX-512 should produce same results as Scalar
            let sizes = vec![1, 7, 15, 16, 17, 32, 63, 100, 1000];

            for size in sizes {
                let a: Vec<f32> = (0..size).map(|i| (i as f32 * 1.5) - 50.0).collect();
                let b: Vec<f32> = (0..size).map(|i| (i as f32 * 0.7) + 20.0).collect();

                let result_avx512 = unsafe { Avx512Backend::dot(&a, &b) };
                let result_scalar = unsafe { ScalarBackend::dot(&a, &b) };

                // Use relative tolerance for larger values
                let tolerance = if result_scalar.abs() > 1.0 {
                    result_scalar.abs() * 1e-5
                } else {
                    1e-5
                };

                assert!(
                    (result_avx512 - result_scalar).abs() < tolerance,
                    "Backend mismatch at size {}: AVX512={}, Scalar={}, diff={}",
                    size,
                    result_avx512,
                    result_scalar,
                    (result_avx512 - result_scalar).abs()
                );
            }
        });
    }

    #[test]
    fn test_avx512_dot_special_values() {
        avx512_test(|| {
            // Test with zero, negative, small, and large values
            let a = vec![
                0.0, -1.0, 1.0, -5.0, 5.0, 1e-10, 1e10, -1e10, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                9.0,
            ];
            let b = vec![
                10.0, 2.0, 3.0, -2.0, 4.0, 2e-10, 2e10, -2e10, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0,
            ];

            // Expected: 0*10 + (-1)*2 + 1*3 + (-5)*(-2) + 5*4 + (1e-10)*(2e-10) + (1e10)*(2e10) + (-1e10)*(-2e10) + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
            //         = 0 - 2 + 3 + 10 + 20 + 2e-20 + 2e20 + 2e20 + 44
            //         = 75 + 2e-20 + 4e20
            // Note: 2e-20 is negligible compared to 4e20

            let result = unsafe { Avx512Backend::dot(&a, &b) };
            let expected = unsafe { ScalarBackend::dot(&a, &b) };

            // Use relative tolerance due to large values
            let rel_error = if expected.abs() > 1.0 {
                (result - expected).abs() / expected.abs()
            } else {
                (result - expected).abs()
            };

            assert!(
                rel_error < 1e-5,
                "Expected {}, got {}, relative error: {}",
                expected,
                result,
                rel_error
            );
        });
    }

    #[test]
    fn test_avx512_dot_remainder_sizes() {
        avx512_test(|| {
            // Test all remainder sizes from 16+1 to 16+15
            for remainder in 1..=15 {
                let size = 16 + remainder;
                let a: Vec<f32> = (0..size).map(|i| (i as f32) + 1.0).collect();
                let b: Vec<f32> = (0..size).map(|i| (i as f32) + 2.0).collect();

                let result_avx512 = unsafe { Avx512Backend::dot(&a, &b) };
                let result_scalar = unsafe { ScalarBackend::dot(&a, &b) };

                let tolerance = if result_scalar.abs() > 1.0 {
                    result_scalar.abs() * 1e-5
                } else {
                    1e-5
                };

                assert!(
                    (result_avx512 - result_scalar).abs() < tolerance,
                    "Remainder test failed at size {} (remainder {}): AVX512={}, Scalar={}",
                    size,
                    remainder,
                    result_avx512,
                    result_scalar
                );
            }
        });
    }

    #[test]
    fn test_avx512_dot_zero_vector() {
        avx512_test(|| {
            let a = vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ];
            let b = vec![0.0; 16];

            let result = unsafe { Avx512Backend::dot(&a, &b) };
            assert_eq!(
                result, 0.0,
                "Dot product with zero vector should be 0.0, got {}",
                result
            );
        });
    }

    #[test]
    fn test_avx512_dot_orthogonal() {
        avx512_test(|| {
            // Orthogonal vectors: [1, 0, 0, 0, ...] and [0, 1, 0, 0, ...]
            let mut a = vec![0.0; 16];
            let mut b = vec![0.0; 16];
            a[0] = 1.0;
            b[1] = 1.0;

            let result = unsafe { Avx512Backend::dot(&a, &b) };
            assert_eq!(
                result, 0.0,
                "Dot product of orthogonal vectors should be 0.0, got {}",
                result
            );
        });
    }

    // =====================
    // AVX-512 sum() tests
    // =====================

    #[test]
    fn test_avx512_sum_basic() {
        avx512_test(|| {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            // Expected: 1 + 2 + 3 + 4 = 10
            let result = unsafe { Avx512Backend::sum(&a) };
            assert!(
                (result - 10.0).abs() < 1e-5,
                "Expected 10.0, got {}",
                result
            );
        });
    }

    #[test]
    fn test_avx512_sum_aligned_16() {
        avx512_test(|| {
            // Test with exactly 16 elements (one AVX-512 register)
            let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
            // Expected: sum of 0..16 = 0+1+2+...+15 = 120
            let expected: f32 = (0..16).map(|i| i as f32).sum();

            let result = unsafe { Avx512Backend::sum(&a) };
            assert!(
                (result - expected).abs() < 1e-4,
                "Expected {}, got {}",
                expected,
                result
            );
        });
    }

    #[test]
    fn test_avx512_sum_non_aligned() {
        avx512_test(|| {
            // Test with 18 elements (16 + 2 remainder)
            let a: Vec<f32> = (0..18).map(|i| (i as f32) * 1.5).collect();
            // Expected: sum of (i * 1.5) for i in 0..18
            let expected: f32 = (0..18).map(|i| (i as f32) * 1.5).sum();

            let result = unsafe { Avx512Backend::sum(&a) };
            assert!(
                (result - expected).abs() < 1e-3,
                "Expected {}, got {}",
                expected,
                result
            );
        });
    }

    #[test]
    fn test_avx512_sum_large() {
        avx512_test(|| {
            // Test with 1000 elements (62 full AVX-512 registers + 8 remainder)
            let size = 1000;
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();
            // Expected: sum of (i * 0.5) for i in 0..1000
            let expected: f32 = (0..size).map(|i| (i as f32) * 0.5).sum();

            let result = unsafe { Avx512Backend::sum(&a) };
            // Larger tolerance for accumulation of floating point errors
            let rel_error = if expected.abs() > 1.0 {
                (result - expected).abs() / expected.abs()
            } else {
                (result - expected).abs()
            };
            assert!(
                rel_error < 1e-4,
                "Expected {}, got {}, relative error: {}",
                expected,
                result,
                rel_error
            );
        });
    }

    #[test]
    fn test_avx512_sum_equivalence_to_scalar() {
        avx512_test(|| {
            // Backend equivalence: AVX-512 should produce same results as Scalar
            let sizes = vec![1, 7, 15, 16, 17, 32, 63, 100, 1000];

            for size in sizes {
                let a: Vec<f32> = (0..size).map(|i| (i as f32 * 1.5) - 50.0).collect();

                let result_avx512 = unsafe { Avx512Backend::sum(&a) };
                let result_scalar = unsafe { ScalarBackend::sum(&a) };

                // Use relative tolerance for larger values
                let tolerance = if result_scalar.abs() > 1.0 {
                    result_scalar.abs() * 1e-5
                } else {
                    1e-5
                };

                assert!(
                    (result_avx512 - result_scalar).abs() < tolerance,
                    "Backend mismatch at size {}: AVX512={}, Scalar={}, diff={}",
                    size,
                    result_avx512,
                    result_scalar,
                    (result_avx512 - result_scalar).abs()
                );
            }
        });
    }

    #[test]
    fn test_avx512_sum_negative_values() {
        avx512_test(|| {
            let a = vec![
                -1.0, -2.0, -3.0, -4.0, 5.0, 6.0, 7.0, 8.0, -9.0, -10.0, 11.0, 12.0, -13.0, 14.0,
                -15.0, 16.0,
            ];
            // Expected: -1 - 2 - 3 - 4 + 5 + 6 + 7 + 8 - 9 - 10 + 11 + 12 - 13 + 14 - 15 + 16 = 22
            let expected = 22.0;

            let result = unsafe { Avx512Backend::sum(&a) };
            assert!(
                (result - expected).abs() < 1e-5,
                "Expected {}, got {}",
                expected,
                result
            );
        });
    }

    #[test]
    fn test_avx512_sum_zero_vector() {
        avx512_test(|| {
            let a = vec![0.0; 16];
            let result = unsafe { Avx512Backend::sum(&a) };
            assert_eq!(result, 0.0, "Sum of zeros should be 0.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_sum_single_element() {
        avx512_test(|| {
            let a = vec![42.0];
            let result = unsafe { Avx512Backend::sum(&a) };
            assert_eq!(
                result, 42.0,
                "Sum of single element should be that element, got {}",
                result
            );
        });
    }

    #[test]
    fn test_avx512_sum_remainder_sizes() {
        avx512_test(|| {
            // Test all remainder sizes from 16+1 to 16+15
            for remainder in 1..=15 {
                let size = 16 + remainder;
                let a: Vec<f32> = (0..size).map(|i| (i as f32) + 1.0).collect();

                let result_avx512 = unsafe { Avx512Backend::sum(&a) };
                let result_scalar = unsafe { ScalarBackend::sum(&a) };

                let tolerance = if result_scalar.abs() > 1.0 {
                    result_scalar.abs() * 1e-5
                } else {
                    1e-5
                };

                assert!(
                    (result_avx512 - result_scalar).abs() < tolerance,
                    "Remainder test failed at size {} (remainder {}): AVX512={}, Scalar={}",
                    size,
                    remainder,
                    result_avx512,
                    result_scalar
                );
            }
        });
    }

    // =====================
    // AVX-512 max() tests
    // =====================

    #[test]
    fn test_avx512_max_basic() {
        avx512_test(|| {
            let a = vec![1.0, 5.0, 3.0, 9.0, 2.0];
            let result = unsafe { Avx512Backend::max(&a) };
            assert_eq!(result, 9.0, "Expected 9.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_max_aligned_16() {
        avx512_test(|| {
            let mut a: Vec<f32> = (0..16).map(|i| i as f32).collect();
            a[8] = 100.0; // Max is in the middle
            let result = unsafe { Avx512Backend::max(&a) };
            assert_eq!(result, 100.0, "Expected 100.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_max_non_aligned() {
        avx512_test(|| {
            let mut a: Vec<f32> = (0..18).map(|i| (i as f32) * 1.5).collect();
            a[17] = 200.0; // Max is in remainder
            let result = unsafe { Avx512Backend::max(&a) };
            assert_eq!(result, 200.0, "Expected 200.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_max_negative_values() {
        avx512_test(|| {
            let a = vec![-5.0, -2.0, -10.0, -1.0, -8.0];
            let result = unsafe { Avx512Backend::max(&a) };
            assert_eq!(result, -1.0, "Expected -1.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_max_equivalence_to_scalar() {
        avx512_test(|| {
            let sizes = vec![1, 7, 15, 16, 17, 32, 63, 100, 1000];
            for size in sizes {
                let a: Vec<f32> = (0..size).map(|i| ((i * 7) % 100) as f32 - 50.0).collect();
                let result_avx512 = unsafe { Avx512Backend::max(&a) };
                let result_scalar = unsafe { ScalarBackend::max(&a) };
                assert_eq!(
                    result_avx512, result_scalar,
                    "Backend mismatch at size {}: AVX512={}, Scalar={}",
                    size, result_avx512, result_scalar
                );
            }
        });
    }

    // =====================
    // AVX-512 min() tests
    // =====================

    #[test]
    fn test_avx512_min_basic() {
        avx512_test(|| {
            let a = vec![5.0, 1.0, 9.0, 3.0, 2.0];
            let result = unsafe { Avx512Backend::min(&a) };
            assert_eq!(result, 1.0, "Expected 1.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_min_aligned_16() {
        avx512_test(|| {
            let mut a: Vec<f32> = (0..16).map(|i| (i + 10) as f32).collect();
            a[8] = -100.0; // Min is in the middle
            let result = unsafe { Avx512Backend::min(&a) };
            assert_eq!(result, -100.0, "Expected -100.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_min_non_aligned() {
        avx512_test(|| {
            let mut a: Vec<f32> = (0..18).map(|i| (i as f32) * 1.5 + 10.0).collect();
            a[17] = -200.0; // Min is in remainder
            let result = unsafe { Avx512Backend::min(&a) };
            assert_eq!(result, -200.0, "Expected -200.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_min_positive_values() {
        avx512_test(|| {
            let a = vec![5.0, 2.0, 10.0, 1.0, 8.0];
            let result = unsafe { Avx512Backend::min(&a) };
            assert_eq!(result, 1.0, "Expected 1.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_min_equivalence_to_scalar() {
        avx512_test(|| {
            let sizes = vec![1, 7, 15, 16, 17, 32, 63, 100, 1000];
            for size in sizes {
                let a: Vec<f32> = (0..size).map(|i| ((i * 7) % 100) as f32 - 50.0).collect();
                let result_avx512 = unsafe { Avx512Backend::min(&a) };
                let result_scalar = unsafe { ScalarBackend::min(&a) };
                assert_eq!(
                    result_avx512, result_scalar,
                    "Backend mismatch at size {}: AVX512={}, Scalar={}",
                    size, result_avx512, result_scalar
                );
            }
        });
    }

    // ============================================================================
    // argmax() tests
    // ============================================================================

    #[test]
    fn test_avx512_argmax_basic() {
        avx512_test(|| {
            let a = vec![1.0, 5.0, 3.0, 9.0, 2.0];
            let result = unsafe { Avx512Backend::argmax(&a) };
            assert_eq!(result, 3); // Index of 9.0
        });
    }

    #[test]
    fn test_avx512_argmax_aligned_16() {
        avx512_test(|| {
            let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
            let result = unsafe { Avx512Backend::argmax(&a) };
            assert_eq!(result, 15); // Maximum is at index 15
        });
    }

    #[test]
    fn test_avx512_argmax_non_aligned_18() {
        avx512_test(|| {
            let a: Vec<f32> = (0..18).map(|i| i as f32).collect();
            let result = unsafe { Avx512Backend::argmax(&a) };
            assert_eq!(result, 17); // Maximum is at index 17
        });
    }

    #[test]
    fn test_avx512_argmax_negative_values() {
        avx512_test(|| {
            let a = vec![-5.0, -2.0, -8.0, -1.0, -10.0];
            let result = unsafe { Avx512Backend::argmax(&a) };
            assert_eq!(result, 3); // Index of -1.0
        });
    }

    #[test]
    fn test_avx512_argmax_max_at_start() {
        avx512_test(|| {
            let a = vec![100.0, 1.0, 2.0, 3.0, 4.0];
            let result = unsafe { Avx512Backend::argmax(&a) };
            assert_eq!(result, 0); // Maximum is at index 0
        });
    }

    #[test]
    fn test_avx512_argmax_backend_equivalence() {
        avx512_test(|| {
            let sizes = [16, 17, 100, 1000, 10000, 16384, 16385, 100000, 1000000];
            for &size in &sizes {
                let a: Vec<f32> = (0..size).map(|i| ((i * 13) % 100) as f32 - 50.0).collect();
                let result_avx512 = unsafe { Avx512Backend::argmax(&a) };
                let result_scalar = unsafe { ScalarBackend::argmax(&a) };
                assert_eq!(
                    result_avx512, result_scalar,
                    "Backend mismatch at size {}: AVX512={}, Scalar={}",
                    size, result_avx512, result_scalar
                );
            }
        });
    }

    // ============================================================================
    // argmin() tests
    // ============================================================================

    #[test]
    fn test_avx512_argmin_basic() {
        avx512_test(|| {
            let a = vec![5.0, 1.0, 9.0, 3.0, 2.0];
            let result = unsafe { Avx512Backend::argmin(&a) };
            assert_eq!(result, 1); // Index of 1.0
        });
    }

    #[test]
    fn test_avx512_argmin_aligned_16() {
        avx512_test(|| {
            let a: Vec<f32> = (0..16).rev().map(|i| i as f32).collect();
            let result = unsafe { Avx512Backend::argmin(&a) };
            assert_eq!(result, 15); // Minimum is at index 15
        });
    }

    #[test]
    fn test_avx512_argmin_non_aligned_18() {
        avx512_test(|| {
            let a: Vec<f32> = (0..18).rev().map(|i| i as f32).collect();
            let result = unsafe { Avx512Backend::argmin(&a) };
            assert_eq!(result, 17); // Minimum is at index 17
        });
    }

    #[test]
    fn test_avx512_argmin_positive_values() {
        avx512_test(|| {
            let a = vec![10.0, 5.0, 8.0, 2.0, 15.0];
            let result = unsafe { Avx512Backend::argmin(&a) };
            assert_eq!(result, 3); // Index of 2.0
        });
    }

    #[test]
    fn test_avx512_argmin_min_at_start() {
        avx512_test(|| {
            let a = vec![1.0, 100.0, 200.0, 300.0, 400.0];
            let result = unsafe { Avx512Backend::argmin(&a) };
            assert_eq!(result, 0); // Minimum is at index 0
        });
    }

    #[test]
    fn test_avx512_argmin_backend_equivalence() {
        avx512_test(|| {
            let sizes = [16, 17, 100, 1000, 10000, 16384, 16385, 100000, 1000000];
            for &size in &sizes {
                let a: Vec<f32> = (0..size).map(|i| ((i * 13) % 100) as f32 - 50.0).collect();
                let result_avx512 = unsafe { Avx512Backend::argmin(&a) };
                let result_scalar = unsafe { ScalarBackend::argmin(&a) };
                assert_eq!(
                    result_avx512, result_scalar,
                    "Backend mismatch at size {}: AVX512={}, Scalar={}",
                    size, result_avx512, result_scalar
                );
            }
        });
    }

    // ============================================================
    // norm_l2 Tests
    // ============================================================

    #[test]
    fn test_avx512_norm_l2_basic() {
        avx512_test(|| {
            let a = vec![3.0, 4.0];
            // Expected: sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5.0
            let result = unsafe { Avx512Backend::norm_l2(&a) };
            assert!((result - 5.0).abs() < 1e-5, "Expected 5.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_norm_l2_empty() {
        avx512_test(|| {
            let a: Vec<f32> = vec![];
            let result = unsafe { Avx512Backend::norm_l2(&a) };
            assert_eq!(result, 0.0, "L2 norm of empty vector should be 0.0");
        });
    }

    #[test]
    fn test_avx512_norm_l2_single() {
        avx512_test(|| {
            let a = vec![7.0];
            // Expected: sqrt(7^2) = 7.0
            let result = unsafe { Avx512Backend::norm_l2(&a) };
            assert!((result - 7.0).abs() < 1e-5, "Expected 7.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_norm_l2_aligned_16() {
        avx512_test(|| {
            // Test with exactly 16 elements (one AVX-512 register)
            let a = vec![1.0; 16];
            // Expected: sqrt(16 * 1^2) = sqrt(16) = 4.0
            let result = unsafe { Avx512Backend::norm_l2(&a) };
            assert!((result - 4.0).abs() < 1e-5, "Expected 4.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_norm_l2_non_aligned() {
        avx512_test(|| {
            // Test with 18 elements (16 + 2 remainder)
            let a: Vec<f32> = (0..18).map(|i| (i as f32) + 1.0).collect();
            // Expected: sqrt(sum((i+1)^2 for i in 0..18))
            let expected = (0..18)
                .map(|i| ((i as f32) + 1.0) * ((i as f32) + 1.0))
                .sum::<f32>()
                .sqrt();

            let result = unsafe { Avx512Backend::norm_l2(&a) };
            assert!(
                (result - expected).abs() < 1e-3,
                "Expected {}, got {}",
                expected,
                result
            );
        });
    }

    #[test]
    fn test_avx512_norm_l2_large() {
        avx512_test(|| {
            // Test with 1000 elements
            let size = 1000;
            let a: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let expected = (0..size)
                .map(|i| ((i as f32) * 0.1) * ((i as f32) * 0.1))
                .sum::<f32>()
                .sqrt();

            let result = unsafe { Avx512Backend::norm_l2(&a) };
            let rel_error = if expected.abs() > 1.0 {
                (result - expected).abs() / expected.abs()
            } else {
                (result - expected).abs()
            };
            assert!(
                rel_error < 1e-4,
                "Expected {}, got {}, relative error: {}",
                expected,
                result,
                rel_error
            );
        });
    }

    #[test]
    fn test_avx512_norm_l2_equivalence_to_scalar() {
        avx512_test(|| {
            // Backend equivalence: AVX-512 should produce same results as Scalar
            let sizes = vec![0, 1, 7, 15, 16, 17, 32, 63, 100, 1000];

            for size in sizes {
                let a: Vec<f32> = (0..size).map(|i| (i as f32 * 0.7) - 10.0).collect();

                let result_avx512 = unsafe { Avx512Backend::norm_l2(&a) };
                let result_scalar = unsafe { ScalarBackend::norm_l2(&a) };

                // Use relative tolerance for larger values
                let tolerance = if result_scalar.abs() > 1.0 {
                    result_scalar.abs() * 1e-5
                } else {
                    1e-5
                };

                assert!(
                    (result_avx512 - result_scalar).abs() < tolerance,
                    "Backend mismatch at size {}: AVX512={}, Scalar={}, diff={}",
                    size,
                    result_avx512,
                    result_scalar,
                    (result_avx512 - result_scalar).abs()
                );
            }
        });
    }

    #[test]
    fn test_avx512_norm_l2_negative_values() {
        avx512_test(|| {
            let a = vec![-3.0, -4.0];
            // Expected: sqrt((-3)^2 + (-4)^2) = sqrt(9 + 16) = 5.0
            let result = unsafe { Avx512Backend::norm_l2(&a) };
            assert!((result - 5.0).abs() < 1e-5, "Expected 5.0, got {}", result);
        });
    }

    #[test]
    fn test_avx512_norm_l1_scalar_fallback() {
        avx512_test(|| {
            // Test scalar fallback for norm_l1 (not yet SIMD optimized)
            let test_cases = vec![
                vec![],
                vec![5.0],
                vec![-3.0, 1.0, -4.0, 1.0, 5.0],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ];

            for test_vec in test_cases {
                // SAFETY: Test code calling backend trait methods
                let avx512_result = unsafe { Avx512Backend::norm_l1(&test_vec) };
                let scalar_result = unsafe { ScalarBackend::norm_l1(&test_vec) };

                assert!(
                    (avx512_result - scalar_result).abs() < 1e-5,
                    "norm_l1 mismatch for {:?}: avx512={}, scalar={}",
                    test_vec,
                    avx512_result,
                    scalar_result
                );
            }
        });
    }

    #[test]
    fn test_avx512_norm_linf_scalar_fallback() {
        avx512_test(|| {
            // Test scalar fallback for norm_linf (not yet SIMD optimized)
            let test_cases = vec![
                vec![],
                vec![5.0],
                vec![-3.0, 1.0, -4.0, 1.0, 5.0],
                vec![-10.0, 5.0, 3.0, 7.0, -2.0, 8.0, 4.0],
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ];

            for test_vec in test_cases {
                // SAFETY: Test code calling backend trait methods
                let avx512_result = unsafe { Avx512Backend::norm_linf(&test_vec) };
                let scalar_result = unsafe { ScalarBackend::norm_linf(&test_vec) };

                assert!(
                    (avx512_result - scalar_result).abs() < 1e-5,
                    "norm_linf mismatch for {:?}: avx512={}, scalar={}",
                    test_vec,
                    avx512_result,
                    scalar_result
                );
            }
        });
    }

    #[test]
    fn test_avx512_sum_kahan_scalar_fallback() {
        avx512_test(|| {
            // Test scalar fallback for sum_kahan (not yet SIMD optimized)
            let test_vec = vec![1.0e10, 1.0, -1.0e10, 1.0]; // Tests numerical stability

            // SAFETY: Test code calling backend trait methods
            let avx512_result = unsafe { Avx512Backend::sum_kahan(&test_vec) };
            let scalar_result = unsafe { ScalarBackend::sum_kahan(&test_vec) };

            assert!(
                (avx512_result - scalar_result).abs() < 1e-5,
                "sum_kahan mismatch: avx512={}, scalar={}",
                avx512_result,
                scalar_result
            );
        });
    }

    #[test]
    fn test_avx512_scale_scalar_fallback() {
        avx512_test(|| {
            // Test scalar fallback for scale (not yet SIMD optimized)
            let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            let scalar = 2.5;
            let mut avx512_result = vec![0.0; a.len()];
            let mut scalar_result = vec![0.0; a.len()];

            // SAFETY: Test code calling backend trait methods
            unsafe {
                Avx512Backend::scale(&a, scalar, &mut avx512_result);
                ScalarBackend::scale(&a, scalar, &mut scalar_result);
            }

            for i in 0..a.len() {
                assert!(
                    (avx512_result[i] - scalar_result[i]).abs() < 1e-5,
                    "scale mismatch at index {}: avx512={}, scalar={}",
                    i,
                    avx512_result[i],
                    scalar_result[i]
                );
            }
        });
    }

    #[test]
    fn test_avx512_abs_scalar_fallback() {
        avx512_test(|| {
            // Test scalar fallback for abs (not yet SIMD optimized)
            let a = vec![-3.0, 1.0, -4.0, 0.0, 5.0, -2.0];
            let mut avx512_result = vec![0.0; a.len()];
            let mut scalar_result = vec![0.0; a.len()];

            // SAFETY: Test code calling backend trait methods
            unsafe {
                Avx512Backend::abs(&a, &mut avx512_result);
                ScalarBackend::abs(&a, &mut scalar_result);
            }

            for i in 0..a.len() {
                assert!(
                    (avx512_result[i] - scalar_result[i]).abs() < 1e-5,
                    "abs mismatch at index {}: avx512={}, scalar={}",
                    i,
                    avx512_result[i],
                    scalar_result[i]
                );
            }
        });
    }

    #[test]
    fn test_avx512_clamp_scalar_fallback() {
        avx512_test(|| {
            // Test scalar fallback for clamp (not yet SIMD optimized)
            let a = vec![-5.0, 0.0, 3.0, 7.0, 10.0];
            let min_val = 0.0;
            let max_val = 5.0;
            let mut avx512_result = vec![0.0; a.len()];
            let mut scalar_result = vec![0.0; a.len()];

            // SAFETY: Test code calling backend trait methods
            unsafe {
                Avx512Backend::clamp(&a, min_val, max_val, &mut avx512_result);
                ScalarBackend::clamp(&a, min_val, max_val, &mut scalar_result);
            }

            for i in 0..a.len() {
                assert!(
                    (avx512_result[i] - scalar_result[i]).abs() < 1e-5,
                    "clamp mismatch at index {}: avx512={}, scalar={}",
                    i,
                    avx512_result[i],
                    scalar_result[i]
                );
            }
        });
    }

    #[test]
    fn test_avx512_lerp_scalar_fallback() {
        avx512_test(|| {
            // Test scalar fallback for lerp (not yet SIMD optimized)
            let a = vec![0.0, 1.0, 2.0, 3.0];
            let b = vec![10.0, 20.0, 30.0, 40.0];
            let t = 0.5;
            let mut avx512_result = vec![0.0; a.len()];
            let mut scalar_result = vec![0.0; a.len()];

            // SAFETY: Test code calling backend trait methods
            unsafe {
                Avx512Backend::lerp(&a, &b, t, &mut avx512_result);
                ScalarBackend::lerp(&a, &b, t, &mut scalar_result);
            }

            for i in 0..a.len() {
                assert!(
                    (avx512_result[i] - scalar_result[i]).abs() < 1e-5,
                    "lerp mismatch at index {}: avx512={}, scalar={}",
                    i,
                    avx512_result[i],
                    scalar_result[i]
                );
            }
        });
    }

    #[test]
    fn test_avx512_fma_scalar_fallback() {
        avx512_test(|| {
            // Test scalar fallback for fma (not yet SIMD optimized)
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let b = vec![2.0, 3.0, 4.0, 5.0];
            let c = vec![1.0, 1.0, 1.0, 1.0];
            let mut avx512_result = vec![0.0; a.len()];
            let mut scalar_result = vec![0.0; a.len()];

            // SAFETY: Test code calling backend trait methods
            unsafe {
                Avx512Backend::fma(&a, &b, &c, &mut avx512_result);
                ScalarBackend::fma(&a, &b, &c, &mut scalar_result);
            }

            for i in 0..a.len() {
                assert!(
                    (avx512_result[i] - scalar_result[i]).abs() < 1e-5,
                    "fma mismatch at index {}: avx512={}, scalar={}",
                    i,
                    avx512_result[i],
                    scalar_result[i]
                );
            }
        });
    }
}
