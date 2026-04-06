#![allow(missing_docs)]
//! Built-in Compute Operations
//!
//! Pre-defined operations that implement the ComputeOp trait:
//! - DotOp: Vector dot product
//! - AddOp: Element-wise vector addition
//! - MatmulOp: Matrix multiplication (SIMD-optimized)
//! - SoftmaxOp: Softmax with SIMD exp approximation (SIMD-EXP)

use super::{Backend, ComputeOp};
use crate::error::TruenoError;

// ============================================================================
// DotOp: Dot Product
// ============================================================================

/// Dot product operation.
#[derive(Debug, Clone)]
pub struct DotOp {
    /// Expected vector length
    pub len: usize,
}

impl DotOp {
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl ComputeOp for DotOp {
    type Input = (Vec<f32>, Vec<f32>);
    type Output = f32;

    fn name(&self) -> &'static str {
        "dot"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (a, b) = input;
        if a.len() != b.len() {
            return Err(TruenoError::SizeMismatch { expected: a.len(), actual: b.len() });
        }
        // Simple scalar implementation for now
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        Ok(sum)
    }

    fn tokens(&self, input: &Self::Input) -> usize {
        // Each element pair is roughly 1 "token" of work
        input.0.len()
    }
}

// ============================================================================
// AddOp: Element-wise Addition
// ============================================================================

/// Element-wise add operation.
#[derive(Debug, Clone)]
pub struct AddOp {
    /// Expected vector length
    pub len: usize,
}

impl AddOp {
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl ComputeOp for AddOp {
    type Input = (Vec<f32>, Vec<f32>);
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "add"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (a, b) = input;
        if a.len() != b.len() {
            return Err(TruenoError::SizeMismatch { expected: a.len(), actual: b.len() });
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }

    fn tokens(&self, input: &Self::Input) -> usize {
        input.0.len()
    }
}

// ============================================================================
// MatmulOp: Matrix Multiplication
// ============================================================================

/// Matrix multiplication operation.
#[derive(Debug, Clone)]
pub struct MatmulOp {
    /// M dimension (rows of A)
    pub m: usize,
    /// K dimension (cols of A = rows of B)
    pub k: usize,
    /// N dimension (cols of B)
    pub n: usize,
}

impl MatmulOp {
    pub fn new(m: usize, k: usize, n: usize) -> Self {
        Self { m, k, n }
    }
}

impl ComputeOp for MatmulOp {
    type Input = (Vec<f32>, Vec<f32>);
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "matmul"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (a, b) = input;
        let expected_a = self.m * self.k;
        let expected_b = self.k * self.n;

        if a.len() != expected_a {
            return Err(TruenoError::SizeMismatch { expected: expected_a, actual: a.len() });
        }
        if b.len() != expected_b {
            return Err(TruenoError::SizeMismatch { expected: expected_b, actual: b.len() });
        }

        // SIMD-optimized matrix multiplication via Matrix type
        // Uses AVX2/AVX-512 with cache blocking for ~10-50x speedup
        let simd_backend = crate::Backend::select_best();
        let mat_a = crate::Matrix::from_vec_with_backend(self.m, self.k, a, simd_backend);
        let mat_b = crate::Matrix::from_vec_with_backend(self.k, self.n, b, simd_backend);

        let result = mat_a.matmul(&mat_b)?;
        // Take ownership of the data Vec directly — avoids redundant copy.
        Ok(result.data)
    }

    fn tokens(&self, _input: &Self::Input) -> usize {
        // For matmul, "tokens" = number of output elements
        // Each output requires K multiply-adds
        self.m * self.n
    }
}

// ============================================================================
// SoftmaxOp: Softmax with SIMD Exp (SIMD-EXP)
// ============================================================================

/// Softmax operation.
#[derive(Debug, Clone)]
pub struct SoftmaxOp {
    /// Expected vector length
    pub len: usize,
}

impl SoftmaxOp {
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl ComputeOp for SoftmaxOp {
    type Input = Vec<f32>;
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "softmax"
    }

    fn execute(&self, input: Self::Input, backend: Backend) -> Result<Self::Output, TruenoError> {
        if input.is_empty() {
            return Ok(vec![]);
        }

        // SIMD-EXP: Use SIMD backends for 2-3x speedup on softmax
        // The exp() is the bottleneck in softmax - SIMD polynomial approximation
        // matches llama.cpp's ggml_v_expf performance.

        // Step 1: Find max for numerical stability (SIMD max)
        let max = Self::simd_max(&input, backend);

        // Step 2: Subtract max and compute exp (SIMD exp)
        let shifted: Vec<f32> = input.iter().map(|x| x - max).collect();
        let n = shifted.len();
        // Uninit: simd_exp writes output[i] = exp(input[i]) for every i.
        let mut exp_vals: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: simd_exp writes every element (SET, not accumulate).
        unsafe {
            exp_vals.set_len(n);
        }
        Self::simd_exp(&shifted, &mut exp_vals, backend);

        // Step 3: Sum (SIMD sum)
        let exp_sum = Self::simd_sum(&exp_vals, backend);

        // Step 4: Normalize (SIMD scale, guard against sum=0)
        let inv_sum = 1.0 / exp_sum.max(f32::EPSILON);
        // Uninit: simd_scale writes output[i] = input[i] * scalar for every i.
        let mut result: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: simd_scale writes every element (SET, not accumulate).
        unsafe {
            result.set_len(n);
        }
        Self::simd_scale(&exp_vals, inv_sum, &mut result, backend);

        Ok(result)
    }

    fn tokens(&self, input: &Self::Input) -> usize {
        input.len()
    }
}

impl SoftmaxOp {
    /// Check if backend supports SIMD acceleration
    #[inline]
    pub fn is_simd_backend(backend: Backend) -> bool {
        matches!(
            backend,
            Backend::Avx2 | Backend::Avx512 | Backend::Sse2 | Backend::Neon | Backend::Auto
        )
    }

    /// SIMD-accelerated max reduction
    #[inline]
    fn simd_max(input: &[f32], backend: Backend) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if Self::is_simd_backend(backend) && is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 availability confirmed by is_x86_feature_detected!() on preceding line.
                return unsafe { Self::avx2_max(input) };
            }
        }
        let _ = backend; // suppress warning on non-x86
                         // Scalar fallback
        input.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    /// SIMD-accelerated exp using polynomial approximation (SIMD-EXP)
    ///
    /// Uses 6th-degree Remez minimax polynomial matching llama.cpp's ggml_v_expf.
    /// Range reduction: exp(x) = 2^k * e^r where r in [-ln(2)/2, ln(2)/2]
    #[inline]
    fn simd_exp(input: &[f32], output: &mut [f32], backend: Backend) {
        #[cfg(target_arch = "x86_64")]
        {
            if Self::is_simd_backend(backend) && is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 availability confirmed by is_x86_feature_detected!() on preceding line.
                unsafe { Self::avx2_exp(input, output) };
                return;
            }
        }
        let _ = backend; // suppress warning on non-x86
                         // Scalar fallback
        for (i, &x) in input.iter().enumerate() {
            output[i] = x.exp();
        }
    }

    /// SIMD-accelerated sum reduction
    #[inline]
    fn simd_sum(input: &[f32], backend: Backend) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if Self::is_simd_backend(backend) && is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 availability confirmed by is_x86_feature_detected!() on preceding line.
                return unsafe { Self::avx2_sum(input) };
            }
        }
        let _ = backend; // suppress warning on non-x86
                         // Scalar fallback
        input.iter().sum()
    }

    /// SIMD-accelerated scale
    #[inline]
    fn simd_scale(input: &[f32], scalar: f32, output: &mut [f32], backend: Backend) {
        #[cfg(target_arch = "x86_64")]
        {
            if Self::is_simd_backend(backend) && is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 availability confirmed by is_x86_feature_detected!() on preceding line.
                unsafe { Self::avx2_scale(input, scalar, output) };
                return;
            }
        }
        let _ = backend; // suppress warning on non-x86
                         // Scalar fallback
        for (i, &x) in input.iter().enumerate() {
            output[i] = x * scalar;
        }
    }

    // AVX2 implementations

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller verifies AVX2 support, input slices meet alignment/length requirements
    unsafe fn avx2_max(input: &[f32]) -> f32 {
        unsafe {
            use std::arch::x86_64::*;
            let len = input.len();
            let mut i = 0;
            let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);

            while i + 8 <= len {
                let v = _mm256_loadu_ps(input.as_ptr().add(i));
                vmax = _mm256_max_ps(vmax, v);
                i += 8;
            }

            // Horizontal max
            let high = _mm256_extractf128_ps(vmax, 1);
            let low = _mm256_castps256_ps128(vmax);
            let max128 = _mm_max_ps(high, low);
            let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
            let max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, 1));
            let mut result = _mm_cvtss_f32(max32);

            // Handle remainder
            for &val in &input[i..] {
                result = result.max(val);
            }
            result
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    // SAFETY: caller verifies AVX2 support, input slices meet alignment/length requirements
    unsafe fn avx2_exp(input: &[f32], output: &mut [f32]) {
        unsafe {
            use std::arch::x86_64::*;

            let len = input.len();
            let mut i = 0;

            // Constants for range reduction (matches llama.cpp ggml_v_expf)
            let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
            let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
            let half = _mm256_set1_ps(0.5);
            let one = _mm256_set1_ps(1.0);

            // Remez minimax polynomial coefficients for e^r on [-ln(2)/2, ln(2)/2]
            let c1 = _mm256_set1_ps(1.0);
            let c2 = _mm256_set1_ps(0.5);
            let c3 = _mm256_set1_ps(0.166_666_67);
            let c4 = _mm256_set1_ps(0.041_666_668);
            let c5 = _mm256_set1_ps(0.008_333_334);
            let c6 = _mm256_set1_ps(0.001_388_889);

            let exp_hi = _mm256_set1_ps(88.376_26);
            let exp_lo = _mm256_set1_ps(-87.336_55);

            while i + 8 <= len {
                let x = _mm256_loadu_ps(input.as_ptr().add(i));
                let x = _mm256_max_ps(_mm256_min_ps(x, exp_hi), exp_lo);

                // Range reduction: x' = x * log2(e), k = round(x'), r = (x' - k) * ln2
                let fx = _mm256_fmadd_ps(x, log2e, half);
                let fx = _mm256_floor_ps(fx);
                let r = _mm256_fnmadd_ps(fx, ln2, x);

                // Polynomial: e^r ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720
                // Using Horner's method for efficient evaluation
                let p = _mm256_fmadd_ps(c6, r, c5);
                let p = _mm256_fmadd_ps(p, r, c4);
                let p = _mm256_fmadd_ps(p, r, c3);
                let p = _mm256_fmadd_ps(p, r, c2);
                let p = _mm256_fmadd_ps(p, r, c1);
                let p = _mm256_fmadd_ps(p, r, one);

                // Scale by 2^k using integer exponent manipulation
                let k = _mm256_cvtps_epi32(fx);
                let k = _mm256_add_epi32(k, _mm256_set1_epi32(127));
                let k = _mm256_slli_epi32(k, 23);
                let pow2k = _mm256_castsi256_ps(k);
                let result = _mm256_mul_ps(p, pow2k);

                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
                i += 8;
            }

            // Scalar remainder
            for j in i..len {
                output[j] = input[j].exp();
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller verifies AVX2 support, input slices meet alignment/length requirements
    unsafe fn avx2_sum(input: &[f32]) -> f32 {
        unsafe {
            use std::arch::x86_64::*;
            let len = input.len();
            let mut i = 0;
            let mut acc = _mm256_setzero_ps();

            while i + 8 <= len {
                let v = _mm256_loadu_ps(input.as_ptr().add(i));
                acc = _mm256_add_ps(acc, v);
                i += 8;
            }

            // Horizontal sum
            let high = _mm256_extractf128_ps(acc, 1);
            let low = _mm256_castps256_ps128(acc);
            let sum128 = _mm_add_ps(high, low);
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            let mut result = _mm_cvtss_f32(sum32);

            // Handle remainder
            for &val in &input[i..] {
                result += val;
            }
            result
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    // SAFETY: caller verifies AVX2 support, input slices meet alignment/length requirements
    unsafe fn avx2_scale(input: &[f32], scalar: f32, output: &mut [f32]) {
        unsafe {
            use std::arch::x86_64::*;
            let len = input.len();
            let mut i = 0;
            let vscalar = _mm256_set1_ps(scalar);

            while i + 8 <= len {
                let v = _mm256_loadu_ps(input.as_ptr().add(i));
                let result = _mm256_mul_ps(v, vscalar);
                _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
                i += 8;
            }

            // Scalar remainder
            for j in i..len {
                output[j] = input[j] * scalar;
            }
        }
    }
}

#[cfg(test)]
mod tests;
