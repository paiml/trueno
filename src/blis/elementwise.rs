//! SIMD-accelerated element-wise operations.
//!
//! AVX2 implementations of ReLU, vector add, and scalar multiply.
//! These are bandwidth-bound at large sizes; SIMD helps at small-to-medium
//! sizes by reducing instruction count and enabling wider stores.
//!
//! # Algorithm
//!
//! ReLU: `_mm256_max_ps(x, zero)` — single instruction per 8 elements
//! Add: `_mm256_add_ps(a, b)` — single instruction per 8 elements
//! Mul scalar: `_mm256_mul_ps(x, scalar_vec)` — single instruction per 8 elements
//!
//! Contract: provable-contracts/contracts/activation-kernel-v1.yaml

use crate::error::TruenoError;

// ============================================================================
// ReLU
// ============================================================================

/// ReLU: output_i = max(0, input_i)
///
/// Uses AVX2 `_mm256_max_ps` when available.
///
/// # Errors
///
/// Returns `Err` if input and output lengths don't match.
pub fn relu(input: &[f32], output: &mut [f32]) -> Result<(), TruenoError> {
    contract_pre_relu!(input);
    let n = input.len();
    if n != output.len() {
        return Err(TruenoError::InvalidInput(format!(
            "relu size mismatch: input[{}], output[{}]",
            n,
            output.len()
        )));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                relu_avx2(input, output);
            }
            return Ok(());
        }
    }

    for i in 0..n {
        output[i] = input[i].max(0.0);
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn relu_avx2(input: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = input.len();
    let chunks = n / 32; // Process 32 elements (4×8) per iteration
    let remainder_32 = chunks * 32;

    unsafe {
        let zero = _mm256_setzero_ps();

        for i in 0..chunks {
            let base = i * 32;
            let v0 = _mm256_loadu_ps(input.as_ptr().add(base));
            let v1 = _mm256_loadu_ps(input.as_ptr().add(base + 8));
            let v2 = _mm256_loadu_ps(input.as_ptr().add(base + 16));
            let v3 = _mm256_loadu_ps(input.as_ptr().add(base + 24));
            _mm256_storeu_ps(output.as_mut_ptr().add(base), _mm256_max_ps(v0, zero));
            _mm256_storeu_ps(output.as_mut_ptr().add(base + 8), _mm256_max_ps(v1, zero));
            _mm256_storeu_ps(output.as_mut_ptr().add(base + 16), _mm256_max_ps(v2, zero));
            _mm256_storeu_ps(output.as_mut_ptr().add(base + 24), _mm256_max_ps(v3, zero));
        }

        // Handle remaining 8-wide chunks
        let mut i = remainder_32;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_max_ps(v, zero));
            i += 8;
        }

        // Scalar tail
        while i < n {
            output[i] = input[i].max(0.0);
            i += 1;
        }
    }
}

// ============================================================================
// Vector Add
// ============================================================================

/// Element-wise add: output_i = a_i + b_i
///
/// Uses AVX2 `_mm256_add_ps` when available.
///
/// # Errors
///
/// Returns `Err` if a, b, and output lengths don't match.
pub fn add(a: &[f32], b: &[f32], output: &mut [f32]) -> Result<(), TruenoError> {
    contract_pre_add!(a, b);
    let n = a.len();
    if n != b.len() || n != output.len() {
        return Err(TruenoError::InvalidInput(format!(
            "add size mismatch: a[{}], b[{}], output[{}]",
            n,
            b.len(),
            output.len()
        )));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                add_avx2(a, b, output);
            }
            return Ok(());
        }
    }

    for i in 0..n {
        output[i] = a[i] + b[i];
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_avx2(a: &[f32], b: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = a.len();
    let chunks = n / 32;
    let remainder_32 = chunks * 32;

    unsafe {
        for i in 0..chunks {
            let base = i * 32;
            let a0 = _mm256_loadu_ps(a.as_ptr().add(base));
            let a1 = _mm256_loadu_ps(a.as_ptr().add(base + 8));
            let a2 = _mm256_loadu_ps(a.as_ptr().add(base + 16));
            let a3 = _mm256_loadu_ps(a.as_ptr().add(base + 24));
            let b0 = _mm256_loadu_ps(b.as_ptr().add(base));
            let b1 = _mm256_loadu_ps(b.as_ptr().add(base + 8));
            let b2 = _mm256_loadu_ps(b.as_ptr().add(base + 16));
            let b3 = _mm256_loadu_ps(b.as_ptr().add(base + 24));
            _mm256_storeu_ps(output.as_mut_ptr().add(base), _mm256_add_ps(a0, b0));
            _mm256_storeu_ps(output.as_mut_ptr().add(base + 8), _mm256_add_ps(a1, b1));
            _mm256_storeu_ps(output.as_mut_ptr().add(base + 16), _mm256_add_ps(a2, b2));
            _mm256_storeu_ps(output.as_mut_ptr().add(base + 24), _mm256_add_ps(a3, b3));
        }

        let mut i = remainder_32;
        while i + 8 <= n {
            let av = _mm256_loadu_ps(a.as_ptr().add(i));
            let bv = _mm256_loadu_ps(b.as_ptr().add(i));
            _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_add_ps(av, bv));
            i += 8;
        }

        while i < n {
            output[i] = a[i] + b[i];
            i += 1;
        }
    }
}

// ============================================================================
// Scalar Multiply
// ============================================================================

/// Element-wise scalar multiply: output_i = input_i * scalar
///
/// Uses AVX2 `_mm256_mul_ps` when available.
///
/// # Errors
///
/// Returns `Err` if input and output lengths don't match.
pub fn mul_scalar(input: &[f32], scalar: f32, output: &mut [f32]) -> Result<(), TruenoError> {
    // Contract: elementwise-kernel-v1.yaml, equation = mul_scalar
    debug_assert!(!input.is_empty(), "Contract mul_scalar: input is empty");
    debug_assert!(scalar.is_finite(), "Contract mul_scalar: scalar is not finite");
    let n = input.len();
    if n != output.len() {
        return Err(TruenoError::InvalidInput(format!(
            "mul_scalar size mismatch: input[{}], output[{}]",
            n,
            output.len()
        )));
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                mul_scalar_avx2(input, scalar, output);
            }
            return Ok(());
        }
    }

    for i in 0..n {
        output[i] = input[i] * scalar;
    }
    Ok(())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_scalar_avx2(input: &[f32], scalar: f32, output: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = input.len();
    let chunks = n / 32;
    let remainder_32 = chunks * 32;

    unsafe {
        let s = _mm256_set1_ps(scalar);

        for i in 0..chunks {
            let base = i * 32;
            let v0 = _mm256_loadu_ps(input.as_ptr().add(base));
            let v1 = _mm256_loadu_ps(input.as_ptr().add(base + 8));
            let v2 = _mm256_loadu_ps(input.as_ptr().add(base + 16));
            let v3 = _mm256_loadu_ps(input.as_ptr().add(base + 24));
            _mm256_storeu_ps(output.as_mut_ptr().add(base), _mm256_mul_ps(v0, s));
            _mm256_storeu_ps(output.as_mut_ptr().add(base + 8), _mm256_mul_ps(v1, s));
            _mm256_storeu_ps(output.as_mut_ptr().add(base + 16), _mm256_mul_ps(v2, s));
            _mm256_storeu_ps(output.as_mut_ptr().add(base + 24), _mm256_mul_ps(v3, s));
        }

        let mut i = remainder_32;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(input.as_ptr().add(i));
            _mm256_storeu_ps(output.as_mut_ptr().add(i), _mm256_mul_ps(v, s));
            i += 8;
        }

        while i < n {
            output[i] = input[i] * scalar;
            i += 1;
        }
    }
}

// ============================================================================
// Allocating variants (skip zero-initialization)
// ============================================================================

/// ReLU with output allocation. Avoids zero-fill overhead of `vec![0.0; n]`.
///
/// # Safety guarantee
///
/// Output Vec is fully initialized by the SIMD/scalar loop before return.
#[must_use]
pub fn relu_alloc(input: &[f32]) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n];
    let _ = relu(input, &mut output);
    output
}

/// Element-wise add with output allocation. Avoids zero-fill overhead.
///
/// # Panics
///
/// Panics if `a` and `b` have different lengths.
#[must_use]
pub fn add_alloc(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "add_alloc: length mismatch");
    let n = a.len();
    let mut output = vec![0.0f32; n];
    let _ = add(a, b, &mut output);
    output
}

/// Scalar multiply with output allocation. Avoids zero-fill overhead.
#[must_use]
pub fn mul_scalar_alloc(input: &[f32], scalar: f32) -> Vec<f32> {
    let n = input.len();
    let mut output = vec![0.0f32; n];
    let _ = mul_scalar(input, scalar, &mut output);
    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── ReLU tests ────────────────────────────────────────────────────────

    #[test]
    fn test_relu_basic() {
        let input = [-1.0, 0.0, 1.0, -0.5, 2.0, -3.0, 0.1, -0.1];
        let expected = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.1, 0.0];
        let mut output = vec![0.0f32; 8];
        relu(&input, &mut output).unwrap();
        assert_eq!(output, expected);
    }

    #[test]
    fn test_relu_large() {
        let n = 11008; // FFN intermediate size
        let input: Vec<f32> =
            (0..n).map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5).collect();
        let mut output = vec![0.0f32; n];
        relu(&input, &mut output).unwrap();
        for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
            assert_eq!(out, inp.max(0.0), "ReLU mismatch at {i}");
        }
    }

    #[test]
    fn test_relu_avx2_scalar_parity() {
        for n in [1, 7, 8, 15, 16, 31, 32, 63, 64, 128, 4096] {
            let input: Vec<f32> =
                (0..n).map(|i| ((i * 17 + 31) % 1000) as f32 / 500.0 - 1.0).collect();
            let mut output = vec![0.0f32; n];
            relu(&input, &mut output).unwrap();
            for (i, (&inp, &out)) in input.iter().zip(output.iter()).enumerate() {
                assert_eq!(out, inp.max(0.0), "ReLU parity at [{i}] n={n}");
            }
        }
    }

    #[test]
    fn test_relu_error_mismatch() {
        let input = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 3];
        assert!(relu(&input, &mut output).is_err());
    }

    // ── Add tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_add_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        let mut output = vec![0.0f32; 4];
        add(&a, &b, &mut output).unwrap();
        assert_eq!(output, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_add_large() {
        let n = 4096;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let mut output = vec![0.0f32; n];
        add(&a, &b, &mut output).unwrap();
        for i in 0..n {
            assert_eq!(output[i], a[i] + b[i], "Add mismatch at {i}");
        }
    }

    #[test]
    fn test_add_avx2_scalar_parity() {
        for n in [1, 7, 8, 15, 16, 31, 32, 63, 64, 128, 4096] {
            let a: Vec<f32> = (0..n).map(|i| ((i * 17 + 31) % 1000) as f32 / 500.0 - 1.0).collect();
            let b: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) % 1000) as f32 / 500.0 - 1.0).collect();
            let mut output = vec![0.0f32; n];
            add(&a, &b, &mut output).unwrap();
            for i in 0..n {
                assert_eq!(output[i], a[i] + b[i], "Add parity at [{i}] n={n}");
            }
        }
    }

    #[test]
    #[should_panic] // contract_pre_add! panics before Err path in debug builds
    fn test_add_error_mismatch() {
        let a = vec![1.0f32; 4];
        let b = vec![1.0f32; 3];
        let mut output = vec![0.0f32; 4];
        let _ = add(&a, &b, &mut output);
    }

    // ── Mul scalar tests ──────────────────────────────────────────────────

    #[test]
    fn test_mul_scalar_basic() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];
        mul_scalar(&input, 2.5, &mut output).unwrap();
        assert_eq!(output, vec![2.5, 5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_mul_scalar_large() {
        let n = 4096;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut output = vec![0.0f32; n];
        mul_scalar(&input, std::f32::consts::PI, &mut output).unwrap();
        for i in 0..n {
            assert!(
                (output[i] - input[i] * std::f32::consts::PI).abs() < 1e-5,
                "Mul scalar mismatch at {i}"
            );
        }
    }

    #[test]
    fn test_mul_scalar_avx2_scalar_parity() {
        for n in [1, 7, 8, 15, 16, 31, 32, 63, 64, 128, 4096] {
            let input: Vec<f32> =
                (0..n).map(|i| ((i * 17 + 31) % 1000) as f32 / 500.0 - 1.0).collect();
            let mut output = vec![0.0f32; n];
            mul_scalar(&input, std::f32::consts::E, &mut output).unwrap();
            for i in 0..n {
                assert!(
                    (output[i] - input[i] * std::f32::consts::E).abs() < 1e-4,
                    "Mul scalar parity at [{i}] n={n}",
                );
            }
        }
    }

    #[test]
    fn test_mul_scalar_error_mismatch() {
        let input = vec![1.0f32; 4];
        let mut output = vec![0.0f32; 3];
        assert!(mul_scalar(&input, 1.0, &mut output).is_err());
    }
}
