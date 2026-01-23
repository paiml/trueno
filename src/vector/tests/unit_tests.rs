use super::super::*;
use crate::Backend;

// Basic construction tests
#[test]
fn test_from_slice() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    assert_eq!(v.len(), 3);
}

#[test]
fn test_from_slice_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    assert_eq!(v.len(), 0);
    assert!(v.is_empty());
}

#[test]
fn test_from_slice_single_element() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.as_slice(), &[42.0]);
    assert_eq!(v.len(), 1);
}

#[test]
fn test_from_slice_with_backend() {
    let v = Vector::from_slice_with_backend(&[1.0, 2.0], Backend::Scalar);
    assert_eq!(v.backend(), Backend::Scalar);
}

#[test]
fn test_scalar_backend_operations() {
    // Test various operations with explicit Scalar backend to cover Backend::Scalar match arms
    let v1 = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0, 4.0], Backend::Scalar);
    let v2 = Vector::from_slice_with_backend(&[4.0, 3.0, 2.0, 1.0], Backend::Scalar);

    // Test dot product (covers line 599)
    let dot = v1.dot(&v2).unwrap();
    assert_eq!(dot, 1.0 * 4.0 + 2.0 * 3.0 + 3.0 * 2.0 + 4.0 * 1.0); // = 20.0

    // Test sum (covers line 856)
    let sum = v1.sum().unwrap();
    assert_eq!(sum, 10.0);

    // Test max (covers line 661)
    let max = v1.max().unwrap();
    assert_eq!(max, 4.0);

    // Test min (covers line 709)
    let min = v1.min().unwrap();
    assert_eq!(min, 1.0);

    // Test argmax (covers line 757)
    let argmax = v1.argmax().unwrap();
    assert_eq!(argmax, 3);

    // Test argmin (covers line 805)
    let argmin = v1.argmin().unwrap();
    assert_eq!(argmin, 0);
}

#[test]
fn test_gpu_and_auto_backend_fallback() {
    // Test operations with GPU/Auto backend which fallback to scalar
    let v1 = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::GPU);
    let v2 = Vector::from_slice_with_backend(&[3.0, 2.0, 1.0], Backend::GPU);

    // These should all work (fallback to scalar)
    let dot = v1.dot(&v2).unwrap();
    assert_eq!(dot, 10.0);

    let sum = v1.sum().unwrap();
    assert_eq!(sum, 6.0);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_avx512_backend_vector_ops() {
    // Test operations with explicit AVX-512 backend to cover Backend::AVX512 match arms
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    // Use large vectors to exercise SIMD paths
    let data1: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let data2: Vec<f32> = (0..1024).map(|i| (1024 - i) as f32).collect();

    let v1 = Vector::from_slice_with_backend(&data1, Backend::AVX512);
    let v2 = Vector::from_slice_with_backend(&data2, Backend::AVX512);

    // Test dot product (covers line 605)
    let dot = v1.dot(&v2).unwrap();
    let expected_dot: f32 = data1.iter().zip(data2.iter()).map(|(a, b)| a * b).sum();
    let dot_rel_err = (dot - expected_dot).abs() / expected_dot.abs().max(1.0);
    assert!(dot_rel_err < 0.01, "dot mismatch: {} vs {}", dot, expected_dot);

    // Test sum
    let sum = v1.sum().unwrap();
    let expected_sum: f32 = data1.iter().sum();
    let sum_rel_err = (sum - expected_sum).abs() / expected_sum.abs().max(1.0);
    assert!(sum_rel_err < 0.01, "sum mismatch: {} vs {}", sum, expected_sum);

    // Test max
    let max = v1.max().unwrap();
    assert_eq!(max, 1023.0);

    // Test min
    let min = v1.min().unwrap();
    assert_eq!(min, 0.0);

    // Test argmax
    let argmax = v1.argmax().unwrap();
    assert_eq!(argmax, 1023);

    // Test argmin
    let argmin = v1.argmin().unwrap();
    assert_eq!(argmin, 0);
}

#[test]
fn test_auto_backend_resolution() {
    let v = Vector::from_slice_with_backend(&[1.0], Backend::Auto);
    // Auto should be resolved to best available backend
    let expected_backend = crate::select_best_available_backend();
    assert_eq!(v.backend(), expected_backend);

    // Verify it's not still Backend::Auto after resolution
    assert_ne!(v.backend(), Backend::Auto);

    // On x86_64, should be a SIMD backend (not Scalar)
    #[cfg(target_arch = "x86_64")]
    {
        assert_ne!(v.backend(), Backend::Scalar);
        assert!(matches!(
            v.backend(),
            Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512
        ));
    }
}

#[test]
fn test_with_alignment_valid() {
    let v = Vector::with_alignment(100, Backend::SSE2, 16).unwrap();
    assert_eq!(v.len(), 100);
    assert_eq!(v.backend(), Backend::SSE2);
}

#[test]
fn test_with_alignment_power_of_two() {
    // Test various power-of-2 alignments
    assert!(Vector::with_alignment(10, Backend::Scalar, 1).is_ok());
    assert!(Vector::with_alignment(10, Backend::Scalar, 2).is_ok());
    assert!(Vector::with_alignment(10, Backend::Scalar, 4).is_ok());
    assert!(Vector::with_alignment(10, Backend::Scalar, 8).is_ok());
    assert!(Vector::with_alignment(10, Backend::Scalar, 16).is_ok());
    assert!(Vector::with_alignment(10, Backend::Scalar, 32).is_ok());
    assert!(Vector::with_alignment(10, Backend::Scalar, 64).is_ok());
}

#[test]
fn test_with_alignment_invalid_zero() {
    let result = Vector::with_alignment(100, Backend::Scalar, 0);
    assert!(result.is_err());
    match result {
        Err(TruenoError::InvalidInput(msg)) => {
            assert!(msg.contains("power of 2"));
            assert!(msg.contains("0"));
        }
        _ => panic!("Expected InvalidInput error for zero alignment"),
    }
}

#[test]
fn test_with_alignment_invalid_not_power_of_two() {
    // Test various non-power-of-2 values
    for alignment in &[3, 5, 6, 7, 9, 10, 12, 15, 17, 20, 24, 31, 33] {
        let result = Vector::with_alignment(100, Backend::Scalar, *alignment);
        assert!(result.is_err(), "Alignment {} should be invalid", alignment);
        match result {
            Err(TruenoError::InvalidInput(msg)) => {
                assert!(
                    msg.contains("power of 2"),
                    "Error message should mention power of 2"
                );
                assert!(
                    msg.contains(&alignment.to_string()),
                    "Error message should mention the invalid alignment"
                );
            }
            _ => panic!(
                "Expected InvalidInput error for non-power-of-2 alignment {}",
                alignment
            ),
        }
    }
}

#[test]
fn test_with_alignment_auto_backend_resolution() {
    let v = Vector::with_alignment(100, Backend::Auto, 16).unwrap();
    // Backend::Auto should be resolved to best available backend
    assert_ne!(v.backend(), Backend::Auto);
}

// Add operation tests
#[test]
fn test_add() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
    let result = a.add(&b).unwrap();
    assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0]);
}

#[test]
fn test_add_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.add(&b).unwrap();
    assert_eq!(result.as_slice(), &[] as &[f32]);
}

#[test]
fn test_add_single() {
    let a = Vector::from_slice(&[1.0]);
    let b = Vector::from_slice(&[2.0]);
    let result = a.add(&b).unwrap();
    assert_eq!(result.as_slice(), &[3.0]);
}

#[test]
fn test_add_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[3.0]);
    let result = a.add(&b);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        TruenoError::SizeMismatch {
            expected: 2,
            actual: 1
        }
    );
}

// Subtract operation tests
#[test]
fn test_sub() {
    let a = Vector::from_slice(&[5.0, 7.0, 9.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.sub(&b).unwrap();
    assert_eq!(result.as_slice(), &[4.0, 5.0, 6.0]);
}

#[test]
fn test_sub_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.sub(&b).unwrap();
    assert_eq!(result.as_slice(), &[] as &[f32]);
}

#[test]
fn test_sub_single() {
    let a = Vector::from_slice(&[5.0]);
    let b = Vector::from_slice(&[2.0]);
    let result = a.sub(&b).unwrap();
    assert_eq!(result.as_slice(), &[3.0]);
}

#[test]
fn test_sub_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[3.0]);
    let result = a.sub(&b);
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        TruenoError::SizeMismatch {
            expected: 2,
            actual: 1
        }
    );
}

#[test]
fn test_sub_negative_result() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
    let result = a.sub(&b).unwrap();
    assert_eq!(result.as_slice(), &[-4.0, -4.0, -4.0]);
}

// Multiply operation tests
#[test]
fn test_mul() {
    let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
    let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
    let result = a.mul(&b).unwrap();
    assert_eq!(result.as_slice(), &[10.0, 18.0, 28.0]);
}

#[test]
fn test_mul_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.mul(&b).unwrap();
    assert_eq!(result.as_slice(), &[] as &[f32]);
}

#[test]
fn test_mul_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0]);
    let result = a.mul(&b);
    assert!(result.is_err());
}

// Division operation tests
#[test]
fn test_div() {
    let a = Vector::from_slice(&[10.0, 20.0, 30.0]);
    let b = Vector::from_slice(&[2.0, 4.0, 5.0]);
    let result = a.div(&b).unwrap();
    assert_eq!(result.as_slice(), &[5.0, 5.0, 6.0]);
}

#[test]
fn test_div_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.div(&b).unwrap();
    assert_eq!(result.as_slice(), &[] as &[f32]);
}

#[test]
fn test_div_single() {
    let a = Vector::from_slice(&[10.0]);
    let b = Vector::from_slice(&[2.0]);
    let result = a.div(&b).unwrap();
    assert_eq!(result.as_slice(), &[5.0]);
}

#[test]
fn test_div_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0]);
    let result = a.div(&b);
    assert!(result.is_err());
}

#[test]
fn test_div_by_one() {
    let a = Vector::from_slice(&[5.0, 10.0, 15.0]);
    let b = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.div(&b).unwrap();
    assert_eq!(result.as_slice(), &[5.0, 10.0, 15.0]);
}

#[test]
fn test_div_fractional() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[2.0, 4.0, 8.0]);
    let result = a.div(&b).unwrap();
    assert_eq!(result.as_slice(), &[0.5, 0.5, 0.375]);
}

// Dot product tests
#[test]
fn test_dot() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
    let result = a.dot(&b).unwrap();
    assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
}

#[test]
fn test_dot_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.dot(&b).unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn test_dot_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[3.0]);
    let result = a.dot(&b);
    assert!(result.is_err());
}

// Sum tests
#[test]
fn test_sum() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(v.sum().unwrap(), 10.0);
}

#[test]
fn test_sum_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    assert_eq!(v.sum().unwrap(), 0.0);
}

#[test]
fn test_sum_single() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.sum().unwrap(), 42.0);
}

// Kahan summation tests (numerically stable)
#[test]
fn test_sum_kahan() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(v.sum_kahan().unwrap(), 10.0);
}

#[test]
fn test_sum_kahan_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    assert_eq!(v.sum_kahan().unwrap(), 0.0);
}

#[test]
fn test_sum_kahan_single() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.sum_kahan().unwrap(), 42.0);
}

#[test]
fn test_sum_kahan_numerical_stability() {
    // Test case that demonstrates rounding error accumulation
    // Using many small values that can lose precision
    let mut data = vec![1e-7f32; 10_000];
    data.push(1.0);

    let v = Vector::from_slice(&data);
    let kahan_result = v.sum_kahan().unwrap();
    let naive_result = v.sum().unwrap();

    // Expected: 1.0 + 10000 * 1e-7 = 1.001
    let expected = 1.001f32;

    // Kahan should be more accurate than naive sum
    let kahan_error = (kahan_result - expected).abs();
    let naive_error = (naive_result - expected).abs();

    // Kahan error should be smaller (or at most equal)
    assert!(
        kahan_error <= naive_error,
        "Kahan sum error ({}) should be <= naive sum error ({})",
        kahan_error,
        naive_error
    );
}

// Max tests
#[test]
fn test_max() {
    let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    assert_eq!(v.max().unwrap(), 5.0);
}

#[test]
fn test_max_single() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.max().unwrap(), 42.0);
}

#[test]
fn test_max_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.max();
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        TruenoError::InvalidInput("Empty vector".to_string())
    );
}

#[test]
fn test_max_negative() {
    let v = Vector::from_slice(&[-5.0, -1.0, -10.0, -3.0]);
    assert_eq!(v.max().unwrap(), -1.0);
}

#[test]
fn test_min() {
    let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    assert_eq!(v.min().unwrap(), 1.0);
}

#[test]
fn test_min_single() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.min().unwrap(), 42.0);
}

#[test]
fn test_min_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.min();
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        TruenoError::InvalidInput("Empty vector".to_string())
    );
}

#[test]
fn test_min_negative() {
    let v = Vector::from_slice(&[-5.0, -1.0, -10.0, -3.0]);
    assert_eq!(v.min().unwrap(), -10.0);
}

#[test]
fn test_argmax() {
    let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    assert_eq!(v.argmax().unwrap(), 1); // max value 5.0 is at index 1
}

#[test]
fn test_argmax_single() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.argmax().unwrap(), 0);
}

#[test]
fn test_argmax_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.argmax();
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        TruenoError::InvalidInput("Empty vector".to_string())
    );
}

#[test]
fn test_argmax_negative() {
    let v = Vector::from_slice(&[-5.0, -1.0, -10.0, -3.0]);
    assert_eq!(v.argmax().unwrap(), 1); // max value -1.0 is at index 1
}

#[test]
fn test_argmax_first_occurrence() {
    // When there are duplicates, should return first occurrence
    let v = Vector::from_slice(&[1.0, 5.0, 3.0, 5.0, 2.0]);
    assert_eq!(v.argmax().unwrap(), 1); // first 5.0 is at index 1
}

#[test]
fn test_argmin() {
    let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    assert_eq!(v.argmin().unwrap(), 0); // min value 1.0 is at index 0
}

#[test]
fn test_argmin_single() {
    let v = Vector::from_slice(&[42.0]);
    assert_eq!(v.argmin().unwrap(), 0);
}

#[test]
fn test_argmin_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.argmin();
    assert!(result.is_err());
    assert_eq!(
        result.unwrap_err(),
        TruenoError::InvalidInput("Empty vector".to_string())
    );
}

#[test]
fn test_argmin_negative() {
    let v = Vector::from_slice(&[-5.0, -1.0, -10.0, -3.0]);
    assert_eq!(v.argmin().unwrap(), 2); // min value -10.0 is at index 2
}

#[test]
fn test_argmin_first_occurrence() {
    // When there are duplicates, should return first occurrence
    let v = Vector::from_slice(&[5.0, 1.0, 3.0, 1.0, 2.0]);
    assert_eq!(v.argmin().unwrap(), 1); // first 1.0 is at index 1
}

// L2 norm (Euclidean norm) tests
#[test]
fn test_norm_l2() {
    let v = Vector::from_slice(&[3.0, 4.0]);
    let result = v.norm_l2().unwrap();
    assert!((result - 5.0).abs() < 1e-5); // sqrt(3^2 + 4^2) = 5
}

#[test]
fn test_norm_l2_single() {
    let v = Vector::from_slice(&[7.0]);
    let result = v.norm_l2().unwrap();
    assert!((result - 7.0).abs() < 1e-5);
}

#[test]
fn test_norm_l2_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.norm_l2().unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn test_norm_l2_unit_vector() {
    let v = Vector::from_slice(&[1.0, 0.0, 0.0]);
    let result = v.norm_l2().unwrap();
    assert!((result - 1.0).abs() < 1e-5);
}

#[test]
fn test_norm_l2_negative() {
    let v = Vector::from_slice(&[-3.0, -4.0]);
    let result = v.norm_l2().unwrap();
    assert!((result - 5.0).abs() < 1e-5); // sqrt((-3)^2 + (-4)^2) = 5
}

// Normalize (unit vector) tests
#[test]
fn test_normalize() {
    let v = Vector::from_slice(&[3.0, 4.0]);
    let result = v.normalize().unwrap();
    // Should be [0.6, 0.8] (3/5, 4/5)
    assert!((result.as_slice()[0] - 0.6).abs() < 1e-5);
    assert!((result.as_slice()[1] - 0.8).abs() < 1e-5);
    // Verify it's a unit vector
    let norm = result.norm_l2().unwrap();
    assert!((norm - 1.0).abs() < 1e-5);
}

#[test]
fn test_normalize_already_unit() {
    let v = Vector::from_slice(&[1.0, 0.0, 0.0]);
    let result = v.normalize().unwrap();
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-5);
    assert!((result.as_slice()[1] - 0.0).abs() < 1e-5);
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
}

#[test]
fn test_normalize_zero_vector() {
    let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = v.normalize();
    // Should error on zero vector (division by zero norm)
    assert!(result.is_err());
}

#[test]
fn test_normalize_negative() {
    let v = Vector::from_slice(&[-3.0, -4.0]);
    let result = v.normalize().unwrap();
    // Should be [-0.6, -0.8]
    assert!((result.as_slice()[0] - (-0.6)).abs() < 1e-5);
    assert!((result.as_slice()[1] - (-0.8)).abs() < 1e-5);
    let norm = result.norm_l2().unwrap();
    assert!((norm - 1.0).abs() < 1e-5);
}

// L1 Norm (Manhattan norm) tests
#[test]
fn test_norm_l1_basic() {
    let v = Vector::from_slice(&[3.0, -4.0, 5.0]);
    let result = v.norm_l1().unwrap();
    // |3| + |-4| + |5| = 3 + 4 + 5 = 12
    assert!((result - 12.0).abs() < 1e-5);
}

#[test]
fn test_norm_l1_all_positive() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let result = v.norm_l1().unwrap();
    // 1 + 2 + 3 + 4 = 10
    assert!((result - 10.0).abs() < 1e-5);
}

#[test]
fn test_norm_l1_all_negative() {
    let v = Vector::from_slice(&[-1.0, -2.0, -3.0]);
    let result = v.norm_l1().unwrap();
    // |-1| + |-2| + |-3| = 1 + 2 + 3 = 6
    assert!((result - 6.0).abs() < 1e-5);
}

#[test]
fn test_norm_l1_zero_vector() {
    let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = v.norm_l1().unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn test_norm_l1_empty_vector() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.norm_l1().unwrap();
    assert_eq!(result, 0.0);
}

// L∞ Norm (infinity/max norm) tests
#[test]
fn test_norm_linf_basic() {
    let v = Vector::from_slice(&[3.0, -7.0, 5.0, -2.0]);
    let result = v.norm_linf().unwrap();
    // max(|3|, |-7|, |5|, |-2|) = max(3, 7, 5, 2) = 7
    assert!((result - 7.0).abs() < 1e-5);
}

#[test]
fn test_norm_linf_all_positive() {
    let v = Vector::from_slice(&[1.0, 2.0, 5.0, 3.0]);
    let result = v.norm_linf().unwrap();
    // max(1, 2, 5, 3) = 5
    assert!((result - 5.0).abs() < 1e-5);
}

#[test]
fn test_norm_linf_all_negative() {
    let v = Vector::from_slice(&[-1.0, -9.0, -3.0]);
    let result = v.norm_linf().unwrap();
    // max(|-1|, |-9|, |-3|) = max(1, 9, 3) = 9
    assert!((result - 9.0).abs() < 1e-5);
}

#[test]
fn test_norm_linf_zero_vector() {
    let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = v.norm_linf().unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn test_norm_linf_empty_vector() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.norm_linf().unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn test_norm_linf_single_element() {
    let v = Vector::from_slice(&[-42.5]);
    let result = v.norm_linf().unwrap();
    assert!((result - 42.5).abs() < 1e-5);
}

// Absolute value tests
#[test]
fn test_abs_mixed() {
    let v = Vector::from_slice(&[3.0, -4.0, 5.0, -2.0]);
    let result = v.abs().unwrap();
    assert_eq!(result.as_slice(), &[3.0, 4.0, 5.0, 2.0]);
}

#[test]
fn test_abs_all_positive() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.abs().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_abs_all_negative() {
    let v = Vector::from_slice(&[-1.0, -2.0, -3.0]);
    let result = v.abs().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_abs_with_zeros() {
    let v = Vector::from_slice(&[0.0, -5.0, 0.0, 3.0]);
    let result = v.abs().unwrap();
    assert_eq!(result.as_slice(), &[0.0, 5.0, 0.0, 3.0]);
}

#[test]
fn test_abs_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.abs().unwrap();
    assert_eq!(result.len(), 0);
}

// Scalar multiplication (scale) tests
#[test]
fn test_scale_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let result = v.scale(2.0).unwrap();
    assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_scale_by_zero() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.scale(0.0).unwrap();
    assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
}

#[test]
fn test_scale_by_negative() {
    let v = Vector::from_slice(&[1.0, -2.0, 3.0]);
    let result = v.scale(-2.0).unwrap();
    assert_eq!(result.as_slice(), &[-2.0, 4.0, -6.0]);
}

#[test]
fn test_scale_by_one() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.scale(1.0).unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_scale_by_fraction() {
    let v = Vector::from_slice(&[2.0, 4.0, 6.0]);
    let result = v.scale(0.5).unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_scale_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.scale(2.0).unwrap();
    assert_eq!(result.len(), 0);
}

// Clamp tests
#[test]
fn test_clamp_basic() {
    let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
    let result = v.clamp(0.0, 10.0).unwrap();
    assert_eq!(result.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
}

#[test]
fn test_clamp_all_within_range() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.clamp(0.0, 10.0).unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_clamp_all_below_min() {
    let v = Vector::from_slice(&[-5.0, -3.0, -1.0]);
    let result = v.clamp(0.0, 10.0).unwrap();
    assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
}

#[test]
fn test_clamp_all_above_max() {
    let v = Vector::from_slice(&[15.0, 20.0, 25.0]);
    let result = v.clamp(0.0, 10.0).unwrap();
    assert_eq!(result.as_slice(), &[10.0, 10.0, 10.0]);
}

#[test]
fn test_clamp_negative_range() {
    let v = Vector::from_slice(&[-10.0, -5.0, 0.0, 5.0]);
    let result = v.clamp(-8.0, -2.0).unwrap();
    assert_eq!(result.as_slice(), &[-8.0, -5.0, -2.0, -2.0]);
}

#[test]
fn test_clamp_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.clamp(0.0, 10.0).unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_clamp_same_min_max() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.clamp(2.5, 2.5).unwrap();
    assert_eq!(result.as_slice(), &[2.5, 2.5, 2.5]);
}

#[test]
fn test_clamp_invalid_range() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.clamp(10.0, 0.0); // min > max
    assert!(result.is_err());
}

// Linear interpolation (lerp) tests
#[test]
fn test_lerp_basic() {
    let a = Vector::from_slice(&[0.0, 10.0, 20.0]);
    let b = Vector::from_slice(&[100.0, 110.0, 120.0]);
    let result = a.lerp(&b, 0.5).unwrap();
    assert_eq!(result.as_slice(), &[50.0, 60.0, 70.0]);
}

#[test]
fn test_lerp_at_zero() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
    let result = a.lerp(&b, 0.0).unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]); // Should return a
}

#[test]
fn test_lerp_at_one() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
    let result = a.lerp(&b, 1.0).unwrap();
    assert_eq!(result.as_slice(), &[4.0, 5.0, 6.0]); // Should return b
}

#[test]
fn test_lerp_extrapolate_above() {
    let a = Vector::from_slice(&[0.0, 10.0]);
    let b = Vector::from_slice(&[10.0, 20.0]);
    let result = a.lerp(&b, 2.0).unwrap();
    assert_eq!(result.as_slice(), &[20.0, 30.0]); // Extrapolation beyond b
}

#[test]
fn test_lerp_extrapolate_below() {
    let a = Vector::from_slice(&[10.0, 20.0]);
    let b = Vector::from_slice(&[20.0, 30.0]);
    let result = a.lerp(&b, -1.0).unwrap();
    assert_eq!(result.as_slice(), &[0.0, 10.0]); // Extrapolation before a
}

#[test]
fn test_lerp_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.lerp(&b, 0.5);
    assert!(result.is_err());
}

#[test]
fn test_lerp_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.lerp(&b, 0.5).unwrap();
    assert_eq!(result.len(), 0);
}

// fma() operation tests (fused multiply-add: a * b + c)
#[test]
fn test_fma_basic() {
    let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
    let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
    let c = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.fma(&b, &c).unwrap();
    // Expected: [2*5+1, 3*6+2, 4*7+3] = [11, 20, 31]
    assert_eq!(result.as_slice(), &[11.0, 20.0, 31.0]);
}

#[test]
fn test_fma_zeros() {
    let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
    let c = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.fma(&b, &c).unwrap();
    // Expected: [0*5+1, 0*6+2, 0*7+3] = [1, 2, 3]
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_fma_ones() {
    let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
    let b = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let c = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = a.fma(&b, &c).unwrap();
    // Expected: [2*1+0, 3*1+0, 4*1+0] = [2, 3, 4]
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0]);
}

#[test]
fn test_fma_negatives() {
    let a = Vector::from_slice(&[-2.0, 3.0, -4.0]);
    let b = Vector::from_slice(&[5.0, -6.0, 7.0]);
    let c = Vector::from_slice(&[1.0, 2.0, -3.0]);
    let result = a.fma(&b, &c).unwrap();
    // Expected: [-2*5+1, 3*(-6)+2, -4*7+(-3)] = [-9, -16, -31]
    assert_eq!(result.as_slice(), &[-9.0, -16.0, -31.0]);
}

#[test]
fn test_fma_size_mismatch_b() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let c = Vector::from_slice(&[1.0, 2.0]);
    let result = a.fma(&b, &c);
    assert!(result.is_err());
}

#[test]
fn test_fma_size_mismatch_c() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[1.0, 2.0]);
    let c = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.fma(&b, &c);
    assert!(result.is_err());
}

#[test]
fn test_fma_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let c: Vector<f32> = Vector::from_slice(&[]);
    let result = a.fma(&b, &c).unwrap();
    assert_eq!(result.len(), 0);
}

// sqrt() operation tests (element-wise square root)
#[test]
fn test_sqrt_basic() {
    let a = Vector::from_slice(&[4.0, 9.0, 16.0, 25.0]);
    let result = a.sqrt().unwrap();
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_sqrt_zeros() {
    let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = a.sqrt().unwrap();
    assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
}

#[test]
fn test_sqrt_one() {
    let a = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.sqrt().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0]);
}

#[test]
fn test_sqrt_fractional() {
    let a = Vector::from_slice(&[0.25, 0.01, 0.0625]);
    let result = a.sqrt().unwrap();
    assert_eq!(result.as_slice(), &[0.5, 0.1, 0.25]);
}

#[test]
fn test_sqrt_negative() {
    let a = Vector::from_slice(&[-1.0, 4.0, -9.0]);
    let result = a.sqrt().unwrap();
    // Negative values produce NaN
    assert!(result.as_slice()[0].is_nan());
    assert_eq!(result.as_slice()[1], 4.0_f32.sqrt());
    assert!(result.as_slice()[2].is_nan());
}

#[test]
fn test_sqrt_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.sqrt().unwrap();
    assert_eq!(result.len(), 0);
}

// recip() operation tests (element-wise reciprocal: 1/x)
#[test]
fn test_recip_basic() {
    let a = Vector::from_slice(&[2.0, 4.0, 5.0, 10.0]);
    let result = a.recip().unwrap();
    assert_eq!(result.as_slice(), &[0.5, 0.25, 0.2, 0.1]);
}

#[test]
fn test_recip_ones() {
    let a = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.recip().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0]);
}

#[test]
fn test_recip_negatives() {
    let a = Vector::from_slice(&[-2.0, -4.0, -0.5]);
    let result = a.recip().unwrap();
    assert_eq!(result.as_slice(), &[-0.5, -0.25, -2.0]);
}

#[test]
fn test_recip_fractional() {
    let a = Vector::from_slice(&[0.5, 0.25, 0.1]);
    let result = a.recip().unwrap();
    assert_eq!(result.as_slice(), &[2.0, 4.0, 10.0]);
}

#[test]
fn test_recip_zero() {
    let a = Vector::from_slice(&[0.0, 2.0, 0.0]);
    let result = a.recip().unwrap();
    // 1/0 = infinity
    assert!(result.as_slice()[0].is_infinite());
    assert_eq!(result.as_slice()[1], 0.5);
    assert!(result.as_slice()[2].is_infinite());
}

#[test]
fn test_recip_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.recip().unwrap();
    assert_eq!(result.len(), 0);
}

// pow() operation tests (element-wise power: x^n)
#[test]
fn test_pow_basic() {
    let a = Vector::from_slice(&[2.0, 3.0, 4.0, 5.0]);
    let result = a.pow(2.0).unwrap();
    assert_eq!(result.as_slice(), &[4.0, 9.0, 16.0, 25.0]);
}

#[test]
fn test_pow_cube() {
    let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
    let result = a.pow(3.0).unwrap();
    assert_eq!(result.as_slice(), &[8.0, 27.0, 64.0]);
}

#[test]
fn test_pow_fractional() {
    let a = Vector::from_slice(&[4.0, 9.0, 16.0]);
    let result = a.pow(0.5).unwrap();
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0]); // Square root
}

#[test]
fn test_pow_zero_exponent() {
    let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
    let result = a.pow(0.0).unwrap();
    assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0]); // x^0 = 1
}

#[test]
fn test_pow_one_exponent() {
    let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
    let result = a.pow(1.0).unwrap();
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0]); // x^1 = x
}

#[test]
fn test_pow_negative_exponent() {
    let a = Vector::from_slice(&[2.0, 4.0, 10.0]);
    let result = a.pow(-1.0).unwrap();
    assert_eq!(result.as_slice(), &[0.5, 0.25, 0.1]); // x^(-1) = 1/x
}

#[test]
fn test_pow_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.pow(2.0).unwrap();
    assert_eq!(result.len(), 0);
}

// exp() operation tests (element-wise exponential: e^x)
#[test]
fn test_exp_basic() {
    let a = Vector::from_slice(&[0.0, 1.0, 2.0]);
    let result = a.exp().unwrap();
    let expected = [1.0, std::f32::consts::E, std::f32::consts::E.powi(2)];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "exp mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_exp_zero() {
    let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = a.exp().unwrap();
    for &val in result.as_slice() {
        assert!((val - 1.0).abs() < 1e-5, "e^0 should be 1.0");
    }
}

#[test]
fn test_exp_negative() {
    let a = Vector::from_slice(&[-1.0, -2.0, -3.0]);
    let result = a.exp().unwrap();
    let expected = [(-1.0f32).exp(), (-2.0f32).exp(), (-3.0f32).exp()];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "exp negative mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_exp_large_positive() {
    let a = Vector::from_slice(&[5.0, 10.0]);
    let result = a.exp().unwrap();
    let expected = [5.0f32.exp(), 10.0f32.exp()];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() / exp < 1e-5,
            "exp large positive mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_exp_large_negative() {
    let a = Vector::from_slice(&[-5.0, -10.0]);
    let result = a.exp().unwrap();
    let expected = [(-5.0f32).exp(), (-10.0f32).exp()];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "exp large negative mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_exp_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.exp().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_exp_backend_auto_unsupported() {
    // EXTREME TDD: Kill mutant that deletes Backend::Auto match arm
    // Direct construction bypasses normal backend resolution (testing defensive code)
    let v = Vector {
        data: vec![1.0, 2.0, 3.0],
        backend: Backend::Auto,
    };

    let result = v.exp();

    // Should return UnsupportedBackend error, not fall through to wildcard
    assert!(result.is_err(), "exp() should error for Backend::Auto");
    match result.unwrap_err() {
        TruenoError::UnsupportedBackend(Backend::Auto) => {
            // Expected error
        }
        other => panic!("Expected UnsupportedBackend(Auto), got {:?}", other),
    }
}

// ln() operation tests (element-wise natural logarithm: ln(x))
#[test]
fn test_ln_basic() {
    let a = Vector::from_slice(&[1.0, std::f32::consts::E, std::f32::consts::E.powi(2)]);
    let result = a.ln().unwrap();
    let expected = [0.0, 1.0, 2.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "ln mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_ln_one() {
    let a = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.ln().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-5, "ln(1) should be 0.0");
    }
}

#[test]
fn test_ln_small_values() {
    let a = Vector::from_slice(&[0.1, 0.5, 0.9]);
    let result = a.ln().unwrap();
    let expected = [0.1f32.ln(), 0.5f32.ln(), 0.9f32.ln()];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "ln small values mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_ln_large_values() {
    let a = Vector::from_slice(&[10.0, 100.0, 1000.0]);
    let result = a.ln().unwrap();
    let expected = [10.0f32.ln(), 100.0f32.ln(), 1000.0f32.ln()];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "ln large values mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_ln_inverse_exp() {
    // Test that ln(exp(x)) = x
    let a = Vector::from_slice(&[0.5, 1.0, 2.0, 3.0]);
    let exp_result = a.exp().unwrap();
    let ln_result = exp_result.ln().unwrap();
    for (i, (&original, &recovered)) in a
        .as_slice()
        .iter()
        .zip(ln_result.as_slice().iter())
        .enumerate()
    {
        assert!(
            (original - recovered).abs() < 1e-5,
            "ln(exp(x)) != x at {}: {} != {}",
            i,
            original,
            recovered
        );
    }
}

#[test]
fn test_ln_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.ln().unwrap();
    assert_eq!(result.len(), 0);
}

// log2() operation tests (element-wise base-2 logarithm)

#[test]
fn test_log2_basic() {
    let a = Vector::from_slice(&[1.0, 2.0, 4.0, 8.0, 16.0]);
    let result = a.log2().unwrap();
    let expected = [0.0, 1.0, 2.0, 3.0, 4.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "log2 mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_log2_one() {
    let a = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.log2().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-5, "log2(1) should be 0.0");
    }
}

#[test]
fn test_log2_fractional() {
    let a = Vector::from_slice(&[0.5, 0.25, 0.125]);
    let result = a.log2().unwrap();
    let expected = [-1.0, -2.0, -3.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "log2 fractional mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_log2_non_powers() {
    let a = Vector::from_slice(&[3.0, 5.0, 10.0]);
    let result = a.log2().unwrap();
    let expected = [3.0f32.log2(), 5.0f32.log2(), 10.0f32.log2()];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "log2 non-powers mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_log2_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.log2().unwrap();
    assert_eq!(result.len(), 0);
}

// log10() operation tests (element-wise base-10 logarithm)

#[test]
fn test_log10_basic() {
    let a = Vector::from_slice(&[1.0, 10.0, 100.0, 1000.0]);
    let result = a.log10().unwrap();
    let expected = [0.0, 1.0, 2.0, 3.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "log10 mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_log10_one() {
    let a = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let result = a.log10().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-5, "log10(1) should be 0.0");
    }
}

#[test]
fn test_log10_fractional() {
    let a = Vector::from_slice(&[0.1, 0.01, 0.001]);
    let result = a.log10().unwrap();
    let expected = [-1.0, -2.0, -3.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "log10 fractional mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_log10_non_powers() {
    let a = Vector::from_slice(&[2.0, 5.0, 50.0]);
    let result = a.log10().unwrap();
    let expected = [2.0f32.log10(), 5.0f32.log10(), 50.0f32.log10()];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "log10 non-powers mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_log10_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.log10().unwrap();
    assert_eq!(result.len(), 0);
}

// sin() operation tests (element-wise sine)
#[test]
fn test_sin_basic() {
    use std::f32::consts::PI;
    let a = Vector::from_slice(&[0.0, PI / 2.0, PI, 3.0 * PI / 2.0]);
    let result = a.sin().unwrap();
    let expected = [0.0, 1.0, 0.0, -1.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "sin mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_sin_zero() {
    let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = a.sin().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-5, "sin(0) should be 0.0");
    }
}

#[test]
fn test_sin_quarter_circle() {
    use std::f32::consts::PI;
    let a = Vector::from_slice(&[PI / 6.0, PI / 4.0, PI / 3.0]);
    let result = a.sin().unwrap();
    let expected = [0.5, std::f32::consts::FRAC_1_SQRT_2, (3.0f32).sqrt() / 2.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "sin quarter circle mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_sin_negative() {
    use std::f32::consts::PI;
    let a = Vector::from_slice(&[-PI / 2.0, -PI, -3.0 * PI / 2.0]);
    let result = a.sin().unwrap();
    let expected = [-1.0, 0.0, 1.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "sin negative mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_sin_periodicity() {
    use std::f32::consts::PI;
    // sin(x + 2π) = sin(x)
    let a = Vector::from_slice(&[0.5, 1.0, 1.5]);
    let b = Vector::from_slice(&[0.5 + 2.0 * PI, 1.0 + 2.0 * PI, 1.5 + 2.0 * PI]);
    let result_a = a.sin().unwrap();
    let result_b = b.sin().unwrap();
    for (i, (&res_a, &res_b)) in result_a
        .as_slice()
        .iter()
        .zip(result_b.as_slice().iter())
        .enumerate()
    {
        assert!(
            (res_a - res_b).abs() < 1e-5,
            "sin periodicity failed at {}: {} != {}",
            i,
            res_a,
            res_b
        );
    }
}

#[test]
fn test_sin_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.sin().unwrap();
    assert_eq!(result.len(), 0);
}

// cos() operation tests (element-wise cosine)
#[test]
fn test_cos_basic() {
    use std::f32::consts::PI;
    let a = Vector::from_slice(&[0.0, PI / 2.0, PI, 3.0 * PI / 2.0]);
    let result = a.cos().unwrap();
    let expected = [1.0, 0.0, -1.0, 0.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "cos mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_cos_zero() {
    let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = a.cos().unwrap();
    for &val in result.as_slice() {
        assert!((val - 1.0).abs() < 1e-5, "cos(0) should be 1.0");
    }
}

#[test]
fn test_cos_quarter_circle() {
    use std::f32::consts::PI;
    let a = Vector::from_slice(&[PI / 6.0, PI / 4.0, PI / 3.0]);
    let result = a.cos().unwrap();
    let expected = [(3.0f32).sqrt() / 2.0, std::f32::consts::FRAC_1_SQRT_2, 0.5];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "cos quarter circle mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_cos_negative() {
    use std::f32::consts::PI;
    let a = Vector::from_slice(&[-PI / 2.0, -PI, -3.0 * PI / 2.0]);
    let result = a.cos().unwrap();
    let expected = [0.0, -1.0, 0.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "cos negative mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_cos_sin_relation() {
    use std::f32::consts::PI;
    // cos(x) = sin(x + π/2)
    let a = Vector::from_slice(&[0.0, PI / 6.0, PI / 4.0, PI / 3.0]);
    let cos_result = a.cos().unwrap();

    let a_plus_pi_2: Vec<f32> = a.as_slice().iter().map(|&x| x + PI / 2.0).collect();
    let shifted = Vector::from_slice(&a_plus_pi_2);
    let sin_result = shifted.sin().unwrap();

    for (i, (&cos_val, &sin_val)) in cos_result
        .as_slice()
        .iter()
        .zip(sin_result.as_slice().iter())
        .enumerate()
    {
        assert!(
            (cos_val - sin_val).abs() < 1e-5,
            "cos(x) = sin(x + π/2) failed at {}: {} != {}",
            i,
            cos_val,
            sin_val
        );
    }
}

#[test]
fn test_cos_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.cos().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_tan_basic() {
    use std::f32::consts::PI;
    // tan(0) = 0, tan(π/4) = 1, tan(-π/4) = -1
    let a = Vector::from_slice(&[0.0, PI / 4.0, -PI / 4.0]);
    let result = a.tan().unwrap();
    let expected = [0.0, 1.0, -1.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "tan basic mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_tan_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.tan().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_tan_quarter_circle() {
    use std::f32::consts::PI;
    // tan(π/4) = 1
    let a = Vector::from_slice(&[PI / 4.0]);
    let result = a.tan().unwrap();
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_tan_negative() {
    use std::f32::consts::PI;
    // tan is odd: tan(-x) = -tan(x)
    let a = Vector::from_slice(&[-PI / 4.0, -PI / 6.0]);
    let result = a.tan().unwrap();
    let expected = [-1.0, -(1.0 / 3.0_f32.sqrt())];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "tan negative mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_tan_sin_cos_relation() {
    use std::f32::consts::PI;
    // tan(x) = sin(x) / cos(x)
    let a = Vector::from_slice(&[PI / 6.0, PI / 4.0, PI / 3.0]);
    let tan_result = a.tan().unwrap();
    let sin_result = a.sin().unwrap();
    let cos_result = a.cos().unwrap();

    for (i, ((&tan_val, &sin_val), &cos_val)) in tan_result
        .as_slice()
        .iter()
        .zip(sin_result.as_slice().iter())
        .zip(cos_result.as_slice().iter())
        .enumerate()
    {
        let expected = sin_val / cos_val;
        assert!(
            (tan_val - expected).abs() < 1e-5,
            "tan(x) != sin(x)/cos(x) at {}: {} != {}",
            i,
            tan_val,
            expected
        );
    }
}

#[test]
fn test_tan_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.tan().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_asin_basic() {
    use std::f32::consts::PI;
    // asin(0) = 0, asin(1) = π/2, asin(-1) = -π/2, asin(0.5) = π/6
    let a = Vector::from_slice(&[0.0, 1.0, -1.0, 0.5]);
    let result = a.asin().unwrap();
    let expected = [0.0, PI / 2.0, -PI / 2.0, PI / 6.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "asin basic mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_asin_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.asin().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_asin_range() {
    use std::f32::consts::PI;
    // asin domain is [-1, 1], range is [-π/2, π/2]
    let a = Vector::from_slice(&[-1.0, -0.5, 0.0, 0.5, 1.0]);
    let result = a.asin().unwrap();
    for (i, &res) in result.as_slice().iter().enumerate() {
        assert!(
            (-PI / 2.0..=PI / 2.0).contains(&res),
            "asin range violation at {}: {} not in [-π/2, π/2]",
            i,
            res
        );
    }
}

#[test]
fn test_asin_negative() {
    use std::f32::consts::PI;
    // asin is odd: asin(-x) = -asin(x)
    let a = Vector::from_slice(&[-0.5, -0.707]);
    let result = a.asin().unwrap();
    let expected = [-PI / 6.0, -PI / 4.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-3,
            "asin negative mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_asin_sin_inverse() {
    use std::f32::consts::PI;
    // asin(sin(x)) = x for x in [-π/2, π/2]
    let a = Vector::from_slice(&[-PI / 4.0, 0.0, PI / 6.0, PI / 4.0]);
    let sin_result = a.sin().unwrap();
    let asin_result = sin_result.asin().unwrap();

    for (i, (&original, &reconstructed)) in a
        .as_slice()
        .iter()
        .zip(asin_result.as_slice().iter())
        .enumerate()
    {
        assert!(
            (original - reconstructed).abs() < 1e-5,
            "asin(sin(x)) != x at {}: {} != {}",
            i,
            reconstructed,
            original
        );
    }
}

#[test]
fn test_asin_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.asin().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_acos_basic() {
    use std::f32::consts::PI;
    // acos(0) = π/2, acos(1) = 0, acos(-1) = π, acos(0.5) = π/3
    let a = Vector::from_slice(&[0.0, 1.0, -1.0, 0.5]);
    let result = a.acos().unwrap();
    let expected = [PI / 2.0, 0.0, PI, PI / 3.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "acos basic mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_acos_zero() {
    use std::f32::consts::PI;
    let a = Vector::from_slice(&[1.0]);
    let result = a.acos().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);

    // Also test acos(0) = π/2
    let b = Vector::from_slice(&[0.0]);
    let result_b = b.acos().unwrap();
    assert!((result_b.as_slice()[0] - PI / 2.0).abs() < 1e-5);
}

#[test]
fn test_acos_range() {
    use std::f32::consts::PI;
    // acos domain is [-1, 1], range is [0, π]
    let a = Vector::from_slice(&[-1.0, -0.5, 0.0, 0.5, 1.0]);
    let result = a.acos().unwrap();
    for (i, &res) in result.as_slice().iter().enumerate() {
        assert!(
            (0.0..=PI).contains(&res),
            "acos range violation at {}: {} not in [0, π]",
            i,
            res
        );
    }
}

#[test]
fn test_acos_symmetry() {
    use std::f32::consts::PI;
    // acos(-x) = π - acos(x)
    let a = Vector::from_slice(&[0.5, 0.707]);
    let result_pos = a.acos().unwrap();

    let a_neg = Vector::from_slice(&[-0.5, -0.707]);
    let result_neg = a_neg.acos().unwrap();

    for (i, (&pos, &neg)) in result_pos
        .as_slice()
        .iter()
        .zip(result_neg.as_slice().iter())
        .enumerate()
    {
        let expected_neg = PI - pos;
        assert!(
            (neg - expected_neg).abs() < 1e-5,
            "acos symmetry failed at {}: acos(-x)={} != π - acos(x)={}",
            i,
            neg,
            expected_neg
        );
    }
}

#[test]
fn test_acos_cos_inverse() {
    use std::f32::consts::PI;
    // acos(cos(x)) = x for x in [0, π]
    let a = Vector::from_slice(&[0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI]);
    let cos_result = a.cos().unwrap();
    let acos_result = cos_result.acos().unwrap();

    for (i, (&original, &reconstructed)) in a
        .as_slice()
        .iter()
        .zip(acos_result.as_slice().iter())
        .enumerate()
    {
        assert!(
            (original - reconstructed).abs() < 1e-5,
            "acos(cos(x)) != x at {}: {} != {}",
            i,
            reconstructed,
            original
        );
    }
}

#[test]
fn test_acos_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.acos().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_atan_basic() {
    use std::f32::consts::PI;
    // atan(0) = 0, atan(1) = π/4, atan(-1) = -π/4
    let a = Vector::from_slice(&[0.0, 1.0, -1.0, 1.732]); // 1.732 ≈ √3 for atan(√3) = π/3
    let result = a.atan().unwrap();
    let expected = [0.0, PI / 4.0, -PI / 4.0, PI / 3.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-3,
            "atan basic mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_atan_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.atan().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_atan_range() {
    use std::f32::consts::PI;
    // atan range is (-π/2, π/2) for all real inputs
    let a = Vector::from_slice(&[-1000.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1000.0]);
    let result = a.atan().unwrap();
    for (i, &res) in result.as_slice().iter().enumerate() {
        assert!(
            (-PI / 2.0..PI / 2.0).contains(&res),
            "atan range violation at {}: {} not in (-π/2, π/2)",
            i,
            res
        );
    }
}

#[test]
fn test_atan_negative() {
    use std::f32::consts::PI;
    // atan is odd: atan(-x) = -atan(x)
    let a = Vector::from_slice(&[-1.0, -1.732]);
    let result = a.atan().unwrap();
    let expected = [-PI / 4.0, -PI / 3.0];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-3,
            "atan negative mismatch at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_atan_tan_inverse() {
    use std::f32::consts::PI;
    // atan(tan(x)) = x for x in (-π/2, π/2)
    let a = Vector::from_slice(&[-PI / 4.0, 0.0, PI / 6.0, PI / 4.0]);
    let tan_result = a.tan().unwrap();
    let atan_result = tan_result.atan().unwrap();

    for (i, (&original, &reconstructed)) in a
        .as_slice()
        .iter()
        .zip(atan_result.as_slice().iter())
        .enumerate()
    {
        assert!(
            (original - reconstructed).abs() < 1e-5,
            "atan(tan(x)) != x at {}: {} != {}",
            i,
            reconstructed,
            original
        );
    }
}

#[test]
fn test_atan_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.atan().unwrap();
    assert_eq!(result.len(), 0);
}

// sinh() tests
#[test]
fn test_sinh_basic() {
    let a = Vector::from_slice(&[0.0, 1.0, -1.0]);
    let result = a.sinh().unwrap();
    let expected = [0.0, 1.0_f32.sinh(), (-1.0_f32).sinh()];
    for (r, e) in result.as_slice().iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
    }
}

#[test]
fn test_sinh_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.sinh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_sinh_positive() {
    let a = Vector::from_slice(&[2.0]);
    let result = a.sinh().unwrap();
    let expected = 2.0_f32.sinh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_sinh_negative() {
    let a = Vector::from_slice(&[-2.0]);
    let result = a.sinh().unwrap();
    let expected = (-2.0_f32).sinh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_sinh_odd_function() {
    // sinh(-x) = -sinh(x)
    let a = Vector::from_slice(&[1.5]);
    let b = Vector::from_slice(&[-1.5]);
    let sinh_a = a.sinh().unwrap();
    let sinh_b = b.sinh().unwrap();
    assert!(
        (sinh_a.as_slice()[0] + sinh_b.as_slice()[0]).abs() < 1e-5,
        "sinh is an odd function: sinh(-x) = -sinh(x)"
    );
}

#[test]
fn test_sinh_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.sinh().unwrap();
    assert_eq!(result.len(), 0);
}

// cosh() tests
#[test]
fn test_cosh_basic() {
    let a = Vector::from_slice(&[0.0, 1.0, -1.0]);
    let result = a.cosh().unwrap();
    let expected = [0.0_f32.cosh(), 1.0_f32.cosh(), (-1.0_f32).cosh()];
    for (r, e) in result.as_slice().iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
    }
}

#[test]
fn test_cosh_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.cosh().unwrap();
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_cosh_positive() {
    let a = Vector::from_slice(&[2.0]);
    let result = a.cosh().unwrap();
    let expected = 2.0_f32.cosh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_cosh_negative() {
    let a = Vector::from_slice(&[-2.0]);
    let result = a.cosh().unwrap();
    let expected = (-2.0_f32).cosh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_cosh_even_function() {
    // cosh(-x) = cosh(x)
    let a = Vector::from_slice(&[1.5]);
    let b = Vector::from_slice(&[-1.5]);
    let cosh_a = a.cosh().unwrap();
    let cosh_b = b.cosh().unwrap();
    assert!(
        (cosh_a.as_slice()[0] - cosh_b.as_slice()[0]).abs() < 1e-5,
        "cosh is an even function: cosh(-x) = cosh(x)"
    );
}

#[test]
fn test_cosh_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.cosh().unwrap();
    assert_eq!(result.len(), 0);
}

// tanh() tests
#[test]
fn test_tanh_basic() {
    let a = Vector::from_slice(&[0.0, 1.0, -1.0]);
    let result = a.tanh().unwrap();
    let expected = [0.0_f32.tanh(), 1.0_f32.tanh(), (-1.0_f32).tanh()];
    for (r, e) in result.as_slice().iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
    }
}

#[test]
fn test_tanh_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.tanh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_tanh_range() {
    // tanh(x) is bounded: -1 <= tanh(x) <= 1
    // For very large values, it approaches ±1 in floating-point
    let a = Vector::from_slice(&[10.0, -10.0, 100.0]);
    let result = a.tanh().unwrap();
    for &val in result.as_slice() {
        assert!(
            (-1.0..=1.0).contains(&val),
            "tanh value {} out of range [-1, 1]",
            val
        );
    }
}

#[test]
fn test_tanh_negative() {
    let a = Vector::from_slice(&[-2.0]);
    let result = a.tanh().unwrap();
    let expected = (-2.0_f32).tanh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_tanh_sinh_cosh_relation() {
    // tanh(x) = sinh(x) / cosh(x)
    let a = Vector::from_slice(&[1.5]);
    let tanh_result = a.tanh().unwrap();
    let sinh_result = a.sinh().unwrap();
    let cosh_result = a.cosh().unwrap();
    let ratio = sinh_result.as_slice()[0] / cosh_result.as_slice()[0];
    assert!(
        (tanh_result.as_slice()[0] - ratio).abs() < 1e-5,
        "tanh(x) = sinh(x)/cosh(x)"
    );
}

#[test]
fn test_tanh_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.tanh();
    assert!(result.is_err());
}

// asinh() tests
#[test]
fn test_asinh_basic() {
    let a = Vector::from_slice(&[0.0, 1.0, -1.0]);
    let result = a.asinh().unwrap();
    let expected = [0.0_f32.asinh(), 1.0_f32.asinh(), (-1.0_f32).asinh()];
    for (r, e) in result.as_slice().iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
    }
}

#[test]
fn test_asinh_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.asinh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_asinh_positive() {
    let a = Vector::from_slice(&[2.0]);
    let result = a.asinh().unwrap();
    let expected = 2.0_f32.asinh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_asinh_negative() {
    let a = Vector::from_slice(&[-2.0]);
    let result = a.asinh().unwrap();
    let expected = (-2.0_f32).asinh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_asinh_odd_function() {
    // asinh(-x) = -asinh(x)
    let a = Vector::from_slice(&[1.5]);
    let b = Vector::from_slice(&[-1.5]);
    let asinh_a = a.asinh().unwrap();
    let asinh_b = b.asinh().unwrap();
    assert!(
        (asinh_a.as_slice()[0] + asinh_b.as_slice()[0]).abs() < 1e-5,
        "asinh is an odd function: asinh(-x) = -asinh(x)"
    );
}

#[test]
fn test_asinh_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.asinh().unwrap();
    assert_eq!(result.len(), 0);
}

// acosh() tests
#[test]
fn test_acosh_basic() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.acosh().unwrap();
    let expected = [1.0_f32.acosh(), 2.0_f32.acosh(), 3.0_f32.acosh()];
    for (r, e) in result.as_slice().iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
    }
}

#[test]
fn test_acosh_one() {
    let a = Vector::from_slice(&[1.0]);
    let result = a.acosh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_acosh_positive() {
    let a = Vector::from_slice(&[2.0]);
    let result = a.acosh().unwrap();
    let expected = 2.0_f32.acosh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_acosh_large() {
    let a = Vector::from_slice(&[10.0]);
    let result = a.acosh().unwrap();
    let expected = 10.0_f32.acosh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_acosh_cosh_relation() {
    // acosh(cosh(x)) = x for x >= 0
    let a = Vector::from_slice(&[1.5]);
    let cosh_result = a.cosh().unwrap();
    let acosh_result = cosh_result.acosh().unwrap();
    assert!(
        (a.as_slice()[0] - acosh_result.as_slice()[0]).abs() < 1e-5,
        "acosh(cosh(x)) = x"
    );
}

#[test]
fn test_acosh_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.acosh().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_atanh_basic() {
    let a = Vector::from_slice(&[0.0, 0.5, -0.5]);
    let result = a.atanh().unwrap();
    let expected: Vec<f32> = vec![0.0_f32.atanh(), 0.5_f32.atanh(), (-0.5_f32).atanh()];
    for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (res - exp).abs() < 1e-5,
            "atanh failed at {}: {} != {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_atanh_zero() {
    let a = Vector::from_slice(&[0.0]);
    let result = a.atanh().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_atanh_positive() {
    let a = Vector::from_slice(&[0.5]);
    let result = a.atanh().unwrap();
    let expected = 0.5_f32.atanh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_atanh_negative() {
    let a = Vector::from_slice(&[-0.5]);
    let result = a.atanh().unwrap();
    let expected = (-0.5_f32).atanh();
    assert!((result.as_slice()[0] - expected).abs() < 1e-5);
}

#[test]
fn test_atanh_odd_function() {
    // atanh(-x) = -atanh(x)
    let a = Vector::from_slice(&[0.5]);
    let neg_a = Vector::from_slice(&[-0.5]);
    let result_a = a.atanh().unwrap();
    let result_neg_a = neg_a.atanh().unwrap();
    assert!(
        (result_a.as_slice()[0] + result_neg_a.as_slice()[0]).abs() < 1e-5,
        "atanh(-x) = -atanh(x)"
    );
}

#[test]
fn test_atanh_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.atanh().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_floor_basic() {
    let a = Vector::from_slice(&[3.7, -2.3, 5.0]);
    let result = a.floor().unwrap();
    assert_eq!(result.as_slice(), &[3.0, -3.0, 5.0]);
}

#[test]
fn test_floor_positive() {
    let a = Vector::from_slice(&[1.1, 2.9, 3.5]);
    let result = a.floor().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn test_floor_negative() {
    let a = Vector::from_slice(&[-1.1, -2.9, -3.5]);
    let result = a.floor().unwrap();
    assert_eq!(result.as_slice(), &[-2.0, -3.0, -4.0]);
}

#[test]
fn test_floor_integers() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, -4.0]);
    let result = a.floor().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, -4.0]);
}

#[test]
fn test_floor_zero() {
    let a = Vector::from_slice(&[0.0, -0.0]);
    let result = a.floor().unwrap();
    assert_eq!(result.as_slice(), &[0.0, -0.0]);
}

#[test]
fn test_floor_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.floor().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_ceil_basic() {
    let a = Vector::from_slice(&[3.2, -2.7, 5.0]);
    let result = a.ceil().unwrap();
    assert_eq!(result.as_slice(), &[4.0, -2.0, 5.0]);
}

#[test]
fn test_ceil_positive() {
    let a = Vector::from_slice(&[1.1, 2.9, 3.5]);
    let result = a.ceil().unwrap();
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0]);
}

#[test]
fn test_ceil_negative() {
    let a = Vector::from_slice(&[-1.1, -2.9, -3.5]);
    let result = a.ceil().unwrap();
    assert_eq!(result.as_slice(), &[-1.0, -2.0, -3.0]);
}

#[test]
fn test_ceil_integers() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, -4.0]);
    let result = a.ceil().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, -4.0]);
}

#[test]
fn test_ceil_zero() {
    let a = Vector::from_slice(&[0.0, -0.0]);
    let result = a.ceil().unwrap();
    assert_eq!(result.as_slice(), &[0.0, -0.0]);
}

#[test]
fn test_ceil_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.ceil().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_round_basic() {
    let a = Vector::from_slice(&[3.2, 3.7, -2.3, -2.8, 5.0]);
    let result = a.round().unwrap();
    assert_eq!(result.as_slice(), &[3.0, 4.0, -2.0, -3.0, 5.0]);
}

#[test]
fn test_round_positive() {
    let a = Vector::from_slice(&[1.4, 1.5, 1.6, 2.5]);
    let result = a.round().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 2.0, 3.0]);
}

#[test]
fn test_round_negative() {
    let a = Vector::from_slice(&[-1.4, -1.5, -1.6, -2.5]);
    let result = a.round().unwrap();
    assert_eq!(result.as_slice(), &[-1.0, -2.0, -2.0, -3.0]);
}

#[test]
fn test_round_halfway() {
    // Rust's round() uses "round half away from zero"
    let a = Vector::from_slice(&[0.5, 1.5, 2.5, 3.5, 4.5]);
    let result = a.round().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_round_zero() {
    let a = Vector::from_slice(&[0.0, -0.0, 0.3, -0.3]);
    let result = a.round().unwrap();
    assert_eq!(result.as_slice(), &[0.0, -0.0, 0.0, -0.0]);
}

#[test]
fn test_round_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.round().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_trunc_basic() {
    let a = Vector::from_slice(&[3.2, 3.7, -2.3, -2.8, 5.0]);
    let result = a.trunc().unwrap();
    assert_eq!(result.as_slice(), &[3.0, 3.0, -2.0, -2.0, 5.0]);
}

#[test]
fn test_trunc_positive() {
    let a = Vector::from_slice(&[1.1, 1.9, 2.5, 3.99]);
    let result = a.trunc().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 1.0, 2.0, 3.0]);
}

#[test]
fn test_trunc_negative() {
    let a = Vector::from_slice(&[-1.1, -1.9, -2.5, -3.99]);
    let result = a.trunc().unwrap();
    assert_eq!(result.as_slice(), &[-1.0, -1.0, -2.0, -3.0]);
}

#[test]
fn test_trunc_toward_zero() {
    // Verify trunc() always moves toward zero
    let a = Vector::from_slice(&[2.7, -2.7, 5.3, -5.3]);
    let result = a.trunc().unwrap();
    assert_eq!(result.as_slice(), &[2.0, -2.0, 5.0, -5.0]);
}

#[test]
fn test_trunc_zero() {
    let a = Vector::from_slice(&[0.0, -0.0, 0.9, -0.9]);
    let result = a.trunc().unwrap();
    assert_eq!(result.as_slice(), &[0.0, -0.0, 0.0, -0.0]);
}

#[test]
fn test_trunc_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.trunc().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_fract_basic() {
    let a = Vector::from_slice(&[3.7, -2.3, 5.0]);
    let result = a.fract().unwrap();
    // fract returns fractional part with same sign
    assert!((result.as_slice()[0] - 0.7).abs() < 1e-5);
    assert!((result.as_slice()[1] - (-0.3)).abs() < 1e-5);
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
}

#[test]
fn test_fract_positive() {
    let a = Vector::from_slice(&[1.2, 2.5, 3.9]);
    let result = a.fract().unwrap();
    assert!((result.as_slice()[0] - 0.2).abs() < 1e-5);
    assert!((result.as_slice()[1] - 0.5).abs() < 1e-5);
    assert!((result.as_slice()[2] - 0.9).abs() < 1e-5);
}

#[test]
fn test_fract_negative() {
    let a = Vector::from_slice(&[-1.2, -2.5, -3.9]);
    let result = a.fract().unwrap();
    assert!((result.as_slice()[0] - (-0.2)).abs() < 1e-5);
    assert!((result.as_slice()[1] - (-0.5)).abs() < 1e-5);
    assert!((result.as_slice()[2] - (-0.9)).abs() < 1e-5);
}

#[test]
fn test_fract_integers() {
    let a = Vector::from_slice(&[1.0, 2.0, -3.0, 0.0]);
    let result = a.fract().unwrap();
    assert_eq!(result.as_slice(), &[0.0, 0.0, -0.0, 0.0]);
}

#[test]
fn test_fract_range() {
    // fract() is always in range [0, 1) for positive, (-1, 0] for negative
    let a = Vector::from_slice(&[0.1, 0.5, 0.9, -0.1, -0.5, -0.9]);
    let result = a.fract().unwrap();
    for &val in result.as_slice() {
        assert!(
            val.abs() < 1.0,
            "fract value should be in range (-1, 1): {}",
            val
        );
    }
}

#[test]
fn test_fract_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.fract().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_signum_basic() {
    let a = Vector::from_slice(&[5.0, -3.0, 0.0, -0.0]);
    let result = a.signum().unwrap();
    assert_eq!(result.as_slice(), &[1.0, -1.0, 1.0, -1.0]);
}

#[test]
fn test_signum_positive() {
    let a = Vector::from_slice(&[0.1, 1.0, 100.0, f32::INFINITY]);
    let result = a.signum().unwrap();
    assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_signum_negative() {
    let a = Vector::from_slice(&[-0.1, -1.0, -100.0, f32::NEG_INFINITY]);
    let result = a.signum().unwrap();
    assert_eq!(result.as_slice(), &[-1.0, -1.0, -1.0, -1.0]);
}

#[test]
fn test_signum_mixed() {
    let a = Vector::from_slice(&[42.5, -17.3, 0.0001, -0.0001]);
    let result = a.signum().unwrap();
    assert_eq!(result.as_slice(), &[1.0, -1.0, 1.0, -1.0]);
}

#[test]
fn test_signum_zero_handling() {
    // Rust's signum treats +0.0 as positive (1.0) and -0.0 as negative (-1.0)
    let a = Vector::from_slice(&[0.0, -0.0]);
    let result = a.signum().unwrap();
    assert_eq!(result.as_slice(), &[1.0, -1.0]);
}

#[test]
fn test_signum_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.signum().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_copysign_basic() {
    let magnitude = Vector::from_slice(&[5.0, 3.0, 2.0, 4.0]);
    let sign = Vector::from_slice(&[-1.0, 1.0, -1.0, 1.0]);
    let result = magnitude.copysign(&sign).unwrap();
    assert_eq!(result.as_slice(), &[-5.0, 3.0, -2.0, 4.0]);
}

#[test]
fn test_copysign_negative_magnitude() {
    // copysign takes absolute magnitude, so negative magnitude becomes positive first
    let magnitude = Vector::from_slice(&[-5.0, -3.0]);
    let sign = Vector::from_slice(&[1.0, -1.0]);
    let result = magnitude.copysign(&sign).unwrap();
    assert_eq!(result.as_slice(), &[5.0, -3.0]);
}

#[test]
fn test_copysign_zero() {
    // copysign handles +0.0 and -0.0
    let magnitude = Vector::from_slice(&[3.0, 3.0]);
    let sign = Vector::from_slice(&[0.0, -0.0]);
    let result = magnitude.copysign(&sign).unwrap();
    assert_eq!(result.as_slice(), &[3.0, -3.0]);
}

#[test]
fn test_copysign_infinity() {
    let magnitude = Vector::from_slice(&[5.0, 5.0]);
    let sign = Vector::from_slice(&[f32::INFINITY, f32::NEG_INFINITY]);
    let result = magnitude.copysign(&sign).unwrap();
    assert_eq!(result.as_slice(), &[5.0, -5.0]);
}

#[test]
fn test_copysign_size_mismatch() {
    let magnitude = Vector::from_slice(&[1.0, 2.0]);
    let sign = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = magnitude.copysign(&sign);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TruenoError::SizeMismatch { .. }
    ));
}

#[test]
fn test_copysign_empty() {
    let magnitude: Vector<f32> = Vector::from_slice(&[]);
    let sign: Vector<f32> = Vector::from_slice(&[]);
    let result = magnitude.copysign(&sign).unwrap();
    assert_eq!(result.len(), 0);
}

// ========================================
// Unit Tests: minimum()
// ========================================

#[test]
fn test_minimum_basic() {
    let a = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    let b = Vector::from_slice(&[2.0, 3.0, 4.0, 1.0]);
    let result = a.minimum(&b).unwrap();
    assert_eq!(result.as_slice(), &[1.0, 3.0, 3.0, 1.0]);
}

#[test]
fn test_minimum_negative() {
    let a = Vector::from_slice(&[-1.0, -5.0, 3.0]);
    let b = Vector::from_slice(&[-2.0, -3.0, 4.0]);
    let result = a.minimum(&b).unwrap();
    assert_eq!(result.as_slice(), &[-2.0, -5.0, 3.0]);
}

#[test]
fn test_minimum_nan() {
    // NaN handling: NAN.min(x) = x (prefers non-NaN)
    let a = Vector::from_slice(&[f32::NAN, 5.0, f32::NAN]);
    let b = Vector::from_slice(&[3.0, f32::NAN, f32::NAN]);
    let result = a.minimum(&b).unwrap();
    assert_eq!(result.as_slice()[0], 3.0);
    assert_eq!(result.as_slice()[1], 5.0);
    assert!(result.as_slice()[2].is_nan());
}

#[test]
fn test_minimum_infinity() {
    let a = Vector::from_slice(&[f32::INFINITY, 5.0, f32::NEG_INFINITY]);
    let b = Vector::from_slice(&[3.0, f32::INFINITY, -10.0]);
    let result = a.minimum(&b).unwrap();
    assert_eq!(result.as_slice(), &[3.0, 5.0, f32::NEG_INFINITY]);
}

#[test]
fn test_minimum_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.minimum(&b);
    assert!(matches!(result, Err(TruenoError::SizeMismatch { .. })));
}

#[test]
fn test_minimum_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.minimum(&b).unwrap();
    assert_eq!(result.len(), 0);
}

// ========================================
// Unit Tests: maximum()
// ========================================

#[test]
fn test_maximum_basic() {
    let a = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    let b = Vector::from_slice(&[2.0, 3.0, 4.0, 1.0]);
    let result = a.maximum(&b).unwrap();
    assert_eq!(result.as_slice(), &[2.0, 5.0, 4.0, 2.0]);
}

#[test]
fn test_maximum_negative() {
    let a = Vector::from_slice(&[-1.0, -5.0, 3.0]);
    let b = Vector::from_slice(&[-2.0, -3.0, 4.0]);
    let result = a.maximum(&b).unwrap();
    assert_eq!(result.as_slice(), &[-1.0, -3.0, 4.0]);
}

#[test]
fn test_maximum_nan() {
    // NaN handling: NAN.max(x) = x (prefers non-NaN)
    let a = Vector::from_slice(&[f32::NAN, 5.0, f32::NAN]);
    let b = Vector::from_slice(&[3.0, f32::NAN, f32::NAN]);
    let result = a.maximum(&b).unwrap();
    assert_eq!(result.as_slice()[0], 3.0);
    assert_eq!(result.as_slice()[1], 5.0);
    assert!(result.as_slice()[2].is_nan());
}

#[test]
fn test_maximum_infinity() {
    let a = Vector::from_slice(&[f32::INFINITY, 5.0, f32::NEG_INFINITY]);
    let b = Vector::from_slice(&[3.0, f32::INFINITY, -10.0]);
    let result = a.maximum(&b).unwrap();
    assert_eq!(result.as_slice(), &[f32::INFINITY, f32::INFINITY, -10.0]);
}

#[test]
fn test_maximum_size_mismatch() {
    let a = Vector::from_slice(&[1.0, 2.0]);
    let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.maximum(&b);
    assert!(matches!(result, Err(TruenoError::SizeMismatch { .. })));
}

#[test]
fn test_maximum_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.maximum(&b).unwrap();
    assert_eq!(result.len(), 0);
}

// ========================================
// Unit Tests: neg()
// ========================================

#[test]
fn test_neg_basic() {
    let a = Vector::from_slice(&[1.0, -2.0, 3.0, -4.0]);
    let result = a.neg().unwrap();
    assert_eq!(result.as_slice(), &[-1.0, 2.0, -3.0, 4.0]);
}

#[test]
fn test_neg_zero() {
    let a = Vector::from_slice(&[0.0, -0.0]);
    let result = a.neg().unwrap();
    // -0.0 becomes 0.0, 0.0 becomes -0.0
    assert_eq!(result.as_slice()[0], -0.0);
    assert_eq!(result.as_slice()[1], 0.0);
}

#[test]
fn test_neg_double_negation() {
    // Property: -(-x) = x (double negation is identity)
    let a = Vector::from_slice(&[1.0, -2.0, 3.0, -4.0, 5.0]);
    let neg_once = a.neg().unwrap();
    let neg_twice = neg_once.neg().unwrap();
    for (i, (&original, &double_neg)) in a
        .as_slice()
        .iter()
        .zip(neg_twice.as_slice().iter())
        .enumerate()
    {
        assert!(
            (original - double_neg).abs() < 1e-6,
            "Double negation failed at {}: -(-{}) = {} != {}",
            i,
            original,
            double_neg,
            original
        );
    }
}

#[test]
fn test_neg_nan() {
    let a = Vector::from_slice(&[f32::NAN, 5.0]);
    let result = a.neg().unwrap();
    assert!(result.as_slice()[0].is_nan());
    assert_eq!(result.as_slice()[1], -5.0);
}

#[test]
fn test_neg_infinity() {
    let a = Vector::from_slice(&[f32::INFINITY, f32::NEG_INFINITY]);
    let result = a.neg().unwrap();
    assert_eq!(result.as_slice(), &[f32::NEG_INFINITY, f32::INFINITY]);
}

#[test]
fn test_neg_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.neg().unwrap();
    assert_eq!(result.len(), 0);
}

// ========================================
// Unit Tests: sum_of_squares()
// ========================================

#[test]
fn test_sum_of_squares_basic() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = a.sum_of_squares().unwrap();
    assert_eq!(result, 14.0); // 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
}

#[test]
fn test_sum_of_squares_negative() {
    let a = Vector::from_slice(&[-1.0, -2.0, 3.0]);
    let result = a.sum_of_squares().unwrap();
    assert_eq!(result, 14.0); // (-1)^2 + (-2)^2 + 3^2 = 1 + 4 + 9 = 14
}

#[test]
fn test_sum_of_squares_single() {
    let a = Vector::from_slice(&[5.0]);
    let result = a.sum_of_squares().unwrap();
    assert_eq!(result, 25.0);
}

#[test]
fn test_sum_of_squares_zero() {
    let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = a.sum_of_squares().unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn test_sum_of_squares_pythagorean() {
    // 3-4-5 Pythagorean triple
    let a = Vector::from_slice(&[3.0, 4.0]);
    let result = a.sum_of_squares().unwrap();
    assert_eq!(result, 25.0); // 3^2 + 4^2 = 9 + 16 = 25
}

#[test]
fn test_sum_of_squares_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.sum_of_squares().unwrap();
    assert_eq!(result, 0.0);
}

// ========================================================================
// Tests for mean() - arithmetic average
// ========================================================================

#[test]
fn test_mean_basic() {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let result = a.mean().unwrap();
    assert!((result - 2.5).abs() < 1e-5); // (1+2+3+4)/4 = 2.5
}

#[test]
fn test_mean_negative() {
    let a = Vector::from_slice(&[-2.0, -4.0, -6.0]);
    let result = a.mean().unwrap();
    assert!((result - (-4.0)).abs() < 1e-5); // (-2-4-6)/3 = -4.0
}

#[test]
fn test_mean_mixed() {
    let a = Vector::from_slice(&[-10.0, 0.0, 10.0]);
    let result = a.mean().unwrap();
    assert!(result.abs() < 1e-5); // (-10+0+10)/3 = 0.0
}

#[test]
fn test_mean_single() {
    let a = Vector::from_slice(&[42.0]);
    let result = a.mean().unwrap();
    assert!((result - 42.0).abs() < 1e-5); // 42/1 = 42
}

#[test]
fn test_mean_all_same() {
    let a = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0, 5.0]);
    let result = a.mean().unwrap();
    assert!((result - 5.0).abs() < 1e-5); // (5+5+5+5+5)/5 = 5
}

#[test]
fn test_mean_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.mean();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// ========================================================================
// Tests for variance() - population variance
// ========================================================================

#[test]
fn test_variance_basic() {
    // Variance of [1,2,3,4,5]: mean=3, var=E[X²]-μ²=11-9=2
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = a.variance().unwrap();
    assert!((result - 2.0).abs() < 1e-5);
}

#[test]
fn test_variance_constant() {
    // Variance of constant vector is 0
    let a = Vector::from_slice(&[7.0, 7.0, 7.0, 7.0]);
    let result = a.variance().unwrap();
    assert!(result.abs() < 1e-5);
}

#[test]
fn test_variance_single() {
    // Variance of single element is 0
    let a = Vector::from_slice(&[42.0]);
    let result = a.variance().unwrap();
    assert!(result.abs() < 1e-5);
}

#[test]
fn test_variance_symmetric() {
    // Variance of [-2, -1, 0, 1, 2]: mean=0, var=E[X²]=2
    let a = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = a.variance().unwrap();
    assert!((result - 2.0).abs() < 1e-5);
}

#[test]
fn test_variance_two_values() {
    // Variance of [1, 5]: mean=3, var=(1-3)²+(5-3)²/2=8/2=4
    let a = Vector::from_slice(&[1.0, 5.0]);
    let result = a.variance().unwrap();
    assert!((result - 4.0).abs() < 1e-5);
}

#[test]
fn test_variance_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.variance();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// ========================================================================
// Tests for stddev() - standard deviation
// ========================================================================

#[test]
fn test_stddev_basic() {
    // stddev of [1,2,3,4,5]: variance=2, stddev=sqrt(2)≈1.414
    let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = a.stddev().unwrap();
    assert!((result - std::f32::consts::SQRT_2).abs() < 1e-5);
}

#[test]
fn test_stddev_constant() {
    // stddev of constant vector is 0
    let a = Vector::from_slice(&[7.0, 7.0, 7.0, 7.0]);
    let result = a.stddev().unwrap();
    assert!(result.abs() < 1e-5);
}

#[test]
fn test_stddev_single() {
    // stddev of single element is 0
    let a = Vector::from_slice(&[42.0]);
    let result = a.stddev().unwrap();
    assert!(result.abs() < 1e-5);
}

#[test]
fn test_stddev_symmetric() {
    // stddev of [-2,-1,0,1,2]: variance=2, stddev=sqrt(2)
    let a = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = a.stddev().unwrap();
    assert!((result - std::f32::consts::SQRT_2).abs() < 1e-5);
}

#[test]
fn test_stddev_two_values() {
    // stddev of [1,5]: variance=4, stddev=2
    let a = Vector::from_slice(&[1.0, 5.0]);
    let result = a.stddev().unwrap();
    assert!((result - 2.0).abs() < 1e-5);
}

#[test]
fn test_stddev_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let result = a.stddev();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// ========================================================================
// Tests for covariance() - population covariance
// ========================================================================

#[test]
fn test_covariance_positive() {
    // Perfect positive linear relationship: y = 2x
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]);
    let result = x.covariance(&y).unwrap();
    // Cov(X,2X) = 2*Var(X) = 2*(2/3) = 4/3 ≈ 1.333
    assert!((result - (4.0 / 3.0)).abs() < 1e-5);
}

#[test]
fn test_covariance_negative() {
    // Negative linear relationship
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y = Vector::from_slice(&[3.0, 2.0, 1.0]);
    let result = x.covariance(&y).unwrap();
    assert!((result - (-2.0 / 3.0)).abs() < 1e-5);
}

#[test]
fn test_covariance_zero() {
    // No linear relationship
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 2.0]);
    let y = Vector::from_slice(&[1.0, 3.0, 1.0, 3.0]);
    let result = x.covariance(&y).unwrap();
    assert!(result.abs() < 1e-5);
}

#[test]
fn test_covariance_self() {
    // Cov(X,X) = Var(X)
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let cov = x.covariance(&x).unwrap();
    let var = x.variance().unwrap();
    assert!((cov - var).abs() < 1e-5);
}

#[test]
fn test_covariance_size_mismatch() {
    let x = Vector::from_slice(&[1.0, 2.0]);
    let y = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = x.covariance(&y);
    assert!(matches!(
        result,
        Err(TruenoError::SizeMismatch {
            expected: 2,
            actual: 3
        })
    ));
}

#[test]
fn test_covariance_empty() {
    let x: Vector<f32> = Vector::from_slice(&[]);
    let y: Vector<f32> = Vector::from_slice(&[]);
    let result = x.covariance(&y);
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// ========================================================================
// Tests for correlation() - Pearson correlation coefficient
// ========================================================================

#[test]
fn test_correlation_perfect_positive() {
    // Perfect positive linear relationship: y = 2x
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y = Vector::from_slice(&[2.0, 4.0, 6.0]);
    let result = x.correlation(&y).unwrap();
    assert!((result - 1.0).abs() < 1e-5);
}

#[test]
fn test_correlation_perfect_negative() {
    // Perfect negative linear relationship
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let y = Vector::from_slice(&[4.0, 3.0, 2.0, 1.0]);
    let result = x.correlation(&y).unwrap();
    assert!((result - (-1.0)).abs() < 1e-5);
}

#[test]
fn test_correlation_zero() {
    // No correlation
    let x = Vector::from_slice(&[1.0, 2.0, 1.0, 2.0]);
    let y = Vector::from_slice(&[1.0, 1.0, 2.0, 2.0]);
    let result = x.correlation(&y).unwrap();
    assert!(result.abs() < 1e-5);
}

#[test]
fn test_correlation_self() {
    // Correlation with self is always 1
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = x.correlation(&x).unwrap();
    assert!((result - 1.0).abs() < 1e-5);
}

#[test]
fn test_correlation_constant_vector() {
    // Constant vector has zero std dev → division by zero
    let x = Vector::from_slice(&[5.0, 5.0, 5.0]);
    let y = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = x.correlation(&y);
    assert!(matches!(result, Err(TruenoError::DivisionByZero)));
}

#[test]
fn test_correlation_size_mismatch() {
    let x = Vector::from_slice(&[1.0, 2.0]);
    let y = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = x.correlation(&y);
    assert!(matches!(
        result,
        Err(TruenoError::SizeMismatch {
            expected: 2,
            actual: 3
        })
    ));
}

// ========================================================================
// Tests for zscore() - Z-score normalization (standardization)
// ========================================================================

#[test]
fn test_zscore_basic() {
    // [1, 2, 3, 4, 5] has mean=3, stddev=sqrt(2)
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let z = v.zscore().unwrap();

    // Verify mean ≈ 0
    let mean = z.mean().unwrap();
    assert!(mean.abs() < 1e-5, "mean = {}, expected ≈ 0", mean);

    // Verify stddev ≈ 1
    let std = z.stddev().unwrap();
    assert!((std - 1.0).abs() < 1e-5, "stddev = {}, expected ≈ 1", std);
}

#[test]
fn test_zscore_negative_values() {
    // [-2, -1, 0, 1, 2] has mean=0, stddev=sqrt(2)
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let z = v.zscore().unwrap();

    let mean = z.mean().unwrap();
    assert!(mean.abs() < 1e-5);

    let std = z.stddev().unwrap();
    assert!((std - 1.0).abs() < 1e-5);
}

#[test]
fn test_zscore_single_element() {
    // Single element has zero stddev → DivisionByZero
    let v = Vector::from_slice(&[5.0]);
    let result = v.zscore();
    assert!(matches!(result, Err(TruenoError::DivisionByZero)));
}

#[test]
fn test_zscore_constant_vector() {
    // All identical elements have zero stddev → DivisionByZero
    let v = Vector::from_slice(&[3.0, 3.0, 3.0, 3.0]);
    let result = v.zscore();
    assert!(matches!(result, Err(TruenoError::DivisionByZero)));
}

#[test]
fn test_zscore_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.zscore();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_zscore_already_normalized() {
    // Vector already with mean≈0, std≈1 should stay similar
    let v = Vector::from_slice(&[-1.0, 0.0, 1.0]);
    let z = v.zscore().unwrap();

    // Should be close to the original (scaling might differ slightly)
    let mean = z.mean().unwrap();
    assert!(mean.abs() < 1e-5);

    let std = z.stddev().unwrap();
    assert!((std - 1.0).abs() < 1e-5);
}

// ========================================================================
// Tests for minmax_normalize() - Min-max normalization to [0, 1]
// ========================================================================

#[test]
fn test_minmax_normalize_basic() {
    // [1, 2, 3, 4, 5] → [0, 0.25, 0.5, 0.75, 1.0]
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let normalized = v.minmax_normalize().unwrap();

    // Verify min = 0
    let min = normalized.min().unwrap();
    assert!((min - 0.0).abs() < 1e-5, "min = {}, expected 0", min);

    // Verify max = 1
    let max = normalized.max().unwrap();
    assert!((max - 1.0).abs() < 1e-5, "max = {}, expected 1", max);

    // Verify specific values
    assert!((normalized.data[0] - 0.0).abs() < 1e-5);
    assert!((normalized.data[2] - 0.5).abs() < 1e-5);
    assert!((normalized.data[4] - 1.0).abs() < 1e-5);
}

#[test]
fn test_minmax_normalize_negative_values() {
    // [-10, -5, 0, 5, 10] → [0, 0.25, 0.5, 0.75, 1.0]
    let v = Vector::from_slice(&[-10.0, -5.0, 0.0, 5.0, 10.0]);
    let normalized = v.minmax_normalize().unwrap();

    let min = normalized.min().unwrap();
    assert!((min - 0.0).abs() < 1e-5);

    let max = normalized.max().unwrap();
    assert!((max - 1.0).abs() < 1e-5);

    // Middle value should be 0.5
    assert!((normalized.data[2] - 0.5).abs() < 1e-5);
}

#[test]
fn test_minmax_normalize_single_element() {
    // Single element has zero range → DivisionByZero
    let v = Vector::from_slice(&[5.0]);
    let result = v.minmax_normalize();
    assert!(matches!(result, Err(TruenoError::DivisionByZero)));
}

#[test]
fn test_minmax_normalize_constant_vector() {
    // All identical elements have zero range → DivisionByZero
    let v = Vector::from_slice(&[3.0, 3.0, 3.0, 3.0]);
    let result = v.minmax_normalize();
    assert!(matches!(result, Err(TruenoError::DivisionByZero)));
}

#[test]
fn test_minmax_normalize_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.minmax_normalize();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_minmax_normalize_already_normalized() {
    // Vector already in [0, 1] should stay in [0, 1]
    let v = Vector::from_slice(&[0.0, 0.25, 0.5, 0.75, 1.0]);
    let normalized = v.minmax_normalize().unwrap();

    let min = normalized.min().unwrap();
    assert!((min - 0.0).abs() < 1e-5);

    let max = normalized.max().unwrap();
    assert!((max - 1.0).abs() < 1e-5);
}

// ========================================================================
// Tests for layer_norm() - Layer normalization (Issue #61)
// ========================================================================

#[test]
fn test_layer_norm_basic() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let gamma = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]);
    let beta = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);

    let y = x.layer_norm(&gamma, &beta, 1e-5).unwrap();

    // Output should have mean ≈ 0
    let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
    assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);

    // Output should have variance ≈ 1
    let var: f32 = y
        .as_slice()
        .iter()
        .map(|&v| (v - mean).powi(2))
        .sum::<f32>()
        / y.len() as f32;
    assert!(
        (var - 1.0).abs() < 1e-3,
        "Variance should be ~1, got {}",
        var
    );
}

#[test]
fn test_layer_norm_with_scale_shift() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let gamma = Vector::from_slice(&[2.0, 2.0, 2.0, 2.0]); // Scale by 2
    let beta = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]); // Shift by 1

    let y = x.layer_norm(&gamma, &beta, 1e-5).unwrap();

    // Output should have mean ≈ 1 (beta)
    let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
    assert!((mean - 1.0).abs() < 1e-3, "Mean should be ~1, got {}", mean);

    // Output should have std ≈ 2 (gamma)
    let var: f32 = y
        .as_slice()
        .iter()
        .map(|&v| (v - mean).powi(2))
        .sum::<f32>()
        / y.len() as f32;
    let std = var.sqrt();
    assert!((std - 2.0).abs() < 1e-3, "Std should be ~2, got {}", std);
}

#[test]
fn test_layer_norm_empty_vector() {
    let x: Vector<f32> = Vector::from_slice(&[]);
    let gamma = Vector::from_slice(&[]);
    let beta = Vector::from_slice(&[]);

    let result = x.layer_norm(&gamma, &beta, 1e-5);
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_layer_norm_size_mismatch_gamma() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let gamma = Vector::from_slice(&[1.0, 1.0]); // Wrong size
    let beta = Vector::from_slice(&[0.0, 0.0, 0.0]);

    let result = x.layer_norm(&gamma, &beta, 1e-5);
    assert!(matches!(result, Err(TruenoError::SizeMismatch { .. })));
}

#[test]
fn test_layer_norm_size_mismatch_beta() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let gamma = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let beta = Vector::from_slice(&[0.0, 0.0]); // Wrong size

    let result = x.layer_norm(&gamma, &beta, 1e-5);
    assert!(matches!(result, Err(TruenoError::SizeMismatch { .. })));
}

#[test]
fn test_layer_norm_simple() {
    let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let y = x.layer_norm_simple(1e-5).unwrap();

    // Output should have mean ≈ 0
    let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
    assert!(mean.abs() < 1e-5, "Mean should be ~0, got {}", mean);
}

#[test]
fn test_layer_norm_constant_input() {
    // Constant input [5, 5, 5, 5] should produce zeros (with gamma=1, beta=0)
    let x = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
    let gamma = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]);
    let beta = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);

    let y = x.layer_norm(&gamma, &beta, 1e-5).unwrap();

    // All outputs should be 0 (or very close due to eps)
    for &v in y.as_slice() {
        assert!(v.abs() < 1e-3, "Expected ~0, got {}", v);
    }
}

#[test]
fn test_layer_norm_single_element() {
    // Single element should normalize to 0 (x - mean = 0)
    let x = Vector::from_slice(&[42.0]);
    let gamma = Vector::from_slice(&[1.0]);
    let beta = Vector::from_slice(&[0.0]);

    let y = x.layer_norm(&gamma, &beta, 1e-5).unwrap();

    assert!(y.as_slice()[0].abs() < 1e-3);
}

#[test]
fn test_layer_norm_negative_values() {
    let x = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let gamma = Vector::from_slice(&[1.0; 5]);
    let beta = Vector::from_slice(&[0.0; 5]);

    let y = x.layer_norm(&gamma, &beta, 1e-5).unwrap();

    // Should still produce mean ≈ 0
    let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
    assert!(mean.abs() < 1e-5);
}

// ========================================================================
// Tests for clip() - Constrain values to [min, max] range
// ========================================================================

#[test]
fn test_clip_basic() {
    // [-5, 0, 5, 10, 15] clipped to [0, 10] → [0, 0, 5, 10, 10]
    let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
    let clipped = v.clip(0.0, 10.0).unwrap();

    assert_eq!(clipped.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
}

#[test]
fn test_clip_no_change() {
    // All values within range should stay unchanged
    let v = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);
    let clipped = v.clip(0.0, 10.0).unwrap();

    assert_eq!(clipped.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_clip_all_below() {
    // All values below min → all become min
    let v = Vector::from_slice(&[-10.0, -5.0, -2.0]);
    let clipped = v.clip(0.0, 10.0).unwrap();

    assert_eq!(clipped.as_slice(), &[0.0, 0.0, 0.0]);
}

#[test]
fn test_clip_all_above() {
    // All values above max → all become max
    let v = Vector::from_slice(&[15.0, 20.0, 25.0]);
    let clipped = v.clip(0.0, 10.0).unwrap();

    assert_eq!(clipped.as_slice(), &[10.0, 10.0, 10.0]);
}

#[test]
fn test_clip_invalid_range() {
    // min > max → InvalidInput error
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.clip(10.0, 5.0);

    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_clip_equal_bounds() {
    // min == max → all values become that value
    let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0]);
    let clipped = v.clip(7.0, 7.0).unwrap();

    assert_eq!(clipped.as_slice(), &[7.0, 7.0, 7.0, 7.0]);
}

// ========================================================================
// Tests for softmax() - Softmax activation (probability distribution)
// ========================================================================

#[test]
fn test_softmax_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let probs = v.softmax().unwrap();

    // Verify sum ≈ 1
    let sum: f32 = probs.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "sum = {}, expected 1", sum);

    // Verify all values in [0, 1]
    for &p in probs.as_slice() {
        assert!((0.0..=1.0).contains(&p), "prob = {} not in [0, 1]", p);
    }

    // Largest input should have largest probability
    assert!(probs.data[2] > probs.data[1]);
    assert!(probs.data[1] > probs.data[0]);
}

#[test]
fn test_softmax_uniform() {
    // All equal inputs → uniform distribution
    let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
    let probs = v.softmax().unwrap();

    // Each should be 1/4 = 0.25
    for &p in probs.as_slice() {
        assert!((p - 0.25).abs() < 1e-5, "prob = {}, expected 0.25", p);
    }
}

#[test]
fn test_softmax_large_values() {
    // Test numerical stability with large values
    let v = Vector::from_slice(&[100.0, 101.0, 102.0]);
    let probs = v.softmax().unwrap();

    // Should still sum to 1
    let sum: f32 = probs.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // Largest value should have largest probability
    assert!(probs.data[2] > probs.data[1]);
    assert!(probs.data[1] > probs.data[0]);
}

#[test]
fn test_softmax_negative_values() {
    let v = Vector::from_slice(&[-3.0, -2.0, -1.0]);
    let probs = v.softmax().unwrap();

    // Verify sum ≈ 1
    let sum: f32 = probs.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // All values should be positive
    for &p in probs.as_slice() {
        assert!(p > 0.0);
    }
}

#[test]
fn test_softmax_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.softmax();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_softmax_single_element() {
    // Single element → probability 1.0
    let v = Vector::from_slice(&[5.0]);
    let probs = v.softmax().unwrap();

    assert!((probs.data[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_log_softmax_basic() {
    // Verify exp(log_softmax(x)) == softmax(x)
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let log_probs = v.log_softmax().unwrap();
    let probs = v.softmax().unwrap();

    for i in 0..v.len() {
        let exp_log_prob = log_probs.data[i].exp();
        assert!(
            (exp_log_prob - probs.data[i]).abs() < 1e-5,
            "exp(log_softmax)[{}] = {}, softmax[{}] = {}",
            i,
            exp_log_prob,
            i,
            probs.data[i]
        );
    }
}

#[test]
fn test_log_softmax_uniform() {
    // All equal inputs → uniform log probabilities
    let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
    let log_probs = v.log_softmax().unwrap();

    // Each should be log(1/4) = log(0.25) ≈ -1.386
    let expected = (0.25_f32).ln();
    for &lp in log_probs.as_slice() {
        assert!(
            (lp - expected).abs() < 1e-5,
            "log_prob = {}, expected {}",
            lp,
            expected
        );
    }
}

#[test]
fn test_log_softmax_large_values() {
    // Test numerical stability with large values
    let v = Vector::from_slice(&[100.0, 101.0, 102.0]);
    let log_probs = v.log_softmax().unwrap();

    // exp(log_probs) should sum to 1
    let sum: f32 = log_probs.as_slice().iter().map(|&lp| lp.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // All log probabilities should be <= 0 (since probabilities <= 1)
    for &lp in log_probs.as_slice() {
        assert!(lp <= 1e-5, "log_prob = {} should be <= 0", lp);
    }

    // Largest input should have largest log probability (least negative)
    assert!(log_probs.data[2] > log_probs.data[1]);
    assert!(log_probs.data[1] > log_probs.data[0]);
}

#[test]
fn test_log_softmax_negative_values() {
    // Negative values should work fine
    let v = Vector::from_slice(&[-1.0, -2.0, -3.0]);
    let log_probs = v.log_softmax().unwrap();

    // exp(log_probs) should sum to 1
    let sum: f32 = log_probs.as_slice().iter().map(|&lp| lp.exp()).sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // All log probabilities should be <= 0
    for &lp in log_probs.as_slice() {
        assert!(lp <= 1e-5);
    }
}

#[test]
fn test_log_softmax_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.log_softmax();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_log_softmax_single_element() {
    // Single element → log probability = log(1.0) = 0.0
    let v = Vector::from_slice(&[5.0]);
    let log_probs = v.log_softmax().unwrap();

    assert!(
        log_probs.data[0].abs() < 1e-5,
        "log_softmax of single element should be 0.0, got {}",
        log_probs.data[0]
    );
}

#[test]
fn test_relu_basic() {
    // Basic ReLU: negative values → 0, positive values unchanged
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.relu().unwrap();

    assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_relu_all_negative() {
    // All negative values should become zero
    let v = Vector::from_slice(&[-5.0, -3.0, -1.0, -0.5]);
    let result = v.relu().unwrap();

    for &val in result.as_slice() {
        assert_eq!(val, 0.0, "All negative values should become 0");
    }
}

#[test]
fn test_relu_all_positive() {
    // All positive values should remain unchanged
    let v = Vector::from_slice(&[0.5, 1.0, 3.0, 5.0]);
    let expected = v.clone();
    let result = v.relu().unwrap();

    for i in 0..v.len() {
        assert_eq!(
            result.data[i], expected.data[i],
            "Positive values should remain unchanged"
        );
    }
}

#[test]
fn test_relu_zero_boundary() {
    // Zero should remain zero (boundary case)
    let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = v.relu().unwrap();

    for &val in result.as_slice() {
        assert_eq!(val, 0.0, "Zero should remain zero");
    }
}

#[test]
fn test_relu_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.relu();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_relu_sparsity() {
    // ReLU creates sparse activations (zeros for negative inputs)
    let v = Vector::from_slice(&[-10.0, 5.0, -3.0, 8.0, -1.0, 2.0]);
    let result = v.relu().unwrap();

    // Count zeros (should be 3)
    let zero_count = result.as_slice().iter().filter(|&&x| x == 0.0).count();
    assert_eq!(zero_count, 3, "ReLU should produce sparse activations");

    // Verify positive values preserved
    assert_eq!(result.data[1], 5.0);
    assert_eq!(result.data[3], 8.0);
    assert_eq!(result.data[5], 2.0);
}

#[test]
fn test_sigmoid_basic() {
    // Basic sigmoid: negative → (0, 0.5), zero → 0.5, positive → (0.5, 1)
    let v = Vector::from_slice(&[-2.0, 0.0, 2.0]);
    let result = v.sigmoid().unwrap();

    // sigmoid(-2) ≈ 0.1192, sigmoid(0) = 0.5, sigmoid(2) ≈ 0.8808
    assert!((result.data[0] - 0.1192).abs() < 0.001);
    assert!((result.data[1] - 0.5).abs() < 0.001);
    assert!((result.data[2] - 0.8808).abs() < 0.001);
}

#[test]
fn test_sigmoid_range() {
    // All outputs should be in [0, 1] range (inclusive for numerical stability)
    let v = Vector::from_slice(&[-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]);
    let result = v.sigmoid().unwrap();

    for &val in result.as_slice() {
        assert!(
            (0.0..=1.0).contains(&val),
            "Sigmoid output {} not in [0, 1]",
            val
        );
    }
}

#[test]
fn test_sigmoid_symmetry() {
    // Test σ(-x) = 1 - σ(x)
    let v = Vector::from_slice(&[-3.0, -1.5, -0.5]);
    let v_neg = Vector::from_slice(&[3.0, 1.5, 0.5]);

    let sig = v.sigmoid().unwrap();
    let sig_neg = v_neg.sigmoid().unwrap();

    for i in 0..v.len() {
        let sum = sig.data[i] + sig_neg.data[i];
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Symmetry violated: σ({}) + σ({}) = {} + {} = {} ≠ 1",
            v.data[i],
            v_neg.data[i],
            sig.data[i],
            sig_neg.data[i],
            sum
        );
    }
}

#[test]
fn test_sigmoid_extreme_values() {
    // Test numerical stability with extreme values
    let v = Vector::from_slice(&[-100.0, -50.0, 50.0, 100.0]);
    let result = v.sigmoid().unwrap();

    // Very negative → close to 0
    assert!(result.data[0] < 1e-6, "sigmoid(-100) should be ≈ 0");
    assert!(result.data[1] < 1e-6, "sigmoid(-50) should be ≈ 0");

    // Very positive → close to 1
    assert!(result.data[2] > 1.0 - 1e-6, "sigmoid(50) should be ≈ 1");
    assert!(result.data[3] > 1.0 - 1e-6, "sigmoid(100) should be ≈ 1");
}

#[test]
fn test_sigmoid_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.sigmoid();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_sigmoid_zero() {
    // sigmoid(0) should be exactly 0.5
    let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = v.sigmoid().unwrap();

    for &val in result.as_slice() {
        assert!((val - 0.5).abs() < 1e-7, "sigmoid(0) = {} ≠ 0.5", val);
    }
}

#[test]
fn test_leaky_relu_basic() {
    // Basic Leaky ReLU with α = 0.01
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.leaky_relu(0.01).unwrap();

    assert_eq!(result.as_slice(), &[-0.02, -0.01, 0.0, 1.0, 2.0]);
}

#[test]
fn test_leaky_relu_different_slopes() {
    // Test with different negative slopes
    let v = Vector::from_slice(&[-10.0, 5.0]);

    // α = 0.01 (default)
    let result_001 = v.leaky_relu(0.01).unwrap();
    assert!((result_001.data[0] - (-0.1)).abs() < 1e-6); // -10 * 0.01
    assert_eq!(result_001.data[1], 5.0);

    // α = 0.1
    let result_01 = v.leaky_relu(0.1).unwrap();
    assert!((result_01.data[0] - (-1.0)).abs() < 1e-6); // -10 * 0.1
    assert_eq!(result_01.data[1], 5.0);

    // α = 0.2
    let result_02 = v.leaky_relu(0.2).unwrap();
    assert!((result_02.data[0] - (-2.0)).abs() < 1e-6); // -10 * 0.2
    assert_eq!(result_02.data[1], 5.0);
}

#[test]
fn test_leaky_relu_reduces_to_relu() {
    // With α = 0, should behave like standard ReLU
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let leaky = v.leaky_relu(0.0).unwrap();
    let relu = v.relu().unwrap();

    for i in 0..v.len() {
        assert_eq!(leaky.data[i], relu.data[i], "α=0 should equal ReLU");
    }
}

#[test]
fn test_leaky_relu_preserves_positive() {
    // Positive values should remain unchanged regardless of α
    let v = Vector::from_slice(&[0.5, 1.0, 5.0, 10.0]);
    let result = v.leaky_relu(0.01).unwrap();

    for i in 0..v.len() {
        assert_eq!(
            result.data[i], v.data[i],
            "Positive values should be preserved"
        );
    }
}

#[test]
fn test_leaky_relu_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.leaky_relu(0.01);
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_leaky_relu_invalid_slope() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);

    // Negative slope should fail
    let result = v.leaky_relu(-0.1);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));

    // Slope >= 1.0 should fail
    let result = v.leaky_relu(1.0);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));

    let result = v.leaky_relu(1.5);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_elu_basic() {
    // Basic ELU with α = 1.0
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.elu(1.0).unwrap();

    // elu(-2, 1) = 1*(e^-2 - 1) ≈ -0.8647
    // elu(-1, 1) = 1*(e^-1 - 1) ≈ -0.6321
    assert!((result.data[0] - (-0.8647)).abs() < 0.001);
    assert!((result.data[1] - (-0.6321)).abs() < 0.001);
    assert_eq!(result.data[2], 0.0);
    assert_eq!(result.data[3], 1.0);
    assert_eq!(result.data[4], 2.0);
}

#[test]
fn test_elu_different_alphas() {
    // Test with different alpha values
    let v = Vector::from_slice(&[-1.0, 2.0]);

    // α = 1.0 (standard)
    let result_1 = v.elu(1.0).unwrap();
    assert!((result_1.data[0] - (-0.6321)).abs() < 0.001);
    assert_eq!(result_1.data[1], 2.0);

    // α = 0.5
    let result_05 = v.elu(0.5).unwrap();
    assert!((result_05.data[0] - (-0.3161)).abs() < 0.001); // 0.5 * (e^-1 - 1)
    assert_eq!(result_05.data[1], 2.0);

    // α = 2.0
    let result_2 = v.elu(2.0).unwrap();
    assert!((result_2.data[0] - (-1.2642)).abs() < 0.001); // 2.0 * (e^-1 - 1)
    assert_eq!(result_2.data[1], 2.0);
}

#[test]
fn test_elu_saturation() {
    // For very negative values, ELU saturates to -α
    let v = Vector::from_slice(&[-10.0, -20.0, -100.0]);
    let result = v.elu(1.0).unwrap();

    // All should be very close to -1.0 (saturation at -α)
    for &val in result.as_slice() {
        assert!(
            (val - (-1.0)).abs() < 0.001,
            "ELU should saturate to -α for very negative inputs, got {}",
            val
        );
    }
}

#[test]
fn test_elu_preserves_positive() {
    // Positive values should remain unchanged
    let v = Vector::from_slice(&[0.5, 1.0, 5.0, 10.0]);
    let result = v.elu(1.0).unwrap();

    for i in 0..v.len() {
        assert_eq!(
            result.data[i], v.data[i],
            "Positive values should be preserved"
        );
    }
}

#[test]
fn test_elu_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.elu(1.0);
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_elu_invalid_alpha() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);

    // Alpha <= 0 should fail
    let result = v.elu(0.0);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));

    let result = v.elu(-1.0);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_gelu_basic() {
    // Basic GELU behavior
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.gelu().unwrap();

    // gelu(0) should be exactly 0
    assert_eq!(result.data[2], 0.0);

    // Negative values should give small negative outputs
    assert!(result.data[0] < 0.0 && result.data[0] > -0.1);
    assert!(result.data[1] < 0.0 && result.data[1] > -0.2);

    // Positive values should be positive and approach linear for large x
    assert!(result.data[3] > 0.8);
    assert!(result.data[4] > 1.8);
}

#[test]
fn test_gelu_zero() {
    // gelu(0) should be exactly 0
    let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let result = v.gelu().unwrap();

    for &val in result.as_slice() {
        assert_eq!(val, 0.0, "gelu(0) should be 0");
    }
}

#[test]
fn test_gelu_smoothness() {
    // GELU is smooth everywhere - test that it produces reasonable values
    let v = Vector::from_slice(&[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
    let result = v.gelu().unwrap();

    // All outputs should be finite
    for &val in result.as_slice() {
        assert!(val.is_finite(), "GELU output should be finite");
    }

    // Verify increasing trend (though not strictly monotonic)
    // Generally gelu increases with x
    assert!(result.data[0] < result.data[3]); // gelu(-3) < gelu(0)
    assert!(result.data[3] < result.data[6]); // gelu(0) < gelu(3)
}

#[test]
fn test_gelu_large_positive() {
    // For large positive x, gelu(x) ≈ x (linear behavior)
    let v = Vector::from_slice(&[5.0, 10.0, 20.0]);
    let result = v.gelu().unwrap();

    for i in 0..v.len() {
        // Should be very close to x for large positive values
        assert!(
            (result.data[i] - v.data[i]).abs() < 0.01,
            "gelu({}) = {} should ≈ {} for large positive x",
            v.data[i],
            result.data[i],
            v.data[i]
        );
    }
}

#[test]
fn test_gelu_large_negative() {
    // For large negative x, gelu(x) ≈ 0
    let v = Vector::from_slice(&[-5.0, -10.0, -20.0]);
    let result = v.gelu().unwrap();

    for &val in result.as_slice() {
        assert!(
            val.abs() < 0.001,
            "gelu should approach 0 for large negative inputs, got {}",
            val
        );
    }
}

#[test]
fn test_gelu_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.gelu();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// ============================================================================
// Swish (SiLU) Tests
// ============================================================================

#[test]
fn test_swish_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.swish().unwrap();

    // swish(-2) ≈ -0.238, swish(-1) ≈ -0.269, swish(0) = 0
    // swish(1) ≈ 0.731, swish(2) ≈ 1.762
    assert!((result.as_slice()[0] - (-0.238)).abs() < 0.01);
    assert!((result.as_slice()[1] - (-0.269)).abs() < 0.01);
    assert_eq!(result.as_slice()[2], 0.0);
    assert!((result.as_slice()[3] - 0.731).abs() < 0.01);
    assert!((result.as_slice()[4] - 1.762).abs() < 0.01);
}

#[test]
fn test_swish_zero() {
    let v = Vector::from_slice(&[0.0]);
    let result = v.swish().unwrap();
    assert_eq!(result.as_slice()[0], 0.0); // swish(0) = 0
}

#[test]
fn test_swish_minimum() {
    // Swish has a minimum value around x ≈ -1.278, value ≈ -0.278
    let v = Vector::from_slice(&[-2.0, -1.5, -1.278, -1.0, -0.5]);
    let result = v.swish().unwrap();

    // All values should be above the minimum
    for &val in result.as_slice() {
        assert!(val > -0.3, "Swish value {} below minimum", val);
    }

    // The middle value (closest to -1.278) should be near the minimum
    assert!(result.as_slice()[2] < -0.27);
    assert!(result.as_slice()[2] > -0.29);
}

#[test]
fn test_swish_large_positive() {
    // For large positive x, swish(x) ≈ x (linear behavior)
    let v = Vector::from_slice(&[10.0, 20.0, 50.0]);
    let result = v.swish().unwrap();

    assert!((result.as_slice()[0] - 10.0).abs() < 0.01);
    assert!((result.as_slice()[1] - 20.0).abs() < 0.01);
    assert!((result.as_slice()[2] - 50.0).abs() < 0.01);
}

#[test]
fn test_swish_large_negative() {
    // For large negative x, swish(x) ≈ 0
    let v = Vector::from_slice(&[-10.0, -20.0, -50.0]);
    let result = v.swish().unwrap();

    // swish(-10) ≈ -0.000454, swish(-20) ≈ -4.1e-9, swish(-50) ≈ 0
    assert!(result.as_slice()[0].abs() < 1e-3);
    assert!(result.as_slice()[1].abs() < 1e-7);
    assert!(result.as_slice()[2].abs() < 1e-15); // Effectively 0
}

#[test]
fn test_swish_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.swish();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// Hardswish activation tests
#[test]
fn test_hardswish_basic() {
    let v = Vector::from_slice(&[-4.0, -3.0, -1.5, 0.0, 1.5, 3.0, 4.0]);
    let result = v.hardswish().unwrap();

    // x <= -3: 0
    assert_eq!(result.as_slice()[0], 0.0);
    assert_eq!(result.as_slice()[1], 0.0);

    // -3 < x < 3: x * (x + 3) / 6
    // hardswish(-1.5) = -1.5 * 1.5 / 6 = -0.375
    assert!((result.as_slice()[2] - (-0.375)).abs() < 1e-5);
    // hardswish(0) = 0 * 3 / 6 = 0
    assert_eq!(result.as_slice()[3], 0.0);
    // hardswish(1.5) = 1.5 * 4.5 / 6 = 1.125
    assert!((result.as_slice()[4] - 1.125).abs() < 1e-5);

    // x >= 3: x
    assert_eq!(result.as_slice()[5], 3.0);
    assert_eq!(result.as_slice()[6], 4.0);
}

#[test]
fn test_hardswish_zero() {
    let v = Vector::from_slice(&[0.0]);
    let result = v.hardswish().unwrap();
    assert_eq!(result.as_slice()[0], 0.0);
}

#[test]
fn test_hardswish_boundary_values() {
    // Test exact boundary values
    let v = Vector::from_slice(&[-3.0, 3.0]);
    let result = v.hardswish().unwrap();

    // At x = -3: 0 (boundary)
    assert_eq!(result.as_slice()[0], 0.0);
    // At x = 3: 3 (boundary)
    assert_eq!(result.as_slice()[1], 3.0);
}

#[test]
fn test_hardswish_large_values() {
    let v = Vector::from_slice(&[-100.0, -10.0, 10.0, 100.0]);
    let result = v.hardswish().unwrap();

    // Large negative: 0
    assert_eq!(result.as_slice()[0], 0.0);
    assert_eq!(result.as_slice()[1], 0.0);

    // Large positive: x
    assert_eq!(result.as_slice()[2], 10.0);
    assert_eq!(result.as_slice()[3], 100.0);
}

#[test]
fn test_hardswish_transition_region() {
    // Test values in the transition region (-3, 3)
    let v = Vector::from_slice(&[-2.0, -1.0, 1.0, 2.0]);
    let result = v.hardswish().unwrap();

    // hardswish(-2) = -2 * 1 / 6 = -0.333...
    assert!((result.as_slice()[0] - (-1.0 / 3.0)).abs() < 1e-5);
    // hardswish(-1) = -1 * 2 / 6 = -0.333...
    assert!((result.as_slice()[1] - (-1.0 / 3.0)).abs() < 1e-5);
    // hardswish(1) = 1 * 4 / 6 = 0.666...
    assert!((result.as_slice()[2] - (2.0 / 3.0)).abs() < 1e-5);
    // hardswish(2) = 2 * 5 / 6 = 1.666...
    assert!((result.as_slice()[3] - (5.0 / 3.0)).abs() < 1e-5);
}

#[test]
fn test_hardswish_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.hardswish();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// Mish activation tests
#[test]
fn test_mish_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.mish().unwrap();

    // mish has small negative values for negative inputs
    assert!(result.as_slice()[0] < 0.0);
    assert!(result.as_slice()[1] < 0.0);

    // mish(0) is a small positive value (0 * tanh(ln(2)) = 0)
    assert!(result.as_slice()[2].abs() < 1e-5);

    // Positive inputs give positive outputs
    assert!(result.as_slice()[3] > 0.0);
    assert!(result.as_slice()[4] > 0.0);
}

#[test]
fn test_mish_zero() {
    let v = Vector::from_slice(&[0.0]);
    let result = v.mish().unwrap();
    // mish(0) = 0 * tanh(ln(2)) = 0
    assert!(result.as_slice()[0].abs() < 1e-10);
}

#[test]
fn test_mish_large_positive() {
    // For large positive x, mish(x) ≈ x
    let v = Vector::from_slice(&[10.0, 20.0, 50.0]);
    let result = v.mish().unwrap();

    // Should be very close to x for large values
    assert!((result.as_slice()[0] - 10.0).abs() < 0.001);
    assert!((result.as_slice()[1] - 20.0).abs() < 0.001);
    assert!((result.as_slice()[2] - 50.0).abs() < 0.001);
}

#[test]
fn test_mish_large_negative() {
    // For large negative x, mish(x) ≈ 0
    let v = Vector::from_slice(&[-10.0, -20.0, -50.0]);
    let result = v.mish().unwrap();

    // Should be very close to 0 for large negative values
    assert!(result.as_slice()[0].abs() < 0.001);
    assert!(result.as_slice()[1].abs() < 1e-6);
    assert!(result.as_slice()[2].abs() < 1e-10);
}

#[test]
fn test_mish_minimum() {
    // Mish has a minimum around x ≈ -1.19 with value ≈ -0.31
    let v = Vector::from_slice(&[-1.19]);
    let result = v.mish().unwrap();

    // Should be close to the minimum value
    assert!(result.as_slice()[0] < -0.2);
    assert!(result.as_slice()[0] > -0.4);
}

#[test]
fn test_mish_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.mish();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

// SELU unit tests

#[test]
fn test_selu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.selu().unwrap();
    let data = result.as_slice();

    // SELU constants
    const LAMBDA: f32 = 1.0507009873554804934193349852946;
    const ALPHA: f32 = 1.6732632423543772848170429916717;

    // Positive values: selu(x) = λ * x
    assert!((data[3] - LAMBDA * 1.0).abs() < 1e-5); // selu(1.0) = λ
    assert!((data[4] - LAMBDA * 2.0).abs() < 1e-5); // selu(2.0) = 2λ

    // Zero: selu(0) = 0
    assert!(data[2].abs() < 1e-5);

    // Negative values: selu(x) = λ * α * (exp(x) - 1)
    let expected_neg1 = LAMBDA * ALPHA * ((-1.0_f32).exp() - 1.0);
    assert!((data[1] - expected_neg1).abs() < 1e-5);
}

#[test]
fn test_selu_zero() {
    let v = Vector::from_slice(&[0.0]);
    let result = v.selu().unwrap();
    assert!(result.as_slice()[0].abs() < 1e-10);
}

#[test]
fn test_selu_positive_scaling() {
    // For positive values, selu(x) = λ * x
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 10.0]);
    let result = v.selu().unwrap();
    let data = result.as_slice();

    const LAMBDA: f32 = 1.0507009873554804934193349852946;

    for (i, &x) in [1.0, 2.0, 3.0, 10.0].iter().enumerate() {
        assert!(
            (data[i] - LAMBDA * x).abs() < 1e-5,
            "selu({}) should be {} but got {}",
            x,
            LAMBDA * x,
            data[i]
        );
    }
}

#[test]
fn test_selu_negative_asymptote() {
    // For very negative x, selu(x) → -λ * α ≈ -1.7581
    let v = Vector::from_slice(&[-100.0]);
    let result = v.selu().unwrap();

    const LAMBDA: f32 = 1.0507009873554804934193349852946;
    const ALPHA: f32 = 1.6732632423543772848170429916717;
    let asymptote = -LAMBDA * ALPHA;

    assert!(
        (result.as_slice()[0] - asymptote).abs() < 1e-4,
        "selu(-100) should approach {} but got {}",
        asymptote,
        result.as_slice()[0]
    );
}

#[test]
fn test_selu_continuity_at_zero() {
    // Test values approaching zero from both sides
    let eps = 1e-6;
    let v = Vector::from_slice(&[-eps, 0.0, eps]);
    let result = v.selu().unwrap();
    let data = result.as_slice();

    // All should be very close to zero (continuous at x=0)
    assert!(data[0].abs() < 1e-3);
    assert!(data[1].abs() < 1e-10);
    assert!(data[2].abs() < 1e-3);
}

#[test]
fn test_selu_empty_vector() {
    let v = Vector::from_slice(&[]);
    let result = v.selu();
    assert!(matches!(result, Err(TruenoError::EmptyVector)));
}

#[test]
fn test_aligned_vector_creation() {
    let v = Vector::with_alignment(100, Backend::SSE2, 16).unwrap();

    // Verify the vector has the correct size
    assert_eq!(v.len(), 100);

    // Check alignment (Vec allocator typically provides good alignment)
    let ptr = v.as_slice().as_ptr() as usize;
    // Note: We can't guarantee specific alignment with standard Vec,
    // but we can verify it's at least naturally aligned for f32 (4 bytes)
    assert_eq!(ptr % 4, 0, "Vector data should be at least 4-byte aligned");

    // Most modern allocators provide 16-byte alignment by default
    // This is informational, not required
    if ptr.is_multiple_of(16) {
        println!("Got 16-byte alignment from standard allocator");
    }
}

#[test]
fn test_aligned_vector_operations() {
    // RED: This test will fail until we implement aligned allocation
    let a = Vector::with_alignment(1000, Backend::SSE2, 16).unwrap();
    let b = Vector::with_alignment(1000, Backend::SSE2, 16).unwrap();

    // Operations on aligned vectors should work correctly
    let result = a.add(&b);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 1000);
}

// Parallel execution tests (for vectors >= 100_000 elements)
#[test]
fn test_add_parallel_large_vector() {
    // Test parallel execution path for add (>= 100_000 elements)
    const SIZE: usize = 150_000;
    let a_data: Vec<f32> = (0..SIZE).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|i| (i * 2) as f32).collect();

    let a = Vector::from_slice(&a_data);
    let b = Vector::from_slice(&b_data);
    let result = a.add(&b).unwrap();

    // Verify correctness
    assert_eq!(result.len(), SIZE);
    for i in 0..SIZE {
        let expected = a_data[i] + b_data[i];
        assert!((result.as_slice()[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_sub_parallel_large_vector() {
    // Test parallel execution path for sub (>= 100_000 elements)
    const SIZE: usize = 150_000;
    let a_data: Vec<f32> = (0..SIZE).map(|i| (i * 3) as f32).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|i| i as f32).collect();

    let a = Vector::from_slice(&a_data);
    let b = Vector::from_slice(&b_data);
    let result = a.sub(&b).unwrap();

    // Verify correctness
    assert_eq!(result.len(), SIZE);
    for i in 0..SIZE {
        let expected = a_data[i] - b_data[i];
        assert!((result.as_slice()[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_mul_parallel_large_vector() {
    // Test parallel execution path for mul (>= 100_000 elements)
    const SIZE: usize = 150_000;
    let a_data: Vec<f32> = (0..SIZE).map(|i| (i % 100) as f32 + 1.0).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|i| 2.0 + (i % 50) as f32).collect();

    let a = Vector::from_slice(&a_data);
    let b = Vector::from_slice(&b_data);
    let result = a.mul(&b).unwrap();

    // Verify correctness
    assert_eq!(result.len(), SIZE);
    for i in 0..SIZE {
        let expected = a_data[i] * b_data[i];
        assert!((result.as_slice()[i] - expected).abs() < 1e-3);
    }
}

#[test]
fn test_div_parallel_large_vector() {
    // Test parallel execution path for div (>= 100_000 elements)
    const SIZE: usize = 150_000;
    let a_data: Vec<f32> = (0..SIZE).map(|i| (i + 100) as f32).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|i| (i % 50) as f32 + 1.0).collect();

    let a = Vector::from_slice(&a_data);
    let b = Vector::from_slice(&b_data);
    let result = a.div(&b).unwrap();

    // Verify correctness
    assert_eq!(result.len(), SIZE);
    for i in 0..SIZE {
        let expected = a_data[i] / b_data[i];
        assert!((result.as_slice()[i] - expected).abs() < 1e-3);
    }
}

#[test]
fn test_dot_parallel_large_vector() {
    // Test parallel execution path for dot (>= 500_000 elements)
    const SIZE: usize = 600_000;
    let a_data: Vec<f32> = (0..SIZE).map(|i| (i % 100) as f32).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|i| 1.0 + (i % 50) as f32).collect();

    let a = Vector::from_slice(&a_data);
    let b = Vector::from_slice(&b_data);
    let result = a.dot(&b).unwrap();

    // Verify it's a reasonable value (not checking exact value due to FP precision)
    assert!(result.is_finite());
    assert!(result > 0.0);
}

#[test]
fn test_fma_parallel_large_vector() {
    // Test parallel execution path for fma (>= 100_000 elements)
    const SIZE: usize = 150_000;
    let a_data: Vec<f32> = (0..SIZE).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|_| 2.0).collect();
    let c_data: Vec<f32> = (0..SIZE).map(|i| 10.0 + i as f32).collect();

    let a = Vector::from_slice(&a_data);
    let b = Vector::from_slice(&b_data);
    let c = Vector::from_slice(&c_data);
    let result = a.fma(&b, &c).unwrap();

    // Verify correctness: fma(a, b, c) = a * b + c
    assert_eq!(result.len(), SIZE);
    for i in 0..SIZE {
        let expected = a_data[i] * b_data[i] + c_data[i];
        assert!((result.as_slice()[i] - expected).abs() < 1e-3);
    }
}

#[test]
fn test_scale_parallel_large_vector() {
    // Test parallel execution path for scale (>= 100_000 elements)
    const SIZE: usize = 150_000;
    let data: Vec<f32> = (0..SIZE).map(|i| i as f32).collect();

    let v = Vector::from_slice(&data);
    let result = v.scale(3.0).unwrap();

    // Verify correctness
    assert_eq!(result.len(), SIZE);
    for (&original, &scaled) in data.iter().zip(result.as_slice().iter()) {
        let expected = original * 3.0;
        assert!((scaled - expected).abs() < 1e-5);
    }
}

#[test]
fn test_parallel_execution_correctness() {
    // Verify parallel and sequential execution produce same results
    const SIZE: usize = 150_000;
    let a_data: Vec<f32> = (0..SIZE).map(|i| (i % 1000) as f32).collect();
    let b_data: Vec<f32> = (0..SIZE).map(|i| (i % 500) as f32 + 1.0).collect();

    let a_large = Vector::from_slice(&a_data);
    let b_large = Vector::from_slice(&b_data);
    let result_parallel = a_large.add(&b_large).unwrap();

    // Compare with small vector (sequential execution)
    const SMALL_SIZE: usize = 100;
    let a_small = Vector::from_slice(&a_data[..SMALL_SIZE]);
    let b_small = Vector::from_slice(&b_data[..SMALL_SIZE]);
    let result_sequential = a_small.add(&b_small).unwrap();

    // First SMALL_SIZE elements should match
    for i in 0..SMALL_SIZE {
        assert_eq!(
            result_parallel.as_slice()[i],
            result_sequential.as_slice()[i]
        );
    }
}

// AVX512 SIMD path tests (need 16+ elements to trigger SIMD loops)
// These tests ensure the SIMD implementations are exercised

#[test]
fn test_norm_l1_avx512_path() {
    // 32 elements to ensure AVX512 loop runs twice (32 / 16 = 2)
    let data: Vec<f32> = (0..32)
        .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
        .collect();
    let v = Vector::from_slice(&data);
    let result = v.norm_l1().unwrap();
    // Sum of |0| + |1| + |2| + ... + |31| = 0 + 1 + 2 + ... + 31 = 31*32/2 = 496
    assert!((result - 496.0).abs() < 1e-3);
}

#[test]
fn test_norm_linf_avx512_path() {
    // 32 elements to ensure AVX512 loop runs twice
    let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    data[17] = -100.0; // Make element 17 the max absolute value
    let v = Vector::from_slice(&data);
    let result = v.norm_linf().unwrap();
    assert!((result - 100.0).abs() < 1e-5);
}

#[test]
fn test_scale_avx512_path() {
    // 64 elements to ensure multiple AVX512 iterations
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let v = Vector::from_slice(&data);
    let result = v.scale(2.0).unwrap();
    for i in 0..64 {
        assert!((result.as_slice()[i] - (i as f32 * 2.0)).abs() < 1e-5);
    }
}

#[test]
fn test_abs_avx512_path() {
    // 48 elements to ensure AVX512 loop runs 3 times (48 / 16 = 3)
    let data: Vec<f32> = (0..48)
        .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
        .collect();
    let v = Vector::from_slice(&data);
    let result = v.abs().unwrap();
    for i in 0..48 {
        assert!((result.as_slice()[i] - (i as f32)).abs() < 1e-5);
    }
}

#[test]
fn test_clamp_avx512_path() {
    // 32 elements with values spanning the clamp range
    let data: Vec<f32> = (0..32).map(|i| (i as f32) - 10.0).collect();
    let v = Vector::from_slice(&data);
    let result = v.clamp(0.0, 15.0).unwrap();
    for i in 0..32 {
        let expected = ((i as f32) - 10.0).clamp(0.0, 15.0);
        assert!((result.as_slice()[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_lerp_avx512_path() {
    // 32 elements
    let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..32).map(|i| (i as f32) * 2.0).collect();
    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);
    let result = va.lerp(&vb, 0.5).unwrap();
    // lerp(a, b, 0.5) = a + 0.5 * (b - a) = 0.5*a + 0.5*b = (a + b) / 2
    for i in 0..32 {
        let expected = (i as f32 + i as f32 * 2.0) / 2.0;
        assert!((result.as_slice()[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_fma_avx512_path() {
    // 32 elements: a*b + c
    let a: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..32).map(|_| 2.0).collect();
    let c: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let va = Vector::from_slice(&a);
    let vb = Vector::from_slice(&b);
    let vc = Vector::from_slice(&c);
    let result = va.fma(&vb, &vc).unwrap();
    // a*b + c = i*2 + i = 3*i
    for i in 0..32 {
        let expected = 3.0 * (i as f32);
        assert!((result.as_slice()[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_argmax_avx512_path() {
    // 32 elements with max at position 25
    let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    data[25] = 1000.0;
    let v = Vector::from_slice(&data);
    let result = v.argmax().unwrap();
    assert_eq!(result, 25);
}

#[test]
fn test_argmin_avx512_path() {
    // 32 elements with min at position 18
    let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    data[18] = -500.0;
    let v = Vector::from_slice(&data);
    let result = v.argmin().unwrap();
    assert_eq!(result, 18);
}
