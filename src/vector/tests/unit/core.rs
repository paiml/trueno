use super::super::super::*;
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
    assert!(
        dot_rel_err < 0.01,
        "dot mismatch: {} vs {}",
        dot,
        expected_dot
    );

    // Test sum
    let sum = v1.sum().unwrap();
    let expected_sum: f32 = data1.iter().sum();
    let sum_rel_err = (sum - expected_sum).abs() / expected_sum.abs().max(1.0);
    assert!(
        sum_rel_err < 0.01,
        "sum mismatch: {} vs {}",
        sum,
        expected_sum
    );

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
