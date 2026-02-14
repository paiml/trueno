use super::super::super::super::*;
use crate::Backend;

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

// L-inf Norm (infinity/max norm) tests
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
