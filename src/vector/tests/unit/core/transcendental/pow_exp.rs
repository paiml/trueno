use super::super::super::super::super::*;
use crate::Backend;

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
