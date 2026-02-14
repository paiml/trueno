use super::super::super::super::*;
use crate::Backend;

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
