use super::super::super::super::super::*;

// Tests for clip() - Constrain values to [min, max] range
// ========================================================================

#[test]
fn test_clip_basic() {
    // [-5, 0, 5, 10, 15] clipped to [0, 10] -> [0, 0, 5, 10, 10]
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
    // All values below min -> all become min
    let v = Vector::from_slice(&[-10.0, -5.0, -2.0]);
    let clipped = v.clip(0.0, 10.0).unwrap();

    assert_eq!(clipped.as_slice(), &[0.0, 0.0, 0.0]);
}

#[test]
fn test_clip_all_above() {
    // All values above max -> all become max
    let v = Vector::from_slice(&[15.0, 20.0, 25.0]);
    let clipped = v.clip(0.0, 10.0).unwrap();

    assert_eq!(clipped.as_slice(), &[10.0, 10.0, 10.0]);
}

#[test]
fn test_clip_invalid_range() {
    // min > max -> InvalidInput error
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.clip(10.0, 5.0);

    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_clip_equal_bounds() {
    // min == max -> all values become that value
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

    // Verify sum ~= 1
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
    // All equal inputs -> uniform distribution
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

    // Verify sum ~= 1
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
    // Single element -> probability 1.0
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
    // All equal inputs -> uniform log probabilities
    let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
    let log_probs = v.log_softmax().unwrap();

    // Each should be log(1/4) = log(0.25) ~= -1.386
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
    // Single element -> log probability = log(1.0) = 0.0
    let v = Vector::from_slice(&[5.0]);
    let log_probs = v.log_softmax().unwrap();

    assert!(
        log_probs.data[0].abs() < 1e-5,
        "log_softmax of single element should be 0.0, got {}",
        log_probs.data[0]
    );
}
