use super::super::super::super::*;

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
