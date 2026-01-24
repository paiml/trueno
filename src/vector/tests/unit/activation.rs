use super::super::super::*;
use crate::Backend;

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
