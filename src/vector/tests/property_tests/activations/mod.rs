mod smooth_activations;

use super::super::super::*;
use proptest::prelude::*;

// ========================================================================
// Property tests for softmax() - Probability distribution
// ========================================================================

proptest! {
    /// Property test: softmax() produces values that sum to 1
    #[test]
    fn test_softmax_sums_to_one(
        a in prop::collection::vec(-50.0f32..50.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let probs = va.softmax().unwrap();
        let sum: f32 = probs.as_slice().iter().sum();

        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax sum = {}, expected 1.0",
            sum
        );
    }
}

proptest! {
    /// Property test: softmax() produces values in [0, 1]
    #[test]
    fn test_softmax_in_unit_range(
        a in prop::collection::vec(-50.0f32..50.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let probs = va.softmax().unwrap();

        for &p in probs.as_slice() {
            prop_assert!(
                (0.0..=1.0).contains(&p),
                "probability {} not in [0, 1]",
                p
            );
        }
    }
}

proptest! {
    /// Property test: softmax() is translation invariant
    /// softmax(x + c) = softmax(x) for any constant c
    #[test]
    fn test_softmax_translation_invariant(
        a in prop::collection::vec(-20.0f32..20.0, 2..50),
        c in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);
        let probs1 = va.softmax().unwrap();

        // Add constant to all elements
        let shifted: Vec<f32> = a.iter().map(|&x| x + c).collect();
        let vb = Vector::from_slice(&shifted);
        let probs2 = vb.softmax().unwrap();

        // Probabilities should be identical
        for i in 0..probs1.len() {
            prop_assert!(
                (probs1.data[i] - probs2.data[i]).abs() < 1e-4,
                "Translation invariance violated at index {}: softmax(x)={}, softmax(x+{})={}",
                i, probs1.data[i], c, probs2.data[i]
            );
        }
    }
}

// ========================================================================
// Property tests for log_softmax() - Log probability distribution
// ========================================================================

proptest! {
    /// Property test: exp(log_softmax(x)) sums to 1
    /// Since log_softmax returns log probabilities, exponentiating should give valid probabilities
    #[test]
    fn test_log_softmax_exp_sums_to_one(
        a in prop::collection::vec(-50.0f32..50.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let log_probs = va.log_softmax().unwrap();

        // Exponentiate to get probabilities
        let sum: f32 = log_probs.as_slice().iter().map(|&lp| lp.exp()).sum();

        prop_assert!(
            (sum - 1.0).abs() < 1e-4,
            "exp(log_softmax) sum = {}, expected 1.0",
            sum
        );
    }
}

proptest! {
    /// Property test: log_softmax() produces values <= 0
    /// Since probabilities are in [0, 1], log(prob) <= 0
    #[test]
    fn test_log_softmax_non_positive(
        a in prop::collection::vec(-50.0f32..50.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let log_probs = va.log_softmax().unwrap();

        for &lp in log_probs.as_slice() {
            prop_assert!(
                lp <= 1e-5,
                "log_probability {} should be <= 0",
                lp
            );
        }
    }
}

proptest! {
    /// Property test: log_softmax() is translation invariant
    /// log_softmax(x + c) = log_softmax(x) for any constant c
    #[test]
    fn test_log_softmax_translation_invariant(
        a in prop::collection::vec(-20.0f32..20.0, 2..50),
        c in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);
        let log_probs1 = va.log_softmax().unwrap();

        // Add constant to all elements
        let shifted: Vec<f32> = a.iter().map(|&x| x + c).collect();
        let vb = Vector::from_slice(&shifted);
        let log_probs2 = vb.log_softmax().unwrap();

        // Log probabilities should be identical
        for i in 0..log_probs1.len() {
            prop_assert!(
                (log_probs1.data[i] - log_probs2.data[i]).abs() < 1e-4,
                "Translation invariance violated at index {}: log_softmax(x)={}, log_softmax(x+{})={}",
                i, log_probs1.data[i], c, log_probs2.data[i]
            );
        }
    }
}

// ========================================================================
// Property tests for relu() - Rectified Linear Unit
// ========================================================================

proptest! {
    /// Property test: relu() produces non-negative outputs
    /// All outputs should be >= 0
    #[test]
    fn test_relu_non_negative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.relu().unwrap();

        for &val in result.as_slice() {
            prop_assert!(
                val >= 0.0,
                "ReLU output {} should be non-negative",
                val
            );
        }
    }
}

proptest! {
    /// Property test: relu() preserves positive values
    /// For all x > 0, relu(x) = x
    #[test]
    fn test_relu_preserves_positive(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.relu().unwrap();

        for (i, &val) in a.iter().enumerate() {
            if val > 0.0 {
                prop_assert!(
                    (result.data[i] - val).abs() < 1e-6,
                    "ReLU should preserve positive value: {} became {}",
                    val, result.data[i]
                );
            }
        }
    }
}

proptest! {
    /// Property test: relu() is idempotent
    /// relu(relu(x)) = relu(x)
    #[test]
    fn test_relu_idempotent(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let relu1 = va.relu().unwrap();
        let relu2 = relu1.relu().unwrap();

        for (i, &orig_val) in a.iter().enumerate() {
            prop_assert!(
                (relu1.data[i] - relu2.data[i]).abs() < 1e-6,
                "ReLU should be idempotent: relu(relu({})) = {} != relu({}) = {}",
                orig_val, relu2.data[i], orig_val, relu1.data[i]
            );
        }
    }
}

// ========================================================================
// Property tests for sigmoid() - Logistic activation
// ========================================================================

proptest! {
    /// Property test: sigmoid() produces values in [0, 1]
    #[test]
    fn test_sigmoid_bounded(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sigmoid().unwrap();

        for &val in result.as_slice() {
            prop_assert!(
                (0.0..=1.0).contains(&val),
                "Sigmoid output {} not in [0, 1]",
                val
            );
        }
    }
}

proptest! {
    /// Property test: sigmoid() symmetry σ(-x) = 1 - σ(x)
    #[test]
    fn test_sigmoid_symmetry_property(
        a in prop::collection::vec(-50.0f32..50.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let sig_pos = va.sigmoid().unwrap();

        // Create negated vector
        let a_neg: Vec<f32> = a.iter().map(|&x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let sig_neg = va_neg.sigmoid().unwrap();

        // σ(-x) + σ(x) should equal 1
        for (i, &val) in a.iter().enumerate() {
            let sum = sig_pos.data[i] + sig_neg.data[i];
            prop_assert!(
                (sum - 1.0).abs() < 1e-5,
                "Symmetry violated: σ({}) + σ({}) = {} + {} = {} ≠ 1",
                val, -val, sig_pos.data[i], sig_neg.data[i], sum
            );
        }
    }
}

proptest! {
    /// Property test: sigmoid() is monotonically increasing
    /// If x < y, then σ(x) < σ(y)
    #[test]
    fn test_sigmoid_monotonic(
        a in prop::collection::vec(-50.0f32..50.0, 2..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sigmoid().unwrap();

        // Check all pairs for monotonicity
        for i in 0..a.len() {
            for j in 0..a.len() {
                if a[i] < a[j] {
                    prop_assert!(
                        result.data[i] < result.data[j] + 1e-6,
                        "Monotonicity violated: {} < {} but σ({}) = {} >= σ({}) = {}",
                        a[i], a[j], a[i], result.data[i], a[j], result.data[j]
                    );
                }
            }
        }
    }
}

// ========================================================================
// Property tests for leaky_relu() - Leaky Rectified Linear Unit
// ========================================================================

proptest! {
    /// Property test: leaky_relu() preserves positive values exactly
    #[test]
    fn test_leaky_relu_preserves_positive_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        alpha in 0.0f32..1.0
    ) {
        let va = Vector::from_slice(&a);
        let result = va.leaky_relu(alpha).unwrap();

        for (i, &val) in a.iter().enumerate() {
            if val > 0.0 {
                prop_assert!(
                    (result.data[i] - val).abs() < 1e-6,
                    "Positive value {} should be preserved, got {}",
                    val, result.data[i]
                );
            }
        }
    }
}

proptest! {
    /// Property test: leaky_relu() scales negative values by alpha
    #[test]
    fn test_leaky_relu_scales_negative_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        alpha in 0.01f32..0.5 // Use smaller range to avoid precision issues
    ) {
        let va = Vector::from_slice(&a);
        let result = va.leaky_relu(alpha).unwrap();

        for (i, &val) in a.iter().enumerate() {
            if val < 0.0 {
                let expected = alpha * val;
                prop_assert!(
                    (result.data[i] - expected).abs() < 1e-4,
                    "Negative value {} should be scaled by {}: expected {}, got {}",
                    val, alpha, expected, result.data[i]
                );
            }
        }
    }
}

proptest! {
    /// Property test: leaky_relu() is monotonically increasing
    /// If x < y, then leaky_relu(x) < leaky_relu(y)
    #[test]
    fn test_leaky_relu_monotonic_property(
        a in prop::collection::vec(-50.0f32..50.0, 2..100),
        alpha in 0.01f32..0.5
    ) {
        let va = Vector::from_slice(&a);
        let result = va.leaky_relu(alpha).unwrap();

        // Check all pairs for monotonicity
        for i in 0..a.len() {
            for j in 0..a.len() {
                if a[i] < a[j] {
                    prop_assert!(
                        result.data[i] < result.data[j] + 1e-5,
                        "Monotonicity violated: {} < {} but leaky_relu({}) = {} >= leaky_relu({}) = {}",
                        a[i], a[j], a[i], result.data[i], a[j], result.data[j]
                    );
                }
            }
        }
    }
}

// ========================================================================
// Property tests for elu() - Exponential Linear Unit
// ========================================================================

proptest! {
    /// Property test: elu() preserves positive values exactly
    #[test]
    fn test_elu_preserves_positive_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        alpha in 0.1f32..5.0
    ) {
        let va = Vector::from_slice(&a);
        let result = va.elu(alpha).unwrap();

        for (i, &val) in a.iter().enumerate() {
            if val > 0.0 {
                prop_assert!(
                    (result.data[i] - val).abs() < 1e-6,
                    "Positive value {} should be preserved, got {}",
                    val, result.data[i]
                );
            }
        }
    }
}

proptest! {
    /// Property test: elu() produces values >= -alpha for negative inputs
    /// ELU saturates to -α as x → -∞
    #[test]
    fn test_elu_bounded_below_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        alpha in 0.1f32..5.0
    ) {
        let va = Vector::from_slice(&a);
        let result = va.elu(alpha).unwrap();

        for &val in result.as_slice() {
            prop_assert!(
                val >= -alpha - 0.01,
                "ELU output {} should be >= -α = {}",
                val, -alpha
            );
        }
    }
}

proptest! {
    /// Property test: elu() is monotonically increasing
    /// If x < y, then elu(x) < elu(y)
    #[test]
    fn test_elu_monotonic_property(
        a in prop::collection::vec(-20.0f32..20.0, 2..50),
        alpha in 0.5f32..2.0
    ) {
        let va = Vector::from_slice(&a);
        let result = va.elu(alpha).unwrap();

        // Check all pairs for monotonicity
        for i in 0..a.len() {
            for j in 0..a.len() {
                if a[i] < a[j] {
                    prop_assert!(
                        result.data[i] < result.data[j] + 1e-5,
                        "Monotonicity violated: {} < {} but elu({}) = {} >= elu({}) = {}",
                        a[i], a[j], a[i], result.data[i], a[j], result.data[j]
                    );
                }
            }
        }
    }
}

// ========================================================================
// Property tests for gelu() - Gaussian Error Linear Unit
// ========================================================================

proptest! {
    /// Property test: gelu() produces finite values
    #[test]
    fn test_gelu_finite_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.gelu().unwrap();

        for &val in result.as_slice() {
            prop_assert!(
                val.is_finite(),
                "GELU output {} should be finite",
                val
            );
        }
    }
}

proptest! {
    /// Property test: gelu(0) = 0
    #[test]
    fn test_gelu_zero_property(
        _a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let v = Vector::from_slice(&[0.0]);
        let result = v.gelu().unwrap();

        prop_assert!(
            result.data[0].abs() < 1e-10,
            "gelu(0) should be 0, got {}",
            result.data[0]
        );
    }
}

proptest! {
    /// Property test: For large positive x, gelu(x) ≈ x
    #[test]
    fn test_gelu_linear_large_positive(
        a in prop::collection::vec(5.0f32..100.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.gelu().unwrap();

        for (i, &val) in a.iter().enumerate() {
            // For large positive values, gelu(x) should be very close to x
            prop_assert!(
                (result.data[i] - val).abs() < 0.01,
                "For large positive {}, gelu should ≈ x, got {} vs {}",
                val, result.data[i], val
            );
        }
    }
}

proptest! {
    /// Property test: swish() produces finite values
    #[test]
    fn test_swish_finite_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.swish().unwrap();

        for &val in result.as_slice() {
            prop_assert!(val.is_finite(), "Swish output should be finite");
        }
    }
}

proptest! {
    /// Property test: swish(0) = 0 always
    #[test]
    fn test_swish_zero_property(
        a in prop::collection::vec(-0.001f32..0.001, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.swish().unwrap();

        // For values very close to 0, swish should also be close to 0
        for &val in result.as_slice() {
            prop_assert!(
                val.abs() < 0.001,
                "Swish of near-zero should be near-zero, got {}",
                val
            );
        }
    }
}

proptest! {
    /// Property test: For large positive x, swish(x) ≈ x (linear)
    #[test]
    fn test_swish_linear_large_positive(
        a in prop::collection::vec(10.0f32..100.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.swish().unwrap();

        for (i, &val) in a.iter().enumerate() {
            // For large positive values, swish(x) should be very close to x
            prop_assert!(
                (result.data[i] - val).abs() < 0.01,
                "For large positive {}, swish should ≈ x, got {} vs {}",
                val, result.data[i], val
            );
        }
    }
}

proptest! {
    /// Property test: hardswish() produces finite values
    #[test]
    fn test_hardswish_finite_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.hardswish().unwrap();

        for &val in result.as_slice() {
            prop_assert!(val.is_finite(), "Hardswish output should be finite");
        }
    }
}

proptest! {
    /// Property test: hardswish(0) = 0 always
    #[test]
    fn test_hardswish_zero_property(
        _a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let v = Vector::from_slice(&[0.0]);
        let result = v.hardswish().unwrap();

        prop_assert!(
            result.data[0].abs() < 1e-10,
            "hardswish(0) should be 0, got {}",
            result.data[0]
        );
    }
}

proptest! {
    /// Property test: For x >= 3, hardswish(x) = x (identity)
    #[test]
    fn test_hardswish_identity_large_positive(
        a in prop::collection::vec(3.0f32..100.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.hardswish().unwrap();

        for (i, &val) in a.iter().enumerate() {
            prop_assert!(
                (result.data[i] - val).abs() < 1e-5,
                "For x >= 3, hardswish(x) should = x, got {} vs {}",
                result.data[i], val
            );
        }
    }
}

proptest! {
    /// Property test: For x <= -3, hardswish(x) = 0
    #[test]
    fn test_hardswish_zero_large_negative(
        a in prop::collection::vec(-100.0f32..-3.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.hardswish().unwrap();

        for &val in result.as_slice() {
            prop_assert!(
                val.abs() < 1e-10,
                "For x <= -3, hardswish(x) should = 0, got {}",
                val
            );
        }
    }
}

proptest! {
    /// Property test: hardswish matches formula in transition region
    #[test]
    fn test_hardswish_transition_property(
        a in prop::collection::vec(-2.999f32..2.999, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.hardswish().unwrap();

        for (i, &x) in a.iter().enumerate() {
            let expected = x * (x + 3.0) / 6.0;
            prop_assert!(
                (result.data[i] - expected).abs() < 1e-5,
                "hardswish({}) should = {} * ({} + 3) / 6 = {}, got {}",
                x, x, x, expected, result.data[i]
            );
        }
    }
}

proptest! {
    /// Property test: mish() produces finite values
    #[test]
    fn test_mish_finite_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.mish().unwrap();

        for &val in result.as_slice() {
            prop_assert!(val.is_finite(), "Mish output should be finite");
        }
    }
}

proptest! {
    /// Property test: mish(0) ≈ 0 (mish(0) = 0 * tanh(softplus(0)) = 0)
    #[test]
    fn test_mish_zero_property(
        _a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let v = Vector::from_slice(&[0.0]);
        let result = v.mish().unwrap();

        prop_assert!(
            result.data[0].abs() < 1e-5,
            "mish(0) should be ≈ 0, got {}",
            result.data[0]
        );
    }
}

proptest! {
    /// Property test: For large positive x, mish(x) ≈ x (linear)
    #[test]
    fn test_mish_linear_large_positive(
        a in prop::collection::vec(20.0f32..100.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.mish().unwrap();

        for (i, &val) in a.iter().enumerate() {
            // For large positive values, mish(x) should be very close to x
            // since softplus(x) → x and tanh(x) → 1
            prop_assert!(
                (result.data[i] - val).abs() < 0.01,
                "For large positive {}, mish should ≈ x, got {} vs {}",
                val, result.data[i], val
            );
        }
    }
}

proptest! {
    /// Property test: For large negative x, mish(x) → 0
    #[test]
    fn test_mish_zero_large_negative(
        a in prop::collection::vec(-100.0f32..-20.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.mish().unwrap();

        for &val in result.as_slice() {
            prop_assert!(
                val.abs() < 1e-5,
                "For large negative x, mish(x) should → 0, got {}",
                val
            );
        }
    }
}

proptest! {
    /// Property test: mish has negative region (unlike ReLU)
    /// mish(x) can be slightly negative for x in (-1.5, 0)
    #[test]
    fn test_mish_negative_region_property(
        a in prop::collection::vec(-1.0f32..-0.1, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.mish().unwrap();

        for (i, &x) in a.iter().enumerate() {
            // Mish should produce negative values in this range
            // The minimum of mish is approximately -0.31 at x ≈ -1.07
            prop_assert!(
                result.data[i] < 0.0,
                "mish({}) should be negative in (-1, -0.1), got {}",
                x, result.data[i]
            );
        }
    }
}

proptest! {
    /// Property test: selu() produces finite values
    #[test]
    fn test_selu_finite_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.selu().unwrap();

        for &val in result.as_slice() {
            prop_assert!(val.is_finite(), "SELU output should be finite");
        }
    }
}

proptest! {
    /// Property test: selu(0) = 0 always
    #[test]
    fn test_selu_zero_property(
        _a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let v = Vector::from_slice(&[0.0]);
        let result = v.selu().unwrap();

        prop_assert!(
            result.data[0].abs() < 1e-10,
            "selu(0) should be 0, got {}",
            result.data[0]
        );
    }
}

proptest! {
    /// Property test: For positive x, selu(x) = λ * x (linear scaling)
    #[test]
    fn test_selu_linear_positive(
        a in prop::collection::vec(0.001f32..100.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.selu().unwrap();

        const LAMBDA: f32 = 1.0507009873554804934193349852946;

        for (i, &val) in a.iter().enumerate() {
            let expected = LAMBDA * val;
            prop_assert!(
                (result.data[i] - expected).abs() < 1e-4,
                "For positive {}, selu should = λ*x = {}, got {}",
                val, expected, result.data[i]
            );
        }
    }
}

proptest! {
    /// Property test: For large negative x, selu(x) → -λ * α ≈ -1.7581
    #[test]
    fn test_selu_asymptote_negative(
        a in prop::collection::vec(-100.0f32..-20.0, 1..50)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.selu().unwrap();

        const LAMBDA: f32 = 1.0507009873554804934193349852946;
        const ALPHA: f32 = 1.6732632423543772848170429916717;
        let asymptote = -LAMBDA * ALPHA;

        for &val in result.as_slice() {
            prop_assert!(
                (val - asymptote).abs() < 1e-3,
                "For large negative x, selu should → {}, got {}",
                asymptote, val
            );
        }
    }
}

proptest! {
    /// Property test: selu is monotonically increasing
    #[test]
    fn test_selu_monotonic_property(
        a in prop::collection::vec(-10.0f32..10.0, 2..50)
    ) {
        let mut sorted = a.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let va = Vector::from_slice(&sorted);
        let result = va.selu().unwrap();

        for i in 1..result.data.len() {
            prop_assert!(
                result.data[i] >= result.data[i-1] - 1e-5,
                "selu should be monotonic: selu({}) = {} >= selu({}) = {}",
                sorted[i], result.data[i], sorted[i-1], result.data[i-1]
            );
        }
    }
}
