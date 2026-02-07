use super::super::super::*;
use proptest::prelude::*;

// Property test: Dot product with self is non-negative (norm property)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_dot_self_nonnegative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.dot(&va).unwrap();

        // ||v||^2 = v·v >= 0 always
        prop_assert!(result >= 0.0);

        // If all zeros, should be exactly zero
        if a.iter().all(|&x| x == 0.0) {
            prop_assert_eq!(result, 0.0);
        } else {
            // If any non-zero element, result should be positive
            prop_assert!(result > 0.0);
        }
    }
}

// Property test: L2 norm is always non-negative
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_l2_nonnegative(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let norm = va.norm_l2().unwrap();

        // ||v|| >= 0 always
        prop_assert!(norm >= 0.0);

        // If all zeros, norm should be exactly zero
        if a.iter().all(|&x| x.abs() < 1e-6) {
            prop_assert!(norm < 1e-5);
        }
    }
}

// Property test: L2 norm equals sqrt(dot(a, a))
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_l2_equals_sqrt_dot(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let norm = va.norm_l2().unwrap();
        let dot_self = va.dot(&va).unwrap();

        // ||a|| = sqrt(a·a)
        // Use relative tolerance for large values
        let relative_error = if dot_self > 0.0 {
            ((norm * norm - dot_self) / dot_self).abs()
        } else {
            (norm * norm - dot_self).abs()
        };
        prop_assert!(relative_error < 1e-4 || (norm * norm - dot_self).abs() < 1e-2);
    }
}

// Property test: Scaling property ||c*a|| = |c| * ||a||
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_l2_scaling(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        c in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);
        let norm_a = va.norm_l2().unwrap();

        // Create c*a
        let scaled: Vec<f32> = a.iter().map(|&x| c * x).collect();
        let v_scaled = Vector::from_slice(&scaled);
        let norm_scaled = v_scaled.norm_l2().unwrap();

        // ||c*a|| = |c| * ||a||
        let expected = c.abs() * norm_a;
        prop_assert!((norm_scaled - expected).abs() < 1e-2);
    }
}

// Property test: Cauchy-Schwarz inequality |a·b| <= ||a|| * ||b||
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_cauchy_schwarz(
        a in prop::collection::vec(-100.0f32..100.0, 1..50),
        b in prop::collection::vec(-100.0f32..100.0, 1..50)
    ) {
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        let dot_ab = va.dot(&vb).unwrap().abs();
        let norm_a = va.dot(&va).unwrap().sqrt();
        let norm_b = vb.dot(&vb).unwrap().sqrt();

        // |a·b| <= ||a|| * ||b||
        // Add small tolerance for floating point
        prop_assert!(dot_ab <= norm_a * norm_b + 1e-3);
    }
}

// Property test: Scaling property (multiply all by same constant)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_scalar_multiplication(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        scalar in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);

        // Create vector of all same scalar
        let scalars = vec![scalar; a.len()];
        let vs = Vector::from_slice(&scalars);

        let result = va.mul(&vs).unwrap();

        // Each element should be a[i] * scalar
        for (i, &val) in result.as_slice().iter().enumerate() {
            let expected = a[i] * scalar;
            prop_assert!((val - expected).abs() < 1e-3);
        }
    }
}

// Property test: Sum of scaled vector = scale * sum
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sum_linearity(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        scalar in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);

        // Create scaled version
        let scalars = vec![scalar; a.len()];
        let vs = Vector::from_slice(&scalars);
        let scaled = va.mul(&vs).unwrap();

        let sum_scaled = scaled.sum().unwrap();
        let sum_original = va.sum().unwrap();

        // sum(scalar * v) = scalar * sum(v)
        let expected = scalar * sum_original;
        prop_assert!((sum_scaled - expected).abs() < 1e-2);
    }
}

// Property test: Normalized vector has unit norm
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_normalize_unit_norm(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        // Skip if vector is too close to zero (would cause division by zero)
        let norm_squared: f32 = a.iter().map(|x| x * x).sum();
        prop_assume!(norm_squared > 1e-6);

        let va = Vector::from_slice(&a);
        let normalized = va.normalize().unwrap();

        // The normalized vector should have L2 norm = 1
        let norm = normalized.norm_l2().unwrap();
        prop_assert!((norm - 1.0).abs() < 1e-4, "norm = {}, expected 1.0", norm);
    }
}

// Property test: Normalization preserves direction (scaling invariance)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_normalize_direction_invariant(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        scale in 0.1f32..10.0
    ) {
        // Skip if vector is too close to zero
        let norm_squared: f32 = a.iter().map(|x| x * x).sum();
        prop_assume!(norm_squared > 1e-6);

        let va = Vector::from_slice(&a);

        // Scale the vector
        let scales = vec![scale; a.len()];
        let vs = Vector::from_slice(&scales);
        let scaled = va.mul(&vs).unwrap();

        // Both should normalize to the same direction
        let norm_a = va.normalize().unwrap();
        let norm_scaled = scaled.normalize().unwrap();

        // Check each element is close
        for (i, (&val_a, &val_scaled)) in norm_a.as_slice().iter()
            .zip(norm_scaled.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (val_a - val_scaled).abs() < 1e-4,
                "Element {} differs: {} vs {}", i, val_a, val_scaled
            );
        }
    }
}

// Property test: L1 norm triangle inequality
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_l1_triangle_inequality(
        len in 1usize..100,
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        // Triangle inequality: ||a + b||₁ <= ||a||₁ + ||b||₁
        // Use same length for both vectors
        let actual_len = len.min(a.len()).min(b.len());
        let a_trimmed = &a[..actual_len];
        let b_trimmed = &b[..actual_len];

        let va = Vector::from_slice(a_trimmed);
        let vb = Vector::from_slice(b_trimmed);

        let norm_a = va.norm_l1().unwrap();
        let norm_b = vb.norm_l1().unwrap();
        let sum = va.add(&vb).unwrap();
        let norm_sum = sum.norm_l1().unwrap();

        // Triangle inequality should hold
        prop_assert!(
            norm_sum <= norm_a + norm_b + 1e-3,
            "Triangle inequality violated: {} > {} + {}",
            norm_sum, norm_a, norm_b
        );
    }
}

// Property test: L1 norm absolute homogeneity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_l1_absolute_homogeneity(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        scalar in -10.0f32..10.0
    ) {
        // Absolute homogeneity: ||c * v||₁ = |c| * ||v||₁
        let va = Vector::from_slice(&a);

        let norm_a = va.norm_l1().unwrap();

        // Scale the vector
        let scalars = vec![scalar; a.len()];
        let vs = Vector::from_slice(&scalars);
        let scaled = va.mul(&vs).unwrap();

        let norm_scaled = scaled.norm_l1().unwrap();

        // Should satisfy: ||c*v||₁ = |c| * ||v||₁
        let expected = scalar.abs() * norm_a;

        // Use relative tolerance for large values
        let tolerance = if expected.abs() > 1.0 {
            expected.abs() * 1e-5 // Relative tolerance
        } else {
            1e-2 // Absolute tolerance for small values
        };

        prop_assert!(
            (norm_scaled - expected).abs() < tolerance,
            "Homogeneity violated: {} != |{}| * {} = {}, diff = {}",
            norm_scaled, scalar, norm_a, expected, (norm_scaled - expected).abs()
        );
    }
}

// Property test: L1 norm equals sum of absolute values
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_l1_definition(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let norm = va.norm_l1().unwrap();

        // Manual calculation of sum(|a[i]|)
        let manual_sum: f32 = a.iter().map(|x| x.abs()).sum();

        // Use relative tolerance for large values
        let tolerance = if manual_sum.abs() > 1.0 {
            manual_sum.abs() * 1e-5 // Relative tolerance
        } else {
            1e-3 // Absolute tolerance for small values
        };

        prop_assert!(
            (norm - manual_sum).abs() < tolerance,
            "L1 norm {} != manual sum {}, diff = {}",
            norm, manual_sum, (norm - manual_sum).abs()
        );
    }
}

// Property test: L∞ norm absolute homogeneity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_linf_absolute_homogeneity(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        scalar in -10.0f32..10.0
    ) {
        prop_assume!(!a.is_empty());

        // Absolute homogeneity: ||c * v||∞ = |c| * ||v||∞
        let va = Vector::from_slice(&a);
        let norm_a = va.norm_linf().unwrap();

        // Scale the vector
        let scalars = vec![scalar; a.len()];
        let vs = Vector::from_slice(&scalars);
        let scaled = va.mul(&vs).unwrap();

        let norm_scaled = scaled.norm_linf().unwrap();

        // Should satisfy: ||c*v||∞ = |c| * ||v||∞
        let expected = scalar.abs() * norm_a;
        prop_assert!(
            (norm_scaled - expected).abs() < 1e-3,
            "Homogeneity violated: {} != |{}| * {} = {}",
            norm_scaled, scalar, norm_a, expected
        );
    }
}

// Property test: L∞ norm equals max of absolute values
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_linf_definition(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        prop_assume!(!a.is_empty());

        let va = Vector::from_slice(&a);
        let norm = va.norm_linf().unwrap();

        // Manual calculation of max(|a[i]|)
        let manual_max = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        prop_assert!(
            (norm - manual_max).abs() < 1e-5,
            "L∞ norm {} != manual max {}",
            norm, manual_max
        );
    }
}

// Property test: L∞ norm submultiplicativity (Hölder's inequality special case)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_norm_linf_submultiplicative(
        len in 1usize..100,
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        // For element-wise multiplication: ||a ⊙ b||∞ <= ||a||∞ * ||b||∞
        let actual_len = len.min(a.len()).min(b.len());
        let a_trimmed = &a[..actual_len];
        let b_trimmed = &b[..actual_len];

        let va = Vector::from_slice(a_trimmed);
        let vb = Vector::from_slice(b_trimmed);

        let norm_a = va.norm_linf().unwrap();
        let norm_b = vb.norm_linf().unwrap();
        let product = va.mul(&vb).unwrap();
        let norm_product = product.norm_linf().unwrap();

        // Submultiplicativity should hold
        prop_assert!(
            norm_product <= norm_a * norm_b + 1e-3,
            "Submultiplicativity violated: {} > {} * {}",
            norm_product, norm_a, norm_b
        );
    }
}

// Property test: abs() idempotence
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_abs_idempotent(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        // abs(abs(v)) = abs(v) - applying twice should be same as once
        let va = Vector::from_slice(&a);
        let abs_once = va.abs().unwrap();
        let abs_twice = abs_once.abs().unwrap();

        for (i, (&val_once, &val_twice)) in abs_once.as_slice().iter()
            .zip(abs_twice.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (val_once - val_twice).abs() < 1e-5,
                "Idempotence failed at {}: {} != {}",
                i, val_once, val_twice
            );
        }
    }
}

// Property test: abs() is always non-negative
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_abs_non_negative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.abs().unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            prop_assert!(
                val >= 0.0,
                "Negative value at {}: {}",
                i, val
            );
        }
    }
}

// Property test: abs() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_abs_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.abs().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.abs();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "Incorrect abs at {}: {} -> {}, expected {}",
                i, input, output, expected
            );
        }
    }
}

// Property test: scale() distributivity over addition
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_scale_distributive(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        scalar in -10.0f32..10.0
    ) {
        // scalar * (a + a) = (scalar * a) + (scalar * a)
        let va = Vector::from_slice(&a);
        let va_plus_va = va.add(&va).unwrap();
        let scaled_sum = va_plus_va.scale(scalar).unwrap();

        let scaled_a = va.scale(scalar).unwrap();
        let sum_of_scaled = scaled_a.add(&scaled_a).unwrap();

        for (i, (&val1, &val2)) in scaled_sum.as_slice().iter()
            .zip(sum_of_scaled.as_slice().iter())
            .enumerate() {
            let tolerance = if val1.abs() > 1.0 {
                val1.abs() * 1e-5
            } else {
                1e-3
            };
            prop_assert!(
                (val1 - val2).abs() < tolerance,
                "Distributivity failed at {}: {} != {}, diff = {}",
                i, val1, val2, (val1 - val2).abs()
            );
        }
    }
}

// Property test: scale() with 1.0 is identity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_scale_identity(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.scale(1.0).unwrap();

        for (i, (&original, &scaled)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (original - scaled).abs() < 1e-5,
                "Identity failed at {}: {} != {}",
                i, original, scaled
            );
        }
    }
}

// Property test: scale() with 0.0 gives zeros
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_scale_zero(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.scale(0.0).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            prop_assert!(
                val.abs() < 1e-10,
                "Zero scaling failed at {}: {} != 0.0",
                i, val
            );
        }
    }
}

// Property test: scale() associativity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_scale_associative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        scalar1 in -10.0f32..10.0,
        scalar2 in -10.0f32..10.0
    ) {
        // (a * s1) * s2 = a * (s1 * s2)
        let va = Vector::from_slice(&a);
        let scaled_once = va.scale(scalar1).unwrap();
        let scaled_twice = scaled_once.scale(scalar2).unwrap();

        let combined_scalar = scalar1 * scalar2;
        let scaled_combined = va.scale(combined_scalar).unwrap();

        for (i, (&val1, &val2)) in scaled_twice.as_slice().iter()
            .zip(scaled_combined.as_slice().iter())
            .enumerate() {
            let tolerance = if val1.abs() > 1.0 {
                val1.abs() * 1e-4  // Slightly higher tolerance for double scaling
            } else {
                1e-3
            };
            prop_assert!(
                (val1 - val2).abs() < tolerance,
                "Associativity failed at {}: {} != {}, diff = {}",
                i, val1, val2, (val1 - val2).abs()
            );
        }
    }
}

// Property test: clamp() bounds enforcement
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_clamp_bounds(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        min_val in -50.0f32..0.0,
        max_val in 0.0f32..50.0
    ) {
        let va = Vector::from_slice(&a);
        let result = va.clamp(min_val, max_val).unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            prop_assert!(
                val >= min_val && val <= max_val,
                "Value {} out of bounds [{}, {}] at index {}",
                val, min_val, max_val, i
            );
        }
    }
}

// Property test: clamp() idempotence
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_clamp_idempotent(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        min_val in -50.0f32..0.0,
        max_val in 0.0f32..50.0
    ) {
        // clamp(clamp(v)) = clamp(v)
        let va = Vector::from_slice(&a);
        let clamped_once = va.clamp(min_val, max_val).unwrap();
        let clamped_twice = clamped_once.clamp(min_val, max_val).unwrap();

        for (i, (&val1, &val2)) in clamped_once.as_slice().iter()
            .zip(clamped_twice.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (val1 - val2).abs() < 1e-10,
                "Idempotence failed at {}: {} != {}",
                i, val1, val2
            );
        }
    }
}

// Property test: clamp() monotonicity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_clamp_monotonic(
        a in prop::collection::vec(-100.0f32..100.0, 2..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.clamp(-50.0, 50.0).unwrap();

        // For any i < j, if a[i] <= a[j], then clamp(a[i]) <= clamp(a[j])
        for i in 0..a.len() - 1 {
            for j in i + 1..a.len() {
                if a[i] <= a[j] {
                    prop_assert!(
                        result.as_slice()[i] <= result.as_slice()[j],
                        "Monotonicity violated: a[{}]={} <= a[{}]={} but clamp[{}]={} > clamp[{}]={}",
                        i, a[i], j, a[j], i, result.as_slice()[i], j, result.as_slice()[j]
                    );
                }
            }
        }
    }
}

// Property test: lerp() at endpoints
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_lerp_endpoints(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let len = a.len().min(b.len());
        let a_trimmed = &a[..len];
        let b_trimmed = &b[..len];

        let va = Vector::from_slice(a_trimmed);
        let vb = Vector::from_slice(b_trimmed);

        // t=0 should return a
        let result_zero = va.lerp(&vb, 0.0).unwrap();
        for (i, (&actual, &expected)) in result_zero.as_slice().iter()
            .zip(a_trimmed.iter())
            .enumerate() {
            prop_assert!(
                (actual - expected).abs() < 1e-5,
                "lerp(t=0) failed at {}: {} != {}",
                i, actual, expected
            );
        }

        // t=1 should return b
        let result_one = va.lerp(&vb, 1.0).unwrap();
        for (i, (&actual, &expected)) in result_one.as_slice().iter()
            .zip(b_trimmed.iter())
            .enumerate() {
            prop_assert!(
                (actual - expected).abs() < 1e-5,
                "lerp(t=1) failed at {}: {} != {}",
                i, actual, expected
            );
        }
    }
}

// Property test: lerp() linearity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_lerp_linearity(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100),
        t in 0.0f32..1.0
    ) {
        let len = a.len().min(b.len());
        let a_trimmed = &a[..len];
        let b_trimmed = &b[..len];

        let va = Vector::from_slice(a_trimmed);
        let vb = Vector::from_slice(b_trimmed);

        let result = va.lerp(&vb, t).unwrap();

        // Verify: result[i] = a[i] + t * (b[i] - a[i])
        for (i, ((&a_val, &b_val), &result_val)) in a_trimmed.iter()
            .zip(b_trimmed.iter())
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val + t * (b_val - a_val);

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-5
            } else {
                1e-4
            };

            prop_assert!(
                (result_val - expected).abs() < tolerance,
                "Linearity failed at {}: {} != {}, diff = {}",
                i, result_val, expected, (result_val - expected).abs()
            );
        }
    }
}

// Property test: lerp() symmetry
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_lerp_symmetry(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100),
        t in 0.0f32..1.0
    ) {
        let len = a.len().min(b.len());
        let a_trimmed = &a[..len];
        let b_trimmed = &b[..len];

        let va = Vector::from_slice(a_trimmed);
        let vb = Vector::from_slice(b_trimmed);

        // lerp(a, b, t) should equal lerp(b, a, 1-t)
        let forward = va.lerp(&vb, t).unwrap();
        let reverse = vb.lerp(&va, 1.0 - t).unwrap();

        for (i, (&fwd, &rev)) in forward.as_slice().iter()
            .zip(reverse.as_slice().iter())
            .enumerate() {
            let tolerance = if fwd.abs() > 1.0 {
                fwd.abs() * 2e-5
            } else {
                1e-4
            };

            prop_assert!(
                (fwd - rev).abs() < tolerance,
                "Symmetry failed at {}: {} != {}, diff = {}",
                i, fwd, rev, (fwd - rev).abs()
            );
        }
    }
}

// Property test: fma() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_fma_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100),
        c in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let len = a.len().min(b.len()).min(c.len());
        let a_trimmed = &a[..len];
        let b_trimmed = &b[..len];
        let c_trimmed = &c[..len];

        let va = Vector::from_slice(a_trimmed);
        let vb = Vector::from_slice(b_trimmed);
        let vc = Vector::from_slice(c_trimmed);

        let result = va.fma(&vb, &vc).unwrap();

        // Verify: result[i] = a[i] * b[i] + c[i]
        for (i, ((&a_val, &b_val), (&c_val, &result_val))) in a_trimmed.iter()
            .zip(b_trimmed.iter())
            .zip(c_trimmed.iter().zip(result.as_slice().iter()))
            .enumerate() {
            let expected = a_val * b_val + c_val;

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-5
            } else {
                1e-4
            };

            prop_assert!(
                (result_val - expected).abs() < tolerance,
                "FMA correctness failed at {}: {} != {}, diff = {}",
                i, result_val, expected, (result_val - expected).abs()
            );
        }
    }
}

// Property test: fma() with zero multiplication
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_fma_zero_mul(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        c in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        // fma(a, 0, c) = 0 * a + c = c
        let len = a.len().min(c.len());
        let a_trimmed = &a[..len];
        let c_trimmed = &c[..len];

        let va = Vector::from_slice(a_trimmed);
        let vc = Vector::from_slice(c_trimmed);
        let zeros = vec![0.0; len];
        let vzero = Vector::from_slice(&zeros);

        let result = va.fma(&vzero, &vc).unwrap();

        for (i, (&result_val, &c_val)) in result.as_slice().iter()
            .zip(c_trimmed.iter())
            .enumerate() {
            prop_assert!(
                (result_val - c_val).abs() < 1e-10,
                "Zero multiplication failed at {}: {} != {}, diff = {}",
                i, result_val, c_val, (result_val - c_val).abs()
            );
        }
    }
}

// Property test: fma() relation to mul and add
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_fma_vs_mul_add(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100),
        c in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let len = a.len().min(b.len()).min(c.len());
        let a_trimmed = &a[..len];
        let b_trimmed = &b[..len];
        let c_trimmed = &c[..len];

        let va = Vector::from_slice(a_trimmed);
        let vb = Vector::from_slice(b_trimmed);
        let vc = Vector::from_slice(c_trimmed);

        // fma(a, b, c) should approximately equal mul(a, b) + c
        let fma_result = va.fma(&vb, &vc).unwrap();
        let mul_result = va.mul(&vb).unwrap();
        let add_result = mul_result.add(&vc).unwrap();

        for (i, (&fma_val, &add_val)) in fma_result.as_slice().iter()
            .zip(add_result.as_slice().iter())
            .enumerate() {
            // FMA can have better accuracy, so use slightly higher tolerance
            let tolerance = if fma_val.abs() > 1.0 {
                fma_val.abs() * 1e-5
            } else {
                1e-4
            };

            prop_assert!(
                (fma_val - add_val).abs() < tolerance,
                "FMA vs mul+add failed at {}: {} != {}, diff = {}",
                i, fma_val, add_val, (fma_val - add_val).abs()
            );
        }
    }
}
