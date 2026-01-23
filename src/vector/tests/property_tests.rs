use super::super::*;
use proptest::prelude::*;

// Property test: Addition is commutative (a + b == b + a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_add_commutative(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100),
        b in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        // Use minimum length to ensure both vectors have same size
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        let result1 = va.add(&vb).unwrap();
        let result2 = vb.add(&va).unwrap();

        prop_assert_eq!(result1.as_slice(), result2.as_slice());
    }
}

// Property test: Addition is associative ((a + b) + c == a + (b + c))
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_add_associative(
        a in prop::collection::vec(-100.0f32..100.0, 1..50),
        b in prop::collection::vec(-100.0f32..100.0, 1..50),
        c in prop::collection::vec(-100.0f32..100.0, 1..50)
    ) {
        let len = a.len().min(b.len()).min(c.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();
        let c_vec: Vec<f32> = c.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);
        let vc = Vector::from_slice(&c_vec);

        let ab = va.add(&vb).unwrap();
        let abc = ab.add(&vc).unwrap();

        let bc = vb.add(&vc).unwrap();
        let a_bc = va.add(&bc).unwrap();

        // Use approximate equality for floating point (relaxed for associativity)
        for (x, y) in abc.as_slice().iter().zip(a_bc.as_slice()) {
            prop_assert!((x - y).abs() < 1e-4);
        }
    }
}

// Property test: Subtraction anti-commutativity (a - b == -(b - a))
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sub_anti_commutative(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100),
        b in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        let result1 = va.sub(&vb).unwrap();
        let result2 = vb.sub(&va).unwrap();

        // a - b should equal -(b - a)
        for (x, y) in result1.as_slice().iter().zip(result2.as_slice()) {
            prop_assert!((x + y).abs() < 1e-5);
        }
    }
}

// Property test: Subtraction identity (a - 0 == a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sub_identity(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let zero = Vector::from_slice(&vec![0.0; a.len()]);

        let result = va.sub(&zero).unwrap();

        prop_assert_eq!(result.as_slice(), va.as_slice());
    }
}

// Property test: Subtraction inverse (a - a == 0)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sub_inverse(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);

        let result = va.sub(&va).unwrap();

        // All elements should be zero (or very close due to floating point)
        for &x in result.as_slice() {
            prop_assert!(x.abs() < 1e-5);
        }
    }
}

// Property test: Multiplication is commutative
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_mul_commutative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        let result1 = va.mul(&vb).unwrap();
        let result2 = vb.mul(&va).unwrap();

        prop_assert_eq!(result1.as_slice(), result2.as_slice());
    }
}

// Property test: Division identity (a / 1 == a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_div_identity(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let ones = Vector::from_slice(&vec![1.0; a.len()]);

        let result = va.div(&ones).unwrap();

        for (x, y) in result.as_slice().iter().zip(va.as_slice()) {
            prop_assert!((x - y).abs() < 1e-5);
        }
    }
}

// Property test: Division inverse (a / a == 1, for non-zero a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_div_inverse(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        // Filter out zeros to avoid division edge cases
        let a_nonzero: Vec<f32> = a.into_iter()
            .map(|x| if x.abs() < 1e-5 { 1.0 } else { x })
            .collect();

        let va = Vector::from_slice(&a_nonzero);
        let result = va.div(&va).unwrap();

        // All elements should be 1.0 (or very close)
        for &x in result.as_slice() {
            prop_assert!((x - 1.0).abs() < 1e-4);
        }
    }
}

// Property test: Division-multiplication inverse (a / b) * b ≈ a
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_div_mul_inverse(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100),
        b in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();

        // Filter out zeros from b to avoid division by zero edge cases
        let b_vec: Vec<f32> = b.into_iter().take(len)
            .map(|x| if x.abs() < 1e-3 { 1.0 } else { x })
            .collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        let divided = va.div(&vb).unwrap();
        let restored = divided.mul(&vb).unwrap();

        // Restored should approximately equal original
        for (original, restored_val) in a_vec.iter().zip(restored.as_slice()) {
            prop_assert!((original - restored_val).abs() < 1e-2);
        }
    }
}

// Property test: Dot product is commutative
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_dot_commutative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        b in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        let result1 = va.dot(&vb).unwrap();
        let result2 = vb.dot(&va).unwrap();

        prop_assert!((result1 - result2).abs() < 1e-3);
    }
}

// Property test: Identity element for addition (a + 0 == a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_add_identity(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let zero = Vector::from_slice(&vec![0.0; a.len()]);

        let result = va.add(&zero).unwrap();

        prop_assert_eq!(result.as_slice(), va.as_slice());
    }
}

// Property test: Identity element for multiplication (a * 1 == a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_mul_identity(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let one = Vector::from_slice(&vec![1.0; a.len()]);

        let result = va.mul(&one).unwrap();

        for (x, y) in result.as_slice().iter().zip(va.as_slice()) {
            prop_assert!((x - y).abs() < 1e-5);
        }
    }
}

// Property test: Zero element for multiplication (a * 0 == 0)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_mul_zero(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let zero = Vector::from_slice(&vec![0.0; a.len()]);

        let result = va.mul(&zero).unwrap();

        for x in result.as_slice() {
            prop_assert_eq!(*x, 0.0);
        }
    }
}

// Property test: Distributive property (a * (b + c) == a * b + a * c)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_distributive(
        a in prop::collection::vec(-10.0f32..10.0, 1..50),
        b in prop::collection::vec(-10.0f32..10.0, 1..50),
        c in prop::collection::vec(-10.0f32..10.0, 1..50)
    ) {
        let len = a.len().min(b.len()).min(c.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();
        let c_vec: Vec<f32> = c.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);
        let vc = Vector::from_slice(&c_vec);

        // a * (b + c)
        let bc = vb.add(&vc).unwrap();
        let left = va.mul(&bc).unwrap();

        // a * b + a * c
        let ab = va.mul(&vb).unwrap();
        let ac = va.mul(&vc).unwrap();
        let right = ab.add(&ac).unwrap();

        for (x, y) in left.as_slice().iter().zip(right.as_slice()) {
            prop_assert!((x - y).abs() < 1e-3);
        }
    }
}

// Property test: Sum is consistent
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sum_matches_manual(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sum().unwrap();
        let manual_sum: f32 = a.iter().sum();

        // Relaxed tolerance for SIMD vs scalar accumulation order differences
        prop_assert!((result - manual_sum).abs() < 1e-2);
    }

    #[test]
    fn test_sum_kahan_correctness(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let kahan_result = va.sum_kahan().unwrap();
        let manual_sum: f32 = a.iter().sum();

        // Kahan result should be close to manual sum
        // Note: Both use same algorithm (iter().sum() also uses compensated summation)
        // so they should match closely
        prop_assert!((kahan_result - manual_sum).abs() < 1e-2,
            "Kahan sum should match manual sum closely");

        // Verify Kahan produces a reasonable result
        let expected_magnitude = a.iter().map(|x| x.abs()).sum::<f32>();
        prop_assert!(kahan_result.abs() <= expected_magnitude + 1.0,
            "Kahan result magnitude should be reasonable");
    }
}

// Property test: Max is actually maximum
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_max_is_maximum(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.max().unwrap();

        // Verify result is >= all elements
        for &x in a.iter() {
            prop_assert!(result >= x);
        }

        // Verify result is actually in the vector
        prop_assert!(a.contains(&result));
    }

    #[test]
    fn test_min_is_minimum(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.min().unwrap();

        // Verify result is <= all elements
        for &x in a.iter() {
            prop_assert!(result <= x);
        }

        // Verify result is actually in the vector
        prop_assert!(a.contains(&result));
    }

    #[test]
    fn test_argmax_correctness(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let idx = va.argmax().unwrap();

        // Verify index is in bounds
        prop_assert!(idx < a.len());

        // Verify value at index is >= all other values
        let max_val = a[idx];
        for &x in a.iter() {
            prop_assert!(max_val >= x);
        }

        // Verify it's the first occurrence (no earlier index has this value)
        for &val in a.iter().take(idx) {
            prop_assert!(val < max_val || val != max_val);
        }
    }

    #[test]
    fn test_argmin_correctness(
        a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let idx = va.argmin().unwrap();

        // Verify index is in bounds
        prop_assert!(idx < a.len());

        // Verify value at index is <= all other values
        let min_val = a[idx];
        for &x in a.iter() {
            prop_assert!(min_val <= x);
        }

        // Verify it's the first occurrence (no earlier index has this value)
        for &val in a.iter().take(idx) {
            prop_assert!(val > min_val || val != min_val);
        }
    }
}

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

// Property test: sqrt() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sqrt_correctness(
        a in prop::collection::vec(0.0f32..100.0, 1..100)  // Non-negative values only
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sqrt().unwrap();

        // Verify: result[i] = sqrt(a[i])
        for (i, (&a_val, &result_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.sqrt();

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-6
            } else {
                1e-6
            };

            prop_assert!(
                (result_val - expected).abs() < tolerance,
                "sqrt correctness failed at {}: {} != {}, diff = {}",
                i, result_val, expected, (result_val - expected).abs()
            );
        }
    }
}

// Property test: sqrt() idempotence with squaring
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sqrt_inverse_square(
        a in prop::collection::vec(0.0f32..100.0, 1..100)
    ) {
        // sqrt(a)^2 = a
        let va = Vector::from_slice(&a);
        let sqrt_result = va.sqrt().unwrap();
        let squared = sqrt_result.mul(&sqrt_result).unwrap();

        for (i, (&original, &recovered)) in a.iter()
            .zip(squared.as_slice().iter())
            .enumerate() {
            let tolerance = if original.abs() > 1.0 {
                original.abs() * 1e-5
            } else {
                1e-5
            };

            prop_assert!(
                (original - recovered).abs() < tolerance,
                "sqrt inverse failed at {}: {} != {}, diff = {}",
                i, original, recovered, (original - recovered).abs()
            );
        }
    }
}

// Property test: sqrt() monotonicity
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sqrt_monotonic(
        a in prop::collection::vec(0.0f32..100.0, 2..100)
    ) {
        // If a[i] < a[j], then sqrt(a[i]) <= sqrt(a[j])
        let va = Vector::from_slice(&a);
        let result = va.sqrt().unwrap();
        let result_slice = result.as_slice();

        for i in 0..a.len()-1 {
            for j in i+1..a.len() {
                // Use a small epsilon to account for f32 precision
                let epsilon = 1e-6;
                if a[i] + epsilon < a[j] {
                    prop_assert!(
                        result_slice[i] <= result_slice[j],
                        "Monotonicity failed: sqrt({}) = {} should be <= sqrt({}) = {}",
                        a[i], result_slice[i], a[j], result_slice[j]
                    );
                } else if a[i] > a[j] + epsilon {
                    prop_assert!(
                        result_slice[i] >= result_slice[j],
                        "Monotonicity failed: sqrt({}) = {} should be >= sqrt({}) = {}",
                        a[i], result_slice[i], a[j], result_slice[j]
                    );
                }
            }
        }
    }
}

// Property test: recip() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_recip_correctness(
        a in prop::collection::vec(0.1f32..100.0, 1..100)  // Avoid zeros and very small values
    ) {
        let va = Vector::from_slice(&a);
        let result = va.recip().unwrap();

        // Verify: result[i] = 1 / a[i]
        for (i, (&a_val, &result_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = 1.0 / a_val;

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-6
            } else {
                1e-6
            };

            prop_assert!(
                (result_val - expected).abs() < tolerance,
                "recip correctness failed at {}: {} != {}, diff = {}",
                i, result_val, expected, (result_val - expected).abs()
            );
        }
    }
}

// Property test: recip() inverse property
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_recip_inverse(
        a in prop::collection::vec(0.1f32..100.0, 1..100)
    ) {
        // recip(recip(a)) = a
        let va = Vector::from_slice(&a);
        let recip_once = va.recip().unwrap();
        let recip_twice = recip_once.recip().unwrap();

        for (i, (&original, &recovered)) in a.iter()
            .zip(recip_twice.as_slice().iter())
            .enumerate() {
            let tolerance = if original.abs() > 1.0 {
                original.abs() * 1e-5
            } else {
                1e-5
            };

            prop_assert!(
                (original - recovered).abs() < tolerance,
                "recip inverse failed at {}: {} != {}, diff = {}",
                i, original, recovered, (original - recovered).abs()
            );
        }
    }
}

// Property test: recip() relation to division
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_recip_vs_division(
        a in prop::collection::vec(0.1f32..100.0, 1..100),
        scalar in 0.1f32..100.0
    ) {
        // scalar * recip(a) should equal scalar / a
        let va = Vector::from_slice(&a);
        let recip_result = va.recip().unwrap();
        let scaled = recip_result.scale(scalar).unwrap();

        for (i, (&a_val, &scaled_val)) in a.iter()
            .zip(scaled.as_slice().iter())
            .enumerate() {
            let expected = scalar / a_val;

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-5
            } else {
                1e-5
            };

            prop_assert!(
                (scaled_val - expected).abs() < tolerance,
                "recip vs division failed at {}: {} != {}, diff = {}",
                i, scaled_val, expected, (scaled_val - expected).abs()
            );
        }
    }
}

// Property test: pow() correctness vs f32::powf()
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_pow_correctness(
        a in prop::collection::vec(0.1f32..100.0, 1..100),
        n in -3.0f32..3.0
    ) {
        let va = Vector::from_slice(&a);
        let result = va.pow(n).unwrap();

        for (i, (&a_val, &pow_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.powf(n);

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-4
            } else {
                1e-4
            };

            prop_assert!(
                (pow_val - expected).abs() < tolerance,
                "pow correctness failed at {}: {} != {}, diff = {}",
                i, pow_val, expected, (pow_val - expected).abs()
            );
        }
    }
}

// Property test: Power laws
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_pow_power_laws(
        a in prop::collection::vec(1.0f32..10.0, 1..50),
        n in 1.0f32..3.0,
        m in 1.0f32..3.0
    ) {
        // Test: (x^n)^m = x^(n*m)
        let va = Vector::from_slice(&a);
        let pow_n = va.pow(n).unwrap();
        let pow_n_then_m = pow_n.pow(m).unwrap();
        let pow_nm = va.pow(n * m).unwrap();

        for (i, (&expected, &actual)) in pow_nm.as_slice().iter()
            .zip(pow_n_then_m.as_slice().iter())
            .enumerate() {
            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-3
            } else {
                1e-3
            };

            prop_assert!(
                (expected - actual).abs() < tolerance,
                "pow power law failed at {}: {} != {}, diff = {}",
                i, expected, actual, (expected - actual).abs()
            );
        }
    }
}

// Property test: pow() special cases
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_pow_special_cases(
        a in prop::collection::vec(0.1f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);

        // x^0 = 1
        let pow_zero = va.pow(0.0).unwrap();
        for &val in pow_zero.as_slice() {
            prop_assert!((val - 1.0).abs() < 1e-5, "x^0 should be 1");
        }

        // x^1 = x
        let pow_one = va.pow(1.0).unwrap();
        for (i, (&original, &pow_val)) in a.iter()
            .zip(pow_one.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (original - pow_val).abs() < 1e-5,
                "x^1 failed at {}: {} != {}",
                i, original, pow_val
            );
        }

        // x^0.5 should equal sqrt(x)
        let pow_half = va.pow(0.5).unwrap();
        let sqrt_result = va.sqrt().unwrap();
        for (i, (&pow_val, &sqrt_val)) in pow_half.as_slice().iter()
            .zip(sqrt_result.as_slice().iter())
            .enumerate() {
            let tolerance = if sqrt_val.abs() > 1.0 {
                sqrt_val.abs() * 1e-5
            } else {
                1e-5
            };
            prop_assert!(
                (pow_val - sqrt_val).abs() < tolerance,
                "x^0.5 vs sqrt failed at {}: {} != {}",
                i, pow_val, sqrt_val
            );
        }
    }
}

// Property test: exp() correctness vs f32::exp()
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_exp_correctness(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.exp().unwrap();

        for (i, (&a_val, &exp_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.exp();

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-5
            } else {
                1e-5
            };

            prop_assert!(
                (exp_val - expected).abs() < tolerance,
                "exp correctness failed at {}: {} != {}, diff = {}",
                i, exp_val, expected, (exp_val - expected).abs()
            );
        }
    }
}

// Property test: exp() identity - exp(0) = 1
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_exp_zero_identity(
        len in 1usize..100
    ) {
        let zeros = vec![0.0f32; len];
        let va = Vector::from_slice(&zeros);
        let result = va.exp().unwrap();

        for (i, &val) in result.as_slice().iter().enumerate() {
            prop_assert!(
                (val - 1.0).abs() < 1e-5,
                "exp(0) identity failed at {}: {} != 1.0",
                i, val
            );
        }
    }
}

// Property test: exp() relation to addition - exp(a+b) = exp(a) * exp(b)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_exp_addition_property(
        a in prop::collection::vec(-5.0f32..5.0, 1..50),
        b in prop::collection::vec(-5.0f32..5.0, 1..50)
    ) {
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        // exp(a + b)
        let sum = va.add(&vb).unwrap();
        let exp_sum = sum.exp().unwrap();

        // exp(a) * exp(b)
        let exp_a = va.exp().unwrap();
        let exp_b = vb.exp().unwrap();
        let product = exp_a.mul(&exp_b).unwrap();

        for (i, (&exp_sum_val, &product_val)) in exp_sum.as_slice().iter()
            .zip(product.as_slice().iter())
            .enumerate() {
            let tolerance = if exp_sum_val.abs() > 1.0 {
                exp_sum_val.abs() * 1e-4
            } else {
                1e-4
            };

            prop_assert!(
                (exp_sum_val - product_val).abs() < tolerance,
                "exp(a+b) = exp(a)*exp(b) failed at {}: {} != {}, diff = {}",
                i, exp_sum_val, product_val, (exp_sum_val - product_val).abs()
            );
        }
    }
}

// Property test: ln() correctness vs f32::ln()
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_ln_correctness(
        a in prop::collection::vec(0.1f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.ln().unwrap();

        for (i, (&a_val, &ln_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.ln();

            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-5
            } else {
                1e-5
            };

            prop_assert!(
                (ln_val - expected).abs() < tolerance,
                "ln correctness failed at {}: {} != {}, diff = {}",
                i, ln_val, expected, (ln_val - expected).abs()
            );
        }
    }
}

// Property test: ln() inverse of exp() - ln(exp(x)) = x
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_ln_inverse_exp_property(
        a in prop::collection::vec(-5.0f32..5.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let exp_result = va.exp().unwrap();
        let ln_result = exp_result.ln().unwrap();

        for (i, (&original, &recovered)) in a.iter()
            .zip(ln_result.as_slice().iter())
            .enumerate() {
            let tolerance = if original.abs() > 1.0 {
                original.abs() * 1e-4
            } else {
                1e-4
            };

            prop_assert!(
                (original - recovered).abs() < tolerance,
                "ln(exp(x)) != x at {}: {} != {}, diff = {}",
                i, original, recovered, (original - recovered).abs()
            );
        }
    }
}

// Property test: ln() product rule - ln(a*b) = ln(a) + ln(b)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_ln_product_rule(
        a in prop::collection::vec(0.1f32..10.0, 1..50),
        b in prop::collection::vec(0.1f32..10.0, 1..50)
    ) {
        let len = a.len().min(b.len());
        let a_vec: Vec<f32> = a.into_iter().take(len).collect();
        let b_vec: Vec<f32> = b.into_iter().take(len).collect();

        let va = Vector::from_slice(&a_vec);
        let vb = Vector::from_slice(&b_vec);

        // ln(a * b)
        let product = va.mul(&vb).unwrap();
        let ln_product = product.ln().unwrap();

        // ln(a) + ln(b)
        let ln_a = va.ln().unwrap();
        let ln_b = vb.ln().unwrap();
        let sum = ln_a.add(&ln_b).unwrap();

        for (i, (&ln_prod_val, &sum_val)) in ln_product.as_slice().iter()
            .zip(sum.as_slice().iter())
            .enumerate() {
            let tolerance = if ln_prod_val.abs() > 1.0 {
                ln_prod_val.abs() * 1e-4
            } else {
                1e-4
            };

            prop_assert!(
                (ln_prod_val - sum_val).abs() < tolerance,
                "ln(a*b) = ln(a)+ln(b) failed at {}: {} != {}, diff = {}",
                i, ln_prod_val, sum_val, (ln_prod_val - sum_val).abs()
            );
        }
    }
}

// Property test: log2() correctness vs f32::log2()
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_log2_correctness(
        a in prop::collection::vec(0.001f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.log2().unwrap();

        for (i, (&a_val, &log2_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.log2();

            prop_assert!(
                (log2_val - expected).abs() < 1e-4,
                "log2 correctness failed at {}: {} != {}, diff = {}",
                i, log2_val, expected, (log2_val - expected).abs()
            );
        }
    }
}

// Property test: log2(2^n) = n (power of 2 property)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn test_log2_power_property(
        n in prop::collection::vec(-10i32..10, 1..50)
    ) {
        let powers: Vec<f32> = n.iter().map(|&exp| 2.0f32.powi(exp)).collect();
        let va = Vector::from_slice(&powers);
        let result = va.log2().unwrap();

        for (&exp_val, &log2_val) in n.iter()
            .zip(result.as_slice().iter()) {
            let expected = exp_val as f32;

            prop_assert!(
                (log2_val - expected).abs() < 1e-4,
                "log2(2^{}) should be {}, got {}, diff = {}",
                exp_val, expected, log2_val, (log2_val - expected).abs()
            );
        }
    }
}

// Property test: log10() correctness vs f32::log10()
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_log10_correctness(
        a in prop::collection::vec(0.001f32..1000.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.log10().unwrap();

        for (i, (&a_val, &log10_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.log10();

            prop_assert!(
                (log10_val - expected).abs() < 1e-4,
                "log10 correctness failed at {}: {} != {}, diff = {}",
                i, log10_val, expected, (log10_val - expected).abs()
            );
        }
    }
}

// Property test: log10(10^n) = n (power of 10 property)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn test_log10_power_property(
        n in prop::collection::vec(-3i32..6, 1..30)
    ) {
        let powers: Vec<f32> = n.iter().map(|&exp| 10.0f32.powi(exp)).collect();
        let va = Vector::from_slice(&powers);
        let result = va.log10().unwrap();

        for (&exp_val, &log10_val) in n.iter()
            .zip(result.as_slice().iter()) {
            let expected = exp_val as f32;

            prop_assert!(
                (log10_val - expected).abs() < 1e-3,
                "log10(10^{}) should be {}, got {}, diff = {}",
                exp_val, expected, log10_val, (log10_val - expected).abs()
            );
        }
    }
}

// Property test: sin() correctness vs f32::sin()
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sin_correctness(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sin().unwrap();

        for (i, (&a_val, &sin_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.sin();

            prop_assert!(
                (sin_val - expected).abs() < 1e-5,
                "sin correctness failed at {}: {} != {}, diff = {}",
                i, sin_val, expected, (sin_val - expected).abs()
            );
        }
    }
}

// Property test: sin() range [-1, 1]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sin_range(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sin().unwrap();

        for (i, &sin_val) in result.as_slice().iter().enumerate() {
            prop_assert!(
                (-1.0..=1.0).contains(&sin_val),
                "sin range failed at {}: {} not in [-1, 1]",
                i, sin_val
            );
        }
    }
}

// Property test: sin() periodicity - sin(x + 2π) = sin(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sin_periodicity_property(
        a in prop::collection::vec(-5.0f32..5.0, 1..50)
    ) {
        use std::f32::consts::PI;

        let va = Vector::from_slice(&a);
        let sin_a = va.sin().unwrap();

        // Add 2π to each element
        let a_plus_2pi: Vec<f32> = a.iter().map(|&x| x + 2.0 * PI).collect();
        let va_plus_2pi = Vector::from_slice(&a_plus_2pi);
        let sin_a_plus_2pi = va_plus_2pi.sin().unwrap();

        for (i, (&sin_val, &sin_periodic_val)) in sin_a.as_slice().iter()
            .zip(sin_a_plus_2pi.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (sin_val - sin_periodic_val).abs() < 1e-5,
                "sin periodicity failed at {}: {} != {}, diff = {}",
                i, sin_val, sin_periodic_val, (sin_val - sin_periodic_val).abs()
            );
        }
    }
}

// Property test: cos() correctness vs f32::cos()
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_cos_correctness(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.cos().unwrap();

        for (i, (&a_val, &cos_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.cos();

            prop_assert!(
                (cos_val - expected).abs() < 1e-5,
                "cos correctness failed at {}: {} != {}, diff = {}",
                i, cos_val, expected, (cos_val - expected).abs()
            );
        }
    }
}

// Property test: cos() range [-1, 1]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_cos_range(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.cos().unwrap();

        for (i, &cos_val) in result.as_slice().iter().enumerate() {
            prop_assert!(
                (-1.0..=1.0).contains(&cos_val),
                "cos range failed at {}: {} not in [-1, 1]",
                i, cos_val
            );
        }
    }
}

// Property test: Pythagorean identity - sin²(x) + cos²(x) = 1
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_pythagorean_identity(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let sin_result = va.sin().unwrap();
        let cos_result = va.cos().unwrap();

        for (i, (&sin_val, &cos_val)) in sin_result.as_slice().iter()
            .zip(cos_result.as_slice().iter())
            .enumerate() {
            let sum_of_squares = sin_val * sin_val + cos_val * cos_val;

            prop_assert!(
                (sum_of_squares - 1.0).abs() < 1e-5,
                "Pythagorean identity failed at {}: sin²+cos² = {} != 1.0",
                i, sum_of_squares
            );
        }
    }
}

// Property test: tan() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_tan_correctness(
        a in prop::collection::vec(-1.5f32..1.5, 1..100)
    ) {
        // Use limited range to avoid tan asymptotes at ±π/2
        let va = Vector::from_slice(&a);
        let result = va.tan().unwrap();

        for (i, (&a_val, &tan_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.tan();

            prop_assert!(
                (tan_val - expected).abs() < 1e-5,
                "tan correctness failed at {}: {} != {}, diff = {}",
                i, tan_val, expected, (tan_val - expected).abs()
            );
        }
    }
}

// Property test: tan(x) = sin(x) / cos(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_tan_sin_cos_identity(
        a in prop::collection::vec(-1.5f32..1.5, 1..100)
    ) {
        // Avoid asymptotes at ±π/2 where cos(x) ≈ 0
        let va = Vector::from_slice(&a);
        let tan_result = va.tan().unwrap();
        let sin_result = va.sin().unwrap();
        let cos_result = va.cos().unwrap();

        for (i, ((&tan_val, &sin_val), &cos_val)) in tan_result.as_slice().iter()
            .zip(sin_result.as_slice().iter())
            .zip(cos_result.as_slice().iter())
            .enumerate() {
            // Skip values where cos is very small (near asymptote)
            if cos_val.abs() > 1e-3 {
                let expected = sin_val / cos_val;
                prop_assert!(
                    (tan_val - expected).abs() < 1e-4,
                    "tan(x) != sin(x)/cos(x) at {}: {} != {}, cos={}",
                    i, tan_val, expected, cos_val
                );
            }
        }
    }
}

// Property test: tan is odd function - tan(-x) = -tan(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_tan_odd_function(
        a in prop::collection::vec(-1.5f32..1.5, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let tan_pos = va.tan().unwrap();

        let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let tan_neg = va_neg.tan().unwrap();

        for (i, (&pos, &neg)) in tan_pos.as_slice().iter()
            .zip(tan_neg.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (pos + neg).abs() < 1e-5,
                "tan odd function failed at {}: tan(-x)={} != -tan(x)={}",
                i, neg, -pos
            );
        }
    }
}

// Property test: asin() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_asin_correctness(
        a in prop::collection::vec(-1.0f32..1.0, 1..100)
    ) {
        // Domain is [-1, 1] for asin
        let va = Vector::from_slice(&a);
        let result = va.asin().unwrap();

        for (i, (&a_val, &asin_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.asin();

            prop_assert!(
                (asin_val - expected).abs() < 1e-5,
                "asin correctness failed at {}: {} != {}, diff = {}",
                i, asin_val, expected, (asin_val - expected).abs()
            );
        }
    }
}

// Property test: asin(sin(x)) = x for x in [-π/2, π/2]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_asin_sin_inverse(
        a in prop::collection::vec(-1.5f32..1.5, 1..100)
    ) {
        // Test range within [-π/2, π/2] to ensure inverse property
        let va = Vector::from_slice(&a);
        let sin_result = va.sin().unwrap();
        let asin_result = sin_result.asin().unwrap();

        for (i, (&original, &reconstructed)) in a.iter()
            .zip(asin_result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (original - reconstructed).abs() < 1e-5,
                "asin(sin(x)) != x at {}: {} != {}",
                i, reconstructed, original
            );
        }
    }
}

// Property test: asin is odd function - asin(-x) = -asin(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_asin_odd_function(
        a in prop::collection::vec(-1.0f32..1.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let asin_pos = va.asin().unwrap();

        let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let asin_neg = va_neg.asin().unwrap();

        for (i, (&pos, &neg)) in asin_pos.as_slice().iter()
            .zip(asin_neg.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (pos + neg).abs() < 1e-5,
                "asin odd function failed at {}: asin(-x)={} != -asin(x)={}",
                i, neg, -pos
            );
        }
    }
}

// Property test: acos() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_acos_correctness(
        a in prop::collection::vec(-1.0f32..1.0, 1..100)
    ) {
        // Domain is [-1, 1] for acos
        let va = Vector::from_slice(&a);
        let result = va.acos().unwrap();

        for (i, (&a_val, &acos_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.acos();

            prop_assert!(
                (acos_val - expected).abs() < 1e-5,
                "acos correctness failed at {}: {} != {}, diff = {}",
                i, acos_val, expected, (acos_val - expected).abs()
            );
        }
    }
}

// Property test: acos(cos(x)) = x for x in [0, π]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_acos_cos_inverse(
        a in prop::collection::vec(0.0f32..std::f32::consts::PI, 1..100)
    ) {
        // Test range within [0, π] to ensure inverse property
        let va = Vector::from_slice(&a);
        let cos_result = va.cos().unwrap();
        let acos_result = cos_result.acos().unwrap();

        for (i, (&original, &reconstructed)) in a.iter()
            .zip(acos_result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (original - reconstructed).abs() < 3e-4,
                "acos(cos(x)) != x at {}: {} != {}",
                i, reconstructed, original
            );
        }
    }
}

// Property test: acos symmetry - acos(-x) = π - acos(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_acos_symmetry(
        a in prop::collection::vec(-1.0f32..1.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let acos_pos = va.acos().unwrap();

        let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let acos_neg = va_neg.acos().unwrap();

        for (i, (&pos, &neg)) in acos_pos.as_slice().iter()
            .zip(acos_neg.as_slice().iter())
            .enumerate() {
            let expected = std::f32::consts::PI - pos;
            prop_assert!(
                (neg - expected).abs() < 1e-5,
                "acos symmetry failed at {}: acos(-x)={} != π - acos(x)={}",
                i, neg, expected
            );
        }
    }
}

// Property test: atan() correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_atan_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        // atan accepts all real numbers
        let va = Vector::from_slice(&a);
        let result = va.atan().unwrap();

        for (i, (&a_val, &atan_val)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = a_val.atan();

            prop_assert!(
                (atan_val - expected).abs() < 1e-5,
                "atan correctness failed at {}: {} != {}, diff = {}",
                i, atan_val, expected, (atan_val - expected).abs()
            );
        }
    }
}

// Property test: atan(tan(x)) = x for x in (-π/2, π/2)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_atan_tan_inverse(
        a in prop::collection::vec(-1.5f32..1.5, 1..100)
    ) {
        // Test range within (-π/2, π/2) to ensure inverse property
        let va = Vector::from_slice(&a);
        let tan_result = va.tan().unwrap();
        let atan_result = tan_result.atan().unwrap();

        for (i, (&original, &reconstructed)) in a.iter()
            .zip(atan_result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (original - reconstructed).abs() < 1e-5,
                "atan(tan(x)) != x at {}: {} != {}",
                i, reconstructed, original
            );
        }
    }
}

// Property test: atan is odd function - atan(-x) = -atan(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_atan_odd_function(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let atan_pos = va.atan().unwrap();

        let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let atan_neg = va_neg.atan().unwrap();

        for (i, (&pos, &neg)) in atan_pos.as_slice().iter()
            .zip(atan_neg.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (pos + neg).abs() < 1e-5,
                "atan odd function failed at {}: atan(-x)={} != -atan(x)={}",
                i, neg, -pos
            );
        }
    }
}

// Property test: sinh correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sinh_correctness(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sinh().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.sinh();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "sinh failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: sinh is odd function - sinh(-x) = -sinh(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sinh_odd_function(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let sinh_pos = va.sinh().unwrap();

        let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let sinh_neg = va_neg.sinh().unwrap();

        for (i, (&pos, &neg)) in sinh_pos.as_slice().iter()
            .zip(sinh_neg.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (pos + neg).abs() < 1e-4,
                "sinh odd function failed at {}: sinh(-x)={} != -sinh(x)={}",
                i, neg, -pos
            );
        }
    }
}

// Property test: sinh definition - sinh(x) = (e^x - e^(-x)) / 2
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sinh_definition(
        a in prop::collection::vec(-5.0f32..5.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sinh().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = (input.exp() - (-input).exp()) / 2.0;
            // Use relative tolerance for larger values
            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-5
            } else {
                1e-5
            };
            prop_assert!(
                (output - expected).abs() < tolerance,
                "sinh definition failed at {}: {} != {} (input: {})",
                i, output, expected, input
            );
        }
    }
}

// Property test: cosh correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_cosh_correctness(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.cosh().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.cosh();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "cosh failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: cosh is even function - cosh(-x) = cosh(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_cosh_even_function(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let cosh_pos = va.cosh().unwrap();

        let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let cosh_neg = va_neg.cosh().unwrap();

        for (i, (&pos, &neg)) in cosh_pos.as_slice().iter()
            .zip(cosh_neg.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (pos - neg).abs() < 1e-4,
                "cosh even function failed at {}: cosh(-x)={} != cosh(x)={}",
                i, neg, pos
            );
        }
    }
}

// Property test: cosh definition - cosh(x) = (e^x + e^(-x)) / 2
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_cosh_definition(
        a in prop::collection::vec(-5.0f32..5.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.cosh().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = (input.exp() + (-input).exp()) / 2.0;
            // Use relative tolerance for larger values
            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-5
            } else {
                1e-5
            };
            prop_assert!(
                (output - expected).abs() < tolerance,
                "cosh definition failed at {}: {} != {} (input: {})",
                i, output, expected, input
            );
        }
    }
}

// Property test: hyperbolic identity - cosh^2(x) - sinh^2(x) = 1
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_cosh_sinh_identity(
        a in prop::collection::vec(-5.0f32..5.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let cosh_result = va.cosh().unwrap();
        let sinh_result = va.sinh().unwrap();

        for (i, (&c, &s)) in cosh_result.as_slice().iter()
            .zip(sinh_result.as_slice().iter())
            .enumerate() {
            let identity = c * c - s * s;
            // Use relative tolerance for numerical stability
            // Since we're computing c^2 - s^2, tolerance scales with squared values
            let max_squared = c.abs().max(s.abs()).powi(2);
            let tolerance = if max_squared > 1.0 {
                max_squared * 1e-4
            } else {
                1e-5
            };
            prop_assert!(
                (identity - 1.0).abs() < tolerance,
                "Hyperbolic identity failed at {}: cosh^2({}) - sinh^2({}) = {} != 1",
                i, c, s, identity
            );
        }
    }
}

// Property test: tanh correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_tanh_correctness(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.tanh().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.tanh();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "tanh failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: tanh is odd function - tanh(-x) = -tanh(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_tanh_odd_function(
        a in prop::collection::vec(-10.0f32..10.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let tanh_pos = va.tanh().unwrap();

        let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let tanh_neg = va_neg.tanh().unwrap();

        for (i, (&pos, &neg)) in tanh_pos.as_slice().iter()
            .zip(tanh_neg.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (pos + neg).abs() < 1e-5,
                "tanh odd function failed at {}: tanh(-x)={} != -tanh(x)={}",
                i, neg, -pos
            );
        }
    }
}

// Property test: tanh = sinh/cosh relation
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_tanh_sinh_cosh_relation(
        a in prop::collection::vec(-5.0f32..5.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let tanh_result = va.tanh().unwrap();
        let sinh_result = va.sinh().unwrap();
        let cosh_result = va.cosh().unwrap();

        for (i, (&t, (&s, &c))) in tanh_result.as_slice().iter()
            .zip(sinh_result.as_slice().iter().zip(cosh_result.as_slice().iter()))
            .enumerate() {
            let ratio = s / c;
            // Use relative tolerance for numerical stability
            let tolerance = if ratio.abs() > 1.0 {
                ratio.abs() * 1e-5
            } else {
                1e-5
            };
            prop_assert!(
                (t - ratio).abs() < tolerance,
                "tanh = sinh/cosh failed at {}: tanh({}) = {} != {}/{}={}",
                i, t, t, s, c, ratio
            );
        }
    }
}

// Property test: tanh range bound -1 <= tanh(x) <= 1
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_tanh_range_bound(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.tanh().unwrap();

        for (i, &output) in result.as_slice().iter().enumerate() {
            prop_assert!(
                (-1.0..=1.0).contains(&output),
                "tanh range bound failed at {}: {} not in [-1, 1]",
                i, output
            );
        }
    }
}

// Property test: asinh correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_asinh_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.asinh().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.asinh();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "asinh failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: asinh is odd function - asinh(-x) = -asinh(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_asinh_odd_function(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let asinh_pos = va.asinh().unwrap();

        let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
        let va_neg = Vector::from_slice(&a_neg);
        let asinh_neg = va_neg.asinh().unwrap();

        for (i, (&pos, &neg)) in asinh_pos.as_slice().iter()
            .zip(asinh_neg.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (pos + neg).abs() < 1e-5,
                "asinh odd function failed at {}: asinh(-x)={} != -asinh(x)={}",
                i, neg, -pos
            );
        }
    }
}

// Property test: asinh(sinh(x)) = x inverse relation
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_asinh_sinh_inverse(
        a in prop::collection::vec(-5.0f32..5.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let sinh_result = va.sinh().unwrap();
        let asinh_result = sinh_result.asinh().unwrap();

        for (i, (&original, &reconstructed)) in a.iter()
            .zip(asinh_result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (original - reconstructed).abs() < 1e-5,
                "asinh(sinh(x)) != x at {}: {} != {}",
                i, reconstructed, original
            );
        }
    }
}

// Property test: acosh correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_acosh_correctness(
        a in prop::collection::vec(1.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.acosh().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.acosh();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "acosh failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: acosh(cosh(x)) = x inverse relation for x >= 0
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_acosh_cosh_inverse(
        a in prop::collection::vec(0.1f32..5.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let cosh_result = va.cosh().unwrap();
        let acosh_result = cosh_result.acosh().unwrap();

        for (i, (&original, &reconstructed)) in a.iter()
            .zip(acosh_result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (original - reconstructed).abs() < 1e-5,
                "acosh(cosh(x)) != x at {}: {} != {}",
                i, reconstructed, original
            );
        }
    }
}

// Property test: acosh range - output >= 0
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_acosh_range(
        a in prop::collection::vec(1.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.acosh().unwrap();

        for (i, &output) in result.as_slice().iter().enumerate() {
            prop_assert!(
                output >= 0.0,
                "acosh range failed at {}: {} not >= 0",
                i, output
            );
        }
    }
}

// Property test: atanh correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_atanh_correctness(
        a in prop::collection::vec(-0.99f32..0.99, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.atanh().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.atanh();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "atanh failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: atanh is odd function: atanh(-x) = -atanh(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_atanh_odd_function(
        a in prop::collection::vec(-0.99f32..0.99, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let neg_a: Vec<f32> = a.iter().map(|x| -x).collect();
        let v_neg_a = Vector::from_slice(&neg_a);

        let result_a = va.atanh().unwrap();
        let result_neg_a = v_neg_a.atanh().unwrap();

        for (i, (&res_a, &res_neg_a)) in result_a.as_slice().iter()
            .zip(result_neg_a.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (res_a + res_neg_a).abs() < 1e-5,
                "atanh(-x) != -atanh(x) at {}: {} != {}",
                i, res_neg_a, -res_a
            );
        }
    }
}

// Property test: atanh(tanh(x)) = x inverse relation
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_atanh_tanh_inverse(
        a in prop::collection::vec(-3.5f32..3.5, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let tanh_result = va.tanh().unwrap();
        let atanh_result = tanh_result.atanh().unwrap();

        for (i, (&original, &reconstructed)) in a.iter()
            .zip(atanh_result.as_slice().iter())
            .enumerate() {
            // Use conservative tolerance: tanh saturates near ±1 causing precision loss
            // Limit test range to [-3.5, 3.5] where tanh is well-conditioned
            let tolerance = 1e-4;
            prop_assert!(
                (original - reconstructed).abs() < tolerance,
                "atanh(tanh(x)) != x at {}: {} != {}",
                i, reconstructed, original
            );
        }
    }
}

// Property test: floor correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_floor_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.floor().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.floor();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "floor failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: floor idempotence - floor(floor(x)) = floor(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_floor_idempotence(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let floor_once = va.floor().unwrap();
        let floor_twice = floor_once.floor().unwrap();

        for (i, (&once, &twice)) in floor_once.as_slice().iter()
            .zip(floor_twice.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (once - twice).abs() < 1e-5,
                "floor idempotence failed at {}: floor(floor({})) = {} != {}",
                i, a[i], twice, once
            );
        }
    }
}

// Property test: floor always <= original value
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_floor_less_than_or_equal(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.floor().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                output <= input,
                "floor should be <= input at {}: floor({}) = {} > {}",
                i, input, output, input
            );
        }
    }
}

// Property test: ceil correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_ceil_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.ceil().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.ceil();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "ceil failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: ceil idempotence - ceil(ceil(x)) = ceil(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_ceil_idempotence(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let ceil_once = va.ceil().unwrap();
        let ceil_twice = ceil_once.ceil().unwrap();

        for (i, (&once, &twice)) in ceil_once.as_slice().iter()
            .zip(ceil_twice.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (once - twice).abs() < 1e-5,
                "ceil idempotence failed at {}: ceil(ceil({})) = {} != {}",
                i, a[i], twice, once
            );
        }
    }
}

// Property test: ceil always >= original value
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_ceil_greater_than_or_equal(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.ceil().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                output >= input,
                "ceil should be >= input at {}: ceil({}) = {} < {}",
                i, input, output, input
            );
        }
    }
}

// Property test: round correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_round_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.round().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.round();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "round failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: round idempotence - round(round(x)) = round(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_round_idempotence(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let round_once = va.round().unwrap();
        let round_twice = round_once.round().unwrap();

        for (i, (&once, &twice)) in round_once.as_slice().iter()
            .zip(round_twice.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (once - twice).abs() < 1e-5,
                "round idempotence failed at {}: round(round({})) = {} != {}",
                i, a[i], twice, once
            );
        }
    }
}

// Property test: round distance - |round(x) - x| <= 0.5
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_round_distance(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.round().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let distance = (output - input).abs();
            prop_assert!(
                distance <= 0.5 + 1e-5,  // Small epsilon for floating point precision
                "round distance should be <= 0.5 at {}: |round({}) - {}| = {} > 0.5",
                i, input, input, distance
            );
        }
    }
}

// Property test: trunc correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_trunc_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.trunc().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.trunc();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "trunc failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: trunc idempotence - trunc(trunc(x)) = trunc(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_trunc_idempotence(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let trunc_once = va.trunc().unwrap();
        let trunc_twice = trunc_once.trunc().unwrap();

        for (i, (&once, &twice)) in trunc_once.as_slice().iter()
            .zip(trunc_twice.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (once - twice).abs() < 1e-5,
                "trunc idempotence failed at {}: trunc(trunc({})) = {} != {}",
                i, a[i], twice, once
            );
        }
    }
}

// Property test: trunc moves toward zero - |trunc(x)| <= |x|
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_trunc_toward_zero(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.trunc().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                output.abs() <= input.abs() + 1e-5,  // Small epsilon for floating point
                "trunc should move toward zero at {}: |trunc({})| = {} > |{}| = {}",
                i, input, output.abs(), input, input.abs()
            );
        }
    }
}

// Property test: fract correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_fract_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.fract().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.fract();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "fract failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: fract decomposition - x = trunc(x) + fract(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_fract_decomposition(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let trunc_result = va.trunc().unwrap();
        let fract_result = va.fract().unwrap();

        for (i, (&input, (&t, &f))) in a.iter()
            .zip(trunc_result.as_slice().iter().zip(fract_result.as_slice().iter()))
            .enumerate() {
            let reconstructed = t + f;
            prop_assert!(
                (reconstructed - input).abs() < 1e-5,
                "decomposition failed at {}: {} != trunc({}) + fract({}) = {} + {} = {}",
                i, input, input, input, t, f, reconstructed
            );
        }
    }
}

// Property test: fract magnitude - |fract(x)| < 1
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_fract_magnitude(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.fract().unwrap();

        for (i, &output) in result.as_slice().iter().enumerate() {
            prop_assert!(
                output.abs() < 1.0,
                "fract magnitude should be < 1 at {}: |fract({})| = {} >= 1",
                i, a[i], output.abs()
            );
        }
    }
}

// Property test: signum correctness
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_signum_correctness(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.signum().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            let expected = input.signum();
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "signum failed at {}: {} != {}",
                i, output, expected
            );
        }
    }
}

// Property test: signum range - always -1, 0, or 1
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_signum_range(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.signum().unwrap();

        for (i, &output) in result.as_slice().iter().enumerate() {
            prop_assert!(
                output == 1.0 || output == -1.0 || output.is_nan(),
                "signum should be 1.0, -1.0, or NaN at {}: signum({}) = {}",
                i, a[i], output
            );
        }
    }
}

// Property test: signum * abs = identity (for non-zero)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_signum_abs_identity(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let signum_result = va.signum().unwrap();
        let abs_result = va.abs().unwrap();

        for (i, (&input, (&sign, &magnitude))) in a.iter()
            .zip(signum_result.as_slice().iter().zip(abs_result.as_slice().iter()))
            .enumerate() {
            // Skip zero values as they have special behavior
            if input.abs() > 1e-10 {
                let reconstructed = sign * magnitude;
                prop_assert!(
                    (reconstructed - input).abs() < 1e-5,
                    "signum*abs identity failed at {}: {} != signum({}) * abs({}) = {} * {} = {}",
                    i, input, input, input, sign, magnitude, reconstructed
                );
            }
        }
    }
}

// ========================================
// Property Tests: copysign()
// ========================================

// Property test: copysign correctness - matches f32::copysign
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_copysign_correctness(
        ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let result = va.copysign(&vb).unwrap();

        for (i, (&mag, (&sgn, &output))) in a.iter()
            .zip(b.iter().zip(result.as_slice().iter()))
            .enumerate() {
            let expected = mag.copysign(sgn);
            prop_assert!(
                (output - expected).abs() < 1e-5 || (output.is_nan() && expected.is_nan()),
                "copysign failed at {}: copysign({}, {}) = {} != {}",
                i, mag, sgn, output, expected
            );
        }
    }
}

// Property test: magnitude preservation - abs(copysign(a, b)) = abs(a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_copysign_magnitude_preservation(
        ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let result = va.copysign(&vb).unwrap();
        let abs_a = va.abs().unwrap();
        let abs_result = result.abs().unwrap();

        for (i, (&expected, &output)) in abs_a.as_slice().iter()
            .zip(abs_result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "magnitude not preserved at {}: abs(copysign({}, {})) = {} != abs({}) = {}",
                i, a[i], b[i], output, a[i], expected
            );
        }
    }
}

// Property test: sign copy - sign(copysign(a, b)) = sign(b) for non-zero b
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_copysign_sign_copy(
        ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let result = va.copysign(&vb).unwrap();
        let signum_b = vb.signum().unwrap();
        let signum_result = result.signum().unwrap();

        for (i, (&sign_b, &sign_result)) in signum_b.as_slice().iter()
            .zip(signum_result.as_slice().iter())
            .enumerate() {
            // Skip NaN cases
            if !sign_b.is_nan() && !sign_result.is_nan() {
                prop_assert!(
                    (sign_result - sign_b).abs() < 1e-5,
                    "sign not copied at {}: sign(copysign({}, {})) = {} != sign({}) = {}",
                    i, a[i], b[i], sign_result, b[i], sign_b
                );
            }
        }
    }
}

// ========================================
// Property Tests: minimum()
// ========================================

// Property test: minimum correctness - matches f32::min
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_minimum_correctness(
        ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let result = va.minimum(&vb).unwrap();

        for (i, (&x, (&y, &output))) in a.iter()
            .zip(b.iter().zip(result.as_slice().iter()))
            .enumerate() {
            let expected = x.min(y);
            prop_assert!(
                (output - expected).abs() < 1e-5 || (output.is_nan() && expected.is_nan()),
                "minimum failed at {}: minimum({}, {}) = {} != {}",
                i, x, y, output, expected
            );
        }
    }
}

// Property test: commutativity - minimum(a, b) = minimum(b, a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_minimum_commutative(
        ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let result1 = va.minimum(&vb).unwrap();
        let result2 = vb.minimum(&va).unwrap();

        for (i, (&r1, &r2)) in result1.as_slice().iter()
            .zip(result2.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (r1 - r2).abs() < 1e-5 || (r1.is_nan() && r2.is_nan()),
                "commutativity failed at {}: minimum({}, {}) = {} != minimum({}, {}) = {}",
                i, a[i], b[i], r1, b[i], a[i], r2
            );
        }
    }
}

// Property test: idempotence - minimum(a, a) = a
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_minimum_idempotent(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.minimum(&va).unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (output - input).abs() < 1e-5 || (output.is_nan() && input.is_nan()),
                "idempotence failed at {}: minimum({}, {}) = {} != {}",
                i, input, input, output, input
            );
        }
    }
}

// ========================================
// Property Tests: maximum()
// ========================================

// Property test: maximum correctness - matches f32::max
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_maximum_correctness(
        ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let result = va.maximum(&vb).unwrap();

        for (i, (&x, (&y, &output))) in a.iter()
            .zip(b.iter().zip(result.as_slice().iter()))
            .enumerate() {
            let expected = x.max(y);
            prop_assert!(
                (output - expected).abs() < 1e-5 || (output.is_nan() && expected.is_nan()),
                "maximum failed at {}: maximum({}, {}) = {} != {}",
                i, x, y, output, expected
            );
        }
    }
}

// Property test: commutativity - maximum(a, b) = maximum(b, a)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_maximum_commutative(
        ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let result1 = va.maximum(&vb).unwrap();
        let result2 = vb.maximum(&va).unwrap();

        for (i, (&r1, &r2)) in result1.as_slice().iter()
            .zip(result2.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (r1 - r2).abs() < 1e-5 || (r1.is_nan() && r2.is_nan()),
                "commutativity failed at {}: maximum({}, {}) = {} != maximum({}, {}) = {}",
                i, a[i], b[i], r1, b[i], a[i], r2
            );
        }
    }
}

// Property test: idempotence - maximum(a, a) = a
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_maximum_idempotent(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.maximum(&va).unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(result.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (output - input).abs() < 1e-5 || (output.is_nan() && input.is_nan()),
                "idempotence failed at {}: maximum({}, {}) = {} != {}",
                i, input, input, output, input
            );
        }
    }
}

// ========================================
// Property Tests: neg()
// ========================================

// Property test: double negation is identity - -(-x) = x
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_neg_double_negation_property(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let neg_once = va.neg().unwrap();
        let neg_twice = neg_once.neg().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(neg_twice.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (output - input).abs() < 1e-5 || (output.is_nan() && input.is_nan()),
                "double negation failed at {}: -(-{}) = {} != {}",
                i, input, output, input
            );
        }
    }
}

// Property test: negation sign flip - sign(neg(x)) = -sign(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_neg_sign_flip(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let neg_result = va.neg().unwrap();

        for (i, (&input, &output)) in a.iter()
            .zip(neg_result.as_slice().iter())
            .enumerate() {
            // Skip zero and NaN
            if input.abs() > 1e-10 && !input.is_nan() {
                prop_assert!(
                    (input.signum() + output.signum()).abs() < 1e-5,
                    "sign flip failed at {}: sign({}) + sign(-{}) = {} + {} != 0",
                    i, input, input, input.signum(), output.signum()
                );
            }
        }
    }
}

// Property test: negation preserves magnitude - abs(neg(x)) = abs(x)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_neg_magnitude_preservation(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let neg_result = va.neg().unwrap();
        let abs_a = va.abs().unwrap();
        let abs_neg_a = neg_result.abs().unwrap();

        for (i, (&expected, &output)) in abs_a.as_slice().iter()
            .zip(abs_neg_a.as_slice().iter())
            .enumerate() {
            prop_assert!(
                (output - expected).abs() < 1e-5,
                "magnitude not preserved at {}: abs(-{}) = {} != abs({}) = {}",
                i, a[i], output, a[i], expected
            );
        }
    }
}

// ========================================
// Property Tests: sum_of_squares()
// ========================================

// Property test: non-negativity - sum_of_squares is always >= 0
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sum_of_squares_non_negative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let result = va.sum_of_squares().unwrap();

        prop_assert!(
            result >= 0.0,
            "sum_of_squares should be non-negative: {} < 0",
            result
        );
    }
}

// Property test: equivalence with dot product - sum_of_squares(v) = dot(v, v)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sum_of_squares_equals_dot_self(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let sum_sq = va.sum_of_squares().unwrap();
        let dot_self = va.dot(&va).unwrap();

        prop_assert!(
            (sum_sq - dot_self).abs() < 1e-4,
            "sum_of_squares should equal dot(self, self): {} != {}",
            sum_sq, dot_self
        );
    }
}

// Property test: scaling - sum_of_squares(k*v) = k^2 * sum_of_squares(v)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_sum_of_squares_scaling(
        a in prop::collection::vec(-10.0f32..10.0, 1..50),
        k in -5.0f32..5.0
    ) {
        let va = Vector::from_slice(&a);
        let scaled = va.scale(k).unwrap();

        let sum_sq_original = va.sum_of_squares().unwrap();
        let sum_sq_scaled = scaled.sum_of_squares().unwrap();
        let expected = k * k * sum_sq_original;

        // Use relative tolerance for larger values
        let tolerance = 1e-3 * expected.abs().max(1.0);
        prop_assert!(
            (sum_sq_scaled - expected).abs() < tolerance,
            "sum_of_squares({} * v) = {} != {}^2 * {} = {}",
            k, sum_sq_scaled, k, sum_sq_original, expected
        );
    }

    /// Property test: mean(v) is between min(v) and max(v)
    #[test]
    fn test_mean_bounds(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let mean_val = va.mean().unwrap();
        let min_val = va.min().unwrap();
        let max_val = va.max().unwrap();

        prop_assert!(
            mean_val >= min_val && mean_val <= max_val,
            "mean({}) = {} not in range [{}, {}]",
            mean_val, mean_val, min_val, max_val
        );
    }

    /// Property test: mean(v + c) = mean(v) + c (translation property)
    #[test]
    fn test_mean_translation(
        a in prop::collection::vec(-50.0f32..50.0, 1..100),
        c in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);
        let mean_original = va.mean().unwrap();

        // Create translated vector: v + c
        let translated: Vec<f32> = a.iter().map(|x| x + c).collect();
        let vt = Vector::from_slice(&translated);
        let mean_translated = vt.mean().unwrap();

        let expected = mean_original + c;
        let tolerance = 1e-4 * expected.abs().max(1.0);
        prop_assert!(
            (mean_translated - expected).abs() < tolerance,
            "mean(v + {}) = {} != mean(v) + {} = {}",
            c, mean_translated, c, expected
        );
    }

    /// Property test: mean(k*v) = k*mean(v) (scaling property)
    #[test]
    fn test_mean_scaling(
        a in prop::collection::vec(-50.0f32..50.0, 1..100),
        k in -5.0f32..5.0
    ) {
        let va = Vector::from_slice(&a);
        let mean_original = va.mean().unwrap();
        let scaled = va.scale(k).unwrap();
        let mean_scaled = scaled.mean().unwrap();

        let expected = k * mean_original;
        let tolerance = 1e-4 * expected.abs().max(1.0);
        prop_assert!(
            (mean_scaled - expected).abs() < tolerance,
            "mean({} * v) = {} != {} * mean(v) = {}",
            k, mean_scaled, k, expected
        );
    }

    /// Property test: variance(v) >= 0 (non-negativity)
    #[test]
    fn test_variance_non_negative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let var = va.variance().unwrap();

        prop_assert!(
            var >= -1e-5, // Allow small numerical error
            "variance = {} should be non-negative",
            var
        );
    }

    /// Property test: variance(k*v) = k²*variance(v) (scaling property)
    #[test]
    fn test_variance_scaling(
        a in prop::collection::vec(-50.0f32..50.0, 1..100),
        k in -5.0f32..5.0
    ) {
        let va = Vector::from_slice(&a);
        let var_original = va.variance().unwrap();
        let scaled = va.scale(k).unwrap();
        let var_scaled = scaled.variance().unwrap();

        let expected = k * k * var_original;
        // Use absolute tolerance for small values, relative for large
        // Note: f32 has ~7 significant digits; variance involves squaring which
        // doubles relative error, then k² multiplies it again. Use 0.5% tolerance.
        let tolerance = if expected.abs() < 1.0 {
            1e-2  // Absolute tolerance for small variance
        } else {
            5e-3 * expected.abs()  // 0.5% relative tolerance for large variance
        };
        prop_assert!(
            (var_scaled - expected).abs() < tolerance,
            "variance({} * v) = {} != {}² * variance(v) = {}",
            k, var_scaled, k, expected
        );
    }

    /// Property test: variance(v + c) = variance(v) (translation invariance)
    #[test]
    fn test_variance_translation_invariance(
        a in prop::collection::vec(-50.0f32..50.0, 1..100),
        c in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);
        let var_original = va.variance().unwrap();

        // Create translated vector: v + c
        let translated: Vec<f32> = a.iter().map(|x| x + c).collect();
        let vt = Vector::from_slice(&translated);
        let var_translated = vt.variance().unwrap();

        // Use absolute tolerance for small values, relative for large
        let tolerance = if var_original.abs() < 1.0 {
            1e-2  // Absolute tolerance for small variance
        } else {
            1e-3 * var_original.abs()  // Relative tolerance for large variance
        };
        prop_assert!(
            (var_translated - var_original).abs() < tolerance,
            "variance(v + {}) = {} != variance(v) = {}",
            c, var_translated, var_original
        );
    }

    /// Property test: stddev(v) >= 0 (non-negativity)
    #[test]
    fn test_stddev_non_negative(
        a in prop::collection::vec(-100.0f32..100.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let sd = va.stddev().unwrap();

        prop_assert!(
            sd >= -1e-5, // Allow small numerical error
            "stddev = {} should be non-negative",
            sd
        );
    }

    /// Property test: stddev(k*v) = |k|*stddev(v) (linear scaling)
    #[test]
    fn test_stddev_scaling(
        a in prop::collection::vec(-50.0f32..50.0, 1..100),
        k in -5.0f32..5.0
    ) {
        let va = Vector::from_slice(&a);
        let sd_original = va.stddev().unwrap();
        let scaled = va.scale(k).unwrap();
        let sd_scaled = scaled.stddev().unwrap();

        let expected = k.abs() * sd_original;
        // Use absolute tolerance for small values, relative for large
        // Note: stddev is numerically unstable for very similar values due to
        // catastrophic cancellation in variance computation
        let tolerance = if expected.abs() < 1.0 {
            5e-2  // Absolute tolerance for small stddev (increased for numerical stability)
        } else {
            1e-3 * expected.abs()  // Relative tolerance for large stddev
        };
        prop_assert!(
            (sd_scaled - expected).abs() < tolerance,
            "stddev({} * v) = {} != |{}| * stddev(v) = {}",
            k, sd_scaled, k, expected
        );
    }

    /// Property test: stddev(v + c) = stddev(v) (translation invariance)
    #[test]
    fn test_stddev_translation_invariance(
        a in prop::collection::vec(-50.0f32..50.0, 1..100),
        c in -10.0f32..10.0
    ) {
        let va = Vector::from_slice(&a);
        let sd_original = va.stddev().unwrap();

        // Create translated vector: v + c
        let translated: Vec<f32> = a.iter().map(|x| x + c).collect();
        let vt = Vector::from_slice(&translated);
        let sd_translated = vt.stddev().unwrap();

        // Use absolute tolerance for small values, relative for large
        let tolerance = if sd_original.abs() < 1.0 {
            1e-2  // Absolute tolerance for small stddev
        } else {
            1e-3 * sd_original.abs()  // Relative tolerance for large stddev
        };
        prop_assert!(
            (sd_translated - sd_original).abs() < tolerance,
            "stddev(v + {}) = {} != stddev(v) = {}",
            c, sd_translated, sd_original
        );
    }

    /// Property test: Cov(X,X) = Var(X) (covariance with self equals variance)
    #[test]
    fn test_covariance_self_equals_variance(
        a in prop::collection::vec(-50.0f32..50.0, 1..100)
    ) {
        let va = Vector::from_slice(&a);
        let cov = va.covariance(&va).unwrap();
        let var = va.variance().unwrap();

        let tolerance = 1e-3 * var.abs().max(1e-5);
        prop_assert!(
            (cov - var).abs() < tolerance,
            "Cov(X,X) = {} != Var(X) = {}",
            cov, var
        );
    }

    /// Property test: Cov(X,Y) = Cov(Y,X) (symmetry/commutativity)
    #[test]
    fn test_covariance_symmetric(
        ab in prop::collection::vec((-50.0f32..50.0, -50.0f32..50.0), 1..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();
        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);

        let cov_ab = va.covariance(&vb).unwrap();
        let cov_ba = vb.covariance(&va).unwrap();

        let tolerance = 1e-4 * cov_ab.abs().max(1e-5);
        prop_assert!(
            (cov_ab - cov_ba).abs() < tolerance,
            "Cov(X,Y) = {} != Cov(Y,X) = {}",
            cov_ab, cov_ba
        );
    }

    /// Property test: Cov(aX, bY) = ab*Cov(X,Y) (bilinearity)
    #[test]
    fn test_covariance_bilinearity(
        ab in prop::collection::vec((-20.0f32..20.0, -20.0f32..20.0), 1..50),
        scale_a in -3.0f32..3.0,
        scale_b in -3.0f32..3.0
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();
        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);

        let cov_original = va.covariance(&vb).unwrap();

        let scaled_a = va.scale(scale_a).unwrap();
        let scaled_b = vb.scale(scale_b).unwrap();
        let cov_scaled = scaled_a.covariance(&scaled_b).unwrap();

        let expected = scale_a * scale_b * cov_original;
        // Use relative tolerance accounting for compounding floating-point errors
        // Small covariances need larger relative tolerance due to precision limits
        let tolerance = 0.5 * expected.abs().max(1e-3);
        prop_assert!(
            (cov_scaled - expected).abs() < tolerance,
            "Cov({}*X, {}*Y) = {} != {}*{}*Cov(X,Y) = {}",
            scale_a, scale_b, cov_scaled, scale_a, scale_b, expected
        );
    }

    /// Property test: -1 ≤ ρ(X,Y) ≤ 1 (correlation is bounded)
    #[test]
    fn test_correlation_bounded(
        ab in prop::collection::vec((-50.0f32..50.0, -50.0f32..50.0), 2..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        // Ensure vectors are not constant
        let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                       - (a.iter().sum::<f32>() / a.len() as f32).powi(2);
        let std_b: f32 = b.iter().map(|y| y * y).sum::<f32>() / b.len() as f32
                       - (b.iter().sum::<f32>() / b.len() as f32).powi(2);

        if std_a < 1e-6 || std_b < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);
        let corr = va.correlation(&vb).unwrap();

        prop_assert!(
            (-1.0 - 1e-5..=1.0 + 1e-5).contains(&corr),
            "correlation = {} not in range [-1, 1]",
            corr
        );
    }

    /// Property test: ρ(X,Y) = ρ(Y,X) (symmetry)
    #[test]
    fn test_correlation_symmetric(
        ab in prop::collection::vec((-50.0f32..50.0, -50.0f32..50.0), 2..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        // Ensure vectors are not constant
        let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                       - (a.iter().sum::<f32>() / a.len() as f32).powi(2);
        let std_b: f32 = b.iter().map(|y| y * y).sum::<f32>() / b.len() as f32
                       - (b.iter().sum::<f32>() / b.len() as f32).powi(2);

        if std_a < 1e-6 || std_b < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);

        let corr_ab = va.correlation(&vb).unwrap();
        let corr_ba = vb.correlation(&va).unwrap();

        let tolerance = 1e-5;
        prop_assert!(
            (corr_ab - corr_ba).abs() < tolerance,
            "ρ(X,Y) = {} != ρ(Y,X) = {}",
            corr_ab, corr_ba
        );
    }

    /// Property test: ρ(X,X) = 1 (perfect self-correlation)
    #[test]
    fn test_correlation_self_is_one(
        a in prop::collection::vec(-50.0f32..50.0, 2..100)
    ) {
        // Ensure vector is not constant
        let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                       - (a.iter().sum::<f32>() / a.len() as f32).powi(2);

        if std_a < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let corr = va.correlation(&va).unwrap();

        prop_assert!(
            (corr - 1.0).abs() < 1e-5,
            "ρ(X,X) = {} != 1.0",
            corr
        );
    }
}

// ========================================================================
// Property tests for zscore() - Z-score normalization
// ========================================================================

proptest! {
    /// Property test: zscore() produces mean ≈ 0
    #[test]
    fn test_zscore_produces_zero_mean(
        a in prop::collection::vec(-100.0f32..100.0, 2..100)
    ) {
        // Ensure vector is not constant
        let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                       - (a.iter().sum::<f32>() / a.len() as f32).powi(2);

        if std_a < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let z = va.zscore().unwrap();
        let mean = z.mean().unwrap();

        prop_assert!(
            mean.abs() < 1e-4,
            "zscore mean = {}, expected ≈ 0",
            mean
        );
    }
}

proptest! {
    /// Property test: zscore() produces stddev ≈ 1
    #[test]
    fn test_zscore_produces_unit_stddev(
        a in prop::collection::vec(-100.0f32..100.0, 3..100)
    ) {
        // Ensure vector is not constant
        let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                       - (a.iter().sum::<f32>() / a.len() as f32).powi(2);

        if std_a < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let z = va.zscore().unwrap();
        let std = z.stddev().unwrap();

        // Use relaxed tolerance for floating-point precision, especially for small n
        prop_assert!(
            (std - 1.0).abs() < 1e-3,
            "zscore stddev = {}, expected ≈ 1",
            std
        );
    }
}

proptest! {
    /// Property test: zscore() preserves correlation structure
    /// ρ(zscore(X), zscore(Y)) = ρ(X, Y)
    #[test]
    fn test_zscore_preserves_correlation(
        ab in prop::collection::vec((-50.0f32..50.0, -50.0f32..50.0), 2..100)
    ) {
        let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
        let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

        // Ensure vectors have sufficient variance for stable correlation
        // Small variances cause numerical instability in zscore normalization
        let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                       - (a.iter().sum::<f32>() / a.len() as f32).powi(2);
        let std_b: f32 = b.iter().map(|x| x * x).sum::<f32>() / b.len() as f32
                       - (b.iter().sum::<f32>() / b.len() as f32).powi(2);

        if std_a < 0.1 || std_b < 0.1 {
            return Ok(());  // Skip near-constant vectors (variance < 0.1)
        }

        let va = Vector::from_slice(&a);
        let vb = Vector::from_slice(&b);

        // Original correlation
        let corr_orig = va.correlation(&vb).unwrap();

        // Correlation after zscore
        let za = va.zscore().unwrap();
        let zb = vb.zscore().unwrap();
        let corr_zscore = za.correlation(&zb).unwrap();

        let tolerance = 1e-3;
        prop_assert!(
            (corr_orig - corr_zscore).abs() < tolerance,
            "ρ(X,Y) = {} != ρ(zscore(X), zscore(Y)) = {}",
            corr_orig, corr_zscore
        );
    }
}

// ========================================================================
// Property tests for minmax_normalize() - Min-max normalization
// ========================================================================

proptest! {
    /// Property test: minmax_normalize() produces min = 0
    #[test]
    fn test_minmax_normalize_produces_zero_min(
        a in prop::collection::vec(-100.0f32..100.0, 2..100)
    ) {
        // Ensure vector is not constant
        let min_a = a.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_a = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if (max_a - min_a).abs() < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let normalized = va.minmax_normalize().unwrap();
        let min = normalized.min().unwrap();

        prop_assert!(
            min.abs() < 1e-4,
            "minmax min = {}, expected ≈ 0",
            min
        );
    }
}

proptest! {
    /// Property test: minmax_normalize() produces max = 1
    #[test]
    fn test_minmax_normalize_produces_one_max(
        a in prop::collection::vec(-100.0f32..100.0, 2..100)
    ) {
        // Ensure vector is not constant
        let min_a = a.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_a = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if (max_a - min_a).abs() < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let normalized = va.minmax_normalize().unwrap();
        let max = normalized.max().unwrap();

        prop_assert!(
            (max - 1.0).abs() < 1e-4,
            "minmax max = {}, expected ≈ 1",
            max
        );
    }
}

proptest! {
    /// Property test: minmax_normalize() preserves order (monotonicity)
    /// If a\[i\] <= a\[j\], then normalized\[i\] <= normalized\[j\]
    #[test]
    fn test_minmax_normalize_preserves_order(
        a in prop::collection::vec(-100.0f32..100.0, 2..100)
    ) {
        // Ensure vector is not constant
        let min_a = a.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_a = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if (max_a - min_a).abs() < 1e-6 {
            return Ok(());  // Skip constant vectors
        }

        let va = Vector::from_slice(&a);
        let normalized = va.minmax_normalize().unwrap();

        // Check that order is preserved for all pairs
        for i in 0..a.len() {
            for j in 0..a.len() {
                if a[i] <= a[j] {
                    prop_assert!(
                        normalized.data[i] <= normalized.data[j] + 1e-5,
                        "Order not preserved: a[{}]={} <= a[{}]={}, but norm[{}]={} > norm[{}]={}",
                        i, a[i], j, a[j], i, normalized.data[i], j, normalized.data[j]
                    );
                }
            }
        }
    }
}

// ========================================================================
// Property tests for clip() - Range clipping
// ========================================================================

proptest! {
    /// Property test: clip() produces values within [min_val, max_val]
    #[test]
    fn test_clip_within_bounds(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        min_val in -50.0f32..50.0,
        max_val in -50.0f32..50.0
    ) {
        // Ensure min <= max
        let (min_val, max_val) = if min_val <= max_val {
            (min_val, max_val)
        } else {
            (max_val, min_val)
        };

        let va = Vector::from_slice(&a);
        let clipped = va.clip(min_val, max_val).unwrap();

        // All values must be within [min_val, max_val]
        for &val in clipped.as_slice() {
            prop_assert!(
                (min_val..=max_val).contains(&val),
                "Value {} not in range [{}, {}]",
                val, min_val, max_val
            );
        }
    }
}

proptest! {
    /// Property test: clip() preserves order (monotonicity)
    /// If a\[i\] <= a\[j\], then clip(a)[i] <= clip(a)[j]
    #[test]
    fn test_clip_preserves_order(
        a in prop::collection::vec(-100.0f32..100.0, 2..100),
        min_val in -50.0f32..50.0,
        max_val in -50.0f32..50.0
    ) {
        // Ensure min <= max
        let (min_val, max_val) = if min_val <= max_val {
            (min_val, max_val)
        } else {
            (max_val, min_val)
        };

        let va = Vector::from_slice(&a);
        let clipped = va.clip(min_val, max_val).unwrap();

        // Check order preservation
        for i in 0..a.len() {
            for j in 0..a.len() {
                if a[i] <= a[j] {
                    prop_assert!(
                        clipped.data[i] <= clipped.data[j] + 1e-5,
                        "Order not preserved: a[{}]={} <= a[{}]={}, but clip[{}]={} > clip[{}]={}",
                        i, a[i], j, a[j], i, clipped.data[i], j, clipped.data[j]
                    );
                }
            }
        }
    }
}

proptest! {
    /// Property test: clip() is idempotent
    /// clip(clip(X)) = clip(X)
    #[test]
    fn test_clip_idempotent(
        a in prop::collection::vec(-100.0f32..100.0, 1..100),
        min_val in -50.0f32..50.0,
        max_val in -50.0f32..50.0
    ) {
        // Ensure min <= max
        let (min_val, max_val) = if min_val <= max_val {
            (min_val, max_val)
        } else {
            (max_val, min_val)
        };

        let va = Vector::from_slice(&a);
        let clipped1 = va.clip(min_val, max_val).unwrap();
        let clipped2 = clipped1.clip(min_val, max_val).unwrap();

        // Clipping twice should give same result
        for i in 0..clipped1.len() {
            prop_assert!(
                (clipped1.data[i] - clipped2.data[i]).abs() < 1e-5,
                "Idempotency violated at index {}: clip_once={}, clip_twice={}",
                i, clipped1.data[i], clipped2.data[i]
            );
        }
    }
}

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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
