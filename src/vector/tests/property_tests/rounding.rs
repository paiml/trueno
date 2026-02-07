use super::super::super::*;
use proptest::prelude::*;

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
