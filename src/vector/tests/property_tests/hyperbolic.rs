use super::super::super::*;
use proptest::prelude::*;

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
