use super::super::super::*;
use proptest::prelude::*;

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
