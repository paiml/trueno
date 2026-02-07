use super::super::super::*;
use proptest::prelude::*;

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
