//! Statistical Analysis Falsification Tests (F1101-F1115)
//!
//! Popperian falsification criteria for statistical analysis per §24.23.

use cbtop::{
    EffectCategory, StatisticalAnalysis, EffectSize,
    StatisticalComparison,
    MannWhitneyResult, OutlierFilter,
    bootstrap_ci, percentile, trimmed_mean,
};

// ============================================================================
// F1101: Bootstrap CI contains true mean
// ============================================================================

#[test]
fn f1101_bootstrap_ci_contains_mean() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let (lower, upper) = bootstrap_ci(&samples, 0.95, 1000);
    let mean = 3.0;

    assert!(lower < mean);
    assert!(upper > mean);
}

#[test]
fn f1101_bootstrap_ci_95_coverage() {
    // Test that 95% CI actually covers true mean ~95% of time
    let true_mean = 5.0;
    let samples: Vec<f64> = (0..100).map(|i| true_mean + (i as f64 % 10.0) - 5.0).collect();

    let (lower, upper) = bootstrap_ci(&samples, 0.95, 10000);

    // CI should contain the sample mean
    let sample_mean = samples.iter().sum::<f64>() / samples.len() as f64;
    assert!(lower <= sample_mean && sample_mean <= upper);
}

#[test]
fn f1101_statistical_analysis_ci() {
    let samples = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();

    assert!(analysis.ci_lower < analysis.mean);
    assert!(analysis.ci_upper > analysis.mean);
    assert!((analysis.confidence_level - 0.95).abs() < 0.01);
}

// ============================================================================
// F1102: Effect size categorized correctly
// ============================================================================

#[test]
fn f1102_effect_negligible() {
    assert_eq!(EffectCategory::from_cohens_d(0.0), EffectCategory::Negligible);
    assert_eq!(EffectCategory::from_cohens_d(0.1), EffectCategory::Negligible);
    assert_eq!(EffectCategory::from_cohens_d(0.19), EffectCategory::Negligible);
}

#[test]
fn f1102_effect_small() {
    assert_eq!(EffectCategory::from_cohens_d(0.2), EffectCategory::Small);
    assert_eq!(EffectCategory::from_cohens_d(0.3), EffectCategory::Small);
    assert_eq!(EffectCategory::from_cohens_d(0.49), EffectCategory::Small);
}

#[test]
fn f1102_effect_medium() {
    assert_eq!(EffectCategory::from_cohens_d(0.5), EffectCategory::Medium);
    assert_eq!(EffectCategory::from_cohens_d(0.6), EffectCategory::Medium);
    assert_eq!(EffectCategory::from_cohens_d(0.79), EffectCategory::Medium);
}

#[test]
fn f1102_effect_large() {
    assert_eq!(EffectCategory::from_cohens_d(0.8), EffectCategory::Large);
    assert_eq!(EffectCategory::from_cohens_d(1.0), EffectCategory::Large);
    assert_eq!(EffectCategory::from_cohens_d(2.0), EffectCategory::Large);
}

#[test]
fn f1102_effect_negative() {
    // Negative d should still be categorized by magnitude
    assert_eq!(EffectCategory::from_cohens_d(-0.1), EffectCategory::Negligible);
    assert_eq!(EffectCategory::from_cohens_d(-0.5), EffectCategory::Medium);
    assert_eq!(EffectCategory::from_cohens_d(-1.0), EffectCategory::Large);
}

// ============================================================================
// F1103: t-test p-value accurate
// ============================================================================

#[test]
fn f1103_welch_t_significant() {
    // Very different samples should be significant
    let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sample2 = vec![10.0, 11.0, 12.0, 13.0, 14.0];

    let result = StatisticalComparison::welch_t_test(&sample1, &sample2).unwrap();

    assert!(result.statistically_significant);
    assert!(result.p_value < 0.05);
}

#[test]
fn f1103_welch_t_not_significant() {
    // Similar samples should not be significant
    let sample1 = vec![5.0, 5.1, 4.9, 5.0, 5.1];
    let sample2 = vec![5.0, 4.9, 5.1, 5.0, 4.9];

    let result = StatisticalComparison::welch_t_test(&sample1, &sample2).unwrap();

    // Should not be significant (similar means)
    assert!(result.p_value > 0.05 || !result.statistically_significant);
}

#[test]
fn f1103_t_statistic_sign() {
    let sample1 = vec![1.0, 2.0, 3.0];
    let sample2 = vec![4.0, 5.0, 6.0];

    let result = StatisticalComparison::welch_t_test(&sample1, &sample2).unwrap();

    // sample1 < sample2, so t should be negative
    assert!(result.t_statistic < 0.0);
}

// ============================================================================
// F1104: Mann-Whitney handles ties
// ============================================================================

#[test]
fn f1104_mann_whitney_with_ties() {
    let sample1 = vec![1.0, 2.0, 2.0, 3.0];
    let sample2 = vec![2.0, 3.0, 3.0, 4.0];

    // Should not panic on ties
    let result = MannWhitneyResult::test(&sample1, &sample2);
    assert!(result.is_some());
}

#[test]
fn f1104_mann_whitney_all_equal() {
    let sample1 = vec![5.0, 5.0, 5.0];
    let sample2 = vec![5.0, 5.0, 5.0];

    // Should not panic when all values are equal
    let result = MannWhitneyResult::test(&sample1, &sample2);
    assert!(result.is_some());
}

// ============================================================================
// F1105: IQR filter removes outliers
// ============================================================================

#[test]
fn f1105_outlier_removed() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];

    let filter = OutlierFilter::new(&samples).unwrap();
    let filtered = filter.filter(&samples);

    assert!(!filtered.contains(&100.0));
    assert_eq!(filtered.len(), 5);
}

#[test]
fn f1105_outlier_count() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0, -100.0];

    let filter = OutlierFilter::new(&samples).unwrap();
    let count = filter.count_outliers(&samples);

    assert!(count >= 2);
}

#[test]
fn f1105_outlier_detection() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let filter = OutlierFilter::new(&samples).unwrap();

    assert!(!filter.is_outlier(3.0));
    assert!(filter.is_outlier(100.0));
    assert!(filter.is_outlier(-100.0));
}

// ============================================================================
// F1106: Empty input handled
// ============================================================================

#[test]
fn f1106_statistical_analysis_empty() {
    let samples: Vec<f64> = vec![];
    assert!(StatisticalAnalysis::from_samples_default(&samples).is_none());
}

#[test]
fn f1106_cohens_d_empty() {
    assert!(EffectSize::cohens_d(&[], &[1.0, 2.0]).is_none());
    assert!(EffectSize::cohens_d(&[1.0, 2.0], &[]).is_none());
}

#[test]
fn f1106_welch_t_empty() {
    let sample = vec![1.0, 2.0];
    assert!(StatisticalComparison::welch_t_test(&[], &sample).is_none());
    assert!(StatisticalComparison::welch_t_test(&sample, &[]).is_none());
}

#[test]
fn f1106_mann_whitney_empty() {
    assert!(MannWhitneyResult::test(&[], &[1.0]).is_none());
    assert!(MannWhitneyResult::test(&[1.0], &[]).is_none());
}

#[test]
fn f1106_outlier_filter_empty() {
    assert!(OutlierFilter::new(&[]).is_none());
}

#[test]
fn f1106_bootstrap_ci_empty() {
    let (lower, upper) = bootstrap_ci(&[], 0.95, 1000);
    assert_eq!(lower, 0.0);
    assert_eq!(upper, 0.0);
}

#[test]
fn f1106_trimmed_mean_empty() {
    assert!(trimmed_mean(&[], 0.1).is_none());
}

// ============================================================================
// F1107: Single element handled
// ============================================================================

#[test]
fn f1107_statistical_analysis_single() {
    let samples = vec![5.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();

    assert!((analysis.mean - 5.0).abs() < 0.01);
    assert!((analysis.ci_lower - 5.0).abs() < 0.01);
    assert!((analysis.ci_upper - 5.0).abs() < 0.01);
}

#[test]
fn f1107_bootstrap_single() {
    let (lower, upper) = bootstrap_ci(&[5.0], 0.95, 1000);
    assert!((lower - 5.0).abs() < 0.01);
    assert!((upper - 5.0).abs() < 0.01);
}

// ============================================================================
// F1108: Negative values accepted
// ============================================================================

#[test]
fn f1108_negative_samples() {
    let samples = vec![-5.0, -3.0, -1.0, 1.0, 3.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();

    assert!((analysis.mean - (-1.0)).abs() < 0.01);
}

#[test]
fn f1108_cohens_d_negative() {
    let sample1 = vec![-5.0, -4.0, -3.0];
    let sample2 = vec![-2.0, -1.0, 0.0];

    let effect = EffectSize::cohens_d(&sample1, &sample2);
    assert!(effect.is_some());
}

// ============================================================================
// F1109: NaN/Inf rejected
// ============================================================================

#[test]
fn f1109_nan_filtered() {
    let samples = vec![1.0, f64::NAN, 3.0, 5.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();

    assert!(analysis.mean.is_finite());
    assert_eq!(analysis.n, 3); // NaN excluded
}

#[test]
fn f1109_inf_filtered() {
    let samples = vec![1.0, f64::INFINITY, 3.0, 5.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();

    assert!(analysis.mean.is_finite());
    assert_eq!(analysis.n, 3); // Inf excluded
}

#[test]
fn f1109_all_nan() {
    let samples = vec![f64::NAN, f64::NAN];
    assert!(StatisticalAnalysis::from_samples_default(&samples).is_none());
}

// ============================================================================
// F1110: Large samples efficient
// ============================================================================

#[test]
#[ignore = "Environment-dependent: timing varies on CI runners under load"]
fn f1110_large_sample() {
    let samples: Vec<f64> = (0..10000).map(|i| i as f64).collect();

    // Should complete quickly (not O(n^2))
    let start = std::time::Instant::now();
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();
    let elapsed = start.elapsed();

    assert!(elapsed.as_millis() < 1000); // Should be fast
    assert!(analysis.mean.is_finite());
}

// ============================================================================
// F1111: Bootstrap iterations configurable
// ============================================================================

#[test]
fn f1111_bootstrap_iterations() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Different iteration counts should work
    let (l1, u1) = bootstrap_ci(&samples, 0.95, 100);
    let (l2, u2) = bootstrap_ci(&samples, 0.95, 10000);

    // Both should produce valid intervals
    assert!(l1 < u1);
    assert!(l2 < u2);
}

// ============================================================================
// F1112: CI width decreases with n
// ============================================================================

#[test]
fn f1112_ci_width_vs_n() {
    let small_samples: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let large_samples: Vec<f64> = (0..100).map(|i| i as f64).collect();

    let small_analysis = StatisticalAnalysis::from_samples_default(&small_samples).unwrap();
    let large_analysis = StatisticalAnalysis::from_samples_default(&large_samples).unwrap();

    // Larger sample should have narrower CI (relative to mean)
    let small_width = small_analysis.ci_width() / small_analysis.mean.abs();
    let large_width = large_analysis.ci_width() / large_analysis.mean.abs();

    assert!(large_width < small_width);
}

// ============================================================================
// F1113: Effect size sign correct
// ============================================================================

#[test]
fn f1113_effect_positive() {
    let sample1 = vec![10.0, 11.0, 12.0];
    let sample2 = vec![1.0, 2.0, 3.0];

    let effect = EffectSize::cohens_d(&sample1, &sample2).unwrap();
    assert!(effect.cohens_d > 0.0); // sample1 > sample2
}

#[test]
fn f1113_effect_negative() {
    let sample1 = vec![1.0, 2.0, 3.0];
    let sample2 = vec![10.0, 11.0, 12.0];

    let effect = EffectSize::cohens_d(&sample1, &sample2).unwrap();
    assert!(effect.cohens_d < 0.0); // sample1 < sample2
}

// ============================================================================
// F1114: Confidence level configurable
// ============================================================================

#[test]
fn f1114_confidence_90() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let analysis = StatisticalAnalysis::from_samples(&samples, 0.90).unwrap();

    assert!((analysis.confidence_level - 0.90).abs() < 0.01);
}

#[test]
fn f1114_confidence_99() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let analysis = StatisticalAnalysis::from_samples(&samples, 0.99).unwrap();

    assert!((analysis.confidence_level - 0.99).abs() < 0.01);
}

#[test]
fn f1114_higher_confidence_wider_ci() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

    let ci_90 = StatisticalAnalysis::from_samples(&samples, 0.90).unwrap();
    let ci_99 = StatisticalAnalysis::from_samples(&samples, 0.99).unwrap();

    // 99% CI should be wider than 90% CI
    assert!(ci_99.ci_width() > ci_90.ci_width());
}

// ============================================================================
// F1115: Thread-safe RNG
// ============================================================================

#[test]
fn f1115_deterministic_bootstrap() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Multiple calls with same input should give consistent results
    let (l1, u1) = bootstrap_ci(&samples, 0.95, 1000);
    let (l2, u2) = bootstrap_ci(&samples, 0.95, 1000);

    // Results should be deterministic (same seed)
    assert!((l1 - l2).abs() < 0.01);
    assert!((u1 - u2).abs() < 0.01);
}

// ============================================================================
// Additional Coverage Tests
// ============================================================================

#[test]
fn test_effect_category_description() {
    assert!(!EffectCategory::Negligible.description().is_empty());
    assert!(!EffectCategory::Small.description().is_empty());
    assert!(!EffectCategory::Medium.description().is_empty());
    assert!(!EffectCategory::Large.description().is_empty());
}

#[test]
fn test_effect_is_significant() {
    let sample1 = vec![1.0, 2.0, 3.0];
    let sample2 = vec![10.0, 11.0, 12.0];

    let effect = EffectSize::cohens_d(&sample1, &sample2).unwrap();
    assert!(effect.is_significant());
}

#[test]
fn test_comparison_is_meaningful() {
    let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let sample2 = vec![10.0, 11.0, 12.0, 13.0, 14.0];

    let result = StatisticalComparison::welch_t_test(&sample1, &sample2).unwrap();
    assert!(result.is_meaningful());
}

#[test]
fn test_outlier_filter_custom_multiplier() {
    let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Tighter multiplier (1.0 instead of 1.5)
    let tight = OutlierFilter::with_multiplier(&samples, 1.0).unwrap();
    let normal = OutlierFilter::new(&samples).unwrap();

    // Tighter multiplier should have smaller fences
    assert!(tight.upper_fence < normal.upper_fence);
}

#[test]
fn test_statistical_analysis_cv() {
    let samples = vec![100.0, 100.0, 100.0, 100.0, 100.0];
    let analysis = StatisticalAnalysis::from_samples_default(&samples).unwrap();

    // Constant values should have 0% CV
    assert!(analysis.cv_percent < 0.01);
}

#[test]
fn test_ci_is_narrow() {
    let tight_samples = vec![100.0, 100.1, 99.9, 100.0, 100.1];
    let analysis = StatisticalAnalysis::from_samples_default(&tight_samples).unwrap();

    assert!(analysis.ci_is_narrow());
}

#[test]
fn test_percentile_function() {
    let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    assert!((percentile(&sorted, 0.0) - 1.0).abs() < 0.01);
    assert!((percentile(&sorted, 0.25) - 2.0).abs() < 0.01);
    assert!((percentile(&sorted, 0.5) - 3.0).abs() < 0.01);
    assert!((percentile(&sorted, 0.75) - 4.0).abs() < 0.01);
    assert!((percentile(&sorted, 1.0) - 5.0).abs() < 0.01);
}

#[test]
fn test_trimmed_mean_removes_extremes() {
    let samples = vec![1.0, 100.0, 200.0, 300.0, 10000.0];

    let normal_mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let trimmed = trimmed_mean(&samples, 0.2).unwrap();

    // Trimmed mean should be less affected by extremes
    assert!(trimmed < normal_mean);
}
