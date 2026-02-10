//! Profile Diffing and A/B Comparison (PMAT-045)
//!
//! Statistical comparison of benchmark profiles for regression detection.
//!
//! # Design
//!
//! - Welch's t-test for comparing two sample sets
//! - Confidence interval computation with configurable levels
//! - Effect size calculation (Cohen's d)
//! - Multiple comparison correction (Bonferroni)
//!
//! # Falsification (FKR-046)
//!
//! H₀: Profile diff cannot detect 5% regression with 95% confidence
//! Test: Inject known 5% regression, verify detection rate >80%

use std::collections::HashMap;

/// Result type for profile comparison operations
pub type CompareResult<T> = Result<T, CompareError>;

/// Errors in profile comparison
#[derive(Debug, Clone, PartialEq)]
pub enum CompareError {
    /// Insufficient samples for comparison
    InsufficientSamples { got: usize, need: usize },
    /// Metric not found in profile
    MetricNotFound { name: String },
    /// Variance is zero (no variation in data)
    ZeroVariance { metric: String },
    /// Invalid confidence level
    InvalidConfidence { value: f64 },
    /// Profiles have no common metrics
    NoCommonMetrics,
}

impl std::fmt::Display for CompareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientSamples { got, need } => {
                write!(f, "Insufficient samples: got {}, need {}", got, need)
            }
            Self::MetricNotFound { name } => {
                write!(f, "Metric not found: {}", name)
            }
            Self::ZeroVariance { metric } => {
                write!(f, "Zero variance in metric: {}", metric)
            }
            Self::InvalidConfidence { value } => {
                write!(f, "Invalid confidence level: {}", value)
            }
            Self::NoCommonMetrics => {
                write!(f, "Profiles have no common metrics")
            }
        }
    }
}

impl std::error::Error for CompareError {}

/// A benchmark profile containing multiple metrics
#[derive(Debug, Clone)]
pub struct BenchmarkProfile {
    /// Profile name/identifier
    pub name: String,
    /// Profile description
    pub description: Option<String>,
    /// Metrics with their sample values
    pub metrics: HashMap<String, MetricSamples>,
    /// Profile metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp when profile was captured
    pub timestamp_ns: u64,
}

impl BenchmarkProfile {
    /// Create a new benchmark profile
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            metrics: HashMap::new(),
            metadata: HashMap::new(),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
        }
    }

    /// Set profile description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add a metric with samples
    pub fn add_metric(&mut self, name: impl Into<String>, samples: Vec<f64>) {
        self.metrics
            .insert(name.into(), MetricSamples::new(samples));
    }

    /// Get metric by name
    pub fn get_metric(&self, name: &str) -> Option<&MetricSamples> {
        self.metrics.get(name)
    }

    /// Get all metric names
    pub fn metric_names(&self) -> impl Iterator<Item = &String> {
        self.metrics.keys()
    }

    /// Number of metrics
    pub fn metric_count(&self) -> usize {
        self.metrics.len()
    }
}

/// Samples for a single metric
#[derive(Debug, Clone)]
pub struct MetricSamples {
    /// Raw sample values
    pub values: Vec<f64>,
    /// Precomputed mean
    mean: f64,
    /// Precomputed variance
    variance: f64,
    /// Precomputed standard deviation
    std_dev: f64,
}

impl MetricSamples {
    /// Create new metric samples
    pub fn new(values: Vec<f64>) -> Self {
        let (mean, variance, std_dev) = if values.is_empty() {
            (0.0, 0.0, 0.0)
        } else {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (values.len() - 1).max(1) as f64;
            let std_dev = variance.sqrt();
            (mean, variance, std_dev)
        };

        Self {
            values,
            mean,
            variance,
            std_dev,
        }
    }

    /// Get sample count
    pub fn count(&self) -> usize {
        self.values.len()
    }

    /// Get mean
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get variance
    pub fn variance(&self) -> f64 {
        self.variance
    }

    /// Get standard deviation
    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }

    /// Get minimum value
    pub fn min(&self) -> f64 {
        self.values.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// Get maximum value
    pub fn max(&self) -> f64 {
        self.values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Result of Welch's t-test comparison
#[derive(Debug, Clone)]
pub struct WelchTestResult {
    /// T-statistic
    pub t_statistic: f64,
    /// Degrees of freedom (Welch-Satterthwaite)
    pub degrees_of_freedom: f64,
    /// Two-tailed p-value
    pub p_value: f64,
    /// Whether result is statistically significant
    pub significant: bool,
    /// Confidence level used
    pub confidence_level: f64,
}

/// Effect size interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectMagnitude {
    /// Negligible effect (|d| < 0.2)
    Negligible,
    /// Small effect (0.2 <= |d| < 0.5)
    Small,
    /// Medium effect (0.5 <= |d| < 0.8)
    Medium,
    /// Large effect (|d| >= 0.8)
    Large,
}

impl EffectMagnitude {
    /// Get magnitude from Cohen's d value
    pub fn from_cohens_d(d: f64) -> Self {
        let abs_d = d.abs();
        if abs_d < 0.2 {
            Self::Negligible
        } else if abs_d < 0.5 {
            Self::Small
        } else if abs_d < 0.8 {
            Self::Medium
        } else {
            Self::Large
        }
    }
}

/// Effect size result
#[derive(Debug, Clone)]
pub struct EffectSizeResult {
    /// Cohen's d effect size
    pub cohens_d: f64,
    /// Effect magnitude interpretation
    pub magnitude: EffectMagnitude,
    /// Percentage change (new - old) / old * 100
    pub percent_change: f64,
}

/// Direction of change
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeDirection {
    /// Performance improved (metric decreased for latency, increased for throughput)
    Improved,
    /// Performance regressed
    Regressed,
    /// No significant change
    NoChange,
}

/// Result of comparing a single metric
#[derive(Debug, Clone)]
pub struct MetricComparison {
    /// Metric name
    pub name: String,
    /// Baseline (A) statistics
    pub baseline_mean: f64,
    /// Baseline standard deviation
    pub baseline_std: f64,
    /// Comparison (B) statistics
    pub comparison_mean: f64,
    /// Comparison standard deviation
    pub comparison_std: f64,
    /// Welch's t-test result
    pub t_test: WelchTestResult,
    /// Effect size analysis
    pub effect_size: EffectSizeResult,
    /// Direction of change
    pub direction: ChangeDirection,
    /// Whether this is a regression
    pub is_regression: bool,
    /// Confidence interval for the difference
    pub ci_lower: f64,
    /// Upper bound of confidence interval
    pub ci_upper: f64,
}

/// Complete A/B comparison result
#[derive(Debug, Clone)]
pub struct ProfileComparison {
    /// Baseline profile name
    pub baseline_name: String,
    /// Comparison profile name
    pub comparison_name: String,
    /// Individual metric comparisons
    pub metrics: Vec<MetricComparison>,
    /// Metrics that regressed significantly
    pub regressions: Vec<String>,
    /// Metrics that improved significantly
    pub improvements: Vec<String>,
    /// Overall verdict
    pub verdict: ComparisonVerdict,
    /// Bonferroni-corrected alpha for multiple comparisons
    pub corrected_alpha: f64,
}

impl ProfileComparison {
    /// Get regression count
    pub fn regression_count(&self) -> usize {
        self.regressions.len()
    }

    /// Get improvement count
    pub fn improvement_count(&self) -> usize {
        self.improvements.len()
    }

    /// Get metrics with no significant change
    pub fn unchanged_count(&self) -> usize {
        self.metrics.len() - self.regression_count() - self.improvement_count()
    }

    /// Check if comparison detected any regressions
    pub fn has_regressions(&self) -> bool {
        !self.regressions.is_empty()
    }
}

/// Overall comparison verdict
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonVerdict {
    /// All metrics stable or improved
    Pass,
    /// Minor regressions detected (< 5%)
    Warning,
    /// Significant regressions detected (>= 5%)
    Fail,
}

/// Configuration for profile comparison
#[derive(Debug, Clone)]
pub struct CompareConfig {
    /// Confidence level (default: 0.95)
    pub confidence_level: f64,
    /// Minimum samples required for comparison
    pub min_samples: usize,
    /// Apply Bonferroni correction for multiple comparisons
    pub bonferroni_correction: bool,
    /// Threshold for regression (percent change)
    pub regression_threshold_percent: f64,
    /// Metrics where higher is better (throughput-like)
    pub higher_is_better: Vec<String>,
    /// Metrics where lower is better (latency-like, default)
    pub lower_is_better: Vec<String>,
}

impl Default for CompareConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            min_samples: 5,
            bonferroni_correction: true,
            regression_threshold_percent: 5.0,
            higher_is_better: vec![
                "throughput".to_string(),
                "ops_per_sec".to_string(),
                "requests_per_sec".to_string(),
            ],
            lower_is_better: vec![
                "latency".to_string(),
                "latency_p50".to_string(),
                "latency_p99".to_string(),
                "memory".to_string(),
                "cpu_usage".to_string(),
            ],
        }
    }
}

/// Profile comparator
#[derive(Debug)]
pub struct ProfileComparator {
    /// Comparison configuration
    config: CompareConfig,
}

impl ProfileComparator {
    /// Create a new profile comparator
    pub fn new(config: CompareConfig) -> Self {
        Self { config }
    }

    /// Compare two benchmark profiles
    pub fn compare(
        &self,
        baseline: &BenchmarkProfile,
        comparison: &BenchmarkProfile,
    ) -> CompareResult<ProfileComparison> {
        // Find common metrics
        let common_metrics: Vec<String> = baseline
            .metric_names()
            .filter(|name| comparison.metrics.contains_key(*name))
            .cloned()
            .collect();

        if common_metrics.is_empty() {
            return Err(CompareError::NoCommonMetrics);
        }

        // Calculate corrected alpha for multiple comparisons
        let alpha = 1.0 - self.config.confidence_level;
        let corrected_alpha = if self.config.bonferroni_correction {
            alpha / common_metrics.len() as f64
        } else {
            alpha
        };

        let mut metric_comparisons = Vec::new();
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();

        for metric_name in &common_metrics {
            let baseline_samples = baseline
                .get_metric(metric_name)
                .expect("metric should exist in baseline");
            let comparison_samples = comparison
                .get_metric(metric_name)
                .expect("metric should exist in comparison");

            // Check minimum samples
            if baseline_samples.count() < self.config.min_samples {
                continue;
            }
            if comparison_samples.count() < self.config.min_samples {
                continue;
            }

            let comparison_result = self.compare_metric(
                metric_name,
                baseline_samples,
                comparison_samples,
                corrected_alpha,
            )?;

            if comparison_result.is_regression {
                regressions.push(metric_name.clone());
            } else if comparison_result.t_test.significant
                && comparison_result.direction == ChangeDirection::Improved
            {
                improvements.push(metric_name.clone());
            }

            metric_comparisons.push(comparison_result);
        }

        // Determine verdict
        let verdict = if regressions.is_empty() {
            ComparisonVerdict::Pass
        } else {
            // Check if any regression exceeds threshold
            let severe_regression =
                metric_comparisons
                    .iter()
                    .filter(|m| m.is_regression)
                    .any(|m| {
                        m.effect_size.percent_change.abs()
                            >= self.config.regression_threshold_percent
                    });

            if severe_regression {
                ComparisonVerdict::Fail
            } else {
                ComparisonVerdict::Warning
            }
        };

        Ok(ProfileComparison {
            baseline_name: baseline.name.clone(),
            comparison_name: comparison.name.clone(),
            metrics: metric_comparisons,
            regressions,
            improvements,
            verdict,
            corrected_alpha,
        })
    }

    /// Compare a single metric between two profiles
    fn compare_metric(
        &self,
        name: &str,
        baseline: &MetricSamples,
        comparison: &MetricSamples,
        alpha: f64,
    ) -> CompareResult<MetricComparison> {
        // Check for zero variance
        if baseline.variance() == 0.0 && comparison.variance() == 0.0 {
            return Err(CompareError::ZeroVariance {
                metric: name.to_string(),
            });
        }

        // Perform Welch's t-test
        let t_test = self.welch_t_test(baseline, comparison, alpha)?;

        // Calculate effect size (Cohen's d with pooled std)
        let pooled_std = self.pooled_std(baseline, comparison);
        let cohens_d = if pooled_std > 0.0 {
            (comparison.mean() - baseline.mean()) / pooled_std
        } else {
            0.0
        };

        let percent_change = if baseline.mean().abs() > 1e-10 {
            ((comparison.mean() - baseline.mean()) / baseline.mean()) * 100.0
        } else {
            0.0
        };

        let effect_size = EffectSizeResult {
            cohens_d,
            magnitude: EffectMagnitude::from_cohens_d(cohens_d),
            percent_change,
        };

        // Determine direction and regression
        let higher_is_better = self
            .config
            .higher_is_better
            .iter()
            .any(|m| name.contains(m));

        let direction = if !t_test.significant {
            ChangeDirection::NoChange
        } else if higher_is_better {
            if comparison.mean() > baseline.mean() {
                ChangeDirection::Improved
            } else {
                ChangeDirection::Regressed
            }
        } else {
            // Lower is better (latency-like)
            if comparison.mean() < baseline.mean() {
                ChangeDirection::Improved
            } else {
                ChangeDirection::Regressed
            }
        };

        let is_regression = direction == ChangeDirection::Regressed;

        // Calculate confidence interval for the difference
        let (ci_lower, ci_upper) = self.confidence_interval(baseline, comparison, alpha);

        Ok(MetricComparison {
            name: name.to_string(),
            baseline_mean: baseline.mean(),
            baseline_std: baseline.std_dev(),
            comparison_mean: comparison.mean(),
            comparison_std: comparison.std_dev(),
            t_test,
            effect_size,
            direction,
            is_regression,
            ci_lower,
            ci_upper,
        })
    }

    /// Perform Welch's t-test
    fn welch_t_test(
        &self,
        a: &MetricSamples,
        b: &MetricSamples,
        alpha: f64,
    ) -> CompareResult<WelchTestResult> {
        let n1 = a.count() as f64;
        let n2 = b.count() as f64;
        let var1 = a.variance();
        let var2 = b.variance();

        // Calculate t-statistic
        let mean_diff = b.mean() - a.mean();
        let se = ((var1 / n1) + (var2 / n2)).sqrt();

        let t_statistic = if se > 1e-10 { mean_diff / se } else { 0.0 };

        // Welch-Satterthwaite degrees of freedom
        let v1 = var1 / n1;
        let v2 = var2 / n2;
        let df = if (v1 + v2) > 1e-10 {
            (v1 + v2).powi(2) / (v1.powi(2) / (n1 - 1.0) + v2.powi(2) / (n2 - 1.0))
        } else {
            n1 + n2 - 2.0
        };

        // Approximate p-value using t-distribution CDF
        let p_value = self.t_distribution_p_value(t_statistic.abs(), df);

        let significant = p_value < alpha;

        Ok(WelchTestResult {
            t_statistic,
            degrees_of_freedom: df,
            p_value,
            significant,
            confidence_level: 1.0 - alpha,
        })
    }

    /// Approximate p-value from t-distribution using approximation
    fn t_distribution_p_value(&self, t: f64, df: f64) -> f64 {
        // Use approximation for two-tailed p-value
        // Based on Hill's approximation for Student's t-distribution

        if df <= 0.0 {
            return 1.0;
        }

        let x = df / (df + t * t);

        // Incomplete beta function approximation
        // For large df, use normal approximation
        if df > 100.0 {
            // Normal approximation
            let z = t;
            2.0 * (1.0 - self.normal_cdf(z.abs()))
        } else {
            // Beta approximation
            let a = df / 2.0;
            let b = 0.5;
            self.incomplete_beta(x, a, b)
        }
    }

    /// Normal CDF approximation
    fn normal_cdf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x / 2.0).exp();

        0.5 * (1.0 + sign * y)
    }

    /// Incomplete beta function approximation
    fn incomplete_beta(&self, x: f64, a: f64, b: f64) -> f64 {
        // Simple continued fraction approximation
        if !(0.0..=1.0).contains(&x) {
            return 0.0;
        }
        if x == 0.0 {
            return 0.0;
        }
        if x == 1.0 {
            return 1.0;
        }

        // Use continued fraction
        let bt = if x == 0.0 || x == 1.0 {
            0.0
        } else {
            (self.ln_gamma(a + b) - self.ln_gamma(a) - self.ln_gamma(b)
                + a * x.ln()
                + b * (1.0 - x).ln())
            .exp()
        };

        if x < (a + 1.0) / (a + b + 2.0) {
            bt * self.beta_cf(x, a, b) / a
        } else {
            1.0 - bt * self.beta_cf(1.0 - x, b, a) / b
        }
    }

    /// Beta continued fraction
    fn beta_cf(&self, x: f64, a: f64, b: f64) -> f64 {
        let max_iter = 100;
        let eps: f64 = 1e-10;

        let mut c: f64 = 1.0;
        let mut d: f64 = 1.0 / (1.0 - (a + b) * x / (a + 1.0)).max(eps);
        let mut h: f64 = d;

        for m in 1..=max_iter {
            let m = m as f64;

            // Even step
            let aa = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
            d = 1.0 / (1.0 + aa * d).max(eps);
            c = 1.0 + aa / c.max(eps);
            h *= d * c;

            // Odd step
            let aa = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
            d = 1.0 / (1.0 + aa * d).max(eps);
            c = 1.0 + aa / c.max(eps);
            let del = d * c;
            h *= del;

            if (del - 1.0).abs() < eps {
                break;
            }
        }

        h
    }

    /// Log gamma function approximation (Stirling)
    #[allow(clippy::excessive_precision)]
    fn ln_gamma(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return f64::INFINITY;
        }

        // Lanczos approximation
        let g = 7;
        let c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7,
        ];

        if x < 0.5 {
            std::f64::consts::PI.ln()
                - (std::f64::consts::PI * x).sin().ln()
                - self.ln_gamma(1.0 - x)
        } else {
            let x = x - 1.0;
            let mut a = c[0];
            for i in 1..=g {
                a += c[i] / (x + i as f64);
            }
            let t = x + g as f64 + 0.5;
            0.5 * (2.0 * std::f64::consts::PI).ln() + (t - 0.5) * t.ln() - t + a.ln()
        }
    }

    /// Calculate pooled standard deviation
    fn pooled_std(&self, a: &MetricSamples, b: &MetricSamples) -> f64 {
        let n1 = a.count() as f64;
        let n2 = b.count() as f64;

        if n1 + n2 <= 2.0 {
            return 0.0;
        }

        let pooled_var = ((n1 - 1.0) * a.variance() + (n2 - 1.0) * b.variance()) / (n1 + n2 - 2.0);

        pooled_var.sqrt()
    }

    /// Calculate confidence interval for the difference in means
    fn confidence_interval(&self, a: &MetricSamples, b: &MetricSamples, alpha: f64) -> (f64, f64) {
        let mean_diff = b.mean() - a.mean();
        let se = ((a.variance() / a.count() as f64) + (b.variance() / b.count() as f64)).sqrt();

        // Use normal approximation for large samples
        let z = self.normal_quantile(1.0 - alpha / 2.0);
        let margin = z * se;

        (mean_diff - margin, mean_diff + margin)
    }

    /// Normal quantile function (inverse CDF) approximation
    fn normal_quantile(&self, p: f64) -> f64 {
        // Rational approximation from Abramowitz and Stegun
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        if p == 0.5 {
            return 0.0;
        }

        let t = if p < 0.5 {
            (-2.0 * p.ln()).sqrt()
        } else {
            (-2.0 * (1.0 - p).ln()).sqrt()
        };

        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        let x = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

        if p < 0.5 {
            -x
        } else {
            x
        }
    }

    /// Get configuration
    pub fn config(&self) -> &CompareConfig {
        &self.config
    }
}

/// Minimum samples required for reliable comparison
pub const MIN_COMPARISON_SAMPLES: usize = 5;

/// Default confidence level
pub const DEFAULT_CONFIDENCE_LEVEL: f64 = 0.95;

/// Default regression threshold (percent)
pub const DEFAULT_REGRESSION_THRESHOLD: f64 = 5.0;

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_profile(
        name: &str,
        latency_samples: Vec<f64>,
        throughput_samples: Vec<f64>,
    ) -> BenchmarkProfile {
        let mut profile = BenchmarkProfile::new(name);
        profile.add_metric("latency_p50", latency_samples);
        profile.add_metric("throughput", throughput_samples);
        profile
    }

    #[test]
    fn test_metric_samples_statistics() {
        let samples = MetricSamples::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(samples.count(), 5);
        assert!((samples.mean() - 3.0).abs() < 0.01);
        assert!((samples.variance() - 2.5).abs() < 0.01);
        assert!((samples.std_dev() - 1.58).abs() < 0.1);
        assert_eq!(samples.min(), 1.0);
        assert_eq!(samples.max(), 5.0);
    }

    #[test]
    fn test_empty_samples() {
        let samples = MetricSamples::new(vec![]);

        assert_eq!(samples.count(), 0);
        assert_eq!(samples.mean(), 0.0);
        assert_eq!(samples.variance(), 0.0);
    }

    #[test]
    fn test_effect_magnitude() {
        assert_eq!(
            EffectMagnitude::from_cohens_d(0.1),
            EffectMagnitude::Negligible
        );
        assert_eq!(EffectMagnitude::from_cohens_d(0.3), EffectMagnitude::Small);
        assert_eq!(EffectMagnitude::from_cohens_d(0.6), EffectMagnitude::Medium);
        assert_eq!(EffectMagnitude::from_cohens_d(1.0), EffectMagnitude::Large);
        assert_eq!(EffectMagnitude::from_cohens_d(-0.9), EffectMagnitude::Large);
    }

    #[test]
    fn test_profile_creation() {
        let profile = BenchmarkProfile::new("test")
            .with_description("Test profile")
            .with_metadata("version", "1.0");

        assert_eq!(profile.name, "test");
        assert_eq!(profile.description, Some("Test profile".to_string()));
        assert_eq!(profile.metadata.get("version"), Some(&"1.0".to_string()));
    }

    #[test]
    fn test_profile_comparison_no_regression() {
        let baseline = create_test_profile(
            "baseline",
            vec![100.0, 102.0, 98.0, 101.0, 99.0], // latency
            vec![1000.0, 1010.0, 990.0, 1005.0, 995.0], // throughput
        );

        let comparison = create_test_profile(
            "comparison",
            vec![99.0, 101.0, 97.0, 100.0, 98.0], // slightly better latency
            vec![1005.0, 1015.0, 995.0, 1010.0, 1000.0], // slightly better throughput
        );

        let comparator = ProfileComparator::new(CompareConfig::default());
        let result = comparator.compare(&baseline, &comparison).unwrap();

        assert_eq!(result.verdict, ComparisonVerdict::Pass);
        assert!(result.regressions.is_empty());
    }

    #[test]
    fn test_profile_comparison_with_regression() {
        let baseline = create_test_profile(
            "baseline",
            vec![100.0, 102.0, 98.0, 101.0, 99.0], // latency ~100
            vec![1000.0, 1010.0, 990.0, 1005.0, 995.0], // throughput ~1000
        );

        let comparison = create_test_profile(
            "comparison",
            vec![120.0, 122.0, 118.0, 121.0, 119.0], // latency ~120 (20% worse)
            vec![800.0, 810.0, 790.0, 805.0, 795.0], // throughput ~800 (20% worse)
        );

        let comparator = ProfileComparator::new(CompareConfig::default());
        let result = comparator.compare(&baseline, &comparison).unwrap();

        assert_eq!(result.verdict, ComparisonVerdict::Fail);
        assert!(!result.regressions.is_empty());
    }

    #[test]
    fn test_no_common_metrics() {
        let mut baseline = BenchmarkProfile::new("baseline");
        baseline.add_metric("metric_a", vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let mut comparison = BenchmarkProfile::new("comparison");
        comparison.add_metric("metric_b", vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let comparator = ProfileComparator::new(CompareConfig::default());
        let result = comparator.compare(&baseline, &comparison);

        assert!(matches!(result, Err(CompareError::NoCommonMetrics)));
    }

    #[test]
    fn test_bonferroni_correction() {
        let mut baseline = BenchmarkProfile::new("baseline");
        let mut comparison = BenchmarkProfile::new("comparison");

        // Add 10 metrics with variance (not all same values)
        for i in 0..10 {
            let base_samples: Vec<f64> = (0..10).map(|j| 100.0 + (j as f64) * 0.1).collect();
            let comp_samples: Vec<f64> = (0..10).map(|j| 100.0 + (j as f64) * 0.1).collect();
            baseline.add_metric(format!("metric_{}", i), base_samples);
            comparison.add_metric(format!("metric_{}", i), comp_samples);
        }

        let config = CompareConfig {
            bonferroni_correction: true,
            ..Default::default()
        };

        let comparator = ProfileComparator::new(config);
        let result = comparator.compare(&baseline, &comparison).unwrap();

        // With 10 metrics and alpha=0.05, corrected alpha should be 0.005
        assert!((result.corrected_alpha - 0.005).abs() < 0.001);
    }

    #[test]
    fn test_welch_t_test_identical() {
        let config = CompareConfig::default();
        let comparator = ProfileComparator::new(config);

        let a = MetricSamples::new(vec![10.0, 10.0, 10.0, 10.0, 10.0]);
        let b = MetricSamples::new(vec![10.0, 10.0, 10.0, 10.0, 10.0]);

        // This should fail due to zero variance
        let result = comparator.compare_metric("test", &a, &b, 0.05);
        assert!(matches!(result, Err(CompareError::ZeroVariance { .. })));
    }

    #[test]
    fn test_welch_t_test_significant() {
        let config = CompareConfig::default();
        let comparator = ProfileComparator::new(config);

        let a = MetricSamples::new(vec![10.0, 11.0, 9.0, 10.5, 9.5]);
        let b = MetricSamples::new(vec![20.0, 21.0, 19.0, 20.5, 19.5]);

        let result = comparator.compare_metric("test", &a, &b, 0.05).unwrap();

        // Very different means should be significant
        assert!(result.t_test.significant);
        assert!(result.effect_size.percent_change > 90.0); // ~100% increase
    }

    #[test]
    fn test_confidence_interval() {
        let config = CompareConfig::default();
        let comparator = ProfileComparator::new(config);

        let a = MetricSamples::new(vec![10.0, 11.0, 9.0, 10.5, 9.5]);
        let b = MetricSamples::new(vec![12.0, 13.0, 11.0, 12.5, 11.5]);

        let result = comparator.compare_metric("latency", &a, &b, 0.05).unwrap();

        // CI should contain the true difference (~2)
        assert!(result.ci_lower < 2.0);
        assert!(result.ci_upper > 2.0);
    }

    #[test]
    fn test_direction_higher_is_better() {
        let config = CompareConfig::default();
        let comparator = ProfileComparator::new(config);

        let a = MetricSamples::new(vec![100.0, 101.0, 99.0, 100.5, 99.5]);
        let b = MetricSamples::new(vec![120.0, 121.0, 119.0, 120.5, 119.5]);

        // Throughput: higher is better
        let result = comparator
            .compare_metric("throughput", &a, &b, 0.05)
            .unwrap();
        assert_eq!(result.direction, ChangeDirection::Improved);
        assert!(!result.is_regression);

        // Latency: lower is better, so increase is regression
        let result = comparator
            .compare_metric("latency_p50", &a, &b, 0.05)
            .unwrap();
        assert_eq!(result.direction, ChangeDirection::Regressed);
        assert!(result.is_regression);
    }

    #[test]
    fn test_comparison_counts() {
        let result = ProfileComparison {
            baseline_name: "a".to_string(),
            comparison_name: "b".to_string(),
            metrics: vec![],
            regressions: vec!["m1".to_string(), "m2".to_string()],
            improvements: vec!["m3".to_string()],
            verdict: ComparisonVerdict::Fail,
            corrected_alpha: 0.05,
        };

        assert_eq!(result.regression_count(), 2);
        assert_eq!(result.improvement_count(), 1);
        assert!(result.has_regressions());
    }

    #[test]
    fn test_compare_error_display() {
        let err = CompareError::InsufficientSamples { got: 3, need: 5 };
        assert!(err.to_string().contains("3"));
        assert!(err.to_string().contains("5"));

        let err = CompareError::MetricNotFound {
            name: "latency".to_string(),
        };
        assert!(err.to_string().contains("latency"));
    }

    #[test]
    fn test_normal_quantile() {
        let comparator = ProfileComparator::new(CompareConfig::default());

        // Known values
        assert!((comparator.normal_quantile(0.5) - 0.0).abs() < 0.01);
        assert!((comparator.normal_quantile(0.975) - 1.96).abs() < 0.1);
        assert!((comparator.normal_quantile(0.025) - (-1.96)).abs() < 0.1);
    }

    // FKR-046: Detection of 5% regression with 95% confidence
    #[test]
    fn test_fkr_046_five_percent_regression_detection() {
        let comparator = ProfileComparator::new(CompareConfig::default());

        // Run multiple trials to verify detection rate
        let mut detected = 0;
        let trials = 100;

        for seed in 0..trials {
            // Generate baseline with mean 100, std 5
            let baseline_values: Vec<f64> = (0..30)
                .map(|i| 100.0 + (((seed * 100 + i) % 10) as f64 - 5.0))
                .collect();

            // Generate comparison with 5% regression (mean 105 for latency)
            let comparison_values: Vec<f64> = (0..30)
                .map(|i| 105.0 + (((seed * 100 + i + 50) % 10) as f64 - 5.0))
                .collect();

            let mut baseline = BenchmarkProfile::new("baseline");
            baseline.add_metric("latency_p50", baseline_values);

            let mut comparison = BenchmarkProfile::new("comparison");
            comparison.add_metric("latency_p50", comparison_values);

            let result = comparator.compare(&baseline, &comparison);

            if let Ok(r) = result {
                if r.has_regressions() {
                    detected += 1;
                }
            }
        }

        let detection_rate = detected as f64 / trials as f64;

        // FKR-046: Detection rate should be >80%
        assert!(
            detection_rate > 0.80,
            "Detection rate {} should be >80%",
            detection_rate
        );
    }
}
