//! Statistical Analysis Module (PMAT-024)
//!
//! Implements statistical analysis per F221 for 95% nonparametric confidence
//! intervals, effect size calculation, and bootstrap sampling.
//!
//! # Components
//!
//! | Component | Formula | Use Case |
//! |-----------|---------|----------|
//! | Bootstrap CI | Resampling with replacement | Nonparametric 95% CI |
//! | Cohen's d | (M1-M2) / pooled_std | Effect size magnitude |
//! | Welch's t-test | t-statistic with unequal variances | A/B comparison |
//! | Mann-Whitney U | Nonparametric rank test | Non-normal distributions |
//! | IQR Outlier Filter | Q1 - 1.5×IQR to Q3 + 1.5×IQR | Robust statistics |
//!
//! # Citations
//!
//! - [Efron & Tibshirani 1993] "An Introduction to the Bootstrap"
//! - [Cohen 1988] "Statistical Power Analysis for Behavioral Sciences"
//! - [Hoefler & Belli 2015] "Scientific Benchmarking of Parallel Computing Systems"

use std::cmp::Ordering;

/// Effect size category per Cohen's conventions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectCategory {
    /// |d| < 0.2 - negligible practical significance
    Negligible,
    /// 0.2 ≤ |d| < 0.5 - small effect
    Small,
    /// 0.5 ≤ |d| < 0.8 - medium effect
    Medium,
    /// |d| ≥ 0.8 - large effect
    Large,
}

impl EffectCategory {
    /// Get description of effect category
    pub fn description(&self) -> &'static str {
        match self {
            EffectCategory::Negligible => "negligible practical significance",
            EffectCategory::Small => "small effect",
            EffectCategory::Medium => "medium effect",
            EffectCategory::Large => "large effect",
        }
    }

    /// Categorize effect size from Cohen's d value
    pub fn from_cohens_d(d: f64) -> Self {
        let abs_d = d.abs();
        if abs_d < 0.2 {
            EffectCategory::Negligible
        } else if abs_d < 0.5 {
            EffectCategory::Small
        } else if abs_d < 0.8 {
            EffectCategory::Medium
        } else {
            EffectCategory::Large
        }
    }
}

/// Statistical analysis result with confidence interval
#[derive(Debug, Clone)]
pub struct StatisticalAnalysis {
    /// Sample mean
    pub mean: f64,
    /// Sample standard deviation
    pub std_dev: f64,
    /// Standard error of the mean
    pub std_error: f64,
    /// Lower bound of confidence interval
    pub ci_lower: f64,
    /// Upper bound of confidence interval
    pub ci_upper: f64,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
    /// Sample size
    pub n: usize,
    /// Coefficient of variation (std_dev / mean * 100)
    pub cv_percent: f64,
}

impl StatisticalAnalysis {
    /// Compute statistical analysis from samples
    pub fn from_samples(samples: &[f64], confidence_level: f64) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        // Filter out NaN/Inf
        let valid: Vec<f64> = samples.iter().copied().filter(|x| x.is_finite()).collect();

        if valid.is_empty() {
            return None;
        }

        let n = valid.len();
        let mean = valid.iter().sum::<f64>() / n as f64;

        if n == 1 {
            return Some(Self {
                mean,
                std_dev: 0.0,
                std_error: 0.0,
                ci_lower: mean,
                ci_upper: mean,
                confidence_level,
                n,
                cv_percent: 0.0,
            });
        }

        let variance = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std_dev = variance.sqrt();
        let std_error = std_dev / (n as f64).sqrt();

        // Bootstrap confidence interval
        let (ci_lower, ci_upper) = bootstrap_ci(&valid, confidence_level, 10000);

        let cv_percent = if mean != 0.0 {
            (std_dev / mean.abs()) * 100.0
        } else {
            0.0
        };

        Some(Self {
            mean,
            std_dev,
            std_error,
            ci_lower,
            ci_upper,
            confidence_level,
            n,
            cv_percent,
        })
    }

    /// Compute with default 95% confidence level
    pub fn from_samples_default(samples: &[f64]) -> Option<Self> {
        Self::from_samples(samples, 0.95)
    }

    /// Get CI width
    pub fn ci_width(&self) -> f64 {
        self.ci_upper - self.ci_lower
    }

    /// Check if CI is narrow (< 10% of mean)
    pub fn ci_is_narrow(&self) -> bool {
        if self.mean == 0.0 {
            return self.ci_width() < 0.1;
        }
        (self.ci_width() / self.mean.abs()) < 0.1
    }
}

/// Effect size calculation result
#[derive(Debug, Clone)]
pub struct EffectSize {
    /// Cohen's d value
    pub cohens_d: f64,
    /// Effect category
    pub category: EffectCategory,
    /// 95% CI lower for effect size
    pub ci_lower: f64,
    /// 95% CI upper for effect size
    pub ci_upper: f64,
}

impl EffectSize {
    /// Calculate Cohen's d between two samples
    pub fn cohens_d(sample1: &[f64], sample2: &[f64]) -> Option<Self> {
        if sample1.is_empty() || sample2.is_empty() {
            return None;
        }

        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;

        let mean1 = sample1.iter().sum::<f64>() / n1;
        let mean2 = sample2.iter().sum::<f64>() / n2;

        let var1 = sample1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0).max(1.0);
        let var2 = sample2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0).max(1.0);

        // Pooled standard deviation
        let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0).max(1.0);
        let pooled_std = pooled_var.sqrt();

        if pooled_std == 0.0 {
            return Some(Self {
                cohens_d: 0.0,
                category: EffectCategory::Negligible,
                ci_lower: 0.0,
                ci_upper: 0.0,
            });
        }

        let d = (mean1 - mean2) / pooled_std;
        let category = EffectCategory::from_cohens_d(d);

        // Approximate 95% CI for Cohen's d using non-central t approximation
        let se_d = ((n1 + n2) / (n1 * n2) + d.powi(2) / (2.0 * (n1 + n2))).sqrt();
        let ci_lower = d - 1.96 * se_d;
        let ci_upper = d + 1.96 * se_d;

        Some(Self {
            cohens_d: d,
            category,
            ci_lower,
            ci_upper,
        })
    }

    /// Check if effect is practically significant
    pub fn is_significant(&self) -> bool {
        self.category != EffectCategory::Negligible
    }
}

/// Result of statistical comparison between two samples
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Welch's t-statistic
    pub t_statistic: f64,
    /// Two-tailed p-value
    pub p_value: f64,
    /// Effect size
    pub effect_size: EffectSize,
    /// Whether difference is statistically significant (p < 0.05)
    pub statistically_significant: bool,
    /// Whether difference is practically significant (|d| >= 0.2)
    pub practically_significant: bool,
    /// Degrees of freedom (Welch-Satterthwaite)
    pub degrees_of_freedom: f64,
}

impl ComparisonResult {
    /// Perform Welch's t-test between two samples
    pub fn welch_t_test(sample1: &[f64], sample2: &[f64]) -> Option<Self> {
        if sample1.len() < 2 || sample2.len() < 2 {
            return None;
        }

        let n1 = sample1.len() as f64;
        let n2 = sample2.len() as f64;

        let mean1 = sample1.iter().sum::<f64>() / n1;
        let mean2 = sample2.iter().sum::<f64>() / n2;

        let var1 = sample1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
        let var2 = sample2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

        let se1 = var1 / n1;
        let se2 = var2 / n2;
        let se_diff = (se1 + se2).sqrt();

        if se_diff == 0.0 {
            return None;
        }

        let t = (mean1 - mean2) / se_diff;

        // Welch-Satterthwaite degrees of freedom
        let df = (se1 + se2).powi(2) / (se1.powi(2) / (n1 - 1.0) + se2.powi(2) / (n2 - 1.0));

        // Approximate p-value using normal distribution for large df
        let p_value = 2.0 * (1.0 - normal_cdf(t.abs()));

        let effect_size = EffectSize::cohens_d(sample1, sample2)?;

        Some(Self {
            t_statistic: t,
            p_value,
            effect_size: effect_size.clone(),
            statistically_significant: p_value < 0.05,
            practically_significant: effect_size.is_significant(),
            degrees_of_freedom: df,
        })
    }

    /// Both statistically and practically significant
    pub fn is_meaningful(&self) -> bool {
        self.statistically_significant && self.practically_significant
    }
}

/// Mann-Whitney U test result (nonparametric)
#[derive(Debug, Clone)]
pub struct MannWhitneyResult {
    /// U statistic
    pub u_statistic: f64,
    /// Approximate p-value (normal approximation)
    pub p_value: f64,
    /// Rank-biserial correlation (effect size)
    pub effect_size: f64,
    /// Whether difference is significant (p < 0.05)
    pub significant: bool,
}

impl MannWhitneyResult {
    /// Perform Mann-Whitney U test between two samples
    pub fn test(sample1: &[f64], sample2: &[f64]) -> Option<Self> {
        if sample1.is_empty() || sample2.is_empty() {
            return None;
        }

        let n1 = sample1.len();
        let n2 = sample2.len();

        // Combine and rank
        let mut combined: Vec<(f64, usize)> = sample1
            .iter()
            .map(|&x| (x, 0))
            .chain(sample2.iter().map(|&x| (x, 1)))
            .collect();

        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Assign ranks (handling ties with average rank)
        let mut ranks: Vec<f64> = vec![0.0; combined.len()];
        let mut i = 0;
        while i < combined.len() {
            let mut j = i;
            while j < combined.len() && combined[j].0 == combined[i].0 {
                j += 1;
            }
            let avg_rank = (i + j + 1) as f64 / 2.0;
            for k in i..j {
                ranks[k] = avg_rank;
            }
            i = j;
        }

        // Sum of ranks for sample 1
        let r1: f64 = combined
            .iter()
            .enumerate()
            .filter(|(_, (_, group))| *group == 0)
            .map(|(idx, _)| ranks[idx])
            .sum();

        // U statistics
        let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
        let u2 = (n1 * n2) as f64 - u1;
        let u = u1.min(u2);

        // Normal approximation for p-value
        let mean_u = (n1 * n2) as f64 / 2.0;
        let std_u = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();

        let z = if std_u > 0.0 {
            (u - mean_u) / std_u
        } else {
            0.0
        };
        let p_value = 2.0 * (1.0 - normal_cdf(z.abs()));

        // Rank-biserial correlation as effect size
        let effect_size = 1.0 - (2.0 * u) / (n1 * n2) as f64;

        Some(Self {
            u_statistic: u,
            p_value,
            effect_size,
            significant: p_value < 0.05,
        })
    }
}

/// IQR-based outlier filter
#[derive(Debug, Clone)]
pub struct OutlierFilter {
    /// Lower fence (Q1 - 1.5*IQR)
    pub lower_fence: f64,
    /// Upper fence (Q3 + 1.5*IQR)
    pub upper_fence: f64,
    /// Q1 (25th percentile)
    pub q1: f64,
    /// Q3 (75th percentile)
    pub q3: f64,
    /// IQR (Q3 - Q1)
    pub iqr: f64,
    /// Multiplier (default 1.5)
    pub multiplier: f64,
}

impl OutlierFilter {
    /// Create outlier filter from samples
    pub fn new(samples: &[f64]) -> Option<Self> {
        Self::with_multiplier(samples, 1.5)
    }

    /// Create outlier filter with custom multiplier
    pub fn with_multiplier(samples: &[f64], multiplier: f64) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let mut sorted: Vec<f64> = samples.iter().copied().filter(|x| x.is_finite()).collect();

        if sorted.is_empty() {
            return None;
        }

        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let q1 = percentile(&sorted, 0.25);
        let q3 = percentile(&sorted, 0.75);
        let iqr = q3 - q1;

        let lower_fence = q1 - multiplier * iqr;
        let upper_fence = q3 + multiplier * iqr;

        Some(Self {
            lower_fence,
            upper_fence,
            q1,
            q3,
            iqr,
            multiplier,
        })
    }

    /// Check if value is an outlier
    pub fn is_outlier(&self, value: f64) -> bool {
        value < self.lower_fence || value > self.upper_fence
    }

    /// Filter outliers from samples
    pub fn filter(&self, samples: &[f64]) -> Vec<f64> {
        samples
            .iter()
            .copied()
            .filter(|&x| !self.is_outlier(x))
            .collect()
    }

    /// Count outliers in samples
    pub fn count_outliers(&self, samples: &[f64]) -> usize {
        samples.iter().filter(|&&x| self.is_outlier(x)).count()
    }
}

/// Bootstrap confidence interval calculation
pub fn bootstrap_ci(samples: &[f64], confidence_level: f64, iterations: usize) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }

    if samples.len() == 1 {
        return (samples[0], samples[0]);
    }

    let n = samples.len();
    let mut bootstrap_means: Vec<f64> = Vec::with_capacity(iterations);

    // Simple LCG for deterministic but varied resampling
    let mut rng_state: u64 = 12345;
    let lcg = |state: &mut u64| -> usize {
        *state = state.wrapping_mul(1103515245).wrapping_add(12345);
        ((*state >> 16) as usize) % n
    };

    for _ in 0..iterations {
        let mut sum = 0.0;
        for _ in 0..n {
            sum += samples[lcg(&mut rng_state)];
        }
        bootstrap_means.push(sum / n as f64);
    }

    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let alpha = 1.0 - confidence_level;
    let lower_idx = ((alpha / 2.0) * iterations as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * iterations as f64) as usize;

    let ci_lower = bootstrap_means
        .get(lower_idx)
        .copied()
        .unwrap_or(bootstrap_means[0]);
    let ci_upper = bootstrap_means
        .get(upper_idx.min(iterations - 1))
        .copied()
        .unwrap_or(*bootstrap_means.last().unwrap());

    (ci_lower, ci_upper)
}

/// Calculate percentile (0.0 to 1.0)
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    // Approximation using error function
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun)
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Calculate robust mean (trimmed mean, removing top/bottom 10%)
pub fn trimmed_mean(samples: &[f64], trim_percent: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }

    let mut sorted: Vec<f64> = samples.iter().copied().filter(|x| x.is_finite()).collect();

    if sorted.is_empty() {
        return None;
    }

    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let trim_count = (sorted.len() as f64 * trim_percent) as usize;
    let start = trim_count;
    let end = sorted.len() - trim_count;

    if start >= end {
        return Some(sorted[sorted.len() / 2]);
    }

    let trimmed = &sorted[start..end];
    Some(trimmed.iter().sum::<f64>() / trimmed.len() as f64)
}


#[cfg(test)]
mod tests;
