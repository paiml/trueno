//! Latency Distribution Analysis Module (PMAT-026)
//!
//! Enhanced latency distribution analysis with tail latency detection,
//! jitter calculation, and histogram statistics for identifying performance anomalies.
//!
//! # Motivation
//!
//! While PMAT-024 provides confidence intervals, detailed latency distribution
//! analysis is needed for:
//! - Detecting bimodal distributions indicating cache misses
//! - Identifying P99.9 tail latency spikes
//! - Calculating jitter (latency variance) for stability assessment
//! - Histogram bucket analysis for distribution shape
//!
//! # Components
//!
//! | Component | Formula | Use Case |
//! |-----------|---------|----------|
//! | Jitter (IPDV) | std_dev(\|latency[i] - latency[i-1]\|) | Connection stability |
//! | Tail Ratio | P99/P50 | Tail latency severity |
//! | Bimodality Coefficient | (skewness² + 1) / kurtosis | Distribution shape |
//! | Histogram Entropy | -Σ(p × log(p)) | Distribution uniformity |

use crate::statistics::percentile;

/// Latency distribution analysis result
#[derive(Debug, Clone)]
pub struct LatencyDistribution {
    /// Median latency (50th percentile)
    pub p50: f64,
    /// 90th percentile latency
    pub p90: f64,
    /// 99th percentile latency
    pub p99: f64,
    /// 99.9th percentile latency
    pub p999: f64,
    /// Jitter (inter-packet delay variation)
    pub jitter: f64,
    /// Tail ratio (P99/P50) - higher means worse tail latency
    pub tail_ratio: f64,
    /// Bimodality coefficient - >0.555 suggests bimodal distribution
    pub bimodality_coefficient: f64,
    /// Latency histogram
    pub histogram: LatencyHistogram,
    /// Number of samples
    pub sample_count: usize,
    /// Minimum latency
    pub min: f64,
    /// Maximum latency
    pub max: f64,
    /// Mean latency
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Skewness (asymmetry)
    pub skewness: f64,
    /// Kurtosis (tail heaviness)
    pub kurtosis: f64,
    /// Outlier ratio (% beyond 3σ)
    pub outlier_ratio: f64,
}

/// Histogram bucket for latency distribution
#[derive(Debug, Clone)]
pub struct HistogramBucket {
    /// Lower bound of bucket (inclusive)
    pub lower: f64,
    /// Upper bound of bucket (exclusive)
    pub upper: f64,
    /// Count of samples in bucket
    pub count: usize,
    /// Percentage of total samples
    pub percentage: f64,
}

/// Latency histogram with statistical properties
#[derive(Debug, Clone)]
pub struct LatencyHistogram {
    /// Histogram buckets
    pub buckets: Vec<HistogramBucket>,
    /// Total number of samples
    pub total_samples: usize,
    /// Shannon entropy (0-1 normalized)
    pub entropy: f64,
    /// Index of the mode bucket (most frequent)
    pub mode_bucket: usize,
    /// Number of buckets
    pub bucket_count: usize,
}

/// Tail latency severity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailSeverity {
    /// Excellent: P99/P50 < 2
    Excellent,
    /// Good: P99/P50 < 3
    Good,
    /// Warning: P99/P50 < 5
    Warning,
    /// Critical: P99/P50 >= 5
    Critical,
}

impl TailSeverity {
    /// Classify based on tail ratio
    pub fn from_ratio(ratio: f64) -> Self {
        if ratio < 2.0 {
            TailSeverity::Excellent
        } else if ratio < 3.0 {
            TailSeverity::Good
        } else if ratio < 5.0 {
            TailSeverity::Warning
        } else {
            TailSeverity::Critical
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            TailSeverity::Excellent => "excellent",
            TailSeverity::Good => "good",
            TailSeverity::Warning => "warning",
            TailSeverity::Critical => "critical",
        }
    }
}

/// Distribution shape classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributionShape {
    /// Unimodal (single peak)
    Unimodal,
    /// Bimodal (two peaks) - often indicates cache hit/miss
    Bimodal,
    /// Multimodal (multiple peaks)
    Multimodal,
    /// Uniform (flat)
    Uniform,
}

impl DistributionShape {
    /// Classify based on bimodality coefficient and entropy
    pub fn classify(bimodality_coeff: f64, entropy: f64) -> Self {
        if entropy > 0.95 {
            DistributionShape::Uniform
        } else if bimodality_coeff > 0.555 {
            DistributionShape::Bimodal
        } else if bimodality_coeff > 0.7 {
            DistributionShape::Multimodal
        } else {
            DistributionShape::Unimodal
        }
    }

    /// Get name
    pub fn name(&self) -> &'static str {
        match self {
            DistributionShape::Unimodal => "unimodal",
            DistributionShape::Bimodal => "bimodal",
            DistributionShape::Multimodal => "multimodal",
            DistributionShape::Uniform => "uniform",
        }
    }
}

impl LatencyDistribution {
    /// Analyze latency samples to produce distribution statistics
    pub fn analyze(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let n = samples.len();

        // Basic statistics
        let mean = samples.iter().sum::<f64>() / n as f64;
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Standard deviation
        let variance = if n > 1 {
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        // Percentiles
        let p50 = percentile(samples, 0.50);
        let p90 = percentile(samples, 0.90);
        let p99 = percentile(samples, 0.99);
        let p999 = percentile(samples, 0.999);

        // Jitter (inter-sample delay variation)
        let jitter = calculate_jitter(samples);

        // Tail ratio
        let tail_ratio = if p50 > 0.0 { p99 / p50 } else { 1.0 };

        // Higher moments (skewness, kurtosis)
        let (skewness, kurtosis) = calculate_moments(samples, mean, std_dev);

        // Bimodality coefficient: (skewness² + 1) / kurtosis
        // For normal distribution: BC ≈ 0.333
        // BC > 0.555 suggests bimodal distribution
        let bimodality_coefficient = if kurtosis > 0.0 {
            (skewness.powi(2) + 1.0) / kurtosis
        } else {
            0.0
        };

        // Outlier ratio (beyond 3σ)
        let outlier_count = samples
            .iter()
            .filter(|&&x| (x - mean).abs() > 3.0 * std_dev)
            .count();
        let outlier_ratio = outlier_count as f64 / n as f64 * 100.0;

        // Build histogram
        let histogram = LatencyHistogram::build(samples, 20);

        Some(Self {
            p50,
            p90,
            p99,
            p999,
            jitter,
            tail_ratio,
            bimodality_coefficient,
            histogram,
            sample_count: n,
            min,
            max,
            mean,
            std_dev,
            skewness,
            kurtosis,
            outlier_ratio,
        })
    }

    /// Classify tail latency severity
    pub fn tail_severity(&self) -> TailSeverity {
        TailSeverity::from_ratio(self.tail_ratio)
    }

    /// Classify distribution shape
    pub fn distribution_shape(&self) -> DistributionShape {
        DistributionShape::classify(self.bimodality_coefficient, self.histogram.entropy)
    }

    /// Check if tail latency is problematic (P99/P50 > 3)
    pub fn has_tail_problem(&self) -> bool {
        self.tail_ratio > 3.0
    }

    /// Check if distribution appears bimodal
    pub fn is_bimodal(&self) -> bool {
        self.bimodality_coefficient > 0.555
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "n={} p50={:.2}µs p99={:.2}µs tail_ratio={:.2} jitter={:.2}µs shape={}",
            self.sample_count,
            self.p50,
            self.p99,
            self.tail_ratio,
            self.jitter,
            self.distribution_shape().name()
        )
    }
}

impl LatencyHistogram {
    /// Build histogram from samples with specified bucket count
    pub fn build(samples: &[f64], bucket_count: usize) -> Self {
        if samples.is_empty() || bucket_count == 0 {
            return Self {
                buckets: Vec::new(),
                total_samples: 0,
                entropy: 0.0,
                mode_bucket: 0,
                bucket_count: 0,
            };
        }

        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Handle case where all values are identical
        let range = max - min;
        let bucket_width = if range > 0.0 {
            range / bucket_count as f64
        } else {
            1.0
        };

        // Initialize buckets
        let mut buckets: Vec<HistogramBucket> = (0..bucket_count)
            .map(|i| {
                let lower = min + i as f64 * bucket_width;
                let upper = if i == bucket_count - 1 {
                    max + f64::EPSILON // Include max in last bucket
                } else {
                    min + (i + 1) as f64 * bucket_width
                };
                HistogramBucket {
                    lower,
                    upper,
                    count: 0,
                    percentage: 0.0,
                }
            })
            .collect();

        // Count samples per bucket
        for &sample in samples {
            let bucket_idx = if range > 0.0 {
                ((sample - min) / bucket_width).floor() as usize
            } else {
                0
            };
            let idx = bucket_idx.min(bucket_count - 1);
            buckets[idx].count += 1;
        }

        // Calculate percentages
        let total = samples.len();
        for bucket in &mut buckets {
            bucket.percentage = bucket.count as f64 / total as f64 * 100.0;
        }

        // Find mode bucket
        let mode_bucket = buckets
            .iter()
            .enumerate()
            .max_by_key(|(_, b)| b.count)
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Calculate entropy
        let entropy = calculate_entropy(&buckets, total);

        Self {
            buckets,
            total_samples: total,
            entropy,
            mode_bucket,
            bucket_count,
        }
    }

    /// Get the mode (most frequent) bucket
    pub fn mode(&self) -> Option<&HistogramBucket> {
        self.buckets.get(self.mode_bucket)
    }

    /// Verify bucket counts sum to total
    pub fn verify_counts(&self) -> bool {
        let sum: usize = self.buckets.iter().map(|b| b.count).sum();
        sum == self.total_samples
    }
}

/// Calculate jitter (inter-packet delay variation)
fn calculate_jitter(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }

    // Calculate differences between consecutive samples
    let diffs: Vec<f64> = samples.windows(2).map(|w| (w[1] - w[0]).abs()).collect();

    if diffs.is_empty() {
        return 0.0;
    }

    // Jitter is the standard deviation of the differences
    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let variance = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / diffs.len() as f64;

    variance.sqrt()
}

/// Calculate skewness and kurtosis
fn calculate_moments(samples: &[f64], mean: f64, std_dev: f64) -> (f64, f64) {
    if samples.len() < 4 || std_dev == 0.0 {
        return (0.0, 3.0); // Return normal distribution values
    }

    let n = samples.len() as f64;

    // Calculate third and fourth moments
    let mut m3 = 0.0;
    let mut m4 = 0.0;

    for &x in samples {
        let z = (x - mean) / std_dev;
        m3 += z.powi(3);
        m4 += z.powi(4);
    }

    // Sample skewness (Fisher's definition)
    let skewness = (n / ((n - 1.0) * (n - 2.0))) * m3;

    // Sample excess kurtosis (Fisher's definition)
    let kurtosis = ((n * (n + 1.0)) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * m4
        - (3.0 * (n - 1.0).powi(2)) / ((n - 2.0) * (n - 3.0));

    // Return kurtosis + 3 (excess kurtosis + 3 = kurtosis)
    (skewness, kurtosis + 3.0)
}

/// Calculate Shannon entropy of histogram (normalized 0-1)
fn calculate_entropy(buckets: &[HistogramBucket], total: usize) -> f64 {
    if total == 0 || buckets.is_empty() {
        return 0.0;
    }

    let mut entropy = 0.0;
    for bucket in buckets {
        if bucket.count > 0 {
            let p = bucket.count as f64 / total as f64;
            entropy -= p * p.ln();
        }
    }

    // Normalize by max entropy (uniform distribution)
    let max_entropy = (buckets.len() as f64).ln();
    if max_entropy > 0.0 {
        entropy / max_entropy
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_distribution_basic() {
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let dist = LatencyDistribution::analyze(&samples).unwrap();

        assert_eq!(dist.sample_count, 100);
        // Median of 1-100 is ~50.5
        assert!((dist.p50 - 50.5).abs() < 2.0);
        // P99 of 1-100 is ~99
        assert!((dist.p99 - 99.0).abs() < 2.0);
        assert!(dist.min == 1.0);
        assert!(dist.max == 100.0);
    }

    #[test]
    fn test_empty_samples() {
        let samples: Vec<f64> = vec![];
        assert!(LatencyDistribution::analyze(&samples).is_none());
    }

    #[test]
    fn test_single_sample() {
        let samples = vec![42.0];
        let dist = LatencyDistribution::analyze(&samples).unwrap();

        assert_eq!(dist.sample_count, 1);
        assert_eq!(dist.jitter, 0.0);
        assert_eq!(dist.p50, 42.0);
    }

    #[test]
    fn test_tail_ratio() {
        // Samples with heavy tail
        let mut samples: Vec<f64> = (1..=95).map(|x| x as f64).collect();
        samples.extend(vec![500.0, 600.0, 700.0, 800.0, 1000.0]); // Tail

        let dist = LatencyDistribution::analyze(&samples).unwrap();
        assert!(dist.tail_ratio > 3.0);
        assert!(dist.has_tail_problem());
    }

    #[test]
    fn test_jitter_calculation() {
        // Stable samples (zero jitter)
        let stable: Vec<f64> = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let dist = LatencyDistribution::analyze(&stable).unwrap();
        assert_eq!(dist.jitter, 0.0);

        // Unstable samples (high jitter)
        let unstable: Vec<f64> = vec![10.0, 50.0, 10.0, 50.0, 10.0, 50.0];
        let dist2 = LatencyDistribution::analyze(&unstable).unwrap();
        assert!(dist2.jitter >= dist.jitter);
    }

    #[test]
    fn test_histogram_counts() {
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let hist = LatencyHistogram::build(&samples, 10);

        assert!(hist.verify_counts());
        assert_eq!(hist.total_samples, 100);
    }

    #[test]
    fn test_entropy_uniform() {
        // Uniform distribution should have high entropy
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let hist = LatencyHistogram::build(&samples, 10);

        // Should be close to 1.0 for uniform
        assert!(hist.entropy > 0.9);
    }

    #[test]
    fn test_entropy_concentrated() {
        // Concentrated distribution should have low entropy
        let samples: Vec<f64> = vec![50.0; 100];
        let hist = LatencyHistogram::build(&samples, 10);

        // All in one bucket = 0 entropy
        assert!(hist.entropy < 0.1);
    }

    #[test]
    fn test_tail_severity() {
        assert_eq!(TailSeverity::from_ratio(1.5), TailSeverity::Excellent);
        assert_eq!(TailSeverity::from_ratio(2.5), TailSeverity::Good);
        assert_eq!(TailSeverity::from_ratio(4.0), TailSeverity::Warning);
        assert_eq!(TailSeverity::from_ratio(10.0), TailSeverity::Critical);
    }

    #[test]
    fn test_percentile_ordering() {
        let samples: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
        let dist = LatencyDistribution::analyze(&samples).unwrap();

        assert!(dist.p50 <= dist.p90);
        assert!(dist.p90 <= dist.p99);
        assert!(dist.p99 <= dist.p999);
    }
}
