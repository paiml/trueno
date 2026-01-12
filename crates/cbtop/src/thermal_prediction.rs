//! Thermal Trend Prediction (PMAT-030)
//!
//! Enhanced thermal analysis with trend prediction, throttle forecasting,
//! and cooldown recommendations.
//!
//! # Features
//!
//! - Temperature trend prediction
//! - Throttle risk calculation
//! - Cooldown time recommendations
//! - Thermal-latency correlation analysis
//!
//! # References
//!
//! - Brooks 2000: "Dynamic Thermal Management for High-Performance Microprocessors" HPCA
//! - Rotem et al. 2012: "Power-Management Architecture of Intel Microarchitectures" IEEE Micro
//!
//! # Falsification Criteria (F1221-F1230)
//!
//! See `tests/thermal_prediction_f1221.rs` for falsification tests.

use std::collections::VecDeque;

/// Default throttle threshold temperature in Celsius
pub const DEFAULT_THROTTLE_THRESHOLD_C: f64 = 85.0;

/// Minimum samples required for analysis
pub const MIN_SAMPLES_FOR_ANALYSIS: usize = 3;

/// Thermal sample with timestamp
#[derive(Debug, Clone, Copy)]
pub struct ThermalSample {
    /// Temperature in Celsius
    pub temperature_c: f64,
    /// Timestamp in seconds (relative to start)
    pub timestamp_sec: f64,
    /// Optional latency measurement at this time
    pub latency_us: Option<f64>,
}

impl ThermalSample {
    /// Create new sample
    pub fn new(temperature_c: f64, timestamp_sec: f64) -> Self {
        Self {
            temperature_c,
            timestamp_sec,
            latency_us: None,
        }
    }

    /// Create sample with latency
    pub fn with_latency(temperature_c: f64, timestamp_sec: f64, latency_us: f64) -> Self {
        Self {
            temperature_c,
            timestamp_sec,
            latency_us: Some(latency_us),
        }
    }
}

/// Thermal trend prediction result
#[derive(Debug, Clone)]
pub struct ThermalPrediction {
    /// Predicted temperature at horizon
    pub predicted_temp_c: f64,
    /// Prediction horizon in seconds
    pub horizon_sec: f64,
    /// Trend slope (°C/second)
    pub trend_slope: f64,
    /// Confidence (0.0 - 1.0) based on R² fit
    pub confidence: f64,
    /// Number of samples used
    pub sample_count: usize,
}

impl ThermalPrediction {
    /// Check if temperature is predicted to exceed threshold
    pub fn will_throttle(&self, threshold_c: f64) -> bool {
        self.predicted_temp_c >= threshold_c
    }

    /// Get time until throttle (if trending up)
    pub fn time_to_throttle(&self, current_temp: f64, threshold_c: f64) -> Option<f64> {
        if self.trend_slope <= 0.0 {
            return None; // Not trending up
        }

        let delta = threshold_c - current_temp;
        if delta <= 0.0 {
            return Some(0.0); // Already at or above threshold
        }

        Some(delta / self.trend_slope)
    }
}

/// Throttle risk assessment
#[derive(Debug, Clone)]
pub struct ThrottleRisk {
    /// Risk probability (0.0 - 1.0)
    pub probability: f64,
    /// Current temperature
    pub current_temp_c: f64,
    /// Throttle threshold
    pub threshold_c: f64,
    /// Temperature margin to threshold
    pub margin_c: f64,
    /// Risk category
    pub category: RiskCategory,
}

impl ThrottleRisk {
    /// Create from temperature and trend
    pub fn assess(
        current_temp: f64,
        threshold: f64,
        trend_slope: f64,
        horizon_sec: f64,
    ) -> Self {
        let margin = threshold - current_temp;
        let predicted_temp = current_temp + trend_slope * horizon_sec;

        // Calculate risk based on:
        // 1. How close we are to threshold
        // 2. Whether we're trending toward it
        // 3. How fast we're approaching it

        let proximity_risk = if margin <= 0.0 {
            1.0
        } else {
            1.0 - (margin / threshold).clamp(0.0, 1.0)
        };

        let trend_risk = if trend_slope > 0.0 {
            // Approaching threshold
            let time_to_threshold = margin / trend_slope;
            if time_to_threshold <= horizon_sec {
                1.0
            } else {
                (horizon_sec / time_to_threshold).clamp(0.0, 1.0)
            }
        } else {
            0.0 // Cooling down
        };

        // Combined risk (weighted average)
        let probability = (0.4 * proximity_risk + 0.6 * trend_risk).clamp(0.0, 1.0);

        let category = RiskCategory::from_probability(probability);

        Self {
            probability,
            current_temp_c: current_temp,
            threshold_c: threshold,
            margin_c: margin,
            category,
        }
    }
}

/// Risk category classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskCategory {
    /// Low risk (< 25%)
    Low,
    /// Moderate risk (25% - 50%)
    Moderate,
    /// High risk (50% - 75%)
    High,
    /// Critical risk (> 75%)
    Critical,
}

impl RiskCategory {
    /// Create from probability
    pub fn from_probability(prob: f64) -> Self {
        if prob < 0.25 {
            Self::Low
        } else if prob < 0.50 {
            Self::Moderate
        } else if prob < 0.75 {
            Self::High
        } else {
            Self::Critical
        }
    }

    /// Get category name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Moderate => "moderate",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }
}

/// Cooldown recommendation
#[derive(Debug, Clone)]
pub struct CooldownRecommendation {
    /// Recommended cooldown time in seconds
    pub duration_sec: f64,
    /// Target temperature after cooldown
    pub target_temp_c: f64,
    /// Current temperature
    pub current_temp_c: f64,
    /// Cooling rate used (°C/second)
    pub cooling_rate: f64,
}

impl CooldownRecommendation {
    /// Calculate cooldown time needed
    pub fn calculate(
        current_temp: f64,
        target_temp: f64,
        cooling_rate: f64,
    ) -> Self {
        let temp_delta = current_temp - target_temp;
        let duration = if temp_delta > 0.0 && cooling_rate > 0.0 {
            temp_delta / cooling_rate
        } else {
            0.0
        };

        Self {
            duration_sec: duration.max(0.0),
            target_temp_c: target_temp,
            current_temp_c: current_temp,
            cooling_rate,
        }
    }

    /// Check if cooldown is needed
    pub fn is_needed(&self) -> bool {
        self.duration_sec > 0.0
    }
}

/// Thermal-latency correlation result
#[derive(Debug, Clone)]
pub struct ThermalCorrelation {
    /// Pearson correlation coefficient (-1.0 to 1.0)
    pub pearson_r: f64,
    /// Number of paired samples
    pub sample_count: usize,
    /// Is correlation significant?
    pub is_significant: bool,
    /// Estimated latency increase per degree (μs/°C)
    pub latency_per_degree: f64,
}

impl ThermalCorrelation {
    /// Check if there's positive correlation (hotter = slower)
    pub fn has_thermal_impact(&self) -> bool {
        self.pearson_r > 0.3 && self.is_significant
    }
}

/// Thermal variance contribution
#[derive(Debug, Clone)]
pub struct ThermalVariance {
    /// Percentage of total variance explained by thermal
    pub contribution_percent: f64,
    /// Temperature range during measurement
    pub temp_range_c: f64,
    /// Average temperature
    pub avg_temp_c: f64,
}

/// Thermal trend analyzer with sliding window
#[derive(Debug)]
pub struct ThermalAnalyzer {
    /// Sample buffer (sliding window)
    samples: VecDeque<ThermalSample>,
    /// Maximum buffer size
    max_samples: usize,
    /// Throttle threshold temperature
    throttle_threshold_c: f64,
    /// Default cooling rate (°C/sec) for recommendations
    default_cooling_rate: f64,
}

impl ThermalAnalyzer {
    /// Create new analyzer
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            throttle_threshold_c: DEFAULT_THROTTLE_THRESHOLD_C,
            default_cooling_rate: 0.5, // Typical passive cooling rate
        }
    }

    /// Set throttle threshold
    pub fn with_threshold(mut self, threshold_c: f64) -> Self {
        self.throttle_threshold_c = threshold_c;
        self
    }

    /// Set cooling rate
    pub fn with_cooling_rate(mut self, rate: f64) -> Self {
        self.default_cooling_rate = rate;
        self
    }

    /// Add a thermal sample
    pub fn add_sample(&mut self, sample: ThermalSample) {
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    /// Add sample from values
    pub fn add(&mut self, temperature_c: f64, timestamp_sec: f64) {
        self.add_sample(ThermalSample::new(temperature_c, timestamp_sec));
    }

    /// Add sample with latency
    pub fn add_with_latency(&mut self, temperature_c: f64, timestamp_sec: f64, latency_us: f64) {
        self.add_sample(ThermalSample::with_latency(
            temperature_c,
            timestamp_sec,
            latency_us,
        ));
    }

    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Check if enough samples for analysis
    pub fn has_sufficient_samples(&self) -> bool {
        self.samples.len() >= MIN_SAMPLES_FOR_ANALYSIS
    }

    /// Get current (latest) temperature
    pub fn current_temperature(&self) -> Option<f64> {
        self.samples.back().map(|s| s.temperature_c)
    }

    /// Get average temperature
    pub fn average_temperature(&self) -> Option<f64> {
        if self.samples.is_empty() {
            return None;
        }

        let sum: f64 = self.samples.iter().map(|s| s.temperature_c).sum();
        Some(sum / self.samples.len() as f64)
    }

    /// Get temperature range
    pub fn temperature_range(&self) -> Option<(f64, f64)> {
        if self.samples.is_empty() {
            return None;
        }

        let min = self
            .samples
            .iter()
            .map(|s| s.temperature_c)
            .fold(f64::INFINITY, f64::min);
        let max = self
            .samples
            .iter()
            .map(|s| s.temperature_c)
            .fold(f64::NEG_INFINITY, f64::max);

        Some((min, max))
    }

    /// Calculate trend slope using linear regression
    pub fn calculate_trend(&self) -> Option<f64> {
        if !self.has_sufficient_samples() {
            return None;
        }

        let n = self.samples.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for sample in &self.samples {
            sum_x += sample.timestamp_sec;
            sum_y += sample.temperature_c;
            sum_xy += sample.timestamp_sec * sample.temperature_c;
            sum_xx += sample.timestamp_sec * sample.timestamp_sec;
        }

        let denominator = n * sum_xx - sum_x * sum_x;
        if denominator.abs() < 1e-10 {
            return Some(0.0); // No time variation
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denominator;
        Some(slope)
    }

    /// Predict temperature at future time
    pub fn predict_trend(&self, horizon_sec: f64) -> Option<ThermalPrediction> {
        if !self.has_sufficient_samples() {
            return None;
        }

        let trend_slope = self.calculate_trend()?;
        let current_temp = self.current_temperature()?;

        // Calculate R² for confidence
        let predicted_temp = current_temp + trend_slope * horizon_sec;
        let confidence = self.calculate_r_squared(trend_slope);

        Some(ThermalPrediction {
            predicted_temp_c: predicted_temp,
            horizon_sec,
            trend_slope,
            confidence,
            sample_count: self.samples.len(),
        })
    }

    /// Calculate R² (coefficient of determination)
    fn calculate_r_squared(&self, slope: f64) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }

        let n = self.samples.len() as f64;
        let mean_y: f64 = self.samples.iter().map(|s| s.temperature_c).sum::<f64>() / n;
        let mean_x: f64 = self.samples.iter().map(|s| s.timestamp_sec).sum::<f64>() / n;

        // Intercept
        let intercept = mean_y - slope * mean_x;

        // Calculate SS_res and SS_tot
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;

        for sample in &self.samples {
            let y_pred = intercept + slope * sample.timestamp_sec;
            ss_res += (sample.temperature_c - y_pred).powi(2);
            ss_tot += (sample.temperature_c - mean_y).powi(2);
        }

        if ss_tot < 1e-10 {
            return 1.0; // Perfect fit (no variation)
        }

        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    }

    /// Calculate throttle risk
    pub fn throttle_risk(&self) -> Option<ThrottleRisk> {
        let current_temp = self.current_temperature()?;
        let trend_slope = self.calculate_trend().unwrap_or(0.0);

        Some(ThrottleRisk::assess(
            current_temp,
            self.throttle_threshold_c,
            trend_slope,
            10.0, // Default 10 second horizon
        ))
    }

    /// Get recommended cooldown
    pub fn recommended_cooldown(&self) -> Option<CooldownRecommendation> {
        let current_temp = self.current_temperature()?;

        // Target is 10°C below threshold for safety margin
        let target_temp = self.throttle_threshold_c - 10.0;

        if current_temp <= target_temp {
            // No cooldown needed
            return Some(CooldownRecommendation {
                duration_sec: 0.0,
                target_temp_c: target_temp,
                current_temp_c: current_temp,
                cooling_rate: self.default_cooling_rate,
            });
        }

        Some(CooldownRecommendation::calculate(
            current_temp,
            target_temp,
            self.default_cooling_rate,
        ))
    }

    /// Calculate thermal-latency correlation
    pub fn correlation_to_latency(&self) -> Option<ThermalCorrelation> {
        // Filter samples that have latency
        let paired: Vec<(f64, f64)> = self
            .samples
            .iter()
            .filter_map(|s| s.latency_us.map(|l| (s.temperature_c, l)))
            .collect();

        if paired.len() < MIN_SAMPLES_FOR_ANALYSIS {
            return None;
        }

        let n = paired.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_yy = 0.0;

        for &(temp, latency) in &paired {
            sum_x += temp;
            sum_y += latency;
            sum_xy += temp * latency;
            sum_xx += temp * temp;
            sum_yy += latency * latency;
        }

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();

        let pearson_r = if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        };

        // Calculate slope for latency_per_degree
        let slope_denom = n * sum_xx - sum_x * sum_x;
        let latency_per_degree = if slope_denom.abs() < 1e-10 {
            0.0
        } else {
            (n * sum_xy - sum_x * sum_y) / slope_denom
        };

        // Significance: |r| > 0.3 and n > 5
        let is_significant = pearson_r.abs() > 0.3 && paired.len() > 5;

        Some(ThermalCorrelation {
            pearson_r,
            sample_count: paired.len(),
            is_significant,
            latency_per_degree,
        })
    }

    /// Calculate thermal variance contribution
    pub fn thermal_variance(&self) -> Option<ThermalVariance> {
        if !self.has_sufficient_samples() {
            return None;
        }

        let avg_temp = self.average_temperature()?;
        let (min_temp, max_temp) = self.temperature_range()?;
        let temp_range = max_temp - min_temp;

        // Calculate thermal contribution to variance
        // Based on correlation if latency data available
        let contribution = if let Some(corr) = self.correlation_to_latency() {
            // R² gives proportion of variance explained
            (corr.pearson_r.powi(2) * 100.0).clamp(0.0, 100.0)
        } else {
            // Estimate based on temperature range
            // Higher range = likely more thermal impact
            (temp_range / 10.0 * 20.0).clamp(0.0, 50.0)
        };

        Some(ThermalVariance {
            contribution_percent: contribution,
            temp_range_c: temp_range,
            avg_temp_c: avg_temp,
        })
    }

    /// Clear all samples
    pub fn clear(&mut self) {
        self.samples.clear();
    }

    /// Get all samples (for export)
    pub fn samples(&self) -> &VecDeque<ThermalSample> {
        &self.samples
    }
}

impl Default for ThermalAnalyzer {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Convenience function to analyze thermal data
pub fn analyze_thermal(
    samples: &[(f64, f64)], // (temp, timestamp) pairs
    horizon_sec: f64,
) -> Option<ThermalPrediction> {
    let mut analyzer = ThermalAnalyzer::new(samples.len());
    for &(temp, time) in samples {
        analyzer.add(temp, time);
    }
    analyzer.predict_trend(horizon_sec)
}

/// Convenience function to assess throttle risk
pub fn assess_throttle_risk(
    current_temp: f64,
    threshold: f64,
    trend_slope: f64,
) -> ThrottleRisk {
    ThrottleRisk::assess(current_temp, threshold, trend_slope, 10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_sample() {
        let sample = ThermalSample::new(65.0, 1.0);
        assert_eq!(sample.temperature_c, 65.0);
        assert!(sample.latency_us.is_none());

        let sample_with_latency = ThermalSample::with_latency(70.0, 2.0, 100.0);
        assert_eq!(sample_with_latency.latency_us, Some(100.0));
    }

    #[test]
    fn test_analyzer_basic() {
        let mut analyzer = ThermalAnalyzer::new(10);

        analyzer.add(60.0, 0.0);
        analyzer.add(65.0, 1.0);
        analyzer.add(70.0, 2.0);

        assert_eq!(analyzer.sample_count(), 3);
        assert!(analyzer.has_sufficient_samples());
        assert_eq!(analyzer.current_temperature(), Some(70.0));
    }

    #[test]
    fn test_trend_calculation() {
        let mut analyzer = ThermalAnalyzer::new(10);

        // Linear increase: 5°C/sec
        analyzer.add(60.0, 0.0);
        analyzer.add(65.0, 1.0);
        analyzer.add(70.0, 2.0);
        analyzer.add(75.0, 3.0);

        let trend = analyzer.calculate_trend().unwrap();
        assert!((trend - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_constant_temperature() {
        let mut analyzer = ThermalAnalyzer::new(10);

        analyzer.add(70.0, 0.0);
        analyzer.add(70.0, 1.0);
        analyzer.add(70.0, 2.0);

        let trend = analyzer.calculate_trend().unwrap();
        assert!(trend.abs() < 0.1); // No trend
    }

    #[test]
    fn test_prediction() {
        let mut analyzer = ThermalAnalyzer::new(10);

        // 5°C/sec increase
        analyzer.add(60.0, 0.0);
        analyzer.add(65.0, 1.0);
        analyzer.add(70.0, 2.0);

        let prediction = analyzer.predict_trend(10.0).unwrap();

        // At t=12, should be 70 + 5*10 = 120°C
        assert!((prediction.predicted_temp_c - 120.0).abs() < 1.0);
        assert!((prediction.trend_slope - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_throttle_risk() {
        let risk = ThrottleRisk::assess(80.0, 85.0, 1.0, 10.0);

        // Close to threshold and trending up
        assert!(risk.probability > 0.5);
        assert_eq!(risk.margin_c, 5.0);
    }

    #[test]
    fn test_risk_category() {
        assert_eq!(RiskCategory::from_probability(0.1), RiskCategory::Low);
        assert_eq!(RiskCategory::from_probability(0.3), RiskCategory::Moderate);
        assert_eq!(RiskCategory::from_probability(0.6), RiskCategory::High);
        assert_eq!(RiskCategory::from_probability(0.9), RiskCategory::Critical);
    }

    #[test]
    fn test_cooldown_recommendation() {
        let cooldown = CooldownRecommendation::calculate(
            90.0,  // Current temp
            75.0,  // Target temp
            0.5,   // Cooling rate
        );

        // Need to cool 15°C at 0.5°C/sec = 30 seconds
        assert!((cooldown.duration_sec - 30.0).abs() < 0.1);
        assert!(cooldown.is_needed());
    }

    #[test]
    fn test_no_cooldown_needed() {
        let cooldown = CooldownRecommendation::calculate(
            70.0,  // Current temp
            75.0,  // Target temp (already below target)
            0.5,
        );

        assert_eq!(cooldown.duration_sec, 0.0);
        assert!(!cooldown.is_needed());
    }

    #[test]
    fn test_thermal_correlation() {
        let mut analyzer = ThermalAnalyzer::new(10);

        // Positive correlation: higher temp = higher latency
        analyzer.add_with_latency(60.0, 0.0, 100.0);
        analyzer.add_with_latency(65.0, 1.0, 110.0);
        analyzer.add_with_latency(70.0, 2.0, 120.0);
        analyzer.add_with_latency(75.0, 3.0, 130.0);
        analyzer.add_with_latency(80.0, 4.0, 140.0);
        analyzer.add_with_latency(85.0, 5.0, 150.0);

        let corr = analyzer.correlation_to_latency().unwrap();

        assert!(corr.pearson_r > 0.9); // Strong positive correlation
        assert!(corr.is_significant);
        assert!(corr.has_thermal_impact());
    }

    #[test]
    fn test_insufficient_samples() {
        let mut analyzer = ThermalAnalyzer::new(10);

        analyzer.add(60.0, 0.0);
        analyzer.add(65.0, 1.0);

        assert!(!analyzer.has_sufficient_samples());
        assert!(analyzer.predict_trend(10.0).is_none());
    }

    #[test]
    fn test_sliding_window() {
        let mut analyzer = ThermalAnalyzer::new(3);

        analyzer.add(60.0, 0.0);
        analyzer.add(65.0, 1.0);
        analyzer.add(70.0, 2.0);
        assert_eq!(analyzer.sample_count(), 3);

        analyzer.add(75.0, 3.0);
        assert_eq!(analyzer.sample_count(), 3); // Still 3

        // Oldest sample (60.0) should be gone
        let (min, _) = analyzer.temperature_range().unwrap();
        assert_eq!(min, 65.0);
    }
}
