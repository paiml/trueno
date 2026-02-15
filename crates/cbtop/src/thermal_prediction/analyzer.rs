//! Thermal trend analyzer with sliding window.

use std::collections::VecDeque;

use super::types::{
    CooldownRecommendation, ThermalCorrelation, ThermalPrediction, ThermalSample,
    ThermalVariance, ThrottleRisk,
};
use super::{DEFAULT_THROTTLE_THRESHOLD_C, MIN_SAMPLES_FOR_ANALYSIS};

/// Thermal trend analyzer with sliding window
#[derive(Debug)]
pub struct ThermalAnalyzer {
    /// Sample buffer (sliding window)
    samples: VecDeque<ThermalSample>,
    /// Maximum buffer size
    max_samples: usize,
    /// Throttle threshold temperature
    throttle_threshold_c: f64,
    /// Default cooling rate (degrees C/sec) for recommendations
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

        // Calculate R-squared for confidence
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

    /// Calculate R-squared (coefficient of determination)
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

        // Target is 10 degrees C below threshold for safety margin
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
            // R-squared gives proportion of variance explained
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
