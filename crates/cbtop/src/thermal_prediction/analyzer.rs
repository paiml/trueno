//! Thermal trend analyzer with sliding window.

use std::collections::VecDeque;

use super::regression;
use super::types::{ThermalPrediction, ThermalSample};
use super::{DEFAULT_THROTTLE_THRESHOLD_C, MIN_SAMPLES_FOR_ANALYSIS};

/// Thermal trend analyzer with sliding window
#[derive(Debug)]
pub struct ThermalAnalyzer {
    /// Sample buffer (sliding window)
    pub(super) samples: VecDeque<ThermalSample>,
    /// Maximum buffer size
    max_samples: usize,
    /// Throttle threshold temperature
    pub(super) throttle_threshold_c: f64,
    /// Default cooling rate (degrees C/sec) for recommendations
    pub(super) default_cooling_rate: f64,
}

impl ThermalAnalyzer {
    /// Create new analyzer
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            throttle_threshold_c: DEFAULT_THROTTLE_THRESHOLD_C,
            default_cooling_rate: 0.5,
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

    /// Collect (timestamp, temperature) pairs for regression.
    fn time_temp_pairs(&self) -> Vec<(f64, f64)> {
        self.samples
            .iter()
            .map(|s| (s.timestamp_sec, s.temperature_c))
            .collect()
    }

    /// Calculate trend slope using linear regression
    pub fn calculate_trend(&self) -> Option<f64> {
        if !self.has_sufficient_samples() {
            return None;
        }
        let (slope, _, _) = regression::ols_fit(&self.time_temp_pairs())?;
        Some(slope)
    }

    /// Predict temperature at future time
    pub fn predict_trend(&self, horizon_sec: f64) -> Option<ThermalPrediction> {
        if !self.has_sufficient_samples() {
            return None;
        }
        let (trend_slope, _, confidence) = regression::ols_fit(&self.time_temp_pairs())?;
        let current_temp = self.current_temperature()?;

        Some(ThermalPrediction {
            predicted_temp_c: current_temp + trend_slope * horizon_sec,
            horizon_sec,
            trend_slope,
            confidence,
            sample_count: self.samples.len(),
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
