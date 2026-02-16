//! Thermal diagnostic methods: risk assessment, cooldown, correlation, variance.

use super::regression;
use super::types::{
    CooldownRecommendation, ThermalCorrelation, ThermalVariance, ThrottleRisk,
};
use super::MIN_SAMPLES_FOR_ANALYSIS;
use super::ThermalAnalyzer;

impl ThermalAnalyzer {
    /// Calculate throttle risk
    pub fn throttle_risk(&self) -> Option<ThrottleRisk> {
        let current_temp = self.current_temperature()?;
        let trend_slope = self.calculate_trend().unwrap_or(0.0);

        Some(ThrottleRisk::assess(
            current_temp,
            self.throttle_threshold_c,
            trend_slope,
            10.0,
        ))
    }

    /// Get recommended cooldown
    pub fn recommended_cooldown(&self) -> Option<CooldownRecommendation> {
        let current_temp = self.current_temperature()?;
        let target_temp = self.throttle_threshold_c - 10.0;

        if current_temp <= target_temp {
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
        let paired: Vec<(f64, f64)> = self
            .samples
            .iter()
            .filter_map(|s| s.latency_us.map(|l| (s.temperature_c, l)))
            .collect();

        if paired.len() < MIN_SAMPLES_FOR_ANALYSIS {
            return None;
        }

        let (pearson_r, latency_per_degree) = regression::pearson_r(&paired)?;
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

        let contribution = if let Some(corr) = self.correlation_to_latency() {
            (corr.pearson_r.powi(2) * 100.0).clamp(0.0, 100.0)
        } else {
            (temp_range / 10.0 * 20.0).clamp(0.0, 50.0)
        };

        Some(ThermalVariance {
            contribution_percent: contribution,
            temp_range_c: temp_range,
            avg_temp_c: avg_temp,
        })
    }
}
