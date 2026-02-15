//! Context-aware regression predictor with adaptive thresholds.

use std::collections::HashMap;

use super::{
    BaselineEntry, RegressionCheck, RegressionThreshold, SystemContext, Trend,
    DEFAULT_COLD_START_MARGIN, DEFAULT_STALENESS_SEC, MIN_SAMPLES_FOR_CONTEXT,
};

/// Context-aware regression predictor
#[derive(Debug)]
pub struct ContextRegressionPredictor {
    /// Historical baselines by metric
    baselines: HashMap<String, Vec<BaselineEntry>>,
    /// Maximum history size
    max_history: usize,
    /// Cold start margin (%)
    cold_start_margin: f64,
    /// Minimum learned margin (%)
    min_margin: f64,
    /// Context staleness threshold (seconds)
    staleness_sec: u64,
    /// Temperature variance factor (% per 10°C)
    temp_factor: f64,
    /// Memory variance factor (% per 10% utilization)
    memory_factor: f64,
    /// Frequency variance factor (% per 10% reduction)
    freq_factor: f64,
    /// Cache cold penalty (%)
    cache_cold_penalty: f64,
}

impl Default for ContextRegressionPredictor {
    fn default() -> Self {
        Self {
            baselines: HashMap::new(),
            max_history: 100,
            cold_start_margin: DEFAULT_COLD_START_MARGIN,
            min_margin: 3.0,
            staleness_sec: DEFAULT_STALENESS_SEC,
            temp_factor: 2.0,
            memory_factor: 1.0,
            freq_factor: 5.0,
            cache_cold_penalty: 10.0,
        }
    }
}

impl ContextRegressionPredictor {
    /// Create new predictor
    pub fn new() -> Self {
        Self::default()
    }

    /// Set cold start margin
    pub fn with_cold_start_margin(mut self, margin: f64) -> Self {
        self.cold_start_margin = margin.max(5.0);
        self
    }

    /// Set minimum margin
    pub fn with_min_margin(mut self, margin: f64) -> Self {
        self.min_margin = margin.max(1.0);
        self
    }

    /// Set temperature factor
    pub fn with_temp_factor(mut self, factor: f64) -> Self {
        self.temp_factor = factor.max(0.0);
        self
    }

    /// Set staleness threshold
    pub fn with_staleness(mut self, sec: u64) -> Self {
        self.staleness_sec = sec;
        self
    }

    /// Add baseline entry
    pub fn add_baseline(&mut self, metric: &str, value: f64, context: SystemContext) {
        let entry = BaselineEntry::new(metric, value, context);

        self.baselines
            .entry(metric.to_string())
            .or_default()
            .push(entry);

        // Trim old entries
        if let Some(entries) = self.baselines.get_mut(metric) {
            while entries.len() > self.max_history {
                entries.remove(0);
            }
        }
    }

    /// Get baseline count for metric
    pub fn baseline_count(&self, metric: &str) -> usize {
        self.baselines.get(metric).map(|e| e.len()).unwrap_or(0)
    }

    /// Check if sufficient history
    pub fn has_sufficient_history(&self, metric: &str) -> bool {
        self.baseline_count(metric) >= MIN_SAMPLES_FOR_CONTEXT
    }

    /// Compute context-aware threshold
    pub fn compute_threshold(
        &self,
        metric: &str,
        current_context: &SystemContext,
    ) -> RegressionThreshold {
        let sample_count = self.baseline_count(metric);

        // Cold start: use conservative margin
        if sample_count < MIN_SAMPLES_FOR_CONTEXT {
            return RegressionThreshold {
                base_percent: self.cold_start_margin,
                temp_adjustment: 0.0,
                memory_adjustment: 0.0,
                freq_adjustment: 0.0,
                cache_adjustment: 0.0,
                final_percent: self.cold_start_margin,
                confidence: 0.1,
                sample_count,
            };
        }

        let entries = self
            .baselines
            .get(metric)
            .expect("metric should exist in baselines after sufficient history check");

        // Compute base threshold from historical variance
        let values: Vec<f64> = entries.iter().map(|e| e.value).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (values.len() - 1).max(1) as f64;
        let std_dev = variance.sqrt();
        let cv = if mean.abs() > 1e-10 {
            (std_dev / mean.abs()) * 100.0
        } else {
            5.0
        };

        // Base threshold: 2σ + min margin
        let base_percent = (cv * 2.0).max(self.min_margin);

        // Temperature adjustment: warmer = more variance expected
        let avg_temp: f64 =
            entries.iter().map(|e| e.context.cpu_temp_c).sum::<f64>() / entries.len() as f64;
        let temp_diff = current_context.cpu_temp_c - avg_temp;
        let temp_adjustment = (temp_diff / 10.0) * self.temp_factor;

        // Memory adjustment: higher pressure = more variance
        let avg_mem: f64 = entries
            .iter()
            .map(|e| e.context.memory_percent)
            .sum::<f64>()
            / entries.len() as f64;
        let mem_diff = current_context.memory_percent - avg_mem;
        let memory_adjustment = (mem_diff / 10.0).max(0.0) * self.memory_factor;

        // Frequency adjustment: lower frequency = expect slower
        let avg_freq_util: f64 = entries
            .iter()
            .map(|e| e.context.freq_utilization())
            .sum::<f64>()
            / entries.len() as f64;
        let freq_diff = avg_freq_util - current_context.freq_utilization();
        let freq_adjustment = (freq_diff * 10.0).max(0.0) * self.freq_factor;

        // Cache adjustment: cold cache = expect slower
        let cache_adjustment = if !current_context.cache_warm {
            self.cache_cold_penalty
        } else {
            0.0
        };

        // Final threshold
        let final_percent = (base_percent
            + temp_adjustment
            + memory_adjustment
            + freq_adjustment
            + cache_adjustment)
            .max(self.min_margin);

        // Confidence increases with more samples
        let confidence = (sample_count as f64 / 50.0).min(1.0);

        RegressionThreshold {
            base_percent,
            temp_adjustment,
            memory_adjustment,
            freq_adjustment,
            cache_adjustment,
            final_percent,
            confidence,
            sample_count,
        }
    }

    /// Detect trend in baselines
    pub fn detect_trend(&self, metric: &str) -> Option<Trend> {
        let entries = self.baselines.get(metric)?;
        if entries.len() < MIN_SAMPLES_FOR_CONTEXT {
            return None;
        }

        // Simple linear regression: value vs time
        let n = entries.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        let base_time = entries.first()?.context.timestamp;
        for entry in entries {
            let x = (entry.context.timestamp - base_time) as f64 / 86400.0; // days
            let y = entry.value;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }

        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return None;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;

        // Compute R²
        let mean_y = sum_y / n;
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for entry in entries {
            let x = (entry.context.timestamp - base_time) as f64 / 86400.0;
            let y_pred = slope * x + intercept;
            ss_res += (entry.value - y_pred).powi(2);
            ss_tot += (entry.value - mean_y).powi(2);
        }
        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        let direction = if slope > 0.1 {
            "increasing"
        } else if slope < -0.1 {
            "decreasing"
        } else {
            "stable"
        };

        Some(Trend {
            slope_per_day: slope,
            r_squared,
            direction,
        })
    }

    /// Check for regression
    pub fn check_regression(
        &self,
        metric: &str,
        current_value: f64,
        context: &SystemContext,
    ) -> RegressionCheck {
        let threshold = self.compute_threshold(metric, context);

        let entries = self.baselines.get(metric);
        let baseline_mean = entries
            .map(|e| e.iter().map(|x| x.value).sum::<f64>() / e.len() as f64)
            .unwrap_or(current_value);

        let percent_change = if baseline_mean.abs() > 1e-10 {
            ((current_value - baseline_mean) / baseline_mean) * 100.0
        } else {
            0.0
        };

        let is_regression = threshold.is_regression(percent_change);
        let trend = self.detect_trend(metric);

        RegressionCheck {
            metric: metric.to_string(),
            current_value,
            baseline_mean,
            percent_change,
            threshold,
            is_regression,
            trend,
        }
    }

    /// Clear history for metric
    pub fn clear(&mut self, metric: &str) {
        self.baselines.remove(metric);
    }

    /// Clear all history
    pub fn clear_all(&mut self) {
        self.baselines.clear();
    }

    /// Export baselines to JSON
    pub fn export_json(&self, metric: &str) -> Option<String> {
        let entries = self.baselines.get(metric)?;
        let entries_json: Vec<String> = entries
            .iter()
            .map(|e| {
                format!(
                    r#"{{"value":{},"context":{}}}"#,
                    e.value,
                    e.context.to_json()
                )
            })
            .collect();
        Some(format!(
            r#"{{"metric":"{}","entries":[{}]}}"#,
            metric,
            entries_json.join(",")
        ))
    }
}
