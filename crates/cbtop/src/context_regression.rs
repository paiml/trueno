//! Context-Aware Regression Predictor (PMAT-039)
//!
//! Context-aware regression thresholds accounting for system state and historical trends.
//!
//! # Features
//!
//! - Context capture (temperature, memory, frequency)
//! - Adaptive threshold computation based on context
//! - Trend detection from historical data
//! - False positive reduction through learned patterns
//!
//! # Falsification Criteria (F1311-F1320)
//!
//! See `tests/context_regression_f1311.rs` for falsification tests.

use std::collections::HashMap;

/// Default cold start margin (15%)
pub const DEFAULT_COLD_START_MARGIN: f64 = 15.0;

/// Default minimum samples for learning
pub const MIN_SAMPLES_FOR_CONTEXT: usize = 5;

/// Default context staleness (seconds)
pub const DEFAULT_STALENESS_SEC: u64 = 3600;

/// System context snapshot
#[derive(Debug, Clone)]
pub struct SystemContext {
    /// Timestamp (Unix seconds)
    pub timestamp: u64,
    /// CPU temperature (Celsius)
    pub cpu_temp_c: f64,
    /// GPU temperature (Celsius)
    pub gpu_temp_c: f64,
    /// Memory utilization (0-100)
    pub memory_percent: f64,
    /// CPU frequency (MHz)
    pub cpu_freq_mhz: f64,
    /// Maximum CPU frequency (MHz)
    pub cpu_freq_max_mhz: f64,
    /// Is cache warm
    pub cache_warm: bool,
    /// Load average (1 min)
    pub load_average: f64,
}

impl Default for SystemContext {
    fn default() -> Self {
        Self {
            timestamp: 0,
            cpu_temp_c: 50.0,
            gpu_temp_c: 50.0,
            memory_percent: 50.0,
            cpu_freq_mhz: 3000.0,
            cpu_freq_max_mhz: 4000.0,
            cache_warm: false,
            load_average: 1.0,
        }
    }
}

impl SystemContext {
    /// Create new context
    pub fn new() -> Self {
        Self::default()
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }

    /// Set CPU temperature
    pub fn with_cpu_temp(mut self, temp_c: f64) -> Self {
        self.cpu_temp_c = temp_c;
        self
    }

    /// Set GPU temperature
    pub fn with_gpu_temp(mut self, temp_c: f64) -> Self {
        self.gpu_temp_c = temp_c;
        self
    }

    /// Set memory utilization
    pub fn with_memory(mut self, percent: f64) -> Self {
        self.memory_percent = percent.clamp(0.0, 100.0);
        self
    }

    /// Set CPU frequency
    pub fn with_cpu_freq(mut self, freq_mhz: f64, max_mhz: f64) -> Self {
        self.cpu_freq_mhz = freq_mhz;
        self.cpu_freq_max_mhz = max_mhz;
        self
    }

    /// Set cache state
    pub fn with_cache_warm(mut self, warm: bool) -> Self {
        self.cache_warm = warm;
        self
    }

    /// Set load average
    pub fn with_load(mut self, load: f64) -> Self {
        self.load_average = load;
        self
    }

    /// Capture current system context
    pub fn capture() -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // In a real implementation, these would read from sysfs/procfs
        // For now, return reasonable defaults
        Self {
            timestamp,
            cpu_temp_c: 60.0,
            gpu_temp_c: 55.0,
            memory_percent: 50.0,
            cpu_freq_mhz: 3000.0,
            cpu_freq_max_mhz: 4000.0,
            cache_warm: false,
            load_average: 1.0,
        }
    }

    /// Get frequency utilization (0-1)
    pub fn freq_utilization(&self) -> f64 {
        if self.cpu_freq_max_mhz > 0.0 {
            self.cpu_freq_mhz / self.cpu_freq_max_mhz
        } else {
            1.0
        }
    }

    /// Get thermal headroom (degrees below throttle threshold)
    pub fn thermal_headroom(&self, throttle_temp: f64) -> f64 {
        (throttle_temp - self.cpu_temp_c).max(0.0)
    }

    /// Check if context is stale
    pub fn is_stale(&self, max_age_sec: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        now.saturating_sub(self.timestamp) > max_age_sec
    }

    /// Export to JSON
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"timestamp":{},"cpu_temp_c":{},"memory_percent":{},"cpu_freq_mhz":{},"cache_warm":{}}}"#,
            self.timestamp,
            self.cpu_temp_c,
            self.memory_percent,
            self.cpu_freq_mhz,
            self.cache_warm
        )
    }
}

/// Historical baseline entry
#[derive(Debug, Clone)]
pub struct BaselineEntry {
    /// Context at measurement time
    pub context: SystemContext,
    /// Measured value
    pub value: f64,
    /// Metric name
    pub metric: String,
}

impl BaselineEntry {
    /// Create new entry
    pub fn new(metric: &str, value: f64, context: SystemContext) -> Self {
        Self {
            metric: metric.to_string(),
            value,
            context,
        }
    }
}

/// Computed regression threshold
#[derive(Debug, Clone)]
pub struct RegressionThreshold {
    /// Base threshold (from historical mean)
    pub base_percent: f64,
    /// Temperature adjustment
    pub temp_adjustment: f64,
    /// Memory adjustment
    pub memory_adjustment: f64,
    /// Frequency adjustment
    pub freq_adjustment: f64,
    /// Cache adjustment
    pub cache_adjustment: f64,
    /// Final threshold
    pub final_percent: f64,
    /// Confidence in threshold
    pub confidence: f64,
    /// Number of samples used
    pub sample_count: usize,
}

impl RegressionThreshold {
    /// Check if regression detected
    pub fn is_regression(&self, percent_change: f64) -> bool {
        percent_change.abs() > self.final_percent
    }
}

/// Detected trend
#[derive(Debug, Clone)]
pub struct Trend {
    /// Slope (change per day)
    pub slope_per_day: f64,
    /// R² of linear fit
    pub r_squared: f64,
    /// Direction description
    pub direction: &'static str,
}

impl Trend {
    /// Check if trend is significant
    pub fn is_significant(&self) -> bool {
        self.r_squared > 0.5 && self.slope_per_day.abs() > 0.1
    }
}

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
    pub fn compute_threshold(&self, metric: &str, current_context: &SystemContext) -> RegressionThreshold {
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

        let entries = self.baselines.get(metric).unwrap();

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
        let avg_temp: f64 = entries.iter().map(|e| e.context.cpu_temp_c).sum::<f64>()
            / entries.len() as f64;
        let temp_diff = current_context.cpu_temp_c - avg_temp;
        let temp_adjustment = (temp_diff / 10.0) * self.temp_factor;

        // Memory adjustment: higher pressure = more variance
        let avg_mem: f64 = entries.iter().map(|e| e.context.memory_percent).sum::<f64>()
            / entries.len() as f64;
        let mem_diff = current_context.memory_percent - avg_mem;
        let memory_adjustment = (mem_diff / 10.0).max(0.0) * self.memory_factor;

        // Frequency adjustment: lower frequency = expect slower
        let avg_freq_util: f64 = entries.iter().map(|e| e.context.freq_utilization()).sum::<f64>()
            / entries.len() as f64;
        let freq_diff = avg_freq_util - current_context.freq_utilization();
        let freq_adjustment = (freq_diff * 10.0).max(0.0) * self.freq_factor;

        // Cache adjustment: cold cache = expect slower
        let cache_adjustment = if !current_context.cache_warm { self.cache_cold_penalty } else { 0.0 };

        // Final threshold
        let final_percent = (base_percent + temp_adjustment + memory_adjustment + freq_adjustment + cache_adjustment)
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
        let r_squared = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };

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
    pub fn check_regression(&self, metric: &str, current_value: f64, context: &SystemContext) -> RegressionCheck {
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
            .map(|e| format!(r#"{{"value":{},"context":{}}}"#, e.value, e.context.to_json()))
            .collect();
        Some(format!(r#"{{"metric":"{}","entries":[{}]}}"#, metric, entries_json.join(",")))
    }
}

/// Result of regression check
#[derive(Debug, Clone)]
pub struct RegressionCheck {
    /// Metric name
    pub metric: String,
    /// Current value
    pub current_value: f64,
    /// Baseline mean
    pub baseline_mean: f64,
    /// Percent change from baseline
    pub percent_change: f64,
    /// Computed threshold
    pub threshold: RegressionThreshold,
    /// Is regression detected
    pub is_regression: bool,
    /// Detected trend
    pub trend: Option<Trend>,
}

impl RegressionCheck {
    /// Check if passed (no regression)
    pub fn passed(&self) -> bool {
        !self.is_regression
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_context() {
        let ctx = SystemContext::new()
            .with_cpu_temp(70.0)
            .with_memory(60.0)
            .with_cpu_freq(3500.0, 4000.0);

        assert_eq!(ctx.cpu_temp_c, 70.0);
        assert_eq!(ctx.memory_percent, 60.0);
        assert!((ctx.freq_utilization() - 0.875).abs() < 0.001);
    }

    #[test]
    fn test_context_json() {
        let ctx = SystemContext::new().with_cpu_temp(65.0);
        let json = ctx.to_json();

        assert!(json.contains("\"cpu_temp_c\":65"));
    }

    #[test]
    fn test_predictor_creation() {
        let predictor = ContextRegressionPredictor::new();
        assert_eq!(predictor.baseline_count("test"), 0);
    }

    #[test]
    fn test_cold_start_margin() {
        let predictor = ContextRegressionPredictor::new();
        let ctx = SystemContext::new();

        let threshold = predictor.compute_threshold("test", &ctx);
        assert_eq!(threshold.final_percent, DEFAULT_COLD_START_MARGIN);
        assert!(threshold.confidence < 0.5);
    }

    #[test]
    fn test_learned_threshold() {
        let mut predictor = ContextRegressionPredictor::new();

        // Add baseline entries
        for i in 0..10 {
            let ctx = SystemContext::new().with_timestamp(i as u64 * 86400);
            predictor.add_baseline("latency", 100.0 + (i % 3) as f64, ctx);
        }

        let current = SystemContext::new().with_timestamp(100 * 86400);
        let threshold = predictor.compute_threshold("latency", &current);

        assert!(threshold.final_percent < DEFAULT_COLD_START_MARGIN);
        assert!(threshold.confidence > 0.1);
    }

    #[test]
    fn test_regression_check() {
        let mut predictor = ContextRegressionPredictor::new();

        for i in 0..10 {
            let ctx = SystemContext::new();
            predictor.add_baseline("throughput", 1000.0 + (i % 5) as f64, ctx);
        }

        let ctx = SystemContext::new();

        // No regression
        let check = predictor.check_regression("throughput", 1002.0, &ctx);
        assert!(!check.is_regression);

        // Clear regression (50% drop)
        let check = predictor.check_regression("throughput", 500.0, &ctx);
        assert!(check.is_regression);
    }

    #[test]
    fn test_trend_detection() {
        let mut predictor = ContextRegressionPredictor::new();

        // Add increasing trend
        for i in 0..20 {
            let ctx = SystemContext::new().with_timestamp(i as u64 * 86400);
            predictor.add_baseline("metric", 100.0 + i as f64 * 2.0, ctx);
        }

        let trend = predictor.detect_trend("metric").unwrap();
        assert!(trend.slope_per_day > 0.0);
        assert_eq!(trend.direction, "increasing");
    }
}
