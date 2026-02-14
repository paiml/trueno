//! Anomaly Detection Engine (PMAT-034)
//!
//! Automated anomaly detection and outlier classification for performance data.
//!
//! # Features
//!
//! - Z-score outlier detection (>3σ)
//! - IQR-based robust outlier detection
//! - Change point detection for performance cliffs
//! - Anomaly classification and root cause identification
//!
//! # Falsification Criteria (F1261-F1270)
//!
//! See `tests/anomaly_detection_f1261.rs` for falsification tests.

/// Default Z-score threshold for outlier detection
pub const DEFAULT_ZSCORE_THRESHOLD: f64 = 3.0;

/// Default IQR multiplier for outlier detection
pub const DEFAULT_IQR_MULTIPLIER: f64 = 1.5;

/// Minimum samples for statistical analysis
pub const MIN_SAMPLES_FOR_DETECTION: usize = 10;

/// Anomaly severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    /// Informational - minor deviation
    Info,
    /// Warning - notable deviation
    Warning,
    /// Critical - severe deviation requiring attention
    Critical,
}

impl AnomalySeverity {
    /// Get severity name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Critical => "critical",
        }
    }

    /// Get severity from deviation magnitude
    pub fn from_deviation(deviation: f64) -> Self {
        if deviation >= 5.0 {
            Self::Critical
        } else if deviation >= 3.0 {
            Self::Warning
        } else {
            Self::Info
        }
    }
}

/// Anomaly type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    /// Statistical outlier (Z-score based)
    Outlier,
    /// Performance spike (sudden increase)
    Spike,
    /// Performance drop (sudden decrease)
    Drop,
    /// Change point (sustained shift)
    ChangePoint,
    /// Periodic anomaly (recurring pattern)
    Periodic,
    /// Correlated anomaly (multi-metric)
    Correlated,
}

impl AnomalyType {
    /// Get anomaly type name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Outlier => "outlier",
            Self::Spike => "spike",
            Self::Drop => "drop",
            Self::ChangePoint => "change_point",
            Self::Periodic => "periodic",
            Self::Correlated => "correlated",
        }
    }
}

/// Detected anomaly
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Index in the data series
    pub index: usize,
    /// The anomalous value
    pub value: f64,
    /// Expected value (mean or predicted)
    pub expected: f64,
    /// Deviation from expected (in standard deviations)
    pub deviation: f64,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Optional description
    pub description: Option<String>,
}

impl Anomaly {
    /// Create new anomaly
    pub fn new(
        index: usize,
        value: f64,
        expected: f64,
        deviation: f64,
        anomaly_type: AnomalyType,
    ) -> Self {
        Self {
            index,
            value,
            expected,
            deviation,
            anomaly_type,
            severity: AnomalySeverity::from_deviation(deviation.abs()),
            description: None,
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Check if anomaly is critical
    pub fn is_critical(&self) -> bool {
        self.severity == AnomalySeverity::Critical
    }

    /// Export to JSON format
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"index":{},"value":{},"expected":{},"deviation":{},"type":"{}","severity":"{}"}}"#,
            self.index,
            self.value,
            self.expected,
            self.deviation,
            self.anomaly_type.name(),
            self.severity.name()
        )
    }
}

/// Change point in data series
#[derive(Debug, Clone)]
pub struct ChangePoint {
    /// Index where change occurs
    pub index: usize,
    /// Mean before change
    pub mean_before: f64,
    /// Mean after change
    pub mean_after: f64,
    /// Magnitude of change
    pub magnitude: f64,
    /// Direction of change (positive = increase)
    pub direction: f64,
}

impl ChangePoint {
    /// Create new change point
    pub fn new(index: usize, mean_before: f64, mean_after: f64) -> Self {
        let magnitude = (mean_after - mean_before).abs();
        let direction = mean_after - mean_before;
        Self {
            index,
            mean_before,
            mean_after,
            magnitude,
            direction,
        }
    }

    /// Check if change is significant (>10% shift)
    pub fn is_significant(&self) -> bool {
        if self.mean_before.abs() < 1e-10 {
            return self.magnitude > 1e-10;
        }
        (self.magnitude / self.mean_before.abs()) > 0.1
    }
}

/// Anomaly detection result summary
#[derive(Debug, Clone)]
pub struct AnomalyReport {
    /// Total data points analyzed
    pub total_points: usize,
    /// Detected anomalies
    pub anomalies: Vec<Anomaly>,
    /// Detected change points
    pub change_points: Vec<ChangePoint>,
    /// Data mean
    pub mean: f64,
    /// Data standard deviation
    pub std_dev: f64,
    /// Detection method used
    pub method: &'static str,
}

impl AnomalyReport {
    /// Count anomalies by severity
    pub fn count_by_severity(&self, severity: AnomalySeverity) -> usize {
        self.anomalies
            .iter()
            .filter(|a| a.severity == severity)
            .count()
    }

    /// Get critical anomalies
    pub fn critical_anomalies(&self) -> Vec<&Anomaly> {
        self.anomalies.iter().filter(|a| a.is_critical()).collect()
    }

    /// Check if any critical anomalies exist
    pub fn has_critical(&self) -> bool {
        self.anomalies.iter().any(|a| a.is_critical())
    }

    /// Export report to JSON
    pub fn to_json(&self) -> String {
        let anomalies_json: Vec<String> = self.anomalies.iter().map(|a| a.to_json()).collect();
        format!(
            r#"{{"total_points":{},"anomaly_count":{},"critical_count":{},"mean":{},"std_dev":{},"method":"{}","anomalies":[{}]}}"#,
            self.total_points,
            self.anomalies.len(),
            self.count_by_severity(AnomalySeverity::Critical),
            self.mean,
            self.std_dev,
            self.method,
            anomalies_json.join(",")
        )
    }
}

/// Anomaly detector
#[derive(Debug)]
pub struct AnomalyDetector {
    /// Data buffer
    data: Vec<f64>,
    /// Z-score threshold
    zscore_threshold: f64,
    /// IQR multiplier
    iqr_multiplier: f64,
    /// Sliding window size for real-time detection
    window_size: usize,
    /// Detected anomalies
    anomalies: Vec<Anomaly>,
    /// Detected change points
    change_points: Vec<ChangePoint>,
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            zscore_threshold: DEFAULT_ZSCORE_THRESHOLD,
            iqr_multiplier: DEFAULT_IQR_MULTIPLIER,
            window_size: 50,
            anomalies: Vec::new(),
            change_points: Vec::new(),
        }
    }
}

impl AnomalyDetector {
    /// Create new detector
    pub fn new() -> Self {
        Self::default()
    }

    /// Set Z-score threshold
    pub fn with_zscore_threshold(mut self, threshold: f64) -> Self {
        self.zscore_threshold = threshold.max(1.0);
        self
    }

    /// Set IQR multiplier
    pub fn with_iqr_multiplier(mut self, multiplier: f64) -> Self {
        self.iqr_multiplier = multiplier.max(0.5);
        self
    }

    /// Set sliding window size
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size.max(10);
        self
    }

    /// Add data point
    pub fn add(&mut self, value: f64) {
        self.data.push(value);
    }

    /// Add multiple data points
    pub fn add_all(&mut self, values: &[f64]) {
        self.data.extend_from_slice(values);
    }

    /// Get data count
    pub fn data_count(&self) -> usize {
        self.data.len()
    }

    /// Check if sufficient data for analysis
    pub fn has_sufficient_data(&self) -> bool {
        self.data.len() >= MIN_SAMPLES_FOR_DETECTION
    }

    /// Calculate mean of slice
    fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Calculate standard deviation
    fn std_dev(data: &[f64], mean: f64) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let variance =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        variance.sqrt()
    }

    /// Calculate percentile
    fn percentile(sorted_data: &[f64], p: f64) -> f64 {
        if sorted_data.is_empty() {
            return 0.0;
        }
        let idx = (p / 100.0 * (sorted_data.len() - 1) as f64).round() as usize;
        sorted_data[idx.min(sorted_data.len() - 1)]
    }

    /// Detect outliers using Z-score method
    pub fn detect_zscore_outliers(&mut self) -> Vec<Anomaly> {
        if !self.has_sufficient_data() {
            return Vec::new();
        }

        let mean = Self::mean(&self.data);
        let std_dev = Self::std_dev(&self.data, mean);

        if std_dev < 1e-10 {
            return Vec::new(); // No variation
        }

        let mut outliers = Vec::new();
        for (i, &value) in self.data.iter().enumerate() {
            let z_score = (value - mean) / std_dev;
            if z_score.abs() > self.zscore_threshold {
                let anomaly_type = if z_score > 0.0 {
                    AnomalyType::Spike
                } else {
                    AnomalyType::Drop
                };
                outliers.push(Anomaly::new(i, value, mean, z_score, anomaly_type));
            }
        }

        self.anomalies.extend(outliers.clone());
        outliers
    }

    /// Detect outliers using IQR method (robust to heavy tails)
    pub fn detect_iqr_outliers(&mut self) -> Vec<Anomaly> {
        if !self.has_sufficient_data() {
            return Vec::new();
        }

        let mut sorted = self.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1 = Self::percentile(&sorted, 25.0);
        let q3 = Self::percentile(&sorted, 75.0);
        let iqr = q3 - q1;

        if iqr < 1e-10 {
            return Vec::new(); // No variation
        }

        let lower_bound = q1 - self.iqr_multiplier * iqr;
        let upper_bound = q3 + self.iqr_multiplier * iqr;
        let median = Self::percentile(&sorted, 50.0);

        let mut outliers = Vec::new();
        for (i, &value) in self.data.iter().enumerate() {
            if value < lower_bound || value > upper_bound {
                let deviation = if value < lower_bound {
                    (lower_bound - value) / iqr
                } else {
                    (value - upper_bound) / iqr
                };
                let anomaly_type = if value > median {
                    AnomalyType::Spike
                } else {
                    AnomalyType::Drop
                };
                outliers.push(Anomaly::new(i, value, median, deviation, anomaly_type));
            }
        }

        self.anomalies.extend(outliers.clone());
        outliers
    }

    /// Detect change points using CUSUM-like algorithm
    pub fn detect_change_points(&mut self) -> Vec<ChangePoint> {
        if self.data.len() < 20 {
            return Vec::new();
        }

        let overall_mean = Self::mean(&self.data);
        let overall_std = Self::std_dev(&self.data, overall_mean);

        if overall_std < 1e-10 {
            return Vec::new();
        }

        let mut change_points = Vec::new();
        let min_segment = 10;

        // Simple change point detection: find points where mean shifts significantly
        for i in min_segment..(self.data.len() - min_segment) {
            let before = &self.data[..i];
            let after = &self.data[i..];

            let mean_before = Self::mean(before);
            let mean_after = Self::mean(after);

            let change = ChangePoint::new(i, mean_before, mean_after);

            // Check if change is significant (>1 std dev shift)
            if change.magnitude > overall_std {
                change_points.push(change);
            }
        }

        // Filter to keep only most significant change points
        if change_points.len() > 1 {
            change_points.sort_by(|a, b| {
                b.magnitude
                    .partial_cmp(&a.magnitude)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            // Keep top change points
            change_points.truncate(3);
        }

        self.change_points.extend(change_points.clone());
        change_points
    }

    /// Classify anomaly based on context
    pub fn classify_anomaly(&self, anomaly: &Anomaly) -> AnomalyType {
        // Check if it's part of a change point
        for cp in &self.change_points {
            if (anomaly.index as i64 - cp.index as i64).abs() < 5 {
                return AnomalyType::ChangePoint;
            }
        }

        // Check if correlated with other anomalies (cluster)
        let nearby_count = self
            .anomalies
            .iter()
            .filter(|a| {
                (a.index as i64 - anomaly.index as i64).abs() < 3 && a.index != anomaly.index
            })
            .count();

        if nearby_count >= 2 {
            return AnomalyType::Correlated;
        }

        anomaly.anomaly_type
    }

    /// Run all detection methods and generate report
    pub fn analyze(&mut self) -> AnomalyReport {
        self.anomalies.clear();
        self.change_points.clear();

        let zscore_outliers = self.detect_zscore_outliers();
        let _iqr_outliers = self.detect_iqr_outliers();
        let change_points = self.detect_change_points();

        // Deduplicate anomalies by index
        let mut seen = std::collections::HashSet::new();
        self.anomalies.retain(|a| seen.insert(a.index));

        // Sort by severity (critical first)
        self.anomalies.sort_by(|a, b| b.severity.cmp(&a.severity));

        let mean = Self::mean(&self.data);
        let std_dev = Self::std_dev(&self.data, mean);

        AnomalyReport {
            total_points: self.data.len(),
            anomalies: self.anomalies.clone(),
            change_points,
            mean,
            std_dev,
            method: if zscore_outliers.is_empty() {
                "iqr"
            } else {
                "zscore"
            },
        }
    }

    /// Real-time anomaly detection on sliding window
    pub fn detect_realtime(&mut self, new_value: f64) -> Option<Anomaly> {
        self.data.push(new_value);

        if self.data.len() < self.window_size {
            return None;
        }

        // Use sliding window
        let start = self.data.len().saturating_sub(self.window_size);
        let window = &self.data[start..self.data.len() - 1]; // Exclude new value

        let mean = Self::mean(window);
        let std_dev = Self::std_dev(window, mean);

        if std_dev < 1e-10 {
            return None;
        }

        let z_score = (new_value - mean) / std_dev;
        if z_score.abs() > self.zscore_threshold {
            let anomaly_type = if z_score > 0.0 {
                AnomalyType::Spike
            } else {
                AnomalyType::Drop
            };
            let anomaly = Anomaly::new(self.data.len() - 1, new_value, mean, z_score, anomaly_type);
            self.anomalies.push(anomaly.clone());
            return Some(anomaly);
        }

        None
    }

    /// Get all detected anomalies
    pub fn get_anomalies(&self) -> &[Anomaly] {
        &self.anomalies
    }

    /// Get all detected change points
    pub fn get_change_points(&self) -> &[ChangePoint] {
        &self.change_points
    }

    /// Clear all data and detections
    pub fn clear(&mut self) {
        self.data.clear();
        self.anomalies.clear();
        self.change_points.clear();
    }
}


#[cfg(test)]
mod tests;
