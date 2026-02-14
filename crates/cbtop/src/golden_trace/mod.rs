//! Golden Trace Comparison (PMAT-029)
//!
//! Capture and compare performance traces against golden baselines for regression detection.
//!
//! # Features
//!
//! - Capture golden performance traces
//! - Compare current metrics against baseline
//! - Detect regressions (>10% deviation)
//! - Export traces for review
//!
//! # Falsification Criteria (F1211-F1220)
//!
//! See `tests/golden_trace_f1211.rs` for falsification tests.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Golden trace error
#[derive(Debug, Clone, PartialEq)]
pub enum GoldenTraceError {
    /// No golden trace exists
    NoBaseline,
    /// IO error
    IoError(String),
    /// Parse error
    ParseError(String),
    /// Invalid trace data
    InvalidTrace(String),
    /// Version mismatch
    VersionMismatch { expected: String, actual: String },
}

impl std::fmt::Display for GoldenTraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoBaseline => write!(f, "No golden baseline exists"),
            Self::IoError(msg) => write!(f, "IO error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::InvalidTrace(msg) => write!(f, "Invalid trace: {}", msg),
            Self::VersionMismatch { expected, actual } => {
                write!(f, "Version mismatch: expected {}, got {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for GoldenTraceError {}

/// Result type for golden trace operations
pub type GoldenTraceResult<T> = Result<T, GoldenTraceError>;

/// Syscall breakdown for trace comparison
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SyscallBreakdown {
    /// Read syscall count
    pub read_count: u64,
    /// Write syscall count
    pub write_count: u64,
    /// Mmap syscall count
    pub mmap_count: u64,
    /// Futex syscall count
    pub futex_count: u64,
    /// Other syscall count
    pub other_count: u64,
}

impl SyscallBreakdown {
    /// Create new breakdown
    pub fn new() -> Self {
        Self::default()
    }

    /// Total syscall count
    pub fn total(&self) -> u64 {
        self.read_count + self.write_count + self.mmap_count + self.futex_count + self.other_count
    }

    /// Calculate percentage difference from baseline
    pub fn percentage_diff(&self, baseline: &SyscallBreakdown) -> SyscallBreakdownDelta {
        SyscallBreakdownDelta {
            read_delta: Self::calc_delta(self.read_count, baseline.read_count),
            write_delta: Self::calc_delta(self.write_count, baseline.write_count),
            mmap_delta: Self::calc_delta(self.mmap_count, baseline.mmap_count),
            futex_delta: Self::calc_delta(self.futex_count, baseline.futex_count),
            other_delta: Self::calc_delta(self.other_count, baseline.other_count),
            total_delta: Self::calc_delta(self.total(), baseline.total()),
        }
    }

    fn calc_delta(current: u64, baseline: u64) -> f64 {
        if baseline == 0 {
            if current == 0 {
                0.0
            } else {
                100.0 // New syscalls appeared
            }
        } else {
            ((current as f64 - baseline as f64) / baseline as f64) * 100.0
        }
    }
}

/// Syscall breakdown delta (percentage changes)
#[derive(Debug, Clone)]
pub struct SyscallBreakdownDelta {
    /// Read syscall delta %
    pub read_delta: f64,
    /// Write syscall delta %
    pub write_delta: f64,
    /// Mmap syscall delta %
    pub mmap_delta: f64,
    /// Futex syscall delta %
    pub futex_delta: f64,
    /// Other syscall delta %
    pub other_delta: f64,
    /// Total syscall delta %
    pub total_delta: f64,
}

impl SyscallBreakdownDelta {
    /// Get maximum absolute delta
    pub fn max_delta(&self) -> f64 {
        self.read_delta
            .abs()
            .max(self.write_delta.abs())
            .max(self.mmap_delta.abs())
            .max(self.futex_delta.abs())
            .max(self.other_delta.abs())
    }

    /// Check if any delta exceeds threshold
    pub fn exceeds_threshold(&self, threshold_percent: f64) -> bool {
        self.read_delta.abs() > threshold_percent
            || self.write_delta.abs() > threshold_percent
            || self.mmap_delta.abs() > threshold_percent
            || self.futex_delta.abs() > threshold_percent
            || self.other_delta.abs() > threshold_percent
    }
}

/// Performance metrics for trace comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMetrics {
    /// Total execution time in microseconds
    pub total_time_us: f64,
    /// P50 latency in microseconds
    pub p50_latency_us: f64,
    /// P99 latency in microseconds
    pub p99_latency_us: f64,
    /// Throughput (ops/sec)
    pub throughput: f64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,
    /// Syscall breakdown
    pub syscalls: SyscallBreakdown,
    /// Custom metrics
    #[serde(default)]
    pub custom: HashMap<String, f64>,
}

impl Default for TraceMetrics {
    fn default() -> Self {
        Self {
            total_time_us: 0.0,
            p50_latency_us: 0.0,
            p99_latency_us: 0.0,
            throughput: 0.0,
            peak_memory_bytes: 0,
            syscalls: SyscallBreakdown::default(),
            custom: HashMap::new(),
        }
    }
}

impl TraceMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set total time
    pub fn total_time_us(mut self, us: f64) -> Self {
        self.total_time_us = us;
        self
    }

    /// Builder: set P50 latency
    pub fn p50_latency_us(mut self, us: f64) -> Self {
        self.p50_latency_us = us;
        self
    }

    /// Builder: set P99 latency
    pub fn p99_latency_us(mut self, us: f64) -> Self {
        self.p99_latency_us = us;
        self
    }

    /// Builder: set throughput
    pub fn throughput(mut self, ops_per_sec: f64) -> Self {
        self.throughput = ops_per_sec;
        self
    }

    /// Builder: set peak memory
    pub fn peak_memory_bytes(mut self, bytes: u64) -> Self {
        self.peak_memory_bytes = bytes;
        self
    }

    /// Builder: set syscall breakdown
    pub fn syscalls(mut self, syscalls: SyscallBreakdown) -> Self {
        self.syscalls = syscalls;
        self
    }

    /// Builder: add custom metric
    pub fn with_custom(mut self, key: &str, value: f64) -> Self {
        self.custom.insert(key.to_string(), value);
        self
    }

    /// Check if metrics are valid
    pub fn is_valid(&self) -> bool {
        self.total_time_us >= 0.0
            && self.p50_latency_us >= 0.0
            && self.p99_latency_us >= 0.0
            && self.throughput >= 0.0
            && !self.total_time_us.is_nan()
            && !self.p50_latency_us.is_nan()
            && !self.p99_latency_us.is_nan()
            && !self.throughput.is_nan()
    }
}

/// Golden trace containing baseline metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenTrace {
    /// Trace version (for compatibility checking)
    pub version: String,
    /// Trace name/identifier
    pub name: String,
    /// Capture timestamp (Unix epoch seconds)
    pub timestamp: u64,
    /// Git commit hash (if available)
    #[serde(default)]
    pub git_commit: Option<String>,
    /// Metrics captured
    pub metrics: TraceMetrics,
    /// Hash of the trace for verification
    #[serde(default)]
    pub hash: String,
    /// Environment info
    #[serde(default)]
    pub environment: HashMap<String, String>,
}

impl GoldenTrace {
    /// Create a new golden trace
    pub fn new(name: &str, metrics: TraceMetrics) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();

        let mut trace = Self {
            version: "1.0".to_string(),
            name: name.to_string(),
            timestamp,
            git_commit: None,
            metrics,
            hash: String::new(),
            environment: HashMap::new(),
        };

        trace.hash = trace.compute_hash();
        trace
    }

    /// Create with specific version
    pub fn with_version(name: &str, version: &str, metrics: TraceMetrics) -> Self {
        let mut trace = Self::new(name, metrics);
        trace.version = version.to_string();
        trace.hash = trace.compute_hash();
        trace
    }

    /// Set git commit
    pub fn git_commit(mut self, commit: &str) -> Self {
        self.git_commit = Some(commit.to_string());
        self.hash = self.compute_hash();
        self
    }

    /// Add environment info
    pub fn with_env(mut self, key: &str, value: &str) -> Self {
        self.environment.insert(key.to_string(), value.to_string());
        self
    }

    /// Compute deterministic hash of trace
    pub fn compute_hash(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.name.hash(&mut hasher);
        self.version.hash(&mut hasher);
        self.timestamp.hash(&mut hasher);
        (self.metrics.total_time_us as u64).hash(&mut hasher);
        (self.metrics.p50_latency_us as u64).hash(&mut hasher);
        (self.metrics.p99_latency_us as u64).hash(&mut hasher);
        (self.metrics.throughput as u64).hash(&mut hasher);
        self.metrics.peak_memory_bytes.hash(&mut hasher);

        format!("{:016x}", hasher.finish())
    }

    /// Verify hash matches
    pub fn verify_hash(&self) -> bool {
        self.hash == self.compute_hash()
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> GoldenTraceResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| GoldenTraceError::ParseError(e.to_string()))
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> GoldenTraceResult<Self> {
        serde_json::from_str(json).map_err(|e| GoldenTraceError::ParseError(e.to_string()))
    }

    /// Serialize to TOML
    pub fn to_toml(&self) -> GoldenTraceResult<String> {
        toml::to_string_pretty(self).map_err(|e| GoldenTraceError::ParseError(e.to_string()))
    }

    /// Deserialize from TOML
    pub fn from_toml(toml_str: &str) -> GoldenTraceResult<Self> {
        toml::from_str(toml_str).map_err(|e| GoldenTraceError::ParseError(e.to_string()))
    }

    /// Save to file (auto-detects format from extension)
    pub fn save(&self, path: &Path) -> GoldenTraceResult<()> {
        let content = if path.extension().map_or(false, |ext| ext == "json") {
            self.to_json()?
        } else {
            self.to_toml()?
        };

        std::fs::write(path, content).map_err(|e| GoldenTraceError::IoError(e.to_string()))
    }

    /// Load from file (auto-detects format from extension)
    pub fn load(path: &Path) -> GoldenTraceResult<Self> {
        if !path.exists() {
            return Err(GoldenTraceError::NoBaseline);
        }

        let content =
            std::fs::read_to_string(path).map_err(|e| GoldenTraceError::IoError(e.to_string()))?;

        if path.extension().map_or(false, |ext| ext == "json") {
            Self::from_json(&content)
        } else {
            Self::from_toml(&content)
        }
    }
}

/// Comparison result between current and golden trace
#[derive(Debug, Clone)]
pub struct TraceComparison {
    /// Golden trace name
    pub golden_name: String,
    /// Golden trace version
    pub golden_version: String,
    /// Total time delta (percentage)
    pub time_delta_percent: f64,
    /// P50 latency delta (percentage)
    pub p50_delta_percent: f64,
    /// P99 latency delta (percentage)
    pub p99_delta_percent: f64,
    /// Throughput delta (percentage)
    pub throughput_delta_percent: f64,
    /// Memory delta (percentage)
    pub memory_delta_percent: f64,
    /// Syscall breakdown delta
    pub syscall_delta: SyscallBreakdownDelta,
    /// Is this a regression?
    pub is_regression: bool,
    /// Regression threshold used (percentage)
    pub threshold_percent: f64,
    /// Summary message
    pub summary: String,
}

impl TraceComparison {
    /// Get maximum positive delta (regression)
    pub fn max_regression(&self) -> f64 {
        self.time_delta_percent
            .max(self.p50_delta_percent)
            .max(self.p99_delta_percent)
            .max(-self.throughput_delta_percent) // Negative throughput is regression
            .max(self.memory_delta_percent)
    }

    /// Get all regressions above threshold
    pub fn regressions(&self) -> Vec<(String, f64)> {
        let mut regressions = Vec::new();
        let threshold = self.threshold_percent;

        if self.time_delta_percent > threshold {
            regressions.push(("total_time".to_string(), self.time_delta_percent));
        }
        if self.p50_delta_percent > threshold {
            regressions.push(("p50_latency".to_string(), self.p50_delta_percent));
        }
        if self.p99_delta_percent > threshold {
            regressions.push(("p99_latency".to_string(), self.p99_delta_percent));
        }
        if -self.throughput_delta_percent > threshold {
            regressions.push(("throughput".to_string(), self.throughput_delta_percent));
        }
        if self.memory_delta_percent > threshold {
            regressions.push(("memory".to_string(), self.memory_delta_percent));
        }

        regressions
    }
}

/// Golden trace comparator
#[derive(Debug, Clone)]
pub struct GoldenComparator {
    /// Regression threshold (default 10%)
    pub threshold_percent: f64,
    /// Allow version mismatch
    pub allow_version_mismatch: bool,
}

impl Default for GoldenComparator {
    fn default() -> Self {
        Self {
            threshold_percent: 10.0,
            allow_version_mismatch: false,
        }
    }
}

impl GoldenComparator {
    /// Create new comparator
    pub fn new() -> Self {
        Self::default()
    }

    /// Set threshold
    pub fn with_threshold(mut self, percent: f64) -> Self {
        self.threshold_percent = percent;
        self
    }

    /// Allow version mismatch
    pub fn allow_version_mismatch(mut self) -> Self {
        self.allow_version_mismatch = true;
        self
    }

    /// Compare current metrics against golden trace
    pub fn compare(
        &self,
        current: &TraceMetrics,
        golden: &GoldenTrace,
    ) -> GoldenTraceResult<TraceComparison> {
        if !current.is_valid() {
            return Err(GoldenTraceError::InvalidTrace(
                "Current metrics are invalid".to_string(),
            ));
        }

        if !golden.metrics.is_valid() {
            return Err(GoldenTraceError::InvalidTrace(
                "Golden metrics are invalid".to_string(),
            ));
        }

        let time_delta = Self::calc_delta(current.total_time_us, golden.metrics.total_time_us);
        let p50_delta = Self::calc_delta(current.p50_latency_us, golden.metrics.p50_latency_us);
        let p99_delta = Self::calc_delta(current.p99_latency_us, golden.metrics.p99_latency_us);
        let throughput_delta = Self::calc_delta(current.throughput, golden.metrics.throughput);
        let memory_delta = Self::calc_delta(
            current.peak_memory_bytes as f64,
            golden.metrics.peak_memory_bytes as f64,
        );

        let syscall_delta = current.syscalls.percentage_diff(&golden.metrics.syscalls);

        // Regression is when metrics get worse
        // For time/latency/memory: higher is worse
        // For throughput: lower is worse
        let is_regression = time_delta > self.threshold_percent
            || p50_delta > self.threshold_percent
            || p99_delta > self.threshold_percent
            || -throughput_delta > self.threshold_percent
            || memory_delta > self.threshold_percent;

        let summary = if is_regression {
            format!(
                "REGRESSION detected vs {}: max delta {:.1}%",
                golden.name,
                time_delta
                    .max(p50_delta)
                    .max(p99_delta)
                    .max(-throughput_delta)
                    .max(memory_delta)
            )
        } else {
            format!(
                "No regression vs {}: within {:.1}% threshold",
                golden.name, self.threshold_percent
            )
        };

        Ok(TraceComparison {
            golden_name: golden.name.clone(),
            golden_version: golden.version.clone(),
            time_delta_percent: time_delta,
            p50_delta_percent: p50_delta,
            p99_delta_percent: p99_delta,
            throughput_delta_percent: throughput_delta,
            memory_delta_percent: memory_delta,
            syscall_delta,
            is_regression,
            threshold_percent: self.threshold_percent,
            summary,
        })
    }

    fn calc_delta(current: f64, baseline: f64) -> f64 {
        if baseline == 0.0 {
            if current == 0.0 {
                0.0
            } else {
                100.0
            }
        } else {
            ((current - baseline) / baseline) * 100.0
        }
    }
}

/// Golden trace manager for storing multiple versions
#[derive(Debug, Clone)]
pub struct GoldenTraceManager {
    /// Storage directory
    storage_dir: std::path::PathBuf,
    /// Cached traces
    cache: HashMap<String, GoldenTrace>,
}

impl GoldenTraceManager {
    /// Create new manager
    pub fn new(storage_dir: std::path::PathBuf) -> Self {
        Self {
            storage_dir,
            cache: HashMap::new(),
        }
    }

    /// Ensure storage directory exists
    pub fn ensure_directory(&self) -> GoldenTraceResult<()> {
        if !self.storage_dir.exists() {
            std::fs::create_dir_all(&self.storage_dir)
                .map_err(|e| GoldenTraceError::IoError(e.to_string()))?;
        }
        Ok(())
    }

    /// Capture current metrics as golden trace
    pub fn capture_golden(&mut self, name: &str, metrics: TraceMetrics) -> GoldenTraceResult<()> {
        self.ensure_directory()?;

        let trace = GoldenTrace::new(name, metrics);
        let path = self.storage_dir.join(format!("{}.toml", name));
        trace.save(&path)?;

        self.cache.insert(name.to_string(), trace);
        Ok(())
    }

    /// Load golden trace by name
    pub fn load_golden(&mut self, name: &str) -> GoldenTraceResult<GoldenTrace> {
        if let Some(cached) = self.cache.get(name) {
            return Ok(cached.clone());
        }

        let path = self.storage_dir.join(format!("{}.toml", name));
        let trace = GoldenTrace::load(&path)?;
        self.cache.insert(name.to_string(), trace.clone());
        Ok(trace)
    }

    /// Compare current metrics to golden
    pub fn compare_to_golden(
        &mut self,
        name: &str,
        current: &TraceMetrics,
    ) -> GoldenTraceResult<TraceComparison> {
        let golden = self.load_golden(name)?;
        let comparator = GoldenComparator::new();
        comparator.compare(current, &golden)
    }

    /// Detect regression against golden
    pub fn detect_regression(
        &mut self,
        name: &str,
        current: &TraceMetrics,
        threshold_percent: f64,
    ) -> GoldenTraceResult<bool> {
        let golden = self.load_golden(name)?;
        let comparator = GoldenComparator::new().with_threshold(threshold_percent);
        let comparison = comparator.compare(current, &golden)?;
        Ok(comparison.is_regression)
    }

    /// List all golden traces
    pub fn list_goldens(&self) -> GoldenTraceResult<Vec<String>> {
        if !self.storage_dir.exists() {
            return Ok(vec![]);
        }

        let entries = std::fs::read_dir(&self.storage_dir)
            .map_err(|e| GoldenTraceError::IoError(e.to_string()))?;

        let mut names = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .extension()
                .map_or(false, |ext| ext == "toml" || ext == "json")
            {
                if let Some(stem) = path.file_stem() {
                    if let Some(name) = stem.to_str() {
                        names.push(name.to_string());
                    }
                }
            }
        }

        names.sort();
        Ok(names)
    }

    /// Export golden trace to path
    pub fn export_trace(&self, name: &str, export_path: &Path) -> GoldenTraceResult<()> {
        let source_path = self.storage_dir.join(format!("{}.toml", name));
        let trace = GoldenTrace::load(&source_path)?;
        trace.save(export_path)
    }

    /// Delete golden trace
    pub fn delete_golden(&mut self, name: &str) -> GoldenTraceResult<()> {
        let path = self.storage_dir.join(format!("{}.toml", name));
        if !path.exists() {
            return Err(GoldenTraceError::NoBaseline);
        }

        std::fs::remove_file(&path).map_err(|e| GoldenTraceError::IoError(e.to_string()))?;
        self.cache.remove(name);
        Ok(())
    }

    /// Check if golden exists
    pub fn golden_exists(&self, name: &str) -> bool {
        self.cache.contains_key(name)
            || self.storage_dir.join(format!("{}.toml", name)).exists()
            || self.storage_dir.join(format!("{}.json", name)).exists()
    }
}


#[cfg(test)]
mod tests;
