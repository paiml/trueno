//! Stress Test Mode (TRUENO-SPEC-025)
//!
//! Comprehensive stress testing for CPU, GPU, memory, and PCIe to validate
//! system behavior under load.
//!
//! # CLI Interface
//!
//! ```bash
//! trueno-monitor --stress-test
//! trueno-monitor --stress-test --target cpu
//! trueno-monitor --stress-test --target gpu:0
//! trueno-monitor --stress-test --duration 60s --intensity 0.8
//! trueno-monitor --stress-test --chaos gentle
//! ```
//!
//! # References
//!
//! - [Volkov2008] GPU stress via GEMM workloads
//! - renacer chaos engineering integration

use std::time::{Duration, Instant};

use super::device::DeviceId;

// ============================================================================
// Stress Test Configuration (TRUENO-SPEC-025 Section 7.3)
// ============================================================================

/// Stress test configuration
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    /// Target resource(s) to stress
    pub target: StressTarget,
    /// Test duration
    pub duration: Duration,
    /// Intensity (0.0-1.0, where 1.0 = maximum load)
    pub intensity: f64,
    /// Ramp-up duration (gradual increase)
    pub ramp_up: Duration,
    /// Chaos engineering preset (optional)
    pub chaos_preset: Option<ChaosPreset>,
    /// Whether to collect detailed metrics
    pub collect_metrics: bool,
    /// Whether to export report on completion
    pub export_report: bool,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            target: StressTarget::All,
            duration: Duration::from_secs(60),
            intensity: 1.0,
            ramp_up: Duration::from_secs(5),
            chaos_preset: None,
            collect_metrics: true,
            export_report: true,
        }
    }
}

impl StressTestConfig {
    /// Create a new stress test configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set target
    #[must_use]
    pub fn with_target(mut self, target: StressTarget) -> Self {
        self.target = target;
        self
    }

    /// Set duration
    #[must_use]
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration = duration;
        self
    }

    /// Set intensity
    #[must_use]
    pub fn with_intensity(mut self, intensity: f64) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Set ramp-up duration
    #[must_use]
    pub fn with_ramp_up(mut self, ramp_up: Duration) -> Self {
        self.ramp_up = ramp_up;
        self
    }

    /// Set chaos preset
    #[must_use]
    pub fn with_chaos(mut self, preset: ChaosPreset) -> Self {
        self.chaos_preset = Some(preset);
        self
    }

    /// Parse duration from string (e.g., "60s", "5m", "1h")
    #[must_use]
    pub fn parse_duration(s: &str) -> Option<Duration> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }

        let (num, unit) = s.split_at(s.len() - 1);
        let value: u64 = num.parse().ok()?;

        match unit {
            "s" => Some(Duration::from_secs(value)),
            "m" => Some(Duration::from_secs(value * 60)),
            "h" => Some(Duration::from_secs(value * 3600)),
            _ => None,
        }
    }
}

/// Stress target
#[derive(Debug, Clone, PartialEq)]
pub enum StressTarget {
    /// Stress all resources
    All,
    /// Stress CPU only
    Cpu,
    /// Stress GPU (optionally specific device)
    Gpu(Option<DeviceId>),
    /// Stress memory (RAM + VRAM)
    Memory,
    /// Stress PCIe bandwidth
    Pcie,
    /// Custom combination
    Custom(Vec<StressTarget>),
}

impl StressTarget {
    /// Parse from string (e.g., "cpu", "gpu", "gpu:0", "memory", "pcie")
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_lowercase();

        if s == "all" {
            return Some(Self::All);
        }
        if s == "cpu" {
            return Some(Self::Cpu);
        }
        if s == "memory" {
            return Some(Self::Memory);
        }
        if s == "pcie" {
            return Some(Self::Pcie);
        }
        if s == "gpu" {
            return Some(Self::Gpu(None));
        }
        if let Some(idx_str) = s.strip_prefix("gpu:") {
            let idx: u32 = idx_str.parse().ok()?;
            return Some(Self::Gpu(Some(DeviceId::nvidia(idx))));
        }

        None
    }

    /// Check if target includes CPU
    #[must_use]
    pub fn includes_cpu(&self) -> bool {
        match self {
            Self::All | Self::Cpu => true,
            Self::Custom(targets) => targets.iter().any(|t| t.includes_cpu()),
            _ => false,
        }
    }

    /// Check if target includes GPU
    #[must_use]
    pub fn includes_gpu(&self) -> bool {
        match self {
            Self::All | Self::Gpu(_) => true,
            Self::Custom(targets) => targets.iter().any(|t| t.includes_gpu()),
            _ => false,
        }
    }

    /// Check if target includes memory
    #[must_use]
    pub fn includes_memory(&self) -> bool {
        match self {
            Self::All | Self::Memory => true,
            Self::Custom(targets) => targets.iter().any(|t| t.includes_memory()),
            _ => false,
        }
    }

    /// Check if target includes PCIe
    #[must_use]
    pub fn includes_pcie(&self) -> bool {
        match self {
            Self::All | Self::Pcie => true,
            Self::Custom(targets) => targets.iter().any(|t| t.includes_pcie()),
            _ => false,
        }
    }
}

// ============================================================================
// Chaos Engineering Presets (from renacer)
// ============================================================================

/// Chaos engineering preset
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChaosPreset {
    /// Gentle chaos (minimal impact)
    Gentle,
    /// Moderate chaos
    Moderate,
    /// Aggressive chaos
    Aggressive,
    /// Extreme chaos (maximum stress)
    Extreme,
}

impl ChaosPreset {
    /// Parse from string
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "gentle" => Some(Self::Gentle),
            "moderate" => Some(Self::Moderate),
            "aggressive" => Some(Self::Aggressive),
            "extreme" => Some(Self::Extreme),
            _ => None,
        }
    }

    /// Get memory limit factor (0.0-1.0)
    #[must_use]
    pub fn memory_limit_factor(&self) -> f64 {
        match self {
            Self::Gentle => 0.9,      // 90% of available
            Self::Moderate => 0.75,   // 75%
            Self::Aggressive => 0.5,  // 50%
            Self::Extreme => 0.25,    // 25%
        }
    }

    /// Get CPU throttle factor (0.0-1.0)
    #[must_use]
    pub fn cpu_throttle_factor(&self) -> f64 {
        match self {
            Self::Gentle => 1.0,      // No throttle
            Self::Moderate => 0.9,    // 90%
            Self::Aggressive => 0.7,  // 70%
            Self::Extreme => 0.5,     // 50%
        }
    }

    /// Get network latency injection in milliseconds
    #[must_use]
    pub fn network_latency_ms(&self) -> u32 {
        match self {
            Self::Gentle => 0,
            Self::Moderate => 10,
            Self::Aggressive => 50,
            Self::Extreme => 200,
        }
    }

    /// Get failure injection rate (0.0-1.0)
    #[must_use]
    pub fn failure_rate(&self) -> f64 {
        match self {
            Self::Gentle => 0.0,
            Self::Moderate => 0.01,
            Self::Aggressive => 0.05,
            Self::Extreme => 0.10,
        }
    }
}

// ============================================================================
// Stress Test Metrics
// ============================================================================

/// Stress test metrics collected during execution
#[derive(Debug, Clone, Default)]
pub struct StressMetrics {
    // Peak values
    /// Peak CPU utilization
    pub peak_cpu_utilization: f64,
    /// Peak GPU utilization
    pub peak_gpu_utilization: f64,
    /// Peak memory utilization
    pub peak_memory_utilization: f64,
    /// Peak temperature in Celsius
    pub peak_temperature_c: f64,
    /// Peak power in Watts
    pub peak_power_watts: f64,
    /// Peak PCIe bandwidth in GB/s
    pub peak_pcie_bandwidth_gbps: f64,

    // Throttling events
    /// Number of thermal throttle events
    pub thermal_throttle_count: u32,
    /// Number of power throttle events
    pub power_throttle_count: u32,
    /// Number of memory pressure events
    pub memory_pressure_events: u32,

    // Errors
    /// GPU errors
    pub gpu_errors: Vec<String>,
    /// Memory errors
    pub memory_errors: Vec<String>,
    /// Transfer errors
    pub transfer_errors: Vec<String>,

    // Performance
    /// Baseline FLOPS (before stress)
    pub baseline_flops: f64,
    /// Achieved FLOPS (during stress)
    pub achieved_flops: f64,
    /// Performance degradation percentage
    pub performance_degradation_pct: f64,

    // Timing
    /// Actual duration
    pub duration_actual: Duration,
    /// Number of samples collected
    pub sample_count: u32,
}

impl StressMetrics {
    /// Create new metrics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Update peak values from current sample
    pub fn update_peaks(
        &mut self,
        cpu_util: f64,
        gpu_util: f64,
        mem_util: f64,
        temp_c: f64,
        power_w: f64,
        pcie_gbps: f64,
    ) {
        self.peak_cpu_utilization = self.peak_cpu_utilization.max(cpu_util);
        self.peak_gpu_utilization = self.peak_gpu_utilization.max(gpu_util);
        self.peak_memory_utilization = self.peak_memory_utilization.max(mem_util);
        self.peak_temperature_c = self.peak_temperature_c.max(temp_c);
        self.peak_power_watts = self.peak_power_watts.max(power_w);
        self.peak_pcie_bandwidth_gbps = self.peak_pcie_bandwidth_gbps.max(pcie_gbps);
        self.sample_count += 1;
    }

    /// Record a thermal throttle event
    pub fn record_thermal_throttle(&mut self) {
        self.thermal_throttle_count += 1;
    }

    /// Record a power throttle event
    pub fn record_power_throttle(&mut self) {
        self.power_throttle_count += 1;
    }

    /// Record a memory pressure event
    pub fn record_memory_pressure(&mut self) {
        self.memory_pressure_events += 1;
    }

    /// Add a GPU error
    pub fn add_gpu_error(&mut self, error: impl Into<String>) {
        self.gpu_errors.push(error.into());
    }

    /// Add a memory error
    pub fn add_memory_error(&mut self, error: impl Into<String>) {
        self.memory_errors.push(error.into());
    }

    /// Add a transfer error
    pub fn add_transfer_error(&mut self, error: impl Into<String>) {
        self.transfer_errors.push(error.into());
    }

    /// Calculate performance degradation
    pub fn calculate_degradation(&mut self) {
        if self.baseline_flops > 0.0 {
            let diff = self.baseline_flops - self.achieved_flops;
            self.performance_degradation_pct = (diff / self.baseline_flops) * 100.0;
        }
    }

    /// Check if there were any errors
    #[must_use]
    pub fn has_errors(&self) -> bool {
        !self.gpu_errors.is_empty()
            || !self.memory_errors.is_empty()
            || !self.transfer_errors.is_empty()
    }

    /// Get total error count
    #[must_use]
    pub fn total_errors(&self) -> usize {
        self.gpu_errors.len() + self.memory_errors.len() + self.transfer_errors.len()
    }
}

// ============================================================================
// Stress Test Report
// ============================================================================

/// Stress test report
#[derive(Debug, Clone)]
pub struct StressTestReport {
    /// Test configuration
    pub config: StressTestConfig,
    /// Collected metrics
    pub metrics: StressMetrics,
    /// Actual test duration
    pub duration_actual: Duration,
    /// Overall verdict
    pub verdict: StressTestVerdict,
    /// Recommendations
    pub recommendations: Vec<String>,
    /// Report timestamp
    pub timestamp: Instant,
}

impl StressTestReport {
    /// Create a new report
    #[must_use]
    pub fn new(config: StressTestConfig, metrics: StressMetrics, duration: Duration) -> Self {
        let verdict = Self::calculate_verdict(&metrics);
        let recommendations = Self::generate_recommendations(&metrics, verdict);

        Self {
            config,
            metrics,
            duration_actual: duration,
            verdict,
            recommendations,
            timestamp: Instant::now(),
        }
    }

    fn calculate_verdict(metrics: &StressMetrics) -> StressTestVerdict {
        // Critical failures
        if metrics.has_errors() {
            return StressTestVerdict::Fail;
        }

        // Check for severe issues
        if metrics.thermal_throttle_count > 10 {
            return StressTestVerdict::Fail;
        }
        if metrics.peak_temperature_c > 95.0 {
            return StressTestVerdict::Fail;
        }
        if metrics.performance_degradation_pct > 50.0 {
            return StressTestVerdict::Fail;
        }

        // Minor issues
        if metrics.thermal_throttle_count > 0
            || metrics.power_throttle_count > 0
            || metrics.memory_pressure_events > 0
        {
            return StressTestVerdict::PassWithNotes;
        }

        StressTestVerdict::Pass
    }

    fn generate_recommendations(
        metrics: &StressMetrics,
        verdict: StressTestVerdict,
    ) -> Vec<String> {
        let mut recs = Vec::new();

        if metrics.peak_temperature_c > 85.0 {
            recs.push("Consider improving cooling - peak temperature exceeded 85°C".to_string());
        }

        if metrics.thermal_throttle_count > 0 {
            recs.push(format!(
                "Thermal throttling detected {} times - reduce workload or improve cooling",
                metrics.thermal_throttle_count
            ));
        }

        if metrics.power_throttle_count > 0 {
            recs.push(format!(
                "Power throttling detected {} times - check power supply capacity",
                metrics.power_throttle_count
            ));
        }

        if metrics.memory_pressure_events > 0 {
            recs.push(format!(
                "Memory pressure detected {} times - consider reducing parallel jobs",
                metrics.memory_pressure_events
            ));
        }

        if metrics.performance_degradation_pct > 10.0 {
            recs.push(format!(
                "Performance degraded by {:.1}% under load - investigate bottlenecks",
                metrics.performance_degradation_pct
            ));
        }

        if verdict == StressTestVerdict::Pass && recs.is_empty() {
            recs.push("System passed all stress tests - no issues detected".to_string());
        }

        recs
    }

    /// Export report to JSON
    #[must_use]
    pub fn to_json(&self) -> String {
        // Simplified JSON format
        format!(
            r#"{{
  "verdict": "{}",
  "duration_seconds": {:.1},
  "peak_cpu_pct": {:.1},
  "peak_gpu_pct": {:.1},
  "peak_memory_pct": {:.1},
  "peak_temp_c": {:.1},
  "peak_power_w": {:.1},
  "thermal_throttles": {},
  "power_throttles": {},
  "memory_pressure_events": {},
  "total_errors": {},
  "performance_degradation_pct": {:.1},
  "recommendations": {:?}
}}"#,
            self.verdict,
            self.duration_actual.as_secs_f64(),
            self.metrics.peak_cpu_utilization,
            self.metrics.peak_gpu_utilization,
            self.metrics.peak_memory_utilization,
            self.metrics.peak_temperature_c,
            self.metrics.peak_power_watts,
            self.metrics.thermal_throttle_count,
            self.metrics.power_throttle_count,
            self.metrics.memory_pressure_events,
            self.metrics.total_errors(),
            self.metrics.performance_degradation_pct,
            self.recommendations
        )
    }
}

/// Stress test verdict
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressTestVerdict {
    /// All tests passed
    Pass,
    /// Passed with minor notes
    PassWithNotes,
    /// Failed
    Fail,
}

impl std::fmt::Display for StressTestVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::PassWithNotes => write!(f, "PASS_WITH_NOTES"),
            Self::Fail => write!(f, "FAIL"),
        }
    }
}

// ============================================================================
// Stress Test Runner State
// ============================================================================

/// Stress test runner state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressTestState {
    /// Not started
    Idle,
    /// Ramping up intensity
    RampUp,
    /// Running at full intensity
    Running,
    /// Cooling down
    CoolDown,
    /// Completed
    Completed,
    /// Failed/aborted
    Aborted,
}

impl std::fmt::Display for StressTestState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "Idle"),
            Self::RampUp => write!(f, "Ramp-Up"),
            Self::Running => write!(f, "Running"),
            Self::CoolDown => write!(f, "Cool-Down"),
            Self::Completed => write!(f, "Completed"),
            Self::Aborted => write!(f, "Aborted"),
        }
    }
}

// ============================================================================
// Tests (Extreme TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // H052: Stress Config Tests
    // =========================================================================

    #[test]
    fn h052_stress_config_default() {
        let config = StressTestConfig::default();
        assert_eq!(config.target, StressTarget::All);
        assert_eq!(config.duration, Duration::from_secs(60));
        assert!((config.intensity - 1.0).abs() < 0.001);
    }

    #[test]
    fn h052_stress_config_builder() {
        let config = StressTestConfig::new()
            .with_target(StressTarget::Cpu)
            .with_duration(Duration::from_secs(30))
            .with_intensity(0.8)
            .with_ramp_up(Duration::from_secs(10))
            .with_chaos(ChaosPreset::Gentle);

        assert_eq!(config.target, StressTarget::Cpu);
        assert_eq!(config.duration, Duration::from_secs(30));
        assert!((config.intensity - 0.8).abs() < 0.001);
        assert_eq!(config.chaos_preset, Some(ChaosPreset::Gentle));
    }

    #[test]
    fn h052_stress_config_intensity_clamp() {
        let config = StressTestConfig::new().with_intensity(1.5);
        assert_eq!(config.intensity, 1.0);

        let config2 = StressTestConfig::new().with_intensity(-0.5);
        assert_eq!(config2.intensity, 0.0);
    }

    #[test]
    fn h052_parse_duration() {
        assert_eq!(
            StressTestConfig::parse_duration("60s"),
            Some(Duration::from_secs(60))
        );
        assert_eq!(
            StressTestConfig::parse_duration("5m"),
            Some(Duration::from_secs(300))
        );
        assert_eq!(
            StressTestConfig::parse_duration("1h"),
            Some(Duration::from_secs(3600))
        );
        assert_eq!(StressTestConfig::parse_duration(""), None);
        assert_eq!(StressTestConfig::parse_duration("invalid"), None);
    }

    // =========================================================================
    // H053: Stress Target Tests
    // =========================================================================

    #[test]
    fn h053_stress_target_parse() {
        assert_eq!(StressTarget::parse("all"), Some(StressTarget::All));
        assert_eq!(StressTarget::parse("cpu"), Some(StressTarget::Cpu));
        assert_eq!(StressTarget::parse("GPU"), Some(StressTarget::Gpu(None)));
        assert_eq!(StressTarget::parse("memory"), Some(StressTarget::Memory));
        assert_eq!(StressTarget::parse("pcie"), Some(StressTarget::Pcie));
        assert!(matches!(
            StressTarget::parse("gpu:0"),
            Some(StressTarget::Gpu(Some(_)))
        ));
        assert_eq!(StressTarget::parse("invalid"), None);
    }

    #[test]
    fn h053_stress_target_includes() {
        let all = StressTarget::All;
        assert!(all.includes_cpu());
        assert!(all.includes_gpu());
        assert!(all.includes_memory());
        assert!(all.includes_pcie());

        let cpu = StressTarget::Cpu;
        assert!(cpu.includes_cpu());
        assert!(!cpu.includes_gpu());
        assert!(!cpu.includes_memory());
        assert!(!cpu.includes_pcie());
    }

    #[test]
    fn h053_stress_target_custom() {
        let custom = StressTarget::Custom(vec![StressTarget::Cpu, StressTarget::Memory]);
        assert!(custom.includes_cpu());
        assert!(!custom.includes_gpu());
        assert!(custom.includes_memory());
        assert!(!custom.includes_pcie());
    }

    // =========================================================================
    // H054: Chaos Preset Tests
    // =========================================================================

    #[test]
    fn h054_chaos_preset_parse() {
        assert_eq!(ChaosPreset::parse("gentle"), Some(ChaosPreset::Gentle));
        assert_eq!(ChaosPreset::parse("MODERATE"), Some(ChaosPreset::Moderate));
        assert_eq!(ChaosPreset::parse("aggressive"), Some(ChaosPreset::Aggressive));
        assert_eq!(ChaosPreset::parse("extreme"), Some(ChaosPreset::Extreme));
        assert_eq!(ChaosPreset::parse("invalid"), None);
    }

    #[test]
    fn h054_chaos_preset_factors() {
        let gentle = ChaosPreset::Gentle;
        assert!((gentle.memory_limit_factor() - 0.9).abs() < 0.001);
        assert!((gentle.cpu_throttle_factor() - 1.0).abs() < 0.001);
        assert_eq!(gentle.network_latency_ms(), 0);
        assert!((gentle.failure_rate() - 0.0).abs() < 0.001);

        let extreme = ChaosPreset::Extreme;
        assert!((extreme.memory_limit_factor() - 0.25).abs() < 0.001);
        assert!((extreme.cpu_throttle_factor() - 0.5).abs() < 0.001);
        assert_eq!(extreme.network_latency_ms(), 200);
        assert!((extreme.failure_rate() - 0.10).abs() < 0.001);
    }

    // =========================================================================
    // H055: Stress Metrics Tests
    // =========================================================================

    #[test]
    fn h055_stress_metrics_default() {
        let metrics = StressMetrics::new();
        assert_eq!(metrics.peak_cpu_utilization, 0.0);
        assert_eq!(metrics.thermal_throttle_count, 0);
        assert!(!metrics.has_errors());
    }

    #[test]
    fn h055_stress_metrics_update_peaks() {
        let mut metrics = StressMetrics::new();

        metrics.update_peaks(50.0, 60.0, 70.0, 75.0, 300.0, 15.0);
        assert!((metrics.peak_cpu_utilization - 50.0).abs() < 0.01);
        assert_eq!(metrics.sample_count, 1);

        metrics.update_peaks(80.0, 40.0, 60.0, 70.0, 200.0, 10.0);
        assert!((metrics.peak_cpu_utilization - 80.0).abs() < 0.01);
        assert!((metrics.peak_gpu_utilization - 60.0).abs() < 0.01); // Previous was higher
        assert_eq!(metrics.sample_count, 2);
    }

    #[test]
    fn h055_stress_metrics_events() {
        let mut metrics = StressMetrics::new();

        metrics.record_thermal_throttle();
        metrics.record_thermal_throttle();
        metrics.record_power_throttle();
        metrics.record_memory_pressure();

        assert_eq!(metrics.thermal_throttle_count, 2);
        assert_eq!(metrics.power_throttle_count, 1);
        assert_eq!(metrics.memory_pressure_events, 1);
    }

    #[test]
    fn h055_stress_metrics_errors() {
        let mut metrics = StressMetrics::new();
        assert!(!metrics.has_errors());
        assert_eq!(metrics.total_errors(), 0);

        metrics.add_gpu_error("GPU timeout");
        metrics.add_memory_error("OOM");
        metrics.add_transfer_error("Transfer failed");

        assert!(metrics.has_errors());
        assert_eq!(metrics.total_errors(), 3);
    }

    #[test]
    fn h055_stress_metrics_degradation() {
        let mut metrics = StressMetrics::new();
        metrics.baseline_flops = 1000.0;
        metrics.achieved_flops = 750.0;
        metrics.calculate_degradation();

        assert!((metrics.performance_degradation_pct - 25.0).abs() < 0.01);
    }

    // =========================================================================
    // H056: Stress Report Tests
    // =========================================================================

    #[test]
    fn h056_stress_report_pass() {
        let config = StressTestConfig::default();
        let metrics = StressMetrics::new();
        let report = StressTestReport::new(config, metrics, Duration::from_secs(60));

        assert_eq!(report.verdict, StressTestVerdict::Pass);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn h056_stress_report_pass_with_notes() {
        let config = StressTestConfig::default();
        let mut metrics = StressMetrics::new();
        metrics.thermal_throttle_count = 2;

        let report = StressTestReport::new(config, metrics, Duration::from_secs(60));

        assert_eq!(report.verdict, StressTestVerdict::PassWithNotes);
    }

    #[test]
    fn h056_stress_report_fail_errors() {
        let config = StressTestConfig::default();
        let mut metrics = StressMetrics::new();
        metrics.add_gpu_error("Critical error");

        let report = StressTestReport::new(config, metrics, Duration::from_secs(60));

        assert_eq!(report.verdict, StressTestVerdict::Fail);
    }

    #[test]
    fn h056_stress_report_fail_thermal() {
        let config = StressTestConfig::default();
        let mut metrics = StressMetrics::new();
        metrics.peak_temperature_c = 100.0;

        let report = StressTestReport::new(config, metrics, Duration::from_secs(60));

        assert_eq!(report.verdict, StressTestVerdict::Fail);
    }

    #[test]
    fn h056_stress_report_to_json() {
        let config = StressTestConfig::default();
        let metrics = StressMetrics::new();
        let report = StressTestReport::new(config, metrics, Duration::from_secs(60));

        let json = report.to_json();
        assert!(json.contains("\"verdict\": \"PASS\""));
        assert!(json.contains("\"duration_seconds\""));
    }

    // =========================================================================
    // H057: Stress Test State Tests
    // =========================================================================

    #[test]
    fn h057_stress_state_display() {
        assert_eq!(format!("{}", StressTestState::Idle), "Idle");
        assert_eq!(format!("{}", StressTestState::RampUp), "Ramp-Up");
        assert_eq!(format!("{}", StressTestState::Running), "Running");
        assert_eq!(format!("{}", StressTestState::CoolDown), "Cool-Down");
        assert_eq!(format!("{}", StressTestState::Completed), "Completed");
        assert_eq!(format!("{}", StressTestState::Aborted), "Aborted");
    }

    // =========================================================================
    // H058: Verdict Display Tests
    // =========================================================================

    #[test]
    fn h058_verdict_display() {
        assert_eq!(format!("{}", StressTestVerdict::Pass), "PASS");
        assert_eq!(format!("{}", StressTestVerdict::PassWithNotes), "PASS_WITH_NOTES");
        assert_eq!(format!("{}", StressTestVerdict::Fail), "FAIL");
    }
}
