//! Memory Hierarchy Monitoring (TRUENO-SPEC-021)
//!
//! Comprehensive memory metrics including RAM, SWAP, and GPU VRAM
//! with pressure level detection based on LAMBDA-0002 specification.
//!
//! # Memory Pressure Levels (from lambda-lab-rust-development)
//!
//! | Level | Available | Action |
//! |-------|-----------|--------|
//! | Ok | >= 50% | Normal operation |
//! | Elevated | 30-50% | Monitor closely |
//! | Warning | 15-30% | Reduce parallelism |
//! | Critical | < 15% | Block new builds |
//!
//! # References
//!
//! - [Hennessy2017] Memory hierarchy model
//! - [McCalpin1995] STREAM bandwidth benchmarking
//! - [Drepper2007] Memory access patterns

use std::collections::VecDeque;
use std::fmt;

use super::device::DeviceId;

// ============================================================================
// Memory Pressure Levels (LAMBDA-0002)
// ============================================================================

/// Memory pressure level based on available memory percentage
///
/// From lambda-lab-rust-development LAMBDA-0002 specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PressureLevel {
    /// Normal operation (>= 50% available)
    Ok,
    /// Monitor closely (30-50% available)
    Elevated,
    /// Reduce parallelism (15-30% available)
    Warning,
    /// Block new builds (< 15% available)
    Critical,
}

impl PressureLevel {
    /// Determine pressure level from available percentage
    #[must_use]
    pub fn from_available_percent(percent: f64) -> Self {
        match percent {
            x if x >= 50.0 => Self::Ok,
            x if x >= 30.0 => Self::Elevated,
            x if x >= 15.0 => Self::Warning,
            _ => Self::Critical,
        }
    }

    /// Get recommendation text for this pressure level
    #[must_use]
    pub fn recommendation(&self) -> &'static str {
        match self {
            Self::Ok => "System healthy - normal operation",
            Self::Elevated => "Memory usage elevated - monitor closely",
            Self::Warning => "High memory usage - reduce parallel jobs",
            Self::Critical => "Critical memory pressure - block new allocations",
        }
    }

    /// Check if new allocations should be blocked
    #[must_use]
    pub fn should_block_allocations(&self) -> bool {
        matches!(self, Self::Critical)
    }

    /// Get ANSI color code for TUI display
    #[must_use]
    pub fn ansi_color(&self) -> &'static str {
        match self {
            Self::Ok => "\x1b[32m",      // Green
            Self::Elevated => "\x1b[33m", // Yellow
            Self::Warning => "\x1b[38;5;208m", // Orange
            Self::Critical => "\x1b[31m", // Red
        }
    }
}

impl fmt::Display for PressureLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ok => write!(f, "OK"),
            Self::Elevated => write!(f, "ELEVATED"),
            Self::Warning => write!(f, "WARNING"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

// ============================================================================
// Memory Metrics (TRUENO-SPEC-021 Section 3.2)
// ============================================================================

/// Comprehensive memory metrics for system and GPU
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    // System RAM
    /// RAM used in bytes
    pub ram_used_bytes: u64,
    /// RAM total in bytes
    pub ram_total_bytes: u64,
    /// RAM available in bytes (accounts for cache/buffers)
    pub ram_available_bytes: u64,
    /// RAM cached in bytes
    pub ram_cached_bytes: u64,
    /// RAM buffers in bytes
    pub ram_buffers_bytes: u64,

    // Swap
    /// Swap used in bytes
    pub swap_used_bytes: u64,
    /// Swap total in bytes
    pub swap_total_bytes: u64,

    // Per-GPU VRAM
    /// GPU VRAM metrics for each device
    pub gpu_vram: Vec<GpuVramMetrics>,

    // Derived metrics
    /// Current pressure level
    pub pressure_level: PressureLevel,
    /// Safe number of parallel jobs (based on 3GB/job heuristic)
    pub safe_parallel_jobs: u32,

    // Bandwidth (if measurable)
    /// RAM read bandwidth in GB/s
    pub ram_read_bandwidth_gbps: Option<f64>,
    /// RAM write bandwidth in GB/s
    pub ram_write_bandwidth_gbps: Option<f64>,

    // History (60-point sparkline, ~60 seconds at 1Hz)
    /// RAM usage history (percentage, 0.0-100.0)
    pub ram_history: VecDeque<f64>,
    /// Swap usage history (percentage, 0.0-100.0)
    pub swap_history: VecDeque<f64>,
}

impl MemoryMetrics {
    /// Maximum history points (60 seconds at 1Hz)
    pub const MAX_HISTORY_POINTS: usize = 60;

    /// Create new memory metrics by reading system state
    #[must_use]
    pub fn new() -> Self {
        let mut metrics = Self::default();
        metrics.refresh();
        metrics
    }

    /// Refresh all memory metrics from system
    pub fn refresh(&mut self) {
        self.read_meminfo();
        self.read_swapinfo();
        self.calculate_pressure();
        self.update_history();
    }

    /// Read /proc/meminfo on Linux
    fn read_meminfo(&mut self) {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let value_kb: u64 = parts[1].parse().unwrap_or(0);
                        let value_bytes = value_kb * 1024;

                        match parts[0] {
                            "MemTotal:" => self.ram_total_bytes = value_bytes,
                            "MemAvailable:" => self.ram_available_bytes = value_bytes,
                            "Cached:" => self.ram_cached_bytes = value_bytes,
                            "Buffers:" => self.ram_buffers_bytes = value_bytes,
                            _ => {}
                        }
                    }
                }
                // Used = Total - Available
                self.ram_used_bytes = self.ram_total_bytes.saturating_sub(self.ram_available_bytes);
            }
        }
    }

    /// Read swap information
    fn read_swapinfo(&mut self) {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let value_kb: u64 = parts[1].parse().unwrap_or(0);
                        let value_bytes = value_kb * 1024;

                        match parts[0] {
                            "SwapTotal:" => self.swap_total_bytes = value_bytes,
                            "SwapFree:" => {
                                self.swap_used_bytes =
                                    self.swap_total_bytes.saturating_sub(value_bytes);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    /// Calculate pressure level and safe jobs
    fn calculate_pressure(&mut self) {
        let available_pct = self.ram_available_percent();
        self.pressure_level = PressureLevel::from_available_percent(available_pct);

        // Safe jobs = min(available_gb / 3.0, cpu_cores)
        // Based on 3GB/job heuristic [Volkov2008]
        let available_gb = self.ram_available_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);

        self.safe_parallel_jobs = ((available_gb / 3.0) as u32).min(cpu_cores).max(1);
    }

    /// Update history sparklines
    fn update_history(&mut self) {
        // Add current RAM usage percentage
        self.ram_history.push_back(self.ram_usage_percent());
        if self.ram_history.len() > Self::MAX_HISTORY_POINTS {
            self.ram_history.pop_front();
        }

        // Add current swap usage percentage
        self.swap_history.push_back(self.swap_usage_percent());
        if self.swap_history.len() > Self::MAX_HISTORY_POINTS {
            self.swap_history.pop_front();
        }
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    /// Get RAM usage percentage (0.0-100.0)
    #[must_use]
    pub fn ram_usage_percent(&self) -> f64 {
        if self.ram_total_bytes == 0 {
            return 0.0;
        }
        (self.ram_used_bytes as f64 / self.ram_total_bytes as f64) * 100.0
    }

    /// Get RAM available percentage (0.0-100.0)
    #[must_use]
    pub fn ram_available_percent(&self) -> f64 {
        if self.ram_total_bytes == 0 {
            return 100.0;
        }
        (self.ram_available_bytes as f64 / self.ram_total_bytes as f64) * 100.0
    }

    /// Get swap usage percentage (0.0-100.0)
    #[must_use]
    pub fn swap_usage_percent(&self) -> f64 {
        if self.swap_total_bytes == 0 {
            return 0.0;
        }
        (self.swap_used_bytes as f64 / self.swap_total_bytes as f64) * 100.0
    }

    /// Get RAM used in GB
    #[must_use]
    pub fn ram_used_gb(&self) -> f64 {
        self.ram_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get RAM total in GB
    #[must_use]
    pub fn ram_total_gb(&self) -> f64 {
        self.ram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get swap used in GB
    #[must_use]
    pub fn swap_used_gb(&self) -> f64 {
        self.swap_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get swap total in GB
    #[must_use]
    pub fn swap_total_gb(&self) -> f64 {
        self.swap_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get total GPU VRAM used across all devices
    #[must_use]
    pub fn total_vram_used_bytes(&self) -> u64 {
        self.gpu_vram.iter().map(|v| v.used_bytes).sum()
    }

    /// Get total GPU VRAM capacity across all devices
    #[must_use]
    pub fn total_vram_total_bytes(&self) -> u64 {
        self.gpu_vram.iter().map(|v| v.total_bytes).sum()
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            ram_used_bytes: 0,
            ram_total_bytes: 0,
            ram_available_bytes: 0,
            ram_cached_bytes: 0,
            ram_buffers_bytes: 0,
            swap_used_bytes: 0,
            swap_total_bytes: 0,
            gpu_vram: Vec::new(),
            pressure_level: PressureLevel::Ok,
            safe_parallel_jobs: 1,
            ram_read_bandwidth_gbps: None,
            ram_write_bandwidth_gbps: None,
            ram_history: VecDeque::with_capacity(Self::MAX_HISTORY_POINTS),
            swap_history: VecDeque::with_capacity(Self::MAX_HISTORY_POINTS),
        }
    }
}

// ============================================================================
// GPU VRAM Metrics (TRUENO-SPEC-021 Section 3.2)
// ============================================================================

/// GPU VRAM metrics for a single device
#[derive(Debug, Clone)]
pub struct GpuVramMetrics {
    /// Device ID
    pub device_id: DeviceId,
    /// VRAM used in bytes
    pub used_bytes: u64,
    /// VRAM total in bytes
    pub total_bytes: u64,
    /// VRAM reserved by driver/system
    pub reserved_bytes: u64,
    /// PCIe BAR1 aperture usage (for large memory)
    pub bar1_used_bytes: u64,
    /// Usage history (percentage, 0.0-100.0)
    pub history: VecDeque<f64>,
}

impl GpuVramMetrics {
    /// Maximum history points
    pub const MAX_HISTORY_POINTS: usize = 60;

    /// Create new GPU VRAM metrics
    #[must_use]
    pub fn new(device_id: DeviceId, used: u64, total: u64) -> Self {
        Self {
            device_id,
            used_bytes: used,
            total_bytes: total,
            reserved_bytes: 0,
            bar1_used_bytes: 0,
            history: VecDeque::with_capacity(Self::MAX_HISTORY_POINTS),
        }
    }

    /// Get VRAM usage percentage (0.0-100.0)
    #[must_use]
    pub fn usage_percent(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        (self.used_bytes as f64 / self.total_bytes as f64) * 100.0
    }

    /// Get VRAM available in bytes
    #[must_use]
    pub fn available_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.used_bytes)
    }

    /// Get VRAM used in GB
    #[must_use]
    pub fn used_gb(&self) -> f64 {
        self.used_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Get VRAM total in GB
    #[must_use]
    pub fn total_gb(&self) -> f64 {
        self.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Update usage and add to history
    pub fn update(&mut self, used: u64) {
        self.used_bytes = used;
        self.history.push_back(self.usage_percent());
        if self.history.len() > Self::MAX_HISTORY_POINTS {
            self.history.pop_front();
        }
    }
}

// ============================================================================
// Pressure Analysis Result
// ============================================================================

/// Detailed memory pressure analysis
#[derive(Debug, Clone)]
pub struct PressureAnalysis {
    /// Current pressure level
    pub level: PressureLevel,
    /// Available memory percentage
    pub available_percent: f64,
    /// Available memory in GB
    pub available_gb: f64,
    /// Safe number of parallel jobs
    pub safe_jobs: u32,
    /// Whether to block new builds
    pub block_builds: bool,
    /// Human-readable recommendation
    pub recommendation: String,
}

impl PressureAnalysis {
    /// Analyze memory metrics and return detailed analysis
    #[must_use]
    pub fn from_metrics(metrics: &MemoryMetrics) -> Self {
        let available_pct = metrics.ram_available_percent();
        let available_gb = metrics.ram_available_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let level = metrics.pressure_level;

        Self {
            level,
            available_percent: available_pct,
            available_gb,
            safe_jobs: metrics.safe_parallel_jobs,
            block_builds: level.should_block_allocations(),
            recommendation: level.recommendation().to_string(),
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
    // H011: Pressure Level Tests
    // =========================================================================

    #[test]
    fn h011_pressure_level_from_percent_ok() {
        assert_eq!(PressureLevel::from_available_percent(100.0), PressureLevel::Ok);
        assert_eq!(PressureLevel::from_available_percent(75.0), PressureLevel::Ok);
        assert_eq!(PressureLevel::from_available_percent(50.0), PressureLevel::Ok);
    }

    #[test]
    fn h011_pressure_level_from_percent_elevated() {
        assert_eq!(PressureLevel::from_available_percent(49.9), PressureLevel::Elevated);
        assert_eq!(PressureLevel::from_available_percent(40.0), PressureLevel::Elevated);
        assert_eq!(PressureLevel::from_available_percent(30.0), PressureLevel::Elevated);
    }

    #[test]
    fn h011_pressure_level_from_percent_warning() {
        assert_eq!(PressureLevel::from_available_percent(29.9), PressureLevel::Warning);
        assert_eq!(PressureLevel::from_available_percent(20.0), PressureLevel::Warning);
        assert_eq!(PressureLevel::from_available_percent(15.0), PressureLevel::Warning);
    }

    #[test]
    fn h011_pressure_level_from_percent_critical() {
        assert_eq!(PressureLevel::from_available_percent(14.9), PressureLevel::Critical);
        assert_eq!(PressureLevel::from_available_percent(5.0), PressureLevel::Critical);
        assert_eq!(PressureLevel::from_available_percent(0.0), PressureLevel::Critical);
    }

    #[test]
    fn h011_pressure_level_display() {
        assert_eq!(format!("{}", PressureLevel::Ok), "OK");
        assert_eq!(format!("{}", PressureLevel::Elevated), "ELEVATED");
        assert_eq!(format!("{}", PressureLevel::Warning), "WARNING");
        assert_eq!(format!("{}", PressureLevel::Critical), "CRITICAL");
    }

    #[test]
    fn h011_pressure_level_recommendation() {
        assert!(PressureLevel::Ok.recommendation().contains("healthy"));
        assert!(PressureLevel::Critical.recommendation().contains("block"));
    }

    #[test]
    fn h011_pressure_level_should_block() {
        assert!(!PressureLevel::Ok.should_block_allocations());
        assert!(!PressureLevel::Elevated.should_block_allocations());
        assert!(!PressureLevel::Warning.should_block_allocations());
        assert!(PressureLevel::Critical.should_block_allocations());
    }

    #[test]
    fn h011_pressure_level_ansi_color() {
        // All levels should have ANSI color codes
        assert!(PressureLevel::Ok.ansi_color().contains("\x1b["));
        assert!(PressureLevel::Elevated.ansi_color().contains("\x1b["));
        assert!(PressureLevel::Warning.ansi_color().contains("\x1b["));
        assert!(PressureLevel::Critical.ansi_color().contains("\x1b["));
    }

    // =========================================================================
    // H012: Memory Metrics Tests
    // =========================================================================

    #[test]
    fn h012_memory_metrics_default() {
        let metrics = MemoryMetrics::default();
        assert_eq!(metrics.ram_used_bytes, 0);
        assert_eq!(metrics.ram_total_bytes, 0);
        assert_eq!(metrics.pressure_level, PressureLevel::Ok);
    }

    #[test]
    fn h012_memory_metrics_new() {
        let metrics = MemoryMetrics::new();
        // On Linux, should have read some values
        #[cfg(target_os = "linux")]
        {
            assert!(metrics.ram_total_bytes > 0);
        }
    }

    #[test]
    fn h012_memory_metrics_ram_usage_percent() {
        let mut metrics = MemoryMetrics::default();
        metrics.ram_used_bytes = 50 * 1024 * 1024 * 1024; // 50 GB
        metrics.ram_total_bytes = 100 * 1024 * 1024 * 1024; // 100 GB

        assert!((metrics.ram_usage_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn h012_memory_metrics_ram_usage_percent_zero() {
        let metrics = MemoryMetrics::default();
        assert!((metrics.ram_usage_percent() - 0.0).abs() < 0.01);
    }

    #[test]
    fn h012_memory_metrics_available_percent() {
        let mut metrics = MemoryMetrics::default();
        metrics.ram_available_bytes = 75 * 1024 * 1024 * 1024; // 75 GB
        metrics.ram_total_bytes = 100 * 1024 * 1024 * 1024; // 100 GB

        assert!((metrics.ram_available_percent() - 75.0).abs() < 0.01);
    }

    #[test]
    fn h012_memory_metrics_available_percent_zero_total() {
        let metrics = MemoryMetrics::default();
        // When total is 0, should return 100% available (not divide by zero)
        assert!((metrics.ram_available_percent() - 100.0).abs() < 0.01);
    }

    #[test]
    fn h012_memory_metrics_swap_percent() {
        let mut metrics = MemoryMetrics::default();
        metrics.swap_used_bytes = 2 * 1024 * 1024 * 1024; // 2 GB
        metrics.swap_total_bytes = 16 * 1024 * 1024 * 1024; // 16 GB

        assert!((metrics.swap_usage_percent() - 12.5).abs() < 0.01);
    }

    #[test]
    fn h012_memory_metrics_swap_percent_zero() {
        let metrics = MemoryMetrics::default();
        assert!((metrics.swap_usage_percent() - 0.0).abs() < 0.01);
    }

    #[test]
    fn h012_memory_metrics_gb_helpers() {
        let mut metrics = MemoryMetrics::default();
        metrics.ram_used_bytes = 32 * 1024 * 1024 * 1024;
        metrics.ram_total_bytes = 64 * 1024 * 1024 * 1024;
        metrics.swap_used_bytes = 1 * 1024 * 1024 * 1024;
        metrics.swap_total_bytes = 16 * 1024 * 1024 * 1024;

        assert!((metrics.ram_used_gb() - 32.0).abs() < 0.01);
        assert!((metrics.ram_total_gb() - 64.0).abs() < 0.01);
        assert!((metrics.swap_used_gb() - 1.0).abs() < 0.01);
        assert!((metrics.swap_total_gb() - 16.0).abs() < 0.01);
    }

    // =========================================================================
    // H013: History Tracking Tests
    // =========================================================================

    #[test]
    fn h013_memory_metrics_history_max() {
        let mut metrics = MemoryMetrics::default();
        metrics.ram_total_bytes = 100;
        metrics.swap_total_bytes = 100;

        // Add more than MAX_HISTORY_POINTS
        for i in 0..100 {
            metrics.ram_used_bytes = i;
            metrics.swap_used_bytes = i;
            metrics.update_history();
        }

        assert_eq!(metrics.ram_history.len(), MemoryMetrics::MAX_HISTORY_POINTS);
        assert_eq!(metrics.swap_history.len(), MemoryMetrics::MAX_HISTORY_POINTS);
    }

    // =========================================================================
    // H014: GPU VRAM Metrics Tests
    // =========================================================================

    #[test]
    fn h014_gpu_vram_new() {
        let vram = GpuVramMetrics::new(DeviceId::nvidia(0), 8 * 1024 * 1024 * 1024, 24 * 1024 * 1024 * 1024);
        assert_eq!(vram.device_id, DeviceId::nvidia(0));
        assert_eq!(vram.used_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(vram.total_bytes, 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn h014_gpu_vram_usage_percent() {
        let vram = GpuVramMetrics::new(DeviceId::nvidia(0), 6 * 1024 * 1024 * 1024, 24 * 1024 * 1024 * 1024);
        assert!((vram.usage_percent() - 25.0).abs() < 0.01);
    }

    #[test]
    fn h014_gpu_vram_usage_percent_zero() {
        let vram = GpuVramMetrics::new(DeviceId::nvidia(0), 0, 0);
        assert!((vram.usage_percent() - 0.0).abs() < 0.01);
    }

    #[test]
    fn h014_gpu_vram_available() {
        let vram = GpuVramMetrics::new(DeviceId::nvidia(0), 8 * 1024 * 1024 * 1024, 24 * 1024 * 1024 * 1024);
        assert_eq!(vram.available_bytes(), 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn h014_gpu_vram_gb_helpers() {
        let vram = GpuVramMetrics::new(DeviceId::nvidia(0), 8 * 1024 * 1024 * 1024, 24 * 1024 * 1024 * 1024);
        assert!((vram.used_gb() - 8.0).abs() < 0.01);
        assert!((vram.total_gb() - 24.0).abs() < 0.01);
    }

    #[test]
    fn h014_gpu_vram_update_history() {
        let mut vram = GpuVramMetrics::new(DeviceId::nvidia(0), 0, 24 * 1024 * 1024 * 1024);

        for i in 0..100 {
            vram.update(i * 1024 * 1024 * 1024);
        }

        assert_eq!(vram.history.len(), GpuVramMetrics::MAX_HISTORY_POINTS);
    }

    // =========================================================================
    // H015: Safe Jobs Calculation Tests
    // =========================================================================

    #[test]
    fn h015_safe_jobs_calculation() {
        let mut metrics = MemoryMetrics::default();
        // 30 GB available = 10 safe jobs (at 3GB/job)
        metrics.ram_available_bytes = 30 * 1024 * 1024 * 1024;
        metrics.ram_total_bytes = 64 * 1024 * 1024 * 1024;
        metrics.calculate_pressure();

        // Should be min(10, cpu_cores)
        assert!(metrics.safe_parallel_jobs >= 1);
        assert!(metrics.safe_parallel_jobs <= 10);
    }

    #[test]
    fn h015_safe_jobs_minimum_one() {
        let mut metrics = MemoryMetrics::default();
        // Very low memory - should still allow at least 1 job
        metrics.ram_available_bytes = 512 * 1024 * 1024; // 512 MB
        metrics.ram_total_bytes = 64 * 1024 * 1024 * 1024;
        metrics.calculate_pressure();

        assert_eq!(metrics.safe_parallel_jobs, 1);
    }

    // =========================================================================
    // H016: Pressure Analysis Tests
    // =========================================================================

    #[test]
    fn h016_pressure_analysis_from_metrics() {
        let mut metrics = MemoryMetrics::default();
        // 16 GB available of 64 GB = 25% available -> Warning level
        metrics.ram_available_bytes = 16 * 1024 * 1024 * 1024;
        metrics.ram_total_bytes = 64 * 1024 * 1024 * 1024;
        metrics.calculate_pressure();

        let analysis = PressureAnalysis::from_metrics(&metrics);

        assert_eq!(analysis.level, PressureLevel::Warning);
        assert!(!analysis.block_builds);
        assert!(analysis.safe_jobs >= 1);
    }

    #[test]
    fn h016_pressure_analysis_critical() {
        let mut metrics = MemoryMetrics::default();
        metrics.ram_available_bytes = 5 * 1024 * 1024 * 1024; // 5 GB
        metrics.ram_total_bytes = 64 * 1024 * 1024 * 1024; // 64 GB (~7.8%)
        metrics.calculate_pressure();

        let analysis = PressureAnalysis::from_metrics(&metrics);

        assert_eq!(analysis.level, PressureLevel::Critical);
        assert!(analysis.block_builds);
    }

    // =========================================================================
    // H017: Total VRAM Tests
    // =========================================================================

    #[test]
    fn h017_total_vram_single_gpu() {
        let mut metrics = MemoryMetrics::default();
        metrics.gpu_vram.push(GpuVramMetrics::new(
            DeviceId::nvidia(0),
            8 * 1024 * 1024 * 1024,
            24 * 1024 * 1024 * 1024,
        ));

        assert_eq!(metrics.total_vram_used_bytes(), 8 * 1024 * 1024 * 1024);
        assert_eq!(metrics.total_vram_total_bytes(), 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn h017_total_vram_multi_gpu() {
        let mut metrics = MemoryMetrics::default();
        metrics.gpu_vram.push(GpuVramMetrics::new(
            DeviceId::nvidia(0),
            8 * 1024 * 1024 * 1024,
            24 * 1024 * 1024 * 1024,
        ));
        metrics.gpu_vram.push(GpuVramMetrics::new(
            DeviceId::nvidia(1),
            4 * 1024 * 1024 * 1024,
            24 * 1024 * 1024 * 1024,
        ));

        assert_eq!(metrics.total_vram_used_bytes(), 12 * 1024 * 1024 * 1024);
        assert_eq!(metrics.total_vram_total_bytes(), 48 * 1024 * 1024 * 1024);
    }

    #[test]
    fn h017_total_vram_empty() {
        let metrics = MemoryMetrics::default();
        assert_eq!(metrics.total_vram_used_bytes(), 0);
        assert_eq!(metrics.total_vram_total_bytes(), 0);
    }
}
