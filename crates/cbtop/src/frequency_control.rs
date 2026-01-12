//! CPU Frequency Control Backend (PMAT-038)
//!
//! Interface with Linux cpufreq to lock CPU frequency for deterministic benchmarks.
//!
//! # Features
//!
//! - Read current CPU frequency
//! - Detect CPU governor (performance/powersave/ondemand)
//! - Frequency lock with RAII guard
//! - Measure variance before/after locking
//!
//! # Falsification Criteria (F1301-F1310)
//!
//! See `tests/frequency_control_f1301.rs` for falsification tests.

use std::path::PathBuf;

/// CPU governor types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuGovernor {
    /// Performance governor (max frequency)
    Performance,
    /// Powersave governor (min frequency)
    Powersave,
    /// Ondemand governor (dynamic)
    Ondemand,
    /// Conservative governor (gradual)
    Conservative,
    /// Schedutil governor (scheduler-based)
    Schedutil,
    /// Userspace governor (manual)
    Userspace,
    /// Unknown governor
    Unknown,
}

impl CpuGovernor {
    /// Get governor name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Performance => "performance",
            Self::Powersave => "powersave",
            Self::Ondemand => "ondemand",
            Self::Conservative => "conservative",
            Self::Schedutil => "schedutil",
            Self::Userspace => "userspace",
            Self::Unknown => "unknown",
        }
    }

    /// Parse from string
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "performance" => Self::Performance,
            "powersave" => Self::Powersave,
            "ondemand" => Self::Ondemand,
            "conservative" => Self::Conservative,
            "schedutil" => Self::Schedutil,
            "userspace" => Self::Userspace,
            _ => Self::Unknown,
        }
    }

    /// Check if deterministic (fixed frequency)
    pub fn is_deterministic(&self) -> bool {
        matches!(self, Self::Performance | Self::Powersave | Self::Userspace)
    }
}

/// CPU frequency info for a single CPU
#[derive(Debug, Clone)]
pub struct CpuFrequencyInfo {
    /// CPU ID
    pub cpu_id: usize,
    /// Current frequency in kHz
    pub current_khz: u64,
    /// Minimum frequency in kHz
    pub min_khz: u64,
    /// Maximum frequency in kHz
    pub max_khz: u64,
    /// Current governor
    pub governor: CpuGovernor,
    /// Available governors
    pub available_governors: Vec<CpuGovernor>,
}

impl CpuFrequencyInfo {
    /// Get current frequency in MHz
    pub fn current_mhz(&self) -> f64 {
        self.current_khz as f64 / 1000.0
    }

    /// Get current frequency in GHz
    pub fn current_ghz(&self) -> f64 {
        self.current_khz as f64 / 1_000_000.0
    }

    /// Get frequency utilization (current / max)
    pub fn utilization(&self) -> f64 {
        if self.max_khz > 0 {
            self.current_khz as f64 / self.max_khz as f64
        } else {
            1.0
        }
    }
}

/// Frequency reading result
#[derive(Debug, Clone)]
pub struct FrequencyReading {
    /// CPU info for each core
    pub cpus: Vec<CpuFrequencyInfo>,
    /// Timestamp
    pub timestamp_ns: u64,
}

impl FrequencyReading {
    /// Get average frequency in MHz
    pub fn average_mhz(&self) -> f64 {
        if self.cpus.is_empty() {
            return 0.0;
        }
        let total: f64 = self.cpus.iter().map(|c| c.current_mhz()).sum();
        total / self.cpus.len() as f64
    }

    /// Get min frequency across cores in MHz
    pub fn min_mhz(&self) -> f64 {
        self.cpus
            .iter()
            .map(|c| c.current_mhz())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    /// Get max frequency across cores in MHz
    pub fn max_mhz(&self) -> f64 {
        self.cpus
            .iter()
            .map(|c| c.current_mhz())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    /// Get frequency variance in MHz
    pub fn variance_mhz(&self) -> f64 {
        self.max_mhz() - self.min_mhz()
    }

    /// Check if all cores have same governor
    pub fn uniform_governor(&self) -> bool {
        if self.cpus.is_empty() {
            return true;
        }
        let first = self.cpus[0].governor;
        self.cpus.iter().all(|c| c.governor == first)
    }

    /// Get most common governor
    pub fn common_governor(&self) -> CpuGovernor {
        self.cpus.first().map(|c| c.governor).unwrap_or(CpuGovernor::Unknown)
    }
}

/// Frequency controller for reading and managing CPU frequencies
#[derive(Debug)]
pub struct FrequencyController {
    /// Sysfs base path
    sysfs_path: PathBuf,
    /// Number of CPUs
    cpu_count: usize,
    /// Mock mode for testing
    mock_mode: bool,
    /// Mock frequency value
    mock_frequency: u64,
    /// Mock governor
    mock_governor: CpuGovernor,
}

impl Default for FrequencyController {
    fn default() -> Self {
        Self::new()
    }
}

impl FrequencyController {
    /// Create new controller
    pub fn new() -> Self {
        let cpu_count = num_cpus();
        Self {
            sysfs_path: PathBuf::from("/sys/devices/system/cpu"),
            cpu_count,
            mock_mode: false,
            mock_frequency: 3_000_000, // 3 GHz default
            mock_governor: CpuGovernor::Performance,
        }
    }

    /// Enable mock mode for testing
    pub fn with_mock(mut self, frequency_khz: u64, governor: CpuGovernor) -> Self {
        self.mock_mode = true;
        self.mock_frequency = frequency_khz;
        self.mock_governor = governor;
        self
    }

    /// Get CPU count
    pub fn cpu_count(&self) -> usize {
        self.cpu_count
    }

    /// Read frequency for single CPU
    pub fn read_cpu_frequency(&self, cpu_id: usize) -> Option<CpuFrequencyInfo> {
        if self.mock_mode {
            return Some(CpuFrequencyInfo {
                cpu_id,
                current_khz: self.mock_frequency,
                min_khz: 800_000,
                max_khz: 4_000_000,
                governor: self.mock_governor,
                available_governors: vec![CpuGovernor::Performance, CpuGovernor::Powersave],
            });
        }

        // Read from sysfs
        let cpu_path = self.sysfs_path.join(format!("cpu{}/cpufreq", cpu_id));

        let current_khz = read_sysfs_value(&cpu_path.join("scaling_cur_freq")).unwrap_or(0);
        let min_khz = read_sysfs_value(&cpu_path.join("scaling_min_freq")).unwrap_or(0);
        let max_khz = read_sysfs_value(&cpu_path.join("scaling_max_freq")).unwrap_or(0);

        let governor_str = read_sysfs_string(&cpu_path.join("scaling_governor"))
            .unwrap_or_else(|| "unknown".to_string());
        let governor = CpuGovernor::from_str(&governor_str);

        let available_str = read_sysfs_string(&cpu_path.join("scaling_available_governors"))
            .unwrap_or_default();
        let available_governors: Vec<CpuGovernor> = available_str
            .split_whitespace()
            .map(CpuGovernor::from_str)
            .collect();

        Some(CpuFrequencyInfo {
            cpu_id,
            current_khz,
            min_khz,
            max_khz,
            governor,
            available_governors,
        })
    }

    /// Read all CPU frequencies
    pub fn read_all_frequencies(&self) -> FrequencyReading {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        let cpus: Vec<CpuFrequencyInfo> = (0..self.cpu_count)
            .filter_map(|id| self.read_cpu_frequency(id))
            .collect();

        FrequencyReading { cpus, timestamp_ns }
    }

    /// Detect current governor
    pub fn detect_governor(&self) -> CpuGovernor {
        self.read_all_frequencies().common_governor()
    }

    /// Check if frequency can be controlled
    pub fn can_control(&self) -> bool {
        if self.mock_mode {
            return true;
        }

        // Check if we have userspace governor available
        let reading = self.read_all_frequencies();
        if reading.cpus.is_empty() {
            return false;
        }

        reading.cpus[0]
            .available_governors
            .contains(&CpuGovernor::Userspace)
            || reading.cpus[0]
                .available_governors
                .contains(&CpuGovernor::Performance)
    }

    /// Measure frequency variance over time
    pub fn measure_variance(&self, samples: usize, interval_ms: u64) -> FrequencyVariance {
        let mut readings = Vec::with_capacity(samples);

        for _ in 0..samples {
            let reading = self.read_all_frequencies();
            readings.push(reading.average_mhz());
            std::thread::sleep(std::time::Duration::from_millis(interval_ms));
        }

        if readings.is_empty() {
            return FrequencyVariance::default();
        }

        let mean: f64 = readings.iter().sum::<f64>() / readings.len() as f64;
        let variance: f64 = readings.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (readings.len() - 1).max(1) as f64;
        let std_dev = variance.sqrt();
        let cv = if mean > 0.0 { std_dev / mean * 100.0 } else { 0.0 };

        let min = readings.iter().copied().fold(f64::INFINITY, f64::min);
        let max = readings.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        FrequencyVariance {
            mean_mhz: mean,
            std_dev_mhz: std_dev,
            cv_percent: cv,
            min_mhz: min,
            max_mhz: max,
            sample_count: readings.len(),
        }
    }
}

/// Frequency variance measurement
#[derive(Debug, Clone, Default)]
pub struct FrequencyVariance {
    /// Mean frequency in MHz
    pub mean_mhz: f64,
    /// Standard deviation in MHz
    pub std_dev_mhz: f64,
    /// Coefficient of variation (%)
    pub cv_percent: f64,
    /// Minimum frequency in MHz
    pub min_mhz: f64,
    /// Maximum frequency in MHz
    pub max_mhz: f64,
    /// Number of samples
    pub sample_count: usize,
}

impl FrequencyVariance {
    /// Check if variance is acceptable (<3% CV)
    pub fn is_stable(&self) -> bool {
        self.cv_percent < 3.0
    }

    /// Get range in MHz
    pub fn range_mhz(&self) -> f64 {
        self.max_mhz - self.min_mhz
    }
}

/// RAII frequency lock guard
#[derive(Debug)]
pub struct FrequencyLock {
    /// Original governor per CPU
    original_governors: Vec<(usize, CpuGovernor)>,
    /// Original frequencies per CPU
    original_frequencies: Vec<(usize, u64)>,
    /// Controller reference path
    sysfs_path: PathBuf,
    /// Was lock successful
    locked: bool,
    /// Mock mode
    mock_mode: bool,
}

impl FrequencyLock {
    /// Try to lock frequency to maximum
    pub fn try_lock(controller: &FrequencyController) -> Self {
        let reading = controller.read_all_frequencies();

        let original_governors: Vec<(usize, CpuGovernor)> = reading
            .cpus
            .iter()
            .map(|c| (c.cpu_id, c.governor))
            .collect();

        let original_frequencies: Vec<(usize, u64)> = reading
            .cpus
            .iter()
            .map(|c| (c.cpu_id, c.current_khz))
            .collect();

        let mut lock = Self {
            original_governors,
            original_frequencies,
            sysfs_path: controller.sysfs_path.clone(),
            locked: false,
            mock_mode: controller.mock_mode,
        };

        // Try to set performance governor
        if controller.mock_mode {
            lock.locked = true;
        } else {
            lock.locked = lock.try_set_performance();
        }

        lock
    }

    /// Try to set performance governor on all CPUs
    fn try_set_performance(&self) -> bool {
        for (cpu_id, _) in &self.original_governors {
            let path = self
                .sysfs_path
                .join(format!("cpu{}/cpufreq/scaling_governor", cpu_id));

            if std::fs::write(&path, "performance").is_err() {
                return false;
            }
        }
        true
    }

    /// Check if lock was successful
    pub fn is_locked(&self) -> bool {
        self.locked
    }
}

impl Drop for FrequencyLock {
    fn drop(&mut self) {
        if !self.locked || self.mock_mode {
            return;
        }

        // Restore original governors
        for (cpu_id, governor) in &self.original_governors {
            let path = self
                .sysfs_path
                .join(format!("cpu{}/cpufreq/scaling_governor", cpu_id));

            let _ = std::fs::write(&path, governor.name());
        }
    }
}

/// Get number of CPUs
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

/// Read u64 value from sysfs
fn read_sysfs_value(path: &std::path::Path) -> Option<u64> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Read string from sysfs
fn read_sysfs_string(path: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_governor_names() {
        assert_eq!(CpuGovernor::Performance.name(), "performance");
        assert_eq!(CpuGovernor::Powersave.name(), "powersave");
        assert_eq!(CpuGovernor::Ondemand.name(), "ondemand");
    }

    #[test]
    fn test_governor_from_str() {
        assert_eq!(CpuGovernor::from_str("performance"), CpuGovernor::Performance);
        assert_eq!(CpuGovernor::from_str("POWERSAVE"), CpuGovernor::Powersave);
        assert_eq!(CpuGovernor::from_str("unknown_gov"), CpuGovernor::Unknown);
    }

    #[test]
    fn test_governor_deterministic() {
        assert!(CpuGovernor::Performance.is_deterministic());
        assert!(!CpuGovernor::Ondemand.is_deterministic());
    }

    #[test]
    fn test_mock_controller() {
        let controller = FrequencyController::new().with_mock(3_500_000, CpuGovernor::Performance);

        let info = controller.read_cpu_frequency(0).unwrap();
        assert_eq!(info.current_khz, 3_500_000);
        assert_eq!(info.governor, CpuGovernor::Performance);
    }

    #[test]
    fn test_frequency_info_conversions() {
        let info = CpuFrequencyInfo {
            cpu_id: 0,
            current_khz: 3_500_000,
            min_khz: 800_000,
            max_khz: 4_000_000,
            governor: CpuGovernor::Performance,
            available_governors: vec![],
        };

        assert_eq!(info.current_mhz(), 3500.0);
        assert!((info.current_ghz() - 3.5).abs() < 0.001);
        assert!((info.utilization() - 0.875).abs() < 0.001);
    }

    #[test]
    fn test_frequency_reading() {
        let controller = FrequencyController::new().with_mock(3_000_000, CpuGovernor::Performance);
        let reading = controller.read_all_frequencies();

        assert!(!reading.cpus.is_empty());
        assert!(reading.average_mhz() > 0.0);
    }

    #[test]
    fn test_frequency_variance() {
        let variance = FrequencyVariance {
            mean_mhz: 3000.0,
            std_dev_mhz: 50.0,
            cv_percent: 1.67,
            min_mhz: 2900.0,
            max_mhz: 3100.0,
            sample_count: 10,
        };

        assert!(variance.is_stable());
        assert_eq!(variance.range_mhz(), 200.0);
    }
}
