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

/// Governor spec: (variant, string name).
const GOVERNOR_SPECS: &[(CpuGovernor, &str)] = &[
    (CpuGovernor::Performance, "performance"),
    (CpuGovernor::Powersave, "powersave"),
    (CpuGovernor::Ondemand, "ondemand"),
    (CpuGovernor::Conservative, "conservative"),
    (CpuGovernor::Schedutil, "schedutil"),
    (CpuGovernor::Userspace, "userspace"),
];

impl CpuGovernor {
    /// Get governor name
    pub fn name(&self) -> &'static str {
        GOVERNOR_SPECS
            .iter()
            .find(|(v, _)| v == self)
            .map(|(_, n)| *n)
            .unwrap_or("unknown")
    }

    /// Parse from string
    pub fn parse(s: &str) -> Self {
        let lower = s.trim().to_lowercase();
        GOVERNOR_SPECS
            .iter()
            .find(|(_, n)| *n == lower)
            .map(|(v, _)| *v)
            .unwrap_or(Self::Unknown)
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
    /// Convert kHz to MHz.
    fn khz_to_mhz(khz: u64) -> f64 {
        khz as f64 / 1_000.0
    }

    /// Get current frequency in MHz
    pub fn current_mhz(&self) -> f64 {
        Self::khz_to_mhz(self.current_khz)
    }

    /// Get current frequency in GHz
    pub fn current_ghz(&self) -> f64 {
        Self::khz_to_mhz(self.current_khz) / 1_000.0
    }

    /// Get frequency utilization (current / max)
    pub fn utilization(&self) -> f64 {
        if self.max_khz > 0 {
            self.current_khz as f64 / self.max_khz as f64
        } else {
            1.0
        }
    }

    /// Build a mock CpuFrequencyInfo for testing.
    fn mock(cpu_id: usize, frequency_khz: u64, governor: CpuGovernor) -> Self {
        Self {
            cpu_id,
            current_khz: frequency_khz,
            min_khz: 800_000,
            max_khz: 4_000_000,
            governor,
            available_governors: vec![CpuGovernor::Performance, CpuGovernor::Powersave],
        }
    }

    /// Read CpuFrequencyInfo from sysfs for the given CPU.
    fn from_sysfs(cpu_id: usize, sysfs_path: &std::path::Path) -> Self {
        let cpu_path = sysfs_path.join(format!("cpu{cpu_id}/cpufreq"));

        let read_khz = |name: &str| read_sysfs_value(&cpu_path.join(name)).unwrap_or(0);

        let governor = read_sysfs_string(&cpu_path.join("scaling_governor"))
            .map(|s| CpuGovernor::parse(&s))
            .unwrap_or(CpuGovernor::Unknown);

        let available_governors: Vec<CpuGovernor> =
            read_sysfs_string(&cpu_path.join("scaling_available_governors"))
                .unwrap_or_default()
                .split_whitespace()
                .map(CpuGovernor::parse)
                .collect();

        Self {
            cpu_id,
            current_khz: read_khz("scaling_cur_freq"),
            min_khz: read_khz("scaling_min_freq"),
            max_khz: read_khz("scaling_max_freq"),
            governor,
            available_governors,
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
        self.extremum_mhz(f64::min)
    }

    /// Get max frequency across cores in MHz
    pub fn max_mhz(&self) -> f64 {
        self.extremum_mhz(f64::max)
    }

    /// Get frequency variance in MHz
    pub fn variance_mhz(&self) -> f64 {
        self.max_mhz() - self.min_mhz()
    }

    /// Shared extremum helper (min or max across core frequencies).
    fn extremum_mhz(&self, cmp: fn(f64, f64) -> f64) -> f64 {
        self.cpus
            .iter()
            .map(|c| c.current_mhz())
            .reduce(cmp)
            .expect("non-empty collection")
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
        self.cpus
            .first()
            .map(|c| c.governor)
            .unwrap_or(CpuGovernor::Unknown)
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
        Some(if self.mock_mode {
            CpuFrequencyInfo::mock(cpu_id, self.mock_frequency, self.mock_governor)
        } else {
            CpuFrequencyInfo::from_sysfs(cpu_id, &self.sysfs_path)
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

        let govs = &reading.cpus[0].available_governors;
        govs.contains(&CpuGovernor::Userspace) || govs.contains(&CpuGovernor::Performance)
    }

    /// Measure frequency variance over time
    pub fn measure_variance(&self, samples: usize, interval_ms: u64) -> FrequencyVariance {
        let mut readings = Vec::with_capacity(samples);
        for _ in 0..samples {
            readings.push(self.read_all_frequencies().average_mhz());
            std::thread::sleep(std::time::Duration::from_millis(interval_ms));
        }
        FrequencyVariance::from_samples(&readings)
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
    /// Compute variance statistics from a set of frequency samples (MHz).
    pub fn from_samples(readings: &[f64]) -> Self {
        if readings.is_empty() {
            return Self::default();
        }
        let n = readings.len() as f64;
        let mean = readings.iter().sum::<f64>() / n;
        let var = readings.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let std_dev = var.sqrt();
        let cv = if mean > 0.0 { std_dev / mean * 100.0 } else { 0.0 };
        Self {
            mean_mhz: mean,
            std_dev_mhz: std_dev,
            cv_percent: cv,
            min_mhz: readings.iter().copied().fold(f64::INFINITY, f64::min),
            max_mhz: readings.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            sample_count: readings.len(),
        }
    }

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

        let (original_governors, original_frequencies): (Vec<_>, Vec<_>) = reading
            .cpus
            .iter()
            .map(|c| ((c.cpu_id, c.governor), (c.cpu_id, c.current_khz)))
            .unzip();

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
            if std::fs::write(governor_path(&self.sysfs_path, *cpu_id), "performance").is_err() {
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
            let _ = std::fs::write(governor_path(&self.sysfs_path, *cpu_id), governor.name());
        }
    }
}

/// Get number of CPUs
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

/// Read and trim a sysfs file. Returns None on I/O error.
fn read_sysfs_trimmed(path: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

/// Read u64 value from sysfs
fn read_sysfs_value(path: &std::path::Path) -> Option<u64> {
    read_sysfs_trimmed(path).and_then(|s| s.parse().ok())
}

/// Read string from sysfs
fn read_sysfs_string(path: &std::path::Path) -> Option<String> {
    read_sysfs_trimmed(path)
}

/// Build sysfs governor path for a given CPU.
fn governor_path(sysfs_path: &std::path::Path, cpu_id: usize) -> PathBuf {
    sysfs_path.join(format!("cpu{cpu_id}/cpufreq/scaling_governor"))
}


#[cfg(test)]
mod tests;
