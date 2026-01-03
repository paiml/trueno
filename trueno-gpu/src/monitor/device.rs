//! Unified Compute Device Abstraction (TRUENO-SPEC-020)
//!
//! Hardware abstraction layer providing a unified interface for CPU, NVIDIA GPU,
//! and AMD GPU monitoring.
//!
//! # Design Principles (Toyota Way)
//!
//! | Principle | Application |
//! |-----------|-------------|
//! | **Genchi Genbutsu** | Direct hardware sampling via native APIs |
//! | **Poka-Yoke** | Type-safe metrics prevent unit confusion |
//!
//! # References
//!
//! - [Nickolls2008] CUDA programming model
//! - [Jia2018] GPU microarchitecture analysis

use std::fmt;

use crate::GpuError;

// ============================================================================
// Device Identification (TRUENO-SPEC-020 Section 2.1)
// ============================================================================

/// Unique identifier for a compute device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId {
    /// Device type discriminant
    pub device_type: DeviceType,
    /// Device index within its type (e.g., GPU 0, GPU 1)
    pub index: u32,
}

impl DeviceId {
    /// Create a new device ID
    #[must_use]
    pub const fn new(device_type: DeviceType, index: u32) -> Self {
        Self { device_type, index }
    }

    /// Create CPU device ID
    #[must_use]
    pub const fn cpu() -> Self {
        Self::new(DeviceType::Cpu, 0)
    }

    /// Create NVIDIA GPU device ID
    #[must_use]
    pub const fn nvidia(index: u32) -> Self {
        Self::new(DeviceType::NvidiaGpu, index)
    }

    /// Create AMD GPU device ID
    #[must_use]
    pub const fn amd(index: u32) -> Self {
        Self::new(DeviceType::AmdGpu, index)
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.device_type {
            DeviceType::Cpu => write!(f, "CPU"),
            DeviceType::NvidiaGpu => write!(f, "NVIDIA:{}", self.index),
            DeviceType::AmdGpu => write!(f, "AMD:{}", self.index),
            DeviceType::IntelGpu => write!(f, "Intel:{}", self.index),
            DeviceType::AppleSilicon => write!(f, "Apple:{}", self.index),
        }
    }
}

/// Type of compute device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU (x86, ARM, etc.)
    Cpu,
    /// NVIDIA GPU (CUDA)
    NvidiaGpu,
    /// AMD GPU (ROCm/HIP)
    AmdGpu,
    /// Intel GPU (oneAPI)
    IntelGpu,
    /// Apple Silicon (Metal)
    AppleSilicon,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::NvidiaGpu => write!(f, "NVIDIA GPU"),
            Self::AmdGpu => write!(f, "AMD GPU"),
            Self::IntelGpu => write!(f, "Intel GPU"),
            Self::AppleSilicon => write!(f, "Apple Silicon"),
        }
    }
}

// ============================================================================
// Unified Device Trait (TRUENO-SPEC-020 Section 2.1)
// ============================================================================

/// Unified compute device abstraction
///
/// All compute devices (CPU, NVIDIA GPU, AMD GPU) implement this trait
/// for consistent monitoring across heterogeneous hardware.
///
/// # Example
///
/// ```rust,ignore
/// use trueno_gpu::monitor::{ComputeDevice, CpuDevice};
///
/// let cpu = CpuDevice::new();
/// println!("CPU: {} @ {:.1}%", cpu.device_name(), cpu.compute_utilization()?);
/// ```
pub trait ComputeDevice: Send + Sync {
    /// Get the unique device identifier
    fn device_id(&self) -> DeviceId;

    /// Get the device name (e.g., "NVIDIA GeForce RTX 4090")
    fn device_name(&self) -> &str;

    /// Get the device type
    fn device_type(&self) -> DeviceType;

    /// Get compute utilization (0.0-100.0%)
    fn compute_utilization(&self) -> Result<f64, GpuError>;

    /// Get compute clock speed in MHz
    fn compute_clock_mhz(&self) -> Result<u32, GpuError>;

    /// Get compute temperature in Celsius
    fn compute_temperature_c(&self) -> Result<f64, GpuError>;

    /// Get current power consumption in Watts
    fn compute_power_watts(&self) -> Result<f64, GpuError>;

    /// Get power limit in Watts
    fn compute_power_limit_watts(&self) -> Result<f64, GpuError>;

    /// Get used memory in bytes
    fn memory_used_bytes(&self) -> Result<u64, GpuError>;

    /// Get total memory in bytes
    fn memory_total_bytes(&self) -> Result<u64, GpuError>;

    /// Get memory bandwidth in GB/s (if available)
    fn memory_bandwidth_gbps(&self) -> Result<f64, GpuError>;

    /// Get number of compute units (SMs for NVIDIA, CUs for AMD, cores for CPU)
    fn compute_unit_count(&self) -> u32;

    /// Get number of active compute units
    fn active_compute_units(&self) -> Result<u32, GpuError>;

    /// Get PCIe TX bytes per second (GPU only)
    fn pcie_tx_bytes_per_sec(&self) -> Result<u64, GpuError>;

    /// Get PCIe RX bytes per second (GPU only)
    fn pcie_rx_bytes_per_sec(&self) -> Result<u64, GpuError>;

    /// Get PCIe generation (1, 2, 3, 4, 5)
    fn pcie_generation(&self) -> u8;

    /// Get PCIe width (x1, x4, x8, x16)
    fn pcie_width(&self) -> u8;

    /// Refresh metrics from hardware
    fn refresh(&mut self) -> Result<(), GpuError>;

    // =========================================================================
    // Default implementations for derived metrics
    // =========================================================================

    /// Get memory usage percentage (0.0-100.0)
    fn memory_usage_percent(&self) -> Result<f64, GpuError> {
        let used = self.memory_used_bytes()?;
        let total = self.memory_total_bytes()?;
        if total == 0 {
            return Ok(0.0);
        }
        Ok((used as f64 / total as f64) * 100.0)
    }

    /// Get available memory in bytes
    fn memory_available_bytes(&self) -> Result<u64, GpuError> {
        let used = self.memory_used_bytes()?;
        let total = self.memory_total_bytes()?;
        Ok(total.saturating_sub(used))
    }

    /// Get memory used in MB
    fn memory_used_mb(&self) -> Result<u64, GpuError> {
        Ok(self.memory_used_bytes()? / (1024 * 1024))
    }

    /// Get memory total in MB
    fn memory_total_mb(&self) -> Result<u64, GpuError> {
        Ok(self.memory_total_bytes()? / (1024 * 1024))
    }

    /// Get memory total in GB
    fn memory_total_gb(&self) -> Result<f64, GpuError> {
        Ok(self.memory_total_bytes()? as f64 / (1024.0 * 1024.0 * 1024.0))
    }

    /// Get power usage percentage (current/limit * 100)
    fn power_usage_percent(&self) -> Result<f64, GpuError> {
        let current = self.compute_power_watts()?;
        let limit = self.compute_power_limit_watts()?;
        if limit == 0.0 {
            return Ok(0.0);
        }
        Ok((current / limit) * 100.0)
    }

    /// Check if device is throttling due to temperature
    fn is_thermal_throttling(&self) -> Result<bool, GpuError> {
        let temp = self.compute_temperature_c()?;
        // Conservative threshold - most GPUs throttle around 83-85°C
        Ok(temp > 80.0)
    }

    /// Check if device is throttling due to power
    fn is_power_throttling(&self) -> Result<bool, GpuError> {
        let percent = self.power_usage_percent()?;
        Ok(percent > 95.0)
    }
}

// ============================================================================
// Throttle Reason (TRUENO-SPEC-020 Section 4.2)
// ============================================================================

/// Reason for compute throttling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThrottleReason {
    /// No throttling
    None,
    /// Thermal throttling (temperature limit)
    Thermal,
    /// Power throttling (power limit)
    Power,
    /// Application-set clock limits
    ApplicationClocks,
    /// Software power cap
    SwPowerCap,
    /// Hardware slowdown (external factors)
    HwSlowdown,
    /// Sync boost throttling
    SyncBoost,
}

impl fmt::Display for ThrottleReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Thermal => write!(f, "Thermal"),
            Self::Power => write!(f, "Power"),
            Self::ApplicationClocks => write!(f, "AppClocks"),
            Self::SwPowerCap => write!(f, "SwPowerCap"),
            Self::HwSlowdown => write!(f, "HwSlowdown"),
            Self::SyncBoost => write!(f, "SyncBoost"),
        }
    }
}

// ============================================================================
// Device Snapshot (for history tracking)
// ============================================================================

/// Point-in-time snapshot of device metrics
#[derive(Debug, Clone)]
pub struct DeviceSnapshot {
    /// Device ID
    pub device_id: DeviceId,
    /// Timestamp (milliseconds since epoch)
    pub timestamp_ms: u64,
    /// Compute utilization (0.0-100.0)
    pub compute_utilization: f64,
    /// Memory used bytes
    pub memory_used_bytes: u64,
    /// Memory total bytes
    pub memory_total_bytes: u64,
    /// Temperature in Celsius
    pub temperature_c: f64,
    /// Power in Watts
    pub power_watts: f64,
    /// Clock speed in MHz
    pub clock_mhz: u32,
}

impl DeviceSnapshot {
    /// Create snapshot from a compute device
    pub fn capture<D: ComputeDevice>(device: &D) -> Result<Self, GpuError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Ok(Self {
            device_id: device.device_id(),
            timestamp_ms: now,
            compute_utilization: device.compute_utilization().unwrap_or(0.0),
            memory_used_bytes: device.memory_used_bytes().unwrap_or(0),
            memory_total_bytes: device.memory_total_bytes().unwrap_or(0),
            temperature_c: device.compute_temperature_c().unwrap_or(0.0),
            power_watts: device.compute_power_watts().unwrap_or(0.0),
            clock_mhz: device.compute_clock_mhz().unwrap_or(0),
        })
    }

    /// Get memory usage percentage
    #[must_use]
    pub fn memory_usage_percent(&self) -> f64 {
        if self.memory_total_bytes == 0 {
            return 0.0;
        }
        (self.memory_used_bytes as f64 / self.memory_total_bytes as f64) * 100.0
    }
}

// ============================================================================
// CPU Device Implementation
// ============================================================================

/// CPU compute device using sysinfo
#[derive(Debug)]
pub struct CpuDevice {
    name: String,
    core_count: u32,
    total_memory: u64,
    // Cached metrics (updated on refresh)
    cpu_usage: f64,
    memory_used: u64,
    temperature: Option<f64>,
}

impl CpuDevice {
    /// Create a new CPU device monitor
    #[must_use]
    pub fn new() -> Self {
        // Get CPU info from /proc/cpuinfo on Linux
        let name = Self::read_cpu_name().unwrap_or_else(|| "Unknown CPU".to_string());
        let core_count = Self::read_core_count();
        let total_memory = Self::read_total_memory();

        Self {
            name,
            core_count,
            total_memory,
            cpu_usage: 0.0,
            memory_used: 0,
            temperature: None,
        }
    }

    fn read_cpu_name() -> Option<String> {
        #[cfg(target_os = "linux")]
        {
            let content = std::fs::read_to_string("/proc/cpuinfo").ok()?;
            for line in content.lines() {
                if line.starts_with("model name") {
                    return line.split(':').nth(1).map(|s| s.trim().to_string());
                }
            }
        }
        None
    }

    fn read_core_count() -> u32 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
                return content
                    .lines()
                    .filter(|line| line.starts_with("processor"))
                    .count() as u32;
            }
        }
        // Fallback
        std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1)
    }

    fn read_total_memory() -> u64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        // Parse "MemTotal:       32847868 kB"
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            if let Ok(kb) = parts[1].parse::<u64>() {
                                return kb * 1024; // Convert to bytes
                            }
                        }
                    }
                }
            }
        }
        0
    }

    fn read_memory_used() -> u64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                let mut total = 0u64;
                let mut available = 0u64;

                for line in content.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if line.starts_with("MemTotal:") {
                            total = parts[1].parse().unwrap_or(0) * 1024;
                        } else if line.starts_with("MemAvailable:") {
                            available = parts[1].parse().unwrap_or(0) * 1024;
                        }
                    }
                }
                return total.saturating_sub(available);
            }
        }
        0
    }

    fn read_cpu_usage() -> f64 {
        #[cfg(target_os = "linux")]
        {
            // Read /proc/stat for CPU usage
            // This is a simplified version - real implementation would track deltas
            if let Ok(content) = std::fs::read_to_string("/proc/stat") {
                for line in content.lines() {
                    if line.starts_with("cpu ") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 5 {
                            let user: u64 = parts[1].parse().unwrap_or(0);
                            let nice: u64 = parts[2].parse().unwrap_or(0);
                            let system: u64 = parts[3].parse().unwrap_or(0);
                            let idle: u64 = parts[4].parse().unwrap_or(0);

                            let total = user + nice + system + idle;
                            let busy = user + nice + system;
                            if total > 0 {
                                return (busy as f64 / total as f64) * 100.0;
                            }
                        }
                    }
                }
            }
        }
        0.0
    }

    fn read_temperature() -> Option<f64> {
        #[cfg(target_os = "linux")]
        {
            // Try hwmon thermal zones
            if let Ok(entries) = std::fs::read_dir("/sys/class/hwmon") {
                for entry in entries.flatten() {
                    let temp_path = entry.path().join("temp1_input");
                    if let Ok(content) = std::fs::read_to_string(&temp_path) {
                        if let Ok(millidegrees) = content.trim().parse::<i64>() {
                            return Some(millidegrees as f64 / 1000.0);
                        }
                    }
                }
            }
            // Fallback to thermal_zone
            if let Ok(content) = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp") {
                if let Ok(millidegrees) = content.trim().parse::<i64>() {
                    return Some(millidegrees as f64 / 1000.0);
                }
            }
        }
        None
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeDevice for CpuDevice {
    fn device_id(&self) -> DeviceId {
        DeviceId::cpu()
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn compute_utilization(&self) -> Result<f64, GpuError> {
        Ok(self.cpu_usage)
    }

    fn compute_clock_mhz(&self) -> Result<u32, GpuError> {
        // Read current CPU frequency
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) =
                std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            {
                if let Ok(khz) = content.trim().parse::<u64>() {
                    return Ok((khz / 1000) as u32);
                }
            }
        }
        Err(GpuError::NotSupported(
            "CPU frequency not available".to_string(),
        ))
    }

    fn compute_temperature_c(&self) -> Result<f64, GpuError> {
        self.temperature.ok_or_else(|| {
            GpuError::NotSupported("CPU temperature not available".to_string())
        })
    }

    fn compute_power_watts(&self) -> Result<f64, GpuError> {
        // CPU power estimation based on TDP and utilization
        // This is a rough estimate - RAPL provides better data on supported CPUs
        Err(GpuError::NotSupported(
            "CPU power not available".to_string(),
        ))
    }

    fn compute_power_limit_watts(&self) -> Result<f64, GpuError> {
        Err(GpuError::NotSupported(
            "CPU power limit not available".to_string(),
        ))
    }

    fn memory_used_bytes(&self) -> Result<u64, GpuError> {
        Ok(self.memory_used)
    }

    fn memory_total_bytes(&self) -> Result<u64, GpuError> {
        Ok(self.total_memory)
    }

    fn memory_bandwidth_gbps(&self) -> Result<f64, GpuError> {
        // Would need memory controller stats - not easily available
        Err(GpuError::NotSupported(
            "Memory bandwidth not available".to_string(),
        ))
    }

    fn compute_unit_count(&self) -> u32 {
        self.core_count
    }

    fn active_compute_units(&self) -> Result<u32, GpuError> {
        // All cores are typically active
        Ok(self.core_count)
    }

    fn pcie_tx_bytes_per_sec(&self) -> Result<u64, GpuError> {
        Err(GpuError::NotSupported("CPU has no PCIe metrics".to_string()))
    }

    fn pcie_rx_bytes_per_sec(&self) -> Result<u64, GpuError> {
        Err(GpuError::NotSupported("CPU has no PCIe metrics".to_string()))
    }

    fn pcie_generation(&self) -> u8 {
        0 // N/A for CPU
    }

    fn pcie_width(&self) -> u8 {
        0 // N/A for CPU
    }

    fn refresh(&mut self) -> Result<(), GpuError> {
        self.cpu_usage = Self::read_cpu_usage();
        self.memory_used = Self::read_memory_used();
        self.temperature = Self::read_temperature();
        Ok(())
    }
}

// ============================================================================
// Tests (Extreme TDD - TRUENO-SPEC-020)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // H001: Device ID Tests
    // =========================================================================

    #[test]
    fn h001_device_id_cpu() {
        let id = DeviceId::cpu();
        assert_eq!(id.device_type, DeviceType::Cpu);
        assert_eq!(id.index, 0);
        assert_eq!(format!("{}", id), "CPU");
    }

    #[test]
    fn h001_device_id_nvidia() {
        let id = DeviceId::nvidia(0);
        assert_eq!(id.device_type, DeviceType::NvidiaGpu);
        assert_eq!(id.index, 0);
        assert_eq!(format!("{}", id), "NVIDIA:0");

        let id2 = DeviceId::nvidia(1);
        assert_eq!(format!("{}", id2), "NVIDIA:1");
    }

    #[test]
    fn h001_device_id_amd() {
        let id = DeviceId::amd(0);
        assert_eq!(id.device_type, DeviceType::AmdGpu);
        assert_eq!(id.index, 0);
        assert_eq!(format!("{}", id), "AMD:0");
    }

    #[test]
    fn h001_device_id_equality() {
        let id1 = DeviceId::nvidia(0);
        let id2 = DeviceId::nvidia(0);
        let id3 = DeviceId::nvidia(1);
        let id4 = DeviceId::amd(0);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_ne!(id1, id4);
    }

    // =========================================================================
    // H002: Device Type Tests
    // =========================================================================

    #[test]
    fn h002_device_type_display() {
        assert_eq!(format!("{}", DeviceType::Cpu), "CPU");
        assert_eq!(format!("{}", DeviceType::NvidiaGpu), "NVIDIA GPU");
        assert_eq!(format!("{}", DeviceType::AmdGpu), "AMD GPU");
        assert_eq!(format!("{}", DeviceType::IntelGpu), "Intel GPU");
        assert_eq!(format!("{}", DeviceType::AppleSilicon), "Apple Silicon");
    }

    // =========================================================================
    // H003: CPU Device Tests
    // =========================================================================

    #[test]
    fn h003_cpu_device_creation() {
        let cpu = CpuDevice::new();
        assert_eq!(cpu.device_type(), DeviceType::Cpu);
        assert_eq!(cpu.device_id(), DeviceId::cpu());
        assert!(cpu.core_count > 0);
    }

    #[test]
    fn h003_cpu_device_default() {
        let cpu = CpuDevice::default();
        assert!(cpu.compute_unit_count() > 0);
    }

    #[test]
    fn h003_cpu_device_name() {
        let cpu = CpuDevice::new();
        // Name should be non-empty
        assert!(!cpu.device_name().is_empty());
    }

    #[test]
    fn h003_cpu_device_memory_total() {
        let cpu = CpuDevice::new();
        // Should have some memory (at least 1GB in practice)
        let total = cpu.memory_total_bytes().unwrap_or(0);
        assert!(total > 0, "CPU should report total memory");
    }

    #[test]
    fn h003_cpu_device_refresh() {
        let mut cpu = CpuDevice::new();
        assert!(cpu.refresh().is_ok());
        // After refresh, metrics should be populated
    }

    // =========================================================================
    // H004: Device Snapshot Tests
    // =========================================================================

    #[test]
    fn h004_device_snapshot_capture() {
        let cpu = CpuDevice::new();
        let snapshot = DeviceSnapshot::capture(&cpu);
        assert!(snapshot.is_ok());

        let snap = snapshot.unwrap();
        assert_eq!(snap.device_id, DeviceId::cpu());
        assert!(snap.timestamp_ms > 0);
    }

    #[test]
    fn h004_device_snapshot_memory_percent() {
        let snap = DeviceSnapshot {
            device_id: DeviceId::cpu(),
            timestamp_ms: 0,
            compute_utilization: 50.0,
            memory_used_bytes: 50 * 1024 * 1024 * 1024, // 50 GB
            memory_total_bytes: 100 * 1024 * 1024 * 1024, // 100 GB
            temperature_c: 45.0,
            power_watts: 100.0,
            clock_mhz: 3000,
        };

        assert!((snap.memory_usage_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn h004_device_snapshot_memory_percent_zero_total() {
        let snap = DeviceSnapshot {
            device_id: DeviceId::cpu(),
            timestamp_ms: 0,
            compute_utilization: 0.0,
            memory_used_bytes: 0,
            memory_total_bytes: 0, // Division by zero case
            temperature_c: 0.0,
            power_watts: 0.0,
            clock_mhz: 0,
        };

        assert!((snap.memory_usage_percent() - 0.0).abs() < 0.01);
    }

    // =========================================================================
    // H005: Throttle Reason Tests
    // =========================================================================

    #[test]
    fn h005_throttle_reason_display() {
        assert_eq!(format!("{}", ThrottleReason::None), "None");
        assert_eq!(format!("{}", ThrottleReason::Thermal), "Thermal");
        assert_eq!(format!("{}", ThrottleReason::Power), "Power");
    }

    // =========================================================================
    // H006: Derived Metrics Tests (ComputeDevice trait defaults)
    // =========================================================================

    #[test]
    fn h006_memory_usage_percent() {
        let cpu = CpuDevice::new();
        let percent = cpu.memory_usage_percent();
        // Should not error and be in valid range
        if let Ok(p) = percent {
            assert!(p >= 0.0 && p <= 100.0);
        }
    }

    #[test]
    fn h006_memory_available_bytes() {
        let cpu = CpuDevice::new();
        if let (Ok(avail), Ok(total)) = (cpu.memory_available_bytes(), cpu.memory_total_bytes()) {
            assert!(avail <= total);
        }
    }

    #[test]
    fn h006_memory_mb_helpers() {
        let cpu = CpuDevice::new();
        if let (Ok(used_mb), Ok(total_mb)) = (cpu.memory_used_mb(), cpu.memory_total_mb()) {
            assert!(used_mb <= total_mb);
        }
    }

    #[test]
    fn h006_memory_gb_helper() {
        let cpu = CpuDevice::new();
        if let Ok(total_gb) = cpu.memory_total_gb() {
            // Should be positive (most systems have > 1GB)
            assert!(total_gb > 0.0);
        }
    }

    // =========================================================================
    // H007: Thermal Throttling Detection
    // =========================================================================

    #[test]
    fn h007_thermal_throttling_detection() {
        let cpu = CpuDevice::new();
        // Just verify it doesn't panic
        let _ = cpu.is_thermal_throttling();
    }

    // =========================================================================
    // H008: Power Throttling Detection
    // =========================================================================

    #[test]
    fn h008_power_throttling_detection() {
        let cpu = CpuDevice::new();
        // CPU doesn't support power metrics, but shouldn't panic
        let _ = cpu.is_power_throttling();
    }

    // =========================================================================
    // H009: Edge Cases
    // =========================================================================

    #[test]
    fn h009_cpu_unsupported_metrics() {
        let cpu = CpuDevice::new();

        // PCIe metrics should return NotSupported
        assert!(cpu.pcie_tx_bytes_per_sec().is_err());
        assert!(cpu.pcie_rx_bytes_per_sec().is_err());
        assert_eq!(cpu.pcie_generation(), 0);
        assert_eq!(cpu.pcie_width(), 0);
    }

    #[test]
    fn h009_device_id_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(DeviceId::cpu());
        set.insert(DeviceId::nvidia(0));
        set.insert(DeviceId::nvidia(1));
        set.insert(DeviceId::amd(0));

        assert_eq!(set.len(), 4);

        // Duplicate should not increase size
        set.insert(DeviceId::cpu());
        assert_eq!(set.len(), 4);
    }

    // =========================================================================
    // H010: Additional Display Coverage
    // =========================================================================

    #[test]
    fn h010_device_id_intel_display() {
        let id = DeviceId::new(DeviceType::IntelGpu, 0);
        assert_eq!(format!("{}", id), "Intel:0");
        assert_eq!(id.device_type, DeviceType::IntelGpu);

        let id2 = DeviceId::new(DeviceType::IntelGpu, 2);
        assert_eq!(format!("{}", id2), "Intel:2");
    }

    #[test]
    fn h010_device_id_apple_display() {
        let id = DeviceId::new(DeviceType::AppleSilicon, 0);
        assert_eq!(format!("{}", id), "Apple:0");
        assert_eq!(id.device_type, DeviceType::AppleSilicon);
    }

    #[test]
    fn h010_throttle_reason_all_variants() {
        // Cover all ThrottleReason Display variants
        assert_eq!(format!("{}", ThrottleReason::None), "None");
        assert_eq!(format!("{}", ThrottleReason::Thermal), "Thermal");
        assert_eq!(format!("{}", ThrottleReason::Power), "Power");
        assert_eq!(format!("{}", ThrottleReason::ApplicationClocks), "AppClocks");
        assert_eq!(format!("{}", ThrottleReason::SwPowerCap), "SwPowerCap");
        assert_eq!(format!("{}", ThrottleReason::HwSlowdown), "HwSlowdown");
        assert_eq!(format!("{}", ThrottleReason::SyncBoost), "SyncBoost");
    }

    // =========================================================================
    // H011: Default Trait Impl Edge Cases
    // =========================================================================

    /// Mock device to test default trait implementations with controlled values
    struct MockDevice {
        mem_used: u64,
        mem_total: u64,
        power_current: f64,
        power_limit: f64,
        temperature: f64,
    }

    impl MockDevice {
        fn new(mem_used: u64, mem_total: u64, power_current: f64, power_limit: f64, temperature: f64) -> Self {
            Self { mem_used, mem_total, power_current, power_limit, temperature }
        }
    }

    impl ComputeDevice for MockDevice {
        fn device_id(&self) -> DeviceId { DeviceId::cpu() }
        fn device_name(&self) -> &str { "Mock" }
        fn device_type(&self) -> DeviceType { DeviceType::Cpu }
        fn compute_utilization(&self) -> Result<f64, GpuError> { Ok(50.0) }
        fn compute_clock_mhz(&self) -> Result<u32, GpuError> { Ok(3000) }
        fn compute_temperature_c(&self) -> Result<f64, GpuError> { Ok(self.temperature) }
        fn compute_power_watts(&self) -> Result<f64, GpuError> { Ok(self.power_current) }
        fn compute_power_limit_watts(&self) -> Result<f64, GpuError> { Ok(self.power_limit) }
        fn memory_used_bytes(&self) -> Result<u64, GpuError> { Ok(self.mem_used) }
        fn memory_total_bytes(&self) -> Result<u64, GpuError> { Ok(self.mem_total) }
        fn memory_bandwidth_gbps(&self) -> Result<f64, GpuError> { Err(GpuError::NotSupported("mock".into())) }
        fn compute_unit_count(&self) -> u32 { 8 }
        fn active_compute_units(&self) -> Result<u32, GpuError> { Ok(8) }
        fn pcie_tx_bytes_per_sec(&self) -> Result<u64, GpuError> { Err(GpuError::NotSupported("mock".into())) }
        fn pcie_rx_bytes_per_sec(&self) -> Result<u64, GpuError> { Err(GpuError::NotSupported("mock".into())) }
        fn pcie_generation(&self) -> u8 { 0 }
        fn pcie_width(&self) -> u8 { 0 }
        fn refresh(&mut self) -> Result<(), GpuError> { Ok(()) }
    }

    #[test]
    fn h011_memory_usage_percent_zero_total() {
        let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
        // Zero total should return 0.0, not divide by zero
        assert!((mock.memory_usage_percent().unwrap() - 0.0).abs() < 0.01);
    }

    #[test]
    fn h011_memory_usage_percent_normal() {
        let mock = MockDevice::new(50 * 1024 * 1024 * 1024, 100 * 1024 * 1024 * 1024, 0.0, 0.0, 0.0);
        // 50% usage
        assert!((mock.memory_usage_percent().unwrap() - 50.0).abs() < 0.01);
    }

    #[test]
    fn h011_memory_available_bytes() {
        let mock = MockDevice::new(30 * 1024 * 1024 * 1024, 100 * 1024 * 1024 * 1024, 0.0, 0.0, 0.0);
        // 70GB available
        let available = mock.memory_available_bytes().unwrap();
        assert_eq!(available, 70 * 1024 * 1024 * 1024);
    }

    #[test]
    fn h011_memory_mb_gb_conversions() {
        let mock = MockDevice::new(1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024, 0.0, 0.0, 0.0);
        // 1GB used = 1024MB
        assert_eq!(mock.memory_used_mb().unwrap(), 1024);
        // 16GB total = 16384MB
        assert_eq!(mock.memory_total_mb().unwrap(), 16384);
        // 16GB as f64
        assert!((mock.memory_total_gb().unwrap() - 16.0).abs() < 0.01);
    }

    #[test]
    fn h011_power_usage_percent_zero_limit() {
        let mock = MockDevice::new(0, 0, 100.0, 0.0, 0.0);
        // Zero limit should return 0.0, not divide by zero
        assert!((mock.power_usage_percent().unwrap() - 0.0).abs() < 0.01);
    }

    #[test]
    fn h011_power_usage_percent_normal() {
        let mock = MockDevice::new(0, 0, 150.0, 300.0, 0.0);
        // 50% power usage
        assert!((mock.power_usage_percent().unwrap() - 50.0).abs() < 0.01);
    }

    #[test]
    fn h011_thermal_throttling_below_threshold() {
        let mock = MockDevice::new(0, 0, 0.0, 0.0, 75.0);
        // Below 80°C - no throttling
        assert!(!mock.is_thermal_throttling().unwrap());
    }

    #[test]
    fn h011_thermal_throttling_above_threshold() {
        let mock = MockDevice::new(0, 0, 0.0, 0.0, 85.0);
        // Above 80°C - throttling
        assert!(mock.is_thermal_throttling().unwrap());
    }

    #[test]
    fn h011_power_throttling_below_threshold() {
        let mock = MockDevice::new(0, 0, 90.0, 100.0, 0.0);
        // 90% - below 95% threshold
        assert!(!mock.is_power_throttling().unwrap());
    }

    #[test]
    fn h011_power_throttling_above_threshold() {
        let mock = MockDevice::new(0, 0, 98.0, 100.0, 0.0);
        // 98% - above 95% threshold
        assert!(mock.is_power_throttling().unwrap());
    }

    // =========================================================================
    // H012: DeviceSnapshot Edge Cases
    // =========================================================================

    #[test]
    fn h012_device_snapshot_from_mock() {
        let mock = MockDevice::new(8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024, 150.0, 300.0, 65.0);
        let snapshot = DeviceSnapshot::capture(&mock).unwrap();

        assert_eq!(snapshot.device_id, DeviceId::cpu());
        assert!((snapshot.compute_utilization - 50.0).abs() < 0.01);
        assert_eq!(snapshot.memory_used_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(snapshot.memory_total_bytes, 16 * 1024 * 1024 * 1024);
        assert!((snapshot.temperature_c - 65.0).abs() < 0.01);
        assert!((snapshot.power_watts - 150.0).abs() < 0.01);
        assert_eq!(snapshot.clock_mhz, 3000);
    }
}
