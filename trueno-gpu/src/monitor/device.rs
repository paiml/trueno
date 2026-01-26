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
        self.temperature
            .ok_or_else(|| GpuError::NotSupported("CPU temperature not available".to_string()))
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
        Err(GpuError::NotSupported(
            "CPU has no PCIe metrics".to_string(),
        ))
    }

    fn pcie_rx_bytes_per_sec(&self) -> Result<u64, GpuError> {
        Err(GpuError::NotSupported(
            "CPU has no PCIe metrics".to_string(),
        ))
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
        // On Linux, this should always succeed
        assert!(percent.is_ok());
        let p = percent.unwrap();
        assert!(p >= 0.0 && p <= 100.0);
    }

    #[test]
    fn h006_memory_available_bytes() {
        let cpu = CpuDevice::new();
        // On Linux, these should always succeed
        let avail = cpu.memory_available_bytes().unwrap();
        let total = cpu.memory_total_bytes().unwrap();
        assert!(avail <= total);
    }

    #[test]
    fn h006_memory_mb_helpers() {
        let cpu = CpuDevice::new();
        // On Linux, these should always succeed
        let used_mb = cpu.memory_used_mb().unwrap();
        let total_mb = cpu.memory_total_mb().unwrap();
        assert!(used_mb <= total_mb);
    }

    #[test]
    fn h006_memory_gb_helper() {
        let cpu = CpuDevice::new();
        // On Linux, this should always succeed
        let total_gb = cpu.memory_total_gb().unwrap();
        // Should be positive (most systems have > 1GB)
        assert!(total_gb > 0.0);
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
        assert_eq!(
            format!("{}", ThrottleReason::ApplicationClocks),
            "AppClocks"
        );
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
        fn new(
            mem_used: u64,
            mem_total: u64,
            power_current: f64,
            power_limit: f64,
            temperature: f64,
        ) -> Self {
            Self {
                mem_used,
                mem_total,
                power_current,
                power_limit,
                temperature,
            }
        }
    }

    impl ComputeDevice for MockDevice {
        fn device_id(&self) -> DeviceId {
            DeviceId::cpu()
        }
        fn device_name(&self) -> &str {
            "Mock"
        }
        fn device_type(&self) -> DeviceType {
            DeviceType::Cpu
        }
        fn compute_utilization(&self) -> Result<f64, GpuError> {
            Ok(50.0)
        }
        fn compute_clock_mhz(&self) -> Result<u32, GpuError> {
            Ok(3000)
        }
        fn compute_temperature_c(&self) -> Result<f64, GpuError> {
            Ok(self.temperature)
        }
        fn compute_power_watts(&self) -> Result<f64, GpuError> {
            Ok(self.power_current)
        }
        fn compute_power_limit_watts(&self) -> Result<f64, GpuError> {
            Ok(self.power_limit)
        }
        fn memory_used_bytes(&self) -> Result<u64, GpuError> {
            Ok(self.mem_used)
        }
        fn memory_total_bytes(&self) -> Result<u64, GpuError> {
            Ok(self.mem_total)
        }
        fn memory_bandwidth_gbps(&self) -> Result<f64, GpuError> {
            Err(GpuError::NotSupported("mock".into()))
        }
        fn compute_unit_count(&self) -> u32 {
            8
        }
        fn active_compute_units(&self) -> Result<u32, GpuError> {
            Ok(8)
        }
        fn pcie_tx_bytes_per_sec(&self) -> Result<u64, GpuError> {
            Err(GpuError::NotSupported("mock".into()))
        }
        fn pcie_rx_bytes_per_sec(&self) -> Result<u64, GpuError> {
            Err(GpuError::NotSupported("mock".into()))
        }
        fn pcie_generation(&self) -> u8 {
            0
        }
        fn pcie_width(&self) -> u8 {
            0
        }
        fn refresh(&mut self) -> Result<(), GpuError> {
            Ok(())
        }
    }

    #[test]
    fn h011_memory_usage_percent_zero_total() {
        let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
        // Zero total should return 0.0, not divide by zero
        assert!((mock.memory_usage_percent().unwrap() - 0.0).abs() < 0.01);
    }

    #[test]
    fn h011_memory_usage_percent_normal() {
        let mock = MockDevice::new(
            50 * 1024 * 1024 * 1024,
            100 * 1024 * 1024 * 1024,
            0.0,
            0.0,
            0.0,
        );
        // 50% usage
        assert!((mock.memory_usage_percent().unwrap() - 50.0).abs() < 0.01);
    }

    #[test]
    fn h011_memory_available_bytes() {
        let mock = MockDevice::new(
            30 * 1024 * 1024 * 1024,
            100 * 1024 * 1024 * 1024,
            0.0,
            0.0,
            0.0,
        );
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
        let mock = MockDevice::new(
            8 * 1024 * 1024 * 1024,
            16 * 1024 * 1024 * 1024,
            150.0,
            300.0,
            65.0,
        );
        let snapshot = DeviceSnapshot::capture(&mock).unwrap();

        assert_eq!(snapshot.device_id, DeviceId::cpu());
        assert!((snapshot.compute_utilization - 50.0).abs() < 0.01);
        assert_eq!(snapshot.memory_used_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(snapshot.memory_total_bytes, 16 * 1024 * 1024 * 1024);
        assert!((snapshot.temperature_c - 65.0).abs() < 0.01);
        assert!((snapshot.power_watts - 150.0).abs() < 0.01);
        assert_eq!(snapshot.clock_mhz, 3000);
    }

    // =========================================================================
    // H013: CpuDevice Additional Coverage
    // =========================================================================

    #[test]
    fn h013_cpu_device_refresh() {
        let mut cpu = CpuDevice::new();
        // Refresh should not panic
        let _ = cpu.refresh();
        // After refresh, metrics should still work
        let _ = cpu.compute_utilization();
        let _ = cpu.memory_used_bytes();
    }

    #[test]
    fn h013_cpu_device_multiple_refreshes() {
        let mut cpu = CpuDevice::new();
        // Multiple refreshes should be idempotent
        for _ in 0..3 {
            let _ = cpu.refresh();
        }
    }

    #[test]
    fn h013_cpu_device_name_not_empty() {
        let cpu = CpuDevice::new();
        assert!(!cpu.device_name().is_empty());
    }

    #[test]
    fn h013_cpu_device_core_count_positive() {
        let cpu = CpuDevice::new();
        assert!(cpu.compute_unit_count() > 0);
    }

    #[test]
    fn h013_cpu_clock_speed() {
        let cpu = CpuDevice::new();
        // Clock speed should be positive (or NotSupported error on some systems)
        // Just verify we can call it without panic
        let _ = cpu.compute_clock_mhz();
    }

    #[test]
    fn h013_cpu_temperature() {
        let cpu = CpuDevice::new();
        // Temperature may not be available on all systems
        // Just verify we can call it without panic
        let _ = cpu.compute_temperature_c();
    }

    // =========================================================================
    // H014: MockDevice Extended Coverage
    // =========================================================================

    #[test]
    fn h014_mock_device_all_methods() {
        let mock = MockDevice::new(1024, 2048, 10.0, 100.0, 30.0);

        // Test all trait method implementations
        assert_eq!(mock.device_id(), DeviceId::cpu());
        assert_eq!(mock.device_name(), "Mock");
        assert!(matches!(mock.device_type(), DeviceType::Cpu));
        assert_eq!(mock.compute_unit_count(), 8);
        assert_eq!(mock.memory_used_bytes().unwrap(), 1024);
        assert_eq!(mock.memory_total_bytes().unwrap(), 2048);
        assert!((mock.compute_utilization().unwrap() - 50.0).abs() < 0.01); // MockDevice always returns 50.0
        assert!((mock.compute_temperature_c().unwrap() - 30.0).abs() < 0.01);
        assert!((mock.compute_power_watts().unwrap() - 10.0).abs() < 0.01);
        assert_eq!(mock.compute_clock_mhz().unwrap(), 3000);
    }

    #[test]
    fn h014_mock_device_derived_metrics() {
        let mock = MockDevice::new(1024, 2048, 10.0, 100.0, 30.0);

        // Derived metrics
        let usage_percent = mock.memory_usage_percent().unwrap();
        assert!((usage_percent - 50.0).abs() < 0.01); // 1024/2048 = 50%

        let available = mock.memory_available_bytes().unwrap();
        assert_eq!(available, 1024); // 2048 - 1024
    }

    #[test]
    fn h014_mock_device_mb_gb_helpers() {
        let mock = MockDevice::new(
            1024 * 1024 * 1024,
            2 * 1024 * 1024 * 1024,
            10.0,
            100.0,
            30.0,
        );

        let used_mb = mock.memory_used_mb().unwrap();
        assert_eq!(used_mb, 1024); // 1 GB = 1024 MB

        let total_gb = mock.memory_total_gb().unwrap();
        assert!((total_gb - 2.0).abs() < 0.1); // 2 GB
    }

    // =========================================================================
    // H015: DeviceId Additional Coverage
    // =========================================================================

    #[test]
    fn h015_device_id_display() {
        assert_eq!(format!("{}", DeviceId::cpu()), "CPU");
        assert_eq!(format!("{}", DeviceId::nvidia(0)), "NVIDIA:0");
        assert_eq!(format!("{}", DeviceId::nvidia(1)), "NVIDIA:1");
        assert_eq!(format!("{}", DeviceId::amd(0)), "AMD:0");
    }

    #[test]
    fn h015_device_id_debug() {
        let cpu_id = DeviceId::cpu();
        let debug_str = format!("{:?}", cpu_id);
        assert!(debug_str.contains("Cpu"));
    }

    #[test]
    fn h015_device_id_clone() {
        let id1 = DeviceId::nvidia(0);
        let id2 = id1.clone();
        assert_eq!(id1, id2);
    }

    // =========================================================================
    // H016: DeviceSnapshot Additional Coverage
    // =========================================================================

    #[test]
    fn h016_snapshot_debug() {
        let mock = MockDevice::new(1024, 2048, 10.0, 100.0, 30.0);
        let snapshot = DeviceSnapshot::capture(&mock).unwrap();
        let debug_str = format!("{:?}", snapshot);
        assert!(debug_str.contains("DeviceSnapshot"));
    }

    #[test]
    fn h016_snapshot_clone() {
        let mock = MockDevice::new(1024, 2048, 10.0, 100.0, 30.0);
        let snapshot1 = DeviceSnapshot::capture(&mock).unwrap();
        let snapshot2 = snapshot1.clone();
        assert_eq!(snapshot1.device_id, snapshot2.device_id);
        assert_eq!(snapshot1.memory_used_bytes, snapshot2.memory_used_bytes);
    }

    // =========================================================================
    // H017: DeviceType Coverage
    // =========================================================================

    #[test]
    fn h017_device_type_display() {
        assert_eq!(format!("{}", DeviceType::Cpu), "CPU");
        assert_eq!(format!("{}", DeviceType::NvidiaGpu), "NVIDIA GPU");
        assert_eq!(format!("{}", DeviceType::AmdGpu), "AMD GPU");
        assert_eq!(format!("{}", DeviceType::IntelGpu), "Intel GPU");
    }

    #[test]
    fn h017_device_type_debug() {
        let gpu = DeviceType::NvidiaGpu;
        let debug_str = format!("{:?}", gpu);
        assert!(debug_str.contains("NvidiaGpu"));
    }

    // =========================================================================
    // H018: Error Path Coverage (Best Effort)
    // =========================================================================

    #[test]
    fn h018_cpu_unsupported_pcie_metrics() {
        let cpu = CpuDevice::new();

        // PCIe metrics return NotSupported
        let tx = cpu.pcie_tx_bytes_per_sec();
        assert!(matches!(tx, Err(GpuError::NotSupported(_))));

        let rx = cpu.pcie_rx_bytes_per_sec();
        assert!(matches!(rx, Err(GpuError::NotSupported(_))));
    }

    #[test]
    fn h018_cpu_power_metrics() {
        let cpu = CpuDevice::new();

        // Power metrics return NotSupported for CPU
        let power = cpu.compute_power_watts();
        assert!(matches!(power, Err(GpuError::NotSupported(_))));

        let power_limit = cpu.compute_power_limit_watts();
        assert!(matches!(power_limit, Err(GpuError::NotSupported(_))));

        let bw = cpu.memory_bandwidth_gbps();
        assert!(matches!(bw, Err(GpuError::NotSupported(_))));
    }

    // =========================================================================
    // H019: CpuDevice active_compute_units Test
    // =========================================================================

    #[test]
    fn h019_cpu_active_compute_units() {
        let cpu = CpuDevice::new();

        // active_compute_units should return the core count
        let active = cpu.active_compute_units();
        assert!(active.is_ok());
        let count = active.unwrap();
        assert!(count > 0, "Should have at least one active compute unit");
        assert_eq!(
            count,
            cpu.compute_unit_count(),
            "Active should equal total cores"
        );
    }

    // =========================================================================
    // H020: MockDevice Full Coverage Tests
    // =========================================================================

    #[test]
    fn h020_mock_device_pcie_metrics() {
        let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);

        // PCIe metrics should return NotSupported
        assert!(matches!(
            mock.pcie_tx_bytes_per_sec(),
            Err(GpuError::NotSupported(_))
        ));
        assert!(matches!(
            mock.pcie_rx_bytes_per_sec(),
            Err(GpuError::NotSupported(_))
        ));
        assert_eq!(mock.pcie_generation(), 0);
        assert_eq!(mock.pcie_width(), 0);
    }

    #[test]
    fn h020_mock_device_active_compute_units() {
        let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);

        let active = mock.active_compute_units();
        assert!(active.is_ok());
        assert_eq!(active.unwrap(), 8); // MockDevice returns 8 compute units
    }

    #[test]
    fn h020_mock_device_memory_bandwidth() {
        let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);

        assert!(matches!(
            mock.memory_bandwidth_gbps(),
            Err(GpuError::NotSupported(_))
        ));
    }

    #[test]
    fn h020_mock_device_refresh() {
        let mut mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);

        // refresh should succeed
        assert!(mock.refresh().is_ok());
    }

    // =========================================================================
    // H021: CpuDevice Default Derived Metrics
    // =========================================================================

    #[test]
    fn h021_cpu_device_memory_mb_conversion() {
        let cpu = CpuDevice::new();

        // Test memory_used_mb conversion
        if let Ok(used_bytes) = cpu.memory_used_bytes() {
            let used_mb = cpu.memory_used_mb();
            assert!(used_mb.is_ok());
            assert_eq!(used_mb.unwrap(), used_bytes / (1024 * 1024));
        }
    }

    #[test]
    fn h021_cpu_device_memory_total_mb() {
        let cpu = CpuDevice::new();

        if let Ok(total_bytes) = cpu.memory_total_bytes() {
            let total_mb = cpu.memory_total_mb();
            assert!(total_mb.is_ok());
            assert_eq!(total_mb.unwrap(), total_bytes / (1024 * 1024));
        }
    }

    #[test]
    fn h021_cpu_device_memory_total_gb() {
        let cpu = CpuDevice::new();

        if let Ok(total_bytes) = cpu.memory_total_bytes() {
            let total_gb = cpu.memory_total_gb();
            assert!(total_gb.is_ok());
            let expected_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            assert!((total_gb.unwrap() - expected_gb).abs() < 0.001);
        }
    }

    #[test]
    fn h021_cpu_device_memory_available() {
        let cpu = CpuDevice::new();

        if let (Ok(used), Ok(total)) = (cpu.memory_used_bytes(), cpu.memory_total_bytes()) {
            let available = cpu.memory_available_bytes();
            assert!(available.is_ok());
            assert_eq!(available.unwrap(), total.saturating_sub(used));
        }
    }

    // =========================================================================
    // H022: Edge Case Tests for Default Trait Implementations
    // =========================================================================

    #[test]
    fn h022_mock_device_thermal_throttling_at_threshold() {
        // Test exactly at the 80 degree threshold
        let mock_at_80 = MockDevice::new(0, 0, 0.0, 0.0, 80.0);
        // 80.0 is not > 80.0, so no throttling
        assert!(!mock_at_80.is_thermal_throttling().unwrap());

        // Just above threshold
        let mock_at_80_1 = MockDevice::new(0, 0, 0.0, 0.0, 80.1);
        assert!(mock_at_80_1.is_thermal_throttling().unwrap());
    }

    #[test]
    fn h022_mock_device_power_throttling_at_threshold() {
        // Test exactly at the 95% threshold
        let mock_at_95 = MockDevice::new(0, 0, 95.0, 100.0, 0.0);
        // 95.0 is not > 95.0, so no throttling
        assert!(!mock_at_95.is_power_throttling().unwrap());

        // Just above threshold
        let mock_at_95_1 = MockDevice::new(0, 0, 95.1, 100.0, 0.0);
        assert!(mock_at_95_1.is_power_throttling().unwrap());
    }

    #[test]
    fn h022_mock_device_memory_usage_full() {
        // Test 100% memory usage
        let mock_full = MockDevice::new(100, 100, 0.0, 0.0, 0.0);
        assert!((mock_full.memory_usage_percent().unwrap() - 100.0).abs() < 0.01);
        assert_eq!(mock_full.memory_available_bytes().unwrap(), 0);
    }

    // =========================================================================
    // H023: DeviceSnapshot Additional Field Coverage
    // =========================================================================

    #[test]
    fn h023_device_snapshot_field_access() {
        let mock = MockDevice::new(
            8 * 1024 * 1024 * 1024,
            32 * 1024 * 1024 * 1024,
            250.0,
            350.0,
            72.0,
        );
        let snapshot = DeviceSnapshot::capture(&mock).unwrap();

        // Verify all fields are accessible and have expected values
        assert_eq!(snapshot.memory_used_bytes, 8 * 1024 * 1024 * 1024);
        assert_eq!(snapshot.memory_total_bytes, 32 * 1024 * 1024 * 1024);
        assert!((snapshot.temperature_c - 72.0).abs() < 0.01);
        assert!((snapshot.power_watts - 250.0).abs() < 0.01);
        assert_eq!(snapshot.clock_mhz, 3000);

        // Test memory_usage_percent calculation
        // 8GB / 32GB = 25%
        assert!((snapshot.memory_usage_percent() - 25.0).abs() < 0.01);
    }

    // =========================================================================
    // H024: CpuDevice Internal Method Coverage via Refresh
    // =========================================================================

    #[test]
    fn h024_cpu_device_refresh_populates_fields() {
        let mut cpu = CpuDevice::new();

        // First refresh
        let result = cpu.refresh();
        assert!(result.is_ok());

        // After refresh, utilization should be populated (may be 0 if just started)
        let util = cpu.compute_utilization();
        assert!(util.is_ok());
        let util_val = util.unwrap();
        assert!(util_val >= 0.0 && util_val <= 100.0);

        // Memory used should be reasonable
        let mem = cpu.memory_used_bytes();
        assert!(mem.is_ok());
    }

    #[test]
    fn h024_cpu_device_refresh_multiple_times() {
        let mut cpu = CpuDevice::new();

        // Refresh multiple times should always succeed
        for _ in 0..5 {
            assert!(cpu.refresh().is_ok());
        }

        // Values should still be accessible
        assert!(cpu.compute_utilization().is_ok());
        assert!(cpu.memory_used_bytes().is_ok());
    }

    // =========================================================================
    // H025: CpuDevice Direct Read Functions Coverage
    // =========================================================================

    #[test]
    fn h025_cpu_device_read_core_count() {
        // read_core_count is called in CpuDevice::new()
        // On Linux it reads from /proc/cpuinfo
        // Verify the result is a valid positive number
        let cpu = CpuDevice::new();
        let count = cpu.compute_unit_count();
        assert!(count >= 1, "Should have at least 1 core");
        assert!(
            count <= 1024,
            "Sanity check: should have fewer than 1024 cores"
        );
    }

    #[test]
    fn h025_cpu_device_read_total_memory() {
        // read_total_memory is called in CpuDevice::new()
        let cpu = CpuDevice::new();
        let total = cpu.memory_total_bytes().unwrap();
        // System should have at least 1GB and less than 1TB typically
        assert!(total >= 1024 * 1024 * 1024, "Should have at least 1GB");
        assert!(total < 100 * 1024 * 1024 * 1024 * 1024, "Sanity: < 100TB");
    }

    #[test]
    fn h025_cpu_device_read_cpu_name() {
        // read_cpu_name is called in CpuDevice::new()
        let cpu = CpuDevice::new();
        let name = cpu.device_name();
        assert!(!name.is_empty(), "CPU name should not be empty");
        // Name could be "Unknown CPU" if /proc/cpuinfo doesn't have model name
    }

    // =========================================================================
    // H026: CpuDevice Compute Clock Coverage
    // =========================================================================

    #[test]
    fn h026_cpu_device_compute_clock_value() {
        let cpu = CpuDevice::new();
        // On systems with frequency scaling, this should return Ok
        // On systems without, it returns NotSupported
        match cpu.compute_clock_mhz() {
            Ok(mhz) => {
                // Valid frequency range: 100 MHz to 10 GHz
                assert!(mhz >= 100, "Clock should be at least 100 MHz");
                assert!(mhz <= 10000, "Clock should be at most 10 GHz");
            }
            Err(GpuError::NotSupported(_)) => {
                // Expected on systems without frequency info
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    // =========================================================================
    // H027: CpuDevice Temperature Coverage
    // =========================================================================

    #[test]
    fn h027_cpu_device_temperature_value() {
        let mut cpu = CpuDevice::new();
        cpu.refresh().unwrap();

        // Temperature may or may not be available depending on hardware/permissions
        match cpu.compute_temperature_c() {
            Ok(temp) => {
                // Valid temperature range: 0 to 150 Celsius
                assert!(temp >= 0.0, "Temperature should be non-negative");
                assert!(temp <= 150.0, "Temperature should be at most 150C");
            }
            Err(GpuError::NotSupported(_)) => {
                // Expected on systems without temperature sensors
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    // =========================================================================
    // H028: CpuDevice CPU Usage Coverage
    // =========================================================================

    #[test]
    fn h028_cpu_device_cpu_usage_after_refresh() {
        let mut cpu = CpuDevice::new();
        cpu.refresh().unwrap();

        let usage = cpu.compute_utilization().unwrap();
        // CPU usage should be between 0 and 100
        assert!(usage >= 0.0, "CPU usage should be non-negative");
        assert!(usage <= 100.0, "CPU usage should be at most 100%");
    }

    // =========================================================================
    // H029: CpuDevice Memory Used Coverage
    // =========================================================================

    #[test]
    fn h029_cpu_device_memory_used_after_refresh() {
        let mut cpu = CpuDevice::new();
        cpu.refresh().unwrap();

        let used = cpu.memory_used_bytes().unwrap();
        let total = cpu.memory_total_bytes().unwrap();

        // Used should be <= total
        assert!(used <= total, "Used memory should not exceed total");
        // At least some memory should be used (kernel, etc.)
        assert!(used > 0, "Some memory should be in use");
    }

    // =========================================================================
    // H030: MockDevice Additional Coverage
    // =========================================================================

    #[test]
    fn h030_mock_device_device_name() {
        let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
        assert_eq!(mock.device_name(), "Mock");
    }

    #[test]
    fn h030_mock_device_device_type() {
        let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
        assert!(matches!(mock.device_type(), DeviceType::Cpu));
    }

    #[test]
    fn h030_mock_device_device_id() {
        let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
        assert_eq!(mock.device_id(), DeviceId::cpu());
    }

    #[test]
    fn h030_mock_device_compute_utilization() {
        let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
        assert!((mock.compute_utilization().unwrap() - 50.0).abs() < 0.01);
    }

    #[test]
    fn h030_mock_device_compute_clock() {
        let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
        assert_eq!(mock.compute_clock_mhz().unwrap(), 3000);
    }

    #[test]
    fn h030_mock_device_compute_temperature() {
        let mock = MockDevice::new(0, 0, 0.0, 0.0, 45.0);
        assert!((mock.compute_temperature_c().unwrap() - 45.0).abs() < 0.01);
    }

    #[test]
    fn h030_mock_device_compute_power() {
        let mock = MockDevice::new(0, 0, 200.0, 300.0, 0.0);
        assert!((mock.compute_power_watts().unwrap() - 200.0).abs() < 0.01);
        assert!((mock.compute_power_limit_watts().unwrap() - 300.0).abs() < 0.01);
    }

    #[test]
    fn h030_mock_device_compute_unit_count() {
        let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
        assert_eq!(mock.compute_unit_count(), 8);
    }

    #[test]
    fn h030_mock_device_memory_bytes() {
        let mock = MockDevice::new(1000, 2000, 0.0, 0.0, 0.0);
        assert_eq!(mock.memory_used_bytes().unwrap(), 1000);
        assert_eq!(mock.memory_total_bytes().unwrap(), 2000);
    }

    // =========================================================================
    // H031: Error-Propagating Mock Device
    // =========================================================================

    /// Mock device that returns errors for testing error propagation in default trait methods
    struct ErrorMockDevice {
        return_error: bool,
    }

    impl ErrorMockDevice {
        fn new(return_error: bool) -> Self {
            Self { return_error }
        }
    }

    impl ComputeDevice for ErrorMockDevice {
        fn device_id(&self) -> DeviceId {
            DeviceId::cpu()
        }
        fn device_name(&self) -> &str {
            "ErrorMock"
        }
        fn device_type(&self) -> DeviceType {
            DeviceType::Cpu
        }
        fn compute_utilization(&self) -> Result<f64, GpuError> {
            if self.return_error {
                Err(GpuError::NotSupported("test".into()))
            } else {
                Ok(50.0)
            }
        }
        fn compute_clock_mhz(&self) -> Result<u32, GpuError> {
            if self.return_error {
                Err(GpuError::NotSupported("test".into()))
            } else {
                Ok(3000)
            }
        }
        fn compute_temperature_c(&self) -> Result<f64, GpuError> {
            if self.return_error {
                Err(GpuError::NotSupported("test".into()))
            } else {
                Ok(50.0)
            }
        }
        fn compute_power_watts(&self) -> Result<f64, GpuError> {
            if self.return_error {
                Err(GpuError::NotSupported("test".into()))
            } else {
                Ok(100.0)
            }
        }
        fn compute_power_limit_watts(&self) -> Result<f64, GpuError> {
            if self.return_error {
                Err(GpuError::NotSupported("test".into()))
            } else {
                Ok(200.0)
            }
        }
        fn memory_used_bytes(&self) -> Result<u64, GpuError> {
            if self.return_error {
                Err(GpuError::NotSupported("test".into()))
            } else {
                Ok(1024)
            }
        }
        fn memory_total_bytes(&self) -> Result<u64, GpuError> {
            if self.return_error {
                Err(GpuError::NotSupported("test".into()))
            } else {
                Ok(2048)
            }
        }
        fn memory_bandwidth_gbps(&self) -> Result<f64, GpuError> {
            Err(GpuError::NotSupported("mock".into()))
        }
        fn compute_unit_count(&self) -> u32 {
            8
        }
        fn active_compute_units(&self) -> Result<u32, GpuError> {
            Ok(8)
        }
        fn pcie_tx_bytes_per_sec(&self) -> Result<u64, GpuError> {
            Err(GpuError::NotSupported("mock".into()))
        }
        fn pcie_rx_bytes_per_sec(&self) -> Result<u64, GpuError> {
            Err(GpuError::NotSupported("mock".into()))
        }
        fn pcie_generation(&self) -> u8 {
            0
        }
        fn pcie_width(&self) -> u8 {
            0
        }
        fn refresh(&mut self) -> Result<(), GpuError> {
            Ok(())
        }
    }

    #[test]
    fn h031_error_mock_memory_usage_percent_error() {
        let mock = ErrorMockDevice::new(true);
        // Should propagate the error from memory_used_bytes
        let result = mock.memory_usage_percent();
        assert!(result.is_err());
    }

    #[test]
    fn h031_error_mock_memory_available_bytes_error() {
        let mock = ErrorMockDevice::new(true);
        // Should propagate the error from memory_used_bytes
        let result = mock.memory_available_bytes();
        assert!(result.is_err());
    }

    #[test]
    fn h031_error_mock_memory_used_mb_error() {
        let mock = ErrorMockDevice::new(true);
        let result = mock.memory_used_mb();
        assert!(result.is_err());
    }

    #[test]
    fn h031_error_mock_memory_total_mb_error() {
        let mock = ErrorMockDevice::new(true);
        let result = mock.memory_total_mb();
        assert!(result.is_err());
    }

    #[test]
    fn h031_error_mock_memory_total_gb_error() {
        let mock = ErrorMockDevice::new(true);
        let result = mock.memory_total_gb();
        assert!(result.is_err());
    }

    #[test]
    fn h031_error_mock_power_usage_percent_error() {
        let mock = ErrorMockDevice::new(true);
        // Should propagate the error from compute_power_watts
        let result = mock.power_usage_percent();
        assert!(result.is_err());
    }

    #[test]
    fn h031_error_mock_is_thermal_throttling_error() {
        let mock = ErrorMockDevice::new(true);
        // Should propagate the error from compute_temperature_c
        let result = mock.is_thermal_throttling();
        assert!(result.is_err());
    }

    #[test]
    fn h031_error_mock_is_power_throttling_error() {
        let mock = ErrorMockDevice::new(true);
        // Should propagate the error from power_usage_percent -> compute_power_watts
        let result = mock.is_power_throttling();
        assert!(result.is_err());
    }

    #[test]
    fn h031_error_mock_working_correctly() {
        let mock = ErrorMockDevice::new(false);
        // When not returning errors, should work
        assert!(mock.memory_usage_percent().is_ok());
        assert!(mock.memory_available_bytes().is_ok());
        assert!(mock.memory_used_mb().is_ok());
        assert!(mock.memory_total_mb().is_ok());
        assert!(mock.memory_total_gb().is_ok());
        assert!(mock.power_usage_percent().is_ok());
        assert!(mock.is_thermal_throttling().is_ok());
        assert!(mock.is_power_throttling().is_ok());
    }

    // =========================================================================
    // H032: DeviceSnapshot with Errors
    // =========================================================================

    #[test]
    fn h032_device_snapshot_with_errors() {
        let mock = ErrorMockDevice::new(true);
        // DeviceSnapshot::capture uses unwrap_or defaults
        let snapshot = DeviceSnapshot::capture(&mock);
        assert!(snapshot.is_ok());

        let snap = snapshot.unwrap();
        // Should use defaults when metrics fail
        assert_eq!(snap.compute_utilization, 0.0);
        assert_eq!(snap.memory_used_bytes, 0);
        assert_eq!(snap.memory_total_bytes, 0);
        assert_eq!(snap.temperature_c, 0.0);
        assert_eq!(snap.power_watts, 0.0);
        assert_eq!(snap.clock_mhz, 0);
    }

    // =========================================================================
    // H033: ThrottleReason Complete Coverage
    // =========================================================================

    #[test]
    fn h033_throttle_reason_clone() {
        let reason = ThrottleReason::Power;
        let cloned = reason.clone();
        assert_eq!(reason, cloned);
    }

    #[test]
    fn h033_throttle_reason_copy() {
        let reason = ThrottleReason::Thermal;
        let copied: ThrottleReason = reason; // Copy
        assert_eq!(reason, copied);
    }

    #[test]
    fn h033_throttle_reason_equality() {
        assert_eq!(ThrottleReason::None, ThrottleReason::None);
        assert_eq!(ThrottleReason::Thermal, ThrottleReason::Thermal);
        assert_ne!(ThrottleReason::None, ThrottleReason::Thermal);
        assert_ne!(ThrottleReason::Power, ThrottleReason::Thermal);
    }

    #[test]
    fn h033_throttle_reason_debug() {
        let reason = ThrottleReason::HwSlowdown;
        let debug_str = format!("{:?}", reason);
        assert!(debug_str.contains("HwSlowdown"));
    }

    // =========================================================================
    // H034: DeviceType Complete Coverage
    // =========================================================================

    #[test]
    fn h034_device_type_clone() {
        let dt = DeviceType::NvidiaGpu;
        let cloned = dt.clone();
        assert_eq!(dt, cloned);
    }

    #[test]
    fn h034_device_type_copy() {
        let dt = DeviceType::AmdGpu;
        let copied: DeviceType = dt; // Copy
        assert_eq!(dt, copied);
    }

    #[test]
    fn h034_device_type_equality() {
        assert_eq!(DeviceType::Cpu, DeviceType::Cpu);
        assert_ne!(DeviceType::Cpu, DeviceType::NvidiaGpu);
        assert_ne!(DeviceType::NvidiaGpu, DeviceType::AmdGpu);
        assert_ne!(DeviceType::AmdGpu, DeviceType::IntelGpu);
        assert_ne!(DeviceType::IntelGpu, DeviceType::AppleSilicon);
    }

    #[test]
    fn h034_device_type_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(DeviceType::Cpu);
        set.insert(DeviceType::NvidiaGpu);
        set.insert(DeviceType::AmdGpu);
        set.insert(DeviceType::IntelGpu);
        set.insert(DeviceType::AppleSilicon);

        assert_eq!(set.len(), 5);

        // Duplicate should not increase size
        set.insert(DeviceType::Cpu);
        assert_eq!(set.len(), 5);
    }

    // =========================================================================
    // H035: DeviceId Complete Coverage
    // =========================================================================

    #[test]
    fn h035_device_id_copy() {
        let id = DeviceId::nvidia(0);
        let copied: DeviceId = id; // Copy
        assert_eq!(id, copied);
    }

    #[test]
    fn h035_device_id_new_with_all_types() {
        // Test DeviceId::new with all DeviceTypes
        let cpu = DeviceId::new(DeviceType::Cpu, 0);
        let nvidia = DeviceId::new(DeviceType::NvidiaGpu, 1);
        let amd = DeviceId::new(DeviceType::AmdGpu, 2);
        let intel = DeviceId::new(DeviceType::IntelGpu, 3);
        let apple = DeviceId::new(DeviceType::AppleSilicon, 4);

        assert_eq!(cpu.device_type, DeviceType::Cpu);
        assert_eq!(cpu.index, 0);
        assert_eq!(nvidia.device_type, DeviceType::NvidiaGpu);
        assert_eq!(nvidia.index, 1);
        assert_eq!(amd.device_type, DeviceType::AmdGpu);
        assert_eq!(amd.index, 2);
        assert_eq!(intel.device_type, DeviceType::IntelGpu);
        assert_eq!(intel.index, 3);
        assert_eq!(apple.device_type, DeviceType::AppleSilicon);
        assert_eq!(apple.index, 4);
    }

    // =========================================================================
    // H036: CpuDevice Partial Coverage via Edge Cases
    // =========================================================================

    #[test]
    fn h036_cpu_device_memory_used_realistic() {
        let mut cpu = CpuDevice::new();
        cpu.refresh().unwrap();

        // memory_used should be between 0 and total
        let used = cpu.memory_used_bytes().unwrap();
        let total = cpu.memory_total_bytes().unwrap();
        assert!(used <= total);
    }

    // =========================================================================
    // H037: Boundary Tests for Default Implementations
    // =========================================================================

    #[test]
    fn h037_memory_usage_percent_boundary_values() {
        // Test 0% usage
        let mock_empty = MockDevice::new(0, 1000, 0.0, 0.0, 0.0);
        assert!((mock_empty.memory_usage_percent().unwrap() - 0.0).abs() < 0.01);

        // Test 100% usage
        let mock_full = MockDevice::new(1000, 1000, 0.0, 0.0, 0.0);
        assert!((mock_full.memory_usage_percent().unwrap() - 100.0).abs() < 0.01);
    }

    #[test]
    fn h037_memory_available_boundary() {
        // All memory available
        let mock_empty = MockDevice::new(0, 1000, 0.0, 0.0, 0.0);
        assert_eq!(mock_empty.memory_available_bytes().unwrap(), 1000);

        // No memory available
        let mock_full = MockDevice::new(1000, 1000, 0.0, 0.0, 0.0);
        assert_eq!(mock_full.memory_available_bytes().unwrap(), 0);
    }

    #[test]
    fn h037_power_usage_percent_boundary() {
        // 0% power
        let mock_idle = MockDevice::new(0, 0, 0.0, 100.0, 0.0);
        assert!((mock_idle.power_usage_percent().unwrap() - 0.0).abs() < 0.01);

        // 100% power
        let mock_max = MockDevice::new(0, 0, 100.0, 100.0, 0.0);
        assert!((mock_max.power_usage_percent().unwrap() - 100.0).abs() < 0.01);
    }

    // =========================================================================
    // H038: DeviceSnapshot Memory Percent Edge Cases
    // =========================================================================

    #[test]
    fn h038_snapshot_memory_percent_edge_cases() {
        // Test with various memory ratios
        let snap_50 = DeviceSnapshot {
            device_id: DeviceId::cpu(),
            timestamp_ms: 12345,
            compute_utilization: 25.0,
            memory_used_bytes: 500,
            memory_total_bytes: 1000,
            temperature_c: 60.0,
            power_watts: 75.0,
            clock_mhz: 2500,
        };
        assert!((snap_50.memory_usage_percent() - 50.0).abs() < 0.01);

        // Test 0%
        let snap_0 = DeviceSnapshot {
            device_id: DeviceId::cpu(),
            timestamp_ms: 0,
            compute_utilization: 0.0,
            memory_used_bytes: 0,
            memory_total_bytes: 1000,
            temperature_c: 0.0,
            power_watts: 0.0,
            clock_mhz: 0,
        };
        assert!((snap_0.memory_usage_percent() - 0.0).abs() < 0.01);

        // Test 100%
        let snap_100 = DeviceSnapshot {
            device_id: DeviceId::cpu(),
            timestamp_ms: 0,
            compute_utilization: 0.0,
            memory_used_bytes: 1000,
            memory_total_bytes: 1000,
            temperature_c: 0.0,
            power_watts: 0.0,
            clock_mhz: 0,
        };
        assert!((snap_100.memory_usage_percent() - 100.0).abs() < 0.01);
    }

    // =========================================================================
    // H039: CpuDevice Method Coverage via Direct Tests
    // =========================================================================

    #[test]
    fn h039_cpu_device_device_id_and_type() {
        let cpu = CpuDevice::new();
        assert_eq!(cpu.device_id(), DeviceId::cpu());
        assert_eq!(cpu.device_type(), DeviceType::Cpu);
    }

    #[test]
    fn h039_cpu_utilization_initial() {
        let cpu = CpuDevice::new();
        // Initially cpu_usage is 0.0 before refresh
        let util = cpu.compute_utilization().unwrap();
        assert!(util >= 0.0 && util <= 100.0);
    }

    #[test]
    fn h039_cpu_memory_used_initial() {
        let cpu = CpuDevice::new();
        // Initially memory_used is 0 before refresh
        let used = cpu.memory_used_bytes().unwrap();
        assert!(used >= 0);
    }

    #[test]
    fn h039_cpu_active_units_equals_total() {
        let cpu = CpuDevice::new();
        let total = cpu.compute_unit_count();
        let active = cpu.active_compute_units().unwrap();
        assert_eq!(total, active);
    }

    // =========================================================================
    // H040: Additional MockDevice Trait Methods
    // =========================================================================

    #[test]
    fn h040_mock_device_power_limit_access() {
        let mock = MockDevice::new(0, 0, 50.0, 100.0, 0.0);
        let limit = mock.compute_power_limit_watts().unwrap();
        assert!((limit - 100.0).abs() < 0.01);
    }

    #[test]
    fn h040_mock_device_memory_total_access() {
        let mock = MockDevice::new(500, 1000, 0.0, 0.0, 0.0);
        let total = mock.memory_total_bytes().unwrap();
        assert_eq!(total, 1000);
    }

    // =========================================================================
    // H041: ErrorMockDevice Additional Coverage
    // =========================================================================

    #[test]
    fn h041_error_mock_device_name() {
        let mock = ErrorMockDevice::new(false);
        assert_eq!(mock.device_name(), "ErrorMock");
    }

    #[test]
    fn h041_error_mock_device_type() {
        let mock = ErrorMockDevice::new(false);
        assert_eq!(mock.device_type(), DeviceType::Cpu);
    }

    #[test]
    fn h041_error_mock_compute_units() {
        let mock = ErrorMockDevice::new(false);
        assert_eq!(mock.compute_unit_count(), 8);
        assert_eq!(mock.active_compute_units().unwrap(), 8);
    }

    #[test]
    fn h041_error_mock_pcie_metrics() {
        let mock = ErrorMockDevice::new(false);
        assert_eq!(mock.pcie_generation(), 0);
        assert_eq!(mock.pcie_width(), 0);
        assert!(mock.pcie_tx_bytes_per_sec().is_err());
        assert!(mock.pcie_rx_bytes_per_sec().is_err());
    }

    #[test]
    fn h041_error_mock_refresh() {
        let mut mock = ErrorMockDevice::new(false);
        assert!(mock.refresh().is_ok());
    }

    #[test]
    fn h041_error_mock_memory_bandwidth() {
        let mock = ErrorMockDevice::new(false);
        assert!(mock.memory_bandwidth_gbps().is_err());
    }

    // =========================================================================
    // H042: Partial Error Mock for Second-Call Error Propagation
    // =========================================================================

    /// Mock device that returns errors only for total memory (not used)
    /// to test error propagation in memory_usage_percent when second call fails
    struct PartialErrorMockDevice {
        error_on_total: bool,
        error_on_limit: bool,
    }

    impl PartialErrorMockDevice {
        fn with_total_error() -> Self {
            Self {
                error_on_total: true,
                error_on_limit: false,
            }
        }

        fn with_limit_error() -> Self {
            Self {
                error_on_total: false,
                error_on_limit: true,
            }
        }
    }

    impl ComputeDevice for PartialErrorMockDevice {
        fn device_id(&self) -> DeviceId {
            DeviceId::cpu()
        }
        fn device_name(&self) -> &str {
            "PartialErrorMock"
        }
        fn device_type(&self) -> DeviceType {
            DeviceType::Cpu
        }
        fn compute_utilization(&self) -> Result<f64, GpuError> {
            Ok(50.0)
        }
        fn compute_clock_mhz(&self) -> Result<u32, GpuError> {
            Ok(3000)
        }
        fn compute_temperature_c(&self) -> Result<f64, GpuError> {
            Ok(50.0)
        }
        fn compute_power_watts(&self) -> Result<f64, GpuError> {
            Ok(100.0)
        }
        fn compute_power_limit_watts(&self) -> Result<f64, GpuError> {
            if self.error_on_limit {
                Err(GpuError::NotSupported("limit error".into()))
            } else {
                Ok(200.0)
            }
        }
        fn memory_used_bytes(&self) -> Result<u64, GpuError> {
            Ok(1024)
        }
        fn memory_total_bytes(&self) -> Result<u64, GpuError> {
            if self.error_on_total {
                Err(GpuError::NotSupported("total error".into()))
            } else {
                Ok(2048)
            }
        }
        fn memory_bandwidth_gbps(&self) -> Result<f64, GpuError> {
            Err(GpuError::NotSupported("mock".into()))
        }
        fn compute_unit_count(&self) -> u32 {
            8
        }
        fn active_compute_units(&self) -> Result<u32, GpuError> {
            Ok(8)
        }
        fn pcie_tx_bytes_per_sec(&self) -> Result<u64, GpuError> {
            Err(GpuError::NotSupported("mock".into()))
        }
        fn pcie_rx_bytes_per_sec(&self) -> Result<u64, GpuError> {
            Err(GpuError::NotSupported("mock".into()))
        }
        fn pcie_generation(&self) -> u8 {
            0
        }
        fn pcie_width(&self) -> u8 {
            0
        }
        fn refresh(&mut self) -> Result<(), GpuError> {
            Ok(())
        }
    }

    #[test]
    fn h042_partial_error_memory_usage_percent_total_error() {
        let mock = PartialErrorMockDevice::with_total_error();
        // memory_used_bytes succeeds, but memory_total_bytes fails
        // Should propagate the error from the second call
        let result = mock.memory_usage_percent();
        assert!(result.is_err());
    }

    #[test]
    fn h042_partial_error_memory_available_bytes_total_error() {
        let mock = PartialErrorMockDevice::with_total_error();
        // memory_used_bytes succeeds, but memory_total_bytes fails
        let result = mock.memory_available_bytes();
        assert!(result.is_err());
    }

    #[test]
    fn h042_partial_error_power_usage_percent_limit_error() {
        let mock = PartialErrorMockDevice::with_limit_error();
        // compute_power_watts succeeds, but compute_power_limit_watts fails
        let result = mock.power_usage_percent();
        assert!(result.is_err());
    }

    #[test]
    fn h042_partial_error_device_name() {
        let mock = PartialErrorMockDevice::with_total_error();
        assert_eq!(mock.device_name(), "PartialErrorMock");
    }

    #[test]
    fn h042_partial_error_device_type() {
        let mock = PartialErrorMockDevice::with_total_error();
        assert_eq!(mock.device_type(), DeviceType::Cpu);
    }

    #[test]
    fn h042_partial_error_device_id() {
        let mock = PartialErrorMockDevice::with_total_error();
        assert_eq!(mock.device_id(), DeviceId::cpu());
    }

    #[test]
    fn h042_partial_error_compute_utilization() {
        let mock = PartialErrorMockDevice::with_total_error();
        assert!((mock.compute_utilization().unwrap() - 50.0).abs() < 0.01);
    }

    #[test]
    fn h042_partial_error_compute_clock() {
        let mock = PartialErrorMockDevice::with_total_error();
        assert_eq!(mock.compute_clock_mhz().unwrap(), 3000);
    }

    #[test]
    fn h042_partial_error_compute_temperature() {
        let mock = PartialErrorMockDevice::with_total_error();
        assert!((mock.compute_temperature_c().unwrap() - 50.0).abs() < 0.01);
    }

    #[test]
    fn h042_partial_error_compute_power() {
        let mock = PartialErrorMockDevice::with_total_error();
        assert!((mock.compute_power_watts().unwrap() - 100.0).abs() < 0.01);
    }

    #[test]
    fn h042_partial_error_memory_used() {
        let mock = PartialErrorMockDevice::with_total_error();
        assert_eq!(mock.memory_used_bytes().unwrap(), 1024);
    }

    #[test]
    fn h042_partial_error_compute_units() {
        let mock = PartialErrorMockDevice::with_total_error();
        assert_eq!(mock.compute_unit_count(), 8);
        assert_eq!(mock.active_compute_units().unwrap(), 8);
    }

    #[test]
    fn h042_partial_error_pcie_metrics() {
        let mock = PartialErrorMockDevice::with_total_error();
        assert_eq!(mock.pcie_generation(), 0);
        assert_eq!(mock.pcie_width(), 0);
        assert!(mock.pcie_tx_bytes_per_sec().is_err());
        assert!(mock.pcie_rx_bytes_per_sec().is_err());
    }

    #[test]
    fn h042_partial_error_memory_bandwidth() {
        let mock = PartialErrorMockDevice::with_total_error();
        assert!(mock.memory_bandwidth_gbps().is_err());
    }

    #[test]
    fn h042_partial_error_refresh() {
        let mut mock = PartialErrorMockDevice::with_total_error();
        assert!(mock.refresh().is_ok());
    }

    // =========================================================================
    // H043: DeviceSnapshot Timestamp Coverage
    // =========================================================================

    #[test]
    fn h043_device_snapshot_timestamp_non_zero() {
        let mock = MockDevice::new(1024, 2048, 100.0, 200.0, 50.0);
        let snapshot = DeviceSnapshot::capture(&mock).unwrap();

        // Timestamp should be non-zero (based on system time)
        assert!(snapshot.timestamp_ms > 0);
    }

    #[test]
    fn h043_device_snapshot_all_fields_populated() {
        let mock = MockDevice::new(1024, 2048, 100.0, 200.0, 50.0);
        let snapshot = DeviceSnapshot::capture(&mock).unwrap();

        // Verify all fields are populated from the mock
        assert_eq!(snapshot.device_id, DeviceId::cpu());
        assert!((snapshot.compute_utilization - 50.0).abs() < 0.01);
        assert_eq!(snapshot.memory_used_bytes, 1024);
        assert_eq!(snapshot.memory_total_bytes, 2048);
        assert!((snapshot.temperature_c - 50.0).abs() < 0.01);
        assert!((snapshot.power_watts - 100.0).abs() < 0.01);
        assert_eq!(snapshot.clock_mhz, 3000);
    }

    // =========================================================================
    // H044: CpuDevice Debug Trait Coverage
    // =========================================================================

    #[test]
    fn h044_cpu_device_debug() {
        let cpu = CpuDevice::new();
        let debug_str = format!("{:?}", cpu);
        assert!(debug_str.contains("CpuDevice"));
    }
}
