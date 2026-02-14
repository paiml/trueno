//! Error propagation tests: ErrorMockDevice, PartialErrorMockDevice,
//! ThrottleReason/DeviceType trait coverage, and boundary tests (H031-H044)

use super::*;

// =========================================================================
// MockDevice (shared test fixture for snapshot/field tests)
// =========================================================================

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
