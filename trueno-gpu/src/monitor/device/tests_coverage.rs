//! Extended coverage tests: CpuDevice coverage, MockDevice extended,
//! derived metrics, and boundary tests (H013-H030)

use super::*;

// =========================================================================
// MockDevice (shared test fixture)
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
