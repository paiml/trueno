//! Tests for GPU monitoring (EXTREME TDD - Tests First!)

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::time::Duration;

    use crate::monitor::*;

    // =========================================================================
    // H0-MON-01: GpuVendor identification
    // =========================================================================

    #[test]
    fn h0_mon_01_vendor_nvidia_id() {
        let vendor = GpuVendor::from_vendor_id(0x10de);
        assert_eq!(vendor, GpuVendor::Nvidia);
        assert!(vendor.is_nvidia());
        assert_eq!(vendor.name(), "NVIDIA");
    }

    #[test]
    fn h0_mon_02_vendor_amd_id() {
        let vendor = GpuVendor::from_vendor_id(0x1002);
        assert_eq!(vendor, GpuVendor::Amd);
        assert!(!vendor.is_nvidia());
        assert_eq!(vendor.name(), "AMD");
    }

    #[test]
    fn h0_mon_03_vendor_intel_id() {
        let vendor = GpuVendor::from_vendor_id(0x8086);
        assert_eq!(vendor, GpuVendor::Intel);
        assert!(!vendor.is_nvidia());
        assert_eq!(vendor.name(), "Intel");
    }

    #[test]
    fn h0_mon_04_vendor_apple_id() {
        let vendor = GpuVendor::from_vendor_id(0x106b);
        assert_eq!(vendor, GpuVendor::Apple);
        assert!(!vendor.is_nvidia());
        assert_eq!(vendor.name(), "Apple");
    }

    #[test]
    fn h0_mon_05_vendor_unknown_id() {
        let vendor = GpuVendor::from_vendor_id(0x9999);
        assert_eq!(vendor, GpuVendor::Unknown(0x9999));
        assert!(!vendor.is_nvidia());
        assert_eq!(vendor.name(), "Unknown");
    }

    #[test]
    fn h0_mon_06_vendor_display() {
        assert_eq!(format!("{}", GpuVendor::Nvidia), "NVIDIA");
        assert_eq!(format!("{}", GpuVendor::Amd), "AMD");
        assert_eq!(
            format!("{}", GpuVendor::Unknown(0x1234)),
            "Unknown (0x1234)"
        );
    }

    // =========================================================================
    // H0-MON-10: GpuBackend identification
    // =========================================================================

    #[test]
    fn h0_mon_10_backend_names() {
        assert_eq!(GpuBackend::Vulkan.name(), "Vulkan");
        assert_eq!(GpuBackend::Metal.name(), "Metal");
        assert_eq!(GpuBackend::Dx12.name(), "DirectX 12");
        assert_eq!(GpuBackend::Dx11.name(), "DirectX 11");
        assert_eq!(GpuBackend::WebGpu.name(), "WebGPU");
        assert_eq!(GpuBackend::Cuda.name(), "CUDA");
        assert_eq!(GpuBackend::OpenGl.name(), "OpenGL");
        assert_eq!(GpuBackend::Cpu.name(), "CPU");
    }

    #[test]
    fn h0_mon_11_backend_is_gpu() {
        assert!(GpuBackend::Vulkan.is_gpu());
        assert!(GpuBackend::Metal.is_gpu());
        assert!(GpuBackend::Dx12.is_gpu());
        assert!(GpuBackend::Dx11.is_gpu());
        assert!(GpuBackend::WebGpu.is_gpu());
        assert!(GpuBackend::Cuda.is_gpu());
        assert!(GpuBackend::OpenGl.is_gpu());
        assert!(!GpuBackend::Cpu.is_gpu());
    }

    #[test]
    fn h0_mon_12_backend_supports_compute() {
        assert!(GpuBackend::Vulkan.supports_compute());
        assert!(GpuBackend::Metal.supports_compute());
        assert!(GpuBackend::Dx12.supports_compute());
        assert!(GpuBackend::WebGpu.supports_compute());
        assert!(GpuBackend::Cuda.supports_compute());
        assert!(!GpuBackend::Dx11.supports_compute()); // DX11 compute shaders limited
        assert!(!GpuBackend::OpenGl.supports_compute());
        assert!(!GpuBackend::Cpu.supports_compute());
    }

    // =========================================================================
    // H0-MON-20: GpuDeviceInfo construction
    // =========================================================================

    #[test]
    fn h0_mon_20_device_info_basic() {
        let info = GpuDeviceInfo::new(0, "Test GPU", GpuVendor::Nvidia, GpuBackend::Vulkan);

        assert_eq!(info.index, 0);
        assert_eq!(info.name, "Test GPU");
        assert_eq!(info.vendor, GpuVendor::Nvidia);
        assert_eq!(info.backend, GpuBackend::Vulkan);
        assert_eq!(info.vram_total, 0);
        assert!(info.compute_capability.is_none());
    }

    #[test]
    fn h0_mon_21_device_info_builder() {
        let info = GpuDeviceInfo::new(0, "RTX 4090", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(24_000_000_000)
            .with_compute_capability(8, 9)
            .with_driver_version("535.154.05")
            .with_pci_bus_id("0000:01:00.0");

        assert_eq!(info.vram_total, 24_000_000_000);
        assert_eq!(info.compute_capability, Some((8, 9)));
        assert_eq!(info.driver_version, Some("535.154.05".to_string()));
        assert_eq!(info.pci_bus_id, Some("0000:01:00.0".to_string()));
    }

    #[test]
    fn h0_mon_22_device_info_vram_helpers() {
        let info = GpuDeviceInfo::new(0, "Test", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(24 * 1024 * 1024 * 1024); // 24 GB

        assert_eq!(info.vram_mb(), 24 * 1024);
        assert!((info.vram_gb() - 24.0).abs() < 0.01);
    }

    #[test]
    fn h0_mon_23_device_info_supports_cuda() {
        let nvidia = GpuDeviceInfo::new(0, "RTX", GpuVendor::Nvidia, GpuBackend::Vulkan);
        let amd = GpuDeviceInfo::new(0, "RX", GpuVendor::Amd, GpuBackend::Vulkan);

        assert!(nvidia.supports_cuda());
        assert!(!amd.supports_cuda());
    }

    #[test]
    fn h0_mon_24_device_info_display() {
        let info = GpuDeviceInfo::new(0, "RTX 4090", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(24 * 1024 * 1024 * 1024);

        let display = format!("{info}");
        assert!(display.contains("RTX 4090"));
        assert!(display.contains("Vulkan"));
        assert!(display.contains("24.0"));
    }

    // =========================================================================
    // H0-MON-30: GpuMemoryMetrics
    // =========================================================================

    #[test]
    fn h0_mon_30_memory_metrics_basic() {
        let mem = GpuMemoryMetrics::new(24_000_000_000, 8_000_000_000, 16_000_000_000);

        assert_eq!(mem.total, 24_000_000_000);
        assert_eq!(mem.used, 8_000_000_000);
        assert_eq!(mem.free, 16_000_000_000);
    }

    #[test]
    fn h0_mon_31_memory_metrics_usage_percent() {
        let mem = GpuMemoryMetrics::new(100, 25, 75);
        assert!((mem.usage_percent() - 25.0).abs() < 0.01);
    }

    #[test]
    fn h0_mon_32_memory_metrics_usage_percent_zero_total() {
        let mem = GpuMemoryMetrics::new(0, 0, 0);
        assert!((mem.usage_percent() - 0.0).abs() < 0.01);
    }

    #[test]
    fn h0_mon_33_memory_metrics_mb_helpers() {
        let mem = GpuMemoryMetrics::new(
            24 * 1024 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
            16 * 1024 * 1024 * 1024,
        );

        assert_eq!(mem.used_mb(), 8 * 1024);
        assert_eq!(mem.free_mb(), 16 * 1024);
    }

    // =========================================================================
    // H0-MON-40: GpuThermalMetrics
    // =========================================================================

    #[test]
    fn h0_mon_40_thermal_safe() {
        let thermal = GpuThermalMetrics {
            temperature_celsius: 50,
            ..Default::default()
        };
        assert!(thermal.is_safe());
        assert!(!thermal.is_critical());
        assert_eq!(thermal.status(), "COOL");
    }

    #[test]
    fn h0_mon_41_thermal_warm() {
        let thermal = GpuThermalMetrics {
            temperature_celsius: 65,
            ..Default::default()
        };
        assert!(thermal.is_safe());
        assert!(!thermal.is_critical());
        assert_eq!(thermal.status(), "WARM");
    }

    #[test]
    fn h0_mon_42_thermal_hot() {
        let thermal = GpuThermalMetrics {
            temperature_celsius: 82,
            ..Default::default()
        };
        assert!(!thermal.is_safe());
        assert!(!thermal.is_critical());
        assert_eq!(thermal.status(), "HOT");
    }

    #[test]
    fn h0_mon_43_thermal_critical() {
        let thermal = GpuThermalMetrics {
            temperature_celsius: 95,
            ..Default::default()
        };
        assert!(!thermal.is_safe());
        assert!(thermal.is_critical());
        assert_eq!(thermal.status(), "CRITICAL");
    }

    // =========================================================================
    // H0-MON-50: GpuPowerMetrics
    // =========================================================================

    #[test]
    fn h0_mon_50_power_usage_percent() {
        let power = GpuPowerMetrics {
            power_draw_watts: 225.0,
            power_limit_watts: 450.0,
            power_state: 0,
        };
        assert!((power.usage_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn h0_mon_51_power_usage_percent_zero_limit() {
        let power = GpuPowerMetrics {
            power_draw_watts: 100.0,
            power_limit_watts: 0.0,
            power_state: 0,
        };
        assert!((power.usage_percent() - 0.0).abs() < 0.01);
    }

    // =========================================================================
    // H0-MON-60: GpuMetrics
    // =========================================================================

    #[test]
    fn h0_mon_60_metrics_creation() {
        let mem = GpuMemoryMetrics::new(1000, 500, 500);
        let metrics = GpuMetrics::new(0, mem);

        assert_eq!(metrics.device_index, 0);
        assert_eq!(metrics.memory.total, 1000);
        assert!(metrics.thermal.is_none());
        assert!(metrics.power.is_none());
    }

    #[test]
    fn h0_mon_61_metrics_age() {
        let mem = GpuMemoryMetrics::new(1000, 500, 500);
        let metrics = GpuMetrics::new(0, mem);

        // Age should be very small immediately after creation
        assert!(metrics.age() < Duration::from_millis(100));
    }

    // =========================================================================
    // H0-MON-70: MonitorConfig
    // =========================================================================

    #[test]
    fn h0_mon_70_config_default() {
        let config = MonitorConfig::default();

        assert_eq!(config.poll_interval, Duration::from_millis(100));
        assert_eq!(config.history_size, 600);
        assert!(!config.background_collection);
    }

    #[test]
    fn h0_mon_71_config_high_frequency() {
        let config = MonitorConfig::high_frequency();

        assert_eq!(config.poll_interval, Duration::from_millis(50));
        assert_eq!(config.history_size, 1200);
        assert!(config.background_collection);
    }

    #[test]
    fn h0_mon_72_config_low_overhead() {
        let config = MonitorConfig::low_overhead();

        assert_eq!(config.poll_interval, Duration::from_millis(500));
        assert_eq!(config.history_size, 120);
        assert!(!config.background_collection);
    }

    // =========================================================================
    // H0-MON-80: MonitorError
    // =========================================================================

    #[test]
    fn h0_mon_80_error_display() {
        assert_eq!(
            format!("{}", MonitorError::NoDevice),
            "No GPU device available"
        );
        assert_eq!(
            format!("{}", MonitorError::InvalidDevice(5)),
            "Invalid device index: 5"
        );
        assert_eq!(
            format!("{}", MonitorError::BackendInit("test".to_string())),
            "Backend initialization failed: test"
        );
    }

    // =========================================================================
    // Integration tests (require GPU feature)
    // =========================================================================

    #[test]
    #[cfg(feature = "gpu")]
    fn h0_mon_90_query_device_info() {
        // This test requires actual GPU hardware
        match GpuDeviceInfo::query() {
            Ok(info) => {
                // Verify we got valid data
                assert!(!info.name.is_empty());
                assert!(info.backend.is_gpu());
                println!("Found GPU: {info}");
            }
            Err(MonitorError::NoDevice) => {
                // No GPU is OK for CI environments
                println!("No GPU available (expected in CI)");
            }
            Err(e) => {
                panic!("Unexpected error: {e}");
            }
        }
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn h0_mon_91_enumerate_devices() {
        match GpuDeviceInfo::enumerate() {
            Ok(devices) => {
                for dev in &devices {
                    println!("Found: {dev}");
                }
            }
            Err(MonitorError::NoDevice) => {
                println!("No GPU available (expected in CI)");
            }
            Err(e) => {
                panic!("Unexpected error: {e}");
            }
        }
    }

    // =========================================================================
    // H0-MON-100: GpuMonitor (mock)
    // =========================================================================

    #[test]
    fn h0_mon_100_monitor_mock_creation() {
        let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(24 * 1024 * 1024 * 1024);
        let config = MonitorConfig::default();
        let monitor = GpuMonitor::mock(info, config);

        assert_eq!(monitor.device_info().name, "Mock GPU");
        assert_eq!(monitor.config().poll_interval, Duration::from_millis(100));
    }

    #[test]
    fn h0_mon_101_monitor_collect() {
        let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(24 * 1024 * 1024 * 1024);
        let config = MonitorConfig::default();
        let monitor = GpuMonitor::mock(info, config);

        // Initially no samples
        assert_eq!(monitor.sample_count(), 0);

        // Collect a sample
        let metrics = monitor.collect().expect("collect should work");
        assert_eq!(metrics.device_index, 0);
        assert_eq!(monitor.sample_count(), 1);

        // Collect more samples
        monitor.collect().expect("collect should work");
        monitor.collect().expect("collect should work");
        assert_eq!(monitor.sample_count(), 3);
    }

    #[test]
    fn h0_mon_102_monitor_history_buffer() {
        let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(1024);

        // Small history size to test ring buffer
        let config = MonitorConfig {
            history_size: 3,
            ..Default::default()
        };
        let monitor = GpuMonitor::mock(info, config);

        // Fill beyond capacity
        for _ in 0..5 {
            monitor.collect().expect("collect should work");
        }

        // Should only have 3 samples (ring buffer)
        assert_eq!(monitor.sample_count(), 3);

        // History should return 3 items
        let history = monitor.history();
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn h0_mon_103_monitor_latest() {
        let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(1024);
        let config = MonitorConfig::default();
        let monitor = GpuMonitor::mock(info, config);

        // No samples yet - should error
        assert!(monitor.latest().is_err());

        // After collecting, latest should work
        monitor.collect().expect("collect should work");
        let latest = monitor.latest().expect("latest should work");
        assert_eq!(latest.device_index, 0);
    }

    #[test]
    fn h0_mon_104_monitor_clear_history() {
        let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(1024);
        let config = MonitorConfig::default();
        let monitor = GpuMonitor::mock(info, config);

        // Collect some samples
        monitor.collect().expect("collect should work");
        monitor.collect().expect("collect should work");
        assert_eq!(monitor.sample_count(), 2);

        // Clear history
        monitor.clear_history();
        assert_eq!(monitor.sample_count(), 0);
    }

    #[test]
    fn h0_mon_105_monitor_is_collecting() {
        let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan);
        let config = MonitorConfig::default();
        let monitor = GpuMonitor::mock(info, config);

        // Mock monitor is not actively collecting in background
        assert!(!monitor.is_collecting());
    }

    #[test]
    fn h0_mon_106_monitor_stop() {
        let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan);
        let config = MonitorConfig::default();
        let monitor = GpuMonitor::mock(info, config);

        // Stop should not panic even without background collection
        monitor.stop();
    }

    // =========================================================================
    // H0-MON-110: GpuMonitor integration tests (require GPU feature)
    // =========================================================================

    #[test]
    #[cfg(feature = "gpu")]
    fn h0_mon_110_monitor_real_gpu() {
        match GpuMonitor::new(0, MonitorConfig::default()) {
            Ok(monitor) => {
                println!("GPU Monitor: {}", monitor.device_info());

                // Collect a sample
                match monitor.collect() {
                    Ok(metrics) => {
                        println!("Collected metrics: device={}", metrics.device_index);
                        assert_eq!(monitor.sample_count(), 1);
                    }
                    Err(e) => {
                        println!("Collect failed (expected in CI): {e}");
                    }
                }
            }
            Err(MonitorError::NoDevice) => {
                println!("No GPU available (expected in CI)");
            }
            Err(e) => {
                panic!("Unexpected error: {e}");
            }
        }
    }

    // =========================================================================
    // H0-MON-120: CUDA monitoring integration tests
    // =========================================================================

    #[test]
    fn h0_mon_120_cuda_monitor_available_check() {
        // Should return false without cuda-monitor feature
        let available = cuda_monitor_available();
        #[cfg(feature = "cuda-monitor")]
        {
            // With feature, returns true/false based on hardware
            println!("CUDA monitoring available: {}", available);
        }
        #[cfg(not(feature = "cuda-monitor"))]
        {
            assert!(!available, "Should be false without cuda-monitor feature");
        }
    }

    #[test]
    #[cfg(feature = "cuda-monitor")]
    fn h0_mon_121_query_cuda_device_info() {
        use crate::monitor::backends::query_cuda_device_info;

        match query_cuda_device_info(0) {
            Ok(info) => {
                assert!(!info.name.is_empty());
                assert_eq!(info.vendor, GpuVendor::Nvidia);
                assert_eq!(info.backend, GpuBackend::Cuda);
                assert!(info.vram_total > 0);
                println!("CUDA Device: {}", info);
            }
            Err(e) => {
                println!("No CUDA device (expected in CI): {}", e);
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda-monitor")]
    fn h0_mon_122_enumerate_cuda_devices() {
        use crate::monitor::backends::enumerate_cuda_devices;

        match enumerate_cuda_devices() {
            Ok(devices) => {
                for dev in &devices {
                    assert_eq!(dev.vendor, GpuVendor::Nvidia);
                    assert_eq!(dev.backend, GpuBackend::Cuda);
                    println!("Found CUDA device: {}", dev);
                }
            }
            Err(e) => {
                println!("CUDA enumeration failed (expected in CI): {}", e);
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda-monitor")]
    fn h0_mon_123_query_cuda_memory() {
        use crate::monitor::backends::query_cuda_memory;

        match query_cuda_memory(0) {
            Ok(mem) => {
                assert!(mem.total > 0);
                assert!(mem.free <= mem.total);
                println!(
                    "CUDA Memory: {} / {} MB ({:.1}% used)",
                    mem.used_mb(),
                    mem.total / (1024 * 1024),
                    mem.usage_percent()
                );
            }
            Err(e) => {
                println!("CUDA memory query failed (expected in CI): {}", e);
            }
        }
    }
}
