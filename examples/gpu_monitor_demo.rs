//! GPU Monitoring Demo (TRUENO-SPEC-010)
//!
//! Demonstrates GPU device discovery and real-time monitoring capabilities.
//!
//! # Usage
//!
//! ```bash
//! # With wgpu backend (cross-platform)
//! cargo run --example gpu_monitor_demo --features gpu
//!
//! # With native CUDA backend (NVIDIA only, most accurate)
//! cargo run --example gpu_monitor_demo --features "gpu,cuda-monitor"
//! ```
//!
//! # Features
//!
//! - Device enumeration (wgpu + optional CUDA)
//! - Real-time memory metrics
//! - Vendor identification (NVIDIA, AMD, Intel, Apple)
//! - Backend detection (Vulkan, Metal, DX12, CUDA)
//!
//! # References
//!
//! - TRUENO-SPEC-010: GPU Monitoring, Tracing, and Visualization
//! - Nickolls et al. (2008): GPU parallel computing model

use trueno::monitor::{GpuBackend, GpuDeviceInfo, GpuMemoryMetrics, GpuVendor};

fn main() {
    println!("=================================================");
    println!("  trueno GPU Monitoring Demo (TRUENO-SPEC-010)");
    println!("=================================================\n");

    // Phase 1: Check available backends
    println!("Phase 1: Backend Detection");
    println!("--------------------------");

    #[cfg(feature = "gpu")]
    {
        println!("  [OK] wgpu backend enabled (cross-platform)");
    }
    #[cfg(not(feature = "gpu"))]
    {
        println!("  [--] wgpu backend not enabled");
        println!("       Run with: --features gpu");
    }

    #[cfg(feature = "cuda-monitor")]
    {
        if cuda_monitor_available() {
            println!("  [OK] CUDA monitoring available (native NVIDIA)");
        } else {
            println!("  [--] CUDA feature enabled but no NVIDIA GPU found");
        }
    }
    #[cfg(not(feature = "cuda-monitor"))]
    {
        println!("  [--] CUDA monitoring not enabled");
        println!("       Run with: --features cuda-monitor");
    }
    println!();

    // Phase 2: Enumerate devices
    println!("Phase 2: Device Enumeration");
    println!("---------------------------");

    // Try CUDA first (most accurate)
    #[cfg(feature = "cuda-monitor")]
    {
        println!("\n  CUDA Devices (native driver API):");
        match trueno::enumerate_cuda_devices() {
            Ok(devices) => {
                for dev in &devices {
                    print_device_info(dev, "    ");
                }
                if devices.is_empty() {
                    println!("    No CUDA devices found");
                }
            }
            Err(e) => {
                println!("    CUDA enumeration failed: {}", e);
            }
        }
    }

    // Then try wgpu (cross-platform)
    #[cfg(feature = "gpu")]
    {
        println!("\n  wgpu Devices (cross-platform):");
        match GpuDeviceInfo::enumerate() {
            Ok(devices) => {
                for dev in &devices {
                    print_device_info(dev, "    ");
                }
                if devices.is_empty() {
                    println!("    No wgpu devices found");
                }
            }
            Err(e) => {
                println!("    wgpu enumeration failed: {}", e);
            }
        }
    }
    println!();

    // Phase 3: Real-time memory monitoring
    println!("Phase 3: Real-Time Memory Monitoring");
    println!("------------------------------------");

    #[cfg(feature = "cuda-monitor")]
    {
        println!("\n  CUDA Memory (cuMemGetInfo):");
        match trueno::query_cuda_memory(0) {
            Ok(mem) => {
                print_memory_metrics(&mem, "    ");
            }
            Err(e) => {
                println!("    CUDA memory query failed: {}", e);
            }
        }
    }
    println!();

    // Phase 4: GpuMonitor with history
    println!("Phase 4: GpuMonitor with History Buffer");
    println!("---------------------------------------");

    #[cfg(feature = "gpu")]
    {
        match GpuMonitor::new(0, MonitorConfig::default()) {
            Ok(monitor) => {
                println!("  Monitor created for: {}", monitor.device_info().name);
                println!(
                    "  Config: poll_interval={:?}, history_size={}",
                    monitor.config().poll_interval,
                    monitor.config().history_size
                );

                // Collect a few samples
                println!("\n  Collecting 5 samples...");
                for i in 0..5 {
                    match monitor.collect() {
                        Ok(metrics) => {
                            println!(
                                "    Sample {}: memory={} bytes, age={:?}",
                                i + 1,
                                metrics.memory.total,
                                metrics.age()
                            );
                        }
                        Err(e) => {
                            println!("    Sample {} failed: {}", i + 1, e);
                        }
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }

                println!("\n  History buffer: {} samples", monitor.sample_count());

                // Get latest
                if let Ok(latest) = monitor.latest() {
                    println!("  Latest sample age: {:?}", latest.age());
                }
            }
            Err(MonitorError::NoDevice) => {
                println!("  No GPU device available");
            }
            Err(e) => {
                println!("  Monitor creation failed: {}", e);
            }
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        println!("  GpuMonitor requires --features gpu");
    }
    println!();

    // Phase 5: Vendor identification
    println!("Phase 5: Vendor Identification");
    println!("------------------------------");
    demonstrate_vendor_identification();
    println!();

    // Phase 6: Backend capabilities
    println!("Phase 6: Backend Capabilities");
    println!("-----------------------------");
    demonstrate_backend_capabilities();
    println!();

    println!("=================================================");
    println!("  Demo complete!");
    println!("=================================================");
}

#[allow(dead_code)]
fn print_device_info(dev: &GpuDeviceInfo, indent: &str) {
    println!("{}[{}] {} ({})", indent, dev.index, dev.name, dev.backend);
    println!("{}    Vendor: {}", indent, dev.vendor);
    println!(
        "{}    VRAM: {:.2} GB ({} bytes)",
        indent,
        dev.vram_gb(),
        dev.vram_total
    );
    if let Some((major, minor)) = dev.compute_capability {
        println!("{}    Compute Capability: {}.{}", indent, major, minor);
    }
    if let Some(ref driver) = dev.driver_version {
        println!("{}    Driver: {}", indent, driver);
    }
    if dev.supports_cuda() {
        println!("{}    CUDA: Supported", indent);
    }
}

#[allow(dead_code)]
fn print_memory_metrics(mem: &GpuMemoryMetrics, indent: &str) {
    println!("{}Total: {} MB", indent, mem.total / (1024 * 1024));
    println!("{}Used:  {} MB", indent, mem.used / (1024 * 1024));
    println!("{}Free:  {} MB", indent, mem.free / (1024 * 1024));
    println!("{}Usage: {:.1}%", indent, mem.usage_percent());
}

fn demonstrate_vendor_identification() {
    let vendors = [
        (0x10de, "NVIDIA"),
        (0x1002, "AMD"),
        (0x8086, "Intel"),
        (0x106b, "Apple"),
        (0x9999, "Unknown"),
    ];

    println!("  PCI Vendor ID Mapping:");
    for (id, _expected) in vendors {
        let vendor = GpuVendor::from_vendor_id(id);
        println!(
            "    0x{:04x} -> {} (is_nvidia={})",
            id,
            vendor,
            vendor.is_nvidia()
        );
    }
}

fn demonstrate_backend_capabilities() {
    let backends = [
        GpuBackend::Vulkan,
        GpuBackend::Metal,
        GpuBackend::Dx12,
        GpuBackend::Cuda,
        GpuBackend::WebGpu,
        GpuBackend::OpenGl,
        GpuBackend::Cpu,
    ];

    println!("  Backend Capabilities:");
    println!("  {:<12} | {:>6} | {:>7}", "Backend", "is_gpu", "compute");
    println!("  {:-<12}-+-{:-^6}-+-{:-^7}", "", "", "");
    for backend in backends {
        println!(
            "  {:<12} | {:>6} | {:>7}",
            backend.name(),
            backend.is_gpu(),
            backend.supports_compute()
        );
    }
}
