//! CUDA Device Monitoring Example (TRUENO-SPEC-010)
//!
//! Demonstrates native CUDA device discovery and memory monitoring.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p trueno-gpu --example cuda_monitor --features cuda
//! ```
//!
//! # Requirements
//!
//! - NVIDIA GPU with CUDA driver installed
//! - Build with `--features cuda`
//!
//! # References
//!
//! - TRUENO-SPEC-010: GPU Monitoring, Tracing, and Visualization
//! - CUDA Driver API: cuDeviceGetName, cuDeviceTotalMem, cuMemGetInfo

fn main() {
    println!("================================================");
    println!("  trueno-gpu CUDA Monitoring (TRUENO-SPEC-010)");
    println!("================================================\n");

    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA feature not enabled.");
        println!("Run with: cargo run -p trueno-gpu --example cuda_monitor --features cuda");
        return;
    }

    #[cfg(feature = "cuda")]
    run_cuda_demo();
}

#[cfg(feature = "cuda")]
fn phase_check_availability() -> bool {
    use trueno_gpu::monitor::cuda_monitoring_available;

    println!("Phase 1: CUDA Availability Check");
    println!("---------------------------------");

    if !cuda_monitoring_available() {
        println!("  [ERROR] CUDA not available.");
        println!("  Check that NVIDIA driver is installed and GPU is present.");
        return false;
    }
    println!("  [OK] CUDA driver detected");
    true
}

#[cfg(feature = "cuda")]
fn phase_device_count() -> bool {
    use trueno_gpu::monitor::cuda_device_count;

    println!("\nPhase 2: Device Count");
    println!("---------------------");

    match cuda_device_count() {
        Ok(count) => {
            println!("  Found {} CUDA device(s)", count);
            true
        }
        Err(e) => {
            println!("  [ERROR] Failed to get device count: {}", e);
            false
        }
    }
}

#[cfg(feature = "cuda")]
fn phase_enumerate_devices() {
    use trueno_gpu::monitor::CudaDeviceInfo;

    println!("\nPhase 3: Device Enumeration");
    println!("---------------------------");

    match CudaDeviceInfo::enumerate() {
        Ok(devices) => {
            for dev in &devices {
                println!("  [{}] {}", dev.index, dev.name);
                println!(
                    "      Total Memory: {:.2} GB ({} bytes)",
                    dev.total_memory_gb(),
                    dev.total_memory
                );
                println!("      Memory (MB):  {} MB", dev.total_memory_mb());
            }
            if devices.is_empty() {
                println!("  No CUDA devices found");
            }
        }
        Err(e) => {
            println!("  [ERROR] Device enumeration failed: {}", e);
        }
    }
}

#[cfg(feature = "cuda")]
fn phase_query_device() {
    use trueno_gpu::monitor::CudaDeviceInfo;

    println!("\nPhase 4: Query Device 0");
    println!("-----------------------");

    match CudaDeviceInfo::query(0) {
        Ok(info) => {
            println!("  Device: {}", info);
            println!("  Name:   {}", info.name);
            println!("  Index:  {}", info.index);
            println!("  VRAM:   {} MB", info.total_memory_mb());
        }
        Err(e) => {
            println!("  [ERROR] Query failed: {}", e);
        }
    }
}

#[cfg(feature = "cuda")]
fn phase_memory_info() {
    use trueno_gpu::driver::CudaContext;
    use trueno_gpu::monitor::CudaMemoryInfo;

    println!("\nPhase 5: Real-Time Memory Info");
    println!("------------------------------");

    match CudaContext::new(0) {
        Ok(ctx) => match CudaMemoryInfo::query(&ctx) {
            Ok(mem) => {
                println!("  Total:  {} MB", mem.total_mb());
                println!("  Free:   {} MB", mem.free_mb());
                println!("  Used:   {} MB", mem.used_mb());
                println!("  Usage:  {:.1}%", mem.usage_percent());
                println!("\n  Display format: {}", mem);
            }
            Err(e) => {
                println!("  [ERROR] Memory query failed: {}", e);
            }
        },
        Err(e) => {
            println!("  [ERROR] Context creation failed: {}", e);
        }
    }
}

#[cfg(feature = "cuda")]
fn phase_memory_monitoring() {
    use trueno_gpu::driver::CudaContext;
    use trueno_gpu::monitor::CudaMemoryInfo;

    println!("\nPhase 6: Memory Monitoring Over Time");
    println!("------------------------------------");

    match CudaContext::new(0) {
        Ok(ctx) => {
            println!("  Sampling memory 5 times...\n");
            for i in 0..5 {
                match CudaMemoryInfo::query(&ctx) {
                    Ok(mem) => {
                        println!(
                            "    Sample {}: {} MB free ({:.1}% used)",
                            i + 1,
                            mem.free_mb(),
                            mem.usage_percent()
                        );
                    }
                    Err(e) => {
                        println!("    Sample {}: Error - {}", i + 1, e);
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        }
        Err(e) => {
            println!("  [ERROR] Context creation failed: {}", e);
        }
    }
}

#[cfg(feature = "cuda")]
fn run_cuda_demo() {
    if !phase_check_availability() {
        return;
    }

    if !phase_device_count() {
        return;
    }

    phase_enumerate_devices();
    phase_query_device();
    phase_memory_info();
    phase_memory_monitoring();

    println!("\n================================================");
    println!("  Demo complete!");
    println!("================================================");
}
