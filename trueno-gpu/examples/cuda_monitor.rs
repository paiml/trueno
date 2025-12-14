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
fn run_cuda_demo() {
    use trueno_gpu::monitor::{cuda_device_count, cuda_monitoring_available, CudaDeviceInfo, CudaMemoryInfo};
    use trueno_gpu::driver::CudaContext;

    // Phase 1: Check CUDA availability
    println!("Phase 1: CUDA Availability Check");
    println!("---------------------------------");

    if !cuda_monitoring_available() {
        println!("  [ERROR] CUDA not available.");
        println!("  Check that NVIDIA driver is installed and GPU is present.");
        return;
    }
    println!("  [OK] CUDA driver detected");

    // Phase 2: Device count
    println!("\nPhase 2: Device Count");
    println!("---------------------");

    match cuda_device_count() {
        Ok(count) => {
            println!("  Found {} CUDA device(s)", count);
        }
        Err(e) => {
            println!("  [ERROR] Failed to get device count: {}", e);
            return;
        }
    }

    // Phase 3: Enumerate devices
    println!("\nPhase 3: Device Enumeration");
    println!("---------------------------");

    match CudaDeviceInfo::enumerate() {
        Ok(devices) => {
            for dev in &devices {
                println!("  [{}] {}", dev.index, dev.name);
                println!("      Total Memory: {:.2} GB ({} bytes)", dev.total_memory_gb(), dev.total_memory);
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

    // Phase 4: Query individual device
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

    // Phase 5: Memory information (requires context)
    println!("\nPhase 5: Real-Time Memory Info");
    println!("------------------------------");

    match CudaContext::new(0) {
        Ok(ctx) => {
            match CudaMemoryInfo::query(&ctx) {
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
            }
        }
        Err(e) => {
            println!("  [ERROR] Context creation failed: {}", e);
        }
    }

    // Phase 6: Memory under load simulation
    println!("\nPhase 6: Memory Monitoring Over Time");
    println!("------------------------------------");

    match CudaContext::new(0) {
        Ok(ctx) => {
            println!("  Sampling memory 5 times...\n");
            for i in 0..5 {
                match CudaMemoryInfo::query(&ctx) {
                    Ok(mem) => {
                        println!("    Sample {}: {} MB free ({:.1}% used)",
                            i + 1, mem.free_mb(), mem.usage_percent());
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

    println!("\n================================================");
    println!("  Demo complete!");
    println!("================================================");
}
