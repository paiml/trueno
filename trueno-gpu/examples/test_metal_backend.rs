//! Test Metal backend detection
//!
//! Run with: cargo run --example test_metal_backend --features metal

fn main() {
    use trueno_gpu::backend::{Backend, MetalBackend};

    let backend = MetalBackend;
    println!("Metal backend name: {}", backend.name());
    println!("Metal available: {}", backend.is_available());
    println!("Metal device count: {}", backend.device_count());

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use trueno_gpu::backend::{MetalCompute, MetalDevice};

        // List all Metal devices
        let devices = MetalCompute::devices();
        for (i, device) in devices.iter().enumerate() {
            println!("\nDevice {}:", i);
            println!("  Name: {}", device.name);
            println!("  VRAM: {:.1} GB", device.vram_gb());
            println!("  Unified Memory: {}", device.has_unified_memory);
            println!("  Max Threads: {}", device.max_threads_per_threadgroup);
        }

        // Test creating a compute pipeline
        if let Ok(compute) = MetalCompute::default_device() {
            println!("\nMetal compute pipeline created on: {}", compute.device_name());

            // Test shader compilation
            let shader = compute.compile_shader(
                r#"
                kernel void add(device float* a [[buffer(0)]],
                               device float* b [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
                    a[id] = a[id] + b[id];
                }
                "#,
                "add",
            );
            match shader {
                Ok(s) => println!("Shader compiled: {}", s.name()),
                Err(e) => println!("Shader compilation error: {}", e),
            }
        }
    }

    println!("\nMetal backend detection test complete!");
}
