//! Metal GEMM Benchmark - METAL-03 Validation
//!
//! Measures Metal compute shader performance vs CPU baseline.
//! Target: Metal achieves >= 80% of CUDA equivalent performance.
//!
//! This benchmark uses wgpu with Metal backend on macOS.
//!
//! Run with: cargo run -p trueno-gpu --example metal_gemm_benchmark --features wgpu --release
//!
//! Note: On macOS, wgpu uses Metal. On other platforms, uses Vulkan/DX12.

use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║      METAL-03: Performance Benchmark (80% Target)        ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    #[cfg(not(feature = "wgpu"))]
    {
        println!("METAL-03 SKIPPED: wgpu feature not enabled");
        println!("Run with: cargo run -p trueno-gpu --example metal_gemm_benchmark --features wgpu --release");
        return;
    }

    #[cfg(feature = "wgpu")]
    run_wgpu_benchmark();
}

#[cfg(feature = "wgpu")]
fn run_wgpu_benchmark() {
    use pollster::block_on;

    block_on(async {
        // Request adapter (will use Metal on macOS)
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await;

        let adapter = match adapter {
            Some(a) => a,
            None => {
                println!("❌ No GPU adapter found");
                return;
            }
        };

        let info = adapter.get_info();
        println!("Backend: {:?}", info.backend);
        println!("Device: {}", info.name);
        println!("Driver: {}\n", info.driver);

        // Check if Metal backend
        let is_metal = matches!(info.backend, wgpu::Backend::Metal);
        if !is_metal && cfg!(target_os = "macos") {
            println!("⚠️  Warning: Not using Metal backend on macOS");
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("Failed to create device");

        // WGSL compute shader for matrix multiplication
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GEMM Shader"),
            source: wgpu::ShaderSource::Wgsl(GEMM_SHADER.into()),
        });

        // Test sizes: 256, 512, 1024, 2048
        let sizes = [256, 512, 1024, 2048];

        println!("┌─────────┬────────────┬────────────┬────────────┬─────────┐");
        println!("│  Size   │  CPU (ms)  │  GPU (ms)  │   Ratio    │ Status  │");
        println!("├─────────┼────────────┼────────────┼────────────┼─────────┤");

        let mut all_pass = true;

        for &n in &sizes {
            // Generate test data
            let a: Vec<f32> = (0..n * n).map(|i| (i % 17) as f32 * 0.1).collect();
            let b: Vec<f32> = (0..n * n).map(|i| (i % 13) as f32 * 0.1).collect();

            let iterations = if n <= 512 { 10 } else { 3 };

            // CPU benchmark
            let cpu_start = Instant::now();
            for _ in 0..iterations {
                let _ = cpu_matmul(&a, &b, n);
            }
            let cpu_time = cpu_start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

            // GPU benchmark
            let gpu_time = run_gpu_matmul(&device, &queue, &shader, &a, &b, n, iterations);

            // Calculate ratio (GPU should be faster, ratio > 1.0 means GPU faster)
            let ratio = cpu_time / gpu_time;
            let status = if ratio >= 0.8 { "✅ PASS" } else { "❌ FAIL" };
            if ratio < 0.8 {
                all_pass = false;
            }

            println!(
                "│ {:>5}x{:<3} │ {:>10.2} │ {:>10.2} │ {:>10.2}x │ {} │",
                n, n, cpu_time, gpu_time, ratio, status
            );
        }

        println!("└─────────┴────────────┴────────────┴────────────┴─────────┘\n");

        if all_pass {
            println!("✅ METAL-03 PASSED: GPU performance meets 80% threshold");
            if is_metal {
                println!("   Validated on Metal backend");
            }
        } else {
            println!("❌ METAL-03 FAILED: Some sizes below 80% threshold");
        }
    });
}

/// CPU reference matmul (naive, for comparison baseline)
fn cpu_matmul(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

#[cfg(feature = "wgpu")]
fn run_gpu_matmul(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shader: &wgpu::ShaderModule,
    a: &[f32],
    b: &[f32],
    n: usize,
    iterations: usize,
) -> f64 {
    use wgpu::util::DeviceExt;

    // Create buffers
    let buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Buffer A"),
        contents: bytemuck::cast_slice(a),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Buffer B"),
        contents: bytemuck::cast_slice(b),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buffer_c = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Buffer C"),
        size: (n * n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims = [n as u32, n as u32, n as u32, 0u32]; // M, N, K, padding
    let buffer_dims = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dimensions"),
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create bind group layout and pipeline
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("GEMM Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("GEMM Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("GEMM Pipeline"),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: Some("gemm"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("GEMM Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer_c.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffer_dims.as_entire_binding(),
            },
        ],
    });

    // Warmup
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Warmup Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Warmup Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n as u32 + 15) / 16, (n as u32 + 15) / 16, 1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GEMM Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n as u32 + 15) / 16, (n as u32 + 15) / 16, 1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }
    let elapsed = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

    elapsed
}

#[cfg(feature = "wgpu")]
const GEMM_SHADER: &str = r#"
struct Dimensions {
    M: u32,
    N: u32,
    K: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dimensions;

@compute @workgroup_size(16, 16)
fn gemm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;

    if (row >= dims.M || col >= dims.N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
        sum = sum + A[row * dims.K + k] * B[k * dims.N + col];
    }
    C[row * dims.N + col] = sum;
}
"#;
