//! Adversarial GEMM Correctness Test (Dr. Popper's Falsification Protocol)
//!
//! Tests boundary conditions with non-aligned dimensions:
//! - N=1023 (tile_size - 1)
//! - N=17 (tile_size + 1)
//! - N=1 (minimum)
//! - N=33 (2*tile_size + 1)
//!
//! Run with: `cargo run -p trueno-gpu --example gemm_adversarial_correctness --features cuda`

#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::kernels::{GemmKernel, Kernel};
use trueno_gpu::ptx::PtxModule;

/// CPU reference GEMM for validation
fn cpu_gemm(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                sum += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Check for NaN or Inf in results
fn check_nan_inf(data: &[f32], name: &str) -> bool {
    for (i, &val) in data.iter().enumerate() {
        if val.is_nan() {
            eprintln!("ERROR: NaN detected at index {} in {}", i, name);
            return false;
        }
        if val.is_infinite() {
            eprintln!("ERROR: Inf detected at index {} in {}", i, name);
            return false;
        }
    }
    true
}

/// Compare GPU results to CPU reference
fn compare_results(gpu: &[f32], cpu: &[f32], tolerance: f32, name: &str) -> bool {
    if gpu.len() != cpu.len() {
        eprintln!(
            "ERROR: Size mismatch in {}: GPU={}, CPU={}",
            name,
            gpu.len(),
            cpu.len()
        );
        return false;
    }

    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    let mut errors = 0;

    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let diff = (g - c).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
        if diff > tolerance {
            errors += 1;
            if errors <= 5 {
                eprintln!(
                    "  Mismatch at index {}: GPU={:.6}, CPU={:.6}, diff={:.6}",
                    i, g, c, diff
                );
            }
        }
    }

    if errors > 0 {
        eprintln!(
            "ERROR: {} mismatches in {} (max diff={:.6} at index {})",
            errors, name, max_diff, max_idx
        );
        return false;
    }

    println!(
        "  {} PASSED (max diff={:.6} at index {})",
        name, max_diff, max_idx
    );
    true
}

#[cfg(feature = "cuda")]
fn test_gemm_dimension(
    ctx: &CudaContext,
    m: u32,
    n: u32,
    k: u32,
    tile_size: u32,
) -> Result<bool, String> {
    println!(
        "\n=== Testing M={}, N={}, K={} (tile_size={}) ===",
        m, n, k, tile_size
    );

    // Create input matrices with known values
    let a_host: Vec<f32> = (0..(m * k))
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let b_host: Vec<f32> = (0..(k * n))
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();

    // Compute CPU reference
    let c_cpu = cpu_gemm(&a_host, &b_host, m as usize, n as usize, k as usize);

    // Check CPU reference for NaN/Inf
    if !check_nan_inf(&c_cpu, "CPU reference") {
        return Err("CPU reference produced NaN/Inf".to_string());
    }

    // Generate kernel
    let kernel = GemmKernel::tiled(m, n, k, tile_size);
    let ptx = PtxModule::new()
        .version(8, 0)
        .target("sm_89")
        .address_size(64)
        .add_kernel(kernel.build_ptx())
        .emit();

    // Load module
    let mut module =
        CudaModule::from_ptx(ctx, &ptx).map_err(|e| format!("PTX compile failed: {}", e))?;

    let stream = CudaStream::new(ctx).map_err(|e| format!("Stream failed: {}", e))?;

    // Allocate GPU buffers
    let mut a_gpu: GpuBuffer<f32> =
        GpuBuffer::new(ctx, (m * k) as usize).map_err(|e| format!("A alloc failed: {}", e))?;
    let mut b_gpu: GpuBuffer<f32> =
        GpuBuffer::new(ctx, (k * n) as usize).map_err(|e| format!("B alloc failed: {}", e))?;
    let mut c_gpu: GpuBuffer<f32> =
        GpuBuffer::new(ctx, (m * n) as usize).map_err(|e| format!("C alloc failed: {}", e))?;

    // Copy inputs to GPU
    a_gpu
        .copy_from_host(&a_host)
        .map_err(|e| format!("A copy failed: {}", e))?;
    b_gpu
        .copy_from_host(&b_host)
        .map_err(|e| format!("B copy failed: {}", e))?;

    // Initialize C to zero
    let c_zeros = vec![0.0f32; (m * n) as usize];
    c_gpu
        .copy_from_host(&c_zeros)
        .map_err(|e| format!("C init failed: {}", e))?;

    // Launch kernel
    let grid_x = (n + tile_size - 1) / tile_size;
    let grid_y = (m + tile_size - 1) / tile_size;
    let config = LaunchConfig {
        grid: (grid_x, grid_y, 1),
        block: (tile_size, tile_size, 1),
        shared_mem: tile_size * tile_size * 4 * 2,
    };

    println!(
        "  Launch config: grid=({}, {}, 1), block=({}, {}, 1)",
        grid_x, grid_y, tile_size, tile_size
    );

    let mut args: [*mut c_void; 6] = [
        a_gpu.as_kernel_arg(),
        b_gpu.as_kernel_arg(),
        c_gpu.as_kernel_arg(),
        &m as *const u32 as *mut c_void,
        &n as *const u32 as *mut c_void,
        &k as *const u32 as *mut c_void,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, "gemm_tiled", &config, &mut args)
            .map_err(|e| format!("Kernel launch failed: {}", e))?;
    }

    stream
        .synchronize()
        .map_err(|e| format!("Sync failed: {}", e))?;

    // Copy results back
    let mut c_host = vec![0.0f32; (m * n) as usize];
    c_gpu
        .copy_to_host(&mut c_host)
        .map_err(|e| format!("C copy failed: {}", e))?;

    // Validate results
    if !check_nan_inf(&c_host, "GPU result") {
        return Ok(false);
    }

    // Compare with tolerance (FMA rounding differences)
    let tolerance = 1e-4 * k as f32; // Scale tolerance with K
    let passed = compare_results(&c_host, &c_cpu, tolerance, "GPU vs CPU");

    Ok(passed)
}

#[cfg(feature = "cuda")]
fn main() {
    println!("=== GEMM Adversarial Correctness Test (Dr. Popper's Falsification Protocol) ===");
    println!("Testing boundary conditions with non-aligned dimensions...\n");

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to create CUDA context: {}", e);
            std::process::exit(1);
        }
    };

    let tile_size = 16u32;
    let mut all_passed = true;
    let mut tests_run = 0;
    let mut tests_passed = 0;

    // Adversarial test cases (Dr. Popper's H0 falsification)
    let test_cases: Vec<(u32, u32, u32, &str)> = vec![
        // (M, N, K, description)
        (1, 1, 1, "Minimum dimension (N=1)"),
        (
            tile_size + 1,
            tile_size + 1,
            tile_size + 1,
            "Tile+1 (17x17x17)",
        ),
        (
            tile_size - 1,
            tile_size - 1,
            tile_size - 1,
            "Tile-1 (15x15x15)",
        ),
        (31, 31, 31, "2*Tile-1 (31x31x31)"),
        (33, 33, 33, "2*Tile+1 (33x33x33)"),
        (64, 63, 65, "Mixed non-aligned"),
        (1, 64, 64, "Single row"),
        (64, 1, 64, "Single column"),
        (64, 64, 1, "Single K"),
        (1023, 1023, 1023, "Large non-aligned (1023x1023x1023)"),
    ];

    for (m, n, k, desc) in test_cases {
        tests_run += 1;
        println!("\n--- Test {}: {} ---", tests_run, desc);

        match test_gemm_dimension(&ctx, m, n, k, tile_size) {
            Ok(true) => {
                tests_passed += 1;
                println!("  RESULT: PASSED");
            }
            Ok(false) => {
                all_passed = false;
                println!("  RESULT: FAILED (incorrect values or NaN/Inf)");
            }
            Err(e) => {
                all_passed = false;
                println!("  RESULT: ERROR - {}", e);
            }
        }
    }

    println!("\n=== SUMMARY ===");
    println!("Tests run: {}", tests_run);
    println!("Tests passed: {}", tests_passed);
    println!("Tests failed: {}", tests_run - tests_passed);

    if all_passed {
        println!("\n*** H0 NOT FALSIFIED: Boundary conditions appear correct ***");
    } else {
        println!("\n*** H0 FALSIFIED: Boundary condition bugs detected! ***");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled. Run with --features cuda");

    // Still validate PTX generation for adversarial dimensions
    println!("\n=== PTX Generation Test (no GPU) ===");

    let test_cases: Vec<(u32, u32, u32, &str)> = vec![
        (1, 1, 1, "Minimum"),
        (17, 17, 17, "Tile+1"),
        (15, 15, 15, "Tile-1"),
        (1023, 1023, 1023, "Large non-aligned"),
    ];

    for (m, n, k, desc) in test_cases {
        let kernel = GemmKernel::tiled(m, n, k, 16);
        let ptx = kernel.emit_ptx();

        if ptx.contains(".entry gemm_tiled") {
            println!(
                "  {} ({}x{}x{}): PTX generated OK ({} bytes)",
                desc,
                m,
                n,
                k,
                ptx.len()
            );
        } else {
            println!("  {} ({}x{}x{}): PTX GENERATION FAILED", desc, m, n, k);
        }
    }
}
