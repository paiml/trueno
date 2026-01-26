//! Minimal Q4K GEMV CUDA test
//!
//! Run with: cargo run --release --features cuda --example test_q4k_cuda

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::kernels::{Kernel, Q4KGemvKernel};
use trueno_gpu::ptx::PtxModule;

fn main() {
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║      trueno-gpu: Q4KGemvKernel CUDA Test             ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // Initialize CUDA
    println!("[1/7] Initializing CUDA...");
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to create CUDA context: {}", e);
            return;
        }
    };
    let device_name = ctx.device_name().unwrap_or_else(|_| "Unknown".to_string());
    println!("       ✓ GPU: {}", device_name);

    // Create kernel for small dimensions (N=256, K=256)
    let n: u32 = 256;
    let k: u32 = 256;
    println!("[2/7] Generating Q4KGemvKernel PTX (N={}, K={})...", n, k);
    let kernel = Q4KGemvKernel::new(n, k);

    // Generate PTX with proper module wrapper
    let ptx = PtxModule::new()
        .version(8, 0)
        .target("sm_89")
        .address_size(64)
        .add_kernel(kernel.build_ptx())
        .emit();
    println!(
        "       PTX generated ({} bytes, {} lines)",
        ptx.len(),
        ptx.lines().count()
    );

    // Load module
    println!("[3/7] JIT compiling PTX...");
    let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load PTX module: {}", e);
            eprintln!("\nPTX dump (first 80 lines):");
            for (i, line) in ptx.lines().take(80).enumerate() {
                eprintln!("{:4}: {}", i + 1, line);
            }
            return;
        }
    };
    println!("       ✓ Module compiled");

    // Create stream
    println!("[4/7] Creating stream...");
    let stream = match CudaStream::new(&ctx) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create stream: {}", e);
            return;
        }
    };
    println!("       ✓ Stream created");

    // Allocate GPU buffers
    // Q4_K format: 144 bytes per super-block (256 elements)
    // Layout: d(2) + dmin(2) + scales(12) + qs(128) = 144 bytes
    let n_super_blocks = (k as usize + 255) / 256;
    let weights_size = n as usize * n_super_blocks * 144;
    let input_size = k as usize;
    let output_size = n as usize;

    println!("[5/7] Allocating buffers...");
    println!(
        "       Weights: {} bytes ({} super-blocks)",
        weights_size, n_super_blocks
    );
    println!("       Input: {} floats", input_size);
    println!("       Output: {} floats", output_size);

    // Create test data - all zeros for weights (will produce zero output)
    let weights_data = vec![0u8; weights_size];
    let input_data = vec![1.0f32; input_size];

    // Allocate and upload
    let weights_buf: GpuBuffer<u8> = match GpuBuffer::from_host(&ctx, &weights_data) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to allocate weights buffer: {}", e);
            return;
        }
    };
    let input_buf: GpuBuffer<f32> = match GpuBuffer::from_host(&ctx, &input_data) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to allocate input buffer: {}", e);
            return;
        }
    };
    let output_buf: GpuBuffer<f32> = match GpuBuffer::new(&ctx, output_size) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to allocate output buffer: {}", e);
            return;
        }
    };
    println!("       ✓ Buffers allocated");

    // Build kernel args
    println!("[6/7] Launching kernel...");
    let mut output_ptr = output_buf.as_ptr();
    let mut weights_ptr = weights_buf.as_ptr();
    let mut input_ptr = input_buf.as_ptr();
    let mut n_val = n;
    let mut k_val = k;

    let mut args: [*mut c_void; 5] = [
        &mut output_ptr as *mut _ as *mut c_void,
        &mut weights_ptr as *mut _ as *mut c_void,
        &mut input_ptr as *mut _ as *mut c_void,
        &mut n_val as *mut _ as *mut c_void,
        &mut k_val as *mut _ as *mut c_void,
    ];

    // Q4KGemvKernel: one warp (32 threads) per output row
    // Grid: ceil(N / 32), Block: 32
    let config = LaunchConfig {
        grid: ((n + 31) / 32, 1, 1),
        block: (32, 1, 1),
        shared_mem: 0,
    };
    println!("       Grid: ({}, 1, 1), Block: (32, 1, 1)", (n + 31) / 32);

    let start = std::time::Instant::now();
    unsafe {
        match stream.launch_kernel(&mut module, kernel.name(), &config, &mut args) {
            Ok(()) => println!("       ✓ Kernel launched"),
            Err(e) => {
                eprintln!("Failed to launch kernel: {}", e);
                return;
            }
        }
    }

    // Synchronize
    match stream.synchronize() {
        Ok(()) => {
            let elapsed = start.elapsed();
            println!("       ✓ Kernel executed in {:?}", elapsed);
        }
        Err(e) => {
            eprintln!("❌ Stream sync failed: {}", e);
            return;
        }
    }

    // Read back results
    println!("[7/7] Verifying...");
    let mut output = vec![0.0f32; output_size];
    match output_buf.copy_to_host(&mut output) {
        Ok(()) => println!("       ✓ Results copied back"),
        Err(e) => {
            eprintln!("Failed to copy results: {}", e);
            return;
        }
    }

    println!("\n✓ SUCCESS! Q4KGemvKernel executed without errors.");
    println!("Output[0..5]: {:?}", &output[..5.min(output.len())]);
}
