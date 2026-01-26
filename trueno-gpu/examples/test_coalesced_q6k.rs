//! Minimal CoalescedQ6K GEMV CUDA test
//!
//! Run with: cargo run --release --features cuda -p trueno-gpu --example test_coalesced_q6k

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::kernels::{CoalescedQ6KGemvKernel, Kernel};
use trueno_gpu::ptx::PtxModule;

fn main() {
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║   trueno-gpu: CoalescedQ6KGemvKernel CUDA Test       ║");
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

    // Create kernel for small dimensions (N=4, K=256 = 1 super-block)
    let n: u32 = 4;
    let k: u32 = 256;
    println!(
        "[2/7] Generating CoalescedQ6KGemvKernel PTX (N={}, K={})...",
        n, k
    );
    let kernel = CoalescedQ6KGemvKernel::new(k, n);

    // Generate PTX with proper module wrapper
    let ptx = PtxModule::new()
        .version(8, 0)
        .target("sm_86")
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
            eprintln!("\nPTX dump (first 100 lines):");
            for (i, line) in ptx.lines().take(100).enumerate() {
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

    // Q6_K format: 210 bytes per super-block (256 elements)
    // Layout: ql[128] + qh[64] + scales[16] + d[2] = 210 bytes
    let n_super_blocks = (k as usize + 255) / 256;
    let weights_size = n as usize * n_super_blocks * 210;
    let input_size = k as usize;
    let output_size = n as usize;

    println!("[5/7] Allocating buffers...");
    println!(
        "       Weights: {} bytes ({} super-blocks per row)",
        weights_size, n_super_blocks
    );
    println!("       Input: {} floats", input_size);
    println!("       Output: {} floats", output_size);

    // Create test data with non-zero values
    // Set d to 1.0f16, scales to small values, and quants to mid-range
    let mut weights_data = vec![0u8; weights_size];
    for row in 0..n as usize {
        let row_offset = row * 210;
        // Set d (f16) at offset 208 to 1.0
        // 1.0f16 = 0x3C00 in little-endian
        weights_data[row_offset + 208] = 0x00;
        weights_data[row_offset + 209] = 0x3C;
        // Set all scales to 1 (small positive value)
        for i in 0..16 {
            weights_data[row_offset + 192 + i] = 1;
        }
    }
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
    let mut k_val = k;
    let mut n_val = n;

    let mut args: [*mut c_void; 5] = [
        &mut output_ptr as *mut _ as *mut c_void,
        &mut weights_ptr as *mut _ as *mut c_void,
        &mut input_ptr as *mut _ as *mut c_void,
        &mut k_val as *mut _ as *mut c_void,
        &mut n_val as *mut _ as *mut c_void,
    ];

    // CoalescedQ6KGemvKernel: one warp (32 threads) per output row
    // Grid: N, Block: 32
    let config = LaunchConfig {
        grid: (n, 1, 1),
        block: (32, 1, 1),
        shared_mem: 0,
    };
    println!("       Grid: ({}, 1, 1), Block: (32, 1, 1)", n);

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

    println!("\n✓ SUCCESS! CoalescedQ6KGemvKernel executed without errors.");
    println!("Output[0..{}]: {:?}", output.len(), &output);
}
