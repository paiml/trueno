//! Minimal CoalescedQ6K GEMV CUDA test
//!
//! Run with: cargo run --release --features cuda -p trueno-gpu --example test_coalesced_q6k

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::kernels::{CoalescedQ6KGemvKernel, Kernel};
use trueno_gpu::ptx::PtxModule;

fn init_cuda() -> Option<CudaContext> {
    println!("[1/7] Initializing CUDA...");
    match CudaContext::new(0) {
        Ok(c) => {
            let device_name = c.device_name().unwrap_or_else(|_| "Unknown".to_string());
            println!("       ✓ GPU: {}", device_name);
            Some(c)
        }
        Err(e) => {
            eprintln!("Failed to create CUDA context: {}", e);
            None
        }
    }
}

fn generate_and_load_module(
    ctx: &CudaContext,
    n: u32,
    k: u32,
) -> Option<(CudaModule, CoalescedQ6KGemvKernel)> {
    println!("[2/7] Generating CoalescedQ6KGemvKernel PTX (N={}, K={})...", n, k);
    let kernel = CoalescedQ6KGemvKernel::new(k, n);

    let ptx = PtxModule::new()
        .version(8, 0)
        .target("sm_86")
        .address_size(64)
        .add_kernel(kernel.build_ptx())
        .emit();
    println!("       PTX generated ({} bytes, {} lines)", ptx.len(), ptx.lines().count());

    println!("[3/7] JIT compiling PTX...");
    match CudaModule::from_ptx(ctx, &ptx) {
        Ok(m) => {
            println!("       ✓ Module compiled");
            Some((m, kernel))
        }
        Err(e) => {
            eprintln!("Failed to load PTX module: {}", e);
            eprintln!("\nPTX dump (first 100 lines):");
            for (i, line) in ptx.lines().take(100).enumerate() {
                eprintln!("{:4}: {}", i + 1, line);
            }
            None
        }
    }
}

fn create_stream(ctx: &CudaContext) -> Option<CudaStream> {
    println!("[4/7] Creating stream...");
    match CudaStream::new(ctx) {
        Ok(s) => {
            println!("       ✓ Stream created");
            Some(s)
        }
        Err(e) => {
            eprintln!("Failed to create stream: {}", e);
            None
        }
    }
}

fn build_q6k_weights(n: u32, n_super_blocks: usize) -> Vec<u8> {
    let weights_size = n as usize * n_super_blocks * 210;
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
    weights_data
}

struct GpuBuffers {
    weights: GpuBuffer<u8>,
    input: GpuBuffer<f32>,
    output: GpuBuffer<f32>,
    output_size: usize,
}

fn allocate_buffers(ctx: &CudaContext, n: u32, k: u32) -> Option<GpuBuffers> {
    // Q6_K format: 210 bytes per super-block (256 elements)
    // Layout: ql[128] + qh[64] + scales[16] + d[2] = 210 bytes
    let n_super_blocks = (k as usize + 255) / 256;
    let weights_size = n as usize * n_super_blocks * 210;
    let input_size = k as usize;
    let output_size = n as usize;

    println!("[5/7] Allocating buffers...");
    println!("       Weights: {} bytes ({} super-blocks per row)", weights_size, n_super_blocks);
    println!("       Input: {} floats", input_size);
    println!("       Output: {} floats", output_size);

    let weights_data = build_q6k_weights(n, n_super_blocks);
    let input_data = vec![1.0f32; input_size];

    let weights = match GpuBuffer::from_host(ctx, &weights_data) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to allocate weights buffer: {}", e);
            return None;
        }
    };
    let input = match GpuBuffer::from_host(ctx, &input_data) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to allocate input buffer: {}", e);
            return None;
        }
    };
    let output = match GpuBuffer::new(ctx, output_size) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to allocate output buffer: {}", e);
            return None;
        }
    };
    println!("       ✓ Buffers allocated");

    Some(GpuBuffers { weights, input, output, output_size })
}

fn launch_and_verify(
    stream: &CudaStream,
    module: &mut CudaModule,
    kernel: &CoalescedQ6KGemvKernel,
    buffers: &GpuBuffers,
    n: u32,
    k: u32,
) -> Option<Vec<f32>> {
    println!("[6/7] Launching kernel...");
    let mut output_ptr = buffers.output.as_ptr();
    let mut weights_ptr = buffers.weights.as_ptr();
    let mut input_ptr = buffers.input.as_ptr();
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
    let config = LaunchConfig { grid: (n, 1, 1), block: (32, 1, 1), shared_mem: 0 };
    println!("       Grid: ({}, 1, 1), Block: (32, 1, 1)", n);

    let start = std::time::Instant::now();
    unsafe {
        match stream.launch_kernel(module, kernel.name(), &config, &mut args) {
            Ok(()) => println!("       ✓ Kernel launched"),
            Err(e) => {
                eprintln!("Failed to launch kernel: {}", e);
                return None;
            }
        }
    }

    match stream.synchronize() {
        Ok(()) => {
            let elapsed = start.elapsed();
            println!("       ✓ Kernel executed in {:?}", elapsed);
        }
        Err(e) => {
            eprintln!("❌ Stream sync failed: {}", e);
            return None;
        }
    }

    println!("[7/7] Verifying...");
    let mut output = vec![0.0f32; buffers.output_size];
    match buffers.output.copy_to_host(&mut output) {
        Ok(()) => {
            println!("       ✓ Results copied back");
            Some(output)
        }
        Err(e) => {
            eprintln!("Failed to copy results: {}", e);
            None
        }
    }
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║   trueno-gpu: CoalescedQ6KGemvKernel CUDA Test       ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let n: u32 = 4;
    let k: u32 = 256;

    let ctx = match init_cuda() {
        Some(c) => c,
        None => return,
    };
    let (mut module, kernel) = match generate_and_load_module(&ctx, n, k) {
        Some(pair) => pair,
        None => return,
    };
    let stream = match create_stream(&ctx) {
        Some(s) => s,
        None => return,
    };
    let buffers = match allocate_buffers(&ctx, n, k) {
        Some(b) => b,
        None => return,
    };

    if let Some(output) = launch_and_verify(&stream, &mut module, &kernel, &buffers, n, k) {
        println!("\n✓ SUCCESS! CoalescedQ6KGemvKernel executed without errors.");
        println!("Output[0..{}]: {:?}", output.len(), &output);
    }
}
