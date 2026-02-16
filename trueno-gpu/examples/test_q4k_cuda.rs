//! Minimal Q4K GEMV CUDA test
//!
//! Run with: cargo run --release --features cuda --example test_q4k_cuda

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::kernels::{Kernel, Q4KGemvKernel};
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

fn generate_and_load_module(ctx: &CudaContext, n: u32, k: u32) -> Option<(CudaModule, Q4KGemvKernel)> {
    println!("[2/7] Generating Q4KGemvKernel PTX (N={}, K={})...", n, k);
    let kernel = Q4KGemvKernel::new(n, k);

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

    println!("[3/7] JIT compiling PTX...");
    match CudaModule::from_ptx(ctx, &ptx) {
        Ok(m) => {
            println!("       ✓ Module compiled");
            Some((m, kernel))
        }
        Err(e) => {
            eprintln!("Failed to load PTX module: {}", e);
            eprintln!("\nPTX dump (first 80 lines):");
            for (i, line) in ptx.lines().take(80).enumerate() {
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

struct GpuBuffers {
    weights: GpuBuffer<u8>,
    input: GpuBuffer<f32>,
    output: GpuBuffer<f32>,
    output_size: usize,
}

fn allocate_buffers(ctx: &CudaContext, n: u32, k: u32) -> Option<GpuBuffers> {
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

    Some(GpuBuffers {
        weights,
        input,
        output,
        output_size,
    })
}

fn launch_and_verify(
    stream: &CudaStream,
    module: &mut CudaModule,
    kernel: &Q4KGemvKernel,
    buffers: &GpuBuffers,
    n: u32,
    k: u32,
) -> Option<Vec<f32>> {
    println!("[6/7] Launching kernel...");
    let mut output_ptr = buffers.output.as_ptr();
    let mut weights_ptr = buffers.weights.as_ptr();
    let mut input_ptr = buffers.input.as_ptr();
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
    println!("║      trueno-gpu: Q4KGemvKernel CUDA Test             ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    let n: u32 = 256;
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
        println!("\n✓ SUCCESS! Q4KGemvKernel executed without errors.");
        println!("Output[0..5]: {:?}", &output[..5.min(output.len())]);
    }
}
