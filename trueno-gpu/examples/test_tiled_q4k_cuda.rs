//! TiledQ4KGemvKernel CUDA test at various dimensions
//!
//! Tests K=896 (0.5B hidden) and K=1536 (1.5B hidden)

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::kernels::{Kernel, TiledQ4KGemvKernel};
use trueno_gpu::ptx::PtxModule;

fn test_tiled_kernel(ctx: &CudaContext, n: u32, k: u32) -> Result<(), String> {
    println!("\n--- Testing TiledQ4KGemvKernel N={}, K={} ---", n, k);
    
    let kernel = TiledQ4KGemvKernel::new(n, k);
    
    // Generate PTX
    let ptx = PtxModule::new()
        .version(8, 0)
        .target("sm_89")
        .address_size(64)
        .add_kernel(kernel.build_ptx())
        .emit();
    println!("PTX: {} bytes", ptx.len());

    // Load module
    let mut module = CudaModule::from_ptx(ctx, &ptx)
        .map_err(|e| format!("PTX compile failed: {}", e))?;
    println!("✓ Module compiled");

    let stream = CudaStream::new(ctx).map_err(|e| format!("Stream failed: {}", e))?;

    // Allocate buffers
    // Q4_K super-block = 144 bytes: d(2) + dmin(2) + scales(12) + qs(128)
    let n_super_blocks = (k as usize + 255) / 256;
    let weights_size = n as usize * n_super_blocks * 144;
    let input_size = k as usize;
    let output_size = n as usize;
    
    // Q4_K needs K * 4 bytes of shared memory for input tile
    let shared_mem_bytes = k as usize * 4;
    println!("Shared memory needed: {} bytes", shared_mem_bytes);

    let weights_data = vec![0u8; weights_size];
    let input_data = vec![1.0f32; input_size];

    let weights_buf: GpuBuffer<u8> = GpuBuffer::from_host(ctx, &weights_data)
        .map_err(|e| format!("Weights alloc failed: {}", e))?;
    let input_buf: GpuBuffer<f32> = GpuBuffer::from_host(ctx, &input_data)
        .map_err(|e| format!("Input alloc failed: {}", e))?;
    let output_buf: GpuBuffer<f32> = GpuBuffer::new(ctx, output_size)
        .map_err(|e| format!("Output alloc failed: {}", e))?;
    println!("✓ Buffers allocated");

    // Build args
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

    // TiledQ4KGemvKernel: 4 tiles per row, one warp per tile
    // Grid: ceil(N / 4), Block: 128 (4 warps)
    let tiles_per_row = 4u32;
    let config = LaunchConfig {
        grid: ((n + tiles_per_row - 1) / tiles_per_row, 1, 1),
        block: (32 * tiles_per_row, 1, 1),
        shared_mem: shared_mem_bytes as u32,
    };
    println!("Grid: ({}, 1, 1), Block: ({}, 1, 1), SharedMem: {}", 
             config.grid.0, config.block.0, config.shared_mem);

    let start = std::time::Instant::now();
    unsafe {
        stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .map_err(|e| format!("Launch failed: {}", e))?;
    }
    
    stream.synchronize().map_err(|e| format!("Sync failed: {}", e))?;
    let elapsed = start.elapsed();
    println!("✓ Kernel executed in {:?}", elapsed);

    let mut output = vec![0.0f32; output_size];
    output_buf.copy_to_host(&mut output)
        .map_err(|e| format!("D2H failed: {}", e))?;
    
    println!("✓ SUCCESS for N={}, K={}", n, k);
    Ok(())
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║    TiledQ4KGemvKernel CUDA Dimension Test            ║");
    println!("╚══════════════════════════════════════════════════════╝");

    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to create CUDA context: {}", e);
            return;
        }
    };
    let device_name = ctx.device_name().unwrap_or_else(|_| "Unknown".to_string());
    println!("GPU: {}", device_name);

    // Test dimensions from various models
    let test_cases = [
        (256, 256),      // Baseline
        (896, 896),      // 0.5B hidden (WORKS)
        (4864, 896),     // 0.5B gate/up (N=intermediate, K=hidden)
        (1536, 1536),    // 1.5B hidden
        (8960, 1536),    // 1.5B gate/up (N=intermediate, K=hidden) - SUSPECT
        (1536, 8960),    // 1.5B down (K > 8192, should use fallback)
        (3584, 3584),    // 7B hidden
    ];

    for (n, k) in test_cases {
        match test_tiled_kernel(&ctx, n, k) {
            Ok(()) => println!("✓ PASS: N={}, K={}", n, k),
            Err(e) => println!("✗ FAIL: N={}, K={}: {}", n, k, e),
        }
    }

    println!("\n═══ Test Complete ═══");
}
