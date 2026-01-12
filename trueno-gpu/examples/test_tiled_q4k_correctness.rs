//! TiledQ4KGemvKernel correctness test - reproduces NaN bug
//!
//! This test uses valid Q4_K weight data to check for NaN outputs.

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::kernels::{Kernel, TiledQ4KGemvKernel};
use trueno_gpu::ptx::PtxModule;

/// Convert f32 to f16 bytes (simple approximation for valid values)
fn f32_to_f16_bytes(val: f32) -> [u8; 2] {
    // Simple f16 encoding for small positive values
    // f16 format: 1 sign + 5 exp + 10 mantissa
    if val == 0.0 {
        return [0, 0];
    }
    let bits = val.to_bits();
    let sign = (bits >> 31) as u16;
    let exp = ((bits >> 23) & 0xFF) as i16 - 127;  // f32 bias
    let mantissa = (bits >> 13) & 0x3FF;  // Top 10 bits of f32 mantissa

    // f16 bias is 15
    let f16_exp = (exp + 15).clamp(0, 31) as u16;
    let f16_bits = (sign << 15) | (f16_exp << 10) | (mantissa as u16);
    f16_bits.to_le_bytes()
}

/// Pack a valid Q4_K super-block with known values
fn create_valid_q4k_block(d: f32, dmin: f32, scale: u8, min_val: u8) -> [u8; 144] {
    let mut block = [0u8; 144];

    // d as f16 (bytes 0-1)
    let d_bytes = f32_to_f16_bytes(d);
    block[0] = d_bytes[0];
    block[1] = d_bytes[1];

    // dmin as f16 (bytes 2-3)
    let dmin_bytes = f32_to_f16_bytes(dmin);
    block[2] = dmin_bytes[0];
    block[3] = dmin_bytes[1];

    // scales (bytes 4-15) - 12 bytes
    // For sub-blocks 0-3: scale in low 6 bits, min in next 6 bits
    for i in 0..4 {
        block[4 + i] = scale & 0x3F;        // scale for sub-block i
        block[4 + 4 + i] = min_val & 0x3F;  // min for sub-block i+4
    }
    // Bytes 12-15 for sub-blocks 4-7 (high bits)
    for i in 0..4 {
        block[12 + i] = scale & 0x0F;  // low 4 bits of scale for sub-block 4+i
    }

    // qs (bytes 16-143) - 128 bytes = 256 x 4-bit values
    // Fill with known pattern: alternating 0-15
    for i in 0..128 {
        block[16 + i] = ((i & 0x0F) as u8) | ((((i + 1) & 0x0F) as u8) << 4);
    }

    block
}

fn test_correctness(ctx: &CudaContext, n: u32, k: u32) -> Result<bool, String> {
    println!("\n--- Correctness Test N={}, K={} ---", n, k);

    let kernel = TiledQ4KGemvKernel::new(n, k);

    // Generate PTX
    let ptx = PtxModule::new()
        .version(8, 0)
        .target("sm_89")
        .address_size(64)
        .add_kernel(kernel.build_ptx())
        .emit();

    // Load module
    let mut module = CudaModule::from_ptx(ctx, &ptx)
        .map_err(|e| format!("PTX compile failed: {}", e))?;

    let stream = CudaStream::new(ctx).map_err(|e| format!("Stream failed: {}", e))?;

    // Create valid Q4_K weights
    let n_super_blocks = (k as usize + 255) / 256;
    let weights_size = n as usize * n_super_blocks * 144;

    // Create weight data with valid Q4_K super-blocks
    let mut weights_data = Vec::with_capacity(weights_size);
    for row in 0..n {
        for sb in 0..n_super_blocks {
            // Use different d, dmin values per row/block to create variation
            let d = 0.1 * (1.0 + (row % 4) as f32 * 0.25);
            let dmin = 0.05 * (1.0 + (sb % 3) as f32 * 0.1);
            let block = create_valid_q4k_block(d, dmin, 3, 1);
            weights_data.extend_from_slice(&block);
        }
    }

    // Input vector: all 1.0
    let input_data = vec![1.0f32; k as usize];

    let shared_mem_bytes = k as usize * 4;

    let weights_buf: GpuBuffer<u8> = GpuBuffer::from_host(ctx, &weights_data)
        .map_err(|e| format!("Weights alloc failed: {}", e))?;
    let input_buf: GpuBuffer<f32> = GpuBuffer::from_host(ctx, &input_data)
        .map_err(|e| format!("Input alloc failed: {}", e))?;
    let output_buf: GpuBuffer<f32> = GpuBuffer::new(ctx, n as usize)
        .map_err(|e| format!("Output alloc failed: {}", e))?;

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

    let tiles_per_row = 4u32;
    let config = LaunchConfig {
        grid: ((n + tiles_per_row - 1) / tiles_per_row, 1, 1),
        block: (32 * tiles_per_row, 1, 1),
        shared_mem: shared_mem_bytes as u32,
    };

    unsafe {
        stream.launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .map_err(|e| format!("Launch failed: {}", e))?;
    }

    stream.synchronize().map_err(|e| format!("Sync failed: {}", e))?;

    let mut output = vec![0.0f32; n as usize];
    output_buf.copy_to_host(&mut output)
        .map_err(|e| format!("D2H failed: {}", e))?;

    // Check for NaN and report which outputs are affected
    let mut nan_outputs = Vec::new();
    let mut inf_outputs = Vec::new();
    let mut ok_outputs = Vec::new();

    for (i, &val) in output.iter().enumerate() {
        if val.is_nan() {
            nan_outputs.push(i);
        } else if val.is_infinite() {
            inf_outputs.push(i);
        } else {
            ok_outputs.push(i);
        }
    }

    println!("  Total outputs: {}", n);
    println!("  OK outputs: {} (first 8: {:?})", ok_outputs.len(), &ok_outputs[..ok_outputs.len().min(8)]);
    println!("  NaN outputs: {} (first 8: {:?})", nan_outputs.len(), &nan_outputs[..nan_outputs.len().min(8)]);
    println!("  Inf outputs: {}", inf_outputs.len());

    if !nan_outputs.is_empty() {
        // Show pattern of NaN outputs
        if nan_outputs.len() <= 32 {
            println!("  NaN indices: {:?}", nan_outputs);
        } else {
            // Check for pattern
            let mod4: Vec<usize> = nan_outputs.iter().map(|x| x % 4).collect();
            let unique_mod4: std::collections::HashSet<_> = mod4.iter().collect();
            println!("  NaN indices mod 4: {:?} (unique: {:?})", &mod4[..mod4.len().min(16)], unique_mod4);
        }

        // Show first few OK values
        if !ok_outputs.is_empty() {
            let first_ok_vals: Vec<f32> = ok_outputs.iter().take(8).map(|&i| output[i]).collect();
            println!("  First OK values: {:?}", first_ok_vals);
        }

        return Ok(false);
    }

    // Show first few output values
    println!("  First 8 outputs: {:?}", &output[..8.min(n as usize)]);

    Ok(true)
}

fn main() {
    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║    TiledQ4KGemvKernel Correctness Test               ║");
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

    let test_cases = [
        (8, 256),      // Minimal: 8 outputs, 1 super-block
        (16, 256),     // 16 outputs, 4 per block
        (32, 256),     // 32 outputs, 8 blocks
        (256, 256),    // Square
        (256, 512),    // 2 super-blocks per row
        (896, 896),    // 0.5B hidden
        (1536, 1536),  // 1.5B hidden
    ];

    let mut all_passed = true;
    for (n, k) in test_cases {
        match test_correctness(&ctx, n, k) {
            Ok(passed) => {
                if passed {
                    println!("✓ PASS: N={}, K={}", n, k);
                } else {
                    println!("✗ FAIL: N={}, K={} - NaN detected!", n, k);
                    all_passed = false;
                }
            }
            Err(e) => {
                println!("✗ ERROR: N={}, K={}: {}", n, k, e);
                all_passed = false;
            }
        }
    }

    println!("\n═══ {} ═══", if all_passed { "ALL TESTS PASSED" } else { "SOME TESTS FAILED" });
}
