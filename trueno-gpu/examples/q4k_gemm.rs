//! Q4_K Quantized GEMM Kernel Example
//!
//! Demonstrates the Q4_K quantized matrix multiplication kernel
//! for efficient inference with 4-bit quantized weights.
//!
//! Run with: cargo run --example q4k_gemm
//!
//! ## Q4_K Simplified Format
//!
//! Each 32-value block is stored as 18 bytes:
//! - 2 bytes: f16 scale factor
//! - 16 bytes: 32 × 4-bit quantized values
//!
//! Dequantization: `value = scale * quant_value`

use trueno_gpu::kernels::{Kernel, QuantizeKernel};

fn main() {
    println!("=== Q4_K Quantized GEMM Kernel ===\n");

    // Create a small kernel for demonstration
    let m = 64;  // Output rows
    let n = 64;  // Output columns
    let k = 128; // Inner dimension (must be divisible by 32)

    // Create simplified Q4_K kernel (32 values per block, 18 bytes per block)
    let kernel = QuantizeKernel::new(m, n, k);
    println!("Kernel name: {}", kernel.name());
    println!("Configuration: {}x{}x{}", m, n, k);
    println!("Blocks per row: {}", kernel.num_blocks_per_row());
    println!();

    // Generate PTX
    let ptx = kernel.emit_ptx();
    println!("=== Generated PTX (first 80 lines) ===");
    for (i, line) in ptx.lines().take(80).enumerate() {
        println!("{:3}: {}", i + 1, line);
    }

    // Verify key PTX features
    println!("\n=== PTX Verification ===");
    assert!(
        ptx.contains(".param .u64 a_ptr"),
        "Missing input activation pointer"
    );
    assert!(
        ptx.contains(".param .u64 b_quant_ptr"),
        "Missing quantized weights pointer"
    );
    assert!(
        ptx.contains(".param .u64 c_ptr"),
        "Missing output pointer"
    );
    assert!(ptx.contains("ld.global.b16"), "Missing f16 load (b16 format)");
    assert!(ptx.contains("cvt.f32.f16"), "Missing f16→f32 conversion");
    assert!(ptx.contains("shfl.sync"), "Missing warp shuffle reduction");
    assert!(ptx.contains("min.u32"), "Missing address clamping");
    println!("All PTX features verified!");

    // Also demonstrate GGML format kernel
    println!("\n=== GGML Q4_K Super-block Format ===");
    let ggml_kernel = QuantizeKernel::ggml(m, n, 256); // K must be divisible by 256
    println!("Kernel name: {}", ggml_kernel.name());
    println!("Super-blocks per row: {}", ggml_kernel.num_super_blocks_per_row());

    let ggml_ptx = ggml_kernel.emit_ptx();
    assert!(ggml_ptx.contains("sb_loop"), "Missing super-block loop");
    assert!(
        ggml_ptx.contains("sub_block_loop"),
        "Missing sub-block loop"
    );
    println!("GGML kernel verified!");

    println!("\n=== Example Complete ===");
}
