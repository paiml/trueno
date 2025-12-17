//! Q5_K and Q6_K Quantized GEMM Kernel Example
//!
//! Demonstrates 5-bit and 6-bit quantized matrix multiplication kernels
//! for efficient inference with higher accuracy than Q4_K.
//!
//! Run with: cargo run -p trueno-gpu --example q5k_q6k_gemm
//!
//! ## Quantization Formats
//!
//! | Format | Bits | Bytes/256 | Accuracy |
//! |--------|------|-----------|----------|
//! | Q4_K   | 4    | 144       | Good     |
//! | Q5_K   | 5    | 176       | Better   |
//! | Q6_K   | 6    | 210       | Best     |

use trueno_gpu::kernels::{Kernel, Q5KKernel, Q6KKernel};

fn main() {
    println!("=== Q5_K and Q6_K Quantized GEMM Kernels ===\n");

    let m = 64; // Output rows
    let n = 64; // Output columns
    let k = 256; // Inner dimension (must be divisible by 256)

    // =========================================================================
    // Q5_K Kernel (PARITY-116)
    // =========================================================================
    println!("--- Q5_K Kernel (5-bit, 176 bytes/256 values) ---");
    let q5k = Q5KKernel::new(m, n, k);
    println!("Kernel name: {}", q5k.name());
    println!("Configuration: {}x{}x{}", m, n, k);
    println!("Super-blocks per row: {}", q5k.num_super_blocks_per_row());

    let ptx_q5k = q5k.emit_ptx();
    println!("PTX size: {} bytes", ptx_q5k.len());

    // Verify Q5_K-specific features
    assert!(
        ptx_q5k.contains("q5k_gemm_ggml"),
        "Missing Q5_K kernel name"
    );
    assert!(ptx_q5k.contains("sb_loop"), "Missing super-block loop");
    assert!(ptx_q5k.contains("sub_block_loop"), "Missing sub-block loop");
    // Q5_K loads both ql (4-bit) and qh (1-bit high) values
    let u8_loads = ptx_q5k.matches("ld.global.u8").count();
    assert!(
        u8_loads >= 4,
        "Q5_K should have multiple u8 loads for ql/qh"
    );
    println!("Q5_K verified: {} u8 loads for scales/ql/qh", u8_loads);

    // =========================================================================
    // Q6_K Kernel (PARITY-117)
    // =========================================================================
    println!("\n--- Q6_K Kernel (6-bit, 210 bytes/256 values) ---");
    let q6k = Q6KKernel::new(m, n, k);
    println!("Kernel name: {}", q6k.name());
    println!("Configuration: {}x{}x{}", m, n, k);
    println!("Super-blocks per row: {}", q6k.num_super_blocks_per_row());

    let ptx_q6k = q6k.emit_ptx();
    println!("PTX size: {} bytes", ptx_q6k.len());

    // Verify Q6_K-specific features
    assert!(
        ptx_q6k.contains("q6k_gemm_ggml"),
        "Missing Q6_K kernel name"
    );
    assert!(ptx_q6k.contains("sb_loop"), "Missing super-block loop");
    assert!(ptx_q6k.contains("sub_block_loop"), "Missing sub-block loop");
    // Q6_K uses signed offset (-32) for symmetric quantization
    assert!(
        ptx_q6k.contains("sub.f32") || ptx_q6k.contains("sub.rn.f32"),
        "Q6_K should have subtraction for signed offset"
    );
    println!("Q6_K verified: signed offset subtraction present");

    // =========================================================================
    // Comparison
    // =========================================================================
    println!("\n--- Format Comparison ---");
    println!("Q5_K PTX: {} bytes", ptx_q5k.len());
    println!("Q6_K PTX: {} bytes", ptx_q6k.len());
    assert_ne!(
        ptx_q5k, ptx_q6k,
        "Q5_K and Q6_K should produce different PTX"
    );
    println!("Kernels verified as distinct!");

    println!("\n=== Example Complete ===");
}
