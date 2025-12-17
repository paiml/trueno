//! GEMM Kernel Generation Example
//!
//! This example demonstrates generating a GEMM (General Matrix Multiply)
//! kernel in PTX using trueno-gpu.
//!
//! Run with: `cargo run -p trueno-gpu --example gemm_kernel`

use trueno_gpu::kernels::{GemmKernel, Kernel};
use trueno_gpu::ptx::PtxModule;

fn main() {
    println!("=== trueno-gpu: GEMM Kernel Generation ===\n");

    // Create GEMM kernels for different use cases
    let naive_gemm = GemmKernel::naive(1024, 1024, 1024);
    let tiled_gemm = GemmKernel::tiled(1024, 1024, 1024, 32);
    let tensor_gemm = GemmKernel::tensor_core(1024, 1024, 1024);

    println!("Available GEMM variants:");
    println!(
        "  1. {} - Simple implementation for correctness testing",
        naive_gemm.name()
    );
    println!(
        "  2. {} - Shared memory tiling for performance",
        tiled_gemm.name()
    );
    println!(
        "  3. {} - Tensor Core acceleration (SM 7.0+)",
        tensor_gemm.name()
    );

    println!("\n--- Naive GEMM PTX Generation ---\n");

    // Generate PTX for naive GEMM
    let ptx = naive_gemm.emit_ptx();

    // Print first 80 lines of PTX
    let lines: Vec<&str> = ptx.lines().collect();
    for (i, line) in lines.iter().take(80).enumerate() {
        println!("{:4}: {}", i + 1, line);
    }

    if lines.len() > 80 {
        println!("... ({} more lines)", lines.len() - 80);
    }

    println!("\n--- GEMM Kernel Details ---");
    println!("Matrix dimensions: M=1024, N=1024, K=1024");
    println!("Operation: C[i,j] = sum(A[i,k] * B[k,j]) for k=0..K");
    println!("PTX size: {} bytes", ptx.len());
    println!("PTX lines: {}", lines.len());

    // Create a full module with multiple kernels
    println!("\n--- Multi-Kernel Module ---\n");

    let module = PtxModule::new()
        .version(8, 0)
        .target("sm_70")
        .address_size(64)
        .add_kernel(naive_gemm.build_ptx())
        .add_kernel(tiled_gemm.build_ptx());

    let full_ptx = module.emit();
    println!("Combined module size: {} bytes", full_ptx.len());
    println!("Contains {} entry points", 2);
}
