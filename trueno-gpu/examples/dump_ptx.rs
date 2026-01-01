// Quick script to dump PTX
use trueno_gpu::kernels::{AttentionKernel, GemmKernel, Kernel};

fn main() {
    println!("=== TENSOR CORE ATTENTION PTX ===\n");
    let kernel = AttentionKernel::tensor_core(64, 64);
    let ptx = kernel.emit_ptx();
    println!("{}", ptx);
    println!("\n=== PTX Size: {} bytes ===\n", ptx.len());

    // Check for WMMA instructions
    let wmma_count = ptx.matches("wmma.").count();
    println!("WMMA instruction count: {}", wmma_count);

    if ptx.contains("wmma.load.a") {
        println!("✓ wmma.load.a found");
    } else {
        println!("✗ wmma.load.a NOT found");
    }
    if ptx.contains("wmma.load.b") {
        println!("✓ wmma.load.b found");
    } else {
        println!("✗ wmma.load.b NOT found");
    }
    if ptx.contains("wmma.mma") {
        println!("✓ wmma.mma found");
    } else {
        println!("✗ wmma.mma NOT found");
    }
    if ptx.contains("wmma.store") {
        println!("✓ wmma.store found");
    } else {
        println!("✗ wmma.store NOT found");
    }

    println!("\n=== TILED GEMM PTX (for comparison) ===\n");
    let gemm_kernel = GemmKernel::tiled(128, 128, 128, 32);
    let gemm_ptx = gemm_kernel.emit_ptx();
    println!("GEMM PTX Size: {} bytes", gemm_ptx.len());
}
