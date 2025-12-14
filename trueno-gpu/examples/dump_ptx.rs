// Quick script to dump PTX
use trueno_gpu::kernels::{GemmKernel, Kernel};

fn main() {
    let kernel = GemmKernel::tiled(128, 128, 128, 32);
    let ptx = kernel.emit_ptx();
    println!("=== TILED GEMM PTX ===\n{}", ptx);
}
