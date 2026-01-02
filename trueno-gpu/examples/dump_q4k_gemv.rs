// Quick script to dump Q4K GEMV PTX
use trueno_gpu::kernels::{Kernel, Q4KGemvKernel};

fn main() {
    let kernel = Q4KGemvKernel::new(1536, 1536);
    let ptx = kernel.emit_ptx();
    println!("{}", ptx);
}
