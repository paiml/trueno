use trueno_gpu::kernels::{Kernel, RopeNeoxIndirectKernel};

fn main() {
    let kernel = RopeNeoxIndirectKernel::new(12, 128, 1_000_000.0);
    let ptx = kernel.emit_ptx();
    println!("{}", ptx);
}
