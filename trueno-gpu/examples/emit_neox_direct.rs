use trueno_gpu::kernels::{Kernel, RopeNeoxKernel};

fn main() {
    let kernel = RopeNeoxKernel::new(12, 128, 1_000_000.0);
    let ptx = kernel.emit_ptx();
    println!("{}", ptx);
}
