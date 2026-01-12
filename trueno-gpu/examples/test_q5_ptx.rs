use trueno_gpu::kernels::{Kernel, Q5_0GemvKernel};

fn main() {
    let kernel = Q5_0GemvKernel::new(896, 896);
    let ptx = kernel.emit_ptx();
    println!("{}", ptx);
}
