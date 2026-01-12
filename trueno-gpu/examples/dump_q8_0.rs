use trueno_gpu::kernels::{Q8_0GemvKernel, Kernel};

fn main() {
    let kernel = Q8_0GemvKernel::new(5504, 2048);
    let ptx = kernel.emit_ptx();
    println!("{}", ptx);
}
