use trueno_gpu::kernels::{Kernel, ResidualAddKernel};
fn main() {
    let k = ResidualAddKernel::new(1536);
    let ptx = k.emit_ptx();
    println!("{}", ptx);
    eprintln!("PTX size: {} bytes", ptx.len());
}
