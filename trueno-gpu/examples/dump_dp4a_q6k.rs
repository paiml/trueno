use trueno_gpu::kernels::{Dp4aQ6KGemvKernel, Kernel};
fn main() {
    let k = Dp4aQ6KGemvKernel::with_warps(1536, 1536, 4);
    let ptx = k.emit_ptx();
    println!("{}", ptx);
    eprintln!("PTX size: {} bytes", ptx.len());
}
