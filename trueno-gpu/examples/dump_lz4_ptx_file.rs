use std::fs;
use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(1); // Single page to match test
    let ptx = kernel.emit_ptx();
    fs::write("/tmp/lz4_kernel.ptx", &ptx).expect("write failed");
    println!("PTX written to /tmp/lz4_kernel.ptx ({} bytes)", ptx.len());
}
