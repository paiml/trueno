use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    for (i, line) in ptx.lines().enumerate() {
        if line.contains("%r123") {
            println!("{:4}: {}", i + 1, line);
        }
    }
}
