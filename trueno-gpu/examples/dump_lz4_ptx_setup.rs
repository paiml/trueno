use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    // Print lines 30-60 to see the setup
    for (i, line) in ptx.lines().enumerate() {
        if i >= 30 && i <= 65 {
            println!("{:4}: {}", i + 1, line);
        }
    }
}
