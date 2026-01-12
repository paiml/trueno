use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    println!("=== All labels ===");
    for (i, line) in ptx.lines().enumerate() {
        if line.trim().starts_with("L_") && line.trim().ends_with(":") {
            println!("{:4}: {}", i + 1, line);
        }
    }
}
