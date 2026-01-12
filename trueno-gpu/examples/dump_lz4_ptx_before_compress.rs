use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    println!("=== Code before L_compress_start (lines 850-900) ===");
    let lines: Vec<&str> = ptx.lines().collect();
    for i in 850..900 {
        if i < lines.len() {
            println!("{:4}: {}", i + 1, lines[i]);
        }
    }
}
