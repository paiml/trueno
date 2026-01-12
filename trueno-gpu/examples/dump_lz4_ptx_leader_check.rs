use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    println!("=== Code around leader check (lines 700-730) ===");
    let lines: Vec<&str> = ptx.lines().collect();
    for i in 700..730 {
        if i < lines.len() {
            println!("{:4}: {}", i + 1, lines[i]);
        }
    }
}
