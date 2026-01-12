use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    println!("=== Skip global load section (lines 180-270) ===");
    let lines: Vec<&str> = ptx.lines().collect();
    for i in 180..270 {
        if i < lines.len() {
            println!("{:4}: {}", i + 1, lines[i]);
        }
    }
}
