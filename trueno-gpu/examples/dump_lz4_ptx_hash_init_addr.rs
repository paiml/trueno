use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    println!("=== Hash table init code (lines 263-280) ===");
    let lines: Vec<&str> = ptx.lines().collect();
    for i in 263..320 {
        if i < lines.len() {
            println!("{:4}: {}", i + 1, lines[i]);
        }
    }
}
