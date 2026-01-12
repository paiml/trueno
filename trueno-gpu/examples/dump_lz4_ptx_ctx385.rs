use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();
    
    let lines: Vec<&str> = ptx.lines().collect();
    // Print lines 380-400
    for i in 380..400 {
        if i < lines.len() {
            println!("{:4}: {}", i + 1, lines[i]);
        }
    }
}
