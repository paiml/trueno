use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();
    
    // Find all setp instructions to see the leader check
    for (i, line) in ptx.lines().enumerate() {
        if line.contains("setp") && (line.contains("eq") || line.contains("lt")) {
            println!("{:4}: {}", i + 1, line);
        }
    }
}
