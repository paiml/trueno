use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    // Find where %rd11 is first used/defined
    for (i, line) in ptx.lines().enumerate() {
        if line.contains("%rd11")
            && (line.contains("cvta") || line.contains("add.u64") || line.contains("mov"))
        {
            println!("{:4}: {}", i + 1, line);
        }
    }
}
