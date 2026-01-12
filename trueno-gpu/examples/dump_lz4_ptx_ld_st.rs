use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    println!("=== ALL load/store instructions (excluding global) ===");
    for (i, line) in ptx.lines().enumerate() {
        if (line.contains("ld.") || line.contains("st.")) && !line.contains(".global") {
            println!("{:4}: {}", i + 1, line);
        }
    }
}
