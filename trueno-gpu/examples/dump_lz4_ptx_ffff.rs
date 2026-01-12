use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    println!("=== Lines with 4294967295 (0xFFFFFFFF) ===");
    for (i, line) in ptx.lines().enumerate() {
        if line.contains("4294967295") {
            println!("{:4}: {}", i + 1, line);
        }
    }

    println!("\n=== st.u32 instructions that might be hash init ===");
    for (i, line) in ptx.lines().enumerate() {
        if line.contains("st.u32") && i > 200 && i < 900 {
            println!("{:4}: {}", i + 1, line);
        }
    }
}
