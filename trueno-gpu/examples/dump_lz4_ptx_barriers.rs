use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    println!("=== ALL barrier instructions and labels around them ===");
    let lines: Vec<&str> = ptx.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if line.contains("bar.sync") || line.contains("L_") && line.trim().ends_with(':') {
            // Print 2 lines before for context
            if i >= 2 {
                println!("{:4}: {}", i - 1, lines[i - 2]);
                println!("{:4}: {}", i, lines[i - 1]);
            }
            println!("{:4}: {} <===", i + 1, line);
            if i + 2 < lines.len() {
                println!("{:4}: {}", i + 2, lines[i + 1]);
                println!("{:4}: {}", i + 3, lines[i + 2]);
            }
            println!();
        }
    }
}
