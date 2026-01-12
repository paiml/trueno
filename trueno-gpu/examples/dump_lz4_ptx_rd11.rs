use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    println!("=== ALL lines containing %rd11 ===");
    for (i, line) in ptx.lines().enumerate() {
        if line.contains("%rd11") {
            // Check if %rd11 is the destination (written to)
            let is_dest = line.trim().starts_with("mov")
                || line.trim().starts_with("cvta")
                || line.trim().starts_with("add")
                || line.trim().starts_with("ld")
                || line.trim().starts_with("cvt");

            // Check if it appears as first register after instruction
            let trimmed = line.trim();
            let writes_rd11 = if let Some(rest) = trimmed.split_whitespace().nth(1) {
                rest.starts_with("%rd11,") || rest == "%rd11,"
            } else {
                false
            };

            let marker = if writes_rd11 { " <-- WRITES" } else { "" };
            println!("{:4}: {}{}", i + 1, line, marker);
        }
    }

    println!("\n=== First 50 lines of PTX ===");
    for (i, line) in ptx.lines().enumerate().take(50) {
        println!("{:4}: {}", i + 1, line);
    }
}
