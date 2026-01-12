use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    println!("=== ALL uses of %rd359 (hash_table_base) ===");
    for (i, line) in ptx.lines().enumerate() {
        if line.contains("%rd359") {
            println!("{:4}: {}", i + 1, line);
        }
    }

    println!("\n=== Checking if %rd359 is modified inside L_compress_loop ===");
    let lines: Vec<&str> = ptx.lines().collect();
    let mut in_loop = false;
    for (i, line) in lines.iter().enumerate() {
        if line.contains("L_compress_loop:") {
            in_loop = true;
        }
        if line.contains("L_emit_remaining:") || line.contains("L_compress_done:") {
            in_loop = false;
        }
        if in_loop && line.contains("%rd359") {
            // Check if %rd359 is the DESTINATION (being written to)
            let trimmed = line.trim();
            if let Some(after_instr) = trimmed.split_whitespace().nth(1) {
                if after_instr.starts_with("%rd359,") || after_instr.starts_with("%rd359") {
                    println!("{:4}: {} <-- WRITES TO %rd359!", i + 1, line);
                } else {
                    println!("{:4}: {} (reads)", i + 1, line);
                }
            }
        }
    }
}
