use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn is_write_to_rd359(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.split_whitespace().nth(1).is_some_and(|operand| operand.starts_with("%rd359"))
}

fn check_rd359_in_loop(lines: &[&str]) {
    println!("\n=== Checking if %rd359 is modified inside L_compress_loop ===");
    let mut in_loop = false;
    for (i, line) in lines.iter().enumerate() {
        if line.contains("L_compress_loop:") {
            in_loop = true;
        }
        if line.contains("L_emit_remaining:") || line.contains("L_compress_done:") {
            in_loop = false;
        }
        if in_loop && line.contains("%rd359") {
            let suffix =
                if is_write_to_rd359(line) { " <-- WRITES TO %rd359!" } else { " (reads)" };
            println!("{:4}: {}{}", i + 1, line, suffix);
        }
    }
}

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    println!("=== ALL uses of %rd359 (hash_table_base) ===");
    for (i, line) in ptx.lines().enumerate() {
        if line.contains("%rd359") {
            println!("{:4}: {}", i + 1, line);
        }
    }

    let lines: Vec<&str> = ptx.lines().collect();
    check_rd359_in_loop(&lines);
}
