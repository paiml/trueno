use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    // Find hash_table_base definition (should be smem_base + 4096)
    println!("=== Looking for %rd359 definition ===");
    for (i, line) in ptx.lines().enumerate() {
        // Look for any instruction that writes to %rd359
        if line.contains("%rd359") && !line.contains("add.u64 %rd369") {
            println!("{:4}: {}", i + 1, line);
        }
    }

    println!("\n=== Looking for 4096 (PAGE_SIZE) ===");
    for (i, line) in ptx.lines().enumerate() {
        if line.contains("4096") {
            println!("{:4}: {}", i + 1, line);
        }
    }

    println!("\n=== Looking for smem_base (%rd11) setup ===");
    for (i, line) in ptx.lines().enumerate() {
        if line.contains("%rd11")
            && (line.contains("cvta")
                || line.contains("add") && !line.contains("add.u64 %rd")
                || line.trim().starts_with("mov"))
        {
            println!("{:4}: {}", i + 1, line);
        }
    }

    println!("\n=== Lines 880-920 (around hash table access) ===");
    let lines: Vec<&str> = ptx.lines().collect();
    for i in 880..920 {
        if i < lines.len() {
            println!("{:4}: {}", i + 1, lines[i]);
        }
    }
}
