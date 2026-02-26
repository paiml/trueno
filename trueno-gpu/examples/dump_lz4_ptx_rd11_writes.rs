use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

/// Check if a PTX instruction writes to `%rd11` (is it the destination register).
fn writes_to_rd11(trimmed: &str) -> bool {
    trimmed
        .split_whitespace()
        .nth(1)
        .is_some_and(|dest| dest.starts_with("%rd11,") || dest == "%rd11" || dest == "%rd11;")
}

/// Extract all 64-bit register names from PTX source.
fn collect_rd_registers(ptx: &str) -> std::collections::HashSet<&str> {
    let mut regs = std::collections::HashSet::new();
    for line in ptx.lines() {
        for word in line.split_whitespace() {
            if word.starts_with("%rd") {
                let reg = word.trim_end_matches(',').trim_end_matches(';');
                regs.insert(reg);
            }
        }
    }
    regs
}

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    println!("=== ALL instructions that WRITE to %rd11 ===");
    for (i, line) in ptx.lines().enumerate() {
        if writes_to_rd11(line.trim()) {
            println!("{:4}: {} <-- WRITES", i + 1, line);
        }
    }

    println!("\n=== Checking for register reuse issues ===");
    let rd_regs = collect_rd_registers(&ptx);
    println!("Total 64-bit registers used: {}", rd_regs.len());

    // Check max register number
    let max_rd: u32 = rd_regs
        .iter()
        .filter_map(|r| r.strip_prefix("%rd").and_then(|n| n.parse().ok()))
        .max()
        .unwrap_or(0);
    println!("Max register number: %rd{}", max_rd);
}
