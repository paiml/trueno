use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    println!("=== ALL instructions that WRITE to %rd11 ===");
    for (i, line) in ptx.lines().enumerate() {
        let trimmed = line.trim();
        // Check if %rd11 is the destination register
        // PTX format: instr dest, src1, src2  or  instr dest, src
        if let Some(after_instr) = trimmed.split_whitespace().nth(1) {
            if after_instr.starts_with("%rd11,") || after_instr == "%rd11" || after_instr == "%rd11;" {
                println!("{:4}: {} <-- WRITES", i + 1, line);
            }
        }
    }

    println!("\n=== Checking for register reuse issues ===");
    // Count unique registers used
    let mut rd_regs: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for line in ptx.lines() {
        for word in line.split_whitespace() {
            if word.starts_with("%rd") {
                // Extract register name
                let reg = word.trim_end_matches(',').trim_end_matches(';');
                rd_regs.insert(reg);
            }
        }
    }
    println!("Total 64-bit registers used: {}", rd_regs.len());

    // Check max register number
    let max_rd: u32 = rd_regs.iter()
        .filter_map(|r| r.strip_prefix("%rd").and_then(|n| n.parse().ok()))
        .max()
        .unwrap_or(0);
    println!("Max register number: %rd{}", max_rd);
}
