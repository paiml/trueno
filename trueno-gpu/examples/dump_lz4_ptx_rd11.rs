use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn check_writes_rd11(line: &str) -> bool {
    let trimmed = line.trim();
    let is_write_instr = trimmed.starts_with("mov")
        || trimmed.starts_with("cvta")
        || trimmed.starts_with("add")
        || trimmed.starts_with("ld")
        || trimmed.starts_with("cvt");

    if !is_write_instr {
        return false;
    }

    trimmed
        .split_whitespace()
        .nth(1)
        .is_some_and(|rest| rest.starts_with("%rd11,") || rest == "%rd11,")
}

fn dump_rd11_lines(ptx: &str) {
    println!("=== ALL lines containing %rd11 ===");
    for (i, line) in ptx.lines().enumerate() {
        if line.contains("%rd11") {
            let marker = if check_writes_rd11(line) { " <-- WRITES" } else { "" };
            println!("{:4}: {}{}", i + 1, line, marker);
        }
    }
}

fn dump_ptx_header(ptx: &str) {
    println!("\n=== First 50 lines of PTX ===");
    for (i, line) in ptx.lines().enumerate().take(50) {
        println!("{:4}: {}", i + 1, line);
    }
}

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    dump_rd11_lines(&ptx);
    dump_ptx_header(&ptx);
}
