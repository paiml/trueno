use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn print_matching(ptx: &str, header: &str, predicate: impl Fn(&str) -> bool) {
    println!("{header}");
    for (i, line) in ptx.lines().enumerate() {
        if predicate(line) {
            println!("{:4}: {}", i + 1, line);
        }
    }
}

fn is_rd11_setup(line: &str) -> bool {
    line.contains("%rd11")
        && (line.contains("cvta")
            || (line.contains("add") && !line.contains("add.u64 %rd"))
            || line.trim().starts_with("mov"))
}

fn print_line_range(ptx: &str, start: usize, end: usize) {
    println!("\n=== Lines {}-{} (around hash table access) ===", start + 1, end);
    let lines: Vec<&str> = ptx.lines().collect();
    for i in start..end.min(lines.len()) {
        println!("{:4}: {}", i + 1, lines[i]);
    }
}

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    print_matching(&ptx, "=== Looking for %rd359 definition ===", |line| {
        line.contains("%rd359") && !line.contains("add.u64 %rd369")
    });
    print_matching(&ptx, "\n=== Looking for 4096 (PAGE_SIZE) ===", |line| {
        line.contains("4096")
    });
    print_matching(&ptx, "\n=== Looking for smem_base (%rd11) setup ===", is_rd11_setup);
    print_line_range(&ptx, 880, 920);
}
