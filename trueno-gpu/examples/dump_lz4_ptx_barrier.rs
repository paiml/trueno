use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

const BARRIER_PATTERNS: &[&str] = &[
    "bar.sync",
    "L_leader",
    "L_warp_done",
    "p_leader",
    "L_compress_loop",
];

fn is_barrier_related(line: &str) -> bool {
    BARRIER_PATTERNS.iter().any(|p| line.contains(p))
        || (line.contains("setp.eq.u32 %p") && line.contains(", 0;"))
}

fn print_context(lines: &[&str], center: usize) {
    let start = center.saturating_sub(2);
    let end = (center + 3).min(lines.len());
    for j in start..end {
        let marker = if j == center { ">>>" } else { "   " };
        println!("{} {:4}: {}", marker, j + 1, lines[j]);
    }
    println!();
}

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();
    let lines: Vec<&str> = ptx.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        if is_barrier_related(line) {
            print_context(&lines, i);
        }
    }
}
