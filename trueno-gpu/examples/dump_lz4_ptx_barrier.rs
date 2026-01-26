use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();

    // Print lines around barriers and leader check
    let lines: Vec<&str> = ptx.lines().collect();
    for (i, line) in lines.iter().enumerate() {
        if line.contains("bar.sync")
            || line.contains("L_leader")
            || line.contains("L_warp_done")
            || line.contains("p_leader")
            || line.contains("L_compress_loop")
            || line.contains("setp.eq.u32 %p") && line.contains(", 0;")
        {
            // Print context
            let start = i.saturating_sub(2);
            let end = (i + 3).min(lines.len());
            for j in start..end {
                let marker = if j == i { ">>>" } else { "   " };
                println!("{} {:4}: {}", marker, j + 1, lines[j]);
            }
            println!();
        }
    }
}
