use trueno_gpu::kernels::{GemmKernel, Kernel};

fn main() {
    let kernel = GemmKernel::wmma_fp16(64, 64, 64);
    let ptx = kernel.emit_ptx();

    // Print lines with cvta to verify the fix
    for (i, line) in ptx.lines().enumerate() {
        if line.contains("cvta") || line.contains("wmma.load") || line.contains("shared_base") {
            println!("{:4}: {}", i + 1, line);
        }
    }

    println!("\n=== Full WMMA section ===");
    let mut in_wmma = false;
    for line in ptx.lines() {
        if line.contains("Synchronize before WMMA") || line.contains("bar.sync 0") {
            in_wmma = true;
        }
        if in_wmma {
            println!("{}", line);
            if line.contains("bar.sync 1") {
                break;
            }
        }
    }
}
