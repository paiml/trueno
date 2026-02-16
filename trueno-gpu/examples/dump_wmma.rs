use trueno_gpu::kernels::{GemmKernel, Kernel};

fn is_wmma_related(line: &str) -> bool {
    line.contains("cvta") || line.contains("wmma.load") || line.contains("shared_base")
}

fn print_wmma_section(ptx: &str) {
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

fn main() {
    let kernel = GemmKernel::wmma_fp16(64, 64, 64);
    let ptx = kernel.emit_ptx();

    for (i, line) in ptx.lines().enumerate() {
        if is_wmma_related(line) {
            println!("{:4}: {}", i + 1, line);
        }
    }

    print_wmma_section(&ptx);
}
