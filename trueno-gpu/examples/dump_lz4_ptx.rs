//! Dump LZ4 kernel PTX around hash table code

use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

fn main() {
    let kernel = Lz4WarpCompressKernel::new(3);
    let ptx = kernel.emit_ptx();
    
    println!("=== LZ4 Kernel PTX (hash table section) ===");
    let lines: Vec<&str> = ptx.lines().collect();
    
    // Find L_compress_loop and print 80 lines
    let mut in_section = false;
    let mut count = 0;
    for (i, line) in lines.iter().enumerate() {
        if line.contains("L_compress_loop:") {
            in_section = true;
        }
        if in_section {
            println!("{:4}: {}", i + 1, line);
            count += 1;
            if count > 80 {
                break;
            }
        }
    }
}
