use trueno_gpu::kernels::{GemmKernel, Kernel};

fn main() {
    // Test with k=64, tile_size=32 -> n_tiles should be 2
    let kernel = GemmKernel::tiled(4, 192, 64, 32);
    let ptx = kernel.emit_ptx();
    
    println!("=== PTX for k=64, tile_size=32 ===");
    
    // Look for the n_tiles constant (should be 2)
    for line in ptx.lines() {
        if line.contains("mov.u32") && (line.contains(", 2;") || line.contains(", 1;") || line.contains(", 32;")) {
            println!("CONSTANT: {}", line);
        }
        if line.contains("tile_loop") {
            println!("{}", line);
        }
    }
    
    // Count occurrences
    println!("\n=== Value Analysis ===");
    
    // The n_tiles register should have value 2 (k=64, tile=32)
    let mov_2_count = ptx.lines()
        .filter(|l: &&str| l.contains("mov.u32") && l.trim().ends_with(", 2;"))
        .count();
    println!("mov.u32 ..., 2; occurrences: {}", mov_2_count);
    
    // Check for potential bugs - if n_tiles is 1 instead of 2
    let mov_1_count = ptx.lines()
        .filter(|l: &&str| l.contains("mov.u32") && l.trim().ends_with(", 1;"))
        .count();
    println!("mov.u32 ..., 1; occurrences: {}", mov_1_count);
    
    // Print all mov.u32 lines
    println!("\n=== All mov.u32 instructions ===");
    for line in ptx.lines() {
        if line.contains("mov.u32") {
            println!("{}", line);
        }
    }
}
