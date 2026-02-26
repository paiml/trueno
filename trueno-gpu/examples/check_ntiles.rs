use trueno_gpu::kernels::{GemmKernel, Kernel};

/// Check if a PTX line is a constant assignment of interest.
fn is_interesting_constant(trimmed: &str) -> bool {
    trimmed.contains("mov.u32")
        && (trimmed.ends_with(", 2;") || trimmed.ends_with(", 1;") || trimmed.ends_with(", 32;"))
}

/// Count occurrences of `mov.u32 ..., <suffix>;` in PTX.
fn count_mov_u32_with_suffix(ptx: &str, suffix: &str) -> usize {
    ptx.lines().filter(|l| l.contains("mov.u32") && l.trim().ends_with(suffix)).count()
}

fn main() {
    // Test with k=64, tile_size=32 -> n_tiles should be 2
    let kernel = GemmKernel::tiled(4, 192, 64, 32);
    let ptx = kernel.emit_ptx();

    println!("=== PTX for k=64, tile_size=32 ===");

    // Look for the n_tiles constant (should be 2)
    for line in ptx.lines() {
        let trimmed = line.trim();
        if is_interesting_constant(trimmed) {
            println!("CONSTANT: {}", line);
        }
        if trimmed.contains("tile_loop") {
            println!("{}", line);
        }
    }

    // Count occurrences
    println!("\n=== Value Analysis ===");

    // The n_tiles register should have value 2 (k=64, tile=32)
    println!("mov.u32 ..., 2; occurrences: {}", count_mov_u32_with_suffix(&ptx, ", 2;"));

    // Check for potential bugs - if n_tiles is 1 instead of 2
    println!("mov.u32 ..., 1; occurrences: {}", count_mov_u32_with_suffix(&ptx, ", 1;"));

    // Print all mov.u32 lines
    println!("\n=== All mov.u32 instructions ===");
    for line in ptx.lines() {
        if line.contains("mov.u32") {
            println!("{}", line);
        }
    }
}
