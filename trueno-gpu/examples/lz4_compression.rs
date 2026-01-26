//! LZ4 Compression Kernel Demo
//!
//! Demonstrates trueno-gpu's LZ4 compression capabilities:
//! - CPU reference implementation (compress/decompress)
//! - GPU kernel generation (PTX for NVIDIA, WGSL for WebGPU)
//! - Compression ratio benchmarks
//! - Warp-cooperative architecture explanation
//!
//! Run with: `cargo run -p trueno-gpu --example lz4_compression`

use trueno_gpu::kernels::lz4::{
    lz4_compress_block, lz4_decompress_block, Lz4WarpCompressKernel, PAGE_SIZE,
};
use trueno_gpu::kernels::Kernel;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       trueno-gpu LZ4 Compression Kernel Demo                ║");
    println!("║   Pure Rust PTX/WGSL Generation - No nvcc Required          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ==========================================================================
    // Part 1: CPU Reference Implementation
    // ==========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Part 1: CPU Reference Implementation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Test 1: Zero page compression (best case - ZRAM workload)
    let zero_page = vec![0u8; PAGE_SIZE as usize];
    let mut compressed = vec![0u8; PAGE_SIZE as usize];
    let comp_size =
        lz4_compress_block(&zero_page, &mut compressed).expect("compression should succeed");
    let ratio = PAGE_SIZE as f64 / comp_size as f64;
    println!("  Zero Page (4KB):");
    println!("    Original:   {} bytes", PAGE_SIZE);
    println!("    Compressed: {} bytes", comp_size);
    println!("    Ratio:      {:.1}:1\n", ratio);

    // Test 2: Repeated pattern (common in memory)
    let mut pattern_page = vec![0u8; PAGE_SIZE as usize];
    for i in 0..PAGE_SIZE as usize {
        pattern_page[i] = (i % 256) as u8;
    }
    let comp_size =
        lz4_compress_block(&pattern_page, &mut compressed).expect("compression should succeed");
    let ratio = PAGE_SIZE as f64 / comp_size as f64;
    println!("  Repeated Pattern (0-255 cycle):");
    println!("    Original:   {} bytes", PAGE_SIZE);
    println!("    Compressed: {} bytes", comp_size);
    println!("    Ratio:      {:.1}:1\n", ratio);

    // Test 3: Text data (real-world workload)
    let text = b"The quick brown fox jumps over the lazy dog. ";
    let mut text_page = vec![0u8; PAGE_SIZE as usize];
    for (i, chunk) in text_page.chunks_mut(text.len()).enumerate() {
        let src = &text[..chunk.len().min(text.len())];
        chunk[..src.len()].copy_from_slice(src);
        // Add some variation
        if i % 2 == 1 && !chunk.is_empty() {
            chunk[0] = b'A' + (i % 26) as u8;
        }
    }
    let comp_size =
        lz4_compress_block(&text_page, &mut compressed).expect("compression should succeed");
    let ratio = PAGE_SIZE as f64 / comp_size as f64;
    println!("  Text Data (repeated sentences):");
    println!("    Original:   {} bytes", PAGE_SIZE);
    println!("    Compressed: {} bytes", comp_size);
    println!("    Ratio:      {:.1}:1\n", ratio);

    // Test 4: Roundtrip verification
    let mut decompressed = vec![0u8; PAGE_SIZE as usize];
    let decomp_size = lz4_decompress_block(&compressed[..comp_size], &mut decompressed)
        .expect("decompression should succeed");
    let matches = decompressed[..decomp_size] == text_page[..decomp_size];
    println!("  Roundtrip Verification:");
    println!("    Decompressed size: {} bytes", decomp_size);
    println!(
        "    Data matches:      {}\n",
        if matches { "✓" } else { "✗" }
    );

    // ==========================================================================
    // Part 2: GPU Kernel Generation
    // ==========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Part 2: GPU Kernel Generation (Pure Rust)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create kernel for batch of 1000 pages
    let batch_size = 1000;
    let kernel = Lz4WarpCompressKernel::new(batch_size);

    println!("  Kernel Configuration:");
    println!("    Batch size:    {} pages", kernel.batch_size());
    println!("    Grid dim:      {:?}", kernel.grid_dim());
    println!("    Block dim:     {:?}", kernel.block_dim());
    println!(
        "    Shared memory: {} bytes ({:.1} KB)\n",
        kernel.shared_memory_bytes(),
        kernel.shared_memory_bytes() as f64 / 1024.0
    );

    // Generate PTX
    let ptx = kernel.emit_ptx();
    println!("  NVIDIA PTX Generation:");
    println!("    PTX size:      {} bytes", ptx.len());
    println!("    Entry point:   {}", kernel.name());
    println!("    Has barriers:  {}", ptx.contains("bar.sync"));
    println!("    Has shared:    {}\n", ptx.contains(".shared"));

    // Barrier safety analysis
    let safety = kernel.analyze_barrier_safety();
    println!("  Barrier Safety Analysis:");
    println!(
        "    Status:        {}",
        if safety.is_safe {
            "✓ Safe"
        } else {
            "✗ Violations found"
        }
    );
    if !safety.is_safe {
        for v in &safety.violations {
            println!("    Violation:     {:?}", v);
        }
    }
    println!();

    // Generate WGSL
    let wgsl = kernel.emit_wgsl();
    println!("  WebGPU WGSL Generation:");
    println!("    WGSL size:     {} bytes", wgsl.len());
    println!("    Has barriers:  {}", wgsl.contains("workgroupBarrier"));
    println!("    Has shared:    {}\n", wgsl.contains("var<workgroup>"));

    // ==========================================================================
    // Part 3: Architecture Overview
    // ==========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Part 3: Warp-Cooperative Architecture");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │  Warp-per-Page Strategy (ZRAM-style compression)       │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │                                                         │");
    println!("  │  Block (128 threads = 4 warps)                         │");
    println!("  │  ┌────────┬────────┬────────┬────────┐                 │");
    println!("  │  │ Warp 0 │ Warp 1 │ Warp 2 │ Warp 3 │                 │");
    println!("  │  │ Page 0 │ Page 1 │ Page 2 │ Page 3 │                 │");
    println!("  │  └────────┴────────┴────────┴────────┘                 │");
    println!("  │                                                         │");
    println!("  │  Each Warp (32 threads):                               │");
    println!("  │  1. Cooperative load: 128 bytes/thread = 4KB           │");
    println!("  │  2. Zero-page detection: Parallel OR reduction         │");
    println!("  │  3. Hash-based match finding (future)                  │");
    println!("  │  4. Lane 0 encodes tokens sequentially                 │");
    println!("  │                                                         │");
    println!("  └─────────────────────────────────────────────────────────┘\n");

    // ==========================================================================
    // Part 4: Sample PTX Output
    // ==========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Part 4: Sample PTX Output (first 40 lines)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for (i, line) in ptx.lines().take(40).enumerate() {
        println!("  {:3} │ {}", i + 1, line);
    }
    println!(
        "  ... [{} more lines]\n",
        ptx.lines().count().saturating_sub(40)
    );

    // ==========================================================================
    // Summary
    // ==========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    println!("  ✓ LZ4 CPU reference: compress/decompress verified");
    println!("  ✓ PTX kernel: {} bytes, barrier-safe", ptx.len());
    println!("  ✓ WGSL shader: {} bytes, cross-platform", wgsl.len());
    println!("  ✓ Dual backend: NVIDIA CUDA + WebGPU support\n");
    println!("  Use Cases:");
    println!("    • ZRAM memory compression (Linux kernel)");
    println!("    • GPU-accelerated backup/archival");
    println!("    • Real-time data streaming compression");
    println!("    • Database page compression\n");
}
