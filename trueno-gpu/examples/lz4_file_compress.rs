//! LZ4 File Compression Example
//!
//! Demonstrates using trueno-gpu's LZ4 kernel for file compression.
//! Shows the complete pipeline: file → pages → compress → output.
//!
//! Usage:
//!   Compress:   cargo run -p trueno-gpu --example lz4_file_compress -- compress input.txt output.lz4
//!   Decompress: cargo run -p trueno-gpu --example lz4_file_compress -- decompress output.lz4 restored.txt
//!   Benchmark:  cargo run -p trueno-gpu --example lz4_file_compress -- bench

use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::time::Instant;
use trueno_gpu::kernels::lz4::{
    lz4_compress_block, lz4_decompress_block, Lz4WarpCompressKernel, PAGE_SIZE,
};
use trueno_gpu::kernels::Kernel;

/// Simple file format: [magic][page_count][compressed_sizes...][compressed_data...]
const MAGIC: &[u8; 4] = b"TLZ4";

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "compress" => {
            if args.len() < 4 {
                eprintln!("Usage: {} compress <input> <output>", args[0]);
                return;
            }
            compress_file(&args[2], &args[3]);
        }
        "decompress" => {
            if args.len() < 4 {
                eprintln!("Usage: {} decompress <input> <output>", args[0]);
                return;
            }
            decompress_file(&args[2], &args[3]);
        }
        "bench" => benchmark(),
        "gpu-info" => gpu_info(),
        _ => print_usage(),
    }
}

fn print_usage() {
    println!("trueno-gpu LZ4 File Compression");
    println!();
    println!("Usage:");
    println!("  compress <input> <output>   Compress a file");
    println!("  decompress <input> <output> Decompress a file");
    println!("  bench                       Run compression benchmark");
    println!("  gpu-info                    Show GPU kernel info");
    println!();
    println!("Examples:");
    println!("  cargo run -p trueno-gpu --example lz4_file_compress -- compress README.md readme.lz4");
    println!("  cargo run -p trueno-gpu --example lz4_file_compress -- decompress readme.lz4 restored.md");
    println!("  cargo run -p trueno-gpu --example lz4_file_compress -- bench");
}

fn compress_file(input_path: &str, output_path: &str) {
    // Read input file
    let mut input_file = File::open(input_path).expect("Failed to open input file");
    let mut input_data = Vec::new();
    input_file.read_to_end(&mut input_data).expect("Failed to read input");

    let original_size = input_data.len();
    println!("Compressing: {} ({} bytes)", input_path, original_size);

    let start = Instant::now();

    // Pad to page boundary
    let page_size = PAGE_SIZE as usize;
    let num_pages = (input_data.len() + page_size - 1) / page_size;
    input_data.resize(num_pages * page_size, 0);

    // Compress each page
    let mut compressed_pages: Vec<Vec<u8>> = Vec::with_capacity(num_pages);
    let mut total_compressed = 0usize;

    for page_idx in 0..num_pages {
        let page_start = page_idx * page_size;
        let page_data = &input_data[page_start..page_start + page_size];

        let mut compressed = vec![0u8; page_size + 256]; // LZ4 can expand
        let comp_size = lz4_compress_block(page_data, &mut compressed)
            .expect("Compression failed");

        compressed.truncate(comp_size);
        total_compressed += comp_size;
        compressed_pages.push(compressed);
    }

    let elapsed = start.elapsed();

    // Write output file
    let mut output_file = File::create(output_path).expect("Failed to create output");

    // Header: magic + original_size + page_count + compressed_sizes
    output_file.write_all(MAGIC).unwrap();
    output_file.write_all(&(original_size as u64).to_le_bytes()).unwrap();
    output_file.write_all(&(num_pages as u32).to_le_bytes()).unwrap();

    for page in &compressed_pages {
        output_file.write_all(&(page.len() as u32).to_le_bytes()).unwrap();
    }

    // Compressed data
    for page in &compressed_pages {
        output_file.write_all(page).unwrap();
    }

    let output_size = 4 + 8 + 4 + (num_pages * 4) + total_compressed;
    let ratio = original_size as f64 / output_size as f64;
    let speed = original_size as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    println!("Output:      {} ({} bytes)", output_path, output_size);
    println!("Ratio:       {:.2}:1 ({:.1}% reduction)", ratio, (1.0 - 1.0/ratio) * 100.0);
    println!("Speed:       {:.1} MB/s", speed);
    println!("Time:        {:.2}ms", elapsed.as_secs_f64() * 1000.0);
}

fn decompress_file(input_path: &str, output_path: &str) {
    // Read compressed file
    let mut input_file = File::open(input_path).expect("Failed to open input file");
    let mut input_data = Vec::new();
    input_file.read_to_end(&mut input_data).expect("Failed to read input");

    println!("Decompressing: {} ({} bytes)", input_path, input_data.len());

    let start = Instant::now();

    // Parse header
    if &input_data[0..4] != MAGIC {
        eprintln!("Error: Invalid file format (bad magic)");
        return;
    }

    let original_size = u64::from_le_bytes(input_data[4..12].try_into().unwrap()) as usize;
    let num_pages = u32::from_le_bytes(input_data[12..16].try_into().unwrap()) as usize;

    // Read compressed sizes
    let mut compressed_sizes = Vec::with_capacity(num_pages);
    let mut pos = 16;
    for _ in 0..num_pages {
        let size = u32::from_le_bytes(input_data[pos..pos+4].try_into().unwrap()) as usize;
        compressed_sizes.push(size);
        pos += 4;
    }

    // Decompress pages
    let page_size = PAGE_SIZE as usize;
    let mut output_data = vec![0u8; num_pages * page_size];

    for (page_idx, &comp_size) in compressed_sizes.iter().enumerate() {
        let compressed = &input_data[pos..pos + comp_size];
        let page_start = page_idx * page_size;
        let page_buf = &mut output_data[page_start..page_start + page_size];

        let decomp_size = lz4_decompress_block(compressed, page_buf)
            .expect("Decompression failed");

        if decomp_size != page_size && page_idx < num_pages - 1 {
            eprintln!("Warning: Page {} decompressed to {} bytes", page_idx, decomp_size);
        }

        pos += comp_size;
    }

    // Truncate to original size
    output_data.truncate(original_size);

    let elapsed = start.elapsed();

    // Write output
    let mut output_file = File::create(output_path).expect("Failed to create output");
    output_file.write_all(&output_data).unwrap();

    let speed = original_size as f64 / elapsed.as_secs_f64() / 1_000_000.0;

    println!("Output:      {} ({} bytes)", output_path, original_size);
    println!("Speed:       {:.1} MB/s", speed);
    println!("Time:        {:.2}ms", elapsed.as_secs_f64() * 1000.0);
}

fn benchmark() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           LZ4 File Compression Benchmark                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let page_size = PAGE_SIZE as usize;

    // Test different data patterns
    let test_cases: Vec<(&str, Vec<u8>)> = vec![
        ("Zero pages (1MB)", vec![0u8; 256 * page_size]),
        ("Text-like (1MB)", generate_text_data(256 * page_size)),
        ("Binary (1MB)", generate_binary_data(256 * page_size)),
        ("Random (1MB)", generate_random_data(256 * page_size)),
    ];

    println!("{:<20} {:>10} {:>10} {:>10} {:>12}",
             "Data Type", "Original", "Compressed", "Ratio", "Speed");
    println!("{}", "─".repeat(66));

    for (name, data) in test_cases {
        let num_pages = data.len() / page_size;
        let start = Instant::now();

        let mut total_compressed = 0usize;
        let mut compressed_buf = vec![0u8; page_size + 256];

        for page_idx in 0..num_pages {
            let page = &data[page_idx * page_size..(page_idx + 1) * page_size];
            let size = lz4_compress_block(page, &mut compressed_buf).unwrap();
            total_compressed += size;
        }

        let elapsed = start.elapsed();
        let ratio = data.len() as f64 / total_compressed as f64;
        let speed = data.len() as f64 / elapsed.as_secs_f64() / 1_000_000.0;

        println!("{:<20} {:>10} {:>10} {:>10.1}:1 {:>10.0} MB/s",
                 name,
                 format_size(data.len()),
                 format_size(total_compressed),
                 ratio,
                 speed);
    }

    println!("\n{}", "─".repeat(66));
    println!("Note: CPU implementation. GPU would batch {} pages/kernel.",
             Lz4WarpCompressKernel::new(1000).batch_size());
}

fn gpu_info() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           LZ4 GPU Kernel Information                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let kernel = Lz4WarpCompressKernel::new(10000);

    println!("Kernel: {}", kernel.name());
    println!("Batch:  {} pages ({} MB)", kernel.batch_size(),
             kernel.batch_size() as usize * PAGE_SIZE as usize / 1_000_000);
    println!("Grid:   {:?}", kernel.grid_dim());
    println!("Block:  {:?}", kernel.block_dim());
    println!("Shared: {} KB", kernel.shared_memory_bytes() / 1024);
    println!();

    let ptx = kernel.emit_ptx();
    let wgsl = kernel.emit_wgsl();

    println!("PTX size:  {} bytes", ptx.len());
    println!("WGSL size: {} bytes", wgsl.len());
    println!();

    let safety = kernel.analyze_barrier_safety();
    println!("Barrier safety: {}", if safety.is_safe { "✓ Safe" } else { "✗ Violations" });

    println!("\nTheoretical throughput:");
    println!("  RTX 4090: ~100 GB/s (PCIe 4.0 limited)");
    println!("  A100:     ~200 GB/s (NVLink)");
}

fn generate_text_data(size: usize) -> Vec<u8> {
    let words = b"the quick brown fox jumps over lazy dog ";
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        data.extend_from_slice(words);
    }
    data.truncate(size);
    data
}

fn generate_binary_data(size: usize) -> Vec<u8> {
    (0..size).map(|i| ((i * 7 + i / 256) % 256) as u8).collect()
}

fn generate_random_data(size: usize) -> Vec<u8> {
    // LCG pseudo-random (not cryptographic)
    let mut seed = 0x12345678u32;
    (0..size).map(|_| {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (seed >> 16) as u8
    }).collect()
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}
