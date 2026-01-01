//! Kernel Generation Benchmark
//!
//! Run: `cargo run -p trueno-gpu --release --example bench_kernel_gen`

use std::time::Instant;
use trueno_gpu::kernels::{
    AttentionKernel, GemmKernel, Kernel, LayerNormKernel, QuantizeKernel, SoftmaxKernel,
};

fn main() {
    let iterations = 1000;

    println!("Kernel Generation Performance ({iterations} iterations each)");
    println!("═══════════════════════════════════════════════════════════════");

    // Warm up
    let _ = GemmKernel::naive(32, 32, 32).emit_ptx();

    let tests: Vec<(&str, Box<dyn Fn() -> String>)> = vec![
        (
            "gemm_naive_64",
            Box::new(|| GemmKernel::naive(64, 64, 64).emit_ptx()),
        ),
        (
            "gemm_tiled_128",
            Box::new(|| GemmKernel::tiled(128, 128, 128, 32).emit_ptx()),
        ),
        (
            "gemm_tensor_core",
            Box::new(|| GemmKernel::tensor_core(64, 64, 64).emit_ptx()),
        ),
        (
            "gemm_wmma_fp16",
            Box::new(|| GemmKernel::wmma_fp16(64, 64, 64).emit_ptx()),
        ),
        (
            "softmax_1024",
            Box::new(|| SoftmaxKernel::new(1024).emit_ptx()),
        ),
        (
            "layernorm_1024",
            Box::new(|| LayerNormKernel::new(1024).emit_ptx()),
        ),
        (
            "attention_64_64",
            Box::new(|| AttentionKernel::new(64, 64).emit_ptx()),
        ),
        (
            "q4k_32",
            Box::new(|| QuantizeKernel::ggml(32, 32, 256).emit_ptx()),
        ),
    ];

    for (name, gen_fn) in &tests {
        let start = Instant::now();
        for _ in 0..iterations {
            std::hint::black_box(gen_fn());
        }
        let elapsed = start.elapsed();
        let per_iter_ns = elapsed.as_nanos() as f64 / iterations as f64;
        let ptx = gen_fn();
        println!(
            "{:20} {:8.2} us  ({} bytes, {} lines)",
            name,
            per_iter_ns / 1000.0,
            ptx.len(),
            ptx.lines().count()
        );
    }

    println!("\n═══════════════════════════════════════════════════════════════");

    // Throughput test
    let start = Instant::now();
    let heavy_iterations = 10000;
    for _ in 0..heavy_iterations {
        std::hint::black_box(GemmKernel::tiled(64, 64, 64, 16).emit_ptx());
    }
    let elapsed = start.elapsed();
    let kernels_per_sec = heavy_iterations as f64 / elapsed.as_secs_f64();
    println!("Throughput: {:.0} kernels/sec (gemm_tiled_64)", kernels_per_sec);
}
