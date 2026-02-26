//! BLIS GEMM Benchmark
//!
//! Measures performance of the BLIS-style matrix multiplication implementation.
//!
//! # Running
//!
//! ```bash
//! cargo run --release --example blis_benchmark
//! ```
//!
//! # Performance Expectations
//!
//! On a modern x86_64 CPU with AVX2:
//! - 64×64: ~25 GFLOP/s
//! - 256×256: ~60 GFLOP/s
//! - 512×512: ~70 GFLOP/s
//! - 1024×1024: ~72 GFLOP/s
//!
//! # Algorithm
//!
//! Uses the BLIS framework (Van Zee & Van de Geijn, 2015):
//! - 8×6 register-blocked microkernel for AVX2
//! - Cache-optimized packing for L1/L2/L3
//! - 5-loop blocking structure
//!
//! # References
//!
//! - Goto, K., & Van de Geijn, R. A. (2008). Anatomy of High-Performance Matrix Multiplication.
//! - Van Zee, F. G., & Van de Geijn, R. A. (2015). BLIS: A Framework for BLAS.

use std::time::Instant;
use trueno::blis::{gemm_blis, gemm_reference, BlisProfiler};

fn benchmark_gemm(name: &str, n: usize, iterations: usize) {
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let mut c = vec![0.0f32; n * n];

    // Warmup
    for _ in 0..3 {
        c.fill(0.0);
        gemm_blis(n, n, n, &a, &b, &mut c, None).unwrap();
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        c.fill(0.0);
        gemm_blis(n, n, n, &a, &b, &mut c, None).unwrap();
    }
    let elapsed = start.elapsed();

    let total_flops = 2u64 * (n as u64) * (n as u64) * (n as u64) * (iterations as u64);
    let gflops = total_flops as f64 / elapsed.as_secs_f64() / 1e9;
    let time_per_op = elapsed.as_micros() as f64 / iterations as f64;

    println!("{:20} {:4}x{:4}: {:8.1} us, {:6.1} GFLOP/s", name, n, n, time_per_op, gflops);
}

fn benchmark_with_profiler(n: usize) {
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let mut c = vec![0.0f32; n * n];
    let mut profiler = BlisProfiler::enabled();

    // Run with profiling
    gemm_blis(n, n, n, &a, &b, &mut c, Some(&mut profiler)).unwrap();

    println!("\nProfiler Results for {}x{}:", n, n);
    println!("{}", profiler.summary());
}

fn compare_reference_vs_blis(n: usize) {
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();

    // Reference
    let mut c_ref = vec![0.0f32; n * n];
    let start = Instant::now();
    gemm_reference(n, n, n, &a, &b, &mut c_ref).unwrap();
    let ref_time = start.elapsed();

    // BLIS
    let mut c_blis = vec![0.0f32; n * n];
    let start = Instant::now();
    gemm_blis(n, n, n, &a, &b, &mut c_blis, None).unwrap();
    let blis_time = start.elapsed();

    // Verify correctness
    let max_diff: f32 =
        c_ref.iter().zip(c_blis.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);

    let speedup = ref_time.as_secs_f64() / blis_time.as_secs_f64();

    println!(
        "{}x{}: Reference {:8.1}ms, BLIS {:8.1}ms, Speedup: {:5.1}x, MaxDiff: {:.2e}",
        n,
        n,
        ref_time.as_secs_f64() * 1000.0,
        blis_time.as_secs_f64() * 1000.0,
        speedup,
        max_diff
    );
}

fn main() {
    println!("=== BLIS GEMM Benchmark ===\n");

    println!("--- Reference vs BLIS Comparison ---");
    for n in [64, 128, 256, 512] {
        compare_reference_vs_blis(n);
    }

    println!("\n--- BLIS Performance (multiple iterations) ---");
    benchmark_gemm("BLIS", 64, 1000);
    benchmark_gemm("BLIS", 128, 100);
    benchmark_gemm("BLIS", 256, 20);
    benchmark_gemm("BLIS", 512, 5);
    benchmark_gemm("BLIS", 1024, 2);

    println!("\n--- Detailed Profiler Output ---");
    benchmark_with_profiler(256);
}
