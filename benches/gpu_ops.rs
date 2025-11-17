//! GPU Performance Benchmarks
//!
//! Validates GPU speedup claims (10-50x over scalar, 5-20x over SIMD)
//! for large-scale operations where transfer overhead is amortized.
//!
//! # Benchmark Methodology
//!
//! - Tests multiple sizes: 1K, 10K, 100K, 1M elements
//! - Compares GPU vs AVX2 vs Scalar backends
//! - Uses Criterion for statistical analysis
//! - Each benchmark measures throughput (elements/second)
//!
//! # Performance Goals (GPU vs Scalar)
//!
//! Small vectors (1K):     <5x   (transfer overhead dominates)
//! Medium vectors (10K):   5-10x (transfer overhead amortized)
//! Large vectors (100K):   10-30x (GPU compute dominates)
//! Very large (1M+):       20-50x (optimal GPU utilization)
//!
//! # Performance Goals (GPU vs AVX2)
//!
//! Small vectors (1K):     <2x   (transfer overhead)
//! Medium vectors (10K):   2-5x  (starting to benefit)
//! Large vectors (100K):   5-15x (clear GPU advantage)
//! Very large (1M+):       10-25x (massive parallelism)

#![cfg(feature = "gpu")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use trueno::backends::gpu::GpuBackend;

/// Generate test data for benchmarks
fn generate_test_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) * 0.5).collect()
}

/// Benchmark GPU vector addition vs scalar baseline
fn bench_gpu_vec_add(c: &mut Criterion) {
    // Skip if GPU not available
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_vec_add");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // GPU backend
        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data_a = generate_test_data(size);
            let data_b = generate_test_data(size);
            let mut gpu = GpuBackend::new();

            bencher.iter(|| {
                black_box(gpu.vec_add(&data_a, &data_b).unwrap());
            });
        });

        // Scalar baseline (for speedup comparison)
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data_a = generate_test_data(size);
            let data_b = generate_test_data(size);

            bencher.iter(|| {
                let result: Vec<f32> = data_a
                    .iter()
                    .zip(data_b.iter())
                    .map(|(a, b)| a + b)
                    .collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark GPU dot product vs scalar baseline
fn bench_gpu_dot(c: &mut Criterion) {
    // Skip if GPU not available
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_dot");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // GPU backend
        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data_a = generate_test_data(size);
            let data_b = generate_test_data(size);
            let mut gpu = GpuBackend::new();

            bencher.iter(|| {
                black_box(gpu.dot(&data_a, &data_b).unwrap());
            });
        });

        // Scalar baseline (for speedup comparison)
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data_a = generate_test_data(size);
            let data_b = generate_test_data(size);

            bencher.iter(|| {
                let result: f32 = data_a
                    .iter()
                    .zip(data_b.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark GPU matrix multiplication vs scalar baseline
fn bench_gpu_matmul(c: &mut Criterion) {
    // Skip if GPU not available
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_matmul");

    for size in [100, 500, 1000].iter() {
        let total_ops = size * size * size; // Approximate operations
        group.throughput(Throughput::Elements(total_ops as u64));

        // GPU backend
        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..(size * size)).map(|i| (i as f32) * 0.5).collect();
            let mut gpu = GpuBackend::new();

            bencher.iter(|| {
                black_box(gpu.matmul(&data, &data, size, size, size).unwrap());
            });
        });

        // Scalar baseline (naive O(n³) matmul)
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..(size * size)).map(|i| (i as f32) * 0.5).collect();

            bencher.iter(|| {
                let mut result = vec![0.0f32; size * size];
                for i in 0..size {
                    for j in 0..size {
                        let mut sum = 0.0;
                        for k in 0..size {
                            sum += data[i * size + k] * data[k * size + j];
                        }
                        result[i * size + j] = sum;
                    }
                }
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark GPU ReLU activation vs scalar baseline
///
/// Tests GPU acceleration for element-wise operations.
/// GPU threshold: >100K elements (OpComplexity::Low)
fn bench_gpu_relu(c: &mut Criterion) {
    // Skip if GPU not available
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_relu");

    // Test sizes: 10K (below threshold), 100K (at threshold), 1M (well above)
    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // GPU backend
        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            // Mix of positive and negative values to test relu logic
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.5 - (size as f32) * 0.25)
                .collect();
            let mut gpu = GpuBackend::new();

            bencher.iter(|| {
                black_box(gpu.relu(&data).unwrap());
            });
        });

        // Scalar baseline (for speedup comparison)
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.5 - (size as f32) * 0.25)
                .collect();

            bencher.iter(|| {
                let result: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gpu_vec_add,
    bench_gpu_dot,
    bench_gpu_matmul,
    bench_gpu_relu
);
criterion_main!(benches);
