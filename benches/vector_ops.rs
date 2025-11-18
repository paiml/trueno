//! Benchmarks for Vector operations comparing Scalar vs SSE2 vs AVX2 backends
//!
//! This benchmark suite verifies SIMD performance improvements across backends.
//!
//! # Benchmark Methodology
//!
//! - Tests multiple vector sizes: 100, 1000, 10000 elements
//! - Compares Scalar, SSE2, and AVX2 backends explicitly
//! - Uses Criterion for statistical analysis
//! - Each benchmark measures throughput (elements/second)
//!
//! # Performance Goals
//!
//! Expected SSE2 speedup over Scalar:
//! - Small vectors (100):   1.5-2x (some overhead from SIMD setup)
//! - Medium vectors (1K):   3-4x (optimal SIMD utilization)
//! - Large vectors (10K+):  3-4x (memory bandwidth limited)
//!
//! Expected AVX2 speedup over SSE2:
//! - Element-wise ops:      2x (8-wide vs 4-wide SIMD)
//! - Dot product:           2x+ (FMA acceleration)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use trueno::{Backend, Vector};

/// Generate test data for benchmarks
fn generate_test_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) * 0.5).collect()
}

/// Benchmark element-wise addition
fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);
            let b = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.add(&b).unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);
            let b = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.add(&b).unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);
            let b = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.add(&b).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark element-wise multiplication
fn bench_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);
            let b = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.mul(&b).unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);
            let b = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.mul(&b).unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);
            let b = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.mul(&b).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark dot product
fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);
            let b = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.dot(&b).unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);
            let b = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.dot(&b).unwrap());
            });
        });

        // AVX2 backend (with FMA)
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);
            let b = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.dot(&b).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark sum reduction
fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.sum().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.sum().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.sum().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark max reduction
fn bench_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("max");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.max().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.max().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.max().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark argmax (find index of maximum value)
fn bench_argmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("argmax");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.argmax().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.argmax().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.argmax().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark argmin (find index of minimum value)
fn bench_argmin(c: &mut Criterion) {
    let mut group = c.benchmark_group("argmin");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.argmin().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.argmin().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.argmin().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark ReLU activation function
fn bench_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu");

    // Test various sizes: small (100), medium (1K, 10K), large (100K, 1M - GPU candidates)
    for size in [100, 1000, 10000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Generate data with mix of positive and negative values
        let data: Vec<f32> = (0..*size)
            .map(|i| (i as f32) * 0.5 - (*size as f32) * 0.25)
            .collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.relu().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.relu().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.relu().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark softmax activation function
fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    // Softmax is more expensive (exp, sum, div) - test up to 100K
    for size in [100, 1000, 10000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.01).collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.softmax().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.softmax().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.softmax().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark log_softmax activation function
fn bench_log_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_softmax");

    for size in [100, 1000, 10000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.01).collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.log_softmax().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.log_softmax().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.log_softmax().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark clip (clamp) operation
fn bench_clip(c: &mut Criterion) {
    let mut group = c.benchmark_group("clip");

    for size in [100, 1000, 10000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.5).collect();
        let min_val = 100.0;
        let max_val = 5000.0;

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.clip(min_val, max_val).unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.clip(min_val, max_val).unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.clip(min_val, max_val).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark sigmoid activation function
fn bench_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid");

    // Sigmoid requires exp() - test up to 100K
    for size in [100, 1000, 10000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Generate data with mix of positive and negative values
        let data: Vec<f32> = (0..*size)
            .map(|i| (i as f32) * 0.1 - (*size as f32) * 0.05)
            .collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.sigmoid().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.sigmoid().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.sigmoid().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark GELU activation function
fn bench_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gelu");

    // GELU requires tanh() - test up to 100K
    for size in [100, 1000, 10000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Generate data with mix of positive and negative values
        let data: Vec<f32> = (0..*size)
            .map(|i| (i as f32) * 0.1 - (*size as f32) * 0.05)
            .collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.gelu().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.gelu().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.gelu().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark swish activation function
fn bench_swish(c: &mut Criterion) {
    let mut group = c.benchmark_group("swish");

    // Swish requires exp() for sigmoid - test up to 100K
    for size in [100, 1000, 10000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Generate data with mix of positive and negative values
        let data: Vec<f32> = (0..*size)
            .map(|i| (i as f32) * 0.1 - (*size as f32) * 0.05)
            .collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.swish().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.swish().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.swish().unwrap());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_add,
    bench_mul,
    bench_dot,
    bench_sum,
    bench_max,
    bench_argmax,
    bench_argmin,
    bench_relu,
    bench_sigmoid,
    bench_gelu,
    bench_swish,
    bench_softmax,
    bench_log_softmax,
    bench_clip
);
criterion_main!(benches);
