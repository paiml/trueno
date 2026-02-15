//! GPU activation function benchmarks (relu, leaky_relu, elu, clip, sigmoid, tanh, swish, gelu)

use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use trueno::backends::gpu::GpuBackend;

/// Benchmark GPU ReLU activation vs scalar baseline
///
/// Tests GPU acceleration for element-wise operations.
/// GPU threshold: >100K elements (OpComplexity::Low)
pub fn bench_gpu_relu(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_relu");

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.5 - (size as f32) * 0.25)
                .collect();
            let mut gpu = GpuBackend::new();

            bencher.iter(|| {
                black_box(gpu.relu(&data).unwrap());
            });
        });

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

/// Benchmark GPU leaky ReLU activation vs scalar baseline
pub fn bench_gpu_leaky_relu(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_leaky_relu");

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let negative_slope = 0.01;

        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.5 - (size as f32) * 0.25)
                .collect();
            let mut gpu = GpuBackend::new();

            bencher.iter(|| {
                black_box(gpu.leaky_relu(&data, negative_slope).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.5 - (size as f32) * 0.25)
                .collect();

            bencher.iter(|| {
                let result: Vec<f32> = data
                    .iter()
                    .map(|&x| if x > 0.0 { x } else { negative_slope * x })
                    .collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark GPU ELU activation vs scalar baseline
pub fn bench_gpu_elu(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_elu");

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let alpha = 1.0;

        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.5 - (size as f32) * 0.25)
                .collect();
            let mut gpu = GpuBackend::new();

            bencher.iter(|| {
                black_box(gpu.elu(&data, alpha).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.5 - (size as f32) * 0.25)
                .collect();

            bencher.iter(|| {
                let result: Vec<f32> = data
                    .iter()
                    .map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
                    .collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark GPU clip operation vs scalar baseline
pub fn bench_gpu_clip(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_clip");

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let min_val = 100.0;
        let max_val = 5000.0;

        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();
            let mut gpu = GpuBackend::new();

            bencher.iter(|| {
                black_box(gpu.clip(&data, min_val, max_val).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();

            bencher.iter(|| {
                let result: Vec<f32> = data.iter().map(|&x| x.max(min_val).min(max_val)).collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark GPU sigmoid activation vs scalar baseline
pub fn bench_gpu_sigmoid(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_sigmoid");

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.001 - (size as f32) * 0.0005)
                .collect();
            let mut gpu = GpuBackend::new();

            bencher.iter(|| {
                black_box(gpu.sigmoid(&data).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.001 - (size as f32) * 0.0005)
                .collect();

            bencher.iter(|| {
                let result: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark GPU tanh activation vs scalar baseline
pub fn bench_gpu_tanh(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_tanh");

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.001 - (size as f32) * 0.0005)
                .collect();
            let mut gpu = GpuBackend::new();

            bencher.iter(|| {
                black_box(gpu.tanh(&data).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.001 - (size as f32) * 0.0005)
                .collect();

            bencher.iter(|| {
                let result: Vec<f32> = data.iter().map(|&x| x.tanh()).collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark GPU swish activation vs scalar baseline
pub fn bench_gpu_swish(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_swish");

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.001 - (size as f32) * 0.0005)
                .collect();
            let mut gpu = GpuBackend::new();

            bencher.iter(|| {
                black_box(gpu.swish(&data).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.001 - (size as f32) * 0.0005)
                .collect();

            bencher.iter(|| {
                let result: Vec<f32> = data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark GPU GELU activation vs scalar baseline
pub fn bench_gpu_gelu(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_gelu");

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.001 - (size as f32) * 0.0005)
                .collect();
            let mut gpu = GpuBackend::new();

            bencher.iter(|| {
                black_box(gpu.gelu(&data).unwrap());
            });
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size)
                .map(|i| (i as f32) * 0.001 - (size as f32) * 0.0005)
                .collect();

            bencher.iter(|| {
                const SQRT_2_OVER_PI: f32 = 0.7978846;
                const COEFF: f32 = 0.044715;

                let result: Vec<f32> = data
                    .iter()
                    .map(|&x| {
                        let x_cubed = x * x * x;
                        let inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
                        0.5 * x * (1.0 + inner.tanh())
                    })
                    .collect();
                black_box(result);
            });
        });
    }

    group.finish();
}
