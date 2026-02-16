//! GPU matrix multiplication and softmax benchmarks

use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use trueno::backends::gpu::GpuBackend;

/// Generate centered benchmark data: values in `[-size*0.005 .. +size*0.005]`.
fn centered_data(size: usize) -> Vec<f32> {
    let offset = size as f32 * 0.005;
    (0..size).map(|i| i as f32 * 0.01 - offset).collect()
}

/// Generate matrix data for matmul benchmarks.
fn matrix_data(size: usize) -> Vec<f32> {
    (0..(size * size)).map(|i| i as f32 * 0.5).collect()
}

/// Scalar numerically-stable softmax reference implementation.
fn scalar_softmax(data: &[f32]) -> Vec<f32> {
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = data.iter().map(|&x| (x - max_val).exp()).collect();
    let sum_exp: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|&e| e / sum_exp).collect()
}

/// Scalar log-softmax reference implementation.
fn scalar_log_softmax(data: &[f32]) -> Vec<f32> {
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = data.iter().map(|&x| (x - max_val).exp()).collect();
    let log_sum_exp = exp_vals.iter().sum::<f32>().ln();
    data.iter().map(|&x| x - max_val - log_sum_exp).collect()
}

/// Benchmark GPU matrix multiplication vs scalar baseline
pub fn bench_gpu_matmul(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_matmul");

    for size in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements((size * size * size) as u64));

        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data = matrix_data(size);
            let mut gpu = GpuBackend::new();
            bencher.iter(|| black_box(gpu.matmul(&data, &data, size, size, size).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = matrix_data(size);
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

/// Benchmark GPU softmax activation vs scalar baseline
pub fn bench_gpu_softmax(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_softmax");

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data = centered_data(size);
            let mut gpu = GpuBackend::new();
            bencher.iter(|| black_box(gpu.softmax(&data).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = centered_data(size);
            bencher.iter(|| black_box(scalar_softmax(&data)));
        });
    }

    group.finish();
}

/// Benchmark GPU log_softmax activation vs scalar baseline
pub fn bench_gpu_log_softmax(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_log_softmax");

    for size in [10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data = centered_data(size);
            let mut gpu = GpuBackend::new();
            bencher.iter(|| black_box(gpu.log_softmax(&data).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = centered_data(size);
            bencher.iter(|| black_box(scalar_log_softmax(&data)));
        });
    }

    group.finish();
}
