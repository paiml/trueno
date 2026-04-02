#![allow(clippy::disallowed_methods)]
//! Head-to-head comparison: trueno vs ndarray across ALL operations
//!
//! Run: cargo bench --bench gemm_comparison
//! Run parallel: cargo bench --bench gemm_comparison --features parallel

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

fn gen_data(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7 + 3) % 100) as f32 / 100.0)
        .collect()
}

// ── GEMM ──────────────────────────────────────────────────────────────

fn bench_gemm(c: &mut Criterion) {
    let sizes = [64, 128, 256, 512, 1024];
    let mut group = c.benchmark_group("gemm");
    group.sample_size(20);

    for &n in &sizes {
        let a_data = gen_data(n * n);
        let b_data = gen_data(n * n);

        group.bench_with_input(BenchmarkId::new("trueno", n), &n, |bench, &n| {
            let mut c_data = vec![0.0f32; n * n];
            bench.iter(|| {
                trueno::blis::gemm(n, n, n, black_box(&a_data), black_box(&b_data), black_box(&mut c_data)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |bench, &n| {
            let a = ndarray::Array2::from_shape_vec((n, n), a_data.clone()).unwrap();
            let b = ndarray::Array2::from_shape_vec((n, n), b_data.clone()).unwrap();
            bench.iter(|| black_box(black_box(&a).dot(black_box(&b))));
        });
    }
    group.finish();
}

// ── GEMV (matrix-vector) ──────────────────────────────────────────────

fn bench_gemv(c: &mut Criterion) {
    let sizes = [64, 128, 256, 512, 1024];
    let mut group = c.benchmark_group("gemv");
    group.sample_size(50);

    for &n in &sizes {
        let mat_data = gen_data(n * n);
        let vec_data = gen_data(n);

        group.bench_with_input(BenchmarkId::new("trueno", n), &n, |bench, &n| {
            let mut out = vec![0.0f32; n];
            bench.iter(|| {
                trueno::blis::gemv::gemv(n, n, black_box(&mat_data), black_box(&vec_data), black_box(&mut out));
            });
        });

        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |bench, &n| {
            let a = ndarray::Array2::from_shape_vec((n, n), mat_data.clone()).unwrap();
            let x = ndarray::Array1::from_vec(vec_data.clone());
            bench.iter(|| black_box(black_box(&a).dot(black_box(&x))));
        });
    }
    group.finish();
}

// ── Vector add (elementwise) ──────────────────────────────────────────

fn bench_add(c: &mut Criterion) {
    let sizes = [1000, 10_000, 100_000, 1_000_000];
    let mut group = c.benchmark_group("vec_add");
    group.sample_size(50);

    for &n in &sizes {
        let a_data = gen_data(n);
        let b_data = gen_data(n);

        group.bench_with_input(BenchmarkId::new("trueno", n), &n, |bench, &n| {
            let mut out = vec![0.0f32; n];
            bench.iter(|| {
                trueno::blis::elementwise::add(
                    black_box(&a_data), black_box(&b_data), black_box(&mut out),
                ).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |bench, &n| {
            let a = ndarray::Array1::from_vec(a_data.clone());
            let b = ndarray::Array1::from_vec(b_data.clone());
            bench.iter(|| black_box(black_box(&a) + black_box(&b)));
        });
    }
    group.finish();
}

// ── Softmax ───────────────────────────────────────────────────────────

fn bench_softmax(c: &mut Criterion) {
    let sizes = [128, 1024, 4096, 32768];
    let mut group = c.benchmark_group("softmax");
    group.sample_size(50);

    for &n in &sizes {
        let data = gen_data(n);

        group.bench_with_input(BenchmarkId::new("trueno", n), &n, |bench, &_n| {
            bench.iter(|| {
                black_box(trueno::blis::softmax::softmax_1d_alloc(black_box(&data)))
            });
        });

        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |bench, &_n| {
            let a = ndarray::Array1::from_vec(data.clone());
            bench.iter(|| {
                let max = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exp: ndarray::Array1<f32> = a.mapv(|x| (x - max).exp());
                let sum = exp.sum();
                black_box(exp / sum)
            });
        });
    }
    group.finish();
}

// ── Transpose ─────────────────────────────────────────────────────────

fn bench_transpose(c: &mut Criterion) {
    let sizes = [64, 128, 256, 512];
    let mut group = c.benchmark_group("transpose");
    group.sample_size(50);

    for &n in &sizes {
        let data = gen_data(n * n);

        group.bench_with_input(BenchmarkId::new("trueno", n), &n, |bench, &n| {
            let mut out = vec![0.0f32; n * n];
            bench.iter(|| {
                trueno::blis::transpose(n, n, black_box(&data), black_box(&mut out)).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |bench, &n| {
            let a = ndarray::Array2::from_shape_vec((n, n), data.clone()).unwrap();
            bench.iter(|| black_box(black_box(&a).t().to_owned()));
        });
    }
    group.finish();
}

// ── ReLU (elementwise activation) ─────────────────────────────────────

fn bench_relu(c: &mut Criterion) {
    let sizes = [1000, 10_000, 100_000, 1_000_000];
    let mut group = c.benchmark_group("relu");
    group.sample_size(50);

    for &n in &sizes {
        let data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) / 100.0).collect();

        group.bench_with_input(BenchmarkId::new("trueno", n), &n, |bench, &n| {
            let mut out = vec![0.0f32; n];
            bench.iter(|| {
                trueno::blis::elementwise::relu(
                    black_box(&data), black_box(&mut out),
                ).unwrap();
            });
        });

        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |bench, &_n| {
            let a = ndarray::Array1::from_vec(data.clone());
            bench.iter(|| black_box(black_box(&a).mapv(|x| x.max(0.0))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_gemm, bench_gemv, bench_add, bench_softmax, bench_transpose, bench_relu);
criterion_main!(benches);
