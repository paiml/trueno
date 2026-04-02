#![allow(clippy::disallowed_methods)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
fn gen(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0).collect()
}
fn bench_transpose(c: &mut Criterion) {
    let mut g = c.benchmark_group("transpose");
    for &n in &[64, 128, 256, 512] {
        let d = gen(n * n);
        g.bench_with_input(BenchmarkId::new("trueno", n), &n, |b, &n| {
            let mut o = vec![0.0f32; n * n];
            b.iter(|| trueno::blis::transpose(n, n, black_box(&d), black_box(&mut o)).unwrap());
        });
        g.bench_with_input(BenchmarkId::new("ndarray", n), &n, |b, &n| {
            let a = ndarray::Array2::from_shape_vec((n, n), d.clone()).unwrap();
            b.iter(|| black_box(black_box(&a).t().to_owned()));
        });
    }
    g.finish();
}
fn bench_gemm(c: &mut Criterion) {
    let mut g = c.benchmark_group("gemm");
    g.sample_size(20);
    for &n in &[64, 128, 256, 512, 1024] {
        let a = gen(n * n);
        let b = gen(n * n);
        g.bench_with_input(BenchmarkId::new("trueno", n), &n, |bench, &n| {
            let mut c = vec![0.0f32; n * n];
            bench.iter(|| {
                trueno::blis::gemm(n, n, n, black_box(&a), black_box(&b), black_box(&mut c))
                    .unwrap()
            });
        });
        g.bench_with_input(BenchmarkId::new("ndarray", n), &n, |bench, &n| {
            let a = ndarray::Array2::from_shape_vec((n, n), a.clone()).unwrap();
            let b = ndarray::Array2::from_shape_vec((n, n), b.clone()).unwrap();
            bench.iter(|| black_box(black_box(&a).dot(black_box(&b))));
        });
    }
    g.finish();
}
criterion_group!(benches, bench_transpose, bench_gemm);
criterion_main!(benches);
