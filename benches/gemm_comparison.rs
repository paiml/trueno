#![allow(clippy::disallowed_methods)]
//! Head-to-head GEMM comparison: trueno BLIS vs ndarray
//!
//! Run: cargo bench --bench gemm_comparison

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

fn gen_data(len: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0).collect()
}

fn bench_gemm_comparison(c: &mut Criterion) {
    let sizes = [64, 128, 256, 512, 1024];

    let mut group = c.benchmark_group("gemm_comparison");
    group.sample_size(30);

    for &n in &sizes {
        let a_data = gen_data(n * n);
        let b_data = gen_data(n * n);

        // trueno BLIS
        group.bench_with_input(BenchmarkId::new("trueno", n), &n, |bench, &n| {
            let mut c_data = vec![0.0f32; n * n];
            bench.iter(|| {
                trueno::blis::gemm(
                    n,
                    n,
                    n,
                    black_box(&a_data),
                    black_box(&b_data),
                    black_box(&mut c_data),
                )
                .unwrap();
            });
        });

        // ndarray (matrixmultiply backend)
        group.bench_with_input(BenchmarkId::new("ndarray", n), &n, |bench, &n| {
            let a = ndarray::Array2::from_shape_vec((n, n), a_data.clone()).unwrap();
            let b = ndarray::Array2::from_shape_vec((n, n), b_data.clone()).unwrap();
            bench.iter(|| {
                black_box(black_box(&a).dot(black_box(&b)));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_gemm_comparison);
criterion_main!(benches);
