use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use trueno::Matrix;

fn bench_matmul_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    // Test various matrix sizes to show SIMD threshold effect
    let sizes = vec![
        (16, 16, 16),    // Small: below SIMD threshold (64)
        (32, 32, 32),    // Medium: below threshold
        (64, 64, 64),    // At threshold
        (128, 128, 128), // Large: above threshold (SIMD should shine)
        (256, 256, 256), // Very large: maximum SIMD benefit
    ];

    for (m, n, p) in sizes {
        let id = format!("{}x{}_x_{}x{}", m, n, n, p);

        // Create test matrices with non-trivial data
        let a = Matrix::from_vec(m, n, (0..m * n).map(|i| (i % 100) as f32).collect()).unwrap();
        let b =
            Matrix::from_vec(n, p, (0..n * p).map(|i| ((i * 2) % 100) as f32).collect()).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(&id),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let result = black_box(a).matmul(black_box(b)).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_matmul_rectangular(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_rectangular");

    // Common ML shapes (batch processing)
    let shapes = vec![
        (32, 128, 64),   // Small batch
        (64, 256, 128),  // Medium batch
        (128, 512, 256), // Large batch (neural network layer)
    ];

    for (m, n, p) in shapes {
        let id = format!("{}x{}_x_{}x{}", m, n, n, p);

        let a = Matrix::from_vec(m, n, (0..m * n).map(|i| (i % 100) as f32).collect()).unwrap();
        let b =
            Matrix::from_vec(n, p, (0..n * p).map(|i| ((i * 3) % 100) as f32).collect()).unwrap();

        group.bench_with_input(
            BenchmarkId::from_parameter(&id),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| {
                    let result = black_box(a).matmul(black_box(b)).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");

    let sizes = vec![
        (64, 64),
        (128, 128),
        (256, 256),
        (128, 256), // Rectangular
    ];

    for (rows, cols) in sizes {
        let id = format!("{}x{}", rows, cols);
        let m = Matrix::from_vec(
            rows,
            cols,
            (0..rows * cols).map(|i| (i % 100) as f32).collect(),
        )
        .unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(&id), &m, |bench, m| {
            bench.iter(|| {
                let result = black_box(m).transpose();
                black_box(result);
            });
        });
    }

    group.finish();
}

fn bench_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec");

    let sizes = vec![(64, 64), (128, 128), (256, 256), (512, 512)];

    for (rows, cols) in sizes {
        let id = format!("{}x{}_x_{}", rows, cols, cols);
        let m = Matrix::from_vec(
            rows,
            cols,
            (0..rows * cols).map(|i| (i % 100) as f32).collect(),
        )
        .unwrap();
        let v =
            trueno::Vector::from_slice(&(0..cols).map(|i| (i % 100) as f32).collect::<Vec<_>>());

        group.bench_with_input(
            BenchmarkId::from_parameter(&id),
            &(&m, &v),
            |bench, (m, v)| {
                bench.iter(|| {
                    let result = black_box(m).matvec(black_box(v)).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_sizes,
    bench_matmul_rectangular,
    bench_transpose,
    bench_matvec
);
criterion_main!(benches);
