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
        (512, 512, 512), // Phase 3: Large matrix baseline
        (1024, 1024, 1024), // Phase 3: Very large matrix baseline
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

fn bench_convolve2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolve2d");

    // Test various image and kernel sizes for CNN/image processing workloads
    // Format: (input_rows, input_cols, kernel_size, description)
    let configs = vec![
        (32, 32, 3, "small_3x3"),     // Small image, edge detection kernel
        (64, 64, 3, "medium_3x3"),    // Medium image, standard convolution
        (128, 128, 3, "large_3x3"),   // Large image, typical CNN layer
        (256, 256, 3, "xlarge_3x3"),  // Very large, approaching GPU threshold
        (128, 128, 5, "large_5x5"),   // Larger kernel (blur, Gaussian)
        (256, 256, 5, "xlarge_5x5"),  // Large image + large kernel
        (512, 512, 3, "xxlarge_3x3"), // Huge image (GPU threshold: >10K elements)
    ];

    for (rows, cols, kernel_size, desc) in configs {
        // Create input image with non-trivial data
        let input_data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 100) as f32) / 100.0)
            .collect();
        let input = Matrix::from_vec(rows, cols, input_data).unwrap();

        // Create kernel (e.g., averaging filter)
        let kernel_val = 1.0 / (kernel_size * kernel_size) as f32;
        let kernel = Matrix::from_vec(
            kernel_size,
            kernel_size,
            vec![kernel_val; kernel_size * kernel_size],
        )
        .unwrap();

        let id = format!("{}x{}_k{}", rows, cols, kernel_size);

        group.bench_with_input(
            BenchmarkId::new(desc, &id),
            &(&input, &kernel),
            |bench, (input, kernel)| {
                bench.iter(|| {
                    let result = black_box(input).convolve2d(black_box(kernel)).unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_convolve2d_edge_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolve2d_edge_detection");

    // Benchmark common edge detection filters (Sobel, Prewitt)
    let sizes = vec![
        (128, 128, "small"),
        (256, 256, "medium"),
        (512, 512, "large"), // GPU threshold
    ];

    for (rows, cols, desc) in sizes {
        // Create input image
        let input_data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 256) as f32) / 255.0)
            .collect();
        let input = Matrix::from_vec(rows, cols, input_data).unwrap();

        // Sobel horizontal kernel
        #[rustfmt::skip]
        let sobel_h = Matrix::from_vec(
            3,
            3,
            vec![
                -1.0, -2.0, -1.0,
                 0.0,  0.0,  0.0,
                 1.0,  2.0,  1.0,
            ],
        )
        .unwrap();

        let id = format!("sobel_{}x{}", rows, cols);

        group.bench_with_input(
            BenchmarkId::new(desc, &id),
            &(&input, &sobel_h),
            |bench, (input, kernel)| {
                bench.iter(|| {
                    let result = black_box(input).convolve2d(black_box(kernel)).unwrap();
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
    bench_matvec,
    bench_convolve2d,
    bench_convolve2d_edge_detection
);
criterion_main!(benches);
