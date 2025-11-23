use std::time::Instant;
use trueno::{Matrix, Vector};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Trueno Matrix Operations Performance Benchmark Suite        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Test sizes
    let sizes = vec![(256, 256), (512, 512), (1024, 1024), (2048, 2048)];

    for (rows, cols) in sizes {
        println!("\n═══════════════════════════════════════════════════════════════════");
        println!("  Matrix Size: {}×{}", rows, cols);
        println!("═══════════════════════════════════════════════════════════════════\n");

        // Create test data
        let matrix_a = Matrix::from_vec(
            rows,
            cols,
            (0..rows * cols)
                .map(|i| ((i % 100) as f32) / 10.0)
                .collect(),
        )
        .expect("Failed to create matrix A");

        let matrix_b = Matrix::from_vec(
            rows,
            cols,
            (0..rows * cols)
                .map(|i| (((i * 7) % 100) as f32) / 10.0)
                .collect(),
        )
        .expect("Failed to create matrix B");

        let vector_data: Vec<f32> = (0..cols).map(|i| ((i % 50) as f32) / 5.0).collect();
        let vector = Vector::from_slice(&vector_data);

        // === Matrix Multiplication Benchmark ===
        if rows <= 1024 {
            // Skip very large for speed
            print!("  Matrix Multiplication ({}×{}×{})... ", rows, cols, cols);

            // Warmup
            for _ in 0..2 {
                let _ = matrix_a.matmul(&matrix_b).expect("Warmup matmul failed");
            }

            let iterations = if rows <= 512 { 10 } else { 5 };
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = matrix_a.matmul(&matrix_b).expect("Benchmark matmul failed");
            }
            let elapsed = start.elapsed();
            let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
            let gflops = (2.0 * rows as f64 * cols as f64 * cols as f64) / (avg_ms * 1e6);

            println!("{:>8.2} ms  ({:.2} GFLOPS)", avg_ms, gflops);
        }

        // === Matrix-Vector Multiplication Benchmark ===
        print!("  Matrix-Vector ({}×{} × {})... ", rows, cols, cols);

        // Warmup
        for _ in 0..3 {
            let _ = matrix_a.matvec(&vector).expect("Warmup matvec failed");
        }

        let iterations = if rows <= 512 { 100 } else { 20 };
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = matrix_a.matvec(&vector).expect("Benchmark matvec failed");
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / iterations as f64;
        let gflops = (2.0 * rows as f64 * cols as f64) / (avg_us * 1000.0);

        println!("{:>8.2} µs  ({:.2} GFLOPS)", avg_us, gflops);

        // === Vector-Matrix Multiplication Benchmark ===
        print!("  Vector-Matrix ({} × {}×{})... ", rows, rows, cols);

        // Warmup
        for _ in 0..3 {
            let _ = Matrix::vecmat(&vector, &matrix_a).expect("Warmup vecmat failed");
        }

        let iterations = if rows <= 512 { 100 } else { 20 };
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = Matrix::vecmat(&vector, &matrix_a).expect("Benchmark vecmat failed");
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / iterations as f64;
        let gflops = (2.0 * rows as f64 * cols as f64) / (avg_us * 1000.0);

        println!("{:>8.2} µs  ({:.2} GFLOPS)", avg_us, gflops);

        // === Transpose Benchmark ===
        print!("  Transpose ({}×{})... ", rows, cols);

        // Warmup
        for _ in 0..3 {
            let _ = matrix_a.transpose();
        }

        let iterations = if rows <= 512 { 100 } else { 20 };
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = matrix_a.transpose();
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / iterations as f64;
        let bandwidth_gb = (rows as f64 * cols as f64 * 8.0) / (avg_us * 1000.0); // Read + Write

        println!("{:>8.2} µs  ({:.2} GB/s bandwidth)", avg_us, bandwidth_gb);
    }

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  Optimizations Applied:");
    println!("  • Matrix Multiplication: SIMD + 3-level cache blocking + parallel");
    println!("  • Matrix-Vector: SIMD dot products (AVX2/SSE2)");
    println!("  • Vector-Matrix: SIMD row accumulation (cache-friendly)");
    println!("  • Transpose: Cache-blocked (64×64 blocks)");
    println!("═══════════════════════════════════════════════════════════════════\n");

    #[cfg(feature = "parallel")]
    println!("  [Parallel feature: ENABLED for matrices ≥1024×1024]\n");
    #[cfg(not(feature = "parallel"))]
    println!("  [Parallel feature: DISABLED]\n");
}
