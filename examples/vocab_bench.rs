//! Benchmark vector-matrix multiply (vocab projection pattern)
use std::time::Instant;
use trueno::{Matrix, Vector};

fn main() {
    // Whisper vocab projection: 1×384 @ 384×51865
    let rows = 1;
    let inner = 384;
    let cols = 51865;
    
    println!("Benchmarking vocab projection pattern: {rows}×{inner} @ {inner}×{cols}");
    println!("Total ops: {} million", (rows * inner * cols) as f64 / 1e6);
    
    // Create test data
    let a: Vec<f32> = (0..rows * inner).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..inner * cols).map(|i| (i as f32) * 0.0001).collect();
    
    // Method 1: Full matmul via Matrix
    let ma = Matrix::from_vec(rows, inner, a.clone()).unwrap();
    let mb = Matrix::from_vec(inner, cols, b.clone()).unwrap();
    
    // Warmup
    for _ in 0..3 {
        let _ = ma.matmul(&mb);
    }
    
    let iterations = 10;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ma.matmul(&mb).unwrap();
    }
    let matmul_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let matmul_gflops = (2.0 * rows as f64 * inner as f64 * cols as f64) / (matmul_time / 1000.0) / 1e9;
    println!("\nMatrix::matmul: {:.1}ms ({:.2} GFLOPS)", matmul_time, matmul_gflops);
    
    // Method 2: Transpose B to column-major, then use dots
    let mut b_t = vec![0.0_f32; inner * cols];
    for i in 0..inner {
        for j in 0..cols {
            b_t[j * inner + i] = b[i * cols + j];
        }
    }
    
    let va = Vector::from_slice(&a);
    
    // Warmup
    for _ in 0..3 {
        let mut result = vec![0.0_f32; cols];
        for (j, result_elem) in result.iter_mut().enumerate() {
            let col_start = j * inner;
            let vb = Vector::from_slice(&b_t[col_start..col_start + inner]);
            *result_elem = va.dot(&vb).unwrap();
        }
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let mut result = vec![0.0_f32; cols];
        for (j, result_elem) in result.iter_mut().enumerate() {
            let col_start = j * inner;
            let vb = Vector::from_slice(&b_t[col_start..col_start + inner]);
            *result_elem = va.dot(&vb).unwrap();
        }
    }
    let transposed_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let transposed_gflops = (2.0 * rows as f64 * inner as f64 * cols as f64) / (transposed_time / 1000.0) / 1e9;
    println!("Vector dots (transposed): {:.1}ms ({:.2} GFLOPS)", transposed_time, transposed_gflops);
    
    // Method 3: Direct scalar (baseline)
    let start = Instant::now();
    for _ in 0..iterations {
        let mut result = vec![0.0_f32; cols];
        for j in 0..cols {
            let mut sum = 0.0_f32;
            for i in 0..inner {
                sum += a[i] * b[i * cols + j];
            }
            result[j] = sum;
        }
    }
    let scalar_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;
    let scalar_gflops = (2.0 * rows as f64 * inner as f64 * cols as f64) / (scalar_time / 1000.0) / 1e9;
    println!("Scalar (naive): {:.1}ms ({:.2} GFLOPS)", scalar_time, scalar_gflops);
    
    println!("\n=== ANALYSIS ===");
    println!("Matrix::matmul vs scalar: {:.1}x", scalar_time / matmul_time);
    println!("Transposed dots vs scalar: {:.1}x", scalar_time / transposed_time);
    println!("Potential speedup for matmul: {:.1}x", matmul_time / transposed_time.min(scalar_time));
}
