use super::*;

#[test]
fn test_matmul_blocking_small_matrices() {
    // Small matrices (<=32) should use simple path (no blocking overhead)
    let sizes = vec![8, 16, 32];
    for size in sizes {
        let a = Matrix::from_vec(size, size, (0..size * size).map(|i| i as f32).collect()).unwrap();
        let b = Matrix::from_vec(
            size,
            size,
            (0..size * size).map(|i| (i * 2) as f32).collect(),
        )
        .unwrap();

        let mut result_naive = Matrix::zeros(size, size);
        let mut result_simd = Matrix::zeros(size, size);

        a.matmul_naive(&b, &mut result_naive).unwrap();
        a.matmul_simd(&b, &mut result_simd).unwrap();

        // Verify correctness
        for i in 0..size {
            for j in 0..size {
                let naive_val = result_naive.get(i, j).unwrap();
                let simd_val = result_simd.get(i, j).unwrap();
                let diff = (naive_val - simd_val).abs();
                let tolerance = if naive_val.abs() > 1.0 {
                    naive_val.abs() * 1e-4
                } else {
                    1e-4
                };
                assert!(
                    diff < tolerance,
                    "Size {}: Mismatch at ({}, {}): naive={}, simd={}, diff={}",
                    size,
                    i,
                    j,
                    naive_val,
                    simd_val,
                    diff
                );
            }
        }
    }
}

#[test]
fn test_matmul_blocking_medium_matrices() {
    // Medium matrices (>32, <512) should benefit from L2 blocking
    let sizes = vec![64, 128, 256];
    for size in sizes {
        let a = Matrix::from_vec(
            size,
            size,
            (0..size * size).map(|i| (i % 100) as f32).collect(),
        )
        .unwrap();
        let b = Matrix::from_vec(
            size,
            size,
            (0..size * size).map(|i| ((i * 3) % 100) as f32).collect(),
        )
        .unwrap();

        let mut result_naive = Matrix::zeros(size, size);
        let mut result_simd = Matrix::zeros(size, size);

        a.matmul_naive(&b, &mut result_naive).unwrap();
        a.matmul_simd(&b, &mut result_simd).unwrap();

        // Verify correctness with relative tolerance for large accumulated values
        for i in 0..size {
            for j in 0..size {
                let naive_val = result_naive.get(i, j).unwrap();
                let simd_val = result_simd.get(i, j).unwrap();
                let diff = (naive_val - simd_val).abs();
                let tolerance = if naive_val.abs() > 1.0 {
                    naive_val.abs() * 1e-3 // More relaxed for large values
                } else {
                    1e-3
                };
                assert!(
                    diff < tolerance,
                    "Size {}: Mismatch at ({}, {}): naive={}, simd={}, diff={}",
                    size,
                    i,
                    j,
                    naive_val,
                    simd_val,
                    diff
                );
            }
        }
    }
}

#[test]
fn test_matmul_blocking_non_aligned_sizes() {
    // Test matrices with sizes not aligned to block boundaries
    let test_cases = vec![
        (33, 33, 33),    // Just over small threshold
        (65, 65, 65),    // Just over L2 block size
        (100, 100, 100), // Middle of L2 block
        (127, 127, 127), // Just under 2x L2 block size
    ];

    for (m, k, n) in test_cases {
        let a = Matrix::from_vec(m, k, (0..m * k).map(|i| (i % 50) as f32).collect()).unwrap();
        let b =
            Matrix::from_vec(k, n, (0..k * n).map(|i| ((i * 2) % 50) as f32).collect()).unwrap();

        let mut result_naive = Matrix::zeros(m, n);
        let mut result_simd = Matrix::zeros(m, n);

        a.matmul_naive(&b, &mut result_naive).unwrap();
        a.matmul_simd(&b, &mut result_simd).unwrap();

        // Verify correctness
        for i in 0..m {
            for j in 0..n {
                let naive_val = result_naive.get(i, j).unwrap();
                let simd_val = result_simd.get(i, j).unwrap();
                let diff = (naive_val - simd_val).abs();
                let tolerance = if naive_val.abs() > 1.0 {
                    naive_val.abs() * 1e-3
                } else {
                    1e-3
                };
                assert!(
                    diff < tolerance,
                    "Size {}x{}x{}: Mismatch at ({}, {}): naive={}, simd={}, diff={}",
                    m,
                    k,
                    n,
                    i,
                    j,
                    naive_val,
                    simd_val,
                    diff
                );
            }
        }
    }
}

#[test]
fn test_matmul_blocking_large_matrices() {
    // Large matrix to verify blocking algorithm correctness
    // Keep size manageable for test speed but large enough to trigger blocking
    let size = 256;
    let a = Matrix::from_vec(
        size,
        size,
        (0..size * size)
            .map(|i| ((i % 100) as f32) / 10.0)
            .collect(),
    )
    .unwrap();
    let b = Matrix::from_vec(
        size,
        size,
        (0..size * size)
            .map(|i| (((i * 7) % 100) as f32) / 10.0)
            .collect(),
    )
    .unwrap();

    let mut result_naive = Matrix::zeros(size, size);
    let mut result_simd = Matrix::zeros(size, size);

    a.matmul_naive(&b, &mut result_naive).unwrap();
    a.matmul_simd(&b, &mut result_simd).unwrap();

    // Verify correctness with appropriate tolerance for accumulated floating-point errors
    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..size {
        for j in 0..size {
            let naive_val = result_naive.get(i, j).unwrap();
            let simd_val = result_simd.get(i, j).unwrap();
            let diff = (naive_val - simd_val).abs();
            let tolerance = if naive_val.abs() > 1.0 {
                naive_val.abs() * 1e-2 // Relaxed tolerance for large accumulated values
            } else {
                1e-2
            };
            if diff >= tolerance {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "Mismatch at ({}, {}): naive={}, simd={}, diff={}, tolerance={}",
                        i, j, naive_val, simd_val, diff, tolerance
                    );
                }
            }
            max_diff = max_diff.max(diff);
        }
    }
    assert_eq!(
        mismatches, 0,
        "Found {} mismatches in {}x{} matmul, max_diff={}",
        mismatches, size, size, max_diff
    );
}

#[test]
fn test_matmul_3level_blocking() {
    // Phase 3: Test 3-level cache blocking for very large matrices (>=512x512)
    // This test ensures the L3 -> L2 -> micro-kernel hierarchy works correctly
    let size = 512; // Triggers 3-level blocking (L3_THRESHOLD = 512)
    let a = Matrix::from_vec(
        size,
        size,
        (0..size * size)
            .map(|i| ((i % 100) as f32) / 10.0)
            .collect(),
    )
    .unwrap();
    let b = Matrix::from_vec(
        size,
        size,
        (0..size * size)
            .map(|i| (((i * 7) % 100) as f32) / 10.0)
            .collect(),
    )
    .unwrap();

    let mut result_naive = Matrix::zeros(size, size);
    let mut result_simd = Matrix::zeros(size, size);

    a.matmul_naive(&b, &mut result_naive).unwrap();
    a.matmul_simd(&b, &mut result_simd).unwrap();

    // Verify correctness with appropriate tolerance
    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..size {
        for j in 0..size {
            let naive_val = result_naive.get(i, j).unwrap();
            let simd_val = result_simd.get(i, j).unwrap();
            let diff = (naive_val - simd_val).abs();
            let tolerance = if naive_val.abs() > 1.0 {
                naive_val.abs() * 1e-2
            } else {
                1e-2
            };
            if diff >= tolerance {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "Mismatch at ({}, {}): naive={}, simd={}, diff={}, tolerance={}",
                        i, j, naive_val, simd_val, diff, tolerance
                    );
                }
            }
            max_diff = max_diff.max(diff);
        }
    }
    assert_eq!(
        mismatches, 0,
        "Found {} mismatches in {}x{} matmul (3-level blocking), max_diff={}",
        mismatches, size, size, max_diff
    );
}

#[test]
#[cfg(feature = "parallel")]
fn test_matmul_parallel_1024() {
    // Phase 4: Test parallel matmul for 1024x1024 matrices
    // This triggers the parallel path (PARALLEL_THRESHOLD = 1024)
    let size = 1024;
    let a = Matrix::from_vec(
        size,
        size,
        (0..size * size)
            .map(|i| ((i % 100) as f32) / 10.0)
            .collect(),
    )
    .unwrap();
    let b = Matrix::from_vec(
        size,
        size,
        (0..size * size)
            .map(|i| (((i * 7) % 100) as f32) / 10.0)
            .collect(),
    )
    .unwrap();

    let mut result_naive = Matrix::zeros(size, size);
    let mut result_parallel = Matrix::zeros(size, size);

    a.matmul_naive(&b, &mut result_naive).unwrap();
    a.matmul_simd(&b, &mut result_parallel).unwrap(); // Uses parallel path with 'parallel' feature

    // Verify correctness with appropriate tolerance
    let mut max_diff = 0.0f32;
    let mut mismatches = 0;
    for i in 0..size {
        for j in 0..size {
            let naive_val = result_naive.get(i, j).unwrap();
            let parallel_val = result_parallel.get(i, j).unwrap();
            let diff = (naive_val - parallel_val).abs();
            let tolerance = if naive_val.abs() > 1.0 {
                naive_val.abs() * 1e-2
            } else {
                1e-2
            };
            if diff >= tolerance {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!(
                        "Mismatch at ({}, {}): naive={}, parallel={}, diff={}, tolerance={}",
                        i, j, naive_val, parallel_val, diff, tolerance
                    );
                }
            }
            max_diff = max_diff.max(diff);
        }
    }
    assert_eq!(
        mismatches, 0,
        "Found {} mismatches in {}x{} parallel matmul, max_diff={}",
        mismatches, size, size, max_diff
    );
}

#[test]
#[cfg(feature = "parallel")]
fn test_matvec_parallel_4096() {
    // Test parallel matvec for very large matrices (>=4096 rows)
    // This triggers the parallel path (PARALLEL_THRESHOLD = 4096)
    let rows = 4096;
    let cols = 512;

    let matrix = Matrix::from_vec(
        rows,
        cols,
        (0..rows * cols)
            .map(|i| ((i % 100) as f32) / 10.0)
            .collect(),
    )
    .unwrap();

    let vector = Vector::from_slice(
        &(0..cols)
            .map(|i| ((i % 50) as f32) / 5.0)
            .collect::<Vec<f32>>(),
    );

    // Compute result (should use parallel path)
    let result = matrix.matvec(&vector).unwrap();

    // Verify result shape
    assert_eq!(result.len(), rows);

    // Verify correctness by comparing with manual dot product calculation
    // Check a few sample rows
    for sample_row in [0, 1024, 2048, 3072, 4095] {
        let row_start = sample_row * cols;
        let row = &matrix.data[row_start..(row_start + cols)];

        // Manual dot product
        let expected: f32 = row
            .iter()
            .zip(vector.as_slice().iter())
            .map(|(a, b)| a * b)
            .sum();

        let actual = result.as_slice()[sample_row];
        let diff = (expected - actual).abs();
        let tolerance = if expected.abs() > 1.0 {
            expected.abs() * 1e-3
        } else {
            1e-3
        };

        assert!(
            diff < tolerance,
            "Mismatch at row {}: expected={}, actual={}, diff={}",
            sample_row,
            expected,
            actual,
            diff
        );
    }
}

// ===== Phase 2 Micro-kernel Tests (Issue #10) =====

#[test]
#[cfg(target_arch = "x86_64")]
fn test_horizontal_sum_avx2() {
    // Test the AVX2 horizontal sum helper function
    if !is_x86_feature_detected!("avx2") {
        println!("Skipping AVX2 horizontal sum test (CPU doesn't support AVX2)");
        return;
    }

    use std::arch::x86_64::*;

    // SAFETY: CPU feature verified at runtime, slices bounds-checked
    unsafe {
        // Test case 1: All ones
        let v = _mm256_set1_ps(1.0);
        let sum = Matrix::<f32>::horizontal_sum_avx2(v);
        assert!((sum - 8.0).abs() < 1e-6, "Expected 8.0, got {}", sum);

        // Test case 2: Sequence 1..8
        let v = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let sum = Matrix::<f32>::horizontal_sum_avx2(v);
        assert!((sum - 36.0).abs() < 1e-6, "Expected 36.0, got {}", sum);

        // Test case 3: Alternating signs
        let v = _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
        let sum = Matrix::<f32>::horizontal_sum_avx2(v);
        assert!(sum.abs() < 1e-6, "Expected ~0.0, got {}", sum);

        // Test case 4: Large values
        let v = _mm256_setr_ps(100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0);
        let sum = Matrix::<f32>::horizontal_sum_avx2(v);
        assert!((sum - 3600.0).abs() < 1e-3, "Expected 3600.0, got {}", sum);

        // Test case 5: Mixed positive/negative
        let v = _mm256_setr_ps(10.5, -5.25, 3.75, -8.0, 12.0, -6.5, 4.25, -2.75);
        let expected = 10.5 - 5.25 + 3.75 - 8.0 + 12.0 - 6.5 + 4.25 - 2.75;
        let sum = Matrix::<f32>::horizontal_sum_avx2(v);
        assert!(
            (sum - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            sum
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matmul_microkernel_4x1_avx2() {
    // Test the 4x1 AVX2 micro-kernel for matrix multiplication
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        println!("Skipping AVX2 micro-kernel test (CPU doesn't support AVX2/FMA)");
        return;
    }

    // Test case 1: Simple dot products
    // A rows: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    // B col:  [1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,  1,  1]
    // Expected: Row sums
    {
        let row0: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let row1: Vec<f32> = (17..=32).map(|x| x as f32).collect();
        let row2: Vec<f32> = (33..=48).map(|x| x as f32).collect();
        let row3: Vec<f32> = (49..=64).map(|x| x as f32).collect();
        let b_col = vec![1.0f32; 16];

        let a_rows = [
            row0.as_slice(),
            row1.as_slice(),
            row2.as_slice(),
            row3.as_slice(),
        ];
        let mut results = [0.0f32; 4];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_4x1_avx2(a_rows, &b_col, &mut results);
        }

        // Expected: sum(1..16), sum(17..32), sum(33..48), sum(49..64)
        let expected = [
            (1..=16).sum::<i32>() as f32,
            (17..=32).sum::<i32>() as f32,
            (33..=48).sum::<i32>() as f32,
            (49..=64).sum::<i32>() as f32,
        ];

        for i in 0..4 {
            assert!(
                (results[i] - expected[i]).abs() < 1e-3,
                "Row {}: expected {}, got {}",
                i,
                expected[i],
                results[i]
            );
        }
    }

    // Test case 2: Identity-like pattern
    // Each row is all zeros except one 1.0
    {
        let row0 = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let row1 = vec![
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let row2 = vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let row3 = vec![
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let b_col: Vec<f32> = (1..=16).map(|x| x as f32).collect();

        let a_rows = [
            row0.as_slice(),
            row1.as_slice(),
            row2.as_slice(),
            row3.as_slice(),
        ];
        let mut results = [0.0f32; 4];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_4x1_avx2(a_rows, &b_col, &mut results);
        }

        // Expected: Each result picks one element from b_col
        let expected = [1.0, 2.0, 3.0, 4.0];
        for i in 0..4 {
            assert!(
                (results[i] - expected[i]).abs() < 1e-6,
                "Row {}: expected {}, got {}",
                i,
                expected[i],
                results[i]
            );
        }
    }

    // Test case 3: Non-aligned size (not multiple of 8)
    // Size 10 (8 + 2 remainder)
    {
        let row0: Vec<f32> = (1..=10).map(|x| x as f32).collect();
        let row1: Vec<f32> = (11..=20).map(|x| x as f32).collect();
        let row2: Vec<f32> = (21..=30).map(|x| x as f32).collect();
        let row3: Vec<f32> = (31..=40).map(|x| x as f32).collect();
        let b_col = vec![2.0f32; 10];

        let a_rows = [
            row0.as_slice(),
            row1.as_slice(),
            row2.as_slice(),
            row3.as_slice(),
        ];
        let mut results = [0.0f32; 4];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_4x1_avx2(a_rows, &b_col, &mut results);
        }

        // Expected: 2x each row sum
        let expected = [
            2.0 * (1..=10).sum::<i32>() as f32,
            2.0 * (11..=20).sum::<i32>() as f32,
            2.0 * (21..=30).sum::<i32>() as f32,
            2.0 * (31..=40).sum::<i32>() as f32,
        ];

        for i in 0..4 {
            assert!(
                (results[i] - expected[i]).abs() < 1e-3,
                "Row {}: expected {}, got {}",
                i,
                expected[i],
                results[i]
            );
        }
    }

    // Test case 4: Mixed positive/negative values
    {
        let row0 = vec![
            1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0, 13.0, -14.0, 15.0,
            -16.0,
        ];
        let row1 = vec![
            2.0, -4.0, 6.0, -8.0, 10.0, -12.0, 14.0, -16.0, 18.0, -20.0, 22.0, -24.0, 26.0, -28.0,
            30.0, -32.0,
        ];
        let row2 = vec![
            0.5, -1.0, 1.5, -2.0, 2.5, -3.0, 3.5, -4.0, 4.5, -5.0, 5.5, -6.0, 6.5, -7.0, 7.5, -8.0,
        ];
        let row3 = vec![
            10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0,
            -10.0, 10.0, -10.0,
        ];
        let b_col = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];

        let a_rows = [
            row0.as_slice(),
            row1.as_slice(),
            row2.as_slice(),
            row3.as_slice(),
        ];
        let mut results = [0.0f32; 4];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_4x1_avx2(a_rows, &b_col, &mut results);
        }

        // Compute expected manually
        let expected = [
            row0.iter().sum::<f32>(),
            row1.iter().sum::<f32>(),
            row2.iter().sum::<f32>(),
            row3.iter().sum::<f32>(),
        ];

        for i in 0..4 {
            assert!(
                (results[i] - expected[i]).abs() < 1e-4,
                "Row {}: expected {}, got {}",
                i,
                expected[i],
                results[i]
            );
        }
    }

    // Test case 5: Zero accumulation
    {
        let row0 = vec![0.0f32; 16];
        let row1 = vec![0.0f32; 16];
        let row2 = vec![0.0f32; 16];
        let row3 = vec![0.0f32; 16];
        let b_col: Vec<f32> = (1..=16).map(|x| x as f32).collect();

        let a_rows = [
            row0.as_slice(),
            row1.as_slice(),
            row2.as_slice(),
            row3.as_slice(),
        ];
        let mut results = [0.0f32; 4];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_4x1_avx2(a_rows, &b_col, &mut results);
        }

        for (i, &result) in results.iter().enumerate() {
            assert!(
                result.abs() < 1e-6,
                "Row {}: expected 0.0, got {}",
                i,
                result
            );
        }
    }

    // Test case 6: Verify FMA correctness (a * b + c pattern)
    // Micro-kernel computes: sum(a[i] * b[i])
    {
        let row0 = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let row1 = vec![
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
            32.0,
        ];
        let row2 = vec![
            0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
        ];
        let row3 = vec![
            3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0, 42.0, 45.0,
            48.0,
        ];
        let b_col = vec![
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        ];

        let a_rows = [
            row0.as_slice(),
            row1.as_slice(),
            row2.as_slice(),
            row3.as_slice(),
        ];
        let mut results = [0.0f32; 4];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_4x1_avx2(a_rows, &b_col, &mut results);
        }

        // Expected: 0.5 x each row sum
        let expected = [
            0.5 * row0.iter().sum::<f32>(),
            0.5 * row1.iter().sum::<f32>(),
            0.5 * row2.iter().sum::<f32>(),
            0.5 * row3.iter().sum::<f32>(),
        ];

        for i in 0..4 {
            assert!(
                (results[i] - expected[i]).abs() < 1e-3,
                "Row {}: expected {}, got {}",
                i,
                expected[i],
                results[i]
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matmul_microkernel_8x1_avx512() {
    // Test the 8x1 AVX-512 micro-kernel for matrix multiplication (Phase 3)
    if !is_x86_feature_detected!("avx512f") {
        println!("Skipping AVX-512 micro-kernel test (CPU doesn't support AVX-512F)");
        return;
    }

    // Test case 1: Simple dot products with 32 elements (2 AVX-512 iterations)
    {
        let row0: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let row1: Vec<f32> = (33..=64).map(|x| x as f32).collect();
        let row2: Vec<f32> = (65..=96).map(|x| x as f32).collect();
        let row3: Vec<f32> = (97..=128).map(|x| x as f32).collect();
        let row4: Vec<f32> = (129..=160).map(|x| x as f32).collect();
        let row5: Vec<f32> = (161..=192).map(|x| x as f32).collect();
        let row6: Vec<f32> = (193..=224).map(|x| x as f32).collect();
        let row7: Vec<f32> = (225..=256).map(|x| x as f32).collect();
        let b_col = vec![1.0f32; 32];

        let a_rows = [
            row0.as_slice(),
            row1.as_slice(),
            row2.as_slice(),
            row3.as_slice(),
            row4.as_slice(),
            row5.as_slice(),
            row6.as_slice(),
            row7.as_slice(),
        ];
        let mut results = [0.0f32; 8];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_8x1_avx512(a_rows, &b_col, &mut results);
        }

        // Expected: sum of each row
        let expected = [
            (1..=32).sum::<i32>() as f32,
            (33..=64).sum::<i32>() as f32,
            (65..=96).sum::<i32>() as f32,
            (97..=128).sum::<i32>() as f32,
            (129..=160).sum::<i32>() as f32,
            (161..=192).sum::<i32>() as f32,
            (193..=224).sum::<i32>() as f32,
            (225..=256).sum::<i32>() as f32,
        ];

        for i in 0..8 {
            assert!(
                (results[i] - expected[i]).abs() < 1e-2,
                "Row {}: expected {}, got {}",
                i,
                expected[i],
                results[i]
            );
        }
    }

    // Test case 2: Scaled dot products
    {
        let row: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let rows: [&[f32]; 8] = [&row, &row, &row, &row, &row, &row, &row, &row];
        let b_col = vec![0.5f32; 32];
        let mut results = [0.0f32; 8];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_8x1_avx512(rows, &b_col, &mut results);
        }

        let expected = 0.5 * (1..=32).sum::<i32>() as f32;
        for (i, &result) in results.iter().enumerate() {
            assert!(
                (result - expected).abs() < 1e-2,
                "Row {}: expected {}, got {}",
                i,
                expected,
                result
            );
        }
    }

    // Test case 3: Zero accumulation
    {
        let zeros = vec![0.0f32; 32];
        let rows: [&[f32]; 8] = [
            &zeros, &zeros, &zeros, &zeros, &zeros, &zeros, &zeros, &zeros,
        ];
        let b_col: Vec<f32> = (1..=32).map(|x| x as f32).collect();
        let mut results = [0.0f32; 8];

        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        unsafe {
            Matrix::<f32>::matmul_microkernel_8x1_avx512(rows, &b_col, &mut results);
        }

        for (i, &result) in results.iter().enumerate() {
            assert!(
                result.abs() < 1e-6,
                "Row {}: expected 0.0, got {}",
                i,
                result
            );
        }
    }
}

// ===== AVX-512 Full Matmul Test =====

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matmul_avx512_backend_large_matrix() {
    // Test full matmul with AVX-512 backend on large matrices
    // This exercises the AVX-512 code path in matmul_simd
    if !is_x86_feature_detected!("avx512f") {
        println!("Skipping AVX-512 matmul test (CPU doesn't support AVX-512F)");
        return;
    }

    // Create large matrices that will trigger the SIMD optimization path
    // Using 256x256 to exercise 3-level cache blocking
    let size = 256;
    let a_data: Vec<f32> = (0..size * size).map(|i| (i % 10) as f32).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| ((i + 5) % 10) as f32).collect();

    let a = Matrix::from_vec_with_backend(size, size, a_data, Backend::AVX512);
    let b = Matrix::from_vec_with_backend(size, size, b_data, Backend::AVX512);

    let result = a.matmul(&b).expect("matmul should succeed");

    // Verify result dimensions
    assert_eq!(result.rows, size);
    assert_eq!(result.cols, size);

    // Spot check a few values against scalar reference
    let a_ref = Matrix::from_vec(
        size,
        size,
        (0..size * size).map(|i| (i % 10) as f32).collect(),
    )
    .expect("valid data");
    let b_ref = Matrix::from_vec(
        size,
        size,
        (0..size * size).map(|i| ((i + 5) % 10) as f32).collect(),
    )
    .expect("valid data");
    let expected = a_ref
        .matmul(&b_ref)
        .expect("reference matmul should succeed");

    // Check first few and last few elements
    for i in 0..5 {
        for j in 0..5 {
            let diff = (result[(i, j)] - expected[(i, j)]).abs();
            assert!(
                diff < 1e-3,
                "Mismatch at ({}, {}): AVX512={}, scalar={}",
                i,
                j,
                result[(i, j)],
                expected[(i, j)]
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matmul_avx512_remainder_handling() {
    // Test AVX-512 matmul with non-aligned matrix sizes to exercise remainder code
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    // Size not divisible by 8 or 16 to exercise remainder handling
    let size = 67;
    let a_data: Vec<f32> = (0..size * size).map(|i| i as f32 * 0.01).collect();
    let b_data: Vec<f32> = (0..size * size).map(|i| i as f32 * 0.01 + 0.5).collect();

    let a = Matrix::from_vec_with_backend(size, size, a_data.clone(), Backend::AVX512);
    let b = Matrix::from_vec_with_backend(size, size, b_data.clone(), Backend::AVX512);

    let result = a.matmul(&b).expect("matmul should succeed");

    // Compare with scalar
    let a_scalar = Matrix::from_vec_with_backend(size, size, a_data, Backend::Scalar);
    let b_scalar = Matrix::from_vec_with_backend(size, size, b_data, Backend::Scalar);
    let expected = a_scalar
        .matmul(&b_scalar)
        .expect("scalar matmul should succeed");

    for i in 0..size {
        for j in 0..size {
            let diff = (result[(i, j)] - expected[(i, j)]).abs();
            let max_val = expected[(i, j)].abs().max(1.0);
            assert!(
                diff / max_val < 1e-4,
                "Mismatch at ({}, {}): AVX512={}, scalar={}",
                i,
                j,
                result[(i, j)],
                expected[(i, j)]
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matmul_avx512_l3_blocking() {
    // Test AVX-512 matmul with L3 blocking (requires size >= 512)
    // This exercises the L3 cache blocking path with AVX-512 8x1 micro-kernel
    if !is_x86_feature_detected!("avx512f") {
        println!("Skipping AVX-512 L3 blocking test (CPU doesn't support AVX-512F)");
        return;
    }

    // Size must be >= 512 to trigger L3 blocking path
    // Use 520 to also exercise remainder handling (520 = 512 + 8, 520 % 16 != 0)
    let size = 520;
    let a_data: Vec<f32> = (0..size * size).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..size * size)
        .map(|i| ((i + 3) % 11) as f32 * 0.1)
        .collect();

    let a = Matrix::from_vec_with_backend(size, size, a_data.clone(), Backend::AVX512);
    let b = Matrix::from_vec_with_backend(size, size, b_data.clone(), Backend::AVX512);

    let result = a
        .matmul(&b)
        .expect("AVX-512 L3 blocking matmul should succeed");

    // Verify dimensions
    assert_eq!(result.rows, size);
    assert_eq!(result.cols, size);

    // Compare with scalar reference
    let a_scalar = Matrix::from_vec_with_backend(size, size, a_data, Backend::Scalar);
    let b_scalar = Matrix::from_vec_with_backend(size, size, b_data, Backend::Scalar);
    let expected = a_scalar
        .matmul(&b_scalar)
        .expect("scalar matmul should succeed");

    // Check corners and some middle elements
    let check_indices = [
        (0, 0),
        (0, size - 1),
        (size - 1, 0),
        (size - 1, size - 1),
        (size / 2, size / 2),
        (8, 8),   // Near 8x1 microkernel boundary
        (15, 15), // Near 16 element boundary
    ];
    for &(i, j) in &check_indices {
        let diff = (result[(i, j)] - expected[(i, j)]).abs();
        let max_val = expected[(i, j)].abs().max(1.0);
        assert!(
            diff / max_val < 1e-3,
            "Mismatch at ({}, {}): AVX512={}, scalar={}, diff={}",
            i,
            j,
            result[(i, j)],
            expected[(i, j)],
            diff
        );
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matmul_avx512_l3_nonaligned_cols() {
    // Test L3 blocking with column count that exercises remainder in 8x1 microkernel
    // k-dimension (inner loop) not divisible by 16 triggers remainder handling
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    // 513 columns: 513 = 32*16 + 1, exercises remainder handling
    let rows = 512;
    let cols = 513; // Not divisible by 16
    let a_data: Vec<f32> = (0..rows * cols).map(|i| (i % 13) as f32 * 0.05).collect();
    let b_data: Vec<f32> = (0..cols * rows).map(|i| (i % 17) as f32 * 0.05).collect();

    let a = Matrix::from_vec_with_backend(rows, cols, a_data.clone(), Backend::AVX512);
    let b = Matrix::from_vec_with_backend(cols, rows, b_data.clone(), Backend::AVX512);

    let result = a.matmul(&b).expect("matmul should succeed");
    assert_eq!(result.shape(), (rows, rows));

    // Verify against scalar
    let a_scalar = Matrix::from_vec_with_backend(rows, cols, a_data, Backend::Scalar);
    let b_scalar = Matrix::from_vec_with_backend(cols, rows, b_data, Backend::Scalar);
    let expected = a_scalar.matmul(&b_scalar).expect("scalar matmul");

    // Spot check
    for i in [0, 7, 8, 15, 16, 63, 64, 255, 256, rows - 1] {
        for j in [0, 7, 8, 15, 16, 63, 64, 255, 256, rows - 1] {
            let diff = (result[(i, j)] - expected[(i, j)]).abs();
            assert!(
                diff < 0.1,
                "Mismatch at ({},{}): got={}, expected={}",
                i,
                j,
                result[(i, j)],
                expected[(i, j)]
            );
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_matmul_avx512_l3_row_remainder() {
    // Test AVX-512 L3 blocking with row count that triggers 4x1 AVX2 and scalar remainder
    // 517 rows = 64*8 + 5 = L2_BLOCK*8 + 5, so remainder of 5 rows per L2 block
    // This exercises lines 1216-1252 (AVX2 4x1 for 4 rows, scalar for 1 row)
    if !is_x86_feature_detected!("avx512f") {
        return;
    }

    let rows = 517; // Not divisible by 8 to trigger remainder handling
    let cols = 512;
    let a_data: Vec<f32> = (0..rows * cols).map(|i| (i % 11) as f32 * 0.03).collect();
    let b_data: Vec<f32> = (0..cols * rows).map(|i| (i % 13) as f32 * 0.03).collect();

    let a = Matrix::from_vec_with_backend(rows, cols, a_data.clone(), Backend::AVX512);
    let b = Matrix::from_vec_with_backend(cols, rows, b_data.clone(), Backend::AVX512);

    let result = a.matmul(&b).expect("matmul should succeed");
    assert_eq!(result.shape(), (rows, rows));

    // Verify against scalar reference
    let a_scalar = Matrix::from_vec_with_backend(rows, cols, a_data, Backend::Scalar);
    let b_scalar = Matrix::from_vec_with_backend(cols, rows, b_data, Backend::Scalar);
    let expected = a_scalar.matmul(&b_scalar).expect("scalar matmul");

    // Check scattered points
    for i in [0, 5, 8, 63, 64, 256, 512, rows - 5, rows - 1] {
        for j in [0, 5, 8, 63, 64, 256, 512, rows - 5, rows - 1] {
            if i < rows && j < rows {
                let diff = (result[(i, j)] - expected[(i, j)]).abs();
                assert!(diff < 0.1, "Mismatch at ({},{})", i, j);
            }
        }
    }
}

#[test]
#[cfg(all(target_arch = "x86_64", feature = "parallel"))]
fn test_matmul_avx512_parallel_large() {
    // Test parallel AVX-512 matmul with 1024x1024 to hit parallel L3 blocking path
    // Requires: AVX-512F + parallel feature
    if !is_x86_feature_detected!("avx512f") {
        println!("Skipping: CPU doesn't support AVX-512F");
        return;
    }

    let size = 1024; // Triggers parallel path (PARALLEL_THRESHOLD = 1024)
    let a_data: Vec<f32> = (0..size * size).map(|i| ((i % 10) as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..size * size)
        .map(|i| (((i + 3) % 10) as f32) * 0.1)
        .collect();

    let a = Matrix::from_vec_with_backend(size, size, a_data.clone(), Backend::AVX512);
    let b = Matrix::from_vec_with_backend(size, size, b_data.clone(), Backend::AVX512);

    let result = a
        .matmul(&b)
        .expect("parallel AVX-512 matmul should succeed");
    assert_eq!(result.shape(), (size, size));

    // Spot check against scalar reference
    let a_scalar = Matrix::from_vec_with_backend(size, size, a_data, Backend::Scalar);
    let b_scalar = Matrix::from_vec_with_backend(size, size, b_data, Backend::Scalar);
    let expected = a_scalar.matmul(&b_scalar).expect("scalar matmul");

    // Check corners
    for (i, j) in [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)] {
        let diff = (result[(i, j)] - expected[(i, j)]).abs();
        let max_val = expected[(i, j)].abs().max(1.0);
        assert!(
            diff / max_val < 0.01,
            "Mismatch at ({},{}): got={}, expected={}",
            i,
            j,
            result[(i, j)],
            expected[(i, j)]
        );
    }
}
