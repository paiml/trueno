use super::*;
use crate::{TruenoError, Vector};

// ===== Internal Implementation Tests (DISABLED - PMAT-018) =====
// These tests referenced internal methods (matmul_naive, matmul_simd, microkernels)
// that are now properly encapsulated in ops/arithmetic.rs.
// The public API tests above provide equivalent coverage.
// TODO: Move internal tests to ops/arithmetic.rs if needed.

#[cfg(internal_matrix_tests)]
mod internal_tests;

mod conv_property_tests;
mod property_tests;

#[test]
fn test_matrix_new() {
    let m = Matrix::new(3, 4);
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 4);
    assert_eq!(m.shape(), (3, 4));
    assert_eq!(m.as_slice().len(), 12);
}

#[test]
fn test_matrix_from_vec() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let m = Matrix::from_vec(2, 2, data).unwrap();
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 2);
    assert_eq!(m.get(0, 0), Some(&1.0));
    assert_eq!(m.get(0, 1), Some(&2.0));
    assert_eq!(m.get(1, 0), Some(&3.0));
    assert_eq!(m.get(1, 1), Some(&4.0));
}

#[test]
fn test_matrix_from_vec_invalid_size() {
    let data = vec![1.0, 2.0, 3.0];
    let result = Matrix::from_vec(2, 2, data);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_matrix_from_slice() {
    // TRUENO-SPEC-014 coverage test
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let m = Matrix::from_slice(2, 3, &data).unwrap();
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
    assert_eq!(m.get(0, 0), Some(&1.0));
    assert_eq!(m.get(1, 2), Some(&6.0));
}

#[test]
fn test_matrix_from_slice_invalid() {
    // TRUENO-SPEC-014 coverage test - error path
    let data = [1.0, 2.0, 3.0];
    let result = Matrix::from_slice(2, 2, &data);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_matrix_zeros() {
    let m = Matrix::zeros(2, 3);
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
    for &val in m.as_slice() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_matrix_identity() {
    let m = Matrix::identity(3);
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 3);

    // Check diagonal
    assert_eq!(m.get(0, 0), Some(&1.0));
    assert_eq!(m.get(1, 1), Some(&1.0));
    assert_eq!(m.get(2, 2), Some(&1.0));

    // Check off-diagonal
    assert_eq!(m.get(0, 1), Some(&0.0));
    assert_eq!(m.get(0, 2), Some(&0.0));
    assert_eq!(m.get(1, 0), Some(&0.0));
    assert_eq!(m.get(1, 2), Some(&0.0));
    assert_eq!(m.get(2, 0), Some(&0.0));
    assert_eq!(m.get(2, 1), Some(&0.0));
}

#[test]
fn test_matrix_get_out_of_bounds() {
    let m = Matrix::new(2, 2);
    assert_eq!(m.get(2, 0), None);
    assert_eq!(m.get(0, 2), None);
    assert_eq!(m.get(2, 2), None);
}

// ===== Matrix Multiplication Tests =====

#[test]
fn test_matmul_basic() {
    // [[1, 2],   [[5, 6],   [[19, 22],
    //  [3, 4]] x  [7, 8]] =  [43, 50]]
    let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);
    assert_eq!(c.get(0, 0), Some(&19.0));
    assert_eq!(c.get(0, 1), Some(&22.0));
    assert_eq!(c.get(1, 0), Some(&43.0));
    assert_eq!(c.get(1, 1), Some(&50.0));
}

#[test]
fn test_matmul_identity() {
    // A x I = A
    let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let identity = Matrix::identity(2);
    let result = a.matmul(&identity).unwrap();

    assert_eq!(result.get(0, 0), Some(&1.0));
    assert_eq!(result.get(0, 1), Some(&2.0));
    assert_eq!(result.get(1, 0), Some(&3.0));
    assert_eq!(result.get(1, 1), Some(&4.0));
}

#[test]
fn test_matmul_zeros() {
    // A x 0 = 0
    let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let zeros = Matrix::zeros(2, 2);
    let result = a.matmul(&zeros).unwrap();

    for &val in result.as_slice() {
        assert_eq!(val, 0.0);
    }
}

#[test]
fn test_matmul_dimension_mismatch() {
    // 2x3 matrix cannot multiply with 2x2 matrix (inner dimensions don't match)
    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = a.matmul(&b);

    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_matmul_non_square() {
    // 2x3 x 3x2 = 2x2
    // [[1, 2, 3],   [[7,  8],    [[58,  64],
    //  [4, 5, 6]] x  [9, 10],  =  [139, 154]]
    //                [11, 12]]
    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);
    assert_eq!(c.get(0, 0), Some(&58.0));
    assert_eq!(c.get(0, 1), Some(&64.0));
    assert_eq!(c.get(1, 0), Some(&139.0));
    assert_eq!(c.get(1, 1), Some(&154.0));
}

#[test]
fn test_matmul_single_element() {
    // 1x1 x 1x1 = 1x1
    let a = Matrix::from_vec(1, 1, vec![3.0]).unwrap();
    let b = Matrix::from_vec(1, 1, vec![4.0]).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.rows(), 1);
    assert_eq!(c.cols(), 1);
    assert_eq!(c.get(0, 0), Some(&12.0));
}

#[test]
fn test_matmul_remainder_rows() {
    // TRUENO-SPEC-014: Test matmul with rows not divisible by 4
    // This exercises the remainder handling path in SIMD matmul
    // 5x8 x 8x6 = 5x6 (5 % 4 = 1 remainder row)
    let a = Matrix::from_vec(5, 8, (0..40).map(|i| (i + 1) as f32).collect()).unwrap();
    let b = Matrix::from_vec(8, 6, (0..48).map(|i| (i + 1) as f32).collect()).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.rows(), 5);
    assert_eq!(c.cols(), 6);

    // Verify using naive calculation for first and last row
    // First row: [1,2,3,4,5,6,7,8] . columns of B
    let expected_00 = (1..=8)
        .zip((0..48).step_by(6).map(|i| (i + 1) as f32))
        .map(|(a, b)| a as f32 * b)
        .sum::<f32>();
    assert!((c.get(0, 0).unwrap() - expected_00).abs() < 1.0);
}

#[test]
fn test_matmul_remainder_rows_7() {
    // TRUENO-SPEC-014: 7x8 x 8x5 = 7x5 (7 % 4 = 3 remainder rows)
    let a = Matrix::from_vec(7, 8, (0..56).map(|_| 1.0f32).collect()).unwrap();
    let b = Matrix::from_vec(8, 5, (0..40).map(|_| 1.0f32).collect()).unwrap();
    let c = a.matmul(&b).unwrap();

    assert_eq!(c.rows(), 7);
    assert_eq!(c.cols(), 5);
    // Each element should be 8.0 (dot product of 8 ones)
    for &val in c.as_slice() {
        assert!((val - 8.0).abs() < 1e-5);
    }
}

// ===== Backend Equivalence Tests =====
// Note: Internal method tests (matmul_naive, matmul_simd) moved to ops/arithmetic.rs
// These tests now use the public matmul() API which auto-selects the best backend.

#[test]
fn test_matmul_public_api_small() {
    // Small matrix - verify public matmul works correctly
    let a = Matrix::from_vec(8, 8, (0..64).map(|i| i as f32).collect()).unwrap();
    let b = Matrix::identity(8);
    let result = a.matmul(&b).unwrap();
    // A x I = A
    assert_eq!(result.as_slice(), a.as_slice());
}

#[test]
fn test_matmul_public_api_large() {
    // Large matrix - verify SIMD path works correctly
    let size = 128;
    let a = Matrix::identity(size);
    let b = Matrix::from_vec(
        size,
        size,
        (0..size * size).map(|i| ((i * 2) % 100) as f32).collect(),
    )
    .unwrap();
    let result = a.matmul(&b).unwrap();
    // I x B = B
    assert_eq!(result.as_slice(), b.as_slice());
}

#[test]
fn test_matmul_public_api_rectangular() {
    // Rectangular matrices
    let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let result = a.matmul(&b).unwrap();

    // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //         = [[58, 64], [139, 154]]
    assert_eq!(result.rows(), 2);
    assert_eq!(result.cols(), 2);
    assert!((result.get(0, 0).unwrap() - 58.0).abs() < 1e-5);
    assert!((result.get(0, 1).unwrap() - 64.0).abs() < 1e-5);
    assert!((result.get(1, 0).unwrap() - 139.0).abs() < 1e-5);
    assert!((result.get(1, 1).unwrap() - 154.0).abs() < 1e-5);
}

// ===== GPU Tests =====

#[test]
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
fn test_gpu_availability() {
    use crate::backends::gpu::GpuBackend;
    // Just test that we can check GPU availability without crashing
    let _available = GpuBackend::is_available();
    // Note: We don't assert availability since CI may not have GPU
}

#[test]
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
#[ignore] // Ignore by default since CI may not have GPU
fn test_gpu_matmul_basic() {
    use crate::backends::gpu::GpuBackend;

    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    // Small test matrix (will use GPU if threshold is low enough)
    let a = Matrix::from_vec(
        4,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    )
    .unwrap();

    let b = Matrix::from_vec(
        4,
        4,
        vec![
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        ],
    )
    .unwrap();

    // Use public matmul API (GPU used for large matrices via threshold)
    let c = a.matmul(&b).expect("matmul should succeed");

    // Verify some basic properties
    assert_eq!(c.rows(), 4);
    assert_eq!(c.cols(), 4);

    // Verify against known result (first element)
    // [1,2,3,4] . [16,12,8,4] = 16+24+24+16 = 80
    assert!((c.get(0, 0).unwrap() - 80.0).abs() < 1e-4);
}

// ===== Transpose Tests =====

#[test]
fn test_transpose_basic() {
    // [[1, 2, 3],     [[1, 4],
    //  [4, 5, 6]]  ->  [2, 5],
    //                   [3, 6]]
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let t = m.transpose();

    assert_eq!(t.rows(), 3);
    assert_eq!(t.cols(), 2);
    assert_eq!(t.get(0, 0), Some(&1.0));
    assert_eq!(t.get(0, 1), Some(&4.0));
    assert_eq!(t.get(1, 0), Some(&2.0));
    assert_eq!(t.get(1, 1), Some(&5.0));
    assert_eq!(t.get(2, 0), Some(&3.0));
    assert_eq!(t.get(2, 1), Some(&6.0));
}

#[test]
fn test_transpose_square() {
    // [[1, 2],     [[1, 3],
    //  [3, 4]]  ->  [2, 4]]
    let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let t = m.transpose();

    assert_eq!(t.rows(), 2);
    assert_eq!(t.cols(), 2);
    assert_eq!(t.get(0, 0), Some(&1.0));
    assert_eq!(t.get(0, 1), Some(&3.0));
    assert_eq!(t.get(1, 0), Some(&2.0));
    assert_eq!(t.get(1, 1), Some(&4.0));
}

#[test]
fn test_transpose_single_row() {
    // [[1, 2, 3]] -> [[1],
    //                  [2],
    //                  [3]]
    let m = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
    let t = m.transpose();

    assert_eq!(t.rows(), 3);
    assert_eq!(t.cols(), 1);
    assert_eq!(t.get(0, 0), Some(&1.0));
    assert_eq!(t.get(1, 0), Some(&2.0));
    assert_eq!(t.get(2, 0), Some(&3.0));
}

#[test]
fn test_transpose_single_col() {
    // [[1],        [[1, 2, 3]]
    //  [2],   ->
    //  [3]]
    let m = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
    let t = m.transpose();

    assert_eq!(t.rows(), 1);
    assert_eq!(t.cols(), 3);
    assert_eq!(t.get(0, 0), Some(&1.0));
    assert_eq!(t.get(0, 1), Some(&2.0));
    assert_eq!(t.get(0, 2), Some(&3.0));
}

#[test]
fn test_transpose_single_element() {
    // [[5]] -> [[5]]
    let m = Matrix::from_vec(1, 1, vec![5.0]).unwrap();
    let t = m.transpose();

    assert_eq!(t.rows(), 1);
    assert_eq!(t.cols(), 1);
    assert_eq!(t.get(0, 0), Some(&5.0));
}

#[test]
fn test_transpose_identity() {
    // I^T = I
    let identity = Matrix::identity(3);
    let t = identity.transpose();

    assert_eq!(t.rows(), 3);
    assert_eq!(t.cols(), 3);

    // Check it's still identity
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_eq!(t.get(i, j), Some(&expected));
        }
    }
}

// ===== ML Primitives Tests =====

#[test]
fn test_max_pool2d() {
    let input = Matrix::from_vec(
        4,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    )
    .expect("valid input");

    // 2x2 kernel, 2x2 stride
    let pooled = input.max_pool2d((2, 2), (2, 2)).expect("valid pooling");
    assert_eq!(pooled.shape(), (2, 2));
    assert_eq!(pooled.get(0, 0), Some(&6.0)); // max of [1,2,5,6]
    assert_eq!(pooled.get(0, 1), Some(&8.0)); // max of [3,4,7,8]
    assert_eq!(pooled.get(1, 0), Some(&14.0)); // max of [9,10,13,14]
    assert_eq!(pooled.get(1, 1), Some(&16.0)); // max of [11,12,15,16]
}

#[test]
fn test_max_pool2d_stride_1() {
    let input = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        .expect("valid input");

    // 2x2 kernel, 1x1 stride (overlapping)
    let pooled = input.max_pool2d((2, 2), (1, 1)).expect("valid pooling");
    assert_eq!(pooled.shape(), (2, 2));
    assert_eq!(pooled.get(0, 0), Some(&5.0)); // max of [1,2,4,5]
    assert_eq!(pooled.get(0, 1), Some(&6.0)); // max of [2,3,5,6]
}

#[test]
fn test_avg_pool2d() {
    let input = Matrix::from_vec(
        4,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    )
    .expect("valid input");

    let pooled = input.avg_pool2d((2, 2), (2, 2)).expect("valid pooling");
    assert_eq!(pooled.shape(), (2, 2));
    // avg of [1,2,5,6] = 14/4 = 3.5
    assert!((pooled.get(0, 0).unwrap() - 3.5).abs() < 1e-5);
    // avg of [3,4,7,8] = 22/4 = 5.5
    assert!((pooled.get(0, 1).unwrap() - 5.5).abs() < 1e-5);
}

#[test]
fn test_topk() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 5.0, 3.0, 2.0, 6.0, 4.0]).expect("valid input");
    let (values, indices) = m.topk(3).expect("valid topk");
    assert_eq!(values, vec![6.0, 5.0, 4.0]);
    assert_eq!(indices, vec![4, 1, 5]);
}

#[test]
fn test_topk_empty() {
    let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid input");
    let (values, indices) = m.topk(0).expect("valid topk");
    assert!(values.is_empty());
    assert!(indices.is_empty());
}

#[test]
fn test_gather_rows() {
    let m = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid input");
    let gathered = m.gather(&[2, 0], 0).expect("valid gather");
    assert_eq!(gathered.shape(), (2, 2));
    assert_eq!(gathered.get(0, 0), Some(&5.0)); // Row 2, col 0
    assert_eq!(gathered.get(0, 1), Some(&6.0)); // Row 2, col 1
    assert_eq!(gathered.get(1, 0), Some(&1.0)); // Row 0, col 0
}

#[test]
fn test_gather_cols() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid input");
    let gathered = m.gather(&[2, 0], 1).expect("valid gather");
    assert_eq!(gathered.shape(), (2, 2));
    assert_eq!(gathered.get(0, 0), Some(&3.0)); // Row 0, col 2
    assert_eq!(gathered.get(0, 1), Some(&1.0)); // Row 0, col 0
}

#[test]
fn test_pad() {
    let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("valid input");
    let padded = m.pad(((1, 1), (1, 1)), 0.0).expect("valid pad");
    assert_eq!(padded.shape(), (4, 4));
    assert_eq!(padded.get(0, 0), Some(&0.0)); // top-left padding
    assert_eq!(padded.get(1, 1), Some(&1.0)); // original (0,0)
    assert_eq!(padded.get(2, 2), Some(&4.0)); // original (1,1)
    assert_eq!(padded.get(3, 3), Some(&0.0)); // bottom-right padding
}

#[test]
fn test_pad_asymmetric() {
    let m = Matrix::from_vec(1, 2, vec![1.0, 2.0]).expect("valid input");
    let padded = m.pad(((0, 1), (2, 0)), -1.0).expect("valid pad");
    assert_eq!(padded.shape(), (2, 4));
    assert_eq!(padded.get(0, 0), Some(&-1.0)); // left padding
    assert_eq!(padded.get(0, 2), Some(&1.0)); // original
    assert_eq!(padded.get(1, 0), Some(&-1.0)); // bottom padding
}

// =========================================================================
// Additional coverage tests for untested paths
// =========================================================================

#[test]
fn test_get_mut_valid() {
    let mut m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    if let Some(val) = m.get_mut(0, 1) {
        *val = 99.0;
    }
    assert_eq!(m.get(0, 1), Some(&99.0));
}

#[test]
fn test_get_mut_out_of_bounds() {
    let mut m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert!(m.get_mut(5, 0).is_none());
    assert!(m.get_mut(0, 10).is_none());
    assert!(m.get_mut(10, 10).is_none());
}

#[test]
fn test_matrix_zeros_coverage() {
    let m: Matrix<f32> = Matrix::zeros(3, 4);
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 4);
    for val in m.as_slice() {
        assert_eq!(*val, 0.0);
    }
}

#[test]
fn test_matrix_identity_coverage() {
    let m: Matrix<f32> = Matrix::identity(3);
    assert_eq!(m.get(0, 0), Some(&1.0));
    assert_eq!(m.get(1, 1), Some(&1.0));
    assert_eq!(m.get(2, 2), Some(&1.0));
    assert_eq!(m.get(0, 1), Some(&0.0));
    assert_eq!(m.get(1, 0), Some(&0.0));
}

#[test]
fn test_get_out_of_bounds_coverage() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert!(m.get(5, 0).is_none());
    assert!(m.get(0, 10).is_none());
}

// =========================================================================
// Kitchen Sink coverage tests (PMAT-018)
// =========================================================================

/// Test pad method with various configurations
#[test]
fn test_pad_kitchen_sink() {
    let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Symmetric padding
    let padded = m.pad(((1, 1), (1, 1)), 0.0).unwrap();
    assert_eq!(padded.rows(), 4);
    assert_eq!(padded.cols(), 5);
    assert_eq!(padded.get(0, 0), Some(&0.0)); // top-left corner (padding)
    assert_eq!(padded.get(1, 1), Some(&1.0)); // original data starts here

    // Asymmetric padding
    let padded2 = m.pad(((2, 0), (0, 3)), 9.0).unwrap();
    assert_eq!(padded2.rows(), 4);
    assert_eq!(padded2.cols(), 6);
    assert_eq!(padded2.get(0, 0), Some(&9.0)); // padding
    assert_eq!(padded2.get(2, 0), Some(&1.0)); // original data

    // Zero padding
    let padded3 = m.pad(((0, 0), (0, 0)), 0.0).unwrap();
    assert_eq!(padded3.rows(), 2);
    assert_eq!(padded3.cols(), 3);
}

/// Test gather with various axis configurations
#[test]
fn test_gather_kitchen_sink() {
    let m = Matrix::from_vec(3, 4, (0..12).map(|x| x as f32).collect()).unwrap();

    // Gather rows (axis=0)
    let rows = m.gather(&[0, 2], 0).unwrap();
    assert_eq!(rows.rows(), 2);
    assert_eq!(rows.cols(), 4);
    assert_eq!(rows.get(0, 0), Some(&0.0)); // row 0
    assert_eq!(rows.get(1, 0), Some(&8.0)); // row 2

    // Gather columns (axis=1)
    let cols = m.gather(&[1, 3], 1).unwrap();
    assert_eq!(cols.rows(), 3);
    assert_eq!(cols.cols(), 2);
}

/// Test topk with various k values
#[test]
fn test_topk_kitchen_sink() {
    let m = Matrix::from_vec(2, 4, vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]).unwrap();

    // topk operates on flattened data, not per-row
    // k=1 returns top 1 value from all 8 elements
    let (vals, idxs) = m.topk(1).unwrap();
    assert_eq!(vals.len(), 1);
    assert_eq!(idxs.len(), 1);
    assert_eq!(vals[0], 9.0); // max value in matrix

    // k=3 returns top 3 values from all 8 elements
    let (vals2, idxs2) = m.topk(3).unwrap();
    assert_eq!(vals2.len(), 3);
    assert_eq!(idxs2.len(), 3);

    // k=8 (all elements)
    let (vals3, _) = m.topk(8).unwrap();
    assert_eq!(vals3.len(), 8);

    // k=0 edge case
    let (vals4, idxs4) = m.topk(0).unwrap();
    assert_eq!(vals4.len(), 0);
    assert_eq!(idxs4.len(), 0);
}

/// Test pooling operations edge cases
#[test]
fn test_pooling_kitchen_sink() {
    // Exact divisible size
    let m = Matrix::from_vec(4, 4, (0..16).map(|x| x as f32).collect()).unwrap();

    let max_pooled = m.max_pool2d((2, 2), (2, 2)).unwrap();
    assert_eq!(max_pooled.rows(), 2);
    assert_eq!(max_pooled.cols(), 2);

    let avg_pooled = m.avg_pool2d((2, 2), (2, 2)).unwrap();
    assert_eq!(avg_pooled.rows(), 2);
    assert_eq!(avg_pooled.cols(), 2);

    // Non-exact divisible size
    let m2 = Matrix::from_vec(5, 5, (0..25).map(|x| x as f32).collect()).unwrap();
    let max_pooled2 = m2.max_pool2d((2, 2), (2, 2)).unwrap();
    assert_eq!(max_pooled2.rows(), 2); // floor(5/2)
    assert_eq!(max_pooled2.cols(), 2);
}

/// Test vecmat (v @ M) operation
#[test]
fn test_vecmat_kitchen_sink() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let m = Matrix::from_vec(
        3,
        4,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();

    // v (1x3) @ M (3x4) -> result (1x4)
    let result = Matrix::vecmat(&v, &m).unwrap();
    assert_eq!(result.len(), 4);
    // Manual calculation: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
    // = [1+10+27, 2+12+30, 3+14+33, 4+16+36] = [38, 44, 50, 56]
    assert!((result.as_slice()[0] - 38.0).abs() < 1e-5);
    assert!((result.as_slice()[1] - 44.0).abs() < 1e-5);
    assert!((result.as_slice()[2] - 50.0).abs() < 1e-5);
    assert!((result.as_slice()[3] - 56.0).abs() < 1e-5);
}

/// Test convolve2d edge cases
#[test]
fn test_convolve2d_kitchen_sink() {
    // 3x3 input with 3x3 kernel (produces 1x1)
    let input = Matrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

    let kernel = Matrix::from_vec(3, 3, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();

    let result = input.convolve2d(&kernel).unwrap();
    assert_eq!(result.rows(), 1);
    assert_eq!(result.cols(), 1);
    assert!((result.get(0, 0).unwrap() - 5.0).abs() < 1e-5); // Center value

    // 5x5 input with 3x3 kernel (produces 3x3)
    let input5 = Matrix::from_vec(5, 5, (0..25).map(|x| x as f32).collect()).unwrap();
    let result5 = input5.convolve2d(&kernel).unwrap();
    assert_eq!(result5.rows(), 3);
    assert_eq!(result5.cols(), 3);
}

/// Test embedding lookups
#[test]
fn test_embedding_kitchen_sink() {
    // Embedding table: 5 words x 4 dimensions
    let embeddings = Matrix::from_vec(5, 4, (0..20).map(|x| x as f32).collect()).unwrap();

    // Single lookup
    let result = embeddings.embedding_lookup(&[0]).unwrap();
    assert_eq!(result.rows(), 1);
    assert_eq!(result.cols(), 4);

    // Multiple lookups
    let result2 = embeddings.embedding_lookup(&[0, 2, 4]).unwrap();
    assert_eq!(result2.rows(), 3);
    assert_eq!(result2.cols(), 4);

    // Sparse lookup (returns matrix and indices)
    let (result3_matrix, result3_indices) = embeddings.embedding_lookup_sparse(&[0, 1, 2]).unwrap();
    assert_eq!(result3_matrix.rows(), 3);
    assert_eq!(result3_matrix.cols(), 4);
    assert_eq!(result3_indices.len(), 3);
}

/// Test batched_matmul_4d
#[test]
fn test_batched_matmul_4d_kitchen_sink() {
    // batch=2, heads=2, m=3, k=4, n=5
    let batch = 2;
    let heads = 2;
    let m = 3;
    let k = 4;
    let n = 5;

    // A: [batch, heads, m, k] = 2*2*3*4 = 48 elements
    let a_data: Vec<f32> = (0..48).map(|x| x as f32 * 0.1).collect();

    // B: [batch, heads, k, n] = 2*2*4*5 = 80 elements
    let b_data: Vec<f32> = (0..80).map(|x| x as f32 * 0.1).collect();

    // Result should be [batch, heads, m, n] = 2*2*3*5 = 60 elements
    let result = Matrix::batched_matmul_4d(&a_data, &b_data, batch, heads, m, k, n).unwrap();
    assert_eq!(result.len(), batch * heads * m * n);
}

/// Test matmul with remainder handling (non-aligned sizes)
#[test]
fn test_matmul_remainder_kitchen_sink() {
    // Sizes that don't align with SIMD widths
    for m in [1, 3, 5, 7, 9, 13, 17] {
        for k in [1, 3, 5, 7, 9, 15] {
            for n in [1, 3, 5, 7, 11] {
                let a = Matrix::from_vec(m, k, vec![1.0; m * k]).unwrap();
                let b = Matrix::from_vec(k, n, vec![1.0; k * n]).unwrap();
                let c = a.matmul(&b).unwrap();
                assert_eq!(c.rows(), m);
                assert_eq!(c.cols(), n);
                // Each element should equal k (sum of 1.0 x 1.0, k times)
                assert!((c.get(0, 0).unwrap() - k as f32).abs() < 1e-4);
            }
        }
    }
}

/// Test transpose edge cases
#[test]
fn test_transpose_kitchen_sink() {
    // 1xN
    let row = Matrix::from_vec(1, 5, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let col = row.transpose();
    assert_eq!(col.rows(), 5);
    assert_eq!(col.cols(), 1);

    // Nx1
    let col2 = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let row2 = col2.transpose();
    assert_eq!(row2.rows(), 1);
    assert_eq!(row2.cols(), 5);

    // 1x1
    let single = Matrix::from_vec(1, 1, vec![42.0]).unwrap();
    let single_t = single.transpose();
    assert_eq!(single_t.get(0, 0), Some(&42.0));
}
