//! Tests for trueno-sparse.
//!
//! Follows provable contract sparse-spmv-v1.yaml falsification tests.

use crate::*;

// ============================================================================
// FALSIFY-SPARSE-001: CSR format validity
// ============================================================================

#[test]
fn test_reject_non_monotonic_offsets() {
    let result = CsrMatrix::<f32>::new(
        3, 3,
        vec![0, 2, 1, 3],  // non-monotonic: offsets[1]=2 > offsets[2]=1
        vec![0, 1, 0],
        vec![1.0, 2.0, 3.0],
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        SparseError::NonMonotonicOffsets { index: 1, .. } => {}
        e => panic!("Expected NonMonotonicOffsets at index 1, got {e:?}"),
    }
}

#[test]
fn test_reject_nonzero_first_offset() {
    let result = CsrMatrix::<f32>::new(
        2, 2,
        vec![1, 2, 3],  // offsets[0] != 0
        vec![0, 1, 0],
        vec![1.0, 2.0, 3.0],
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        SparseError::NonZeroFirstOffset { value: 1 } => {}
        e => panic!("Expected NonZeroFirstOffset, got {e:?}"),
    }
}

#[test]
fn test_reject_offsets_nnz_mismatch() {
    let result = CsrMatrix::<f32>::new(
        2, 2,
        vec![0, 1, 5],  // offsets[2]=5 but only 2 elements
        vec![0, 1],
        vec![1.0, 2.0],
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        SparseError::OffsetNnzMismatch { .. } => {}
        e => panic!("Expected OffsetNnzMismatch, got {e:?}"),
    }
}

#[test]
fn test_reject_wrong_offsets_length() {
    let result = CsrMatrix::<f32>::new(
        3, 3,
        vec![0, 1],  // length 2, expected 4 (rows+1)
        vec![0],
        vec![1.0],
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        SparseError::InvalidOffsetsLength { actual: 2, expected: 4 } => {}
        e => panic!("Expected InvalidOffsetsLength, got {e:?}"),
    }
}

// ============================================================================
// FALSIFY-SPARSE-002: Dimension correctness
// ============================================================================

#[test]
fn test_reject_column_out_of_bounds() {
    let result = CsrMatrix::<f32>::new(
        2, 3,
        vec![0, 1, 2],
        vec![0, 5],  // col=5 >= cols=3
        vec![1.0, 2.0],
    );
    assert!(result.is_err());
    match result.unwrap_err() {
        SparseError::ColumnOutOfBounds { col: 5, cols: 3, .. } => {}
        e => panic!("Expected ColumnOutOfBounds, got {e:?}"),
    }
}

#[test]
fn test_spmv_dimension_mismatch() {
    let a = CsrMatrix::<f32>::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
    let x = vec![1.0, 2.0];  // length 2, but cols=3
    let mut y = vec![0.0; 2];
    let result = a.spmv(1.0, &x, 0.0, &mut y);
    assert!(result.is_err());
}

#[test]
fn test_spmv_output_dimension_mismatch() {
    let a = CsrMatrix::<f32>::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();
    let x = vec![1.0, 2.0, 3.0];
    let mut y = vec![0.0; 5];  // length 5, but rows=2
    let result = a.spmv(1.0, &x, 0.0, &mut y);
    assert!(result.is_err());
}

// ============================================================================
// FALSIFY-SPARSE-003: Numerical accuracy (backward error bound)
// ============================================================================

#[test]
fn test_spmv_identity() {
    let n = 4;
    let a = CsrMatrix::<f32>::identity(n);
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let mut y = vec![0.0; n];

    a.spmv(1.0, &x, 0.0, &mut y).unwrap();

    for i in 0..n {
        assert!(
            (y[i] - x[i]).abs() < 1e-7,
            "Identity SpMV failed at i={i}: y={}, x={}",
            y[i], x[i]
        );
    }
}

#[test]
fn test_spmv_alpha_beta() {
    let a = CsrMatrix::<f32>::identity(3);
    let x = vec![1.0, 2.0, 3.0];
    let mut y = vec![10.0, 20.0, 30.0];

    // y = 2.0 * I * x + 0.5 * y = [2+5, 4+10, 6+15] = [7, 14, 21]
    a.spmv(2.0, &x, 0.5, &mut y).unwrap();

    assert!((y[0] - 7.0).abs() < 1e-5, "y[0]={}", y[0]);
    assert!((y[1] - 14.0).abs() < 1e-5, "y[1]={}", y[1]);
    assert!((y[2] - 21.0).abs() < 1e-5, "y[2]={}", y[2]);
}

#[test]
fn test_spmv_sparse_matrix() {
    // A = [[1, 0, 2],
    //      [0, 3, 0],
    //      [4, 0, 5]]
    let a = CsrMatrix::<f32>::new(
        3, 3,
        vec![0, 2, 3, 5],
        vec![0, 2, 1, 0, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
    ).unwrap();

    let x = vec![1.0, 2.0, 3.0];
    let mut y = vec![0.0; 3];

    a.spmv(1.0, &x, 0.0, &mut y).unwrap();

    // y[0] = 1*1 + 2*3 = 7
    // y[1] = 3*2 = 6
    // y[2] = 4*1 + 5*3 = 19
    assert!((y[0] - 7.0).abs() < 1e-5);
    assert!((y[1] - 6.0).abs() < 1e-5);
    assert!((y[2] - 19.0).abs() < 1e-5);
}

#[test]
fn test_spmv_empty_rows() {
    // A = [[0, 0],
    //      [1, 0],
    //      [0, 0]]
    let a = CsrMatrix::<f32>::new(
        3, 2,
        vec![0, 0, 1, 1],
        vec![0],
        vec![1.0],
    ).unwrap();

    let x = vec![5.0, 3.0];
    let mut y = vec![0.0; 3];
    a.spmv(1.0, &x, 0.0, &mut y).unwrap();

    assert!((y[0]).abs() < 1e-7);
    assert!((y[1] - 5.0).abs() < 1e-5);
    assert!((y[2]).abs() < 1e-7);
}

// ============================================================================
// COO construction and conversion
// ============================================================================

#[test]
fn test_coo_to_csr_basic() {
    let coo = CooMatrix::new(
        3, 3,
        vec![0, 1, 2, 0],
        vec![0, 1, 2, 2],
        vec![1.0_f32, 2.0, 3.0, 4.0],
    ).unwrap();

    let csr = CsrMatrix::from_coo(&coo);
    assert_eq!(csr.rows(), 3);
    assert_eq!(csr.cols(), 3);
    assert_eq!(csr.nnz(), 4);

    // Row 0 has 2 entries: (0,0)=1.0, (0,2)=4.0
    assert_eq!(csr.offsets()[0], 0);
    assert_eq!(csr.offsets()[1], 2);
    // Row 1 has 1 entry: (1,1)=2.0
    assert_eq!(csr.offsets()[2], 3);
    // Row 2 has 1 entry: (2,2)=3.0
    assert_eq!(csr.offsets()[3], 4);
}

#[test]
fn test_coo_to_csr_empty() {
    let coo = CooMatrix::<f32>::empty(5, 5);
    let csr = CsrMatrix::from_coo(&coo);
    assert_eq!(csr.rows(), 5);
    assert_eq!(csr.nnz(), 0);
}

#[test]
fn test_coo_rejects_bad_row() {
    let result = CooMatrix::new(2, 2, vec![0, 5], vec![0, 1], vec![1.0_f32, 2.0]);
    assert!(result.is_err());
}

#[test]
fn test_coo_rejects_bad_col() {
    let result = CooMatrix::new(2, 2, vec![0, 1], vec![0, 5], vec![1.0_f32, 2.0]);
    assert!(result.is_err());
}

#[test]
fn test_coo_rejects_mismatched_lengths() {
    let result = CooMatrix::new(2, 2, vec![0], vec![0, 1], vec![1.0_f32, 2.0]);
    assert!(result.is_err());
}

// ============================================================================
// SpMM tests
// ============================================================================

#[test]
fn test_spmm_identity() {
    let n = 3;
    let a = CsrMatrix::<f32>::identity(n);
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3x3
    let mut c = vec![0.0; 9];

    a.spmm(1.0, &b, 3, 0.0, &mut c).unwrap();

    for i in 0..9 {
        assert!((c[i] - b[i]).abs() < 1e-5, "c[{i}]={}, b={}", c[i], b[i]);
    }
}

#[test]
fn test_spmm_sparse() {
    // A = [[1, 0], [0, 2]]
    let a = CsrMatrix::<f32>::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).unwrap();

    // B = [[1, 2], [3, 4]]
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let mut c = vec![0.0; 4];

    a.spmm(1.0, &b, 2, 0.0, &mut c).unwrap();

    // C = [[1*1, 1*2], [2*3, 2*4]] = [[1, 2], [6, 8]]
    assert!((c[0] - 1.0).abs() < 1e-5);
    assert!((c[1] - 2.0).abs() < 1e-5);
    assert!((c[2] - 6.0).abs() < 1e-5);
    assert!((c[3] - 8.0).abs() < 1e-5);
}

// ============================================================================
// Dense roundtrip
// ============================================================================

#[test]
fn test_to_dense_roundtrip() {
    let coo = CooMatrix::new(
        2, 3,
        vec![0, 0, 1],
        vec![0, 2, 1],
        vec![1.0_f32, 2.0, 3.0],
    ).unwrap();

    let csr = CsrMatrix::from_coo(&coo);
    let dense = csr.to_dense();

    // [[1, 0, 2], [0, 3, 0]]
    assert!((dense[0] - 1.0).abs() < 1e-7);
    assert!((dense[1]).abs() < 1e-7);
    assert!((dense[2] - 2.0).abs() < 1e-7);
    assert!((dense[3]).abs() < 1e-7);
    assert!((dense[4] - 3.0).abs() < 1e-7);
    assert!((dense[5]).abs() < 1e-7);
}

// ============================================================================
// Statistics
// ============================================================================

#[test]
fn test_avg_nnz_per_row() {
    // 3 rows, 5 nonzeros
    let a = CsrMatrix::<f32>::new(
        3, 3,
        vec![0, 2, 3, 5],
        vec![0, 1, 2, 0, 1],
        vec![1.0; 5],
    ).unwrap();
    let avg = a.avg_nnz_per_row();
    assert!((avg - 5.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_row_length_variance() {
    // Row lengths: [2, 1, 2], mean = 5/3
    let a = CsrMatrix::<f32>::new(
        3, 3,
        vec![0, 2, 3, 5],
        vec![0, 1, 2, 0, 1],
        vec![1.0; 5],
    ).unwrap();
    let var = a.row_length_variance();
    // Variance = ((2-5/3)^2 + (1-5/3)^2 + (2-5/3)^2) / 3
    //          = (1/9 + 4/9 + 1/9) / 3 = 6/27 = 2/9
    assert!((var - 2.0 / 9.0).abs() < 1e-10);
}

#[test]
fn test_empty_matrix_stats() {
    let a = CsrMatrix::<f32>::new(0, 0, vec![0], vec![], vec![]).unwrap();
    assert_eq!(a.avg_nnz_per_row(), 0.0);
    assert_eq!(a.row_length_variance(), 0.0);
}

// ============================================================================
// Property-based tests (proptest)
// ============================================================================

mod proptests {
    use super::*;
    use proptest::prelude::*;

    /// Generate a random CSR matrix via COO.
    fn arb_csr(max_dim: usize, max_nnz: usize) -> impl Strategy<Value = CsrMatrix<f32>> {
        (1..=max_dim, 1..=max_dim, 0..=max_nnz).prop_flat_map(|(rows, cols, nnz)| {
            let row_idx = proptest::collection::vec(0..rows as u32, nnz);
            let col_idx = proptest::collection::vec(0..cols as u32, nnz);
            let vals = proptest::collection::vec(-100.0_f32..100.0, nnz);
            (Just(rows), Just(cols), row_idx, col_idx, vals)
        }).prop_map(|(rows, cols, ri, ci, vals)| {
            let coo = CooMatrix::new(rows, cols, ri, ci, vals).unwrap();
            CsrMatrix::from_coo(&coo)
        })
    }

    proptest! {
        /// FALSIFY-SPARSE-002: SpMV output has correct dimensions.
        #[test]
        fn prop_spmv_output_dimension(a in arb_csr(20, 50)) {
            let x = vec![1.0_f32; a.cols()];
            let mut y = vec![0.0_f32; a.rows()];
            a.spmv(1.0, &x, 0.0, &mut y).unwrap();
            prop_assert_eq!(y.len(), a.rows());
        }

        /// FALSIFY-SPARSE-003: SpMV matches dense matvec.
        #[test]
        fn prop_spmv_matches_dense(a in arb_csr(15, 40)) {
            let x: Vec<f32> = (0..a.cols()).map(|i| (i as f32 + 1.0) * 0.1).collect();
            let mut y_sparse = vec![0.0_f32; a.rows()];
            a.spmv(1.0, &x, 0.0, &mut y_sparse).unwrap();

            // Dense reference
            let dense = a.to_dense();
            let mut y_dense = vec![0.0_f32; a.rows()];
            for i in 0..a.rows() {
                for j in 0..a.cols() {
                    y_dense[i] += dense[i * a.cols() + j] * x[j];
                }
            }

            for i in 0..a.rows() {
                let err = (y_sparse[i] - y_dense[i]).abs();
                let scale = y_dense[i].abs().max(1.0);
                prop_assert!(
                    err / scale < 1e-4,
                    "SpMV mismatch at row {}: sparse={}, dense={}, err={}",
                    i, y_sparse[i], y_dense[i], err
                );
            }
        }

        /// FALSIFY-SPARSE-005: Linearity: spmv(A, alpha*x) == alpha * spmv(A, x).
        #[test]
        fn prop_spmv_linearity(a in arb_csr(10, 30), alpha in -10.0_f32..10.0) {
            let x: Vec<f32> = (0..a.cols()).map(|i| (i as f32 + 1.0) * 0.1).collect();

            // y1 = A * (alpha * x)
            let scaled_x: Vec<f32> = x.iter().map(|xi| alpha * xi).collect();
            let mut y1 = vec![0.0_f32; a.rows()];
            a.spmv(1.0, &scaled_x, 0.0, &mut y1).unwrap();

            // y2 = alpha * A * x
            let mut y2 = vec![0.0_f32; a.rows()];
            a.spmv(alpha, &x, 0.0, &mut y2).unwrap();

            for i in 0..a.rows() {
                let err = (y1[i] - y2[i]).abs();
                let scale = y1[i].abs().max(y2[i].abs()).max(1.0);
                prop_assert!(
                    err / scale < 1e-3,
                    "Linearity violated at row {}: y1={}, y2={}, alpha={}",
                    i, y1[i], y2[i], alpha
                );
            }
        }

        /// CSR from_coo always produces valid CSR.
        #[test]
        fn prop_coo_to_csr_valid(a in arb_csr(20, 50)) {
            // If we got here, from_coo succeeded. Verify invariants.
            let offsets = a.offsets();
            prop_assert_eq!(offsets[0], 0);
            prop_assert_eq!(offsets[a.rows()] as usize, a.nnz());
            for i in 0..a.rows() {
                prop_assert!(offsets[i] <= offsets[i + 1]);
            }
            for &col in a.col_indices() {
                prop_assert!((col as usize) < a.cols());
            }
        }
    }
}
