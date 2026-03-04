//! Tests for trueno-tensor: tensor ops, einsum, and provable contracts.

use crate::{batch_matmul, einsum, matmul, outer, trace, Tensor};

// ── Tensor basics ──────────────────────────────────────────────────

#[test]
fn test_tensor_new() {
    let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.ndim(), 2);
    assert_eq!(t.len(), 6);
    assert!(!t.is_empty());
}

#[test]
fn test_tensor_zeros() {
    let t = Tensor::zeros(vec![3, 4]);
    assert_eq!(t.len(), 12);
    assert!(t.data().iter().all(|&x| x == 0.0));
}

#[test]
fn test_tensor_get_set() {
    let mut t = Tensor::zeros(vec![2, 3]);
    t.set(&[1, 2], 42.0);
    assert!((t.get(&[1, 2]) - 42.0).abs() < 1e-10);
    assert!((t.get(&[0, 0]) - 0.0).abs() < 1e-10);
}

#[test]
fn test_tensor_reshape() {
    let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let r = t.reshape(vec![3, 2]).unwrap();
    assert_eq!(r.shape(), &[3, 2]);
    assert_eq!(r.data(), t.data());
}

#[test]
fn test_tensor_reshape_mismatch() {
    let t = Tensor::zeros(vec![2, 3]);
    assert!(t.reshape(vec![2, 4]).is_err());
}

#[test]
fn test_tensor_data_length_mismatch() {
    let result = Tensor::new(vec![2, 3], vec![1.0, 2.0]);
    assert!(result.is_err());
}

#[test]
fn test_tensor_transpose_2d() {
    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let tr = t.transpose(&[1, 0]);
    assert_eq!(tr.shape(), &[3, 2]);
    assert!((tr.get(&[0, 0]) - 1.0).abs() < 1e-10);
    assert!((tr.get(&[0, 1]) - 4.0).abs() < 1e-10);
    assert!((tr.get(&[1, 0]) - 2.0).abs() < 1e-10);
    assert!((tr.get(&[2, 1]) - 6.0).abs() < 1e-10);
}

// ── Matrix multiply via einsum ─────────────────────────────────────

#[test]
fn test_matmul_2x3_3x2() {
    // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    let c = matmul(&a, &b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    // C[0,0] = 1*7+2*9+3*11 = 58
    assert!((c.get(&[0, 0]) - 58.0).abs() < 1e-4);
    // C[0,1] = 1*8+2*10+3*12 = 64
    assert!((c.get(&[0, 1]) - 64.0).abs() < 1e-4);
    // C[1,0] = 4*7+5*9+6*11 = 139
    assert!((c.get(&[1, 0]) - 139.0).abs() < 1e-4);
    // C[1,1] = 4*8+5*10+6*12 = 154
    assert!((c.get(&[1, 1]) - 154.0).abs() < 1e-4);
}

#[test]
fn test_matmul_identity() {
    // I * A = A
    let eye = Tensor::new(vec![3, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
    let a = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let c = matmul(&eye, &a).unwrap();
    assert_eq!(c.shape(), &[3, 2]);
    for i in 0..6 {
        assert!((c.data()[i] - a.data()[i]).abs() < 1e-6);
    }
}

// ── Einsum variations ──────────────────────────────────────────────

#[test]
fn test_einsum_outer_product() {
    let a = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::new(vec![2], vec![4.0, 5.0]).unwrap();
    let c = outer(&a, &b).unwrap();
    assert_eq!(c.shape(), &[3, 2]);
    assert!((c.get(&[0, 0]) - 4.0).abs() < 1e-6);
    assert!((c.get(&[1, 1]) - 10.0).abs() < 1e-6);
    assert!((c.get(&[2, 0]) - 12.0).abs() < 1e-6);
}

#[test]
fn test_einsum_dot_product() {
    // "i,i->" is inner product, but we output scalar as 0-dim
    let a = Tensor::new(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::new(vec![3], vec![4.0, 5.0, 6.0]).unwrap();
    let c = einsum("i,i->", &a, &b).unwrap();
    assert_eq!(c.shape(), &[] as &[usize]);
    // 1*4+2*5+3*6 = 32
    assert!((c.data()[0] - 32.0).abs() < 1e-6);
}

#[test]
fn test_einsum_trace() {
    let a = Tensor::new(vec![3, 3], vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]).unwrap();
    let t = trace(&a).unwrap();
    assert!((t - 6.0).abs() < 1e-6);
}

#[test]
fn test_einsum_batch_matmul() {
    // 2 batches of 2x2 matmul
    let a = Tensor::new(
        vec![2, 2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();
    let b = Tensor::new(
        vec![2, 2, 2],
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    // B is identity in each batch, so C = A
    let c = batch_matmul(&a, &b).unwrap();
    assert_eq!(c.shape(), &[2, 2, 2]);
    for i in 0..8 {
        assert!((c.data()[i] - a.data()[i]).abs() < 1e-6);
    }
}

#[test]
fn test_einsum_3d_contraction() {
    // "ijk,jkl->il" — contract middle two indices
    let a = Tensor::new(vec![2, 3, 4], (0..24).map(|i| i as f32).collect()).unwrap();
    let b = Tensor::new(vec![3, 4, 5], (0..60).map(|i| i as f32).collect()).unwrap();
    let c = einsum("ijk,jkl->il", &a, &b).unwrap();
    assert_eq!(c.shape(), &[2, 5]);

    // Verify one element: C[0,0] = sum_j sum_k A[0,j,k]*B[j,k,0]
    let mut expected = 0.0f32;
    for j in 0..3 {
        for k in 0..4 {
            expected += a.get(&[0, j, k]) * b.get(&[j, k, 0]);
        }
    }
    assert!((c.get(&[0, 0]) - expected).abs() < 1e-2);
}

// ── Contract: matmul associativity ─────────────────────────────────

#[test]
fn test_contract_matmul_associativity() {
    // (A*B)*C ≈ A*(B*C)
    let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::new(vec![3, 4], (1..=12).map(|i| i as f32).collect()).unwrap();
    let c = Tensor::new(vec![4, 2], (1..=8).map(|i| i as f32).collect()).unwrap();

    let ab = matmul(&a, &b).unwrap();
    let ab_c = matmul(&ab, &c).unwrap();

    let bc = matmul(&b, &c).unwrap();
    let a_bc = matmul(&a, &bc).unwrap();

    assert_eq!(ab_c.shape(), a_bc.shape());
    for i in 0..ab_c.len() {
        assert!(
            (ab_c.data()[i] - a_bc.data()[i]).abs() < 1e-1,
            "associativity violated at {i}: {} vs {}",
            ab_c.data()[i],
            a_bc.data()[i]
        );
    }
}

// ── Contract: transpose involution ─────────────────────────────────

#[test]
fn test_contract_transpose_involution() {
    // (A^T)^T = A
    let a = Tensor::new(vec![3, 4], (0..12).map(|i| i as f32).collect()).unwrap();
    let att = a.transpose(&[1, 0]).transpose(&[1, 0]);
    assert_eq!(att.shape(), a.shape());
    for i in 0..a.len() {
        assert!((att.data()[i] - a.data()[i]).abs() < 1e-10);
    }
}

// ── Error handling ─────────────────────────────────────────────────

#[test]
fn test_einsum_no_arrow() {
    let a = Tensor::zeros(vec![2, 3]);
    let b = Tensor::zeros(vec![3, 2]);
    assert!(einsum("ij,jk", &a, &b).is_err());
}

#[test]
fn test_einsum_dimension_mismatch() {
    let a = Tensor::zeros(vec![2, 3]);
    let b = Tensor::zeros(vec![4, 2]); // j=3 in a, j=4 in b
    assert!(einsum("ij,jk->ik", &a, &b).is_err());
}

#[test]
fn test_einsum_label_count_mismatch() {
    let a = Tensor::zeros(vec![2, 3]);
    let b = Tensor::zeros(vec![3, 2]);
    // "ijk" has 3 labels but a has 2 dims
    assert!(einsum("ijk,jk->ik", &a, &b).is_err());
}

#[test]
fn test_trace_non_square() {
    let a = Tensor::zeros(vec![2, 3]);
    assert!(trace(&a).is_err());
}

// ── Proptest ───────────────────────────────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_matmul_zero_row(n in 2..8usize, k in 2..8usize) {
            // Zero matrix times anything = zero
            let a = Tensor::zeros(vec![n, k]);
            let b = Tensor::new(vec![k, n], (0..k*n).map(|i| i as f32).collect()).unwrap();
            let c = matmul(&a, &b).unwrap();
            for &v in c.data() {
                prop_assert!((v).abs() < 1e-6);
            }
        }

        #[test]
        fn prop_reshape_preserves_data(m in 1..6usize, n in 1..6usize) {
            let data: Vec<f32> = (0..m*n).map(|i| i as f32).collect();
            let t = Tensor::new(vec![m, n], data.clone()).unwrap();
            let r = t.reshape(vec![n, m]).unwrap();
            prop_assert_eq!(r.data(), t.data());
        }
    }
}
