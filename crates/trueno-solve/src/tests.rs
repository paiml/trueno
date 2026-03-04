//! Solver tests — provable contracts and falsification.

use crate::cholesky::cholesky;
use crate::lu::lu_factorize;
use crate::qr::qr_factorize;
use crate::svd::svd;
use crate::trsm::{trsm, DiagonalType, TriangularSide};

// ============================================================================
// LU factorization tests
// ============================================================================

#[test]
fn test_lu_identity() {
    let a = [1.0, 0.0, 0.0, 1.0_f32];
    let lu = lu_factorize(&a, 2).expect("lu ok");
    let x = lu.solve(&[3.0, 7.0]).expect("solve ok");
    assert!((x[0] - 3.0).abs() < 1e-5);
    assert!((x[1] - 7.0).abs() < 1e-5);
}

#[test]
fn test_lu_2x2() {
    // A = [[2, 1], [1, 3]], b = [5, 7]
    // x = [1.6, 1.8]
    let a = [2.0, 1.0, 1.0, 3.0_f32];
    let lu = lu_factorize(&a, 2).expect("lu ok");
    let x = lu.solve(&[5.0, 7.0]).expect("solve ok");
    assert!((x[0] - 1.6).abs() < 1e-5, "x[0]={}", x[0]);
    assert!((x[1] - 1.8).abs() < 1e-5, "x[1]={}", x[1]);
}

#[test]
fn test_lu_3x3() {
    // A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]] (tridiagonal)
    let a = [2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0_f32];
    let lu = lu_factorize(&a, 3).expect("lu ok");
    let b = [1.0, 0.0, 1.0_f32];
    let x = lu.solve(&b).expect("solve ok");

    // Verify Ax ≈ b
    for i in 0..3 {
        let mut ax_i = 0.0f32;
        for j in 0..3 {
            ax_i += a[i * 3 + j] * x[j];
        }
        assert!((ax_i - b[i]).abs() < 1e-4, "Ax[{i}]={ax_i}, b[{i}]={}", b[i]);
    }
}

#[test]
fn test_lu_backward_error() {
    let a = [4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0_f32];
    let lu = lu_factorize(&a, 3).expect("lu ok");

    let l = lu.extract_l();
    let u = lu.extract_u();
    let p = lu.extract_p();

    // Verify PA ≈ LU
    let n = 3;
    let mut pa = vec![0.0f32; n * n];
    let mut lu_product = vec![0.0f32; n * n];

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                pa[i * n + j] += p[i * n + k] * a[k * n + j];
                lu_product[i * n + j] += l[i * n + k] * u[k * n + j];
            }
        }
    }

    let mut max_err = 0.0f32;
    for i in 0..n * n {
        max_err = max_err.max((pa[i] - lu_product[i]).abs());
    }
    assert!(max_err < 1e-4, "PA ≠ LU, max_err={max_err}");
}

#[test]
fn test_lu_singular_detected() {
    let a = [1.0, 2.0, 2.0, 4.0_f32]; // Singular: row2 = 2*row1
    assert!(lu_factorize(&a, 2).is_err());
}

// ============================================================================
// QR factorization tests
// ============================================================================

#[test]
fn test_qr_identity() {
    let a = [1.0, 0.0, 0.0, 1.0_f32];
    let qr = qr_factorize(&a, 2, 2).expect("qr ok");
    let r = qr.extract_r();

    // R should be ±I (signs may flip)
    assert!(r[0].abs() > 0.9);
    assert!(r[3].abs() > 0.9);
}

#[test]
fn test_qr_orthogonality() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f32]; // 3×2
    let qr = qr_factorize(&a, 3, 2).expect("qr ok");
    let q = qr.extract_q();

    // Check Q^T Q ≈ I
    let m = 3;
    let mut qtq = vec![0.0f32; m * m];
    for i in 0..m {
        for j in 0..m {
            for k in 0..m {
                qtq[i * m + j] += q[k * m + i] * q[k * m + j];
            }
        }
    }

    for i in 0..m {
        for j in 0..m {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (qtq[i * m + j] - expected).abs() < 1e-4,
                "Q^TQ[{i},{j}] = {}, expected {expected}",
                qtq[i * m + j]
            );
        }
    }
}

#[test]
fn test_qr_reconstruction() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f32]; // 3×2
    let m = 3;
    let n = 2;
    let qr = qr_factorize(&a, m, n).expect("qr ok");
    let q = qr.extract_q();
    let r = qr.extract_r();

    // QR should reconstruct A
    for i in 0..m {
        for j in 0..n {
            let mut qr_ij = 0.0f32;
            for k in 0..n {
                qr_ij += q[i * m + k] * r[k * n + j];
            }
            assert!(
                (qr_ij - a[i * n + j]).abs() < 1e-3,
                "QR[{i},{j}]={qr_ij}, A[{i},{j}]={}",
                a[i * n + j]
            );
        }
    }
}

#[test]
fn test_qr_solve_least_squares() {
    // Overdetermined system: 3×2
    let a = [1.0, 1.0, 1.0, 2.0, 1.0, 3.0_f32];
    let b = [1.0, 2.0, 3.0_f32]; // y ≈ x (least squares fit)

    let qr = qr_factorize(&a, 3, 2).expect("qr ok");
    let x = qr.solve(&b).expect("solve ok");

    // Verify the residual is reasonable
    let mut residual = 0.0f32;
    for i in 0..3 {
        let mut ax = 0.0f32;
        for j in 0..2 {
            ax += a[i * 2 + j] * x[j];
        }
        residual += (ax - b[i]).powi(2);
    }
    assert!(residual.sqrt() < 1.0, "Residual too large: {}", residual.sqrt());
}

// ============================================================================
// SVD tests
// ============================================================================

#[test]
fn test_svd_2x2_identity() {
    let a = [1.0, 0.0, 0.0, 1.0_f32];
    let result = svd(&a, 2, 2).expect("svd ok");

    // Singular values should be [1, 1]
    assert!((result.sigma[0] - 1.0).abs() < 1e-4, "σ[0]={}", result.sigma[0]);
    assert!((result.sigma[1] - 1.0).abs() < 1e-4, "σ[1]={}", result.sigma[1]);
}

#[test]
fn test_svd_singular_values_nonneg_decreasing() {
    let a = [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0_f32];
    let result = svd(&a, 3, 3).expect("svd ok");

    for s in &result.sigma {
        assert!(*s >= -1e-6, "Negative singular value: {s}");
    }
    for i in 0..result.sigma.len() - 1 {
        assert!(
            result.sigma[i] >= result.sigma[i + 1] - 1e-6,
            "σ[{}]={} < σ[{}]={}",
            i,
            result.sigma[i],
            i + 1,
            result.sigma[i + 1]
        );
    }
}

#[test]
fn test_svd_reconstruction() {
    let a = [1.0, 2.0, 3.0, 4.0_f32]; // 2×2
    let m = 2;
    let n = 2;
    let result = svd(&a, m, n).expect("svd ok");

    // Reconstruct: A ≈ U * diag(σ) * V^T
    let min_mn = m.min(n);
    let mut recon = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for k in 0..min_mn {
                sum += f64::from(result.u[i * m + k])
                    * f64::from(result.sigma[k])
                    * f64::from(result.vt[k * n + j]);
            }
            recon[i * n + j] = sum as f32;
        }
    }

    for i in 0..m * n {
        assert!(
            (recon[i] - a[i]).abs() < 1e-3,
            "Reconstruction error at {i}: recon={}, orig={}",
            recon[i],
            a[i]
        );
    }
}

#[test]
fn test_svd_orthogonality_u() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0_f32]; // 3×2
    let result = svd(&a, 3, 2).expect("svd ok");

    let m = result.m;
    let min_mn = result.m.min(result.n);
    // Check first min_mn columns of U are orthonormal
    for i in 0..min_mn {
        for j in 0..min_mn {
            let mut dot = 0.0f64;
            for k in 0..m {
                dot += f64::from(result.u[k * m + i]) * f64::from(result.u[k * m + j]);
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-3,
                "U orthogonality: U^T U[{i},{j}]={dot}, expected {expected}"
            );
        }
    }
}

// ============================================================================
// Cholesky tests
// ============================================================================

#[test]
fn test_cholesky_2x2() {
    // A = [[4, 2], [2, 3]] (positive definite)
    let a = [4.0, 2.0, 2.0, 3.0_f32];
    let chol = cholesky(&a, 2).expect("cholesky ok");

    // L L^T should reconstruct A
    let n = 2;
    let mut recon = vec![0.0f32; 4];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                recon[i * n + j] += chol.l[i * n + k] * chol.l[j * n + k];
            }
        }
    }

    for i in 0..4 {
        assert!(
            (recon[i] - a[i]).abs() < 1e-5,
            "LL^T[{i}]={}, A[{i}]={}",
            recon[i],
            a[i]
        );
    }
}

#[test]
fn test_cholesky_solve() {
    let a = [4.0, 2.0, 2.0, 3.0_f32];
    let chol = cholesky(&a, 2).expect("cholesky ok");
    let x = chol.solve(&[8.0, 7.0]).expect("solve ok");

    // Verify Ax ≈ b
    let ax0 = 4.0 * x[0] + 2.0 * x[1];
    let ax1 = 2.0 * x[0] + 3.0 * x[1];
    assert!((ax0 - 8.0).abs() < 1e-4);
    assert!((ax1 - 7.0).abs() < 1e-4);
}

#[test]
fn test_cholesky_not_positive_definite() {
    // Not positive definite: eigenvalues include negative
    let a = [1.0, 3.0, 3.0, 1.0_f32];
    assert!(cholesky(&a, 2).is_err());
}

#[test]
fn test_cholesky_3x3() {
    // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
    let a = [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0_f32];
    let chol = cholesky(&a, 3).expect("cholesky ok");

    // Reconstruct
    let n = 3;
    let mut recon = vec![0.0f32; 9];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                recon[i * n + j] += chol.l[i * n + k] * chol.l[j * n + k];
            }
        }
    }

    for i in 0..9 {
        assert!(
            (recon[i] - a[i]).abs() < 1e-3,
            "LL^T[{i}]={}, A[{i}]={}",
            recon[i],
            a[i]
        );
    }
}

// ============================================================================
// Property-based tests
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_lu_solve_residual(
            a00 in -10.0f32..10.0,
            a01 in -10.0f32..10.0,
            a10 in -10.0f32..10.0,
            a11 in -10.0f32..10.0,
            b0 in -10.0f32..10.0,
            b1 in -10.0f32..10.0,
        ) {
            let a = [a00, a01, a10, a11];
            if let Ok(lu) = lu_factorize(&a, 2) {
                if let Ok(x) = lu.solve(&[b0, b1]) {
                    // Verify ||Ax - b|| is small
                    let r0 = a00 * x[0] + a01 * x[1] - b0;
                    let r1 = a10 * x[0] + a11 * x[1] - b1;
                    let residual = (r0 * r0 + r1 * r1).sqrt();
                    let b_norm = (b0 * b0 + b1 * b1).sqrt() + 1e-10;
                    prop_assert!(residual / b_norm < 1e-3, "Residual too large: {residual}");
                }
            }
        }

        #[test]
        fn prop_svd_values_nonneg(
            a00 in -5.0f32..5.0,
            a01 in -5.0f32..5.0,
            a10 in -5.0f32..5.0,
            a11 in -5.0f32..5.0,
        ) {
            let a = [a00, a01, a10, a11];
            if let Ok(result) = svd(&a, 2, 2) {
                for s in &result.sigma {
                    prop_assert!(*s >= -1e-5, "Negative singular value: {s}");
                }
                if result.sigma.len() >= 2 {
                    prop_assert!(result.sigma[0] >= result.sigma[1] - 1e-5);
                }
            }
        }
    }
}

// ============================================================================
// TRSM (Triangular Solve AX = B)
// ============================================================================

#[test]
fn test_trsm_lower_2x2() -> Result<(), Box<dyn std::error::Error>> {
    // L = [[2, 0], [3, 4]], B = [[1], [2]]
    // Lx = b => x0 = 1/2, x1 = (2 - 3*0.5)/4 = 0.125
    let l = [2.0, 0.0, 3.0, 4.0_f32];
    let b = [1.0, 2.0_f32];
    let result = trsm(&l, &b, 2, 1, TriangularSide::Lower, DiagonalType::NonUnit)?;
    assert!((result.x[0] - 0.5).abs() < 1e-5);
    assert!((result.x[1] - 0.125).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_trsm_upper_2x2() -> Result<(), Box<dyn std::error::Error>> {
    // U = [[3, 1], [0, 2]], B = [[5], [4]]
    // x1 = 4/2 = 2, x0 = (5 - 1*2)/3 = 1
    let u = [3.0, 1.0, 0.0, 2.0_f32];
    let b = [5.0, 4.0_f32];
    let result = trsm(&u, &b, 2, 1, TriangularSide::Upper, DiagonalType::NonUnit)?;
    assert!((result.x[0] - 1.0).abs() < 1e-5);
    assert!((result.x[1] - 2.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_trsm_unit_diagonal() -> Result<(), Box<dyn std::error::Error>> {
    // L = [[1, 0], [3, 1]], B = [[1], [5]]  (unit diagonal)
    // x0 = 1, x1 = 5 - 3*1 = 2
    let l = [1.0, 0.0, 3.0, 1.0_f32];
    let b = [1.0, 5.0_f32];
    let result = trsm(&l, &b, 2, 1, TriangularSide::Lower, DiagonalType::Unit)?;
    assert!((result.x[0] - 1.0).abs() < 1e-5);
    assert!((result.x[1] - 2.0).abs() < 1e-5);
    Ok(())
}

#[test]
fn test_trsm_multiple_rhs() -> Result<(), Box<dyn std::error::Error>> {
    // L = [[2, 0], [1, 3]], B = [[2, 4], [5, 7]]  (n=2, nrhs=2)
    let l = [2.0, 0.0, 1.0, 3.0_f32];
    let b = [2.0, 4.0, 5.0, 7.0_f32];
    let result = trsm(&l, &b, 2, 2, TriangularSide::Lower, DiagonalType::NonUnit)?;
    // Column 0: x0=2/2=1, x1=(5-1*1)/3=4/3
    assert!((result.x[0] - 1.0).abs() < 1e-5);
    assert!((result.x[2] - 4.0 / 3.0).abs() < 1e-4);
    // Column 1: x0=4/2=2, x1=(7-1*2)/3=5/3
    assert!((result.x[1] - 2.0).abs() < 1e-5);
    assert!((result.x[3] - 5.0 / 3.0).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_trsm_backward_error() -> Result<(), Box<dyn std::error::Error>> {
    // Verify ||AX - B||/||B|| is small
    let l = [3.0, 0.0, 0.0, 2.0, 5.0, 0.0, 1.0, 4.0, 6.0_f32];
    let b = [1.0, 2.0, 3.0_f32];
    let result = trsm(&l, &b, 3, 1, TriangularSide::Lower, DiagonalType::NonUnit)?;

    // Compute AX
    let n = 3;
    let mut ax = vec![0.0f32; n];
    for i in 0..n {
        for j in 0..n {
            ax[i] += l[i * n + j] * result.x[j];
        }
    }
    // ||AX - B|| / ||B||
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for i in 0..n {
        num += (ax[i] - b[i]).powi(2);
        den += b[i].powi(2);
    }
    let rel_err = (num / den).sqrt();
    assert!(rel_err < 1e-5, "TRSM backward error {rel_err} too large");
    Ok(())
}

#[test]
fn test_trsm_singular_detected() {
    let a = [1.0, 0.0, 0.0, 0.0_f32]; // zero diagonal at (1,1)
    let b = [1.0, 2.0_f32];
    let result = trsm(&a, &b, 2, 1, TriangularSide::Lower, DiagonalType::NonUnit);
    assert!(result.is_err());
}
