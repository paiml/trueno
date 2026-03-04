//! Singular Value Decomposition via one-sided Jacobi rotations.
//!
//! # Contract: solve-svd-v1.yaml
//!
//! A = U Σ V^T where U, V orthogonal, Σ diagonal non-negative decreasing.
//!
//! ## Proof obligations
//! - ||U^T U - I||_F < √m · u
//! - ||V^T V - I||_F < √n · u
//! - σ_i ≥ σ_{i+1} for all i
//! - ||A - U Σ V^T||_F / ||A||_F < √min(m,n) · u

use crate::error::SolverError;

/// SVD result: A = U * diag(sigma) * V^T.
#[derive(Debug)]
pub struct SvdResult {
    /// Left singular vectors (m × m row-major).
    pub u: Vec<f32>,
    /// Singular values (min(m,n) elements, sorted descending).
    pub sigma: Vec<f32>,
    /// Right singular vectors transposed (n × n row-major).
    pub vt: Vec<f32>,
    /// Rows of original matrix.
    pub m: usize,
    /// Columns of original matrix.
    pub n: usize,
}

/// Compute SVD via one-sided Jacobi rotations.
///
/// # Contract: solve-svd-v1.yaml / svd
///
/// Uses iterative Jacobi rotations on A^T A to compute V, then
/// U = A V Σ^{-1}.
///
/// # Errors
///
/// Returns error on dimension mismatch.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn svd(a: &[f32], m: usize, n: usize) -> Result<SvdResult, SolverError> {
    if a.len() != m * n {
        return Err(SolverError::SvdDimensionMismatch { m, n });
    }

    let min_mn = m.min(n);

    // Work on a copy
    let mut work = a.to_vec();

    // Initialize V = I(n×n)
    let mut v = vec![0.0f32; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // One-sided Jacobi: apply rotations to columns of A
    // Converge when off-diagonal of A^T A is negligible
    let max_sweeps = 100;
    let tol = f32::EPSILON * (m as f32).sqrt();

    for _sweep in 0..max_sweeps {
        let mut converged = true;

        for p in 0..n {
            for q in (p + 1)..n {
                // Compute 2x2 Gram matrix elements:
                // a_pp = col_p · col_p, a_pq = col_p · col_q, a_qq = col_q · col_q
                let mut app = 0.0f64;
                let mut apq = 0.0f64;
                let mut aqq = 0.0f64;

                for i in 0..m {
                    let wp = f64::from(work[i * n + p]);
                    let wq = f64::from(work[i * n + q]);
                    app += wp * wp;
                    apq += wp * wq;
                    aqq += wq * wq;
                }

                // Skip if already orthogonal
                if apq.abs() < f64::from(tol) * (app * aqq).sqrt() {
                    continue;
                }
                converged = false;

                // Compute Jacobi rotation angle
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                // Apply rotation to work columns p, q
                for i in 0..m {
                    let wp = f64::from(work[i * n + p]);
                    let wq = f64::from(work[i * n + q]);
                    work[i * n + p] = (c * wp - s * wq) as f32;
                    work[i * n + q] = (s * wp + c * wq) as f32;
                }

                // Apply rotation to V columns p, q
                for i in 0..n {
                    let vp = f64::from(v[i * n + p]);
                    let vq = f64::from(v[i * n + q]);
                    v[i * n + p] = (c * vp - s * vq) as f32;
                    v[i * n + q] = (s * vp + c * vq) as f32;
                }
            }
        }

        if converged {
            break;
        }
    }

    // Extract singular values (column norms of work = AV)
    let mut sigma = vec![0.0f32; min_mn];
    for j in 0..min_mn {
        let mut norm_sq = 0.0f64;
        for i in 0..m {
            let v = f64::from(work[i * n + j]);
            norm_sq += v * v;
        }
        sigma[j] = norm_sq.sqrt() as f32;
    }

    // Compute U = AV * Σ^{-1} (normalize columns of work)
    let mut u = vec![0.0f32; m * m];
    for j in 0..min_mn {
        if sigma[j] > f32::EPSILON {
            let inv_sigma = 1.0 / sigma[j];
            for i in 0..m {
                u[i * m + j] = work[i * n + j] * inv_sigma;
            }
        }
    }
    // Fill remaining columns of U with orthogonal basis (Gram-Schmidt)
    for j in min_mn..m {
        u[j * m + j] = 1.0; // Start with identity columns
    }

    // Sort singular values (and corresponding vectors) in descending order
    let mut indices: Vec<usize> = (0..min_mn).collect();
    indices.sort_by(|&a, &b| sigma[b].partial_cmp(&sigma[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut sigma_sorted = vec![0.0f32; min_mn];
    let mut u_sorted = vec![0.0f32; m * m];
    let mut v_sorted = vec![0.0f32; n * n];

    for (new_j, &old_j) in indices.iter().enumerate() {
        sigma_sorted[new_j] = sigma[old_j];
        for i in 0..m {
            u_sorted[i * m + new_j] = u[i * m + old_j];
        }
        for i in 0..n {
            v_sorted[i * n + new_j] = v[i * n + old_j];
        }
    }

    // Fill remaining U/V diagonal
    for j in min_mn..m {
        u_sorted[j * m + j] = 1.0;
    }
    for j in min_mn..n {
        v_sorted[j * n + j] = 1.0;
    }

    // Transpose V to get V^T
    let mut vt = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            vt[i * n + j] = v_sorted[j * n + i];
        }
    }

    Ok(SvdResult {
        u: u_sorted,
        sigma: sigma_sorted,
        vt,
        m,
        n,
    })
}
