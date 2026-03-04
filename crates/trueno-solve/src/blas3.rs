//! BLAS Level-3 operations: syrk, trmm, symm.
//!
//! # Contract: blas-level3-v1.yaml
//!
//! Reference CPU implementations for cuBLAS parity.

use crate::error::SolverError;

/// Symmetric rank-k update: C = α·A·Aᵀ + β·C
///
/// A is n×k, C is n×n (symmetric, stored as full matrix row-major).
///
/// # Errors
///
/// Returns error on dimension mismatch.
pub fn syrk(
    a: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), SolverError> {
    validate_buffer(a, n * k, n, k)?;
    validate_buffer(c, n * n, n, n)?;

    // C = β·C
    for val in c.iter_mut() {
        *val *= beta;
    }

    // C += α·A·Aᵀ
    for i in 0..n {
        for j in 0..=i {
            let dot = dot_row_row(a, k, i, j);
            let update = alpha * dot;
            c[i * n + j] += update;
            if i != j {
                c[j * n + i] += update; // symmetric
            }
        }
    }

    Ok(())
}

/// Dot product of rows i and j of matrix a (n×k layout).
fn dot_row_row(a: &[f32], k: usize, i: usize, j: usize) -> f32 {
    let mut sum = 0.0_f32;
    for p in 0..k {
        sum += a[i * k + p] * a[j * k + p];
    }
    sum
}

/// Triangular matrix multiply: B = α·A·B (left side, lower triangular).
///
/// A is n×n lower triangular, B is n×nrhs (row-major).
///
/// # Errors
///
/// Returns error on dimension mismatch.
pub fn trmm(
    a: &[f32],
    b: &mut [f32],
    n: usize,
    nrhs: usize,
    alpha: f32,
) -> Result<(), SolverError> {
    validate_buffer(a, n * n, n, n)?;
    validate_buffer(b, n * nrhs, n, nrhs)?;

    // Process from bottom row to top to avoid overwriting
    // For lower triangular: row i depends on rows 0..=i
    // We compute new_b[i] = α · Σ_{j<=i} A[i,j] · old_b[j]
    // Process top-to-bottom, accumulate into temporary
    let mut temp = vec![0.0_f32; n * nrhs];

    for i in 0..n {
        for j in 0..=i {
            let a_val = alpha * a[i * n + j];
            for col in 0..nrhs {
                temp[i * nrhs + col] += a_val * b[j * nrhs + col];
            }
        }
    }

    b[..n * nrhs].copy_from_slice(&temp[..n * nrhs]);
    Ok(())
}

/// Symmetric matrix multiply: C = α·A·B + β·C
///
/// A is n×n symmetric (stored full), B is n×m, C is n×m (row-major).
///
/// # Errors
///
/// Returns error on dimension mismatch.
pub fn symm(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    m: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), SolverError> {
    validate_buffer(a, n * n, n, n)?;
    validate_buffer(b, n * m, n, m)?;
    validate_buffer(c, n * m, n, m)?;

    // C = β·C + α·A·B (standard GEMM since A is stored as full matrix)
    for i in 0..n {
        for j in 0..m {
            let mut sum = 0.0_f32;
            for p in 0..n {
                sum += a[i * n + p] * b[p * m + j];
            }
            c[i * m + j] = alpha * sum + beta * c[i * m + j];
        }
    }

    Ok(())
}

/// Validate buffer length matches expected dimensions.
fn validate_buffer(buf: &[f32], expected: usize, rows: usize, cols: usize) -> Result<(), SolverError> {
    if buf.len() != expected {
        return Err(SolverError::BufferLengthMismatch {
            expected,
            got: buf.len(),
            rows,
            cols,
        });
    }
    Ok(())
}
