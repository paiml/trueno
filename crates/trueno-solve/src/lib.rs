//! Dense linear algebra solvers with provable contracts.
//!
//! # Solvers
//!
//! - **LU**: Partial pivoting, PA = LU
//! - **QR**: Householder reflections, A = QR
//! - **SVD**: One-sided Jacobi, A = UΣV^T
//! - **Cholesky**: A = LL^T for positive definite matrices
//!
//! # Example
//!
//! ```
//! use trueno_solve::lu_factorize;
//!
//! // Solve Ax = b where A = [[2, 1], [1, 3]]
//! let a = [2.0, 1.0, 1.0, 3.0_f32];
//! let lu = lu_factorize(&a, 2).unwrap();
//! let x = lu.solve(&[5.0, 7.0]).unwrap();
//! // x ≈ [1.6, 1.8]
//! assert!((x[0] - 1.6).abs() < 1e-5);
//! assert!((x[1] - 1.8).abs() < 1e-5);
//! ```

mod blas3;
mod cholesky;
mod error;
mod lu;
mod qr;
mod svd;
mod trsm;

#[cfg(test)]
mod tests;

pub use blas3::{f32_to_f16, gemm_ex, gemm_strided_batched, symm, syr2k, syrk, trmm};
pub use cholesky::{cholesky, CholeskyFactorization};
pub use error::SolverError;
pub use lu::{lu_factorize, LuFactorization};
pub use qr::{qr_factorize, QrFactorization};
pub use svd::{svd, SvdResult};
pub use trsm::{trsm, DiagonalType, TriangularSide, TrsmResult};
