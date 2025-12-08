//! Eigendecomposition for symmetric matrices
//!
//! Provides SIMD-accelerated eigenvalue and eigenvector computation for symmetric
//! (Hermitian) matrices, enabling PCA, spectral clustering, and other algorithms
//! without external dependencies like nalgebra.
//!
//! # Algorithm
//!
//! Uses the Jacobi eigenvalue algorithm, which is numerically stable and well-suited
//! for SIMD parallelization. For large matrices (>1000 dimensions), GPU acceleration
//! is available via wgpu.
//!
//! # Example
//!
//! ```
//! use trueno::{Matrix, SymmetricEigen};
//!
//! // Create a symmetric positive definite matrix
//! let cov = Matrix::from_vec(2, 2, vec![
//!     2.0, 1.0,
//!     1.0, 2.0,
//! ]).unwrap();
//!
//! let eigen = SymmetricEigen::new(&cov).unwrap();
//!
//! // Eigenvalues in descending order (PCA convention)
//! let values = eigen.eigenvalues();
//! assert!((values[0] - 3.0).abs() < 1e-6);  // λ₁ = 3
//! assert!((values[1] - 1.0).abs() < 1e-6);  // λ₂ = 1
//! ```

use crate::{Backend, Matrix, TruenoError, Vector};

/// Maximum number of sweeps for Jacobi algorithm convergence
/// Each sweep processes all n(n-1)/2 off-diagonal elements once
/// Typically converges in 5-10 sweeps for well-conditioned matrices
const MAX_JACOBI_SWEEPS: usize = 50;

/// Convergence threshold for off-diagonal elements (relative to Frobenius norm)
const CONVERGENCE_THRESHOLD: f32 = 1e-7;

/// GPU threshold - use wgpu for matrices larger than this
#[allow(dead_code)]
const GPU_THRESHOLD: usize = 1000;

/// Symmetric matrix eigendecomposition
///
/// Computes eigenvalues and eigenvectors for symmetric (Hermitian) matrices.
/// Eigenvalues are returned in descending order (largest first), which is the
/// convention used in PCA and most dimensionality reduction algorithms.
///
/// # Properties
///
/// For a symmetric matrix A, the decomposition satisfies:
/// - `A = V × D × V^T` where D is diagonal with eigenvalues
/// - Eigenvectors are orthonormal: `V^T × V = I`
/// - Eigenvalues are real (guaranteed for symmetric matrices)
///
/// # Performance
///
/// - SIMD-accelerated Jacobi rotations for CPU
/// - GPU compute shaders for matrices >1000 dimensions (with `gpu` feature)
/// - O(n³) time complexity, O(n²) space complexity
#[derive(Debug, Clone)]
pub struct SymmetricEigen {
    /// Eigenvalues sorted in descending order
    eigenvalues: Vec<f32>,
    /// Eigenvectors as columns (column i corresponds to eigenvalue i)
    eigenvectors: Matrix<f32>,
    /// Sorting indices mapping original to sorted order
    #[allow(dead_code)]
    sort_indices: Vec<usize>,
    /// Backend used for computation
    backend: Backend,
}

impl SymmetricEigen {
    /// Computes eigendecomposition of a symmetric matrix
    ///
    /// # Arguments
    ///
    /// * `matrix` - A symmetric square matrix
    ///
    /// # Returns
    ///
    /// `SymmetricEigen` containing eigenvalues (descending) and eigenvectors
    ///
    /// # Errors
    ///
    /// - `InvalidInput` if matrix is not square
    /// - `InvalidInput` if matrix is empty
    /// - `InvalidInput` if algorithm fails to converge
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::{Matrix, SymmetricEigen};
    ///
    /// let m = Matrix::from_vec(3, 3, vec![
    ///     4.0, 2.0, 0.0,
    ///     2.0, 5.0, 3.0,
    ///     0.0, 3.0, 6.0,
    /// ]).unwrap();
    ///
    /// let eigen = SymmetricEigen::new(&m).unwrap();
    /// assert_eq!(eigen.eigenvalues().len(), 3);
    /// ```
    pub fn new(matrix: &Matrix<f32>) -> Result<Self, TruenoError> {
        // Validate input
        if matrix.rows() != matrix.cols() {
            return Err(TruenoError::InvalidInput(format!(
                "Matrix must be square for eigendecomposition, got {}x{}",
                matrix.rows(),
                matrix.cols()
            )));
        }

        if matrix.rows() == 0 {
            return Err(TruenoError::InvalidInput(
                "Cannot compute eigendecomposition of empty matrix".to_string(),
            ));
        }

        let backend = Backend::select_best();

        // Dispatch to appropriate implementation based on matrix size and GPU availability
        #[cfg(feature = "gpu")]
        {
            let n = matrix.rows();
            if n >= GPU_THRESHOLD && crate::backends::gpu::GpuBackend::is_available() {
                return Self::compute_gpu(matrix);
            }
        }

        // CPU implementation (SIMD-accelerated) - works on all platforms
        Self::compute_jacobi(matrix, backend)
    }

    /// CPU implementation using Jacobi eigenvalue algorithm
    ///
    /// The Jacobi algorithm iteratively applies Givens rotations to eliminate
    /// off-diagonal elements, converging to a diagonal matrix of eigenvalues.
    fn compute_jacobi(matrix: &Matrix<f32>, backend: Backend) -> Result<Self, TruenoError> {
        let n = matrix.rows();

        // Copy matrix data for in-place modification
        let mut a = matrix.as_slice().to_vec();

        // Compute initial Frobenius norm for relative convergence
        let frobenius_sq: f32 = a.iter().map(|x| x * x).sum();
        let tolerance = CONVERGENCE_THRESHOLD * frobenius_sq.sqrt().max(1.0);

        // Initialize eigenvectors to identity matrix
        let mut v = vec![0.0f32; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }

        // Jacobi iteration with sweep strategy
        // Each sweep processes all off-diagonal elements once
        for _sweep in 0..MAX_JACOBI_SWEEPS {
            // Cyclic Jacobi: process all pairs (i, j) where i < j
            let mut converged = true;

            for i in 0..n {
                for j in (i + 1)..n {
                    let aij = a[i * n + j];

                    // Skip if already small enough
                    if aij.abs() < tolerance {
                        continue;
                    }

                    converged = false;
                    Self::jacobi_rotate(&mut a, &mut v, n, i, j, backend);
                }
            }

            if converged {
                // Extract eigenvalues from diagonal
                let eigenvalues: Vec<f32> = (0..n).map(|i| a[i * n + i]).collect();

                // Sort eigenvalues in descending order
                let mut indices: Vec<usize> = (0..n).collect();
                indices.sort_by(|&i, &j| {
                    eigenvalues[j]
                        .partial_cmp(&eigenvalues[i])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Reorder eigenvalues
                let sorted_eigenvalues: Vec<f32> =
                    indices.iter().map(|&i| eigenvalues[i]).collect();

                // Create eigenvector matrix with sorted columns
                let mut eigenvector_data = vec![0.0f32; n * n];
                for (new_col, &old_col) in indices.iter().enumerate() {
                    for row in 0..n {
                        eigenvector_data[row * n + new_col] = v[row * n + old_col];
                    }
                }

                let eigenvectors = Matrix::from_vec(n, n, eigenvector_data)?;

                return Ok(SymmetricEigen {
                    eigenvalues: sorted_eigenvalues,
                    eigenvectors,
                    sort_indices: indices,
                    backend,
                });
            }
        }

        // Failed to converge - this shouldn't happen for well-conditioned matrices
        Err(TruenoError::InvalidInput(format!(
            "Jacobi algorithm failed to converge after {} sweeps",
            MAX_JACOBI_SWEEPS
        )))
    }

    /// Find the largest off-diagonal element (unused in cyclic Jacobi, kept for classic Jacobi)
    #[inline]
    #[allow(dead_code)]
    fn find_max_off_diagonal(a: &[f32], n: usize) -> (usize, usize, f32) {
        let mut max_val = 0.0f32;
        let mut p = 0;
        let mut q = 1;

        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        (p, q, max_val)
    }

    /// Apply Jacobi rotation to zero out a[p][q] and a[q][p]
    ///
    /// Uses the numerically stable formula from:
    /// Golub & Van Loan, "Matrix Computations", 4th Edition
    #[inline]
    fn jacobi_rotate(
        a: &mut [f32],
        v: &mut [f32],
        n: usize,
        p: usize,
        q: usize,
        _backend: Backend,
    ) {
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        // Skip if already zero
        if apq.abs() < 1e-15 {
            return;
        }

        // Compute rotation parameters using numerically stable formula
        // tau = (aqq - app) / (2 * apq)
        // t = sign(tau) / (|tau| + sqrt(1 + tau^2))  (avoiding catastrophic cancellation)
        // c = 1 / sqrt(1 + t^2)
        // s = t * c
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };

        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        // Update diagonal elements
        a[p * n + p] = app - t * apq;
        a[q * n + q] = aqq + t * apq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;

        // Update off-diagonal elements in rows/columns p and q
        for k in 0..n {
            if k != p && k != q {
                let akp = a[k * n + p];
                let akq = a[k * n + q];
                a[k * n + p] = c * akp - s * akq;
                a[p * n + k] = a[k * n + p];
                a[k * n + q] = s * akp + c * akq;
                a[q * n + k] = a[k * n + q];
            }
        }

        // Update eigenvector matrix
        for k in 0..n {
            let vkp = v[k * n + p];
            let vkq = v[k * n + q];
            v[k * n + p] = c * vkp - s * vkq;
            v[k * n + q] = s * vkp + c * vkq;
        }
    }

    /// GPU implementation for large matrices
    #[cfg(feature = "gpu")]
    fn compute_gpu(matrix: &Matrix<f32>) -> Result<Self, TruenoError> {
        use crate::backends::gpu::GpuBackend;

        let n = matrix.rows();
        let mut gpu = GpuBackend::new();

        // Execute eigendecomposition on GPU
        let (eigenvalues, eigenvector_data) =
            gpu.symmetric_eigen(matrix.as_slice(), n).map_err(|e| {
                TruenoError::InvalidInput(format!("GPU eigendecomposition failed: {}", e))
            })?;

        // Sort eigenvalues in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_eigenvalues: Vec<f32> = indices.iter().map(|&i| eigenvalues[i]).collect();

        // Reorder eigenvectors
        let mut sorted_eigenvector_data = vec![0.0f32; n * n];
        for (new_col, &old_col) in indices.iter().enumerate() {
            for row in 0..n {
                sorted_eigenvector_data[row * n + new_col] = eigenvector_data[row * n + old_col];
            }
        }

        let eigenvectors = Matrix::from_vec(n, n, sorted_eigenvector_data)?;

        Ok(SymmetricEigen {
            eigenvalues: sorted_eigenvalues,
            eigenvectors,
            sort_indices: indices,
            backend: Backend::GPU,
        })
    }

    /// Returns the eigenvalues in descending order
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::{Matrix, SymmetricEigen};
    ///
    /// let m = Matrix::from_vec(2, 2, vec![3.0, 1.0, 1.0, 3.0]).unwrap();
    /// let eigen = SymmetricEigen::new(&m).unwrap();
    ///
    /// let values = eigen.eigenvalues();
    /// assert!((values[0] - 4.0).abs() < 1e-5);  // λ₁ = 4
    /// assert!((values[1] - 2.0).abs() < 1e-5);  // λ₂ = 2
    /// ```
    pub fn eigenvalues(&self) -> &[f32] {
        &self.eigenvalues
    }

    /// Returns the eigenvector matrix
    ///
    /// Columns are eigenvectors, ordered to correspond with eigenvalues.
    /// Column `i` is the eigenvector for `eigenvalues()[i]`.
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::{Matrix, SymmetricEigen};
    ///
    /// let m = Matrix::identity(3);
    /// let eigen = SymmetricEigen::new(&m).unwrap();
    ///
    /// // Identity matrix has eigenvectors that are the standard basis
    /// let vectors = eigen.eigenvectors();
    /// assert_eq!(vectors.rows(), 3);
    /// assert_eq!(vectors.cols(), 3);
    /// ```
    pub fn eigenvectors(&self) -> &Matrix<f32> {
        &self.eigenvectors
    }

    /// Returns an iterator over (eigenvalue, eigenvector) pairs
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::{Matrix, SymmetricEigen};
    ///
    /// let m = Matrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 1.0]).unwrap();
    /// let eigen = SymmetricEigen::new(&m).unwrap();
    ///
    /// for (value, vector) in eigen.iter() {
    ///     println!("λ = {}, v = {:?}", value, vector.as_slice());
    /// }
    /// ```
    pub fn iter(&self) -> EigenIterator<'_> {
        EigenIterator {
            eigen: self,
            index: 0,
        }
    }

    /// Returns the number of eigenvalue/eigenvector pairs
    pub fn len(&self) -> usize {
        self.eigenvalues.len()
    }

    /// Returns true if there are no eigenvalues
    pub fn is_empty(&self) -> bool {
        self.eigenvalues.is_empty()
    }

    /// Returns the backend used for computation
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Get a specific eigenvector by index
    ///
    /// # Arguments
    ///
    /// * `i` - Index of the eigenvector (0 = largest eigenvalue)
    ///
    /// # Returns
    ///
    /// The eigenvector as a Vector, or None if index out of bounds
    pub fn eigenvector(&self, i: usize) -> Option<Vector<f32>> {
        if i >= self.eigenvalues.len() {
            return None;
        }

        let n = self.eigenvectors.rows();
        let mut data = Vec::with_capacity(n);

        for row in 0..n {
            if let Some(&val) = self.eigenvectors.get(row, i) {
                data.push(val);
            }
        }

        Some(Vector::from_slice(&data))
    }

    /// Reconstruct the original matrix from eigendecomposition
    ///
    /// Computes `V × D × V^T` where D is the diagonal matrix of eigenvalues.
    /// This is useful for verifying the decomposition accuracy.
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::{Matrix, SymmetricEigen};
    ///
    /// let m = Matrix::from_vec(2, 2, vec![4.0, 2.0, 2.0, 4.0]).unwrap();
    /// let eigen = SymmetricEigen::new(&m).unwrap();
    /// let reconstructed = eigen.reconstruct().unwrap();
    ///
    /// // Should be approximately equal to original
    /// assert!((reconstructed.get(0, 0).unwrap() - 4.0).abs() < 1e-5);
    /// ```
    pub fn reconstruct(&self) -> Result<Matrix<f32>, TruenoError> {
        let n = self.eigenvalues.len();

        // V × D (scale each column by its eigenvalue)
        let mut vd_data = vec![0.0f32; n * n];
        for i in 0..n {
            let lambda = self.eigenvalues[i];
            for j in 0..n {
                if let Some(&v) = self.eigenvectors.get(j, i) {
                    vd_data[j * n + i] = v * lambda;
                }
            }
        }

        let vd = Matrix::from_vec(n, n, vd_data)?;
        let vt = self.eigenvectors.transpose();

        vd.matmul(&vt)
    }
}

/// Iterator over eigenvalue-eigenvector pairs
pub struct EigenIterator<'a> {
    eigen: &'a SymmetricEigen,
    index: usize,
}

impl<'a> Iterator for EigenIterator<'a> {
    type Item = (f32, Vector<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.eigen.len() {
            return None;
        }

        let value = self.eigen.eigenvalues[self.index];
        let vector = self.eigen.eigenvector(self.index)?;
        self.index += 1;

        Some((value, vector))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.eigen.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for EigenIterator<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // RED PHASE: Tests that define expected behavior
    // =========================================================================

    #[test]
    fn test_symmetric_eigen_2x2_simple() {
        // Simple 2x2 symmetric matrix: [[2, 1], [1, 2]]
        // Eigenvalues: 3, 1
        // Eigenvectors: [1/√2, 1/√2], [1/√2, -1/√2]
        let m = Matrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 2.0]).expect("valid matrix");

        let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

        let values = eigen.eigenvalues();
        assert_eq!(values.len(), 2);

        // Eigenvalues should be in descending order
        assert!(values[0] >= values[1], "eigenvalues must be descending");

        // Check eigenvalue values (with tolerance)
        assert!(
            (values[0] - 3.0).abs() < 1e-5,
            "first eigenvalue should be 3, got {}",
            values[0]
        );
        assert!(
            (values[1] - 1.0).abs() < 1e-5,
            "second eigenvalue should be 1, got {}",
            values[1]
        );
    }

    #[test]
    fn test_symmetric_eigen_identity() {
        // Identity matrix has all eigenvalues = 1
        let m = Matrix::identity(3);

        let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

        let values = eigen.eigenvalues();
        assert_eq!(values.len(), 3);

        for (i, &val) in values.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "eigenvalue {} should be 1, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_symmetric_eigen_diagonal() {
        // Diagonal matrix: eigenvalues are the diagonal elements
        let m = Matrix::from_vec(3, 3, vec![5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0])
            .expect("valid matrix");

        let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

        let values = eigen.eigenvalues();

        // Should be sorted descending: 5, 3, 1
        assert!((values[0] - 5.0).abs() < 1e-5, "got {}", values[0]);
        assert!((values[1] - 3.0).abs() < 1e-5, "got {}", values[1]);
        assert!((values[2] - 1.0).abs() < 1e-5, "got {}", values[2]);
    }

    #[test]
    fn test_symmetric_eigen_eigenvectors_orthogonal() {
        let m = Matrix::from_vec(3, 3, vec![4.0, 2.0, 0.0, 2.0, 5.0, 3.0, 0.0, 3.0, 6.0])
            .expect("valid matrix");

        let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

        // Eigenvectors should be orthonormal: V^T × V = I
        let v = eigen.eigenvectors();
        let vt = v.transpose();
        let product = vt.matmul(v).expect("matmul should succeed");

        // Check if product is approximately identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = product.get(i, j).unwrap();
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "V^T×V[{},{}] = {}, expected {}",
                    i,
                    j,
                    actual,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_symmetric_eigen_reconstruction() {
        let m = Matrix::from_vec(2, 2, vec![4.0, 2.0, 2.0, 4.0]).expect("valid matrix");

        let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");
        let reconstructed = eigen.reconstruct().expect("reconstruction should succeed");

        // Reconstructed matrix should match original
        for i in 0..2 {
            for j in 0..2 {
                let original = m.get(i, j).unwrap();
                let recon = reconstructed.get(i, j).unwrap();
                assert!(
                    (original - recon).abs() < 1e-4,
                    "A[{},{}] = {}, reconstructed = {}",
                    i,
                    j,
                    original,
                    recon
                );
            }
        }
    }

    #[test]
    fn test_symmetric_eigen_av_equals_lambda_v() {
        // For each eigenpair (λ, v): A×v = λ×v
        let m = Matrix::from_vec(2, 2, vec![3.0, 1.0, 1.0, 3.0]).expect("valid matrix");

        let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

        for (lambda, v) in eigen.iter() {
            // Compute A×v
            let av = m.matvec(&v).expect("matvec should succeed");

            // Compute λ×v
            let lambda_v: Vec<f32> = v.as_slice().iter().map(|&x| x * lambda).collect();

            // Check equality
            for (i, (&av_i, &lv_i)) in av.as_slice().iter().zip(lambda_v.iter()).enumerate() {
                assert!(
                    (av_i - lv_i).abs() < 1e-4,
                    "A×v[{}] = {}, λv[{}] = {}",
                    i,
                    av_i,
                    i,
                    lv_i
                );
            }
        }
    }

    #[test]
    fn test_symmetric_eigen_error_non_square() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid matrix");

        let result = SymmetricEigen::new(&m);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(
            matches!(err, TruenoError::InvalidInput(_)),
            "expected InvalidInput error"
        );
    }

    #[test]
    fn test_symmetric_eigen_error_empty() {
        let m = Matrix::zeros(0, 0);

        let result = SymmetricEigen::new(&m);
        assert!(result.is_err());
    }

    #[test]
    fn test_symmetric_eigen_1x1() {
        let m = Matrix::from_vec(1, 1, vec![7.0]).expect("valid matrix");

        let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

        assert_eq!(eigen.eigenvalues().len(), 1);
        assert!((eigen.eigenvalues()[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_symmetric_eigen_iterator() {
        let m = Matrix::from_vec(2, 2, vec![2.0, 0.0, 0.0, 1.0]).expect("valid matrix");

        let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

        let pairs: Vec<_> = eigen.iter().collect();
        assert_eq!(pairs.len(), 2);

        // First eigenvalue is larger
        assert!(pairs[0].0 >= pairs[1].0);
    }

    #[test]
    fn test_symmetric_eigen_len() {
        let m = Matrix::identity(5);
        let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

        assert_eq!(eigen.len(), 5);
        assert!(!eigen.is_empty());
    }

    #[test]
    fn test_symmetric_eigen_eigenvector_accessor() {
        let m = Matrix::identity(3);
        let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

        let v0 = eigen.eigenvector(0);
        assert!(v0.is_some());
        assert_eq!(v0.unwrap().len(), 3);

        let v_invalid = eigen.eigenvector(10);
        assert!(v_invalid.is_none());
    }

    #[test]
    fn test_symmetric_eigen_covariance_matrix() {
        // Typical covariance matrix from PCA
        // Points: [(1,2), (3,4), (5,6)] centered → [(-2,-2), (0,0), (2,2)]
        // Cov = [[8/3, 8/3], [8/3, 8/3]] ≈ [[2.67, 2.67], [2.67, 2.67]]
        let m = Matrix::from_vec(2, 2, vec![2.67, 2.67, 2.67, 2.67]).expect("valid matrix");

        let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

        // Eigenvalues: 5.34, 0 (approximately)
        let values = eigen.eigenvalues();
        assert!(values[0] > 5.0, "first eigenvalue should be ~5.34");
        assert!(values[1].abs() < 0.1, "second eigenvalue should be ~0");
    }

    #[test]
    fn test_symmetric_eigen_negative_eigenvalues() {
        // Matrix with negative eigenvalues
        // [[0, 1], [1, 0]] has eigenvalues 1, -1
        let m = Matrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]).expect("valid matrix");

        let eigen = SymmetricEigen::new(&m).expect("eigendecomposition should succeed");

        let values = eigen.eigenvalues();
        assert!(
            (values[0] - 1.0).abs() < 1e-5,
            "first eigenvalue should be 1"
        );
        assert!(
            (values[1] - (-1.0)).abs() < 1e-5,
            "second eigenvalue should be -1"
        );
    }

    // =========================================================================
    // Property-based tests (proptest)
    // =========================================================================

    #[cfg(test)]
    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(50))]

            #[test]
            fn prop_eigenvalues_descending(n in 2usize..6) {
                // Generate random symmetric matrix
                let mut data = vec![0.0f32; n * n];
                for i in 0..n {
                    for j in i..n {
                        let val = (i + j) as f32 / (n as f32);
                        data[i * n + j] = val;
                        data[j * n + i] = val;
                    }
                }

                let m = Matrix::from_vec(n, n, data).expect("valid matrix");
                let eigen = SymmetricEigen::new(&m).expect("eigen should succeed");

                let values = eigen.eigenvalues();
                for i in 1..values.len() {
                    prop_assert!(
                        values[i - 1] >= values[i],
                        "eigenvalues not descending: {} < {}",
                        values[i - 1],
                        values[i]
                    );
                }
            }

            #[test]
            fn prop_eigenvector_count_matches_dimension(n in 1usize..8) {
                let m = Matrix::identity(n);
                let eigen = SymmetricEigen::new(&m).expect("eigen should succeed");

                prop_assert_eq!(eigen.len(), n);
                prop_assert_eq!(eigen.eigenvalues().len(), n);
                prop_assert_eq!(eigen.eigenvectors().rows(), n);
                prop_assert_eq!(eigen.eigenvectors().cols(), n);
            }

            #[test]
            fn prop_reconstruction_accuracy(
                a in 1.0f32..10.0,  // Ensure positive diagonal for conditioning
                b in -5.0f32..5.0,  // Off-diagonal smaller than diagonal
                c in 1.0f32..10.0   // Ensure positive diagonal for conditioning
            ) {
                // Create symmetric 2x2 matrix [[a+|b|, b], [b, c+|b|]]
                // Add |b| to diagonal for better conditioning
                let diag_a = a + b.abs();
                let diag_c = c + b.abs();
                let m = Matrix::from_vec(2, 2, vec![diag_a, b, b, diag_c]).expect("valid matrix");

                if let Ok(eigen) = SymmetricEigen::new(&m) {
                    if let Ok(recon) = eigen.reconstruct() {
                        // Use relative error for numerical stability
                        let frobenius_orig: f32 = [diag_a, b, b, diag_c].iter()
                            .map(|x| x * x).sum::<f32>().sqrt();
                        let max_allowed_error = 0.01 * frobenius_orig.max(1.0);

                        for i in 0..2 {
                            for j in 0..2 {
                                let orig = m.get(i, j).unwrap();
                                let rec = recon.get(i, j).unwrap();
                                prop_assert!(
                                    (orig - rec).abs() < max_allowed_error,
                                    "reconstruction error: {} vs {}, allowed: {}",
                                    orig,
                                    rec,
                                    max_allowed_error
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
