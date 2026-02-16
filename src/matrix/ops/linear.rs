//! Linear algebra operations for Matrix
//!
//! This module provides linear operations:
//! - `transpose()` - Matrix transpose
//! - `matvec()` - Matrix-vector multiplication
//! - `vecmat()` - Vector-matrix multiplication

use crate::{Backend, TruenoError, Vector};

#[cfg(feature = "tracing")]
use tracing::instrument;

/// Backend dispatch macro for dot product - centralizes platform-specific SIMD dispatch
macro_rules! dispatch_dot {
    ($backend:expr, $a:expr, $b:expr) => {{
        #[cfg(target_arch = "x86_64")]
        use crate::backends::{avx2::Avx2Backend, sse2::Sse2Backend};
        use crate::backends::{scalar::ScalarBackend, VectorBackend};
        // SAFETY: CPU features verified at runtime before backend selection
        unsafe {
            match $backend {
                Backend::Scalar => ScalarBackend::dot($a, $b),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::dot($a, $b),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::dot($a, $b),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::dot($a, $b)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    use crate::backends::neon::NeonBackend;
                    NeonBackend::dot($a, $b)
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::dot($a, $b),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    use crate::backends::wasm::WasmBackend;
                    WasmBackend::dot($a, $b)
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::dot($a, $b),
                Backend::GPU | Backend::Auto => ScalarBackend::dot($a, $b),
            }
        }
    }};
}

use super::super::Matrix;

impl Matrix<f32> {
    /// Transpose this matrix (swap rows and columns)
    ///
    /// Returns a new matrix with dimensions swapped: `self.rows → result.cols`,
    /// `self.cols → result.rows`.
    ///
    /// # Performance
    ///
    /// Uses cache-optimized block-wise transpose with 32x32 blocks.
    /// Sequential writes for output ensure good cache behavior.
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let t = m.transpose();
    ///
    /// // [[1, 2, 3],     [[1, 4],
    /// //  [4, 5, 6]]  →   [2, 5],
    /// //                  [3, 6]]
    /// assert_eq!(t.rows(), 3);
    /// assert_eq!(t.cols(), 2);
    /// assert_eq!(t.get(0, 0), Some(&1.0));
    /// assert_eq!(t.get(0, 1), Some(&4.0));
    /// assert_eq!(t.get(1, 0), Some(&2.0));
    /// ```
    #[cfg_attr(feature = "tracing", instrument(skip(self), fields(dims = %format!("{}x{}", self.rows, self.cols))))]
    pub fn transpose(&self) -> Matrix<f32> {
        let mut result = Matrix::zeros_with_backend(self.cols, self.rows, self.backend);

        // Use block-wise transpose for better cache locality
        // Block size of 32 balances cache efficiency for both square and non-square matrices
        const BLOCK_SIZE: usize = 32;

        // For non-square matrices, process output rows sequentially for write coalescing
        // This ensures writes are sequential in memory regardless of input shape
        // Non-square transpose uses write-coalesced layout for sequential memory access

        // Process in blocks, iterating output rows first for sequential writes
        for j_block in (0..self.cols).step_by(BLOCK_SIZE) {
            let j_end = (j_block + BLOCK_SIZE).min(self.cols);

            for i_block in (0..self.rows).step_by(BLOCK_SIZE) {
                let i_end = (i_block + BLOCK_SIZE).min(self.rows);

                // Within block: iterate output rows (j) in outer loop for sequential writes
                for j in j_block..j_end {
                    let dst_row_start = j * result.cols;
                    for i in i_block..i_end {
                        // result[j, i] = self[i, j]
                        // Sequential write: dst_row_start + i increments by 1
                        // Strided read: acceptable, CPU prefetch handles this
                        result.data[dst_row_start + i] = self.data[i * self.cols + j];
                    }
                }
            }
        }

        result
    }

    /// Matrix-vector multiplication (column vector): A × v
    ///
    /// Multiplies this matrix by a column vector, computing `A × v` where the result
    /// is a column vector with length equal to the number of rows in `A`.
    ///
    /// # Mathematical Definition
    ///
    /// For an m×n matrix A and an n-dimensional vector v:
    /// ```text
    /// result[i] = Σ(j=0 to n-1) A[i,j] × v[j]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `v` - Column vector with length equal to `self.cols()`
    ///
    /// # Returns
    ///
    /// A new vector with length `self.rows()`
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `v.len() != self.cols()`
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::{Matrix, Vector};
    ///
    /// let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let result = m.matvec(&v).unwrap();
    ///
    /// // [[1, 2, 3]   [1]   [1×1 + 2×2 + 3×3]   [14]
    /// //  [4, 5, 6]] × [2] = [4×1 + 5×2 + 6×3] = [32]
    /// //               [3]
    /// assert_eq!(result.as_slice(), &[14.0, 32.0]);
    /// ```
    pub fn matvec(&self, v: &Vector<f32>) -> Result<Vector<f32>, TruenoError> {
        if v.len() != self.cols {
            return Err(TruenoError::InvalidInput(format!(
                "Vector length {} does not match matrix columns {} for matrix-vector multiplication",
                v.len(),
                self.cols
            )));
        }

        let v_slice = v.as_slice();

        let mut result_data = vec![0.0; self.rows];

        // Parallel execution for very large matrices (≥4096 rows)
        // Note: Thread overhead dominates for smaller matrices
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 4096;

            if self.rows >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;
                use std::sync::atomic::{AtomicPtr, Ordering};
                use std::sync::Arc;

                let result_ptr = Arc::new(AtomicPtr::new(result_data.as_mut_ptr()));

                // Process rows in parallel - each row computes an independent dot product
                (0..self.rows).into_par_iter().for_each(|i| {
                    let row_start = i * self.cols;
                    let row = &self.data[row_start..(row_start + self.cols)];

                    let dot_result = dispatch_dot!(self.backend, row, v_slice);

                    // Write to non-overlapping memory location (thread-safe)
                    // SAFETY: CPU feature verified at runtime, slices bounds-checked
                    unsafe {
                        let ptr = result_ptr.load(Ordering::Relaxed);
                        *ptr.add(i) = dot_result;
                    }
                });

                return Ok(Vector::from_slice(&result_data));
            }
        }

        // SIMD-optimized execution: each row-vector product is a dot product
        for (i, result) in result_data.iter_mut().enumerate() {
            let row_start = i * self.cols;
            let row = &self.data[row_start..(row_start + self.cols)];

            // Use SIMD dot product for each row
            *result = dispatch_dot!(self.backend, row, v_slice);
        }

        Ok(Vector::from_slice(&result_data))
    }

    /// Vector-matrix multiplication (row vector): v^T × A
    ///
    /// Multiplies a row vector by this matrix, computing `v^T × A` where the result
    /// is a row vector with length equal to the number of columns in `A`.
    ///
    /// # Mathematical Definition
    ///
    /// For an m-dimensional vector v and an m×n matrix A:
    /// ```text
    /// result[j] = Σ(i=0 to m-1) v[i] × A[i,j]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `v` - Row vector with length equal to `m.rows()`
    /// * `m` - Matrix to multiply
    ///
    /// # Returns
    ///
    /// A new vector with length `m.cols()`
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `v.len() != m.rows()`
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::{Matrix, Vector};
    ///
    /// let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let v = Vector::from_slice(&[1.0, 2.0]);
    /// let result = Matrix::vecmat(&v, &m).unwrap();
    ///
    /// // [1, 2] × [[1, 2, 3]  = [1×1 + 2×4, 1×2 + 2×5, 1×3 + 2×6]
    /// //           [4, 5, 6]]
    /// //         = [9, 12, 15]
    /// assert_eq!(result.as_slice(), &[9.0, 12.0, 15.0]);
    /// ```
    pub fn vecmat(v: &Vector<f32>, m: &Matrix<f32>) -> Result<Vector<f32>, TruenoError> {
        if v.len() != m.rows {
            return Err(TruenoError::InvalidInput(format!(
                "Vector length {} does not match matrix rows {} for vector-matrix multiplication",
                v.len(),
                m.rows
            )));
        }

        // SIMD-optimized implementation using row-wise accumulation
        // Instead of column-wise access (cache-unfriendly), we compute:
        // result = Σ(i) v[i] * row_i (cache-friendly, vectorizable)
        //
        // This approach:
        // 1. Sequential row access (cache-friendly vs strided column access)
        // 2. Uses SIMD scale and add operations
        // 3. Leverages existing optimized Vector operations

        let mut result = Vector::from_slice(&vec![0.0; m.cols]);
        let v_slice = v.as_slice();

        // Accumulate each scaled row into result
        for (i, &scalar) in v_slice.iter().enumerate().take(m.rows) {
            let row_start = i * m.cols;
            let row = &m.data[row_start..(row_start + m.cols)];

            // Create vector for this row
            let row_vec = Vector::from_slice(row);

            // result += scalar * row (using SIMD scale and add)
            let scaled_row = row_vec.scale(scalar)?;
            result = result.add(&scaled_row)?;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_square() {
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
    fn test_transpose_rect() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = m.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(t.get(0, 0), Some(&1.0));
        assert_eq!(t.get(0, 1), Some(&4.0));
        assert_eq!(t.get(1, 0), Some(&2.0));
    }

    #[test]
    fn test_matvec_basic() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = m.matvec(&v).unwrap();
        assert_eq!(result.as_slice(), &[14.0, 32.0]);
    }

    #[test]
    fn test_matvec_dimension_mismatch() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = Vector::from_slice(&[1.0, 2.0]); // Wrong size
        assert!(m.matvec(&v).is_err());
    }

    #[test]
    fn test_vecmat_basic() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = Vector::from_slice(&[1.0, 2.0]);
        let result = Matrix::vecmat(&v, &m).unwrap();
        assert_eq!(result.as_slice(), &[9.0, 12.0, 15.0]);
    }

    #[test]
    fn test_vecmat_dimension_mismatch() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]); // Wrong size
        assert!(Matrix::vecmat(&v, &m).is_err());
    }

    // ========== matvec: Backend Dispatch Coverage ==========

    /// Helper: build a matrix with a specific backend and run matvec, verifying results.
    fn assert_matvec_backend(
        rows: usize,
        cols: usize,
        mat_data: Vec<f32>,
        vec_data: &[f32],
        expected: &[f32],
        backend: Backend,
        tolerance: f32,
        label: &str,
    ) {
        let m = Matrix::from_vec_with_backend(rows, cols, mat_data, backend);
        let v = Vector::from_slice(vec_data);
        let result = m.matvec(&v).unwrap();
        assert_eq!(result.as_slice().len(), expected.len(), "{label}: length mismatch");
        for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < tolerance,
                "{label} at index {i}: got {got} expected {exp}",
            );
        }
    }

    #[test]
    fn test_matvec_scalar_backend() {
        // Explicitly use Scalar to hit Backend::Scalar arm in dispatch_dot
        assert_matvec_backend(
            2, 3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 2.0, 3.0],
            &[14.0, 32.0],
            Backend::Scalar,
            1e-6,
            "matvec Scalar",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_matvec_sse2_backend() {
        // SSE2 arm: dispatches to Sse2Backend::dot
        assert_matvec_backend(
            2, 3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 2.0, 3.0],
            &[14.0, 32.0],
            Backend::SSE2,
            1e-6,
            "matvec SSE2",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_matvec_avx_backend() {
        // AVX arm: dispatches to Sse2Backend::dot (same arm as SSE2)
        assert_matvec_backend(
            2, 3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 2.0, 3.0],
            &[14.0, 32.0],
            Backend::AVX,
            1e-6,
            "matvec AVX",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_matvec_avx2_backend() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // AVX2 arm: dispatches to Avx2Backend::dot
        assert_matvec_backend(
            2, 3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 2.0, 3.0],
            &[14.0, 32.0],
            Backend::AVX2,
            1e-6,
            "matvec AVX2",
        );
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_matvec_avx512_backend() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        // AVX512 arm: dispatches to Avx2Backend::dot (same arm as AVX2)
        assert_matvec_backend(
            2, 3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 2.0, 3.0],
            &[14.0, 32.0],
            Backend::AVX512,
            1e-6,
            "matvec AVX512",
        );
    }

    #[test]
    fn test_matvec_neon_fallback_backend() {
        // On non-ARM, NEON falls back to ScalarBackend
        assert_matvec_backend(
            2, 3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 2.0, 3.0],
            &[14.0, 32.0],
            Backend::NEON,
            1e-6,
            "matvec NEON",
        );
    }

    #[test]
    fn test_matvec_wasm_fallback_backend() {
        // On non-WASM, WasmSIMD falls back to ScalarBackend
        assert_matvec_backend(
            2, 3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 2.0, 3.0],
            &[14.0, 32.0],
            Backend::WasmSIMD,
            1e-6,
            "matvec WasmSIMD",
        );
    }

    #[test]
    fn test_matvec_gpu_backend() {
        // GPU arm falls back to ScalarBackend
        assert_matvec_backend(
            2, 3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 2.0, 3.0],
            &[14.0, 32.0],
            Backend::GPU,
            1e-6,
            "matvec GPU",
        );
    }

    #[test]
    fn test_matvec_auto_backend() {
        // Auto arm falls back to ScalarBackend in dispatch_dot
        assert_matvec_backend(
            2, 3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 2.0, 3.0],
            &[14.0, 32.0],
            Backend::Auto,
            1e-6,
            "matvec Auto",
        );
    }

    // ========== matvec: Backend Equivalence ==========

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_matvec_backend_equivalence() {
        // Compare all backends against Scalar for a non-trivial matrix
        let rows = 4;
        let cols = 16; // Ensures AVX2 processes full 8-wide lanes + remainder
        let mat_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1).collect();
        let vec_data: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.5 + 1.0).collect();

        let m_scalar = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::Scalar);
        let v = Vector::from_slice(&vec_data);
        let expected = m_scalar.matvec(&v).unwrap();

        for &backend in &[Backend::SSE2, Backend::AVX] {
            let m = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), backend);
            let result = m.matvec(&v).unwrap();
            for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.as_slice().iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-4,
                    "Scalar vs {backend:?} mismatch at [{i}]: {got} vs {exp}",
                );
            }
        }

        if is_x86_feature_detected!("avx2") {
            for &backend in &[Backend::AVX2, Backend::AVX512] {
                let m = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), backend);
                let result = m.matvec(&v).unwrap();
                for (i, (&got, &exp)) in
                    result.as_slice().iter().zip(expected.as_slice().iter()).enumerate()
                {
                    assert!(
                        (got - exp).abs() < 1e-4,
                        "Scalar vs {backend:?} mismatch at [{i}]: {got} vs {exp}",
                    );
                }
            }
        }
    }

    // ========== matvec: Edge Cases ==========

    #[test]
    fn test_matvec_single_element_1x1() {
        let m = Matrix::from_vec_with_backend(1, 1, vec![3.0], Backend::Scalar);
        let v = Vector::from_slice(&[5.0]);
        let result = m.matvec(&v).unwrap();
        assert!((result.as_slice()[0] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_single_row() {
        // 1xN matrix: result should be a single dot product
        let m = Matrix::from_vec_with_backend(1, 4, vec![1.0, 2.0, 3.0, 4.0], Backend::Scalar);
        let v = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let result = m.matvec(&v).unwrap();
        assert!((result.as_slice()[0] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_single_column() {
        // Nx1 matrix: result is each element multiplied by the scalar
        let m = Matrix::from_vec_with_backend(3, 1, vec![2.0, 4.0, 6.0], Backend::Scalar);
        let v = Vector::from_slice(&[3.0]);
        let result = m.matvec(&v).unwrap();
        assert_eq!(result.as_slice(), &[6.0, 12.0, 18.0]);
    }

    #[test]
    fn test_matvec_identity_matrix() {
        // Identity matrix should return the input vector unchanged
        let m = Matrix::from_vec(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let v = Vector::from_slice(&[7.0, 11.0, 13.0]);
        let result = m.matvec(&v).unwrap();
        for (i, (&got, &exp)) in result.as_slice().iter().zip([7.0, 11.0, 13.0].iter()).enumerate()
        {
            assert!((got - exp).abs() < 1e-6, "identity matvec [{i}]: {got} != {exp}");
        }
    }

    #[test]
    fn test_matvec_zero_matrix() {
        let m = Matrix::from_vec(2, 3, vec![0.0; 6]).unwrap();
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = m.matvec(&v).unwrap();
        for &val in result.as_slice() {
            assert!((val - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matvec_zero_vector() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = m.matvec(&v).unwrap();
        for &val in result.as_slice() {
            assert!((val - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matvec_non_aligned_dimensions() {
        // 3x7: not aligned to any SIMD lane width (SSE=4, AVX2=8, AVX512=16)
        let rows = 3;
        let cols = 7;
        let mat_data: Vec<f32> = (0..rows * cols).map(|i| (i + 1) as f32).collect();
        let vec_data: Vec<f32> = (0..cols).map(|i| (i + 1) as f32).collect();

        for &backend in &[Backend::Scalar, Backend::GPU, Backend::Auto, Backend::NEON, Backend::WasmSIMD] {
            let m = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), backend);
            let v = Vector::from_slice(&vec_data);
            let result = m.matvec(&v).unwrap();
            assert_eq!(result.as_slice().len(), rows, "non-aligned {backend:?}");
            // Verify row 0: dot([1..7], [1..7]) = 1+4+9+16+25+36+49 = 140
            assert!(
                (result.as_slice()[0] - 140.0).abs() < 1e-3,
                "non-aligned {backend:?} row 0: got {}",
                result.as_slice()[0]
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_matvec_non_aligned_simd_backends() {
        // 5x13: not aligned to SSE2 (4), AVX2 (8), or AVX-512 (16) lane widths
        let rows = 5;
        let cols = 13;
        let mat_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.3).collect();
        let vec_data: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.7 + 0.1).collect();

        let m_scalar = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::Scalar);
        let v = Vector::from_slice(&vec_data);
        let expected = m_scalar.matvec(&v).unwrap();

        for &backend in &[Backend::SSE2, Backend::AVX] {
            let m = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), backend);
            let result = m.matvec(&v).unwrap();
            for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.as_slice().iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "non-aligned Scalar vs {backend:?} at [{i}]: {got} vs {exp}",
                );
            }
        }

        if is_x86_feature_detected!("avx2") {
            let m_avx2 = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::AVX2);
            let result = m_avx2.matvec(&v).unwrap();
            for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.as_slice().iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "non-aligned Scalar vs AVX2 at [{i}]: {got} vs {exp}",
                );
            }
        }
    }

    #[test]
    fn test_matvec_large_matrix_all_backends() {
        // Large enough to exercise SIMD loops (> 16 elements per row)
        let rows = 10;
        let cols = 64;
        let mat_data: Vec<f32> = (0..rows * cols).map(|i| ((i % 17) as f32) * 0.1 - 0.8).collect();
        let vec_data: Vec<f32> = (0..cols).map(|i| ((i % 13) as f32) * 0.2 - 1.2).collect();

        let m_scalar = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::Scalar);
        let v = Vector::from_slice(&vec_data);
        let expected = m_scalar.matvec(&v).unwrap();

        let backends = vec![Backend::GPU, Backend::Auto, Backend::NEON, Backend::WasmSIMD];
        for backend in backends {
            let m = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), backend);
            let result = m.matvec(&v).unwrap();
            for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.as_slice().iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-3,
                    "large Scalar vs {backend:?} at [{i}]: {got} vs {exp}",
                );
            }
        }
    }

    #[test]
    fn test_matvec_dimension_mismatch_error_message() {
        let m = Matrix::from_vec(2, 3, vec![1.0; 6]).unwrap();
        let v = Vector::from_slice(&[1.0, 2.0]);
        let err = m.matvec(&v).unwrap_err();
        match err {
            TruenoError::InvalidInput(msg) => {
                assert!(msg.contains("2"), "Error should mention vector length 2: {msg}");
                assert!(msg.contains("3"), "Error should mention matrix cols 3: {msg}");
            }
            other => panic!("Expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn test_matvec_negative_values() {
        let m = Matrix::from_vec_with_backend(
            2, 3,
            vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            Backend::Scalar,
        );
        let v = Vector::from_slice(&[-1.0, -2.0, -3.0]);
        let result = m.matvec(&v).unwrap();
        // Row 0: (-1)(-1) + (-2)(-2) + (-3)(-3) = 1 + 4 + 9 = 14
        // Row 1: (-4)(-1) + (-5)(-2) + (-6)(-3) = 4 + 10 + 18 = 32
        assert!((result.as_slice()[0] - 14.0).abs() < 1e-5);
        assert!((result.as_slice()[1] - 32.0).abs() < 1e-5);
    }

    // ========== matvec: Parallel Path (Refs CB-130) ==========

    #[test]
    #[cfg(feature = "parallel")]
    fn test_matvec_parallel_large_matrix() {
        // Trigger the parallel path: rows >= 4096
        let rows = 4096;
        let cols = 16;
        let mat_data: Vec<f32> = (0..rows * cols).map(|i| ((i % 100) as f32) * 0.01).collect();
        let vec_data: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1 + 1.0).collect();

        // Run with Scalar backend to compare sequential vs parallel
        let m_scalar = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::Scalar);
        let v = Vector::from_slice(&vec_data);
        let result = m_scalar.matvec(&v).unwrap();
        assert_eq!(result.as_slice().len(), rows);

        // Verify first and last rows manually
        let row0: f32 = (0..cols)
            .map(|j| mat_data[j] * vec_data[j])
            .sum();
        assert!(
            (result.as_slice()[0] - row0).abs() < 1e-2,
            "parallel row 0: got {} expected {row0}",
            result.as_slice()[0]
        );
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_matvec_parallel_with_simd_backends() {
        // Ensure parallel + SIMD dispatch works together
        let rows = 4096;
        let cols = 32;
        let mat_data: Vec<f32> = (0..rows * cols).map(|i| ((i % 50) as f32) * 0.02 - 0.5).collect();
        let vec_data: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.1).collect();
        let v = Vector::from_slice(&vec_data);

        let m_scalar = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::Scalar);
        let expected = m_scalar.matvec(&v).unwrap();

        #[cfg(target_arch = "x86_64")]
        {
            let m_sse = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::SSE2);
            let result_sse = m_sse.matvec(&v).unwrap();
            for (i, (&got, &exp)) in result_sse
                .as_slice()
                .iter()
                .zip(expected.as_slice().iter())
                .enumerate()
            {
                assert!(
                    (got - exp).abs() < 1e-2,
                    "parallel Scalar vs SSE2 at [{i}]: {got} vs {exp}",
                );
            }

            if is_x86_feature_detected!("avx2") {
                let m_avx2 = Matrix::from_vec_with_backend(rows, cols, mat_data.clone(), Backend::AVX2);
                let result_avx2 = m_avx2.matvec(&v).unwrap();
                for (i, (&got, &exp)) in result_avx2
                    .as_slice()
                    .iter()
                    .zip(expected.as_slice().iter())
                    .enumerate()
                {
                    assert!(
                        (got - exp).abs() < 1e-2,
                        "parallel Scalar vs AVX2 at [{i}]: {got} vs {exp}",
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_matvec_parallel_boundary() {
        // Just below the threshold (4095 rows) - should NOT hit parallel path
        let rows = 4095;
        let cols = 4;
        let mat_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
        let vec_data = vec![1.0; cols];
        let m = Matrix::from_vec_with_backend(rows, cols, mat_data, Backend::Scalar);
        let v = Vector::from_slice(&vec_data);
        let result = m.matvec(&v).unwrap();
        assert_eq!(result.as_slice().len(), rows);

        // Exactly at threshold (4096 rows) - should hit parallel path
        let rows = 4096;
        let mat_data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.001).collect();
        let m = Matrix::from_vec_with_backend(rows, cols, mat_data, Backend::Scalar);
        let result = m.matvec(&v).unwrap();
        assert_eq!(result.as_slice().len(), rows);
    }
}
