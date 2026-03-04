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
    // KAIZEN-040: Delegate to crate::blis::transpose which has AVX2 8×8
    // in-register micro-kernel with 64×64 L1-resident tiling and prefetch.
    // Previous implementation used scalar 32×32 blocks.
    #[cfg_attr(feature = "tracing", instrument(skip(self), fields(dims = %format!("{}x{}", self.rows, self.cols))))]
    pub fn transpose(&self) -> Matrix<f32> {
        let mut result = Matrix::zeros_with_backend(self.cols, self.rows, self.backend);

        // BLIS transpose handles AVX2 dispatch, remainder edges, and shape-adaptive
        // loop ordering internally. Dimensions are correct by construction so
        // the only possible error (size mismatch) cannot occur.
        if let Err(e) = crate::blis::transpose::transpose(
            self.rows,
            self.cols,
            &self.data,
            &mut result.data,
        ) {
            // Unreachable: result is allocated as cols×rows which matches rows×cols elements.
            // If somehow triggered, fall back to scalar element-wise transpose.
            debug_assert!(false, "BLIS transpose dimension mismatch: {e}");
            for i in 0..self.rows {
                for j in 0..self.cols {
                    result.data[j * self.rows + i] = self.data[i * self.cols + j];
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
mod tests;
