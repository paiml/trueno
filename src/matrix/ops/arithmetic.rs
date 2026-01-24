//! Matrix arithmetic operations
//!
//! This module provides matrix multiplication and related operations:
//! - `matmul()` - Standard matrix multiplication with SIMD optimization
//! - `batched_matmul()` - Batched 3D tensor multiplication
//! - `batched_matmul_4d()` - 4D tensor multiplication for attention
//!
//! ## Domain Separation (PMAT-018)
//!
//! Arithmetic operations (multiplication, addition) are separate from storage
//! operations (allocation, indexing). This allows optimizing compute kernels
//! independently of memory layout decisions.
//!
//! ## Performance Hierarchy
//!
//! 1. GPU for large matrices (≥500×500) - 2-10x speedup
//! 2. BLIS/SIMD for medium-large matrices (>64×64) - 2-8x speedup
//! 3. Naive for small matrices - lowest overhead

use crate::TruenoError;

#[cfg(feature = "tracing")]
use tracing::instrument;

use super::super::Matrix;

impl Matrix<f32> {
    /// Matrix multiplication (matmul)
    ///
    /// Computes `C = A × B` where A is `m×n`, B is `n×p`, and C is `m×p`.
    ///
    /// # Arguments
    ///
    /// * `other` - The matrix to multiply with (right operand)
    ///
    /// # Returns
    ///
    /// A new matrix containing the result of matrix multiplication
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if matrix dimensions are incompatible
    /// (i.e., `self.cols != other.rows`)
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    /// let c = a.matmul(&b).unwrap();
    ///
    /// // [[1, 2],   [[5, 6],   [[19, 22],
    /// //  [3, 4]] ×  [7, 8]] =  [43, 50]]
    /// assert_eq!(c.get(0, 0), Some(&19.0));
    /// assert_eq!(c.get(0, 1), Some(&22.0));
    /// assert_eq!(c.get(1, 0), Some(&43.0));
    /// assert_eq!(c.get(1, 1), Some(&50.0));
    /// ```
    // =========================================================================
    // HOT PATH - PERFORMANCE CRITICAL
    // =========================================================================
    // Core matrix operation used in neural network forward passes.
    // Changes to inner loops REQUIRE benchmark verification: make bench-check
    // =========================================================================
    #[cfg_attr(feature = "tracing", instrument(skip(self, other), fields(dims = %format!("{}x{} @ {}x{}", self.rows, self.cols, other.rows, other.cols))))]
    pub fn matmul(&self, other: &Matrix<f32>) -> Result<Matrix<f32>, TruenoError> {
        if self.cols != other.rows {
            return Err(TruenoError::InvalidInput(format!(
                "Matrix dimension mismatch for multiplication: {}×{} × {}×{} (inner dimensions {} and {} must match)",
                self.rows, self.cols, other.rows, other.cols, self.cols, other.rows
            )));
        }

        // Fast path for vector-matrix multiply (rows=1)
        if self.rows == 1 {
            return self.matmul_vector_matrix(other);
        }

        let mut result = Matrix::zeros_with_backend(self.rows, other.cols, self.backend);

        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = 500;
        const SIMD_THRESHOLD: usize = 64;

        // Try GPU first for very large matrices
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.rows >= GPU_THRESHOLD
                && self.cols >= GPU_THRESHOLD
                && other.cols >= GPU_THRESHOLD
            {
                if let Ok(gpu_result) = self.matmul_gpu(other) {
                    return Ok(gpu_result);
                }
            }
        }

        // Use SIMD for medium-large matrices
        if self.rows >= SIMD_THRESHOLD
            || self.cols >= SIMD_THRESHOLD
            || other.cols >= SIMD_THRESHOLD
        {
            #[cfg(target_arch = "wasm32")]
            {
                self.matmul_wasm_tiled(other, &mut result)?;
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                crate::blis::gemm_blis(
                    self.rows,
                    other.cols,
                    self.cols,
                    &self.data,
                    &other.data,
                    &mut result.data,
                    None,
                )?;
            }
        } else {
            self.matmul_naive(other, &mut result)?;
        }

        Ok(result)
    }

    /// Batched matrix multiplication for 3D tensors.
    ///
    /// Computes `[batch, m, k] @ [batch, k, n] -> [batch, m, n]` using SIMD for each batch.
    #[cfg_attr(
        feature = "tracing",
        instrument(skip(a_data, b_data), fields(batch, m, k, n))
    )]
    pub fn batched_matmul(
        a_data: &[f32],
        b_data: &[f32],
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, TruenoError> {
        let a_stride = m * k;
        let b_stride = k * n;
        let out_stride = m * n;

        if a_data.len() != batch * a_stride {
            return Err(TruenoError::InvalidInput(format!(
                "A data size mismatch: expected {} ({}×{}×{}), got {}",
                batch * a_stride, batch, m, k, a_data.len()
            )));
        }
        if b_data.len() != batch * b_stride {
            return Err(TruenoError::InvalidInput(format!(
                "B data size mismatch: expected {} ({}×{}×{}), got {}",
                batch * b_stride, batch, k, n, b_data.len()
            )));
        }

        let mut output = vec![0.0f32; batch * out_stride];

        for ba in 0..batch {
            let a_offset = ba * a_stride;
            let b_offset = ba * b_stride;
            let out_offset = ba * out_stride;

            let a_slice = &a_data[a_offset..a_offset + a_stride];
            let b_slice = &b_data[b_offset..b_offset + b_stride];

            let a_mat = Matrix::from_slice(m, k, a_slice)?;
            let b_mat = Matrix::from_slice(k, n, b_slice)?;

            let result = a_mat.matmul(&b_mat)?;
            output[out_offset..out_offset + out_stride].copy_from_slice(result.as_slice());
        }

        Ok(output)
    }

    /// Batched matrix multiplication for 4D tensors (attention pattern).
    ///
    /// Computes `[batch, heads, m, k] @ [batch, heads, k, n] -> [batch, heads, m, n]`
    #[cfg_attr(
        feature = "tracing",
        instrument(skip(a_data, b_data), fields(batch, heads, m, k, n))
    )]
    pub fn batched_matmul_4d(
        a_data: &[f32],
        b_data: &[f32],
        batch: usize,
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, TruenoError> {
        let a_head_stride = m * k;
        let b_head_stride = k * n;
        let out_head_stride = m * n;
        let total_heads = batch * heads;

        let expected_a = total_heads * a_head_stride;
        let expected_b = total_heads * b_head_stride;
        if a_data.len() != expected_a {
            return Err(TruenoError::InvalidInput(format!(
                "A data size mismatch: expected {} ({}×{}×{}×{}), got {}",
                expected_a, batch, heads, m, k, a_data.len()
            )));
        }
        if b_data.len() != expected_b {
            return Err(TruenoError::InvalidInput(format!(
                "B data size mismatch: expected {} ({}×{}×{}×{}), got {}",
                expected_b, batch, heads, k, n, b_data.len()
            )));
        }

        let mut output = vec![0.0f32; total_heads * out_head_stride];

        for bh in 0..total_heads {
            let a_offset = bh * a_head_stride;
            let b_offset = bh * b_head_stride;
            let out_offset = bh * out_head_stride;

            let a_slice = &a_data[a_offset..a_offset + a_head_stride];
            let b_slice = &b_data[b_offset..b_offset + b_head_stride];

            let a_mat = Matrix::from_slice(m, k, a_slice)?;
            let b_mat = Matrix::from_slice(k, n, b_slice)?;

            let result = a_mat.matmul(&b_mat)?;
            output[out_offset..out_offset + out_head_stride].copy_from_slice(result.as_slice());
        }

        Ok(output)
    }

    /// Fast path for vector-matrix multiplication (1×K @ K×N → 1×N)
    #[cfg_attr(feature = "tracing", instrument(skip(self, other), fields(k = self.cols, n = other.cols)))]
    fn matmul_vector_matrix(&self, other: &Matrix<f32>) -> Result<Matrix<f32>, TruenoError> {
        debug_assert_eq!(self.rows, 1);

        let k = self.cols;
        let n = other.cols;
        let mut result = Matrix::zeros_with_backend(1, n, self.backend);

        for ki in 0..k {
            let a_k = self.data[ki];
            if a_k == 0.0 {
                continue;
            }

            let b_row_start = ki * n;
            for j in 0..n {
                result.data[j] += a_k * other.data[b_row_start + j];
            }
        }

        Ok(result)
    }

    /// Naive O(n³) matrix multiplication (baseline for small matrices)
    fn matmul_naive(
        &self,
        other: &Matrix<f32>,
        result: &mut Matrix<f32>,
    ) -> Result<(), TruenoError> {
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k).expect("bounds validated")
                        * other.get(k, j).expect("bounds validated");
                }
                *result.get_mut(i, j).expect("bounds validated") = sum;
            }
        }
        Ok(())
    }

    /// WASM-optimized tiled matrix multiplication
    #[allow(dead_code)]
    fn matmul_wasm_tiled(
        &self,
        other: &Matrix<f32>,
        result: &mut Matrix<f32>,
    ) -> Result<(), TruenoError> {
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        for i in 0..m {
            let a_row_start = i * k;
            let result_row_start = i * n;

            let simd_width = 8;
            let n_simd = (n / simd_width) * simd_width;

            #[allow(clippy::needless_range_loop)]
            for j0 in (0..n_simd).step_by(simd_width) {
                let mut acc = [0.0f32; 8];

                for kk in 0..k {
                    let a_val = self.data[a_row_start + kk];
                    let b_row_start = kk * n + j0;

                    for jj in 0..simd_width {
                        acc[jj] += a_val * other.data[b_row_start + jj];
                    }
                }

                for jj in 0..simd_width {
                    result.data[result_row_start + j0 + jj] = acc[jj];
                }
            }

            for j in n_simd..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += self.data[a_row_start + kk] * other.data[kk * n + j];
                }
                result.data[result_row_start + j] = sum;
            }
        }

        Ok(())
    }

    /// GPU-accelerated matrix multiplication
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    fn matmul_gpu(&self, other: &Matrix<f32>) -> Result<Matrix<f32>, TruenoError> {
        use crate::backends::gpu::GpuBackend;

        if !GpuBackend::is_available() {
            return Err(TruenoError::InvalidInput("GPU not available".to_string()));
        }

        let mut gpu = GpuBackend::new();
        let result_data = gpu
            .matmul(&self.data, &other.data, self.rows, self.cols, other.cols)
            .map_err(|e| TruenoError::InvalidInput(format!("GPU matmul failed: {}", e)))?;

        let mut result = Matrix::zeros(self.rows, other.cols);
        result.data = result_data;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_basic() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.get(0, 0), Some(&19.0));
        assert_eq!(c.get(0, 1), Some(&22.0));
        assert_eq!(c.get(1, 0), Some(&43.0));
        assert_eq!(c.get(1, 1), Some(&50.0));
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        let a = Matrix::from_vec(2, 3, vec![1.0; 6]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![1.0; 4]).unwrap();
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn test_matmul_identity() {
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let i = Matrix::identity(2);
        let result = a.matmul(&i).unwrap();
        assert_eq!(result.as_slice(), a.as_slice());
    }

    #[test]
    fn test_batched_matmul() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 2 batches of 2×2
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]; // 2 identity matrices
        let result = Matrix::batched_matmul(&a, &b, 2, 2, 2, 2).unwrap();
        assert_eq!(result, a); // A × I = A
    }
}
