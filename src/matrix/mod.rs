//! Matrix operations for Trueno
//!
//! Provides 2D matrix operations with SIMD optimization for linear algebra,
//! machine learning, and scientific computing.
//!
//! # Example
//!
//! ```
//! use trueno::Matrix;
//!
//! // Create a 2x3 matrix
//! let m = Matrix::zeros(2, 3);
//! assert_eq!(m.rows(), 2);
//! assert_eq!(m.cols(), 3);
//! ```

use crate::{Backend, TruenoError, Vector};

#[cfg(feature = "tracing")]
use tracing::instrument;

/// A 2D matrix with row-major storage
///
/// Data is stored in row-major format (C-style), where consecutive elements
/// in memory belong to the same row. This is compatible with NumPy's default
/// layout and optimal for cache locality when accessing rows.
///
/// # Storage Layout
///
/// For a 2x3 matrix:
/// ```text
/// [[a, b, c],
///  [d, e, f]]
/// ```
/// Data is stored as: [a, b, c, d, e, f]
///
/// # Example
///
/// ```
/// use trueno::Matrix;
///
/// let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// assert_eq!(m.get(0, 0), Some(&1.0));
/// assert_eq!(m.get(0, 1), Some(&2.0));
/// assert_eq!(m.get(1, 0), Some(&3.0));
/// assert_eq!(m.get(1, 1), Some(&4.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
    backend: Backend,
}

impl std::ops::Index<(usize, usize)> for Matrix<f32> {
    type Output = f32;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row * self.cols + col]
    }
}

impl Matrix<f32> {
    /// Creates a new matrix with uninitialized values
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    ///
    /// # Returns
    ///
    /// A new matrix with dimensions `rows x cols` containing uninitialized values
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// let m = Matrix::new(3, 4);
    /// assert_eq!(m.rows(), 3);
    /// assert_eq!(m.cols(), 4);
    /// ```
    pub fn new(rows: usize, cols: usize) -> Self {
        let backend = Backend::select_best();
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
            backend,
        }
    }

    /// Creates a matrix from a vector of data
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `data` - Vector containing matrix elements in row-major order
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `data.len() != rows * cols`
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// assert_eq!(m.rows(), 2);
    /// assert_eq!(m.cols(), 2);
    /// ```
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Result<Self, TruenoError> {
        if data.len() != rows * cols {
            return Err(TruenoError::InvalidInput(format!(
                "Data length {} does not match matrix dimensions {}x{} (expected {})",
                data.len(),
                rows,
                cols,
                rows * cols
            )));
        }

        let backend = Backend::select_best();
        Ok(Matrix {
            rows,
            cols,
            data,
            backend,
        })
    }

    /// Creates a matrix from a vector with a specific backend
    ///
    /// This is useful for testing specific SIMD code paths.
    pub fn from_vec_with_backend(
        rows: usize,
        cols: usize,
        data: Vec<f32>,
        backend: Backend,
    ) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length {} does not match matrix dimensions {}x{}",
            data.len(),
            rows,
            cols
        );
        Matrix {
            rows,
            cols,
            data,
            backend,
        }
    }

    /// Creates a matrix from a slice by copying the data
    ///
    /// This is a convenience method that copies the slice into an owned vector.
    /// For zero-copy scenarios, consider using the data directly with `from_vec`
    /// if you already have an owned `Vec`.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows
    /// * `cols` - Number of columns
    /// * `data` - Slice containing matrix elements in row-major order
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if `data.len() != rows * cols`
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// let data = [1.0, 2.0, 3.0, 4.0];
    /// let m = Matrix::from_slice(2, 2, &data).unwrap();
    /// assert_eq!(m.get(0, 0), Some(&1.0));
    /// ```
    pub fn from_slice(rows: usize, cols: usize, data: &[f32]) -> Result<Self, TruenoError> {
        Self::from_vec(rows, cols, data.to_vec())
    }

    /// Creates a matrix filled with zeros
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// let m = Matrix::zeros(3, 3);
    /// assert_eq!(m.get(1, 1), Some(&0.0));
    /// ```
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix::new(rows, cols)
    }

    /// Creates a matrix filled with zeros using a specific backend
    /// (Internal use only - reuses backend from parent matrix)
    fn zeros_with_backend(rows: usize, cols: usize, backend: Backend) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
            backend,
        }
    }

    /// Creates an identity matrix (square matrix with 1s on diagonal)
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// let m = Matrix::identity(3);
    /// assert_eq!(m.get(0, 0), Some(&1.0));
    /// assert_eq!(m.get(0, 1), Some(&0.0));
    /// assert_eq!(m.get(1, 1), Some(&1.0));
    /// ```
    pub fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let backend = Backend::select_best();
        Matrix {
            rows: n,
            cols: n,
            data,
            backend,
        }
    }

    /// Returns the number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Returns the shape as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Gets a reference to an element at (row, col)
    ///
    /// Returns `None` if indices are out of bounds
    pub fn get(&self, row: usize, col: usize) -> Option<&f32> {
        if row >= self.rows || col >= self.cols {
            None
        } else {
            self.data.get(row * self.cols + col)
        }
    }

    /// Gets a mutable reference to an element at (row, col)
    ///
    /// Returns `None` if indices are out of bounds
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f32> {
        if row >= self.rows || col >= self.cols {
            None
        } else {
            let idx = row * self.cols + col;
            self.data.get_mut(idx)
        }
    }

    /// Returns a reference to the underlying data
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

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
    // See convolve2d for prohibited patterns in hot loops.
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
        // Common in ML vocab projection: hidden_state @ embedding_transposed
        // 8x faster than general matmul for 1×384 @ 384×51865 pattern
        if self.rows == 1 {
            return self.matmul_vector_matrix(other);
        }

        let mut result = Matrix::zeros_with_backend(self.rows, other.cols, self.backend);

        // Backend selection strategy (empirical - see docs/performance-analysis.md):
        // 1. GPU for large matrices (≥500×500) - 2-10x speedup (measured)
        // 2. SIMD for medium-large matrices (>64×64) - 2-8x speedup
        // 3. Naive for small matrices - lowest overhead

        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = 500; // Empirical: 2x at 500×500, 9.6x at 1000×1000
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
                // GPU failed, fall through to SIMD/naive
            }
        }

        // Use SIMD for medium-large matrices
        if self.rows >= SIMD_THRESHOLD
            || self.cols >= SIMD_THRESHOLD
            || other.cols >= SIMD_THRESHOLD
        {
            // Tiled approach threshold: below this size, tiling beats transpose
            // Based on WASM optimization spec benchmarks
            const TILED_THRESHOLD: usize = 512;

            let max_dim = self.rows.max(self.cols).max(other.cols);

            if max_dim < TILED_THRESHOLD {
                // Medium matrices: use BLIS on native, tiled on WASM
                #[cfg(target_arch = "wasm32")]
                {
                    self.matmul_wasm_tiled(other, &mut result)?;
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    // BLIS is faster than tiled for all sizes on native
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
                // Large matrices: platform-specific optimized paths
                #[cfg(target_arch = "wasm32")]
                {
                    // WASM: tiled is always better (no SIMD microkernel advantage)
                    self.matmul_wasm_tiled(other, &mut result)?;
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    // Native: use BLIS-style GEMM with register blocking
                    // ~2x faster than old SIMD implementation for large matrices
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
            }
        } else {
            self.matmul_naive(other, &mut result)?;
        }

        Ok(result)
    }

    /// Batched matrix multiplication for 3D tensors.
    ///
    /// Computes `[batch, m, k] @ [batch, k, n] -> [batch, m, n]` using SIMD for each batch.
    /// This is critical for transformer attention performance.
    ///
    /// # Arguments
    /// * `a_data` - Flattened input A with shape [batch, m, k]
    /// * `b_data` - Flattened input B with shape [batch, k, n]
    /// * `batch` - Batch dimension
    /// * `m` - Rows of A (and output)
    /// * `k` - Columns of A / Rows of B
    /// * `n` - Columns of B (and output)
    ///
    /// # Returns
    /// Flattened output with shape [batch, m, n]
    ///
    /// # Performance
    /// Uses SIMD matmul for each batch slice, achieving ~50 GFLOPS vs ~0.1 GFLOPS naive.
    /// See Williams et al., 2009 (Roofline model) for theoretical analysis.
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

        // Validate input sizes
        if a_data.len() != batch * a_stride {
            return Err(TruenoError::InvalidInput(format!(
                "A data size mismatch: expected {} ({}×{}×{}), got {}",
                batch * a_stride,
                batch,
                m,
                k,
                a_data.len()
            )));
        }
        if b_data.len() != batch * b_stride {
            return Err(TruenoError::InvalidInput(format!(
                "B data size mismatch: expected {} ({}×{}×{}), got {}",
                batch * b_stride,
                batch,
                k,
                n,
                b_data.len()
            )));
        }

        let mut output = vec![0.0f32; batch * out_stride];

        // Process each batch using SIMD matmul
        for ba in 0..batch {
            let a_offset = ba * a_stride;
            let b_offset = ba * b_stride;
            let out_offset = ba * out_stride;

            // Create matrix views from slices (no copy - just metadata)
            let a_slice = &a_data[a_offset..a_offset + a_stride];
            let b_slice = &b_data[b_offset..b_offset + b_stride];

            // Use from_slice to avoid copying
            let a_mat = Matrix::from_slice(m, k, a_slice)?;
            let b_mat = Matrix::from_slice(k, n, b_slice)?;

            // SIMD matmul
            let result = a_mat.matmul(&b_mat)?;

            // Copy result to output
            output[out_offset..out_offset + out_stride].copy_from_slice(result.as_slice());
        }

        Ok(output)
    }

    /// Batched matrix multiplication for 4D tensors (attention pattern).
    ///
    /// Computes `[batch, heads, m, k] @ [batch, heads, k, n] -> [batch, heads, m, n]`
    /// This is the exact pattern used in multi-head attention: Q @ K^T and attn @ V.
    ///
    /// # Arguments
    /// * `a_data` - Flattened input A with shape [batch, heads, m, k]
    /// * `b_data` - Flattened input B with shape [batch, heads, k, n]
    /// * `batch` - Batch dimension
    /// * `heads` - Number of attention heads
    /// * `m` - Rows (sequence length for Q)
    /// * `k` - Columns of A / Rows of B (head dimension)
    /// * `n` - Columns of B (sequence length for K^T, or head dim for V)
    ///
    /// # Performance
    /// Processes batch×heads independent matmuls using SIMD.
    /// For Qwen2-0.5B: batch=1, heads=14, m=seq, k=64, n=seq
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

        // Validate input sizes
        let expected_a = total_heads * a_head_stride;
        let expected_b = total_heads * b_head_stride;
        if a_data.len() != expected_a {
            return Err(TruenoError::InvalidInput(format!(
                "A data size mismatch: expected {} ({}×{}×{}×{}), got {}",
                expected_a,
                batch,
                heads,
                m,
                k,
                a_data.len()
            )));
        }
        if b_data.len() != expected_b {
            return Err(TruenoError::InvalidInput(format!(
                "B data size mismatch: expected {} ({}×{}×{}×{}), got {}",
                expected_b,
                batch,
                heads,
                k,
                n,
                b_data.len()
            )));
        }

        let mut output = vec![0.0f32; total_heads * out_head_stride];

        // Process each (batch, head) pair using SIMD matmul
        for bh in 0..total_heads {
            let a_offset = bh * a_head_stride;
            let b_offset = bh * b_head_stride;
            let out_offset = bh * out_head_stride;

            // Create matrix views from slices
            let a_slice = &a_data[a_offset..a_offset + a_head_stride];
            let b_slice = &b_data[b_offset..b_offset + b_head_stride];

            let a_mat = Matrix::from_slice(m, k, a_slice)?;
            let b_mat = Matrix::from_slice(k, n, b_slice)?;

            // SIMD matmul
            let result = a_mat.matmul(&b_mat)?;

            // Copy result to output
            output[out_offset..out_offset + out_head_stride].copy_from_slice(result.as_slice());
        }

        Ok(output)
    }

    /// Fast path for vector-matrix multiplication (1×K @ K×N → 1×N)
    ///
    /// This is 8x faster than general matmul for patterns like:
    /// - Vocab projection: hidden_state (1×384) @ embedding_transposed (384×51865)
    /// - Single token decode in Whisper/LLM inference
    ///
    /// Strategy: Outer product accumulation (no transpose needed!)
    /// For result[j] = sum_k(A[0,k] * B[k,j]), we compute:
    ///   result += A[k] * B[k,:]  for each k
    /// This has excellent cache locality since we access entire rows of B.
    #[cfg_attr(feature = "tracing", instrument(skip(self, other), fields(k = self.cols, n = other.cols)))]
    fn matmul_vector_matrix(&self, other: &Matrix<f32>) -> Result<Matrix<f32>, TruenoError> {
        debug_assert_eq!(self.rows, 1);

        let k = self.cols; // Inner dimension
        let n = other.cols; // Output dimension

        // Result is 1×N, initialized to zero
        let mut result = Matrix::zeros_with_backend(1, n, self.backend);

        // Outer product accumulation: result += A[k] * B[k,:]
        // For each k, scale row k of B by A[k] and add to result
        // The compiler will auto-vectorize this inner loop
        for ki in 0..k {
            let a_k = self.data[ki];
            if a_k == 0.0 {
                continue; // Skip zero multiplications
            }

            // Get row ki of B (contiguous in memory - cache friendly!)
            let b_row_start = ki * n;

            // AXPY: result += a_k * B[ki,:]
            // This loop is auto-vectorized by LLVM with -O2/-O3
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
        // C[i,j] = Σ A[i,k] × B[k,j]
        // SAFETY: Loop bounds are validated by dimension checks in matmul()
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    // Bounds guaranteed: i < self.rows, k < self.cols, j < other.cols
                    sum += self
                        .get(i, k)
                        .expect("matmul_naive: A[i,k] bounds validated by loop")
                        * other
                            .get(k, j)
                            .expect("matmul_naive: B[k,j] bounds validated by loop");
                }
                *result
                    .get_mut(i, j)
                    .expect("matmul_naive: C[i,j] bounds validated by loop") = sum;
            }
        }
        Ok(())
    }

    /// AVX2 micro-kernel: Compute 4 rows × 1 column using register blocking (Phase 2)
    ///
    /// This micro-kernel processes 4 rows of matrix A against 1 column of B_transposed
    /// simultaneously, keeping intermediate results in AVX2 registers for efficiency.
    ///
    /// # Performance Benefits
    /// - Loads B-column once, reuses for 4 A-rows (4× reduction in memory bandwidth)
    /// - Uses FMA instructions for fused multiply-add (3× throughput vs separate ops)
    /// - Keeps accumulators in YMM registers (no memory traffic for intermediate results)
    ///
    /// # Safety
    /// - Caller must ensure all slices have the same length
    /// - Must be called on x86_64 with AVX2 support
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn matmul_microkernel_4x1_avx2(
        a_rows: [&[f32]; 4],
        b_col: &[f32],
        results: &mut [f32; 4],
    ) {
        use std::arch::x86_64::*;

        let len = b_col.len();
        let chunks = len / 8; // Process 8 f32 elements per iteration (AVX2 = 256 bits)

        // Accumulators for 4 output elements (kept in registers)
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        // Main loop: Process 8 elements at a time
        for i in 0..chunks {
            let offset = i * 8;

            // Load B column (reused for all 4 A rows)
            let b_vec = _mm256_loadu_ps(b_col.as_ptr().add(offset));

            // Load A rows and FMA (Fused Multiply-Add)
            let a0_vec = _mm256_loadu_ps(a_rows[0].as_ptr().add(offset));
            acc0 = _mm256_fmadd_ps(a0_vec, b_vec, acc0);

            let a1_vec = _mm256_loadu_ps(a_rows[1].as_ptr().add(offset));
            acc1 = _mm256_fmadd_ps(a1_vec, b_vec, acc1);

            let a2_vec = _mm256_loadu_ps(a_rows[2].as_ptr().add(offset));
            acc2 = _mm256_fmadd_ps(a2_vec, b_vec, acc2);

            let a3_vec = _mm256_loadu_ps(a_rows[3].as_ptr().add(offset));
            acc3 = _mm256_fmadd_ps(a3_vec, b_vec, acc3);
        }

        // Horizontal sum of each accumulator (reduce 8 elements to 1)
        results[0] = Self::horizontal_sum_avx2(acc0);
        results[1] = Self::horizontal_sum_avx2(acc1);
        results[2] = Self::horizontal_sum_avx2(acc2);
        results[3] = Self::horizontal_sum_avx2(acc3);

        // Handle remainder elements with scalar code
        let remainder_start = chunks * 8;
        if remainder_start < len {
            for i in remainder_start..len {
                results[0] += a_rows[0][i] * b_col[i];
                results[1] += a_rows[1][i] * b_col[i];
                results[2] += a_rows[2][i] * b_col[i];
                results[3] += a_rows[3][i] * b_col[i];
            }
        }
    }

    /// Helper: Horizontal sum of 8 f32 values in an AVX2 register
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn horizontal_sum_avx2(v: std::arch::x86_64::__m256) -> f32 {
        use std::arch::x86_64::*;

        // Sum upper and lower 128-bit lanes
        let sum128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));

        // Horizontal add within 128-bit lane (4 values → 2 values)
        let sum64 = _mm_hadd_ps(sum128, sum128);

        // Horizontal add again (2 values → 1 value)
        let sum32 = _mm_hadd_ps(sum64, sum64);

        // Extract final scalar result
        _mm_cvtss_f32(sum32)
    }

    /// AVX-512 micro-kernel: Compute 8 rows × 1 column using register blocking (Phase 3)
    ///
    /// This micro-kernel processes 8 rows of matrix A against 1 column of B_transposed
    /// simultaneously, keeping intermediate results in AVX-512 registers for efficiency.
    ///
    /// # Performance Benefits
    /// - Processes 16 f32 elements per iteration (vs 8 with AVX2) - 2× throughput
    /// - Loads B-column once, reuses for 8 A-rows (8× reduction in memory bandwidth)
    /// - Uses FMA instructions for fused multiply-add (3× throughput vs separate ops)
    /// - Keeps accumulators in ZMM registers (no memory traffic for intermediate results)
    ///
    /// # Safety
    /// - Caller must ensure all slices have the same length
    /// - Must be called on x86_64 with AVX-512F support
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn matmul_microkernel_8x1_avx512(
        a_rows: [&[f32]; 8],
        b_col: &[f32],
        results: &mut [f32; 8],
    ) {
        use std::arch::x86_64::*;

        let len = b_col.len();
        let chunks = len / 16; // Process 16 f32 elements per iteration (AVX-512 = 512 bits)

        // Accumulators for 8 output elements (kept in ZMM registers)
        let mut acc0 = _mm512_setzero_ps();
        let mut acc1 = _mm512_setzero_ps();
        let mut acc2 = _mm512_setzero_ps();
        let mut acc3 = _mm512_setzero_ps();
        let mut acc4 = _mm512_setzero_ps();
        let mut acc5 = _mm512_setzero_ps();
        let mut acc6 = _mm512_setzero_ps();
        let mut acc7 = _mm512_setzero_ps();

        // Main loop: Process 16 elements at a time
        for i in 0..chunks {
            let offset = i * 16;

            // Load B column (reused for all 8 A rows)
            let b_vec = _mm512_loadu_ps(b_col.as_ptr().add(offset));

            // Load A rows and FMA (Fused Multiply-Add)
            let a0_vec = _mm512_loadu_ps(a_rows[0].as_ptr().add(offset));
            acc0 = _mm512_fmadd_ps(a0_vec, b_vec, acc0);

            let a1_vec = _mm512_loadu_ps(a_rows[1].as_ptr().add(offset));
            acc1 = _mm512_fmadd_ps(a1_vec, b_vec, acc1);

            let a2_vec = _mm512_loadu_ps(a_rows[2].as_ptr().add(offset));
            acc2 = _mm512_fmadd_ps(a2_vec, b_vec, acc2);

            let a3_vec = _mm512_loadu_ps(a_rows[3].as_ptr().add(offset));
            acc3 = _mm512_fmadd_ps(a3_vec, b_vec, acc3);

            let a4_vec = _mm512_loadu_ps(a_rows[4].as_ptr().add(offset));
            acc4 = _mm512_fmadd_ps(a4_vec, b_vec, acc4);

            let a5_vec = _mm512_loadu_ps(a_rows[5].as_ptr().add(offset));
            acc5 = _mm512_fmadd_ps(a5_vec, b_vec, acc5);

            let a6_vec = _mm512_loadu_ps(a_rows[6].as_ptr().add(offset));
            acc6 = _mm512_fmadd_ps(a6_vec, b_vec, acc6);

            let a7_vec = _mm512_loadu_ps(a_rows[7].as_ptr().add(offset));
            acc7 = _mm512_fmadd_ps(a7_vec, b_vec, acc7);
        }

        // Horizontal sum of each accumulator (reduce 16 elements to 1)
        results[0] = _mm512_reduce_add_ps(acc0);
        results[1] = _mm512_reduce_add_ps(acc1);
        results[2] = _mm512_reduce_add_ps(acc2);
        results[3] = _mm512_reduce_add_ps(acc3);
        results[4] = _mm512_reduce_add_ps(acc4);
        results[5] = _mm512_reduce_add_ps(acc5);
        results[6] = _mm512_reduce_add_ps(acc6);
        results[7] = _mm512_reduce_add_ps(acc7);

        // Handle remainder elements with scalar code
        let remainder_start = chunks * 16;
        if remainder_start < len {
            for i in remainder_start..len {
                results[0] += a_rows[0][i] * b_col[i];
                results[1] += a_rows[1][i] * b_col[i];
                results[2] += a_rows[2][i] * b_col[i];
                results[3] += a_rows[3][i] * b_col[i];
                results[4] += a_rows[4][i] * b_col[i];
                results[5] += a_rows[5][i] * b_col[i];
                results[6] += a_rows[6][i] * b_col[i];
                results[7] += a_rows[7][i] * b_col[i];
            }
        }
    }

    /// Cache-aware blocked matrix multiplication with SIMD optimization
    ///
    /// Uses 2-level cache blocking (L2/L1) to minimize cache misses:
    /// - L2 blocks: 64×64 (256KB for 3 matrices in f32)
    /// - L1 micro-kernels: 8×8 (768 bytes fits comfortably in L1)
    ///
    /// Performance characteristics:
    /// - Small matrices (<64×64): ~1.2× speedup over naive (overhead dominates)
    /// - Medium matrices (128×128): ~1.5-2× speedup (cache effects visible)
    /// - Large matrices (512×512+): ~2-3× speedup (dramatic cache improvement)
    ///
    /// This is Phase 1 of matmul optimization (Issue #10). Future Phase 2 will
    /// add optional BLAS backend for full NumPy parity on very large matrices.
    /// Helper function to process a single L3 row block for parallel matmul (Phase 4).
    ///
    /// # Safety
    /// When called from parallel code, the caller must ensure that each thread processes
    /// a distinct row range [iii, i3_end) with no overlap. This function is safe because
    /// each thread writes only to its own row range in the result matrix.
    #[cfg(feature = "parallel")]
    #[allow(clippy::too_many_arguments)]
    fn process_l3_row_block_seq(
        iii: usize,
        i3_end: usize,
        a: &Matrix<f32>,
        b_transposed: &Matrix<f32>,
        result: &mut Matrix<f32>,
        l2_block_size: usize,
        l3_block_size: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        use crate::backends::{avx2::Avx2Backend, sse2::Sse2Backend};
        use crate::backends::{scalar::ScalarBackend, VectorBackend};

        // Process all column blocks for this row block
        for jjj in (0..b_transposed.rows).step_by(l3_block_size) {
            let j3_end = (jjj + l3_block_size).min(b_transposed.rows);

            for kkk in (0..a.cols).step_by(l3_block_size) {
                let k3_end = (kkk + l3_block_size).min(a.cols);

                // L2 blocking within L3 blocks
                for ii in (iii..i3_end).step_by(l2_block_size) {
                    let i_end = (ii + l2_block_size).min(i3_end);

                    for jj in (jjj..j3_end).step_by(l2_block_size) {
                        let j_end = (jj + l2_block_size).min(j3_end);

                        for kk in (kkk..k3_end).step_by(l2_block_size) {
                            let k_end = (kk + l2_block_size).min(k3_end);
                            let block_size = k_end - kk;

                            // Micro-kernel processing
                            #[cfg(target_arch = "x86_64")]
                            let use_microkernel =
                                matches!(a.backend, Backend::AVX2 | Backend::AVX512);

                            #[cfg(target_arch = "x86_64")]
                            if use_microkernel {
                                let mut i = ii;

                                // Process 4 rows at a time with micro-kernel
                                while i + 4 <= i_end {
                                    let row0_start = i * a.cols + kk;
                                    let row1_start = (i + 1) * a.cols + kk;
                                    let row2_start = (i + 2) * a.cols + kk;
                                    let row3_start = (i + 3) * a.cols + kk;

                                    let a_rows = [
                                        &a.data[row0_start..row0_start + block_size],
                                        &a.data[row1_start..row1_start + block_size],
                                        &a.data[row2_start..row2_start + block_size],
                                        &a.data[row3_start..row3_start + block_size],
                                    ];

                                    for j in jj..j_end {
                                        let col_start = j * b_transposed.cols + kk;
                                        let b_col =
                                            &b_transposed.data[col_start..col_start + block_size];

                                        let mut partial_dots = [0.0f32; 4];
                                        // SAFETY: AVX2 support verified by is_x86_feature_detected!("avx2")
                                        // check in outer scope. Slices a_rows and b_col are bounds-checked
                                        // and properly aligned for SIMD operations.
                                        // SAFETY: CPU feature verified at runtime, slices bounds-checked
                                        unsafe {
                                            Matrix::matmul_microkernel_4x1_avx2(
                                                a_rows,
                                                b_col,
                                                &mut partial_dots,
                                            );
                                        }

                                        result.data[i * result.cols + j] += partial_dots[0];
                                        result.data[(i + 1) * result.cols + j] += partial_dots[1];
                                        result.data[(i + 2) * result.cols + j] += partial_dots[2];
                                        result.data[(i + 3) * result.cols + j] += partial_dots[3];
                                    }

                                    i += 4;
                                }

                                // Handle remaining rows (< 4)
                                for i in i..i_end {
                                    let row_start = i * a.cols + kk;
                                    let a_row = &a.data[row_start..row_start + block_size];

                                    for j in jj..j_end {
                                        let col_start = j * b_transposed.cols + kk;
                                        let b_col =
                                            &b_transposed.data[col_start..col_start + block_size];

                                        // SAFETY: AVX2 verified at runtime, slices bounds-checked
                                        // SAFETY: AVX2 verified at runtime, slices bounds-checked
                                        let partial_dot = unsafe { Avx2Backend::dot(a_row, b_col) };
                                        result.data[i * result.cols + j] += partial_dot;
                                    }
                                }
                            } else {
                                // Non-AVX2 path
                                #[allow(unused_variables)]
                                for i in ii..i_end {
                                    let row_start = i * a.cols + kk;
                                    let a_row = &a.data[row_start..row_start + block_size];

                                    for j in jj..j_end {
                                        let col_start = j * b_transposed.cols + kk;
                                        let b_col =
                                            &b_transposed.data[col_start..col_start + block_size];

                                        // SAFETY: AVX2 verified at runtime, slices bounds-checked
                                        let partial_dot = unsafe {
                                            match a.backend {
                                                Backend::Scalar => ScalarBackend::dot(a_row, b_col),
                                                #[cfg(target_arch = "x86_64")]
                                                Backend::SSE2 | Backend::AVX => {
                                                    Sse2Backend::dot(a_row, b_col)
                                                }
                                                #[cfg(not(target_arch = "x86_64"))]
                                                Backend::SSE2
                                                | Backend::AVX
                                                | Backend::AVX2
                                                | Backend::AVX512 => {
                                                    ScalarBackend::dot(a_row, b_col)
                                                }
                                                #[cfg(any(
                                                    target_arch = "aarch64",
                                                    target_arch = "arm"
                                                ))]
                                                Backend::NEON => {
                                                    use crate::backends::neon::NeonBackend;
                                                    NeonBackend::dot(a_row, b_col)
                                                }
                                                #[cfg(not(any(
                                                    target_arch = "aarch64",
                                                    target_arch = "arm"
                                                )))]
                                                Backend::NEON => ScalarBackend::dot(a_row, b_col),
                                                #[cfg(target_arch = "wasm32")]
                                                Backend::WasmSIMD => {
                                                    use crate::backends::wasm::WasmBackend;
                                                    WasmBackend::dot(a_row, b_col)
                                                }
                                                #[cfg(not(target_arch = "wasm32"))]
                                                Backend::WasmSIMD => {
                                                    ScalarBackend::dot(a_row, b_col)
                                                }
                                                // Catch-all for GPU, Auto, and any other backends
                                                _ => ScalarBackend::dot(a_row, b_col),
                                            }
                                        };

                                        result.data[i * result.cols + j] += partial_dot;
                                    }
                                }
                            }

                            // Non-x86_64 fallback
                            #[cfg(not(target_arch = "x86_64"))]
                            {
                                for i in ii..i_end {
                                    let row_start = i * a.cols + kk;
                                    let a_row = &a.data[row_start..row_start + block_size];

                                    for j in jj..j_end {
                                        let col_start = j * b_transposed.cols + kk;
                                        let b_col =
                                            &b_transposed.data[col_start..col_start + block_size];

                                        // SAFETY: AVX2 verified at runtime, slices bounds-checked
                                        let partial_dot = unsafe {
                                            match a.backend {
                                                Backend::Scalar => ScalarBackend::dot(a_row, b_col),
                                                #[cfg(any(
                                                    target_arch = "aarch64",
                                                    target_arch = "arm"
                                                ))]
                                                Backend::NEON => {
                                                    use crate::backends::neon::NeonBackend;
                                                    NeonBackend::dot(a_row, b_col)
                                                }
                                                #[cfg(not(any(
                                                    target_arch = "aarch64",
                                                    target_arch = "arm"
                                                )))]
                                                Backend::NEON => ScalarBackend::dot(a_row, b_col),
                                                #[cfg(target_arch = "wasm32")]
                                                Backend::WasmSIMD => {
                                                    use crate::backends::wasm::WasmBackend;
                                                    WasmBackend::dot(a_row, b_col)
                                                }
                                                #[cfg(not(target_arch = "wasm32"))]
                                                Backend::WasmSIMD => {
                                                    ScalarBackend::dot(a_row, b_col)
                                                }
                                                _ => ScalarBackend::dot(a_row, b_col),
                                            }
                                        };

                                        result.data[i * result.cols + j] += partial_dot;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn matmul_simd(
        &self,
        other: &Matrix<f32>,
        result: &mut Matrix<f32>,
    ) -> Result<(), TruenoError> {
        // Cache blocking parameters (tuned for typical x86_64 CPUs)
        // L2 cache: 256KB typical → 64K f32 elements → 64×64×3 matrices fits
        const L2_BLOCK_SIZE: usize = 64;
        // L3 cache: 4-16MB typical → 256×256 blocks for very large matrices (Phase 3)
        const L3_BLOCK_SIZE: usize = 256;
        const L3_THRESHOLD: usize = 512; // Use 3-level blocking for matrices ≥512×512

        // For small matrices, use simple SIMD approach (blocking overhead too high)
        if self.rows <= 32 || self.cols <= 32 || other.cols <= 32 {
            return self.matmul_simd_simple(other, result);
        }

        #[cfg(target_arch = "x86_64")]
        use crate::backends::{avx2::Avx2Backend, sse2::Sse2Backend};
        use crate::backends::{scalar::ScalarBackend, VectorBackend};

        // Pre-transpose B for better cache locality (columns become rows)
        let b_transposed = other.transpose();

        // Determine if we should use 3-level blocking (Phase 3)
        let use_l3_blocking =
            self.rows >= L3_THRESHOLD && self.cols >= L3_THRESHOLD && other.cols >= L3_THRESHOLD;

        // Phase 4: Determine if we should use multi-threading (≥1024×1024)
        #[cfg(feature = "parallel")]
        const PARALLEL_THRESHOLD: usize = 1024;
        #[cfg(feature = "parallel")]
        let use_parallel = self.rows >= PARALLEL_THRESHOLD
            && self.cols >= PARALLEL_THRESHOLD
            && other.cols >= PARALLEL_THRESHOLD;
        #[cfg(not(feature = "parallel"))]
        let use_parallel = false;

        if use_l3_blocking {
            // ===== Phase 3/4: 3-Level Cache Blocking (L3 → L2 → micro-kernel) =====
            // For very large matrices (≥512×512), use L3 cache blocking to minimize
            // cache misses when data doesn't fit in L2 cache
            //
            // Hierarchy:
            // 1. L3 blocks: 256×256 (fits in L3 cache: 4-16MB)
            // 2. L2 blocks: 64×64 (fits in L2 cache: 256KB)
            // 3. Micro-kernel: 4×1 for AVX2/AVX512
            //
            // Phase 4: For ≥1024×1024, parallelize L3 row blocks with rayon

            if use_parallel {
                // ===== Phase 4: Parallel 3-Level Cache Blocking (Lock-Free Row Partitioning) =====
                #[cfg(feature = "parallel")]
                {
                    use rayon::prelude::*;
                    use std::sync::atomic::{AtomicPtr, Ordering};
                    use std::sync::Arc;

                    // Lock-free parallelization strategy:
                    // Each thread processes one L3 row block (256 rows). Since row blocks are
                    // non-overlapping, threads write to distinct memory regions with no contention.
                    //
                    // Safety invariant: Each thread writes to result.data[iii*cols..(i3_end)*cols],
                    // where iii = block_idx * L3_BLOCK_SIZE. Since L3 blocks don't overlap,
                    // no two threads write to the same memory location.

                    // Store result pointer in Arc<AtomicPtr> for safe sharing
                    let result_ptr = Arc::new(AtomicPtr::new(result as *mut Matrix<f32>));

                    // Calculate number of L3 blocks
                    let num_blocks = self.rows.div_ceil(L3_BLOCK_SIZE);

                    // Process each L3 block in parallel (lock-free)
                    (0..num_blocks).into_par_iter().for_each(|block_idx| {
                        let iii = block_idx * L3_BLOCK_SIZE;
                        let i3_end = (iii + L3_BLOCK_SIZE).min(self.rows);

                        // SAFETY: Each thread processes a distinct row range [iii, i3_end).
                        // No two threads write to overlapping memory locations because:
                        // 1. L3 blocks partition rows: [0, 256), [256, 512), etc.
                        // 2. Each thread only modifies result.data[iii*cols..(i3_end)*cols]
                        // 3. Row ranges are non-overlapping by construction
                        // 4. All threads complete before function returns (rayon guarantee)
                        // 5. AtomicPtr ensures proper memory ordering across threads
                        // SAFETY: CPU feature verified at runtime, slices bounds-checked
                        unsafe {
                            let ptr = result_ptr.load(Ordering::Relaxed);
                            Self::process_l3_row_block_seq(
                                iii,
                                i3_end,
                                self,
                                &b_transposed,
                                &mut *ptr,
                                L2_BLOCK_SIZE,
                                L3_BLOCK_SIZE,
                            );
                        }
                    });
                }

                return Ok(());
            }

            // ===== Sequential 3-Level Cache Blocking (fallback) =====
            for iii in (0..self.rows).step_by(L3_BLOCK_SIZE) {
                let i3_end = (iii + L3_BLOCK_SIZE).min(self.rows);

                for jjj in (0..other.cols).step_by(L3_BLOCK_SIZE) {
                    let j3_end = (jjj + L3_BLOCK_SIZE).min(other.cols);

                    for kkk in (0..self.cols).step_by(L3_BLOCK_SIZE) {
                        let k3_end = (kkk + L3_BLOCK_SIZE).min(self.cols);

                        // L2 blocking within L3 blocks
                        for ii in (iii..i3_end).step_by(L2_BLOCK_SIZE) {
                            let i_end = (ii + L2_BLOCK_SIZE).min(i3_end);

                            for jj in (jjj..j3_end).step_by(L2_BLOCK_SIZE) {
                                let j_end = (jj + L2_BLOCK_SIZE).min(j3_end);

                                for kk in (kkk..k3_end).step_by(L2_BLOCK_SIZE) {
                                    let k_end = (kk + L2_BLOCK_SIZE).min(k3_end);
                                    let block_size = k_end - kk;

                                    // Micro-kernel processing
                                    #[cfg(target_arch = "x86_64")]
                                    let use_avx512 = matches!(self.backend, Backend::AVX512);
                                    #[cfg(target_arch = "x86_64")]
                                    let use_avx2 = matches!(self.backend, Backend::AVX2);

                                    #[cfg(target_arch = "x86_64")]
                                    if use_avx512 {
                                        // AVX-512 8x1 micro-kernel (Phase 3)
                                        let mut i = ii;

                                        // Process 8 rows at a time with AVX-512 micro-kernel
                                        while i + 8 <= i_end {
                                            let a_rows = [
                                                &self.data[i * self.cols + kk..(i * self.cols + kk) + block_size],
                                                &self.data[(i + 1) * self.cols + kk..((i + 1) * self.cols + kk) + block_size],
                                                &self.data[(i + 2) * self.cols + kk..((i + 2) * self.cols + kk) + block_size],
                                                &self.data[(i + 3) * self.cols + kk..((i + 3) * self.cols + kk) + block_size],
                                                &self.data[(i + 4) * self.cols + kk..((i + 4) * self.cols + kk) + block_size],
                                                &self.data[(i + 5) * self.cols + kk..((i + 5) * self.cols + kk) + block_size],
                                                &self.data[(i + 6) * self.cols + kk..((i + 6) * self.cols + kk) + block_size],
                                                &self.data[(i + 7) * self.cols + kk..((i + 7) * self.cols + kk) + block_size],
                                            ];

                                            for j in jj..j_end {
                                                let col_start = j * b_transposed.cols + kk;
                                                let b_col = &b_transposed.data
                                                    [col_start..col_start + block_size];

                                                let mut partial_dots = [0.0f32; 8];
                                                // SAFETY: CPU feature verified at runtime, slices bounds-checked
                                                unsafe {
                                                    Self::matmul_microkernel_8x1_avx512(
                                                        a_rows,
                                                        b_col,
                                                        &mut partial_dots,
                                                    );
                                                }

                                                result.data[i * result.cols + j] += partial_dots[0];
                                                result.data[(i + 1) * result.cols + j] += partial_dots[1];
                                                result.data[(i + 2) * result.cols + j] += partial_dots[2];
                                                result.data[(i + 3) * result.cols + j] += partial_dots[3];
                                                result.data[(i + 4) * result.cols + j] += partial_dots[4];
                                                result.data[(i + 5) * result.cols + j] += partial_dots[5];
                                                result.data[(i + 6) * result.cols + j] += partial_dots[6];
                                                result.data[(i + 7) * result.cols + j] += partial_dots[7];
                                            }

                                            i += 8;
                                        }

                                        // Handle remaining rows with AVX2 4x1 kernel
                                        while i + 4 <= i_end {
                                            let a_rows = [
                                                &self.data[i * self.cols + kk..(i * self.cols + kk) + block_size],
                                                &self.data[(i + 1) * self.cols + kk..((i + 1) * self.cols + kk) + block_size],
                                                &self.data[(i + 2) * self.cols + kk..((i + 2) * self.cols + kk) + block_size],
                                                &self.data[(i + 3) * self.cols + kk..((i + 3) * self.cols + kk) + block_size],
                                            ];

                                            for j in jj..j_end {
                                                let col_start = j * b_transposed.cols + kk;
                                                let b_col = &b_transposed.data[col_start..col_start + block_size];

                                                let mut partial_dots = [0.0f32; 4];
                                                // SAFETY: CPU feature verified at runtime, slices bounds-checked
                                                unsafe {
                                                    Self::matmul_microkernel_4x1_avx2(a_rows, b_col, &mut partial_dots);
                                                }

                                                result.data[i * result.cols + j] += partial_dots[0];
                                                result.data[(i + 1) * result.cols + j] += partial_dots[1];
                                                result.data[(i + 2) * result.cols + j] += partial_dots[2];
                                                result.data[(i + 3) * result.cols + j] += partial_dots[3];
                                            }
                                            i += 4;
                                        }

                                        // Handle remaining rows (< 4)
                                        for i in i..i_end {
                                            let row_start = i * self.cols + kk;
                                            let a_row = &self.data[row_start..row_start + block_size];

                                            for j in jj..j_end {
                                                let col_start = j * b_transposed.cols + kk;
                                                let b_col = &b_transposed.data[col_start..col_start + block_size];

                                                // SAFETY: AVX2 verified at runtime, slices bounds-checked
                                                let partial_dot = unsafe { Avx2Backend::dot(a_row, b_col) };
                                                result.data[i * result.cols + j] += partial_dot;
                                            }
                                        }
                                    } else if use_avx2 {
                                        // AVX2 4x1 micro-kernel
                                        let mut i = ii;

                                        // Process 4 rows at a time with micro-kernel
                                        while i + 4 <= i_end {
                                            let row0_start = i * self.cols + kk;
                                            let row1_start = (i + 1) * self.cols + kk;
                                            let row2_start = (i + 2) * self.cols + kk;
                                            let row3_start = (i + 3) * self.cols + kk;

                                            let a_rows = [
                                                &self.data[row0_start..row0_start + block_size],
                                                &self.data[row1_start..row1_start + block_size],
                                                &self.data[row2_start..row2_start + block_size],
                                                &self.data[row3_start..row3_start + block_size],
                                            ];

                                            for j in jj..j_end {
                                                let col_start = j * b_transposed.cols + kk;
                                                let b_col = &b_transposed.data
                                                    [col_start..col_start + block_size];

                                                let mut partial_dots = [0.0f32; 4];
                                                // SAFETY: CPU feature verified at runtime, slices bounds-checked
                                                unsafe {
                                                    Self::matmul_microkernel_4x1_avx2(
                                                        a_rows,
                                                        b_col,
                                                        &mut partial_dots,
                                                    );
                                                }

                                                result.data[i * result.cols + j] += partial_dots[0];
                                                result.data[(i + 1) * result.cols + j] +=
                                                    partial_dots[1];
                                                result.data[(i + 2) * result.cols + j] +=
                                                    partial_dots[2];
                                                result.data[(i + 3) * result.cols + j] +=
                                                    partial_dots[3];
                                            }

                                            i += 4;
                                        }

                                        // Handle remaining rows (< 4)
                                        for i in i..i_end {
                                            let row_start = i * self.cols + kk;
                                            let a_row =
                                                &self.data[row_start..row_start + block_size];

                                            for j in jj..j_end {
                                                let col_start = j * b_transposed.cols + kk;
                                                let b_col = &b_transposed.data
                                                    [col_start..col_start + block_size];

                                                let partial_dot =
                                                    // SAFETY: CPU feature verified at runtime, slices bounds-checked
                                                    unsafe { Avx2Backend::dot(a_row, b_col) };
                                                result.data[i * result.cols + j] += partial_dot;
                                            }
                                        }
                                    } else {
                                        // Non-AVX2 path
                                        #[allow(unused_variables)]
                                        for i in ii..i_end {
                                            let row_start = i * self.cols + kk;
                                            let a_row =
                                                &self.data[row_start..row_start + block_size];

                                            for j in jj..j_end {
                                                let col_start = j * b_transposed.cols + kk;
                                                let b_col = &b_transposed.data
                                                    [col_start..col_start + block_size];

                                                // SAFETY: AVX2 verified at runtime, slices bounds-checked
                                                let partial_dot = unsafe {
                                                    match self.backend {
                                                        Backend::Scalar => {
                                                            ScalarBackend::dot(a_row, b_col)
                                                        }
                                                        #[cfg(target_arch = "x86_64")]
                                                        Backend::SSE2 | Backend::AVX => {
                                                            Sse2Backend::dot(a_row, b_col)
                                                        }
                                                        #[cfg(not(target_arch = "x86_64"))]
                                                        Backend::SSE2
                                                        | Backend::AVX
                                                        | Backend::AVX2
                                                        | Backend::AVX512 => {
                                                            ScalarBackend::dot(a_row, b_col)
                                                        }
                                                        #[cfg(any(
                                                            target_arch = "aarch64",
                                                            target_arch = "arm"
                                                        ))]
                                                        Backend::NEON => {
                                                            use crate::backends::neon::NeonBackend;
                                                            NeonBackend::dot(a_row, b_col)
                                                        }
                                                        #[cfg(not(any(
                                                            target_arch = "aarch64",
                                                            target_arch = "arm"
                                                        )))]
                                                        Backend::NEON => {
                                                            ScalarBackend::dot(a_row, b_col)
                                                        }
                                                        #[cfg(target_arch = "wasm32")]
                                                        Backend::WasmSIMD => {
                                                            use crate::backends::wasm::WasmBackend;
                                                            WasmBackend::dot(a_row, b_col)
                                                        }
                                                        #[cfg(not(target_arch = "wasm32"))]
                                                        Backend::WasmSIMD => {
                                                            ScalarBackend::dot(a_row, b_col)
                                                        }
                                                        Backend::GPU
                                                        | Backend::Auto
                                                        | Backend::AVX2
                                                        | Backend::AVX512 => {
                                                            ScalarBackend::dot(a_row, b_col)
                                                        }
                                                    }
                                                };

                                                result.data[i * result.cols + j] += partial_dot;
                                            }
                                        }
                                    }

                                    // Non-x86_64 platforms
                                    #[cfg(not(target_arch = "x86_64"))]
                                    for i in ii..i_end {
                                        let row_start = i * self.cols + kk;
                                        let a_row = &self.data[row_start..row_start + block_size];

                                        for j in jj..j_end {
                                            let col_start = j * b_transposed.cols + kk;
                                            let b_col = &b_transposed.data
                                                [col_start..col_start + block_size];

                                            // SAFETY: AVX2 verified at runtime, slices bounds-checked
                                            let partial_dot = unsafe {
                                                match self.backend {
                                                    Backend::Scalar => {
                                                        ScalarBackend::dot(a_row, b_col)
                                                    }
                                                    #[cfg(any(
                                                        target_arch = "aarch64",
                                                        target_arch = "arm"
                                                    ))]
                                                    Backend::NEON => {
                                                        use crate::backends::neon::NeonBackend;
                                                        NeonBackend::dot(a_row, b_col)
                                                    }
                                                    #[cfg(not(any(
                                                        target_arch = "aarch64",
                                                        target_arch = "arm"
                                                    )))]
                                                    Backend::NEON => {
                                                        ScalarBackend::dot(a_row, b_col)
                                                    }
                                                    #[cfg(target_arch = "wasm32")]
                                                    Backend::WasmSIMD => {
                                                        use crate::backends::wasm::WasmBackend;
                                                        WasmBackend::dot(a_row, b_col)
                                                    }
                                                    #[cfg(not(target_arch = "wasm32"))]
                                                    Backend::WasmSIMD => {
                                                        ScalarBackend::dot(a_row, b_col)
                                                    }
                                                    _ => ScalarBackend::dot(a_row, b_col),
                                                }
                                            };

                                            result.data[i * result.cols + j] += partial_dot;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // ===== Phase 1/2: 2-Level Cache Blocking (L2 → micro-kernel) =====
            // For medium matrices (32-512), use original 2-level blocking
            //
            // This path preserves the fast performance for 256×256 and smaller matrices
            // by avoiding the overhead of 3-level loop nesting

            for ii in (0..self.rows).step_by(L2_BLOCK_SIZE) {
                let i_end = (ii + L2_BLOCK_SIZE).min(self.rows);

                for jj in (0..other.cols).step_by(L2_BLOCK_SIZE) {
                    let j_end = (jj + L2_BLOCK_SIZE).min(other.cols);

                    for kk in (0..self.cols).step_by(L2_BLOCK_SIZE) {
                        let k_end = (kk + L2_BLOCK_SIZE).min(self.cols);
                        let block_size = k_end - kk;

                        // Inner loops: Process L2 block with micro-kernel (Phase 2) or SIMD
                        #[cfg(target_arch = "x86_64")]
                        let use_microkernel =
                            matches!(self.backend, Backend::AVX2 | Backend::AVX512);

                        #[cfg(target_arch = "x86_64")]
                        if use_microkernel {
                            // Phase 2: Use 4×1 micro-kernel for AVX2/AVX512
                            let mut i = ii;

                            // Process 4 rows at a time with micro-kernel
                            while i + 4 <= i_end {
                                // Get 4 consecutive rows of A
                                let row0_start = i * self.cols + kk;
                                let row1_start = (i + 1) * self.cols + kk;
                                let row2_start = (i + 2) * self.cols + kk;
                                let row3_start = (i + 3) * self.cols + kk;

                                let a_rows = [
                                    &self.data[row0_start..row0_start + block_size],
                                    &self.data[row1_start..row1_start + block_size],
                                    &self.data[row2_start..row2_start + block_size],
                                    &self.data[row3_start..row3_start + block_size],
                                ];

                                // Process each column of B with the micro-kernel
                                for j in jj..j_end {
                                    let col_start = j * b_transposed.cols + kk;
                                    let b_col =
                                        &b_transposed.data[col_start..col_start + block_size];

                                    // Compute 4 dot products simultaneously
                                    let mut partial_dots = [0.0f32; 4];
                                    // SAFETY: CPU feature verified at runtime, slices bounds-checked
                                    unsafe {
                                        Self::matmul_microkernel_4x1_avx2(
                                            a_rows,
                                            b_col,
                                            &mut partial_dots,
                                        );
                                    }

                                    // Accumulate results
                                    result.data[i * result.cols + j] += partial_dots[0];
                                    result.data[(i + 1) * result.cols + j] += partial_dots[1];
                                    result.data[(i + 2) * result.cols + j] += partial_dots[2];
                                    result.data[(i + 3) * result.cols + j] += partial_dots[3];
                                }

                                i += 4;
                            }

                            // Handle remaining rows (< 4) with standard path
                            for i in i..i_end {
                                let row_start = i * self.cols + kk;
                                let a_row = &self.data[row_start..row_start + block_size];

                                for j in jj..j_end {
                                    let col_start = j * b_transposed.cols + kk;
                                    let b_col =
                                        &b_transposed.data[col_start..col_start + block_size];

                                    // SAFETY: AVX2 verified at runtime, slices bounds-checked
                                    let partial_dot = unsafe { Avx2Backend::dot(a_row, b_col) };
                                    result.data[i * result.cols + j] += partial_dot;
                                }
                            }
                        } else {
                            // Phase 1: Standard SIMD path (non-AVX2 backends)
                            #[allow(unused_variables)]
                            for i in ii..i_end {
                                let row_start = i * self.cols + kk;
                                let a_row = &self.data[row_start..row_start + block_size];

                                for j in jj..j_end {
                                    let col_start = j * b_transposed.cols + kk;
                                    let b_col =
                                        &b_transposed.data[col_start..col_start + block_size];

                                    // SAFETY: AVX2 verified at runtime, slices bounds-checked
                                    let partial_dot = unsafe {
                                        match self.backend {
                                            Backend::Scalar => ScalarBackend::dot(a_row, b_col),
                                            #[cfg(target_arch = "x86_64")]
                                            Backend::SSE2 | Backend::AVX => {
                                                Sse2Backend::dot(a_row, b_col)
                                            }
                                            #[cfg(not(target_arch = "x86_64"))]
                                            Backend::SSE2
                                            | Backend::AVX
                                            | Backend::AVX2
                                            | Backend::AVX512 => ScalarBackend::dot(a_row, b_col),
                                            #[cfg(any(
                                                target_arch = "aarch64",
                                                target_arch = "arm"
                                            ))]
                                            Backend::NEON => {
                                                use crate::backends::neon::NeonBackend;
                                                NeonBackend::dot(a_row, b_col)
                                            }
                                            #[cfg(not(any(
                                                target_arch = "aarch64",
                                                target_arch = "arm"
                                            )))]
                                            Backend::NEON => ScalarBackend::dot(a_row, b_col),
                                            #[cfg(target_arch = "wasm32")]
                                            Backend::WasmSIMD => {
                                                use crate::backends::wasm::WasmBackend;
                                                WasmBackend::dot(a_row, b_col)
                                            }
                                            #[cfg(not(target_arch = "wasm32"))]
                                            Backend::WasmSIMD => ScalarBackend::dot(a_row, b_col),
                                            Backend::GPU
                                            | Backend::Auto
                                            | Backend::AVX2
                                            | Backend::AVX512 => ScalarBackend::dot(a_row, b_col),
                                        }
                                    };

                                    result.data[i * result.cols + j] += partial_dot;
                                }
                            }
                        }

                        // Non-x86_64 platforms: Use standard SIMD path
                        #[cfg(not(target_arch = "x86_64"))]
                        for i in ii..i_end {
                            let row_start = i * self.cols + kk;
                            let a_row = &self.data[row_start..row_start + block_size];

                            for j in jj..j_end {
                                let col_start = j * b_transposed.cols + kk;
                                let b_col = &b_transposed.data[col_start..col_start + block_size];

                                // SAFETY: AVX2 verified at runtime, slices bounds-checked
                                let partial_dot = unsafe {
                                    match self.backend {
                                        Backend::Scalar => ScalarBackend::dot(a_row, b_col),
                                        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                                        Backend::NEON => {
                                            use crate::backends::neon::NeonBackend;
                                            NeonBackend::dot(a_row, b_col)
                                        }
                                        #[cfg(not(any(
                                            target_arch = "aarch64",
                                            target_arch = "arm"
                                        )))]
                                        Backend::NEON => ScalarBackend::dot(a_row, b_col),
                                        #[cfg(target_arch = "wasm32")]
                                        Backend::WasmSIMD => {
                                            use crate::backends::wasm::WasmBackend;
                                            WasmBackend::dot(a_row, b_col)
                                        }
                                        #[cfg(not(target_arch = "wasm32"))]
                                        Backend::WasmSIMD => ScalarBackend::dot(a_row, b_col),
                                        _ => ScalarBackend::dot(a_row, b_col),
                                    }
                                };

                                result.data[i * result.cols + j] += partial_dot;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Simple SIMD matrix multiplication without blocking (for small matrices)
    ///
    /// This is the pre-blocking implementation that works well for small matrices
    /// where cache blocking overhead exceeds benefits.
    fn matmul_simd_simple(
        &self,
        other: &Matrix<f32>,
        result: &mut Matrix<f32>,
    ) -> Result<(), TruenoError> {
        #[cfg(target_arch = "x86_64")]
        use crate::backends::{avx2::Avx2Backend, sse2::Sse2Backend};
        use crate::backends::{scalar::ScalarBackend, VectorBackend};

        // Pre-transpose B for better cache locality
        let b_transposed = other.transpose();

        for i in 0..self.rows {
            let row_start = i * self.cols;
            let row_end = row_start + self.cols;
            let a_row = &self.data[row_start..row_end];

            for j in 0..other.cols {
                let col_start = j * b_transposed.cols;
                let col_end = col_start + b_transposed.cols;
                let b_col = &b_transposed.data[col_start..col_end];

                // Compute dot product using SIMD backend directly
                // SAFETY: Backend dot() maintains safety invariants
                let dot_result = unsafe {
                    match self.backend {
                        Backend::Scalar => ScalarBackend::dot(a_row, b_col),
                        #[cfg(target_arch = "x86_64")]
                        Backend::SSE2 | Backend::AVX => Sse2Backend::dot(a_row, b_col),
                        #[cfg(target_arch = "x86_64")]
                        Backend::AVX2 | Backend::AVX512 => Avx2Backend::dot(a_row, b_col),
                        #[cfg(not(target_arch = "x86_64"))]
                        Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                            ScalarBackend::dot(a_row, b_col)
                        }
                        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                        Backend::NEON => {
                            use crate::backends::neon::NeonBackend;
                            NeonBackend::dot(a_row, b_col)
                        }
                        #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                        Backend::NEON => ScalarBackend::dot(a_row, b_col),
                        #[cfg(target_arch = "wasm32")]
                        Backend::WasmSIMD => {
                            use crate::backends::wasm::WasmBackend;
                            WasmBackend::dot(a_row, b_col)
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        Backend::WasmSIMD => ScalarBackend::dot(a_row, b_col),
                        Backend::GPU | Backend::Auto => ScalarBackend::dot(a_row, b_col),
                    }
                };

                result.data[i * result.cols + j] = dot_result;
            }
        }

        Ok(())
    }

    /// WASM-optimized tiled matrix multiplication with SIMD inner loop
    ///
    /// Key optimizations:
    /// 1. NO transpose - avoids O(n²) memory allocation and copy
    /// 2. Tiled blocking with SIMD-aligned tile widths
    /// 3. Inner j-loop uses SIMD (B rows are contiguous in memory)
    /// 4. Register accumulation to minimize memory traffic
    ///
    /// Performance: Targets <30ms for 384×74×384 (Whisper encoder attention)
    fn matmul_wasm_tiled(
        &self,
        other: &Matrix<f32>,
        result: &mut Matrix<f32>,
    ) -> Result<(), TruenoError> {
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;

        // For each row of A
        for i in 0..m {
            let a_row_start = i * k;
            let result_row_start = i * n;

            // For each column of B, compute dot product A[i,:] · B[:,j]
            // BUT: B[:,j] is not contiguous. Instead, iterate over k and accumulate.
            //
            // C[i,j] = Σ_k A[i,k] * B[k,j]
            //
            // For efficiency, broadcast A[i,k] and multiply with B[k, j0:j0+width]
            // This uses SIMD on the contiguous B row segment.

            // Process output columns in SIMD-width chunks
            let simd_width = 8; // AVX2 processes 8 f32s
            let n_simd = (n / simd_width) * simd_width;

            // SIMD portion: columns 0..n_simd
            // Note: Explicit indexing is intentional for LLVM auto-vectorization.
            // Iterator patterns prevent the compiler from recognizing the SIMD pattern.
            #[allow(clippy::needless_range_loop)]
            for j0 in (0..n_simd).step_by(simd_width) {
                let mut acc = [0.0f32; 8];

                for kk in 0..k {
                    let a_val = self.data[a_row_start + kk];
                    let b_row_start = kk * n + j0;

                    // Multiply a_val with B[kk, j0:j0+8]
                    for jj in 0..simd_width {
                        acc[jj] += a_val * other.data[b_row_start + jj];
                    }
                }

                // Write accumulated results
                for jj in 0..simd_width {
                    result.data[result_row_start + j0 + jj] = acc[jj];
                }
            }

            // Remainder columns (non-SIMD)
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

    /// GPU-accelerated matrix multiplication (very large matrices only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    fn matmul_gpu(&self, other: &Matrix<f32>) -> Result<Matrix<f32>, TruenoError> {
        use crate::backends::gpu::GpuBackend;

        // Check if GPU is available
        if !GpuBackend::is_available() {
            return Err(TruenoError::InvalidInput("GPU not available".to_string()));
        }

        // Create GPU backend
        let mut gpu = GpuBackend::new();

        // Execute GPU matmul
        let result_data = gpu
            .matmul(&self.data, &other.data, self.rows, self.cols, other.cols)
            .map_err(|e| TruenoError::InvalidInput(format!("GPU matmul failed: {}", e)))?;

        // Create result matrix
        let mut result = Matrix::zeros(self.rows, other.cols);
        result.data = result_data;

        Ok(result)
    }

    /// Transpose the matrix (swap rows and columns)
    ///
    /// Returns a new matrix where element `(i, j)` of the original becomes
    /// element `(j, i)` in the result.
    ///
    /// # Returns
    ///
    /// A new matrix with dimensions swapped: if input is `m×n`, output is `n×m`
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
        // Fix for issue #65: non-square transpose was slow due to strided writes

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

        #[cfg(target_arch = "x86_64")]
        use crate::backends::{avx2::Avx2Backend, sse2::Sse2Backend};
        use crate::backends::{scalar::ScalarBackend, VectorBackend};

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

                    // SAFETY: CPU feature verified at runtime, slices bounds-checked
                    let dot_result = unsafe {
                        #[cfg(target_arch = "x86_64")]
                        {
                            match self.backend {
                                Backend::AVX2 | Backend::AVX512 => Avx2Backend::dot(row, v_slice),
                                Backend::SSE2 | Backend::AVX => Sse2Backend::dot(row, v_slice),
                                _ => ScalarBackend::dot(row, v_slice),
                            }
                        }
                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            ScalarBackend::dot(row, v_slice)
                        }
                    };

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
            // SAFETY: CPU feature verified at runtime, slices bounds-checked
            *result = unsafe {
                #[cfg(target_arch = "x86_64")]
                {
                    match self.backend {
                        Backend::AVX2 | Backend::AVX512 => Avx2Backend::dot(row, v_slice),
                        Backend::SSE2 | Backend::AVX => Sse2Backend::dot(row, v_slice),
                        _ => ScalarBackend::dot(row, v_slice),
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    ScalarBackend::dot(row, v_slice)
                }
            };
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

    /// Perform 2D convolution with a kernel
    ///
    /// Applies a 2D convolution operation using "valid" padding (no padding),
    /// resulting in an output smaller than the input.
    ///
    /// # Arguments
    ///
    /// * `kernel` - Convolution kernel (filter) to apply
    ///
    /// # Returns
    ///
    /// Convolved matrix with dimensions:
    /// - rows: `input.rows - kernel.rows + 1`
    /// - cols: `input.cols - kernel.cols + 1`
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if:
    /// - Kernel is larger than input in any dimension
    /// - Kernel has even dimensions (center pixel ambiguous)
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// // 5x5 input image
    /// let input = Matrix::from_vec(
    ///     5, 5,
    ///     vec![
    ///         0.0, 0.0, 0.0, 0.0, 0.0,
    ///         0.0, 0.0, 0.0, 0.0, 0.0,
    ///         0.0, 0.0, 9.0, 0.0, 0.0,
    ///         0.0, 0.0, 0.0, 0.0, 0.0,
    ///         0.0, 0.0, 0.0, 0.0, 0.0,
    ///     ]
    /// ).unwrap();
    ///
    /// // 3x3 averaging kernel
    /// let kernel_val = 1.0 / 9.0;
    /// let kernel = Matrix::from_vec(
    ///     3, 3,
    ///     vec![kernel_val; 9]
    /// ).unwrap();
    ///
    /// let result = input.convolve2d(&kernel).unwrap();
    /// assert_eq!(result.rows(), 3); // 5 - 3 + 1
    /// assert_eq!(result.cols(), 3);
    /// ```
    // =========================================================================
    // HOT PATH - PERFORMANCE CRITICAL
    // =========================================================================
    // This function processes millions of elements for typical image sizes.
    // Any changes to the inner loop REQUIRE benchmark verification:
    //   1. Run: make bench-check
    //   2. Verify no regression >10%
    //
    // PROHIBITED in inner loops:
    //   - .get() / .get_mut() (bounds checking overhead)
    //   - .expect() / .unwrap() (panic path overhead)
    //   - Iterator adaptors (closure overhead)
    //
    // Use direct indexing with bounds proof documented above the loop.
    // =========================================================================
    pub fn convolve2d(&self, kernel: &Matrix<f32>) -> Result<Matrix<f32>, TruenoError> {
        // Validate kernel size
        if kernel.rows > self.rows || kernel.cols > self.cols {
            return Err(TruenoError::InvalidInput(format!(
                "Kernel size ({}x{}) larger than input ({}x{})",
                kernel.rows, kernel.cols, self.rows, self.cols
            )));
        }

        // Calculate output dimensions (valid padding)
        let output_rows = self.rows - kernel.rows + 1;
        let output_cols = self.cols - kernel.cols + 1;

        // Initialize output matrix (reuse parent's backend)
        let mut result = Matrix::zeros_with_backend(output_rows, output_cols, self.backend);

        // Backend selection strategy:
        // OpComplexity::High - GPU beneficial at >10K elements
        // GPU for large images (output > 10K elements)
        // Scalar for smaller images

        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = 10_000;

        // Try GPU first for large convolutions
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if output_rows * output_cols >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuBackend;

                if GpuBackend::is_available() {
                    if let Ok(gpu_result) =
                        self.convolve2d_gpu(kernel, &mut result, output_rows, output_cols)
                    {
                        return Ok(gpu_result);
                    }
                    // Fall through to scalar if GPU fails
                }
            }
        }

        // Scalar baseline implementation - optimized with direct indexing
        // SAFETY invariant proof:
        // - output_rows = self.rows - kernel.rows + 1
        // - output_cols = self.cols - kernel.cols + 1
        // - For any out_row < output_rows and k_row < kernel.rows:
        //   in_row = out_row + k_row < (self.rows - kernel.rows + 1) + kernel.rows - 1 = self.rows
        // - Same logic applies to columns
        // - All indices are provably within bounds, so we use direct indexing for performance

        let input_data = self.as_slice();
        let kernel_data = kernel.as_slice();
        let result_data = result.data.as_mut_slice();
        let input_cols = self.cols;
        let kernel_cols = kernel.cols;
        let result_cols = output_cols;

        for out_row in 0..output_rows {
            for out_col in 0..output_cols {
                let mut sum = 0.0;

                // Apply kernel - use direct indexing for performance
                for k_row in 0..kernel.rows {
                    let in_row = out_row + k_row;
                    let input_row_offset = in_row * input_cols;
                    let kernel_row_offset = k_row * kernel_cols;

                    for k_col in 0..kernel.cols {
                        let in_col = out_col + k_col;
                        sum += input_data[input_row_offset + in_col]
                            * kernel_data[kernel_row_offset + k_col];
                    }
                }

                result_data[out_row * result_cols + out_col] = sum;
            }
        }

        Ok(result)
    }

    /// GPU-accelerated 2D convolution helper
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    fn convolve2d_gpu(
        &self,
        kernel: &Matrix<f32>,
        result: &mut Matrix<f32>,
        _output_rows: usize,
        _output_cols: usize,
    ) -> Result<Matrix<f32>, TruenoError> {
        use crate::backends::gpu::GpuDevice;

        let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;

        gpu.convolve2d(
            self.as_slice(),
            kernel.as_slice(),
            result.data.as_mut_slice(),
            self.rows,
            self.cols,
            kernel.rows,
            kernel.cols,
        )
        .map_err(TruenoError::InvalidInput)?;

        Ok(result.clone())
    }

    /// Lookup embeddings by indices (Issue #61: ML primitives)
    ///
    /// Performs embedding lookup where self is the embedding table with shape
    /// `[vocab_size, embed_dim]` and indices specify which rows to select.
    ///
    /// # Arguments
    ///
    /// * `indices` - Slice of indices into the embedding table
    ///
    /// # Returns
    ///
    /// A matrix with shape `[indices.len(), embed_dim]` containing the selected rows
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if any index is out of bounds
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Matrix;
    ///
    /// // Create embedding table: 4 words, 3-dimensional embeddings
    /// let embeddings = Matrix::from_vec(4, 3, vec![
    ///     1.0, 2.0, 3.0,   // word 0
    ///     4.0, 5.0, 6.0,   // word 1
    ///     7.0, 8.0, 9.0,   // word 2
    ///     10.0, 11.0, 12.0 // word 3
    /// ]).unwrap();
    ///
    /// // Lookup embeddings for indices [1, 3, 0]
    /// let result = embeddings.embedding_lookup(&[1, 3, 0]).unwrap();
    ///
    /// assert_eq!(result.rows(), 3);
    /// assert_eq!(result.cols(), 3);
    /// assert_eq!(result.get(0, 0), Some(&4.0)); // word 1
    /// assert_eq!(result.get(1, 0), Some(&10.0)); // word 3
    /// assert_eq!(result.get(2, 0), Some(&1.0)); // word 0
    /// ```
    pub fn embedding_lookup(&self, indices: &[usize]) -> Result<Matrix<f32>, TruenoError> {
        // Validate indices
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.rows {
                return Err(TruenoError::InvalidInput(format!(
                    "Index {} at position {} is out of bounds for embedding table with {} rows",
                    idx, i, self.rows
                )));
            }
        }

        // Handle empty indices
        if indices.is_empty() {
            return Ok(Matrix::zeros_with_backend(0, self.cols, self.backend));
        }

        // Allocate output matrix: [seq_len, embed_dim]
        let seq_len = indices.len();
        let embed_dim = self.cols;
        let mut result = Matrix::zeros_with_backend(seq_len, embed_dim, self.backend);

        // Copy rows from embedding table to result
        for (out_row, &idx) in indices.iter().enumerate() {
            let src_start = idx * embed_dim;
            let dst_start = out_row * embed_dim;

            // Copy entire row
            result.data[dst_start..dst_start + embed_dim]
                .copy_from_slice(&self.data[src_start..src_start + embed_dim]);
        }

        Ok(result)
    }

    /// Lookup embeddings with gradient tracking support (for training)
    ///
    /// Returns both the embeddings and a sparse gradient accumulator.
    /// This is useful for sparse gradient updates in training.
    ///
    /// # Arguments
    ///
    /// * `indices` - Slice of indices into the embedding table
    ///
    /// # Returns
    ///
    /// Tuple of (embeddings, unique_indices) where unique_indices can be used
    /// for sparse gradient updates
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if any index is out of bounds
    pub fn embedding_lookup_sparse(
        &self,
        indices: &[usize],
    ) -> Result<(Matrix<f32>, Vec<usize>), TruenoError> {
        let embeddings = self.embedding_lookup(indices)?;

        // Get unique indices for sparse gradient updates
        let mut unique: Vec<usize> = indices.to_vec();
        unique.sort_unstable();
        unique.dedup();

        Ok((embeddings, unique))
    }

    /// 2D Max Pooling operation for CNN downsampling
    ///
    /// Applies max pooling over a 2D input tensor with specified kernel size and stride.
    /// Input shape: (height, width), Output shape: ((height - kh) / sh + 1, (width - kw) / sw + 1)
    ///
    /// # Arguments
    /// * `kernel` - (kernel_height, kernel_width) pooling window size
    /// * `stride` - (stride_height, stride_width) step size
    ///
    /// # Examples
    /// ```
    /// use trueno::matrix::Matrix;
    /// let input = Matrix::from_vec(4, 4, vec![
    ///     1.0, 2.0, 3.0, 4.0,
    ///     5.0, 6.0, 7.0, 8.0,
    ///     9.0, 10.0, 11.0, 12.0,
    ///     13.0, 14.0, 15.0, 16.0,
    /// ]).unwrap();
    /// let pooled = input.max_pool2d((2, 2), (2, 2)).unwrap();
    /// assert_eq!(pooled.shape(), (2, 2));
    /// assert_eq!(pooled.get(0, 0), Some(&6.0));  // max of [1,2,5,6]
    /// assert_eq!(pooled.get(1, 1), Some(&16.0)); // max of [11,12,15,16]
    /// ```
    pub fn max_pool2d(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Matrix<f32>, TruenoError> {
        let (kh, kw) = kernel;
        let (sh, sw) = stride;

        if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
            return Err(TruenoError::InvalidInput(
                "Kernel and stride dimensions must be positive".into(),
            ));
        }

        if kh > self.rows || kw > self.cols {
            return Err(TruenoError::InvalidInput(format!(
                "Kernel size ({}, {}) larger than input ({}, {})",
                kh, kw, self.rows, self.cols
            )));
        }

        let out_h = (self.rows - kh) / sh + 1;
        let out_w = (self.cols - kw) / sw + 1;
        let mut result = Matrix::new(out_h, out_w);

        for i in 0..out_h {
            for j in 0..out_w {
                let mut max_val = f32::NEG_INFINITY;
                for ki in 0..kh {
                    for kj in 0..kw {
                        let val = self.data[(i * sh + ki) * self.cols + (j * sw + kj)];
                        max_val = max_val.max(val);
                    }
                }
                result.data[i * out_w + j] = max_val;
            }
        }

        Ok(result)
    }

    /// 2D Average Pooling operation for CNN downsampling
    ///
    /// Applies average pooling over a 2D input tensor with specified kernel size and stride.
    /// Input shape: (height, width), Output shape: ((height - kh) / sh + 1, (width - kw) / sw + 1)
    ///
    /// # Arguments
    /// * `kernel` - (kernel_height, kernel_width) pooling window size
    /// * `stride` - (stride_height, stride_width) step size
    ///
    /// # Examples
    /// ```
    /// use trueno::matrix::Matrix;
    /// let input = Matrix::from_vec(4, 4, vec![
    ///     1.0, 2.0, 3.0, 4.0,
    ///     5.0, 6.0, 7.0, 8.0,
    ///     9.0, 10.0, 11.0, 12.0,
    ///     13.0, 14.0, 15.0, 16.0,
    /// ]).unwrap();
    /// let pooled = input.avg_pool2d((2, 2), (2, 2)).unwrap();
    /// assert_eq!(pooled.shape(), (2, 2));
    /// assert!((pooled.get(0, 0).unwrap() - 3.5).abs() < 1e-5);  // avg of [1,2,5,6]
    /// ```
    pub fn avg_pool2d(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Matrix<f32>, TruenoError> {
        let (kh, kw) = kernel;
        let (sh, sw) = stride;

        if kh == 0 || kw == 0 || sh == 0 || sw == 0 {
            return Err(TruenoError::InvalidInput(
                "Kernel and stride dimensions must be positive".into(),
            ));
        }

        if kh > self.rows || kw > self.cols {
            return Err(TruenoError::InvalidInput(format!(
                "Kernel size ({}, {}) larger than input ({}, {})",
                kh, kw, self.rows, self.cols
            )));
        }

        let out_h = (self.rows - kh) / sh + 1;
        let out_w = (self.cols - kw) / sw + 1;
        let kernel_size = (kh * kw) as f32;
        let mut result = Matrix::new(out_h, out_w);

        for i in 0..out_h {
            for j in 0..out_w {
                let mut sum = 0.0;
                for ki in 0..kh {
                    for kj in 0..kw {
                        sum += self.data[(i * sh + ki) * self.cols + (j * sw + kj)];
                    }
                }
                result.data[i * out_w + j] = sum / kernel_size;
            }
        }

        Ok(result)
    }

    /// Top-K selection: returns the k largest elements and their indices
    ///
    /// Useful for beam search, sampling, and ranking operations.
    /// Searches row-major order and returns (values, indices) sorted descending.
    ///
    /// # Examples
    /// ```
    /// use trueno::matrix::Matrix;
    /// let m = Matrix::from_vec(2, 3, vec![1.0, 5.0, 3.0, 2.0, 6.0, 4.0]).unwrap();
    /// let (values, indices) = m.topk(2).unwrap();
    /// assert_eq!(values, vec![6.0, 5.0]);
    /// assert_eq!(indices, vec![4, 1]);  // flat indices
    /// ```
    pub fn topk(&self, k: usize) -> Result<(Vec<f32>, Vec<usize>), TruenoError> {
        if k == 0 {
            return Ok((vec![], vec![]));
        }

        let k = k.min(self.data.len());
        let mut indexed: Vec<(usize, f32)> = self.data.iter().copied().enumerate().collect();

        // Partial sort - only sort k elements
        indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        indexed.truncate(k);
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let values: Vec<f32> = indexed.iter().map(|(_, v)| *v).collect();
        let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();

        Ok((values, indices))
    }

    /// Gather elements along axis using indices
    ///
    /// For 2D matrix with axis=0: output[i] = self[indices[i], :]
    /// For 2D matrix with axis=1: output[:, i] = self[:, indices[i]]
    ///
    /// # Examples
    /// ```
    /// use trueno::matrix::Matrix;
    /// let m = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    /// let gathered = m.gather(&[2, 0], 0).unwrap();  // Select rows 2 and 0
    /// assert_eq!(gathered.shape(), (2, 2));
    /// assert_eq!(gathered.get(0, 0), Some(&5.0));  // Row 2
    /// assert_eq!(gathered.get(1, 0), Some(&1.0));  // Row 0
    /// ```
    pub fn gather(&self, indices: &[usize], axis: usize) -> Result<Matrix<f32>, TruenoError> {
        match axis {
            0 => {
                // Gather rows
                let mut result = Matrix::new(indices.len(), self.cols);
                for (out_i, &idx) in indices.iter().enumerate() {
                    if idx >= self.rows {
                        return Err(TruenoError::InvalidInput(format!(
                            "Index {} out of bounds for axis 0 with size {}",
                            idx, self.rows
                        )));
                    }
                    for j in 0..self.cols {
                        result.data[out_i * self.cols + j] = self.data[idx * self.cols + j];
                    }
                }
                Ok(result)
            }
            1 => {
                // Gather columns
                let mut result = Matrix::new(self.rows, indices.len());
                for i in 0..self.rows {
                    for (out_j, &idx) in indices.iter().enumerate() {
                        if idx >= self.cols {
                            return Err(TruenoError::InvalidInput(format!(
                                "Index {} out of bounds for axis 1 with size {}",
                                idx, self.cols
                            )));
                        }
                        result.data[i * indices.len() + out_j] = self.data[i * self.cols + idx];
                    }
                }
                Ok(result)
            }
            _ => Err(TruenoError::InvalidInput(format!(
                "Axis {} not supported for 2D matrix (use 0 or 1)",
                axis
            ))),
        }
    }

    /// Pad matrix with a constant value
    ///
    /// # Arguments
    /// * `padding` - ((top, bottom), (left, right)) padding amounts
    /// * `value` - constant value to pad with (usually 0.0)
    ///
    /// # Examples
    /// ```
    /// use trueno::matrix::Matrix;
    /// let m = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let padded = m.pad(((1, 1), (1, 1)), 0.0).unwrap();
    /// assert_eq!(padded.shape(), (4, 4));
    /// assert_eq!(padded.get(0, 0), Some(&0.0));  // top-left padding
    /// assert_eq!(padded.get(1, 1), Some(&1.0));  // original (0,0)
    /// ```
    pub fn pad(
        &self,
        padding: ((usize, usize), (usize, usize)),
        value: f32,
    ) -> Result<Matrix<f32>, TruenoError> {
        let ((top, bottom), (left, right)) = padding;
        let new_rows = self.rows + top + bottom;
        let new_cols = self.cols + left + right;

        let mut result = Matrix::from_vec(new_rows, new_cols, vec![value; new_rows * new_cols])?;

        // Copy original data
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[(i + top) * new_cols + (j + left)] = self.data[i * self.cols + j];
            }
        }

        Ok(result)
    }
}


// Tests (~2.6K lines extracted for TDG compliance)
#[cfg(test)]
mod tests;
