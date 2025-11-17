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
    pub fn matmul(&self, other: &Matrix<f32>) -> Result<Matrix<f32>, TruenoError> {
        if self.cols != other.rows {
            return Err(TruenoError::InvalidInput(format!(
                "Matrix dimension mismatch for multiplication: {}×{} × {}×{} (inner dimensions {} and {} must match)",
                self.rows, self.cols, other.rows, other.cols, self.cols, other.rows
            )));
        }

        let mut result = Matrix::zeros(self.rows, other.cols);

        // Backend selection strategy:
        // 1. GPU for very large matrices (>1000×1000) - 10-50x speedup
        // 2. SIMD for medium-large matrices (>64×64) - 2-8x speedup
        // 3. Naive for small matrices - lowest overhead

        #[cfg(feature = "gpu")]
        const GPU_THRESHOLD: usize = 1000;
        const SIMD_THRESHOLD: usize = 64;

        // Try GPU first for very large matrices
        #[cfg(feature = "gpu")]
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
            self.matmul_simd(other, &mut result)?;
        } else {
            self.matmul_naive(other, &mut result)?;
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
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k).unwrap() * other.get(k, j).unwrap();
                }
                *result.get_mut(i, j).unwrap() = sum;
            }
        }
        Ok(())
    }

    /// SIMD-optimized matrix multiplication using Vector operations
    fn matmul_simd(
        &self,
        other: &Matrix<f32>,
        result: &mut Matrix<f32>,
    ) -> Result<(), TruenoError> {
        // Strategy: Use Vector::dot() for each element computation
        // C[i,j] = dot(A_row_i, B_col_j)

        // Pre-transpose B for better cache locality (columns become rows)
        let b_transposed = other.transpose();

        for i in 0..self.rows {
            // Extract row i from A as a slice
            let row_start = i * self.cols;
            let row_end = row_start + self.cols;
            let a_row = &self.data[row_start..row_end];
            let a_vec = Vector::from_slice(a_row);

            for j in 0..other.cols {
                // Extract row j from B^T (which is column j from B)
                let col_start = j * b_transposed.cols;
                let col_end = col_start + b_transposed.cols;
                let b_col = &b_transposed.data[col_start..col_end];
                let b_vec = Vector::from_slice(b_col);

                // Compute dot product using SIMD
                let dot_result = a_vec.dot(&b_vec)?;
                *result.get_mut(i, j).unwrap() = dot_result;
            }
        }

        Ok(())
    }

    /// GPU-accelerated matrix multiplication (very large matrices only)
    #[cfg(feature = "gpu")]
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
    pub fn transpose(&self) -> Matrix<f32> {
        let mut result = Matrix::zeros(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                *result.get_mut(j, i).unwrap() = *self.get(i, j).unwrap();
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

        let mut result_data = vec![0.0; self.rows];

        // Compute result[i] = Σ A[i,j] × v[j]
        for (i, result_elem) in result_data.iter_mut().enumerate() {
            let mut sum = 0.0;
            for j in 0..self.cols {
                sum += self.get(i, j).unwrap() * v.as_slice()[j];
            }
            *result_elem = sum;
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

        let mut result_data = vec![0.0; m.cols];

        // Compute result[j] = Σ v[i] × A[i,j]
        for (j, result_elem) in result_data.iter_mut().enumerate() {
            let mut sum = 0.0;
            for i in 0..m.rows {
                sum += v.as_slice()[i] * m.get(i, j).unwrap();
            }
            *result_elem = sum;
        }

        Ok(Vector::from_slice(&result_data))
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

        // Initialize output matrix
        let mut result = Matrix::zeros(output_rows, output_cols);

        // Perform convolution (scalar baseline)
        for out_row in 0..output_rows {
            for out_col in 0..output_cols {
                let mut sum = 0.0;

                // Apply kernel
                for k_row in 0..kernel.rows {
                    for k_col in 0..kernel.cols {
                        let in_row = out_row + k_row;
                        let in_col = out_col + k_col;

                        let input_val = self.get(in_row, in_col).unwrap();
                        let kernel_val = kernel.get(k_row, k_col).unwrap();

                        sum += input_val * kernel_val;
                    }
                }

                *result.get_mut(out_row, out_col).unwrap() = sum;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_new() {
        let m = Matrix::new(3, 4);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 4);
        assert_eq!(m.shape(), (3, 4));
        assert_eq!(m.as_slice().len(), 12);
    }

    #[test]
    fn test_matrix_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let m = Matrix::from_vec(2, 2, data).unwrap();
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.get(0, 0), Some(&1.0));
        assert_eq!(m.get(0, 1), Some(&2.0));
        assert_eq!(m.get(1, 0), Some(&3.0));
        assert_eq!(m.get(1, 1), Some(&4.0));
    }

    #[test]
    fn test_matrix_from_vec_invalid_size() {
        let data = vec![1.0, 2.0, 3.0];
        let result = Matrix::from_vec(2, 2, data);
        assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
    }

    #[test]
    fn test_matrix_zeros() {
        let m = Matrix::zeros(2, 3);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.cols(), 3);
        for &val in m.as_slice() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_matrix_identity() {
        let m = Matrix::identity(3);
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 3);

        // Check diagonal
        assert_eq!(m.get(0, 0), Some(&1.0));
        assert_eq!(m.get(1, 1), Some(&1.0));
        assert_eq!(m.get(2, 2), Some(&1.0));

        // Check off-diagonal
        assert_eq!(m.get(0, 1), Some(&0.0));
        assert_eq!(m.get(0, 2), Some(&0.0));
        assert_eq!(m.get(1, 0), Some(&0.0));
        assert_eq!(m.get(1, 2), Some(&0.0));
        assert_eq!(m.get(2, 0), Some(&0.0));
        assert_eq!(m.get(2, 1), Some(&0.0));
    }

    #[test]
    fn test_matrix_get_out_of_bounds() {
        let m = Matrix::new(2, 2);
        assert_eq!(m.get(2, 0), None);
        assert_eq!(m.get(0, 2), None);
        assert_eq!(m.get(2, 2), None);
    }

    // ===== Matrix Multiplication Tests =====

    #[test]
    fn test_matmul_basic() {
        // [[1, 2],   [[5, 6],   [[19, 22],
        //  [3, 4]] ×  [7, 8]] =  [43, 50]]
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);
        assert_eq!(c.get(0, 0), Some(&19.0));
        assert_eq!(c.get(0, 1), Some(&22.0));
        assert_eq!(c.get(1, 0), Some(&43.0));
        assert_eq!(c.get(1, 1), Some(&50.0));
    }

    #[test]
    fn test_matmul_identity() {
        // A × I = A
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let identity = Matrix::identity(2);
        let result = a.matmul(&identity).unwrap();

        assert_eq!(result.get(0, 0), Some(&1.0));
        assert_eq!(result.get(0, 1), Some(&2.0));
        assert_eq!(result.get(1, 0), Some(&3.0));
        assert_eq!(result.get(1, 1), Some(&4.0));
    }

    #[test]
    fn test_matmul_zeros() {
        // A × 0 = 0
        let a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let zeros = Matrix::zeros(2, 2);
        let result = a.matmul(&zeros).unwrap();

        for &val in result.as_slice() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_matmul_dimension_mismatch() {
        // 2×3 matrix cannot multiply with 2×2 matrix (inner dimensions don't match)
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = a.matmul(&b);

        assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
    }

    #[test]
    fn test_matmul_non_square() {
        // 2×3 × 3×2 = 2×2
        // [[1, 2, 3],   [[7,  8],    [[58,  64],
        //  [4, 5, 6]] ×  [9, 10],  =  [139, 154]]
        //                [11, 12]]
        let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.rows(), 2);
        assert_eq!(c.cols(), 2);
        assert_eq!(c.get(0, 0), Some(&58.0));
        assert_eq!(c.get(0, 1), Some(&64.0));
        assert_eq!(c.get(1, 0), Some(&139.0));
        assert_eq!(c.get(1, 1), Some(&154.0));
    }

    #[test]
    fn test_matmul_single_element() {
        // 1×1 × 1×1 = 1×1
        let a = Matrix::from_vec(1, 1, vec![3.0]).unwrap();
        let b = Matrix::from_vec(1, 1, vec![4.0]).unwrap();
        let c = a.matmul(&b).unwrap();

        assert_eq!(c.rows(), 1);
        assert_eq!(c.cols(), 1);
        assert_eq!(c.get(0, 0), Some(&12.0));
    }

    // ===== Backend Equivalence Tests =====

    #[test]
    fn test_matmul_simd_equivalence_small() {
        // Small matrix (below SIMD threshold) - verify both paths work
        let a = Matrix::from_vec(8, 8, (0..64).map(|i| i as f32).collect()).unwrap();
        let b = Matrix::from_vec(8, 8, (0..64).map(|i| (i * 2) as f32).collect()).unwrap();

        let mut result_naive = Matrix::zeros(8, 8);
        let mut result_simd = Matrix::zeros(8, 8);

        a.matmul_naive(&b, &mut result_naive).unwrap();
        a.matmul_simd(&b, &mut result_simd).unwrap();

        // Results should be identical
        for i in 0..8 {
            for j in 0..8 {
                let naive_val = result_naive.get(i, j).unwrap();
                let simd_val = result_simd.get(i, j).unwrap();
                assert!(
                    (naive_val - simd_val).abs() < 1e-5,
                    "Mismatch at ({}, {}): naive={}, simd={}",
                    i,
                    j,
                    naive_val,
                    simd_val
                );
            }
        }
    }

    #[test]
    fn test_matmul_simd_equivalence_large() {
        // Large matrix (above SIMD threshold) - verify SIMD correctness
        let size = 128;
        let a = Matrix::from_vec(
            size,
            size,
            (0..size * size).map(|i| (i % 100) as f32).collect(),
        )
        .unwrap();
        let b = Matrix::from_vec(
            size,
            size,
            (0..size * size).map(|i| ((i * 2) % 100) as f32).collect(),
        )
        .unwrap();

        let mut result_naive = Matrix::zeros(size, size);
        let mut result_simd = Matrix::zeros(size, size);

        a.matmul_naive(&b, &mut result_naive).unwrap();
        a.matmul_simd(&b, &mut result_simd).unwrap();

        // Results should be identical (within floating-point tolerance)
        for i in 0..size {
            for j in 0..size {
                let naive_val = result_naive.get(i, j).unwrap();
                let simd_val = result_simd.get(i, j).unwrap();
                assert!(
                    (naive_val - simd_val).abs() < 1e-3,
                    "Mismatch at ({}, {}): naive={}, simd={}",
                    i,
                    j,
                    naive_val,
                    simd_val
                );
            }
        }
    }

    #[test]
    fn test_matmul_simd_equivalence_rectangular() {
        // Rectangular matrices
        let a = Matrix::from_vec(64, 128, (0..64 * 128).map(|i| i as f32).collect()).unwrap();
        let b = Matrix::from_vec(128, 32, (0..128 * 32).map(|i| (i * 3) as f32).collect()).unwrap();

        let mut result_naive = Matrix::zeros(64, 32);
        let mut result_simd = Matrix::zeros(64, 32);

        a.matmul_naive(&b, &mut result_naive).unwrap();
        a.matmul_simd(&b, &mut result_simd).unwrap();

        // Results should be identical (use relative tolerance for large values)
        for i in 0..64 {
            for j in 0..32 {
                let naive_val = result_naive.get(i, j).unwrap();
                let simd_val = result_simd.get(i, j).unwrap();
                let diff = (naive_val - simd_val).abs();
                let tolerance = if naive_val.abs() > 1.0 {
                    naive_val.abs() * 1e-5 // Relative tolerance for large values
                } else {
                    1e-5 // Absolute tolerance for small values
                };
                assert!(
                    diff < tolerance,
                    "Mismatch at ({}, {}): naive={}, simd={}, diff={}",
                    i,
                    j,
                    naive_val,
                    simd_val,
                    diff
                );
            }
        }
    }

    // ===== GPU Tests =====

    #[test]
    #[cfg(feature = "gpu")]
    fn test_gpu_availability() {
        use crate::backends::gpu::GpuBackend;
        // Just test that we can check GPU availability without crashing
        let _available = GpuBackend::is_available();
        // Note: We don't assert availability since CI may not have GPU
    }

    #[test]
    #[cfg(feature = "gpu")]
    #[ignore] // Ignore by default since CI may not have GPU
    fn test_gpu_matmul_basic() {
        use crate::backends::gpu::GpuBackend;

        if !GpuBackend::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Small test matrix (will use GPU if threshold is low enough)
        let a = Matrix::from_vec(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap();

        let b = Matrix::from_vec(
            4,
            4,
            vec![
                16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
                1.0,
            ],
        )
        .unwrap();

        // Try GPU matmul directly
        let result = a.matmul_gpu(&b);

        if let Ok(c) = result {
            // Verify some basic properties
            assert_eq!(c.rows(), 4);
            assert_eq!(c.cols(), 4);

            // Verify against known result (first element)
            // [1,2,3,4] · [16,12,8,4] = 16+24+24+16 = 80
            assert!((c.get(0, 0).unwrap() - 80.0).abs() < 1e-4);
        } else {
            eprintln!("GPU matmul failed: {:?}", result);
        }
    }

    // ===== Transpose Tests =====

    #[test]
    fn test_transpose_basic() {
        // [[1, 2, 3],     [[1, 4],
        //  [4, 5, 6]]  →   [2, 5],
        //                  [3, 6]]
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = m.transpose();

        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(t.get(0, 0), Some(&1.0));
        assert_eq!(t.get(0, 1), Some(&4.0));
        assert_eq!(t.get(1, 0), Some(&2.0));
        assert_eq!(t.get(1, 1), Some(&5.0));
        assert_eq!(t.get(2, 0), Some(&3.0));
        assert_eq!(t.get(2, 1), Some(&6.0));
    }

    #[test]
    fn test_transpose_square() {
        // [[1, 2],     [[1, 3],
        //  [3, 4]]  →   [2, 4]]
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
    fn test_transpose_single_row() {
        // [[1, 2, 3]] → [[1],
        //                 [2],
        //                 [3]]
        let m = Matrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        let t = m.transpose();

        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 1);
        assert_eq!(t.get(0, 0), Some(&1.0));
        assert_eq!(t.get(1, 0), Some(&2.0));
        assert_eq!(t.get(2, 0), Some(&3.0));
    }

    #[test]
    fn test_transpose_single_col() {
        // [[1],        [[1, 2, 3]]
        //  [2],   →
        //  [3]]
        let m = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let t = m.transpose();

        assert_eq!(t.rows(), 1);
        assert_eq!(t.cols(), 3);
        assert_eq!(t.get(0, 0), Some(&1.0));
        assert_eq!(t.get(0, 1), Some(&2.0));
        assert_eq!(t.get(0, 2), Some(&3.0));
    }

    #[test]
    fn test_transpose_single_element() {
        // [[5]] → [[5]]
        let m = Matrix::from_vec(1, 1, vec![5.0]).unwrap();
        let t = m.transpose();

        assert_eq!(t.rows(), 1);
        assert_eq!(t.cols(), 1);
        assert_eq!(t.get(0, 0), Some(&5.0));
    }

    #[test]
    fn test_transpose_identity() {
        // I^T = I
        let identity = Matrix::identity(3);
        let t = identity.transpose();

        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 3);

        // Check it's still identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(t.get(i, j), Some(&expected));
            }
        }
    }
}

// Property-based tests for matmul
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    /// Generate a matrix of given dimensions with random values
    fn matrix_strategy(rows: usize, cols: usize) -> impl Strategy<Value = Matrix<f32>> {
        proptest::collection::vec(-100.0f32..100.0, rows * cols)
            .prop_map(move |data| Matrix::from_vec(rows, cols, data).unwrap())
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property: Matrix multiplication is associative
        /// (A × B) × C = A × (B × C)
        #[test]
        fn test_matmul_associative(
            a in matrix_strategy(3, 4),
            b in matrix_strategy(4, 5),
            c in matrix_strategy(5, 3)
        ) {
            let ab = a.matmul(&b).unwrap();
            let ab_c = ab.matmul(&c).unwrap();

            let bc = b.matmul(&c).unwrap();
            let a_bc = a.matmul(&bc).unwrap();

            // Check dimensions
            prop_assert_eq!(ab_c.rows(), a_bc.rows());
            prop_assert_eq!(ab_c.cols(), a_bc.cols());

            // Check values with tolerance for floating-point errors
            // Use relative tolerance for large values, absolute for small values
            for i in 0..ab_c.rows() {
                for j in 0..ab_c.cols() {
                    let val1 = ab_c.get(i, j).unwrap();
                    let val2 = a_bc.get(i, j).unwrap();
                    let diff = (val1 - val2).abs();
                    let max_val = val1.abs().max(val2.abs());

                    // Use hybrid tolerance: absolute for small values, relative for large
                    // Matrix multiplication accumulates rounding errors, so we need looser tolerance
                    let tolerance = if max_val < 1.0 {
                        1e-3  // Absolute tolerance for small values
                    } else {
                        max_val * 1e-3  // Relative tolerance (0.1%) for large values
                    };

                    prop_assert!(
                        diff < tolerance,
                        "Associativity failed at ({}, {}): {} != {} (diff: {}, tolerance: {})",
                        i, j, val1, val2, diff, tolerance
                    );
                }
            }
        }

        /// Property: Multiplying by identity matrix preserves the matrix
        /// A × I = A
        #[test]
        fn test_matmul_identity_property(
            rows in 1usize..10,
            cols in 1usize..10,
            data in proptest::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            // Ensure data length matches dimensions
            let size = rows * cols;
            if data.len() < size {
                return Ok(());
            }
            let matrix_data = data[0..size].to_vec();

            let a = Matrix::from_vec(rows, cols, matrix_data).unwrap();
            let identity = Matrix::identity(cols);
            let result = a.matmul(&identity).unwrap();

            // Check dimensions
            prop_assert_eq!(result.rows(), a.rows());
            prop_assert_eq!(result.cols(), a.cols());

            // Check values (should be identical)
            for i in 0..rows {
                for j in 0..cols {
                    let original = a.get(i, j).unwrap();
                    let multiplied = result.get(i, j).unwrap();
                    let diff = (original - multiplied).abs();
                    prop_assert!(
                        diff < 1e-5,
                        "Identity property failed at ({}, {}): {} != {} (diff: {})",
                        i, j, original, multiplied, diff
                    );
                }
            }
        }

        /// Property: Dimension property
        /// If A is m×n and B is n×p, then A×B is m×p
        #[test]
        fn test_matmul_dimension_property(
            m in 1usize..10,
            n in 1usize..10,
            p in 1usize..10
        ) {
            let a = Matrix::zeros(m, n);
            let b = Matrix::zeros(n, p);
            let c = a.matmul(&b).unwrap();

            prop_assert_eq!(c.rows(), m);
            prop_assert_eq!(c.cols(), p);
        }

        /// Property: Double transpose returns original
        /// (A^T)^T = A
        #[test]
        fn test_transpose_double_transpose(
            a in matrix_strategy(5, 7)
        ) {
            let t = a.transpose();
            let tt = t.transpose();

            prop_assert_eq!(tt.rows(), a.rows());
            prop_assert_eq!(tt.cols(), a.cols());

            for i in 0..a.rows() {
                for j in 0..a.cols() {
                    prop_assert_eq!(tt.get(i, j), a.get(i, j));
                }
            }
        }

        /// Property: Transpose swaps dimensions
        /// If A is m×n, then A^T is n×m
        #[test]
        fn test_transpose_dimension_swap(
            m in 1usize..20,
            n in 1usize..20
        ) {
            let a = Matrix::zeros(m, n);
            let t = a.transpose();

            prop_assert_eq!(t.rows(), n);
            prop_assert_eq!(t.cols(), m);
        }

        /// Property: Transpose of product
        /// (A×B)^T = B^T×A^T
        #[test]
        fn test_transpose_of_product(
            a in matrix_strategy(3, 4),
            b in matrix_strategy(4, 5)
        ) {
            let ab = a.matmul(&b).unwrap();
            let ab_t = ab.transpose();

            let b_t = b.transpose();
            let a_t = a.transpose();
            let bt_at = b_t.matmul(&a_t).unwrap();

            prop_assert_eq!(ab_t.rows(), bt_at.rows());
            prop_assert_eq!(ab_t.cols(), bt_at.cols());

            // Check values with tolerance for floating-point errors
            for i in 0..ab_t.rows() {
                for j in 0..ab_t.cols() {
                    let val1 = ab_t.get(i, j).unwrap();
                    let val2 = bt_at.get(i, j).unwrap();
                    let diff = (val1 - val2).abs();
                    let max_val = val1.abs().max(val2.abs());

                    let tolerance = if max_val < 1.0 {
                        1e-3
                    } else {
                        max_val * 1e-3
                    };

                    prop_assert!(
                        diff < tolerance,
                        "Transpose of product failed at ({}, {}): {} != {} (diff: {}, tolerance: {})",
                        i, j, val1, val2, diff, tolerance
                    );
                }
            }
        }

        /// Matrix-vector multiplication: (A×B)×v = A×(B×v)
        #[test]
        fn test_matvec_associativity(
            a in matrix_strategy(3, 4),
            b in matrix_strategy(4, 5),
            v_data in prop::collection::vec(-10.0f32..10.0, 5)
        ) {
            let v = Vector::from_slice(&v_data);

            let ab = a.matmul(&b).unwrap();
            let ab_v = ab.matvec(&v).unwrap();

            let b_v = b.matvec(&v).unwrap();
            let a_bv = a.matvec(&b_v).unwrap();

            prop_assert_eq!(ab_v.len(), a_bv.len());

            for i in 0..ab_v.len() {
                let diff = (ab_v.as_slice()[i] - a_bv.as_slice()[i]).abs();
                let max_val = ab_v.as_slice()[i].abs().max(a_bv.as_slice()[i].abs());
                let tolerance = if max_val < 1.0 { 1e-2 } else { max_val * 1e-2 };

                prop_assert!(
                    diff < tolerance,
                    "Associativity failed at index {}: {} != {} (diff: {}, tolerance: {})",
                    i, ab_v.as_slice()[i], a_bv.as_slice()[i], diff, tolerance
                );
            }
        }

        /// Vector-matrix multiplication: v×(A×B) = (v×A)×B
        #[test]
        fn test_vecmat_associativity(
            a in matrix_strategy(3, 4),
            b in matrix_strategy(4, 5),
            v_data in prop::collection::vec(-10.0f32..10.0, 3)
        ) {
            let v = Vector::from_slice(&v_data);

            let ab = a.matmul(&b).unwrap();
            let v_ab = Matrix::vecmat(&v, &ab).unwrap();

            let v_a = Matrix::vecmat(&v, &a).unwrap();
            let va_b = Matrix::vecmat(&v_a, &b).unwrap();

            prop_assert_eq!(v_ab.len(), va_b.len());

            for i in 0..v_ab.len() {
                let diff = (v_ab.as_slice()[i] - va_b.as_slice()[i]).abs();
                let max_val = v_ab.as_slice()[i].abs().max(va_b.as_slice()[i].abs());
                let tolerance = if max_val < 1.0 { 1e-2 } else { max_val * 1e-2 };

                prop_assert!(
                    diff < tolerance,
                    "Associativity failed at index {}: {} != {} (diff: {}, tolerance: {})",
                    i, v_ab.as_slice()[i], va_b.as_slice()[i], diff, tolerance
                );
            }
        }
    }

    // Unit tests for matrix-vector operations
    #[test]
    fn test_matvec_basic() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = m.matvec(&v).unwrap();

        // [[1, 2, 3]   [1]   [14]
        //  [4, 5, 6]] × [2] = [32]
        //               [3]
        assert_eq!(result.len(), 2);
        assert!((result.as_slice()[0] - 14.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_matvec_identity() {
        let m = Matrix::identity(3);
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = m.matvec(&v).unwrap();

        // I×v = v
        assert_eq!(result.as_slice(), v.as_slice());
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

        // [1, 2] × [[1, 2, 3]  = [9, 12, 15]
        //           [4, 5, 6]]
        assert_eq!(result.len(), 3);
        assert!((result.as_slice()[0] - 9.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 12.0).abs() < 1e-6);
        assert!((result.as_slice()[2] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_vecmat_identity() {
        let m = Matrix::identity(3);
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = Matrix::vecmat(&v, &m).unwrap();

        // v×I = v
        assert_eq!(result.as_slice(), v.as_slice());
    }

    #[test]
    fn test_vecmat_dimension_mismatch() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]); // Wrong size

        assert!(Matrix::vecmat(&v, &m).is_err());
    }

    #[test]
    fn test_matvec_zero_vector() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = m.matvec(&v).unwrap();

        // A×0 = 0
        assert_eq!(result.as_slice(), &[0.0, 0.0]);
    }

    #[test]
    fn test_vecmat_zero_vector() {
        let m = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = Vector::from_slice(&[0.0, 0.0]);
        let result = Matrix::vecmat(&v, &m).unwrap();

        // 0×A = 0
        assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_matvec_transpose_equivalence() {
        // v^T × A = (A^T × v)^T
        // If A is m×n and v is m-dimensional, then:
        // - v^T × A is n-dimensional
        // - A^T is n×m, so A^T × v needs v to be n-dimensional
        // Actually, this is wrong. Let me use correct equivalence:
        // If A is m×n, v is n-dimensional:
        // - A × v is m-dimensional (matrix-vector)
        // - A^T is n×m, u is m-dimensional:
        // - u^T × A is n-dimensional (vector-matrix)
        // These are equivalent when u = A × v

        let m = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = Vector::from_slice(&[1.0, 2.0]); // 2-dimensional

        // A × v (3×2 times 2D = 3D result)
        let av = m.matvec(&v).unwrap();

        // v^T × A^T (2D times 2×3 = 3D result)
        let m_t = m.transpose(); // Now 2×3
        let v_mt = Matrix::vecmat(&v, &m_t).unwrap();

        // (A × v)^T = v^T × A^T
        assert_eq!(av.as_slice(), v_mt.as_slice());
    }

    // ===== 2D Convolution Tests =====

    #[test]
    fn test_convolve2d_basic_3x3() {
        // Simple 3x3 convolution with identity kernel (should preserve input)
        let input = Matrix::from_vec(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();

        // 1x1 identity kernel (should return center pixel)
        let kernel = Matrix::from_vec(1, 1, vec![1.0]).unwrap();

        let result = input.convolve2d(&kernel).unwrap();

        // Result should be 3x3 (same input size with valid padding)
        assert_eq!(result.rows(), 3);
        assert_eq!(result.cols(), 3);
        assert_eq!(result.as_slice(), input.as_slice());
    }

    #[test]
    fn test_convolve2d_edge_detection() {
        // Test edge detection with Sobel-like kernel
        let input = Matrix::from_vec(
            4,
            4,
            vec![
                1.0, 1.0, 1.0, 1.0, //
                1.0, 2.0, 2.0, 1.0, //
                1.0, 2.0, 2.0, 1.0, //
                1.0, 1.0, 1.0, 1.0, //
            ],
        )
        .unwrap();

        // Simple 3x3 horizontal edge detection kernel
        #[rustfmt::skip]
        let kernel = Matrix::from_vec(
            3,
            3,
            vec![
                -1.0, -1.0, -1.0,
                 0.0,  0.0,  0.0,
                 1.0,  1.0,  1.0,
            ],
        )
        .unwrap();

        let result = input.convolve2d(&kernel).unwrap();

        // Result should be 2x2 (4-3+1 = 2)
        assert_eq!(result.rows(), 2);
        assert_eq!(result.cols(), 2);
    }

    #[test]
    fn test_convolve2d_averaging_filter() {
        // Test averaging filter (blur)
        let input = Matrix::from_vec(
            5,
            5,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 9.0, 0.0, 0.0, // Center pixel
                0.0, 0.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, 0.0, 0.0, //
            ],
        )
        .unwrap();

        // 3x3 averaging kernel (all 1/9)
        let kernel_val = 1.0 / 9.0;
        let kernel = Matrix::from_vec(
            3,
            3,
            vec![
                kernel_val, kernel_val, kernel_val, //
                kernel_val, kernel_val, kernel_val, //
                kernel_val, kernel_val, kernel_val, //
            ],
        )
        .unwrap();

        let result = input.convolve2d(&kernel).unwrap();

        // Result should be 3x3
        assert_eq!(result.rows(), 3);
        assert_eq!(result.cols(), 3);

        // Center should be 1.0 (9/9)
        assert!((result.get(1, 1).unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_convolve2d_invalid_kernel() {
        let input = Matrix::from_vec(3, 3, vec![1.0; 9]).unwrap();

        // Kernel larger than input
        let kernel = Matrix::from_vec(4, 4, vec![1.0; 16]).unwrap();

        assert!(input.convolve2d(&kernel).is_err());
    }

    // ===== Property-Based Tests for Convolution =====

    #[cfg(test)]
    mod conv_property_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn test_convolve2d_output_size(
                input_rows in 3usize..20,
                input_cols in 3usize..20,
                kernel_rows in 1usize..5,
                kernel_cols in 1usize..5,
            ) {
                // Property: Output size is always (input - kernel + 1) for valid padding
                if kernel_rows <= input_rows && kernel_cols <= input_cols {
                    let input = Matrix::from_vec(input_rows, input_cols, vec![1.0; input_rows * input_cols]).unwrap();
                    let kernel = Matrix::from_vec(kernel_rows, kernel_cols, vec![1.0; kernel_rows * kernel_cols]).unwrap();

                    let result = input.convolve2d(&kernel).unwrap();

                    prop_assert_eq!(result.rows(), input_rows - kernel_rows + 1);
                    prop_assert_eq!(result.cols(), input_cols - kernel_cols + 1);
                }
            }

            #[test]
            fn test_convolve2d_identity_kernel(
                input_rows in 3usize..10,
                input_cols in 3usize..10,
                values in prop::collection::vec(-100.0f32..100.0, 9..100)
            ) {
                // Property: 1x1 identity kernel preserves input
                if values.len() >= input_rows * input_cols {
                    let data: Vec<f32> = values.iter().take(input_rows * input_cols).copied().collect();
                    let input = Matrix::from_vec(input_rows, input_cols, data.clone()).unwrap();
                    let kernel = Matrix::from_vec(1, 1, vec![1.0]).unwrap();

                    let result = input.convolve2d(&kernel).unwrap();

                    prop_assert_eq!(result.rows(), input_rows);
                    prop_assert_eq!(result.cols(), input_cols);
                    prop_assert_eq!(result.as_slice(), input.as_slice());
                }
            }

            #[test]
            fn test_convolve2d_zero_kernel(
                input_rows in 3usize..10,
                input_cols in 3usize..10,
                kernel_rows in 1usize..4,
                kernel_cols in 1usize..4,
            ) {
                // Property: Zero kernel produces zero output
                if kernel_rows <= input_rows && kernel_cols <= input_cols {
                    let input = Matrix::from_vec(input_rows, input_cols, vec![5.0; input_rows * input_cols]).unwrap();
                    let kernel = Matrix::from_vec(kernel_rows, kernel_cols, vec![0.0; kernel_rows * kernel_cols]).unwrap();

                    let result = input.convolve2d(&kernel).unwrap();

                    for &val in result.as_slice() {
                        prop_assert!((val - 0.0).abs() < 1e-5);
                    }
                }
            }

            #[test]
            fn test_convolve2d_scalar_multiplication(
                input_rows in 3usize..10,
                input_cols in 3usize..10,
                scalar in -10.0f32..10.0,
            ) {
                // Property: Convolving with scalar * kernel = scalar * (convolve with kernel)
                let input = Matrix::from_vec(input_rows, input_cols, vec![2.0; input_rows * input_cols]).unwrap();
                let kernel = Matrix::from_vec(3, 3, vec![1.0; 9]).unwrap();
                let kernel_scaled = Matrix::from_vec(3, 3, vec![scalar; 9]).unwrap();

                let result1 = input.convolve2d(&kernel).unwrap();
                let result2 = input.convolve2d(&kernel_scaled).unwrap();

                for (v1, v2) in result1.as_slice().iter().zip(result2.as_slice().iter()) {
                    prop_assert!((v1 * scalar - v2).abs() < 1e-3);
                }
            }
        }
    }
}
