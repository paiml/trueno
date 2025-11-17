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

use crate::{Backend, TruenoError};

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

        // Naive O(n³) matrix multiplication: C[i,j] = Σ A[i,k] × B[k,j]
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k).unwrap() * other.get(k, j).unwrap();
                }
                *result.get_mut(i, j).unwrap() = sum;
            }
        }

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
        proptest::collection::vec(-100.0f32..100.0, rows * cols).prop_map(move |data| {
            Matrix::from_vec(rows, cols, data).unwrap()
        })
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
    }
}
