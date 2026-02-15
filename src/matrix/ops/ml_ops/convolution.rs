//! 2D convolution operations for Matrix

use crate::TruenoError;

use super::super::super::Matrix;

impl Matrix<f32> {
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
    ///
    /// # Example
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    /// )?;
    ///
    /// // 3x3 averaging kernel
    /// let kernel_val = 1.0 / 9.0;
    /// let kernel = Matrix::from_vec(
    ///     3, 3,
    ///     vec![kernel_val; 9]
    /// )?;
    ///
    /// let result = input.convolve2d(&kernel)?;
    /// assert_eq!(result.rows(), 3); // 5 - 3 + 1
    /// assert_eq!(result.cols(), 3);
    /// # Ok(())
    /// # }
    /// ```
    // =========================================================================
    // HOT PATH - PERFORMANCE CRITICAL
    // =========================================================================
    // This function processes millions of elements for typical image sizes.
    // Any changes to the inner loop REQUIRE benchmark verification.
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

        // GPU acceleration for large convolutions
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = 10_000;

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
                }
            }
        }

        // Scalar baseline implementation - optimized with direct indexing
        let input_data = self.as_slice();
        let kernel_data = kernel.as_slice();
        let result_data = result.data.as_mut_slice();
        let input_cols = self.cols;
        let kernel_cols = kernel.cols;
        let result_cols = output_cols;

        for out_row in 0..output_rows {
            for out_col in 0..output_cols {
                let mut sum = 0.0;

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
}
