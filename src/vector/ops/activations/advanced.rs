//! Advanced activation functions: leaky_relu, elu, gelu, swish, hardswish, mish, selu
//!
//! These are separated from the basic activations (softmax, log_softmax, relu, sigmoid)
//! for file size management while remaining part of the same `activations` module.

use crate::backends::scalar::ScalarBackend;
use crate::backends::VectorBackend;
use crate::vector::Vector;
use crate::{Backend, Result, TruenoError};

use super::dispatch_unary_op;

impl Vector<f32> {
    /// Leaky ReLU activation function
    ///
    /// Computes the element-wise Leaky ReLU with a configurable negative slope.
    /// Leaky ReLU addresses the "dying ReLU" problem by allowing small negative values.
    ///
    /// # Formula
    ///
    /// ```text
    /// leaky_relu(x, α)[i] = max(αx\[i\], x\[i\])
    ///                     = x\[i\]    if x\[i\] > 0
    ///                     = αx\[i\]   if x\[i\] ≤ 0
    /// ```
    ///
    /// # Parameters
    ///
    /// - `negative_slope`: The slope for negative values (typically 0.01)
    ///   - Must be in range [0.0, 1.0)
    ///   - Common values: 0.01 (default), 0.1, 0.2
    ///   - α = 0 reduces to standard ReLU
    ///   - α = 1 reduces to identity function
    ///
    /// # Properties
    ///
    /// - **Fixes dying ReLU**: Neurons can't completely die (always has gradient)
    /// - **Non-zero gradient**: Gradient is α for negative inputs (not zero)
    /// - **Unbounded positive**: No saturation for positive values
    /// - **Parameterized**: Negative slope can be tuned or learned (PReLU)
    ///
    /// # Applications
    ///
    /// - **Deep networks**: Prevents dying neurons in very deep networks
    /// - **GANs**: Often used in generator and discriminator networks
    /// - **Better gradient flow**: Helps with vanishing gradient problem
    /// - **Empirical improvements**: Often outperforms ReLU in practice
    ///
    /// # Performance
    ///
    /// This operation is memory-bound (simple multiplication and comparison).
    /// SIMD provides modest speedups.
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    /// Returns `InvalidInput` if negative_slope is not in [0.0, 1.0).
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.leaky_relu(0.01)?;
    ///
    /// // Negative values multiplied by 0.01, positive unchanged
    /// assert_eq!(result.as_slice(), &[-0.02, -0.01, 0.0, 1.0, 2.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn leaky_relu(&self, negative_slope: f32) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // Validate negative_slope parameter
        if !(0.0..1.0).contains(&negative_slope) {
            return Err(TruenoError::InvalidInput(format!(
                "negative_slope must be in [0.0, 1.0), got {}",
                negative_slope
            )));
        }

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu
                        .leaky_relu(&self.data, &mut result, negative_slope)
                        .is_ok()
                    {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Scalar fallback: leaky_relu(x, α) = x if x > 0, αx otherwise
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| if x > 0.0 { x } else { negative_slope * x })
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// ELU (Exponential Linear Unit) activation function
    ///
    /// Computes the element-wise ELU with a configurable alpha parameter.
    /// ELU pushes mean activations closer to zero, improving learning.
    ///
    /// # Formula
    ///
    /// ```text
    /// elu(x, α)[i] = x\[i\]           if x\[i\] > 0
    ///              = α(e^x\[i\] - 1)  if x\[i\] ≤ 0
    /// ```
    ///
    /// # Parameters
    ///
    /// - `alpha`: Controls the saturation value for negative inputs (typically 1.0)
    ///   - Must be > 0
    ///   - Common value: 1.0 (original ELU paper)
    ///   - Larger α → slower saturation for negative inputs
    ///
    /// # Properties
    ///
    /// - **Smooth**: Unlike ReLU/Leaky ReLU, has smooth gradients everywhere
    /// - **Negative values**: Allows negative outputs (pushes mean closer to zero)
    /// - **Bounded below**: Saturates to -α for very negative inputs
    /// - **Unbounded above**: No saturation for positive values
    /// - **Non-zero gradient**: Has gradient everywhere (no dead neurons)
    ///
    /// # Applications
    ///
    /// - **Deep networks**: Better gradient flow than ReLU
    /// - **Mean activation near zero**: Reduces internal covariate shift
    /// - **Noise robustness**: Smooth activation helps with noisy gradients
    /// - **Empirical improvements**: Often outperforms ReLU and Leaky ReLU
    ///
    /// # Performance
    ///
    /// This operation is compute-bound due to exp() for negative values.
    /// More expensive than ReLU/Leaky ReLU but provides better properties.
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    /// Returns `InvalidInput` if alpha <= 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.elu(1.0)?;
    ///
    /// // Negative values: α(e^x - 1), positive unchanged
    /// // elu(-2, 1) ≈ -0.865, elu(-1, 1) ≈ -0.632
    /// assert!((result.as_slice()[0] - (-0.865)).abs() < 0.01);
    /// assert!((result.as_slice()[1] - (-0.632)).abs() < 0.01);
    /// assert_eq!(result.as_slice()[2], 0.0);
    /// assert_eq!(result.as_slice()[3], 1.0);
    /// assert_eq!(result.as_slice()[4], 2.0);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn elu(&self, alpha: f32) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // Validate alpha parameter
        if alpha <= 0.0 {
            return Err(TruenoError::InvalidInput(format!(
                "alpha must be > 0, got {}",
                alpha
            )));
        }

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.elu(&self.data, &mut result, alpha).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Scalar fallback: elu(x, α) = x if x > 0, α(e^x - 1) otherwise
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// GELU (Gaussian Error Linear Unit) activation function
    ///
    /// Computes the element-wise GELU activation using the tanh approximation.
    /// GELU is the activation function used in transformers (BERT, GPT, etc.).
    ///
    /// # Formula
    ///
    /// ```text
    /// gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    /// ```
    ///
    /// This is the tanh approximation which is faster than the exact form
    /// involving the error function (erf).
    ///
    /// # Properties
    ///
    /// - **Smooth**: Infinitely differentiable everywhere
    /// - **Non-monotonic**: Unlike ReLU variants, has slight non-monotonicity near zero
    /// - **Stochastic regularizer**: Can be viewed as adaptive dropout
    /// - **Zero-centered**: Mean activation close to zero
    /// - **Bounded below**: Approaches 0 as x → -∞
    /// - **Unbounded above**: Linear growth for large positive x
    ///
    /// # Applications
    ///
    /// - **Transformers**: BERT, GPT-2, GPT-3, GPT-4 (default activation)
    /// - **Vision transformers**: ViT, DINO, MAE
    /// - **Modern architectures**: State-of-the-art NLP and vision models
    /// - **Better than ReLU**: Empirically outperforms ReLU in many tasks
    ///
    /// # Performance
    ///
    /// This operation is compute-intensive (tanh, x³ calculations).
    /// More expensive than ReLU but comparable to ELU.
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.gelu()?;
    ///
    /// // GELU is smooth and non-monotonic near zero
    /// assert!(result.as_slice()[0] < 0.0); // Negative inputs → small negative outputs
    /// assert_eq!(result.as_slice()[2], 0.0); // gelu(0) = 0
    /// assert!(result.as_slice()[4] > 1.5); // Large positive → ~linear
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn gelu(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.gelu(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        let mut result = vec![0.0; self.len()];

        // Dispatch to appropriate backend
        dispatch_unary_op!(self.backend, gelu, &self.data, &mut result);

        Ok(Vector::from_vec(result))
    }

    /// Swish activation function (also known as SiLU - Sigmoid Linear Unit)
    ///
    /// Applies the Swish activation element-wise: swish(x) = x * sigmoid(x) = x / (1 + e^(-x)).
    ///
    /// Swish is a smooth, non-monotonic activation function that consistently matches or
    /// outperforms ReLU in deep networks. It's used in EfficientNet, MobileNet v3, and
    /// many modern architectures. The function is self-gated: it adaptively gates the
    /// input based on its value.
    ///
    /// Properties:
    /// - Smooth and differentiable everywhere
    /// - Non-monotonic: has a slight "dip" for negative values
    /// - swish(0) = 0
    /// - swish(x) ≈ x for large positive x (linear)
    /// - swish(x) ≈ 0 for large negative x
    /// - Unbounded above, bounded below by ≈ -0.278 at x ≈ -1.278
    ///
    /// # Performance
    ///
    /// Compute-bound operation requiring exponential and division.
    /// Future SIMD optimizations planned for Phase 9 (GPU backend).
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.swish()?;
    ///
    /// // Swish is smooth and self-gated
    /// assert!(result.as_slice()[0] < 0.0); // Negative inputs → small negative outputs
    /// assert_eq!(result.as_slice()[2], 0.0); // swish(0) = 0
    /// assert!(result.as_slice()[4] > 1.5); // Large positive → ~linear
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # References
    ///
    /// - Ramachandran et al. (2017): "Searching for Activation Functions"
    /// - Also known as SiLU (Sigmoid Linear Unit): Elfwing et al. (2018)
    pub fn swish(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.swish(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        let mut result = vec![0.0; self.len()];

        // Dispatch to appropriate SIMD backend
        dispatch_unary_op!(self.backend, swish, &self.data, &mut result);

        Ok(Vector::from_vec(result))
    }

    /// Hard Swish activation function
    ///
    /// Applies the hardswish activation element-wise: hardswish(x) = x * relu6(x + 3) / 6
    ///
    /// Hardswish is a piece-wise linear approximation to swish, designed for efficient
    /// computation in mobile neural networks. It's used in MobileNetV3 and avoids the
    /// expensive sigmoid computation of standard swish.
    ///
    /// Properties:
    /// - Piece-wise linear: efficient to compute
    /// - hardswish(x) = 0 for x ≤ -3
    /// - hardswish(x) = x for x ≥ 3
    /// - hardswish(x) = x * (x + 3) / 6 for -3 < x < 3
    /// - hardswish(0) = 0
    /// - Smooth transitions at boundaries
    ///
    /// # Performance
    ///
    /// More efficient than swish as it uses only multiply/divide operations
    /// instead of expensive exponential functions. Ideal for inference on
    /// resource-constrained devices.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-4.0, -3.0, 0.0, 3.0, 4.0]);
    /// let result = v.hardswish()?;
    ///
    /// // Piece-wise linear behavior
    /// assert_eq!(result.as_slice()[0], 0.0); // x ≤ -3 → 0
    /// assert_eq!(result.as_slice()[1], 0.0); // x = -3 → 0
    /// assert_eq!(result.as_slice()[2], 0.0); // x = 0 → 0
    /// assert_eq!(result.as_slice()[3], 3.0); // x = 3 → x
    /// assert_eq!(result.as_slice()[4], 4.0); // x ≥ 3 → x
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # References
    ///
    /// - Howard et al. (2019): "Searching for MobileNetV3"
    pub fn hardswish(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // Scalar implementation: hardswish(x) = x * relu6(x + 3) / 6
        // Simplified piece-wise:
        // - x <= -3: 0
        // - x >= 3: x
        // - else: x * (x + 3) / 6
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                if x <= -3.0 {
                    0.0
                } else if x >= 3.0 {
                    x
                } else {
                    x * (x + 3.0) / 6.0
                }
            })
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// Mish activation function
    ///
    /// Applies the mish activation element-wise: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    ///
    /// Mish is a self-regularizing non-monotonic activation function that often outperforms
    /// ReLU and swish in computer vision tasks. It's used in YOLOv4 and many modern architectures.
    ///
    /// Properties:
    /// - Smooth and non-monotonic (similar to swish)
    /// - Self-regularizing: prevents dying neurons
    /// - mish(0) ≈ 0 (small positive value)
    /// - mish(x) ≈ x for large positive x (nearly linear)
    /// - mish(x) ≈ 0 for large negative x
    /// - Bounded below by ≈ -0.31 at x ≈ -1.19
    ///
    /// # Performance
    ///
    /// Compute-bound operation requiring exponential, logarithm, and tanh.
    /// More expensive than ReLU/swish but often provides better accuracy.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.mish()?;
    ///
    /// // Mish is smooth and self-gated
    /// assert!(result.as_slice()[0] < 0.0); // Small negative output for negative inputs
    /// assert!(result.as_slice()[2].abs() < 1e-5); // mish(0) = 0
    /// assert!(result.as_slice()[4] > 1.5); // Large positive → near linear
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # References
    ///
    /// - Misra (2019): "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    pub fn mish(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // Scalar implementation: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                // Handle extreme values for numerical stability
                if x < -20.0 {
                    // For very negative x: softplus ≈ 0, tanh(0) ≈ 0, so mish ≈ 0
                    0.0
                } else if x > 20.0 {
                    // For very positive x: softplus ≈ x, tanh(x) ≈ 1, so mish ≈ x
                    x
                } else {
                    // Normal case: x * tanh(ln(1 + e^x))
                    let softplus = (1.0 + x.exp()).ln();
                    x * softplus.tanh()
                }
            })
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// SELU (Scaled Exponential Linear Unit) activation function
    ///
    /// Computes selu(x) = λ * (x if x > 0 else α * (exp(x) - 1))
    /// where λ ≈ 1.0507 and α ≈ 1.6733
    ///
    /// # Properties
    ///
    /// - **Self-normalizing**: Activations converge to zero mean and unit variance
    /// - **Vanishing gradient prevention**: Non-zero gradient for negative inputs
    /// - **Automatic normalization**: Reduces need for batch normalization
    ///
    /// # Performance
    ///
    /// Uses scalar implementation (GPU disabled for element-wise ops).
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.selu()?;
    ///
    /// // Positive values scaled by λ ≈ 1.0507
    /// assert!((result.as_slice()[3] - 1.0507).abs() < 0.001);
    /// assert!((result.as_slice()[4] - 2.1014).abs() < 0.001);
    ///
    /// // Zero stays zero
    /// assert!(result.as_slice()[2].abs() < 1e-5);
    ///
    /// // Negative values use ELU-like formula
    /// assert!(result.as_slice()[0] < 0.0);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # References
    ///
    /// - Klambauer et al. (2017): "Self-Normalizing Neural Networks"
    pub fn selu(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // SELU constants from Klambauer et al. (2017)
        // These specific values ensure self-normalizing property
        const LAMBDA: f32 = 1.0507009873554804934193349852946;
        const ALPHA: f32 = 1.6732632423543772848170429916717;

        // Scalar implementation: selu(x) = λ * (x if x > 0 else α * (exp(x) - 1))
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                if x > 0.0 {
                    LAMBDA * x
                } else {
                    LAMBDA * ALPHA * (x.exp() - 1.0)
                }
            })
            .collect();

        Ok(Vector::from_vec(data))
    }
}
