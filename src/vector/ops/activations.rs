//! Activation functions for Vector<f32>
//!
//! This module provides neural network activation functions optimized with
//! multi-backend SIMD support (Scalar, SSE2, AVX2, AVX-512, NEON, WASM SIMD).
//!
//! ## Activation Functions
//!
//! - [`softmax`](crate::Vector::softmax): Softmax normalization for classification
//! - [`log_softmax`](crate::Vector::log_softmax): Numerically stable log-softmax
//! - [`relu`](crate::Vector::relu): Rectified Linear Unit
//! - [`sigmoid`](crate::Vector::sigmoid): Logistic sigmoid
//! - [`leaky_relu`](crate::Vector::leaky_relu): Leaky ReLU with configurable slope
//! - [`elu`](crate::Vector::elu): Exponential Linear Unit
//! - [`gelu`](crate::Vector::gelu): Gaussian Error Linear Unit
//! - [`swish`](crate::Vector::swish): Self-gated activation (SiLU)
//! - [`hardswish`](crate::Vector::hardswish): Efficient hardware-friendly swish
//! - [`mish`](crate::Vector::mish): Self-regularizing activation
//! - [`selu`](crate::Vector::selu): Scaled Exponential Linear Unit

#[cfg(target_arch = "x86_64")]
use crate::backends::avx2::Avx2Backend;
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
use crate::backends::neon::NeonBackend;
use crate::backends::scalar::ScalarBackend;
#[cfg(target_arch = "x86_64")]
use crate::backends::sse2::Sse2Backend;
#[cfg(target_arch = "wasm32")]
use crate::backends::wasm::WasmBackend;
use crate::backends::VectorBackend;
use crate::vector::Vector;
use crate::{Backend, Result, TruenoError};

impl Vector<f32> {
    /// Softmax activation function
    ///
    /// Converts a vector of real values into a probability distribution.
    /// Formula: softmax(x)\[i\] = exp(x\[i\] - max(x)) / sum(exp(x\[j\] - max(x)))
    ///
    /// Uses the numerically stable version with max subtraction to prevent overflow.
    /// The output is a probability distribution: all values in [0, 1] and sum to 1.
    ///
    /// This is the standard activation function for multi-class classification in neural networks.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let logits = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let probs = logits.softmax()?;
    ///
    /// // Verify sum ≈ 1
    /// let sum: f32 = probs.as_slice().iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-5);
    ///
    /// // Verify all values in [0, 1]
    /// for &p in probs.as_slice() {
    ///     assert!(p >= 0.0 && p <= 1.0);
    /// }
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns EmptyVector error for empty vectors (cannot compute softmax).
    pub fn softmax(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // OpComplexity::Medium - GPU threshold: >10K elements (multi-pass overhead)
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 4-368x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.softmax(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Scalar fallback: Multi-pass softmax for numerical stability
        // Find max for numerical stability (prevents overflow in exp)
        let max_val = self.max()?;

        // Compute exp(x - max) for each element
        let exp_vals: Vec<f32> = self.data.iter().map(|&x| (x - max_val).exp()).collect();

        // Compute sum of exponentials
        let sum_exp: f32 = exp_vals.iter().sum();

        // Normalize by sum
        let data: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        Ok(Vector::from_vec(data))
    }

    /// Log-softmax activation function
    ///
    /// Computes the logarithm of the softmax function in a numerically stable way.
    /// Formula: log_softmax(x)\[i\] = x\[i\] - max(x) - log(sum(exp(x\[j\] - max(x))))
    ///
    /// This is more numerically stable than computing log(softmax(x)) and is commonly
    /// used in neural networks for computing cross-entropy loss.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let logits = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let log_probs = logits.log_softmax()?;
    ///
    /// // Verify exp(log_softmax) = softmax
    /// let probs_from_log: Vec<f32> = log_probs.as_slice().iter().map(|&x| x.exp()).collect();
    /// let sum: f32 = probs_from_log.iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-5);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns EmptyVector error for empty vectors.
    pub fn log_softmax(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // OpComplexity::Medium - GPU threshold: >10K elements (multi-pass overhead)
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 4-368x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.log_softmax(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Scalar fallback: Multi-pass log_softmax for numerical stability
        // Find max for numerical stability
        let max_val = self.max()?;

        // Compute exp(x - max) for each element
        let exp_vals: Vec<f32> = self.data.iter().map(|&x| (x - max_val).exp()).collect();

        // Compute log of sum of exponentials
        let sum_exp: f32 = exp_vals.iter().sum();
        let log_sum_exp = sum_exp.ln();

        // log_softmax(x)[i] = x[i] - max - log_sum_exp
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| x - max_val - log_sum_exp)
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// ReLU (Rectified Linear Unit) activation function
    ///
    /// Computes the element-wise ReLU: max(0, x).
    /// ReLU is one of the most widely used activation functions in neural networks.
    ///
    /// # Formula
    ///
    /// ```text
    /// relu(x)[i] = max(0, x\[i\])
    ///            = x\[i\]  if x\[i\] > 0
    ///            = 0     otherwise
    /// ```
    ///
    /// # Properties
    ///
    /// - **Non-linearity**: Introduces non-linearity while preserving linearity for positive values
    /// - **Sparsity**: Produces exactly zero for negative inputs (sparse activations)
    /// - **Gradient**: Derivative is 1 for positive inputs, 0 for negative (solves vanishing gradient)
    /// - **Computational efficiency**: Simple max operation, no exponentials
    ///
    /// # Applications
    ///
    /// - **Deep neural networks**: Default activation for hidden layers
    /// - **Convolutional networks**: Standard activation in CNNs
    /// - **Feature learning**: Encourages sparse representations
    ///
    /// # Performance
    ///
    /// This operation is memory-bound. SIMD provides modest speedups since
    /// the computation (comparison and selection) is simpler than memory access.
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
    /// let result = v.relu()?;
    /// assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn relu(&self) -> Result<Self> {
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
                    if gpu.relu(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        let mut result = vec![0.0; self.len()];

        // Use parallel processing for very large arrays (reduces TLB pressure and improves cache utilization)
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 500_000; // Increased to avoid overhead at smaller sizes
            const CHUNK_SIZE: usize = 65536; // 64K elements = 256KB, cache-friendly

            if self.len() >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;

                self.data
                    .par_chunks(CHUNK_SIZE)
                    .zip(result.par_chunks_mut(CHUNK_SIZE))
                    .for_each(|(chunk_in, chunk_out)| {
                        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
                        unsafe {
                            match self.backend {
                                Backend::Scalar => {
                                    ScalarBackend::relu(chunk_in, chunk_out);
                                }
                                #[cfg(target_arch = "x86_64")]
                                Backend::SSE2 | Backend::AVX => {
                                    Sse2Backend::relu(chunk_in, chunk_out);
                                }
                                #[cfg(target_arch = "x86_64")]
                                Backend::AVX2 | Backend::AVX512 => {
                                    Avx2Backend::relu(chunk_in, chunk_out);
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                                    ScalarBackend::relu(chunk_in, chunk_out);
                                }
                                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                                Backend::NEON => {
                                    NeonBackend::relu(chunk_in, chunk_out);
                                }
                                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                                Backend::NEON => {
                                    ScalarBackend::relu(chunk_in, chunk_out);
                                }
                                #[cfg(target_arch = "wasm32")]
                                Backend::WasmSIMD => {
                                    WasmBackend::relu(chunk_in, chunk_out);
                                }
                                #[cfg(not(target_arch = "wasm32"))]
                                Backend::WasmSIMD => {
                                    ScalarBackend::relu(chunk_in, chunk_out);
                                }
                                Backend::GPU | Backend::Auto => {
                                    ScalarBackend::relu(chunk_in, chunk_out);
                                }
                            }
                        }
                    });

                return Ok(Vector::from_vec(result)); // Use from_vec to avoid extra copy
            }
        }

        // Sequential processing for small arrays or when parallel feature disabled
        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::relu(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::relu(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    Avx2Backend::relu(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::relu(&self.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::relu(&self.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    ScalarBackend::relu(&self.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::relu(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    ScalarBackend::relu(&self.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::relu(&self.data, &mut result);
                }
            }
        }

        Ok(Vector::from_vec(result)) // Use from_vec to avoid extra copy
    }

    /// Sigmoid (logistic) activation function
    ///
    /// Computes the element-wise sigmoid: σ(x) = 1 / (1 + e^(-x)).
    /// Sigmoid is a classic activation function that squashes inputs to the range (0, 1).
    ///
    /// # Formula
    ///
    /// ```text
    /// sigmoid(x)[i] = 1 / (1 + exp(-x\[i\]))
    ///               = exp(x\[i\]) / (1 + exp(x\[i\]))
    /// ```
    ///
    /// # Properties
    ///
    /// - **Bounded output**: Maps all inputs to (0, 1) range
    /// - **Smooth**: Infinitely differentiable (C^∞)
    /// - **Symmetric**: σ(-x) = 1 - σ(x)
    /// - **Derivative**: σ'(x) = σ(x) * (1 - σ(x))
    /// - **Interpretable**: Output can be interpreted as probability
    ///
    /// # Applications
    ///
    /// - **Binary classification**: Final layer for binary output (0 or 1)
    /// - **Logistic regression**: Traditional ML algorithm
    /// - **Gating mechanisms**: LSTM/GRU gates (input, forget, output)
    /// - **Attention mechanisms**: Soft attention weights
    ///
    /// # Numerical Considerations
    ///
    /// For very large negative inputs (x < -50), exp(-x) overflows to infinity.
    /// However, sigmoid(x) approaches 0, so we return 0 for numerical stability.
    /// For very large positive inputs (x > 50), exp(-x) underflows to 0,
    /// and sigmoid(x) approaches 1.
    ///
    /// # Performance
    ///
    /// This operation is compute-bound due to the exp() operation. SIMD provides
    /// modest speedups, but the exponential is the bottleneck.
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
    /// let v = Vector::from_slice(&[-2.0, 0.0, 2.0]);
    /// let result = v.sigmoid()?;
    ///
    /// // sigmoid(-2) ≈ 0.119, sigmoid(0) = 0.5, sigmoid(2) ≈ 0.881
    /// assert!((result.as_slice()[0] - 0.119).abs() < 0.001);
    /// assert!((result.as_slice()[1] - 0.5).abs() < 0.001);
    /// assert!((result.as_slice()[2] - 0.881).abs() < 0.001);
    /// # Ok::<(), trueno::TruenoError>(())
    /// ```
    pub fn sigmoid(&self) -> Result<Self> {
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
                    if gpu.sigmoid(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        let mut result = vec![0.0; self.len()];

        // Dispatch to appropriate backend
        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::sigmoid(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::sigmoid(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    Avx2Backend::sigmoid(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::sigmoid(&self.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::sigmoid(&self.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    ScalarBackend::sigmoid(&self.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::sigmoid(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    ScalarBackend::sigmoid(&self.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::sigmoid(&self.data, &mut result);
                }
            }
        }

        Ok(Vector::from_vec(result))
    }

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
        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::gelu(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::gelu(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    Avx2Backend::gelu(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::gelu(&self.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::gelu(&self.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    ScalarBackend::gelu(&self.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::gelu(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    ScalarBackend::gelu(&self.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::gelu(&self.data, &mut result);
                }
            }
        }

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
        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::swish(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::swish(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    Avx2Backend::swish(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::swish(&self.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::swish(&self.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    ScalarBackend::swish(&self.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::swish(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    ScalarBackend::swish(&self.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
                    // Not yet implemented, use scalar
                    ScalarBackend::swish(&self.data, &mut result);
                }
            }
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Softmax ==========

    #[test]
    fn test_softmax_basic() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = v.softmax().unwrap();
        // Check sum = 1
        let sum: f32 = result.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Check all values in [0, 1]
        for &val in result.as_slice() {
            assert!(val >= 0.0 && val <= 1.0);
        }
        // Check highest input has highest probability
        assert!(result.as_slice()[2] > result.as_slice()[1]);
        assert!(result.as_slice()[1] > result.as_slice()[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        assert!(matches!(v.softmax(), Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_softmax_single() {
        let v = Vector::from_slice(&[5.0]);
        let result = v.softmax().unwrap();
        assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_uniform() {
        let v = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let result = v.softmax().unwrap();
        // All equal inputs should give equal outputs
        for &val in result.as_slice() {
            assert!((val - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_large_values() {
        // Test numerical stability with large values
        let v = Vector::from_slice(&[1000.0, 1001.0, 1002.0]);
        let result = v.softmax().unwrap();
        let sum: f32 = result.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ========== Log Softmax ==========

    #[test]
    fn test_log_softmax_basic() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = v.log_softmax().unwrap();
        // All log probabilities should be <= 0
        for &val in result.as_slice() {
            assert!(val <= 0.0);
        }
    }

    #[test]
    fn test_log_softmax_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        assert!(matches!(v.log_softmax(), Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_log_softmax_single() {
        let v = Vector::from_slice(&[5.0]);
        let result = v.log_softmax().unwrap();
        // log(1) = 0
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    }

    // ========== ReLU ==========

    #[test]
    fn test_relu_basic() {
        let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = v.relu().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[3] - 1.0).abs() < 1e-6);
        assert!((result.as_slice()[4] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        assert!(matches!(v.relu(), Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_relu_all_negative() {
        let v = Vector::from_slice(&[-5.0, -3.0, -1.0]);
        let result = v.relu().unwrap();
        for &val in result.as_slice() {
            assert!((val - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_relu_all_positive() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = v.relu().unwrap();
        assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 2.0).abs() < 1e-6);
        assert!((result.as_slice()[2] - 3.0).abs() < 1e-6);
    }

    // ========== Sigmoid ==========

    #[test]
    fn test_sigmoid_basic() {
        let v = Vector::from_slice(&[-10.0, 0.0, 10.0]);
        let result = v.sigmoid().unwrap();
        // sigmoid(-10) ≈ 0
        assert!(result.as_slice()[0] < 0.001);
        // sigmoid(0) = 0.5
        assert!((result.as_slice()[1] - 0.5).abs() < 1e-6);
        // sigmoid(10) ≈ 1
        assert!(result.as_slice()[2] > 0.999);
    }

    #[test]
    fn test_sigmoid_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        assert!(matches!(v.sigmoid(), Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_sigmoid_range() {
        let v = Vector::from_slice(&[-100.0, -1.0, 0.0, 1.0, 100.0]);
        let result = v.sigmoid().unwrap();
        for &val in result.as_slice() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    // ========== Leaky ReLU ==========

    #[test]
    fn test_leaky_relu_basic() {
        let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = v.leaky_relu(0.01).unwrap();
        assert!((result.as_slice()[0] - (-0.02)).abs() < 1e-6);
        assert!((result.as_slice()[1] - (-0.01)).abs() < 1e-6);
        assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[3] - 1.0).abs() < 1e-6);
        assert!((result.as_slice()[4] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_leaky_relu_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        assert!(matches!(v.leaky_relu(0.01), Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_leaky_relu_different_slopes() {
        let v = Vector::from_slice(&[-1.0]);
        // slope 0.1
        let result = v.leaky_relu(0.1).unwrap();
        assert!((result.as_slice()[0] - (-0.1)).abs() < 1e-6);
        // slope 0.2
        let result = v.leaky_relu(0.2).unwrap();
        assert!((result.as_slice()[0] - (-0.2)).abs() < 1e-6);
    }

    // ========== ELU ==========

    #[test]
    fn test_elu_basic() {
        let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = v.elu(1.0).unwrap();
        // Positive values unchanged
        assert!((result.as_slice()[3] - 1.0).abs() < 1e-6);
        assert!((result.as_slice()[4] - 2.0).abs() < 1e-6);
        // Zero stays zero
        assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
        // Negative values: alpha * (exp(x) - 1)
        assert!(result.as_slice()[0] < 0.0);
        assert!(result.as_slice()[1] < 0.0);
    }

    #[test]
    fn test_elu_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        assert!(matches!(v.elu(1.0), Err(TruenoError::EmptyVector)));
    }

    // ========== GELU ==========

    #[test]
    fn test_gelu_basic() {
        let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = v.gelu().unwrap();
        // GELU(0) = 0
        assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
        // GELU is approximately linear for positive values
        assert!(result.as_slice()[3] > 0.5);
        assert!(result.as_slice()[4] > 1.5);
        // Negative values are small but not zero
        assert!(result.as_slice()[0].abs() < 0.1);
    }

    #[test]
    fn test_gelu_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        assert!(matches!(v.gelu(), Err(TruenoError::EmptyVector)));
    }

    // ========== Swish ==========

    #[test]
    fn test_swish_basic() {
        let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = v.swish().unwrap();
        // Swish(0) = 0
        assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
        // Swish(x) = x * sigmoid(x)
        // Swish(1) ≈ 0.731
        assert!((result.as_slice()[3] - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_swish_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        assert!(matches!(v.swish(), Err(TruenoError::EmptyVector)));
    }

    // ========== Hardswish ==========

    #[test]
    fn test_hardswish_basic() {
        let v = Vector::from_slice(&[-4.0, -3.0, 0.0, 3.0, 4.0]);
        let result = v.hardswish().unwrap();
        // x <= -3: hardswish(x) = 0
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 0.0).abs() < 1e-6);
        // x >= 3: hardswish(x) = x
        assert!((result.as_slice()[3] - 3.0).abs() < 1e-6);
        assert!((result.as_slice()[4] - 4.0).abs() < 1e-6);
        // x = 0: hardswish(0) = 0
        assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_hardswish_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        assert!(matches!(v.hardswish(), Err(TruenoError::EmptyVector)));
    }

    // ========== Mish ==========

    #[test]
    fn test_mish_basic() {
        let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = v.mish().unwrap();
        // Mish(0) = 0
        assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
        // Mish is smooth and non-monotonic for negative values
        assert!(result.as_slice()[0] < 0.0);
    }

    #[test]
    fn test_mish_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        assert!(matches!(v.mish(), Err(TruenoError::EmptyVector)));
    }

    // ========== SELU ==========

    #[test]
    fn test_selu_basic() {
        let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = v.selu().unwrap();
        // SELU(0) = 0
        assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
        // Positive values scaled by λ ≈ 1.0507
        assert!((result.as_slice()[3] - 1.0507).abs() < 0.001);
        assert!((result.as_slice()[4] - 2.1014).abs() < 0.001);
        // Negative values use ELU-like formula
        assert!(result.as_slice()[0] < 0.0);
        assert!(result.as_slice()[1] < 0.0);
    }

    #[test]
    fn test_selu_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        assert!(matches!(v.selu(), Err(TruenoError::EmptyVector)));
    }

    // ========== Backend Tests ==========

    #[test]
    fn test_relu_scalar_backend() {
        let v = Vector::from_slice_with_backend(&[-1.0, 0.0, 1.0], Backend::Scalar);
        let result = v.relu().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_relu_sse2_backend() {
        let v = Vector::from_slice_with_backend(&[-1.0, 0.0, 1.0, 2.0], Backend::SSE2);
        let result = v.relu().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_scalar_backend() {
        let v = Vector::from_slice_with_backend(&[0.0], Backend::Scalar);
        let result = v.sigmoid().unwrap();
        assert!((result.as_slice()[0] - 0.5).abs() < 1e-6);
    }

    // ========== Large Array Tests ==========

    #[test]
    fn test_relu_large() {
        let v = Vector::from_slice(&[-1.0; 1000]);
        let result = v.relu().unwrap();
        for &val in result.as_slice() {
            assert!((val - 0.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sigmoid_large() {
        let v = Vector::from_slice(&[0.0; 1000]);
        let result = v.sigmoid().unwrap();
        for &val in result.as_slice() {
            assert!((val - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_large() {
        let v = Vector::from_slice(&[1.0; 100]);
        let result = v.softmax().unwrap();
        let sum: f32 = result.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-4);
        // All equal inputs should give equal probabilities
        for &val in result.as_slice() {
            assert!((val - 0.01).abs() < 1e-4);
        }
    }
}
