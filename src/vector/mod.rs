//! Vector type with multi-backend support
//!
//! This module provides the core `Vector<T>` type with SIMD-optimized operations
//! across multiple backends (Scalar, SSE2, AVX2, AVX-512, NEON, WASM SIMD).
//!
//! GPU thresholds intentionally set to usize::MAX to disable GPU for element-wise ops.
//! See docs/performance-analysis.md - GPU is 2-65,000x SLOWER than scalar for these ops.

#![allow(clippy::absurd_extreme_comparisons)]

// Submodules
pub mod dispatch;
mod ops;

// Tests (~10K lines extracted for TDG compliance)
#[cfg(test)]
mod tests;

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
use crate::{Backend, Result, TruenoError};

// Use the dispatch macros from the dispatch submodule (exported at crate root)
use crate::dispatch_unary_op;

/// High-performance vector with multi-backend support
///
/// # Examples
///
/// ```
/// use trueno::Vector;
///
/// let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
/// let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
/// let result = a.add(&b).unwrap();
///
/// assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T> {
    data: Vec<T>,
    backend: Backend,
}

impl<T> Vector<T>
where
    T: Clone,
{
    /// Create vector from slice using auto-selected optimal backend
    ///
    /// # Performance
    ///
    /// Auto-selects the best available backend at creation time based on:
    /// - CPU feature detection (AVX-512 > AVX2 > AVX > SSE2)
    /// - Vector size (GPU for large workloads)
    /// - Platform availability (NEON on ARM, WASM SIMD in browser)
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(v.len(), 4);
    /// ```
    pub fn from_slice(data: &[T]) -> Self {
        Self {
            data: data.to_vec(),
            backend: crate::select_best_available_backend(),
        }
    }

    /// Create vector from an existing Vec (takes ownership, no copy)
    ///
    /// This is more efficient than `from_slice` when you already have a Vec
    /// and don't need to keep it, as it avoids an extra allocation and copy.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let data = vec![1.0, 2.0, 3.0];
    /// let v = Vector::from_vec(data);
    /// assert_eq!(v.len(), 3);
    /// ```
    pub fn from_vec(data: Vec<T>) -> Self {
        Self {
            data,
            backend: crate::select_best_available_backend(),
        }
    }

    /// Create vector with specific backend (for benchmarking or testing)
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::{Vector, Backend};
    ///
    /// let v = Vector::from_slice_with_backend(&[1.0, 2.0], Backend::Scalar);
    /// assert_eq!(v.len(), 2);
    /// ```
    pub fn from_slice_with_backend(data: &[T], backend: Backend) -> Self {
        let resolved_backend = match backend {
            Backend::Auto => crate::select_best_available_backend(),
            _ => backend,
        };

        Self {
            data: data.to_vec(),
            backend: resolved_backend,
        }
    }
}

impl Vector<f32> {
    /// Create vector with specified alignment for optimal SIMD performance
    ///
    /// This method attempts to create a vector with memory aligned to the specified byte boundary.
    /// Note: Rust's Vec allocator may already provide sufficient alignment for most use cases.
    /// This method validates the alignment requirement but uses standard Vec allocation.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of elements to allocate
    /// * `backend` - Backend to use for operations
    /// * `alignment` - Requested alignment in bytes (must be power of 2: 16, 32, 64)
    ///
    /// # Recommended Alignments
    ///
    /// - SSE2: 16 bytes (128-bit)
    /// - AVX2: 32 bytes (256-bit)
    /// - AVX-512: 64 bytes (512-bit)
    ///
    /// # Note on Implementation
    ///
    /// Currently uses Rust's default Vec allocator, which typically provides 16-byte alignment
    /// on modern systems. Custom allocators for specific alignments will be added in future versions.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::{Vector, Backend};
    ///
    /// // Create vector with requested 16-byte alignment
    /// let v = Vector::with_alignment(100, Backend::SSE2, 16).unwrap();
    /// assert_eq!(v.len(), 100);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `TruenoError::InvalidInput` if alignment is not a power of 2.
    pub fn with_alignment(size: usize, backend: Backend, alignment: usize) -> Result<Self> {
        // Validate alignment is power of 2
        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return Err(TruenoError::InvalidInput(format!(
                "Alignment must be power of 2, got {}",
                alignment
            )));
        }

        // Resolve backend
        let resolved_backend = match backend {
            Backend::Auto => crate::select_best_available_backend(),
            _ => backend,
        };

        // For now, use standard Vec allocation which typically provides good alignment
        // Future enhancement: use custom allocator for guaranteed alignment > 16 bytes
        let data = vec![0.0f32; size];

        // Verify actual alignment (for informational purposes)
        let ptr = data.as_ptr() as usize;
        let actual_alignment = ptr & !(ptr - 1); // Find lowest set bit

        // Log warning if alignment requirement not met (for future enhancement)
        if alignment > actual_alignment {
            // Note: This is not an error, just informational
            // The unaligned loads in SSE2 (_mm_loadu_ps) will still work correctly
            eprintln!(
                "Note: Requested {}-byte alignment, got {}-byte alignment. Using unaligned loads.",
                alignment, actual_alignment
            );
        }

        Ok(Self {
            data,
            backend: resolved_backend,
        })
    }
}

impl<T> Vector<T>
where
    T: Clone,
{
    /// Get underlying data as slice
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    /// ```
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Get vector length
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// assert_eq!(v.len(), 5);
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if vector is empty
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v1: Vector<f32> = Vector::from_slice(&[]);
    /// assert!(v1.is_empty());
    ///
    /// let v2 = Vector::from_slice(&[1.0]);
    /// assert!(!v2.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the backend being used
    pub fn backend(&self) -> Backend {
        self.backend
    }
}

impl Vector<f32> {
    // Note: add, sub, mul, div moved to ops/arithmetic.rs
    // Note: dot, sum, max, min, argmax, argmin moved to ops/reductions.rs
    // Note: mean, variance, stddev, covariance, correlation moved to ops/reductions.rs

    // Reductions (dot, sum, max, min, etc.) moved to ops/reductions.rs
    // Arithmetic (add, sub, mul, div, scale, fma) moved to ops/arithmetic.rs

    /// Z-score normalization (standardization)
    ///
    /// Transforms the vector to have mean = 0 and standard deviation = 1.
    /// Each element is transformed as: z\[i\] = (x\[i\] - μ) / σ
    ///
    /// This is a fundamental preprocessing step in machine learning and statistics,
    /// ensuring features have comparable scales and are centered around zero.
    ///
    /// # Performance
    ///
    /// Uses optimized SIMD implementations via mean() and stddev(), then applies
    /// element-wise operations (sub, scale) which also use SIMD.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let z = v.zscore().unwrap();
    ///
    /// // Verify mean ≈ 0
    /// let mean = z.mean().unwrap();
    /// assert!(mean.abs() < 1e-5);
    ///
    /// // Verify stddev ≈ 1
    /// let std = z.stddev().unwrap();
    /// assert!((std - 1.0).abs() < 1e-5);
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns EmptyVector error for empty vectors (cannot compute mean/stddev).
    ///
    /// # Division by zero
    ///
    /// Returns DivisionByZero error if the vector has zero standard deviation
    /// (i.e., all elements are identical/constant).
    ///
    /// ```
    /// use trueno::{Vector, TruenoError};
    ///
    /// let v = Vector::from_slice(&[5.0, 5.0, 5.0]); // Constant
    /// assert!(matches!(v.zscore(), Err(TruenoError::DivisionByZero)));
    /// ```
    pub fn zscore(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        let mean_val = self.mean()?;
        let std_val = self.stddev()?;

        // Check for zero standard deviation (constant vector)
        if std_val.abs() < 1e-10 {
            return Err(TruenoError::DivisionByZero);
        }

        // Transform: z[i] = (x[i] - μ) / σ
        let inv_std = 1.0 / std_val;
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| (x - mean_val) * inv_std)
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// Min-max normalization (scaling to [0, 1] range)
    ///
    /// Transforms the vector so that the minimum value becomes 0 and the maximum
    /// value becomes 1, with all other values scaled proportionally.
    /// Formula: x'\[i\] = (x\[i\] - min) / (max - min)
    ///
    /// This is a fundamental preprocessing technique in machine learning, especially
    /// for algorithms sensitive to feature magnitudes (e.g., neural networks, k-NN).
    ///
    /// # Performance
    ///
    /// Uses optimized SIMD implementations via min() and max() operations, then
    /// applies element-wise transformation.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let normalized = v.minmax_normalize().unwrap();
    ///
    /// // Verify range [0, 1]
    /// let min = normalized.min().unwrap();
    /// let max = normalized.max().unwrap();
    /// assert!((min - 0.0).abs() < 1e-5);
    /// assert!((max - 1.0).abs() < 1e-5);
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns EmptyVector error for empty vectors (cannot compute min/max).
    ///
    /// # Division by zero
    ///
    /// Returns DivisionByZero error if the vector has all identical elements
    /// (i.e., min = max, causing division by zero in the normalization formula).
    ///
    /// ```
    /// use trueno::{Vector, TruenoError};
    ///
    /// let v = Vector::from_slice(&[5.0, 5.0, 5.0]); // Constant
    /// assert!(matches!(v.minmax_normalize(), Err(TruenoError::DivisionByZero)));
    /// ```
    pub fn minmax_normalize(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        let min_val = self.min()?;
        let max_val = self.max()?;
        let range = max_val - min_val;

        // Check for zero range (constant vector)
        if range.abs() < 1e-10 {
            return Err(TruenoError::DivisionByZero);
        }

        // Transform: x'[i] = (x[i] - min) / (max - min)
        let inv_range = 1.0 / range;
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| (x - min_val) * inv_range)
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// Layer normalization with learnable parameters (Issue #61: ML primitives)
    ///
    /// Applies layer normalization: `y = gamma * (x - mean) / sqrt(variance + eps) + beta`
    ///
    /// This is a fundamental normalization technique in transformers and other
    /// modern neural network architectures. Unlike batch normalization, layer norm
    /// normalizes across the feature dimension, making it suitable for sequence models.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Scale parameter (typically learned, initialized to 1.0)
    /// * `beta` - Shift parameter (typically learned, initialized to 0.0)
    /// * `eps` - Small constant for numerical stability (typically 1e-5 or 1e-6)
    ///
    /// # Returns
    ///
    /// Normalized vector with the same shape as input
    ///
    /// # Errors
    ///
    /// Returns `SizeMismatch` if gamma or beta have different lengths than self
    /// Returns `EmptyVector` if input is empty
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// let gamma = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]); // Scale = 1
    /// let beta = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);  // Shift = 0
    ///
    /// let y = x.layer_norm(&gamma, &beta, 1e-5).unwrap();
    ///
    /// // Output should be approximately standardized (mean ≈ 0, std ≈ 1)
    /// let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
    /// assert!(mean.abs() < 1e-5);
    /// ```
    ///
    /// # Performance
    ///
    /// Single-pass computation using Welford's algorithm for numerical stability.
    /// Time complexity: O(n), Space complexity: O(n).
    pub fn layer_norm(&self, gamma: &Self, beta: &Self, eps: f32) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        if self.len() != gamma.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: gamma.len(),
            });
        }

        if self.len() != beta.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: beta.len(),
            });
        }

        // Compute mean
        let mean_val = self.mean()?;

        // Compute variance: E[(x - mean)^2]
        let variance: f32 = self
            .data
            .iter()
            .map(|&x| {
                let diff = x - mean_val;
                diff * diff
            })
            .sum::<f32>()
            / self.len() as f32;

        // Compute inverse standard deviation for numerical stability
        let inv_std = 1.0 / (variance + eps).sqrt();

        // Apply normalization: y = gamma * (x - mean) * inv_std + beta
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(gamma.data.iter())
            .zip(beta.data.iter())
            .map(|((&x, &g), &b)| g * (x - mean_val) * inv_std + b)
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// Layer normalization without learnable parameters
    ///
    /// Simplified version that just standardizes the input: `y = (x - mean) / sqrt(variance + eps)`
    ///
    /// This is equivalent to calling `layer_norm` with gamma=1 and beta=0.
    ///
    /// # Arguments
    ///
    /// * `eps` - Small constant for numerical stability (typically 1e-5)
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// let y = x.layer_norm_simple(1e-5).unwrap();
    ///
    /// // Output should be standardized
    /// let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
    /// assert!(mean.abs() < 1e-5);
    /// ```
    pub fn layer_norm_simple(&self, eps: f32) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        let mean_val = self.mean()?;

        // Compute variance
        let variance: f32 = self
            .data
            .iter()
            .map(|&x| {
                let diff = x - mean_val;
                diff * diff
            })
            .sum::<f32>()
            / self.len() as f32;

        let inv_std = 1.0 / (variance + eps).sqrt();

        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| (x - mean_val) * inv_std)
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// Clip values to a specified range [min_val, max_val]
    ///
    /// Constrains each element to be within the specified range:
    /// - Values below min_val become min_val
    /// - Values above max_val become max_val
    /// - Values within range stay unchanged
    ///
    /// This is useful for outlier handling, gradient clipping in neural networks,
    /// and ensuring values stay within valid bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
    /// let clipped = v.clip(0.0, 10.0).unwrap();
    ///
    /// // Values: [-5, 0, 5, 10, 15] → [0, 0, 5, 10, 10]
    /// assert_eq!(clipped.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
    /// ```
    ///
    /// # Invalid range
    ///
    /// Returns InvalidInput error if min_val > max_val.
    ///
    /// ```
    /// use trueno::{Vector, TruenoError};
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let result = v.clip(10.0, 5.0); // min > max
    /// assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
    /// ```
    pub fn clip(&self, min_val: f32, max_val: f32) -> Result<Self> {
        if min_val > max_val {
            return Err(TruenoError::InvalidInput(format!(
                "min_val ({}) must be <= max_val ({})",
                min_val, max_val
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
                    if gpu.clip(&self.data, &mut result, min_val, max_val).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Scalar fallback: Element-wise clamp
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| x.max(min_val).min(max_val))
            .collect();

        Ok(Vector::from_vec(data))
    }


    /// L2 norm (Euclidean norm)
    ///
    /// Computes the Euclidean length of the vector: sqrt(sum(a\[i\]^2)).
    /// This is mathematically equivalent to sqrt(dot(self, self)).
    ///
    /// # Performance
    ///
    /// Uses optimized SIMD implementations via the dot product operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.0, 4.0]);
    /// let norm = v.norm_l2().unwrap();
    /// assert!((norm - 5.0).abs() < 1e-5); // sqrt(3^2 + 4^2) = 5
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns 0.0 for empty vectors (consistent with the mathematical definition).
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v: Vector<f32> = Vector::from_slice(&[]);
    /// assert_eq!(v.norm_l2().unwrap(), 0.0);
    /// ```
    pub fn norm_l2(&self) -> Result<f32> {
        if self.data.is_empty() {
            return Ok(0.0);
        }

        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let result = unsafe {
            match self.backend {
                Backend::Scalar => ScalarBackend::norm_l2(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::norm_l2(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::norm_l2(&self.data),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::norm_l2(&self.data)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => NeonBackend::norm_l2(&self.data),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::norm_l2(&self.data),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => WasmBackend::norm_l2(&self.data),
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::norm_l2(&self.data),
                Backend::GPU | Backend::Auto => ScalarBackend::norm_l2(&self.data),
            }
        };

        Ok(result)
    }

    /// Normalize the vector to unit length (L2 norm = 1)
    ///
    /// Returns a new vector in the same direction but with magnitude 1.
    ///
    /// # Errors
    ///
    /// Returns `TruenoError::DivisionByZero` if the vector has zero norm (cannot normalize zero vector).
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.0, 4.0]);
    /// let unit = v.normalize().unwrap();
    ///
    /// // Result is [0.6, 0.8] (a unit vector)
    /// assert!((unit.as_slice()[0] - 0.6).abs() < 1e-5);
    /// assert!((unit.as_slice()[1] - 0.8).abs() < 1e-5);
    ///
    /// // Verify it's a unit vector (norm = 1)
    /// assert!((unit.norm_l2().unwrap() - 1.0).abs() < 1e-5);
    /// ```
    ///
    /// # Zero Vector Error
    ///
    /// ```
    /// use trueno::{Vector, TruenoError};
    ///
    /// let v = Vector::from_slice(&[0.0, 0.0]);
    /// assert!(matches!(v.normalize(), Err(TruenoError::DivisionByZero)));
    /// ```
    pub fn normalize(&self) -> Result<Vector<f32>> {
        let norm = self.norm_l2()?;

        // Check for zero or near-zero norm (cannot normalize zero vector)
        if norm.abs() < 1e-10 {
            return Err(TruenoError::DivisionByZero);
        }

        // Divide each element by the norm using scalar multiplication
        // This avoids creating an intermediate vector
        self.scale(1.0 / norm)
    }

    /// Compute the L1 norm (Manhattan norm) of the vector
    ///
    /// Returns the sum of absolute values: ||v||₁ = sum(|v\[i\]|)
    ///
    /// The L1 norm is used in:
    /// - Machine learning (L1 regularization, Lasso regression)
    /// - Distance metrics (Manhattan distance)
    /// - Sparse modeling and feature selection
    /// - Signal processing
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.0, -4.0, 5.0]);
    /// let norm = v.norm_l1().unwrap();
    ///
    /// // |3| + |-4| + |5| = 12
    /// assert!((norm - 12.0).abs() < 1e-5);
    /// ```
    ///
    /// # Empty Vector
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v: Vector<f32> = Vector::from_slice(&[]);
    /// assert_eq!(v.norm_l1().unwrap(), 0.0);
    /// ```
    pub fn norm_l1(&self) -> Result<f32> {
        if self.data.is_empty() {
            return Ok(0.0);
        }

        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let result = unsafe {
            match self.backend {
                Backend::Scalar => ScalarBackend::norm_l1(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::norm_l1(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::norm_l1(&self.data),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::norm_l1(&self.data)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => NeonBackend::norm_l1(&self.data),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::norm_l1(&self.data),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => WasmBackend::norm_l1(&self.data),
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::norm_l1(&self.data),
                Backend::GPU | Backend::Auto => ScalarBackend::norm_l1(&self.data),
            }
        };

        Ok(result)
    }

    /// Compute the L∞ norm (infinity norm / max norm) of the vector
    ///
    /// Returns the maximum absolute value: ||v||∞ = max(|v\[i\]|)
    ///
    /// The L∞ norm is used in:
    /// - Numerical analysis (error bounds, stability analysis)
    /// - Optimization (Chebyshev approximation)
    /// - Signal processing (peak detection)
    /// - Distance metrics (Chebyshev distance)
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.0, -7.0, 5.0, -2.0]);
    /// let norm = v.norm_linf().unwrap();
    ///
    /// // max(|3|, |-7|, |5|, |-2|) = 7
    /// assert!((norm - 7.0).abs() < 1e-5);
    /// ```
    ///
    /// # Empty Vector
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v: Vector<f32> = Vector::from_slice(&[]);
    /// assert_eq!(v.norm_linf().unwrap(), 0.0);
    /// ```
    pub fn norm_linf(&self) -> Result<f32> {
        if self.data.is_empty() {
            return Ok(0.0);
        }

        // Use optimized SIMD backend for single-pass abs+max
        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let max_abs = unsafe {
            match self.backend {
                Backend::Scalar => ScalarBackend::norm_linf(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::norm_linf(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::norm_linf(&self.data),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::norm_linf(&self.data)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => ScalarBackend::norm_linf(&self.data), // NEON fallback
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::norm_linf(&self.data),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => ScalarBackend::norm_linf(&self.data), // WASM fallback
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::norm_linf(&self.data),
                Backend::GPU | Backend::Auto => ScalarBackend::norm_linf(&self.data),
            }
        };

        Ok(max_abs)
    }

    /// Compute element-wise absolute value
    ///
    /// Returns a new vector where each element is the absolute value of the corresponding input element.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.0, -4.0, 5.0, -2.0]);
    /// let result = v.abs().unwrap();
    ///
    /// assert_eq!(result.as_slice(), &[3.0, 4.0, 5.0, 2.0]);
    /// ```
    ///
    /// # Empty Vector
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v: Vector<f32> = Vector::from_slice(&[]);
    /// let result = v.abs().unwrap();
    /// assert_eq!(result.len(), 0);
    /// ```
    pub fn abs(&self) -> Result<Vector<f32>> {
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend {
                    Backend::Scalar => ScalarBackend::abs(&self.data, &mut result_data),
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => Sse2Backend::abs(&self.data, &mut result_data),
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::abs(&self.data, &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => NeonBackend::abs(&self.data, &mut result_data),
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => WasmBackend::abs(&self.data, &mut result_data),
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        // Auto should have been resolved at creation time
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[allow(unreachable_patterns)]
                    _ => ScalarBackend::abs(&self.data, &mut result_data),
                }
            }
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }


    /// Clamp elements to range [min_val, max_val]
    ///
    /// Returns a new vector where each element is constrained to the specified range.
    /// Elements below min_val become min_val, elements above max_val become max_val.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
    /// let result = v.clamp(0.0, 10.0).unwrap();
    ///
    /// assert_eq!(result.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
    /// ```
    ///
    /// # Negative Range
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-10.0, -5.0, 0.0, 5.0]);
    /// let result = v.clamp(-8.0, -2.0).unwrap();
    /// assert_eq!(result.as_slice(), &[-8.0, -5.0, -2.0, -2.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `InvalidInput` if min_val > max_val.
    pub fn clamp(&self, min_val: f32, max_val: f32) -> Result<Vector<f32>> {
        // Validate range
        if min_val > max_val {
            return Err(TruenoError::InvalidInput(format!(
                "Invalid clamp range: min ({}) > max ({})",
                min_val, max_val
            )));
        }

        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend {
                    Backend::Scalar => {
                        ScalarBackend::clamp(&self.data, min_val, max_val, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => {
                        Sse2Backend::clamp(&self.data, min_val, max_val, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::clamp(&self.data, min_val, max_val, &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => {
                        NeonBackend::clamp(&self.data, min_val, max_val, &mut result_data)
                    }
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => {
                        WasmBackend::clamp(&self.data, min_val, max_val, &mut result_data)
                    }
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        // Auto should have been resolved at creation time
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[allow(unreachable_patterns)]
                    _ => ScalarBackend::clamp(&self.data, min_val, max_val, &mut result_data),
                }
            }
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Linear interpolation between two vectors
    ///
    /// Computes element-wise linear interpolation: `result\[i\] = a\[i\] + t * (b\[i\] - a\[i\])`
    ///
    /// - When `t = 0.0`, returns `self`
    /// - When `t = 1.0`, returns `other`
    /// - Values outside `[0, 1]` perform extrapolation
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[0.0, 10.0, 20.0]);
    /// let b = Vector::from_slice(&[100.0, 110.0, 120.0]);
    /// let result = a.lerp(&b, 0.5).unwrap();
    ///
    /// assert_eq!(result.as_slice(), &[50.0, 60.0, 70.0]);
    /// ```
    ///
    /// # Extrapolation
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[0.0, 10.0]);
    /// let b = Vector::from_slice(&[10.0, 20.0]);
    ///
    /// // t > 1.0 extrapolates beyond b
    /// let result = a.lerp(&b, 2.0).unwrap();
    /// assert_eq!(result.as_slice(), &[20.0, 30.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `SizeMismatch` if vectors have different lengths.
    pub fn lerp(&self, other: &Vector<f32>, t: f32) -> Result<Vector<f32>> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend {
                    Backend::Scalar => {
                        ScalarBackend::lerp(&self.data, &other.data, t, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => {
                        Sse2Backend::lerp(&self.data, &other.data, t, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::lerp(&self.data, &other.data, t, &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => {
                        NeonBackend::lerp(&self.data, &other.data, t, &mut result_data)
                    }
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => {
                        WasmBackend::lerp(&self.data, &other.data, t, &mut result_data)
                    }
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[allow(unreachable_patterns)]
                    _ => ScalarBackend::lerp(&self.data, &other.data, t, &mut result_data),
                }
            }
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }


    /// Element-wise square root: result\[i\] = sqrt(self\[i\])
    ///
    /// Computes the square root of each element. For negative values, returns NaN
    /// following IEEE 754 floating-point semantics.
    ///
    /// # Returns
    ///
    /// A new vector where each element is the square root of the corresponding input element
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[4.0, 9.0, 16.0, 25.0]);
    /// let result = a.sqrt().unwrap();
    /// assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
    /// ```
    ///
    /// Negative values produce NaN:
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[-1.0, 4.0]);
    /// let result = a.sqrt().unwrap();
    /// assert!(result.as_slice()[0].is_nan());
    /// assert_eq!(result.as_slice()[1], 2.0);
    /// ```
    ///
    /// # Use Cases
    ///
    /// - Distance calculations: Euclidean distance computation
    /// - Statistics: Standard deviation, RMS (root mean square)
    /// - Machine learning: Normalization, gradient descent with adaptive learning rates
    /// - Signal processing: Amplitude calculations, power spectrum analysis
    /// - Physics simulations: Velocity from kinetic energy, wave propagation
    pub fn sqrt(&self) -> Result<Vector<f32>> {
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            // Use parallel processing for large arrays
            #[cfg(feature = "parallel")]
            {
                const PARALLEL_THRESHOLD: usize = 100_000;
                const CHUNK_SIZE: usize = 65536;

                if self.len() >= PARALLEL_THRESHOLD {
                    use rayon::prelude::*;

                    self.data
                        .par_chunks(CHUNK_SIZE)
                        .zip(result_data.par_chunks_mut(CHUNK_SIZE))
                        .for_each(|(chunk_in, chunk_out)| {
                            dispatch_unary_op!(self.backend, sqrt, chunk_in, chunk_out);
                        });

                    return Ok(Vector {
                        data: result_data,
                        backend: self.backend,
                    });
                }
            }

            dispatch_unary_op!(self.backend, sqrt, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Element-wise reciprocal: result\[i\] = 1 / self\[i\]
    ///
    /// Computes the reciprocal (multiplicative inverse) of each element.
    /// For zero values, returns infinity following IEEE 754 floating-point semantics.
    ///
    /// # Returns
    ///
    /// A new vector where each element is the reciprocal of the corresponding input element
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[2.0, 4.0, 5.0, 10.0]);
    /// let result = a.recip().unwrap();
    /// assert_eq!(result.as_slice(), &[0.5, 0.25, 0.2, 0.1]);
    /// ```
    ///
    /// Zero values produce infinity:
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[0.0, 2.0]);
    /// let result = a.recip().unwrap();
    /// assert!(result.as_slice()[0].is_infinite());
    /// assert_eq!(result.as_slice()[1], 0.5);
    /// ```
    ///
    /// # Use Cases
    ///
    /// - Division optimization: `a / b` → `a * recip(b)` (multiplication is faster)
    /// - Neural networks: Learning rate schedules, weight normalization
    /// - Statistics: Harmonic mean calculations, inverse transformations
    /// - Physics: Resistance (R = 1/G), optical power (P = 1/f)
    /// - Signal processing: Frequency to period conversion, filter design
    pub fn recip(&self) -> Result<Vector<f32>> {
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, recip, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Element-wise power: result\[i\] = base\[i\]^n
    ///
    /// Raises each element to the given power `n`.
    /// Uses Rust's optimized f32::powf() method.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[2.0, 3.0, 4.0]);
    /// let squared = v.pow(2.0).unwrap();
    /// assert_eq!(squared.as_slice(), &[4.0, 9.0, 16.0]);
    ///
    /// let sqrt = v.pow(0.5).unwrap();  // Fractional power = root
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `x.pow(0.0)` returns 1.0 for all x (even x=0)
    /// - `x.pow(1.0)` returns x (identity)
    /// - `x.pow(-1.0)` returns 1/x (reciprocal)
    /// - `x.pow(0.5)` returns sqrt(x) (square root)
    ///
    /// # Applications
    ///
    /// - Statistics: Power transformations (Box-Cox, Yeo-Johnson)
    /// - Machine learning: Polynomial features, activation functions
    /// - Physics: Inverse square law (1/r²), power laws
    /// - Signal processing: Power spectral density, root mean square
    pub fn pow(&self, n: f32) -> Result<Vector<f32>> {
        let pow_data: Vec<f32> = self.data.iter().map(|x| x.powf(n)).collect();
        Ok(Vector {
            data: pow_data,
            backend: self.backend,
        })
    }


}

