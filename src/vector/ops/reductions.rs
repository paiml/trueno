//! Reduction operations for Vector<f32>
//!
//! This module provides reduction operations that aggregate vector elements:
//! - Basic: `sum`, `dot`, `max`, `min`
//! - Index-finding: `argmax`, `argmin`
//! - Statistical: `mean`, `variance`, `stddev`, `covariance`, `correlation`
//! - Numerically stable: `sum_kahan`, `sum_of_squares`

#[cfg(target_arch = "x86_64")]
use crate::backends::avx2::Avx2Backend;
#[cfg(target_arch = "x86_64")]
use crate::backends::avx512::Avx512Backend;
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
use crate::backends::neon::NeonBackend;
use crate::backends::scalar::ScalarBackend;
#[cfg(target_arch = "x86_64")]
use crate::backends::sse2::Sse2Backend;
#[cfg(target_arch = "wasm32")]
use crate::backends::wasm::WasmBackend;
use crate::backends::VectorBackend;
use crate::vector::Vector;
use crate::{dispatch_reduction, Backend, Result, TruenoError};

impl Vector<f32> {
    /// Dot product
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
    /// let result = a.dot(&b).unwrap();
    ///
    /// assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    /// ```
    pub fn dot(&self, other: &Self) -> Result<f32> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let result = unsafe {
            match self.backend {
                Backend::Scalar => ScalarBackend::dot(&self.data, &other.data),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::dot(&self.data, &other.data),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 => Avx2Backend::dot(&self.data, &other.data),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX512 => Avx512Backend::dot(&self.data, &other.data),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::dot(&self.data, &other.data)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => NeonBackend::dot(&self.data, &other.data),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::dot(&self.data, &other.data),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => WasmBackend::dot(&self.data, &other.data),
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::dot(&self.data, &other.data),
                Backend::GPU | Backend::Auto => ScalarBackend::dot(&self.data, &other.data),
            }
        };

        Ok(result)
    }

    /// Sum all elements
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(v.sum().unwrap(), 10.0);
    /// ```
    pub fn sum(&self) -> Result<f32> {
        Ok(dispatch_reduction!(self.backend, sum, &self.data))
    }

    /// Find maximum element
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    /// assert_eq!(v.max().unwrap(), 5.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TruenoError::InvalidInput`] if vector is empty.
    pub fn max(&self) -> Result<f32> {
        if self.data.is_empty() {
            return Err(TruenoError::InvalidInput("Empty vector".to_string()));
        }

        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let result = unsafe {
            match self.backend {
                Backend::Scalar => ScalarBackend::max(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::max(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::max(&self.data),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::max(&self.data)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => NeonBackend::max(&self.data),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::max(&self.data),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => WasmBackend::max(&self.data),
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::max(&self.data),
                Backend::GPU | Backend::Auto => ScalarBackend::max(&self.data),
            }
        };

        Ok(result)
    }

    /// Find minimum value in the vector
    ///
    /// Returns the smallest element in the vector using SIMD optimization.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    /// assert_eq!(v.min().unwrap(), 1.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TruenoError::InvalidInput`] if vector is empty.
    pub fn min(&self) -> Result<f32> {
        if self.data.is_empty() {
            return Err(TruenoError::InvalidInput("Empty vector".to_string()));
        }

        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let result = unsafe {
            match self.backend {
                Backend::Scalar => ScalarBackend::min(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::min(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::min(&self.data),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::min(&self.data)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => NeonBackend::min(&self.data),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::min(&self.data),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => WasmBackend::min(&self.data),
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::min(&self.data),
                Backend::GPU | Backend::Auto => ScalarBackend::min(&self.data),
            }
        };

        Ok(result)
    }

    /// Find index of maximum value in the vector
    ///
    /// Returns the index of the first occurrence of the maximum value using SIMD optimization.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    /// assert_eq!(v.argmax().unwrap(), 1); // max value 5.0 is at index 1
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TruenoError::InvalidInput`] if vector is empty.
    pub fn argmax(&self) -> Result<usize> {
        if self.data.is_empty() {
            return Err(TruenoError::InvalidInput("Empty vector".to_string()));
        }

        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let result = unsafe {
            match self.backend {
                Backend::Scalar => ScalarBackend::argmax(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::argmax(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::argmax(&self.data),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::argmax(&self.data)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => NeonBackend::argmax(&self.data),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::argmax(&self.data),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => WasmBackend::argmax(&self.data),
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::argmax(&self.data),
                Backend::GPU | Backend::Auto => ScalarBackend::argmax(&self.data),
            }
        };

        Ok(result)
    }

    /// Find index of minimum value in the vector
    ///
    /// Returns the index of the first occurrence of the minimum value using SIMD optimization.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
    /// assert_eq!(v.argmin().unwrap(), 0); // min value 1.0 is at index 0
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TruenoError::InvalidInput`] if vector is empty.
    pub fn argmin(&self) -> Result<usize> {
        if self.data.is_empty() {
            return Err(TruenoError::InvalidInput("Empty vector".to_string()));
        }

        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let result = unsafe {
            match self.backend {
                Backend::Scalar => ScalarBackend::argmin(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::argmin(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::argmin(&self.data),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::argmin(&self.data)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => NeonBackend::argmin(&self.data),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::argmin(&self.data),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => WasmBackend::argmin(&self.data),
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::argmin(&self.data),
                Backend::GPU | Backend::Auto => ScalarBackend::argmin(&self.data),
            }
        };

        Ok(result)
    }

    /// Kahan summation (numerically stable sum)
    ///
    /// Uses the Kahan summation algorithm to reduce floating-point rounding errors
    /// when summing many numbers. This is more accurate than the standard sum() method
    /// for vectors with many elements or elements of vastly different magnitudes.
    ///
    /// # Performance
    ///
    /// Note: Kahan summation is inherently sequential and cannot be effectively
    /// parallelized with SIMD. All backends use the scalar implementation.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(v.sum_kahan().unwrap(), 10.0);
    /// ```
    pub fn sum_kahan(&self) -> Result<f32> {
        if self.data.is_empty() {
            return Ok(0.0);
        }

        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let result = unsafe {
            match self.backend {
                Backend::Scalar => ScalarBackend::sum_kahan(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::sum_kahan(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::sum_kahan(&self.data),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::sum_kahan(&self.data)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => NeonBackend::sum_kahan(&self.data),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::sum_kahan(&self.data),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => WasmBackend::sum_kahan(&self.data),
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::sum_kahan(&self.data),
                Backend::GPU | Backend::Auto => ScalarBackend::sum_kahan(&self.data),
            }
        };

        Ok(result)
    }

    /// Sum of squared elements
    ///
    /// Computes the sum of squares: sum(a\[i\]^2).
    /// This is the building block for computing L2 norm and variance.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let sum_sq = v.sum_of_squares().unwrap();
    /// assert_eq!(sum_sq, 14.0); // 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns 0.0 for empty vectors.
    pub fn sum_of_squares(&self) -> Result<f32> {
        if self.data.is_empty() {
            return Ok(0.0);
        }

        // Use dot product with self: dot(self, self) = sum(a[i]^2)
        self.dot(self)
    }

    /// Arithmetic mean (average)
    ///
    /// Computes the arithmetic mean of all elements: sum(a\[i\]) / n.
    ///
    /// # Performance
    ///
    /// Uses optimized SIMD sum() implementation, then divides by length.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// let avg = v.mean().unwrap();
    /// assert!((avg - 2.5).abs() < 1e-5); // (1+2+3+4)/4 = 2.5
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns an error for empty vectors (division by zero).
    ///
    /// ```
    /// use trueno::{Vector, TruenoError};
    ///
    /// let v: Vector<f32> = Vector::from_slice(&[]);
    /// assert!(matches!(v.mean(), Err(TruenoError::EmptyVector)));
    /// ```
    pub fn mean(&self) -> Result<f32> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        let total = self.sum()?;
        Ok(total / self.len() as f32)
    }

    /// Population variance
    ///
    /// Computes the population variance: Var(X) = E\[(X - μ)²\] = E\[X²\] - μ²
    /// Uses the computational formula to avoid two passes over the data.
    ///
    /// # Performance
    ///
    /// Uses optimized SIMD implementations via sum_of_squares() and mean().
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let var = v.variance().unwrap();
    /// assert!((var - 2.0).abs() < 1e-5); // Population variance
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns an error for empty vectors.
    ///
    /// ```
    /// use trueno::{Vector, TruenoError};
    ///
    /// let v: Vector<f32> = Vector::from_slice(&[]);
    /// assert!(matches!(v.variance(), Err(TruenoError::EmptyVector)));
    /// ```
    pub fn variance(&self) -> Result<f32> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        let mean_val = self.mean()?;
        let sum_sq = self.sum_of_squares()?;
        let mean_sq = sum_sq / self.len() as f32;

        // Var(X) = E[X²] - μ²
        Ok(mean_sq - mean_val * mean_val)
    }

    /// Population standard deviation
    ///
    /// Computes the population standard deviation: σ = sqrt(Var(X)).
    /// This is the square root of the variance.
    ///
    /// # Performance
    ///
    /// Uses optimized SIMD implementations via variance().
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let sd = v.stddev().unwrap();
    /// assert!((sd - 1.4142135).abs() < 1e-5); // sqrt(2) ≈ 1.414
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns an error for empty vectors.
    ///
    /// ```
    /// use trueno::{Vector, TruenoError};
    ///
    /// let v: Vector<f32> = Vector::from_slice(&[]);
    /// assert!(matches!(v.stddev(), Err(TruenoError::EmptyVector)));
    /// ```
    pub fn stddev(&self) -> Result<f32> {
        let var = self.variance()?;
        Ok(var.sqrt())
    }

    /// Population covariance between two vectors
    ///
    /// Computes the population covariance: Cov(X,Y) = E[(X - μx)(Y - μy)]
    /// Uses the computational formula: Cov(X,Y) = E\[XY\] - μx·μy
    ///
    /// # Performance
    ///
    /// Uses optimized SIMD implementations via dot() and mean().
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let y = Vector::from_slice(&[2.0, 4.0, 6.0]);
    /// let cov = x.covariance(&y).unwrap();
    /// assert!((cov - 1.333).abs() < 0.01); // Perfect positive covariance
    /// ```
    ///
    /// # Size mismatch
    ///
    /// Returns an error if vectors have different lengths.
    ///
    /// ```
    /// use trueno::{Vector, TruenoError};
    ///
    /// let x = Vector::from_slice(&[1.0, 2.0]);
    /// let y = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// assert!(matches!(x.covariance(&y), Err(TruenoError::SizeMismatch { .. })));
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns an error for empty vectors.
    ///
    /// ```
    /// use trueno::{Vector, TruenoError};
    ///
    /// let x: Vector<f32> = Vector::from_slice(&[]);
    /// let y: Vector<f32> = Vector::from_slice(&[]);
    /// assert!(matches!(x.covariance(&y), Err(TruenoError::EmptyVector)));
    /// ```
    pub fn covariance(&self, other: &Self) -> Result<f32> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        let mean_x = self.mean()?;
        let mean_y = other.mean()?;
        let dot_xy = self.dot(other)?;
        let mean_xy = dot_xy / self.len() as f32;

        // Cov(X,Y) = E[XY] - μx·μy
        Ok(mean_xy - mean_x * mean_y)
    }

    /// Pearson correlation coefficient
    ///
    /// Computes the Pearson correlation coefficient: ρ(X,Y) = Cov(X,Y) / (σx·σy)
    /// Normalized covariance in range [-1, 1].
    ///
    /// # Performance
    ///
    /// Uses optimized SIMD implementations via covariance() and stddev().
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let y = Vector::from_slice(&[2.0, 4.0, 6.0]);
    /// let corr = x.correlation(&y).unwrap();
    /// assert!((corr - 1.0).abs() < 1e-5); // Perfect positive correlation
    /// ```
    ///
    /// # Size mismatch
    ///
    /// Returns an error if vectors have different lengths.
    ///
    /// # Division by zero
    ///
    /// Returns DivisionByZero error if either vector has zero standard deviation
    /// (i.e., is constant).
    ///
    /// ```
    /// use trueno::{Vector, TruenoError};
    ///
    /// let x = Vector::from_slice(&[5.0, 5.0, 5.0]); // Constant
    /// let y = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// assert!(matches!(x.correlation(&y), Err(TruenoError::DivisionByZero)));
    /// ```
    pub fn correlation(&self, other: &Self) -> Result<f32> {
        let cov = self.covariance(other)?;
        let std_x = self.stddev()?;
        let std_y = other.stddev()?;

        // Check for zero standard deviation (constant vectors)
        if std_x.abs() < 1e-10 || std_y.abs() < 1e-10 {
            return Err(TruenoError::DivisionByZero);
        }

        // ρ(X,Y) = Cov(X,Y) / (σx·σy)
        // Clamp to [-1, 1] to handle floating-point precision errors
        let corr = cov / (std_x * std_y);
        Ok(corr.clamp(-1.0, 1.0))
    }
}
