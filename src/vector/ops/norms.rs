//! Vector norm operations
//!
//! This module provides vector norm calculations:
//! - `norm_l1()` - L1 norm (Manhattan norm)
//! - `norm_l2()` - L2 norm (Euclidean norm)
//! - `norm_linf()` - L∞ norm (infinity norm / max norm)

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
use crate::{Backend, Result, Vector};

impl Vector<f32> {
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
        if self.as_slice().is_empty() {
            return Ok(0.0);
        }

        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let result = unsafe {
            match self.backend() {
                Backend::Scalar => ScalarBackend::norm_l2(self.as_slice()),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::norm_l2(self.as_slice()),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::norm_l2(self.as_slice()),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::norm_l2(self.as_slice())
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => NeonBackend::norm_l2(self.as_slice()),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::norm_l2(self.as_slice()),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => WasmBackend::norm_l2(self.as_slice()),
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::norm_l2(self.as_slice()),
                Backend::GPU | Backend::Auto => ScalarBackend::norm_l2(self.as_slice()),
            }
        };

        Ok(result)
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
        if self.as_slice().is_empty() {
            return Ok(0.0);
        }

        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let result = unsafe {
            match self.backend() {
                Backend::Scalar => ScalarBackend::norm_l1(self.as_slice()),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::norm_l1(self.as_slice()),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::norm_l1(self.as_slice()),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::norm_l1(self.as_slice())
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => NeonBackend::norm_l1(self.as_slice()),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::norm_l1(self.as_slice()),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => WasmBackend::norm_l1(self.as_slice()),
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::norm_l1(self.as_slice()),
                Backend::GPU | Backend::Auto => ScalarBackend::norm_l1(self.as_slice()),
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
        if self.as_slice().is_empty() {
            return Ok(0.0);
        }

        // Use optimized SIMD backend for single-pass abs+max
        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let max_abs = unsafe {
            match self.backend() {
                Backend::Scalar => ScalarBackend::norm_linf(self.as_slice()),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::norm_linf(self.as_slice()),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::norm_linf(self.as_slice()),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::norm_linf(self.as_slice())
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => ScalarBackend::norm_linf(self.as_slice()), // NEON fallback
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::norm_linf(self.as_slice()),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => ScalarBackend::norm_linf(self.as_slice()), // WASM fallback
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::norm_linf(self.as_slice()),
                Backend::GPU | Backend::Auto => ScalarBackend::norm_linf(self.as_slice()),
            }
        };

        Ok(max_abs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_norm_l2_pythagorean() {
        let v = Vector::from_slice(&[3.0, 4.0]);
        let norm = v.norm_l2().unwrap();
        assert!((norm - 5.0).abs() < 1e-5); // 3-4-5 triangle
    }

    #[test]
    fn test_norm_l2_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        assert_eq!(v.norm_l2().unwrap(), 0.0);
    }

    #[test]
    fn test_norm_l2_unit() {
        let v = Vector::from_slice(&[1.0, 0.0, 0.0]);
        assert!((v.norm_l2().unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l1_basic() {
        let v = Vector::from_slice(&[3.0, -4.0, 5.0]);
        let norm = v.norm_l1().unwrap();
        assert!((norm - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l1_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        assert_eq!(v.norm_l1().unwrap(), 0.0);
    }

    #[test]
    fn test_norm_linf_basic() {
        let v = Vector::from_slice(&[3.0, -7.0, 5.0, -2.0]);
        let norm = v.norm_linf().unwrap();
        assert!((norm - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_linf_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        assert_eq!(v.norm_linf().unwrap(), 0.0);
    }

    #[test]
    fn test_norm_linf_all_negative() {
        let v = Vector::from_slice(&[-1.0, -5.0, -3.0]);
        let norm = v.norm_linf().unwrap();
        assert!((norm - 5.0).abs() < 1e-5);
    }
}
