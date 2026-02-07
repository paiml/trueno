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

    // =========================================================================
    // L2 norm: additional edge cases
    // =========================================================================

    #[test]
    fn test_norm_l2_single_element() {
        let v = Vector::from_slice(&[7.0]);
        let norm = v.norm_l2().unwrap();
        assert!((norm - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l2_single_negative() {
        let v = Vector::from_slice(&[-5.0]);
        let norm = v.norm_l2().unwrap();
        assert!((norm - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l2_all_zeros() {
        let v = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);
        let norm = v.norm_l2().unwrap();
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_norm_l2_large_vector() {
        // Large vector to exercise SIMD paths
        let data: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01).collect();
        let v = Vector::from_slice(&data);
        let norm = v.norm_l2().unwrap();
        // norm = sqrt(sum(i^2 * 0.0001)) for i in 0..1024
        let expected: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - expected).abs() < 1e-2, "Got {} expected {}", norm, expected);
    }

    #[test]
    fn test_norm_l2_mixed_positive_negative() {
        let v = Vector::from_slice(&[3.0, -4.0, 0.0]);
        let norm = v.norm_l2().unwrap();
        // sqrt(9 + 16 + 0) = 5
        assert!((norm - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l2_known_identity() {
        // L2 norm of unit vectors
        let v = Vector::from_slice(&[0.0, 0.0, 1.0]);
        assert!((v.norm_l2().unwrap() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l2_non_aligned_size() {
        // 5 elements - not a power of 2, exercises remainder handling
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let expected = (1.0 + 4.0 + 9.0 + 16.0 + 25.0_f32).sqrt();
        assert!((v.norm_l2().unwrap() - expected).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l2_very_small_values() {
        let v = Vector::from_slice(&[1e-20, 1e-20, 1e-20]);
        let norm = v.norm_l2().unwrap();
        assert!(norm > 0.0, "Norm of small values should be positive");
        assert!(norm < 1e-10, "Norm should be very small");
    }

    // =========================================================================
    // L1 norm: additional edge cases
    // =========================================================================

    #[test]
    fn test_norm_l1_single_element() {
        let v = Vector::from_slice(&[-7.0]);
        let norm = v.norm_l1().unwrap();
        assert!((norm - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l1_all_zeros() {
        let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let norm = v.norm_l1().unwrap();
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_norm_l1_all_positive() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let norm = v.norm_l1().unwrap();
        assert!((norm - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l1_all_negative() {
        let v = Vector::from_slice(&[-1.0, -2.0, -3.0]);
        let norm = v.norm_l1().unwrap();
        assert!((norm - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l1_large_vector() {
        let data: Vec<f32> = (0..512).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let v = Vector::from_slice(&data);
        let norm = v.norm_l1().unwrap();
        assert!((norm - 512.0).abs() < 1e-3);
    }

    #[test]
    fn test_norm_l1_non_aligned_size() {
        let v = Vector::from_slice(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0]);
        let expected = 1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0;
        assert!((v.norm_l1().unwrap() - expected).abs() < 1e-5);
    }

    // =========================================================================
    // L-infinity norm: additional edge cases
    // =========================================================================

    #[test]
    fn test_norm_linf_single_element() {
        let v = Vector::from_slice(&[-42.0]);
        let norm = v.norm_linf().unwrap();
        assert!((norm - 42.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_linf_all_zeros() {
        let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let norm = v.norm_linf().unwrap();
        assert_eq!(norm, 0.0);
    }

    #[test]
    fn test_norm_linf_max_at_end() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 100.0]);
        let norm = v.norm_linf().unwrap();
        assert!((norm - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_linf_max_at_beginning() {
        let v = Vector::from_slice(&[-100.0, 2.0, 3.0, 4.0]);
        let norm = v.norm_linf().unwrap();
        assert!((norm - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_linf_large_vector() {
        let mut data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        data[200] = -99.9;
        let v = Vector::from_slice(&data);
        let norm = v.norm_linf().unwrap();
        assert!((norm - 99.9).abs() < 1e-3);
    }

    #[test]
    fn test_norm_linf_all_equal() {
        let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
        let norm = v.norm_linf().unwrap();
        assert!((norm - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_linf_non_aligned_size() {
        let v = Vector::from_slice(&[1.0, -9.0, 3.0, -4.0, 5.0]);
        assert!((v.norm_linf().unwrap() - 9.0).abs() < 1e-5);
    }

    // =========================================================================
    // Cross-norm property: L-inf <= L2 <= L1 for any vector
    // =========================================================================

    #[test]
    fn test_norm_ordering_property() {
        let v = Vector::from_slice(&[3.0, -4.0, 5.0, -2.0, 1.0]);
        let l1 = v.norm_l1().unwrap();
        let l2 = v.norm_l2().unwrap();
        let linf = v.norm_linf().unwrap();

        assert!(
            linf <= l2 + 1e-5,
            "L-inf ({}) should be <= L2 ({})", linf, l2
        );
        assert!(
            l2 <= l1 + 1e-5,
            "L2 ({}) should be <= L1 ({})", l2, l1
        );
    }

    #[test]
    fn test_norm_ordering_property_large() {
        let data: Vec<f32> = (0..100).map(|i| ((i as f32) * 0.37).sin()).collect();
        let v = Vector::from_slice(&data);
        let l1 = v.norm_l1().unwrap();
        let l2 = v.norm_l2().unwrap();
        let linf = v.norm_linf().unwrap();

        assert!(linf <= l2 + 1e-4, "L-inf <= L2 failed");
        assert!(l2 <= l1 + 1e-4, "L2 <= L1 failed");
    }
}
