//! Arithmetic operations for Vector<f32>
//!
//! This module provides element-wise arithmetic operations:
//! - Basic: `add`, `sub`, `mul`, `div`
//! - Scalar: `scale`
//! - Fused: `fma` (fused multiply-add)

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
use crate::{dispatch_binary_op, Backend, Result, TruenoError};

impl Vector<f32> {
    /// Element-wise addition
    ///
    /// # Performance
    ///
    /// Auto-selects the best available backend:
    /// - **AVX2**: ~4x faster than scalar for 1K+ elements
    /// - **GPU**: ~50x faster than scalar for 10M+ elements
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
    ///
    /// # Errors
    ///
    /// Returns [`TruenoError::SizeMismatch`] if vectors have different lengths.
    pub fn add(&self, other: &Self) -> Result<Self> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        let mut result = vec![0.0; self.len()];

        // Use parallel processing for large arrays
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 100_000; // Threshold for element-wise ops
            const CHUNK_SIZE: usize = 65536; // 64K elements = 256KB, cache-friendly

            if self.len() >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;

                self.data
                    .par_chunks(CHUNK_SIZE)
                    .zip(other.data.par_chunks(CHUNK_SIZE))
                    .zip(result.par_chunks_mut(CHUNK_SIZE))
                    .for_each(|((chunk_a, chunk_b), chunk_out)| {
                        dispatch_binary_op!(self.backend, add, chunk_a, chunk_b, chunk_out);
                    });

                return Ok(Self {
                    data: result,
                    backend: self.backend,
                });
            }
        }

        dispatch_binary_op!(self.backend, add, &self.data, &other.data, &mut result);

        Ok(Self {
            data: result,
            backend: self.backend,
        })
    }

    /// Element-wise subtraction
    ///
    /// # Performance
    ///
    /// Auto-selects the best available backend:
    /// - **AVX2**: ~4x faster than scalar for 1K+ elements
    /// - **GPU**: ~50x faster than scalar for 10M+ elements
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[5.0, 7.0, 9.0]);
    /// let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let result = a.sub(&b).unwrap();
    ///
    /// assert_eq!(result.as_slice(), &[4.0, 5.0, 6.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`TruenoError::SizeMismatch`] if vectors have different lengths.
    pub fn sub(&self, other: &Self) -> Result<Self> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        let mut result = vec![0.0; self.len()];

        // Use parallel processing for large arrays
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 100_000;
            const CHUNK_SIZE: usize = 65536;

            if self.len() >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;

                self.data
                    .par_chunks(CHUNK_SIZE)
                    .zip(other.data.par_chunks(CHUNK_SIZE))
                    .zip(result.par_chunks_mut(CHUNK_SIZE))
                    .for_each(|((chunk_a, chunk_b), chunk_out)| {
                        dispatch_binary_op!(self.backend, sub, chunk_a, chunk_b, chunk_out);
                    });

                return Ok(Self {
                    data: result,
                    backend: self.backend,
                });
            }
        }

        dispatch_binary_op!(self.backend, sub, &self.data, &other.data, &mut result);

        Ok(Self {
            data: result,
            backend: self.backend,
        })
    }

    /// Element-wise multiplication
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
    /// let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
    /// let result = a.mul(&b).unwrap();
    ///
    /// assert_eq!(result.as_slice(), &[10.0, 18.0, 28.0]);
    /// ```
    pub fn mul(&self, other: &Self) -> Result<Self> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        let mut result = vec![0.0; self.len()];

        // Use parallel processing for large arrays
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 100_000;
            const CHUNK_SIZE: usize = 65536;

            if self.len() >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;

                self.data
                    .par_chunks(CHUNK_SIZE)
                    .zip(other.data.par_chunks(CHUNK_SIZE))
                    .zip(result.par_chunks_mut(CHUNK_SIZE))
                    .for_each(|((chunk_a, chunk_b), chunk_out)| {
                        dispatch_binary_op!(self.backend, mul, chunk_a, chunk_b, chunk_out);
                    });

                return Ok(Self {
                    data: result,
                    backend: self.backend,
                });
            }
        }

        dispatch_binary_op!(self.backend, mul, &self.data, &other.data, &mut result);

        Ok(Self {
            data: result,
            backend: self.backend,
        })
    }

    /// Element-wise division
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[10.0, 20.0, 30.0]);
    /// let b = Vector::from_slice(&[2.0, 4.0, 5.0]);
    /// let result = a.div(&b).unwrap();
    ///
    /// assert_eq!(result.as_slice(), &[5.0, 5.0, 6.0]);
    /// ```
    pub fn div(&self, other: &Self) -> Result<Self> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        let mut result = vec![0.0; self.len()];

        // Use parallel processing for large arrays
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 100_000;
            const CHUNK_SIZE: usize = 65536;

            if self.len() >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;

                self.data
                    .par_chunks(CHUNK_SIZE)
                    .zip(other.data.par_chunks(CHUNK_SIZE))
                    .zip(result.par_chunks_mut(CHUNK_SIZE))
                    .for_each(|((chunk_a, chunk_b), chunk_out)| {
                        dispatch_binary_op!(self.backend, div, chunk_a, chunk_b, chunk_out);
                    });

                return Ok(Self {
                    data: result,
                    backend: self.backend,
                });
            }
        }

        dispatch_binary_op!(self.backend, div, &self.data, &other.data, &mut result);

        Ok(Self {
            data: result,
            backend: self.backend,
        })
    }

    /// Scalar multiplication (scale all elements by a scalar value)
    ///
    /// Returns a new vector where each element is multiplied by the scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// let result = v.scale(2.0).unwrap();
    ///
    /// assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
    /// ```
    ///
    /// # Scaling by Zero
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let result = v.scale(0.0).unwrap();
    /// assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
    /// ```
    ///
    /// # Negative Scaling
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, -2.0, 3.0]);
    /// let result = v.scale(-2.0).unwrap();
    /// assert_eq!(result.as_slice(), &[-2.0, 4.0, -6.0]);
    /// ```
    pub fn scale(&self, scalar: f32) -> Result<Vector<f32>> {
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend {
                    Backend::Scalar => ScalarBackend::scale(&self.data, scalar, &mut result_data),
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => {
                        Sse2Backend::scale(&self.data, scalar, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::scale(&self.data, scalar, &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => NeonBackend::scale(&self.data, scalar, &mut result_data),
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => WasmBackend::scale(&self.data, scalar, &mut result_data),
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        // Auto should have been resolved at creation time
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[allow(unreachable_patterns)]
                    _ => ScalarBackend::scale(&self.data, scalar, &mut result_data),
                }
            }
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Fused multiply-add: result\[i\] = self\[i\] * b\[i\] + c\[i\]
    ///
    /// Computes element-wise fused multiply-add operation. On hardware with FMA support
    /// (AVX2, NEON), this is a single instruction with better performance and numerical
    /// accuracy (no intermediate rounding). On platforms without FMA (SSE2, WASM), uses
    /// separate multiply and add operations.
    ///
    /// # Arguments
    ///
    /// * `b` - The second vector to multiply with
    /// * `c` - The vector to add to the product
    ///
    /// # Returns
    ///
    /// A new vector where each element is `self\[i\] * b\[i\] + c\[i\]`
    ///
    /// # Errors
    ///
    /// Returns `SizeMismatch` if vector lengths don't match
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
    /// let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
    /// let c = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let result = a.fma(&b, &c).unwrap();
    /// assert_eq!(result.as_slice(), &[11.0, 20.0, 31.0]);  // [2*5+1, 3*6+2, 4*7+3]
    /// ```
    ///
    /// # Use Cases
    ///
    /// - Neural networks: matrix multiplication, backpropagation
    /// - Scientific computing: polynomial evaluation, numerical integration
    /// - Graphics: transformation matrices, shader computations
    /// - Physics simulations: force calculations, particle systems
    pub fn fma(&self, b: &Vector<f32>, c: &Vector<f32>) -> Result<Vector<f32>> {
        if self.len() != b.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: b.len(),
            });
        }
        if self.len() != c.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: c.len(),
            });
        }

        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend {
                    Backend::Scalar => {
                        ScalarBackend::fma(&self.data, &b.data, &c.data, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => {
                        Sse2Backend::fma(&self.data, &b.data, &c.data, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::fma(&self.data, &b.data, &c.data, &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => {
                        NeonBackend::fma(&self.data, &b.data, &c.data, &mut result_data)
                    }
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => {
                        WasmBackend::fma(&self.data, &b.data, &c.data, &mut result_data)
                    }
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[allow(unreachable_patterns)]
                    _ => ScalarBackend::fma(&self.data, &b.data, &c.data, &mut result_data),
                }
            }
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Add Tests =====

    #[test]
    fn test_add_basic() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
        let result = a.add(&b).unwrap();
        assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_size_mismatch() {
        let a = Vector::from_slice(&[1.0, 2.0]);
        let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = a.add(&b);
        assert!(result.is_err());
        match result {
            Err(TruenoError::SizeMismatch { expected, actual }) => {
                assert_eq!(expected, 2);
                assert_eq!(actual, 3);
            }
            _ => panic!("Expected SizeMismatch error"),
        }
    }

    #[test]
    fn test_add_empty() {
        let a = Vector::from_slice(&[]);
        let b = Vector::from_slice(&[]);
        let result = a.add(&b).unwrap();
        assert!(result.as_slice().is_empty());
    }

    #[test]
    fn test_add_single_element() {
        let a = Vector::from_slice(&[1.5]);
        let b = Vector::from_slice(&[2.5]);
        let result = a.add(&b).unwrap();
        assert!((result.as_slice()[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_negatives() {
        let a = Vector::from_slice(&[-1.0, -2.0, -3.0]);
        let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = a.add(&b).unwrap();
        assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_add_large_array() {
        let n = 10000;
        let a = Vector::from_slice(&vec![1.0; n]);
        let b = Vector::from_slice(&vec![2.0; n]);
        let result = a.add(&b).unwrap();
        for val in result.as_slice() {
            assert!((val - 3.0).abs() < 1e-6);
        }
    }

    // ===== Sub Tests =====

    #[test]
    fn test_sub_basic() {
        let a = Vector::from_slice(&[5.0, 7.0, 9.0]);
        let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = a.sub(&b).unwrap();
        assert_eq!(result.as_slice(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sub_size_mismatch() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[1.0]);
        let result = a.sub(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_sub_empty() {
        let a = Vector::from_slice(&[]);
        let b = Vector::from_slice(&[]);
        let result = a.sub(&b).unwrap();
        assert!(result.as_slice().is_empty());
    }

    #[test]
    fn test_sub_self() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let result = a.sub(&a).unwrap();
        for val in result.as_slice() {
            assert!((val - 0.0).abs() < 1e-6);
        }
    }

    // ===== Mul Tests =====

    #[test]
    fn test_mul_basic() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[2.0, 3.0, 4.0]);
        let result = a.mul(&b).unwrap();
        assert_eq!(result.as_slice(), &[2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_mul_size_mismatch() {
        let a = Vector::from_slice(&[1.0]);
        let b = Vector::from_slice(&[1.0, 2.0]);
        let result = a.mul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_mul_empty() {
        let a = Vector::from_slice(&[]);
        let b = Vector::from_slice(&[]);
        let result = a.mul(&b).unwrap();
        assert!(result.as_slice().is_empty());
    }

    #[test]
    fn test_mul_by_zero() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = a.mul(&b).unwrap();
        assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mul_by_one() {
        let a = Vector::from_slice(&[5.0, 10.0, 15.0]);
        let b = Vector::from_slice(&[1.0, 1.0, 1.0]);
        let result = a.mul(&b).unwrap();
        assert_eq!(result.as_slice(), &[5.0, 10.0, 15.0]);
    }

    // ===== Div Tests =====

    #[test]
    fn test_div_basic() {
        let a = Vector::from_slice(&[4.0, 6.0, 8.0]);
        let b = Vector::from_slice(&[2.0, 2.0, 2.0]);
        let result = a.div(&b).unwrap();
        assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_div_size_mismatch() {
        let a = Vector::from_slice(&[1.0, 2.0]);
        let b = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let result = a.div(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_div_empty() {
        let a = Vector::from_slice(&[]);
        let b = Vector::from_slice(&[]);
        let result = a.div(&b).unwrap();
        assert!(result.as_slice().is_empty());
    }

    #[test]
    fn test_div_by_one() {
        let a = Vector::from_slice(&[5.0, 10.0, 15.0]);
        let b = Vector::from_slice(&[1.0, 1.0, 1.0]);
        let result = a.div(&b).unwrap();
        assert_eq!(result.as_slice(), &[5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_div_by_zero_produces_inf() {
        let a = Vector::from_slice(&[1.0, 2.0]);
        let b = Vector::from_slice(&[0.0, 0.0]);
        let result = a.div(&b).unwrap();
        assert!(result.as_slice()[0].is_infinite());
        assert!(result.as_slice()[1].is_infinite());
    }

    // ===== Scale Tests =====

    #[test]
    fn test_scale_basic() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = a.scale(2.0).unwrap();
        assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_scale_empty() {
        let a = Vector::from_slice(&[]);
        let result = a.scale(5.0).unwrap();
        assert!(result.as_slice().is_empty());
    }

    #[test]
    fn test_scale_by_zero() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = a.scale(0.0).unwrap();
        assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_scale_by_one() {
        let a = Vector::from_slice(&[5.0, 10.0, 15.0]);
        let result = a.scale(1.0).unwrap();
        assert_eq!(result.as_slice(), &[5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_scale_negative() {
        let a = Vector::from_slice(&[1.0, -2.0, 3.0]);
        let result = a.scale(-1.0).unwrap();
        assert_eq!(result.as_slice(), &[-1.0, 2.0, -3.0]);
    }

    // ===== FMA Tests =====

    #[test]
    fn test_fma_basic() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[2.0, 2.0, 2.0]);
        let c = Vector::from_slice(&[1.0, 1.0, 1.0]);
        // a * b + c = [2+1, 4+1, 6+1] = [3, 5, 7]
        let result = a.fma(&b, &c).unwrap();
        assert_eq!(result.as_slice(), &[3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_fma_size_mismatch_b() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[2.0]);
        let c = Vector::from_slice(&[1.0, 1.0, 1.0]);
        let result = a.fma(&b, &c);
        assert!(result.is_err());
    }

    #[test]
    fn test_fma_size_mismatch_c() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[2.0, 2.0, 2.0]);
        let c = Vector::from_slice(&[1.0]);
        let result = a.fma(&b, &c);
        assert!(result.is_err());
    }

    #[test]
    fn test_fma_empty() {
        let a = Vector::from_slice(&[]);
        let b = Vector::from_slice(&[]);
        let c = Vector::from_slice(&[]);
        let result = a.fma(&b, &c).unwrap();
        assert!(result.as_slice().is_empty());
    }

    #[test]
    fn test_fma_multiply_by_zero() {
        let a = Vector::from_slice(&[5.0, 10.0, 15.0]);
        let b = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let c = Vector::from_slice(&[1.0, 2.0, 3.0]);
        // a * 0 + c = c
        let result = a.fma(&b, &c).unwrap();
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_fma_add_zero() {
        let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
        let b = Vector::from_slice(&[3.0, 2.0, 1.0]);
        let c = Vector::from_slice(&[0.0, 0.0, 0.0]);
        // a * b + 0 = a * b
        let result = a.fma(&b, &c).unwrap();
        assert_eq!(result.as_slice(), &[6.0, 6.0, 4.0]);
    }

    // ===== Backend Tests =====

    #[test]
    fn test_add_scalar_backend() {
        let a = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::Scalar);
        let b = Vector::from_slice_with_backend(&[4.0, 5.0, 6.0], Backend::Scalar);
        let result = a.add(&b).unwrap();
        assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_add_sse2_backend() {
        let a = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0, 4.0], Backend::SSE2);
        let b = Vector::from_slice_with_backend(&[4.0, 5.0, 6.0, 7.0], Backend::SSE2);
        let result = a.add(&b).unwrap();
        assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0, 11.0]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_add_avx2_backend() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return; // Skip if AVX2 not available
        }
        let data: Vec<f32> = vec![1.0; 16];
        let a = Vector::from_slice_with_backend(&data, Backend::AVX2);
        let b_data: Vec<f32> = vec![2.0; 16];
        let b = Vector::from_slice_with_backend(&b_data, Backend::AVX2);
        let result = a.add(&b).unwrap();
        for &val in result.as_slice() {
            assert!((val - 3.0).abs() < 1e-6);
        }
    }

    // ===== Edge Cases =====

    #[test]
    fn test_add_non_aligned_size() {
        // Test with sizes that don't align to SIMD register widths
        let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]); // 7 elements
        let b = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let result = a.add(&b).unwrap();
        assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_mul_preserves_sign() {
        let a = Vector::from_slice(&[2.0, -2.0, 2.0, -2.0]);
        let b = Vector::from_slice(&[3.0, 3.0, -3.0, -3.0]);
        let result = a.mul(&b).unwrap();
        assert_eq!(result.as_slice(), &[6.0, -6.0, -6.0, 6.0]);
    }

    #[test]
    fn test_operations_with_special_floats() {
        let a = Vector::from_slice(&[f32::INFINITY, f32::NEG_INFINITY, 0.0]);
        let b = Vector::from_slice(&[1.0, 1.0, 1.0]);
        let result = a.add(&b).unwrap();
        assert!(result.as_slice()[0].is_infinite());
        assert!(result.as_slice()[1].is_infinite());
        assert!((result.as_slice()[2] - 1.0).abs() < 1e-6);
    }
}
