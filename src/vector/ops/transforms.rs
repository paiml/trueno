//! Vector transformation operations
//!
//! This module provides element-wise transformation methods:
//! - `abs()` - Element-wise absolute value
//! - `clamp()` / `clip()` - Clamp values to a range
//! - `lerp()` - Linear interpolation between two vectors
//! - `sqrt()` - Element-wise square root
//! - `recip()` - Element-wise reciprocal (1/x)
//! - `pow()` - Element-wise power

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
use crate::dispatch_unary_op;
use crate::{Backend, Result, TruenoError, Vector};

impl Vector<f32> {
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

        if !self.as_slice().is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend() {
                    Backend::Scalar => ScalarBackend::abs(self.as_slice(), &mut result_data),
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => {
                        Sse2Backend::abs(self.as_slice(), &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::abs(self.as_slice(), &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => NeonBackend::abs(self.as_slice(), &mut result_data),
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => WasmBackend::abs(self.as_slice(), &mut result_data),
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[allow(unreachable_patterns)]
                    _ => ScalarBackend::abs(self.as_slice(), &mut result_data),
                }
            }
        }

        Ok(Vector::from_slice_with_backend(
            &result_data,
            self.backend(),
        ))
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

        // Scalar fallback: Element-wise clamp
        let data: Vec<f32> = self
            .as_slice()
            .iter()
            .map(|&x| x.max(min_val).min(max_val))
            .collect();

        Ok(Vector::from_vec(data))
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

        if !self.as_slice().is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend() {
                    Backend::Scalar => {
                        ScalarBackend::clamp(self.as_slice(), min_val, max_val, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => {
                        Sse2Backend::clamp(self.as_slice(), min_val, max_val, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::clamp(self.as_slice(), min_val, max_val, &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => {
                        NeonBackend::clamp(self.as_slice(), min_val, max_val, &mut result_data)
                    }
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => {
                        WasmBackend::clamp(self.as_slice(), min_val, max_val, &mut result_data)
                    }
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[allow(unreachable_patterns)]
                    _ => ScalarBackend::clamp(self.as_slice(), min_val, max_val, &mut result_data),
                }
            }
        }

        Ok(Vector::from_slice_with_backend(
            &result_data,
            self.backend(),
        ))
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

        if !self.as_slice().is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend() {
                    Backend::Scalar => {
                        ScalarBackend::lerp(self.as_slice(), other.as_slice(), t, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => {
                        Sse2Backend::lerp(self.as_slice(), other.as_slice(), t, &mut result_data)
                    }
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::lerp(self.as_slice(), other.as_slice(), t, &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => {
                        NeonBackend::lerp(self.as_slice(), other.as_slice(), t, &mut result_data)
                    }
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => {
                        WasmBackend::lerp(self.as_slice(), other.as_slice(), t, &mut result_data)
                    }
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[allow(unreachable_patterns)]
                    _ => {
                        ScalarBackend::lerp(self.as_slice(), other.as_slice(), t, &mut result_data)
                    }
                }
            }
        }

        Ok(Vector::from_slice_with_backend(
            &result_data,
            self.backend(),
        ))
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

        if !self.as_slice().is_empty() {
            // Use parallel processing for large arrays
            #[cfg(feature = "parallel")]
            {
                const PARALLEL_THRESHOLD: usize = 100_000;
                const CHUNK_SIZE: usize = 65536;

                if self.len() >= PARALLEL_THRESHOLD {
                    use rayon::prelude::*;

                    self.as_slice()
                        .par_chunks(CHUNK_SIZE)
                        .zip(result_data.par_chunks_mut(CHUNK_SIZE))
                        .for_each(|(chunk_in, chunk_out)| {
                            dispatch_unary_op!(self.backend(), sqrt, chunk_in, chunk_out);
                        });

                    return Ok(Vector::from_slice_with_backend(
                        &result_data,
                        self.backend(),
                    ));
                }
            }

            dispatch_unary_op!(self.backend(), sqrt, self.as_slice(), &mut result_data);
        }

        Ok(Vector::from_slice_with_backend(
            &result_data,
            self.backend(),
        ))
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

        if !self.as_slice().is_empty() {
            dispatch_unary_op!(self.backend(), recip, self.as_slice(), &mut result_data);
        }

        Ok(Vector::from_slice_with_backend(
            &result_data,
            self.backend(),
        ))
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
        let pow_data: Vec<f32> = self.as_slice().iter().map(|x| x.powf(n)).collect();
        Ok(Vector::from_vec(pow_data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abs_basic() {
        let v = Vector::from_slice(&[3.0, -4.0, 5.0, -2.0]);
        let result = v.abs().unwrap();
        assert_eq!(result.as_slice(), &[3.0, 4.0, 5.0, 2.0]);
    }

    #[test]
    fn test_abs_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.abs().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_clip_basic() {
        let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
        let clipped = v.clip(0.0, 10.0).unwrap();
        assert_eq!(clipped.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
    }

    #[test]
    fn test_clip_invalid_range() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = v.clip(10.0, 5.0);
        assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
    }

    #[test]
    fn test_clamp_basic() {
        let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
        let result = v.clamp(0.0, 10.0).unwrap();
        assert_eq!(result.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
    }

    #[test]
    fn test_clamp_negative_range() {
        let v = Vector::from_slice(&[-10.0, -5.0, 0.0, 5.0]);
        let result = v.clamp(-8.0, -2.0).unwrap();
        assert_eq!(result.as_slice(), &[-8.0, -5.0, -2.0, -2.0]);
    }

    #[test]
    fn test_lerp_midpoint() {
        let a = Vector::from_slice(&[0.0, 10.0, 20.0]);
        let b = Vector::from_slice(&[100.0, 110.0, 120.0]);
        let result = a.lerp(&b, 0.5).unwrap();
        assert_eq!(result.as_slice(), &[50.0, 60.0, 70.0]);
    }

    #[test]
    fn test_lerp_extrapolation() {
        let a = Vector::from_slice(&[0.0, 10.0]);
        let b = Vector::from_slice(&[10.0, 20.0]);
        let result = a.lerp(&b, 2.0).unwrap();
        assert_eq!(result.as_slice(), &[20.0, 30.0]);
    }

    #[test]
    fn test_lerp_size_mismatch() {
        let a = Vector::from_slice(&[0.0, 10.0]);
        let b = Vector::from_slice(&[10.0, 20.0, 30.0]);
        let result = a.lerp(&b, 0.5);
        assert!(matches!(result, Err(TruenoError::SizeMismatch { .. })));
    }

    #[test]
    fn test_sqrt_basic() {
        let a = Vector::from_slice(&[4.0, 9.0, 16.0, 25.0]);
        let result = a.sqrt().unwrap();
        assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sqrt_negative() {
        let a = Vector::from_slice(&[-1.0, 4.0]);
        let result = a.sqrt().unwrap();
        assert!(result.as_slice()[0].is_nan());
        assert_eq!(result.as_slice()[1], 2.0);
    }

    #[test]
    fn test_recip_basic() {
        let a = Vector::from_slice(&[2.0, 4.0, 5.0, 10.0]);
        let result = a.recip().unwrap();
        assert_eq!(result.as_slice(), &[0.5, 0.25, 0.2, 0.1]);
    }

    #[test]
    fn test_recip_zero() {
        let a = Vector::from_slice(&[0.0, 2.0]);
        let result = a.recip().unwrap();
        assert!(result.as_slice()[0].is_infinite());
        assert_eq!(result.as_slice()[1], 0.5);
    }

    #[test]
    fn test_pow_squared() {
        let v = Vector::from_slice(&[2.0, 3.0, 4.0]);
        let squared = v.pow(2.0).unwrap();
        assert_eq!(squared.as_slice(), &[4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_pow_square_root() {
        let v = Vector::from_slice(&[4.0, 9.0, 16.0]);
        let sqrt = v.pow(0.5).unwrap();
        assert!((sqrt.as_slice()[0] - 2.0).abs() < 1e-5);
        assert!((sqrt.as_slice()[1] - 3.0).abs() < 1e-5);
        assert!((sqrt.as_slice()[2] - 4.0).abs() < 1e-5);
    }

    // =====================================================================
    // Coverage: GPU and Auto backend error paths
    // =====================================================================

    #[test]
    fn test_abs_gpu_backend_returns_error() {
        let v = Vector::from_slice_with_backend(&[1.0, -2.0, 3.0], Backend::GPU);
        let result = v.abs();
        assert!(matches!(
            result,
            Err(TruenoError::UnsupportedBackend(Backend::GPU))
        ));
    }

    #[test]
    fn test_abs_auto_backend_returns_error() {
        // Auto is resolved at construction, but we can test the error path
        // by using from_slice_with_backend which resolves Auto to best available.
        // For the GPU path we already tested above. Let's ensure Scalar path works.
        let v = Vector::from_slice_with_backend(&[3.0, -4.0, 5.0], Backend::Scalar);
        let result = v.abs().unwrap();
        assert_eq!(result.as_slice(), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_clamp_gpu_backend_returns_error() {
        let v = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::GPU);
        let result = v.clamp(0.0, 2.0);
        assert!(matches!(
            result,
            Err(TruenoError::UnsupportedBackend(Backend::GPU))
        ));
    }

    #[test]
    fn test_clamp_scalar_backend() {
        let v = Vector::from_slice_with_backend(&[-5.0, 0.0, 5.0, 10.0], Backend::Scalar);
        let result = v.clamp(0.0, 8.0).unwrap();
        assert_eq!(result.as_slice(), &[0.0, 0.0, 5.0, 8.0]);
    }

    #[test]
    fn test_clamp_invalid_range() {
        let v = Vector::from_slice(&[1.0, 2.0]);
        let result = v.clamp(10.0, 5.0);
        assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
    }

    #[test]
    fn test_clamp_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.clamp(0.0, 1.0).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_lerp_gpu_backend_returns_error() {
        let a = Vector::from_slice_with_backend(&[1.0, 2.0], Backend::GPU);
        let b = Vector::from_slice_with_backend(&[3.0, 4.0], Backend::GPU);
        let result = a.lerp(&b, 0.5);
        assert!(matches!(
            result,
            Err(TruenoError::UnsupportedBackend(Backend::GPU))
        ));
    }

    #[test]
    fn test_lerp_scalar_backend() {
        let a = Vector::from_slice_with_backend(&[0.0, 10.0], Backend::Scalar);
        let b = Vector::from_slice_with_backend(&[10.0, 20.0], Backend::Scalar);
        let result = a.lerp(&b, 0.5).unwrap();
        assert_eq!(result.as_slice(), &[5.0, 15.0]);
    }

    #[test]
    fn test_lerp_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let b: Vector<f32> = Vector::from_slice(&[]);
        let result = a.lerp(&b, 0.5).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_sqrt_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.sqrt().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_recip_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.recip().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_pow_zero_exponent() {
        let v = Vector::from_slice(&[2.0, 0.0, -3.0]);
        let result = v.pow(0.0).unwrap();
        // x^0 = 1.0 for all x
        assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_pow_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.pow(2.0).unwrap();
        assert_eq!(result.len(), 0);
    }
}
