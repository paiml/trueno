//! Vector norm operations
//!
//! This module provides vector norm calculations:
//! - `norm_l1()` - L1 norm (Manhattan norm)
//! - `norm_l2()` - L2 norm (Euclidean norm)
//! - `norm_linf()` - L∞ norm (infinity norm / max norm)
//!
//! All three norms share the same `unsafe fn(&[f32]) -> f32` backend
//! signature, so dispatch is handled by the crate-level
//! [`dispatch_reduction!`] macro -- no per-module dispatch struct needed.

use crate::backends::VectorBackend;
use crate::{dispatch_reduction, Result, Vector};

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
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.0, 4.0]);
    /// let norm = v.norm_l2()?;
    /// assert!((norm - 5.0).abs() < 1e-5); // sqrt(3^2 + 4^2) = 5
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns 0.0 for empty vectors (consistent with the mathematical definition).
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v: Vector<f32> = Vector::from_slice(&[]);
    /// assert_eq!(v.norm_l2()?, 0.0);
    /// # Ok(())
    /// # }
    /// ```
    pub fn norm_l2(&self) -> Result<f32> {
        if self.data.is_empty() {
            return Ok(0.0);
        }
        Ok(dispatch_reduction!(self.backend, norm_l2, &self.data))
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
        Ok(dispatch_reduction!(self.backend, norm_l1, &self.data))
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
        Ok(dispatch_reduction!(self.backend, norm_linf, &self.data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend;

    // =========================================================================
    // Parametric test helpers — eliminates repeated backend/norm boilerplate
    // =========================================================================

    type NormMethod = fn(&Vector<f32>) -> Result<f32>;

    fn norm_l1(v: &Vector<f32>) -> Result<f32> { v.norm_l1() }
    fn norm_l2(v: &Vector<f32>) -> Result<f32> { v.norm_l2() }
    fn norm_linf(v: &Vector<f32>) -> Result<f32> { v.norm_linf() }

    fn assert_norm(norm_fn: NormMethod, data: &[f32], expected: f32, tol: f32) {
        let v = Vector::from_slice(data);
        let result = norm_fn(&v).unwrap();
        assert!((result - expected).abs() <= tol, "expected {expected} got {result}");
    }

    fn assert_norm_with_backend(
        norm_fn: NormMethod, data: &[f32], expected: f32, tol: f32, backend: Backend,
    ) {
        let v = Vector::from_slice_with_backend(data, backend);
        let result = norm_fn(&v).unwrap();
        assert!((result - expected).abs() <= tol, "expected {expected} got {result} ({backend:?})");
    }

    fn assert_norm_backend_equivalence(norm_fn: NormMethod, data: &[f32], tol: f32) {
        let scalar = norm_fn(&Vector::from_slice_with_backend(data, Backend::Scalar)).unwrap();
        for &backend in &[Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto] {
            let val = norm_fn(&Vector::from_slice_with_backend(data, backend)).unwrap();
            assert!((scalar - val).abs() < tol, "Scalar vs {backend:?}: {scalar} vs {val}");
        }
        #[cfg(target_arch = "x86_64")]
        {
            let sse2 = norm_fn(&Vector::from_slice_with_backend(data, Backend::SSE2)).unwrap();
            assert!((scalar - sse2).abs() < tol, "Scalar vs SSE2: {scalar} vs {sse2}");
            if is_x86_feature_detected!("avx2") {
                let avx2 = norm_fn(&Vector::from_slice_with_backend(data, Backend::AVX2)).unwrap();
                assert!((scalar - avx2).abs() < tol, "Scalar vs AVX2: {scalar} vs {avx2}");
            }
        }
    }

    fn assert_norm_non_aligned(
        norm_fn: NormMethod,
        make_data: fn(usize) -> Vec<f32>,
        make_expected: fn(&[f32]) -> f32,
        tol: f32,
    ) {
        for size in [1, 2, 3, 5, 7, 9, 13, 15, 17, 31, 33] {
            let data = make_data(size);
            let result = norm_fn(&Vector::from_slice(&data)).unwrap();
            let expected = make_expected(&data);
            assert!((result - expected).abs() < tol, "size {size}: {result} vs {expected}");
        }
    }

    fn assert_norm_ordering(data: &[f32]) {
        let v = Vector::from_slice(data);
        let l1 = v.norm_l1().unwrap();
        let l2 = v.norm_l2().unwrap();
        let linf = v.norm_linf().unwrap();
        assert!(linf <= l2 + 1e-4, "L-inf ({linf}) should be <= L2 ({l2})");
        assert!(l2 <= l1 + 1e-4, "L2 ({l2}) should be <= L1 ({l1})");
    }

    // =========================================================================
    // L2 norm: edge cases
    // =========================================================================

    #[test]
    fn test_norm_l2_pythagorean() { assert_norm(norm_l2, &[3.0, 4.0], 5.0, 1e-5); }

    #[test]
    fn test_norm_l2_empty() { assert_norm(norm_l2, &[], 0.0, 0.0); }

    #[test]
    fn test_norm_l2_unit() { assert_norm(norm_l2, &[1.0, 0.0, 0.0], 1.0, 1e-5); }

    #[test]
    fn test_norm_l2_single_element() { assert_norm(norm_l2, &[7.0], 7.0, 1e-5); }

    #[test]
    fn test_norm_l2_single_negative() { assert_norm(norm_l2, &[-5.0], 5.0, 1e-5); }

    #[test]
    fn test_norm_l2_all_zeros() { assert_norm(norm_l2, &[0.0, 0.0, 0.0, 0.0], 0.0, 0.0); }

    #[test]
    fn test_norm_l2_mixed_positive_negative() { assert_norm(norm_l2, &[3.0, -4.0, 0.0], 5.0, 1e-5); }

    #[test]
    fn test_norm_l2_known_identity() { assert_norm(norm_l2, &[0.0, 0.0, 1.0], 1.0, 1e-5); }

    #[test]
    fn test_norm_l2_non_aligned_size() {
        let expected = (1.0 + 4.0 + 9.0 + 16.0 + 25.0_f32).sqrt();
        assert_norm(norm_l2, &[1.0, 2.0, 3.0, 4.0, 5.0], expected, 1e-5);
    }

    #[test]
    fn test_norm_l2_large_vector() {
        let data: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01).collect();
        let expected: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_norm(norm_l2, &data, expected, 1e-2);
    }

    #[test]
    fn test_norm_l2_very_small_values() {
        let v = Vector::from_slice(&[1e-20, 1e-20, 1e-20]);
        let norm = v.norm_l2().unwrap();
        assert!(norm > 0.0, "Norm of small values should be positive");
        assert!(norm < 1e-10, "Norm should be very small");
    }

    // =========================================================================
    // L1 norm: edge cases
    // =========================================================================

    #[test]
    fn test_norm_l1_basic() { assert_norm(norm_l1, &[3.0, -4.0, 5.0], 12.0, 1e-5); }

    #[test]
    fn test_norm_l1_empty() { assert_norm(norm_l1, &[], 0.0, 0.0); }

    #[test]
    fn test_norm_l1_single_element() { assert_norm(norm_l1, &[-7.0], 7.0, 1e-5); }

    #[test]
    fn test_norm_l1_all_zeros() { assert_norm(norm_l1, &[0.0, 0.0, 0.0], 0.0, 0.0); }

    #[test]
    fn test_norm_l1_all_positive() { assert_norm(norm_l1, &[1.0, 2.0, 3.0, 4.0], 10.0, 1e-5); }

    #[test]
    fn test_norm_l1_all_negative() { assert_norm(norm_l1, &[-1.0, -2.0, -3.0], 6.0, 1e-5); }

    #[test]
    fn test_norm_l1_non_aligned_size() { assert_norm(norm_l1, &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0], 28.0, 1e-5); }

    #[test]
    fn test_norm_l1_large_vector() {
        let data: Vec<f32> = (0..512).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert_norm(norm_l1, &data, 512.0, 1e-3);
    }

    // =========================================================================
    // L-infinity norm: edge cases
    // =========================================================================

    #[test]
    fn test_norm_linf_basic() { assert_norm(norm_linf, &[3.0, -7.0, 5.0, -2.0], 7.0, 1e-5); }

    #[test]
    fn test_norm_linf_empty() { assert_norm(norm_linf, &[], 0.0, 0.0); }

    #[test]
    fn test_norm_linf_all_negative() { assert_norm(norm_linf, &[-1.0, -5.0, -3.0], 5.0, 1e-5); }

    #[test]
    fn test_norm_linf_single_element() { assert_norm(norm_linf, &[-42.0], 42.0, 1e-5); }

    #[test]
    fn test_norm_linf_all_zeros() { assert_norm(norm_linf, &[0.0, 0.0, 0.0], 0.0, 0.0); }

    #[test]
    fn test_norm_linf_max_at_end() { assert_norm(norm_linf, &[1.0, 2.0, 3.0, 100.0], 100.0, 1e-5); }

    #[test]
    fn test_norm_linf_max_at_beginning() { assert_norm(norm_linf, &[-100.0, 2.0, 3.0, 4.0], 100.0, 1e-5); }

    #[test]
    fn test_norm_linf_all_equal() { assert_norm(norm_linf, &[5.0, 5.0, 5.0, 5.0], 5.0, 1e-5); }

    #[test]
    fn test_norm_linf_non_aligned_size() { assert_norm(norm_linf, &[1.0, -9.0, 3.0, -4.0, 5.0], 9.0, 1e-5); }

    #[test]
    fn test_norm_linf_large_vector() {
        let mut data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        data[200] = -99.9;
        assert_norm(norm_linf, &data, 99.9, 1e-3);
    }

    // =========================================================================
    // Cross-norm property: L-inf <= L2 <= L1
    // =========================================================================

    #[test]
    fn test_norm_ordering_property() { assert_norm_ordering(&[3.0, -4.0, 5.0, -2.0, 1.0]); }

    #[test]
    fn test_norm_ordering_property_large() {
        let data: Vec<f32> = (0..100).map(|i| ((i as f32) * 0.37).sin()).collect();
        assert_norm_ordering(&data);
    }

    // =========================================================================
    // Norm spec table — (method, name, basic_data, expected) for loop-based tests
    // =========================================================================

    fn norm_specs() -> [(NormMethod, &'static str, &'static [f32], f32); 3] {
        [
            (norm_l1, "l1", &[3.0, -4.0, 5.0], 12.0),
            (norm_l2, "l2", &[3.0, 4.0, 0.0, 0.0], 5.0),
            (norm_linf, "linf", &[3.0, -7.0, 5.0, -2.0], 7.0),
        ]
    }

    // =========================================================================
    // Backend dispatch — all norms x all backends in single tests
    // =========================================================================

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_all_norms_sse2_backend() {
        for (method, name, data, expected) in norm_specs() {
            assert_norm_with_backend(method, data, expected, 1e-5, Backend::SSE2);
            let _ = name;
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_all_norms_avx2_backend() {
        if !is_x86_feature_detected!("avx2") { return; }
        for (method, _name, data, expected) in norm_specs() {
            assert_norm_with_backend(method, data, expected, 1e-3, Backend::AVX2);
        }
    }

    #[test]
    fn test_all_norms_fallback_backends() {
        for (method, _name, data, expected) in norm_specs() {
            for &b in &[Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto, Backend::Scalar] {
                assert_norm_with_backend(method, data, expected, 1e-5, b);
            }
        }
    }

    // =========================================================================
    // Backend equivalence — all norms in single test
    // =========================================================================

    #[test]
    fn test_all_norms_backend_equivalence() {
        let data: Vec<f32> = (0..100).map(|i| ((i as f32) * 0.13).sin()).collect();
        for (method, _name, _, _) in norm_specs() {
            assert_norm_backend_equivalence(method, &data, 1e-3);
        }
    }

    // =========================================================================
    // Non-aligned sizes — all norms in single test
    // =========================================================================

    #[test]
    fn test_all_norms_non_aligned_sizes() {
        let norms: [(NormMethod, fn(usize) -> Vec<f32>, fn(&[f32]) -> f32); 3] = [
            (norm_l2, |sz| (0..sz).map(|i| (i as f32 + 1.0) * 0.1).collect(), |d| d.iter().map(|x| x * x).sum::<f32>().sqrt()),
            (norm_l1, |sz| (0..sz).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect(), |d| d.len() as f32),
            (norm_linf, |sz| (0..sz).map(|i| i as f32 + 1.0).collect(), |d| d.len() as f32),
        ];
        for (method, make_data, make_expected) in norms {
            assert_norm_non_aligned(method, make_data, make_expected, 1e-3);
        }
    }
}
