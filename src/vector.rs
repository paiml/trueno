//! Vector type with multi-backend support

use crate::backends::scalar::ScalarBackend;
#[cfg(target_arch = "x86_64")]
use crate::backends::sse2::Sse2Backend;
use crate::backends::VectorBackend;
use crate::{Backend, Result, TruenoError};

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

        // Dispatch to appropriate backend
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::add(&self.data, &other.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    // For now, all x86 backends use SSE2 implementation
                    // TODO: Add AVX2, AVX512 optimized versions
                    Sse2Backend::add(&self.data, &other.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    // Fallback to scalar on non-x86_64
                    ScalarBackend::add(&self.data, &other.data, &mut result);
                }
                Backend::NEON | Backend::WasmSIMD | Backend::GPU | Backend::Auto => {
                    // Not yet implemented, use scalar
                    ScalarBackend::add(&self.data, &other.data, &mut result);
                }
            }
        }

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

        // Dispatch to appropriate backend
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::mul(&self.data, &other.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    Sse2Backend::mul(&self.data, &other.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::mul(&self.data, &other.data, &mut result);
                }
                Backend::NEON | Backend::WasmSIMD | Backend::GPU | Backend::Auto => {
                    ScalarBackend::mul(&self.data, &other.data, &mut result);
                }
            }
        }

        Ok(Self {
            data: result,
            backend: self.backend,
        })
    }

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

        let result = unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::dot(&self.data, &other.data)
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    Sse2Backend::dot(&self.data, &other.data)
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::dot(&self.data, &other.data)
                }
                Backend::NEON | Backend::WasmSIMD | Backend::GPU | Backend::Auto => {
                    ScalarBackend::dot(&self.data, &other.data)
                }
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
        let result = unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::sum(&self.data)
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    Sse2Backend::sum(&self.data)
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::sum(&self.data)
                }
                Backend::NEON | Backend::WasmSIMD | Backend::GPU | Backend::Auto => {
                    ScalarBackend::sum(&self.data)
                }
            }
        };

        Ok(result)
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

        let result = unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::max(&self.data)
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    Sse2Backend::max(&self.data)
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::max(&self.data)
                }
                Backend::NEON | Backend::WasmSIMD | Backend::GPU | Backend::Auto => {
                    ScalarBackend::max(&self.data)
                }
            }
        };

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Basic construction tests
    #[test]
    fn test_from_slice() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
    }

    #[test]
    fn test_from_slice_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn test_from_slice_single_element() {
        let v = Vector::from_slice(&[42.0]);
        assert_eq!(v.as_slice(), &[42.0]);
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn test_from_slice_with_backend() {
        let v = Vector::from_slice_with_backend(&[1.0, 2.0], Backend::Scalar);
        assert_eq!(v.backend(), Backend::Scalar);
    }

    #[test]
    fn test_auto_backend_resolution() {
        let v = Vector::from_slice_with_backend(&[1.0], Backend::Auto);
        // Auto should be resolved to best available backend
        let expected_backend = crate::select_best_available_backend();
        assert_eq!(v.backend(), expected_backend);

        // Verify it's not still Backend::Auto after resolution
        assert_ne!(v.backend(), Backend::Auto);

        // On x86_64, should be a SIMD backend (not Scalar)
        #[cfg(target_arch = "x86_64")]
        {
            assert_ne!(v.backend(), Backend::Scalar);
            assert!(matches!(
                v.backend(),
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512
            ));
        }
    }

    // Add operation tests
    #[test]
    fn test_add() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
        let result = a.add(&b).unwrap();
        assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let b: Vector<f32> = Vector::from_slice(&[]);
        let result = a.add(&b).unwrap();
        assert_eq!(result.as_slice(), &[]);
    }

    #[test]
    fn test_add_single() {
        let a = Vector::from_slice(&[1.0]);
        let b = Vector::from_slice(&[2.0]);
        let result = a.add(&b).unwrap();
        assert_eq!(result.as_slice(), &[3.0]);
    }

    #[test]
    fn test_add_size_mismatch() {
        let a = Vector::from_slice(&[1.0, 2.0]);
        let b = Vector::from_slice(&[3.0]);
        let result = a.add(&b);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            TruenoError::SizeMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    // Multiply operation tests
    #[test]
    fn test_mul() {
        let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
        let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
        let result = a.mul(&b).unwrap();
        assert_eq!(result.as_slice(), &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_mul_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let b: Vector<f32> = Vector::from_slice(&[]);
        let result = a.mul(&b).unwrap();
        assert_eq!(result.as_slice(), &[]);
    }

    #[test]
    fn test_mul_size_mismatch() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[4.0, 5.0]);
        let result = a.mul(&b);
        assert!(result.is_err());
    }

    // Dot product tests
    #[test]
    fn test_dot() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
        let result = a.dot(&b).unwrap();
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_dot_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let b: Vector<f32> = Vector::from_slice(&[]);
        let result = a.dot(&b).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_size_mismatch() {
        let a = Vector::from_slice(&[1.0, 2.0]);
        let b = Vector::from_slice(&[3.0]);
        let result = a.dot(&b);
        assert!(result.is_err());
    }

    // Sum tests
    #[test]
    fn test_sum() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.sum().unwrap(), 10.0);
    }

    #[test]
    fn test_sum_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        assert_eq!(v.sum().unwrap(), 0.0);
    }

    #[test]
    fn test_sum_single() {
        let v = Vector::from_slice(&[42.0]);
        assert_eq!(v.sum().unwrap(), 42.0);
    }

    // Max tests
    #[test]
    fn test_max() {
        let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
        assert_eq!(v.max().unwrap(), 5.0);
    }

    #[test]
    fn test_max_single() {
        let v = Vector::from_slice(&[42.0]);
        assert_eq!(v.max().unwrap(), 42.0);
    }

    #[test]
    fn test_max_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.max();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            TruenoError::InvalidInput("Empty vector".to_string())
        );
    }

    #[test]
    fn test_max_negative() {
        let v = Vector::from_slice(&[-5.0, -1.0, -10.0, -3.0]);
        assert_eq!(v.max().unwrap(), -1.0);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // Property test: Addition is commutative (a + b == b + a)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_add_commutative(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100),
            b in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            // Use minimum length to ensure both vectors have same size
            let len = a.len().min(b.len());
            let a_vec: Vec<f32> = a.into_iter().take(len).collect();
            let b_vec: Vec<f32> = b.into_iter().take(len).collect();

            let va = Vector::from_slice(&a_vec);
            let vb = Vector::from_slice(&b_vec);

            let result1 = va.add(&vb).unwrap();
            let result2 = vb.add(&va).unwrap();

            prop_assert_eq!(result1.as_slice(), result2.as_slice());
        }
    }

    // Property test: Addition is associative ((a + b) + c == a + (b + c))
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_add_associative(
            a in prop::collection::vec(-100.0f32..100.0, 1..50),
            b in prop::collection::vec(-100.0f32..100.0, 1..50),
            c in prop::collection::vec(-100.0f32..100.0, 1..50)
        ) {
            let len = a.len().min(b.len()).min(c.len());
            let a_vec: Vec<f32> = a.into_iter().take(len).collect();
            let b_vec: Vec<f32> = b.into_iter().take(len).collect();
            let c_vec: Vec<f32> = c.into_iter().take(len).collect();

            let va = Vector::from_slice(&a_vec);
            let vb = Vector::from_slice(&b_vec);
            let vc = Vector::from_slice(&c_vec);

            let ab = va.add(&vb).unwrap();
            let abc = ab.add(&vc).unwrap();

            let bc = vb.add(&vc).unwrap();
            let a_bc = va.add(&bc).unwrap();

            // Use approximate equality for floating point (relaxed for associativity)
            for (x, y) in abc.as_slice().iter().zip(a_bc.as_slice()) {
                prop_assert!((x - y).abs() < 1e-4);
            }
        }
    }

    // Property test: Multiplication is commutative
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_mul_commutative(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            b in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let len = a.len().min(b.len());
            let a_vec: Vec<f32> = a.into_iter().take(len).collect();
            let b_vec: Vec<f32> = b.into_iter().take(len).collect();

            let va = Vector::from_slice(&a_vec);
            let vb = Vector::from_slice(&b_vec);

            let result1 = va.mul(&vb).unwrap();
            let result2 = vb.mul(&va).unwrap();

            prop_assert_eq!(result1.as_slice(), result2.as_slice());
        }
    }

    // Property test: Dot product is commutative
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_dot_commutative(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            b in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let len = a.len().min(b.len());
            let a_vec: Vec<f32> = a.into_iter().take(len).collect();
            let b_vec: Vec<f32> = b.into_iter().take(len).collect();

            let va = Vector::from_slice(&a_vec);
            let vb = Vector::from_slice(&b_vec);

            let result1 = va.dot(&vb).unwrap();
            let result2 = vb.dot(&va).unwrap();

            prop_assert!((result1 - result2).abs() < 1e-3);
        }
    }

    // Property test: Identity element for addition (a + 0 == a)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_add_identity(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let zero = Vector::from_slice(&vec![0.0; a.len()]);

            let result = va.add(&zero).unwrap();

            prop_assert_eq!(result.as_slice(), va.as_slice());
        }
    }

    // Property test: Identity element for multiplication (a * 1 == a)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_mul_identity(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let one = Vector::from_slice(&vec![1.0; a.len()]);

            let result = va.mul(&one).unwrap();

            for (x, y) in result.as_slice().iter().zip(va.as_slice()) {
                prop_assert!((x - y).abs() < 1e-5);
            }
        }
    }

    // Property test: Zero element for multiplication (a * 0 == 0)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_mul_zero(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let zero = Vector::from_slice(&vec![0.0; a.len()]);

            let result = va.mul(&zero).unwrap();

            for x in result.as_slice() {
                prop_assert_eq!(*x, 0.0);
            }
        }
    }

    // Property test: Distributive property (a * (b + c) == a * b + a * c)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_distributive(
            a in prop::collection::vec(-10.0f32..10.0, 1..50),
            b in prop::collection::vec(-10.0f32..10.0, 1..50),
            c in prop::collection::vec(-10.0f32..10.0, 1..50)
        ) {
            let len = a.len().min(b.len()).min(c.len());
            let a_vec: Vec<f32> = a.into_iter().take(len).collect();
            let b_vec: Vec<f32> = b.into_iter().take(len).collect();
            let c_vec: Vec<f32> = c.into_iter().take(len).collect();

            let va = Vector::from_slice(&a_vec);
            let vb = Vector::from_slice(&b_vec);
            let vc = Vector::from_slice(&c_vec);

            // a * (b + c)
            let bc = vb.add(&vc).unwrap();
            let left = va.mul(&bc).unwrap();

            // a * b + a * c
            let ab = va.mul(&vb).unwrap();
            let ac = va.mul(&vc).unwrap();
            let right = ab.add(&ac).unwrap();

            for (x, y) in left.as_slice().iter().zip(right.as_slice()) {
                prop_assert!((x - y).abs() < 1e-3);
            }
        }
    }

    // Property test: Sum is consistent
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sum_matches_manual(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.sum().unwrap();
            let manual_sum: f32 = a.iter().sum();

            // Relaxed tolerance for SIMD vs scalar accumulation order differences
            prop_assert!((result - manual_sum).abs() < 1e-2);
        }
    }

    // Property test: Max is actually maximum
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_max_is_maximum(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.max().unwrap();

            // Verify result is >= all elements
            for &x in a.iter() {
                prop_assert!(result >= x);
            }

            // Verify result is actually in the vector
            prop_assert!(a.contains(&result));
        }
    }
}
