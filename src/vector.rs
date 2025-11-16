//! Vector type with multi-backend support

use crate::backends::scalar::ScalarBackend;
#[cfg(target_arch = "x86_64")]
use crate::backends::sse2::Sse2Backend;
#[cfg(target_arch = "x86_64")]
use crate::backends::avx2::Avx2Backend;
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
use crate::backends::neon::NeonBackend;
#[cfg(target_arch = "wasm32")]
use crate::backends::wasm::WasmBackend;
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
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::add(&self.data, &other.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    // AVX2 backend (AVX-512 uses AVX2 for now)
                    Avx2Backend::add(&self.data, &other.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    // Fallback to scalar on non-x86_64
                    ScalarBackend::add(&self.data, &other.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::add(&self.data, &other.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    // Fallback to scalar on non-ARM
                    ScalarBackend::add(&self.data, &other.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::add(&self.data, &other.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    // Fallback to scalar on non-WASM
                    ScalarBackend::add(&self.data, &other.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
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

        // Dispatch to appropriate backend
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::sub(&self.data, &other.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::sub(&self.data, &other.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    // AVX2 backend (AVX-512 uses AVX2 for now)
                    Avx2Backend::sub(&self.data, &other.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    // Fallback to scalar on non-x86_64
                    ScalarBackend::sub(&self.data, &other.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::sub(&self.data, &other.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    // Fallback to scalar on non-ARM
                    ScalarBackend::sub(&self.data, &other.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::sub(&self.data, &other.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    // Fallback to scalar on non-WASM
                    ScalarBackend::sub(&self.data, &other.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
                    // Not yet implemented, use scalar
                    ScalarBackend::sub(&self.data, &other.data, &mut result);
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
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::mul(&self.data, &other.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    Avx2Backend::mul(&self.data, &other.data, &mut result);
                }
#[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::mul(&self.data, &other.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::mul(&self.data, &other.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    ScalarBackend::mul(&self.data, &other.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::mul(&self.data, &other.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    ScalarBackend::mul(&self.data, &other.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::mul(&self.data, &other.data, &mut result);
                }
            }
        }

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

        // Dispatch to appropriate backend
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::div(&self.data, &other.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::div(&self.data, &other.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    Avx2Backend::div(&self.data, &other.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::div(&self.data, &other.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::div(&self.data, &other.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    ScalarBackend::div(&self.data, &other.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::div(&self.data, &other.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    ScalarBackend::div(&self.data, &other.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::div(&self.data, &other.data, &mut result);
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
                Backend::SSE2 | Backend::AVX => Sse2Backend::dot(&self.data, &other.data),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::dot(&self.data, &other.data),
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
                Backend::GPU | Backend::Auto => {
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
                Backend::SSE2 | Backend::AVX => Sse2Backend::sum(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::sum(&self.data),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::sum(&self.data)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => NeonBackend::sum(&self.data),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::sum(&self.data),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => WasmBackend::sum(&self.data),
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::sum(&self.data),
                Backend::GPU | Backend::Auto => {
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
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::max(&self.data)
                }
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

        let result = unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::min(&self.data)
                }
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
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::min(&self.data)
                }
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

        let result = unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::argmax(&self.data)
                }
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
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::argmax(&self.data)
                }
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

        let result = unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::argmin(&self.data)
                }
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
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::argmin(&self.data)
                }
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

        let result = unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::sum_kahan(&self.data)
                }
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
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::sum_kahan(&self.data)
                }
            }
        };

        Ok(result)
    }

    /// L2 norm (Euclidean norm)
    ///
    /// Computes the Euclidean length of the vector: sqrt(sum(a[i]^2)).
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

        let result = unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::norm_l2(&self.data)
                }
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
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::norm_l2(&self.data)
                }
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

        // Divide each element by the norm
        // Create a vector filled with the norm value
        let norm_vec = Vector::from_slice(&vec![norm; self.len()]);
        self.div(&norm_vec)
    }

    /// Compute the L1 norm (Manhattan norm) of the vector
    ///
    /// Returns the sum of absolute values: ||v||₁ = sum(|v[i]|)
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

        let result = unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::norm_l1(&self.data)
                }
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
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::norm_l1(&self.data)
                }
            }
        };

        Ok(result)
    }

    /// Compute the L∞ norm (infinity norm / max norm) of the vector
    ///
    /// Returns the maximum absolute value: ||v||∞ = max(|v[i]|)
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

        // Create a vector of absolute values
        let abs_values: Vec<f32> = self.data.iter().map(|x| x.abs()).collect();

        // Find the maximum absolute value using existing max() implementation
        let max_abs = unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::max(&abs_values)
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::max(&abs_values),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::max(&abs_values),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::max(&abs_values)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => NeonBackend::max(&abs_values),
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::max(&abs_values),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => WasmBackend::max(&abs_values),
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::max(&abs_values),
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::max(&abs_values)
                }
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
        // Create a new vector with absolute values
        let abs_data: Vec<f32> = self.data.iter().map(|x| x.abs()).collect();

        Ok(Vector {
            data: abs_data,
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
                    Backend::WASM => WasmBackend::scale(&self.data, scalar, &mut result_data),
                    Backend::GPU => {
                        return Err(TruenoError::UnsupportedBackend(Backend::GPU))
                    }
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
                    Backend::WASM => {
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
    /// Computes element-wise linear interpolation: `result[i] = a[i] + t * (b[i] - a[i])`
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
                    Backend::WASM => {
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

    /// Fused multiply-add: result[i] = self[i] * b[i] + c[i]
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
    /// A new vector where each element is `self[i] * b[i] + c[i]`
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
                    Backend::WASM => {
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

    /// Element-wise square root: result[i] = sqrt(self[i])
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
        let sqrt_data: Vec<f32> = self.data.iter().map(|x| x.sqrt()).collect();
        Ok(Vector {
            data: sqrt_data,
            backend: self.backend,
        })
    }

    /// Element-wise reciprocal: result[i] = 1 / self[i]
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
        let recip_data: Vec<f32> = self.data.iter().map(|x| x.recip()).collect();
        Ok(Vector {
            data: recip_data,
            backend: self.backend,
        })
    }

    /// Element-wise power: result[i] = base[i]^n
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

    /// Element-wise exponential: result[i] = e^x[i]
    ///
    /// Computes the natural exponential (e^x) for each element.
    /// Uses Rust's optimized f32::exp() method.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, 2.0]);
    /// let result = v.exp().unwrap();
    /// // result ≈ [1.0, 2.718, 7.389]
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `exp(0.0)` returns 1.0
    /// - `exp(1.0)` returns e ≈ 2.71828
    /// - `exp(-∞)` returns 0.0
    /// - `exp(+∞)` returns +∞
    ///
    /// # Applications
    ///
    /// - Machine learning: Softmax activation, sigmoid, exponential loss
    /// - Statistics: Exponential distribution, log-normal distribution
    /// - Physics: Radioactive decay, population growth models
    /// - Signal processing: Exponential smoothing, envelope detection
    /// - Numerical methods: Solving differential equations
    pub fn exp(&self) -> Result<Vector<f32>> {
        let exp_data: Vec<f32> = self.data.iter().map(|x| x.exp()).collect();
        Ok(Vector {
            data: exp_data,
            backend: self.backend,
        })
    }

    /// Element-wise natural logarithm: result[i] = ln(x[i])
    ///
    /// Computes the natural logarithm (base e) for each element.
    /// Uses Rust's optimized f32::ln() method.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, std::f32::consts::E, std::f32::consts::E.powi(2)]);
    /// let result = v.ln().unwrap();
    /// // result ≈ [0.0, 1.0, 2.0]
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `ln(1.0)` returns 0.0
    /// - `ln(e)` returns 1.0
    /// - `ln(x)` for x ≤ 0 returns NaN
    /// - `ln(0.0)` returns -∞
    /// - `ln(+∞)` returns +∞
    ///
    /// # Applications
    ///
    /// - Machine learning: Log loss, log-likelihood, softmax normalization
    /// - Statistics: Log-normal distribution, log transformation for skewed data
    /// - Information theory: Entropy calculation, mutual information
    /// - Economics: Log returns, elasticity calculations
    /// - Signal processing: Decibel conversion, log-frequency analysis
    pub fn ln(&self) -> Result<Vector<f32>> {
        let ln_data: Vec<f32> = self.data.iter().map(|x| x.ln()).collect();
        Ok(Vector {
            data: ln_data,
            backend: self.backend,
        })
    }

    /// Element-wise sine: result[i] = sin(x[i])
    ///
    /// Computes the sine for each element (input in radians).
    /// Uses Rust's optimized f32::sin() method.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let v = Vector::from_slice(&[0.0, PI / 2.0, PI]);
    /// let result = v.sin().unwrap();
    /// // result ≈ [0.0, 1.0, 0.0]
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `sin(0)` returns 0.0
    /// - `sin(π/2)` returns 1.0
    /// - `sin(π)` returns 0.0 (approximately)
    /// - `sin(-x)` returns -sin(x) (odd function)
    /// - Periodic with period 2π: sin(x + 2π) = sin(x)
    ///
    /// # Applications
    ///
    /// - Signal processing: Waveform generation, oscillators, modulation
    /// - Physics: Harmonic motion, wave propagation, pendulums
    /// - Audio: Synthesizers, tone generation, effects processing
    /// - Graphics: Animation, rotation transformations, procedural generation
    /// - Fourier analysis: Frequency decomposition, spectral analysis
    pub fn sin(&self) -> Result<Vector<f32>> {
        let sin_data: Vec<f32> = self.data.iter().map(|x| x.sin()).collect();
        Ok(Vector {
            data: sin_data,
            backend: self.backend,
        })
    }

    /// Element-wise cosine: result[i] = cos(x[i])
    ///
    /// Computes the cosine for each element (input in radians).
    /// Uses Rust's optimized f32::cos() method.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let v = Vector::from_slice(&[0.0, PI / 2.0, PI]);
    /// let result = v.cos().unwrap();
    /// // result ≈ [1.0, 0.0, -1.0]
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `cos(0)` returns 1.0
    /// - `cos(π/2)` returns 0.0 (approximately)
    /// - `cos(π)` returns -1.0
    /// - `cos(-x)` returns cos(x) (even function)
    /// - Periodic with period 2π: cos(x + 2π) = cos(x)
    /// - Relation to sine: cos(x) = sin(x + π/2)
    ///
    /// # Applications
    ///
    /// - Signal processing: Phase-shifted waveforms, I/Q modulation, quadrature signals
    /// - Physics: Projectile motion, wave interference, damped oscillations
    /// - Graphics: Rotation matrices, camera transforms, circular motion
    /// - Audio: Stereo panning, spatial audio, frequency synthesis
    /// - Engineering: Control systems, frequency response, AC circuits
    pub fn cos(&self) -> Result<Vector<f32>> {
        let cos_data: Vec<f32> = self.data.iter().map(|x| x.cos()).collect();
        Ok(Vector {
            data: cos_data,
            backend: self.backend,
        })
    }

    /// Computes element-wise tangent (tan) of the vector.
    ///
    /// Returns a new vector where each element is the tangent of the corresponding input element.
    /// tan(x) = sin(x) / cos(x)
    ///
    /// # Returns
    /// - `Ok(Vector<f32>)`: New vector with tan(x) for each element
    ///
    /// # Properties
    /// - Odd function: tan(-x) = -tan(x)
    /// - Period: 2π (not π, despite common misconception)
    /// - Undefined at x = π/2 + nπ (where n is any integer)
    /// - tan(x) = sin(x) / cos(x)
    /// - Range: (-∞, +∞)
    ///
    /// # Performance
    /// - Iterator map pattern for cache efficiency
    /// - Leverages Rust's optimized f32::tan()
    /// - Auto-vectorized by LLVM on supporting platforms
    ///
    /// # Examples
    /// ```
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let angles = Vector::from_slice(&[0.0, PI / 4.0, -PI / 4.0]);
    /// let result = angles.tan().unwrap();
    /// // Result: [0.0, 1.0, -1.0] (approximately)
    /// ```
    ///
    /// # Use Cases
    /// - Trigonometry: Slope calculations, angle relationships
    /// - Signal processing: Phase analysis, modulation
    /// - Physics: Projectile trajectories, optics (Snell's law angles)
    /// - Graphics: Perspective projection, field of view calculations
    /// - Engineering: Slope gradients, tangent lines to curves
    pub fn tan(&self) -> Result<Vector<f32>> {
        let tan_data: Vec<f32> = self.data.iter().map(|x| x.tan()).collect();
        Ok(Vector {
            data: tan_data,
            backend: self.backend,
        })
    }

    /// Computes element-wise arcsine (asin/sin⁻¹) of the vector.
    ///
    /// Returns a new vector where each element is the inverse sine of the corresponding input element.
    /// This is the inverse function of sin: if y = sin(x), then x = asin(y).
    ///
    /// # Returns
    /// - `Ok(Vector<f32>)`: New vector with asin(x) for each element
    ///
    /// # Properties
    /// - Domain: [-1, 1] (inputs outside this range produce NaN)
    /// - Range: [-π/2, π/2]
    /// - Odd function: asin(-x) = -asin(x)
    /// - Inverse relation: asin(sin(x)) = x for x ∈ [-π/2, π/2]
    /// - asin(0) = 0
    /// - asin(1) = π/2
    /// - asin(-1) = -π/2
    ///
    /// # Performance
    /// - Iterator map pattern for cache efficiency
    /// - Leverages Rust's optimized f32::asin()
    /// - Auto-vectorized by LLVM on supporting platforms
    ///
    /// # Examples
    /// ```
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let values = Vector::from_slice(&[0.0, 0.5, 1.0]);
    /// let result = values.asin().unwrap();
    /// // Result: [0.0, π/6, π/2] (approximately)
    /// ```
    ///
    /// # Use Cases
    /// - Physics: Calculating angles from sine values in mechanics, optics
    /// - Signal processing: Phase recovery, demodulation
    /// - Graphics: Inverse transformations, angle calculations
    /// - Navigation: GPS calculations, spherical trigonometry
    /// - Control systems: Inverse kinematics, servo positioning
    pub fn asin(&self) -> Result<Vector<f32>> {
        let asin_data: Vec<f32> = self.data.iter().map(|x| x.asin()).collect();
        Ok(Vector {
            data: asin_data,
            backend: self.backend,
        })
    }

    /// Computes element-wise arccosine (acos/cos⁻¹) of the vector.
    ///
    /// Returns a new vector where each element is the inverse cosine of the corresponding input element.
    /// This is the inverse function of cos: if y = cos(x), then x = acos(y).
    ///
    /// # Returns
    /// - `Ok(Vector<f32>)`: New vector with acos(x) for each element
    ///
    /// # Properties
    /// - Domain: [-1, 1] (inputs outside this range produce NaN)
    /// - Range: [0, π]
    /// - Symmetry: acos(-x) = π - acos(x)
    /// - Inverse relation: acos(cos(x)) = x for x ∈ [0, π]
    /// - acos(0) = π/2
    /// - acos(1) = 0
    /// - acos(-1) = π
    ///
    /// # Performance
    /// - Iterator map pattern for cache efficiency
    /// - Leverages Rust's optimized f32::acos()
    /// - Auto-vectorized by LLVM on supporting platforms
    ///
    /// # Examples
    /// ```
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let values = Vector::from_slice(&[0.0, 0.5, 1.0]);
    /// let result = values.acos().unwrap();
    /// // Result: [π/2, π/3, 0.0] (approximately)
    /// ```
    ///
    /// # Use Cases
    /// - Physics: Angle calculations in mechanics, optics, reflections
    /// - Signal processing: Phase analysis, correlation functions
    /// - Graphics: View angle calculations, lighting models
    /// - Navigation: Bearing calculations, great circle distances
    /// - Robotics: Joint angle solving, orientation calculations
    pub fn acos(&self) -> Result<Vector<f32>> {
        let acos_data: Vec<f32> = self.data.iter().map(|x| x.acos()).collect();
        Ok(Vector {
            data: acos_data,
            backend: self.backend,
        })
    }

    /// Computes element-wise arctangent (atan/tan⁻¹) of the vector.
    ///
    /// Returns a new vector where each element is the inverse tangent of the corresponding input element.
    /// This is the inverse function of tan: if y = tan(x), then x = atan(y).
    ///
    /// # Returns
    /// - `Ok(Vector<f32>)`: New vector with atan(x) for each element
    ///
    /// # Properties
    /// - Domain: All real numbers (-∞, +∞)
    /// - Range: (-π/2, π/2)
    /// - Odd function: atan(-x) = -atan(x)
    /// - Inverse relation: atan(tan(x)) = x for x ∈ (-π/2, π/2)
    /// - atan(0) = 0
    /// - atan(1) = π/4
    /// - atan(-1) = -π/4
    /// - lim(x→∞) atan(x) = π/2
    /// - lim(x→-∞) atan(x) = -π/2
    ///
    /// # Performance
    /// - Iterator map pattern for cache efficiency
    /// - Leverages Rust's optimized f32::atan()
    /// - Auto-vectorized by LLVM on supporting platforms
    ///
    /// # Examples
    /// ```
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let values = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = values.atan().unwrap();
    /// // Result: [0.0, π/4, -π/4] (approximately)
    /// ```
    ///
    /// # Use Cases
    /// - Physics: Angle calculations from slopes, velocity components
    /// - Signal processing: Phase unwrapping, FM demodulation
    /// - Graphics: Rotation calculations, camera orientation
    /// - Robotics: Inverse kinematics, steering angles
    /// - Navigation: Heading calculations from coordinates
    pub fn atan(&self) -> Result<Vector<f32>> {
        let atan_data: Vec<f32> = self.data.iter().map(|x| x.atan()).collect();
        Ok(Vector {
            data: atan_data,
            backend: self.backend,
        })
    }

    /// Computes the hyperbolic sine (sinh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// sinh(x) = (e^x - e^(-x)) / 2
    ///
    /// # Properties
    ///
    /// - Domain: (-∞, +∞)
    /// - Range: (-∞, +∞)
    /// - Odd function: sinh(-x) = -sinh(x)
    /// - sinh(0) = 0
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = v.sinh().unwrap();
    /// assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    /// ```
    pub fn sinh(&self) -> Result<Vector<f32>> {
        let sinh_data: Vec<f32> = self.data.iter().map(|x| x.sinh()).collect();
        Ok(Vector {
            data: sinh_data,
            backend: self.backend,
        })
    }

    /// Computes the hyperbolic cosine (cosh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// cosh(x) = (e^x + e^(-x)) / 2
    ///
    /// # Properties
    ///
    /// - Domain: (-∞, +∞)
    /// - Range: [1, +∞)
    /// - Even function: cosh(-x) = cosh(x)
    /// - cosh(0) = 1
    /// - Always positive: cosh(x) ≥ 1 for all x
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = v.cosh().unwrap();
    /// assert!((result.as_slice()[0] - 1.0).abs() < 1e-5);
    /// ```
    pub fn cosh(&self) -> Result<Vector<f32>> {
        let cosh_data: Vec<f32> = self.data.iter().map(|x| x.cosh()).collect();
        Ok(Vector {
            data: cosh_data,
            backend: self.backend,
        })
    }

    /// Computes the hyperbolic tangent (tanh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// tanh(x) = sinh(x) / cosh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    ///
    /// # Properties
    ///
    /// - Domain: (-∞, +∞)
    /// - Range: (-1, 1)
    /// - Odd function: tanh(-x) = -tanh(x)
    /// - tanh(0) = 0
    /// - Bounded: -1 < tanh(x) < 1 for all x
    /// - Commonly used as activation function in neural networks
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = v.tanh().unwrap();
    /// assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    /// // All values are in range (-1, 1)
    /// assert!(result.as_slice().iter().all(|&x| x > -1.0 && x < 1.0));
    /// ```
    pub fn tanh(&self) -> Result<Vector<f32>> {
        let tanh_data: Vec<f32> = self.data.iter().map(|x| x.tanh()).collect();
        Ok(Vector {
            data: tanh_data,
            backend: self.backend,
        })
    }

    /// Computes the inverse hyperbolic sine (asinh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// asinh(x) = ln(x + sqrt(x² + 1))
    ///
    /// # Properties
    ///
    /// - Domain: (-∞, +∞)
    /// - Range: (-∞, +∞)
    /// - Odd function: asinh(-x) = -asinh(x)
    /// - asinh(0) = 0
    /// - Inverse of sinh: asinh(sinh(x)) = x
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = v.asinh().unwrap();
    /// assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    /// ```
    pub fn asinh(&self) -> Result<Vector<f32>> {
        let asinh_data: Vec<f32> = self.data.iter().map(|x| x.asinh()).collect();
        Ok(Vector {
            data: asinh_data,
            backend: self.backend,
        })
    }

    /// Computes the inverse hyperbolic cosine (acosh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// acosh(x) = ln(x + sqrt(x² - 1))
    ///
    /// # Properties
    ///
    /// - Domain: [1, +∞)
    /// - Range: [0, +∞)
    /// - acosh(1) = 0
    /// - Inverse of cosh: acosh(cosh(x)) = x for x >= 0
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let result = v.acosh().unwrap();
    /// assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    /// ```
    pub fn acosh(&self) -> Result<Vector<f32>> {
        let acosh_data: Vec<f32> = self.data.iter().map(|x| x.acosh()).collect();
        Ok(Vector {
            data: acosh_data,
            backend: self.backend,
        })
    }

    /// Computes the inverse hyperbolic tangent (atanh) of each element.
    ///
    /// Domain: (-1, 1)
    /// Range: (-∞, +∞)
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 0.5, -0.5]);
    /// let result = v.atanh().unwrap();
    /// // atanh(0) = 0, atanh(0.5) ≈ 0.549, atanh(-0.5) ≈ -0.549
    /// ```
    pub fn atanh(&self) -> Result<Vector<f32>> {
        let atanh_data: Vec<f32> = self.data.iter().map(|x| x.atanh()).collect();
        Ok(Vector {
            data: atanh_data,
            backend: self.backend,
        })
    }

    /// Computes the floor (round down to nearest integer) of each element.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.7, -2.3, 5.0]);
    /// let result = v.floor().unwrap();
    /// assert_eq!(result.as_slice(), &[3.0, -3.0, 5.0]);
    /// ```
    pub fn floor(&self) -> Result<Vector<f32>> {
        let floor_data: Vec<f32> = self.data.iter().map(|x| x.floor()).collect();
        Ok(Vector {
            data: floor_data,
            backend: self.backend,
        })
    }

    /// Computes the ceiling (round up to nearest integer) of each element.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.2, -2.7, 5.0]);
    /// let result = v.ceil().unwrap();
    /// assert_eq!(result.as_slice(), &[4.0, -2.0, 5.0]);
    /// ```
    pub fn ceil(&self) -> Result<Vector<f32>> {
        let ceil_data: Vec<f32> = self.data.iter().map(|x| x.ceil()).collect();
        Ok(Vector {
            data: ceil_data,
            backend: self.backend,
        })
    }

    /// Rounds each element to the nearest integer.
    ///
    /// Uses "round half away from zero" strategy:
    /// - 0.5 rounds to 1.0, 1.5 rounds to 2.0, -1.5 rounds to -2.0, etc.
    /// - Positive halfway cases round up, negative halfway cases round down.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.2, 3.7, -2.3, -2.8]);
    /// let result = v.round().unwrap();
    /// assert_eq!(result.as_slice(), &[3.0, 4.0, -2.0, -3.0]);
    /// ```
    pub fn round(&self) -> Result<Vector<f32>> {
        let round_data: Vec<f32> = self.data.iter().map(|x| x.round()).collect();
        Ok(Vector {
            data: round_data,
            backend: self.backend,
        })
    }

    /// Truncates each element toward zero (removes fractional part).
    ///
    /// Truncation always moves toward zero:
    /// - Positive values: equivalent to floor() (e.g., 3.7 → 3.0)
    /// - Negative values: equivalent to ceil() (e.g., -3.7 → -3.0)
    /// - This differs from floor() which always rounds down
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.7, -2.7, 5.0]);
    /// let result = v.trunc().unwrap();
    /// assert_eq!(result.as_slice(), &[3.0, -2.0, 5.0]);
    /// ```
    pub fn trunc(&self) -> Result<Vector<f32>> {
        let trunc_data: Vec<f32> = self.data.iter().map(|x| x.trunc()).collect();
        Ok(Vector {
            data: trunc_data,
            backend: self.backend,
        })
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

    // Subtract operation tests
    #[test]
    fn test_sub() {
        let a = Vector::from_slice(&[5.0, 7.0, 9.0]);
        let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = a.sub(&b).unwrap();
        assert_eq!(result.as_slice(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sub_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let b: Vector<f32> = Vector::from_slice(&[]);
        let result = a.sub(&b).unwrap();
        assert_eq!(result.as_slice(), &[]);
    }

    #[test]
    fn test_sub_single() {
        let a = Vector::from_slice(&[5.0]);
        let b = Vector::from_slice(&[2.0]);
        let result = a.sub(&b).unwrap();
        assert_eq!(result.as_slice(), &[3.0]);
    }

    #[test]
    fn test_sub_size_mismatch() {
        let a = Vector::from_slice(&[1.0, 2.0]);
        let b = Vector::from_slice(&[3.0]);
        let result = a.sub(&b);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            TruenoError::SizeMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn test_sub_negative_result() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
        let result = a.sub(&b).unwrap();
        assert_eq!(result.as_slice(), &[-4.0, -4.0, -4.0]);
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

    // Division operation tests
    #[test]
    fn test_div() {
        let a = Vector::from_slice(&[10.0, 20.0, 30.0]);
        let b = Vector::from_slice(&[2.0, 4.0, 5.0]);
        let result = a.div(&b).unwrap();
        assert_eq!(result.as_slice(), &[5.0, 5.0, 6.0]);
    }

    #[test]
    fn test_div_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let b: Vector<f32> = Vector::from_slice(&[]);
        let result = a.div(&b).unwrap();
        assert_eq!(result.as_slice(), &[]);
    }

    #[test]
    fn test_div_single() {
        let a = Vector::from_slice(&[10.0]);
        let b = Vector::from_slice(&[2.0]);
        let result = a.div(&b).unwrap();
        assert_eq!(result.as_slice(), &[5.0]);
    }

    #[test]
    fn test_div_size_mismatch() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[4.0, 5.0]);
        let result = a.div(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_div_by_one() {
        let a = Vector::from_slice(&[5.0, 10.0, 15.0]);
        let b = Vector::from_slice(&[1.0, 1.0, 1.0]);
        let result = a.div(&b).unwrap();
        assert_eq!(result.as_slice(), &[5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_div_fractional() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[2.0, 4.0, 8.0]);
        let result = a.div(&b).unwrap();
        assert_eq!(result.as_slice(), &[0.5, 0.5, 0.375]);
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

    // Kahan summation tests (numerically stable)
    #[test]
    fn test_sum_kahan() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v.sum_kahan().unwrap(), 10.0);
    }

    #[test]
    fn test_sum_kahan_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        assert_eq!(v.sum_kahan().unwrap(), 0.0);
    }

    #[test]
    fn test_sum_kahan_single() {
        let v = Vector::from_slice(&[42.0]);
        assert_eq!(v.sum_kahan().unwrap(), 42.0);
    }

    #[test]
    fn test_sum_kahan_numerical_stability() {
        // Test case that demonstrates rounding error accumulation
        // Using many small values that can lose precision
        let mut data = vec![1e-7f32; 10_000];
        data.push(1.0);

        let v = Vector::from_slice(&data);
        let kahan_result = v.sum_kahan().unwrap();
        let naive_result = v.sum().unwrap();

        // Expected: 1.0 + 10000 * 1e-7 = 1.001
        let expected = 1.001f32;

        // Kahan should be more accurate than naive sum
        let kahan_error = (kahan_result - expected).abs();
        let naive_error = (naive_result - expected).abs();

        // Kahan error should be smaller (or at most equal)
        assert!(kahan_error <= naive_error,
            "Kahan sum error ({}) should be <= naive sum error ({})",
            kahan_error, naive_error);
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

    #[test]
    fn test_min() {
        let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
        assert_eq!(v.min().unwrap(), 1.0);
    }

    #[test]
    fn test_min_single() {
        let v = Vector::from_slice(&[42.0]);
        assert_eq!(v.min().unwrap(), 42.0);
    }

    #[test]
    fn test_min_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.min();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            TruenoError::InvalidInput("Empty vector".to_string())
        );
    }

    #[test]
    fn test_min_negative() {
        let v = Vector::from_slice(&[-5.0, -1.0, -10.0, -3.0]);
        assert_eq!(v.min().unwrap(), -10.0);
    }

    #[test]
    fn test_argmax() {
        let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
        assert_eq!(v.argmax().unwrap(), 1); // max value 5.0 is at index 1
    }

    #[test]
    fn test_argmax_single() {
        let v = Vector::from_slice(&[42.0]);
        assert_eq!(v.argmax().unwrap(), 0);
    }

    #[test]
    fn test_argmax_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.argmax();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            TruenoError::InvalidInput("Empty vector".to_string())
        );
    }

    #[test]
    fn test_argmax_negative() {
        let v = Vector::from_slice(&[-5.0, -1.0, -10.0, -3.0]);
        assert_eq!(v.argmax().unwrap(), 1); // max value -1.0 is at index 1
    }

    #[test]
    fn test_argmax_first_occurrence() {
        // When there are duplicates, should return first occurrence
        let v = Vector::from_slice(&[1.0, 5.0, 3.0, 5.0, 2.0]);
        assert_eq!(v.argmax().unwrap(), 1); // first 5.0 is at index 1
    }

    #[test]
    fn test_argmin() {
        let v = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
        assert_eq!(v.argmin().unwrap(), 0); // min value 1.0 is at index 0
    }

    #[test]
    fn test_argmin_single() {
        let v = Vector::from_slice(&[42.0]);
        assert_eq!(v.argmin().unwrap(), 0);
    }

    #[test]
    fn test_argmin_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.argmin();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            TruenoError::InvalidInput("Empty vector".to_string())
        );
    }

    #[test]
    fn test_argmin_negative() {
        let v = Vector::from_slice(&[-5.0, -1.0, -10.0, -3.0]);
        assert_eq!(v.argmin().unwrap(), 2); // min value -10.0 is at index 2
    }

    #[test]
    fn test_argmin_first_occurrence() {
        // When there are duplicates, should return first occurrence
        let v = Vector::from_slice(&[5.0, 1.0, 3.0, 1.0, 2.0]);
        assert_eq!(v.argmin().unwrap(), 1); // first 1.0 is at index 1
    }

    // L2 norm (Euclidean norm) tests
    #[test]
    fn test_norm_l2() {
        let v = Vector::from_slice(&[3.0, 4.0]);
        let result = v.norm_l2().unwrap();
        assert!((result - 5.0).abs() < 1e-5); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_norm_l2_single() {
        let v = Vector::from_slice(&[7.0]);
        let result = v.norm_l2().unwrap();
        assert!((result - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l2_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.norm_l2().unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_norm_l2_unit_vector() {
        let v = Vector::from_slice(&[1.0, 0.0, 0.0]);
        let result = v.norm_l2().unwrap();
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l2_negative() {
        let v = Vector::from_slice(&[-3.0, -4.0]);
        let result = v.norm_l2().unwrap();
        assert!((result - 5.0).abs() < 1e-5); // sqrt((-3)^2 + (-4)^2) = 5
    }

    // Normalize (unit vector) tests
    #[test]
    fn test_normalize() {
        let v = Vector::from_slice(&[3.0, 4.0]);
        let result = v.normalize().unwrap();
        // Should be [0.6, 0.8] (3/5, 4/5)
        assert!((result.as_slice()[0] - 0.6).abs() < 1e-5);
        assert!((result.as_slice()[1] - 0.8).abs() < 1e-5);
        // Verify it's a unit vector
        let norm = result.norm_l2().unwrap();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_already_unit() {
        let v = Vector::from_slice(&[1.0, 0.0, 0.0]);
        let result = v.normalize().unwrap();
        assert!((result.as_slice()[0] - 1.0).abs() < 1e-5);
        assert!((result.as_slice()[1] - 0.0).abs() < 1e-5);
        assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = v.normalize();
        // Should error on zero vector (division by zero norm)
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_negative() {
        let v = Vector::from_slice(&[-3.0, -4.0]);
        let result = v.normalize().unwrap();
        // Should be [-0.6, -0.8]
        assert!((result.as_slice()[0] - (-0.6)).abs() < 1e-5);
        assert!((result.as_slice()[1] - (-0.8)).abs() < 1e-5);
        let norm = result.norm_l2().unwrap();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    // L1 Norm (Manhattan norm) tests
    #[test]
    fn test_norm_l1_basic() {
        let v = Vector::from_slice(&[3.0, -4.0, 5.0]);
        let result = v.norm_l1().unwrap();
        // |3| + |-4| + |5| = 3 + 4 + 5 = 12
        assert!((result - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l1_all_positive() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let result = v.norm_l1().unwrap();
        // 1 + 2 + 3 + 4 = 10
        assert!((result - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l1_all_negative() {
        let v = Vector::from_slice(&[-1.0, -2.0, -3.0]);
        let result = v.norm_l1().unwrap();
        // |-1| + |-2| + |-3| = 1 + 2 + 3 = 6
        assert!((result - 6.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_l1_zero_vector() {
        let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = v.norm_l1().unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_norm_l1_empty_vector() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.norm_l1().unwrap();
        assert_eq!(result, 0.0);
    }

    // L∞ Norm (infinity/max norm) tests
    #[test]
    fn test_norm_linf_basic() {
        let v = Vector::from_slice(&[3.0, -7.0, 5.0, -2.0]);
        let result = v.norm_linf().unwrap();
        // max(|3|, |-7|, |5|, |-2|) = max(3, 7, 5, 2) = 7
        assert!((result - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_linf_all_positive() {
        let v = Vector::from_slice(&[1.0, 2.0, 5.0, 3.0]);
        let result = v.norm_linf().unwrap();
        // max(1, 2, 5, 3) = 5
        assert!((result - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_linf_all_negative() {
        let v = Vector::from_slice(&[-1.0, -9.0, -3.0]);
        let result = v.norm_linf().unwrap();
        // max(|-1|, |-9|, |-3|) = max(1, 9, 3) = 9
        assert!((result - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_norm_linf_zero_vector() {
        let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = v.norm_linf().unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_norm_linf_empty_vector() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.norm_linf().unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_norm_linf_single_element() {
        let v = Vector::from_slice(&[-42.5]);
        let result = v.norm_linf().unwrap();
        assert!((result - 42.5).abs() < 1e-5);
    }

    // Absolute value tests
    #[test]
    fn test_abs_mixed() {
        let v = Vector::from_slice(&[3.0, -4.0, 5.0, -2.0]);
        let result = v.abs().unwrap();
        assert_eq!(result.as_slice(), &[3.0, 4.0, 5.0, 2.0]);
    }

    #[test]
    fn test_abs_all_positive() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = v.abs().unwrap();
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_abs_all_negative() {
        let v = Vector::from_slice(&[-1.0, -2.0, -3.0]);
        let result = v.abs().unwrap();
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_abs_with_zeros() {
        let v = Vector::from_slice(&[0.0, -5.0, 0.0, 3.0]);
        let result = v.abs().unwrap();
        assert_eq!(result.as_slice(), &[0.0, 5.0, 0.0, 3.0]);
    }

    #[test]
    fn test_abs_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.abs().unwrap();
        assert_eq!(result.len(), 0);
    }

    // Scalar multiplication (scale) tests
    #[test]
    fn test_scale_basic() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let result = v.scale(2.0).unwrap();
        assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_scale_by_zero() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = v.scale(0.0).unwrap();
        assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_scale_by_negative() {
        let v = Vector::from_slice(&[1.0, -2.0, 3.0]);
        let result = v.scale(-2.0).unwrap();
        assert_eq!(result.as_slice(), &[-2.0, 4.0, -6.0]);
    }

    #[test]
    fn test_scale_by_one() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = v.scale(1.0).unwrap();
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_scale_by_fraction() {
        let v = Vector::from_slice(&[2.0, 4.0, 6.0]);
        let result = v.scale(0.5).unwrap();
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_scale_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.scale(2.0).unwrap();
        assert_eq!(result.len(), 0);
    }

    // Clamp tests
    #[test]
    fn test_clamp_basic() {
        let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
        let result = v.clamp(0.0, 10.0).unwrap();
        assert_eq!(result.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
    }

    #[test]
    fn test_clamp_all_within_range() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = v.clamp(0.0, 10.0).unwrap();
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_clamp_all_below_min() {
        let v = Vector::from_slice(&[-5.0, -3.0, -1.0]);
        let result = v.clamp(0.0, 10.0).unwrap();
        assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_clamp_all_above_max() {
        let v = Vector::from_slice(&[15.0, 20.0, 25.0]);
        let result = v.clamp(0.0, 10.0).unwrap();
        assert_eq!(result.as_slice(), &[10.0, 10.0, 10.0]);
    }

    #[test]
    fn test_clamp_negative_range() {
        let v = Vector::from_slice(&[-10.0, -5.0, 0.0, 5.0]);
        let result = v.clamp(-8.0, -2.0).unwrap();
        assert_eq!(result.as_slice(), &[-8.0, -5.0, -2.0, -2.0]);
    }

    #[test]
    fn test_clamp_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        let result = v.clamp(0.0, 10.0).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_clamp_same_min_max() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = v.clamp(2.5, 2.5).unwrap();
        assert_eq!(result.as_slice(), &[2.5, 2.5, 2.5]);
    }

    #[test]
    fn test_clamp_invalid_range() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = v.clamp(10.0, 0.0); // min > max
        assert!(result.is_err());
    }

    // Linear interpolation (lerp) tests
    #[test]
    fn test_lerp_basic() {
        let a = Vector::from_slice(&[0.0, 10.0, 20.0]);
        let b = Vector::from_slice(&[100.0, 110.0, 120.0]);
        let result = a.lerp(&b, 0.5).unwrap();
        assert_eq!(result.as_slice(), &[50.0, 60.0, 70.0]);
    }

    #[test]
    fn test_lerp_at_zero() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
        let result = a.lerp(&b, 0.0).unwrap();
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]); // Should return a
    }

    #[test]
    fn test_lerp_at_one() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
        let result = a.lerp(&b, 1.0).unwrap();
        assert_eq!(result.as_slice(), &[4.0, 5.0, 6.0]); // Should return b
    }

    #[test]
    fn test_lerp_extrapolate_above() {
        let a = Vector::from_slice(&[0.0, 10.0]);
        let b = Vector::from_slice(&[10.0, 20.0]);
        let result = a.lerp(&b, 2.0).unwrap();
        assert_eq!(result.as_slice(), &[20.0, 30.0]); // Extrapolation beyond b
    }

    #[test]
    fn test_lerp_extrapolate_below() {
        let a = Vector::from_slice(&[10.0, 20.0]);
        let b = Vector::from_slice(&[20.0, 30.0]);
        let result = a.lerp(&b, -1.0).unwrap();
        assert_eq!(result.as_slice(), &[0.0, 10.0]); // Extrapolation before a
    }

    #[test]
    fn test_lerp_size_mismatch() {
        let a = Vector::from_slice(&[1.0, 2.0]);
        let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = a.lerp(&b, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_lerp_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let b: Vector<f32> = Vector::from_slice(&[]);
        let result = a.lerp(&b, 0.5).unwrap();
        assert_eq!(result.len(), 0);
    }

    // fma() operation tests (fused multiply-add: a * b + c)
    #[test]
    fn test_fma_basic() {
        let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
        let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
        let c = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = a.fma(&b, &c).unwrap();
        // Expected: [2*5+1, 3*6+2, 4*7+3] = [11, 20, 31]
        assert_eq!(result.as_slice(), &[11.0, 20.0, 31.0]);
    }

    #[test]
    fn test_fma_zeros() {
        let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let b = Vector::from_slice(&[5.0, 6.0, 7.0]);
        let c = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = a.fma(&b, &c).unwrap();
        // Expected: [0*5+1, 0*6+2, 0*7+3] = [1, 2, 3]
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_fma_ones() {
        let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
        let b = Vector::from_slice(&[1.0, 1.0, 1.0]);
        let c = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = a.fma(&b, &c).unwrap();
        // Expected: [2*1+0, 3*1+0, 4*1+0] = [2, 3, 4]
        assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_fma_negatives() {
        let a = Vector::from_slice(&[-2.0, 3.0, -4.0]);
        let b = Vector::from_slice(&[5.0, -6.0, 7.0]);
        let c = Vector::from_slice(&[1.0, 2.0, -3.0]);
        let result = a.fma(&b, &c).unwrap();
        // Expected: [-2*5+1, 3*(-6)+2, -4*7+(-3)] = [-9, -16, -31]
        assert_eq!(result.as_slice(), &[-9.0, -16.0, -31.0]);
    }

    #[test]
    fn test_fma_size_mismatch_b() {
        let a = Vector::from_slice(&[1.0, 2.0]);
        let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let c = Vector::from_slice(&[1.0, 2.0]);
        let result = a.fma(&b, &c);
        assert!(result.is_err());
    }

    #[test]
    fn test_fma_size_mismatch_c() {
        let a = Vector::from_slice(&[1.0, 2.0]);
        let b = Vector::from_slice(&[1.0, 2.0]);
        let c = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = a.fma(&b, &c);
        assert!(result.is_err());
    }

    #[test]
    fn test_fma_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let b: Vector<f32> = Vector::from_slice(&[]);
        let c: Vector<f32> = Vector::from_slice(&[]);
        let result = a.fma(&b, &c).unwrap();
        assert_eq!(result.len(), 0);
    }

    // sqrt() operation tests (element-wise square root)
    #[test]
    fn test_sqrt_basic() {
        let a = Vector::from_slice(&[4.0, 9.0, 16.0, 25.0]);
        let result = a.sqrt().unwrap();
        assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sqrt_zeros() {
        let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = a.sqrt().unwrap();
        assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sqrt_one() {
        let a = Vector::from_slice(&[1.0, 1.0, 1.0]);
        let result = a.sqrt().unwrap();
        assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sqrt_fractional() {
        let a = Vector::from_slice(&[0.25, 0.01, 0.0625]);
        let result = a.sqrt().unwrap();
        assert_eq!(result.as_slice(), &[0.5, 0.1, 0.25]);
    }

    #[test]
    fn test_sqrt_negative() {
        let a = Vector::from_slice(&[-1.0, 4.0, -9.0]);
        let result = a.sqrt().unwrap();
        // Negative values produce NaN
        assert!(result.as_slice()[0].is_nan());
        assert_eq!(result.as_slice()[1], 4.0_f32.sqrt());
        assert!(result.as_slice()[2].is_nan());
    }

    #[test]
    fn test_sqrt_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.sqrt().unwrap();
        assert_eq!(result.len(), 0);
    }

    // recip() operation tests (element-wise reciprocal: 1/x)
    #[test]
    fn test_recip_basic() {
        let a = Vector::from_slice(&[2.0, 4.0, 5.0, 10.0]);
        let result = a.recip().unwrap();
        assert_eq!(result.as_slice(), &[0.5, 0.25, 0.2, 0.1]);
    }

    #[test]
    fn test_recip_ones() {
        let a = Vector::from_slice(&[1.0, 1.0, 1.0]);
        let result = a.recip().unwrap();
        assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_recip_negatives() {
        let a = Vector::from_slice(&[-2.0, -4.0, -0.5]);
        let result = a.recip().unwrap();
        assert_eq!(result.as_slice(), &[-0.5, -0.25, -2.0]);
    }

    #[test]
    fn test_recip_fractional() {
        let a = Vector::from_slice(&[0.5, 0.25, 0.1]);
        let result = a.recip().unwrap();
        assert_eq!(result.as_slice(), &[2.0, 4.0, 10.0]);
    }

    #[test]
    fn test_recip_zero() {
        let a = Vector::from_slice(&[0.0, 2.0, 0.0]);
        let result = a.recip().unwrap();
        // 1/0 = infinity
        assert!(result.as_slice()[0].is_infinite());
        assert_eq!(result.as_slice()[1], 0.5);
        assert!(result.as_slice()[2].is_infinite());
    }

    #[test]
    fn test_recip_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.recip().unwrap();
        assert_eq!(result.len(), 0);
    }

    // pow() operation tests (element-wise power: x^n)
    #[test]
    fn test_pow_basic() {
        let a = Vector::from_slice(&[2.0, 3.0, 4.0, 5.0]);
        let result = a.pow(2.0).unwrap();
        assert_eq!(result.as_slice(), &[4.0, 9.0, 16.0, 25.0]);
    }

    #[test]
    fn test_pow_cube() {
        let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
        let result = a.pow(3.0).unwrap();
        assert_eq!(result.as_slice(), &[8.0, 27.0, 64.0]);
    }

    #[test]
    fn test_pow_fractional() {
        let a = Vector::from_slice(&[4.0, 9.0, 16.0]);
        let result = a.pow(0.5).unwrap();
        assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0]); // Square root
    }

    #[test]
    fn test_pow_zero_exponent() {
        let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
        let result = a.pow(0.0).unwrap();
        assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0]); // x^0 = 1
    }

    #[test]
    fn test_pow_one_exponent() {
        let a = Vector::from_slice(&[2.0, 3.0, 4.0]);
        let result = a.pow(1.0).unwrap();
        assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0]); // x^1 = x
    }

    #[test]
    fn test_pow_negative_exponent() {
        let a = Vector::from_slice(&[2.0, 4.0, 10.0]);
        let result = a.pow(-1.0).unwrap();
        assert_eq!(result.as_slice(), &[0.5, 0.25, 0.1]); // x^(-1) = 1/x
    }

    #[test]
    fn test_pow_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.pow(2.0).unwrap();
        assert_eq!(result.len(), 0);
    }

    // exp() operation tests (element-wise exponential: e^x)
    #[test]
    fn test_exp_basic() {
        let a = Vector::from_slice(&[0.0, 1.0, 2.0]);
        let result = a.exp().unwrap();
        let expected = [1.0, std::f32::consts::E, std::f32::consts::E.powi(2)];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "exp mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_exp_zero() {
        let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = a.exp().unwrap();
        for &val in result.as_slice() {
            assert!((val - 1.0).abs() < 1e-5, "e^0 should be 1.0");
        }
    }

    #[test]
    fn test_exp_negative() {
        let a = Vector::from_slice(&[-1.0, -2.0, -3.0]);
        let result = a.exp().unwrap();
        let expected = [
            (-1.0f32).exp(),
            (-2.0f32).exp(),
            (-3.0f32).exp(),
        ];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "exp negative mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_exp_large_positive() {
        let a = Vector::from_slice(&[5.0, 10.0]);
        let result = a.exp().unwrap();
        let expected = [5.0f32.exp(), 10.0f32.exp()];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() / exp < 1e-5,
                "exp large positive mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_exp_large_negative() {
        let a = Vector::from_slice(&[-5.0, -10.0]);
        let result = a.exp().unwrap();
        let expected = [(-5.0f32).exp(), (-10.0f32).exp()];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "exp large negative mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_exp_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.exp().unwrap();
        assert_eq!(result.len(), 0);
    }

    // ln() operation tests (element-wise natural logarithm: ln(x))
    #[test]
    fn test_ln_basic() {
        let a = Vector::from_slice(&[1.0, std::f32::consts::E, std::f32::consts::E.powi(2)]);
        let result = a.ln().unwrap();
        let expected = [0.0, 1.0, 2.0];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "ln mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_ln_one() {
        let a = Vector::from_slice(&[1.0, 1.0, 1.0]);
        let result = a.ln().unwrap();
        for &val in result.as_slice() {
            assert!((val - 0.0).abs() < 1e-5, "ln(1) should be 0.0");
        }
    }

    #[test]
    fn test_ln_small_values() {
        let a = Vector::from_slice(&[0.1, 0.5, 0.9]);
        let result = a.ln().unwrap();
        let expected = [0.1f32.ln(), 0.5f32.ln(), 0.9f32.ln()];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "ln small values mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_ln_large_values() {
        let a = Vector::from_slice(&[10.0, 100.0, 1000.0]);
        let result = a.ln().unwrap();
        let expected = [10.0f32.ln(), 100.0f32.ln(), 1000.0f32.ln()];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "ln large values mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_ln_inverse_exp() {
        // Test that ln(exp(x)) = x
        let a = Vector::from_slice(&[0.5, 1.0, 2.0, 3.0]);
        let exp_result = a.exp().unwrap();
        let ln_result = exp_result.ln().unwrap();
        for (i, (&original, &recovered)) in a.as_slice().iter().zip(ln_result.as_slice().iter()).enumerate() {
            assert!(
                (original - recovered).abs() < 1e-5,
                "ln(exp(x)) != x at {}: {} != {}",
                i,
                original,
                recovered
            );
        }
    }

    #[test]
    fn test_ln_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.ln().unwrap();
        assert_eq!(result.len(), 0);
    }

    // sin() operation tests (element-wise sine)
    #[test]
    fn test_sin_basic() {
        use std::f32::consts::PI;
        let a = Vector::from_slice(&[0.0, PI / 2.0, PI, 3.0 * PI / 2.0]);
        let result = a.sin().unwrap();
        let expected = [0.0, 1.0, 0.0, -1.0];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "sin mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_sin_zero() {
        let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = a.sin().unwrap();
        for &val in result.as_slice() {
            assert!((val - 0.0).abs() < 1e-5, "sin(0) should be 0.0");
        }
    }

    #[test]
    fn test_sin_quarter_circle() {
        use std::f32::consts::PI;
        let a = Vector::from_slice(&[PI / 6.0, PI / 4.0, PI / 3.0]);
        let result = a.sin().unwrap();
        let expected = [0.5, std::f32::consts::FRAC_1_SQRT_2, (3.0f32).sqrt() / 2.0];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "sin quarter circle mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_sin_negative() {
        use std::f32::consts::PI;
        let a = Vector::from_slice(&[-PI / 2.0, -PI, -3.0 * PI / 2.0]);
        let result = a.sin().unwrap();
        let expected = [-1.0, 0.0, 1.0];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "sin negative mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_sin_periodicity() {
        use std::f32::consts::PI;
        // sin(x + 2π) = sin(x)
        let a = Vector::from_slice(&[0.5, 1.0, 1.5]);
        let b = Vector::from_slice(&[0.5 + 2.0 * PI, 1.0 + 2.0 * PI, 1.5 + 2.0 * PI]);
        let result_a = a.sin().unwrap();
        let result_b = b.sin().unwrap();
        for (i, (&res_a, &res_b)) in result_a.as_slice().iter().zip(result_b.as_slice().iter()).enumerate() {
            assert!(
                (res_a - res_b).abs() < 1e-5,
                "sin periodicity failed at {}: {} != {}",
                i,
                res_a,
                res_b
            );
        }
    }

    #[test]
    fn test_sin_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.sin().unwrap();
        assert_eq!(result.len(), 0);
    }

    // cos() operation tests (element-wise cosine)
    #[test]
    fn test_cos_basic() {
        use std::f32::consts::PI;
        let a = Vector::from_slice(&[0.0, PI / 2.0, PI, 3.0 * PI / 2.0]);
        let result = a.cos().unwrap();
        let expected = [1.0, 0.0, -1.0, 0.0];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "cos mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_cos_zero() {
        let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = a.cos().unwrap();
        for &val in result.as_slice() {
            assert!((val - 1.0).abs() < 1e-5, "cos(0) should be 1.0");
        }
    }

    #[test]
    fn test_cos_quarter_circle() {
        use std::f32::consts::PI;
        let a = Vector::from_slice(&[PI / 6.0, PI / 4.0, PI / 3.0]);
        let result = a.cos().unwrap();
        let expected = [(3.0f32).sqrt() / 2.0, std::f32::consts::FRAC_1_SQRT_2, 0.5];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "cos quarter circle mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_cos_negative() {
        use std::f32::consts::PI;
        let a = Vector::from_slice(&[-PI / 2.0, -PI, -3.0 * PI / 2.0]);
        let result = a.cos().unwrap();
        let expected = [0.0, -1.0, 0.0];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "cos negative mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_cos_sin_relation() {
        use std::f32::consts::PI;
        // cos(x) = sin(x + π/2)
        let a = Vector::from_slice(&[0.0, PI / 6.0, PI / 4.0, PI / 3.0]);
        let cos_result = a.cos().unwrap();

        let a_plus_pi_2: Vec<f32> = a.as_slice().iter().map(|&x| x + PI / 2.0).collect();
        let shifted = Vector::from_slice(&a_plus_pi_2);
        let sin_result = shifted.sin().unwrap();

        for (i, (&cos_val, &sin_val)) in cos_result.as_slice().iter().zip(sin_result.as_slice().iter()).enumerate() {
            assert!(
                (cos_val - sin_val).abs() < 1e-5,
                "cos(x) = sin(x + π/2) failed at {}: {} != {}",
                i,
                cos_val,
                sin_val
            );
        }
    }

    #[test]
    fn test_cos_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.cos().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_tan_basic() {
        use std::f32::consts::PI;
        // tan(0) = 0, tan(π/4) = 1, tan(-π/4) = -1
        let a = Vector::from_slice(&[0.0, PI / 4.0, -PI / 4.0]);
        let result = a.tan().unwrap();
        let expected = [0.0, 1.0, -1.0];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "tan basic mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_tan_zero() {
        let a = Vector::from_slice(&[0.0]);
        let result = a.tan().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_tan_quarter_circle() {
        use std::f32::consts::PI;
        // tan(π/4) = 1
        let a = Vector::from_slice(&[PI / 4.0]);
        let result = a.tan().unwrap();
        assert!((result.as_slice()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_tan_negative() {
        use std::f32::consts::PI;
        // tan is odd: tan(-x) = -tan(x)
        let a = Vector::from_slice(&[-PI / 4.0, -PI / 6.0]);
        let result = a.tan().unwrap();
        let expected = [-1.0, -(1.0 / 3.0_f32.sqrt())];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "tan negative mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_tan_sin_cos_relation() {
        use std::f32::consts::PI;
        // tan(x) = sin(x) / cos(x)
        let a = Vector::from_slice(&[PI / 6.0, PI / 4.0, PI / 3.0]);
        let tan_result = a.tan().unwrap();
        let sin_result = a.sin().unwrap();
        let cos_result = a.cos().unwrap();

        for (i, ((&tan_val, &sin_val), &cos_val)) in tan_result
            .as_slice()
            .iter()
            .zip(sin_result.as_slice().iter())
            .zip(cos_result.as_slice().iter())
            .enumerate()
        {
            let expected = sin_val / cos_val;
            assert!(
                (tan_val - expected).abs() < 1e-5,
                "tan(x) != sin(x)/cos(x) at {}: {} != {}",
                i,
                tan_val,
                expected
            );
        }
    }

    #[test]
    fn test_tan_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.tan().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_asin_basic() {
        use std::f32::consts::PI;
        // asin(0) = 0, asin(1) = π/2, asin(-1) = -π/2, asin(0.5) = π/6
        let a = Vector::from_slice(&[0.0, 1.0, -1.0, 0.5]);
        let result = a.asin().unwrap();
        let expected = [0.0, PI / 2.0, -PI / 2.0, PI / 6.0];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "asin basic mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_asin_zero() {
        let a = Vector::from_slice(&[0.0]);
        let result = a.asin().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_asin_range() {
        use std::f32::consts::PI;
        // asin domain is [-1, 1], range is [-π/2, π/2]
        let a = Vector::from_slice(&[-1.0, -0.5, 0.0, 0.5, 1.0]);
        let result = a.asin().unwrap();
        for (i, &res) in result.as_slice().iter().enumerate() {
            assert!(
                (-PI / 2.0..=PI / 2.0).contains(&res),
                "asin range violation at {}: {} not in [-π/2, π/2]",
                i,
                res
            );
        }
    }

    #[test]
    fn test_asin_negative() {
        use std::f32::consts::PI;
        // asin is odd: asin(-x) = -asin(x)
        let a = Vector::from_slice(&[-0.5, -0.707]);
        let result = a.asin().unwrap();
        let expected = [-PI / 6.0, -PI / 4.0];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-3,
                "asin negative mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_asin_sin_inverse() {
        use std::f32::consts::PI;
        // asin(sin(x)) = x for x in [-π/2, π/2]
        let a = Vector::from_slice(&[-PI / 4.0, 0.0, PI / 6.0, PI / 4.0]);
        let sin_result = a.sin().unwrap();
        let asin_result = sin_result.asin().unwrap();

        for (i, (&original, &reconstructed)) in
            a.as_slice().iter().zip(asin_result.as_slice().iter()).enumerate()
        {
            assert!(
                (original - reconstructed).abs() < 1e-5,
                "asin(sin(x)) != x at {}: {} != {}",
                i,
                reconstructed,
                original
            );
        }
    }

    #[test]
    fn test_asin_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.asin().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_acos_basic() {
        use std::f32::consts::PI;
        // acos(0) = π/2, acos(1) = 0, acos(-1) = π, acos(0.5) = π/3
        let a = Vector::from_slice(&[0.0, 1.0, -1.0, 0.5]);
        let result = a.acos().unwrap();
        let expected = [PI / 2.0, 0.0, PI, PI / 3.0];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "acos basic mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_acos_zero() {
        use std::f32::consts::PI;
        let a = Vector::from_slice(&[1.0]);
        let result = a.acos().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);

        // Also test acos(0) = π/2
        let b = Vector::from_slice(&[0.0]);
        let result_b = b.acos().unwrap();
        assert!((result_b.as_slice()[0] - PI / 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_acos_range() {
        use std::f32::consts::PI;
        // acos domain is [-1, 1], range is [0, π]
        let a = Vector::from_slice(&[-1.0, -0.5, 0.0, 0.5, 1.0]);
        let result = a.acos().unwrap();
        for (i, &res) in result.as_slice().iter().enumerate() {
            assert!(
                (0.0..=PI).contains(&res),
                "acos range violation at {}: {} not in [0, π]",
                i,
                res
            );
        }
    }

    #[test]
    fn test_acos_symmetry() {
        use std::f32::consts::PI;
        // acos(-x) = π - acos(x)
        let a = Vector::from_slice(&[0.5, 0.707]);
        let result_pos = a.acos().unwrap();

        let a_neg = Vector::from_slice(&[-0.5, -0.707]);
        let result_neg = a_neg.acos().unwrap();

        for (i, (&pos, &neg)) in result_pos.as_slice().iter().zip(result_neg.as_slice().iter()).enumerate() {
            let expected_neg = PI - pos;
            assert!(
                (neg - expected_neg).abs() < 1e-5,
                "acos symmetry failed at {}: acos(-x)={} != π - acos(x)={}",
                i,
                neg,
                expected_neg
            );
        }
    }

    #[test]
    fn test_acos_cos_inverse() {
        use std::f32::consts::PI;
        // acos(cos(x)) = x for x in [0, π]
        let a = Vector::from_slice(&[0.0, PI / 6.0, PI / 4.0, PI / 2.0, PI]);
        let cos_result = a.cos().unwrap();
        let acos_result = cos_result.acos().unwrap();

        for (i, (&original, &reconstructed)) in
            a.as_slice().iter().zip(acos_result.as_slice().iter()).enumerate()
        {
            assert!(
                (original - reconstructed).abs() < 1e-5,
                "acos(cos(x)) != x at {}: {} != {}",
                i,
                reconstructed,
                original
            );
        }
    }

    #[test]
    fn test_acos_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.acos().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_atan_basic() {
        use std::f32::consts::PI;
        // atan(0) = 0, atan(1) = π/4, atan(-1) = -π/4
        let a = Vector::from_slice(&[0.0, 1.0, -1.0, 1.732]); // 1.732 ≈ √3 for atan(√3) = π/3
        let result = a.atan().unwrap();
        let expected = [0.0, PI / 4.0, -PI / 4.0, PI / 3.0];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-3,
                "atan basic mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_atan_zero() {
        let a = Vector::from_slice(&[0.0]);
        let result = a.atan().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_atan_range() {
        use std::f32::consts::PI;
        // atan range is (-π/2, π/2) for all real inputs
        let a = Vector::from_slice(&[-1000.0, -10.0, -1.0, 0.0, 1.0, 10.0, 1000.0]);
        let result = a.atan().unwrap();
        for (i, &res) in result.as_slice().iter().enumerate() {
            assert!(
                (-PI / 2.0..PI / 2.0).contains(&res),
                "atan range violation at {}: {} not in (-π/2, π/2)",
                i,
                res
            );
        }
    }

    #[test]
    fn test_atan_negative() {
        use std::f32::consts::PI;
        // atan is odd: atan(-x) = -atan(x)
        let a = Vector::from_slice(&[-1.0, -1.732]);
        let result = a.atan().unwrap();
        let expected = [-PI / 4.0, -PI / 3.0];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-3,
                "atan negative mismatch at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_atan_tan_inverse() {
        use std::f32::consts::PI;
        // atan(tan(x)) = x for x in (-π/2, π/2)
        let a = Vector::from_slice(&[-PI / 4.0, 0.0, PI / 6.0, PI / 4.0]);
        let tan_result = a.tan().unwrap();
        let atan_result = tan_result.atan().unwrap();

        for (i, (&original, &reconstructed)) in
            a.as_slice().iter().zip(atan_result.as_slice().iter()).enumerate()
        {
            assert!(
                (original - reconstructed).abs() < 1e-5,
                "atan(tan(x)) != x at {}: {} != {}",
                i,
                reconstructed,
                original
            );
        }
    }

    #[test]
    fn test_atan_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.atan().unwrap();
        assert_eq!(result.len(), 0);
    }

    // sinh() tests
    #[test]
    fn test_sinh_basic() {
        let a = Vector::from_slice(&[0.0, 1.0, -1.0]);
        let result = a.sinh().unwrap();
        let expected = [0.0, 1.0_f32.sinh(), (-1.0_f32).sinh()];
        for (r, e) in result.as_slice().iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
        }
    }

    #[test]
    fn test_sinh_zero() {
        let a = Vector::from_slice(&[0.0]);
        let result = a.sinh().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_sinh_positive() {
        let a = Vector::from_slice(&[2.0]);
        let result = a.sinh().unwrap();
        let expected = 2.0_f32.sinh();
        assert!((result.as_slice()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_sinh_negative() {
        let a = Vector::from_slice(&[-2.0]);
        let result = a.sinh().unwrap();
        let expected = (-2.0_f32).sinh();
        assert!((result.as_slice()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_sinh_odd_function() {
        // sinh(-x) = -sinh(x)
        let a = Vector::from_slice(&[1.5]);
        let b = Vector::from_slice(&[-1.5]);
        let sinh_a = a.sinh().unwrap();
        let sinh_b = b.sinh().unwrap();
        assert!(
            (sinh_a.as_slice()[0] + sinh_b.as_slice()[0]).abs() < 1e-5,
            "sinh is an odd function: sinh(-x) = -sinh(x)"
        );
    }

    #[test]
    fn test_sinh_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.sinh().unwrap();
        assert_eq!(result.len(), 0);
    }

    // cosh() tests
    #[test]
    fn test_cosh_basic() {
        let a = Vector::from_slice(&[0.0, 1.0, -1.0]);
        let result = a.cosh().unwrap();
        let expected = [0.0_f32.cosh(), 1.0_f32.cosh(), (-1.0_f32).cosh()];
        for (r, e) in result.as_slice().iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
        }
    }

    #[test]
    fn test_cosh_zero() {
        let a = Vector::from_slice(&[0.0]);
        let result = a.cosh().unwrap();
        assert!((result.as_slice()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosh_positive() {
        let a = Vector::from_slice(&[2.0]);
        let result = a.cosh().unwrap();
        let expected = 2.0_f32.cosh();
        assert!((result.as_slice()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_cosh_negative() {
        let a = Vector::from_slice(&[-2.0]);
        let result = a.cosh().unwrap();
        let expected = (-2.0_f32).cosh();
        assert!((result.as_slice()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_cosh_even_function() {
        // cosh(-x) = cosh(x)
        let a = Vector::from_slice(&[1.5]);
        let b = Vector::from_slice(&[-1.5]);
        let cosh_a = a.cosh().unwrap();
        let cosh_b = b.cosh().unwrap();
        assert!(
            (cosh_a.as_slice()[0] - cosh_b.as_slice()[0]).abs() < 1e-5,
            "cosh is an even function: cosh(-x) = cosh(x)"
        );
    }

    #[test]
    fn test_cosh_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.cosh().unwrap();
        assert_eq!(result.len(), 0);
    }

    // tanh() tests
    #[test]
    fn test_tanh_basic() {
        let a = Vector::from_slice(&[0.0, 1.0, -1.0]);
        let result = a.tanh().unwrap();
        let expected = [0.0_f32.tanh(), 1.0_f32.tanh(), (-1.0_f32).tanh()];
        for (r, e) in result.as_slice().iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
        }
    }

    #[test]
    fn test_tanh_zero() {
        let a = Vector::from_slice(&[0.0]);
        let result = a.tanh().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_tanh_range() {
        // tanh(x) is bounded: -1 <= tanh(x) <= 1
        // For very large values, it approaches ±1 in floating-point
        let a = Vector::from_slice(&[10.0, -10.0, 100.0]);
        let result = a.tanh().unwrap();
        for &val in result.as_slice() {
            assert!((-1.0..=1.0).contains(&val), "tanh value {} out of range [-1, 1]", val);
        }
    }

    #[test]
    fn test_tanh_negative() {
        let a = Vector::from_slice(&[-2.0]);
        let result = a.tanh().unwrap();
        let expected = (-2.0_f32).tanh();
        assert!((result.as_slice()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_tanh_sinh_cosh_relation() {
        // tanh(x) = sinh(x) / cosh(x)
        let a = Vector::from_slice(&[1.5]);
        let tanh_result = a.tanh().unwrap();
        let sinh_result = a.sinh().unwrap();
        let cosh_result = a.cosh().unwrap();
        let ratio = sinh_result.as_slice()[0] / cosh_result.as_slice()[0];
        assert!(
            (tanh_result.as_slice()[0] - ratio).abs() < 1e-5,
            "tanh(x) = sinh(x)/cosh(x)"
        );
    }

    #[test]
    fn test_tanh_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.tanh().unwrap();
        assert_eq!(result.len(), 0);
    }

    // asinh() tests
    #[test]
    fn test_asinh_basic() {
        let a = Vector::from_slice(&[0.0, 1.0, -1.0]);
        let result = a.asinh().unwrap();
        let expected = [0.0_f32.asinh(), 1.0_f32.asinh(), (-1.0_f32).asinh()];
        for (r, e) in result.as_slice().iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
        }
    }

    #[test]
    fn test_asinh_zero() {
        let a = Vector::from_slice(&[0.0]);
        let result = a.asinh().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_asinh_positive() {
        let a = Vector::from_slice(&[2.0]);
        let result = a.asinh().unwrap();
        let expected = 2.0_f32.asinh();
        assert!((result.as_slice()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_asinh_negative() {
        let a = Vector::from_slice(&[-2.0]);
        let result = a.asinh().unwrap();
        let expected = (-2.0_f32).asinh();
        assert!((result.as_slice()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_asinh_odd_function() {
        // asinh(-x) = -asinh(x)
        let a = Vector::from_slice(&[1.5]);
        let b = Vector::from_slice(&[-1.5]);
        let asinh_a = a.asinh().unwrap();
        let asinh_b = b.asinh().unwrap();
        assert!(
            (asinh_a.as_slice()[0] + asinh_b.as_slice()[0]).abs() < 1e-5,
            "asinh is an odd function: asinh(-x) = -asinh(x)"
        );
    }

    #[test]
    fn test_asinh_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.asinh().unwrap();
        assert_eq!(result.len(), 0);
    }

    // acosh() tests
    #[test]
    fn test_acosh_basic() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = a.acosh().unwrap();
        let expected = [1.0_f32.acosh(), 2.0_f32.acosh(), 3.0_f32.acosh()];
        for (r, e) in result.as_slice().iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5, "Expected {}, got {}", e, r);
        }
    }

    #[test]
    fn test_acosh_one() {
        let a = Vector::from_slice(&[1.0]);
        let result = a.acosh().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_acosh_positive() {
        let a = Vector::from_slice(&[2.0]);
        let result = a.acosh().unwrap();
        let expected = 2.0_f32.acosh();
        assert!((result.as_slice()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_acosh_large() {
        let a = Vector::from_slice(&[10.0]);
        let result = a.acosh().unwrap();
        let expected = 10.0_f32.acosh();
        assert!((result.as_slice()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_acosh_cosh_relation() {
        // acosh(cosh(x)) = x for x >= 0
        let a = Vector::from_slice(&[1.5]);
        let cosh_result = a.cosh().unwrap();
        let acosh_result = cosh_result.acosh().unwrap();
        assert!(
            (a.as_slice()[0] - acosh_result.as_slice()[0]).abs() < 1e-5,
            "acosh(cosh(x)) = x"
        );
    }

    #[test]
    fn test_acosh_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.acosh().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_atanh_basic() {
        let a = Vector::from_slice(&[0.0, 0.5, -0.5]);
        let result = a.atanh().unwrap();
        let expected: Vec<f32> = vec![0.0_f32.atanh(), 0.5_f32.atanh(), (-0.5_f32).atanh()];
        for (i, (&res, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert!(
                (res - exp).abs() < 1e-5,
                "atanh failed at {}: {} != {}",
                i,
                res,
                exp
            );
        }
    }

    #[test]
    fn test_atanh_zero() {
        let a = Vector::from_slice(&[0.0]);
        let result = a.atanh().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_atanh_positive() {
        let a = Vector::from_slice(&[0.5]);
        let result = a.atanh().unwrap();
        let expected = 0.5_f32.atanh();
        assert!((result.as_slice()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_atanh_negative() {
        let a = Vector::from_slice(&[-0.5]);
        let result = a.atanh().unwrap();
        let expected = (-0.5_f32).atanh();
        assert!((result.as_slice()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_atanh_odd_function() {
        // atanh(-x) = -atanh(x)
        let a = Vector::from_slice(&[0.5]);
        let neg_a = Vector::from_slice(&[-0.5]);
        let result_a = a.atanh().unwrap();
        let result_neg_a = neg_a.atanh().unwrap();
        assert!(
            (result_a.as_slice()[0] + result_neg_a.as_slice()[0]).abs() < 1e-5,
            "atanh(-x) = -atanh(x)"
        );
    }

    #[test]
    fn test_atanh_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.atanh().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_floor_basic() {
        let a = Vector::from_slice(&[3.7, -2.3, 5.0]);
        let result = a.floor().unwrap();
        assert_eq!(result.as_slice(), &[3.0, -3.0, 5.0]);
    }

    #[test]
    fn test_floor_positive() {
        let a = Vector::from_slice(&[1.1, 2.9, 3.5]);
        let result = a.floor().unwrap();
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_floor_negative() {
        let a = Vector::from_slice(&[-1.1, -2.9, -3.5]);
        let result = a.floor().unwrap();
        assert_eq!(result.as_slice(), &[-2.0, -3.0, -4.0]);
    }

    #[test]
    fn test_floor_integers() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0, -4.0]);
        let result = a.floor().unwrap();
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, -4.0]);
    }

    #[test]
    fn test_floor_zero() {
        let a = Vector::from_slice(&[0.0, -0.0]);
        let result = a.floor().unwrap();
        assert_eq!(result.as_slice(), &[0.0, -0.0]);
    }

    #[test]
    fn test_floor_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.floor().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_ceil_basic() {
        let a = Vector::from_slice(&[3.2, -2.7, 5.0]);
        let result = a.ceil().unwrap();
        assert_eq!(result.as_slice(), &[4.0, -2.0, 5.0]);
    }

    #[test]
    fn test_ceil_positive() {
        let a = Vector::from_slice(&[1.1, 2.9, 3.5]);
        let result = a.ceil().unwrap();
        assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_ceil_negative() {
        let a = Vector::from_slice(&[-1.1, -2.9, -3.5]);
        let result = a.ceil().unwrap();
        assert_eq!(result.as_slice(), &[-1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_ceil_integers() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0, -4.0]);
        let result = a.ceil().unwrap();
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, -4.0]);
    }

    #[test]
    fn test_ceil_zero() {
        let a = Vector::from_slice(&[0.0, -0.0]);
        let result = a.ceil().unwrap();
        assert_eq!(result.as_slice(), &[0.0, -0.0]);
    }

    #[test]
    fn test_ceil_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.ceil().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_round_basic() {
        let a = Vector::from_slice(&[3.2, 3.7, -2.3, -2.8, 5.0]);
        let result = a.round().unwrap();
        assert_eq!(result.as_slice(), &[3.0, 4.0, -2.0, -3.0, 5.0]);
    }

    #[test]
    fn test_round_positive() {
        let a = Vector::from_slice(&[1.4, 1.5, 1.6, 2.5]);
        let result = a.round().unwrap();
        assert_eq!(result.as_slice(), &[1.0, 2.0, 2.0, 3.0]);
    }

    #[test]
    fn test_round_negative() {
        let a = Vector::from_slice(&[-1.4, -1.5, -1.6, -2.5]);
        let result = a.round().unwrap();
        assert_eq!(result.as_slice(), &[-1.0, -2.0, -2.0, -3.0]);
    }

    #[test]
    fn test_round_halfway() {
        // Rust's round() uses "round half away from zero"
        let a = Vector::from_slice(&[0.5, 1.5, 2.5, 3.5, 4.5]);
        let result = a.round().unwrap();
        assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_round_zero() {
        let a = Vector::from_slice(&[0.0, -0.0, 0.3, -0.3]);
        let result = a.round().unwrap();
        assert_eq!(result.as_slice(), &[0.0, -0.0, 0.0, -0.0]);
    }

    #[test]
    fn test_round_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.round().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_trunc_basic() {
        let a = Vector::from_slice(&[3.2, 3.7, -2.3, -2.8, 5.0]);
        let result = a.trunc().unwrap();
        assert_eq!(result.as_slice(), &[3.0, 3.0, -2.0, -2.0, 5.0]);
    }

    #[test]
    fn test_trunc_positive() {
        let a = Vector::from_slice(&[1.1, 1.9, 2.5, 3.99]);
        let result = a.trunc().unwrap();
        assert_eq!(result.as_slice(), &[1.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_trunc_negative() {
        let a = Vector::from_slice(&[-1.1, -1.9, -2.5, -3.99]);
        let result = a.trunc().unwrap();
        assert_eq!(result.as_slice(), &[-1.0, -1.0, -2.0, -3.0]);
    }

    #[test]
    fn test_trunc_toward_zero() {
        // Verify trunc() always moves toward zero
        let a = Vector::from_slice(&[2.7, -2.7, 5.3, -5.3]);
        let result = a.trunc().unwrap();
        assert_eq!(result.as_slice(), &[2.0, -2.0, 5.0, -5.0]);
    }

    #[test]
    fn test_trunc_zero() {
        let a = Vector::from_slice(&[0.0, -0.0, 0.9, -0.9]);
        let result = a.trunc().unwrap();
        assert_eq!(result.as_slice(), &[0.0, -0.0, 0.0, -0.0]);
    }

    #[test]
    fn test_trunc_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.trunc().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_aligned_vector_creation() {
        let v = Vector::with_alignment(100, Backend::SSE2, 16).unwrap();

        // Verify the vector has the correct size
        assert_eq!(v.len(), 100);

        // Check alignment (Vec allocator typically provides good alignment)
        let ptr = v.as_slice().as_ptr() as usize;
        // Note: We can't guarantee specific alignment with standard Vec,
        // but we can verify it's at least naturally aligned for f32 (4 bytes)
        assert_eq!(ptr % 4, 0, "Vector data should be at least 4-byte aligned");

        // Most modern allocators provide 16-byte alignment by default
        // This is informational, not required
        if ptr.is_multiple_of(16) {
            println!("Got 16-byte alignment from standard allocator");
        }
    }

    #[test]
    fn test_aligned_vector_operations() {
        // RED: This test will fail until we implement aligned allocation
        let a = Vector::with_alignment(1000, Backend::SSE2, 16).unwrap();
        let b = Vector::with_alignment(1000, Backend::SSE2, 16).unwrap();

        // Operations on aligned vectors should work correctly
        let result = a.add(&b);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1000);
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

    // Property test: Subtraction anti-commutativity (a - b == -(b - a))
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sub_anti_commutative(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100),
            b in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let len = a.len().min(b.len());
            let a_vec: Vec<f32> = a.into_iter().take(len).collect();
            let b_vec: Vec<f32> = b.into_iter().take(len).collect();

            let va = Vector::from_slice(&a_vec);
            let vb = Vector::from_slice(&b_vec);

            let result1 = va.sub(&vb).unwrap();
            let result2 = vb.sub(&va).unwrap();

            // a - b should equal -(b - a)
            for (x, y) in result1.as_slice().iter().zip(result2.as_slice()) {
                prop_assert!((x + y).abs() < 1e-5);
            }
        }
    }

    // Property test: Subtraction identity (a - 0 == a)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sub_identity(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let zero = Vector::from_slice(&vec![0.0; a.len()]);

            let result = va.sub(&zero).unwrap();

            prop_assert_eq!(result.as_slice(), va.as_slice());
        }
    }

    // Property test: Subtraction inverse (a - a == 0)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sub_inverse(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);

            let result = va.sub(&va).unwrap();

            // All elements should be zero (or very close due to floating point)
            for &x in result.as_slice() {
                prop_assert!(x.abs() < 1e-5);
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

    // Property test: Division identity (a / 1 == a)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_div_identity(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let ones = Vector::from_slice(&vec![1.0; a.len()]);

            let result = va.div(&ones).unwrap();

            for (x, y) in result.as_slice().iter().zip(va.as_slice()) {
                prop_assert!((x - y).abs() < 1e-5);
            }
        }
    }

    // Property test: Division inverse (a / a == 1, for non-zero a)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_div_inverse(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            // Filter out zeros to avoid division edge cases
            let a_nonzero: Vec<f32> = a.into_iter()
                .map(|x| if x.abs() < 1e-5 { 1.0 } else { x })
                .collect();

            let va = Vector::from_slice(&a_nonzero);
            let result = va.div(&va).unwrap();

            // All elements should be 1.0 (or very close)
            for &x in result.as_slice() {
                prop_assert!((x - 1.0).abs() < 1e-4);
            }
        }
    }

    // Property test: Division-multiplication inverse (a / b) * b ≈ a
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_div_mul_inverse(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100),
            b in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let len = a.len().min(b.len());
            let a_vec: Vec<f32> = a.into_iter().take(len).collect();

            // Filter out zeros from b to avoid division by zero edge cases
            let b_vec: Vec<f32> = b.into_iter().take(len)
                .map(|x| if x.abs() < 1e-3 { 1.0 } else { x })
                .collect();

            let va = Vector::from_slice(&a_vec);
            let vb = Vector::from_slice(&b_vec);

            let divided = va.div(&vb).unwrap();
            let restored = divided.mul(&vb).unwrap();

            // Restored should approximately equal original
            for (original, restored_val) in a_vec.iter().zip(restored.as_slice()) {
                prop_assert!((original - restored_val).abs() < 1e-2);
            }
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

        #[test]
        fn test_sum_kahan_correctness(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let kahan_result = va.sum_kahan().unwrap();
            let manual_sum: f32 = a.iter().sum();

            // Kahan result should be close to manual sum
            // Note: Both use same algorithm (iter().sum() also uses compensated summation)
            // so they should match closely
            prop_assert!((kahan_result - manual_sum).abs() < 1e-2,
                "Kahan sum should match manual sum closely");

            // Verify Kahan produces a reasonable result
            let expected_magnitude = a.iter().map(|x| x.abs()).sum::<f32>();
            prop_assert!(kahan_result.abs() <= expected_magnitude + 1.0,
                "Kahan result magnitude should be reasonable");
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

        #[test]
        fn test_min_is_minimum(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.min().unwrap();

            // Verify result is <= all elements
            for &x in a.iter() {
                prop_assert!(result <= x);
            }

            // Verify result is actually in the vector
            prop_assert!(a.contains(&result));
        }

        #[test]
        fn test_argmax_correctness(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let idx = va.argmax().unwrap();

            // Verify index is in bounds
            prop_assert!(idx < a.len());

            // Verify value at index is >= all other values
            let max_val = a[idx];
            for &x in a.iter() {
                prop_assert!(max_val >= x);
            }

            // Verify it's the first occurrence (no earlier index has this value)
            for &val in a.iter().take(idx) {
                prop_assert!(val < max_val || val != max_val);
            }
        }

        #[test]
        fn test_argmin_correctness(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let idx = va.argmin().unwrap();

            // Verify index is in bounds
            prop_assert!(idx < a.len());

            // Verify value at index is <= all other values
            let min_val = a[idx];
            for &x in a.iter() {
                prop_assert!(min_val <= x);
            }

            // Verify it's the first occurrence (no earlier index has this value)
            for &val in a.iter().take(idx) {
                prop_assert!(val > min_val || val != min_val);
            }
        }
    }

    // Property test: Dot product with self is non-negative (norm property)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_dot_self_nonnegative(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.dot(&va).unwrap();

            // ||v||^2 = v·v >= 0 always
            prop_assert!(result >= 0.0);

            // If all zeros, should be exactly zero
            if a.iter().all(|&x| x == 0.0) {
                prop_assert_eq!(result, 0.0);
            } else {
                // If any non-zero element, result should be positive
                prop_assert!(result > 0.0);
            }
        }
    }

    // Property test: L2 norm is always non-negative
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_norm_l2_nonnegative(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let norm = va.norm_l2().unwrap();

            // ||v|| >= 0 always
            prop_assert!(norm >= 0.0);

            // If all zeros, norm should be exactly zero
            if a.iter().all(|&x| x.abs() < 1e-6) {
                prop_assert!(norm < 1e-5);
            }
        }
    }

    // Property test: L2 norm equals sqrt(dot(a, a))
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_norm_l2_equals_sqrt_dot(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let norm = va.norm_l2().unwrap();
            let dot_self = va.dot(&va).unwrap();

            // ||a|| = sqrt(a·a)
            // Use relative tolerance for large values
            let relative_error = if dot_self > 0.0 {
                ((norm * norm - dot_self) / dot_self).abs()
            } else {
                (norm * norm - dot_self).abs()
            };
            prop_assert!(relative_error < 1e-4 || (norm * norm - dot_self).abs() < 1e-2);
        }
    }

    // Property test: Scaling property ||c*a|| = |c| * ||a||
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_norm_l2_scaling(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            c in -10.0f32..10.0
        ) {
            let va = Vector::from_slice(&a);
            let norm_a = va.norm_l2().unwrap();

            // Create c*a
            let scaled: Vec<f32> = a.iter().map(|&x| c * x).collect();
            let v_scaled = Vector::from_slice(&scaled);
            let norm_scaled = v_scaled.norm_l2().unwrap();

            // ||c*a|| = |c| * ||a||
            let expected = c.abs() * norm_a;
            prop_assert!((norm_scaled - expected).abs() < 1e-2);
        }
    }

    // Property test: Cauchy-Schwarz inequality |a·b| <= ||a|| * ||b||
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_cauchy_schwarz(
            a in prop::collection::vec(-100.0f32..100.0, 1..50),
            b in prop::collection::vec(-100.0f32..100.0, 1..50)
        ) {
            let len = a.len().min(b.len());
            let a_vec: Vec<f32> = a.into_iter().take(len).collect();
            let b_vec: Vec<f32> = b.into_iter().take(len).collect();

            let va = Vector::from_slice(&a_vec);
            let vb = Vector::from_slice(&b_vec);

            let dot_ab = va.dot(&vb).unwrap().abs();
            let norm_a = va.dot(&va).unwrap().sqrt();
            let norm_b = vb.dot(&vb).unwrap().sqrt();

            // |a·b| <= ||a|| * ||b||
            // Add small tolerance for floating point
            prop_assert!(dot_ab <= norm_a * norm_b + 1e-3);
        }
    }

    // Property test: Scaling property (multiply all by same constant)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_scalar_multiplication(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            scalar in -10.0f32..10.0
        ) {
            let va = Vector::from_slice(&a);

            // Create vector of all same scalar
            let scalars = vec![scalar; a.len()];
            let vs = Vector::from_slice(&scalars);

            let result = va.mul(&vs).unwrap();

            // Each element should be a[i] * scalar
            for (i, &val) in result.as_slice().iter().enumerate() {
                let expected = a[i] * scalar;
                prop_assert!((val - expected).abs() < 1e-3);
            }
        }
    }

    // Property test: Sum of scaled vector = scale * sum
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sum_linearity(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            scalar in -10.0f32..10.0
        ) {
            let va = Vector::from_slice(&a);

            // Create scaled version
            let scalars = vec![scalar; a.len()];
            let vs = Vector::from_slice(&scalars);
            let scaled = va.mul(&vs).unwrap();

            let sum_scaled = scaled.sum().unwrap();
            let sum_original = va.sum().unwrap();

            // sum(scalar * v) = scalar * sum(v)
            let expected = scalar * sum_original;
            prop_assert!((sum_scaled - expected).abs() < 1e-2);
        }
    }

    // Property test: Normalized vector has unit norm
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_normalize_unit_norm(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            // Skip if vector is too close to zero (would cause division by zero)
            let norm_squared: f32 = a.iter().map(|x| x * x).sum();
            prop_assume!(norm_squared > 1e-6);

            let va = Vector::from_slice(&a);
            let normalized = va.normalize().unwrap();

            // The normalized vector should have L2 norm = 1
            let norm = normalized.norm_l2().unwrap();
            prop_assert!((norm - 1.0).abs() < 1e-4, "norm = {}, expected 1.0", norm);
        }
    }

    // Property test: Normalization preserves direction (scaling invariance)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_normalize_direction_invariant(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            scale in 0.1f32..10.0
        ) {
            // Skip if vector is too close to zero
            let norm_squared: f32 = a.iter().map(|x| x * x).sum();
            prop_assume!(norm_squared > 1e-6);

            let va = Vector::from_slice(&a);

            // Scale the vector
            let scales = vec![scale; a.len()];
            let vs = Vector::from_slice(&scales);
            let scaled = va.mul(&vs).unwrap();

            // Both should normalize to the same direction
            let norm_a = va.normalize().unwrap();
            let norm_scaled = scaled.normalize().unwrap();

            // Check each element is close
            for (i, (&val_a, &val_scaled)) in norm_a.as_slice().iter()
                .zip(norm_scaled.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (val_a - val_scaled).abs() < 1e-4,
                    "Element {} differs: {} vs {}", i, val_a, val_scaled
                );
            }
        }
    }

    // Property test: L1 norm triangle inequality
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_norm_l1_triangle_inequality(
            len in 1usize..100,
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            b in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            // Triangle inequality: ||a + b||₁ <= ||a||₁ + ||b||₁
            // Use same length for both vectors
            let actual_len = len.min(a.len()).min(b.len());
            let a_trimmed = &a[..actual_len];
            let b_trimmed = &b[..actual_len];

            let va = Vector::from_slice(a_trimmed);
            let vb = Vector::from_slice(b_trimmed);

            let norm_a = va.norm_l1().unwrap();
            let norm_b = vb.norm_l1().unwrap();
            let sum = va.add(&vb).unwrap();
            let norm_sum = sum.norm_l1().unwrap();

            // Triangle inequality should hold
            prop_assert!(
                norm_sum <= norm_a + norm_b + 1e-3,
                "Triangle inequality violated: {} > {} + {}",
                norm_sum, norm_a, norm_b
            );
        }
    }

    // Property test: L1 norm absolute homogeneity
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_norm_l1_absolute_homogeneity(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            scalar in -10.0f32..10.0
        ) {
            // Absolute homogeneity: ||c * v||₁ = |c| * ||v||₁
            let va = Vector::from_slice(&a);

            let norm_a = va.norm_l1().unwrap();

            // Scale the vector
            let scalars = vec![scalar; a.len()];
            let vs = Vector::from_slice(&scalars);
            let scaled = va.mul(&vs).unwrap();

            let norm_scaled = scaled.norm_l1().unwrap();

            // Should satisfy: ||c*v||₁ = |c| * ||v||₁
            let expected = scalar.abs() * norm_a;

            // Use relative tolerance for large values
            let tolerance = if expected.abs() > 1.0 {
                expected.abs() * 1e-5 // Relative tolerance
            } else {
                1e-2 // Absolute tolerance for small values
            };

            prop_assert!(
                (norm_scaled - expected).abs() < tolerance,
                "Homogeneity violated: {} != |{}| * {} = {}, diff = {}",
                norm_scaled, scalar, norm_a, expected, (norm_scaled - expected).abs()
            );
        }
    }

    // Property test: L1 norm equals sum of absolute values
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_norm_l1_definition(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let norm = va.norm_l1().unwrap();

            // Manual calculation of sum(|a[i]|)
            let manual_sum: f32 = a.iter().map(|x| x.abs()).sum();

            // Use relative tolerance for large values
            let tolerance = if manual_sum.abs() > 1.0 {
                manual_sum.abs() * 1e-5 // Relative tolerance
            } else {
                1e-3 // Absolute tolerance for small values
            };

            prop_assert!(
                (norm - manual_sum).abs() < tolerance,
                "L1 norm {} != manual sum {}, diff = {}",
                norm, manual_sum, (norm - manual_sum).abs()
            );
        }
    }

    // Property test: L∞ norm absolute homogeneity
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_norm_linf_absolute_homogeneity(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            scalar in -10.0f32..10.0
        ) {
            prop_assume!(!a.is_empty());

            // Absolute homogeneity: ||c * v||∞ = |c| * ||v||∞
            let va = Vector::from_slice(&a);
            let norm_a = va.norm_linf().unwrap();

            // Scale the vector
            let scalars = vec![scalar; a.len()];
            let vs = Vector::from_slice(&scalars);
            let scaled = va.mul(&vs).unwrap();

            let norm_scaled = scaled.norm_linf().unwrap();

            // Should satisfy: ||c*v||∞ = |c| * ||v||∞
            let expected = scalar.abs() * norm_a;
            prop_assert!(
                (norm_scaled - expected).abs() < 1e-3,
                "Homogeneity violated: {} != |{}| * {} = {}",
                norm_scaled, scalar, norm_a, expected
            );
        }
    }

    // Property test: L∞ norm equals max of absolute values
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_norm_linf_definition(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            prop_assume!(!a.is_empty());

            let va = Vector::from_slice(&a);
            let norm = va.norm_linf().unwrap();

            // Manual calculation of max(|a[i]|)
            let manual_max = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

            prop_assert!(
                (norm - manual_max).abs() < 1e-5,
                "L∞ norm {} != manual max {}",
                norm, manual_max
            );
        }
    }

    // Property test: L∞ norm submultiplicativity (Hölder's inequality special case)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_norm_linf_submultiplicative(
            len in 1usize..100,
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            b in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            // For element-wise multiplication: ||a ⊙ b||∞ <= ||a||∞ * ||b||∞
            let actual_len = len.min(a.len()).min(b.len());
            let a_trimmed = &a[..actual_len];
            let b_trimmed = &b[..actual_len];

            let va = Vector::from_slice(a_trimmed);
            let vb = Vector::from_slice(b_trimmed);

            let norm_a = va.norm_linf().unwrap();
            let norm_b = vb.norm_linf().unwrap();
            let product = va.mul(&vb).unwrap();
            let norm_product = product.norm_linf().unwrap();

            // Submultiplicativity should hold
            prop_assert!(
                norm_product <= norm_a * norm_b + 1e-3,
                "Submultiplicativity violated: {} > {} * {}",
                norm_product, norm_a, norm_b
            );
        }
    }

    // Property test: abs() idempotence
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_abs_idempotent(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            // abs(abs(v)) = abs(v) - applying twice should be same as once
            let va = Vector::from_slice(&a);
            let abs_once = va.abs().unwrap();
            let abs_twice = abs_once.abs().unwrap();

            for (i, (&val_once, &val_twice)) in abs_once.as_slice().iter()
                .zip(abs_twice.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (val_once - val_twice).abs() < 1e-5,
                    "Idempotence failed at {}: {} != {}",
                    i, val_once, val_twice
                );
            }
        }
    }

    // Property test: abs() is always non-negative
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_abs_non_negative(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.abs().unwrap();

            for (i, &val) in result.as_slice().iter().enumerate() {
                prop_assert!(
                    val >= 0.0,
                    "Negative value at {}: {}",
                    i, val
                );
            }
        }
    }

    // Property test: abs() correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_abs_correctness(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.abs().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = input.abs();
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "Incorrect abs at {}: {} -> {}, expected {}",
                    i, input, output, expected
                );
            }
        }
    }

    // Property test: scale() distributivity over addition
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_scale_distributive(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            scalar in -10.0f32..10.0
        ) {
            // scalar * (a + a) = (scalar * a) + (scalar * a)
            let va = Vector::from_slice(&a);
            let va_plus_va = va.add(&va).unwrap();
            let scaled_sum = va_plus_va.scale(scalar).unwrap();

            let scaled_a = va.scale(scalar).unwrap();
            let sum_of_scaled = scaled_a.add(&scaled_a).unwrap();

            for (i, (&val1, &val2)) in scaled_sum.as_slice().iter()
                .zip(sum_of_scaled.as_slice().iter())
                .enumerate() {
                let tolerance = if val1.abs() > 1.0 {
                    val1.abs() * 1e-5
                } else {
                    1e-3
                };
                prop_assert!(
                    (val1 - val2).abs() < tolerance,
                    "Distributivity failed at {}: {} != {}, diff = {}",
                    i, val1, val2, (val1 - val2).abs()
                );
            }
        }
    }

    // Property test: scale() with 1.0 is identity
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_scale_identity(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.scale(1.0).unwrap();

            for (i, (&original, &scaled)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (original - scaled).abs() < 1e-5,
                    "Identity failed at {}: {} != {}",
                    i, original, scaled
                );
            }
        }
    }

    // Property test: scale() with 0.0 gives zeros
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_scale_zero(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.scale(0.0).unwrap();

            for (i, &val) in result.as_slice().iter().enumerate() {
                prop_assert!(
                    val.abs() < 1e-10,
                    "Zero scaling failed at {}: {} != 0.0",
                    i, val
                );
            }
        }
    }

    // Property test: scale() associativity
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_scale_associative(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            scalar1 in -10.0f32..10.0,
            scalar2 in -10.0f32..10.0
        ) {
            // (a * s1) * s2 = a * (s1 * s2)
            let va = Vector::from_slice(&a);
            let scaled_once = va.scale(scalar1).unwrap();
            let scaled_twice = scaled_once.scale(scalar2).unwrap();

            let combined_scalar = scalar1 * scalar2;
            let scaled_combined = va.scale(combined_scalar).unwrap();

            for (i, (&val1, &val2)) in scaled_twice.as_slice().iter()
                .zip(scaled_combined.as_slice().iter())
                .enumerate() {
                let tolerance = if val1.abs() > 1.0 {
                    val1.abs() * 1e-4  // Slightly higher tolerance for double scaling
                } else {
                    1e-3
                };
                prop_assert!(
                    (val1 - val2).abs() < tolerance,
                    "Associativity failed at {}: {} != {}, diff = {}",
                    i, val1, val2, (val1 - val2).abs()
                );
            }
        }
    }

    // Property test: clamp() bounds enforcement
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_clamp_bounds(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            min_val in -50.0f32..0.0,
            max_val in 0.0f32..50.0
        ) {
            let va = Vector::from_slice(&a);
            let result = va.clamp(min_val, max_val).unwrap();

            for (i, &val) in result.as_slice().iter().enumerate() {
                prop_assert!(
                    val >= min_val && val <= max_val,
                    "Value {} out of bounds [{}, {}] at index {}",
                    val, min_val, max_val, i
                );
            }
        }
    }

    // Property test: clamp() idempotence
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_clamp_idempotent(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            min_val in -50.0f32..0.0,
            max_val in 0.0f32..50.0
        ) {
            // clamp(clamp(v)) = clamp(v)
            let va = Vector::from_slice(&a);
            let clamped_once = va.clamp(min_val, max_val).unwrap();
            let clamped_twice = clamped_once.clamp(min_val, max_val).unwrap();

            for (i, (&val1, &val2)) in clamped_once.as_slice().iter()
                .zip(clamped_twice.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (val1 - val2).abs() < 1e-10,
                    "Idempotence failed at {}: {} != {}",
                    i, val1, val2
                );
            }
        }
    }

    // Property test: clamp() monotonicity
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_clamp_monotonic(
            a in prop::collection::vec(-100.0f32..100.0, 2..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.clamp(-50.0, 50.0).unwrap();

            // For any i < j, if a[i] <= a[j], then clamp(a[i]) <= clamp(a[j])
            for i in 0..a.len() - 1 {
                for j in i + 1..a.len() {
                    if a[i] <= a[j] {
                        prop_assert!(
                            result.as_slice()[i] <= result.as_slice()[j],
                            "Monotonicity violated: a[{}]={} <= a[{}]={} but clamp[{}]={} > clamp[{}]={}",
                            i, a[i], j, a[j], i, result.as_slice()[i], j, result.as_slice()[j]
                        );
                    }
                }
            }
        }
    }

    // Property test: lerp() at endpoints
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_lerp_endpoints(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            b in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let len = a.len().min(b.len());
            let a_trimmed = &a[..len];
            let b_trimmed = &b[..len];

            let va = Vector::from_slice(a_trimmed);
            let vb = Vector::from_slice(b_trimmed);

            // t=0 should return a
            let result_zero = va.lerp(&vb, 0.0).unwrap();
            for (i, (&actual, &expected)) in result_zero.as_slice().iter()
                .zip(a_trimmed.iter())
                .enumerate() {
                prop_assert!(
                    (actual - expected).abs() < 1e-5,
                    "lerp(t=0) failed at {}: {} != {}",
                    i, actual, expected
                );
            }

            // t=1 should return b
            let result_one = va.lerp(&vb, 1.0).unwrap();
            for (i, (&actual, &expected)) in result_one.as_slice().iter()
                .zip(b_trimmed.iter())
                .enumerate() {
                prop_assert!(
                    (actual - expected).abs() < 1e-5,
                    "lerp(t=1) failed at {}: {} != {}",
                    i, actual, expected
                );
            }
        }
    }

    // Property test: lerp() linearity
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_lerp_linearity(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            b in prop::collection::vec(-100.0f32..100.0, 1..100),
            t in 0.0f32..1.0
        ) {
            let len = a.len().min(b.len());
            let a_trimmed = &a[..len];
            let b_trimmed = &b[..len];

            let va = Vector::from_slice(a_trimmed);
            let vb = Vector::from_slice(b_trimmed);

            let result = va.lerp(&vb, t).unwrap();

            // Verify: result[i] = a[i] + t * (b[i] - a[i])
            for (i, ((&a_val, &b_val), &result_val)) in a_trimmed.iter()
                .zip(b_trimmed.iter())
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = a_val + t * (b_val - a_val);

                let tolerance = if expected.abs() > 1.0 {
                    expected.abs() * 1e-5
                } else {
                    1e-4
                };

                prop_assert!(
                    (result_val - expected).abs() < tolerance,
                    "Linearity failed at {}: {} != {}, diff = {}",
                    i, result_val, expected, (result_val - expected).abs()
                );
            }
        }
    }

    // Property test: lerp() symmetry
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_lerp_symmetry(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            b in prop::collection::vec(-100.0f32..100.0, 1..100),
            t in 0.0f32..1.0
        ) {
            let len = a.len().min(b.len());
            let a_trimmed = &a[..len];
            let b_trimmed = &b[..len];

            let va = Vector::from_slice(a_trimmed);
            let vb = Vector::from_slice(b_trimmed);

            // lerp(a, b, t) should equal lerp(b, a, 1-t)
            let forward = va.lerp(&vb, t).unwrap();
            let reverse = vb.lerp(&va, 1.0 - t).unwrap();

            for (i, (&fwd, &rev)) in forward.as_slice().iter()
                .zip(reverse.as_slice().iter())
                .enumerate() {
                let tolerance = if fwd.abs() > 1.0 {
                    fwd.abs() * 2e-5
                } else {
                    1e-4
                };

                prop_assert!(
                    (fwd - rev).abs() < tolerance,
                    "Symmetry failed at {}: {} != {}, diff = {}",
                    i, fwd, rev, (fwd - rev).abs()
                );
            }
        }
    }

    // Property test: fma() correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_fma_correctness(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            b in prop::collection::vec(-100.0f32..100.0, 1..100),
            c in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let len = a.len().min(b.len()).min(c.len());
            let a_trimmed = &a[..len];
            let b_trimmed = &b[..len];
            let c_trimmed = &c[..len];

            let va = Vector::from_slice(a_trimmed);
            let vb = Vector::from_slice(b_trimmed);
            let vc = Vector::from_slice(c_trimmed);

            let result = va.fma(&vb, &vc).unwrap();

            // Verify: result[i] = a[i] * b[i] + c[i]
            for (i, ((&a_val, &b_val), (&c_val, &result_val))) in a_trimmed.iter()
                .zip(b_trimmed.iter())
                .zip(c_trimmed.iter().zip(result.as_slice().iter()))
                .enumerate() {
                let expected = a_val * b_val + c_val;

                let tolerance = if expected.abs() > 1.0 {
                    expected.abs() * 1e-5
                } else {
                    1e-4
                };

                prop_assert!(
                    (result_val - expected).abs() < tolerance,
                    "FMA correctness failed at {}: {} != {}, diff = {}",
                    i, result_val, expected, (result_val - expected).abs()
                );
            }
        }
    }

    // Property test: fma() with zero multiplication
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_fma_zero_mul(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            c in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            // fma(a, 0, c) = 0 * a + c = c
            let len = a.len().min(c.len());
            let a_trimmed = &a[..len];
            let c_trimmed = &c[..len];

            let va = Vector::from_slice(a_trimmed);
            let vc = Vector::from_slice(c_trimmed);
            let zeros = vec![0.0; len];
            let vzero = Vector::from_slice(&zeros);

            let result = va.fma(&vzero, &vc).unwrap();

            for (i, (&result_val, &c_val)) in result.as_slice().iter()
                .zip(c_trimmed.iter())
                .enumerate() {
                prop_assert!(
                    (result_val - c_val).abs() < 1e-10,
                    "Zero multiplication failed at {}: {} != {}, diff = {}",
                    i, result_val, c_val, (result_val - c_val).abs()
                );
            }
        }
    }

    // Property test: fma() relation to mul and add
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_fma_vs_mul_add(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            b in prop::collection::vec(-100.0f32..100.0, 1..100),
            c in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let len = a.len().min(b.len()).min(c.len());
            let a_trimmed = &a[..len];
            let b_trimmed = &b[..len];
            let c_trimmed = &c[..len];

            let va = Vector::from_slice(a_trimmed);
            let vb = Vector::from_slice(b_trimmed);
            let vc = Vector::from_slice(c_trimmed);

            // fma(a, b, c) should approximately equal mul(a, b) + c
            let fma_result = va.fma(&vb, &vc).unwrap();
            let mul_result = va.mul(&vb).unwrap();
            let add_result = mul_result.add(&vc).unwrap();

            for (i, (&fma_val, &add_val)) in fma_result.as_slice().iter()
                .zip(add_result.as_slice().iter())
                .enumerate() {
                // FMA can have better accuracy, so use slightly higher tolerance
                let tolerance = if fma_val.abs() > 1.0 {
                    fma_val.abs() * 1e-5
                } else {
                    1e-4
                };

                prop_assert!(
                    (fma_val - add_val).abs() < tolerance,
                    "FMA vs mul+add failed at {}: {} != {}, diff = {}",
                    i, fma_val, add_val, (fma_val - add_val).abs()
                );
            }
        }
    }

    // Property test: sqrt() correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sqrt_correctness(
            a in prop::collection::vec(0.0f32..100.0, 1..100)  // Non-negative values only
        ) {
            let va = Vector::from_slice(&a);
            let result = va.sqrt().unwrap();

            // Verify: result[i] = sqrt(a[i])
            for (i, (&a_val, &result_val)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = a_val.sqrt();

                let tolerance = if expected.abs() > 1.0 {
                    expected.abs() * 1e-6
                } else {
                    1e-6
                };

                prop_assert!(
                    (result_val - expected).abs() < tolerance,
                    "sqrt correctness failed at {}: {} != {}, diff = {}",
                    i, result_val, expected, (result_val - expected).abs()
                );
            }
        }
    }

    // Property test: sqrt() idempotence with squaring
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sqrt_inverse_square(
            a in prop::collection::vec(0.0f32..100.0, 1..100)
        ) {
            // sqrt(a)^2 = a
            let va = Vector::from_slice(&a);
            let sqrt_result = va.sqrt().unwrap();
            let squared = sqrt_result.mul(&sqrt_result).unwrap();

            for (i, (&original, &recovered)) in a.iter()
                .zip(squared.as_slice().iter())
                .enumerate() {
                let tolerance = if original.abs() > 1.0 {
                    original.abs() * 1e-5
                } else {
                    1e-5
                };

                prop_assert!(
                    (original - recovered).abs() < tolerance,
                    "sqrt inverse failed at {}: {} != {}, diff = {}",
                    i, original, recovered, (original - recovered).abs()
                );
            }
        }
    }

    // Property test: sqrt() monotonicity
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sqrt_monotonic(
            a in prop::collection::vec(0.0f32..100.0, 2..100)
        ) {
            // If a[i] < a[j], then sqrt(a[i]) < sqrt(a[j])
            let va = Vector::from_slice(&a);
            let result = va.sqrt().unwrap();
            let result_slice = result.as_slice();

            for i in 0..a.len()-1 {
                for j in i+1..a.len() {
                    if a[i] < a[j] {
                        prop_assert!(
                            result_slice[i] < result_slice[j],
                            "Monotonicity failed: sqrt({}) = {} should be < sqrt({}) = {}",
                            a[i], result_slice[i], a[j], result_slice[j]
                        );
                    } else if a[i] > a[j] {
                        prop_assert!(
                            result_slice[i] > result_slice[j],
                            "Monotonicity failed: sqrt({}) = {} should be > sqrt({}) = {}",
                            a[i], result_slice[i], a[j], result_slice[j]
                        );
                    }
                }
            }
        }
    }

    // Property test: recip() correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_recip_correctness(
            a in prop::collection::vec(0.1f32..100.0, 1..100)  // Avoid zeros and very small values
        ) {
            let va = Vector::from_slice(&a);
            let result = va.recip().unwrap();

            // Verify: result[i] = 1 / a[i]
            for (i, (&a_val, &result_val)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = 1.0 / a_val;

                let tolerance = if expected.abs() > 1.0 {
                    expected.abs() * 1e-6
                } else {
                    1e-6
                };

                prop_assert!(
                    (result_val - expected).abs() < tolerance,
                    "recip correctness failed at {}: {} != {}, diff = {}",
                    i, result_val, expected, (result_val - expected).abs()
                );
            }
        }
    }

    // Property test: recip() inverse property
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_recip_inverse(
            a in prop::collection::vec(0.1f32..100.0, 1..100)
        ) {
            // recip(recip(a)) = a
            let va = Vector::from_slice(&a);
            let recip_once = va.recip().unwrap();
            let recip_twice = recip_once.recip().unwrap();

            for (i, (&original, &recovered)) in a.iter()
                .zip(recip_twice.as_slice().iter())
                .enumerate() {
                let tolerance = if original.abs() > 1.0 {
                    original.abs() * 1e-5
                } else {
                    1e-5
                };

                prop_assert!(
                    (original - recovered).abs() < tolerance,
                    "recip inverse failed at {}: {} != {}, diff = {}",
                    i, original, recovered, (original - recovered).abs()
                );
            }
        }
    }

    // Property test: recip() relation to division
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_recip_vs_division(
            a in prop::collection::vec(0.1f32..100.0, 1..100),
            scalar in 0.1f32..100.0
        ) {
            // scalar * recip(a) should equal scalar / a
            let va = Vector::from_slice(&a);
            let recip_result = va.recip().unwrap();
            let scaled = recip_result.scale(scalar).unwrap();

            for (i, (&a_val, &scaled_val)) in a.iter()
                .zip(scaled.as_slice().iter())
                .enumerate() {
                let expected = scalar / a_val;

                let tolerance = if expected.abs() > 1.0 {
                    expected.abs() * 1e-5
                } else {
                    1e-5
                };

                prop_assert!(
                    (scaled_val - expected).abs() < tolerance,
                    "recip vs division failed at {}: {} != {}, diff = {}",
                    i, scaled_val, expected, (scaled_val - expected).abs()
                );
            }
        }
    }

    // Property test: pow() correctness vs f32::powf()
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_pow_correctness(
            a in prop::collection::vec(0.1f32..100.0, 1..100),
            n in -3.0f32..3.0
        ) {
            let va = Vector::from_slice(&a);
            let result = va.pow(n).unwrap();

            for (i, (&a_val, &pow_val)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = a_val.powf(n);

                let tolerance = if expected.abs() > 1.0 {
                    expected.abs() * 1e-4
                } else {
                    1e-4
                };

                prop_assert!(
                    (pow_val - expected).abs() < tolerance,
                    "pow correctness failed at {}: {} != {}, diff = {}",
                    i, pow_val, expected, (pow_val - expected).abs()
                );
            }
        }
    }

    // Property test: Power laws
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_pow_power_laws(
            a in prop::collection::vec(1.0f32..10.0, 1..50),
            n in 1.0f32..3.0,
            m in 1.0f32..3.0
        ) {
            // Test: (x^n)^m = x^(n*m)
            let va = Vector::from_slice(&a);
            let pow_n = va.pow(n).unwrap();
            let pow_n_then_m = pow_n.pow(m).unwrap();
            let pow_nm = va.pow(n * m).unwrap();

            for (i, (&expected, &actual)) in pow_nm.as_slice().iter()
                .zip(pow_n_then_m.as_slice().iter())
                .enumerate() {
                let tolerance = if expected.abs() > 1.0 {
                    expected.abs() * 1e-3
                } else {
                    1e-3
                };

                prop_assert!(
                    (expected - actual).abs() < tolerance,
                    "pow power law failed at {}: {} != {}, diff = {}",
                    i, expected, actual, (expected - actual).abs()
                );
            }
        }
    }

    // Property test: pow() special cases
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_pow_special_cases(
            a in prop::collection::vec(0.1f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);

            // x^0 = 1
            let pow_zero = va.pow(0.0).unwrap();
            for &val in pow_zero.as_slice() {
                prop_assert!((val - 1.0).abs() < 1e-5, "x^0 should be 1");
            }

            // x^1 = x
            let pow_one = va.pow(1.0).unwrap();
            for (i, (&original, &pow_val)) in a.iter()
                .zip(pow_one.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (original - pow_val).abs() < 1e-5,
                    "x^1 failed at {}: {} != {}",
                    i, original, pow_val
                );
            }

            // x^0.5 should equal sqrt(x)
            let pow_half = va.pow(0.5).unwrap();
            let sqrt_result = va.sqrt().unwrap();
            for (i, (&pow_val, &sqrt_val)) in pow_half.as_slice().iter()
                .zip(sqrt_result.as_slice().iter())
                .enumerate() {
                let tolerance = if sqrt_val.abs() > 1.0 {
                    sqrt_val.abs() * 1e-5
                } else {
                    1e-5
                };
                prop_assert!(
                    (pow_val - sqrt_val).abs() < tolerance,
                    "x^0.5 vs sqrt failed at {}: {} != {}",
                    i, pow_val, sqrt_val
                );
            }
        }
    }

    // Property test: exp() correctness vs f32::exp()
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_exp_correctness(
            a in prop::collection::vec(-10.0f32..10.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.exp().unwrap();

            for (i, (&a_val, &exp_val)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = a_val.exp();

                let tolerance = if expected.abs() > 1.0 {
                    expected.abs() * 1e-5
                } else {
                    1e-5
                };

                prop_assert!(
                    (exp_val - expected).abs() < tolerance,
                    "exp correctness failed at {}: {} != {}, diff = {}",
                    i, exp_val, expected, (exp_val - expected).abs()
                );
            }
        }
    }

    // Property test: exp() identity - exp(0) = 1
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_exp_zero_identity(
            len in 1usize..100
        ) {
            let zeros = vec![0.0f32; len];
            let va = Vector::from_slice(&zeros);
            let result = va.exp().unwrap();

            for (i, &val) in result.as_slice().iter().enumerate() {
                prop_assert!(
                    (val - 1.0).abs() < 1e-5,
                    "exp(0) identity failed at {}: {} != 1.0",
                    i, val
                );
            }
        }
    }

    // Property test: exp() relation to addition - exp(a+b) = exp(a) * exp(b)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_exp_addition_property(
            a in prop::collection::vec(-5.0f32..5.0, 1..50),
            b in prop::collection::vec(-5.0f32..5.0, 1..50)
        ) {
            let len = a.len().min(b.len());
            let a_vec: Vec<f32> = a.into_iter().take(len).collect();
            let b_vec: Vec<f32> = b.into_iter().take(len).collect();

            let va = Vector::from_slice(&a_vec);
            let vb = Vector::from_slice(&b_vec);

            // exp(a + b)
            let sum = va.add(&vb).unwrap();
            let exp_sum = sum.exp().unwrap();

            // exp(a) * exp(b)
            let exp_a = va.exp().unwrap();
            let exp_b = vb.exp().unwrap();
            let product = exp_a.mul(&exp_b).unwrap();

            for (i, (&exp_sum_val, &product_val)) in exp_sum.as_slice().iter()
                .zip(product.as_slice().iter())
                .enumerate() {
                let tolerance = if exp_sum_val.abs() > 1.0 {
                    exp_sum_val.abs() * 1e-4
                } else {
                    1e-4
                };

                prop_assert!(
                    (exp_sum_val - product_val).abs() < tolerance,
                    "exp(a+b) = exp(a)*exp(b) failed at {}: {} != {}, diff = {}",
                    i, exp_sum_val, product_val, (exp_sum_val - product_val).abs()
                );
            }
        }
    }

    // Property test: ln() correctness vs f32::ln()
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_ln_correctness(
            a in prop::collection::vec(0.1f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.ln().unwrap();

            for (i, (&a_val, &ln_val)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = a_val.ln();

                let tolerance = if expected.abs() > 1.0 {
                    expected.abs() * 1e-5
                } else {
                    1e-5
                };

                prop_assert!(
                    (ln_val - expected).abs() < tolerance,
                    "ln correctness failed at {}: {} != {}, diff = {}",
                    i, ln_val, expected, (ln_val - expected).abs()
                );
            }
        }
    }

    // Property test: ln() inverse of exp() - ln(exp(x)) = x
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_ln_inverse_exp_property(
            a in prop::collection::vec(-5.0f32..5.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let exp_result = va.exp().unwrap();
            let ln_result = exp_result.ln().unwrap();

            for (i, (&original, &recovered)) in a.iter()
                .zip(ln_result.as_slice().iter())
                .enumerate() {
                let tolerance = if original.abs() > 1.0 {
                    original.abs() * 1e-4
                } else {
                    1e-4
                };

                prop_assert!(
                    (original - recovered).abs() < tolerance,
                    "ln(exp(x)) != x at {}: {} != {}, diff = {}",
                    i, original, recovered, (original - recovered).abs()
                );
            }
        }
    }

    // Property test: ln() product rule - ln(a*b) = ln(a) + ln(b)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_ln_product_rule(
            a in prop::collection::vec(0.1f32..10.0, 1..50),
            b in prop::collection::vec(0.1f32..10.0, 1..50)
        ) {
            let len = a.len().min(b.len());
            let a_vec: Vec<f32> = a.into_iter().take(len).collect();
            let b_vec: Vec<f32> = b.into_iter().take(len).collect();

            let va = Vector::from_slice(&a_vec);
            let vb = Vector::from_slice(&b_vec);

            // ln(a * b)
            let product = va.mul(&vb).unwrap();
            let ln_product = product.ln().unwrap();

            // ln(a) + ln(b)
            let ln_a = va.ln().unwrap();
            let ln_b = vb.ln().unwrap();
            let sum = ln_a.add(&ln_b).unwrap();

            for (i, (&ln_prod_val, &sum_val)) in ln_product.as_slice().iter()
                .zip(sum.as_slice().iter())
                .enumerate() {
                let tolerance = if ln_prod_val.abs() > 1.0 {
                    ln_prod_val.abs() * 1e-4
                } else {
                    1e-4
                };

                prop_assert!(
                    (ln_prod_val - sum_val).abs() < tolerance,
                    "ln(a*b) = ln(a)+ln(b) failed at {}: {} != {}, diff = {}",
                    i, ln_prod_val, sum_val, (ln_prod_val - sum_val).abs()
                );
            }
        }
    }

    // Property test: sin() correctness vs f32::sin()
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sin_correctness(
            a in prop::collection::vec(-10.0f32..10.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.sin().unwrap();

            for (i, (&a_val, &sin_val)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = a_val.sin();

                prop_assert!(
                    (sin_val - expected).abs() < 1e-5,
                    "sin correctness failed at {}: {} != {}, diff = {}",
                    i, sin_val, expected, (sin_val - expected).abs()
                );
            }
        }
    }

    // Property test: sin() range [-1, 1]
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sin_range(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.sin().unwrap();

            for (i, &sin_val) in result.as_slice().iter().enumerate() {
                prop_assert!(
                    (-1.0..=1.0).contains(&sin_val),
                    "sin range failed at {}: {} not in [-1, 1]",
                    i, sin_val
                );
            }
        }
    }

    // Property test: sin() periodicity - sin(x + 2π) = sin(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sin_periodicity_property(
            a in prop::collection::vec(-5.0f32..5.0, 1..50)
        ) {
            use std::f32::consts::PI;

            let va = Vector::from_slice(&a);
            let sin_a = va.sin().unwrap();

            // Add 2π to each element
            let a_plus_2pi: Vec<f32> = a.iter().map(|&x| x + 2.0 * PI).collect();
            let va_plus_2pi = Vector::from_slice(&a_plus_2pi);
            let sin_a_plus_2pi = va_plus_2pi.sin().unwrap();

            for (i, (&sin_val, &sin_periodic_val)) in sin_a.as_slice().iter()
                .zip(sin_a_plus_2pi.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (sin_val - sin_periodic_val).abs() < 1e-5,
                    "sin periodicity failed at {}: {} != {}, diff = {}",
                    i, sin_val, sin_periodic_val, (sin_val - sin_periodic_val).abs()
                );
            }
        }
    }

    // Property test: cos() correctness vs f32::cos()
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_cos_correctness(
            a in prop::collection::vec(-10.0f32..10.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.cos().unwrap();

            for (i, (&a_val, &cos_val)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = a_val.cos();

                prop_assert!(
                    (cos_val - expected).abs() < 1e-5,
                    "cos correctness failed at {}: {} != {}, diff = {}",
                    i, cos_val, expected, (cos_val - expected).abs()
                );
            }
        }
    }

    // Property test: cos() range [-1, 1]
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_cos_range(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.cos().unwrap();

            for (i, &cos_val) in result.as_slice().iter().enumerate() {
                prop_assert!(
                    (-1.0..=1.0).contains(&cos_val),
                    "cos range failed at {}: {} not in [-1, 1]",
                    i, cos_val
                );
            }
        }
    }

    // Property test: Pythagorean identity - sin²(x) + cos²(x) = 1
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_pythagorean_identity(
            a in prop::collection::vec(-10.0f32..10.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let sin_result = va.sin().unwrap();
            let cos_result = va.cos().unwrap();

            for (i, (&sin_val, &cos_val)) in sin_result.as_slice().iter()
                .zip(cos_result.as_slice().iter())
                .enumerate() {
                let sum_of_squares = sin_val * sin_val + cos_val * cos_val;

                prop_assert!(
                    (sum_of_squares - 1.0).abs() < 1e-5,
                    "Pythagorean identity failed at {}: sin²+cos² = {} != 1.0",
                    i, sum_of_squares
                );
            }
        }
    }

    // Property test: tan() correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_tan_correctness(
            a in prop::collection::vec(-1.5f32..1.5, 1..100)
        ) {
            // Use limited range to avoid tan asymptotes at ±π/2
            let va = Vector::from_slice(&a);
            let result = va.tan().unwrap();

            for (i, (&a_val, &tan_val)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = a_val.tan();

                prop_assert!(
                    (tan_val - expected).abs() < 1e-5,
                    "tan correctness failed at {}: {} != {}, diff = {}",
                    i, tan_val, expected, (tan_val - expected).abs()
                );
            }
        }
    }

    // Property test: tan(x) = sin(x) / cos(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_tan_sin_cos_identity(
            a in prop::collection::vec(-1.5f32..1.5, 1..100)
        ) {
            // Avoid asymptotes at ±π/2 where cos(x) ≈ 0
            let va = Vector::from_slice(&a);
            let tan_result = va.tan().unwrap();
            let sin_result = va.sin().unwrap();
            let cos_result = va.cos().unwrap();

            for (i, ((&tan_val, &sin_val), &cos_val)) in tan_result.as_slice().iter()
                .zip(sin_result.as_slice().iter())
                .zip(cos_result.as_slice().iter())
                .enumerate() {
                // Skip values where cos is very small (near asymptote)
                if cos_val.abs() > 1e-3 {
                    let expected = sin_val / cos_val;
                    prop_assert!(
                        (tan_val - expected).abs() < 1e-4,
                        "tan(x) != sin(x)/cos(x) at {}: {} != {}, cos={}",
                        i, tan_val, expected, cos_val
                    );
                }
            }
        }
    }

    // Property test: tan is odd function - tan(-x) = -tan(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_tan_odd_function(
            a in prop::collection::vec(-1.5f32..1.5, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let tan_pos = va.tan().unwrap();

            let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
            let va_neg = Vector::from_slice(&a_neg);
            let tan_neg = va_neg.tan().unwrap();

            for (i, (&pos, &neg)) in tan_pos.as_slice().iter()
                .zip(tan_neg.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (pos + neg).abs() < 1e-5,
                    "tan odd function failed at {}: tan(-x)={} != -tan(x)={}",
                    i, neg, -pos
                );
            }
        }
    }

    // Property test: asin() correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_asin_correctness(
            a in prop::collection::vec(-1.0f32..1.0, 1..100)
        ) {
            // Domain is [-1, 1] for asin
            let va = Vector::from_slice(&a);
            let result = va.asin().unwrap();

            for (i, (&a_val, &asin_val)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = a_val.asin();

                prop_assert!(
                    (asin_val - expected).abs() < 1e-5,
                    "asin correctness failed at {}: {} != {}, diff = {}",
                    i, asin_val, expected, (asin_val - expected).abs()
                );
            }
        }
    }

    // Property test: asin(sin(x)) = x for x in [-π/2, π/2]
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_asin_sin_inverse(
            a in prop::collection::vec(-1.5f32..1.5, 1..100)
        ) {
            // Test range within [-π/2, π/2] to ensure inverse property
            let va = Vector::from_slice(&a);
            let sin_result = va.sin().unwrap();
            let asin_result = sin_result.asin().unwrap();

            for (i, (&original, &reconstructed)) in a.iter()
                .zip(asin_result.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (original - reconstructed).abs() < 1e-5,
                    "asin(sin(x)) != x at {}: {} != {}",
                    i, reconstructed, original
                );
            }
        }
    }

    // Property test: asin is odd function - asin(-x) = -asin(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_asin_odd_function(
            a in prop::collection::vec(-1.0f32..1.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let asin_pos = va.asin().unwrap();

            let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
            let va_neg = Vector::from_slice(&a_neg);
            let asin_neg = va_neg.asin().unwrap();

            for (i, (&pos, &neg)) in asin_pos.as_slice().iter()
                .zip(asin_neg.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (pos + neg).abs() < 1e-5,
                    "asin odd function failed at {}: asin(-x)={} != -asin(x)={}",
                    i, neg, -pos
                );
            }
        }
    }

    // Property test: acos() correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_acos_correctness(
            a in prop::collection::vec(-1.0f32..1.0, 1..100)
        ) {
            // Domain is [-1, 1] for acos
            let va = Vector::from_slice(&a);
            let result = va.acos().unwrap();

            for (i, (&a_val, &acos_val)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = a_val.acos();

                prop_assert!(
                    (acos_val - expected).abs() < 1e-5,
                    "acos correctness failed at {}: {} != {}, diff = {}",
                    i, acos_val, expected, (acos_val - expected).abs()
                );
            }
        }
    }

    // Property test: acos(cos(x)) = x for x in [0, π]
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_acos_cos_inverse(
            a in prop::collection::vec(0.0f32..std::f32::consts::PI, 1..100)
        ) {
            // Test range within [0, π] to ensure inverse property
            let va = Vector::from_slice(&a);
            let cos_result = va.cos().unwrap();
            let acos_result = cos_result.acos().unwrap();

            for (i, (&original, &reconstructed)) in a.iter()
                .zip(acos_result.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (original - reconstructed).abs() < 3e-4,
                    "acos(cos(x)) != x at {}: {} != {}",
                    i, reconstructed, original
                );
            }
        }
    }

    // Property test: acos symmetry - acos(-x) = π - acos(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_acos_symmetry(
            a in prop::collection::vec(-1.0f32..1.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let acos_pos = va.acos().unwrap();

            let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
            let va_neg = Vector::from_slice(&a_neg);
            let acos_neg = va_neg.acos().unwrap();

            for (i, (&pos, &neg)) in acos_pos.as_slice().iter()
                .zip(acos_neg.as_slice().iter())
                .enumerate() {
                let expected = std::f32::consts::PI - pos;
                prop_assert!(
                    (neg - expected).abs() < 1e-5,
                    "acos symmetry failed at {}: acos(-x)={} != π - acos(x)={}",
                    i, neg, expected
                );
            }
        }
    }

    // Property test: atan() correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_atan_correctness(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            // atan accepts all real numbers
            let va = Vector::from_slice(&a);
            let result = va.atan().unwrap();

            for (i, (&a_val, &atan_val)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = a_val.atan();

                prop_assert!(
                    (atan_val - expected).abs() < 1e-5,
                    "atan correctness failed at {}: {} != {}, diff = {}",
                    i, atan_val, expected, (atan_val - expected).abs()
                );
            }
        }
    }

    // Property test: atan(tan(x)) = x for x in (-π/2, π/2)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_atan_tan_inverse(
            a in prop::collection::vec(-1.5f32..1.5, 1..100)
        ) {
            // Test range within (-π/2, π/2) to ensure inverse property
            let va = Vector::from_slice(&a);
            let tan_result = va.tan().unwrap();
            let atan_result = tan_result.atan().unwrap();

            for (i, (&original, &reconstructed)) in a.iter()
                .zip(atan_result.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (original - reconstructed).abs() < 1e-5,
                    "atan(tan(x)) != x at {}: {} != {}",
                    i, reconstructed, original
                );
            }
        }
    }

    // Property test: atan is odd function - atan(-x) = -atan(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_atan_odd_function(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let atan_pos = va.atan().unwrap();

            let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
            let va_neg = Vector::from_slice(&a_neg);
            let atan_neg = va_neg.atan().unwrap();

            for (i, (&pos, &neg)) in atan_pos.as_slice().iter()
                .zip(atan_neg.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (pos + neg).abs() < 1e-5,
                    "atan odd function failed at {}: atan(-x)={} != -atan(x)={}",
                    i, neg, -pos
                );
            }
        }
    }

    // Property test: sinh correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sinh_correctness(
            a in prop::collection::vec(-10.0f32..10.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.sinh().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = input.sinh();
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "sinh failed at {}: {} != {}",
                    i, output, expected
                );
            }
        }
    }

    // Property test: sinh is odd function - sinh(-x) = -sinh(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sinh_odd_function(
            a in prop::collection::vec(-10.0f32..10.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let sinh_pos = va.sinh().unwrap();

            let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
            let va_neg = Vector::from_slice(&a_neg);
            let sinh_neg = va_neg.sinh().unwrap();

            for (i, (&pos, &neg)) in sinh_pos.as_slice().iter()
                .zip(sinh_neg.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (pos + neg).abs() < 1e-4,
                    "sinh odd function failed at {}: sinh(-x)={} != -sinh(x)={}",
                    i, neg, -pos
                );
            }
        }
    }

    // Property test: sinh definition - sinh(x) = (e^x - e^(-x)) / 2
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sinh_definition(
            a in prop::collection::vec(-5.0f32..5.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.sinh().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = (input.exp() - (-input).exp()) / 2.0;
                // Use relative tolerance for larger values
                let tolerance = if expected.abs() > 1.0 {
                    expected.abs() * 1e-5
                } else {
                    1e-5
                };
                prop_assert!(
                    (output - expected).abs() < tolerance,
                    "sinh definition failed at {}: {} != {} (input: {})",
                    i, output, expected, input
                );
            }
        }
    }

    // Property test: cosh correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_cosh_correctness(
            a in prop::collection::vec(-10.0f32..10.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.cosh().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = input.cosh();
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "cosh failed at {}: {} != {}",
                    i, output, expected
                );
            }
        }
    }

    // Property test: cosh is even function - cosh(-x) = cosh(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_cosh_even_function(
            a in prop::collection::vec(-10.0f32..10.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let cosh_pos = va.cosh().unwrap();

            let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
            let va_neg = Vector::from_slice(&a_neg);
            let cosh_neg = va_neg.cosh().unwrap();

            for (i, (&pos, &neg)) in cosh_pos.as_slice().iter()
                .zip(cosh_neg.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (pos - neg).abs() < 1e-4,
                    "cosh even function failed at {}: cosh(-x)={} != cosh(x)={}",
                    i, neg, pos
                );
            }
        }
    }

    // Property test: cosh definition - cosh(x) = (e^x + e^(-x)) / 2
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_cosh_definition(
            a in prop::collection::vec(-5.0f32..5.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.cosh().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = (input.exp() + (-input).exp()) / 2.0;
                // Use relative tolerance for larger values
                let tolerance = if expected.abs() > 1.0 {
                    expected.abs() * 1e-5
                } else {
                    1e-5
                };
                prop_assert!(
                    (output - expected).abs() < tolerance,
                    "cosh definition failed at {}: {} != {} (input: {})",
                    i, output, expected, input
                );
            }
        }
    }

    // Property test: hyperbolic identity - cosh^2(x) - sinh^2(x) = 1
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_cosh_sinh_identity(
            a in prop::collection::vec(-5.0f32..5.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let cosh_result = va.cosh().unwrap();
            let sinh_result = va.sinh().unwrap();

            for (i, (&c, &s)) in cosh_result.as_slice().iter()
                .zip(sinh_result.as_slice().iter())
                .enumerate() {
                let identity = c * c - s * s;
                // Use relative tolerance for numerical stability
                // Since we're computing c^2 - s^2, tolerance scales with squared values
                let max_squared = c.abs().max(s.abs()).powi(2);
                let tolerance = if max_squared > 1.0 {
                    max_squared * 1e-4
                } else {
                    1e-5
                };
                prop_assert!(
                    (identity - 1.0).abs() < tolerance,
                    "Hyperbolic identity failed at {}: cosh^2({}) - sinh^2({}) = {} != 1",
                    i, c, s, identity
                );
            }
        }
    }

    // Property test: tanh correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_tanh_correctness(
            a in prop::collection::vec(-10.0f32..10.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.tanh().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = input.tanh();
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "tanh failed at {}: {} != {}",
                    i, output, expected
                );
            }
        }
    }

    // Property test: tanh is odd function - tanh(-x) = -tanh(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_tanh_odd_function(
            a in prop::collection::vec(-10.0f32..10.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let tanh_pos = va.tanh().unwrap();

            let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
            let va_neg = Vector::from_slice(&a_neg);
            let tanh_neg = va_neg.tanh().unwrap();

            for (i, (&pos, &neg)) in tanh_pos.as_slice().iter()
                .zip(tanh_neg.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (pos + neg).abs() < 1e-5,
                    "tanh odd function failed at {}: tanh(-x)={} != -tanh(x)={}",
                    i, neg, -pos
                );
            }
        }
    }

    // Property test: tanh = sinh/cosh relation
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_tanh_sinh_cosh_relation(
            a in prop::collection::vec(-5.0f32..5.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let tanh_result = va.tanh().unwrap();
            let sinh_result = va.sinh().unwrap();
            let cosh_result = va.cosh().unwrap();

            for (i, (&t, (&s, &c))) in tanh_result.as_slice().iter()
                .zip(sinh_result.as_slice().iter().zip(cosh_result.as_slice().iter()))
                .enumerate() {
                let ratio = s / c;
                // Use relative tolerance for numerical stability
                let tolerance = if ratio.abs() > 1.0 {
                    ratio.abs() * 1e-5
                } else {
                    1e-5
                };
                prop_assert!(
                    (t - ratio).abs() < tolerance,
                    "tanh = sinh/cosh failed at {}: tanh({}) = {} != {}/{}={}",
                    i, t, t, s, c, ratio
                );
            }
        }
    }

    // Property test: tanh range bound -1 <= tanh(x) <= 1
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_tanh_range_bound(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.tanh().unwrap();

            for (i, &output) in result.as_slice().iter().enumerate() {
                prop_assert!(
                    (-1.0..=1.0).contains(&output),
                    "tanh range bound failed at {}: {} not in [-1, 1]",
                    i, output
                );
            }
        }
    }

    // Property test: asinh correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_asinh_correctness(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.asinh().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = input.asinh();
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "asinh failed at {}: {} != {}",
                    i, output, expected
                );
            }
        }
    }

    // Property test: asinh is odd function - asinh(-x) = -asinh(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_asinh_odd_function(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let asinh_pos = va.asinh().unwrap();

            let a_neg: Vec<f32> = a.iter().map(|x| -x).collect();
            let va_neg = Vector::from_slice(&a_neg);
            let asinh_neg = va_neg.asinh().unwrap();

            for (i, (&pos, &neg)) in asinh_pos.as_slice().iter()
                .zip(asinh_neg.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (pos + neg).abs() < 1e-5,
                    "asinh odd function failed at {}: asinh(-x)={} != -asinh(x)={}",
                    i, neg, -pos
                );
            }
        }
    }

    // Property test: asinh(sinh(x)) = x inverse relation
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_asinh_sinh_inverse(
            a in prop::collection::vec(-5.0f32..5.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let sinh_result = va.sinh().unwrap();
            let asinh_result = sinh_result.asinh().unwrap();

            for (i, (&original, &reconstructed)) in a.iter()
                .zip(asinh_result.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (original - reconstructed).abs() < 1e-5,
                    "asinh(sinh(x)) != x at {}: {} != {}",
                    i, reconstructed, original
                );
            }
        }
    }

    // Property test: acosh correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_acosh_correctness(
            a in prop::collection::vec(1.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.acosh().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = input.acosh();
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "acosh failed at {}: {} != {}",
                    i, output, expected
                );
            }
        }
    }

    // Property test: acosh(cosh(x)) = x inverse relation for x >= 0
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_acosh_cosh_inverse(
            a in prop::collection::vec(0.1f32..5.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let cosh_result = va.cosh().unwrap();
            let acosh_result = cosh_result.acosh().unwrap();

            for (i, (&original, &reconstructed)) in a.iter()
                .zip(acosh_result.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (original - reconstructed).abs() < 1e-5,
                    "acosh(cosh(x)) != x at {}: {} != {}",
                    i, reconstructed, original
                );
            }
        }
    }

    // Property test: acosh range - output >= 0
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_acosh_range(
            a in prop::collection::vec(1.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.acosh().unwrap();

            for (i, &output) in result.as_slice().iter().enumerate() {
                prop_assert!(
                    output >= 0.0,
                    "acosh range failed at {}: {} not >= 0",
                    i, output
                );
            }
        }
    }

    // Property test: atanh correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_atanh_correctness(
            a in prop::collection::vec(-0.99f32..0.99, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.atanh().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = input.atanh();
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "atanh failed at {}: {} != {}",
                    i, output, expected
                );
            }
        }
    }

    // Property test: atanh is odd function: atanh(-x) = -atanh(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_atanh_odd_function(
            a in prop::collection::vec(-0.99f32..0.99, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let neg_a: Vec<f32> = a.iter().map(|x| -x).collect();
            let v_neg_a = Vector::from_slice(&neg_a);

            let result_a = va.atanh().unwrap();
            let result_neg_a = v_neg_a.atanh().unwrap();

            for (i, (&res_a, &res_neg_a)) in result_a.as_slice().iter()
                .zip(result_neg_a.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (res_a + res_neg_a).abs() < 1e-5,
                    "atanh(-x) != -atanh(x) at {}: {} != {}",
                    i, res_neg_a, -res_a
                );
            }
        }
    }

    // Property test: atanh(tanh(x)) = x inverse relation
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_atanh_tanh_inverse(
            a in prop::collection::vec(-5.0f32..5.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let tanh_result = va.tanh().unwrap();
            let atanh_result = tanh_result.atanh().unwrap();

            for (i, (&original, &reconstructed)) in a.iter()
                .zip(atanh_result.as_slice().iter())
                .enumerate() {
                // Use adaptive tolerance: numerical precision degrades as tanh(x) approaches ±1
                // For large |x|, tanh(x) ≈ ±1 and atanh near ±1 has compounding errors
                let tolerance = 1e-5 * (1.0 + original.abs() * 10.0);
                prop_assert!(
                    (original - reconstructed).abs() < tolerance,
                    "atanh(tanh(x)) != x at {}: {} != {}",
                    i, reconstructed, original
                );
            }
        }
    }

    // Property test: floor correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_floor_correctness(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.floor().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = input.floor();
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "floor failed at {}: {} != {}",
                    i, output, expected
                );
            }
        }
    }

    // Property test: floor idempotence - floor(floor(x)) = floor(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_floor_idempotence(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let floor_once = va.floor().unwrap();
            let floor_twice = floor_once.floor().unwrap();

            for (i, (&once, &twice)) in floor_once.as_slice().iter()
                .zip(floor_twice.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (once - twice).abs() < 1e-5,
                    "floor idempotence failed at {}: floor(floor({})) = {} != {}",
                    i, a[i], twice, once
                );
            }
        }
    }

    // Property test: floor always <= original value
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_floor_less_than_or_equal(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.floor().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    output <= input,
                    "floor should be <= input at {}: floor({}) = {} > {}",
                    i, input, output, input
                );
            }
        }
    }

    // Property test: ceil correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_ceil_correctness(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.ceil().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = input.ceil();
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "ceil failed at {}: {} != {}",
                    i, output, expected
                );
            }
        }
    }

    // Property test: ceil idempotence - ceil(ceil(x)) = ceil(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_ceil_idempotence(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let ceil_once = va.ceil().unwrap();
            let ceil_twice = ceil_once.ceil().unwrap();

            for (i, (&once, &twice)) in ceil_once.as_slice().iter()
                .zip(ceil_twice.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (once - twice).abs() < 1e-5,
                    "ceil idempotence failed at {}: ceil(ceil({})) = {} != {}",
                    i, a[i], twice, once
                );
            }
        }
    }

    // Property test: ceil always >= original value
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_ceil_greater_than_or_equal(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.ceil().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    output >= input,
                    "ceil should be >= input at {}: ceil({}) = {} < {}",
                    i, input, output, input
                );
            }
        }
    }

    // Property test: round correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_round_correctness(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.round().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = input.round();
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "round failed at {}: {} != {}",
                    i, output, expected
                );
            }
        }
    }

    // Property test: round idempotence - round(round(x)) = round(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_round_idempotence(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let round_once = va.round().unwrap();
            let round_twice = round_once.round().unwrap();

            for (i, (&once, &twice)) in round_once.as_slice().iter()
                .zip(round_twice.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (once - twice).abs() < 1e-5,
                    "round idempotence failed at {}: round(round({})) = {} != {}",
                    i, a[i], twice, once
                );
            }
        }
    }

    // Property test: round distance - |round(x) - x| <= 0.5
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_round_distance(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.round().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let distance = (output - input).abs();
                prop_assert!(
                    distance <= 0.5 + 1e-5,  // Small epsilon for floating point precision
                    "round distance should be <= 0.5 at {}: |round({}) - {}| = {} > 0.5",
                    i, input, input, distance
                );
            }
        }
    }

    // Property test: trunc correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_trunc_correctness(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.trunc().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = input.trunc();
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "trunc failed at {}: {} != {}",
                    i, output, expected
                );
            }
        }
    }

    // Property test: trunc idempotence - trunc(trunc(x)) = trunc(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_trunc_idempotence(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let trunc_once = va.trunc().unwrap();
            let trunc_twice = trunc_once.trunc().unwrap();

            for (i, (&once, &twice)) in trunc_once.as_slice().iter()
                .zip(trunc_twice.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (once - twice).abs() < 1e-5,
                    "trunc idempotence failed at {}: trunc(trunc({})) = {} != {}",
                    i, a[i], twice, once
                );
            }
        }
    }

    // Property test: trunc moves toward zero - |trunc(x)| <= |x|
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_trunc_toward_zero(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.trunc().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    output.abs() <= input.abs() + 1e-5,  // Small epsilon for floating point
                    "trunc should move toward zero at {}: |trunc({})| = {} > |{}| = {}",
                    i, input, output.abs(), input, input.abs()
                );
            }
        }
    }
}
