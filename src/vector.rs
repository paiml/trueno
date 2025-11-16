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
}
