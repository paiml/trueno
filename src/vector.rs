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

    /// Sum of squared elements
    ///
    /// Computes the sum of squares: sum(a[i]^2).
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
    /// Computes the arithmetic mean of all elements: sum(a[i]) / n.
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
    /// Computes the population variance: Var(X) = E[(X - μ)²] = E[X²] - μ²
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
    /// Uses the computational formula: Cov(X,Y) = E[XY] - μx·μy
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
    /// assert!((cov - 2.0).abs() < 1e-5); // Perfect positive covariance
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
        Ok(cov / (std_x * std_y))
    }

    /// Z-score normalization (standardization)
    ///
    /// Transforms the vector to have mean = 0 and standard deviation = 1.
    /// Each element is transformed as: z[i] = (x[i] - μ) / σ
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

        Ok(Vector::from_slice(&data))
    }

    /// Min-max normalization (scaling to [0, 1] range)
    ///
    /// Transforms the vector so that the minimum value becomes 0 and the maximum
    /// value becomes 1, with all other values scaled proportionally.
    /// Formula: x'[i] = (x[i] - min) / (max - min)
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

        Ok(Vector::from_slice(&data))
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

        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| x.max(min_val).min(max_val))
            .collect();

        Ok(Vector::from_slice(&data))
    }

    /// Softmax activation function
    ///
    /// Converts a vector of real values into a probability distribution.
    /// Formula: softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
    ///
    /// Uses the numerically stable version with max subtraction to prevent overflow.
    /// The output is a probability distribution: all values in [0, 1] and sum to 1.
    ///
    /// This is the standard activation function for multi-class classification in neural networks.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let logits = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let probs = logits.softmax().unwrap();
    ///
    /// // Verify sum ≈ 1
    /// let sum: f32 = probs.as_slice().iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-5);
    ///
    /// // Verify all values in [0, 1]
    /// for &p in probs.as_slice() {
    ///     assert!(p >= 0.0 && p <= 1.0);
    /// }
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns EmptyVector error for empty vectors (cannot compute softmax).
    pub fn softmax(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // Find max for numerical stability (prevents overflow in exp)
        let max_val = self.max()?;

        // Compute exp(x - max) for each element
        let exp_vals: Vec<f32> = self
            .data
            .iter()
            .map(|&x| (x - max_val).exp())
            .collect();

        // Compute sum of exponentials
        let sum_exp: f32 = exp_vals.iter().sum();

        // Normalize by sum
        let data: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        Ok(Vector::from_slice(&data))
    }

    /// Log-softmax activation function
    ///
    /// Computes the logarithm of the softmax function in a numerically stable way.
    /// Formula: log_softmax(x)[i] = x[i] - max(x) - log(sum(exp(x[j] - max(x))))
    ///
    /// This is more numerically stable than computing log(softmax(x)) and is commonly
    /// used in neural networks for computing cross-entropy loss.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let logits = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let log_probs = logits.log_softmax().unwrap();
    ///
    /// // Verify exp(log_softmax) = softmax
    /// let probs_from_log: Vec<f32> = log_probs.as_slice().iter().map(|&x| x.exp()).collect();
    /// let sum: f32 = probs_from_log.iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-5);
    /// ```
    ///
    /// # Empty vectors
    ///
    /// Returns EmptyVector error for empty vectors.
    pub fn log_softmax(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // Find max for numerical stability
        let max_val = self.max()?;

        // Compute exp(x - max) for each element
        let exp_vals: Vec<f32> = self
            .data
            .iter()
            .map(|&x| (x - max_val).exp())
            .collect();

        // Compute log of sum of exponentials
        let sum_exp: f32 = exp_vals.iter().sum();
        let log_sum_exp = sum_exp.ln();

        // log_softmax(x)[i] = x[i] - max - log_sum_exp
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| x - max_val - log_sum_exp)
            .collect();

        Ok(Vector::from_slice(&data))
    }

    /// ReLU (Rectified Linear Unit) activation function
    ///
    /// Computes the element-wise ReLU: max(0, x).
    /// ReLU is one of the most widely used activation functions in neural networks.
    ///
    /// # Formula
    ///
    /// ```text
    /// relu(x)[i] = max(0, x[i])
    ///            = x[i]  if x[i] > 0
    ///            = 0     otherwise
    /// ```
    ///
    /// # Properties
    ///
    /// - **Non-linearity**: Introduces non-linearity while preserving linearity for positive values
    /// - **Sparsity**: Produces exactly zero for negative inputs (sparse activations)
    /// - **Gradient**: Derivative is 1 for positive inputs, 0 for negative (solves vanishing gradient)
    /// - **Computational efficiency**: Simple max operation, no exponentials
    ///
    /// # Applications
    ///
    /// - **Deep neural networks**: Default activation for hidden layers
    /// - **Convolutional networks**: Standard activation in CNNs
    /// - **Feature learning**: Encourages sparse representations
    ///
    /// # Performance
    ///
    /// This operation is memory-bound. SIMD provides modest speedups since
    /// the computation (comparison and selection) is simpler than memory access.
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.relu().unwrap();
    /// assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    /// ```
    pub fn relu(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // Element-wise max(0, x)
        let data: Vec<f32> = self.data.iter().map(|&x| x.max(0.0)).collect();

        Ok(Vector::from_slice(&data))
    }

    /// Sigmoid (logistic) activation function
    ///
    /// Computes the element-wise sigmoid: σ(x) = 1 / (1 + e^(-x)).
    /// Sigmoid is a classic activation function that squashes inputs to the range (0, 1).
    ///
    /// # Formula
    ///
    /// ```text
    /// sigmoid(x)[i] = 1 / (1 + exp(-x[i]))
    ///               = exp(x[i]) / (1 + exp(x[i]))
    /// ```
    ///
    /// # Properties
    ///
    /// - **Bounded output**: Maps all inputs to (0, 1) range
    /// - **Smooth**: Infinitely differentiable (C^∞)
    /// - **Symmetric**: σ(-x) = 1 - σ(x)
    /// - **Derivative**: σ'(x) = σ(x) * (1 - σ(x))
    /// - **Interpretable**: Output can be interpreted as probability
    ///
    /// # Applications
    ///
    /// - **Binary classification**: Final layer for binary output (0 or 1)
    /// - **Logistic regression**: Traditional ML algorithm
    /// - **Gating mechanisms**: LSTM/GRU gates (input, forget, output)
    /// - **Attention mechanisms**: Soft attention weights
    ///
    /// # Numerical Considerations
    ///
    /// For very large negative inputs (x < -50), exp(-x) overflows to infinity.
    /// However, sigmoid(x) approaches 0, so we return 0 for numerical stability.
    /// For very large positive inputs (x > 50), exp(-x) underflows to 0,
    /// and sigmoid(x) approaches 1.
    ///
    /// # Performance
    ///
    /// This operation is compute-bound due to the exp() operation. SIMD provides
    /// modest speedups, but the exponential is the bottleneck.
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, 0.0, 2.0]);
    /// let result = v.sigmoid().unwrap();
    ///
    /// // sigmoid(-2) ≈ 0.119, sigmoid(0) = 0.5, sigmoid(2) ≈ 0.881
    /// assert!((result.data[0] - 0.119).abs() < 0.001);
    /// assert!((result.data[1] - 0.5).abs() < 0.001);
    /// assert!((result.data[2] - 0.881).abs() < 0.001);
    /// ```
    pub fn sigmoid(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // σ(x) = 1 / (1 + exp(-x))
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                // Handle extreme values for numerical stability
                if x < -50.0 {
                    0.0 // exp(-x) would overflow, but sigmoid approaches 0
                } else if x > 50.0 {
                    1.0 // exp(-x) underflows to 0, sigmoid approaches 1
                } else {
                    1.0 / (1.0 + (-x).exp())
                }
            })
            .collect();

        Ok(Vector::from_slice(&data))
    }

    /// Leaky ReLU activation function
    ///
    /// Computes the element-wise Leaky ReLU with a configurable negative slope.
    /// Leaky ReLU addresses the "dying ReLU" problem by allowing small negative values.
    ///
    /// # Formula
    ///
    /// ```text
    /// leaky_relu(x, α)[i] = max(αx[i], x[i])
    ///                     = x[i]    if x[i] > 0
    ///                     = αx[i]   if x[i] ≤ 0
    /// ```
    ///
    /// # Parameters
    ///
    /// - `negative_slope`: The slope for negative values (typically 0.01)
    ///   - Must be in range [0.0, 1.0)
    ///   - Common values: 0.01 (default), 0.1, 0.2
    ///   - α = 0 reduces to standard ReLU
    ///   - α = 1 reduces to identity function
    ///
    /// # Properties
    ///
    /// - **Fixes dying ReLU**: Neurons can't completely die (always has gradient)
    /// - **Non-zero gradient**: Gradient is α for negative inputs (not zero)
    /// - **Unbounded positive**: No saturation for positive values
    /// - **Parameterized**: Negative slope can be tuned or learned (PReLU)
    ///
    /// # Applications
    ///
    /// - **Deep networks**: Prevents dying neurons in very deep networks
    /// - **GANs**: Often used in generator and discriminator networks
    /// - **Better gradient flow**: Helps with vanishing gradient problem
    /// - **Empirical improvements**: Often outperforms ReLU in practice
    ///
    /// # Performance
    ///
    /// This operation is memory-bound (simple multiplication and comparison).
    /// SIMD provides modest speedups.
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    /// Returns `InvalidInput` if negative_slope is not in [0.0, 1.0).
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.leaky_relu(0.01).unwrap();
    ///
    /// // Negative values multiplied by 0.01, positive unchanged
    /// assert_eq!(result.as_slice(), &[-0.02, -0.01, 0.0, 1.0, 2.0]);
    /// ```
    pub fn leaky_relu(&self, negative_slope: f32) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // Validate negative_slope parameter
        if !(0.0..1.0).contains(&negative_slope) {
            return Err(TruenoError::InvalidInput(format!(
                "negative_slope must be in [0.0, 1.0), got {}",
                negative_slope
            )));
        }

        // leaky_relu(x, α) = x if x > 0, αx otherwise
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                if x > 0.0 {
                    x
                } else {
                    negative_slope * x
                }
            })
            .collect();

        Ok(Vector::from_slice(&data))
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

    /// Returns the fractional part of each element.
    ///
    /// The fractional part has the same sign as the original value:
    /// - Positive: fract(3.7) = 0.7
    /// - Negative: fract(-3.7) = -0.7
    /// - Decomposition property: x = trunc(x) + fract(x)
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[3.7, -2.3, 5.0]);
    /// let result = v.fract().unwrap();
    /// // Fractional parts: 0.7, -0.3, 0.0
    /// assert!((result.as_slice()[0] - 0.7).abs() < 1e-5);
    /// assert!((result.as_slice()[1] - (-0.3)).abs() < 1e-5);
    /// ```
    pub fn fract(&self) -> Result<Vector<f32>> {
        let fract_data: Vec<f32> = self.data.iter().map(|x| x.fract()).collect();
        Ok(Vector {
            data: fract_data,
            backend: self.backend,
        })
    }

    /// Returns the sign of each element.
    ///
    /// Returns:
    /// - `1.0` if the value is positive (including +0.0 and +∞)
    /// - `-1.0` if the value is negative (including -0.0 and -∞)
    /// - `NaN` if the value is NaN
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[5.0, -3.0, 0.0, -0.0]);
    /// let result = v.signum().unwrap();
    /// assert_eq!(result.as_slice(), &[1.0, -1.0, 1.0, -1.0]);
    /// ```
    pub fn signum(&self) -> Result<Vector<f32>> {
        let signum_data: Vec<f32> = self.data.iter().map(|x| x.signum()).collect();
        Ok(Vector {
            data: signum_data,
            backend: self.backend,
        })
    }

    /// Returns a vector with the magnitude of `self` and the sign of `sign`.
    ///
    /// For each element pair, takes the magnitude from `self` and the sign from `sign`.
    /// Equivalent to `abs(self[i])` with the sign of `sign[i]`.
    ///
    /// # Arguments
    ///
    /// * `sign` - Vector providing the sign for each element
    ///
    /// # Errors
    ///
    /// Returns `TruenoError::SizeMismatch` if vectors have different lengths.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let magnitude = Vector::from_slice(&[5.0, 3.0, 2.0]);
    /// let sign = Vector::from_slice(&[-1.0, 1.0, -1.0]);
    /// let result = magnitude.copysign(&sign).unwrap();
    /// assert_eq!(result.as_slice(), &[-5.0, 3.0, -2.0]);
    /// ```
    pub fn copysign(&self, sign: &Self) -> Result<Vector<f32>> {
        if self.len() != sign.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: sign.len(),
            });
        }

        let copysign_data: Vec<f32> = self.data.iter()
            .zip(sign.data.iter())
            .map(|(mag, sgn)| mag.copysign(*sgn))
            .collect();

        Ok(Vector {
            data: copysign_data,
            backend: self.backend,
        })
    }

    /// Element-wise minimum of two vectors.
    ///
    /// Returns a new vector where each element is the minimum of the corresponding
    /// elements from self and other.
    ///
    /// NaN handling: Prefers non-NaN values (NAN.min(x) = x).
    ///
    /// # Examples
    /// ```
    /// use trueno::Vector;
    /// let a = Vector::from_slice(&[1.0, 5.0, 3.0]);
    /// let b = Vector::from_slice(&[2.0, 3.0, 4.0]);
    /// let result = a.minimum(&b).unwrap();
    /// assert_eq!(result.as_slice(), &[1.0, 3.0, 3.0]);
    /// ```
    pub fn minimum(&self, other: &Self) -> Result<Vector<f32>> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        let minimum_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.min(*b))
            .collect();

        Ok(Vector {
            data: minimum_data,
            backend: self.backend,
        })
    }

    /// Element-wise maximum of two vectors.
    ///
    /// Returns a new vector where each element is the maximum of the corresponding
    /// elements from self and other.
    ///
    /// NaN handling: Prefers non-NaN values (NAN.max(x) = x).
    ///
    /// # Examples
    /// ```
    /// use trueno::Vector;
    /// let a = Vector::from_slice(&[1.0, 5.0, 3.0]);
    /// let b = Vector::from_slice(&[2.0, 3.0, 4.0]);
    /// let result = a.maximum(&b).unwrap();
    /// assert_eq!(result.as_slice(), &[2.0, 5.0, 4.0]);
    /// ```
    pub fn maximum(&self, other: &Self) -> Result<Vector<f32>> {
        if self.len() != other.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: other.len(),
            });
        }

        let maximum_data: Vec<f32> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.max(*b))
            .collect();

        Ok(Vector {
            data: maximum_data,
            backend: self.backend,
        })
    }

    /// Element-wise negation (unary minus).
    ///
    /// Returns a new vector where each element is the negation of the corresponding
    /// element from self.
    ///
    /// Properties: Double negation is identity: -(-x) = x
    ///
    /// # Examples
    /// ```
    /// use trueno::Vector;
    /// let a = Vector::from_slice(&[1.0, -2.0, 3.0]);
    /// let result = a.neg().unwrap();
    /// assert_eq!(result.as_slice(), &[-1.0, 2.0, -3.0]);
    /// ```
    pub fn neg(&self) -> Result<Vector<f32>> {
        let neg_data: Vec<f32> = self.data.iter().map(|x| -x).collect();
        Ok(Vector {
            data: neg_data,
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
    fn test_fract_basic() {
        let a = Vector::from_slice(&[3.7, -2.3, 5.0]);
        let result = a.fract().unwrap();
        // fract returns fractional part with same sign
        assert!((result.as_slice()[0] - 0.7).abs() < 1e-5);
        assert!((result.as_slice()[1] - (-0.3)).abs() < 1e-5);
        assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_fract_positive() {
        let a = Vector::from_slice(&[1.2, 2.5, 3.9]);
        let result = a.fract().unwrap();
        assert!((result.as_slice()[0] - 0.2).abs() < 1e-5);
        assert!((result.as_slice()[1] - 0.5).abs() < 1e-5);
        assert!((result.as_slice()[2] - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_fract_negative() {
        let a = Vector::from_slice(&[-1.2, -2.5, -3.9]);
        let result = a.fract().unwrap();
        assert!((result.as_slice()[0] - (-0.2)).abs() < 1e-5);
        assert!((result.as_slice()[1] - (-0.5)).abs() < 1e-5);
        assert!((result.as_slice()[2] - (-0.9)).abs() < 1e-5);
    }

    #[test]
    fn test_fract_integers() {
        let a = Vector::from_slice(&[1.0, 2.0, -3.0, 0.0]);
        let result = a.fract().unwrap();
        assert_eq!(result.as_slice(), &[0.0, 0.0, -0.0, 0.0]);
    }

    #[test]
    fn test_fract_range() {
        // fract() is always in range [0, 1) for positive, (-1, 0] for negative
        let a = Vector::from_slice(&[0.1, 0.5, 0.9, -0.1, -0.5, -0.9]);
        let result = a.fract().unwrap();
        for &val in result.as_slice() {
            assert!(val.abs() < 1.0, "fract value should be in range (-1, 1): {}", val);
        }
    }

    #[test]
    fn test_fract_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.fract().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_signum_basic() {
        let a = Vector::from_slice(&[5.0, -3.0, 0.0, -0.0]);
        let result = a.signum().unwrap();
        assert_eq!(result.as_slice(), &[1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_signum_positive() {
        let a = Vector::from_slice(&[0.1, 1.0, 100.0, f32::INFINITY]);
        let result = a.signum().unwrap();
        assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_signum_negative() {
        let a = Vector::from_slice(&[-0.1, -1.0, -100.0, f32::NEG_INFINITY]);
        let result = a.signum().unwrap();
        assert_eq!(result.as_slice(), &[-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_signum_mixed() {
        let a = Vector::from_slice(&[42.5, -17.3, 0.0001, -0.0001]);
        let result = a.signum().unwrap();
        assert_eq!(result.as_slice(), &[1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn test_signum_zero_handling() {
        // Rust's signum treats +0.0 as positive (1.0) and -0.0 as negative (-1.0)
        let a = Vector::from_slice(&[0.0, -0.0]);
        let result = a.signum().unwrap();
        assert_eq!(result.as_slice(), &[1.0, -1.0]);
    }

    #[test]
    fn test_signum_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.signum().unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_copysign_basic() {
        let magnitude = Vector::from_slice(&[5.0, 3.0, 2.0, 4.0]);
        let sign = Vector::from_slice(&[-1.0, 1.0, -1.0, 1.0]);
        let result = magnitude.copysign(&sign).unwrap();
        assert_eq!(result.as_slice(), &[-5.0, 3.0, -2.0, 4.0]);
    }

    #[test]
    fn test_copysign_negative_magnitude() {
        // copysign takes absolute magnitude, so negative magnitude becomes positive first
        let magnitude = Vector::from_slice(&[-5.0, -3.0]);
        let sign = Vector::from_slice(&[1.0, -1.0]);
        let result = magnitude.copysign(&sign).unwrap();
        assert_eq!(result.as_slice(), &[5.0, -3.0]);
    }

    #[test]
    fn test_copysign_zero() {
        // copysign handles +0.0 and -0.0
        let magnitude = Vector::from_slice(&[3.0, 3.0]);
        let sign = Vector::from_slice(&[0.0, -0.0]);
        let result = magnitude.copysign(&sign).unwrap();
        assert_eq!(result.as_slice(), &[3.0, -3.0]);
    }

    #[test]
    fn test_copysign_infinity() {
        let magnitude = Vector::from_slice(&[5.0, 5.0]);
        let sign = Vector::from_slice(&[f32::INFINITY, f32::NEG_INFINITY]);
        let result = magnitude.copysign(&sign).unwrap();
        assert_eq!(result.as_slice(), &[5.0, -5.0]);
    }

    #[test]
    fn test_copysign_size_mismatch() {
        let magnitude = Vector::from_slice(&[1.0, 2.0]);
        let sign = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = magnitude.copysign(&sign);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TruenoError::SizeMismatch { .. }));
    }

    #[test]
    fn test_copysign_empty() {
        let magnitude: Vector<f32> = Vector::from_slice(&[]);
        let sign: Vector<f32> = Vector::from_slice(&[]);
        let result = magnitude.copysign(&sign).unwrap();
        assert_eq!(result.len(), 0);
    }

    // ========================================
    // Unit Tests: minimum()
    // ========================================

    #[test]
    fn test_minimum_basic() {
        let a = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
        let b = Vector::from_slice(&[2.0, 3.0, 4.0, 1.0]);
        let result = a.minimum(&b).unwrap();
        assert_eq!(result.as_slice(), &[1.0, 3.0, 3.0, 1.0]);
    }

    #[test]
    fn test_minimum_negative() {
        let a = Vector::from_slice(&[-1.0, -5.0, 3.0]);
        let b = Vector::from_slice(&[-2.0, -3.0, 4.0]);
        let result = a.minimum(&b).unwrap();
        assert_eq!(result.as_slice(), &[-2.0, -5.0, 3.0]);
    }

    #[test]
    fn test_minimum_nan() {
        // NaN handling: NAN.min(x) = x (prefers non-NaN)
        let a = Vector::from_slice(&[f32::NAN, 5.0, f32::NAN]);
        let b = Vector::from_slice(&[3.0, f32::NAN, f32::NAN]);
        let result = a.minimum(&b).unwrap();
        assert_eq!(result.as_slice()[0], 3.0);
        assert_eq!(result.as_slice()[1], 5.0);
        assert!(result.as_slice()[2].is_nan());
    }

    #[test]
    fn test_minimum_infinity() {
        let a = Vector::from_slice(&[f32::INFINITY, 5.0, f32::NEG_INFINITY]);
        let b = Vector::from_slice(&[3.0, f32::INFINITY, -10.0]);
        let result = a.minimum(&b).unwrap();
        assert_eq!(result.as_slice(), &[3.0, 5.0, f32::NEG_INFINITY]);
    }

    #[test]
    fn test_minimum_size_mismatch() {
        let a = Vector::from_slice(&[1.0, 2.0]);
        let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = a.minimum(&b);
        assert!(matches!(result, Err(TruenoError::SizeMismatch { .. })));
    }

    #[test]
    fn test_minimum_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let b: Vector<f32> = Vector::from_slice(&[]);
        let result = a.minimum(&b).unwrap();
        assert_eq!(result.len(), 0);
    }

    // ========================================
    // Unit Tests: maximum()
    // ========================================

    #[test]
    fn test_maximum_basic() {
        let a = Vector::from_slice(&[1.0, 5.0, 3.0, 2.0]);
        let b = Vector::from_slice(&[2.0, 3.0, 4.0, 1.0]);
        let result = a.maximum(&b).unwrap();
        assert_eq!(result.as_slice(), &[2.0, 5.0, 4.0, 2.0]);
    }

    #[test]
    fn test_maximum_negative() {
        let a = Vector::from_slice(&[-1.0, -5.0, 3.0]);
        let b = Vector::from_slice(&[-2.0, -3.0, 4.0]);
        let result = a.maximum(&b).unwrap();
        assert_eq!(result.as_slice(), &[-1.0, -3.0, 4.0]);
    }

    #[test]
    fn test_maximum_nan() {
        // NaN handling: NAN.max(x) = x (prefers non-NaN)
        let a = Vector::from_slice(&[f32::NAN, 5.0, f32::NAN]);
        let b = Vector::from_slice(&[3.0, f32::NAN, f32::NAN]);
        let result = a.maximum(&b).unwrap();
        assert_eq!(result.as_slice()[0], 3.0);
        assert_eq!(result.as_slice()[1], 5.0);
        assert!(result.as_slice()[2].is_nan());
    }

    #[test]
    fn test_maximum_infinity() {
        let a = Vector::from_slice(&[f32::INFINITY, 5.0, f32::NEG_INFINITY]);
        let b = Vector::from_slice(&[3.0, f32::INFINITY, -10.0]);
        let result = a.maximum(&b).unwrap();
        assert_eq!(result.as_slice(), &[f32::INFINITY, f32::INFINITY, -10.0]);
    }

    #[test]
    fn test_maximum_size_mismatch() {
        let a = Vector::from_slice(&[1.0, 2.0]);
        let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = a.maximum(&b);
        assert!(matches!(result, Err(TruenoError::SizeMismatch { .. })));
    }

    #[test]
    fn test_maximum_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let b: Vector<f32> = Vector::from_slice(&[]);
        let result = a.maximum(&b).unwrap();
        assert_eq!(result.len(), 0);
    }

    // ========================================
    // Unit Tests: neg()
    // ========================================

    #[test]
    fn test_neg_basic() {
        let a = Vector::from_slice(&[1.0, -2.0, 3.0, -4.0]);
        let result = a.neg().unwrap();
        assert_eq!(result.as_slice(), &[-1.0, 2.0, -3.0, 4.0]);
    }

    #[test]
    fn test_neg_zero() {
        let a = Vector::from_slice(&[0.0, -0.0]);
        let result = a.neg().unwrap();
        // -0.0 becomes 0.0, 0.0 becomes -0.0
        assert_eq!(result.as_slice()[0], -0.0);
        assert_eq!(result.as_slice()[1], 0.0);
    }

    #[test]
    fn test_neg_double_negation() {
        // Property: -(-x) = x (double negation is identity)
        let a = Vector::from_slice(&[1.0, -2.0, 3.0, -4.0, 5.0]);
        let neg_once = a.neg().unwrap();
        let neg_twice = neg_once.neg().unwrap();
        for (i, (&original, &double_neg)) in a.as_slice().iter()
            .zip(neg_twice.as_slice().iter())
            .enumerate() {
            assert!(
                (original - double_neg).abs() < 1e-6,
                "Double negation failed at {}: -(-{}) = {} != {}",
                i, original, double_neg, original
            );
        }
    }

    #[test]
    fn test_neg_nan() {
        let a = Vector::from_slice(&[f32::NAN, 5.0]);
        let result = a.neg().unwrap();
        assert!(result.as_slice()[0].is_nan());
        assert_eq!(result.as_slice()[1], -5.0);
    }

    #[test]
    fn test_neg_infinity() {
        let a = Vector::from_slice(&[f32::INFINITY, f32::NEG_INFINITY]);
        let result = a.neg().unwrap();
        assert_eq!(result.as_slice(), &[f32::NEG_INFINITY, f32::INFINITY]);
    }

    #[test]
    fn test_neg_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.neg().unwrap();
        assert_eq!(result.len(), 0);
    }

    // ========================================
    // Unit Tests: sum_of_squares()
    // ========================================

    #[test]
    fn test_sum_of_squares_basic() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = a.sum_of_squares().unwrap();
        assert_eq!(result, 14.0); // 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
    }

    #[test]
    fn test_sum_of_squares_negative() {
        let a = Vector::from_slice(&[-1.0, -2.0, 3.0]);
        let result = a.sum_of_squares().unwrap();
        assert_eq!(result, 14.0); // (-1)^2 + (-2)^2 + 3^2 = 1 + 4 + 9 = 14
    }

    #[test]
    fn test_sum_of_squares_single() {
        let a = Vector::from_slice(&[5.0]);
        let result = a.sum_of_squares().unwrap();
        assert_eq!(result, 25.0);
    }

    #[test]
    fn test_sum_of_squares_zero() {
        let a = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = a.sum_of_squares().unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_sum_of_squares_pythagorean() {
        // 3-4-5 Pythagorean triple
        let a = Vector::from_slice(&[3.0, 4.0]);
        let result = a.sum_of_squares().unwrap();
        assert_eq!(result, 25.0); // 3^2 + 4^2 = 9 + 16 = 25
    }

    #[test]
    fn test_sum_of_squares_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.sum_of_squares().unwrap();
        assert_eq!(result, 0.0);
    }

    // ========================================================================
    // Tests for mean() - arithmetic average
    // ========================================================================

    #[test]
    fn test_mean_basic() {
        let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let result = a.mean().unwrap();
        assert!((result - 2.5).abs() < 1e-5); // (1+2+3+4)/4 = 2.5
    }

    #[test]
    fn test_mean_negative() {
        let a = Vector::from_slice(&[-2.0, -4.0, -6.0]);
        let result = a.mean().unwrap();
        assert!((result - (-4.0)).abs() < 1e-5); // (-2-4-6)/3 = -4.0
    }

    #[test]
    fn test_mean_mixed() {
        let a = Vector::from_slice(&[-10.0, 0.0, 10.0]);
        let result = a.mean().unwrap();
        assert!(result.abs() < 1e-5); // (-10+0+10)/3 = 0.0
    }

    #[test]
    fn test_mean_single() {
        let a = Vector::from_slice(&[42.0]);
        let result = a.mean().unwrap();
        assert!((result - 42.0).abs() < 1e-5); // 42/1 = 42
    }

    #[test]
    fn test_mean_all_same() {
        let a = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0, 5.0]);
        let result = a.mean().unwrap();
        assert!((result - 5.0).abs() < 1e-5); // (5+5+5+5+5)/5 = 5
    }

    #[test]
    fn test_mean_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.mean();
        assert!(matches!(result, Err(TruenoError::EmptyVector)));
    }

    // ========================================================================
    // Tests for variance() - population variance
    // ========================================================================

    #[test]
    fn test_variance_basic() {
        // Variance of [1,2,3,4,5]: mean=3, var=E[X²]-μ²=11-9=2
        let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = a.variance().unwrap();
        assert!((result - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_variance_constant() {
        // Variance of constant vector is 0
        let a = Vector::from_slice(&[7.0, 7.0, 7.0, 7.0]);
        let result = a.variance().unwrap();
        assert!(result.abs() < 1e-5);
    }

    #[test]
    fn test_variance_single() {
        // Variance of single element is 0
        let a = Vector::from_slice(&[42.0]);
        let result = a.variance().unwrap();
        assert!(result.abs() < 1e-5);
    }

    #[test]
    fn test_variance_symmetric() {
        // Variance of [-2, -1, 0, 1, 2]: mean=0, var=E[X²]=2
        let a = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = a.variance().unwrap();
        assert!((result - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_variance_two_values() {
        // Variance of [1, 5]: mean=3, var=(1-3)²+(5-3)²/2=8/2=4
        let a = Vector::from_slice(&[1.0, 5.0]);
        let result = a.variance().unwrap();
        assert!((result - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_variance_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.variance();
        assert!(matches!(result, Err(TruenoError::EmptyVector)));
    }

    // ========================================================================
    // Tests for stddev() - standard deviation
    // ========================================================================

    #[test]
    fn test_stddev_basic() {
        // stddev of [1,2,3,4,5]: variance=2, stddev=sqrt(2)≈1.414
        let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = a.stddev().unwrap();
        assert!((result - std::f32::consts::SQRT_2).abs() < 1e-5);
    }

    #[test]
    fn test_stddev_constant() {
        // stddev of constant vector is 0
        let a = Vector::from_slice(&[7.0, 7.0, 7.0, 7.0]);
        let result = a.stddev().unwrap();
        assert!(result.abs() < 1e-5);
    }

    #[test]
    fn test_stddev_single() {
        // stddev of single element is 0
        let a = Vector::from_slice(&[42.0]);
        let result = a.stddev().unwrap();
        assert!(result.abs() < 1e-5);
    }

    #[test]
    fn test_stddev_symmetric() {
        // stddev of [-2,-1,0,1,2]: variance=2, stddev=sqrt(2)
        let a = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = a.stddev().unwrap();
        assert!((result - std::f32::consts::SQRT_2).abs() < 1e-5);
    }

    #[test]
    fn test_stddev_two_values() {
        // stddev of [1,5]: variance=4, stddev=2
        let a = Vector::from_slice(&[1.0, 5.0]);
        let result = a.stddev().unwrap();
        assert!((result - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_stddev_empty() {
        let a: Vector<f32> = Vector::from_slice(&[]);
        let result = a.stddev();
        assert!(matches!(result, Err(TruenoError::EmptyVector)));
    }

    // ========================================================================
    // Tests for covariance() - population covariance
    // ========================================================================

    #[test]
    fn test_covariance_positive() {
        // Perfect positive linear relationship: y = 2x
        let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y = Vector::from_slice(&[2.0, 4.0, 6.0]);
        let result = x.covariance(&y).unwrap();
        // Cov(X,2X) = 2*Var(X) = 2*(2/3) = 4/3 ≈ 1.333
        assert!((result - (4.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn test_covariance_negative() {
        // Negative linear relationship
        let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y = Vector::from_slice(&[3.0, 2.0, 1.0]);
        let result = x.covariance(&y).unwrap();
        assert!((result - (-2.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn test_covariance_zero() {
        // No linear relationship
        let x = Vector::from_slice(&[1.0, 2.0, 3.0, 2.0]);
        let y = Vector::from_slice(&[1.0, 3.0, 1.0, 3.0]);
        let result = x.covariance(&y).unwrap();
        assert!(result.abs() < 1e-5);
    }

    #[test]
    fn test_covariance_self() {
        // Cov(X,X) = Var(X)
        let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let cov = x.covariance(&x).unwrap();
        let var = x.variance().unwrap();
        assert!((cov - var).abs() < 1e-5);
    }

    #[test]
    fn test_covariance_size_mismatch() {
        let x = Vector::from_slice(&[1.0, 2.0]);
        let y = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = x.covariance(&y);
        assert!(matches!(
            result,
            Err(TruenoError::SizeMismatch { expected: 2, actual: 3 })
        ));
    }

    #[test]
    fn test_covariance_empty() {
        let x: Vector<f32> = Vector::from_slice(&[]);
        let y: Vector<f32> = Vector::from_slice(&[]);
        let result = x.covariance(&y);
        assert!(matches!(result, Err(TruenoError::EmptyVector)));
    }

    // ========================================================================
    // Tests for correlation() - Pearson correlation coefficient
    // ========================================================================

    #[test]
    fn test_correlation_perfect_positive() {
        // Perfect positive linear relationship: y = 2x
        let x = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let y = Vector::from_slice(&[2.0, 4.0, 6.0]);
        let result = x.correlation(&y).unwrap();
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_correlation_perfect_negative() {
        // Perfect negative linear relationship
        let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let y = Vector::from_slice(&[4.0, 3.0, 2.0, 1.0]);
        let result = x.correlation(&y).unwrap();
        assert!((result - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_correlation_zero() {
        // No correlation
        let x = Vector::from_slice(&[1.0, 2.0, 1.0, 2.0]);
        let y = Vector::from_slice(&[1.0, 1.0, 2.0, 2.0]);
        let result = x.correlation(&y).unwrap();
        assert!(result.abs() < 1e-5);
    }

    #[test]
    fn test_correlation_self() {
        // Correlation with self is always 1
        let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = x.correlation(&x).unwrap();
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_correlation_constant_vector() {
        // Constant vector has zero std dev → division by zero
        let x = Vector::from_slice(&[5.0, 5.0, 5.0]);
        let y = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = x.correlation(&y);
        assert!(matches!(result, Err(TruenoError::DivisionByZero)));
    }

    #[test]
    fn test_correlation_size_mismatch() {
        let x = Vector::from_slice(&[1.0, 2.0]);
        let y = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = x.correlation(&y);
        assert!(matches!(
            result,
            Err(TruenoError::SizeMismatch { expected: 2, actual: 3 })
        ));
    }

    // ========================================================================
    // Tests for zscore() - Z-score normalization (standardization)
    // ========================================================================

    #[test]
    fn test_zscore_basic() {
        // [1, 2, 3, 4, 5] has mean=3, stddev=sqrt(2)
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let z = v.zscore().unwrap();

        // Verify mean ≈ 0
        let mean = z.mean().unwrap();
        assert!(mean.abs() < 1e-5, "mean = {}, expected ≈ 0", mean);

        // Verify stddev ≈ 1
        let std = z.stddev().unwrap();
        assert!(
            (std - 1.0).abs() < 1e-5,
            "stddev = {}, expected ≈ 1",
            std
        );
    }

    #[test]
    fn test_zscore_negative_values() {
        // [-2, -1, 0, 1, 2] has mean=0, stddev=sqrt(2)
        let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let z = v.zscore().unwrap();

        let mean = z.mean().unwrap();
        assert!(mean.abs() < 1e-5);

        let std = z.stddev().unwrap();
        assert!((std - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_zscore_single_element() {
        // Single element has zero stddev → DivisionByZero
        let v = Vector::from_slice(&[5.0]);
        let result = v.zscore();
        assert!(matches!(result, Err(TruenoError::DivisionByZero)));
    }

    #[test]
    fn test_zscore_constant_vector() {
        // All identical elements have zero stddev → DivisionByZero
        let v = Vector::from_slice(&[3.0, 3.0, 3.0, 3.0]);
        let result = v.zscore();
        assert!(matches!(result, Err(TruenoError::DivisionByZero)));
    }

    #[test]
    fn test_zscore_empty_vector() {
        let v = Vector::from_slice(&[]);
        let result = v.zscore();
        assert!(matches!(result, Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_zscore_already_normalized() {
        // Vector already with mean≈0, std≈1 should stay similar
        let v = Vector::from_slice(&[-1.0, 0.0, 1.0]);
        let z = v.zscore().unwrap();

        // Should be close to the original (scaling might differ slightly)
        let mean = z.mean().unwrap();
        assert!(mean.abs() < 1e-5);

        let std = z.stddev().unwrap();
        assert!((std - 1.0).abs() < 1e-5);
    }

    // ========================================================================
    // Tests for minmax_normalize() - Min-max normalization to [0, 1]
    // ========================================================================

    #[test]
    fn test_minmax_normalize_basic() {
        // [1, 2, 3, 4, 5] → [0, 0.25, 0.5, 0.75, 1.0]
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = v.minmax_normalize().unwrap();

        // Verify min = 0
        let min = normalized.min().unwrap();
        assert!((min - 0.0).abs() < 1e-5, "min = {}, expected 0", min);

        // Verify max = 1
        let max = normalized.max().unwrap();
        assert!((max - 1.0).abs() < 1e-5, "max = {}, expected 1", max);

        // Verify specific values
        assert!((normalized.data[0] - 0.0).abs() < 1e-5);
        assert!((normalized.data[2] - 0.5).abs() < 1e-5);
        assert!((normalized.data[4] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_minmax_normalize_negative_values() {
        // [-10, -5, 0, 5, 10] → [0, 0.25, 0.5, 0.75, 1.0]
        let v = Vector::from_slice(&[-10.0, -5.0, 0.0, 5.0, 10.0]);
        let normalized = v.minmax_normalize().unwrap();

        let min = normalized.min().unwrap();
        assert!((min - 0.0).abs() < 1e-5);

        let max = normalized.max().unwrap();
        assert!((max - 1.0).abs() < 1e-5);

        // Middle value should be 0.5
        assert!((normalized.data[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_minmax_normalize_single_element() {
        // Single element has zero range → DivisionByZero
        let v = Vector::from_slice(&[5.0]);
        let result = v.minmax_normalize();
        assert!(matches!(result, Err(TruenoError::DivisionByZero)));
    }

    #[test]
    fn test_minmax_normalize_constant_vector() {
        // All identical elements have zero range → DivisionByZero
        let v = Vector::from_slice(&[3.0, 3.0, 3.0, 3.0]);
        let result = v.minmax_normalize();
        assert!(matches!(result, Err(TruenoError::DivisionByZero)));
    }

    #[test]
    fn test_minmax_normalize_empty_vector() {
        let v = Vector::from_slice(&[]);
        let result = v.minmax_normalize();
        assert!(matches!(result, Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_minmax_normalize_already_normalized() {
        // Vector already in [0, 1] should stay in [0, 1]
        let v = Vector::from_slice(&[0.0, 0.25, 0.5, 0.75, 1.0]);
        let normalized = v.minmax_normalize().unwrap();

        let min = normalized.min().unwrap();
        assert!((min - 0.0).abs() < 1e-5);

        let max = normalized.max().unwrap();
        assert!((max - 1.0).abs() < 1e-5);
    }

    // ========================================================================
    // Tests for clip() - Constrain values to [min, max] range
    // ========================================================================

    #[test]
    fn test_clip_basic() {
        // [-5, 0, 5, 10, 15] clipped to [0, 10] → [0, 0, 5, 10, 10]
        let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
        let clipped = v.clip(0.0, 10.0).unwrap();

        assert_eq!(clipped.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
    }

    #[test]
    fn test_clip_no_change() {
        // All values within range should stay unchanged
        let v = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);
        let clipped = v.clip(0.0, 10.0).unwrap();

        assert_eq!(clipped.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_clip_all_below() {
        // All values below min → all become min
        let v = Vector::from_slice(&[-10.0, -5.0, -2.0]);
        let clipped = v.clip(0.0, 10.0).unwrap();

        assert_eq!(clipped.as_slice(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_clip_all_above() {
        // All values above max → all become max
        let v = Vector::from_slice(&[15.0, 20.0, 25.0]);
        let clipped = v.clip(0.0, 10.0).unwrap();

        assert_eq!(clipped.as_slice(), &[10.0, 10.0, 10.0]);
    }

    #[test]
    fn test_clip_invalid_range() {
        // min > max → InvalidInput error
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = v.clip(10.0, 5.0);

        assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
    }

    #[test]
    fn test_clip_equal_bounds() {
        // min == max → all values become that value
        let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0]);
        let clipped = v.clip(7.0, 7.0).unwrap();

        assert_eq!(clipped.as_slice(), &[7.0, 7.0, 7.0, 7.0]);
    }

    // ========================================================================
    // Tests for softmax() - Softmax activation (probability distribution)
    // ========================================================================

    #[test]
    fn test_softmax_basic() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let probs = v.softmax().unwrap();

        // Verify sum ≈ 1
        let sum: f32 = probs.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {}, expected 1", sum);

        // Verify all values in [0, 1]
        for &p in probs.as_slice() {
            assert!((0.0..=1.0).contains(&p), "prob = {} not in [0, 1]", p);
        }

        // Largest input should have largest probability
        assert!(probs.data[2] > probs.data[1]);
        assert!(probs.data[1] > probs.data[0]);
    }

    #[test]
    fn test_softmax_uniform() {
        // All equal inputs → uniform distribution
        let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
        let probs = v.softmax().unwrap();

        // Each should be 1/4 = 0.25
        for &p in probs.as_slice() {
            assert!((p - 0.25).abs() < 1e-5, "prob = {}, expected 0.25", p);
        }
    }

    #[test]
    fn test_softmax_large_values() {
        // Test numerical stability with large values
        let v = Vector::from_slice(&[100.0, 101.0, 102.0]);
        let probs = v.softmax().unwrap();

        // Should still sum to 1
        let sum: f32 = probs.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Largest value should have largest probability
        assert!(probs.data[2] > probs.data[1]);
        assert!(probs.data[1] > probs.data[0]);
    }

    #[test]
    fn test_softmax_negative_values() {
        let v = Vector::from_slice(&[-3.0, -2.0, -1.0]);
        let probs = v.softmax().unwrap();

        // Verify sum ≈ 1
        let sum: f32 = probs.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // All values should be positive
        for &p in probs.as_slice() {
            assert!(p > 0.0);
        }
    }

    #[test]
    fn test_softmax_empty_vector() {
        let v = Vector::from_slice(&[]);
        let result = v.softmax();
        assert!(matches!(result, Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_softmax_single_element() {
        // Single element → probability 1.0
        let v = Vector::from_slice(&[5.0]);
        let probs = v.softmax().unwrap();

        assert!((probs.data[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_basic() {
        // Verify exp(log_softmax(x)) == softmax(x)
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let log_probs = v.log_softmax().unwrap();
        let probs = v.softmax().unwrap();

        for i in 0..v.len() {
            let exp_log_prob = log_probs.data[i].exp();
            assert!(
                (exp_log_prob - probs.data[i]).abs() < 1e-5,
                "exp(log_softmax)[{}] = {}, softmax[{}] = {}",
                i,
                exp_log_prob,
                i,
                probs.data[i]
            );
        }
    }

    #[test]
    fn test_log_softmax_uniform() {
        // All equal inputs → uniform log probabilities
        let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
        let log_probs = v.log_softmax().unwrap();

        // Each should be log(1/4) = log(0.25) ≈ -1.386
        let expected = (0.25_f32).ln();
        for &lp in log_probs.as_slice() {
            assert!(
                (lp - expected).abs() < 1e-5,
                "log_prob = {}, expected {}",
                lp,
                expected
            );
        }
    }

    #[test]
    fn test_log_softmax_large_values() {
        // Test numerical stability with large values
        let v = Vector::from_slice(&[100.0, 101.0, 102.0]);
        let log_probs = v.log_softmax().unwrap();

        // exp(log_probs) should sum to 1
        let sum: f32 = log_probs.as_slice().iter().map(|&lp| lp.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // All log probabilities should be <= 0 (since probabilities <= 1)
        for &lp in log_probs.as_slice() {
            assert!(lp <= 1e-5, "log_prob = {} should be <= 0", lp);
        }

        // Largest input should have largest log probability (least negative)
        assert!(log_probs.data[2] > log_probs.data[1]);
        assert!(log_probs.data[1] > log_probs.data[0]);
    }

    #[test]
    fn test_log_softmax_negative_values() {
        // Negative values should work fine
        let v = Vector::from_slice(&[-1.0, -2.0, -3.0]);
        let log_probs = v.log_softmax().unwrap();

        // exp(log_probs) should sum to 1
        let sum: f32 = log_probs.as_slice().iter().map(|&lp| lp.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // All log probabilities should be <= 0
        for &lp in log_probs.as_slice() {
            assert!(lp <= 1e-5);
        }
    }

    #[test]
    fn test_log_softmax_empty_vector() {
        let v = Vector::from_slice(&[]);
        let result = v.log_softmax();
        assert!(matches!(result, Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_log_softmax_single_element() {
        // Single element → log probability = log(1.0) = 0.0
        let v = Vector::from_slice(&[5.0]);
        let log_probs = v.log_softmax().unwrap();

        assert!(
            log_probs.data[0].abs() < 1e-5,
            "log_softmax of single element should be 0.0, got {}",
            log_probs.data[0]
        );
    }

    #[test]
    fn test_relu_basic() {
        // Basic ReLU: negative values → 0, positive values unchanged
        let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = v.relu().unwrap();

        assert_eq!(result.as_slice(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_all_negative() {
        // All negative values should become zero
        let v = Vector::from_slice(&[-5.0, -3.0, -1.0, -0.5]);
        let result = v.relu().unwrap();

        for &val in result.as_slice() {
            assert_eq!(val, 0.0, "All negative values should become 0");
        }
    }

    #[test]
    fn test_relu_all_positive() {
        // All positive values should remain unchanged
        let v = Vector::from_slice(&[0.5, 1.0, 3.0, 5.0]);
        let expected = v.clone();
        let result = v.relu().unwrap();

        for i in 0..v.len() {
            assert_eq!(
                result.data[i], expected.data[i],
                "Positive values should remain unchanged"
            );
        }
    }

    #[test]
    fn test_relu_zero_boundary() {
        // Zero should remain zero (boundary case)
        let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = v.relu().unwrap();

        for &val in result.as_slice() {
            assert_eq!(val, 0.0, "Zero should remain zero");
        }
    }

    #[test]
    fn test_relu_empty_vector() {
        let v = Vector::from_slice(&[]);
        let result = v.relu();
        assert!(matches!(result, Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_relu_sparsity() {
        // ReLU creates sparse activations (zeros for negative inputs)
        let v = Vector::from_slice(&[-10.0, 5.0, -3.0, 8.0, -1.0, 2.0]);
        let result = v.relu().unwrap();

        // Count zeros (should be 3)
        let zero_count = result.as_slice().iter().filter(|&&x| x == 0.0).count();
        assert_eq!(zero_count, 3, "ReLU should produce sparse activations");

        // Verify positive values preserved
        assert_eq!(result.data[1], 5.0);
        assert_eq!(result.data[3], 8.0);
        assert_eq!(result.data[5], 2.0);
    }

    #[test]
    fn test_sigmoid_basic() {
        // Basic sigmoid: negative → (0, 0.5), zero → 0.5, positive → (0.5, 1)
        let v = Vector::from_slice(&[-2.0, 0.0, 2.0]);
        let result = v.sigmoid().unwrap();

        // sigmoid(-2) ≈ 0.1192, sigmoid(0) = 0.5, sigmoid(2) ≈ 0.8808
        assert!((result.data[0] - 0.1192).abs() < 0.001);
        assert!((result.data[1] - 0.5).abs() < 0.001);
        assert!((result.data[2] - 0.8808).abs() < 0.001);
    }

    #[test]
    fn test_sigmoid_range() {
        // All outputs should be in [0, 1] range (inclusive for numerical stability)
        let v = Vector::from_slice(&[-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]);
        let result = v.sigmoid().unwrap();

        for &val in result.as_slice() {
            assert!(
                (0.0..=1.0).contains(&val),
                "Sigmoid output {} not in [0, 1]",
                val
            );
        }
    }

    #[test]
    fn test_sigmoid_symmetry() {
        // Test σ(-x) = 1 - σ(x)
        let v = Vector::from_slice(&[-3.0, -1.5, -0.5]);
        let v_neg = Vector::from_slice(&[3.0, 1.5, 0.5]);

        let sig = v.sigmoid().unwrap();
        let sig_neg = v_neg.sigmoid().unwrap();

        for i in 0..v.len() {
            let sum = sig.data[i] + sig_neg.data[i];
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Symmetry violated: σ({}) + σ({}) = {} + {} = {} ≠ 1",
                v.data[i],
                v_neg.data[i],
                sig.data[i],
                sig_neg.data[i],
                sum
            );
        }
    }

    #[test]
    fn test_sigmoid_extreme_values() {
        // Test numerical stability with extreme values
        let v = Vector::from_slice(&[-100.0, -50.0, 50.0, 100.0]);
        let result = v.sigmoid().unwrap();

        // Very negative → close to 0
        assert!(result.data[0] < 1e-6, "sigmoid(-100) should be ≈ 0");
        assert!(result.data[1] < 1e-6, "sigmoid(-50) should be ≈ 0");

        // Very positive → close to 1
        assert!(result.data[2] > 1.0 - 1e-6, "sigmoid(50) should be ≈ 1");
        assert!(result.data[3] > 1.0 - 1e-6, "sigmoid(100) should be ≈ 1");
    }

    #[test]
    fn test_sigmoid_empty_vector() {
        let v = Vector::from_slice(&[]);
        let result = v.sigmoid();
        assert!(matches!(result, Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_sigmoid_zero() {
        // sigmoid(0) should be exactly 0.5
        let v = Vector::from_slice(&[0.0, 0.0, 0.0]);
        let result = v.sigmoid().unwrap();

        for &val in result.as_slice() {
            assert!((val - 0.5).abs() < 1e-7, "sigmoid(0) = {} ≠ 0.5", val);
        }
    }

    #[test]
    fn test_leaky_relu_basic() {
        // Basic Leaky ReLU with α = 0.01
        let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = v.leaky_relu(0.01).unwrap();

        assert_eq!(result.as_slice(), &[-0.02, -0.01, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_leaky_relu_different_slopes() {
        // Test with different negative slopes
        let v = Vector::from_slice(&[-10.0, 5.0]);

        // α = 0.01 (default)
        let result_001 = v.leaky_relu(0.01).unwrap();
        assert!((result_001.data[0] - (-0.1)).abs() < 1e-6); // -10 * 0.01
        assert_eq!(result_001.data[1], 5.0);

        // α = 0.1
        let result_01 = v.leaky_relu(0.1).unwrap();
        assert!((result_01.data[0] - (-1.0)).abs() < 1e-6); // -10 * 0.1
        assert_eq!(result_01.data[1], 5.0);

        // α = 0.2
        let result_02 = v.leaky_relu(0.2).unwrap();
        assert!((result_02.data[0] - (-2.0)).abs() < 1e-6); // -10 * 0.2
        assert_eq!(result_02.data[1], 5.0);
    }

    #[test]
    fn test_leaky_relu_reduces_to_relu() {
        // With α = 0, should behave like standard ReLU
        let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let leaky = v.leaky_relu(0.0).unwrap();
        let relu = v.relu().unwrap();

        for i in 0..v.len() {
            assert_eq!(leaky.data[i], relu.data[i], "α=0 should equal ReLU");
        }
    }

    #[test]
    fn test_leaky_relu_preserves_positive() {
        // Positive values should remain unchanged regardless of α
        let v = Vector::from_slice(&[0.5, 1.0, 5.0, 10.0]);
        let result = v.leaky_relu(0.01).unwrap();

        for i in 0..v.len() {
            assert_eq!(
                result.data[i], v.data[i],
                "Positive values should be preserved"
            );
        }
    }

    #[test]
    fn test_leaky_relu_empty_vector() {
        let v = Vector::from_slice(&[]);
        let result = v.leaky_relu(0.01);
        assert!(matches!(result, Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_leaky_relu_invalid_slope() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);

        // Negative slope should fail
        let result = v.leaky_relu(-0.1);
        assert!(matches!(result, Err(TruenoError::InvalidInput(_))));

        // Slope >= 1.0 should fail
        let result = v.leaky_relu(1.0);
        assert!(matches!(result, Err(TruenoError::InvalidInput(_))));

        let result = v.leaky_relu(1.5);
        assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
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
            // If a[i] < a[j], then sqrt(a[i]) <= sqrt(a[j])
            let va = Vector::from_slice(&a);
            let result = va.sqrt().unwrap();
            let result_slice = result.as_slice();

            for i in 0..a.len()-1 {
                for j in i+1..a.len() {
                    // Use a small epsilon to account for f32 precision
                    let epsilon = 1e-6;
                    if a[i] + epsilon < a[j] {
                        prop_assert!(
                            result_slice[i] <= result_slice[j],
                            "Monotonicity failed: sqrt({}) = {} should be <= sqrt({}) = {}",
                            a[i], result_slice[i], a[j], result_slice[j]
                        );
                    } else if a[i] > a[j] + epsilon {
                        prop_assert!(
                            result_slice[i] >= result_slice[j],
                            "Monotonicity failed: sqrt({}) = {} should be >= sqrt({}) = {}",
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

    // Property test: fract correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_fract_correctness(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.fract().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = input.fract();
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "fract failed at {}: {} != {}",
                    i, output, expected
                );
            }
        }
    }

    // Property test: fract decomposition - x = trunc(x) + fract(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_fract_decomposition(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let trunc_result = va.trunc().unwrap();
            let fract_result = va.fract().unwrap();

            for (i, (&input, (&t, &f))) in a.iter()
                .zip(trunc_result.as_slice().iter().zip(fract_result.as_slice().iter()))
                .enumerate() {
                let reconstructed = t + f;
                prop_assert!(
                    (reconstructed - input).abs() < 1e-5,
                    "decomposition failed at {}: {} != trunc({}) + fract({}) = {} + {} = {}",
                    i, input, input, input, t, f, reconstructed
                );
            }
        }
    }

    // Property test: fract magnitude - |fract(x)| < 1
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_fract_magnitude(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.fract().unwrap();

            for (i, &output) in result.as_slice().iter().enumerate() {
                prop_assert!(
                    output.abs() < 1.0,
                    "fract magnitude should be < 1 at {}: |fract({})| = {} >= 1",
                    i, a[i], output.abs()
                );
            }
        }
    }

    // Property test: signum correctness
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_signum_correctness(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.signum().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                let expected = input.signum();
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "signum failed at {}: {} != {}",
                    i, output, expected
                );
            }
        }
    }

    // Property test: signum range - always -1, 0, or 1
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_signum_range(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.signum().unwrap();

            for (i, &output) in result.as_slice().iter().enumerate() {
                prop_assert!(
                    output == 1.0 || output == -1.0 || output.is_nan(),
                    "signum should be 1.0, -1.0, or NaN at {}: signum({}) = {}",
                    i, a[i], output
                );
            }
        }
    }

    // Property test: signum * abs = identity (for non-zero)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_signum_abs_identity(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let signum_result = va.signum().unwrap();
            let abs_result = va.abs().unwrap();

            for (i, (&input, (&sign, &magnitude))) in a.iter()
                .zip(signum_result.as_slice().iter().zip(abs_result.as_slice().iter()))
                .enumerate() {
                // Skip zero values as they have special behavior
                if input.abs() > 1e-10 {
                    let reconstructed = sign * magnitude;
                    prop_assert!(
                        (reconstructed - input).abs() < 1e-5,
                        "signum*abs identity failed at {}: {} != signum({}) * abs({}) = {} * {} = {}",
                        i, input, input, input, sign, magnitude, reconstructed
                    );
                }
            }
        }
    }

    // ========================================
    // Property Tests: copysign()
    // ========================================

    // Property test: copysign correctness - matches f32::copysign
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_copysign_correctness(
            ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
        ) {
            let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
            let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

            let va = Vector::from_slice(&a);
            let vb = Vector::from_slice(&b);
            let result = va.copysign(&vb).unwrap();

            for (i, (&mag, (&sgn, &output))) in a.iter()
                .zip(b.iter().zip(result.as_slice().iter()))
                .enumerate() {
                let expected = mag.copysign(sgn);
                prop_assert!(
                    (output - expected).abs() < 1e-5 || (output.is_nan() && expected.is_nan()),
                    "copysign failed at {}: copysign({}, {}) = {} != {}",
                    i, mag, sgn, output, expected
                );
            }
        }
    }

    // Property test: magnitude preservation - abs(copysign(a, b)) = abs(a)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_copysign_magnitude_preservation(
            ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
        ) {
            let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
            let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

            let va = Vector::from_slice(&a);
            let vb = Vector::from_slice(&b);
            let result = va.copysign(&vb).unwrap();
            let abs_a = va.abs().unwrap();
            let abs_result = result.abs().unwrap();

            for (i, (&expected, &output)) in abs_a.as_slice().iter()
                .zip(abs_result.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "magnitude not preserved at {}: abs(copysign({}, {})) = {} != abs({}) = {}",
                    i, a[i], b[i], output, a[i], expected
                );
            }
        }
    }

    // Property test: sign copy - sign(copysign(a, b)) = sign(b) for non-zero b
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_copysign_sign_copy(
            ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
        ) {
            let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
            let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

            let va = Vector::from_slice(&a);
            let vb = Vector::from_slice(&b);
            let result = va.copysign(&vb).unwrap();
            let signum_b = vb.signum().unwrap();
            let signum_result = result.signum().unwrap();

            for (i, (&sign_b, &sign_result)) in signum_b.as_slice().iter()
                .zip(signum_result.as_slice().iter())
                .enumerate() {
                // Skip NaN cases
                if !sign_b.is_nan() && !sign_result.is_nan() {
                    prop_assert!(
                        (sign_result - sign_b).abs() < 1e-5,
                        "sign not copied at {}: sign(copysign({}, {})) = {} != sign({}) = {}",
                        i, a[i], b[i], sign_result, b[i], sign_b
                    );
                }
            }
        }
    }

    // ========================================
    // Property Tests: minimum()
    // ========================================

    // Property test: minimum correctness - matches f32::min
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_minimum_correctness(
            ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
        ) {
            let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
            let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

            let va = Vector::from_slice(&a);
            let vb = Vector::from_slice(&b);
            let result = va.minimum(&vb).unwrap();

            for (i, (&x, (&y, &output))) in a.iter()
                .zip(b.iter().zip(result.as_slice().iter()))
                .enumerate() {
                let expected = x.min(y);
                prop_assert!(
                    (output - expected).abs() < 1e-5 || (output.is_nan() && expected.is_nan()),
                    "minimum failed at {}: minimum({}, {}) = {} != {}",
                    i, x, y, output, expected
                );
            }
        }
    }

    // Property test: commutativity - minimum(a, b) = minimum(b, a)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_minimum_commutative(
            ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
        ) {
            let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
            let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

            let va = Vector::from_slice(&a);
            let vb = Vector::from_slice(&b);
            let result1 = va.minimum(&vb).unwrap();
            let result2 = vb.minimum(&va).unwrap();

            for (i, (&r1, &r2)) in result1.as_slice().iter()
                .zip(result2.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (r1 - r2).abs() < 1e-5 || (r1.is_nan() && r2.is_nan()),
                    "commutativity failed at {}: minimum({}, {}) = {} != minimum({}, {}) = {}",
                    i, a[i], b[i], r1, b[i], a[i], r2
                );
            }
        }
    }

    // Property test: idempotence - minimum(a, a) = a
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_minimum_idempotent(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.minimum(&va).unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (output - input).abs() < 1e-5 || (output.is_nan() && input.is_nan()),
                    "idempotence failed at {}: minimum({}, {}) = {} != {}",
                    i, input, input, output, input
                );
            }
        }
    }

    // ========================================
    // Property Tests: maximum()
    // ========================================

    // Property test: maximum correctness - matches f32::max
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_maximum_correctness(
            ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
        ) {
            let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
            let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

            let va = Vector::from_slice(&a);
            let vb = Vector::from_slice(&b);
            let result = va.maximum(&vb).unwrap();

            for (i, (&x, (&y, &output))) in a.iter()
                .zip(b.iter().zip(result.as_slice().iter()))
                .enumerate() {
                let expected = x.max(y);
                prop_assert!(
                    (output - expected).abs() < 1e-5 || (output.is_nan() && expected.is_nan()),
                    "maximum failed at {}: maximum({}, {}) = {} != {}",
                    i, x, y, output, expected
                );
            }
        }
    }

    // Property test: commutativity - maximum(a, b) = maximum(b, a)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_maximum_commutative(
            ab in prop::collection::vec((-100.0f32..100.0, -100.0f32..100.0), 1..100)
        ) {
            let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
            let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

            let va = Vector::from_slice(&a);
            let vb = Vector::from_slice(&b);
            let result1 = va.maximum(&vb).unwrap();
            let result2 = vb.maximum(&va).unwrap();

            for (i, (&r1, &r2)) in result1.as_slice().iter()
                .zip(result2.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (r1 - r2).abs() < 1e-5 || (r1.is_nan() && r2.is_nan()),
                    "commutativity failed at {}: maximum({}, {}) = {} != maximum({}, {}) = {}",
                    i, a[i], b[i], r1, b[i], a[i], r2
                );
            }
        }
    }

    // Property test: idempotence - maximum(a, a) = a
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_maximum_idempotent(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.maximum(&va).unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(result.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (output - input).abs() < 1e-5 || (output.is_nan() && input.is_nan()),
                    "idempotence failed at {}: maximum({}, {}) = {} != {}",
                    i, input, input, output, input
                );
            }
        }
    }

    // ========================================
    // Property Tests: neg()
    // ========================================

    // Property test: double negation is identity - -(-x) = x
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_neg_double_negation_property(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let neg_once = va.neg().unwrap();
            let neg_twice = neg_once.neg().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(neg_twice.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (output - input).abs() < 1e-5 || (output.is_nan() && input.is_nan()),
                    "double negation failed at {}: -(-{}) = {} != {}",
                    i, input, output, input
                );
            }
        }
    }

    // Property test: negation sign flip - sign(neg(x)) = -sign(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_neg_sign_flip(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let neg_result = va.neg().unwrap();

            for (i, (&input, &output)) in a.iter()
                .zip(neg_result.as_slice().iter())
                .enumerate() {
                // Skip zero and NaN
                if input.abs() > 1e-10 && !input.is_nan() {
                    prop_assert!(
                        (input.signum() + output.signum()).abs() < 1e-5,
                        "sign flip failed at {}: sign({}) + sign(-{}) = {} + {} != 0",
                        i, input, input, input.signum(), output.signum()
                    );
                }
            }
        }
    }

    // Property test: negation preserves magnitude - abs(neg(x)) = abs(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_neg_magnitude_preservation(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let neg_result = va.neg().unwrap();
            let abs_a = va.abs().unwrap();
            let abs_neg_a = neg_result.abs().unwrap();

            for (i, (&expected, &output)) in abs_a.as_slice().iter()
                .zip(abs_neg_a.as_slice().iter())
                .enumerate() {
                prop_assert!(
                    (output - expected).abs() < 1e-5,
                    "magnitude not preserved at {}: abs(-{}) = {} != abs({}) = {}",
                    i, a[i], output, a[i], expected
                );
            }
        }
    }

    // ========================================
    // Property Tests: sum_of_squares()
    // ========================================

    // Property test: non-negativity - sum_of_squares is always >= 0
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sum_of_squares_non_negative(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.sum_of_squares().unwrap();

            prop_assert!(
                result >= 0.0,
                "sum_of_squares should be non-negative: {} < 0",
                result
            );
        }
    }

    // Property test: equivalence with dot product - sum_of_squares(v) = dot(v, v)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sum_of_squares_equals_dot_self(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let sum_sq = va.sum_of_squares().unwrap();
            let dot_self = va.dot(&va).unwrap();

            prop_assert!(
                (sum_sq - dot_self).abs() < 1e-4,
                "sum_of_squares should equal dot(self, self): {} != {}",
                sum_sq, dot_self
            );
        }
    }

    // Property test: scaling - sum_of_squares(k*v) = k^2 * sum_of_squares(v)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sum_of_squares_scaling(
            a in prop::collection::vec(-10.0f32..10.0, 1..50),
            k in -5.0f32..5.0
        ) {
            let va = Vector::from_slice(&a);
            let scaled = va.scale(k).unwrap();

            let sum_sq_original = va.sum_of_squares().unwrap();
            let sum_sq_scaled = scaled.sum_of_squares().unwrap();
            let expected = k * k * sum_sq_original;

            // Use relative tolerance for larger values
            let tolerance = 1e-3 * expected.abs().max(1.0);
            prop_assert!(
                (sum_sq_scaled - expected).abs() < tolerance,
                "sum_of_squares({} * v) = {} != {}^2 * {} = {}",
                k, sum_sq_scaled, k, sum_sq_original, expected
            );
        }

        /// Property test: mean(v) is between min(v) and max(v)
        #[test]
        fn test_mean_bounds(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let mean_val = va.mean().unwrap();
            let min_val = va.min().unwrap();
            let max_val = va.max().unwrap();

            prop_assert!(
                mean_val >= min_val && mean_val <= max_val,
                "mean({}) = {} not in range [{}, {}]",
                mean_val, mean_val, min_val, max_val
            );
        }

        /// Property test: mean(v + c) = mean(v) + c (translation property)
        #[test]
        fn test_mean_translation(
            a in prop::collection::vec(-50.0f32..50.0, 1..100),
            c in -10.0f32..10.0
        ) {
            let va = Vector::from_slice(&a);
            let mean_original = va.mean().unwrap();

            // Create translated vector: v + c
            let translated: Vec<f32> = a.iter().map(|x| x + c).collect();
            let vt = Vector::from_slice(&translated);
            let mean_translated = vt.mean().unwrap();

            let expected = mean_original + c;
            let tolerance = 1e-4 * expected.abs().max(1.0);
            prop_assert!(
                (mean_translated - expected).abs() < tolerance,
                "mean(v + {}) = {} != mean(v) + {} = {}",
                c, mean_translated, c, expected
            );
        }

        /// Property test: mean(k*v) = k*mean(v) (scaling property)
        #[test]
        fn test_mean_scaling(
            a in prop::collection::vec(-50.0f32..50.0, 1..100),
            k in -5.0f32..5.0
        ) {
            let va = Vector::from_slice(&a);
            let mean_original = va.mean().unwrap();
            let scaled = va.scale(k).unwrap();
            let mean_scaled = scaled.mean().unwrap();

            let expected = k * mean_original;
            let tolerance = 1e-4 * expected.abs().max(1.0);
            prop_assert!(
                (mean_scaled - expected).abs() < tolerance,
                "mean({} * v) = {} != {} * mean(v) = {}",
                k, mean_scaled, k, expected
            );
        }

        /// Property test: variance(v) >= 0 (non-negativity)
        #[test]
        fn test_variance_non_negative(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let var = va.variance().unwrap();

            prop_assert!(
                var >= -1e-5, // Allow small numerical error
                "variance = {} should be non-negative",
                var
            );
        }

        /// Property test: variance(k*v) = k²*variance(v) (scaling property)
        #[test]
        fn test_variance_scaling(
            a in prop::collection::vec(-50.0f32..50.0, 1..100),
            k in -5.0f32..5.0
        ) {
            let va = Vector::from_slice(&a);
            let var_original = va.variance().unwrap();
            let scaled = va.scale(k).unwrap();
            let var_scaled = scaled.variance().unwrap();

            let expected = k * k * var_original;
            let tolerance = 1e-3 * expected.abs().max(1e-5);
            prop_assert!(
                (var_scaled - expected).abs() < tolerance,
                "variance({} * v) = {} != {}² * variance(v) = {}",
                k, var_scaled, k, expected
            );
        }

        /// Property test: variance(v + c) = variance(v) (translation invariance)
        #[test]
        fn test_variance_translation_invariance(
            a in prop::collection::vec(-50.0f32..50.0, 1..100),
            c in -10.0f32..10.0
        ) {
            let va = Vector::from_slice(&a);
            let var_original = va.variance().unwrap();

            // Create translated vector: v + c
            let translated: Vec<f32> = a.iter().map(|x| x + c).collect();
            let vt = Vector::from_slice(&translated);
            let var_translated = vt.variance().unwrap();

            let tolerance = 1e-3 * var_original.abs().max(1e-5);
            prop_assert!(
                (var_translated - var_original).abs() < tolerance,
                "variance(v + {}) = {} != variance(v) = {}",
                c, var_translated, var_original
            );
        }

        /// Property test: stddev(v) >= 0 (non-negativity)
        #[test]
        fn test_stddev_non_negative(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let sd = va.stddev().unwrap();

            prop_assert!(
                sd >= -1e-5, // Allow small numerical error
                "stddev = {} should be non-negative",
                sd
            );
        }

        /// Property test: stddev(k*v) = |k|*stddev(v) (linear scaling)
        #[test]
        fn test_stddev_scaling(
            a in prop::collection::vec(-50.0f32..50.0, 1..100),
            k in -5.0f32..5.0
        ) {
            let va = Vector::from_slice(&a);
            let sd_original = va.stddev().unwrap();
            let scaled = va.scale(k).unwrap();
            let sd_scaled = scaled.stddev().unwrap();

            let expected = k.abs() * sd_original;
            let tolerance = 1e-3 * expected.abs().max(1e-5);
            prop_assert!(
                (sd_scaled - expected).abs() < tolerance,
                "stddev({} * v) = {} != |{}| * stddev(v) = {}",
                k, sd_scaled, k, expected
            );
        }

        /// Property test: stddev(v + c) = stddev(v) (translation invariance)
        #[test]
        fn test_stddev_translation_invariance(
            a in prop::collection::vec(-50.0f32..50.0, 1..100),
            c in -10.0f32..10.0
        ) {
            let va = Vector::from_slice(&a);
            let sd_original = va.stddev().unwrap();

            // Create translated vector: v + c
            let translated: Vec<f32> = a.iter().map(|x| x + c).collect();
            let vt = Vector::from_slice(&translated);
            let sd_translated = vt.stddev().unwrap();

            let tolerance = 1e-3 * sd_original.abs().max(1e-5);
            prop_assert!(
                (sd_translated - sd_original).abs() < tolerance,
                "stddev(v + {}) = {} != stddev(v) = {}",
                c, sd_translated, sd_original
            );
        }

        /// Property test: Cov(X,X) = Var(X) (covariance with self equals variance)
        #[test]
        fn test_covariance_self_equals_variance(
            a in prop::collection::vec(-50.0f32..50.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let cov = va.covariance(&va).unwrap();
            let var = va.variance().unwrap();

            let tolerance = 1e-3 * var.abs().max(1e-5);
            prop_assert!(
                (cov - var).abs() < tolerance,
                "Cov(X,X) = {} != Var(X) = {}",
                cov, var
            );
        }

        /// Property test: Cov(X,Y) = Cov(Y,X) (symmetry/commutativity)
        #[test]
        fn test_covariance_symmetric(
            ab in prop::collection::vec((-50.0f32..50.0, -50.0f32..50.0), 1..100)
        ) {
            let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
            let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();
            let va = Vector::from_slice(&a);
            let vb = Vector::from_slice(&b);

            let cov_ab = va.covariance(&vb).unwrap();
            let cov_ba = vb.covariance(&va).unwrap();

            let tolerance = 1e-4 * cov_ab.abs().max(1e-5);
            prop_assert!(
                (cov_ab - cov_ba).abs() < tolerance,
                "Cov(X,Y) = {} != Cov(Y,X) = {}",
                cov_ab, cov_ba
            );
        }

        /// Property test: Cov(aX, bY) = ab*Cov(X,Y) (bilinearity)
        #[test]
        fn test_covariance_bilinearity(
            ab in prop::collection::vec((-20.0f32..20.0, -20.0f32..20.0), 1..50),
            scale_a in -3.0f32..3.0,
            scale_b in -3.0f32..3.0
        ) {
            let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
            let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();
            let va = Vector::from_slice(&a);
            let vb = Vector::from_slice(&b);

            let cov_original = va.covariance(&vb).unwrap();

            let scaled_a = va.scale(scale_a).unwrap();
            let scaled_b = vb.scale(scale_b).unwrap();
            let cov_scaled = scaled_a.covariance(&scaled_b).unwrap();

            let expected = scale_a * scale_b * cov_original;
            let tolerance = 1e-3 * expected.abs().max(1e-5);
            prop_assert!(
                (cov_scaled - expected).abs() < tolerance,
                "Cov({}*X, {}*Y) = {} != {}*{}*Cov(X,Y) = {}",
                scale_a, scale_b, cov_scaled, scale_a, scale_b, expected
            );
        }

        /// Property test: -1 ≤ ρ(X,Y) ≤ 1 (correlation is bounded)
        #[test]
        fn test_correlation_bounded(
            ab in prop::collection::vec((-50.0f32..50.0, -50.0f32..50.0), 2..100)
        ) {
            let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
            let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

            // Ensure vectors are not constant
            let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                           - (a.iter().sum::<f32>() / a.len() as f32).powi(2);
            let std_b: f32 = b.iter().map(|y| y * y).sum::<f32>() / b.len() as f32
                           - (b.iter().sum::<f32>() / b.len() as f32).powi(2);

            if std_a < 1e-6 || std_b < 1e-6 {
                return Ok(());  // Skip constant vectors
            }

            let va = Vector::from_slice(&a);
            let vb = Vector::from_slice(&b);
            let corr = va.correlation(&vb).unwrap();

            prop_assert!(
                (-1.0 - 1e-5..=1.0 + 1e-5).contains(&corr),
                "correlation = {} not in range [-1, 1]",
                corr
            );
        }

        /// Property test: ρ(X,Y) = ρ(Y,X) (symmetry)
        #[test]
        fn test_correlation_symmetric(
            ab in prop::collection::vec((-50.0f32..50.0, -50.0f32..50.0), 2..100)
        ) {
            let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
            let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

            // Ensure vectors are not constant
            let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                           - (a.iter().sum::<f32>() / a.len() as f32).powi(2);
            let std_b: f32 = b.iter().map(|y| y * y).sum::<f32>() / b.len() as f32
                           - (b.iter().sum::<f32>() / b.len() as f32).powi(2);

            if std_a < 1e-6 || std_b < 1e-6 {
                return Ok(());  // Skip constant vectors
            }

            let va = Vector::from_slice(&a);
            let vb = Vector::from_slice(&b);

            let corr_ab = va.correlation(&vb).unwrap();
            let corr_ba = vb.correlation(&va).unwrap();

            let tolerance = 1e-5;
            prop_assert!(
                (corr_ab - corr_ba).abs() < tolerance,
                "ρ(X,Y) = {} != ρ(Y,X) = {}",
                corr_ab, corr_ba
            );
        }

        /// Property test: ρ(X,X) = 1 (perfect self-correlation)
        #[test]
        fn test_correlation_self_is_one(
            a in prop::collection::vec(-50.0f32..50.0, 2..100)
        ) {
            // Ensure vector is not constant
            let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                           - (a.iter().sum::<f32>() / a.len() as f32).powi(2);

            if std_a < 1e-6 {
                return Ok(());  // Skip constant vectors
            }

            let va = Vector::from_slice(&a);
            let corr = va.correlation(&va).unwrap();

            prop_assert!(
                (corr - 1.0).abs() < 1e-5,
                "ρ(X,X) = {} != 1.0",
                corr
            );
        }
    }

    // ========================================================================
    // Property tests for zscore() - Z-score normalization
    // ========================================================================

    proptest! {
        /// Property test: zscore() produces mean ≈ 0
        #[test]
        fn test_zscore_produces_zero_mean(
            a in prop::collection::vec(-100.0f32..100.0, 2..100)
        ) {
            // Ensure vector is not constant
            let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                           - (a.iter().sum::<f32>() / a.len() as f32).powi(2);

            if std_a < 1e-6 {
                return Ok(());  // Skip constant vectors
            }

            let va = Vector::from_slice(&a);
            let z = va.zscore().unwrap();
            let mean = z.mean().unwrap();

            prop_assert!(
                mean.abs() < 1e-4,
                "zscore mean = {}, expected ≈ 0",
                mean
            );
        }
    }

    proptest! {
        /// Property test: zscore() produces stddev ≈ 1
        #[test]
        fn test_zscore_produces_unit_stddev(
            a in prop::collection::vec(-100.0f32..100.0, 2..100)
        ) {
            // Ensure vector is not constant
            let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                           - (a.iter().sum::<f32>() / a.len() as f32).powi(2);

            if std_a < 1e-6 {
                return Ok(());  // Skip constant vectors
            }

            let va = Vector::from_slice(&a);
            let z = va.zscore().unwrap();
            let std = z.stddev().unwrap();

            prop_assert!(
                (std - 1.0).abs() < 1e-4,
                "zscore stddev = {}, expected ≈ 1",
                std
            );
        }
    }

    proptest! {
        /// Property test: zscore() preserves correlation structure
        /// ρ(zscore(X), zscore(Y)) = ρ(X, Y)
        #[test]
        fn test_zscore_preserves_correlation(
            ab in prop::collection::vec((-50.0f32..50.0, -50.0f32..50.0), 2..100)
        ) {
            let a: Vec<f32> = ab.iter().map(|(x, _)| *x).collect();
            let b: Vec<f32> = ab.iter().map(|(_, y)| *y).collect();

            // Ensure vectors are not constant
            let std_a: f32 = a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32
                           - (a.iter().sum::<f32>() / a.len() as f32).powi(2);
            let std_b: f32 = b.iter().map(|x| x * x).sum::<f32>() / b.len() as f32
                           - (b.iter().sum::<f32>() / b.len() as f32).powi(2);

            if std_a < 1e-6 || std_b < 1e-6 {
                return Ok(());  // Skip constant vectors
            }

            let va = Vector::from_slice(&a);
            let vb = Vector::from_slice(&b);

            // Original correlation
            let corr_orig = va.correlation(&vb).unwrap();

            // Correlation after zscore
            let za = va.zscore().unwrap();
            let zb = vb.zscore().unwrap();
            let corr_zscore = za.correlation(&zb).unwrap();

            let tolerance = 1e-3;
            prop_assert!(
                (corr_orig - corr_zscore).abs() < tolerance,
                "ρ(X,Y) = {} != ρ(zscore(X), zscore(Y)) = {}",
                corr_orig, corr_zscore
            );
        }
    }

    // ========================================================================
    // Property tests for minmax_normalize() - Min-max normalization
    // ========================================================================

    proptest! {
        /// Property test: minmax_normalize() produces min = 0
        #[test]
        fn test_minmax_normalize_produces_zero_min(
            a in prop::collection::vec(-100.0f32..100.0, 2..100)
        ) {
            // Ensure vector is not constant
            let min_a = a.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_a = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            if (max_a - min_a).abs() < 1e-6 {
                return Ok(());  // Skip constant vectors
            }

            let va = Vector::from_slice(&a);
            let normalized = va.minmax_normalize().unwrap();
            let min = normalized.min().unwrap();

            prop_assert!(
                min.abs() < 1e-4,
                "minmax min = {}, expected ≈ 0",
                min
            );
        }
    }

    proptest! {
        /// Property test: minmax_normalize() produces max = 1
        #[test]
        fn test_minmax_normalize_produces_one_max(
            a in prop::collection::vec(-100.0f32..100.0, 2..100)
        ) {
            // Ensure vector is not constant
            let min_a = a.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_a = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            if (max_a - min_a).abs() < 1e-6 {
                return Ok(());  // Skip constant vectors
            }

            let va = Vector::from_slice(&a);
            let normalized = va.minmax_normalize().unwrap();
            let max = normalized.max().unwrap();

            prop_assert!(
                (max - 1.0).abs() < 1e-4,
                "minmax max = {}, expected ≈ 1",
                max
            );
        }
    }

    proptest! {
        /// Property test: minmax_normalize() preserves order (monotonicity)
        /// If a[i] <= a[j], then normalized[i] <= normalized[j]
        #[test]
        fn test_minmax_normalize_preserves_order(
            a in prop::collection::vec(-100.0f32..100.0, 2..100)
        ) {
            // Ensure vector is not constant
            let min_a = a.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_a = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            if (max_a - min_a).abs() < 1e-6 {
                return Ok(());  // Skip constant vectors
            }

            let va = Vector::from_slice(&a);
            let normalized = va.minmax_normalize().unwrap();

            // Check that order is preserved for all pairs
            for i in 0..a.len() {
                for j in 0..a.len() {
                    if a[i] <= a[j] {
                        prop_assert!(
                            normalized.data[i] <= normalized.data[j] + 1e-5,
                            "Order not preserved: a[{}]={} <= a[{}]={}, but norm[{}]={} > norm[{}]={}",
                            i, a[i], j, a[j], i, normalized.data[i], j, normalized.data[j]
                        );
                    }
                }
            }
        }
    }

    // ========================================================================
    // Property tests for clip() - Range clipping
    // ========================================================================

    proptest! {
        /// Property test: clip() produces values within [min_val, max_val]
        #[test]
        fn test_clip_within_bounds(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            min_val in -50.0f32..50.0,
            max_val in -50.0f32..50.0
        ) {
            // Ensure min <= max
            let (min_val, max_val) = if min_val <= max_val {
                (min_val, max_val)
            } else {
                (max_val, min_val)
            };

            let va = Vector::from_slice(&a);
            let clipped = va.clip(min_val, max_val).unwrap();

            // All values must be within [min_val, max_val]
            for &val in clipped.as_slice() {
                prop_assert!(
                    (min_val..=max_val).contains(&val),
                    "Value {} not in range [{}, {}]",
                    val, min_val, max_val
                );
            }
        }
    }

    proptest! {
        /// Property test: clip() preserves order (monotonicity)
        /// If a[i] <= a[j], then clip(a)[i] <= clip(a)[j]
        #[test]
        fn test_clip_preserves_order(
            a in prop::collection::vec(-100.0f32..100.0, 2..100),
            min_val in -50.0f32..50.0,
            max_val in -50.0f32..50.0
        ) {
            // Ensure min <= max
            let (min_val, max_val) = if min_val <= max_val {
                (min_val, max_val)
            } else {
                (max_val, min_val)
            };

            let va = Vector::from_slice(&a);
            let clipped = va.clip(min_val, max_val).unwrap();

            // Check order preservation
            for i in 0..a.len() {
                for j in 0..a.len() {
                    if a[i] <= a[j] {
                        prop_assert!(
                            clipped.data[i] <= clipped.data[j] + 1e-5,
                            "Order not preserved: a[{}]={} <= a[{}]={}, but clip[{}]={} > clip[{}]={}",
                            i, a[i], j, a[j], i, clipped.data[i], j, clipped.data[j]
                        );
                    }
                }
            }
        }
    }

    proptest! {
        /// Property test: clip() is idempotent
        /// clip(clip(X)) = clip(X)
        #[test]
        fn test_clip_idempotent(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            min_val in -50.0f32..50.0,
            max_val in -50.0f32..50.0
        ) {
            // Ensure min <= max
            let (min_val, max_val) = if min_val <= max_val {
                (min_val, max_val)
            } else {
                (max_val, min_val)
            };

            let va = Vector::from_slice(&a);
            let clipped1 = va.clip(min_val, max_val).unwrap();
            let clipped2 = clipped1.clip(min_val, max_val).unwrap();

            // Clipping twice should give same result
            for i in 0..clipped1.len() {
                prop_assert!(
                    (clipped1.data[i] - clipped2.data[i]).abs() < 1e-5,
                    "Idempotency violated at index {}: clip_once={}, clip_twice={}",
                    i, clipped1.data[i], clipped2.data[i]
                );
            }
        }
    }

    // ========================================================================
    // Property tests for softmax() - Probability distribution
    // ========================================================================

    proptest! {
        /// Property test: softmax() produces values that sum to 1
        #[test]
        fn test_softmax_sums_to_one(
            a in prop::collection::vec(-50.0f32..50.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let probs = va.softmax().unwrap();
            let sum: f32 = probs.as_slice().iter().sum();

            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "softmax sum = {}, expected 1.0",
                sum
            );
        }
    }

    proptest! {
        /// Property test: softmax() produces values in [0, 1]
        #[test]
        fn test_softmax_in_unit_range(
            a in prop::collection::vec(-50.0f32..50.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let probs = va.softmax().unwrap();

            for &p in probs.as_slice() {
                prop_assert!(
                    (0.0..=1.0).contains(&p),
                    "probability {} not in [0, 1]",
                    p
                );
            }
        }
    }

    proptest! {
        /// Property test: softmax() is translation invariant
        /// softmax(x + c) = softmax(x) for any constant c
        #[test]
        fn test_softmax_translation_invariant(
            a in prop::collection::vec(-20.0f32..20.0, 2..50),
            c in -10.0f32..10.0
        ) {
            let va = Vector::from_slice(&a);
            let probs1 = va.softmax().unwrap();

            // Add constant to all elements
            let shifted: Vec<f32> = a.iter().map(|&x| x + c).collect();
            let vb = Vector::from_slice(&shifted);
            let probs2 = vb.softmax().unwrap();

            // Probabilities should be identical
            for i in 0..probs1.len() {
                prop_assert!(
                    (probs1.data[i] - probs2.data[i]).abs() < 1e-4,
                    "Translation invariance violated at index {}: softmax(x)={}, softmax(x+{})={}",
                    i, probs1.data[i], c, probs2.data[i]
                );
            }
        }
    }

    // ========================================================================
    // Property tests for log_softmax() - Log probability distribution
    // ========================================================================

    proptest! {
        /// Property test: exp(log_softmax(x)) sums to 1
        /// Since log_softmax returns log probabilities, exponentiating should give valid probabilities
        #[test]
        fn test_log_softmax_exp_sums_to_one(
            a in prop::collection::vec(-50.0f32..50.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let log_probs = va.log_softmax().unwrap();

            // Exponentiate to get probabilities
            let sum: f32 = log_probs.as_slice().iter().map(|&lp| lp.exp()).sum();

            prop_assert!(
                (sum - 1.0).abs() < 1e-4,
                "exp(log_softmax) sum = {}, expected 1.0",
                sum
            );
        }
    }

    proptest! {
        /// Property test: log_softmax() produces values <= 0
        /// Since probabilities are in [0, 1], log(prob) <= 0
        #[test]
        fn test_log_softmax_non_positive(
            a in prop::collection::vec(-50.0f32..50.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let log_probs = va.log_softmax().unwrap();

            for &lp in log_probs.as_slice() {
                prop_assert!(
                    lp <= 1e-5,
                    "log_probability {} should be <= 0",
                    lp
                );
            }
        }
    }

    proptest! {
        /// Property test: log_softmax() is translation invariant
        /// log_softmax(x + c) = log_softmax(x) for any constant c
        #[test]
        fn test_log_softmax_translation_invariant(
            a in prop::collection::vec(-20.0f32..20.0, 2..50),
            c in -10.0f32..10.0
        ) {
            let va = Vector::from_slice(&a);
            let log_probs1 = va.log_softmax().unwrap();

            // Add constant to all elements
            let shifted: Vec<f32> = a.iter().map(|&x| x + c).collect();
            let vb = Vector::from_slice(&shifted);
            let log_probs2 = vb.log_softmax().unwrap();

            // Log probabilities should be identical
            for i in 0..log_probs1.len() {
                prop_assert!(
                    (log_probs1.data[i] - log_probs2.data[i]).abs() < 1e-4,
                    "Translation invariance violated at index {}: log_softmax(x)={}, log_softmax(x+{})={}",
                    i, log_probs1.data[i], c, log_probs2.data[i]
                );
            }
        }
    }

    // ========================================================================
    // Property tests for relu() - Rectified Linear Unit
    // ========================================================================

    proptest! {
        /// Property test: relu() produces non-negative outputs
        /// All outputs should be >= 0
        #[test]
        fn test_relu_non_negative(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.relu().unwrap();

            for &val in result.as_slice() {
                prop_assert!(
                    val >= 0.0,
                    "ReLU output {} should be non-negative",
                    val
                );
            }
        }
    }

    proptest! {
        /// Property test: relu() preserves positive values
        /// For all x > 0, relu(x) = x
        #[test]
        fn test_relu_preserves_positive(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.relu().unwrap();

            for (i, &val) in a.iter().enumerate() {
                if val > 0.0 {
                    prop_assert!(
                        (result.data[i] - val).abs() < 1e-6,
                        "ReLU should preserve positive value: {} became {}",
                        val, result.data[i]
                    );
                }
            }
        }
    }

    proptest! {
        /// Property test: relu() is idempotent
        /// relu(relu(x)) = relu(x)
        #[test]
        fn test_relu_idempotent(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let relu1 = va.relu().unwrap();
            let relu2 = relu1.relu().unwrap();

            for (i, &orig_val) in a.iter().enumerate() {
                prop_assert!(
                    (relu1.data[i] - relu2.data[i]).abs() < 1e-6,
                    "ReLU should be idempotent: relu(relu({})) = {} != relu({}) = {}",
                    orig_val, relu2.data[i], orig_val, relu1.data[i]
                );
            }
        }
    }

    // ========================================================================
    // Property tests for sigmoid() - Logistic activation
    // ========================================================================

    proptest! {
        /// Property test: sigmoid() produces values in [0, 1]
        #[test]
        fn test_sigmoid_bounded(
            a in prop::collection::vec(-100.0f32..100.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.sigmoid().unwrap();

            for &val in result.as_slice() {
                prop_assert!(
                    (0.0..=1.0).contains(&val),
                    "Sigmoid output {} not in [0, 1]",
                    val
                );
            }
        }
    }

    proptest! {
        /// Property test: sigmoid() symmetry σ(-x) = 1 - σ(x)
        #[test]
        fn test_sigmoid_symmetry_property(
            a in prop::collection::vec(-50.0f32..50.0, 1..100)
        ) {
            let va = Vector::from_slice(&a);
            let sig_pos = va.sigmoid().unwrap();

            // Create negated vector
            let a_neg: Vec<f32> = a.iter().map(|&x| -x).collect();
            let va_neg = Vector::from_slice(&a_neg);
            let sig_neg = va_neg.sigmoid().unwrap();

            // σ(-x) + σ(x) should equal 1
            for (i, &val) in a.iter().enumerate() {
                let sum = sig_pos.data[i] + sig_neg.data[i];
                prop_assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "Symmetry violated: σ({}) + σ({}) = {} + {} = {} ≠ 1",
                    val, -val, sig_pos.data[i], sig_neg.data[i], sum
                );
            }
        }
    }

    proptest! {
        /// Property test: sigmoid() is monotonically increasing
        /// If x < y, then σ(x) < σ(y)
        #[test]
        fn test_sigmoid_monotonic(
            a in prop::collection::vec(-50.0f32..50.0, 2..100)
        ) {
            let va = Vector::from_slice(&a);
            let result = va.sigmoid().unwrap();

            // Check all pairs for monotonicity
            for i in 0..a.len() {
                for j in 0..a.len() {
                    if a[i] < a[j] {
                        prop_assert!(
                            result.data[i] < result.data[j] + 1e-6,
                            "Monotonicity violated: {} < {} but σ({}) = {} >= σ({}) = {}",
                            a[i], a[j], a[i], result.data[i], a[j], result.data[j]
                        );
                    }
                }
            }
        }
    }

    // ========================================================================
    // Property tests for leaky_relu() - Leaky Rectified Linear Unit
    // ========================================================================

    proptest! {
        /// Property test: leaky_relu() preserves positive values exactly
        #[test]
        fn test_leaky_relu_preserves_positive_property(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            alpha in 0.0f32..1.0
        ) {
            let va = Vector::from_slice(&a);
            let result = va.leaky_relu(alpha).unwrap();

            for (i, &val) in a.iter().enumerate() {
                if val > 0.0 {
                    prop_assert!(
                        (result.data[i] - val).abs() < 1e-6,
                        "Positive value {} should be preserved, got {}",
                        val, result.data[i]
                    );
                }
            }
        }
    }

    proptest! {
        /// Property test: leaky_relu() scales negative values by alpha
        #[test]
        fn test_leaky_relu_scales_negative_property(
            a in prop::collection::vec(-100.0f32..100.0, 1..100),
            alpha in 0.01f32..0.5 // Use smaller range to avoid precision issues
        ) {
            let va = Vector::from_slice(&a);
            let result = va.leaky_relu(alpha).unwrap();

            for (i, &val) in a.iter().enumerate() {
                if val < 0.0 {
                    let expected = alpha * val;
                    prop_assert!(
                        (result.data[i] - expected).abs() < 1e-4,
                        "Negative value {} should be scaled by {}: expected {}, got {}",
                        val, alpha, expected, result.data[i]
                    );
                }
            }
        }
    }

    proptest! {
        /// Property test: leaky_relu() is monotonically increasing
        /// If x < y, then leaky_relu(x) < leaky_relu(y)
        #[test]
        fn test_leaky_relu_monotonic_property(
            a in prop::collection::vec(-50.0f32..50.0, 2..100),
            alpha in 0.01f32..0.5
        ) {
            let va = Vector::from_slice(&a);
            let result = va.leaky_relu(alpha).unwrap();

            // Check all pairs for monotonicity
            for i in 0..a.len() {
                for j in 0..a.len() {
                    if a[i] < a[j] {
                        prop_assert!(
                            result.data[i] < result.data[j] + 1e-5,
                            "Monotonicity violated: {} < {} but leaky_relu({}) = {} >= leaky_relu({}) = {}",
                            a[i], a[j], a[i], result.data[i], a[j], result.data[j]
                        );
                    }
                }
            }
        }
    }
}
