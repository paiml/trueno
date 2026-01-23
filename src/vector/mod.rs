//! Vector type with multi-backend support
//!
//! This module provides the core `Vector<T>` type with SIMD-optimized operations
//! across multiple backends (Scalar, SSE2, AVX2, AVX-512, NEON, WASM SIMD).
//!
//! GPU thresholds intentionally set to usize::MAX to disable GPU for element-wise ops.
//! See docs/performance-analysis.md - GPU is 2-65,000x SLOWER than scalar for these ops.

#![allow(clippy::absurd_extreme_comparisons)]

// Submodules
pub mod dispatch;

// Tests (~10K lines extracted for TDG compliance)
#[cfg(test)]
mod tests;

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
use crate::{Backend, Result, TruenoError};

// Use the dispatch macros from the dispatch submodule (exported at crate root)
use crate::{dispatch_binary_op, dispatch_reduction, dispatch_unary_op};

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

    /// Create vector from an existing Vec (takes ownership, no copy)
    ///
    /// This is more efficient than `from_slice` when you already have a Vec
    /// and don't need to keep it, as it avoids an extra allocation and copy.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let data = vec![1.0, 2.0, 3.0];
    /// let v = Vector::from_vec(data);
    /// assert_eq!(v.len(), 3);
    /// ```
    pub fn from_vec(data: Vec<T>) -> Self {
        Self {
            data,
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

    /// Z-score normalization (standardization)
    ///
    /// Transforms the vector to have mean = 0 and standard deviation = 1.
    /// Each element is transformed as: z\[i\] = (x\[i\] - μ) / σ
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

        Ok(Vector::from_vec(data))
    }

    /// Min-max normalization (scaling to [0, 1] range)
    ///
    /// Transforms the vector so that the minimum value becomes 0 and the maximum
    /// value becomes 1, with all other values scaled proportionally.
    /// Formula: x'\[i\] = (x\[i\] - min) / (max - min)
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

        Ok(Vector::from_vec(data))
    }

    /// Layer normalization with learnable parameters (Issue #61: ML primitives)
    ///
    /// Applies layer normalization: `y = gamma * (x - mean) / sqrt(variance + eps) + beta`
    ///
    /// This is a fundamental normalization technique in transformers and other
    /// modern neural network architectures. Unlike batch normalization, layer norm
    /// normalizes across the feature dimension, making it suitable for sequence models.
    ///
    /// # Arguments
    ///
    /// * `gamma` - Scale parameter (typically learned, initialized to 1.0)
    /// * `beta` - Shift parameter (typically learned, initialized to 0.0)
    /// * `eps` - Small constant for numerical stability (typically 1e-5 or 1e-6)
    ///
    /// # Returns
    ///
    /// Normalized vector with the same shape as input
    ///
    /// # Errors
    ///
    /// Returns `SizeMismatch` if gamma or beta have different lengths than self
    /// Returns `EmptyVector` if input is empty
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// let gamma = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]); // Scale = 1
    /// let beta = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);  // Shift = 0
    ///
    /// let y = x.layer_norm(&gamma, &beta, 1e-5).unwrap();
    ///
    /// // Output should be approximately standardized (mean ≈ 0, std ≈ 1)
    /// let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
    /// assert!(mean.abs() < 1e-5);
    /// ```
    ///
    /// # Performance
    ///
    /// Single-pass computation using Welford's algorithm for numerical stability.
    /// Time complexity: O(n), Space complexity: O(n).
    pub fn layer_norm(&self, gamma: &Self, beta: &Self, eps: f32) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        if self.len() != gamma.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: gamma.len(),
            });
        }

        if self.len() != beta.len() {
            return Err(TruenoError::SizeMismatch {
                expected: self.len(),
                actual: beta.len(),
            });
        }

        // Compute mean
        let mean_val = self.mean()?;

        // Compute variance: E[(x - mean)^2]
        let variance: f32 = self
            .data
            .iter()
            .map(|&x| {
                let diff = x - mean_val;
                diff * diff
            })
            .sum::<f32>()
            / self.len() as f32;

        // Compute inverse standard deviation for numerical stability
        let inv_std = 1.0 / (variance + eps).sqrt();

        // Apply normalization: y = gamma * (x - mean) * inv_std + beta
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(gamma.data.iter())
            .zip(beta.data.iter())
            .map(|((&x, &g), &b)| g * (x - mean_val) * inv_std + b)
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// Layer normalization without learnable parameters
    ///
    /// Simplified version that just standardizes the input: `y = (x - mean) / sqrt(variance + eps)`
    ///
    /// This is equivalent to calling `layer_norm` with gamma=1 and beta=0.
    ///
    /// # Arguments
    ///
    /// * `eps` - Small constant for numerical stability (typically 1e-5)
    ///
    /// # Example
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let x = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    /// let y = x.layer_norm_simple(1e-5).unwrap();
    ///
    /// // Output should be standardized
    /// let mean: f32 = y.as_slice().iter().sum::<f32>() / y.len() as f32;
    /// assert!(mean.abs() < 1e-5);
    /// ```
    pub fn layer_norm_simple(&self, eps: f32) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        let mean_val = self.mean()?;

        // Compute variance
        let variance: f32 = self
            .data
            .iter()
            .map(|&x| {
                let diff = x - mean_val;
                diff * diff
            })
            .sum::<f32>()
            / self.len() as f32;

        let inv_std = 1.0 / (variance + eps).sqrt();

        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| (x - mean_val) * inv_std)
            .collect();

        Ok(Vector::from_vec(data))
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

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.clip(&self.data, &mut result, min_val, max_val).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Scalar fallback: Element-wise clamp
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| x.max(min_val).min(max_val))
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// Softmax activation function
    ///
    /// Converts a vector of real values into a probability distribution.
    /// Formula: softmax(x)\[i\] = exp(x\[i\] - max(x)) / sum(exp(x\[j\] - max(x)))
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

        // OpComplexity::Medium - GPU threshold: >10K elements (multi-pass overhead)
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 4-368x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.softmax(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Scalar fallback: Multi-pass softmax for numerical stability
        // Find max for numerical stability (prevents overflow in exp)
        let max_val = self.max()?;

        // Compute exp(x - max) for each element
        let exp_vals: Vec<f32> = self.data.iter().map(|&x| (x - max_val).exp()).collect();

        // Compute sum of exponentials
        let sum_exp: f32 = exp_vals.iter().sum();

        // Normalize by sum
        let data: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

        Ok(Vector::from_vec(data))
    }

    /// Log-softmax activation function
    ///
    /// Computes the logarithm of the softmax function in a numerically stable way.
    /// Formula: log_softmax(x)\[i\] = x\[i\] - max(x) - log(sum(exp(x\[j\] - max(x))))
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

        // OpComplexity::Medium - GPU threshold: >10K elements (multi-pass overhead)
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 4-368x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.log_softmax(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Scalar fallback: Multi-pass log_softmax for numerical stability
        // Find max for numerical stability
        let max_val = self.max()?;

        // Compute exp(x - max) for each element
        let exp_vals: Vec<f32> = self.data.iter().map(|&x| (x - max_val).exp()).collect();

        // Compute log of sum of exponentials
        let sum_exp: f32 = exp_vals.iter().sum();
        let log_sum_exp = sum_exp.ln();

        // log_softmax(x)[i] = x[i] - max - log_sum_exp
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| x - max_val - log_sum_exp)
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// ReLU (Rectified Linear Unit) activation function
    ///
    /// Computes the element-wise ReLU: max(0, x).
    /// ReLU is one of the most widely used activation functions in neural networks.
    ///
    /// # Formula
    ///
    /// ```text
    /// relu(x)[i] = max(0, x\[i\])
    ///            = x\[i\]  if x\[i\] > 0
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

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.relu(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        let mut result = vec![0.0; self.len()];

        // Use parallel processing for very large arrays (reduces TLB pressure and improves cache utilization)
        #[cfg(feature = "parallel")]
        {
            const PARALLEL_THRESHOLD: usize = 500_000; // Increased to avoid overhead at smaller sizes
            const CHUNK_SIZE: usize = 65536; // 64K elements = 256KB, cache-friendly

            if self.len() >= PARALLEL_THRESHOLD {
                use rayon::prelude::*;

                self.data
                    .par_chunks(CHUNK_SIZE)
                    .zip(result.par_chunks_mut(CHUNK_SIZE))
                    .for_each(|(chunk_in, chunk_out)| {
                        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
                        unsafe {
                            match self.backend {
                                Backend::Scalar => {
                                    ScalarBackend::relu(chunk_in, chunk_out);
                                }
                                #[cfg(target_arch = "x86_64")]
                                Backend::SSE2 | Backend::AVX => {
                                    Sse2Backend::relu(chunk_in, chunk_out);
                                }
                                #[cfg(target_arch = "x86_64")]
                                Backend::AVX2 | Backend::AVX512 => {
                                    Avx2Backend::relu(chunk_in, chunk_out);
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                                    ScalarBackend::relu(chunk_in, chunk_out);
                                }
                                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                                Backend::NEON => {
                                    NeonBackend::relu(chunk_in, chunk_out);
                                }
                                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                                Backend::NEON => {
                                    ScalarBackend::relu(chunk_in, chunk_out);
                                }
                                #[cfg(target_arch = "wasm32")]
                                Backend::WasmSIMD => {
                                    WasmBackend::relu(chunk_in, chunk_out);
                                }
                                #[cfg(not(target_arch = "wasm32"))]
                                Backend::WasmSIMD => {
                                    ScalarBackend::relu(chunk_in, chunk_out);
                                }
                                Backend::GPU | Backend::Auto => {
                                    ScalarBackend::relu(chunk_in, chunk_out);
                                }
                            }
                        }
                    });

                return Ok(Vector::from_vec(result)); // Use from_vec to avoid extra copy
            }
        }

        // Sequential processing for small arrays or when parallel feature disabled
        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::relu(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::relu(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    Avx2Backend::relu(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::relu(&self.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::relu(&self.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    ScalarBackend::relu(&self.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::relu(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    ScalarBackend::relu(&self.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::relu(&self.data, &mut result);
                }
            }
        }

        Ok(Vector::from_vec(result)) // Use from_vec to avoid extra copy
    }

    /// Sigmoid (logistic) activation function
    ///
    /// Computes the element-wise sigmoid: σ(x) = 1 / (1 + e^(-x)).
    /// Sigmoid is a classic activation function that squashes inputs to the range (0, 1).
    ///
    /// # Formula
    ///
    /// ```text
    /// sigmoid(x)[i] = 1 / (1 + exp(-x\[i\]))
    ///               = exp(x\[i\]) / (1 + exp(x\[i\]))
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
    /// assert!((result.as_slice()[0] - 0.119).abs() < 0.001);
    /// assert!((result.as_slice()[1] - 0.5).abs() < 0.001);
    /// assert!((result.as_slice()[2] - 0.881).abs() < 0.001);
    /// ```
    pub fn sigmoid(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.sigmoid(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        let mut result = vec![0.0; self.len()];

        // Dispatch to appropriate backend
        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::sigmoid(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::sigmoid(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    Avx2Backend::sigmoid(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::sigmoid(&self.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::sigmoid(&self.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    ScalarBackend::sigmoid(&self.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::sigmoid(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    ScalarBackend::sigmoid(&self.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::sigmoid(&self.data, &mut result);
                }
            }
        }

        Ok(Vector::from_vec(result))
    }

    /// Leaky ReLU activation function
    ///
    /// Computes the element-wise Leaky ReLU with a configurable negative slope.
    /// Leaky ReLU addresses the "dying ReLU" problem by allowing small negative values.
    ///
    /// # Formula
    ///
    /// ```text
    /// leaky_relu(x, α)[i] = max(αx\[i\], x\[i\])
    ///                     = x\[i\]    if x\[i\] > 0
    ///                     = αx\[i\]   if x\[i\] ≤ 0
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

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu
                        .leaky_relu(&self.data, &mut result, negative_slope)
                        .is_ok()
                    {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Scalar fallback: leaky_relu(x, α) = x if x > 0, αx otherwise
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| if x > 0.0 { x } else { negative_slope * x })
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// ELU (Exponential Linear Unit) activation function
    ///
    /// Computes the element-wise ELU with a configurable alpha parameter.
    /// ELU pushes mean activations closer to zero, improving learning.
    ///
    /// # Formula
    ///
    /// ```text
    /// elu(x, α)[i] = x\[i\]           if x\[i\] > 0
    ///              = α(e^x\[i\] - 1)  if x\[i\] ≤ 0
    /// ```
    ///
    /// # Parameters
    ///
    /// - `alpha`: Controls the saturation value for negative inputs (typically 1.0)
    ///   - Must be > 0
    ///   - Common value: 1.0 (original ELU paper)
    ///   - Larger α → slower saturation for negative inputs
    ///
    /// # Properties
    ///
    /// - **Smooth**: Unlike ReLU/Leaky ReLU, has smooth gradients everywhere
    /// - **Negative values**: Allows negative outputs (pushes mean closer to zero)
    /// - **Bounded below**: Saturates to -α for very negative inputs
    /// - **Unbounded above**: No saturation for positive values
    /// - **Non-zero gradient**: Has gradient everywhere (no dead neurons)
    ///
    /// # Applications
    ///
    /// - **Deep networks**: Better gradient flow than ReLU
    /// - **Mean activation near zero**: Reduces internal covariate shift
    /// - **Noise robustness**: Smooth activation helps with noisy gradients
    /// - **Empirical improvements**: Often outperforms ReLU and Leaky ReLU
    ///
    /// # Performance
    ///
    /// This operation is compute-bound due to exp() for negative values.
    /// More expensive than ReLU/Leaky ReLU but provides better properties.
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    /// Returns `InvalidInput` if alpha <= 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.elu(1.0).unwrap();
    ///
    /// // Negative values: α(e^x - 1), positive unchanged
    /// // elu(-2, 1) ≈ -0.865, elu(-1, 1) ≈ -0.632
    /// assert!((result.as_slice()[0] - (-0.865)).abs() < 0.01);
    /// assert!((result.as_slice()[1] - (-0.632)).abs() < 0.01);
    /// assert_eq!(result.as_slice()[2], 0.0);
    /// assert_eq!(result.as_slice()[3], 1.0);
    /// assert_eq!(result.as_slice()[4], 2.0);
    /// ```
    pub fn elu(&self, alpha: f32) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // Validate alpha parameter
        if alpha <= 0.0 {
            return Err(TruenoError::InvalidInput(format!(
                "alpha must be > 0, got {}",
                alpha
            )));
        }

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.elu(&self.data, &mut result, alpha).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Scalar fallback: elu(x, α) = x if x > 0, α(e^x - 1) otherwise
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// GELU (Gaussian Error Linear Unit) activation function
    ///
    /// Computes the element-wise GELU activation using the tanh approximation.
    /// GELU is the activation function used in transformers (BERT, GPT, etc.).
    ///
    /// # Formula
    ///
    /// ```text
    /// gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    /// ```
    ///
    /// This is the tanh approximation which is faster than the exact form
    /// involving the error function (erf).
    ///
    /// # Properties
    ///
    /// - **Smooth**: Infinitely differentiable everywhere
    /// - **Non-monotonic**: Unlike ReLU variants, has slight non-monotonicity near zero
    /// - **Stochastic regularizer**: Can be viewed as adaptive dropout
    /// - **Zero-centered**: Mean activation close to zero
    /// - **Bounded below**: Approaches 0 as x → -∞
    /// - **Unbounded above**: Linear growth for large positive x
    ///
    /// # Applications
    ///
    /// - **Transformers**: BERT, GPT-2, GPT-3, GPT-4 (default activation)
    /// - **Vision transformers**: ViT, DINO, MAE
    /// - **Modern architectures**: State-of-the-art NLP and vision models
    /// - **Better than ReLU**: Empirically outperforms ReLU in many tasks
    ///
    /// # Performance
    ///
    /// This operation is compute-intensive (tanh, x³ calculations).
    /// More expensive than ReLU but comparable to ELU.
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
    /// let result = v.gelu().unwrap();
    ///
    /// // GELU is smooth and non-monotonic near zero
    /// assert!(result.as_slice()[0] < 0.0); // Negative inputs → small negative outputs
    /// assert_eq!(result.as_slice()[2], 0.0); // gelu(0) = 0
    /// assert!(result.as_slice()[4] > 1.5); // Large positive → ~linear
    /// ```
    pub fn gelu(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.gelu(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        let mut result = vec![0.0; self.len()];

        // Dispatch to appropriate backend
        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::gelu(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::gelu(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    Avx2Backend::gelu(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::gelu(&self.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::gelu(&self.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    ScalarBackend::gelu(&self.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::gelu(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    ScalarBackend::gelu(&self.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
                    ScalarBackend::gelu(&self.data, &mut result);
                }
            }
        }

        Ok(Vector::from_vec(result))
    }

    /// Swish activation function (also known as SiLU - Sigmoid Linear Unit)
    ///
    /// Applies the Swish activation element-wise: swish(x) = x * sigmoid(x) = x / (1 + e^(-x)).
    ///
    /// Swish is a smooth, non-monotonic activation function that consistently matches or
    /// outperforms ReLU in deep networks. It's used in EfficientNet, MobileNet v3, and
    /// many modern architectures. The function is self-gated: it adaptively gates the
    /// input based on its value.
    ///
    /// Properties:
    /// - Smooth and differentiable everywhere
    /// - Non-monotonic: has a slight "dip" for negative values
    /// - swish(0) = 0
    /// - swish(x) ≈ x for large positive x (linear)
    /// - swish(x) ≈ 0 for large negative x
    /// - Unbounded above, bounded below by ≈ -0.278 at x ≈ -1.278
    ///
    /// # Performance
    ///
    /// Compute-bound operation requiring exponential and division.
    /// Future SIMD optimizations planned for Phase 9 (GPU backend).
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.swish().unwrap();
    ///
    /// // Swish is smooth and self-gated
    /// assert!(result.as_slice()[0] < 0.0); // Negative inputs → small negative outputs
    /// assert_eq!(result.as_slice()[2], 0.0); // swish(0) = 0
    /// assert!(result.as_slice()[4] > 1.5); // Large positive → ~linear
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # References
    ///
    /// - Ramachandran et al. (2017): "Searching for Activation Functions"
    /// - Also known as SiLU (Sigmoid Linear Unit): Elfwing et al. (2018)
    pub fn swish(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.swish(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        let mut result = vec![0.0; self.len()];

        // Dispatch to appropriate SIMD backend
        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::swish(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::swish(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    Avx2Backend::swish(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::swish(&self.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::swish(&self.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    ScalarBackend::swish(&self.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::swish(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    ScalarBackend::swish(&self.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
                    // Not yet implemented, use scalar
                    ScalarBackend::swish(&self.data, &mut result);
                }
            }
        }

        Ok(Vector::from_vec(result))
    }

    /// Hard Swish activation function
    ///
    /// Applies the hardswish activation element-wise: hardswish(x) = x * relu6(x + 3) / 6
    ///
    /// Hardswish is a piece-wise linear approximation to swish, designed for efficient
    /// computation in mobile neural networks. It's used in MobileNetV3 and avoids the
    /// expensive sigmoid computation of standard swish.
    ///
    /// Properties:
    /// - Piece-wise linear: efficient to compute
    /// - hardswish(x) = 0 for x ≤ -3
    /// - hardswish(x) = x for x ≥ 3
    /// - hardswish(x) = x * (x + 3) / 6 for -3 < x < 3
    /// - hardswish(0) = 0
    /// - Smooth transitions at boundaries
    ///
    /// # Performance
    ///
    /// More efficient than swish as it uses only multiply/divide operations
    /// instead of expensive exponential functions. Ideal for inference on
    /// resource-constrained devices.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-4.0, -3.0, 0.0, 3.0, 4.0]);
    /// let result = v.hardswish().unwrap();
    ///
    /// // Piece-wise linear behavior
    /// assert_eq!(result.as_slice()[0], 0.0); // x ≤ -3 → 0
    /// assert_eq!(result.as_slice()[1], 0.0); // x = -3 → 0
    /// assert_eq!(result.as_slice()[2], 0.0); // x = 0 → 0
    /// assert_eq!(result.as_slice()[3], 3.0); // x = 3 → x
    /// assert_eq!(result.as_slice()[4], 4.0); // x ≥ 3 → x
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # References
    ///
    /// - Howard et al. (2019): "Searching for MobileNetV3"
    pub fn hardswish(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // Scalar implementation: hardswish(x) = x * relu6(x + 3) / 6
        // Simplified piece-wise:
        // - x <= -3: 0
        // - x >= 3: x
        // - else: x * (x + 3) / 6
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                if x <= -3.0 {
                    0.0
                } else if x >= 3.0 {
                    x
                } else {
                    x * (x + 3.0) / 6.0
                }
            })
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// Mish activation function
    ///
    /// Applies the mish activation element-wise: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    ///
    /// Mish is a self-regularizing non-monotonic activation function that often outperforms
    /// ReLU and swish in computer vision tasks. It's used in YOLOv4 and many modern architectures.
    ///
    /// Properties:
    /// - Smooth and non-monotonic (similar to swish)
    /// - Self-regularizing: prevents dying neurons
    /// - mish(0) ≈ 0 (small positive value)
    /// - mish(x) ≈ x for large positive x (nearly linear)
    /// - mish(x) ≈ 0 for large negative x
    /// - Bounded below by ≈ -0.31 at x ≈ -1.19
    ///
    /// # Performance
    ///
    /// Compute-bound operation requiring exponential, logarithm, and tanh.
    /// More expensive than ReLU/swish but often provides better accuracy.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.mish().unwrap();
    ///
    /// // Mish is smooth and self-gated
    /// assert!(result.as_slice()[0] < 0.0); // Small negative output for negative inputs
    /// assert!(result.as_slice()[2].abs() < 1e-5); // mish(0) = 0
    /// assert!(result.as_slice()[4] > 1.5); // Large positive → near linear
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # References
    ///
    /// - Misra (2019): "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
    pub fn mish(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // Scalar implementation: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                // Handle extreme values for numerical stability
                if x < -20.0 {
                    // For very negative x: softplus ≈ 0, tanh(0) ≈ 0, so mish ≈ 0
                    0.0
                } else if x > 20.0 {
                    // For very positive x: softplus ≈ x, tanh(x) ≈ 1, so mish ≈ x
                    x
                } else {
                    // Normal case: x * tanh(ln(1 + e^x))
                    let softplus = (1.0 + x.exp()).ln();
                    x * softplus.tanh()
                }
            })
            .collect();

        Ok(Vector::from_vec(data))
    }

    /// SELU (Scaled Exponential Linear Unit) activation function
    ///
    /// Computes selu(x) = λ * (x if x > 0 else α * (exp(x) - 1))
    /// where λ ≈ 1.0507 and α ≈ 1.6733
    ///
    /// # Properties
    ///
    /// - **Self-normalizing**: Activations converge to zero mean and unit variance
    /// - **Vanishing gradient prevention**: Non-zero gradient for negative inputs
    /// - **Automatic normalization**: Reduces need for batch normalization
    ///
    /// # Performance
    ///
    /// Uses scalar implementation (GPU disabled for element-wise ops).
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    /// let result = v.selu().unwrap();
    ///
    /// // Positive values scaled by λ ≈ 1.0507
    /// assert!((result.as_slice()[3] - 1.0507).abs() < 0.001);
    /// assert!((result.as_slice()[4] - 2.1014).abs() < 0.001);
    ///
    /// // Zero stays zero
    /// assert!(result.as_slice()[2].abs() < 1e-5);
    ///
    /// // Negative values use ELU-like formula
    /// assert!(result.as_slice()[0] < 0.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `EmptyVector` if the input vector is empty.
    ///
    /// # References
    ///
    /// - Klambauer et al. (2017): "Self-Normalizing Neural Networks"
    pub fn selu(&self) -> Result<Self> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // SELU constants from Klambauer et al. (2017)
        // These specific values ensure self-normalizing property
        const LAMBDA: f32 = 1.0507009873554804934193349852946;
        const ALPHA: f32 = 1.6732632423543772848170429916717;

        // Scalar implementation: selu(x) = λ * (x if x > 0 else α * (exp(x) - 1))
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| {
                if x > 0.0 {
                    LAMBDA * x
                } else {
                    LAMBDA * ALPHA * (x.exp() - 1.0)
                }
            })
            .collect();

        Ok(Vector::from_vec(data))
    }

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
        if self.data.is_empty() {
            return Ok(0.0);
        }

        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let result = unsafe {
            match self.backend {
                Backend::Scalar => ScalarBackend::norm_l2(&self.data),
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
                Backend::GPU | Backend::Auto => ScalarBackend::norm_l2(&self.data),
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

        // Divide each element by the norm using scalar multiplication
        // This avoids creating an intermediate vector
        self.scale(1.0 / norm)
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

        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let result = unsafe {
            match self.backend {
                Backend::Scalar => ScalarBackend::norm_l1(&self.data),
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
                Backend::GPU | Backend::Auto => ScalarBackend::norm_l1(&self.data),
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
        if self.data.is_empty() {
            return Ok(0.0);
        }

        // Use optimized SIMD backend for single-pass abs+max
        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        let max_abs = unsafe {
            match self.backend {
                Backend::Scalar => ScalarBackend::norm_linf(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => Sse2Backend::norm_linf(&self.data),
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => Avx2Backend::norm_linf(&self.data),
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::norm_linf(&self.data)
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => ScalarBackend::norm_linf(&self.data), // NEON fallback
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => ScalarBackend::norm_linf(&self.data),
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => ScalarBackend::norm_linf(&self.data), // WASM fallback
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => ScalarBackend::norm_linf(&self.data),
                Backend::GPU | Backend::Auto => ScalarBackend::norm_linf(&self.data),
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
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend {
                    Backend::Scalar => ScalarBackend::abs(&self.data, &mut result_data),
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => Sse2Backend::abs(&self.data, &mut result_data),
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::abs(&self.data, &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => NeonBackend::abs(&self.data, &mut result_data),
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => WasmBackend::abs(&self.data, &mut result_data),
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        // Auto should have been resolved at creation time
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[allow(unreachable_patterns)]
                    _ => ScalarBackend::abs(&self.data, &mut result_data),
                }
            }
        }

        Ok(Vector {
            data: result_data,
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
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
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
                    Backend::WasmSIMD => {
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

        if !self.data.is_empty() {
            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
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
                    Backend::WasmSIMD => {
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

        if !self.data.is_empty() {
            // Use parallel processing for large arrays
            #[cfg(feature = "parallel")]
            {
                const PARALLEL_THRESHOLD: usize = 100_000;
                const CHUNK_SIZE: usize = 65536;

                if self.len() >= PARALLEL_THRESHOLD {
                    use rayon::prelude::*;

                    self.data
                        .par_chunks(CHUNK_SIZE)
                        .zip(result_data.par_chunks_mut(CHUNK_SIZE))
                        .for_each(|(chunk_in, chunk_out)| {
                            dispatch_unary_op!(self.backend, sqrt, chunk_in, chunk_out);
                        });

                    return Ok(Vector {
                        data: result_data,
                        backend: self.backend,
                    });
                }
            }

            dispatch_unary_op!(self.backend, sqrt, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
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

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, recip, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
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
        let pow_data: Vec<f32> = self.data.iter().map(|x| x.powf(n)).collect();
        Ok(Vector {
            data: pow_data,
            backend: self.backend,
        })
    }

    /// Element-wise exponential: result\[i\] = e^x\[i\]
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
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            // Use parallel processing for large arrays
            #[cfg(feature = "parallel")]
            {
                const PARALLEL_THRESHOLD: usize = 100_000;
                const CHUNK_SIZE: usize = 65536;

                if self.len() >= PARALLEL_THRESHOLD {
                    use rayon::prelude::*;

                    self.data
                        .par_chunks(CHUNK_SIZE)
                        .zip(result_data.par_chunks_mut(CHUNK_SIZE))
                        .for_each(|(chunk_in, chunk_out)| {
                            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
                            unsafe {
                                match self.backend {
                                    Backend::Scalar => ScalarBackend::exp(chunk_in, chunk_out),
                                    #[cfg(target_arch = "x86_64")]
                                    Backend::SSE2 | Backend::AVX => {
                                        Sse2Backend::exp(chunk_in, chunk_out)
                                    }
                                    #[cfg(target_arch = "x86_64")]
                                    Backend::AVX2 | Backend::AVX512 => {
                                        Avx2Backend::exp(chunk_in, chunk_out)
                                    }
                                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                                    Backend::NEON => NeonBackend::exp(chunk_in, chunk_out),
                                    #[cfg(target_arch = "wasm32")]
                                    Backend::WasmSIMD => WasmBackend::exp(chunk_in, chunk_out),
                                    Backend::GPU => ScalarBackend::exp(chunk_in, chunk_out),
                                    Backend::Auto => ScalarBackend::exp(chunk_in, chunk_out),
                                    #[allow(unreachable_patterns)]
                                    _ => ScalarBackend::exp(chunk_in, chunk_out),
                                }
                            }
                        });

                    return Ok(Vector {
                        data: result_data,
                        backend: self.backend,
                    });
                }
            }

            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend {
                    Backend::Scalar => ScalarBackend::exp(&self.data, &mut result_data),
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => Sse2Backend::exp(&self.data, &mut result_data),
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::exp(&self.data, &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => NeonBackend::exp(&self.data, &mut result_data),
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => WasmBackend::exp(&self.data, &mut result_data),
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        // Auto should have been resolved at creation time
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[allow(unreachable_patterns)]
                    _ => ScalarBackend::exp(&self.data, &mut result_data),
                }
            }
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Element-wise natural logarithm: result\[i\] = ln(x\[i\])
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
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, ln, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Element-wise base-2 logarithm: result\[i\] = log₂(x\[i\])
    ///
    /// Computes the base-2 logarithm for each element.
    /// Uses Rust's optimized f32::log2() method.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 4.0, 8.0]);
    /// let result = v.log2().unwrap();
    /// // result ≈ [0.0, 1.0, 2.0, 3.0]
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `log2(1.0)` returns 0.0
    /// - `log2(2.0)` returns 1.0
    /// - `log2(x)` for x ≤ 0 returns NaN
    /// - `log2(0.0)` returns -∞
    /// - `log2(+∞)` returns +∞
    ///
    /// # Applications
    ///
    /// - Information theory: Entropy in bits, mutual information
    /// - Computer science: Bit manipulation, binary search complexity
    /// - Audio: Octave calculations, pitch detection
    /// - Data compression: Huffman coding, arithmetic coding
    pub fn log2(&self) -> Result<Vector<f32>> {
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, log2, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Element-wise base-10 logarithm: result\[i\] = log₁₀(x\[i\])
    ///
    /// Computes the base-10 (common) logarithm for each element.
    /// Uses Rust's optimized f32::log10() method.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 10.0, 100.0, 1000.0]);
    /// let result = v.log10().unwrap();
    /// // result ≈ [0.0, 1.0, 2.0, 3.0]
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `log10(1.0)` returns 0.0
    /// - `log10(10.0)` returns 1.0
    /// - `log10(x)` for x ≤ 0 returns NaN
    /// - `log10(0.0)` returns -∞
    /// - `log10(+∞)` returns +∞
    ///
    /// # Applications
    ///
    /// - Audio: Decibel calculations (dB = 20 * log10(amplitude))
    /// - Chemistry: pH calculations (-log10(H+ concentration))
    /// - Seismology: Richter scale
    /// - Scientific notation: Order of magnitude calculations
    pub fn log10(&self) -> Result<Vector<f32>> {
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, log10, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Element-wise sine: result\[i\] = sin(x\[i\])
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
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, sin, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Element-wise cosine: result\[i\] = cos(x\[i\])
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
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, cos, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
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
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, tan, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
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
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.tanh(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        let mut result = vec![0.0; self.len()];

        // Dispatch to appropriate SIMD backend
        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::tanh(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::tanh(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    Avx2Backend::tanh(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::tanh(&self.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::tanh(&self.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    ScalarBackend::tanh(&self.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::tanh(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    ScalarBackend::tanh(&self.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
                    // Auto should have been resolved at Vector creation
                    // GPU falls back to best available SIMD
                    #[cfg(target_arch = "x86_64")]
                    {
                        if is_x86_feature_detected!("avx2") {
                            Avx2Backend::tanh(&self.data, &mut result);
                        } else {
                            Sse2Backend::tanh(&self.data, &mut result);
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        ScalarBackend::tanh(&self.data, &mut result);
                    }
                }
            }
        }

        Ok(Vector {
            data: result,
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
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, floor, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
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
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, ceil, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
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
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, round, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
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
    /// Equivalent to `abs(self\[i\])` with the sign of `sign\[i\]`.
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

        let copysign_data: Vec<f32> = self
            .data
            .iter()
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

        let minimum_data: Vec<f32> = self
            .data
            .iter()
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

        let maximum_data: Vec<f32> = self
            .data
            .iter()
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

