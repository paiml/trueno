//! Backend implementations for different SIMD instruction sets
//!
//! This module contains the actual SIMD implementations for each backend.
//! All backends implement the same trait-based interface to ensure API consistency.
//!
//! # Safety
//!
//! All `unsafe` code is isolated within backend implementations. The public API
//! remains 100% safe.
//!
//! # Backends
//!
//! - `scalar`: Portable baseline implementation (no SIMD)
//! - `sse2`: x86_64 baseline SIMD (128-bit)
//! - `avx2`: x86_64 advanced SIMD (256-bit with FMA)
//! - `avx512`: x86_64 maximum SIMD (512-bit)
//! - `neon`: ARM SIMD (128-bit)
//! - `wasm`: WebAssembly SIMD128

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod sse2;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
pub mod neon;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

// Future backends
//
// #[cfg(target_arch = "x86_64")]
// pub mod avx512;

/// Backend trait defining common operations
///
/// All backend implementations must implement this trait to ensure
/// consistent behavior across different SIMD instruction sets.
///
/// # Safety
///
/// Implementations may use unsafe SIMD intrinsics. Callers must ensure:
/// - Input slices are valid
/// - Result slice has sufficient capacity
/// - Slices `a` and `b` have the same length
pub trait VectorBackend {
    /// Element-wise addition: a[i] + b[i]
    ///
    /// # Safety
    ///
    /// - `a` and `b` must have the same length
    /// - `result` must have length >= `a.len()`
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]);

    /// Element-wise subtraction: a[i] - b[i]
    ///
    /// # Safety
    ///
    /// - `a` and `b` must have the same length
    /// - `result` must have length >= `a.len()`
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]);

    /// Element-wise multiplication: a[i] * b[i]
    ///
    /// # Safety
    ///
    /// - `a` and `b` must have the same length
    /// - `result` must have length >= `a.len()`
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]);

    /// Element-wise division: a[i] / b[i]
    ///
    /// # Safety
    ///
    /// - `a` and `b` must have the same length
    /// - `result` must have length >= `a.len()`
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]);

    /// Dot product: sum(a[i] * b[i])
    ///
    /// # Safety
    ///
    /// - `a` and `b` must have the same length
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32;

    /// Sum reduction: sum(a[i])
    ///
    /// # Safety
    ///
    /// - `a` must not be empty
    unsafe fn sum(a: &[f32]) -> f32;

    /// Max reduction: max(a[i])
    ///
    /// # Safety
    ///
    /// - `a` must not be empty
    unsafe fn max(a: &[f32]) -> f32;

    /// Min reduction: min(a[i])
    ///
    /// # Safety
    ///
    /// - `a` must not be empty
    unsafe fn min(a: &[f32]) -> f32;

    /// Argmax: index of maximum value
    ///
    /// Returns the index of the first occurrence of the maximum value.
    ///
    /// # Safety
    ///
    /// - `a` must not be empty
    unsafe fn argmax(a: &[f32]) -> usize;

    /// Argmin: index of minimum value
    ///
    /// Returns the index of the first occurrence of the minimum value.
    ///
    /// # Safety
    ///
    /// - `a` must not be empty
    unsafe fn argmin(a: &[f32]) -> usize;

    /// Kahan summation: numerically stable sum(a[i])
    ///
    /// Uses the Kahan summation algorithm to reduce floating-point rounding errors
    /// when summing many numbers. Tracks a running compensation for lost low-order bits.
    ///
    /// # Safety
    ///
    /// - Can handle empty slice (returns 0.0)
    unsafe fn sum_kahan(a: &[f32]) -> f32;

    /// L2 norm (Euclidean norm): sqrt(sum(a[i]^2))
    ///
    /// Computes the Euclidean length of the vector. This is equivalent to sqrt(dot(a, a)).
    ///
    /// # Safety
    ///
    /// - Can handle empty slice (returns 0.0)
    unsafe fn norm_l2(a: &[f32]) -> f32;

    /// L1 norm (Manhattan norm): sum(|a[i]|)
    ///
    /// Computes the sum of absolute values of all elements.
    /// Used in machine learning (L1 regularization), distance metrics, and sparse modeling.
    ///
    /// # Safety
    ///
    /// - Can handle empty slice (returns 0.0)
    unsafe fn norm_l1(a: &[f32]) -> f32;

    /// Scalar multiplication: result[i] = a[i] * scalar
    ///
    /// Multiplies all elements by a scalar value.
    /// Used in vector scaling, normalization, and linear transformations.
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slice
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]);

    /// Clamp elements to range [min_val, max_val]: result[i] = max(min_val, min(a[i], max_val))
    ///
    /// Constrains each element to the specified range.
    /// Used in neural networks (gradient clipping), graphics (color clamping), and signal processing.
    ///
    /// # Safety
    ///
    /// - `result` must have the same length as `a`
    /// - Can handle empty slice
    /// - Assumes min_val <= max_val (caller must validate)
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]);
}
