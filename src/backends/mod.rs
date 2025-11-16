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

// Future backends
//
// #[cfg(target_arch = "x86_64")]
// pub mod avx512;
//
// #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
// pub mod neon;
//
// #[cfg(target_arch = "wasm32")]
// pub mod wasm;

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

    /// Element-wise multiplication: a[i] * b[i]
    ///
    /// # Safety
    ///
    /// - `a` and `b` must have the same length
    /// - `result` must have length >= `a.len()`
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]);

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
}
