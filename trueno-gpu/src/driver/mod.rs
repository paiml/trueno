//! CUDA Driver API (Minimal FFI)
//!
//! Provides minimal bindings to the CUDA driver API for module loading and kernel execution.
//! Only enabled with the `cuda` feature.
//!
//! ## Design Philosophy
//!
//! **OWN THE STACK**: Hand-written FFI for the ~20 CUDA driver functions we need.
//! No external dependencies (cudarc, cuda-sys). Total control.
//!
//! - **Minimal FFI**: Only bind what we need (~400 lines in sys.rs)
//! - **Safe wrappers**: All unsafe code isolated in sys.rs
//! - **RAII**: Automatic cleanup for contexts, modules, streams, memory
//! - **Typestate pattern**: Compile-time GPU state machine verification (Poka-Yoke)
//!
//! ## Modules
//!
//! - [`sys`] - Raw CUDA FFI (unsafe, internal use only)
//! - [`context`] - CUDA context management with Primary Context API
//! - [`module`] - PTX loading and JIT compilation
//! - [`stream`] - Async execution streams
//! - [`memory`] - GPU memory allocation and transfer
//!
//! ## Example
//!
//! ```ignore
//! use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer};
//!
//! // Create context for GPU 0
//! let ctx = CudaContext::new(0)?;
//!
//! // Load PTX module
//! let ptx = include_str!("kernel.ptx");
//! let mut module = CudaModule::from_ptx(&ctx, ptx)?;
//!
//! // Allocate GPU memory
//! let data: Vec<f32> = vec![1.0; 1024];
//! let mut buf = GpuBuffer::from_host(&ctx, &data)?;
//!
//! // Create stream and launch kernel
//! let stream = CudaStream::new(&ctx)?;
//! // ... launch kernel ...
//! stream.synchronize()?;
//!
//! // Download results
//! let mut result = vec![0.0f32; 1024];
//! buf.copy_to_host(&mut result)?;
//! ```

// FFI layer - uses FFI-specific patterns that trigger clippy lints
// (borrow_as_ptr, ptr_as_ptr, cast_sign_loss, wildcard_imports are normal for CUDA bindings)
#[cfg(feature = "cuda")]
#[allow(
    clippy::borrow_as_ptr,
    clippy::ptr_as_ptr,
    clippy::cast_sign_loss,
    clippy::wildcard_imports
)]
pub mod sys;

#[cfg(feature = "cuda")]
#[allow(clippy::borrow_as_ptr, clippy::ptr_as_ptr, clippy::cast_sign_loss)]
mod context;
#[cfg(feature = "cuda")]
#[allow(clippy::ptr_as_ptr, clippy::borrow_as_ptr)]
mod memory;
#[cfg(feature = "cuda")]
#[allow(clippy::borrow_as_ptr, clippy::ptr_as_ptr)]
mod module;
#[cfg(feature = "cuda")]
#[allow(clippy::borrow_as_ptr)]
mod stream;

// Re-export for use without cuda feature (types only)
mod types;
pub use types::*;

// Re-export CUDA wrappers when feature enabled
#[cfg(feature = "cuda")]
pub use context::{cuda_available, device_count, CudaContext};
#[cfg(feature = "cuda")]
pub use memory::GpuBuffer;
#[cfg(feature = "cuda")]
pub use module::CudaModule;
#[cfg(feature = "cuda")]
pub use stream::{CudaStream, DEFAULT_STREAM};

/// Check if CUDA is available at runtime
///
/// Returns `true` if:
/// - CUDA driver is installed (libcuda.so/nvcuda.dll exists)
/// - At least one CUDA device is available
/// - cuInit succeeds
#[cfg(not(feature = "cuda"))]
#[must_use]
pub fn cuda_available() -> bool {
    false
}

/// Get the number of CUDA devices
///
/// Returns 0 if CUDA is not available.
#[cfg(not(feature = "cuda"))]
#[must_use]
pub fn device_count() -> usize {
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_available_returns_bool() {
        // Just verify it compiles and returns a bool
        let _available: bool = cuda_available();
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_cuda_available_without_feature() {
        assert!(!cuda_available());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_device_count_without_feature() {
        assert_eq!(device_count(), 0);
    }
}
