//! CUDA Driver API (Minimal FFI)
//!
//! Provides minimal bindings to the CUDA driver API for module loading and kernel execution.
//! Only enabled with the `cuda` feature.
//!
//! ## Design Philosophy
//!
//! - **Minimal FFI**: Only bind what we need
//! - **Safe wrappers**: All unsafe code isolated here
//! - **Typestate pattern**: Compile-time GPU state machine verification (Poka-Yoke)

#[cfg(feature = "cuda")]
mod context;
#[cfg(feature = "cuda")]
mod module;
#[cfg(feature = "cuda")]
mod stream;
#[cfg(feature = "cuda")]
mod memory;

// Re-export for use without cuda feature (types only)
mod types;
pub use types::*;

/// Check if CUDA is available at runtime
#[must_use]
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // TODO: Check for cuInit success
        false
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_available_without_feature() {
        // Without cuda feature, should return false
        #[cfg(not(feature = "cuda"))]
        assert!(!cuda_available());
    }
}
