//! # trueno-gpu: Pure Rust PTX Generation for NVIDIA CUDA
//!
//! Generate PTX assembly directly from Rust - no LLVM, no nvcc, no external dependencies.
//!
//! ## Philosophy
//!
//! **Own the Stack** - Build everything from first principles for complete control,
//! auditability, and reproducibility.
//!
//! ## Quick Start
//!
//! ```rust
//! use trueno_gpu::ptx::{PtxModule, PtxKernel, PtxType};
//!
//! // Build a vector addition kernel
//! let module = PtxModule::new()
//!     .version(8, 0)
//!     .target("sm_70")
//!     .address_size(64);
//!
//! let ptx_source = module.emit();
//! assert!(ptx_source.contains(".version 8.0"));
//! ```
//!
//! ## Modules
//!
//! - [`ptx`] - PTX code generation (builder pattern)
//! - [`driver`] - CUDA driver API (minimal FFI, optional)
//! - [`kernels`] - Hand-optimized GPU kernels
//! - [`memory`] - GPU memory management
//! - [`backend`] - Multi-backend abstraction

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![deny(unsafe_op_in_unsafe_fn)]
// ============================================================================
// Development-phase lint allows - to be addressed incrementally
// ============================================================================
// Allow dead code during development - will be used as API expands
#![allow(dead_code)]
// Allow precision loss in non-critical floating point calculations
#![allow(clippy::cast_precision_loss)]
// Allow possible truncation - we handle 64-bit correctly
#![allow(clippy::cast_possible_truncation)]
// Allow format push string - not a critical performance path
#![allow(clippy::format_push_string)]
// Allow doc markdown for code references - these are placeholders
#![allow(clippy::doc_markdown)]
// Allow missing errors doc during initial development
#![allow(clippy::missing_errors_doc)]
// Allow unnecessary literal bound for backend trait
#![allow(clippy::unnecessary_literal_bound)]
// Allow manual div_ceil - will use std when stabilized
#![allow(clippy::manual_div_ceil)]
// Allow missing panics doc during initial development
#![allow(clippy::missing_panics_doc)]
// Allow cast_lossless - we intentionally use as for u32->u64
#![allow(clippy::cast_lossless)]
// Allow uninlined format args - stylistic preference
#![allow(clippy::uninlined_format_args)]
// Allow map_unwrap_or - more readable with map().unwrap_or()
#![allow(clippy::map_unwrap_or)]
// Allow redundant closure for method calls - clearer intent
#![allow(clippy::redundant_closure_for_method_calls)]
// Allow unused self - methods will use self as API expands
#![allow(clippy::unused_self)]
// Allow expect_used in tests and non-critical paths
#![allow(clippy::expect_used)]
// Allow too_many_lines during development - will be refactored
#![allow(clippy::too_many_lines)]
// Allow needless_range_loop - clearer intent in some algorithms
#![allow(clippy::needless_range_loop)]
// Allow float_cmp in tests where exact comparison is intended
#![allow(clippy::float_cmp)]
// Allow unused comparisons - some are defensive checks
#![allow(unused_comparisons)]
// Allow unwrap_used in tests
#![allow(clippy::unwrap_used)]
// Allow cast_sign_loss - we know values are positive
#![allow(clippy::cast_sign_loss)]
// Allow field_reassign_with_default - clearer test setup
#![allow(clippy::field_reassign_with_default)]
// Allow panic in tests
#![allow(clippy::panic)]
// Allow manual_range_contains - clearer in assertions
#![allow(clippy::manual_range_contains)]
// Allow default_constructed_unit_structs
#![allow(clippy::default_constructed_unit_structs)]
// Allow clone_on_copy - clearer intent
#![allow(clippy::clone_on_copy)]
// Allow absurd_extreme_comparisons - defensive checks
#![allow(clippy::absurd_extreme_comparisons)]
// Allow no_effect_underscore_binding - intentional in tests
#![allow(clippy::no_effect_underscore_binding)]
// Allow useless_vec in tests - clearer intent
#![allow(clippy::useless_vec)]

pub mod backend;
pub mod driver;
pub mod kernels;
pub mod memory;
pub mod monitor;
pub mod ptx;

/// Error types for trueno-gpu operations
pub mod error;

/// E2E visual testing framework for GPU kernels
pub mod testing;

/// WASM visual testing bindings (requires viz feature)
#[cfg(feature = "viz")]
pub mod wasm;

pub use error::{GpuError, Result};
pub use monitor::{cuda_device_count, cuda_monitoring_available, CudaDeviceInfo, CudaMemoryInfo};

#[cfg(test)]
mod tests {
    #[test]
    fn test_crate_compiles() {
        // Smoke test - crate compiles
        let _ = super::error::Result::<()>::Ok(());
    }
}
