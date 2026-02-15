//! Tensor Core GEMM variants (simulated 16x16 and true WMMA FP16)
//!
//! ## Submodules
//!
//! - [`simulated`]: Simulated Tensor Core GEMM using 16x16 shared memory tiles
//! - [`wmma`]: True WMMA FP16 GEMM using hardware Tensor Core PTX intrinsics

mod simulated;
mod wmma;
