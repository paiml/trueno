//! Tensor Core GEMM variants (simulated 16x16 and true WMMA FP16)
//!
//! ## Submodules
//!
//! - [`simulated`]: Simulated Tensor Core GEMM using 16x16 shared memory tiles
//! - [`wmma`]: True WMMA FP16 GEMM using hardware Tensor Core PTX intrinsics
//! - [`cta_wmma`]: CTA-level WMMA with 4 warps sharing 32×32 tiles

pub mod cta_wmma;
mod simulated;
mod wmma;
