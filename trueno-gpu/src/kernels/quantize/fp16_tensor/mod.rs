//! FP16 and Tensor Core Q4K Kernels
//!
//! High-performance quantized inference kernels optimized for memory bandwidth
//! and tensor core utilization.
//!
//! ## Kernels
//!
//! - [`Fp16Q4KGemvKernel`] - FP16 input/output Q4K GEMV with 4x bandwidth reduction
//! - [`TensorCoreQ4KGemmKernel`] - Tensor Core accelerated Q4K GEMM for batched decode
//! - [`MultiWarpTensorCoreQ4KGemmKernel`] - 4-warp WMMA Q4K GEMM (PMAT-045)

mod fp16_gemv;
mod mw_tensor_core_gemm;
mod tensor_core_gemm;

pub use fp16_gemv::Fp16Q4KGemvKernel;
pub use mw_tensor_core_gemm::MultiWarpTensorCoreQ4KGemmKernel;
pub use tensor_core_gemm::TensorCoreQ4KGemmKernel;
