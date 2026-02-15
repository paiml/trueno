//! FP16 and Tensor Core Q4K Kernels
//!
//! High-performance quantized inference kernels optimized for memory bandwidth
//! and tensor core utilization.
//!
//! ## Kernels
//!
//! - [`Fp16Q4KGemvKernel`] - FP16 input/output Q4K GEMV with 4x bandwidth reduction
//! - [`TensorCoreQ4KGemmKernel`] - Tensor Core accelerated Q4K GEMM for batched decode

mod fp16_gemv;
mod tensor_core_gemm;

pub use fp16_gemv::Fp16Q4KGemvKernel;
pub use tensor_core_gemm::TensorCoreQ4KGemmKernel;
