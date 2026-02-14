//! Fused Q4K GEMV Kernels
//!
//! These kernels fuse multiple operations to reduce memory bandwidth:
//! - FusedRmsNormQ4KGemvKernel: RMSNorm + Q4K GEMV in single pass
//! - FusedGateUpQ4KGemvKernel: Gate + Up projections sharing input load
//! - FusedRmsNormGateUpSwigluQ4KKernel: RMSNorm + Gate+Up + SwiGLU (3-way fusion)

mod gate_up_gemv;
mod rmsnorm_gate_up_swiglu;
mod rmsnorm_gemv;

pub use gate_up_gemv::FusedGateUpQ4KGemvKernel;
pub use rmsnorm_gate_up_swiglu::FusedRmsNormGateUpSwigluQ4KKernel;
pub use rmsnorm_gemv::FusedRmsNormQ4KGemvKernel;
