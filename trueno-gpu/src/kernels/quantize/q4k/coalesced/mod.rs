//! Coalesced Q4_K GEMV Kernels with Optimized Memory Access
//!
//! - `CoalescedQ4KGemvKernel`: Scale loading via broadcast, vectorized qs
//! - `VectorizedQ4KGemvKernel`: Coalesced u32 loads for high bandwidth
//! - `MwvDp4aQ4KGemvKernel`: DP4A integer dot products with Q8_1-quantized activations

mod coalesced_kernel;
mod multi_warp_vectorized;
mod mwv_dp4a;
mod vectorized_kernel;
mod wide_kernel;

pub use coalesced_kernel::CoalescedQ4KGemvKernel;
pub use multi_warp_vectorized::MultiWarpVectorizedQ4KGemvKernel;
pub use mwv_dp4a::MwvDp4aQ4KGemvKernel;
pub use vectorized_kernel::VectorizedQ4KGemvKernel;
pub use wide_kernel::WideQ4KGemvKernel;
