//! Q6_K GEMV Kernels
//!
//! Implements Q6_K quantized GEMV operations for decode throughput.
//!
//! ## Q6_K Layout (210 bytes per 256 values)
//!
//! - ql[128]: bytes 0-127, low 4-bits packed 2 per byte
//! - qh[64]: bytes 128-191, high 2-bits packed 4 per byte
//! - scales[16]: bytes 192-207, signed i8 per 16-element sub-block
//! - d: bytes 208-209, f16 scale factor
//!
//! ## Kernels
//!
//! - [`Q6KGemvKernel`]: Basic Q6_K GEMV with warp shuffle reduction
//! - [`CoalescedQ6KGemvKernel`]: Optimized with vectorized scale loading (PAR-066)
//! - [`BatchedQ6KGemvKernel`]: Batched version for M>1 processing (PAR-130)
//! - [`Q6KKernel`]: Fused Q6_K GEMM kernel (PARITY-117)

mod batched;
mod coalesced;
mod gemm;
mod gemv;

pub use batched::BatchedQ6KGemvKernel;
pub use coalesced::CoalescedQ6KGemvKernel;
pub use gemm::Q6KKernel;
pub use gemv::Q6KGemvKernel;
