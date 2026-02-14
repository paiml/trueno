//! RoPE (Rotary Position Embedding) Kernels
//!
//! GPU kernels for rotary position embeddings in transformer models.
//!
//! ## Kernel Variants
//!
//! - `RopeKernel`: Standard adjacent-pair RoPE
//! - `RopeIndirectKernel`: CUDA Graph compatible version
//! - `RopeNeoxKernel`: NEOX/GPT-NeoX style (split halves)
//! - `RopeNeoxIndirectKernel`: CUDA Graph compatible NEOX version
//! - `BatchedRopeKernel`: Multi-sequence batched RoPE
//! - `PreciseRopeKernel`: High-precision for theta=1M (Qwen2.5)
//! - `PreciseRopeIndirectKernel`: Precise + CUDA Graph compatible

mod standard;
mod neox;
mod batched;
mod precise;

pub use standard::{RopeKernel, RopeIndirectKernel};
pub use neox::{RopeNeoxKernel, RopeNeoxIndirectKernel};
pub use batched::BatchedRopeKernel;
pub use precise::{PreciseRopeKernel, PreciseRopeIndirectKernel};

#[cfg(test)]
mod tests;
