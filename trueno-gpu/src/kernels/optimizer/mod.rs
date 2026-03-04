//! Optimizer Kernels
//!
//! GPU kernels for fused optimizer weight updates, eliminating CPU↔GPU synchronization.
//!
//! ## Available Kernels
//!
//! - [`AdamWStepKernel`]: Fused AdamW with weight decay
//! - [`AdamStepKernel`]: Vanilla Adam without weight decay
//! - [`GradientClipKernel`]: L2 gradient norm clipping
//! - [`SquaredSumKernel`]: GPU-side sum-of-squares reduction for L2 norm (KAIZEN-049)
//!
//! ## Performance Benefits
//!
//! Traditional training loop:
//! 1. Forward pass (GPU)
//! 2. Backward pass (GPU)
//! 3. Copy gradients GPU → CPU
//! 4. Optimizer step (CPU)
//! 5. Copy weights CPU → GPU
//!
//! With fused kernels:
//! 1. Forward pass (GPU)
//! 2. Backward pass (GPU)
//! 3. Optimizer step (GPU) ← All on GPU!
//!
//! This eliminates the PCIe bottleneck entirely.
//!
//! # Issue #89: Fused optimizer kernels

#![allow(clippy::similar_names)]

mod adamw;
mod clip;
mod squared_sum;

pub use adamw::{AdamStepKernel, AdamWStepKernel};
pub use clip::GradientClipKernel;
pub use squared_sum::SquaredSumKernel;
