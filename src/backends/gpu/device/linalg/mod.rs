//! GPU linear algebra operations
//!
//! Matrix multiplication, vector addition, dot product, and 2D convolution.

pub mod cached_matmul;
pub mod wgsl_forward;
mod convolve2d;
mod dot;
mod matmul;
mod vec_ops;
