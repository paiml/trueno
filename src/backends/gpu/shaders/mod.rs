//! WGSL compute shaders for GPU operations

mod basic_ops;
mod reductions;
mod advanced;

pub use basic_ops::*;
pub use reductions::*;
pub use advanced::*;
