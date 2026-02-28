//! WGSL compute shaders for GPU operations

mod advanced;
mod basic_ops;
mod reductions;

pub(crate) use advanced::*;
pub(crate) use basic_ops::*;
pub(crate) use reductions::*;
