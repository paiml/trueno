//! Vector operation modules
//!
//! This module organizes Vector operations into focused submodules:
//! - `activations`: Neural network activation functions (relu, sigmoid, gelu, etc.)
//! - `arithmetic`: Binary arithmetic operations (add, sub, mul, div, fma, scale)
//! - `reductions`: Reduction operations (sum, dot, max, min, argmax, mean, variance, etc.)
//! - `rounding`: Rounding and sign functions (floor, ceil, round, trunc, etc.)
//! - `transcendental`: Mathematical functions (exp, log, sin, cos, etc.)

pub mod activations;
pub mod arithmetic;
pub mod reductions;
pub mod rounding;
pub mod transcendental;
