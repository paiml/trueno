//! Vector operation modules
//!
//! This module organizes Vector operations into focused submodules:
//! - `activations`: Neural network activation functions (relu, sigmoid, gelu, etc.)
//! - `reductions`: Reduction operations (sum, dot, max, min, argmax, etc.)
//! - `arithmetic`: Binary arithmetic operations (add, sub, mul, div, fma)
//! - `transforms`: Normalization and transformation functions
//! - `transcendental`: Mathematical functions (exp, log, sin, cos, etc.)

pub mod activations;
pub mod rounding;
pub mod transcendental;
