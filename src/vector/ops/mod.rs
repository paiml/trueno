//! Vector operation modules
//!
//! This module organizes Vector operations into focused submodules:
//! - `activations`: Neural network activation functions (relu, sigmoid, gelu, etc.)
//! - `arithmetic`: Binary arithmetic operations (add, sub, mul, div, fma, scale)
//! - `normalization`: Normalization methods (zscore, minmax, layer_norm, normalize)
//! - `norms`: Vector norm calculations (norm_l1, norm_l2, norm_linf)
//! - `reductions`: Reduction operations (sum, dot, max, min, argmax, mean, variance, etc.)
//! - `rounding`: Rounding and sign functions (floor, ceil, round, trunc, etc.)
//! - `transcendental`: Mathematical functions (exp, log, sin, cos, etc.)
//! - `transforms`: Element-wise transforms (abs, clamp, clip, lerp, sqrt, recip, pow)

pub mod activations;
pub mod arithmetic;
pub mod normalization;
pub mod norms;
pub mod reductions;
pub mod rounding;
pub mod transcendental;
pub mod transforms;
