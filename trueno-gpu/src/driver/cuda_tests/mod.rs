//! CUDA Driver Tests (PMAT-018: 95% Coverage Strike)
//!
//! These tests REQUIRE CUDA hardware. They WILL NOT SKIP.
//! The RTX 4090 is present. Execute the tests.

#![cfg(all(test, feature = "cuda"))]

use super::context::{cuda_available, device_count, get_driver, CudaContext};
use super::graph::{CaptureMode, CudaGraph};
use super::memory::GpuBuffer;
use super::module::CudaModule;
use super::stream::CudaStream;
use super::types::LaunchConfig;
use std::ffi::c_void;

mod driver_and_context;
mod streams;
mod gpu_buffer;
mod cuda_graph_tests;
mod module_tests;
mod stress_and_advanced;
