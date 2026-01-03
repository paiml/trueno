//! TUI Compute Mode Flow: CPU, GPU, and Memory Monitoring
//!
//! TRUENO-SPEC-020: Unified compute device abstraction for real-time monitoring.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
//! │   CPU       │   │ NVIDIA GPU  │   │  AMD GPU    │
//! │  Backend    │   │   Backend   │   │   Backend   │
//! └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
//!        └────────────┬────┴────────────┬────┘
//!              ┌──────▼─────────────────▼──────┐
//!              │   Unified Telemetry Collector │
//!              └───────────────┬───────────────┘
//!                              │
//!              ┌───────────────▼───────────────┐
//!              │      TUI Renderer (ratatui)   │
//!              └───────────────────────────────┘
//! ```
//!
//! # References
//!
//! - [Volkov2008] Tile size optimization, memory bandwidth modeling
//! - [Liker2004] Toyota Way principles (Genchi Genbutsu, Jidoka)
//! - [LAMBDA-0002] Memory pressure levels specification

mod device;
mod memory;
mod compute;
mod data_flow;
mod tui_layout;
mod stress_test;

pub use device::*;
pub use memory::*;
pub use compute::*;
pub use data_flow::*;
pub use tui_layout::*;
pub use stress_test::*;

// Re-export legacy CUDA types for backwards compatibility
mod cuda;
pub use cuda::{
    cuda_device_count, cuda_monitoring_available, CudaDeviceInfo, CudaMemoryInfo,
};
