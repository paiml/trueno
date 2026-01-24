//! Tiling Compute Blocks (TCB) - Work Partitioning for High-Performance Kernels
//!
//! TCBs represent the fundamental unit of work partitioning within `ComputeBrick` kernels.
//! While a `ComputeBrick` defines a logical operation (e.g., Q4_K MatMul), a TCB defines
//! the physical execution strategy—how data is partitioned across the memory hierarchy.
//!
//! # Architecture
//!
//! Tiling occurs at three levels:
//! 1. **Macro-Tile (L3/Global Memory)**: Partitioning across CPU sockets or GPU SMs
//! 2. **Midi-Tile (L2/Shared Memory)**: Partitioning within a thread block or Rayon task
//! 3. **Micro-Tile (Registers)**: Smallest unit processed by SIMD or CUDA warps
//!
//! # Modules
//!
//! - `geometry` - TcbGeometry dimensions and level definitions
//! - `config` - TilingConfig and backend selection
//! - `calculator` - TcbIndexCalculator for index computation
//! - `packing` - Memory layout packing utilities
//! - `prefetch` - Prefetch locality hints
//! - `q4k_matvec` - Q4_K quantized matrix-vector tiling
//! - `error` - TilingError types

mod geometry;
mod config;
mod calculator;
mod packing;
mod prefetch;
mod q4k_matvec;
mod error;

pub use geometry::{TcbGeometry, TcbLevel};
pub use config::{TilingConfig, TilingBackend};
pub use calculator::TcbIndexCalculator;
pub use packing::{PackingLayout, pack_a_index, pack_b_index, swizzle_index};
pub use prefetch::{PrefetchLocality, optimal_prefetch_distance};
pub use q4k_matvec::{TiledQ4KMatvec, TilingStats, Q4K_SUPERBLOCK_SIZE, Q4K_SUPERBLOCK_BYTES, f16_to_f32, extract_scale_min_6bit};
pub use error::TilingError;

#[cfg(test)]
mod tests;
