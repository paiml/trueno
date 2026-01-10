//! Collector bricks (Layer 1)
//!
//! Data collection from hardware (Genchi Genbutsu).

pub mod cpu;
pub mod gpu;
pub mod memory;
pub mod thermal;
pub mod pcie;
pub mod wos;
pub mod pepita;
pub mod zram;

pub use cpu::CpuCollectorBrick;
pub use gpu::GpuCollectorBrick;
pub use memory::MemoryCollectorBrick;
pub use thermal::ThermalCollectorBrick;
pub use pcie::PcieCollectorBrick;
pub use wos::{WosCollectorBrick, WosKernelMetrics, JidokaHealthStatus};
pub use pepita::{PepitaCollectorBrick, IoUringMetrics};
pub use zram::{ZramCollectorBrick, ZramMetrics, ZramAlgorithm};