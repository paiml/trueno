//! Panel bricks (Layer 3)
//!
//! Rendering using presentar-terminal widgets.

pub mod overview;
pub mod cpu;
pub mod gpu;
pub mod help;
pub mod memory;
pub mod thermal;
pub mod pcie;
pub mod load;
pub mod config;

pub use overview::OverviewPanelBrick;
pub use cpu::CpuPanelBrick;
pub use gpu::GpuPanelBrick;
pub use help::HelpPanelBrick;
pub use memory::MemoryPanelBrick;
pub use thermal::ThermalPanelBrick;
pub use pcie::PciePanelBrick;
pub use load::LoadControlPanelBrick;
pub use config::ConfigPanelBrick;