//! Load generator bricks (Layer 4)
//!
//! Generate controlled compute workloads.

mod simd;
mod cuda;
mod wgpu;
mod memory;

pub use simd::SimdLoadBrick;
pub use cuda::CudaLoadBrick;
pub use wgpu::WgpuLoadBrick;
pub use memory::MemBandwidthBrick;
