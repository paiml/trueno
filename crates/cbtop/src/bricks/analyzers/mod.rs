//! Analyzer bricks (Layer 2)
//!
//! Business logic and derived metrics.

mod throughput;
mod bottleneck;
mod thermal;
mod efficiency;

pub use throughput::ThroughputAnalyzerBrick;
pub use bottleneck::BottleneckAnalyzerBrick;
pub use thermal::ThermalAnalyzerBrick;
pub use efficiency::EfficiencyAnalyzerBrick;
