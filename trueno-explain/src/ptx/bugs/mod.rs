//! PTX Bug Detection - Static Analysis for GPU Kernel Bugs
//!
//! Shattered from monolithic bugs.rs into:
//! - `types` - Bug severity, classification, and report types
//! - `analyzer` - PtxBugAnalyzer implementation
//! - `coverage` - PTX coverage tracking
//!
//! Bug Classes (from probar `gpu_pixels`):
//! - P0 Critical: `SharedMemU64Addressing`, `LoopBranchToEnd`, `MissingBarrierSync`
//! - P1 High: `NonInPlaceLoopAccumulator`, `RegisterSpills`
//! - P2 Medium: `RedundantMoves`, `UnoptimizedMemoryPattern`

mod types;
mod analyzer;
mod coverage;

pub use types::{BugSeverity, PtxBug, PtxBugClass, PtxBugReport};
pub use analyzer::{PtxBugAnalyzer, WhitelistEntry};
pub use coverage::{PtxCoverageReport, PtxCoverageTracker, PtxCoverageTrackerBuilder, PtxFeature};

#[cfg(test)]
mod tests;
