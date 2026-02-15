//! Fuzz Testing Integration (PMAT-023)
//!
//! Property-based testing and fuzz-like input validation per section 36.3 to address
//! the Resilience score. Uses proptest for stable Rust compatibility.
//!
//! # Fuzz Targets
//!
//! | Target | Component | Description |
//! |--------|-----------|-------------|
//! | `fuzz_syscall_breakdown` | TracingEscalation | Syscall name/duration inputs |
//! | `fuzz_workload_metrics` | RooflineAnalysis | FLOP/byte/time values |
//! | `fuzz_escalation_thresholds` | TracingEscalation | Threshold configurations |
//! | `fuzz_hardware_profile` | HardwareProfile | Peak GFLOPS/bandwidth values |
//! | `fuzz_brick_scoring` | BrickScore | Score calculation inputs |
//!
//! # Falsification Criteria
//!
//! F1081-F1095: Input validation and error path testing

mod types;
mod suite;
mod edge_cases;

pub use types::*;
pub use suite::*;
pub use edge_cases::*;

#[cfg(test)]
mod tests;
