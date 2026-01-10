//! Brick implementations for cbtop
//!
//! Four-layer architecture following PROBAR-SPEC-009:
//! - Layer 1: Collectors - Data collection from hardware
//! - Layer 2: Analyzers - Business logic and derived metrics
//! - Layer 3: Panels - Rendering using presentar-terminal widgets
//! - Layer 4: Generators - Load generation

pub mod collectors;
pub mod analyzers;
pub mod panels;
pub mod generators;

// Re-exports
pub use collectors::*;
pub use analyzers::*;
pub use generators::*;
