//! cbtop - Compute Block Top
//!
//! Real-time load testing and hardware monitoring TUI built on the Brick Architecture.
//!
// Allow new_without_default - explicit new() is clearer for these types
#![allow(clippy::new_without_default)]
// Allow derivable_impls - explicit Default is clearer
#![allow(clippy::derivable_impls)]
// Allow missing_panics_doc - not critical for internal methods
#![allow(clippy::missing_panics_doc)]
// Allow missing_errors_doc - errors are self-explanatory
#![allow(clippy::missing_errors_doc)]
// Allow unnecessary_map_or - map_or is clearer
#![allow(clippy::unnecessary_map_or)]
// Allow collapsible_if - clarity over conciseness
#![allow(clippy::collapsible_if)]
// Allow needless_range_loop - clearer in some cases
#![allow(clippy::needless_range_loop)]
// Allow cast_precision_loss - acceptable for display values
#![allow(clippy::cast_precision_loss)]
// Allow cast_possible_truncation - handled appropriately
#![allow(clippy::cast_possible_truncation)]
// Allow dead_code - development in progress
#![allow(dead_code)]
// Allow field_reassign_with_default - clearer initialization
#![allow(clippy::field_reassign_with_default)]
// Allow manual_flatten - clearer error handling
#![allow(clippy::manual_flatten)]
//!
//! # Design Philosophy
//!
//! - **Test-as-Interface**: Every component is a falsifiable Brick (PROBAR-SPEC-009)
//! - **presentar-terminal**: All widgets and canvas from presentar-terminal (no custom reimplementation)
//! - **Toyota Way**: Jidoka, Poka-Yoke, Genchi Genbutsu principles throughout
//!
//! # Architecture
//!
//! ```text
//! Layer 4: Load Generators  → SimdLoadBrick, CudaLoadBrick, WgpuLoadBrick
//! Layer 3: Panels           → Overview, CPU, GPU, PCIe, Memory, Thermal
//! Layer 2: Analyzers        → Throughput, Bottleneck, Thermal
//! Layer 1: Collectors       → CPU, GPU, PCIe, Memory, Thermal
//! ```
//!
//! # Widget Source Policy
//!
//! All widgets and canvas implementations come from `presentar-terminal`.
//! cbtop does NOT implement its own widgets. If a widget is missing, it MUST
//! be added to presentar-terminal FIRST, then used here.

pub mod brick;
pub mod bricks;
pub mod ring_buffer;
pub mod app;
pub mod config;
pub mod error;
pub mod headless;

// Core brick traits (cbtop-specific)
pub use brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
// ComputeBrick Scoring Framework (§29)
pub use brick::{BrickScore, BrickGrade, Scorable};

// Application
pub use app::CbtopApp;
pub use config::Config;
pub use error::CbtopError;

// Re-export presentar-terminal widgets and canvas for convenience
// All widgets MUST come from presentar-terminal - DO NOT reimplement
pub use presentar_terminal::{
    BrailleGraph, GraphMode, Meter, Table,
    ColorMode,
};
pub use presentar_terminal::direct::{
    CellBuffer, DiffRenderer, DirectTerminalCanvas,
};

// Re-export presentar-core traits
pub use presentar_core::{
    Canvas, Color, Point, Rect, Size, TextStyle, Constraints,
};
