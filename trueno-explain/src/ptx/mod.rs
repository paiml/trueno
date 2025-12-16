//! PTX analysis module
//!
//! Parses and analyzes NVIDIA PTX assembly to detect:
//! - Register pressure (Muda of Transport when spills occur)
//! - Memory access patterns (Muda of Waiting when uncoalesced)
//! - Warp divergence (Heijunka imbalance)

mod parser;

pub use parser::PtxAnalyzer;
