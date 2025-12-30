//! PTX Optimization Passes
//!
//! Inspired by CUDA Tile IR patterns (spec: cuda-tile-behavior.md).
//!
//! ## Available Passes
//!
//! - **FMA Fusion**: Detect `mul` + `add` patterns and fuse to `fma`
//! - **Tile Validation**: Validate tile constraints to prevent register pressure issues
//!
//! ## Usage
//!
//! ```rust,ignore
//! use trueno_gpu::ptx::optimize::{fma_fusion, tile_validation};
//!
//! let instructions = vec![/* PTX instructions */];
//! let fused = fma_fusion::pass(instructions);
//! tile_validation::validate(&fused)?;
//! ```
//!
//! ## Academic Foundation
//!
//! - FMA fusion based on Click & Paleczny (1995) SSA pattern matching
//! - Tile constraints based on Volkov & Demmel (2008) GPU optimization

pub mod fma_fusion;
pub mod tile_validation;

use super::instructions::PtxInstruction;
use crate::error::Result;

/// Apply all optimization passes to a sequence of PTX instructions.
///
/// # Arguments
///
/// * `instructions` - The PTX instructions to optimize
///
/// # Returns
///
/// Optimized instruction sequence
///
/// # cuda-tile-behavior.md References
///
/// - Section 3.5: FMA Fusion Detection
/// - Section 3.4: Tile Dimension Constraints
pub fn optimize(instructions: Vec<PtxInstruction>) -> Result<Vec<PtxInstruction>> {
    // Pass 1: FMA fusion
    let fused = fma_fusion::pass(instructions);

    // Pass 2: Tile validation (just validates, doesn't transform)
    tile_validation::validate(&fused)?;

    Ok(fused)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::instructions::PtxOp;
    use crate::ptx::types::PtxType;

    #[test]
    fn test_optimize_empty() {
        let instructions = vec![];
        let result = optimize(instructions).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_optimize_passthrough() {
        // Instructions that shouldn't be modified
        let instructions = vec![PtxInstruction::new(PtxOp::Ret, PtxType::Pred)];
        let result = optimize(instructions).unwrap();
        assert_eq!(result.len(), 1);
    }
}
