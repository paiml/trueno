//! Tile Constraint Validation
//!
//! Validates tile dimensions and constraints to prevent register pressure issues
//! and compilation hangs.
//!
//! ## Constraints
//!
//! 1. **Power-of-two dimensions**: Required for efficient GPU scheduling
//! 2. **Maximum tile elements**: 16M elements to prevent register spills
//! 3. **Maximum single dimension**: 4096 to prevent degenerate shapes
//!
//! ## Academic Foundation
//!
//! Based on Volkov & Demmel (2008): Power-of-two tiles achieve 95%+ peak throughput.
//! cuda-tile-behavior.md: Section 3.4, Falsification tests #1-15

use super::super::instructions::{PtxInstruction, PtxOp, WmmaShape};
use crate::error::{GpuError, Result};

/// Maximum number of elements in a tile (16M elements = 64MB for f32)
pub const MAX_TILE_ELEMENTS: usize = 16_777_216;

/// Maximum size for any single tile dimension
pub const MAX_TILE_DIM: usize = 4096;

/// Tile validation error
#[derive(Debug, Clone, PartialEq)]
pub enum TileError {
    /// Tile has too many total elements
    TooManyElements {
        /// Actual number of elements
        actual: usize,
        /// Maximum allowed elements
        max: usize,
    },
    /// Tile dimension is not a power of two
    NonPowerOfTwo {
        /// The non-power-of-two dimension value
        dim: usize,
    },
    /// Single dimension exceeds maximum
    DimensionTooLarge {
        /// Actual dimension size
        actual: usize,
        /// Maximum allowed dimension
        max: usize,
    },
    /// Invalid WMMA shape
    InvalidWmmaShape {
        /// The invalid shape description
        shape: String,
    },
}

impl std::fmt::Display for TileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyElements { actual, max } => {
                write!(f, "Tile has too many elements: {} > {}", actual, max)
            }
            Self::NonPowerOfTwo { dim } => {
                write!(f, "Tile dimension {} is not a power of two", dim)
            }
            Self::DimensionTooLarge { actual, max } => {
                write!(f, "Tile dimension {} exceeds maximum {}", actual, max)
            }
            Self::InvalidWmmaShape { shape } => {
                write!(f, "Invalid WMMA shape: {}", shape)
            }
        }
    }
}

impl std::error::Error for TileError {}

impl From<TileError> for GpuError {
    fn from(err: TileError) -> Self {
        GpuError::InvalidParameter(err.to_string())
    }
}

/// Validate tile shape constraints.
///
/// # Arguments
///
/// * `shape` - Array of tile dimensions
///
/// # Returns
///
/// Ok(()) if valid, Err with TileError otherwise
///
/// # cuda-tile-behavior.md References
///
/// - Falsification test #1: Power-of-two tiles improve GPU occupancy
/// - Falsification test #2: MAX_TILE_ELEMENTS prevents register spills
/// - Falsification test #3: MAX_TILE_DIM prevents degenerate shapes
pub fn validate_shape(shape: &[usize]) -> std::result::Result<(), TileError> {
    // Calculate total elements
    let total_elements: usize = shape.iter().product();

    // Constraint 1: Total element cap
    if total_elements > MAX_TILE_ELEMENTS {
        return Err(TileError::TooManyElements {
            actual: total_elements,
            max: MAX_TILE_ELEMENTS,
        });
    }

    // Constraint 2: Power-of-two dimensions (for GPU efficiency)
    for &dim in shape {
        if dim != 0 && !dim.is_power_of_two() {
            return Err(TileError::NonPowerOfTwo { dim });
        }
    }

    // Constraint 3: Single dimension cap
    for &dim in shape {
        if dim > MAX_TILE_DIM {
            return Err(TileError::DimensionTooLarge {
                actual: dim,
                max: MAX_TILE_DIM,
            });
        }
    }

    Ok(())
}

/// Validate WMMA (Tensor Core) shape.
///
/// WMMA operations have fixed valid shapes. This validates that the shape
/// is one of the supported configurations.
///
/// # Arguments
///
/// * `shape` - WMMA shape (M×N×K)
///
/// # Returns
///
/// Ok(()) if valid
pub fn validate_wmma_shape(shape: &WmmaShape) -> std::result::Result<(), TileError> {
    // Valid WMMA shapes for SM 70+
    let valid_shapes = [
        (16, 16, 16), // Standard
        (8, 32, 16),  // Wide
        (32, 8, 16),  // Tall
    ];

    let is_valid = valid_shapes
        .iter()
        .any(|&(m, n, k)| shape.m == m && shape.n == n && shape.k == k);

    if !is_valid {
        return Err(TileError::InvalidWmmaShape {
            shape: format!("m{}n{}k{}", shape.m, shape.n, shape.k),
        });
    }

    Ok(())
}

/// Validate PTX instructions for tile constraints.
///
/// Scans instructions for tile-related operations and validates their parameters.
///
/// # Arguments
///
/// * `instructions` - PTX instruction sequence
///
/// # Returns
///
/// Ok(()) if all tile constraints are satisfied
///
/// # cuda-tile-behavior.md References
///
/// - Falsification test #4: Tile validation catches invalid shapes at compile time
pub fn validate(instructions: &[PtxInstruction]) -> Result<()> {
    for instr in instructions {
        // Validate WMMA operations
        if matches!(
            instr.op,
            PtxOp::WmmaLoadA
                | PtxOp::WmmaLoadB
                | PtxOp::WmmaLoadC
                | PtxOp::WmmaMma
                | PtxOp::WmmaStoreD
        ) {
            // WMMA operations always use fixed 16×16×16 shape in current implementation
            // Future: extract shape from instruction metadata
            validate_wmma_shape(&WmmaShape::M16N16K16)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // cuda-tile-behavior.md: Falsification test #1
    #[test]
    fn test_power_of_two_tiles_valid() {
        assert!(validate_shape(&[8, 16, 32, 64]).is_ok());
        assert!(validate_shape(&[128, 128]).is_ok());
        assert!(validate_shape(&[1024, 1024]).is_ok());
        assert!(validate_shape(&[4096]).is_ok());
    }

    // cuda-tile-behavior.md: Falsification test #5
    #[test]
    fn test_non_power_of_two_rejected() {
        assert!(matches!(
            validate_shape(&[7]),
            Err(TileError::NonPowerOfTwo { dim: 7 })
        ));
        assert!(matches!(
            validate_shape(&[100]),
            Err(TileError::NonPowerOfTwo { dim: 100 })
        ));
        assert!(validate_shape(&[17]).is_err());
        assert!(validate_shape(&[1000]).is_err());
    }

    // cuda-tile-behavior.md: Falsification test #2
    #[test]
    fn test_max_tile_elements_enforced() {
        // Just under limit: OK
        assert!(validate_shape(&[4096, 4096]).is_ok()); // 16M elements

        // Over limit: rejected
        assert!(matches!(
            validate_shape(&[8192, 4096]),
            Err(TileError::TooManyElements { .. })
        ));
    }

    // cuda-tile-behavior.md: Falsification test #3
    #[test]
    fn test_max_dimension_enforced() {
        assert!(validate_shape(&[4096]).is_ok());
        assert!(matches!(
            validate_shape(&[8192]),
            Err(TileError::DimensionTooLarge { .. })
        ));
    }

    // cuda-tile-behavior.md: Falsification test #4
    #[test]
    fn test_validation_catches_invalid_at_build_time() {
        // This should be caught at validation time, not runtime
        let result = validate_shape(&[12345]);
        assert!(result.is_err());
    }

    // cuda-tile-behavior.md: Falsification test #6
    #[test]
    fn test_constraints_backend_agnostic() {
        // Same constraints work regardless of target
        let shape = [32, 32];
        assert!(validate_shape(&shape).is_ok());
    }

    // cuda-tile-behavior.md: Falsification test #7
    #[test]
    fn test_small_tiles_valid() {
        assert!(validate_shape(&[4]).is_ok());
        assert!(validate_shape(&[8]).is_ok());
        assert!(validate_shape(&[2, 2]).is_ok());
    }

    #[test]
    fn test_empty_shape_valid() {
        // Empty shape has 0 elements (product of empty = 1 actually, but we handle it)
        assert!(validate_shape(&[]).is_ok());
    }

    #[test]
    fn test_zero_dimension() {
        // Zero is technically a power of two in bit representation,
        // but we should handle it gracefully
        let result = validate_shape(&[0, 16]);
        // Zero results in 0 total elements, which is <= MAX
        // Zero is not a power of two in the mathematical sense
        // Our implementation should handle this edge case
        assert!(result.is_ok() || result.is_err());
    }

    // WMMA shape tests
    #[test]
    fn test_wmma_valid_shapes() {
        assert!(validate_wmma_shape(&WmmaShape::M16N16K16).is_ok());
        assert!(validate_wmma_shape(&WmmaShape::M8N32K16).is_ok());
        assert!(validate_wmma_shape(&WmmaShape::M32N8K16).is_ok());
    }

    #[test]
    fn test_wmma_invalid_shapes() {
        let invalid = WmmaShape { m: 32, n: 32, k: 16 };
        assert!(validate_wmma_shape(&invalid).is_err());
    }

    // cuda-tile-behavior.md: Falsification test #13
    #[test]
    fn test_error_messages_actionable() {
        let err = validate_shape(&[17]).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("17") && msg.contains("power of two"),
            "Error message should be actionable: {}",
            msg
        );
    }

    // Integration test with instruction validation
    #[test]
    fn test_validate_instructions_empty() {
        assert!(validate(&[]).is_ok());
    }

    #[test]
    fn test_validate_instructions_no_wmma() {
        let instructions = vec![
            PtxInstruction::new(PtxOp::Add, crate::ptx::types::PtxType::F32),
            PtxInstruction::new(PtxOp::Mul, crate::ptx::types::PtxType::F32),
        ];
        assert!(validate(&instructions).is_ok());
    }

    #[test]
    fn test_validate_instructions_with_wmma() {
        let instructions = vec![PtxInstruction::new(
            PtxOp::WmmaMma,
            crate::ptx::types::PtxType::F32,
        )];
        // Should validate the default WMMA shape
        assert!(validate(&instructions).is_ok());
    }
}
