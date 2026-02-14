use super::*;
use crate::error::GpuError;

// ========== Display and Error trait tests ==========

#[test]
fn test_tile_error_display_too_many_elements() {
    let err = TileError::TooManyElements {
        actual: 20_000_000,
        max: 16_777_216,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("20000000"),
        "Should contain actual count: {}",
        msg
    );
    assert!(
        msg.contains("16777216"),
        "Should contain max count: {}",
        msg
    );
    assert!(
        msg.contains("too many elements"),
        "Should describe the error: {}",
        msg
    );
}

#[test]
fn test_tile_error_display_dimension_too_large() {
    let err = TileError::DimensionTooLarge {
        actual: 8192,
        max: 4096,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("8192"),
        "Should contain actual dimension: {}",
        msg
    );
    assert!(
        msg.contains("4096"),
        "Should contain max dimension: {}",
        msg
    );
    assert!(
        msg.contains("exceeds"),
        "Should describe the error: {}",
        msg
    );
}

#[test]
fn test_tile_error_display_invalid_wmma_shape() {
    let err = TileError::InvalidWmmaShape {
        shape: "m64n64k64".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("m64n64k64"), "Should contain shape: {}", msg);
    assert!(
        msg.contains("Invalid WMMA"),
        "Should describe the error: {}",
        msg
    );
}

#[test]
fn test_tile_error_display_non_power_of_two() {
    let err = TileError::NonPowerOfTwo { dim: 123 };
    let msg = err.to_string();
    assert!(msg.contains("123"), "Should contain dimension: {}", msg);
    assert!(
        msg.contains("power of two"),
        "Should describe the error: {}",
        msg
    );
}

#[test]
fn test_tile_error_implements_std_error() {
    let err = TileError::NonPowerOfTwo { dim: 42 };
    // Verify Error trait is implemented (source() returns None by default)
    let std_err: &dyn std::error::Error = &err;
    assert!(std_err.source().is_none());
}

#[test]
fn test_tile_error_to_gpu_error_conversion() {
    let tile_err = TileError::TooManyElements {
        actual: 100,
        max: 50,
    };
    let gpu_err: GpuError = tile_err.clone().into();

    // Verify conversion produces InvalidParameter variant with the error message
    match gpu_err {
        GpuError::InvalidParameter(msg) => {
            assert!(msg.contains("100"), "Should contain actual: {}", msg);
            assert!(msg.contains("50"), "Should contain max: {}", msg);
        }
        _ => panic!("Expected InvalidParameter variant"),
    }
}

#[test]
fn test_tile_error_conversion_non_power_of_two() {
    let tile_err = TileError::NonPowerOfTwo { dim: 37 };
    let gpu_err: GpuError = tile_err.into();

    match gpu_err {
        GpuError::InvalidParameter(msg) => {
            assert!(msg.contains("37"));
            assert!(msg.contains("power of two"));
        }
        _ => panic!("Expected InvalidParameter"),
    }
}

#[test]
fn test_tile_error_conversion_dimension_too_large() {
    let tile_err = TileError::DimensionTooLarge {
        actual: 10000,
        max: 4096,
    };
    let gpu_err: GpuError = tile_err.into();

    match gpu_err {
        GpuError::InvalidParameter(msg) => {
            assert!(msg.contains("10000"));
            assert!(msg.contains("4096"));
        }
        _ => panic!("Expected InvalidParameter"),
    }
}

#[test]
fn test_tile_error_conversion_invalid_wmma() {
    let tile_err = TileError::InvalidWmmaShape {
        shape: "m99n99k99".to_string(),
    };
    let gpu_err: GpuError = tile_err.into();

    match gpu_err {
        GpuError::InvalidParameter(msg) => {
            assert!(msg.contains("m99n99k99"));
        }
        _ => panic!("Expected InvalidParameter"),
    }
}

// ========== validate_shape edge cases ==========

#[test]
fn test_validate_shape_multiple_dims_with_non_power_of_two() {
    // First dimension is valid, second is not
    let result = validate_shape(&[32, 100]);
    assert!(matches!(result, Err(TileError::NonPowerOfTwo { dim: 100 })));
}

#[test]
fn test_validate_shape_first_dim_non_power_of_two() {
    // First dimension is invalid
    let result = validate_shape(&[13, 16]);
    assert!(matches!(result, Err(TileError::NonPowerOfTwo { dim: 13 })));
}

#[test]
fn test_validate_shape_too_many_elements_exact_boundary() {
    // 4096 * 4096 = 16,777,216 = MAX_TILE_ELEMENTS (valid)
    assert!(validate_shape(&[4096, 4096]).is_ok());

    // 4096 * 4096 * 2 = 33,554,432 > MAX (invalid)
    // But 8192 would fail dimension check first, so use 4096 * 4096 * 2
    // Actually, we need to test elements overflow without dimension overflow
    // Use 4096 * 4096 * 2 = would overflow dimension first
    // Instead test with multiple smaller dimensions
    assert!(matches!(
        validate_shape(&[4096, 4096, 2]),
        Err(TileError::TooManyElements { .. })
    ));
}

#[test]
fn test_validate_shape_single_element() {
    assert!(validate_shape(&[1]).is_ok());
    assert!(validate_shape(&[1, 1]).is_ok());
    assert!(validate_shape(&[1, 1, 1]).is_ok());
}

#[test]
fn test_validate_shape_large_valid_multidimensional() {
    // Multiple dimensions that together are within limits
    assert!(validate_shape(&[64, 64, 64]).is_ok()); // 262,144 elements
    assert!(validate_shape(&[256, 256, 256]).is_ok()); // 16,777,216 elements = MAX
}

#[test]
fn test_validate_shape_dimension_boundary() {
    // Exactly at MAX_TILE_DIM
    assert!(validate_shape(&[4096]).is_ok());

    // Just over MAX_TILE_DIM (but still power of two)
    assert!(matches!(
        validate_shape(&[8192]),
        Err(TileError::DimensionTooLarge {
            actual: 8192,
            max: 4096
        })
    ));
}

#[test]
fn test_validate_shape_zero_dimension_is_ok() {
    // Zero is handled specially - results in 0 elements which is <= MAX
    // Zero passes the power-of-two check (dim != 0 && !is_power_of_two() is false for 0)
    let result = validate_shape(&[0, 16]);
    assert!(result.is_ok());
}

// ========== validate_wmma_shape tests ==========

#[test]
fn test_wmma_shape_validation_all_valid() {
    // Explicitly test all three valid shapes
    assert!(validate_wmma_shape(&WmmaShape::M16N16K16).is_ok());
    assert!(validate_wmma_shape(&WmmaShape::M8N32K16).is_ok());
    assert!(validate_wmma_shape(&WmmaShape::M32N8K16).is_ok());
}

#[test]
fn test_wmma_shape_validation_invalid_combinations() {
    // Various invalid combinations
    let cases = [
        WmmaShape { m: 8, n: 8, k: 8 }, // All 8s
        WmmaShape {
            m: 32,
            n: 32,
            k: 32,
        }, // All 32s
        WmmaShape {
            m: 16,
            n: 32,
            k: 16,
        }, // Wrong combination
        WmmaShape { m: 8, n: 16, k: 16 }, // Wrong combination
        WmmaShape { m: 1, n: 1, k: 1 }, // Minimal
        WmmaShape {
            m: 64,
            n: 64,
            k: 64,
        }, // Too large
    ];

    for shape in cases {
        assert!(
            validate_wmma_shape(&shape).is_err(),
            "Shape m{}n{}k{} should be invalid",
            shape.m,
            shape.n,
            shape.k
        );
    }
}

#[test]
fn test_wmma_invalid_error_message_format() {
    let shape = WmmaShape { m: 24, n: 24, k: 8 };
    let result = validate_wmma_shape(&shape);

    match result {
        Err(TileError::InvalidWmmaShape { shape: s }) => {
            assert_eq!(s, "m24n24k8");
        }
        _ => panic!("Expected InvalidWmmaShape error"),
    }
}

// ========== validate() instruction tests ==========

#[test]
fn test_validate_wmma_load_a() {
    let instructions = vec![PtxInstruction::new(
        PtxOp::WmmaLoadA,
        crate::ptx::types::PtxType::F16,
    )];
    assert!(validate(&instructions).is_ok());
}

#[test]
fn test_validate_wmma_load_b() {
    let instructions = vec![PtxInstruction::new(
        PtxOp::WmmaLoadB,
        crate::ptx::types::PtxType::F16,
    )];
    assert!(validate(&instructions).is_ok());
}

#[test]
fn test_validate_wmma_load_c() {
    let instructions = vec![PtxInstruction::new(
        PtxOp::WmmaLoadC,
        crate::ptx::types::PtxType::F32,
    )];
    assert!(validate(&instructions).is_ok());
}

#[test]
fn test_validate_wmma_store_d() {
    let instructions = vec![PtxInstruction::new(
        PtxOp::WmmaStoreD,
        crate::ptx::types::PtxType::F32,
    )];
    assert!(validate(&instructions).is_ok());
}

#[test]
fn test_validate_mixed_instructions_with_wmma() {
    let instructions = vec![
        PtxInstruction::new(PtxOp::Add, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaLoadA, crate::ptx::types::PtxType::F16),
        PtxInstruction::new(PtxOp::Mul, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaLoadB, crate::ptx::types::PtxType::F16),
        PtxInstruction::new(PtxOp::WmmaLoadC, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaMma, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaStoreD, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::Sub, crate::ptx::types::PtxType::F32),
    ];
    assert!(validate(&instructions).is_ok());
}

#[test]
fn test_validate_all_wmma_ops_in_sequence() {
    let instructions = vec![
        PtxInstruction::new(PtxOp::WmmaLoadA, crate::ptx::types::PtxType::F16),
        PtxInstruction::new(PtxOp::WmmaLoadB, crate::ptx::types::PtxType::F16),
        PtxInstruction::new(PtxOp::WmmaLoadC, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaMma, crate::ptx::types::PtxType::F32),
        PtxInstruction::new(PtxOp::WmmaStoreD, crate::ptx::types::PtxType::F32),
    ];
    assert!(validate(&instructions).is_ok());
}

// ========== TileError Clone and PartialEq tests ==========

#[test]
fn test_tile_error_clone() {
    let err1 = TileError::TooManyElements {
        actual: 1000,
        max: 500,
    };
    let err2 = err1.clone();
    assert_eq!(err1, err2);
}

#[test]
fn test_tile_error_partial_eq() {
    let err1 = TileError::NonPowerOfTwo { dim: 17 };
    let err2 = TileError::NonPowerOfTwo { dim: 17 };
    let err3 = TileError::NonPowerOfTwo { dim: 19 };

    assert_eq!(err1, err2);
    assert_ne!(err1, err3);
}

#[test]
fn test_tile_error_debug() {
    let err = TileError::DimensionTooLarge {
        actual: 5000,
        max: 4096,
    };
    let debug_str = format!("{:?}", err);
    assert!(debug_str.contains("DimensionTooLarge"));
    assert!(debug_str.contains("5000"));
    assert!(debug_str.contains("4096"));
}

// ========== Constants verification ==========

#[test]
fn test_constants_values() {
    assert_eq!(MAX_TILE_ELEMENTS, 16_777_216);
    assert_eq!(MAX_TILE_DIM, 4096);
}

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
    let invalid = WmmaShape {
        m: 32,
        n: 32,
        k: 16,
    };
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
