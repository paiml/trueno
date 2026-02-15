use super::super::*;

// ========================================================================
// TCB Geometry tests
// ========================================================================

#[test]
fn test_tcb_geometry_creation() {
    let geom = TcbGeometry::new(4, 8, 256);
    assert_eq!(geom.m, 4);
    assert_eq!(geom.n, 8);
    assert_eq!(geom.k, 256);
    assert_eq!(geom.alignment, 16);
}

#[test]
fn test_tcb_geometry_alignment() {
    let geom = TcbGeometry::with_alignment(4, 16, 128, 64);
    assert_eq!(geom.alignment, 64);
}

#[test]
#[should_panic(expected = "TCB dimensions must be non-zero")]
fn test_tcb_geometry_zero_dimension() {
    let _ = TcbGeometry::new(0, 8, 256);
}

#[test]
#[should_panic(expected = "Alignment must be power of 2")]
fn test_tcb_geometry_invalid_alignment() {
    let _ = TcbGeometry::with_alignment(4, 8, 256, 17);
}

#[test]
fn test_arithmetic_intensity() {
    // 4×8×256 tile
    let geom = TcbGeometry::new(4, 8, 256);
    let ai = geom.arithmetic_intensity();
    // AI = 2*4*8*256 / ((4*256 + 256*8) * 4) = 16384 / 12288 ≈ 1.33
    assert!((ai - 1.33).abs() < 0.1);
}

#[test]
fn test_q4k_alignment() {
    let aligned = TcbGeometry::new(4, 8, 256);
    assert!(aligned.is_q4k_aligned());

    let unaligned = TcbGeometry::new(4, 8, 128);
    assert!(!unaligned.is_q4k_aligned());
}

#[test]
fn test_cache_fitting() {
    let geom = TcbGeometry::new(64, 64, 64);
    // A: 64*64*4 = 16KB, B: 64*64*4 = 16KB, total = 32KB
    assert!(geom.fits_in_cache(64 * 1024)); // 64KB cache
    assert!(!geom.fits_in_cache(16 * 1024)); // 16KB cache
}

// F334: TcbGeometry helper methods
#[test]
fn test_tcb_geometry_helpers() {
    let geom = TcbGeometry::new(8, 16, 256);

    // total_elements = m * n
    assert_eq!(geom.total_elements(), 8 * 16);

    // total_flops = 2 * m * n * k
    assert_eq!(geom.total_flops(), 2 * 8 * 16 * 256);

    // Tile bytes
    assert_eq!(geom.a_tile_bytes(), 8 * 256 * 4);
    assert_eq!(geom.b_tile_bytes(), 256 * 16 * 4);
    assert_eq!(geom.c_tile_bytes(), 8 * 16 * 4);

    // Q4_0 alignment
    assert!(geom.is_q4_0_aligned()); // 256 % 32 == 0
    let unaligned = TcbGeometry::new(4, 4, 17);
    assert!(!unaligned.is_q4_0_aligned());
}

// F335: TcbGeometry Display
#[test]
fn test_tcb_geometry_display() {
    let geom = TcbGeometry::new(4, 8, 256);
    let display = format!("{}", geom);
    assert!(display.contains("TCB"));
    assert!(display.contains("4×8×256"));
    assert!(display.contains("align=16"));
    assert!(display.contains("AI="));
}

// F336: TcbGeometry Default
#[test]
fn test_tcb_geometry_default() {
    let geom = TcbGeometry::default();
    assert_eq!(geom.m, 4);
    assert_eq!(geom.n, 4);
    assert_eq!(geom.k, 4);
    assert_eq!(geom.alignment, 16);
}

// F337: TcbLevel typical cache bytes
#[test]
fn test_tcb_level_cache_bytes() {
    assert_eq!(TcbLevel::Macro.typical_cache_bytes(), 32 * 1024 * 1024);
    assert_eq!(TcbLevel::Midi.typical_cache_bytes(), 256 * 1024);
    assert_eq!(TcbLevel::Micro.typical_cache_bytes(), 32 * 1024);
}

// F349: Serialization round-trip for TcbGeometry
#[test]
fn test_tcb_geometry_serde() {
    let geom = TcbGeometry::with_alignment(4, 8, 256, 64);
    let json = serde_json::to_string(&geom).unwrap();
    let decoded: TcbGeometry = serde_json::from_str(&json).unwrap();
    assert_eq!(geom, decoded);
}

// F353: Large tile calculations
#[test]
fn test_large_tile_arithmetic() {
    // Very large geometry to test u64 arithmetic
    let geom = TcbGeometry::new(10000, 10000, 1000);
    let total = geom.total_elements();
    assert_eq!(total, 100_000_000);

    let flops = geom.total_flops();
    assert_eq!(flops, 200_000_000_000);
}

// ========================================================================
// TilingConfig tests
// ========================================================================

#[test]
fn test_tiling_config_gpu_q4k_matvec() {
    let config = TilingConfig::gpu_q4k_matvec();
    assert_eq!(config.macro_tile.m, 1);
    assert_eq!(config.macro_tile.k, 256);
    assert!(config.macro_tile.is_q4k_aligned());
    assert!(config.validate().is_ok());
}

#[test]
fn test_tiling_config_cpu_avx2() {
    let config = TilingConfig::cpu_avx2_matmul();
    assert_eq!(config.micro_tile.n, 8); // AVX2 = 8 floats
    assert!(config.validate().is_ok());
}

#[test]
fn test_tiling_config_validation_failure() {
    let mut config = TilingConfig::cpu_avx2_matmul();
    // Make midi larger than macro (invalid)
    config.midi_tile.m = config.macro_tile.m + 1;
    assert!(config.validate().is_err());
}

// TILE-004: AVX-512 Register Tiling Tests
#[test]
fn test_tiling_config_avx512_matmul() {
    let config = TilingConfig::cpu_avx512_matmul();
    assert_eq!(config.micro_tile.n, 16); // AVX-512 = 16 floats
    assert_eq!(config.micro_tile.alignment, 64); // 64-byte alignment
    assert!(config.validate().is_ok());
}

#[test]
fn test_tiling_config_avx512_q4k_matvec() {
    let config = TilingConfig::cpu_avx512_q4k_matvec();
    assert!(config.micro_tile.is_q4k_aligned());
    assert_eq!(config.micro_tile.m, 4); // 4×1 micro-kernel
    assert_eq!(config.micro_tile.n, 1); // Single output column (matvec)
    assert!(config.validate().is_ok());
}

#[test]
fn test_tiling_config_avx512_vnni() {
    let config = TilingConfig::cpu_avx512_vnni_q4k_q8k();
    assert!(config.micro_tile.is_q4k_aligned());
    assert_eq!(config.backend, TilingBackend::CpuAvx512);
    assert!(config.validate().is_ok());
}

#[test]
fn test_avx512_vs_avx2_tile_sizes() {
    let avx2 = TilingConfig::cpu_avx2_matmul();
    let avx512 = TilingConfig::cpu_avx512_matmul();

    // AVX-512 should have 2x wider micro-tiles
    assert_eq!(avx512.micro_tile.n, avx2.micro_tile.n * 2);

    // AVX-512 should have stricter alignment
    assert!(avx512.micro_tile.alignment >= avx2.micro_tile.alignment);
}

// F338: TilingConfig factory methods
#[test]
fn test_tiling_config_gpu_softmax() {
    let config = TilingConfig::gpu_softmax();
    assert_eq!(config.name, "Softmax_GPU");
    assert_eq!(config.macro_tile.m, 1);
    assert_eq!(config.macro_tile.n, 32000); // Vocab size
    assert_eq!(config.backend, TilingBackend::Gpu);
    assert!(config.validate().is_ok());
}

#[test]
fn test_tiling_config_cpu_rmsnorm() {
    let config = TilingConfig::cpu_rmsnorm();
    assert_eq!(config.name, "RMSNorm_CPU");
    assert_eq!(config.macro_tile.m, 1);
    assert_eq!(config.backend, TilingBackend::CpuAvx512);
    assert!(config.validate().is_ok());
}

#[test]
fn test_tiling_config_gpu_q4k_matmul() {
    let config = TilingConfig::gpu_q4k_matmul();
    assert_eq!(config.name, "Q4K_MatMul_GPU");
    assert_eq!(config.macro_tile.m, 128);
    assert!(config.macro_tile.is_q4k_aligned());
    assert!(config.validate().is_ok());
}

#[test]
fn test_tiling_config_cpu_avx2_q4k_matvec() {
    let config = TilingConfig::cpu_avx2_q4k_matvec();
    assert_eq!(config.name, "Q4K_MatVec_AVX2");
    assert!(config.micro_tile.is_q4k_aligned());
    assert_eq!(config.backend, TilingBackend::CpuAvx2);
    assert!(config.validate().is_ok());
}

// F350: Serialization round-trip for TilingConfig
#[test]
fn test_tiling_config_serde() {
    let config = TilingConfig::cpu_avx512_matmul();
    let json = serde_json::to_string(&config).unwrap();
    let decoded: TilingConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config.name, decoded.name);
    assert_eq!(config.backend, decoded.backend);
}

// F330: Midi/micro tile divisibility
#[test]
fn test_tile_divisibility() {
    let config = TilingConfig::cpu_avx512_matmul();

    // Macro should be divisible by midi
    assert_eq!(config.macro_tile.m % config.midi_tile.m, 0);
    assert_eq!(config.macro_tile.n % config.midi_tile.n, 0);

    // Midi should be divisible by micro
    assert_eq!(config.midi_tile.m % config.micro_tile.m, 0);
    assert_eq!(config.midi_tile.n % config.micro_tile.n, 0);
}

// F343: TilingConfig validation - divisibility errors
#[test]
fn test_tiling_config_divisibility_error() {
    let mut config = TilingConfig::cpu_avx2_matmul();
    // Make midi not divide evenly into macro
    config.midi_tile.m = 17; // 256 % 17 != 0
    let result = config.validate();
    assert!(result.is_err());
    if let Err(TilingError::DivisibilityError { level, .. }) = result {
        assert_eq!(level, "macro/midi");
    } else {
        panic!("Expected DivisibilityError");
    }
}

#[test]
fn test_tiling_config_micro_divisibility_error() {
    let mut config = TilingConfig::cpu_avx2_matmul();
    // Make micro not divide evenly into midi
    config.micro_tile.m = 17; // midi.m % 17 != 0
    let result = config.validate();
    assert!(result.is_err());
}

// F341: TilingError Display implementations
#[test]
fn test_tiling_error_display() {
    let err1 = TilingError::InvalidHierarchy {
        reason: "test".into(),
    };
    assert!(format!("{}", err1).contains("Invalid tiling hierarchy"));
    assert!(format!("{}", err1).contains("test"));

    let err2 = TilingError::DivisibilityError {
        level: "macro/midi",
        dimension: "M",
        larger: 256,
        smaller: 17,
    };
    assert!(format!("{}", err2).contains("Tiling divisibility error"));
    assert!(format!("{}", err2).contains("256"));

    let err3 = TilingError::CacheOverflow {
        level: TcbLevel::Midi,
        required_bytes: 1000,
        available_bytes: 500,
    };
    assert!(format!("{}", err3).contains("exceeds"));
    assert!(format!("{}", err3).contains("Midi"));

    let err4 = TilingError::AlignmentError {
        required: 64,
        actual: 32,
    };
    assert!(format!("{}", err4).contains("Alignment error"));

    let err5 = TilingError::QuantAlignmentError {
        format: "Q4_K",
        required_k: 256,
        actual_k: 100,
    };
    assert!(format!("{}", err5).contains("Quantization alignment"));
    assert!(format!("{}", err5).contains("Q4_K"));
}

// F342: TilingError as std::error::Error
#[test]
fn test_tiling_error_trait() {
    let err = TilingError::InvalidHierarchy {
        reason: "test".into(),
    };
    // Ensure it implements Error trait
    let _: &dyn std::error::Error = &err;
}

// F346: TilingBackend enum
#[test]
fn test_tiling_backend_equality() {
    assert_eq!(TilingBackend::CpuAvx2, TilingBackend::CpuAvx2);
    assert_ne!(TilingBackend::CpuAvx2, TilingBackend::CpuAvx512);
    assert_ne!(TilingBackend::Gpu, TilingBackend::Scalar);
    assert_eq!(TilingBackend::CpuNeon, TilingBackend::CpuNeon);
}

// F347: PackingLayout enum
#[test]
fn test_packing_layout_equality() {
    assert_eq!(PackingLayout::RowMajor, PackingLayout::RowMajor);
    assert_ne!(PackingLayout::RowMajor, PackingLayout::ColumnMajor);
    assert_ne!(PackingLayout::PanelMajorA, PackingLayout::PanelMajorB);
}

// F348: TcbLevel enum
#[test]
fn test_tcb_level_equality() {
    assert_eq!(TcbLevel::Macro, TcbLevel::Macro);
    assert_ne!(TcbLevel::Macro, TcbLevel::Midi);
    assert_ne!(TcbLevel::Midi, TcbLevel::Micro);
}
