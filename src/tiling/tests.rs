//! Tiling module tests.

use super::*;

// F301: TCB Output Equivalence - tested via property tests
// F302: Tile Size Power of 2 - static analysis

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

#[test]
fn test_index_calculator_macro_offset() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config.clone(), 1024, 1024, 1024);

    let (row, col) = calc.macro_tile_offset(0);
    assert_eq!((row, col), (0, 0));

    let (_row, col) = calc.macro_tile_offset(1);
    assert_eq!(col, config.macro_tile.n);
}

#[test]
fn test_index_calculator_boundary() {
    let config = TilingConfig::cpu_avx2_matmul();

    // With 512×512 problem and 256×256 tiles, first tile is NOT a boundary
    let calc_large = TcbIndexCalculator::new(config.clone(), 512, 512, 256);
    assert!(!calc_large.is_boundary_tile(0));

    // With 100×100 problem and 256×256 tiles, first (only) tile IS a boundary
    let calc_small = TcbIndexCalculator::new(config, 100, 100, 256);
    assert!(calc_small.is_boundary_tile(0));

    // Actual dimensions should be clamped to problem size
    let (actual_m, actual_n) = calc_small.actual_tile_dims(0);
    assert_eq!(actual_m, 100);
    assert_eq!(actual_n, 100);
}

#[test]
fn test_pack_a_index() {
    // mr=4, kc=256, panel 0
    let idx = pack_a_index(0, 0, 4, 256, 64);
    assert_eq!(idx, 0);

    // Second element in first panel
    let idx = pack_a_index(1, 0, 4, 256, 64);
    assert_eq!(idx, 1);

    // First element, second k
    let idx = pack_a_index(0, 1, 4, 256, 64);
    assert_eq!(idx, 4);
}

#[test]
fn test_swizzle_index() {
    // XOR swizzling should avoid bank conflicts
    let idx0 = swizzle_index(0);
    let idx32 = swizzle_index(32);
    // These would conflict without swizzling (both bank 0)
    // With swizzling: 0 ^ 0 = 0, 32 ^ 1 = 33
    assert_ne!(idx0 % 32, idx32 % 32);
}

#[test]
fn test_optimal_prefetch_distance() {
    let geom = TcbGeometry::new(4, 8, 64);
    let dist = optimal_prefetch_distance(&geom, TcbLevel::Midi);
    assert!(dist >= 1);
}

// F321: Odd-Sized Matrix Handling
#[test]
fn test_odd_sized_matrices() {
    let config = TilingConfig::cpu_avx2_matmul();

    // Test various odd sizes
    for (m, n, k) in [(127, 255, 513), (1, 1, 1), (7, 13, 31)] {
        let calc = TcbIndexCalculator::new(config.clone(), m, n, k);
        let num_tiles = calc.num_k_blocks();
        assert!(num_tiles >= 1);
    }
}

// F322: Zero-Padding Efficiency
#[test]
fn test_tile_count_calculation() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config.clone(), 1024, 1024, 1024);

    let num_macro = calc.config.num_macro_tiles(1024, 1024);
    let num_midi = calc.config.midi_tiles_per_macro();
    let num_micro = calc.config.micro_tiles_per_midi();

    assert!(num_macro > 0);
    assert!(num_midi > 0);
    assert!(num_micro > 0);
}

// TILE-003: Q4K MatVec Tests
#[test]
fn test_tiled_q4k_matvec_creation() {
    let matvec = TiledQ4KMatvec::new(4096, 4096);
    assert_eq!(matvec.m, 4096);
    assert_eq!(matvec.k, 4096);
    assert_eq!(matvec.superblocks_per_row(), 16); // 4096 / 256
    assert_eq!(matvec.total_superblocks(), 4096 * 16);
}

#[test]
#[should_panic(expected = "K dimension")]
fn test_tiled_q4k_matvec_unaligned_k() {
    let _ = TiledQ4KMatvec::new(4096, 100); // Not aligned to 256
}

#[test]
fn test_tiled_q4k_matvec_weight_offset() {
    let matvec = TiledQ4KMatvec::new(100, 512);
    // Row 0: offset 0
    assert_eq!(matvec.weight_row_offset(0), 0);
    // Row 1: offset = 2 superblocks * 144 bytes = 288
    assert_eq!(matvec.weight_row_offset(1), 2 * Q4K_SUPERBLOCK_BYTES);
}

#[test]
fn test_tiled_q4k_matvec_optimal_rows() {
    let matvec = TiledQ4KMatvec::new(4096, 4096);
    // With 256KB L2, should fit many rows
    let rows = matvec.optimal_parallel_rows(256 * 1024);
    assert!(rows >= 4); // At least micro-kernel size
    assert!(rows <= 4096); // At most all rows
}

#[test]
fn test_tiled_q4k_matvec_stats() {
    let matvec = TiledQ4KMatvec::new(4096, 4096);
    let stats = matvec.stats();

    // Weight bytes: 4096 * 16 * 144 = 9,437,184 bytes
    assert_eq!(stats.superblocks, 4096 * 16);
    // Arithmetic ops: 4096 * 4096 * 2 = 33,554,432
    assert_eq!(stats.arithmetic_ops, 4096 * 4096 * 2);
    // AI should be reasonable for Q4K
    assert!(stats.arithmetic_intensity > 1.0);
}

#[test]
fn test_q4k_constants() {
    assert_eq!(Q4K_SUPERBLOCK_SIZE, 256);
    assert_eq!(Q4K_SUPERBLOCK_BYTES, 144);
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

// TILE-005: F321-F340 Boundary Handling Tests
// F321: Odd-Sized Matrix Handling (already exists above)

// F323: Single-element matrices
#[test]
fn test_single_element_matrix() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config, 1, 1, 256);

    assert!(calc.is_boundary_tile(0));
    let (actual_m, actual_n) = calc.actual_tile_dims(0);
    assert_eq!(actual_m, 1);
    assert_eq!(actual_n, 1);
}

// F324: Prime-sized matrices (no clean tiling)
#[test]
fn test_prime_sized_matrices() {
    let config = TilingConfig::cpu_avx2_matmul();

    // Prime sizes: 127, 251, 509 (all < macro_tile.m which is 256)
    for size in [127, 251] {
        let calc = TcbIndexCalculator::new(config.clone(), size, size, 256);
        let num_tiles = config.num_macro_tiles(size, size);
        assert!(num_tiles >= 1);

        // Tiles smaller than macro size are boundary tiles
        assert!(calc.is_boundary_tile(0));
    }

    // 509 > 256, so first tile is NOT a boundary, but second tile IS
    let calc = TcbIndexCalculator::new(config.clone(), 509, 509, 256);
    // First tile (0,0 to 255,255) is not boundary for 509×509
    assert!(!calc.is_boundary_tile(0));
    // Second tile (0,256 to 255,508) IS boundary (509-256=253 < 256)
    assert!(calc.is_boundary_tile(1));
}

// F325: K dimension exactly equals superblock
#[test]
fn test_k_equals_superblock() {
    let matvec = TiledQ4KMatvec::new(100, 256);
    assert_eq!(matvec.superblocks_per_row(), 1);
    assert_eq!(matvec.total_superblocks(), 100);
}

// F326: Very large M dimension
#[test]
fn test_large_m_dimension() {
    let matvec = TiledQ4KMatvec::new(100_000, 256);
    assert_eq!(matvec.superblocks_per_row(), 1);
    assert_eq!(matvec.total_superblocks(), 100_000);
    // Should still compute optimal rows
    let rows = matvec.optimal_parallel_rows(256 * 1024);
    assert!(rows >= 4);
}

// F327: Very large K dimension
#[test]
fn test_large_k_dimension() {
    let matvec = TiledQ4KMatvec::new(10, 32768); // 32K hidden dim
    assert_eq!(matvec.superblocks_per_row(), 128);
    let stats = matvec.stats();
    assert!(stats.arithmetic_intensity > 0.0);
}

// F328: Tile offset at boundaries
#[test]
fn test_tile_offset_boundaries() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config.clone(), 1000, 1000, 256);

    // Last tile index
    let num_tiles = config.num_macro_tiles(1000, 1000);
    let last_idx = num_tiles - 1;

    let (row, col) = calc.macro_tile_offset(last_idx);
    // Should be within bounds
    assert!(row < 1000 + config.macro_tile.m);
    assert!(col < 1000 + config.macro_tile.n);
}

// F329: Index calculator consistency
#[test]
fn test_index_calculator_consistency() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config.clone(), 512, 512, 256);

    // Macro offset for tile 0 should be (0, 0)
    let (r0, c0) = calc.macro_tile_offset(0);
    assert_eq!((r0, c0), (0, 0));

    // Linear offset should match
    let linear = calc.block_to_linear_offset(0, 512);
    assert_eq!(linear, 0);

    // A and B offsets at k_block=0 should also be 0
    let a_off = calc.a_offset(0, 0);
    let b_off = calc.b_offset(0, 0);
    assert_eq!(a_off, 0);
    assert_eq!(b_off, 0);
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

// F331: f16 to f32 conversion
#[test]
fn test_f16_conversion() {
    // Zero
    assert_eq!(f16_to_f32(&[0x00, 0x00]), 0.0);

    // One (0x3C00 in f16)
    let one = f16_to_f32(&[0x00, 0x3C]);
    assert!((one - 1.0).abs() < 0.001);

    // Negative one (0xBC00)
    let neg_one = f16_to_f32(&[0x00, 0xBC]);
    assert!((neg_one - (-1.0)).abs() < 0.001);

    // Infinity (0x7C00)
    assert!(f16_to_f32(&[0x00, 0x7C]).is_infinite());

    // NaN (0x7C01)
    assert!(f16_to_f32(&[0x01, 0x7C]).is_nan());
}

// F332: f16 subnormal conversion
#[test]
fn test_f16_subnormal() {
    // Smallest positive subnormal: 0x0001
    let subnormal = f16_to_f32(&[0x01, 0x00]);
    assert!(subnormal > 0.0);
    assert!(subnormal < 0.001); // Very small

    // Negative zero: 0x8000
    let neg_zero = f16_to_f32(&[0x00, 0x80]);
    assert_eq!(neg_zero, -0.0);
    assert!(neg_zero.is_sign_negative());

    // Negative infinity: 0xFC00
    let neg_inf = f16_to_f32(&[0x00, 0xFC]);
    assert!(neg_inf.is_infinite());
    assert!(neg_inf.is_sign_negative());
}

// F333: Execute scalar implementation
#[test]
fn test_execute_scalar() {
    let matvec = TiledQ4KMatvec::new(2, 256);

    // Create minimal valid Q4K weights (2 rows × 1 superblock each)
    let mut weights = vec![0u8; 2 * Q4K_SUPERBLOCK_BYTES];

    // Set up first row: d=1.0, dmin=0.0, all qs=0
    // f16 for 1.0 is 0x3C00
    weights[0] = 0x00;
    weights[1] = 0x3C;
    // dmin = 0
    weights[2] = 0x00;
    weights[3] = 0x00;
    // scales all zero (simplified)
    // qs all zero -> dequantized values will be 0

    // Second row: same setup
    let offset = Q4K_SUPERBLOCK_BYTES;
    weights[offset] = 0x00;
    weights[offset + 1] = 0x3C;

    let input = vec![1.0f32; 256];
    let mut output = vec![0.0f32; 2];

    matvec.execute_scalar(&weights, &input, &mut output);

    // With zero quantized values, output should be 0 or near 0
    // (The exact value depends on the scale/min extraction)
    assert!(output[0].is_finite());
    assert!(output[1].is_finite());
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

// F339: TcbIndexCalculator midi/micro offsets
#[test]
fn test_index_calculator_midi_offset() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config, 1024, 1024, 1024);

    let (row, col) = calc.midi_tile_offset(0);
    assert_eq!((row, col), (0, 0));

    let (row1, col1) = calc.midi_tile_offset(1);
    // Second midi tile should be one midi_tile.n to the right
    assert_eq!(row1, 0);
    assert!(col1 > 0);
}

#[test]
fn test_index_calculator_micro_offset() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config, 1024, 1024, 1024);

    let (row, col) = calc.micro_tile_offset(0);
    assert_eq!((row, col), (0, 0));

    let (row1, col1) = calc.micro_tile_offset(1);
    assert_eq!(row1, 0);
    assert!(col1 > 0);
}

// F340: pack_b_index
#[test]
fn test_pack_b_index() {
    // nr=8, kc=64, panel 0
    let idx = pack_b_index(0, 0, 8, 64, 64);
    assert_eq!(idx, 0);

    // Second element in first panel (col 1)
    let idx = pack_b_index(0, 1, 8, 64, 64);
    assert_eq!(idx, 1);

    // First element, second row
    let idx = pack_b_index(1, 0, 8, 64, 64);
    assert_eq!(idx, 8);

    // Second panel (col 8)
    let idx = pack_b_index(0, 8, 8, 64, 64);
    // panel 1 * 64 * 8 + 0 * 8 + 0 = 512
    assert_eq!(idx, 512);
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

// F344: TilingStats fields
#[test]
fn test_tiling_stats_complete() {
    let matvec = TiledQ4KMatvec::new(100, 512);
    let stats = matvec.stats();

    assert_eq!(stats.input_bytes, 512 * 4);
    assert_eq!(stats.output_bytes, 100 * 4);
    assert_eq!(stats.superblocks, 100 * 2); // 512/256 = 2 per row
    assert!(stats.total_weight_bytes > 0);
}

// F345: extract_scale_min_6bit function
#[test]
fn test_extract_scale_min_6bit() {
    // Test with known byte patterns
    let scales = [0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];

    // idx 0: scale from bits 0-5 of byte 0 = 0x3F = 63
    let (sc, _m) = extract_scale_min_6bit(&scales, 0);
    assert_eq!(sc, 63.0);

    // Test odd index
    let scales2 = [0xC0, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
    let (sc1, _m1) = extract_scale_min_6bit(&scales2, 1);
    assert!(sc1 >= 0.0); // Just ensure it doesn't panic
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

// F349: Serialization round-trip for TcbGeometry
#[test]
fn test_tcb_geometry_serde() {
    let geom = TcbGeometry::with_alignment(4, 8, 256, 64);
    let json = serde_json::to_string(&geom).unwrap();
    let decoded: TcbGeometry = serde_json::from_str(&json).unwrap();
    assert_eq!(geom, decoded);
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

// F351: Index calculator k_blocks
#[test]
fn test_index_calculator_k_blocks() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config.clone(), 512, 512, 1024);

    // 1024 / 256 = 4 K blocks
    assert_eq!(calc.num_k_blocks(), 4);

    // Non-divisible case
    let calc2 = TcbIndexCalculator::new(config, 512, 512, 300);
    // ceil(300 / 256) = 2
    assert_eq!(calc2.num_k_blocks(), 2);
}

// F352: A and B offset calculations
#[test]
fn test_ab_offset_calculations() {
    let config = TilingConfig::cpu_avx2_matmul();
    let calc = TcbIndexCalculator::new(config.clone(), 512, 512, 512);

    // A offset: row * problem_k + col
    let a_off = calc.a_offset(1, 0); // macro_row=1, k_block=0
    assert_eq!(a_off, (config.macro_tile.m * 512) as usize);

    // B offset: row * problem_n + col
    let b_off = calc.b_offset(0, 1); // k_block=0, macro_col=1
    assert_eq!(b_off, config.macro_tile.n as usize);
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

// F354: Prefetch with different levels
#[test]
fn test_prefetch_all_levels() {
    let geom = TcbGeometry::new(4, 8, 64);

    let dist_micro = optimal_prefetch_distance(&geom, TcbLevel::Micro);
    let dist_midi = optimal_prefetch_distance(&geom, TcbLevel::Midi);
    let dist_macro = optimal_prefetch_distance(&geom, TcbLevel::Macro);

    // Macro should have larger distance (higher latency)
    assert!(dist_macro >= dist_midi);
    assert!(dist_midi >= dist_micro);
    // All should be at least 1
    assert!(dist_micro >= 1);
}

// F355: PrefetchLocality Debug
#[test]
fn test_prefetch_locality_debug() {
    let loc = PrefetchLocality::T0;
    let debug = format!("{:?}", loc);
    assert!(debug.contains("T0"));

    let loc2 = PrefetchLocality::NonTemporal;
    let debug2 = format!("{:?}", loc2);
    assert!(debug2.contains("NonTemporal"));
}
