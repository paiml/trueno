use super::super::*;

// ========================================================================
// TILE-003: Q4K MatVec Tests
// ========================================================================

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

// ========================================================================
// f16 conversion tests
// ========================================================================

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
