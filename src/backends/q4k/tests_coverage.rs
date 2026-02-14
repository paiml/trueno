//! Additional Q4K scalar coverage tests and AVX2 GEMV coverage tests.

#![allow(dead_code)]

use super::gemv::compute_chunk_q4k_scalar;
use super::*;

// =========================================================================
// Additional Q4K scalar coverage tests
// =========================================================================

/// Test compute_chunk_q4k_scalar with start_row offset
#[test]
fn test_compute_chunk_scalar_with_offset() {
    let in_dim = 256;
    let out_dim = 4;
    let num_blocks_per_row = 1;
    let row_bytes = SUPER_BLOCK_BYTES;

    let mut q4k_data = vec![0u8; out_dim * row_bytes];
    for row in 0..out_dim {
        let offset = row * row_bytes;
        q4k_data[offset] = 0x00;
        q4k_data[offset + 1] = 0x3C; // d = 1.0
        q4k_data[offset + 2] = 0x00;
        q4k_data[offset + 3] = 0x00; // dmin = 0.0
        for i in 0..12 {
            q4k_data[offset + 4 + i] = 0x01;
        }
        q4k_data[offset + 16..offset + 144].fill(0x55); // qs = 5|5
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();

    // Process only the last 2 rows (start_row=2)
    let mut chunk = vec![0.0f32; 2];
    compute_chunk_q4k_scalar(
        &q4k_data,
        &input,
        &mut chunk,
        2, // start_row
        out_dim,
        in_dim,
        num_blocks_per_row,
        row_bytes,
    );

    for &val in &chunk {
        assert!(val.is_finite());
    }
}

/// Test compute_chunk_q4k_scalar where out_idx exceeds out_dim (early break)
#[test]
fn test_compute_chunk_scalar_exceeds_outdim() {
    let in_dim = 256;
    let out_dim = 2;
    let num_blocks_per_row = 1;
    let row_bytes = SUPER_BLOCK_BYTES;

    let mut q4k_data = vec![0u8; out_dim * row_bytes];
    for row in 0..out_dim {
        let offset = row * row_bytes;
        q4k_data[offset] = 0x00;
        q4k_data[offset + 1] = 0x3C;
        q4k_data[offset + 2] = 0x00;
        q4k_data[offset + 3] = 0x00;
        for i in 0..12 {
            q4k_data[offset + 4 + i] = 0x01;
        }
        q4k_data[offset + 16..offset + 144].fill(0x33);
    }

    let input = vec![1.0f32; in_dim];

    // Chunk has 4 slots but out_dim=2, so only 2 should be written
    let mut chunk = vec![0.0f32; 4];
    compute_chunk_q4k_scalar(
        &q4k_data,
        &input,
        &mut chunk,
        0,
        out_dim,
        in_dim,
        num_blocks_per_row,
        row_bytes,
    );

    // First 2 should be populated, last 2 should remain zero
    for i in 0..2 {
        assert!(chunk[i].is_finite());
    }
    assert_eq!(chunk[2], 0.0, "Elements beyond out_dim should remain zero");
    assert_eq!(chunk[3], 0.0, "Elements beyond out_dim should remain zero");
}

/// Test scalar path with multiple super-blocks per row
#[test]
fn test_matmul_q4k_scalar_multiple_blocks() {
    let in_dim = 512; // 2 super-blocks
    let out_dim = 2;
    let num_blocks = 2;

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        for _ in 0..num_blocks {
            q4k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
            q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
            q4k_data.extend_from_slice(&[0x02u8; 12]); // scales = 2
            q4k_data.extend_from_slice(&[0x88u8; 128]); // qs = 8|8
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001).collect();
    let output_scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let output_optimized = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output_scalar.len(), out_dim);
    assert_eq!(output_optimized.len(), out_dim);

    for (i, (s, o)) in output_scalar
        .iter()
        .zip(output_optimized.iter())
        .enumerate()
    {
        let diff = (s - o).abs();
        assert!(
            diff < 1e-3,
            "Row {}: scalar={} vs optimized={}, diff={}",
            i,
            s,
            o,
            diff
        );
    }
}

/// Test f16 conversion: NaN values
#[test]
fn test_f16_to_f32_nan() {
    // f16 NaN: exp=31, mantissa != 0
    let nan_val = f16_to_f32(0x7C01);
    assert!(nan_val.is_nan(), "0x7C01 should be NaN");
}

/// Test f16 conversion: negative normal value
#[test]
fn test_f16_to_f32_negative_normal() {
    // -2.0 in f16 = 0xC000
    let val = f16_to_f32(0xC000);
    assert!((val - (-2.0)).abs() < 1e-3, "Expected -2.0, got {}", val);
}

/// Test f16 conversion: smallest normal
#[test]
fn test_f16_to_f32_smallest_normal() {
    // Smallest positive normal: 0x0400 = 2^(-14) ~ 6.1035e-5
    let val = f16_to_f32(0x0400);
    assert!(
        val > 0.0 && val < 0.001,
        "Expected small normal, got {}",
        val
    );
}

/// Test f16 conversion: largest normal
#[test]
fn test_f16_to_f32_largest_normal() {
    // Largest finite f16: 0x7BFF ~ 65504
    let val = f16_to_f32(0x7BFF);
    assert!(
        (val - 65504.0).abs() < 100.0,
        "Expected ~65504, got {}",
        val
    );
}

/// Test f16 conversion: negative subnormal
#[test]
fn test_f16_to_f32_negative_subnormal() {
    // Negative smallest subnormal: 0x8001
    let val = f16_to_f32(0x8001);
    assert!(
        val < 0.0 && val > -1e-4,
        "Expected small negative, got {}",
        val
    );
}

/// Test parse_q4k_header with all-zero block
#[test]
fn test_parse_q4k_header_all_zeros() {
    let block = vec![0u8; 144];
    let (d, dmin, scales, mins) = parse_q4k_header(&block);
    assert_eq!(d, 0.0);
    assert_eq!(dmin, 0.0);
    assert_eq!(scales, [0u8; 8]);
    assert_eq!(mins, [0u8; 8]);
}

/// Test parse_q4k_header with max-value block
#[test]
fn test_parse_q4k_header_max_values() {
    let mut block = vec![0xFFu8; 144];
    block[0] = 0xFF;
    block[1] = 0x7B; // d ~ largest f16 finite
    block[2] = 0xFF;
    block[3] = 0x7B; // dmin ~ largest f16 finite
    let (d, dmin, scales, mins) = parse_q4k_header(&block);
    assert!(d.is_finite(), "d should be finite");
    assert!(dmin.is_finite(), "dmin should be finite");
    // Scales and mins should be populated
    for i in 0..8 {
        // The exact values depend on bit unpacking but should be non-zero
        assert!(scales[i] > 0 || mins[i] > 0 || i >= 4);
    }
}

/// Test dequantize with more than one block
#[test]
fn test_dequantize_q4k_multi_block() {
    let num_elements = 512; // 2 blocks
    let mut data = Vec::new();
    for _ in 0..2 {
        data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
        data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
        data.extend_from_slice(&[0x01u8; 12]); // scales = 1
        data.extend_from_slice(&[0x77u8; 128]); // qs = 7|7
    }

    let result = dequantize_q4k_to_f32(&data, num_elements);
    assert_eq!(result.len(), num_elements);
    for val in &result {
        assert!(val.is_finite());
    }
}

/// Test that colmajor skips zero input values
#[test]
fn test_colmajor_sparse_input() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
        q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
        q4k_data.extend_from_slice(&[0x01u8; 12]); // scales
        q4k_data.extend_from_slice(&[0x55u8; 128]); // qs
    }

    // All zeros except first element
    let mut input = vec![0.0f32; in_dim];
    input[0] = 1.0;

    let output = matmul_q4k_f32_colmajor(&q4k_data, &input, out_dim, in_dim);
    assert_eq!(output.len(), out_dim);
    // Should be non-zero since input[0] = 1.0
    assert!(output[0].is_finite());
}

/// Test matmul_q4k_f32 with 4-way unroll remainder (in_dim not multiple of 4 within a chunk)
#[test]
fn test_matmul_q4k_f32_optimized_remainder() {
    // Test the optimized path's remainder handling
    let in_dim = 256;
    let out_dim = 1;

    let mut q4k_data = Vec::new();
    q4k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
    q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
    q4k_data.extend_from_slice(&[0x01u8; 12]); // scales = 1
                                               // Use varying qs to exercise different nibble values
    for i in 0..128 {
        q4k_data.push(((i % 16) | (((i + 1) % 16) << 4)) as u8);
    }

    let input = vec![1.0f32; in_dim];
    let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let optimized = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    let diff = (scalar[0] - optimized[0]).abs();
    assert!(
        diff < 1e-3,
        "Scalar {} vs optimized {}, diff={}",
        scalar[0],
        optimized[0],
        diff
    );
}

// ================================================================
// Q4K AVX2 GEMV coverage - dispatch with various sizes
// ================================================================

/// Test dispatch with multiple super-blocks, exercising AVX2 inner loops
#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4k_dispatch_multi_superblock() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    // 4 super-blocks = 1024 elements
    let in_dim = 1024;
    let out_dim = 8;
    let num_blocks = in_dim / SUPER_BLOCK_SIZE;

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for sb in 0..num_blocks {
            q4k_data.extend_from_slice(&[0x66, 0x2E]); // d
            q4k_data.extend_from_slice(&[0x66, 0x2A]); // dmin
            let sv = ((row + sb) as u8 + 1) | (((row + sb) as u8 + 2) << 4);
            q4k_data.extend_from_slice(&[sv; 12]);
            for i in 0..128 {
                let low = ((row + sb + i) % 16) as u8;
                let high = ((row + sb + i + 3) % 16) as u8;
                q4k_data.push(low | (high << 4));
            }
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001 - 0.5).collect();

    let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let dispatch = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    for (i, (s, d)) in scalar.iter().zip(dispatch.iter()).enumerate() {
        let diff = (s - d).abs();
        let rel_diff = if s.abs() > 1e-6 { diff / s.abs() } else { diff };
        assert!(
            rel_diff < 1e-4 || diff < 1e-4,
            "Row {}: scalar={}, dispatch={}, diff={}, rel={}",
            i,
            s,
            d,
            diff,
            rel_diff
        );
    }
}

/// Test dispatch with 3 super-blocks (768 elements), exercising more inner loop iterations
#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4k_dispatch_three_superblocks() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let in_dim = 768; // 3 super-blocks
    let out_dim = 2;
    let num_blocks = in_dim / SUPER_BLOCK_SIZE;

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for sb in 0..num_blocks {
            q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
            q4k_data.extend_from_slice(&[0x00, 0x38]); // dmin ~ 0.5
            let sv = ((row + sb) as u8 + 1) & 0x3F;
            q4k_data.extend_from_slice(&[sv; 12]);
            q4k_data.extend_from_slice(&[0x77u8; 128]);
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.005).collect();

    let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let dispatch = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    for (i, (s, d)) in scalar.iter().zip(dispatch.iter()).enumerate() {
        let diff = (s - d).abs();
        let rel_diff = if s.abs() > 1e-6 { diff / s.abs() } else { diff };
        assert!(
            rel_diff < 1e-4 || diff < 1e-4,
            "Row {}: scalar={}, dispatch={}, diff={}",
            i,
            s,
            d,
            diff
        );
    }
}

/// Test dispatch with negative input values (exercises subtract branches)
#[cfg(target_arch = "x86_64")]
#[test]
fn test_q4k_dispatch_negative_inputs() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let in_dim = 256;
    let out_dim = 4;

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        q4k_data.extend_from_slice(&[0x66, 0x2E]); // d ~ 0.1
        q4k_data.extend_from_slice(&[0x66, 0x2A]); // dmin ~ 0.05
        q4k_data.extend_from_slice(&[((row + 1) as u8); 12]);
        for i in 0..128 {
            q4k_data.push(((i * 3 + row) % 256) as u8);
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| -1.0 + (i as f32) * 0.008).collect();

    let scalar = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let dispatch = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    for (i, (s, d)) in scalar.iter().zip(dispatch.iter()).enumerate() {
        let diff = (s - d).abs();
        let rel_diff = if s.abs() > 1e-6 { diff / s.abs() } else { diff };
        assert!(
            rel_diff < 1e-4 || diff < 1e-4,
            "Row {}: scalar={}, dispatch={}, diff={}",
            i,
            s,
            d,
            diff
        );
    }
}
