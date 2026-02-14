//! Extended Q6K tests: parallel dispatch, compute_chunk_scalar, colmajor edge cases, coverage

use super::gemv::compute_chunk_scalar;
use super::*;

#[test]
fn test_parallel_dispatch_large_matrix() {
    // Test parallel path: total_work >= 8_000_000
    // Use 4096 x 2048 = 8_388_608 ops (triggers parallel)
    let out_dim = 4096;
    let in_dim = 2048; // Must be multiple of 256 (SUPER_BLOCK_SIZE)
    let total_work = out_dim * in_dim;
    assert!(total_work >= 8_000_000, "Test must trigger parallel path");

    let num_superblocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_superblocks_per_row * SUPER_BLOCK_BYTES;
    let total_bytes = out_dim * row_bytes;

    // Create deterministic test data
    let mut q6k_data = vec![0u8; total_bytes];
    for row in 0..out_dim {
        for sb in 0..num_superblocks_per_row {
            let offset = row * row_bytes + sb * SUPER_BLOCK_BYTES;
            // d = 1.0 as f16
            q6k_data[offset] = 0x00;
            q6k_data[offset + 1] = 0x3C;
            // ql: 6-bit low parts
            for i in 0..128 {
                q6k_data[offset + 2 + i] = ((row + sb + i) % 64) as u8;
            }
            // qh: 2-bit high parts
            for i in 0..64 {
                q6k_data[offset + 130 + i] = ((row ^ sb ^ i) % 4) as u8;
            }
            // scales
            for i in 0..16 {
                q6k_data[offset + 194 + i] = 0x10;
            }
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i % 10) as f32 * 0.1).collect();

    // Call dispatch - should use parallel path
    let result = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

    // Verify dimensions and finiteness
    assert_eq!(result.len(), out_dim);
    for (i, &val) in result.iter().enumerate() {
        assert!(val.is_finite(), "Result[{}] is not finite: {}", i, val);
    }

    // Compare a few rows against scalar for consistency
    let scalar_result = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
    for i in (0..out_dim).step_by(512) {
        let diff = (result[i] - scalar_result[i]).abs();
        let tol = scalar_result[i].abs() * 0.01 + 1e-4;
        assert!(
            diff < tol,
            "Parallel vs scalar mismatch at row {}: parallel={}, scalar={}, diff={}",
            i,
            result[i],
            scalar_result[i],
            diff
        );
    }
}

#[test]
fn test_parallel_colmajor_large_matrix() {
    // Test colmajor path
    // ne0 = output dimension, ne1 = input dimension
    let ne0 = 2048; // output dimension, must be multiple of 256
    let ne1 = 4096; // input dimension

    let blocks_per_col = (ne0 + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let col_bytes = blocks_per_col * SUPER_BLOCK_BYTES;
    let total_bytes = ne1 * col_bytes;

    let mut q6k_data = vec![0u8; total_bytes];
    for col in 0..ne1 {
        for sb in 0..blocks_per_col {
            let offset = col * col_bytes + sb * SUPER_BLOCK_BYTES;
            // d = 0.5 as f16
            q6k_data[offset] = 0x00;
            q6k_data[offset + 1] = 0x38;
            // ql
            for i in 0..128 {
                q6k_data[offset + 2 + i] = ((col ^ sb ^ i) % 64) as u8;
            }
            // qh
            for i in 0..64 {
                q6k_data[offset + 130 + i] = ((col + sb) % 4) as u8;
            }
            // scales
            for i in 0..16 {
                q6k_data[offset + 194 + i] = 0x20;
            }
        }
    }

    // Input must have length ne1
    let input: Vec<f32> = (0..ne1).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

    // Use colmajor dispatch
    let result = matmul_q6k_f32_colmajor_dispatch(&q6k_data, &input, ne0, ne1);

    // Output has ne0 elements
    assert_eq!(result.len(), ne0);
    for (i, &val) in result.iter().enumerate() {
        assert!(val.is_finite(), "Result[{}] is not finite: {}", i, val);
    }
}

#[test]
fn test_compute_chunk_scalar_small() {
    // Directly test compute_chunk_scalar
    let in_dim = 256;
    let out_dim = 4;
    let num_blocks_per_row = 1;
    let row_bytes = SUPER_BLOCK_BYTES;

    let mut q6k_data = vec![0u8; out_dim * row_bytes];
    for row in 0..out_dim {
        let offset = row * row_bytes;
        // d = 1.0 as f16
        q6k_data[offset] = 0x00;
        q6k_data[offset + 1] = 0x3C;
        // ql = all zeros
        for i in 0..128 {
            q6k_data[offset + 2 + i] = 0x00;
        }
        // qh = all zeros
        for i in 0..64 {
            q6k_data[offset + 130 + i] = 0x00;
        }
        // scales = 1
        for i in 0..16 {
            q6k_data[offset + 194 + i] = 0x01;
        }
    }

    let input = vec![1.0f32; in_dim];
    let mut chunk = vec![0.0f32; out_dim];

    compute_chunk_scalar(
        &q6k_data,
        &input,
        &mut chunk,
        0,
        out_dim,
        in_dim,
        num_blocks_per_row,
        row_bytes,
    );

    // Verify results are finite
    for (i, &val) in chunk.iter().enumerate() {
        assert!(val.is_finite(), "Chunk[{}] is not finite: {}", i, val);
    }
}

// =========================================================================
// Additional Q6K scalar coverage tests
// =========================================================================

/// Test compute_chunk_scalar with start_row offset (exercises the
/// `out_idx = start_row + local_idx` path and skips first rows)
#[test]
fn test_compute_chunk_scalar_with_offset() {
    let in_dim = 256;
    let out_dim = 4;
    let num_blocks_per_row = 1;
    let row_bytes = SUPER_BLOCK_BYTES;

    // Build Q6K data: ql[128], qh[64], scales[16], d[2] = 210 bytes per row
    let mut q6k_data = Vec::new();
    for row in 0..out_dim {
        // ql: varying values
        for i in 0..128 {
            q6k_data.push(((row * 17 + i) % 256) as u8);
        }
        // qh: varying
        for i in 0..64 {
            q6k_data.push(((row * 7 + i) % 256) as u8);
        }
        // scales
        q6k_data.extend_from_slice(&[0x02u8; 16]);
        // d = 1.0 as f16
        q6k_data.extend_from_slice(&[0x00, 0x3C]);
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();

    // Process only the last 2 rows (start_row=2)
    let mut chunk = vec![0.0f32; 2];
    compute_chunk_scalar(
        &q6k_data,
        &input,
        &mut chunk,
        2, // start_row offset
        out_dim,
        in_dim,
        num_blocks_per_row,
        row_bytes,
    );

    for &val in &chunk {
        assert!(val.is_finite());
    }
}

/// Test compute_chunk_scalar where out_idx exceeds out_dim (early break)
#[test]
fn test_compute_chunk_scalar_exceeds_outdim() {
    let in_dim = 256;
    let out_dim = 2;
    let num_blocks_per_row = 1;
    let row_bytes = SUPER_BLOCK_BYTES;

    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        q6k_data.extend_from_slice(&[0x33u8; 128]); // ql
        q6k_data.extend_from_slice(&[0x11u8; 64]); // qh
        q6k_data.extend_from_slice(&[0x01u8; 16]); // scales
        q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
    }

    let input = vec![1.0f32; in_dim];

    // Chunk has 4 slots but out_dim=2, so only 2 should be written
    let mut chunk = vec![0.0f32; 4];
    compute_chunk_scalar(
        &q6k_data,
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
fn test_matmul_q6k_scalar_multiple_blocks() {
    let in_dim = 512; // 2 super-blocks
    let out_dim = 2;
    let num_blocks = 2;

    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        for _ in 0..num_blocks {
            q6k_data.extend_from_slice(&[0x44u8; 128]); // ql
            q6k_data.extend_from_slice(&[0x00u8; 64]); // qh
            q6k_data.extend_from_slice(&[0x02u8; 16]); // scales = 2
            q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001).collect();
    let output_scalar = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
    let output_dispatch = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(output_scalar.len(), out_dim);
    assert_eq!(output_dispatch.len(), out_dim);

    for (i, (s, d)) in output_scalar.iter().zip(output_dispatch.iter()).enumerate() {
        let diff = (s - d).abs();
        assert!(
            diff < 1e-3,
            "Row {}: scalar={} vs dispatch={}, diff={}",
            i,
            s,
            d,
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
    // Negative NaN
    let neg_nan = f16_to_f32(0xFC01);
    assert!(neg_nan.is_nan(), "0xFC01 should be NaN");
}

/// Test f16 conversion: negative normal value
#[test]
fn test_f16_to_f32_negative_normal() {
    // -2.0 in f16 = 0xC000
    let val = f16_to_f32(0xC000);
    assert!((val - (-2.0)).abs() < 1e-3, "Expected -2.0, got {}", val);
    // -0.5 in f16 = 0xB800
    let val = f16_to_f32(0xB800);
    assert!((val - (-0.5)).abs() < 1e-3, "Expected -0.5, got {}", val);
}

/// Test f16 conversion: smallest positive normal
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

/// Test f16 conversion: largest finite f16
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

/// Test f16 conversion: half value (0.5)
#[test]
fn test_f16_to_f32_half() {
    // 0.5 in f16 = 0x3800
    let val = f16_to_f32(0x3800);
    assert!((val - 0.5).abs() < 1e-6, "Expected 0.5, got {}", val);
}

/// Test f16 conversion: largest subnormal
#[test]
fn test_f16_to_f32_largest_subnormal() {
    // 0x03FF = largest subnormal, should be close to smallest normal
    let largest_subnormal = f16_to_f32(0x03FF);
    let smallest_normal = f16_to_f32(0x0400);
    assert!(largest_subnormal > 0.0);
    assert!(
        largest_subnormal < smallest_normal,
        "Largest subnormal {} should be less than smallest normal {}",
        largest_subnormal,
        smallest_normal
    );
}

/// Test colmajor sparse input optimization path (x_j == 0.0 skip)
#[test]
fn test_colmajor_sparse_input() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        q6k_data.extend_from_slice(&[0x33u8; 128]); // ql
        q6k_data.extend_from_slice(&[0x00u8; 64]); // qh
        q6k_data.extend_from_slice(&[0x01u8; 16]); // scales
        q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
    }

    // All zeros except first element -- test x_j == 0.0 skip path
    let mut input = vec![0.0f32; in_dim];
    input[0] = 1.0;

    let output = matmul_q6k_f32_colmajor(&q6k_data, &input, out_dim, in_dim);
    assert_eq!(output.len(), out_dim);
    // Should be non-zero since input[0] = 1.0
    assert!(output[0].is_finite());
}

/// Test colmajor with fully sparse input (all zeros)
#[test]
fn test_colmajor_all_zero_input() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        q6k_data.extend_from_slice(&[0xFFu8; 128]); // ql
        q6k_data.extend_from_slice(&[0xFFu8; 64]); // qh
        q6k_data.extend_from_slice(&[0x7Fu8; 16]); // scales
        q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
    }

    let input = vec![0.0f32; in_dim];
    let output = matmul_q6k_f32_colmajor(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert_eq!(*val, 0.0, "All-zero input should produce zero output");
    }
}

/// Test Q6K matmul consistency: scalar, dispatch, and colmajor
/// (for small enough dimensions that colmajor won't diverge from row-major
///  due to layout differences)
#[test]
fn test_q6k_scalar_vs_dispatch_small() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q6k_data = Vec::new();
    for row in 0..out_dim {
        for i in 0..128 {
            q6k_data.push(((row * 13 + i) % 256) as u8);
        }
        for i in 0..64 {
            q6k_data.push(((row * 7 + i) % 256) as u8);
        }
        q6k_data.extend_from_slice(&[0x03u8; 16]);
        q6k_data.extend_from_slice(&[0x66, 0x2E]); // d ~ 0.1
    }

    let input: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.013).sin()).collect();

    let scalar = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
    let dispatch = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

    for (i, (s, d)) in scalar.iter().zip(dispatch.iter()).enumerate() {
        let diff = (s - d).abs();
        assert!(
            diff < 1e-3,
            "Row {}: scalar={} vs dispatch={}, diff={}",
            i,
            s,
            d,
            diff
        );
    }
}

/// Test matmul_q6k_f32 (the public alias routes through dispatch)
#[test]
fn test_matmul_q6k_f32_public_api() {
    let in_dim = 256;
    let out_dim = 3;

    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        q6k_data.extend_from_slice(&[0x22u8; 128]); // ql
        q6k_data.extend_from_slice(&[0x11u8; 64]); // qh
        q6k_data.extend_from_slice(&[0x04u8; 16]); // scales
        q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
    }

    let input = vec![0.5f32; in_dim];

    // matmul_q6k_f32 is public alias for dispatch
    let result = matmul_q6k_f32(&q6k_data, &input, out_dim, in_dim);
    assert_eq!(result.len(), out_dim);
    for val in &result {
        assert!(val.is_finite());
    }
}

/// Test with minimum dimension: single element (in_dim < SUPER_BLOCK_SIZE)
#[test]
fn test_q6k_small_input_dim() {
    let in_dim = 16; // Much smaller than one super-block (256)
    let out_dim = 1;

    // Still need one full super-block worth of data per row
    let mut q6k_data = Vec::new();
    q6k_data.extend_from_slice(&[0x55u8; 128]); // ql
    q6k_data.extend_from_slice(&[0x00u8; 64]); // qh
    q6k_data.extend_from_slice(&[0x01u8; 16]); // scales
    q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0

    let input = vec![1.0f32; in_dim];
    let output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), 1);
    assert!(output[0].is_finite());
}

/// Test colmajor with unaligned output dimension (ne0 not multiple of 256)
#[test]
fn test_colmajor_unaligned_output_dim() {
    let ne0 = 300; // Not multiple of 256
    let ne1 = 2; // 2 input columns

    let blocks_per_col = (ne0 + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE; // = 2
    let col_bytes = blocks_per_col * SUPER_BLOCK_BYTES;

    let mut q6k_data = Vec::new();
    for _ in 0..ne1 {
        for _ in 0..blocks_per_col {
            q6k_data.extend_from_slice(&[0x22u8; 128]); // ql
            q6k_data.extend_from_slice(&[0x00u8; 64]); // qh
            q6k_data.extend_from_slice(&[0x01u8; 16]); // scales
            q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
        }
    }

    assert_eq!(q6k_data.len(), ne1 * col_bytes);

    let input = vec![1.0f32; ne1];
    let output = matmul_q6k_f32_colmajor(&q6k_data, &input, ne0, ne1);

    assert_eq!(output.len(), ne0);
    for val in &output {
        assert!(val.is_finite());
    }
}

/// Test compute_chunk_scalar where data is truncated (sb_start + SUPER_BLOCK_BYTES > data.len())
#[test]
fn test_compute_chunk_scalar_truncated_data() {
    let in_dim = 512; // Needs 2 blocks per row
    let out_dim = 1;
    let num_blocks_per_row = 2;
    let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;

    // Provide only 1.5 blocks worth of data (truncated second block)
    let mut q6k_data = Vec::new();
    // Full first block
    q6k_data.extend_from_slice(&[0x11u8; 128]); // ql
    q6k_data.extend_from_slice(&[0x00u8; 64]); // qh
    q6k_data.extend_from_slice(&[0x01u8; 16]); // scales
    q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
                                               // Partial second block (100 bytes instead of 210)
    q6k_data.extend_from_slice(&[0x00u8; 100]);

    let input = vec![1.0f32; in_dim];
    let mut chunk = vec![0.0f32; 1];

    compute_chunk_scalar(
        &q6k_data,
        &input,
        &mut chunk,
        0,
        out_dim,
        in_dim,
        num_blocks_per_row,
        row_bytes,
    );

    // Should process first block and skip truncated second block
    assert!(chunk[0].is_finite());
}

/// Test Q6K constants are correct
#[test]
fn test_q6k_constants() {
    assert_eq!(SUPER_BLOCK_SIZE, 256);
    assert_eq!(SUPER_BLOCK_BYTES, 210);
    // 210 = 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
    assert_eq!(128 + 64 + 16 + 2, SUPER_BLOCK_BYTES);
}

/// Test that negative input produces finite output with positive weights
#[test]
fn test_q6k_negative_input() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        q6k_data.extend_from_slice(&[0x55u8; 128]); // ql
        q6k_data.extend_from_slice(&[0x00u8; 64]); // qh
        q6k_data.extend_from_slice(&[0x01u8; 16]); // scales = 1
        q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
    }

    let input: Vec<f32> = (0..in_dim).map(|i| -(i as f32) * 0.01).collect();
    let output = matmul_q6k_f32(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

/// Test that scalar output matches across identical rows
#[test]
fn test_q6k_identical_rows_produce_same_output() {
    let in_dim = 256;
    let out_dim = 4;

    // All rows identical
    let mut q6k_data = Vec::new();
    for _ in 0..out_dim {
        q6k_data.extend_from_slice(&[0x42u8; 128]); // ql
        q6k_data.extend_from_slice(&[0x11u8; 64]); // qh
        q6k_data.extend_from_slice(&[0x05u8; 16]); // scales
        q6k_data.extend_from_slice(&[0x66, 0x2E]); // d ~ 0.1
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.005).collect();
    let output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);

    // All rows should produce the same result
    for i in 1..out_dim {
        assert!(
            (output[0] - output[i]).abs() < 1e-6,
            "Row 0 ({}) != Row {} ({})",
            output[0],
            i,
            output[i]
        );
    }
}

/// Test Q6K with d = 0 (global scale zero should zero out results)
#[test]
fn test_q6k_zero_d_scale() {
    let in_dim = 256;
    let out_dim = 1;

    let mut q6k_data = Vec::new();
    q6k_data.extend_from_slice(&[0xFFu8; 128]); // ql = max
    q6k_data.extend_from_slice(&[0xFFu8; 64]); // qh = max
    q6k_data.extend_from_slice(&[0x7Fu8; 16]); // scales = max positive
    q6k_data.extend_from_slice(&[0x00, 0x00]); // d = 0.0

    let input = vec![1.0f32; in_dim];
    let output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);

    assert_eq!(output[0], 0.0, "Zero d-scale should produce zero output");
}
