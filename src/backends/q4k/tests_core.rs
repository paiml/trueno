//! Core Q4K tests: golden parity, scalar vs optimized, NaN/Inf, AVX2 parity,
//! determinism, f16 conversion, dequant, colmajor, parse header, dimensions.

#![allow(dead_code)]

use super::*;

/// Golden Test: Fused kernel must match dequant+matmul within e = 1e-3
/// This is the core falsification test from Section 12.4 of the spec.
#[test]
fn test_fused_q4k_golden_parity() {
    // Create synthetic Q4K data (one super-block = 256 elements)
    let in_dim = 256;
    let out_dim = 4;
    let num_blocks = 1;

    // Build Q4K test data
    let mut q4k_data = Vec::with_capacity(out_dim * num_blocks * SUPER_BLOCK_BYTES);

    for row in 0..out_dim {
        // d = 0.1, dmin = 0.05 (as f16)
        let d: u16 = 0x2E66; // ~0.1 in f16
        let dmin: u16 = 0x2A66; // ~0.05 in f16
        q4k_data.extend_from_slice(&d.to_le_bytes());
        q4k_data.extend_from_slice(&dmin.to_le_bytes());

        // Scales and mins (all set to 1 for simplicity)
        let scales_packed = [0x01u8; 12];
        q4k_data.extend_from_slice(&scales_packed);

        // Quantized values: pattern based on row
        let mut qs = [0u8; 128];
        for (i, q) in qs.iter_mut().enumerate() {
            // Low nibble: (row + i) % 16, High nibble: (row + i + 1) % 16
            let low = ((row + i) % 16) as u8;
            let high = ((row + i + 1) % 16) as u8;
            *q = low | (high << 4);
        }
        q4k_data.extend_from_slice(&qs);
    }

    // Create input vector
    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();

    // Compute using fused kernel
    let fused_output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    // Compute reference: dequant then matmul
    let mut reference_output = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let row_start = row * SUPER_BLOCK_BYTES;
        let row_q4k = &q4k_data[row_start..row_start + SUPER_BLOCK_BYTES];
        let f32_weights = dequantize_q4k_to_f32(row_q4k, in_dim);

        let mut sum = 0.0f32;
        for (w, x) in f32_weights.iter().zip(input.iter()) {
            sum += w * x;
        }
        reference_output[row] = sum;
    }

    // Golden parity check: |fused - reference| < 1e-3
    for (i, (fused, reference)) in fused_output.iter().zip(reference_output.iter()).enumerate()
    {
        let diff = (fused - reference).abs();
        assert!(
            diff < 1e-3,
            "Row {}: Fused kernel divergence: {} vs {} (d={})",
            i,
            fused,
            reference,
            diff
        );
    }
}

/// Test scalar implementation matches optimized version
#[test]
fn test_scalar_vs_optimized_parity() {
    let in_dim = 256;
    let out_dim = 2;

    // Build simple Q4K test data
    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
        q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
        q4k_data.extend_from_slice(&[0x01u8; 12]); // scales
        q4k_data.extend_from_slice(&[0x55u8; 128]); // qs = 5 | (5 << 4)
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001).collect();

    let scalar_output = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
    let optimized_output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    for (i, (s, o)) in scalar_output
        .iter()
        .zip(optimized_output.iter())
        .enumerate()
    {
        let diff = (s - o).abs();
        // Allow small FP differences from mul_add vs separate multiply-add
        assert!(
            diff < 1e-4,
            "Row {}: Scalar vs optimized divergence: {} vs {} (d={})",
            i,
            s,
            o,
            diff
        );
    }
}

/// Test that output contains no NaN or Inf
#[test]
fn test_no_nan_inf() {
    let in_dim = 256;
    let out_dim = 4;

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
        q4k_data.extend_from_slice(&[0x00, 0x38]); // dmin ~ 0.5
        q4k_data.extend_from_slice(&[0x3Fu8; 12]); // max scales
        q4k_data.extend_from_slice(&[0xFFu8; 128]); // max qs
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
    let output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    for (i, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "Row {}: Output is not finite: {}", i, val);
    }
}

/// Test AVX2 implementation matches scalar within tolerance
#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_vs_scalar_parity() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 test - CPU doesn't support AVX2+FMA");
        return;
    }

    let in_dim = 512; // 2 super-blocks
    let out_dim = 4;

    // Build Q4K test data with varied values
    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        // d ~ 0.1, dmin ~ 0.05
        q4k_data.extend_from_slice(&[0x66, 0x2E]); // d
        q4k_data.extend_from_slice(&[0x66, 0x2A]); // dmin
                                                   // Varied scales
        let scale_val = (row as u8 + 1) | ((row as u8 + 2) << 4);
        q4k_data.extend_from_slice(&[scale_val; 12]);
        // Varied quantized values
        for i in 0..128 {
            let low = ((row + i) % 16) as u8;
            let high = ((row + i + 3) % 16) as u8;
            q4k_data.push(low | (high << 4));
        }
    }
    // Duplicate for second super-block
    let single_row_bytes = q4k_data.len() / out_dim;
    let mut full_data = Vec::with_capacity(out_dim * single_row_bytes * 2);
    for row in 0..out_dim {
        let row_start = row * single_row_bytes;
        full_data.extend_from_slice(&q4k_data[row_start..row_start + single_row_bytes]);
        full_data.extend_from_slice(&q4k_data[row_start..row_start + single_row_bytes]);
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.002 - 0.5).collect();

    let scalar_output = matmul_q4k_f32(&full_data, &input, out_dim, in_dim);
    let dispatch_output = matmul_q4k_f32_dispatch(&full_data, &input, out_dim, in_dim);

    for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate()
    {
        let diff = (scalar - dispatch).abs();
        let rel_diff = if scalar.abs() > 1e-6 {
            diff / scalar.abs()
        } else {
            diff
        };
        // Allow 1e-5 relative error for FMA differences
        assert!(
            rel_diff < 1e-5 || diff < 1e-5,
            "Row {}: AVX2 vs scalar divergence: {} vs {} (d={}, rel={})",
            i,
            dispatch,
            scalar,
            diff,
            rel_diff
        );
    }
}

/// Test determinism: same input produces same output
#[test]
fn test_determinism() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        q4k_data.extend_from_slice(&[0x66, 0x2E]); // d
        q4k_data.extend_from_slice(&[0x66, 0x2A]); // dmin
        q4k_data.extend_from_slice(&[0x15u8; 12]);
        q4k_data.extend_from_slice(&[0xABu8; 128]);
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.005).collect();

    let output1 = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);
    let output2 = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    for (i, (a, b)) in output1.iter().zip(output2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "Row {}: Non-deterministic output: {} vs {}",
            i,
            a,
            b
        );
    }
}

/// Test f16 conversion correctness
#[test]
fn test_f16_to_f32() {
    // Test normal values
    assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-3); // 1.0
    assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-3); // 2.0
    assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-3); // 0.5

    // Test zero
    assert_eq!(f16_to_f32(0x0000), 0.0);
    assert_eq!(f16_to_f32(0x8000), -0.0);

    // Test subnormals (small values)
    let small = f16_to_f32(0x0001);
    assert!(small > 0.0 && small < 1e-4);
}

#[test]
fn test_f16_to_f32_infinity_nan() {
    // Positive infinity = 0x7C00
    let inf = f16_to_f32(0x7C00);
    assert!(inf.is_infinite() && inf.is_sign_positive());

    // Negative infinity = 0xFC00
    let neg_inf = f16_to_f32(0xFC00);
    assert!(neg_inf.is_infinite() && neg_inf.is_sign_negative());

    // Negative value
    let neg_one = f16_to_f32(0xBC00); // -1.0
    assert!((neg_one + 1.0).abs() < 1e-3);
}

#[test]
fn test_dequantize_q4k_to_f32_basic() {
    // Create a single Q4K block (144 bytes for 256 elements)
    let mut block = vec![0u8; SUPER_BLOCK_BYTES];
    // d = 1.0 (0x3C00)
    block[0] = 0x00;
    block[1] = 0x3C;
    // dmin = 0 (0x0000)
    block[2] = 0x00;
    block[3] = 0x00;
    // scales = all zeros
    block[4..16].fill(0x00);
    // qs = 0x55 (5 | 5<<4) for all values
    block[16..144].fill(0x55);

    let result = dequantize_q4k_to_f32(&block, 256);
    assert_eq!(result.len(), 256);

    // All values should be finite
    for val in &result {
        assert!(val.is_finite());
    }
}

#[test]
fn test_dequantize_q4k_to_f32_varies_scales() {
    let mut block = vec![0u8; SUPER_BLOCK_BYTES];
    block[0] = 0x00;
    block[1] = 0x3C; // d = 1.0
    block[2] = 0x00;
    block[3] = 0x00; // dmin = 0

    // Set different scales for each group
    for i in 0..12 {
        block[4 + i] = (i * 10) as u8;
    }

    // Set quantized values
    block[16..144].fill(0x33); // 3 | 3<<4

    let result = dequantize_q4k_to_f32(&block, 256);
    assert_eq!(result.len(), 256);
    for val in &result {
        assert!(val.is_finite());
    }
}

#[test]
fn test_matmul_q4k_f32_colmajor_basic() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
        q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
        q4k_data.extend_from_slice(&[0x01u8; 12]); // scales
        q4k_data.extend_from_slice(&[0x55u8; 128]); // qs
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
    let output = matmul_q4k_f32_colmajor(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_matmul_q4k_f32_colmajor_dispatch_basic() {
    let in_dim = 256;
    let out_dim = 4;

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        q4k_data.extend_from_slice(&[0x66, 0x2E]); // d ~ 0.1
        q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
        q4k_data.extend_from_slice(&[(row as u8 + 1); 12]); // varying scales
        q4k_data.extend_from_slice(&[(row as u8 * 17).wrapping_add(0x44); 128]);
        // varying qs
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01 - 1.0).collect();
    let output = matmul_q4k_f32_colmajor_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_matmul_q4k_colmajor_produces_finite() {
    // Column-major layout test: verify it produces valid finite outputs
    // Note: colmajor and rowmajor have different data layout assumptions
    let in_dim = 256;
    let out_dim = 2;

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
        q4k_data.extend_from_slice(&[0x00, 0x38]); // dmin ~ 0.5
        q4k_data.extend_from_slice(&[0x01u8; 12]);
        q4k_data.extend_from_slice(&[0x55u8; 128]);
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.005).collect();

    let rowmajor = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);
    let colmajor = matmul_q4k_f32_colmajor(&q4k_data, &input, out_dim, in_dim);

    // Both should produce finite results
    for (i, r) in rowmajor.iter().enumerate() {
        assert!(r.is_finite(), "Row {}: rowmajor non-finite", i);
    }
    for (i, c) in colmajor.iter().enumerate() {
        assert!(c.is_finite(), "Row {}: colmajor non-finite", i);
    }
}

#[test]
fn test_matmul_q4k_unaligned_dimensions() {
    // Test with dimensions not aligned to block size (256)
    let in_dim = 300;
    let out_dim = 3;
    let num_blocks = (in_dim + 255) / 256; // = 2 blocks

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        for _ in 0..num_blocks {
            q4k_data.extend_from_slice(&[0x00, 0x3C]); // d
            q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin
            q4k_data.extend_from_slice(&[0x01u8; 12]); // scales
            q4k_data.extend_from_slice(&[0x33u8; 128]); // qs
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
    let output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_matmul_q4k_zero_input() {
    let in_dim = 256;
    let out_dim = 2;

    let mut q4k_data = Vec::new();
    for _ in 0..out_dim {
        q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
        q4k_data.extend_from_slice(&[0x00, 0x38]); // dmin ~ 0.5
        q4k_data.extend_from_slice(&[0x7Fu8; 12]); // max scales
        q4k_data.extend_from_slice(&[0xFFu8; 128]); // max qs
    }

    let input: Vec<f32> = vec![0.0; in_dim];
    let output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert_eq!(*val, 0.0, "Output should be zero when input is zero");
    }
}

#[test]
fn test_matmul_q4k_large_dimensions() {
    let in_dim = 1024;
    let out_dim = 8;
    let num_blocks = in_dim / 256;

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for blk in 0..num_blocks {
            let val = ((row * num_blocks + blk) as u8).wrapping_mul(17);
            q4k_data.extend_from_slice(&[0x66, 0x2E]); // d ~ 0.1
            q4k_data.extend_from_slice(&[0x33, 0x2A]); // dmin ~ 0.05
            q4k_data.extend_from_slice(&[(val.wrapping_add(1)); 12]);
            q4k_data.extend_from_slice(&[val.wrapping_add(0x55); 128]);
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| ((i % 100) as f32) * 0.01).collect();
    let output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for val in &output {
        assert!(val.is_finite());
    }
}

#[test]
fn test_parse_q4k_header() {
    let mut block = vec![0u8; 144];
    // d = 1.0 (0x3C00), dmin = 0.5 (0x3800)
    block[0] = 0x00;
    block[1] = 0x3C;
    block[2] = 0x00;
    block[3] = 0x38;
    // scales_bytes[0..12] for llama.cpp format
    // bytes 0-3: lower 6 bits = scales[0-3], upper 2 bits = scales[4-7] upper bits
    // bytes 4-7: lower 6 bits = mins[0-3], upper 2 bits = mins[4-7] upper bits
    // bytes 8-11: lower 4 bits = scales[4-7] lower bits, upper 4 bits = mins[4-7] lower bits
    block[4..8].copy_from_slice(&[0x01, 0x02, 0x03, 0x04]); // scales[0-3] = 1,2,3,4
    block[8..12].copy_from_slice(&[0x0A, 0x0B, 0x0C, 0x0D]); // mins[0-3] = 10,11,12,13
    block[12..16].copy_from_slice(&[0x55, 0x66, 0x77, 0x88]); // combined lower nibbles

    let (d, dmin, scales, mins) = parse_q4k_header(&block);

    assert!((d - 1.0).abs() < 0.01, "d should be ~1.0, got {}", d);
    assert!(
        (dmin - 0.5).abs() < 0.01,
        "dmin should be ~0.5, got {}",
        dmin
    );
    // Check first scales/mins have expected low 6-bit values
    assert_eq!(scales[0], 0x01, "scales[0] should be 1");
    assert_eq!(scales[1], 0x02, "scales[1] should be 2");
    assert_eq!(mins[0], 0x0A, "mins[0] should be 10");
    assert_eq!(mins[1], 0x0B, "mins[1] should be 11");
}

#[test]
fn test_matmul_q4k_single_row() {
    let in_dim = 256;
    let out_dim = 1;

    let mut q4k_data = Vec::new();
    q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
    q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
    q4k_data.extend_from_slice(&[0x01u8; 12]); // scales
    q4k_data.extend_from_slice(&[0xAAu8; 128]); // qs

    let input: Vec<f32> = vec![1.0; in_dim];
    let output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), 1);
    assert!(output[0].is_finite());
}

/// Test AVX2 matmul with large dimensions (exercises full SIMD paths)
#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_large_matrix_mul() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 large matrix test - CPU doesn't support AVX2+FMA");
        return;
    }

    let in_dim = 4096; // 16 super-blocks
    let out_dim = 32;

    // Build Q4K test data with realistic values
    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for _sb in 0..(in_dim / 256) {
            // d ~ 0.1, dmin ~ 0.05
            q4k_data.extend_from_slice(&[0x66, 0x2E]); // d
            q4k_data.extend_from_slice(&[0x66, 0x2A]); // dmin
                                                       // Varied scales based on row
            let scale_val = (row as u8 % 16) | (((row + 1) as u8 % 16) << 4);
            q4k_data.extend_from_slice(&[scale_val; 12]);
            // Varied quantized values
            for i in 0..128 {
                let low = ((row + i) % 16) as u8;
                let high = ((row + i + 3) % 16) as u8;
                q4k_data.push(low | (high << 4));
            }
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001 - 2.0).collect();

    let scalar_output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);
    let dispatch_output = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate()
    {
        let diff = (scalar - dispatch).abs();
        let rel_diff = if scalar.abs() > 1e-6 {
            diff / scalar.abs()
        } else {
            diff
        };
        assert!(
            rel_diff < 1e-4 || diff < 1e-4,
            "Row {}: AVX2 vs scalar divergence: {} vs {} (d={}, rel={})",
            i,
            dispatch,
            scalar,
            diff,
            rel_diff
        );
    }
}

/// Test colmajor AVX2 path with realistic dimensions
#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_colmajor_large() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 colmajor test - CPU doesn't support AVX2+FMA");
        return;
    }

    let in_dim = 2048; // 8 super-blocks
    let out_dim = 16;

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for sb in 0..(in_dim / 256) {
            q4k_data.extend_from_slice(&[0x66, 0x2E]); // d
            q4k_data.extend_from_slice(&[0x33, 0x2A]); // dmin
            let scale_val = ((row + sb) as u8 % 16) | (((row + sb + 1) as u8 % 16) << 4);
            q4k_data.extend_from_slice(&[scale_val; 12]);
            for i in 0..128 {
                q4k_data.push(((i % 16) | (((i + 1) % 16) << 4)) as u8);
            }
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.002 - 1.0).collect();

    let output = matmul_q4k_f32_colmajor(&q4k_data, &input, out_dim, in_dim);
    let output_dispatch = matmul_q4k_f32_colmajor_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    assert_eq!(output_dispatch.len(), out_dim);

    for (i, (base, dispatched)) in output.iter().zip(output_dispatch.iter()).enumerate() {
        let diff = (base - dispatched).abs();
        assert!(
            diff < 1e-3 || (diff / base.abs()) < 1e-4,
            "Row {}: colmajor mismatch: {} vs {} (diff={})",
            i,
            base,
            dispatched,
            diff
        );
    }
}

/// Test non-aligned dimensions (exercises scalar remainder handling)
#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_non_aligned_dimensions() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping AVX2 non-aligned test - CPU doesn't support AVX2+FMA");
        return;
    }

    // Non-aligned: 768 = 3 super-blocks (not power of 2)
    let in_dim = 768;
    let out_dim = 7; // Odd number

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for _sb in 0..(in_dim / 256) {
            q4k_data.extend_from_slice(&[0x66, 0x2E]);
            q4k_data.extend_from_slice(&[0x66, 0x2A]);
            let scale_val = (row as u8 % 16) | (((row + 1) as u8 % 16) << 4);
            q4k_data.extend_from_slice(&[scale_val; 12]);
            for i in 0..128 {
                q4k_data.push(((i % 16) | (((i + 5) % 16) << 4)) as u8);
            }
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.003).sin()).collect();

    let scalar_output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);
    let dispatch_output = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(scalar_output.len(), out_dim);
    assert_eq!(dispatch_output.len(), out_dim);

    for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate()
    {
        let diff = (scalar - dispatch).abs();
        let rel_diff = if scalar.abs() > 1e-6 {
            diff / scalar.abs()
        } else {
            diff
        };
        // FMA operations can have ordering differences, allow 1e-5 relative error
        assert!(
            rel_diff < 1e-5 || diff < 1e-2,
            "Row {}: non-aligned AVX2 mismatch: {} vs {} (diff={}, rel={})",
            i,
            scalar,
            dispatch,
            diff,
            rel_diff
        );
    }
}

/// Test parallel SIMD execution (exercises compute_chunk_q4k_avx2)
#[cfg(all(target_arch = "x86_64", feature = "parallel"))]
#[test]
fn test_parallel_avx2_large_batch() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("Skipping parallel AVX2 test - CPU doesn't support AVX2+FMA");
        return;
    }

    // Large enough to trigger parallel path (>1000 rows)
    let in_dim = 1024;
    let out_dim = 2048; // Large output dim for parallel execution

    let mut q4k_data = Vec::new();
    for row in 0..out_dim {
        for _sb in 0..(in_dim / 256) {
            q4k_data.extend_from_slice(&[0x66, 0x2E]);
            q4k_data.extend_from_slice(&[0x33, 0x2A]);
            let scale_val = ((row % 256) as u8) | (((row / 256) % 16) as u8 * 16);
            q4k_data.extend_from_slice(&[scale_val; 12]);
            for i in 0..128 {
                q4k_data.push(((i * row) % 256) as u8);
            }
        }
    }

    let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.001).collect();

    let output = matmul_q4k_f32_colmajor_dispatch(&q4k_data, &input, out_dim, in_dim);

    assert_eq!(output.len(), out_dim);
    for (i, val) in output.iter().enumerate() {
        assert!(
            val.is_finite(),
            "Row {}: parallel AVX2 produced non-finite: {}",
            i,
            val
        );
    }
}
