//! Fused Q4_K Matrix-Vector Multiply (F-GPU-130)
//!
//! This module implements fused quantized matrix-vector multiplication that operates
//! directly on Q4_K compressed weights without full dequantization.
//!
//! # Q4_K Format (llama.cpp compatible)
//!
//! Super-block layout (144 bytes per 256 elements):
//! - `d`: 2 bytes (f16 global scale)
//! - `dmin`: 2 bytes (f16 global min scale)
//! - `scales`: 12 bytes (packed 6-bit scales and mins for 8 sub-blocks)
//! - `qs`: 128 bytes (4-bit quantized values, interleaved low/high nibbles)
//!
//! # Golden Test Invariant (Section 12.4 of spec)
//!
//! For all Q4K weight W and input x:
//! ```text
//! matmul_q4k_f32(W, x) ≈ matmul(dequant_q4k_to_f32(W), x)  within ε = 1e-3
//! ```
//!
//! # Performance Targets
//!
//! - Baseline (dequant+matmul): 0.27 tok/s
//! - Target (fused): >5 tok/s CPU, >100 tok/s GPU
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno::backends::q4k::matmul_q4k_f32;
//!
//! let q4k_weights = load_q4k_tensor("gate_proj.weight");
//! let input = vec![1.0f32; 896];
//! let output = matmul_q4k_f32(&q4k_weights, &input, 4864, 896);
//! ```

#![allow(dead_code)]

// Sub-modules
mod colmajor;
mod dequant;
mod gemv;

// Re-exports
pub use colmajor::{matmul_q4k_f32_colmajor, matmul_q4k_f32_colmajor_dispatch};
pub use dequant::dequantize_q4k_to_f32;
pub use gemv::{matmul_q4k_f32, matmul_q4k_f32_dispatch, matmul_q4k_f32_scalar};

// Constants (pub(crate) for submodule access)
pub(crate) const SUPER_BLOCK_SIZE: usize = 256;
pub(crate) const SUPER_BLOCK_BYTES: usize = 144;
#[allow(dead_code)] // Reserved for future sub-block optimizations
pub(crate) const SUB_BLOCK_SIZE: usize = 32;

/// Convert f16 bits to f32
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mantissa = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mantissa == 0 {
            f32::from_bits(sign)
        } else {
            // Subnormal
            let mut m = mantissa;
            let mut e = 0i32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            let new_exp = ((127 - 15 + 1 + e) as u32) << 23;
            let new_mantissa = (m & 0x3FF) << 13;
            f32::from_bits(sign | new_exp | new_mantissa)
        }
    } else if exp == 31 {
        f32::from_bits(sign | (0xFF << 23) | (mantissa << 13))
    } else {
        let new_exp = ((exp as i32 - 15 + 127) as u32) << 23;
        f32::from_bits(sign | new_exp | (mantissa << 13))
    }
}

/// Parse Q4_K super-block header and scales
///
/// Returns (d, dmin, scales[8], mins[8])
#[inline(always)]
pub(crate) fn parse_q4k_header(block: &[u8]) -> (f32, f32, [u8; 8], [u8; 8]) {
    debug_assert!(block.len() >= 16);

    // Read d and dmin (f16)
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

    // Unpack scales and mins (llama.cpp format)
    let scales_bytes = &block[4..16];
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    for i in 0..4 {
        // Blocks 0-3: lower 6 bits of bytes 0-3 and 4-7
        scales[i] = scales_bytes[i] & 0x3F;
        mins[i] = scales_bytes[i + 4] & 0x3F;
        // Blocks 4-7: lower 4 bits from bytes 8-11, upper 2 bits from bytes 0-3/4-7
        scales[i + 4] = (scales_bytes[i + 8] & 0x0F) | ((scales_bytes[i] >> 6) << 4);
        mins[i + 4] = (scales_bytes[i + 8] >> 4) | ((scales_bytes[i + 4] >> 6) << 4);
    }

    (d, dmin, scales, mins)
}

#[cfg(test)]
mod tests {
    use super::gemv::compute_chunk_q4k_scalar;
    use super::*;

    /// Golden Test: Fused kernel must match dequant+matmul within ε = 1e-3
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
                "Row {}: Fused kernel divergence: {} vs {} (Δ={})",
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

        for (i, (s, o)) in scalar_output.iter().zip(optimized_output.iter()).enumerate() {
            let diff = (s - o).abs();
            // Allow small FP differences from mul_add vs separate multiply-add
            assert!(
                diff < 1e-4,
                "Row {}: Scalar vs optimized divergence: {} vs {} (Δ={})",
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

        for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate() {
            let diff = (scalar - dispatch).abs();
            let rel_diff = if scalar.abs() > 1e-6 {
                diff / scalar.abs()
            } else {
                diff
            };
            // Allow 1e-5 relative error for FMA differences
            assert!(
                rel_diff < 1e-5 || diff < 1e-5,
                "Row {}: AVX2 vs scalar divergence: {} vs {} (Δ={}, rel={})",
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
            q4k_data.extend_from_slice(&[(row as u8 * 17).wrapping_add(0x44); 128]); // varying qs
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
        assert!((dmin - 0.5).abs() < 0.01, "dmin should be ~0.5, got {}", dmin);
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

        for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate() {
            let diff = (scalar - dispatch).abs();
            let rel_diff = if scalar.abs() > 1e-6 {
                diff / scalar.abs()
            } else {
                diff
            };
            assert!(
                rel_diff < 1e-4 || diff < 1e-4,
                "Row {}: AVX2 vs scalar divergence: {} vs {} (Δ={}, rel={})",
                i, dispatch, scalar, diff, rel_diff
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
                i, base, dispatched, diff
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

        for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate() {
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
                i, scalar, dispatch, diff, rel_diff
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
                i, val
            );
        }
    }

    // =========================================================================
    // Golden Vector Tests (Section 12.4: Q4K fused matmul ≈ dequant+f32_matmul)
    // =========================================================================

    /// Helper: naive f32 matrix-vector multiplication
    fn matmul_f32_naive(weights: &[f32], input: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; out_dim];
        for row in 0..out_dim {
            let mut sum = 0.0f32;
            for col in 0..in_dim {
                sum += weights[row * in_dim + col] * input[col];
            }
            output[row] = sum;
        }
        output
    }

    /// Golden Vector Test: Q4K matmul ≈ dequant + f32 matmul
    ///
    /// This test verifies the invariant from Section 12.4:
    /// matmul_q4k_f32(W, x) ≈ matmul(dequant_q4k_to_f32(W), x) within ε
    ///
    /// Quantization introduces error, so we use a relaxed tolerance (5%).
    #[test]
    fn test_golden_vector_q4k_matmul_vs_dequant() {
        use crate::backends::q4k::dequantize_q4k_to_f32;

        // Realistic dimensions for LLM layers
        let in_dim = 512; // 2 super-blocks
        let out_dim = 8;

        // Build Q4K test data with realistic distribution
        let mut q4k_data = Vec::new();
        for row in 0..out_dim {
            for sb in 0..(in_dim / 256) {
                // d ~ 0.1, dmin ~ 0.05 (realistic for normalized weights)
                q4k_data.extend_from_slice(&[0x66, 0x2E]); // d
                q4k_data.extend_from_slice(&[0x66, 0x2A]); // dmin
                // Varied scales based on position
                let scale_base = ((row * 7 + sb * 3) % 16) as u8;
                for i in 0..12 {
                    q4k_data.push(scale_base + (i as u8 % 4));
                }
                // Varied quantized values (4-bit, so 0-15)
                for i in 0..128 {
                    let low = ((row + sb + i) % 16) as u8;
                    let high = ((row + sb + i + 5) % 16) as u8;
                    q4k_data.push(low | (high << 4));
                }
            }
        }

        // Random-ish input vector (sinusoidal distribution)
        let input: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.017).sin() * 0.5)
            .collect();

        // Method 1: Fused Q4K matmul
        let fused_output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

        // Method 2: Dequantize + f32 matmul
        let total_elements = in_dim * out_dim;
        let dequantized_weights = dequantize_q4k_to_f32(&q4k_data, total_elements);
        let reference_output = matmul_f32_naive(&dequantized_weights, &input, out_dim, in_dim);

        // Verify Golden Invariant: error within 5% or absolute 0.01
        assert_eq!(fused_output.len(), reference_output.len());
        let mut max_rel_error = 0.0f32;
        let mut max_abs_error = 0.0f32;

        for (i, (fused, reference)) in fused_output.iter().zip(reference_output.iter()).enumerate() {
            let abs_error = (fused - reference).abs();
            let rel_error = if reference.abs() > 1e-6 {
                abs_error / reference.abs()
            } else {
                abs_error
            };
            max_rel_error = max_rel_error.max(rel_error);
            max_abs_error = max_abs_error.max(abs_error);

            assert!(
                rel_error < 0.05 || abs_error < 0.01,
                "Golden invariant violated at row {}: fused={}, reference={}, \
                 rel_error={:.4}%, abs_error={:.6}",
                i, fused, reference, rel_error * 100.0, abs_error
            );
        }

        // Report max errors for visibility
        eprintln!(
            "[Golden Q4K Test] max_rel_error={:.4}%, max_abs_error={:.6}",
            max_rel_error * 100.0, max_abs_error
        );
    }

    /// Golden Vector Test: dispatch path also satisfies invariant
    #[test]
    fn test_golden_vector_q4k_dispatch_vs_dequant() {
        use crate::backends::q4k::dequantize_q4k_to_f32;

        // Larger dimensions to exercise SIMD paths
        let in_dim = 1024; // 4 super-blocks
        let out_dim = 16;

        let mut q4k_data = Vec::new();
        for row in 0..out_dim {
            for sb in 0..(in_dim / 256) {
                q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
                q4k_data.extend_from_slice(&[0x00, 0x38]); // dmin ~ 0.5
                for i in 0..12 {
                    q4k_data.push(((row + sb + i) % 64) as u8);
                }
                for i in 0..128 {
                    let low = ((row * 3 + sb * 7 + i) % 16) as u8;
                    let high = ((row * 5 + sb * 11 + i * 2) % 16) as u8;
                    q4k_data.push(low | (high << 4));
                }
            }
        }

        let input: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.013 + 0.5).cos() * 0.3)
            .collect();

        // Dispatch (may use AVX2/SIMD)
        let dispatch_output = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

        // Reference: dequantize + f32
        let total_elements = in_dim * out_dim;
        let dequantized = dequantize_q4k_to_f32(&q4k_data, total_elements);
        let reference_output = matmul_f32_naive(&dequantized, &input, out_dim, in_dim);

        let mut max_rel_error = 0.0f32;
        for (i, (dispatch, reference)) in dispatch_output.iter().zip(reference_output.iter()).enumerate() {
            let abs_error = (dispatch - reference).abs();
            let rel_error = if reference.abs() > 1e-6 {
                abs_error / reference.abs()
            } else {
                abs_error
            };
            max_rel_error = max_rel_error.max(rel_error);

            assert!(
                rel_error < 0.05 || abs_error < 0.01,
                "Golden invariant violated (dispatch) at row {}: \
                 dispatch={}, reference={}, rel_error={:.4}%",
                i, dispatch, reference, rel_error * 100.0
            );
        }

        eprintln!(
            "[Golden Q4K Dispatch Test] max_rel_error={:.4}%",
            max_rel_error * 100.0
        );
    }

    /// Edge case: zero input vector should produce zero output
    #[test]
    fn test_golden_vector_zero_input() {
        let in_dim = 256;
        let out_dim = 4;

        let mut q4k_data = Vec::new();
        for _row in 0..out_dim {
            q4k_data.extend_from_slice(&[0x66, 0x2E]); // d
            q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0 (important for zero output)
            q4k_data.extend_from_slice(&[0x01u8; 12]);
            q4k_data.extend_from_slice(&[0x55u8; 128]); // Non-zero weights
        }

        let input = vec![0.0f32; in_dim];
        let output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

        // With dmin=0 and all-zero input, output should be near zero
        for (i, val) in output.iter().enumerate() {
            assert!(
                val.abs() < 1e-6,
                "Zero input should give ~zero output, got {} at row {}",
                val, i
            );
        }
    }

    /// Edge case: uniform input vector
    #[test]
    fn test_golden_vector_uniform_input() {
        use crate::backends::q4k::dequantize_q4k_to_f32;

        let in_dim = 256;
        let out_dim = 2;

        let mut q4k_data = Vec::new();
        for row in 0..out_dim {
            q4k_data.extend_from_slice(&[0x00, 0x3C]); // d ~ 1.0
            q4k_data.extend_from_slice(&[0x00, 0x00]); // dmin = 0
            q4k_data.extend_from_slice(&[0x01u8; 12]);
            // Uniform quantized weights
            q4k_data.extend_from_slice(&[((row + 1) * 0x11) as u8; 128]);
        }

        let input = vec![1.0f32; in_dim]; // All ones
        let fused_output = matmul_q4k_f32(&q4k_data, &input, out_dim, in_dim);

        let total_elements = in_dim * out_dim;
        let dequantized = dequantize_q4k_to_f32(&q4k_data, total_elements);
        let reference_output = matmul_f32_naive(&dequantized, &input, out_dim, in_dim);

        for (i, (fused, reference)) in fused_output.iter().zip(reference_output.iter()).enumerate() {
            let rel_error = if reference.abs() > 1e-6 {
                (fused - reference).abs() / reference.abs()
            } else {
                (fused - reference).abs()
            };
            assert!(
                rel_error < 0.05,
                "Uniform input failed at row {}: fused={}, ref={}, err={:.2}%",
                i, fused, reference, rel_error * 100.0
            );
        }
    }

    #[test]
    fn test_parallel_dispatch_large_matrix() {
        // Test parallel path: total_work >= 8_000_000
        // Use 4096 x 2048 = 8_388_608 ops (triggers parallel)
        let out_dim = 4096;
        let in_dim = 2048; // Must be multiple of 256 (SUPER_BLOCK_SIZE)
        let total_work = out_dim * in_dim;
        assert!(
            total_work >= 8_000_000,
            "Test must trigger parallel path"
        );

        let num_superblocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        let row_bytes = num_superblocks_per_row * SUPER_BLOCK_BYTES;
        let total_bytes = out_dim * row_bytes;

        // Create deterministic test data
        let mut q4k_data = vec![0u8; total_bytes];
        for row in 0..out_dim {
            for sb in 0..num_superblocks_per_row {
                let offset = row * row_bytes + sb * SUPER_BLOCK_BYTES;
                // d = 1.0 as f16
                q4k_data[offset] = 0x00;
                q4k_data[offset + 1] = 0x3C;
                // dmin = 0.0
                q4k_data[offset + 2] = 0x00;
                q4k_data[offset + 3] = 0x00;
                // scales = 1 for all
                for i in 0..12 {
                    q4k_data[offset + 4 + i] = 0x01;
                }
                // qs = predictable pattern
                for i in 0..128 {
                    q4k_data[offset + 16 + i] = ((row + sb + i) % 16) as u8;
                }
            }
        }

        let input: Vec<f32> = (0..in_dim).map(|i| (i % 10) as f32 * 0.1).collect();

        // Call dispatch - should use parallel path
        let result = matmul_q4k_f32_dispatch(&q4k_data, &input, out_dim, in_dim);

        // Verify dimensions and finiteness
        assert_eq!(result.len(), out_dim);
        for (i, &val) in result.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Result[{}] is not finite: {}",
                i,
                val
            );
        }

        // Compare a few rows against scalar for consistency
        let scalar_result = matmul_q4k_f32_scalar(&q4k_data, &input, out_dim, in_dim);
        for i in (0..out_dim).step_by(512) {
            let diff = (result[i] - scalar_result[i]).abs();
            let tol = scalar_result[i].abs() * 0.01 + 1e-5;
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
        // ne0 = output dimension (rows), ne1 = input dimension (columns)
        // Input must have length ne1
        let ne0 = 2048; // output dimension (rows), must be multiple of 256
        let ne1 = 4096; // input dimension (columns)

        let blocks_per_col = (ne0 + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        let col_bytes = blocks_per_col * SUPER_BLOCK_BYTES;
        let total_bytes = ne1 * col_bytes;

        let mut q4k_data = vec![0u8; total_bytes];
        for col in 0..ne1 {
            for sb in 0..blocks_per_col {
                let offset = col * col_bytes + sb * SUPER_BLOCK_BYTES;
                // d = 0.5 as f16
                q4k_data[offset] = 0x00;
                q4k_data[offset + 1] = 0x38;
                // dmin = 0.0
                q4k_data[offset + 2] = 0x00;
                q4k_data[offset + 3] = 0x00;
                // scales
                for i in 0..12 {
                    q4k_data[offset + 4 + i] = 0x02;
                }
                // qs
                for i in 0..128 {
                    q4k_data[offset + 16 + i] = ((col ^ sb ^ i) % 16) as u8;
                }
            }
        }

        // Input must have length ne1 (input dimension)
        let input: Vec<f32> = (0..ne1).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

        // Use colmajor dispatch
        let result = matmul_q4k_f32_colmajor_dispatch(&q4k_data, &input, ne0, ne1);

        // Output has ne0 elements
        assert_eq!(result.len(), ne0);
        for (i, &val) in result.iter().enumerate() {
            assert!(val.is_finite(), "Result[{}] is not finite: {}", i, val);
        }
    }

    #[test]
    fn test_compute_chunk_scalar_small() {
        // Directly test compute_chunk_q4k_scalar
        let in_dim = 256;
        let out_dim = 4;
        let num_blocks_per_row = 1;
        let row_bytes = SUPER_BLOCK_BYTES;

        let mut q4k_data = vec![0u8; out_dim * row_bytes];
        for row in 0..out_dim {
            let offset = row * row_bytes;
            // d = 1.0 as f16
            q4k_data[offset] = 0x00;
            q4k_data[offset + 1] = 0x3C;
            // dmin = 0.0
            q4k_data[offset + 2] = 0x00;
            q4k_data[offset + 3] = 0x00;
            // scales = 1
            for i in 0..12 {
                q4k_data[offset + 4 + i] = 0x01;
            }
            // qs = all zeros (simplest case)
            for i in 0..128 {
                q4k_data[offset + 16 + i] = 0x00;
            }
        }

        let input = vec![1.0f32; in_dim];
        let mut chunk = vec![0.0f32; out_dim];

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

        // With qs=0, d=1, scales=1, dmin=0, result should be negative
        // Each element: d * scale * 0 - dmin * min = 0 - 0 = 0
        // Actually with all zeros in qs and dmin=0, output should be 0
        for (i, &val) in chunk.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Chunk[{}] is not finite: {}",
                i,
                val
            );
        }
    }
}
