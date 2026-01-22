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

const SUPER_BLOCK_SIZE: usize = 256;
const SUPER_BLOCK_BYTES: usize = 144;
const SUB_BLOCK_SIZE: usize = 32;

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
fn parse_q4k_header(block: &[u8]) -> (f32, f32, [u8; 8], [u8; 8]) {
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

/// Fused Q4_K matrix-vector multiply (scalar reference implementation)
///
/// Computes: output = Q4K_weight @ input
/// where weight is stored in Q4_K format (144 bytes per 256 elements)
///
/// This is the **scalar reference implementation** for correctness verification.
/// SIMD-optimized versions should produce identical results within ε = 1e-3.
///
/// # Arguments
/// * `q4k_data` - Raw Q4K bytes for the entire weight matrix [out_dim, in_dim]
/// * `input` - F32 input vector [in_dim]
/// * `out_dim` - Number of output elements (rows of weight matrix)
/// * `in_dim` - Number of input elements (columns of weight matrix)
///
/// # Returns
/// F32 output vector [out_dim]
///
/// # Panics
/// Panics if:
/// - `in_dim` is not a multiple of 256
/// - `q4k_data` length doesn't match expected size
/// - `input` length doesn't match `in_dim`
pub fn matmul_q4k_f32_scalar(
    q4k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), in_dim, "Input length mismatch");
    assert!(
        in_dim % SUPER_BLOCK_SIZE == 0 || in_dim < SUPER_BLOCK_SIZE,
        "in_dim must be multiple of 256 (or smaller for padding)"
    );

    let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;
    let expected_size = out_dim * row_bytes;

    assert!(
        q4k_data.len() >= expected_size,
        "Q4K data too small: {} < {}",
        q4k_data.len(),
        expected_size
    );

    let mut output = vec![0.0f32; out_dim];

    for out_idx in 0..out_dim {
        let row_start = out_idx * row_bytes;
        let mut sum = 0.0f32;

        for sb_idx in 0..num_blocks_per_row {
            let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
            let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];

            // Parse header
            let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
            let qs = &sb_data[16..144];

            // Input offset for this super-block
            let input_offset = sb_idx * SUPER_BLOCK_SIZE;

            // Process 4 chunks of 64 values each
            for chunk in 0..4 {
                let chunk_start = chunk * 64;
                let q_start = chunk * 32;

                // Scale indices for this chunk
                let scale_idx_low = chunk * 2;
                let scale_idx_high = chunk * 2 + 1;

                let d1 = d * f32::from(scales[scale_idx_low]);
                let dm1 = dmin * f32::from(mins[scale_idx_low]);
                let d2 = d * f32::from(scales[scale_idx_high]);
                let dm2 = dmin * f32::from(mins[scale_idx_high]);

                // First 32 values: low nibbles
                for i in 0..32 {
                    let q_val = (qs[q_start + i] & 0x0F) as f32;
                    let dequant = d1 * q_val - dm1;
                    let input_idx = input_offset + chunk_start + i;
                    if input_idx < in_dim {
                        sum += dequant * input[input_idx];
                    }
                }

                // Next 32 values: high nibbles
                for i in 0..32 {
                    let q_val = (qs[q_start + i] >> 4) as f32;
                    let dequant = d2 * q_val - dm2;
                    let input_idx = input_offset + chunk_start + 32 + i;
                    if input_idx < in_dim {
                        sum += dequant * input[input_idx];
                    }
                }
            }
        }

        output[out_idx] = sum;
    }

    output
}

/// Fused Q4_K matrix-vector multiply (optimized with 4-way unrolling)
///
/// This version uses 4 independent accumulators to improve instruction-level
/// parallelism while maintaining scalar correctness.
///
/// # Arguments
/// Same as `matmul_q4k_f32_scalar`
pub fn matmul_q4k_f32(
    q4k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), in_dim, "Input length mismatch");

    let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;

    let mut output = vec![0.0f32; out_dim];

    for out_idx in 0..out_dim {
        let row_start = out_idx * row_bytes;

        // 4 independent accumulators for better ILP
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;

        for sb_idx in 0..num_blocks_per_row {
            let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
            let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];

            // Parse header
            let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
            let qs = &sb_data[16..144];

            // Input offset for this super-block
            let input_offset = sb_idx * SUPER_BLOCK_SIZE;

            // Process 4 chunks, accumulating to different registers
            for chunk in 0..4 {
                let chunk_start = chunk * 64;
                let q_start = chunk * 32;

                let scale_idx_low = chunk * 2;
                let scale_idx_high = chunk * 2 + 1;

                let d1 = d * f32::from(scales[scale_idx_low]);
                let dm1 = dmin * f32::from(mins[scale_idx_low]);
                let d2 = d * f32::from(scales[scale_idx_high]);
                let dm2 = dmin * f32::from(mins[scale_idx_high]);

                // Process low nibbles (first 32) with 4-way unroll
                let mut i = 0;
                while i + 3 < 32 {
                    let input_base = input_offset + chunk_start + i;
                    if input_base + 3 < in_dim {
                        let q0 = (qs[q_start + i] & 0x0F) as f32;
                        let q1 = (qs[q_start + i + 1] & 0x0F) as f32;
                        let q2 = (qs[q_start + i + 2] & 0x0F) as f32;
                        let q3 = (qs[q_start + i + 3] & 0x0F) as f32;

                        acc0 = (d1 * q0 - dm1).mul_add(input[input_base], acc0);
                        acc1 = (d1 * q1 - dm1).mul_add(input[input_base + 1], acc1);
                        acc2 = (d1 * q2 - dm1).mul_add(input[input_base + 2], acc2);
                        acc3 = (d1 * q3 - dm1).mul_add(input[input_base + 3], acc3);
                    }
                    i += 4;
                }
                // Handle remainder
                while i < 32 {
                    let input_idx = input_offset + chunk_start + i;
                    if input_idx < in_dim {
                        let q_val = (qs[q_start + i] & 0x0F) as f32;
                        acc0 = (d1 * q_val - dm1).mul_add(input[input_idx], acc0);
                    }
                    i += 1;
                }

                // Process high nibbles (next 32) with 4-way unroll
                let mut i = 0;
                while i + 3 < 32 {
                    let input_base = input_offset + chunk_start + 32 + i;
                    if input_base + 3 < in_dim {
                        let q0 = (qs[q_start + i] >> 4) as f32;
                        let q1 = (qs[q_start + i + 1] >> 4) as f32;
                        let q2 = (qs[q_start + i + 2] >> 4) as f32;
                        let q3 = (qs[q_start + i + 3] >> 4) as f32;

                        acc0 = (d2 * q0 - dm2).mul_add(input[input_base], acc0);
                        acc1 = (d2 * q1 - dm2).mul_add(input[input_base + 1], acc1);
                        acc2 = (d2 * q2 - dm2).mul_add(input[input_base + 2], acc2);
                        acc3 = (d2 * q3 - dm2).mul_add(input[input_base + 3], acc3);
                    }
                    i += 4;
                }
                // Handle remainder
                while i < 32 {
                    let input_idx = input_offset + chunk_start + 32 + i;
                    if input_idx < in_dim {
                        let q_val = (qs[q_start + i] >> 4) as f32;
                        acc0 = (d2 * q_val - dm2).mul_add(input[input_idx], acc0);
                    }
                    i += 1;
                }
            }
        }

        // Combine all accumulators
        output[out_idx] = (acc0 + acc1) + (acc2 + acc3);
    }

    output
}

/// Dequantize Q4_K data to F32 (for golden test comparison)
///
/// This function fully dequantizes Q4K data to F32, matching the
/// `dequantize_q4_k_to_f32` in aprender/src/format/converter.rs.
pub fn dequantize_q4k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let mut result = vec![0.0f32; num_blocks * SUPER_BLOCK_SIZE];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * SUPER_BLOCK_SIZE;

        if sb_start + SUPER_BLOCK_BYTES > data.len() {
            break;
        }

        let sb_data = &data[sb_start..sb_start + SUPER_BLOCK_BYTES];
        let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
        let qs = &sb_data[16..144];

        let mut ys_index = out_start;

        for chunk in 0..4 {
            let q = &qs[chunk * 32..(chunk + 1) * 32];

            let scale_idx_low = chunk * 2;
            let scale_idx_high = chunk * 2 + 1;

            let d1 = d * f32::from(scales[scale_idx_low]);
            let dm1 = dmin * f32::from(mins[scale_idx_low]);
            let d2 = d * f32::from(scales[scale_idx_high]);
            let dm2 = dmin * f32::from(mins[scale_idx_high]);

            // First pass: 32 low nibbles
            for &byte in q {
                result[ys_index] = d1 * (byte & 0xF) as f32 - dm1;
                ys_index += 1;
            }

            // Second pass: 32 high nibbles
            for &byte in q {
                result[ys_index] = d2 * (byte >> 4) as f32 - dm2;
                ys_index += 1;
            }
        }
    }

    result.truncate(num_elements);
    result
}

#[cfg(test)]
mod tests {
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
}
