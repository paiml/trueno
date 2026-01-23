//! Column-major Q4_K matrix-vector multiplication.
//!
//! This module implements column-major GEMV for GGML/GGUF format weights,
//! where weights are stored column-first for cache-efficient streaming.

use super::{parse_q4k_header, SUPER_BLOCK_BYTES, SUPER_BLOCK_SIZE};

/// Fused Q4_K matrix-vector multiply for GGML column-major layout
///
/// Computes: output = input @ Q4K_weight (GGML convention: y = x @ W)
/// where weight is stored in Q4_K format with GGML column-major super-block organization.
///
/// # GGML Column-Major Layout
///
/// For a weight tensor with shape [ne0, ne1] in GGML notation:
/// - ne0 is the output dimension (rows)
/// - ne1 is the input/reduction dimension (columns)
/// - Elements are stored column-major: W[i,j] at offset i + j*ne0
/// - Each column j (length ne0) contains weights from input[j] to all outputs
///
/// # Arguments
/// * `q4k_data` - Raw Q4K bytes in GGML column-major layout [ne0, ne1]
/// * `input` - F32 input vector [ne1] (input/reduction dimension)
/// * `ne0` - Size of output dimension (rows in GGML, output size)
/// * `ne1` - Size of input/reduction dimension (columns in GGML, input size)
///
/// # Returns
/// F32 output vector [ne0]
pub fn matmul_q4k_f32_colmajor(
    q4k_data: &[u8],
    input: &[f32],
    ne0: usize, // output dimension (rows)
    ne1: usize, // input/reduction dimension (columns)
) -> Vec<f32> {
    assert_eq!(input.len(), ne1, "Input length must match ne1 (input dimension)");

    // Number of super-blocks per column (each column has ne0 elements = output_dim)
    let blocks_per_col = (ne0 + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let col_bytes = blocks_per_col * SUPER_BLOCK_BYTES;

    let mut output = vec![0.0f32; ne0];

    // Process each input column and accumulate to outputs
    // Column j contains weights from input[j] to all ne0 outputs
    for col_idx in 0..ne1 {
        let col_start = col_idx * col_bytes;
        let x_j = input[col_idx]; // Input value for this column

        // Skip if input is zero (common in sparse activations)
        if x_j == 0.0 {
            continue;
        }

        // Process super-blocks for this column
        for sb_idx in 0..blocks_per_col {
            let sb_start = col_start + sb_idx * SUPER_BLOCK_BYTES;

            if sb_start + SUPER_BLOCK_BYTES > q4k_data.len() {
                break;
            }

            let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];

            // Parse header
            let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
            let qs = &sb_data[16..144];

            // Output offset for this super-block
            let output_offset = sb_idx * SUPER_BLOCK_SIZE;

            // Process 4 chunks of 64 values each
            for chunk in 0..4 {
                let chunk_start = chunk * 64;
                let q_start = chunk * 32;

                let scale_idx_low = chunk * 2;
                let scale_idx_high = chunk * 2 + 1;

                let d1 = d * f32::from(scales[scale_idx_low]);
                let dm1 = dmin * f32::from(mins[scale_idx_low]);
                let d2 = d * f32::from(scales[scale_idx_high]);
                let dm2 = dmin * f32::from(mins[scale_idx_high]);

                // Process low nibbles (first 32 values)
                for i in 0..32 {
                    let output_idx = output_offset + chunk_start + i;
                    if output_idx < ne0 {
                        let q_val = (qs[q_start + i] & 0x0F) as f32;
                        let dequant = d1 * q_val - dm1;
                        output[output_idx] += x_j * dequant;
                    }
                }

                // Process high nibbles (next 32 values)
                for i in 0..32 {
                    let output_idx = output_offset + chunk_start + 32 + i;
                    if output_idx < ne0 {
                        let q_val = (qs[q_start + i] >> 4) as f32;
                        let dequant = d2 * q_val - dm2;
                        output[output_idx] += x_j * dequant;
                    }
                }
            }
        }
    }

    output
}

/// Runtime dispatch for column-major Q4K matmul
///
/// Uses scalar implementation for correctness.
/// Matches GGUF tensor layout without requiring transposition.
#[inline]
pub fn matmul_q4k_f32_colmajor_dispatch(
    q4k_data: &[u8],
    input: &[f32],
    ne0: usize,
    ne1: usize,
) -> Vec<f32> {
    matmul_q4k_f32_colmajor(q4k_data, input, ne0, ne1)
}
