//! Column-major Q6_K matrix-vector multiplication.
//!
//! This module implements column-major GEMV for GGML/GGUF format weights,
//! where weights are stored column-first for cache-efficient streaming.

use super::{f16_to_f32, SUPER_BLOCK_BYTES, SUPER_BLOCK_SIZE};

/// Fused Q6_K matrix-vector multiply for GGML column-major layout
///
/// Computes: output = input @ Q6K_weight (GGML convention: y = x @ W)
/// where weight is stored in Q6_K format with GGML column-major super-block organization.
///
/// # Arguments
/// * `q6k_data` - Raw Q6K bytes in GGML column-major layout [ne0, ne1]
/// * `input` - F32 input vector [ne1] (input/reduction dimension)
/// * `ne0` - Size of output dimension (rows in GGML, output size)
/// * `ne1` - Size of input/reduction dimension (columns in GGML, input size)
///
/// # Returns
/// F32 output vector [ne0]
pub fn matmul_q6k_f32_colmajor(
    q6k_data: &[u8],
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

        for sb_idx in 0..blocks_per_col {
            let sb_start = col_start + sb_idx * SUPER_BLOCK_BYTES;
            if sb_start + SUPER_BLOCK_BYTES > q6k_data.len() {
                break;
            }
            let sb_data = &q6k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];

            let ql = &sb_data[0..128];
            let qh = &sb_data[128..192];
            let scales = &sb_data[192..208];
            let d = f16_to_f32(u16::from_le_bytes([sb_data[208], sb_data[209]]));

            let output_offset = sb_idx * SUPER_BLOCK_SIZE;

            for group in 0..16 {
                let scale = (scales[group] as i8) as f32;
                let group_offset = group * 16;

                for j in 0..16 {
                    let idx = group_offset + j;
                    let output_idx = output_offset + idx;
                    if output_idx >= ne0 {
                        continue;
                    }

                    let ql_byte = ql[idx / 2];
                    let low4 = if idx % 2 == 0 {
                        ql_byte & 0x0F
                    } else {
                        ql_byte >> 4
                    };

                    let qh_byte = qh[idx / 4];
                    let qh_shift = (idx % 4) * 2;
                    let high2 = (qh_byte >> qh_shift) & 0x03;

                    let q6 = (low4 | (high2 << 4)) as i8 - 32;
                    let dequant = d * scale * q6 as f32;
                    output[output_idx] += x_j * dequant;
                }
            }
        }
    }

    output
}

/// Runtime dispatch for column-major Q6K matmul
///
/// Uses scalar implementation for correctness.
/// Critical for lm_head which is typically 151936 x 1536 (233M elements).
#[inline]
pub fn matmul_q6k_f32_colmajor_dispatch(
    q6k_data: &[u8],
    input: &[f32],
    ne0: usize,
    ne1: usize,
) -> Vec<f32> {
    matmul_q6k_f32_colmajor(q6k_data, input, ne0, ne1)
}
