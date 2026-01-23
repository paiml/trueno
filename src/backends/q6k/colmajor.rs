//! Column-major Q6_K matrix-vector multiplication.
//!
//! This module implements column-major GEMV for GGML/GGUF format weights,
//! where weights are stored column-first for cache-efficient streaming.

use super::{f16_to_f32, SUPER_BLOCK_BYTES, SUPER_BLOCK_SIZE};

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

/// Parallel column-major Q6K matmul with AVX2
///
/// Uses multiple threads and AVX2 SIMD for large matrices like lm_head.
/// Processes columns in parallel, each thread using SIMD for the dot product.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)] // TODO: Wire into colmajor_dispatch when validated
fn matmul_q6k_f32_colmajor_parallel_avx2(
    q6k_data: &[u8],
    input: &[f32],
    ne0: usize,
    ne1: usize,
) -> Vec<f32> {
    use std::arch::x86_64::*;
    use std::thread;

    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");

    let num_threads = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(12);

    let chunk_size = (ne1 + num_threads - 1) / num_threads;
    let blocks_per_col = (ne0 + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let col_bytes = blocks_per_col * SUPER_BLOCK_BYTES;

    let mut output = vec![0.0f32; ne1];

    thread::scope(|s| {
        let input_ref = input;
        let q6k_ref = q6k_data;
        let chunks: Vec<_> = output.chunks_mut(chunk_size).enumerate().collect();

        for (chunk_idx, chunk) in chunks {
            let start_col = chunk_idx * chunk_size;

            s.spawn(move || {
                if has_avx2 {
                    // SAFETY: AVX2 verified above
                    unsafe {
                        for (local_idx, out_val) in chunk.iter_mut().enumerate() {
                            let col_idx = start_col + local_idx;
                            if col_idx >= ne1 {
                                break;
                            }

                            let col_start = col_idx * col_bytes;
                            let mut acc = _mm256_setzero_ps();
                            let mut scalar_acc = 0.0f32;

                            for sb_idx in 0..blocks_per_col {
                                let sb_start = col_start + sb_idx * SUPER_BLOCK_BYTES;
                                if sb_start + SUPER_BLOCK_BYTES > q6k_ref.len() {
                                    break;
                                }

                                let sb_data = &q6k_ref[sb_start..sb_start + SUPER_BLOCK_BYTES];
                                let ql = &sb_data[0..128];
                                let qh = &sb_data[128..192];
                                let scales = &sb_data[192..208];
                                let d = f16_to_f32(u16::from_le_bytes([sb_data[208], sb_data[209]]));

                                let input_offset = sb_idx * SUPER_BLOCK_SIZE;

                                // Process 16 groups of 16 values each
                                for group in 0..16 {
                                    let scale = (scales[group] as i8) as f32;
                                    let ds = d * scale;
                                    let ds_vec = _mm256_set1_ps(ds);
                                    let group_offset = group * 16;

                                    // Process 8 values at a time with AVX2
                                    for j_base in (0..16).step_by(8) {
                                        let input_idx = input_offset + group_offset + j_base;
                                        if input_idx + 8 <= ne0 {
                                            // Load 8 input values
                                            let x = _mm256_loadu_ps(input_ref.as_ptr().add(input_idx));

                                            // Dequantize 8 Q6 values manually
                                            let mut q_vals = [0i32; 8];
                                            for k in 0..8 {
                                                let idx = group_offset + j_base + k;
                                                let ql_byte = ql[idx / 2];
                                                let low4 = if idx % 2 == 0 {
                                                    ql_byte & 0x0F
                                                } else {
                                                    ql_byte >> 4
                                                };
                                                let qh_byte = qh[idx / 4];
                                                let qh_shift = (idx % 4) * 2;
                                                let high2 = (qh_byte >> qh_shift) & 0x03;
                                                q_vals[k] = (low4 | (high2 << 4)) as i32 - 32;
                                            }

                                            // Load q values into vector
                                            let q_i32 = _mm256_loadu_si256(q_vals.as_ptr() as *const __m256i);
                                            let q_f32 = _mm256_cvtepi32_ps(q_i32);

                                            // acc += ds * q * x
                                            let dq = _mm256_mul_ps(ds_vec, q_f32);
                                            acc = _mm256_fmadd_ps(dq, x, acc);
                                        } else {
                                            // Handle boundary with scalar
                                            for k in 0..8 {
                                                let idx = group_offset + j_base + k;
                                                let input_idx_k = input_offset + idx;
                                                if input_idx_k < ne0 {
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
                                                    scalar_acc += ds * q6 as f32 * input_ref[input_idx_k];
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            // Horizontal sum
                            let hi128 = _mm256_extractf128_ps(acc, 1);
                            let lo128 = _mm256_castps256_ps128(acc);
                            let sum128 = _mm_add_ps(lo128, hi128);
                            let hi64 = _mm_movehl_ps(sum128, sum128);
                            let sum64 = _mm_add_ps(sum128, hi64);
                            let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
                            let sum32 = _mm_add_ss(sum64, hi32);

                            *out_val = _mm_cvtss_f32(sum32) + scalar_acc;
                        }
                    }
                } else {
                    // Scalar fallback
                    for (local_idx, out_val) in chunk.iter_mut().enumerate() {
                        let col_idx = start_col + local_idx;
                        if col_idx >= ne1 {
                            break;
                        }

                        let col_start = col_idx * col_bytes;
                        let mut sum = 0.0f32;

                        for sb_idx in 0..blocks_per_col {
                            let sb_start = col_start + sb_idx * SUPER_BLOCK_BYTES;
                            if sb_start + SUPER_BLOCK_BYTES > q6k_ref.len() {
                                break;
                            }
                            let sb_data = &q6k_ref[sb_start..sb_start + SUPER_BLOCK_BYTES];
                            let ql = &sb_data[0..128];
                            let qh = &sb_data[128..192];
                            let scales = &sb_data[192..208];
                            let d = f16_to_f32(u16::from_le_bytes([sb_data[208], sb_data[209]]));

                            let input_offset = sb_idx * SUPER_BLOCK_SIZE;

                            for group in 0..16 {
                                let scale = (scales[group] as i8) as f32;
                                let group_offset = group * 16;

                                for j in 0..16 {
                                    let idx = group_offset + j;
                                    let input_idx = input_offset + idx;
                                    if input_idx >= ne0 {
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
                                    sum += dequant * input_ref[input_idx];
                                }
                            }
                        }

                        *out_val = sum;
                    }
                }
            });
        }
    });

    output
}

/// Runtime dispatch for column-major Q6K matmul
///
/// Uses scalar implementation for now (correctness first, then optimize).
/// Critical for lm_head which is typically 151936 x 1536 (233M elements).
#[inline]
pub fn matmul_q6k_f32_colmajor_dispatch(
    q6k_data: &[u8],
    input: &[f32],
    ne0: usize,
    ne1: usize,
) -> Vec<f32> {
    // Use scalar kernel for correctness verification
    // TODO: Add parallel version with output partitioning
    matmul_q6k_f32_colmajor(q6k_data, input, ne0, ne1)
}

