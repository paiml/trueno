//! Row-major Q4_K matrix-vector multiplication.
//!
//! This module implements row-major GEMV where weights are stored row-first.
//! Includes scalar, AVX2-optimized, and parallel dispatch implementations.

use super::{parse_q4k_header, SUPER_BLOCK_BYTES, SUPER_BLOCK_SIZE};

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

/// Fused Q4_K matrix-vector multiply with AVX2 SIMD (8-wide)
///
/// Processes 8 elements at a time using AVX2 intrinsics.
/// Falls back to scalar for remainder elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matmul_q4k_f32_avx2(
    q4k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;

    let mut output = vec![0.0f32; out_dim];

    // Mask for extracting low 4 bits
    let low_mask = _mm256_set1_epi32(0x0F);

    for out_idx in 0..out_dim {
        let row_start = out_idx * row_bytes;

        // 8-wide accumulator
        let mut acc = _mm256_setzero_ps();

        for sb_idx in 0..num_blocks_per_row {
            let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
            let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];

            // Parse header
            let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
            let qs = &sb_data[16..144];

            let input_offset = sb_idx * SUPER_BLOCK_SIZE;

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

                // Broadcast scales
                let d1_vec = _mm256_set1_ps(d1);
                let dm1_vec = _mm256_set1_ps(dm1);
                let d2_vec = _mm256_set1_ps(d2);
                let dm2_vec = _mm256_set1_ps(dm2);

                // Process low nibbles (32 values) in groups of 8
                let mut i = 0;
                while i + 8 <= 32 {
                    let input_base = input_offset + chunk_start + i;
                    if input_base + 8 <= in_dim {
                        // Load 8 bytes of quantized values
                        let q_bytes = _mm_loadl_epi64(
                            qs.as_ptr().add(q_start + i) as *const __m128i
                        );

                        // Zero-extend u8 to i32: [b0, b1, ..., b7, 0, 0, ...] -> [b0, b1, ..., b7] as i32
                        let q_i32 = _mm256_cvtepu8_epi32(q_bytes);

                        // Mask low nibbles
                        let q_low = _mm256_and_si256(q_i32, low_mask);

                        // Convert to f32
                        let q_f32 = _mm256_cvtepi32_ps(q_low);

                        // Load 8 input values
                        let x = _mm256_loadu_ps(input.as_ptr().add(input_base));

                        // dequant = d1 * q - dm1
                        let dequant = _mm256_fmsub_ps(d1_vec, q_f32, dm1_vec);

                        // acc += dequant * x
                        acc = _mm256_fmadd_ps(dequant, x, acc);
                    }
                    i += 8;
                }

                // Process high nibbles (32 values) in groups of 8
                let mut i = 0;
                while i + 8 <= 32 {
                    let input_base = input_offset + chunk_start + 32 + i;
                    if input_base + 8 <= in_dim {
                        // Load 8 bytes of quantized values
                        let q_bytes = _mm_loadl_epi64(
                            qs.as_ptr().add(q_start + i) as *const __m128i
                        );

                        // Zero-extend u8 to i32
                        let q_i32 = _mm256_cvtepu8_epi32(q_bytes);

                        // Shift right 4 bits to get high nibbles
                        let q_high = _mm256_srli_epi32(q_i32, 4);

                        // Convert to f32
                        let q_f32 = _mm256_cvtepi32_ps(q_high);

                        // Load 8 input values
                        let x = _mm256_loadu_ps(input.as_ptr().add(input_base));

                        // dequant = d2 * q - dm2
                        let dequant = _mm256_fmsub_ps(d2_vec, q_f32, dm2_vec);

                        // acc += dequant * x
                        acc = _mm256_fmadd_ps(dequant, x, acc);
                    }
                    i += 8;
                }
            }
        }

        // Horizontal sum of 8-wide accumulator
        // acc = [a0, a1, a2, a3, a4, a5, a6, a7]
        let hi128 = _mm256_extractf128_ps(acc, 1);
        let lo128 = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo128, hi128);
        // sum128 = [a0+a4, a1+a5, a2+a6, a3+a7]
        let hi64 = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, hi64);
        // sum64 = [a0+a2+a4+a6, a1+a3+a5+a7, ...]
        let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
        let sum32 = _mm_add_ss(sum64, hi32);

        output[out_idx] = _mm_cvtss_f32(sum32);
    }

    output
}

/// Runtime dispatch for Q4K matmul - uses AVX2 if available, otherwise scalar
#[inline]
pub fn matmul_q4k_f32_dispatch(
    q4k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    {
        // For large matmuls (total work >= ~8M ops), use parallel execution
        // This catches FFN layers (8960x1536) and lm_head (151936x1536)
        // Also catches ffn_down (1536x8960) where out_dim is small but in_dim is large
        let total_work = out_dim * in_dim;
        if total_work >= 8_000_000 {
            return matmul_q4k_f32_parallel(q4k_data, input, out_dim, in_dim);
        }

        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We just verified AVX2 + FMA are available
            return unsafe { matmul_q4k_f32_avx2(q4k_data, input, out_dim, in_dim) };
        }
    }

    // Fallback to scalar with 4-way unroll
    matmul_q4k_f32(q4k_data, input, out_dim, in_dim)
}

/// Fused Q4_K matrix-vector multiply for GGML column-major layout
///
/// Computes: output = input @ Q4K_weight (GGML convention: y = x @ W)
/// where weight is stored in Q4_K format with GGML column-major super-block organization.
///
/// # GGML Column-Major Layout (PMAT-103)
///
/// For a weight tensor with shape [ne0, ne1] in GGML notation:
/// - ne0 is the output dimension (rows)
/// - ne1 is the input/reduction dimension (columns)
/// - Elements are stored column-major: W[i,j] at offset i + j*ne0
/// - Each column j (length ne0) contains weights from input[j] to all outputs
/// - Super-blocks are organized by columns: column j uses super-blocks [j*blocks_per_col, (j+1)*blocks_per_col)
///
/// This matches GGUF tensor storage and enables fused kernel execution without transposition.
///
/// # Arguments
/// * `q4k_data` - Raw Q4K bytes in GGML column-major layout [ne0, ne1]
/// * `input` - F32 input vector [ne1] (input/reduction dimension)
/// * `ne0` - Size of output dimension (rows in GGML, output size)
/// * `ne1` - Size of input/reduction dimension (columns in GGML, input size)
///
/// # Returns
/// F32 output vector [ne0]
///
/// # Example
/// ```rust,ignore
/// // GGUF ffn_gate: shape [intermediate_dim, hidden_dim] = [8960, 1536]
/// // Computes: intermediate = hidden @ ffn_gate
/// let output = matmul_q4k_f32_colmajor(&q4k_bytes, &hidden, 8960, 1536);
/// // output has 8960 elements
/// ```

// ============================================================================
// Parallel Execution Helpers
// ============================================================================

#[cfg(target_arch = "x86_64")]
fn matmul_q4k_f32_parallel(
    q4k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    use std::thread;

    // Use fewer threads with larger chunks for better cache efficiency
    let num_threads = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(12);

    let chunk_size = (out_dim + num_threads - 1) / num_threads;
    let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;

    let mut output = vec![0.0f32; out_dim];
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");

    thread::scope(|s| {
        let input_ref = input;
        let q4k_ref = q4k_data;
        let chunks: Vec<_> = output.chunks_mut(chunk_size).enumerate().collect();

        for (chunk_idx, chunk) in chunks {
            let start_row = chunk_idx * chunk_size;

            s.spawn(move || {
                if has_avx2 {
                    unsafe {
                        compute_chunk_q4k_avx2(
                            q4k_ref,
                            input_ref,
                            chunk,
                            start_row,
                            out_dim,
                            in_dim,
                            num_blocks_per_row,
                            row_bytes,
                        );
                    }
                } else {
                    compute_chunk_q4k_scalar(
                        q4k_ref,
                        input_ref,
                        chunk,
                        start_row,
                        out_dim,
                        in_dim,
                        num_blocks_per_row,
                        row_bytes,
                    );
                }
            });
        }
    });

    output
}

/// Fallback for non-x86_64
#[cfg(not(target_arch = "x86_64"))]
fn matmul_q4k_f32_parallel(
    q4k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    matmul_q4k_f32(q4k_data, input, out_dim, in_dim)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn compute_chunk_q4k_avx2(
    q4k_data: &[u8],
    input: &[f32],
    chunk: &mut [f32],
    start_row: usize,
    out_dim: usize,
    in_dim: usize,
    num_blocks_per_row: usize,
    row_bytes: usize,
) {
    use std::arch::x86_64::*;

    let low_mask = _mm256_set1_epi32(0x0F);

    for (local_idx, out_val) in chunk.iter_mut().enumerate() {
        let out_idx = start_row + local_idx;
        if out_idx >= out_dim {
            break;
        }

        let row_start = out_idx * row_bytes;
        let mut acc = _mm256_setzero_ps();

        for sb_idx in 0..num_blocks_per_row {
            let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
            if sb_start + SUPER_BLOCK_BYTES > q4k_data.len() {
                break;
            }
            let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];

            let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
            let qs = &sb_data[16..144];

            let input_offset = sb_idx * SUPER_BLOCK_SIZE;

            // Process 4 chunks of 64 values each
            for chunk_i in 0..4 {
                let chunk_start = chunk_i * 64;
                let q_start = chunk_i * 32;

                let scale_idx_low = chunk_i * 2;
                let scale_idx_high = chunk_i * 2 + 1;

                let d1 = d * f32::from(scales[scale_idx_low]);
                let dm1 = dmin * f32::from(mins[scale_idx_low]);
                let d2 = d * f32::from(scales[scale_idx_high]);
                let dm2 = dmin * f32::from(mins[scale_idx_high]);

                let d1_vec = _mm256_set1_ps(d1);
                let dm1_vec = _mm256_set1_ps(dm1);
                let d2_vec = _mm256_set1_ps(d2);
                let dm2_vec = _mm256_set1_ps(dm2);

                // Process low nibbles (32 values) in groups of 8
                let mut i = 0;
                while i + 8 <= 32 {
                    let input_base = input_offset + chunk_start + i;
                    if input_base + 8 <= in_dim {
                        let q_bytes = _mm_loadl_epi64(
                            qs.as_ptr().add(q_start + i) as *const __m128i
                        );
                        let q_i32 = _mm256_cvtepu8_epi32(q_bytes);
                        let q_low = _mm256_and_si256(q_i32, low_mask);
                        let q_f32 = _mm256_cvtepi32_ps(q_low);
                        let x = _mm256_loadu_ps(input.as_ptr().add(input_base));
                        let dequant = _mm256_fmsub_ps(d1_vec, q_f32, dm1_vec);
                        acc = _mm256_fmadd_ps(dequant, x, acc);
                    }
                    i += 8;
                }

                // Process high nibbles (32 values) in groups of 8
                let mut i = 0;
                while i + 8 <= 32 {
                    let input_base = input_offset + chunk_start + 32 + i;
                    if input_base + 8 <= in_dim {
                        let q_bytes = _mm_loadl_epi64(
                            qs.as_ptr().add(q_start + i) as *const __m128i
                        );
                        let q_i32 = _mm256_cvtepu8_epi32(q_bytes);
                        let q_high = _mm256_srli_epi32(q_i32, 4);
                        let q_f32 = _mm256_cvtepi32_ps(q_high);
                        let x = _mm256_loadu_ps(input.as_ptr().add(input_base));
                        let dequant = _mm256_fmsub_ps(d2_vec, q_f32, dm2_vec);
                        acc = _mm256_fmadd_ps(dequant, x, acc);
                    }
                    i += 8;
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

        *out_val = _mm_cvtss_f32(sum32);
    }
}

#[allow(dead_code)]
pub(crate) fn compute_chunk_q4k_scalar(
    q4k_data: &[u8],
    input: &[f32],
    chunk: &mut [f32],
    start_row: usize,
    out_dim: usize,
    in_dim: usize,
    num_blocks_per_row: usize,
    row_bytes: usize,
) {
    for (local_idx, out_val) in chunk.iter_mut().enumerate() {
        let out_idx = start_row + local_idx;
        if out_idx >= out_dim {
            break;
        }

        let row_start = out_idx * row_bytes;
        let mut sum = 0.0f32;

        for sb_idx in 0..num_blocks_per_row {
            let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
            if sb_start + SUPER_BLOCK_BYTES > q4k_data.len() {
                break;
            }
            let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];

            let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
            let qs = &sb_data[16..144];

            let input_offset = sb_idx * SUPER_BLOCK_SIZE;

            for chunk_i in 0..4 {
                let chunk_start = chunk_i * 64;
                let q_start = chunk_i * 32;

                let scale_idx_low = chunk_i * 2;
                let scale_idx_high = chunk_i * 2 + 1;

                let d1 = d * f32::from(scales[scale_idx_low]);
                let dm1 = dmin * f32::from(mins[scale_idx_low]);
                let d2 = d * f32::from(scales[scale_idx_high]);
                let dm2 = dmin * f32::from(mins[scale_idx_high]);

                // Low nibbles
                for i in 0..32 {
                    let input_idx = input_offset + chunk_start + i;
                    if input_idx < in_dim {
                        let q_val = (qs[q_start + i] & 0x0F) as f32;
                        sum += (d1 * q_val - dm1) * input[input_idx];
                    }
                }

                // High nibbles
                for i in 0..32 {
                    let input_idx = input_offset + chunk_start + 32 + i;
                    if input_idx < in_dim {
                        let q_val = (qs[q_start + i] >> 4) as f32;
                        sum += (d2 * q_val - dm2) * input[input_idx];
                    }
                }
            }
        }

        *out_val = sum;
    }
}

