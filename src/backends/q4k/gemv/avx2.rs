//! AVX2 SIMD Q4_K GEMV implementations.
//!
//! Contains 8-wide AVX2+FMA optimized GEMV, super-block processor,
//! horizontal sum helper, and the AVX2 chunk processor for parallel dispatch.

use super::super::{parse_q4k_header, SUPER_BLOCK_BYTES, SUPER_BLOCK_SIZE};

/// Fused Q4_K matrix-vector multiply with AVX2 SIMD (8-wide)
///
/// Processes 8 elements at a time using AVX2 intrinsics.
/// Falls back to scalar for remainder elements.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn matmul_q4k_f32_avx2(
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
            let qs = sb_data.get(16..144).expect("Q4_K: need ≥144 bytes for qs");

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
                        let q_bytes =
                            _mm_loadl_epi64(qs.as_ptr().add(q_start + i) as *const __m128i);

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
                        let q_bytes =
                            _mm_loadl_epi64(qs.as_ptr().add(q_start + i) as *const __m128i);

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

/// Process one Q4K super-block row with AVX2 and accumulate into `acc`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn process_q4k_superblock_avx2(
    sb_data: &[u8],
    input: &[f32],
    input_offset: usize,
    in_dim: usize,
    low_mask: std::arch::x86_64::__m256i,
    acc: &mut std::arch::x86_64::__m256,
) {
    use std::arch::x86_64::*;

    let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
    let qs = sb_data.get(16..144).expect("Q4_K: need ≥144 bytes for qs");

    for chunk_i in 0..4 {
        let chunk_start = chunk_i * 64;
        let q_start = chunk_i * 32;

        let d1 = d * f32::from(scales[chunk_i * 2]);
        let dm1 = dmin * f32::from(mins[chunk_i * 2]);
        let d2 = d * f32::from(scales[chunk_i * 2 + 1]);
        let dm2 = dmin * f32::from(mins[chunk_i * 2 + 1]);

        let d1_vec = _mm256_set1_ps(d1);
        let dm1_vec = _mm256_set1_ps(dm1);
        let d2_vec = _mm256_set1_ps(d2);
        let dm2_vec = _mm256_set1_ps(dm2);

        // Process low nibbles (32 values) in groups of 8
        let mut i = 0;
        while i + 8 <= 32 {
            let input_base = input_offset + chunk_start + i;
            if input_base + 8 <= in_dim {
                let q_bytes = _mm_loadl_epi64(qs.as_ptr().add(q_start + i) as *const __m128i);
                let q_i32 = _mm256_cvtepu8_epi32(q_bytes);
                let q_low = _mm256_and_si256(q_i32, low_mask);
                let q_f32 = _mm256_cvtepi32_ps(q_low);
                let x = _mm256_loadu_ps(input.as_ptr().add(input_base));
                let dequant = _mm256_fmsub_ps(d1_vec, q_f32, dm1_vec);
                *acc = _mm256_fmadd_ps(dequant, x, *acc);
            }
            i += 8;
        }

        // Process high nibbles (32 values) in groups of 8
        let mut i = 0;
        while i + 8 <= 32 {
            let input_base = input_offset + chunk_start + 32 + i;
            if input_base + 8 <= in_dim {
                let q_bytes = _mm_loadl_epi64(qs.as_ptr().add(q_start + i) as *const __m128i);
                let q_i32 = _mm256_cvtepu8_epi32(q_bytes);
                let q_high = _mm256_srli_epi32(q_i32, 4);
                let q_f32 = _mm256_cvtepi32_ps(q_high);
                let x = _mm256_loadu_ps(input.as_ptr().add(input_base));
                let dequant = _mm256_fmsub_ps(d2_vec, q_f32, dm2_vec);
                *acc = _mm256_fmadd_ps(dequant, x, *acc);
            }
            i += 8;
        }
    }
}

/// AVX2 horizontal sum of 8 f32 lanes to a single f32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn hsum_avx2(acc: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn compute_chunk_q4k_avx2(
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
            let input_offset = sb_idx * SUPER_BLOCK_SIZE;
            process_q4k_superblock_avx2(sb_data, input, input_offset, in_dim, low_mask, &mut acc);
        }

        *out_val = hsum_avx2(acc);
    }
}
