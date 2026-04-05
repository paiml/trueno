//! AVX-512 SIMD Q4_K GEMV implementation.
//!
//! Contract: avx512-q4k-v1.yaml (C-AVX512-Q4K-001)
//! Processes 16 elements per iteration using zmm registers (2× throughput vs AVX2).
//! References: [46] GPTQ, [47] QuIP# AVX-512 dequant methodology.

use super::super::{parse_q4k_header, SUPER_BLOCK_BYTES, SUPER_BLOCK_SIZE};

/// Fused Q4_K matrix-vector multiply with AVX-512 SIMD (16-wide)
///
/// Contract: avx512-q4k-v1.yaml (C-AVX512-Q4K-001, C-AVX512-Q4K-002)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "fma")]
pub(crate) unsafe fn matmul_q4k_f32_avx512(
    q4k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    unsafe {
        use std::arch::x86_64::*;

        let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;
        let low_mask = _mm512_set1_epi32(0x0F);

        let mut output = vec![0.0f32; out_dim];

        for out_idx in 0..out_dim {
            let row_start = out_idx * row_bytes;
            let mut acc = _mm512_setzero_ps();

            for sb_idx in 0..num_blocks_per_row {
                let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
                let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];
                let input_offset = sb_idx * SUPER_BLOCK_SIZE;
                process_q4k_superblock_avx512(
                    sb_data,
                    input,
                    input_offset,
                    in_dim,
                    low_mask,
                    &mut acc,
                );
            }

            output[out_idx] = hsum_avx512(acc);
        }

        output
    }
}

/// Process one Q4K super-block with AVX-512 (16-wide).
///
/// Each super-block = 256 elements in 4 chunks of 64.
/// Each chunk: 32 low nibbles + 32 high nibbles.
/// AVX-512: 16 elements per iteration → 2 iterations per 32 nibbles.
///
/// NOTE: Dual-accumulator (low→acc0, high→acc1) was tested (2026-04-05)
/// but showed NO improvement. Zen 4's OOO engine already hides the FMA
/// dependency chain across iterations — adding a second accumulator just
/// adds merge overhead without helping the pipeline.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "fma")]
unsafe fn process_q4k_superblock_avx512(
    sb_data: &[u8],
    input: &[f32],
    input_offset: usize,
    in_dim: usize,
    low_mask: std::arch::x86_64::__m512i,
    acc: &mut std::arch::x86_64::__m512,
) {
    unsafe {
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

            let d1_vec = _mm512_set1_ps(d1);
            let dm1_vec = _mm512_set1_ps(dm1);
            let d2_vec = _mm512_set1_ps(d2);
            let dm2_vec = _mm512_set1_ps(dm2);

            // Process low nibbles (32 values) in groups of 16
            let mut i = 0;
            while i + 16 <= 32 {
                let input_base = input_offset + chunk_start + i;
                if input_base + 16 <= in_dim {
                    let q_bytes = _mm_loadu_si128(qs.as_ptr().add(q_start + i) as *const __m128i);
                    let q_i32 = _mm512_cvtepu8_epi32(q_bytes);
                    let q_low = _mm512_and_si512(q_i32, low_mask);
                    let q_f32 = _mm512_cvtepi32_ps(q_low);
                    let x = _mm512_loadu_ps(input.as_ptr().add(input_base));
                    let dequant = _mm512_fmsub_ps(d1_vec, q_f32, dm1_vec);
                    *acc = _mm512_fmadd_ps(dequant, x, *acc);
                }
                i += 16;
            }

            // Process high nibbles (32 values) in groups of 16
            let mut i = 0;
            while i + 16 <= 32 {
                let input_base = input_offset + chunk_start + 32 + i;
                if input_base + 16 <= in_dim {
                    let q_bytes = _mm_loadu_si128(qs.as_ptr().add(q_start + i) as *const __m128i);
                    let q_i32 = _mm512_cvtepu8_epi32(q_bytes);
                    let q_high = _mm512_srli_epi32(q_i32, 4);
                    let q_f32 = _mm512_cvtepi32_ps(q_high);
                    let x = _mm512_loadu_ps(input.as_ptr().add(input_base));
                    let dequant = _mm512_fmsub_ps(d2_vec, q_f32, dm2_vec);
                    *acc = _mm512_fmadd_ps(dequant, x, *acc);
                }
                i += 16;
            }
        }
    }
}

/// AVX-512 horizontal sum of 16 f32 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn hsum_avx512(v: std::arch::x86_64::__m512) -> f32 {
    use std::arch::x86_64::*;
    // Reduce 512→256→128→scalar
    let hi256 = _mm512_extractf32x8_ps(v, 1);
    let lo256 = _mm512_castps512_ps256(v);
    let sum256 = _mm256_add_ps(lo256, hi256);
    // Now reduce 256→scalar using AVX2 hsum
    let hi128 = _mm256_extractf128_ps(sum256, 1);
    let lo128 = _mm256_castps256_ps128(sum256);
    let sum128 = _mm_add_ps(lo128, hi128);
    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);
    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, hi32);
    _mm_cvtss_f32(sum32)
}

/// AVX-512 chunk processor for parallel dispatch.
/// Contract: avx512-q4k-v1.yaml
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "fma")]
pub(crate) unsafe fn compute_chunk_q4k_avx512(
    q4k_data: &[u8],
    input: &[f32],
    chunk: &mut [f32],
    start_row: usize,
    out_dim: usize,
    in_dim: usize,
    num_blocks_per_row: usize,
    row_bytes: usize,
) {
    unsafe {
        use std::arch::x86_64::*;

        let low_mask = _mm512_set1_epi32(0x0F);

        for (local_idx, out_val) in chunk.iter_mut().enumerate() {
            let out_idx = start_row + local_idx;
            if out_idx >= out_dim {
                break;
            }
            let row_start = out_idx * row_bytes;
            let mut acc = _mm512_setzero_ps();

            for sb_idx in 0..num_blocks_per_row {
                let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
                let sb_data = &q4k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];
                let input_offset = sb_idx * SUPER_BLOCK_SIZE;
                process_q4k_superblock_avx512(
                    sb_data,
                    input,
                    input_offset,
                    in_dim,
                    low_mask,
                    &mut acc,
                );
            }

            *out_val = hsum_avx512(acc);
        }
    }
}
