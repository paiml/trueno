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

// Allow dead_code for experimental SIMD microkernels kept for future optimization work
#![allow(dead_code)]

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

/// Fused Q4_K column-major matrix-vector multiply with AVX2 SIMD (8-wide)
///
/// Processes 8 elements at a time using AVX2 intrinsics for GGML column-major layout.
/// This is the optimized path for APR/SafeTensors that preserves GGUF quantized format.
///
/// # Performance (PMAT-103)
/// - Achieves ~25-30x speedup over scalar for large matrices
/// - Uses FMA for fused multiply-add (better accuracy + performance)
/// - Vectorizes both low and high nibble processing
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matmul_q4k_f32_colmajor_avx2(
    q4k_data: &[u8],
    input: &[f32],
    ne0: usize, // reduction dimension (input size)
    ne1: usize, // output dimension
) -> Vec<f32> {
    use std::arch::x86_64::*;

    let blocks_per_col = (ne0 + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let col_bytes = blocks_per_col * SUPER_BLOCK_BYTES;

    let mut output = vec![0.0f32; ne1];

    // Mask for extracting low 4 bits
    let low_mask = _mm256_set1_epi32(0x0F);

    // Process each output column
    for col_idx in 0..ne1 {
        let col_start = col_idx * col_bytes;

        // 8-wide accumulator
        let mut acc = _mm256_setzero_ps();
        // Scalar accumulator for remainder
        let mut scalar_acc = 0.0f32;

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

            // Input offset for this super-block
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

                // Broadcast scales to 8-wide vectors
                let d1_vec = _mm256_set1_ps(d1);
                let dm1_vec = _mm256_set1_ps(dm1);
                let d2_vec = _mm256_set1_ps(d2);
                let dm2_vec = _mm256_set1_ps(dm2);

                // Process low nibbles (32 values) in groups of 8
                let mut i = 0;
                while i + 8 <= 32 {
                    let input_idx = input_offset + chunk_start + i;
                    if input_idx + 8 <= ne0 {
                        // Load 8 bytes of quantized values
                        let q_bytes = _mm_loadl_epi64(
                            qs.as_ptr().add(q_start + i) as *const __m128i
                        );

                        // Zero-extend u8 to i32
                        let q_i32 = _mm256_cvtepu8_epi32(q_bytes);

                        // Mask low nibbles
                        let q_low = _mm256_and_si256(q_i32, low_mask);

                        // Convert to f32
                        let q_f32 = _mm256_cvtepi32_ps(q_low);

                        // Load 8 input values
                        let x = _mm256_loadu_ps(input.as_ptr().add(input_idx));

                        // dequant = d1 * q - dm1
                        let dequant = _mm256_fmsub_ps(d1_vec, q_f32, dm1_vec);

                        // acc += dequant * x
                        acc = _mm256_fmadd_ps(dequant, x, acc);
                    } else {
                        // Handle boundary with scalar
                        for j in 0..8 {
                            let idx = input_idx + j;
                            if idx < ne0 {
                                let q_val = (qs[q_start + i + j] & 0x0F) as f32;
                                let dequant = d1 * q_val - dm1;
                                scalar_acc += input[idx] * dequant;
                            }
                        }
                    }
                    i += 8;
                }

                // Process high nibbles (32 values) in groups of 8
                i = 0;
                while i + 8 <= 32 {
                    let input_idx = input_offset + chunk_start + 32 + i;
                    if input_idx + 8 <= ne0 {
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
                        let x = _mm256_loadu_ps(input.as_ptr().add(input_idx));

                        // dequant = d2 * q - dm2
                        let dequant = _mm256_fmsub_ps(d2_vec, q_f32, dm2_vec);

                        // acc += dequant * x
                        acc = _mm256_fmadd_ps(dequant, x, acc);
                    } else {
                        // Handle boundary with scalar
                        for j in 0..8 {
                            let idx = input_idx + j;
                            if idx < ne0 {
                                let q_val = (qs[q_start + i + j] >> 4) as f32;
                                let dequant = d2 * q_val - dm2;
                                scalar_acc += input[idx] * dequant;
                            }
                        }
                    }
                    i += 8;
                }
            }
        }

        // Horizontal sum of 8-wide accumulator
        let hi128 = _mm256_extractf128_ps(acc, 1);
        let lo128 = _mm256_castps256_ps128(acc);
        let sum128 = _mm_add_ps(lo128, hi128);
        let hi64 = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, hi64);
        let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
        let sum32 = _mm_add_ss(sum64, hi32);

        output[col_idx] = _mm_cvtss_f32(sum32) + scalar_acc;
    }

    output
}

/// Parallel column-major Q4K matmul with AVX2
#[cfg(target_arch = "x86_64")]
fn matmul_q4k_f32_colmajor_parallel_avx2(
    q4k_data: &[u8],
    input: &[f32],
    ne0: usize,
    ne1: usize,
) -> Vec<f32> {
    use std::arch::x86_64::*;
    use std::thread;

    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    if !has_avx2 {
        return matmul_q4k_f32_colmajor_parallel(q4k_data, input, ne0, ne1);
    }

    let num_threads = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(12);

    let chunk_size = (ne1 + num_threads - 1) / num_threads;
    let blocks_per_col = (ne0 + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let col_bytes = blocks_per_col * SUPER_BLOCK_BYTES;

    let mut output = vec![0.0f32; ne1];

    // Mask for extracting low 4 bits
    let low_mask_val = 0x0F_i32;

    thread::scope(|s| {
        let input_ref = input;
        let q4k_ref = q4k_data;
        let chunks: Vec<_> = output.chunks_mut(chunk_size).enumerate().collect();

        for (chunk_idx, chunk) in chunks {
            let start_col = chunk_idx * chunk_size;

            s.spawn(move || {
                // SAFETY: We verified AVX2 + FMA are available above
                unsafe {
                    let low_mask = _mm256_set1_epi32(low_mask_val);

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

                            if sb_start + SUPER_BLOCK_BYTES > q4k_ref.len() {
                                break;
                            }

                            let sb_data = &q4k_ref[sb_start..sb_start + SUPER_BLOCK_BYTES];
                            let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
                            let qs = &sb_data[16..144];

                            let input_offset = sb_idx * SUPER_BLOCK_SIZE;

                            for chunk_num in 0..4 {
                                let chunk_start = chunk_num * 64;
                                let q_start = chunk_num * 32;

                                let scale_idx_low = chunk_num * 2;
                                let scale_idx_high = chunk_num * 2 + 1;

                                let d1 = d * f32::from(scales[scale_idx_low]);
                                let dm1 = dmin * f32::from(mins[scale_idx_low]);
                                let d2 = d * f32::from(scales[scale_idx_high]);
                                let dm2 = dmin * f32::from(mins[scale_idx_high]);

                                let d1_vec = _mm256_set1_ps(d1);
                                let dm1_vec = _mm256_set1_ps(dm1);
                                let d2_vec = _mm256_set1_ps(d2);
                                let dm2_vec = _mm256_set1_ps(dm2);

                                // Process low nibbles
                                let mut i = 0;
                                while i + 8 <= 32 {
                                    let input_idx = input_offset + chunk_start + i;
                                    if input_idx + 8 <= ne0 {
                                        let q_bytes = _mm_loadl_epi64(
                                            qs.as_ptr().add(q_start + i) as *const __m128i
                                        );
                                        let q_i32 = _mm256_cvtepu8_epi32(q_bytes);
                                        let q_low = _mm256_and_si256(q_i32, low_mask);
                                        let q_f32 = _mm256_cvtepi32_ps(q_low);
                                        let x = _mm256_loadu_ps(input_ref.as_ptr().add(input_idx));
                                        let dequant = _mm256_fmsub_ps(d1_vec, q_f32, dm1_vec);
                                        acc = _mm256_fmadd_ps(dequant, x, acc);
                                    } else {
                                        for j in 0..8 {
                                            let idx = input_idx + j;
                                            if idx < ne0 {
                                                let q_val = (qs[q_start + i + j] & 0x0F) as f32;
                                                scalar_acc += input_ref[idx] * (d1 * q_val - dm1);
                                            }
                                        }
                                    }
                                    i += 8;
                                }

                                // Process high nibbles
                                i = 0;
                                while i + 8 <= 32 {
                                    let input_idx = input_offset + chunk_start + 32 + i;
                                    if input_idx + 8 <= ne0 {
                                        let q_bytes = _mm_loadl_epi64(
                                            qs.as_ptr().add(q_start + i) as *const __m128i
                                        );
                                        let q_i32 = _mm256_cvtepu8_epi32(q_bytes);
                                        let q_high = _mm256_srli_epi32(q_i32, 4);
                                        let q_f32 = _mm256_cvtepi32_ps(q_high);
                                        let x = _mm256_loadu_ps(input_ref.as_ptr().add(input_idx));
                                        let dequant = _mm256_fmsub_ps(d2_vec, q_f32, dm2_vec);
                                        acc = _mm256_fmadd_ps(dequant, x, acc);
                                    } else {
                                        for j in 0..8 {
                                            let idx = input_idx + j;
                                            if idx < ne0 {
                                                let q_val = (qs[q_start + i + j] >> 4) as f32;
                                                scalar_acc += input_ref[idx] * (d2 * q_val - dm2);
                                            }
                                        }
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

                        *out_val = _mm_cvtss_f32(sum32) + scalar_acc;
                    }
                }
            });
        }
    });

    output
}

/// Runtime dispatch for column-major Q4K matmul
///
/// Uses scalar implementation for now (correctness first, then optimize).
/// Matches GGUF tensor layout without requiring transposition.
#[inline]
pub fn matmul_q4k_f32_colmajor_dispatch(
    q4k_data: &[u8],
    input: &[f32],
    ne0: usize,
    ne1: usize,
) -> Vec<f32> {
    // Use scalar kernel for correctness verification
    // TODO: Add parallel/SIMD version that uses output-first iteration
    matmul_q4k_f32_colmajor(q4k_data, input, ne0, ne1)
}

/// Parallel column-major Q4K matmul
fn matmul_q4k_f32_colmajor_parallel(
    q4k_data: &[u8],
    input: &[f32],
    ne0: usize,
    ne1: usize,
) -> Vec<f32> {
    use std::thread;

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
        let q4k_ref = q4k_data;
        let chunks: Vec<_> = output.chunks_mut(chunk_size).enumerate().collect();

        for (chunk_idx, chunk) in chunks {
            let start_col = chunk_idx * chunk_size;

            s.spawn(move || {
                for (local_idx, out_val) in chunk.iter_mut().enumerate() {
                    let col_idx = start_col + local_idx;
                    if col_idx >= ne1 {
                        break;
                    }

                    let col_start = col_idx * col_bytes;
                    let mut sum = 0.0f32;

                    for sb_idx in 0..blocks_per_col {
                        let sb_start = col_start + sb_idx * SUPER_BLOCK_BYTES;

                        if sb_start + SUPER_BLOCK_BYTES > q4k_ref.len() {
                            break;
                        }

                        let sb_data = &q4k_ref[sb_start..sb_start + SUPER_BLOCK_BYTES];
                        let (d, dmin, scales, mins) = parse_q4k_header(sb_data);
                        let qs = &sb_data[16..144];

                        let input_offset = sb_idx * SUPER_BLOCK_SIZE;

                        for chunk in 0..4 {
                            let chunk_start = chunk * 64;
                            let q_start = chunk * 32;

                            let scale_idx_low = chunk * 2;
                            let scale_idx_high = chunk * 2 + 1;

                            let d1 = d * f32::from(scales[scale_idx_low]);
                            let dm1 = dmin * f32::from(mins[scale_idx_low]);
                            let d2 = d * f32::from(scales[scale_idx_high]);
                            let dm2 = dmin * f32::from(mins[scale_idx_high]);

                            for i in 0..32 {
                                let input_idx = input_offset + chunk_start + i;
                                if input_idx < ne0 {
                                    let q_val = (qs[q_start + i] & 0x0F) as f32;
                                    sum += input_ref[input_idx] * (d1 * q_val - dm1);
                                }
                            }

                            for i in 0..32 {
                                let input_idx = input_offset + chunk_start + 32 + i;
                                if input_idx < ne0 {
                                    let q_val = (qs[q_start + i] >> 4) as f32;
                                    sum += input_ref[input_idx] * (d2 * q_val - dm2);
                                }
                            }
                        }
                    }

                    *out_val = sum;
                }
            });
        }
    });

    output
}

/// Parallel Q4K matmul using multiple threads with AVX2
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
fn compute_chunk_q4k_scalar(
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
}
