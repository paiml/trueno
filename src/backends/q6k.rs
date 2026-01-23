//! Fused Q6_K Matrix-Vector Multiply
//!
//! Q6_K format (210 bytes per 256 elements):
//! - `ql`: 128 bytes (lower 4 bits of each value)
//! - `qh`: 64 bytes (upper 2 bits, packed 4 values per byte)
//! - `scales`: 16 bytes (8-bit scales for 16 groups of 16 values)
//! - `d`: 2 bytes (f16 global scale)

// Allow dead_code for experimental SIMD microkernels kept for future optimization work
#![allow(dead_code)]

const SUPER_BLOCK_SIZE: usize = 256;
const SUPER_BLOCK_BYTES: usize = 210;

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

/// Fused Q6_K matrix-vector multiply (scalar reference)
pub fn matmul_q6k_f32_scalar(
    q6k_data: &[u8],
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
        let mut sum = 0.0f32;

        for sb_idx in 0..num_blocks_per_row {
            let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
            if sb_start + SUPER_BLOCK_BYTES > q6k_data.len() {
                break;
            }
            let sb_data = &q6k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];

            // Parse Q6K block: ql[128], qh[64], scales[16], d[2]
            let ql = &sb_data[0..128];
            let qh = &sb_data[128..192];
            let scales = &sb_data[192..208];
            let d = f16_to_f32(u16::from_le_bytes([sb_data[208], sb_data[209]]));

            let input_offset = sb_idx * SUPER_BLOCK_SIZE;

            // Process 16 groups of 16 values each (256 total)
            for group in 0..16 {
                let scale = (scales[group] as i8) as f32;
                let group_offset = group * 16;

                for j in 0..16 {
                    let idx = group_offset + j;
                    let input_idx = input_offset + idx;
                    if input_idx >= in_dim {
                        continue;
                    }

                    // Extract 6-bit value: 4 low bits from ql, 2 high bits from qh
                    let ql_byte = ql[idx / 2];
                    let low4 = if idx % 2 == 0 {
                        ql_byte & 0x0F
                    } else {
                        ql_byte >> 4
                    };

                    // qh is packed: 4 values per byte (2 bits each)
                    let qh_byte = qh[idx / 4];
                    let qh_shift = (idx % 4) * 2;
                    let high2 = (qh_byte >> qh_shift) & 0x03;

                    // Combine to 6-bit value (0-63) then center to signed (-32 to 31)
                    let q6 = (low4 | (high2 << 4)) as i8 - 32;

                    // Dequantize: d * scale * q6
                    let dequant = d * scale * q6 as f32;
                    sum += dequant * input[input_idx];
                }
            }
        }

        output[out_idx] = sum;
    }

    output
}

/// Fused Q6_K matrix-vector multiply with AVX2 SIMD
///
/// Optimized to process groups of 8 values at a time, computing
/// dequant and dot product in one pass without intermediate buffer.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn matmul_q6k_f32_avx2(
    q6k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;

    let mut output = vec![0.0f32; out_dim];

    for out_idx in 0..out_dim {
        let row_start = out_idx * row_bytes;
        let mut acc = _mm256_setzero_ps();

        for sb_idx in 0..num_blocks_per_row {
            let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
            if sb_start + SUPER_BLOCK_BYTES > q6k_data.len() {
                break;
            }
            let sb_data = &q6k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];

            // Parse Q6K block
            let ql = &sb_data[0..128];
            let qh = &sb_data[128..192];
            let scales = &sb_data[192..208];
            let d = f16_to_f32(u16::from_le_bytes([sb_data[208], sb_data[209]]));

            let input_offset = sb_idx * SUPER_BLOCK_SIZE;
            let d_vec = _mm256_set1_ps(d);

            // Process each group of 16 values (scale is constant per group)
            for group in 0..16 {
                let scale = (scales[group] as i8) as f32;
                let scale_vec = _mm256_set1_ps(scale);
                let ds_vec = _mm256_mul_ps(d_vec, scale_vec);
                let group_offset = group * 16;
                let input_group = input_offset + group_offset;

                // Process 8 values at a time (2 iterations per group of 16)
                for half in 0..2 {
                    let half_offset = half * 8;
                    let idx_base = group_offset + half_offset;
                    let input_base = input_group + half_offset;

                    if input_base + 8 > in_dim {
                        continue;
                    }

                    // Extract 8 quantized values
                    // Q6 value = (ql_low4 | (qh_2bit << 4)) - 32
                    let mut q6_vals = [0i32; 8];
                    for i in 0..8 {
                        let idx = idx_base + i;
                        let ql_byte = ql[idx / 2];
                        let low4 = if idx % 2 == 0 {
                            ql_byte & 0x0F
                        } else {
                            ql_byte >> 4
                        };
                        let qh_byte = qh[idx / 4];
                        let qh_shift = (idx % 4) * 2;
                        let high2 = (qh_byte >> qh_shift) & 0x03;
                        q6_vals[i] = ((low4 | (high2 << 4)) as i32) - 32;
                    }

                    // Load into SIMD
                    let q6_i32 = _mm256_loadu_si256(q6_vals.as_ptr() as *const __m256i);
                    let q6_f32 = _mm256_cvtepi32_ps(q6_i32);

                    // Load input
                    let x = _mm256_loadu_ps(input.as_ptr().add(input_base));

                    // Compute: acc += (d * scale * q6) * x
                    let dequant = _mm256_mul_ps(ds_vec, q6_f32);
                    acc = _mm256_fmadd_ps(dequant, x, acc);
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

        output[out_idx] = _mm_cvtss_f32(sum32);
    }

    output
}

/// Runtime dispatch for Q6K matmul - uses AVX2 if available
#[inline]
pub fn matmul_q6k_f32_dispatch(
    q6k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    // For large matmuls (total work >= ~8M ops), use parallel execution
    // This catches FFN layers (8960x1536) and lm_head (151936x1536)
    // Also catches ffn_down (1536x8960) where out_dim is small but in_dim is large
    let total_work = out_dim * in_dim;
    if total_work >= 8_000_000 {
        return matmul_q6k_f32_parallel(q6k_data, input, out_dim, in_dim);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { matmul_q6k_f32_avx2(q6k_data, input, out_dim, in_dim) };
        }
    }
    matmul_q6k_f32_scalar(q6k_data, input, out_dim, in_dim)
}

/// Parallel Q6K matmul using multiple threads with AVX2
#[cfg(target_arch = "x86_64")]
fn matmul_q6k_f32_parallel(
    q6k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    use std::thread;

    // Use fewer threads with larger chunks for better cache efficiency
    let num_threads = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .min(12); // Use 12 threads max for better cache behavior

    let chunk_size = (out_dim + num_threads - 1) / num_threads;
    let num_blocks_per_row = (in_dim + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let row_bytes = num_blocks_per_row * SUPER_BLOCK_BYTES;

    let mut output = vec![0.0f32; out_dim];
    let has_avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");

    thread::scope(|s| {
        let input_ref = input;
        let q6k_ref = q6k_data;
        let chunks: Vec<_> = output.chunks_mut(chunk_size).enumerate().collect();

        for (chunk_idx, chunk) in chunks {
            let start_row = chunk_idx * chunk_size;

            s.spawn(move || {
                if has_avx2 {
                    // Call AVX2 kernel for this chunk
                    unsafe {
                        compute_chunk_avx2(
                            q6k_ref,
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
                    compute_chunk_scalar(
                        q6k_ref,
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
fn matmul_q6k_f32_parallel(
    q6k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    matmul_q6k_f32_scalar(q6k_data, input, out_dim, in_dim)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn compute_chunk_avx2(
    q6k_data: &[u8],
    input: &[f32],
    chunk: &mut [f32],
    start_row: usize,
    out_dim: usize,
    in_dim: usize,
    num_blocks_per_row: usize,
    row_bytes: usize,
) {
    use std::arch::x86_64::*;

    for (local_idx, out_val) in chunk.iter_mut().enumerate() {
        let out_idx = start_row + local_idx;
        if out_idx >= out_dim {
            break;
        }

        let row_start = out_idx * row_bytes;
        let mut acc = _mm256_setzero_ps();

        for sb_idx in 0..num_blocks_per_row {
            let sb_start = row_start + sb_idx * SUPER_BLOCK_BYTES;
            if sb_start + SUPER_BLOCK_BYTES > q6k_data.len() {
                break;
            }
            let sb_data = &q6k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];

            let ql = &sb_data[0..128];
            let qh = &sb_data[128..192];
            let scales = &sb_data[192..208];
            let d = f16_to_f32(u16::from_le_bytes([sb_data[208], sb_data[209]]));

            let input_offset = sb_idx * SUPER_BLOCK_SIZE;
            let d_vec = _mm256_set1_ps(d);

            for group in 0..16 {
                let scale = (scales[group] as i8) as f32;
                let scale_vec = _mm256_set1_ps(scale);
                let ds_vec = _mm256_mul_ps(d_vec, scale_vec);
                let group_offset = group * 16;
                let input_group = input_offset + group_offset;

                for half in 0..2 {
                    let half_offset = half * 8;
                    let idx_base = group_offset + half_offset;
                    let input_base = input_group + half_offset;

                    if input_base + 8 > in_dim {
                        continue;
                    }

                    // Extract 8 quantized values
                    let mut q6_vals = [0i32; 8];
                    for i in 0..8 {
                        let idx = idx_base + i;
                        let ql_byte = ql[idx / 2];
                        let low4 = if idx % 2 == 0 {
                            ql_byte & 0x0F
                        } else {
                            ql_byte >> 4
                        };
                        let qh_byte = qh[idx / 4];
                        let qh_shift = (idx % 4) * 2;
                        let high2 = (qh_byte >> qh_shift) & 0x03;
                        q6_vals[i] = ((low4 | (high2 << 4)) as i32) - 32;
                    }

                    let q6_i32 = _mm256_loadu_si256(q6_vals.as_ptr() as *const __m256i);
                    let q6_f32 = _mm256_cvtepi32_ps(q6_i32);
                    let x = _mm256_loadu_ps(input.as_ptr().add(input_base));
                    let dequant = _mm256_mul_ps(ds_vec, q6_f32);
                    acc = _mm256_fmadd_ps(dequant, x, acc);
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
fn compute_chunk_scalar(
    q6k_data: &[u8],
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
            if sb_start + SUPER_BLOCK_BYTES > q6k_data.len() {
                break;
            }
            let sb_data = &q6k_data[sb_start..sb_start + SUPER_BLOCK_BYTES];

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
                    if input_idx >= in_dim {
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
                    sum += dequant * input[input_idx];
                }
            }
        }

        *out_val = sum;
    }
}

/// Public alias for the optimized Q6K matmul
pub fn matmul_q6k_f32(
    q6k_data: &[u8],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    matmul_q6k_f32_dispatch(q6k_data, input, out_dim, in_dim)
}

/// Fused Q6_K matrix-vector multiply for GGML column-major layout (PMAT-103)
///
/// Computes: output = input @ Q6K_weight (GGML convention: y = x @ W)
/// where weight is stored in Q6_K format with GGML column-major super-block organization.
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

/// Parallel column-major Q6K matmul with AVX2
///
/// Uses multiple threads and AVX2 SIMD for large matrices like lm_head.
/// Processes columns in parallel, each thread using SIMD for the dot product.
#[cfg(target_arch = "x86_64")]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q6k_basic() {
        let in_dim = 256;
        let out_dim = 2;

        // Create Q6K test data (210 bytes per block)
        let mut q6k_data = Vec::new();
        for _ in 0..out_dim {
            // ql: 128 bytes (all zeros = q4 part is 0)
            q6k_data.extend_from_slice(&[0x55u8; 128]); // 5 in each nibble
            // qh: 64 bytes (all zeros = q2 part is 0)
            q6k_data.extend_from_slice(&[0x00u8; 64]);
            // scales: 16 bytes (all ones)
            q6k_data.extend_from_slice(&[0x01u8; 16]);
            // d: f16 = 1.0
            q6k_data.extend_from_slice(&[0x00, 0x3C]);
        }

        let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
        let output = matmul_q6k_f32(&q6k_data, &input, out_dim, in_dim);

        assert_eq!(output.len(), out_dim);
        for val in &output {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_q6k_avx2_vs_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        let in_dim = 512;
        let out_dim = 4;

        let mut q6k_data = Vec::new();
        for row in 0..out_dim {
            for _ in 0..2 {
                // 2 blocks per row
                q6k_data.extend_from_slice(&[(row as u8 * 17).wrapping_add(0x33); 128]);
                q6k_data.extend_from_slice(&[(row as u8).wrapping_add(0x11); 64]);
                q6k_data.extend_from_slice(&[0x02u8; 16]);
                q6k_data.extend_from_slice(&[0x66, 0x2E]); // d ~ 0.1
            }
        }

        let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.002 - 0.5).collect();

        let scalar = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
        let dispatch = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

        for (i, (s, d)) in scalar.iter().zip(dispatch.iter()).enumerate() {
            let diff = (s - d).abs();
            assert!(
                diff < 1e-4,
                "Row {}: scalar {} vs dispatch {} (diff {})",
                i, s, d, diff
            );
        }
    }
}
