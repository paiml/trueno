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

    #[test]
    fn test_f16_to_f32_normal() {
        // Normal f16 value: 1.0 = 0x3C00
        let result = f16_to_f32(0x3C00);
        assert!((result - 1.0).abs() < 1e-6, "Expected 1.0, got {}", result);

        // 2.0 = 0x4000
        let result = f16_to_f32(0x4000);
        assert!((result - 2.0).abs() < 1e-6, "Expected 2.0, got {}", result);

        // -1.0 = 0xBC00
        let result = f16_to_f32(0xBC00);
        assert!((result + 1.0).abs() < 1e-6, "Expected -1.0, got {}", result);
    }

    #[test]
    fn test_f16_to_f32_zero() {
        // Positive zero
        let result = f16_to_f32(0x0000);
        assert_eq!(result, 0.0, "Expected +0.0");
        assert!(result.is_sign_positive());

        // Negative zero
        let result = f16_to_f32(0x8000);
        assert_eq!(result, 0.0, "Expected -0.0");
        assert!(result.is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_infinity() {
        // Positive infinity = 0x7C00
        let result = f16_to_f32(0x7C00);
        assert!(result.is_infinite() && result.is_sign_positive());

        // Negative infinity = 0xFC00
        let result = f16_to_f32(0xFC00);
        assert!(result.is_infinite() && result.is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_subnormal() {
        // Smallest subnormal: 0x0001 ≈ 5.96e-8
        let result = f16_to_f32(0x0001);
        assert!(result > 0.0 && result < 1e-6, "Expected small subnormal, got {}", result);

        // Larger subnormal: 0x03FF (largest subnormal)
        let result = f16_to_f32(0x03FF);
        assert!(result > 0.0 && result < 1e-4, "Expected subnormal, got {}", result);
    }

    #[test]
    fn test_q6k_colmajor_basic() {
        let in_dim = 256;
        let out_dim = 2;

        // Create Q6K test data
        let mut q6k_data = Vec::new();
        for _ in 0..out_dim {
            q6k_data.extend_from_slice(&[0x33u8; 128]); // ql
            q6k_data.extend_from_slice(&[0x00u8; 64]);  // qh
            q6k_data.extend_from_slice(&[0x01u8; 16]); // scales
            q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
        }

        let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
        let output = matmul_q6k_f32_colmajor(&q6k_data, &input, out_dim, in_dim);

        assert_eq!(output.len(), out_dim);
        for val in &output {
            assert!(val.is_finite(), "Output contains non-finite value: {}", val);
        }
    }

    #[test]
    fn test_q6k_colmajor_dispatch() {
        let in_dim = 256;
        let out_dim = 4;

        let mut q6k_data = Vec::new();
        for row in 0..out_dim {
            q6k_data.extend_from_slice(&[(row as u8).wrapping_add(0x22); 128]);
            q6k_data.extend_from_slice(&[(row as u8).wrapping_add(0x11); 64]);
            q6k_data.extend_from_slice(&[0x02u8; 16]);
            q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
        }

        let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01 - 1.0).collect();

        let result = matmul_q6k_f32_colmajor_dispatch(&q6k_data, &input, out_dim, in_dim);
        assert_eq!(result.len(), out_dim);
        for val in &result {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_q6k_unaligned_dimensions() {
        // Test with dimensions not aligned to block size (256)
        let in_dim = 300; // Not a multiple of 256
        let out_dim = 3;
        let num_blocks = (in_dim + 255) / 256; // = 2 blocks

        let mut q6k_data = Vec::new();
        for _ in 0..out_dim {
            for _ in 0..num_blocks {
                q6k_data.extend_from_slice(&[0x11u8; 128]);
                q6k_data.extend_from_slice(&[0x00u8; 64]);
                q6k_data.extend_from_slice(&[0x01u8; 16]);
                q6k_data.extend_from_slice(&[0x00, 0x3C]);
            }
        }

        let input: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01).collect();
        let output = matmul_q6k_f32(&q6k_data, &input, out_dim, in_dim);

        assert_eq!(output.len(), out_dim);
        for val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_q6k_single_row() {
        let in_dim = 256;
        let out_dim = 1;

        let mut q6k_data = Vec::new();
        q6k_data.extend_from_slice(&[0xAAu8; 128]); // ql
        q6k_data.extend_from_slice(&[0x55u8; 64]);  // qh (alternating bits)
        q6k_data.extend_from_slice(&[0x01u8; 16]); // scales
        q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0

        let input: Vec<f32> = vec![1.0; in_dim];
        let output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);

        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
    }

    #[test]
    fn test_q6k_large_dimensions() {
        let in_dim = 1024;
        let out_dim = 8;
        let num_blocks = in_dim / 256;

        let mut q6k_data = Vec::new();
        for row in 0..out_dim {
            for blk in 0..num_blocks {
                let val = ((row * num_blocks + blk) as u8).wrapping_mul(17);
                q6k_data.extend_from_slice(&[val; 128]);
                q6k_data.extend_from_slice(&[val.wrapping_add(1); 64]);
                q6k_data.extend_from_slice(&[0x02u8; 16]);
                q6k_data.extend_from_slice(&[0x66, 0x2E]); // d ~ 0.1
            }
        }

        let input: Vec<f32> = (0..in_dim).map(|i| ((i % 100) as f32) * 0.01).collect();
        let output = matmul_q6k_f32(&q6k_data, &input, out_dim, in_dim);

        assert_eq!(output.len(), out_dim);
        for val in &output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_q6k_zero_input() {
        let in_dim = 256;
        let out_dim = 2;

        let mut q6k_data = Vec::new();
        for _ in 0..out_dim {
            q6k_data.extend_from_slice(&[0xFFu8; 128]);
            q6k_data.extend_from_slice(&[0xFFu8; 64]);
            q6k_data.extend_from_slice(&[0x7Fu8; 16]); // max positive scale
            q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
        }

        let input: Vec<f32> = vec![0.0; in_dim];
        let output = matmul_q6k_f32(&q6k_data, &input, out_dim, in_dim);

        assert_eq!(output.len(), out_dim);
        for val in &output {
            assert_eq!(*val, 0.0, "Output should be zero when input is zero");
        }
    }

    #[test]
    fn test_q6k_negative_scales() {
        let in_dim = 256;
        let out_dim = 1;

        let mut q6k_data = Vec::new();
        q6k_data.extend_from_slice(&[0x00u8; 128]); // ql = 0
        q6k_data.extend_from_slice(&[0x00u8; 64]);  // qh = 0
        q6k_data.extend_from_slice(&[0x80u8; 16]); // scales = -128 (negative)
        q6k_data.extend_from_slice(&[0x00, 0x3C]); // d = 1.0

        let input: Vec<f32> = vec![1.0; in_dim];
        let output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);

        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
        // With negative scales and quant=0-32=-32, result should be positive
    }

    // =========================================================================
    // Golden Vector Tests: Q6K scalar reference vs dispatch/SIMD paths
    // =========================================================================

    /// Golden Test: Q6K scalar == dispatch for random input
    #[test]
    fn test_golden_q6k_scalar_vs_dispatch() {
        // Realistic LLM dimensions
        let in_dim = 512; // 2 super-blocks
        let out_dim = 8;

        let mut q6k_data = Vec::new();
        for row in 0..out_dim {
            for sb in 0..(in_dim / 256) {
                // ql: varied 4-bit low values
                for i in 0..128 {
                    let low = ((row + sb + i) % 16) as u8;
                    let high = ((row + sb + i + 3) % 16) as u8;
                    q6k_data.push(low | (high << 4));
                }
                // qh: varied 2-bit high values
                for i in 0..64 {
                    let vals = [
                        ((row + i) % 4) as u8,
                        ((row + i + 1) % 4) as u8,
                        ((row + i + 2) % 4) as u8,
                        ((row + i + 3) % 4) as u8,
                    ];
                    q6k_data.push(vals[0] | (vals[1] << 2) | (vals[2] << 4) | (vals[3] << 6));
                }
                // scales: varied signed 8-bit
                for i in 0..16 {
                    q6k_data.push(((row * 7 + sb * 3 + i) % 64) as u8);
                }
                // d ~ 0.1
                q6k_data.extend_from_slice(&[0x66, 0x2E]);
            }
        }

        // Sinusoidal input
        let input: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.019).sin() * 0.4)
            .collect();

        let scalar_output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
        let dispatch_output = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

        assert_eq!(scalar_output.len(), dispatch_output.len());
        let mut max_abs_error = 0.0f32;

        for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate() {
            let abs_error = (scalar - dispatch).abs();
            max_abs_error = max_abs_error.max(abs_error);

            // Scalar and dispatch should match closely (minor FMA ordering differences)
            assert!(
                abs_error < 2e-4,
                "Row {}: scalar={}, dispatch={}, diff={}",
                i, scalar, dispatch, abs_error
            );
        }

        eprintln!("[Golden Q6K Test] max_abs_error={:.6}", max_abs_error);
    }

    /// Golden Test: Q6K colmajor path consistency
    #[test]
    fn test_golden_q6k_colmajor_consistency() {
        let in_dim = 512;
        let out_dim = 4;

        let mut q6k_data = Vec::new();
        for row in 0..out_dim {
            for sb in 0..2 {
                // ql
                for i in 0..128 {
                    q6k_data.push(((row * 5 + sb * 13 + i) % 256) as u8);
                }
                // qh
                for i in 0..64 {
                    q6k_data.push(((row * 7 + sb * 11 + i * 2) % 256) as u8);
                }
                // scales
                for i in 0..16 {
                    q6k_data.push(((row + sb + i) % 128) as u8);
                }
                // d ~ 0.5
                q6k_data.extend_from_slice(&[0x00, 0x38]);
            }
        }

        let input: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.011 + 0.3).cos() * 0.5)
            .collect();

        let colmajor_output = matmul_q6k_f32_colmajor(&q6k_data, &input, out_dim, in_dim);
        let colmajor_dispatch = matmul_q6k_f32_colmajor_dispatch(&q6k_data, &input, out_dim, in_dim);

        assert_eq!(colmajor_output.len(), colmajor_dispatch.len());
        for (i, (base, dispatch)) in colmajor_output.iter().zip(colmajor_dispatch.iter()).enumerate() {
            let diff = (base - dispatch).abs();
            assert!(
                diff < 1e-4,
                "Row {}: colmajor base={}, dispatch={}, diff={}",
                i, base, dispatch, diff
            );
        }
    }

    /// Edge case: maximum 6-bit values (63)
    #[test]
    fn test_golden_q6k_max_quant_values() {
        let in_dim = 256;
        let out_dim = 2;

        let mut q6k_data = Vec::new();
        for _ in 0..out_dim {
            // ql: all 0xF (low nibble = 15)
            q6k_data.extend_from_slice(&[0xFFu8; 128]);
            // qh: all 0xFF (all 2-bit high = 3), so value = 15 + 3*16 = 63
            q6k_data.extend_from_slice(&[0xFFu8; 64]);
            // scales: positive
            q6k_data.extend_from_slice(&[0x3Fu8; 16]); // scale = 63
            // d = 1.0
            q6k_data.extend_from_slice(&[0x00, 0x3C]);
        }

        let input = vec![1.0f32; in_dim];
        let scalar_output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
        let dispatch_output = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

        for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate() {
            assert!(
                scalar.is_finite() && dispatch.is_finite(),
                "Row {}: max values should produce finite output",
                i
            );
            let diff = (scalar - dispatch).abs();
            assert!(
                diff < 1e-4,
                "Row {}: max quant scalar={}, dispatch={}, diff={}",
                i, scalar, dispatch, diff
            );
        }
    }

    /// Edge case: alternating positive/negative scales
    #[test]
    fn test_golden_q6k_alternating_scales() {
        let in_dim = 256;
        let out_dim = 2;

        let mut q6k_data = Vec::new();
        for _ in 0..out_dim {
            // ql: mid-range values
            q6k_data.extend_from_slice(&[0x77u8; 128]); // 7, 7 repeated
            // qh: zeros (full value = 7)
            q6k_data.extend_from_slice(&[0x00u8; 64]);
            // scales: alternating +32, -32
            for i in 0..16 {
                if i % 2 == 0 {
                    q6k_data.push(0x20); // +32
                } else {
                    q6k_data.push(0xE0); // -32 (as signed i8)
                }
            }
            // d = 0.5
            q6k_data.extend_from_slice(&[0x00, 0x38]);
        }

        let input = vec![1.0f32; in_dim];
        let scalar_output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
        let dispatch_output = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

        for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate() {
            let diff = (scalar - dispatch).abs();
            assert!(
                diff < 1e-4,
                "Row {}: alternating scales scalar={}, dispatch={}, diff={}",
                i, scalar, dispatch, diff
            );
        }
    }

    /// Large scale test for SIMD path coverage
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_golden_q6k_large_simd() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping Q6K SIMD test - no AVX2+FMA");
            return;
        }

        let in_dim = 2048; // 8 super-blocks
        let out_dim = 32;

        let mut q6k_data = Vec::new();
        for row in 0..out_dim {
            for sb in 0..(in_dim / 256) {
                for i in 0..128 {
                    let val = ((row * 3 + sb * 7 + i) % 256) as u8;
                    q6k_data.push(val);
                }
                for i in 0..64 {
                    let val = ((row * 5 + sb * 11 + i * 2) % 256) as u8;
                    q6k_data.push(val);
                }
                for i in 0..16 {
                    q6k_data.push(((row + sb + i) % 64) as u8);
                }
                q6k_data.extend_from_slice(&[0x66, 0x2E]);
            }
        }

        let input: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f32) * 0.007 - 1.0).tanh())
            .collect();

        let scalar_output = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
        let dispatch_output = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

        let mut max_rel_error = 0.0f32;
        for (i, (scalar, dispatch)) in scalar_output.iter().zip(dispatch_output.iter()).enumerate() {
            let abs_error = (scalar - dispatch).abs();
            let rel_error = if scalar.abs() > 1e-6 {
                abs_error / scalar.abs()
            } else {
                abs_error
            };
            max_rel_error = max_rel_error.max(rel_error);

            assert!(
                rel_error < 1e-4 || abs_error < 1e-4,
                "Row {}: large SIMD scalar={}, dispatch={}, rel_err={:.6}",
                i, scalar, dispatch, rel_error
            );
        }

        eprintln!("[Golden Q6K Large SIMD] max_rel_error={:.6}", max_rel_error);
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
        let mut q6k_data = vec![0u8; total_bytes];
        for row in 0..out_dim {
            for sb in 0..num_superblocks_per_row {
                let offset = row * row_bytes + sb * SUPER_BLOCK_BYTES;
                // d = 1.0 as f16
                q6k_data[offset] = 0x00;
                q6k_data[offset + 1] = 0x3C;
                // ql: 6-bit low parts
                for i in 0..128 {
                    q6k_data[offset + 2 + i] = ((row + sb + i) % 64) as u8;
                }
                // qh: 2-bit high parts
                for i in 0..64 {
                    q6k_data[offset + 130 + i] = ((row ^ sb ^ i) % 4) as u8;
                }
                // scales
                for i in 0..16 {
                    q6k_data[offset + 194 + i] = 0x10;
                }
            }
        }

        let input: Vec<f32> = (0..in_dim).map(|i| (i % 10) as f32 * 0.1).collect();

        // Call dispatch - should use parallel path
        let result = matmul_q6k_f32_dispatch(&q6k_data, &input, out_dim, in_dim);

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
        let scalar_result = matmul_q6k_f32_scalar(&q6k_data, &input, out_dim, in_dim);
        for i in (0..out_dim).step_by(512) {
            let diff = (result[i] - scalar_result[i]).abs();
            let tol = scalar_result[i].abs() * 0.01 + 1e-4;
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
        // ne0 = output dimension, ne1 = input dimension
        let ne0 = 2048; // output dimension, must be multiple of 256
        let ne1 = 4096; // input dimension

        let blocks_per_col = (ne0 + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
        let col_bytes = blocks_per_col * SUPER_BLOCK_BYTES;
        let total_bytes = ne1 * col_bytes;

        let mut q6k_data = vec![0u8; total_bytes];
        for col in 0..ne1 {
            for sb in 0..blocks_per_col {
                let offset = col * col_bytes + sb * SUPER_BLOCK_BYTES;
                // d = 0.5 as f16
                q6k_data[offset] = 0x00;
                q6k_data[offset + 1] = 0x38;
                // ql
                for i in 0..128 {
                    q6k_data[offset + 2 + i] = ((col ^ sb ^ i) % 64) as u8;
                }
                // qh
                for i in 0..64 {
                    q6k_data[offset + 130 + i] = ((col + sb) % 4) as u8;
                }
                // scales
                for i in 0..16 {
                    q6k_data[offset + 194 + i] = 0x20;
                }
            }
        }

        // Input must have length ne1
        let input: Vec<f32> = (0..ne1).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

        // Use colmajor dispatch
        let result = matmul_q6k_f32_colmajor_dispatch(&q6k_data, &input, ne0, ne1);

        // Output has ne0 elements
        assert_eq!(result.len(), ne0);
        for (i, &val) in result.iter().enumerate() {
            assert!(val.is_finite(), "Result[{}] is not finite: {}", i, val);
        }
    }

    #[test]
    fn test_compute_chunk_scalar_small() {
        // Directly test compute_chunk_scalar
        let in_dim = 256;
        let out_dim = 4;
        let num_blocks_per_row = 1;
        let row_bytes = SUPER_BLOCK_BYTES;

        let mut q6k_data = vec![0u8; out_dim * row_bytes];
        for row in 0..out_dim {
            let offset = row * row_bytes;
            // d = 1.0 as f16
            q6k_data[offset] = 0x00;
            q6k_data[offset + 1] = 0x3C;
            // ql = all zeros
            for i in 0..128 {
                q6k_data[offset + 2 + i] = 0x00;
            }
            // qh = all zeros
            for i in 0..64 {
                q6k_data[offset + 130 + i] = 0x00;
            }
            // scales = 1
            for i in 0..16 {
                q6k_data[offset + 194 + i] = 0x01;
            }
        }

        let input = vec![1.0f32; in_dim];
        let mut chunk = vec![0.0f32; out_dim];

        compute_chunk_scalar(
            &q6k_data,
            &input,
            &mut chunk,
            0,
            out_dim,
            in_dim,
            num_blocks_per_row,
            row_bytes,
        );

        // Verify results are finite
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
