//! K-Quantization formats for GGUF/APR model weights (Toyota Way: ONE source of truth)
//!
//! This crate provides quantization functions for converting F32 data to
//! K-quantization formats (Q4_K, Q5_K, Q6_K). This is the ONLY implementation
//! in the Sovereign AI Stack - aprender and realizar import from here.
//!
//! ## Stack Architecture (Toyota Way)
//!
//! ```text
//!        ┌─────────┐
//!        │ apr CLI │
//!        └────┬────┘
//!             │
//!     ┌───────┼───────┬───────────┐
//!     ▼       ▼       ▼           ▼
//! ┌────────┐ ┌────────┐ ┌─────────┐
//! │entrenar│ │aprender│ │realizar │
//! └───┬────┘ └───┬────┘ └────┬────┘
//!     │          │           │
//!     └────┬─────┴───────────┴────┘
//!          ▼
//!       ┌────────────────┐
//!       │  trueno-quant  │  ← YOU ARE HERE
//!       └───────┬────────┘
//!               ▼
//!       ┌────────────────┐
//!       │     trueno     │
//!       └────────────────┘
//! ```
//!
//! ## Format Specifications
//!
//! - Q4_K: 256-element super-blocks, 144 bytes (4.5 bits/weight)
//! - Q5_K: 256-element super-blocks, 176 bytes (5.5 bits/weight)
//! - Q6_K: 256-element super-blocks, 210 bytes (6.5 bits/weight)
//!
//! ## Usage
//!
//! ```rust
//! use trueno_quant::{quantize_q4_k, dequantize_q4_k_to_f32};
//!
//! let data: Vec<f32> = (0..256).map(|i| i as f32 / 10.0).collect();
//! let quantized = quantize_q4_k(&data);
//! let restored = dequantize_q4_k_to_f32(&quantized, 256);
//! ```

#![warn(missing_docs)]

// ============================================================================
// Constants
// ============================================================================

/// Minimum valid f16 normal value (~6.1e-5)
/// Prevents NaN on round-trip through f16 encoding
pub const F16_MIN_NORMAL: f32 = 6.1e-5;

/// Q4_K super-block size (elements per block)
pub const Q4_K_BLOCK_SIZE: usize = 256;

/// Q4_K super-block byte size
pub const Q4_K_BLOCK_BYTES: usize = 144;

/// Q5_K super-block size (elements per block)
pub const Q5_K_BLOCK_SIZE: usize = 256;

/// Q5_K super-block byte size
pub const Q5_K_BLOCK_BYTES: usize = 176;

/// Q6_K super-block size (elements per block)
pub const Q6_K_BLOCK_SIZE: usize = 256;

/// Q6_K super-block byte size
pub const Q6_K_BLOCK_BYTES: usize = 210;

// ============================================================================
// f16 Conversion Helpers
// ============================================================================

/// Convert f32 to f16 (using half crate)
#[inline]
pub fn f32_to_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

/// Convert f16 to f32 (using half crate)
#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

// ============================================================================
// Shared K-Quant Helpers (extracted for cognitive complexity reduction)
// ============================================================================

/// Compute per-sub-block scale and min values from padded data.
///
/// Returns (sub_scales, sub_mins) for 8 sub-blocks of 32 elements each.
/// `quant_max` is the maximum quantized value (15 for Q4_K, 31 for Q5_K).
fn compute_sub_block_stats(padded: &[f32; 256], quant_max: f32) -> ([f32; 8], [f32; 8]) {
    const SUB_BLOCK_SIZE: usize = 32;
    let mut sub_scales = [0.0f32; 8];
    let mut sub_mins = [0.0f32; 8];

    for (j, sub_block) in padded.chunks(SUB_BLOCK_SIZE).enumerate().take(8) {
        let min = sub_block.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = sub_block.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max - min;

        sub_scales[j] = if range > F16_MIN_NORMAL {
            range / quant_max
        } else {
            F16_MIN_NORMAL
        };
        sub_mins[j] = (-min).max(0.0);
    }

    (sub_scales, sub_mins)
}

/// Compute global d and dmin from sub-block statistics, plus quantized 6-bit scales/mins.
fn compute_global_scales(
    sub_scales: &[f32; 8],
    sub_mins: &[f32; 8],
) -> (f32, f32, [u8; 8], [u8; 8]) {
    let max_scale = sub_scales.iter().fold(0.0f32, |a, &b| a.max(b));
    let max_min = sub_mins.iter().fold(0.0f32, |a, &b| a.max(b));

    let d = if max_scale > F16_MIN_NORMAL {
        max_scale / 63.0
    } else {
        F16_MIN_NORMAL
    };
    let dmin = if max_min > F16_MIN_NORMAL {
        max_min / 63.0
    } else {
        F16_MIN_NORMAL
    };

    let mut scales_6bit = [0u8; 8];
    let mut mins_6bit = [0u8; 8];
    for j in 0..8 {
        scales_6bit[j] = ((sub_scales[j] / d).round() as u8).min(63);
        mins_6bit[j] = ((sub_mins[j] / dmin).round() as u8).min(63);
    }

    (d, dmin, scales_6bit, mins_6bit)
}

/// Write the K-quant header: d (f16) + dmin (f16) + packed 12-byte scales.
fn write_kquant_header(
    result: &mut Vec<u8>,
    d: f32,
    dmin: f32,
    scales_6bit: &[u8; 8],
    mins_6bit: &[u8; 8],
) {
    result.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    result.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());

    let mut scales_packed = [0u8; 12];
    for i in 0..4 {
        scales_packed[i] = (scales_6bit[i] & 0x3F) | ((scales_6bit[i + 4] & 0x30) << 2);
        scales_packed[i + 4] = (mins_6bit[i] & 0x3F) | ((mins_6bit[i + 4] & 0x30) << 2);
    }
    for i in 0..4 {
        scales_packed[i + 8] = (scales_6bit[i + 4] & 0x0F) | ((mins_6bit[i + 4] & 0x0F) << 4);
    }
    result.extend_from_slice(&scales_packed);
}

/// Quantize a single value: (value + min_val) / scale, clamped to [0, max_q].
#[inline]
fn quantize_one(value: f32, min_val: f32, scale: f32, max_q: f32) -> u8 {
    if scale > 1e-10 {
        ((value + min_val) / scale).round().clamp(0.0, max_q) as u8
    } else {
        0
    }
}

// ============================================================================
// Q4_K Quantization
// ============================================================================

/// Quantize F32 data to Q4_K format (llama.cpp/candle compatible)
///
/// Q4_K format: 256 elements per super-block, 144 bytes per block
/// Layout: d (2B) + dmin (2B) + scales (12B) + qs (128B)
///
/// Value packing (candle/llama.cpp layout):
/// - For each 64-value chunk: 32 bytes store low nibbles first, then high nibbles
/// - Low nibbles use scale[is], high nibbles use scale[is+1]
pub fn quantize_q4_k(data: &[f32]) -> Vec<u8> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 144;

    if data.is_empty() {
        return vec![];
    }

    let num_blocks = (data.len() + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let mut result = Vec::with_capacity(num_blocks * SUPER_BLOCK_BYTES);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * SUPER_BLOCK_SIZE;
        let block_end = (block_start + SUPER_BLOCK_SIZE).min(data.len());
        let block_data = &data[block_start..block_end];

        let mut padded = [0.0f32; SUPER_BLOCK_SIZE];
        padded[..block_data.len()].copy_from_slice(block_data);

        let (sub_scales, sub_mins) = compute_sub_block_stats(&padded, 15.0);
        let (d, dmin, scales_6bit, mins_6bit) = compute_global_scales(&sub_scales, &sub_mins);
        write_kquant_header(&mut result, d, dmin, &scales_6bit, &mins_6bit);

        // Quantize values into 4-bit packed nibbles
        let mut qs = [0u8; 128];
        for chunk in 0..4 {
            let chunk_start = chunk * 64;
            let is = chunk * 2;
            let scale_lo = d * f32::from(scales_6bit[is]);
            let min_lo = dmin * f32::from(mins_6bit[is]);
            let scale_hi = d * f32::from(scales_6bit[is + 1]);
            let min_hi = dmin * f32::from(mins_6bit[is + 1]);

            for l in 0..32 {
                let q_lo = quantize_one(padded[chunk_start + l], min_lo, scale_lo, 15.0);
                let q_hi = quantize_one(padded[chunk_start + l + 32], min_hi, scale_hi, 15.0);
                qs[chunk * 32 + l] = (q_lo & 0x0F) | ((q_hi & 0x0F) << 4);
            }
        }
        result.extend_from_slice(&qs);
    }

    result
}

/// Quantize F32 matrix to Q4_K format with proper row layout
///
/// Processes each row independently to maintain row-major layout.
pub fn quantize_q4_k_matrix(data: &[f32], shape: &[usize]) -> Vec<u8> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 144;

    if shape.len() != 2 {
        return quantize_q4_k(data);
    }

    let rows = shape[0];
    let cols = shape[1];

    let super_blocks_per_row = (cols + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let padded_cols = super_blocks_per_row * SUPER_BLOCK_SIZE;

    let mut result = Vec::with_capacity(rows * super_blocks_per_row * SUPER_BLOCK_BYTES);

    for row_idx in 0..rows {
        let mut padded_row = vec![0.0f32; padded_cols];
        let row_start = row_idx * cols;
        let row_end = row_start + cols;
        if row_end <= data.len() {
            padded_row[..cols].copy_from_slice(&data[row_start..row_end]);
        }

        let row_q4k = quantize_q4_k(&padded_row);
        result.extend_from_slice(&row_q4k);
    }

    result
}

// ============================================================================
// Q5_K Quantization
// ============================================================================

/// Quantize F32 data to Q5_K format
///
/// Q5_K: 256 elements per super-block, 176 bytes per block
/// Layout: d (2B) + dmin (2B) + scales (12B) + qh (32B) + qs (128B)
pub fn quantize_q5_k(data: &[f32]) -> Vec<u8> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 176;

    if data.is_empty() {
        return vec![];
    }

    let num_blocks = (data.len() + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let mut result = Vec::with_capacity(num_blocks * SUPER_BLOCK_BYTES);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * SUPER_BLOCK_SIZE;
        let block_end = (block_start + SUPER_BLOCK_SIZE).min(data.len());
        let block_data = &data[block_start..block_end];

        let mut padded = [0.0f32; SUPER_BLOCK_SIZE];
        padded[..block_data.len()].copy_from_slice(block_data);

        let (sub_scales, sub_mins) = compute_sub_block_stats(&padded, 31.0);
        let (d, dmin, scales_6bit, mins_6bit) = compute_global_scales(&sub_scales, &sub_mins);
        write_kquant_header(&mut result, d, dmin, &scales_6bit, &mins_6bit);

        // Quantize all 256 values to 5-bit
        let mut q5_vals = [0u8; 256];
        for j in 0..8 {
            let scale = d * f32::from(scales_6bit[j]);
            let min_val = dmin * f32::from(mins_6bit[j]);
            for k in 0..32 {
                q5_vals[j * 32 + k] = quantize_one(padded[j * 32 + k], min_val, scale, 31.0);
            }
        }

        // Pack high bits (qh)
        result.extend_from_slice(&pack_q5k_high_bits(&q5_vals));

        // Pack low 4 bits (qs)
        result.extend_from_slice(&pack_q5k_low_nibbles(&q5_vals));
    }

    result
}

/// Pack Q5_K high bits: extract bit 4 from each value into 32 bytes.
fn pack_q5k_high_bits(q5_vals: &[u8; 256]) -> [u8; 32] {
    let mut qh = [0u8; 32];
    for i in 0..32 {
        let mut h = 0u8;
        for j in 0..8 {
            h |= ((q5_vals[j * 32 + i] >> 4) & 1) << j;
        }
        qh[i] = h;
    }
    qh
}

/// Pack Q5_K low nibbles: combine pairs of 4-bit values into 128 bytes.
fn pack_q5k_low_nibbles(q5_vals: &[u8; 256]) -> [u8; 128] {
    let mut qs = [0u8; 128];
    for j in 0..8 {
        for k in 0..16 {
            let idx1 = j * 32 + k;
            let idx2 = j * 32 + k + 16;
            qs[j * 16 + k] = (q5_vals[idx1] & 0x0F) | ((q5_vals[idx2] & 0x0F) << 4);
        }
    }
    qs
}

/// Quantize F32 matrix to Q5_K format with proper row layout
pub fn quantize_q5_k_matrix(data: &[f32], shape: &[usize]) -> Vec<u8> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 176;

    if shape.len() != 2 {
        return quantize_q5_k(data);
    }

    let rows = shape[0];
    let cols = shape[1];
    let super_blocks_per_row = (cols + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let padded_cols = super_blocks_per_row * SUPER_BLOCK_SIZE;

    let mut result = Vec::with_capacity(rows * super_blocks_per_row * SUPER_BLOCK_BYTES);

    for row_idx in 0..rows {
        let mut padded_row = vec![0.0f32; padded_cols];
        let row_start = row_idx * cols;
        let row_end = row_start + cols;
        if row_end <= data.len() {
            padded_row[..cols].copy_from_slice(&data[row_start..row_end]);
        }

        let row_q5k = quantize_q5_k(&padded_row);
        result.extend_from_slice(&row_q5k);
    }

    result
}

// ============================================================================
// Q6_K Quantization
// ============================================================================

/// Quantize F32 data to Q6_K format (candle/GGUF compatible)
///
/// Q6_K format: 256-element super-blocks
/// Each super block: ql (128 bytes) + qh (64 bytes) + scales (16 bytes) + d (f16) = 210 bytes
/// - 6-bit values stored split: low 4 bits in ql, high 2 bits in qh
/// - 16 sub-blocks of 16 elements each, with int8 scale per sub-block
pub fn quantize_q6_k(data: &[f32]) -> Vec<u8> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 210;

    if data.is_empty() {
        return vec![];
    }

    let num_blocks = (data.len() + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let mut result = Vec::with_capacity(num_blocks * SUPER_BLOCK_BYTES);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * SUPER_BLOCK_SIZE;
        let block_end = (block_start + SUPER_BLOCK_SIZE).min(data.len());
        let block_data = &data[block_start..block_end];

        let mut padded = [0.0f32; SUPER_BLOCK_SIZE];
        padded[..block_data.len()].copy_from_slice(block_data);

        let (d, scales_i8) = compute_q6k_scales(&padded);
        let q6_vals = quantize_q6k_values(&padded, d, &scales_i8);
        let (ql, qh) = pack_q6k_bits(&q6_vals);

        // Write in candle order: ql, qh, scales, d
        result.extend_from_slice(&ql);
        result.extend_from_slice(&qh);
        for s in &scales_i8 {
            result.push(*s as u8);
        }
        result.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }

    result
}

/// Compute Q6_K global scale and per-sub-block int8 scales.
fn compute_q6k_scales(padded: &[f32; 256]) -> (f32, [i8; 16]) {
    let mut sub_scales = [0.0f32; 16];
    for (j, sub_block) in padded.chunks(16).enumerate().take(16) {
        let max_abs = sub_block.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        sub_scales[j] = if max_abs > F16_MIN_NORMAL {
            max_abs / 31.0
        } else {
            F16_MIN_NORMAL
        };
    }

    let max_scale = sub_scales.iter().fold(0.0f32, |a, &b| a.max(b));
    let d = if max_scale > F16_MIN_NORMAL {
        max_scale / 127.0
    } else {
        F16_MIN_NORMAL
    };

    let mut scales_i8 = [0i8; 16];
    for j in 0..16 {
        scales_i8[j] = (sub_scales[j] / d).round().clamp(-127.0, 127.0) as i8;
    }

    (d, scales_i8)
}

/// Quantize 256 padded values to 6-bit Q6_K format.
fn quantize_q6k_values(padded: &[f32; 256], d: f32, scales_i8: &[i8; 16]) -> [u8; 256] {
    let mut q6_vals = [0u8; 256];
    for j in 0..16 {
        let scale = d * f32::from(scales_i8[j]);
        let inv_scale = if scale.abs() > 1e-10 {
            1.0 / scale
        } else {
            0.0
        };
        for k in 0..16 {
            let idx = j * 16 + k;
            let q = (padded[idx] * inv_scale).round().clamp(-32.0, 31.0) as i8;
            q6_vals[idx] = (q + 32) as u8;
        }
    }
    q6_vals
}

/// Pack 256 Q6_K values into ql (128 bytes) and qh (64 bytes) candle/GGUF layout.
fn pack_q6k_bits(q6_vals: &[u8; 256]) -> ([u8; 128], [u8; 64]) {
    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];

    for half in 0..2 {
        let n = half * 128;
        let ql_base = half * 64;
        let qh_base = half * 32;

        for l in 0..32 {
            let q1 = q6_vals[n + l];
            let q2 = q6_vals[n + l + 32];
            let q3 = q6_vals[n + l + 64];
            let q4 = q6_vals[n + l + 96];

            ql[ql_base + l] = (q1 & 0x0F) | ((q3 & 0x0F) << 4);
            ql[ql_base + l + 32] = (q2 & 0x0F) | ((q4 & 0x0F) << 4);

            qh[qh_base + l] = ((q1 >> 4) & 0x03)
                | (((q2 >> 4) & 0x03) << 2)
                | (((q3 >> 4) & 0x03) << 4)
                | (((q4 >> 4) & 0x03) << 6);
        }
    }

    (ql, qh)
}

/// Quantize F32 matrix to Q6_K format with proper row layout
pub fn quantize_q6_k_matrix(data: &[f32], shape: &[usize]) -> Vec<u8> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 210;

    if shape.len() != 2 {
        return quantize_q6_k(data);
    }

    let rows = shape[0];
    let cols = shape[1];
    let super_blocks_per_row = (cols + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let padded_cols = super_blocks_per_row * SUPER_BLOCK_SIZE;

    let mut result = Vec::with_capacity(rows * super_blocks_per_row * SUPER_BLOCK_BYTES);

    for row_idx in 0..rows {
        let mut padded_row = vec![0.0f32; padded_cols];
        let row_start = row_idx * cols;
        let row_end = row_start + cols;
        if row_end <= data.len() {
            padded_row[..cols].copy_from_slice(&data[row_start..row_end]);
        }

        let row_q6k = quantize_q6_k(&padded_row);
        result.extend_from_slice(&row_q6k);
    }

    result
}

// ============================================================================
// Dequantization
// ============================================================================

/// Dequantize Q4_K bytes to F32
pub fn dequantize_q4_k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 144;

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let mut result = vec![0.0f32; num_blocks * SUPER_BLOCK_SIZE];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * SUPER_BLOCK_SIZE;

        if sb_start + SUPER_BLOCK_BYTES > data.len() {
            break;
        }

        let d = sanitize_f16_scale(data[sb_start], data[sb_start + 1]);
        let dmin = sanitize_f16_scale(data[sb_start + 2], data[sb_start + 3]);

        let (scales, mins) = unpack_q4k_scales(&data[sb_start + 4..sb_start + 16]);
        let qs = &data[sb_start + 16..sb_start + 144];

        dequantize_q4k_block(d, dmin, &scales, &mins, qs, &mut result[out_start..]);
    }

    result.truncate(num_elements);
    result
}

/// Sanitize an f16-encoded scale value: return 0.0 for NaN, infinity, or subnormals.
#[inline]
fn sanitize_f16_scale(lo: u8, hi: u8) -> f32 {
    let raw = f16_to_f32(u16::from_le_bytes([lo, hi]));
    if raw.is_nan() || raw.is_infinite() || raw.abs() < F16_MIN_NORMAL {
        0.0
    } else {
        raw
    }
}

/// Unpack Q4_K 12-byte packed scales into 8 scale + 8 min values.
fn unpack_q4k_scales(scales_bytes: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for i in 0..4 {
        scales[i] = scales_bytes[i] & 0x3F;
        mins[i] = scales_bytes[i + 4] & 0x3F;
        scales[i + 4] = (scales_bytes[i + 8] & 0x0F) | ((scales_bytes[i] >> 6) << 4);
        mins[i + 4] = (scales_bytes[i + 8] >> 4) | ((scales_bytes[i + 4] >> 6) << 4);
    }
    (scales, mins)
}

/// Dequantize one Q4_K block (256 values) from packed nibbles.
fn dequantize_q4k_block(
    d: f32,
    dmin: f32,
    scales: &[u8; 8],
    mins: &[u8; 8],
    qs: &[u8],
    output: &mut [f32],
) {
    let mut ys_index = 0;
    for chunk in 0..4 {
        let is = chunk * 2;
        let scale_lo = d * f32::from(scales[is]);
        let min_lo = dmin * f32::from(mins[is]);
        let scale_hi = d * f32::from(scales[is + 1]);
        let min_hi = dmin * f32::from(mins[is + 1]);

        for l in 0..32 {
            let byte = qs[chunk * 32 + l];
            output[ys_index] = scale_lo * (byte & 0x0F) as f32 - min_lo;
            ys_index += 1;
        }
        for l in 0..32 {
            let byte = qs[chunk * 32 + l];
            output[ys_index] = scale_hi * ((byte >> 4) & 0x0F) as f32 - min_hi;
            ys_index += 1;
        }
    }
}

/// Dequantize Q5_K bytes to F32
pub fn dequantize_q5_k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 176;

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let mut result = vec![0.0f32; num_blocks * SUPER_BLOCK_SIZE];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * SUPER_BLOCK_SIZE;

        if sb_start + SUPER_BLOCK_BYTES > data.len() {
            break;
        }

        let d = f16_to_f32(u16::from_le_bytes([data[sb_start], data[sb_start + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([data[sb_start + 2], data[sb_start + 3]]));

        let scales_bytes = &data[sb_start + 4..sb_start + 16];
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];

        for i in 0..4 {
            scales[i] = scales_bytes[i] & 0x3F;
            mins[i] = scales_bytes[i + 4] & 0x3F;
            scales[i + 4] = (scales_bytes[i + 8] & 0x0F) | ((scales_bytes[i] >> 6) << 4);
            mins[i + 4] = (scales_bytes[i + 8] >> 4) | ((scales_bytes[i + 4] >> 6) << 4);
        }

        let qh = &data[sb_start + 16..sb_start + 48];
        let qs = &data[sb_start + 48..sb_start + 176];

        for j in 0..8 {
            let scale = d * f32::from(scales[j]);
            let min_val = dmin * f32::from(mins[j]);
            for k in 0..32 {
                let idx = j * 32 + k;
                let qs_idx = j * 16 + (k % 16);
                let q_lo = if k < 16 {
                    qs[qs_idx] & 0x0F
                } else {
                    (qs[qs_idx] >> 4) & 0x0F
                };
                let q_hi = (qh[k] >> j) & 1;
                let q = q_lo | (q_hi << 4);
                result[out_start + idx] = scale * f32::from(q) - min_val;
            }
        }
    }

    result.truncate(num_elements);
    result
}

/// Dequantize Q6_K bytes to F32
pub fn dequantize_q6_k_to_f32(data: &[u8], num_elements: usize) -> Vec<f32> {
    const SUPER_BLOCK_SIZE: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 210;

    let num_blocks = (num_elements + SUPER_BLOCK_SIZE - 1) / SUPER_BLOCK_SIZE;
    let mut result = vec![0.0f32; num_blocks * SUPER_BLOCK_SIZE];

    for sb_idx in 0..num_blocks {
        let sb_start = sb_idx * SUPER_BLOCK_BYTES;
        let out_start = sb_idx * SUPER_BLOCK_SIZE;

        if sb_start + SUPER_BLOCK_BYTES > data.len() {
            break;
        }

        let ql = &data[sb_start..sb_start + 128];
        let qh = &data[sb_start + 128..sb_start + 192];
        let scales = &data[sb_start + 192..sb_start + 208];
        let d = f16_to_f32(u16::from_le_bytes([
            data[sb_start + 208],
            data[sb_start + 209],
        ]));

        for half in 0..2 {
            let ql_base = half * 64;
            let qh_base = half * 32;
            let out_base = out_start + half * 128;

            for l in 0..32 {
                let q1_lo = ql[ql_base + l] & 0x0F;
                let q2_lo = ql[ql_base + l + 32] & 0x0F;
                let q3_lo = (ql[ql_base + l] >> 4) & 0x0F;
                let q4_lo = (ql[ql_base + l + 32] >> 4) & 0x0F;

                let qh_byte = qh[qh_base + l];
                let q1_hi = (qh_byte & 0x03) << 4;
                let q2_hi = ((qh_byte >> 2) & 0x03) << 4;
                let q3_hi = ((qh_byte >> 4) & 0x03) << 4;
                let q4_hi = ((qh_byte >> 6) & 0x03) << 4;

                let q1 = (q1_lo | q1_hi) as i8 - 32;
                let q2 = (q2_lo | q2_hi) as i8 - 32;
                let q3 = (q3_lo | q3_hi) as i8 - 32;
                let q4 = (q4_lo | q4_hi) as i8 - 32;

                let scale_idx_1 = (half * 8) + (l / 16);
                let scale_idx_2 = (half * 8) + (l / 16) + 2;
                let scale_idx_3 = (half * 8) + (l / 16) + 4;
                let scale_idx_4 = (half * 8) + (l / 16) + 6;

                let s1 = scales[scale_idx_1] as i8;
                let s2 = scales[scale_idx_2] as i8;
                let s3 = scales[scale_idx_3] as i8;
                let s4 = scales[scale_idx_4] as i8;

                result[out_base + l] = d * f32::from(s1) * f32::from(q1);
                result[out_base + l + 32] = d * f32::from(s2) * f32::from(q2);
                result[out_base + l + 64] = d * f32::from(s3) * f32::from(q3);
                result[out_base + l + 96] = d * f32::from(s4) * f32::from(q4);
            }
        }
    }

    result.truncate(num_elements);
    result
}

// ============================================================================
// Transpose Functions (LAYOUT-002: GGUF column-major → APR row-major)
// ============================================================================

/// Transpose Q4K tensor from GGUF column-major to APR row-major layout
///
/// GGUF stores weights as [cols, rows] in column-major order.
/// APR requires [rows, cols] in row-major order.
/// This function dequantizes, transposes, and re-quantizes.
pub fn transpose_q4k_for_matmul(data: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    if shape.len() != 2 {
        return (data.to_vec(), shape.to_vec());
    }

    let cols = shape[0];
    let rows = shape[1];
    let num_elements = rows * cols;

    let f32_data = dequantize_q4_k_to_f32(data, num_elements);

    let mut transposed = vec![0.0f32; num_elements];
    for r in 0..rows {
        for c in 0..cols {
            transposed[r * cols + c] = f32_data[c * rows + r];
        }
    }

    let new_shape = vec![rows, cols];
    let quantized = quantize_q4_k_matrix(&transposed, &new_shape);

    (quantized, new_shape)
}

/// Transpose Q5K tensor from GGUF column-major to APR row-major layout
pub fn transpose_q5k_for_matmul(data: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    if shape.len() != 2 {
        return (data.to_vec(), shape.to_vec());
    }

    let cols = shape[0];
    let rows = shape[1];
    let num_elements = rows * cols;

    let f32_data = dequantize_q5_k_to_f32(data, num_elements);

    let mut transposed = vec![0.0f32; num_elements];
    for r in 0..rows {
        for c in 0..cols {
            transposed[r * cols + c] = f32_data[c * rows + r];
        }
    }

    // Note: APR doesn't have native Q5K, convert to Q6K for better precision
    let new_shape = vec![rows, cols];
    let quantized = quantize_q6_k_matrix(&transposed, &new_shape);

    (quantized, new_shape)
}

/// Transpose Q6K tensor from GGUF column-major to APR row-major layout
pub fn transpose_q6k_for_matmul(data: &[u8], shape: &[usize]) -> (Vec<u8>, Vec<usize>) {
    if shape.len() != 2 {
        return (data.to_vec(), shape.to_vec());
    }

    let cols = shape[0];
    let rows = shape[1];
    let num_elements = rows * cols;

    let f32_data = dequantize_q6_k_to_f32(data, num_elements);

    let mut transposed = vec![0.0f32; num_elements];
    for r in 0..rows {
        for c in 0..cols {
            transposed[r * cols + c] = f32_data[c * rows + r];
        }
    }

    let new_shape = vec![rows, cols];
    let quantized = quantize_q6_k_matrix(&transposed, &new_shape);

    (quantized, new_shape)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4k_roundtrip() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 10.0).collect();

        let quantized = quantize_q4_k(&data);
        assert_eq!(quantized.len(), 144);

        let dequantized = dequantize_q4_k_to_f32(&quantized, 256);

        let data_range =
            data.iter().fold(0.0f32, |a, &b| a.max(b)) - data.iter().fold(0.0f32, |a, &b| a.min(b));

        let max_error: f32 = data
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let relaxed_threshold = data_range * 0.5;
        assert!(
            max_error < relaxed_threshold,
            "Q4K roundtrip error {} exceeds threshold {}",
            max_error,
            relaxed_threshold
        );
    }

    #[test]
    fn test_q6k_roundtrip() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 10.0).collect();

        let quantized = quantize_q6_k(&data);
        assert_eq!(quantized.len(), 210);

        let dequantized = dequantize_q6_k_to_f32(&quantized, 256);

        let max_error: f32 = data
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_error < 1.0,
            "Q6K roundtrip error too high: {}",
            max_error
        );
    }

    #[test]
    fn test_q4k_matrix() {
        let data: Vec<f32> = (0..512).map(|i| i as f32 / 100.0).collect();
        let shape = vec![2, 256];

        let quantized = quantize_q4_k_matrix(&data, &shape);
        assert_eq!(quantized.len(), 2 * 144);
    }

    #[test]
    fn test_transpose_q4k() {
        let cols = 256;
        let rows = 2;
        let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32 / 10.0).collect();

        let quantized = quantize_q4_k(&data);
        let shape = vec![cols, rows];

        let (transposed_data, new_shape) = transpose_q4k_for_matmul(&quantized, &shape);

        assert_eq!(new_shape, vec![rows, cols]);
        assert!(!transposed_data.is_empty());
    }

    #[test]
    fn test_f16_min_normal() {
        let f16_val = half::f16::from_f32(F16_MIN_NORMAL);
        let roundtrip = f16_val.to_f32();
        assert!(
            roundtrip > 0.0,
            "F16_MIN_NORMAL should be positive after f16 roundtrip"
        );
        assert!(roundtrip < 1e-4, "F16_MIN_NORMAL should be small");
    }

    #[test]
    fn test_q5k_roundtrip() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 10.0).collect();

        let quantized = quantize_q5_k(&data);
        assert_eq!(quantized.len(), 176);

        let dequantized = dequantize_q5_k_to_f32(&quantized, 256);

        let max_error: f32 = data
            .iter()
            .zip(dequantized.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        // Q5K should have error between Q4K and Q6K
        let data_range =
            data.iter().fold(0.0f32, |a, &b| a.max(b)) - data.iter().fold(0.0f32, |a, &b| a.min(b));
        let relaxed_threshold = data_range * 0.4;
        assert!(
            max_error < relaxed_threshold,
            "Q5K roundtrip error {} exceeds threshold {}",
            max_error,
            relaxed_threshold
        );
    }

    #[test]
    fn test_constants() {
        assert_eq!(Q4_K_BLOCK_SIZE, 256);
        assert_eq!(Q4_K_BLOCK_BYTES, 144);
        assert_eq!(Q5_K_BLOCK_SIZE, 256);
        assert_eq!(Q5_K_BLOCK_BYTES, 176);
        assert_eq!(Q6_K_BLOCK_SIZE, 256);
        assert_eq!(Q6_K_BLOCK_BYTES, 210);
    }
}
