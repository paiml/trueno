//! Legacy Quantization GEMV Kernels
//!
//! This module contains GEMV kernels for older quantization formats that are still
//! used in some models but are not the primary focus for optimization.
//!
//! ## Kernels
//!
//! - [`Q8_0GemvKernel`] - 8-bit quantization (32 int8 + fp16 scale)
//! - [`Q4_0GemvKernel`] - 4-bit centered quantization (nibble - 8)
//! - [`Q4_1GemvKernel`] - 4-bit affine quantization (d * nibble + m)
//! - [`Q5_0GemvKernel`] - 5-bit quantization with high bits

use super::{Q5_0_BLOCK_BYTES, Q5_0_BLOCK_SIZE, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_SIZE};
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

// =============================================================================
// Q8_0 GEMV KERNEL
// =============================================================================

/// Q8_0 quantized GEMV kernel for M=1 decode throughput
///
/// Q8_0 is simpler than Q4K: 32 int8 values + 1 fp16 scale per block.
/// Layout: d (fp16, 2 bytes) + qs[32] (32 int8 values) = 34 bytes per block
/// Dequant: value[i] = d * qs[i]
#[derive(Debug, Clone)]
pub struct Q8_0GemvKernel {
    /// K dimension (input dimension)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl Q8_0GemvKernel {
    /// Create a new Q8_0 GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Get number of blocks per row (ceiling division)
    #[must_use]
    pub const fn num_blocks_per_row(&self) -> u32 {
        (self.k + Q8_0_BLOCK_SIZE - 1) / Q8_0_BLOCK_SIZE
    }
}

impl Kernel for Q8_0GemvKernel {
    fn name(&self) -> &str {
        "q8_0_gemv_warp_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("q8_0_gemv_warp_reduce")
            .param(PtxType::U64, "y_ptr") // Output vector (N)
            .param(PtxType::U64, "w_ptr") // Q8_0 weights (N x K/32 blocks)
            .param(PtxType::U64, "x_ptr") // Input vector (K)
            .param(PtxType::U32, "k_dim") // K dimension
            .param(PtxType::U32, "n_dim") // N dimension
            .build(|ctx| {
                // Block = 32 threads (one warp), grid = N blocks
                // Each block computes one output element y[block_id]

                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                // Bounds check
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                // Number of blocks per row: ceil(K / 32)
                let k_rounded = ctx.add_u32(k_dim, Q8_0_BLOCK_SIZE - 1);
                let num_blocks = ctx.div_u32(k_rounded, Q8_0_BLOCK_SIZE);

                // Row base address: w_ptr + block_id * num_blocks * 34
                let block_bytes = ctx.mov_u32_imm(Q8_0_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_blocks, block_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Loop over blocks (each thread handles one value per block)
                let blk_idx = ctx.mov_u32_imm(0);

                ctx.label("blk_loop");
                let blk_done = ctx.setp_ge_u32(blk_idx, num_blocks);
                ctx.branch_if(blk_done, "blk_loop_end");

                // Block address = row_base + blk_idx * 34
                let blk_offset = ctx.mul_wide_u32(blk_idx, Q8_0_BLOCK_BYTES);
                let blk_addr = ctx.add_u64(row_base, blk_offset);

                // Load scale d (fp16 at offset 0)
                let d_f16 = ctx.ld_global_f16(blk_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Load quantized value qs[thread_id] (int8 at offset 2 + thread_id)
                let two_64 = ctx.mov_u64_imm(2);
                let qs_base = ctx.add_u64(blk_addr, two_64);
                let tid_64 = ctx.cvt_u64_u32(thread_id);
                let qs_addr = ctx.add_u64(qs_base, tid_64);
                let q_u8 = ctx.ld_global_u8(qs_addr);

                // Convert int8 to signed: treat as signed byte
                // PTX cvt.s32.s8 interprets the byte as signed
                let q_s32 = ctx.cvt_s32_s8(q_u8);
                let q_f32 = ctx.cvt_f32_s32(q_s32);

                // Dequantize: val = d * q
                let dequant = ctx.mul_f32(d, q_f32);

                // Load activation x[blk_idx * 32 + thread_id]
                let blk_k_base = ctx.mul_u32(blk_idx, Q8_0_BLOCK_SIZE);
                let x_idx = ctx.add_u32_reg(blk_k_base, thread_id);

                // Bounds check for last block (K may not be multiple of 32)
                let x_oob = ctx.setp_ge_u32(x_idx, k_dim);
                ctx.branch_if(x_oob, "skip_mul");

                let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                let x_bytes = ctx.mul_u64(x_idx_64, 4);
                let x_addr = ctx.add_u64(x_ptr, x_bytes);
                let x_val = ctx.ld_global_f32(x_addr);

                ctx.fma_f32_inplace(acc, x_val, dequant);

                ctx.label("skip_mul");
                ctx.add_u32_inplace(blk_idx, 1);
                ctx.branch("blk_loop");

                ctx.label("blk_loop_end");

                // Warp reduce
                let tmp16 = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp16);
                let tmp8 = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp8);
                let tmp4 = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp4);
                let tmp2 = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp2);
                let tmp1 = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp1);

                // Thread 0 writes result
                let one_u32 = ctx.mov_u32_imm(1);
                let is_thread0 = ctx.setp_lt_u32(thread_id, one_u32);
                ctx.branch_if_not(is_thread0, "exit");

                let y_offset = ctx.mul_wide_u32(block_id, 4);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// =============================================================================
// PAR-058-FIX: Q4_0 DEQUANTIZE + GEMV
// =============================================================================

/// Q4_0 block size (32 values per block)
const Q4_0_BLOCK_SIZE: u32 = 32;
/// Q4_0 bytes per block: 2 (d fp16) + 16 (qs) = 18 bytes
const Q4_0_BLOCK_BYTES: u32 = 18;

/// Q4_0 GEMV kernel (fused dequantization + matrix-vector multiply).
///
/// Q4_0 format: 18 bytes per 32 elements (2-byte fp16 scale + 16 bytes packed nibbles).
/// This format is simpler than Q4_K but has a higher compression ratio.
/// Used when GGUF header claims a different qtype but data is actually Q4_0.
///
/// # PTX Implementation
///
/// Each warp processes one output element (row of weight matrix).
/// Lane i processes elements [i, i+32, i+64, ...] within the row.
/// Uses warp shuffle reduction to sum across lanes.
///
/// ```text
/// Block layout: [scale: f16, packed: u8[16]]  (18 bytes for 32 elements)
/// Nibble extraction: low = byte & 0x0F, high = (byte >> 4) & 0x0F
/// Dequant: value = (nibble - 8) * scale
/// ```
#[derive(Debug, Clone)]
pub struct Q4_0GemvKernel {
    /// K dimension (input dimension)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl Q4_0GemvKernel {
    /// Create a new Q4_0 GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Get number of blocks per row (ceiling division)
    #[must_use]
    pub const fn num_blocks_per_row(&self) -> u32 {
        (self.k + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE
    }
}

impl Kernel for Q4_0GemvKernel {
    fn name(&self) -> &str {
        "q4_0_gemv_warp_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("q4_0_gemv_warp_reduce")
            .param(PtxType::U64, "y_ptr") // Output vector (N)
            .param(PtxType::U64, "w_ptr") // Q4_0 weights (N x K/32 blocks)
            .param(PtxType::U64, "x_ptr") // Input vector (K)
            .param(PtxType::U32, "k_dim") // K dimension
            .param(PtxType::U32, "n_dim") // N dimension
            .build(|ctx| {
                // Block = 32 threads (one warp), grid = N blocks
                // Each block computes one output element y[block_id]

                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                // Bounds check
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                // Number of blocks per row: ceil(K / 32)
                let k_rounded = ctx.add_u32(k_dim, Q4_0_BLOCK_SIZE - 1);
                let num_blocks = ctx.div_u32(k_rounded, Q4_0_BLOCK_SIZE);

                // Row base address: w_ptr + block_id * num_blocks * 18
                let block_bytes = ctx.mov_u32_imm(Q4_0_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_blocks, block_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Loop over blocks (each thread handles one value per block)
                let blk_idx = ctx.mov_u32_imm(0);

                ctx.label("blk_loop");
                let blk_done = ctx.setp_ge_u32(blk_idx, num_blocks);
                ctx.branch_if(blk_done, "blk_loop_end");

                // Block address = row_base + blk_idx * 18
                let blk_offset = ctx.mul_wide_u32(blk_idx, Q4_0_BLOCK_BYTES);
                let blk_addr = ctx.add_u64(row_base, blk_offset);

                // Load scale d (fp16 at offset 0)
                let d_f16 = ctx.ld_global_f16(blk_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Load nibble for this thread from qs (offset 2)
                // qs layout: 32 4-bit values packed into 16 bytes
                // Nibble index = thread_id, byte index = thread_id / 2
                // Low/high nibble = thread_id % 2
                let two_64 = ctx.mov_u64_imm(2);
                let qs_base = ctx.add_u64(blk_addr, two_64);

                // byte_idx = thread_id / 2
                let byte_idx = ctx.div_u32(thread_id, 2);
                let byte_idx_64 = ctx.cvt_u64_u32(byte_idx);
                let qs_addr = ctx.add_u64(qs_base, byte_idx_64);

                // Load the byte containing our nibble
                let qs_byte = ctx.ld_global_u8(qs_addr);
                let qs_byte_u32 = ctx.cvt_u32_u8(qs_byte);

                // Extract nibble: if thread_id is odd, use high nibble (>> 4)
                // nibble_select = (thread_id % 2) * 4 = (thread_id & 1) << 2
                let one_u32 = ctx.mov_u32_imm(1);
                let nibble_select = ctx.and_u32(thread_id, one_u32);
                let shift_amount = ctx.mul_u32(nibble_select, 4);
                let shifted = ctx.shr_u32(qs_byte_u32, shift_amount);
                let fifteen_u32 = ctx.mov_u32_imm(15);
                let nibble = ctx.and_u32(shifted, fifteen_u32);

                // Center: q_centered = nibble - 8 (result may be negative, -8 to +7)
                let eight_u32 = ctx.mov_u32_imm(8);
                let q_centered = ctx.sub_u32_reg(nibble, eight_u32);

                // Convert to float and dequantize
                let q_f32 = ctx.cvt_f32_s32(q_centered);
                let dequant = ctx.mul_f32(d, q_f32);

                // Load activation x[blk_idx * 32 + thread_id]
                let blk_k_base = ctx.mul_u32(blk_idx, Q4_0_BLOCK_SIZE);
                let x_idx = ctx.add_u32_reg(blk_k_base, thread_id);

                // Bounds check for last block (K may not be multiple of 32)
                let x_oob = ctx.setp_ge_u32(x_idx, k_dim);
                ctx.branch_if(x_oob, "skip_mul");

                let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                let x_bytes = ctx.mul_u64(x_idx_64, 4);
                let x_addr = ctx.add_u64(x_ptr, x_bytes);
                let x_val = ctx.ld_global_f32(x_addr);

                ctx.fma_f32_inplace(acc, x_val, dequant);

                ctx.label("skip_mul");
                ctx.add_u32_inplace(blk_idx, 1);
                ctx.branch("blk_loop");

                ctx.label("blk_loop_end");

                // Warp reduce
                let tmp16 = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp16);
                let tmp8 = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp8);
                let tmp4 = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp4);
                let tmp2 = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp2);
                let tmp1 = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp1);

                // Thread 0 writes result
                let is_thread0 = ctx.setp_lt_u32(thread_id, one_u32);
                ctx.branch_if_not(is_thread0, "exit");

                let y_offset = ctx.mul_wide_u32(block_id, 4);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// =============================================================================
// PAR-058-FIX: Q4_1 DEQUANTIZE + GEMV (for Qwen2.5-0.5B FFN down weights)
// =============================================================================

/// Q4_1 block size (32 values per block, same as Q4_0)
const Q4_1_BLOCK_SIZE: u32 = 32;
/// Q4_1 bytes per block: 2 (d fp16) + 2 (m fp16) + 16 (qs) = 20 bytes
const Q4_1_BLOCK_BYTES: u32 = 20;

/// Q4_1 GEMV kernel - handles affine quantization with scale + offset
///
/// Q4_1 format (per block of 32 elements):
/// - d: fp16 scale (2 bytes, offset 0)
/// - m: fp16 min/offset (2 bytes, offset 2)
/// - qs: packed 4-bit nibbles (16 bytes, offset 4)
///
/// Dequantization: val = d * nibble + m
///
/// Used by Qwen2.5-0.5B which has some FFN down weights in Q4_1 format
/// despite GGUF metadata saying Q4_K.
#[derive(Debug, Clone)]
pub struct Q4_1GemvKernel {
    /// K dimension (input dimension)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl Q4_1GemvKernel {
    /// Create a new Q4_1 GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Get number of blocks per row (ceiling division)
    #[must_use]
    pub const fn num_blocks_per_row(&self) -> u32 {
        (self.k + Q4_1_BLOCK_SIZE - 1) / Q4_1_BLOCK_SIZE
    }
}

impl Kernel for Q4_1GemvKernel {
    fn name(&self) -> &str {
        "q4_1_gemv_warp_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("q4_1_gemv_warp_reduce")
            .param(PtxType::U64, "y_ptr") // Output vector (N)
            .param(PtxType::U64, "w_ptr") // Q4_1 weights (N x K/32 blocks)
            .param(PtxType::U64, "x_ptr") // Input vector (K)
            .param(PtxType::U32, "k_dim") // K dimension
            .param(PtxType::U32, "n_dim") // N dimension
            .build(|ctx| {
                // Block = 32 threads (one warp), grid = N blocks
                // Each block computes one output element y[block_id]

                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                // Bounds check
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                // Number of blocks per row: ceil(K / 32)
                let k_rounded = ctx.add_u32(k_dim, Q4_1_BLOCK_SIZE - 1);
                let num_blocks = ctx.div_u32(k_rounded, Q4_1_BLOCK_SIZE);

                // Row base address: w_ptr + block_id * num_blocks * 20
                let block_bytes = ctx.mov_u32_imm(Q4_1_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_blocks, block_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Loop over blocks (each thread handles one value per block)
                let blk_idx = ctx.mov_u32_imm(0);

                ctx.label("blk_loop");
                let blk_done = ctx.setp_ge_u32(blk_idx, num_blocks);
                ctx.branch_if(blk_done, "blk_loop_end");

                // Block address = row_base + blk_idx * 20
                let blk_offset = ctx.mul_wide_u32(blk_idx, Q4_1_BLOCK_BYTES);
                let blk_addr = ctx.add_u64(row_base, blk_offset);

                // Load scale d (fp16 at offset 0)
                let d_f16 = ctx.ld_global_f16(blk_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Load min m (fp16 at offset 2)
                let two_64 = ctx.mov_u64_imm(2);
                let m_addr = ctx.add_u64(blk_addr, two_64);
                let m_f16 = ctx.ld_global_f16(m_addr);
                let m = ctx.cvt_f32_f16(m_f16);

                // Load nibble for this thread from qs (offset 4)
                // qs layout: 32 4-bit values packed into 16 bytes
                // Nibble index = thread_id, byte index = thread_id / 2
                // Low/high nibble = thread_id % 2
                let four_64 = ctx.mov_u64_imm(4);
                let qs_base = ctx.add_u64(blk_addr, four_64);

                // byte_idx = thread_id / 2
                let byte_idx = ctx.div_u32(thread_id, 2);
                let byte_idx_64 = ctx.cvt_u64_u32(byte_idx);
                let qs_addr = ctx.add_u64(qs_base, byte_idx_64);

                // Load the byte containing our nibble
                let qs_byte = ctx.ld_global_u8(qs_addr);
                let qs_byte_u32 = ctx.cvt_u32_u8(qs_byte);

                // Extract nibble: if thread_id is odd, use high nibble (>> 4)
                // nibble_select = (thread_id % 2) * 4 = (thread_id & 1) << 2
                let one_u32 = ctx.mov_u32_imm(1);
                let nibble_select = ctx.and_u32(thread_id, one_u32);
                let shift_amount = ctx.mul_u32(nibble_select, 4);
                let shifted = ctx.shr_u32(qs_byte_u32, shift_amount);
                let fifteen_u32 = ctx.mov_u32_imm(15);
                let nibble = ctx.and_u32(shifted, fifteen_u32);

                // Q4_1: val = d * nibble + m (affine quantization, no centering)
                let q_f32 = ctx.cvt_f32_u32(nibble);
                let dequant = ctx.fma_f32(d, q_f32, m);

                // Load activation x[blk_idx * 32 + thread_id]
                let blk_k_base = ctx.mul_u32(blk_idx, Q4_1_BLOCK_SIZE);
                let x_idx = ctx.add_u32_reg(blk_k_base, thread_id);

                // Bounds check for last block (K may not be multiple of 32)
                let x_oob = ctx.setp_ge_u32(x_idx, k_dim);
                ctx.branch_if(x_oob, "skip_mul");

                let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                let x_bytes = ctx.mul_u64(x_idx_64, 4);
                let x_addr = ctx.add_u64(x_ptr, x_bytes);
                let x_val = ctx.ld_global_f32(x_addr);

                ctx.fma_f32_inplace(acc, x_val, dequant);

                ctx.label("skip_mul");
                ctx.add_u32_inplace(blk_idx, 1);
                ctx.branch("blk_loop");

                ctx.label("blk_loop_end");

                // Warp reduce
                let tmp16 = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp16);
                let tmp8 = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp8);
                let tmp4 = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp4);
                let tmp2 = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp2);
                let tmp1 = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp1);

                // Thread 0 writes result
                let is_thread0 = ctx.setp_lt_u32(thread_id, one_u32);
                ctx.branch_if_not(is_thread0, "exit");

                let y_offset = ctx.mul_wide_u32(block_id, 4);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// =============================================================================
// PAR-054: Q5_0 DEQUANTIZE + GEMV (for Qwen2.5 attention weights)
// =============================================================================

/// Q5_0 GEMV kernel - handles Qwen 0.5B and similar models
///
/// Q5_0 format (per block of 32 elements):
/// - d: fp16 scale (2 bytes, offset 0)
/// - qh: u32 with 32 high bits (4 bytes, offset 2)
/// - qs: packed 4-bit nibbles (16 bytes, offset 6)
///
/// Dequantization: val = d * ((nibble | (high_bit << 4)) - 16)
#[derive(Debug, Clone)]
pub struct Q5_0GemvKernel {
    /// K dimension (input dimension)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl Q5_0GemvKernel {
    /// Create a new Q5_0 GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Get number of blocks per row (ceiling division)
    #[must_use]
    pub const fn num_blocks_per_row(&self) -> u32 {
        (self.k + Q5_0_BLOCK_SIZE - 1) / Q5_0_BLOCK_SIZE
    }
}

impl Kernel for Q5_0GemvKernel {
    fn name(&self) -> &str {
        "q5_0_gemv_warp_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("q5_0_gemv_warp_reduce")
            .param(PtxType::U64, "y_ptr") // Output vector (N)
            .param(PtxType::U64, "w_ptr") // Q5_0 weights (N x K/32 blocks)
            .param(PtxType::U64, "x_ptr") // Input vector (K)
            .param(PtxType::U32, "k_dim") // K dimension
            .param(PtxType::U32, "n_dim") // N dimension
            .build(|ctx| {
                // Block = 32 threads (one warp), grid = N blocks
                // Each block computes one output element y[block_id]

                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                // Bounds check
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                // Number of blocks per row: ceil(K / 32)
                let k_rounded = ctx.add_u32(k_dim, Q5_0_BLOCK_SIZE - 1);
                let num_blocks = ctx.div_u32(k_rounded, Q5_0_BLOCK_SIZE);

                // Row base address: w_ptr + block_id * num_blocks * 22
                let block_bytes = ctx.mov_u32_imm(Q5_0_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_blocks, block_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Loop over blocks (each thread handles one value per block)
                let blk_idx = ctx.mov_u32_imm(0);

                ctx.label("blk_loop");
                let blk_done = ctx.setp_ge_u32(blk_idx, num_blocks);
                ctx.branch_if(blk_done, "blk_loop_end");

                // Block address = row_base + blk_idx * 22
                let blk_offset = ctx.mul_wide_u32(blk_idx, Q5_0_BLOCK_BYTES);
                let blk_addr = ctx.add_u64(row_base, blk_offset);

                // Load scale d (fp16 at offset 0)
                let d_f16 = ctx.ld_global_f16(blk_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Load qh (u32 at offset 2) - contains high bits for all 32 values
                // PAR-061-FIX: Use byte loads to avoid misaligned u32 access
                // Q5_0 blocks are 22 bytes, so offset 2 is not guaranteed 4-byte aligned
                let two_64 = ctx.mov_u64_imm(2);
                let qh_addr = ctx.add_u64(blk_addr, two_64);
                let qh_b0 = ctx.ld_global_u8(qh_addr);
                let three_64 = ctx.mov_u64_imm(3);
                let qh_addr1 = ctx.add_u64(blk_addr, three_64);
                let qh_b1 = ctx.ld_global_u8(qh_addr1);
                let four_64 = ctx.mov_u64_imm(4);
                let qh_addr2 = ctx.add_u64(blk_addr, four_64);
                let qh_b2 = ctx.ld_global_u8(qh_addr2);
                let five_64 = ctx.mov_u64_imm(5);
                let qh_addr3 = ctx.add_u64(blk_addr, five_64);
                let qh_b3 = ctx.ld_global_u8(qh_addr3);
                // Combine bytes: qh = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                let qh_b0_u32 = ctx.cvt_u32_u8(qh_b0);
                let qh_b1_u32 = ctx.cvt_u32_u8(qh_b1);
                let qh_b2_u32 = ctx.cvt_u32_u8(qh_b2);
                let qh_b3_u32 = ctx.cvt_u32_u8(qh_b3);
                let qh_b1_shifted = ctx.shl_u32_imm(qh_b1_u32, 8);
                let qh_b2_shifted = ctx.shl_u32_imm(qh_b2_u32, 16);
                let qh_b3_shifted = ctx.shl_u32_imm(qh_b3_u32, 24);
                let qh_01 = ctx.or_u32(qh_b0_u32, qh_b1_shifted);
                let qh_012 = ctx.or_u32(qh_01, qh_b2_shifted);
                let qh = ctx.or_u32(qh_012, qh_b3_shifted);

                // Extract high bit for this thread: (qh >> thread_id) & 1
                let high_bit = ctx.shr_u32(qh, thread_id);
                let one_u32 = ctx.mov_u32_imm(1);
                let high_bit_masked = ctx.and_u32(high_bit, one_u32);

                // Load nibble for this thread from qs (offset 6)
                // qs layout: 32 4-bit values packed into 16 bytes
                // Nibble index = thread_id, byte index = thread_id / 2
                // Low/high nibble = thread_id % 2
                let six_64 = ctx.mov_u64_imm(6);
                let qs_base = ctx.add_u64(blk_addr, six_64);

                // byte_idx = thread_id / 2
                let byte_idx = ctx.div_u32(thread_id, 2);
                let byte_idx_64 = ctx.cvt_u64_u32(byte_idx);
                let qs_addr = ctx.add_u64(qs_base, byte_idx_64);

                // Load the byte containing our nibble
                let qs_byte = ctx.ld_global_u8(qs_addr);
                let qs_byte_u32 = ctx.cvt_u32_u8(qs_byte);

                // Extract nibble: if thread_id is odd, use high nibble (>> 4)
                // nibble_select = (thread_id % 2) * 4 = (thread_id & 1) << 2
                let nibble_select = ctx.and_u32(thread_id, one_u32);
                let shift_amount = ctx.mul_u32(nibble_select, 4);
                let shifted = ctx.shr_u32(qs_byte_u32, shift_amount);
                let fifteen_u32 = ctx.mov_u32_imm(15);
                let nibble = ctx.and_u32(shifted, fifteen_u32);

                // Combine nibble with high bit: q = nibble | (high_bit << 4)
                let high_shifted = ctx.shl_u32_imm(high_bit_masked, 4);
                let q_5bit = ctx.or_u32(nibble, high_shifted);

                // Center: q_centered = q - 16 (result may be negative, -16 to +15)
                let sixteen_u32 = ctx.mov_u32_imm(16);
                let q_centered = ctx.sub_u32_reg(q_5bit, sixteen_u32);

                // Convert to float and dequantize
                // cvt_f32_s32 interprets the bits as signed, so negative values work correctly
                let q_f32 = ctx.cvt_f32_s32(q_centered);
                let dequant = ctx.mul_f32(d, q_f32);

                // Load activation x[blk_idx * 32 + thread_id]
                let blk_k_base = ctx.mul_u32(blk_idx, Q5_0_BLOCK_SIZE);
                let x_idx = ctx.add_u32_reg(blk_k_base, thread_id);

                // Bounds check for last block (K may not be multiple of 32)
                let x_oob = ctx.setp_ge_u32(x_idx, k_dim);
                ctx.branch_if(x_oob, "skip_mul");

                let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                let x_bytes = ctx.mul_u64(x_idx_64, 4);
                let x_addr = ctx.add_u64(x_ptr, x_bytes);
                let x_val = ctx.ld_global_f32(x_addr);

                ctx.fma_f32_inplace(acc, x_val, dequant);

                ctx.label("skip_mul");
                ctx.add_u32_inplace(blk_idx, 1);
                ctx.branch("blk_loop");

                ctx.label("blk_loop_end");

                // Warp reduce
                let tmp16 = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp16);
                let tmp8 = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp8);
                let tmp4 = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp4);
                let tmp2 = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp2);
                let tmp1 = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, tmp1);

                // Thread 0 writes result
                let is_thread0 = ctx.setp_lt_u32(thread_id, one_u32);
                ctx.branch_if_not(is_thread0, "exit");

                let y_offset = ctx.mul_wide_u32(block_id, 4);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
