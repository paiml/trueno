//! Q6_K GEMV Kernels
//!
//! Implements Q6_K quantized GEMV operations for decode throughput.
//!
//! ## Q6_K Layout (210 bytes per 256 values)
//!
//! - ql[128]: bytes 0-127, low 4-bits packed 2 per byte
//! - qh[64]: bytes 128-191, high 2-bits packed 4 per byte
//! - scales[16]: bytes 192-207, signed i8 per 16-element sub-block
//! - d: bytes 208-209, f16 scale factor
//!
//! ## Kernels
//!
//! - [`Q6KGemvKernel`]: Basic Q6_K GEMV with warp shuffle reduction
//! - [`CoalescedQ6KGemvKernel`]: Optimized with vectorized scale loading (PAR-066)
//! - [`BatchedQ6KGemvKernel`]: Batched version for M>1 processing (PAR-130)

use super::{Kernel, Q6K_SUPER_BLOCK_BYTES, Q6K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

// =============================================================================
// Q6_K FUSED GEMV KERNEL (PAR-003)
// =============================================================================

/// Q6_K quantized GEMV kernel for M=1 decode throughput
#[derive(Debug, Clone)]
pub struct Q6KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl Q6KGemvKernel {
    /// Create a new Q6_K GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }
}

impl Kernel for Q6KGemvKernel {
    fn name(&self) -> &str {
        "q6k_gemv_warp_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Q6_K super-block layout (210 bytes for 256 values):
        // - ql[128]: bytes 0-127, low 4-bits packed 2 per byte
        // - qh[64]: bytes 128-191, high 2-bits packed 4 per byte
        // - scales[16]: bytes 192-207, signed i8 per 16-element sub-block
        // - d: bytes 208-209, f16 scale factor
        //
        // Q6_K dequant formula (from llama.cpp):
        // For 256 values, processed in two 128-value halves (n=0, n=128):
        //   For each half, 4 groups of 32 values at positions l, l+32, l+64, l+96
        //   q1: ql[l] low nibble + qh[l] bits 0-1, shifted left 4
        //   q2: ql[l+32] low nibble + qh[l] bits 2-3, shifted left 4
        //   q3: ql[l] high nibble + qh[l] bits 4-5, shifted left 4
        //   q4: ql[l+32] high nibble + qh[l] bits 6-7, shifted left 4
        //   quant = q_combined - 32 (signed range -32 to +31)
        //   scale = scales[8*half + l/16 + 2*group] (signed i8)
        //   dequant = d * scale * quant
        PtxKernel::new("q6k_gemv_warp_reduce")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .build(|ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let acc = ctx.mov_f32_imm(0.0);
                // Ceiling division: (k + 255) / 256 for GGUF super-block count
                let k_rounded = ctx.add_u32(k_dim, Q6K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q6K_SUPER_BLOCK_SIZE);

                let sb_bytes = ctx.mov_u32_imm(Q6K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_end");

                let sb_offset = ctx.mul_wide_u32(sb_idx, Q6K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d (f16 at offset 208)
                let d_offset = ctx.mov_u64_imm(208);
                let d_addr = ctx.add_u64(sb_addr, d_offset);
                let d_f16 = ctx.ld_global_f16(d_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Each thread handles 8 values at offsets 0, 32, 64, 96, 128, 160, 192, 224
                let thread_partial = ctx.mov_f32_imm(0.0);

                // Process each of 8 values per thread
                // For val_idx = thread_id + offset (offset in [0, 32, 64, 96, 128, 160, 192, 224]):
                //   n_idx = val_idx / 128 (0 or 1, which 128-block half)
                //   pos = val_idx % 128 (position within 128-block)
                //   group = pos / 32 (0, 1, 2, or 3)
                //   l = pos % 32 (0-31)
                //   is = l / 16 (0 or 1)
                //
                //   scale_idx = 8 * n_idx + is + 2 * group
                //   ql_byte_offset = 64 * n_idx + l + (32 if group in [1, 3] else 0)
                //   ql_use_high_nibble = (group >= 2)
                //   qh_byte_offset = 32 * n_idx + l
                //   qh_bit_shift = 2 * group

                for offset in [0u32, 32, 64, 96, 128, 160, 192, 224] {
                    let offset_reg = ctx.mov_u32_imm(offset);
                    let val_idx = ctx.add_u32_reg(thread_id, offset_reg);

                    // n_idx = val_idx / 128
                    let n_idx = ctx.div_u32(val_idx, 128);
                    // pos = val_idx % 128
                    let pos = ctx.rem_u32(val_idx, 128);
                    // group = pos / 32
                    let group = ctx.div_u32(pos, 32);
                    // l = pos % 32
                    let l = ctx.rem_u32(pos, 32);
                    // is = l / 16
                    let is = ctx.div_u32(l, 16);

                    // scale_idx = 8 * n_idx + is + 2 * group
                    let eight = ctx.mov_u32_imm(8);
                    let two = ctx.mov_u32_imm(2);
                    let n_idx_x8 = ctx.mul_u32_reg(n_idx, eight);
                    let group_x2 = ctx.mul_u32_reg(group, two);
                    let scale_idx_temp = ctx.add_u32_reg(n_idx_x8, is);
                    let scale_idx = ctx.add_u32_reg(scale_idx_temp, group_x2);

                    // Load scale (signed i8 at offset 192 + scale_idx)
                    let scales_offset = ctx.mov_u64_imm(192);
                    let scales_base = ctx.add_u64(sb_addr, scales_offset);
                    let scale_idx_64 = ctx.cvt_u64_u32(scale_idx);
                    let scale_addr = ctx.add_u64(scales_base, scale_idx_64);
                    let scale_u8 = ctx.ld_global_u8(scale_addr);
                    // Convert u8 to signed i8 then to f32
                    // i8 is stored as u8, reinterpret: if >= 128, subtract 256
                    // Using: scale_f32 = (scale_u8 as f32) - 256.0 * (scale_u8 >> 7)
                    let scale_u32 = ctx.cvt_u32_u8(scale_u8);
                    let seven = ctx.mov_u32_imm(7);
                    let sign_bit = ctx.shr_u32(scale_u32, seven); // 0 or 1
                    let scale_u32_f32 = ctx.cvt_f32_u32(scale_u32);
                    let sign_bit_f32 = ctx.cvt_f32_u32(sign_bit);
                    let twofiftysix_f32 = ctx.mov_f32_imm(256.0);
                    let correction_f32 = ctx.mul_f32(sign_bit_f32, twofiftysix_f32);
                    let scale_f32 = ctx.sub_f32(scale_u32_f32, correction_f32);

                    // ql_byte_offset = 64 * n_idx + l + (32 * group_is_odd)
                    // where group_is_odd = group & 1
                    let sixty_four = ctx.mov_u32_imm(64);
                    let thirty_two = ctx.mov_u32_imm(32);
                    let one = ctx.mov_u32_imm(1);
                    let n_idx_x64 = ctx.mul_u32_reg(n_idx, sixty_four);
                    let ql_base = ctx.add_u32_reg(n_idx_x64, l);
                    let group_is_odd = ctx.and_u32(group, one);
                    let ql_offset_add = ctx.mul_u32_reg(group_is_odd, thirty_two);
                    let ql_byte_offset = ctx.add_u32_reg(ql_base, ql_offset_add);

                    // Load ql byte
                    let ql_byte_offset_64 = ctx.cvt_u64_u32(ql_byte_offset);
                    let ql_addr = ctx.add_u64(sb_addr, ql_byte_offset_64);
                    let ql_byte = ctx.ld_global_u8(ql_addr);
                    let ql_byte_32 = ctx.cvt_u32_u8(ql_byte);

                    // Extract nibble: low if group < 2, high if group >= 2
                    // nibble_shift = (group / 2) * 4 = (group >> 1) * 4
                    let group_div_2 = ctx.shr_u32(group, one);
                    let four = ctx.mov_u32_imm(4);
                    let nibble_shift = ctx.mul_u32_reg(group_div_2, four);
                    let ql_shifted = ctx.shr_u32(ql_byte_32, nibble_shift);
                    let mask_0xf = ctx.mov_u32_imm(0xF);
                    let ql_nibble = ctx.and_u32(ql_shifted, mask_0xf);

                    // qh_byte_offset = 32 * n_idx + l
                    let n_idx_x32 = ctx.mul_u32_reg(n_idx, thirty_two);
                    let qh_byte_offset = ctx.add_u32_reg(n_idx_x32, l);

                    // Load qh byte (offset 128 + qh_byte_offset)
                    let qh_base_offset = ctx.mov_u64_imm(128);
                    let qh_base = ctx.add_u64(sb_addr, qh_base_offset);
                    let qh_byte_offset_64 = ctx.cvt_u64_u32(qh_byte_offset);
                    let qh_addr = ctx.add_u64(qh_base, qh_byte_offset_64);
                    let qh_byte = ctx.ld_global_u8(qh_addr);
                    let qh_byte_32 = ctx.cvt_u32_u8(qh_byte);

                    // qh_bit_shift = 2 * group
                    let qh_shift = ctx.mul_u32_reg(group, two);
                    let qh_shifted = ctx.shr_u32(qh_byte_32, qh_shift);
                    let mask_0x3 = ctx.mov_u32_imm(0x3);
                    let qh_2bits = ctx.and_u32(qh_shifted, mask_0x3);

                    // Combine: quant = ql_nibble | (qh_2bits << 4) - 32
                    let qh_shifted_up = ctx.shl_u32(qh_2bits, four);
                    let combined = ctx.or_u32(ql_nibble, qh_shifted_up);
                    let combined_f32 = ctx.cvt_f32_u32(combined);
                    let thirty_two_f32 = ctx.mov_f32_imm(32.0);
                    let quant_signed = ctx.sub_f32(combined_f32, thirty_two_f32);

                    // Dequantize: val = d × scale × quant
                    let d_scale = ctx.mul_f32(d, scale_f32);
                    let dequant = ctx.mul_f32(d_scale, quant_signed);

                    // Load activation
                    let sb_k_base = ctx.mul_u32(sb_idx, Q6K_SUPER_BLOCK_SIZE);
                    let x_idx = ctx.add_u32_reg(sb_k_base, val_idx);
                    let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                    let x_bytes = ctx.mul_u64(x_idx_64, 4);
                    let x_addr = ctx.add_u64(x_ptr, x_bytes);
                    let x_val = ctx.ld_global_f32(x_addr);

                    ctx.fma_f32_inplace(thread_partial, x_val, dequant);
                }

                ctx.add_f32_inplace(acc, thread_partial);
                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_end");

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
// PAR-066: COALESCED Q6_K GEMV KERNEL
// =============================================================================

/// Coalesced Q6_K GEMV kernel with vectorized scale loading (PAR-066)
///
/// Five-Whys Root Cause: Q6KGemvKernel uses single-byte loads for all 16 scales,
/// causing 16 separate memory transactions per super-block. This kernel loads
/// all scales as 4 x u32 via lane 0, then broadcasts via warp shuffle.
///
/// # Memory Access Pattern
///
/// **Before (Q6KGemvKernel):** 16 × ld_global_u8 = 16 memory transactions
/// **After (Coalesced):** 4 × ld_global_u32 + warp shuffle = 4 transactions
///
/// # Q6_K Layout (210 bytes per 256 values)
///
/// - ql[128]: bytes 0-127, low 4-bits packed 2 per byte
/// - qh[64]: bytes 128-191, high 2-bits packed 4 per byte
/// - scales[16]: bytes 192-207, signed i8 per 16-element sub-block
/// - d: bytes 208-209, f16 scale factor
///
/// # Performance Target
///
/// - Qwen2 1.5B FFN down_proj uses Q6_K (bottleneck identified in PAR-065)
/// - Expected 20-30% improvement from reduced memory transactions
#[derive(Debug, Clone)]
pub struct CoalescedQ6KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl CoalescedQ6KGemvKernel {
    /// Create a new coalesced Q6_K GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Get number of super-blocks per row
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        (self.k + Q6K_SUPER_BLOCK_SIZE - 1) / Q6K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for CoalescedQ6KGemvKernel {
    fn name(&self) -> &str {
        "coalesced_q6k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("coalesced_q6k_gemv")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .build(|ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);

                // Bounds check
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                // Calculate super-blocks per row
                let k_rounded = ctx.add_u32(k_dim, Q6K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q6K_SUPER_BLOCK_SIZE);

                // Row base address
                let sb_bytes = ctx.mov_u32_imm(Q6K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_end");

                let sb_offset = ctx.mul_wide_u32(sb_idx, Q6K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d (f16 at offset 208)
                let d_offset = ctx.mov_u64_imm(208);
                let d_addr = ctx.add_u64(sb_addr, d_offset);
                let d_f16 = ctx.ld_global_f16(d_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // ========================================================
                // PAR-066 OPTIMIZATION: Byte-wise scale loading + warp shuffle
                // Q6K super-blocks are 210 bytes (NOT 4-byte aligned!)
                // So we use byte loads + warp shuffle to share scales
                // Lanes 0-15 each load one scale byte, then broadcast via shuffle
                // ========================================================
                let scales_base_offset = ctx.mov_u64_imm(192);
                let scales_base = ctx.add_u64(sb_addr, scales_base_offset);

                // Each of lanes 0-15 loads one scale byte
                // Lanes 16-31 will get their values via warp shuffle
                let lane_mod_16 = ctx.rem_u32(lane_id, 16);
                let lane_offset = ctx.cvt_u64_u32(lane_mod_16);
                let scale_addr = ctx.add_u64(scales_base, lane_offset);

                // Load scale byte for this lane (lanes 0-15) or 0 (lanes 16-31)
                let my_scale_byte = ctx.mov_u32_imm(0);
                let sixteen_const = ctx.mov_u32_imm(16);
                let is_low_lane = ctx.setp_lt_u32(lane_id, sixteen_const);
                ctx.branch_if_not(is_low_lane, "skip_scale_load");
                let scale_u8 = ctx.ld_global_u8(scale_addr);
                let scale_u32 = ctx.cvt_u32_u8(scale_u8);
                ctx.mov_u32_reg(my_scale_byte, scale_u32);
                ctx.label("skip_scale_load");

                // Broadcast all 16 scales via warp shuffle
                // Each lane gets scale[0..15] by shuffling from lanes 0..15
                let s0_u32 = ctx.shfl_idx_u32(my_scale_byte, 0, 0xFFFF_FFFF);
                let s1_u32 = ctx.shfl_idx_u32(my_scale_byte, 1, 0xFFFF_FFFF);
                let s2_u32 = ctx.shfl_idx_u32(my_scale_byte, 2, 0xFFFF_FFFF);
                let s3_u32 = ctx.shfl_idx_u32(my_scale_byte, 3, 0xFFFF_FFFF);
                let s4_u32 = ctx.shfl_idx_u32(my_scale_byte, 4, 0xFFFF_FFFF);
                let s5_u32 = ctx.shfl_idx_u32(my_scale_byte, 5, 0xFFFF_FFFF);
                let s6_u32 = ctx.shfl_idx_u32(my_scale_byte, 6, 0xFFFF_FFFF);
                let s7_u32 = ctx.shfl_idx_u32(my_scale_byte, 7, 0xFFFF_FFFF);
                let s8_u32 = ctx.shfl_idx_u32(my_scale_byte, 8, 0xFFFF_FFFF);
                let s9_u32 = ctx.shfl_idx_u32(my_scale_byte, 9, 0xFFFF_FFFF);
                let s10_u32 = ctx.shfl_idx_u32(my_scale_byte, 10, 0xFFFF_FFFF);
                let s11_u32 = ctx.shfl_idx_u32(my_scale_byte, 11, 0xFFFF_FFFF);
                let s12_u32 = ctx.shfl_idx_u32(my_scale_byte, 12, 0xFFFF_FFFF);
                let s13_u32 = ctx.shfl_idx_u32(my_scale_byte, 13, 0xFFFF_FFFF);
                let s14_u32 = ctx.shfl_idx_u32(my_scale_byte, 14, 0xFFFF_FFFF);
                let s15_u32 = ctx.shfl_idx_u32(my_scale_byte, 15, 0xFFFF_FFFF);

                // For packing into u32, create combined values (for reference only)
                let _scales_0_3_bcast = s0_u32; // placeholder for old code compatibility
                let _scales_4_7_bcast = s4_u32;
                let _scales_8_11_bcast = s8_u32;
                let _scales_12_15_bcast = s12_u32;

                // Convert individual scale bytes to signed f32
                // Scale bytes are already in s0_u32..s15_u32 from warp shuffle above
                // Convert u8 to signed i8 as f32: if >= 128, subtract 256
                let seven = ctx.mov_u32_imm(7);
                let twofiftysix_f32 = ctx.mov_f32_imm(256.0);

                // Helper: convert u8 to signed f32
                // sign = (val >> 7), correction = sign * 256, result = val - correction
                let s0_sign = ctx.shr_u32(s0_u32, seven);
                let s0_f32_raw = ctx.cvt_f32_u32(s0_u32);
                let s0_sign_f32 = ctx.cvt_f32_u32(s0_sign);
                let s0_correction = ctx.mul_f32(s0_sign_f32, twofiftysix_f32);
                let scale0 = ctx.sub_f32(s0_f32_raw, s0_correction);

                let s1_sign = ctx.shr_u32(s1_u32, seven);
                let s1_f32_raw = ctx.cvt_f32_u32(s1_u32);
                let s1_sign_f32 = ctx.cvt_f32_u32(s1_sign);
                let s1_correction = ctx.mul_f32(s1_sign_f32, twofiftysix_f32);
                let scale1 = ctx.sub_f32(s1_f32_raw, s1_correction);

                let s2_sign = ctx.shr_u32(s2_u32, seven);
                let s2_f32_raw = ctx.cvt_f32_u32(s2_u32);
                let s2_sign_f32 = ctx.cvt_f32_u32(s2_sign);
                let s2_correction = ctx.mul_f32(s2_sign_f32, twofiftysix_f32);
                let scale2 = ctx.sub_f32(s2_f32_raw, s2_correction);

                let s3_sign = ctx.shr_u32(s3_u32, seven);
                let s3_f32_raw = ctx.cvt_f32_u32(s3_u32);
                let s3_sign_f32 = ctx.cvt_f32_u32(s3_sign);
                let s3_correction = ctx.mul_f32(s3_sign_f32, twofiftysix_f32);
                let scale3 = ctx.sub_f32(s3_f32_raw, s3_correction);

                let s4_sign = ctx.shr_u32(s4_u32, seven);
                let s4_f32_raw = ctx.cvt_f32_u32(s4_u32);
                let s4_sign_f32 = ctx.cvt_f32_u32(s4_sign);
                let s4_correction = ctx.mul_f32(s4_sign_f32, twofiftysix_f32);
                let scale4 = ctx.sub_f32(s4_f32_raw, s4_correction);

                let s5_sign = ctx.shr_u32(s5_u32, seven);
                let s5_f32_raw = ctx.cvt_f32_u32(s5_u32);
                let s5_sign_f32 = ctx.cvt_f32_u32(s5_sign);
                let s5_correction = ctx.mul_f32(s5_sign_f32, twofiftysix_f32);
                let scale5 = ctx.sub_f32(s5_f32_raw, s5_correction);

                let s6_sign = ctx.shr_u32(s6_u32, seven);
                let s6_f32_raw = ctx.cvt_f32_u32(s6_u32);
                let s6_sign_f32 = ctx.cvt_f32_u32(s6_sign);
                let s6_correction = ctx.mul_f32(s6_sign_f32, twofiftysix_f32);
                let scale6 = ctx.sub_f32(s6_f32_raw, s6_correction);

                let s7_sign = ctx.shr_u32(s7_u32, seven);
                let s7_f32_raw = ctx.cvt_f32_u32(s7_u32);
                let s7_sign_f32 = ctx.cvt_f32_u32(s7_sign);
                let s7_correction = ctx.mul_f32(s7_sign_f32, twofiftysix_f32);
                let scale7 = ctx.sub_f32(s7_f32_raw, s7_correction);

                let s8_sign = ctx.shr_u32(s8_u32, seven);
                let s8_f32_raw = ctx.cvt_f32_u32(s8_u32);
                let s8_sign_f32 = ctx.cvt_f32_u32(s8_sign);
                let s8_correction = ctx.mul_f32(s8_sign_f32, twofiftysix_f32);
                let scale8 = ctx.sub_f32(s8_f32_raw, s8_correction);

                let s9_sign = ctx.shr_u32(s9_u32, seven);
                let s9_f32_raw = ctx.cvt_f32_u32(s9_u32);
                let s9_sign_f32 = ctx.cvt_f32_u32(s9_sign);
                let s9_correction = ctx.mul_f32(s9_sign_f32, twofiftysix_f32);
                let scale9 = ctx.sub_f32(s9_f32_raw, s9_correction);

                let s10_sign = ctx.shr_u32(s10_u32, seven);
                let s10_f32_raw = ctx.cvt_f32_u32(s10_u32);
                let s10_sign_f32 = ctx.cvt_f32_u32(s10_sign);
                let s10_correction = ctx.mul_f32(s10_sign_f32, twofiftysix_f32);
                let scale10 = ctx.sub_f32(s10_f32_raw, s10_correction);

                let s11_sign = ctx.shr_u32(s11_u32, seven);
                let s11_f32_raw = ctx.cvt_f32_u32(s11_u32);
                let s11_sign_f32 = ctx.cvt_f32_u32(s11_sign);
                let s11_correction = ctx.mul_f32(s11_sign_f32, twofiftysix_f32);
                let scale11 = ctx.sub_f32(s11_f32_raw, s11_correction);

                let s12_sign = ctx.shr_u32(s12_u32, seven);
                let s12_f32_raw = ctx.cvt_f32_u32(s12_u32);
                let s12_sign_f32 = ctx.cvt_f32_u32(s12_sign);
                let s12_correction = ctx.mul_f32(s12_sign_f32, twofiftysix_f32);
                let scale12 = ctx.sub_f32(s12_f32_raw, s12_correction);

                let s13_sign = ctx.shr_u32(s13_u32, seven);
                let s13_f32_raw = ctx.cvt_f32_u32(s13_u32);
                let s13_sign_f32 = ctx.cvt_f32_u32(s13_sign);
                let s13_correction = ctx.mul_f32(s13_sign_f32, twofiftysix_f32);
                let scale13 = ctx.sub_f32(s13_f32_raw, s13_correction);

                let s14_sign = ctx.shr_u32(s14_u32, seven);
                let s14_f32_raw = ctx.cvt_f32_u32(s14_u32);
                let s14_sign_f32 = ctx.cvt_f32_u32(s14_sign);
                let s14_correction = ctx.mul_f32(s14_sign_f32, twofiftysix_f32);
                let scale14 = ctx.sub_f32(s14_f32_raw, s14_correction);

                let s15_sign = ctx.shr_u32(s15_u32, seven);
                let s15_f32_raw = ctx.cvt_f32_u32(s15_u32);
                let s15_sign_f32 = ctx.cvt_f32_u32(s15_sign);
                let s15_correction = ctx.mul_f32(s15_sign_f32, twofiftysix_f32);
                let scale15 = ctx.sub_f32(s15_f32_raw, s15_correction);

                // Precompute d * scale for all 16 scales
                let ds0 = ctx.mul_f32(d, scale0);
                let ds1 = ctx.mul_f32(d, scale1);
                let ds2 = ctx.mul_f32(d, scale2);
                let ds3 = ctx.mul_f32(d, scale3);
                let ds4 = ctx.mul_f32(d, scale4);
                let ds5 = ctx.mul_f32(d, scale5);
                let ds6 = ctx.mul_f32(d, scale6);
                let ds7 = ctx.mul_f32(d, scale7);
                let ds8 = ctx.mul_f32(d, scale8);
                let ds9 = ctx.mul_f32(d, scale9);
                let ds10 = ctx.mul_f32(d, scale10);
                let ds11 = ctx.mul_f32(d, scale11);
                let ds12 = ctx.mul_f32(d, scale12);
                let ds13 = ctx.mul_f32(d, scale13);
                let ds14 = ctx.mul_f32(d, scale14);
                let ds15 = ctx.mul_f32(d, scale15);

                // Process 8 values per thread at offsets 0, 32, 64, 96, 128, 160, 192, 224
                // PAR-066 OPTIMIZATION: Scale index is deterministic per offset
                // scale_idx = 8 * n_idx + is + 2 * group
                // For lanes 0-15: is=0, so scale_idx = 8*n_idx + 2*group
                // For lanes 16-31: is=1, so scale_idx = 8*n_idx + 2*group + 1
                // This means each offset needs only 2 ds values, selected by lane_id < 16
                let thread_partial = ctx.mov_f32_imm(0.0);
                let thirty_two_f32 = ctx.mov_f32_imm(32.0);

                // Hardcoded ds pairs for each offset (determined by n_idx, group):
                // offset 0:   n=0, g=0, base=0  -> ds0 or ds1
                // offset 32:  n=0, g=1, base=2  -> ds2 or ds3
                // offset 64:  n=0, g=2, base=4  -> ds4 or ds5
                // offset 96:  n=0, g=3, base=6  -> ds6 or ds7
                // offset 128: n=1, g=0, base=8  -> ds8 or ds9
                // offset 160: n=1, g=1, base=10 -> ds10 or ds11
                // offset 192: n=1, g=2, base=12 -> ds12 or ds13
                // offset 224: n=1, g=3, base=14 -> ds14 or ds15

                // Precompute ds_selected for each offset using conditional add
                // ds_selected = is_low_lane ? ds[base] : ds[base+1]
                // Use FMA: ds[base] + (is_high_lane * (ds[base+1] - ds[base]))
                let ds_diff_0 = ctx.sub_f32(ds1, ds0);
                let ds_diff_1 = ctx.sub_f32(ds3, ds2);
                let ds_diff_2 = ctx.sub_f32(ds5, ds4);
                let ds_diff_3 = ctx.sub_f32(ds7, ds6);
                let ds_diff_4 = ctx.sub_f32(ds9, ds8);
                let ds_diff_5 = ctx.sub_f32(ds11, ds10);
                let ds_diff_6 = ctx.sub_f32(ds13, ds12);
                let ds_diff_7 = ctx.sub_f32(ds15, ds14);

                // Compute lane_is (0 for lanes 0-15, 1 for lanes 16-31)
                // div_u32 takes a constant, so use 16 directly
                let lane_is = ctx.div_u32(lane_id, 16);
                let lane_is_f32 = ctx.cvt_f32_u32(lane_is);

                // ds_selected[i] = ds[base_i] + lane_is_f32 * ds_diff_i
                let ds_sel_0 = ctx.fma_f32(lane_is_f32, ds_diff_0, ds0);
                let ds_sel_1 = ctx.fma_f32(lane_is_f32, ds_diff_1, ds2);
                let ds_sel_2 = ctx.fma_f32(lane_is_f32, ds_diff_2, ds4);
                let ds_sel_3 = ctx.fma_f32(lane_is_f32, ds_diff_3, ds6);
                let ds_sel_4 = ctx.fma_f32(lane_is_f32, ds_diff_4, ds8);
                let ds_sel_5 = ctx.fma_f32(lane_is_f32, ds_diff_5, ds10);
                let ds_sel_6 = ctx.fma_f32(lane_is_f32, ds_diff_6, ds12);
                let ds_sel_7 = ctx.fma_f32(lane_is_f32, ds_diff_7, ds14);

                // Process each of 8 offsets with hardcoded parameters
                // (offset, n_idx, group, ds_selected)
                let offset_params: [(u32, u32, u32); 8] = [
                    (0, 0, 0),
                    (32, 0, 1),
                    (64, 0, 2),
                    (96, 0, 3),
                    (128, 1, 0),
                    (160, 1, 1),
                    (192, 1, 2),
                    (224, 1, 3),
                ];

                for (i, (offset, n_idx_val, group_val)) in offset_params.iter().enumerate() {
                    let offset_reg = ctx.mov_u32_imm(*offset);
                    let val_idx = ctx.add_u32_reg(lane_id, offset_reg);

                    // Select the precomputed ds_selected for this offset
                    let ds_selected = match i {
                        0 => ds_sel_0,
                        1 => ds_sel_1,
                        2 => ds_sel_2,
                        3 => ds_sel_3,
                        4 => ds_sel_4,
                        5 => ds_sel_5,
                        6 => ds_sel_6,
                        _ => ds_sel_7,
                    };

                    // l = lane_id (since all offsets are multiples of 32)
                    let l = lane_id;

                    // n_idx and group are compile-time constants for this offset
                    let n_idx = ctx.mov_u32_imm(*n_idx_val);
                    let group = ctx.mov_u32_imm(*group_val);

                    // ql_byte_offset = 64 * n_idx + l + (32 * group_is_odd)
                    let sixty_four = ctx.mov_u32_imm(64);
                    let thirty_two = ctx.mov_u32_imm(32);
                    let one_32 = ctx.mov_u32_imm(1);
                    let n_idx_x64 = ctx.mul_u32_reg(n_idx, sixty_four);
                    let ql_base = ctx.add_u32_reg(n_idx_x64, l);
                    let group_is_odd = ctx.and_u32(group, one_32);
                    let ql_offset_add = ctx.mul_u32_reg(group_is_odd, thirty_two);
                    let ql_byte_offset = ctx.add_u32_reg(ql_base, ql_offset_add);

                    // Load ql byte
                    let ql_byte_offset_64 = ctx.cvt_u64_u32(ql_byte_offset);
                    let ql_addr = ctx.add_u64(sb_addr, ql_byte_offset_64);
                    let ql_byte = ctx.ld_global_u8(ql_addr);
                    let ql_byte_32 = ctx.cvt_u32_u8(ql_byte);

                    // Extract nibble: low if group < 2, high if group >= 2
                    let group_div_2 = ctx.shr_u32(group, one_32);
                    let four = ctx.mov_u32_imm(4);
                    let nibble_shift = ctx.mul_u32_reg(group_div_2, four);
                    let ql_shifted = ctx.shr_u32(ql_byte_32, nibble_shift);
                    let mask_0xf = ctx.mov_u32_imm(0xF);
                    let ql_nibble = ctx.and_u32(ql_shifted, mask_0xf);

                    // qh_byte_offset = 32 * n_idx + l
                    let n_idx_x32 = ctx.mul_u32_reg(n_idx, thirty_two);
                    let qh_byte_offset = ctx.add_u32_reg(n_idx_x32, l);

                    // Load qh byte (offset 128 + qh_byte_offset)
                    let qh_base_offset = ctx.mov_u64_imm(128);
                    let qh_base = ctx.add_u64(sb_addr, qh_base_offset);
                    let qh_byte_offset_64 = ctx.cvt_u64_u32(qh_byte_offset);
                    let qh_addr = ctx.add_u64(qh_base, qh_byte_offset_64);
                    let qh_byte = ctx.ld_global_u8(qh_addr);
                    let qh_byte_32 = ctx.cvt_u32_u8(qh_byte);

                    // qh_bit_shift = 2 * group
                    let two = ctx.mov_u32_imm(2);
                    let qh_shift = ctx.mul_u32_reg(group, two);
                    let qh_shifted = ctx.shr_u32(qh_byte_32, qh_shift);
                    let mask_0x3 = ctx.mov_u32_imm(0x3);
                    let qh_2bits = ctx.and_u32(qh_shifted, mask_0x3);

                    // Combine: quant = ql_nibble | (qh_2bits << 4) - 32
                    let qh_shifted_up = ctx.shl_u32(qh_2bits, four);
                    let combined = ctx.or_u32(ql_nibble, qh_shifted_up);
                    let combined_f32 = ctx.cvt_f32_u32(combined);
                    let quant_signed = ctx.sub_f32(combined_f32, thirty_two_f32);

                    // Dequantize: val = d * scale * quant = ds_selected * quant
                    let dequant = ctx.mul_f32(ds_selected, quant_signed);

                    // Load activation
                    let sb_k_base = ctx.mul_u32(sb_idx, Q6K_SUPER_BLOCK_SIZE);
                    let x_idx = ctx.add_u32_reg(sb_k_base, val_idx);
                    let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                    let x_bytes = ctx.mul_u64(x_idx_64, 4);
                    let x_addr = ctx.add_u64(x_ptr, x_bytes);
                    let x_val = ctx.ld_global_f32(x_addr);

                    ctx.fma_f32_inplace(thread_partial, x_val, dequant);
                }

                ctx.add_f32_inplace(acc, thread_partial);
                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_end");

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

                // Only lane 0 writes
                let one_u32 = ctx.mov_u32_imm(1);
                let is_thread0 = ctx.setp_lt_u32(lane_id, one_u32);
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
// BATCHED Q6_K GEMV KERNEL (PAR-130)
// =============================================================================
//
// Batched version of CoalescedQ6KGemvKernel for M>1 batch processing.
// Eliminates 896 sequential kernel launches for M=32 batch decode.
//
// Strategy:
// - One warp (32 threads) per output row
// - Each thread processes 8 elements per super-block (256/32 = 8)
// - All M batch elements processed within single kernel launch
// - Weights loaded once, reused for all M inputs (L1 cache efficient)
//
// Memory: Q6K = 210 bytes per 256 values = 0.82 bytes/value

/// Batched Q6_K GEMV kernel for batch decode throughput (PAR-130)
///
/// Processes M input vectors against the same weight matrix in one kernel launch.
/// This eliminates M-1 kernel launches per layer, critical for batched decode.
#[derive(Debug, Clone)]
pub struct BatchedQ6KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// M dimension (batch size)
    pub m: u32,
}

impl BatchedQ6KGemvKernel {
    /// Create a new batched Q6_K GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32, m: u32) -> Self {
        Self { k, n, m }
    }

    /// Get number of super-blocks per row
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        (self.k + Q6K_SUPER_BLOCK_SIZE - 1) / Q6K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for BatchedQ6KGemvKernel {
    fn name(&self) -> &str {
        "batched_q6k_gemv_warp_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        let m = self.m;
        PtxKernel::new("batched_q6k_gemv_warp_reduce")
            .param(PtxType::U64, "y_ptr") // Output matrix (M × N)
            .param(PtxType::U64, "w_ptr") // Q6_K weights (N × K/256 super-blocks)
            .param(PtxType::U64, "x_ptr") // Input matrix (M × K)
            .param(PtxType::U32, "k_dim") // K dimension
            .param(PtxType::U32, "n_dim") // N dimension
            .param(PtxType::U32, "m_dim") // M dimension (batch size)
            .build(move |ctx| {
                // Block = 32 threads (one warp), grid = N blocks
                // Each block computes one output row: y[:, block_id]

                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let lane_id = ctx.special_reg(PtxReg::TidX);

                // Bounds check
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let _m_dim = ctx.load_param_u32("m_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                // Initialize M accumulators
                let mut accs = Vec::with_capacity(m as usize);
                for _ in 0..m {
                    accs.push(ctx.mov_f32_imm(0.0));
                }

                // Calculate super-blocks per row
                let k_rounded = ctx.add_u32(k_dim, Q6K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q6K_SUPER_BLOCK_SIZE);

                // Row base address for weights
                let sb_bytes = ctx.mov_u32_imm(Q6K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_end");

                let sb_offset = ctx.mul_wide_u32(sb_idx, Q6K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d (f16 at offset 208)
                let d_offset = ctx.mov_u64_imm(208);
                let d_addr = ctx.add_u64(sb_addr, d_offset);
                let d_f16 = ctx.ld_global_f16(d_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Each thread processes 8 values (256/32)
                // Thread lane processes values: lane*8, lane*8+1, ..., lane*8+7
                let eight = ctx.mov_u32_imm(8);
                let thread_base_val = ctx.mul_u32_reg(lane_id, eight);

                // Initialize per-thread partial sums for all M batch elements
                let mut thread_partials = Vec::with_capacity(m as usize);
                for _ in 0..m {
                    thread_partials.push(ctx.mov_f32_imm(0.0));
                }

                // Process 8 values per thread
                let val_offset = ctx.mov_u32_imm(0);

                ctx.label("val_loop");
                let val_done = ctx.setp_ge_u32(val_offset, eight);
                ctx.branch_if(val_done, "val_loop_end");

                let val_idx = ctx.add_u32_reg(thread_base_val, val_offset);

                // Determine which sub-block (16 values each, 16 sub-blocks total)
                let sub_block_idx = ctx.div_u32(val_idx, 16);
                let _sub_val_idx = ctx.rem_u32(val_idx, 16);

                // Load scale for this sub-block (offset 192 + sub_block_idx)
                let scales_offset = ctx.mov_u64_imm(192);
                let scales_base = ctx.add_u64(sb_addr, scales_offset);
                let sub_block_idx_64 = ctx.cvt_u64_u32(sub_block_idx);
                let scale_addr = ctx.add_u64(scales_base, sub_block_idx_64);
                let scale_u8 = ctx.ld_global_u8(scale_addr);
                let scale_u32 = ctx.cvt_u32_u8(scale_u8);

                // Convert scale to signed: if >= 128, subtract 256
                let seven = ctx.mov_u32_imm(7);
                let scale_sign = ctx.shr_u32(scale_u32, seven);
                let twofiftysix_f32 = ctx.mov_f32_imm(256.0);
                let scale_f32_raw = ctx.cvt_f32_u32(scale_u32);
                let scale_sign_f32 = ctx.cvt_f32_u32(scale_sign);
                let scale_correction = ctx.mul_f32(scale_sign_f32, twofiftysix_f32);
                let scale_f32 = ctx.sub_f32(scale_f32_raw, scale_correction);

                // Load low 4-bit value from ql (offset 0 + val_idx / 2)
                let ql_byte_idx = ctx.div_u32(val_idx, 2);
                let ql_nibble_idx = ctx.rem_u32(val_idx, 2);
                let ql_byte_idx_64 = ctx.cvt_u64_u32(ql_byte_idx);
                let ql_addr = ctx.add_u64(sb_addr, ql_byte_idx_64);
                let ql_packed = ctx.ld_global_u8(ql_addr);
                let ql_packed_32 = ctx.cvt_u32_u8(ql_packed);
                let four = ctx.mov_u32_imm(4);
                let ql_shift = ctx.mul_u32_reg(ql_nibble_idx, four);
                let ql_shifted = ctx.shr_u32(ql_packed_32, ql_shift);
                let mask_4bit = ctx.mov_u32_imm(0xF);
                let ql = ctx.and_u32(ql_shifted, mask_4bit);

                // Load high 2-bit value from qh (offset 128 + val_idx / 4)
                let qh_offset = ctx.mov_u64_imm(128);
                let qh_base = ctx.add_u64(sb_addr, qh_offset);
                let qh_byte_idx = ctx.div_u32(val_idx, 4);
                let qh_bit_pos = ctx.rem_u32(val_idx, 4);
                let qh_byte_idx_64 = ctx.cvt_u64_u32(qh_byte_idx);
                let qh_addr = ctx.add_u64(qh_base, qh_byte_idx_64);
                let qh_packed = ctx.ld_global_u8(qh_addr);
                let qh_packed_32 = ctx.cvt_u32_u8(qh_packed);
                let two = ctx.mov_u32_imm(2);
                let qh_shift = ctx.mul_u32_reg(qh_bit_pos, two);
                let qh_shifted = ctx.shr_u32(qh_packed_32, qh_shift);
                let mask_2bit = ctx.mov_u32_imm(0x3);
                let qh = ctx.and_u32(qh_shifted, mask_2bit);

                // Combine: quant = ql + 4 * qh - 32 (6-bit signed)
                let qh_scaled = ctx.mul_u32_reg(qh, four);
                let ql_qh = ctx.add_u32_reg(ql, qh_scaled);
                let ql_qh_f32 = ctx.cvt_f32_u32(ql_qh);
                let thirty_two_f32 = ctx.mov_f32_imm(32.0);
                let quant_signed = ctx.sub_f32(ql_qh_f32, thirty_two_f32);

                // Dequantize: val = d × scale × quant
                let ds = ctx.mul_f32(d, scale_f32);
                let dequant = ctx.mul_f32(ds, quant_signed);

                // Calculate K index
                let sb_k_base = ctx.mul_u32(sb_idx, Q6K_SUPER_BLOCK_SIZE);
                let k_idx = ctx.add_u32_reg(sb_k_base, val_idx);

                // Accumulate for all M batch elements
                for batch_idx in 0..m as usize {
                    // Load x[batch_idx, k_idx]
                    let batch_offset = ctx.mov_u32_imm((batch_idx as u32) * self.k);
                    let x_offset = ctx.add_u32_reg(batch_offset, k_idx);
                    let x_offset_64 = ctx.cvt_u64_u32(x_offset);
                    let x_bytes = ctx.mul_u64(x_offset_64, 4);
                    let x_addr = ctx.add_u64(x_ptr, x_bytes);
                    let x_val = ctx.ld_global_f32(x_addr);

                    ctx.fma_f32_inplace(thread_partials[batch_idx], x_val, dequant);
                }

                ctx.add_u32_inplace(val_offset, 1);
                ctx.branch("val_loop");

                ctx.label("val_loop_end");

                // Accumulate thread partials into main accumulators
                for batch_idx in 0..m as usize {
                    ctx.add_f32_inplace(accs[batch_idx], thread_partials[batch_idx]);
                }

                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_end");

                // Warp reduce each accumulator and store
                for batch_idx in 0..m as usize {
                    let tmp16 = ctx.shfl_down_f32(accs[batch_idx], 16, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(accs[batch_idx], tmp16);
                    let tmp8 = ctx.shfl_down_f32(accs[batch_idx], 8, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(accs[batch_idx], tmp8);
                    let tmp4 = ctx.shfl_down_f32(accs[batch_idx], 4, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(accs[batch_idx], tmp4);
                    let tmp2 = ctx.shfl_down_f32(accs[batch_idx], 2, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(accs[batch_idx], tmp2);
                    let tmp1 = ctx.shfl_down_f32(accs[batch_idx], 1, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(accs[batch_idx], tmp1);
                }

                // Only lane 0 writes
                let one_u32 = ctx.mov_u32_imm(1);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one_u32);
                ctx.branch_if_not(is_lane0, "exit");

                // Write M outputs: y[batch_idx, block_id]
                for batch_idx in 0..m as usize {
                    // y[batch_idx * n + block_id]
                    let batch_row_offset = ctx.mov_u32_imm((batch_idx as u32) * self.n);
                    let y_idx = ctx.add_u32_reg(batch_row_offset, block_id);
                    let y_offset = ctx.mul_wide_u32(y_idx, 4);
                    let y_addr = ctx.add_u64(y_ptr, y_offset);
                    ctx.st_global_f32(y_addr, accs[batch_idx]);
                }

                ctx.label("exit");
                ctx.ret();
            })
    }
}
// =============================================================================
// Q6_K FUSED GEMM KERNEL (PARITY-117)
// =============================================================================
//
// Q6_K Super-block Layout (210 bytes for 256 values):
// - Offset 0-127: ql (128 bytes, 256 × 4-bit low values packed)
// - Offset 128-191: qh (64 bytes, 256 × 2-bit high values packed)
// - Offset 192-207: scales (16 bytes, 16 × 8-bit scales for 16 sub-blocks of 16)
// - Offset 208-209: d (f16 super-block scale)
//
// Dequantization: val = d × scale_b × (ql + 4*qh - 32)
// Where ql is 4-bit (0-15), qh is 2-bit (0-3), giving 6-bit signed range (-32 to 31)

/// Q6_K quantized GEMM kernel configuration
#[derive(Debug, Clone)]
pub struct Q6KKernel {
    /// Output rows (M)
    pub m: u32,
    /// Output columns (N)
    pub n: u32,
    /// Inner dimension (K) - must be divisible by 256
    pub k: u32,
    /// Tile size for output
    pub tile_size: u32,
}

impl Q6KKernel {
    /// Create a new Q6_K quantized GEMM kernel
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            tile_size: 32,
        }
    }

    /// Set output tile size
    #[must_use]
    pub const fn with_tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Get number of super-blocks per row
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        self.k / Q6K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for Q6KKernel {
    fn name(&self) -> &str {
        "q6k_gemm_ggml"
    }

    fn build_ptx(&self) -> PtxKernel {
        let tile_size = self.tile_size;
        let smem_size = Q6K_SUPER_BLOCK_SIZE * 4; // 256 f32 values

        PtxKernel::new("q6k_gemm_ggml")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_quant_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Thread and block indices
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

                // Load parameters
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_quant_ptr = ctx.load_param_u64("b_quant_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Calculate output position
                let tile_size_reg = ctx.mov_u32_imm(tile_size);
                let out_row = ctx.mul_u32_reg(ctaid_y, tile_size_reg);
                let out_col = ctx.mul_u32_reg(ctaid_x, tile_size_reg);

                let local_row = ctx.div_u32(tid, tile_size);
                let local_col = ctx.rem_u32(tid, tile_size);

                let global_row = ctx.add_u32_reg(out_row, local_row);
                let global_col = ctx.add_u32_reg(out_col, local_col);

                // Bounds check predicates
                let row_oob = ctx.setp_ge_u32(global_row, m_param);
                let col_oob = ctx.setp_ge_u32(global_col, n_param);

                // Clamp to valid range
                let one = ctx.mov_u32_imm(1);
                let m_minus_1 = ctx.sub_u32_reg(m_param, one);
                let n_minus_1 = ctx.sub_u32_reg(n_param, one);
                let clamped_row = ctx.min_u32(global_row, m_minus_1);
                let clamped_col = ctx.min_u32(global_col, n_minus_1);

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Number of super-blocks (K / 256)
                let num_k_super_blocks = ctx.div_u32(k_param, Q6K_SUPER_BLOCK_SIZE);

                // Super-block loop
                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_k_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_done");

                // Calculate super-block address
                let sb_per_row = num_k_super_blocks;
                let row_sb_offset = ctx.mul_u32_reg(clamped_col, sb_per_row);
                let total_sb_offset = ctx.add_u32_reg(row_sb_offset, sb_idx);
                let byte_offset = ctx.mul_wide_u32(total_sb_offset, Q6K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(b_quant_ptr, byte_offset);

                // Load d (f16 at offset 208)
                let d_offset = ctx.mov_u64_imm(208);
                let d_addr = ctx.add_u64(sb_addr, d_offset);
                let d_f16 = ctx.ld_global_f16(d_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Process 16 sub-blocks of 16 values each (Q6_K uses 16-element sub-blocks)
                let sub_block_idx = ctx.mov_u32_imm(0);
                let sixteen_blocks = ctx.mov_u32_imm(16);
                let sixteen_values = ctx.mov_u32_imm(16);

                ctx.label("sub_block_loop");
                let sub_done = ctx.setp_ge_u32(sub_block_idx, sixteen_blocks);
                ctx.branch_if(sub_done, "sub_block_done");

                // Load 8-bit scale for this sub-block (offset 192 + sub_block_idx)
                let scales_offset = ctx.mov_u64_imm(192);
                let scales_base = ctx.add_u64(sb_addr, scales_offset);
                let sub_block_idx_64 = ctx.cvt_u64_u32(sub_block_idx);
                let scale_addr = ctx.add_u64(scales_base, sub_block_idx_64);
                let scale_u8 = ctx.ld_global_u8(scale_addr);
                let scale_u32 = ctx.cvt_u32_u8(scale_u8);
                // Q6_K scales are signed 8-bit, center at 32 for proper range
                let scale_f32 = ctx.cvt_f32_u32(scale_u32);

                // Thread's lane within sub-block (0-15)
                let lane = ctx.rem_u32(tid, 16);

                // Global value index within super-block
                let global_val_idx = ctx.mul_u32(sub_block_idx, 16);
                let global_val_idx_full = ctx.add_u32_reg(global_val_idx, lane);

                // Load low 4-bit value from ql (offset 0 + global_val_idx / 2)
                let ql_byte_idx = ctx.div_u32(global_val_idx_full, 2);
                let ql_nibble_idx = ctx.rem_u32(global_val_idx_full, 2);
                let ql_byte_idx_64 = ctx.cvt_u64_u32(ql_byte_idx);
                let ql_addr = ctx.add_u64(sb_addr, ql_byte_idx_64);
                let ql_packed = ctx.ld_global_u8(ql_addr);
                let ql_packed_32 = ctx.cvt_u32_u8(ql_packed);
                let four = ctx.mov_u32_imm(4);
                let ql_shift = ctx.mul_u32_reg(ql_nibble_idx, four);
                let ql_shifted = ctx.shr_u32(ql_packed_32, ql_shift);
                let mask_4bit = ctx.mov_u32_imm(0xF);
                let ql = ctx.and_u32(ql_shifted, mask_4bit);

                // Load high 2-bit value from qh (offset 128 + global_val_idx / 4)
                let qh_offset = ctx.mov_u64_imm(128);
                let qh_base = ctx.add_u64(sb_addr, qh_offset);
                let qh_byte_idx = ctx.div_u32(global_val_idx_full, 4);
                let qh_bit_pos = ctx.rem_u32(global_val_idx_full, 4);
                let qh_byte_idx_64 = ctx.cvt_u64_u32(qh_byte_idx);
                let qh_addr = ctx.add_u64(qh_base, qh_byte_idx_64);
                let qh_packed = ctx.ld_global_u8(qh_addr);
                let qh_packed_32 = ctx.cvt_u32_u8(qh_packed);
                let two = ctx.mov_u32_imm(2);
                let qh_shift = ctx.mul_u32_reg(qh_bit_pos, two);
                let qh_shifted = ctx.shr_u32(qh_packed_32, qh_shift);
                let mask_2bit = ctx.mov_u32_imm(0x3);
                let qh = ctx.and_u32(qh_shifted, mask_2bit);

                // Combine: quant = ql + 4 * qh - 32 (6-bit signed: -32 to 31)
                let qh_scaled = ctx.mul_u32_reg(qh, four);
                let ql_qh = ctx.add_u32_reg(ql, qh_scaled);
                // Convert to signed by subtracting 32
                let ql_qh_f32 = ctx.cvt_f32_u32(ql_qh);
                let thirty_two_f32 = ctx.mov_f32_imm(32.0);
                let quant_signed = ctx.sub_f32(ql_qh_f32, thirty_two_f32);

                // Dequantize: val = d × scale × quant
                let d_scale = ctx.mul_f32(d, scale_f32);
                let dequant = ctx.mul_f32(d_scale, quant_signed);

                // Load activation and accumulate
                let two_fifty_six = ctx.mov_u32_imm(256);
                let sb_k_offset = ctx.mul_u32_reg(sb_idx, two_fifty_six);
                let sub_k_offset = ctx.mul_u32_reg(sub_block_idx, sixteen_values);
                let k_offset = ctx.add_u32_reg(sb_k_offset, sub_k_offset);
                let k_offset_full = ctx.add_u32_reg(k_offset, lane);

                let a_row_offset = ctx.mul_wide_u32_reg(clamped_row, k_param);
                let k_offset_64 = ctx.cvt_u64_u32(k_offset_full);
                let a_elem_offset = ctx.add_u64(a_row_offset, k_offset_64);
                let a_elem_bytes = ctx.mul_u64(a_elem_offset, 4);
                let a_addr = ctx.add_u64(a_ptr, a_elem_bytes);

                let a_val = ctx.ld_global_f32(a_addr);

                let prod = ctx.mul_f32(a_val, dequant);

                // Warp reduce (16 threads for Q6_K sub-blocks)
                let shuffled_8 = ctx.shfl_down_f32(prod, 8, 0xFFFF_FFFF);
                let prod_1 = ctx.add_f32(prod, shuffled_8);
                let shuffled_4 = ctx.shfl_down_f32(prod_1, 4, 0xFFFF_FFFF);
                let prod_2 = ctx.add_f32(prod_1, shuffled_4);
                let shuffled_2 = ctx.shfl_down_f32(prod_2, 2, 0xFFFF_FFFF);
                let prod_3 = ctx.add_f32(prod_2, shuffled_2);
                let shuffled_1 = ctx.shfl_down_f32(prod_3, 1, 0xFFFF_FFFF);
                let sub_block_sum = ctx.add_f32(prod_3, shuffled_1);

                let broadcast_sum = ctx.shfl_idx_f32(sub_block_sum, 0, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, broadcast_sum);

                ctx.add_u32_inplace(sub_block_idx, 1);
                ctx.branch("sub_block_loop");

                ctx.label("sub_block_done");

                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_done");

                // Store result
                ctx.branch_if(row_oob, "exit");
                ctx.branch_if(col_oob, "exit");

                let c_row_offset = ctx.mul_wide_u32_reg(global_row, n_param);
                let global_col_64 = ctx.cvt_u64_u32(global_col);
                let c_elem_offset = ctx.add_u64(c_row_offset, global_col_64);
                let c_elem_bytes = ctx.mul_u64(c_elem_offset, 4);
                let c_addr = ctx.add_u64(c_ptr, c_elem_bytes);

                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}
