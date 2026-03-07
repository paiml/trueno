//! Half-warp DP4A Q6_K GEMV kernel (PMAT-030)
//!
//! 16 threads per super-block (half-warp), matching the Q4K HW DP4A design.
//! Each thread handles 16 Q6K values — exactly one scale's worth — so each thread
//! loads its own scale directly. Eliminates ALL shfl broadcasts and binary tree
//! selection (126 instructions/SB → ~7 instructions/SB for scales).
//!
//! # Provable Contracts
//!
//! - **C1 (Thread mapping)**: 16 threads per SB via `half_lane = lane_id & 15`.
//!   Each warp processes 2 SBs independently (lanes 0-15 and 16-31).
//!
//! - **C2 (Value coverage)**: 16 threads × 16 values = 256 = Q6K_SUPER_BLOCK_SIZE.
//!   Each thread processes 4 dp4a iterations of 4 values each.
//!
//! - **C3 (Scale efficiency)**: 1 byte load + 6 instructions per thread per SB
//!   (vs 126 instructions for 16 shfl + sign-extension + binary tree in Dp4a kernel).
//!
//! - **C4 (Reduction correctness)**: Half-warp reduction via shfl_down with
//!   deltas 8,4,2,1. Same algebraic proof as HW DP4A Q4K (GH-176).

use crate::kernels::quantize::{Kernel, Q6K_SUPER_BLOCK_BYTES, Q6K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Half-warp DP4A Q6_K GEMV kernel.
pub struct HalfWarpDp4aQ6KGemvKernel {
    /// Number of elements per row (K dimension).
    pub k: u32,
    /// Number of rows (N dimension).
    pub n: u32,
    /// Number of warps per CTA.
    pub num_warps: u32,
}

impl HalfWarpDp4aQ6KGemvKernel {
    /// Create a new HW DP4A Q6K GEMV kernel with default 3 warps per CTA.
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n, num_warps: 3 }
    }
}

impl Kernel for HalfWarpDp4aQ6KGemvKernel {
    fn name(&self) -> &str {
        "hw_dp4a_q6k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let num_half_warps = num_warps * 2;
        let smem_size = (num_half_warps * 4) as usize;

        PtxKernel::new("hw_dp4a_q6k_gemv")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U64, "q8_ptr")
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .shared_memory(smem_size)
            .max_regs(255)
            .build(move |ctx| {
                // ===== Thread identity =====
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);
                let grid_dim = ctx.special_reg(PtxReg::NctaIdX);

                // ===== Parameters =====
                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let q8_ptr = ctx.load_param_u64("q8_ptr");

                let k_rounded = ctx.add_u32(k_dim, Q6K_SUPER_BLOCK_SIZE - 1);
                let num_sb = ctx.div_u32(k_rounded, Q6K_SUPER_BLOCK_SIZE);
                let sb_bytes_reg = ctx.mov_u32_imm(Q6K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_sb, sb_bytes_reg);

                // ===== C1: Half-warp thread mapping =====
                let half_lane = ctx.and_u32_imm(lane_id, 15);
                let half_warp_in_warp = ctx.shr_u32_imm(lane_id, 4);
                let warp_x2 = ctx.shl_u32_imm(warp_id, 1);
                let half_warp_id = ctx.add_u32_reg(warp_x2, half_warp_in_warp);
                let num_hw = ctx.mov_u32_imm(num_half_warps);

                // ===== C2: Per-thread data mapping =====
                // half_lane 0-15 maps to Q6K values [half_lane*16 .. half_lane*16+15]
                //
                // Derived addressing (see dp4a.rs for Q6K layout):
                //   n_idx = half_lane / 8          (0 or 1: which 128-element half)
                //   q_path = (half_lane % 8) / 2   (0-3: q1/q2/q3/q4)
                //   l_half = half_lane % 2          (0 or 1: first or second 16 in group)
                //
                //   ql_base = n_idx*64 + (q_path & 1)*32 + l_half*16
                //   nibble_shift = (q_path / 2) * 4  (0 or 4)
                //   qh_base = 128 + n_idx*32 + l_half*16
                //   qh_shift = q_path * 2             (0, 2, 4, or 6)
                //   scale_idx = half_lane             (direct!)
                //   q8_block = half_lane / 2          (which Q8_1 block within SB)
                //   q8_sub_offset = (half_lane % 2) * 16

                let c1 = ctx.mov_u32_imm(1);
                let c2 = ctx.mov_u32_imm(2);
                let c3 = ctx.mov_u32_imm(3);
                let c4 = ctx.mov_u32_imm(4);
                let c7 = ctx.mov_u32_imm(7);
                let _c8 = ctx.mov_u32_imm(8);
                let c16 = ctx.mov_u32_imm(16);
                let c32 = ctx.mov_u32_imm(32);
                let c64 = ctx.mov_u32_imm(64);

                // n_idx = half_lane >> 3
                let n_idx = ctx.shr_u32(half_lane, c3);
                // q_path = (half_lane & 7) >> 1
                let hl_mod8 = ctx.and_u32(half_lane, c7);
                let q_path = ctx.shr_u32(hl_mod8, c1);
                // l_half = half_lane & 1
                let l_half = ctx.and_u32(half_lane, c1);

                // ql_base = n_idx*64 + (q_path & 1)*32 + l_half*16
                let n_x64 = ctx.mul_u32_reg(n_idx, c64);
                let qp_low = ctx.and_u32(q_path, c1);
                let qp_x32 = ctx.mul_u32_reg(qp_low, c32);
                let lh_x16 = ctx.mul_u32_reg(l_half, c16);
                let ql_base = ctx.add_u32_reg(n_x64, qp_x32);
                let ql_base = ctx.add_u32_reg(ql_base, lh_x16);
                let ql_base_64 = ctx.cvt_u64_u32(ql_base);

                // nibble_shift = (q_path >> 1) << 2  (0 or 4)
                let qp_div2 = ctx.shr_u32(q_path, c1);
                let nibble_shift = ctx.shl_u32(qp_div2, c2);

                // qh_base = 128 + n_idx*32 + l_half*16
                let n_x32 = ctx.mul_u32_reg(n_idx, c32);
                let qh_base_off = ctx.add_u32(n_x32, 128);
                let qh_base = ctx.add_u32_reg(qh_base_off, lh_x16);
                let qh_base_64 = ctx.cvt_u64_u32(qh_base);

                // qh_shift = q_path * 2
                let qh_shift = ctx.shl_u32(q_path, c1);

                // Q8 block within SB = half_lane / 2 (8 Q8 blocks per SB)
                let q8_block_in_sb = ctx.shr_u32(half_lane, c1);
                // Q8 sub-offset within block = (half_lane & 1) * 16
                let q8_sub = ctx.mul_u32_reg(l_half, c16);

                // Pre-computed Q8 block byte offset (block * 36)
                let c36 = ctx.mov_u32_imm(36);
                let q8_blk_bytes = ctx.mul_u32_reg(q8_block_in_sb, c36);
                let q8_blk_bytes_64 = ctx.cvt_u64_u32(q8_blk_bytes);
                let q8_sub_64 = ctx.cvt_u64_u32(q8_sub);

                // DP4A constants
                let mask_0f = ctx.mov_u32_imm(0x0F0F_0F0F);
                let mask_03 = ctx.mov_u32_imm(0x0303_0303);
                let ones_packed = ctx.mov_u32_imm(0x0101_0101);
                let c5 = ctx.mov_u32_imm(5);

                // Hoisted 64-bit constants (reserved for multi-word load patterns)
                let _c4_64 = ctx.mov_u64_imm(4);
                let _c8_64 = ctx.mov_u64_imm(8);
                let _c12_64 = ctx.mov_u64_imm(12);
                let c32_64 = ctx.mov_u64_imm(32);
                let c192_64 = ctx.mov_u64_imm(192);
                let c208_64 = ctx.mov_u64_imm(208);
                let c256_f32 = ctx.mov_f32_imm(256.0);

                // Q8 SB stride: 8 blocks × 36 bytes = 288
                let c288 = ctx.mov_u32_imm(288);

                let zero = ctx.mov_u32_imm(0);

                // ===== Grid-stride row loop =====
                let row_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(row_idx, block_id);

                ctx.label("hwq6_row_loop");
                let row_oob = ctx.setp_ge_u32(row_idx, n_dim);
                ctx.branch_if(row_oob, "hwq6_exit");

                let row_off = ctx.mul_wide_u32_reg(row_idx, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_off);

                let acc = ctx.mov_f32_imm(0.0);

                // SB loop: stride by num_half_warps
                let sb_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(sb_idx, half_warp_id);

                ctx.label("hwq6_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_sb);
                ctx.branch_if(sb_done, "hwq6_sb_end");

                // Super-block base address
                let sb_off = ctx.mul_wide_u32(sb_idx, Q6K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                // ===== C3: Direct scale loading (1 byte per thread!) =====
                // scale_idx = half_lane (each thread owns its scale)
                let sc_addr = ctx.add_u64(sb_addr, c192_64);
                let sc_lane_off = ctx.cvt_u64_u32(half_lane);
                let sc_my_addr = ctx.add_u64(sc_addr, sc_lane_off);
                let sc_u8 = ctx.ld_global_u8(sc_my_addr);
                let sc_u32 = ctx.cvt_u32_u8(sc_u8);

                // Signed i8 → f32: if bit 7 set, subtract 256
                let sign_bit = ctx.shr_u32(sc_u32, c7);
                let raw_f32 = ctx.cvt_f32_u32(sc_u32);
                let sign_f32 = ctx.cvt_f32_u32(sign_bit);
                let correction = ctx.mul_f32(sign_f32, c256_f32);
                let scale_f32 = ctx.sub_f32(raw_f32, correction);

                // Load d (f16 at offset 208)
                let d_addr = ctx.add_u64(sb_addr, c208_64);
                let d_f16 = ctx.ld_global_f16(d_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // d * scale (precomputed for all 4 iterations)
                let d_scale = ctx.mul_f32(d, scale_f32);

                // Q8 base for this SB
                let q8_sb_off = ctx.mul_wide_u32_reg(sb_idx, c288);
                let q8_sb_base = ctx.add_u64(q8_ptr, q8_sb_off);
                let q8_blk_addr = ctx.add_u64(q8_sb_base, q8_blk_bytes_64);

                // Q8 d scale (f16 at block+32, same for all 4 iterations)
                let q8_d_addr = ctx.add_u64(q8_blk_addr, c32_64);
                let q8_d_f16 = ctx.ld_global_f16(q8_d_addr);
                let q8_d = ctx.cvt_f32_f16(q8_d_f16);

                // Combined scale = d * scale * q8_d
                let combined_scale = ctx.mul_f32(d_scale, q8_d);

                // Q8 qs base for this thread
                let q8_qs_base = ctx.add_u64(q8_blk_addr, q8_sub_64);

                // ===== 4 dp4a iterations (4 values each = 16 total) =====
                // Accumulate integer: int_acc = Σ(dot - 32*sum)
                let int_acc = ctx.mov_u32_imm(0);

                for i in 0..4u32 {
                    let i_x4 = ctx.mov_u64_imm(u64::from(i * 4));

                    // ql load: sb_addr + ql_base + i*4
                    let ql_iter_addr = ctx.add_u64(sb_addr, ql_base_64);
                    let ql_addr = ctx.add_u64(ql_iter_addr, i_x4);
                    let ql_raw = ctx.ld_global_u32_unaligned(ql_addr);

                    // Extract nibbles
                    let ql_shifted = ctx.shr_u32(ql_raw, nibble_shift);
                    let ql_nibs = ctx.and_u32(ql_shifted, mask_0f);

                    // qh load: sb_addr + qh_base + i*4
                    let qh_iter_addr = ctx.add_u64(sb_addr, qh_base_64);
                    let qh_addr = ctx.add_u64(qh_iter_addr, i_x4);
                    let qh_raw = ctx.ld_global_u32_unaligned(qh_addr);

                    // Extract 2-bit pairs
                    let qh_shifted = ctx.shr_u32(qh_raw, qh_shift);
                    let qh_2bits = ctx.and_u32(qh_shifted, mask_03);

                    // Combine: q6k_unsigned = ql_nibs | (qh_2bits << 4)
                    let qh_up = ctx.shl_u32(qh_2bits, c4);
                    let combined = ctx.or_u32(ql_nibs, qh_up);

                    // Q8 qs load
                    let q8_addr = ctx.add_u64(q8_qs_base, i_x4);
                    let q8_int32 = ctx.ld_global_u32(q8_addr);

                    // dp4a: unsigned Q6K × signed Q8
                    ctx.dp4a_u32_s32_inplace(int_acc, combined, q8_int32);

                    // dp4a: sum of Q8 bytes (for -32 bias)
                    let sum_iter = ctx.mov_u32_imm(0);
                    ctx.dp4a_u32_s32_inplace(sum_iter, ones_packed, q8_int32);

                    // Subtract bias: int_acc -= 32 * sum = sum << 5
                    let sum_x32 = ctx.shl_u32(sum_iter, c5);
                    let int_acc_new = ctx.sub_u32(int_acc, sum_x32);
                    ctx.mov_u32_reg(int_acc, int_acc_new);
                }

                // Convert accumulated integer to f32 and scale
                let int_f32 = ctx.cvt_f32_s32(int_acc);
                ctx.fma_f32_inplace(acc, combined_scale, int_f32);

                // Stride by num_half_warps
                ctx.add_u32_reg_inplace(sb_idx, num_hw);
                ctx.branch("hwq6_sb_loop");

                ctx.label("hwq6_sb_end");

                // ===== C4: Half-warp reduction via shfl_down =====
                let t = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t);
                let t = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t);
                let t = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t);
                let t = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t);

                // Half-warp lane 0 stores to shared memory
                let is_hl0 = ctx.setp_eq_u32(half_lane, zero);
                ctx.branch_if_not(is_hl0, "hwq6_skip_sm");

                let sm_off = ctx.shl_u32_imm(half_warp_id, 2);
                let sm_addr = ctx.cvt_u64_u32(sm_off);
                ctx.st_shared_f32(sm_addr, acc);

                ctx.label("hwq6_skip_sm");
                ctx.bar_sync(0);

                // Thread 0 reduces all half-warps and stores
                let is_t0 = ctx.setp_eq_u32(thread_id, zero);
                ctx.branch_if_not(is_t0, "hwq6_skip_store");

                let result = ctx.mov_f32_imm(0.0);
                for hw in 0..num_half_warps {
                    let off = ctx.mov_u64_imm(u64::from(hw * 4));
                    let val = ctx.ld_shared_f32(off);
                    ctx.add_f32_inplace(result, val);
                }

                let y_off = ctx.mul_wide_u32(row_idx, 4);
                let y_addr = ctx.add_u64(y_ptr, y_off);
                ctx.st_global_f32(y_addr, result);

                ctx.label("hwq6_skip_store");

                // Next row (grid-stride)
                ctx.add_u32_reg_inplace(row_idx, grid_dim);
                ctx.bar_sync(0);
                ctx.branch("hwq6_row_loop");

                ctx.label("hwq6_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_emits_valid() {
        let k = HalfWarpDp4aQ6KGemvKernel::new(1536, 1536);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("hw_dp4a_q6k_gemv"), "kernel name");
        assert!(ptx.contains("dp4a.u32.s32"), "DP4A instruction");
        assert!(ptx.contains("and.b32"), "half_lane masking");
    }

    #[test]
    fn test_ptx_lm_head() {
        let k = HalfWarpDp4aQ6KGemvKernel::new(1536, 151936);
        let ptx = k.emit_ptx();
        assert!(ptx.contains("hw_dp4a_q6k_gemv"));
    }

    #[test]
    fn test_value_coverage() {
        // 16 threads × 16 values/thread = 256 = Q6K_SUPER_BLOCK_SIZE
        assert_eq!(16 * 16, Q6K_SUPER_BLOCK_SIZE as usize);
    }

    #[test]
    fn test_scale_mapping() {
        // Verify that half_lane i maps to scale[i] covering values [i*16..(i+1)*16-1]
        for half_lane in 0..16u32 {
            let n_idx = half_lane / 8;
            let q_path = (half_lane % 8) / 2;
            let l_half = half_lane % 2;

            // The 16 values this thread handles start at position:
            //   For n_idx=0: q_path*32 + l_half*16 (within first 128)
            //   For n_idx=1: 128 + q_path*32 + l_half*16
            let start_pos = n_idx * 128 + q_path * 32 + l_half * 16;

            // Scale index for position start_pos should be start_pos / 16 = half_lane
            assert_eq!(
                start_pos / 16,
                half_lane,
                "half_lane {} maps to values starting at {}, scale should be {}",
                half_lane,
                start_pos,
                half_lane
            );
        }
    }

    #[test]
    fn test_addressing_derivation() {
        // Verify ql_base, qh_base formulas against known good values
        for half_lane in 0..16u32 {
            let n_idx = half_lane / 8;
            let q_path = (half_lane % 8) / 2;
            let l_half = half_lane % 2;

            let ql_base = n_idx * 64 + (q_path & 1) * 32 + l_half * 16;
            let nibble_shift = (q_path / 2) * 4;
            let qh_base = 128 + n_idx * 32 + l_half * 16;
            let qh_shift = q_path * 2;

            // Cross-check: decode value at position half_lane*16 using GGML formula
            let pos = half_lane * 16;
            let n = if pos < 128 { 0 } else { 128 };
            let pos_in_half = pos - n;
            let l = pos_in_half % 32;
            let q_idx = pos_in_half / 32; // which of q1/q2/q3/q4

            // Expected ql index for first value
            let expected_ql = n / 2 + (q_idx & 1) * 32 + l;
            assert_eq!(
                ql_base, expected_ql,
                "half_lane {}: ql_base={} expected={}",
                half_lane, ql_base, expected_ql
            );

            // Expected nibble: q1/q2 use low nibble (shift=0), q3/q4 use high (shift=4)
            let expected_nibble = if q_idx < 2 { 0 } else { 4 };
            assert_eq!(
                nibble_shift, expected_nibble,
                "half_lane {}: nibble_shift={} expected={}",
                half_lane, nibble_shift, expected_nibble
            );

            // Expected qh index
            let expected_qh = 128 + n / 4 + l;
            assert_eq!(
                qh_base, expected_qh,
                "half_lane {}: qh_base={} expected={}",
                half_lane, qh_base, expected_qh
            );

            // Expected qh shift
            let expected_qh_shift = q_idx * 2;
            assert_eq!(
                qh_shift, expected_qh_shift,
                "half_lane {}: qh_shift={} expected={}",
                half_lane, qh_shift, expected_qh_shift
            );
        }
    }

    #[test]
    fn dump_ptx() {
        let k = HalfWarpDp4aQ6KGemvKernel::new(1536, 256);
        let ptx = k.emit_ptx();
        eprintln!("{ptx}");
    }
}
