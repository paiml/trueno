//! PMAT-295: Inline Q8 DP4A Q4K GEMV kernel.
//!
//! Fuses Q8_1 activation quantization INTO the DP4A GEMV inner loop.
//! Eliminates the separate Q8_1 quantize kernel launch per projection.
//!
//! Key difference from BatchedHwDp4aQ4KGemvKernel:
//! - Input is `f32_ptr` (FP32 activation buffer), NOT `q8_ptr`
//! - Each thread cooperatively quantizes FP32 to INT8 in-register using
//!   4-thread shuffle reduction within each bq8_group
//! - Then proceeds with standard DP4A dot product
//!
//! # Performance Model
//!
//! Per SB iteration overhead vs pre-computed Q8:
//! - 8 extra FP32 loads per thread (from L2 cache, ~100ns)
//! - 4 abs.f32 + 4 max.f32 + 2 shfl_down (4-thread reduction) = ~12 insn
//! - 8 quantize (mul+cvt+clamp) = ~24 insn
//! - 4 pack_to_u32 = ~8 insn
//! Total: ~44 extra insn per SB per thread vs ~4 insn (2 loads) = ~40 extra
//!
//! Savings: eliminates 1 kernel launch per GEMV call = ~12us.
//! At 4 Q8 launches saved per layer x 28 layers = 112 saved = 1.3ms.
//! Extra compute: 40 insn x 6 SBs x 7 proj x 28 layers / 1.9GHz = ~25us.
//! Net: ~1.3ms saved. On 13ms step = ~10% improvement.

use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory, PtxSync};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// PMAT-295: Batched DP4A Q4K GEMV with inline FP32-to-Q8 quantization.
///
/// Same structure as `BatchedHwDp4aQ4KGemvKernel` but reads FP32 input
/// directly and quantizes to INT8 in registers before DP4A.
pub struct InlineQ8Dp4aQ4KGemvKernel {
    pub k: u32,
    pub n: u32,
    pub m: u32,
    pub num_warps: u32,
}

impl InlineQ8Dp4aQ4KGemvKernel {
    pub fn new(k: u32, n: u32, m: u32) -> Self {
        Self { k, n, m, num_warps: 3 }
    }
}

impl Kernel for InlineQ8Dp4aQ4KGemvKernel {
    fn name(&self) -> &str {
        "inline_q8_dp4a_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let num_half_warps = num_warps * 2;
        let m = self.m;
        let smem_size = (num_half_warps * m * 4) as usize;

        PtxKernel::new("inline_q8_dp4a_q4k_gemv")
            .param(PtxType::U64, "y_ptr") // Output: [N * M] f32
            .param(PtxType::U64, "w_ptr") // Q4K weights: [N * ceil(K/256) * 144] bytes
            .param(PtxType::U64, "x_ptr") // FP32 input: [M * K] f32 (NOT Q8!)
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .param(PtxType::U32, "m_dim")
            .shared_memory(smem_size)
            .max_regs(255)
            .build(move |ctx| {
                // ===== Thread identity (same as HwDp4a) =====
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);
                let grid_dim = ctx.special_reg(PtxReg::NctaIdX);

                let n_dim = ctx.load_param_u32("n_dim");
                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_sb = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);
                let sb_bytes_reg = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_sb, sb_bytes_reg);

                // Half-warp mapping
                let half_lane = ctx.and_u32_imm(lane_id, 15);
                let half_warp_in_warp = ctx.shr_u32_imm(lane_id, 4);
                let warp_x2 = ctx.shl_u32_imm(warp_id, 1);
                let half_warp_id = ctx.add_u32_reg(warp_x2, half_warp_in_warp);
                let num_hw = ctx.mov_u32_imm(num_half_warps);

                // Per-thread data mapping
                let bq8_group = ctx.shr_u32_imm(half_lane, 2);
                let lane_in_group = ctx.and_u32_imm(half_lane, 3);
                let bq8_offset = ctx.shl_u32_imm(bq8_group, 1);

                // Q4K qs offset
                let t1 = ctx.shl_u32_imm(bq8_offset, 4);
                let t2 = ctx.shl_u32_imm(lane_in_group, 2);
                let q4_local = ctx.add_u32_reg(t1, t2);
                let q4_off = ctx.add_u32(q4_local, 16);
                let q4_off_64 = ctx.cvt_u64_u32(q4_off);

                // FP32 activation stride per batch element: K * 4 bytes
                let c_4_bytes = ctx.mov_u32_imm(4);
                let x_vec_stride = ctx.mul_u32_reg(k_dim, c_4_bytes);

                // FP32 activation base offsets for inline Q8:
                // K offset for this thread's values:
                //   sb_idx * 256 + bq8_offset * 32 + lane_in_group * 4
                // For QR=0 (low nibbles): above offset
                // For QR=1 (high nibbles): above offset + 128
                let c_256_u32 = ctx.mov_u32_imm(256);
                let bq8_k_base = ctx.shl_u32_imm(bq8_offset, 5); // * 32
                let lig_k_base = ctx.shl_u32_imm(lane_in_group, 2); // * 4
                let thread_k_off = ctx.add_u32_reg(bq8_k_base, lig_k_base);

                // Hoisted constants
                let c_2_64 = ctx.mov_u64_imm(2);
                let c_4_64 = ctx.mov_u64_imm(4);
                let c_8_64 = ctx.mov_u64_imm(8);
                let c_16_64 = ctx.mov_u64_imm(16);
                let c_128_u32 = ctx.mov_u32_imm(128);

                // Scale extraction invariants
                let ci_mod2 = ctx.and_u32_imm(bq8_group, 1);
                let c_16_u32 = ctx.mov_u32_imm(16);
                let byte_shift = ctx.mul_u32_reg(ci_mod2, c_16_u32);
                let c_8_u32 = ctx.mov_u32_imm(8);
                let byte_shift_hi = ctx.add_u32_reg(byte_shift, c_8_u32);
                let c_2_u32 = ctx.mov_u32_imm(2);
                let p_hi = ctx.setp_ge_u32(bq8_group, c_2_u32);

                // DP4A and Q8 constants
                let c_ones = ctx.mov_u32_imm(0x0101_0101);
                let c_mask_6bit = ctx.mov_u32_imm(0x3F3F_3F3F);
                let c_mask_4bit = ctx.mov_u32_imm(0x0F0F_0F0F);
                let c_mask_2bit = ctx.mov_u32_imm(0x0303_0303);
                let c_inv_127 = ctx.mov_f32_imm(1.0 / 127.0);
                let c_eps = ctx.mov_f32_imm(1e-10);
                let c_127_s32 = ctx.mov_s32_imm(127);
                let c_neg127_u32 = ctx.mov_u32_imm(0xFFFF_FF81);
                let c_neg127_s32 = ctx.mov_s32_from_u32(c_neg127_u32);

                // M accumulators
                let f32_zero = ctx.mov_f32_imm(0.0);
                let mut accs = Vec::with_capacity(m as usize);
                for _ in 0..m {
                    accs.push(ctx.mov_f32_imm(0.0));
                }

                // ===== Grid-stride row loop =====
                let row_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(row_idx, block_id);

                ctx.label("iq8_row_loop");
                let row_oob = ctx.setp_ge_u32(row_idx, n_dim);
                ctx.branch_if(row_oob, "iq8_exit");

                let row_off = ctx.mul_wide_u32_reg(row_idx, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_off);

                for acc in &accs {
                    ctx.mov_f32_reg(*acc, f32_zero);
                }

                // SB loop
                let sb_idx = ctx.mov_u32_imm(0);
                ctx.add_u32_reg_inplace(sb_idx, half_warp_id);

                ctx.label("iq8_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_sb);
                ctx.branch_if(sb_done, "iq8_sb_end");

                // ===== Q4K weight loading (shared across M, same as HwDp4a) =====
                let sb_off = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let dmin_addr = ctx.add_u64(sb_addr, c_2_64);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);
                let neg_dmin = ctx.neg_f32(dmin);

                let sc_base = ctx.add_u64(sb_addr, c_4_64);
                let sc03 = ctx.ld_global_u32(sc_base);
                let sc47_addr = ctx.add_u64(sc_base, c_4_64);
                let sc47 = ctx.ld_global_u32(sc47_addr);
                let sc811_addr = ctx.add_u64(sc_base, c_8_64);
                let sc811 = ctx.ld_global_u32(sc811_addr);

                // Scale extraction (same as HwDp4a)
                let sc_lo4 = ctx.and_u32(sc03, c_mask_6bit);
                let mn_lo4 = ctx.and_u32(sc47, c_mask_6bit);
                let sc_hi_low = ctx.and_u32(sc811, c_mask_4bit);
                let t = ctx.shr_u32_imm(sc03, 6);
                let t = ctx.and_u32(t, c_mask_2bit);
                let sc_hi_top = ctx.shl_u32_imm(t, 4);
                let sc_hi4 = ctx.or_u32(sc_hi_low, sc_hi_top);

                let mn_hi_raw = ctx.shr_u32_imm(sc811, 4);
                let mn_hi_low = ctx.and_u32(mn_hi_raw, c_mask_4bit);
                let t = ctx.shr_u32_imm(sc47, 6);
                let t = ctx.and_u32(t, c_mask_2bit);
                let mn_hi_top = ctx.shl_u32_imm(t, 4);
                let mn_hi4 = ctx.or_u32(mn_hi_low, mn_hi_top);

                let sc_src = ctx.selp_u32(p_hi, sc_hi4, sc_lo4);
                let mn_src = ctx.selp_u32(p_hi, mn_hi4, mn_lo4);

                let sc0 = ctx.bfe_u32_reg(sc_src, byte_shift, 8);
                let sc1 = ctx.bfe_u32_reg(sc_src, byte_shift_hi, 8);
                let mn0 = ctx.bfe_u32_reg(mn_src, byte_shift, 8);
                let mn1 = ctx.bfe_u32_reg(mn_src, byte_shift_hi, 8);

                // Q4K data (shared across M)
                let q4_addr = ctx.add_u64(sb_addr, q4_off_64);
                let v0 = ctx.ld_global_u32(q4_addr);
                let v1_addr = ctx.add_u64(q4_addr, c_16_64);
                let v1 = ctx.ld_global_u32(v1_addr);

                let v0_lo = ctx.and_u32(v0, c_mask_4bit);
                let v1_lo = ctx.and_u32(v1, c_mask_4bit);
                let v0_hi = ctx.shr_u32_imm(v0, 4);
                let v0_hi = ctx.and_u32(v0_hi, c_mask_4bit);
                let v1_hi = ctx.shr_u32_imm(v1, 4);
                let v1_hi = ctx.and_u32(v1_hi, c_mask_4bit);

                // ===== Compute K base offset for this SB =====
                let sb_k_base = ctx.mul_u32_reg(sb_idx, c_256_u32);

                // ===== Per-batch inline Q8 quantize + DP4A =====
                for mi in 0..m {
                    let x_m_off = if mi == 0 {
                        ctx.mov_u64_imm(0)
                    } else {
                        let mi_reg = ctx.mov_u32_imm(mi);
                        ctx.mul_wide_u32_reg(mi_reg, x_vec_stride)
                    };
                    let x_m_base = ctx.add_u64(x_ptr, x_m_off);

                    // ===== QR=0: Low nibbles — inline Q8 quantize 8 values =====
                    // K offset for this thread's 4 values (QR=0 block)
                    let k_off_lo = ctx.add_u32_reg(sb_k_base, thread_k_off);
                    let k_bytes_lo = ctx.mul_wide_u32(k_off_lo, 4);
                    let x_lo_addr = ctx.add_u64(x_m_base, k_bytes_lo);

                    // Load 4 FP32 values
                    let f0 = ctx.ld_global_f32(x_lo_addr);
                    let f1_addr = ctx.add_u64(x_lo_addr, c_4_64);
                    let f1 = ctx.ld_global_f32(f1_addr);
                    let f2_addr = ctx.add_u64(x_lo_addr, c_8_64);
                    let f2 = ctx.ld_global_f32(f2_addr);
                    let f3_addr = ctx.add_u64(f2_addr, c_4_64);
                    let f3 = ctx.ld_global_f32(f3_addr);

                    // Find absmax across all 4 values
                    let a0 = ctx.abs_f32(f0);
                    let a1 = ctx.abs_f32(f1);
                    let a2 = ctx.abs_f32(f2);
                    let a3 = ctx.abs_f32(f3);
                    let m01 = ctx.max_f32(a0, a1);
                    let m23 = ctx.max_f32(a2, a3);
                    let local_max = ctx.max_f32(m01, m23);

                    // 4-thread shuffle reduction within bq8_group for 16-element Q8 block
                    // (This covers the 4 threads x 4 values = 16 elements for one half of the Q8 block.
                    // The full 32-element Q8 block requires 2 halves, but we quantize each half independently
                    // with its own scale. This is a simplification vs true Q8_1 format.)
                    let mask_4 = ctx.mov_u32_imm(0x0000_000F); // 4-thread mask
                    let t1 = ctx.shfl_down_f32(local_max, 2, mask_4);
                    let max_2 = ctx.max_f32(local_max, t1);
                    let t2 = ctx.shfl_down_f32(max_2, 1, mask_4);
                    let group_max = ctx.max_f32(max_2, t2);

                    // Broadcast max to all 4 threads in group
                    // Use shfl_idx with lane_in_group=0 offset
                    let group_base_lane = ctx.and_u32_imm(half_lane, 0xFC); // round down to group start
                    let group_max = ctx.shfl_idx_f32(group_max, group_base_lane, 0xFFFF_FFFF);

                    // Compute Q8 scale: d = max / 127, inv_scale = 127 / (max + eps)
                    let q8_scale = ctx.mul_f32(group_max, c_inv_127);
                    let scale_eps = ctx.add_f32(q8_scale, c_eps);
                    let inv_scale = ctx.rcp_f32(scale_eps);

                    // Quantize 4 values to INT8
                    let s0 = ctx.mul_f32(f0, inv_scale);
                    let s1 = ctx.mul_f32(f1, inv_scale);
                    let s2 = ctx.mul_f32(f2, inv_scale);
                    let s3 = ctx.mul_f32(f3, inv_scale);

                    let i0 = ctx.cvt_rni_s32_f32(s0);
                    let i1 = ctx.cvt_rni_s32_f32(s1);
                    let i2 = ctx.cvt_rni_s32_f32(s2);
                    let i3 = ctx.cvt_rni_s32_f32(s3);

                    let i0 = ctx.max_s32(i0, c_neg127_s32);
                    let i0 = ctx.min_s32(i0, c_127_s32);
                    let i1 = ctx.max_s32(i1, c_neg127_s32);
                    let i1 = ctx.min_s32(i1, c_127_s32);
                    let i2 = ctx.max_s32(i2, c_neg127_s32);
                    let i2 = ctx.min_s32(i2, c_127_s32);
                    let i3 = ctx.max_s32(i3, c_neg127_s32);
                    let i3 = ctx.min_s32(i3, c_127_s32);

                    // Pack 4 INT8 into u32 for DP4A: byte0=i0, byte1=i1, byte2=i2, byte3=i3
                    let b0 = {
                        let _u8 = ctx.cvt_u8_s32(i0);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let b1 = {
                        let _u8 = ctx.cvt_u8_s32(i1);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let b1 = ctx.shl_u32_imm(b1, 8);
                    let b2 = {
                        let _u8 = ctx.cvt_u8_s32(i2);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let b2 = ctx.shl_u32_imm(b2, 16);
                    let b3 = {
                        let _u8 = ctx.cvt_u8_s32(i3);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let b3 = ctx.shl_u32_imm(b3, 24);
                    let u0_lo = ctx.or_u32(b0, b1);
                    let t = ctx.or_u32(b2, b3);
                    let u0_lo = ctx.or_u32(u0_lo, t);

                    // Also need the v1 part (next 16 elements in the SB)
                    // v1 corresponds to activation at k_off_lo + 16
                    let c_16_u32_k = ctx.mov_u32_imm(16);
                    let k_off_v1 = ctx.add_u32_reg(k_off_lo, c_16_u32_k);
                    let k_bytes_v1 = ctx.mul_wide_u32(k_off_v1, 4);
                    let x_v1_addr = ctx.add_u64(x_m_base, k_bytes_v1);

                    let g0 = ctx.ld_global_f32(x_v1_addr);
                    let g1_addr = ctx.add_u64(x_v1_addr, c_4_64);
                    let g1 = ctx.ld_global_f32(g1_addr);
                    let g2_addr = ctx.add_u64(x_v1_addr, c_8_64);
                    let g2 = ctx.ld_global_f32(g2_addr);
                    let g3_addr = ctx.add_u64(g2_addr, c_4_64);
                    let g3 = ctx.ld_global_f32(g3_addr);

                    // Quantize v1 values with SAME scale (same Q8 block)
                    let gs0 = ctx.mul_f32(g0, inv_scale);
                    let gs1 = ctx.mul_f32(g1, inv_scale);
                    let gs2 = ctx.mul_f32(g2, inv_scale);
                    let gs3 = ctx.mul_f32(g3, inv_scale);

                    let gi0 = ctx.cvt_rni_s32_f32(gs0);
                    let gi1 = ctx.cvt_rni_s32_f32(gs1);
                    let gi2 = ctx.cvt_rni_s32_f32(gs2);
                    let gi3 = ctx.cvt_rni_s32_f32(gs3);

                    let gi0 = ctx.max_s32(gi0, c_neg127_s32);
                    let gi0 = ctx.min_s32(gi0, c_127_s32);
                    let gi1 = ctx.max_s32(gi1, c_neg127_s32);
                    let gi1 = ctx.min_s32(gi1, c_127_s32);
                    let gi2 = ctx.max_s32(gi2, c_neg127_s32);
                    let gi2 = ctx.min_s32(gi2, c_127_s32);
                    let gi3 = ctx.max_s32(gi3, c_neg127_s32);
                    let gi3 = ctx.min_s32(gi3, c_127_s32);

                    let gb0 = {
                        let _u8 = ctx.cvt_u8_s32(gi0);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let gb1 = {
                        let _u8 = ctx.cvt_u8_s32(gi1);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let gb1 = ctx.shl_u32_imm(gb1, 8);
                    let gb2 = {
                        let _u8 = ctx.cvt_u8_s32(gi2);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let gb2 = ctx.shl_u32_imm(gb2, 16);
                    let gb3 = {
                        let _u8 = ctx.cvt_u8_s32(gi3);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let gb3 = ctx.shl_u32_imm(gb3, 24);
                    let u1_lo = ctx.or_u32(gb0, gb1);
                    let gt = ctx.or_u32(gb2, gb3);
                    let u1_lo = ctx.or_u32(u1_lo, gt);

                    // DP4A: same as original (Q4K nibbles x Q8 values)
                    let dot0 = ctx.mov_u32_imm(0);
                    ctx.dp4a_u32_s32_inplace(dot0, v0_lo, u0_lo);
                    ctx.dp4a_u32_s32_inplace(dot0, v1_lo, u1_lo);

                    let sum0 = ctx.mov_u32_imm(0);
                    ctx.dp4a_u32_s32_inplace(sum0, c_ones, u0_lo);
                    ctx.dp4a_u32_s32_inplace(sum0, c_ones, u1_lo);

                    // Accumulate (same formula, using inline Q8 scale)
                    let sdot0 = ctx.mul_lo_s32(sc0, dot0);
                    let msum0 = ctx.mul_lo_s32(mn0, sum0);
                    let sdot0_f = ctx.cvt_f32_s32(sdot0);
                    let msum0_f = ctx.cvt_f32_s32(msum0);
                    let t1 = ctx.mul_f32(d, sdot0_f);
                    let t3 = ctx.fma_f32(neg_dmin, msum0_f, t1);
                    let q8_d0_t3 = ctx.mul_f32(q8_scale, t3);
                    ctx.add_f32_inplace(accs[mi as usize], q8_d0_t3);

                    // ===== QR=1: High nibbles — same pattern with k_off + 128 =====
                    let k_off_hi = ctx.add_u32_reg(k_off_lo, c_128_u32);
                    let k_bytes_hi = ctx.mul_wide_u32(k_off_hi, 4);
                    let x_hi_addr = ctx.add_u64(x_m_base, k_bytes_hi);

                    let h0 = ctx.ld_global_f32(x_hi_addr);
                    let h1_addr = ctx.add_u64(x_hi_addr, c_4_64);
                    let h1 = ctx.ld_global_f32(h1_addr);
                    let h2_addr = ctx.add_u64(x_hi_addr, c_8_64);
                    let h2 = ctx.ld_global_f32(h2_addr);
                    let h3_addr = ctx.add_u64(h2_addr, c_4_64);
                    let h3 = ctx.ld_global_f32(h3_addr);

                    // Absmax for high nibble Q8 block
                    let ha0 = ctx.abs_f32(h0);
                    let ha1 = ctx.abs_f32(h1);
                    let ha2 = ctx.abs_f32(h2);
                    let ha3 = ctx.abs_f32(h3);
                    let hm01 = ctx.max_f32(ha0, ha1);
                    let hm23 = ctx.max_f32(ha2, ha3);
                    let h_local_max = ctx.max_f32(hm01, hm23);

                    let ht1 = ctx.shfl_down_f32(h_local_max, 2, mask_4);
                    let h_max_2 = ctx.max_f32(h_local_max, ht1);
                    let ht2 = ctx.shfl_down_f32(h_max_2, 1, mask_4);
                    let h_group_max = ctx.max_f32(h_max_2, ht2);
                    let h_group_max = ctx.shfl_idx_f32(h_group_max, group_base_lane, 0xFFFF_FFFF);

                    let hq8_scale = ctx.mul_f32(h_group_max, c_inv_127);
                    let h_scale_eps = ctx.add_f32(hq8_scale, c_eps);
                    let h_inv_scale = ctx.rcp_f32(h_scale_eps);

                    // Quantize + pack high 4 values
                    let hs0 = ctx.mul_f32(h0, h_inv_scale);
                    let hs1 = ctx.mul_f32(h1, h_inv_scale);
                    let hs2 = ctx.mul_f32(h2, h_inv_scale);
                    let hs3 = ctx.mul_f32(h3, h_inv_scale);

                    let hi0 = ctx.cvt_rni_s32_f32(hs0);
                    let hi1 = ctx.cvt_rni_s32_f32(hs1);
                    let hi2 = ctx.cvt_rni_s32_f32(hs2);
                    let hi3 = ctx.cvt_rni_s32_f32(hs3);
                    let hi0 = ctx.max_s32(hi0, c_neg127_s32);
                    let hi0 = ctx.min_s32(hi0, c_127_s32);
                    let hi1 = ctx.max_s32(hi1, c_neg127_s32);
                    let hi1 = ctx.min_s32(hi1, c_127_s32);
                    let hi2 = ctx.max_s32(hi2, c_neg127_s32);
                    let hi2 = ctx.min_s32(hi2, c_127_s32);
                    let hi3 = ctx.max_s32(hi3, c_neg127_s32);
                    let hi3 = ctx.min_s32(hi3, c_127_s32);

                    let hb0 = {
                        let _u8 = ctx.cvt_u8_s32(hi0);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let hb1 = {
                        let _u8 = ctx.cvt_u8_s32(hi1);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let hb1 = ctx.shl_u32_imm(hb1, 8);
                    let hb2 = {
                        let _u8 = ctx.cvt_u8_s32(hi2);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let hb2 = ctx.shl_u32_imm(hb2, 16);
                    let hb3 = {
                        let _u8 = ctx.cvt_u8_s32(hi3);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let hb3 = ctx.shl_u32_imm(hb3, 24);
                    let u0_hi = ctx.or_u32(hb0, hb1);
                    let ht = ctx.or_u32(hb2, hb3);
                    let u0_hi = ctx.or_u32(u0_hi, ht);

                    // v1_hi part (k_off_hi + 16)
                    let k_off_v1h = ctx.add_u32_reg(k_off_hi, c_16_u32_k);
                    let k_bytes_v1h = ctx.mul_wide_u32(k_off_v1h, 4);
                    let x_v1h_addr = ctx.add_u64(x_m_base, k_bytes_v1h);

                    let hg0 = ctx.ld_global_f32(x_v1h_addr);
                    let hg1_addr = ctx.add_u64(x_v1h_addr, c_4_64);
                    let hg1 = ctx.ld_global_f32(hg1_addr);
                    let hg2_addr = ctx.add_u64(x_v1h_addr, c_8_64);
                    let hg2 = ctx.ld_global_f32(hg2_addr);
                    let hg3_addr = ctx.add_u64(hg2_addr, c_4_64);
                    let hg3 = ctx.ld_global_f32(hg3_addr);

                    let hgs0 = ctx.mul_f32(hg0, h_inv_scale);
                    let hgs1 = ctx.mul_f32(hg1, h_inv_scale);
                    let hgs2 = ctx.mul_f32(hg2, h_inv_scale);
                    let hgs3 = ctx.mul_f32(hg3, h_inv_scale);

                    let hgi0 = ctx.cvt_rni_s32_f32(hgs0);
                    let hgi1 = ctx.cvt_rni_s32_f32(hgs1);
                    let hgi2 = ctx.cvt_rni_s32_f32(hgs2);
                    let hgi3 = ctx.cvt_rni_s32_f32(hgs3);
                    let hgi0 = ctx.max_s32(hgi0, c_neg127_s32);
                    let hgi0 = ctx.min_s32(hgi0, c_127_s32);
                    let hgi1 = ctx.max_s32(hgi1, c_neg127_s32);
                    let hgi1 = ctx.min_s32(hgi1, c_127_s32);
                    let hgi2 = ctx.max_s32(hgi2, c_neg127_s32);
                    let hgi2 = ctx.min_s32(hgi2, c_127_s32);
                    let hgi3 = ctx.max_s32(hgi3, c_neg127_s32);
                    let hgi3 = ctx.min_s32(hgi3, c_127_s32);

                    let hgb0 = {
                        let _u8 = ctx.cvt_u8_s32(hgi0);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let hgb1 = {
                        let _u8 = ctx.cvt_u8_s32(hgi1);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let hgb1 = ctx.shl_u32_imm(hgb1, 8);
                    let hgb2 = {
                        let _u8 = ctx.cvt_u8_s32(hgi2);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let hgb2 = ctx.shl_u32_imm(hgb2, 16);
                    let hgb3 = {
                        let _u8 = ctx.cvt_u8_s32(hgi3);
                        ctx.cvt_u32_u8(_u8)
                    };
                    let hgb3 = ctx.shl_u32_imm(hgb3, 24);
                    let u1_hi = ctx.or_u32(hgb0, hgb1);
                    let hgt = ctx.or_u32(hgb2, hgb3);
                    let u1_hi = ctx.or_u32(u1_hi, hgt);

                    // DP4A for high nibbles
                    let dot1 = ctx.mov_u32_imm(0);
                    ctx.dp4a_u32_s32_inplace(dot1, v0_hi, u0_hi);
                    ctx.dp4a_u32_s32_inplace(dot1, v1_hi, u1_hi);

                    let sum1 = ctx.mov_u32_imm(0);
                    ctx.dp4a_u32_s32_inplace(sum1, c_ones, u0_hi);
                    ctx.dp4a_u32_s32_inplace(sum1, c_ones, u1_hi);

                    let sdot1 = ctx.mul_lo_s32(sc1, dot1);
                    let msum1 = ctx.mul_lo_s32(mn1, sum1);
                    let sdot1_f = ctx.cvt_f32_s32(sdot1);
                    let msum1_f = ctx.cvt_f32_s32(msum1);
                    let t1 = ctx.mul_f32(d, sdot1_f);
                    let t3 = ctx.fma_f32(neg_dmin, msum1_f, t1);
                    let q8_d1_t3 = ctx.mul_f32(hq8_scale, t3);
                    ctx.add_f32_inplace(accs[mi as usize], q8_d1_t3);
                }

                // Stride
                ctx.add_u32_reg_inplace(sb_idx, num_hw);
                ctx.branch("iq8_sb_loop");

                ctx.label("iq8_sb_end");

                // ===== Half-warp reduction (same as HwDp4a) =====
                for acc in &accs {
                    let t = ctx.shfl_down_f32(*acc, 8, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(*acc, t);
                    let t = ctx.shfl_down_f32(*acc, 4, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(*acc, t);
                    let t = ctx.shfl_down_f32(*acc, 2, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(*acc, t);
                    let t = ctx.shfl_down_f32(*acc, 1, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(*acc, t);
                }

                // Cross-half-warp reduction via shared memory
                let c_zero_u32 = ctx.mov_u32_imm(0);
                let is_lane0 = ctx.setp_eq_u32(half_lane, c_zero_u32);
                let smem_base = ctx.mov_u64_imm(0);
                ctx.branch_if_not(is_lane0, "iq8_skip_smem_write");

                let c_m_u32 = ctx.mov_u32_imm(m);
                for (mi, acc) in accs.iter().enumerate() {
                    let mi_reg = ctx.mov_u32_imm(mi as u32);
                    let hw_m = ctx.mul_u32_reg(half_warp_id, c_m_u32);
                    let smem_idx = ctx.add_u32_reg(hw_m, mi_reg);
                    let smem_off = ctx.mul_wide_u32(smem_idx, 4);
                    let smem_addr = ctx.add_u64(smem_base, smem_off);
                    ctx.st_shared_f32(smem_addr, *acc);
                }

                ctx.label("iq8_skip_smem_write");
                ctx.bar_sync(0);

                let c_zero2 = ctx.mov_u32_imm(0);
                let is_first_hw = ctx.setp_eq_u32(half_warp_id, c_zero2);
                ctx.branch_if_not(is_first_hw, "iq8_skip_output");

                let c_zero3 = ctx.mov_u32_imm(0);
                let is_lane0_first = ctx.setp_eq_u32(half_lane, c_zero3);
                ctx.branch_if_not(is_lane0_first, "iq8_skip_output");

                let c_m_red = ctx.mov_u32_imm(m);
                for mi in 0..m {
                    let mut total = ctx.mov_f32_imm(0.0);
                    for hw in 0..num_half_warps {
                        let mi_reg = ctx.mov_u32_imm(mi);
                        let hw_reg = ctx.mov_u32_imm(hw);
                        let hw_m = ctx.mul_u32_reg(hw_reg, c_m_red);
                        let idx = ctx.add_u32_reg(hw_m, mi_reg);
                        let off = ctx.mul_wide_u32(idx, 4);
                        let addr = ctx.add_u64(smem_base, off);
                        let val = ctx.ld_shared_f32(addr);
                        ctx.add_f32_inplace(total, val);
                    }

                    let mi_reg = ctx.mov_u32_imm(mi);
                    let row_m = ctx.mul_u32_reg(row_idx, c_m_red);
                    let out_idx = ctx.add_u32_reg(row_m, mi_reg);
                    let out_off = ctx.mul_wide_u32(out_idx, 4);
                    let out_addr = ctx.add_u64(y_ptr, out_off);
                    ctx.st_global_f32(out_addr, total);
                }

                ctx.label("iq8_skip_output");
                ctx.bar_sync(1);

                ctx.add_u32_reg_inplace(row_idx, grid_dim);
                ctx.branch("iq8_row_loop");

                ctx.label("iq8_exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inline_q8_kernel_builds_ptx() {
        let kernel = InlineQ8Dp4aQ4KGemvKernel::new(1536, 1536, 4);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("inline_q8_dp4a_q4k_gemv"));
        assert!(ptx.contains(".entry"));
        assert!(ptx.is_ascii(), "PTX must be pure ASCII");
    }

    #[test]
    fn test_inline_q8_kernel_m1() {
        let kernel = InlineQ8Dp4aQ4KGemvKernel::new(1536, 1536, 1);
        let ptx = kernel.emit_ptx();
        assert!(ptx.is_ascii());
        assert!(ptx.contains("dp4a"));
    }
}
