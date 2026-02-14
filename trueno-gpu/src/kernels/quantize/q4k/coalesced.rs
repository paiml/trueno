//! Coalesced Q4_K GEMV Kernels with Optimized Memory Access
//!
//! - `CoalescedQ4KGemvKernel`: Scale loading via broadcast, vectorized qs
//! - `VectorizedQ4KGemvKernel`: Coalesced u32 loads for high bandwidth
//! - `MwvDp4aQ4KGemvKernel`: DP4A integer dot products with Q8_1-quantized activations

use crate::kernels::quantize::{Kernel, Q4K_SUPER_BLOCK_BYTES, Q4K_SUPER_BLOCK_SIZE};
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Coalesced Q4_K GEMV kernel with optimized memory access (PAR-062)
///
/// Key optimizations over basic Q4KGemvKernel:
/// 1. **Scale loading**: Lane 0 loads 12 scale bytes as 3 x u32, broadcasts via shuffle
///    - Reduces 384 redundant byte loads to 3 loads + 3 broadcasts per super-block
/// 2. **Vectorized qs access**: Uses u32 loads for quantized values (4 bytes at once)
///    - Improves memory transaction efficiency
///
/// # Performance Target
/// - Memory bandwidth: 100+ GB/s (vs 7 GB/s in basic kernel)
/// - Goal: 2x llama.cpp performance
///
/// # References
/// - llama.cpp vec_dot_q4_K_q8_1 (vecdotq.cuh:792-818)
/// - "Optimizing CUDA Memory Transactions" (NVIDIA Best Practices Guide)
#[derive(Debug, Clone)]
pub struct CoalescedQ4KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl CoalescedQ4KGemvKernel {
    /// Create a new coalesced Q4_K GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }

    /// Get number of super-blocks per row (ceiling division)
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        (self.k + Q4K_SUPER_BLOCK_SIZE - 1) / Q4K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for CoalescedQ4KGemvKernel {
    fn name(&self) -> &str {
        "coalesced_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("coalesced_q4k_gemv")
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
                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                // Row base address
                let sb_bytes = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_end");

                let sb_offset = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d and dmin (all lanes)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // ========================================================
                // PAR-062 OPTIMIZATION: Vectorized scale loading
                // Only lane 0 loads scales as 3 x u32, then broadcasts
                // ========================================================
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);

                // Lane 0 loads 12 bytes as 3 x u32
                let one = ctx.mov_u32_imm(1);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);

                // Initialize scale registers (will be overwritten by lane 0)
                let scales_0_3 = ctx.mov_u32_imm(0);
                let scales_4_7 = ctx.mov_u32_imm(0);
                let scales_8_11 = ctx.mov_u32_imm(0);

                ctx.branch_if_not(is_lane0, "skip_scale_load");

                // Lane 0: Load scales as 3 x u32 (coalesced within transaction)
                ctx.ld_global_u32_into(scales_0_3, scales_base);
                let four_64b = ctx.mov_u64_imm(4);
                let scales_4_addr = ctx.add_u64(scales_base, four_64b);
                ctx.ld_global_u32_into(scales_4_7, scales_4_addr);
                let eight_64 = ctx.mov_u64_imm(8);
                let scales_8_addr = ctx.add_u64(scales_base, eight_64);
                ctx.ld_global_u32_into(scales_8_11, scales_8_addr);

                ctx.label("skip_scale_load");

                // Broadcast scales from lane 0 to all lanes
                let scales_0_3_bcast = ctx.shfl_idx_u32(scales_0_3, 0, 0xFFFF_FFFF);
                let scales_4_7_bcast = ctx.shfl_idx_u32(scales_4_7, 0, 0xFFFF_FFFF);
                let scales_8_11_bcast = ctx.shfl_idx_u32(scales_8_11, 0, 0xFFFF_FFFF);

                // Extract individual scale bytes using bit operations
                let mask_8bit = ctx.mov_u32_imm(0xFF);
                let eight = ctx.mov_u32_imm(8);
                let sixteen = ctx.mov_u32_imm(16);
                let twenty_four = ctx.mov_u32_imm(24);

                // s0-s3 from scales_0_3_bcast
                let s0_32 = ctx.and_u32(scales_0_3_bcast, mask_8bit);
                let s0_shifted = ctx.shr_u32(scales_0_3_bcast, eight);
                let s1_32 = ctx.and_u32(s0_shifted, mask_8bit);
                let s1_shifted = ctx.shr_u32(scales_0_3_bcast, sixteen);
                let s2_32 = ctx.and_u32(s1_shifted, mask_8bit);
                let s3_32 = ctx.shr_u32(scales_0_3_bcast, twenty_four);

                // s4-s7 from scales_4_7_bcast
                let s4_32 = ctx.and_u32(scales_4_7_bcast, mask_8bit);
                let s4_shifted = ctx.shr_u32(scales_4_7_bcast, eight);
                let s5_32 = ctx.and_u32(s4_shifted, mask_8bit);
                let s5_shifted = ctx.shr_u32(scales_4_7_bcast, sixteen);
                let s6_32 = ctx.and_u32(s5_shifted, mask_8bit);
                let s7_32 = ctx.shr_u32(scales_4_7_bcast, twenty_four);

                // s8-s11 from scales_8_11_bcast
                let s8_32 = ctx.and_u32(scales_8_11_bcast, mask_8bit);
                let s8_shifted = ctx.shr_u32(scales_8_11_bcast, eight);
                let s9_32 = ctx.and_u32(s8_shifted, mask_8bit);
                let s9_shifted = ctx.shr_u32(scales_8_11_bcast, sixteen);
                let s10_32 = ctx.and_u32(s9_shifted, mask_8bit);
                let s11_32 = ctx.shr_u32(scales_8_11_bcast, twenty_four);

                // Constants for scale/min extraction
                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let four = ctx.mov_u32_imm(4);
                let six = ctx.mov_u32_imm(6);

                // Extract scale/min for all 8 blocks (same logic as original)
                // Block 0-3: simple extraction
                let scale0 = ctx.and_u32(s0_32, mask_6bit);
                let min0 = ctx.and_u32(s4_32, mask_6bit);
                let scale0_f = ctx.cvt_f32_u32(scale0);
                let min0_f = ctx.cvt_f32_u32(min0);

                let scale1 = ctx.and_u32(s1_32, mask_6bit);
                let min1 = ctx.and_u32(s5_32, mask_6bit);
                let scale1_f = ctx.cvt_f32_u32(scale1);
                let min1_f = ctx.cvt_f32_u32(min1);

                let scale2 = ctx.and_u32(s2_32, mask_6bit);
                let min2 = ctx.and_u32(s6_32, mask_6bit);
                let scale2_f = ctx.cvt_f32_u32(scale2);
                let min2_f = ctx.cvt_f32_u32(min2);

                let scale3 = ctx.and_u32(s3_32, mask_6bit);
                let min3 = ctx.and_u32(s7_32, mask_6bit);
                let scale3_f = ctx.cvt_f32_u32(scale3);
                let min3_f = ctx.cvt_f32_u32(min3);

                // Block 4-7: complex extraction
                let s8_lo = ctx.and_u32(s8_32, mask_4bit);
                let s0_hi = ctx.shr_u32(s0_32, six);
                let s0_hi_shifted = ctx.shl_u32(s0_hi, four);
                let scale4 = ctx.or_u32(s8_lo, s0_hi_shifted);
                let s8_hi = ctx.shr_u32(s8_32, four);
                let s4_hi = ctx.shr_u32(s4_32, six);
                let s4_hi_shifted = ctx.shl_u32(s4_hi, four);
                let min4 = ctx.or_u32(s8_hi, s4_hi_shifted);
                let scale4_f = ctx.cvt_f32_u32(scale4);
                let min4_f = ctx.cvt_f32_u32(min4);

                let s9_lo = ctx.and_u32(s9_32, mask_4bit);
                let s1_hi = ctx.shr_u32(s1_32, six);
                let s1_hi_shifted = ctx.shl_u32(s1_hi, four);
                let scale5 = ctx.or_u32(s9_lo, s1_hi_shifted);
                let s9_hi = ctx.shr_u32(s9_32, four);
                let s5_hi = ctx.shr_u32(s5_32, six);
                let s5_hi_shifted = ctx.shl_u32(s5_hi, four);
                let min5 = ctx.or_u32(s9_hi, s5_hi_shifted);
                let scale5_f = ctx.cvt_f32_u32(scale5);
                let min5_f = ctx.cvt_f32_u32(min5);

                let s10_lo = ctx.and_u32(s10_32, mask_4bit);
                let s2_hi = ctx.shr_u32(s2_32, six);
                let s2_hi_shifted = ctx.shl_u32(s2_hi, four);
                let scale6 = ctx.or_u32(s10_lo, s2_hi_shifted);
                let s10_hi = ctx.shr_u32(s10_32, four);
                let s6_hi = ctx.shr_u32(s6_32, six);
                let s6_hi_shifted = ctx.shl_u32(s6_hi, four);
                let min6 = ctx.or_u32(s10_hi, s6_hi_shifted);
                let scale6_f = ctx.cvt_f32_u32(scale6);
                let min6_f = ctx.cvt_f32_u32(min6);

                let s11_lo = ctx.and_u32(s11_32, mask_4bit);
                let s3_hi = ctx.shr_u32(s3_32, six);
                let s3_hi_shifted = ctx.shl_u32(s3_hi, four);
                let scale7 = ctx.or_u32(s11_lo, s3_hi_shifted);
                let s11_hi = ctx.shr_u32(s11_32, four);
                let s7_hi = ctx.shr_u32(s7_32, six);
                let s7_hi_shifted = ctx.shl_u32(s7_hi, four);
                let min7 = ctx.or_u32(s11_hi, s7_hi_shifted);
                let scale7_f = ctx.cvt_f32_u32(scale7);
                let min7_f = ctx.cvt_f32_u32(min7);

                // Precompute d*scale and dmin*min
                let ds0 = ctx.mul_f32(d, scale0_f);
                let dm0 = ctx.mul_f32(dmin, min0_f);
                let ds1 = ctx.mul_f32(d, scale1_f);
                let dm1 = ctx.mul_f32(dmin, min1_f);
                let ds2 = ctx.mul_f32(d, scale2_f);
                let dm2 = ctx.mul_f32(dmin, min2_f);
                let ds3 = ctx.mul_f32(d, scale3_f);
                let dm3 = ctx.mul_f32(dmin, min3_f);
                let ds4 = ctx.mul_f32(d, scale4_f);
                let dm4 = ctx.mul_f32(dmin, min4_f);
                let ds5 = ctx.mul_f32(d, scale5_f);
                let dm5 = ctx.mul_f32(dmin, min5_f);
                let ds6 = ctx.mul_f32(d, scale6_f);
                let dm6 = ctx.mul_f32(dmin, min6_f);
                let ds7 = ctx.mul_f32(d, scale7_f);
                let dm7 = ctx.mul_f32(dmin, min7_f);

                // qs base
                let sixteen_64 = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen_64);

                let thread_partial = ctx.mov_f32_imm(0.0);

                // Process 8 values per thread (unrolled)
                let offsets_and_blocks: [(u32, u32); 8] = [
                    (0, 0),
                    (32, 1),
                    (64, 2),
                    (96, 3),
                    (128, 4),
                    (160, 5),
                    (192, 6),
                    (224, 7),
                ];

                for (offset, block_idx) in offsets_and_blocks {
                    let (ds, dm) = match block_idx {
                        0 => (ds0, dm0),
                        1 => (ds1, dm1),
                        2 => (ds2, dm2),
                        3 => (ds3, dm3),
                        4 => (ds4, dm4),
                        5 => (ds5, dm5),
                        6 => (ds6, dm6),
                        _ => (ds7, dm7),
                    };

                    let offset_reg = ctx.mov_u32_imm(offset);
                    let val_idx = ctx.add_u32_reg(lane_id, offset_reg);

                    // Calculate byte address (same logic, already coalesced for qs)
                    let chunk_idx = ctx.div_u32(val_idx, 64);
                    let val_in_chunk = ctx.rem_u32(val_idx, 64);
                    let byte_in_chunk = ctx.rem_u32(val_in_chunk, 32);

                    let chunk_offset = ctx.mul_u32(chunk_idx, 32);
                    let qs_byte_offset = ctx.add_u32_reg(chunk_offset, byte_in_chunk);
                    let qs_byte_offset_64 = ctx.cvt_u64_u32(qs_byte_offset);
                    let qs_addr = ctx.add_u64(qs_base, qs_byte_offset_64);
                    let packed = ctx.ld_global_u8(qs_addr);
                    let packed_32 = ctx.cvt_u32_u8(packed);

                    let mask_4bit_q = ctx.mov_u32_imm(0xF);
                    let four_q = ctx.mov_u32_imm(4);
                    let val_in_chunk_div_32 = ctx.div_u32(val_in_chunk, 32);
                    let shift_amount = ctx.mul_u32_reg(val_in_chunk_div_32, four_q);
                    let shifted = ctx.shr_u32(packed_32, shift_amount);
                    let quant = ctx.and_u32(shifted, mask_4bit_q);

                    let quant_f32 = ctx.cvt_f32_u32(quant);
                    let scaled = ctx.mul_f32(ds, quant_f32);
                    let dequant = ctx.sub_f32(scaled, dm);

                    // Load activation
                    let sb_k_base = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
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

                // Warp shuffle reduce
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

/// Wide Q4_K GEMV kernel with 256 threads (8 warps) and cross-warp reduction (PAR-082)
///
/// Root cause of 3x Ollama performance gap:
/// - CoalescedQ4KGemvKernel uses 32 threads (1 warp) per output row
/// - Only 33% occupancy on RTX 4090 SMs (16 blocks × 1 warp = 16/48 warps)
/// - Cannot hide memory latency — SM idles during ~400 cycle global loads
///
/// Fix: 8 warps per block, each processes every 8th super-block, then reduce via shared memory
/// - 256 threads per block → 67-100% occupancy (6 blocks × 8 warps = 48 warps/SM)
/// - 8x more warps to hide memory latency
/// - Shared memory: only 32 bytes (8 partial sums)
///
/// # Performance Target
/// - Memory bandwidth: 500+ GB/s (vs ~190 GB/s with 1 warp)
/// - Decode: 100+ tok/s 7B Q4K (vs 39 tok/s with 1 warp)
///
/// # References
/// - llama.cpp `dequantize_mul_mat_vec_q4_K` uses 256 threads
/// - Pope et al. (2023) "Efficiently Scaling Transformer Inference"
#[derive(Debug, Clone)]
pub struct WideQ4KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Number of warps per block (default: 8 = 256 threads)
    pub num_warps: u32,
}

impl WideQ4KGemvKernel {
    /// Create a new wide Q4_K GEMV kernel with 8 warps (256 threads)
    ///
    /// Empirically measured: 8 warps (67 tok/s) > 4 warps (61 tok/s) on RTX 4090
    /// The extra SM occupancy from 8 warps hides memory latency better
    /// than the reduced cross-warp reduction overhead of 4 warps.
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n, num_warps: 8 }
    }
}

impl Kernel for WideQ4KGemvKernel {
    fn name(&self) -> &str {
        "wide_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;

        // Shared memory for cross-warp reduction: num_warps × 4 bytes
        let smem_size = (num_warps * 4) as usize;

        PtxKernel::new("wide_q4k_gemv")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .shared_memory(smem_size)
            .build(move |ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);

                // Bounds check (block_id = output row index)
                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "wide_exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                // Calculate super-blocks per row
                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                // Row base address
                let sb_bytes = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Each warp starts at its own super-block index, strides by num_warps
                let sb_idx_init = ctx.mov_u32_imm(0);
                let sb_idx = ctx.add_u32_reg(sb_idx_init, warp_id);

                let num_warps_reg = ctx.mov_u32_imm(num_warps);

                ctx.label("wide_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "wide_sb_loop_end");

                let sb_offset = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d and dmin (all lanes)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // ========================================================
                // Scale loading: Lane 0 loads 12 bytes as 3 x u32, broadcasts
                // (Same as CoalescedQ4KGemvKernel)
                // ========================================================
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);

                let one = ctx.mov_u32_imm(1);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);

                let scales_0_3 = ctx.mov_u32_imm(0);
                let scales_4_7 = ctx.mov_u32_imm(0);
                let scales_8_11 = ctx.mov_u32_imm(0);

                ctx.branch_if_not(is_lane0, "wide_skip_scale_load");

                ctx.ld_global_u32_into(scales_0_3, scales_base);
                let four_64b = ctx.mov_u64_imm(4);
                let scales_4_addr = ctx.add_u64(scales_base, four_64b);
                ctx.ld_global_u32_into(scales_4_7, scales_4_addr);
                let eight_64 = ctx.mov_u64_imm(8);
                let scales_8_addr = ctx.add_u64(scales_base, eight_64);
                ctx.ld_global_u32_into(scales_8_11, scales_8_addr);

                ctx.label("wide_skip_scale_load");

                let scales_0_3_bcast = ctx.shfl_idx_u32(scales_0_3, 0, 0xFFFF_FFFF);
                let scales_4_7_bcast = ctx.shfl_idx_u32(scales_4_7, 0, 0xFFFF_FFFF);
                let scales_8_11_bcast = ctx.shfl_idx_u32(scales_8_11, 0, 0xFFFF_FFFF);

                // Extract scale bytes
                let mask_8bit = ctx.mov_u32_imm(0xFF);
                let eight = ctx.mov_u32_imm(8);
                let sixteen = ctx.mov_u32_imm(16);
                let twenty_four = ctx.mov_u32_imm(24);

                let s0_32 = ctx.and_u32(scales_0_3_bcast, mask_8bit);
                let s0_shifted = ctx.shr_u32(scales_0_3_bcast, eight);
                let s1_32 = ctx.and_u32(s0_shifted, mask_8bit);
                let s1_shifted = ctx.shr_u32(scales_0_3_bcast, sixteen);
                let s2_32 = ctx.and_u32(s1_shifted, mask_8bit);
                let s3_32 = ctx.shr_u32(scales_0_3_bcast, twenty_four);

                let s4_32 = ctx.and_u32(scales_4_7_bcast, mask_8bit);
                let s4_shifted = ctx.shr_u32(scales_4_7_bcast, eight);
                let s5_32 = ctx.and_u32(s4_shifted, mask_8bit);
                let s5_shifted = ctx.shr_u32(scales_4_7_bcast, sixteen);
                let s6_32 = ctx.and_u32(s5_shifted, mask_8bit);
                let s7_32 = ctx.shr_u32(scales_4_7_bcast, twenty_four);

                let s8_32 = ctx.and_u32(scales_8_11_bcast, mask_8bit);
                let s8_shifted = ctx.shr_u32(scales_8_11_bcast, eight);
                let s9_32 = ctx.and_u32(s8_shifted, mask_8bit);
                let s9_shifted = ctx.shr_u32(scales_8_11_bcast, sixteen);
                let s10_32 = ctx.and_u32(s9_shifted, mask_8bit);
                let s11_32 = ctx.shr_u32(scales_8_11_bcast, twenty_four);

                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let four = ctx.mov_u32_imm(4);
                let six = ctx.mov_u32_imm(6);

                // Block 0-3
                let scale0 = ctx.and_u32(s0_32, mask_6bit);
                let min0 = ctx.and_u32(s4_32, mask_6bit);
                let scale0_f = ctx.cvt_f32_u32(scale0);
                let min0_f = ctx.cvt_f32_u32(min0);
                let scale1 = ctx.and_u32(s1_32, mask_6bit);
                let min1 = ctx.and_u32(s5_32, mask_6bit);
                let scale1_f = ctx.cvt_f32_u32(scale1);
                let min1_f = ctx.cvt_f32_u32(min1);
                let scale2 = ctx.and_u32(s2_32, mask_6bit);
                let min2 = ctx.and_u32(s6_32, mask_6bit);
                let scale2_f = ctx.cvt_f32_u32(scale2);
                let min2_f = ctx.cvt_f32_u32(min2);
                let scale3 = ctx.and_u32(s3_32, mask_6bit);
                let min3 = ctx.and_u32(s7_32, mask_6bit);
                let scale3_f = ctx.cvt_f32_u32(scale3);
                let min3_f = ctx.cvt_f32_u32(min3);

                // Block 4-7
                let s8_lo = ctx.and_u32(s8_32, mask_4bit);
                let s0_hi = ctx.shr_u32(s0_32, six);
                let s0_hi_shifted = ctx.shl_u32(s0_hi, four);
                let scale4 = ctx.or_u32(s8_lo, s0_hi_shifted);
                let s8_hi = ctx.shr_u32(s8_32, four);
                let s4_hi = ctx.shr_u32(s4_32, six);
                let s4_hi_shifted = ctx.shl_u32(s4_hi, four);
                let min4 = ctx.or_u32(s8_hi, s4_hi_shifted);
                let scale4_f = ctx.cvt_f32_u32(scale4);
                let min4_f = ctx.cvt_f32_u32(min4);

                let s9_lo = ctx.and_u32(s9_32, mask_4bit);
                let s1_hi = ctx.shr_u32(s1_32, six);
                let s1_hi_shifted = ctx.shl_u32(s1_hi, four);
                let scale5 = ctx.or_u32(s9_lo, s1_hi_shifted);
                let s9_hi = ctx.shr_u32(s9_32, four);
                let s5_hi = ctx.shr_u32(s5_32, six);
                let s5_hi_shifted = ctx.shl_u32(s5_hi, four);
                let min5 = ctx.or_u32(s9_hi, s5_hi_shifted);
                let scale5_f = ctx.cvt_f32_u32(scale5);
                let min5_f = ctx.cvt_f32_u32(min5);

                let s10_lo = ctx.and_u32(s10_32, mask_4bit);
                let s2_hi = ctx.shr_u32(s2_32, six);
                let s2_hi_shifted = ctx.shl_u32(s2_hi, four);
                let scale6 = ctx.or_u32(s10_lo, s2_hi_shifted);
                let s10_hi = ctx.shr_u32(s10_32, four);
                let s6_hi = ctx.shr_u32(s6_32, six);
                let s6_hi_shifted = ctx.shl_u32(s6_hi, four);
                let min6 = ctx.or_u32(s10_hi, s6_hi_shifted);
                let scale6_f = ctx.cvt_f32_u32(scale6);
                let min6_f = ctx.cvt_f32_u32(min6);

                let s11_lo = ctx.and_u32(s11_32, mask_4bit);
                let s3_hi = ctx.shr_u32(s3_32, six);
                let s3_hi_shifted = ctx.shl_u32(s3_hi, four);
                let scale7 = ctx.or_u32(s11_lo, s3_hi_shifted);
                let s11_hi = ctx.shr_u32(s11_32, four);
                let s7_hi = ctx.shr_u32(s7_32, six);
                let s7_hi_shifted = ctx.shl_u32(s7_hi, four);
                let min7 = ctx.or_u32(s11_hi, s7_hi_shifted);
                let scale7_f = ctx.cvt_f32_u32(scale7);
                let min7_f = ctx.cvt_f32_u32(min7);

                // Precompute d*scale and dmin*min
                let ds0 = ctx.mul_f32(d, scale0_f);
                let dm0 = ctx.mul_f32(dmin, min0_f);
                let ds1 = ctx.mul_f32(d, scale1_f);
                let dm1 = ctx.mul_f32(dmin, min1_f);
                let ds2 = ctx.mul_f32(d, scale2_f);
                let dm2 = ctx.mul_f32(dmin, min2_f);
                let ds3 = ctx.mul_f32(d, scale3_f);
                let dm3 = ctx.mul_f32(dmin, min3_f);
                let ds4 = ctx.mul_f32(d, scale4_f);
                let dm4 = ctx.mul_f32(dmin, min4_f);
                let ds5 = ctx.mul_f32(d, scale5_f);
                let dm5 = ctx.mul_f32(dmin, min5_f);
                let ds6 = ctx.mul_f32(d, scale6_f);
                let dm6 = ctx.mul_f32(dmin, min6_f);
                let ds7 = ctx.mul_f32(d, scale7_f);
                let dm7 = ctx.mul_f32(dmin, min7_f);

                // qs base
                let sixteen_64 = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen_64);

                let thread_partial = ctx.mov_f32_imm(0.0);

                // Process 8 values per thread (same as CoalescedQ4K)
                let offsets_and_blocks: [(u32, u32); 8] = [
                    (0, 0),
                    (32, 1),
                    (64, 2),
                    (96, 3),
                    (128, 4),
                    (160, 5),
                    (192, 6),
                    (224, 7),
                ];

                for (offset, block_idx) in offsets_and_blocks {
                    let (ds, dm) = match block_idx {
                        0 => (ds0, dm0),
                        1 => (ds1, dm1),
                        2 => (ds2, dm2),
                        3 => (ds3, dm3),
                        4 => (ds4, dm4),
                        5 => (ds5, dm5),
                        6 => (ds6, dm6),
                        _ => (ds7, dm7),
                    };

                    let offset_reg = ctx.mov_u32_imm(offset);
                    let val_idx = ctx.add_u32_reg(lane_id, offset_reg);

                    let chunk_idx = ctx.div_u32(val_idx, 64);
                    let val_in_chunk = ctx.rem_u32(val_idx, 64);
                    let byte_in_chunk = ctx.rem_u32(val_in_chunk, 32);
                    let chunk_offset = ctx.mul_u32(chunk_idx, 32);
                    let qs_byte_offset = ctx.add_u32_reg(chunk_offset, byte_in_chunk);
                    let qs_byte_offset_64 = ctx.cvt_u64_u32(qs_byte_offset);
                    let qs_addr = ctx.add_u64(qs_base, qs_byte_offset_64);
                    let packed = ctx.ld_global_u8(qs_addr);
                    let packed_32 = ctx.cvt_u32_u8(packed);

                    let mask_4bit_q = ctx.mov_u32_imm(0xF);
                    let four_q = ctx.mov_u32_imm(4);
                    let val_in_chunk_div_32 = ctx.div_u32(val_in_chunk, 32);
                    let shift_amount = ctx.mul_u32_reg(val_in_chunk_div_32, four_q);
                    let shifted = ctx.shr_u32(packed_32, shift_amount);
                    let quant = ctx.and_u32(shifted, mask_4bit_q);

                    let quant_f32 = ctx.cvt_f32_u32(quant);
                    let scaled = ctx.mul_f32(ds, quant_f32);
                    let dequant = ctx.sub_f32(scaled, dm);

                    let sb_k_base = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                    let x_idx = ctx.add_u32_reg(sb_k_base, val_idx);
                    let x_idx_64 = ctx.cvt_u64_u32(x_idx);
                    let x_bytes = ctx.mul_u64(x_idx_64, 4);
                    let x_addr = ctx.add_u64(x_ptr, x_bytes);
                    let x_val = ctx.ld_global_f32(x_addr);

                    ctx.fma_f32_inplace(thread_partial, x_val, dequant);
                }

                ctx.add_f32_inplace(acc, thread_partial);

                // Stride by num_warps (not 1)
                ctx.add_u32_reg_inplace(sb_idx, num_warps_reg);
                ctx.branch("wide_sb_loop");

                ctx.label("wide_sb_loop_end");

                // ====================================================
                // Phase 1: Intra-warp reduction (warp shuffle)
                // ====================================================
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

                // ====================================================
                // Phase 2: Cross-warp reduction via shared memory
                // Lane 0 of each warp writes partial sum to shared[warp_id]
                // ====================================================
                let zero_u32 = ctx.mov_u32_imm(0);
                let is_lane0_final = ctx.setp_eq_u32(lane_id, zero_u32);
                ctx.branch_if_not(is_lane0_final, "wide_skip_smem_write");

                // shared[warp_id] = acc
                let four_smem = ctx.mov_u32_imm(4);
                let warp_offset = ctx.mul_u32_reg(warp_id, four_smem); // 4 bytes per f32
                let warp_smem_addr = ctx.cvt_u64_u32(warp_offset);
                ctx.st_shared_f32(warp_smem_addr, acc);

                ctx.label("wide_skip_smem_write");

                // Full block synchronization
                ctx.bar_sync(0);

                // Thread 0 reads all partial sums and reduces
                let is_thread0 = ctx.setp_eq_u32(thread_id, zero_u32);
                ctx.branch_if_not(is_thread0, "wide_exit");

                let final_sum = ctx.mov_f32_imm(0.0);
                // Unrolled loop over num_warps (8) partial sums
                for w in 0..num_warps {
                    let w_offset = ctx.mov_u64_imm(u64::from(w * 4));
                    let partial = ctx.ld_shared_f32(w_offset);
                    ctx.add_f32_inplace(final_sum, partial);
                }

                // Write final result to global memory
                let y_offset = ctx.mul_wide_u32(block_id, 4);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, final_sum);

                ctx.label("wide_exit");
                ctx.ret();
            })
    }
}

/// Vectorized Q4_K GEMV kernel with coalesced u32 loads (PAR-069)
///
/// Achieves high memory bandwidth by loading weights as u32:
/// - Each thread loads 4 consecutive bytes (8 nibbles = 8 Q4 values)
/// - 32 threads × 4 bytes = 128 bytes per warp transaction (perfectly coalesced!)
pub struct VectorizedQ4KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
}

impl VectorizedQ4KGemvKernel {
    /// Create a new vectorized Q4_K GEMV kernel
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }
}

impl Kernel for VectorizedQ4KGemvKernel {
    fn name(&self) -> &str {
        "vectorized_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Grid: one warp (32 threads) per output row
        // Each thread loads 4 bytes = 8 nibbles = 8 values per super-block
        // Total: 32 threads × 8 values = 256 values = 1 super-block per iteration
        PtxKernel::new("vectorized_q4k_gemv")
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
                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                // Row base address
                let sb_bytes = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop_v");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_end_v");

                let sb_offset = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_offset);

                // Load d and dmin (all lanes)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // ========================================================
                // PAR-069: VECTORIZED SCALE LOADING (from CoalescedQ4K)
                // Lane 0 loads scales as 3 x u32, broadcasts via shuffle
                // ========================================================
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);
                let one = ctx.mov_u32_imm(1);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);

                let scales_0_3 = ctx.mov_u32_imm(0);
                let scales_4_7 = ctx.mov_u32_imm(0);
                let scales_8_11 = ctx.mov_u32_imm(0);

                ctx.branch_if_not(is_lane0, "skip_scale_load_v");
                ctx.ld_global_u32_into(scales_0_3, scales_base);
                let four_64b = ctx.mov_u64_imm(4);
                let scales_4_addr = ctx.add_u64(scales_base, four_64b);
                ctx.ld_global_u32_into(scales_4_7, scales_4_addr);
                let eight_64 = ctx.mov_u64_imm(8);
                let scales_8_addr = ctx.add_u64(scales_base, eight_64);
                ctx.ld_global_u32_into(scales_8_11, scales_8_addr);
                ctx.label("skip_scale_load_v");

                let scales_0_3_bcast = ctx.shfl_idx_u32(scales_0_3, 0, 0xFFFF_FFFF);
                let scales_4_7_bcast = ctx.shfl_idx_u32(scales_4_7, 0, 0xFFFF_FFFF);
                let scales_8_11_bcast = ctx.shfl_idx_u32(scales_8_11, 0, 0xFFFF_FFFF);

                // Extract individual scale bytes
                let mask_8bit = ctx.mov_u32_imm(0xFF);
                let eight = ctx.mov_u32_imm(8);
                let sixteen = ctx.mov_u32_imm(16);
                let twenty_four = ctx.mov_u32_imm(24);

                let s0_32 = ctx.and_u32(scales_0_3_bcast, mask_8bit);
                let s0_shifted = ctx.shr_u32(scales_0_3_bcast, eight);
                let s1_32 = ctx.and_u32(s0_shifted, mask_8bit);
                let s1_shifted = ctx.shr_u32(scales_0_3_bcast, sixteen);
                let s2_32 = ctx.and_u32(s1_shifted, mask_8bit);
                let s3_32 = ctx.shr_u32(scales_0_3_bcast, twenty_four);

                let s4_32 = ctx.and_u32(scales_4_7_bcast, mask_8bit);
                let s4_shifted = ctx.shr_u32(scales_4_7_bcast, eight);
                let s5_32 = ctx.and_u32(s4_shifted, mask_8bit);
                let s5_shifted = ctx.shr_u32(scales_4_7_bcast, sixteen);
                let s6_32 = ctx.and_u32(s5_shifted, mask_8bit);
                let s7_32 = ctx.shr_u32(scales_4_7_bcast, twenty_four);

                let s8_32 = ctx.and_u32(scales_8_11_bcast, mask_8bit);
                let s8_shifted = ctx.shr_u32(scales_8_11_bcast, eight);
                let s9_32 = ctx.and_u32(s8_shifted, mask_8bit);
                let s9_shifted = ctx.shr_u32(scales_8_11_bcast, sixteen);
                let s10_32 = ctx.and_u32(s9_shifted, mask_8bit);
                let s11_32 = ctx.shr_u32(scales_8_11_bcast, twenty_four);

                // Constants for scale/min extraction
                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let mask_4bit = ctx.mov_u32_imm(0x0F);
                let four_const = ctx.mov_u32_imm(4);
                let six = ctx.mov_u32_imm(6);

                // Extract scale/min for all 8 blocks
                // Block 0-3: simple extraction
                let scale0 = ctx.and_u32(s0_32, mask_6bit);
                let min0 = ctx.and_u32(s4_32, mask_6bit);
                let scale1 = ctx.and_u32(s1_32, mask_6bit);
                let min1 = ctx.and_u32(s5_32, mask_6bit);
                let scale2 = ctx.and_u32(s2_32, mask_6bit);
                let min2 = ctx.and_u32(s6_32, mask_6bit);
                let scale3 = ctx.and_u32(s3_32, mask_6bit);
                let min3 = ctx.and_u32(s7_32, mask_6bit);

                // Block 4-7: complex extraction (6-bit packed)
                let s8_lo = ctx.and_u32(s8_32, mask_4bit);
                let s0_hi = ctx.shr_u32(s0_32, six);
                let s0_hi_shifted = ctx.shl_u32(s0_hi, four_const);
                let scale4 = ctx.or_u32(s8_lo, s0_hi_shifted);
                let s8_hi = ctx.shr_u32(s8_32, four_const);
                let s4_hi = ctx.shr_u32(s4_32, six);
                let s4_hi_shifted = ctx.shl_u32(s4_hi, four_const);
                let min4 = ctx.or_u32(s8_hi, s4_hi_shifted);

                let s9_lo = ctx.and_u32(s9_32, mask_4bit);
                let s1_hi = ctx.shr_u32(s1_32, six);
                let s1_hi_shifted = ctx.shl_u32(s1_hi, four_const);
                let scale5 = ctx.or_u32(s9_lo, s1_hi_shifted);
                let s9_hi = ctx.shr_u32(s9_32, four_const);
                let s5_hi = ctx.shr_u32(s5_32, six);
                let s5_hi_shifted = ctx.shl_u32(s5_hi, four_const);
                let min5 = ctx.or_u32(s9_hi, s5_hi_shifted);

                let s10_lo = ctx.and_u32(s10_32, mask_4bit);
                let s2_hi = ctx.shr_u32(s2_32, six);
                let s2_hi_shifted = ctx.shl_u32(s2_hi, four_const);
                let scale6 = ctx.or_u32(s10_lo, s2_hi_shifted);
                let s10_hi = ctx.shr_u32(s10_32, four_const);
                let s6_hi = ctx.shr_u32(s6_32, six);
                let s6_hi_shifted = ctx.shl_u32(s6_hi, four_const);
                let min6 = ctx.or_u32(s10_hi, s6_hi_shifted);

                let s11_lo = ctx.and_u32(s11_32, mask_4bit);
                let s3_hi = ctx.shr_u32(s3_32, six);
                let s3_hi_shifted = ctx.shl_u32(s3_hi, four_const);
                let scale7 = ctx.or_u32(s11_lo, s3_hi_shifted);
                let s11_hi = ctx.shr_u32(s11_32, four_const);
                let s7_hi = ctx.shr_u32(s7_32, six);
                let s7_hi_shifted = ctx.shl_u32(s7_hi, four_const);
                let min7 = ctx.or_u32(s11_hi, s7_hi_shifted);

                // Convert to f32 and precompute d*scale, dmin*min for all blocks
                let scale0_f = ctx.cvt_f32_u32(scale0);
                let min0_f = ctx.cvt_f32_u32(min0);
                let ds0 = ctx.mul_f32(d, scale0_f);
                let dm0 = ctx.mul_f32(dmin, min0_f);

                let scale1_f = ctx.cvt_f32_u32(scale1);
                let min1_f = ctx.cvt_f32_u32(min1);
                let ds1 = ctx.mul_f32(d, scale1_f);
                let dm1 = ctx.mul_f32(dmin, min1_f);

                let scale2_f = ctx.cvt_f32_u32(scale2);
                let min2_f = ctx.cvt_f32_u32(min2);
                let ds2 = ctx.mul_f32(d, scale2_f);
                let dm2 = ctx.mul_f32(dmin, min2_f);

                let scale3_f = ctx.cvt_f32_u32(scale3);
                let min3_f = ctx.cvt_f32_u32(min3);
                let ds3 = ctx.mul_f32(d, scale3_f);
                let dm3 = ctx.mul_f32(dmin, min3_f);

                let scale4_f = ctx.cvt_f32_u32(scale4);
                let min4_f = ctx.cvt_f32_u32(min4);
                let ds4 = ctx.mul_f32(d, scale4_f);
                let dm4 = ctx.mul_f32(dmin, min4_f);

                let scale5_f = ctx.cvt_f32_u32(scale5);
                let min5_f = ctx.cvt_f32_u32(min5);
                let ds5 = ctx.mul_f32(d, scale5_f);
                let dm5 = ctx.mul_f32(dmin, min5_f);

                let scale6_f = ctx.cvt_f32_u32(scale6);
                let min6_f = ctx.cvt_f32_u32(min6);
                let ds6 = ctx.mul_f32(d, scale6_f);
                let dm6 = ctx.mul_f32(dmin, min6_f);

                let scale7_f = ctx.cvt_f32_u32(scale7);
                let min7_f = ctx.cvt_f32_u32(min7);
                let ds7 = ctx.mul_f32(d, scale7_f);
                let dm7 = ctx.mul_f32(dmin, min7_f);

                // ============================================================
                // PAR-069: COALESCED WEIGHT LOADING
                // Each thread loads 4 consecutive bytes (u32) = 8 nibbles
                // 32 threads × 4 bytes = 128 bytes per warp (1 memory transaction!)
                // ============================================================
                let sixteen_64 = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen_64);

                let four = ctx.mov_u32_imm(4);
                let thread_byte_offset = ctx.mul_u32_reg(lane_id, four);
                let thread_byte_offset_64 = ctx.cvt_u64_u32(thread_byte_offset);
                let qs_addr = ctx.add_u64(qs_base, thread_byte_offset_64);

                // COALESCED u32 LOAD: 32 threads × 4 bytes = 128 bytes per transaction
                let packed_u32 = ctx.ld_global_u32(qs_addr);

                // Unpack 8 nibbles from 4 bytes
                let nib0 = ctx.and_u32(packed_u32, mask_4bit);
                let shift4 = ctx.mov_u32_imm(4);
                let nib1 = ctx.shr_u32(packed_u32, shift4);
                let nib1 = ctx.and_u32(nib1, mask_4bit);
                let shift8_const = ctx.mov_u32_imm(8);
                let nib2 = ctx.shr_u32(packed_u32, shift8_const);
                let nib2 = ctx.and_u32(nib2, mask_4bit);
                let shift12 = ctx.mov_u32_imm(12);
                let nib3 = ctx.shr_u32(packed_u32, shift12);
                let nib3 = ctx.and_u32(nib3, mask_4bit);
                let shift16_const = ctx.mov_u32_imm(16);
                let nib4 = ctx.shr_u32(packed_u32, shift16_const);
                let nib4 = ctx.and_u32(nib4, mask_4bit);
                let shift20 = ctx.mov_u32_imm(20);
                let nib5 = ctx.shr_u32(packed_u32, shift20);
                let nib5 = ctx.and_u32(nib5, mask_4bit);
                let shift24_const = ctx.mov_u32_imm(24);
                let nib6 = ctx.shr_u32(packed_u32, shift24_const);
                let nib6 = ctx.and_u32(nib6, mask_4bit);
                let shift28 = ctx.mov_u32_imm(28);
                let nib7 = ctx.shr_u32(packed_u32, shift28);

                // ============================================================
                // CORRECTNESS-002 FIX: Q4K DEINTERLEAVED NIBBLE LAYOUT
                //
                // Q4K stores 256 values per super-block in 4 chunks of 64 values:
                //   - qs[0..32]: low nibbles → values 0-31, high nibbles → values 32-63
                //   - qs[32..64]: low nibbles → values 64-95, high nibbles → values 96-127
                //   - qs[64..96]: low nibbles → values 128-159, high nibbles → values 160-191
                //   - qs[96..128]: low nibbles → values 192-223, high nibbles → values 224-255
                //
                // Thread t loads bytes t*4..t*4+3:
                //   - chunk = (t*4) / 32 = t / 8
                //   - Low nibbles need scale chunk*2, high nibbles need scale chunk*2+1
                //   - Low nibble activations: chunk*64 + (t*4 % 32) + byte_offset
                //   - High nibble activations: chunk*64 + 32 + (t*4 % 32) + byte_offset
                // ============================================================

                // Compute chunk index (which 64-value block we're in)
                let three_const = ctx.mov_u32_imm(3);
                let chunk_idx = ctx.shr_u32(lane_id, three_const); // lane_id / 8

                // Compute scale indices for low and high nibbles
                let low_scale_idx = ctx.shl_u32(chunk_idx, one); // chunk * 2
                let high_scale_idx = ctx.add_u32(low_scale_idx, 1); // chunk * 2 + 1

                // Select low scale (for nib0, nib2, nib4, nib6)
                let ds_low = ds0;
                let dm_low = dm0;
                let is_low1 = ctx.setp_eq_u32(low_scale_idx, one);
                let ds_low = ctx.selp_f32(is_low1, ds1, ds_low);
                let dm_low = ctx.selp_f32(is_low1, dm1, dm_low);
                let two_u32 = ctx.mov_u32_imm(2);
                let is_low2 = ctx.setp_eq_u32(low_scale_idx, two_u32);
                let ds_low = ctx.selp_f32(is_low2, ds2, ds_low);
                let dm_low = ctx.selp_f32(is_low2, dm2, dm_low);
                let three_u32 = ctx.mov_u32_imm(3);
                let is_low3 = ctx.setp_eq_u32(low_scale_idx, three_u32);
                let ds_low = ctx.selp_f32(is_low3, ds3, ds_low);
                let dm_low = ctx.selp_f32(is_low3, dm3, dm_low);
                let is_low4 = ctx.setp_eq_u32(low_scale_idx, four);
                let ds_low = ctx.selp_f32(is_low4, ds4, ds_low);
                let dm_low = ctx.selp_f32(is_low4, dm4, dm_low);
                let five_u32 = ctx.mov_u32_imm(5);
                let is_low5 = ctx.setp_eq_u32(low_scale_idx, five_u32);
                let ds_low = ctx.selp_f32(is_low5, ds5, ds_low);
                let dm_low = ctx.selp_f32(is_low5, dm5, dm_low);
                let six_u32 = ctx.mov_u32_imm(6);
                let is_low6 = ctx.setp_eq_u32(low_scale_idx, six_u32);
                let ds_low = ctx.selp_f32(is_low6, ds6, ds_low);
                let dm_low = ctx.selp_f32(is_low6, dm6, dm_low);
                let seven_u32 = ctx.mov_u32_imm(7);
                let is_low7 = ctx.setp_eq_u32(low_scale_idx, seven_u32);
                let ds_low = ctx.selp_f32(is_low7, ds7, ds_low);
                let dm_low = ctx.selp_f32(is_low7, dm7, dm_low);

                // Select high scale (for nib1, nib3, nib5, nib7)
                let ds_high = ds0;
                let dm_high = dm0;
                let is_high1 = ctx.setp_eq_u32(high_scale_idx, one);
                let ds_high = ctx.selp_f32(is_high1, ds1, ds_high);
                let dm_high = ctx.selp_f32(is_high1, dm1, dm_high);
                let is_high2 = ctx.setp_eq_u32(high_scale_idx, two_u32);
                let ds_high = ctx.selp_f32(is_high2, ds2, ds_high);
                let dm_high = ctx.selp_f32(is_high2, dm2, dm_high);
                let is_high3 = ctx.setp_eq_u32(high_scale_idx, three_u32);
                let ds_high = ctx.selp_f32(is_high3, ds3, ds_high);
                let dm_high = ctx.selp_f32(is_high3, dm3, dm_high);
                let is_high4 = ctx.setp_eq_u32(high_scale_idx, four);
                let ds_high = ctx.selp_f32(is_high4, ds4, ds_high);
                let dm_high = ctx.selp_f32(is_high4, dm4, dm_high);
                let is_high5 = ctx.setp_eq_u32(high_scale_idx, five_u32);
                let ds_high = ctx.selp_f32(is_high5, ds5, ds_high);
                let dm_high = ctx.selp_f32(is_high5, dm5, dm_high);
                let is_high6 = ctx.setp_eq_u32(high_scale_idx, six_u32);
                let ds_high = ctx.selp_f32(is_high6, ds6, ds_high);
                let dm_high = ctx.selp_f32(is_high6, dm6, dm_high);
                let is_high7 = ctx.setp_eq_u32(high_scale_idx, seven_u32);
                let ds_high = ctx.selp_f32(is_high7, ds7, ds_high);
                let dm_high = ctx.selp_f32(is_high7, dm7, dm_high);

                // Convert nibbles to f32
                let nib0_f = ctx.cvt_f32_u32(nib0);
                let nib1_f = ctx.cvt_f32_u32(nib1);
                let nib2_f = ctx.cvt_f32_u32(nib2);
                let nib3_f = ctx.cvt_f32_u32(nib3);
                let nib4_f = ctx.cvt_f32_u32(nib4);
                let nib5_f = ctx.cvt_f32_u32(nib5);
                let nib6_f = ctx.cvt_f32_u32(nib6);
                let nib7_f = ctx.cvt_f32_u32(nib7);

                // Dequantize with CORRECT scale selection:
                // - Low nibbles (0, 2, 4, 6) use ds_low/dm_low
                // - High nibbles (1, 3, 5, 7) use ds_high/dm_high
                let dq0 = ctx.mul_f32(ds_low, nib0_f);
                let dq0 = ctx.sub_f32(dq0, dm_low);
                let dq1 = ctx.mul_f32(ds_high, nib1_f); // HIGH nibble
                let dq1 = ctx.sub_f32(dq1, dm_high);
                let dq2 = ctx.mul_f32(ds_low, nib2_f);
                let dq2 = ctx.sub_f32(dq2, dm_low);
                let dq3 = ctx.mul_f32(ds_high, nib3_f); // HIGH nibble
                let dq3 = ctx.sub_f32(dq3, dm_high);
                let dq4 = ctx.mul_f32(ds_low, nib4_f);
                let dq4 = ctx.sub_f32(dq4, dm_low);
                let dq5 = ctx.mul_f32(ds_high, nib5_f); // HIGH nibble
                let dq5 = ctx.sub_f32(dq5, dm_high);
                let dq6 = ctx.mul_f32(ds_low, nib6_f);
                let dq6 = ctx.sub_f32(dq6, dm_low);
                let dq7 = ctx.mul_f32(ds_high, nib7_f); // HIGH nibble
                let dq7 = ctx.sub_f32(dq7, dm_high);

                // ============================================================
                // CORRECTNESS-002 FIX: CORRECT ACTIVATION INDICES
                //
                // Q4K deinterleaved layout:
                //   - Low nibbles from byte b map to value: chunk*64 + (b % 32)
                //   - High nibbles from byte b map to value: chunk*64 + 32 + (b % 32)
                //
                // Thread t loads bytes t*4, t*4+1, t*4+2, t*4+3
                // byte_in_chunk = (t*4) % 32 = (t % 8) * 4
                // ============================================================
                let sb_k_base = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                let sixty_four = ctx.mov_u32_imm(64);
                let chunk_base = ctx.mul_u32_reg(chunk_idx, sixty_four); // chunk * 64
                let chunk_start = ctx.add_u32_reg(sb_k_base, chunk_base); // sb_base + chunk*64

                // byte_in_chunk = (lane_id % 8) * 4
                let seven_mask = ctx.mov_u32_imm(7);
                let lane_in_chunk = ctx.and_u32(lane_id, seven_mask); // lane_id % 8
                let byte_in_chunk = ctx.shl_u32(lane_in_chunk, two_u32); // * 4

                // Base for low nibbles: chunk_start + byte_in_chunk
                let low_base = ctx.add_u32_reg(chunk_start, byte_in_chunk);
                // Base for high nibbles: chunk_start + 32 + byte_in_chunk
                let thirty_two = ctx.mov_u32_imm(32);
                let high_base = ctx.add_u32_reg(chunk_start, thirty_two);
                let high_base = ctx.add_u32_reg(high_base, byte_in_chunk);

                let thread_partial = ctx.mov_f32_imm(0.0);

                // LOW nibbles: values at low_base + 0, 1, 2, 3
                // nib0 (byte0 low) → x[low_base + 0]
                let x_idx0_64 = ctx.cvt_u64_u32(low_base);
                let x_off0 = ctx.mul_u64(x_idx0_64, 4);
                let x_addr0 = ctx.add_u64(x_ptr, x_off0);
                let x0 = ctx.ld_global_f32(x_addr0);
                ctx.fma_f32_inplace(thread_partial, x0, dq0);

                // nib2 (byte1 low) → x[low_base + 1]
                let x_idx2 = ctx.add_u32(low_base, 1);
                let x_idx2_64 = ctx.cvt_u64_u32(x_idx2);
                let x_off2 = ctx.mul_u64(x_idx2_64, 4);
                let x_addr2 = ctx.add_u64(x_ptr, x_off2);
                let x2 = ctx.ld_global_f32(x_addr2);
                ctx.fma_f32_inplace(thread_partial, x2, dq2);

                // nib4 (byte2 low) → x[low_base + 2]
                let x_idx4 = ctx.add_u32(low_base, 2);
                let x_idx4_64 = ctx.cvt_u64_u32(x_idx4);
                let x_off4 = ctx.mul_u64(x_idx4_64, 4);
                let x_addr4 = ctx.add_u64(x_ptr, x_off4);
                let x4 = ctx.ld_global_f32(x_addr4);
                ctx.fma_f32_inplace(thread_partial, x4, dq4);

                // nib6 (byte3 low) → x[low_base + 3]
                let x_idx6 = ctx.add_u32(low_base, 3);
                let x_idx6_64 = ctx.cvt_u64_u32(x_idx6);
                let x_off6 = ctx.mul_u64(x_idx6_64, 4);
                let x_addr6 = ctx.add_u64(x_ptr, x_off6);
                let x6 = ctx.ld_global_f32(x_addr6);
                ctx.fma_f32_inplace(thread_partial, x6, dq6);

                // HIGH nibbles: values at high_base + 0, 1, 2, 3
                // nib1 (byte0 high) → x[high_base + 0]
                let x_idx1_64 = ctx.cvt_u64_u32(high_base);
                let x_off1 = ctx.mul_u64(x_idx1_64, 4);
                let x_addr1 = ctx.add_u64(x_ptr, x_off1);
                let x1 = ctx.ld_global_f32(x_addr1);
                ctx.fma_f32_inplace(thread_partial, x1, dq1);

                // nib3 (byte1 high) → x[high_base + 1]
                let x_idx3 = ctx.add_u32(high_base, 1);
                let x_idx3_64 = ctx.cvt_u64_u32(x_idx3);
                let x_off3 = ctx.mul_u64(x_idx3_64, 4);
                let x_addr3 = ctx.add_u64(x_ptr, x_off3);
                let x3 = ctx.ld_global_f32(x_addr3);
                ctx.fma_f32_inplace(thread_partial, x3, dq3);

                // nib5 (byte2 high) → x[high_base + 2]
                let x_idx5 = ctx.add_u32(high_base, 2);
                let x_idx5_64 = ctx.cvt_u64_u32(x_idx5);
                let x_off5 = ctx.mul_u64(x_idx5_64, 4);
                let x_addr5 = ctx.add_u64(x_ptr, x_off5);
                let x5 = ctx.ld_global_f32(x_addr5);
                ctx.fma_f32_inplace(thread_partial, x5, dq5);

                // nib7 (byte3 high) → x[high_base + 3]
                let x_idx7 = ctx.add_u32(high_base, 3);
                let x_idx7_64 = ctx.cvt_u64_u32(x_idx7);
                let x_off7 = ctx.mul_u64(x_idx7_64, 4);
                let x_addr7 = ctx.add_u64(x_ptr, x_off7);
                let x7 = ctx.ld_global_f32(x_addr7);
                ctx.fma_f32_inplace(thread_partial, x7, dq7);

                ctx.add_f32_inplace(acc, thread_partial);
                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop_v");

                ctx.label("sb_loop_end_v");

                // Warp shuffle reduction
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
                let is_lane0_final = ctx.setp_lt_u32(lane_id, one);
                ctx.branch_if_not(is_lane0_final, "exit");

                let y_offset = ctx.mul_wide_u32(block_id, 4);
                let y_addr = ctx.add_u64(y_ptr, y_offset);
                ctx.st_global_f32(y_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Multi-warp Vectorized Q4_K GEMV kernel (PAR-082-V2)
///
/// Combines the best of both approaches:
/// - **VectorizedQ4K**: u32 coalesced loads (128 bytes/warp/transaction)
/// - **WideQ4K**: Multi-warp parallelism for memory latency hiding
///
/// 4 warps (128 threads) per block, matching llama.cpp `mul_mat_vec_q`.
/// Each warp strides across super-blocks, uses u32 coalesced loads,
/// then cross-warp reduction via shared memory.
///
/// # Performance Target
///
/// 50% BW utilization → ~128 tok/s on RTX 4090 for 7B Q4K (Ollama parity)
pub struct MultiWarpVectorizedQ4KGemvKernel {
    /// K dimension (input dimension, need not be multiple of 256 — bounds-checked)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Number of warps per block (default: 4, matching llama.cpp)
    pub num_warps: u32,
}

impl MultiWarpVectorizedQ4KGemvKernel {
    /// Create with 4 warps (128 threads), matching llama.cpp
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n, num_warps: 4 }
    }
}

impl Kernel for MultiWarpVectorizedQ4KGemvKernel {
    fn name(&self) -> &str {
        "mwv_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let smem_size = (num_warps * 4) as usize;

        PtxKernel::new("mwv_q4k_gemv")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .shared_memory(smem_size)
            .build(move |ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);

                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "mwv_exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let x_ptr = ctx.load_param_u64("x_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                let sb_bytes_c = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes_c);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Each warp starts at warp_id, strides by num_warps
                let sb_idx_z = ctx.mov_u32_imm(0);
                let sb_idx = ctx.add_u32_reg(sb_idx_z, warp_id);
                let nw_reg = ctx.mov_u32_imm(num_warps);

                ctx.label("mwv_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "mwv_sb_end");

                let sb_off = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                // Load d and dmin
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two_64 = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two_64);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // Scale loading: lane 0 loads 3 x u32, broadcasts
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);
                let one = ctx.mov_u32_imm(1);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);

                let sc03r = ctx.mov_u32_imm(0);
                let sc47r = ctx.mov_u32_imm(0);
                let sc811r = ctx.mov_u32_imm(0);

                ctx.branch_if_not(is_lane0, "mwv_skip_sc");
                ctx.ld_global_u32_into(sc03r, scales_base);
                let f64b = ctx.mov_u64_imm(4);
                let s4a = ctx.add_u64(scales_base, f64b);
                ctx.ld_global_u32_into(sc47r, s4a);
                let e64 = ctx.mov_u64_imm(8);
                let s8a = ctx.add_u64(scales_base, e64);
                ctx.ld_global_u32_into(sc811r, s8a);
                ctx.label("mwv_skip_sc");

                let sc03 = ctx.shfl_idx_u32(sc03r, 0, 0xFFFF_FFFF);
                let sc47 = ctx.shfl_idx_u32(sc47r, 0, 0xFFFF_FFFF);
                let sc811 = ctx.shfl_idx_u32(sc811r, 0, 0xFFFF_FFFF);

                let m8 = ctx.mov_u32_imm(0xFF);
                let sh8 = ctx.mov_u32_imm(8);
                let sh16 = ctx.mov_u32_imm(16);
                let sh24 = ctx.mov_u32_imm(24);

                let s0 = ctx.and_u32(sc03, m8);
                let t = ctx.shr_u32(sc03, sh8);
                let s1 = ctx.and_u32(t, m8);
                let t = ctx.shr_u32(sc03, sh16);
                let s2 = ctx.and_u32(t, m8);
                let s3 = ctx.shr_u32(sc03, sh24);

                let s4 = ctx.and_u32(sc47, m8);
                let t = ctx.shr_u32(sc47, sh8);
                let s5 = ctx.and_u32(t, m8);
                let t = ctx.shr_u32(sc47, sh16);
                let s6 = ctx.and_u32(t, m8);
                let s7 = ctx.shr_u32(sc47, sh24);

                let s8 = ctx.and_u32(sc811, m8);
                let t = ctx.shr_u32(sc811, sh8);
                let s9 = ctx.and_u32(t, m8);
                let t = ctx.shr_u32(sc811, sh16);
                let s10 = ctx.and_u32(t, m8);
                let s11 = ctx.shr_u32(sc811, sh24);

                let m6 = ctx.mov_u32_imm(0x3F);
                let m4 = ctx.mov_u32_imm(0x0F);
                let four_c = ctx.mov_u32_imm(4);
                let six_c = ctx.mov_u32_imm(6);

                // Blocks 0-3
                let sc0 = ctx.and_u32(s0, m6);
                let mn0 = ctx.and_u32(s4, m6);
                let sc1 = ctx.and_u32(s1, m6);
                let mn1 = ctx.and_u32(s5, m6);
                let sc2 = ctx.and_u32(s2, m6);
                let mn2 = ctx.and_u32(s6, m6);
                let sc3 = ctx.and_u32(s3, m6);
                let mn3 = ctx.and_u32(s7, m6);

                // Blocks 4-7
                let t = ctx.and_u32(s8, m4);
                let u = ctx.shr_u32(s0, six_c);
                let u = ctx.shl_u32(u, four_c);
                let sc4 = ctx.or_u32(t, u);
                let t = ctx.shr_u32(s8, four_c);
                let u = ctx.shr_u32(s4, six_c);
                let u = ctx.shl_u32(u, four_c);
                let mn4 = ctx.or_u32(t, u);

                let t = ctx.and_u32(s9, m4);
                let u = ctx.shr_u32(s1, six_c);
                let u = ctx.shl_u32(u, four_c);
                let sc5 = ctx.or_u32(t, u);
                let t = ctx.shr_u32(s9, four_c);
                let u = ctx.shr_u32(s5, six_c);
                let u = ctx.shl_u32(u, four_c);
                let mn5 = ctx.or_u32(t, u);

                let t = ctx.and_u32(s10, m4);
                let u = ctx.shr_u32(s2, six_c);
                let u = ctx.shl_u32(u, four_c);
                let sc6 = ctx.or_u32(t, u);
                let t = ctx.shr_u32(s10, four_c);
                let u = ctx.shr_u32(s6, six_c);
                let u = ctx.shl_u32(u, four_c);
                let mn6 = ctx.or_u32(t, u);

                let t = ctx.and_u32(s11, m4);
                let u = ctx.shr_u32(s3, six_c);
                let u = ctx.shl_u32(u, four_c);
                let sc7 = ctx.or_u32(t, u);
                let t = ctx.shr_u32(s11, four_c);
                let u = ctx.shr_u32(s7, six_c);
                let u = ctx.shl_u32(u, four_c);
                let mn7 = ctx.or_u32(t, u);

                // d*scale and dmin*min
                let f0 = ctx.cvt_f32_u32(sc0);
                let g0 = ctx.cvt_f32_u32(mn0);
                let ds0 = ctx.mul_f32(d, f0);
                let dm0 = ctx.mul_f32(dmin, g0);
                let f1 = ctx.cvt_f32_u32(sc1);
                let g1 = ctx.cvt_f32_u32(mn1);
                let ds1 = ctx.mul_f32(d, f1);
                let dm1 = ctx.mul_f32(dmin, g1);
                let f2 = ctx.cvt_f32_u32(sc2);
                let g2 = ctx.cvt_f32_u32(mn2);
                let ds2 = ctx.mul_f32(d, f2);
                let dm2 = ctx.mul_f32(dmin, g2);
                let f3 = ctx.cvt_f32_u32(sc3);
                let g3 = ctx.cvt_f32_u32(mn3);
                let ds3 = ctx.mul_f32(d, f3);
                let dm3 = ctx.mul_f32(dmin, g3);
                let f4 = ctx.cvt_f32_u32(sc4);
                let g4 = ctx.cvt_f32_u32(mn4);
                let ds4 = ctx.mul_f32(d, f4);
                let dm4 = ctx.mul_f32(dmin, g4);
                let f5 = ctx.cvt_f32_u32(sc5);
                let g5 = ctx.cvt_f32_u32(mn5);
                let ds5 = ctx.mul_f32(d, f5);
                let dm5 = ctx.mul_f32(dmin, g5);
                let f6 = ctx.cvt_f32_u32(sc6);
                let g6 = ctx.cvt_f32_u32(mn6);
                let ds6 = ctx.mul_f32(d, f6);
                let dm6 = ctx.mul_f32(dmin, g6);
                let f7 = ctx.cvt_f32_u32(sc7);
                let g7 = ctx.cvt_f32_u32(mn7);
                let ds7 = ctx.mul_f32(d, f7);
                let dm7 = ctx.mul_f32(dmin, g7);

                // COALESCED u32 LOAD: 128 bytes per warp transaction
                let sixteen_64 = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen_64);
                let four = ctx.mov_u32_imm(4);
                let tbo = ctx.mul_u32_reg(lane_id, four);
                let tbo64 = ctx.cvt_u64_u32(tbo);
                let qa = ctx.add_u64(qs_base, tbo64);
                let packed = ctx.ld_global_u32(qa);

                // Unpack 8 nibbles
                let nib0 = ctx.and_u32(packed, m4);
                let sh4 = ctx.mov_u32_imm(4);
                let nib1 = ctx.shr_u32(packed, sh4);
                let nib1 = ctx.and_u32(nib1, m4);
                let nib2 = ctx.shr_u32(packed, sh8);
                let nib2 = ctx.and_u32(nib2, m4);
                let s12 = ctx.mov_u32_imm(12);
                let nib3 = ctx.shr_u32(packed, s12);
                let nib3 = ctx.and_u32(nib3, m4);
                let nib4 = ctx.shr_u32(packed, sh16);
                let nib4 = ctx.and_u32(nib4, m4);
                let s20 = ctx.mov_u32_imm(20);
                let nib5 = ctx.shr_u32(packed, s20);
                let nib5 = ctx.and_u32(nib5, m4);
                let nib6 = ctx.shr_u32(packed, sh24);
                let nib6 = ctx.and_u32(nib6, m4);
                let s28 = ctx.mov_u32_imm(28);
                let nib7 = ctx.shr_u32(packed, s28);

                // Chunk and scale indices (deinterleaved)
                let three_c = ctx.mov_u32_imm(3);
                let ci = ctx.shr_u32(lane_id, three_c);
                let lsi = ctx.shl_u32(ci, one);
                let hsi = ctx.add_u32(lsi, 1);

                // Select low scale
                let dl = ds0;
                let ml = dm0;
                let p = ctx.setp_eq_u32(lsi, one);
                let dl = ctx.selp_f32(p, ds1, dl);
                let ml = ctx.selp_f32(p, dm1, ml);
                let two_u = ctx.mov_u32_imm(2);
                let p = ctx.setp_eq_u32(lsi, two_u);
                let dl = ctx.selp_f32(p, ds2, dl);
                let ml = ctx.selp_f32(p, dm2, ml);
                let three_u = ctx.mov_u32_imm(3);
                let p = ctx.setp_eq_u32(lsi, three_u);
                let dl = ctx.selp_f32(p, ds3, dl);
                let ml = ctx.selp_f32(p, dm3, ml);
                let p = ctx.setp_eq_u32(lsi, four);
                let dl = ctx.selp_f32(p, ds4, dl);
                let ml = ctx.selp_f32(p, dm4, ml);
                let five_u = ctx.mov_u32_imm(5);
                let p = ctx.setp_eq_u32(lsi, five_u);
                let dl = ctx.selp_f32(p, ds5, dl);
                let ml = ctx.selp_f32(p, dm5, ml);
                let six_u = ctx.mov_u32_imm(6);
                let p = ctx.setp_eq_u32(lsi, six_u);
                let dl = ctx.selp_f32(p, ds6, dl);
                let ml = ctx.selp_f32(p, dm6, ml);
                let seven_u = ctx.mov_u32_imm(7);
                let p = ctx.setp_eq_u32(lsi, seven_u);
                let dl = ctx.selp_f32(p, ds7, dl);
                let ml = ctx.selp_f32(p, dm7, ml);

                // Select high scale
                let dh = ds0;
                let mh = dm0;
                let p = ctx.setp_eq_u32(hsi, one);
                let dh = ctx.selp_f32(p, ds1, dh);
                let mh = ctx.selp_f32(p, dm1, mh);
                let p = ctx.setp_eq_u32(hsi, two_u);
                let dh = ctx.selp_f32(p, ds2, dh);
                let mh = ctx.selp_f32(p, dm2, mh);
                let p = ctx.setp_eq_u32(hsi, three_u);
                let dh = ctx.selp_f32(p, ds3, dh);
                let mh = ctx.selp_f32(p, dm3, mh);
                let p = ctx.setp_eq_u32(hsi, four);
                let dh = ctx.selp_f32(p, ds4, dh);
                let mh = ctx.selp_f32(p, dm4, mh);
                let p = ctx.setp_eq_u32(hsi, five_u);
                let dh = ctx.selp_f32(p, ds5, dh);
                let mh = ctx.selp_f32(p, dm5, mh);
                let p = ctx.setp_eq_u32(hsi, six_u);
                let dh = ctx.selp_f32(p, ds6, dh);
                let mh = ctx.selp_f32(p, dm6, mh);
                let p = ctx.setp_eq_u32(hsi, seven_u);
                let dh = ctx.selp_f32(p, ds7, dh);
                let mh = ctx.selp_f32(p, dm7, mh);

                // Nibbles to f32
                let n0f = ctx.cvt_f32_u32(nib0);
                let n1f = ctx.cvt_f32_u32(nib1);
                let n2f = ctx.cvt_f32_u32(nib2);
                let n3f = ctx.cvt_f32_u32(nib3);
                let n4f = ctx.cvt_f32_u32(nib4);
                let n5f = ctx.cvt_f32_u32(nib5);
                let n6f = ctx.cvt_f32_u32(nib6);
                let n7f = ctx.cvt_f32_u32(nib7);

                // Dequantize
                let dq0 = ctx.mul_f32(dl, n0f);
                let dq0 = ctx.sub_f32(dq0, ml);
                let dq1 = ctx.mul_f32(dh, n1f);
                let dq1 = ctx.sub_f32(dq1, mh);
                let dq2 = ctx.mul_f32(dl, n2f);
                let dq2 = ctx.sub_f32(dq2, ml);
                let dq3 = ctx.mul_f32(dh, n3f);
                let dq3 = ctx.sub_f32(dq3, mh);
                let dq4 = ctx.mul_f32(dl, n4f);
                let dq4 = ctx.sub_f32(dq4, ml);
                let dq5 = ctx.mul_f32(dh, n5f);
                let dq5 = ctx.sub_f32(dq5, mh);
                let dq6 = ctx.mul_f32(dl, n6f);
                let dq6 = ctx.sub_f32(dq6, ml);
                let dq7 = ctx.mul_f32(dh, n7f);
                let dq7 = ctx.sub_f32(dq7, mh);

                // Activation indices (deinterleaved)
                let skb = ctx.mul_u32(sb_idx, Q4K_SUPER_BLOCK_SIZE);
                let s64 = ctx.mov_u32_imm(64);
                let cb = ctx.mul_u32_reg(ci, s64);
                let cs = ctx.add_u32_reg(skb, cb);

                let sm = ctx.mov_u32_imm(7);
                let lic = ctx.and_u32(lane_id, sm);
                let bic = ctx.shl_u32(lic, two_u);

                let lb = ctx.add_u32_reg(cs, bic);
                let s32 = ctx.mov_u32_imm(32);
                let hb = ctx.add_u32_reg(cs, s32);
                let hb = ctx.add_u32_reg(hb, bic);

                let pt = ctx.mov_f32_imm(0.0);

                // GH-215 FIX: Bounds-check activation loads for non-256-aligned K.
                // When K is not a multiple of 256, the last super-block has
                // padding elements that would cause OOB reads into uninitialized
                // GPU memory, producing garbage output.

                // LOW nibbles
                let lb64 = ctx.cvt_u64_u32(lb);
                let xo = ctx.mul_u64(lb64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_lb0 = ctx.setp_lt_u32(lb, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_lb0, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq0);

                let v = ctx.add_u32(lb, 1);
                let v64 = ctx.cvt_u64_u32(v);
                let xo = ctx.mul_u64(v64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_lb1 = ctx.setp_lt_u32(v, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_lb1, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq2);

                let v = ctx.add_u32(lb, 2);
                let v64 = ctx.cvt_u64_u32(v);
                let xo = ctx.mul_u64(v64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_lb2 = ctx.setp_lt_u32(v, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_lb2, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq4);

                let v = ctx.add_u32(lb, 3);
                let v64 = ctx.cvt_u64_u32(v);
                let xo = ctx.mul_u64(v64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_lb3 = ctx.setp_lt_u32(v, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_lb3, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq6);

                // HIGH nibbles
                let hb64 = ctx.cvt_u64_u32(hb);
                let xo = ctx.mul_u64(hb64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_hb0 = ctx.setp_lt_u32(hb, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_hb0, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq1);

                let v = ctx.add_u32(hb, 1);
                let v64 = ctx.cvt_u64_u32(v);
                let xo = ctx.mul_u64(v64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_hb1 = ctx.setp_lt_u32(v, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_hb1, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq3);

                let v = ctx.add_u32(hb, 2);
                let v64 = ctx.cvt_u64_u32(v);
                let xo = ctx.mul_u64(v64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_hb2 = ctx.setp_lt_u32(v, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_hb2, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq5);

                let v = ctx.add_u32(hb, 3);
                let v64 = ctx.cvt_u64_u32(v);
                let xo = ctx.mul_u64(v64, 4);
                let xa = ctx.add_u64(x_ptr, xo);
                let p_hb3 = ctx.setp_lt_u32(v, k_dim);
                let xv = ctx.ld_global_f32_predicated(xa, p_hb3, 0.0);
                ctx.fma_f32_inplace(pt, xv, dq7);

                ctx.add_f32_inplace(acc, pt);

                // Stride by num_warps
                ctx.add_u32_reg_inplace(sb_idx, nw_reg);
                ctx.branch("mwv_sb_loop");

                ctx.label("mwv_sb_end");

                // Phase 1: Intra-warp reduction
                let t16 = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t16);
                let t8 = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t8);
                let t4 = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t4);
                let t2 = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t2);
                let t1 = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t1);

                // Phase 2: Cross-warp reduction via shared memory
                let z = ctx.mov_u32_imm(0);
                let is_l0 = ctx.setp_eq_u32(lane_id, z);
                ctx.branch_if_not(is_l0, "mwv_skip_sm");

                let f4 = ctx.mov_u32_imm(4);
                let wo = ctx.mul_u32_reg(warp_id, f4);
                let sa = ctx.cvt_u64_u32(wo);
                ctx.st_shared_f32(sa, acc);

                ctx.label("mwv_skip_sm");
                ctx.bar_sync(0);

                let is_t0 = ctx.setp_eq_u32(thread_id, z);
                ctx.branch_if_not(is_t0, "mwv_exit");

                let fs = ctx.mov_f32_imm(0.0);
                for w in 0..num_warps {
                    let wo = ctx.mov_u64_imm(u64::from(w * 4));
                    let pv = ctx.ld_shared_f32(wo);
                    ctx.add_f32_inplace(fs, pv);
                }

                let yo = ctx.mul_wide_u32(block_id, 4);
                let ya = ctx.add_u64(y_ptr, yo);
                ctx.st_global_f32(ya, fs);

                ctx.label("mwv_exit");
                ctx.ret();
            })
    }
}

/// Multi-warp DP4A Q4_K GEMV kernel (PAR-082-V4)
///
/// Combines MWV multi-warp parallelism with DP4A integer dot products:
/// - **Activations**: Pre-quantized to Q8_1 format (36 bytes/block: 32 qs + f16 scale + f16 sum)
/// - **DP4A**: `dp4a.u32.s32` computes 4 multiply-adds of u4×s8 per instruction
/// - **Instruction reduction**: ~65 → ~20 instructions per inner loop iteration (3.3x)
/// - **Memory reduction**: 8 scattered f32 loads → 2 coalesced u32 Q8 loads
///
/// # Q4K × Q8_1 dot product
///
/// For each sub-block of 32 values:
///   `result = q8_d * (d * scale * dp4a_sum - dmin * min * q8_byte_sum)`
/// where `dp4a_sum = Σ(nibble_i × q8_byte_i)` and `q8_byte_sum = Σ(q8_byte_i)`.
///
/// Reference: NVIDIA PTX ISA §9.7.8.11 dp4a (Compute Capability ≥ 6.1)
/// Citation: Markidis et al., "NVIDIA Tensor Core Programmability, Performance &
///           Precision," IEEE IPDPSW 2018, DOI:10.1109/IPDPSW.2018.00091
pub struct MwvDp4aQ4KGemvKernel {
    /// K dimension (input dimension, must be multiple of 256)
    pub k: u32,
    /// N dimension (output dimension)
    pub n: u32,
    /// Number of warps per block (default: 3)
    pub num_warps: u32,
}

impl MwvDp4aQ4KGemvKernel {
    /// Create with 3 warps (96 threads), empirically optimal on RTX 4090
    #[must_use]
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n, num_warps: 3 }
    }
}

impl Kernel for MwvDp4aQ4KGemvKernel {
    fn name(&self) -> &str {
        "mwv_dp4a_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let num_warps = self.num_warps;
        let smem_size = (num_warps * 4) as usize;

        PtxKernel::new("mwv_dp4a_q4k_gemv")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "w_ptr")
            .param(PtxType::U64, "q8_ptr") // Q8_1 quantized activations (NOT f32)
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .shared_memory(smem_size)
            .build(move |ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);
                let warp_id = ctx.div_u32(thread_id, 32);

                let n_dim = ctx.load_param_u32("n_dim");
                let oob = ctx.setp_ge_u32(block_id, n_dim);
                ctx.branch_if(oob, "dp4a_exit");

                let k_dim = ctx.load_param_u32("k_dim");
                let y_ptr = ctx.load_param_u64("y_ptr");
                let w_ptr = ctx.load_param_u64("w_ptr");
                let q8_ptr = ctx.load_param_u64("q8_ptr");

                let acc = ctx.mov_f32_imm(0.0);

                let k_rounded = ctx.add_u32(k_dim, Q4K_SUPER_BLOCK_SIZE - 1);
                let num_super_blocks = ctx.div_u32(k_rounded, Q4K_SUPER_BLOCK_SIZE);

                let sb_bytes_c = ctx.mov_u32_imm(Q4K_SUPER_BLOCK_BYTES);
                let row_bytes = ctx.mul_u32_reg(num_super_blocks, sb_bytes_c);
                let row_offset = ctx.mul_wide_u32_reg(block_id, row_bytes);
                let row_base = ctx.add_u64(w_ptr, row_offset);

                // Each warp starts at warp_id, strides by num_warps
                let sb_idx_z = ctx.mov_u32_imm(0);
                let sb_idx = ctx.add_u32_reg(sb_idx_z, warp_id);
                let nw_reg = ctx.mov_u32_imm(num_warps);

                ctx.label("dp4a_sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_super_blocks);
                ctx.branch_if(sb_done, "dp4a_sb_end");

                let sb_off = ctx.mul_wide_u32(sb_idx, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(row_base, sb_off);

                // Load d and dmin (same as MWV)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);
                let two_64 = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two_64);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // Scale loading: lane 0 loads 3 x u32, broadcasts (same as MWV)
                let four_64 = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four_64);
                let one = ctx.mov_u32_imm(1);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);

                let sc03r = ctx.mov_u32_imm(0);
                let sc47r = ctx.mov_u32_imm(0);
                let sc811r = ctx.mov_u32_imm(0);

                ctx.branch_if_not(is_lane0, "dp4a_skip_sc");
                ctx.ld_global_u32_into(sc03r, scales_base);
                let f64b = ctx.mov_u64_imm(4);
                let s4a = ctx.add_u64(scales_base, f64b);
                ctx.ld_global_u32_into(sc47r, s4a);
                let e64 = ctx.mov_u64_imm(8);
                let s8a = ctx.add_u64(scales_base, e64);
                ctx.ld_global_u32_into(sc811r, s8a);
                ctx.label("dp4a_skip_sc");

                let sc03 = ctx.shfl_idx_u32(sc03r, 0, 0xFFFF_FFFF);
                let sc47 = ctx.shfl_idx_u32(sc47r, 0, 0xFFFF_FFFF);
                let sc811 = ctx.shfl_idx_u32(sc811r, 0, 0xFFFF_FFFF);

                // Decode 8 scale/min pairs (identical to MWV)
                let m8 = ctx.mov_u32_imm(0xFF);
                let sh8 = ctx.mov_u32_imm(8);
                let sh16 = ctx.mov_u32_imm(16);
                let sh24 = ctx.mov_u32_imm(24);

                let s0 = ctx.and_u32(sc03, m8);
                let t = ctx.shr_u32(sc03, sh8);
                let s1 = ctx.and_u32(t, m8);
                let t = ctx.shr_u32(sc03, sh16);
                let s2 = ctx.and_u32(t, m8);
                let s3 = ctx.shr_u32(sc03, sh24);

                let s4 = ctx.and_u32(sc47, m8);
                let t = ctx.shr_u32(sc47, sh8);
                let s5 = ctx.and_u32(t, m8);
                let t = ctx.shr_u32(sc47, sh16);
                let s6 = ctx.and_u32(t, m8);
                let s7 = ctx.shr_u32(sc47, sh24);

                let s8 = ctx.and_u32(sc811, m8);
                let t = ctx.shr_u32(sc811, sh8);
                let s9 = ctx.and_u32(t, m8);
                let t = ctx.shr_u32(sc811, sh16);
                let s10 = ctx.and_u32(t, m8);
                let s11 = ctx.shr_u32(sc811, sh24);

                let m6 = ctx.mov_u32_imm(0x3F);
                let m4 = ctx.mov_u32_imm(0x0F);
                let four_c = ctx.mov_u32_imm(4);
                let six_c = ctx.mov_u32_imm(6);

                // Blocks 0-3
                let sc0 = ctx.and_u32(s0, m6);
                let mn0 = ctx.and_u32(s4, m6);
                let sc1 = ctx.and_u32(s1, m6);
                let mn1 = ctx.and_u32(s5, m6);
                let sc2 = ctx.and_u32(s2, m6);
                let mn2 = ctx.and_u32(s6, m6);
                let sc3 = ctx.and_u32(s3, m6);
                let mn3 = ctx.and_u32(s7, m6);

                // Blocks 4-7
                let t = ctx.and_u32(s8, m4);
                let u = ctx.shr_u32(s0, six_c);
                let u = ctx.shl_u32(u, four_c);
                let sc4 = ctx.or_u32(t, u);
                let t = ctx.shr_u32(s8, four_c);
                let u = ctx.shr_u32(s4, six_c);
                let u = ctx.shl_u32(u, four_c);
                let mn4 = ctx.or_u32(t, u);

                let t = ctx.and_u32(s9, m4);
                let u = ctx.shr_u32(s1, six_c);
                let u = ctx.shl_u32(u, four_c);
                let sc5 = ctx.or_u32(t, u);
                let t = ctx.shr_u32(s9, four_c);
                let u = ctx.shr_u32(s5, six_c);
                let u = ctx.shl_u32(u, four_c);
                let mn5 = ctx.or_u32(t, u);

                let t = ctx.and_u32(s10, m4);
                let u = ctx.shr_u32(s2, six_c);
                let u = ctx.shl_u32(u, four_c);
                let sc6 = ctx.or_u32(t, u);
                let t = ctx.shr_u32(s10, four_c);
                let u = ctx.shr_u32(s6, six_c);
                let u = ctx.shl_u32(u, four_c);
                let mn6 = ctx.or_u32(t, u);

                let t = ctx.and_u32(s11, m4);
                let u = ctx.shr_u32(s3, six_c);
                let u = ctx.shl_u32(u, four_c);
                let sc7 = ctx.or_u32(t, u);
                let t = ctx.shr_u32(s11, four_c);
                let u = ctx.shr_u32(s7, six_c);
                let u = ctx.shl_u32(u, four_c);
                let mn7 = ctx.or_u32(t, u);

                // d*scale and dmin*min (same as MWV)
                let f0 = ctx.cvt_f32_u32(sc0);
                let g0 = ctx.cvt_f32_u32(mn0);
                let ds0 = ctx.mul_f32(d, f0);
                let dm0 = ctx.mul_f32(dmin, g0);
                let f1 = ctx.cvt_f32_u32(sc1);
                let g1 = ctx.cvt_f32_u32(mn1);
                let ds1 = ctx.mul_f32(d, f1);
                let dm1 = ctx.mul_f32(dmin, g1);
                let f2 = ctx.cvt_f32_u32(sc2);
                let g2 = ctx.cvt_f32_u32(mn2);
                let ds2 = ctx.mul_f32(d, f2);
                let dm2 = ctx.mul_f32(dmin, g2);
                let f3 = ctx.cvt_f32_u32(sc3);
                let g3 = ctx.cvt_f32_u32(mn3);
                let ds3 = ctx.mul_f32(d, f3);
                let dm3 = ctx.mul_f32(dmin, g3);
                let f4 = ctx.cvt_f32_u32(sc4);
                let g4 = ctx.cvt_f32_u32(mn4);
                let ds4 = ctx.mul_f32(d, f4);
                let dm4 = ctx.mul_f32(dmin, g4);
                let f5 = ctx.cvt_f32_u32(sc5);
                let g5 = ctx.cvt_f32_u32(mn5);
                let ds5 = ctx.mul_f32(d, f5);
                let dm5 = ctx.mul_f32(dmin, g5);
                let f6 = ctx.cvt_f32_u32(sc6);
                let g6 = ctx.cvt_f32_u32(mn6);
                let ds6 = ctx.mul_f32(d, f6);
                let dm6 = ctx.mul_f32(dmin, g6);
                let f7 = ctx.cvt_f32_u32(sc7);
                let g7 = ctx.cvt_f32_u32(mn7);
                let ds7 = ctx.mul_f32(d, f7);
                let dm7 = ctx.mul_f32(dmin, g7);

                // === DP4A HOT PATH (replaces nibble unpack + f32 loads + FMA) ===

                // COALESCED u32 LOAD of Q4K weights (same as MWV)
                let sixteen_64 = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen_64);
                let four = ctx.mov_u32_imm(4);
                let tbo = ctx.mul_u32_reg(lane_id, four);
                let tbo64 = ctx.cvt_u64_u32(tbo);
                let qa = ctx.add_u64(qs_base, tbo64);
                let packed = ctx.ld_global_u32(qa);

                // Extract nibble packs: low = 4 unsigned bytes, high = 4 unsigned bytes
                let mask_0f = ctx.mov_u32_imm(0x0F0F_0F0F);
                let sh4 = ctx.mov_u32_imm(4);
                let low_nibs = ctx.and_u32(packed, mask_0f);
                let high_shifted = ctx.shr_u32(packed, sh4);
                let high_nibs = ctx.and_u32(high_shifted, mask_0f);

                // Chunk index: ci = lane_id / 8 (which sub-block pair)
                let three_c = ctx.mov_u32_imm(3);
                let ci = ctx.shr_u32(lane_id, three_c);
                // Local index within chunk: lic = lane_id & 7
                let sm = ctx.mov_u32_imm(7);
                let lic = ctx.and_u32(lane_id, sm);

                // Q8 block indices:
                //   LOW sub-block: sb_idx * 8 + ci * 2
                //   HIGH sub-block: sb_idx * 8 + ci * 2 + 1
                let eight_c = ctx.mov_u32_imm(8);
                let sb8 = ctx.mul_u32_reg(sb_idx, eight_c);
                let ci2 = ctx.shl_u32(ci, one);
                let blk_low = ctx.add_u32_reg(sb8, ci2);
                let blk_high = ctx.add_u32(blk_low, 1);

                // Q8_1 block stride = 36 bytes
                let thirty_six = ctx.mov_u32_imm(36);
                let q8_off_low = ctx.mul_wide_u32_reg(blk_low, thirty_six);
                let q8_off_high = ctx.mul_wide_u32_reg(blk_high, thirty_six);

                // Q8 qs offset within block = lic * 4
                let lic_x4 = ctx.mul_u32_reg(lic, four);
                let lic_x4_64 = ctx.cvt_u64_u32(lic_x4);

                // Load Q8 packed bytes (LOW sub-block)
                let q8_base_low = ctx.add_u64(q8_ptr, q8_off_low);
                let q8_addr_low = ctx.add_u64(q8_base_low, lic_x4_64);
                let q8_low = ctx.ld_global_u32(q8_addr_low);

                // Load Q8 packed bytes (HIGH sub-block)
                let q8_base_high = ctx.add_u64(q8_ptr, q8_off_high);
                let q8_addr_high = ctx.add_u64(q8_base_high, lic_x4_64);
                let q8_high = ctx.ld_global_u32(q8_addr_high);

                // DP4A: dot product of nibbles × q8 bytes
                let dot_low = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(dot_low, low_nibs, q8_low);
                let dot_high = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(dot_high, high_nibs, q8_high);

                // DP4A: sum of Q8 bytes (for min term)
                let ones = ctx.mov_u32_imm(0x0101_0101);
                let sum_low = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(sum_low, ones, q8_low);
                let sum_high = ctx.mov_u32_imm(0);
                ctx.dp4a_u32_s32_inplace(sum_high, ones, q8_high);

                // Load Q8 scales (d = f16 at offset 32 within Q8 block)
                let thirty_two_64 = ctx.mov_u64_imm(32);
                let q8_d_addr_low = ctx.add_u64(q8_base_low, thirty_two_64);
                let q8_d_low_f16 = ctx.ld_global_f16(q8_d_addr_low);
                let q8_d_low = ctx.cvt_f32_f16(q8_d_low_f16);

                let q8_d_addr_high = ctx.add_u64(q8_base_high, thirty_two_64);
                let q8_d_high_f16 = ctx.ld_global_f16(q8_d_addr_high);
                let q8_d_high = ctx.cvt_f32_f16(q8_d_high_f16);

                // Select per-sub-block scales (same selp chain as MWV)
                let lsi = ctx.shl_u32(ci, one);
                let hsi = ctx.add_u32(lsi, 1);

                // Select LOW scale
                let dl = ds0;
                let ml = dm0;
                let p = ctx.setp_eq_u32(lsi, one);
                let dl = ctx.selp_f32(p, ds1, dl);
                let ml = ctx.selp_f32(p, dm1, ml);
                let two_u = ctx.mov_u32_imm(2);
                let p = ctx.setp_eq_u32(lsi, two_u);
                let dl = ctx.selp_f32(p, ds2, dl);
                let ml = ctx.selp_f32(p, dm2, ml);
                let three_u = ctx.mov_u32_imm(3);
                let p = ctx.setp_eq_u32(lsi, three_u);
                let dl = ctx.selp_f32(p, ds3, dl);
                let ml = ctx.selp_f32(p, dm3, ml);
                let p = ctx.setp_eq_u32(lsi, four);
                let dl = ctx.selp_f32(p, ds4, dl);
                let ml = ctx.selp_f32(p, dm4, ml);
                let five_u = ctx.mov_u32_imm(5);
                let p = ctx.setp_eq_u32(lsi, five_u);
                let dl = ctx.selp_f32(p, ds5, dl);
                let ml = ctx.selp_f32(p, dm5, ml);
                let six_u = ctx.mov_u32_imm(6);
                let p = ctx.setp_eq_u32(lsi, six_u);
                let dl = ctx.selp_f32(p, ds6, dl);
                let ml = ctx.selp_f32(p, dm6, ml);
                let seven_u = ctx.mov_u32_imm(7);
                let p = ctx.setp_eq_u32(lsi, seven_u);
                let dl = ctx.selp_f32(p, ds7, dl);
                let ml = ctx.selp_f32(p, dm7, ml);

                // Select HIGH scale
                let dh = ds0;
                let mh = dm0;
                let p = ctx.setp_eq_u32(hsi, one);
                let dh = ctx.selp_f32(p, ds1, dh);
                let mh = ctx.selp_f32(p, dm1, mh);
                let p = ctx.setp_eq_u32(hsi, two_u);
                let dh = ctx.selp_f32(p, ds2, dh);
                let mh = ctx.selp_f32(p, dm2, mh);
                let p = ctx.setp_eq_u32(hsi, three_u);
                let dh = ctx.selp_f32(p, ds3, dh);
                let mh = ctx.selp_f32(p, dm3, mh);
                let p = ctx.setp_eq_u32(hsi, four);
                let dh = ctx.selp_f32(p, ds4, dh);
                let mh = ctx.selp_f32(p, dm4, mh);
                let p = ctx.setp_eq_u32(hsi, five_u);
                let dh = ctx.selp_f32(p, ds5, dh);
                let mh = ctx.selp_f32(p, dm5, mh);
                let p = ctx.setp_eq_u32(hsi, six_u);
                let dh = ctx.selp_f32(p, ds6, dh);
                let mh = ctx.selp_f32(p, dm6, mh);
                let p = ctx.setp_eq_u32(hsi, seven_u);
                let dh = ctx.selp_f32(p, ds7, dh);
                let mh = ctx.selp_f32(p, dm7, mh);

                // Convert DP4A results to f32
                let dot_low_f = ctx.cvt_f32_s32(dot_low);
                let dot_high_f = ctx.cvt_f32_s32(dot_high);
                let sum_low_f = ctx.cvt_f32_s32(sum_low);
                let sum_high_f = ctx.cvt_f32_s32(sum_high);

                // LOW contribution: q8_d * (ds * dot - dm * sum)
                let t1 = ctx.mul_f32(dl, dot_low_f);
                let t2 = ctx.mul_f32(ml, sum_low_f);
                let t3 = ctx.sub_f32(t1, t2);
                let t4 = ctx.mul_f32(q8_d_low, t3);
                ctx.add_f32_inplace(acc, t4);

                // HIGH contribution: q8_d * (ds * dot - dm * sum)
                let t1 = ctx.mul_f32(dh, dot_high_f);
                let t2 = ctx.mul_f32(mh, sum_high_f);
                let t3 = ctx.sub_f32(t1, t2);
                let t4 = ctx.mul_f32(q8_d_high, t3);
                ctx.add_f32_inplace(acc, t4);

                // Stride by num_warps
                ctx.add_u32_reg_inplace(sb_idx, nw_reg);
                ctx.branch("dp4a_sb_loop");

                ctx.label("dp4a_sb_end");

                // Phase 1: Intra-warp reduction (same as MWV)
                let t16 = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t16);
                let t8 = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t8);
                let t4 = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t4);
                let t2 = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t2);
                let t1 = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, t1);

                // Phase 2: Cross-warp reduction via shared memory
                let z = ctx.mov_u32_imm(0);
                let is_l0 = ctx.setp_eq_u32(lane_id, z);
                ctx.branch_if_not(is_l0, "dp4a_skip_sm");

                let f4 = ctx.mov_u32_imm(4);
                let wo = ctx.mul_u32_reg(warp_id, f4);
                let sa = ctx.cvt_u64_u32(wo);
                ctx.st_shared_f32(sa, acc);

                ctx.label("dp4a_skip_sm");
                ctx.bar_sync(0);

                let is_t0 = ctx.setp_eq_u32(thread_id, z);
                ctx.branch_if_not(is_t0, "dp4a_exit");

                let fs = ctx.mov_f32_imm(0.0);
                for w in 0..num_warps {
                    let wo = ctx.mov_u64_imm(u64::from(w * 4));
                    let pv = ctx.ld_shared_f32(wo);
                    ctx.add_f32_inplace(fs, pv);
                }

                let yo = ctx.mul_wide_u32(block_id, 4);
                let ya = ctx.add_u64(y_ptr, yo);
                ctx.st_global_f32(ya, fs);

                ctx.label("dp4a_exit");
                ctx.ret();
            })
    }
}
