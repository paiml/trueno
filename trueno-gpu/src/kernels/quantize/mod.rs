//! Q4_K Dequantization-Fused GEMM Kernel
//!
//! Implements fused dequantization with matrix multiplication per GGML/llama.cpp methodology.
//!
//! ## Q4_K Super-block Layout (144 bytes for 256 values)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │ Offset 0-1: d (f16 super-block scale)                       │
//! │ Offset 2-3: dmin (f16 super-block min)                      │
//! │ Offset 4-15: scales (12 bytes, packed 6-bit scale+min × 8)  │
//! │ Offset 16-143: qs (128 bytes, 256 × 4-bit values packed)    │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Sub-block Structure
//!
//! Each super-block contains 8 sub-blocks of 32 values:
//! - Sub-block b uses: scale_b (6-bit) and min_b (6-bit) from scales[12]
//! - Dequantization: val = d × scale_b × quant - dmin × min_b
//!
//! ## PARITY-041: Fused Q4_K GEMM
//!
//! This kernel fuses dequantization with GEMM to eliminate intermediate buffers:
//! - Memory bandwidth: 144 bytes → 256 values (vs 512 bytes if dequantized to f16)
//! - 3.5x memory bandwidth reduction

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use super::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

mod dot;
mod fp16_tensor;
mod fused;
mod legacy;
mod q4k;
mod q5k;
mod q6k;

pub use dot::{PackedDp4aQ4KQ8Kernel, Q4KQ8DotKernel};
pub use fp16_tensor::{Fp16Q4KGemvKernel, TensorCoreQ4KGemmKernel};
pub use fused::{
    FusedGateUpQ4KGemvKernel, FusedRmsNormGateUpSwigluQ4KKernel, FusedRmsNormQ4KGemvKernel,
};
pub use legacy::{Q4_0GemvKernel, Q4_1GemvKernel, Q5_0GemvKernel, Q8_0GemvKernel};
pub use q4k::{
    BatchedQ4KGemvKernel, ChunkedTiledQ4KGemvKernel, CoalescedQ4KGemvKernel, Dp4aQ4KGemvKernel,
    MultiWarpVectorizedQ4KGemvKernel, Q4KGemvKernel, TiledQ4KGemvKernel,
    TrueDp4aQ4KGemvKernel, VectorizedQ4KGemvKernel, WideQ4KGemvKernel,
};
pub use q5k::{Q5KGemvKernel, Q5KKernel};
pub use q6k::{BatchedQ6KGemvKernel, CoalescedQ6KGemvKernel, Q6KGemvKernel, Q6KKernel};

/// Q4_K sub-block size (number of weights per sub-block)
const Q4K_BLOCK_SIZE: u32 = 32;
/// Q4_K super-block size (number of weights per super-block)
pub(crate) const Q4K_SUPER_BLOCK_SIZE: u32 = 256;
/// Bytes per Q4_K super-block (2 + 2 + 12 + 128 = 144 bytes)
pub(crate) const Q4K_SUPER_BLOCK_BYTES: u32 = 144;
/// Legacy: Bytes per simplified Q4_K block (for backwards compatibility)
const Q4K_BLOCK_BYTES: u32 = 18;

/// Q5_K super-block size (number of weights per super-block)
pub(crate) const Q5K_SUPER_BLOCK_SIZE: u32 = 256;
/// Bytes per Q5_K super-block (2 + 2 + 12 + 128 + 32 = 176 bytes)
/// Layout: d(2) + dmin(2) + scales(12) + qs(128) + qh(32)
pub(crate) const Q5K_SUPER_BLOCK_BYTES: u32 = 176;

/// Q6_K super-block size (number of weights per super-block)
pub(crate) const Q6K_SUPER_BLOCK_SIZE: u32 = 256;
/// Bytes per Q6_K super-block (128 + 64 + 16 + 2 = 210 bytes)
/// Layout: ql(128) + qh(64) + scales(16) + d(2)
pub(crate) const Q6K_SUPER_BLOCK_BYTES: u32 = 210;

/// Q8_0 block size (number of weights per block)
const Q8_0_BLOCK_SIZE: u32 = 32;
/// Bytes per Q8_0 block (2 + 32 = 34 bytes)
/// Layout: d(2 bytes, fp16) + qs[32] (32 int8 values)
const Q8_0_BLOCK_BYTES: u32 = 34;

/// Q5_0 block size (number of weights per block)
const Q5_0_BLOCK_SIZE: u32 = 32;
/// Bytes per Q5_0 block (2 + 4 + 16 = 22 bytes)
/// Layout: d(2 bytes, fp16) + qh(4 bytes, u32 with 32 high bits) + qs[16] (32 nibbles)
const Q5_0_BLOCK_BYTES: u32 = 22;

/// Q4_K format variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q4KFormat {
    /// Simplified format (32 values, 18 bytes) - legacy
    Simplified,
    /// Real GGML format (256 values, 144 bytes per super-block)
    GgmlSuperBlock,
}

/// Q4_K quantized GEMM kernel configuration
#[derive(Debug, Clone)]
pub struct QuantizeKernel {
    /// Output rows (M)
    pub m: u32,
    /// Output columns (N)
    pub n: u32,
    /// Inner dimension (K) - must be divisible by super_block_size (256)
    pub k: u32,
    /// Tile size for output
    pub tile_size: u32,
    /// Quantization block size
    pub block_size: u32,
    /// Format variant (GGML super-block or simplified)
    pub format: Q4KFormat,
}

impl QuantizeKernel {
    /// Create a new Q4_K quantized GEMM kernel (simplified format for compatibility)
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            tile_size: 32,
            block_size: Q4K_BLOCK_SIZE,
            format: Q4KFormat::Simplified,
        }
    }

    /// Create a Q4_K kernel using real GGML super-block format (PARITY-041)
    ///
    /// This is the correct format for GGUF model weights:
    /// - 256 values per super-block
    /// - 144 bytes per super-block (2+2+12+128)
    /// - 8 sub-blocks with 6-bit scale/min each
    #[must_use]
    pub fn ggml(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            tile_size: 32,
            block_size: Q4K_SUPER_BLOCK_SIZE,
            format: Q4KFormat::GgmlSuperBlock,
        }
    }

    /// Set output tile size
    #[must_use]
    pub const fn with_tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Get number of quantization blocks per row
    #[must_use]
    pub const fn num_blocks_per_row(&self) -> u32 {
        self.k / self.block_size
    }

    /// Get number of super-blocks per row (for GGML format)
    #[must_use]
    pub const fn num_super_blocks_per_row(&self) -> u32 {
        self.k / Q4K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for QuantizeKernel {
    fn name(&self) -> &str {
        match self.format {
            Q4KFormat::Simplified => "q4k_gemm_fused",
            Q4KFormat::GgmlSuperBlock => "q4k_gemm_ggml",
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        match self.format {
            Q4KFormat::Simplified => self.build_fused_gemm_simplified(),
            Q4KFormat::GgmlSuperBlock => self.build_fused_gemm_ggml(),
        }
    }
}

impl QuantizeKernel {
    /// Build kernel for simplified Q4_K format (legacy, 32 values/block)
    fn build_fused_gemm_simplified(&self) -> PtxKernel {
        // Q4_K GEMM with fused dequantization
        // Each warp processes one block of 32 weights
        let tile_size = self.tile_size;
        let block_size = self.block_size;

        // Shared memory for dequantized tile
        let smem_size = tile_size * tile_size * 4;

        PtxKernel::new("q4k_gemm_fused")
            .param(PtxType::U64, "a_ptr") // Input activations (f32)
            .param(PtxType::U64, "b_quant_ptr") // Quantized weights (Q4_K)
            .param(PtxType::U64, "c_ptr") // Output (f32)
            .param(PtxType::U32, "m") // Output rows
            .param(PtxType::U32, "n") // Output columns
            .param(PtxType::U32, "k") // Inner dimension
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

                // Thread's position within tile
                let local_row = ctx.div_u32(tid, tile_size);
                let local_col = ctx.rem_u32(tid, tile_size);

                // Global output position
                let global_row = ctx.add_u32_reg(out_row, local_row);
                let global_col = ctx.add_u32_reg(out_col, local_col);

                // Bounds check - compute predicates for later store
                let row_oob = ctx.setp_ge_u32(global_row, m_param);
                let col_oob = ctx.setp_ge_u32(global_col, n_param);

                // Clamp global_row and global_col to valid range [0, m-1] and [0, n-1]
                // This ensures all memory accesses are valid even for out-of-bounds threads.
                // Out-of-bounds threads will compute redundant values but won't store them.
                // This is necessary because all threads in a warp must participate in
                // warp shuffle reductions (shfl.sync with mask 0xFFFFFFFF).
                let one = ctx.mov_u32_imm(1);
                let m_minus_1 = ctx.sub_u32_reg(m_param, one);
                let n_minus_1 = ctx.sub_u32_reg(n_param, one);
                let clamped_row = ctx.min_u32(global_row, m_minus_1);
                let clamped_col = ctx.min_u32(global_col, n_minus_1);

                // Initialize accumulator (all threads)
                let acc = ctx.mov_f32_imm(0.0);

                // Calculate number of blocks in K dimension
                let block_size_reg = ctx.mov_u32_imm(block_size);
                let num_k_blocks = ctx.div_u32(k_param, block_size);

                // Loop over K blocks
                let k_block = ctx.mov_u32_imm(0);

                ctx.label("k_block_loop");
                let k_done = ctx.setp_ge_u32(k_block, num_k_blocks);
                ctx.branch_if(k_done, "k_block_done");

                // ===== Load and dequantize weight block =====
                // Weight layout: each row has (K/32) Q4_K blocks

                // Calculate block address for weight[clamped_col][k_block]
                // Use clamped_col to ensure valid memory access for all threads
                // Block address = b_quant_ptr + clamped_col * (K/32) * 18 + k_block * 18
                let blocks_per_row = num_k_blocks;
                let block_bytes = ctx.mov_u32_imm(Q4K_BLOCK_BYTES);
                let row_offset = ctx.mul_u32_reg(clamped_col, blocks_per_row);
                let block_offset = ctx.add_u32_reg(row_offset, k_block);
                let byte_offset = ctx.mul_wide_u32_reg(block_offset, block_bytes);
                let block_addr = ctx.add_u64(b_quant_ptr, byte_offset);

                // Load scale from block header (f16 at offset 0)
                // Simplified Q4K format: 2-byte f16 scale + 16 bytes data = 18 bytes
                let scale_addr = block_addr;
                let scale_f16 = ctx.ld_global_f16(scale_addr);
                let scale = ctx.cvt_f32_f16(scale_f16);

                // Load packed 4-bit values
                // Thread i loads values at position (i % 32) within block
                let lane = ctx.rem_u32(tid, block_size);
                let byte_idx = ctx.div_u32(lane, 2);
                let nibble_idx = ctx.rem_u32(lane, 2);

                // Data starts at offset 2 (after 2-byte f16 scale)
                let header_size = ctx.mov_u64_imm(2);
                let data_addr = ctx.add_u64(block_addr, header_size);
                let byte_idx_64 = ctx.cvt_u64_u32(byte_idx);
                let packed_addr = ctx.add_u64(data_addr, byte_idx_64);
                let packed = ctx.ld_global_u8(packed_addr);

                // Extract 4-bit value (no branch - use shift/mask)
                let four = ctx.mov_u32_imm(4);
                let shift = ctx.mul_u32_reg(nibble_idx, four);
                let packed_32 = ctx.cvt_u32_u8(packed);
                let fifteen = ctx.mov_u32_imm(0xF);
                let shifted = ctx.shr_u32(packed_32, shift);
                let quant = ctx.and_u32(shifted, fifteen);

                // Fused dequantization: val = scale * quant
                // (simplified format has no min/bias term)
                let quant_f32 = ctx.cvt_f32_u32(quant);
                let dequant = ctx.mul_f32(scale, quant_f32);

                // ===== Load activation value =====
                // A[clamped_row][k_block * 32 + lane]
                // Use clamped_row to ensure valid memory access for all threads
                let k_offset_base = ctx.mul_u32_reg(k_block, block_size_reg);
                let k_offset = ctx.add_u32_reg(k_offset_base, lane);

                // A address = a_ptr + clamped_row * K + k_offset
                let a_row_offset = ctx.mul_wide_u32_reg(clamped_row, k_param);
                let k_offset_64 = ctx.cvt_u64_u32(k_offset);
                let a_elem_offset = ctx.add_u64(a_row_offset, k_offset_64);
                let a_elem_offset_bytes = ctx.mul_u64(a_elem_offset, 4);
                let a_addr = ctx.add_u64(a_ptr, a_elem_offset_bytes);

                let a_val = ctx.ld_global_f32(a_addr);

                // ===== Accumulate: acc += a_val * dequant =====
                let prod = ctx.mul_f32(a_val, dequant);

                // Warp reduce for dot product
                let shuffled_16 = ctx.shfl_down_f32(prod, 16, 0xFFFF_FFFF);
                let prod_1 = ctx.add_f32(prod, shuffled_16);

                let shuffled_8 = ctx.shfl_down_f32(prod_1, 8, 0xFFFF_FFFF);
                let prod_2 = ctx.add_f32(prod_1, shuffled_8);

                let shuffled_4 = ctx.shfl_down_f32(prod_2, 4, 0xFFFF_FFFF);
                let prod_3 = ctx.add_f32(prod_2, shuffled_4);

                let shuffled_2 = ctx.shfl_down_f32(prod_3, 2, 0xFFFF_FFFF);
                let prod_4 = ctx.add_f32(prod_3, shuffled_2);

                let shuffled_1 = ctx.shfl_down_f32(prod_4, 1, 0xFFFF_FFFF);
                let block_sum = ctx.add_f32(prod_4, shuffled_1);

                // Broadcast sum to all lanes (use shfl_idx, NOT shfl_down with 0!)
                // shfl_down(x, 0) is a no-op - it returns x unchanged
                // shfl_idx(x, 0) broadcasts lane 0's value to all lanes
                let broadcast_sum = ctx.shfl_idx_f32(block_sum, 0, 0xFFFF_FFFF);

                // Add to accumulator IN-PLACE (not shadowing!)
                // Previous: let acc = ctx.add_f32(acc, broadcast_sum); // WRONG: creates new reg
                ctx.add_f32_inplace(acc, broadcast_sum);

                // Increment K block counter IN-PLACE and loop back
                // Previous: let _k_next = ctx.add_u32(k_block, 1); // WRONG: discarded
                // Previous: ctx.branch("k_block_done"); // WRONG: exits loop
                ctx.add_u32_inplace(k_block, 1);
                ctx.branch("k_block_loop"); // CORRECT: loop back

                ctx.label("k_block_done");

                // ===== Store result =====
                ctx.branch_if(row_oob, "exit");
                ctx.branch_if(col_oob, "exit");

                // C address = c_ptr + global_row * N + global_col
                let c_row_offset = ctx.mul_wide_u32_reg(global_row, n_param);
                let global_col_64 = ctx.cvt_u64_u32(global_col);
                let c_elem_offset = ctx.add_u64(c_row_offset, global_col_64);
                let c_elem_offset_bytes = ctx.mul_u64(c_elem_offset, 4);
                let c_addr = ctx.add_u64(c_ptr, c_elem_offset_bytes);

                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }

    /// Build kernel for real GGML Q4_K super-block format (PARITY-041)
    ///
    /// Super-block layout (144 bytes for 256 values):
    /// - Offset 0-1: d (f16 super-block scale)
    /// - Offset 2-3: dmin (f16 super-block min)
    /// - Offset 4-15: scales (12 bytes, packed 6-bit scale+min × 8 sub-blocks)
    /// - Offset 16-143: qs (128 bytes, 256 × 4-bit values packed)
    ///
    /// Dequantization: val = d × scale_b × quant - dmin × min_b
    fn build_fused_gemm_ggml(&self) -> PtxKernel {
        let tile_size = self.tile_size;

        // Shared memory for dequantized values
        let smem_size = Q4K_SUPER_BLOCK_SIZE * 4; // 256 f32 values

        PtxKernel::new("q4k_gemm_ggml")
            .param(PtxType::U64, "a_ptr") // Input activations (f32)
            .param(PtxType::U64, "b_quant_ptr") // Quantized weights (Q4_K GGML)
            .param(PtxType::U64, "c_ptr") // Output (f32)
            .param(PtxType::U32, "m") // Output rows
            .param(PtxType::U32, "n") // Output columns
            .param(PtxType::U32, "k") // Inner dimension
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

                // Thread's position within tile
                let local_row = ctx.div_u32(tid, tile_size);
                let local_col = ctx.rem_u32(tid, tile_size);

                // Global output position
                let global_row = ctx.add_u32_reg(out_row, local_row);
                let global_col = ctx.add_u32_reg(out_col, local_col);

                // Bounds check - compute predicates for later store
                let row_oob = ctx.setp_ge_u32(global_row, m_param);
                let col_oob = ctx.setp_ge_u32(global_col, n_param);

                // Clamp global_row and global_col to valid range [0, m-1] and [0, n-1]
                // This ensures all memory accesses are valid even for out-of-bounds threads.
                // Out-of-bounds threads will compute redundant values but won't store them.
                // This is necessary because all threads in a warp must participate in
                // warp shuffle reductions (shfl.sync with mask 0xFFFFFFFF).
                let one = ctx.mov_u32_imm(1);
                let m_minus_1 = ctx.sub_u32_reg(m_param, one);
                let n_minus_1 = ctx.sub_u32_reg(n_param, one);
                let clamped_row = ctx.min_u32(global_row, m_minus_1);
                let clamped_col = ctx.min_u32(global_col, n_minus_1);

                // Initialize accumulator (all threads, including out-of-bounds)
                let acc = ctx.mov_f32_imm(0.0);

                // Calculate number of super-blocks in K dimension (K / 256)
                let num_k_super_blocks = ctx.div_u32(k_param, Q4K_SUPER_BLOCK_SIZE);

                // Loop over K super-blocks
                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_k_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_done");

                // ===== Load Q4_K super-block header =====
                // Super-block address = b_quant_ptr + clamped_col * (K/256) * 144 + sb_idx * 144
                // Use clamped_col to ensure valid memory access for all threads
                let sb_per_row = num_k_super_blocks;
                let row_sb_offset = ctx.mul_u32_reg(clamped_col, sb_per_row);
                let total_sb_offset = ctx.add_u32_reg(row_sb_offset, sb_idx);
                let byte_offset = ctx.mul_wide_u32(total_sb_offset, Q4K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(b_quant_ptr, byte_offset);

                // Load d (f16 at offset 0)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Load dmin (f16 at offset 2)
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // ===== Process 8 sub-blocks of 32 values each =====
                // Each thread handles multiple values within the sub-block
                let sub_block_idx = ctx.mov_u32_imm(0);
                let eight = ctx.mov_u32_imm(8);
                let thirty_two = ctx.mov_u32_imm(32);

                ctx.label("sub_block_loop");
                let sub_done = ctx.setp_ge_u32(sub_block_idx, eight);
                ctx.branch_if(sub_done, "sub_block_done");

                // ===== Extract 6-bit scale and min for this sub-block =====
                // scales[12] contains packed 12-bit entries (6-bit scale + 6-bit min)
                // bit_offset = sub_block_idx * 12
                let bit_offset = ctx.mul_u32(sub_block_idx, 12);
                let byte_idx = ctx.div_u32(bit_offset, 8);
                let bit_in_byte = ctx.rem_u32(bit_offset, 8);

                // Load 2-3 bytes from scales (offset 4 in super-block)
                let four = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four);
                let byte_idx_64 = ctx.cvt_u64_u32(byte_idx);
                let scales_addr = ctx.add_u64(scales_base, byte_idx_64);
                let scale_b0 = ctx.ld_global_u8(scales_addr);
                let one_64 = ctx.mov_u64_imm(1);
                let scales_addr1 = ctx.add_u64(scales_addr, one_64);
                let scale_b1 = ctx.ld_global_u8(scales_addr1);

                // Combine bytes and extract 12 bits
                let b0_32 = ctx.cvt_u32_u8(scale_b0);
                let b1_32 = ctx.cvt_u32_u8(scale_b1);
                let eight_shift = ctx.mov_u32_imm(8);
                let b1_shifted = ctx.shl_u32(b1_32, eight_shift);
                let combined = ctx.or_u32(b0_32, b1_shifted);
                let bits_12 = ctx.shr_u32(combined, bit_in_byte);

                // Extract 6-bit scale (lower 6 bits) and min (upper 6 bits)
                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let scale_6bit = ctx.and_u32(bits_12, mask_6bit);
                let six_shift = ctx.mov_u32_imm(6);
                let min_shifted = ctx.shr_u32(bits_12, six_shift);
                let min_6bit = ctx.and_u32(min_shifted, mask_6bit);

                // Convert to floats and normalize to [0,1]
                let scale_f32 = ctx.cvt_f32_u32(scale_6bit);
                let min_f32 = ctx.cvt_f32_u32(min_6bit);
                let inv_63 = ctx.mov_f32_imm(1.0 / 63.0);
                let scale_norm = ctx.mul_f32(scale_f32, inv_63);
                let min_norm = ctx.mul_f32(min_f32, inv_63);

                // ===== Process 32 values in this sub-block =====
                // Thread tid handles value (tid % 32) within sub-block
                let lane = ctx.rem_u32(tid, 32);

                // Load quantized 4-bit value
                // qs offset = 16 + sub_block_idx * 16 + lane/2
                let sixteen = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen);
                let sub_block_offset = ctx.mul_u32(sub_block_idx, 16);
                let sub_block_offset_64 = ctx.cvt_u64_u32(sub_block_offset);
                let qs_sub_base = ctx.add_u64(qs_base, sub_block_offset_64);

                let byte_in_sub = ctx.div_u32(lane, 2);
                let nibble_idx = ctx.rem_u32(lane, 2);
                let byte_in_sub_64 = ctx.cvt_u64_u32(byte_in_sub);
                let qs_addr = ctx.add_u64(qs_sub_base, byte_in_sub_64);
                let packed = ctx.ld_global_u8(qs_addr);

                // Extract 4-bit value
                let shift_amt = ctx.mul_u32(nibble_idx, 4);
                let packed_32 = ctx.cvt_u32_u8(packed);
                let shifted = ctx.shr_u32(packed_32, shift_amt);
                let mask_4bit = ctx.mov_u32_imm(0xF);
                let quant = ctx.and_u32(shifted, mask_4bit);

                // Dequantize: val = d × scale × quant - dmin × min
                let quant_f32 = ctx.cvt_f32_u32(quant);
                let d_scale = ctx.mul_f32(d, scale_norm);
                let scaled = ctx.mul_f32(d_scale, quant_f32);
                let dmin_min = ctx.mul_f32(dmin, min_norm);
                let dequant = ctx.sub_f32(scaled, dmin_min);

                // ===== Load activation and accumulate =====
                // A[clamped_row][sb_idx * 256 + sub_block_idx * 32 + lane]
                // Use clamped_row to ensure valid memory access for all threads
                let two_fifty_six = ctx.mov_u32_imm(256);
                let sb_k_offset = ctx.mul_u32_reg(sb_idx, two_fifty_six);
                let sub_k_offset = ctx.mul_u32_reg(sub_block_idx, thirty_two);
                let k_offset = ctx.add_u32_reg(sb_k_offset, sub_k_offset);
                let k_offset_full = ctx.add_u32_reg(k_offset, lane);

                let a_row_offset = ctx.mul_wide_u32_reg(clamped_row, k_param);
                let k_offset_64 = ctx.cvt_u64_u32(k_offset_full);
                let a_elem_offset = ctx.add_u64(a_row_offset, k_offset_64);
                let a_elem_bytes = ctx.mul_u64(a_elem_offset, 4);
                let a_addr = ctx.add_u64(a_ptr, a_elem_bytes);

                let a_val = ctx.ld_global_f32(a_addr);

                // Multiply and reduce
                let prod = ctx.mul_f32(a_val, dequant);

                // Warp reduce for dot product
                let shuffled_16 = ctx.shfl_down_f32(prod, 16, 0xFFFF_FFFF);
                let prod_1 = ctx.add_f32(prod, shuffled_16);
                let shuffled_8 = ctx.shfl_down_f32(prod_1, 8, 0xFFFF_FFFF);
                let prod_2 = ctx.add_f32(prod_1, shuffled_8);
                let shuffled_4 = ctx.shfl_down_f32(prod_2, 4, 0xFFFF_FFFF);
                let prod_3 = ctx.add_f32(prod_2, shuffled_4);
                let shuffled_2 = ctx.shfl_down_f32(prod_3, 2, 0xFFFF_FFFF);
                let prod_4 = ctx.add_f32(prod_3, shuffled_2);
                let shuffled_1 = ctx.shfl_down_f32(prod_4, 1, 0xFFFF_FFFF);
                let sub_block_sum = ctx.add_f32(prod_4, shuffled_1);

                // Broadcast and accumulate
                let broadcast_sum = ctx.shfl_idx_f32(sub_block_sum, 0, 0xFFFF_FFFF);
                ctx.add_f32_inplace(acc, broadcast_sum);

                // Next sub-block
                ctx.add_u32_inplace(sub_block_idx, 1);
                ctx.branch("sub_block_loop");

                ctx.label("sub_block_done");

                // Next super-block
                ctx.add_u32_inplace(sb_idx, 1);
                ctx.branch("sb_loop");

                ctx.label("sb_loop_done");

                // ===== Store result =====
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

// =============================================================================
// PAR-063-V4: Q8 QUANTIZATION KERNEL (ACTIVATION QUANTIZATION)
// =============================================================================

/// Q8_1 Quantization kernel for activations (PAR-063-V4)
///
/// Converts f32 activations to Q8_1 format for use with DP4A dot products.
/// This is the key optimization used by llama.cpp to enable true DP4A SIMD.
///
/// # Q8_1 Format
///
/// Each block of 32 values is stored as:
/// - qs[32]: 32 x int8 quantized values
/// - d: f16 scale factor
/// - s: f16 sum of values (for min contribution in Q4K dot product)
///
/// Total: 34 bytes per 32 values = 8.5 bits per value
///
/// # Quantization Formula
///
/// ```text
/// max_abs = max(|x_0|, |x_1|, ..., |x_31|)
/// scale = max_abs / 127
/// q_i = round(x_i / scale)  // clamped to [-127, 127]
/// ```
///
/// # Performance Impact
///
/// By pre-quantizing activations:
/// - GEMV can use pure integer DP4A (4 MADs per instruction)
/// - Eliminates f32 activation loads in inner loop
/// - Expected 2-4x instruction reduction
///
/// # References
///
/// - llama.cpp: ggml_quantize_q8_1 in ggml-quants.c
/// - NVIDIA: dp4a.u32.s32 for unsigned weights × signed activations
#[derive(Debug, Clone)]
pub struct Q8QuantizeKernel {
    /// Number of elements to quantize (must be multiple of 32)
    pub n: u32,
}

impl Q8QuantizeKernel {
    /// Create a new Q8 quantization kernel
    #[must_use]
    pub fn new(n: u32) -> Self {
        Self { n }
    }

    /// Get number of Q8 blocks (32 values each)
    #[must_use]
    pub const fn num_blocks(&self) -> u32 {
        (self.n + 31) / 32
    }
}

impl Kernel for Q8QuantizeKernel {
    fn name(&self) -> &str {
        "q8_quantize"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Grid: one block per Q8 block (32 values)
        // Each warp (32 threads) processes one Q8 block cooperatively
        PtxKernel::new("q8_quantize")
            .param(PtxType::U64, "out_ptr") // Q8 output: [num_blocks * 34] bytes
            .param(PtxType::U64, "in_ptr") // f32 input: [n] floats
            .param(PtxType::U32, "n_dim")
            .build(|ctx| {
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);
                let lane_id = ctx.rem_u32(thread_id, 32);

                let n_dim = ctx.load_param_u32("n_dim");
                let num_blocks = ctx.add_u32(n_dim, 31);
                let num_blocks = ctx.div_u32(num_blocks, 32);

                // Bounds check
                let oob = ctx.setp_ge_u32(block_id, num_blocks);
                ctx.branch_if(oob, "exit");

                let out_ptr = ctx.load_param_u64("out_ptr");
                let in_ptr = ctx.load_param_u64("in_ptr");

                // Each thread loads 1 value (32 threads = 32 values = 1 Q8 block)
                let block_start = ctx.mul_u32(block_id, 32);
                let idx = ctx.add_u32_reg(block_start, lane_id);

                // Load f32 value
                let idx_64 = ctx.cvt_u64_u32(idx);
                let idx_bytes = ctx.mul_u64(idx_64, 4);
                let in_addr = ctx.add_u64(in_ptr, idx_bytes);
                let val = ctx.ld_global_f32(in_addr);

                // Compute absolute value
                let abs_val = ctx.abs_f32(val);

                // Find max absolute value across warp using shuffle reduction
                let max_abs = abs_val;
                let tmp16 = ctx.shfl_down_f32(max_abs, 16, 0xFFFF_FFFF);
                let max_abs = ctx.max_f32(max_abs, tmp16);
                let tmp8 = ctx.shfl_down_f32(max_abs, 8, 0xFFFF_FFFF);
                let max_abs = ctx.max_f32(max_abs, tmp8);
                let tmp4 = ctx.shfl_down_f32(max_abs, 4, 0xFFFF_FFFF);
                let max_abs = ctx.max_f32(max_abs, tmp4);
                let tmp2 = ctx.shfl_down_f32(max_abs, 2, 0xFFFF_FFFF);
                let max_abs = ctx.max_f32(max_abs, tmp2);
                let tmp1 = ctx.shfl_down_f32(max_abs, 1, 0xFFFF_FFFF);
                let max_abs = ctx.max_f32(max_abs, tmp1);

                // Broadcast max to all lanes
                let max_abs = ctx.shfl_idx_f32(max_abs, 0, 0xFFFF_FFFF);

                // Compute scale: d = max_abs / 127
                let inv_127 = ctx.mov_f32_imm(1.0 / 127.0);
                let scale = ctx.mul_f32(max_abs, inv_127);

                // Compute inverse scale for quantization
                let eps = ctx.mov_f32_imm(1e-10);
                let scale_eps = ctx.add_f32(scale, eps);
                let inv_scale = ctx.rcp_f32(scale_eps);

                // Quantize: q = round(val * inv_scale) clamped to [-127, 127]
                let scaled = ctx.mul_f32(val, inv_scale);
                let rounded = ctx.cvt_rni_s32_f32(scaled);

                // Clamp to [-127, 127]
                let min_val = ctx.mov_u32_imm(0xFFFF_FF81); // -127 as u32
                let min_s32 = ctx.mov_s32_from_u32(min_val);
                let max_val = ctx.mov_s32_imm(127);
                let clamped = ctx.max_s32(rounded, min_s32);
                let clamped = ctx.min_s32(clamped, max_val);

                // Convert to u8 (as signed byte stored in unsigned format)
                let q8_val = ctx.cvt_u8_s32(clamped);

                // Store quantized value
                // Q8_1 layout: [32 bytes qs] [2 bytes d] [2 bytes s]
                // Output offset for this block: block_id * 36 bytes
                let block_bytes = ctx.mov_u32_imm(36);
                let block_offset = ctx.mul_wide_u32_reg(block_id, block_bytes);
                let block_base = ctx.add_u64(out_ptr, block_offset);

                // Store qs[lane_id]
                let lane_64 = ctx.cvt_u64_u32(lane_id);
                let qs_addr = ctx.add_u64(block_base, lane_64);
                ctx.st_global_u8(qs_addr, q8_val);

                // Only lane 0 stores scale (d) and sum (s)
                let one = ctx.mov_u32_imm(1);
                let is_lane0 = ctx.setp_lt_u32(lane_id, one);
                ctx.branch_if_not(is_lane0, "exit");

                // Store scale at offset 32
                let thirty_two_64 = ctx.mov_u64_imm(32);
                let d_addr = ctx.add_u64(block_base, thirty_two_64);
                let scale_f16 = ctx.cvt_f16_f32(scale);
                ctx.st_global_f16(d_addr, scale_f16);

                // Compute sum of values for min contribution (warp reduction)
                // Note: sum is already computed from original values
                let sum = val;
                let sum_tmp16 = ctx.shfl_down_f32(sum, 16, 0xFFFF_FFFF);
                let sum = ctx.add_f32(sum, sum_tmp16);
                let sum_tmp8 = ctx.shfl_down_f32(sum, 8, 0xFFFF_FFFF);
                let sum = ctx.add_f32(sum, sum_tmp8);
                let sum_tmp4 = ctx.shfl_down_f32(sum, 4, 0xFFFF_FFFF);
                let sum = ctx.add_f32(sum, sum_tmp4);
                let sum_tmp2 = ctx.shfl_down_f32(sum, 2, 0xFFFF_FFFF);
                let sum = ctx.add_f32(sum, sum_tmp2);
                let sum_tmp1 = ctx.shfl_down_f32(sum, 1, 0xFFFF_FFFF);
                let sum = ctx.add_f32(sum, sum_tmp1);

                // Store sum at offset 34
                let thirty_four_64 = ctx.mov_u64_imm(34);
                let s_addr = ctx.add_u64(block_base, thirty_four_64);
                let sum_f16 = ctx.cvt_f16_f32(sum);
                ctx.st_global_f16(s_addr, sum_f16);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// Tests (~2K lines extracted for TDG compliance)
#[cfg(test)]
mod tests;
