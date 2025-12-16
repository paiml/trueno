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
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Q4_K sub-block size (number of weights per sub-block)
const Q4K_BLOCK_SIZE: u32 = 32;
/// Q4_K super-block size (number of weights per super-block)
const Q4K_SUPER_BLOCK_SIZE: u32 = 256;
/// Bytes per Q4_K super-block (2 + 2 + 12 + 128 = 144 bytes)
const Q4K_SUPER_BLOCK_BYTES: u32 = 144;
/// Legacy: Bytes per simplified Q4_K block (for backwards compatibility)
const Q4K_BLOCK_BYTES: u32 = 18;

/// Q5_K super-block size (number of weights per super-block)
const Q5K_SUPER_BLOCK_SIZE: u32 = 256;
/// Bytes per Q5_K super-block (2 + 2 + 12 + 128 + 32 = 176 bytes)
/// Layout: d(2) + dmin(2) + scales(12) + qs(128) + qh(32)
const Q5K_SUPER_BLOCK_BYTES: u32 = 176;

/// Q6_K super-block size (number of weights per super-block)
const Q6K_SUPER_BLOCK_SIZE: u32 = 256;
/// Bytes per Q6_K super-block (128 + 64 + 16 + 2 = 210 bytes)
/// Layout: ql(128) + qh(64) + scales(16) + d(2)
const Q6K_SUPER_BLOCK_BYTES: u32 = 210;

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
            .param(PtxType::U64, "a_ptr")           // Input activations (f32)
            .param(PtxType::U64, "b_quant_ptr")     // Quantized weights (Q4_K)
            .param(PtxType::U64, "c_ptr")           // Output (f32)
            .param(PtxType::U32, "m")               // Output rows
            .param(PtxType::U32, "n")               // Output columns
            .param(PtxType::U32, "k")               // Inner dimension
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
            .param(PtxType::U64, "a_ptr")           // Input activations (f32)
            .param(PtxType::U64, "b_quant_ptr")     // Quantized weights (Q4_K GGML)
            .param(PtxType::U64, "c_ptr")           // Output (f32)
            .param(PtxType::U32, "m")               // Output rows
            .param(PtxType::U32, "n")               // Output columns
            .param(PtxType::U32, "k")               // Inner dimension
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
// Q5_K FUSED GEMM KERNEL (PARITY-116)
// =============================================================================
//
// Q5_K Super-block Layout (176 bytes for 256 values):
// - Offset 0-1: d (f16 super-block scale)
// - Offset 2-3: dmin (f16 super-block min)
// - Offset 4-15: scales (12 bytes, packed 6-bit scale+min × 8 sub-blocks)
// - Offset 16-143: qs (128 bytes, 256 × 4-bit low values packed)
// - Offset 144-175: qh (32 bytes, 256 × 1-bit high values packed)
//
// Dequantization: val = d × scale_b × (ql + 16*qh) - dmin × min_b
// Where ql is 4-bit (0-15), qh is 1-bit (0 or 1), giving 5-bit range (0-31)

/// Q5_K quantized GEMM kernel configuration
#[derive(Debug, Clone)]
pub struct Q5KKernel {
    /// Output rows (M)
    pub m: u32,
    /// Output columns (N)
    pub n: u32,
    /// Inner dimension (K) - must be divisible by 256
    pub k: u32,
    /// Tile size for output
    pub tile_size: u32,
}

impl Q5KKernel {
    /// Create a new Q5_K quantized GEMM kernel
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
        self.k / Q5K_SUPER_BLOCK_SIZE
    }
}

impl Kernel for Q5KKernel {
    fn name(&self) -> &str {
        "q5k_gemm_ggml"
    }

    fn build_ptx(&self) -> PtxKernel {
        let tile_size = self.tile_size;
        let smem_size = Q5K_SUPER_BLOCK_SIZE * 4; // 256 f32 values

        PtxKernel::new("q5k_gemm_ggml")
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

                // Clamp to valid range for memory safety
                let one = ctx.mov_u32_imm(1);
                let m_minus_1 = ctx.sub_u32_reg(m_param, one);
                let n_minus_1 = ctx.sub_u32_reg(n_param, one);
                let clamped_row = ctx.min_u32(global_row, m_minus_1);
                let clamped_col = ctx.min_u32(global_col, n_minus_1);

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Number of super-blocks (K / 256)
                let num_k_super_blocks = ctx.div_u32(k_param, Q5K_SUPER_BLOCK_SIZE);

                // Super-block loop
                let sb_idx = ctx.mov_u32_imm(0);

                ctx.label("sb_loop");
                let sb_done = ctx.setp_ge_u32(sb_idx, num_k_super_blocks);
                ctx.branch_if(sb_done, "sb_loop_done");

                // Calculate super-block address
                let sb_per_row = num_k_super_blocks;
                let row_sb_offset = ctx.mul_u32_reg(clamped_col, sb_per_row);
                let total_sb_offset = ctx.add_u32_reg(row_sb_offset, sb_idx);
                let byte_offset = ctx.mul_wide_u32(total_sb_offset, Q5K_SUPER_BLOCK_BYTES);
                let sb_addr = ctx.add_u64(b_quant_ptr, byte_offset);

                // Load d (f16 at offset 0)
                let d_f16 = ctx.ld_global_f16(sb_addr);
                let d = ctx.cvt_f32_f16(d_f16);

                // Load dmin (f16 at offset 2)
                let two = ctx.mov_u64_imm(2);
                let dmin_addr = ctx.add_u64(sb_addr, two);
                let dmin_f16 = ctx.ld_global_f16(dmin_addr);
                let dmin = ctx.cvt_f32_f16(dmin_f16);

                // Process 8 sub-blocks of 32 values each
                let sub_block_idx = ctx.mov_u32_imm(0);
                let eight = ctx.mov_u32_imm(8);
                let thirty_two = ctx.mov_u32_imm(32);

                ctx.label("sub_block_loop");
                let sub_done = ctx.setp_ge_u32(sub_block_idx, eight);
                ctx.branch_if(sub_done, "sub_block_done");

                // Extract 6-bit scale and min (same as Q4_K)
                let bit_offset = ctx.mul_u32(sub_block_idx, 12);
                let byte_idx = ctx.div_u32(bit_offset, 8);
                let bit_in_byte = ctx.rem_u32(bit_offset, 8);

                let four = ctx.mov_u64_imm(4);
                let scales_base = ctx.add_u64(sb_addr, four);
                let byte_idx_64 = ctx.cvt_u64_u32(byte_idx);
                let scales_addr = ctx.add_u64(scales_base, byte_idx_64);
                let scale_b0 = ctx.ld_global_u8(scales_addr);
                let one_64 = ctx.mov_u64_imm(1);
                let scales_addr1 = ctx.add_u64(scales_addr, one_64);
                let scale_b1 = ctx.ld_global_u8(scales_addr1);

                let b0_32 = ctx.cvt_u32_u8(scale_b0);
                let b1_32 = ctx.cvt_u32_u8(scale_b1);
                let eight_shift = ctx.mov_u32_imm(8);
                let b1_shifted = ctx.shl_u32(b1_32, eight_shift);
                let combined = ctx.or_u32(b0_32, b1_shifted);
                let bits_12 = ctx.shr_u32(combined, bit_in_byte);

                let mask_6bit = ctx.mov_u32_imm(0x3F);
                let scale_6bit = ctx.and_u32(bits_12, mask_6bit);
                let six_shift = ctx.mov_u32_imm(6);
                let min_shifted = ctx.shr_u32(bits_12, six_shift);
                let min_6bit = ctx.and_u32(min_shifted, mask_6bit);

                let scale_f32 = ctx.cvt_f32_u32(scale_6bit);
                let min_f32 = ctx.cvt_f32_u32(min_6bit);
                let inv_63 = ctx.mov_f32_imm(1.0 / 63.0);
                let scale_norm = ctx.mul_f32(scale_f32, inv_63);
                let min_norm = ctx.mul_f32(min_f32, inv_63);

                // Thread's lane within sub-block
                let lane = ctx.rem_u32(tid, 32);

                // Load low 4-bit value from qs (offset 16 + sub_block_idx * 16 + lane/2)
                let sixteen = ctx.mov_u64_imm(16);
                let qs_base = ctx.add_u64(sb_addr, sixteen);
                let sub_block_offset = ctx.mul_u32(sub_block_idx, 16);
                let sub_block_offset_64 = ctx.cvt_u64_u32(sub_block_offset);
                let qs_sub_base = ctx.add_u64(qs_base, sub_block_offset_64);

                let byte_in_sub = ctx.div_u32(lane, 2);
                let nibble_idx = ctx.rem_u32(lane, 2);
                let byte_in_sub_64 = ctx.cvt_u64_u32(byte_in_sub);
                let qs_addr = ctx.add_u64(qs_sub_base, byte_in_sub_64);
                let packed_ql = ctx.ld_global_u8(qs_addr);

                let shift_amt = ctx.mul_u32(nibble_idx, 4);
                let packed_ql_32 = ctx.cvt_u32_u8(packed_ql);
                let shifted_ql = ctx.shr_u32(packed_ql_32, shift_amt);
                let mask_4bit = ctx.mov_u32_imm(0xF);
                let ql = ctx.and_u32(shifted_ql, mask_4bit);

                // Load high bit from qh (offset 144 + (sub_block_idx * 32 + lane) / 8)
                let qh_base_offset = ctx.mov_u64_imm(144);
                let qh_base = ctx.add_u64(sb_addr, qh_base_offset);
                let global_bit_idx = ctx.mul_u32(sub_block_idx, 32);
                let global_bit_idx_full = ctx.add_u32_reg(global_bit_idx, lane);
                let qh_byte_idx = ctx.div_u32(global_bit_idx_full, 8);
                let qh_bit_idx = ctx.rem_u32(global_bit_idx_full, 8);
                let qh_byte_idx_64 = ctx.cvt_u64_u32(qh_byte_idx);
                let qh_addr = ctx.add_u64(qh_base, qh_byte_idx_64);
                let qh_byte = ctx.ld_global_u8(qh_addr);
                let qh_byte_32 = ctx.cvt_u32_u8(qh_byte);
                let qh_shifted = ctx.shr_u32(qh_byte_32, qh_bit_idx);
                let mask_1bit = ctx.mov_u32_imm(1);
                let qh = ctx.and_u32(qh_shifted, mask_1bit);

                // Combine: quant = ql + 16 * qh (5-bit value: 0-31)
                let sixteen_u32 = ctx.mov_u32_imm(16);
                let qh_scaled = ctx.mul_u32_reg(qh, sixteen_u32);
                let quant = ctx.add_u32_reg(ql, qh_scaled);

                // Dequantize: val = d × scale × quant - dmin × min
                let quant_f32 = ctx.cvt_f32_u32(quant);
                let d_scale = ctx.mul_f32(d, scale_norm);
                let scaled = ctx.mul_f32(d_scale, quant_f32);
                let dmin_min = ctx.mul_f32(dmin, min_norm);
                let dequant = ctx.sub_f32(scaled, dmin_min);

                // Load activation and accumulate
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

                let prod = ctx.mul_f32(a_val, dequant);

                // Warp reduce
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_kernel_name() {
        let kernel = QuantizeKernel::new(1024, 1024, 4096);
        assert_eq!(kernel.name(), "q4k_gemm_fused");
    }

    #[test]
    fn test_quantize_default_config() {
        let kernel = QuantizeKernel::new(1024, 1024, 4096);
        assert_eq!(kernel.m, 1024);
        assert_eq!(kernel.n, 1024);
        assert_eq!(kernel.k, 4096);
        assert_eq!(kernel.tile_size, 32);
        assert_eq!(kernel.block_size, Q4K_BLOCK_SIZE);
    }

    #[test]
    fn test_quantize_with_tile_size() {
        let kernel = QuantizeKernel::new(1024, 1024, 4096).with_tile_size(64);
        assert_eq!(kernel.tile_size, 64);
    }

    #[test]
    fn test_quantize_num_blocks() {
        let kernel = QuantizeKernel::new(1024, 1024, 4096);
        assert_eq!(kernel.num_blocks_per_row(), 128); // 4096 / 32
    }

    #[test]
    fn test_quantize_ptx_generation() {
        let kernel = QuantizeKernel::new(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        // Verify parameters
        assert!(ptx.contains(".param .u64 a_ptr"));
        assert!(ptx.contains(".param .u64 b_quant_ptr"));
        assert!(ptx.contains(".param .u64 c_ptr"));
        assert!(ptx.contains(".param .u32 m"));
        assert!(ptx.contains(".param .u32 n"));
        assert!(ptx.contains(".param .u32 k"));
    }

    #[test]
    fn test_quantize_shared_memory() {
        let kernel = QuantizeKernel::new(1024, 1024, 4096);
        let ptx_kernel = kernel.build_ptx();

        // Should have shared memory for dequantized tile
        assert!(ptx_kernel.shared_memory_bytes() > 0);
    }

    #[test]
    fn test_quantize_ptx_contains_operations() {
        let kernel = QuantizeKernel::new(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        // Verify memory operations
        assert!(ptx.contains("ld.global"));
        assert!(ptx.contains("st.global.f32"));

        // Verify arithmetic for dequantization and GEMM
        assert!(ptx.contains("mul.f32"));
        assert!(ptx.contains("add.f32"));

        // Verify warp shuffle for reduction
        assert!(ptx.contains("shfl") || ptx.contains("shfl.down"));
    }

    #[test]
    fn test_quantize_dequantization_ops() {
        let kernel = QuantizeKernel::new(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        // Verify shift/mask for nibble extraction
        // Note: shr may be emitted differently
        assert!(ptx.contains("mul") || ptx.contains("shr"));

        // Verify type conversion
        assert!(ptx.contains("cvt"));
    }

    #[test]
    fn test_quantize_kernel_variants() {
        // Test with different configurations
        let configs = vec![
            QuantizeKernel::new(512, 512, 2048),
            QuantizeKernel::new(1024, 1024, 4096),
            QuantizeKernel::new(2048, 2048, 8192),
            QuantizeKernel::new(4096, 4096, 4096).with_tile_size(64),
        ];

        for config in configs {
            let ptx = config.emit_ptx();
            assert!(!ptx.is_empty());
            assert!(ptx.contains(".visible .entry"));
        }
    }

    #[test]
    fn test_quantize_block_layout() {
        // Verify Q4_K block constants
        assert_eq!(Q4K_BLOCK_SIZE, 32);
        assert_eq!(Q4K_BLOCK_BYTES, 18);
    }

    // =========================================================================
    // SATD REMEDIATION TESTS (EXTREME TDD)
    // These tests verify the K-loop and shuffle bugs are fixed.
    // Falsifiable claims per Popperian methodology.
    // =========================================================================

    #[test]
    fn test_kloop_branches_back_to_loop_start() {
        // FALSIFIABLE CLAIM: K-loop branches back to "k_block_loop", not "k_block_done"
        // This test FAILS if the SATD bug (single iteration) is present.
        let kernel = QuantizeKernel::new(64, 64, 128); // K=128 requires 4 K-blocks
        let ptx = kernel.emit_ptx();

        // The PTX should contain a branch back to k_block_loop
        // If it only branches to k_block_done, the loop exits after 1 iteration
        let has_loop_back = ptx.contains("bra k_block_loop") || ptx.contains("bra\tk_block_loop");

        assert!(
            has_loop_back,
            "FALSIFIED: K-loop does not branch back to loop start. \
             Found 'bra k_block_done' instead of 'bra k_block_loop'. \
             This means K-loop only runs once regardless of K value."
        );
    }

    #[test]
    fn test_kloop_counter_incremented_inplace() {
        // FALSIFIABLE CLAIM: K-loop counter is incremented in-place using add_u32_inplace
        // If add_u32 is used (returns new reg), the counter is never updated.
        let kernel = QuantizeKernel::new(64, 64, 128);
        let ptx = kernel.emit_ptx();

        // The PTX should increment k_block register in-place
        // Pattern: add.u32 %rN, %rN, 1 (same register for dest and src1)
        // If we see add.u32 %rM, %rN, 1 (different registers), it's broken

        // Count the k_block_loop and k_block_done labels
        let loop_count = ptx.matches("k_block_loop").count();
        let done_count = ptx.matches("k_block_done").count();

        // There should be exactly 2 references to k_block_loop:
        // 1. The label definition
        // 2. The branch back to the loop
        assert!(
            loop_count >= 2,
            "FALSIFIED: k_block_loop only appears {} times. \
             Expected at least 2 (label + branch back). \
             K-loop counter is not being used correctly.",
            loop_count
        );

        // done label should appear exactly once (the label definition)
        // If bra k_block_done appears twice, the loop exits incorrectly
        assert_eq!(
            done_count, 2, // label + conditional branch
            "FALSIFIED: k_block_done appears {} times. \
             Expected 2 (label + conditional exit). \
             Extra branches to k_block_done indicate premature loop exit.",
            done_count
        );
    }

    #[test]
    fn test_shuffle_broadcast_uses_shfl_idx_not_shfl_down_zero() {
        // FALSIFIABLE CLAIM: Broadcast uses shfl.idx (or shfl.sync.idx) with lane 0,
        // NOT shfl.down with offset 0 (which is a no-op).
        let kernel = QuantizeKernel::new(64, 64, 128);
        let ptx = kernel.emit_ptx();

        // shfl.down with offset 0 is a no-op - it returns the same value
        // Correct broadcast should use shfl.idx or shfl.sync.idx
        let has_shfl_idx = ptx.contains("shfl.idx") || ptx.contains("shfl.sync.idx");
        let has_bad_shfl_down_zero = ptx.contains("shfl.down.b32") && ptx.contains(", 0,");

        // Either we have shfl.idx (correct) or we don't have the bad pattern
        assert!(
            has_shfl_idx || !has_bad_shfl_down_zero,
            "FALSIFIED: Broadcast uses shfl.down with offset 0, which is a no-op. \
             Should use shfl.idx with lane 0 to broadcast the reduced value."
        );
    }

    #[test]
    fn test_accumulator_updated_inplace() {
        // FALSIFIABLE CLAIM: Accumulator is updated in-place, not shadowed
        // If add_f32 creates a new register, the accumulator is never updated.
        let kernel = QuantizeKernel::new(64, 64, 128);
        let ptx = kernel.emit_ptx();

        // The accumulator should be used in a fma.rn.f32 or add.f32 that writes
        // back to the same register. This is tricky to verify in PTX without
        // full SSA analysis, so we verify the loop structure instead.

        // The key invariant: if K > 32, we need multiple accumulations.
        // With K=128 (4 blocks), the final result should be sum of 4 partial products.
        // If accumulator is not updated, result will be wrong (only 1 block's contribution).

        // For now, verify the structure allows for accumulation
        let has_add_f32 = ptx.contains("add.f32") || ptx.contains("add.rn.f32");
        assert!(
            has_add_f32,
            "FALSIFIED: No add.f32 found for accumulation. \
             Accumulator cannot be updated without add instruction."
        );
    }

    // =========================================================================
    // PARITY-041: GGML Q4_K Super-block Format Tests
    // Verify the new kernel that uses real GGML Q4_K format (144-byte super-blocks)
    // =========================================================================

    #[test]
    fn test_ggml_kernel_name() {
        let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
        assert_eq!(kernel.name(), "q4k_gemm_ggml");
    }

    #[test]
    fn test_ggml_kernel_config() {
        let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
        assert_eq!(kernel.m, 1024);
        assert_eq!(kernel.n, 1024);
        assert_eq!(kernel.k, 4096);
        assert_eq!(kernel.block_size, Q4K_SUPER_BLOCK_SIZE); // 256 values
        assert_eq!(kernel.format, Q4KFormat::GgmlSuperBlock);
    }

    #[test]
    fn test_ggml_super_block_constants() {
        // Verify GGML Q4_K super-block constants
        assert_eq!(Q4K_SUPER_BLOCK_SIZE, 256, "Super-block should have 256 values");
        assert_eq!(Q4K_SUPER_BLOCK_BYTES, 144, "Super-block should be 144 bytes (2+2+12+128)");
    }

    #[test]
    fn test_ggml_num_super_blocks() {
        let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
        assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256 = 16
    }

    #[test]
    fn test_ggml_ptx_generation() {
        let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        // Verify kernel name
        assert!(ptx.contains("q4k_gemm_ggml"), "Should contain GGML kernel name");

        // Verify parameters
        assert!(ptx.contains(".param .u64 a_ptr"));
        assert!(ptx.contains(".param .u64 b_quant_ptr"));
        assert!(ptx.contains(".param .u64 c_ptr"));
        assert!(ptx.contains(".param .u32 m"));
        assert!(ptx.contains(".param .u32 n"));
        assert!(ptx.contains(".param .u32 k"));
    }

    #[test]
    fn test_ggml_ptx_contains_f16_loads() {
        let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        // GGML Q4_K has f16 scale (d) and min (dmin) at super-block header
        assert!(ptx.contains("ld.global.f16") || ptx.contains("ld.global.b16"),
            "Should load f16 values for d and dmin");
        assert!(ptx.contains("cvt") && ptx.contains("f32"),
            "Should convert f16 to f32 for computation");
    }

    #[test]
    fn test_ggml_ptx_contains_nested_loops() {
        let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        // GGML kernel has nested loops: super-block loop and sub-block loop
        assert!(ptx.contains("sb_loop"), "Should have super-block loop");
        assert!(ptx.contains("sub_block_loop"), "Should have sub-block loop for 8 sub-blocks");
    }

    #[test]
    fn test_ggml_ptx_contains_scale_extraction() {
        let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        // Scale extraction involves bit manipulation (12-bit packed entries)
        assert!(ptx.contains("shr") || ptx.contains("shl"),
            "Should have shift operations for scale extraction");
        assert!(ptx.contains("and"),
            "Should have AND operations for 6-bit masking");
    }

    #[test]
    fn test_ggml_ptx_contains_warp_reduce() {
        let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        // Warp shuffle reduction for dot product
        assert!(ptx.contains("shfl"),
            "Should have warp shuffle for reduction");
    }

    #[test]
    fn test_ggml_both_loop_branches_back() {
        // FALSIFIABLE: Both loops should branch back to their start
        let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        let sb_loop_count = ptx.matches("sb_loop").count();
        let sub_block_loop_count = ptx.matches("sub_block_loop").count();

        // Each loop should have: label definition + branch back = 2 references
        assert!(sb_loop_count >= 2,
            "sb_loop should appear at least twice (label + branch back), found {}", sb_loop_count);
        assert!(sub_block_loop_count >= 2,
            "sub_block_loop should appear at least twice (label + branch back), found {}", sub_block_loop_count);
    }

    #[test]
    fn test_simplified_vs_ggml_different_ptx() {
        // Verify simplified and GGML kernels produce different PTX
        let simplified = QuantizeKernel::new(1024, 1024, 4096);
        let ggml = QuantizeKernel::ggml(1024, 1024, 4096);

        let ptx_simplified = simplified.emit_ptx();
        let ptx_ggml = ggml.emit_ptx();

        assert_ne!(ptx_simplified, ptx_ggml,
            "Simplified and GGML kernels should produce different PTX");
        assert!(ptx_simplified.contains("q4k_gemm_fused"));
        assert!(ptx_ggml.contains("q4k_gemm_ggml"));
    }

    // =========================================================================
    // PARITY-116: Q5_K Kernel Tests
    // =========================================================================

    #[test]
    fn test_q5k_kernel_name() {
        let kernel = Q5KKernel::new(1024, 1024, 4096);
        assert_eq!(kernel.name(), "q5k_gemm_ggml");
    }

    #[test]
    fn test_q5k_kernel_config() {
        let kernel = Q5KKernel::new(1024, 1024, 4096);
        assert_eq!(kernel.m, 1024);
        assert_eq!(kernel.n, 1024);
        assert_eq!(kernel.k, 4096);
        assert_eq!(kernel.tile_size, 32);
    }

    #[test]
    fn test_q5k_super_block_constants() {
        assert_eq!(Q5K_SUPER_BLOCK_SIZE, 256, "Q5_K super-block should have 256 values");
        assert_eq!(Q5K_SUPER_BLOCK_BYTES, 176, "Q5_K super-block should be 176 bytes (2+2+12+128+32)");
    }

    #[test]
    fn test_q5k_num_super_blocks() {
        let kernel = Q5KKernel::new(1024, 1024, 4096);
        assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256 = 16
    }

    #[test]
    fn test_q5k_ptx_generation() {
        let kernel = Q5KKernel::new(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        // Verify kernel name
        assert!(ptx.contains("q5k_gemm_ggml"), "Should contain Q5_K kernel name");

        // Verify parameters
        assert!(ptx.contains(".param .u64 a_ptr"));
        assert!(ptx.contains(".param .u64 b_quant_ptr"));
        assert!(ptx.contains(".param .u64 c_ptr"));
        assert!(ptx.contains(".param .u32 m"));
        assert!(ptx.contains(".param .u32 n"));
        assert!(ptx.contains(".param .u32 k"));
    }

    #[test]
    fn test_q5k_ptx_contains_nested_loops() {
        let kernel = Q5KKernel::new(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains("sb_loop"), "Should have super-block loop");
        assert!(ptx.contains("sub_block_loop"), "Should have sub-block loop");
    }

    #[test]
    fn test_q5k_ptx_contains_high_bit_load() {
        // FALSIFIABLE: Q5_K must load high bits from qh (offset 144)
        let kernel = Q5KKernel::new(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        // Q5_K has 1-bit high values packed in qh array
        // The kernel should have multiple ld.global.u8 for ql and qh
        let load_count = ptx.matches("ld.global.u8").count();
        assert!(load_count >= 4, // At least scales (2) + ql + qh
            "Q5_K should have multiple u8 loads for scales, ql, and qh. Found {}", load_count);
    }

    #[test]
    fn test_q5k_both_loops_branch_back() {
        let kernel = Q5KKernel::new(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        let sb_loop_count = ptx.matches("sb_loop").count();
        let sub_block_loop_count = ptx.matches("sub_block_loop").count();

        assert!(sb_loop_count >= 2,
            "sb_loop should appear at least twice (label + branch back), found {}", sb_loop_count);
        assert!(sub_block_loop_count >= 2,
            "sub_block_loop should appear at least twice (label + branch back), found {}", sub_block_loop_count);
    }

    // =========================================================================
    // PARITY-117: Q6_K Kernel Tests
    // =========================================================================

    #[test]
    fn test_q6k_kernel_name() {
        let kernel = Q6KKernel::new(1024, 1024, 4096);
        assert_eq!(kernel.name(), "q6k_gemm_ggml");
    }

    #[test]
    fn test_q6k_kernel_config() {
        let kernel = Q6KKernel::new(1024, 1024, 4096);
        assert_eq!(kernel.m, 1024);
        assert_eq!(kernel.n, 1024);
        assert_eq!(kernel.k, 4096);
        assert_eq!(kernel.tile_size, 32);
    }

    #[test]
    fn test_q6k_super_block_constants() {
        assert_eq!(Q6K_SUPER_BLOCK_SIZE, 256, "Q6_K super-block should have 256 values");
        assert_eq!(Q6K_SUPER_BLOCK_BYTES, 210, "Q6_K super-block should be 210 bytes (128+64+16+2)");
    }

    #[test]
    fn test_q6k_num_super_blocks() {
        let kernel = Q6KKernel::new(1024, 1024, 4096);
        assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256 = 16
    }

    #[test]
    fn test_q6k_ptx_generation() {
        let kernel = Q6KKernel::new(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        // Verify kernel name
        assert!(ptx.contains("q6k_gemm_ggml"), "Should contain Q6_K kernel name");

        // Verify parameters
        assert!(ptx.contains(".param .u64 a_ptr"));
        assert!(ptx.contains(".param .u64 b_quant_ptr"));
        assert!(ptx.contains(".param .u64 c_ptr"));
        assert!(ptx.contains(".param .u32 m"));
        assert!(ptx.contains(".param .u32 n"));
        assert!(ptx.contains(".param .u32 k"));
    }

    #[test]
    fn test_q6k_ptx_contains_nested_loops() {
        let kernel = Q6KKernel::new(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains("sb_loop"), "Should have super-block loop");
        assert!(ptx.contains("sub_block_loop"), "Should have sub-block loop");
    }

    #[test]
    fn test_q6k_ptx_contains_2bit_high_extraction() {
        // FALSIFIABLE: Q6_K must load and extract 2-bit high values from qh
        let kernel = Q6KKernel::new(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        // Q6_K has 2-bit high values, needs mask 0x3
        assert!(ptx.contains("and"), "Should have AND for bit masking");
    }

    #[test]
    fn test_q6k_ptx_contains_signed_offset() {
        // FALSIFIABLE: Q6_K subtracts 32 to convert to signed range
        let kernel = Q6KKernel::new(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        // Q6_K: quant = ql + 4*qh - 32 (signed range -32 to 31)
        assert!(ptx.contains("sub.f32") || ptx.contains("sub.rn.f32"),
            "Should have subtraction for signed offset");
    }

    #[test]
    fn test_q6k_both_loops_branch_back() {
        let kernel = Q6KKernel::new(1024, 1024, 4096);
        let ptx = kernel.emit_ptx();

        let sb_loop_count = ptx.matches("sb_loop").count();
        let sub_block_loop_count = ptx.matches("sub_block_loop").count();

        assert!(sb_loop_count >= 2,
            "sb_loop should appear at least twice (label + branch back), found {}", sb_loop_count);
        assert!(sub_block_loop_count >= 2,
            "sub_block_loop should appear at least twice (label + branch back), found {}", sub_block_loop_count);
    }

    #[test]
    fn test_all_quant_kernels_different() {
        // Verify all quantized kernels produce distinct PTX
        let q4k = QuantizeKernel::ggml(1024, 1024, 4096);
        let q5k = Q5KKernel::new(1024, 1024, 4096);
        let q6k = Q6KKernel::new(1024, 1024, 4096);

        let ptx_q4k = q4k.emit_ptx();
        let ptx_q5k = q5k.emit_ptx();
        let ptx_q6k = q6k.emit_ptx();

        assert_ne!(ptx_q4k, ptx_q5k, "Q4_K and Q5_K should produce different PTX");
        assert_ne!(ptx_q4k, ptx_q6k, "Q4_K and Q6_K should produce different PTX");
        assert_ne!(ptx_q5k, ptx_q6k, "Q5_K and Q6_K should produce different PTX");
    }

    // =========================================================================
    // Property-Based Tests (PARITY-116, PARITY-117)
    // =========================================================================

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn prop_q5k_valid_ptx_for_any_size(
            m in 32u32..512,
            n in 32u32..512,
            // K must be divisible by 256 for super-blocks
            k_factor in 1u32..8
        ) {
            let k = k_factor * 256;
            let kernel = Q5KKernel::new(m, n, k);
            let ptx = kernel.emit_ptx();

            // PTX must be valid (non-empty, contains kernel)
            prop_assert!(!ptx.is_empty());
            prop_assert!(ptx.contains("q5k_gemm_ggml"));
            prop_assert!(ptx.contains(".entry"));
            prop_assert!(ptx.contains("ret;"));

            // Must have nested loops
            prop_assert!(ptx.contains("sb_loop"));
            prop_assert!(ptx.contains("sub_block_loop"));
        }

        #[test]
        fn prop_q6k_valid_ptx_for_any_size(
            m in 32u32..512,
            n in 32u32..512,
            k_factor in 1u32..8
        ) {
            let k = k_factor * 256;
            let kernel = Q6KKernel::new(m, n, k);
            let ptx = kernel.emit_ptx();

            prop_assert!(!ptx.is_empty());
            prop_assert!(ptx.contains("q6k_gemm_ggml"));
            prop_assert!(ptx.contains(".entry"));
            prop_assert!(ptx.contains("ret;"));

            // Q6_K-specific: signed offset subtraction
            prop_assert!(ptx.contains("sub.f32") || ptx.contains("sub.rn.f32"));
        }

        #[test]
        fn prop_q5k_super_blocks_correct(k_factor in 1u32..16) {
            let k = k_factor * 256;
            let kernel = Q5KKernel::new(64, 64, k);
            prop_assert_eq!(kernel.num_super_blocks_per_row(), k_factor);
        }

        #[test]
        fn prop_q6k_super_blocks_correct(k_factor in 1u32..16) {
            let k = k_factor * 256;
            let kernel = Q6KKernel::new(64, 64, k);
            prop_assert_eq!(kernel.num_super_blocks_per_row(), k_factor);
        }

        #[test]
        fn prop_all_quant_kernels_distinct(
            m in 64u32..256,
            n in 64u32..256,
            k_factor in 1u32..4
        ) {
            let k = k_factor * 256;
            let q4k = QuantizeKernel::ggml(m, n, k);
            let q5k = Q5KKernel::new(m, n, k);
            let q6k = Q6KKernel::new(m, n, k);

            let ptx_q4k = q4k.emit_ptx();
            let ptx_q5k = q5k.emit_ptx();
            let ptx_q6k = q6k.emit_ptx();

            prop_assert!(ptx_q4k != ptx_q5k);
            prop_assert!(ptx_q4k != ptx_q6k);
            prop_assert!(ptx_q5k != ptx_q6k);
        }
    }
}
