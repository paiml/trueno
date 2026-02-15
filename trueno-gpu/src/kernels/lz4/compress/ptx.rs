//! PTX code generation for the LZ4 warp-cooperative compression kernel
// Allow similar names for offset variables (imm4, imm8, imm12, etc. are intentionally named)
#![allow(clippy::similar_names)]

use super::Lz4WarpCompressKernel;
use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

use super::{
    LZ4_HASH_BITS, LZ4_HASH_MULT, LZ4_HASH_SIZE, LZ4_MAX_OFFSET, LZ4_MIN_MATCH, PAGE_SIZE,
};

impl Kernel for Lz4WarpCompressKernel {
    fn name(&self) -> &str {
        "lz4_compress_warp"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new(self.name())
            .param(PtxType::U64, "input_batch")
            .param(PtxType::U64, "output_batch")
            .param(PtxType::U64, "output_sizes")
            .param(PtxType::U32, "batch_size")
            .shared_memory(self.shared_memory_bytes())
            .build(|ctx| {
                // Phase 1: Calculate page assignment
                let block_id = ctx.special_reg(PtxReg::CtaIdX);
                let thread_id = ctx.special_reg(PtxReg::TidX);

                // warp_id = threadIdx.x / 32
                let shift_5 = ctx.mov_u32_imm(5);
                let warp_id = ctx.shr_u32(thread_id, shift_5);

                // page_id = blockIdx.x * 4 + warp_id
                let page_id = ctx.mul_wide_u32(block_id, 4);
                let warp_id_64 = ctx.cvt_u64_u32(warp_id);
                let page_id = ctx.add_u64(page_id, warp_id_64);
                let page_id_32 = ctx.cvt_u32_u64(page_id);

                // lane_id = threadIdx.x % 32
                let mask_31 = ctx.mov_u32_imm(31);
                let lane_id = ctx.and_u32(thread_id, mask_31);

                // Phase 2: Bounds check
                // CRITICAL: Do NOT early-exit here! All threads must participate in barriers.
                // Use predicated execution instead - out-of-bounds threads skip memory ops
                // but still reach barriers to prevent deadlock.
                let batch_param = ctx.load_param_u32("batch_size");
                let in_bounds_pred = ctx.setp_lt_u32(page_id_32, batch_param);

                // Phase 3: Calculate pointers (safe even if out of bounds - we just won't use them)
                let page_offset = ctx.mul_wide_u32(page_id_32, PAGE_SIZE);
                let input_ptr = ctx.load_param_u64("input_batch");
                let input_page_ptr = ctx.add_u64(input_ptr, page_offset);
                let output_ptr = ctx.load_param_u64("output_batch");
                let output_page_ptr = ctx.add_u64(output_ptr, page_offset);

                // Phase 4: Cooperative load into shared memory
                // Each thread loads 128 bytes (4 x 32-byte chunks)
                //
                // CRITICAL FIX: Partition shared memory by warp_id
                // Each warp gets (PAGE_SIZE + LZ4_HASH_SIZE * 2) = 12KB
                // Without this, multiple warps in a block overwrite each other's data
                const WARP_SMEM_SIZE: u32 = PAGE_SIZE + LZ4_HASH_SIZE * 2; // 12288 bytes
                let warp_smem_offset = ctx.mul_u32(warp_id, WARP_SMEM_SIZE);
                let warp_smem_offset_64 = ctx.cvt_u64_u32(warp_smem_offset);
                // Use shared_base_addr() to get generic address (via cvta.to.shared)
                // Then use ld_generic/st_generic for memory operations
                let raw_smem_base = ctx.shared_base_addr();
                let smem_base = ctx.add_u64(raw_smem_base, warp_smem_offset_64);

                let load_base = ctx.mul_u32(lane_id, 128);

                // Pre-compute offset immediates (avoid nested mutable borrows)
                let imm4 = ctx.mov_u64_imm(4);
                let imm8 = ctx.mov_u64_imm(8);
                let imm12 = ctx.mov_u64_imm(12);
                let imm16 = ctx.mov_u64_imm(16);
                let imm20 = ctx.mov_u64_imm(20);
                let imm24 = ctx.mov_u64_imm(24);
                let imm28 = ctx.mov_u64_imm(28);

                // Skip global loads if out of bounds (but still do shared mem to avoid uninitialized reads)
                ctx.branch_if_not(in_bounds_pred, "L_skip_global_load");

                // Load/store 128 bytes per thread (4 x 32-byte chunks)
                for i in 0..4u32 {
                    let chunk_off = ctx.add_u32(load_base, i * 32);
                    let chunk_off_64 = ctx.cvt_u64_u32(chunk_off);
                    let load_addr = ctx.add_u64(input_page_ptr, chunk_off_64);

                    // Load 32 bytes (8 x u32)
                    let d0 = ctx.ld_global_u32(load_addr);
                    let off4 = ctx.add_u64(load_addr, imm4);
                    let d1 = ctx.ld_global_u32(off4);
                    let off8 = ctx.add_u64(load_addr, imm8);
                    let d2 = ctx.ld_global_u32(off8);
                    let off12 = ctx.add_u64(load_addr, imm12);
                    let d3 = ctx.ld_global_u32(off12);
                    let off16 = ctx.add_u64(load_addr, imm16);
                    let d4 = ctx.ld_global_u32(off16);
                    let off20 = ctx.add_u64(load_addr, imm20);
                    let d5 = ctx.ld_global_u32(off20);
                    let off24 = ctx.add_u64(load_addr, imm24);
                    let d6 = ctx.ld_global_u32(off24);
                    let off28 = ctx.add_u64(load_addr, imm28);
                    let d7 = ctx.ld_global_u32(off28);

                    // Store to shared memory
                    let smem_off = ctx.add_u64(smem_base, chunk_off_64);
                    ctx.st_generic_u32(smem_off, d0);
                    let smem_4 = ctx.add_u64(smem_off, imm4);
                    ctx.st_generic_u32(smem_4, d1);
                    let smem_8 = ctx.add_u64(smem_off, imm8);
                    ctx.st_generic_u32(smem_8, d2);
                    let smem_12 = ctx.add_u64(smem_off, imm12);
                    ctx.st_generic_u32(smem_12, d3);
                    let smem_16 = ctx.add_u64(smem_off, imm16);
                    ctx.st_generic_u32(smem_16, d4);
                    let smem_20 = ctx.add_u64(smem_off, imm20);
                    ctx.st_generic_u32(smem_20, d5);
                    let smem_24 = ctx.add_u64(smem_off, imm24);
                    ctx.st_generic_u32(smem_24, d6);
                    let smem_28 = ctx.add_u64(smem_off, imm28);
                    ctx.st_generic_u32(smem_28, d7);
                }
                ctx.branch("L_after_global_load");

                ctx.label("L_skip_global_load");
                // Initialize shared memory to zeros for out-of-bounds warps
                // (so subsequent reads don't access uninitialized memory)
                let zero_val = ctx.mov_u32_imm(0);
                for i in 0..4u32 {
                    let chunk_off = ctx.add_u32(load_base, i * 32);
                    let chunk_off_64 = ctx.cvt_u64_u32(chunk_off);
                    let smem_off = ctx.add_u64(smem_base, chunk_off_64);
                    ctx.st_generic_u32(smem_off, zero_val);
                    let smem_4 = ctx.add_u64(smem_off, imm4);
                    ctx.st_generic_u32(smem_4, zero_val);
                    let smem_8 = ctx.add_u64(smem_off, imm8);
                    ctx.st_generic_u32(smem_8, zero_val);
                    let smem_12 = ctx.add_u64(smem_off, imm12);
                    ctx.st_generic_u32(smem_12, zero_val);
                    let smem_16 = ctx.add_u64(smem_off, imm16);
                    ctx.st_generic_u32(smem_16, zero_val);
                    let smem_20 = ctx.add_u64(smem_off, imm20);
                    ctx.st_generic_u32(smem_20, zero_val);
                    let smem_24 = ctx.add_u64(smem_off, imm24);
                    ctx.st_generic_u32(smem_24, zero_val);
                    let smem_28 = ctx.add_u64(smem_off, imm28);
                    ctx.st_generic_u32(smem_28, zero_val);
                }

                ctx.label("L_after_global_load");
                // Barrier to ensure all data is loaded
                ctx.bar_sync(0);

                // Phase 5: LZ4 compression with zero-page detection
                //
                // Each warp detects if its page is all zeros using parallel reduction.
                // Zero pages compress to a minimal LZ4 sequence (significant memory savings).
                // Non-zero pages get full LZ4 compression via hash-based matching.

                // Step 5a: Check if thread's 128 bytes are all zeros (load 8 u32s, OR them)
                let chunk_val = ctx.mov_u32_imm(0);
                for i in 0..4u32 {
                    let chunk_off = ctx.add_u32(load_base, i * 32);
                    let chunk_off_64 = ctx.cvt_u64_u32(chunk_off);
                    let smem_off = ctx.add_u64(smem_base, chunk_off_64);

                    // Load 8 u32s and OR them together
                    let d0 = ctx.ld_generic_u32(smem_off);
                    let chunk_val = ctx.or_u32(chunk_val, d0);
                    let off4 = ctx.add_u64(smem_off, imm4);
                    let d1 = ctx.ld_generic_u32(off4);
                    let chunk_val = ctx.or_u32(chunk_val, d1);
                    let off8 = ctx.add_u64(smem_off, imm8);
                    let d2 = ctx.ld_generic_u32(off8);
                    let chunk_val = ctx.or_u32(chunk_val, d2);
                    let off12 = ctx.add_u64(smem_off, imm12);
                    let d3 = ctx.ld_generic_u32(off12);
                    let chunk_val = ctx.or_u32(chunk_val, d3);
                    let off16 = ctx.add_u64(smem_off, imm16);
                    let d4 = ctx.ld_generic_u32(off16);
                    let chunk_val = ctx.or_u32(chunk_val, d4);
                    let off20 = ctx.add_u64(smem_off, imm20);
                    let d5 = ctx.ld_generic_u32(off20);
                    let chunk_val = ctx.or_u32(chunk_val, d5);
                    let off24 = ctx.add_u64(smem_off, imm24);
                    let d6 = ctx.ld_generic_u32(off24);
                    let chunk_val = ctx.or_u32(chunk_val, d6);
                    let off28 = ctx.add_u64(smem_off, imm28);
                    let d7 = ctx.ld_generic_u32(off28);
                    let _ = ctx.or_u32(chunk_val, d7);
                }

                // Step 5b: Warp-level reduction to check if entire page is zeros
                // Each thread has its portion's OR value in chunk_val
                // Use warp shuffle to combine all 32 lanes' values
                // (For simplicity, we use shared memory reduction here)

                // Store each thread's chunk result to shared memory for reduction
                // CRITICAL: lane_id * 4 for 4-byte alignment of u32 stores
                let lane_off_bytes = ctx.mul_u32(lane_id, 4);
                let reduction_off = ctx.add_u32(lane_off_bytes, PAGE_SIZE); // Use space after page data
                let reduction_off_64 = ctx.cvt_u64_u32(reduction_off);
                let reduction_addr = ctx.add_u64(smem_base, reduction_off_64);
                ctx.st_generic_u32(reduction_addr, chunk_val);
                ctx.bar_sync(0);

                // Lane 0 reduces all 32 values (only if in bounds)
                let zero = ctx.mov_u32_imm(0);
                let is_leader = ctx.setp_eq_u32(lane_id, zero);
                // Combined check: must be leader AND in bounds to write size
                let can_write_size = ctx.and_pred(is_leader, in_bounds_pred);
                ctx.branch_if_not(can_write_size, "L_not_leader");

                // Leader: Read and OR all 32 values
                let page_or = ctx.mov_u32_imm(0);
                for lane in 0..32u32 {
                    let lane_off = ctx.mov_u32_imm(PAGE_SIZE + lane * 4);
                    let lane_off_64 = ctx.cvt_u64_u32(lane_off);
                    let lane_addr = ctx.add_u64(smem_base, lane_off_64);
                    let lane_val = ctx.ld_generic_u32(lane_addr);
                    let _ = ctx.or_u32(page_or, lane_val);
                }

                // Check if page is all zeros
                let is_zero_page = ctx.setp_eq_u32(page_or, zero);

                // Write output size based on whether page is zeros
                let size_ptr = ctx.load_param_u64("output_sizes");
                let size_off = ctx.mul_wide_u32(page_id_32, 4);
                let size_addr = ctx.add_u64(size_ptr, size_off);

                // If zero page: compressed size is just 20 bytes (LZ4 sequence for 4KB zeros)
                // If non-zero: for now, output uncompressed size (full LZ4 to be implemented)
                let compressed_zero_size = ctx.mov_u32_imm(20); // Token + extended length + match
                let uncompressed_size = ctx.mov_u32_imm(PAGE_SIZE);
                ctx.branch_if(is_zero_page, "L_write_zero_size");

                // Non-zero path: LZ4 compression
                // Jump to compression loop
                ctx.branch("L_compress_start");

                ctx.label("L_compress_done");
                // After compression, write the actual compressed size (out_pos_final)
                // out_pos_final is set in L_emit_remaining before jumping here
                ctx.branch("L_write_compressed_size");

                // ================================================================
                // LZ4 COMPRESSION LOOP
                // ================================================================
                ctx.label("L_compress_start");

                // LZ4 hash constant: 0x9E3779B1 = 2654435761
                let lz4_prime = ctx.mov_u32_imm(LZ4_HASH_MULT);
                let hash_shift = ctx.mov_u32_imm(32 - LZ4_HASH_BITS); // 20
                let hash_mask = ctx.mov_u32_imm(LZ4_HASH_SIZE - 1); // 4095

                // Compression state - initialize in shared memory for persistence across loop
                // Use warp's hash table area (after PAGE_SIZE + 8KB hash table = offset 12288)
                // We'll use a small area after hash table for state variables
                let state_off = ctx.mov_u32_imm(PAGE_SIZE + LZ4_HASH_SIZE * 2); // 12288
                let state_off_64 = ctx.cvt_u64_u32(state_off);
                let state_base = ctx.add_u64(smem_base, state_off_64);

                // State layout in shared memory:
                // offset 0: in_pos (u32)
                // offset 4: out_pos (u32)
                // offset 8: anchor (u32)

                // Initialize state to 0
                let zero_val = ctx.mov_u32_imm(0);
                ctx.st_generic_u32(state_base, zero_val); // in_pos = 0
                let imm4_state = ctx.mov_u64_imm(4);
                let out_pos_addr = ctx.add_u64(state_base, imm4_state);
                ctx.st_generic_u32(out_pos_addr, zero_val); // out_pos = 0
                let imm8_state = ctx.mov_u64_imm(8);
                let anchor_addr = ctx.add_u64(state_base, imm8_state);
                ctx.st_generic_u32(anchor_addr, zero_val); // anchor = 0

                let limit = ctx.mov_u32_imm(PAGE_SIZE - 12); // LAST_LITERALS + MIN_MATCH

                // Main compression loop
                ctx.label("L_compress_loop");

                // Load current state
                let in_pos = ctx.ld_generic_u32(state_base);
                let out_pos = ctx.ld_generic_u32(out_pos_addr);
                let anchor = ctx.ld_generic_u32(anchor_addr);

                // Check bounds: if in_pos >= limit, emit remaining literals
                let at_limit = ctx.setp_ge_u32(in_pos, limit);
                ctx.branch_if(at_limit, "L_emit_remaining");

                // Load 4 bytes at current position for hashing
                let in_pos_64 = ctx.cvt_u64_u32(in_pos);
                let curr_addr = ctx.add_u64(smem_base, in_pos_64);
                let curr_val = ctx.ld_generic_u32(curr_addr);

                // Compute hash: ((val * prime) >> 20) & 0xFFF
                let hash_tmp = ctx.mul_lo_u32(curr_val, lz4_prime);
                let hash_shifted = ctx.shr_u32(hash_tmp, hash_shift);
                let hash_idx = ctx.and_u32(hash_shifted, hash_mask);

                // Hash table is after page data in shared memory
                // hash_table_base = smem_base + PAGE_SIZE
                let hash_table_off = ctx.mov_u32_imm(PAGE_SIZE);
                let hash_table_off_64 = ctx.cvt_u64_u32(hash_table_off);
                let hash_table_base = ctx.add_u64(smem_base, hash_table_off_64);

                // Look up hash table (2 bytes per entry for position 0-4095)
                let hash_entry_off = ctx.mul_u32(hash_idx, 2);
                let hash_entry_off_64 = ctx.cvt_u64_u32(hash_entry_off);
                let hash_entry_addr = ctx.add_u64(hash_table_base, hash_entry_off_64);

                // Load match candidate position (u16, zero-extended to u32)
                let match_pos_u16 = ctx.ld_generic_u16(hash_entry_addr);
                let match_pos = ctx.cvt_u32_u16(match_pos_u16);

                // Update hash table with current position (truncated to u16)
                ctx.st_generic_u16(hash_entry_addr, in_pos);

                // Check if we have a valid match candidate (0xFFFF = invalid for u16)
                let invalid_marker = ctx.mov_u32_imm(0xFFFF);
                let no_match_candidate = ctx.setp_eq_u32(match_pos, invalid_marker);
                ctx.branch_if(no_match_candidate, "L_no_match");

                // Check if match is within valid offset (<= 65535)
                let offset = ctx.sub_u32_reg(in_pos, match_pos);
                let max_offset_plus_one = ctx.mov_u32_imm(LZ4_MAX_OFFSET + 1);
                let offset_too_large = ctx.setp_ge_u32(offset, max_offset_plus_one);
                ctx.branch_if(offset_too_large, "L_no_match");

                // Zero offset is invalid
                let zero_check = ctx.mov_u32_imm(0);
                let offset_is_zero = ctx.setp_eq_u32(offset, zero_check);
                ctx.branch_if(offset_is_zero, "L_no_match");

                // Load 4 bytes at match position and compare
                let match_pos_64 = ctx.cvt_u64_u32(match_pos);
                let match_addr = ctx.add_u64(smem_base, match_pos_64);
                let match_val = ctx.ld_generic_u32(match_addr);

                // Check if first 4 bytes match
                ctx.label("L_check_match");
                let vals_equal = ctx.setp_eq_u32(curr_val, match_val);
                ctx.branch_if_not(vals_equal, "L_no_match");

                // ============================================================
                // Found a match! Encode the LZ4 sequence
                // ============================================================
                ctx.label("L_found_match");
                ctx.label("L_encode_sequence");

                // Calculate literal length: in_pos - anchor
                let literal_len = ctx.sub_u32_reg(in_pos, anchor);

                // Match length is minimum 4 (we matched 4 bytes)
                // For simplicity, use minimum match length without extension
                let _match_len = ctx.mov_u32_imm(LZ4_MIN_MATCH);

                // Build token byte: (min(literal_len, 15) << 4) | min(match_len - 4, 15)
                // match_len - 4 = 0, so token_match = 0
                let fifteen = ctx.mov_u32_imm(15);
                let lit_ge_15 = ctx.setp_ge_u32(literal_len, fifteen);
                let token_lit = ctx.selp_u32(lit_ge_15, fifteen, literal_len);
                let four_bits = ctx.mov_u32_imm(4);
                let token_lit_shifted = ctx.shl_u32(token_lit, four_bits);
                // match_len - 4 = 0, token_match = 0
                let token = token_lit_shifted; // | 0

                // Write token byte to output
                let out_pos_64 = ctx.cvt_u64_u32(out_pos);
                let out_addr = ctx.add_u64(output_page_ptr, out_pos_64);
                ctx.st_global_u8(out_addr, token);
                let _one_32 = ctx.mov_u32_imm(1);
                let out_pos_1 = ctx.add_u32(out_pos, 1);

                // Write extended literal length if >= 15
                ctx.branch_if_not(lit_ge_15, "L_skip_ext_lit_len");

                // Extended length: write (literal_len - 15) as series of 255s + remainder
                // Simplified: just write single byte for lengths < 270
                let lit_minus_15 = ctx.sub_u32_reg(literal_len, fifteen);
                let out_pos_1_64 = ctx.cvt_u64_u32(out_pos_1);
                let ext_addr = ctx.add_u64(output_page_ptr, out_pos_1_64);
                ctx.st_global_u8(ext_addr, lit_minus_15);
                let out_pos_2 = ctx.add_u32(out_pos_1, 1);
                // Store updated out_pos for use after branch
                ctx.st_generic_u32(out_pos_addr, out_pos_2);
                ctx.branch("L_copy_literals");

                ctx.label("L_skip_ext_lit_len");
                ctx.st_generic_u32(out_pos_addr, out_pos_1);

                // Copy literals from shared memory to output
                ctx.label("L_copy_literals");
                // Reload out_pos after potential update
                let out_pos_cur = ctx.ld_generic_u32(out_pos_addr);

                // Copy loop: copy literal_len bytes from smem[anchor] to output[out_pos]
                // For GPU efficiency, copy 4 bytes at a time when possible
                let lit_copy_idx = ctx.mov_u32_imm(0);
                ctx.st_generic_u32(state_base, lit_copy_idx); // Reuse in_pos slot temporarily

                ctx.label("L_copy_lit_loop");
                let copy_idx = ctx.ld_generic_u32(state_base);
                let copy_done = ctx.setp_ge_u32(copy_idx, literal_len);
                ctx.branch_if(copy_done, "L_copy_lit_done");

                // Load byte from smem[anchor + copy_idx]
                let src_off = ctx.add_u32_reg(anchor, copy_idx);
                let src_off_64 = ctx.cvt_u64_u32(src_off);
                let src_addr = ctx.add_u64(smem_base, src_off_64);
                let byte_val = ctx.ld_generic_u8(src_addr);

                // Store to output[out_pos_cur + copy_idx]
                let dst_off = ctx.add_u32_reg(out_pos_cur, copy_idx);
                let dst_off_64 = ctx.cvt_u64_u32(dst_off);
                let dst_addr = ctx.add_u64(output_page_ptr, dst_off_64);
                ctx.st_global_u8(dst_addr, byte_val);

                // Increment copy index
                let copy_idx_next = ctx.add_u32(copy_idx, 1);
                ctx.st_generic_u32(state_base, copy_idx_next);
                ctx.branch("L_copy_lit_loop");

                ctx.label("L_copy_lit_done");
                // Update out_pos after copying literals
                let out_pos_after_lit = ctx.add_u32_reg(out_pos_cur, literal_len);

                // Write match offset (2 bytes, little-endian)
                let out_pos_after_lit_64 = ctx.cvt_u64_u32(out_pos_after_lit);
                let offset_addr = ctx.add_u64(output_page_ptr, out_pos_after_lit_64);
                let mask_ff = ctx.mov_u32_imm(0xFF);
                let offset_lo = ctx.and_u32(offset, mask_ff);
                ctx.st_global_u8(offset_addr, offset_lo);

                let imm_1 = ctx.mov_u64_imm(1);
                let offset_addr_1 = ctx.add_u64(offset_addr, imm_1);
                let eight_bits = ctx.mov_u32_imm(8);
                let offset_hi = ctx.shr_u32(offset, eight_bits);
                ctx.st_global_u8(offset_addr_1, offset_hi);

                let out_pos_after_offset = ctx.add_u32(out_pos_after_lit, 2);

                // Update state for next iteration
                // new_anchor = in_pos + match_len
                let new_anchor = ctx.add_u32(in_pos, LZ4_MIN_MATCH);
                ctx.st_generic_u32(anchor_addr, new_anchor);
                // new_in_pos = in_pos + match_len
                ctx.st_generic_u32(state_base, new_anchor);
                // new_out_pos
                ctx.st_generic_u32(out_pos_addr, out_pos_after_offset);

                ctx.branch("L_compress_loop");

                ctx.label("L_no_match");
                // No match, advance in_pos by 1
                let in_pos_next = ctx.add_u32(in_pos, 1);
                ctx.st_generic_u32(state_base, in_pos_next);
                ctx.branch("L_compress_loop");

                // ============================================================
                // Emit remaining literals (final sequence with no match)
                // ============================================================
                ctx.label("L_emit_remaining");

                // Reload state
                let _final_in_pos = ctx.ld_generic_u32(state_base);
                let final_out_pos = ctx.ld_generic_u32(out_pos_addr);
                let final_anchor = ctx.ld_generic_u32(anchor_addr);

                // literal_len = PAGE_SIZE - anchor
                let page_size_val = ctx.mov_u32_imm(PAGE_SIZE);
                let final_lit_len = ctx.sub_u32_reg(page_size_val, final_anchor);

                // If no remaining literals, we're done
                let no_remaining = ctx.setp_eq_u32(final_lit_len, zero_val);
                ctx.branch_if(no_remaining, "L_finalize_size");

                // Write token (no match, so token_match = 0)
                let final_lit_ge_15 = ctx.setp_ge_u32(final_lit_len, fifteen);
                let final_token_lit = ctx.selp_u32(final_lit_ge_15, fifteen, final_lit_len);
                let final_token = ctx.shl_u32(final_token_lit, four_bits);

                let final_out_64 = ctx.cvt_u64_u32(final_out_pos);
                let final_token_addr = ctx.add_u64(output_page_ptr, final_out_64);
                ctx.st_global_u8(final_token_addr, final_token);
                let final_out_1 = ctx.add_u32(final_out_pos, 1);

                // Extended literal length if >= 15
                ctx.branch_if_not(final_lit_ge_15, "L_final_copy_literals");
                let final_ext_len = ctx.sub_u32_reg(final_lit_len, fifteen);
                let final_ext_64 = ctx.cvt_u64_u32(final_out_1);
                let final_ext_addr = ctx.add_u64(output_page_ptr, final_ext_64);
                ctx.st_global_u8(final_ext_addr, final_ext_len);
                let final_out_2 = ctx.add_u32(final_out_1, 1);
                ctx.st_generic_u32(out_pos_addr, final_out_2);
                ctx.branch("L_final_do_copy");

                ctx.label("L_final_copy_literals");
                ctx.st_generic_u32(out_pos_addr, final_out_1);

                ctx.label("L_final_do_copy");
                // Copy remaining literals
                let final_out_cur = ctx.ld_generic_u32(out_pos_addr);
                let final_copy_idx = ctx.mov_u32_imm(0);
                ctx.st_generic_u32(state_base, final_copy_idx);

                ctx.label("L_final_copy_loop");
                let fcopy_idx = ctx.ld_generic_u32(state_base);
                let fcopy_done = ctx.setp_ge_u32(fcopy_idx, final_lit_len);
                ctx.branch_if(fcopy_done, "L_finalize_size");

                let fsrc_off = ctx.add_u32_reg(final_anchor, fcopy_idx);
                let fsrc_off_64 = ctx.cvt_u64_u32(fsrc_off);
                let fsrc_addr = ctx.add_u64(smem_base, fsrc_off_64);
                let fbyte_val = ctx.ld_generic_u8(fsrc_addr);

                let fdst_off = ctx.add_u32_reg(final_out_cur, fcopy_idx);
                let fdst_off_64 = ctx.cvt_u64_u32(fdst_off);
                let fdst_addr = ctx.add_u64(output_page_ptr, fdst_off_64);
                ctx.st_global_u8(fdst_addr, fbyte_val);

                let fcopy_next = ctx.add_u32(fcopy_idx, 1);
                ctx.st_generic_u32(state_base, fcopy_next);
                ctx.branch("L_final_copy_loop");

                ctx.label("L_finalize_size");
                // Calculate final compressed size
                let final_size = ctx.ld_generic_u32(out_pos_addr);
                // Add remaining literals that were just copied
                let total_size = ctx.add_u32_reg(final_size, final_lit_len);
                // Store for use at L_write_compressed_size
                ctx.st_generic_u32(out_pos_addr, total_size);

                // Warp lane coordination
                ctx.label("L_lane_sync");
                ctx.bar_sync(2);
                ctx.branch("L_compress_done");

                ctx.label("L_write_compressed_size");
                let compressed_size = ctx.ld_generic_u32(out_pos_addr);
                // If compression expanded the data, store as uncompressed
                let expanded = ctx.setp_ge_u32(compressed_size, uncompressed_size);
                let final_reported_size =
                    ctx.selp_u32(expanded, uncompressed_size, compressed_size);
                ctx.st_global_u32(size_addr, final_reported_size);
                ctx.branch("L_after_size_write");

                ctx.label("L_write_zero_size");
                ctx.st_global_u32(size_addr, compressed_zero_size);

                ctx.label("L_after_size_write");
                ctx.label("L_not_leader");
                ctx.bar_sync(0);

                // Phase 6: Cooperative store to output (skip if out of bounds)
                ctx.branch_if_not(in_bounds_pred, "L_exit");

                for i in 0..4u32 {
                    let chunk_off = ctx.add_u32(load_base, i * 32);
                    let chunk_off_64 = ctx.cvt_u64_u32(chunk_off);
                    let smem_off = ctx.add_u64(smem_base, chunk_off_64);

                    // Load from shared memory
                    let d0 = ctx.ld_generic_u32(smem_off);
                    let ld_4 = ctx.add_u64(smem_off, imm4);
                    let d1 = ctx.ld_generic_u32(ld_4);
                    let ld_8 = ctx.add_u64(smem_off, imm8);
                    let d2 = ctx.ld_generic_u32(ld_8);
                    let ld_12 = ctx.add_u64(smem_off, imm12);
                    let d3 = ctx.ld_generic_u32(ld_12);
                    let ld_16 = ctx.add_u64(smem_off, imm16);
                    let d4 = ctx.ld_generic_u32(ld_16);
                    let ld_20 = ctx.add_u64(smem_off, imm20);
                    let d5 = ctx.ld_generic_u32(ld_20);
                    let ld_24 = ctx.add_u64(smem_off, imm24);
                    let d6 = ctx.ld_generic_u32(ld_24);
                    let ld_28 = ctx.add_u64(smem_off, imm28);
                    let d7 = ctx.ld_generic_u32(ld_28);

                    // Store to global output
                    let store_addr = ctx.add_u64(output_page_ptr, chunk_off_64);
                    ctx.st_global_u32(store_addr, d0);
                    let st_4 = ctx.add_u64(store_addr, imm4);
                    ctx.st_global_u32(st_4, d1);
                    let st_8 = ctx.add_u64(store_addr, imm8);
                    ctx.st_global_u32(st_8, d2);
                    let st_12 = ctx.add_u64(store_addr, imm12);
                    ctx.st_global_u32(st_12, d3);
                    let st_16 = ctx.add_u64(store_addr, imm16);
                    ctx.st_global_u32(st_16, d4);
                    let st_20 = ctx.add_u64(store_addr, imm20);
                    ctx.st_global_u32(st_20, d5);
                    let st_24 = ctx.add_u64(store_addr, imm24);
                    ctx.st_global_u32(st_24, d6);
                    let st_28 = ctx.add_u64(store_addr, imm28);
                    ctx.st_global_u32(st_28, d7);
                }

                ctx.label("L_exit");
            })
    }
}
