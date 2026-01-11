//! GPU ArgMax Kernel for Greedy Sampling
//!
//! PAR-062: Implements GPU-side argmax to eliminate costly logits copy
//! from GPU to CPU (152064 floats = ~600KB per token).
//!
//! Instead of copying all logits, we compute argmax on GPU and only copy
//! the resulting token ID (4 bytes) - a 150,000x reduction in transfer size.
//!
//! ## Algorithm
//!
//! Two-kernel approach:
//! 1. Per-block reduction finds local (max_val, max_idx) using shared memory
//! 2. Final reduction across block results to find global argmax
//!
//! ## Performance Target
//!
//! - Before: 600KB D2H copy per token (~3ms on PCIe)
//! - After: 4B D2H copy per token (~0.001ms)
//! - Expected speedup: ~1.2x overall (from ~163 tok/s to ~200 tok/s)

use super::Kernel;
use crate::ptx::{PtxKernel, PtxType};

/// ArgMax kernel configuration
///
/// Finds the index of the maximum value in a float array.
/// Uses block-level reduction with warp shuffle for efficiency.
#[derive(Debug, Clone)]
pub struct ArgMaxKernel {
    /// Total vector length (vocab_size)
    pub length: u32,
}

impl ArgMaxKernel {
    /// Create a new argmax kernel for the given vector length
    #[must_use]
    pub fn new(length: u32) -> Self {
        Self { length }
    }

    /// Get recommended block size for this kernel
    #[must_use]
    pub fn block_size(&self) -> u32 {
        256 // Good balance between occupancy and shared memory
    }

    /// Get number of blocks needed
    #[must_use]
    pub fn num_blocks(&self) -> u32 {
        // Each thread handles multiple elements via striding
        let elements_per_block = self.block_size() * 4; // 4 elements per thread
        (self.length + elements_per_block - 1) / elements_per_block
    }
}

impl Kernel for ArgMaxKernel {
    fn name(&self) -> &str {
        "argmax_block_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Block-level argmax reduction kernel
        //
        // Each block processes a portion of the input and outputs:
        // - block_max_vals[block_id]: maximum value found by this block
        // - block_max_idxs[block_id]: index of that maximum
        //
        // A second pass (or CPU reduction) finds global max from block results.

        PtxKernel::new("argmax_block_reduce")
            .param(PtxType::U64, "input_ptr")       // f32* input values
            .param(PtxType::U64, "block_max_vals")  // f32* per-block max values
            .param(PtxType::U64, "block_max_idxs")  // u32* per-block max indices
            .param(PtxType::U32, "length")          // Total number of elements
            .shared_memory(256 * 8)                 // 256 * (f32 + u32) = 2KB
            .build(|ctx| {
                // Thread and block IDs
                let tid = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let bid = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let block_dim = ctx.special_reg(crate::ptx::PtxReg::NtidX);

                // Load parameters
                let input_ptr = ctx.load_param_u64("input_ptr");
                let block_max_vals = ctx.load_param_u64("block_max_vals");
                let block_max_idxs = ctx.load_param_u64("block_max_idxs");
                let length = ctx.load_param_u32("length");

                // Shared memory layout: [max_vals (256 f32), max_idxs (256 u32)]
                let shared_base = ctx.shared_ptr();

                // Calculate global start index for this block
                // Each block handles blockDim * 4 elements
                let four = ctx.const_u32(4);
                let elements_per_block = ctx.mul_lo_u32(block_dim, four);
                let block_start = ctx.mul_lo_u32(bid, elements_per_block);

                // Thread's starting index: block_start + tid
                let thread_start = ctx.add_u32_reg(block_start, tid);

                // Initialize with negative infinity and index 0
                let neg_inf = ctx.const_f32(f32::NEG_INFINITY);
                let mut local_max = neg_inf;
                let mut local_idx = ctx.const_u32(0);

                // Stride loop: each thread processes elements at stride blockDim
                // Thread 0: elements 0, 256, 512, 768
                // Thread 1: elements 1, 257, 513, 769
                // etc.
                for i in 0..4u32 {
                    let stride = ctx.mul_u32(block_dim, i);
                    let idx = ctx.add_u32_reg(thread_start, stride);

                    // Bounds check
                    let in_bounds = ctx.setp_lt_u32(idx, length);
                    ctx.branch_if_not(in_bounds, &format!("skip_load_{}", i));

                    // Load value
                    let byte_offset = ctx.mul_wide_u32(idx, 4);
                    let addr = ctx.add_u64(input_ptr, byte_offset);
                    let val = ctx.ld_global_f32(addr);

                    // Update local max if this value is greater
                    let is_greater = ctx.setp_gt_f32(val, local_max);
                    local_max = ctx.selp_f32(is_greater, val, local_max);
                    local_idx = ctx.selp_u32(is_greater, idx, local_idx);

                    ctx.label(&format!("skip_load_{}", i));
                }

                // Store to shared memory
                // Offset for this thread: tid * 4 bytes
                let sh_val_offset = ctx.mul_wide_u32(tid, 4);
                let sh_val_addr = ctx.add_u64(shared_base, sh_val_offset);
                ctx.st_shared_f32(sh_val_addr, local_max);

                // Index array starts after 256 floats (1024 bytes)
                let offset_1024 = ctx.mov_u64_imm(1024);
                let idx_base = ctx.add_u64(shared_base, offset_1024);
                let sh_idx_addr = ctx.add_u64(idx_base, sh_val_offset);
                ctx.st_shared_u32(sh_idx_addr, local_idx);

                // Synchronize before reduction
                ctx.bar_sync(0);

                // Tree reduction in shared memory
                // 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
                let stride_128 = ctx.const_u32(128);
                let is_active_128 = ctx.setp_lt_u32(tid, stride_128);
                ctx.branch_if_not(is_active_128, "skip_reduce_128");
                {
                    // Load other value from tid + 128
                    let other_tid = ctx.add_u32_reg(tid, stride_128);
                    let other_off = ctx.mul_wide_u32(other_tid, 4);
                    let other_val_addr = ctx.add_u64(shared_base, other_off);
                    let other_val = ctx.ld_shared_f32(other_val_addr);
                    let other_idx_addr = ctx.add_u64(idx_base, other_off);
                    let other_idx = ctx.ld_shared_u32(other_idx_addr);

                    let my_val = ctx.ld_shared_f32(sh_val_addr);
                    let my_idx = ctx.ld_shared_u32(sh_idx_addr);

                    let is_greater = ctx.setp_gt_f32(other_val, my_val);
                    let new_val = ctx.selp_f32(is_greater, other_val, my_val);
                    let new_idx = ctx.selp_u32(is_greater, other_idx, my_idx);

                    ctx.st_shared_f32(sh_val_addr, new_val);
                    ctx.st_shared_u32(sh_idx_addr, new_idx);
                }
                ctx.label("skip_reduce_128");
                ctx.bar_sync(0);

                // Continue reduction for smaller strides
                for stride in [64u32, 32, 16, 8, 4, 2, 1] {
                    let stride_reg = ctx.const_u32(stride);
                    let is_active = ctx.setp_lt_u32(tid, stride_reg);
                    ctx.branch_if_not(is_active, &format!("skip_reduce_{}", stride));
                    {
                        let other_tid = ctx.add_u32_reg(tid, stride_reg);
                        let other_off = ctx.mul_wide_u32(other_tid, 4);
                        let other_val_addr = ctx.add_u64(shared_base, other_off);
                        let other_val = ctx.ld_shared_f32(other_val_addr);
                        let other_idx_addr = ctx.add_u64(idx_base, other_off);
                        let other_idx = ctx.ld_shared_u32(other_idx_addr);

                        let my_val = ctx.ld_shared_f32(sh_val_addr);
                        let my_idx = ctx.ld_shared_u32(sh_idx_addr);

                        let is_greater = ctx.setp_gt_f32(other_val, my_val);
                        let new_val = ctx.selp_f32(is_greater, other_val, my_val);
                        let new_idx = ctx.selp_u32(is_greater, other_idx, my_idx);

                        ctx.st_shared_f32(sh_val_addr, new_val);
                        ctx.st_shared_u32(sh_idx_addr, new_idx);
                    }
                    ctx.label(&format!("skip_reduce_{}", stride));
                    ctx.bar_sync(0);
                }

                // Thread 0 writes block result to global memory
                let zero = ctx.const_u32(0);
                let is_thread_0 = ctx.setp_eq_u32(tid, zero);
                ctx.branch_if_not(is_thread_0, "exit");

                // Load final result from shared memory
                let zero_off = ctx.const_u32(0);
                let final_val_addr = ctx.add_u64(shared_base, zero_off);
                let final_val = ctx.ld_shared_f32(final_val_addr);
                let final_idx_addr = ctx.add_u64(idx_base, zero_off);
                let final_idx = ctx.ld_shared_u32(final_idx_addr);

                // Write to block output arrays
                let bid_offset = ctx.mul_wide_u32(bid, 4);
                let out_val_addr = ctx.add_u64(block_max_vals, bid_offset);
                ctx.st_global_f32(out_val_addr, final_val);
                let out_idx_addr = ctx.add_u64(block_max_idxs, bid_offset);
                ctx.st_global_u32(out_idx_addr, final_idx);

                ctx.label("exit");
            })
    }
}

/// Final reduction kernel to find global argmax from block results
///
/// This is a simple single-block kernel that reduces the per-block
/// max values to find the global maximum.
#[derive(Debug, Clone)]
pub struct ArgMaxFinalKernel {
    /// Number of blocks from first pass
    pub num_blocks: u32,
}

impl ArgMaxFinalKernel {
    /// Create kernel for final reduction
    #[must_use]
    pub fn new(num_blocks: u32) -> Self {
        Self { num_blocks }
    }
}

impl Kernel for ArgMaxFinalKernel {
    fn name(&self) -> &str {
        "argmax_final_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("argmax_final_reduce")
            .param(PtxType::U64, "block_max_vals")  // f32* from first pass
            .param(PtxType::U64, "block_max_idxs")  // u32* from first pass
            .param(PtxType::U64, "output_idx")      // u32* single output index
            .param(PtxType::U32, "num_blocks")      // Number of block results
            .shared_memory(256 * 8)
            .build(|ctx| {
                let tid = ctx.special_reg(crate::ptx::PtxReg::TidX);

                let block_max_vals = ctx.load_param_u64("block_max_vals");
                let block_max_idxs = ctx.load_param_u64("block_max_idxs");
                let output_idx = ctx.load_param_u64("output_idx");
                let num_blocks = ctx.load_param_u32("num_blocks");

                let shared_base = ctx.shared_ptr();
                let final_offset_1024 = ctx.mov_u64_imm(1024);
                let idx_base = ctx.add_u64(shared_base, final_offset_1024);

                // Initialize
                let neg_inf = ctx.const_f32(f32::NEG_INFINITY);
                let mut local_max = neg_inf;
                let mut local_idx = ctx.const_u32(0);

                // Stride loop over block results
                let mut idx = tid;
                ctx.label("load_loop");
                let in_bounds = ctx.setp_lt_u32(idx, num_blocks);
                ctx.branch_if_not(in_bounds, "load_done");

                let byte_off = ctx.mul_wide_u32(idx, 4);
                let val_addr = ctx.add_u64(block_max_vals, byte_off);
                let val = ctx.ld_global_f32(val_addr);
                let idx_addr = ctx.add_u64(block_max_idxs, byte_off);
                let block_idx = ctx.ld_global_u32(idx_addr);

                let is_greater = ctx.setp_gt_f32(val, local_max);
                local_max = ctx.selp_f32(is_greater, val, local_max);
                local_idx = ctx.selp_u32(is_greater, block_idx, local_idx);

                idx = ctx.add_u32(idx, 256);
                ctx.branch("load_loop");

                ctx.label("load_done");

                // Store to shared
                let sh_off = ctx.mul_wide_u32(tid, 4);
                let sh_val_addr = ctx.add_u64(shared_base, sh_off);
                ctx.st_shared_f32(sh_val_addr, local_max);
                let sh_idx_addr = ctx.add_u64(idx_base, sh_off);
                ctx.st_shared_u32(sh_idx_addr, local_idx);

                ctx.bar_sync(0);

                // Tree reduction
                for stride in [128u32, 64, 32, 16, 8, 4, 2, 1] {
                    let stride_reg = ctx.const_u32(stride);
                    let is_active = ctx.setp_lt_u32(tid, stride_reg);
                    ctx.branch_if_not(is_active, &format!("final_skip_{}", stride));
                    {
                        let other_tid = ctx.add_u32_reg(tid, stride_reg);
                        let other_off = ctx.mul_wide_u32(other_tid, 4);
                        let other_val_addr = ctx.add_u64(shared_base, other_off);
                        let other_val = ctx.ld_shared_f32(other_val_addr);
                        let other_idx_addr = ctx.add_u64(idx_base, other_off);
                        let other_idx = ctx.ld_shared_u32(other_idx_addr);

                        let my_val = ctx.ld_shared_f32(sh_val_addr);
                        let my_idx = ctx.ld_shared_u32(sh_idx_addr);

                        let is_greater = ctx.setp_gt_f32(other_val, my_val);
                        let new_val = ctx.selp_f32(is_greater, other_val, my_val);
                        let new_idx = ctx.selp_u32(is_greater, other_idx, my_idx);

                        ctx.st_shared_f32(sh_val_addr, new_val);
                        ctx.st_shared_u32(sh_idx_addr, new_idx);
                    }
                    ctx.label(&format!("final_skip_{}", stride));
                    ctx.bar_sync(0);
                }

                // Thread 0 writes final result
                let final_zero = ctx.const_u32(0);
                let is_zero = ctx.setp_eq_u32(tid, final_zero);
                ctx.branch_if_not(is_zero, "final_exit");

                let zero_off_64 = ctx.mov_u64_imm(0);
                let result_addr = ctx.add_u64(idx_base, zero_off_64);
                let result = ctx.ld_shared_u32(result_addr);
                ctx.st_global_u32(output_idx, result);

                ctx.label("final_exit");
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax_kernel_builds() {
        let kernel = ArgMaxKernel::new(152064);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".visible .entry argmax_block_reduce"));
        assert!(ptx.contains("bar.sync"));
    }

    #[test]
    fn test_argmax_final_kernel_builds() {
        let kernel = ArgMaxFinalKernel::new(149);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".visible .entry argmax_final_reduce"));
    }

    #[test]
    fn test_argmax_num_blocks() {
        let kernel = ArgMaxKernel::new(152064);
        // 152064 / (256 * 4) = 148.5 -> 149 blocks
        assert_eq!(kernel.num_blocks(), 149);
    }
}
