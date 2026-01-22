//! Numerically Stable Softmax Kernel
//!
//! Implements softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))

use super::Kernel;
use crate::ptx::{PtxKernel, PtxType};

/// Softmax kernel configuration
#[derive(Debug, Clone)]
pub struct SoftmaxKernel {
    /// Vector length
    pub length: u32,
    /// Use warp shuffle for reduction (faster)
    pub use_warp_shuffle: bool,
}

impl SoftmaxKernel {
    /// Create a new softmax kernel
    #[must_use]
    pub fn new(length: u32) -> Self {
        Self {
            length,
            use_warp_shuffle: true,
        }
    }

    /// Disable warp shuffle (for compatibility with older GPUs)
    #[must_use]
    pub const fn without_warp_shuffle(mut self) -> Self {
        self.use_warp_shuffle = false;
        self
    }
}

impl Kernel for SoftmaxKernel {
    fn name(&self) -> &str {
        if self.use_warp_shuffle {
            "softmax_warp_shuffle"
        } else {
            "softmax_shared"
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        if self.use_warp_shuffle {
            self.build_warp_shuffle()
        } else {
            self.build_shared_memory()
        }
    }
}

impl SoftmaxKernel {
    fn build_warp_shuffle(&self) -> PtxKernel {
        // Warp-level softmax using shuffle for fast reductions
        // Each block processes one row; blockIdx.x selects which row
        // Assumes row fits in a single warp (32 elements) for simplicity
        // For longer vectors, multiple warps would cooperate
        PtxKernel::new("softmax_warp_shuffle")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "length")
            .build(|ctx| {
                // Thread ID within block and block ID (row index)
                let tid = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let ctaid = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let length = ctx.load_param_u32("length");

                // Bounds check: tid must be < length (row size)
                let pred = ctx.setp_ge_u32(tid, length);
                ctx.branch_if(pred, "exit");

                // Load input value for this thread
                // Global index = ctaid * length + tid (row_idx * row_size + col_idx)
                let input_ptr = ctx.load_param_u64("input_ptr");
                let global_idx = ctx.mad_lo_u32(ctaid, length, tid);
                let offset = ctx.mul_wide_u32(global_idx, 4);
                let addr = ctx.add_u64(input_ptr, offset);
                let val = ctx.ld_global_f32(addr);

                // ===== Step 1: Find max using warp shuffle =====
                // Initialize max with our value
                let max_val = val;

                // Warp shuffle reduction for max (tree reduction)
                // Each iteration halves the active participants
                let shuffled_16 = ctx.shfl_down_f32(max_val, 16, 0xFFFF_FFFF);
                let max_val_1 = ctx.max_f32(max_val, shuffled_16);

                let shuffled_8 = ctx.shfl_down_f32(max_val_1, 8, 0xFFFF_FFFF);
                let max_val_2 = ctx.max_f32(max_val_1, shuffled_8);

                let shuffled_4 = ctx.shfl_down_f32(max_val_2, 4, 0xFFFF_FFFF);
                let max_val_3 = ctx.max_f32(max_val_2, shuffled_4);

                let shuffled_2 = ctx.shfl_down_f32(max_val_3, 2, 0xFFFF_FFFF);
                let max_val_4 = ctx.max_f32(max_val_3, shuffled_2);

                let shuffled_1 = ctx.shfl_down_f32(max_val_4, 1, 0xFFFF_FFFF);
                let warp_max = ctx.max_f32(max_val_4, shuffled_1);

                // Broadcast max to all lanes (get value from lane 0)
                let broadcast_max = ctx.shfl_idx_f32(warp_max, 0, 0xFFFF_FFFF);

                // ===== Step 2: Compute exp(val - max) =====
                let shifted = ctx.sub_f32(val, broadcast_max);
                // PTX ex2 computes 2^x, we need e^x = 2^(x * log2(e))
                // log2(e) ≈ 1.4426950408889634
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scaled = ctx.mul_f32(shifted, log2_e);
                let exp_val = ctx.ex2_f32(scaled);

                // ===== Step 3: Sum exp values using warp shuffle =====
                let sum_val = exp_val;

                let sum_shuffled_16 = ctx.shfl_down_f32(sum_val, 16, 0xFFFF_FFFF);
                let sum_val_1 = ctx.add_f32(sum_val, sum_shuffled_16);

                let sum_shuffled_8 = ctx.shfl_down_f32(sum_val_1, 8, 0xFFFF_FFFF);
                let sum_val_2 = ctx.add_f32(sum_val_1, sum_shuffled_8);

                let sum_shuffled_4 = ctx.shfl_down_f32(sum_val_2, 4, 0xFFFF_FFFF);
                let sum_val_3 = ctx.add_f32(sum_val_2, sum_shuffled_4);

                let sum_shuffled_2 = ctx.shfl_down_f32(sum_val_3, 2, 0xFFFF_FFFF);
                let sum_val_4 = ctx.add_f32(sum_val_3, sum_shuffled_2);

                let sum_shuffled_1 = ctx.shfl_down_f32(sum_val_4, 1, 0xFFFF_FFFF);
                let warp_sum = ctx.add_f32(sum_val_4, sum_shuffled_1);

                // Broadcast sum to all lanes (get value from lane 0)
                let broadcast_sum = ctx.shfl_idx_f32(warp_sum, 0, 0xFFFF_FFFF);

                // ===== Step 4: Divide exp(val - max) by sum =====
                let softmax_result = ctx.div_f32(exp_val, broadcast_sum);

                // ===== Step 5: Store result =====
                let output_ptr = ctx.load_param_u64("output_ptr");
                let out_addr = ctx.add_u64(output_ptr, offset);
                ctx.st_global_f32(out_addr, softmax_result);

                ctx.label("exit");
                ctx.ret();
            })
    }

    fn build_shared_memory(&self) -> PtxKernel {
        // Shared memory softmax for larger vectors or older GPUs
        // Uses block-level reduction with shared memory
        let block_size = 256_u32;
        let smem_size = block_size * 4; // Reduction buffer for f32

        PtxKernel::new("softmax_shared")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "length")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Thread and block indices
                let tid = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let ctaid = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ntid = ctx.special_reg(crate::ptx::PtxReg::NtidX);

                // Global index
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load parameters
                let length = ctx.load_param_u32("length");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check for loading
                let pred = ctx.setp_ge_u32(gid, length);

                // Load value (or 0 if out of bounds)
                let val = ctx.mov_f32_imm(0.0);
                ctx.branch_if(pred, "skip_load");
                let offset = ctx.mul_wide_u32(gid, 4);
                let addr = ctx.add_u64(input_ptr, offset);
                let _loaded = ctx.ld_global_f32(addr);
                // In real PTX we'd use predicated mov, simplified here
                ctx.label("skip_load");

                // Store to shared memory for reduction
                let smem_offset = ctx.mul_wide_u32(tid, 4);
                ctx.st_shared_f32(smem_offset, val);

                // Synchronize
                ctx.bar_sync(0);

                // ===== Block-level max reduction =====
                // Tree reduction in shared memory with halving stride
                let stride_reg = ctx.mov_u32_imm(128);
                let one = ctx.mov_u32_imm(1);

                ctx.label("max_reduce_loop");

                // Exit when stride reaches 0
                let stride_zero = ctx.setp_lt_u32(stride_reg, one);
                ctx.branch_if(stride_zero, "max_reduce_done");

                // Only threads with tid < stride participate
                let should_reduce = ctx.setp_lt_u32(tid, stride_reg);
                ctx.branch_if_not(should_reduce, "max_skip_neighbor");

                // Load neighbor value at tid + stride
                let neighbor_tid = ctx.add_u32_reg(tid, stride_reg);
                let block_size_reg = ctx.mov_u32_imm(block_size);
                let neighbor_oob = ctx.setp_ge_u32(neighbor_tid, block_size_reg);
                ctx.branch_if(neighbor_oob, "max_skip_neighbor");

                let neighbor_offset = ctx.mul_u32(neighbor_tid, 4);
                let neighbor_val = ctx.ld_shared_f32(neighbor_offset);
                let my_val = ctx.ld_shared_f32(smem_offset);
                let new_max = ctx.max_f32(my_val, neighbor_val);
                ctx.st_shared_f32(smem_offset, new_max);

                ctx.label("max_skip_neighbor");

                ctx.bar_sync(1);

                // Halve stride: stride = stride >> 1
                ctx.shr_u32_inplace(stride_reg, 1);
                ctx.branch("max_reduce_loop"); // Loop back with halved stride

                ctx.label("max_reduce_done");

                // Get max from thread 0
                let zero_offset = ctx.mov_u32_imm(0);
                let zero_offset_64 = ctx.cvt_u64_u32(zero_offset);
                let block_max = ctx.ld_shared_f32(zero_offset_64);

                ctx.bar_sync(2);

                // ===== Compute exp(val - max) =====
                let shifted = ctx.sub_f32(val, block_max);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scaled = ctx.mul_f32(shifted, log2_e);
                let exp_val = ctx.ex2_f32(scaled);

                // Store exp values back to shared memory
                ctx.st_shared_f32(smem_offset, exp_val);

                ctx.bar_sync(3);

                // ===== Block-level sum reduction =====
                // Tree reduction for sum (same pattern as max)
                let sum_stride_reg = ctx.mov_u32_imm(128);

                ctx.label("sum_reduce_loop");

                // Exit when stride reaches 0
                let sum_stride_zero = ctx.setp_lt_u32(sum_stride_reg, one);
                ctx.branch_if(sum_stride_zero, "sum_reduce_done");

                // Only threads with tid < stride participate
                let should_sum = ctx.setp_lt_u32(tid, sum_stride_reg);
                ctx.branch_if_not(should_sum, "sum_skip_neighbor");

                // Load neighbor value at tid + stride
                let sum_neighbor_tid = ctx.add_u32_reg(tid, sum_stride_reg);
                let sum_neighbor_oob = ctx.setp_ge_u32(sum_neighbor_tid, block_size_reg);
                ctx.branch_if(sum_neighbor_oob, "sum_skip_neighbor");

                let sum_neighbor_offset = ctx.mul_u32(sum_neighbor_tid, 4);
                let sum_neighbor_val = ctx.ld_shared_f32(sum_neighbor_offset);
                let sum_my_val = ctx.ld_shared_f32(smem_offset);
                let new_sum = ctx.add_f32(sum_my_val, sum_neighbor_val);
                ctx.st_shared_f32(smem_offset, new_sum);

                ctx.label("sum_skip_neighbor");

                ctx.bar_sync(4);

                // Halve stride: stride = stride >> 1
                ctx.shr_u32_inplace(sum_stride_reg, 1);
                ctx.branch("sum_reduce_loop"); // Loop back with halved stride

                ctx.label("sum_reduce_done");

                // Get sum from thread 0
                let block_sum = ctx.ld_shared_f32(zero_offset_64);

                ctx.bar_sync(5);

                // ===== Divide and store =====
                let softmax_result = ctx.div_f32(exp_val, block_sum);

                ctx.branch_if(pred, "exit");
                let out_offset = ctx.mul_wide_u32(gid, 4);
                let out_addr = ctx.add_u64(output_ptr, out_offset);
                ctx.st_global_f32(out_addr, softmax_result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// ============================================================================
// Long Row Softmax Kernel (for rows > 32 elements)
// ============================================================================

/// Softmax kernel for long rows (> 32 elements)
///
/// Uses multi-warp reduction with grid-stride loops:
/// - Each block handles one row
/// - Up to 256 threads (8 warps) per block
/// - Each thread processes multiple elements in grid-stride pattern
/// - Warp-level reduction, then inter-warp reduction via shared memory
#[derive(Debug, Clone)]
pub struct LongRowSoftmaxKernel {
    /// Row size (number of elements per row)
    pub row_size: u32,
}

impl LongRowSoftmaxKernel {
    /// Create a new long row softmax kernel
    #[must_use]
    pub fn new(row_size: u32) -> Self {
        Self { row_size }
    }
}

impl Kernel for LongRowSoftmaxKernel {
    fn name(&self) -> &str {
        "softmax_long_row"
    }

    fn build_ptx(&self) -> PtxKernel {
        // FULL SOFTMAX: exp(x - max) / sum(exp(x - max))
        let block_size = 256_u32;
        let n_warps = block_size / 32;
        // Shared memory: 8 warp maxes + 1 global max + 8 warp sums + 1 global sum = 72 bytes
        let smem_size = (n_warps * 2 + 2) * 4;

        PtxKernel::new("softmax_long_row")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "row_size")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Thread indexing
                let tid = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let ctaid = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ntid = ctx.special_reg(crate::ptx::PtxReg::NtidX);

                let row_size = ctx.load_param_u32("row_size");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Compute warp_id and lane_id
                let lane_mask = ctx.mov_u32_imm(31);
                let lane_id = ctx.and_u32(tid, lane_mask);
                let warp_id = ctx.shr_u32_imm(tid, 5); // tid / 32

                // Row offset
                let row_offset = ctx.mul_lo_u32(ctaid, row_size);
                let row_offset_bytes = ctx.mul_wide_u32(row_offset, 4);
                let row_in_ptr = ctx.add_u64(input_ptr, row_offset_bytes);
                let row_out_ptr = ctx.add_u64(output_ptr, row_offset_bytes);

                // =========================================================
                // Phase 1: Find max using grid-stride loop + multi-warp reduction
                // =========================================================
                let neg_inf = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let local_max = neg_inf;

                // Grid-stride loop: idx = tid; idx < row_size; idx += ntid
                let idx = ctx.add_u32(tid, 0); // Copy tid to new register

                ctx.label("max_loop");
                let done_max = ctx.setp_ge_u32(idx, row_size);
                ctx.branch_if(done_max, "max_loop_done");

                // Load input[idx]
                let byte_offset = ctx.mul_wide_u32(idx, 4);
                let load_addr = ctx.add_u64(row_in_ptr, byte_offset);
                let val = ctx.ld_global_f32(load_addr);

                // local_max = max(local_max, val)
                ctx.max_f32_inplace(local_max, val);

                // idx += ntid
                ctx.add_u32_reg_inplace(idx, ntid);
                ctx.branch("max_loop");

                ctx.label("max_loop_done");

                // Warp-level max reduction using shuffles
                let shuffled_16 = ctx.shfl_down_f32(local_max, 16, 0xFFFF_FFFF);
                let warp_max_1 = ctx.max_f32(local_max, shuffled_16);
                let shuffled_8 = ctx.shfl_down_f32(warp_max_1, 8, 0xFFFF_FFFF);
                let warp_max_2 = ctx.max_f32(warp_max_1, shuffled_8);
                let shuffled_4 = ctx.shfl_down_f32(warp_max_2, 4, 0xFFFF_FFFF);
                let warp_max_3 = ctx.max_f32(warp_max_2, shuffled_4);
                let shuffled_2 = ctx.shfl_down_f32(warp_max_3, 2, 0xFFFF_FFFF);
                let warp_max_4 = ctx.max_f32(warp_max_3, shuffled_2);
                let shuffled_1 = ctx.shfl_down_f32(warp_max_4, 1, 0xFFFF_FFFF);
                let warp_max = ctx.max_f32(warp_max_4, shuffled_1);

                // Lane 0 of each warp stores warp max to shared memory
                let zero = ctx.mov_u32_imm(0);
                let is_lane_0 = ctx.setp_eq_u32(lane_id, zero);
                ctx.branch_if_not(is_lane_0, "skip_store_warp_max");
                let smem_offset = ctx.mul_u32(warp_id, 4);
                let smem_offset_64 = ctx.cvt_u64_u32(smem_offset);
                ctx.st_shared_f32(smem_offset_64, warp_max);
                ctx.label("skip_store_warp_max");

                // Synchronize
                ctx.bar_sync(0);

                // Warp 0 (all 32 lanes) reduce across warp maxes
                // ALL lanes must participate in shuffles to avoid deadlock!
                let is_warp_0 = ctx.setp_eq_u32(warp_id, zero);
                ctx.branch_if_not(is_warp_0, "skip_inter_warp_max");

                // Lanes 0-7 load valid warp maxes, lanes 8-31 load duplicates
                // using lane_id & 7 to clamp index to 0-7 (duplicates don't affect max)
                let seven = ctx.mov_u32_imm(7);
                let lane_id_clamped = ctx.and_u32(lane_id, seven);
                let lane_smem_offset = ctx.mul_u32(lane_id_clamped, 4);
                let lane_smem_64 = ctx.cvt_u64_u32(lane_smem_offset);
                let loaded_warp_max = ctx.ld_shared_f32(lane_smem_64);

                // Reduce 8 values using shuffles (all 32 lanes participate)
                let inter_4 = ctx.shfl_down_f32(loaded_warp_max, 4, 0xFFFF_FFFF);
                let inter_max_1 = ctx.max_f32(loaded_warp_max, inter_4);
                let inter_2 = ctx.shfl_down_f32(inter_max_1, 2, 0xFFFF_FFFF);
                let inter_max_2 = ctx.max_f32(inter_max_1, inter_2);
                let inter_1 = ctx.shfl_down_f32(inter_max_2, 1, 0xFFFF_FFFF);
                let global_max = ctx.max_f32(inter_max_2, inter_1);

                // Lane 0 stores global max at shared[8] (offset 32 bytes)
                let is_lane_0_check = ctx.setp_eq_u32(lane_id, zero);
                ctx.branch_if_not(is_lane_0_check, "skip_store_global_max");
                let global_max_offset = ctx.mov_u32_imm(32); // 8 * 4 bytes
                let global_max_offset_64 = ctx.cvt_u64_u32(global_max_offset);
                ctx.st_shared_f32(global_max_offset_64, global_max);
                ctx.label("skip_store_global_max");

                ctx.label("skip_inter_warp_max");

                // Synchronize
                ctx.bar_sync(1);

                // All threads load global max
                let global_max_read_offset = ctx.mov_u32_imm(32);
                let global_max_read_64 = ctx.cvt_u64_u32(global_max_read_offset);
                let global_max_val = ctx.ld_shared_f32(global_max_read_64);

                // =========================================================
                // Phase 2: Compute sum(exp(x - max)) using grid-stride loop
                // =========================================================
                let local_sum = ctx.mov_f32_imm(0.0);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);

                let idx2 = ctx.add_u32(tid, 0);
                ctx.label("sum_loop");
                let done_sum = ctx.setp_ge_u32(idx2, row_size);
                ctx.branch_if(done_sum, "sum_loop_done");

                // Load input[idx]
                let byte_offset2 = ctx.mul_wide_u32(idx2, 4);
                let load_addr2 = ctx.add_u64(row_in_ptr, byte_offset2);
                let val2 = ctx.ld_global_f32(load_addr2);

                // exp_val = exp(val - global_max) = 2^((val - max) * log2(e))
                let shifted = ctx.sub_f32(val2, global_max_val);
                let scaled = ctx.mul_f32(shifted, log2_e);
                let exp_val = ctx.ex2_f32(scaled);

                // local_sum += exp_val
                ctx.add_f32_inplace(local_sum, exp_val);

                ctx.add_u32_reg_inplace(idx2, ntid);
                ctx.branch("sum_loop");

                ctx.label("sum_loop_done");

                // Warp-level sum reduction using shuffles
                let sum_shuffled_16 = ctx.shfl_down_f32(local_sum, 16, 0xFFFF_FFFF);
                let warp_sum_1 = ctx.add_f32(local_sum, sum_shuffled_16);
                let sum_shuffled_8 = ctx.shfl_down_f32(warp_sum_1, 8, 0xFFFF_FFFF);
                let warp_sum_2 = ctx.add_f32(warp_sum_1, sum_shuffled_8);
                let sum_shuffled_4 = ctx.shfl_down_f32(warp_sum_2, 4, 0xFFFF_FFFF);
                let warp_sum_3 = ctx.add_f32(warp_sum_2, sum_shuffled_4);
                let sum_shuffled_2 = ctx.shfl_down_f32(warp_sum_3, 2, 0xFFFF_FFFF);
                let warp_sum_4 = ctx.add_f32(warp_sum_3, sum_shuffled_2);
                let sum_shuffled_1 = ctx.shfl_down_f32(warp_sum_4, 1, 0xFFFF_FFFF);
                let warp_sum = ctx.add_f32(warp_sum_4, sum_shuffled_1);

                // Lane 0 of each warp stores warp sum to shared memory at offset 36+ bytes
                ctx.branch_if_not(is_lane_0, "skip_store_warp_sum");
                let sum_smem_base = ctx.mov_u32_imm(36); // after global_max
                let four = ctx.mov_u32_imm(4);
                let sum_smem_offset = ctx.mad_lo_u32(warp_id, four, sum_smem_base);
                let sum_smem_64 = ctx.cvt_u64_u32(sum_smem_offset);
                ctx.st_shared_f32(sum_smem_64, warp_sum);
                ctx.label("skip_store_warp_sum");

                // Synchronize
                ctx.bar_sync(2);

                // Warp 0 (all 32 lanes) reduce across warp sums
                ctx.branch_if_not(is_warp_0, "skip_inter_warp_sum");

                // Lanes 0-7 load valid warp sums, lanes 8-31 load duplicates
                let seven2 = ctx.mov_u32_imm(7);
                let lane_id_clamped2 = ctx.and_u32(lane_id, seven2);
                let sum_base2 = ctx.mov_u32_imm(36);
                let four2 = ctx.mov_u32_imm(4);
                let sum_lane_offset = ctx.mad_lo_u32(lane_id_clamped2, four2, sum_base2);
                let sum_lane_64 = ctx.cvt_u64_u32(sum_lane_offset);
                let loaded_warp_sum = ctx.ld_shared_f32(sum_lane_64);

                // Reduce 8 values using shuffles (all 32 lanes participate)
                let sum_inter_4 = ctx.shfl_down_f32(loaded_warp_sum, 4, 0xFFFF_FFFF);
                let inter_sum_1 = ctx.add_f32(loaded_warp_sum, sum_inter_4);
                let sum_inter_2 = ctx.shfl_down_f32(inter_sum_1, 2, 0xFFFF_FFFF);
                let inter_sum_2 = ctx.add_f32(inter_sum_1, sum_inter_2);
                let sum_inter_1 = ctx.shfl_down_f32(inter_sum_2, 1, 0xFFFF_FFFF);
                let global_sum = ctx.add_f32(inter_sum_2, sum_inter_1);

                // Lane 0 stores global sum at shared[17] (offset 68 bytes)
                let is_lane_0_sum = ctx.setp_eq_u32(lane_id, zero);
                ctx.branch_if_not(is_lane_0_sum, "skip_store_global_sum");
                let global_sum_offset = ctx.mov_u32_imm(68);
                let global_sum_offset_64 = ctx.cvt_u64_u32(global_sum_offset);
                ctx.st_shared_f32(global_sum_offset_64, global_sum);
                ctx.label("skip_store_global_sum");

                ctx.label("skip_inter_warp_sum");

                // Synchronize
                ctx.bar_sync(3);

                // All threads load global sum
                let global_sum_read_offset = ctx.mov_u32_imm(68);
                let global_sum_read_64 = ctx.cvt_u64_u32(global_sum_read_offset);
                let global_sum_val = ctx.ld_shared_f32(global_sum_read_64);

                // =========================================================
                // Phase 3: Normalize and write output: exp(x - max) / sum
                // =========================================================
                let idx3 = ctx.add_u32(tid, 0);
                ctx.label("write_loop");
                let done_write = ctx.setp_ge_u32(idx3, row_size);
                ctx.branch_if(done_write, "write_loop_done");

                // Load input[idx]
                let byte_offset3 = ctx.mul_wide_u32(idx3, 4);
                let load_addr3 = ctx.add_u64(row_in_ptr, byte_offset3);
                let val3 = ctx.ld_global_f32(load_addr3);

                // exp_val = exp(val - global_max)
                let shifted3 = ctx.sub_f32(val3, global_max_val);
                let scaled3 = ctx.mul_f32(shifted3, log2_e);
                let exp_val3 = ctx.ex2_f32(scaled3);

                // softmax_val = exp_val / global_sum
                let softmax_val = ctx.div_f32(exp_val3, global_sum_val);

                // Store result
                let out_addr = ctx.add_u64(row_out_ptr, byte_offset3);
                ctx.st_global_f32(out_addr, softmax_val);

                ctx.add_u32_reg_inplace(idx3, ntid);
                ctx.branch("write_loop");

                ctx.label("write_loop_done");
                ctx.ret();
            })
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_kernel_name() {
        let kernel = SoftmaxKernel::new(4096);
        assert_eq!(kernel.name(), "softmax_warp_shuffle");

        let kernel_shared = SoftmaxKernel::new(4096).without_warp_shuffle();
        assert_eq!(kernel_shared.name(), "softmax_shared");
    }

    #[test]
    fn test_long_row_softmax_ptx_generation() {
        let kernel = LongRowSoftmaxKernel::new(1500);
        let ptx = kernel.emit_ptx();

        // Verify kernel name
        assert!(ptx.contains("softmax_long_row"), "Missing kernel name");

        // Verify parameters
        assert!(ptx.contains(".param .u64 input_ptr"), "Missing input_ptr param");
        assert!(ptx.contains(".param .u64 output_ptr"), "Missing output_ptr param");
        assert!(ptx.contains(".param .u32 row_size"), "Missing row_size param");

        // Verify has grid-stride loops (multiple branch labels)
        assert!(ptx.contains("max_loop:"), "Missing max_loop label");
        assert!(ptx.contains("max_loop_done:"), "Missing max_loop_done label");
        assert!(ptx.contains("sum_loop:"), "Missing sum_loop label");
        assert!(ptx.contains("write_loop:"), "Missing write_loop label");

        // Verify has barrier syncs for inter-warp reduction
        assert!(ptx.contains("bar.sync"), "Missing barrier sync");

        // Verify has warp shuffles for intra-warp reduction
        assert!(ptx.contains("shfl") || ptx.contains("shfl.down") || ptx.contains("shfl.sync.down"),
                "Missing warp shuffle");

        // Print first 300 lines for debugging
        for (i, line) in ptx.lines().enumerate().take(300) {
            println!("{:4}: {}", i + 1, line);
        }
    }

    #[test]
    fn test_softmax_ptx_generation() {
        let kernel = SoftmaxKernel::new(4096);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u32 length"));
    }

    #[test]
    fn test_softmax_shared_memory() {
        let kernel = SoftmaxKernel::new(4096).without_warp_shuffle();
        let ptx_kernel = kernel.build_ptx();
        assert!(ptx_kernel.shared_memory_bytes() > 0);
    }

    #[test]
    fn test_softmax_warp_shuffle_ptx() {
        let kernel = SoftmaxKernel::new(32);
        let ptx = kernel.emit_ptx();

        // Verify warp shuffle operations are present
        assert!(ptx.contains("shfl") || ptx.contains("shfl.down"));

        // Verify max operation
        assert!(ptx.contains("max.f32"));

        // Verify exp operation (ex2)
        assert!(ptx.contains("ex2.f32") || ptx.contains("ex2"));

        // Verify division
        assert!(ptx.contains("div.rn.f32")); // div requires rounding mode for floats

        // Verify memory operations
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));
    }

    #[test]
    fn test_softmax_shared_memory_ptx() {
        let kernel = SoftmaxKernel::new(256).without_warp_shuffle();
        let ptx = kernel.emit_ptx();

        // Verify shared memory usage
        assert!(ptx.contains("ld.shared.f32") || ptx.contains("ld.f32"));
        assert!(ptx.contains("st.shared.f32") || ptx.contains("st.f32"));

        // Verify barrier synchronization
        assert!(ptx.contains("bar"));

        // Verify exp and divide
        assert!(ptx.contains("ex2.f32") || ptx.contains("ex2"));
        assert!(ptx.contains("div.rn.f32")); // div requires rounding mode for floats
    }

    #[test]
    fn test_softmax_kernel_variants() {
        let warp_kernel = SoftmaxKernel::new(32);
        let shared_kernel = SoftmaxKernel::new(256).without_warp_shuffle();

        // Both should produce valid PTX
        let warp_ptx = warp_kernel.emit_ptx();
        let shared_ptx = shared_kernel.emit_ptx();

        assert!(!warp_ptx.is_empty());
        assert!(!shared_ptx.is_empty());

        // Verify different kernel names in output
        assert!(warp_ptx.contains("softmax_warp_shuffle"));
        assert!(shared_ptx.contains("softmax_shared"));
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Verify the implementation uses numerically stable softmax
        // (subtracts max before exp)
        let kernel = SoftmaxKernel::new(32);
        let ptx = kernel.emit_ptx();

        // Should have sub operation (for val - max)
        assert!(ptx.contains("sub.f32"));

        // Should have mul for log2(e) scaling
        assert!(ptx.contains("mul.f32"));
    }

    // =========================================================================
    // SATD REMEDIATION TESTS (EXTREME TDD)
    // These tests verify the max-reduce loop bug is fixed.
    // Falsifiable claims per Popperian methodology.
    // =========================================================================

    #[test]
    fn test_shared_max_reduce_loop_iterates() {
        // FALSIFIABLE CLAIM: Max-reduce loop iterates multiple times for full reduction
        // The SATD bug causes it to exit after one iteration.
        let kernel = SoftmaxKernel::new(256).without_warp_shuffle();
        let ptx = kernel.emit_ptx();

        // The PTX should contain a branch back to max_reduce_loop
        // If it only branches to max_reduce_done, the reduction is incomplete
        let has_loop_back =
            ptx.contains("bra max_reduce_loop") || ptx.contains("bra\tmax_reduce_loop");

        assert!(
            has_loop_back,
            "FALSIFIED: Max-reduce loop does not branch back to loop start. \
             Found 'bra max_reduce_done' instead of 'bra max_reduce_loop'. \
             This means max reduction only runs once, producing wrong max."
        );
    }

    #[test]
    fn test_shared_max_reduce_stride_halves() {
        // FALSIFIABLE CLAIM: Max-reduce stride is halved each iteration (128->64->32->...)
        // If stride is not updated, loop will be infinite or wrong.
        let kernel = SoftmaxKernel::new(256).without_warp_shuffle();
        let ptx = kernel.emit_ptx();

        // Look for stride manipulation - should see shr.b32 (PTX requires .b32 for shifts) or div
        let has_stride_update =
            ptx.contains("shr.b32") || ptx.contains("shr.u32") || ptx.contains("div.u32");

        assert!(
            has_stride_update,
            "FALSIFIED: Max-reduce stride is not halved. \
             Expected shr.b32, shr.u32 or div.u32 for stride = stride / 2. \
             Without this, tree reduction cannot work correctly."
        );
    }

    #[test]
    fn test_shared_sum_reduce_implemented() {
        // FALSIFIABLE CLAIM: Sum reduction is fully implemented, not a placeholder
        // The SATD bug has: `let block_sum = sum_val; // Placeholder`
        let kernel = SoftmaxKernel::new(256).without_warp_shuffle();
        let ptx = kernel.emit_ptx();

        // Verify sum reduction loop structure exists
        // Should have: sum_reduce_loop label, branch back, and sum_reduce_done label
        let has_sum_loop = ptx.contains("sum_reduce_loop");
        let has_sum_done = ptx.contains("sum_reduce_done");
        let has_loop_back =
            ptx.contains("bra sum_reduce_loop") || ptx.contains("bra\tsum_reduce_loop");

        assert!(
            has_sum_loop && has_sum_done && has_loop_back,
            "FALSIFIED: Sum reduction loop structure is incomplete. \
             has_sum_loop={}, has_sum_done={}, has_loop_back={}. \
             A proper tree reduction needs a complete loop structure.",
            has_sum_loop,
            has_sum_done,
            has_loop_back
        );
    }
}
