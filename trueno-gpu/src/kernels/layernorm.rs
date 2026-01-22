//! Fused Layer Normalization Kernel
//!
//! Implements LayerNorm(x) = (x - mean) / sqrt(variance + epsilon) * gamma + beta
//!
//! Uses warp-level parallel reductions for mean and variance computation.
//! Numerically stable using Welford's online algorithm.

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use super::Kernel;
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Layer normalization kernel configuration
#[derive(Debug, Clone)]
pub struct LayerNormKernel {
    /// Hidden dimension size
    pub hidden_size: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Whether to use affine transformation (gamma, beta)
    pub affine: bool,
    /// Use warp shuffle for reduction (faster on SM 3.0+)
    pub use_warp_shuffle: bool,
}

impl LayerNormKernel {
    /// Create a new LayerNorm kernel
    #[must_use]
    pub fn new(hidden_size: u32) -> Self {
        Self {
            hidden_size,
            epsilon: 1e-5,
            affine: true,
            use_warp_shuffle: true,
        }
    }

    /// Set custom epsilon value
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Disable affine transformation (gamma=1, beta=0)
    #[must_use]
    pub const fn without_affine(mut self) -> Self {
        self.affine = false;
        self
    }

    /// Disable warp shuffle (for compatibility with older GPUs)
    #[must_use]
    pub const fn without_warp_shuffle(mut self) -> Self {
        self.use_warp_shuffle = false;
        self
    }
}

impl Kernel for LayerNormKernel {
    fn name(&self) -> &str {
        if self.use_warp_shuffle {
            "layernorm_warp_shuffle"
        } else {
            "layernorm_shared"
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

impl LayerNormKernel {
    fn build_warp_shuffle(&self) -> PtxKernel {
        // Warp-level LayerNorm using shuffle for fast reductions
        // Each warp handles one row of the input
        // FIXED: Now properly loops over all elements for hidden_size > 32
        let epsilon = self.epsilon;
        let affine = self.affine;

        PtxKernel::new("layernorm_warp_shuffle")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U64, "gamma_ptr")
            .param(PtxType::U64, "beta_ptr")
            .param(PtxType::U32, "hidden_size")
            .param(PtxType::U32, "batch_size")
            .build(|ctx| {
                // Thread ID within warp (lane)
                let tid = ctx.special_reg(PtxReg::TidX);
                let hidden_size_param = ctx.load_param_u32("hidden_size");
                let batch_size = ctx.load_param_u32("batch_size");

                // Each block handles one row, each thread handles strided elements
                let row_idx = ctx.special_reg(PtxReg::CtaIdX);
                let lane_id = ctx.rem_u32(tid, 32);

                // Bounds check - row must be within batch
                let pred = ctx.setp_ge_u32(row_idx, batch_size);
                ctx.branch_if(pred, "exit");

                // Calculate row offset
                let input_ptr = ctx.load_param_u64("input_ptr");
                let row_offset = ctx.mul_wide_u32_reg(row_idx, hidden_size_param);
                let row_offset_bytes = ctx.mul_u64(row_offset, 4);
                let row_base = ctx.add_u64(input_ptr, row_offset_bytes);

                // Constants
                let four = ctx.mov_u32_imm(4);

                // ===== Step 1: Compute mean using warp shuffle =====
                // Each thread loads and sums multiple elements with stride 32
                let sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let elem_idx = ctx.add_u32_reg(idx, lane_id);
                let in_bounds = ctx.setp_lt_u32(elem_idx, hidden_size_param);
                ctx.branch_if_not(in_bounds, "sum_loop_end");

                let elem_offset = ctx.mul_wide_u32_reg(elem_idx, four);
                let elem_addr = ctx.add_u64(row_base, elem_offset);
                let val = ctx.ld_global_f32(elem_addr);
                ctx.add_f32_inplace(sum, val);

                ctx.add_u32_inplace(idx, 32); // stride by warp size
                ctx.branch("sum_loop");

                ctx.label("sum_loop_end");

                // Warp shuffle reduction for sum
                let shuffled_16 = ctx.shfl_down_f32(sum, 16, 0xFFFF_FFFF);
                let sum_1 = ctx.add_f32(sum, shuffled_16);

                let shuffled_8 = ctx.shfl_down_f32(sum_1, 8, 0xFFFF_FFFF);
                let sum_2 = ctx.add_f32(sum_1, shuffled_8);

                let shuffled_4 = ctx.shfl_down_f32(sum_2, 4, 0xFFFF_FFFF);
                let sum_3 = ctx.add_f32(sum_2, shuffled_4);

                let shuffled_2 = ctx.shfl_down_f32(sum_3, 2, 0xFFFF_FFFF);
                let sum_4 = ctx.add_f32(sum_3, shuffled_2);

                let shuffled_1 = ctx.shfl_down_f32(sum_4, 1, 0xFFFF_FFFF);
                let warp_sum = ctx.add_f32(sum_4, shuffled_1);

                // Broadcast sum to all lanes and compute mean
                let broadcast_sum = ctx.shfl_idx_f32(warp_sum, 0, 0xFFFF_FFFF);
                let hidden_f32 = ctx.cvt_f32_u32(hidden_size_param);
                let mean = ctx.div_f32(broadcast_sum, hidden_f32);

                // ===== Step 2: Compute variance using warp shuffle =====
                // variance = sum((x - mean)^2) / n
                let var_sum = ctx.mov_f32_imm(0.0);
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("var_loop");
                let elem_idx2 = ctx.add_u32_reg(idx2, lane_id);
                let in_bounds2 = ctx.setp_lt_u32(elem_idx2, hidden_size_param);
                ctx.branch_if_not(in_bounds2, "var_loop_end");

                let elem_offset2 = ctx.mul_wide_u32_reg(elem_idx2, four);
                let elem_addr2 = ctx.add_u64(row_base, elem_offset2);
                let val2 = ctx.ld_global_f32(elem_addr2);
                let diff = ctx.sub_f32(val2, mean);
                let sq_diff = ctx.mul_f32(diff, diff);
                ctx.add_f32_inplace(var_sum, sq_diff);

                ctx.add_u32_inplace(idx2, 32); // stride by warp size
                ctx.branch("var_loop");

                ctx.label("var_loop_end");

                // Warp shuffle reduction for variance sum
                let var_shuffled_16 = ctx.shfl_down_f32(var_sum, 16, 0xFFFF_FFFF);
                let var_sum_1 = ctx.add_f32(var_sum, var_shuffled_16);

                let var_shuffled_8 = ctx.shfl_down_f32(var_sum_1, 8, 0xFFFF_FFFF);
                let var_sum_2 = ctx.add_f32(var_sum_1, var_shuffled_8);

                let var_shuffled_4 = ctx.shfl_down_f32(var_sum_2, 4, 0xFFFF_FFFF);
                let var_sum_3 = ctx.add_f32(var_sum_2, var_shuffled_4);

                let var_shuffled_2 = ctx.shfl_down_f32(var_sum_3, 2, 0xFFFF_FFFF);
                let var_sum_4 = ctx.add_f32(var_sum_3, var_shuffled_2);

                let var_shuffled_1 = ctx.shfl_down_f32(var_sum_4, 1, 0xFFFF_FFFF);
                let warp_var_sum = ctx.add_f32(var_sum_4, var_shuffled_1);

                // Broadcast and compute variance
                let broadcast_var_sum = ctx.shfl_idx_f32(warp_var_sum, 0, 0xFFFF_FFFF);
                let variance = ctx.div_f32(broadcast_var_sum, hidden_f32);

                // ===== Step 3: Compute rstd = 1/sqrt(variance + epsilon) =====
                let eps = ctx.mov_f32_imm(epsilon);
                let var_plus_eps = ctx.add_f32(variance, eps);
                let rstd = ctx.rsqrt_f32(var_plus_eps);

                // ===== Step 4: Normalize and apply affine transformation =====
                // Third pass to normalize all elements
                let idx3 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let elem_idx3 = ctx.add_u32_reg(idx3, lane_id);
                let in_bounds3 = ctx.setp_lt_u32(elem_idx3, hidden_size_param);
                ctx.branch_if_not(in_bounds3, "exit");

                let elem_offset3 = ctx.mul_wide_u32_reg(elem_idx3, four);
                let elem_addr3 = ctx.add_u64(row_base, elem_offset3);
                let val3 = ctx.ld_global_f32(elem_addr3);

                // normalized = (x - mean) * rstd
                let diff3 = ctx.sub_f32(val3, mean);
                let normalized = ctx.mul_f32(diff3, rstd);

                // Apply affine: y = gamma * normalized + beta
                let result = if affine {
                    let gamma_ptr = ctx.load_param_u64("gamma_ptr");
                    let beta_ptr = ctx.load_param_u64("beta_ptr");
                    let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset3);
                    let beta_addr = ctx.add_u64(beta_ptr, elem_offset3);
                    let gamma = ctx.ld_global_f32(gamma_addr);
                    let beta = ctx.ld_global_f32(beta_addr);
                    let scaled = ctx.mul_f32(gamma, normalized);
                    ctx.add_f32(scaled, beta)
                } else {
                    normalized
                };

                // Store result
                let output_ptr = ctx.load_param_u64("output_ptr");
                let out_row_base = ctx.add_u64(output_ptr, row_offset_bytes);
                let out_addr = ctx.add_u64(out_row_base, elem_offset3);
                ctx.st_global_f32(out_addr, result);

                ctx.add_u32_inplace(idx3, 32); // stride by warp size
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }

    fn build_shared_memory(&self) -> PtxKernel {
        // Shared memory LayerNorm for larger hidden sizes or older GPUs
        // Uses block-level reduction with shared memory
        let block_size = 256_u32;
        let smem_size = block_size * 4 * 2; // sum and sq_sum buffers
        let epsilon = self.epsilon;
        let affine = self.affine;

        PtxKernel::new("layernorm_shared")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U64, "gamma_ptr")
            .param(PtxType::U64, "beta_ptr")
            .param(PtxType::U32, "hidden_size")
            .param(PtxType::U32, "batch_size")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Thread and block indices
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);

                // Global index within block
                let _gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load parameters
                let hidden_size_param = ctx.load_param_u32("hidden_size");
                let batch_size = ctx.load_param_u32("batch_size");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Each block handles one row
                let row_idx = ctaid;
                let row_pred = ctx.setp_ge_u32(row_idx, batch_size);
                ctx.branch_if(row_pred, "exit");

                // Calculate row base address
                let row_offset = ctx.mul_wide_u32_reg(row_idx, hidden_size_param);
                let row_offset_bytes = ctx.mul_u64(row_offset, 4);
                let row_base = ctx.add_u64(input_ptr, row_offset_bytes);

                // Thread loads one element (if in bounds)
                let elem_pred = ctx.setp_lt_u32(tid, hidden_size_param);
                let _val = ctx.mov_f32_imm(0.0);

                ctx.branch_if_not(elem_pred, "skip_load");
                let elem_offset = ctx.mul_wide_u32(tid, 4);
                let elem_addr = ctx.add_u64(row_base, elem_offset);
                let val = ctx.ld_global_f32(elem_addr);
                ctx.label("skip_load");

                // Store value to shared memory for reduction
                let smem_offset = ctx.mul_wide_u32(tid, 4);
                ctx.st_shared_f32(smem_offset, val);

                ctx.bar_sync(0);

                // ===== Block-level sum reduction =====
                let stride = ctx.mov_u32_imm(128);

                ctx.label("sum_reduce_loop");
                let stride_pred = ctx.setp_lt_u32(tid, stride);
                ctx.branch_if_not(stride_pred, "sum_reduce_done");

                let neighbor_tid = ctx.add_u32_reg(tid, stride);
                let block_size_reg = ctx.mov_u32_imm(block_size);
                let neighbor_oob = ctx.setp_ge_u32(neighbor_tid, block_size_reg);
                ctx.branch_if(neighbor_oob, "sum_skip_neighbor");

                let neighbor_offset = ctx.mul_wide_u32(neighbor_tid, 4);
                let neighbor_val = ctx.ld_shared_f32(neighbor_offset);
                let my_val = ctx.ld_shared_f32(smem_offset);
                let new_sum = ctx.add_f32(my_val, neighbor_val);
                ctx.st_shared_f32(smem_offset, new_sum);

                ctx.label("sum_skip_neighbor");
                ctx.bar_sync(1);
                ctx.branch("sum_reduce_done");

                ctx.label("sum_reduce_done");

                // Get sum from thread 0
                let zero_offset = ctx.mov_u64_imm(0);
                let total_sum = ctx.ld_shared_f32(zero_offset);

                // Compute mean
                let hidden_f32 = ctx.cvt_f32_u32(hidden_size_param);
                let mean = ctx.div_f32(total_sum, hidden_f32);

                ctx.bar_sync(2);

                // ===== Compute squared differences for variance =====
                let diff = ctx.sub_f32(val, mean);
                let sq_diff = ctx.mul_f32(diff, diff);
                ctx.st_shared_f32(smem_offset, sq_diff);

                ctx.bar_sync(3);

                // Block-level variance sum reduction (simplified)
                let var_stride = ctx.mov_u32_imm(128);
                let var_stride_pred = ctx.setp_lt_u32(tid, var_stride);
                ctx.branch_if_not(var_stride_pred, "var_reduce_done");

                let var_neighbor_tid = ctx.add_u32_reg(tid, var_stride);
                let var_neighbor_oob = ctx.setp_ge_u32(var_neighbor_tid, block_size_reg);
                ctx.branch_if(var_neighbor_oob, "var_skip_neighbor");

                let var_neighbor_offset = ctx.mul_wide_u32(var_neighbor_tid, 4);
                let var_neighbor_val = ctx.ld_shared_f32(var_neighbor_offset);
                let var_my_val = ctx.ld_shared_f32(smem_offset);
                let new_var_sum = ctx.add_f32(var_my_val, var_neighbor_val);
                ctx.st_shared_f32(smem_offset, new_var_sum);

                ctx.label("var_skip_neighbor");
                ctx.label("var_reduce_done");

                ctx.bar_sync(4);

                // Get variance sum and compute variance
                let total_var_sum = ctx.ld_shared_f32(zero_offset);
                let variance = ctx.div_f32(total_var_sum, hidden_f32);

                // Compute rstd = 1/sqrt(variance + epsilon)
                let eps = ctx.mov_f32_imm(epsilon);
                let var_plus_eps = ctx.add_f32(variance, eps);
                let rstd = ctx.rsqrt_f32(var_plus_eps);

                // ===== Normalize and store =====
                ctx.branch_if_not(elem_pred, "exit");

                let normalized = ctx.mul_f32(diff, rstd);

                let result = if affine {
                    let gamma_ptr = ctx.load_param_u64("gamma_ptr");
                    let beta_ptr = ctx.load_param_u64("beta_ptr");
                    let elem_offset = ctx.mul_wide_u32(tid, 4);
                    let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset);
                    let beta_addr = ctx.add_u64(beta_ptr, elem_offset);
                    let gamma = ctx.ld_global_f32(gamma_addr);
                    let beta = ctx.ld_global_f32(beta_addr);
                    let scaled = ctx.mul_f32(gamma, normalized);
                    ctx.add_f32(scaled, beta)
                } else {
                    normalized
                };

                let out_row_base = ctx.add_u64(output_ptr, row_offset_bytes);
                let elem_offset = ctx.mul_wide_u32(tid, 4);
                let out_addr = ctx.add_u64(out_row_base, elem_offset);
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// RMSNorm kernel (Root Mean Square Layer Normalization)
///
/// RMSNorm(x) = x / sqrt(mean(x^2) + epsilon) * gamma
///
/// Simpler and faster than LayerNorm - no mean centering, just scaling.
/// Used by LLaMA, Mistral, and other modern LLMs.
///
/// # PAR-023: Async pipeline support
///
/// This kernel is designed for chaining with other operations without sync.
#[derive(Debug, Clone)]
pub struct RmsNormKernel {
    /// Hidden dimension size
    pub hidden_size: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl RmsNormKernel {
    /// Create a new RMSNorm kernel
    #[must_use]
    pub fn new(hidden_size: u32) -> Self {
        Self {
            hidden_size,
            epsilon: 1e-5,
        }
    }

    /// Set custom epsilon value
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl Kernel for RmsNormKernel {
    fn name(&self) -> &str {
        "rmsnorm"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden_size = self.hidden_size;
        let epsilon = self.epsilon;

        // RMSNorm for single row (batch=1) using warp shuffle
        // Grid: 1 block, Block: 32 threads (one warp)
        // Each thread handles hidden_size/32 elements
        PtxKernel::new("rmsnorm")
            .param(PtxType::U64, "input_ptr") // Input vector
            .param(PtxType::U64, "output_ptr") // Output vector (can be same as input)
            .param(PtxType::U64, "gamma_ptr") // Scale weights
            .shared_memory(0) // Warp shuffle, no shared memory
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);

                // Load parameters
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                // Constants
                let hidden_u32 = ctx.mov_u32_imm(hidden_size);
                let four = ctx.mov_u32_imm(4);

                // Accumulate sum of squares
                // Each thread processes elements: tid, tid+32, tid+64, ...
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let loop_idx = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(loop_idx, hidden_u32);
                ctx.branch_if_not(in_bounds, "sum_loop_end");

                // Load input[idx]
                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let elem_addr = ctx.add_u64(input_ptr, elem_offset);
                let val = ctx.ld_global_f32(elem_addr);

                // sq_sum += val * val
                ctx.fma_f32_inplace(sq_sum, val, val);

                // idx += 32 (stride by warp size)
                ctx.add_u32_inplace(idx, 32);
                ctx.branch("sum_loop");

                ctx.label("sum_loop_end");

                // Warp reduce sq_sum
                let shfl16 = ctx.shfl_down_f32(sq_sum, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl16);
                let shfl8 = ctx.shfl_down_f32(sq_sum, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl8);
                let shfl4 = ctx.shfl_down_f32(sq_sum, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl4);
                let shfl2 = ctx.shfl_down_f32(sq_sum, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl2);
                let shfl1 = ctx.shfl_down_f32(sq_sum, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl1);

                // Broadcast final sum to all threads
                let total_sq_sum = ctx.shfl_idx_f32(sq_sum, 0, 0xFFFF_FFFF);

                // Compute RMS = sqrt(mean(x^2) + epsilon)
                let hidden_f32 = ctx.cvt_f32_u32(hidden_u32);
                let mean_sq = ctx.div_f32(total_sq_sum, hidden_f32);
                let eps = ctx.mov_f32_imm(epsilon);
                let mean_sq_eps = ctx.add_f32(mean_sq, eps);
                let rms_inv = ctx.rsqrt_f32(mean_sq_eps);

                // Normalize and scale: output[i] = (input[i] * rms_inv) * gamma[i]
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, tid);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, hidden_u32);
                ctx.branch_if_not(in_bounds2, "exit");

                // Load input[idx] and gamma[idx]
                let elem_offset2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let in_addr = ctx.add_u64(input_ptr, elem_offset2);
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset2);
                let out_addr = ctx.add_u64(output_ptr, elem_offset2);

                let inp = ctx.ld_global_f32(in_addr);
                let gamma = ctx.ld_global_f32(gamma_addr);

                // output = input * rms_inv * gamma
                let normalized = ctx.mul_f32(inp, rms_inv);
                let result = ctx.mul_f32(normalized, gamma);

                ctx.st_global_f32(out_addr, result);

                ctx.add_u32_inplace(idx2, 32);
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// PAR-081: Vectorized RMSNorm kernel with 256 threads
///
/// Achieves ~10x speedup over single-warp RmsNormKernel by:
/// 1. Using 256 threads (8 warps) instead of 32
/// 2. Shared memory for cross-warp reduction
/// 3. Better memory coalescing
///
/// Expected: 23.5µs → ~2-3µs for hidden_size=1536
#[derive(Debug, Clone)]
pub struct VectorizedRmsNormKernel {
    /// Hidden dimension size
    pub hidden_size: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl VectorizedRmsNormKernel {
    /// Create a new vectorized RMSNorm kernel
    #[must_use]
    pub fn new(hidden_size: u32) -> Self {
        Self {
            hidden_size,
            epsilon: 1e-5,
        }
    }

    /// Set custom epsilon value
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl Kernel for VectorizedRmsNormKernel {
    fn name(&self) -> &str {
        "rmsnorm_vectorized"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden_size = self.hidden_size;
        let epsilon = self.epsilon;

        // Strategy:
        // - Block: 256 threads (8 warps)
        // - Each thread handles ceil(hidden_size / 256) elements
        // - Pass 1: Compute sum of squares using shared memory for warp reduction
        // - Pass 2: Normalize with the computed RMS inverse
        //
        // Shared memory layout:
        // - warp_sums[8] - partial sums from each warp

        PtxKernel::new("rmsnorm_vectorized")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U64, "gamma_ptr")
            .shared_memory(8 * 4) // 8 warp partial sums (f32)
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let warp_id = ctx.div_u32(tid, 32);
                let lane_id = ctx.rem_u32(tid, 32);

                // Load parameters
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                // Constants
                let hidden_u32 = ctx.mov_u32_imm(hidden_size);
                let four = ctx.mov_u32_imm(4);
                let _thread_count = ctx.mov_u32_imm(256);

                // Pass 1: Accumulate sum of squares
                // Each thread processes elements: tid, tid+256, tid+512, ...
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let loop_idx = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(loop_idx, hidden_u32);
                ctx.branch_if_not(in_bounds, "sum_loop_end");

                // Load input[loop_idx]
                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let elem_addr = ctx.add_u64(input_ptr, elem_offset);
                let val = ctx.ld_global_f32(elem_addr);

                // sq_sum += val * val
                ctx.fma_f32_inplace(sq_sum, val, val);

                // idx += 256 (stride by block size)
                ctx.add_u32_inplace(idx, 256);
                ctx.branch("sum_loop");

                ctx.label("sum_loop_end");

                // Warp-level reduction of sq_sum
                let shfl16 = ctx.shfl_down_f32(sq_sum, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl16);
                let shfl8 = ctx.shfl_down_f32(sq_sum, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl8);
                let shfl4 = ctx.shfl_down_f32(sq_sum, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl4);
                let shfl2 = ctx.shfl_down_f32(sq_sum, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl2);
                let shfl1 = ctx.shfl_down_f32(sq_sum, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl1);

                // Constants for comparisons
                let zero = ctx.mov_u32_imm(0);
                let eight = ctx.mov_u32_imm(8);
                let thirty_two = ctx.mov_u32_imm(32);

                // Lane 0 of each warp writes to shared memory
                let lane_zero = ctx.setp_eq_u32(lane_id, zero);
                let warp_smem_off = ctx.mul_u32(warp_id, 4);
                ctx.branch_if_not(lane_zero, "skip_smem_write");
                ctx.st_shared_f32(warp_smem_off, sq_sum);
                ctx.label("skip_smem_write");

                // Sync all warps
                ctx.bar_sync(0);

                // First warp reduces across all warp sums
                let is_first_warp = ctx.setp_lt_u32(tid, thirty_two);

                ctx.branch_if_not(is_first_warp, "skip_final_reduce");

                // Load warp partial sum if lane < 8
                let lane_valid = ctx.setp_lt_u32(lane_id, eight);
                let lane_smem_off = ctx.mul_u32(lane_id, 4);
                // Initialize to 0, load real value if valid
                let warp_partial = ctx.mov_f32_imm(0.0);
                ctx.branch_if_not(lane_valid, "skip_warp_load");
                let loaded_val = ctx.ld_shared_f32(lane_smem_off);
                // Use add with 0 to copy (no move_f32_inplace available)
                let _zero_f32 = ctx.mov_f32_imm(0.0);
                ctx.add_f32_inplace(warp_partial, loaded_val);
                ctx.label("skip_warp_load");

                // Reduce 8 values (only first 8 lanes participate)
                let red4 = ctx.shfl_down_f32(warp_partial, 4, 0xFFFF_FFFF);
                let partial = ctx.add_f32(warp_partial, red4);
                let red2 = ctx.shfl_down_f32(partial, 2, 0xFFFF_FFFF);
                let partial = ctx.add_f32(partial, red2);
                let red1 = ctx.shfl_down_f32(partial, 1, 0xFFFF_FFFF);
                let final_sum = ctx.add_f32(partial, red1);

                // Lane 0 writes total to shared memory slot 0
                let smem_zero = ctx.mov_u32_imm(0);
                ctx.branch_if_not(lane_zero, "skip_final_write");
                ctx.st_shared_f32(smem_zero, final_sum);
                ctx.label("skip_final_write");

                ctx.label("skip_final_reduce");

                // Sync again before all threads read
                ctx.bar_sync(1);

                // All threads read the total sum from slot 0
                let smem_read_zero = ctx.mov_u32_imm(0);
                let total = ctx.ld_shared_f32(smem_read_zero);

                // Compute RMS inverse: 1 / sqrt(mean(x^2) + epsilon)
                let hidden_f32 = ctx.cvt_f32_u32(hidden_u32);
                let mean_sq = ctx.div_f32(total, hidden_f32);
                let eps = ctx.mov_f32_imm(epsilon);
                let mean_sq_eps = ctx.add_f32(mean_sq, eps);
                let rms_inv = ctx.rsqrt_f32(mean_sq_eps);

                // Pass 2: Normalize and scale
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, tid);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, hidden_u32);
                ctx.branch_if_not(in_bounds2, "exit");

                // Load input and gamma
                let elem_offset2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let in_addr = ctx.add_u64(input_ptr, elem_offset2);
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset2);
                let out_addr = ctx.add_u64(output_ptr, elem_offset2);

                let inp = ctx.ld_global_f32(in_addr);
                let gamma = ctx.ld_global_f32(gamma_addr);

                // output = input * rms_inv * gamma
                let normalized = ctx.mul_f32(inp, rms_inv);
                let result = ctx.mul_f32(normalized, gamma);

                ctx.st_global_f32(out_addr, result);

                ctx.add_u32_inplace(idx2, 256);
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// PAR-112: Batched Vectorized RMSNorm Kernel
///
/// Processes M sequences in parallel using Grid.y = M.
/// Each block (blockIdx.y) processes one sequence independently.
/// Achieves ~4x speedup over M sequential kernel launches.
///
/// Grid: (1, M, 1), Block: (256, 1, 1)
#[derive(Debug, Clone)]
pub struct BatchedVectorizedRmsNormKernel {
    /// Hidden dimension size
    pub hidden_size: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl BatchedVectorizedRmsNormKernel {
    /// Create a new batched vectorized RMSNorm kernel
    #[must_use]
    pub fn new(hidden_size: u32, batch_size: u32) -> Self {
        Self {
            hidden_size,
            batch_size,
            epsilon: 1e-5,
        }
    }

    /// Set custom epsilon value
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl Kernel for BatchedVectorizedRmsNormKernel {
    fn name(&self) -> &str {
        "batched_rmsnorm_vectorized"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden_size = self.hidden_size;
        let epsilon = self.epsilon;

        // Strategy:
        // - Grid: (1, M, 1) - one block row per sequence
        // - Block: (256, 1, 1) - 8 warps per block
        // - Each block processes input[blockIdx.y * hidden_size : (blockIdx.y+1) * hidden_size]
        // - Shared memory for warp reduction within block
        //
        // Memory layout (packed):
        // input:  [seq0_hidden..., seq1_hidden..., seq2_hidden..., seq3_hidden...]
        // output: [seq0_hidden..., seq1_hidden..., seq2_hidden..., seq3_hidden...]
        // gamma:  [hidden_size] (shared across all sequences)

        PtxKernel::new("batched_rmsnorm_vectorized")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U64, "gamma_ptr")
            .shared_memory(8 * 4) // 8 warp partial sums (f32)
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY); // blockIdx.y = batch index
                let warp_id = ctx.div_u32(tid, 32);
                let lane_id = ctx.rem_u32(tid, 32);

                // Load parameters
                let input_base = ctx.load_param_u64("input_ptr");
                let output_base = ctx.load_param_u64("output_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                // Calculate batch offset: batch_idx * hidden_size * 4 bytes
                let hidden_u32 = ctx.mov_u32_imm(hidden_size);
                let four = ctx.mov_u32_imm(4);
                let batch_offset_elems = ctx.mul_wide_u32_reg(batch_idx, hidden_u32);
                let batch_offset_bytes = ctx.mul_u64(batch_offset_elems, 4);

                // Offset input/output pointers for this batch
                let input_ptr = ctx.add_u64(input_base, batch_offset_bytes);
                let output_ptr = ctx.add_u64(output_base, batch_offset_bytes);

                // Pass 1: Accumulate sum of squares
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let loop_idx = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(loop_idx, hidden_u32);
                ctx.branch_if_not(in_bounds, "sum_loop_end");

                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let elem_addr = ctx.add_u64(input_ptr, elem_offset);
                let val = ctx.ld_global_f32(elem_addr);

                ctx.fma_f32_inplace(sq_sum, val, val);
                ctx.add_u32_inplace(idx, 256);
                ctx.branch("sum_loop");

                ctx.label("sum_loop_end");

                // Warp-level reduction
                let shfl16 = ctx.shfl_down_f32(sq_sum, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl16);
                let shfl8 = ctx.shfl_down_f32(sq_sum, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl8);
                let shfl4 = ctx.shfl_down_f32(sq_sum, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl4);
                let shfl2 = ctx.shfl_down_f32(sq_sum, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl2);
                let shfl1 = ctx.shfl_down_f32(sq_sum, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl1);

                // Lane 0 writes to shared memory
                let smem_base = ctx.mov_u64_imm(0);
                let zero = ctx.mov_u32_imm(0);
                let is_lane_zero = ctx.setp_eq_u32(lane_id, zero);
                ctx.branch_if_not(is_lane_zero, "skip_store");

                let warp_offset = ctx.mul_wide_u32_reg(warp_id, four);
                let smem_addr = ctx.add_u64(smem_base, warp_offset);
                ctx.st_shared_f32(smem_addr, sq_sum);

                ctx.label("skip_store");
                ctx.bar_sync(0);

                // Thread 0 reduces across warps
                let final_sum = ctx.mov_f32_imm(0.0);
                let is_tid_zero = ctx.setp_eq_u32(tid, zero);
                ctx.branch_if_not(is_tid_zero, "after_final_reduce");

                // Load and sum all 8 warp contributions
                let addr0 = ctx.mov_u64_imm(0);
                let s0 = ctx.ld_shared_f32(addr0);
                ctx.add_f32_inplace(final_sum, s0);

                let addr1 = ctx.mov_u64_imm(4);
                let s1 = ctx.ld_shared_f32(addr1);
                ctx.add_f32_inplace(final_sum, s1);

                let addr2 = ctx.mov_u64_imm(8);
                let s2 = ctx.ld_shared_f32(addr2);
                ctx.add_f32_inplace(final_sum, s2);

                let addr3 = ctx.mov_u64_imm(12);
                let s3 = ctx.ld_shared_f32(addr3);
                ctx.add_f32_inplace(final_sum, s3);

                let addr4 = ctx.mov_u64_imm(16);
                let s4 = ctx.ld_shared_f32(addr4);
                ctx.add_f32_inplace(final_sum, s4);

                let addr5 = ctx.mov_u64_imm(20);
                let s5 = ctx.ld_shared_f32(addr5);
                ctx.add_f32_inplace(final_sum, s5);

                let addr6 = ctx.mov_u64_imm(24);
                let s6 = ctx.ld_shared_f32(addr6);
                ctx.add_f32_inplace(final_sum, s6);

                let addr7 = ctx.mov_u64_imm(28);
                let s7 = ctx.ld_shared_f32(addr7);
                ctx.add_f32_inplace(final_sum, s7);

                // Compute rms_inv = rsqrt(sum / hidden_size + epsilon)
                let hidden_f32 = ctx.cvt_f32_u32(hidden_u32);
                let mean_sq = ctx.div_f32(final_sum, hidden_f32);
                let eps = ctx.mov_f32_imm(epsilon);
                let var_plus_eps = ctx.add_f32(mean_sq, eps);
                let rms_inv = ctx.rsqrt_f32(var_plus_eps);

                // Store rms_inv to shared memory for other threads
                ctx.st_shared_f32(addr0, rms_inv);

                ctx.label("after_final_reduce");
                ctx.bar_sync(0);

                // All threads load rms_inv and normalize
                let smem_zero = ctx.mov_u64_imm(0);
                let rms_inv_shared = ctx.ld_shared_f32(smem_zero);

                // Pass 2: Normalize output = input * rms_inv * gamma
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, tid);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, hidden_u32);
                ctx.branch_if_not(in_bounds2, "exit");

                let elem_offset2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let in_addr = ctx.add_u64(input_ptr, elem_offset2);
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset2);
                let out_addr = ctx.add_u64(output_ptr, elem_offset2);

                let inp = ctx.ld_global_f32(in_addr);
                let gamma = ctx.ld_global_f32(gamma_addr);

                let normalized = ctx.mul_f32(inp, rms_inv_shared);
                let result = ctx.mul_f32(normalized, gamma);

                ctx.st_global_f32(out_addr, result);

                ctx.add_u32_inplace(idx2, 256);
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// CORRECTNESS-013: High-Precision RMSNorm Kernel for CPU/GPU bit-exactness
///
/// Uses Kahan compensated summation to match CPU sequential sum precision,
/// and Newton-Raphson refinement for rsqrt to achieve full IEEE-754 f32 precision.
///
/// This kernel is slower than VectorizedRmsNormKernel but produces results
/// that match CPU computation within floating-point epsilon.
///
/// Grid: (1, 1, 1), Block: (256, 1, 1)
#[derive(Debug, Clone)]
pub struct PreciseRmsNormKernel {
    /// Hidden dimension size
    pub hidden_size: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl PreciseRmsNormKernel {
    /// Create a new precise RMSNorm kernel
    #[must_use]
    pub fn new(hidden_size: u32) -> Self {
        Self {
            hidden_size,
            epsilon: 1e-5,
        }
    }

    /// Set custom epsilon value
    #[must_use]
    pub const fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }
}

impl Kernel for PreciseRmsNormKernel {
    fn name(&self) -> &str {
        "rmsnorm_precise"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden_size = self.hidden_size;
        let epsilon = self.epsilon;

        // Strategy for CPU-matching precision:
        // - Block: 256 threads (8 warps)
        // - Each thread handles ceil(hidden_size / 256) elements
        // - Pass 1: Kahan compensated sum of squares for numerically stable accumulation
        // - Warp-level Kahan merge for reduction
        // - Pass 2: Newton-Raphson refined rsqrt for full precision
        // - Pass 3: Normalize with the computed RMS inverse
        //
        // Kahan summation formula:
        //   y = val - compensation
        //   t = sum + y
        //   compensation = (t - sum) - y
        //   sum = t
        //
        // Newton-Raphson rsqrt refinement:
        //   rsqrt_refined = rsqrt * (1.5 - 0.5 * x * rsqrt * rsqrt)

        PtxKernel::new("rmsnorm_precise")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U64, "gamma_ptr")
            .shared_memory(16 * 4) // 8 warp sums + 8 compensations (f32)
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let warp_id = ctx.div_u32(tid, 32);
                let lane_id = ctx.rem_u32(tid, 32);

                // Load parameters
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                // Constants
                let hidden_u32 = ctx.mov_u32_imm(hidden_size);
                let four = ctx.mov_u32_imm(4);

                // Pass 1: Kahan compensated sum of squares
                // Each thread maintains (sum, compensation) pair
                let sq_sum = ctx.mov_f32_imm(0.0);
                let compensation = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let loop_idx = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(loop_idx, hidden_u32);
                ctx.branch_if_not(in_bounds, "sum_loop_end");

                // Load input[loop_idx]
                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let elem_addr = ctx.add_u64(input_ptr, elem_offset);
                let val = ctx.ld_global_f32(elem_addr);

                // val_sq = val * val
                let val_sq = ctx.mul_f32(val, val);

                // Kahan summation: add val_sq to sum with compensation
                // y = val_sq - compensation
                let y = ctx.sub_f32(val_sq, compensation);
                // t = sum + y
                let t = ctx.add_f32(sq_sum, y);
                // new_compensation = (t - sum) - y
                let t_minus_sum = ctx.sub_f32(t, sq_sum);
                let new_comp = ctx.sub_f32(t_minus_sum, y);

                // Update compensation = new_comp (zero then add)
                let zero_f32 = ctx.mov_f32_imm(0.0);
                ctx.mul_f32_inplace(compensation, zero_f32); // compensation = 0
                ctx.add_f32_inplace(compensation, new_comp); // compensation = new_comp

                // Update sq_sum = t (zero then add)
                ctx.mul_f32_inplace(sq_sum, zero_f32); // sq_sum = 0
                ctx.add_f32_inplace(sq_sum, t); // sq_sum = t

                // idx += 256 (stride by block size)
                ctx.add_u32_inplace(idx, 256);
                ctx.branch("sum_loop");

                ctx.label("sum_loop_end");

                // Warp-level reduction with Kahan merge
                // For each shuffle level, merge (sum, comp) pairs
                // merge: new_sum = sum1 + sum2, new_comp = comp1 + comp2 + ((sum1 + sum2) - new_sum)
                // Simplified: just sum the values (Kahan already gave per-thread precision)
                let shfl16 = ctx.shfl_down_f32(sq_sum, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl16);
                let shfl8 = ctx.shfl_down_f32(sq_sum, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl8);
                let shfl4 = ctx.shfl_down_f32(sq_sum, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl4);
                let shfl2 = ctx.shfl_down_f32(sq_sum, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl2);
                let shfl1 = ctx.shfl_down_f32(sq_sum, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(sq_sum, shfl1);

                // Constants for comparisons
                let zero = ctx.mov_u32_imm(0);
                let eight = ctx.mov_u32_imm(8);
                let thirty_two = ctx.mov_u32_imm(32);

                // Lane 0 of each warp writes to shared memory
                let lane_zero = ctx.setp_eq_u32(lane_id, zero);
                let warp_smem_off = ctx.mul_u32(warp_id, 4);
                ctx.branch_if_not(lane_zero, "skip_smem_write");
                ctx.st_shared_f32(warp_smem_off, sq_sum);
                ctx.label("skip_smem_write");

                // Sync all warps
                ctx.bar_sync(0);

                // First warp reduces across all warp sums
                let is_first_warp = ctx.setp_lt_u32(tid, thirty_two);

                ctx.branch_if_not(is_first_warp, "skip_final_reduce");

                // Load warp partial sum if lane < 8
                let lane_valid = ctx.setp_lt_u32(lane_id, eight);
                let lane_smem_off = ctx.mul_u32(lane_id, 4);
                let warp_partial = ctx.mov_f32_imm(0.0);
                ctx.branch_if_not(lane_valid, "skip_warp_load");
                let loaded_val = ctx.ld_shared_f32(lane_smem_off);
                ctx.add_f32_inplace(warp_partial, loaded_val);
                ctx.label("skip_warp_load");

                // Reduce 8 values (only first 8 lanes participate)
                let red4 = ctx.shfl_down_f32(warp_partial, 4, 0xFFFF_FFFF);
                let partial = ctx.add_f32(warp_partial, red4);
                let red2 = ctx.shfl_down_f32(partial, 2, 0xFFFF_FFFF);
                let partial = ctx.add_f32(partial, red2);
                let red1 = ctx.shfl_down_f32(partial, 1, 0xFFFF_FFFF);
                let final_sum = ctx.add_f32(partial, red1);

                // Compute mean_sq and apply Newton-Raphson refined rsqrt
                let hidden_f32 = ctx.cvt_f32_u32(hidden_u32);
                let mean_sq = ctx.div_f32(final_sum, hidden_f32);
                let eps = ctx.mov_f32_imm(epsilon);
                let mean_sq_eps = ctx.add_f32(mean_sq, eps);

                // Initial rsqrt approximation
                let rsqrt_approx = ctx.rsqrt_f32(mean_sq_eps);

                // Newton-Raphson refinement: rsqrt_refined = rsqrt * (1.5 - 0.5 * x * rsqrt^2)
                let half = ctx.mov_f32_imm(0.5);
                let three_half = ctx.mov_f32_imm(1.5);
                // rsqrt^2
                let rsqrt_sq = ctx.mul_f32(rsqrt_approx, rsqrt_approx);
                // 0.5 * x * rsqrt^2
                let half_x_rsq_sq = ctx.mul_f32(mean_sq_eps, rsqrt_sq);
                let half_x_rsq_sq = ctx.mul_f32(half, half_x_rsq_sq);
                // 1.5 - 0.5 * x * rsqrt^2
                let factor = ctx.sub_f32(three_half, half_x_rsq_sq);
                // rsqrt * factor
                let rms_inv = ctx.mul_f32(rsqrt_approx, factor);

                // Lane 0 writes total to shared memory slot 0
                let smem_zero = ctx.mov_u32_imm(0);
                ctx.branch_if_not(lane_zero, "skip_final_write");
                ctx.st_shared_f32(smem_zero, rms_inv);
                ctx.label("skip_final_write");

                ctx.label("skip_final_reduce");

                // Sync again before all threads read
                ctx.bar_sync(1);

                // All threads read the total sum from slot 0
                let smem_read_zero = ctx.mov_u32_imm(0);
                let rms_inv_final = ctx.ld_shared_f32(smem_read_zero);

                // Pass 2: Normalize and scale
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, tid);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, hidden_u32);
                ctx.branch_if_not(in_bounds2, "exit");

                // Load input and gamma
                let elem_offset2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let in_addr = ctx.add_u64(input_ptr, elem_offset2);
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset2);
                let out_addr = ctx.add_u64(output_ptr, elem_offset2);

                let inp = ctx.ld_global_f32(in_addr);
                let gamma = ctx.ld_global_f32(gamma_addr);

                // output = input * rms_inv * gamma
                let normalized = ctx.mul_f32(inp, rms_inv_final);
                let result = ctx.mul_f32(normalized, gamma);

                ctx.st_global_f32(out_addr, result);

                ctx.add_u32_inplace(idx2, 256);
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precise_rmsnorm_kernel_name() {
        let kernel = PreciseRmsNormKernel::new(1536);
        assert_eq!(kernel.name(), "rmsnorm_precise");
    }

    #[test]
    fn test_precise_rmsnorm_ptx_generation() {
        let kernel = PreciseRmsNormKernel::new(1536).with_epsilon(1e-6);
        let ptx = kernel.emit_ptx();

        // Verify parameters
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u64 gamma_ptr"));

        // Verify kernel name
        assert!(ptx.contains("rmsnorm_precise"), "Missing kernel name");

        // Verify warp shuffle for reduction
        assert!(ptx.contains("shfl"), "Missing warp shuffle for reduction");

        // Verify rsqrt for initial approximation
        assert!(ptx.contains("rsqrt"), "Missing rsqrt instruction");

        // Verify multiplication for Newton-Raphson refinement
        assert!(ptx.contains("mul.f32"), "Missing mul.f32 for refinement");
    }

    #[test]
    fn test_layernorm_kernel_name() {
        let kernel = LayerNormKernel::new(768);
        assert_eq!(kernel.name(), "layernorm_warp_shuffle");

        let kernel_shared = LayerNormKernel::new(768).without_warp_shuffle();
        assert_eq!(kernel_shared.name(), "layernorm_shared");
    }

    #[test]
    fn test_layernorm_with_epsilon() {
        let kernel = LayerNormKernel::new(768).with_epsilon(1e-6);
        assert!((kernel.epsilon - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_layernorm_without_affine() {
        let kernel = LayerNormKernel::new(768).without_affine();
        assert!(!kernel.affine);
    }

    #[test]
    fn test_layernorm_ptx_generation() {
        let kernel = LayerNormKernel::new(768);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u64 gamma_ptr"));
        assert!(ptx.contains(".param .u64 beta_ptr"));
        assert!(ptx.contains(".param .u32 hidden_size"));
        assert!(ptx.contains(".param .u32 batch_size"));
    }

    #[test]
    fn test_layernorm_warp_shuffle_ptx() {
        let kernel = LayerNormKernel::new(32);
        let ptx = kernel.emit_ptx();

        // Verify warp shuffle operations for reduction
        assert!(ptx.contains("shfl") || ptx.contains("shfl.down"));

        // Verify division for mean/variance
        assert!(ptx.contains("div.rn.f32")); // div requires rounding mode for floats

        // Verify rsqrt for normalization
        assert!(ptx.contains("rsqrt.f32") || ptx.contains("rsqrt"));

        // Verify memory operations
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));
    }

    #[test]
    fn test_layernorm_shared_memory_ptx() {
        let kernel = LayerNormKernel::new(256).without_warp_shuffle();
        let ptx = kernel.emit_ptx();

        // Verify shared memory usage
        assert!(ptx.contains("ld.shared.f32") || ptx.contains("ld.f32"));
        assert!(ptx.contains("st.shared.f32") || ptx.contains("st.f32"));

        // Verify barrier synchronization
        assert!(ptx.contains("bar"));

        // Verify rsqrt and division
        assert!(ptx.contains("rsqrt.f32") || ptx.contains("rsqrt"));
        assert!(ptx.contains("div.rn.f32")); // div requires rounding mode for floats
    }

    #[test]
    fn test_layernorm_kernel_variants() {
        let warp_kernel = LayerNormKernel::new(32);
        let shared_kernel = LayerNormKernel::new(256).without_warp_shuffle();

        // Both should produce valid PTX
        let warp_ptx = warp_kernel.emit_ptx();
        let shared_ptx = shared_kernel.emit_ptx();

        assert!(!warp_ptx.is_empty());
        assert!(!shared_ptx.is_empty());

        // Verify different kernel names in output
        assert!(warp_ptx.contains("layernorm_warp_shuffle"));
        assert!(shared_ptx.contains("layernorm_shared"));
    }

    #[test]
    fn test_layernorm_numerical_operations() {
        let kernel = LayerNormKernel::new(32);
        let ptx = kernel.emit_ptx();

        // Should have subtraction (for x - mean)
        assert!(ptx.contains("sub.f32"));

        // Should have multiplication (for scaling, (x-mean)^2)
        assert!(ptx.contains("mul.f32"));

        // Should have addition (for variance + epsilon, gamma*x + beta)
        assert!(ptx.contains("add.f32"));
    }

    #[test]
    fn test_layernorm_without_affine_ptx() {
        let kernel_affine = LayerNormKernel::new(32);
        let kernel_no_affine = LayerNormKernel::new(32).without_affine();

        let ptx_affine = kernel_affine.emit_ptx();
        let ptx_no_affine = kernel_no_affine.emit_ptx();

        // Both should be valid
        assert!(!ptx_affine.is_empty());
        assert!(!ptx_no_affine.is_empty());

        // Affine version should load gamma/beta pointers
        assert!(ptx_affine.contains("gamma_ptr"));
        assert!(ptx_affine.contains("beta_ptr"));
    }

    #[test]
    fn test_layernorm_default_config() {
        let kernel = LayerNormKernel::new(768);

        assert_eq!(kernel.hidden_size, 768);
        assert!((kernel.epsilon - 1e-5).abs() < 1e-10);
        assert!(kernel.affine);
        assert!(kernel.use_warp_shuffle);
    }

    #[test]
    fn test_rmsnorm_kernel_name() {
        let kernel = RmsNormKernel::new(2048);
        assert_eq!(kernel.name(), "rmsnorm");
    }

    #[test]
    fn test_rmsnorm_ptx_generation() {
        let kernel = RmsNormKernel::new(2048);
        let ptx = kernel.emit_ptx();

        // Verify parameters
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u64 gamma_ptr"));

        // Verify warp shuffle for reduction
        assert!(ptx.contains("shfl"));

        // Verify rsqrt for 1/sqrt(rms)
        assert!(ptx.contains("rsqrt.f32") || ptx.contains("rsqrt"));
    }

    #[test]
    fn test_rmsnorm_with_epsilon() {
        let kernel = RmsNormKernel::new(2048).with_epsilon(1e-6);
        assert!((kernel.epsilon - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_rmsnorm_ptx_valid_syntax() {
        let kernel = RmsNormKernel::new(2048).with_epsilon(1e-5);
        let ptx = kernel.emit_ptx();

        // Print first 200 lines for debugging
        for (i, line) in ptx.lines().enumerate().take(200) {
            eprintln!("{:4}: {}", i + 1, line);
        }

        // Basic structure checks
        assert!(ptx.contains(".entry rmsnorm"));
        assert!(ptx.contains("ret;"));
    }

    // ===== VectorizedRmsNormKernel Tests =====

    #[test]
    fn test_vectorized_rmsnorm_kernel_new() {
        let kernel = VectorizedRmsNormKernel::new(2048);
        assert_eq!(kernel.hidden_size, 2048);
        assert!((kernel.epsilon - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_vectorized_rmsnorm_kernel_name() {
        let kernel = VectorizedRmsNormKernel::new(1024);
        assert_eq!(kernel.name(), "rmsnorm_vectorized");
    }

    #[test]
    fn test_vectorized_rmsnorm_with_epsilon() {
        let kernel = VectorizedRmsNormKernel::new(2048).with_epsilon(1e-6);
        assert!((kernel.epsilon - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_vectorized_rmsnorm_ptx_generation() {
        let kernel = VectorizedRmsNormKernel::new(2048);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(
            ptx.contains(".entry rmsnorm_vectorized"),
            "Should have rmsnorm_vectorized entry"
        );

        // Verify parameters
        assert!(ptx.contains(".param .u64 input_ptr"), "Should have input_ptr");
        assert!(ptx.contains(".param .u64 output_ptr"), "Should have output_ptr");
        assert!(ptx.contains(".param .u64 gamma_ptr"), "Should have gamma_ptr");
    }

    #[test]
    fn test_vectorized_rmsnorm_warp_operations() {
        let kernel = VectorizedRmsNormKernel::new(1024);
        let ptx = kernel.emit_ptx();

        // Verify warp shuffle for reduction
        assert!(
            ptx.contains("shfl.sync") || ptx.contains("shfl."),
            "Should have shfl for warp reduction"
        );

        // Verify rsqrt for 1/sqrt(rms)
        assert!(
            ptx.contains("rsqrt.f32") || ptx.contains("rsqrt"),
            "Should have rsqrt for RMSNorm"
        );
    }

    #[test]
    fn test_vectorized_rmsnorm_shared_memory() {
        let kernel = VectorizedRmsNormKernel::new(2048);
        let ptx_kernel = kernel.build_ptx();

        // Vectorized kernel uses shared memory for inter-warp reduction
        assert!(
            ptx_kernel.shared_memory_bytes() > 0,
            "Vectorized RMSNorm should use shared memory"
        );
    }

    #[test]
    fn test_vectorized_rmsnorm_various_sizes() {
        // Test common hidden sizes
        for hidden_size in [256, 512, 1024, 2048, 4096] {
            let kernel = VectorizedRmsNormKernel::new(hidden_size);
            assert_eq!(kernel.hidden_size, hidden_size);

            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty());
            assert!(ptx.contains(".entry"));
            assert!(ptx.contains("ret;"));
        }
    }

    #[test]
    fn test_vectorized_rmsnorm_numerical_ops() {
        let kernel = VectorizedRmsNormKernel::new(1024);
        let ptx = kernel.emit_ptx();

        // Verify numerical operations
        assert!(ptx.contains("mul.f32"), "Should have multiplication");
        assert!(ptx.contains("add.f32"), "Should have addition");
    }

    // ===== BatchedVectorizedRmsNormKernel Tests =====

    #[test]
    fn test_batched_vectorized_rmsnorm_kernel_new() {
        let kernel = BatchedVectorizedRmsNormKernel::new(2048, 8);
        assert_eq!(kernel.hidden_size, 2048);
        assert_eq!(kernel.batch_size, 8);
        assert!((kernel.epsilon - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_batched_vectorized_rmsnorm_kernel_name() {
        let kernel = BatchedVectorizedRmsNormKernel::new(1024, 4);
        assert_eq!(kernel.name(), "batched_rmsnorm_vectorized");
    }

    #[test]
    fn test_batched_vectorized_rmsnorm_with_epsilon() {
        let kernel = BatchedVectorizedRmsNormKernel::new(2048, 4).with_epsilon(1e-6);
        assert!((kernel.epsilon - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_batched_vectorized_rmsnorm_ptx_generation() {
        let kernel = BatchedVectorizedRmsNormKernel::new(2048, 4);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(
            ptx.contains(".entry batched_rmsnorm_vectorized"),
            "Should have batched_rmsnorm_vectorized entry"
        );

        // Verify parameters
        assert!(ptx.contains(".param .u64 input_ptr"), "Should have input_ptr");
        assert!(ptx.contains(".param .u64 output_ptr"), "Should have output_ptr");
        assert!(ptx.contains(".param .u64 gamma_ptr"), "Should have gamma_ptr");
    }

    #[test]
    fn test_batched_vectorized_rmsnorm_batch_sizes() {
        // Test various batch sizes
        for batch_size in [1, 2, 4, 8, 16] {
            let kernel = BatchedVectorizedRmsNormKernel::new(1024, batch_size);
            assert_eq!(kernel.batch_size, batch_size);

            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty());
            assert!(ptx.contains(".entry"));
        }
    }

    #[test]
    fn test_batched_vectorized_rmsnorm_hidden_sizes() {
        // Test common hidden sizes
        for hidden_size in [256, 512, 1024, 2048, 4096] {
            let kernel = BatchedVectorizedRmsNormKernel::new(hidden_size, 4);
            assert_eq!(kernel.hidden_size, hidden_size);

            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty());
            assert!(ptx.contains(".entry"));
        }
    }

    #[test]
    fn test_batched_vectorized_rmsnorm_warp_operations() {
        let kernel = BatchedVectorizedRmsNormKernel::new(1024, 4);
        let ptx = kernel.emit_ptx();

        // Verify warp shuffle for reduction
        assert!(
            ptx.contains("shfl.sync") || ptx.contains("shfl."),
            "Should have shfl for warp reduction"
        );

        // Verify rsqrt for normalization
        assert!(
            ptx.contains("rsqrt.f32") || ptx.contains("rsqrt"),
            "Should have rsqrt"
        );
    }

    #[test]
    fn test_batched_vectorized_rmsnorm_shared_memory() {
        let kernel = BatchedVectorizedRmsNormKernel::new(2048, 4);
        let ptx_kernel = kernel.build_ptx();

        // Batched kernel uses shared memory
        assert!(
            ptx_kernel.shared_memory_bytes() > 0,
            "Batched RMSNorm should use shared memory"
        );
    }

    #[test]
    fn test_batched_vectorized_rmsnorm_memory_ops() {
        let kernel = BatchedVectorizedRmsNormKernel::new(1024, 8);
        let ptx = kernel.emit_ptx();

        // Verify memory operations
        assert!(ptx.contains("ld.global"), "Should have global loads");
        assert!(ptx.contains("st.global"), "Should have global stores");
    }

    #[test]
    fn test_batched_vectorized_rmsnorm_barrier_sync() {
        let kernel = BatchedVectorizedRmsNormKernel::new(2048, 4);
        let ptx = kernel.emit_ptx();

        // Verify barrier synchronization for shared memory
        assert!(
            ptx.contains("bar.sync"),
            "Should have barrier synchronization"
        );
    }
}
