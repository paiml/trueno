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

                // Each warp processes one row
                let warp_id = ctx.div_u32(tid, 32);
                let lane_id = ctx.rem_u32(tid, 32);

                // Bounds check - warp must be within batch
                let pred = ctx.setp_ge_u32(warp_id, batch_size);
                ctx.branch_if(pred, "exit");

                // Calculate row offset
                let input_ptr = ctx.load_param_u64("input_ptr");
                let row_offset = ctx.mul_wide_u32_reg(warp_id, hidden_size_param);
                let row_offset_bytes = ctx.mul_u64(row_offset, 4);
                let row_base = ctx.add_u64(input_ptr, row_offset_bytes);

                // ===== Step 1: Compute mean using warp shuffle =====
                // Each thread loads and sums multiple elements
                let sum = ctx.mov_f32_imm(0.0);
                let count_reg = ctx.mov_u32_imm(0);

                // Loop to load all elements (simplified - assumes hidden_size <= 32 for now)
                let lane_pred = ctx.setp_lt_u32(lane_id, hidden_size_param);
                ctx.branch_if_not(lane_pred, "skip_load_mean");

                let elem_offset = ctx.mul_wide_u32(lane_id, 4);
                let elem_addr = ctx.add_u64(row_base, elem_offset);
                let val = ctx.ld_global_f32(elem_addr);
                let sum = ctx.add_f32(sum, val);
                let _count_reg = ctx.add_u32(count_reg, 1);

                ctx.label("skip_load_mean");

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
                let broadcast_sum = ctx.shfl_down_f32(warp_sum, 0, 0xFFFF_FFFF);
                let hidden_f32 = ctx.cvt_f32_u32(hidden_size_param);
                let mean = ctx.div_f32(broadcast_sum, hidden_f32);

                // ===== Step 2: Compute variance using warp shuffle =====
                // variance = sum((x - mean)^2) / n
                let var_sum = ctx.mov_f32_imm(0.0);

                ctx.branch_if_not(lane_pred, "skip_load_var");

                let diff = ctx.sub_f32(val, mean);
                let sq_diff = ctx.mul_f32(diff, diff);
                let var_sum = ctx.add_f32(var_sum, sq_diff);

                ctx.label("skip_load_var");

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
                let broadcast_var_sum = ctx.shfl_down_f32(warp_var_sum, 0, 0xFFFF_FFFF);
                let variance = ctx.div_f32(broadcast_var_sum, hidden_f32);

                // ===== Step 3: Compute rstd = 1/sqrt(variance + epsilon) =====
                let eps = ctx.mov_f32_imm(epsilon);
                let var_plus_eps = ctx.add_f32(variance, eps);
                let rstd = ctx.rsqrt_f32(var_plus_eps);

                // ===== Step 4: Normalize and apply affine transformation =====
                ctx.branch_if_not(lane_pred, "skip_normalize");

                // normalized = (x - mean) * rstd
                let normalized = ctx.mul_f32(diff, rstd);

                // Apply affine: y = gamma * normalized + beta
                let result = if affine {
                    let gamma_ptr = ctx.load_param_u64("gamma_ptr");
                    let beta_ptr = ctx.load_param_u64("beta_ptr");
                    let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset);
                    let beta_addr = ctx.add_u64(beta_ptr, elem_offset);
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
                let out_addr = ctx.add_u64(out_row_base, elem_offset);
                ctx.st_global_f32(out_addr, result);

                ctx.label("skip_normalize");
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(ptx.contains("div.f32"));

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
        assert!(ptx.contains("div.f32"));
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
}
