//! RMSNorm Backward Kernel
//!
//! Backward (gradient) kernel for Root Mean Square Layer Normalization.
//!
//! ## Mathematical Specification
//!
//! Forward: `y_i = x_i / rms(x) * γ_i` where `rms(x) = sqrt(mean(x²) + ε)`
//!
//! Backward:
//! - `∂L/∂x_i = γ_i/rms * (∂L/∂y_i - x_i/rms² * mean(x · ∂L/∂y · γ))`
//! - `∂L/∂γ_i = Σ_batch (∂L/∂y_i * x_i / rms)`
//!
//! ## Implementation
//!
//! Uses warp shuffle reductions for computing mean(x · ∂L/∂y · γ).
//! One warp processes one row (hidden_dim ≤ 32).
//!
//! ## Falsifiable Prediction (P-RMS-BACK-001)
//!
//! RMSNorm backward matches finite-difference within ε < 1e-5.

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// RMSNorm Backward Kernel (warp-parallel, one row per warp)
///
/// Computes gradients for RMSNorm using warp shuffle reductions.
///
/// # Parameters
/// - `input_ptr`: Original input (x)
/// - `gamma_ptr`: Learned scale parameter (γ)
/// - `rms_ptr`: RMS values from forward pass
/// - `grad_output_ptr`: Gradient from upstream (∂L/∂y)
/// - `grad_input_ptr`: Output gradient for input (∂L/∂x)
/// - `num_rows`: Number of rows (batch size)
/// - `hidden_dim`: Hidden dimension (must be ≤ 32)
/// - `eps`: Epsilon for numerical stability
#[derive(Debug, Clone)]
pub struct RmsNormBackwardKernel {
    /// Number of rows (batch size)
    pub num_rows: u32,
    /// Hidden dimension (max 32 for warp reduction)
    pub hidden_dim: u32,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl RmsNormBackwardKernel {
    /// Create a new RMSNorm backward kernel
    ///
    /// # Arguments
    /// - `num_rows`: Batch size
    /// - `hidden_dim`: Hidden dimension (must be ≤ 32)
    /// - `eps`: Epsilon for numerical stability
    ///
    /// # Panics
    /// Panics if `hidden_dim` > 32
    #[must_use]
    pub fn new(num_rows: u32, hidden_dim: u32, eps: f32) -> Self {
        assert!(
            hidden_dim <= 32,
            "hidden_dim must be ≤ 32 for warp reduction"
        );
        Self {
            num_rows,
            hidden_dim,
            eps,
        }
    }
}

impl Kernel for RmsNormBackwardKernel {
    fn name(&self) -> &str {
        "rms_norm_backward"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden_dim = self.hidden_dim;
        let eps = self.eps;

        PtxKernel::new("rms_norm_backward")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "gamma_ptr")
            .param(PtxType::U64, "rms_ptr")
            .param(PtxType::U64, "grad_output_ptr")
            .param(PtxType::U64, "grad_input_ptr")
            .param(PtxType::U32, "num_rows")
            .param(PtxType::U32, "hidden_dim")
            .build(move |ctx| {
                // Thread indexing: one warp (32 threads) per row
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let global_tid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Lane within warp (0-31)
                let lane = ctx.and_u32_imm(global_tid, 31);

                // Warp ID (row index)
                let warp_id = ctx.shr_u32_imm(global_tid, 5);

                // Load parameters
                let num_rows_param = ctx.load_param_u32("num_rows");
                let hidden_dim_param = ctx.load_param_u32("hidden_dim");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");
                let rms_ptr = ctx.load_param_u64("rms_ptr");
                let grad_output_ptr = ctx.load_param_u64("grad_output_ptr");
                let grad_input_ptr = ctx.load_param_u64("grad_input_ptr");

                // Bounds check: warp_id < num_rows
                let valid_row = ctx.setp_lt_u32(warp_id, num_rows_param);
                ctx.branch_if_not(valid_row, "exit");

                // Bounds check: lane < hidden_dim
                let valid_lane = ctx.setp_lt_u32(lane, hidden_dim_param);

                // Calculate row base offset
                let row_elem_offset = ctx.mul_lo_u32(warp_id, hidden_dim_param);
                let row_byte_offset = ctx.mul_wide_u32(row_elem_offset, 4);
                let input_row_base = ctx.add_u64(input_ptr, row_byte_offset);
                let grad_out_row_base = ctx.add_u64(grad_output_ptr, row_byte_offset);
                let grad_in_row_base = ctx.add_u64(grad_input_ptr, row_byte_offset);

                // Calculate element address within row
                let lane_offset = ctx.mul_wide_u32(lane, 4);
                let input_addr = ctx.add_u64(input_row_base, lane_offset);
                let gamma_addr = ctx.add_u64(gamma_ptr, lane_offset);
                let grad_out_addr = ctx.add_u64(grad_out_row_base, lane_offset);
                let grad_in_addr = ctx.add_u64(grad_in_row_base, lane_offset);

                // Load RMS value for this row
                let rms_offset = ctx.mul_wide_u32(warp_id, 4);
                let rms_addr = ctx.add_u64(rms_ptr, rms_offset);
                let rms = ctx.ld_global_f32(rms_addr);

                // Load x[i], γ[i], and ∂L/∂y[i] with bounds checking
                let x_i = ctx.ld_global_f32_predicated(input_addr, valid_lane, 0.0);
                let gamma_i = ctx.ld_global_f32_predicated(gamma_addr, valid_lane, 0.0);
                let grad_y_i = ctx.ld_global_f32_predicated(grad_out_addr, valid_lane, 0.0);

                // Compute x_i * grad_y_i * gamma_i
                let x_grad_gamma = ctx.mul_f32(x_i, grad_y_i);
                let x_grad_gamma = ctx.mul_f32(x_grad_gamma, gamma_i);

                // Warp reduction for mean(x * grad_y * gamma)
                let mut sum = x_grad_gamma;
                let warp_mask = 0xFFFF_FFFFu32;

                for offset in [16u32, 8, 4, 2, 1] {
                    if offset < hidden_dim {
                        let shuffled = ctx.shfl_down_f32(sum, offset, warp_mask);
                        sum = ctx.add_f32(sum, shuffled);
                    }
                }

                // Broadcast sum to all lanes
                let total_sum = ctx.shfl_idx_f32(sum, 0, warp_mask);

                // Compute mean: divide by hidden_dim
                let hidden_dim_f32 = ctx.cvt_f32_u32(hidden_dim_param);
                let mean_term = ctx.div_f32(total_sum, hidden_dim_f32);

                // Compute grad_x_i = gamma_i/rms * (grad_y_i - x_i/rms² * mean_term)
                // = gamma_i/rms * grad_y_i - gamma_i * x_i * mean_term / rms³
                let eps_const = ctx.mov_f32_imm(eps);
                let rms_sq = ctx.mul_f32(rms, rms);
                let rms_sq_eps = ctx.add_f32(rms_sq, eps_const);
                let rms_safe = ctx.sqrt_f32(rms_sq_eps);

                // gamma_i / rms
                let gamma_over_rms = ctx.div_f32(gamma_i, rms_safe);

                // x_i / rms²
                let x_over_rms_sq = ctx.div_f32(x_i, rms_sq_eps);

                // x_i / rms² * mean_term
                let correction = ctx.mul_f32(x_over_rms_sq, mean_term);

                // grad_y_i - correction
                let adjusted_grad = ctx.sub_f32(grad_y_i, correction);

                // final gradient
                let grad_x_i = ctx.mul_f32(gamma_over_rms, adjusted_grad);

                // Store result only for valid lanes
                ctx.branch_if_not(valid_lane, "exit");
                ctx.st_global_f32(grad_in_addr, grad_x_i);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm_backward_name() {
        let kernel = RmsNormBackwardKernel::new(64, 32, 1e-6);
        assert_eq!(kernel.name(), "rms_norm_backward");
    }

    #[test]
    fn test_rms_norm_backward_ptx_generation() {
        let kernel = RmsNormBackwardKernel::new(64, 32, 1e-6);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry rms_norm_backward"));
        // Verify parameters
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 gamma_ptr"));
        assert!(ptx.contains(".param .u64 rms_ptr"));
        assert!(ptx.contains(".param .u64 grad_output_ptr"));
        assert!(ptx.contains(".param .u64 grad_input_ptr"));
        // Verify warp shuffle for reduction
        assert!(ptx.contains("shfl.sync.down"));
        // Verify sqrt for RMS - check both rn and approx variants
        assert!(
            ptx.contains("sqrt.rn.f32") || ptx.contains("sqrt"),
            "PTX should contain sqrt: {}",
            ptx
        );
    }

    #[test]
    fn test_rms_norm_backward_small_hidden() {
        let kernel = RmsNormBackwardKernel::new(128, 16, 1e-5);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry rms_norm_backward"));
        assert!(ptx.contains("shfl.sync"));
    }

    #[test]
    fn test_rms_norm_backward_barrier_safety() {
        let kernel = RmsNormBackwardKernel::new(64, 32, 1e-6);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "RMSNorm backward should be barrier-safe: {:?}",
            result.violations
        );
    }

    #[test]
    #[should_panic(expected = "hidden_dim must be ≤ 32")]
    fn test_rms_norm_backward_hidden_dim_limit() {
        let _ = RmsNormBackwardKernel::new(64, 64, 1e-6);
    }
}
