//! Softmax Backward Kernel
//!
//! Backward (gradient) kernel for softmax activation.
//!
//! ## Mathematical Specification
//!
//! Forward: `y_i = exp(x_i - max) / sum(exp(x_j - max))`
//!
//! Backward: `∂L/∂x_i = y_i · (∂L/∂y_i - Σⱼ ∂L/∂y_j · y_j)`
//!
//! The key insight is that we need to compute `dot(grad_y, y)` for each row,
//! then use it to compute the gradient for each element.
//!
//! ## Implementation
//!
//! Uses warp shuffle reductions for rows that fit within a warp (≤32 elements).
//! For larger rows, this kernel should be called with row_size ≤ 32.
//!
//! ## Falsifiable Prediction (P-SOFT-BACK-001)
//!
//! Softmax backward matches finite-difference within ε < 1e-5.

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Softmax Backward Kernel (warp-parallel, one row per warp)
///
/// Computes gradients for softmax using warp shuffle reductions.
///
/// # Parameters
/// - `output_ptr`: Softmax forward output (y)
/// - `grad_output_ptr`: Gradient from upstream (∂L/∂y)
/// - `grad_input_ptr`: Output gradient (∂L/∂x)
/// - `num_rows`: Number of rows
/// - `row_size`: Elements per row (must be ≤ 32 for warp-level reduction)
#[derive(Debug, Clone)]
pub struct SoftmaxBackwardKernel {
    /// Number of rows
    pub num_rows: u32,
    /// Elements per row (max 32 for warp reduction)
    pub row_size: u32,
}

impl SoftmaxBackwardKernel {
    /// Create a new Softmax backward kernel
    ///
    /// # Arguments
    /// - `num_rows`: Number of rows to process
    /// - `row_size`: Elements per row (must be ≤ 32)
    ///
    /// # Panics
    /// Panics if `row_size` > 32
    #[must_use]
    pub fn new(num_rows: u32, row_size: u32) -> Self {
        assert!(row_size <= 32, "row_size must be ≤ 32 for warp reduction");
        Self { num_rows, row_size }
    }
}

impl Kernel for SoftmaxBackwardKernel {
    fn name(&self) -> &str {
        "softmax_backward"
    }

    fn build_ptx(&self) -> PtxKernel {
        let row_size = self.row_size;

        PtxKernel::new("softmax_backward")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U64, "grad_output_ptr")
            .param(PtxType::U64, "grad_input_ptr")
            .param(PtxType::U32, "num_rows")
            .param(PtxType::U32, "row_size")
            .build(move |ctx| {
                // Thread indexing: one warp (32 threads) per row
                // Global warp ID = (blockIdx.x * blockDim.x + threadIdx.x) / 32
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let global_tid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Lane within warp (0-31): lane = global_tid & 31
                let lane = ctx.and_u32_imm(global_tid, 31);

                // Warp ID (row index): warp_id = global_tid >> 5
                let warp_id = ctx.shr_u32_imm(global_tid, 5);

                // Load parameters
                let num_rows = ctx.load_param_u32("num_rows");
                let row_size_param = ctx.load_param_u32("row_size");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let grad_output_ptr = ctx.load_param_u64("grad_output_ptr");
                let grad_input_ptr = ctx.load_param_u64("grad_input_ptr");

                // Bounds check: warp_id < num_rows
                let valid_row = ctx.setp_lt_u32(warp_id, num_rows);
                ctx.branch_if_not(valid_row, "exit");

                // Bounds check: lane < row_size
                let valid_lane = ctx.setp_lt_u32(lane, row_size_param);

                // Calculate row base offset: row_offset = warp_id * row_size * 4
                let row_elem_offset = ctx.mul_lo_u32(warp_id, row_size_param);
                let row_byte_offset = ctx.mul_wide_u32(row_elem_offset, 4);
                let output_row_base = ctx.add_u64(output_ptr, row_byte_offset);
                let grad_out_row_base = ctx.add_u64(grad_output_ptr, row_byte_offset);
                let grad_in_row_base = ctx.add_u64(grad_input_ptr, row_byte_offset);

                // Calculate element address within row: lane * 4
                let lane_offset = ctx.mul_wide_u32(lane, 4);
                let output_addr = ctx.add_u64(output_row_base, lane_offset);
                let grad_out_addr = ctx.add_u64(grad_out_row_base, lane_offset);
                let grad_in_addr = ctx.add_u64(grad_in_row_base, lane_offset);

                // Load y[i] and grad_y[i] with bounds checking
                let y_i = ctx.ld_global_f32_predicated(output_addr, valid_lane, 0.0);
                let grad_y_i = ctx.ld_global_f32_predicated(grad_out_addr, valid_lane, 0.0);

                // Compute local contribution: y_i * grad_y_i
                let local_product = ctx.mul_f32(y_i, grad_y_i);

                // Warp-level reduction to compute dot(grad_y, y) = Σ(y_i * grad_y_i)
                // Using butterfly reduction pattern with shfl.down
                let mut sum = local_product;

                // Full warp mask for shuffle operations
                let warp_mask = 0xFFFF_FFFFu32;

                // Unroll reduction: 16, 8, 4, 2, 1
                for offset in [16u32, 8, 4, 2, 1] {
                    if offset < row_size {
                        let shuffled = ctx.shfl_down_f32(sum, offset, warp_mask);
                        sum = ctx.add_f32(sum, shuffled);
                    }
                }

                // Broadcast sum to all lanes using shfl.idx from lane 0
                let dot_product = ctx.shfl_idx_f32(sum, 0, warp_mask);

                // Compute gradient: grad_x_i = y_i * (grad_y_i - dot_product)
                let diff = ctx.sub_f32(grad_y_i, dot_product);
                let grad_x_i = ctx.mul_f32(y_i, diff);

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
    fn test_softmax_backward_name() {
        let kernel = SoftmaxBackwardKernel::new(64, 32);
        assert_eq!(kernel.name(), "softmax_backward");
    }

    #[test]
    fn test_softmax_backward_ptx_generation() {
        let kernel = SoftmaxBackwardKernel::new(64, 32);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry softmax_backward"));
        // Verify parameters
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u64 grad_output_ptr"));
        assert!(ptx.contains(".param .u64 grad_input_ptr"));
        assert!(ptx.contains(".param .u32 num_rows"));
        // Verify warp shuffle for reduction
        assert!(ptx.contains("shfl.sync.down"));
        // Verify broadcast
        assert!(ptx.contains("shfl.sync.idx"));
    }

    #[test]
    fn test_softmax_backward_small_row() {
        // Test with smaller row size
        let kernel = SoftmaxBackwardKernel::new(128, 16);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry softmax_backward"));
        assert!(ptx.contains("shfl.sync"));
    }

    #[test]
    fn test_softmax_backward_barrier_safety() {
        let kernel = SoftmaxBackwardKernel::new(64, 32);
        let result = kernel.analyze_barrier_safety();
        assert!(result.is_safe, "Softmax backward should be barrier-safe: {:?}", result.violations);
    }

    #[test]
    #[should_panic(expected = "row_size must be ≤ 32")]
    fn test_softmax_backward_row_size_limit() {
        let _ = SoftmaxBackwardKernel::new(64, 64);
    }
}
