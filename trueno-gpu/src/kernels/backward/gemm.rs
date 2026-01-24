//! GEMM Backward Kernel
//!
//! Backward (gradient) kernel for matrix multiplication.
//!
//! ## Mathematical Specification
//!
//! Forward: `C = A @ B` where A is (M, K) and B is (K, N)
//!
//! Backward:
//! - `∂L/∂A = ∂L/∂C @ B^T` (shape: M×N @ N×K = M×K)
//! - `∂L/∂B = A^T @ ∂L/∂C` (shape: K×M @ M×N = K×N)
//!
//! ## Implementation
//!
//! This kernel computes gradient w.r.t. input A (grad_a = grad_c @ B^T).
//! For grad_b, transpose the roles or use a separate kernel call.
//!
//! Uses naive approach with one thread per output element.
//! For production use, consider tiled implementations.
//!
//! ## Falsifiable Prediction (P-GEMM-BACK-001)
//!
//! GEMM backward matches finite-difference within ε < 1e-4.

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// GEMM Backward Kernel (gradient w.r.t. A)
///
/// Computes `grad_a = grad_c @ B^T` for GEMM C = A @ B.
///
/// # Parameters
/// - `grad_c_ptr`: Gradient from upstream (∂L/∂C), shape M×N
/// - `b_ptr`: Input B from forward pass, shape K×N
/// - `grad_a_ptr`: Output gradient for A (∂L/∂A), shape M×K
/// - `m`: Number of rows in A and grad_c
/// - `n`: Number of columns in B and grad_c
/// - `k`: Number of columns in A / rows in B
///
/// # Algorithm
/// For computing grad_a[i,j]:
/// ```text
/// grad_a[i,j] = Σ_l grad_c[i,l] * B[j,l]  // B^T[l,j] = B[j,l]
/// ```
#[derive(Debug, Clone)]
pub struct GemmBackwardAKernel {
    /// M dimension (rows of A)
    pub m: u32,
    /// N dimension (cols of B)
    pub n: u32,
    /// K dimension (cols of A / rows of B)
    pub k: u32,
}

impl GemmBackwardAKernel {
    /// Create a new GEMM backward kernel for gradient w.r.t. A
    #[must_use]
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }
}

impl Kernel for GemmBackwardAKernel {
    fn name(&self) -> &str {
        "gemm_backward_a"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("gemm_backward_a")
            .param(PtxType::U64, "grad_c_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "grad_a_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .build(|ctx| {
                // Thread coordinates: (row, col) in output grad_a
                let tid_x = ctx.special_reg(PtxReg::TidX);
                let tid_y = ctx.special_reg(PtxReg::TidY);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);
                let ntid_x = ctx.special_reg(PtxReg::NtidX);
                let ntid_y = ctx.special_reg(PtxReg::NtidY);

                // Global position (row, col) in grad_a[m, k]
                let row = ctx.mad_lo_u32(ctaid_y, ntid_y, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);

                // Load parameters
                let m = ctx.load_param_u32("m");
                let n = ctx.load_param_u32("n");
                let k = ctx.load_param_u32("k");
                let grad_c_ptr = ctx.load_param_u64("grad_c_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let grad_a_ptr = ctx.load_param_u64("grad_a_ptr");

                // Bounds check: row < m && col < k
                let valid_row = ctx.setp_lt_u32(row, m);
                ctx.branch_if_not(valid_row, "exit");
                let valid_col = ctx.setp_lt_u32(col, k);
                ctx.branch_if_not(valid_col, "exit");

                // Accumulator for dot product
                let acc = ctx.mov_f32_imm(0.0);

                // Constants
                let four = ctx.mov_u32_imm(4);

                // Loop counter
                let i = ctx.mov_u32_imm(0);

                ctx.label("loop_start");
                // Check i < n
                let loop_cond = ctx.setp_lt_u32(i, n);
                ctx.branch_if_not(loop_cond, "loop_end");

                // Load grad_c[row, i]: offset = (row * n + i) * 4
                let grad_c_row_offset = ctx.mul_lo_u32(row, n);
                let grad_c_elem_idx = ctx.add_u32_reg(grad_c_row_offset, i);
                let grad_c_byte_offset = ctx.mul_wide_u32_reg(grad_c_elem_idx, four);
                let grad_c_addr = ctx.add_u64(grad_c_ptr, grad_c_byte_offset);
                let grad_c_val = ctx.ld_global_f32(grad_c_addr);

                // Load B[col, i]: offset = (col * n + i) * 4
                // B is K×N, so B[col, i] where col indexes K, i indexes N
                let b_row_offset = ctx.mul_lo_u32(col, n);
                let b_elem_idx = ctx.add_u32_reg(b_row_offset, i);
                let b_byte_offset = ctx.mul_wide_u32_reg(b_elem_idx, four);
                let b_addr = ctx.add_u64(b_ptr, b_byte_offset);
                let b_val = ctx.ld_global_f32(b_addr);

                // Accumulate: acc += grad_c[row, i] * B[col, i]
                let prod = ctx.mul_f32(grad_c_val, b_val);
                ctx.add_f32_inplace(acc, prod);

                // Increment loop counter
                ctx.add_u32_inplace(i, 1);
                ctx.branch("loop_start");

                ctx.label("loop_end");

                // Store result: grad_a[row, col] = acc
                // offset = (row * k + col) * 4
                let grad_a_row_offset = ctx.mul_lo_u32(row, k);
                let grad_a_elem_idx = ctx.add_u32_reg(grad_a_row_offset, col);
                let grad_a_byte_offset = ctx.mul_wide_u32_reg(grad_a_elem_idx, four);
                let grad_a_addr = ctx.add_u64(grad_a_ptr, grad_a_byte_offset);
                ctx.st_global_f32(grad_a_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// GEMM Backward Kernel (gradient w.r.t. B)
///
/// Computes `grad_b = A^T @ grad_c` for GEMM C = A @ B.
///
/// # Parameters
/// - `a_ptr`: Input A from forward pass, shape M×K
/// - `grad_c_ptr`: Gradient from upstream (∂L/∂C), shape M×N
/// - `grad_b_ptr`: Output gradient for B (∂L/∂B), shape K×N
/// - `m`: Number of rows in A
/// - `n`: Number of columns in B
/// - `k`: Number of columns in A / rows in B
///
/// # Algorithm
/// For computing grad_b[i,j]:
/// ```text
/// grad_b[i,j] = Σ_l A[l,i] * grad_c[l,j]  // A^T[i,l] = A[l,i]
/// ```
#[derive(Debug, Clone)]
pub struct GemmBackwardBKernel {
    /// M dimension (rows of A)
    pub m: u32,
    /// N dimension (cols of B)
    pub n: u32,
    /// K dimension (cols of A / rows of B)
    pub k: u32,
}

impl GemmBackwardBKernel {
    /// Create a new GEMM backward kernel for gradient w.r.t. B
    #[must_use]
    pub const fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }
}

impl Kernel for GemmBackwardBKernel {
    fn name(&self) -> &str {
        "gemm_backward_b"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("gemm_backward_b")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "grad_c_ptr")
            .param(PtxType::U64, "grad_b_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .build(|ctx| {
                // Thread coordinates: (row, col) in output grad_b[k, n]
                let tid_x = ctx.special_reg(PtxReg::TidX);
                let tid_y = ctx.special_reg(PtxReg::TidY);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);
                let ntid_x = ctx.special_reg(PtxReg::NtidX);
                let ntid_y = ctx.special_reg(PtxReg::NtidY);

                // Global position (row, col) in grad_b[k, n]
                let row = ctx.mad_lo_u32(ctaid_y, ntid_y, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);

                // Load parameters
                let m = ctx.load_param_u32("m");
                let n = ctx.load_param_u32("n");
                let k = ctx.load_param_u32("k");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let grad_c_ptr = ctx.load_param_u64("grad_c_ptr");
                let grad_b_ptr = ctx.load_param_u64("grad_b_ptr");

                // Bounds check: row < k && col < n
                let valid_row = ctx.setp_lt_u32(row, k);
                ctx.branch_if_not(valid_row, "exit");
                let valid_col = ctx.setp_lt_u32(col, n);
                ctx.branch_if_not(valid_col, "exit");

                // Accumulator for dot product
                let acc = ctx.mov_f32_imm(0.0);

                // Constants
                let four = ctx.mov_u32_imm(4);

                // Loop counter
                let i = ctx.mov_u32_imm(0);

                ctx.label("loop_start");
                // Check i < m
                let loop_cond = ctx.setp_lt_u32(i, m);
                ctx.branch_if_not(loop_cond, "loop_end");

                // Load A[i, row]: offset = (i * k + row) * 4
                // A is M×K, A^T[row, i] = A[i, row]
                let a_row_offset = ctx.mul_lo_u32(i, k);
                let a_elem_idx = ctx.add_u32_reg(a_row_offset, row);
                let a_byte_offset = ctx.mul_wide_u32_reg(a_elem_idx, four);
                let a_addr = ctx.add_u64(a_ptr, a_byte_offset);
                let a_val = ctx.ld_global_f32(a_addr);

                // Load grad_c[i, col]: offset = (i * n + col) * 4
                let grad_c_row_offset = ctx.mul_lo_u32(i, n);
                let grad_c_elem_idx = ctx.add_u32_reg(grad_c_row_offset, col);
                let grad_c_byte_offset = ctx.mul_wide_u32_reg(grad_c_elem_idx, four);
                let grad_c_addr = ctx.add_u64(grad_c_ptr, grad_c_byte_offset);
                let grad_c_val = ctx.ld_global_f32(grad_c_addr);

                // Accumulate: acc += A[i, row] * grad_c[i, col]
                let prod = ctx.mul_f32(a_val, grad_c_val);
                ctx.add_f32_inplace(acc, prod);

                // Increment loop counter
                ctx.add_u32_inplace(i, 1);
                ctx.branch("loop_start");

                ctx.label("loop_end");

                // Store result: grad_b[row, col] = acc
                // offset = (row * n + col) * 4
                let grad_b_row_offset = ctx.mul_lo_u32(row, n);
                let grad_b_elem_idx = ctx.add_u32_reg(grad_b_row_offset, col);
                let grad_b_byte_offset = ctx.mul_wide_u32_reg(grad_b_elem_idx, four);
                let grad_b_addr = ctx.add_u64(grad_b_ptr, grad_b_byte_offset);
                ctx.st_global_f32(grad_b_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_backward_a_name() {
        let kernel = GemmBackwardAKernel::new(64, 64, 64);
        assert_eq!(kernel.name(), "gemm_backward_a");
    }

    #[test]
    fn test_gemm_backward_a_ptx_generation() {
        let kernel = GemmBackwardAKernel::new(64, 64, 64);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry gemm_backward_a"));
        // Verify parameters
        assert!(ptx.contains(".param .u64 grad_c_ptr"));
        assert!(ptx.contains(".param .u64 b_ptr"));
        assert!(ptx.contains(".param .u64 grad_a_ptr"));
        // Verify loop for accumulation
        assert!(ptx.contains("loop_start"));
        assert!(ptx.contains("loop_end"));
    }

    #[test]
    fn test_gemm_backward_a_barrier_safety() {
        let kernel = GemmBackwardAKernel::new(32, 32, 32);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "GEMM backward A should be barrier-safe: {:?}",
            result.violations
        );
    }

    #[test]
    fn test_gemm_backward_b_name() {
        let kernel = GemmBackwardBKernel::new(64, 64, 64);
        assert_eq!(kernel.name(), "gemm_backward_b");
    }

    #[test]
    fn test_gemm_backward_b_ptx_generation() {
        let kernel = GemmBackwardBKernel::new(64, 64, 64);
        let ptx = kernel.emit_ptx();

        // Verify entry point
        assert!(ptx.contains(".entry gemm_backward_b"));
        // Verify parameters
        assert!(ptx.contains(".param .u64 a_ptr"));
        assert!(ptx.contains(".param .u64 grad_c_ptr"));
        assert!(ptx.contains(".param .u64 grad_b_ptr"));
        // Verify loop for accumulation
        assert!(ptx.contains("loop_start"));
    }

    #[test]
    fn test_gemm_backward_b_barrier_safety() {
        let kernel = GemmBackwardBKernel::new(32, 32, 32);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "GEMM backward B should be barrier-safe: {:?}",
            result.violations
        );
    }
}
