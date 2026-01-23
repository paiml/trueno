//! Fused Transformer Kernels (PMAT-PERF-009)
//!
//! Implements fused operations for transformer inference to reduce kernel launch overhead:
//! - FusedQKVKernel: Q/K/V projection in single kernel (3x reduction in launches)
//! - FusedGateUpKernel: Gate+Up FFN with SwiGLU activation (2x reduction)
//!
//! # Five-Whys Root Cause (PMAT-PERF-009)
//!
//! ```text
//! Why 1: Why is decode throughput 131 tok/s vs 400 tok/s target?
//! → 280+ kernel launches per token (10+ per layer × 28 layers)
//!
//! Why 2: Why so many kernel launches?
//! → Q, K, V computed as 3 separate GEMV operations
//!
//! Why 3: Why separate operations?
//! → Original implementation didn't consider launch overhead
//!
//! Why 4: Why does launch overhead matter?
//! → GPU kernel launch: ~5-10µs, 280 launches = 1.4-2.8ms overhead/token
//!
//! Why 5: ROOT CAUSE
//! → Kernel launch overhead (2.8ms) exceeds compute time for small batch decode
//! → FIX: Fuse Q/K/V into single kernel, reducing launches by 2/3
//! ```

// Allow similar names for related variables (wq/wk/wv, shfl_q/shfl_k/shfl_v, etc.)
// Allow unused_assignments/unused_mut because PTX branch semantics aren't tracked by Rust
#![allow(clippy::similar_names, unused_assignments, unused_mut)]

use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

use super::Kernel;

/// Fused Q/K/V projection kernel
///
/// Computes Q, K, V projections in a single kernel launch:
/// - Q = x @ W_q^T (hidden_size → hidden_size)
/// - K = x @ W_k^T (hidden_size → kv_dim)
/// - V = x @ W_v^T (hidden_size → kv_dim)
///
/// Grid: (max(hidden_size, kv_dim), 1, 1)
/// Block: (32, 1, 1) - one warp per output element
#[derive(Debug, Clone)]
pub struct FusedQKVKernel {
    /// Hidden dimension size
    pub hidden_size: usize,
    /// KV dimension (may differ for GQA)
    pub kv_dim: usize,
}

impl FusedQKVKernel {
    /// Create a new fused QKV kernel.
    pub fn new(hidden_size: usize, kv_dim: usize) -> Self {
        Self { hidden_size, kv_dim }
    }
}

impl Kernel for FusedQKVKernel {
    fn name(&self) -> &str {
        "fused_qkv_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden = self.hidden_size as u32;
        let kv = self.kv_dim as u32;

        PtxKernel::new("fused_qkv_gemv")
            // Parameters: x, W_q, W_k, W_v, out_q, out_k, out_v
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "wq_ptr")
            .param(PtxType::U64, "wk_ptr")
            .param(PtxType::U64, "wv_ptr")
            .param(PtxType::U64, "out_q_ptr")
            .param(PtxType::U64, "out_k_ptr")
            .param(PtxType::U64, "out_v_ptr")
            .build(move |ctx| {
                // Get thread/block IDs
                let tid = ctx.special_reg(PtxReg::TidX);
                let row = ctx.special_reg(PtxReg::CtaIdX);

                // lane = tid & 31
                let lane = ctx.and_u32_imm(tid, 31);

                // Load constants
                let hidden_size = ctx.mov_u32_imm(hidden);
                let kv_dim_val = ctx.mov_u32_imm(kv);

                // Initialize accumulators
                let mut acc_q = ctx.mov_f32_imm(0.0);
                let mut acc_k = ctx.mov_f32_imm(0.0);
                let mut acc_v = ctx.mov_f32_imm(0.0);

                // Load base pointers
                let x_ptr = ctx.load_param_u64("x_ptr");
                let wq_ptr = ctx.load_param_u64("wq_ptr");
                let wk_ptr = ctx.load_param_u64("wk_ptr");
                let wv_ptr = ctx.load_param_u64("wv_ptr");

                // k = lane (start offset)
                let mut k = lane;

                // Main loop: stride by 32 (warp size)
                ctx.label("loop_start");
                let pred_exit = ctx.setp_ge_u32(k, hidden_size);
                ctx.branch_if(pred_exit, "loop_end");

                // Load x[k]
                let offset_k = ctx.mul_wide_u32(k, 4);
                let x_addr = ctx.add_u64(x_ptr, offset_k);
                let x_val = ctx.ld_global_f32(x_addr);

                // Compute weight offset: row * hidden + k
                let row_offset = ctx.mul_u32_reg(row, hidden_size);
                let weight_idx = ctx.add_u32_reg(row_offset, k);
                let weight_byte_offset = ctx.mul_wide_u32(weight_idx, 4);

                // Load and accumulate Q
                let wq_addr = ctx.add_u64(wq_ptr, weight_byte_offset);
                let wq_val = ctx.ld_global_f32(wq_addr);
                acc_q = ctx.fma_f32(x_val, wq_val, acc_q);

                // Load and accumulate K
                let wk_addr = ctx.add_u64(wk_ptr, weight_byte_offset);
                let wk_val = ctx.ld_global_f32(wk_addr);
                acc_k = ctx.fma_f32(x_val, wk_val, acc_k);

                // Load and accumulate V
                let wv_addr = ctx.add_u64(wv_ptr, weight_byte_offset);
                let wv_val = ctx.ld_global_f32(wv_addr);
                acc_v = ctx.fma_f32(x_val, wv_val, acc_v);

                // k += 32 (must be in-place to update loop variable)
                ctx.add_u32_inplace(k, 32);
                ctx.branch("loop_start");

                ctx.label("loop_end");

                // Warp reduction for all accumulators
                // acc_q
                let shfl_q_16 = ctx.shfl_down_f32(acc_q, 16, 0xFFFF_FFFF);
                acc_q = ctx.add_f32(acc_q, shfl_q_16);
                let shfl_q_8 = ctx.shfl_down_f32(acc_q, 8, 0xFFFF_FFFF);
                acc_q = ctx.add_f32(acc_q, shfl_q_8);
                let shfl_q_4 = ctx.shfl_down_f32(acc_q, 4, 0xFFFF_FFFF);
                acc_q = ctx.add_f32(acc_q, shfl_q_4);
                let shfl_q_2 = ctx.shfl_down_f32(acc_q, 2, 0xFFFF_FFFF);
                acc_q = ctx.add_f32(acc_q, shfl_q_2);
                let shfl_q_1 = ctx.shfl_down_f32(acc_q, 1, 0xFFFF_FFFF);
                acc_q = ctx.add_f32(acc_q, shfl_q_1);

                // acc_k
                let shfl_k_16 = ctx.shfl_down_f32(acc_k, 16, 0xFFFF_FFFF);
                acc_k = ctx.add_f32(acc_k, shfl_k_16);
                let shfl_k_8 = ctx.shfl_down_f32(acc_k, 8, 0xFFFF_FFFF);
                acc_k = ctx.add_f32(acc_k, shfl_k_8);
                let shfl_k_4 = ctx.shfl_down_f32(acc_k, 4, 0xFFFF_FFFF);
                acc_k = ctx.add_f32(acc_k, shfl_k_4);
                let shfl_k_2 = ctx.shfl_down_f32(acc_k, 2, 0xFFFF_FFFF);
                acc_k = ctx.add_f32(acc_k, shfl_k_2);
                let shfl_k_1 = ctx.shfl_down_f32(acc_k, 1, 0xFFFF_FFFF);
                acc_k = ctx.add_f32(acc_k, shfl_k_1);

                // acc_v
                let shfl_v_16 = ctx.shfl_down_f32(acc_v, 16, 0xFFFF_FFFF);
                acc_v = ctx.add_f32(acc_v, shfl_v_16);
                let shfl_v_8 = ctx.shfl_down_f32(acc_v, 8, 0xFFFF_FFFF);
                acc_v = ctx.add_f32(acc_v, shfl_v_8);
                let shfl_v_4 = ctx.shfl_down_f32(acc_v, 4, 0xFFFF_FFFF);
                acc_v = ctx.add_f32(acc_v, shfl_v_4);
                let shfl_v_2 = ctx.shfl_down_f32(acc_v, 2, 0xFFFF_FFFF);
                acc_v = ctx.add_f32(acc_v, shfl_v_2);
                let shfl_v_1 = ctx.shfl_down_f32(acc_v, 1, 0xFFFF_FFFF);
                acc_v = ctx.add_f32(acc_v, shfl_v_1);

                // Lane 0 stores results (skip if lane != 0)
                let zero = ctx.mov_u32_imm(0);
                let is_lane0 = ctx.setp_eq_u32(lane, zero);
                ctx.branch_if_not(is_lane0, "done");

                // Store outputs
                let out_q_ptr = ctx.load_param_u64("out_q_ptr");
                let out_k_ptr = ctx.load_param_u64("out_k_ptr");
                let out_v_ptr = ctx.load_param_u64("out_v_ptr");

                let row_byte_offset = ctx.mul_wide_u32(row, 4);

                // Store Q (always for row < hidden_size, which is ensured by grid size)
                let out_q_addr = ctx.add_u64(out_q_ptr, row_byte_offset);
                ctx.st_global_f32(out_q_addr, acc_q);

                // Store K/V only if row < kv_dim
                let pred_kv = ctx.setp_lt_u32(row, kv_dim_val);
                ctx.branch_if_not(pred_kv, "done");

                let out_k_addr = ctx.add_u64(out_k_ptr, row_byte_offset);
                ctx.st_global_f32(out_k_addr, acc_k);

                let out_v_addr = ctx.add_u64(out_v_ptr, row_byte_offset);
                ctx.st_global_f32(out_v_addr, acc_v);

                ctx.label("done");
                ctx.ret();
            })
    }
}

/// Fused Gate+Up FFN kernel with SwiGLU activation
///
/// Computes: output = SiLU(gate) * up
/// Where:
/// - gate = x @ W_gate^T
/// - up = x @ W_up^T
/// - SiLU(x) = x * sigmoid(x)
///
/// Grid: (intermediate_size, 1, 1)
/// Block: (32, 1, 1) - one warp per output element
#[derive(Debug, Clone)]
pub struct FusedGateUpKernel {
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Intermediate FFN dimension
    pub intermediate_size: usize,
}

impl FusedGateUpKernel {
    /// Create a new fused gate+up kernel.
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
        }
    }
}

impl Kernel for FusedGateUpKernel {
    fn name(&self) -> &str {
        "fused_gate_up_swiglu"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden = self.hidden_size as u32;

        PtxKernel::new("fused_gate_up_swiglu")
            // Parameters: x, W_gate, W_up, output
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "wg_ptr")
            .param(PtxType::U64, "wu_ptr")
            .param(PtxType::U64, "out_ptr")
            .build(move |ctx| {
                // Get thread/block IDs
                let tid = ctx.special_reg(PtxReg::TidX);
                let row = ctx.special_reg(PtxReg::CtaIdX);
                let lane = ctx.and_u32_imm(tid, 31);
                let hidden_size = ctx.mov_u32_imm(hidden);

                // Initialize accumulators
                let mut acc_gate = ctx.mov_f32_imm(0.0);
                let mut acc_up = ctx.mov_f32_imm(0.0);

                // Load base pointers
                let x_ptr = ctx.load_param_u64("x_ptr");
                let wg_ptr = ctx.load_param_u64("wg_ptr");
                let wu_ptr = ctx.load_param_u64("wu_ptr");

                // k = lane
                let mut k = lane;

                // Main loop
                ctx.label("loop_start");
                let pred_exit = ctx.setp_ge_u32(k, hidden_size);
                ctx.branch_if(pred_exit, "loop_end");

                // Load x[k]
                let offset_k = ctx.mul_wide_u32(k, 4);
                let x_addr = ctx.add_u64(x_ptr, offset_k);
                let x_val = ctx.ld_global_f32(x_addr);

                // Weight offset: row * hidden + k
                let row_offset = ctx.mul_u32_reg(row, hidden_size);
                let weight_idx = ctx.add_u32_reg(row_offset, k);
                let weight_byte_offset = ctx.mul_wide_u32(weight_idx, 4);

                // Load W_gate and accumulate
                let wg_addr = ctx.add_u64(wg_ptr, weight_byte_offset);
                let wg_val = ctx.ld_global_f32(wg_addr);
                acc_gate = ctx.fma_f32(x_val, wg_val, acc_gate);

                // Load W_up and accumulate
                let wu_addr = ctx.add_u64(wu_ptr, weight_byte_offset);
                let wu_val = ctx.ld_global_f32(wu_addr);
                acc_up = ctx.fma_f32(x_val, wu_val, acc_up);

                // k += 32 (must be in-place to update loop variable)
                ctx.add_u32_inplace(k, 32);
                ctx.branch("loop_start");

                ctx.label("loop_end");

                // Warp reduction for acc_gate
                let shfl_g_16 = ctx.shfl_down_f32(acc_gate, 16, 0xFFFF_FFFF);
                acc_gate = ctx.add_f32(acc_gate, shfl_g_16);
                let shfl_g_8 = ctx.shfl_down_f32(acc_gate, 8, 0xFFFF_FFFF);
                acc_gate = ctx.add_f32(acc_gate, shfl_g_8);
                let shfl_g_4 = ctx.shfl_down_f32(acc_gate, 4, 0xFFFF_FFFF);
                acc_gate = ctx.add_f32(acc_gate, shfl_g_4);
                let shfl_g_2 = ctx.shfl_down_f32(acc_gate, 2, 0xFFFF_FFFF);
                acc_gate = ctx.add_f32(acc_gate, shfl_g_2);
                let shfl_g_1 = ctx.shfl_down_f32(acc_gate, 1, 0xFFFF_FFFF);
                acc_gate = ctx.add_f32(acc_gate, shfl_g_1);

                // Warp reduction for acc_up
                let shfl_u_16 = ctx.shfl_down_f32(acc_up, 16, 0xFFFF_FFFF);
                acc_up = ctx.add_f32(acc_up, shfl_u_16);
                let shfl_u_8 = ctx.shfl_down_f32(acc_up, 8, 0xFFFF_FFFF);
                acc_up = ctx.add_f32(acc_up, shfl_u_8);
                let shfl_u_4 = ctx.shfl_down_f32(acc_up, 4, 0xFFFF_FFFF);
                acc_up = ctx.add_f32(acc_up, shfl_u_4);
                let shfl_u_2 = ctx.shfl_down_f32(acc_up, 2, 0xFFFF_FFFF);
                acc_up = ctx.add_f32(acc_up, shfl_u_2);
                let shfl_u_1 = ctx.shfl_down_f32(acc_up, 1, 0xFFFF_FFFF);
                acc_up = ctx.add_f32(acc_up, shfl_u_1);

                // Lane 0 computes SiLU and stores (skip if lane != 0)
                let zero = ctx.mov_u32_imm(0);
                let is_lane0 = ctx.setp_eq_u32(lane, zero);
                ctx.branch_if_not(is_lane0, "done");

                // SiLU(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
                // exp(-x) = 2^(-x * log2(e))
                let neg_gate = ctx.neg_f32(acc_gate);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scaled = ctx.mul_f32(neg_gate, log2_e);
                let exp_val = ctx.ex2_f32(scaled);
                let one = ctx.mov_f32_imm(1.0);
                let one_plus_exp = ctx.add_f32(one, exp_val);
                let sigmoid = ctx.rcp_f32(one_plus_exp);
                let silu = ctx.mul_f32(acc_gate, sigmoid);

                // output = SiLU(gate) * up
                let output = ctx.mul_f32(silu, acc_up);

                // Store output[row]
                let out_ptr = ctx.load_param_u64("out_ptr");
                let row_byte_offset = ctx.mul_wide_u32(row, 4);
                let out_addr = ctx.add_u64(out_ptr, row_byte_offset);
                ctx.st_global_f32(out_addr, output);

                ctx.label("done");
                ctx.ret();
            })
    }
}

/// Fused GEMM + Bias + GELU kernel (WAPR-PERF-007)
///
/// Computes: output = GELU(A @ B + bias) in a single kernel launch.
/// Eliminates 3 kernel launches (GEMM, Bias, GELU) into 1.
///
/// For FFN in Whisper encoder:
/// - First linear: [seq, hidden] @ [hidden, intermediate] + bias → GELU → [seq, intermediate]
/// - This kernel handles that in ONE launch instead of THREE.
///
/// Grid: (N / block_size, M / block_size, 1)
/// Block: (block_size, block_size, 1) - typically 16x16
///
/// # Citations
///
/// - [Dao2022] FlashAttention: Fast and Memory-Efficient Exact Attention
/// - [Kwon2023] PagedAttention: Efficient LLM Serving
#[derive(Debug, Clone)]
pub struct FusedGemmBiasGeluKernel {
    /// M: Number of rows in A and C
    pub m: u32,
    /// N: Number of columns in B and C
    pub n: u32,
    /// K: Shared dimension (columns of A, rows of B)
    pub k: u32,
}

impl FusedGemmBiasGeluKernel {
    /// Create a new fused GEMM+Bias+GELU kernel
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k }
    }
}

impl Kernel for FusedGemmBiasGeluKernel {
    fn name(&self) -> &str {
        "fused_gemm_bias_gelu"
    }

    fn build_ptx(&self) -> PtxKernel {
        let k_val = self.k;
        let n_val = self.n;

        PtxKernel::new("fused_gemm_bias_gelu")
            .param(PtxType::U64, "a_ptr")      // Input matrix A [M, K]
            .param(PtxType::U64, "b_ptr")      // Weight matrix B [K, N]
            .param(PtxType::U64, "bias_ptr")   // Bias vector [N]
            .param(PtxType::U64, "c_ptr")      // Output matrix C [M, N]
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .build(move |ctx| {
                // Calculate row and column from thread/block IDs
                // row = ctaid.y * ntid.y + tid.y
                // col = ctaid.x * ntid.x + tid.x
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);
                let ntid_y = ctx.special_reg(PtxReg::NtidY);
                let tid_y = ctx.special_reg(PtxReg::TidY);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ntid_x = ctx.special_reg(PtxReg::NtidX);
                let tid_x = ctx.special_reg(PtxReg::TidX);

                let row = ctx.mad_lo_u32(ctaid_y, ntid_y, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);

                // Load params
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                // Bounds check
                let pred_m = ctx.setp_ge_u32(row, m_param);
                ctx.branch_if(pred_m, "exit");
                let pred_n = ctx.setp_ge_u32(col, n_param);
                ctx.branch_if(pred_n, "exit");

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let bias_ptr = ctx.load_param_u64("bias_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Calculate base offset for A[row, 0]
                let row_offset = ctx.mul_wide_u32(row, k_val * 4);
                let a_row_ptr = ctx.add_u64(a_ptr, row_offset);

                // Calculate base offset for B[0, col]
                let col_offset = ctx.mul_wide_u32(col, 4);
                let b_col_base = ctx.add_u64(b_ptr, col_offset);

                // Loop over K dimension
                let i = ctx.mov_u32_imm(0);

                ctx.label("loop_k");
                let pred_k = ctx.setp_ge_u32(i, k_param);
                ctx.branch_if(pred_k, "loop_end");

                // Load A[row, i]
                let i_offset = ctx.mul_wide_u32(i, 4);
                let a_addr = ctx.add_u64(a_row_ptr, i_offset);
                let a_val = ctx.ld_global_f32(a_addr);

                // Load B[i, col]
                let b_row_offset = ctx.mul_wide_u32(i, n_val * 4);
                let b_addr = ctx.add_u64(b_col_base, b_row_offset);
                let b_val = ctx.ld_global_f32(b_addr);

                // acc += a_val * b_val
                ctx.fma_f32_inplace(acc, a_val, b_val);

                // i++
                ctx.add_u32_inplace(i, 1);
                ctx.branch("loop_k");

                ctx.label("loop_end");

                // Load and add bias[col]
                let bias_offset = ctx.mul_wide_u32(col, 4);
                let bias_addr = ctx.add_u64(bias_ptr, bias_offset);
                let bias_val = ctx.ld_global_f32(bias_addr);
                let acc_biased = ctx.add_f32(acc, bias_val);

                // ============================================
                // GELU approximation (fused in same kernel)
                // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                // ============================================
                let x = acc_biased;

                // Constants
                let sqrt_2_pi = ctx.mov_f32_imm(0.797_884_6);  // sqrt(2/π)
                let c = ctx.mov_f32_imm(0.044_715);
                let half = ctx.mov_f32_imm(0.5);
                let one = ctx.mov_f32_imm(1.0);
                let two = ctx.mov_f32_imm(2.0);
                let zero = ctx.mov_f32_imm(0.0);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);

                // x³
                let x2 = ctx.mul_f32(x, x);
                let x3 = ctx.mul_f32(x2, x);

                // 0.044715 * x³
                let cx3 = ctx.mul_f32(c, x3);

                // x + 0.044715 * x³
                let inner = ctx.add_f32(x, cx3);

                // sqrt(2/π) * (x + 0.044715 * x³)
                let scaled = ctx.mul_f32(sqrt_2_pi, inner);

                // tanh approximation: tanh(x) = 2*sigmoid(2x) - 1
                let two_x = ctx.mul_f32(two, scaled);
                let neg_two_x = ctx.sub_f32(zero, two_x);
                let scaled_exp = ctx.mul_f32(neg_two_x, log2_e);
                let exp_neg = ctx.ex2_f32(scaled_exp);
                let denom = ctx.add_f32(one, exp_neg);
                let sigmoid = ctx.div_f32(one, denom);
                let two_sigmoid = ctx.mul_f32(two, sigmoid);
                let tanh = ctx.sub_f32(two_sigmoid, one);

                // 1 + tanh(...)
                let one_plus_tanh = ctx.add_f32(one, tanh);

                // 0.5 * x
                let half_x = ctx.mul_f32(half, x);

                // result = 0.5 * x * (1 + tanh(...))
                let result = ctx.mul_f32(half_x, one_plus_tanh);

                // Store result: C[row, col]
                let c_row_offset = ctx.mul_wide_u32(row, n_val * 4);
                let c_row_ptr = ctx.add_u64(c_ptr, c_row_offset);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_addr = ctx.add_u64(c_row_ptr, c_col_offset);
                ctx.st_global_f32(c_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fused_gemm_bias_gelu_kernel_builds() {
        let kernel = FusedGemmBiasGeluKernel::new(1500, 1536, 384);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("fused_gemm_bias_gelu"));
        assert!(ptx.contains(".entry"));
        // Verify GELU constants are present (hex format: 0F{bits:08X})
        // sqrt(2/π) ≈ 0.7978846 -> 0F3F4C422A
        // 0.044715 -> 0F3D372713
        assert!(ptx.contains("0F3F4C422A"), "Missing sqrt(2/π) constant");
        assert!(ptx.contains("0F3D372713"), "Missing 0.044715 constant");
    }

    #[test]
    fn test_fused_qkv_kernel_builds() {
        let kernel = FusedQKVKernel::new(3584, 512);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("fused_qkv_gemv"));
        assert!(ptx.contains(".entry"));
    }

    #[test]
    fn test_fused_gate_up_kernel_builds() {
        let kernel = FusedGateUpKernel::new(3584, 18944);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("fused_gate_up_swiglu"));
        assert!(ptx.contains(".entry"));
    }

    #[test]
    fn test_fused_qkv_kernel_name() {
        let kernel = FusedQKVKernel::new(1024, 256);
        assert_eq!(kernel.name(), "fused_qkv_gemv");
    }

    #[test]
    fn test_fused_gate_up_kernel_name() {
        let kernel = FusedGateUpKernel::new(1024, 4096);
        assert_eq!(kernel.name(), "fused_gate_up_swiglu");
    }

    #[test]
    fn test_fused_qkv_kernel_clone() {
        let kernel = FusedQKVKernel::new(1024, 256);
        let cloned = kernel.clone();
        assert_eq!(cloned.hidden_size, kernel.hidden_size);
        assert_eq!(cloned.kv_dim, kernel.kv_dim);
    }

    #[test]
    fn test_fused_gate_up_kernel_clone() {
        let kernel = FusedGateUpKernel::new(1024, 4096);
        let cloned = kernel.clone();
        assert_eq!(cloned.hidden_size, kernel.hidden_size);
        assert_eq!(cloned.intermediate_size, kernel.intermediate_size);
    }

    #[test]
    fn test_fused_qkv_kernel_debug() {
        let kernel = FusedQKVKernel::new(1024, 256);
        let debug = format!("{:?}", kernel);
        assert!(debug.contains("FusedQKVKernel"));
        assert!(debug.contains("1024"));
    }

    #[test]
    fn test_fused_gate_up_kernel_debug() {
        let kernel = FusedGateUpKernel::new(1024, 4096);
        let debug = format!("{:?}", kernel);
        assert!(debug.contains("FusedGateUpKernel"));
        assert!(debug.contains("4096"));
    }
}
