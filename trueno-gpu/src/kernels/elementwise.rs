//! Element-wise GPU Kernels
//!
//! Simple element-wise operations for transformer forward passes.
//!
//! ## Available Kernels
//!
//! - **ResidualAddKernel**: Element-wise addition for residual connections
//!
//! # PAR-023: Async pipeline support
//!
//! These kernels are designed for GPU-resident execution without sync.

use super::Kernel;
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Residual Add Kernel: output = input1 + input2
///
/// Element-wise addition for residual connections in transformers.
/// Used for: x = x + attn(x) and x = x + ffn(x)
///
/// # Parameters
///
/// - `input1_ptr`: First input vector (u64 pointer)
/// - `input2_ptr`: Second input vector (u64 pointer)
/// - `output_ptr`: Output vector (u64 pointer, can alias input1 or input2)
/// - `n`: Number of elements (u32)
///
/// # Grid Configuration
///
/// - Block: 256 threads
/// - Grid: ceil(n / 256) blocks
#[derive(Debug, Clone)]
pub struct ResidualAddKernel {
    /// Number of elements
    pub n: u32,
}

impl ResidualAddKernel {
    /// Create a new residual add kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for ResidualAddKernel {
    fn name(&self) -> &str {
        "residual_add"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Simple element-wise addition
        // Each thread processes one element
        // Block: 256 threads, Grid: ceil(n/256)
        PtxKernel::new("residual_add")
            .param(PtxType::U64, "input1_ptr")
            .param(PtxType::U64, "input2_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load parameters
                let n = ctx.load_param_u32("n");
                let input1_ptr = ctx.load_param_u64("input1_ptr");
                let input2_ptr = ctx.load_param_u64("input2_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate address (gid * 4 bytes)
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let addr1 = ctx.add_u64(input1_ptr, offset);
                let addr2 = ctx.add_u64(input2_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load both values
                let val1 = ctx.ld_global_f32(addr1);
                let val2 = ctx.ld_global_f32(addr2);

                // Add
                let result = ctx.add_f32(val1, val2);

                // Store
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Fused Residual Add + RMSNorm Kernel
///
/// Combines residual addition and RMSNorm in a single kernel pass.
/// Reduces memory bandwidth by avoiding intermediate writes.
///
/// output = rmsnorm(input1 + input2, gamma, epsilon)
///
/// # PAR-023: This fused kernel eliminates one memory round-trip
#[derive(Debug, Clone)]
pub struct FusedResidualRmsNormKernel {
    /// Hidden dimension size
    pub hidden_size: u32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
}

impl FusedResidualRmsNormKernel {
    /// Create a new fused residual+rmsnorm kernel
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

impl Kernel for FusedResidualRmsNormKernel {
    fn name(&self) -> &str {
        "fused_residual_rmsnorm"
    }

    fn build_ptx(&self) -> PtxKernel {
        let hidden_size = self.hidden_size;
        let epsilon = self.epsilon;

        // Fused residual add + RMSNorm for single row using warp shuffle
        // Grid: 1 block, Block: 32 threads (one warp)
        PtxKernel::new("fused_residual_rmsnorm")
            .param(PtxType::U64, "residual_ptr") // Residual input
            .param(PtxType::U64, "input_ptr") // Input to add
            .param(PtxType::U64, "output_ptr") // Output (can alias residual)
            .param(PtxType::U64, "gamma_ptr") // Scale weights
            .shared_memory(0)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);

                // Load parameters
                let residual_ptr = ctx.load_param_u64("residual_ptr");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let gamma_ptr = ctx.load_param_u64("gamma_ptr");

                // Constants
                let hidden_u32 = ctx.mov_u32_imm(hidden_size);
                let four = ctx.mov_u32_imm(4);

                // ===== Phase 1: Add residual and accumulate sum of squares =====
                // Each thread processes elements: tid, tid+32, tid+64, ...
                let sq_sum = ctx.mov_f32_imm(0.0);
                let idx = ctx.mov_u32_imm(0);

                ctx.label("sum_loop");
                let loop_idx = ctx.add_u32_reg(idx, tid);
                let in_bounds = ctx.setp_lt_u32(loop_idx, hidden_u32);
                ctx.branch_if_not(in_bounds, "sum_loop_end");

                // Load residual[idx] and input[idx]
                let elem_offset = ctx.mul_wide_u32_reg(loop_idx, four);
                let res_addr = ctx.add_u64(residual_ptr, elem_offset);
                let inp_addr = ctx.add_u64(input_ptr, elem_offset);

                let res_val = ctx.ld_global_f32(res_addr);
                let inp_val = ctx.ld_global_f32(inp_addr);

                // sum_val = residual + input
                let sum_val = ctx.add_f32(res_val, inp_val);

                // sq_sum += sum_val * sum_val
                ctx.fma_f32_inplace(sq_sum, sum_val, sum_val);

                // Store intermediate sum for phase 2
                // Using output buffer as scratch (will be overwritten)
                let out_addr = ctx.add_u64(output_ptr, elem_offset);
                ctx.st_global_f32(out_addr, sum_val);

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

                // ===== Phase 2: Normalize and scale =====
                let idx2 = ctx.mov_u32_imm(0);

                ctx.label("norm_loop");
                let loop_idx2 = ctx.add_u32_reg(idx2, tid);
                let in_bounds2 = ctx.setp_lt_u32(loop_idx2, hidden_u32);
                ctx.branch_if_not(in_bounds2, "exit");

                // Load sum_val from output buffer and gamma
                let elem_offset2 = ctx.mul_wide_u32_reg(loop_idx2, four);
                let out_addr2 = ctx.add_u64(output_ptr, elem_offset2);
                let gamma_addr = ctx.add_u64(gamma_ptr, elem_offset2);

                let sum_val2 = ctx.ld_global_f32(out_addr2);
                let gamma = ctx.ld_global_f32(gamma_addr);

                // output = sum_val * rms_inv * gamma
                let normalized = ctx.mul_f32(sum_val2, rms_inv);
                let result = ctx.mul_f32(normalized, gamma);

                ctx.st_global_f32(out_addr2, result);

                ctx.add_u32_inplace(idx2, 32);
                ctx.branch("norm_loop");

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// SiLU (Swish) Activation Kernel: output = x * sigmoid(x)
///
/// Sigmoid Linear Unit activation function used in LLaMA/TinyLlama FFN.
/// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// # PAR-023: Used in GPU-resident FFN block
#[derive(Debug, Clone)]
pub struct SiluKernel {
    /// Number of elements
    pub n: u32,
}

impl SiluKernel {
    /// Create a new SiLU activation kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for SiluKernel {
    fn name(&self) -> &str {
        "silu"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("silu")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load parameters
                let n = ctx.load_param_u32("n");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate address
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let in_addr = ctx.add_u64(input_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load x
                let x = ctx.ld_global_f32(in_addr);

                // Compute SiLU: x * sigmoid(x) = x / (1 + exp(-x))
                // Step 1: neg_x = -x (0 - x)
                let zero = ctx.mov_f32_imm(0.0);
                let neg_x = ctx.sub_f32(zero, x);
                // Step 2: exp_neg_x = exp(-x) using ex2 (base-2 exp)
                // exp(x) = 2^(x * log2(e)) where log2(e) ≈ 1.4426950408889634
                let log2_e = ctx.mov_f32_imm(1.442_695);
                let scaled = ctx.mul_f32(neg_x, log2_e);
                let exp_neg_x = ctx.ex2_f32(scaled);
                // Step 3: denom = 1 + exp(-x)
                let one = ctx.mov_f32_imm(1.0);
                let denom = ctx.add_f32(one, exp_neg_x);
                // Step 4: sigmoid = 1 / denom (using division)
                let sigmoid = ctx.div_f32(one, denom);
                // Step 5: result = x * sigmoid
                let result = ctx.mul_f32(x, sigmoid);

                // Store
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// GELU Activation Kernel (approximate): output ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
///
/// Gaussian Error Linear Unit activation function used in GPT/BERT models.
///
/// # PAR-023: Used in GPU-resident FFN block for models using GELU
#[derive(Debug, Clone)]
pub struct GeluKernel {
    /// Number of elements
    pub n: u32,
}

impl GeluKernel {
    /// Create a new GELU activation kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for GeluKernel {
    fn name(&self) -> &str {
        "gelu"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("gelu")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load parameters
                let n = ctx.load_param_u32("n");
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate address
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let in_addr = ctx.add_u64(input_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load x
                let x = ctx.ld_global_f32(in_addr);

                // GELU approximation:
                // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
                // sqrt(2/π) ≈ 0.7978845608
                let sqrt_2_pi = ctx.mov_f32_imm(0.797_884_6);
                let c = ctx.mov_f32_imm(0.044_715);
                let half = ctx.mov_f32_imm(0.5);
                let one = ctx.mov_f32_imm(1.0);

                // x³
                let x2 = ctx.mul_f32(x, x);
                let x3 = ctx.mul_f32(x2, x);

                // 0.044715 * x³
                let cx3 = ctx.mul_f32(c, x3);

                // x + 0.044715 * x³
                let inner = ctx.add_f32(x, cx3);

                // sqrt(2/π) * (x + 0.044715 * x³)
                let scaled = ctx.mul_f32(sqrt_2_pi, inner);

                // tanh approximation using (exp(2x) - 1) / (exp(2x) + 1)
                // For better precision, use: tanh(x) = 2*sigmoid(2x) - 1
                let two = ctx.mov_f32_imm(2.0);
                let zero = ctx.mov_f32_imm(0.0);
                let two_x = ctx.mul_f32(two, scaled);
                let neg_two_x = ctx.sub_f32(zero, two_x);
                let log2_e = ctx.mov_f32_imm(1.442_695);
                let scaled_exp = ctx.mul_f32(neg_two_x, log2_e);
                let exp_neg = ctx.ex2_f32(scaled_exp);
                let denom = ctx.add_f32(one, exp_neg);
                let sigmoid = ctx.div_f32(one, denom);
                // tanh = 2*sigmoid - 1
                let two_sigmoid = ctx.mul_f32(two, sigmoid);
                let tanh = ctx.sub_f32(two_sigmoid, one);

                // 1 + tanh(...)
                let one_plus_tanh = ctx.add_f32(one, tanh);

                // 0.5 * x
                let half_x = ctx.mul_f32(half, x);

                // result = 0.5 * x * (1 + tanh(...))
                let result = ctx.mul_f32(half_x, one_plus_tanh);

                // Store
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Element-wise Multiply Kernel: output = input1 * input2
///
/// Used for gated activations in SwiGLU: silu(gate) * up
///
/// # PAR-023: Used in GPU-resident FFN block
#[derive(Debug, Clone)]
pub struct ElementwiseMulKernel {
    /// Number of elements
    pub n: u32,
}

impl ElementwiseMulKernel {
    /// Create a new element-wise multiply kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for ElementwiseMulKernel {
    fn name(&self) -> &str {
        "elementwise_mul"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("elementwise_mul")
            .param(PtxType::U64, "input1_ptr")
            .param(PtxType::U64, "input2_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load parameters
                let n = ctx.load_param_u32("n");
                let input1_ptr = ctx.load_param_u64("input1_ptr");
                let input2_ptr = ctx.load_param_u64("input2_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate address
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let addr1 = ctx.add_u64(input1_ptr, offset);
                let addr2 = ctx.add_u64(input2_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load both values
                let val1 = ctx.ld_global_f32(addr1);
                let val2 = ctx.ld_global_f32(addr2);

                // Multiply
                let result = ctx.mul_f32(val1, val2);

                // Store
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Fused SwiGLU Kernel: output = silu(gate) * up
///
/// Combines SiLU activation and element-wise multiply in one pass.
/// This is the gated activation used in LLaMA FFN.
///
/// gate_proj = x @ W_gate
/// up_proj = x @ W_up
/// output = silu(gate_proj) * up_proj
///
/// # PAR-023: Fused kernel eliminates one memory round-trip
#[derive(Debug, Clone)]
pub struct FusedSwigluKernel {
    /// Number of elements
    pub n: u32,
}

impl FusedSwigluKernel {
    /// Create a new fused SwiGLU kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for FusedSwigluKernel {
    fn name(&self) -> &str {
        "fused_swiglu"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("fused_swiglu")
            .param(PtxType::U64, "gate_ptr") // gate_proj
            .param(PtxType::U64, "up_ptr") // up_proj
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load parameters
                let n = ctx.load_param_u32("n");
                let gate_ptr = ctx.load_param_u64("gate_ptr");
                let up_ptr = ctx.load_param_u64("up_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate address
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let gate_addr = ctx.add_u64(gate_ptr, offset);
                let up_addr = ctx.add_u64(up_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load gate and up
                let gate = ctx.ld_global_f32(gate_addr);
                let up = ctx.ld_global_f32(up_addr);

                // Compute SiLU(gate): gate * sigmoid(gate)
                let zero = ctx.mov_f32_imm(0.0);
                let neg_gate = ctx.sub_f32(zero, gate);
                let log2_e = ctx.mov_f32_imm(1.442_695);
                let scaled = ctx.mul_f32(neg_gate, log2_e);
                let exp_neg = ctx.ex2_f32(scaled);
                let one = ctx.mov_f32_imm(1.0);
                let denom = ctx.add_f32(one, exp_neg);
                let sigmoid = ctx.div_f32(one, denom);
                let silu_gate = ctx.mul_f32(gate, sigmoid);

                // Multiply: silu(gate) * up
                let result = ctx.mul_f32(silu_gate, up);

                // Store
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
    fn test_residual_add_kernel_name() {
        let kernel = ResidualAddKernel::new(2048);
        assert_eq!(kernel.name(), "residual_add");
    }

    #[test]
    fn test_residual_add_ptx_generation() {
        let kernel = ResidualAddKernel::new(2048);
        let ptx = kernel.emit_ptx();

        // Verify parameters
        assert!(ptx.contains(".param .u64 input1_ptr"));
        assert!(ptx.contains(".param .u64 input2_ptr"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u32 n"));

        // Verify basic structure
        assert!(ptx.contains(".entry residual_add"));
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn test_fused_residual_rmsnorm_kernel_name() {
        let kernel = FusedResidualRmsNormKernel::new(2048);
        assert_eq!(kernel.name(), "fused_residual_rmsnorm");
    }

    #[test]
    fn test_fused_residual_rmsnorm_ptx_generation() {
        let kernel = FusedResidualRmsNormKernel::new(2048);
        let ptx = kernel.emit_ptx();

        // Verify parameters
        assert!(ptx.contains(".param .u64 residual_ptr"));
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u64 gamma_ptr"));

        // Verify warp shuffle for reduction
        assert!(ptx.contains("shfl"));

        // Verify rsqrt for normalization
        assert!(ptx.contains("rsqrt.approx"));

        // Verify fused add
        assert!(ptx.contains("add.f32"));
    }

    #[test]
    fn test_fused_residual_rmsnorm_with_epsilon() {
        let kernel = FusedResidualRmsNormKernel::new(2048).with_epsilon(1e-6);
        assert!((kernel.epsilon - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_residual_add_ptx_valid() {
        let kernel = ResidualAddKernel::new(256);
        let ptx = kernel.emit_ptx();

        // Print first 50 lines for debugging
        for (i, line) in ptx.lines().enumerate().take(50) {
            eprintln!("{:4}: {}", i + 1, line);
        }

        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".target sm_89"));
    }

    // =========================================================================
    // PAR-023: SiLU/GELU/SwiGLU Kernel Tests
    // =========================================================================

    #[test]
    fn test_silu_kernel_name() {
        let kernel = SiluKernel::new(2048);
        assert_eq!(kernel.name(), "silu");
    }

    #[test]
    fn test_silu_ptx_generation() {
        let kernel = SiluKernel::new(2048);
        let ptx = kernel.emit_ptx();

        // Verify parameters
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u32 n"));

        // Verify basic structure
        assert!(ptx.contains(".entry silu"));

        // Verify sigmoid computation (exp, div)
        assert!(ptx.contains("ex2.approx")); // base-2 exp
        assert!(ptx.contains("div.rn")); // division for 1/denom

        // Verify multiplication for x * sigmoid(x)
        assert!(ptx.contains("mul.f32"));
    }

    #[test]
    fn test_gelu_kernel_name() {
        let kernel = GeluKernel::new(2048);
        assert_eq!(kernel.name(), "gelu");
    }

    #[test]
    fn test_gelu_ptx_generation() {
        let kernel = GeluKernel::new(2048);
        let ptx = kernel.emit_ptx();

        // Verify parameters
        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u32 n"));

        // Verify basic structure
        assert!(ptx.contains(".entry gelu"));

        // Verify tanh approximation (exp, div for sigmoid-based)
        assert!(ptx.contains("ex2.approx"));
        assert!(ptx.contains("div.rn"));

        // Verify x³ computation
        assert!(ptx.contains("mul.f32"));
    }

    #[test]
    fn test_elementwise_mul_kernel_name() {
        let kernel = ElementwiseMulKernel::new(2048);
        assert_eq!(kernel.name(), "elementwise_mul");
    }

    #[test]
    fn test_elementwise_mul_ptx_generation() {
        let kernel = ElementwiseMulKernel::new(2048);
        let ptx = kernel.emit_ptx();

        // Verify parameters
        assert!(ptx.contains(".param .u64 input1_ptr"));
        assert!(ptx.contains(".param .u64 input2_ptr"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u32 n"));

        // Verify basic structure
        assert!(ptx.contains(".entry elementwise_mul"));
        assert!(ptx.contains("mul.f32"));
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));
    }

    #[test]
    fn test_fused_swiglu_kernel_name() {
        let kernel = FusedSwigluKernel::new(2048);
        assert_eq!(kernel.name(), "fused_swiglu");
    }

    #[test]
    fn test_fused_swiglu_ptx_generation() {
        let kernel = FusedSwigluKernel::new(2048);
        let ptx = kernel.emit_ptx();

        // Verify parameters
        assert!(ptx.contains(".param .u64 gate_ptr"));
        assert!(ptx.contains(".param .u64 up_ptr"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u32 n"));

        // Verify basic structure
        assert!(ptx.contains(".entry fused_swiglu"));

        // Verify SiLU computation
        assert!(ptx.contains("ex2.approx"));
        assert!(ptx.contains("div.rn"));

        // Verify final multiply
        assert!(ptx.contains("mul.f32"));
    }

    // PARITY-114: Barrier safety tests for new kernels
    #[test]
    fn test_barrier_safety_silu() {
        let kernel = SiluKernel::new(1024);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "SiLU should be barrier-safe: {:?}",
            result.violations
        );
    }

    #[test]
    fn test_barrier_safety_gelu() {
        let kernel = GeluKernel::new(1024);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "GELU should be barrier-safe: {:?}",
            result.violations
        );
    }

    #[test]
    fn test_barrier_safety_elementwise_mul() {
        let kernel = ElementwiseMulKernel::new(1024);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "ElementwiseMul should be barrier-safe: {:?}",
            result.violations
        );
    }

    #[test]
    fn test_barrier_safety_fused_swiglu() {
        let kernel = FusedSwigluKernel::new(1024);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "FusedSwiGLU should be barrier-safe: {:?}",
            result.violations
        );
    }
}
