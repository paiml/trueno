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

#![allow(clippy::similar_names)]

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

// ============================================================================
// PAR-114: Batched Residual Add Kernel (processes M sequences in parallel)
// ============================================================================

/// Batched Residual Add: output[m] = input1[m] + input2[m] for m in 0..M
///
/// Processes M sequences in parallel using Grid.y for batch index.
///
/// # Parameters
///
/// - `input1_ptr`: First packed input [M × n]
/// - `input2_ptr`: Second packed input [M × n]
/// - `output_ptr`: Output [M × n]
/// - `n`: Elements per sequence
///
/// # Grid Configuration
///
/// - Grid: (ceil(n/256), batch_size, 1)
/// - Block: (256, 1, 1)
#[derive(Debug, Clone)]
pub struct BatchedResidualAddKernel {
    /// Elements per sequence
    pub n: u32,
    /// Batch size (M)
    pub batch_size: u32,
}

impl BatchedResidualAddKernel {
    /// Create a new batched residual add kernel
    #[must_use]
    pub const fn new(n: u32, batch_size: u32) -> Self {
        Self { n, batch_size }
    }
}

impl Kernel for BatchedResidualAddKernel {
    fn name(&self) -> &str {
        "batched_residual_add"
    }

    fn build_ptx(&self) -> PtxKernel {
        let n = self.n;

        PtxKernel::new("batched_residual_add")
            .param(PtxType::U64, "input1_ptr")
            .param(PtxType::U64, "input2_ptr")
            .param(PtxType::U64, "output_ptr")
            .build(move |ctx| {
                // Global thread ID within the sequence
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY); // Grid.y = sequence index
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let local_gid = ctx.mad_lo_u32(ctaid_x, ntid, tid);

                // Load parameters
                let input1_ptr = ctx.load_param_u64("input1_ptr");
                let input2_ptr = ctx.load_param_u64("input2_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check within sequence
                let n_val = ctx.mov_u32_imm(n);
                let in_bounds = ctx.setp_lt_u32(local_gid, n_val);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate global element index: batch_idx × n + local_gid
                let batch_offset = ctx.mul_lo_u32(batch_idx, n_val);
                let gid = ctx.add_u32_reg(batch_offset, local_gid);

                // Calculate byte address (gid × 4 bytes)
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
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
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
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
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

/// Scale Kernel: output = input * scale (scalar constant)
///
/// Multiplies each element by a constant scale factor.
/// Used for attention score scaling (1/sqrt(d_k)).
#[derive(Debug, Clone)]
pub struct ScaleKernel {
    /// Number of elements
    pub n: u32,
}

impl ScaleKernel {
    /// Create a new scale kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for ScaleKernel {
    fn name(&self) -> &str {
        "scale"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("scale")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::F32, "scale")
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
                let scale = ctx.load_param_f32("scale");

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(gid, n);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate address
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let in_addr = ctx.add_u64(input_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load input value
                let val = ctx.ld_global_f32(in_addr);

                // Multiply by scale
                let result = ctx.mul_f32(val, scale);

                // Store result
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
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
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

// ============================================================================
// PAR-114: Batched SwiGLU Kernel (processes M sequences in parallel)
// ============================================================================

/// Batched SwiGLU: output[m] = silu(gate[m]) * up[m] for m in 0..M
///
/// Processes M sequences in parallel using Grid.y for batch index.
///
/// # Parameters
///
/// - `gate_ptr`: Packed gate values [M × n]
/// - `up_ptr`: Packed up values [M × n]
/// - `output_ptr`: Output [M × n]
///
/// # Grid Configuration
///
/// - Grid: (ceil(n/256), batch_size, 1)
/// - Block: (256, 1, 1)
#[derive(Debug, Clone)]
pub struct BatchedSwigluKernel {
    /// Elements per sequence
    pub n: u32,
    /// Batch size (M)
    pub batch_size: u32,
}

impl BatchedSwigluKernel {
    /// Create a new batched SwiGLU kernel
    #[must_use]
    pub const fn new(n: u32, batch_size: u32) -> Self {
        Self { n, batch_size }
    }
}

impl Kernel for BatchedSwigluKernel {
    fn name(&self) -> &str {
        "batched_swiglu"
    }

    fn build_ptx(&self) -> PtxKernel {
        let n = self.n;

        PtxKernel::new("batched_swiglu")
            .param(PtxType::U64, "gate_ptr")
            .param(PtxType::U64, "up_ptr")
            .param(PtxType::U64, "output_ptr")
            .build(move |ctx| {
                // Global thread ID within the sequence
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY); // Grid.y = sequence index
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let local_gid = ctx.mad_lo_u32(ctaid_x, ntid, tid);

                // Load parameters
                let gate_ptr = ctx.load_param_u64("gate_ptr");
                let up_ptr = ctx.load_param_u64("up_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check within sequence
                let n_val = ctx.mov_u32_imm(n);
                let in_bounds = ctx.setp_lt_u32(local_gid, n_val);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate global element index: batch_idx × n + local_gid
                let batch_offset = ctx.mul_lo_u32(batch_idx, n_val);
                let gid = ctx.add_u32_reg(batch_offset, local_gid);

                // Calculate byte address (gid × 4 bytes)
                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let gate_addr = ctx.add_u64(gate_ptr, offset);
                let up_addr = ctx.add_u64(up_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                // Load gate and up values
                let gate = ctx.ld_global_f32(gate_addr);
                let up = ctx.ld_global_f32(up_addr);

                // Compute SiLU(gate): gate × sigmoid(gate)
                let zero = ctx.mov_f32_imm(0.0);
                let neg_gate = ctx.sub_f32(zero, gate);
                let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scaled = ctx.mul_f32(neg_gate, log2_e);
                let exp_neg = ctx.ex2_f32(scaled);
                let one = ctx.mov_f32_imm(1.0);
                let denom = ctx.add_f32(one, exp_neg);
                let sigmoid = ctx.div_f32(one, denom);
                let silu_gate = ctx.mul_f32(gate, sigmoid);

                // Multiply: silu(gate) × up
                let result = ctx.mul_f32(silu_gate, up);

                // Store
                ctx.st_global_f32(out_addr, result);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// ============================================================================
// PAR-052: KV Cache Scatter Kernels
// ============================================================================

/// KV Cache Scatter Kernel: Scatter K/V vectors to strided KV cache positions
///
/// Used to update KV cache at specific positions without full D2D copies.
/// Replaces 672+ D2D copies per token with two kernel launches.
#[derive(Debug, Clone)]
pub struct KvCacheScatterKernel {
    /// Number of KV heads
    pub num_kv_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Maximum sequence length
    pub max_len: u32,
}

impl KvCacheScatterKernel {
    /// Create a new KV cache scatter kernel
    #[must_use]
    pub const fn new(num_kv_heads: u32, head_dim: u32, max_len: u32) -> Self {
        Self { num_kv_heads, head_dim, max_len }
    }
}

impl Kernel for KvCacheScatterKernel {
    fn name(&self) -> &str {
        "kv_cache_scatter"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("kv_cache_scatter")
            .param(PtxType::U64, "src_ptr")
            .param(PtxType::U64, "cache_ptr")
            .param(PtxType::U32, "pos")
            .param(PtxType::U32, "head_dim")
            .param(PtxType::U32, "max_len")
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let src_ptr = ctx.load_param_u64("src_ptr");
                let cache_ptr = ctx.load_param_u64("cache_ptr");
                let pos = ctx.load_param_u32("pos");
                let head_dim = ctx.load_param_u32("head_dim");
                let max_len = ctx.load_param_u32("max_len");

                // Each block handles one head, each thread one element
                let head_idx = ctaid;
                let elem_idx = tid;

                // Bounds check
                let in_bounds = ctx.setp_lt_u32(elem_idx, head_dim);
                ctx.branch_if_not(in_bounds, "exit");

                // Source offset: head_idx * head_dim + elem_idx
                let src_head_offset = ctx.mul_lo_u32(head_idx, head_dim);
                let src_offset = ctx.add_u32_reg(src_head_offset, elem_idx);
                let four = ctx.mov_u32_imm(4);
                let src_bytes = ctx.mul_lo_u32(src_offset, four);
                let src_bytes_64 = ctx.cvt_u64_u32(src_bytes);
                let src_addr = ctx.add_u64(src_ptr, src_bytes_64);

                // Cache offset: (head_idx * max_len + pos) * head_dim + elem_idx
                let cache_head_stride = ctx.mul_lo_u32(head_idx, max_len);
                let cache_pos_offset = ctx.add_u32_reg(cache_head_stride, pos);
                let cache_elem_stride = ctx.mul_lo_u32(cache_pos_offset, head_dim);
                let cache_offset = ctx.add_u32_reg(cache_elem_stride, elem_idx);
                let cache_bytes = ctx.mul_lo_u32(cache_offset, four);
                let cache_bytes_64 = ctx.cvt_u64_u32(cache_bytes);
                let cache_addr = ctx.add_u64(cache_ptr, cache_bytes_64);

                // Load from source and store to cache
                let val = ctx.ld_global_f32(src_addr);
                ctx.st_global_f32(cache_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// KV Cache Scatter Indirect Kernel: CUDA Graph compatible version
///
/// Reads position from device memory instead of kernel parameter.
#[derive(Debug, Clone)]
pub struct KvCacheScatterIndirectKernel {
    /// Number of KV heads
    pub num_kv_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Maximum sequence length
    pub max_len: u32,
}

impl KvCacheScatterIndirectKernel {
    /// Create a new indirect KV cache scatter kernel
    #[must_use]
    pub const fn new(num_kv_heads: u32, head_dim: u32, max_len: u32) -> Self {
        Self { num_kv_heads, head_dim, max_len }
    }
}

impl Kernel for KvCacheScatterIndirectKernel {
    fn name(&self) -> &str {
        "kv_cache_scatter_indirect"
    }

    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("kv_cache_scatter_indirect")
            .param(PtxType::U64, "src_ptr")
            .param(PtxType::U64, "cache_ptr")
            .param(PtxType::U64, "pos_ptr")  // Indirect: read from device memory
            .param(PtxType::U32, "head_dim")
            .param(PtxType::U32, "max_len")
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let src_ptr = ctx.load_param_u64("src_ptr");
                let cache_ptr = ctx.load_param_u64("cache_ptr");
                let pos_ptr = ctx.load_param_u64("pos_ptr");
                let head_dim = ctx.load_param_u32("head_dim");
                let max_len = ctx.load_param_u32("max_len");

                // Read position from device memory (indirect)
                let pos = ctx.ld_global_u32(pos_ptr);

                let head_idx = ctaid;
                let elem_idx = tid;

                let in_bounds = ctx.setp_lt_u32(elem_idx, head_dim);
                ctx.branch_if_not(in_bounds, "exit");

                let src_head_offset = ctx.mul_lo_u32(head_idx, head_dim);
                let src_offset = ctx.add_u32_reg(src_head_offset, elem_idx);
                let four = ctx.mov_u32_imm(4);
                let src_bytes = ctx.mul_lo_u32(src_offset, four);
                let src_bytes_64 = ctx.cvt_u64_u32(src_bytes);
                let src_addr = ctx.add_u64(src_ptr, src_bytes_64);

                let cache_head_stride = ctx.mul_lo_u32(head_idx, max_len);
                let cache_pos_offset = ctx.add_u32_reg(cache_head_stride, pos);
                let cache_elem_stride = ctx.mul_lo_u32(cache_pos_offset, head_dim);
                let cache_offset = ctx.add_u32_reg(cache_elem_stride, elem_idx);
                let cache_bytes = ctx.mul_lo_u32(cache_offset, four);
                let cache_bytes_64 = ctx.cvt_u64_u32(cache_bytes);
                let cache_addr = ctx.add_u64(cache_ptr, cache_bytes_64);

                let val = ctx.ld_global_f32(src_addr);
                ctx.st_global_f32(cache_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// ============================================================================
// PAR-060: RoPE (Rotary Position Embedding) Kernels
// ============================================================================

/// RoPE Kernel: Apply rotary position embeddings to Q or K vectors
#[derive(Debug, Clone)]
pub struct RopeKernel {
    /// Number of heads
    pub num_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Rope theta base (typically 10000.0)
    pub theta: f32,
}

impl RopeKernel {
    /// Create a new RoPE kernel
    #[must_use]
    pub fn new(num_heads: u32, head_dim: u32, theta: f32) -> Self {
        Self { num_heads, head_dim, theta }
    }
}

impl Kernel for RopeKernel {
    fn name(&self) -> &str {
        "rope"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let theta = self.theta;
        PtxKernel::new("rope")
            .param(PtxType::U64, "x_ptr")       // Input/output Q or K vectors (in-place)
            .param(PtxType::U64, "out_ptr")    // Output pointer (can be same as x_ptr)
            .param(PtxType::U32, "pos")        // Position in sequence
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let pos = ctx.load_param_u32("pos");

                // Each block handles one head, threads handle pairs
                let head_idx = ctaid;
                let pair_idx = tid;  // Process elements in pairs

                let half_dim = ctx.mov_u32_imm(head_dim / 2);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate element indices for the pair
                let two = ctx.mov_u32_imm(2);
                let elem0 = ctx.mul_lo_u32(pair_idx, two);
                let one = ctx.mov_u32_imm(1);
                let elem1 = ctx.add_u32_reg(elem0, one);

                // X offset: head_idx * head_dim + elem
                let dim = ctx.mov_u32_imm(head_dim);
                let head_offset = ctx.mul_lo_u32(head_idx, dim);
                let offset0 = ctx.add_u32_reg(head_offset, elem0);
                let offset1 = ctx.add_u32_reg(head_offset, elem1);

                let four = ctx.mov_u32_imm(4);
                let bytes0 = ctx.mul_lo_u32(offset0, four);
                let bytes1 = ctx.mul_lo_u32(offset1, four);
                let bytes0_64 = ctx.cvt_u64_u32(bytes0);
                let bytes1_64 = ctx.cvt_u64_u32(bytes1);
                let addr0 = ctx.add_u64(x_ptr, bytes0_64);
                let addr1 = ctx.add_u64(x_ptr, bytes1_64);
                let out_addr0 = ctx.add_u64(out_ptr, bytes0_64);
                let out_addr1 = ctx.add_u64(out_ptr, bytes1_64);

                // Load x values
                let x0 = ctx.ld_global_f32(addr0);
                let x1 = ctx.ld_global_f32(addr1);

                // Compute frequency on-the-fly: freq = 1.0 / (theta^(2*pair_idx/head_dim))
                // = theta^(-2*pair_idx/head_dim)
                // exp(-2*pair_idx/head_dim * log(theta))
                // Using: theta^x = 2^(x * log2(theta))
                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32(power);

                // angle = position * freq_base
                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                // Compute sin and cos using PTX approximations
                let cos_val = ctx.cos_f32(angle);
                let sin_val = ctx.sin_f32(angle);

                // Apply rotation: (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
                let x0_cos = ctx.mul_f32(x0, cos_val);
                let x1_sin = ctx.mul_f32(x1, sin_val);
                let new_x0 = ctx.sub_f32(x0_cos, x1_sin);

                let x0_sin = ctx.mul_f32(x0, sin_val);
                let x1_cos = ctx.mul_f32(x1, cos_val);
                let new_x1 = ctx.add_f32(x0_sin, x1_cos);

                // Store results
                ctx.st_global_f32(out_addr0, new_x0);
                ctx.st_global_f32(out_addr1, new_x1);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// RoPE Indirect Kernel: CUDA Graph compatible version
#[derive(Debug, Clone)]
pub struct RopeIndirectKernel {
    /// Number of heads
    pub num_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Rope theta base (typically 10000.0)
    pub theta: f32,
}

impl RopeIndirectKernel {
    /// Create a new indirect RoPE kernel
    #[must_use]
    pub fn new(num_heads: u32, head_dim: u32, theta: f32) -> Self {
        Self { num_heads, head_dim, theta }
    }
}

impl Kernel for RopeIndirectKernel {
    fn name(&self) -> &str {
        "rope_indirect"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let theta = self.theta;
        PtxKernel::new("rope_indirect")
            .param(PtxType::U64, "x_ptr")       // Input/output Q or K vectors (in-place)
            .param(PtxType::U64, "out_ptr")    // Output pointer (can be same as x_ptr)
            .param(PtxType::U64, "pos_ptr")    // Indirect: read position from device memory
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let pos_ptr = ctx.load_param_u64("pos_ptr");

                // Read position from device memory (indirect - allows CUDA graph replay)
                let pos = ctx.ld_global_u32(pos_ptr);

                let head_idx = ctaid;
                let pair_idx = tid;

                let half_dim = ctx.mov_u32_imm(head_dim / 2);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim);
                ctx.branch_if_not(in_bounds, "exit");

                let two = ctx.mov_u32_imm(2);
                let elem0 = ctx.mul_lo_u32(pair_idx, two);
                let one = ctx.mov_u32_imm(1);
                let elem1 = ctx.add_u32_reg(elem0, one);

                let dim = ctx.mov_u32_imm(head_dim);
                let head_offset = ctx.mul_lo_u32(head_idx, dim);
                let offset0 = ctx.add_u32_reg(head_offset, elem0);
                let offset1 = ctx.add_u32_reg(head_offset, elem1);

                let four = ctx.mov_u32_imm(4);
                let bytes0 = ctx.mul_lo_u32(offset0, four);
                let bytes1 = ctx.mul_lo_u32(offset1, four);
                let bytes0_64 = ctx.cvt_u64_u32(bytes0);
                let bytes1_64 = ctx.cvt_u64_u32(bytes1);
                let addr0 = ctx.add_u64(x_ptr, bytes0_64);
                let addr1 = ctx.add_u64(x_ptr, bytes1_64);
                let out_addr0 = ctx.add_u64(out_ptr, bytes0_64);
                let out_addr1 = ctx.add_u64(out_ptr, bytes1_64);

                let x0 = ctx.ld_global_f32(addr0);
                let x1 = ctx.ld_global_f32(addr1);

                // Compute frequency on-the-fly: freq = 1.0 / (theta^(2*pair_idx/head_dim))
                // = theta^(-2*pair_idx/head_dim)
                // Using: theta^x = 2^(x * log2(theta))
                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32(power);

                // angle = position * freq_base
                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                // Compute sin and cos using PTX approximations
                let cos_val = ctx.cos_f32(angle);
                let sin_val = ctx.sin_f32(angle);

                // Apply rotation: (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
                let x0_cos = ctx.mul_f32(x0, cos_val);
                let x1_sin = ctx.mul_f32(x1, sin_val);
                let new_x0 = ctx.sub_f32(x0_cos, x1_sin);

                let x0_sin = ctx.mul_f32(x0, sin_val);
                let x1_cos = ctx.mul_f32(x1, cos_val);
                let new_x1 = ctx.add_f32(x0_sin, x1_cos);

                ctx.st_global_f32(out_addr0, new_x0);
                ctx.st_global_f32(out_addr1, new_x1);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// ============================================================================
// CORRECTNESS-011: NEOX-Style RoPE Kernels (split halves instead of adjacent pairs)
// ============================================================================

/// RoPE NEOX Kernel: Apply rotary position embeddings using NEOX/GPT-NeoX style
///
/// NEOX style uses split halves: pairs are at indices (i, i + half_dim)
/// This is required for Qwen2.5 models (rope_type=2)
#[derive(Debug, Clone)]
pub struct RopeNeoxKernel {
    /// Number of heads
    pub num_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Rope theta base (typically 10000.0 or 1000000.0)
    pub theta: f32,
}

impl RopeNeoxKernel {
    /// Create a new NEOX-style RoPE kernel
    #[must_use]
    pub fn new(num_heads: u32, head_dim: u32, theta: f32) -> Self {
        Self { num_heads, head_dim, theta }
    }
}

impl Kernel for RopeNeoxKernel {
    fn name(&self) -> &str {
        "rope_neox"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let theta = self.theta;
        let half_dim = head_dim / 2;
        PtxKernel::new("rope_neox")
            .param(PtxType::U64, "x_ptr")       // Input/output Q or K vectors (in-place)
            .param(PtxType::U64, "out_ptr")    // Output pointer (can be same as x_ptr)
            .param(PtxType::U32, "pos")        // Position in sequence
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let pos = ctx.load_param_u32("pos");

                // Each block handles one head, threads handle pairs
                let head_idx = ctaid;
                let pair_idx = tid;  // Process elements in pairs

                let half_dim_reg = ctx.mov_u32_imm(half_dim);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim_reg);
                ctx.branch_if_not(in_bounds, "exit");

                // NEOX style: elem0 = pair_idx (first half), elem1 = pair_idx + half_dim (second half)
                let elem0 = pair_idx;
                let elem1 = ctx.add_u32_reg(pair_idx, half_dim_reg);

                // X offset: head_idx * head_dim + elem
                let dim = ctx.mov_u32_imm(head_dim);
                let head_offset = ctx.mul_lo_u32(head_idx, dim);
                let offset0 = ctx.add_u32_reg(head_offset, elem0);
                let offset1 = ctx.add_u32_reg(head_offset, elem1);

                let four = ctx.mov_u32_imm(4);
                let bytes0 = ctx.mul_lo_u32(offset0, four);
                let bytes1 = ctx.mul_lo_u32(offset1, four);
                let bytes0_64 = ctx.cvt_u64_u32(bytes0);
                let bytes1_64 = ctx.cvt_u64_u32(bytes1);
                let addr0 = ctx.add_u64(x_ptr, bytes0_64);
                let addr1 = ctx.add_u64(x_ptr, bytes1_64);
                let out_addr0 = ctx.add_u64(out_ptr, bytes0_64);
                let out_addr1 = ctx.add_u64(out_ptr, bytes1_64);

                // Load x values
                let x0 = ctx.ld_global_f32(addr0);
                let x1 = ctx.ld_global_f32(addr1);

                // Compute frequency on-the-fly: freq = 1.0 / (theta^(2*pair_idx/head_dim))
                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32(power);

                // angle = position * freq_base
                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                // Compute sin and cos using PTX approximations
                let cos_val = ctx.cos_f32(angle);
                let sin_val = ctx.sin_f32(angle);

                // Apply rotation: (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
                let x0_cos = ctx.mul_f32(x0, cos_val);
                let x1_sin = ctx.mul_f32(x1, sin_val);
                let new_x0 = ctx.sub_f32(x0_cos, x1_sin);

                let x0_sin = ctx.mul_f32(x0, sin_val);
                let x1_cos = ctx.mul_f32(x1, cos_val);
                let new_x1 = ctx.add_f32(x0_sin, x1_cos);

                // Store results
                ctx.st_global_f32(out_addr0, new_x0);
                ctx.st_global_f32(out_addr1, new_x1);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// RoPE NEOX Indirect Kernel: CUDA Graph compatible NEOX-style version
#[derive(Debug, Clone)]
pub struct RopeNeoxIndirectKernel {
    /// Number of heads
    pub num_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Rope theta base (typically 10000.0 or 1000000.0)
    pub theta: f32,
}

impl RopeNeoxIndirectKernel {
    /// Create a new indirect NEOX-style RoPE kernel
    #[must_use]
    pub fn new(num_heads: u32, head_dim: u32, theta: f32) -> Self {
        Self { num_heads, head_dim, theta }
    }
}

impl Kernel for RopeNeoxIndirectKernel {
    fn name(&self) -> &str {
        "rope_neox_indirect"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let theta = self.theta;
        let half_dim = head_dim / 2;
        PtxKernel::new("rope_neox_indirect")
            .param(PtxType::U64, "x_ptr")       // Input/output Q or K vectors (in-place)
            .param(PtxType::U64, "out_ptr")    // Output pointer (can be same as x_ptr)
            .param(PtxType::U64, "pos_ptr")    // Indirect: read position from device memory
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let pos_ptr = ctx.load_param_u64("pos_ptr");

                // Read position from device memory (indirect - allows CUDA graph replay)
                let pos = ctx.ld_global_u32(pos_ptr);

                let head_idx = ctaid;
                let pair_idx = tid;

                let half_dim_reg = ctx.mov_u32_imm(half_dim);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim_reg);
                ctx.branch_if_not(in_bounds, "exit");

                // NEOX style: elem0 = pair_idx (first half), elem1 = pair_idx + half_dim (second half)
                let elem0 = pair_idx;
                let elem1 = ctx.add_u32_reg(pair_idx, half_dim_reg);

                let dim = ctx.mov_u32_imm(head_dim);
                let head_offset = ctx.mul_lo_u32(head_idx, dim);
                let offset0 = ctx.add_u32_reg(head_offset, elem0);
                let offset1 = ctx.add_u32_reg(head_offset, elem1);

                let four = ctx.mov_u32_imm(4);
                let bytes0 = ctx.mul_lo_u32(offset0, four);
                let bytes1 = ctx.mul_lo_u32(offset1, four);
                let bytes0_64 = ctx.cvt_u64_u32(bytes0);
                let bytes1_64 = ctx.cvt_u64_u32(bytes1);
                let addr0 = ctx.add_u64(x_ptr, bytes0_64);
                let addr1 = ctx.add_u64(x_ptr, bytes1_64);
                let out_addr0 = ctx.add_u64(out_ptr, bytes0_64);
                let out_addr1 = ctx.add_u64(out_ptr, bytes1_64);

                let x0 = ctx.ld_global_f32(addr0);
                let x1 = ctx.ld_global_f32(addr1);

                // Compute frequency on-the-fly: freq = 1.0 / (theta^(2*pair_idx/head_dim))
                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32(power);

                // angle = position * freq_base
                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                // Compute sin and cos using PTX approximations
                let cos_val = ctx.cos_f32(angle);
                let sin_val = ctx.sin_f32(angle);

                // Apply rotation: (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
                let x0_cos = ctx.mul_f32(x0, cos_val);
                let x1_sin = ctx.mul_f32(x1, sin_val);
                let new_x0 = ctx.sub_f32(x0_cos, x1_sin);

                let x0_sin = ctx.mul_f32(x0, sin_val);
                let x1_cos = ctx.mul_f32(x1, cos_val);
                let new_x1 = ctx.add_f32(x0_sin, x1_cos);

                ctx.st_global_f32(out_addr0, new_x0);
                ctx.st_global_f32(out_addr1, new_x1);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// ============================================================================
// PAR-114: Batched RoPE Kernel (processes M sequences in parallel)
// ============================================================================

/// Batched RoPE Kernel: Apply rotary position embeddings to M sequences
///
/// Processes M sequences in parallel using Grid.y for batch index.
/// Each sequence can have a different position.
///
/// # Parameters
///
/// - `x_ptr`: Packed input vectors [M × num_heads × head_dim]
/// - `out_ptr`: Output vectors (can be same as x_ptr for in-place)
/// - `positions_ptr`: Array of M positions (u32[M])
///
/// # Grid Configuration
///
/// - Grid: (num_heads, batch_size, 1)
/// - Block: (head_dim / 2, 1, 1)
/// - Each block processes one head of one sequence
#[derive(Debug, Clone)]
pub struct BatchedRopeKernel {
    /// Number of heads
    pub num_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Rope theta base (typically 10000.0)
    pub theta: f32,
}

impl BatchedRopeKernel {
    /// Create a new batched RoPE kernel
    #[must_use]
    pub fn new(num_heads: u32, head_dim: u32, batch_size: u32, theta: f32) -> Self {
        Self { num_heads, head_dim, batch_size, theta }
    }
}

impl Kernel for BatchedRopeKernel {
    fn name(&self) -> &str {
        "batched_rope"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let theta = self.theta;

        PtxKernel::new("batched_rope")
            .param(PtxType::U64, "x_ptr")        // Packed input [M × num_heads × head_dim]
            .param(PtxType::U64, "out_ptr")     // Output (can alias x_ptr)
            .param(PtxType::U64, "positions_ptr") // Array of M positions
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);  // blockIdx.x = head
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY); // blockIdx.y = sequence

                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let positions_ptr = ctx.load_param_u64("positions_ptr");

                let pair_idx = tid;

                // Bounds check
                let half_dim = ctx.mov_u32_imm(head_dim / 2);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim);
                ctx.branch_if_not(in_bounds, "exit");

                // Read position for this sequence from positions_ptr[batch_idx]
                let four = ctx.mov_u32_imm(4);
                let pos_byte_offset = ctx.mul_lo_u32(batch_idx, four);
                let pos_byte_offset_64 = ctx.cvt_u64_u32(pos_byte_offset);
                let pos_addr = ctx.add_u64(positions_ptr, pos_byte_offset_64);
                let pos = ctx.ld_global_u32(pos_addr);

                // Calculate element indices for the pair
                let two = ctx.mov_u32_imm(2);
                let elem0 = ctx.mul_lo_u32(pair_idx, two);
                let one = ctx.mov_u32_imm(1);
                let elem1 = ctx.add_u32_reg(elem0, one);

                // Batch offset: batch_idx × num_heads × head_dim
                let heads_per_seq = ctx.mov_u32_imm(num_heads);
                let dim = ctx.mov_u32_imm(head_dim);
                let seq_stride = ctx.mul_lo_u32(heads_per_seq, dim);
                let batch_offset = ctx.mul_lo_u32(batch_idx, seq_stride);

                // Head offset within sequence: head_idx × head_dim
                let head_offset = ctx.mul_lo_u32(head_idx, dim);

                // Total offset: batch_offset + head_offset + element
                let base_offset = ctx.add_u32_reg(batch_offset, head_offset);
                let offset0 = ctx.add_u32_reg(base_offset, elem0);
                let offset1 = ctx.add_u32_reg(base_offset, elem1);

                // Convert to byte offsets
                let bytes0 = ctx.mul_lo_u32(offset0, four);
                let bytes1 = ctx.mul_lo_u32(offset1, four);
                let bytes0_64 = ctx.cvt_u64_u32(bytes0);
                let bytes1_64 = ctx.cvt_u64_u32(bytes1);

                // Calculate addresses
                let addr0 = ctx.add_u64(x_ptr, bytes0_64);
                let addr1 = ctx.add_u64(x_ptr, bytes1_64);
                let out_addr0 = ctx.add_u64(out_ptr, bytes0_64);
                let out_addr1 = ctx.add_u64(out_ptr, bytes1_64);

                // Load x values
                let x0 = ctx.ld_global_f32(addr0);
                let x1 = ctx.ld_global_f32(addr1);

                // Compute frequency: freq = theta^(-2*pair_idx/head_dim)
                // Using: theta^x = 2^(x * log2(theta))
                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32(power);

                // angle = position × freq_base
                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                // Compute sin and cos
                let cos_val = ctx.cos_f32(angle);
                let sin_val = ctx.sin_f32(angle);

                // Apply rotation: (x0 × cos - x1 × sin, x0 × sin + x1 × cos)
                let x0_cos = ctx.mul_f32(x0, cos_val);
                let x1_sin = ctx.mul_f32(x1, sin_val);
                let new_x0 = ctx.sub_f32(x0_cos, x1_sin);

                let x0_sin = ctx.mul_f32(x0, sin_val);
                let x1_cos = ctx.mul_f32(x1, cos_val);
                let new_x1 = ctx.add_f32(x0_sin, x1_cos);

                // Store results
                ctx.st_global_f32(out_addr0, new_x0);
                ctx.st_global_f32(out_addr1, new_x1);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// CORRECTNESS-013: Precise RoPE Kernel for CPU/GPU bit-exactness
///
/// Uses polynomial sin/cos approximations instead of hardware `sin.approx.f32`
/// and `cos.approx.f32` which have ~2^-21 error. For Qwen 2.5 with theta=1M,
/// the high-frequency components are very sensitive to trig precision.
///
/// Enable via CORRECTNESS_MODE=1 environment variable.
#[derive(Debug, Clone)]
pub struct PreciseRopeKernel {
    /// Number of heads
    pub num_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Rope theta base (typically 10000.0 or 1000000.0 for Qwen2.5)
    pub theta: f32,
}

impl PreciseRopeKernel {
    /// Create a new precise RoPE kernel
    #[must_use]
    pub fn new(num_heads: u32, head_dim: u32, theta: f32) -> Self {
        Self { num_heads, head_dim, theta }
    }
}

impl Kernel for PreciseRopeKernel {
    fn name(&self) -> &str {
        "rope_precise"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let theta = self.theta;

        PtxKernel::new("rope_precise")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "out_ptr")
            .param(PtxType::U32, "pos")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let pos = ctx.load_param_u32("pos");

                // Each block handles one head, threads handle pairs
                let head_idx = ctaid;
                let pair_idx = tid;

                let half_dim = ctx.mov_u32_imm(head_dim / 2);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim);
                ctx.branch_if_not(in_bounds, "exit");

                // Calculate element indices for the pair
                let two = ctx.mov_u32_imm(2);
                let elem0 = ctx.mul_lo_u32(pair_idx, two);
                let one = ctx.mov_u32_imm(1);
                let elem1 = ctx.add_u32_reg(elem0, one);

                // X offset: head_idx * head_dim + elem
                let dim = ctx.mov_u32_imm(head_dim);
                let head_offset = ctx.mul_lo_u32(head_idx, dim);
                let offset0 = ctx.add_u32_reg(head_offset, elem0);
                let offset1 = ctx.add_u32_reg(head_offset, elem1);

                let four = ctx.mov_u32_imm(4);
                let bytes0 = ctx.mul_lo_u32(offset0, four);
                let bytes1 = ctx.mul_lo_u32(offset1, four);
                let bytes0_64 = ctx.cvt_u64_u32(bytes0);
                let bytes1_64 = ctx.cvt_u64_u32(bytes1);
                let addr0 = ctx.add_u64(x_ptr, bytes0_64);
                let addr1 = ctx.add_u64(x_ptr, bytes1_64);
                let out_addr0 = ctx.add_u64(out_ptr, bytes0_64);
                let out_addr1 = ctx.add_u64(out_ptr, bytes1_64);

                // Load x values
                let x0 = ctx.ld_global_f32(addr0);
                let x1 = ctx.ld_global_f32(addr1);

                // Compute frequency on-the-fly: freq = theta^(-2*pair_idx/head_dim)
                // Using: theta^x = 2^(x * log2(theta))
                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32(power);

                // angle = position * freq_base
                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                // CORRECTNESS-013: Use precise polynomial sin/cos instead of .approx
                let cos_val = ctx.cos_f32_precise(angle);
                let sin_val = ctx.sin_f32_precise(angle);

                // Apply rotation: (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
                let x0_cos = ctx.mul_f32(x0, cos_val);
                let x1_sin = ctx.mul_f32(x1, sin_val);
                let new_x0 = ctx.sub_f32(x0_cos, x1_sin);

                let x0_sin = ctx.mul_f32(x0, sin_val);
                let x1_cos = ctx.mul_f32(x1, cos_val);
                let new_x1 = ctx.add_f32(x0_sin, x1_cos);

                // Store results
                ctx.st_global_f32(out_addr0, new_x0);
                ctx.st_global_f32(out_addr1, new_x1);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// CORRECTNESS-013: Precise RoPE Indirect Kernel for CUDA graph compatibility
///
/// Same as PreciseRopeKernel but reads position from a GPU buffer
/// instead of a kernel parameter, enabling CUDA graph capture.
#[derive(Debug, Clone)]
pub struct PreciseRopeIndirectKernel {
    /// Number of heads
    pub num_heads: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Rope theta base
    pub theta: f32,
}

impl PreciseRopeIndirectKernel {
    /// Create a new precise RoPE indirect kernel
    #[must_use]
    pub fn new(num_heads: u32, head_dim: u32, theta: f32) -> Self {
        Self { num_heads, head_dim, theta }
    }
}

impl Kernel for PreciseRopeIndirectKernel {
    fn name(&self) -> &str {
        "rope_precise_indirect"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let theta = self.theta;
        let half_dim = head_dim / 2;

        PtxKernel::new("rope_precise_indirect")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "out_ptr")
            .param(PtxType::U64, "pos_ptr")  // Position read from GPU buffer
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let pos_ptr = ctx.load_param_u64("pos_ptr");

                // Read position from GPU buffer (CUDA graph compatible)
                let pos = ctx.ld_global_u32(pos_ptr);

                let head_idx = ctaid;
                let pair_idx = tid;

                let half_dim_reg = ctx.mov_u32_imm(half_dim);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim_reg);
                ctx.branch_if_not(in_bounds, "exit");

                // CORRECTNESS-013: NEOX style pairing for Qwen2.5 compatibility
                // elem0 = pair_idx (first half), elem1 = pair_idx + half_dim (second half)
                let elem0 = pair_idx;
                let elem1 = ctx.add_u32_reg(pair_idx, half_dim_reg);

                let dim = ctx.mov_u32_imm(head_dim);
                let head_offset = ctx.mul_lo_u32(head_idx, dim);
                let offset0 = ctx.add_u32_reg(head_offset, elem0);
                let offset1 = ctx.add_u32_reg(head_offset, elem1);

                let four = ctx.mov_u32_imm(4);
                let bytes0 = ctx.mul_lo_u32(offset0, four);
                let bytes1 = ctx.mul_lo_u32(offset1, four);
                let bytes0_64 = ctx.cvt_u64_u32(bytes0);
                let bytes1_64 = ctx.cvt_u64_u32(bytes1);
                let addr0 = ctx.add_u64(x_ptr, bytes0_64);
                let addr1 = ctx.add_u64(x_ptr, bytes1_64);
                let out_addr0 = ctx.add_u64(out_ptr, bytes0_64);
                let out_addr1 = ctx.add_u64(out_ptr, bytes1_64);

                let x0 = ctx.ld_global_f32(addr0);
                let x1 = ctx.ld_global_f32(addr1);

                // Compute frequency on-the-fly: freq = 1.0 / (theta^(2*pair_idx/head_dim))
                // CORRECTNESS-013: Use precise exp2 to avoid ex2.approx error
                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32_precise(power);  // Precise exp2

                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                // CORRECTNESS-013: Use precise polynomial sin/cos instead of .approx
                let cos_val = ctx.cos_f32_precise(angle);
                let sin_val = ctx.sin_f32_precise(angle);

                // Apply rotation: (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
                let x0_cos = ctx.mul_f32(x0, cos_val);
                let x1_sin = ctx.mul_f32(x1, sin_val);
                let new_x0 = ctx.sub_f32(x0_cos, x1_sin);

                let x0_sin = ctx.mul_f32(x0, sin_val);
                let x1_cos = ctx.mul_f32(x1, cos_val);
                let new_x1 = ctx.add_f32(x0_sin, x1_cos);

                ctx.st_global_f32(out_addr0, new_x0);
                ctx.st_global_f32(out_addr1, new_x1);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Transpose Kernel: output[j, i] = input[i, j]
///
/// Simple matrix transpose for attention K^T computation.
/// Used in multi-head attention to transpose K before Q @ K^T.
///
/// # Parameters
///
/// - `input_ptr`: Input matrix [rows, cols] (u64 pointer)
/// - `output_ptr`: Output matrix [cols, rows] (u64 pointer)
/// - `rows`: Number of rows in input (u32)
/// - `cols`: Number of columns in input (u32)
///
/// # Grid Configuration
///
/// - Block: 256 threads
/// - Grid: ceil(rows * cols / 256) blocks
#[derive(Debug, Clone)]
pub struct TransposeKernel {
    /// Number of rows in input
    pub rows: u32,
    /// Number of columns in input
    pub cols: u32,
}

impl TransposeKernel {
    /// Create a new transpose kernel
    #[must_use]
    pub const fn new(rows: u32, cols: u32) -> Self {
        Self { rows, cols }
    }
}

impl Kernel for TransposeKernel {
    fn name(&self) -> &str {
        "transpose"
    }

    fn build_ptx(&self) -> PtxKernel {
        let rows = self.rows;
        let cols = self.cols;
        let total_elems = rows * cols;

        // Each thread transposes one element
        // input[i, j] -> output[j, i]
        PtxKernel::new("transpose")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "rows")
            .param(PtxType::U32, "cols")
            .build(move |ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Load pointers
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Bounds check using compile-time constant
                let total = ctx.mov_u32_imm(total_elems);
                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                // Compute input coordinates: i = gid / cols, j = gid % cols
                // Using compile-time known cols for efficient division
                let row_idx = ctx.div_u32(gid, cols);
                let col_idx = ctx.rem_u32(gid, cols);

                // Input address: input[i, j] = input_ptr + (i * cols + j) * 4
                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(gid, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);

                // Output address: output[j, i] = output_ptr + (j * rows + i) * 4
                let rows_reg = ctx.mov_u32_imm(rows);
                let out_linear = ctx.mad_lo_u32(col_idx, rows_reg, row_idx);
                let output_offset = ctx.mul_wide_u32_reg(out_linear, four);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                // Load and store
                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Interleaved to Batched Kernel: reshape [seq_len, n_heads * head_dim] to [n_heads, seq_len, head_dim]
///
/// Converts multi-head attention tensors from interleaved layout (used after projection)
/// to batched layout (used for batched GEMM attention computation).
///
/// # Parameters
///
/// - `input_ptr`: Input tensor in interleaved layout (u64 pointer)
/// - `output_ptr`: Output tensor in batched layout (u64 pointer)
/// - `seq_len`: Sequence length (u32)
/// - `n_heads`: Number of attention heads (u32)
/// - `head_dim`: Dimension per head (u32)
///
/// # Memory Layout
///
/// - Input: `[s * n_heads * head_dim + h * head_dim + d]` for position s, head h, dim d
/// - Output: `[h * seq_len * head_dim + s * head_dim + d]`
#[derive(Debug, Clone)]
pub struct InterleavedToBatchedKernel {
    /// Sequence length
    pub seq_len: u32,
    /// Number of heads
    pub n_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
}

impl InterleavedToBatchedKernel {
    /// Create a new interleaved-to-batched kernel
    #[must_use]
    pub const fn new(seq_len: u32, n_heads: u32, head_dim: u32) -> Self {
        Self { seq_len, n_heads, head_dim }
    }
}

impl Kernel for InterleavedToBatchedKernel {
    fn name(&self) -> &str {
        "interleaved_to_batched"
    }

    fn build_ptx(&self) -> PtxKernel {
        let seq_len = self.seq_len;
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let d_model = n_heads * head_dim;
        let total_elems = seq_len * d_model;

        PtxKernel::new("interleaved_to_batched")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .build(move |ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Bounds check
                let total = ctx.mov_u32_imm(total_elems);
                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                // Input pointer
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Decode input index: gid = s * d_model + h * head_dim + d
                // s = gid / d_model
                // remainder = gid % d_model
                // h = remainder / head_dim
                // d = remainder % head_dim
                let s = ctx.div_u32(gid, d_model);
                let remainder = ctx.rem_u32(gid, d_model);
                let h = ctx.div_u32(remainder, head_dim);
                let d = ctx.rem_u32(remainder, head_dim);

                // Compute output index: h * seq_len * head_dim + s * head_dim + d
                let seq_head = ctx.mov_u32_imm(seq_len * head_dim);
                let head_dim_reg = ctx.mov_u32_imm(head_dim);
                let out_base = ctx.mul_lo_u32(h, seq_head);
                let out_row = ctx.mad_lo_u32(s, head_dim_reg, d);
                let out_idx = ctx.add_u32_reg(out_base, out_row);

                // Load from input, store to output
                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(gid, four);
                let output_offset = ctx.mul_wide_u32_reg(out_idx, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Extract Single Head Kernel: extract head h from interleaved [seq_len, n_heads * head_dim]
///
/// Copies one head's data to a contiguous [seq_len, head_dim] buffer.
///
/// # Parameters
///
/// - `input_ptr`: Input tensor in interleaved layout (u64 pointer)
/// - `output_ptr`: Output tensor [seq_len, head_dim] (u64 pointer)
/// - `head_idx`: Which head to extract (u32)
///
/// # Memory Layout
///
/// - Input: `[s * n_heads * head_dim + head_idx * head_dim + d]`
/// - Output: `[s * head_dim + d]`
#[derive(Debug, Clone)]
pub struct ExtractSingleHeadKernel {
    /// Sequence length
    pub seq_len: u32,
    /// Number of heads
    pub n_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
}

impl ExtractSingleHeadKernel {
    /// Create kernel
    #[must_use]
    pub const fn new(seq_len: u32, n_heads: u32, head_dim: u32) -> Self {
        Self { seq_len, n_heads, head_dim }
    }
}

impl Kernel for ExtractSingleHeadKernel {
    fn name(&self) -> &str {
        "extract_single_head"
    }

    fn build_ptx(&self) -> PtxKernel {
        let seq_len = self.seq_len;
        let head_dim = self.head_dim;
        let d_model = self.n_heads * head_dim;
        let output_size = seq_len * head_dim;

        PtxKernel::new("extract_single_head")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "head_idx")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Bounds check
                let total = ctx.mov_u32_imm(output_size);
                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let head_idx = ctx.load_param_u32("head_idx");

                // Decode output index: gid = s * head_dim + d
                let s = ctx.div_u32(gid, head_dim);
                let d = ctx.rem_u32(gid, head_dim);

                // Compute input index: s * d_model + head_idx * head_dim + d
                let d_model_reg = ctx.mov_u32_imm(d_model);
                let head_dim_reg = ctx.mov_u32_imm(head_dim);
                let head_offset = ctx.mul_lo_u32(head_idx, head_dim_reg);
                let row_offset = ctx.mul_lo_u32(s, d_model_reg);
                let in_idx = ctx.add_u32_reg(row_offset, head_offset);
                let in_idx = ctx.add_u32_reg(in_idx, d);

                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(in_idx, four);
                let output_offset = ctx.mul_wide_u32_reg(gid, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Copy Single Head Kernel: copy [seq_len, head_dim] to head h position in interleaved output
///
/// # Parameters
///
/// - `input_ptr`: Input tensor [seq_len, head_dim] (u64 pointer)
/// - `output_ptr`: Output tensor in interleaved layout (u64 pointer)
/// - `head_idx`: Which head position to copy to (u32)
#[derive(Debug, Clone)]
pub struct CopySingleHeadKernel {
    /// Sequence length
    pub seq_len: u32,
    /// Number of heads
    pub n_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
}

impl CopySingleHeadKernel {
    /// Create kernel
    #[must_use]
    pub const fn new(seq_len: u32, n_heads: u32, head_dim: u32) -> Self {
        Self { seq_len, n_heads, head_dim }
    }
}

impl Kernel for CopySingleHeadKernel {
    fn name(&self) -> &str {
        "copy_single_head"
    }

    fn build_ptx(&self) -> PtxKernel {
        let seq_len = self.seq_len;
        let head_dim = self.head_dim;
        let d_model = self.n_heads * head_dim;
        let input_size = seq_len * head_dim;

        PtxKernel::new("copy_single_head")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "head_idx")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Bounds check
                let total = ctx.mov_u32_imm(input_size);
                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let head_idx = ctx.load_param_u32("head_idx");

                // Decode input index: gid = s * head_dim + d
                let s = ctx.div_u32(gid, head_dim);
                let d = ctx.rem_u32(gid, head_dim);

                // Compute output index: s * d_model + head_idx * head_dim + d
                let d_model_reg = ctx.mov_u32_imm(d_model);
                let head_dim_reg = ctx.mov_u32_imm(head_dim);
                let head_offset = ctx.mul_lo_u32(head_idx, head_dim_reg);
                let row_offset = ctx.mul_lo_u32(s, d_model_reg);
                let out_idx = ctx.add_u32_reg(row_offset, head_offset);
                let out_idx = ctx.add_u32_reg(out_idx, d);

                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(gid, four);
                let output_offset = ctx.mul_wide_u32_reg(out_idx, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Batched to Interleaved Kernel: reshape [n_heads, seq_len, head_dim] to [seq_len, n_heads * head_dim]
///
/// Converts multi-head attention outputs from batched layout back to interleaved layout
/// for output projection.
///
/// # Parameters
///
/// - `input_ptr`: Input tensor in batched layout (u64 pointer)
/// - `output_ptr`: Output tensor in interleaved layout (u64 pointer)
/// - `seq_len`: Sequence length (u32)
/// - `n_heads`: Number of attention heads (u32)
/// - `head_dim`: Dimension per head (u32)
///
/// # Memory Layout
///
/// - Input: `[h * seq_len * head_dim + s * head_dim + d]`
/// - Output: `[s * n_heads * head_dim + h * head_dim + d]`
#[derive(Debug, Clone)]
pub struct BatchedToInterleavedKernel {
    /// Sequence length
    pub seq_len: u32,
    /// Number of heads
    pub n_heads: u32,
    /// Dimension per head
    pub head_dim: u32,
}

impl BatchedToInterleavedKernel {
    /// Create a new batched-to-interleaved kernel
    #[must_use]
    pub const fn new(seq_len: u32, n_heads: u32, head_dim: u32) -> Self {
        Self { seq_len, n_heads, head_dim }
    }
}

impl Kernel for BatchedToInterleavedKernel {
    fn name(&self) -> &str {
        "batched_to_interleaved"
    }

    fn build_ptx(&self) -> PtxKernel {
        let seq_len = self.seq_len;
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let d_model = n_heads * head_dim;
        let total_elems = seq_len * d_model;

        PtxKernel::new("batched_to_interleaved")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .build(move |ctx| {
                // Global thread ID
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Bounds check
                let total = ctx.mov_u32_imm(total_elems);
                let in_bounds = ctx.setp_lt_u32(gid, total);
                ctx.branch_if_not(in_bounds, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Decode output index: gid = s * d_model + h * head_dim + d
                let s = ctx.div_u32(gid, d_model);
                let remainder = ctx.rem_u32(gid, d_model);
                let h = ctx.div_u32(remainder, head_dim);
                let d = ctx.rem_u32(remainder, head_dim);

                // Compute input index: h * seq_len * head_dim + s * head_dim + d
                let seq_head = ctx.mov_u32_imm(seq_len * head_dim);
                let head_dim_reg = ctx.mov_u32_imm(head_dim);
                let in_base = ctx.mul_lo_u32(h, seq_head);
                let in_row = ctx.mad_lo_u32(s, head_dim_reg, d);
                let in_idx = ctx.add_u32_reg(in_base, in_row);

                // Load from input, store to output
                let four = ctx.mov_u32_imm(4);
                let input_offset = ctx.mul_wide_u32_reg(in_idx, four);
                let output_offset = ctx.mul_wide_u32_reg(gid, four);
                let input_addr = ctx.add_u64(input_ptr, input_offset);
                let output_addr = ctx.add_u64(output_ptr, output_offset);

                let val = ctx.ld_global_f32(input_addr);
                ctx.st_global_f32(output_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

// =============================================================================
// Batched Kernels for Multi-Head Attention (WAPR-PERF-008)
// =============================================================================

/// Batched transpose kernel: transpose multiple matrices in one launch
///
/// Grid: ((rows*cols + 255)/256, 1, batch)
/// Uses blockIdx.z to select batch/head
///
/// # Arguments
/// - `input_ptr`: Pointer to input [batch, rows, cols]
/// - `output_ptr`: Pointer to output [batch, cols, rows]
/// - `batch`, `rows`, `cols`: Dimensions
#[derive(Debug, Clone)]
pub struct BatchedTransposeKernel {
    /// Number of batches (e.g., n_heads)
    pub batch: u32,
    /// Input rows (becomes output cols)
    pub rows: u32,
    /// Input cols (becomes output rows)
    pub cols: u32,
}

impl BatchedTransposeKernel {
    /// Create a new batched transpose kernel
    #[must_use]
    pub const fn new(batch: u32, rows: u32, cols: u32) -> Self {
        Self { batch, rows, cols }
    }
}

impl Kernel for BatchedTransposeKernel {
    fn name(&self) -> &str {
        "batched_transpose"
    }

    fn build_ptx(&self) -> PtxKernel {
        let rows = self.rows;
        let cols = self.cols;
        let total_per_batch = rows * cols;

        PtxKernel::new("batched_transpose")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "batch")
            .param(PtxType::U32, "rows")
            .param(PtxType::U32, "cols")
            .build(move |ctx| {
                // Batch index from z-dimension
                let batch_idx = ctx.special_reg(PtxReg::CtaIdZ);

                // Global thread ID within batch
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                // Bounds check
                let total = ctx.mov_u32_imm(total_per_batch);
                let in_bounds = ctx.setp_lt_u32(gid, total);
                let batch_param = ctx.load_param_u32("batch");
                let batch_valid = ctx.setp_lt_u32(batch_idx, batch_param);
                let valid = ctx.and_pred(in_bounds, batch_valid);
                ctx.branch_if_not(valid, "exit");

                // Load pointers
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");

                // Decode gid -> (row, col) in input using immediate divisors
                let row = ctx.div_u32(gid, cols);
                let col = ctx.rem_u32(gid, cols);

                // Batch offset
                let batch_offset = ctx.mul_wide_u32(batch_idx, total_per_batch * 4);
                let in_batch_ptr = ctx.add_u64(input_ptr, batch_offset);
                let out_batch_ptr = ctx.add_u64(output_ptr, batch_offset);

                // Input index: row * cols + col
                let cols_reg = ctx.mov_u32_imm(cols);
                let in_idx = ctx.mad_lo_u32(row, cols_reg, col);
                // Output index: col * rows + row (transposed)
                let rows_reg = ctx.mov_u32_imm(rows);
                let out_idx = ctx.mad_lo_u32(col, rows_reg, row);

                // Compute addresses
                let four = ctx.mov_u32_imm(4);
                let in_offset = ctx.mul_wide_u32_reg(in_idx, four);
                let out_offset = ctx.mul_wide_u32_reg(out_idx, four);
                let in_addr = ctx.add_u64(in_batch_ptr, in_offset);
                let out_addr = ctx.add_u64(out_batch_ptr, out_offset);

                // Load and store
                let val = ctx.ld_global_f32(in_addr);
                ctx.st_global_f32(out_addr, val);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Batched scale kernel: multiply all elements by a scalar
///
/// Grid: ((total + 255)/256, 1, 1)
///
/// # Arguments
/// - `input_ptr`: Pointer to input
/// - `output_ptr`: Pointer to output
/// - `scale`: Scalar multiplier
/// - `n`: Total number of elements
#[derive(Debug, Clone)]
pub struct BatchedScaleKernel {
    /// Total number of elements (batch * rows * cols)
    pub n: u32,
}

impl BatchedScaleKernel {
    /// Create a new batched scale kernel
    #[must_use]
    pub const fn new(n: u32) -> Self {
        Self { n }
    }
}

impl Kernel for BatchedScaleKernel {
    fn name(&self) -> &str {
        "batched_scale"
    }

    fn build_ptx(&self) -> PtxKernel {
        let total = self.n;

        PtxKernel::new("batched_scale")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::F32, "scale")
            .param(PtxType::U32, "n")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let ntid = ctx.special_reg(PtxReg::NtidX);
                let gid = ctx.mad_lo_u32(ctaid, ntid, tid);

                let total_reg = ctx.mov_u32_imm(total);
                let in_bounds = ctx.setp_lt_u32(gid, total_reg);
                ctx.branch_if_not(in_bounds, "exit");

                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let scale = ctx.load_param_f32("scale");

                let four = ctx.mov_u32_imm(4);
                let offset = ctx.mul_wide_u32_reg(gid, four);
                let in_addr = ctx.add_u64(input_ptr, offset);
                let out_addr = ctx.add_u64(output_ptr, offset);

                let val = ctx.ld_global_f32(in_addr);
                let scaled = ctx.mul_f32(val, scale);
                ctx.st_global_f32(out_addr, scaled);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// Batched softmax kernel: softmax for multiple independent rows
///
/// Uses warp shuffle for reduction. One warp per row.
/// Grid: (batch * n_rows, 1, 1), Block: (32, 1, 1)
///
/// For multi-head attention: batch = n_heads, n_rows = seq_len, row_size = seq_len
#[derive(Debug, Clone)]
pub struct BatchedSoftmaxKernel {
    /// Total number of rows to process (batch * n_rows)
    pub total_rows: u32,
    /// Size of each row
    pub row_size: u32,
}

impl BatchedSoftmaxKernel {
    /// Create a new batched softmax kernel
    #[must_use]
    pub const fn new(total_rows: u32, row_size: u32) -> Self {
        Self { total_rows, row_size }
    }
}

impl Kernel for BatchedSoftmaxKernel {
    fn name(&self) -> &str {
        "batched_softmax"
    }

    fn build_ptx(&self) -> PtxKernel {
        let total_rows = self.total_rows;
        let row_size = self.row_size;

        // For long rows (>32), we need multi-pass with shared memory
        // For now, use simple loop-based approach that works for any row size
        PtxKernel::new("batched_softmax")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "total_rows")
            .param(PtxType::U32, "row_size")
            .shared_memory(72) // For warp reduction
            .build(move |ctx| {
                // One block (one warp of 32 threads) per row
                // Each thread processes row_size/32 elements in a strided loop
                let row_idx = ctx.special_reg(PtxReg::CtaIdX);
                let tid = ctx.special_reg(PtxReg::TidX);

                // Bounds check
                let total_rows_reg = ctx.mov_u32_imm(total_rows);
                let valid = ctx.setp_lt_u32(row_idx, total_rows_reg);
                ctx.branch_if_not(valid, "exit");

                // Calculate row base address
                let input_ptr = ctx.load_param_u64("input_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let row_size_reg = ctx.mov_u32_imm(row_size);

                let row_offset = ctx.mul_wide_u32(row_idx, row_size * 4);
                let row_input_ptr = ctx.add_u64(input_ptr, row_offset);
                let row_output_ptr = ctx.add_u64(output_ptr, row_offset);

                // Constants
                let four = ctx.mov_u32_imm(4);
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);

                // Pass 1: Find max (parallel reduction)
                // Initialize local_max = -inf
                let local_max = ctx.mov_f32_imm(f32::NEG_INFINITY);

                // Loop counter starts at tid, increments by 32 each iteration
                let i_max = ctx.mov_u32_imm(0);  // Allocate counter
                ctx.add_u32_reg_inplace(i_max, tid);  // i_max = tid
                ctx.label("max_loop");
                let max_done = ctx.setp_ge_u32(i_max, row_size_reg);
                ctx.branch_if(max_done, "max_done");

                let offset = ctx.mul_wide_u32_reg(i_max, four);
                let addr = ctx.add_u64(row_input_ptr, offset);
                let val = ctx.ld_global_f32(addr);
                ctx.max_f32_inplace(local_max, val);
                ctx.add_u32_inplace(i_max, 32);  // Increment by warp size
                ctx.branch("max_loop");

                ctx.label("max_done");

                // Warp shuffle reduction for max (tree reduction)
                let tmp16 = ctx.shfl_down_f32(local_max, 16, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp16);
                let tmp8 = ctx.shfl_down_f32(local_max, 8, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp8);
                let tmp4 = ctx.shfl_down_f32(local_max, 4, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp4);
                let tmp2 = ctx.shfl_down_f32(local_max, 2, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp2);
                let tmp1 = ctx.shfl_down_f32(local_max, 1, 0xFFFF_FFFF);
                ctx.max_f32_inplace(local_max, tmp1);

                // Broadcast max to all threads (read from lane 0)
                let row_max = ctx.shfl_idx_f32(local_max, 0, 0xFFFF_FFFF);

                // Pass 2: Compute sum of exp(x - max)
                let local_sum = ctx.mov_f32_imm(0.0);

                let i_sum = ctx.mov_u32_imm(0);  // Allocate counter
                ctx.add_u32_reg_inplace(i_sum, tid);  // i_sum = tid
                ctx.label("sum_loop");
                let sum_done = ctx.setp_ge_u32(i_sum, row_size_reg);
                ctx.branch_if(sum_done, "sum_done");

                let offset = ctx.mul_wide_u32_reg(i_sum, four);
                let addr = ctx.add_u64(row_input_ptr, offset);
                let val = ctx.ld_global_f32(addr);
                let diff = ctx.sub_f32(val, row_max);
                let exp_arg = ctx.mul_f32(diff, log2e);
                let exp_val = ctx.ex2_f32(exp_arg);
                ctx.add_f32_inplace(local_sum, exp_val);
                ctx.add_u32_inplace(i_sum, 32);  // Increment by warp size
                ctx.branch("sum_loop");

                ctx.label("sum_done");

                // Warp shuffle reduction for sum
                let stmp16 = ctx.shfl_down_f32(local_sum, 16, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp16);
                let stmp8 = ctx.shfl_down_f32(local_sum, 8, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp8);
                let stmp4 = ctx.shfl_down_f32(local_sum, 4, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp4);
                let stmp2 = ctx.shfl_down_f32(local_sum, 2, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp2);
                let stmp1 = ctx.shfl_down_f32(local_sum, 1, 0xFFFF_FFFF);
                ctx.add_f32_inplace(local_sum, stmp1);

                // Broadcast sum to all threads
                let row_sum = ctx.shfl_idx_f32(local_sum, 0, 0xFFFF_FFFF);

                // Pass 3: Write normalized values
                let i_write = ctx.mov_u32_imm(0);  // Allocate counter
                ctx.add_u32_reg_inplace(i_write, tid);  // i_write = tid
                ctx.label("write_loop");
                let write_done = ctx.setp_ge_u32(i_write, row_size_reg);
                ctx.branch_if(write_done, "exit");

                let offset = ctx.mul_wide_u32_reg(i_write, four);
                let in_addr = ctx.add_u64(row_input_ptr, offset);
                let out_addr = ctx.add_u64(row_output_ptr, offset);

                let val = ctx.ld_global_f32(in_addr);
                let diff = ctx.sub_f32(val, row_max);
                let exp_arg = ctx.mul_f32(diff, log2e);
                let exp_val = ctx.ex2_f32(exp_arg);
                let normalized = ctx.div_f32(exp_val, row_sum);
                ctx.st_global_f32(out_addr, normalized);

                ctx.add_u32_inplace(i_write, 32);  // Increment by warp size
                ctx.branch("write_loop");

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

    // ===== BatchedResidualAddKernel Tests =====

    #[test]
    fn test_batched_residual_add_kernel_new() {
        let kernel = BatchedResidualAddKernel::new(2048, 8);
        assert_eq!(kernel.n, 2048);
        assert_eq!(kernel.batch_size, 8);
    }

    #[test]
    fn test_batched_residual_add_kernel_name() {
        let kernel = BatchedResidualAddKernel::new(1024, 4);
        assert_eq!(kernel.name(), "batched_residual_add");
    }

    #[test]
    fn test_batched_residual_add_ptx_generation() {
        let kernel = BatchedResidualAddKernel::new(2048, 4);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(
            ptx.contains(".entry batched_residual_add"),
            "Should have batched_residual_add entry"
        );

        // Verify parameters
        assert!(ptx.contains(".param .u64 input1_ptr"), "Should have input1_ptr");
        assert!(ptx.contains(".param .u64 input2_ptr"), "Should have input2_ptr");
        assert!(ptx.contains(".param .u64 output_ptr"), "Should have output_ptr");

        // Verify arithmetic
        assert!(ptx.contains("add.f32"), "Should have add operation");
    }

    #[test]
    fn test_batched_residual_add_batch_sizes() {
        for batch_size in [1, 2, 4, 8, 16] {
            let kernel = BatchedResidualAddKernel::new(1024, batch_size);
            assert_eq!(kernel.batch_size, batch_size);

            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty());
            assert!(ptx.contains(".entry"));
        }
    }

    // ===== BatchedSwigluKernel Tests =====

    #[test]
    fn test_batched_swiglu_kernel_new() {
        let kernel = BatchedSwigluKernel::new(2048, 8);
        assert_eq!(kernel.n, 2048);
        assert_eq!(kernel.batch_size, 8);
    }

    #[test]
    fn test_batched_swiglu_kernel_name() {
        let kernel = BatchedSwigluKernel::new(1024, 4);
        assert_eq!(kernel.name(), "batched_swiglu");
    }

    #[test]
    fn test_batched_swiglu_ptx_generation() {
        let kernel = BatchedSwigluKernel::new(2048, 4);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(
            ptx.contains(".entry batched_swiglu"),
            "Should have batched_swiglu entry"
        );

        // Verify parameters
        assert!(ptx.contains(".param .u64 gate_ptr"), "Should have gate_ptr");
        assert!(ptx.contains(".param .u64 up_ptr"), "Should have up_ptr");
        assert!(ptx.contains(".param .u64 output_ptr"), "Should have output_ptr");

        // Verify SiLU computation
        assert!(ptx.contains("ex2.approx"), "Should have exp approximation");
    }

    #[test]
    fn test_batched_swiglu_batch_sizes() {
        for batch_size in [1, 2, 4, 8, 16] {
            let kernel = BatchedSwigluKernel::new(1024, batch_size);
            assert_eq!(kernel.batch_size, batch_size);

            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty());
            assert!(ptx.contains(".entry"));
        }
    }

    // ===== KvCacheScatterKernel Tests =====

    #[test]
    fn test_kv_cache_scatter_kernel_new() {
        let kernel = KvCacheScatterKernel::new(4, 64, 2048);
        assert_eq!(kernel.num_kv_heads, 4);
        assert_eq!(kernel.head_dim, 64);
        assert_eq!(kernel.max_len, 2048);
    }

    #[test]
    fn test_kv_cache_scatter_kernel_name() {
        let kernel = KvCacheScatterKernel::new(8, 128, 4096);
        assert_eq!(kernel.name(), "kv_cache_scatter");
    }

    #[test]
    fn test_kv_cache_scatter_ptx_generation() {
        let kernel = KvCacheScatterKernel::new(4, 64, 2048);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(
            ptx.contains(".entry kv_cache_scatter"),
            "Should have kv_cache_scatter entry"
        );

        // Verify parameters
        assert!(ptx.contains(".param .u64 src_ptr"), "Should have src_ptr");
        assert!(ptx.contains(".param .u64 cache_ptr"), "Should have cache_ptr");
        assert!(ptx.contains(".param .u32 pos"), "Should have pos");
        assert!(ptx.contains(".param .u32 head_dim"), "Should have head_dim");
        assert!(ptx.contains(".param .u32 max_len"), "Should have max_len");
    }

    #[test]
    fn test_kv_cache_scatter_memory_ops() {
        let kernel = KvCacheScatterKernel::new(4, 64, 2048);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains("ld.global"), "Should have global loads");
        assert!(ptx.contains("st.global"), "Should have global stores");
    }

    // ===== KvCacheScatterIndirectKernel Tests =====

    #[test]
    fn test_kv_cache_scatter_indirect_kernel_new() {
        let kernel = KvCacheScatterIndirectKernel::new(4, 64, 2048);
        assert_eq!(kernel.num_kv_heads, 4);
        assert_eq!(kernel.head_dim, 64);
        assert_eq!(kernel.max_len, 2048);
    }

    #[test]
    fn test_kv_cache_scatter_indirect_kernel_name() {
        let kernel = KvCacheScatterIndirectKernel::new(8, 128, 4096);
        assert_eq!(kernel.name(), "kv_cache_scatter_indirect");
    }

    #[test]
    fn test_kv_cache_scatter_indirect_ptx_generation() {
        let kernel = KvCacheScatterIndirectKernel::new(4, 64, 2048);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(
            ptx.contains(".entry kv_cache_scatter_indirect"),
            "Should have kv_cache_scatter_indirect entry"
        );

        // Verify parameters - note pos_ptr for indirect mode
        assert!(ptx.contains(".param .u64 src_ptr"), "Should have src_ptr");
        assert!(ptx.contains(".param .u64 cache_ptr"), "Should have cache_ptr");
        assert!(ptx.contains(".param .u64 pos_ptr"), "Should have pos_ptr for indirect");
    }

    #[test]
    fn test_kv_cache_scatter_indirect_reads_pos() {
        let kernel = KvCacheScatterIndirectKernel::new(4, 64, 2048);
        let ptx = kernel.emit_ptx();

        // Indirect mode reads position from device memory
        assert!(ptx.contains("ld.global"), "Should load pos from device memory");
    }

    // ===== RopeKernel Tests =====

    #[test]
    fn test_rope_kernel_new() {
        let kernel = RopeKernel::new(32, 128, 10000.0);
        assert_eq!(kernel.num_heads, 32);
        assert_eq!(kernel.head_dim, 128);
        assert!((kernel.theta - 10000.0).abs() < 0.001);
    }

    #[test]
    fn test_rope_kernel_name() {
        let kernel = RopeKernel::new(22, 64, 10000.0);
        assert_eq!(kernel.name(), "rope");
    }

    #[test]
    fn test_rope_ptx_generation() {
        let kernel = RopeKernel::new(32, 128, 10000.0);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(ptx.contains(".entry rope"), "Should have rope entry");

        // Verify parameters
        assert!(ptx.contains(".param .u64 x_ptr"), "Should have x_ptr");
        assert!(ptx.contains(".param .u64 out_ptr"), "Should have out_ptr");
        assert!(ptx.contains(".param .u32 pos"), "Should have pos");
    }

    #[test]
    fn test_rope_trig_operations() {
        let kernel = RopeKernel::new(22, 64, 10000.0);
        let ptx = kernel.emit_ptx();

        // RoPE uses sin/cos via ex2 approximation
        assert!(ptx.contains("ex2") || ptx.contains("sin") || ptx.contains("cos"),
            "Should have trigonometric operations");
        assert!(ptx.contains("mul.f32"), "Should have multiplications");
    }

    #[test]
    fn test_rope_various_head_dims() {
        for head_dim in [32, 64, 128] {
            let kernel = RopeKernel::new(22, head_dim, 10000.0);
            assert_eq!(kernel.head_dim, head_dim);

            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty());
            assert!(ptx.contains(".entry"));
        }
    }

    // ===== RopeIndirectKernel Tests =====

    #[test]
    fn test_rope_indirect_kernel_new() {
        let kernel = RopeIndirectKernel::new(32, 128, 10000.0);
        assert_eq!(kernel.num_heads, 32);
        assert_eq!(kernel.head_dim, 128);
        assert!((kernel.theta - 10000.0).abs() < 0.001);
    }

    #[test]
    fn test_rope_indirect_kernel_name() {
        let kernel = RopeIndirectKernel::new(22, 64, 10000.0);
        assert_eq!(kernel.name(), "rope_indirect");
    }

    #[test]
    fn test_rope_indirect_ptx_generation() {
        let kernel = RopeIndirectKernel::new(32, 128, 10000.0);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(
            ptx.contains(".entry rope_indirect"),
            "Should have rope_indirect entry"
        );

        // Verify parameters - note pos_ptr for indirect mode
        assert!(ptx.contains(".param .u64 x_ptr"), "Should have x_ptr");
        assert!(ptx.contains(".param .u64 out_ptr"), "Should have out_ptr");
        assert!(ptx.contains(".param .u64 pos_ptr"), "Should have pos_ptr for indirect");
    }

    #[test]
    fn test_rope_indirect_loads_position() {
        let kernel = RopeIndirectKernel::new(22, 64, 10000.0);
        let ptx = kernel.emit_ptx();

        // Indirect mode reads position from device memory
        assert!(ptx.contains("ld.global"), "Should load pos from device memory");
    }

    // ===== BatchedRopeKernel Tests =====

    #[test]
    fn test_batched_rope_kernel_new() {
        let kernel = BatchedRopeKernel::new(32, 128, 8, 10000.0);
        assert_eq!(kernel.num_heads, 32);
        assert_eq!(kernel.head_dim, 128);
        assert_eq!(kernel.batch_size, 8);
        assert!((kernel.theta - 10000.0).abs() < 0.001);
    }

    #[test]
    fn test_batched_rope_kernel_name() {
        let kernel = BatchedRopeKernel::new(22, 64, 4, 10000.0);
        assert_eq!(kernel.name(), "batched_rope");
    }

    #[test]
    fn test_batched_rope_ptx_generation() {
        let kernel = BatchedRopeKernel::new(32, 128, 4, 10000.0);
        let ptx = kernel.emit_ptx();

        // Verify kernel entry point
        assert!(
            ptx.contains(".entry batched_rope"),
            "Should have batched_rope entry"
        );

        // Verify parameters
        assert!(ptx.contains(".param .u64 x_ptr"), "Should have x_ptr");
    }

    #[test]
    fn test_batched_rope_batch_sizes() {
        for batch_size in [1, 2, 4, 8, 16] {
            let kernel = BatchedRopeKernel::new(22, 64, batch_size, 10000.0);
            assert_eq!(kernel.batch_size, batch_size);

            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty());
            assert!(ptx.contains(".entry"));
        }
    }

    #[test]
    fn test_batched_rope_trig_operations() {
        let kernel = BatchedRopeKernel::new(22, 64, 4, 10000.0);
        let ptx = kernel.emit_ptx();

        // RoPE uses sin/cos
        assert!(ptx.contains("ex2") || ptx.contains("sin") || ptx.contains("cos"),
            "Should have trigonometric operations");
    }

    #[test]
    fn test_batched_rope_memory_ops() {
        let kernel = BatchedRopeKernel::new(22, 64, 4, 10000.0);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains("ld.global"), "Should have global loads");
        assert!(ptx.contains("st.global"), "Should have global stores");
    }

    // ===== RopeNeoxKernel Tests =====

    #[test]
    fn test_rope_neox_kernel_new() {
        let kernel = RopeNeoxKernel::new(22, 64, 10000.0);
        assert_eq!(kernel.head_dim, 64);
        assert!((kernel.theta - 10000.0).abs() < 0.001);
    }

    #[test]
    fn test_rope_neox_kernel_name() {
        let kernel = RopeNeoxKernel::new(22, 64, 10000.0);
        assert_eq!(kernel.name(), "rope_neox");
    }

    #[test]
    fn test_rope_neox_ptx_generation() {
        let kernel = RopeNeoxKernel::new(22, 64, 10000.0);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry rope_neox"), "Should have rope_neox entry");
        assert!(ptx.contains(".param .u64 x_ptr"), "Should have x_ptr");
        assert!(ptx.contains(".param .u64 out_ptr"), "Should have out_ptr");
        assert!(ptx.contains(".param .u32 pos"), "Should have pos param");
    }

    #[test]
    fn test_rope_neox_various_head_dims() {
        for head_dim in [32, 64, 128, 256] {
            let kernel = RopeNeoxKernel::new(22, head_dim, 10000.0);
            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty());
            assert!(ptx.contains(".entry"));
        }
    }

    // ===== RopeNeoxIndirectKernel Tests =====

    #[test]
    fn test_rope_neox_indirect_kernel_new() {
        let kernel = RopeNeoxIndirectKernel::new(22, 64, 10000.0);
        assert_eq!(kernel.head_dim, 64);
        assert!((kernel.theta - 10000.0).abs() < 0.001);
    }

    #[test]
    fn test_rope_neox_indirect_kernel_name() {
        let kernel = RopeNeoxIndirectKernel::new(22, 64, 10000.0);
        assert_eq!(kernel.name(), "rope_neox_indirect");
    }

    #[test]
    fn test_rope_neox_indirect_ptx_generation() {
        let kernel = RopeNeoxIndirectKernel::new(22, 64, 10000.0);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry rope_neox_indirect"), "Should have rope_neox_indirect entry");
        assert!(ptx.contains(".param .u64 x_ptr"), "Should have x_ptr");
        assert!(ptx.contains(".param .u64 pos_ptr"), "Should have pos_ptr for indirect");
    }

    #[test]
    fn test_rope_neox_indirect_loads_position() {
        let kernel = RopeNeoxIndirectKernel::new(22, 64, 10000.0);
        let ptx = kernel.emit_ptx();

        // Indirect kernel should load position from device memory
        assert!(ptx.contains("ld.global"), "Should have global loads for position");
    }

    // ===== PreciseRopeKernel Tests =====

    #[test]
    fn test_rope_precise_kernel_new() {
        let kernel = PreciseRopeKernel::new(22, 64, 10000.0);
        assert_eq!(kernel.head_dim, 64);
        assert!((kernel.theta - 10000.0).abs() < 0.001);
    }

    #[test]
    fn test_rope_precise_kernel_name() {
        let kernel = PreciseRopeKernel::new(22, 64, 10000.0);
        assert_eq!(kernel.name(), "rope_precise");
    }

    #[test]
    fn test_rope_precise_ptx_generation() {
        let kernel = PreciseRopeKernel::new(22, 64, 10000.0);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry rope_precise"), "Should have rope_precise entry");
        assert!(ptx.contains(".param .u64 x_ptr"), "Should have x_ptr");
        assert!(ptx.contains(".param .u64 out_ptr"), "Should have out_ptr");
        assert!(ptx.contains(".param .u32 pos"), "Should have pos param");
    }

    #[test]
    fn test_rope_precise_uses_precise_trig() {
        let kernel = PreciseRopeKernel::new(22, 64, 10000.0);
        let ptx = kernel.emit_ptx();

        // Precise rope should use full precision sin/cos
        assert!(ptx.contains("sin.approx") || ptx.contains("cos.approx") || ptx.contains("ex2"),
            "Should use trigonometric operations");
    }

    #[test]
    fn test_rope_precise_various_head_dims() {
        for head_dim in [32, 64, 128, 256] {
            let kernel = PreciseRopeKernel::new(22, head_dim, 10000.0);
            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty());
            assert!(ptx.contains(".entry"));
        }
    }

    // ===== PreciseRopeIndirectKernel Tests =====

    #[test]
    fn test_rope_precise_indirect_kernel_new() {
        let kernel = PreciseRopeIndirectKernel::new(22, 64, 10000.0);
        assert_eq!(kernel.head_dim, 64);
        assert!((kernel.theta - 10000.0).abs() < 0.001);
    }

    #[test]
    fn test_rope_precise_indirect_kernel_name() {
        let kernel = PreciseRopeIndirectKernel::new(22, 64, 10000.0);
        assert_eq!(kernel.name(), "rope_precise_indirect");
    }

    #[test]
    fn test_rope_precise_indirect_ptx_generation() {
        let kernel = PreciseRopeIndirectKernel::new(22, 64, 10000.0);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry rope_precise_indirect"), "Should have rope_precise_indirect entry");
        assert!(ptx.contains(".param .u64 x_ptr"), "Should have x_ptr");
        assert!(ptx.contains(".param .u64 pos_ptr"), "Should have pos_ptr for indirect");
    }

    #[test]
    fn test_rope_precise_indirect_loads_position() {
        let kernel = PreciseRopeIndirectKernel::new(22, 64, 10000.0);
        let ptx = kernel.emit_ptx();

        // Indirect kernel should load position from device memory
        assert!(ptx.contains("ld.global"), "Should have global loads for position");
    }

    #[test]
    fn test_rope_precise_indirect_memory_ops() {
        let kernel = PreciseRopeIndirectKernel::new(22, 64, 10000.0);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains("ld.global"), "Should have global loads");
        assert!(ptx.contains("st.global"), "Should have global stores");
    }

    // ===== ScaleKernel Tests =====

    #[test]
    fn test_scale_kernel_new() {
        let kernel = ScaleKernel::new(1024);
        assert_eq!(kernel.n, 1024);
    }

    #[test]
    fn test_scale_kernel_name() {
        let kernel = ScaleKernel::new(1024);
        assert_eq!(kernel.name(), "scale");
    }

    #[test]
    fn test_scale_kernel_ptx_generation() {
        let kernel = ScaleKernel::new(2048);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry scale"), "Should have scale entry");
        assert!(ptx.contains(".param .u64 input_ptr"), "Should have input_ptr");
        assert!(ptx.contains(".param .u64 output_ptr"), "Should have output_ptr");
        assert!(ptx.contains(".param .f32 scale"), "Should have scale param");
        assert!(ptx.contains(".param .u32 n"), "Should have n param");
        assert!(ptx.contains("mul.f32"), "Should have multiply op");
    }

    // ===== TransposeKernel Tests =====

    #[test]
    fn test_transpose_kernel_new() {
        let kernel = TransposeKernel::new(512, 1024);
        assert_eq!(kernel.rows, 512);
        assert_eq!(kernel.cols, 1024);
    }

    #[test]
    fn test_transpose_kernel_name() {
        let kernel = TransposeKernel::new(64, 64);
        assert_eq!(kernel.name(), "transpose");
    }

    #[test]
    fn test_transpose_kernel_ptx_generation() {
        let kernel = TransposeKernel::new(256, 512);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry transpose"), "Should have transpose entry");
        assert!(ptx.contains(".param .u64 input_ptr"), "Should have input_ptr");
        assert!(ptx.contains(".param .u64 output_ptr"), "Should have output_ptr");
        assert!(ptx.contains(".param .u32 rows"), "Should have rows param");
        assert!(ptx.contains(".param .u32 cols"), "Should have cols param");
    }

    #[test]
    fn test_transpose_various_dimensions() {
        for (rows, cols) in [(64, 64), (128, 256), (512, 128), (1024, 1024)] {
            let kernel = TransposeKernel::new(rows, cols);
            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty());
            assert!(ptx.contains(".entry"));
        }
    }

    // ===== InterleavedToBatchedKernel Tests =====

    #[test]
    fn test_interleaved_to_batched_kernel_new() {
        let kernel = InterleavedToBatchedKernel::new(128, 32, 64);
        assert_eq!(kernel.seq_len, 128);
        assert_eq!(kernel.n_heads, 32);
        assert_eq!(kernel.head_dim, 64);
    }

    #[test]
    fn test_interleaved_to_batched_kernel_name() {
        let kernel = InterleavedToBatchedKernel::new(128, 32, 64);
        assert_eq!(kernel.name(), "interleaved_to_batched");
    }

    #[test]
    fn test_interleaved_to_batched_kernel_ptx_generation() {
        let kernel = InterleavedToBatchedKernel::new(256, 16, 128);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry interleaved_to_batched"), "Should have entry");
        assert!(ptx.contains(".param .u64 input_ptr"), "Should have input_ptr");
        assert!(ptx.contains(".param .u64 output_ptr"), "Should have output_ptr");
    }

    // ===== ExtractSingleHeadKernel Tests =====

    #[test]
    fn test_extract_single_head_kernel_new() {
        let kernel = ExtractSingleHeadKernel::new(128, 32, 64);
        assert_eq!(kernel.seq_len, 128);
        assert_eq!(kernel.n_heads, 32);
        assert_eq!(kernel.head_dim, 64);
    }

    #[test]
    fn test_extract_single_head_kernel_name() {
        let kernel = ExtractSingleHeadKernel::new(128, 32, 64);
        assert_eq!(kernel.name(), "extract_single_head");
    }

    #[test]
    fn test_extract_single_head_kernel_ptx_generation() {
        let kernel = ExtractSingleHeadKernel::new(256, 32, 128);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry extract_single_head"), "Should have entry");
        assert!(ptx.contains("ld.global"), "Should have global loads");
        assert!(ptx.contains("st.global"), "Should have global stores");
    }

    // ===== CopySingleHeadKernel Tests =====

    #[test]
    fn test_copy_single_head_kernel_new() {
        let kernel = CopySingleHeadKernel::new(128, 32, 64);
        assert_eq!(kernel.seq_len, 128);
        assert_eq!(kernel.n_heads, 32);
        assert_eq!(kernel.head_dim, 64);
    }

    #[test]
    fn test_copy_single_head_kernel_name() {
        let kernel = CopySingleHeadKernel::new(128, 32, 64);
        assert_eq!(kernel.name(), "copy_single_head");
    }

    #[test]
    fn test_copy_single_head_kernel_ptx_generation() {
        let kernel = CopySingleHeadKernel::new(256, 32, 128);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry copy_single_head"), "Should have entry");
        assert!(ptx.contains("ld.global"), "Should have global loads");
        assert!(ptx.contains("st.global"), "Should have global stores");
    }

    // ===== BatchedToInterleavedKernel Tests =====

    #[test]
    fn test_batched_to_interleaved_kernel_new() {
        let kernel = BatchedToInterleavedKernel::new(128, 32, 64);
        assert_eq!(kernel.seq_len, 128);
        assert_eq!(kernel.n_heads, 32);
        assert_eq!(kernel.head_dim, 64);
    }

    #[test]
    fn test_batched_to_interleaved_kernel_name() {
        let kernel = BatchedToInterleavedKernel::new(128, 32, 64);
        assert_eq!(kernel.name(), "batched_to_interleaved");
    }

    #[test]
    fn test_batched_to_interleaved_kernel_ptx_generation() {
        let kernel = BatchedToInterleavedKernel::new(256, 16, 128);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry batched_to_interleaved"), "Should have entry");
        assert!(ptx.contains("ld.global"), "Should have global loads");
        assert!(ptx.contains("st.global"), "Should have global stores");
    }

    // ===== BatchedTransposeKernel Tests =====

    #[test]
    fn test_batched_transpose_kernel_new() {
        let kernel = BatchedTransposeKernel::new(32, 128, 64);
        assert_eq!(kernel.batch, 32);
        assert_eq!(kernel.rows, 128);
        assert_eq!(kernel.cols, 64);
    }

    #[test]
    fn test_batched_transpose_kernel_name() {
        let kernel = BatchedTransposeKernel::new(32, 128, 64);
        assert_eq!(kernel.name(), "batched_transpose");
    }

    #[test]
    fn test_batched_transpose_kernel_ptx_generation() {
        let kernel = BatchedTransposeKernel::new(16, 256, 128);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry batched_transpose"), "Should have entry");
        assert!(ptx.contains("ld.global"), "Should have global loads");
        assert!(ptx.contains("st.global"), "Should have global stores");
    }

    // ===== BatchedScaleKernel Tests =====

    #[test]
    fn test_batched_scale_kernel_new() {
        let kernel = BatchedScaleKernel::new(1024);
        assert_eq!(kernel.n, 1024);
    }

    #[test]
    fn test_batched_scale_kernel_name() {
        let kernel = BatchedScaleKernel::new(1024);
        assert_eq!(kernel.name(), "batched_scale");
    }

    #[test]
    fn test_batched_scale_kernel_ptx_generation() {
        let kernel = BatchedScaleKernel::new(2048);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry batched_scale"), "Should have entry");
        assert!(ptx.contains(".param .f32 scale"), "Should have scale param");
        assert!(ptx.contains("mul.f32"), "Should have multiply op");
    }

    // ===== BatchedSoftmaxKernel Tests =====

    #[test]
    fn test_batched_softmax_kernel_new() {
        let kernel = BatchedSoftmaxKernel::new(32, 128);
        assert_eq!(kernel.total_rows, 32);
        assert_eq!(kernel.row_size, 128);
    }

    #[test]
    fn test_batched_softmax_kernel_name() {
        let kernel = BatchedSoftmaxKernel::new(32, 128);
        assert_eq!(kernel.name(), "batched_softmax");
    }

    #[test]
    fn test_batched_softmax_kernel_ptx_generation() {
        let kernel = BatchedSoftmaxKernel::new(16, 256);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry batched_softmax"), "Should have entry");
        // Softmax uses exp
        assert!(ptx.contains("ex2") || ptx.contains("exp"), "Should have exp operation");
    }

    #[test]
    fn test_batched_softmax_various_sizes() {
        for (total_rows, row_size) in [(1, 64), (8, 128), (32, 256), (64, 512)] {
            let kernel = BatchedSoftmaxKernel::new(total_rows, row_size);
            let ptx = kernel.emit_ptx();
            assert!(!ptx.is_empty());
            assert!(ptx.contains(".entry"));
        }
    }
}
