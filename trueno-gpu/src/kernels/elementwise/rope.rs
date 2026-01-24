//! RoPE (Rotary Position Embedding) Kernels
//!
//! GPU kernels for rotary position embeddings in transformer models.
//!
//! ## Kernel Variants
//!
//! - `RopeKernel`: Standard adjacent-pair RoPE
//! - `RopeIndirectKernel`: CUDA Graph compatible version
//! - `RopeNeoxKernel`: NEOX/GPT-NeoX style (split halves)
//! - `RopeNeoxIndirectKernel`: CUDA Graph compatible NEOX version
//! - `BatchedRopeKernel`: Multi-sequence batched RoPE
//! - `PreciseRopeKernel`: High-precision for theta=1M (Qwen2.5)
//! - `PreciseRopeIndirectKernel`: Precise + CUDA Graph compatible

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

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
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "out_ptr")
            .param(PtxType::U32, "pos")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let pos = ctx.load_param_u32("pos");

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

                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32(power);

                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                let cos_val = ctx.cos_f32(angle);
                let sin_val = ctx.sin_f32(angle);

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
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "out_ptr")
            .param(PtxType::U64, "pos_ptr")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let pos_ptr = ctx.load_param_u64("pos_ptr");

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

                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32(power);

                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                let cos_val = ctx.cos_f32(angle);
                let sin_val = ctx.sin_f32(angle);

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
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "out_ptr")
            .param(PtxType::U32, "pos")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let pos = ctx.load_param_u32("pos");

                let head_idx = ctaid;
                let pair_idx = tid;

                let half_dim_reg = ctx.mov_u32_imm(half_dim);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim_reg);
                ctx.branch_if_not(in_bounds, "exit");

                // NEOX style: elem0 = pair_idx, elem1 = pair_idx + half_dim
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

                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32(power);

                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                let cos_val = ctx.cos_f32(angle);
                let sin_val = ctx.sin_f32(angle);

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
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "out_ptr")
            .param(PtxType::U64, "pos_ptr")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let pos_ptr = ctx.load_param_u64("pos_ptr");

                let pos = ctx.ld_global_u32(pos_ptr);

                let head_idx = ctaid;
                let pair_idx = tid;

                let half_dim_reg = ctx.mov_u32_imm(half_dim);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim_reg);
                ctx.branch_if_not(in_bounds, "exit");

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

                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32(power);

                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                let cos_val = ctx.cos_f32(angle);
                let sin_val = ctx.sin_f32(angle);

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
/// # Grid Configuration
///
/// - Grid: (num_heads, batch_size, 1)
/// - Block: (head_dim / 2, 1, 1)
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
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "out_ptr")
            .param(PtxType::U64, "positions_ptr")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);

                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let positions_ptr = ctx.load_param_u64("positions_ptr");

                let pair_idx = tid;

                let half_dim = ctx.mov_u32_imm(head_dim / 2);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim);
                ctx.branch_if_not(in_bounds, "exit");

                let four = ctx.mov_u32_imm(4);
                let pos_byte_offset = ctx.mul_lo_u32(batch_idx, four);
                let pos_byte_offset_64 = ctx.cvt_u64_u32(pos_byte_offset);
                let pos_addr = ctx.add_u64(positions_ptr, pos_byte_offset_64);
                let pos = ctx.ld_global_u32(pos_addr);

                let two = ctx.mov_u32_imm(2);
                let elem0 = ctx.mul_lo_u32(pair_idx, two);
                let one = ctx.mov_u32_imm(1);
                let elem1 = ctx.add_u32_reg(elem0, one);

                let heads_per_seq = ctx.mov_u32_imm(num_heads);
                let dim = ctx.mov_u32_imm(head_dim);
                let seq_stride = ctx.mul_lo_u32(heads_per_seq, dim);
                let batch_offset = ctx.mul_lo_u32(batch_idx, seq_stride);
                let head_offset = ctx.mul_lo_u32(head_idx, dim);
                let base_offset = ctx.add_u32_reg(batch_offset, head_offset);
                let offset0 = ctx.add_u32_reg(base_offset, elem0);
                let offset1 = ctx.add_u32_reg(base_offset, elem1);

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

                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32(power);

                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                let cos_val = ctx.cos_f32(angle);
                let sin_val = ctx.sin_f32(angle);

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

/// CORRECTNESS-013: Precise RoPE Kernel for CPU/GPU bit-exactness
///
/// Uses polynomial sin/cos approximations instead of hardware `sin.approx.f32`
/// and `cos.approx.f32` which have ~2^-21 error. For Qwen 2.5 with theta=1M,
/// the high-frequency components are very sensitive to trig precision.
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

                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32(power);

                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                // CORRECTNESS-013: Use precise polynomial sin/cos
                let cos_val = ctx.cos_f32_precise(angle);
                let sin_val = ctx.sin_f32_precise(angle);

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

/// CORRECTNESS-013: Precise RoPE Indirect Kernel for CUDA graph compatibility
///
/// Same as PreciseRopeKernel but reads position from a GPU buffer.
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
            .param(PtxType::U64, "pos_ptr")
            .build(move |ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid = ctx.special_reg(PtxReg::CtaIdX);
                let x_ptr = ctx.load_param_u64("x_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let pos_ptr = ctx.load_param_u64("pos_ptr");

                let pos = ctx.ld_global_u32(pos_ptr);

                let head_idx = ctaid;
                let pair_idx = tid;

                let half_dim_reg = ctx.mov_u32_imm(half_dim);
                let in_bounds = ctx.setp_lt_u32(pair_idx, half_dim_reg);
                ctx.branch_if_not(in_bounds, "exit");

                // NEOX style pairing for Qwen2.5 compatibility
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

                let pair_f32 = ctx.cvt_f32_u32(pair_idx);
                let dim_f32 = ctx.mov_f32_imm(head_dim as f32);
                let neg_two = ctx.mov_f32_imm(-2.0);
                let exponent = ctx.mul_f32(pair_f32, neg_two);
                let exponent_scaled = ctx.div_f32(exponent, dim_f32);
                let log2_theta = ctx.mov_f32_imm(theta.log2());
                let power = ctx.mul_f32(exponent_scaled, log2_theta);
                let freq_base = ctx.ex2_f32_precise(power);

                let pos_f32 = ctx.cvt_f32_u32(pos);
                let angle = ctx.mul_f32(pos_f32, freq_base);

                let cos_val = ctx.cos_f32_precise(angle);
                let sin_val = ctx.sin_f32_precise(angle);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_kernel_name() {
        let kernel = RopeKernel::new(32, 64, 10000.0);
        assert_eq!(kernel.name(), "rope");
    }

    #[test]
    fn test_rope_ptx_generation() {
        let kernel = RopeKernel::new(32, 64, 10000.0);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry rope"));
        assert!(ptx.contains("sin.approx.f32"));
        assert!(ptx.contains("cos.approx.f32"));
    }

    #[test]
    fn test_rope_indirect_kernel_name() {
        let kernel = RopeIndirectKernel::new(32, 64, 10000.0);
        assert_eq!(kernel.name(), "rope_indirect");
    }

    #[test]
    fn test_rope_indirect_ptx_generation() {
        let kernel = RopeIndirectKernel::new(32, 64, 10000.0);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry rope_indirect"));
        assert!(ptx.contains(".param .u64 pos_ptr"));
        assert!(ptx.contains("ld.global.u32"));
    }

    #[test]
    fn test_rope_neox_kernel_name() {
        let kernel = RopeNeoxKernel::new(32, 64, 1_000_000.0);
        assert_eq!(kernel.name(), "rope_neox");
    }

    #[test]
    fn test_rope_neox_indirect_kernel_name() {
        let kernel = RopeNeoxIndirectKernel::new(32, 64, 1_000_000.0);
        assert_eq!(kernel.name(), "rope_neox_indirect");
    }

    #[test]
    fn test_batched_rope_kernel_name() {
        let kernel = BatchedRopeKernel::new(32, 64, 4, 10000.0);
        assert_eq!(kernel.name(), "batched_rope");
    }

    #[test]
    fn test_batched_rope_ptx_generation() {
        let kernel = BatchedRopeKernel::new(32, 64, 4, 10000.0);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry batched_rope"));
        assert!(ptx.contains(".param .u64 positions_ptr"));
    }

    #[test]
    fn test_precise_rope_kernel_name() {
        let kernel = PreciseRopeKernel::new(32, 64, 1_000_000.0);
        assert_eq!(kernel.name(), "rope_precise");
    }

    #[test]
    fn test_precise_rope_indirect_kernel_name() {
        let kernel = PreciseRopeIndirectKernel::new(32, 64, 1_000_000.0);
        assert_eq!(kernel.name(), "rope_precise_indirect");
    }
}
