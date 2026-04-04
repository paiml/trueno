//! Fused QKV hardware DP4A Q4K GEMV kernel (stub).
//!
//! Uses `selp_u64` to conditionally select between Q/K/V weight pointers.
//! Full implementation pending (PMAT-053).

#![allow(unused_imports)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// Fused QKV DP4A kernel — selects Q/K/V weights via predicated pointer.
#[derive(Debug, Clone)]
pub struct FusedQKVHwDp4aQ4KGemvKernel {
    /// Hidden dimension
    pub hidden_dim: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Number of Q heads
    pub num_q_heads: u32,
    /// Number of KV heads
    pub num_kv_heads: u32,
}

impl FusedQKVHwDp4aQ4KGemvKernel {
    /// Create a new fused QKV DP4A kernel.
    #[must_use]
    pub fn new(hidden_dim: u32, head_dim: u32, num_q_heads: u32, num_kv_heads: u32) -> Self {
        Self { hidden_dim, head_dim, num_q_heads, num_kv_heads }
    }
}

impl Kernel for FusedQKVHwDp4aQ4KGemvKernel {
    fn name(&self) -> &str {
        "fused_qkv_hw_dp4a_q4k_gemv"
    }

    fn build_ptx(&self) -> PtxKernel {
        // Stub: generates a no-op kernel. Full implementation in PMAT-053.
        PtxKernel::new("fused_qkv_hw_dp4a_q4k_gemv")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "wq_ptr")
            .param(PtxType::U64, "wk_ptr")
            .param(PtxType::U64, "wv_ptr")
            .param(PtxType::U64, "y_q_ptr")
            .param(PtxType::U64, "y_k_ptr")
            .param(PtxType::U64, "y_v_ptr")
            .param(PtxType::U32, "hidden_dim")
            .build(|ctx| {
                ctx.ret();
            })
    }
}
