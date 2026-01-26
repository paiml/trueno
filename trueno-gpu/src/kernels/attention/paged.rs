//! Paged/Incremental Attention Kernels for VRAM-bound block management.
//!
//! This module implements incremental attention kernels optimized for
//! autoregressive LLM decoding. Unlike FlashAttention which tiles SRAM,
//! these kernels manage GPU-resident KV caches with efficient block access.
//!
//! ## Kernels
//!
//! - **IncrementalAttentionKernel**: Single-query (M=1) autoregressive attention
//! - **MultiWarpIncrementalAttentionKernel**: Multi-warp version for larger sequences
//! - **BatchedIncrementalAttentionKernel**: Batched incremental attention
//! - **FlashDecodingChunkKernel**: Split-K parallel decoding chunks
//! - **FlashDecodingReduceKernel**: Reduction kernel for Flash Decoding
//!
//! ## References
//!
//! - [Kwon2023] PagedAttention for LLM Serving with vLLM
//! - Flash Decoding (Split-K) for parallel sequence processing

#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

// =============================================================================
// PAR-020: Incremental Attention Kernel for M=1 Autoregressive Decoding
// =============================================================================

/// Incremental attention kernel for single-query autoregressive decoding (PAR-020)
///
/// Optimized for the critical path of LLM token generation where each new token
/// requires attention over the entire KV cache with a single query vector.
///
/// # Memory Layout
///
/// - Q: [head_dim] - single query vector for current position
/// - K: [seq_len, head_dim] - cached keys (GPU-resident)
/// - V: [seq_len, head_dim] - cached values (GPU-resident)
/// - Output: [head_dim] - weighted sum of values
///
/// # Algorithm
///
/// 1. Compute attention scores: score[i] = dot(Q, K[i]) * scale
/// 2. Apply causal mask (positions > current are masked)
/// 3. Online softmax: max_score, sum_exp tracked incrementally
/// 4. Compute weighted V sum: output = sum(softmax[i] * V[i])
///
/// # Performance
///
/// - Avoids materializing [seq_len, seq_len] attention matrix
/// - Uses warp shuffle for efficient parallel reduction
/// - Designed for GPU-resident KV cache (no D2H transfer)
/// - Target: O(seq_len * head_dim) memory, O(seq_len * head_dim) compute
#[derive(Debug, Clone)]
pub struct IncrementalAttentionKernel {
    /// Maximum sequence length to support
    pub max_seq_len: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Number of key-value heads (for GQA, <= num_heads)
    pub num_kv_heads: u32,
    /// Scaling factor for attention scores (1/sqrt(head_dim))
    pub scale: f32,
    /// PAR-061: Read seq_len from device memory (for CUDA graph compatibility)
    pub indirect_seq_len: bool,
}

impl IncrementalAttentionKernel {
    /// Create new incremental attention kernel (MHA - num_kv_heads = num_heads)
    ///
    /// # Arguments
    ///
    /// * `max_seq_len` - Maximum sequence length to support
    /// * `head_dim` - Dimension per attention head
    /// * `num_heads` - Number of attention heads
    #[must_use]
    pub fn new(max_seq_len: u32, head_dim: u32, num_heads: u32) -> Self {
        Self::with_gqa(max_seq_len, head_dim, num_heads, num_heads)
    }

    /// Create new incremental attention kernel with GQA support (PAR-021)
    ///
    /// # Arguments
    ///
    /// * `max_seq_len` - Maximum sequence length to support
    /// * `head_dim` - Dimension per attention head
    /// * `num_heads` - Number of query attention heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    #[must_use]
    pub fn with_gqa(max_seq_len: u32, head_dim: u32, num_heads: u32, num_kv_heads: u32) -> Self {
        Self {
            max_seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            scale: 1.0 / (head_dim as f32).sqrt(),
            indirect_seq_len: false,
        }
    }

    /// PAR-061: Enable indirect seq_len mode (reads from device memory)
    /// Required for CUDA graph compatibility
    #[must_use]
    pub fn with_indirect_seq_len(mut self, indirect: bool) -> Self {
        self.indirect_seq_len = indirect;
        self
    }

    /// Check if this kernel is configured for GQA
    #[must_use]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads != self.num_heads
    }
}

impl Kernel for IncrementalAttentionKernel {
    fn name(&self) -> &str {
        // PAR-061: Different kernel name for indirect mode
        if self.indirect_seq_len {
            "incremental_attention_indirect"
        } else {
            "incremental_attention"
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let scale = self.scale;
        let max_seq_len = self.max_seq_len;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let indirect = self.indirect_seq_len;

        // Kernel strategy (PAR-020 + PAR-021 GQA):
        // - Grid: (num_heads, 1, 1) - one block per Q head
        // - Block: (32, 1, 1) - one warp per block
        // - Each warp computes attention for one Q head using online softmax
        //
        // Memory layout:
        // - q: [num_heads, head_dim] - query vectors for current position
        // - k: [num_kv_heads, max_seq_len, head_dim] - key cache (GPU-resident)
        // - v: [num_kv_heads, max_seq_len, head_dim] - value cache (GPU-resident)
        // - output: [num_heads, head_dim] - attention output
        //
        // GQA mapping (PAR-021):
        // - Each Q head uses kv_head_idx = q_head_idx * num_kv_heads / num_heads
        // - For MHA: kv_head_idx = q_head_idx
        // - For GQA: multiple Q heads share the same KV head
        //
        // Algorithm:
        // 1. Thread i loads Q[lane_id], Q[lane_id+32], ... (strided)
        // 2. Loop over seq positions, computing Q·K dot product per position
        // 3. Warp-reduce dot product using shfl_down
        // 4. Online softmax: track running max and sum_exp
        // 5. Accumulate weighted V vectors
        // 6. Normalize and store output

        // PAR-061: Use different kernel name and parameter type for indirect mode
        let kernel_name = if indirect {
            "incremental_attention_indirect"
        } else {
            "incremental_attention"
        };

        let mut builder = PtxKernel::new(kernel_name)
            .param(PtxType::U64, "q_ptr")
            .param(PtxType::U64, "k_ptr")
            .param(PtxType::U64, "v_ptr")
            .param(PtxType::U64, "out_ptr");

        // PAR-061: Indirect mode takes seq_len_ptr (U64), direct mode takes seq_len (U32)
        builder = if indirect {
            builder.param(PtxType::U64, "seq_len_ptr")
        } else {
            builder.param(PtxType::U32, "seq_len")
        };

        builder
            .shared_memory(0) // Register-only, warp shuffle for reduction
            .build(move |ctx| {
                // Get indices
                let q_head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let lane_id = ctx.special_reg(PtxReg::TidX);

                // Load parameters
                // PAR-061: In indirect mode, load seq_len from device memory
                let seq_len = if indirect {
                    let seq_len_ptr = ctx.load_param_u64("seq_len_ptr");
                    ctx.ld_global_u32(seq_len_ptr)
                } else {
                    ctx.load_param_u32("seq_len")
                };
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptr = ctx.load_param_u64("k_ptr");
                let v_ptr = ctx.load_param_u64("v_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");

                // Pre-compute constants
                let four = ctx.mov_u32_imm(4);
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);

                // Compute Q/output head offset
                // Q/output: q_head_idx * head_dim
                let q_head_off = ctx.mul_lo_u32(q_head_idx, head_dim_u32);
                let q_head_off_bytes = ctx.mul_wide_u32_reg(q_head_off, four);
                let q_head_ptr = ctx.add_u64(q_ptr, q_head_off_bytes);
                let out_head_ptr = ctx.add_u64(out_ptr, q_head_off_bytes);

                // PAR-021 GQA: Compute KV head index
                // kv_head_idx = q_head_idx * num_kv_heads / num_heads
                // This maps multiple Q heads to the same KV head
                // Use literal values since they're known at kernel build time
                let kv_head_idx = ctx.mul_u32(q_head_idx, num_kv_heads);
                let kv_head_idx = ctx.div_u32(kv_head_idx, num_heads);

                // K/V: kv_head_idx * max_seq_len * head_dim
                let kv_stride = ctx.mov_u32_imm(max_seq_len * head_dim);
                let kv_head_off = ctx.mul_lo_u32(kv_head_idx, kv_stride);
                let kv_head_off_bytes = ctx.mul_wide_u32_reg(kv_head_off, four);
                let k_head_ptr = ctx.add_u64(k_ptr, kv_head_off_bytes);
                let v_head_ptr = ctx.add_u64(v_ptr, kv_head_off_bytes);

                // CORRECTNESS-002: Each thread handles 4 elements (strided by 32) for head_dim up to 128
                // Thread 0 handles [0,32,64,96], thread 1 handles [1,33,65,97], etc.
                // Supports head_dim: 32, 64, 96, 128

                // Load Q values into registers (persistent across seq loop)
                // Using predicated loads for bounds checking
                // CORRECTNESS-002: Support head_dim up to 128 (4 elements per thread)
                let q0_off_bytes = ctx.mul_wide_u32_reg(lane_id, four);
                let q0_addr = ctx.add_u64(q_head_ptr, q0_off_bytes);
                let in_bounds0 = ctx.setp_lt_u32(lane_id, head_dim_u32);
                let q0 = ctx.ld_global_f32_predicated(q0_addr, in_bounds0, 0.0);

                // Second element (if head_dim > 32)
                let lane_plus_32 = ctx.add_u32(lane_id, 32);
                let q1_off_bytes = ctx.mul_wide_u32_reg(lane_plus_32, four);
                let q1_addr = ctx.add_u64(q_head_ptr, q1_off_bytes);
                let in_bounds1 = ctx.setp_lt_u32(lane_plus_32, head_dim_u32);
                let q1 = ctx.ld_global_f32_predicated(q1_addr, in_bounds1, 0.0);

                // CORRECTNESS-002: Third element (if head_dim > 64)
                let lane_plus_64 = ctx.add_u32(lane_id, 64);
                let q2_off_bytes = ctx.mul_wide_u32_reg(lane_plus_64, four);
                let q2_addr = ctx.add_u64(q_head_ptr, q2_off_bytes);
                let in_bounds2 = ctx.setp_lt_u32(lane_plus_64, head_dim_u32);
                let q2 = ctx.ld_global_f32_predicated(q2_addr, in_bounds2, 0.0);

                // CORRECTNESS-002: Fourth element (if head_dim > 96)
                let lane_plus_96 = ctx.add_u32(lane_id, 96);
                let q3_off_bytes = ctx.mul_wide_u32_reg(lane_plus_96, four);
                let q3_addr = ctx.add_u64(q_head_ptr, q3_off_bytes);
                let in_bounds3 = ctx.setp_lt_u32(lane_plus_96, head_dim_u32);
                let q3 = ctx.ld_global_f32_predicated(q3_addr, in_bounds3, 0.0);

                // Initialize output accumulators
                let out0 = ctx.mov_f32_imm(0.0);
                let out1 = ctx.mov_f32_imm(0.0);
                // CORRECTNESS-002: Additional accumulators for head_dim > 64
                let out2 = ctx.mov_f32_imm(0.0);
                let out3 = ctx.mov_f32_imm(0.0);

                // Online softmax state
                let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let sum_exp = ctx.mov_f32_imm(0.0);

                // Log2(e) for exp approximation via ex2
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scale_reg = ctx.mov_f32_imm(scale);

                // Loop counter
                let pos = ctx.mov_u32_imm(0);

                ctx.label("seq_loop");

                // Check loop condition
                let loop_cond = ctx.setp_lt_u32(pos, seq_len);
                ctx.branch_if_not(loop_cond, "seq_loop_end");

                // Compute K offset for this position: pos * head_dim
                let k_pos_off = ctx.mul_lo_u32(pos, head_dim_u32);

                // Load K[pos, lane_id] and K[pos, lane_id+32]
                let k0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
                let k0_off_bytes = ctx.mul_wide_u32_reg(k0_elem_off, four);
                let k0_addr = ctx.add_u64(k_head_ptr, k0_off_bytes);
                let k0 = ctx.ld_global_f32_predicated(k0_addr, in_bounds0, 0.0);

                let k1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
                let k1_off_bytes = ctx.mul_wide_u32_reg(k1_elem_off, four);
                let k1_addr = ctx.add_u64(k_head_ptr, k1_off_bytes);
                let k1 = ctx.ld_global_f32_predicated(k1_addr, in_bounds1, 0.0);

                // CORRECTNESS-002: Load K[pos, lane_id+64] and K[pos, lane_id+96]
                let k2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
                let k2_off_bytes = ctx.mul_wide_u32_reg(k2_elem_off, four);
                let k2_addr = ctx.add_u64(k_head_ptr, k2_off_bytes);
                let k2 = ctx.ld_global_f32_predicated(k2_addr, in_bounds2, 0.0);

                let k3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
                let k3_off_bytes = ctx.mul_wide_u32_reg(k3_elem_off, four);
                let k3_addr = ctx.add_u64(k_head_ptr, k3_off_bytes);
                let k3 = ctx.ld_global_f32_predicated(k3_addr, in_bounds3, 0.0);

                // Compute partial dot product: q0*k0 + q1*k1 + q2*k2 + q3*k3
                // CORRECTNESS-002: Now handles full head_dim=128
                let dot_partial = ctx.mul_f32(q0, k0);
                let dot_partial = ctx.fma_f32(q1, k1, dot_partial);
                let dot_partial = ctx.fma_f32(q2, k2, dot_partial);
                let dot_partial = ctx.fma_f32(q3, k3, dot_partial);

                // Warp-reduce the dot product using shfl.down
                // sum += shfl_down(sum, 16)
                // sum += shfl_down(sum, 8)
                // sum += shfl_down(sum, 4)
                // sum += shfl_down(sum, 2)
                // sum += shfl_down(sum, 1)
                let dot16 = ctx.shfl_down_f32(dot_partial, 16, 0xFFFF_FFFF);
                let dot_partial = ctx.add_f32(dot_partial, dot16);
                let dot8 = ctx.shfl_down_f32(dot_partial, 8, 0xFFFF_FFFF);
                let dot_partial = ctx.add_f32(dot_partial, dot8);
                let dot4 = ctx.shfl_down_f32(dot_partial, 4, 0xFFFF_FFFF);
                let dot_partial = ctx.add_f32(dot_partial, dot4);
                let dot2 = ctx.shfl_down_f32(dot_partial, 2, 0xFFFF_FFFF);
                let dot_partial = ctx.add_f32(dot_partial, dot2);
                let dot1 = ctx.shfl_down_f32(dot_partial, 1, 0xFFFF_FFFF);
                let dot_reduced = ctx.add_f32(dot_partial, dot1);

                // Broadcast result to all threads via shfl.idx lane 0
                let dot_broadcast = ctx.shfl_idx_f32(dot_reduced, 0, 0xFFFF_FFFF);

                // Scale the attention score
                let score = ctx.mul_f32(dot_broadcast, scale_reg);

                // Online softmax update (Milakov & Gimelshein 2018):
                // new_max = max(old_max, score)
                // correction = exp(old_max - new_max)
                // sum_exp = sum_exp * correction + exp(score - new_max)
                // output = output * correction + exp(score - new_max) * V

                let new_max = ctx.max_f32(max_score, score);

                // exp(old_max - new_max) using 2^(x * log2(e))
                let max_diff = ctx.sub_f32(max_score, new_max);
                let max_diff_scaled = ctx.mul_f32(max_diff, log2e);
                let correction = ctx.ex2_f32(max_diff_scaled);

                // exp(score - new_max)
                let score_diff = ctx.sub_f32(score, new_max);
                let score_diff_scaled = ctx.mul_f32(score_diff, log2e);
                let exp_score = ctx.ex2_f32(score_diff_scaled);

                // Load V[pos, lane_id] and V[pos, lane_id+32]
                let v0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
                let v0_off_bytes = ctx.mul_wide_u32_reg(v0_elem_off, four);
                let v0_addr = ctx.add_u64(v_head_ptr, v0_off_bytes);
                let v0 = ctx.ld_global_f32_predicated(v0_addr, in_bounds0, 0.0);

                let v1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
                let v1_off_bytes = ctx.mul_wide_u32_reg(v1_elem_off, four);
                let v1_addr = ctx.add_u64(v_head_ptr, v1_off_bytes);
                let v1 = ctx.ld_global_f32_predicated(v1_addr, in_bounds1, 0.0);

                // CORRECTNESS-002: Load V[pos, lane_id+64] and V[pos, lane_id+96]
                let v2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
                let v2_off_bytes = ctx.mul_wide_u32_reg(v2_elem_off, four);
                let v2_addr = ctx.add_u64(v_head_ptr, v2_off_bytes);
                let v2 = ctx.ld_global_f32_predicated(v2_addr, in_bounds2, 0.0);

                let v3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
                let v3_off_bytes = ctx.mul_wide_u32_reg(v3_elem_off, four);
                let v3_addr = ctx.add_u64(v_head_ptr, v3_off_bytes);
                let v3 = ctx.ld_global_f32_predicated(v3_addr, in_bounds3, 0.0);

                // Update loop state using in-place operations
                // Online softmax: max_score = max(max_score, score)
                ctx.max_f32_inplace(max_score, score);

                // sum_exp = sum_exp * correction + exp_score
                ctx.mul_f32_inplace(sum_exp, correction);
                ctx.add_f32_inplace(sum_exp, exp_score);

                // out = out * correction + exp_score * V
                ctx.mul_f32_inplace(out0, correction);
                ctx.fma_f32_inplace(out0, exp_score, v0);
                ctx.mul_f32_inplace(out1, correction);
                ctx.fma_f32_inplace(out1, exp_score, v1);
                // CORRECTNESS-002: Update out2 and out3
                ctx.mul_f32_inplace(out2, correction);
                ctx.fma_f32_inplace(out2, exp_score, v2);
                ctx.mul_f32_inplace(out3, correction);
                ctx.fma_f32_inplace(out3, exp_score, v3);

                // Increment position
                ctx.add_u32_inplace(pos, 1);
                ctx.branch("seq_loop");

                ctx.label("seq_loop_end");

                // Normalize output: out /= sum_exp
                // Use reciprocal approximation for speed
                let one = ctx.mov_f32_imm(1.0);
                let inv_sum = ctx.div_f32(one, sum_exp);

                ctx.mul_f32_inplace(out0, inv_sum);
                ctx.mul_f32_inplace(out1, inv_sum);
                // CORRECTNESS-002: Normalize out2 and out3
                ctx.mul_f32_inplace(out2, inv_sum);
                ctx.mul_f32_inplace(out3, inv_sum);

                // Store output (only for valid indices)
                // Thread writes to output[head_idx, lane_id]
                let out0_addr = ctx.add_u64(out_head_ptr, q0_off_bytes);
                ctx.branch_if_not(in_bounds0, "skip_store0");
                ctx.st_global_f32(out0_addr, out0);
                ctx.label("skip_store0");

                let out1_addr = ctx.add_u64(out_head_ptr, q1_off_bytes);
                ctx.branch_if_not(in_bounds1, "skip_store1");
                ctx.st_global_f32(out1_addr, out1);
                ctx.label("skip_store1");

                // CORRECTNESS-002: Store out2 and out3
                let out2_addr = ctx.add_u64(out_head_ptr, q2_off_bytes);
                ctx.branch_if_not(in_bounds2, "skip_store2");
                ctx.st_global_f32(out2_addr, out2);
                ctx.label("skip_store2");

                let out3_addr = ctx.add_u64(out_head_ptr, q3_off_bytes);
                ctx.branch_if_not(in_bounds3, "skip_store3");
                ctx.st_global_f32(out3_addr, out3);
                ctx.label("skip_store3");

                ctx.ret();
            })
    }
}

/// PAR-070: Multi-warp incremental attention for decode phase
///
/// Uses multiple warps per head to parallelize across KV cache positions.
/// Each warp processes a chunk of positions, then cross-warp reduction combines
/// the partial softmax states.
///
/// Performance target: 8x speedup over single-warp (from 81µs to ~10µs)
///
/// # Algorithm
///
/// 1. Launch `num_heads × num_warps_per_head` blocks
/// 2. Each warp handles positions [warp_idx * chunk, (warp_idx + 1) * chunk)
/// 3. Compute local max_score, sum_exp, weighted_output
/// 4. Cross-warp reduction in shared memory to get global max
/// 5. Correction pass to align all warps to global max
/// 6. Final sum and normalization
#[derive(Debug, Clone)]
pub struct MultiWarpIncrementalAttentionKernel {
    /// Maximum sequence length to support
    pub max_seq_len: u32,
    /// Head dimension (must be <= 128)
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Number of key-value heads (for GQA, <= num_heads)
    pub num_kv_heads: u32,
    /// Number of warps per head (parallelism factor)
    pub num_warps_per_head: u32,
    /// Scaling factor for attention scores (1/sqrt(head_dim))
    pub scale: f32,
    /// PAR-061: Read seq_len from device memory (for CUDA graph compatibility)
    pub indirect_seq_len: bool,
}

impl MultiWarpIncrementalAttentionKernel {
    /// Create new multi-warp incremental attention kernel
    ///
    /// # Arguments
    ///
    /// * `max_seq_len` - Maximum sequence length to support
    /// * `head_dim` - Dimension per attention head (must be <= 128)
    /// * `num_heads` - Number of query attention heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    /// * `num_warps` - Number of warps per head (4-8 recommended)
    #[must_use]
    pub fn new(
        max_seq_len: u32,
        head_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
        num_warps: u32,
    ) -> Self {
        Self {
            max_seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            num_warps_per_head: num_warps,
            scale: 1.0 / (head_dim as f32).sqrt(),
            indirect_seq_len: false,
        }
    }

    /// Enable indirect seq_len mode (reads from device memory)
    #[must_use]
    pub fn with_indirect_seq_len(mut self, indirect: bool) -> Self {
        self.indirect_seq_len = indirect;
        self
    }
}

impl Kernel for MultiWarpIncrementalAttentionKernel {
    fn name(&self) -> &str {
        if self.indirect_seq_len {
            "multi_warp_attention_indirect"
        } else {
            "multi_warp_attention"
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let scale = self.scale;
        let max_seq_len = self.max_seq_len;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let num_warps = self.num_warps_per_head;
        let indirect = self.indirect_seq_len;

        // Shared memory for cross-warp reduction:
        // - max_scores[num_warps]: local max per warp (offset 0)
        // - sum_exps[num_warps]: local sum_exp per warp (offset num_warps*4)
        // - global_max: 1 float (offset num_warps*8) - CORRECTNESS-013: separate from local values
        // - global_sum: 1 float (offset num_warps*8+4)
        // - outputs[num_warps * head_dim]: partial outputs per warp (offset num_warps*8+8)
        let smem_size = (num_warps * 2 + 2 + num_warps * head_dim) * 4;

        let kernel_name = if indirect {
            "multi_warp_attention_indirect"
        } else {
            "multi_warp_attention"
        };

        let mut builder = PtxKernel::new(kernel_name)
            .param(PtxType::U64, "q_ptr")
            .param(PtxType::U64, "k_ptr")
            .param(PtxType::U64, "v_ptr")
            .param(PtxType::U64, "out_ptr");

        builder = if indirect {
            builder.param(PtxType::U64, "seq_len_ptr")
        } else {
            builder.param(PtxType::U32, "seq_len")
        };

        builder.shared_memory(smem_size as usize).build(move |ctx| {
            // Grid: (num_heads, 1, 1) - one block per head
            // Block: (32 * num_warps, 1, 1) - multiple warps per block
            let q_head_idx = ctx.special_reg(PtxReg::CtaIdX);
            let tid = ctx.special_reg(PtxReg::TidX);

            // Compute warp_idx and lane_id from tid
            // warp_idx = tid / 32, lane_id = tid % 32
            let warp_idx = ctx.div_u32(tid, 32);
            let lane_id = ctx.rem_u32(tid, 32);

            // Load seq_len
            let seq_len = if indirect {
                let seq_len_ptr = ctx.load_param_u64("seq_len_ptr");
                ctx.ld_global_u32(seq_len_ptr)
            } else {
                ctx.load_param_u32("seq_len")
            };

            let q_ptr = ctx.load_param_u64("q_ptr");
            let k_ptr = ctx.load_param_u64("k_ptr");
            let v_ptr = ctx.load_param_u64("v_ptr");
            let out_ptr = ctx.load_param_u64("out_ptr");

            // Constants
            let four = ctx.mov_u32_imm(4);
            let head_dim_u32 = ctx.mov_u32_imm(head_dim);
            let num_warps_u32 = ctx.mov_u32_imm(num_warps);

            // Compute chunk boundaries for this warp
            // chunk_size = ceil(seq_len / num_warps) = (seq_len + num_warps - 1) / num_warps
            let seq_plus_nw = ctx.add_u32(seq_len, num_warps - 1);
            let chunk_size = ctx.div_u32(seq_plus_nw, num_warps);

            let start_pos = ctx.mul_lo_u32(warp_idx, chunk_size);
            let end_pos = ctx.add_u32_reg(start_pos, chunk_size);
            let end_pos = ctx.min_u32(end_pos, seq_len);

            // Compute Q/output head offset
            let q_head_off = ctx.mul_lo_u32(q_head_idx, head_dim_u32);
            let q_head_off_bytes = ctx.mul_wide_u32_reg(q_head_off, four);
            let q_head_ptr = ctx.add_u64(q_ptr, q_head_off_bytes);
            let out_head_ptr = ctx.add_u64(out_ptr, q_head_off_bytes);

            // GQA: kv_head_idx = q_head_idx * num_kv_heads / num_heads
            let kv_head_idx = ctx.mul_u32(q_head_idx, num_kv_heads);
            let kv_head_idx = ctx.div_u32(kv_head_idx, num_heads);

            // K/V: kv_head_idx * max_seq_len * head_dim
            let kv_stride = ctx.mov_u32_imm(max_seq_len * head_dim);
            let kv_head_off = ctx.mul_lo_u32(kv_head_idx, kv_stride);
            let kv_head_off_bytes = ctx.mul_wide_u32_reg(kv_head_off, four);
            let k_head_ptr = ctx.add_u64(k_ptr, kv_head_off_bytes);
            let v_head_ptr = ctx.add_u64(v_ptr, kv_head_off_bytes);

            // Load Q values (persistent across loop)
            let q0_off_bytes = ctx.mul_wide_u32_reg(lane_id, four);
            let q0_addr = ctx.add_u64(q_head_ptr, q0_off_bytes);
            let in_bounds0 = ctx.setp_lt_u32(lane_id, head_dim_u32);
            let q0 = ctx.ld_global_f32_predicated(q0_addr, in_bounds0, 0.0);

            let lane_plus_32 = ctx.add_u32(lane_id, 32);
            let q1_off_bytes = ctx.mul_wide_u32_reg(lane_plus_32, four);
            let q1_addr = ctx.add_u64(q_head_ptr, q1_off_bytes);
            let in_bounds1 = ctx.setp_lt_u32(lane_plus_32, head_dim_u32);
            let q1 = ctx.ld_global_f32_predicated(q1_addr, in_bounds1, 0.0);

            let lane_plus_64 = ctx.add_u32(lane_id, 64);
            let q2_off_bytes = ctx.mul_wide_u32_reg(lane_plus_64, four);
            let q2_addr = ctx.add_u64(q_head_ptr, q2_off_bytes);
            let in_bounds2 = ctx.setp_lt_u32(lane_plus_64, head_dim_u32);
            let q2 = ctx.ld_global_f32_predicated(q2_addr, in_bounds2, 0.0);

            let lane_plus_96 = ctx.add_u32(lane_id, 96);
            let q3_off_bytes = ctx.mul_wide_u32_reg(lane_plus_96, four);
            let q3_addr = ctx.add_u64(q_head_ptr, q3_off_bytes);
            let in_bounds3 = ctx.setp_lt_u32(lane_plus_96, head_dim_u32);
            let q3 = ctx.ld_global_f32_predicated(q3_addr, in_bounds3, 0.0);

            // Initialize accumulators
            let out0 = ctx.mov_f32_imm(0.0);
            let out1 = ctx.mov_f32_imm(0.0);
            let out2 = ctx.mov_f32_imm(0.0);
            let out3 = ctx.mov_f32_imm(0.0);
            let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
            let sum_exp = ctx.mov_f32_imm(0.0);

            let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
            let scale_reg = ctx.mov_f32_imm(scale);

            // Loop over this warp's chunk of positions
            let pos = ctx.add_u32(start_pos, 0);

            ctx.label("chunk_loop");
            let loop_cond = ctx.setp_lt_u32(pos, end_pos);
            ctx.branch_if_not(loop_cond, "chunk_loop_end");

            // Load K[pos]
            let k_pos_off = ctx.mul_lo_u32(pos, head_dim_u32);
            let k0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
            let k0_off_bytes = ctx.mul_wide_u32_reg(k0_elem_off, four);
            let k0_addr = ctx.add_u64(k_head_ptr, k0_off_bytes);
            let k0 = ctx.ld_global_f32_predicated(k0_addr, in_bounds0, 0.0);

            let k1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
            let k1_off_bytes = ctx.mul_wide_u32_reg(k1_elem_off, four);
            let k1_addr = ctx.add_u64(k_head_ptr, k1_off_bytes);
            let k1 = ctx.ld_global_f32_predicated(k1_addr, in_bounds1, 0.0);

            let k2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
            let k2_off_bytes = ctx.mul_wide_u32_reg(k2_elem_off, four);
            let k2_addr = ctx.add_u64(k_head_ptr, k2_off_bytes);
            let k2 = ctx.ld_global_f32_predicated(k2_addr, in_bounds2, 0.0);

            let k3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
            let k3_off_bytes = ctx.mul_wide_u32_reg(k3_elem_off, four);
            let k3_addr = ctx.add_u64(k_head_ptr, k3_off_bytes);
            let k3 = ctx.ld_global_f32_predicated(k3_addr, in_bounds3, 0.0);

            // Dot product Q·K
            let dot = ctx.mul_f32(q0, k0);
            let dot = ctx.fma_f32(q1, k1, dot);
            let dot = ctx.fma_f32(q2, k2, dot);
            let dot = ctx.fma_f32(q3, k3, dot);

            // Warp-reduce
            let dot16 = ctx.shfl_down_f32(dot, 16, 0xFFFF_FFFF);
            let dot = ctx.add_f32(dot, dot16);
            let dot8 = ctx.shfl_down_f32(dot, 8, 0xFFFF_FFFF);
            let dot = ctx.add_f32(dot, dot8);
            let dot4 = ctx.shfl_down_f32(dot, 4, 0xFFFF_FFFF);
            let dot = ctx.add_f32(dot, dot4);
            let dot2 = ctx.shfl_down_f32(dot, 2, 0xFFFF_FFFF);
            let dot = ctx.add_f32(dot, dot2);
            let dot1 = ctx.shfl_down_f32(dot, 1, 0xFFFF_FFFF);
            let dot = ctx.add_f32(dot, dot1);
            let score = ctx.shfl_idx_f32(dot, 0, 0xFFFF_FFFF);
            let score = ctx.mul_f32(score, scale_reg);

            // Online softmax update
            let new_max = ctx.max_f32(max_score, score);
            let max_diff = ctx.sub_f32(max_score, new_max);
            let max_diff_scaled = ctx.mul_f32(max_diff, log2e);
            let correction = ctx.ex2_f32(max_diff_scaled);
            let score_diff = ctx.sub_f32(score, new_max);
            let score_diff_scaled = ctx.mul_f32(score_diff, log2e);
            let exp_score = ctx.ex2_f32(score_diff_scaled);

            // Load V[pos]
            let v0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
            let v0_off_bytes = ctx.mul_wide_u32_reg(v0_elem_off, four);
            let v0_addr = ctx.add_u64(v_head_ptr, v0_off_bytes);
            let v0 = ctx.ld_global_f32_predicated(v0_addr, in_bounds0, 0.0);

            let v1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
            let v1_off_bytes = ctx.mul_wide_u32_reg(v1_elem_off, four);
            let v1_addr = ctx.add_u64(v_head_ptr, v1_off_bytes);
            let v1 = ctx.ld_global_f32_predicated(v1_addr, in_bounds1, 0.0);

            let v2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
            let v2_off_bytes = ctx.mul_wide_u32_reg(v2_elem_off, four);
            let v2_addr = ctx.add_u64(v_head_ptr, v2_off_bytes);
            let v2 = ctx.ld_global_f32_predicated(v2_addr, in_bounds2, 0.0);

            let v3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
            let v3_off_bytes = ctx.mul_wide_u32_reg(v3_elem_off, four);
            let v3_addr = ctx.add_u64(v_head_ptr, v3_off_bytes);
            let v3 = ctx.ld_global_f32_predicated(v3_addr, in_bounds3, 0.0);

            // Update state
            ctx.max_f32_inplace(max_score, score);
            ctx.mul_f32_inplace(sum_exp, correction);
            ctx.add_f32_inplace(sum_exp, exp_score);

            ctx.mul_f32_inplace(out0, correction);
            ctx.fma_f32_inplace(out0, exp_score, v0);
            ctx.mul_f32_inplace(out1, correction);
            ctx.fma_f32_inplace(out1, exp_score, v1);
            ctx.mul_f32_inplace(out2, correction);
            ctx.fma_f32_inplace(out2, exp_score, v2);
            ctx.mul_f32_inplace(out3, correction);
            ctx.fma_f32_inplace(out3, exp_score, v3);

            ctx.add_u32_inplace(pos, 1);
            ctx.branch("chunk_loop");

            ctx.label("chunk_loop_end");

            // Store local max and sum_exp to shared memory for cross-warp reduction
            // smem layout: [max_0..max_n, sum_0..sum_n, out_0..out_n*head_dim]
            // Shared memory uses offset from 0, not a base pointer in PTX
            let warp_off = ctx.mul_u32(warp_idx, 4);
            let warp_off_64 = ctx.cvt_u64_u32(warp_off);
            let max_off_base = ctx.mov_u64_imm(0);
            let max_addr = ctx.add_u64(max_off_base, warp_off_64);

            let sum_off_base = ctx.mov_u64_imm((num_warps * 4) as u64);
            let sum_addr = ctx.add_u64(sum_off_base, warp_off_64);

            // Only lane 0 writes to shared memory
            let zero_u32 = ctx.mov_u32_imm(0);
            let is_lane0 = ctx.setp_eq_u32(lane_id, zero_u32);
            ctx.branch_if_not(is_lane0, "skip_smem_write");
            ctx.st_shared_f32(max_addr, max_score);
            ctx.st_shared_f32(sum_addr, sum_exp);
            ctx.label("skip_smem_write");

            // Barrier for all threads in this block
            ctx.bar_sync(0);

            // Lane 0 of warp 0 reduces global max across warps
            let is_warp0 = ctx.setp_eq_u32(warp_idx, zero_u32);
            let is_warp0_lane0 = ctx.and_pred(is_warp0, is_lane0);

            ctx.branch_if_not(is_warp0_lane0, "skip_reduce");

            // Compute global max
            let global_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
            let reduce_i = ctx.mov_u32_imm(0);
            ctx.label("reduce_max_loop");
            let reduce_cond = ctx.setp_lt_u32(reduce_i, num_warps_u32);
            ctx.branch_if_not(reduce_cond, "reduce_max_done");

            let i_off = ctx.mul_u32(reduce_i, 4);
            let i_off_64 = ctx.cvt_u64_u32(i_off);
            let max_i_addr = ctx.add_u64(max_off_base, i_off_64);
            let max_i = ctx.ld_shared_f32(max_i_addr);
            ctx.max_f32_inplace(global_max, max_i);
            ctx.add_u32_inplace(reduce_i, 1);
            ctx.branch("reduce_max_loop");

            ctx.label("reduce_max_done");

            // Compute global sum_exp = sum(sum_exp_i * exp(max_i - global_max))
            let global_sum = ctx.mov_f32_imm(0.0);
            let reduce_i = ctx.mov_u32_imm(0);
            ctx.label("reduce_sum_loop");
            let reduce_cond = ctx.setp_lt_u32(reduce_i, num_warps_u32);
            ctx.branch_if_not(reduce_cond, "reduce_sum_done");

            let i_off = ctx.mul_u32(reduce_i, 4);
            let i_off_64 = ctx.cvt_u64_u32(i_off);
            let max_i_addr = ctx.add_u64(max_off_base, i_off_64);
            let max_i = ctx.ld_shared_f32(max_i_addr);
            let sum_i_addr = ctx.add_u64(sum_off_base, i_off_64);
            let sum_i = ctx.ld_shared_f32(sum_i_addr);

            let diff = ctx.sub_f32(max_i, global_max);
            let diff_scaled = ctx.mul_f32(diff, log2e);
            let correction = ctx.ex2_f32(diff_scaled);
            let corrected_sum = ctx.mul_f32(sum_i, correction);
            ctx.add_f32_inplace(global_sum, corrected_sum);

            ctx.add_u32_inplace(reduce_i, 1);
            ctx.branch("reduce_sum_loop");

            ctx.label("reduce_sum_done");

            // CORRECTNESS-013: Store global_max and global_sum to SEPARATE locations
            // Previous bug: storing to max_off_base (0) overwrote warp 0's local max
            let global_max_off = ctx.mov_u64_imm((num_warps * 8) as u64);
            let global_sum_off = ctx.mov_u64_imm((num_warps * 8 + 4) as u64);
            ctx.st_shared_f32(global_max_off, global_max);
            ctx.st_shared_f32(global_sum_off, global_sum);

            ctx.label("skip_reduce");

            // Barrier to wait for reduction
            ctx.bar_sync(1);

            // CORRECTNESS-013: Load global max and sum from dedicated locations
            let global_max_off = ctx.mov_u64_imm((num_warps * 8) as u64);
            let global_sum_off = ctx.mov_u64_imm((num_warps * 8 + 4) as u64);
            let global_max = ctx.ld_shared_f32(global_max_off);
            let global_sum = ctx.ld_shared_f32(global_sum_off);

            // Compute correction for this warp's contribution
            let my_max = ctx.ld_shared_f32(max_addr);
            let my_diff = ctx.sub_f32(my_max, global_max);
            let my_diff_scaled = ctx.mul_f32(my_diff, log2e);
            let my_correction = ctx.ex2_f32(my_diff_scaled);

            // Correct and normalize this warp's output
            let one = ctx.mov_f32_imm(1.0);
            let inv_sum = ctx.div_f32(one, global_sum);
            let final_scale = ctx.mul_f32(my_correction, inv_sum);

            ctx.mul_f32_inplace(out0, final_scale);
            ctx.mul_f32_inplace(out1, final_scale);
            ctx.mul_f32_inplace(out2, final_scale);
            ctx.mul_f32_inplace(out3, final_scale);

            // Store corrected outputs to shared memory for warp 0 to sum
            // CORRECTNESS-013: Output area starts at offset: num_warps*8 + 8 (after global max/sum)
            // Each warp stores head_dim * 4 bytes
            let out_area_base = ctx.mov_u32_imm(num_warps * 8 + 8);
            let warp_out_offset = ctx.mul_u32(warp_idx, head_dim * 4);
            let out_base = ctx.add_u32_reg(out_area_base, warp_out_offset);

            // Store out0-out3 based on lane positions
            let lane_off_0 = ctx.mul_u32(lane_id, 4);
            let out0_smem_off = ctx.add_u32_reg(out_base, lane_off_0);
            let out0_smem_addr = ctx.cvt_u64_u32(out0_smem_off);
            ctx.branch_if_not(in_bounds0, "skip_store_out0");
            ctx.st_shared_f32(out0_smem_addr, out0);
            ctx.label("skip_store_out0");

            let lane_off_1 = ctx.mul_u32(lane_plus_32, 4);
            let out1_smem_off = ctx.add_u32_reg(out_base, lane_off_1);
            let out1_smem_addr = ctx.cvt_u64_u32(out1_smem_off);
            ctx.branch_if_not(in_bounds1, "skip_store_out1");
            ctx.st_shared_f32(out1_smem_addr, out1);
            ctx.label("skip_store_out1");

            let lane_off_2 = ctx.mul_u32(lane_plus_64, 4);
            let out2_smem_off = ctx.add_u32_reg(out_base, lane_off_2);
            let out2_smem_addr = ctx.cvt_u64_u32(out2_smem_off);
            ctx.branch_if_not(in_bounds2, "skip_store_out2");
            ctx.st_shared_f32(out2_smem_addr, out2);
            ctx.label("skip_store_out2");

            let lane_off_3 = ctx.mul_u32(lane_plus_96, 4);
            let out3_smem_off = ctx.add_u32_reg(out_base, lane_off_3);
            let out3_smem_addr = ctx.cvt_u64_u32(out3_smem_off);
            ctx.branch_if_not(in_bounds3, "skip_store_out3");
            ctx.st_shared_f32(out3_smem_addr, out3);
            ctx.label("skip_store_out3");

            // Barrier to ensure all warps stored their outputs
            ctx.bar_sync(2);

            // Only warp 0 sums outputs and stores to global memory
            ctx.branch_if_not(is_warp0, "skip_final_sum");

            // Sum outputs from all warps for each element this warp handles
            // Thread i in warp 0 sums element i across all warps
            let final_out0 = ctx.mov_f32_imm(0.0);
            let sum_w = ctx.mov_u32_imm(0);
            ctx.label("sum_warps_loop0");
            let sum_cond = ctx.setp_lt_u32(sum_w, num_warps_u32);
            ctx.branch_if_not(sum_cond, "sum_warps_done0");

            let w_out_offset = ctx.mul_u32(sum_w, head_dim * 4);
            let w_out_base = ctx.add_u32_reg(out_area_base, w_out_offset);
            let elem_off = ctx.mul_u32(lane_id, 4);
            let elem_addr_off = ctx.add_u32_reg(w_out_base, elem_off);
            let elem_addr = ctx.cvt_u64_u32(elem_addr_off);
            let elem_val = ctx.ld_shared_f32(elem_addr);
            ctx.add_f32_inplace(final_out0, elem_val);
            ctx.add_u32_inplace(sum_w, 1);
            ctx.branch("sum_warps_loop0");

            ctx.label("sum_warps_done0");

            // Store final output to global memory
            let out0_addr = ctx.add_u64(out_head_ptr, q0_off_bytes);
            ctx.branch_if_not(in_bounds0, "skip_final_store0");
            ctx.st_global_f32(out0_addr, final_out0);
            ctx.label("skip_final_store0");

            // Repeat for elements 32-63 (out1)
            let final_out1 = ctx.mov_f32_imm(0.0);
            let sum_w = ctx.mov_u32_imm(0);
            ctx.label("sum_warps_loop1");
            let sum_cond = ctx.setp_lt_u32(sum_w, num_warps_u32);
            ctx.branch_if_not(sum_cond, "sum_warps_done1");

            let w_out_offset = ctx.mul_u32(sum_w, head_dim * 4);
            let w_out_base = ctx.add_u32_reg(out_area_base, w_out_offset);
            let elem_off = ctx.mul_u32(lane_plus_32, 4);
            let elem_addr_off = ctx.add_u32_reg(w_out_base, elem_off);
            let elem_addr = ctx.cvt_u64_u32(elem_addr_off);
            let elem_val = ctx.ld_shared_f32(elem_addr);
            ctx.add_f32_inplace(final_out1, elem_val);
            ctx.add_u32_inplace(sum_w, 1);
            ctx.branch("sum_warps_loop1");

            ctx.label("sum_warps_done1");

            let out1_addr = ctx.add_u64(out_head_ptr, q1_off_bytes);
            ctx.branch_if_not(in_bounds1, "skip_final_store1");
            ctx.st_global_f32(out1_addr, final_out1);
            ctx.label("skip_final_store1");

            // Repeat for elements 64-95 (out2)
            let final_out2 = ctx.mov_f32_imm(0.0);
            let sum_w = ctx.mov_u32_imm(0);
            ctx.label("sum_warps_loop2");
            let sum_cond = ctx.setp_lt_u32(sum_w, num_warps_u32);
            ctx.branch_if_not(sum_cond, "sum_warps_done2");

            let w_out_offset = ctx.mul_u32(sum_w, head_dim * 4);
            let w_out_base = ctx.add_u32_reg(out_area_base, w_out_offset);
            let elem_off = ctx.mul_u32(lane_plus_64, 4);
            let elem_addr_off = ctx.add_u32_reg(w_out_base, elem_off);
            let elem_addr = ctx.cvt_u64_u32(elem_addr_off);
            let elem_val = ctx.ld_shared_f32(elem_addr);
            ctx.add_f32_inplace(final_out2, elem_val);
            ctx.add_u32_inplace(sum_w, 1);
            ctx.branch("sum_warps_loop2");

            ctx.label("sum_warps_done2");

            let out2_addr = ctx.add_u64(out_head_ptr, q2_off_bytes);
            ctx.branch_if_not(in_bounds2, "skip_final_store2");
            ctx.st_global_f32(out2_addr, final_out2);
            ctx.label("skip_final_store2");

            // Repeat for elements 96-127 (out3)
            let final_out3 = ctx.mov_f32_imm(0.0);
            let sum_w = ctx.mov_u32_imm(0);
            ctx.label("sum_warps_loop3");
            let sum_cond = ctx.setp_lt_u32(sum_w, num_warps_u32);
            ctx.branch_if_not(sum_cond, "sum_warps_done3");

            let w_out_offset = ctx.mul_u32(sum_w, head_dim * 4);
            let w_out_base = ctx.add_u32_reg(out_area_base, w_out_offset);
            let elem_off = ctx.mul_u32(lane_plus_96, 4);
            let elem_addr_off = ctx.add_u32_reg(w_out_base, elem_off);
            let elem_addr = ctx.cvt_u64_u32(elem_addr_off);
            let elem_val = ctx.ld_shared_f32(elem_addr);
            ctx.add_f32_inplace(final_out3, elem_val);
            ctx.add_u32_inplace(sum_w, 1);
            ctx.branch("sum_warps_loop3");

            ctx.label("sum_warps_done3");

            let out3_addr = ctx.add_u64(out_head_ptr, q3_off_bytes);
            ctx.branch_if_not(in_bounds3, "skip_final_store3");
            ctx.st_global_f32(out3_addr, final_out3);
            ctx.label("skip_final_store3");

            ctx.label("skip_final_sum");

            ctx.ret();
        })
    }
}

/// PAR-118: Batched Incremental Attention for M sequences in parallel
///
/// Processes M independent sequences in a single kernel launch, reducing
/// kernel launch overhead from 3M to 3 per layer (batched KV scatter + batched attention).
///
/// Grid: (num_heads, batch_size, 1)
/// Block: (32, 1, 1) - one warp per head
///
/// Memory layout:
/// - q: [M, num_heads, head_dim] - contiguous query vectors
/// - k_ptrs: [M] - array of M pointers to K caches
/// - v_ptrs: [M] - array of M pointers to V caches
/// - output: [M, num_heads, head_dim] - contiguous output
/// - seq_lens: [M] - array of M sequence lengths (indirect mode)
#[derive(Debug, Clone)]
pub struct BatchedIncrementalAttentionKernel {
    /// Maximum sequence length to support
    pub max_seq_len: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Scaling factor for attention scores
    pub scale: f32,
}

impl BatchedIncrementalAttentionKernel {
    /// Create a new batched incremental attention kernel
    #[must_use]
    pub fn new(
        max_seq_len: u32,
        head_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
        batch_size: u32,
    ) -> Self {
        Self {
            max_seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            batch_size,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }
}

impl Kernel for BatchedIncrementalAttentionKernel {
    fn name(&self) -> &str {
        "batched_incremental_attention"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let scale = self.scale;
        let max_seq_len = self.max_seq_len;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let _batch_size = self.batch_size;

        // Grid: (num_heads, batch_size, 1)
        // Block: (32, 1, 1) - one warp per block
        //
        // Each block handles one (head, batch) pair
        // batch_idx = blockIdx.y selects which sequence
        // head_idx = blockIdx.x selects which Q head

        PtxKernel::new("batched_incremental_attention")
            .param(PtxType::U64, "q_ptr") // [M, num_heads, head_dim]
            .param(PtxType::U64, "k_ptrs_ptr") // [M] array of K cache pointers
            .param(PtxType::U64, "v_ptrs_ptr") // [M] array of V cache pointers
            .param(PtxType::U64, "out_ptr") // [M, num_heads, head_dim]
            .param(PtxType::U64, "seq_lens_ptr") // [M] array of sequence lengths
            .shared_memory(0)
            .build(move |ctx| {
                // Get indices
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);
                let lane_id = ctx.special_reg(PtxReg::TidX);

                // Load parameters
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptrs_ptr = ctx.load_param_u64("k_ptrs_ptr");
                let v_ptrs_ptr = ctx.load_param_u64("v_ptrs_ptr");
                let out_ptr = ctx.load_param_u64("out_ptr");
                let seq_lens_ptr = ctx.load_param_u64("seq_lens_ptr");

                // Load seq_len for this batch element
                let four = ctx.mov_u32_imm(4);
                let eight = ctx.mov_u32_imm(8);
                let batch_idx_bytes = ctx.mul_wide_u32_reg(batch_idx, four);
                let seq_len_addr = ctx.add_u64(seq_lens_ptr, batch_idx_bytes);
                let seq_len = ctx.ld_global_u32(seq_len_addr);

                // Load K and V cache pointers for this batch element
                let batch_ptr_off = ctx.mul_wide_u32_reg(batch_idx, eight);
                let k_ptr_addr = ctx.add_u64(k_ptrs_ptr, batch_ptr_off);
                let v_ptr_addr = ctx.add_u64(v_ptrs_ptr, batch_ptr_off);
                let k_cache_ptr = ctx.ld_global_u64(k_ptr_addr);
                let v_cache_ptr = ctx.ld_global_u64(v_ptr_addr);

                // Compute Q/output offset: batch_idx * num_heads * head_dim + head_idx * head_dim
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);
                let num_heads_u32 = ctx.mov_u32_imm(num_heads);
                let batch_head_stride = ctx.mul_lo_u32(num_heads_u32, head_dim_u32);
                let batch_off = ctx.mul_lo_u32(batch_idx, batch_head_stride);
                let head_off = ctx.mul_lo_u32(head_idx, head_dim_u32);
                let q_head_off = ctx.add_u32_reg(batch_off, head_off);
                let q_head_off_bytes = ctx.mul_wide_u32_reg(q_head_off, four);
                let q_head_ptr = ctx.add_u64(q_ptr, q_head_off_bytes);
                let out_head_ptr = ctx.add_u64(out_ptr, q_head_off_bytes);

                // GQA: Compute KV head index
                let kv_head_idx = ctx.mul_u32(head_idx, num_kv_heads);
                let kv_head_idx = ctx.div_u32(kv_head_idx, num_heads);

                // K/V: kv_head_idx * max_seq_len * head_dim
                let kv_stride = ctx.mov_u32_imm(max_seq_len * head_dim);
                let kv_head_off = ctx.mul_lo_u32(kv_head_idx, kv_stride);
                let kv_head_off_bytes = ctx.mul_wide_u32_reg(kv_head_off, four);
                let k_head_ptr = ctx.add_u64(k_cache_ptr, kv_head_off_bytes);
                let v_head_ptr = ctx.add_u64(v_cache_ptr, kv_head_off_bytes);

                // Load Q values (same as IncrementalAttentionKernel)
                let q0_off_bytes = ctx.mul_wide_u32_reg(lane_id, four);
                let q0_addr = ctx.add_u64(q_head_ptr, q0_off_bytes);
                let in_bounds0 = ctx.setp_lt_u32(lane_id, head_dim_u32);
                let q0 = ctx.ld_global_f32_predicated(q0_addr, in_bounds0, 0.0);

                let lane_plus_32 = ctx.add_u32(lane_id, 32);
                let q1_off_bytes = ctx.mul_wide_u32_reg(lane_plus_32, four);
                let q1_addr = ctx.add_u64(q_head_ptr, q1_off_bytes);
                let in_bounds1 = ctx.setp_lt_u32(lane_plus_32, head_dim_u32);
                let q1 = ctx.ld_global_f32_predicated(q1_addr, in_bounds1, 0.0);

                let lane_plus_64 = ctx.add_u32(lane_id, 64);
                let q2_off_bytes = ctx.mul_wide_u32_reg(lane_plus_64, four);
                let q2_addr = ctx.add_u64(q_head_ptr, q2_off_bytes);
                let in_bounds2 = ctx.setp_lt_u32(lane_plus_64, head_dim_u32);
                let q2 = ctx.ld_global_f32_predicated(q2_addr, in_bounds2, 0.0);

                let lane_plus_96 = ctx.add_u32(lane_id, 96);
                let q3_off_bytes = ctx.mul_wide_u32_reg(lane_plus_96, four);
                let q3_addr = ctx.add_u64(q_head_ptr, q3_off_bytes);
                let in_bounds3 = ctx.setp_lt_u32(lane_plus_96, head_dim_u32);
                let q3 = ctx.ld_global_f32_predicated(q3_addr, in_bounds3, 0.0);

                // Initialize accumulators
                let out0 = ctx.mov_f32_imm(0.0);
                let out1 = ctx.mov_f32_imm(0.0);
                let out2 = ctx.mov_f32_imm(0.0);
                let out3 = ctx.mov_f32_imm(0.0);

                // Online softmax state
                let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let sum_exp = ctx.mov_f32_imm(0.0);
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scale_reg = ctx.mov_f32_imm(scale);

                // Loop over sequence positions
                let pos = ctx.mov_u32_imm(0);
                ctx.label("batched_seq_loop");
                let loop_cond = ctx.setp_lt_u32(pos, seq_len);
                ctx.branch_if_not(loop_cond, "batched_seq_loop_end");

                // Load K[pos] and compute Q·K dot product
                let k_pos_off = ctx.mul_lo_u32(pos, head_dim_u32);

                let k0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
                let k0_off_bytes = ctx.mul_wide_u32_reg(k0_elem_off, four);
                let k0_addr = ctx.add_u64(k_head_ptr, k0_off_bytes);
                let k0 = ctx.ld_global_f32_predicated(k0_addr, in_bounds0, 0.0);

                let k1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
                let k1_off_bytes = ctx.mul_wide_u32_reg(k1_elem_off, four);
                let k1_addr = ctx.add_u64(k_head_ptr, k1_off_bytes);
                let k1 = ctx.ld_global_f32_predicated(k1_addr, in_bounds1, 0.0);

                let k2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
                let k2_off_bytes = ctx.mul_wide_u32_reg(k2_elem_off, four);
                let k2_addr = ctx.add_u64(k_head_ptr, k2_off_bytes);
                let k2 = ctx.ld_global_f32_predicated(k2_addr, in_bounds2, 0.0);

                let k3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
                let k3_off_bytes = ctx.mul_wide_u32_reg(k3_elem_off, four);
                let k3_addr = ctx.add_u64(k_head_ptr, k3_off_bytes);
                let k3 = ctx.ld_global_f32_predicated(k3_addr, in_bounds3, 0.0);

                // Dot product Q·K
                let dot = ctx.mul_f32(q0, k0);
                ctx.fma_f32_inplace(dot, q1, k1);
                ctx.fma_f32_inplace(dot, q2, k2);
                ctx.fma_f32_inplace(dot, q3, k3);

                // Warp reduce - use full warp mask for all 32 threads
                for delta in [16, 8, 4, 2, 1] {
                    let other = ctx.shfl_down_f32(dot, delta, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(dot, other);
                }

                // Scale score
                let score = ctx.mul_f32(dot, scale_reg);

                // Online softmax update
                let old_max = max_score;
                ctx.max_f32_inplace(max_score, score);
                let score_minus_max = ctx.sub_f32(score, max_score);
                let score_log2 = ctx.mul_f32(score_minus_max, log2e);
                let exp_score = ctx.ex2_f32(score_log2);

                // Rescale sum_exp if max changed
                let old_minus_new = ctx.sub_f32(old_max, max_score);
                let log2_old = ctx.mul_f32(old_minus_new, log2e);
                let correction = ctx.ex2_f32(log2_old);
                ctx.mul_f32_inplace(sum_exp, correction);
                ctx.add_f32_inplace(sum_exp, exp_score);

                // Rescale existing output
                ctx.mul_f32_inplace(out0, correction);
                ctx.mul_f32_inplace(out1, correction);
                ctx.mul_f32_inplace(out2, correction);
                ctx.mul_f32_inplace(out3, correction);

                // Load V[pos] and accumulate
                let v0_addr = ctx.add_u64(v_head_ptr, k0_off_bytes);
                let v0 = ctx.ld_global_f32_predicated(v0_addr, in_bounds0, 0.0);
                ctx.fma_f32_inplace(out0, exp_score, v0);

                let v1_addr = ctx.add_u64(v_head_ptr, k1_off_bytes);
                let v1 = ctx.ld_global_f32_predicated(v1_addr, in_bounds1, 0.0);
                ctx.fma_f32_inplace(out1, exp_score, v1);

                let v2_addr = ctx.add_u64(v_head_ptr, k2_off_bytes);
                let v2 = ctx.ld_global_f32_predicated(v2_addr, in_bounds2, 0.0);
                ctx.fma_f32_inplace(out2, exp_score, v2);

                let v3_addr = ctx.add_u64(v_head_ptr, k3_off_bytes);
                let v3 = ctx.ld_global_f32_predicated(v3_addr, in_bounds3, 0.0);
                ctx.fma_f32_inplace(out3, exp_score, v3);

                ctx.add_u32_inplace(pos, 1);
                ctx.branch("batched_seq_loop");

                ctx.label("batched_seq_loop_end");

                // Normalize output
                let one = ctx.mov_f32_imm(1.0);
                let inv_sum = ctx.div_f32(one, sum_exp);
                ctx.mul_f32_inplace(out0, inv_sum);
                ctx.mul_f32_inplace(out1, inv_sum);
                ctx.mul_f32_inplace(out2, inv_sum);
                ctx.mul_f32_inplace(out3, inv_sum);

                // Store output
                let out0_addr = ctx.add_u64(out_head_ptr, q0_off_bytes);
                ctx.branch_if_not(in_bounds0, "batched_skip_store0");
                ctx.st_global_f32(out0_addr, out0);
                ctx.label("batched_skip_store0");

                let out1_addr = ctx.add_u64(out_head_ptr, q1_off_bytes);
                ctx.branch_if_not(in_bounds1, "batched_skip_store1");
                ctx.st_global_f32(out1_addr, out1);
                ctx.label("batched_skip_store1");

                let out2_addr = ctx.add_u64(out_head_ptr, q2_off_bytes);
                ctx.branch_if_not(in_bounds2, "batched_skip_store2");
                ctx.st_global_f32(out2_addr, out2);
                ctx.label("batched_skip_store2");

                let out3_addr = ctx.add_u64(out_head_ptr, q3_off_bytes);
                ctx.branch_if_not(in_bounds3, "batched_skip_store3");
                ctx.st_global_f32(out3_addr, out3);
                ctx.label("batched_skip_store3");

                ctx.ret();
            })
    }
}

// =============================================================================
// PAR-118: Flash Decoding - Split-K Attention for 2X Ollama Performance
// =============================================================================
//
// Flash Decoding splits the KV cache into chunks processed in parallel,
// then reduces partial results. This amortizes memory bandwidth across
// multiple thread blocks, achieving higher throughput for long sequences.
//
// Algorithm:
// 1. Split sequence into K chunks of CHUNK_SIZE positions
// 2. Each chunk computes partial attention: (max_score, sum_exp, weighted_out)
// 3. Reduction combines partials with proper softmax rescaling:
//    - new_max = max(chunk_max[0], chunk_max[1], ...)
//    - For each chunk: scale = exp(chunk_max - new_max)
//    - new_sum = sum(chunk_sum[i] * scale[i])
//    - output = sum(chunk_out[i] * chunk_sum[i] * scale[i]) / new_sum
//
// Performance:
// - Current: Sequential loop over seq_len (memory-bandwidth limited)
// - Flash Decoding: K parallel blocks (K = ceil(seq_len / CHUNK_SIZE))
// - Expected speedup: ~1.5-2x for typical seq_len (512-2048)
// =============================================================================

/// Chunk size for Flash Decoding split-K attention
/// Trade-off: smaller = more parallelism, larger = less reduction overhead
pub const FLASH_DECODE_CHUNK_SIZE: u32 = 128;

/// PAR-118: Flash Decoding kernel for split-K attention
///
/// Splits the KV cache into chunks and processes them in parallel.
/// Requires a separate reduction kernel to combine partial results.
///
/// Memory layout:
/// - q: [M, num_heads, head_dim] - contiguous query vectors
/// - k_ptrs: [M] - array of M pointers to K caches
/// - v_ptrs: [M] - array of M pointers to V caches
/// - partials: [M, num_heads, num_chunks, head_dim + 2] - partial results
///   - [0..head_dim]: weighted output (sum of exp_score * V)
///   - [head_dim]: max_score for this chunk
///   - [head_dim + 1]: sum_exp for this chunk
/// - seq_lens: [M] - array of M sequence lengths
#[derive(Debug, Clone)]
pub struct FlashDecodingChunkKernel {
    /// Maximum sequence length to support
    pub max_seq_len: u32,
    /// Head dimension
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Chunk size for split-K
    pub chunk_size: u32,
    /// Scaling factor for attention scores
    pub scale: f32,
}

impl FlashDecodingChunkKernel {
    /// Create a new Flash Decoding chunk kernel
    #[must_use]
    pub fn new(
        max_seq_len: u32,
        head_dim: u32,
        num_heads: u32,
        num_kv_heads: u32,
        batch_size: u32,
    ) -> Self {
        Self {
            max_seq_len,
            head_dim,
            num_heads,
            num_kv_heads,
            batch_size,
            chunk_size: FLASH_DECODE_CHUNK_SIZE,
            scale: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    /// Get the number of chunks for a given sequence length
    #[must_use]
    pub fn num_chunks(&self, seq_len: u32) -> u32 {
        (seq_len + self.chunk_size - 1) / self.chunk_size
    }

    /// Get the size of the partials buffer per (head, batch) pair
    /// Layout: [num_chunks, head_dim + 2]
    #[must_use]
    pub fn partials_size_per_head(&self, max_chunks: u32) -> u32 {
        max_chunks * (self.head_dim + 2)
    }
}

impl Kernel for FlashDecodingChunkKernel {
    fn name(&self) -> &str {
        "flash_decoding_chunk"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let scale = self.scale;
        let max_seq_len = self.max_seq_len;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let chunk_size = self.chunk_size;
        let _batch_size = self.batch_size;

        // Grid: (num_heads, batch_size, num_chunks)
        // Block: (32, 1, 1) - one warp per block
        //
        // Each block handles one (head, batch, chunk) triple
        // chunk_idx = blockIdx.z selects which chunk of the sequence
        // batch_idx = blockIdx.y selects which sequence
        // head_idx = blockIdx.x selects which Q head

        PtxKernel::new("flash_decoding_chunk")
            .param(PtxType::U64, "q_ptr") // [M, num_heads, head_dim]
            .param(PtxType::U64, "k_ptrs_ptr") // [M] array of K cache pointers
            .param(PtxType::U64, "v_ptrs_ptr") // [M] array of V cache pointers
            .param(PtxType::U64, "partials_ptr") // [M, num_heads, num_chunks, head_dim + 2]
            .param(PtxType::U64, "seq_lens_ptr") // [M] array of sequence lengths
            .param(PtxType::U32, "max_chunks") // Maximum number of chunks
            .shared_memory(0)
            .build(move |ctx| {
                // Get indices
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);
                let chunk_idx = ctx.special_reg(PtxReg::CtaIdZ);
                let lane_id = ctx.special_reg(PtxReg::TidX);

                // Load parameters
                let q_ptr = ctx.load_param_u64("q_ptr");
                let k_ptrs_ptr = ctx.load_param_u64("k_ptrs_ptr");
                let v_ptrs_ptr = ctx.load_param_u64("v_ptrs_ptr");
                let partials_ptr = ctx.load_param_u64("partials_ptr");
                let seq_lens_ptr = ctx.load_param_u64("seq_lens_ptr");
                let max_chunks_param = ctx.load_param_u32("max_chunks");

                let four = ctx.mov_u32_imm(4);
                let eight = ctx.mov_u32_imm(8);

                // Load seq_len for this batch element
                let batch_idx_bytes = ctx.mul_wide_u32_reg(batch_idx, four);
                let seq_len_addr = ctx.add_u64(seq_lens_ptr, batch_idx_bytes);
                let seq_len = ctx.ld_global_u32(seq_len_addr);

                // Compute chunk boundaries
                let chunk_size_u32 = ctx.mov_u32_imm(chunk_size);
                let chunk_start = ctx.mul_lo_u32(chunk_idx, chunk_size_u32);
                let chunk_end_raw = ctx.add_u32(chunk_start, chunk_size); // Use literal for add_u32
                                                                          // Clamp chunk_end to seq_len
                let chunk_end = ctx.min_u32(chunk_end_raw, seq_len); // Both are VirtualReg

                // Early exit if chunk_start >= seq_len (this chunk has no work)
                let has_work = ctx.setp_lt_u32(chunk_start, seq_len);
                ctx.branch_if_not(has_work, "flash_decode_chunk_empty");

                // Load K and V cache pointers for this batch element
                let batch_ptr_off = ctx.mul_wide_u32_reg(batch_idx, eight);
                let k_ptr_addr = ctx.add_u64(k_ptrs_ptr, batch_ptr_off);
                let v_ptr_addr = ctx.add_u64(v_ptrs_ptr, batch_ptr_off);
                let k_cache_ptr = ctx.ld_global_u64(k_ptr_addr);
                let v_cache_ptr = ctx.ld_global_u64(v_ptr_addr);

                // Compute Q offset: batch_idx * num_heads * head_dim + head_idx * head_dim
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);
                let num_heads_u32 = ctx.mov_u32_imm(num_heads);
                let batch_head_stride = ctx.mul_lo_u32(num_heads_u32, head_dim_u32);
                let batch_off = ctx.mul_lo_u32(batch_idx, batch_head_stride);
                let head_off = ctx.mul_lo_u32(head_idx, head_dim_u32);
                let q_head_off = ctx.add_u32_reg(batch_off, head_off);
                let q_head_off_bytes = ctx.mul_wide_u32_reg(q_head_off, four);
                let q_head_ptr = ctx.add_u64(q_ptr, q_head_off_bytes);

                // GQA: Compute KV head index
                let kv_head_idx = ctx.mul_u32(head_idx, num_kv_heads);
                let kv_head_idx = ctx.div_u32(kv_head_idx, num_heads);

                // K/V: kv_head_idx * max_seq_len * head_dim
                let kv_stride = ctx.mov_u32_imm(max_seq_len * head_dim);
                let kv_head_off = ctx.mul_lo_u32(kv_head_idx, kv_stride);
                let kv_head_off_bytes = ctx.mul_wide_u32_reg(kv_head_off, four);
                let k_head_ptr = ctx.add_u64(k_cache_ptr, kv_head_off_bytes);
                let v_head_ptr = ctx.add_u64(v_cache_ptr, kv_head_off_bytes);

                // Load Q values (head_dim up to 128, 4 elements per lane)
                let q0_off_bytes = ctx.mul_wide_u32_reg(lane_id, four);
                let q0_addr = ctx.add_u64(q_head_ptr, q0_off_bytes);
                let in_bounds0 = ctx.setp_lt_u32(lane_id, head_dim_u32);
                let q0 = ctx.ld_global_f32_predicated(q0_addr, in_bounds0, 0.0);

                let lane_plus_32 = ctx.add_u32(lane_id, 32);
                let q1_off_bytes = ctx.mul_wide_u32_reg(lane_plus_32, four);
                let q1_addr = ctx.add_u64(q_head_ptr, q1_off_bytes);
                let in_bounds1 = ctx.setp_lt_u32(lane_plus_32, head_dim_u32);
                let q1 = ctx.ld_global_f32_predicated(q1_addr, in_bounds1, 0.0);

                let lane_plus_64 = ctx.add_u32(lane_id, 64);
                let q2_off_bytes = ctx.mul_wide_u32_reg(lane_plus_64, four);
                let q2_addr = ctx.add_u64(q_head_ptr, q2_off_bytes);
                let in_bounds2 = ctx.setp_lt_u32(lane_plus_64, head_dim_u32);
                let q2 = ctx.ld_global_f32_predicated(q2_addr, in_bounds2, 0.0);

                let lane_plus_96 = ctx.add_u32(lane_id, 96);
                let q3_off_bytes = ctx.mul_wide_u32_reg(lane_plus_96, four);
                let q3_addr = ctx.add_u64(q_head_ptr, q3_off_bytes);
                let in_bounds3 = ctx.setp_lt_u32(lane_plus_96, head_dim_u32);
                let q3 = ctx.ld_global_f32_predicated(q3_addr, in_bounds3, 0.0);

                // Initialize accumulators for this chunk
                let out0 = ctx.mov_f32_imm(0.0);
                let out1 = ctx.mov_f32_imm(0.0);
                let out2 = ctx.mov_f32_imm(0.0);
                let out3 = ctx.mov_f32_imm(0.0);

                // Online softmax state for this chunk
                let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let sum_exp = ctx.mov_f32_imm(0.0);
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let scale_reg = ctx.mov_f32_imm(scale);

                // Loop over positions in this chunk [chunk_start, chunk_end)
                let pos = chunk_start;
                ctx.label("flash_decode_chunk_loop");
                let loop_cond = ctx.setp_lt_u32(pos, chunk_end);
                ctx.branch_if_not(loop_cond, "flash_decode_chunk_loop_end");

                // Load K[pos] and compute Q·K dot product
                let k_pos_off = ctx.mul_lo_u32(pos, head_dim_u32);

                let k0_elem_off = ctx.add_u32_reg(k_pos_off, lane_id);
                let k0_off_bytes = ctx.mul_wide_u32_reg(k0_elem_off, four);
                let k0_addr = ctx.add_u64(k_head_ptr, k0_off_bytes);
                let k0 = ctx.ld_global_f32_predicated(k0_addr, in_bounds0, 0.0);

                let k1_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_32);
                let k1_off_bytes = ctx.mul_wide_u32_reg(k1_elem_off, four);
                let k1_addr = ctx.add_u64(k_head_ptr, k1_off_bytes);
                let k1 = ctx.ld_global_f32_predicated(k1_addr, in_bounds1, 0.0);

                let k2_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_64);
                let k2_off_bytes = ctx.mul_wide_u32_reg(k2_elem_off, four);
                let k2_addr = ctx.add_u64(k_head_ptr, k2_off_bytes);
                let k2 = ctx.ld_global_f32_predicated(k2_addr, in_bounds2, 0.0);

                let k3_elem_off = ctx.add_u32_reg(k_pos_off, lane_plus_96);
                let k3_off_bytes = ctx.mul_wide_u32_reg(k3_elem_off, four);
                let k3_addr = ctx.add_u64(k_head_ptr, k3_off_bytes);
                let k3 = ctx.ld_global_f32_predicated(k3_addr, in_bounds3, 0.0);

                // Dot product Q·K
                let dot = ctx.mul_f32(q0, k0);
                ctx.fma_f32_inplace(dot, q1, k1);
                ctx.fma_f32_inplace(dot, q2, k2);
                ctx.fma_f32_inplace(dot, q3, k3);

                // Warp reduce
                for delta in [16, 8, 4, 2, 1] {
                    let other = ctx.shfl_down_f32(dot, delta, 0xFFFF_FFFF);
                    ctx.add_f32_inplace(dot, other);
                }

                // Scale score
                let score = ctx.mul_f32(dot, scale_reg);

                // Online softmax update
                let old_max = max_score;
                ctx.max_f32_inplace(max_score, score);
                let score_minus_max = ctx.sub_f32(score, max_score);
                let score_log2 = ctx.mul_f32(score_minus_max, log2e);
                let exp_score = ctx.ex2_f32(score_log2);

                // Rescale sum_exp if max changed
                let old_minus_new = ctx.sub_f32(old_max, max_score);
                let log2_old = ctx.mul_f32(old_minus_new, log2e);
                let correction = ctx.ex2_f32(log2_old);
                ctx.mul_f32_inplace(sum_exp, correction);
                ctx.add_f32_inplace(sum_exp, exp_score);

                // Rescale existing output
                ctx.mul_f32_inplace(out0, correction);
                ctx.mul_f32_inplace(out1, correction);
                ctx.mul_f32_inplace(out2, correction);
                ctx.mul_f32_inplace(out3, correction);

                // Load V[pos] and accumulate (NOT normalized yet)
                let v0_addr = ctx.add_u64(v_head_ptr, k0_off_bytes);
                let v0 = ctx.ld_global_f32_predicated(v0_addr, in_bounds0, 0.0);
                ctx.fma_f32_inplace(out0, exp_score, v0);

                let v1_addr = ctx.add_u64(v_head_ptr, k1_off_bytes);
                let v1 = ctx.ld_global_f32_predicated(v1_addr, in_bounds1, 0.0);
                ctx.fma_f32_inplace(out1, exp_score, v1);

                let v2_addr = ctx.add_u64(v_head_ptr, k2_off_bytes);
                let v2 = ctx.ld_global_f32_predicated(v2_addr, in_bounds2, 0.0);
                ctx.fma_f32_inplace(out2, exp_score, v2);

                let v3_addr = ctx.add_u64(v_head_ptr, k3_off_bytes);
                let v3 = ctx.ld_global_f32_predicated(v3_addr, in_bounds3, 0.0);
                ctx.fma_f32_inplace(out3, exp_score, v3);

                ctx.add_u32_inplace(pos, 1);
                ctx.branch("flash_decode_chunk_loop");

                ctx.label("flash_decode_chunk_loop_end");

                // Compute partials offset:
                // partials_ptr + (batch_idx * num_heads * max_chunks + head_idx * max_chunks + chunk_idx) * (head_dim + 2) * 4
                let head_dim_plus_2 = ctx.mov_u32_imm(head_dim + 2);
                let partial_stride = ctx.mul_lo_u32(max_chunks_param, head_dim_plus_2);
                let batch_partial_stride = ctx.mul_lo_u32(num_heads_u32, partial_stride);
                let batch_partial_off = ctx.mul_lo_u32(batch_idx, batch_partial_stride);
                let head_partial_off = ctx.mul_lo_u32(head_idx, partial_stride);
                let chunk_partial_off = ctx.mul_lo_u32(chunk_idx, head_dim_plus_2);
                let partial_off = ctx.add_u32_reg(batch_partial_off, head_partial_off);
                let partial_off = ctx.add_u32_reg(partial_off, chunk_partial_off);
                let partial_off_bytes = ctx.mul_wide_u32_reg(partial_off, four);
                let partial_base = ctx.add_u64(partials_ptr, partial_off_bytes);

                // Store weighted output (out0..out3)
                let out0_addr = ctx.add_u64(partial_base, q0_off_bytes);
                ctx.branch_if_not(in_bounds0, "flash_decode_skip_out0");
                ctx.st_global_f32(out0_addr, out0);
                ctx.label("flash_decode_skip_out0");

                let out1_addr = ctx.add_u64(partial_base, q1_off_bytes);
                ctx.branch_if_not(in_bounds1, "flash_decode_skip_out1");
                ctx.st_global_f32(out1_addr, out1);
                ctx.label("flash_decode_skip_out1");

                let out2_addr = ctx.add_u64(partial_base, q2_off_bytes);
                ctx.branch_if_not(in_bounds2, "flash_decode_skip_out2");
                ctx.st_global_f32(out2_addr, out2);
                ctx.label("flash_decode_skip_out2");

                let out3_addr = ctx.add_u64(partial_base, q3_off_bytes);
                ctx.branch_if_not(in_bounds3, "flash_decode_skip_out3");
                ctx.st_global_f32(out3_addr, out3);
                ctx.label("flash_decode_skip_out3");

                // Store max_score at offset head_dim (only lane 0)
                let zero_u32 = ctx.mov_u32_imm(0);
                let is_lane0 = ctx.setp_eq_u32(lane_id, zero_u32);
                ctx.branch_if_not(is_lane0, "flash_decode_skip_meta");
                let max_off = ctx.mov_u32_imm(head_dim);
                let max_off_bytes = ctx.mul_wide_u32_reg(max_off, four);
                let max_addr = ctx.add_u64(partial_base, max_off_bytes);
                ctx.st_global_f32(max_addr, max_score);

                // Store sum_exp at offset head_dim + 1
                let sum_off = ctx.mov_u32_imm(head_dim + 1);
                let sum_off_bytes = ctx.mul_wide_u32_reg(sum_off, four);
                let sum_addr = ctx.add_u64(partial_base, sum_off_bytes);
                ctx.st_global_f32(sum_addr, sum_exp);
                ctx.label("flash_decode_skip_meta");

                ctx.ret();

                // Empty chunk handler - store sentinel values
                ctx.label("flash_decode_chunk_empty");
                // Same partial offset calculation
                let head_dim_plus_2_e = ctx.mov_u32_imm(head_dim + 2);
                let partial_stride_e = ctx.mul_lo_u32(max_chunks_param, head_dim_plus_2_e);
                let batch_partial_stride_e = ctx.mul_lo_u32(num_heads_u32, partial_stride_e);
                let batch_partial_off_e = ctx.mul_lo_u32(batch_idx, batch_partial_stride_e);
                let head_partial_off_e = ctx.mul_lo_u32(head_idx, partial_stride_e);
                let chunk_partial_off_e = ctx.mul_lo_u32(chunk_idx, head_dim_plus_2_e);
                let partial_off_e = ctx.add_u32_reg(batch_partial_off_e, head_partial_off_e);
                let partial_off_e = ctx.add_u32_reg(partial_off_e, chunk_partial_off_e);
                let partial_off_bytes_e = ctx.mul_wide_u32_reg(partial_off_e, four);
                let partial_base_e = ctx.add_u64(partials_ptr, partial_off_bytes_e);

                // Store -inf for max_score (sentinel for empty chunk)
                let zero_u32_e = ctx.mov_u32_imm(0);
                let is_lane0_e = ctx.setp_eq_u32(lane_id, zero_u32_e);
                ctx.branch_if_not(is_lane0_e, "flash_decode_empty_done");
                let neg_inf = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let max_off_e = ctx.mov_u32_imm(head_dim);
                let max_off_bytes_e = ctx.mul_wide_u32_reg(max_off_e, four);
                let max_addr_e = ctx.add_u64(partial_base_e, max_off_bytes_e);
                ctx.st_global_f32(max_addr_e, neg_inf);

                // Store 0 for sum_exp
                let zero = ctx.mov_f32_imm(0.0);
                let sum_off_e = ctx.mov_u32_imm(head_dim + 1);
                let sum_off_bytes_e = ctx.mul_wide_u32_reg(sum_off_e, four);
                let sum_addr_e = ctx.add_u64(partial_base_e, sum_off_bytes_e);
                ctx.st_global_f32(sum_addr_e, zero);
                ctx.label("flash_decode_empty_done");

                ctx.ret();
            })
    }
}

/// PAR-118: Flash Decoding reduction kernel
///
/// Reduces partial attention results from multiple chunks into final output.
/// Uses online softmax rescaling to combine results correctly.
///
/// Memory layout:
/// - partials: [M, num_heads, num_chunks, head_dim + 2]
/// - output: [M, num_heads, head_dim]
/// - seq_lens: [M] - sequence lengths (to compute actual num_chunks)
#[derive(Debug, Clone)]
pub struct FlashDecodingReduceKernel {
    /// Head dimension
    pub head_dim: u32,
    /// Number of query attention heads
    pub num_heads: u32,
    /// Batch size (M)
    pub batch_size: u32,
    /// Chunk size used in chunk kernel
    pub chunk_size: u32,
}

impl FlashDecodingReduceKernel {
    /// Create a new Flash Decoding reduce kernel
    #[must_use]
    pub fn new(head_dim: u32, num_heads: u32, batch_size: u32) -> Self {
        Self {
            head_dim,
            num_heads,
            batch_size,
            chunk_size: FLASH_DECODE_CHUNK_SIZE,
        }
    }
}

impl Kernel for FlashDecodingReduceKernel {
    fn name(&self) -> &str {
        "flash_decoding_reduce"
    }

    fn build_ptx(&self) -> PtxKernel {
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let chunk_size = self.chunk_size;
        let _batch_size = self.batch_size;

        // Grid: (num_heads, batch_size, 1)
        // Block: (32, 1, 1) - one warp per block
        //
        // Each block reduces all chunks for one (head, batch) pair

        PtxKernel::new("flash_decoding_reduce")
            .param(PtxType::U64, "partials_ptr") // [M, num_heads, max_chunks, head_dim + 2]
            .param(PtxType::U64, "output_ptr") // [M, num_heads, head_dim]
            .param(PtxType::U64, "seq_lens_ptr") // [M] sequence lengths
            .param(PtxType::U32, "max_chunks") // Maximum number of chunks
            .shared_memory(0)
            .build(move |ctx| {
                let head_idx = ctx.special_reg(PtxReg::CtaIdX);
                let batch_idx = ctx.special_reg(PtxReg::CtaIdY);
                let lane_id = ctx.special_reg(PtxReg::TidX);

                let partials_ptr = ctx.load_param_u64("partials_ptr");
                let output_ptr = ctx.load_param_u64("output_ptr");
                let seq_lens_ptr = ctx.load_param_u64("seq_lens_ptr");
                let max_chunks = ctx.load_param_u32("max_chunks");

                let four = ctx.mov_u32_imm(4);

                // Load seq_len and compute actual number of chunks
                let batch_idx_bytes = ctx.mul_wide_u32_reg(batch_idx, four);
                let seq_len_addr = ctx.add_u64(seq_lens_ptr, batch_idx_bytes);
                let seq_len = ctx.ld_global_u32(seq_len_addr);
                // num_chunks = (seq_len + chunk_size - 1) / chunk_size
                let seq_plus_chunk_m1 = ctx.add_u32(seq_len, chunk_size - 1);
                let num_chunks = ctx.div_u32(seq_plus_chunk_m1, chunk_size);

                // Compute partials base for this (batch, head) pair
                let head_dim_u32 = ctx.mov_u32_imm(head_dim);
                let num_heads_u32 = ctx.mov_u32_imm(num_heads);
                let head_dim_plus_2 = ctx.mov_u32_imm(head_dim + 2);
                let partial_stride = ctx.mul_lo_u32(max_chunks, head_dim_plus_2);
                let batch_partial_stride = ctx.mul_lo_u32(num_heads_u32, partial_stride);
                let batch_partial_off = ctx.mul_lo_u32(batch_idx, batch_partial_stride);
                let head_partial_off = ctx.mul_lo_u32(head_idx, partial_stride);
                let partial_base_off = ctx.add_u32_reg(batch_partial_off, head_partial_off);
                let partial_base_off_bytes = ctx.mul_wide_u32_reg(partial_base_off, four);
                let partial_base = ctx.add_u64(partials_ptr, partial_base_off_bytes);

                // First pass: find global max across all chunks
                let global_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
                let chunk_iter = ctx.mov_u32_imm(0);
                ctx.label("reduce_max_loop");
                let max_loop_cond = ctx.setp_lt_u32(chunk_iter, num_chunks);
                ctx.branch_if_not(max_loop_cond, "reduce_max_loop_end");

                // Load this chunk's max_score (at offset head_dim within chunk)
                let chunk_off = ctx.mul_lo_u32(chunk_iter, head_dim_plus_2);
                let max_elem_off = ctx.add_u32(chunk_off, head_dim);
                let max_elem_off_bytes = ctx.mul_wide_u32_reg(max_elem_off, four);
                let chunk_max_addr = ctx.add_u64(partial_base, max_elem_off_bytes);
                // Only lane 0 needs to read (broadcast later)
                let zero_lane = ctx.mov_u32_imm(0);
                let is_lane0 = ctx.setp_eq_u32(lane_id, zero_lane);
                let chunk_max =
                    ctx.ld_global_f32_predicated(chunk_max_addr, is_lane0, f32::NEG_INFINITY);
                // Broadcast from lane 0 to all lanes
                let chunk_max = ctx.shfl_idx_f32(chunk_max, 0, 0xFFFF_FFFF);
                ctx.max_f32_inplace(global_max, chunk_max);

                ctx.add_u32_inplace(chunk_iter, 1);
                ctx.branch("reduce_max_loop");
                ctx.label("reduce_max_loop_end");

                // Second pass: accumulate rescaled outputs
                let log2e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                let global_sum = ctx.mov_f32_imm(0.0);
                let acc0 = ctx.mov_f32_imm(0.0);
                let acc1 = ctx.mov_f32_imm(0.0);
                let acc2 = ctx.mov_f32_imm(0.0);
                let acc3 = ctx.mov_f32_imm(0.0);

                let lane_plus_32 = ctx.add_u32(lane_id, 32);
                let lane_plus_64 = ctx.add_u32(lane_id, 64);
                let lane_plus_96 = ctx.add_u32(lane_id, 96);
                let in_bounds0 = ctx.setp_lt_u32(lane_id, head_dim_u32);
                let in_bounds1 = ctx.setp_lt_u32(lane_plus_32, head_dim_u32);
                let in_bounds2 = ctx.setp_lt_u32(lane_plus_64, head_dim_u32);
                let in_bounds3 = ctx.setp_lt_u32(lane_plus_96, head_dim_u32);

                let chunk_iter2 = ctx.mov_u32_imm(0);
                ctx.label("reduce_acc_loop");
                let acc_loop_cond = ctx.setp_lt_u32(chunk_iter2, num_chunks);
                ctx.branch_if_not(acc_loop_cond, "reduce_acc_loop_end");

                // Load chunk max and sum_exp
                let chunk_off2 = ctx.mul_lo_u32(chunk_iter2, head_dim_plus_2);
                let max_elem_off2 = ctx.add_u32(chunk_off2, head_dim);
                let sum_elem_off2 = ctx.add_u32(chunk_off2, head_dim);
                let sum_elem_off2 = ctx.add_u32(sum_elem_off2, 1);
                let max_off_bytes2 = ctx.mul_wide_u32_reg(max_elem_off2, four);
                let sum_off_bytes2 = ctx.mul_wide_u32_reg(sum_elem_off2, four);
                let chunk_max_addr2 = ctx.add_u64(partial_base, max_off_bytes2);
                let chunk_sum_addr2 = ctx.add_u64(partial_base, sum_off_bytes2);

                let chunk_max2 =
                    ctx.ld_global_f32_predicated(chunk_max_addr2, is_lane0, f32::NEG_INFINITY);
                let chunk_sum2 = ctx.ld_global_f32_predicated(chunk_sum_addr2, is_lane0, 0.0);
                let chunk_max2 = ctx.shfl_idx_f32(chunk_max2, 0, 0xFFFF_FFFF);
                let chunk_sum2 = ctx.shfl_idx_f32(chunk_sum2, 0, 0xFFFF_FFFF);

                // Skip empty chunks (max = -inf)
                let neg_inf_check = ctx.mov_f32_imm(-1e30);
                let is_valid = ctx.setp_gt_f32(chunk_max2, neg_inf_check);
                ctx.branch_if_not(is_valid, "reduce_skip_chunk");

                // Compute scale factor: exp(chunk_max - global_max)
                let max_diff = ctx.sub_f32(chunk_max2, global_max);
                let max_diff_log2 = ctx.mul_f32(max_diff, log2e);
                let scale_factor = ctx.ex2_f32(max_diff_log2);

                // Accumulate scaled sum_exp
                let scaled_sum = ctx.mul_f32(chunk_sum2, scale_factor);
                ctx.add_f32_inplace(global_sum, scaled_sum);

                // Load and accumulate scaled output
                let chunk_base_off_bytes = ctx.mul_wide_u32_reg(chunk_off2, four);
                let chunk_base = ctx.add_u64(partial_base, chunk_base_off_bytes);

                let out0_off_bytes = ctx.mul_wide_u32_reg(lane_id, four);
                let out0_addr = ctx.add_u64(chunk_base, out0_off_bytes);
                let out0 = ctx.ld_global_f32_predicated(out0_addr, in_bounds0, 0.0);
                let scaled_out0 = ctx.mul_f32(out0, scale_factor);
                ctx.add_f32_inplace(acc0, scaled_out0);

                let out1_off_bytes = ctx.mul_wide_u32_reg(lane_plus_32, four);
                let out1_addr = ctx.add_u64(chunk_base, out1_off_bytes);
                let out1 = ctx.ld_global_f32_predicated(out1_addr, in_bounds1, 0.0);
                let scaled_out1 = ctx.mul_f32(out1, scale_factor);
                ctx.add_f32_inplace(acc1, scaled_out1);

                let out2_off_bytes = ctx.mul_wide_u32_reg(lane_plus_64, four);
                let out2_addr = ctx.add_u64(chunk_base, out2_off_bytes);
                let out2 = ctx.ld_global_f32_predicated(out2_addr, in_bounds2, 0.0);
                let scaled_out2 = ctx.mul_f32(out2, scale_factor);
                ctx.add_f32_inplace(acc2, scaled_out2);

                let out3_off_bytes = ctx.mul_wide_u32_reg(lane_plus_96, four);
                let out3_addr = ctx.add_u64(chunk_base, out3_off_bytes);
                let out3 = ctx.ld_global_f32_predicated(out3_addr, in_bounds3, 0.0);
                let scaled_out3 = ctx.mul_f32(out3, scale_factor);
                ctx.add_f32_inplace(acc3, scaled_out3);

                ctx.label("reduce_skip_chunk");
                ctx.add_u32_inplace(chunk_iter2, 1);
                ctx.branch("reduce_acc_loop");
                ctx.label("reduce_acc_loop_end");

                // Final normalization: output = acc / global_sum
                let one = ctx.mov_f32_imm(1.0);
                let inv_sum = ctx.div_f32(one, global_sum);
                ctx.mul_f32_inplace(acc0, inv_sum);
                ctx.mul_f32_inplace(acc1, inv_sum);
                ctx.mul_f32_inplace(acc2, inv_sum);
                ctx.mul_f32_inplace(acc3, inv_sum);

                // Compute output offset: batch_idx * num_heads * head_dim + head_idx * head_dim
                let batch_head_stride = ctx.mul_lo_u32(num_heads_u32, head_dim_u32);
                let batch_off = ctx.mul_lo_u32(batch_idx, batch_head_stride);
                let head_off = ctx.mul_lo_u32(head_idx, head_dim_u32);
                let out_base_off = ctx.add_u32_reg(batch_off, head_off);
                let out_base_off_bytes = ctx.mul_wide_u32_reg(out_base_off, four);
                let out_base = ctx.add_u64(output_ptr, out_base_off_bytes);

                // Store output
                let final_out0_addr = ctx.add_u64(out_base, out0_off_bytes);
                ctx.branch_if_not(in_bounds0, "reduce_skip_store0");
                ctx.st_global_f32(final_out0_addr, acc0);
                ctx.label("reduce_skip_store0");

                let final_out1_addr = ctx.add_u64(out_base, out1_off_bytes);
                ctx.branch_if_not(in_bounds1, "reduce_skip_store1");
                ctx.st_global_f32(final_out1_addr, acc1);
                ctx.label("reduce_skip_store1");

                let final_out2_addr = ctx.add_u64(out_base, out2_off_bytes);
                ctx.branch_if_not(in_bounds2, "reduce_skip_store2");
                ctx.st_global_f32(final_out2_addr, acc2);
                ctx.label("reduce_skip_store2");

                let final_out3_addr = ctx.add_u64(out_base, out3_off_bytes);
                ctx.branch_if_not(in_bounds3, "reduce_skip_store3");
                ctx.st_global_f32(final_out3_addr, acc3);
                ctx.label("reduce_skip_store3");

                ctx.ret();
            })
    }
}
