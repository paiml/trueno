//! GPU-resident weight structures and encoder/decoder configurations.
//!
//! This module contains weight structs for GPU-resident transformer blocks,
//! along with configuration types and forward pass implementations that
//! operate with zero host transfers.

#[cfg(feature = "cuda")]
use super::{batched_multihead_attention_optimized, GpuResidentTensor};
#[cfg(feature = "cuda")]
use crate::driver::CudaContext;
#[cfg(feature = "cuda")]
use crate::error::Result;

// ============================================================================
// GPU-Resident Encoder Block (Total Offload)
// ============================================================================

/// Weights for a single GPU-resident encoder block
///
/// Pre-upload all weights to GPU at model load time.
/// Then run forward passes with ZERO host transfers.
#[cfg(feature = "cuda")]
pub struct GpuEncoderBlockWeights {
    /// Layer norm 1: gamma [d_model]
    pub ln1_gamma: GpuResidentTensor<f32>,
    /// Layer norm 1: beta [d_model]
    pub ln1_beta: GpuResidentTensor<f32>,
    /// Query projection: weight [d_model, d_model]
    pub w_q: GpuResidentTensor<f32>,
    /// Query projection: bias [d_model]
    pub b_q: GpuResidentTensor<f32>,
    /// Key projection: weight [d_model, d_model]
    pub w_k: GpuResidentTensor<f32>,
    /// Key projection: bias [d_model]
    pub b_k: GpuResidentTensor<f32>,
    /// Value projection: weight [d_model, d_model]
    pub w_v: GpuResidentTensor<f32>,
    /// Value projection: bias [d_model]
    pub b_v: GpuResidentTensor<f32>,
    /// Output projection: weight [d_model, d_model]
    pub w_o: GpuResidentTensor<f32>,
    /// Output projection: bias [d_model]
    pub b_o: GpuResidentTensor<f32>,
    /// Layer norm 2: gamma [d_model]
    pub ln2_gamma: GpuResidentTensor<f32>,
    /// Layer norm 2: beta [d_model]
    pub ln2_beta: GpuResidentTensor<f32>,
    /// FFN up projection: weight [d_model, ffn_dim]
    pub ffn_up_w: GpuResidentTensor<f32>,
    /// FFN up projection: bias [ffn_dim]
    pub ffn_up_b: GpuResidentTensor<f32>,
    /// FFN down projection: weight [ffn_dim, d_model]
    pub ffn_down_w: GpuResidentTensor<f32>,
    /// FFN down projection: bias [d_model]
    pub ffn_down_b: GpuResidentTensor<f32>,
}

/// WAPR-PERF-012: GPU Conv Frontend Weights
#[cfg(feature = "cuda")]
pub struct GpuConvFrontendWeights {
    /// Conv1: weight [out_channels, in_channels, kernel_size] = [384, 80, 3]
    pub conv1_weight: GpuResidentTensor<f32>,
    /// Conv1: bias [out_channels] = [384]
    pub conv1_bias: GpuResidentTensor<f32>,
    /// Conv2: weight [out_channels, in_channels, kernel_size] = [384, 384, 3]
    pub conv2_weight: GpuResidentTensor<f32>,
    /// Conv2: bias [out_channels] = [384]
    pub conv2_bias: GpuResidentTensor<f32>,
}

/// WAPR-PERF-013: GPU Decoder Block Weights (similar to encoder but with cross-attention)
#[cfg(feature = "cuda")]
pub struct GpuDecoderBlockWeights {
    // Self-Attention weights
    /// Layer norm 1: gamma [d_model]
    pub ln1_gamma: GpuResidentTensor<f32>,
    /// Layer norm 1: beta [d_model]
    pub ln1_beta: GpuResidentTensor<f32>,
    /// Self-Attention Q: weight [d_model, d_model]
    pub self_w_q: GpuResidentTensor<f32>,
    /// Self-Attention Q: bias [d_model]
    pub self_b_q: GpuResidentTensor<f32>,
    /// Self-Attention K: weight [d_model, d_model]
    pub self_w_k: GpuResidentTensor<f32>,
    /// Self-Attention K: bias [d_model]
    pub self_b_k: GpuResidentTensor<f32>,
    /// Self-Attention V: weight [d_model, d_model]
    pub self_w_v: GpuResidentTensor<f32>,
    /// Self-Attention V: bias [d_model]
    pub self_b_v: GpuResidentTensor<f32>,
    /// Self-Attention O: weight [d_model, d_model]
    pub self_w_o: GpuResidentTensor<f32>,
    /// Self-Attention O: bias [d_model]
    pub self_b_o: GpuResidentTensor<f32>,

    // Cross-Attention weights
    /// Layer norm 2: gamma [d_model]
    pub ln2_gamma: GpuResidentTensor<f32>,
    /// Layer norm 2: beta [d_model]
    pub ln2_beta: GpuResidentTensor<f32>,
    /// Cross-Attention Q: weight [d_model, d_model]
    pub cross_w_q: GpuResidentTensor<f32>,
    /// Cross-Attention Q: bias [d_model]
    pub cross_b_q: GpuResidentTensor<f32>,
    /// Cross-Attention K: weight [d_model, d_model]
    pub cross_w_k: GpuResidentTensor<f32>,
    /// Cross-Attention K: bias [d_model]
    pub cross_b_k: GpuResidentTensor<f32>,
    /// Cross-Attention V: weight [d_model, d_model]
    pub cross_w_v: GpuResidentTensor<f32>,
    /// Cross-Attention V: bias [d_model]
    pub cross_b_v: GpuResidentTensor<f32>,
    /// Cross-Attention O: weight [d_model, d_model]
    pub cross_w_o: GpuResidentTensor<f32>,
    /// Cross-Attention O: bias [d_model]
    pub cross_b_o: GpuResidentTensor<f32>,

    // FFN weights
    /// Layer norm 3: gamma [d_model]
    pub ln3_gamma: GpuResidentTensor<f32>,
    /// Layer norm 3: beta [d_model]
    pub ln3_beta: GpuResidentTensor<f32>,
    /// FFN up projection: weight [d_model, ffn_dim]
    pub ffn_up_w: GpuResidentTensor<f32>,
    /// FFN up projection: bias [ffn_dim]
    pub ffn_up_b: GpuResidentTensor<f32>,
    /// FFN down projection: weight [ffn_dim, d_model]
    pub ffn_down_w: GpuResidentTensor<f32>,
    /// FFN down projection: bias [d_model]
    pub ffn_down_b: GpuResidentTensor<f32>,
}

/// WAPR-PERF-013: GPU-Resident KV Cache for decoder
///
/// Stores K/V tensors on GPU to avoid D2H/H2D transfers during decoding.
#[cfg(feature = "cuda")]
pub struct GpuKvCache {
    /// Key cache [max_seq_len, d_model] - grows incrementally
    pub key: GpuResidentTensor<f32>,
    /// Value cache [max_seq_len, d_model] - grows incrementally
    pub value: GpuResidentTensor<f32>,
    /// Current sequence length (number of tokens cached)
    pub seq_len: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Model dimension
    pub d_model: usize,
}

#[cfg(feature = "cuda")]
impl GpuKvCache {
    /// Create new GPU KV cache
    pub fn new(ctx: &CudaContext, max_seq_len: usize, d_model: usize) -> Result<Self> {
        let total_size = max_seq_len * d_model;
        let zeros = vec![0.0f32; total_size];

        let key = GpuResidentTensor::from_host(ctx, &zeros)?;
        let value = GpuResidentTensor::from_host(ctx, &zeros)?;

        Ok(Self {
            key,
            value,
            seq_len: 0,
            max_seq_len,
            d_model,
        })
    }

    /// Reset cache (for new sequence)
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Get current sequence length
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }
}

/// Configuration for GPU decoder
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct GpuDecoderConfig {
    /// Model dimension (d_model)
    pub d_model: u32,
    /// Number of attention heads
    pub n_heads: u32,
    /// FFN hidden dimension (typically 4 * d_model)
    pub ffn_dim: u32,
    /// Maximum sequence length
    pub max_seq_len: u32,
    /// Number of decoder layers
    pub n_layers: u32,
}

/// Configuration for GPU encoder
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub struct GpuEncoderConfig {
    /// Model dimension (d_model)
    pub d_model: u32,
    /// Number of attention heads
    pub n_heads: u32,
    /// FFN hidden dimension (typically 4 * d_model)
    pub ffn_dim: u32,
}

/// Forward pass through one encoder block (100% GPU-resident)
///
/// Architecture: Pre-norm with residual connections
/// x + Attention(LN(x)) then x + FFN(LN(x))
///
/// # Arguments
/// * `ctx` - CUDA context
/// * `x` - Input tensor [seq_len * d_model] on GPU
/// * `weights` - Pre-uploaded encoder block weights
/// * `config` - Encoder configuration
/// * `seq_len` - Sequence length
///
/// # Returns
/// Output tensor [seq_len * d_model] on GPU
#[cfg(feature = "cuda")]
pub fn forward_encoder_block_gpu(
    ctx: &CudaContext,
    x: &GpuResidentTensor<f32>,
    weights: &GpuEncoderBlockWeights,
    config: &GpuEncoderConfig,
) -> Result<GpuResidentTensor<f32>> {
    let d_model = config.d_model;
    let n_heads = config.n_heads;
    let head_dim = d_model / n_heads;
    let ffn_dim = config.ffn_dim;
    let seq_len = (x.len() / d_model as usize) as u32;

    // Debug flag for intermediate value inspection
    let debug = std::env::var("WHISPER_DEBUG_GPU_INTERNALS").is_ok();

    // ====== Self-Attention Block ======

    // Pre-norm: x_norm = LayerNorm(x)
    let x_norm = x.layer_norm(ctx, &weights.ln1_gamma, &weights.ln1_beta, d_model, seq_len)?;

    if debug {
        let ln1_host = x_norm.peek_host()?;
        let mean = ln1_host.iter().sum::<f32>() / ln1_host.len() as f32;
        let std = (ln1_host.iter().map(|v| v.powi(2)).sum::<f32>() / ln1_host.len() as f32).sqrt();
        eprintln!(
            "[DEBUG-GPU-INTERNAL] LN1 output: mean={:.6}, std={:.6}",
            mean, std
        );

        // Check weight matrices
        let wq_host = weights.w_q.peek_host()?;
        let bq_host = weights.b_q.peek_host()?;
        eprintln!(
            "[DEBUG-GPU-INTERNAL] w_q: len={}, mean={:.6}, max={:.6}",
            wq_host.len(),
            wq_host.iter().sum::<f32>() / wq_host.len() as f32,
            wq_host.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
        eprintln!(
            "[DEBUG-GPU-INTERNAL] b_q: len={}, mean={:.6}",
            bq_host.len(),
            bq_host.iter().sum::<f32>() / bq_host.len() as f32
        );
    }

    // Q, K, V projections (all on GPU)
    let q = x_norm.linear(
        ctx,
        &weights.w_q,
        Some(&weights.b_q),
        seq_len,
        d_model,
        d_model,
    )?;
    let k = x_norm.linear(
        ctx,
        &weights.w_k,
        Some(&weights.b_k),
        seq_len,
        d_model,
        d_model,
    )?;
    let v = x_norm.linear(
        ctx,
        &weights.w_v,
        Some(&weights.b_v),
        seq_len,
        d_model,
        d_model,
    )?;

    if debug {
        let q_host = q.peek_host()?;
        let k_host = k.peek_host()?;
        let v_host = v.peek_host()?;
        eprintln!(
            "[DEBUG-GPU-INTERNAL] Q: mean={:.6}, K: mean={:.6}, V: mean={:.6}",
            q_host.iter().sum::<f32>() / q_host.len() as f32,
            k_host.iter().sum::<f32>() / k_host.len() as f32,
            v_host.iter().sum::<f32>() / v_host.len() as f32
        );
    }

    // Multi-head attention (on GPU)
    // WAPR-PERF-008: Batched attention (reduces 54 kernel launches to 9, correct output)
    let attn_out =
        batched_multihead_attention_optimized(ctx, &q, &k, &v, n_heads, head_dim, seq_len)?;

    if debug {
        let attn_host = attn_out.peek_host()?;
        eprintln!(
            "[DEBUG-GPU-INTERNAL] attn_out: mean={:.6}, std={:.6}",
            attn_host.iter().sum::<f32>() / attn_host.len() as f32,
            (attn_host.iter().map(|v| v.powi(2)).sum::<f32>() / attn_host.len() as f32).sqrt()
        );
    }

    // Output projection
    let attn_proj = attn_out.linear(
        ctx,
        &weights.w_o,
        Some(&weights.b_o),
        seq_len,
        d_model,
        d_model,
    )?;

    if debug {
        let proj_host = attn_proj.peek_host()?;
        eprintln!(
            "[DEBUG-GPU-INTERNAL] attn_proj: mean={:.6}, std={:.6}",
            proj_host.iter().sum::<f32>() / proj_host.len() as f32,
            (proj_host.iter().map(|v| v.powi(2)).sum::<f32>() / proj_host.len() as f32).sqrt()
        );
    }

    // Residual connection: x + attn_proj
    let residual1 = x.add(ctx, &attn_proj)?;

    if debug {
        let res1_host = residual1.peek_host()?;
        eprintln!(
            "[DEBUG-GPU-INTERNAL] residual1: mean={:.6}, std={:.6}",
            res1_host.iter().sum::<f32>() / res1_host.len() as f32,
            (res1_host.iter().map(|v| v.powi(2)).sum::<f32>() / res1_host.len() as f32).sqrt()
        );
    }

    // ====== FFN Block ======

    // Pre-norm: x_norm2 = LayerNorm(residual1)
    let x_norm2 =
        residual1.layer_norm(ctx, &weights.ln2_gamma, &weights.ln2_beta, d_model, seq_len)?;

    if debug {
        let ln2_host = x_norm2.peek_host()?;
        eprintln!(
            "[DEBUG-GPU-INTERNAL] LN2 output: mean={:.6}, std={:.6}",
            ln2_host.iter().sum::<f32>() / ln2_host.len() as f32,
            (ln2_host.iter().map(|v| v.powi(2)).sum::<f32>() / ln2_host.len() as f32).sqrt()
        );
    }

    // FFN up projection + GELU (FUSED - WAPR-PERF-007)
    // Uses single kernel instead of 3 (GEMM + Bias + GELU)
    let ffn_gelu = x_norm2.fused_linear_gelu(
        ctx,
        &weights.ffn_up_w,
        &weights.ffn_up_b,
        seq_len,
        d_model,
        ffn_dim,
    )?;

    if debug {
        let gelu_host = ffn_gelu.peek_host()?;
        eprintln!(
            "[DEBUG-GPU-INTERNAL] ffn_gelu (fused): mean={:.6}, std={:.6}",
            gelu_host.iter().sum::<f32>() / gelu_host.len() as f32,
            (gelu_host.iter().map(|v| v.powi(2)).sum::<f32>() / gelu_host.len() as f32).sqrt()
        );
    }

    // FFN down projection
    let ffn_down = ffn_gelu.linear(
        ctx,
        &weights.ffn_down_w,
        Some(&weights.ffn_down_b),
        seq_len,
        ffn_dim,
        d_model,
    )?;

    if debug {
        let down_host = ffn_down.peek_host()?;
        eprintln!(
            "[DEBUG-GPU-INTERNAL] ffn_down: mean={:.6}, std={:.6}",
            down_host.iter().sum::<f32>() / down_host.len() as f32,
            (down_host.iter().map(|v| v.powi(2)).sum::<f32>() / down_host.len() as f32).sqrt()
        );
    }

    // Residual connection: residual1 + ffn_down
    let output = residual1.add(ctx, &ffn_down)?;

    if debug {
        let out_host = output.peek_host()?;
        eprintln!(
            "[DEBUG-GPU-INTERNAL] block_output: mean={:.6}, std={:.6}",
            out_host.iter().sum::<f32>() / out_host.len() as f32,
            (out_host.iter().map(|v| v.powi(2)).sum::<f32>() / out_host.len() as f32).sqrt()
        );
    }

    Ok(output)
}
