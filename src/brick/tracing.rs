//! Model-Level Inference Tracing (Phase 13, E.11)
//!
//! Comprehensive tracing system for transformer model inference:
//! - MLT-01: LayerActivationTrace - anomaly detection per layer
//! - MLT-02: AttentionWeightTrace - attention pattern analysis
//! - MLT-03: LogitEvolutionTrace - token probability evolution
//! - MLT-04: QuantizationErrorTrace - quantization quality metrics
//! - MLT-05: KvCacheStateTrace - KV cache efficiency tracking
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno::brick::{ModelTracer, ModelTracerConfig};
//!
//! let config = ModelTracerConfig::lightweight();
//! let mut tracer = ModelTracer::new(config);
//!
//! tracer.begin_forward(position);
//! // ... forward pass with trace hooks ...
//! if let Some(anomaly) = tracer.end_forward() {
//!     log::warn!("Anomaly: {}", anomaly);
//! }
//! ```

use std::fmt;

use super::exec_graph::BrickId;

// ============================================================================
// QuantType - Quantization type tracking
// ============================================================================

/// Quantization type for tracking quantization errors (MLT-04).
///
/// Note: Variant names follow GGML conventions (e.g., Q4_K) for interoperability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[allow(non_camel_case_types)]
pub enum QuantType {
    /// Full precision (FP32)
    #[default]
    F32,
    /// Half precision (FP16)
    F16,
    /// Brain floating point (BF16)
    Bf16,
    /// 8-bit integer quantization
    Q8_0,
    /// 4-bit quantization (GGML)
    Q4_0,
    /// 4-bit quantization with k-quants
    Q4_K,
    /// 5-bit quantization with k-quants
    Q5_K,
    /// 6-bit quantization with k-quants
    Q6_K,
    /// 2-bit quantization
    Q2_K,
    /// 3-bit quantization
    Q3_K,
}

impl QuantType {
    /// Get bits per element for this quantization type.
    pub fn bits_per_element(self) -> f32 {
        match self {
            Self::F32 => 32.0,
            Self::F16 | Self::Bf16 => 16.0,
            Self::Q8_0 => 8.0,
            Self::Q6_K => 6.5,
            Self::Q5_K => 5.5,
            Self::Q4_0 | Self::Q4_K => 4.5,
            Self::Q3_K => 3.5,
            Self::Q2_K => 2.5,
        }
    }

    /// Get compression ratio vs FP32.
    pub fn compression_ratio(self) -> f32 {
        32.0 / self.bits_per_element()
    }
}

// ============================================================================
// E.11.2: LayerActivationTrace (MLT-01)
// ============================================================================

/// Statistics for a tensor without storing the tensor itself.
///
/// Computes min, max, mean, std, L2 norm, NaN/Inf counts in a single pass.
/// Used for anomaly detection (explosion, vanishing gradients, NaN propagation).
///
/// # Example
/// ```rust,ignore
/// let stats = TensorStats::from_slice(&tensor_data);
/// if stats.has_anomaly() {
///     log::warn!("Anomaly detected: {}", stats.anomaly_description());
/// }
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TensorStats {
    /// Number of elements analyzed
    pub count: usize,
    /// Minimum value (ignoring NaN/Inf)
    pub min: f32,
    /// Maximum value (ignoring NaN/Inf)
    pub max: f32,
    /// Mean value (ignoring NaN/Inf)
    pub mean: f32,
    /// Standard deviation (ignoring NaN/Inf)
    pub std: f32,
    /// Count of NaN values
    pub nan_count: usize,
    /// Count of Inf values
    pub inf_count: usize,
    /// L2 norm (sqrt of sum of squares)
    pub l2_norm: f32,
}

impl TensorStats {
    /// Compute statistics from a slice in a single pass.
    ///
    /// Uses Welford's algorithm for numerically stable mean/variance.
    pub fn from_slice(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let mut count = 0usize;
        let mut nan_count = 0usize;
        let mut inf_count = 0usize;
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut sum_sq = 0.0f64;

        // Welford's algorithm for online mean/variance
        let mut mean = 0.0f64;
        let mut m2 = 0.0f64;

        for &val in data {
            if val.is_nan() {
                nan_count += 1;
                continue;
            }
            if val.is_infinite() {
                inf_count += 1;
                continue;
            }

            count += 1;
            min = min.min(val);
            max = max.max(val);
            sum_sq += (val as f64) * (val as f64);

            // Welford's update
            let delta = val as f64 - mean;
            mean += delta / count as f64;
            let delta2 = val as f64 - mean;
            m2 += delta * delta2;
        }

        let std = if count > 1 {
            (m2 / (count - 1) as f64).sqrt() as f32
        } else {
            0.0
        };

        let l2_norm = sum_sq.sqrt() as f32;

        Self {
            count: data.len(),
            min: if count > 0 { min } else { 0.0 },
            max: if count > 0 { max } else { 0.0 },
            mean: mean as f32,
            std,
            nan_count,
            inf_count,
            l2_norm,
        }
    }

    /// Check if this tensor has any anomalies.
    ///
    /// Anomaly detection rules (from E.11.2):
    /// - NaN detected: `nan_count > 0`
    /// - Explosion: `max.abs() > 1e6` or `std > 1e4`
    /// - Vanishing: `std < 1e-6` (should check after first few layers)
    pub fn has_anomaly(&self) -> bool {
        self.nan_count > 0
            || self.inf_count > 0
            || self.max.abs() > 1e6
            || self.min.abs() > 1e6
            || self.std > 1e4
    }

    /// Check if values are vanishing (for layers past warmup).
    pub fn is_vanishing(&self) -> bool {
        self.std < 1e-6 && self.count > 0
    }

    /// Get a description of any anomaly detected.
    pub fn anomaly_description(&self) -> Option<String> {
        if self.nan_count > 0 {
            return Some(format!("NaN detected: {} values", self.nan_count));
        }
        if self.inf_count > 0 {
            return Some(format!("Inf detected: {} values", self.inf_count));
        }
        if self.max.abs() > 1e6 || self.min.abs() > 1e6 {
            return Some(format!(
                "Explosion: min={:.2e}, max={:.2e}",
                self.min, self.max
            ));
        }
        if self.std > 1e4 {
            return Some(format!("High variance: std={:.2e}", self.std));
        }
        None
    }
}

/// Activation trace for a single transformer layer.
///
/// Records tensor statistics at each stage of a transformer layer:
/// input → norm → attention → residual → ffn → output
#[derive(Debug, Clone, Default)]
pub struct LayerActivationTrace {
    /// Layer index (0-indexed)
    pub layer_idx: usize,
    /// Input hidden state statistics
    pub input_stats: TensorStats,
    /// After RMSNorm/LayerNorm statistics
    pub post_norm_stats: TensorStats,
    /// After attention statistics
    pub post_attn_stats: TensorStats,
    /// After FFN statistics
    pub post_ffn_stats: TensorStats,
    /// Output hidden state statistics
    pub output_stats: TensorStats,
    /// Residual connection magnitude ratio (output_norm / (output_norm + attn_norm))
    pub residual_ratio: f32,
}

impl LayerActivationTrace {
    /// Create a new layer activation trace.
    pub fn new(layer_idx: usize) -> Self {
        Self {
            layer_idx,
            ..Default::default()
        }
    }

    /// Check if this layer has any anomalies.
    pub fn has_anomaly(&self) -> bool {
        self.input_stats.has_anomaly()
            || self.post_norm_stats.has_anomaly()
            || self.post_attn_stats.has_anomaly()
            || self.post_ffn_stats.has_anomaly()
            || self.output_stats.has_anomaly()
            || self.residual_ratio > 0.99 // Skip connection bypass
    }

    /// Get anomaly description for this layer.
    pub fn anomaly_description(&self) -> Option<String> {
        if let Some(desc) = self.input_stats.anomaly_description() {
            return Some(format!("Layer {} input: {}", self.layer_idx, desc));
        }
        if let Some(desc) = self.post_norm_stats.anomaly_description() {
            return Some(format!("Layer {} post_norm: {}", self.layer_idx, desc));
        }
        if let Some(desc) = self.post_attn_stats.anomaly_description() {
            return Some(format!("Layer {} post_attn: {}", self.layer_idx, desc));
        }
        if let Some(desc) = self.post_ffn_stats.anomaly_description() {
            return Some(format!("Layer {} post_ffn: {}", self.layer_idx, desc));
        }
        if let Some(desc) = self.output_stats.anomaly_description() {
            return Some(format!("Layer {} output: {}", self.layer_idx, desc));
        }
        if self.residual_ratio > 0.99 {
            return Some(format!(
                "Layer {} residual dominance: ratio={:.4}",
                self.layer_idx, self.residual_ratio
            ));
        }
        None
    }
}

/// Full model activation trace for one forward pass.
#[derive(Debug, Clone, Default)]
pub struct ModelActivationTrace {
    /// Per-layer activation traces
    pub layers: Vec<LayerActivationTrace>,
    /// Embedding output statistics
    pub embedding_stats: TensorStats,
    /// Final logits statistics
    pub logits_stats: TensorStats,
    /// Whether any anomaly was detected
    pub has_anomaly: bool,
    /// Description of first anomaly found
    pub anomaly_desc: Option<String>,
}

impl ModelActivationTrace {
    /// Create a new model activation trace with expected layer count.
    pub fn with_capacity(num_layers: usize) -> Self {
        Self {
            layers: Vec::with_capacity(num_layers),
            ..Default::default()
        }
    }

    /// Add a layer trace.
    pub fn add_layer(&mut self, trace: LayerActivationTrace) {
        if !self.has_anomaly {
            if let Some(desc) = trace.anomaly_description() {
                self.has_anomaly = true;
                self.anomaly_desc = Some(desc);
            }
        }
        self.layers.push(trace);
    }

    /// Finalize the trace and check embedding/logits.
    pub fn finalize(&mut self) {
        if !self.has_anomaly {
            if let Some(desc) = self.embedding_stats.anomaly_description() {
                self.has_anomaly = true;
                self.anomaly_desc = Some(format!("Embedding: {}", desc));
            }
        }
        if !self.has_anomaly {
            if let Some(desc) = self.logits_stats.anomaly_description() {
                self.has_anomaly = true;
                self.anomaly_desc = Some(format!("Logits: {}", desc));
            }
        }
    }
}

// ============================================================================
// E.11.3: AttentionWeightTrace (MLT-02)
// ============================================================================

/// Sparse attention weight storage for a single head.
///
/// Records top-k attended positions to avoid storing the full attention matrix.
/// Useful for debugging repetition, context loss, and attention sinks.
#[derive(Debug, Clone, Default)]
pub struct AttentionWeightTrace {
    /// Layer index
    pub layer_idx: usize,
    /// Head index within the layer
    pub head_idx: usize,
    /// Query position (current token being generated)
    pub query_pos: usize,
    /// Top-k attended positions (sorted by weight descending)
    pub top_k_positions: Vec<usize>,
    /// Corresponding attention weights
    pub top_k_weights: Vec<f32>,
    /// Sum of weights outside top-k (attention mass lost to tail)
    pub tail_mass: f32,
    /// Entropy of attention distribution (higher = more uniform)
    pub entropy: f32,
}

impl AttentionWeightTrace {
    /// Create from full attention weights, extracting top-k.
    pub fn from_weights(
        layer_idx: usize,
        head_idx: usize,
        query_pos: usize,
        weights: &[f32],
        k: usize,
    ) -> Self {
        let k = k.min(weights.len());

        // Create position-weight pairs and sort by weight descending
        let mut pairs: Vec<(usize, f32)> = weights.iter().copied().enumerate().collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k_positions: Vec<usize> = pairs.iter().take(k).map(|(pos, _)| *pos).collect();
        let top_k_weights: Vec<f32> = pairs.iter().take(k).map(|(_, w)| *w).collect();

        let top_k_mass: f32 = top_k_weights.iter().sum();
        let total_mass: f32 = weights.iter().sum();
        let tail_mass = (total_mass - top_k_mass).max(0.0);

        // Compute entropy: H = -sum(p * log(p)) for non-zero probabilities
        let entropy = weights
            .iter()
            .filter(|&&w| w > 1e-10)
            .map(|&w| -w * w.ln())
            .sum();

        Self {
            layer_idx,
            head_idx,
            query_pos,
            top_k_positions,
            top_k_weights,
            tail_mass,
            entropy,
        }
    }

    /// Check if attention is concentrated on first position (attention sink).
    pub fn is_attention_sink(&self, threshold: f32) -> bool {
        self.top_k_positions.first() == Some(&0)
            && self.top_k_weights.first().copied().unwrap_or(0.0) > threshold
    }

    /// Check if attention is too uniform (confused model).
    pub fn is_uniform(&self, entropy_threshold: f32) -> bool {
        self.entropy > entropy_threshold
    }

    /// Check for repetition pattern (high weight on recent positions).
    pub fn has_recency_bias(&self, recency_window: usize, threshold: f32) -> bool {
        if self.query_pos == 0 {
            return false;
        }
        let recency_start = self.query_pos.saturating_sub(recency_window);
        let recent_mass: f32 = self
            .top_k_positions
            .iter()
            .zip(self.top_k_weights.iter())
            .filter(|(pos, _)| **pos >= recency_start)
            .map(|(_, w)| w)
            .sum();
        recent_mass > threshold
    }
}

/// Configuration for attention weight tracing.
#[derive(Debug, Clone)]
pub struct AttentionTraceConfig {
    /// Number of top positions to record per head
    pub top_k: usize,
    /// Layers to trace (None = all)
    pub layers: Option<Vec<usize>>,
    /// Heads to trace (None = all)
    pub heads: Option<Vec<usize>>,
    /// Minimum weight to consider (positions with weight below this are ignored)
    pub weight_threshold: f32,
}

impl Default for AttentionTraceConfig {
    fn default() -> Self {
        Self {
            top_k: 10,
            layers: None,
            heads: None,
            weight_threshold: 0.01,
        }
    }
}

impl AttentionTraceConfig {
    /// Check if a layer should be traced.
    pub fn should_trace_layer(&self, layer_idx: usize) -> bool {
        self.layers
            .as_ref()
            .is_none_or(|layers| layers.contains(&layer_idx))
    }

    /// Check if a head should be traced.
    pub fn should_trace_head(&self, head_idx: usize) -> bool {
        self.heads
            .as_ref()
            .is_none_or(|heads| heads.contains(&head_idx))
    }
}

// ============================================================================
// E.11.4: LogitEvolutionTrace (MLT-03)
// ============================================================================

/// Logit evolution for a single token through layers.
///
/// Tracks how a token's logit value and rank change as hidden states
/// pass through transformer layers.
#[derive(Debug, Clone, Default)]
pub struct TokenLogitEvolution {
    /// Token ID being tracked
    pub token_id: u32,
    /// Token string representation (for display)
    pub token_str: String,
    /// Logit value after each layer's contribution
    pub per_layer_logit: Vec<f32>,
    /// Rank among vocabulary at each layer (0 = highest probability)
    pub per_layer_rank: Vec<usize>,
    /// Final probability after softmax
    pub final_probability: f32,
    /// Final rank (0 = selected token)
    pub final_rank: usize,
}

impl TokenLogitEvolution {
    /// Create a new token evolution tracker.
    pub fn new(token_id: u32, token_str: String) -> Self {
        Self {
            token_id,
            token_str,
            ..Default::default()
        }
    }

    /// Record logit value at a layer.
    pub fn record_layer(&mut self, logit: f32, rank: usize) {
        self.per_layer_logit.push(logit);
        self.per_layer_rank.push(rank);
    }

    /// Get the layer where this token's rank changed most dramatically.
    pub fn decisive_layer(&self) -> Option<usize> {
        if self.per_layer_rank.len() < 2 {
            return None;
        }

        let mut max_change = 0i64;
        let mut decisive = 0;

        for i in 1..self.per_layer_rank.len() {
            let change =
                (self.per_layer_rank[i] as i64 - self.per_layer_rank[i - 1] as i64).abs();
            if change > max_change {
                max_change = change;
                decisive = i;
            }
        }

        Some(decisive)
    }
}

/// Full logit trace for one generation step.
#[derive(Debug, Clone, Default)]
pub struct LogitEvolutionTrace {
    /// Position being generated
    pub position: usize,
    /// Tokens being tracked (typically top-k candidates + ground truth)
    pub tracked_tokens: Vec<TokenLogitEvolution>,
    /// Which layer had the largest impact on the selected token
    pub decisive_layer: usize,
    /// Temperature used for sampling
    pub temperature: f32,
    /// Top-p (nucleus) value used
    pub top_p: f32,
}

impl LogitEvolutionTrace {
    /// Create a new logit evolution trace.
    pub fn new(position: usize, temperature: f32, top_p: f32) -> Self {
        Self {
            position,
            temperature,
            top_p,
            ..Default::default()
        }
    }

    /// Add a token to track.
    pub fn track_token(&mut self, token_id: u32, token_str: String) -> &mut TokenLogitEvolution {
        self.tracked_tokens
            .push(TokenLogitEvolution::new(token_id, token_str));
        self.tracked_tokens.last_mut().expect("invariant: just pushed")
    }

    /// Compute rank of a token in a logit distribution.
    pub fn compute_rank(logits: &[f32], token_id: u32) -> usize {
        let target_logit = logits.get(token_id as usize).copied().unwrap_or(f32::MIN);

        logits.iter().filter(|&&l| l > target_logit).count()
    }

    /// Finalize the trace after generation completes.
    pub fn finalize(&mut self, selected_token_id: u32) {
        // Find the decisive layer for the selected token
        for token in &self.tracked_tokens {
            if token.token_id == selected_token_id {
                if let Some(layer) = token.decisive_layer() {
                    self.decisive_layer = layer;
                }
                break;
            }
        }
    }
}

// ============================================================================
// E.11.5: QuantizationErrorTrace (MLT-04)
// ============================================================================

/// Quantization error measurement for a single operation.
///
/// Compares quantized computation against FP32 reference using multiple metrics.
#[derive(Debug, Clone)]
pub struct QuantizationErrorTrace {
    /// Brick type being measured
    pub brick_id: BrickId,
    /// Layer index
    pub layer_idx: usize,
    /// Mean squared error vs FP32 reference
    pub mse: f32,
    /// Maximum absolute error
    pub max_abs_error: f32,
    /// Cosine similarity (1.0 = perfect match)
    pub cosine_similarity: f32,
    /// Signal-to-noise ratio in dB
    pub snr_db: f32,
    /// Quantization type used
    pub quant_type: QuantType,
}

impl QuantizationErrorTrace {
    /// Compute error metrics between quantized and reference outputs.
    pub fn compute(
        brick_id: BrickId,
        layer_idx: usize,
        quantized: &[f32],
        reference: &[f32],
        quant_type: QuantType,
    ) -> Self {
        assert_eq!(quantized.len(), reference.len(), "Length mismatch");
        let n = quantized.len();
        if n == 0 {
            return Self {
                brick_id,
                layer_idx,
                mse: 0.0,
                max_abs_error: 0.0,
                cosine_similarity: 1.0, // Perfect match when both empty
                snr_db: f32::INFINITY,
                quant_type,
            };
        }

        // MSE and max abs error
        let mut sum_sq_error = 0.0f64;
        let mut max_abs_error = 0.0f32;
        for (q, r) in quantized.iter().zip(reference.iter()) {
            let error = q - r;
            sum_sq_error += (error as f64) * (error as f64);
            max_abs_error = max_abs_error.max(error.abs());
        }
        let mse = (sum_sq_error / n as f64) as f32;

        // Cosine similarity
        let mut dot = 0.0f64;
        let mut norm_q = 0.0f64;
        let mut norm_r = 0.0f64;
        for (q, r) in quantized.iter().zip(reference.iter()) {
            dot += (*q as f64) * (*r as f64);
            norm_q += (*q as f64) * (*q as f64);
            norm_r += (*r as f64) * (*r as f64);
        }
        let cosine_similarity = if norm_q > 0.0 && norm_r > 0.0 {
            (dot / (norm_q.sqrt() * norm_r.sqrt())) as f32
        } else {
            0.0
        };

        // SNR in dB: 10 * log10(signal_power / noise_power)
        let signal_power = norm_r / n as f64;
        let noise_power = sum_sq_error / n as f64;
        let snr_db = if noise_power > 1e-10 {
            (10.0 * (signal_power / noise_power).log10()) as f32
        } else {
            f32::INFINITY
        };

        Self {
            brick_id,
            layer_idx,
            mse,
            max_abs_error,
            cosine_similarity,
            snr_db,
            quant_type,
        }
    }

    /// Check if error is acceptable (cosine > 0.995).
    pub fn is_acceptable(&self) -> bool {
        self.cosine_similarity > 0.995
    }

    /// Check if error is in warning zone (0.99 < cosine < 0.995).
    pub fn is_warning(&self) -> bool {
        self.cosine_similarity > 0.99 && self.cosine_similarity <= 0.995
    }

    /// Check if error is critical (cosine < 0.99).
    pub fn is_critical(&self) -> bool {
        self.cosine_similarity < 0.99
    }
}

/// Cumulative quantization error across an entire model.
#[derive(Debug, Clone, Default)]
pub struct ModelQuantizationError {
    /// Per-brick error traces
    pub brick_errors: Vec<QuantizationErrorTrace>,
    /// Overall cosine similarity of final logits
    pub logits_cosine: f32,
    /// KL divergence of output probability distributions
    pub output_kl_divergence: f32,
    /// Perplexity difference (PPL_quant - PPL_fp32)
    pub perplexity_delta: f32,
}

impl ModelQuantizationError {
    /// Add a brick error trace.
    pub fn add_error(&mut self, trace: QuantizationErrorTrace) {
        self.brick_errors.push(trace);
    }

    /// Get count of critical errors.
    pub fn critical_count(&self) -> usize {
        self.brick_errors.iter().filter(|e| e.is_critical()).count()
    }

    /// Get count of warning errors.
    pub fn warning_count(&self) -> usize {
        self.brick_errors.iter().filter(|e| e.is_warning()).count()
    }

    /// Get worst brick by cosine similarity.
    pub fn worst_brick(&self) -> Option<&QuantizationErrorTrace> {
        self.brick_errors
            .iter()
            .min_by(|a, b| a.cosine_similarity.partial_cmp(&b.cosine_similarity).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// ============================================================================
// E.11.6: KvCacheStateTrace (MLT-05)
// ============================================================================

/// KV cache state at a single generation step.
#[derive(Debug, Clone, Default)]
pub struct KvCacheStateTrace {
    /// Generation step (0-indexed)
    pub step: usize,
    /// Total cache size in bytes
    pub cache_size_bytes: usize,
    /// Number of valid (filled) positions in cache
    pub valid_positions: usize,
    /// Maximum positions (context window size)
    pub max_positions: usize,
    /// Evictions performed this step
    pub evictions_this_step: usize,
    /// Cache hit rate (reused positions / total lookups)
    pub cache_hit_rate: f32,
    /// Oldest position still in cache
    pub oldest_position: usize,
    /// Memory fragmentation (0.0 = compact, 1.0 = fully scattered)
    pub fragmentation: f32,
    /// Positions accessed this step (for locality analysis)
    pub accessed_positions: Vec<usize>,
}

impl KvCacheStateTrace {
    /// Create a new trace for a step.
    pub fn new(step: usize, max_positions: usize) -> Self {
        Self {
            step,
            max_positions,
            ..Default::default()
        }
    }

    /// Check if context window is exhausted.
    pub fn is_window_exhausted(&self) -> bool {
        self.valid_positions >= self.max_positions
    }

    /// Get cache utilization ratio.
    pub fn utilization(&self) -> f32 {
        if self.max_positions == 0 {
            return 0.0;
        }
        self.valid_positions as f32 / self.max_positions as f32
    }
}

/// Full KV cache trace for a generation session.
#[derive(Debug, Clone, Default)]
pub struct KvCacheSessionTrace {
    /// Per-step traces
    pub steps: Vec<KvCacheStateTrace>,
    /// Total evictions across the session
    pub total_evictions: usize,
    /// Average cache hit rate
    pub avg_hit_rate: f32,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
}

impl KvCacheSessionTrace {
    /// Add a step trace.
    pub fn add_step(&mut self, trace: KvCacheStateTrace) {
        self.total_evictions += trace.evictions_this_step;
        self.peak_memory_bytes = self.peak_memory_bytes.max(trace.cache_size_bytes);

        // Update rolling average
        let n = self.steps.len() as f32 + 1.0;
        self.avg_hit_rate =
            (self.avg_hit_rate * (n - 1.0) + trace.cache_hit_rate) / n;

        self.steps.push(trace);
    }

    /// Check if eviction rate is concerning (>10% of steps).
    pub fn has_high_eviction_rate(&self) -> bool {
        if self.steps.is_empty() {
            return false;
        }
        let eviction_steps = self.steps.iter().filter(|s| s.evictions_this_step > 0).count();
        eviction_steps as f32 / self.steps.len() as f32 > 0.1
    }

    /// Check if KV cache is thrashing (high evictions + low hit rate).
    ///
    /// Returns true if the recent window shows both high eviction rate and low hit rate.
    /// Uses all available steps if fewer than `window` steps exist.
    ///
    /// # Arguments
    /// - `window`: Number of recent steps to consider (uses available if fewer)
    /// - `min_hit_rate`: Minimum acceptable hit rate (0.0-1.0)
    pub fn has_thrashing(&self, window: usize, min_hit_rate: f32) -> bool {
        if self.steps.is_empty() {
            return false;
        }

        // Use all steps if fewer than window
        let actual_window = std::cmp::min(window, self.steps.len());
        let recent_steps = &self.steps[self.steps.len() - actual_window..];
        let recent_evictions: usize = recent_steps.iter().map(|s| s.evictions_this_step).sum();
        let recent_hit_rate: f32 =
            recent_steps.iter().map(|s| s.cache_hit_rate).sum::<f32>() / actual_window as f32;

        // Thrashing: more than half the steps have evictions AND hit rate below threshold
        recent_evictions > actual_window / 2 && recent_hit_rate < min_hit_rate
    }
}

// ============================================================================
// E.11.7: Unified ModelTracer
// ============================================================================

/// Configuration for model-level tracing.
#[derive(Debug, Clone, Default)]
pub struct ModelTracerConfig {
    /// Enable layer activation tracing (MLT-01)
    pub trace_activations: bool,
    /// Enable attention weight tracing (MLT-02)
    pub trace_attention: bool,
    /// Attention trace configuration
    pub attention_config: AttentionTraceConfig,
    /// Enable logit evolution tracing (MLT-03)
    pub trace_logits: bool,
    /// Specific tokens to track (None = auto-select top-k)
    pub tracked_tokens: Option<Vec<u32>>,
    /// Enable quantization error tracing (MLT-04) - expensive!
    pub trace_quant_error: bool,
    /// Enable KV cache state tracing (MLT-05)
    pub trace_kv_cache: bool,
}

impl ModelTracerConfig {
    /// Create a config that traces everything (for debugging).
    pub fn full() -> Self {
        Self {
            trace_activations: true,
            trace_attention: true,
            attention_config: AttentionTraceConfig::default(),
            trace_logits: true,
            tracked_tokens: None,
            trace_quant_error: true,
            trace_kv_cache: true,
        }
    }

    /// Create a lightweight config (activations + KV cache only).
    pub fn lightweight() -> Self {
        Self {
            trace_activations: true,
            trace_kv_cache: true,
            ..Default::default()
        }
    }

    /// Check if any tracing is enabled.
    pub fn is_enabled(&self) -> bool {
        self.trace_activations
            || self.trace_attention
            || self.trace_logits
            || self.trace_quant_error
            || self.trace_kv_cache
    }
}

/// Unified model tracer that coordinates all trace types.
///
/// # Example
/// ```rust,ignore
/// let config = ModelTracerConfig::lightweight();
/// let mut tracer = ModelTracer::new(config);
///
/// tracer.begin_forward(position);
/// // ... forward pass with trace hooks ...
/// if let Some(anomaly) = tracer.end_forward() {
///     log::warn!("Anomaly: {}", anomaly);
/// }
/// ```
pub struct ModelTracer {
    config: ModelTracerConfig,
    /// Current forward pass position
    current_position: usize,
    /// Accumulated activation traces
    activation_traces: Vec<ModelActivationTrace>,
    /// Current activation trace (in progress)
    current_activation_trace: Option<ModelActivationTrace>,
    /// Accumulated attention traces
    attention_traces: Vec<AttentionWeightTrace>,
    /// Accumulated logit evolution traces
    logit_traces: Vec<LogitEvolutionTrace>,
    /// Current logit trace (in progress)
    current_logit_trace: Option<LogitEvolutionTrace>,
    /// Accumulated quantization error traces
    quant_traces: Vec<ModelQuantizationError>,
    /// KV cache session trace
    kv_trace: KvCacheSessionTrace,
}

impl ModelTracer {
    /// Create a new tracer with the given configuration.
    pub fn new(config: ModelTracerConfig) -> Self {
        Self {
            config,
            current_position: 0,
            activation_traces: Vec::new(),
            current_activation_trace: None,
            attention_traces: Vec::new(),
            logit_traces: Vec::new(),
            current_logit_trace: None,
            quant_traces: Vec::new(),
            kv_trace: KvCacheSessionTrace::default(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &ModelTracerConfig {
        &self.config
    }

    /// Get a reference to the current logit trace (if any).
    pub fn current_logit_trace(&self) -> Option<&LogitEvolutionTrace> {
        self.current_logit_trace.as_ref()
    }

    /// Set the current logit trace (for testing purposes).
    pub fn set_current_logit_trace(&mut self, trace: Option<LogitEvolutionTrace>) {
        self.current_logit_trace = trace;
    }

    /// Begin a forward pass at the given position.
    pub fn begin_forward(&mut self, position: usize) {
        self.current_position = position;

        if self.config.trace_activations {
            self.current_activation_trace = Some(ModelActivationTrace::default());
        }

        if self.config.trace_logits {
            self.current_logit_trace = Some(LogitEvolutionTrace::new(position, 1.0, 1.0));
        }
    }

    /// Record layer activation (called by executor after each layer).
    pub fn record_layer_activation(&mut self, trace: LayerActivationTrace) {
        if let Some(ref mut activation) = self.current_activation_trace {
            activation.add_layer(trace);
        }
    }

    /// Record attention weights (called by attention brick).
    pub fn record_attention(&mut self, trace: AttentionWeightTrace) {
        if self.config.trace_attention {
            self.attention_traces.push(trace);
        }
    }

    /// Record logit state at a layer (called by lm_head or probe).
    pub fn record_logits(&mut self, layer_idx: usize, logits: &[f32]) {
        if let Some(ref mut logit_trace) = self.current_logit_trace {
            for token_evo in &mut logit_trace.tracked_tokens {
                let logit = logits.get(token_evo.token_id as usize).copied().unwrap_or(0.0);
                let rank = LogitEvolutionTrace::compute_rank(logits, token_evo.token_id);
                token_evo.record_layer(logit, rank);
            }
            // Store decisive layer based on rank changes
            logit_trace.decisive_layer = layer_idx;
        }
    }

    /// Record KV cache state (called after each generation step).
    pub fn record_kv_state(&mut self, trace: KvCacheStateTrace) {
        if self.config.trace_kv_cache {
            self.kv_trace.add_step(trace);
        }
    }

    /// Record quantization error for a brick.
    pub fn record_quant_error(&mut self, trace: QuantizationErrorTrace) {
        if self.config.trace_quant_error {
            if self.quant_traces.is_empty() {
                self.quant_traces.push(ModelQuantizationError::default());
            }
            if let Some(model_error) = self.quant_traces.last_mut() {
                model_error.add_error(trace);
            }
        }
    }

    /// Complete forward pass and check for anomalies.
    ///
    /// Returns a description of the first anomaly detected, if any.
    pub fn end_forward(&mut self) -> Option<String> {
        let mut anomaly = None;

        // Finalize activation trace
        if let Some(mut trace) = self.current_activation_trace.take() {
            trace.finalize();
            if trace.has_anomaly {
                anomaly = trace.anomaly_desc.clone();
            }
            self.activation_traces.push(trace);
        }

        // Finalize logit trace
        if let Some(trace) = self.current_logit_trace.take() {
            self.logit_traces.push(trace);
        }

        anomaly
    }

    /// Get summary statistics.
    pub fn summary(&self) -> ModelTracerSummary {
        ModelTracerSummary {
            total_forwards: self.activation_traces.len(),
            anomalies_detected: self.activation_traces.iter().filter(|t| t.has_anomaly).count(),
            attention_traces: self.attention_traces.len(),
            logit_traces: self.logit_traces.len(),
            kv_steps: self.kv_trace.steps.len(),
            total_evictions: self.kv_trace.total_evictions,
            avg_hit_rate: self.kv_trace.avg_hit_rate,
            quant_warnings: self.quant_traces.iter().map(|t| t.warning_count()).sum(),
            quant_criticals: self.quant_traces.iter().map(|t| t.critical_count()).sum(),
        }
    }

    /// Export summary as JSON for artifact validation.
    pub fn summary_to_json(&self) -> String {
        let summary = self.summary();
        format!(
            r#"{{"total_forwards":{},"anomalies_detected":{},"attention_traces":{},"logit_traces":{},"kv_steps":{},"total_evictions":{},"avg_hit_rate":{:.4},"quant_warnings":{},"quant_criticals":{}}}"#,
            summary.total_forwards,
            summary.anomalies_detected,
            summary.attention_traces,
            summary.logit_traces,
            summary.kv_steps,
            summary.total_evictions,
            summary.avg_hit_rate,
            summary.quant_warnings,
            summary.quant_criticals
        )
    }

    /// Clear all accumulated traces (free memory).
    pub fn clear(&mut self) {
        self.activation_traces.clear();
        self.attention_traces.clear();
        self.logit_traces.clear();
        self.quant_traces.clear();
        self.kv_trace = KvCacheSessionTrace::default();
    }
}

/// Summary of model tracer state.
#[derive(Debug, Clone, Default)]
pub struct ModelTracerSummary {
    /// Total forward passes traced
    pub total_forwards: usize,
    /// Number of forward passes with anomalies
    pub anomalies_detected: usize,
    /// Total attention traces collected
    pub attention_traces: usize,
    /// Total logit evolution traces
    pub logit_traces: usize,
    /// Total KV cache steps traced
    pub kv_steps: usize,
    /// Total KV cache evictions
    pub total_evictions: usize,
    /// Average KV cache hit rate
    pub avg_hit_rate: f32,
    /// Quantization warning count
    pub quant_warnings: usize,
    /// Quantization critical count
    pub quant_criticals: usize,
}

impl fmt::Display for ModelTracerSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ModelTracer Summary:")?;
        writeln!(f, "  Forward passes: {}", self.total_forwards)?;
        writeln!(f, "  Anomalies: {}", self.anomalies_detected)?;
        writeln!(f, "  Attention traces: {}", self.attention_traces)?;
        writeln!(f, "  Logit traces: {}", self.logit_traces)?;
        writeln!(f, "  KV cache steps: {}", self.kv_steps)?;
        writeln!(f, "  KV evictions: {}", self.total_evictions)?;
        writeln!(f, "  Avg hit rate: {:.2}%", self.avg_hit_rate * 100.0)?;
        writeln!(f, "  Quant warnings: {}", self.quant_warnings)?;
        write!(f, "  Quant criticals: {}", self.quant_criticals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // QuantType Tests
    // ========================================================================

    #[test]
    fn test_quant_type_bits() {
        assert_eq!(QuantType::F32.bits_per_element(), 32.0);
        assert_eq!(QuantType::F16.bits_per_element(), 16.0);
        assert_eq!(QuantType::Q8_0.bits_per_element(), 8.0);
        assert_eq!(QuantType::Q4_K.bits_per_element(), 4.5);
    }

    #[test]
    fn test_quant_type_compression_ratio() {
        // F32 -> F32 = 1x
        assert!((QuantType::F32.compression_ratio() - 1.0).abs() < 0.01);
        // F32 -> F16 = 2x
        assert!((QuantType::F16.compression_ratio() - 2.0).abs() < 0.01);
        // F32 -> Q4_K = ~7.1x
        assert!(QuantType::Q4_K.compression_ratio() > 7.0);
    }

    // ========================================================================
    // TensorStats Tests
    // ========================================================================

    #[test]
    fn test_tensor_stats_empty() {
        let stats = TensorStats::from_slice(&[]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.nan_count, 0);
        assert!(!stats.has_anomaly());
    }

    #[test]
    fn test_tensor_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStats::from_slice(&data);
        assert_eq!(stats.count, 5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!((stats.mean - 3.0).abs() < 0.01);
        assert!(!stats.has_anomaly());
    }

    #[test]
    fn test_tensor_stats_nan_detection() {
        let data = vec![1.0, f32::NAN, 3.0];
        let stats = TensorStats::from_slice(&data);
        assert_eq!(stats.nan_count, 1);
        assert!(stats.has_anomaly());
        assert!(stats.anomaly_description().unwrap().contains("NaN"));
    }

    #[test]
    fn test_tensor_stats_inf_detection() {
        let data = vec![1.0, f32::INFINITY, 3.0];
        let stats = TensorStats::from_slice(&data);
        assert_eq!(stats.inf_count, 1);
        assert!(stats.has_anomaly());
    }

    #[test]
    fn test_tensor_stats_explosion() {
        let data = vec![1e7, 2e7];
        let stats = TensorStats::from_slice(&data);
        assert!(stats.has_anomaly());
        assert!(stats.anomaly_description().unwrap().contains("Explosion"));
    }

    #[test]
    fn test_tensor_stats_vanishing() {
        let data = vec![1e-8, 1e-8, 1e-8];
        let stats = TensorStats::from_slice(&data);
        assert!(stats.is_vanishing());
    }

    // ========================================================================
    // LayerActivationTrace Tests
    // ========================================================================

    #[test]
    fn test_layer_activation_trace_new() {
        let trace = LayerActivationTrace::new(5);
        assert_eq!(trace.layer_idx, 5);
        assert!(!trace.has_anomaly());
    }

    #[test]
    fn test_layer_activation_trace_anomaly() {
        let mut trace = LayerActivationTrace::new(0);
        trace.input_stats = TensorStats::from_slice(&[f32::NAN]);
        assert!(trace.has_anomaly());
        assert!(trace.anomaly_description().is_some());
    }

    #[test]
    fn test_layer_activation_trace_residual_dominance() {
        let mut trace = LayerActivationTrace::new(0);
        trace.residual_ratio = 0.999;
        assert!(trace.has_anomaly());
        assert!(trace.anomaly_description().unwrap().contains("residual"));
    }

    // ========================================================================
    // ModelActivationTrace Tests
    // ========================================================================

    #[test]
    fn test_model_activation_trace_add_layer() {
        let mut trace = ModelActivationTrace::with_capacity(32);
        trace.add_layer(LayerActivationTrace::new(0));
        trace.add_layer(LayerActivationTrace::new(1));
        assert_eq!(trace.layers.len(), 2);
        assert!(!trace.has_anomaly);
    }

    #[test]
    fn test_model_activation_trace_anomaly_propagation() {
        let mut trace = ModelActivationTrace::default();
        let mut bad_layer = LayerActivationTrace::new(0);
        bad_layer.input_stats = TensorStats::from_slice(&[f32::NAN]);
        trace.add_layer(bad_layer);
        assert!(trace.has_anomaly);
    }

    // ========================================================================
    // AttentionWeightTrace Tests
    // ========================================================================

    #[test]
    fn test_attention_weight_trace_from_weights() {
        let weights = vec![0.1, 0.3, 0.4, 0.2];
        let trace = AttentionWeightTrace::from_weights(0, 0, 3, &weights, 2);

        assert_eq!(trace.layer_idx, 0);
        assert_eq!(trace.head_idx, 0);
        assert_eq!(trace.query_pos, 3);
        assert_eq!(trace.top_k_positions.len(), 2);
        // Position 2 has highest weight (0.4), then position 1 (0.3)
        assert_eq!(trace.top_k_positions[0], 2);
        assert_eq!(trace.top_k_positions[1], 1);
    }

    #[test]
    fn test_attention_sink_detection() {
        let weights = vec![0.8, 0.1, 0.05, 0.05];
        let trace = AttentionWeightTrace::from_weights(0, 0, 3, &weights, 4);
        assert!(trace.is_attention_sink(0.5));
    }

    #[test]
    fn test_recency_bias_detection() {
        // Position 3 attending mostly to positions 1 and 2
        let weights = vec![0.05, 0.4, 0.5, 0.05];
        let trace = AttentionWeightTrace::from_weights(0, 0, 3, &weights, 4);
        assert!(trace.has_recency_bias(2, 0.5));
    }

    // ========================================================================
    // TokenLogitEvolution Tests
    // ========================================================================

    #[test]
    fn test_token_logit_evolution() {
        let mut evo = TokenLogitEvolution::new(42, "test".to_string());
        evo.record_layer(1.0, 100);
        evo.record_layer(2.0, 50);
        evo.record_layer(3.0, 10);

        assert_eq!(evo.per_layer_logit.len(), 3);
        assert_eq!(evo.per_layer_rank.len(), 3);
        assert_eq!(evo.decisive_layer(), Some(1)); // 100->50 is biggest jump
    }

    #[test]
    fn test_logit_evolution_trace_compute_rank() {
        let logits = vec![1.0, 5.0, 3.0, 2.0]; // sorted: 5, 3, 2, 1
        // Token 0 has logit 1.0, rank 3 (3 values above it)
        assert_eq!(LogitEvolutionTrace::compute_rank(&logits, 0), 3);
        // Token 1 has logit 5.0, rank 0 (nothing above it)
        assert_eq!(LogitEvolutionTrace::compute_rank(&logits, 1), 0);
    }

    // ========================================================================
    // QuantizationErrorTrace Tests
    // ========================================================================

    #[test]
    fn test_quant_error_perfect_match() {
        let reference = vec![1.0, 2.0, 3.0];
        let quantized = vec![1.0, 2.0, 3.0];
        let trace = QuantizationErrorTrace::compute(
            BrickId::RmsNorm,
            0,
            &quantized,
            &reference,
            QuantType::Q4_K,
        );

        assert!((trace.mse - 0.0).abs() < 1e-6);
        assert!((trace.cosine_similarity - 1.0).abs() < 1e-6);
        assert!(trace.is_acceptable());
    }

    #[test]
    fn test_quant_error_significant_difference() {
        let reference = vec![1.0, 2.0, 3.0];
        // Non-proportional: adds different offsets, changing direction
        let quantized = vec![1.5, 2.1, 2.9];
        let trace = QuantizationErrorTrace::compute(
            BrickId::RmsNorm,
            0,
            &quantized,
            &reference,
            QuantType::Q4_K,
        );

        assert!(trace.mse > 0.0);
        assert!(trace.cosine_similarity < 1.0);
        // Cosine should still be high since vectors are close
        assert!(trace.cosine_similarity > 0.99);
    }

    // ========================================================================
    // KvCacheStateTrace Tests
    // ========================================================================

    #[test]
    fn test_kv_cache_state_trace() {
        let trace = KvCacheStateTrace::new(0, 2048);
        assert_eq!(trace.step, 0);
        assert_eq!(trace.max_positions, 2048);
        assert!(!trace.is_window_exhausted());
    }

    #[test]
    fn test_kv_cache_state_utilization() {
        let mut trace = KvCacheStateTrace::new(0, 1000);
        trace.valid_positions = 500;
        assert!((trace.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_kv_cache_session_trace() {
        let mut session = KvCacheSessionTrace::default();
        session.add_step(KvCacheStateTrace {
            step: 0,
            cache_hit_rate: 0.9,
            evictions_this_step: 0,
            cache_size_bytes: 1000,
            ..Default::default()
        });
        session.add_step(KvCacheStateTrace {
            step: 1,
            cache_hit_rate: 0.8,
            evictions_this_step: 5,
            cache_size_bytes: 2000,
            ..Default::default()
        });

        assert_eq!(session.steps.len(), 2);
        assert_eq!(session.total_evictions, 5);
        assert_eq!(session.peak_memory_bytes, 2000);
        assert!((session.avg_hit_rate - 0.85).abs() < 0.01);
    }

    // ========================================================================
    // ModelTracer Tests
    // ========================================================================

    #[test]
    fn test_model_tracer_lightweight() {
        let config = ModelTracerConfig::lightweight();
        assert!(config.trace_activations);
        assert!(config.trace_kv_cache);
        assert!(!config.trace_attention);
        assert!(!config.trace_quant_error);
    }

    #[test]
    fn test_model_tracer_full() {
        let config = ModelTracerConfig::full();
        assert!(config.trace_activations);
        assert!(config.trace_attention);
        assert!(config.trace_logits);
        assert!(config.trace_quant_error);
        assert!(config.trace_kv_cache);
    }

    #[test]
    fn test_model_tracer_forward_pass() {
        let config = ModelTracerConfig::lightweight();
        let mut tracer = ModelTracer::new(config);

        tracer.begin_forward(0);
        tracer.record_layer_activation(LayerActivationTrace::new(0));
        tracer.record_layer_activation(LayerActivationTrace::new(1));
        let anomaly = tracer.end_forward();

        assert!(anomaly.is_none());
        let summary = tracer.summary();
        assert_eq!(summary.total_forwards, 1);
        assert_eq!(summary.anomalies_detected, 0);
    }

    #[test]
    fn test_model_tracer_detects_anomaly() {
        let config = ModelTracerConfig::lightweight();
        let mut tracer = ModelTracer::new(config);

        tracer.begin_forward(0);
        let mut bad_layer = LayerActivationTrace::new(0);
        bad_layer.input_stats = TensorStats::from_slice(&[f32::NAN]);
        tracer.record_layer_activation(bad_layer);
        let anomaly = tracer.end_forward();

        assert!(anomaly.is_some());
        assert!(anomaly.unwrap().contains("NaN"));
        assert_eq!(tracer.summary().anomalies_detected, 1);
    }

    #[test]
    fn test_model_tracer_json_output() {
        let config = ModelTracerConfig::lightweight();
        let mut tracer = ModelTracer::new(config);

        tracer.begin_forward(0);
        tracer.end_forward();

        let json = tracer.summary_to_json();
        assert!(json.contains("\"total_forwards\":1"));
        assert!(json.contains("\"anomalies_detected\":0"));
    }

    // ========================================================================
    // Falsification Tests
    // ========================================================================

    /// FALSIFICATION TEST: TensorStats Welford algorithm numerical stability
    ///
    /// Welford's algorithm must produce correct mean/std even for large values.
    #[test]
    fn test_falsify_tensor_stats_welford_stability() {
        // Test with large offset - naive algorithm would lose precision
        let large_offset = 1e9;
        let data: Vec<f32> = (0..1000).map(|i| large_offset + i as f32).collect();
        let stats = TensorStats::from_slice(&data);

        // Mean should be large_offset + 499.5
        let expected_mean = large_offset + 499.5;
        assert!(
            (stats.mean - expected_mean as f32).abs() < 1.0,
            "FALSIFICATION FAILED: Welford mean {} != expected {} (relative error too high)",
            stats.mean,
            expected_mean
        );

        // Std should be ~288.7 (uniform distribution 0-999)
        assert!(
            stats.std > 280.0 && stats.std < 300.0,
            "FALSIFICATION FAILED: Welford std {} outside expected range [280, 300]",
            stats.std
        );
    }

    /// FALSIFICATION TEST: Cosine similarity must be 1.0 for identical vectors
    #[test]
    fn test_falsify_cosine_identical_vectors() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trace = QuantizationErrorTrace::compute(
            BrickId::RmsNorm,
            0,
            &data,
            &data,
            QuantType::F32,
        );

        assert!(
            (trace.cosine_similarity - 1.0).abs() < 1e-6,
            "FALSIFICATION FAILED: identical vectors have cosine {} != 1.0",
            trace.cosine_similarity
        );
    }

    /// FALSIFICATION TEST: Cosine similarity must be symmetric
    #[test]
    fn test_falsify_cosine_symmetry() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let trace_ab = QuantizationErrorTrace::compute(
            BrickId::RmsNorm, 0, &a, &b, QuantType::F32,
        );
        let trace_ba = QuantizationErrorTrace::compute(
            BrickId::RmsNorm, 0, &b, &a, QuantType::F32,
        );

        assert!(
            (trace_ab.cosine_similarity - trace_ba.cosine_similarity).abs() < 1e-6,
            "FALSIFICATION FAILED: cosine(a,b) {} != cosine(b,a) {}",
            trace_ab.cosine_similarity,
            trace_ba.cosine_similarity
        );
    }

    /// FALSIFICATION TEST: ModelTracer layer count must match recorded layers
    #[test]
    fn test_falsify_tracer_layer_count() {
        let config = ModelTracerConfig::lightweight();
        let mut tracer = ModelTracer::new(config);

        tracer.begin_forward(0);
        let num_layers = 32;
        for i in 0..num_layers {
            tracer.record_layer_activation(LayerActivationTrace::new(i));
        }
        tracer.end_forward();

        // The activation trace should have exactly num_layers entries
        assert_eq!(
            tracer.activation_traces[0].layers.len(),
            num_layers,
            "FALSIFICATION FAILED: recorded {} layers but expected {}",
            tracer.activation_traces[0].layers.len(),
            num_layers
        );
    }
}
