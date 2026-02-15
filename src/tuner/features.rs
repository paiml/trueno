//! Feature Extraction for ML Tuning
//!
//! Implements TunerFeatures, TunerFeaturesBuilder, FeatureExtractor, and RunConfig.

use crate::brick::{BrickCategory, BrickProfiler};
use crate::hardware::HardwareCapability;
use serde::{Deserialize, Serialize};

use super::error::TunerError;
use super::types::{BottleneckClass, KernelType, QuantType};

// ============================================================================
// TunerFeatures
// ============================================================================

/// Feature vector for ML-based kernel tuning.
///
/// All fields normalized to [0, 1] for model input.
/// Total dimension: 42 features.
///
/// # Feature Categories
///
/// - **Static (11)**: Known before execution (model size, batch size, etc.)
/// - **Quant one-hot (8)**: Quantization type encoding
/// - **Kernel one-hot (16)**: Kernel type encoding
/// - **Hardware (5)**: GPU capabilities
/// - **Derived (2)**: Computed features (arithmetic intensity, efficiency)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunerFeatures {
    // === Static features (11) ===
    /// Model size in billions (log10 normalized)
    pub model_params_b: f32,
    /// Hidden dimension / 16384
    pub hidden_dim_norm: f32,
    /// Number of layers / 128
    pub num_layers_norm: f32,
    /// Number of attention heads / 128
    pub num_heads_norm: f32,
    /// Head dimension / 256
    pub head_dim_norm: f32,
    /// Vocabulary size (log10 normalized)
    pub vocab_size_log: f32,
    /// Batch size M / 64
    pub batch_size_norm: f32,
    /// Sequence length (log2 / 15)
    pub seq_len_log: f32,
    /// CUDA graphs enabled (0 or 1)
    pub cuda_graphs: f32,
    /// Number of KV caches / batch_size (for multi-cache detection)
    pub kv_cache_ratio: f32,
    /// Prefill vs decode (0=decode, 1=prefill)
    pub is_prefill: f32,

    // === Quantization one-hot (8) ===
    pub quant_type_onehot: [f32; 8],

    // === Kernel one-hot (16) ===
    pub kernel_type_onehot: [f32; 16],

    // === Hardware features (5) === [v1.1.0: added L2 cache + zero-copy]
    /// Memory bandwidth / 3000 GB/s
    pub gpu_mem_bw_norm: f32,
    /// Compute TFLOPS / 500
    pub gpu_compute_norm: f32,
    /// SM count / 200
    pub gpu_sm_norm: f32,
    /// L2 cache size / 128 MB (v1.1.0: critical for occupancy)
    pub gpu_l2_cache_norm: f32,
    /// Zero-copy memory path enabled (0 or 1) (v1.1.0: pinned memory)
    pub is_zero_copy: f32,

    // === Derived features (2) ===
    /// Arithmetic intensity (FLOP/byte), normalized
    pub arithmetic_intensity: f32,
    /// Theoretical efficiency (measured / roofline)
    pub theoretical_efficiency: f32,

    // === Labels (for training) ===
    /// Measured throughput (tokens/second) - training label
    #[serde(skip_serializing_if = "Option::is_none")]
    pub measured_tps: Option<f32>,
    /// Best kernel ID - classification label
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_kernel_id: Option<u8>,
    /// Bottleneck class - classification label
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bottleneck_class: Option<BottleneckClass>,
}

impl Default for TunerFeatures {
    fn default() -> Self {
        Self {
            model_params_b: 0.0,
            hidden_dim_norm: 0.0,
            num_layers_norm: 0.0,
            num_heads_norm: 0.0,
            head_dim_norm: 0.0,
            vocab_size_log: 0.0,
            batch_size_norm: 0.0,
            seq_len_log: 0.0,
            cuda_graphs: 0.0,
            kv_cache_ratio: 1.0,
            is_prefill: 0.0,
            quant_type_onehot: [0.0; 8],
            kernel_type_onehot: [0.0; 16],
            gpu_mem_bw_norm: 0.0,
            gpu_compute_norm: 0.0,
            gpu_sm_norm: 0.0,
            gpu_l2_cache_norm: 0.0,
            is_zero_copy: 0.0,
            arithmetic_intensity: 0.0,
            theoretical_efficiency: 0.0,
            measured_tps: None,
            best_kernel_id: None,
            bottleneck_class: None,
        }
    }
}

impl TunerFeatures {
    /// Total feature dimension (excluding labels)
    /// v1.1.0: 11 static + 8 quant + 16 kernel + 5 hardware + 2 derived = 42
    pub const DIM: usize = 11 + 8 + 16 + 5 + 2; // 42 features (v1.1.0)

    /// Create a new feature builder
    pub fn builder() -> TunerFeaturesBuilder {
        TunerFeaturesBuilder::default()
    }

    /// Convert to flat vector for model input
    pub fn to_vector(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(Self::DIM);

        // Static features
        v.push(self.model_params_b);
        v.push(self.hidden_dim_norm);
        v.push(self.num_layers_norm);
        v.push(self.num_heads_norm);
        v.push(self.head_dim_norm);
        v.push(self.vocab_size_log);
        v.push(self.batch_size_norm);
        v.push(self.seq_len_log);
        v.push(self.cuda_graphs);
        v.push(self.kv_cache_ratio);
        v.push(self.is_prefill);

        // One-hot encodings
        v.extend_from_slice(&self.quant_type_onehot);
        v.extend_from_slice(&self.kernel_type_onehot);

        // Hardware features (5) [v1.1.0]
        v.push(self.gpu_mem_bw_norm);
        v.push(self.gpu_compute_norm);
        v.push(self.gpu_sm_norm);
        v.push(self.gpu_l2_cache_norm); // v1.1.0
        v.push(self.is_zero_copy); // v1.1.0

        // Derived features
        v.push(self.arithmetic_intensity);
        v.push(self.theoretical_efficiency);

        v
    }

    /// Validate features (F021-F030 falsification criteria)
    pub fn validate(&self) -> Result<(), TunerError> {
        let v = self.to_vector();

        // F021: No NaN features
        if v.iter().any(|x| x.is_nan()) {
            return Err(TunerError::InvalidFeature("NaN value in features".into()));
        }

        // F022: No infinite features
        if v.iter().any(|x| x.is_infinite()) {
            return Err(TunerError::InvalidFeature(
                "Infinite value in features".into(),
            ));
        }

        // F023: All features in [0, 1] (with small tolerance for floating point)
        if v.iter().any(|x| *x < -0.001 || *x > 1.001) {
            return Err(TunerError::InvalidFeature(
                "Feature value outside [0, 1]".into(),
            ));
        }

        // F029: One-hot sums = 1
        let quant_sum: f32 = self.quant_type_onehot.iter().sum();
        if (quant_sum - 1.0).abs() > 0.001 && quant_sum > 0.001 {
            return Err(TunerError::InvalidFeature(
                "Quant one-hot does not sum to 1".into(),
            ));
        }

        let kernel_sum: f32 = self.kernel_type_onehot.iter().sum();
        if (kernel_sum - 1.0).abs() > 0.001 && kernel_sum > 0.001 {
            return Err(TunerError::InvalidFeature(
                "Kernel one-hot does not sum to 1".into(),
            ));
        }

        Ok(())
    }
}

// ============================================================================
// TunerFeaturesBuilder
// ============================================================================

/// Builder for TunerFeatures with automatic normalization.
#[derive(Default)]
pub struct TunerFeaturesBuilder {
    model_params_b: Option<f32>,
    hidden_dim: Option<u32>,
    num_layers: Option<u32>,
    num_heads: Option<u32>,
    head_dim: Option<u32>,
    vocab_size: Option<u32>,
    batch_size: Option<u32>,
    seq_len: Option<u32>,
    cuda_graphs: bool,
    kv_caches: Option<u32>,
    is_prefill: bool,
    quant_type: Option<QuantType>,
    kernel_type: Option<KernelType>,
    gpu_mem_bw_gbs: Option<f32>,
    gpu_compute_tflops: Option<f32>,
    gpu_sm_count: Option<u32>,
    gpu_l2_cache_mb: Option<f32>, // v1.1.0
    is_zero_copy: bool,           // v1.1.0
    measured_tps: Option<f32>,
}

impl TunerFeaturesBuilder {
    /// Set model size in billions of parameters
    pub fn model_params_b(mut self, params: f32) -> Self {
        self.model_params_b = Some(params);
        self
    }

    /// Set hidden dimension
    pub fn hidden_dim(mut self, dim: u32) -> Self {
        self.hidden_dim = Some(dim);
        self
    }

    /// Set number of layers
    pub fn num_layers(mut self, layers: u32) -> Self {
        self.num_layers = Some(layers);
        self
    }

    /// Set number of attention heads
    pub fn num_heads(mut self, heads: u32) -> Self {
        self.num_heads = Some(heads);
        self
    }

    /// Set head dimension
    pub fn head_dim(mut self, dim: u32) -> Self {
        self.head_dim = Some(dim);
        self
    }

    /// Set vocabulary size
    pub fn vocab_size(mut self, size: u32) -> Self {
        self.vocab_size = Some(size);
        self
    }

    /// Set batch size (M)
    pub fn batch_size(mut self, m: u32) -> Self {
        self.batch_size = Some(m);
        self
    }

    /// Set sequence length
    pub fn seq_len(mut self, len: u32) -> Self {
        self.seq_len = Some(len);
        self
    }

    /// Enable CUDA graphs
    pub fn cuda_graphs(mut self, enabled: bool) -> Self {
        self.cuda_graphs = enabled;
        self
    }

    /// Set number of KV caches
    pub fn kv_caches(mut self, count: u32) -> Self {
        self.kv_caches = Some(count);
        self
    }

    /// Set prefill mode
    pub fn is_prefill(mut self, prefill: bool) -> Self {
        self.is_prefill = prefill;
        self
    }

    /// Set quantization type
    pub fn quant_type(mut self, qt: QuantType) -> Self {
        self.quant_type = Some(qt);
        self
    }

    /// Set kernel type
    pub fn kernel_type(mut self, kt: KernelType) -> Self {
        self.kernel_type = Some(kt);
        self
    }

    /// Set GPU memory bandwidth in GB/s
    pub fn gpu_mem_bw_gbs(mut self, bw: f32) -> Self {
        self.gpu_mem_bw_gbs = Some(bw);
        self
    }

    /// Set GPU compute in TFLOPS
    pub fn gpu_compute_tflops(mut self, tflops: f32) -> Self {
        self.gpu_compute_tflops = Some(tflops);
        self
    }

    /// Set GPU SM count
    pub fn gpu_sm_count(mut self, count: u32) -> Self {
        self.gpu_sm_count = Some(count);
        self
    }

    /// Set measured throughput (for training data)
    pub fn measured_tps(mut self, tps: f32) -> Self {
        self.measured_tps = Some(tps);
        self
    }

    /// Set L2 cache size in MB (v1.1.0)
    pub fn gpu_l2_cache_mb(mut self, l2_mb: f32) -> Self {
        self.gpu_l2_cache_mb = Some(l2_mb);
        self
    }

    /// Set zero-copy memory path enabled (v1.1.0)
    pub fn is_zero_copy(mut self, enabled: bool) -> Self {
        self.is_zero_copy = enabled;
        self
    }

    /// Set hardware capability (auto-fills GPU features)
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    // SAFETY: GPU bandwidth/TFLOPS values fit in f32 (practical max ~10K); f64→f32 truncation negligible.
    pub fn hardware(mut self, hw: &HardwareCapability) -> Self {
        if let Some(gpu) = &hw.gpu {
            self.gpu_mem_bw_gbs = Some(gpu.memory_bw_gbps as f32);
            self.gpu_compute_tflops = Some(gpu.peak_tflops_fp32 as f32);
            // SM count not directly available; estimate from compute capability
            self.gpu_sm_count = None;
        }
        self
    }

    /// Build the feature vector with normalization
    #[allow(clippy::cast_precision_loss)]
    // SAFETY: All u32 values are model hyperparams (hidden_dim ≤ 16384, layers ≤ 128,
    //         heads ≤ 128, vocab ≤ 1M, batch ≤ 64, seq_len ≤ 32768) — well within f32 mantissa.
    pub fn build(self) -> TunerFeatures {
        let batch_size = self.batch_size.unwrap_or(1);
        let kv_caches = self.kv_caches.unwrap_or(batch_size);

        // Create one-hot encodings
        let mut quant_onehot = [0.0f32; 8];
        if let Some(qt) = self.quant_type {
            quant_onehot[qt.to_index()] = 1.0;
        }

        let mut kernel_onehot = [0.0f32; 16];
        if let Some(kt) = self.kernel_type {
            kernel_onehot[kt.to_index()] = 1.0;
        }

        // Calculate derived features
        let hidden_dim = self.hidden_dim.unwrap_or(1536) as f32;
        let batch_size_f = batch_size as f32;
        let quant_bytes = self
            .quant_type
            .map(|q| q.bytes_per_param())
            .unwrap_or(0.5625);

        // Arithmetic intensity for GEMV: 2*N*K FLOPs / (N*K*bytes + K + N) bytes
        // Simplified: ~2 / bytes_per_param for memory-bound inference
        let arithmetic_intensity = (2.0 / quant_bytes).min(10.0) / 10.0;

        // Theoretical efficiency starts at 0 (unknown until measured)
        let theoretical_efficiency = 0.0;

        TunerFeatures {
            // Normalized static features
            model_params_b: self
                .model_params_b
                .map(|p| (p.log10() + 1.0) / 3.0) // log10(0.1)=-1, log10(100)=2 → [0, 1]
                .unwrap_or(0.0)
                .clamp(0.0, 1.0),
            hidden_dim_norm: (hidden_dim / 16384.0).clamp(0.0, 1.0),
            num_layers_norm: (self.num_layers.unwrap_or(28) as f32 / 128.0).clamp(0.0, 1.0),
            num_heads_norm: (self.num_heads.unwrap_or(12) as f32 / 128.0).clamp(0.0, 1.0),
            head_dim_norm: (self.head_dim.unwrap_or(128) as f32 / 256.0).clamp(0.0, 1.0),
            vocab_size_log: self
                .vocab_size
                .map(|v| (v as f32).log10() / 6.0) // log10(1M)=6
                .unwrap_or(0.0)
                .clamp(0.0, 1.0),
            batch_size_norm: (batch_size_f / 64.0).clamp(0.0, 1.0),
            seq_len_log: self
                .seq_len
                .map(|s| (s as f32).log2() / 15.0) // log2(32K)≈15
                .unwrap_or(0.0)
                .clamp(0.0, 1.0),
            cuda_graphs: if self.cuda_graphs { 1.0 } else { 0.0 },
            kv_cache_ratio: (kv_caches as f32 / batch_size_f).clamp(0.0, 1.0),
            is_prefill: if self.is_prefill { 1.0 } else { 0.0 },

            // One-hot encodings
            quant_type_onehot: quant_onehot,
            kernel_type_onehot: kernel_onehot,

            // Hardware features (5) [v1.1.0]
            gpu_mem_bw_norm: (self.gpu_mem_bw_gbs.unwrap_or(1000.0) / 3000.0).clamp(0.0, 1.0),
            gpu_compute_norm: (self.gpu_compute_tflops.unwrap_or(100.0) / 500.0).clamp(0.0, 1.0),
            gpu_sm_norm: (self.gpu_sm_count.unwrap_or(128) as f32 / 200.0).clamp(0.0, 1.0),
            gpu_l2_cache_norm: (self.gpu_l2_cache_mb.unwrap_or(48.0) / 128.0).clamp(0.0, 1.0), // v1.1.0
            is_zero_copy: if self.is_zero_copy { 1.0 } else { 0.0 }, // v1.1.0

            // Derived features
            arithmetic_intensity,
            theoretical_efficiency,

            // Labels
            measured_tps: self.measured_tps,
            best_kernel_id: None,
            bottleneck_class: None,
        }
    }
}

// ============================================================================
// FeatureExtractor
// ============================================================================

/// Extracts features from BrickProfiler and runtime configuration.
#[derive(Debug)]
pub struct FeatureExtractor {
    /// Hardware capability (cached)
    pub(crate) hardware: Option<HardwareCapability>,
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new() -> Self {
        Self { hardware: None }
    }

    /// Create with hardware capability
    pub fn with_hardware(hardware: HardwareCapability) -> Self {
        Self {
            hardware: Some(hardware),
        }
    }

    /// Extract features from profiler and configuration
    pub fn extract(&self, profiler: &BrickProfiler, config: &RunConfig) -> TunerFeatures {
        let mut builder = TunerFeatures::builder()
            .model_params_b(config.model_params_b)
            .hidden_dim(config.hidden_dim)
            .num_layers(config.num_layers)
            .num_heads(config.num_heads)
            .batch_size(config.batch_size)
            .seq_len(config.seq_len)
            .cuda_graphs(config.cuda_graphs)
            .quant_type(config.quant_type)
            .kernel_type(config.kernel_type);

        // Add hardware features if available
        if let Some(hw) = &self.hardware {
            builder = builder.hardware(hw);
        }

        // Add measured throughput if available
        if let Some(tps) = profiler.tokens_per_sec() {
            builder = builder.measured_tps(tps);
        }

        let mut features = builder.build();

        // Update derived features from profiler
        if let Some(efficiency) = self.calculate_efficiency(profiler, config) {
            features.theoretical_efficiency = efficiency;
        }

        // Classify bottleneck from profiler data
        features.bottleneck_class = Some(self.classify_bottleneck(profiler));

        features
    }

    /// Calculate efficiency from profiler data
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    // SAFETY: GPU bandwidth f64→f32 truncation is negligible for roofline efficiency calculation.
    pub fn calculate_efficiency(
        &self,
        profiler: &BrickProfiler,
        config: &RunConfig,
    ) -> Option<f32> {
        let measured_tps = profiler.tokens_per_sec()?;
        let hw = self.hardware.as_ref()?;
        let gpu = hw.gpu.as_ref()?;

        // Calculate theoretical max based on roofline
        let bytes_per_token = config.model_params_b * 1e9 * config.quant_type.bytes_per_param();
        let theoretical_tps = (gpu.memory_bw_gbps as f32) * 1e9 / bytes_per_token;

        Some((measured_tps / theoretical_tps).clamp(0.0, 1.0))
    }

    /// Classify bottleneck from profiler brick breakdown.
    ///
    /// PAR-200: Uses category_stats() for efficient aggregation.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    // SAFETY: percentage() returns f64 in 0–100; f64→f32 truncation is negligible for threshold comparisons.
    pub fn classify_bottleneck(&self, profiler: &BrickProfiler) -> BottleneckClass {
        let cats = profiler.category_stats();
        let total_ns = profiler.total_ns();

        if total_ns == 0 {
            return BottleneckClass::Unknown;
        }

        // Get category percentages
        let attention_pct =
            cats[BrickCategory::Attention as usize].percentage(total_ns) as f32 / 100.0;
        let ffn_pct = cats[BrickCategory::Ffn as usize].percentage(total_ns) as f32 / 100.0;
        let norm_pct = cats[BrickCategory::Norm as usize].percentage(total_ns) as f32 / 100.0;

        // Classify based on dominant component
        if attention_pct > 0.35 {
            BottleneckClass::AttentionBound
        } else if ffn_pct > 0.50 {
            // FFN is memory-bound (large GEMV operations)
            BottleneckClass::MemoryBound
        } else if norm_pct > 0.20 {
            // High norm percentage indicates launch overhead
            BottleneckClass::LaunchBound
        } else {
            BottleneckClass::MemoryBound // Default for inference
        }
    }
}

// ============================================================================
// RunConfig
// ============================================================================

/// Runtime configuration for feature extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub model_params_b: f32,
    pub hidden_dim: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub batch_size: u32,
    pub seq_len: u32,
    pub cuda_graphs: bool,
    pub quant_type: QuantType,
    pub kernel_type: KernelType,
}

/// Default hidden dimension for 1.5B parameter model
const DEFAULT_HIDDEN_DIM: u32 = 1536;

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            model_params_b: 1.5,
            hidden_dim: DEFAULT_HIDDEN_DIM,
            num_layers: 28,
            num_heads: 12,
            batch_size: 1,
            seq_len: 1,
            cuda_graphs: false,
            quant_type: QuantType::Q4K,
            kernel_type: KernelType::VectorizedQ4K,
        }
    }
}
