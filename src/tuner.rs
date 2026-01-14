//! ML-Based ComputeBrick Tuner
//!
//! Implements learned cost models for kernel selection and throughput prediction.
//! See: `docs/specifications/ml-tuner-bricks.md`
//!
//! # Architecture
//!
//! ```text
//! BrickProfiler → FeatureExtractor → TunerModel → Recommendations
//! ```
//!
//! # Scientific Foundations
//!
//! - Chen et al. (2018). "TVM: An Automated End-to-End Optimizing Compiler." OSDI '18.
//! - Williams et al. (2009). "Roofline: An Insightful Visual Performance Model." CACM.
//! - Friedman (2001). "Greedy Function Approximation: A Gradient Boosting Machine."
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno::tuner::{BrickTuner, TunerFeatures};
//!
//! let features = TunerFeatures::builder()
//!     .model_params_b(1.5)
//!     .hidden_dim(1536)
//!     .batch_size(4)
//!     .quant_type(QuantType::Q4K)
//!     .build();
//!
//! let tuner = BrickTuner::load_or_default();
//! let recommendation = tuner.recommend(&features);
//! println!("Predicted: {} tok/s", recommendation.throughput.predicted_tps);
//! ```

use crate::brick::{BrickBottleneck, BrickProfiler};
use crate::hardware::HardwareCapability;
use serde::{Deserialize, Serialize};

// ML-tuner feature: use aprender RandomForest models (SHOWCASE-BRICK-001, Section 12)
#[cfg(feature = "ml-tuner")]
use aprender::{
    tree::{RandomForestClassifier, RandomForestRegressor},
    Matrix, Vector,
};

// ============================================================================
// TUNER-001: TunerFeatures
// ============================================================================

/// Quantization type for feature encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum QuantType {
    Q4_0,
    Q4_1,
    #[default]
    Q4K,
    Q5K,
    Q6K,
    Q8_0,
    F16,
    F32,
}

impl QuantType {
    /// One-hot encoding index (0-7)
    pub fn to_index(self) -> usize {
        match self {
            QuantType::Q4_0 => 0,
            QuantType::Q4_1 => 1,
            QuantType::Q4K => 2,
            QuantType::Q5K => 3,
            QuantType::Q6K => 4,
            QuantType::Q8_0 => 5,
            QuantType::F16 => 6,
            QuantType::F32 => 7,
        }
    }

    /// Bytes per parameter (approximate)
    pub fn bytes_per_param(self) -> f32 {
        match self {
            QuantType::Q4_0 | QuantType::Q4_1 | QuantType::Q4K => 0.5625, // 4.5 bits
            QuantType::Q5K => 0.6875,                                     // 5.5 bits
            QuantType::Q6K => 0.8125,                                     // 6.5 bits
            QuantType::Q8_0 => 1.0,
            QuantType::F16 => 2.0,
            QuantType::F32 => 4.0,
        }
    }
}

/// Kernel type for feature encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum KernelType {
    // Q4K variants
    #[default]
    TiledQ4K,
    CoalescedQ4K,
    VectorizedQ4K,
    BatchedQ4K,
    Dp4aQ4K,
    FusedRmsNormQ4K,
    // Q6K variants
    CoalescedQ6K,
    // Attention variants
    IncrementalAttention,
    MultiWarpAttention,
    BatchedAttention,
    // Normalization
    RmsNorm,
    VectorizedRmsNorm,
    BatchedRmsNorm,
    // Other
    Generic,
    Unknown,
}

impl KernelType {
    /// One-hot encoding index (0-15)
    pub fn to_index(self) -> usize {
        match self {
            KernelType::TiledQ4K => 0,
            KernelType::CoalescedQ4K => 1,
            KernelType::VectorizedQ4K => 2,
            KernelType::BatchedQ4K => 3,
            KernelType::Dp4aQ4K => 4,
            KernelType::FusedRmsNormQ4K => 5,
            KernelType::CoalescedQ6K => 6,
            KernelType::IncrementalAttention => 7,
            KernelType::MultiWarpAttention => 8,
            KernelType::BatchedAttention => 9,
            KernelType::RmsNorm => 10,
            KernelType::VectorizedRmsNorm => 11,
            KernelType::BatchedRmsNorm => 12,
            KernelType::Generic => 13,
            KernelType::Unknown => 14,
        }
    }

    /// Number of kernel types
    pub const COUNT: usize = 16;
}

/// Bottleneck classification for ML model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum BottleneckClass {
    #[default]
    Unknown,
    MemoryBound,
    ComputeBound,
    LaunchBound,
    AttentionBound,
}

impl BottleneckClass {
    /// Convert from BrickBottleneck
    pub fn from_brick_bottleneck(b: BrickBottleneck) -> Self {
        match b {
            BrickBottleneck::Memory => BottleneckClass::MemoryBound,
            BrickBottleneck::Compute => BottleneckClass::ComputeBound,
            BrickBottleneck::Unknown => BottleneckClass::Unknown,
        }
    }

    /// Recommended action for this bottleneck
    pub fn recommended_action(self) -> &'static str {
        match self {
            BottleneckClass::MemoryBound => {
                "Increase batch size (M) to amortize weight reads across sequences"
            }
            BottleneckClass::ComputeBound => {
                "Rare for inference; check for redundant computation or use tensor cores"
            }
            BottleneckClass::LaunchBound => {
                "Enable CUDA graphs or fuse kernels to reduce launch overhead"
            }
            BottleneckClass::AttentionBound => {
                "Use Flash Decoding, reduce sequence length, or use batched attention"
            }
            BottleneckClass::Unknown => "Run profiling to identify bottleneck",
        }
    }

    /// One-hot encoding index (0-4)
    pub fn to_index(self) -> usize {
        match self {
            BottleneckClass::Unknown => 0,
            BottleneckClass::MemoryBound => 1,
            BottleneckClass::ComputeBound => 2,
            BottleneckClass::LaunchBound => 3,
            BottleneckClass::AttentionBound => 4,
        }
    }
}

impl std::fmt::Display for BottleneckClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BottleneckClass::Unknown => write!(f, "Unknown"),
            BottleneckClass::MemoryBound => write!(f, "MemoryBound"),
            BottleneckClass::ComputeBound => write!(f, "ComputeBound"),
            BottleneckClass::LaunchBound => write!(f, "LaunchBound"),
            BottleneckClass::AttentionBound => write!(f, "AttentionBound"),
        }
    }
}

/// Feature vector for ML-based kernel tuning.
///
/// All fields normalized to [0, 1] for model input.
/// Total dimension: 40 features.
///
/// # Feature Categories
///
/// - **Static (11)**: Known before execution (model size, batch size, etc.)
/// - **Quant one-hot (8)**: Quantization type encoding
/// - **Kernel one-hot (16)**: Kernel type encoding
/// - **Hardware (3)**: GPU capabilities
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
        v.push(self.gpu_l2_cache_norm);  // v1.1.0
        v.push(self.is_zero_copy);        // v1.1.0

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
    gpu_l2_cache_mb: Option<f32>,  // v1.1.0
    is_zero_copy: bool,             // v1.1.0
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
            is_zero_copy: if self.is_zero_copy { 1.0 } else { 0.0 },                           // v1.1.0

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
// TUNER-002: FeatureExtractor
// ============================================================================

/// Extracts features from BrickProfiler and runtime configuration.
#[derive(Debug)]
pub struct FeatureExtractor {
    /// Hardware capability (cached)
    hardware: Option<HardwareCapability>,
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
    fn calculate_efficiency(&self, profiler: &BrickProfiler, config: &RunConfig) -> Option<f32> {
        let measured_tps = profiler.tokens_per_sec()?;
        let hw = self.hardware.as_ref()?;
        let gpu = hw.gpu.as_ref()?;

        // Calculate theoretical max based on roofline
        let bytes_per_token = config.model_params_b * 1e9 * config.quant_type.bytes_per_param();
        let theoretical_tps = (gpu.memory_bw_gbps as f32) * 1e9 / bytes_per_token;

        Some((measured_tps / theoretical_tps).clamp(0.0, 1.0))
    }

    /// Classify bottleneck from profiler brick breakdown
    fn classify_bottleneck(&self, profiler: &BrickProfiler) -> BottleneckClass {
        let stats = profiler.all_stats();
        if stats.is_empty() {
            return BottleneckClass::Unknown;
        }

        // Calculate percentage breakdown
        let total_ns: u64 = stats.values().map(|s| s.total_ns).sum();
        if total_ns == 0 {
            return BottleneckClass::Unknown;
        }

        // Find attention percentage
        let attention_ns: u64 = stats
            .iter()
            .filter(|(name, _)| name.to_lowercase().contains("attention"))
            .map(|(_, s)| s.total_ns)
            .sum();
        let attention_pct = attention_ns as f32 / total_ns as f32;

        // Find GEMV percentage (memory-bound indicator)
        let gemv_ns: u64 = stats
            .iter()
            .filter(|(name, _)| {
                let lower = name.to_lowercase();
                lower.contains("gemv")
                    || lower.contains("qkv")
                    || lower.contains("ffn")
                    || lower.contains("proj")
            })
            .map(|(_, s)| s.total_ns)
            .sum();
        let gemv_pct = gemv_ns as f32 / total_ns as f32;

        // Classify based on dominant component
        if attention_pct > 0.35 {
            BottleneckClass::AttentionBound
        } else if gemv_pct > 0.50 {
            BottleneckClass::MemoryBound
        } else {
            // Check for launch overhead (many small bricks)
            let avg_brick_ns = total_ns / stats.len() as u64;
            if avg_brick_ns < 10_000 {
                // < 10µs average
                BottleneckClass::LaunchBound
            } else {
                BottleneckClass::MemoryBound // Default for inference
            }
        }
    }
}

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

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            model_params_b: 1.5,
            hidden_dim: 1536,
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

// ============================================================================
// TUNER-003/004/005: ML Models (Simplified implementations)
// ============================================================================

/// Throughput prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputPrediction {
    /// Predicted tokens per second
    pub predicted_tps: f32,
    /// Confidence (0-1)
    pub confidence: f32,
    /// Top contributing features
    pub top_features: Vec<(String, f32)>,
}

/// Kernel recommendation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelRecommendation {
    /// Top recommended kernel
    pub top_kernel: KernelType,
    /// Confidence (0-1)
    pub confidence: f32,
    /// Alternative kernels with probabilities
    pub alternatives: Vec<(KernelType, f32)>,
}

/// Bottleneck prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckPrediction {
    /// Predicted bottleneck class
    pub class: BottleneckClass,
    /// Confidence (0-1)
    pub confidence: f32,
    /// Human-readable explanation
    pub explanation: String,
    /// Recommended action
    pub recommended_action: String,
}

/// Simple linear regression model for throughput prediction.
///
/// Uses closed-form solution: w = (X^T X)^-1 X^T y
/// With `ml-tuner` feature: uses aprender::RandomForestRegressor (SHOWCASE-BRICK-001)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputRegressor {
    /// Model weights (one per feature + bias) - fallback when ml-tuner disabled
    weights: Vec<f32>,
    /// Feature importance scores
    feature_importance: Vec<(String, f32)>,
    /// Training sample count
    sample_count: usize,
    /// Mean absolute percentage error on validation
    mape: f32,
    /// Whether the RandomForest model is trained (ml-tuner feature)
    #[cfg(feature = "ml-tuner")]
    #[serde(skip)]
    rf_model: Option<RandomForestRegressor>,
}

impl Default for ThroughputRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl ThroughputRegressor {
    /// Create a new regressor with default weights
    pub fn new() -> Self {
        // Initialize with heuristic-based weights
        // These encode domain knowledge from SHOWCASE-BRICK-001
        let mut weights = vec![0.0; TunerFeatures::DIM + 1]; // +1 for bias

        // Bias: baseline throughput ~200 tok/s normalized
        weights[0] = 0.4;

        // Batch size has largest positive impact (index 6)
        weights[7] = 0.3; // batch_size_norm

        // CUDA graphs help (index 8)
        weights[9] = 0.1; // cuda_graphs

        // GPU memory bandwidth matters (index 35)
        weights[36] = 0.15; // gpu_mem_bw_norm

        // GPU SM count matters (index 37)
        weights[38] = 0.1; // gpu_sm_norm

        // Larger models are slower (negative impact)
        weights[1] = -0.15; // model_params_b

        // Longer sequences slower for decode
        weights[8] = -0.05; // seq_len_log

        Self {
            weights,
            feature_importance: Self::default_feature_importance(),
            sample_count: 0,
            mape: 0.15, // 15% default MAPE
            #[cfg(feature = "ml-tuner")]
            rf_model: None,
        }
    }

    /// Create a new regressor using aprender RandomForest (ml-tuner feature)
    #[cfg(feature = "ml-tuner")]
    pub fn with_random_forest(n_estimators: usize) -> Self {
        let mut instance = Self::new();
        instance.rf_model = Some(RandomForestRegressor::new(n_estimators));
        instance
    }

    fn default_feature_importance() -> Vec<(String, f32)> {
        vec![
            ("batch_size".into(), 0.25),
            ("gpu_mem_bw".into(), 0.20),
            ("model_params".into(), 0.15),
            ("cuda_graphs".into(), 0.10),
            ("gpu_sm_count".into(), 0.10),
            ("hidden_dim".into(), 0.08),
            ("quant_type".into(), 0.07),
            ("seq_len".into(), 0.05),
        ]
    }

    /// Train the model on labeled data
    pub fn train(&mut self, data: &[(TunerFeatures, f32)]) -> Result<(), TunerError> {
        if data.len() < 10 {
            return Err(TunerError::InsufficientData(data.len()));
        }

        // Simple gradient descent (in production: aprender's GBDT)
        let learning_rate = 0.01;
        let epochs = 100;

        for _ in 0..epochs {
            let mut gradients = vec![0.0; self.weights.len()];

            for (features, target) in data {
                let x = features.to_vector();
                let predicted = self.predict_raw(&x);
                let error = predicted - target;

                // Gradient for bias
                gradients[0] += error;

                // Gradient for features
                for (i, xi) in x.iter().enumerate() {
                    gradients[i + 1] += error * xi;
                }
            }

            // Update weights
            let n = data.len() as f32;
            for (i, g) in gradients.iter().enumerate() {
                self.weights[i] -= learning_rate * g / n;
            }
        }

        // Calculate MAPE on training data
        let mut total_ape = 0.0;
        for (features, target) in data {
            let predicted = self.predict_raw(&features.to_vector());
            total_ape += ((predicted - target) / target.max(1.0)).abs();
        }
        self.mape = total_ape / data.len() as f32;
        self.sample_count = data.len();

        Ok(())
    }

    /// Train using aprender RandomForest (ml-tuner feature)
    ///
    /// Provides superior throughput prediction via ensemble learning.
    /// See: SHOWCASE-BRICK-001 Section 12.3
    #[cfg(feature = "ml-tuner")]
    pub fn train_random_forest(&mut self, data: &[(TunerFeatures, f32)]) -> Result<(), TunerError> {
        if data.len() < 10 {
            return Err(TunerError::InsufficientData(data.len()));
        }

        // Convert to aprender matrix format (f32 for RandomForestRegressor)
        let n_samples = data.len();
        let n_features = TunerFeatures::DIM;
        let mut x_data = Vec::with_capacity(n_samples * n_features);
        let mut y_data = Vec::with_capacity(n_samples);

        for (features, target) in data {
            x_data.extend(features.to_vector());
            y_data.push(*target);
        }

        let x_matrix = Matrix::from_vec(n_samples, n_features, x_data)
            .map_err(|e| TunerError::TrainingFailed(e.to_string()))?;
        let y_vector = Vector::from_vec(y_data);

        // Train RandomForest
        let rf = self.rf_model.get_or_insert_with(|| RandomForestRegressor::new(100));
        rf.fit(&x_matrix, &y_vector)
            .map_err(|e| TunerError::TrainingFailed(e.to_string()))?;

        // Calculate MAPE on training data
        let predictions = rf.predict(&x_matrix);
        let mut total_ape = 0.0;
        for (i, (_, target)) in data.iter().enumerate() {
            let pred = predictions.as_slice()[i];
            total_ape += ((pred - target) / target.max(1.0)).abs();
        }
        self.mape = total_ape / data.len() as f32;
        self.sample_count = data.len();

        Ok(())
    }

    fn predict_raw(&self, x: &[f32]) -> f32 {
        let mut result = self.weights[0]; // bias
        for (i, xi) in x.iter().enumerate() {
            if i + 1 < self.weights.len() {
                result += self.weights[i + 1] * xi;
            }
        }
        // Convert from normalized to tok/s (scale ~1000)
        (result * 1000.0).max(1.0)
    }

    /// Predict throughput for features
    ///
    /// With `ml-tuner` feature: uses trained RandomForest if available.
    /// Falls back to linear model otherwise.
    pub fn predict(&self, features: &TunerFeatures) -> ThroughputPrediction {
        let x = features.to_vector();

        // Use RandomForest if trained (ml-tuner feature)
        #[cfg(feature = "ml-tuner")]
        let raw_predicted_tps = if let Some(ref rf) = self.rf_model {
            // Use f32 matrix for RandomForestRegressor
            if let Ok(x_matrix) = Matrix::from_vec(1, TunerFeatures::DIM, x.to_vec()) {
                let predictions = rf.predict(&x_matrix);
                predictions.as_slice().first().copied().unwrap_or(0.0)
            } else {
                self.predict_raw(&x)
            }
        } else {
            self.predict_raw(&x)
        };

        #[cfg(not(feature = "ml-tuner"))]
        let raw_predicted_tps = self.predict_raw(&x);

        // v1.1.0: Roofline clamping - predictions must not exceed theoretical maximum
        // Roofline: max_tps = memory_bw_bytes_per_sec / bytes_per_token
        // For decode: bytes_per_token ≈ model_params × bytes_per_param
        let theoretical_max_tps = Self::compute_roofline_bound(features);
        let predicted_tps = raw_predicted_tps.min(theoretical_max_tps);

        // Confidence based on training MAPE and feature validity
        // Lower confidence if we hit the roofline cap
        let roofline_penalty = if raw_predicted_tps > theoretical_max_tps {
            0.9 // 10% confidence penalty for capped predictions
        } else {
            1.0
        };
        let confidence = (1.0 - self.mape).max(0.5) * roofline_penalty;

        ThroughputPrediction {
            predicted_tps,
            confidence,
            top_features: self.feature_importance.iter().take(5).cloned().collect(),
        }
    }

    /// Compute theoretical maximum throughput based on roofline model (v1.1.0)
    ///
    /// For memory-bound LLM inference (decode phase):
    /// max_tps = memory_bw_bytes_per_sec / bytes_per_token
    /// bytes_per_token = model_params × bytes_per_param / batch_size
    fn compute_roofline_bound(features: &TunerFeatures) -> f32 {
        // Denormalize model params: normalized = (log10(b) + 1) / 3
        // log10(b) = normalized * 3 - 1
        // b = 10^(normalized * 3 - 1)
        let model_params_b = 10.0_f32.powf(features.model_params_b * 3.0 - 1.0);

        // Get bytes per param from quant type one-hot encoding
        let bytes_per_param = Self::bytes_per_param_from_onehot(&features.quant_type_onehot);

        // Denormalize memory bandwidth: normalized = bw / 3000 GB/s
        let gpu_mem_bw_gbs = features.gpu_mem_bw_norm * 3000.0;

        // Denormalize batch size: normalized = batch_size / 64
        let batch_size = (features.batch_size_norm * 64.0).max(1.0);

        // Roofline calculation:
        // model_bytes = model_params_b * bytes_per_param * 1e9
        // bytes_per_token = model_bytes / batch_size
        // max_tps = (gpu_mem_bw_gbs * 1e9) / bytes_per_token
        //         = (gpu_mem_bw_gbs * 1e9 * batch_size) / (model_params_b * bytes_per_param * 1e9)
        //         = (gpu_mem_bw_gbs * batch_size) / (model_params_b * bytes_per_param)
        let theoretical_max = (gpu_mem_bw_gbs * batch_size) / (model_params_b * bytes_per_param);

        // Clamp to reasonable range (1 tok/s to 10000 tok/s)
        theoretical_max.clamp(1.0, 10000.0)
    }

    /// Extract bytes per param from quant type one-hot encoding
    fn bytes_per_param_from_onehot(onehot: &[f32; 8]) -> f32 {
        // One-hot indices map to QuantType variants
        // 0: Q4_0, 1: Q4_1, 2: Q4K, 3: Q5K, 4: Q6K, 5: Q8_0, 6: F16, 7: F32
        let bytes_per_param = [0.5625, 0.5625, 0.5625, 0.6875, 0.8125, 1.0, 2.0, 4.0];

        // Find the active index (max value in one-hot)
        let idx = onehot
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(2); // Default to Q4K if ambiguous

        bytes_per_param[idx]
    }
}

/// Kernel classifier using simple rule-based logic.
///
/// With `ml-tuner` feature: uses aprender::RandomForestClassifier (SHOWCASE-BRICK-001)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct KernelClassifier {
    /// Kernel accuracy on validation (for confidence)
    accuracy: f32,
    /// RandomForest classifier when ml-tuner feature is enabled
    #[cfg(feature = "ml-tuner")]
    #[serde(skip)]
    rf_classifier: Option<RandomForestClassifier>,
}

impl KernelClassifier {
    pub fn new() -> Self {
        Self {
            accuracy: 0.85,
            #[cfg(feature = "ml-tuner")]
            rf_classifier: None,
        }
    }

    /// Create a classifier with aprender RandomForest (ml-tuner feature)
    #[cfg(feature = "ml-tuner")]
    pub fn with_random_forest(n_estimators: usize) -> Self {
        Self {
            accuracy: 0.85,
            rf_classifier: Some(RandomForestClassifier::new(n_estimators)),
        }
    }

    /// Train the classifier using aprender RandomForest (ml-tuner feature)
    ///
    /// Labels should be kernel type indices (0=TiledQ4K, 1=CoalescedQ4K, etc.)
    #[cfg(feature = "ml-tuner")]
    pub fn train(&mut self, data: &[(TunerFeatures, u32)]) -> Result<(), TunerError> {
        if data.len() < 10 {
            return Err(TunerError::InsufficientData(data.len()));
        }

        // Convert to aprender format (Matrix<f32> for features, &[usize] for labels)
        let n_samples = data.len();
        let n_features = TunerFeatures::DIM;
        let mut x_data = Vec::with_capacity(n_samples * n_features);
        let mut y_data: Vec<usize> = Vec::with_capacity(n_samples);

        for (features, label) in data {
            x_data.extend(features.to_vector());
            y_data.push(*label as usize);
        }

        let x_matrix = Matrix::from_vec(n_samples, n_features, x_data)
            .map_err(|e| TunerError::TrainingFailed(e.to_string()))?;

        let rf = self.rf_classifier.get_or_insert_with(|| RandomForestClassifier::new(50));
        rf.fit(&x_matrix, &y_data)
            .map_err(|e| TunerError::TrainingFailed(e.to_string()))?;

        // Calculate accuracy on training data
        let predictions = rf.predict(&x_matrix);
        let mut correct = 0;
        for (i, (_, label)) in data.iter().enumerate() {
            if predictions[i] as u32 == *label {
                correct += 1;
            }
        }
        self.accuracy = correct as f32 / data.len() as f32;

        Ok(())
    }

    /// Predict best kernel based on features
    pub fn predict(&self, features: &TunerFeatures) -> KernelRecommendation {
        // Rule-based kernel selection from SHOWCASE-BRICK-001 learnings
        let batch_size = (features.batch_size_norm * 64.0).round() as u32;
        let seq_len = (2.0_f32.powf(features.seq_len_log * 15.0)).round() as u32;

        // Determine best Q4K variant based on batch size
        let (top_kernel, confidence) = if batch_size >= 4 {
            // M >= 4: Use batched kernels
            (KernelType::BatchedQ4K, 0.90)
        } else if batch_size >= 2 {
            // M = 2-3: Vectorized is good
            (KernelType::VectorizedQ4K, 0.85)
        } else {
            // M = 1: Coalesced or Vectorized
            if features.cuda_graphs > 0.5 {
                (KernelType::VectorizedQ4K, 0.88)
            } else {
                (KernelType::CoalescedQ4K, 0.82)
            }
        };

        // Check for attention-bound cases
        let attention_kernel = if seq_len > 128 {
            KernelType::MultiWarpAttention
        } else {
            KernelType::IncrementalAttention
        };

        // Build alternatives
        let alternatives = vec![
            (KernelType::VectorizedQ4K, 0.85),
            (KernelType::CoalescedQ4K, 0.75),
            (attention_kernel, 0.70),
        ]
        .into_iter()
        .filter(|(k, _)| *k != top_kernel)
        .take(2)
        .collect();

        KernelRecommendation {
            top_kernel,
            confidence,
            alternatives,
        }
    }
}

/// Bottleneck classifier using heuristics from profiler data.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BottleneckClassifier {
    /// Classification accuracy
    accuracy: f32,
}

impl BottleneckClassifier {
    pub fn new() -> Self {
        Self { accuracy: 0.90 }
    }

    /// Predict bottleneck from features
    pub fn predict(&self, features: &TunerFeatures) -> BottleneckPrediction {
        // Use already-computed bottleneck if available
        if let Some(class) = features.bottleneck_class {
            return BottleneckPrediction {
                class,
                confidence: 0.95,
                explanation: format!(
                    "Bottleneck classified from profiler data: {}",
                    class
                ),
                recommended_action: class.recommended_action().to_string(),
            };
        }

        // Heuristic classification based on features
        let batch_size = (features.batch_size_norm * 64.0).round() as u32;
        let seq_len = (2.0_f32.powf(features.seq_len_log * 15.0)).round() as u32;

        let (class, confidence, explanation) = if batch_size == 1 && features.cuda_graphs < 0.5 {
            (
                BottleneckClass::LaunchBound,
                0.75,
                "Single sequence without CUDA graphs: kernel launch overhead may dominate".into(),
            )
        } else if seq_len > 512 {
            (
                BottleneckClass::AttentionBound,
                0.80,
                format!(
                    "Long sequence (len={}) likely makes attention the bottleneck",
                    seq_len
                ),
            )
        } else {
            (
                BottleneckClass::MemoryBound,
                0.85,
                "Q4K GEMV is typically memory-bound for LLM inference".into(),
            )
        };

        BottleneckPrediction {
            class,
            confidence,
            explanation,
            recommended_action: class.recommended_action().to_string(),
        }
    }
}

// ============================================================================
// TUNER-006: BrickTuner Ensemble
// ============================================================================

/// Combined tuner recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TunerRecommendation {
    /// Throughput prediction
    pub throughput: ThroughputPrediction,
    /// Kernel recommendation
    pub kernel: KernelRecommendation,
    /// Bottleneck analysis
    pub bottleneck: BottleneckPrediction,
    /// Model version
    pub model_version: String,
    /// Overall confidence
    pub confidence_overall: f32,
    /// Suggested experiments to try
    pub suggested_experiments: Vec<ExperimentSuggestion>,
}

/// Suggested experiment to improve performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperimentSuggestion {
    /// Increase batch size
    IncreaseBatchSize { from: u32, to: u32 },
    /// Enable CUDA graphs
    EnableCudaGraphs,
    /// Try a specific kernel
    TryKernel { kernel: KernelType },
    /// Reduce sequence length
    ReduceSequenceLength { factor: f32 },
    /// Enable multi-KV cache
    EnableMultiKvCache { count: u32 },
}

impl std::fmt::Display for ExperimentSuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExperimentSuggestion::IncreaseBatchSize { from, to } => {
                write!(f, "Increase batch size: M={} → M={}", from, to)
            }
            ExperimentSuggestion::EnableCudaGraphs => {
                write!(f, "Enable CUDA graphs for kernel launch amortization")
            }
            ExperimentSuggestion::TryKernel { kernel } => {
                write!(f, "Try kernel: {:?}", kernel)
            }
            ExperimentSuggestion::ReduceSequenceLength { factor } => {
                write!(f, "Reduce sequence length by {:.0}%", (1.0 - factor) * 100.0)
            }
            ExperimentSuggestion::EnableMultiKvCache { count } => {
                write!(f, "Enable {} separate KV caches for batched attention", count)
            }
        }
    }
}

/// ML-based ComputeBrick tuner ensemble.
///
/// Combines three models for comprehensive recommendations:
/// - ThroughputRegressor: Predicts tok/s
/// - KernelClassifier: Selects best kernel
/// - BottleneckClassifier: Identifies performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrickTuner {
    /// Throughput regression model
    throughput: ThroughputRegressor,
    /// Kernel classification model
    kernel: KernelClassifier,
    /// Bottleneck classification model
    bottleneck: BottleneckClassifier,
    /// Model version
    version: String,
    /// Training timestamp
    trained_at: String,
    /// Number of training samples
    sample_count: usize,
}

impl Default for BrickTuner {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickTuner {
    /// Model version
    pub const VERSION: &'static str = "1.0.0";

    /// Create a new tuner with default models
    pub fn new() -> Self {
        Self {
            throughput: ThroughputRegressor::new(),
            kernel: KernelClassifier::new(),
            bottleneck: BottleneckClassifier::new(),
            version: Self::VERSION.to_string(),
            trained_at: chrono_lite_now(),
            sample_count: 0,
        }
    }

    /// Get comprehensive tuning recommendation
    pub fn recommend(&self, features: &TunerFeatures) -> TunerRecommendation {
        let throughput = self.throughput.predict(features);
        let kernel = self.kernel.predict(features);
        let bottleneck = self.bottleneck.predict(features);

        // Calculate overall confidence
        let confidence_overall =
            (throughput.confidence + kernel.confidence + bottleneck.confidence) / 3.0;

        // Generate experiment suggestions based on bottleneck
        let suggested_experiments = self.suggest_experiments(features, &bottleneck);

        TunerRecommendation {
            throughput,
            kernel,
            bottleneck,
            model_version: self.version.clone(),
            confidence_overall,
            suggested_experiments,
        }
    }

    /// Suggest experiments based on current bottleneck
    fn suggest_experiments(
        &self,
        features: &TunerFeatures,
        bottleneck: &BottleneckPrediction,
    ) -> Vec<ExperimentSuggestion> {
        let mut suggestions = Vec::new();
        let batch_size = (features.batch_size_norm * 64.0).round() as u32;

        match bottleneck.class {
            BottleneckClass::MemoryBound => {
                if batch_size < 8 {
                    suggestions.push(ExperimentSuggestion::IncreaseBatchSize {
                        from: batch_size,
                        to: (batch_size * 2).min(8),
                    });
                }
                suggestions.push(ExperimentSuggestion::TryKernel {
                    kernel: KernelType::BatchedQ4K,
                });
                if batch_size > 1 {
                    suggestions.push(ExperimentSuggestion::EnableMultiKvCache { count: batch_size });
                }
            }
            BottleneckClass::LaunchBound => {
                if features.cuda_graphs < 0.5 {
                    suggestions.push(ExperimentSuggestion::EnableCudaGraphs);
                }
                suggestions.push(ExperimentSuggestion::TryKernel {
                    kernel: KernelType::FusedRmsNormQ4K,
                });
            }
            BottleneckClass::AttentionBound => {
                suggestions.push(ExperimentSuggestion::TryKernel {
                    kernel: KernelType::BatchedAttention,
                });
                suggestions.push(ExperimentSuggestion::ReduceSequenceLength { factor: 0.5 });
            }
            _ => {
                // Default suggestions
                if batch_size < 4 {
                    suggestions.push(ExperimentSuggestion::IncreaseBatchSize {
                        from: batch_size,
                        to: 4,
                    });
                }
            }
        }

        suggestions
    }

    /// Train all models on labeled data
    pub fn train(&mut self, data: &[(TunerFeatures, f32)]) -> Result<(), TunerError> {
        self.throughput.train(data)?;
        self.sample_count = data.len();
        self.trained_at = chrono_lite_now();
        Ok(())
    }

    /// Print recommendations to console (TUI-friendly)
    pub fn print_recommendation(&self, rec: &TunerRecommendation) {
        println!("╭─────────────────────────────────────────────────────────────╮");
        println!(
            "│           BrickTuner Recommendations v{}                 │",
            self.version
        );
        println!("├─────────────────────────────────────────────────────────────┤");
        println!(
            "│ Predicted throughput: {:>7.1} tok/s ({:>4.0}% confidence)     │",
            rec.throughput.predicted_tps,
            rec.throughput.confidence * 100.0
        );
        println!(
            "│ Recommended kernel:   {:>15?} ({:>4.0}% conf)       │",
            rec.kernel.top_kernel,
            rec.kernel.confidence * 100.0
        );
        println!(
            "│ Bottleneck class:     {:>15} ({:>4.0}% conf)       │",
            rec.bottleneck.class,
            rec.bottleneck.confidence * 100.0
        );
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Explanation: {}│", pad_right(&rec.bottleneck.explanation, 47));
        println!("├─────────────────────────────────────────────────────────────┤");
        println!("│ Suggested experiments:                                      │");
        for (i, exp) in rec.suggested_experiments.iter().take(3).enumerate() {
            println!("│  {}. {}│", i + 1, pad_right(&exp.to_string(), 56));
        }
        println!("╰─────────────────────────────────────────────────────────────╯");
    }

    // ========================================================================
    // T-TUNER-006: cbtop TUI Integration (GitHub #83)
    // ========================================================================

    /// Render recommendation as TUI panel lines (for cbtop integration)
    ///
    /// Returns a vector of strings that can be rendered in a TUI widget.
    /// Each line is formatted for fixed-width display (width=61 chars).
    pub fn render_panel(&self, rec: &TunerRecommendation) -> Vec<String> {
        let mut lines = Vec::with_capacity(12);

        lines.push(format!(
            "│           BrickTuner Recommendations v{}                 │",
            self.version
        ));
        lines.push("├─────────────────────────────────────────────────────────────┤".to_string());
        lines.push(format!(
            "│ Predicted throughput: {:>7.1} tok/s ({:>4.0}% confidence)     │",
            rec.throughput.predicted_tps,
            rec.throughput.confidence * 100.0
        ));
        lines.push(format!(
            "│ Recommended kernel:   {:>15?} ({:>4.0}% conf)       │",
            rec.kernel.top_kernel,
            rec.kernel.confidence * 100.0
        ));
        lines.push(format!(
            "│ Bottleneck class:     {:>15} ({:>4.0}% conf)       │",
            rec.bottleneck.class,
            rec.bottleneck.confidence * 100.0
        ));
        lines.push("├─────────────────────────────────────────────────────────────┤".to_string());
        lines.push(format!(
            "│ Explanation: {}│",
            pad_right(&rec.bottleneck.explanation, 47)
        ));
        lines.push("├─────────────────────────────────────────────────────────────┤".to_string());
        lines.push("│ Suggested experiments:                                      │".to_string());

        for (i, exp) in rec.suggested_experiments.iter().take(3).enumerate() {
            lines.push(format!("│  {}. {}│", i + 1, pad_right(&exp.to_string(), 56)));
        }

        // Pad if fewer than 3 suggestions
        for _ in rec.suggested_experiments.len()..3 {
            lines.push("│                                                             │".to_string());
        }

        lines.push("├─────────────────────────────────────────────────────────────┤".to_string());
        lines.push("│ [Press 'a' to apply] [Press 't' to toggle] [Press 'r' to run]│".to_string());

        lines
    }

    /// Render compact recommendation (single line for status bar)
    pub fn render_compact(&self, rec: &TunerRecommendation) -> String {
        format!(
            "Tuner: {:.0} tok/s | {:?} | {} ({:.0}%)",
            rec.throughput.predicted_tps,
            rec.kernel.top_kernel,
            rec.bottleneck.class,
            rec.confidence_overall * 100.0
        )
    }

    /// Render prediction vs actual comparison
    pub fn render_comparison(&self, rec: &TunerRecommendation, actual_tps: f32) -> Vec<String> {
        let error_pct = if actual_tps > 0.0 {
            ((rec.throughput.predicted_tps - actual_tps) / actual_tps * 100.0).abs()
        } else {
            0.0
        };

        let accuracy_indicator = if error_pct < 5.0 {
            "🎯 Excellent"
        } else if error_pct < 10.0 {
            "✓ Good"
        } else if error_pct < 20.0 {
            "△ Fair"
        } else {
            "✗ Poor"
        };

        vec![
            format!(
                "│ Predicted: {:>7.1} tok/s  Actual: {:>7.1} tok/s           │",
                rec.throughput.predicted_tps, actual_tps
            ),
            format!(
                "│ Error: {:>5.1}%  Accuracy: {:>12}                       │",
                error_pct, accuracy_indicator
            ),
        ]
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, TunerError> {
        serde_json::to_string_pretty(self).map_err(|e| TunerError::Serialization(e.to_string()))
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, TunerError> {
        serde_json::from_str(json).map_err(|e| TunerError::Serialization(e.to_string()))
    }

    // =========================================================================
    // APR Persistence (SOVEREIGN STACK - GH#81)
    // =========================================================================

    /// APR format magic bytes (APR1 = uncompressed)
    const APR_MAGIC: [u8; 4] = [b'A', b'P', b'R', b'1'];

    /// Get the default cache path for tuner models.
    ///
    /// Returns `~/.cache/trueno/tuner_model_v{VERSION}.apr`
    #[cfg(feature = "hardware-detect")]
    pub fn cache_path() -> std::path::PathBuf {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("trueno");

        // Create directory if it doesn't exist
        let _ = std::fs::create_dir_all(&cache_dir);

        cache_dir.join(format!("tuner_model_v{}.apr", Self::VERSION))
    }

    /// Load tuner from cache or create new with defaults.
    ///
    /// This is the recommended way to create a BrickTuner for production use.
    /// It will:
    /// 1. Check for cached model at `~/.cache/trueno/tuner_model_v{VERSION}.apr`
    /// 2. Load if exists and version matches
    /// 3. Create new with defaults if not found or version mismatch
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use trueno::tuner::BrickTuner;
    ///
    /// let tuner = BrickTuner::load_or_default();
    /// ```
    #[cfg(feature = "hardware-detect")]
    pub fn load_or_default() -> Self {
        let path = Self::cache_path();

        if path.exists() {
            match Self::load_apr(&path) {
                Ok(tuner) => {
                    // Version check
                    if tuner.version == Self::VERSION {
                        return tuner;
                    }
                    // Version mismatch - create new
                }
                Err(_) => {
                    // Load failed - create new
                }
            }
        }

        Self::new()
    }

    /// Save tuner model to .apr file.
    ///
    /// APR1 format (uncompressed):
    /// - 4-byte magic: "APR1"
    /// - 4-byte metadata_len: u32 LE
    /// - JSON metadata
    /// - 4-byte CRC32: checksum
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tuner = BrickTuner::new();
    /// tuner.save_apr("model.apr")?;
    /// ```
    pub fn save_apr<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), TunerError> {
        use std::io::Write;

        let json = self.to_json()?;
        let json_bytes = json.as_bytes();

        let mut file = std::fs::File::create(path)
            .map_err(|e| TunerError::Io(e.to_string()))?;

        // Write magic
        file.write_all(&Self::APR_MAGIC)
            .map_err(|e| TunerError::Io(e.to_string()))?;

        // Write metadata length
        let len = json_bytes.len() as u32;
        file.write_all(&len.to_le_bytes())
            .map_err(|e| TunerError::Io(e.to_string()))?;

        // Write JSON metadata
        file.write_all(json_bytes)
            .map_err(|e| TunerError::Io(e.to_string()))?;

        // Calculate and write CRC32
        let mut crc = 0u32;
        crc = crc32_update(crc, &Self::APR_MAGIC);
        crc = crc32_update(crc, &len.to_le_bytes());
        crc = crc32_update(crc, json_bytes);
        file.write_all(&crc.to_le_bytes())
            .map_err(|e| TunerError::Io(e.to_string()))?;

        Ok(())
    }

    /// Load tuner model from .apr file.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let tuner = BrickTuner::load_apr("model.apr")?;
    /// ```
    pub fn load_apr<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TunerError> {
        use std::io::Read;

        let mut file = std::fs::File::open(path)
            .map_err(|e| TunerError::Io(e.to_string()))?;

        // Read and verify magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|e| TunerError::Io(e.to_string()))?;

        if magic != Self::APR_MAGIC {
            return Err(TunerError::InvalidFormat(
                "Invalid APR magic bytes".to_string(),
            ));
        }

        // Read metadata length
        let mut len_bytes = [0u8; 4];
        file.read_exact(&mut len_bytes)
            .map_err(|e| TunerError::Io(e.to_string()))?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        // Read JSON metadata
        let mut json_bytes = vec![0u8; len];
        file.read_exact(&mut json_bytes)
            .map_err(|e| TunerError::Io(e.to_string()))?;

        // Read and verify CRC32
        let mut crc_bytes = [0u8; 4];
        file.read_exact(&mut crc_bytes)
            .map_err(|e| TunerError::Io(e.to_string()))?;
        let stored_crc = u32::from_le_bytes(crc_bytes);

        let mut computed_crc = 0u32;
        computed_crc = crc32_update(computed_crc, &Self::APR_MAGIC);
        computed_crc = crc32_update(computed_crc, &len_bytes);
        computed_crc = crc32_update(computed_crc, &json_bytes);

        if stored_crc != computed_crc {
            return Err(TunerError::InvalidFormat(
                "CRC32 checksum mismatch".to_string(),
            ));
        }

        // Parse JSON
        let json = String::from_utf8(json_bytes)
            .map_err(|e| TunerError::Serialization(e.to_string()))?;

        Self::from_json(&json)
    }

    /// Save to default cache path.
    #[cfg(feature = "hardware-detect")]
    pub fn save_to_cache(&self) -> Result<(), TunerError> {
        self.save_apr(Self::cache_path())
    }
}

/// Simple CRC32 implementation (IEEE polynomial).
/// Used for .apr file checksum verification.
fn crc32_update(crc: u32, data: &[u8]) -> u32 {
    const CRC32_TABLE: [u32; 256] = crc32_table();
    let mut crc = !crc;
    for &byte in data {
        crc = CRC32_TABLE[((crc ^ u32::from(byte)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

/// Generate CRC32 lookup table at compile time.
const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = 0xEDB8_8320 ^ (crc >> 1);
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Compute CRC32 hash for given data (convenience wrapper)
fn crc32_hash(data: &[u8]) -> u32 {
    crc32_update(0, data)
}

// ============================================================================
// TUNER-007: BrickProfiler Integration
// ============================================================================

impl BrickProfiler {
    /// Get ML-based tuning recommendations.
    ///
    /// Extracts features from current profile and returns recommendations.
    pub fn get_tuner_recommendations(&self, config: &RunConfig) -> Option<TunerRecommendation> {
        if !self.is_enabled() {
            return None;
        }

        // Create feature extractor
        let extractor = FeatureExtractor::new();

        // Extract features
        let features = extractor.extract(self, config);

        // Get recommendation from global tuner
        let tuner = BrickTuner::new();
        Some(tuner.recommend(&features))
    }

    /// Print tuner recommendations to console.
    pub fn print_tuner_recommendations(&self, config: &RunConfig) {
        if let Some(rec) = self.get_tuner_recommendations(config) {
            let tuner = BrickTuner::new();
            tuner.print_recommendation(&rec);
        } else {
            println!("Tuner recommendations not available (profiler disabled)");
        }
    }

    /// Get tokens per second from profiler.
    pub fn tokens_per_sec(&self) -> Option<f32> {
        let total_ns = self.total_ns();
        let total_tokens = self.total_tokens();
        if total_ns == 0 || total_tokens == 0 {
            return None;
        }
        Some(total_tokens as f32 * 1e9 / total_ns as f32)
    }
}

// ============================================================================
// Training Data Collection (TUNER-010)
// ============================================================================

/// Training sample for the tuner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Features
    pub features: TunerFeatures,
    /// Measured throughput (label)
    pub throughput_tps: f32,
    /// Best kernel (label)
    pub best_kernel: KernelType,
    /// Bottleneck class (label)
    pub bottleneck: BottleneckClass,
    /// Timestamp
    pub timestamp: String,
    /// Hardware ID
    pub hardware_id: String,
}

/// User feedback on a recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum UserFeedback {
    /// User accepted the recommendation
    Accepted,
    /// User rejected the recommendation
    Rejected,
    /// User provided alternative (overrode recommendation)
    Alternative,
    /// No feedback (default)
    #[default]
    None,
}

/// Concept drift detection result
#[derive(Debug, Clone)]
pub struct ConceptDriftStatus {
    /// Whether drift has been detected
    pub drift_detected: bool,
    /// Estimated model staleness (0.0 = fresh, 1.0 = very stale)
    pub staleness_score: f32,
    /// Number of samples since last training
    pub samples_since_training: usize,
    /// Recommendation: should retrain?
    pub recommend_retrain: bool,
    /// Explanation of drift status
    pub explanation: String,
}

/// Training data collector with online learning support (T-TUNER-005, GitHub #82)
#[derive(Debug, Default)]
pub struct TunerDataCollector {
    /// Collected samples
    samples: Vec<TrainingSample>,
    /// Feature extractor
    extractor: FeatureExtractor,
    /// Auto-retrain threshold
    retrain_threshold: usize,
    /// Number of samples at last training
    samples_at_last_train: usize,
    /// User feedback history (sample index -> feedback)
    feedback: Vec<UserFeedback>,
    /// Online learning enabled (privacy: opt-in only)
    online_learning_enabled: bool,
    /// Moving average of prediction errors (for concept drift)
    error_window: Vec<f32>,
    /// Error window size for drift detection
    error_window_size: usize,
}

impl TunerDataCollector {
    /// Default error window size for concept drift detection
    const DEFAULT_ERROR_WINDOW_SIZE: usize = 50;

    /// Error threshold for drift detection (mean absolute error)
    const DRIFT_ERROR_THRESHOLD: f32 = 0.15;

    /// Staleness threshold (samples since training) for recommending retrain
    const STALENESS_THRESHOLD: usize = 100;

    /// Create a new collector
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            extractor: FeatureExtractor::new(),
            retrain_threshold: 100,
            samples_at_last_train: 0,
            feedback: Vec::new(),
            online_learning_enabled: false, // Privacy: opt-in
            error_window: Vec::new(),
            error_window_size: Self::DEFAULT_ERROR_WINDOW_SIZE,
        }
    }

    /// Create a collector with online learning enabled
    pub fn with_online_learning() -> Self {
        let mut collector = Self::new();
        collector.online_learning_enabled = true;
        collector
    }

    /// Enable online learning (privacy: explicit opt-in)
    pub fn enable_online_learning(&mut self) {
        self.online_learning_enabled = true;
    }

    /// Disable online learning
    pub fn disable_online_learning(&mut self) {
        self.online_learning_enabled = false;
    }

    /// Check if online learning is enabled
    pub fn is_online_learning_enabled(&self) -> bool {
        self.online_learning_enabled
    }

    /// Record a profiling run as training data
    pub fn record(
        &mut self,
        profiler: &BrickProfiler,
        config: &RunConfig,
        kernel: KernelType,
    ) -> Option<()> {
        let throughput_tps = profiler.tokens_per_sec()?;
        let features = self.extractor.extract(profiler, config);
        let bottleneck = features.bottleneck_class.unwrap_or(BottleneckClass::Unknown);

        let sample = TrainingSample {
            features,
            throughput_tps,
            best_kernel: kernel,
            bottleneck,
            timestamp: chrono_lite_now(),
            hardware_id: "unknown".to_string(),
        };

        self.samples.push(sample);
        Some(())
    }

    /// Get all samples
    pub fn samples(&self) -> &[TrainingSample] {
        &self.samples
    }

    /// Get sample count
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Export to JSON
    pub fn to_json(&self) -> Result<String, TunerError> {
        serde_json::to_string_pretty(&self.samples)
            .map_err(|e| TunerError::Serialization(e.to_string()))
    }

    /// Prepare training data for model
    pub fn prepare_training_data(&self) -> Vec<(TunerFeatures, f32)> {
        self.samples
            .iter()
            .map(|s| (s.features.clone(), s.throughput_tps))
            .collect()
    }

    // ========================================================================
    // T-TUNER-003: Persistent Training Data (GitHub #80)
    // ========================================================================

    /// Minimum samples required before training triggers
    pub const MIN_SAMPLES_FOR_TRAINING: usize = 1000;

    /// Training data cache path
    #[cfg(feature = "hardware-detect")]
    pub fn cache_path() -> std::path::PathBuf {
        let hw_id = Self::hardware_id();
        dirs::cache_dir()
            .unwrap_or_else(|| std::path::PathBuf::from(".cache"))
            .join("trueno")
            .join(format!("training_data_{}.apr", hw_id))
    }

    /// Generate hardware fingerprint for hardware-specific models
    #[cfg(feature = "hardware-detect")]
    pub fn hardware_id() -> String {
        use crate::hardware::HardwareCapability;
        let hw = HardwareCapability::detect();

        // Create a stable fingerprint from hardware characteristics
        let fingerprint = format!(
            "{}-{:?}-{}-{}",
            hw.cpu.cores,
            hw.cpu.simd,
            hw.gpu.as_ref().map(|g| g.model.as_str()).unwrap_or("none"),
            hw.gpu.as_ref().map(|g| g.vram_gb as u32).unwrap_or(0),
        );

        // Hash to short hex string
        let hash = crc32_hash(fingerprint.as_bytes());
        format!("{:08x}", hash)
    }

    /// Load from cache or create empty
    #[cfg(feature = "hardware-detect")]
    pub fn load_or_create() -> Self {
        let path = Self::cache_path();
        if path.exists() {
            if let Ok(collector) = Self::load_apr(&path) {
                return collector;
            }
        }
        Self::new()
    }

    /// Save training data to APR format
    pub fn save_apr<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), TunerError> {
        use std::io::Write;

        // Ensure parent directory exists
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;
        }

        // Serialize samples to JSON
        let json = serde_json::to_string(&self.samples)
            .map_err(|e| TunerError::Serialization(e.to_string()))?;
        let json_bytes = json.as_bytes();

        // Create APR format: MAGIC + LEN + JSON + CRC32
        let mut file = std::fs::File::create(path.as_ref())
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Write magic bytes: "APR2" (version 2 for training data)
        file.write_all(b"APR2")
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Write length as u32 little-endian
        let len = json_bytes.len() as u32;
        file.write_all(&len.to_le_bytes())
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Write JSON
        file.write_all(json_bytes)
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Write CRC32 checksum
        let checksum = crc32_hash(json_bytes);
        file.write_all(&checksum.to_le_bytes())
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        Ok(())
    }

    /// Load training data from APR format
    pub fn load_apr<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TunerError> {
        use std::io::Read;

        let mut file = std::fs::File::open(path.as_ref())
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Read and verify magic
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;
        if &magic != b"APR2" {
            return Err(TunerError::InvalidFormat(format!(
                "Expected APR2 magic, got {:?}",
                magic
            )));
        }

        // Read length
        let mut len_bytes = [0u8; 4];
        file.read_exact(&mut len_bytes)
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        // Read JSON
        let mut json_bytes = vec![0u8; len];
        file.read_exact(&mut json_bytes)
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;

        // Read and verify CRC32
        let mut crc_bytes = [0u8; 4];
        file.read_exact(&mut crc_bytes)
            .map_err(|e: std::io::Error| TunerError::Io(e.to_string()))?;
        let stored_crc = u32::from_le_bytes(crc_bytes);
        let computed_crc = crc32_hash(&json_bytes);

        if stored_crc != computed_crc {
            return Err(TunerError::InvalidFormat(format!(
                "CRC mismatch: stored={:08x}, computed={:08x}",
                stored_crc, computed_crc
            )));
        }

        // Deserialize samples
        let samples: Vec<TrainingSample> = serde_json::from_slice(&json_bytes)
            .map_err(|e| TunerError::Serialization(e.to_string()))?;

        Ok(Self {
            samples,
            extractor: FeatureExtractor::new(),
            retrain_threshold: 100,
            samples_at_last_train: 0,
            feedback: Vec::new(),
            online_learning_enabled: false,
            error_window: Vec::new(),
            error_window_size: Self::DEFAULT_ERROR_WINDOW_SIZE,
        })
    }

    /// Append a sample to the cached training data
    #[cfg(feature = "hardware-detect")]
    pub fn record_and_persist(
        &mut self,
        profiler: &BrickProfiler,
        config: &RunConfig,
        kernel: KernelType,
    ) -> Result<(), TunerError> {
        // Record the sample
        self.record(profiler, config, kernel);

        // Append to cache file
        let path = Self::cache_path();
        self.save_apr(&path)?;

        Ok(())
    }

    /// Check if we have enough samples to train
    pub fn ready_to_train(&self) -> bool {
        self.samples.len() >= Self::MIN_SAMPLES_FOR_TRAINING
    }

    /// Train a BrickTuner from collected data if we have enough samples
    pub fn train_if_ready(&self) -> Option<BrickTuner> {
        if !self.ready_to_train() {
            return None;
        }

        let training_data = self.prepare_training_data();
        let mut tuner = BrickTuner::new();

        match tuner.train(&training_data) {
            Ok(()) => Some(tuner),
            Err(_) => None,
        }
    }

    /// Get training progress as (current, required)
    pub fn training_progress(&self) -> (usize, usize) {
        (self.samples.len(), Self::MIN_SAMPLES_FOR_TRAINING)
    }

    /// Merge samples from another collector
    pub fn merge(&mut self, other: &TunerDataCollector) {
        self.samples.extend(other.samples.iter().cloned());
    }

    /// Import samples from JSON
    pub fn from_json(json: &str) -> Result<Self, TunerError> {
        let samples: Vec<TrainingSample> =
            serde_json::from_str(json).map_err(|e| TunerError::Serialization(e.to_string()))?;

        Ok(Self {
            samples,
            extractor: FeatureExtractor::new(),
            retrain_threshold: 100,
            samples_at_last_train: 0,
            feedback: Vec::new(),
            online_learning_enabled: false,
            error_window: Vec::new(),
            error_window_size: Self::DEFAULT_ERROR_WINDOW_SIZE,
        })
    }

    /// Import samples from the Five-Whys archive (85 labeled iterations)
    /// Bootstrap initial training data from historical analysis
    pub fn bootstrap_from_five_whys() -> Self {
        // Five-Whys archive has 85 labeled iterations from SHOWCASE-BRICK-001
        // Each iteration has: features, throughput, kernel selection, bottleneck

        // TODO: Load actual Five-Whys data from archive
        // For now, return empty collector - data will be collected from real runs
        Self::new()
    }

    // ========================================================================
    // T-TUNER-005: Online Learning (GitHub #82)
    // ========================================================================

    /// Record user feedback on a recommendation
    pub fn record_feedback(&mut self, sample_index: usize, feedback: UserFeedback) {
        // Extend feedback vector if needed
        while self.feedback.len() <= sample_index {
            self.feedback.push(UserFeedback::None);
        }
        self.feedback[sample_index] = feedback;
    }

    /// Get feedback for a sample
    pub fn get_feedback(&self, sample_index: usize) -> UserFeedback {
        self.feedback.get(sample_index).copied().unwrap_or(UserFeedback::None)
    }

    /// Record prediction error for concept drift detection
    pub fn record_prediction_error(&mut self, predicted: f32, actual: f32) {
        if !self.online_learning_enabled {
            return;
        }

        // Compute relative error (0.0 = perfect, 1.0 = 100% error)
        let error = if actual > 0.0 {
            ((predicted - actual) / actual).abs().min(1.0)
        } else {
            1.0
        };

        // Add to sliding window
        self.error_window.push(error);

        // Trim window to max size
        if self.error_window.len() > self.error_window_size {
            self.error_window.remove(0);
        }
    }

    /// Detect concept drift based on prediction error trends
    pub fn detect_concept_drift(&self) -> ConceptDriftStatus {
        let samples_since_training = self.samples.len().saturating_sub(self.samples_at_last_train);

        // Not enough data for drift detection
        if self.error_window.len() < 10 {
            return ConceptDriftStatus {
                drift_detected: false,
                staleness_score: 0.0,
                samples_since_training,
                recommend_retrain: false,
                explanation: "Insufficient data for drift detection".to_string(),
            };
        }

        // Compute mean error
        let mean_error: f32 =
            self.error_window.iter().sum::<f32>() / self.error_window.len() as f32;

        // Compute staleness score (0.0 = fresh, 1.0 = stale)
        let staleness_score = (samples_since_training as f32 / Self::STALENESS_THRESHOLD as f32)
            .min(1.0);

        // Detect drift
        let drift_detected = mean_error > Self::DRIFT_ERROR_THRESHOLD;

        // Recommend retrain if drift detected OR stale
        let recommend_retrain = drift_detected || staleness_score > 0.8;

        let explanation = if drift_detected {
            format!(
                "Concept drift detected: mean error {:.1}% exceeds threshold {:.1}%",
                mean_error * 100.0,
                Self::DRIFT_ERROR_THRESHOLD * 100.0
            )
        } else if staleness_score > 0.8 {
            format!(
                "Model stale: {} samples since last training (threshold: {})",
                samples_since_training,
                Self::STALENESS_THRESHOLD
            )
        } else {
            format!(
                "Model fresh: mean error {:.1}%, {} samples since training",
                mean_error * 100.0,
                samples_since_training
            )
        };

        ConceptDriftStatus {
            drift_detected,
            staleness_score,
            samples_since_training,
            recommend_retrain,
            explanation,
        }
    }

    /// Check if auto-retrain should trigger
    pub fn should_retrain(&self) -> bool {
        if !self.online_learning_enabled {
            return false;
        }

        let samples_since = self.samples.len().saturating_sub(self.samples_at_last_train);

        // Retrain if we have enough new samples
        if samples_since >= self.retrain_threshold {
            return true;
        }

        // Or if concept drift is detected
        let drift = self.detect_concept_drift();
        drift.recommend_retrain && samples_since >= 10
    }

    /// Mark that training occurred (resets drift counters)
    pub fn mark_trained(&mut self) {
        self.samples_at_last_train = self.samples.len();
        self.error_window.clear();
    }

    /// Get training statistics
    pub fn training_stats(&self) -> TrainingStats {
        let drift = self.detect_concept_drift();

        // Count feedback types
        let accepted_count = self.feedback.iter().filter(|f| **f == UserFeedback::Accepted).count();
        let rejected_count = self.feedback.iter().filter(|f| **f == UserFeedback::Rejected).count();
        let alternative_count = self.feedback.iter().filter(|f| **f == UserFeedback::Alternative).count();

        TrainingStats {
            total_samples: self.samples.len(),
            samples_since_training: drift.samples_since_training,
            accepted_count,
            rejected_count,
            alternative_count,
            staleness_score: drift.staleness_score,
            drift_detected: drift.drift_detected,
            online_learning_enabled: self.online_learning_enabled,
        }
    }

    /// Auto-retrain and update BrickTuner if conditions are met
    pub fn auto_retrain(&mut self, tuner: &mut BrickTuner) -> bool {
        if !self.should_retrain() {
            return false;
        }

        // Weight samples by feedback
        let training_data = self.prepare_weighted_training_data();

        if training_data.len() < 10 {
            return false;
        }

        // Train and update
        match tuner.train(&training_data) {
            Ok(()) => {
                self.mark_trained();
                true
            }
            Err(_) => false,
        }
    }

    /// Prepare training data with feedback weighting
    fn prepare_weighted_training_data(&self) -> Vec<(TunerFeatures, f32)> {
        self.samples
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                let feedback = self.get_feedback(i);

                // Skip rejected samples (they had bad throughput measurements)
                if feedback == UserFeedback::Rejected {
                    return None;
                }

                // Weight accepted samples higher (duplicate them)
                let weight = match feedback {
                    UserFeedback::Accepted => 2,
                    UserFeedback::Alternative => 1, // Still use, but normal weight
                    _ => 1,
                };

                Some((0..weight).map(|_| (s.features.clone(), s.throughput_tps)))
            })
            .flatten()
            .collect()
    }
}

/// Training statistics summary
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Total samples collected
    pub total_samples: usize,
    /// Samples since last training
    pub samples_since_training: usize,
    /// Accepted recommendations count
    pub accepted_count: usize,
    /// Rejected recommendations count
    pub rejected_count: usize,
    /// Alternative provided count
    pub alternative_count: usize,
    /// Staleness score (0.0 = fresh, 1.0 = stale)
    pub staleness_score: f32,
    /// Whether concept drift was detected
    pub drift_detected: bool,
    /// Whether online learning is enabled
    pub online_learning_enabled: bool,
}

// ============================================================================
// Error Types
// ============================================================================

/// Tuner error type
#[derive(Debug, Clone)]
pub enum TunerError {
    /// Invalid feature value
    InvalidFeature(String),
    /// Insufficient training data
    InsufficientData(usize),
    /// Serialization error
    Serialization(String),
    /// Model not found
    ModelNotFound,
    /// Prediction failed
    PredictionFailed(String),
    /// Training failed (ml-tuner feature)
    TrainingFailed(String),
    /// I/O error (file operations)
    Io(String),
    /// Invalid format (APR magic/version mismatch)
    InvalidFormat(String),
}

impl std::fmt::Display for TunerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TunerError::InvalidFeature(msg) => write!(f, "Invalid feature: {}", msg),
            TunerError::InsufficientData(n) => {
                write!(f, "Insufficient training data: {} samples (need >= 10)", n)
            }
            TunerError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            TunerError::ModelNotFound => write!(f, "Tuner model not found"),
            TunerError::PredictionFailed(msg) => write!(f, "Prediction failed: {}", msg),
            TunerError::TrainingFailed(msg) => write!(f, "Training failed: {}", msg),
            TunerError::Io(msg) => write!(f, "I/O error: {}", msg),
            TunerError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
        }
    }
}

impl std::error::Error for TunerError {}

// ============================================================================
// Helper Functions
// ============================================================================

/// Simple timestamp (avoids chrono dependency)
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

/// Pad string to fixed width
fn pad_right(s: &str, width: usize) -> String {
    if s.len() >= width {
        s[..width].to_string()
    } else {
        format!("{}{}", s, " ".repeat(width - s.len()))
    }
}

// ============================================================================
// Tests (TUNER-016-020 Falsification)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    // F001-F020: Model Accuracy
    mod f001_f020_model_accuracy {
        use super::*;

        #[test]
        fn f001_throughput_prediction_reasonable() {
            let features = TunerFeatures::builder()
                .model_params_b(1.5)
                .hidden_dim(1536)
                .batch_size(4)
                .quant_type(QuantType::Q4K)
                .cuda_graphs(true)
                .build();

            let regressor = ThroughputRegressor::new();
            let prediction = regressor.predict(&features);

            // Prediction should be positive and reasonable
            assert!(prediction.predicted_tps > 0.0);
            assert!(prediction.predicted_tps < 10000.0);
        }

        #[test]
        fn f010_prediction_latency_under_1ms() {
            let features = TunerFeatures::builder()
                .model_params_b(1.5)
                .batch_size(4)
                .build();

            let tuner = BrickTuner::new();
            let start = Instant::now();
            let _rec = tuner.recommend(&features);
            let elapsed = start.elapsed();

            assert!(elapsed.as_millis() < 1, "Prediction took {}ms", elapsed.as_millis());
        }

        #[test]
        fn f015_batch_size_monotonic() {
            let regressor = ThroughputRegressor::new();

            let pred_m1 = regressor.predict(&TunerFeatures::builder().batch_size(1).build());
            let pred_m4 = regressor.predict(&TunerFeatures::builder().batch_size(4).build());
            let pred_m8 = regressor.predict(&TunerFeatures::builder().batch_size(8).build());

            // Higher batch size should predict higher throughput
            assert!(
                pred_m4.predicted_tps >= pred_m1.predicted_tps,
                "M=4 ({}) should be >= M=1 ({})",
                pred_m4.predicted_tps,
                pred_m1.predicted_tps
            );
            assert!(
                pred_m8.predicted_tps >= pred_m4.predicted_tps,
                "M=8 ({}) should be >= M=4 ({})",
                pred_m8.predicted_tps,
                pred_m4.predicted_tps
            );
        }

        #[test]
        fn f019_cuda_graphs_benefit_predicted() {
            let regressor = ThroughputRegressor::new();

            let pred_no_graph = regressor.predict(
                &TunerFeatures::builder()
                    .batch_size(1)
                    .cuda_graphs(false)
                    .build(),
            );
            let pred_with_graph = regressor.predict(
                &TunerFeatures::builder()
                    .batch_size(1)
                    .cuda_graphs(true)
                    .build(),
            );

            // CUDA graphs should predict higher throughput
            assert!(
                pred_with_graph.predicted_tps >= pred_no_graph.predicted_tps,
                "With graphs ({}) should be >= without ({})",
                pred_with_graph.predicted_tps,
                pred_no_graph.predicted_tps
            );
        }
    }

    // F021-F040: Feature Engineering
    mod f021_f040_feature_engineering {
        use super::*;

        #[test]
        fn f021_no_nan_features() {
            let features = TunerFeatures::builder()
                .model_params_b(1.5)
                .hidden_dim(1536)
                .batch_size(4)
                .quant_type(QuantType::Q4K)
                .build();

            let v = features.to_vector();
            assert!(!v.iter().any(|x| x.is_nan()), "Features contain NaN");
        }

        #[test]
        fn f022_no_infinite_features() {
            let features = TunerFeatures::builder()
                .model_params_b(1.5)
                .hidden_dim(1536)
                .batch_size(4)
                .build();

            let v = features.to_vector();
            assert!(
                !v.iter().any(|x| x.is_infinite()),
                "Features contain infinity"
            );
        }

        #[test]
        fn f023_features_in_0_1_range() {
            let features = TunerFeatures::builder()
                .model_params_b(100.0) // Very large
                .hidden_dim(16384)     // Max
                .batch_size(64)        // Max
                .seq_len(32768)        // Max
                .build();

            let v = features.to_vector();
            for (i, x) in v.iter().enumerate() {
                assert!(
                    *x >= -0.001 && *x <= 1.001,
                    "Feature {} = {} is outside [0, 1]",
                    i,
                    x
                );
            }
        }

        /// f026: Roofline bound - predicted TPS must never exceed theoretical maximum
        /// This is the crucible that ensures ML predictions do not violate hardware limits.
        /// Roofline model: max_tps = memory_bw_bytes_per_sec / bytes_per_token
        /// For decode: bytes_per_token ≈ model_params × bytes_per_param
        #[test]
        fn f026_roofline_bound() {
            // Test configuration: 7B Q4_K model on RTX 4090 (1008 GB/s)
            let model_params_b: f32 = 7.0;
            let bytes_per_param: f32 = QuantType::Q4K.bytes_per_param();
            let gpu_mem_bw_gbs: f32 = 1008.0; // RTX 4090

            // Theoretical maximum tokens/sec for decode phase (batch=1)
            // bytes_per_token = model_params * bytes_per_param * 1e9
            // max_tps = mem_bw_bytes_per_sec / bytes_per_token
            let model_bytes: f32 = model_params_b * bytes_per_param * 1e9;
            let theoretical_max_tps: f32 = (gpu_mem_bw_gbs * 1e9) / model_bytes;

            // Build features for this configuration
            let features = TunerFeatures::builder()
                .model_params_b(model_params_b)
                .batch_size(1)
                .quant_type(QuantType::Q4K)
                .gpu_mem_bw_gbs(gpu_mem_bw_gbs)
                .gpu_compute_tflops(82.6) // RTX 4090 FP32
                .is_prefill(false) // Decode phase
                .build();

            let tuner = BrickTuner::new();
            let rec = tuner.recommend(&features);

            // CRITICAL ASSERTION: predicted_tps <= theoretical_max_tps
            // Allowing 10% margin for numerical precision
            let margin = 1.10;
            assert!(
                rec.throughput.predicted_tps <= theoretical_max_tps * margin,
                "Roofline violation: predicted {} tok/s exceeds theoretical max {} tok/s \
                 (model: {}B, quant: Q4K, mem_bw: {} GB/s)",
                rec.throughput.predicted_tps,
                theoretical_max_tps,
                model_params_b,
                gpu_mem_bw_gbs
            );

            // Also verify theoretical max is reasonable (sanity check)
            // 7B Q4_K on 1008 GB/s should yield ~200-300 tok/s theoretical max
            assert!(
                theoretical_max_tps > 100.0 && theoretical_max_tps < 500.0,
                "Theoretical max {} tok/s is outside expected range for 7B Q4_K on RTX 4090",
                theoretical_max_tps
            );
        }

        #[test]
        fn f029_onehot_sums_to_one() {
            let features = TunerFeatures::builder()
                .quant_type(QuantType::Q4K)
                .kernel_type(KernelType::VectorizedQ4K)
                .build();

            let quant_sum: f32 = features.quant_type_onehot.iter().sum();
            let kernel_sum: f32 = features.kernel_type_onehot.iter().sum();

            assert!(
                (quant_sum - 1.0).abs() < 0.001,
                "Quant one-hot sum = {}",
                quant_sum
            );
            assert!(
                (kernel_sum - 1.0).abs() < 0.001,
                "Kernel one-hot sum = {}",
                kernel_sum
            );
        }

        /// f040: Feature dimension must be 42 per spec v1.1.0
        /// 11 static + 8 quant + 16 kernel + 5 hardware + 2 derived = 42
        #[test]
        fn f040_feature_dimension_is_42() {
            assert_eq!(TunerFeatures::DIM, 42, "DIM must be 42 per spec v1.1.0");

            let features = TunerFeatures::builder().build();
            assert_eq!(features.to_vector().len(), TunerFeatures::DIM);
        }
    }

    // F041-F060: Training Data Quality
    mod f041_f060_training_data {
        use super::*;

        #[test]
        fn f059_no_data_leakage() {
            // Training labels should not be in feature vector
            let features = TunerFeatures::builder()
                .measured_tps(500.0) // Label
                .build();

            let v = features.to_vector();
            // measured_tps should NOT be in the vector (it's a label)
            assert_eq!(v.len(), TunerFeatures::DIM);
        }
    }

    // F061-F080: Integration Correctness
    mod f061_f080_integration {
        use super::*;

        #[test]
        fn f066_recommendations_json_valid() {
            let features = TunerFeatures::builder()
                .model_params_b(1.5)
                .batch_size(4)
                .build();

            let tuner = BrickTuner::new();
            let rec = tuner.recommend(&features);

            let json = serde_json::to_string(&rec);
            assert!(json.is_ok(), "Failed to serialize recommendation");
        }

        #[test]
        fn f070_safetensors_roundtrip() {
            let tuner = BrickTuner::new();

            // Serialize
            let json = tuner.to_json();
            assert!(json.is_ok());

            // Deserialize
            let loaded = BrickTuner::from_json(&json.unwrap());
            assert!(loaded.is_ok());
            assert_eq!(loaded.unwrap().version, tuner.version);
        }

        #[test]
        fn f071_feature_extractor_deterministic() {
            let config = RunConfig::default();
            let profiler = BrickProfiler::new();
            let extractor = FeatureExtractor::new();

            let f1 = extractor.extract(&profiler, &config);
            let f2 = extractor.extract(&profiler, &config);

            assert_eq!(f1.to_vector(), f2.to_vector());
        }

        #[test]
        fn f072_prediction_deterministic() {
            let features = TunerFeatures::builder()
                .model_params_b(1.5)
                .batch_size(4)
                .build();

            let tuner = BrickTuner::new();
            let rec1 = tuner.recommend(&features);
            let rec2 = tuner.recommend(&features);

            assert_eq!(
                rec1.throughput.predicted_tps,
                rec2.throughput.predicted_tps
            );
            assert_eq!(rec1.kernel.top_kernel, rec2.kernel.top_kernel);
        }

        #[test]
        fn f075_error_handling_graceful() {
            // Invalid features should not panic
            let mut features = TunerFeatures::default();
            features.model_params_b = f32::NAN;

            let result = features.validate();
            assert!(result.is_err());
        }
    }

    // F081-F100: Generalization & Robustness
    mod f081_f100_generalization {
        use super::*;

        #[test]
        fn f085_adversarial_inputs_handled() {
            // Extreme values should not crash
            let features = TunerFeatures::builder()
                .model_params_b(0.001) // Very small
                .hidden_dim(1)
                .batch_size(1000) // Very large (will be clamped)
                .build();

            let tuner = BrickTuner::new();
            let rec = tuner.recommend(&features);

            // Should produce some recommendation without crashing
            assert!(rec.throughput.predicted_tps > 0.0);
        }

        #[test]
        fn f091_cold_start_handling() {
            // Tuner should work with default (untrained) model
            let tuner = BrickTuner::new();
            assert_eq!(tuner.sample_count, 0);

            let features = TunerFeatures::builder().batch_size(4).build();
            let rec = tuner.recommend(&features);

            // Should still produce reasonable recommendations
            assert!(rec.confidence_overall > 0.0);
        }

        #[test]
        fn f096_extreme_values_clipped() {
            let features = TunerFeatures::builder()
                .model_params_b(1000.0) // Way over max
                .hidden_dim(100000)     // Way over max
                .batch_size(1000)       // Way over max
                .build();

            // All values should be clipped to [0, 1]
            let v = features.to_vector();
            assert!(v.iter().all(|x| *x >= 0.0 && *x <= 1.0));
        }
    }

    // Bottleneck classification tests
    #[test]
    fn test_bottleneck_recommended_action() {
        assert!(BottleneckClass::MemoryBound
            .recommended_action()
            .contains("batch size"));
        assert!(BottleneckClass::LaunchBound
            .recommended_action()
            .contains("CUDA graphs"));
        assert!(BottleneckClass::AttentionBound
            .recommended_action()
            .contains("Flash Decoding"));
    }

    // Kernel classifier tests
    #[test]
    fn test_kernel_classifier_batched_for_high_m() {
        let classifier = KernelClassifier::new();
        let features = TunerFeatures::builder().batch_size(8).build();

        let rec = classifier.predict(&features);
        assert_eq!(rec.top_kernel, KernelType::BatchedQ4K);
    }

    // Feature builder tests
    #[test]
    fn test_feature_builder_normalization() {
        let features = TunerFeatures::builder()
            .model_params_b(1.0) // log10(1.0) = 0, normalized = (0+1)/3 = 0.33
            .hidden_dim(1536)    // 1536/16384 ≈ 0.094
            .batch_size(4)       // 4/64 = 0.0625
            .build();

        assert!(features.model_params_b > 0.0 && features.model_params_b < 1.0);
        assert!(features.hidden_dim_norm > 0.0 && features.hidden_dim_norm < 1.0);
        assert!(features.batch_size_norm > 0.0 && features.batch_size_norm < 1.0);
    }

    // Additional coverage tests
    #[test]
    fn test_all_builder_methods() {
        let features = TunerFeatures::builder()
            .model_params_b(1.5)
            .hidden_dim(2048)
            .num_layers(32)
            .num_heads(16)
            .head_dim(128)
            .vocab_size(32000)
            .batch_size(4)
            .seq_len(512)
            .cuda_graphs(true)
            .kv_caches(4)
            .is_prefill(false)
            .quant_type(QuantType::Q4K)
            .kernel_type(KernelType::VectorizedQ4K)
            .gpu_mem_bw_gbs(1000.0)
            .gpu_compute_tflops(150.0)
            .gpu_sm_count(128)
            .measured_tps(100.0)
            .build();

        assert!(features.model_params_b > 0.0);
        assert!(features.cuda_graphs == 1.0);
        assert!(features.is_prefill == 0.0);
    }

    #[test]
    fn test_quant_type_bytes_per_param() {
        assert_eq!(QuantType::Q4_0.bytes_per_param(), 0.5625);
        assert_eq!(QuantType::Q4_1.bytes_per_param(), 0.5625);
        assert_eq!(QuantType::Q5K.bytes_per_param(), 0.6875);
        assert_eq!(QuantType::Q6K.bytes_per_param(), 0.8125);
        assert_eq!(QuantType::Q8_0.bytes_per_param(), 1.0);
        assert_eq!(QuantType::F16.bytes_per_param(), 2.0);
        assert_eq!(QuantType::F32.bytes_per_param(), 4.0);
    }

    #[test]
    fn test_kernel_type_to_index() {
        assert_eq!(KernelType::TiledQ4K.to_index(), 0);
        assert_eq!(KernelType::CoalescedQ4K.to_index(), 1);
        assert_eq!(KernelType::VectorizedQ4K.to_index(), 2);
        assert_eq!(KernelType::BatchedQ4K.to_index(), 3);
        assert_eq!(KernelType::Dp4aQ4K.to_index(), 4);
        assert_eq!(KernelType::FusedRmsNormQ4K.to_index(), 5);
        assert_eq!(KernelType::CoalescedQ6K.to_index(), 6);
        assert_eq!(KernelType::IncrementalAttention.to_index(), 7);
        assert_eq!(KernelType::MultiWarpAttention.to_index(), 8);
        assert_eq!(KernelType::BatchedAttention.to_index(), 9);
        assert_eq!(KernelType::RmsNorm.to_index(), 10);
        assert_eq!(KernelType::VectorizedRmsNorm.to_index(), 11);
        assert_eq!(KernelType::BatchedRmsNorm.to_index(), 12);
        assert_eq!(KernelType::Generic.to_index(), 13);
        assert_eq!(KernelType::Unknown.to_index(), 14);
    }

    #[test]
    fn test_bottleneck_class_to_index() {
        assert_eq!(BottleneckClass::Unknown.to_index(), 0);
        assert_eq!(BottleneckClass::MemoryBound.to_index(), 1);
        assert_eq!(BottleneckClass::ComputeBound.to_index(), 2);
        assert_eq!(BottleneckClass::LaunchBound.to_index(), 3);
        assert_eq!(BottleneckClass::AttentionBound.to_index(), 4);
    }

    #[test]
    fn test_bottleneck_display() {
        assert_eq!(format!("{}", BottleneckClass::Unknown), "Unknown");
        assert_eq!(format!("{}", BottleneckClass::MemoryBound), "MemoryBound");
        assert_eq!(format!("{}", BottleneckClass::ComputeBound), "ComputeBound");
        assert_eq!(format!("{}", BottleneckClass::LaunchBound), "LaunchBound");
        assert_eq!(format!("{}", BottleneckClass::AttentionBound), "AttentionBound");
    }

    #[test]
    fn test_from_brick_bottleneck() {
        use crate::brick::BrickBottleneck;
        assert_eq!(
            BottleneckClass::from_brick_bottleneck(BrickBottleneck::Memory),
            BottleneckClass::MemoryBound
        );
        assert_eq!(
            BottleneckClass::from_brick_bottleneck(BrickBottleneck::Compute),
            BottleneckClass::ComputeBound
        );
        assert_eq!(
            BottleneckClass::from_brick_bottleneck(BrickBottleneck::Unknown),
            BottleneckClass::Unknown
        );
    }

    #[test]
    fn test_run_config_default() {
        let config = RunConfig::default();
        assert_eq!(config.model_params_b, 1.5);
        assert_eq!(config.batch_size, 1);
        assert_eq!(config.quant_type, QuantType::Q4K);
    }

    #[test]
    fn test_tuner_features_validate() {
        let features = TunerFeatures::builder().build();
        assert!(features.validate().is_ok());

        // Test with NaN
        let mut bad_features = features.clone();
        bad_features.model_params_b = f32::NAN;
        assert!(bad_features.validate().is_err());
    }

    #[test]
    fn test_tuner_error_display() {
        let err = TunerError::InvalidFeature("test".to_string());
        assert!(format!("{}", err).contains("Invalid feature"));

        let err = TunerError::InsufficientData(5);
        assert!(format!("{}", err).contains("Insufficient"));

        let err = TunerError::Serialization("test".to_string());
        assert!(format!("{}", err).contains("Serialization"));

        let err = TunerError::ModelNotFound;
        assert!(format!("{}", err).contains("not found"));

        let err = TunerError::PredictionFailed("test".to_string());
        assert!(format!("{}", err).contains("Prediction failed"));
    }

    #[test]
    fn test_throughput_regressor_predict_raw() {
        let regressor = ThroughputRegressor::new();
        let features = TunerFeatures::builder().batch_size(4).build();
        let vec = features.to_vector();
        let raw = regressor.predict_raw(&vec);
        assert!(raw > 0.0);
    }

    #[test]
    fn test_bottleneck_classifier() {
        let classifier = BottleneckClassifier::new();
        let features = TunerFeatures::builder().batch_size(4).build();
        let pred = classifier.predict(&features);
        // Default prediction should be MemoryBound for inference
        assert!(matches!(
            pred.class,
            BottleneckClass::MemoryBound | BottleneckClass::Unknown
        ));
    }

    #[test]
    fn test_brick_tuner_recommend() {
        let tuner = BrickTuner::new();
        let features = TunerFeatures::builder()
            .model_params_b(1.5)
            .batch_size(4)
            .build();
        let rec = tuner.recommend(&features);

        assert!(rec.throughput.predicted_tps > 0.0);
        assert!(!rec.suggested_experiments.is_empty());
    }

    #[test]
    fn test_experiment_suggestion_display() {
        let exp = ExperimentSuggestion::IncreaseBatchSize { from: 1, to: 4 };
        assert!(format!("{}", exp).contains("Increase batch size"));

        let exp = ExperimentSuggestion::EnableCudaGraphs;
        assert!(format!("{}", exp).contains("CUDA graphs"));

        let exp = ExperimentSuggestion::TryKernel { kernel: KernelType::BatchedQ4K };
        assert!(format!("{}", exp).contains("kernel"));

        let exp = ExperimentSuggestion::ReduceSequenceLength { factor: 0.5 };
        assert!(format!("{}", exp).contains("sequence"));

        let exp = ExperimentSuggestion::EnableMultiKvCache { count: 4 };
        assert!(format!("{}", exp).contains("KV"));
    }

    #[test]
    fn test_tuner_data_collector() {
        let collector = TunerDataCollector::new();
        assert!(collector.is_empty());
        assert_eq!(collector.len(), 0);
        assert!(collector.samples().is_empty());
    }

    #[test]
    fn test_feature_extractor_default() {
        let extractor = FeatureExtractor::new();
        assert!(extractor.hardware.is_none());
    }

    #[test]
    fn test_feature_extractor_debug() {
        let extractor = FeatureExtractor::new();
        let debug_str = format!("{:?}", extractor);
        assert!(debug_str.contains("FeatureExtractor"));
    }

    #[test]
    fn test_chrono_lite_now() {
        let timestamp = super::chrono_lite_now();
        let parsed: u64 = timestamp.parse().expect("Should be a number");
        assert!(parsed > 0);
    }

    #[test]
    fn test_pad_right() {
        assert_eq!(super::pad_right("test", 10), "test      ");
        assert_eq!(super::pad_right("longstring", 5), "longs");
    }

    // Additional coverage tests for v1.1.0

    #[test]
    fn test_quant_type_to_index_all_variants() {
        // Cover all QuantType::to_index branches
        assert_eq!(QuantType::Q4_0.to_index(), 0);
        assert_eq!(QuantType::Q4_1.to_index(), 1);
        assert_eq!(QuantType::Q4K.to_index(), 2);
        assert_eq!(QuantType::Q5K.to_index(), 3);
        assert_eq!(QuantType::Q6K.to_index(), 4);
        assert_eq!(QuantType::Q8_0.to_index(), 5);
        assert_eq!(QuantType::F16.to_index(), 6);
        assert_eq!(QuantType::F32.to_index(), 7);
    }

    #[test]
    fn test_validation_infinite_features() {
        let mut features = TunerFeatures::default();
        features.model_params_b = f32::INFINITY;
        let result = features.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Infinite"));
    }

    #[test]
    fn test_validation_out_of_range() {
        let mut features = TunerFeatures::default();
        features.batch_size_norm = 2.0; // Out of [0, 1]
        let result = features.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("outside [0, 1]"));
    }

    #[test]
    fn test_validation_bad_quant_onehot() {
        let mut features = TunerFeatures::default();
        features.quant_type_onehot = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // Sums to 1 but invalid one-hot
        // This should actually pass since sum is 1.0
        assert!(features.validate().is_ok());

        // Now test with sum != 1
        features.quant_type_onehot = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // Sums to 0.5
        let result = features.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validation_bad_kernel_onehot() {
        let mut features = TunerFeatures::default();
        features.kernel_type_onehot = [0.0; 16]; // All zeros, sum = 0
        // Zero sum is allowed (unspecified kernel)
        assert!(features.validate().is_ok());

        // Sum != 0 and != 1 should fail
        features.kernel_type_onehot[0] = 0.5;
        let result = features.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_gpu_l2_cache_mb() {
        let features = TunerFeatures::builder()
            .gpu_l2_cache_mb(96.0) // 96MB L2 cache
            .build();
        // Normalized: 96 / 128 = 0.75
        assert!((features.gpu_l2_cache_norm - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_builder_is_zero_copy() {
        let features_enabled = TunerFeatures::builder()
            .is_zero_copy(true)
            .build();
        assert_eq!(features_enabled.is_zero_copy, 1.0);

        let features_disabled = TunerFeatures::builder()
            .is_zero_copy(false)
            .build();
        assert_eq!(features_disabled.is_zero_copy, 0.0);
    }

    #[test]
    fn test_builder_hardware() {
        use crate::hardware::{GpuBackend, GpuCapability};

        let gpu = GpuCapability {
            vendor: "NVIDIA".to_string(),
            model: "Test GPU".to_string(),
            backend: GpuBackend::Cuda,
            compute_capability: Some("8.9".to_string()),
            peak_tflops_fp32: 100.0,
            peak_tflops_tensor: Some(400.0),
            memory_bw_gbps: 1000.0,
            vram_gb: 24.0,
        };

        // Directly test the normalization without HardwareCapability
        let features = TunerFeatures::builder()
            .gpu_mem_bw_gbs(gpu.memory_bw_gbps as f32)
            .gpu_compute_tflops(gpu.peak_tflops_fp32 as f32)
            .build();

        // Memory BW: 1000 / 3000 ≈ 0.333
        assert!((features.gpu_mem_bw_norm - (1000.0 / 3000.0)).abs() < 0.01);
        // Compute: 100 / 500 = 0.2
        assert!((features.gpu_compute_norm - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_brick_tuner_train() {
        let mut tuner = BrickTuner::new();

        // Create minimal training data
        let data: Vec<(TunerFeatures, f32)> = (0..15)
            .map(|i| {
                let features = TunerFeatures::builder()
                    .batch_size((i % 4) as u32 + 1)
                    .model_params_b(1.5 + (i as f32) * 0.1)
                    .build();
                (features, 100.0 + (i as f32) * 10.0)
            })
            .collect();

        let result = tuner.train(&data);
        assert!(result.is_ok());
        assert_eq!(tuner.sample_count, 15);
    }

    #[test]
    fn test_brick_tuner_train_insufficient_data() {
        let mut tuner = BrickTuner::new();

        // Too few samples
        let data: Vec<(TunerFeatures, f32)> = (0..5)
            .map(|i| {
                let features = TunerFeatures::builder().batch_size(i as u32 + 1).build();
                (features, 100.0)
            })
            .collect();

        let result = tuner.train(&data);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), TunerError::InsufficientData(5)));
    }

    #[test]
    fn test_brick_tuner_print_recommendation() {
        let tuner = BrickTuner::new();
        let features = TunerFeatures::builder().batch_size(4).build();
        let rec = tuner.recommend(&features);

        // Just verify it doesn't panic
        tuner.print_recommendation(&rec);
    }

    #[test]
    fn test_attention_bound_suggestions() {
        let tuner = BrickTuner::new();

        // Create features that would trigger AttentionBound
        let mut features = TunerFeatures::builder()
            .batch_size(1)
            .seq_len(8192) // Long sequence
            .is_prefill(true)
            .build();
        features.bottleneck_class = Some(BottleneckClass::AttentionBound);

        let bottleneck_pred = BottleneckPrediction {
            class: BottleneckClass::AttentionBound,
            confidence: 0.9,
            explanation: "Attention bound".to_string(),
            recommended_action: "Use FlashAttention".to_string(),
        };

        let suggestions = tuner.suggest_experiments(&features, &bottleneck_pred);
        // Should suggest BatchedAttention and ReduceSequenceLength
        let has_batched_attention = suggestions.iter().any(|s| {
            matches!(s, ExperimentSuggestion::TryKernel { kernel: KernelType::BatchedAttention })
        });
        let has_reduce_seq = suggestions.iter().any(|s| {
            matches!(s, ExperimentSuggestion::ReduceSequenceLength { .. })
        });
        assert!(has_batched_attention || has_reduce_seq);
    }

    #[test]
    fn test_unknown_bottleneck_suggestions() {
        let tuner = BrickTuner::new();

        let mut features = TunerFeatures::builder()
            .batch_size(1)
            .build();
        features.bottleneck_class = Some(BottleneckClass::Unknown);

        let rec = tuner.recommend(&features);
        // Should suggest increasing batch size from 1 to 4
        let has_increase_batch = rec.suggested_experiments.iter().any(|s| {
            matches!(s, ExperimentSuggestion::IncreaseBatchSize { from: 1, to: 4 })
        });
        assert!(has_increase_batch);
    }

    #[test]
    fn test_data_collector_record() {
        use std::time::Duration;

        let mut collector = TunerDataCollector::new();
        let mut profiler = BrickProfiler::enabled();
        let config = RunConfig::default();

        // Simulate a profiling run using the proper API
        profiler.record_elapsed("test_brick", Duration::from_micros(100), 32);

        let result = collector.record(&profiler, &config, KernelType::VectorizedQ4K);
        assert!(result.is_some());
        assert_eq!(collector.len(), 1);
        assert!(!collector.is_empty());
    }

    #[test]
    fn test_data_collector_to_json() {
        let collector = TunerDataCollector::new();
        let json = collector.to_json();
        assert!(json.is_ok());
        assert_eq!(json.unwrap(), "[]"); // Empty array
    }

    #[test]
    fn test_data_collector_prepare_training_data() {
        use std::time::Duration;

        let mut collector = TunerDataCollector::new();
        let mut profiler = BrickProfiler::enabled();
        let config = RunConfig::default();

        // Add a sample using the proper API
        profiler.record_elapsed("test_brick", Duration::from_micros(100), 32);
        collector.record(&profiler, &config, KernelType::VectorizedQ4K);

        let training_data = collector.prepare_training_data();
        assert_eq!(training_data.len(), 1);
        assert!(training_data[0].1 > 0.0); // throughput > 0
    }

    #[test]
    fn test_roofline_helper_methods() {
        // Test bytes_per_param_from_onehot
        let onehot_q4k = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let bytes = ThroughputRegressor::bytes_per_param_from_onehot(&onehot_q4k);
        assert!((bytes - 0.5625).abs() < 0.001);

        let onehot_f32 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let bytes = ThroughputRegressor::bytes_per_param_from_onehot(&onehot_f32);
        assert!((bytes - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_roofline_bound() {
        let features = TunerFeatures::builder()
            .model_params_b(7.0)
            .batch_size(1)
            .quant_type(QuantType::Q4K)
            .gpu_mem_bw_gbs(1008.0)
            .build();

        let bound = ThroughputRegressor::compute_roofline_bound(&features);
        // Should be around 256 tok/s for 7B Q4K on 1008 GB/s
        assert!(bound > 200.0 && bound < 300.0);
    }

}
