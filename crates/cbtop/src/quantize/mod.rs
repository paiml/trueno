//! QuantizedBrick Implementation (PMAT-013)
//!
//! Implements quantized weight support for ComputeBricks per cbtop spec §17.
//!
//! # Supported Formats
//!
//! | Format | Bits/Weight | Memory | Perplexity Delta |
//! |--------|------------|--------|------------------|
//! | Q4_0   | 4.0        | 25%    | ~0.5%            |
//! | Q4_K   | 4.5        | 28%    | ~0.3%            |
//! | Q5_K   | 5.5        | 34%    | ~0.1%            |
//! | Q8_0   | 8.0        | 50%    | ~0.01%           |
//!
//! # Citations
//!
//! - [Dettmers et al. 2022] "LLM.int8(): 8-bit Matrix Multiplication" NeurIPS
//! - [Frantar et al. 2023] "GPTQ: Accurate Post-Training Quantization" ICLR
//! - [Lin et al. 2023] "AWQ: Activation-aware Weight Quantization" MLSys

// Allow non-camel-case for GGML standard quantization type names
#![allow(non_camel_case_types)]

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// Supported quantization formats for ComputeBricks.
///
/// Based on GGML/llama.cpp quantization types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    /// Full precision (baseline)
    F32,
    /// Half precision
    F16,
    /// Brain float 16
    BF16,

    /// 4-bit quantization, no scales per block (32 values/block)
    Q4_0,
    /// 4-bit quantization with K-quants (256 values/super-block, 6-bit scales)
    Q4_K,
    /// 5-bit quantization with K-quants
    Q5_K,
    /// 6-bit quantization with K-quants
    Q6_K,
    /// 8-bit quantization, simple (32 values/block)
    Q8_0,

    /// GPTQ format (ExLlama compatible)
    Gptq { bits: u8, group_size: u16 },
    /// AWQ format (activation-aware)
    Awq { bits: u8 },
}

impl QuantFormat {
    /// Effective bits per weight (including scales and metadata).
    pub fn bits_per_weight(&self) -> f64 {
        match self {
            QuantFormat::F32 => 32.0,
            QuantFormat::F16 => 16.0,
            QuantFormat::BF16 => 16.0,
            QuantFormat::Q4_0 => 4.5, // 4 bits + scale overhead
            QuantFormat::Q4_K => 4.5, // 4 bits + 6-bit scales
            QuantFormat::Q5_K => 5.5,
            QuantFormat::Q6_K => 6.5,
            QuantFormat::Q8_0 => 8.5, // 8 bits + scale
            QuantFormat::Gptq { bits, .. } => *bits as f64 + 0.5,
            QuantFormat::Awq { bits } => *bits as f64 + 0.5,
        }
    }

    /// Memory ratio compared to F16 (lower is better).
    pub fn memory_ratio(&self) -> f64 {
        self.bits_per_weight() / 16.0
    }

    /// Expected perplexity delta compared to F16 (lower is better).
    ///
    /// Based on llama.cpp benchmarks.
    pub fn expected_ppl_delta(&self) -> f64 {
        match self {
            QuantFormat::F32 => 0.0,
            QuantFormat::F16 => 0.0,
            QuantFormat::BF16 => 0.01,
            QuantFormat::Q4_0 => 0.5,
            QuantFormat::Q4_K => 0.3,
            QuantFormat::Q5_K => 0.1,
            QuantFormat::Q6_K => 0.05,
            QuantFormat::Q8_0 => 0.01,
            QuantFormat::Gptq { bits, .. } => match bits {
                4 => 0.4,
                8 => 0.02,
                _ => 0.5,
            },
            QuantFormat::Awq { bits } => match bits {
                4 => 0.2, // AWQ tends to have better quality
                _ => 0.3,
            },
        }
    }

    /// Block size (number of weights per quantization block).
    pub fn block_size(&self) -> usize {
        match self {
            QuantFormat::F32 | QuantFormat::F16 | QuantFormat::BF16 => 1,
            QuantFormat::Q4_0 | QuantFormat::Q8_0 => 32,
            QuantFormat::Q4_K | QuantFormat::Q5_K | QuantFormat::Q6_K => 256, // Super-block
            QuantFormat::Gptq { group_size, .. } => *group_size as usize,
            QuantFormat::Awq { .. } => 128,
        }
    }

    /// Bytes per block.
    pub fn bytes_per_block(&self) -> usize {
        match self {
            QuantFormat::F32 => 4,
            QuantFormat::F16 | QuantFormat::BF16 => 2,
            QuantFormat::Q4_0 => 18,  // 2 (scale) + 16 (32 × 4-bit)
            QuantFormat::Q4_K => 144, // 2+2+12+128 (super-block)
            QuantFormat::Q5_K => 176, // 2+2+12+128+32
            QuantFormat::Q6_K => 210, // 128+64+16+2
            QuantFormat::Q8_0 => 34,  // 2 (scale) + 32 (8-bit values)
            QuantFormat::Gptq { bits, group_size } => {
                let data_bytes = (*group_size as usize * *bits as usize).div_ceil(8);
                data_bytes + 4 // + scale/zero
            }
            QuantFormat::Awq { bits } => {
                let data_bytes = (128 * *bits as usize).div_ceil(8);
                data_bytes + 4
            }
        }
    }
}

impl fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantFormat::F32 => write!(f, "F32"),
            QuantFormat::F16 => write!(f, "F16"),
            QuantFormat::BF16 => write!(f, "BF16"),
            QuantFormat::Q4_0 => write!(f, "Q4_0"),
            QuantFormat::Q4_K => write!(f, "Q4_K"),
            QuantFormat::Q5_K => write!(f, "Q5_K"),
            QuantFormat::Q6_K => write!(f, "Q6_K"),
            QuantFormat::Q8_0 => write!(f, "Q8_0"),
            QuantFormat::Gptq { bits, group_size } => {
                write!(f, "GPTQ-{}bit-g{}", bits, group_size)
            }
            QuantFormat::Awq { bits } => write!(f, "AWQ-{}bit", bits),
        }
    }
}

/// Dequantization strategy.
///
/// Controls when and how quantized weights are dequantized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DequantStrategy {
    /// Fused: dequantize during matmul (best for GPU, saves memory bandwidth)
    Fused,
    /// Prefetch: dequantize ahead of compute (good for pipelining)
    Prefetch { lookahead_blocks: usize },
    /// On-demand: dequantize per block (lowest memory footprint)
    OnDemand,
}

impl Default for DequantStrategy {
    fn default() -> Self {
        DequantStrategy::Fused
    }
}

/// Quantized weight storage for a single layer.
#[derive(Debug, Clone)]
pub struct QuantizedWeights {
    /// Quantization format
    pub format: QuantFormat,
    /// Raw quantized data
    pub data: Vec<u8>,
    /// Shape: [rows, cols] for 2D weights
    pub shape: (usize, usize),
    /// Layer name (for debugging)
    pub layer_name: String,
}

impl QuantizedWeights {
    /// Create new quantized weights.
    pub fn new(format: QuantFormat, data: Vec<u8>, shape: (usize, usize), name: &str) -> Self {
        Self {
            format,
            data,
            shape,
            layer_name: name.to_string(),
        }
    }

    /// Total number of weights.
    pub fn num_weights(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    /// Memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }

    /// Memory footprint if stored as F16.
    pub fn f16_memory_bytes(&self) -> usize {
        self.num_weights() * 2
    }

    /// Compression ratio (F16 / quantized).
    pub fn compression_ratio(&self) -> f64 {
        self.f16_memory_bytes() as f64 / self.memory_bytes() as f64
    }

    /// Effective bits per weight (actual).
    pub fn actual_bits_per_weight(&self) -> f64 {
        (self.data.len() * 8) as f64 / self.num_weights() as f64
    }
}

/// Quantization statistics for a model or layer.
#[derive(Debug, Clone, Default)]
pub struct QuantStats {
    /// Total weights across all layers
    pub total_weights: usize,
    /// Total memory (bytes) for quantized weights
    pub total_memory_bytes: usize,
    /// Memory if stored as F16
    pub f16_memory_bytes: usize,
    /// Weights per format
    pub weights_by_format: HashMap<QuantFormat, usize>,
    /// Memory per format
    pub memory_by_format: HashMap<QuantFormat, usize>,
    /// Per-layer stats
    pub layer_stats: Vec<LayerQuantStats>,
}

/// Per-layer quantization statistics.
#[derive(Debug, Clone)]
pub struct LayerQuantStats {
    /// Layer name
    pub name: String,
    /// Quantization format
    pub format: QuantFormat,
    /// Weight count
    pub weights: usize,
    /// Memory bytes
    pub memory_bytes: usize,
    /// Compression ratio
    pub compression_ratio: f64,
}

impl QuantStats {
    /// Create new empty stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add layer statistics.
    pub fn add_layer(&mut self, weights: &QuantizedWeights) {
        self.total_weights += weights.num_weights();
        self.total_memory_bytes += weights.memory_bytes();
        self.f16_memory_bytes += weights.f16_memory_bytes();

        *self.weights_by_format.entry(weights.format).or_default() += weights.num_weights();
        *self.memory_by_format.entry(weights.format).or_default() += weights.memory_bytes();

        self.layer_stats.push(LayerQuantStats {
            name: weights.layer_name.clone(),
            format: weights.format,
            weights: weights.num_weights(),
            memory_bytes: weights.memory_bytes(),
            compression_ratio: weights.compression_ratio(),
        });
    }

    /// Overall compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        if self.total_memory_bytes == 0 {
            1.0
        } else {
            self.f16_memory_bytes as f64 / self.total_memory_bytes as f64
        }
    }

    /// Effective bits per weight (average).
    pub fn avg_bits_per_weight(&self) -> f64 {
        if self.total_weights == 0 {
            0.0
        } else {
            (self.total_memory_bytes * 8) as f64 / self.total_weights as f64
        }
    }

    /// Dominant format (most weights).
    pub fn dominant_format(&self) -> Option<QuantFormat> {
        self.weights_by_format
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(format, _)| *format)
    }
}

impl fmt::Display for QuantStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Quantization Statistics")?;
        writeln!(f, "======================")?;
        writeln!(f, "Total Weights: {}", self.total_weights)?;
        writeln!(
            f,
            "Total Memory: {:.2} MB (quantized)",
            self.total_memory_bytes as f64 / 1_000_000.0
        )?;
        writeln!(
            f,
            "F16 Memory: {:.2} MB (baseline)",
            self.f16_memory_bytes as f64 / 1_000_000.0
        )?;
        writeln!(f, "Compression: {:.2}x", self.compression_ratio())?;
        writeln!(f, "Avg Bits/Weight: {:.2}", self.avg_bits_per_weight())?;

        if !self.weights_by_format.is_empty() {
            writeln!(f)?;
            writeln!(f, "By Format:")?;
            for (format, weights) in &self.weights_by_format {
                let memory = self.memory_by_format.get(format).unwrap_or(&0);
                writeln!(
                    f,
                    "  {}: {} weights, {:.2} MB",
                    format,
                    weights,
                    *memory as f64 / 1_000_000.0
                )?;
            }
        }

        Ok(())
    }
}

/// GGUF file header (simplified parsing).
///
/// GGUF format specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// Magic number ("GGUF")
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Number of tensors
    pub tensor_count: u64,
    /// Number of metadata key-value pairs
    pub metadata_kv_count: u64,
}

/// GGUF metadata value types.
#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

/// GGUF tensor info.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    /// Tensor name
    pub name: String,
    /// Number of dimensions
    pub n_dims: u32,
    /// Dimensions
    pub dims: Vec<u64>,
    /// Data type (GGML type)
    pub dtype: u32,
    /// Offset in data section
    pub offset: u64,
}

/// Result type for GGUF operations.
pub type GgufResult<T> = Result<T, GgufError>;

/// GGUF parsing errors.
#[derive(Debug, Clone)]
pub enum GgufError {
    /// Invalid magic number
    InvalidMagic([u8; 4]),
    /// Unsupported version
    UnsupportedVersion(u32),
    /// IO error
    Io(String),
    /// Invalid data
    InvalidData(String),
    /// Tensor not found
    TensorNotFound(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::InvalidMagic(magic) => {
                write!(f, "Invalid GGUF magic: {:?}", magic)
            }
            GgufError::UnsupportedVersion(v) => {
                write!(f, "Unsupported GGUF version: {}", v)
            }
            GgufError::Io(msg) => write!(f, "IO error: {}", msg),
            GgufError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            GgufError::TensorNotFound(name) => write!(f, "Tensor not found: {}", name),
        }
    }
}

impl std::error::Error for GgufError {}

/// GGUF file loader (basic implementation).
#[derive(Debug)]
pub struct GgufLoader {
    /// File path
    path: String,
    /// Header info
    header: Option<GgufHeader>,
    /// Tensor metadata
    tensors: Vec<GgufTensorInfo>,
    /// Model metadata
    metadata: HashMap<String, GgufValue>,
}

impl GgufLoader {
    /// Create a new GGUF loader for a file path.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_string_lossy().to_string(),
            header: None,
            tensors: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Check if path exists and has GGUF extension.
    pub fn validate_path(&self) -> GgufResult<()> {
        let path = Path::new(&self.path);
        if !path.exists() {
            return Err(GgufError::Io(format!("File not found: {}", self.path)));
        }
        if path.extension().map_or(true, |ext| ext != "gguf") {
            return Err(GgufError::InvalidData(
                "File does not have .gguf extension".to_string(),
            ));
        }
        Ok(())
    }

    /// Parse GGUF header from bytes.
    pub fn parse_header(&mut self, data: &[u8]) -> GgufResult<()> {
        if data.len() < 24 {
            return Err(GgufError::InvalidData(
                "File too small for header".to_string(),
            ));
        }

        let magic: [u8; 4] = data[0..4].try_into().unwrap();
        if &magic != b"GGUF" {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let metadata_kv_count = u64::from_le_bytes(data[16..24].try_into().unwrap());

        self.header = Some(GgufHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        });

        Ok(())
    }

    /// Get parsed header.
    pub fn header(&self) -> Option<&GgufHeader> {
        self.header.as_ref()
    }

    /// Get tensor count.
    pub fn tensor_count(&self) -> u64 {
        self.header.as_ref().map_or(0, |h| h.tensor_count)
    }

    /// Get file path.
    pub fn path(&self) -> &str {
        &self.path
    }
}

/// QuantizedBrick wraps compute operations with quantized weights.
///
/// Per cbtop spec §17.2.
#[derive(Debug, Clone)]
pub struct QuantizedBrick {
    /// Brick name
    pub name: String,
    /// Quantized weights for this brick
    pub weights: Option<QuantizedWeights>,
    /// Dequantization strategy
    pub dequant_strategy: DequantStrategy,
    /// Performance budget (tokens per second)
    pub budget_tok_per_sec: Option<u64>,
}

impl QuantizedBrick {
    /// Create a new quantized brick.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            weights: None,
            dequant_strategy: DequantStrategy::default(),
            budget_tok_per_sec: None,
        }
    }

    /// Set quantized weights.
    pub fn with_weights(mut self, weights: QuantizedWeights) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set dequantization strategy.
    pub fn with_dequant_strategy(mut self, strategy: DequantStrategy) -> Self {
        self.dequant_strategy = strategy;
        self
    }

    /// Set performance budget.
    pub fn with_budget(mut self, tok_per_sec: u64) -> Self {
        self.budget_tok_per_sec = Some(tok_per_sec);
        self
    }

    /// Get memory footprint (bytes).
    pub fn memory_bytes(&self) -> usize {
        self.weights.as_ref().map_or(0, |w| w.memory_bytes())
    }

    /// Get effective bits per weight.
    pub fn bits_per_weight(&self) -> f64 {
        self.weights
            .as_ref()
            .map_or(0.0, |w| w.actual_bits_per_weight())
    }

    /// Get quantization format.
    pub fn format(&self) -> Option<QuantFormat> {
        self.weights.as_ref().map(|w| w.format)
    }

    /// Check if weights are loaded.
    pub fn has_weights(&self) -> bool {
        self.weights.is_some()
    }
}

impl fmt::Display for QuantizedBrick {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QuantizedBrick[{}]", self.name)?;
        if let Some(weights) = &self.weights {
            write!(
                f,
                " format={} weights={} memory={:.2}MB",
                weights.format,
                weights.num_weights(),
                weights.memory_bytes() as f64 / 1_000_000.0
            )?;
        }
        Ok(())
    }
}

/// GGML tensor type to QuantFormat mapping.
pub fn ggml_type_to_format(ggml_type: u32) -> Option<QuantFormat> {
    match ggml_type {
        0 => Some(QuantFormat::F32),
        1 => Some(QuantFormat::F16),
        2 => Some(QuantFormat::Q4_0),
        3 => Some(QuantFormat::Q4_K), // Q4_1 in GGML, map to Q4_K
        8 => Some(QuantFormat::Q8_0),
        12 => Some(QuantFormat::Q4_K),
        13 => Some(QuantFormat::Q5_K),
        14 => Some(QuantFormat::Q6_K),
        _ => None,
    }
}


#[cfg(test)]
mod tests;
