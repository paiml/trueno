//! Execution Graph Node Types and Profiling Primitives
//!
//! This module contains all type definitions for execution path tracking:
//!
//! - **PAR-073**: BrickSample, BrickBottleneck - foundational profiling primitives
//! - **PAR-200**: BrickId, BrickCategory, SyncMode - O(1) hot path brick identification
//! - **PAR-201**: ExecutionNode, EdgeType, etc. - execution hierarchy types

use std::collections::HashMap;
use std::fmt;

// ============================================================================
// BrickProfiler: FOUNDATIONAL Real-Time Per-Brick Timing (PAR-073)
// ============================================================================

/// Individual brick timing sample.
/// Pure Rust timing using `std::time::Instant`.
#[derive(Debug, Clone, Copy)]
pub struct BrickSample {
    /// Brick name hash (for fast lookup)
    pub brick_id: u64,
    /// Elapsed time in nanoseconds
    pub elapsed_ns: u64,
    /// Number of elements processed
    pub elements: u64,
}

/// Bottleneck classification for roofline analysis (PMAT-451)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BrickBottleneck {
    /// Not classified
    #[default]
    Unknown,
    /// Limited by memory bandwidth
    Memory,
    /// Limited by compute throughput
    Compute,
}

impl fmt::Display for BrickBottleneck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BrickBottleneck::Unknown => write!(f, "unknown"),
            BrickBottleneck::Memory => write!(f, "memory"),
            BrickBottleneck::Compute => write!(f, "compute"),
        }
    }
}

// ============================================================================
// PAR-200: BrickProfiler v2 - O(1) Hot Path with BrickId Enum
// ============================================================================

/// Well-known brick types for O(1) lookup on hot path.
///
/// PAR-200: Eliminates string allocation and HashMap hashing during profiling.
/// Use `BrickId::Custom` with string fallback for unknown brick types.
///
/// # Example
/// ```rust
/// use trueno::brick::BrickId;
///
/// let brick = BrickId::RmsNorm;
/// assert_eq!(brick.category(), trueno::brick::BrickCategory::Norm);
/// assert_eq!(brick.name(), "RmsNorm");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum BrickId {
    // Normalization (0-1)
    /// RMS normalization layer
    RmsNorm = 0,
    /// Layer normalization
    LayerNorm = 1,

    // Attention (2-7)
    /// Q/K/V projection (combined or separate)
    QkvProjection = 2,
    /// Rotary position embedding
    RopeEmbedding = 3,
    /// Attention score computation (Q @ K^T)
    AttentionScore = 4,
    /// Attention softmax
    AttentionSoftmax = 5,
    /// Attention output (scores @ V)
    AttentionOutput = 6,
    /// Output projection after attention
    OutputProjection = 7,

    // FFN (8-11)
    /// Gate projection (for gated FFN)
    GateProjection = 8,
    /// Up projection
    UpProjection = 9,
    /// SiLU/GELU/ReLU activation
    Activation = 10,
    /// Down projection
    DownProjection = 11,

    // Other (12-14)
    /// Token embedding lookup
    Embedding = 12,
    /// Language model head (logits)
    LmHead = 13,
    /// Token sampling
    Sampling = 14,
}

impl BrickId {
    /// Number of well-known brick types.
    pub const COUNT: usize = 15;

    /// Get the category for hierarchical aggregation.
    #[inline]
    pub fn category(self) -> BrickCategory {
        match self {
            Self::RmsNorm | Self::LayerNorm => BrickCategory::Norm,
            Self::QkvProjection
            | Self::RopeEmbedding
            | Self::AttentionScore
            | Self::AttentionSoftmax
            | Self::AttentionOutput
            | Self::OutputProjection => BrickCategory::Attention,
            Self::GateProjection | Self::UpProjection | Self::Activation | Self::DownProjection => {
                BrickCategory::Ffn
            }
            Self::Embedding | Self::LmHead | Self::Sampling => BrickCategory::Other,
        }
    }

    /// Get the string name of this brick.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::RmsNorm => "RmsNorm",
            Self::LayerNorm => "LayerNorm",
            Self::QkvProjection => "QkvProjection",
            Self::RopeEmbedding => "RopeEmbedding",
            Self::AttentionScore => "AttentionScore",
            Self::AttentionSoftmax => "AttentionSoftmax",
            Self::AttentionOutput => "AttentionOutput",
            Self::OutputProjection => "OutputProjection",
            Self::GateProjection => "GateProjection",
            Self::UpProjection => "UpProjection",
            Self::Activation => "Activation",
            Self::DownProjection => "DownProjection",
            Self::Embedding => "Embedding",
            Self::LmHead => "LmHead",
            Self::Sampling => "Sampling",
        }
    }

    /// Try to parse a string into a BrickId.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "RmsNorm" => Some(Self::RmsNorm),
            "LayerNorm" => Some(Self::LayerNorm),
            "QkvProjection" | "Qkv" => Some(Self::QkvProjection),
            "RopeEmbedding" | "Rope" | "RoPE" => Some(Self::RopeEmbedding),
            "AttentionScore" => Some(Self::AttentionScore),
            "AttentionSoftmax" | "Softmax" => Some(Self::AttentionSoftmax),
            "AttentionOutput" => Some(Self::AttentionOutput),
            "OutputProjection" | "OutProj" => Some(Self::OutputProjection),
            "GateProjection" | "Gate" => Some(Self::GateProjection),
            "UpProjection" | "Up" => Some(Self::UpProjection),
            "Activation" | "SiLU" | "GELU" | "ReLU" => Some(Self::Activation),
            "DownProjection" | "Down" => Some(Self::DownProjection),
            "Embedding" | "Embed" => Some(Self::Embedding),
            "LmHead" | "Head" => Some(Self::LmHead),
            "Sampling" | "Sample" => Some(Self::Sampling),
            _ => None,
        }
    }
}

impl fmt::Display for BrickId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Category for hierarchical aggregation of brick statistics.
///
/// PAR-200: Groups related bricks for high-level performance analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum BrickCategory {
    /// Normalization layers (RmsNorm, LayerNorm)
    Norm = 0,
    /// Attention mechanism (QKV, RoPE, scores, softmax, output)
    Attention = 1,
    /// Feed-forward network (gate, up, activation, down)
    Ffn = 2,
    /// Other operations (embedding, lm_head, sampling)
    #[default]
    Other = 3,
}

impl BrickCategory {
    /// Number of categories.
    pub const COUNT: usize = 4;

    /// Get the string name of this category.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Norm => "Norm",
            Self::Attention => "Attention",
            Self::Ffn => "FFN",
            Self::Other => "Other",
        }
    }
}

impl fmt::Display for BrickCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Synchronization mode for GPU profiling.
///
/// PAR-200: Controls the trade-off between accuracy and overhead.
///
/// # Performance Characteristics
///
/// | Mode | Overhead | Accuracy | Use Case |
/// |------|----------|----------|----------|
/// | `Immediate` | ~200% | Exact per-kernel | Debugging |
/// | `PerLayer` | ~20% | Per-layer exact | Development |
/// | `Deferred` | ~5% | Approximate | Production |
/// | `None` | 0% | N/A | Disabled |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncMode {
    /// Sync after each kernel (accurate but slow).
    /// Best for debugging and detailed optimization.
    Immediate,
    /// Sync once per transformer layer.
    /// Good balance for development.
    PerLayer,
    /// Sync once per forward pass (fast, approximate).
    /// Best for production profiling.
    #[default]
    Deferred,
    /// No synchronization (profiling disabled or CPU-only).
    None,
}

// ============================================================================
// PAR-201: Execution Path Graph Types
// ============================================================================

/// Node ID in the execution graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionNodeId(pub u32);

/// Execution graph node types.
///
/// PAR-201: Represents different levels of the execution hierarchy.
#[derive(Debug, Clone)]
pub enum ExecutionNode {
    /// High-level brick (BrickId from v2)
    Brick {
        id: BrickId,
        timing_ns: u64,
        elements: u64,
    },
    /// GPU kernel launch
    Kernel {
        name: String,
        /// FNV-1a hash of PTX source for identity
        ptx_hash: u64,
        /// Grid dimensions (blocks)
        grid: (u32, u32, u32),
        /// Block dimensions (threads)
        block: (u32, u32, u32),
        /// Shared memory bytes
        shared_mem: u32,
        /// Kernel execution time in nanoseconds (Phase 9: for CPA)
        timing_ns: Option<u64>,
        /// Arithmetic intensity (FLOPs/byte) for roofline analysis (Phase 9)
        arithmetic_intensity: Option<f32>,
        /// Achieved throughput in TFLOP/s (Phase 9)
        achieved_tflops: Option<f32>,
    },
    /// Memory transfer operation (Phase 9: data movement topology)
    Transfer {
        /// Source location description
        src: String,
        /// Destination location description
        dst: String,
        /// Bytes transferred
        bytes: u64,
        /// Transfer direction
        direction: TransferDirection,
        /// Transfer time in nanoseconds
        timing_ns: Option<u64>,
    },
    /// Rust function (from DWARF or manual annotation)
    Function {
        name: String,
        file: Option<String>,
        line: Option<u32>,
    },
    /// Transformer layer grouping
    Layer { index: u32 },
    /// Phase 11 (E.9.4): Async task metrics for poll efficiency tracking
    AsyncTask {
        /// Task name for identification
        name: String,
        /// Number of times poll() was called
        poll_count: u64,
        /// Number of times poll() returned Pending
        yield_count: u64,
        /// Total time spent in poll() (nanoseconds)
        total_poll_ns: u64,
    },
}

impl ExecutionNode {
    /// Get the display name of this node.
    pub fn name(&self) -> String {
        match self {
            Self::Brick { id, .. } => id.name().to_string(),
            Self::Kernel { name, .. } => name.clone(),
            Self::Function { name, .. } => name.clone(),
            Self::Layer { index } => format!("Layer{}", index),
            Self::Transfer {
                src,
                dst,
                direction,
                ..
            } => {
                let dir = match direction {
                    TransferDirection::H2D => "H2D",
                    TransferDirection::D2H => "D2H",
                    TransferDirection::D2D => "D2D",
                };
                format!("{}:{}->{}", dir, src, dst)
            }
            Self::AsyncTask { name, .. } => name.clone(),
        }
    }

    /// Check if this is a kernel node.
    pub fn is_kernel(&self) -> bool {
        matches!(self, Self::Kernel { .. })
    }

    /// Check if this is a brick node.
    pub fn is_brick(&self) -> bool {
        matches!(self, Self::Brick { .. })
    }

    /// Check if this is a transfer node.
    pub fn is_transfer(&self) -> bool {
        matches!(self, Self::Transfer { .. })
    }

    /// Get timing if available (bricks, kernels, and transfers).
    pub fn timing_ns(&self) -> Option<u64> {
        match self {
            Self::Brick { timing_ns, .. } => Some(*timing_ns),
            Self::Kernel { timing_ns, .. } => *timing_ns,
            Self::Transfer { timing_ns, .. } => *timing_ns,
            _ => None,
        }
    }

    /// Get PTX hash if available (kernels only).
    pub fn ptx_hash(&self) -> Option<u64> {
        match self {
            Self::Kernel { ptx_hash, .. } => Some(*ptx_hash),
            _ => None,
        }
    }

    /// Get arithmetic intensity if available (kernels only, Phase 9).
    pub fn arithmetic_intensity(&self) -> Option<f32> {
        match self {
            Self::Kernel {
                arithmetic_intensity,
                ..
            } => *arithmetic_intensity,
            _ => None,
        }
    }

    /// Get achieved TFLOP/s if available (kernels only, Phase 9).
    pub fn achieved_tflops(&self) -> Option<f32> {
        match self {
            Self::Kernel {
                achieved_tflops, ..
            } => *achieved_tflops,
            _ => None,
        }
    }

    /// Get transfer bytes if available (transfers only, Phase 9).
    pub fn transfer_bytes(&self) -> Option<u64> {
        match self {
            Self::Transfer { bytes, .. } => Some(*bytes),
            _ => None,
        }
    }
}

/// Edge types in execution graph.
///
/// PAR-201: Describes relationships between execution nodes.
/// Phase 9 (E.7.12): Added DependsOn and Transfer for advanced profiling.
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeType {
    /// Function calls function
    Calls,
    /// Brick contains sub-operations
    Contains,
    /// Function launches GPU kernel
    Launches,
    /// Temporal sequence (A happens before B)
    Sequence,
    /// Dependency edge for critical path analysis (CUDA events, stream sync)
    /// PAR-201 Phase 9: CPA requires tracking true dependencies vs containment
    DependsOn,
    /// Data transfer edge with byte count (H2D/D2H/D2D)
    /// PAR-201 Phase 9: For data movement topology and ping-pong detection
    Transfer {
        /// Bytes transferred
        bytes: u64,
        /// Transfer direction
        direction: TransferDirection,
    },
}

/// Direction of memory transfer.
///
/// PAR-201 Phase 9: Used with EdgeType::Transfer for data movement analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host to Device
    H2D,
    /// Device to Host
    D2H,
    /// Device to Device
    D2D,
}

/// An edge in the execution graph.
#[derive(Debug, Clone)]
pub struct ExecutionEdge {
    /// Source node ID
    pub src: ExecutionNodeId,
    /// Destination node ID
    pub dst: ExecutionNodeId,
    /// Edge type
    pub edge_type: EdgeType,
    /// Optional weight (e.g., call count, timing)
    pub weight: f32,
}

// ============================================================================
// PTX Registry and Statistics Types
// ============================================================================

/// PTX kernel registry for execution graph correlation.
///
/// PAR-201: Maps PTX hashes to source code for debugging and analysis.
#[derive(Debug, Default)]
pub struct PtxRegistry {
    /// Hash → (kernel_name, ptx_source, file_path)
    kernels: HashMap<u64, (String, String, Option<std::path::PathBuf>)>,
}

impl PtxRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register PTX source code.
    ///
    /// # Arguments
    /// - `name`: Kernel name (e.g., "batched_q4k_gemv")
    /// - `ptx`: PTX source code
    /// - `path`: Optional file path for source correlation
    pub fn register(&mut self, name: &str, ptx: &str, path: Option<&std::path::Path>) {
        debug_assert!(!name.is_empty(), "CB-BUDGET: kernel name must not be empty");
        debug_assert!(!ptx.is_empty(), "CB-BUDGET: PTX source must not be empty");
        let hash = Self::hash_ptx(ptx);
        self.kernels.insert(
            hash,
            (
                name.to_string(),
                ptx.to_string(),
                path.map(|p| p.to_path_buf()),
            ),
        );
    }

    /// Compute FNV-1a hash of PTX source.
    #[inline]
    pub fn hash_ptx(ptx: &str) -> u64 {
        // FNV-1a hash
        let mut hash: u64 = 0xcbf29ce484222325;
        for byte in ptx.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Lookup PTX source by hash.
    pub fn lookup(&self, hash: u64) -> Option<&str> {
        self.kernels.get(&hash).map(|(_, ptx, _)| ptx.as_str())
    }

    /// Lookup kernel name by hash.
    pub fn lookup_name(&self, hash: u64) -> Option<&str> {
        self.kernels.get(&hash).map(|(name, _, _)| name.as_str())
    }

    /// Lookup file path by hash.
    pub fn lookup_path(&self, hash: u64) -> Option<&std::path::Path> {
        self.kernels
            .get(&hash)
            .and_then(|(_, _, path)| path.as_deref())
    }

    /// Get all registered hashes.
    pub fn hashes(&self) -> impl Iterator<Item = u64> + '_ {
        self.kernels.keys().copied()
    }

    /// Number of registered kernels.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Check if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }
}

/// Aggregated statistics for a brick category.
#[derive(Debug, Clone, Copy, Default)]
pub struct CategoryStats {
    /// Total elapsed time (nanoseconds)
    pub total_ns: u64,
    /// Total elements processed
    pub total_elements: u64,
    /// Total samples
    pub count: u64,
}

impl CategoryStats {
    /// Average time per sample in microseconds.
    #[inline]
    pub fn avg_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_ns as f64 / self.count as f64 / 1000.0
        }
    }

    /// Throughput in elements per second.
    #[inline]
    pub fn throughput(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.total_elements as f64 / (self.total_ns as f64 / 1_000_000_000.0)
        }
    }

    /// Percentage of total time (given total_ns across all categories).
    #[inline]
    pub fn percentage(&self, total: u64) -> f64 {
        if total == 0 {
            0.0
        } else {
            100.0 * self.total_ns as f64 / total as f64
        }
    }
}

/// Accumulated per-brick statistics.
#[derive(Debug, Clone, Default)]
pub struct BrickStats {
    /// Brick name
    pub name: String,
    /// Total samples collected
    pub count: u64,
    /// Total elapsed time (nanoseconds)
    pub total_ns: u64,
    /// Min elapsed time (nanoseconds)
    pub min_ns: u64,
    /// Max elapsed time (nanoseconds)
    pub max_ns: u64,
    /// Total elements processed
    pub total_elements: u64,
    /// PMAT-451: Total bytes processed (for throughput calculation)
    pub total_bytes: u64,
    /// PMAT-451: Total compressed bytes (for compression ratio)
    pub total_compressed_bytes: u64,
    /// PMAT-451: Bottleneck classification
    pub bottleneck: BrickBottleneck,
    /// Phase 11 (E.9.2): Total CPU cycles (from RDTSCP/CNTVCT)
    pub total_cycles: u64,
    /// Phase 11: Minimum CPU cycles observed
    pub min_cycles: u64,
    /// Phase 11: Maximum CPU cycles observed
    pub max_cycles: u64,
}

impl BrickStats {
    /// Create new stats for a brick.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            count: 0,
            total_ns: 0,
            min_ns: u64::MAX,
            max_ns: 0,
            total_elements: 0,
            total_bytes: 0,
            total_compressed_bytes: 0,
            bottleneck: BrickBottleneck::Unknown,
            total_cycles: 0,
            min_cycles: u64::MAX,
            max_cycles: 0,
        }
    }

    /// Add a sample to statistics.
    pub fn add_sample(&mut self, elapsed_ns: u64, elements: u64) {
        debug_assert!(elements > 0, "CB-BUDGET: elements must be > 0");
        self.count += 1;
        self.total_ns += elapsed_ns;
        self.min_ns = self.min_ns.min(elapsed_ns);
        self.max_ns = self.max_ns.max(elapsed_ns);
        self.total_elements += elements;
    }

    /// Phase 11 (E.9.2): Add a sample with CPU cycle count.
    ///
    /// Use this for frequency-invariant performance analysis.
    /// Cycles are immune to CPU frequency scaling (turbo boost).
    pub fn add_sample_with_cycles(&mut self, elapsed_ns: u64, elements: u64, cycles: u64) {
        self.add_sample(elapsed_ns, elements);
        self.total_cycles += cycles;
        self.min_cycles = self.min_cycles.min(cycles);
        self.max_cycles = self.max_cycles.max(cycles);
    }

    /// Phase 11: Cycles per element (frequency-invariant throughput metric).
    ///
    /// Lower is better. This metric is immune to CPU frequency scaling.
    #[must_use]
    pub fn cycles_per_element(&self) -> f64 {
        if self.total_elements == 0 {
            0.0
        } else {
            self.total_cycles as f64 / self.total_elements as f64
        }
    }

    /// Phase 11: Average cycles per sample.
    #[must_use]
    pub fn avg_cycles(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_cycles as f64 / self.count as f64
        }
    }

    /// Phase 11: Estimated IPC (Instructions Per Cycle).
    ///
    /// Approximation assuming ~1 instruction per element for simple ops.
    /// - Low IPC (<1.0): Memory stalls (cache misses, memory latency)
    /// - High IPC (>2.0): Compute bound (efficient execution)
    #[must_use]
    pub fn estimated_ipc(&self) -> f64 {
        if self.total_cycles == 0 {
            0.0
        } else {
            // Rough approximation: assume 1 instruction per element
            self.total_elements as f64 / self.total_cycles as f64
        }
    }

    /// Phase 11: Diagnose bottleneck based on cycles vs time ratio.
    ///
    /// High cycles + low time = likely cache misses
    /// Low cycles + high time = likely CPU throttling or context switches
    #[must_use]
    pub fn diagnose_from_cycles(&self) -> &'static str {
        if self.total_cycles == 0 || self.total_ns == 0 {
            return "insufficient data";
        }

        let ipc = self.estimated_ipc();
        let ns_per_cycle = self.total_ns as f64 / self.total_cycles as f64;

        // Typical CPU runs at ~3GHz, so 1 cycle ≈ 0.33ns
        // If ns_per_cycle >> 0.33, we're seeing stalls or throttling
        if ipc < 0.5 {
            "memory-bound (low IPC, likely cache misses)"
        } else if ipc > 2.0 {
            "compute-bound (efficient)"
        } else if ns_per_cycle > 1.0 {
            "throttled or context-switched"
        } else {
            "balanced"
        }
    }

    /// PMAT-451: Add a sample with byte metrics for compression workloads.
    ///
    /// # Arguments
    /// - `elapsed_ns`: Time taken in nanoseconds
    /// - `elements`: Number of elements processed (e.g., pages)
    /// - `input_bytes`: Original uncompressed size
    /// - `output_bytes`: Compressed output size
    pub fn add_sample_with_bytes(
        &mut self,
        elapsed_ns: u64,
        elements: u64,
        input_bytes: u64,
        output_bytes: u64,
    ) {
        self.add_sample(elapsed_ns, elements);
        self.total_bytes += input_bytes;
        self.total_compressed_bytes += output_bytes;
    }

    /// PMAT-451: Calculate compression ratio (input_size / output_size).
    /// Returns 1.0 if no compression data available.
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.total_compressed_bytes == 0 {
            1.0
        } else {
            self.total_bytes as f64 / self.total_compressed_bytes as f64
        }
    }

    /// PMAT-451: Calculate throughput in GB/s.
    /// Based on total input bytes processed.
    #[must_use]
    pub fn throughput_gbps(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            let bytes_per_ns = self.total_bytes as f64 / self.total_ns as f64;
            bytes_per_ns * 1e9 / 1e9 // Convert to GB/s (ns to sec, bytes to GB)
        }
    }

    /// PMAT-451: Set bottleneck classification.
    pub fn set_bottleneck(&mut self, bottleneck: BrickBottleneck) {
        self.bottleneck = bottleneck;
    }

    /// PMAT-451: Get bottleneck classification.
    #[must_use]
    pub fn get_bottleneck(&self) -> BrickBottleneck {
        self.bottleneck
    }

    /// Average time in microseconds.
    #[must_use]
    pub fn avg_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_ns as f64 / self.count as f64 / 1000.0
        }
    }

    /// Throughput in elements/second.
    #[must_use]
    pub fn throughput(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.total_elements as f64 / (self.total_ns as f64 / 1_000_000_000.0)
        }
    }

    /// Throughput in tokens/second (alias for throughput).
    #[must_use]
    pub fn tokens_per_sec(&self) -> f64 {
        self.throughput()
    }

    /// Minimum time in microseconds.
    #[must_use]
    pub fn min_us(&self) -> f64 {
        if self.min_ns == u64::MAX {
            0.0
        } else {
            self.min_ns as f64 / 1000.0
        }
    }

    /// Maximum time in microseconds.
    #[must_use]
    pub fn max_us(&self) -> f64 {
        self.max_ns as f64 / 1000.0
    }
}
