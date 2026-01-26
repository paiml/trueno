//! Grammar of ComputeBlock (§32)
//!
//! A declarative, composable framework for specifying compute workloads,
//! inspired by Wilkinson's Grammar of Graphics (2005).
//!
//! # Conceptual Foundation
//!
//! Just as the Grammar of Graphics decomposes visualization:
//! ```text
//! Data + Aesthetics + Geometry + Statistics + Scales + Coordinates + Facets + Theme → Visualization
//! ```
//!
//! The Grammar of ComputeBlock decomposes computation:
//! ```text
//! Workload + Resources + Strategy + Transform + Scales + Context + Composition + Policy → Execution
//! ```
//!
//! # Example
//!
//! ```ignore
//! use cbtop::grammar::*;
//!
//! let result = ComputeBlock::builder()
//!     .workload(Workload::matmul(1024, 1024, 1024))
//!     .strategy(Strategy::Gpu(GpuDevice::Auto))
//!     .strategy(Strategy::Simd(SimdWidth::Avx2))  // Fallback
//!     .transform(Transform::Tile { tile_size: 64 })
//!     .policy(Policy::realtime())
//!     .build()?
//!     .execute()?;
//! ```
//!
//! # References
//!
//! - [Wilkinson 2005] "The Grammar of Graphics" Springer
//! - [Wickham 2010] "A Layered Grammar of Graphics" JCGS
//! - [Halide 2013] "Halide: Optimizing Parallelism" PLDI
//! - [TVM 2018] "TVM: End-to-End Optimizing Compiler" OSDI

use std::collections::HashMap;
use std::time::Duration;

/// Result type for grammar operations
pub type GrammarResult<T> = Result<T, GrammarError>;

/// Error types for grammar operations
#[derive(Debug, Clone, PartialEq)]
pub enum GrammarError {
    /// Missing required workload specification
    MissingWorkload,
    /// Invalid dimensions (zero or negative)
    InvalidDimensions(String),
    /// Invalid scale domain (min >= max)
    InvalidScaleDomain { min: f64, max: f64 },
    /// Device not found
    DeviceNotFound(u32),
    /// Execution timeout
    Timeout(Duration),
    /// Strategy not supported on current hardware
    UnsupportedStrategy(String),
    /// Validation error
    ValidationError(String),
    /// Execution error
    ExecutionError(String),
}

impl std::fmt::Display for GrammarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GrammarError::MissingWorkload => write!(f, "Missing required workload specification"),
            GrammarError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            GrammarError::InvalidScaleDomain { min, max } => {
                write!(f, "Invalid scale domain: min {} >= max {}", min, max)
            }
            GrammarError::DeviceNotFound(id) => write!(f, "Device {} not found", id),
            GrammarError::Timeout(d) => write!(f, "Execution timeout after {:?}", d),
            GrammarError::UnsupportedStrategy(s) => write!(f, "Unsupported strategy: {}", s),
            GrammarError::ValidationError(s) => write!(f, "Validation error: {}", s),
            GrammarError::ExecutionError(s) => write!(f, "Execution error: {}", s),
        }
    }
}

impl std::error::Error for GrammarError {}

// ============================================================================
// Workload Specification (Data equivalent)
// ============================================================================

/// Compute operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Element-wise operations
    Elementwise,
    /// Dot product
    Dot,
    /// Matrix multiplication
    Matmul,
    /// 2D convolution
    Conv2d,
    /// Multi-head attention
    Attention,
    /// Softmax
    Softmax,
    /// Layer normalization
    LayerNorm,
    /// Feed-forward network
    Ffn,
    /// Reduction (sum, mean, max, etc.)
    Reduce,
    /// Custom operation
    Custom(u32),
}

/// Data type for compute operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point
    F16,
    /// Brain float 16
    Bf16,
    /// 8-bit integer
    I8,
    /// Unsigned 8-bit integer
    U8,
    /// 4-bit quantized (packed)
    Q4,
}

impl DataType {
    /// Get byte size of data type
    pub fn byte_size(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 | DataType::Bf16 => 2,
            DataType::I8 | DataType::U8 => 1,
            DataType::Q4 => 1, // 2 values per byte
        }
    }
}

/// Problem dimensions
#[derive(Debug, Clone, PartialEq)]
pub struct Dimensions {
    /// Batch size
    pub batch: usize,
    /// Sequence length (for attention)
    pub seq_len: usize,
    /// Number of heads (for attention)
    pub num_heads: usize,
    /// Head dimension (for attention)
    pub head_dim: usize,
    /// Hidden dimension (for FFN)
    pub hidden_dim: usize,
    /// M dimension (for matmul)
    pub m: usize,
    /// N dimension (for matmul)
    pub n: usize,
    /// K dimension (for matmul)
    pub k: usize,
}

impl Default for Dimensions {
    fn default() -> Self {
        Self {
            batch: 1,
            seq_len: 1,
            num_heads: 1,
            head_dim: 64,
            hidden_dim: 1,
            m: 1,
            n: 1,
            k: 1,
        }
    }
}

impl Dimensions {
    /// Create dimensions for vector operation
    pub fn vector(size: usize) -> Self {
        Self {
            n: size,
            ..Default::default()
        }
    }

    /// Create dimensions for matrix multiplication
    pub fn matmul(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            ..Default::default()
        }
    }

    /// Create dimensions for attention
    pub fn attention(batch: usize, seq_len: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            batch,
            seq_len,
            num_heads,
            head_dim,
            ..Default::default()
        }
    }
}

/// Tensor specification
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSpec {
    /// Tensor name/identifier
    pub name: String,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DataType,
    /// Stride (optional, for non-contiguous tensors)
    pub stride: Option<Vec<usize>>,
}

impl TensorSpec {
    /// Create a new tensor spec
    pub fn new(name: impl Into<String>, shape: Vec<usize>, dtype: DataType) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype,
            stride: None,
        }
    }

    /// Total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total byte size
    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.byte_size()
    }
}

/// Workload specification (analogous to DataFrame)
#[derive(Debug, Clone, PartialEq)]
pub struct WorkloadSpec {
    /// Operation type
    pub operation: Operation,
    /// Problem dimensions
    pub dimensions: Dimensions,
    /// Primary data type
    pub dtype: DataType,
    /// Input tensor specifications
    pub inputs: Vec<TensorSpec>,
    /// Output tensor specifications
    pub outputs: Vec<TensorSpec>,
}

impl WorkloadSpec {
    /// Create a dot product workload
    pub fn dot(size: usize) -> Self {
        Self {
            operation: Operation::Dot,
            dimensions: Dimensions::vector(size),
            dtype: DataType::F32,
            inputs: vec![
                TensorSpec::new("a", vec![size], DataType::F32),
                TensorSpec::new("b", vec![size], DataType::F32),
            ],
            outputs: vec![TensorSpec::new("result", vec![1], DataType::F32)],
        }
    }

    /// Create a matrix multiplication workload
    pub fn matmul(m: usize, n: usize, k: usize) -> Self {
        Self {
            operation: Operation::Matmul,
            dimensions: Dimensions::matmul(m, n, k),
            dtype: DataType::F32,
            inputs: vec![
                TensorSpec::new("a", vec![m, k], DataType::F32),
                TensorSpec::new("b", vec![k, n], DataType::F32),
            ],
            outputs: vec![TensorSpec::new("c", vec![m, n], DataType::F32)],
        }
    }

    /// Create an attention workload
    pub fn attention(batch: usize, seq_len: usize, num_heads: usize, head_dim: usize) -> Self {
        let embed_dim = num_heads * head_dim;
        Self {
            operation: Operation::Attention,
            dimensions: Dimensions::attention(batch, seq_len, num_heads, head_dim),
            dtype: DataType::F32,
            inputs: vec![
                TensorSpec::new("q", vec![batch, seq_len, embed_dim], DataType::F32),
                TensorSpec::new("k", vec![batch, seq_len, embed_dim], DataType::F32),
                TensorSpec::new("v", vec![batch, seq_len, embed_dim], DataType::F32),
            ],
            outputs: vec![TensorSpec::new(
                "out",
                vec![batch, seq_len, embed_dim],
                DataType::F32,
            )],
        }
    }

    /// Create an elementwise workload
    pub fn elementwise(size: usize) -> Self {
        Self {
            operation: Operation::Elementwise,
            dimensions: Dimensions::vector(size),
            dtype: DataType::F32,
            inputs: vec![TensorSpec::new("input", vec![size], DataType::F32)],
            outputs: vec![TensorSpec::new("output", vec![size], DataType::F32)],
        }
    }

    /// Total FLOP count estimate
    pub fn flop_count(&self) -> usize {
        match self.operation {
            Operation::Dot => self.dimensions.n * 2, // mul + add
            Operation::Matmul => self.dimensions.m * self.dimensions.n * self.dimensions.k * 2,
            Operation::Attention => {
                let b = self.dimensions.batch;
                let s = self.dimensions.seq_len;
                let h = self.dimensions.num_heads;
                let d = self.dimensions.head_dim;
                // QK^T + softmax + AV
                b * h * (s * s * d * 2 + s * s + s * s * d * 2)
            }
            // Default estimate for remaining operations
            Operation::Elementwise
            | Operation::Conv2d
            | Operation::Softmax
            | Operation::LayerNorm
            | Operation::Ffn
            | Operation::Reduce
            | Operation::Custom(_) => self.dimensions.n,
        }
    }
}

// ============================================================================
// Resource Mapping (Aesthetics equivalent)
// ============================================================================

/// Scale binding for resource mapping
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleBinding {
    /// Bind to problem size
    ProblemSize,
    /// Bind to data volume
    DataVolume,
    /// Bind to throughput requirement
    Throughput,
    /// Custom binding expression
    Custom(String),
}

/// Byte size helper
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteSize(pub usize);

impl ByteSize {
    /// Create from megabytes
    pub fn mb(mb: usize) -> Self {
        ByteSize(mb * 1024 * 1024)
    }

    /// Create from gigabytes
    pub fn gb(gb: usize) -> Self {
        ByteSize(gb * 1024 * 1024 * 1024)
    }

    /// Get raw bytes
    pub fn bytes(&self) -> usize {
        self.0
    }
}

/// Resource mapping (analogous to Aesthetics)
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResourceMapping {
    /// Map problem size to cores
    pub cores: Option<ScaleBinding>,
    /// Map data volume to memory
    pub memory: Option<ScaleBinding>,
    /// Map throughput to bandwidth
    pub bandwidth: Option<ScaleBinding>,
    /// Map latency constraints
    pub latency: Option<ScaleBinding>,
    /// Fixed core count override
    pub cores_value: Option<usize>,
    /// Fixed memory limit override
    pub memory_value: Option<ByteSize>,
}

impl ResourceMapping {
    /// Create empty resource mapping
    pub fn new() -> Self {
        Self::default()
    }

    /// Set core binding
    pub fn cores(mut self, binding: ScaleBinding) -> Self {
        self.cores = Some(binding);
        self
    }

    /// Set fixed core count
    pub fn cores_value(mut self, count: usize) -> Self {
        self.cores_value = Some(count);
        self
    }

    /// Set memory binding
    pub fn memory(mut self, binding: ScaleBinding) -> Self {
        self.memory = Some(binding);
        self
    }

    /// Set fixed memory limit
    pub fn memory_value(mut self, size: ByteSize) -> Self {
        self.memory_value = Some(size);
        self
    }
}

// ============================================================================
// Execution Strategy (Geometry equivalent)
// ============================================================================

/// SIMD width specification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdWidth {
    /// Auto-detect best available
    Auto,
    /// SSE2 (128-bit)
    Sse2,
    /// AVX2 (256-bit)
    Avx2,
    /// AVX-512 (512-bit)
    Avx512,
    /// ARM NEON (128-bit)
    Neon,
    /// WASM SIMD128
    Wasm,
}

/// GPU device specification
#[derive(Debug, Clone, PartialEq)]
pub enum GpuDevice {
    /// Auto-select best available
    Auto,
    /// Specific device by ID
    Id(u32),
    /// CUDA device
    Cuda(u32),
    /// wgpu device
    Wgpu(u32),
}

/// Kernel specification for GPU
#[derive(Debug, Clone, PartialEq)]
pub struct KernelSpec {
    /// Kernel name
    pub name: String,
    /// Block size (threads per block)
    pub block_size: (u32, u32, u32),
    /// Grid size (number of blocks)
    pub grid_size: Option<(u32, u32, u32)>,
    /// Shared memory per block
    pub shared_mem: usize,
}

/// Execution strategy (analogous to Geometry)
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionStrategy {
    /// Sequential execution (baseline)
    Sequential,
    /// SIMD vectorization
    Simd { width: SimdWidth },
    /// Multi-threaded parallel
    Parallel { threads: usize, chunk_size: usize },
    /// GPU acceleration
    Gpu {
        device: GpuDevice,
        kernel: Option<KernelSpec>,
    },
    /// Distributed across nodes
    Distributed { nodes: Vec<String> },
    /// Hybrid CPU+GPU
    Hybrid { cpu_fraction: f64 },
}

impl ExecutionStrategy {
    /// Create SIMD strategy with auto width
    pub fn simd_auto() -> Self {
        ExecutionStrategy::Simd {
            width: SimdWidth::Auto,
        }
    }

    /// Create SIMD strategy with specific width
    pub fn simd(width: SimdWidth) -> Self {
        ExecutionStrategy::Simd { width }
    }

    /// Create parallel strategy
    pub fn parallel(threads: usize) -> Self {
        ExecutionStrategy::Parallel {
            threads,
            chunk_size: 1024,
        }
    }

    /// Create GPU strategy with auto device
    pub fn gpu_auto() -> Self {
        ExecutionStrategy::Gpu {
            device: GpuDevice::Auto,
            kernel: None,
        }
    }

    /// Create GPU strategy with specific device
    pub fn gpu(device: GpuDevice) -> Self {
        ExecutionStrategy::Gpu {
            device,
            kernel: None,
        }
    }
}

// ============================================================================
// Data Transform (Statistics equivalent)
// ============================================================================

/// Quantization scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// Symmetric quantization
    Symmetric,
    /// Asymmetric quantization
    Asymmetric,
    /// Block-wise quantization (GGML-style)
    BlockWise { block_size: usize },
}

/// Data transform (analogous to Statistics)
#[derive(Debug, Clone, PartialEq, Default)]
pub enum DataTransform {
    /// No transformation
    #[default]
    Identity,
    /// Quantize to lower precision
    Quantize { bits: u8, scheme: QuantScheme },
    /// Tile for cache efficiency
    Tile { tile_size: usize },
    /// Transpose for memory layout
    Transpose { order: Vec<usize> },
    /// Pad for alignment
    Pad { alignment: usize },
    /// Fuse multiple operations
    Fuse { ops: Vec<Operation> },
}

impl DataTransform {
    /// Create identity transform
    pub fn identity() -> Self {
        DataTransform::Identity
    }

    /// Create tiling transform
    pub fn tile(size: usize) -> Self {
        DataTransform::Tile { tile_size: size }
    }

    /// Create quantization transform
    pub fn quantize(bits: u8) -> Self {
        DataTransform::Quantize {
            bits,
            scheme: QuantScheme::Symmetric,
        }
    }

    /// Create padding transform
    pub fn pad(alignment: usize) -> Self {
        DataTransform::Pad { alignment }
    }
}

// ============================================================================
// Execution Context (Coordinates equivalent)
// ============================================================================

/// CPU affinity specification
#[derive(Debug, Clone, PartialEq)]
pub struct CpuAffinity {
    /// List of CPU cores to use
    pub cores: Vec<usize>,
}

/// Execution context (analogous to Coordinates)
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionContext {
    /// Local CPU execution
    Cpu {
        affinity: Option<CpuAffinity>,
        numa_node: Option<usize>,
    },
    /// GPU execution
    Gpu { device_id: u32, stream: Option<u32> },
    /// Heterogeneous (multiple contexts)
    Heterogeneous { contexts: Vec<ExecutionContext> },
}

impl Default for ExecutionContext {
    fn default() -> Self {
        ExecutionContext::Cpu {
            affinity: None,
            numa_node: None,
        }
    }
}

impl ExecutionContext {
    /// Create CPU context
    pub fn cpu() -> Self {
        ExecutionContext::Cpu {
            affinity: None,
            numa_node: None,
        }
    }

    /// Create GPU context
    pub fn gpu(device_id: u32) -> Self {
        ExecutionContext::Gpu {
            device_id,
            stream: None,
        }
    }
}

// ============================================================================
// Composition Mode (Facets equivalent)
// ============================================================================

/// Composition mode (analogous to Facets)
#[derive(Debug, Clone, PartialEq, Default)]
pub enum CompositionMode {
    /// Single execution
    #[default]
    None,
    /// Data parallelism (same op, different data)
    DataParallel { shards: usize },
    /// Model parallelism (different ops, same data)
    ModelParallel { stages: usize },
    /// Pipeline parallelism
    Pipeline { depth: usize, overlap: bool },
    /// Batch processing
    Batch { batch_size: usize, prefetch: usize },
}

impl CompositionMode {
    /// Create data parallel mode
    pub fn data_parallel(shards: usize) -> Self {
        CompositionMode::DataParallel { shards }
    }

    /// Create batch mode
    pub fn batch(size: usize) -> Self {
        CompositionMode::Batch {
            batch_size: size,
            prefetch: 2,
        }
    }

    /// Create pipeline mode
    pub fn pipeline(depth: usize) -> Self {
        CompositionMode::Pipeline {
            depth,
            overlap: true,
        }
    }
}

// ============================================================================
// Execution Policy (Theme equivalent)
// ============================================================================

/// Quality of Service level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QosLevel {
    /// Best effort (no guarantees)
    BestEffort,
    /// Background (lowest priority)
    Background,
    /// Interactive (balanced)
    Interactive,
    /// Realtime (highest priority)
    Realtime,
}

/// Retry policy
#[derive(Debug, Clone, PartialEq)]
pub struct RetryPolicy {
    /// Maximum retry attempts
    pub max_retries: usize,
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(100),
            backoff_multiplier: 2.0,
        }
    }
}

/// Resource limits
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResourceLimits {
    /// Maximum memory usage
    pub max_memory: Option<ByteSize>,
    /// Maximum CPU cores
    pub max_cores: Option<usize>,
    /// Maximum GPU memory
    pub max_gpu_memory: Option<ByteSize>,
}

/// Observability configuration
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ObservabilityConfig {
    /// Enable tracing
    pub tracing: bool,
    /// Enable metrics
    pub metrics: bool,
    /// Sampling rate (0.0-1.0)
    pub sampling_rate: f64,
}

/// Execution policy (analogous to Theme)
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionPolicy {
    /// Quality of Service level
    pub qos: QosLevel,
    /// Preemption allowed
    pub preemptible: bool,
    /// Timeout constraint
    pub timeout: Option<Duration>,
    /// Retry policy
    pub retry: RetryPolicy,
    /// Resource limits
    pub limits: ResourceLimits,
    /// Observability config
    pub observability: ObservabilityConfig,
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self {
            qos: QosLevel::Interactive,
            preemptible: true,
            timeout: None,
            retry: RetryPolicy::default(),
            limits: ResourceLimits::default(),
            observability: ObservabilityConfig::default(),
        }
    }
}

impl ExecutionPolicy {
    /// Create realtime policy (low latency, non-preemptible)
    pub fn realtime() -> Self {
        Self {
            qos: QosLevel::Realtime,
            preemptible: false,
            timeout: Some(Duration::from_millis(100)),
            ..Default::default()
        }
    }

    /// Create batch policy (high throughput, preemptible)
    pub fn batch() -> Self {
        Self {
            qos: QosLevel::BestEffort,
            preemptible: true,
            timeout: None,
            ..Default::default()
        }
    }

    /// Create interactive policy (balanced)
    pub fn interactive() -> Self {
        Self::default()
    }

    /// Create debug policy (full tracing, relaxed limits)
    pub fn debug() -> Self {
        Self {
            qos: QosLevel::BestEffort,
            preemptible: true,
            timeout: None,
            observability: ObservabilityConfig {
                tracing: true,
                metrics: true,
                sampling_rate: 1.0,
            },
            ..Default::default()
        }
    }
}

// ============================================================================
// Strategy Layer (Layer equivalent)
// ============================================================================

/// Strategy layer (analogous to ggplot2 Layer)
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyLayer {
    /// Execution strategy
    pub strategy: ExecutionStrategy,
    /// Layer-specific workload override
    pub workload: Option<WorkloadSpec>,
    /// Layer-specific resource mapping
    pub resources: ResourceMapping,
    /// Layer priority (higher = try first)
    pub priority: i32,
}

impl StrategyLayer {
    /// Create new strategy layer
    pub fn new(strategy: ExecutionStrategy) -> Self {
        Self {
            strategy,
            workload: None,
            resources: ResourceMapping::default(),
            priority: 0,
        }
    }

    /// Set layer priority
    pub fn priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

// ============================================================================
// ComputeBlock (GGPlot equivalent)
// ============================================================================

/// Execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Execution time
    pub duration: Duration,
    /// GFLOP/s achieved
    pub gflops: f64,
    /// Memory bandwidth achieved (GB/s)
    pub bandwidth_gbps: f64,
    /// Strategy that was used
    pub strategy_used: String,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Validated ComputeBlock ready for execution
#[derive(Debug, Clone)]
pub struct BuiltComputeBlock {
    inner: ComputeBlock,
}

impl BuiltComputeBlock {
    /// Execute the compute block
    pub fn execute(&self) -> GrammarResult<ExecutionResult> {
        let start = std::time::Instant::now();

        // Select strategy (in order of priority)
        let mut strategies = self.inner.strategies.clone();
        strategies.sort_by(|a, b| b.priority.cmp(&a.priority));

        let strategy_used = if let Some(layer) = strategies.first() {
            format!("{:?}", layer.strategy)
        } else {
            "Sequential".to_string()
        };

        // Simulate execution (real implementation would dispatch to backends)
        let duration = start.elapsed();
        let flops = self
            .inner
            .workload
            .as_ref()
            .map(|w| w.flop_count())
            .unwrap_or(0);
        let gflops = if duration.as_secs_f64() > 0.0 {
            flops as f64 / duration.as_secs_f64() / 1e9
        } else {
            0.0
        };

        Ok(ExecutionResult {
            duration,
            gflops,
            bandwidth_gbps: 0.0,
            strategy_used,
            metrics: HashMap::new(),
        })
    }

    /// Get the workload spec
    pub fn workload(&self) -> Option<&WorkloadSpec> {
        self.inner.workload.as_ref()
    }
}

/// ComputeBlock - the main orchestrator (analogous to GGPlot)
#[derive(Debug, Clone, Default)]
pub struct ComputeBlock {
    /// Workload specification
    workload: Option<WorkloadSpec>,
    /// Resource mapping
    resources: ResourceMapping,
    /// Strategy layers (multiple, with priority)
    strategies: Vec<StrategyLayer>,
    /// Data transform
    transform: DataTransform,
    /// Execution context
    context: ExecutionContext,
    /// Composition mode
    composition: CompositionMode,
    /// Execution policy
    policy: ExecutionPolicy,
    /// Facet parameters for parameter sweep
    facet_params: Option<(String, Vec<f64>)>,
}

impl ComputeBlock {
    /// Create a new ComputeBlock builder
    pub fn builder() -> ComputeBlockBuilder {
        ComputeBlockBuilder::new()
    }

    /// Validate the ComputeBlock configuration
    fn validate(&self) -> GrammarResult<()> {
        // F701: Builder rejects incomplete spec
        if self.workload.is_none() {
            return Err(GrammarError::MissingWorkload);
        }

        // F711: Scale domain validation
        // (handled in ResourceScale)

        Ok(())
    }

    /// Build and validate the ComputeBlock
    pub fn build(self) -> GrammarResult<BuiltComputeBlock> {
        self.validate()?;
        Ok(BuiltComputeBlock { inner: self })
    }
}

/// Builder for ComputeBlock (fluent API)
#[derive(Debug, Clone, Default)]
pub struct ComputeBlockBuilder {
    inner: ComputeBlock,
}

impl ComputeBlockBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            inner: ComputeBlock {
                transform: DataTransform::Identity,
                context: ExecutionContext::Cpu {
                    affinity: None,
                    numa_node: None,
                },
                composition: CompositionMode::None,
                policy: ExecutionPolicy::default(),
                ..Default::default()
            },
        }
    }

    /// Set workload specification
    pub fn workload(mut self, workload: WorkloadSpec) -> Self {
        self.inner.workload = Some(workload);
        self
    }

    /// Set resource mapping
    pub fn resources(mut self, resources: ResourceMapping) -> Self {
        self.inner.resources = resources;
        self
    }

    /// Add a strategy layer
    pub fn strategy(mut self, strategy: ExecutionStrategy) -> Self {
        self.inner.strategies.push(StrategyLayer::new(strategy));
        self
    }

    /// Add a strategy layer with priority
    pub fn strategy_with_priority(mut self, strategy: ExecutionStrategy, priority: i32) -> Self {
        self.inner
            .strategies
            .push(StrategyLayer::new(strategy).priority(priority));
        self
    }

    /// Set data transform
    pub fn transform(mut self, transform: DataTransform) -> Self {
        self.inner.transform = transform;
        self
    }

    /// Set execution context
    pub fn context(mut self, context: ExecutionContext) -> Self {
        self.inner.context = context;
        self
    }

    /// Set composition mode
    pub fn composition(mut self, composition: CompositionMode) -> Self {
        self.inner.composition = composition;
        self
    }

    /// Set execution policy
    pub fn policy(mut self, policy: ExecutionPolicy) -> Self {
        self.inner.policy = policy;
        self
    }

    /// Set facet parameters for parameter sweep
    pub fn facet_by(mut self, param: impl Into<String>, values: Vec<f64>) -> Self {
        self.inner.facet_params = Some((param.into(), values));
        self
    }

    /// Build and validate
    pub fn build(self) -> GrammarResult<BuiltComputeBlock> {
        self.inner.build()
    }
}

// ============================================================================
// Resource Scaling Traits
// ============================================================================

/// Resource scaling trait (analogous to graphics Scale<D, R>)
pub trait ResourceScale<D, R> {
    /// Scale a request from domain to range
    fn scale(&self, request: D) -> R;
    /// Get domain bounds
    fn domain(&self) -> (D, D);
    /// Get range bounds
    fn range(&self) -> (R, R);
}

/// Linear resource scaling
#[derive(Debug, Clone)]
pub struct LinearResourceScale {
    domain: (f64, f64),
    range: (f64, f64),
}

impl LinearResourceScale {
    /// Create new linear scale
    pub fn new(domain: (f64, f64), range: (f64, f64)) -> GrammarResult<Self> {
        if domain.0 >= domain.1 {
            return Err(GrammarError::InvalidScaleDomain {
                min: domain.0,
                max: domain.1,
            });
        }
        Ok(Self { domain, range })
    }
}

impl ResourceScale<f64, f64> for LinearResourceScale {
    fn scale(&self, request: f64) -> f64 {
        let t = (request - self.domain.0) / (self.domain.1 - self.domain.0);
        self.range.0 + t * (self.range.1 - self.range.0)
    }

    fn domain(&self) -> (f64, f64) {
        self.domain
    }

    fn range(&self) -> (f64, f64) {
        self.range
    }
}

/// Logarithmic resource scaling (for exponential resources)
#[derive(Debug, Clone)]
pub struct LogResourceScale {
    base: f64,
    domain: (f64, f64),
    range: (f64, f64),
}

impl LogResourceScale {
    /// Create new log scale
    pub fn new(base: f64, domain: (f64, f64), range: (f64, f64)) -> GrammarResult<Self> {
        if domain.0 >= domain.1 {
            return Err(GrammarError::InvalidScaleDomain {
                min: domain.0,
                max: domain.1,
            });
        }
        Ok(Self {
            base,
            domain,
            range,
        })
    }
}

impl ResourceScale<f64, f64> for LogResourceScale {
    fn scale(&self, request: f64) -> f64 {
        let log_request = request.log(self.base);
        let log_min = self.domain.0.log(self.base);
        let log_max = self.domain.1.log(self.base);
        let t = (log_request - log_min) / (log_max - log_min);
        self.range.0 + t * (self.range.1 - self.range.0)
    }

    fn domain(&self) -> (f64, f64) {
        self.domain
    }

    fn range(&self) -> (f64, f64) {
        self.range
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_spec_dot() {
        let workload = WorkloadSpec::dot(1000);
        assert_eq!(workload.operation, Operation::Dot);
        assert_eq!(workload.dimensions.n, 1000);
        assert_eq!(workload.flop_count(), 2000);
    }

    #[test]
    fn test_workload_spec_matmul() {
        let workload = WorkloadSpec::matmul(64, 64, 64);
        assert_eq!(workload.operation, Operation::Matmul);
        assert_eq!(workload.dimensions.m, 64);
        assert_eq!(workload.dimensions.n, 64);
        assert_eq!(workload.dimensions.k, 64);
    }

    #[test]
    fn test_builder_missing_workload() {
        let result = ComputeBlock::builder().build();
        assert!(matches!(result, Err(GrammarError::MissingWorkload)));
    }

    #[test]
    fn test_builder_with_workload() {
        let result = ComputeBlock::builder()
            .workload(WorkloadSpec::dot(1000))
            .build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_strategy_fallback() {
        let block = ComputeBlock::builder()
            .workload(WorkloadSpec::matmul(1024, 1024, 1024))
            .strategy_with_priority(ExecutionStrategy::gpu_auto(), 10)
            .strategy_with_priority(ExecutionStrategy::simd_auto(), 5)
            .strategy_with_priority(ExecutionStrategy::Sequential, 0)
            .build()
            .unwrap();

        let result = block.execute().unwrap();
        assert!(result.strategy_used.contains("Gpu"));
    }

    #[test]
    fn test_execution_policy_realtime() {
        let policy = ExecutionPolicy::realtime();
        assert_eq!(policy.qos, QosLevel::Realtime);
        assert!(!policy.preemptible);
        assert!(policy.timeout.is_some());
    }

    #[test]
    fn test_linear_scale() {
        let scale = LinearResourceScale::new((0.0, 100.0), (0.0, 8.0)).unwrap();
        assert_eq!(scale.scale(50.0), 4.0);
        assert_eq!(scale.scale(0.0), 0.0);
        assert_eq!(scale.scale(100.0), 8.0);
    }

    #[test]
    fn test_linear_scale_invalid_domain() {
        let result = LinearResourceScale::new((100.0, 0.0), (0.0, 8.0));
        assert!(matches!(
            result,
            Err(GrammarError::InvalidScaleDomain { .. })
        ));
    }

    #[test]
    fn test_data_type_byte_size() {
        assert_eq!(DataType::F32.byte_size(), 4);
        assert_eq!(DataType::F16.byte_size(), 2);
        assert_eq!(DataType::I8.byte_size(), 1);
    }

    #[test]
    fn test_tensor_spec() {
        let spec = TensorSpec::new("test", vec![10, 20, 30], DataType::F32);
        assert_eq!(spec.numel(), 6000);
        assert_eq!(spec.byte_size(), 24000);
    }

    #[test]
    fn test_composition_modes() {
        let batch = CompositionMode::batch(32);
        assert!(matches!(
            batch,
            CompositionMode::Batch { batch_size: 32, .. }
        ));

        let dp = CompositionMode::data_parallel(4);
        assert!(matches!(dp, CompositionMode::DataParallel { shards: 4 }));
    }

    #[test]
    fn test_transform_identity() {
        let t = DataTransform::identity();
        assert!(matches!(t, DataTransform::Identity));
    }

    #[test]
    fn test_facet_by() {
        let block = ComputeBlock::builder()
            .workload(WorkloadSpec::matmul(64, 64, 64))
            .facet_by("tile_size", vec![16.0, 32.0, 64.0])
            .build()
            .unwrap();

        // Facet params should be set
        assert!(block.inner.facet_params.is_some());
    }
}
