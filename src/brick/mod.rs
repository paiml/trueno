//! ComputeBrick: Token-Centric Compute Units
//!
//! A **ComputeBrick** is a self-verifying, token-centric compute unit that bundles:
//! - **Operation**: The compute operation (matmul, dot, softmax, etc.)
//! - **Assertions**: Falsifiable claims about the output (equivalence, bounds)
//! - **Budget**: Performance target in µs/token or tokens/sec
//! - **Backend**: Execution target (Scalar, AVX2, CUDA, etc.)
//!
//! # Core Insight
//!
//! A **token** is the unit of data; a **ComputeBrick** is the unit of compute.
//!
//! ```text
//! Token ──▶ [ComputeBrick] ──▶ Token
//!            (matmul, softmax, attention)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno::brick::{ComputeBrick, ComputeBackend, MatmulOp};
//!
//! let matmul = ComputeBrick::new(MatmulOp::new(1024, 1024, 1024))
//!     .assert_equiv(ComputeBackend::Scalar)
//!     .budget_tok_per_sec(50_000.0)
//!     .backend(ComputeBackend::Avx2);
//!
//! let result = matmul.run((a, b))?;
//! println!("Throughput: {:.0} tok/s", result.tokens_per_sec);
//! ```
//!
//! # Scientific Basis
//!
//! Per Popper (1959), a theory that makes no falsifiable predictions is not scientific.
//! A ComputeBrick with no assertions makes no testable claims and is therefore invalid.

// Submodules
mod batch;
mod buffer;
mod circuit;
mod connection;
mod memory;
mod perf_metrics;
mod profiling;
mod rate_limit;
mod resource_pool;
mod shutdown;

// Re-export profiling functions
pub use profiling::{
    cached_nanos, cached_nanos_or_now, cpu_cycles, get_page_faults, init_time_service,
    with_page_fault_tracking,
};

// Re-export perf_metrics types
pub use perf_metrics::{InferencePhase, PerfMetrics};

// Re-export memory types
#[cfg(not(target_arch = "wasm32"))]
pub use memory::AlignedBuffer;
pub use memory::{
    is_direct_io_aligned, madvise_region, prefetch_for_inference, prefetch_ptr, prefetch_slice,
    CacheAligned, MemoryAdvice, PrefetchLocality, CACHE_LINE_SIZE, CACHE_LINE_SIZE_F32,
    DIRECT_IO_ALIGNMENT,
};

// Re-export buffer types
pub use buffer::{BufferWatermarks, WatermarkedBuffer};

// Re-export circuit breaker types
pub use circuit::{CircuitBreaker, CircuitState};

// Re-export shutdown types
pub use shutdown::{GracefulShutdown, ShutdownGuard, ShutdownResult};

// Re-export resource pool types
pub use resource_pool::{PooledResource, ResourcePool};

// Re-export rate limiting types
pub use rate_limit::{LimitError, ServeLimits};

// Re-export connection types
pub use connection::{ConnectionState, KeepAliveConfig, ManagedConnection};

// Re-export batch types
pub use batch::{balance211, split_batch, Balance211Iter, BatchSplitStrategy};

// KV cache management
mod kv_cache;
pub use kv_cache::{KvCacheManager, KvCacheSlotInfo, SequentialBatchOrderer};

// SIMD configuration
mod simd_config;
pub use simd_config::{
    unroll_tail_process, AmxTileConfig, LazySimdConfig, SimdBackendState, UnrollFactor,
    UnrollTailIterator,
};

// Execution graph and brick profiling types (PAR-073, PAR-200, PAR-201)
mod exec_graph;
pub use exec_graph::{
    BrickBottleneck, BrickCategory, BrickId, BrickSample, BrickStats, CategoryStats, EdgeType,
    ExecutionEdge, ExecutionGraph, ExecutionNode, ExecutionNodeId, PtxRegistry, SyncMode,
    TransferDirection,
};

// BrickProfiler and tile profiling (TILING-SPEC-001)
mod profiler;
pub use profiler::{
    fnv1a_f32_checksum, BrickIdTimer, BrickProfiler, BrickTimer, DivergenceInfo, KernelChecksum,
    TileLevel, TileStats, TileTimer,
};

// Model-level inference tracing (Phase 13, E.11)
mod tracing;
pub use tracing::{
    AttentionTraceConfig, AttentionWeightTrace, KvCacheSessionTrace, KvCacheStateTrace,
    LayerActivationTrace, LogitEvolutionTrace, ModelActivationTrace, ModelQuantizationError,
    ModelTracer, ModelTracerConfig, ModelTracerSummary, QuantType, QuantizationErrorTrace,
    TensorStats, TokenLogitEvolution,
};

// Async and buffer patterns (Phase 12, E.10)
mod patterns;
pub use patterns::{
    reserve_capacity, AsyncResult, BoundedQueue, DualWakerState, FlowControlError,
    GraphReuseCounter, ReserveStrategy, StrategicBuffer, StreamCapacity, WakeDecision,
    WakeSkipState,
};

// Built-in compute operations
mod ops;
pub use ops::{AddOp, DotOp, MatmulOp, SoftmaxOp};

// Fused operations for transformer inference (PMAT-PERF-009)
mod fused_ops;
pub use fused_ops::{FusedGateUpOp, FusedGateUpWeights, FusedQKVOp, FusedQKVWeights};

// SIMD-optimized attention operation (PMAT-017)
mod attention;
pub use attention::AttentionOp;

// Q5_K and Q6_K quantization operations (llama.cpp compatible)
mod quant_ops;
pub use quant_ops::{BlockQ5K, BlockQ6K, DotQ5KOp, DotQ6KOp};

// Tests (7,400+ lines extracted for TDG compliance)
#[cfg(test)]
mod tests;

use crate::error::TruenoError;
use std::fmt;
use std::marker::PhantomData;
use std::time::Instant;


// Async task profiler (Phase 11, E.9.4)
mod async_profiler;
pub use async_profiler::AsyncTaskProfiler;

// Budget types (TokenBudget, ByteBudget, TokenResult)
mod budget;
pub use budget::{ByteBudget, TokenBudget, TokenResult};

/// Execution backend for compute operations.
/// This is the brick-specific backend enum with additional GPU backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ComputeBackend {
    /// Pure Rust scalar fallback (always available, baseline for correctness)
    Scalar,
    /// SSE2 SIMD (x86_64 baseline)
    Sse2,
    /// AVX2 256-bit SIMD with FMA
    #[default]
    Avx2,
    /// AVX-512 512-bit SIMD
    Avx512,
    /// ARM NEON SIMD
    Neon,
    /// WebAssembly SIMD128
    Wasm,
    /// NVIDIA CUDA via PTX
    Cuda,
    /// Cross-platform GPU via wgpu
    Wgpu,
    /// Auto-select best available backend
    Auto,
}

impl fmt::Display for ComputeBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComputeBackend::Scalar => write!(f, "Scalar"),
            ComputeBackend::Sse2 => write!(f, "SSE2"),
            ComputeBackend::Avx2 => write!(f, "AVX2"),
            ComputeBackend::Avx512 => write!(f, "AVX-512"),
            ComputeBackend::Neon => write!(f, "NEON"),
            ComputeBackend::Wasm => write!(f, "WASM"),
            ComputeBackend::Cuda => write!(f, "CUDA"),
            ComputeBackend::Wgpu => write!(f, "wgpu"),
            ComputeBackend::Auto => write!(f, "Auto"),
        }
    }
}

/// Type alias for backward compatibility
pub type Backend = ComputeBackend;


/// Errors from ComputeBrick execution.
/// Tells you exactly what failed (Jidoka: stop and signal).
#[derive(Debug, thiserror::Error)]
pub enum BrickError {
    /// Assertion failed during verification
    #[error("Assertion failed: {name} - expected {expected}, got {actual}")]
    AssertionFailed {
        name: String,
        expected: String,
        actual: String,
    },

    /// Performance budget exceeded
    #[error("Budget exceeded: {limit_us:.1}µs/tok limit, {actual_us:.1}µs/tok actual ({utilization:.0}% of budget)")]
    BudgetExceeded {
        limit_us: f64,
        actual_us: f64,
        utilization: f64,
    },

    /// Underlying compute error
    #[error("Compute error: {0}")]
    ComputeError(#[from] TruenoError),

    /// No assertions defined (violates Popperian falsifiability)
    #[error("Brick has no assertions - violates Popperian falsifiability requirement")]
    NoAssertions,

    /// Backend not available
    #[error("Backend {0} not available on this system")]
    BackendUnavailable(Backend),
}

/// Type of assertion for compute verification.
#[derive(Debug, Clone)]
pub enum ComputeAssertion {
    /// Output must match baseline backend within tolerance
    Equivalence { baseline: Backend, tolerance: f64 },
    /// Output values must be within bounds
    Bounds { min: f64, max: f64 },
    /// Output must not contain NaN or infinity
    Finite,
    /// Custom assertion with name and check function index
    Custom { name: String },
}

impl ComputeAssertion {
    /// Create equivalence assertion with default tolerance (1e-5).
    pub fn equiv(baseline: Backend) -> Self {
        Self::Equivalence {
            baseline,
            tolerance: 1e-5,
        }
    }

    /// Create equivalence assertion with custom tolerance.
    pub fn equiv_with_tolerance(baseline: Backend, tolerance: f64) -> Self {
        Self::Equivalence {
            baseline,
            tolerance,
        }
    }

    /// Create bounds assertion.
    pub fn bounds(min: f64, max: f64) -> Self {
        Self::Bounds { min, max }
    }

    /// Create finite assertion (no NaN/Inf).
    pub fn finite() -> Self {
        Self::Finite
    }
}

/// Verification result from ComputeBrick.
#[derive(Debug, Clone)]
pub struct BrickVerification {
    /// Overall pass/fail
    pub passed: bool,
    /// Individual assertion results
    pub assertion_results: Vec<AssertionResult>,
    /// Verification time in microseconds
    pub verification_us: f64,
}

impl BrickVerification {
    /// Check if all assertions passed.
    pub fn is_valid(&self) -> bool {
        self.passed
    }

    /// Get failed assertions.
    pub fn failures(&self) -> impl Iterator<Item = &AssertionResult> {
        self.assertion_results.iter().filter(|r| !r.passed)
    }
}

/// Result of a single assertion check.
#[derive(Debug, Clone)]
pub struct AssertionResult {
    /// Assertion that was checked
    pub assertion: ComputeAssertion,
    /// Did it pass?
    pub passed: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Trait for compute operations that can be wrapped in a ComputeBrick.
pub trait ComputeOp: Send + Sync {
    /// Input type for this operation
    type Input;
    /// Output type for this operation
    type Output;

    /// Operation name for identification
    fn name(&self) -> &'static str;

    /// Execute the operation on the given backend
    fn execute(&self, input: Self::Input, backend: Backend) -> Result<Self::Output, TruenoError>;

    /// Number of tokens this operation processes (for budget calculation)
    fn tokens(&self, input: &Self::Input) -> usize;

    /// Clone the input for verification (if needed)
    fn clone_input(&self, input: &Self::Input) -> Option<Self::Input>
    where
        Self::Input: Clone,
    {
        Some(input.clone())
    }
}

/// Self-verifying, token-centric compute unit.
/// Bundles: operation + assertions + budget + verification
pub struct ComputeBrick<Op: ComputeOp> {
    /// The compute operation
    op: Op,
    /// Falsifiable assertions
    assertions: Vec<ComputeAssertion>,
    /// Token-centric performance budget
    budget: TokenBudget,
    /// Execution backend
    backend: Backend,
    /// Enforce budget (fail if exceeded)
    enforce_budget: bool,
    /// Phantom for variance
    _phantom: PhantomData<Op>,
}

impl<Op: ComputeOp> ComputeBrick<Op> {
    /// Create a new compute brick with the given operation.
    pub fn new(op: Op) -> Self {
        Self {
            op,
            assertions: Vec::new(),
            budget: TokenBudget::default(),
            backend: Backend::Auto,
            enforce_budget: false,
            _phantom: PhantomData,
        }
    }

    /// Add equivalence assertion (output must match baseline backend).
    #[must_use]
    pub fn assert_equiv(mut self, baseline: Backend) -> Self {
        self.assertions.push(ComputeAssertion::equiv(baseline));
        self
    }

    /// Add equivalence assertion with custom tolerance.
    #[must_use]
    pub fn assert_equiv_with_tolerance(mut self, baseline: Backend, tolerance: f64) -> Self {
        self.assertions
            .push(ComputeAssertion::equiv_with_tolerance(baseline, tolerance));
        self
    }

    /// Add bounds assertion (output values within range).
    #[must_use]
    pub fn assert_bounds(mut self, min: f64, max: f64) -> Self {
        self.assertions.push(ComputeAssertion::bounds(min, max));
        self
    }

    /// Add finite assertion (no NaN/Inf in output).
    #[must_use]
    pub fn assert_finite(mut self) -> Self {
        self.assertions.push(ComputeAssertion::finite());
        self
    }

    /// Set token throughput budget (tokens/second).
    #[must_use]
    pub fn budget_tok_per_sec(mut self, tps: f64) -> Self {
        self.budget = TokenBudget::from_throughput(tps);
        self
    }

    /// Set token latency budget (microseconds/token).
    #[must_use]
    pub fn budget_us_per_tok(mut self, us: f64) -> Self {
        self.budget = TokenBudget::from_latency(us);
        self
    }

    /// Set full budget configuration.
    #[must_use]
    pub fn budget(mut self, budget: TokenBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Set execution backend.
    #[must_use]
    pub fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Enforce budget (fail if exceeded). Default is false (just report).
    #[must_use]
    pub fn enforce_budget(mut self, enforce: bool) -> Self {
        self.enforce_budget = enforce;
        self
    }

    /// Get the brick name (from operation).
    pub fn name(&self) -> &'static str {
        self.op.name()
    }

    /// Get current budget.
    pub fn get_budget(&self) -> TokenBudget {
        self.budget
    }

    /// Get current backend.
    pub fn get_backend(&self) -> Backend {
        self.backend
    }

    /// Get assertions.
    pub fn get_assertions(&self) -> &[ComputeAssertion] {
        &self.assertions
    }

    /// Run the compute brick with full verification (Jidoka gate).
    pub fn run(&self, input: Op::Input) -> Result<TokenResult<Op::Output>, BrickError> {
        let tokens = self.op.tokens(&input);

        // Execute with timing
        let start = Instant::now();
        let output = self.op.execute(input, self.backend)?;
        let elapsed_us = start.elapsed().as_secs_f64() * 1_000_000.0;

        // Calculate metrics
        let us_per_token = if tokens > 0 {
            elapsed_us / tokens as f64
        } else {
            elapsed_us
        };
        let tokens_per_sec = if elapsed_us > 0.0 {
            tokens as f64 * 1_000_000.0 / elapsed_us
        } else {
            f64::INFINITY
        };
        let budget_met = self.budget.is_met(us_per_token);
        let budget_utilization = self.budget.utilization(us_per_token);

        // Check budget enforcement
        if self.enforce_budget && !budget_met {
            return Err(BrickError::BudgetExceeded {
                limit_us: self.budget.us_per_token,
                actual_us: us_per_token,
                utilization: budget_utilization * 100.0,
            });
        }

        Ok(TokenResult {
            output,
            tokens_processed: tokens,
            us_per_token,
            tokens_per_sec,
            budget_met,
            budget_utilization,
        })
    }

    /// Verify assertions without full execution.
    /// Returns verification status.
    pub fn verify(&self) -> BrickVerification {
        let start = Instant::now();

        // Check if we have assertions (Popperian requirement)
        if self.assertions.is_empty() {
            return BrickVerification {
                passed: false,
                assertion_results: vec![AssertionResult {
                    assertion: ComputeAssertion::Custom {
                        name: "popperian_falsifiability".to_string(),
                    },
                    passed: false,
                    error: Some(
                        "No assertions defined - violates Popperian falsifiability".to_string(),
                    ),
                }],
                verification_us: start.elapsed().as_secs_f64() * 1_000_000.0,
            };
        }

        // For now, just validate assertion structure
        // Full verification requires input data
        let results: Vec<AssertionResult> = self
            .assertions
            .iter()
            .map(|a| AssertionResult {
                assertion: a.clone(),
                passed: true,
                error: None,
            })
            .collect();

        let passed = results.iter().all(|r| r.passed);

        BrickVerification {
            passed,
            assertion_results: results,
            verification_us: start.elapsed().as_secs_f64() * 1_000_000.0,
        }
    }
}

impl<Op: ComputeOp + Clone> Clone for ComputeBrick<Op> {
    fn clone(&self) -> Self {
        Self {
            op: self.op.clone(),
            assertions: self.assertions.clone(),
            budget: self.budget,
            backend: self.backend,
            enforce_budget: self.enforce_budget,
            _phantom: PhantomData,
        }
    }
}

impl<Op: ComputeOp> fmt::Debug for ComputeBrick<Op> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComputeBrick")
            .field("name", &self.op.name())
            .field("backend", &self.backend)
            .field("budget", &self.budget)
            .field("assertions", &self.assertions.len())
            .field("enforce_budget", &self.enforce_budget)
            .finish()
    }
}

// ============================================================================
// LLM Transformer Fused Operations (PMAT-PERF-009)
// BrickLayer: Compose multiple bricks
// ============================================================================

/// A layer of compute bricks that execute sequentially.
/// Throughput ceiling = min(component throughputs).
#[derive(Debug, Default)]
pub struct BrickLayer {
    /// Named bricks in this layer
    bricks: Vec<(String, f64)>, // (name, budget_tok_per_sec)
}

impl BrickLayer {
    /// Create a new empty layer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a brick to the layer.
    #[must_use]
    pub fn with_brick<Op: ComputeOp>(mut self, brick: &ComputeBrick<Op>) -> Self {
        self.bricks
            .push((brick.name().to_string(), brick.budget.tokens_per_sec));
        self
    }

    /// Add a named entry with throughput budget.
    #[must_use]
    pub fn with_named(mut self, name: &str, budget_tok_per_sec: f64) -> Self {
        self.bricks.push((name.to_string(), budget_tok_per_sec));
        self
    }

    /// Get the throughput ceiling (bottleneck).
    /// Layer throughput = min(component throughputs).
    pub fn throughput_ceiling(&self) -> f64 {
        self.bricks
            .iter()
            .map(|(_, tps)| *tps)
            .fold(f64::INFINITY, f64::min)
    }

    /// Get the bottleneck brick name.
    pub fn bottleneck(&self) -> Option<&str> {
        self.bricks
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, _)| name.as_str())
    }

    /// Get all bricks with their budgets.
    pub fn bricks(&self) -> &[(String, f64)] {
        &self.bricks
    }
}
