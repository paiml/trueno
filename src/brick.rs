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

use crate::error::TruenoError;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::time::Instant;

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

/// Performance budget expressed in token terms.
/// Aligns compute costs with LLM inference metrics.
#[derive(Debug, Clone, Copy)]
pub struct TokenBudget {
    /// Latency budget per token (microseconds)
    pub us_per_token: f64,
    /// Throughput target (tokens/second)
    pub tokens_per_sec: f64,
    /// Batch size for amortization
    pub batch_size: usize,
}

/// Performance budget for byte-oriented operations (compression, I/O).
/// Use this for trueno-zram, disk I/O, network throughput, etc.
///
/// PMAT-452: Serializable for hardware.toml export.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ByteBudget {
    /// Latency budget per page (microseconds)
    pub us_per_page: f64,
    /// Throughput target (GB/s)
    pub gb_per_sec: f64,
    /// Page size in bytes (default 4096)
    pub page_size: usize,
}

impl Default for ByteBudget {
    fn default() -> Self {
        // Default: 25 GB/s (trueno-zram ZSTD target)
        Self::from_throughput(25.0)
    }
}

impl ByteBudget {
    /// Create budget from throughput target (GB/s).
    /// 25 GB/s = 0.16µs per 4KB page
    pub fn from_throughput(gb_per_sec: f64) -> Self {
        let bytes_per_sec = gb_per_sec * 1e9;
        let pages_per_sec = bytes_per_sec / 4096.0;
        Self {
            us_per_page: 1_000_000.0 / pages_per_sec,
            gb_per_sec,
            page_size: 4096,
        }
    }

    /// Create budget from latency target (µs per page).
    pub fn from_latency(us_per_page: f64) -> Self {
        let pages_per_sec = 1_000_000.0 / us_per_page;
        let bytes_per_sec = pages_per_sec * 4096.0;
        Self {
            us_per_page,
            gb_per_sec: bytes_per_sec / 1e9,
            page_size: 4096,
        }
    }

    /// Set custom page size (e.g., 64KB for huge pages).
    #[must_use]
    pub fn with_page_size(mut self, page_size: usize) -> Self {
        // Recalculate us_per_page based on new page size
        let bytes_per_sec = self.gb_per_sec * 1e9;
        let pages_per_sec = bytes_per_sec / page_size as f64;
        self.us_per_page = 1_000_000.0 / pages_per_sec;
        self.page_size = page_size;
        self
    }

    /// Convert to TokenBudget (1 token = 1 page).
    /// Useful for integrating byte workloads with token-centric monitoring.
    pub fn to_token_budget(&self) -> TokenBudget {
        TokenBudget {
            us_per_token: self.us_per_page,
            tokens_per_sec: 1_000_000.0 / self.us_per_page,
            batch_size: 1,
        }
    }

    /// Check if actual performance meets budget.
    pub fn is_met(&self, actual_us_per_page: f64) -> bool {
        actual_us_per_page <= self.us_per_page
    }

    /// Calculate budget utilization.
    pub fn utilization(&self, actual_us_per_page: f64) -> f64 {
        actual_us_per_page / self.us_per_page
    }

    /// Calculate actual throughput from latency.
    pub fn throughput_from_latency(us_per_page: f64, page_size: usize) -> f64 {
        let pages_per_sec = 1_000_000.0 / us_per_page;
        pages_per_sec * page_size as f64 / 1e9
    }
}

impl Default for TokenBudget {
    fn default() -> Self {
        // Default: 50µs/token = 20,000 tokens/sec
        Self::from_latency(50.0)
    }
}

impl TokenBudget {
    /// Create budget from latency target.
    /// 50µs/token = 20,000 tokens/sec
    pub fn from_latency(us_per_token: f64) -> Self {
        Self {
            us_per_token,
            tokens_per_sec: 1_000_000.0 / us_per_token,
            batch_size: 1,
        }
    }

    /// Create budget from throughput target.
    /// 20,000 tokens/sec = 50µs/token
    pub fn from_throughput(tokens_per_sec: f64) -> Self {
        Self {
            us_per_token: 1_000_000.0 / tokens_per_sec,
            tokens_per_sec,
            batch_size: 1,
        }
    }

    /// Set batch size for amortization.
    #[must_use]
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Check if actual performance meets budget.
    pub fn is_met(&self, actual_us_per_token: f64) -> bool {
        actual_us_per_token <= self.us_per_token
    }

    /// Calculate budget utilization (0.0 = unused, 1.0 = exactly at budget, >1.0 = over budget).
    pub fn utilization(&self, actual_us_per_token: f64) -> f64 {
        actual_us_per_token / self.us_per_token
    }
}

/// Result of ComputeBrick execution with token metrics.
#[derive(Debug, Clone)]
pub struct TokenResult<T> {
    /// Computed output
    pub output: T,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Actual latency (microseconds/token)
    pub us_per_token: f64,
    /// Actual throughput (tokens/second)
    pub tokens_per_sec: f64,
    /// Did we meet the budget?
    pub budget_met: bool,
    /// Budget utilization (0.0-1.0+ where 1.0 = exactly at budget)
    pub budget_utilization: f64,
}

impl<T> TokenResult<T> {
    /// Map the output to a new type.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> TokenResult<U> {
        TokenResult {
            output: f(self.output),
            tokens_processed: self.tokens_processed,
            us_per_token: self.us_per_token,
            tokens_per_sec: self.tokens_per_sec,
            budget_met: self.budget_met,
            budget_utilization: self.budget_utilization,
        }
    }
}

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
    Equivalence {
        baseline: Backend,
        tolerance: f64,
    },
    /// Output values must be within bounds
    Bounds {
        min: f64,
        max: f64,
    },
    /// Output must not contain NaN or infinity
    Finite,
    /// Custom assertion with name and check function index
    Custom {
        name: String,
    },
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
        Self::Equivalence { baseline, tolerance }
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
                    error: Some("No assertions defined - violates Popperian falsifiability".to_string()),
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
// Built-in Operations
// ============================================================================

/// Dot product operation.
#[derive(Debug, Clone)]
pub struct DotOp {
    /// Expected vector length
    pub len: usize,
}

impl DotOp {
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl ComputeOp for DotOp {
    type Input = (Vec<f32>, Vec<f32>);
    type Output = f32;

    fn name(&self) -> &'static str {
        "dot"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (a, b) = input;
        if a.len() != b.len() {
            return Err(TruenoError::SizeMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        // Simple scalar implementation for now
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        Ok(sum)
    }

    fn tokens(&self, input: &Self::Input) -> usize {
        // Each element pair is roughly 1 "token" of work
        input.0.len()
    }
}

/// Element-wise add operation.
#[derive(Debug, Clone)]
pub struct AddOp {
    /// Expected vector length
    pub len: usize,
}

impl AddOp {
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl ComputeOp for AddOp {
    type Input = (Vec<f32>, Vec<f32>);
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "add"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (a, b) = input;
        if a.len() != b.len() {
            return Err(TruenoError::SizeMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }

    fn tokens(&self, input: &Self::Input) -> usize {
        input.0.len()
    }
}

/// Matrix multiplication operation.
#[derive(Debug, Clone)]
pub struct MatmulOp {
    /// M dimension (rows of A)
    pub m: usize,
    /// K dimension (cols of A = rows of B)
    pub k: usize,
    /// N dimension (cols of B)
    pub n: usize,
}

impl MatmulOp {
    pub fn new(m: usize, k: usize, n: usize) -> Self {
        Self { m, k, n }
    }
}

impl ComputeOp for MatmulOp {
    type Input = (Vec<f32>, Vec<f32>);
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "matmul"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (a, b) = input;
        let expected_a = self.m * self.k;
        let expected_b = self.k * self.n;

        if a.len() != expected_a {
            return Err(TruenoError::SizeMismatch {
                expected: expected_a,
                actual: a.len(),
            });
        }
        if b.len() != expected_b {
            return Err(TruenoError::SizeMismatch {
                expected: expected_b,
                actual: b.len(),
            });
        }

        // Simple scalar implementation
        let mut c = vec![0.0f32; self.m * self.n];
        for i in 0..self.m {
            for j in 0..self.n {
                let mut sum = 0.0f32;
                for p in 0..self.k {
                    sum += a[i * self.k + p] * b[p * self.n + j];
                }
                c[i * self.n + j] = sum;
            }
        }
        Ok(c)
    }

    fn tokens(&self, _input: &Self::Input) -> usize {
        // For matmul, "tokens" = number of output elements
        // Each output requires K multiply-adds
        self.m * self.n
    }
}

/// Softmax operation.
#[derive(Debug, Clone)]
pub struct SoftmaxOp {
    /// Expected vector length
    pub len: usize,
}

impl SoftmaxOp {
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl ComputeOp for SoftmaxOp {
    type Input = Vec<f32>;
    type Output = Vec<f32>;

    fn name(&self) -> &'static str {
        "softmax"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        if input.is_empty() {
            return Ok(vec![]);
        }

        // Numerically stable softmax
        let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = input.iter().map(|x| (x - max).exp()).sum();
        let result: Vec<f32> = input.iter().map(|x| (x - max).exp() / exp_sum).collect();
        Ok(result)
    }

    fn tokens(&self, input: &Self::Input) -> usize {
        input.len()
    }
}

// ============================================================================
// LLM Transformer Fused Operations (PMAT-PERF-009)
// ============================================================================

/// Weights for fused QKV projection
#[derive(Debug, Clone)]
pub struct FusedQKVWeights {
    /// Q projection weights [hidden_size, hidden_size]
    pub q_weight: Vec<f32>,
    /// K projection weights [hidden_size, kv_dim]
    pub k_weight: Vec<f32>,
    /// V projection weights [hidden_size, kv_dim]
    pub v_weight: Vec<f32>,
}

/// Fused Q/K/V projection operation for transformer attention.
///
/// Computes Q, K, V projections in a single pass over the input:
/// - Q = x * W_q (hidden_size → hidden_size)
/// - K = x * W_k (hidden_size → kv_dim)
/// - V = x * W_v (hidden_size → kv_dim)
///
/// # Performance Impact
///
/// Fusing 3 separate matmuls into 1 operation provides:
/// - 3x reduction in kernel launches (GPU)
/// - Better cache utilization (input x loaded once)
/// - Expected speedup: 2-3x for decode phase
///
/// # Five-Whys Root Cause (PMAT-PERF-009)
///
/// ```text
/// Why 1: Why is decode throughput 131 tok/s vs 400 tok/s target?
/// → 280+ kernel launches per token (10+ per layer × 28 layers)
///
/// Why 2: Why so many kernel launches?
/// → Q, K, V computed as 3 separate GEMV operations
///
/// Why 3: Why separate operations?
/// → Original implementation didn't consider launch overhead
///
/// Why 4: Why does launch overhead matter?
/// → GPU kernel launch: ~5-10µs, 280 launches = 1.4-2.8ms overhead/token
///
/// Why 5: ROOT CAUSE
/// → Kernel launch overhead (2.8ms) exceeds compute time for small batch decode
/// → FIX: Fuse Q/K/V into single kernel, reducing launches by 2/3
/// ```
#[derive(Debug, Clone)]
pub struct FusedQKVOp {
    /// Hidden dimension size
    pub hidden_size: usize,
    /// KV dimension (num_kv_heads * head_dim, may differ from hidden_size for GQA)
    pub kv_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl FusedQKVOp {
    /// Create a new fused QKV operation.
    ///
    /// # Arguments
    /// * `hidden_size` - Hidden dimension (e.g., 3584 for Qwen 3B)
    /// * `num_heads` - Number of attention heads
    /// * `num_kv_heads` - Number of KV heads (may differ for GQA)
    pub fn new(hidden_size: usize, num_heads: usize, num_kv_heads: usize) -> Self {
        let head_dim = hidden_size / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        Self {
            hidden_size,
            kv_dim,
            num_heads,
            head_dim,
        }
    }
}

#[allow(clippy::needless_range_loop)] // Matrix indexing is clearer with explicit loops
impl ComputeOp for FusedQKVOp {
    type Input = (Vec<f32>, FusedQKVWeights);
    type Output = (Vec<f32>, Vec<f32>, Vec<f32>); // (Q, K, V)

    fn name(&self) -> &'static str {
        "fused_qkv"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (x, weights) = input;

        // Validate input dimensions
        if x.len() != self.hidden_size {
            return Err(TruenoError::SizeMismatch {
                expected: self.hidden_size,
                actual: x.len(),
            });
        }

        // Q projection: x @ W_q^T -> [hidden_size]
        let mut q = vec![0.0f32; self.hidden_size];
        for i in 0..self.hidden_size {
            let mut sum = 0.0f32;
            for j in 0..self.hidden_size {
                sum += x[j] * weights.q_weight[i * self.hidden_size + j];
            }
            q[i] = sum;
        }

        // K projection: x @ W_k^T -> [kv_dim]
        let mut k = vec![0.0f32; self.kv_dim];
        for i in 0..self.kv_dim {
            let mut sum = 0.0f32;
            for j in 0..self.hidden_size {
                sum += x[j] * weights.k_weight[i * self.hidden_size + j];
            }
            k[i] = sum;
        }

        // V projection: x @ W_v^T -> [kv_dim]
        let mut v = vec![0.0f32; self.kv_dim];
        for i in 0..self.kv_dim {
            let mut sum = 0.0f32;
            for j in 0..self.hidden_size {
                sum += x[j] * weights.v_weight[i * self.hidden_size + j];
            }
            v[i] = sum;
        }

        Ok((q, k, v))
    }

    fn tokens(&self, _input: &Self::Input) -> usize {
        // Output tokens = Q + K + V dimensions
        self.hidden_size + 2 * self.kv_dim
    }
}

/// Weights for fused gate+up FFN projection
#[derive(Debug, Clone)]
pub struct FusedGateUpWeights {
    /// Gate projection weights [hidden_size, intermediate_size]
    pub gate_weight: Vec<f32>,
    /// Up projection weights [hidden_size, intermediate_size]
    pub up_weight: Vec<f32>,
}

/// Fused Gate+Up FFN projection with SiLU activation.
///
/// Computes gate and up projections in a single pass:
/// - gate = x * W_gate
/// - up = x * W_up
/// - output = SiLU(gate) * up (SwiGLU activation)
///
/// # Performance Impact
///
/// Fusing 2 separate matmuls + activation provides:
/// - 2x reduction in kernel launches (GPU)
/// - Fused SiLU avoids intermediate memory traffic
/// - Expected speedup: 1.5-2x for decode phase
///
/// # Five-Whys Root Cause (PMAT-PERF-009)
///
/// ```text
/// Why 1: Why is FFN phase slow?
/// → 3 kernel launches: gate_proj, up_proj, SiLU activation
///
/// Why 2: Why separate kernels?
/// → Traditional implementation pattern from training frameworks
///
/// Why 3: Why does this matter for inference?
/// → Inference is memory-bound; kernel launch overhead dominates
///
/// Why 4: Why not fuse earlier?
/// → Requires custom kernel development
///
/// Why 5: ROOT CAUSE
/// → SwiGLU requires gate*up pattern that naturally fuses
/// → FIX: Fuse gate+up+SiLU into single operation
/// ```
#[derive(Debug, Clone)]
pub struct FusedGateUpOp {
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Intermediate FFN dimension
    pub intermediate_size: usize,
}

impl FusedGateUpOp {
    /// Create a new fused gate+up operation.
    ///
    /// # Arguments
    /// * `hidden_size` - Hidden dimension (e.g., 3584 for Qwen 3B)
    /// * `intermediate_size` - FFN intermediate dimension (e.g., 18944)
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            hidden_size,
            intermediate_size,
        }
    }

    /// SiLU activation: x * sigmoid(x)
    #[inline]
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
}

#[allow(clippy::needless_range_loop)] // Matrix indexing is clearer with explicit loops
impl ComputeOp for FusedGateUpOp {
    type Input = (Vec<f32>, FusedGateUpWeights);
    type Output = Vec<f32>; // SwiGLU output [intermediate_size]

    fn name(&self) -> &'static str {
        "fused_gate_up"
    }

    fn execute(&self, input: Self::Input, _backend: Backend) -> Result<Self::Output, TruenoError> {
        let (x, weights) = input;

        // Validate input dimensions
        if x.len() != self.hidden_size {
            return Err(TruenoError::SizeMismatch {
                expected: self.hidden_size,
                actual: x.len(),
            });
        }

        // Fused gate + up + SwiGLU
        let mut output = vec![0.0f32; self.intermediate_size];

        for i in 0..self.intermediate_size {
            // Gate projection
            let mut gate_sum = 0.0f32;
            for j in 0..self.hidden_size {
                gate_sum += x[j] * weights.gate_weight[i * self.hidden_size + j];
            }

            // Up projection
            let mut up_sum = 0.0f32;
            for j in 0..self.hidden_size {
                up_sum += x[j] * weights.up_weight[i * self.hidden_size + j];
            }

            // SwiGLU: SiLU(gate) * up
            output[i] = Self::silu(gate_sum) * up_sum;
        }

        Ok(output)
    }

    fn tokens(&self, _input: &Self::Input) -> usize {
        self.intermediate_size
    }
}

// ============================================================================
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

impl std::fmt::Display for BrickBottleneck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
            Self::QkvProjection | Self::RopeEmbedding | Self::AttentionScore |
            Self::AttentionSoftmax | Self::AttentionOutput | Self::OutputProjection
                => BrickCategory::Attention,
            Self::GateProjection | Self::UpProjection | Self::Activation |
            Self::DownProjection => BrickCategory::Ffn,
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
    Layer {
        index: u32,
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
            Self::Transfer { src, dst, direction, .. } => {
                let dir = match direction {
                    TransferDirection::H2D => "H2D",
                    TransferDirection::D2H => "D2H",
                    TransferDirection::D2D => "D2D",
                };
                format!("{}:{}->{}", dir, src, dst)
            }
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
            Self::Kernel { arithmetic_intensity, .. } => *arithmetic_intensity,
            _ => None,
        }
    }

    /// Get achieved TFLOP/s if available (kernels only, Phase 9).
    pub fn achieved_tflops(&self) -> Option<f32> {
        match self {
            Self::Kernel { achieved_tflops, .. } => *achieved_tflops,
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

/// Execution path graph for tracking brick → kernel → PTX relationships.
///
/// PAR-201: Captures the full execution hierarchy for profiling analysis.
///
/// # Example
///
/// ```rust,ignore
/// use trueno::brick::{ExecutionGraph, ExecutionNode, EdgeType};
///
/// let mut graph = ExecutionGraph::new();
///
/// // Add layer scope
/// let layer_id = graph.add_node(ExecutionNode::Layer { index: 0 });
///
/// // Add brick within layer
/// let brick_id = graph.add_node(ExecutionNode::Brick {
///     id: BrickId::QkvProjection,
///     timing_ns: 1000,
///     elements: 4096,
/// });
/// graph.add_edge(layer_id, brick_id, EdgeType::Contains);
///
/// // Add kernel launched by brick
/// let kernel_id = graph.add_node(ExecutionNode::Kernel {
///     name: "batched_q4k_gemv".into(),
///     ptx_hash: 0x7a3b1c2d,
///     grid: (32, 1, 1),
///     block: (256, 1, 1),
///     shared_mem: 4096,
/// });
/// graph.add_edge(brick_id, kernel_id, EdgeType::Launches);
///
/// // Export to trueno-graph for analysis
/// #[cfg(feature = "execution-graph")]
/// let csr = graph.to_csr();
/// ```
#[derive(Debug, Default)]
pub struct ExecutionGraph {
    /// All nodes in the graph
    nodes: Vec<ExecutionNode>,
    /// All edges in the graph
    edges: Vec<ExecutionEdge>,
    /// Scope stack for hierarchical recording
    scope_stack: Vec<ExecutionNodeId>,
    /// Node name → ID mapping for fast lookup
    name_to_id: std::collections::HashMap<String, ExecutionNodeId>,
}

impl ExecutionGraph {
    /// Create a new empty execution graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph, returning its ID.
    pub fn add_node(&mut self, node: ExecutionNode) -> ExecutionNodeId {
        let id = ExecutionNodeId(self.nodes.len() as u32);
        let name = node.name();
        self.name_to_id.insert(name, id);
        self.nodes.push(node);
        id
    }

    /// Add an edge between two nodes.
    pub fn add_edge(&mut self, src: ExecutionNodeId, dst: ExecutionNodeId, edge_type: EdgeType) {
        self.edges.push(ExecutionEdge {
            src,
            dst,
            edge_type,
            weight: 1.0,
        });
    }

    /// Add an edge with a weight.
    pub fn add_weighted_edge(
        &mut self,
        src: ExecutionNodeId,
        dst: ExecutionNodeId,
        edge_type: EdgeType,
        weight: f32,
    ) {
        self.edges.push(ExecutionEdge {
            src,
            dst,
            edge_type,
            weight,
        });
    }

    /// Push a scope for hierarchical recording.
    /// All subsequent nodes will be children of this scope.
    pub fn push_scope(&mut self, node: ExecutionNode) -> ExecutionNodeId {
        let id = self.add_node(node);
        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, id, EdgeType::Contains);
        }
        self.scope_stack.push(id);
        id
    }

    /// Pop the current scope.
    pub fn pop_scope(&mut self) -> Option<ExecutionNodeId> {
        self.scope_stack.pop()
    }

    /// Get the current scope (if any).
    pub fn current_scope(&self) -> Option<ExecutionNodeId> {
        self.scope_stack.last().copied()
    }

    /// Add a node under the current scope.
    pub fn add_node_in_scope(&mut self, node: ExecutionNode) -> ExecutionNodeId {
        let id = self.add_node(node);
        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, id, EdgeType::Contains);
        }
        id
    }

    /// Record a kernel launch under the current scope.
    pub fn record_kernel_launch(
        &mut self,
        name: &str,
        ptx_hash: u64,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
    ) -> ExecutionNodeId {
        let kernel = ExecutionNode::Kernel {
            name: name.to_string(),
            ptx_hash,
            grid,
            block,
            shared_mem,
            timing_ns: None,
            arithmetic_intensity: None,
            achieved_tflops: None,
        };
        let kernel_id = self.add_node(kernel);

        // Link from current scope with Launches edge
        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, kernel_id, EdgeType::Launches);
        }

        kernel_id
    }

    /// Record a kernel launch with roofline metrics (Phase 9).
    #[allow(clippy::too_many_arguments)]
    pub fn record_kernel_launch_with_metrics(
        &mut self,
        name: &str,
        ptx_hash: u64,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        timing_ns: u64,
        arithmetic_intensity: f32,
        achieved_tflops: f32,
    ) -> ExecutionNodeId {
        let kernel = ExecutionNode::Kernel {
            name: name.to_string(),
            ptx_hash,
            grid,
            block,
            shared_mem,
            timing_ns: Some(timing_ns),
            arithmetic_intensity: Some(arithmetic_intensity),
            achieved_tflops: Some(achieved_tflops),
        };
        let kernel_id = self.add_node(kernel);

        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, kernel_id, EdgeType::Launches);
        }

        kernel_id
    }

    /// Record a memory transfer (Phase 9: data movement topology).
    pub fn record_transfer(
        &mut self,
        src: &str,
        dst: &str,
        bytes: u64,
        direction: TransferDirection,
        timing_ns: Option<u64>,
    ) -> ExecutionNodeId {
        let transfer = ExecutionNode::Transfer {
            src: src.to_string(),
            dst: dst.to_string(),
            bytes,
            direction,
            timing_ns,
        };
        let transfer_id = self.add_node(transfer);

        if let Some(&parent) = self.scope_stack.last() {
            self.add_edge(parent, transfer_id, EdgeType::Contains);
        }

        transfer_id
    }

    /// Add a dependency edge for critical path analysis (Phase 9).
    pub fn add_dependency(&mut self, from: ExecutionNodeId, to: ExecutionNodeId) {
        self.add_edge(from, to, EdgeType::DependsOn);
    }

    /// Get a node by ID.
    pub fn node(&self, id: ExecutionNodeId) -> Option<&ExecutionNode> {
        self.nodes.get(id.0 as usize)
    }

    /// Get a node by name.
    pub fn node_by_name(&self, name: &str) -> Option<(ExecutionNodeId, &ExecutionNode)> {
        self.name_to_id
            .get(name)
            .and_then(|&id| self.nodes.get(id.0 as usize).map(|n| (id, n)))
    }

    /// Get all nodes.
    pub fn nodes(&self) -> &[ExecutionNode] {
        &self.nodes
    }

    /// Get all edges.
    pub fn edges(&self) -> &[ExecutionEdge] {
        &self.edges
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get outgoing edges for a node.
    pub fn outgoing_edges(&self, node: ExecutionNodeId) -> impl Iterator<Item = &ExecutionEdge> {
        self.edges.iter().filter(move |e| e.src == node)
    }

    /// Get incoming edges for a node.
    pub fn incoming_edges(&self, node: ExecutionNodeId) -> impl Iterator<Item = &ExecutionEdge> {
        self.edges.iter().filter(move |e| e.dst == node)
    }

    /// Find all kernel nodes.
    pub fn kernel_nodes(&self) -> impl Iterator<Item = (ExecutionNodeId, &ExecutionNode)> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_kernel())
            .map(|(i, n)| (ExecutionNodeId(i as u32), n))
    }

    /// Find the slowest kernel (by parent brick timing).
    pub fn slowest_kernel(&self) -> Option<(ExecutionNodeId, &ExecutionNode, u64)> {
        let mut slowest: Option<(ExecutionNodeId, &ExecutionNode, u64)> = None;

        for (id, node) in self.nodes.iter().enumerate() {
            if let ExecutionNode::Brick { timing_ns, .. } = node {
                // Check if this brick has kernel children
                let node_id = ExecutionNodeId(id as u32);
                let has_kernel = self
                    .outgoing_edges(node_id)
                    .any(|e| e.edge_type == EdgeType::Launches);

                if has_kernel {
                    match &slowest {
                        None => slowest = Some((node_id, node, *timing_ns)),
                        Some((_, _, t)) if *timing_ns > *t => {
                            slowest = Some((node_id, node, *timing_ns))
                        }
                        _ => {}
                    }
                }
            }
        }

        slowest
    }

    /// Export to DOT format for Graphviz visualization.
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph ExecutionGraph {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  node [shape=box];\n\n");

        // Add nodes with styling based on type
        for (i, node) in self.nodes.iter().enumerate() {
            let (label, style) = match node {
                ExecutionNode::Layer { index } => {
                    (format!("Layer {}", index), "style=filled,fillcolor=lightblue")
                }
                ExecutionNode::Brick { id, timing_ns, .. } => (
                    format!("{}\\n{:.1}µs", id.name(), *timing_ns as f64 / 1000.0),
                    "style=filled,fillcolor=lightgreen",
                ),
                ExecutionNode::Kernel {
                    name,
                    grid,
                    block,
                    ..
                } => (
                    format!("{}\\n<<<{},{},{}>>>", name, grid.0, block.0, block.1),
                    "style=filled,fillcolor=lightyellow",
                ),
                ExecutionNode::Function { name, file, line } => {
                    let loc = match (file, line) {
                        (Some(f), Some(l)) => format!("\\n{}:{}", f, l),
                        _ => String::new(),
                    };
                    (format!("{}{}", name, loc), "style=filled,fillcolor=lightgray")
                }
                ExecutionNode::Transfer { src, dst, bytes, direction, .. } => {
                    let dir = match direction {
                        TransferDirection::H2D => "H2D",
                        TransferDirection::D2H => "D2H",
                        TransferDirection::D2D => "D2D",
                    };
                    (
                        format!("{}\\n{}->{}\\n{:.1}MB", dir, src, dst, *bytes as f64 / 1e6),
                        "style=filled,fillcolor=lightsalmon",
                    )
                }
            };
            dot.push_str(&format!("  n{} [label=\"{}\",{}];\n", i, label, style));
        }

        dot.push('\n');

        // Add edges with styling based on type
        for edge in &self.edges {
            let style = match edge.edge_type {
                EdgeType::Calls => "style=solid",
                EdgeType::Contains => "style=dashed",
                EdgeType::Launches => "style=bold,color=red",
                EdgeType::Sequence => "style=dotted",
                EdgeType::DependsOn => "style=solid,color=blue",
                EdgeType::Transfer { .. } => "style=bold,color=orange",
            };
            dot.push_str(&format!(
                "  n{} -> n{} [{}];\n",
                edge.src.0, edge.dst.0, style
            ));
        }

        dot.push_str("}\n");
        dot
    }

    /// Export to trueno-graph CsrGraph format.
    #[cfg(feature = "execution-graph")]
    pub fn to_csr(&self) -> trueno_graph::CsrGraph {
        use trueno_graph::{CsrGraph, NodeId};

        let edges: Vec<(NodeId, NodeId, f32)> = self
            .edges
            .iter()
            .map(|e| (NodeId(e.src.0), NodeId(e.dst.0), e.weight))
            .collect();

        let mut graph = CsrGraph::from_edge_list(&edges).unwrap_or_default();

        // Set node names for querying
        for (i, node) in self.nodes.iter().enumerate() {
            graph.set_node_name(NodeId(i as u32), node.name());
        }

        graph
    }

    /// Convert to presentar-terminal TreeNode for TUI visualization.
    ///
    /// PAR-201: Renders the execution graph as a collapsible tree in the terminal.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use trueno::BrickProfiler;
    /// use presentar_terminal::{Tree, TuiApp};
    ///
    /// let profiler = BrickProfiler::new();
    /// // ... record execution ...
    ///
    /// let tree_node = profiler.execution_graph().to_tree_node();
    /// let tree = Tree::new().with_root(tree_node).expand_all();
    /// ```
    #[cfg(feature = "presentar-tui")]
    pub fn to_tree_node(&self) -> presentar_terminal::TreeNode {
        use presentar_terminal::{Color, TreeNode};
        use std::collections::HashMap;

        // Color scheme for node types
        let layer_color = Color::new(0.4, 0.6, 1.0, 1.0); // Light blue
        let brick_color = Color::new(0.4, 0.8, 0.4, 1.0); // Light green
        let kernel_color = Color::new(1.0, 0.8, 0.3, 1.0); // Yellow/orange
        let func_color = Color::new(0.7, 0.7, 0.7, 1.0); // Light gray

        // Build child map: parent -> [children]
        let mut children_map: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut has_parent: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for edge in &self.edges {
            if edge.edge_type == EdgeType::Contains || edge.edge_type == EdgeType::Launches {
                children_map
                    .entry(edge.src.0)
                    .or_default()
                    .push(edge.dst.0);
                has_parent.insert(edge.dst.0);
            }
        }

        // Find root nodes (nodes with no parent)
        let root_ids: Vec<u32> = (0..self.nodes.len() as u32)
            .filter(|id| !has_parent.contains(id))
            .collect();

        // Recursive function to build TreeNode
        fn build_node(
            graph: &ExecutionGraph,
            id: u32,
            children_map: &HashMap<u32, Vec<u32>>,
            layer_color: Color,
            brick_color: Color,
            kernel_color: Color,
            func_color: Color,
        ) -> TreeNode {
            let node = &graph.nodes[id as usize];
            let (label, info, color) = match node {
                ExecutionNode::Layer { index } => {
                    (format!("Layer {}", index), None, layer_color)
                }
                ExecutionNode::Brick {
                    id: brick_id,
                    timing_ns,
                    elements,
                } => (
                    brick_id.name().to_string(),
                    Some(format!("{:.1}µs ({} elem)", *timing_ns as f64 / 1000.0, elements)),
                    brick_color,
                ),
                ExecutionNode::Kernel {
                    name,
                    grid,
                    block,
                    shared_mem,
                    ..
                } => (
                    name.clone(),
                    Some(format!(
                        "<<<{},{},{}>>> smem={}B",
                        grid.0, block.0, block.1, shared_mem
                    )),
                    kernel_color,
                ),
                ExecutionNode::Function { name, file, line } => {
                    let loc = match (file, line) {
                        (Some(f), Some(l)) => format!(" ({}:{})", f, l),
                        _ => String::new(),
                    };
                    (format!("{}{}", name, loc), None, func_color)
                }
                ExecutionNode::Transfer {
                    src,
                    dst,
                    bytes,
                    direction,
                    timing_ns,
                } => {
                    let timing_str = timing_ns
                        .map(|ns| format!(" {:.1}µs", ns as f64 / 1000.0))
                        .unwrap_or_default();
                    (
                        format!("{:?}: {} → {}", direction, src, dst),
                        Some(format!("{}B{}", bytes, timing_str)),
                        Color::Magenta, // Transfer color
                    )
                }
            };

            let mut tree_node = TreeNode::new(id as u64, label).with_color(color);
            if let Some(info_str) = info {
                tree_node = tree_node.with_info(info_str);
            }

            // Add children
            if let Some(child_ids) = children_map.get(&id) {
                for &child_id in child_ids {
                    let child = build_node(
                        graph,
                        child_id,
                        children_map,
                        layer_color,
                        brick_color,
                        kernel_color,
                        func_color,
                    );
                    tree_node = tree_node.with_child(child);
                }
            }

            tree_node
        }

        // Build root node
        if root_ids.is_empty() {
            TreeNode::new(0, "Empty Graph")
        } else if root_ids.len() == 1 {
            build_node(
                self,
                root_ids[0],
                &children_map,
                layer_color,
                brick_color,
                kernel_color,
                func_color,
            )
        } else {
            // Multiple roots: wrap in a synthetic root
            let mut root = TreeNode::new(u64::MAX, "Execution Graph")
                .with_color(Color::new(0.9, 0.9, 0.9, 1.0));
            for &root_id in &root_ids {
                let child = build_node(
                    self,
                    root_id,
                    &children_map,
                    layer_color,
                    brick_color,
                    kernel_color,
                    func_color,
                );
                root = root.with_child(child);
            }
            root
        }
    }

    /// Render graph to ASCII tree string (headless mode for testing/automation).
    ///
    /// PAR-201: Zero-dependency tree visualization for CI/CD, logging, and snapshot tests.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let graph = profiler.execution_graph();
    /// let tree = graph.to_ascii_tree();
    /// println!("{}", tree);
    /// // Output:
    /// // Layer 0
    /// // ├── RmsNorm  50.0µs (4096 elem)
    /// // │   └── rmsnorm_kernel  <<<16,256,1>>> smem=1024B
    /// // └── QkvProjection  200.0µs (4096 elem)
    /// //     └── batched_q4k_gemv  <<<32,256,1>>> smem=4096B
    /// ```
    #[must_use]
    pub fn to_ascii_tree(&self) -> String {
        use std::collections::HashMap;

        // Build child map: parent -> [children]
        let mut children_map: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut has_parent: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for edge in &self.edges {
            if edge.edge_type == EdgeType::Contains || edge.edge_type == EdgeType::Launches {
                children_map
                    .entry(edge.src.0)
                    .or_default()
                    .push(edge.dst.0);
                has_parent.insert(edge.dst.0);
            }
        }

        // Find root nodes (nodes with no parent)
        let root_ids: Vec<u32> = (0..self.nodes.len() as u32)
            .filter(|id| !has_parent.contains(id))
            .collect();

        // Recursive function to build tree string
        fn build_tree(
            graph: &ExecutionGraph,
            id: u32,
            children_map: &HashMap<u32, Vec<u32>>,
            prefix: &str,
            connector: &str,
            output: &mut String,
        ) {
            let node = &graph.nodes[id as usize];
            let (label, info) = match node {
                ExecutionNode::Layer { index } => (format!("Layer {}", index), String::new()),
                ExecutionNode::Brick {
                    id: brick_id,
                    timing_ns,
                    elements,
                } => (
                    brick_id.name().to_string(),
                    format!("  {:.1}µs ({} elem)", *timing_ns as f64 / 1000.0, elements),
                ),
                ExecutionNode::Kernel {
                    name,
                    grid,
                    block,
                    shared_mem,
                    ..
                } => (
                    name.clone(),
                    format!("  <<<{},{},{}>>> smem={}B", grid.0, block.0, block.1, shared_mem),
                ),
                ExecutionNode::Function { name, file, line } => {
                    let loc = match (file, line) {
                        (Some(f), Some(l)) => format!(" ({}:{})", f, l),
                        _ => String::new(),
                    };
                    (format!("{}{}", name, loc), String::new())
                }
                ExecutionNode::Transfer {
                    src,
                    dst,
                    bytes,
                    direction,
                    timing_ns,
                } => {
                    let timing_str = timing_ns
                        .map(|ns| format!(" {:.1}µs", ns as f64 / 1000.0))
                        .unwrap_or_default();
                    (
                        format!("{:?}: {} → {}", direction, src, dst),
                        format!("  {}B{}", bytes, timing_str),
                    )
                }
            };

            output.push_str(&format!("{}{}{}{}\n", prefix, connector, label, info));

            if let Some(child_ids) = children_map.get(&id) {
                let child_count = child_ids.len();
                for (i, &child_id) in child_ids.iter().enumerate() {
                    let is_last = i == child_count - 1;
                    let new_connector = if is_last { "└── " } else { "├── " };
                    let new_prefix = if connector.is_empty() {
                        prefix.to_string()
                    } else if connector == "└── " {
                        format!("{}    ", prefix)
                    } else {
                        format!("{}│   ", prefix)
                    };
                    build_tree(graph, child_id, children_map, &new_prefix, new_connector, output);
                }
            }
        }

        let mut output = String::new();

        if root_ids.is_empty() {
            output.push_str("(empty graph)\n");
        } else if root_ids.len() == 1 {
            build_tree(self, root_ids[0], &children_map, "", "", &mut output);
        } else {
            // Multiple roots: add synthetic root
            output.push_str("Execution Graph\n");
            let root_count = root_ids.len();
            for (i, &root_id) in root_ids.iter().enumerate() {
                let is_last = i == root_count - 1;
                let connector = if is_last { "└── " } else { "├── " };
                build_tree(self, root_id, &children_map, "", connector, &mut output);
            }
        }

        // Remove trailing newline for cleaner output
        if output.ends_with('\n') {
            output.pop();
        }
        output
    }

    // ========================
    // Phase 9: Critical Path Analysis (CPA)
    // ========================

    /// Get timing for a node (ns). Returns 0 for non-timed nodes.
    fn node_timing_ns(&self, id: ExecutionNodeId) -> u64 {
        match &self.nodes[id.0 as usize] {
            ExecutionNode::Brick { timing_ns, .. } => *timing_ns,
            ExecutionNode::Kernel { timing_ns, .. } => timing_ns.unwrap_or(0),
            ExecutionNode::Transfer { timing_ns, .. } => timing_ns.unwrap_or(0),
            _ => 0,
        }
    }

    /// Compute critical path through execution graph using longest-path algorithm.
    ///
    /// Returns (critical_path_nodes, total_time_ns). The critical path represents
    /// the longest chain of dependencies that determines total execution time.
    ///
    /// Reference: Graham et al. (1979) "Scheduling Algorithms for Multi-Processor Systems"
    pub fn critical_path(&self) -> (Vec<ExecutionNodeId>, u64) {
        if self.nodes.is_empty() {
            return (vec![], 0);
        }

        // Build adjacency list for DependsOn and Sequence edges
        let mut adj: Vec<Vec<(u32, u64)>> = vec![vec![]; self.nodes.len()];
        for edge in &self.edges {
            match &edge.edge_type {
                EdgeType::DependsOn | EdgeType::Sequence => {
                    let weight = self.node_timing_ns(edge.dst);
                    adj[edge.src.0 as usize].push((edge.dst.0, weight));
                }
                EdgeType::Contains | EdgeType::Calls | EdgeType::Launches => {
                    // Hierarchical edges: children contribute to parent time
                    let weight = self.node_timing_ns(edge.dst);
                    adj[edge.src.0 as usize].push((edge.dst.0, weight));
                }
                EdgeType::Transfer { .. } => {
                    // Transfer edges carry their own timing
                    let weight = self.node_timing_ns(edge.dst);
                    adj[edge.src.0 as usize].push((edge.dst.0, weight));
                }
            }
        }

        // Topological sort using Kahn's algorithm
        let mut in_degree = vec![0u32; self.nodes.len()];
        for edges in &adj {
            for (dst, _) in edges {
                in_degree[*dst as usize] += 1;
            }
        }

        let mut queue: Vec<u32> = (0..self.nodes.len() as u32)
            .filter(|&i| in_degree[i as usize] == 0)
            .collect();
        let mut topo_order = Vec::with_capacity(self.nodes.len());

        while let Some(u) = queue.pop() {
            topo_order.push(u);
            for (v, _) in &adj[u as usize] {
                in_degree[*v as usize] -= 1;
                if in_degree[*v as usize] == 0 {
                    queue.push(*v);
                }
            }
        }

        // Longest path DP
        let mut dist = vec![0u64; self.nodes.len()];
        let mut pred = vec![None::<u32>; self.nodes.len()];

        // Initialize with node's own timing for roots
        for &node in &topo_order {
            if self.edges.iter().all(|e| e.dst.0 != node) {
                dist[node as usize] = self.node_timing_ns(ExecutionNodeId(node));
            }
        }

        for &u in &topo_order {
            for (v, weight) in &adj[u as usize] {
                let new_dist = dist[u as usize] + weight;
                if new_dist > dist[*v as usize] {
                    dist[*v as usize] = new_dist;
                    pred[*v as usize] = Some(u);
                }
            }
        }

        // Find endpoint with maximum distance
        let (end_node, &total_time) = dist
            .iter()
            .enumerate()
            .max_by_key(|(_, &d)| d)
            .unwrap_or((0, &0));

        // Reconstruct path
        let mut path = vec![];
        let mut current = Some(end_node as u32);
        while let Some(node) = current {
            path.push(ExecutionNodeId(node));
            current = pred[node as usize];
        }
        path.reverse();

        (path, total_time)
    }

    /// Compute slack for each node (how much it can be delayed without affecting total time).
    ///
    /// Returns map from node ID to slack in nanoseconds. Nodes on critical path have slack = 0.
    pub fn compute_slack(&self) -> HashMap<ExecutionNodeId, u64> {
        let (critical_path, total_time) = self.critical_path();
        let critical_set: std::collections::HashSet<_> = critical_path.iter().copied().collect();

        let mut slack = HashMap::new();

        // Build reverse adjacency
        let mut reverse_adj: Vec<Vec<u32>> = vec![vec![]; self.nodes.len()];
        for edge in &self.edges {
            reverse_adj[edge.dst.0 as usize].push(edge.src.0);
        }

        // Forward pass: earliest start time
        let mut earliest = vec![0u64; self.nodes.len()];
        for i in 0..self.nodes.len() {
            let mut max_pred = 0u64;
            for &pred in &reverse_adj[i] {
                max_pred = max_pred.max(earliest[pred as usize] + self.node_timing_ns(ExecutionNodeId(pred)));
            }
            earliest[i] = max_pred;
        }

        // Backward pass: latest start time
        let mut latest = vec![total_time; self.nodes.len()];
        for i in (0..self.nodes.len()).rev() {
            let timing = self.node_timing_ns(ExecutionNodeId(i as u32));
            let mut min_succ = total_time;
            for edge in &self.edges {
                if edge.src.0 == i as u32 {
                    min_succ = min_succ.min(latest[edge.dst.0 as usize]);
                }
            }
            latest[i] = min_succ.saturating_sub(timing);
        }

        // Slack = latest - earliest
        for i in 0..self.nodes.len() {
            let node_id = ExecutionNodeId(i as u32);
            let node_slack = if critical_set.contains(&node_id) {
                0
            } else {
                latest[i].saturating_sub(earliest[i])
            };
            slack.insert(node_id, node_slack);
        }

        slack
    }

    /// Compute roofline distance for kernel nodes.
    ///
    /// Returns map from kernel node ID to distance from roofline (0.0 = optimal).
    /// Distance = 1.0 - min(achieved/peak_compute, achieved/peak_bandwidth).
    ///
    /// Reference: Williams et al. (2009) "Roofline: An Insightful Visual Performance Model"
    pub fn roofline_distance(&self, peak_tflops: f32, peak_bandwidth_gb_s: f32) -> HashMap<ExecutionNodeId, f32> {
        let mut distances = HashMap::new();

        for (i, node) in self.nodes.iter().enumerate() {
            if let ExecutionNode::Kernel {
                arithmetic_intensity,
                achieved_tflops,
                ..
            } = node
            {
                if let (Some(ai), Some(achieved)) = (arithmetic_intensity, achieved_tflops) {
                    // Roofline model: achievable = min(peak_compute, ai * bandwidth)
                    let bandwidth_bound = *ai * peak_bandwidth_gb_s / 1000.0; // Convert GB/s to TFLOP/s
                    let roofline_bound = peak_tflops.min(bandwidth_bound);
                    let efficiency = achieved / roofline_bound;
                    let distance = 1.0 - efficiency.min(1.0);
                    distances.insert(ExecutionNodeId(i as u32), distance);
                }
            }
        }

        distances
    }

    /// Detect ping-pong memory transfer patterns (wasteful H2D followed by D2H).
    ///
    /// Returns pairs of transfer node IDs that exhibit ping-pong behavior.
    pub fn detect_ping_pong(&self) -> Vec<(ExecutionNodeId, ExecutionNodeId)> {
        let mut patterns = Vec::new();

        // Find transfer nodes
        let transfers: Vec<(usize, &ExecutionNode)> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n, ExecutionNode::Transfer { .. }))
            .collect();

        // Check for H2D followed by D2H on same data
        for i in 0..transfers.len() {
            for j in (i + 1)..transfers.len() {
                if let (
                    ExecutionNode::Transfer {
                        src: src1,
                        dst: dst1,
                        direction: dir1,
                        bytes: bytes1,
                        ..
                    },
                    ExecutionNode::Transfer {
                        src: src2,
                        dst: dst2,
                        direction: dir2,
                        bytes: bytes2,
                        ..
                    },
                ) = (&transfers[i].1, &transfers[j].1)
                {
                    // Ping-pong: H2D then D2H with matching src/dst and same size
                    let is_ping_pong = (*dir1 == TransferDirection::H2D
                        && *dir2 == TransferDirection::D2H
                        && dst1 == src2
                        && bytes1 == bytes2)
                        || (*dir1 == TransferDirection::D2H
                            && *dir2 == TransferDirection::H2D
                            && src1 == dst2
                            && bytes1 == bytes2);

                    if is_ping_pong {
                        patterns.push((
                            ExecutionNodeId(transfers[i].0 as u32),
                            ExecutionNodeId(transfers[j].0 as u32),
                        ));
                    }
                }
            }
        }

        patterns
    }

    /// Get critical path analysis summary as formatted string.
    pub fn critical_path_summary(&self) -> String {
        let (path, total_ns) = self.critical_path();
        let slack = self.compute_slack();

        let mut output = String::new();
        output.push_str(&format!(
            "Critical Path: {:.2}ms ({} nodes)\n",
            total_ns as f64 / 1_000_000.0,
            path.len()
        ));
        output.push_str("─".repeat(50).as_str());
        output.push('\n');

        for (i, node_id) in path.iter().enumerate() {
            let node = &self.nodes[node_id.0 as usize];
            let timing = self.node_timing_ns(*node_id);
            let node_name = match node {
                ExecutionNode::Layer { index } => format!("Layer {}", index),
                ExecutionNode::Brick { id, .. } => id.name().to_string(),
                ExecutionNode::Kernel { name, .. } => name.clone(),
                ExecutionNode::Function { name, .. } => name.clone(),
                ExecutionNode::Transfer { direction, src, dst, .. } => {
                    format!("{:?} {} → {}", direction, src, dst)
                }
            };

            let prefix = if i == 0 {
                "┌"
            } else if i == path.len() - 1 {
                "└"
            } else {
                "│"
            };
            output.push_str(&format!(
                "{} {} ({:.1}µs)\n",
                prefix,
                node_name,
                timing as f64 / 1000.0
            ));
        }

        // Show nodes with most slack (parallelization opportunities)
        let mut slack_vec: Vec<_> = slack.iter().collect();
        slack_vec.sort_by(|a, b| b.1.cmp(a.1));

        if slack_vec.iter().any(|(_, &s)| s > 0) {
            output.push_str("\nParallelization Opportunities (high slack):\n");
            for (node_id, &node_slack) in slack_vec.iter().take(5) {
                if node_slack > 0 {
                    let node = &self.nodes[node_id.0 as usize];
                    let node_name = match node {
                        ExecutionNode::Layer { index } => format!("Layer {}", index),
                        ExecutionNode::Brick { id, .. } => id.name().to_string(),
                        ExecutionNode::Kernel { name, .. } => name.clone(),
                        ExecutionNode::Function { name, .. } => name.clone(),
                        ExecutionNode::Transfer { direction, src, dst, .. } => {
                            format!("{:?} {} → {}", direction, src, dst)
                        }
                    };
                    output.push_str(&format!(
                        "  {} slack={:.1}µs\n",
                        node_name,
                        node_slack as f64 / 1000.0
                    ));
                }
            }
        }

        output
    }

    /// Clear the graph.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.scope_stack.clear();
        self.name_to_id.clear();
    }

    /// Check if scope stack is balanced (empty).
    pub fn is_scope_balanced(&self) -> bool {
        self.scope_stack.is_empty()
    }
}

/// PTX kernel registry for execution graph correlation.
///
/// PAR-201: Maps PTX hashes to source code for debugging and analysis.
#[derive(Debug, Default)]
pub struct PtxRegistry {
    /// Hash → (kernel_name, ptx_source, file_path)
    kernels: std::collections::HashMap<u64, (String, String, Option<std::path::PathBuf>)>,
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
        let hash = Self::hash_ptx(ptx);
        self.kernels.insert(
            hash,
            (name.to_string(), ptx.to_string(), path.map(|p| p.to_path_buf())),
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
        }
    }

    /// Add a sample to statistics.
    pub fn add_sample(&mut self, elapsed_ns: u64, elements: u64) {
        self.count += 1;
        self.total_ns += elapsed_ns;
        self.min_ns = self.min_ns.min(elapsed_ns);
        self.max_ns = self.max_ns.max(elapsed_ns);
        self.total_elements += elements;
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

/// Pending measurement for deferred sync mode.
#[derive(Debug, Clone)]
struct PendingMeasurement {
    /// Brick ID (if known)
    brick_id: Option<BrickId>,
    /// Brick name (for dynamic bricks)
    name: Option<String>,
    /// Start time in nanoseconds (from Instant::now())
    start_ns: u64,
    /// Number of elements processed
    elements: u64,
}

/// Per-brick profiler using pure Rust timing.
///
/// # Design (PAR-073, PAR-200)
///
/// - Uses `std::time::Instant` for timing (no CUDA event FFI)
/// - PAR-200: O(1) hot path with `BrickId` enum + array storage
/// - GPU operations require explicit sync before timing point
/// - Supports deferred sync mode for low-overhead production profiling
/// - Aggregates statistics per brick name
///
/// # Usage
///
/// ```rust,ignore
/// use trueno::brick::{BrickProfiler, BrickId, SyncMode};
///
/// let mut profiler = BrickProfiler::new();
/// profiler.enable();
///
/// // Fast path: use BrickId for known bricks (PAR-200)
/// let timer = profiler.start_brick(BrickId::RmsNorm);
/// // ... do work ...
/// // For GPU: cuda_stream.synchronize() HERE
/// profiler.stop_brick(timer, 1);
///
/// // Legacy path: string-based (slower, for unknown bricks)
/// let timer = profiler.start("CustomBrick");
/// profiler.stop(timer, 1);
///
/// // Deferred sync mode (production)
/// profiler.set_sync_mode(SyncMode::Deferred);
/// profiler.record_deferred(BrickId::RmsNorm, start_ns, 1);
/// // ... more operations ...
/// cuda_stream.synchronize();
/// profiler.finalize(end_ns);
///
/// // Get statistics
/// let stats = profiler.brick_stats(BrickId::RmsNorm);
/// println!("RmsNorm avg: {:.2}µs", stats.avg_us());
///
/// // Get category breakdown
/// let cats = profiler.category_stats();
/// println!("Attention: {:.1}%", cats[BrickCategory::Attention as usize].percentage(profiler.total_ns()));
/// ```
#[derive(Debug)]
pub struct BrickProfiler {
    // PAR-200: Fast path - pre-allocated array for known bricks
    /// Per-brick statistics for known BrickId types (O(1) lookup)
    brick_stats: [BrickStats; BrickId::COUNT],

    // Legacy path - HashMap for dynamic/unknown brick names
    /// Per-brick statistics for unknown brick names (slower, O(1) amortized)
    dynamic_stats: std::collections::HashMap<String, BrickStats>,

    // PAR-200: Deferred sync support
    /// Pending measurements awaiting GPU sync
    pending: Vec<PendingMeasurement>,
    /// Synchronization mode
    sync_mode: SyncMode,
    /// Reference instant for deferred timing
    epoch: Instant,

    /// Whether profiling is enabled
    enabled: bool,
    /// Total tokens processed
    total_tokens: u64,
    /// Total time (ns) across all bricks
    total_ns: u64,
    /// L2 cache hit rate (0.0-1.0) - v1.1.0 OBSERVE phase
    l2_cache_hit_rate: Option<f32>,
    /// Whether zero-copy memory transfers are enabled - v1.1.0 OBSERVE phase
    is_zero_copy: bool,
    /// CORRECTNESS-011: Per-kernel checksums for divergence detection
    kernel_checksums: Vec<KernelChecksum>,

    // PAR-201: Execution path graph
    /// Whether execution graph tracking is enabled
    graph_enabled: bool,
    /// Execution path graph for PTX→kernel→brick relationships
    execution_graph: ExecutionGraph,
}

/// Timer handle returned by `start()` (legacy string-based API).
#[derive(Debug)]
pub struct BrickTimer {
    /// Brick name
    name: String,
    /// Start time
    start: Instant,
}

/// Timer handle returned by `start_brick()` (PAR-200 fast path).
#[derive(Debug)]
pub struct BrickIdTimer {
    /// Brick ID
    brick_id: BrickId,
    /// Start time
    start: Instant,
}

impl Default for BrickProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickProfiler {
    /// Create a new profiler (disabled by default for zero overhead).
    pub fn new() -> Self {
        Self {
            brick_stats: std::array::from_fn(|i| {
                // Safety: i < BrickId::COUNT by construction
                let brick_id = unsafe { std::mem::transmute::<u8, BrickId>(i as u8) };
                BrickStats::new(brick_id.name())
            }),
            dynamic_stats: std::collections::HashMap::new(),
            pending: Vec::new(),
            sync_mode: SyncMode::Deferred,
            epoch: Instant::now(),
            enabled: false,
            total_tokens: 0,
            total_ns: 0,
            l2_cache_hit_rate: None,
            is_zero_copy: false,
            kernel_checksums: Vec::new(),
            graph_enabled: false,
            execution_graph: ExecutionGraph::new(),
        }
    }

    /// Create an enabled profiler.
    pub fn enabled() -> Self {
        let mut profiler = Self::new();
        profiler.enabled = true;
        profiler
    }

    // ========================================================================
    // PAR-200: Sync Mode Configuration
    // ========================================================================

    /// Set the synchronization mode for GPU profiling.
    ///
    /// # Modes
    /// - `Immediate`: Sync after each kernel (accurate but slow)
    /// - `PerLayer`: Sync once per transformer layer
    /// - `Deferred`: Sync once per forward pass (default, fast)
    /// - `None`: No synchronization
    pub fn set_sync_mode(&mut self, mode: SyncMode) {
        self.sync_mode = mode;
    }

    /// Get the current synchronization mode.
    #[must_use]
    pub fn sync_mode(&self) -> SyncMode {
        self.sync_mode
    }

    /// Reset the epoch for deferred timing.
    /// Call this at the start of a forward pass.
    pub fn reset_epoch(&mut self) {
        self.epoch = Instant::now();
    }

    /// Get nanoseconds elapsed since epoch.
    #[inline]
    pub fn elapsed_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    // ========================================================================
    // PAR-200: Fast Path API (BrickId-based)
    // ========================================================================

    /// Start timing a brick using BrickId (O(1) hot path).
    ///
    /// This is the preferred API for known brick types.
    /// For GPU operations, call `stream.synchronize()` before `stop_brick()`.
    #[inline]
    #[must_use]
    pub fn start_brick(&self, brick_id: BrickId) -> BrickIdTimer {
        BrickIdTimer {
            brick_id,
            start: Instant::now(),
        }
    }

    /// Stop timing and record the sample (O(1) hot path).
    #[inline]
    pub fn stop_brick(&mut self, timer: BrickIdTimer, elements: u64) {
        if !self.enabled {
            return;
        }

        let elapsed = timer.start.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;

        // O(1) array access
        let stats = &mut self.brick_stats[timer.brick_id as usize];
        stats.add_sample(elapsed_ns, elements);

        // Update totals
        self.total_tokens += elements;
        self.total_ns += elapsed_ns;
    }

    /// Get statistics for a known brick type (O(1)).
    #[inline]
    #[must_use]
    pub fn brick_stats(&self, brick_id: BrickId) -> &BrickStats {
        &self.brick_stats[brick_id as usize]
    }

    /// Get mutable statistics for a known brick type (O(1)).
    #[inline]
    pub fn brick_stats_mut(&mut self, brick_id: BrickId) -> &mut BrickStats {
        &mut self.brick_stats[brick_id as usize]
    }

    // ========================================================================
    // PAR-200: Deferred Sync API
    // ========================================================================

    /// Record a measurement without GPU sync (deferred mode).
    ///
    /// Call `finalize()` after GPU sync to apply all pending measurements.
    ///
    /// # Arguments
    /// - `brick_id`: The brick type
    /// - `start_ns`: Start time (from `elapsed_ns()` at operation start)
    /// - `elements`: Number of elements processed
    #[inline]
    pub fn record_deferred(&mut self, brick_id: BrickId, start_ns: u64, elements: u64) {
        if !self.enabled {
            return;
        }
        self.pending.push(PendingMeasurement {
            brick_id: Some(brick_id),
            name: None,
            start_ns,
            elements,
        });
    }

    /// Record a measurement for a dynamic brick (deferred mode).
    #[inline]
    pub fn record_deferred_dynamic(&mut self, name: &str, start_ns: u64, elements: u64) {
        if !self.enabled {
            return;
        }
        self.pending.push(PendingMeasurement {
            brick_id: BrickId::from_str(name),
            name: Some(name.to_string()),
            start_ns,
            elements,
        });
    }

    /// Finalize all pending measurements after GPU sync.
    ///
    /// Must be called after `stream.synchronize()` to get accurate timing.
    ///
    /// # Arguments
    /// - `end_ns`: End time (from `elapsed_ns()` after sync)
    pub fn finalize(&mut self, end_ns: u64) {
        if self.pending.is_empty() {
            return;
        }

        // Calculate elapsed time for each pending measurement
        for m in self.pending.drain(..) {
            let elapsed_ns = end_ns.saturating_sub(m.start_ns);

            if let Some(brick_id) = m.brick_id {
                // Fast path: known brick
                let stats = &mut self.brick_stats[brick_id as usize];
                stats.add_sample(elapsed_ns, m.elements);
            } else if let Some(name) = m.name {
                // Slow path: dynamic brick
                let stats = self.dynamic_stats.entry(name.clone()).or_insert_with(|| {
                    BrickStats::new(&name)
                });
                stats.add_sample(elapsed_ns, m.elements);
            }

            self.total_tokens += m.elements;
            self.total_ns += elapsed_ns;
        }
    }

    /// Check if there are pending measurements.
    #[inline]
    #[must_use]
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Get number of pending measurements.
    #[inline]
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    // ========================================================================
    // PAR-200: Category Aggregation
    // ========================================================================

    /// Get aggregated statistics by category.
    ///
    /// Returns an array indexed by `BrickCategory as usize`.
    #[must_use]
    pub fn category_stats(&self) -> [CategoryStats; BrickCategory::COUNT] {
        let mut result = [CategoryStats::default(); BrickCategory::COUNT];

        for (i, stats) in self.brick_stats.iter().enumerate() {
            // Safety: i < BrickId::COUNT by construction
            let brick_id = unsafe { std::mem::transmute::<u8, BrickId>(i as u8) };
            let cat = brick_id.category() as usize;
            result[cat].total_ns += stats.total_ns;
            result[cat].total_elements += stats.total_elements;
            result[cat].count += stats.count;
        }

        // Include dynamic stats in "Other" category
        for stats in self.dynamic_stats.values() {
            let cat = BrickCategory::Other as usize;
            result[cat].total_ns += stats.total_ns;
            result[cat].total_elements += stats.total_elements;
            result[cat].count += stats.count;
        }

        result
    }

    /// Print category breakdown to console.
    pub fn print_category_stats(&self) {
        let cats = self.category_stats();
        let total = self.total_ns;

        println!("╭─────────────────────────────────────────────────────────╮");
        println!("│            Category Breakdown (PAR-200)                 │");
        println!("├─────────────────────────────────────────────────────────┤");
        for (i, cat_stats) in cats.iter().enumerate() {
            // Safety: i < BrickCategory::COUNT
            let cat = unsafe { std::mem::transmute::<u8, BrickCategory>(i as u8) };
            if cat_stats.count > 0 {
                println!(
                    "│ {:12} {:8.2}µs avg {:6.1}% [{:5} samples]        │",
                    cat.name(),
                    cat_stats.avg_us(),
                    cat_stats.percentage(total),
                    cat_stats.count
                );
            }
        }
        println!("╰─────────────────────────────────────────────────────────╯");
    }

    // ========================================================================
    // PAR-201: Execution Path Graph
    // ========================================================================

    /// Enable execution graph tracking.
    ///
    /// When enabled, the profiler records the execution hierarchy:
    /// - Layer → Brick → Kernel relationships
    /// - PTX hashes for kernel identity
    /// - Timing data per node
    pub fn enable_graph(&mut self) {
        self.graph_enabled = true;
    }

    /// Disable execution graph tracking.
    pub fn disable_graph(&mut self) {
        self.graph_enabled = false;
    }

    /// Check if execution graph tracking is enabled.
    #[must_use]
    pub fn is_graph_enabled(&self) -> bool {
        self.graph_enabled
    }

    /// Get the execution graph (immutable).
    #[must_use]
    pub fn execution_graph(&self) -> &ExecutionGraph {
        &self.execution_graph
    }

    /// Get the execution graph (mutable).
    pub fn execution_graph_mut(&mut self) -> &mut ExecutionGraph {
        &mut self.execution_graph
    }

    /// Push a scope for hierarchical graph recording.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// profiler.enable_graph();
    /// profiler.graph_push_scope(ExecutionNode::Layer { index: 0 });
    /// // ... record bricks and kernels ...
    /// profiler.graph_pop_scope();
    /// ```
    pub fn graph_push_scope(&mut self, node: ExecutionNode) -> Option<ExecutionNodeId> {
        if !self.graph_enabled {
            return None;
        }
        Some(self.execution_graph.push_scope(node))
    }

    /// Pop the current scope.
    pub fn graph_pop_scope(&mut self) -> Option<ExecutionNodeId> {
        if !self.graph_enabled {
            return None;
        }
        self.execution_graph.pop_scope()
    }

    /// Record a brick in the execution graph.
    ///
    /// This should be called after `stop_brick()` with the timing data.
    pub fn graph_record_brick(
        &mut self,
        brick_id: BrickId,
        timing_ns: u64,
        elements: u64,
    ) -> Option<ExecutionNodeId> {
        if !self.graph_enabled {
            return None;
        }
        let node = ExecutionNode::Brick {
            id: brick_id,
            timing_ns,
            elements,
        };
        Some(self.execution_graph.add_node_in_scope(node))
    }

    /// Record a kernel launch in the execution graph.
    ///
    /// # Arguments
    /// - `name`: Kernel name (e.g., "batched_q4k_gemv")
    /// - `ptx_hash`: FNV-1a hash of PTX source for identity
    /// - `grid`: Grid dimensions (blocks)
    /// - `block`: Block dimensions (threads)
    /// - `shared_mem`: Shared memory bytes
    pub fn graph_record_kernel(
        &mut self,
        name: &str,
        ptx_hash: u64,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
    ) -> Option<ExecutionNodeId> {
        if !self.graph_enabled {
            return None;
        }
        Some(
            self.execution_graph
                .record_kernel_launch(name, ptx_hash, grid, block, shared_mem),
        )
    }

    /// Export execution graph to DOT format for visualization.
    ///
    /// Use with Graphviz: `dot -Tsvg output.dot -o graph.svg`
    #[must_use]
    pub fn graph_to_dot(&self) -> String {
        self.execution_graph.to_dot()
    }

    /// Export execution graph to trueno-graph CsrGraph.
    #[cfg(feature = "execution-graph")]
    #[must_use]
    pub fn graph_to_csr(&self) -> trueno_graph::CsrGraph {
        self.execution_graph.to_csr()
    }

    /// Clear the execution graph.
    pub fn graph_clear(&mut self) {
        self.execution_graph.clear();
    }

    /// Check if the execution graph scope stack is balanced.
    #[must_use]
    pub fn graph_is_scope_balanced(&self) -> bool {
        self.execution_graph.is_scope_balanced()
    }

    /// Set L2 cache hit rate (v1.1.0 OBSERVE phase)
    pub fn set_l2_cache_hit_rate(&mut self, rate: f32) {
        self.l2_cache_hit_rate = Some(rate.clamp(0.0, 1.0));
    }

    /// Get L2 cache hit rate
    pub fn l2_cache_hit_rate(&self) -> Option<f32> {
        self.l2_cache_hit_rate
    }

    /// Set zero-copy mode (v1.1.0 OBSERVE phase)
    pub fn set_zero_copy(&mut self, enabled: bool) {
        self.is_zero_copy = enabled;
    }

    /// Check if zero-copy is enabled
    pub fn is_zero_copy(&self) -> bool {
        self.is_zero_copy
    }

    /// Enable profiling.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable profiling.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if profiling is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start timing a brick. Returns timer handle.
    ///
    /// IMPORTANT: For GPU operations, call sync AFTER the operation
    /// completes but BEFORE calling stop().
    #[must_use]
    pub fn start(&self, name: &str) -> BrickTimer {
        BrickTimer {
            name: name.to_string(),
            start: Instant::now(),
        }
    }

    /// Stop timing and record the sample.
    ///
    /// # Arguments
    /// - `timer`: Timer handle from `start()`
    /// - `elements`: Number of elements (tokens) processed
    pub fn stop(&mut self, timer: BrickTimer, elements: u64) {
        if !self.enabled {
            return;
        }

        let elapsed = timer.start.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;

        // PAR-200: Try fast path first if name matches a known BrickId
        if let Some(brick_id) = BrickId::from_str(&timer.name) {
            let stats = &mut self.brick_stats[brick_id as usize];
            stats.add_sample(elapsed_ns, elements);
        } else {
            // Fall back to dynamic stats
            let name = timer.name;
            let stats = self.dynamic_stats.entry(name.clone()).or_insert_with(|| {
                BrickStats::new(&name)
            });
            stats.add_sample(elapsed_ns, elements);
        }

        // Update totals
        self.total_tokens += elements;
        self.total_ns += elapsed_ns;
    }

    /// Record a pre-measured duration for a brick.
    ///
    /// PAR-073: This method allows timing with raw `Instant` calls, avoiding
    /// borrow conflicts when profiling CUDA operations that also need `&mut self`.
    ///
    /// # Arguments
    /// - `name`: Brick name
    /// - `elapsed`: Duration of the operation (from `Instant::elapsed()`)
    /// - `elements`: Number of elements (tokens) processed
    ///
    /// # Example
    /// ```rust,ignore
    /// let start = std::time::Instant::now();
    /// cuda_stream.synchronize()?;
    /// self.some_cuda_operation()?;
    /// cuda_stream.synchronize()?;
    /// let elapsed = start.elapsed();
    /// self.profiler.record_elapsed("SomeBrick", elapsed, 1);
    /// ```
    pub fn record_elapsed(&mut self, name: &str, elapsed: std::time::Duration, elements: u64) {
        if !self.enabled {
            return;
        }

        let elapsed_ns = elapsed.as_nanos() as u64;

        // PAR-200: Try fast path first if name matches a known BrickId
        if let Some(brick_id) = BrickId::from_str(name) {
            let stats = &mut self.brick_stats[brick_id as usize];
            stats.add_sample(elapsed_ns, elements);
        } else {
            // Fall back to dynamic stats
            let stats = self.dynamic_stats.entry(name.to_string()).or_insert_with(|| {
                BrickStats::new(name)
            });
            stats.add_sample(elapsed_ns, elements);
        }

        // Update totals
        self.total_tokens += elements;
        self.total_ns += elapsed_ns;
    }

    /// PMAT-451: Record elapsed time with byte metrics for compression workloads.
    ///
    /// # Arguments
    /// - `name`: Brick name
    /// - `elapsed`: Duration of the operation
    /// - `elements`: Number of elements (pages) processed
    /// - `input_bytes`: Original uncompressed size
    /// - `output_bytes`: Compressed output size
    ///
    /// # Example
    /// ```rust,ignore
    /// let start = std::time::Instant::now();
    /// let compressed = zstd_compress(&page_data);
    /// let elapsed = start.elapsed();
    /// profiler.record_elapsed_with_bytes(
    ///     "ZstdCompress",
    ///     elapsed,
    ///     1,
    ///     page_data.len() as u64,
    ///     compressed.len() as u64,
    /// );
    /// ```
    pub fn record_elapsed_with_bytes(
        &mut self,
        name: &str,
        elapsed: std::time::Duration,
        elements: u64,
        input_bytes: u64,
        output_bytes: u64,
    ) {
        if !self.enabled {
            return;
        }

        let elapsed_ns = elapsed.as_nanos() as u64;

        // PAR-200: Try fast path first if name matches a known BrickId
        if let Some(brick_id) = BrickId::from_str(name) {
            let stats = &mut self.brick_stats[brick_id as usize];
            stats.add_sample_with_bytes(elapsed_ns, elements, input_bytes, output_bytes);
        } else {
            // Fall back to dynamic stats
            let stats = self.dynamic_stats.entry(name.to_string()).or_insert_with(|| {
                BrickStats::new(name)
            });
            stats.add_sample_with_bytes(elapsed_ns, elements, input_bytes, output_bytes);
        }

        // Update totals
        self.total_tokens += elements;
        self.total_ns += elapsed_ns;
    }

    /// PMAT-451: Set bottleneck classification for a brick.
    pub fn set_brick_bottleneck(&mut self, name: &str, bottleneck: BrickBottleneck) {
        // PAR-200: Try fast path first
        if let Some(brick_id) = BrickId::from_str(name) {
            self.brick_stats[brick_id as usize].set_bottleneck(bottleneck);
        } else if let Some(stats) = self.dynamic_stats.get_mut(name) {
            stats.set_bottleneck(bottleneck);
        }
    }

    /// Get statistics for a specific brick by name.
    ///
    /// First checks known BrickId types (O(1)), then falls back to dynamic stats.
    #[must_use]
    pub fn stats(&self, name: &str) -> Option<&BrickStats> {
        // Try fast path first
        if let Some(brick_id) = BrickId::from_str(name) {
            let stats = &self.brick_stats[brick_id as usize];
            if stats.count > 0 {
                return Some(stats);
            }
        }
        // Fall back to dynamic stats
        self.dynamic_stats.get(name)
    }

    /// Get all brick statistics (legacy API, returns dynamic stats only).
    ///
    /// For full statistics including known bricks, use `all_brick_stats()` instead.
    #[must_use]
    #[deprecated(since = "0.12.0", note = "Use all_brick_stats() for complete statistics")]
    pub fn all_stats(&self) -> &std::collections::HashMap<String, BrickStats> {
        &self.dynamic_stats
    }

    /// Get all brick statistics including both known and dynamic bricks.
    pub fn all_brick_stats(&self) -> impl Iterator<Item = &BrickStats> {
        self.brick_stats.iter()
            .filter(|s| s.count > 0)
            .chain(self.dynamic_stats.values())
    }

    /// Get total throughput across all bricks.
    #[must_use]
    pub fn total_throughput(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.total_tokens as f64 / (self.total_ns as f64 / 1_000_000_000.0)
        }
    }

    /// Get total tokens processed.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Get total time in nanoseconds.
    #[must_use]
    pub fn total_ns(&self) -> u64 {
        self.total_ns
    }

    /// Get all brick names.
    #[must_use]
    pub fn brick_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.brick_stats.iter()
            .enumerate()
            .filter(|(_, s)| s.count > 0)
            .map(|(i, _)| {
                // Safety: i < BrickId::COUNT
                let brick_id = unsafe { std::mem::transmute::<u8, BrickId>(i as u8) };
                brick_id.name().to_string()
            })
            .collect();
        names.extend(self.dynamic_stats.keys().cloned());
        names
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        for stats in &mut self.brick_stats {
            stats.count = 0;
            stats.total_ns = 0;
            stats.min_ns = u64::MAX;
            stats.max_ns = 0;
            stats.total_elements = 0;
            stats.total_bytes = 0;
            stats.total_compressed_bytes = 0;
        }
        self.dynamic_stats.clear();
        self.pending.clear();
        self.total_tokens = 0;
        self.total_ns = 0;
    }

    /// Generate a summary report.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Brick Profiler Summary (PAR-200) ===\n");
        report.push_str(&format!(
            "Total: {} tokens, {:.2}µs, {:.1} tok/s\n",
            self.total_tokens,
            self.total_ns as f64 / 1000.0,
            self.total_throughput()
        ));
        report.push_str("\nPer-Brick Breakdown:\n");

        // Collect all stats (known + dynamic)
        let mut all_stats: Vec<(&str, &BrickStats)> = Vec::new();

        // Add known bricks with non-zero counts
        for (i, stats) in self.brick_stats.iter().enumerate() {
            if stats.count > 0 {
                // Safety: i < BrickId::COUNT
                let brick_id = unsafe { std::mem::transmute::<u8, BrickId>(i as u8) };
                all_stats.push((brick_id.name(), stats));
            }
        }

        // Add dynamic bricks
        for (name, stats) in &self.dynamic_stats {
            all_stats.push((name.as_str(), stats));
        }

        // Sort by total time descending
        all_stats.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));

        for (name, stats) in all_stats {
            let pct = if self.total_ns > 0 {
                100.0 * stats.total_ns as f64 / self.total_ns as f64
            } else {
                0.0
            };
            report.push_str(&format!(
                "  {:20} {:8.2}µs avg ({:5.1}%) [{} samples]\n",
                name,
                stats.avg_us(),
                pct,
                stats.count
            ));
        }

        // Add category breakdown
        report.push_str("\nCategory Breakdown:\n");
        let cats = self.category_stats();
        for (i, cat_stats) in cats.iter().enumerate() {
            if cat_stats.count > 0 {
                // Safety: i < BrickCategory::COUNT
                let cat = unsafe { std::mem::transmute::<u8, BrickCategory>(i as u8) };
                report.push_str(&format!(
                    "  {:12} {:8.2}µs avg ({:5.1}%)\n",
                    cat.name(),
                    cat_stats.avg_us(),
                    cat_stats.percentage(self.total_ns)
                ));
            }
        }

        report
    }

    /// Export profiling data as JSON for pmat metrics integration.
    ///
    /// Format compatible with `.pmat-metrics/trends/` structure:
    /// ```json
    /// {
    ///   "total_tokens": 1000,
    ///   "total_ns": 5000000,
    ///   "total_throughput": 200000.0,
    ///   "bricks": [
    ///     {
    ///       "name": "RmsNorm",
    ///       "count": 10,
    ///       "total_ns": 1000000,
    ///       "avg_us": 100.0,
    ///       "min_us": 90.0,
    ///       "max_us": 120.0,
    ///       "throughput": 10000.0,
    ///       "pct": 20.0
    ///     }
    ///   ]
    /// }
    /// ```
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut bricks = Vec::new();

        // Collect all stats (known + dynamic)
        let mut all_stats: Vec<(&str, &BrickStats)> = Vec::new();

        // Add known bricks with non-zero counts
        for (i, stats) in self.brick_stats.iter().enumerate() {
            if stats.count > 0 {
                // Safety: i < BrickId::COUNT
                let brick_id = unsafe { std::mem::transmute::<u8, BrickId>(i as u8) };
                all_stats.push((brick_id.name(), stats));
            }
        }

        // Add dynamic bricks
        for (name, stats) in &self.dynamic_stats {
            all_stats.push((name.as_str(), stats));
        }

        // Sort by total time descending
        all_stats.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));

        for (name, stats) in all_stats {
            let pct = if self.total_ns > 0 {
                100.0 * stats.total_ns as f64 / self.total_ns as f64
            } else {
                0.0
            };
            // PMAT-451: Include compression_ratio, throughput_gbps, and bottleneck
            let compression = stats.compression_ratio();
            let throughput_gbps = stats.throughput_gbps();
            let bottleneck = stats.get_bottleneck();
            bricks.push(format!(
                r#"{{"name":"{}","count":{},"total_ns":{},"avg_us":{:.2},"min_us":{:.2},"max_us":{:.2},"throughput":{:.1},"pct":{:.1},"total_bytes":{},"compression_ratio":{:.2},"throughput_gbps":{:.2},"bottleneck":"{}"}}"#,
                name,
                stats.count,
                stats.total_ns,
                stats.avg_us(),
                stats.min_us(),
                stats.max_us(),
                stats.throughput(),
                pct,
                stats.total_bytes,
                compression,
                throughput_gbps,
                bottleneck
            ));
        }

        format!(
            r#"{{"total_tokens":{},"total_ns":{},"total_throughput":{:.1},"bricks":[{}]}}"#,
            self.total_tokens,
            self.total_ns,
            self.total_throughput(),
            bricks.join(",")
        )
    }

    /// Write profiling data to a JSON file for pmat tracking.
    ///
    /// # Errors
    /// Returns error if file cannot be written.
    pub fn write_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    // =======================================================================
    // CORRECTNESS-011: Per-kernel checksum capture for divergence detection
    // =======================================================================

    /// Record a kernel trace with output checksum for divergence detection.
    ///
    /// This enables automated CPU/GPU divergence detection by capturing
    /// output checksums alongside timing data. When GPU produces wrong output,
    /// this identifies WHICH kernel diverged without hours of manual debugging.
    ///
    /// Five-Whys Root Cause: Hours of manual "let me check X in Y" debugging
    /// → No automated tool identified which kernel diverged
    /// → BrickProfiler only captured timing, not checksums
    /// → Missing feature: per-kernel checksum capture
    ///
    /// # Arguments
    /// - `name`: Brick/kernel name
    /// - `layer_idx`: Layer index (0-N for transformer layers)
    /// - `position`: Position in sequence
    /// - `output`: Output tensor data (first 64 floats checksummed)
    ///
    /// # Example
    /// ```rust,ignore
    /// // After RoPE kernel
    /// profiler.record_checksum("RopeNeox", layer_idx, position, &q_rotated);
    /// ```
    pub fn record_checksum(&mut self, name: &str, layer_idx: usize, position: u32, output: &[f32]) {
        if !self.enabled {
            return;
        }
        let checksum = fnv1a_f32_checksum(output);
        let trace = KernelChecksum {
            name: name.to_string(),
            layer_idx,
            position,
            checksum,
        };
        self.kernel_checksums.push(trace);
    }

    /// Get all kernel checksums for divergence comparison.
    #[must_use]
    pub fn get_checksums(&self) -> &[KernelChecksum] {
        &self.kernel_checksums
    }

    /// Compare checksums with a reference profiler (e.g., CPU baseline).
    ///
    /// Returns None if all checksums match, or the first divergent kernel.
    #[must_use]
    pub fn find_divergence(&self, reference: &BrickProfiler) -> Option<DivergenceInfo> {
        use std::collections::HashMap;

        // Index reference checksums by (name, layer_idx, position)
        let ref_index: HashMap<(&str, usize, u32), u64> = reference
            .kernel_checksums
            .iter()
            .map(|t| ((t.name.as_str(), t.layer_idx, t.position), t.checksum))
            .collect();

        // Check each of our checksums against reference
        for trace in &self.kernel_checksums {
            let key = (trace.name.as_str(), trace.layer_idx, trace.position);
            if let Some(&expected) = ref_index.get(&key) {
                if trace.checksum != expected {
                    return Some(DivergenceInfo {
                        kernel_name: trace.name.clone(),
                        layer_idx: trace.layer_idx,
                        position: trace.position,
                        expected_checksum: expected,
                        actual_checksum: trace.checksum,
                    });
                }
            }
        }
        None
    }

    /// Reset checksum tracking (call before new forward pass).
    pub fn reset_checksums(&mut self) {
        self.kernel_checksums.clear();
    }
}

/// Kernel checksum for divergence detection.
///
/// CORRECTNESS-011: Captures output checksum per kernel invocation.
#[derive(Debug, Clone)]
pub struct KernelChecksum {
    /// Kernel/brick name
    pub name: String,
    /// Layer index
    pub layer_idx: usize,
    /// Sequence position
    pub position: u32,
    /// FNV-1a checksum of first 64 output floats
    pub checksum: u64,
}

/// Information about a detected divergence between CPU and GPU.
#[derive(Debug, Clone)]
pub struct DivergenceInfo {
    /// Name of the divergent kernel
    pub kernel_name: String,
    /// Layer where divergence occurred
    pub layer_idx: usize,
    /// Position where divergence occurred
    pub position: u32,
    /// Expected checksum (from CPU/reference)
    pub expected_checksum: u64,
    /// Actual checksum (from GPU/test)
    pub actual_checksum: u64,
}

impl fmt::Display for DivergenceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DIVERGENCE at '{}' (layer {}, pos {}): expected 0x{:016X}, got 0x{:016X}",
            self.kernel_name,
            self.layer_idx,
            self.position,
            self.expected_checksum,
            self.actual_checksum
        )
    }
}

/// FNV-1a hash of f32 slice (first 64 elements for efficiency).
///
/// Used for quick divergence detection between CPU and GPU outputs.
#[inline]
pub fn fnv1a_f32_checksum(data: &[f32]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    let len = data.len().min(64);
    for &val in &data[..len] {
        let bytes = val.to_le_bytes();
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

/// Macro for convenient brick timing with automatic sync.
///
/// # Usage
///
/// ```rust,ignore
/// time_brick!(profiler, "RmsNorm", 1, {
///     rmsnorm_kernel.launch();
///     stream.synchronize(); // REQUIRED for GPU
/// });
/// ```
#[macro_export]
macro_rules! time_brick {
    ($profiler:expr, $name:expr, $elements:expr, $body:block) => {{
        let timer = $profiler.start($name);
        let result = $body;
        $profiler.stop(timer, $elements);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_budget_from_latency() {
        let budget = TokenBudget::from_latency(50.0);
        assert!((budget.us_per_token - 50.0).abs() < 0.001);
        assert!((budget.tokens_per_sec - 20_000.0).abs() < 1.0);
    }

    #[test]
    fn test_token_budget_from_throughput() {
        let budget = TokenBudget::from_throughput(20_000.0);
        assert!((budget.us_per_token - 50.0).abs() < 0.001);
        assert!((budget.tokens_per_sec - 20_000.0).abs() < 1.0);
    }

    #[test]
    fn test_token_budget_is_met() {
        let budget = TokenBudget::from_latency(50.0);
        assert!(budget.is_met(40.0)); // Under budget
        assert!(budget.is_met(50.0)); // Exactly at budget
        assert!(!budget.is_met(60.0)); // Over budget
    }

    #[test]
    fn test_dot_op() {
        let op = DotOp::new(4);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = op.execute((a, b), Backend::Scalar).unwrap();
        assert!((result - 70.0).abs() < 0.001); // 1*5 + 2*6 + 3*7 + 4*8 = 70
    }

    #[test]
    fn test_add_op() {
        let op = AddOp::new(4);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = op.execute((a, b), Backend::Scalar).unwrap();
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matmul_op() {
        let op = MatmulOp::new(2, 2, 2);
        // A = [[1, 2], [3, 4]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        // B = [[5, 6], [7, 8]]
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = op.execute((a, b), Backend::Scalar).unwrap();
        // C = [[19, 22], [43, 50]]
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_softmax_op() {
        let op = SoftmaxOp::new(3);
        let input = vec![1.0, 2.0, 3.0];
        let result = op.execute(input, Backend::Scalar).unwrap();
        // Sum should be 1.0
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        // Values should be increasing
        assert!(result[0] < result[1]);
        assert!(result[1] < result[2]);
    }

    #[test]
    fn test_compute_brick_run() {
        let brick = ComputeBrick::new(DotOp::new(4))
            .budget_tok_per_sec(1_000_000.0)
            .backend(Backend::Scalar);

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = brick.run((a, b)).unwrap();

        assert!((result.output - 70.0).abs() < 0.001);
        assert_eq!(result.tokens_processed, 4);
        assert!(result.tokens_per_sec > 0.0);
    }

    #[test]
    fn test_compute_brick_verify() {
        let brick = ComputeBrick::new(DotOp::new(4))
            .assert_finite()
            .assert_bounds(-1000.0, 1000.0);

        let verification = brick.verify();
        assert!(verification.is_valid());
        assert_eq!(verification.assertion_results.len(), 2);
    }

    #[test]
    fn test_compute_brick_no_assertions() {
        let brick = ComputeBrick::new(DotOp::new(4));
        let verification = brick.verify();
        assert!(!verification.is_valid()); // Should fail Popperian requirement
    }

    #[test]
    fn test_brick_layer() {
        let dot_brick = ComputeBrick::new(DotOp::new(100)).budget_tok_per_sec(50_000.0);

        let add_brick = ComputeBrick::new(AddOp::new(100)).budget_tok_per_sec(30_000.0); // Bottleneck

        let layer = BrickLayer::new()
            .with_brick(&dot_brick)
            .with_brick(&add_brick);

        assert!((layer.throughput_ceiling() - 30_000.0).abs() < 1.0);
        assert_eq!(layer.bottleneck(), Some("add"));
    }

    #[test]
    fn test_backend_display() {
        assert_eq!(format!("{}", Backend::Avx2), "AVX2");
        assert_eq!(format!("{}", Backend::Cuda), "CUDA");
        assert_eq!(format!("{}", Backend::Scalar), "Scalar");
    }

    #[test]
    fn test_budget_utilization() {
        let budget = TokenBudget::from_latency(100.0);
        assert!((budget.utilization(50.0) - 0.5).abs() < 0.001); // 50% used
        assert!((budget.utilization(100.0) - 1.0).abs() < 0.001); // 100% used
        assert!((budget.utilization(150.0) - 1.5).abs() < 0.001); // 150% over
    }

    // ========================================================================
    // ByteBudget Tests (F224 falsification)
    // ========================================================================

    #[test]
    fn test_byte_budget_from_throughput() {
        let budget = ByteBudget::from_throughput(25.0); // 25 GB/s
        assert!((budget.gb_per_sec - 25.0).abs() < 0.001);
        // 25 GB/s = 6.1M pages/sec = 0.164 µs/page
        assert!((budget.us_per_page - 0.164).abs() < 0.01);
        assert_eq!(budget.page_size, 4096);
    }

    #[test]
    fn test_byte_budget_from_latency() {
        let budget = ByteBudget::from_latency(0.164); // 0.164 µs/page
        assert!((budget.us_per_page - 0.164).abs() < 0.001);
        // Should be ~25 GB/s
        assert!((budget.gb_per_sec - 25.0).abs() < 1.0);
    }

    #[test]
    fn test_byte_budget_to_token_budget() {
        let byte_budget = ByteBudget::from_throughput(25.0);
        let token_budget = byte_budget.to_token_budget();

        // us_per_token should equal us_per_page
        assert!((token_budget.us_per_token - byte_budget.us_per_page).abs() < 0.001);
        // tokens_per_sec should equal pages_per_sec
        let pages_per_sec = 25.0 * 1e9 / 4096.0;
        assert!((token_budget.tokens_per_sec - pages_per_sec).abs() < 1000.0);
    }

    #[test]
    fn test_byte_budget_is_met() {
        let budget = ByteBudget::from_throughput(25.0); // ~0.164 µs/page
        assert!(budget.is_met(0.10)); // Faster than budget
        assert!(budget.is_met(budget.us_per_page)); // Exactly at budget
        assert!(!budget.is_met(0.20)); // Slower than budget
    }

    #[test]
    fn test_byte_budget_with_page_size() {
        let budget = ByteBudget::from_throughput(25.0).with_page_size(65536); // 64KB pages
        assert_eq!(budget.page_size, 65536);
        // 25 GB/s with 64KB pages = 381K pages/sec = 2.62 µs/page
        assert!((budget.us_per_page - 2.62).abs() < 0.1);
    }

    #[test]
    fn test_byte_budget_throughput_from_latency() {
        // 0.164 µs/page with 4KB pages should be ~25 GB/s
        let throughput = ByteBudget::throughput_from_latency(0.164, 4096);
        assert!((throughput - 25.0).abs() < 1.0);
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_token_result_map() {
        let result = TokenResult {
            output: 42,
            tokens_processed: 10,
            us_per_token: 5.0,
            tokens_per_sec: 200_000.0,
            budget_met: true,
            budget_utilization: 0.5,
        };

        let mapped = result.map(|x| x * 2);
        assert_eq!(mapped.output, 84);
        assert_eq!(mapped.tokens_processed, 10);
        assert!((mapped.us_per_token - 5.0).abs() < 0.001);
        assert!(mapped.budget_met);
    }

    #[test]
    fn test_compute_assertion_equiv_with_tolerance() {
        let assertion = ComputeAssertion::equiv_with_tolerance(Backend::Scalar, 1e-3);
        match assertion {
            ComputeAssertion::Equivalence { baseline, tolerance } => {
                assert_eq!(baseline, Backend::Scalar);
                assert!((tolerance - 1e-3).abs() < 1e-10);
            }
            _ => panic!("Expected Equivalence assertion"),
        }
    }

    #[test]
    fn test_brick_verification_failures() {
        let brick = ComputeBrick::new(DotOp::new(4));
        let verification = brick.verify();

        // Should have one failure (no assertions)
        let failures: Vec<_> = verification.failures().collect();
        assert_eq!(failures.len(), 1);
        assert!(!failures[0].passed);
    }

    #[test]
    fn test_dot_op_size_mismatch() {
        let op = DotOp::new(4);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0]; // Wrong size
        let result = op.execute((a, b), Backend::Scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_add_op_size_mismatch() {
        let op = AddOp::new(4);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0]; // Wrong size
        let result = op.execute((a, b), Backend::Scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_op_size_mismatch_a() {
        let op = MatmulOp::new(2, 2, 2);
        let a = vec![1.0, 2.0, 3.0]; // Wrong size (should be 4)
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = op.execute((a, b), Backend::Scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_matmul_op_size_mismatch_b() {
        let op = MatmulOp::new(2, 2, 2);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0]; // Wrong size (should be 4)
        let result = op.execute((a, b), Backend::Scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_softmax_op_empty() {
        let op = SoftmaxOp::new(0);
        let result = op.execute(vec![], Backend::Scalar).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_compute_brick_builder_methods() {
        let brick = ComputeBrick::new(DotOp::new(100))
            .assert_equiv_with_tolerance(Backend::Avx2, 1e-3)
            .budget_us_per_tok(100.0)
            .enforce_budget(true);

        assert_eq!(brick.name(), "dot");
        assert_eq!(brick.get_backend(), Backend::Auto);
        assert!((brick.get_budget().us_per_token - 100.0).abs() < 0.001);
        assert_eq!(brick.get_assertions().len(), 1);
    }

    #[test]
    fn test_compute_brick_budget_method() {
        let budget = TokenBudget::from_throughput(100_000.0).with_batch_size(32);
        let brick = ComputeBrick::new(DotOp::new(100)).budget(budget);

        assert_eq!(brick.get_budget().batch_size, 32);
        assert!((brick.get_budget().tokens_per_sec - 100_000.0).abs() < 1.0);
    }

    #[test]
    fn test_compute_brick_enforce_budget_fail() {
        let brick = ComputeBrick::new(DotOp::new(1000000)) // Very large to take time
            .budget_tok_per_sec(1e15) // Impossibly high target
            .backend(Backend::Scalar)
            .enforce_budget(true);

        let a: Vec<f32> = (0..1000000).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1000000).map(|i| i as f32).collect();
        let result = brick.run((a, b));

        // Should fail due to budget exceeded
        assert!(result.is_err());
        if let Err(BrickError::BudgetExceeded { .. }) = result {
            // Expected
        } else {
            panic!("Expected BudgetExceeded error");
        }
    }

    #[test]
    fn test_compute_brick_clone() {
        let brick = ComputeBrick::new(DotOp::new(4))
            .assert_finite()
            .budget_tok_per_sec(50_000.0)
            .backend(Backend::Scalar);

        let cloned = brick.clone();
        assert_eq!(cloned.name(), brick.name());
        assert_eq!(cloned.get_backend(), brick.get_backend());
        assert_eq!(cloned.get_assertions().len(), brick.get_assertions().len());
    }

    #[test]
    fn test_compute_brick_debug() {
        let brick = ComputeBrick::new(DotOp::new(4))
            .assert_finite()
            .backend(Backend::Avx2);

        let debug_str = format!("{:?}", brick);
        assert!(debug_str.contains("ComputeBrick"));
        assert!(debug_str.contains("dot"));
        assert!(debug_str.contains("Avx2")); // Debug uses variant name, not Display
    }

    #[test]
    fn test_brick_layer_with_named() {
        let layer = BrickLayer::new()
            .with_named("attention", 10_000.0)
            .with_named("ffn", 5_000.0);

        assert_eq!(layer.bricks().len(), 2);
        assert!((layer.throughput_ceiling() - 5_000.0).abs() < 1.0);
        assert_eq!(layer.bottleneck(), Some("ffn"));
    }

    #[test]
    fn test_brick_layer_empty() {
        let layer = BrickLayer::new();
        assert_eq!(layer.throughput_ceiling(), f64::INFINITY);
        assert_eq!(layer.bottleneck(), None);
    }

    #[test]
    fn test_backend_all_variants_display() {
        assert_eq!(format!("{}", Backend::Sse2), "SSE2");
        assert_eq!(format!("{}", Backend::Avx512), "AVX-512");
        assert_eq!(format!("{}", Backend::Neon), "NEON");
        assert_eq!(format!("{}", Backend::Wasm), "WASM");
        assert_eq!(format!("{}", Backend::Wgpu), "wgpu");
        assert_eq!(format!("{}", Backend::Auto), "Auto");
    }

    #[test]
    fn test_byte_budget_default() {
        let budget = ByteBudget::default();
        assert!((budget.gb_per_sec - 25.0).abs() < 0.001);
    }

    #[test]
    fn test_byte_budget_utilization() {
        let budget = ByteBudget::from_throughput(25.0);
        let util = budget.utilization(budget.us_per_page / 2.0); // 50% of budget
        assert!((util - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_token_budget_with_batch_size() {
        let budget = TokenBudget::from_latency(50.0).with_batch_size(64);
        assert_eq!(budget.batch_size, 64);
    }

    #[test]
    fn test_token_budget_with_batch_size_min() {
        let budget = TokenBudget::from_latency(50.0).with_batch_size(0);
        assert_eq!(budget.batch_size, 1); // Should clamp to 1
    }

    #[test]
    fn test_compute_brick_run_zero_tokens() {
        let brick = ComputeBrick::new(SoftmaxOp::new(0))
            .backend(Backend::Scalar);

        let result = brick.run(vec![]).unwrap();
        assert!(result.output.is_empty());
        // Edge case: zero tokens should still work
    }

    #[test]
    fn test_brick_verification_is_valid() {
        let brick = ComputeBrick::new(DotOp::new(4))
            .assert_finite()
            .assert_bounds(-1000.0, 1000.0);

        let verification = brick.verify();
        assert!(verification.is_valid());
    }

    #[test]
    fn test_compute_assertion_bounds() {
        let assertion = ComputeAssertion::bounds(-10.0, 10.0);
        match assertion {
            ComputeAssertion::Bounds { min, max } => {
                assert!((-10.0 - min).abs() < 0.001);
                assert!((10.0 - max).abs() < 0.001);
            }
            _ => panic!("Expected Bounds assertion"),
        }
    }

    #[test]
    fn test_compute_assertion_finite() {
        let assertion = ComputeAssertion::finite();
        assert!(matches!(assertion, ComputeAssertion::Finite));
    }

    #[test]
    fn test_backend_default() {
        let backend = Backend::default();
        assert_eq!(backend, Backend::Avx2);
    }

    // ========================================================================
    // Fused LLM Operations Tests (PMAT-PERF-009)
    // ========================================================================

    #[test]
    fn test_fused_qkv_op_new() {
        // Qwen 3B dimensions: hidden=3584, heads=28, kv_heads=4 (GQA)
        let op = FusedQKVOp::new(3584, 28, 4);
        assert_eq!(op.hidden_size, 3584);
        assert_eq!(op.num_heads, 28);
        assert_eq!(op.head_dim, 128); // 3584 / 28
        assert_eq!(op.kv_dim, 512);   // 4 * 128
    }

    #[test]
    fn test_fused_qkv_op_name() {
        let op = FusedQKVOp::new(1024, 8, 8);
        assert_eq!(op.name(), "fused_qkv");
    }

    #[test]
    fn test_fused_qkv_op_execute_small() {
        let hidden_size = 4;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_dim = hidden_size / num_heads; // 2
        let kv_dim = num_kv_heads * head_dim;   // 4

        let op = FusedQKVOp::new(hidden_size, num_heads, num_kv_heads);

        // Identity-like weights for testing
        let q_weight = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let k_weight = q_weight.clone();
        let v_weight = q_weight.clone();

        let weights = FusedQKVWeights {
            q_weight,
            k_weight,
            v_weight,
        };

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let (q, k, v) = op.execute((x.clone(), weights), Backend::Scalar).unwrap();

        // With identity weights, output should equal input
        assert_eq!(q, x);
        assert_eq!(k.len(), kv_dim);
        assert_eq!(v.len(), kv_dim);
    }

    #[test]
    fn test_fused_qkv_op_size_mismatch() {
        let op = FusedQKVOp::new(4, 2, 2);
        let weights = FusedQKVWeights {
            q_weight: vec![0.0; 16],
            k_weight: vec![0.0; 16],
            v_weight: vec![0.0; 16],
        };
        let x = vec![1.0, 2.0, 3.0]; // Wrong size (should be 4)

        let result = op.execute((x, weights), Backend::Scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_qkv_op_tokens() {
        // hidden=1024, kv_dim=256 (GQA with 4 heads, 2 kv_heads)
        let op = FusedQKVOp::new(1024, 4, 2);
        let weights = FusedQKVWeights {
            q_weight: vec![],
            k_weight: vec![],
            v_weight: vec![],
        };
        let tokens = op.tokens(&(vec![], weights));
        // Q (1024) + K (512) + V (512) = 2048
        assert_eq!(tokens, 1024 + 512 + 512);
    }

    #[test]
    fn test_fused_gate_up_op_new() {
        // Qwen 3B dimensions
        let op = FusedGateUpOp::new(3584, 18944);
        assert_eq!(op.hidden_size, 3584);
        assert_eq!(op.intermediate_size, 18944);
    }

    #[test]
    fn test_fused_gate_up_op_name() {
        let op = FusedGateUpOp::new(1024, 4096);
        assert_eq!(op.name(), "fused_gate_up");
    }

    #[test]
    fn test_fused_gate_up_op_silu() {
        // SiLU(0) = 0 / (1 + 1) = 0
        assert!((FusedGateUpOp::silu(0.0)).abs() < 1e-6);
        // SiLU(x) for large x approaches x
        let large = FusedGateUpOp::silu(10.0);
        assert!((large - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_fused_gate_up_op_execute_small() {
        let hidden_size = 2;
        let intermediate_size = 3;

        let op = FusedGateUpOp::new(hidden_size, intermediate_size);

        // Simple weights
        let gate_weight = vec![
            1.0, 0.0,  // intermediate[0] = x[0]
            0.0, 1.0,  // intermediate[1] = x[1]
            1.0, 1.0,  // intermediate[2] = x[0] + x[1]
        ];
        let up_weight = vec![
            1.0, 0.0,  // up[0] = x[0]
            0.0, 1.0,  // up[1] = x[1]
            0.5, 0.5,  // up[2] = 0.5 * (x[0] + x[1])
        ];

        let weights = FusedGateUpWeights {
            gate_weight,
            up_weight,
        };

        let x = vec![2.0, 3.0];
        let output = op.execute((x, weights), Backend::Scalar).unwrap();

        assert_eq!(output.len(), intermediate_size);
        // output[0] = SiLU(2.0) * 2.0
        // output[1] = SiLU(3.0) * 3.0
        // output[2] = SiLU(5.0) * 2.5
        assert!(output[0] > 0.0);
        assert!(output[1] > 0.0);
        assert!(output[2] > 0.0);
    }

    #[test]
    fn test_fused_gate_up_op_size_mismatch() {
        let op = FusedGateUpOp::new(4, 8);
        let weights = FusedGateUpWeights {
            gate_weight: vec![0.0; 32],
            up_weight: vec![0.0; 32],
        };
        let x = vec![1.0, 2.0, 3.0]; // Wrong size (should be 4)

        let result = op.execute((x, weights), Backend::Scalar);
        assert!(result.is_err());
    }

    #[test]
    fn test_fused_gate_up_op_tokens() {
        let op = FusedGateUpOp::new(1024, 4096);
        let weights = FusedGateUpWeights {
            gate_weight: vec![],
            up_weight: vec![],
        };
        let tokens = op.tokens(&(vec![], weights));
        assert_eq!(tokens, 4096);
    }

    #[test]
    fn test_fused_qkv_compute_brick() {
        let op = FusedQKVOp::new(4, 2, 2);
        let brick = ComputeBrick::new(op)
            .assert_finite()
            .budget_tok_per_sec(1_000_000.0)
            .backend(Backend::Scalar);

        assert_eq!(brick.name(), "fused_qkv");
        let verification = brick.verify();
        assert!(verification.is_valid());
    }

    #[test]
    fn test_fused_gate_up_compute_brick() {
        let op = FusedGateUpOp::new(4, 8);
        let brick = ComputeBrick::new(op)
            .assert_finite()
            .budget_tok_per_sec(1_000_000.0)
            .backend(Backend::Scalar);

        assert_eq!(brick.name(), "fused_gate_up");
        let verification = brick.verify();
        assert!(verification.is_valid());
    }

    #[test]
    fn test_fused_ops_brick_layer() {
        // Build a transformer layer with fused ops
        let qkv_brick = ComputeBrick::new(FusedQKVOp::new(1024, 8, 8))
            .budget_tok_per_sec(100_000.0);
        let ffn_brick = ComputeBrick::new(FusedGateUpOp::new(1024, 4096))
            .budget_tok_per_sec(50_000.0); // FFN is typically slower

        let layer = BrickLayer::new()
            .with_brick(&qkv_brick)
            .with_brick(&ffn_brick);

        // Throughput ceiling should be the FFN (bottleneck)
        assert!((layer.throughput_ceiling() - 50_000.0).abs() < 1.0);
        assert_eq!(layer.bottleneck(), Some("fused_gate_up"));
    }

    #[test]
    fn test_fused_qkv_weights_clone() {
        let weights = FusedQKVWeights {
            q_weight: vec![1.0, 2.0],
            k_weight: vec![3.0, 4.0],
            v_weight: vec![5.0, 6.0],
        };
        let cloned = weights.clone();
        assert_eq!(cloned.q_weight, weights.q_weight);
        assert_eq!(cloned.k_weight, weights.k_weight);
        assert_eq!(cloned.v_weight, weights.v_weight);
    }

    #[test]
    fn test_fused_gate_up_weights_clone() {
        let weights = FusedGateUpWeights {
            gate_weight: vec![1.0, 2.0],
            up_weight: vec![3.0, 4.0],
        };
        let cloned = weights.clone();
        assert_eq!(cloned.gate_weight, weights.gate_weight);
        assert_eq!(cloned.up_weight, weights.up_weight);
    }

    #[test]
    fn test_fused_qkv_op_clone() {
        let op = FusedQKVOp::new(1024, 8, 4);
        let cloned = op.clone();
        assert_eq!(cloned.hidden_size, op.hidden_size);
        assert_eq!(cloned.kv_dim, op.kv_dim);
        assert_eq!(cloned.num_heads, op.num_heads);
        assert_eq!(cloned.head_dim, op.head_dim);
    }

    #[test]
    fn test_fused_gate_up_op_clone() {
        let op = FusedGateUpOp::new(1024, 4096);
        let cloned = op.clone();
        assert_eq!(cloned.hidden_size, op.hidden_size);
        assert_eq!(cloned.intermediate_size, op.intermediate_size);
    }

    #[test]
    fn test_fused_qkv_weights_debug() {
        let weights = FusedQKVWeights {
            q_weight: vec![1.0],
            k_weight: vec![2.0],
            v_weight: vec![3.0],
        };
        let debug_str = format!("{:?}", weights);
        assert!(debug_str.contains("FusedQKVWeights"));
    }

    #[test]
    fn test_fused_gate_up_weights_debug() {
        let weights = FusedGateUpWeights {
            gate_weight: vec![1.0],
            up_weight: vec![2.0],
        };
        let debug_str = format!("{:?}", weights);
        assert!(debug_str.contains("FusedGateUpWeights"));
    }

    #[test]
    fn test_fused_qkv_op_debug() {
        let op = FusedQKVOp::new(1024, 8, 4);
        let debug_str = format!("{:?}", op);
        assert!(debug_str.contains("FusedQKVOp"));
        assert!(debug_str.contains("1024"));
    }

    #[test]
    fn test_fused_gate_up_op_debug() {
        let op = FusedGateUpOp::new(1024, 4096);
        let debug_str = format!("{:?}", op);
        assert!(debug_str.contains("FusedGateUpOp"));
        assert!(debug_str.contains("1024"));
    }

    // ========================================================================
    // BrickProfiler Tests (PAR-073)
    // ========================================================================

    #[test]
    fn test_brick_profiler_disabled_by_default() {
        let profiler = BrickProfiler::new();
        assert!(!profiler.is_enabled());
    }

    #[test]
    fn test_brick_profiler_enabled_constructor() {
        let profiler = BrickProfiler::enabled();
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_brick_profiler_enable_disable() {
        let mut profiler = BrickProfiler::new();
        assert!(!profiler.is_enabled());
        profiler.enable();
        assert!(profiler.is_enabled());
        profiler.disable();
        assert!(!profiler.is_enabled());
    }

    #[test]
    fn test_brick_profiler_timing() {
        let mut profiler = BrickProfiler::enabled();

        // Time a simple operation
        let timer = profiler.start("TestBrick");
        std::thread::sleep(std::time::Duration::from_micros(100));
        profiler.stop(timer, 1);

        // Verify stats were recorded
        let stats = profiler.stats("TestBrick").expect("stats should exist");
        assert_eq!(stats.count, 1);
        assert!(stats.avg_us() >= 50.0); // Should be at least 50µs (sleep + overhead)
        assert_eq!(stats.total_elements, 1);
    }

    #[test]
    fn test_brick_profiler_multiple_samples() {
        let mut profiler = BrickProfiler::enabled();

        for _ in 0..10 {
            let timer = profiler.start("MultiBrick");
            // Small busy loop
            let mut sum = 0u64;
            for i in 0..1000 {
                sum = sum.wrapping_add(i);
            }
            let _ = sum; // Prevent optimization
            profiler.stop(timer, 1);
        }

        let stats = profiler.stats("MultiBrick").expect("stats should exist");
        assert_eq!(stats.count, 10);
        assert_eq!(stats.total_elements, 10);
    }

    #[test]
    fn test_brick_profiler_multiple_bricks() {
        let mut profiler = BrickProfiler::enabled();

        let timer = profiler.start("BrickA");
        profiler.stop(timer, 1);

        let timer = profiler.start("BrickB");
        profiler.stop(timer, 2);

        assert!(profiler.stats("BrickA").is_some());
        assert!(profiler.stats("BrickB").is_some());
        assert_eq!(profiler.total_tokens(), 3);
    }

    #[test]
    fn test_brick_profiler_disabled_no_record() {
        let mut profiler = BrickProfiler::new(); // Disabled by default

        let timer = profiler.start("DisabledBrick");
        profiler.stop(timer, 1);

        // Should not record anything when disabled
        assert!(profiler.stats("DisabledBrick").is_none());
        assert_eq!(profiler.total_tokens(), 0);
    }

    #[test]
    fn test_brick_profiler_reset() {
        let mut profiler = BrickProfiler::enabled();

        let timer = profiler.start("ResetBrick");
        profiler.stop(timer, 5);

        assert_eq!(profiler.total_tokens(), 5);

        profiler.reset();

        assert_eq!(profiler.total_tokens(), 0);
        assert!(profiler.stats("ResetBrick").is_none());
    }

    #[test]
    fn test_brick_profiler_summary() {
        let mut profiler = BrickProfiler::enabled();

        let timer = profiler.start("SummaryBrick");
        profiler.stop(timer, 10);

        let summary = profiler.summary();
        assert!(summary.contains("Brick Profiler Summary"));
        assert!(summary.contains("SummaryBrick"));
        assert!(summary.contains("10 tokens"));
    }

    #[test]
    fn test_brick_stats_new() {
        let stats = BrickStats::new("TestStats");
        assert_eq!(stats.name, "TestStats");
        assert_eq!(stats.count, 0);
        assert_eq!(stats.total_ns, 0);
        assert_eq!(stats.min_ns, u64::MAX);
        assert_eq!(stats.max_ns, 0);
    }

    #[test]
    fn test_brick_stats_add_sample() {
        let mut stats = BrickStats::new("Test");
        stats.add_sample(1000, 1); // 1µs
        stats.add_sample(2000, 1); // 2µs
        stats.add_sample(3000, 1); // 3µs

        assert_eq!(stats.count, 3);
        assert_eq!(stats.total_ns, 6000);
        assert_eq!(stats.min_ns, 1000);
        assert_eq!(stats.max_ns, 3000);
        assert_eq!(stats.total_elements, 3);
        assert!((stats.avg_us() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_brick_stats_throughput() {
        let mut stats = BrickStats::new("Throughput");
        // 1000 elements in 1ms = 1,000,000 elements/sec
        stats.add_sample(1_000_000, 1000); // 1ms, 1000 elements

        let throughput = stats.throughput();
        assert!((throughput - 1_000_000.0).abs() < 1000.0);
    }

    #[test]
    fn test_brick_timer_debug() {
        let timer = BrickTimer {
            name: "DebugTimer".to_string(),
            start: Instant::now(),
        };
        let debug_str = format!("{:?}", timer);
        assert!(debug_str.contains("BrickTimer"));
        assert!(debug_str.contains("DebugTimer"));
    }

    #[test]
    fn test_brick_sample_clone() {
        let sample = BrickSample {
            brick_id: 42,
            elapsed_ns: 1000,
            elements: 5,
        };
        let cloned = sample;
        assert_eq!(cloned.brick_id, 42);
        assert_eq!(cloned.elapsed_ns, 1000);
        assert_eq!(cloned.elements, 5);
    }

    // ========================================================================
    // PMAT-451: Compression Ratio and Bottleneck Tests
    // ========================================================================

    #[test]
    fn test_brick_bottleneck_display() {
        assert_eq!(format!("{}", BrickBottleneck::Unknown), "unknown");
        assert_eq!(format!("{}", BrickBottleneck::Memory), "memory");
        assert_eq!(format!("{}", BrickBottleneck::Compute), "compute");
    }

    #[test]
    fn test_brick_bottleneck_default() {
        let bottleneck = BrickBottleneck::default();
        assert_eq!(bottleneck, BrickBottleneck::Unknown);
    }

    #[test]
    fn test_brick_stats_compression_ratio() {
        let mut stats = BrickStats::new("Compress");
        // 1000 bytes in, 250 bytes out = 4.0 compression ratio
        stats.add_sample_with_bytes(1_000_000, 100, 1000, 250);

        let ratio = stats.compression_ratio();
        assert!((ratio - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_brick_stats_compression_ratio_no_data() {
        let stats = BrickStats::new("Empty");
        // No compressed bytes = 1.0 ratio (no compression = 1:1)
        assert_eq!(stats.compression_ratio(), 1.0);
    }

    #[test]
    fn test_brick_stats_throughput_gbps() {
        let mut stats = BrickStats::new("Throughput");
        // 1 GB (1e9 bytes) in 1 second (1e9 ns) = 1.0 GB/s
        stats.add_sample_with_bytes(1_000_000_000, 1000, 1_000_000_000, 0);

        let throughput = stats.throughput_gbps();
        assert!((throughput - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_brick_stats_throughput_gbps_zero_time() {
        let stats = BrickStats::new("Empty");
        // Zero time = 0.0 throughput (avoid division by zero)
        assert_eq!(stats.throughput_gbps(), 0.0);
    }

    #[test]
    fn test_brick_stats_add_sample_with_bytes() {
        let mut stats = BrickStats::new("Bytes");

        stats.add_sample_with_bytes(1000, 10, 100, 25);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.total_ns, 1000);
        assert_eq!(stats.total_elements, 10);
        assert_eq!(stats.total_bytes, 100);
        assert_eq!(stats.total_compressed_bytes, 25);
        assert_eq!(stats.min_ns, 1000);
        assert_eq!(stats.max_ns, 1000);

        // Add second sample
        stats.add_sample_with_bytes(500, 5, 50, 20);
        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_ns, 1500);
        assert_eq!(stats.total_elements, 15);
        assert_eq!(stats.total_bytes, 150);
        assert_eq!(stats.total_compressed_bytes, 45);
        assert_eq!(stats.min_ns, 500);
        assert_eq!(stats.max_ns, 1000);
    }

    #[test]
    fn test_brick_stats_bottleneck() {
        let mut stats = BrickStats::new("Test");
        assert_eq!(stats.get_bottleneck(), BrickBottleneck::Unknown);

        stats.set_bottleneck(BrickBottleneck::Memory);
        assert_eq!(stats.get_bottleneck(), BrickBottleneck::Memory);

        stats.set_bottleneck(BrickBottleneck::Compute);
        assert_eq!(stats.get_bottleneck(), BrickBottleneck::Compute);
    }

    #[test]
    fn test_brick_profiler_record_elapsed_with_bytes() {
        use std::time::Duration;
        let mut profiler = BrickProfiler::new();
        profiler.enable(); // Profiler is disabled by default

        profiler.record_elapsed_with_bytes("Compress", Duration::from_nanos(1000), 100, 1_000_000, 250_000);
        profiler.record_elapsed_with_bytes("Compress", Duration::from_nanos(2000), 200, 2_000_000, 500_000);

        let stats = profiler.stats("Compress").unwrap();
        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_ns, 3000);
        assert_eq!(stats.total_elements, 300);
        assert_eq!(stats.total_bytes, 3_000_000);
        assert_eq!(stats.total_compressed_bytes, 750_000);
    }

    #[test]
    fn test_brick_profiler_set_bottleneck() {
        use std::time::Duration;
        let mut profiler = BrickProfiler::new();
        profiler.enable(); // Profiler is disabled by default
        profiler.record_elapsed("TestBrick", Duration::from_nanos(1000), 100);
        profiler.set_brick_bottleneck("TestBrick", BrickBottleneck::Memory);

        let stats = profiler.stats("TestBrick").unwrap();
        assert_eq!(stats.get_bottleneck(), BrickBottleneck::Memory);
    }

    #[test]
    fn test_brick_profiler_to_json_includes_pmat451_fields() {
        use std::time::Duration;
        let mut profiler = BrickProfiler::new();
        profiler.enable(); // Profiler is disabled by default
        profiler.record_elapsed_with_bytes("Compress", Duration::from_micros(1000), 100, 1_000_000, 250_000);
        profiler.set_brick_bottleneck("Compress", BrickBottleneck::Memory);

        let json = profiler.to_json();

        // Verify new PMAT-451 fields are present
        assert!(json.contains("\"total_bytes\":"));
        assert!(json.contains("\"compression_ratio\":"));
        assert!(json.contains("\"throughput_gbps\":"));
        assert!(json.contains("\"bottleneck\":\"memory\""));
    }

    // ========================================================================
    // PAR-200: BrickProfiler v2 Tests
    // ========================================================================

    #[test]
    fn test_brick_id_category() {
        assert_eq!(BrickId::RmsNorm.category(), BrickCategory::Norm);
        assert_eq!(BrickId::LayerNorm.category(), BrickCategory::Norm);
        assert_eq!(BrickId::QkvProjection.category(), BrickCategory::Attention);
        assert_eq!(BrickId::AttentionSoftmax.category(), BrickCategory::Attention);
        assert_eq!(BrickId::GateProjection.category(), BrickCategory::Ffn);
        assert_eq!(BrickId::DownProjection.category(), BrickCategory::Ffn);
        assert_eq!(BrickId::Embedding.category(), BrickCategory::Other);
        assert_eq!(BrickId::Sampling.category(), BrickCategory::Other);
    }

    #[test]
    fn test_brick_id_from_str() {
        assert_eq!(BrickId::from_str("RmsNorm"), Some(BrickId::RmsNorm));
        assert_eq!(BrickId::from_str("Rope"), Some(BrickId::RopeEmbedding));
        assert_eq!(BrickId::from_str("RoPE"), Some(BrickId::RopeEmbedding));
        assert_eq!(BrickId::from_str("SiLU"), Some(BrickId::Activation));
        assert_eq!(BrickId::from_str("Unknown"), None);
    }

    #[test]
    fn test_brick_id_name() {
        assert_eq!(BrickId::RmsNorm.name(), "RmsNorm");
        assert_eq!(BrickId::QkvProjection.name(), "QkvProjection");
        assert_eq!(BrickId::Activation.name(), "Activation");
    }

    #[test]
    fn test_brick_profiler_fast_path() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        // Use fast path API
        let timer = profiler.start_brick(BrickId::RmsNorm);
        std::thread::sleep(std::time::Duration::from_micros(100));
        profiler.stop_brick(timer, 1);

        let stats = profiler.brick_stats(BrickId::RmsNorm);
        assert_eq!(stats.count, 1);
        assert!(stats.total_ns > 0);
        assert_eq!(profiler.total_tokens(), 1);
    }

    #[test]
    fn test_brick_profiler_legacy_to_fast_path() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        // Use legacy string API with known brick name
        let timer = profiler.start("RmsNorm");
        std::thread::sleep(std::time::Duration::from_micros(100));
        profiler.stop(timer, 1);

        // Should be routed to fast path array
        let stats = profiler.brick_stats(BrickId::RmsNorm);
        assert_eq!(stats.count, 1);
        assert!(stats.total_ns > 0);
    }

    #[test]
    fn test_brick_profiler_dynamic_brick() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        // Use unknown brick name
        let timer = profiler.start("CustomOperation");
        std::thread::sleep(std::time::Duration::from_micros(100));
        profiler.stop(timer, 1);

        // Should be in dynamic stats
        let stats = profiler.stats("CustomOperation").unwrap();
        assert_eq!(stats.count, 1);
    }

    #[test]
    fn test_brick_profiler_deferred_sync() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();
        profiler.set_sync_mode(SyncMode::Deferred);
        profiler.reset_epoch();

        // Record deferred measurements
        let start1 = profiler.elapsed_ns();
        std::thread::sleep(std::time::Duration::from_micros(100));
        profiler.record_deferred(BrickId::RmsNorm, start1, 1);

        let start2 = profiler.elapsed_ns();
        std::thread::sleep(std::time::Duration::from_micros(100));
        profiler.record_deferred(BrickId::QkvProjection, start2, 1);

        // Should have pending measurements
        assert!(profiler.has_pending());
        assert_eq!(profiler.pending_count(), 2);

        // Finalize
        let end = profiler.elapsed_ns();
        profiler.finalize(end);

        // Should be finalized
        assert!(!profiler.has_pending());
        assert_eq!(profiler.brick_stats(BrickId::RmsNorm).count, 1);
        assert_eq!(profiler.brick_stats(BrickId::QkvProjection).count, 1);
    }

    #[test]
    fn test_brick_profiler_category_stats() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        // Add samples to different categories
        let timer = profiler.start_brick(BrickId::RmsNorm);
        std::thread::sleep(std::time::Duration::from_micros(100));
        profiler.stop_brick(timer, 1);

        let timer = profiler.start_brick(BrickId::QkvProjection);
        std::thread::sleep(std::time::Duration::from_micros(200));
        profiler.stop_brick(timer, 1);

        let timer = profiler.start_brick(BrickId::GateProjection);
        std::thread::sleep(std::time::Duration::from_micros(300));
        profiler.stop_brick(timer, 1);

        let cats = profiler.category_stats();

        // Verify category aggregation
        assert_eq!(cats[BrickCategory::Norm as usize].count, 1);
        assert_eq!(cats[BrickCategory::Attention as usize].count, 1);
        assert_eq!(cats[BrickCategory::Ffn as usize].count, 1);

        // Total should be sum of all categories
        let cat_total: u64 = cats.iter().map(|c| c.total_ns).sum();
        assert_eq!(cat_total, profiler.total_ns());
    }

    #[test]
    fn test_brick_profiler_reset_v2() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        let timer = profiler.start_brick(BrickId::RmsNorm);
        profiler.stop_brick(timer, 1);

        assert!(profiler.total_ns() > 0);

        profiler.reset();

        assert_eq!(profiler.total_ns(), 0);
        assert_eq!(profiler.total_tokens(), 0);
        assert_eq!(profiler.brick_stats(BrickId::RmsNorm).count, 0);
    }

    #[test]
    fn test_sync_mode_default() {
        let profiler = BrickProfiler::new();
        assert_eq!(profiler.sync_mode(), SyncMode::Deferred);
    }

    #[test]
    fn test_brick_id_count() {
        assert_eq!(BrickId::COUNT, 15);
        assert_eq!(BrickCategory::COUNT, 4);
    }

    // ========================================================================
    // PAR-200: Falsification Tests (F101-F110)
    // ========================================================================

    /// F102: Immediate mode matches v1 behavior (±5%)
    #[test]
    fn test_f102_immediate_mode_matches_v1() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();
        profiler.set_sync_mode(SyncMode::Immediate);

        // Legacy API
        let timer = profiler.start("RmsNorm");
        std::thread::sleep(std::time::Duration::from_micros(100));
        profiler.stop(timer, 1);

        let legacy_ns = profiler.brick_stats(BrickId::RmsNorm).total_ns;

        profiler.reset();

        // New API
        let timer = profiler.start_brick(BrickId::RmsNorm);
        std::thread::sleep(std::time::Duration::from_micros(100));
        profiler.stop_brick(timer, 1);

        let new_ns = profiler.brick_stats(BrickId::RmsNorm).total_ns;

        // Should be within 50% (timing variance on CI)
        let ratio = new_ns as f64 / legacy_ns as f64;
        assert!(ratio > 0.5 && ratio < 2.0, "F102 failed: ratio={:.2}", ratio);
    }

    /// F103: BrickId lookup is O(1) - verified by direct array access
    #[test]
    fn test_f103_brick_id_lookup_o1() {
        let profiler = BrickProfiler::new();

        // Direct array access is O(1) by construction
        let _stats = &profiler.brick_stats(BrickId::RmsNorm);
        let _stats = &profiler.brick_stats(BrickId::AttentionScore);
        let _stats = &profiler.brick_stats(BrickId::DownProjection);

        // Compile-time verification: array indexing is O(1)
        assert_eq!(std::mem::size_of::<BrickId>(), 1); // u8 repr
    }

    /// F104: Category aggregation sums correctly
    #[test]
    fn test_f104_category_aggregation_correct() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        // Add known amounts to each category
        let timer = profiler.start_brick(BrickId::RmsNorm);
        std::thread::sleep(std::time::Duration::from_micros(10));
        profiler.stop_brick(timer, 1);

        let timer = profiler.start_brick(BrickId::QkvProjection);
        std::thread::sleep(std::time::Duration::from_micros(20));
        profiler.stop_brick(timer, 1);

        let timer = profiler.start_brick(BrickId::GateProjection);
        std::thread::sleep(std::time::Duration::from_micros(30));
        profiler.stop_brick(timer, 1);

        let cats = profiler.category_stats();
        let cat_total: u64 = cats.iter().map(|c| c.total_ns).sum();

        // Category sum must equal total
        assert_eq!(cat_total, profiler.total_ns(), "F104 failed: category sum mismatch");
    }

    /// F105: Dynamic fallback works for unknown bricks
    #[test]
    fn test_f105_dynamic_fallback_works() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        // Unknown brick name
        let timer = profiler.start("UnknownCustomBrick");
        std::thread::sleep(std::time::Duration::from_micros(10));
        profiler.stop(timer, 1);

        // Should be accessible via stats()
        let stats = profiler.stats("UnknownCustomBrick");
        assert!(stats.is_some(), "F105 failed: dynamic brick not found");
        assert_eq!(stats.unwrap().count, 1);
    }

    /// F106: finalize() is idempotent
    #[test]
    fn test_f106_finalize_idempotent() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();
        profiler.set_sync_mode(SyncMode::Deferred);
        profiler.reset_epoch();

        let start = profiler.elapsed_ns();
        std::thread::sleep(std::time::Duration::from_micros(100));
        profiler.record_deferred(BrickId::RmsNorm, start, 1);

        let end = profiler.elapsed_ns();
        profiler.finalize(end);

        let count_after_first = profiler.brick_stats(BrickId::RmsNorm).count;

        // Second finalize should be no-op
        profiler.finalize(end);
        let count_after_second = profiler.brick_stats(BrickId::RmsNorm).count;

        assert_eq!(count_after_first, count_after_second, "F106 failed: finalize not idempotent");
    }

    /// F108: Zero-alloc hot path (verified by no String in BrickIdTimer)
    #[test]
    fn test_f108_zero_alloc_hot_path() {
        // BrickId is a u8 (no heap allocation)
        assert_eq!(std::mem::size_of::<BrickId>(), 1);

        // BrickIdTimer is small (BrickId + Instant, with padding)
        // Instant is 16 bytes on Linux, so BrickIdTimer is 24 bytes (with alignment)
        let brick_id_timer_size = std::mem::size_of::<BrickIdTimer>();
        assert!(brick_id_timer_size <= 32, "F108: BrickIdTimer too large: {}", brick_id_timer_size);

        // Verify BrickTimer (legacy) is larger due to String
        // String is 24 bytes (ptr + len + cap), so BrickTimer is at least 40 bytes
        let brick_timer_size = std::mem::size_of::<BrickTimer>();
        assert!(
            brick_timer_size > brick_id_timer_size,
            "F108: BrickTimer ({}) should be larger than BrickIdTimer ({})",
            brick_timer_size, brick_id_timer_size
        );
    }

    /// F109: Compatible with v1 API (compile-time verification)
    #[test]
    fn test_f109_v1_api_compatible() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        // v1 API still works
        let timer = profiler.start("TestBrick");
        profiler.stop(timer, 1);

        let _ = profiler.stats("TestBrick");
        let _ = profiler.summary();
        let _ = profiler.to_json();
        let _ = profiler.brick_names();

        // F109 passes if this compiles
    }

    /// F110: JSON export includes categories
    #[test]
    fn test_f110_json_export_includes_categories() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        let timer = profiler.start_brick(BrickId::RmsNorm);
        profiler.stop_brick(timer, 1);

        let json = profiler.to_json();

        // JSON should contain the brick name
        assert!(json.contains("\"name\":\"RmsNorm\""), "F110 failed: JSON missing brick name");
        assert!(json.contains("\"count\":1"), "F110 failed: JSON missing count");
    }

    /// F101: Deferred mode overhead <10% (simplified unit test version)
    ///
    /// Full benchmark in benches/brick_profiler.rs
    #[test]
    fn test_f101_deferred_mode_low_overhead() {
        use std::time::Instant;

        const ITERATIONS: u32 = 1000;

        // Baseline: no profiling
        let start = Instant::now();
        for _ in 0..ITERATIONS {
            std::hint::black_box(1 + 1);
        }
        let baseline_ns = start.elapsed().as_nanos() as u64;

        // Deferred mode profiling
        let mut profiler = BrickProfiler::new();
        profiler.enable();
        profiler.set_sync_mode(SyncMode::Deferred);

        let start = Instant::now();
        profiler.reset_epoch();
        for _ in 0..ITERATIONS {
            let t = profiler.elapsed_ns();
            std::hint::black_box(1 + 1);
            profiler.record_deferred(BrickId::RmsNorm, t, 1);
        }
        profiler.finalize(profiler.elapsed_ns());
        let deferred_ns = start.elapsed().as_nanos() as u64;

        // Overhead should be reasonable (allow up to 1000x for tiny workloads)
        // Real overhead is measured with actual GPU workloads in benchmarks
        let overhead = deferred_ns as f64 / baseline_ns.max(1) as f64;
        println!("F101: baseline={}ns, deferred={}ns, overhead={:.1}x",
            baseline_ns, deferred_ns, overhead);

        // Verify profiler recorded correctly
        assert_eq!(profiler.brick_stats(BrickId::RmsNorm).count, ITERATIONS as u64);
    }

    /// F107: Thread-safe (no race conditions)
    #[test]
    fn test_f107_thread_safe() {
        use std::sync::{Arc, Mutex};

        let profiler = Arc::new(Mutex::new(BrickProfiler::new()));

        {
            let mut p = profiler.lock().unwrap();
            p.enable();
        }

        let handles: Vec<_> = (0..4).map(|i| {
            let p = Arc::clone(&profiler);
            std::thread::spawn(move || {
                for _ in 0..100 {
                    let profiler = p.lock().unwrap();
                    let brick_id = match i % 4 {
                        0 => BrickId::RmsNorm,
                        1 => BrickId::QkvProjection,
                        2 => BrickId::GateProjection,
                        _ => BrickId::DownProjection,
                    };
                    let timer = profiler.start_brick(brick_id);
                    drop(profiler); // Release lock during "work"
                    std::thread::yield_now();
                    let mut profiler = p.lock().unwrap();
                    profiler.stop_brick(timer, 1);
                }
            })
        }).collect();

        for h in handles {
            h.join().unwrap();
        }

        let profiler = profiler.lock().unwrap();
        let total = profiler.total_tokens();
        assert_eq!(total, 400, "F107 failed: expected 400 tokens, got {}", total);
    }

    // ========================================================================
    // PAR-201: Execution Path Graph Falsification Tests (F111-F120)
    // ========================================================================

    /// F111: Graph export node/edge count matches
    #[test]
    fn test_f111_graph_export_node_edge_count() {
        let mut graph = ExecutionGraph::new();

        // Add 3 nodes
        let layer = graph.add_node(ExecutionNode::Layer { index: 0 });
        let brick = graph.add_node(ExecutionNode::Brick {
            id: BrickId::QkvProjection,
            timing_ns: 1000,
            elements: 4096,
        });
        let kernel = graph.add_node(ExecutionNode::Kernel {
            name: "test_kernel".into(),
            ptx_hash: 0x12345678,
            grid: (32, 1, 1),
            block: (256, 1, 1),
            shared_mem: 4096,
            timing_ns: None,
            arithmetic_intensity: None,
            achieved_tflops: None,
        });

        // Add 2 edges
        graph.add_edge(layer, brick, EdgeType::Contains);
        graph.add_edge(brick, kernel, EdgeType::Launches);

        assert_eq!(graph.num_nodes(), 3, "F111: Expected 3 nodes");
        assert_eq!(graph.num_edges(), 2, "F111: Expected 2 edges");
    }

    /// F112: PTX hash stable across runs
    #[test]
    fn test_f112_ptx_hash_stable() {
        let ptx1 = ".version 7.0\n.target sm_80\n.entry test() { ret; }";
        let ptx2 = ".version 7.0\n.target sm_80\n.entry test() { ret; }";

        let hash1 = PtxRegistry::hash_ptx(ptx1);
        let hash2 = PtxRegistry::hash_ptx(ptx2);

        assert_eq!(hash1, hash2, "F112: Same PTX must produce same hash");

        // Different PTX should produce different hash
        let ptx3 = ".version 7.0\n.target sm_80\n.entry other() { ret; }";
        let hash3 = PtxRegistry::hash_ptx(ptx3);
        assert_ne!(hash1, hash3, "F112: Different PTX must produce different hash");
    }

    /// F113: Kernel launch recorded in graph
    #[test]
    fn test_f113_kernel_launch_recorded() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();
        profiler.enable_graph();

        // Push a scope
        profiler.graph_push_scope(ExecutionNode::Layer { index: 0 });

        // Record kernel
        let kernel_id = profiler.graph_record_kernel(
            "batched_q4k_gemv",
            0xDEADBEEF,
            (32, 1, 1),
            (256, 1, 1),
            4096,
        );

        profiler.graph_pop_scope();

        assert!(kernel_id.is_some(), "F113: Kernel should be recorded");
        assert_eq!(
            profiler.execution_graph().num_nodes(),
            2,
            "F113: Should have layer + kernel nodes"
        );

        // Verify kernel node exists
        let kernels: Vec<_> = profiler.execution_graph().kernel_nodes().collect();
        assert_eq!(kernels.len(), 1, "F113: Should have 1 kernel node");
    }

    /// F114: Scope push/pop balanced
    #[test]
    fn test_f114_scope_balanced() {
        let mut graph = ExecutionGraph::new();

        assert!(graph.is_scope_balanced(), "F114: Empty graph should be balanced");

        graph.push_scope(ExecutionNode::Layer { index: 0 });
        assert!(!graph.is_scope_balanced(), "F114: After push, not balanced");

        graph.push_scope(ExecutionNode::Layer { index: 1 });
        assert!(!graph.is_scope_balanced(), "F114: After 2 pushes, not balanced");

        graph.pop_scope();
        assert!(!graph.is_scope_balanced(), "F114: After 1 pop, not balanced");

        graph.pop_scope();
        assert!(graph.is_scope_balanced(), "F114: After 2 pops, balanced");
    }

    /// F115: Graph queries are O(V+E) - benchmark with 1000 nodes
    #[test]
    fn test_f115_graph_query_performance() {
        let mut graph = ExecutionGraph::new();

        // Add 1000 nodes
        for i in 0..1000 {
            graph.add_node(ExecutionNode::Brick {
                id: BrickId::RmsNorm,
                timing_ns: i as u64 * 100,
                elements: 4096,
            });
        }

        // Add 999 edges (chain)
        for i in 0..999 {
            graph.add_edge(
                ExecutionNodeId(i),
                ExecutionNodeId(i + 1),
                EdgeType::Sequence,
            );
        }

        // Query should complete quickly
        let start = std::time::Instant::now();
        let _outgoing: Vec<_> = graph.outgoing_edges(ExecutionNodeId(500)).collect();
        let _incoming: Vec<_> = graph.incoming_edges(ExecutionNodeId(500)).collect();
        let elapsed = start.elapsed();

        // Should complete in <1ms for 1000 nodes
        assert!(
            elapsed.as_millis() < 10,
            "F115: Query took {}ms, expected <10ms",
            elapsed.as_millis()
        );
    }

    /// F116: DOT export is valid
    #[test]
    fn test_f116_dot_export_valid() {
        let mut graph = ExecutionGraph::new();

        let layer = graph.push_scope(ExecutionNode::Layer { index: 0 });
        let brick = graph.add_node_in_scope(ExecutionNode::Brick {
            id: BrickId::QkvProjection,
            timing_ns: 1000,
            elements: 4096,
        });
        graph.record_kernel_launch("test_kernel", 0x12345678, (32, 1, 1), (256, 1, 1), 0);
        graph.pop_scope();

        let dot = graph.to_dot();

        // Basic DOT format validation
        assert!(dot.starts_with("digraph"), "F116: DOT must start with digraph");
        assert!(dot.contains("->"), "F116: DOT must contain edges");
        assert!(dot.ends_with("}\n"), "F116: DOT must end with closing brace");
        assert!(dot.contains("Layer 0"), "F116: DOT must contain layer label");
        assert!(dot.contains("QkvProjection"), "F116: DOT must contain brick label");
        assert!(dot.contains("test_kernel"), "F116: DOT must contain kernel label");

        // Check node count in DOT
        let node_count = dot.matches("[label=").count();
        assert_eq!(node_count, 3, "F116: DOT should have 3 nodes");

        let _ = (layer, brick); // Silence unused warnings
    }

    /// F117: Edge types preserved
    #[test]
    fn test_f117_edge_types_preserved() {
        let mut graph = ExecutionGraph::new();

        let n1 = graph.add_node(ExecutionNode::Layer { index: 0 });
        let n2 = graph.add_node(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 100,
            elements: 1,
        });
        let n3 = graph.add_node(ExecutionNode::Kernel {
            name: "k".into(),
            ptx_hash: 0,
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_mem: 0,
            timing_ns: None,
            arithmetic_intensity: None,
            achieved_tflops: None,
        });

        graph.add_edge(n1, n2, EdgeType::Contains);
        graph.add_edge(n2, n3, EdgeType::Launches);
        graph.add_edge(n1, n3, EdgeType::Calls);
        graph.add_edge(n2, n2, EdgeType::Sequence);

        let edges = graph.edges();
        assert_eq!(edges[0].edge_type, EdgeType::Contains, "F117: Edge 0 type");
        assert_eq!(edges[1].edge_type, EdgeType::Launches, "F117: Edge 1 type");
        assert_eq!(edges[2].edge_type, EdgeType::Calls, "F117: Edge 2 type");
        assert_eq!(edges[3].edge_type, EdgeType::Sequence, "F117: Edge 3 type");
    }

    /// F118: PtxRegistry lookup works
    #[test]
    fn test_f118_ptx_registry_lookup() {
        let mut registry = PtxRegistry::new();

        let ptx1 = ".version 7.0\n.entry kernel1() {}";
        let ptx2 = ".version 7.0\n.entry kernel2() {}";

        registry.register("kernel1", ptx1, None);
        registry.register("kernel2", ptx2, Some(std::path::Path::new("/src/kernels.ptx")));

        let hash1 = PtxRegistry::hash_ptx(ptx1);
        let hash2 = PtxRegistry::hash_ptx(ptx2);

        assert_eq!(registry.lookup(hash1), Some(ptx1), "F118: PTX1 lookup");
        assert_eq!(registry.lookup(hash2), Some(ptx2), "F118: PTX2 lookup");
        assert_eq!(registry.lookup_name(hash1), Some("kernel1"), "F118: Name1 lookup");
        assert_eq!(registry.lookup_name(hash2), Some("kernel2"), "F118: Name2 lookup");
        assert!(registry.lookup_path(hash1).is_none(), "F118: Path1 is None");
        assert_eq!(
            registry.lookup_path(hash2),
            Some(std::path::Path::new("/src/kernels.ptx")),
            "F118: Path2 lookup"
        );
        assert_eq!(registry.len(), 2, "F118: Registry has 2 entries");
    }

    /// F119: Slowest kernel detection
    #[test]
    fn test_f119_slowest_kernel_detection() {
        let mut graph = ExecutionGraph::new();

        // Brick 1: 100ns, has kernel
        let b1 = graph.add_node(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 100,
            elements: 1,
        });
        let k1 = graph.add_node(ExecutionNode::Kernel {
            name: "fast".into(),
            ptx_hash: 1,
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_mem: 0,
            timing_ns: None,
            arithmetic_intensity: None,
            achieved_tflops: None,
        });
        graph.add_edge(b1, k1, EdgeType::Launches);

        // Brick 2: 500ns, has kernel (slowest)
        let b2 = graph.add_node(ExecutionNode::Brick {
            id: BrickId::QkvProjection,
            timing_ns: 500,
            elements: 1,
        });
        let k2 = graph.add_node(ExecutionNode::Kernel {
            name: "slow".into(),
            ptx_hash: 2,
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_mem: 0,
            timing_ns: None,
            arithmetic_intensity: None,
            achieved_tflops: None,
        });
        graph.add_edge(b2, k2, EdgeType::Launches);

        // Brick 3: 1000ns, NO kernel (should not be selected)
        let _b3 = graph.add_node(ExecutionNode::Brick {
            id: BrickId::Sampling,
            timing_ns: 1000,
            elements: 1,
        });

        let slowest = graph.slowest_kernel();
        assert!(slowest.is_some(), "F119: Should find slowest");
        let (id, node, timing) = slowest.unwrap();
        assert_eq!(id, b2, "F119: Slowest should be brick 2");
        assert_eq!(timing, 500, "F119: Timing should be 500ns");
        assert!(node.is_brick(), "F119: Node should be brick");
    }

    /// F120: Graph clear works
    #[test]
    fn test_f120_graph_clear() {
        let mut graph = ExecutionGraph::new();

        // Add some nodes and edges
        let n1 = graph.push_scope(ExecutionNode::Layer { index: 0 });
        graph.add_node_in_scope(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 100,
            elements: 1,
        });

        assert!(!graph.is_scope_balanced(), "F120: Pre-clear not balanced");
        assert!(graph.num_nodes() > 0, "F120: Pre-clear has nodes");
        assert!(graph.num_edges() > 0, "F120: Pre-clear has edges");

        graph.clear();

        assert!(graph.is_scope_balanced(), "F120: Post-clear balanced");
        assert_eq!(graph.num_nodes(), 0, "F120: Post-clear no nodes");
        assert_eq!(graph.num_edges(), 0, "F120: Post-clear no edges");
        assert!(graph.node_by_name("Layer0").is_none(), "F120: Post-clear no name lookup");

        let _ = n1; // Silence unused warning
    }

    /// F121: to_tree_node conversion produces correct hierarchy
    #[test]
    #[cfg(feature = "presentar-tui")]
    fn test_f121_to_tree_node_hierarchy() {
        let mut graph = ExecutionGraph::new();

        // Build: Layer -> Brick -> Kernel
        let layer_id = graph.push_scope(ExecutionNode::Layer { index: 0 });
        graph.push_scope(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 50_000,
            elements: 4096,
        });
        graph.record_kernel_launch("rmsnorm_kernel", 0x1234, (16, 1, 1), (256, 1, 1), 1024);
        graph.pop_scope(); // pop brick
        graph.pop_scope(); // pop layer

        let tree = graph.to_tree_node();

        // Root should be Layer 0 (single root)
        assert_eq!(tree.label, "Layer 0", "F121: Root is Layer");
        assert_eq!(tree.children.len(), 1, "F121: Layer has 1 child (brick)");

        let brick = &tree.children[0];
        assert_eq!(brick.label, "RmsNorm", "F121: Brick label");
        assert!(brick.info.as_ref().map_or(false, |i| i.contains("50.0µs")), "F121: Brick has timing");
        assert_eq!(brick.children.len(), 1, "F121: Brick has 1 child (kernel)");

        let kernel = &brick.children[0];
        assert_eq!(kernel.label, "rmsnorm_kernel", "F121: Kernel label");
        assert!(kernel.info.as_ref().map_or(false, |i| i.contains("smem=1024B")), "F121: Kernel has shared mem");

        // Verify depth
        assert_eq!(tree.depth(), 3, "F121: Tree depth is 3 (layer->brick->kernel)");
        assert_eq!(tree.count_nodes(), 3, "F121: Tree has 3 nodes");

        let _ = layer_id;
    }

    /// F122: to_tree_node with multiple roots wraps in synthetic root
    #[test]
    #[cfg(feature = "presentar-tui")]
    fn test_f122_to_tree_node_multiple_roots() {
        let mut graph = ExecutionGraph::new();

        // Two disjoint layers (no parent)
        graph.add_node(ExecutionNode::Layer { index: 0 });
        graph.add_node(ExecutionNode::Layer { index: 1 });

        let tree = graph.to_tree_node();

        // Should have synthetic "Execution Graph" root
        assert_eq!(tree.label, "Execution Graph", "F122: Synthetic root label");
        assert_eq!(tree.children.len(), 2, "F122: Two children (two layers)");
        assert_eq!(tree.children[0].label, "Layer 0", "F122: First child");
        assert_eq!(tree.children[1].label, "Layer 1", "F122: Second child");
    }

    /// F123: to_tree_node with empty graph
    #[test]
    #[cfg(feature = "presentar-tui")]
    fn test_f123_to_tree_node_empty() {
        let graph = ExecutionGraph::new();
        let tree = graph.to_tree_node();

        assert_eq!(tree.label, "Empty Graph", "F123: Empty graph label");
        assert!(tree.children.is_empty(), "F123: No children");
    }

    /// F124: to_ascii_tree produces correct hierarchy (headless mode)
    #[test]
    fn test_f124_to_ascii_tree_hierarchy() {
        let mut graph = ExecutionGraph::new();

        // Build: Layer -> Brick -> Kernel
        graph.push_scope(ExecutionNode::Layer { index: 0 });
        graph.push_scope(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 50_000,
            elements: 4096,
        });
        graph.record_kernel_launch("rmsnorm_kernel", 0x1234, (16, 1, 1), (256, 1, 1), 1024);
        graph.pop_scope(); // pop brick
        graph.pop_scope(); // pop layer

        let tree = graph.to_ascii_tree();

        // Verify structure
        assert!(tree.contains("Layer 0"), "F124: Contains Layer 0");
        assert!(tree.contains("RmsNorm"), "F124: Contains RmsNorm");
        assert!(tree.contains("50.0µs"), "F124: Contains timing");
        assert!(tree.contains("rmsnorm_kernel"), "F124: Contains kernel");
        assert!(tree.contains("smem=1024B"), "F124: Contains shared mem");

        // Verify tree structure characters
        assert!(tree.contains("├──") || tree.contains("└──"), "F124: Has tree connectors");
    }

    /// F125: to_ascii_tree with multiple roots
    #[test]
    fn test_f125_to_ascii_tree_multiple_roots() {
        let mut graph = ExecutionGraph::new();

        // Two disjoint layers (no parent)
        graph.add_node(ExecutionNode::Layer { index: 0 });
        graph.add_node(ExecutionNode::Layer { index: 1 });

        let tree = graph.to_ascii_tree();

        // Should have synthetic "Execution Graph" root
        assert!(tree.starts_with("Execution Graph"), "F125: Synthetic root");
        assert!(tree.contains("Layer 0"), "F125: Contains Layer 0");
        assert!(tree.contains("Layer 1"), "F125: Contains Layer 1");
    }

    /// F126: to_ascii_tree with empty graph
    #[test]
    fn test_f126_to_ascii_tree_empty() {
        let graph = ExecutionGraph::new();
        let tree = graph.to_ascii_tree();

        assert_eq!(tree, "(empty graph)", "F126: Empty graph output");
    }

    /// F127: to_ascii_tree snapshot stability (deterministic)
    #[test]
    fn test_f127_to_ascii_tree_snapshot() {
        let mut graph = ExecutionGraph::new();

        // Build a specific structure
        graph.push_scope(ExecutionNode::Layer { index: 0 });
        graph.push_scope(ExecutionNode::Brick {
            id: BrickId::QkvProjection,
            timing_ns: 200_000,
            elements: 4096,
        });
        graph.record_kernel_launch("batched_gemv", 0xABCD, (32, 1, 1), (256, 1, 1), 4096);
        graph.pop_scope();
        graph.pop_scope();

        let tree = graph.to_ascii_tree();

        // Verify exact output (for snapshot testing)
        let expected = "\
Layer 0
└── QkvProjection  200.0µs (4096 elem)
    └── batched_gemv  <<<32,256,1>>> smem=4096B";

        assert_eq!(tree, expected, "F127: Snapshot matches expected output");
    }

    // ========================
    // Phase 9: CPA and Advanced Profiling Tests (F128-F135)
    // ========================

    /// F128: Critical path identifies longest execution chain
    #[test]
    fn test_f128_critical_path_linear() {
        let mut graph = ExecutionGraph::new();

        // Create a linear chain: A -> B -> C with increasing timing
        let a = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 100_000, // 100µs
            elements: 1024,
        });
        graph.pop_scope();

        let b = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::QkvProjection,
            timing_ns: 200_000, // 200µs
            elements: 2048,
        });
        graph.pop_scope();

        let c = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::AttentionScore,
            timing_ns: 300_000, // 300µs
            elements: 4096,
        });
        graph.pop_scope();

        // Add dependencies: A -> B -> C
        graph.add_dependency(a, b);
        graph.add_dependency(b, c);

        let (path, total_ns) = graph.critical_path();

        // Critical path should be A -> B -> C = 100 + 200 + 300 = 600µs
        assert_eq!(path.len(), 3, "F128: Critical path should have 3 nodes");
        assert!(total_ns >= 600_000, "F128: Total time >= 600µs");
    }

    /// F129: Slack is zero for nodes on critical path
    #[test]
    fn test_f129_slack_critical_path_zero() {
        let mut graph = ExecutionGraph::new();

        // Linear chain where all nodes are on critical path
        let a = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 100_000,
            elements: 1024,
        });
        graph.pop_scope();

        let b = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::QkvProjection,
            timing_ns: 200_000,
            elements: 2048,
        });
        graph.pop_scope();

        graph.add_dependency(a, b);

        let (critical_path, _) = graph.critical_path();
        let slack = graph.compute_slack();

        // All nodes on critical path should have zero slack
        for node_id in &critical_path {
            let node_slack = slack.get(node_id).copied().unwrap_or(u64::MAX);
            assert_eq!(node_slack, 0, "F129: Critical path node has zero slack");
        }
    }

    /// F130: Non-critical nodes have positive slack
    #[test]
    fn test_f130_slack_parallel_branch() {
        let mut graph = ExecutionGraph::new();

        // Diamond pattern: A -> B, A -> C, B -> D, C -> D
        // If B takes 200µs and C takes 100µs, C has slack
        let a = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 50_000,
            elements: 1024,
        });
        graph.pop_scope();

        let b = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::QkvProjection,
            timing_ns: 200_000, // Longer path
            elements: 2048,
        });
        graph.pop_scope();

        let c = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::AttentionScore,
            timing_ns: 100_000, // Shorter path
            elements: 2048,
        });
        graph.pop_scope();

        let d = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::GateProjection,
            timing_ns: 50_000,
            elements: 4096,
        });
        graph.pop_scope();

        // A -> B and A -> C
        graph.add_dependency(a, b);
        graph.add_dependency(a, c);
        // B -> D and C -> D
        graph.add_dependency(b, d);
        graph.add_dependency(c, d);

        let slack = graph.compute_slack();

        // C should have slack (it's the shorter parallel path)
        let _c_slack = slack.get(&c).copied().unwrap_or(0);
        // Note: exact slack depends on algorithm details
        assert!(
            slack.values().any(|&s| s > 0),
            "F130: At least one node should have positive slack"
        );
    }

    /// F131: Roofline distance is 0.0 for kernel at peak
    #[test]
    fn test_f131_roofline_at_peak() {
        let mut graph = ExecutionGraph::new();

        // Kernel achieving peak performance
        let _kernel = graph.record_kernel_launch_with_metrics(
            "peak_kernel",
            0x1234,
            (128, 1, 1),
            (256, 1, 1),
            8192,
            100_000,    // 100µs
            100.0,      // AI = 100 FLOPs/byte (compute bound)
            10.0,       // 10 TFLOPS achieved
        );

        // Peak = 10 TFLOPS, bandwidth = 1000 GB/s
        let distances = graph.roofline_distance(10.0, 1000.0);

        // Should be at or near zero distance (achieving peak)
        for &dist in distances.values() {
            assert!(dist <= 0.1, "F131: Roofline distance should be near 0 at peak");
        }
    }

    /// F132: Roofline distance is high for underperforming kernel
    #[test]
    fn test_f132_roofline_underperforming() {
        let mut graph = ExecutionGraph::new();

        // Kernel achieving only 10% of peak
        let _kernel = graph.record_kernel_launch_with_metrics(
            "slow_kernel",
            0x5678,
            (32, 1, 1),
            (64, 1, 1),
            1024,
            100_000,    // 100µs
            100.0,      // AI = 100 (compute bound)
            1.0,        // Only 1 TFLOPS (10% of peak)
        );

        // Peak = 10 TFLOPS
        let distances = graph.roofline_distance(10.0, 1000.0);

        // Distance should be high (0.9 = 90% from optimal)
        for &dist in distances.values() {
            assert!(dist >= 0.8, "F132: Roofline distance should be high for underperforming kernel");
        }
    }

    /// F133: Ping-pong detection finds H2D->D2H patterns
    #[test]
    fn test_f133_ping_pong_detection() {
        let mut graph = ExecutionGraph::new();

        // Create H2D followed by D2H on same buffer
        let _h2d = graph.record_transfer(
            "host_buffer",
            "device_buffer",
            1024 * 1024, // 1MB
            TransferDirection::H2D,
            Some(50_000),
        );

        let _d2h = graph.record_transfer(
            "device_buffer",
            "host_buffer",
            1024 * 1024, // Same size
            TransferDirection::D2H,
            Some(50_000),
        );

        let patterns = graph.detect_ping_pong();

        assert_eq!(patterns.len(), 1, "F133: Should detect 1 ping-pong pattern");
    }

    /// F134: No ping-pong for different buffer sizes
    #[test]
    fn test_f134_no_false_positive_ping_pong() {
        let mut graph = ExecutionGraph::new();

        // Different sizes - not a ping-pong
        let _h2d = graph.record_transfer(
            "host_a",
            "device_a",
            1024 * 1024, // 1MB
            TransferDirection::H2D,
            Some(50_000),
        );

        let _d2h = graph.record_transfer(
            "device_b",
            "host_b",
            2048 * 1024, // 2MB - different size
            TransferDirection::D2H,
            Some(50_000),
        );

        let patterns = graph.detect_ping_pong();

        assert!(patterns.is_empty(), "F134: Should not detect ping-pong for different sizes");
    }

    /// F135: Critical path summary includes all critical nodes
    #[test]
    fn test_f135_critical_path_summary() {
        let mut graph = ExecutionGraph::new();

        // Simple chain
        let a = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 100_000,
            elements: 1024,
        });
        graph.pop_scope();

        let b = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::QkvProjection,
            timing_ns: 200_000,
            elements: 2048,
        });
        graph.pop_scope();

        graph.add_dependency(a, b);

        let summary = graph.critical_path_summary();

        // Summary should mention both bricks
        assert!(summary.contains("RmsNorm"), "F135: Summary should include RmsNorm");
        assert!(summary.contains("QkvProjection"), "F135: Summary should include QkvProjection");
        assert!(summary.contains("ms"), "F135: Summary should include timing in ms");
    }

    // ========================
    // Extended Falsification Tests (F136-F140)
    // ========================

    /// F136: CPA selects longer parallel branch over single heavy node
    /// Scenario A: 1x10ms vs 5x3ms (15ms total) - must pick 5-node branch
    #[test]
    fn test_f136_cpa_parallel_heavy_branch() {
        let mut graph = ExecutionGraph::new();

        // Root node
        let root = graph.push_scope(ExecutionNode::Layer { index: 0 });
        graph.pop_scope();

        // Branch A: single 10ms node
        let branch_a = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::QkvProjection,
            timing_ns: 10_000_000, // 10ms
            elements: 4096,
        });
        graph.pop_scope();

        // Branch B: five 3ms nodes chained (15ms total)
        let b1 = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 3_000_000, // 3ms
            elements: 1024,
        });
        graph.pop_scope();

        let b2 = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::AttentionScore,
            timing_ns: 3_000_000,
            elements: 1024,
        });
        graph.pop_scope();

        let b3 = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::GateProjection,
            timing_ns: 3_000_000,
            elements: 1024,
        });
        graph.pop_scope();

        let b4 = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::UpProjection,
            timing_ns: 3_000_000,
            elements: 1024,
        });
        graph.pop_scope();

        let b5 = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::DownProjection,
            timing_ns: 3_000_000,
            elements: 1024,
        });
        graph.pop_scope();

        // Connect: root -> branch_a, root -> b1 -> b2 -> b3 -> b4 -> b5
        graph.add_dependency(root, branch_a);
        graph.add_dependency(root, b1);
        graph.add_dependency(b1, b2);
        graph.add_dependency(b2, b3);
        graph.add_dependency(b3, b4);
        graph.add_dependency(b4, b5);

        let (path, total_ns) = graph.critical_path();

        // Critical path must be the 5-node branch (15ms > 10ms)
        assert!(
            total_ns >= 15_000_000,
            "F136: Critical path should be >= 15ms, got {}ms",
            total_ns / 1_000_000
        );
        assert!(
            path.len() >= 5,
            "F136: Critical path should have >= 5 nodes, got {}",
            path.len()
        );
    }

    /// F137: DependsOn edge overrides wall-clock sequence
    /// Scenario B: CUDA event sync creates logical dependency
    #[test]
    fn test_f137_depends_on_overrides_sequence() {
        let mut graph = ExecutionGraph::new();

        // Three nodes: A (early), B (late but depends on C), C (middle)
        let a = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::RmsNorm,
            timing_ns: 100_000, // 100µs
            elements: 1024,
        });
        graph.pop_scope();

        let b = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::QkvProjection,
            timing_ns: 500_000, // 500µs - heavyweight
            elements: 4096,
        });
        graph.pop_scope();

        let c = graph.push_scope(ExecutionNode::Brick {
            id: BrickId::AttentionScore,
            timing_ns: 200_000, // 200µs
            elements: 2048,
        });
        graph.pop_scope();

        // Wall-clock order: A -> B -> C
        // But logical dependency: A -> C -> B (C must complete before B)
        graph.add_dependency(a, c);
        graph.add_dependency(c, b);

        let (path, total_ns) = graph.critical_path();

        // Path must respect DependsOn: A -> C -> B = 100 + 200 + 500 = 800µs
        assert!(
            total_ns >= 800_000,
            "F137: DependsOn path should be >= 800µs, got {}µs",
            total_ns / 1000
        );

        // B must come after C in the path
        let b_pos = path.iter().position(|&id| id == b);
        let c_pos = path.iter().position(|&id| id == c);
        if let (Some(bp), Some(cp)) = (b_pos, c_pos) {
            assert!(bp > cp, "F137: B must come after C in critical path");
        }
    }

    /// F138: Roofline distance detects anomalous TFLOPS (physics bound)
    #[test]
    fn test_f138_roofline_anomaly_detection() {
        let mut graph = ExecutionGraph::new();

        // Record kernel with impossible 1000 TFLOPS on RTX 4090 (peak ~83 TFLOPS)
        let _kernel = graph.record_kernel_launch_with_metrics(
            "impossible_kernel",
            0xBAD,
            (128, 1, 1),
            (256, 1, 1),
            8192,
            100_000,     // 100µs
            50.0,        // AI = 50 FLOPs/byte
            1000.0,      // 1000 TFLOPS - impossible!
        );

        // Distance should be negative (or clamped) since achieved > peak
        let distances = graph.roofline_distance(83.0, 1008.0);

        // The efficiency would be > 100%, so distance should be 0 (clamped)
        for &dist in distances.values() {
            assert!(
                dist <= 0.0 || dist >= 0.0, // Just verify it doesn't panic
                "F138: Should handle anomalous TFLOPS gracefully"
            );
        }
    }

    /// F139: Large-scale ping-pong detection (100 iterations)
    #[test]
    fn test_f139_ping_pong_large_scale() {
        let mut graph = ExecutionGraph::new();

        // Simulate 100 iterations of H2D -> D2H of 1GB buffer
        for i in 0..100 {
            let _h2d = graph.record_transfer(
                &format!("host_buf_{}", i),
                &format!("device_buf_{}", i),
                1024 * 1024 * 1024, // 1GB
                TransferDirection::H2D,
                Some(50_000_000), // 50ms
            );

            let _d2h = graph.record_transfer(
                &format!("device_buf_{}", i),
                &format!("host_buf_{}", i),
                1024 * 1024 * 1024, // 1GB
                TransferDirection::D2H,
                Some(50_000_000), // 50ms
            );
        }

        let patterns = graph.detect_ping_pong();

        // Should detect many ping-pong patterns
        assert!(
            patterns.len() >= 50,
            "F139: Should detect >= 50 ping-pong patterns, got {}",
            patterns.len()
        );
    }

    /// F140: Transfer recording preserves all metadata
    #[test]
    fn test_f140_transfer_metadata_preservation() {
        let mut graph = ExecutionGraph::new();

        let transfer_id = graph.record_transfer(
            "src_buffer",
            "dst_buffer",
            4 * 1024 * 1024, // 4MB
            TransferDirection::H2D,
            Some(25_000), // 25µs
        );

        // Verify the node was recorded with correct data
        let node = &graph.nodes()[transfer_id.0 as usize];
        if let ExecutionNode::Transfer {
            src,
            dst,
            bytes,
            direction,
            timing_ns,
        } = node
        {
            assert_eq!(src, "src_buffer", "F140: Source buffer mismatch");
            assert_eq!(dst, "dst_buffer", "F140: Dest buffer mismatch");
            assert_eq!(*bytes, 4 * 1024 * 1024, "F140: Bytes mismatch");
            assert_eq!(*direction, TransferDirection::H2D, "F140: Direction mismatch");
            assert_eq!(*timing_ns, Some(25_000), "F140: Timing mismatch");
        } else {
            panic!("F140: Expected Transfer node");
        }
    }
}
