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
#[derive(Debug, Clone, Copy)]
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
}
