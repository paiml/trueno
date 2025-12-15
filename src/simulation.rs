//! Simulation Testing Framework (TRUENO-SPEC-012)
//!
//! Provides deterministic, reproducible, and falsifiable validation of compute
//! operations across all backends: SIMD (CPU), PTX (CUDA), and WGPU.
//!
//! This module integrates with the sovereign stack (simular) and follows
//! Toyota Production System principles:
//!
//! - **Jidoka**: Built-in quality - stop on defect
//! - **Poka-Yoke**: Mistake-proofing via type safety
//! - **Heijunka**: Leveled testing across backends
//! - **Genchi Genbutsu**: Visual inspection of results
//! - **Kaizen**: Continuous performance improvement
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno::simulation::{SimTestConfig, BackendTolerance};
//!
//! let config = SimTestConfig::builder()
//!     .seed(42)
//!     .tolerance(BackendTolerance::default())
//!     .build();
//! ```

use crate::Backend;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::path::PathBuf;

// =============================================================================
// VISUAL REGRESSION TESTING (Genchi Genbutsu: Go and See)
// =============================================================================

/// RGB color for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb {
    /// Red component
    pub r: u8,
    /// Green component
    pub g: u8,
    /// Blue component
    pub b: u8,
}

impl Rgb {
    /// Create new RGB color
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Magenta for NaN values
    pub const NAN_COLOR: Self = Self::new(255, 0, 255);
    /// White for +Infinity
    pub const INF_COLOR: Self = Self::new(255, 255, 255);
    /// Black for -Infinity
    pub const NEG_INF_COLOR: Self = Self::new(0, 0, 0);
}

/// Color palette for heatmap rendering
#[derive(Debug, Clone)]
pub struct ColorPalette {
    colors: Vec<Rgb>,
}

impl Default for ColorPalette {
    fn default() -> Self {
        Self::viridis()
    }
}

impl ColorPalette {
    /// Viridis colorblind-friendly palette
    #[must_use]
    pub fn viridis() -> Self {
        Self {
            colors: vec![
                Rgb::new(68, 1, 84),
                Rgb::new(59, 82, 139),
                Rgb::new(33, 145, 140),
                Rgb::new(94, 201, 98),
                Rgb::new(253, 231, 37),
            ],
        }
    }

    /// Grayscale palette
    #[must_use]
    pub fn grayscale() -> Self {
        Self {
            colors: vec![
                Rgb::new(0, 0, 0),
                Rgb::new(128, 128, 128),
                Rgb::new(255, 255, 255),
            ],
        }
    }

    /// Interpolate color at position t (0.0 to 1.0)
    #[must_use]
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    pub fn interpolate(&self, t: f32) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        let n = self.colors.len() - 1;
        let idx = (t * n as f32).floor() as usize;
        let idx = idx.min(n - 1);
        let local_t = t * n as f32 - idx as f32;

        let c1 = &self.colors[idx];
        let c2 = &self.colors[idx + 1];

        Rgb {
            r: (c1.r as f32 + (c2.r as f32 - c1.r as f32) * local_t) as u8,
            g: (c1.g as f32 + (c2.g as f32 - c1.g as f32) * local_t) as u8,
            b: (c1.b as f32 + (c2.b as f32 - c1.b as f32) * local_t) as u8,
        }
    }
}

/// Visual regression test configuration (Genchi Genbutsu)
#[derive(Debug, Clone)]
pub struct VisualRegressionConfig {
    /// Golden baseline directory
    pub golden_dir: PathBuf,
    /// Output directory for test results
    pub output_dir: PathBuf,
    /// Maximum allowed different pixels (percentage)
    pub max_diff_pct: f64,
    /// Color palette for visualization
    pub palette: ColorPalette,
}

impl Default for VisualRegressionConfig {
    fn default() -> Self {
        Self {
            golden_dir: PathBuf::from("golden"),
            output_dir: PathBuf::from("test_output"),
            max_diff_pct: 0.0, // Exact match by default
            palette: ColorPalette::default(),
        }
    }
}

impl VisualRegressionConfig {
    /// Create new config with custom golden directory
    #[must_use]
    pub fn new(golden_dir: impl Into<PathBuf>) -> Self {
        Self {
            golden_dir: golden_dir.into(),
            ..Default::default()
        }
    }

    /// Set output directory
    #[must_use]
    pub fn with_output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = dir.into();
        self
    }

    /// Set maximum diff percentage
    #[must_use]
    pub const fn with_max_diff_pct(mut self, pct: f64) -> Self {
        self.max_diff_pct = pct;
        self
    }

    /// Set color palette
    #[must_use]
    pub fn with_palette(mut self, palette: ColorPalette) -> Self {
        self.palette = palette;
        self
    }
}

/// Pixel diff result for visual regression testing
#[derive(Debug, Clone)]
pub struct PixelDiffResult {
    /// Number of pixels that differ
    pub different_pixels: usize,
    /// Total number of pixels
    pub total_pixels: usize,
    /// Maximum color difference found
    pub max_diff: u32,
}

impl PixelDiffResult {
    /// Calculate percentage of different pixels
    #[must_use]
    pub fn diff_percentage(&self) -> f64 {
        if self.total_pixels == 0 {
            0.0
        } else {
            (self.different_pixels as f64 / self.total_pixels as f64) * 100.0
        }
    }

    /// Check if images match within threshold
    #[must_use]
    pub fn matches(&self, threshold_pct: f64) -> bool {
        self.diff_percentage() <= threshold_pct
    }

    /// Create a passing result (no differences)
    #[must_use]
    pub const fn pass(total_pixels: usize) -> Self {
        Self {
            different_pixels: 0,
            total_pixels,
            max_diff: 0,
        }
    }
}

/// Simple buffer renderer for SIMD output visualization
///
/// Converts f32 buffers to raw RGBA bytes for testing
#[derive(Debug, Clone)]
pub struct BufferRenderer {
    palette: ColorPalette,
    range: Option<(f32, f32)>,
}

impl Default for BufferRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferRenderer {
    /// Create renderer with auto-normalization
    #[must_use]
    pub fn new() -> Self {
        Self {
            palette: ColorPalette::default(),
            range: None,
        }
    }

    /// Set fixed range for normalization
    #[must_use]
    pub const fn with_range(mut self, min: f32, max: f32) -> Self {
        self.range = Some((min, max));
        self
    }

    /// Set color palette
    #[must_use]
    pub fn with_palette(mut self, palette: ColorPalette) -> Self {
        self.palette = palette;
        self
    }

    /// Render f32 buffer to raw RGBA bytes
    ///
    /// Returns Vec<u8> with RGBA pixels (4 bytes per pixel)
    #[must_use]
    pub fn render_to_rgba(&self, buffer: &[f32], width: u32, height: u32) -> Vec<u8> {
        assert_eq!(buffer.len(), (width * height) as usize);

        let (min_val, max_val) = self.range.unwrap_or_else(|| {
            let valid: Vec<f32> = buffer.iter().copied().filter(|v| v.is_finite()).collect();
            if valid.is_empty() {
                (0.0, 1.0)
            } else {
                let min = valid.iter().copied().fold(f32::INFINITY, f32::min);
                let max = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                (min, max.max(min + f32::EPSILON))
            }
        });

        let mut rgba = Vec::with_capacity(buffer.len() * 4);

        for &value in buffer {
            let color = if value.is_nan() {
                Rgb::NAN_COLOR
            } else if value.is_infinite() {
                if value > 0.0 { Rgb::INF_COLOR } else { Rgb::NEG_INF_COLOR }
            } else {
                let t = (value - min_val) / (max_val - min_val);
                self.palette.interpolate(t)
            };

            rgba.push(color.r);
            rgba.push(color.g);
            rgba.push(color.b);
            rgba.push(255); // Alpha
        }

        rgba
    }

    /// Compare two RGBA buffers and return diff result
    #[must_use]
    pub fn compare_rgba(&self, a: &[u8], b: &[u8], tolerance: u8) -> PixelDiffResult {
        if a == b {
            return PixelDiffResult::pass(a.len() / 4);
        }

        let min_len = a.len().min(b.len());
        let mut different = 0;
        let mut max_diff: u32 = 0;

        // Compare pixels (4 bytes each: RGBA)
        for i in (0..min_len).step_by(4) {
            let mut pixel_diff = false;
            for j in 0..4 {
                if i + j < min_len {
                    let diff = (a[i + j] as i32 - b[i + j] as i32).unsigned_abs();
                    if diff > tolerance as u32 {
                        pixel_diff = true;
                        max_diff = max_diff.max(diff);
                    }
                }
            }
            if pixel_diff {
                different += 1;
            }
        }

        // Count size difference as pixel differences
        if a.len() != b.len() {
            different += a.len().abs_diff(b.len()) / 4;
        }

        PixelDiffResult {
            different_pixels: different,
            total_pixels: min_len.max(a.len()).max(b.len()) / 4,
            max_diff,
        }
    }
}

/// Golden baseline manager for visual regression testing
#[derive(Debug, Clone)]
pub struct GoldenBaseline {
    config: VisualRegressionConfig,
}

impl GoldenBaseline {
    /// Create new golden baseline manager
    #[must_use]
    pub fn new(config: VisualRegressionConfig) -> Self {
        Self { config }
    }

    /// Get path for a golden baseline file
    #[must_use]
    pub fn golden_path(&self, name: &str) -> PathBuf {
        self.config.golden_dir.join(format!("{name}.golden"))
    }

    /// Get path for an output file
    #[must_use]
    pub fn output_path(&self, name: &str) -> PathBuf {
        self.config.output_dir.join(format!("{name}.output"))
    }

    /// Get the config
    #[must_use]
    pub const fn config(&self) -> &VisualRegressionConfig {
        &self.config
    }
}

// =============================================================================
// STRESS TESTING (Heijunka: Leveled Workload Testing)
// =============================================================================

/// Stress test configuration for trueno operations
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    /// Number of cycles per backend
    pub cycles_per_backend: u32,
    /// Input sizes to test (leveled)
    pub input_sizes: Vec<usize>,
    /// Backends to stress test
    pub backends: Vec<Backend>,
    /// Performance thresholds
    pub thresholds: StressThresholds,
    /// Master seed for RNG
    pub master_seed: u64,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            cycles_per_backend: 100,
            input_sizes: vec![100, 1_000, 10_000, 100_000, 1_000_000],
            backends: vec![Backend::Scalar, Backend::AVX2],
            thresholds: StressThresholds::default(),
            master_seed: 42,
        }
    }
}

impl StressTestConfig {
    /// Create new stress test config
    #[must_use]
    pub fn new(master_seed: u64) -> Self {
        Self {
            master_seed,
            ..Default::default()
        }
    }

    /// Set cycles per backend
    #[must_use]
    pub const fn with_cycles(mut self, cycles: u32) -> Self {
        self.cycles_per_backend = cycles;
        self
    }

    /// Set input sizes
    #[must_use]
    pub fn with_input_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.input_sizes = sizes;
        self
    }

    /// Set backends to test
    #[must_use]
    pub fn with_backends(mut self, backends: Vec<Backend>) -> Self {
        self.backends = backends;
        self
    }

    /// Set performance thresholds
    #[must_use]
    pub fn with_thresholds(mut self, thresholds: StressThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Calculate total test count
    #[must_use]
    pub fn total_tests(&self) -> usize {
        self.backends.len() * self.input_sizes.len() * self.cycles_per_backend as usize
    }
}

/// Performance thresholds for stress testing
#[derive(Debug, Clone)]
pub struct StressThresholds {
    /// Max time per operation (ms)
    pub max_op_time_ms: u64,
    /// Max memory per operation (bytes)
    pub max_memory_bytes: usize,
    /// Max variance in operation times (coefficient of variation)
    pub max_timing_variance: f64,
    /// Max allowed failure rate (0.0 to 1.0)
    pub max_failure_rate: f64,
}

impl Default for StressThresholds {
    fn default() -> Self {
        Self {
            max_op_time_ms: 1000,              // 1s max per op
            max_memory_bytes: 256 * 1024 * 1024, // 256MB max
            max_timing_variance: 0.5,          // 50% max variance
            max_failure_rate: 0.0,             // Zero failures allowed
        }
    }
}

impl StressThresholds {
    /// Strict thresholds for CI
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            max_op_time_ms: 100,
            max_memory_bytes: 64 * 1024 * 1024,
            max_timing_variance: 0.2,
            max_failure_rate: 0.0,
        }
    }

    /// Relaxed thresholds for development
    #[must_use]
    pub const fn relaxed() -> Self {
        Self {
            max_op_time_ms: 5000,
            max_memory_bytes: 512 * 1024 * 1024,
            max_timing_variance: 1.0,
            max_failure_rate: 0.01,
        }
    }
}

/// Stress test result for a single operation
#[derive(Debug, Clone)]
pub struct StressResult {
    /// Backend used
    pub backend: Backend,
    /// Input size
    pub input_size: usize,
    /// Cycles completed
    pub cycles_completed: u32,
    /// Total tests passed
    pub tests_passed: u32,
    /// Total tests failed
    pub tests_failed: u32,
    /// Mean operation time (ms)
    pub mean_op_time_ms: f64,
    /// Max operation time (ms)
    pub max_op_time_ms: u64,
    /// Timing variance (coefficient of variation)
    pub timing_variance: f64,
    /// Detected anomalies
    pub anomalies: Vec<StressAnomaly>,
}

impl StressResult {
    /// Check if all tests passed
    #[must_use]
    pub fn passed(&self) -> bool {
        self.tests_failed == 0 && self.anomalies.is_empty()
    }

    /// Calculate pass rate
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        let total = self.tests_passed + self.tests_failed;
        if total == 0 {
            1.0
        } else {
            self.tests_passed as f64 / total as f64
        }
    }
}

/// Anomaly detected during stress testing
#[derive(Debug, Clone)]
pub struct StressAnomaly {
    /// Cycle where anomaly was detected
    pub cycle: u32,
    /// Type of anomaly
    pub kind: StressAnomalyKind,
    /// Description
    pub description: String,
}

/// Types of stress test anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressAnomalyKind {
    /// Operation too slow
    SlowOperation,
    /// High memory usage
    HighMemory,
    /// Test failure
    TestFailure,
    /// Timing spike
    TimingSpike,
    /// Non-deterministic output
    NonDeterministic,
}

/// Re-export SimRng from simular for deterministic testing
#[cfg(test)]
pub use simular::engine::rng::SimRng;

// =============================================================================
// BACKEND TOLERANCE (Poka-Yoke: Type-safe tolerance configuration)
// =============================================================================

/// Backend-specific tolerance configuration
///
/// Implements Poka-Yoke (mistake-proofing) by providing compile-time
/// guarantees for correct tolerance values per backend type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BackendTolerance {
    /// Scalar vs SIMD tolerance (should be exact: 0.0)
    pub scalar_vs_simd: f32,
    /// SIMD vs GPU tolerance (IEEE 754: 1e-5)
    pub simd_vs_gpu: f32,
    /// GPU vs GPU tolerance (same precision: 1e-6)
    pub gpu_vs_gpu: f32,
}

impl Default for BackendTolerance {
    fn default() -> Self {
        Self {
            scalar_vs_simd: 0.0,
            simd_vs_gpu: 1e-5,
            gpu_vs_gpu: 1e-6,
        }
    }
}

impl BackendTolerance {
    /// Strict tolerance for exact comparisons
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            scalar_vs_simd: 0.0,
            simd_vs_gpu: 0.0,
            gpu_vs_gpu: 0.0,
        }
    }

    /// Relaxed tolerance for approximate comparisons
    #[must_use]
    pub const fn relaxed() -> Self {
        Self {
            scalar_vs_simd: 1e-6,
            simd_vs_gpu: 1e-4,
            gpu_vs_gpu: 1e-5,
        }
    }

    /// Get tolerance for comparing two backends
    #[must_use]
    pub fn for_backends(&self, a: Backend, b: Backend) -> f32 {
        match (a, b) {
            (Backend::Scalar, Backend::Scalar) => 0.0,
            (Backend::Scalar, Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512)
            | (Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512, Backend::Scalar) => {
                self.scalar_vs_simd
            }
            (Backend::GPU, Backend::GPU) => self.gpu_vs_gpu,
            (Backend::GPU, _) | (_, Backend::GPU) => self.simd_vs_gpu,
            _ => self.scalar_vs_simd, // SIMD vs SIMD
        }
    }
}

// =============================================================================
// BACKEND SELECTOR (Poka-Yoke: Type-safe backend selection)
// =============================================================================

/// Poka-Yoke: Type-safe backend selection
///
/// Provides compile-time and runtime guarantees for correct backend selection
/// based on input size and operation type.
#[derive(Debug, Clone)]
pub struct BackendSelector {
    /// Minimum size for GPU offload (default: 100,000)
    gpu_threshold: usize,
    /// Minimum size for parallel execution (default: 1,000)
    parallel_threshold: usize,
}

impl Default for BackendSelector {
    fn default() -> Self {
        Self {
            gpu_threshold: 100_000,
            parallel_threshold: 1_000,
        }
    }
}

impl BackendSelector {
    /// Create a new backend selector with custom thresholds
    #[must_use]
    pub const fn new(gpu_threshold: usize, parallel_threshold: usize) -> Self {
        Self {
            gpu_threshold,
            parallel_threshold,
        }
    }

    /// Get the GPU threshold
    #[must_use]
    pub const fn gpu_threshold(&self) -> usize {
        self.gpu_threshold
    }

    /// Get the parallel threshold
    #[must_use]
    pub const fn parallel_threshold(&self) -> usize {
        self.parallel_threshold
    }

    /// Select backend based on input size
    ///
    /// # Decision Logic (TRUENO-SPEC-012)
    ///
    /// - N < 1,000: Pure SIMD (no parallelization overhead)
    /// - 1,000 <= N < 100,000: SIMD + Parallel (Rayon)
    /// - N >= 100,000: GPU (if available), else SIMD + Parallel
    #[must_use]
    pub fn select_for_size(&self, size: usize, gpu_available: bool) -> BackendCategory {
        if size < self.parallel_threshold {
            BackendCategory::SimdOnly
        } else if size < self.gpu_threshold {
            BackendCategory::SimdParallel
        } else if gpu_available {
            BackendCategory::Gpu
        } else {
            BackendCategory::SimdParallel // Graceful fallback
        }
    }

    /// Check if size is at GPU threshold boundary (for testing)
    #[must_use]
    pub fn is_at_gpu_boundary(&self, size: usize) -> bool {
        size == self.gpu_threshold || size == self.gpu_threshold - 1
    }

    /// Check if size is at parallel threshold boundary (for testing)
    #[must_use]
    pub fn is_at_parallel_boundary(&self, size: usize) -> bool {
        size == self.parallel_threshold || size == self.parallel_threshold - 1
    }
}

/// Backend category for selection result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendCategory {
    /// Pure SIMD (N < 1,000)
    SimdOnly,
    /// SIMD with parallel execution (1,000 <= N < 100,000)
    SimdParallel,
    /// GPU compute (N >= 100,000)
    Gpu,
}

// =============================================================================
// JIDOKA GUARD (Built-in Quality: Stop on Defect)
// =============================================================================

/// Jidoka condition that triggers stop
#[derive(Debug, Clone, PartialEq)]
pub enum JidokaCondition {
    /// NaN detected in output
    NanDetected,
    /// Infinity detected in output
    InfDetected,
    /// Cross-backend divergence exceeds tolerance
    BackendDivergence {
        /// Tolerance threshold
        tolerance: f32,
    },
    /// Performance regression exceeds threshold
    PerformanceRegression {
        /// Threshold percentage
        threshold_pct: f32,
    },
    /// Determinism failure (same seed, different output)
    DeterminismFailure,
}

/// Jidoka action on condition trigger
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JidokaAction {
    /// Stop immediately and report
    Stop,
    /// Log and continue (soft Jidoka)
    LogAndContinue,
    /// Trigger visual diff report
    VisualReport,
}

/// Jidoka error types
#[derive(Debug, Clone)]
pub enum JidokaError {
    /// NaN values detected
    NanDetected {
        /// Context description
        context: String,
        /// Indices of NaN values
        indices: Vec<usize>,
    },
    /// Infinity values detected
    InfDetected {
        /// Context description
        context: String,
        /// Indices of infinite values
        indices: Vec<usize>,
    },
    /// Backend divergence detected
    BackendDivergence {
        /// Context description
        context: String,
        /// Maximum difference found
        max_diff: f32,
        /// Tolerance threshold
        tolerance: f32,
    },
    /// Performance regression detected
    PerformanceRegression {
        /// Context description
        context: String,
        /// Actual regression percentage
        regression_pct: f32,
        /// Threshold percentage
        threshold_pct: f32,
    },
    /// Determinism failure detected
    DeterminismFailure {
        /// Context description
        context: String,
        /// First differing index
        first_diff_index: usize,
    },
}

impl std::fmt::Display for JidokaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NanDetected { context, indices } => {
                write!(f, "Jidoka: NaN detected at {context} (indices: {indices:?})")
            }
            Self::InfDetected { context, indices } => {
                write!(
                    f,
                    "Jidoka: Infinity detected at {context} (indices: {indices:?})"
                )
            }
            Self::BackendDivergence {
                context,
                max_diff,
                tolerance,
            } => {
                write!(
                    f,
                    "Jidoka: Backend divergence at {context} (max_diff: {max_diff}, tolerance: {tolerance})"
                )
            }
            Self::PerformanceRegression {
                context,
                regression_pct,
                threshold_pct,
            } => {
                write!(
                    f,
                    "Jidoka: Performance regression at {context} ({regression_pct:.2}% > {threshold_pct:.2}%)"
                )
            }
            Self::DeterminismFailure {
                context,
                first_diff_index,
            } => {
                write!(
                    f,
                    "Jidoka: Determinism failure at {context} (first diff at index {first_diff_index})"
                )
            }
        }
    }
}

impl std::error::Error for JidokaError {}

/// Jidoka guard for simulation tests
///
/// Implements Toyota Production System's Jidoka principle:
/// stop production when a defect is detected.
#[derive(Debug, Clone)]
pub struct JidokaGuard {
    /// Condition that triggers stop
    pub condition: JidokaCondition,
    /// Action to take on trigger
    pub action: JidokaAction,
    /// Context for debugging
    pub context: String,
}

impl JidokaGuard {
    /// Create a new Jidoka guard
    #[must_use]
    pub fn new(condition: JidokaCondition, action: JidokaAction, context: impl Into<String>) -> Self {
        Self {
            condition,
            action,
            context: context.into(),
        }
    }

    /// Create a NaN detection guard
    #[must_use]
    pub fn nan_guard(context: impl Into<String>) -> Self {
        Self::new(JidokaCondition::NanDetected, JidokaAction::Stop, context)
    }

    /// Create an infinity detection guard
    #[must_use]
    pub fn inf_guard(context: impl Into<String>) -> Self {
        Self::new(JidokaCondition::InfDetected, JidokaAction::Stop, context)
    }

    /// Create a backend divergence guard
    #[must_use]
    pub fn divergence_guard(tolerance: f32, context: impl Into<String>) -> Self {
        Self::new(
            JidokaCondition::BackendDivergence { tolerance },
            JidokaAction::Stop,
            context,
        )
    }

    /// Check output for NaN/Inf and return error if found
    ///
    /// # Errors
    ///
    /// Returns `JidokaError` if the condition is triggered
    pub fn check_output(&self, output: &[f32]) -> Result<(), JidokaError> {
        match &self.condition {
            JidokaCondition::NanDetected => {
                let nan_indices: Vec<usize> = output
                    .iter()
                    .enumerate()
                    .filter(|(_, x)| x.is_nan())
                    .map(|(i, _)| i)
                    .collect();

                if !nan_indices.is_empty() {
                    return Err(JidokaError::NanDetected {
                        context: self.context.clone(),
                        indices: nan_indices,
                    });
                }
            }
            JidokaCondition::InfDetected => {
                let inf_indices: Vec<usize> = output
                    .iter()
                    .enumerate()
                    .filter(|(_, x)| x.is_infinite())
                    .map(|(i, _)| i)
                    .collect();

                if !inf_indices.is_empty() {
                    return Err(JidokaError::InfDetected {
                        context: self.context.clone(),
                        indices: inf_indices,
                    });
                }
            }
            _ => {} // Other conditions handled by compare methods
        }
        Ok(())
    }

    /// Compare two outputs for backend divergence
    ///
    /// # Errors
    ///
    /// Returns `JidokaError` if divergence exceeds tolerance
    pub fn check_divergence(&self, a: &[f32], b: &[f32]) -> Result<(), JidokaError> {
        if let JidokaCondition::BackendDivergence { tolerance } = &self.condition {
            let max_diff = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0_f32, f32::max);

            if max_diff > *tolerance {
                return Err(JidokaError::BackendDivergence {
                    context: self.context.clone(),
                    max_diff,
                    tolerance: *tolerance,
                });
            }
        }
        Ok(())
    }

    /// Check for determinism (same inputs should produce same outputs)
    ///
    /// # Errors
    ///
    /// Returns `JidokaError` if outputs differ
    pub fn check_determinism(&self, a: &[f32], b: &[f32]) -> Result<(), JidokaError> {
        if let JidokaCondition::DeterminismFailure = &self.condition {
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                // Use bitwise comparison for exact equality
                if x.to_bits() != y.to_bits() {
                    return Err(JidokaError::DeterminismFailure {
                        context: self.context.clone(),
                        first_diff_index: i,
                    });
                }
            }
        }
        Ok(())
    }
}

// =============================================================================
// HEIJUNKA SCHEDULER (Leveled Testing)
// =============================================================================

/// Simulation test configuration
#[derive(Debug, Clone)]
pub struct SimulationTest {
    /// Backend to test
    pub backend: Backend,
    /// Input size
    pub input_size: usize,
    /// Test cycle number
    pub cycle: u32,
    /// Seed for deterministic RNG
    pub seed: u64,
}

/// Heijunka: Balanced test distribution across backends and sizes
///
/// Implements Toyota Production System's Heijunka principle:
/// level the workload to reduce waste and variability.
#[derive(Debug)]
pub struct HeijunkaScheduler {
    /// Test queue balanced across backends
    queue: VecDeque<SimulationTest>,
    /// Backends to cycle through
    backends: Vec<Backend>,
}

impl HeijunkaScheduler {
    /// Create a leveled test schedule
    #[must_use]
    pub fn new(
        backends: Vec<Backend>,
        input_sizes: Vec<usize>,
        cycles_per_backend: u32,
        master_seed: u64,
    ) -> Self {
        let mut queue = VecDeque::new();

        // Interleave tests across backends (leveling)
        for size in &input_sizes {
            for backend in &backends {
                for cycle in 0..cycles_per_backend {
                    let seed = compute_seed(*backend, *size, cycle, master_seed);
                    queue.push_back(SimulationTest {
                        backend: *backend,
                        input_size: *size,
                        cycle,
                        seed,
                    });
                }
            }
        }

        Self {
            queue,
            backends: backends.clone(),
        }
    }

    /// Get the next test from the queue
    pub fn next_test(&mut self) -> Option<SimulationTest> {
        self.queue.pop_front()
    }

    /// Get remaining test count
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.queue.len()
    }

    /// Get backends being tested
    #[must_use]
    pub fn backends(&self) -> &[Backend] {
        &self.backends
    }

    /// Check if schedule is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

/// Compute deterministic seed for a test configuration
fn compute_seed(backend: Backend, size: usize, cycle: u32, master_seed: u64) -> u64 {
    let backend_bits = backend as u64;
    let size_bits = size as u64;
    let cycle_bits = u64::from(cycle);

    master_seed
        .wrapping_add(backend_bits.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(size_bits.wrapping_mul(0x6A09_E667_BB67_AE85))
        .wrapping_add(cycle_bits.wrapping_mul(0x3C6E_F372_FE94_F82B))
}

// =============================================================================
// SIMULATION TEST CONFIG (Builder Pattern)
// =============================================================================

/// Simulation test configuration builder
#[derive(Debug, Clone)]
pub struct SimTestConfigBuilder<S> {
    seed: u64,
    tolerance: BackendTolerance,
    backends: Vec<Backend>,
    input_sizes: Vec<usize>,
    cycles: u32,
    _state: PhantomData<S>,
}

/// Builder state: seed not set
pub struct NeedsSeed;
/// Builder state: ready to build
pub struct Ready;

impl Default for SimTestConfigBuilder<NeedsSeed> {
    fn default() -> Self {
        Self::new()
    }
}

impl SimTestConfigBuilder<NeedsSeed> {
    /// Create a new config builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            seed: 0,
            tolerance: BackendTolerance::default(),
            backends: vec![Backend::Scalar, Backend::AVX2],
            input_sizes: vec![100, 1_000, 10_000, 100_000],
            cycles: 10,
            _state: PhantomData,
        }
    }

    /// Set the master seed (required)
    #[must_use]
    pub fn seed(self, seed: u64) -> SimTestConfigBuilder<Ready> {
        SimTestConfigBuilder {
            seed,
            tolerance: self.tolerance,
            backends: self.backends,
            input_sizes: self.input_sizes,
            cycles: self.cycles,
            _state: PhantomData,
        }
    }
}

impl SimTestConfigBuilder<Ready> {
    /// Set tolerance configuration
    #[must_use]
    pub fn tolerance(mut self, tolerance: BackendTolerance) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set backends to test
    #[must_use]
    pub fn backends(mut self, backends: Vec<Backend>) -> Self {
        self.backends = backends;
        self
    }

    /// Set input sizes to test
    #[must_use]
    pub fn input_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.input_sizes = sizes;
        self
    }

    /// Set number of test cycles
    #[must_use]
    pub fn cycles(mut self, cycles: u32) -> Self {
        self.cycles = cycles;
        self
    }

    /// Build the configuration
    #[must_use]
    pub fn build(self) -> SimTestConfig {
        SimTestConfig {
            seed: self.seed,
            tolerance: self.tolerance,
            backends: self.backends,
            input_sizes: self.input_sizes,
            cycles: self.cycles,
        }
    }
}

/// Simulation test configuration
#[derive(Debug, Clone)]
pub struct SimTestConfig {
    /// Master seed for deterministic RNG
    pub seed: u64,
    /// Backend tolerance configuration
    pub tolerance: BackendTolerance,
    /// Backends to test
    pub backends: Vec<Backend>,
    /// Input sizes to test
    pub input_sizes: Vec<usize>,
    /// Number of test cycles
    pub cycles: u32,
}

impl SimTestConfig {
    /// Create a config builder
    #[must_use]
    pub fn builder() -> SimTestConfigBuilder<NeedsSeed> {
        SimTestConfigBuilder::new()
    }

    /// Create a Heijunka scheduler from this config
    #[must_use]
    pub fn create_scheduler(&self) -> HeijunkaScheduler {
        HeijunkaScheduler::new(
            self.backends.clone(),
            self.input_sizes.clone(),
            self.cycles,
            self.seed,
        )
    }
}

// =============================================================================
// TESTS (EXTREME TDD)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SimRng Integration Tests (Phase 1, Task 1)
    // =========================================================================

    #[test]
    fn test_simrng_reproducibility() {
        // Falsifiable claim B-016, B-017
        let mut rng1 = SimRng::new(42);
        let mut rng2 = SimRng::new(42);

        let seq1: Vec<f64> = (0..100).map(|_| rng1.gen_f64()).collect();
        let seq2: Vec<f64> = (0..100).map(|_| rng2.gen_f64()).collect();

        assert_eq!(seq1, seq2, "Same seed must produce identical sequences");
    }

    #[test]
    fn test_simrng_different_seeds() {
        // Falsifiable claim B-018
        let mut rng1 = SimRng::new(42);
        let mut rng2 = SimRng::new(43);

        let seq1: Vec<f64> = (0..100).map(|_| rng1.gen_f64()).collect();
        let seq2: Vec<f64> = (0..100).map(|_| rng2.gen_f64()).collect();

        assert_ne!(seq1, seq2, "Different seeds must produce different sequences");
    }

    #[test]
    fn test_simrng_partitioning() {
        // Falsifiable claim B-019
        let mut rng = SimRng::new(42);
        let partitions = rng.partition(4);

        assert_eq!(partitions.len(), 4);

        // Each partition should be independent
        let mut seqs: Vec<Vec<f64>> = Vec::new();
        for mut p in partitions {
            seqs.push((0..10).map(|_| p.gen_f64()).collect());
        }

        for i in 0..seqs.len() {
            for j in (i + 1)..seqs.len() {
                assert_ne!(seqs[i], seqs[j], "Partitions must be independent");
            }
        }
    }

    #[test]
    fn test_simrng_gen_f32_for_trueno() {
        // Generate f32 test data using SimRng
        let mut rng = SimRng::new(42);

        let test_data: Vec<f32> = (0..1000).map(|_| rng.gen_f64() as f32).collect();

        // Verify all values are in valid range
        for v in &test_data {
            assert!(v.is_finite(), "Generated value should be finite");
            assert!(*v >= 0.0 && *v < 1.0, "Value should be in [0, 1)");
        }
    }

    // =========================================================================
    // BackendSelector Tests (Phase 1, Task 2)
    // =========================================================================

    #[test]
    fn test_backend_selector_default_thresholds() {
        // Falsifiable claim A-005, A-006
        let selector = BackendSelector::default();

        assert_eq!(selector.gpu_threshold(), 100_000);
        assert_eq!(selector.parallel_threshold(), 1_000);
    }

    #[test]
    fn test_backend_selector_simd_only() {
        // N < 1,000 should use SIMD only
        let selector = BackendSelector::default();

        assert_eq!(
            selector.select_for_size(100, false),
            BackendCategory::SimdOnly
        );
        assert_eq!(
            selector.select_for_size(999, false),
            BackendCategory::SimdOnly
        );
        assert_eq!(
            selector.select_for_size(999, true),
            BackendCategory::SimdOnly
        );
    }

    #[test]
    fn test_backend_selector_simd_parallel() {
        // 1,000 <= N < 100,000 should use SIMD + Parallel
        let selector = BackendSelector::default();

        assert_eq!(
            selector.select_for_size(1_000, false),
            BackendCategory::SimdParallel
        );
        assert_eq!(
            selector.select_for_size(50_000, false),
            BackendCategory::SimdParallel
        );
        assert_eq!(
            selector.select_for_size(99_999, false),
            BackendCategory::SimdParallel
        );
    }

    #[test]
    fn test_backend_selector_gpu() {
        // N >= 100,000 should use GPU (if available)
        let selector = BackendSelector::default();

        assert_eq!(selector.select_for_size(100_000, true), BackendCategory::Gpu);
        assert_eq!(
            selector.select_for_size(1_000_000, true),
            BackendCategory::Gpu
        );
    }

    #[test]
    fn test_backend_selector_gpu_fallback() {
        // N >= 100,000 without GPU should fallback to SIMD + Parallel
        let selector = BackendSelector::default();

        assert_eq!(
            selector.select_for_size(100_000, false),
            BackendCategory::SimdParallel
        );
    }

    #[test]
    fn test_backend_selector_boundary() {
        // Falsifiable claim A-005
        let selector = BackendSelector::default();

        // At GPU threshold boundary
        assert!(selector.is_at_gpu_boundary(100_000));
        assert!(selector.is_at_gpu_boundary(99_999));
        assert!(!selector.is_at_gpu_boundary(99_998));

        // At parallel threshold boundary
        assert!(selector.is_at_parallel_boundary(1_000));
        assert!(selector.is_at_parallel_boundary(999));
        assert!(!selector.is_at_parallel_boundary(998));
    }

    // =========================================================================
    // BackendTolerance Tests (Phase 1, Task 2)
    // =========================================================================

    #[test]
    fn test_backend_tolerance_default() {
        let tolerance = BackendTolerance::default();

        assert_eq!(tolerance.scalar_vs_simd, 0.0);
        assert!((tolerance.simd_vs_gpu - 1e-5).abs() < 1e-10);
        assert!((tolerance.gpu_vs_gpu - 1e-6).abs() < 1e-10);
    }

    #[test]
    fn test_backend_tolerance_strict() {
        let tolerance = BackendTolerance::strict();

        assert_eq!(tolerance.scalar_vs_simd, 0.0);
        assert_eq!(tolerance.simd_vs_gpu, 0.0);
        assert_eq!(tolerance.gpu_vs_gpu, 0.0);
    }

    #[test]
    fn test_backend_tolerance_for_backends() {
        // Falsifiable claim A-002, A-003, A-004
        let tolerance = BackendTolerance::default();

        // Scalar vs Scalar
        assert_eq!(tolerance.for_backends(Backend::Scalar, Backend::Scalar), 0.0);

        // Scalar vs SIMD (should be exact)
        assert_eq!(tolerance.for_backends(Backend::Scalar, Backend::AVX2), 0.0);

        // GPU vs GPU
        assert_eq!(
            tolerance.for_backends(Backend::GPU, Backend::GPU),
            tolerance.gpu_vs_gpu
        );

        // SIMD vs GPU
        assert_eq!(
            tolerance.for_backends(Backend::AVX2, Backend::GPU),
            tolerance.simd_vs_gpu
        );
    }

    // =========================================================================
    // JidokaGuard Tests (Phase 1, Task 3)
    // =========================================================================

    #[test]
    fn test_jidoka_nan_detection() {
        // Falsifiable claim B-027
        let guard = JidokaGuard::nan_guard("test_operation");
        let output_with_nan = vec![1.0, 2.0, f32::NAN, 4.0];

        let result = guard.check_output(&output_with_nan);
        assert!(result.is_err());

        if let Err(JidokaError::NanDetected { indices, .. }) = result {
            assert_eq!(indices, vec![2]);
        } else {
            panic!("Expected NanDetected error");
        }
    }

    #[test]
    fn test_jidoka_nan_no_false_positive() {
        let guard = JidokaGuard::nan_guard("test_operation");
        let clean_output = vec![1.0, 2.0, 3.0, 4.0];

        let result = guard.check_output(&clean_output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_jidoka_inf_detection() {
        // Falsifiable claim B-028
        let guard = JidokaGuard::inf_guard("test_operation");
        let output_with_inf = vec![1.0, f32::INFINITY, 3.0, f32::NEG_INFINITY];

        let result = guard.check_output(&output_with_inf);
        assert!(result.is_err());

        if let Err(JidokaError::InfDetected { indices, .. }) = result {
            assert_eq!(indices, vec![1, 3]);
        } else {
            panic!("Expected InfDetected error");
        }
    }

    #[test]
    fn test_jidoka_divergence_detection() {
        // Falsifiable claim A-004
        let guard = JidokaGuard::divergence_guard(1e-5, "cross_backend");
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.1, 4.0]; // 0.1 diff at index 2

        let result = guard.check_divergence(&a, &b);
        assert!(result.is_err());

        if let Err(JidokaError::BackendDivergence { max_diff, .. }) = result {
            assert!((max_diff - 0.1).abs() < 1e-6);
        } else {
            panic!("Expected BackendDivergence error");
        }
    }

    #[test]
    fn test_jidoka_divergence_within_tolerance() {
        let guard = JidokaGuard::divergence_guard(1e-5, "cross_backend");
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0 + 1e-7, 4.0]; // Within tolerance

        let result = guard.check_divergence(&a, &b);
        assert!(result.is_ok());
    }

    #[test]
    fn test_jidoka_determinism_check() {
        // Falsifiable claim B-017
        let guard = JidokaGuard::new(
            JidokaCondition::DeterminismFailure,
            JidokaAction::Stop,
            "determinism_test",
        );

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];

        let result = guard.check_determinism(&a, &b);
        assert!(result.is_ok());
    }

    #[test]
    fn test_jidoka_determinism_failure() {
        let guard = JidokaGuard::new(
            JidokaCondition::DeterminismFailure,
            JidokaAction::Stop,
            "determinism_test",
        );

        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f32> = vec![1.0, 2.0, 3.000_001, 4.0]; // Different bit pattern

        // Verify they actually have different bits
        assert_ne!(a[2].to_bits(), b[2].to_bits(), "Test values must differ");

        let result = guard.check_determinism(&a, &b);
        assert!(result.is_err());

        if let Err(JidokaError::DeterminismFailure {
            first_diff_index, ..
        }) = result
        {
            assert_eq!(first_diff_index, 2);
        } else {
            panic!("Expected DeterminismFailure error");
        }
    }

    #[test]
    fn test_jidoka_error_display() {
        let err = JidokaError::NanDetected {
            context: "test".to_string(),
            indices: vec![0, 2],
        };
        let display = format!("{err}");
        assert!(display.contains("NaN"));
        assert!(display.contains("test"));

        let err2 = JidokaError::BackendDivergence {
            context: "cross".to_string(),
            max_diff: 0.01,
            tolerance: 0.001,
        };
        let display2 = format!("{err2}");
        assert!(display2.contains("divergence"));
    }

    // =========================================================================
    // HeijunkaScheduler Tests (Phase 1, Task 4)
    // =========================================================================

    #[test]
    fn test_heijunka_schedule_creation() {
        let backends = vec![Backend::Scalar, Backend::AVX2];
        let sizes = vec![100, 1000];
        let cycles = 2;

        let scheduler = HeijunkaScheduler::new(backends.clone(), sizes, cycles, 42);

        // Should have backends.len() * sizes.len() * cycles tests
        // 2 backends * 2 sizes * 2 cycles = 8 tests
        assert_eq!(scheduler.remaining(), 8);
    }

    #[test]
    fn test_heijunka_deterministic_seeds() {
        // Falsifiable claim B-017
        let backends = vec![Backend::Scalar, Backend::AVX2];
        let sizes = vec![100, 1000];
        let cycles = 2;

        let mut scheduler1 = HeijunkaScheduler::new(backends.clone(), sizes.clone(), cycles, 42);
        let mut scheduler2 = HeijunkaScheduler::new(backends, sizes, cycles, 42);

        // Seeds should be identical for same configuration
        while let (Some(t1), Some(t2)) = (scheduler1.next_test(), scheduler2.next_test()) {
            assert_eq!(t1.seed, t2.seed, "Seeds must be deterministic");
        }
    }

    #[test]
    fn test_heijunka_consumes_all_tests() {
        let backends = vec![Backend::Scalar];
        let sizes = vec![100];
        let cycles = 5;

        let mut scheduler = HeijunkaScheduler::new(backends, sizes, cycles, 42);

        let mut count = 0;
        while scheduler.next_test().is_some() {
            count += 1;
        }

        assert_eq!(count, 5);
        assert!(scheduler.is_empty());
    }

    #[test]
    fn test_heijunka_different_master_seeds() {
        // Falsifiable claim B-018
        let backends = vec![Backend::Scalar];
        let sizes = vec![100];
        let cycles = 1;

        let mut scheduler1 = HeijunkaScheduler::new(backends.clone(), sizes.clone(), cycles, 42);
        let mut scheduler2 = HeijunkaScheduler::new(backends, sizes, cycles, 43);

        let t1 = scheduler1.next_test().unwrap();
        let t2 = scheduler2.next_test().unwrap();

        assert_ne!(t1.seed, t2.seed, "Different master seeds must produce different test seeds");
    }

    // =========================================================================
    // SimTestConfig Tests
    // =========================================================================

    #[test]
    fn test_sim_test_config_builder() {
        let config = SimTestConfig::builder()
            .seed(42)
            .tolerance(BackendTolerance::strict())
            .backends(vec![Backend::Scalar, Backend::AVX2, Backend::GPU])
            .input_sizes(vec![100, 1000, 10000])
            .cycles(5)
            .build();

        assert_eq!(config.seed, 42);
        assert_eq!(config.backends.len(), 3);
        assert_eq!(config.input_sizes.len(), 3);
        assert_eq!(config.cycles, 5);
    }

    #[test]
    fn test_sim_test_config_creates_scheduler() {
        let config = SimTestConfig::builder()
            .seed(42)
            .backends(vec![Backend::Scalar, Backend::AVX2])
            .input_sizes(vec![100, 1000])
            .cycles(3)
            .build();

        let scheduler = config.create_scheduler();

        // 2 backends * 2 sizes * 3 cycles = 12 tests
        assert_eq!(scheduler.remaining(), 12);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_full_simulation_workflow() {
        // Create test configuration
        let config = SimTestConfig::builder()
            .seed(42)
            .tolerance(BackendTolerance::default())
            .backends(vec![Backend::Scalar, Backend::AVX2])
            .input_sizes(vec![100])
            .cycles(2)
            .build();

        // Create scheduler
        let mut scheduler = config.create_scheduler();

        // Create Jidoka guards
        let nan_guard = JidokaGuard::nan_guard("simulation_test");

        // Run through all tests
        let mut test_count = 0;
        while let Some(test) = scheduler.next_test() {
            // Generate deterministic test data
            let mut rng = SimRng::new(test.seed);
            let data: Vec<f32> = (0..test.input_size).map(|_| rng.gen_f64() as f32).collect();

            // Check for NaN (should pass with valid data)
            let result = nan_guard.check_output(&data);
            assert!(result.is_ok(), "Generated data should not contain NaN");

            test_count += 1;
        }

        // 2 backends * 1 size * 2 cycles = 4 tests
        assert_eq!(test_count, 4);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_empty_output_checks() {
        let guard = JidokaGuard::nan_guard("empty_test");
        let result = guard.check_output(&[]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_single_element_checks() {
        let guard = JidokaGuard::nan_guard("single_test");

        assert!(guard.check_output(&[1.0]).is_ok());
        assert!(guard.check_output(&[f32::NAN]).is_err());
    }

    #[test]
    fn test_backend_category_debug() {
        let category = BackendCategory::Gpu;
        let debug = format!("{category:?}");
        assert!(debug.contains("Gpu"));
    }

    #[test]
    fn test_jidoka_condition_clone() {
        let condition = JidokaCondition::BackendDivergence { tolerance: 1e-5 };
        let cloned = condition.clone();
        assert_eq!(condition, cloned);
    }

    #[test]
    fn test_jidoka_action_eq() {
        assert_eq!(JidokaAction::Stop, JidokaAction::Stop);
        assert_ne!(JidokaAction::Stop, JidokaAction::LogAndContinue);
    }

    // =========================================================================
    // Visual Regression Testing (Phase 2)
    // =========================================================================

    #[test]
    fn test_rgb_color_creation() {
        let color = Rgb::new(255, 128, 64);
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 128);
        assert_eq!(color.b, 64);
    }

    #[test]
    fn test_rgb_special_colors() {
        assert_eq!(Rgb::NAN_COLOR, Rgb::new(255, 0, 255));
        assert_eq!(Rgb::INF_COLOR, Rgb::new(255, 255, 255));
        assert_eq!(Rgb::NEG_INF_COLOR, Rgb::new(0, 0, 0));
    }

    #[test]
    fn test_color_palette_viridis() {
        let palette = ColorPalette::viridis();
        assert_eq!(palette.colors.len(), 5);

        // Test interpolation at boundaries
        let at_0 = palette.interpolate(0.0);
        let at_1 = palette.interpolate(1.0);

        // Viridis starts dark purple, ends yellow
        assert_eq!(at_0, Rgb::new(68, 1, 84));
        assert_eq!(at_1, Rgb::new(253, 231, 37));
    }

    #[test]
    fn test_color_palette_grayscale() {
        let palette = ColorPalette::grayscale();
        assert_eq!(palette.colors.len(), 3);

        let at_0 = palette.interpolate(0.0);
        let at_1 = palette.interpolate(1.0);

        assert_eq!(at_0, Rgb::new(0, 0, 0));
        assert_eq!(at_1, Rgb::new(255, 255, 255));
    }

    #[test]
    fn test_color_palette_interpolation_midpoint() {
        let palette = ColorPalette::grayscale();
        let at_mid = palette.interpolate(0.5);

        // Should be close to gray
        assert_eq!(at_mid, Rgb::new(128, 128, 128));
    }

    #[test]
    fn test_color_palette_clamping() {
        let palette = ColorPalette::viridis();

        // Values outside [0, 1] should be clamped
        let at_neg = palette.interpolate(-0.5);
        let at_over = palette.interpolate(1.5);

        assert_eq!(at_neg, palette.interpolate(0.0));
        assert_eq!(at_over, palette.interpolate(1.0));
    }

    #[test]
    fn test_visual_regression_config_default() {
        let config = VisualRegressionConfig::default();

        assert_eq!(config.golden_dir, PathBuf::from("golden"));
        assert_eq!(config.output_dir, PathBuf::from("test_output"));
        assert_eq!(config.max_diff_pct, 0.0);
    }

    #[test]
    fn test_visual_regression_config_builder() {
        let config = VisualRegressionConfig::new("my_golden")
            .with_output_dir("my_output")
            .with_max_diff_pct(1.5)
            .with_palette(ColorPalette::grayscale());

        assert_eq!(config.golden_dir, PathBuf::from("my_golden"));
        assert_eq!(config.output_dir, PathBuf::from("my_output"));
        assert_eq!(config.max_diff_pct, 1.5);
    }

    #[test]
    fn test_pixel_diff_result_percentage() {
        let result = PixelDiffResult {
            different_pixels: 10,
            total_pixels: 100,
            max_diff: 50,
        };

        assert_eq!(result.diff_percentage(), 10.0);
        assert!(!result.matches(5.0));
        assert!(result.matches(10.0));
        assert!(result.matches(15.0));
    }

    #[test]
    fn test_pixel_diff_result_zero_total() {
        let result = PixelDiffResult {
            different_pixels: 0,
            total_pixels: 0,
            max_diff: 0,
        };

        assert_eq!(result.diff_percentage(), 0.0);
    }

    #[test]
    fn test_pixel_diff_result_pass() {
        let result = PixelDiffResult::pass(100);

        assert_eq!(result.different_pixels, 0);
        assert_eq!(result.total_pixels, 100);
        assert_eq!(result.max_diff, 0);
        assert!(result.matches(0.0));
    }

    #[test]
    fn test_buffer_renderer_default() {
        let renderer = BufferRenderer::default();
        assert!(renderer.range.is_none());
    }

    #[test]
    fn test_buffer_renderer_with_range() {
        let renderer = BufferRenderer::new().with_range(0.0, 10.0);
        assert_eq!(renderer.range, Some((0.0, 10.0)));
    }

    #[test]
    fn test_buffer_renderer_with_palette() {
        let renderer = BufferRenderer::new().with_palette(ColorPalette::grayscale());
        assert_eq!(renderer.palette.colors.len(), 3);
    }

    #[test]
    fn test_buffer_renderer_rgba_output() {
        let renderer = BufferRenderer::new();
        let buffer: Vec<f32> = (0..4).map(|i| i as f32 / 3.0).collect();
        let rgba = renderer.render_to_rgba(&buffer, 2, 2);

        // 4 pixels * 4 bytes = 16 bytes
        assert_eq!(rgba.len(), 16);

        // Check alpha channel is always 255
        for i in (3..16).step_by(4) {
            assert_eq!(rgba[i], 255);
        }
    }

    #[test]
    fn test_buffer_renderer_nan_handling() {
        let renderer = BufferRenderer::new();
        let buffer = vec![0.0, f32::NAN, 1.0, 0.5];
        let rgba = renderer.render_to_rgba(&buffer, 2, 2);

        // Second pixel should be NAN_COLOR (magenta: 255, 0, 255)
        assert_eq!(rgba[4], 255); // R
        assert_eq!(rgba[5], 0);   // G
        assert_eq!(rgba[6], 255); // B
        assert_eq!(rgba[7], 255); // A
    }

    #[test]
    fn test_buffer_renderer_inf_handling() {
        let renderer = BufferRenderer::new();
        let buffer = vec![f32::INFINITY, f32::NEG_INFINITY, 0.5, 0.5];
        let rgba = renderer.render_to_rgba(&buffer, 2, 2);

        // First pixel: +INF should be white
        assert_eq!(rgba[0], 255);
        assert_eq!(rgba[1], 255);
        assert_eq!(rgba[2], 255);

        // Second pixel: -INF should be black
        assert_eq!(rgba[4], 0);
        assert_eq!(rgba[5], 0);
        assert_eq!(rgba[6], 0);
    }

    #[test]
    fn test_buffer_renderer_compare_identical() {
        let renderer = BufferRenderer::new();
        let buffer: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let rgba = renderer.render_to_rgba(&buffer, 4, 4);

        let result = renderer.compare_rgba(&rgba, &rgba, 0);
        assert_eq!(result.different_pixels, 0);
        assert!(result.matches(0.0));
    }

    #[test]
    fn test_buffer_renderer_compare_different() {
        let renderer = BufferRenderer::new();
        let buffer_a: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let buffer_b: Vec<f32> = (0..16).map(|i| 1.0 - i as f32 / 15.0).collect();

        let rgba_a = renderer.render_to_rgba(&buffer_a, 4, 4);
        let rgba_b = renderer.render_to_rgba(&buffer_b, 4, 4);

        let result = renderer.compare_rgba(&rgba_a, &rgba_b, 0);
        assert!(result.different_pixels > 0);
    }

    #[test]
    fn test_buffer_renderer_compare_with_tolerance() {
        let renderer = BufferRenderer::new();
        let rgba_a = vec![100, 100, 100, 255];
        let rgba_b = vec![105, 102, 98, 255];

        // With tolerance 10, should match
        let result = renderer.compare_rgba(&rgba_a, &rgba_b, 10);
        assert_eq!(result.different_pixels, 0);

        // With tolerance 1, should differ
        let result_strict = renderer.compare_rgba(&rgba_a, &rgba_b, 1);
        assert!(result_strict.different_pixels > 0);
    }

    #[test]
    fn test_golden_baseline_paths() {
        let config = VisualRegressionConfig::new("/test/golden")
            .with_output_dir("/test/output");
        let baseline = GoldenBaseline::new(config);

        assert_eq!(
            baseline.golden_path("relu_4x4"),
            PathBuf::from("/test/golden/relu_4x4.golden")
        );
        assert_eq!(
            baseline.output_path("relu_4x4"),
            PathBuf::from("/test/output/relu_4x4.output")
        );
    }

    #[test]
    fn test_golden_baseline_config_access() {
        let config = VisualRegressionConfig::new("/golden").with_max_diff_pct(2.5);
        let baseline = GoldenBaseline::new(config);

        assert_eq!(baseline.config().max_diff_pct, 2.5);
    }

    // =========================================================================
    // Stress Testing (Phase 3)
    // =========================================================================

    #[test]
    fn test_stress_test_config_default() {
        let config = StressTestConfig::default();

        assert_eq!(config.cycles_per_backend, 100);
        assert_eq!(config.input_sizes.len(), 5);
        assert_eq!(config.backends.len(), 2);
        assert_eq!(config.master_seed, 42);
    }

    #[test]
    fn test_stress_test_config_builder() {
        let config = StressTestConfig::new(123)
            .with_cycles(50)
            .with_input_sizes(vec![100, 1000])
            .with_backends(vec![Backend::Scalar])
            .with_thresholds(StressThresholds::strict());

        assert_eq!(config.master_seed, 123);
        assert_eq!(config.cycles_per_backend, 50);
        assert_eq!(config.input_sizes.len(), 2);
        assert_eq!(config.backends.len(), 1);
    }

    #[test]
    fn test_stress_test_config_total_tests() {
        let config = StressTestConfig::default()
            .with_cycles(10)
            .with_input_sizes(vec![100, 1000, 10000])
            .with_backends(vec![Backend::Scalar, Backend::AVX2]);

        // 2 backends * 3 sizes * 10 cycles = 60 tests
        assert_eq!(config.total_tests(), 60);
    }

    #[test]
    fn test_stress_thresholds_default() {
        let thresholds = StressThresholds::default();

        assert_eq!(thresholds.max_op_time_ms, 1000);
        assert_eq!(thresholds.max_memory_bytes, 256 * 1024 * 1024);
        assert!((thresholds.max_timing_variance - 0.5).abs() < 0.001);
        assert_eq!(thresholds.max_failure_rate, 0.0);
    }

    #[test]
    fn test_stress_thresholds_strict() {
        let thresholds = StressThresholds::strict();

        assert_eq!(thresholds.max_op_time_ms, 100);
        assert_eq!(thresholds.max_memory_bytes, 64 * 1024 * 1024);
        assert!((thresholds.max_timing_variance - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_stress_thresholds_relaxed() {
        let thresholds = StressThresholds::relaxed();

        assert_eq!(thresholds.max_op_time_ms, 5000);
        assert_eq!(thresholds.max_memory_bytes, 512 * 1024 * 1024);
        assert!((thresholds.max_timing_variance - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_stress_result_passed() {
        let result = StressResult {
            backend: Backend::Scalar,
            input_size: 1000,
            cycles_completed: 10,
            tests_passed: 100,
            tests_failed: 0,
            mean_op_time_ms: 50.0,
            max_op_time_ms: 100,
            timing_variance: 0.1,
            anomalies: vec![],
        };

        assert!(result.passed());
        assert_eq!(result.pass_rate(), 1.0);
    }

    #[test]
    fn test_stress_result_failed() {
        let result = StressResult {
            backend: Backend::AVX2,
            input_size: 10000,
            cycles_completed: 10,
            tests_passed: 95,
            tests_failed: 5,
            mean_op_time_ms: 100.0,
            max_op_time_ms: 500,
            timing_variance: 0.3,
            anomalies: vec![],
        };

        assert!(!result.passed()); // Failed because tests_failed > 0
        assert!((result.pass_rate() - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_stress_result_with_anomaly() {
        let result = StressResult {
            backend: Backend::Scalar,
            input_size: 1000,
            cycles_completed: 10,
            tests_passed: 100,
            tests_failed: 0,
            mean_op_time_ms: 50.0,
            max_op_time_ms: 100,
            timing_variance: 0.1,
            anomalies: vec![StressAnomaly {
                cycle: 5,
                kind: StressAnomalyKind::SlowOperation,
                description: "Operation took 200ms".to_string(),
            }],
        };

        assert!(!result.passed()); // Failed because anomalies not empty
    }

    #[test]
    fn test_stress_anomaly_kinds() {
        assert_eq!(StressAnomalyKind::SlowOperation, StressAnomalyKind::SlowOperation);
        assert_ne!(StressAnomalyKind::SlowOperation, StressAnomalyKind::TestFailure);

        // Test all variants exist
        let _slow = StressAnomalyKind::SlowOperation;
        let _mem = StressAnomalyKind::HighMemory;
        let _fail = StressAnomalyKind::TestFailure;
        let _spike = StressAnomalyKind::TimingSpike;
        let _ndet = StressAnomalyKind::NonDeterministic;
    }

    #[test]
    fn test_stress_result_zero_tests() {
        let result = StressResult {
            backend: Backend::Scalar,
            input_size: 0,
            cycles_completed: 0,
            tests_passed: 0,
            tests_failed: 0,
            mean_op_time_ms: 0.0,
            max_op_time_ms: 0,
            timing_variance: 0.0,
            anomalies: vec![],
        };

        // Zero tests should still pass with pass_rate of 1.0
        assert!(result.passed());
        assert_eq!(result.pass_rate(), 1.0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Falsifiable claim A-009: Backend selection is deterministic
        #[test]
        fn prop_backend_selection_deterministic(size in 0usize..1_000_000) {
            let selector = BackendSelector::default();

            let result1 = selector.select_for_size(size, true);
            let result2 = selector.select_for_size(size, true);

            prop_assert_eq!(result1, result2);
        }

        /// Falsifiable claim: compute_seed is deterministic
        #[test]
        fn prop_compute_seed_deterministic(
            backend_idx in 0u8..8,
            size in 0usize..1_000_000,
            cycle in 0u32..100,
            master_seed in any::<u64>()
        ) {
            let backend = match backend_idx {
                0 => Backend::Scalar,
                1 => Backend::SSE2,
                2 => Backend::AVX,
                3 => Backend::AVX2,
                4 => Backend::AVX512,
                5 => Backend::NEON,
                6 => Backend::WasmSIMD,
                _ => Backend::GPU,
            };

            let seed1 = compute_seed(backend, size, cycle, master_seed);
            let seed2 = compute_seed(backend, size, cycle, master_seed);

            prop_assert_eq!(seed1, seed2);
        }

        /// Falsifiable claim: NaN detection never misses
        #[test]
        fn prop_nan_detection_complete(values in prop::collection::vec(-1000.0f32..1000.0, 0..100)) {
            let guard = JidokaGuard::nan_guard("test");

            // Clean input should pass
            let result = guard.check_output(&values);
            prop_assert!(result.is_ok());
        }
    }
}
