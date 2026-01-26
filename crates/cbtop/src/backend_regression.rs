//! Cross-Backend Regression Detector (PMAT-031)
//!
//! Detect performance regressions when switching between compute backends.
//!
//! # Features
//!
//! - Compare efficiency across backends (Scalar, SSE2, AVX2, CUDA, Metal)
//! - Detect size thresholds where performance cliffs occur
//! - Recommend optimal backend for given workload size
//! - Measure GPU transfer overhead vs compute benefit
//!
//! # Falsification Criteria (F1231-F1240)
//!
//! See `tests/backend_regression_f1231.rs` for falsification tests.

/// Backend identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Scalar (no SIMD)
    Scalar,
    /// SSE2 (128-bit)
    Sse2,
    /// AVX2 (256-bit)
    Avx2,
    /// AVX-512 (512-bit)
    Avx512,
    /// NEON (ARM)
    Neon,
    /// CUDA (NVIDIA GPU)
    Cuda,
    /// Metal (Apple GPU)
    Metal,
    /// Vulkan (Cross-platform GPU)
    Vulkan,
    /// WebGPU
    WebGpu,
}

impl Backend {
    /// Get backend name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Scalar => "Scalar",
            Self::Sse2 => "SSE2",
            Self::Avx2 => "AVX2",
            Self::Avx512 => "AVX-512",
            Self::Neon => "NEON",
            Self::Cuda => "CUDA",
            Self::Metal => "Metal",
            Self::Vulkan => "Vulkan",
            Self::WebGpu => "WebGPU",
        }
    }

    /// Is this a GPU backend?
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Cuda | Self::Metal | Self::Vulkan | Self::WebGpu)
    }

    /// Is this a SIMD backend?
    pub fn is_simd(&self) -> bool {
        matches!(self, Self::Sse2 | Self::Avx2 | Self::Avx512 | Self::Neon)
    }

    /// Get expected speedup over scalar (theoretical)
    pub fn theoretical_speedup(&self) -> f64 {
        match self {
            Self::Scalar => 1.0,
            Self::Sse2 => 4.0,    // 128-bit / 32-bit = 4
            Self::Avx2 => 8.0,    // 256-bit / 32-bit = 8
            Self::Avx512 => 16.0, // 512-bit / 32-bit = 16
            Self::Neon => 4.0,    // 128-bit / 32-bit = 4
            Self::Cuda => 100.0,  // Variable, placeholder
            Self::Metal => 50.0,  // Variable, placeholder
            Self::Vulkan => 50.0, // Variable, placeholder
            Self::WebGpu => 30.0, // Variable, placeholder
        }
    }
}

/// Workload type for benchmark
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadType {
    /// Matrix multiplication
    Gemm,
    /// 2D convolution
    Conv2d,
    /// Element-wise operations
    Elementwise,
    /// Reduction (sum, mean)
    Reduction,
    /// Attention mechanism
    Attention,
    /// Memory bandwidth test
    Bandwidth,
}

impl WorkloadType {
    /// Get workload name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gemm => "GEMM",
            Self::Conv2d => "Conv2D",
            Self::Elementwise => "Elementwise",
            Self::Reduction => "Reduction",
            Self::Attention => "Attention",
            Self::Bandwidth => "Bandwidth",
        }
    }
}

/// Performance measurement for a single backend/size combination
#[derive(Debug, Clone)]
pub struct BackendMeasurement {
    /// Backend used
    pub backend: Backend,
    /// Workload type
    pub workload: WorkloadType,
    /// Problem size (elements)
    pub size: usize,
    /// Latency in microseconds
    pub latency_us: f64,
    /// Throughput (ops/sec or elements/sec)
    pub throughput: f64,
    /// Efficiency (% of theoretical peak)
    pub efficiency_percent: f64,
    /// GPU transfer time (if applicable)
    pub transfer_time_us: Option<f64>,
    /// Compute time (excluding transfer)
    pub compute_time_us: Option<f64>,
}

impl BackendMeasurement {
    /// Create new measurement
    pub fn new(
        backend: Backend,
        workload: WorkloadType,
        size: usize,
        latency_us: f64,
        throughput: f64,
    ) -> Self {
        Self {
            backend,
            workload,
            size,
            latency_us,
            throughput,
            efficiency_percent: 0.0,
            transfer_time_us: None,
            compute_time_us: None,
        }
    }

    /// Set efficiency
    pub fn with_efficiency(mut self, efficiency: f64) -> Self {
        self.efficiency_percent = efficiency;
        self
    }

    /// Set GPU timing breakdown
    pub fn with_gpu_timing(mut self, transfer_us: f64, compute_us: f64) -> Self {
        self.transfer_time_us = Some(transfer_us);
        self.compute_time_us = Some(compute_us);
        self
    }

    /// Get transfer overhead ratio (transfer / total)
    pub fn transfer_overhead(&self) -> Option<f64> {
        match (self.transfer_time_us, self.compute_time_us) {
            (Some(t), Some(c)) if t + c > 0.0 => Some(t / (t + c)),
            _ => None,
        }
    }
}

/// Comparison result between two backends
#[derive(Debug, Clone)]
pub struct BackendComparison {
    /// Baseline backend
    pub baseline: Backend,
    /// Comparison backend
    pub comparison: Backend,
    /// Workload type
    pub workload: WorkloadType,
    /// Problem size
    pub size: usize,
    /// Efficiency ratio (comparison / baseline)
    pub efficiency_ratio: f64,
    /// Speedup (baseline_latency / comparison_latency)
    pub speedup: f64,
    /// Is this a regression? (efficiency_ratio < 1.0 - threshold)
    pub is_regression: bool,
    /// Regression threshold used
    pub threshold: f64,
}

impl BackendComparison {
    /// Get summary message
    pub fn summary(&self) -> String {
        if self.is_regression {
            format!(
                "REGRESSION: {} -> {} on {} size={}: {:.1}% slower",
                self.baseline.name(),
                self.comparison.name(),
                self.workload.name(),
                self.size,
                (1.0 - self.speedup) * 100.0
            )
        } else {
            format!(
                "OK: {} -> {} on {} size={}: {:.1}x speedup",
                self.baseline.name(),
                self.comparison.name(),
                self.workload.name(),
                self.size,
                self.speedup
            )
        }
    }
}

/// Size cliff detection result
#[derive(Debug, Clone)]
pub struct SizeCliff {
    /// Backend where cliff occurs
    pub backend: Backend,
    /// Workload type
    pub workload: WorkloadType,
    /// Size before cliff
    pub size_before: usize,
    /// Size after cliff
    pub size_after: usize,
    /// Efficiency before cliff
    pub efficiency_before: f64,
    /// Efficiency after cliff
    pub efficiency_after: f64,
    /// Drop percentage
    pub drop_percent: f64,
}

impl SizeCliff {
    /// Get summary message
    pub fn summary(&self) -> String {
        format!(
            "CLIFF: {} {} at {}→{}: {:.1}% efficiency drop",
            self.backend.name(),
            self.workload.name(),
            self.size_before,
            self.size_after,
            self.drop_percent
        )
    }
}

/// Backend recommendation
#[derive(Debug, Clone)]
pub struct BackendRecommendation {
    /// Recommended backend
    pub backend: Backend,
    /// Workload type
    pub workload: WorkloadType,
    /// Problem size
    pub size: usize,
    /// Expected efficiency
    pub expected_efficiency: f64,
    /// Confidence (0.0 - 1.0)
    pub confidence: f64,
    /// Reason for recommendation
    pub reason: String,
}

/// Backend regression detector
#[derive(Debug, Clone)]
pub struct BackendRegressionDetector {
    /// Measurements collected
    measurements: Vec<BackendMeasurement>,
    /// Regression threshold (default 10%)
    threshold_percent: f64,
    /// Cliff detection threshold (default 10%)
    cliff_threshold_percent: f64,
    /// Available backends
    available_backends: Vec<Backend>,
}

impl Default for BackendRegressionDetector {
    fn default() -> Self {
        Self {
            measurements: Vec::new(),
            threshold_percent: 10.0,
            cliff_threshold_percent: 10.0,
            available_backends: vec![Backend::Scalar, Backend::Sse2, Backend::Avx2],
        }
    }
}

impl BackendRegressionDetector {
    /// Create new detector
    pub fn new() -> Self {
        Self::default()
    }

    /// Set regression threshold
    pub fn with_threshold(mut self, percent: f64) -> Self {
        self.threshold_percent = percent;
        self
    }

    /// Set cliff detection threshold
    pub fn with_cliff_threshold(mut self, percent: f64) -> Self {
        self.cliff_threshold_percent = percent;
        self
    }

    /// Set available backends
    pub fn with_backends(mut self, backends: Vec<Backend>) -> Self {
        self.available_backends = backends;
        self
    }

    /// Add a measurement
    pub fn add_measurement(&mut self, measurement: BackendMeasurement) {
        self.measurements.push(measurement);
    }

    /// Add measurement from values
    pub fn add(
        &mut self,
        backend: Backend,
        workload: WorkloadType,
        size: usize,
        latency_us: f64,
        throughput: f64,
        efficiency: f64,
    ) {
        self.add_measurement(
            BackendMeasurement::new(backend, workload, size, latency_us, throughput)
                .with_efficiency(efficiency),
        );
    }

    /// Get measurement count
    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }

    /// Compare two backends for a specific workload/size
    pub fn compare_backends(
        &self,
        baseline: Backend,
        comparison: Backend,
        workload: WorkloadType,
        size: usize,
    ) -> Option<BackendComparison> {
        let baseline_m = self.find_measurement(baseline, workload, size)?;
        let comparison_m = self.find_measurement(comparison, workload, size)?;

        let efficiency_ratio = if baseline_m.efficiency_percent > 0.0 {
            comparison_m.efficiency_percent / baseline_m.efficiency_percent
        } else {
            0.0
        };

        let speedup = if comparison_m.latency_us > 0.0 {
            baseline_m.latency_us / comparison_m.latency_us
        } else {
            0.0
        };

        let is_regression = speedup < (1.0 - self.threshold_percent / 100.0);

        Some(BackendComparison {
            baseline,
            comparison,
            workload,
            size,
            efficiency_ratio,
            speedup,
            is_regression,
            threshold: self.threshold_percent,
        })
    }

    /// Find measurement
    fn find_measurement(
        &self,
        backend: Backend,
        workload: WorkloadType,
        size: usize,
    ) -> Option<&BackendMeasurement> {
        self.measurements
            .iter()
            .find(|m| m.backend == backend && m.workload == workload && m.size == size)
    }

    /// Detect size cliffs for a backend
    pub fn detect_size_cliffs(&self, backend: Backend, workload: WorkloadType) -> Vec<SizeCliff> {
        let mut measurements: Vec<_> = self
            .measurements
            .iter()
            .filter(|m| m.backend == backend && m.workload == workload)
            .collect();

        measurements.sort_by_key(|m| m.size);

        let mut cliffs = Vec::new();

        for window in measurements.windows(2) {
            let before = &window[0];
            let after = &window[1];

            if before.efficiency_percent > 0.0 {
                let drop = (before.efficiency_percent - after.efficiency_percent)
                    / before.efficiency_percent
                    * 100.0;

                if drop > self.cliff_threshold_percent {
                    cliffs.push(SizeCliff {
                        backend,
                        workload,
                        size_before: before.size,
                        size_after: after.size,
                        efficiency_before: before.efficiency_percent,
                        efficiency_after: after.efficiency_percent,
                        drop_percent: drop,
                    });
                }
            }
        }

        cliffs
    }

    /// Analyze GPU transfer overhead
    pub fn analyze_transfer_overhead(
        &self,
        backend: Backend,
        workload: WorkloadType,
    ) -> Option<TransferAnalysis> {
        if !backend.is_gpu() {
            return None;
        }

        let measurements: Vec<_> = self
            .measurements
            .iter()
            .filter(|m| {
                m.backend == backend
                    && m.workload == workload
                    && m.transfer_time_us.is_some()
                    && m.compute_time_us.is_some()
            })
            .collect();

        if measurements.is_empty() {
            return None;
        }

        let mut total_transfer = 0.0;
        let mut total_compute = 0.0;
        let mut sizes_with_overhead = Vec::new();

        for m in &measurements {
            let transfer = m.transfer_time_us.unwrap();
            let compute = m.compute_time_us.unwrap();
            total_transfer += transfer;
            total_compute += compute;

            let overhead = transfer / (transfer + compute);
            if overhead > 0.5 {
                sizes_with_overhead.push((m.size, overhead));
            }
        }

        let avg_overhead = total_transfer / (total_transfer + total_compute);

        Some(TransferAnalysis {
            backend,
            workload,
            average_overhead: avg_overhead,
            total_transfer_time_us: total_transfer,
            total_compute_time_us: total_compute,
            sizes_dominated_by_transfer: sizes_with_overhead,
        })
    }

    /// Recommend best backend for given workload/size
    pub fn recommend_backend(
        &self,
        workload: WorkloadType,
        size: usize,
    ) -> Option<BackendRecommendation> {
        let candidates: Vec<_> = self
            .measurements
            .iter()
            .filter(|m| m.workload == workload && m.size == size)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Find best by throughput
        let best = candidates
            .iter()
            .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())?;

        // Calculate confidence based on efficiency
        let confidence = (best.efficiency_percent / 100.0).clamp(0.0, 1.0);

        let reason = if best.backend.is_gpu() {
            if let Some(overhead) = best.transfer_overhead() {
                if overhead > 0.3 {
                    format!(
                        "GPU selected but transfer overhead is {:.1}%",
                        overhead * 100.0
                    )
                } else {
                    "Best throughput with low transfer overhead".to_string()
                }
            } else {
                "Best throughput among available backends".to_string()
            }
        } else {
            "Best CPU backend for this size".to_string()
        };

        Some(BackendRecommendation {
            backend: best.backend,
            workload,
            size,
            expected_efficiency: best.efficiency_percent,
            confidence,
            reason,
        })
    }

    /// Get all comparisons for a workload
    pub fn compare_all_backends(&self, workload: WorkloadType) -> Vec<BackendComparison> {
        let sizes: Vec<usize> = self
            .measurements
            .iter()
            .filter(|m| m.workload == workload)
            .map(|m| m.size)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let backends: Vec<Backend> = self
            .measurements
            .iter()
            .filter(|m| m.workload == workload)
            .map(|m| m.backend)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let mut comparisons = Vec::new();

        for size in &sizes {
            // Compare each backend against Scalar baseline
            if let Some(scalar) = backends.iter().find(|b| **b == Backend::Scalar) {
                for backend in &backends {
                    if *backend != Backend::Scalar {
                        if let Some(cmp) = self.compare_backends(*scalar, *backend, workload, *size)
                        {
                            comparisons.push(cmp);
                        }
                    }
                }
            }
        }

        comparisons
    }

    /// Detect all regressions
    pub fn detect_regressions(&self) -> Vec<BackendComparison> {
        let workloads: Vec<WorkloadType> = self
            .measurements
            .iter()
            .map(|m| m.workload)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let mut regressions = Vec::new();

        for workload in workloads {
            let comparisons = self.compare_all_backends(workload);
            for cmp in comparisons {
                if cmp.is_regression {
                    regressions.push(cmp);
                }
            }
        }

        regressions
    }

    /// Generate summary report
    pub fn summary(&self) -> BackendSummary {
        let workloads: Vec<WorkloadType> = self
            .measurements
            .iter()
            .map(|m| m.workload)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let backends: Vec<Backend> = self
            .measurements
            .iter()
            .map(|m| m.backend)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let regressions = self.detect_regressions();

        let mut all_cliffs = Vec::new();
        for backend in &backends {
            for workload in &workloads {
                all_cliffs.extend(self.detect_size_cliffs(*backend, *workload));
            }
        }

        BackendSummary {
            measurement_count: self.measurements.len(),
            backend_count: backends.len(),
            workload_count: workloads.len(),
            regression_count: regressions.len(),
            cliff_count: all_cliffs.len(),
            regressions,
            cliffs: all_cliffs,
        }
    }

    /// Check if backend is available
    pub fn is_backend_available(&self, backend: Backend) -> bool {
        self.available_backends.contains(&backend)
    }

    /// Get available backends
    pub fn available_backends(&self) -> &[Backend] {
        &self.available_backends
    }

    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
    }
}

/// GPU transfer overhead analysis
#[derive(Debug, Clone)]
pub struct TransferAnalysis {
    /// Backend analyzed
    pub backend: Backend,
    /// Workload type
    pub workload: WorkloadType,
    /// Average transfer overhead (0.0 - 1.0)
    pub average_overhead: f64,
    /// Total transfer time
    pub total_transfer_time_us: f64,
    /// Total compute time
    pub total_compute_time_us: f64,
    /// Sizes where transfer dominates (>50% overhead)
    pub sizes_dominated_by_transfer: Vec<(usize, f64)>,
}

impl TransferAnalysis {
    /// Check if transfer dominates compute
    pub fn transfer_dominated(&self) -> bool {
        self.average_overhead > 0.5
    }

    /// Get summary
    pub fn summary(&self) -> String {
        if self.transfer_dominated() {
            format!(
                "{} {}: Transfer overhead {:.1}% - consider larger batches",
                self.backend.name(),
                self.workload.name(),
                self.average_overhead * 100.0
            )
        } else {
            format!(
                "{} {}: Transfer overhead {:.1}% - GPU efficient",
                self.backend.name(),
                self.workload.name(),
                self.average_overhead * 100.0
            )
        }
    }
}

/// Summary of backend regression analysis
#[derive(Debug, Clone)]
pub struct BackendSummary {
    /// Total measurements
    pub measurement_count: usize,
    /// Number of backends tested
    pub backend_count: usize,
    /// Number of workloads tested
    pub workload_count: usize,
    /// Number of regressions detected
    pub regression_count: usize,
    /// Number of size cliffs detected
    pub cliff_count: usize,
    /// All regressions
    pub regressions: Vec<BackendComparison>,
    /// All cliffs
    pub cliffs: Vec<SizeCliff>,
}

impl BackendSummary {
    /// Check if any regressions detected
    pub fn has_regressions(&self) -> bool {
        self.regression_count > 0
    }

    /// Check if any cliffs detected
    pub fn has_cliffs(&self) -> bool {
        self.cliff_count > 0
    }

    /// Get status message
    pub fn status(&self) -> &'static str {
        if self.regression_count > 0 {
            "FAIL: Regressions detected"
        } else if self.cliff_count > 0 {
            "WARN: Size cliffs detected"
        } else {
            "PASS: No issues detected"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_names() {
        assert_eq!(Backend::Scalar.name(), "Scalar");
        assert_eq!(Backend::Avx2.name(), "AVX2");
        assert_eq!(Backend::Cuda.name(), "CUDA");
    }

    #[test]
    fn test_backend_is_gpu() {
        assert!(!Backend::Scalar.is_gpu());
        assert!(!Backend::Avx2.is_gpu());
        assert!(Backend::Cuda.is_gpu());
        assert!(Backend::Metal.is_gpu());
    }

    #[test]
    fn test_backend_is_simd() {
        assert!(!Backend::Scalar.is_simd());
        assert!(Backend::Sse2.is_simd());
        assert!(Backend::Avx2.is_simd());
        assert!(!Backend::Cuda.is_simd());
    }

    #[test]
    fn test_measurement_creation() {
        let m = BackendMeasurement::new(Backend::Avx2, WorkloadType::Gemm, 1024, 100.0, 10000.0)
            .with_efficiency(85.0);

        assert_eq!(m.backend, Backend::Avx2);
        assert_eq!(m.size, 1024);
        assert_eq!(m.efficiency_percent, 85.0);
    }

    #[test]
    fn test_detector_add_measurement() {
        let mut detector = BackendRegressionDetector::new();

        detector.add(
            Backend::Scalar,
            WorkloadType::Gemm,
            1024,
            1000.0,
            1000.0,
            50.0,
        );
        detector.add(Backend::Avx2, WorkloadType::Gemm, 1024, 250.0, 4000.0, 80.0);

        assert_eq!(detector.measurement_count(), 2);
    }

    #[test]
    fn test_compare_backends() {
        let mut detector = BackendRegressionDetector::new();

        detector.add(
            Backend::Scalar,
            WorkloadType::Gemm,
            1024,
            1000.0,
            1000.0,
            50.0,
        );
        detector.add(Backend::Avx2, WorkloadType::Gemm, 1024, 250.0, 4000.0, 80.0);

        let cmp = detector
            .compare_backends(Backend::Scalar, Backend::Avx2, WorkloadType::Gemm, 1024)
            .unwrap();

        assert!(cmp.speedup > 3.0);
        assert!(!cmp.is_regression);
    }

    #[test]
    fn test_detect_cliff() {
        let mut detector = BackendRegressionDetector::new().with_cliff_threshold(10.0);

        // Normal efficiency at small sizes
        detector.add(
            Backend::Avx2,
            WorkloadType::Gemm,
            1024,
            100.0,
            10000.0,
            90.0,
        );
        detector.add(
            Backend::Avx2,
            WorkloadType::Gemm,
            2048,
            200.0,
            10000.0,
            88.0,
        );
        // Cliff: efficiency drops significantly
        detector.add(Backend::Avx2, WorkloadType::Gemm, 4096, 500.0, 8000.0, 60.0);

        let cliffs = detector.detect_size_cliffs(Backend::Avx2, WorkloadType::Gemm);

        assert!(!cliffs.is_empty());
        assert!(cliffs[0].drop_percent > 10.0);
    }

    #[test]
    fn test_recommend_backend() {
        let mut detector = BackendRegressionDetector::new();

        detector.add(
            Backend::Scalar,
            WorkloadType::Gemm,
            1024,
            1000.0,
            1000.0,
            50.0,
        );
        detector.add(Backend::Avx2, WorkloadType::Gemm, 1024, 250.0, 4000.0, 80.0);
        detector.add(
            Backend::Cuda,
            WorkloadType::Gemm,
            1024,
            100.0,
            10000.0,
            95.0,
        );

        let rec = detector
            .recommend_backend(WorkloadType::Gemm, 1024)
            .unwrap();

        assert_eq!(rec.backend, Backend::Cuda);
    }

    #[test]
    fn test_transfer_overhead() {
        let m = BackendMeasurement::new(Backend::Cuda, WorkloadType::Gemm, 1024, 100.0, 10000.0)
            .with_gpu_timing(30.0, 70.0);

        let overhead = m.transfer_overhead().unwrap();
        assert!((overhead - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_summary() {
        let mut detector = BackendRegressionDetector::new();

        detector.add(
            Backend::Scalar,
            WorkloadType::Gemm,
            1024,
            1000.0,
            1000.0,
            50.0,
        );
        detector.add(Backend::Avx2, WorkloadType::Gemm, 1024, 250.0, 4000.0, 80.0);

        let summary = detector.summary();

        assert_eq!(summary.measurement_count, 2);
        assert_eq!(summary.backend_count, 2);
    }
}
