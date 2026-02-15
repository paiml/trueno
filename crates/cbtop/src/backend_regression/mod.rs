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

mod analysis;
mod types;

pub use analysis::{BackendSummary, TransferAnalysis};
pub use types::{
    Backend, BackendComparison, BackendMeasurement, BackendRecommendation, SizeCliff, WorkloadType,
};

use std::collections::HashSet;

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
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let backends: Vec<Backend> = self
            .measurements
            .iter()
            .filter(|m| m.workload == workload)
            .map(|m| m.backend)
            .collect::<HashSet<_>>()
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
            .collect::<HashSet<_>>()
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
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let backends: Vec<Backend> = self
            .measurements
            .iter()
            .map(|m| m.backend)
            .collect::<HashSet<_>>()
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


#[cfg(test)]
mod tests;
