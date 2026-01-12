//! Falsification Tests for PMAT-031: Cross-Backend Regression Detector
//!
//! F1231-F1240: Backend regression falsification tests
//!
//! These tests verify the backend regression detector for:
//! - Cross-backend comparison
//! - Size cliff detection
//! - Best backend recommendation
//! - Transfer overhead analysis

use cbtop::{
    BackendRegressionDetector, BackendMeasurement, BackendComparison,
    Backend, BackendWorkload, SizeCliff, BackendRecommendation,
};

// =============================================================================
// F1231: Backend Comparison Tests
// =============================================================================

/// F1231.1: Backend comparison works for all backends
#[test]
fn f1231_backend_comparison() {
    let mut detector = BackendRegressionDetector::new();

    // Add measurements for different backends
    detector.add(Backend::Scalar, BackendWorkload::Gemm, 1024, 1000.0, 1000.0, 50.0);
    detector.add(Backend::Sse2, BackendWorkload::Gemm, 1024, 500.0, 2000.0, 60.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 250.0, 4000.0, 80.0);

    let cmp = detector
        .compare_backends(Backend::Scalar, Backend::Avx2, BackendWorkload::Gemm, 1024)
        .unwrap();

    assert_eq!(cmp.baseline, Backend::Scalar);
    assert_eq!(cmp.comparison, Backend::Avx2);
    assert!(cmp.speedup > 1.0);
}

/// F1231.2: Missing backend returns None
#[test]
fn f1231_missing_backend() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(Backend::Scalar, BackendWorkload::Gemm, 1024, 1000.0, 1000.0, 50.0);

    let result = detector.compare_backends(
        Backend::Scalar,
        Backend::Cuda, // Not added
        BackendWorkload::Gemm,
        1024,
    );

    assert!(result.is_none());
}

// =============================================================================
// F1232: Efficiency Ratio Tests
// =============================================================================

/// F1232.1: Efficiency ratio calculated correctly
#[test]
fn f1232_efficiency_ratio() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(Backend::Scalar, BackendWorkload::Gemm, 1024, 1000.0, 1000.0, 50.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 250.0, 4000.0, 80.0);

    let cmp = detector
        .compare_backends(Backend::Scalar, Backend::Avx2, BackendWorkload::Gemm, 1024)
        .unwrap();

    // Efficiency ratio = 80 / 50 = 1.6
    assert!((cmp.efficiency_ratio - 1.6).abs() < 0.1);
}

/// F1232.2: Zero baseline efficiency handled
#[test]
fn f1232_zero_baseline() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(Backend::Scalar, BackendWorkload::Gemm, 1024, 1000.0, 1000.0, 0.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 250.0, 4000.0, 80.0);

    let cmp = detector
        .compare_backends(Backend::Scalar, Backend::Avx2, BackendWorkload::Gemm, 1024)
        .unwrap();

    // Should not divide by zero
    assert_eq!(cmp.efficiency_ratio, 0.0);
}

// =============================================================================
// F1233: Size Cliff Detection Tests
// =============================================================================

/// F1233.1: Size cliff detected when >10% drop
#[test]
fn f1233_cliff_detection() {
    let mut detector = BackendRegressionDetector::new().with_cliff_threshold(10.0);

    // Efficiency drops from 90% to 60% at 4M elements
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1_000_000, 100.0, 10000.0, 90.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 2_000_000, 200.0, 10000.0, 88.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 4_000_000, 500.0, 8000.0, 60.0);

    let cliffs = detector.detect_size_cliffs(Backend::Avx2, BackendWorkload::Gemm);

    assert!(!cliffs.is_empty());
    let cliff = &cliffs[0];
    assert!(cliff.drop_percent > 10.0);
    assert_eq!(cliff.size_after, 4_000_000);
}

/// F1233.2: No cliff for gradual changes
#[test]
fn f1233_no_cliff_gradual() {
    let mut detector = BackendRegressionDetector::new().with_cliff_threshold(10.0);

    // Gradual efficiency changes
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 100.0, 10000.0, 90.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 2048, 200.0, 10000.0, 88.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 4096, 400.0, 10000.0, 85.0);

    let cliffs = detector.detect_size_cliffs(Backend::Avx2, BackendWorkload::Gemm);

    assert!(cliffs.is_empty());
}

// =============================================================================
// F1234: GPU Transfer Overhead Tests
// =============================================================================

/// F1234.1: GPU transfer overhead measured
#[test]
fn f1234_transfer_overhead() {
    let mut detector = BackendRegressionDetector::new();

    let m = BackendMeasurement::new(Backend::Cuda, BackendWorkload::Gemm, 1024, 100.0, 10000.0)
        .with_gpu_timing(30.0, 70.0);

    detector.add_measurement(m);

    let analysis = detector
        .analyze_transfer_overhead(Backend::Cuda, BackendWorkload::Gemm)
        .unwrap();

    assert!((analysis.average_overhead - 0.3).abs() < 0.01);
}

/// F1234.2: Non-GPU returns None
#[test]
fn f1234_non_gpu_no_analysis() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 100.0, 10000.0, 80.0);

    let analysis = detector.analyze_transfer_overhead(Backend::Avx2, BackendWorkload::Gemm);

    assert!(analysis.is_none());
}

// =============================================================================
// F1235: Best Backend Selection Tests
// =============================================================================

/// F1235.1: Best backend selected by throughput
#[test]
fn f1235_best_backend() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(Backend::Scalar, BackendWorkload::Gemm, 1024, 1000.0, 1000.0, 50.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 250.0, 4000.0, 80.0);
    detector.add(Backend::Cuda, BackendWorkload::Gemm, 1024, 100.0, 10000.0, 95.0);

    let rec = detector.recommend_backend(BackendWorkload::Gemm, 1024).unwrap();

    assert_eq!(rec.backend, Backend::Cuda);
    assert!(rec.expected_efficiency >= 90.0);
}

/// F1235.2: Returns None for unknown workload/size
#[test]
fn f1235_unknown_returns_none() {
    let detector = BackendRegressionDetector::new();

    let rec = detector.recommend_backend(BackendWorkload::Gemm, 1024);

    assert!(rec.is_none());
}

// =============================================================================
// F1236: Regression Threshold Configuration Tests
// =============================================================================

/// F1236.1: Custom threshold works
#[test]
fn f1236_custom_threshold() {
    let mut detector = BackendRegressionDetector::new().with_threshold(5.0);

    // 3% slower - within 5% threshold
    detector.add(Backend::Scalar, BackendWorkload::Gemm, 1024, 1000.0, 1000.0, 50.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 1030.0, 970.0, 48.0);

    let cmp = detector
        .compare_backends(Backend::Scalar, Backend::Avx2, BackendWorkload::Gemm, 1024)
        .unwrap();

    assert!(!cmp.is_regression);

    // With stricter threshold
    let detector2 = BackendRegressionDetector::new().with_threshold(2.0);
    // Same measurements would be regression
}

/// F1236.2: Default threshold is 10%
#[test]
fn f1236_default_threshold() {
    let mut detector = BackendRegressionDetector::new();

    // 15% slower - exceeds 10% default threshold
    detector.add(Backend::Scalar, BackendWorkload::Gemm, 1024, 1000.0, 1000.0, 50.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 1200.0, 833.0, 42.0);

    let cmp = detector
        .compare_backends(Backend::Scalar, Backend::Avx2, BackendWorkload::Gemm, 1024)
        .unwrap();

    assert!(cmp.is_regression);
    assert_eq!(cmp.threshold, 10.0);
}

// =============================================================================
// F1237: Backend Availability Tests
// =============================================================================

/// F1237.1: Unavailable backends skipped
#[test]
fn f1237_backend_availability() {
    let detector = BackendRegressionDetector::new()
        .with_backends(vec![Backend::Scalar, Backend::Avx2]);

    assert!(detector.is_backend_available(Backend::Scalar));
    assert!(detector.is_backend_available(Backend::Avx2));
    assert!(!detector.is_backend_available(Backend::Cuda));
}

/// F1237.2: Custom backend list works
#[test]
fn f1237_custom_backends() {
    let detector = BackendRegressionDetector::new()
        .with_backends(vec![Backend::Cuda, Backend::Metal]);

    let backends = detector.available_backends();
    assert_eq!(backends.len(), 2);
    assert!(backends.contains(&Backend::Cuda));
    assert!(backends.contains(&Backend::Metal));
}

// =============================================================================
// F1238: Comparison Summary Tests
// =============================================================================

/// F1238.1: Summary generated
#[test]
fn f1238_summary_generated() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(Backend::Scalar, BackendWorkload::Gemm, 1024, 1000.0, 1000.0, 50.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 250.0, 4000.0, 80.0);

    let summary = detector.summary();

    assert_eq!(summary.measurement_count, 2);
    assert_eq!(summary.backend_count, 2);
    assert_eq!(summary.workload_count, 1);
}

/// F1238.2: Summary status reflects regressions
#[test]
fn f1238_summary_status() {
    let detector = BackendRegressionDetector::new();
    let summary = detector.summary();

    assert!(!summary.has_regressions());
    assert_eq!(summary.status(), "PASS: No issues detected");
}

// =============================================================================
// F1239: Historical Comparison Tests
// =============================================================================

/// F1239.1: Multiple measurements over time
#[test]
fn f1239_historical_comparison() {
    let mut detector = BackendRegressionDetector::new();

    // Add measurements at different times/runs
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 250.0, 4000.0, 80.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 2048, 500.0, 4000.0, 78.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 4096, 1000.0, 4000.0, 75.0);

    // Can detect trends
    let cliffs = detector.detect_size_cliffs(Backend::Avx2, BackendWorkload::Gemm);
    // Gradual decline, no cliff
    assert!(cliffs.is_empty());
}

// =============================================================================
// F1240: Multiple Workload Types Tests
// =============================================================================

/// F1240.1: Different workload types compared
#[test]
fn f1240_multiple_workloads() {
    let mut detector = BackendRegressionDetector::new();

    // GEMM workload
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 250.0, 4000.0, 80.0);

    // Elementwise workload (different characteristics)
    detector.add(Backend::Avx2, BackendWorkload::Elementwise, 1024, 50.0, 20000.0, 95.0);

    // Reduction workload
    detector.add(Backend::Avx2, BackendWorkload::Reduction, 1024, 100.0, 10000.0, 90.0);

    let summary = detector.summary();
    assert_eq!(summary.workload_count, 3);
}

/// F1240.2: Workload names correct
#[test]
fn f1240_workload_names() {
    assert_eq!(BackendWorkload::Gemm.name(), "GEMM");
    assert_eq!(BackendWorkload::Conv2d.name(), "Conv2D");
    assert_eq!(BackendWorkload::Elementwise.name(), "Elementwise");
    assert_eq!(BackendWorkload::Reduction.name(), "Reduction");
    assert_eq!(BackendWorkload::Attention.name(), "Attention");
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test backend GPU detection
#[test]
fn test_backend_is_gpu() {
    assert!(Backend::Cuda.is_gpu());
    assert!(Backend::Metal.is_gpu());
    assert!(Backend::Vulkan.is_gpu());
    assert!(Backend::WebGpu.is_gpu());
    assert!(!Backend::Scalar.is_gpu());
    assert!(!Backend::Avx2.is_gpu());
}

/// Test backend SIMD detection
#[test]
fn test_backend_is_simd() {
    assert!(Backend::Sse2.is_simd());
    assert!(Backend::Avx2.is_simd());
    assert!(Backend::Avx512.is_simd());
    assert!(Backend::Neon.is_simd());
    assert!(!Backend::Scalar.is_simd());
    assert!(!Backend::Cuda.is_simd());
}

/// Test theoretical speedup
#[test]
fn test_theoretical_speedup() {
    assert_eq!(Backend::Scalar.theoretical_speedup(), 1.0);
    assert_eq!(Backend::Sse2.theoretical_speedup(), 4.0);
    assert_eq!(Backend::Avx2.theoretical_speedup(), 8.0);
    assert_eq!(Backend::Avx512.theoretical_speedup(), 16.0);
}

/// Test comparison summary message
#[test]
fn test_comparison_summary() {
    let cmp = BackendComparison {
        baseline: Backend::Scalar,
        comparison: Backend::Avx2,
        workload: BackendWorkload::Gemm,
        size: 1024,
        efficiency_ratio: 1.6,
        speedup: 4.0,
        is_regression: false,
        threshold: 10.0,
    };

    let summary = cmp.summary();
    assert!(summary.contains("OK"));
    assert!(summary.contains("4.0x"));
}

/// Test cliff summary message
#[test]
fn test_cliff_summary() {
    let cliff = SizeCliff {
        backend: Backend::Avx2,
        workload: BackendWorkload::Gemm,
        size_before: 2_000_000,
        size_after: 4_000_000,
        efficiency_before: 90.0,
        efficiency_after: 60.0,
        drop_percent: 33.3,
    };

    let summary = cliff.summary();
    assert!(summary.contains("CLIFF"));
    assert!(summary.contains("33.3%"));
}

/// Test transfer analysis summary
#[test]
fn test_transfer_analysis_summary() {
    let mut detector = BackendRegressionDetector::new();

    let m = BackendMeasurement::new(Backend::Cuda, BackendWorkload::Gemm, 1024, 100.0, 10000.0)
        .with_gpu_timing(60.0, 40.0); // Transfer dominated

    detector.add_measurement(m);

    let analysis = detector
        .analyze_transfer_overhead(Backend::Cuda, BackendWorkload::Gemm)
        .unwrap();

    assert!(analysis.transfer_dominated());
    let summary = analysis.summary();
    assert!(summary.contains("larger batches"));
}

/// Test detector clear
#[test]
fn test_detector_clear() {
    let mut detector = BackendRegressionDetector::new();

    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 250.0, 4000.0, 80.0);
    assert_eq!(detector.measurement_count(), 1);

    detector.clear();
    assert_eq!(detector.measurement_count(), 0);
}

/// Test detect all regressions
#[test]
fn test_detect_all_regressions() {
    let mut detector = BackendRegressionDetector::new();

    // Good comparison
    detector.add(Backend::Scalar, BackendWorkload::Gemm, 1024, 1000.0, 1000.0, 50.0);
    detector.add(Backend::Avx2, BackendWorkload::Gemm, 1024, 250.0, 4000.0, 80.0);

    let regressions = detector.detect_regressions();
    assert!(regressions.is_empty());
}
