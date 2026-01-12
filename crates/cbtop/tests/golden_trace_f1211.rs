//! Falsification Tests for PMAT-029: Golden Trace Comparison
//!
//! F1211-F1220: Golden trace comparison falsification tests
//!
//! These tests verify the golden trace module for:
//! - Trace capture and storage
//! - Comparison and delta calculation
//! - Regression detection
//! - Export/import functionality

use cbtop::{
    GoldenTrace, GoldenTraceManager, GoldenComparator, GoldenTraceError,
    TraceMetrics, TraceSyscallBreakdown as SyscallBreakdown, TraceComparison,
};
use tempfile::TempDir;

// =============================================================================
// F1211: Golden Trace Capture Tests
// =============================================================================

/// F1211.1: Golden trace captures metrics
#[test]
fn f1211_capture_metrics() {
    let metrics = TraceMetrics::new()
        .total_time_us(1234.5)
        .p50_latency_us(50.0)
        .p99_latency_us(100.0)
        .throughput(10000.0)
        .peak_memory_bytes(1024 * 1024);

    let trace = GoldenTrace::new("capture_test", metrics);

    // All fields should be populated
    assert_eq!(trace.name, "capture_test");
    assert_eq!(trace.metrics.total_time_us, 1234.5);
    assert_eq!(trace.metrics.p50_latency_us, 50.0);
    assert_eq!(trace.metrics.p99_latency_us, 100.0);
    assert_eq!(trace.metrics.throughput, 10000.0);
    assert_eq!(trace.metrics.peak_memory_bytes, 1024 * 1024);
    assert!(!trace.hash.is_empty());
    assert!(trace.timestamp > 0);
}

/// F1211.2: Capture with syscall breakdown
#[test]
fn f1211_capture_with_syscalls() {
    let syscalls = SyscallBreakdown {
        read_count: 100,
        write_count: 50,
        mmap_count: 10,
        futex_count: 25,
        other_count: 15,
    };

    let metrics = TraceMetrics::new()
        .total_time_us(1000.0)
        .syscalls(syscalls);

    let trace = GoldenTrace::new("syscall_test", metrics);

    assert_eq!(trace.metrics.syscalls.read_count, 100);
    assert_eq!(trace.metrics.syscalls.write_count, 50);
    assert_eq!(trace.metrics.syscalls.total(), 200);
}

// =============================================================================
// F1212: Trace Comparison Delta Tests
// =============================================================================

/// F1212.1: Trace comparison calculates delta correctly
#[test]
fn f1212_delta_calculation() {
    let golden = GoldenTrace::new(
        "baseline",
        TraceMetrics::new()
            .total_time_us(1000.0)
            .p50_latency_us(50.0)
            .throughput(10000.0),
    );

    let current = TraceMetrics::new()
        .total_time_us(1100.0) // 10% slower
        .p50_latency_us(55.0)  // 10% slower
        .throughput(9000.0);   // 10% lower

    let comparator = GoldenComparator::new();
    let result = comparator.compare(&current, &golden).unwrap();

    // Check delta calculations
    assert!((result.time_delta_percent - 10.0).abs() < 0.1);
    assert!((result.p50_delta_percent - 10.0).abs() < 0.1);
    assert!((result.throughput_delta_percent - (-10.0)).abs() < 0.1);
}

/// F1212.2: Zero baseline handled correctly
#[test]
fn f1212_zero_baseline() {
    let golden = GoldenTrace::new(
        "zero_baseline",
        TraceMetrics::new()
            .total_time_us(0.0)
            .throughput(0.0),
    );

    let current = TraceMetrics::new()
        .total_time_us(100.0)
        .throughput(1000.0);

    let comparator = GoldenComparator::new();
    let result = comparator.compare(&current, &golden).unwrap();

    // New values from zero should be 100% delta
    assert!(result.time_delta_percent > 0.0);
}

// =============================================================================
// F1213: Regression Detection Tests
// =============================================================================

/// F1213.1: Regression detected at >10% threshold
#[test]
fn f1213_regression_detection() {
    let golden = GoldenTrace::new(
        "baseline",
        TraceMetrics::new()
            .total_time_us(1000.0)
            .p50_latency_us(50.0)
            .throughput(10000.0),
    );

    // Current is 15% worse
    let current = TraceMetrics::new()
        .total_time_us(1150.0) // 15% slower
        .p50_latency_us(57.5)
        .throughput(8500.0);

    let comparator = GoldenComparator::new().with_threshold(10.0);
    let result = comparator.compare(&current, &golden).unwrap();

    assert!(result.is_regression);
    assert!(result.time_delta_percent > 10.0);
}

/// F1213.2: No regression below threshold
#[test]
fn f1213_no_regression_below_threshold() {
    let golden = GoldenTrace::new(
        "baseline",
        TraceMetrics::new()
            .total_time_us(1000.0)
            .p50_latency_us(50.0)
            .throughput(10000.0),
    );

    // Current is 5% worse (within 10% threshold)
    let current = TraceMetrics::new()
        .total_time_us(1050.0)
        .p50_latency_us(52.5)
        .throughput(9500.0);

    let comparator = GoldenComparator::new().with_threshold(10.0);
    let result = comparator.compare(&current, &golden).unwrap();

    assert!(!result.is_regression);
}

/// F1213.3: Custom threshold works
#[test]
fn f1213_custom_threshold() {
    let golden = GoldenTrace::new(
        "baseline",
        TraceMetrics::new().total_time_us(1000.0),
    );

    let current = TraceMetrics::new().total_time_us(1060.0); // 6% slower

    // 5% threshold - should be regression
    let comparator = GoldenComparator::new().with_threshold(5.0);
    let result = comparator.compare(&current, &golden).unwrap();
    assert!(result.is_regression);

    // 10% threshold - should not be regression
    let comparator = GoldenComparator::new().with_threshold(10.0);
    let result = comparator.compare(&current, &golden).unwrap();
    assert!(!result.is_regression);
}

// =============================================================================
// F1214: Golden Trace Versioning Tests
// =============================================================================

/// F1214.1: Golden trace versioned
#[test]
fn f1214_version_stored() {
    let trace = GoldenTrace::with_version(
        "versioned",
        "2.0",
        TraceMetrics::new().total_time_us(1000.0),
    );

    assert_eq!(trace.version, "2.0");
}

/// F1214.2: Git commit stored
#[test]
fn f1214_git_commit_stored() {
    let trace = GoldenTrace::new("git_test", TraceMetrics::new())
        .git_commit("abc123def456");

    assert_eq!(trace.git_commit, Some("abc123def456".to_string()));
}

// =============================================================================
// F1215: Trace Timestamps Tests
// =============================================================================

/// F1215.1: Trace timestamps preserved
#[test]
fn f1215_timestamps_preserved() {
    let trace1 = GoldenTrace::new("first", TraceMetrics::new());
    std::thread::sleep(std::time::Duration::from_millis(10));
    let trace2 = GoldenTrace::new("second", TraceMetrics::new());

    // Timestamps should be chronological
    assert!(trace2.timestamp >= trace1.timestamp);
}

/// F1215.2: Timestamp through serialization
#[test]
fn f1215_timestamp_serialization() {
    let original = GoldenTrace::new("timestamp_test", TraceMetrics::new());
    let timestamp = original.timestamp;

    let json = original.to_json().unwrap();
    let parsed = GoldenTrace::from_json(&json).unwrap();

    assert_eq!(parsed.timestamp, timestamp);
}

// =============================================================================
// F1216: Breakdown Delta Tests
// =============================================================================

/// F1216.1: Per-syscall diff calculated
#[test]
fn f1216_syscall_diff() {
    let baseline = SyscallBreakdown {
        read_count: 100,
        write_count: 50,
        mmap_count: 10,
        futex_count: 20,
        other_count: 5,
    };

    let current = SyscallBreakdown {
        read_count: 120,  // 20% increase
        write_count: 50,  // unchanged
        mmap_count: 15,   // 50% increase
        futex_count: 22,  // 10% increase
        other_count: 5,   // unchanged
    };

    let delta = current.percentage_diff(&baseline);

    assert!((delta.read_delta - 20.0).abs() < 0.1);
    assert!(delta.write_delta.abs() < 0.1);
    assert!((delta.mmap_delta - 50.0).abs() < 0.1);
    assert!((delta.futex_delta - 10.0).abs() < 0.1);
}

/// F1216.2: Max delta detection
#[test]
fn f1216_max_delta() {
    let baseline = SyscallBreakdown {
        read_count: 100,
        write_count: 100,
        mmap_count: 100,
        futex_count: 100,
        other_count: 100,
    };

    let current = SyscallBreakdown {
        read_count: 100,
        write_count: 100,
        mmap_count: 200,  // 100% increase - max delta
        futex_count: 100,
        other_count: 100,
    };

    let delta = current.percentage_diff(&baseline);
    assert!((delta.max_delta() - 100.0).abs() < 0.1);
}

// =============================================================================
// F1217: Empty Golden Handling Tests
// =============================================================================

/// F1217.1: Empty golden returns baseline error
#[test]
fn f1217_no_baseline() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = GoldenTraceManager::new(temp_dir.path().to_path_buf());
    manager.ensure_directory().unwrap();

    let result = manager.load_golden("nonexistent");
    assert!(matches!(result, Err(GoldenTraceError::NoBaseline)));
}

/// F1217.2: Compare without golden fails
#[test]
fn f1217_compare_no_golden() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = GoldenTraceManager::new(temp_dir.path().to_path_buf());
    manager.ensure_directory().unwrap();

    let current = TraceMetrics::new().total_time_us(1000.0);
    let result = manager.compare_to_golden("missing", &current);

    assert!(matches!(result, Err(GoldenTraceError::NoBaseline)));
}

// =============================================================================
// F1218: Trace Hash Tests
// =============================================================================

/// F1218.1: Trace hash computed
#[test]
fn f1218_hash_computed() {
    let trace = GoldenTrace::new("hash_test", TraceMetrics::new().total_time_us(1000.0));

    assert!(!trace.hash.is_empty());
    assert_eq!(trace.hash.len(), 16); // 16 hex chars
}

/// F1218.2: Hash is deterministic
#[test]
fn f1218_hash_deterministic() {
    let metrics = TraceMetrics::new().total_time_us(1000.0);

    // Same name and metrics should produce same hash
    let trace1 = GoldenTrace::new("deterministic", metrics.clone());
    let computed = trace1.compute_hash();

    // Hash should be consistent
    assert_eq!(trace1.hash, computed);
    assert!(trace1.verify_hash());
}

// =============================================================================
// F1219: Multiple Goldens Tests
// =============================================================================

/// F1219.1: Multiple goldens supported
#[test]
fn f1219_multiple_goldens() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = GoldenTraceManager::new(temp_dir.path().to_path_buf());

    // Save multiple goldens
    manager.capture_golden("v1", TraceMetrics::new().total_time_us(1000.0)).unwrap();
    manager.capture_golden("v2", TraceMetrics::new().total_time_us(900.0)).unwrap();
    manager.capture_golden("v3", TraceMetrics::new().total_time_us(800.0)).unwrap();

    let names = manager.list_goldens().unwrap();
    assert_eq!(names.len(), 3);
}

/// F1219.2: Version selection works
#[test]
fn f1219_version_selection() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = GoldenTraceManager::new(temp_dir.path().to_path_buf());

    manager.capture_golden("release_1_0", TraceMetrics::new().total_time_us(1000.0)).unwrap();
    manager.capture_golden("release_2_0", TraceMetrics::new().total_time_us(800.0)).unwrap();

    // Can load specific version
    let v1 = manager.load_golden("release_1_0").unwrap();
    assert_eq!(v1.metrics.total_time_us, 1000.0);

    let v2 = manager.load_golden("release_2_0").unwrap();
    assert_eq!(v2.metrics.total_time_us, 800.0);
}

// =============================================================================
// F1220: Export Format Tests
// =============================================================================

/// F1220.1: JSON export valid
#[test]
fn f1220_json_export() {
    let metrics = TraceMetrics::new()
        .total_time_us(1000.0)
        .p50_latency_us(50.0);
    let trace = GoldenTrace::new("json_export", metrics);

    let json = trace.to_json().unwrap();

    // Should be valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["name"], "json_export");
    assert_eq!(parsed["metrics"]["total_time_us"], 1000.0);
}

/// F1220.2: TOML export valid
#[test]
fn f1220_toml_export() {
    let metrics = TraceMetrics::new()
        .total_time_us(1000.0)
        .p50_latency_us(50.0);
    let trace = GoldenTrace::new("toml_export", metrics);

    let toml_str = trace.to_toml().unwrap();

    // Should be valid TOML
    let parsed: toml::Value = toml::from_str(&toml_str).unwrap();
    assert_eq!(parsed["name"].as_str(), Some("toml_export"));
}

/// F1220.3: Export to file
#[test]
fn f1220_export_to_file() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = GoldenTraceManager::new(temp_dir.path().join("goldens"));

    manager.capture_golden("exportable", TraceMetrics::new().total_time_us(1000.0)).unwrap();

    // Export to different location
    let export_path = temp_dir.path().join("exported.toml");
    manager.export_trace("exportable", &export_path).unwrap();

    // File should exist and be valid
    assert!(export_path.exists());
    let content = std::fs::read_to_string(&export_path).unwrap();
    assert!(content.contains("exportable"));
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test trace deletion
#[test]
fn test_trace_deletion() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = GoldenTraceManager::new(temp_dir.path().to_path_buf());

    manager.capture_golden("deletable", TraceMetrics::new()).unwrap();
    assert!(manager.golden_exists("deletable"));

    manager.delete_golden("deletable").unwrap();
    assert!(!manager.golden_exists("deletable"));
}

/// Test regression summary message
#[test]
fn test_regression_summary() {
    let golden = GoldenTrace::new(
        "baseline",
        TraceMetrics::new().total_time_us(1000.0),
    );

    let current = TraceMetrics::new().total_time_us(1500.0); // 50% regression

    let comparator = GoldenComparator::new();
    let result = comparator.compare(&current, &golden).unwrap();

    assert!(result.is_regression);
    assert!(result.summary.contains("REGRESSION"));
}

/// Test regressions list
#[test]
fn test_regressions_list() {
    let golden = GoldenTrace::new(
        "baseline",
        TraceMetrics::new()
            .total_time_us(1000.0)
            .p50_latency_us(50.0)
            .throughput(10000.0),
    );

    let current = TraceMetrics::new()
        .total_time_us(1200.0) // 20% regression
        .p50_latency_us(65.0)  // 30% regression
        .throughput(7000.0);   // 30% regression

    let comparator = GoldenComparator::new().with_threshold(10.0);
    let result = comparator.compare(&current, &golden).unwrap();

    let regressions = result.regressions();
    assert!(!regressions.is_empty());
    assert!(regressions.iter().any(|(name, _)| name == "total_time"));
    assert!(regressions.iter().any(|(name, _)| name == "p50_latency"));
}

/// Test custom metrics
#[test]
fn test_custom_metrics() {
    let metrics = TraceMetrics::new()
        .total_time_us(1000.0)
        .with_custom("gpu_utilization", 95.5)
        .with_custom("memory_bandwidth", 800.0);

    let trace = GoldenTrace::new("custom_test", metrics);

    assert_eq!(trace.metrics.custom.get("gpu_utilization"), Some(&95.5));
    assert_eq!(trace.metrics.custom.get("memory_bandwidth"), Some(&800.0));
}

/// Test environment info
#[test]
fn test_environment_info() {
    let trace = GoldenTrace::new("env_test", TraceMetrics::new())
        .with_env("os", "linux")
        .with_env("cpu", "x86_64");

    assert_eq!(trace.environment.get("os"), Some(&"linux".to_string()));
    assert_eq!(trace.environment.get("cpu"), Some(&"x86_64".to_string()));
}

/// Test metrics validity
#[test]
fn test_metrics_validity() {
    let valid = TraceMetrics::new().total_time_us(1000.0);
    assert!(valid.is_valid());

    let mut invalid = TraceMetrics::new();
    invalid.total_time_us = f64::NAN;
    assert!(!invalid.is_valid());

    let mut negative = TraceMetrics::new();
    negative.total_time_us = -1.0;
    assert!(!negative.is_valid());
}

/// Test invalid metrics comparison
#[test]
fn test_invalid_metrics_comparison() {
    let golden = GoldenTrace::new("baseline", TraceMetrics::new().total_time_us(1000.0));

    let mut invalid = TraceMetrics::new();
    invalid.total_time_us = f64::NAN;

    let comparator = GoldenComparator::new();
    let result = comparator.compare(&invalid, &golden);

    assert!(matches!(result, Err(GoldenTraceError::InvalidTrace(_))));
}

/// Test syscall threshold detection
#[test]
fn test_syscall_threshold_detection() {
    let baseline = SyscallBreakdown {
        read_count: 100,
        write_count: 100,
        mmap_count: 10,
        futex_count: 20,
        other_count: 10,
    };

    let current = SyscallBreakdown {
        read_count: 130, // 30% increase
        write_count: 100,
        mmap_count: 10,
        futex_count: 20,
        other_count: 10,
    };

    let delta = current.percentage_diff(&baseline);

    assert!(delta.exceeds_threshold(20.0));
    assert!(!delta.exceeds_threshold(50.0));
}
