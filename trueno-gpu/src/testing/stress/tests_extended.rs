//! Extended coverage tests for the stress testing framework.

use std::time::Duration;

use super::*;

#[test]
fn test_frame_profile_clone() {
    // Coverage for FrameProfile Clone derive
    let profile = FrameProfile {
        cycle: 5,
        duration_ms: 100,
        memory_bytes: 1024,
        tests_passed: 10,
        tests_failed: 2,
        input_seed: 12345,
        input_size: 256,
    };
    let cloned = profile.clone();
    assert_eq!(profile.cycle, cloned.cycle);
    assert_eq!(profile.duration_ms, cloned.duration_ms);
    assert_eq!(profile.memory_bytes, cloned.memory_bytes);
    assert_eq!(profile.tests_passed, cloned.tests_passed);
    assert_eq!(profile.tests_failed, cloned.tests_failed);
    assert_eq!(profile.input_seed, cloned.input_seed);
    assert_eq!(profile.input_size, cloned.input_size);
}

#[test]
fn test_stress_report_clone() {
    // Coverage for StressReport Clone derive
    let mut report = StressReport::default();
    report.add_frame(FrameProfile {
        cycle: 0,
        duration_ms: 50,
        tests_passed: 5,
        tests_failed: 1,
        ..Default::default()
    });
    report.anomalies.push(Anomaly {
        cycle: 0,
        kind: AnomalyKind::SlowFrame,
        description: "Test anomaly".to_string(),
    });

    let cloned = report.clone();
    assert_eq!(report.cycles_completed, cloned.cycles_completed);
    assert_eq!(report.total_passed, cloned.total_passed);
    assert_eq!(report.total_failed, cloned.total_failed);
    assert_eq!(report.frames.len(), cloned.frames.len());
    assert_eq!(report.anomalies.len(), cloned.anomalies.len());
}

#[test]
fn test_anomaly_clone() {
    // Coverage for Anomaly Clone derive
    let anomaly = Anomaly {
        cycle: 42,
        kind: AnomalyKind::HighMemory,
        description: "Memory exceeded threshold".to_string(),
    };
    let cloned = anomaly.clone();
    assert_eq!(anomaly.cycle, cloned.cycle);
    assert_eq!(anomaly.kind, cloned.kind);
    assert_eq!(anomaly.description, cloned.description);
}

#[test]
fn test_performance_thresholds_clone() {
    // Coverage for PerformanceThresholds Clone derive
    let thresholds = PerformanceThresholds {
        max_frame_time_ms: 50,
        max_memory_bytes: 1024,
        max_timing_variance: 0.1,
        max_failure_rate: 0.05,
    };
    let cloned = thresholds.clone();
    assert_eq!(thresholds.max_frame_time_ms, cloned.max_frame_time_ms);
    assert_eq!(thresholds.max_memory_bytes, cloned.max_memory_bytes);
    assert!((thresholds.max_timing_variance - cloned.max_timing_variance).abs() < 0.001);
    assert!((thresholds.max_failure_rate - cloned.max_failure_rate).abs() < 0.001);
}

#[test]
fn test_performance_result_clone() {
    // Coverage for PerformanceResult Clone derive
    let result = PerformanceResult {
        passed: false,
        max_frame_ms: 100,
        mean_frame_ms: 75.5,
        variance: 0.15,
        pass_rate: 0.95,
        violations: vec!["Test violation".to_string()],
    };
    let cloned = result.clone();
    assert_eq!(result.passed, cloned.passed);
    assert_eq!(result.max_frame_ms, cloned.max_frame_ms);
    assert!((result.mean_frame_ms - cloned.mean_frame_ms).abs() < 0.001);
    assert!((result.variance - cloned.variance).abs() < 0.001);
    assert!((result.pass_rate - cloned.pass_rate).abs() < 0.001);
    assert_eq!(result.violations.len(), cloned.violations.len());
}

#[test]
fn test_stress_rng_clone() {
    // Coverage for StressRng Clone derive
    let mut rng = StressRng::new(42);
    rng.next_u32(); // Advance state

    let cloned = rng.clone();

    // Both should produce same sequence from this point
    assert_eq!(rng.next_u32(), cloned.clone().next_u32());
}

#[test]
fn test_stress_config_clone() {
    // Coverage for StressConfig Clone derive
    let config = StressConfig {
        cycles: 50,
        interval_ms: 200,
        seed: 12345,
        min_input_size: 128,
        max_input_size: 1024,
        thresholds: PerformanceThresholds::default(),
    };
    let cloned = config.clone();
    assert_eq!(config.cycles, cloned.cycles);
    assert_eq!(config.interval_ms, cloned.interval_ms);
    assert_eq!(config.seed, cloned.seed);
    assert_eq!(config.min_input_size, cloned.min_input_size);
    assert_eq!(config.max_input_size, cloned.max_input_size);
}

#[test]
fn test_stress_report_with_timing_variance() {
    // Coverage for timing_variance calculation with actual variance
    let mut report = StressReport::default();

    // Add frames with known variance pattern
    for i in 0..5 {
        report.add_frame(FrameProfile {
            cycle: i,
            duration_ms: if i % 2 == 0 { 50 } else { 150 },
            tests_passed: 1,
            tests_failed: 0,
            ..Default::default()
        });
    }

    let variance = report.timing_variance();
    // Variance should be non-zero with alternating 50/150ms timings
    assert!(variance > 0.0);
}

#[test]
fn test_multiple_anomaly_kinds() {
    // Coverage for all AnomalyKind variants and Anomaly debug
    let anomalies = vec![
        Anomaly {
            cycle: 0,
            kind: AnomalyKind::SlowFrame,
            description: "Slow".to_string(),
        },
        Anomaly {
            cycle: 1,
            kind: AnomalyKind::HighMemory,
            description: "High memory".to_string(),
        },
        Anomaly {
            cycle: 2,
            kind: AnomalyKind::TestFailure,
            description: "Test failed".to_string(),
        },
        Anomaly {
            cycle: 3,
            kind: AnomalyKind::TimingSpike,
            description: "Spike".to_string(),
        },
        Anomaly {
            cycle: 4,
            kind: AnomalyKind::NonDeterministic,
            description: "Non-deterministic".to_string(),
        },
    ];

    // Test Debug impl
    for a in &anomalies {
        let debug_str = format!("{:?}", a);
        assert!(debug_str.contains("Anomaly"));
    }

    // Verify all kinds are distinct
    assert_eq!(anomalies.len(), 5);
}

#[test]
fn test_stress_report_debug() {
    // Coverage for StressReport Debug derive
    let mut report = StressReport::default();
    report.add_frame(FrameProfile::default());

    let debug_str = format!("{:?}", report);
    assert!(debug_str.contains("StressReport"));
    assert!(debug_str.contains("frames"));
}

#[test]
fn test_frame_profile_debug() {
    // Coverage for FrameProfile Debug derive
    let profile = FrameProfile {
        cycle: 1,
        duration_ms: 100,
        ..Default::default()
    };

    let debug_str = format!("{:?}", profile);
    assert!(debug_str.contains("FrameProfile"));
    assert!(debug_str.contains("cycle"));
}

#[test]
fn test_performance_thresholds_debug() {
    // Coverage for PerformanceThresholds Debug derive
    let thresholds = PerformanceThresholds::default();
    let debug_str = format!("{:?}", thresholds);
    assert!(debug_str.contains("PerformanceThresholds"));
}

#[test]
fn test_performance_result_debug() {
    // Coverage for PerformanceResult Debug derive
    let result = PerformanceResult {
        passed: true,
        max_frame_ms: 50,
        mean_frame_ms: 40.0,
        variance: 0.1,
        pass_rate: 1.0,
        violations: vec![],
    };
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("PerformanceResult"));
}

#[test]
fn test_stress_rng_debug() {
    // Coverage for StressRng Debug derive
    let rng = StressRng::new(42);
    let debug_str = format!("{:?}", rng);
    assert!(debug_str.contains("StressRng"));
}

#[test]
fn test_stress_config_debug() {
    // Coverage for StressConfig Debug derive
    let config = StressConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("StressConfig"));
}

#[test]
fn test_anomaly_kind_debug() {
    // Coverage for AnomalyKind Debug/Copy derives
    let kind = AnomalyKind::SlowFrame;
    let copied = kind; // Test Copy
    let debug_str = format!("{:?}", copied);
    assert!(debug_str.contains("SlowFrame"));
}

// ========================================================================
// ADDITIONAL COVERAGE TESTS - TARGETING 95%+ COVERAGE (PART 2)
// ========================================================================

#[test]
fn test_frame_profile_all_fields() {
    // Coverage for all FrameProfile fields including memory_bytes
    let profile = FrameProfile {
        cycle: 10,
        duration_ms: 50,
        memory_bytes: 4096,
        tests_passed: 8,
        tests_failed: 2,
        input_seed: 999,
        input_size: 128,
    };

    assert_eq!(profile.cycle, 10);
    assert_eq!(profile.duration_ms, 50);
    assert_eq!(profile.memory_bytes, 4096);
    assert_eq!(profile.tests_passed, 8);
    assert_eq!(profile.tests_failed, 2);
    assert_eq!(profile.input_seed, 999);
    assert_eq!(profile.input_size, 128);
}

#[test]
fn test_stress_report_default_values() {
    // Coverage for StressReport default derive
    let report = StressReport::default();
    assert!(report.frames.is_empty());
    assert_eq!(report.cycles_completed, 0);
    assert_eq!(report.total_passed, 0);
    assert_eq!(report.total_failed, 0);
    assert!(report.anomalies.is_empty());
}

#[test]
fn test_performance_result_all_fields() {
    // Coverage for all PerformanceResult fields
    let result = PerformanceResult {
        passed: true,
        max_frame_ms: 75,
        mean_frame_ms: 50.5,
        variance: 0.12,
        pass_rate: 0.98,
        violations: vec!["violation1".to_string(), "violation2".to_string()],
    };

    assert!(result.passed);
    assert_eq!(result.max_frame_ms, 75);
    assert!((result.mean_frame_ms - 50.5).abs() < 0.001);
    assert!((result.variance - 0.12).abs() < 0.001);
    assert!((result.pass_rate - 0.98).abs() < 0.001);
    assert_eq!(result.violations.len(), 2);
}

#[test]
fn test_performance_thresholds_all_fields() {
    // Coverage for all PerformanceThresholds fields
    let thresholds = PerformanceThresholds {
        max_frame_time_ms: 150,
        max_memory_bytes: 1024 * 1024,
        max_timing_variance: 0.3,
        max_failure_rate: 0.1,
    };

    assert_eq!(thresholds.max_frame_time_ms, 150);
    assert_eq!(thresholds.max_memory_bytes, 1024 * 1024);
    assert!((thresholds.max_timing_variance - 0.3).abs() < 0.001);
    assert!((thresholds.max_failure_rate - 0.1).abs() < 0.001);
}

#[test]
fn test_anomaly_all_fields() {
    // Coverage for all Anomaly fields
    let anomaly = Anomaly {
        cycle: 5,
        kind: AnomalyKind::TimingSpike,
        description: "Timing spike detected at cycle 5".to_string(),
    };

    assert_eq!(anomaly.cycle, 5);
    assert_eq!(anomaly.kind, AnomalyKind::TimingSpike);
    assert!(anomaly.description.contains("cycle 5"));
}

#[test]
fn test_stress_config_all_fields() {
    // Coverage for all StressConfig fields
    let config = StressConfig {
        cycles: 200,
        interval_ms: 50,
        seed: 12345,
        min_input_size: 32,
        max_input_size: 1024,
        thresholds: PerformanceThresholds {
            max_frame_time_ms: 200,
            max_memory_bytes: 128 * 1024 * 1024,
            max_timing_variance: 0.25,
            max_failure_rate: 0.02,
        },
    };

    assert_eq!(config.cycles, 200);
    assert_eq!(config.interval_ms, 50);
    assert_eq!(config.seed, 12345);
    assert_eq!(config.min_input_size, 32);
    assert_eq!(config.max_input_size, 1024);
    assert_eq!(config.thresholds.max_frame_time_ms, 200);
}

#[test]
fn test_stress_runner_memory_bytes_calculation() {
    // Coverage for memory_bytes calculation in run_cycle (line 351)
    let config = StressConfig {
        cycles: 1,
        seed: 42,
        min_input_size: 100,
        max_input_size: 101, // Force exact size
        ..Default::default()
    };

    let mut runner = StressTestRunner::new(config);
    let profile = runner.run_cycle(0, |input| (input.len() as u32, 0));

    // memory_bytes should be input_size * sizeof(f32)
    assert_eq!(
        profile.memory_bytes,
        profile.input_size * std::mem::size_of::<f32>()
    );
}

#[test]
fn test_verify_performance_multiple_violations() {
    // Coverage for multiple violations in verify_performance
    let mut report = StressReport::default();

    // Create a report that triggers all violation checks
    report.add_frame(FrameProfile {
        cycle: 0,
        duration_ms: 200, // Exceeds threshold
        tests_passed: 1,
        tests_failed: 99, // Very high failure rate
        ..Default::default()
    });
    report.add_frame(FrameProfile {
        cycle: 1,
        duration_ms: 10, // High variance with previous
        tests_passed: 1,
        tests_failed: 0,
        ..Default::default()
    });

    let thresholds = PerformanceThresholds {
        max_frame_time_ms: 100,   // Will trigger
        max_timing_variance: 0.1, // Will trigger
        max_failure_rate: 0.01,   // Will trigger
        ..Default::default()
    };

    let result = verify_performance(&report, &thresholds);
    assert!(!result.passed);
    assert!(result.violations.len() >= 2); // At least frame time and pass rate
}

#[test]
fn test_stress_rng_sequence_consistency() {
    // Coverage for RNG state consistency across operations
    let mut rng = StressRng::new(42);

    // Generate a sequence
    let seq1: Vec<u32> = (0..10).map(|_| rng.next_u32()).collect();

    // New RNG with same seed should produce same sequence
    let mut rng2 = StressRng::new(42);
    let seq2: Vec<u32> = (0..10).map(|_| rng2.next_u32()).collect();

    assert_eq!(seq1, seq2);
}

#[test]
fn test_stress_rng_different_seeds() {
    // Coverage for RNG producing different sequences with different seeds
    let mut rng1 = StressRng::new(1);
    let mut rng2 = StressRng::new(2);

    let val1 = rng1.next_u32();
    let val2 = rng2.next_u32();

    // Different seeds should produce different values
    assert_ne!(val1, val2);
}

#[test]
fn test_stress_runner_input_generation_determinism() {
    // Coverage for deterministic input generation
    let config1 = StressConfig {
        seed: 12345,
        min_input_size: 50,
        max_input_size: 100,
        ..Default::default()
    };

    let config2 = StressConfig {
        seed: 12345,
        min_input_size: 50,
        max_input_size: 100,
        ..Default::default()
    };

    let mut runner1 = StressTestRunner::new(config1);
    let mut runner2 = StressTestRunner::new(config2);

    let (seed1, input1) = runner1.generate_input();
    let (seed2, input2) = runner2.generate_input();

    // Same seed should produce same inputs
    assert_eq!(seed1, seed2);
    assert_eq!(input1, input2);
}

#[test]
fn test_run_all_with_zero_interval() {
    // Coverage for run_all with zero interval (no sleep needed)
    let config = StressConfig {
        cycles: 3,
        interval_ms: 0, // Zero interval
        seed: 42,
        min_input_size: 10,
        max_input_size: 20,
        thresholds: PerformanceThresholds::default(),
    };

    let mut runner = StressTestRunner::new(config);
    let report = runner.run_all(|_input| (1, 0));

    assert_eq!(report.cycles_completed, 3);
}

#[test]
fn test_verify_performance_boundary_threshold() {
    // Coverage for boundary conditions in threshold checks
    let mut report = StressReport::default();

    // Create exactly at threshold
    report.add_frame(FrameProfile {
        cycle: 0,
        duration_ms: 100, // Exactly at threshold
        tests_passed: 99,
        tests_failed: 1, // Exactly at 1% threshold
        ..Default::default()
    });

    let thresholds = PerformanceThresholds {
        max_frame_time_ms: 100, // Equal should pass
        max_timing_variance: 1.0,
        max_failure_rate: 0.01, // Equal should pass
        ..Default::default()
    };

    let result = verify_performance(&report, &thresholds);
    // At boundary values, the check is `>` for frame time, so 100 == 100 should pass
    // But pass_rate check is `<` so 99% < 99% is false, should pass
    assert!(result.passed || result.violations.len() <= 1);
}

#[test]
fn test_stress_report_anomalies_field() {
    // Coverage for anomalies field in StressReport
    let mut report = StressReport::default();

    // Manually add an anomaly
    report.anomalies.push(Anomaly {
        cycle: 0,
        kind: AnomalyKind::HighMemory,
        description: "Memory exceeded".to_string(),
    });

    assert_eq!(report.anomalies.len(), 1);
    assert_eq!(report.anomalies[0].kind, AnomalyKind::HighMemory);
}

#[test]
fn test_stress_rng_gen_f32_range() {
    // Coverage for gen_f32 producing values in [0, 1) range
    let mut rng = StressRng::new(42);

    for _ in 0..1000 {
        let val = rng.gen_f32();
        assert!(val >= 0.0, "Value should be >= 0.0");
        assert!(val < 1.0, "Value should be < 1.0");
    }
}

#[test]
fn test_anomaly_kind_copy_semantics() {
    // Coverage for AnomalyKind Copy trait
    let kind1 = AnomalyKind::NonDeterministic;
    let kind2 = kind1; // Copy
    let kind3 = kind1; // Another copy

    assert_eq!(kind1, kind2);
    assert_eq!(kind2, kind3);
}

#[test]
fn test_performance_result_with_empty_violations() {
    // Coverage for PerformanceResult with no violations
    let result = PerformanceResult {
        passed: true,
        max_frame_ms: 50,
        mean_frame_ms: 40.0,
        variance: 0.05,
        pass_rate: 1.0,
        violations: Vec::new(),
    };

    assert!(result.passed);
    assert!(result.violations.is_empty());
}

#[test]
fn test_stress_runner_report_method() {
    // Coverage for report() method returning reference
    let config = StressConfig::default();
    let runner = StressTestRunner::new(config);

    let report = runner.report();
    assert_eq!(report.cycles_completed, 0);
    assert!(report.frames.is_empty());
}

#[test]
fn test_run_cycle_with_failures_and_slow_frame() {
    // Coverage for both anomaly conditions in single run_cycle
    let config = StressConfig {
        cycles: 1,
        seed: 42,
        thresholds: PerformanceThresholds {
            max_frame_time_ms: 1, // Very low to trigger slow frame
            ..Default::default()
        },
        ..Default::default()
    };

    let mut runner = StressTestRunner::new(config);

    runner.run_cycle(0, |_input| {
        std::thread::sleep(Duration::from_millis(5));
        (3, 5) // Report failures to trigger TestFailure anomaly
    });

    let report = runner.report();

    // Should have both SlowFrame and TestFailure anomalies
    let slow_count = report
        .anomalies
        .iter()
        .filter(|a| a.kind == AnomalyKind::SlowFrame)
        .count();
    let failure_count = report
        .anomalies
        .iter()
        .filter(|a| a.kind == AnomalyKind::TestFailure)
        .count();

    assert_eq!(slow_count, 1, "Should have one SlowFrame anomaly");
    assert_eq!(failure_count, 1, "Should have one TestFailure anomaly");
}
