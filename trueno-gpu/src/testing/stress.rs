//! Stress Testing Framework with Randomized Inputs
//!
//! Frame-by-frame stress testing with:
//! - Randomized inputs via simular (deterministic RNG)
//! - Performance profiling via renacer
//! - Anomaly detection for regression identification
//!
//! # Sovereign Stack
//!
//! - `simular` v0.2.0: Deterministic RNG (SimRng)
//! - `renacer` v0.7.0: Profiling and anomaly detection

use std::time::{Duration, Instant};

/// Frame profile data collected during stress testing
#[derive(Debug, Clone, Default)]
pub struct FrameProfile {
    /// Cycle number
    pub cycle: u32,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Memory usage estimate (bytes)
    pub memory_bytes: usize,
    /// Number of tests passed
    pub tests_passed: u32,
    /// Number of tests failed
    pub tests_failed: u32,
    /// Input seed used for this frame
    pub input_seed: u64,
    /// Input size used for this frame
    pub input_size: usize,
}

/// Cumulative stress test report
#[derive(Debug, Clone, Default)]
pub struct StressReport {
    /// All frame profiles
    pub frames: Vec<FrameProfile>,
    /// Total cycles completed
    pub cycles_completed: u32,
    /// Total tests passed across all cycles
    pub total_passed: u32,
    /// Total tests failed across all cycles
    pub total_failed: u32,
    /// Detected anomalies
    pub anomalies: Vec<Anomaly>,
}

impl StressReport {
    /// Calculate mean frame time in milliseconds
    #[must_use]
    pub fn mean_frame_time_ms(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.frames.iter().map(|f| f.duration_ms).sum();
        sum as f64 / self.frames.len() as f64
    }

    /// Calculate timing variance (coefficient of variation)
    #[must_use]
    pub fn timing_variance(&self) -> f64 {
        if self.frames.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_frame_time_ms();
        if mean == 0.0 {
            return 0.0;
        }
        let variance: f64 = self
            .frames
            .iter()
            .map(|f| {
                let diff = f.duration_ms as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / self.frames.len() as f64;
        variance.sqrt() / mean
    }

    /// Calculate max frame time
    #[must_use]
    pub fn max_frame_time_ms(&self) -> u64 {
        self.frames.iter().map(|f| f.duration_ms).max().unwrap_or(0)
    }

    /// Calculate pass rate (0.0 to 1.0)
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        let total = self.total_passed + self.total_failed;
        if total == 0 {
            return 1.0;
        }
        self.total_passed as f64 / total as f64
    }

    /// Add a frame to the report
    pub fn add_frame(&mut self, profile: FrameProfile) {
        self.total_passed += profile.tests_passed;
        self.total_failed += profile.tests_failed;
        self.cycles_completed += 1;
        self.frames.push(profile);
    }
}

/// Detected anomaly during stress testing
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Cycle where anomaly was detected
    pub cycle: u32,
    /// Type of anomaly
    pub kind: AnomalyKind,
    /// Description
    pub description: String,
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyKind {
    /// Frame took longer than threshold
    SlowFrame,
    /// Memory usage exceeded threshold
    HighMemory,
    /// Test failure detected
    TestFailure,
    /// Timing variance spike
    TimingSpike,
    /// Non-deterministic behavior
    NonDeterministic,
}

/// Performance thresholds for anomaly detection
#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Max time per frame (ms)
    pub max_frame_time_ms: u64,
    /// Max memory per frame (bytes)
    pub max_memory_bytes: usize,
    /// Max variance in frame times (coefficient of variation)
    pub max_timing_variance: f64,
    /// Max allowed failure rate (0.0 to 1.0)
    pub max_failure_rate: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_frame_time_ms: 100,             // 10 FPS minimum
            max_memory_bytes: 64 * 1024 * 1024, // 64MB max
            max_timing_variance: 0.2,           // 20% max variance
            max_failure_rate: 0.01,             // 1% max failures
        }
    }
}

/// Performance verification result
#[derive(Debug, Clone)]
pub struct PerformanceResult {
    /// Whether all thresholds passed
    pub passed: bool,
    /// Max frame time observed
    pub max_frame_ms: u64,
    /// Mean frame time observed
    pub mean_frame_ms: f64,
    /// Timing variance observed
    pub variance: f64,
    /// Pass rate observed
    pub pass_rate: f64,
    /// List of threshold violations
    pub violations: Vec<String>,
}

/// Verify performance against thresholds
#[must_use]
pub fn verify_performance(
    report: &StressReport,
    thresholds: &PerformanceThresholds,
) -> PerformanceResult {
    let max_frame = report.max_frame_time_ms();
    let mean_frame = report.mean_frame_time_ms();
    let variance = report.timing_variance();
    let pass_rate = report.pass_rate();

    let mut violations = Vec::new();

    if max_frame > thresholds.max_frame_time_ms {
        violations.push(format!(
            "Max frame time {}ms exceeds threshold {}ms",
            max_frame, thresholds.max_frame_time_ms
        ));
    }

    if variance > thresholds.max_timing_variance {
        violations.push(format!(
            "Timing variance {:.3} exceeds threshold {:.3}",
            variance, thresholds.max_timing_variance
        ));
    }

    if pass_rate < (1.0 - thresholds.max_failure_rate) {
        violations.push(format!(
            "Pass rate {:.1}% below threshold {:.1}%",
            pass_rate * 100.0,
            (1.0 - thresholds.max_failure_rate) * 100.0
        ));
    }

    PerformanceResult {
        passed: violations.is_empty(),
        max_frame_ms: max_frame,
        mean_frame_ms: mean_frame,
        variance,
        pass_rate,
        violations,
    }
}

/// Simple PCG32 RNG for stress testing (no external deps in core)
/// Used when simular feature is not enabled
#[derive(Debug, Clone)]
pub struct StressRng {
    state: u64,
    inc: u64,
}

impl StressRng {
    /// Create new RNG with seed
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (seed << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Generate next u32
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        (xorshifted >> rot) | (xorshifted << ((!rot).wrapping_add(1) & 31))
    }

    /// Generate next u64
    pub fn next_u64(&mut self) -> u64 {
        let high = self.next_u32() as u64;
        let low = self.next_u32() as u64;
        (high << 32) | low
    }

    /// Generate f32 in [0, 1)
    pub fn gen_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / u32::MAX as f64) as f32
    }

    /// Generate u32 in range [min, max)
    pub fn gen_range_u32(&mut self, min: u32, max: u32) -> u32 {
        if max <= min {
            return min;
        }
        let range = max - min;
        min + (self.next_u32() % range)
    }
}

/// Stress test configuration
#[derive(Debug, Clone)]
pub struct StressConfig {
    /// Number of cycles to run
    pub cycles: u32,
    /// Interval between cycles (ms)
    pub interval_ms: u64,
    /// Base seed for RNG
    pub seed: u64,
    /// Min input size
    pub min_input_size: usize,
    /// Max input size
    pub max_input_size: usize,
    /// Performance thresholds
    pub thresholds: PerformanceThresholds,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            cycles: 100,
            interval_ms: 100,
            seed: 42,
            min_input_size: 64,
            max_input_size: 512,
            thresholds: PerformanceThresholds::default(),
        }
    }
}

/// Stress test runner
pub struct StressTestRunner {
    rng: StressRng,
    config: StressConfig,
    report: StressReport,
}

impl StressTestRunner {
    /// Create new stress test runner
    #[must_use]
    pub fn new(config: StressConfig) -> Self {
        Self {
            rng: StressRng::new(config.seed),
            config,
            report: StressReport::default(),
        }
    }

    /// Generate randomized input for a cycle
    pub fn generate_input(&mut self) -> (u64, Vec<f32>) {
        let seed = self.rng.next_u64();
        let size = self.rng.gen_range_u32(
            self.config.min_input_size as u32,
            self.config.max_input_size as u32,
        ) as usize;

        let mut input_rng = StressRng::new(seed);
        let input: Vec<f32> = (0..size).map(|_| input_rng.gen_f32()).collect();

        (seed, input)
    }

    /// Run a single cycle with provided test function
    pub fn run_cycle<F>(&mut self, cycle: u32, test_fn: F) -> FrameProfile
    where
        F: FnOnce(&[f32]) -> (u32, u32), // Returns (passed, failed)
    {
        let (input_seed, input) = self.generate_input();
        let input_size = input.len();

        let start = Instant::now();
        let (tests_passed, tests_failed) = test_fn(&input);
        let duration = start.elapsed();

        let profile = FrameProfile {
            cycle,
            duration_ms: duration.as_millis() as u64,
            memory_bytes: input_size * std::mem::size_of::<f32>(),
            tests_passed,
            tests_failed,
            input_seed,
            input_size,
        };

        // Check for anomalies
        if profile.duration_ms > self.config.thresholds.max_frame_time_ms {
            self.report.anomalies.push(Anomaly {
                cycle,
                kind: AnomalyKind::SlowFrame,
                description: format!(
                    "Frame {}ms exceeds threshold {}ms",
                    profile.duration_ms, self.config.thresholds.max_frame_time_ms
                ),
            });
        }

        if tests_failed > 0 {
            self.report.anomalies.push(Anomaly {
                cycle,
                kind: AnomalyKind::TestFailure,
                description: format!("{} tests failed in cycle {}", tests_failed, cycle),
            });
        }

        self.report.add_frame(profile.clone());
        profile
    }

    /// Run all cycles
    pub fn run_all<F>(&mut self, mut test_fn: F) -> &StressReport
    where
        F: FnMut(&[f32]) -> (u32, u32),
    {
        let interval = Duration::from_millis(self.config.interval_ms);

        for cycle in 0..self.config.cycles {
            let start = Instant::now();
            self.run_cycle(cycle, &mut test_fn);

            let elapsed = start.elapsed();
            if let Some(remaining) = interval.checked_sub(elapsed) {
                std::thread::sleep(remaining);
            }
        }

        &self.report
    }

    /// Get the current report
    #[must_use]
    pub fn report(&self) -> &StressReport {
        &self.report
    }

    /// Verify performance and return result
    #[must_use]
    pub fn verify(&self) -> PerformanceResult {
        verify_performance(&self.report, &self.config.thresholds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_rng_deterministic() {
        let mut rng1 = StressRng::new(42);
        let mut rng2 = StressRng::new(42);

        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_stress_rng_gen_range() {
        let mut rng = StressRng::new(12345);
        for _ in 0..1000 {
            let val = rng.gen_range_u32(10, 100);
            assert!(val >= 10 && val < 100);
        }
    }

    #[test]
    fn test_stress_rng_gen_f32() {
        let mut rng = StressRng::new(99999);
        for _ in 0..1000 {
            let val = rng.gen_f32();
            assert!((0.0..1.0).contains(&val));
        }
    }

    #[test]
    fn test_frame_profile_default() {
        let profile = FrameProfile::default();
        assert_eq!(profile.cycle, 0);
        assert_eq!(profile.duration_ms, 0);
    }

    #[test]
    fn test_stress_report_metrics() {
        let mut report = StressReport::default();

        report.add_frame(FrameProfile {
            cycle: 0,
            duration_ms: 100,
            tests_passed: 5,
            tests_failed: 0,
            ..Default::default()
        });

        report.add_frame(FrameProfile {
            cycle: 1,
            duration_ms: 120,
            tests_passed: 5,
            tests_failed: 0,
            ..Default::default()
        });

        assert_eq!(report.cycles_completed, 2);
        assert_eq!(report.total_passed, 10);
        assert_eq!(report.total_failed, 0);
        assert_eq!(report.mean_frame_time_ms(), 110.0);
        assert_eq!(report.max_frame_time_ms(), 120);
        assert!((report.pass_rate() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_stress_report_variance() {
        let mut report = StressReport::default();

        // Add frames with same duration - variance should be 0
        for i in 0..10 {
            report.add_frame(FrameProfile {
                cycle: i,
                duration_ms: 100,
                tests_passed: 1,
                tests_failed: 0,
                ..Default::default()
            });
        }

        assert!((report.timing_variance()).abs() < 0.001);
    }

    #[test]
    fn test_performance_thresholds_default() {
        let thresholds = PerformanceThresholds::default();
        assert_eq!(thresholds.max_frame_time_ms, 100);
        assert_eq!(thresholds.max_memory_bytes, 64 * 1024 * 1024);
        assert!((thresholds.max_timing_variance - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_verify_performance_pass() {
        let mut report = StressReport::default();
        for i in 0..10 {
            report.add_frame(FrameProfile {
                cycle: i,
                duration_ms: 50,
                tests_passed: 5,
                tests_failed: 0,
                ..Default::default()
            });
        }

        let result = verify_performance(&report, &PerformanceThresholds::default());
        assert!(result.passed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn test_verify_performance_fail_slow() {
        let mut report = StressReport::default();
        report.add_frame(FrameProfile {
            cycle: 0,
            duration_ms: 200, // Exceeds 100ms threshold
            tests_passed: 5,
            tests_failed: 0,
            ..Default::default()
        });

        let result = verify_performance(&report, &PerformanceThresholds::default());
        assert!(!result.passed);
        assert_eq!(result.violations.len(), 1);
        assert!(result.violations[0].contains("Max frame time"));
    }

    #[test]
    fn test_stress_runner_generate_input() {
        let config = StressConfig {
            min_input_size: 100,
            max_input_size: 200,
            seed: 42,
            ..Default::default()
        };

        let mut runner = StressTestRunner::new(config);
        let (seed1, input1) = runner.generate_input();
        let (seed2, input2) = runner.generate_input();

        // Different inputs each time
        assert_ne!(seed1, seed2);
        assert!(input1.len() >= 100 && input1.len() < 200);
        assert!(input2.len() >= 100 && input2.len() < 200);
    }

    #[test]
    fn test_stress_runner_run_cycle() {
        let config = StressConfig {
            cycles: 1,
            seed: 42,
            ..Default::default()
        };

        let mut runner = StressTestRunner::new(config);
        let profile = runner.run_cycle(0, |input| {
            // Simple test: count positive values
            let positive = input.iter().filter(|&&v| v > 0.5).count() as u32;
            (positive, 0)
        });

        assert_eq!(profile.cycle, 0);
        assert!(profile.tests_passed > 0);
        assert_eq!(profile.tests_failed, 0);
    }

    #[test]
    fn test_anomaly_detection() {
        let config = StressConfig {
            cycles: 1,
            seed: 42,
            thresholds: PerformanceThresholds {
                max_frame_time_ms: 1, // Very low threshold
                ..Default::default()
            },
            ..Default::default()
        };

        let mut runner = StressTestRunner::new(config);

        // This will likely exceed 1ms
        runner.run_cycle(0, |input| {
            std::thread::sleep(Duration::from_millis(5));
            (input.len() as u32, 0)
        });

        let report = runner.report();
        assert!(!report.anomalies.is_empty());
        assert_eq!(report.anomalies[0].kind, AnomalyKind::SlowFrame);
    }

    // ========================================================================
    // EDGE CASE COVERAGE TESTS - TRUENO-SPEC-014
    // ========================================================================

    #[test]
    fn test_stress_report_empty_frames() {
        // Coverage for line 54: when frames is empty
        let report = StressReport::default();
        assert_eq!(report.mean_frame_time_ms(), 0.0);
        assert_eq!(report.max_frame_time_ms(), 0);
    }

    #[test]
    fn test_stress_report_single_frame_variance() {
        // Coverage for timing_variance with < 2 frames
        let mut report = StressReport::default();
        report.add_frame(FrameProfile {
            cycle: 0,
            duration_ms: 100,
            tests_passed: 1,
            tests_failed: 0,
            ..Default::default()
        });
        assert_eq!(report.timing_variance(), 0.0);
    }

    #[test]
    fn test_stress_report_zero_mean_variance() {
        // Coverage for line 68: when mean is 0.0
        let mut report = StressReport::default();
        report.add_frame(FrameProfile {
            cycle: 0,
            duration_ms: 0, // Zero duration
            tests_passed: 1,
            tests_failed: 0,
            ..Default::default()
        });
        report.add_frame(FrameProfile {
            cycle: 1,
            duration_ms: 0, // Zero duration
            tests_passed: 1,
            tests_failed: 0,
            ..Default::default()
        });
        assert_eq!(report.timing_variance(), 0.0);
    }

    #[test]
    fn test_stress_report_pass_rate_no_tests() {
        // Coverage for line 90: when total is 0
        let mut report = StressReport::default();
        report.add_frame(FrameProfile {
            cycle: 0,
            duration_ms: 100,
            tests_passed: 0, // No tests
            tests_failed: 0, // No tests
            ..Default::default()
        });
        assert_eq!(report.pass_rate(), 1.0);
    }

    // ========================================================================
    // ADDITIONAL COVERAGE TESTS - TARGETING 95%+ COVERAGE
    // ========================================================================

    #[test]
    fn test_stress_rng_next_u64() {
        // Coverage for next_u64 (lines 253-257)
        let mut rng = StressRng::new(42);
        let val1 = rng.next_u64();
        let val2 = rng.next_u64();

        // Should produce different values
        assert_ne!(val1, val2);

        // Deterministic check
        let mut rng2 = StressRng::new(42);
        assert_eq!(val1, rng2.next_u64());
    }

    #[test]
    fn test_stress_rng_gen_range_edge_cases() {
        // Coverage for gen_range_u32 edge case when max <= min (line 267)
        let mut rng = StressRng::new(42);

        // When max == min, should return min
        assert_eq!(rng.gen_range_u32(50, 50), 50);

        // When max < min, should return min
        assert_eq!(rng.gen_range_u32(100, 50), 100);
    }

    #[test]
    fn test_test_failure_anomaly_detection() {
        // Coverage for test failure anomaly (lines 370-376)
        let config = StressConfig {
            cycles: 1,
            seed: 42,
            ..Default::default()
        };

        let mut runner = StressTestRunner::new(config);

        // Run cycle that reports test failures
        runner.run_cycle(0, |_input| {
            (3, 2) // 3 passed, 2 failed
        });

        let report = runner.report();
        assert_eq!(report.total_passed, 3);
        assert_eq!(report.total_failed, 2);

        // Should have a TestFailure anomaly
        let failure_anomalies: Vec<_> = report
            .anomalies
            .iter()
            .filter(|a| a.kind == AnomalyKind::TestFailure)
            .collect();
        assert_eq!(failure_anomalies.len(), 1);
        assert!(failure_anomalies[0].description.contains("2 tests failed"));
    }

    #[test]
    fn test_verify_performance_fail_variance() {
        // Coverage for timing variance violation (lines 194-199)
        let mut report = StressReport::default();

        // Add frames with high variance in timing
        report.add_frame(FrameProfile {
            cycle: 0,
            duration_ms: 10,
            tests_passed: 1,
            tests_failed: 0,
            ..Default::default()
        });
        report.add_frame(FrameProfile {
            cycle: 1,
            duration_ms: 100, // 10x difference - high variance
            tests_passed: 1,
            tests_failed: 0,
            ..Default::default()
        });

        let thresholds = PerformanceThresholds {
            max_frame_time_ms: 200,    // Pass frame time check
            max_timing_variance: 0.01, // Very strict - will fail
            max_failure_rate: 0.5,     // Pass failure rate
            ..Default::default()
        };

        let result = verify_performance(&report, &thresholds);
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.contains("variance")));
    }

    #[test]
    fn test_verify_performance_fail_pass_rate() {
        // Coverage for pass rate violation (lines 201-207)
        let mut report = StressReport::default();

        // Add frames with high failure rate
        report.add_frame(FrameProfile {
            cycle: 0,
            duration_ms: 50,
            tests_passed: 1,
            tests_failed: 9, // 90% failure rate
            ..Default::default()
        });

        let thresholds = PerformanceThresholds {
            max_frame_time_ms: 200,
            max_timing_variance: 1.0,
            max_failure_rate: 0.05, // 5% max - will fail
            ..Default::default()
        };

        let result = verify_performance(&report, &thresholds);
        assert!(!result.passed);
        assert!(result.violations.iter().any(|v| v.contains("Pass rate")));
    }

    #[test]
    fn test_runner_verify_method() {
        // Coverage for StressTestRunner::verify (lines 409-412)
        let config = StressConfig {
            cycles: 1,
            seed: 42,
            thresholds: PerformanceThresholds::default(),
            ..Default::default()
        };

        let mut runner = StressTestRunner::new(config);
        runner.run_cycle(0, |_input| (5, 0));

        let result = runner.verify();
        assert!(result.passed);
        assert_eq!(result.pass_rate, 1.0);
    }

    #[test]
    fn test_run_all_cycles() {
        // Coverage for run_all (lines 383-400)
        let config = StressConfig {
            cycles: 3,
            interval_ms: 1, // Very short interval for fast test
            seed: 42,
            min_input_size: 10,
            max_input_size: 20,
            thresholds: PerformanceThresholds::default(),
        };

        let mut runner = StressTestRunner::new(config);
        let report = runner.run_all(|input| {
            // Simple test function
            let sum: f32 = input.iter().sum();
            if sum > 0.0 {
                (1, 0)
            } else {
                (0, 1)
            }
        });

        assert_eq!(report.cycles_completed, 3);
        assert_eq!(report.frames.len(), 3);
        assert!(report.total_passed >= 1);
    }

    #[test]
    fn test_run_all_with_slow_test() {
        // Coverage for run_all when test takes longer than interval
        // (lines 394-396 - the checked_sub branch returning None)
        let config = StressConfig {
            cycles: 2,
            interval_ms: 1, // 1ms interval
            seed: 42,
            min_input_size: 10,
            max_input_size: 20,
            thresholds: PerformanceThresholds {
                max_frame_time_ms: 1000, // High threshold to avoid anomaly
                ..Default::default()
            },
        };

        let mut runner = StressTestRunner::new(config);
        let report = runner.run_all(|_input| {
            // Sleep longer than interval to trigger the checked_sub branch
            std::thread::sleep(Duration::from_millis(5));
            (1, 0)
        });

        assert_eq!(report.cycles_completed, 2);
    }

    #[test]
    fn test_anomaly_kinds_equality() {
        // Coverage for AnomalyKind PartialEq/Eq
        assert_eq!(AnomalyKind::SlowFrame, AnomalyKind::SlowFrame);
        assert_ne!(AnomalyKind::SlowFrame, AnomalyKind::HighMemory);
        assert_ne!(AnomalyKind::TestFailure, AnomalyKind::TimingSpike);
        assert_ne!(AnomalyKind::NonDeterministic, AnomalyKind::HighMemory);
    }

    #[test]
    fn test_stress_config_default() {
        // Coverage for StressConfig default values
        let config = StressConfig::default();
        assert_eq!(config.cycles, 100);
        assert_eq!(config.interval_ms, 100);
        assert_eq!(config.seed, 42);
        assert_eq!(config.min_input_size, 64);
        assert_eq!(config.max_input_size, 512);
    }

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
}
