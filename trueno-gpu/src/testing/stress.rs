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
        let variance: f64 = self.frames.iter()
            .map(|f| {
                let diff = f.duration_ms as f64 - mean;
                diff * diff
            })
            .sum::<f64>() / self.frames.len() as f64;
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
            max_frame_time_ms: 100,           // 10 FPS minimum
            max_memory_bytes: 64 * 1024 * 1024, // 64MB max
            max_timing_variance: 0.2,         // 20% max variance
            max_failure_rate: 0.01,           // 1% max failures
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
pub fn verify_performance(report: &StressReport, thresholds: &PerformanceThresholds) -> PerformanceResult {
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
        let mut rng = Self { state: 0, inc: (seed << 1) | 1 };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Generate next u32
    pub fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6364136223846793005)
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
            if elapsed < interval {
                std::thread::sleep(interval - elapsed);
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
}
