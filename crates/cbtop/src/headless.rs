//! Headless benchmark mode for CI/CD and AI agent integration
//!
//! Enables cbtop to run without a TTY, outputting machine-readable results.
//!
//! # Example
//!
//! ```bash
//! # Run headless benchmark
//! cbtop --headless --format json --duration 5
//!
//! # Use bench subcommand
//! cbtop bench --backend simd --workload gemm --duration 5
//! ```

use crate::config::{ComputeBackend, WorkloadType};
use crate::error::CbtopError;
use crate::bricks::generators::SimdLoadBrick;
use crate::brick::{BrickScore, Scorable};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Output format for benchmark results
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Json,
    Text,
}

/// System information for benchmark context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub cpu: String,
    pub cores: usize,
    pub memory_gb: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
}

impl SystemInfo {
    pub fn detect() -> Self {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Try to get CPU info from /proc/cpuinfo on Linux
        let cpu = Self::detect_cpu();

        // Get memory info
        let memory_gb = Self::detect_memory_gb();

        Self {
            cpu,
            cores,
            memory_gb,
            gpu: None, // GPU detection requires CUDA/wgpu initialization
        }
    }

    fn detect_cpu() -> String {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in content.lines() {
                    if line.starts_with("model name") {
                        if let Some(name) = line.split(':').nth(1) {
                            return name.trim().to_string();
                        }
                    }
                }
            }
        }
        "Unknown CPU".to_string()
    }

    fn detect_memory_gb() -> u64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb / 1024 / 1024; // Convert KB to GB
                            }
                        }
                    }
                }
            }
        }
        0
    }
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub backend: String,
    pub workload: String,
    pub size: usize,
    pub iterations: u64,
}

/// Latency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub cv_percent: f64,
}

/// Score breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreInfo {
    pub total: u8,
    pub grade: String,
    pub performance: u8,
    pub efficiency: u8,
    pub correctness: u8,
    pub stability: u8,
}

impl From<BrickScore> for ScoreInfo {
    fn from(score: BrickScore) -> Self {
        Self {
            total: score.total(),
            grade: format!("{:?}", score.grade()),
            performance: score.performance,
            efficiency: score.efficiency,
            correctness: score.correctness,
            stability: score.stability,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub version: String,
    pub timestamp: String,
    pub duration_secs: f64,
    pub system: SystemInfo,
    pub benchmark: BenchmarkConfig,
    pub results: BenchmarkResults,
    pub score: ScoreInfo,
}

/// Core benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub gflops: f64,
    pub throughput_ops_sec: f64,
    pub latency_ms: LatencyStats,
}

impl BenchmarkResult {
    /// Format result for output
    pub fn format(&self, format: OutputFormat) -> String {
        match format {
            OutputFormat::Json => {
                serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
            }
            OutputFormat::Text => self.format_text(),
        }
    }

    fn format_text(&self) -> String {
        format!(
            r#"
=== cbtop Benchmark Results ===

System:
  CPU: {}
  Cores: {}
  Memory: {} GB

Benchmark:
  Backend: {}
  Workload: {}
  Size: {} elements
  Iterations: {}
  Duration: {:.2}s

Results:
  GFLOP/s: {:.2}
  Throughput: {:.0} ops/sec
  Latency (ms):
    Mean: {:.3}
    P50:  {:.3}
    P95:  {:.3}
    P99:  {:.3}
    CV:   {:.1}%

Score: {}/100 (Grade: {})
  Performance:  {}/40
  Efficiency:   {}/25
  Correctness:  {}/20
  Stability:    {}/15
"#,
            self.system.cpu,
            self.system.cores,
            self.system.memory_gb,
            self.benchmark.backend,
            self.benchmark.workload,
            self.benchmark.size,
            self.benchmark.iterations,
            self.duration_secs,
            self.results.gflops,
            self.results.throughput_ops_sec,
            self.results.latency_ms.mean,
            self.results.latency_ms.p50,
            self.results.latency_ms.p95,
            self.results.latency_ms.p99,
            self.results.latency_ms.cv_percent,
            self.score.total,
            self.score.grade,
            self.score.performance,
            self.score.efficiency,
            self.score.correctness,
            self.score.stability,
        )
    }

    /// Check for regression against baseline
    pub fn check_regression(&self, baseline: &BenchmarkResult, threshold: f64) -> RegressionResult {
        let change_percent =
            (self.results.gflops - baseline.results.gflops) / baseline.results.gflops * 100.0;

        RegressionResult {
            baseline_gflops: baseline.results.gflops,
            current_gflops: self.results.gflops,
            change_percent,
            threshold_percent: threshold,
            is_regression: change_percent < -threshold,
            status: if change_percent < -threshold {
                "REGRESSION".to_string()
            } else if change_percent > threshold {
                "IMPROVEMENT".to_string()
            } else {
                "STABLE".to_string()
            },
        }
    }

    /// Compare multiple benchmark results
    pub fn compare(results: &[(String, BenchmarkResult)]) -> ComparisonResult {
        let comparisons: Vec<_> = results
            .iter()
            .map(|(name, r)| BackendComparison {
                backend: name.clone(),
                gflops: r.results.gflops,
                score: r.score.total,
                latency_mean_ms: r.results.latency_ms.mean,
            })
            .collect();

        let best = comparisons
            .iter()
            .max_by(|a, b| a.gflops.partial_cmp(&b.gflops).unwrap())
            .map(|c| c.backend.clone())
            .unwrap_or_default();

        ComparisonResult {
            backends: comparisons,
            recommended: best,
        }
    }
}

/// Regression check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    pub baseline_gflops: f64,
    pub current_gflops: f64,
    pub change_percent: f64,
    pub threshold_percent: f64,
    pub is_regression: bool,
    pub status: String,
}

impl RegressionResult {
    pub fn format(&self, format: OutputFormat) -> String {
        match format {
            OutputFormat::Json => {
                serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
            }
            OutputFormat::Text => {
                format!(
                    r#"
=== Regression Check ===

Baseline: {:.2} GFLOP/s
Current:  {:.2} GFLOP/s
Change:   {:+.1}%
Threshold: {:.1}%

Status: {}
"#,
                    self.baseline_gflops,
                    self.current_gflops,
                    self.change_percent,
                    self.threshold_percent,
                    self.status,
                )
            }
        }
    }
}

/// Backend comparison for --compare mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendComparison {
    pub backend: String,
    pub gflops: f64,
    pub score: u8,
    pub latency_mean_ms: f64,
}

/// Comparison result for multiple backends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub backends: Vec<BackendComparison>,
    pub recommended: String,
}

impl ComparisonResult {
    pub fn format(&self, format: OutputFormat) -> String {
        match format {
            OutputFormat::Json => {
                serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
            }
            OutputFormat::Text => {
                let mut s = String::from("\n=== Backend Comparison ===\n\n");
                s.push_str("Backend      GFLOP/s   Score   Latency\n");
                s.push_str("----------------------------------------\n");
                for c in &self.backends {
                    s.push_str(&format!(
                        "{:<12} {:>7.2}   {:>3}     {:.3}ms\n",
                        c.backend, c.gflops, c.score, c.latency_mean_ms
                    ));
                }
                s.push_str(&format!("\nRecommended: {}\n", self.recommended));
                s
            }
        }
    }
}

/// Headless benchmark runner
pub struct HeadlessBenchmark {
    backend: ComputeBackend,
    workload: WorkloadType,
    size: usize,
    duration: Duration,
}

impl HeadlessBenchmark {
    /// Create a new headless benchmark
    pub fn new(
        backend: ComputeBackend,
        workload: WorkloadType,
        size: usize,
        duration: Duration,
    ) -> Self {
        Self {
            backend,
            workload,
            size,
            duration,
        }
    }

    /// Run the benchmark and return results
    pub fn run(&self) -> Result<BenchmarkResult, CbtopError> {
        let system = SystemInfo::detect();
        let start_time = Instant::now();

        // Create and configure the load brick
        let mut brick = SimdLoadBrick::new(self.size);
        brick.set_workload(self.workload);
        brick.set_intensity(1.0); // Full intensity for benchmarking
        brick.start();

        // Warmup phase (10% of duration, min 100ms)
        let warmup_duration = Duration::from_millis(
            (self.duration.as_millis() / 10).max(100) as u64
        );
        let warmup_start = Instant::now();
        while warmup_start.elapsed() < warmup_duration {
            brick.run_iteration();
        }

        // Reset metrics after warmup
        let mut brick = SimdLoadBrick::new(self.size);
        brick.set_workload(self.workload);
        brick.set_intensity(1.0);
        brick.start();

        // Measurement phase
        let mut iterations = 0u64;
        let measure_start = Instant::now();

        while measure_start.elapsed() < self.duration {
            brick.run_iteration();
            iterations += 1;
        }

        let total_duration = start_time.elapsed();
        brick.stop();

        // Calculate statistics using brick's internal latency history (PERF-002)
        // This ensures CV calculation matches what score() uses
        let latencies = brick.latency_history_slice();
        let latency_stats = Self::calculate_latency_stats(&latencies);
        let gflops = brick.gflops();
        let throughput = if latency_stats.mean > 0.0 {
            1000.0 / latency_stats.mean
        } else {
            0.0
        };

        // Get score
        let score = brick.score();

        Ok(BenchmarkResult {
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            duration_secs: total_duration.as_secs_f64(),
            system,
            benchmark: BenchmarkConfig {
                backend: format!("{:?}", self.backend),
                workload: format!("{:?}", self.workload),
                size: self.size,
                iterations,
            },
            results: BenchmarkResults {
                gflops,
                throughput_ops_sec: throughput,
                latency_ms: latency_stats,
            },
            score: score.into(),
        })
    }

    fn calculate_latency_stats(latencies: &[f64]) -> LatencyStats {
        if latencies.is_empty() {
            return LatencyStats {
                mean: 0.0,
                min: 0.0,
                max: 0.0,
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
                cv_percent: 0.0,
            };
        }

        let n = latencies.len() as f64;
        let mean = latencies.iter().sum::<f64>() / n;
        let min = latencies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = latencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Calculate standard deviation
        let variance = latencies.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let cv_percent = if mean > 0.0 { (std_dev / mean) * 100.0 } else { 0.0 };

        // Calculate percentiles
        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile = |p: f64| -> f64 {
            let idx = (p * (sorted.len() - 1) as f64).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        };

        LatencyStats {
            mean,
            min,
            max,
            p50: percentile(0.50),
            p95: percentile(0.95),
            p99: percentile(0.99),
            cv_percent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_info_detect() {
        let info = SystemInfo::detect();
        assert!(info.cores > 0);
    }

    #[test]
    fn test_latency_stats_calculation() {
        let latencies = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = HeadlessBenchmark::calculate_latency_stats(&latencies);
        assert!((stats.mean - 3.0).abs() < 0.01);
        assert!((stats.min - 1.0).abs() < 0.01);
        assert!((stats.max - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_benchmark_result_json_format() {
        let result = BenchmarkResult {
            version: "0.1.0".to_string(),
            timestamp: "2026-01-11T10:00:00Z".to_string(),
            duration_secs: 5.0,
            system: SystemInfo {
                cpu: "Test CPU".to_string(),
                cores: 4,
                memory_gb: 16,
                gpu: None,
            },
            benchmark: BenchmarkConfig {
                backend: "Simd".to_string(),
                workload: "Gemm".to_string(),
                size: 1000000,
                iterations: 500,
            },
            results: BenchmarkResults {
                gflops: 25.0,
                throughput_ops_sec: 1000.0,
                latency_ms: LatencyStats {
                    mean: 1.0,
                    min: 0.5,
                    max: 2.0,
                    p50: 0.9,
                    p95: 1.5,
                    p99: 1.8,
                    cv_percent: 10.0,
                },
            },
            score: ScoreInfo {
                total: 85,
                grade: "B".to_string(),
                performance: 35,
                efficiency: 20,
                correctness: 20,
                stability: 10,
            },
        };

        let json = result.format(OutputFormat::Json);
        assert!(json.contains("\"gflops\": 25.0"));
        assert!(json.contains("\"total\": 85"));
    }

    #[test]
    fn test_regression_detection() {
        let baseline = BenchmarkResult {
            version: "0.1.0".to_string(),
            timestamp: "2026-01-11T10:00:00Z".to_string(),
            duration_secs: 5.0,
            system: SystemInfo {
                cpu: "Test".to_string(),
                cores: 4,
                memory_gb: 16,
                gpu: None,
            },
            benchmark: BenchmarkConfig {
                backend: "Simd".to_string(),
                workload: "Gemm".to_string(),
                size: 1000000,
                iterations: 500,
            },
            results: BenchmarkResults {
                gflops: 25.0,
                throughput_ops_sec: 1000.0,
                latency_ms: LatencyStats {
                    mean: 1.0, min: 0.5, max: 2.0,
                    p50: 0.9, p95: 1.5, p99: 1.8,
                    cv_percent: 10.0,
                },
            },
            score: ScoreInfo {
                total: 85, grade: "B".to_string(),
                performance: 35, efficiency: 20,
                correctness: 20, stability: 10,
            },
        };

        let mut current = baseline.clone();
        current.results.gflops = 22.0; // 12% regression

        let regression = current.check_regression(&baseline, 5.0);
        assert!(regression.is_regression);
        assert_eq!(regression.status, "REGRESSION");
        assert!(regression.change_percent < -10.0);
    }

    #[test]
    fn test_headless_benchmark_short_run() {
        let benchmark = HeadlessBenchmark::new(
            ComputeBackend::Simd,
            WorkloadType::Gemm,
            10000,
            Duration::from_millis(100),
        );

        let result = benchmark.run().unwrap();
        assert!(result.results.gflops > 0.0);
        assert!(result.benchmark.iterations > 0);
    }
}
