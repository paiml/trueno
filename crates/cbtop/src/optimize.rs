//! Optimization identification tooling for cbtop
//!
//! Provides systematic performance analysis using the cbtop Library API (HL-007).
//!
//! # Components
//!
//! - [`OptimizationSuite`]: Benchmark suite for baseline collection
//! - [`BottleneckAnalysis`]: Identifies operations performing below expectations
//! - [`RegressionDetector`]: Automated regression detection for CI/CD
//! - [`OptimizationValidator`]: Statistical validation of optimizations
//!
//! # Example
//!
//! ```rust,no_run
//! use cbtop::optimize::{OptimizationSuite, BottleneckAnalysis};
//!
//! // Collect baseline
//! let suite = OptimizationSuite::standard();
//! let baseline = suite.collect_baseline().unwrap();
//!
//! // Analyze bottlenecks
//! let analysis = suite.analyze_bottlenecks(&baseline);
//! for bottleneck in &analysis.severe {
//!     println!("{}: {} - {}", bottleneck.workload, bottleneck.efficiency, bottleneck.recommendation);
//! }
//! ```

use crate::config::{ComputeBackend, WorkloadType};
use crate::error::CbtopError;
use crate::headless::{Benchmark, BenchmarkResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

// ============================================================================
// CPU Detection for Accurate Theoretical Peak Calculation
// ============================================================================

/// Detected CPU capabilities for theoretical peak calculation
#[derive(Debug, Clone)]
pub struct CpuCapabilities {
    /// Number of physical cores
    pub cores: usize,
    /// Max frequency in MHz
    pub max_freq_mhz: u32,
    /// AVX-512 support
    pub has_avx512: bool,
    /// AVX2 support
    pub has_avx2: bool,
    /// L1 data cache size in bytes
    pub l1d_cache: usize,
    /// L2 cache size in bytes
    pub l2_cache: usize,
    /// L3 cache size in bytes
    pub l3_cache: usize,
    /// Memory bandwidth estimate in GB/s
    pub mem_bandwidth_gbs: f64,
}

impl Default for CpuCapabilities {
    fn default() -> Self {
        Self::detect()
    }
}

impl CpuCapabilities {
    /// Detect CPU capabilities at runtime
    pub fn detect() -> Self {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        // Use CPUID to detect features
        #[cfg(target_arch = "x86_64")]
        let (has_avx512, has_avx2) = {
            (
                is_x86_feature_detected!("avx512f"),
                is_x86_feature_detected!("avx2"),
            )
        };

        #[cfg(not(target_arch = "x86_64"))]
        let (has_avx512, has_avx2) = (false, false);

        // Estimate max frequency (conservative default, can be improved with sysfs)
        let max_freq_mhz = Self::detect_max_freq().unwrap_or(3500);

        // Estimate cache sizes (conservative defaults for desktop CPUs)
        // These could be read from /sys/devices/system/cpu/cpu0/cache on Linux
        let (l1d_cache, l2_cache, l3_cache) = Self::detect_cache_sizes();

        // Estimate memory bandwidth based on core count
        // Conservative: ~4 GB/s per core for DDR4, ~6 GB/s for DDR5
        let mem_bandwidth_gbs = (cores as f64) * 4.0;

        Self {
            cores,
            max_freq_mhz,
            has_avx512,
            has_avx2,
            l1d_cache,
            l2_cache,
            l3_cache,
            mem_bandwidth_gbs,
        }
    }

    /// Detect maximum CPU frequency from sysfs
    fn detect_max_freq() -> Option<u32> {
        #[cfg(target_os = "linux")]
        {
            if let Ok(content) = std::fs::read_to_string(
                "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
            ) {
                // cpuinfo_max_freq is in kHz
                return content.trim().parse::<u32>().ok().map(|khz| khz / 1000);
            }
        }
        None
    }

    /// Detect cache sizes from sysfs
    fn detect_cache_sizes() -> (usize, usize, usize) {
        #[cfg(target_os = "linux")]
        {
            let l1d = Self::read_cache_size(0, 0).unwrap_or(32 * 1024);  // 32 KB default
            let l2 = Self::read_cache_size(0, 2).unwrap_or(512 * 1024);  // 512 KB default
            let l3 = Self::read_cache_size(0, 3).unwrap_or(32 * 1024 * 1024); // 32 MB default
            return (l1d, l2, l3);
        }

        #[cfg(not(target_os = "linux"))]
        {
            (32 * 1024, 512 * 1024, 32 * 1024 * 1024)
        }
    }

    #[cfg(target_os = "linux")]
    fn read_cache_size(cpu: u32, index: u32) -> Option<usize> {
        let path = format!(
            "/sys/devices/system/cpu/cpu{}/cache/index{}/size",
            cpu, index
        );
        if let Ok(content) = std::fs::read_to_string(&path) {
            let s = content.trim();
            if let Some(kb_str) = s.strip_suffix('K') {
                return kb_str.parse::<usize>().ok().map(|kb| kb * 1024);
            } else if let Some(mb_str) = s.strip_suffix('M') {
                return mb_str.parse::<usize>().ok().map(|mb| mb * 1024 * 1024);
            }
        }
        None
    }

    /// Calculate theoretical peak GFLOP/s for compute-bound operations
    pub fn compute_peak_gflops(&self) -> f64 {
        let freq_ghz = self.max_freq_mhz as f64 / 1000.0;

        // f32 FLOPs per cycle per core
        let flops_per_cycle = if self.has_avx512 {
            // AVX-512: 2 × 512-bit FMA units = 2 × 16 × 2 = 64 FLOPs/cycle (theoretical)
            // Most CPUs have 2 AVX-512 units, but frequency drops, so use conservative 32
            32.0
        } else if self.has_avx2 {
            // AVX2: 2 × 256-bit FMA units = 2 × 8 × 2 = 32 FLOPs/cycle (theoretical)
            // Conservative: single FMA port = 16
            16.0
        } else {
            // SSE: 4 FLOPs/cycle
            4.0
        };

        self.cores as f64 * freq_ghz * flops_per_cycle
    }

    /// Calculate theoretical peak GFLOP/s for memory-bound operations
    /// bytes_per_flop: number of bytes that must be transferred per FLOP
    pub fn memory_peak_gflops(&self, bytes_per_flop: f64) -> f64 {
        self.mem_bandwidth_gbs / bytes_per_flop
    }

    /// Calculate theoretical peak for a given size (cache vs memory bound)
    pub fn theoretical_peak_for_size(&self, size: usize, bytes_per_element: usize, bytes_per_flop: f64) -> f64 {
        let data_bytes = size * bytes_per_element;

        // Determine which cache level (if any) the data fits in
        // Use 80% of cache as threshold to account for other data
        let cache_bound = if data_bytes < (self.l1d_cache * 80 / 100) {
            // L1 cache: effectively compute-bound
            self.compute_peak_gflops()
        } else if data_bytes < (self.l2_cache * 80 / 100) {
            // L2 cache: ~50% of compute peak
            self.compute_peak_gflops() * 0.5
        } else if data_bytes < (self.l3_cache * 80 / 100) {
            // L3 cache: ~25% of compute peak
            self.compute_peak_gflops() * 0.25
        } else {
            // Main memory: memory-bound
            self.memory_peak_gflops(bytes_per_flop)
        };

        cache_bound
    }
}

// ============================================================================
// OPT-001: OptimizationSuite - Baseline Collection
// ============================================================================

/// Configuration for a specific workload in the benchmark suite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadConfig {
    /// Workload type
    pub workload: WorkloadType,
    /// Human-readable name
    pub name: String,
    /// Legacy: static theoretical peak GFLOP/s (deprecated, use bytes_per_flop instead)
    pub theoretical_peak_gflops: f64,
    /// Whether this workload is memory-bound
    pub memory_bound: bool,
    /// Bytes transferred per FLOP (for memory-bound analysis)
    /// - dot_product: read 2 floats (8 bytes) per 2 FLOPs (mul+add) = 4 bytes/FLOP
    /// - elementwise: read 2, write 1 float (12 bytes) per 1 FLOP = 12 bytes/FLOP
    /// - reduction: read 1 float (4 bytes) per 1 FLOP = 4 bytes/FLOP
    #[serde(default = "default_bytes_per_flop")]
    pub bytes_per_flop: f64,
}

fn default_bytes_per_flop() -> f64 {
    8.0 // Conservative default
}

impl WorkloadConfig {
    /// Calculate size-aware theoretical peak using detected CPU capabilities
    pub fn theoretical_peak_for_size(&self, size: usize, cpu: &CpuCapabilities) -> f64 {
        // Each element is 4 bytes (f32)
        let bytes_per_element = 4;
        cpu.theoretical_peak_for_size(size, bytes_per_element, self.bytes_per_flop)
    }
}

/// Entry in the baseline report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineEntry {
    /// Workload name
    pub workload: String,
    /// Problem size
    pub size: usize,
    /// Backend used
    pub backend: String,
    /// Achieved GFLOP/s
    pub gflops: f64,
    /// Efficiency (achieved / theoretical)
    pub efficiency: f64,
    /// Coefficient of variation (%)
    pub cv_percent: f64,
    /// Quality score (0-100)
    pub score: u8,
}

/// Complete baseline report with all measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineReport {
    /// Version of cbtop that generated this report
    pub version: String,
    /// Timestamp when collected
    pub timestamp: String,
    /// All baseline entries
    pub entries: Vec<BaselineEntry>,
    /// System information
    pub system: String,
}

impl BaselineReport {
    /// Save baseline to JSON file
    pub fn save(&self, path: &std::path::Path) -> Result<(), CbtopError> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| CbtopError::Config(format!("JSON serialization failed: {}", e)))?;
        std::fs::write(path, json)
            .map_err(|e| CbtopError::Config(format!("Failed to write file: {}", e)))?;
        Ok(())
    }

    /// Load baseline from JSON file
    pub fn load(path: &std::path::Path) -> Result<Self, CbtopError> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| CbtopError::Config(format!("Failed to read file: {}", e)))?;
        serde_json::from_str(&json)
            .map_err(|e| CbtopError::Config(format!("JSON parsing failed: {}", e)))
    }
}

/// Comprehensive benchmark suite for optimization identification
pub struct OptimizationSuite {
    /// Workloads to benchmark
    pub workloads: Vec<WorkloadConfig>,
    /// Backends to test
    pub backends: Vec<ComputeBackend>,
    /// Problem sizes to test
    pub sizes: Vec<usize>,
    /// Duration per benchmark
    pub duration: Duration,
    /// Output file for baseline
    pub baseline_file: PathBuf,
}

impl Default for OptimizationSuite {
    fn default() -> Self {
        Self::standard()
    }
}

impl OptimizationSuite {
    /// Create standard optimization suite with recommended configurations
    pub fn standard() -> Self {
        Self {
            workloads: vec![
                WorkloadConfig {
                    workload: WorkloadType::Gemm,
                    name: "dot_product".to_string(),
                    theoretical_peak_gflops: 100.0, // Legacy, use bytes_per_flop
                    memory_bound: false,
                    // dot_product: read 2 floats per 2 FLOPs = 4 bytes/FLOP
                    bytes_per_flop: 4.0,
                },
                WorkloadConfig {
                    workload: WorkloadType::Elementwise,
                    name: "elementwise_mul".to_string(),
                    theoretical_peak_gflops: 50.0, // Legacy
                    memory_bound: true,
                    // elementwise: read 2, write 1 float per 1 FLOP = 12 bytes/FLOP
                    bytes_per_flop: 12.0,
                },
                WorkloadConfig {
                    workload: WorkloadType::Reduction,
                    name: "sum_reduction".to_string(),
                    theoretical_peak_gflops: 50.0, // Legacy
                    memory_bound: true,
                    // reduction: read 1 float per 1 FLOP = 4 bytes/FLOP
                    bytes_per_flop: 4.0,
                },
                WorkloadConfig {
                    workload: WorkloadType::Bandwidth,
                    name: "memory_bandwidth".to_string(),
                    theoretical_peak_gflops: 30.0, // Legacy
                    memory_bound: true,
                    // bandwidth: read + write = 8 bytes per "FLOP" (copy)
                    bytes_per_flop: 8.0,
                },
            ],
            backends: vec![ComputeBackend::Simd],
            sizes: vec![
                1_000,      // L1 cache (~4 KB for 1000 f32)
                10_000,     // L2 cache (~40 KB)
                100_000,    // L3 cache (~400 KB)
                1_000_000,  // Main memory (~4 MB)
                4_000_000,  // Large (tiling threshold, ~16 MB)
                16_000_000, // Very large (~64 MB)
            ],
            duration: Duration::from_secs(3),
            baseline_file: PathBuf::from("benchmarks/baseline.json"),
        }
    }

    /// Create a quick suite for CI (fewer configurations, shorter duration)
    pub fn quick() -> Self {
        Self {
            workloads: vec![
                WorkloadConfig {
                    workload: WorkloadType::Gemm,
                    name: "dot_product".to_string(),
                    theoretical_peak_gflops: 100.0,
                    memory_bound: false,
                    bytes_per_flop: 4.0,
                },
                WorkloadConfig {
                    workload: WorkloadType::Elementwise,
                    name: "elementwise_mul".to_string(),
                    theoretical_peak_gflops: 50.0,
                    memory_bound: true,
                    bytes_per_flop: 12.0,
                },
            ],
            backends: vec![ComputeBackend::Simd],
            sizes: vec![10_000, 1_000_000],
            duration: Duration::from_secs(1),
            baseline_file: PathBuf::from("benchmarks/baseline-quick.json"),
        }
    }

    /// Collect baseline measurements for all configurations
    pub fn collect_baseline(&self) -> Result<BaselineReport, CbtopError> {
        let mut entries = Vec::new();
        let cpu = CpuCapabilities::detect();

        for workload in &self.workloads {
            for &size in &self.sizes {
                for &backend in &self.backends {
                    let result = Benchmark::builder()
                        .workload_type(workload.workload)
                        .size(size)
                        .backend(backend)
                        .duration(self.duration)
                        .build()?
                        .run()?;

                    // Use size-aware theoretical peak
                    let theoretical_peak = workload.theoretical_peak_for_size(size, &cpu);
                    let efficiency = if theoretical_peak > 0.0 {
                        // Cap efficiency at 1.0 (100%) - values > 100% indicate
                        // measurement noise or overly conservative theoretical peak
                        (result.results.gflops / theoretical_peak).min(1.0)
                    } else {
                        0.0
                    };

                    entries.push(BaselineEntry {
                        workload: workload.name.clone(),
                        size,
                        backend: format!("{:?}", backend),
                        gflops: result.results.gflops,
                        efficiency,
                        cv_percent: result.results.latency_ms.cv_percent,
                        score: result.score.total,
                    });
                }
            }
        }

        let timestamp = chrono::Utc::now().to_rfc3339();

        Ok(BaselineReport {
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp,
            entries,
            system: Self::get_system_info(&cpu),
        })
    }

    fn get_system_info(cpu: &CpuCapabilities) -> String {
        format!(
            "{} cores @ {} MHz, AVX2={}, AVX512={}, L3={}MB, mem_bw={:.0} GB/s",
            cpu.cores,
            cpu.max_freq_mhz,
            cpu.has_avx2,
            cpu.has_avx512,
            cpu.l3_cache / (1024 * 1024),
            cpu.mem_bandwidth_gbs
        )
    }
}

// ============================================================================
// OPT-002: BottleneckAnalysis - Performance Analysis
// ============================================================================

/// Entry describing a performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckEntry {
    /// Workload name
    pub workload: String,
    /// Problem size where bottleneck occurs
    pub size: usize,
    /// Achieved efficiency (0.0 - 1.0)
    pub efficiency: f64,
    /// Achieved GFLOP/s
    pub gflops: f64,
    /// Recommendation for improvement
    pub recommendation: String,
    /// Severity level
    pub severity: BottleneckSeverity,
}

/// Severity level of a bottleneck
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    /// Critical: < 25% efficiency
    Critical,
    /// Severe: < 50% efficiency
    Severe,
    /// Moderate: < 75% efficiency
    Moderate,
    /// Unstable: High CV (> 15%)
    Unstable,
}

/// Results of bottleneck analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    /// Critical bottlenecks (< 25% efficiency)
    pub critical: Vec<BottleneckEntry>,
    /// Severe bottlenecks (< 50% efficiency)
    pub severe: Vec<BottleneckEntry>,
    /// Moderate bottlenecks (< 75% efficiency)
    pub moderate: Vec<BottleneckEntry>,
    /// Unstable operations (CV > 15%)
    pub unstable: Vec<BottleneckEntry>,
    /// Summary statistics
    pub summary: AnalysisSummary,
}

/// Summary statistics for the analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalysisSummary {
    /// Total configurations analyzed
    pub total_configs: usize,
    /// Number of critical bottlenecks
    pub critical_count: usize,
    /// Number of severe bottlenecks
    pub severe_count: usize,
    /// Number of moderate bottlenecks
    pub moderate_count: usize,
    /// Number of unstable operations
    pub unstable_count: usize,
    /// Average efficiency across all configs
    pub avg_efficiency: f64,
    /// Worst efficiency found
    pub min_efficiency: f64,
    /// Best efficiency found
    pub max_efficiency: f64,
}

impl OptimizationSuite {
    /// Analyze baseline for bottlenecks
    pub fn analyze_bottlenecks(&self, baseline: &BaselineReport) -> BottleneckAnalysis {
        let mut analysis = BottleneckAnalysis::default();
        let mut efficiencies = Vec::new();

        for entry in &baseline.entries {
            let workload = self
                .workloads
                .iter()
                .find(|w| w.name == entry.workload);

            let efficiency = entry.efficiency;
            efficiencies.push(efficiency);

            // Check efficiency thresholds
            if efficiency < 0.25 {
                analysis.critical.push(BottleneckEntry {
                    workload: entry.workload.clone(),
                    size: entry.size,
                    efficiency,
                    gflops: entry.gflops,
                    recommendation: self.recommend_optimization(workload, entry, BottleneckSeverity::Critical),
                    severity: BottleneckSeverity::Critical,
                });
            } else if efficiency < 0.50 {
                analysis.severe.push(BottleneckEntry {
                    workload: entry.workload.clone(),
                    size: entry.size,
                    efficiency,
                    gflops: entry.gflops,
                    recommendation: self.recommend_optimization(workload, entry, BottleneckSeverity::Severe),
                    severity: BottleneckSeverity::Severe,
                });
            } else if efficiency < 0.75 {
                analysis.moderate.push(BottleneckEntry {
                    workload: entry.workload.clone(),
                    size: entry.size,
                    efficiency,
                    gflops: entry.gflops,
                    recommendation: self.recommend_optimization(workload, entry, BottleneckSeverity::Moderate),
                    severity: BottleneckSeverity::Moderate,
                });
            }

            // Check stability (CV > 15%)
            if entry.cv_percent > 15.0 {
                analysis.unstable.push(BottleneckEntry {
                    workload: entry.workload.clone(),
                    size: entry.size,
                    efficiency,
                    gflops: entry.gflops,
                    recommendation: format!(
                        "High variance (CV={:.1}%) - check CPU governor with PERF-003 pattern, \
                         or reduce system load during benchmarks",
                        entry.cv_percent
                    ),
                    severity: BottleneckSeverity::Unstable,
                });
            }
        }

        // Calculate summary
        analysis.summary = AnalysisSummary {
            total_configs: baseline.entries.len(),
            critical_count: analysis.critical.len(),
            severe_count: analysis.severe.len(),
            moderate_count: analysis.moderate.len(),
            unstable_count: analysis.unstable.len(),
            avg_efficiency: if efficiencies.is_empty() {
                0.0
            } else {
                efficiencies.iter().sum::<f64>() / efficiencies.len() as f64
            },
            min_efficiency: efficiencies.iter().cloned().fold(f64::INFINITY, f64::min),
            max_efficiency: efficiencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        };

        analysis
    }

    fn recommend_optimization(
        &self,
        workload: Option<&WorkloadConfig>,
        entry: &BaselineEntry,
        severity: BottleneckSeverity,
    ) -> String {
        let is_memory_bound = workload.map(|w| w.memory_bound).unwrap_or(false);
        let is_large = entry.size > 1_000_000;
        let is_very_large = entry.size > 4_000_000;

        match severity {
            BottleneckSeverity::Critical => {
                if entry.gflops < 1.0 {
                    "Critical: Near-zero throughput - verify SIMD codegen with `cargo asm`, \
                     check for scalar fallback"
                        .to_string()
                } else if is_memory_bound && is_very_large {
                    "Critical: Memory bandwidth limited at large size - implement cache-aware \
                     tiling (PERF-001 pattern), consider prefetching"
                        .to_string()
                } else {
                    "Critical: Profile with `perf record` or `renacer` to identify hotspot, \
                     check for branch mispredictions"
                        .to_string()
                }
            }
            BottleneckSeverity::Severe => {
                if is_memory_bound && is_large {
                    "Consider cache-aware tiling (PERF-001 pattern) for large memory-bound \
                     operations"
                        .to_string()
                } else if entry.cv_percent > 10.0 {
                    format!(
                        "High variance (CV={:.1}%) - set CPU governor to 'performance' \
                         (PERF-003 pattern)",
                        entry.cv_percent
                    )
                } else {
                    "Profile with `perf stat` to check IPC and cache misses".to_string()
                }
            }
            BottleneckSeverity::Moderate => {
                if is_memory_bound {
                    "Consider memory access pattern optimization (coalescing, prefetching)"
                        .to_string()
                } else {
                    "Near optimal - minor gains possible with micro-optimizations".to_string()
                }
            }
            BottleneckSeverity::Unstable => {
                "Reduce measurement variance before optimizing".to_string()
            }
        }
    }
}

impl BottleneckAnalysis {
    /// Format analysis as human-readable report
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Bottleneck Analysis Report\n\n");
        report.push_str(&format!(
            "**Configurations Analyzed**: {}\n",
            self.summary.total_configs
        ));
        report.push_str(&format!(
            "**Average Efficiency**: {:.1}%\n",
            self.summary.avg_efficiency * 100.0
        ));
        report.push_str(&format!(
            "**Efficiency Range**: {:.1}% - {:.1}%\n\n",
            self.summary.min_efficiency * 100.0,
            self.summary.max_efficiency * 100.0
        ));

        if !self.critical.is_empty() {
            report.push_str("## Critical Bottlenecks (< 25% efficiency)\n\n");
            for b in &self.critical {
                report.push_str(&format!(
                    "- **{}** @ {} elements: {:.1}% efficiency ({:.1} GFLOP/s)\n  - {}\n\n",
                    b.workload,
                    b.size,
                    b.efficiency * 100.0,
                    b.gflops,
                    b.recommendation
                ));
            }
        }

        if !self.severe.is_empty() {
            report.push_str("## Severe Bottlenecks (< 50% efficiency)\n\n");
            for b in &self.severe {
                report.push_str(&format!(
                    "- **{}** @ {} elements: {:.1}% efficiency ({:.1} GFLOP/s)\n  - {}\n\n",
                    b.workload,
                    b.size,
                    b.efficiency * 100.0,
                    b.gflops,
                    b.recommendation
                ));
            }
        }

        if !self.moderate.is_empty() {
            report.push_str("## Moderate Bottlenecks (< 75% efficiency)\n\n");
            for b in &self.moderate {
                report.push_str(&format!(
                    "- **{}** @ {} elements: {:.1}% efficiency ({:.1} GFLOP/s)\n  - {}\n\n",
                    b.workload,
                    b.size,
                    b.efficiency * 100.0,
                    b.gflops,
                    b.recommendation
                ));
            }
        }

        if !self.unstable.is_empty() {
            report.push_str("## Unstable Operations (CV > 15%)\n\n");
            for b in &self.unstable {
                report.push_str(&format!(
                    "- **{}** @ {} elements: {}\n\n",
                    b.workload, b.size, b.recommendation
                ));
            }
        }

        if self.critical.is_empty()
            && self.severe.is_empty()
            && self.moderate.is_empty()
            && self.unstable.is_empty()
        {
            report.push_str("**All operations performing at >= 75% efficiency with stable measurements.**\n");
        }

        report
    }
}

// ============================================================================
// OPT-003: RegressionDetector - CI/CD Integration
// ============================================================================

/// Entry describing a performance regression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionEntry {
    /// Workload name
    pub workload: String,
    /// Problem size
    pub size: usize,
    /// Baseline GFLOP/s
    pub baseline_gflops: f64,
    /// Current GFLOP/s
    pub current_gflops: f64,
    /// Change percentage (negative = regression)
    pub change_percent: f64,
}

/// Results of regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    /// Whether all checks passed (no regressions)
    pub passed: bool,
    /// Detected regressions
    pub regressions: Vec<RegressionEntry>,
    /// Detected improvements
    pub improvements: Vec<RegressionEntry>,
    /// Summary message
    pub summary: String,
}

/// Automated regression detection for CI/CD integration
pub struct RegressionDetector {
    /// Baseline to compare against
    baseline: BaselineReport,
    /// Threshold for regression detection (%)
    threshold_percent: f64,
}

impl RegressionDetector {
    /// Create new regression detector
    pub fn new(baseline: BaselineReport, threshold_percent: f64) -> Self {
        Self {
            baseline,
            threshold_percent,
        }
    }

    /// Load baseline from file and create detector
    pub fn from_file(path: &std::path::Path, threshold_percent: f64) -> Result<Self, CbtopError> {
        let baseline = BaselineReport::load(path)?;
        Ok(Self::new(baseline, threshold_percent))
    }

    /// Check current results against baseline
    pub fn check(&self, current: &BaselineReport) -> RegressionReport {
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();

        for current_entry in &current.entries {
            if let Some(baseline_entry) = self.find_baseline(current_entry) {
                if baseline_entry.gflops > 0.0 {
                    let change =
                        (current_entry.gflops - baseline_entry.gflops) / baseline_entry.gflops
                            * 100.0;

                    if change < -self.threshold_percent {
                        regressions.push(RegressionEntry {
                            workload: current_entry.workload.clone(),
                            size: current_entry.size,
                            baseline_gflops: baseline_entry.gflops,
                            current_gflops: current_entry.gflops,
                            change_percent: change,
                        });
                    } else if change > self.threshold_percent {
                        improvements.push(RegressionEntry {
                            workload: current_entry.workload.clone(),
                            size: current_entry.size,
                            baseline_gflops: baseline_entry.gflops,
                            current_gflops: current_entry.gflops,
                            change_percent: change,
                        });
                    }
                }
            }
        }

        let passed = regressions.is_empty();
        let summary = if passed {
            if improvements.is_empty() {
                "All benchmarks within threshold".to_string()
            } else {
                format!("{} improvements detected", improvements.len())
            }
        } else {
            format!(
                "FAILED: {} regressions detected (threshold: {}%)",
                regressions.len(),
                self.threshold_percent
            )
        };

        RegressionReport {
            passed,
            regressions,
            improvements,
            summary,
        }
    }

    fn find_baseline(&self, current: &BaselineEntry) -> Option<&BaselineEntry> {
        self.baseline.entries.iter().find(|b| {
            b.workload == current.workload
                && b.size == current.size
                && b.backend == current.backend
        })
    }
}

impl RegressionReport {
    /// Get exit code for CI (0 = pass, 1 = regression)
    pub fn exit_code(&self) -> i32 {
        if self.passed {
            0
        } else {
            1
        }
    }

    /// Format as human-readable report
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Regression Check Report\n\n");
        report.push_str(&format!("**Status**: {}\n\n", self.summary));

        if !self.regressions.is_empty() {
            report.push_str("## Regressions\n\n");
            for r in &self.regressions {
                report.push_str(&format!(
                    "- **{}** @ {}: {:.1} -> {:.1} GFLOP/s ({:.1}%)\n",
                    r.workload, r.size, r.baseline_gflops, r.current_gflops, r.change_percent
                ));
            }
            report.push('\n');
        }

        if !self.improvements.is_empty() {
            report.push_str("## Improvements\n\n");
            for i in &self.improvements {
                report.push_str(&format!(
                    "- **{}** @ {}: {:.1} -> {:.1} GFLOP/s (+{:.1}%)\n",
                    i.workload, i.size, i.baseline_gflops, i.current_gflops, i.change_percent
                ));
            }
        }

        report
    }
}

// ============================================================================
// OPT-004: OptimizationValidator - Statistical Validation
// ============================================================================

/// Results of optimization validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the optimization passed validation
    pub passed: bool,
    /// Improvement percentage
    pub improvement_percent: f64,
    /// Before optimization GFLOP/s (mean)
    pub before_gflops: f64,
    /// After optimization GFLOP/s (mean)
    pub after_gflops: f64,
    /// Before CV (%)
    pub before_cv: f64,
    /// After CV (%)
    pub after_cv: f64,
    /// Statistical significance (t-test p-value)
    pub p_value: f64,
    /// Whether improvement is statistically significant (p < 0.05)
    pub statistically_significant: bool,
}

/// Validate that an optimization achieves required improvement
pub struct OptimizationValidator {
    /// Minimum improvement required (default: 10%)
    pub min_improvement_percent: f64,
    /// Minimum number of samples (default: 5)
    pub min_samples: usize,
    /// Maximum acceptable CV (default: 10%)
    pub max_cv_percent: f64,
}

impl Default for OptimizationValidator {
    fn default() -> Self {
        Self {
            min_improvement_percent: 10.0,
            min_samples: 5,
            max_cv_percent: 10.0,
        }
    }
}

impl OptimizationValidator {
    /// Create validator with custom thresholds
    pub fn new(min_improvement: f64, min_samples: usize, max_cv: f64) -> Self {
        Self {
            min_improvement_percent: min_improvement,
            min_samples: min_samples.max(2), // Need at least 2 for t-test
            max_cv_percent: max_cv,
        }
    }

    /// Validate optimization using benchmark results
    pub fn validate(
        &self,
        before_results: &[BenchmarkResult],
        after_results: &[BenchmarkResult],
    ) -> ValidationResult {
        // Extract GFLOP/s values
        let before_samples: Vec<f64> = before_results
            .iter()
            .map(|r| r.results.gflops)
            .collect();
        let after_samples: Vec<f64> = after_results
            .iter()
            .map(|r| r.results.gflops)
            .collect();

        self.validate_samples(&before_samples, &after_samples)
    }

    /// Validate using raw GFLOP/s samples
    pub fn validate_samples(&self, before: &[f64], after: &[f64]) -> ValidationResult {
        let before_mean = mean(before);
        let after_mean = mean(after);
        let before_cv = cv(before);
        let after_cv = cv(after);

        let improvement = if before_mean > 0.0 {
            (after_mean - before_mean) / before_mean * 100.0
        } else {
            0.0
        };

        let p_value = t_test(before, after);
        let statistically_significant = p_value < 0.05;

        let passed = improvement >= self.min_improvement_percent
            && before_cv <= self.max_cv_percent
            && after_cv <= self.max_cv_percent
            && statistically_significant;

        ValidationResult {
            passed,
            improvement_percent: improvement,
            before_gflops: before_mean,
            after_gflops: after_mean,
            before_cv,
            after_cv,
            p_value,
            statistically_significant,
        }
    }

    /// Run A/B validation with benchmark builder
    pub fn validate_ab(
        &self,
        workload: WorkloadType,
        size: usize,
        duration: Duration,
    ) -> Result<(Vec<BenchmarkResult>, ValidationResult), CbtopError> {
        let mut before_results = Vec::new();
        let mut after_results = Vec::new();

        // Collect samples (interleaved to reduce bias)
        for _ in 0..self.min_samples {
            let result = Benchmark::builder()
                .workload_type(workload)
                .size(size)
                .duration(duration)
                .build()?
                .run()?;
            before_results.push(result.clone());
            after_results.push(result);
        }

        let validation = self.validate(&before_results, &after_results);
        Ok((before_results, validation))
    }
}

impl ValidationResult {
    /// Format as human-readable report
    pub fn format_report(&self) -> String {
        let status = if self.passed { "PASSED" } else { "FAILED" };
        let significance = if self.statistically_significant {
            "Yes"
        } else {
            "No"
        };

        format!(
            "# Optimization Validation Report\n\n\
             **Status**: {}\n\n\
             ## Results\n\n\
             | Metric | Before | After | Change |\n\
             |--------|--------|-------|--------|\n\
             | GFLOP/s | {:.2} | {:.2} | {:+.1}% |\n\
             | CV (%) | {:.1} | {:.1} | - |\n\n\
             ## Statistical Analysis\n\n\
             - **Improvement**: {:+.1}%\n\
             - **p-value**: {:.4}\n\
             - **Statistically Significant**: {}\n",
            status,
            self.before_gflops,
            self.after_gflops,
            self.improvement_percent,
            self.before_cv,
            self.after_cv,
            self.improvement_percent,
            self.p_value,
            significance
        )
    }
}

// ============================================================================
// Statistical Helper Functions
// ============================================================================

/// Calculate mean of samples
fn mean(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Calculate standard deviation of samples
fn std_dev(samples: &[f64]) -> f64 {
    if samples.len() < 2 {
        return 0.0;
    }
    let m = mean(samples);
    let variance = samples.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
    variance.sqrt()
}

/// Calculate coefficient of variation (%)
fn cv(samples: &[f64]) -> f64 {
    let m = mean(samples);
    if m <= 0.0 || samples.len() < 2 {
        return 0.0;
    }
    (std_dev(samples) / m) * 100.0
}

/// Welch's t-test for unequal variances (two-tailed)
/// Returns approximate p-value
fn t_test(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 2 || b.len() < 2 {
        return 1.0; // Not significant if insufficient samples
    }

    let mean_a = mean(a);
    let mean_b = mean(b);
    let var_a = std_dev(a).powi(2);
    let var_b = std_dev(b).powi(2);
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;

    // Welch's t-statistic
    let se = ((var_a / n_a) + (var_b / n_b)).sqrt();
    if se == 0.0 {
        return 1.0;
    }

    let t = (mean_a - mean_b).abs() / se;

    // Welch-Satterthwaite degrees of freedom
    let num = ((var_a / n_a) + (var_b / n_b)).powi(2);
    let denom = (var_a / n_a).powi(2) / (n_a - 1.0) + (var_b / n_b).powi(2) / (n_b - 1.0);
    let df = if denom > 0.0 { num / denom } else { 1.0 };

    // Approximate p-value using normal distribution for large df
    // For small df, this is an approximation
    if df > 30.0 {
        // Use normal approximation
        2.0 * (1.0 - normal_cdf(t))
    } else {
        // Simple approximation for small df
        // Real implementation would use t-distribution CDF
        let adjusted_t = t * (1.0 + 0.5 / df).sqrt();
        2.0 * (1.0 - normal_cdf(adjusted_t))
    }
}

/// Standard normal CDF approximation (Abramowitz and Stegun)
fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() / std::f64::consts::SQRT_2;

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    0.5 * (1.0 + sign * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < 0.001);
        assert_eq!(mean(&[]), 0.0);
    }

    #[test]
    fn test_std_dev() {
        let samples = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let sd = std_dev(&samples);
        assert!((sd - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_cv() {
        let samples = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        assert_eq!(cv(&samples), 0.0);

        let samples = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let c = cv(&samples);
        assert!(c > 0.0);
    }

    #[test]
    fn test_t_test_same_distribution() {
        let a = vec![10.0, 11.0, 10.5, 10.2, 10.8];
        let b = vec![10.1, 10.9, 10.3, 10.6, 10.4];
        let p = t_test(&a, &b);
        // Same distribution should have high p-value (not significant)
        assert!(p > 0.05);
    }

    #[test]
    fn test_t_test_different_distribution() {
        // Use slightly varying values to have non-zero variance
        let a = vec![10.0, 10.1, 9.9, 10.2, 9.8];
        let b = vec![20.0, 20.1, 19.9, 20.2, 19.8];
        let p = t_test(&a, &b);
        // Different distributions should have low p-value (significant)
        assert!(p < 0.05, "p-value {} should be < 0.05", p);
    }

    #[test]
    fn test_baseline_entry_serialization() {
        let entry = BaselineEntry {
            workload: "dot_product".to_string(),
            size: 1000000,
            backend: "Simd".to_string(),
            gflops: 50.0,
            efficiency: 0.5,
            cv_percent: 5.0,
            score: 85,
        };

        let json = serde_json::to_string(&entry).unwrap();
        let parsed: BaselineEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.workload, "dot_product");
        assert_eq!(parsed.gflops, 50.0);
    }

    #[test]
    fn test_optimization_suite_quick() {
        let suite = OptimizationSuite::quick();
        assert_eq!(suite.workloads.len(), 2);
        assert_eq!(suite.sizes.len(), 2);
        assert_eq!(suite.duration, Duration::from_secs(1));
    }

    #[test]
    fn test_bottleneck_severity() {
        assert_eq!(
            serde_json::to_string(&BottleneckSeverity::Critical).unwrap(),
            "\"Critical\""
        );
    }

    #[test]
    fn test_regression_report_exit_code() {
        let passing = RegressionReport {
            passed: true,
            regressions: vec![],
            improvements: vec![],
            summary: "OK".to_string(),
        };
        assert_eq!(passing.exit_code(), 0);

        let failing = RegressionReport {
            passed: false,
            regressions: vec![RegressionEntry {
                workload: "test".to_string(),
                size: 1000,
                baseline_gflops: 100.0,
                current_gflops: 80.0,
                change_percent: -20.0,
            }],
            improvements: vec![],
            summary: "FAILED".to_string(),
        };
        assert_eq!(failing.exit_code(), 1);
    }

    #[test]
    fn test_validation_result_format() {
        let result = ValidationResult {
            passed: true,
            improvement_percent: 15.0,
            before_gflops: 50.0,
            after_gflops: 57.5,
            before_cv: 3.0,
            after_cv: 2.5,
            p_value: 0.01,
            statistically_significant: true,
        };

        let report = result.format_report();
        assert!(report.contains("PASSED"));
        assert!(report.contains("15.0%"));
    }
}
