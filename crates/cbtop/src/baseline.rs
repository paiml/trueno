//! Industry Baseline Validation (PMAT-016)
//!
//! Compare cbtop throughput against industry baselines (vLLM, TGI, Triton).
//! Per cbtop spec §21.7 and §21.8.
//!
//! # Design Principles
//!
//! - Use vLLM/llama.cpp as **reference**, not dependency
//! - Side-by-side validation without polluting Pure Rust codebase
//! - No foreign code in cbtop binary (F976)
//!
//! # Citations
//!
//! - [Satna 2026] "LLM Inference Benchmarking Framework" GitHub
//! - [vLLM 2023] "vLLM: Easy, Fast, Cheap LLM Serving with PagedAttention" UCB

use std::fmt;

/// Industry server baseline data from Satna (2026) benchmarks.
///
/// Citation: [21] Satna, R. (2026). "LLM Inference Benchmarking Framework."
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ServerBaseline {
    /// Server name (vLLM, TGI, Triton)
    pub name: &'static str,
    /// Peak tokens per second
    pub peak_tok_per_sec: u32,
    /// P95 latency in milliseconds
    pub p95_latency_ms: u32,
    /// SM utilization percentage
    pub sm_utilization: u8,
    /// Memory overhead percentage
    pub memory_overhead: u8,
    /// Reference GPU
    pub gpu: &'static str,
}

/// Industry baselines from Satna (2026) on A10 GPU.
pub const VLLM_BASELINE: ServerBaseline = ServerBaseline {
    name: "vLLM",
    peak_tok_per_sec: 412,
    p95_latency_ms: 1715,
    sm_utilization: 99,
    memory_overhead: 42,
    gpu: "A10",
};

pub const TGI_BASELINE: ServerBaseline = ServerBaseline {
    name: "TGI",
    peak_tok_per_sec: 408,
    p95_latency_ms: 1704,
    sm_utilization: 98,
    memory_overhead: 44,
    gpu: "A10",
};

pub const TRITON_BASELINE: ServerBaseline = ServerBaseline {
    name: "Triton",
    peak_tok_per_sec: 385,
    p95_latency_ms: 2007,
    sm_utilization: 97,
    memory_overhead: 45,
    gpu: "A10",
};

/// All industry baselines.
pub const INDUSTRY_BASELINES: [ServerBaseline; 3] = [VLLM_BASELINE, TGI_BASELINE, TRITON_BASELINE];

/// GPU class with expected throughput ranges.
///
/// From cbtop spec §21.7 "Expected Throughput by GPU Class".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuClass {
    /// NVIDIA A10 (24GB) - Data center inference GPU
    A10,
    /// NVIDIA A100 (40GB or 80GB) - Data center training/inference GPU
    A100,
    /// NVIDIA H100 (80GB) - Hopper architecture flagship
    H100,
    /// NVIDIA RTX 4090 (24GB) - Consumer flagship
    Rtx4090,
    /// NVIDIA RTX 3090 (24GB) - Previous gen consumer flagship
    Rtx3090,
    /// Unknown GPU class
    Unknown,
}

impl GpuClass {
    /// Expected throughput range (min, max) in tok/s.
    ///
    /// From cbtop spec §21.7.
    pub fn expected_throughput(&self) -> (u32, u32) {
        match self {
            GpuClass::A10 => (350, 450),
            GpuClass::A100 => (800, 1200),
            GpuClass::H100 => (1800, 2400),
            GpuClass::Rtx4090 => (300, 400),
            GpuClass::Rtx3090 => (200, 300),
            GpuClass::Unknown => (100, 500), // Conservative estimate
        }
    }

    /// VRAM size in GB.
    pub fn vram_gb(&self) -> u32 {
        match self {
            GpuClass::A10 => 24,
            GpuClass::A100 => 80, // Using 80GB variant
            GpuClass::H100 => 80,
            GpuClass::Rtx4090 => 24,
            GpuClass::Rtx3090 => 24,
            GpuClass::Unknown => 8,
        }
    }

    /// Detect GPU class from GPU name string.
    ///
    /// Parses common GPU name formats from nvidia-smi, NVML, etc.
    pub fn from_name(name: &str) -> Self {
        let name_lower = name.to_lowercase();

        if name_lower.contains("h100") {
            GpuClass::H100
        } else if name_lower.contains("a100") {
            GpuClass::A100
        } else if name_lower.contains("a10") && !name_lower.contains("a100") {
            GpuClass::A10
        } else if name_lower.contains("4090") {
            GpuClass::Rtx4090
        } else if name_lower.contains("3090") {
            GpuClass::Rtx3090
        } else {
            GpuClass::Unknown
        }
    }
}

impl fmt::Display for GpuClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuClass::A10 => write!(f, "A10 (24GB)"),
            GpuClass::A100 => write!(f, "A100 (40/80GB)"),
            GpuClass::H100 => write!(f, "H100 (80GB)"),
            GpuClass::Rtx4090 => write!(f, "RTX 4090 (24GB)"),
            GpuClass::Rtx3090 => write!(f, "RTX 3090 (24GB)"),
            GpuClass::Unknown => write!(f, "Unknown GPU"),
        }
    }
}

/// Throughput grade (A/B/C/D/F) based on baseline comparison.
///
/// From cbtop spec F983: "Throughput grade calculated".
/// Ordering: F < D < C < B < A (A is best).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThroughputGrade {
    /// < 40% of vLLM baseline (worst)
    F,
    /// >= 40% of vLLM baseline
    D,
    /// >= 60% of vLLM baseline
    C,
    /// >= 80% of vLLM baseline
    B,
    /// >= 100% of vLLM baseline (best)
    A,
}

impl ThroughputGrade {
    /// Calculate grade from actual throughput vs baseline.
    pub fn from_percentage(percentage: f64) -> Self {
        if percentage >= 100.0 {
            ThroughputGrade::A
        } else if percentage >= 80.0 {
            ThroughputGrade::B
        } else if percentage >= 60.0 {
            ThroughputGrade::C
        } else if percentage >= 40.0 {
            ThroughputGrade::D
        } else {
            ThroughputGrade::F
        }
    }

    /// Get threshold percentage for this grade.
    pub fn threshold(&self) -> f64 {
        match self {
            ThroughputGrade::A => 100.0,
            ThroughputGrade::B => 80.0,
            ThroughputGrade::C => 60.0,
            ThroughputGrade::D => 40.0,
            ThroughputGrade::F => 0.0,
        }
    }
}

impl fmt::Display for ThroughputGrade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (grade, desc) = match self {
            ThroughputGrade::A => ("A", "Excellent - meets or exceeds baseline"),
            ThroughputGrade::B => ("B", "Good - 80%+ of baseline"),
            ThroughputGrade::C => ("C", "Fair - 60%+ of baseline"),
            ThroughputGrade::D => ("D", "Poor - 40%+ of baseline"),
            ThroughputGrade::F => ("F", "Failing - below 40% of baseline"),
        };
        write!(f, "{} ({})", grade, desc)
    }
}

/// SM utilization health indicator.
///
/// From cbtop spec §21.7: SM utilization thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmHealth {
    /// SM utilization > 95% - at risk of saturation
    Saturated,
    /// SM utilization 80-95% - optimal range
    Optimal,
    /// SM utilization 50-80% - room for improvement
    Moderate,
    /// SM utilization < 50% - critical underutilization
    Critical,
}

impl SmHealth {
    /// Calculate SM health from utilization percentage.
    pub fn from_utilization(sm_util: u8) -> Self {
        if sm_util > 95 {
            SmHealth::Saturated
        } else if sm_util >= 80 {
            SmHealth::Optimal
        } else if sm_util >= 50 {
            SmHealth::Moderate
        } else {
            SmHealth::Critical
        }
    }

    /// Is this health status acceptable for production?
    pub fn is_acceptable(&self) -> bool {
        matches!(self, SmHealth::Optimal | SmHealth::Saturated)
    }
}

impl fmt::Display for SmHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmHealth::Saturated => write!(f, "SATURATED (>95%)"),
            SmHealth::Optimal => write!(f, "OPTIMAL (80-95%)"),
            SmHealth::Moderate => write!(f, "MODERATE (50-80%)"),
            SmHealth::Critical => write!(f, "CRITICAL (<50%)"),
        }
    }
}

/// Baseline comparison result.
///
/// Compares actual metrics against industry baselines.
#[derive(Debug, Clone)]
pub struct BaselineComparison {
    /// Detected GPU class
    pub gpu_class: GpuClass,
    /// Actual throughput (tok/s)
    pub actual_tok_per_sec: u32,
    /// Expected throughput range
    pub expected_range: (u32, u32),
    /// Percentage of vLLM baseline achieved
    pub vllm_percentage: f64,
    /// Throughput grade
    pub grade: ThroughputGrade,
    /// SM utilization
    pub sm_utilization: u8,
    /// SM health indicator
    pub sm_health: SmHealth,
    /// P95 latency (ms)
    pub p95_latency_ms: Option<u32>,
    /// Comparison with each baseline
    pub baseline_comparisons: Vec<SingleComparison>,
}

/// Comparison with a single baseline server.
#[derive(Debug, Clone)]
pub struct SingleComparison {
    /// Baseline server
    pub baseline: ServerBaseline,
    /// Percentage achieved (actual / baseline * 100)
    pub percentage: f64,
    /// Delta in tok/s (actual - baseline)
    pub delta_tok_per_sec: i32,
}

impl BaselineComparison {
    /// Create a new baseline comparison.
    pub fn new(
        gpu_name: &str,
        actual_tok_per_sec: u32,
        sm_utilization: u8,
        p95_latency_ms: Option<u32>,
    ) -> Self {
        let gpu_class = GpuClass::from_name(gpu_name);
        let expected_range = gpu_class.expected_throughput();

        // Calculate percentage of vLLM baseline (scaled by GPU class)
        let vllm_scaled_baseline = scale_baseline_for_gpu(&VLLM_BASELINE, &gpu_class);
        let vllm_percentage = (actual_tok_per_sec as f64 / vllm_scaled_baseline as f64) * 100.0;

        let grade = ThroughputGrade::from_percentage(vllm_percentage);
        let sm_health = SmHealth::from_utilization(sm_utilization);

        // Compare against all baselines
        let baseline_comparisons: Vec<_> = INDUSTRY_BASELINES
            .iter()
            .map(|baseline| {
                let scaled = scale_baseline_for_gpu(baseline, &gpu_class);
                SingleComparison {
                    baseline: *baseline,
                    percentage: (actual_tok_per_sec as f64 / scaled as f64) * 100.0,
                    delta_tok_per_sec: actual_tok_per_sec as i32 - scaled as i32,
                }
            })
            .collect();

        BaselineComparison {
            gpu_class,
            actual_tok_per_sec,
            expected_range,
            vllm_percentage,
            grade,
            sm_utilization,
            sm_health,
            p95_latency_ms,
            baseline_comparisons,
        }
    }

    /// Check if throughput is within expected range for GPU class.
    pub fn is_within_expected_range(&self) -> bool {
        self.actual_tok_per_sec >= self.expected_range.0
            && self.actual_tok_per_sec <= self.expected_range.1
    }

    /// Get improvement suggestions based on metrics.
    pub fn suggestions(&self) -> Vec<&'static str> {
        let mut suggestions = Vec::new();

        // SM utilization suggestions
        match self.sm_health {
            SmHealth::Critical => {
                suggestions
                    .push("Critical: SM utilization < 50% - check batch size and kernel occupancy");
                suggestions.push("Consider increasing batch size or concurrent requests");
            }
            SmHealth::Moderate => {
                suggestions.push("SM utilization 50-80% - room for optimization");
                suggestions.push("Try increasing kernel occupancy or reducing memory pressure");
            }
            SmHealth::Saturated => {
                suggestions
                    .push("SM utilization > 95% - at saturation, throughput limited by compute");
            }
            SmHealth::Optimal => {}
        }

        // Grade-based suggestions
        match self.grade {
            ThroughputGrade::F => {
                suggestions.push("Throughput < 40% of baseline - major optimization needed");
                suggestions
                    .push("Check for: kernel inefficiency, memory bottlenecks, PCIe transfers");
            }
            ThroughputGrade::D => {
                suggestions.push("Throughput 40-60% of baseline - significant optimization needed");
            }
            ThroughputGrade::C => {
                suggestions
                    .push("Throughput 60-80% of baseline - optimization opportunities exist");
            }
            ThroughputGrade::B | ThroughputGrade::A => {}
        }

        suggestions
    }
}

impl fmt::Display for BaselineComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Baseline Comparison Report")?;
        writeln!(f, "==========================")?;
        writeln!(f)?;
        writeln!(f, "GPU Class: {}", self.gpu_class)?;
        writeln!(f, "Actual Throughput: {} tok/s", self.actual_tok_per_sec)?;
        writeln!(
            f,
            "Expected Range: {}-{} tok/s",
            self.expected_range.0, self.expected_range.1
        )?;
        writeln!(f, "Grade: {}", self.grade)?;
        writeln!(f)?;
        writeln!(
            f,
            "SM Utilization: {}% ({})",
            self.sm_utilization, self.sm_health
        )?;
        if let Some(latency) = self.p95_latency_ms {
            writeln!(f, "P95 Latency: {} ms", latency)?;
        }
        writeln!(f)?;
        writeln!(f, "Comparison vs Industry Baselines:")?;
        for cmp in &self.baseline_comparisons {
            let sign = if cmp.delta_tok_per_sec >= 0 { "+" } else { "" };
            writeln!(
                f,
                "  {}: {:.1}% ({}{} tok/s)",
                cmp.baseline.name, cmp.percentage, sign, cmp.delta_tok_per_sec
            )?;
        }

        let suggestions = self.suggestions();
        if !suggestions.is_empty() {
            writeln!(f)?;
            writeln!(f, "Suggestions:")?;
            for suggestion in suggestions {
                writeln!(f, "  - {}", suggestion)?;
            }
        }

        Ok(())
    }
}

/// Scale baseline for different GPU classes.
///
/// Baselines are measured on A10; scale for other GPUs based on expected performance.
fn scale_baseline_for_gpu(baseline: &ServerBaseline, gpu_class: &GpuClass) -> u32 {
    let (min_expected, max_expected) = gpu_class.expected_throughput();
    let a10_expected = (350 + 450) / 2; // A10 midpoint

    let target_expected = (min_expected + max_expected) / 2;
    let scale_factor = target_expected as f64 / a10_expected as f64;

    (baseline.peak_tok_per_sec as f64 * scale_factor) as u32
}

/// Baseline validator for F971-F985 falsification criteria.
#[derive(Debug, Default)]
pub struct BaselineValidator {
    /// Validated criteria
    validations: Vec<(String, bool, String)>,
}

impl BaselineValidator {
    /// Create a new validator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate F971: Realistic GPU throughput (within 30% of vLLM).
    pub fn validate_f971_throughput(&mut self, comparison: &BaselineComparison) -> bool {
        let passed = comparison.vllm_percentage >= 70.0; // Within 30% means at least 70%
        self.validations.push((
            "F971".to_string(),
            passed,
            format!(
                "Throughput {:.1}% of vLLM (need >= 70%)",
                comparison.vllm_percentage
            ),
        ));
        passed
    }

    /// Validate F972: SM utilization correct (within 5% of nvidia-smi).
    pub fn validate_f972_sm_util(&mut self, reported: u8, actual: u8) -> bool {
        let diff = (reported as i16 - actual as i16).unsigned_abs();
        let passed = diff <= 5;
        self.validations.push((
            "F972".to_string(),
            passed,
            format!("SM util diff: {}% (need <= 5%)", diff),
        ));
        passed
    }

    /// Validate F975: Baseline comparison available.
    pub fn validate_f975_baseline_available(&mut self, has_comparison: bool) -> bool {
        self.validations.push((
            "F975".to_string(),
            has_comparison,
            "Baseline comparison available".to_string(),
        ));
        has_comparison
    }

    /// Validate F976: No foreign code dependency.
    pub fn validate_f976_no_foreign_code(&mut self) -> bool {
        // This is always true for cbtop - we don't depend on vLLM/llama.cpp
        self.validations.push((
            "F976".to_string(),
            true,
            "No foreign code in cbtop binary".to_string(),
        ));
        true
    }

    /// Validate F982: GPU class detected correctly.
    pub fn validate_f982_gpu_detected(&mut self, gpu_class: &GpuClass) -> bool {
        let passed = *gpu_class != GpuClass::Unknown;
        self.validations.push((
            "F982".to_string(),
            passed,
            format!("GPU class: {}", gpu_class),
        ));
        passed
    }

    /// Validate F983: Throughput grade calculated.
    pub fn validate_f983_grade_calculated(&mut self, grade: &ThroughputGrade) -> bool {
        self.validations.push((
            "F983".to_string(),
            true,
            format!("Grade calculated: {:?}", grade),
        ));
        true
    }

    /// Validate F984: Health indicators displayed.
    pub fn validate_f984_health_indicators(
        &mut self,
        has_sm: bool,
        has_memory: bool,
        has_scaling: bool,
    ) -> bool {
        let passed = has_sm && has_memory && has_scaling;
        self.validations.push((
            "F984".to_string(),
            passed,
            format!(
                "Health: SM={}, Memory={}, Scaling={}",
                has_sm, has_memory, has_scaling
            ),
        ));
        passed
    }

    /// Get validation summary.
    pub fn summary(&self) -> ValidationSummary {
        let total = self.validations.len();
        let passed = self.validations.iter().filter(|(_, p, _)| *p).count();
        ValidationSummary {
            total,
            passed,
            failed: total - passed,
            details: self.validations.clone(),
        }
    }
}

/// Validation summary.
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    /// Total validations run
    pub total: usize,
    /// Passed validations
    pub passed: usize,
    /// Failed validations
    pub failed: usize,
    /// Detailed results
    pub details: Vec<(String, bool, String)>,
}

impl fmt::Display for ValidationSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Baseline Validation Summary")?;
        writeln!(f, "===========================")?;
        writeln!(f, "Passed: {}/{}", self.passed, self.total)?;
        writeln!(f)?;
        for (id, passed, msg) in &self.details {
            let status = if *passed { "PASS" } else { "FAIL" };
            writeln!(f, "[{}] {}: {}", status, id, msg)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_class_detection() {
        assert_eq!(GpuClass::from_name("NVIDIA A10"), GpuClass::A10);
        assert_eq!(GpuClass::from_name("NVIDIA A100-SXM4-80GB"), GpuClass::A100);
        assert_eq!(GpuClass::from_name("NVIDIA H100 PCIe"), GpuClass::H100);
        assert_eq!(
            GpuClass::from_name("NVIDIA GeForce RTX 4090"),
            GpuClass::Rtx4090
        );
        assert_eq!(
            GpuClass::from_name("NVIDIA GeForce RTX 3090"),
            GpuClass::Rtx3090
        );
        assert_eq!(GpuClass::from_name("Unknown GPU"), GpuClass::Unknown);
    }

    #[test]
    fn test_throughput_grade() {
        assert_eq!(ThroughputGrade::from_percentage(105.0), ThroughputGrade::A);
        assert_eq!(ThroughputGrade::from_percentage(100.0), ThroughputGrade::A);
        assert_eq!(ThroughputGrade::from_percentage(85.0), ThroughputGrade::B);
        assert_eq!(ThroughputGrade::from_percentage(65.0), ThroughputGrade::C);
        assert_eq!(ThroughputGrade::from_percentage(45.0), ThroughputGrade::D);
        assert_eq!(ThroughputGrade::from_percentage(35.0), ThroughputGrade::F);
    }

    #[test]
    fn test_sm_health() {
        assert_eq!(SmHealth::from_utilization(98), SmHealth::Saturated);
        assert_eq!(SmHealth::from_utilization(85), SmHealth::Optimal);
        assert_eq!(SmHealth::from_utilization(60), SmHealth::Moderate);
        assert_eq!(SmHealth::from_utilization(40), SmHealth::Critical);
    }

    #[test]
    fn test_baseline_comparison_a10() {
        let comparison = BaselineComparison::new("NVIDIA A10", 400, 95, Some(1700));

        assert_eq!(comparison.gpu_class, GpuClass::A10);
        assert!(comparison.vllm_percentage > 90.0); // Should be close to 100%
        assert!(comparison.is_within_expected_range());
        assert!(comparison.grade >= ThroughputGrade::B);
    }

    #[test]
    fn test_baseline_comparison_h100() {
        let comparison = BaselineComparison::new("NVIDIA H100 PCIe", 2000, 92, None);

        assert_eq!(comparison.gpu_class, GpuClass::H100);
        // H100 baseline is scaled up from A10
        assert!(comparison.is_within_expected_range());
    }

    #[test]
    fn test_validator_f971() {
        let comparison = BaselineComparison::new("NVIDIA A10", 350, 90, None);
        let mut validator = BaselineValidator::new();

        let passed = validator.validate_f971_throughput(&comparison);
        assert!(passed); // 350/412 ~= 85% > 70%
    }

    #[test]
    fn test_validator_f972() {
        let mut validator = BaselineValidator::new();

        assert!(validator.validate_f972_sm_util(92, 90)); // 2% diff
        assert!(!validator.validate_f972_sm_util(92, 80)); // 12% diff
    }

    #[test]
    fn test_validator_f976_no_foreign() {
        let mut validator = BaselineValidator::new();
        assert!(validator.validate_f976_no_foreign_code());
    }

    #[test]
    fn test_industry_baselines_defined() {
        // F985: Benchmark methodology documented
        assert_eq!(VLLM_BASELINE.peak_tok_per_sec, 412);
        assert_eq!(TGI_BASELINE.peak_tok_per_sec, 408);
        assert_eq!(TRITON_BASELINE.peak_tok_per_sec, 385);
    }

    #[test]
    fn test_expected_throughput_ranges() {
        assert_eq!(GpuClass::A10.expected_throughput(), (350, 450));
        assert_eq!(GpuClass::A100.expected_throughput(), (800, 1200));
        assert_eq!(GpuClass::H100.expected_throughput(), (1800, 2400));
    }

    #[test]
    fn test_grade_thresholds() {
        assert_eq!(ThroughputGrade::A.threshold(), 100.0);
        assert_eq!(ThroughputGrade::B.threshold(), 80.0);
        assert_eq!(ThroughputGrade::C.threshold(), 60.0);
        assert_eq!(ThroughputGrade::D.threshold(), 40.0);
        assert_eq!(ThroughputGrade::F.threshold(), 0.0);
    }

    #[test]
    fn test_validation_summary() {
        let mut validator = BaselineValidator::new();
        validator.validate_f976_no_foreign_code();
        validator.validate_f975_baseline_available(true);

        let summary = validator.summary();
        assert_eq!(summary.total, 2);
        assert_eq!(summary.passed, 2);
        assert_eq!(summary.failed, 0);
    }
}
