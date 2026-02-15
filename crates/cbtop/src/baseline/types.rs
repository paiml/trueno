//! Baseline types: server baselines, GPU classes, grades, and health indicators.

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
