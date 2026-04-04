//! Roofline model implementation per Williams, Waterman & Patterson (2009) [4].
//! Supports hierarchical GPU roofline per Yang et al. (2020) [13].
//! Uses Empirical Roofline Toolkit (ERT) methodology [6].

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Floating-point precision levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Precision {
    Fp32,
    Fp16,
    Tf32,
    Int8,
    Bf16,
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Precision::Fp32 => write!(f, "FP32"),
            Precision::Fp16 => write!(f, "FP16 Tensor"),
            Precision::Tf32 => write!(f, "TF32 Tensor"),
            Precision::Int8 => write!(f, "INT8 Tensor"),
            Precision::Bf16 => write!(f, "BF16"),
        }
    }
}

/// Memory hierarchy levels for hierarchical roofline [13].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryLevel {
    L1Cache,
    L2Cache,
    Dram,
    Pcie,
}

impl std::fmt::Display for MemoryLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryLevel::L1Cache => write!(f, "L1 Cache"),
            MemoryLevel::L2Cache => write!(f, "L2 Cache"),
            MemoryLevel::Dram => write!(f, "DRAM"),
            MemoryLevel::Pcie => write!(f, "PCIe"),
        }
    }
}

/// Whether a kernel is compute-bound or memory-bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Bound {
    /// Below ridge point: memory bandwidth is the bottleneck.
    Memory { bandwidth_utilization: f64 },
    /// Above ridge point: compute throughput is the bottleneck.
    Compute { compute_utilization: f64 },
}

/// Roofline model for a specific hardware target.
/// Implements the Empirical Roofline Toolkit (ERT) methodology [6].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RooflineModel {
    /// Hardware target name (e.g., "RTX 4090", "AMD EPYC AVX2")
    pub target: String,
    /// Peak compute throughput (FLOP/s) per precision
    pub peak_compute: HashMap<Precision, f64>,
    /// Peak memory bandwidth (bytes/s) per memory level
    pub peak_bandwidth: HashMap<MemoryLevel, f64>,
}

impl RooflineModel {
    /// Compute the ridge point for a given precision and memory level.
    /// Ridge = peak_compute / peak_bandwidth (FLOP/byte).
    /// This is the arithmetic intensity where the kernel transitions
    /// from memory-bound to compute-bound.
    pub fn ridge_point(&self, precision: Precision, mem_level: MemoryLevel) -> Option<f64> {
        let compute = self.peak_compute.get(&precision)?;
        let bandwidth = self.peak_bandwidth.get(&mem_level)?;
        if *bandwidth <= 0.0 {
            return None;
        }
        Some(compute / bandwidth)
    }

    /// Compute the theoretical peak throughput at a given arithmetic intensity.
    /// throughput = min(peak_compute, AI * peak_bandwidth)
    pub fn theoretical_peak(
        &self,
        arithmetic_intensity: f64,
        precision: Precision,
        mem_level: MemoryLevel,
    ) -> Option<f64> {
        let compute = self.peak_compute.get(&precision)?;
        let bandwidth = self.peak_bandwidth.get(&mem_level)?;
        Some(compute.min(arithmetic_intensity * bandwidth))
    }

    /// Classify a kernel as compute-bound or memory-bound.
    pub fn classify(
        &self,
        arithmetic_intensity: f64,
        achieved_throughput: f64,
        precision: Precision,
        mem_level: MemoryLevel,
    ) -> Option<KernelRooflinePoint> {
        let ridge = self.ridge_point(precision, mem_level)?;
        let peak = self.theoretical_peak(arithmetic_intensity, precision, mem_level)?;
        let peak_compute = *self.peak_compute.get(&precision)?;

        let bound = if arithmetic_intensity < ridge {
            Bound::Memory { bandwidth_utilization: achieved_throughput / peak * 100.0 }
        } else {
            Bound::Compute { compute_utilization: achieved_throughput / peak_compute * 100.0 }
        };

        let efficiency = if peak > 0.0 { achieved_throughput / peak * 100.0 } else { 0.0 };

        let distance_to_ridge =
            if arithmetic_intensity > 0.0 { ridge / arithmetic_intensity } else { f64::INFINITY };

        Some(KernelRooflinePoint {
            arithmetic_intensity,
            achieved_throughput,
            peak_throughput: peak,
            efficiency,
            bound,
            distance_to_ridge,
        })
    }

    /// Create the RTX 4090 roofline model with spec values.
    pub fn rtx_4090() -> Self {
        let mut peak_compute = HashMap::new();
        peak_compute.insert(Precision::Fp32, 82.6e12); // 82.6 TFLOP/s
        peak_compute.insert(Precision::Fp16, 330.0e12); // 330 TFLOP/s (Tensor)
        peak_compute.insert(Precision::Tf32, 165.0e12); // 165 TFLOP/s (Tensor)
        peak_compute.insert(Precision::Int8, 660.0e12); // 660 TOP/s (Tensor)

        let mut peak_bandwidth = HashMap::new();
        peak_bandwidth.insert(MemoryLevel::L1Cache, 19.0e12); // ~19 TB/s
        peak_bandwidth.insert(MemoryLevel::L2Cache, 5.3e12); // ~5.3 TB/s
        peak_bandwidth.insert(MemoryLevel::Dram, 1008.0e9); // 1008 GB/s
        peak_bandwidth.insert(MemoryLevel::Pcie, 32.0e9); // 32 GB/s PCIe 4.0 x16

        RooflineModel {
            target: "NVIDIA GeForce RTX 4090 (SM 8.9)".to_string(),
            peak_compute,
            peak_bandwidth,
        }
    }

    /// Create a CPU AVX2+FMA roofline model.
    /// Assumes dual 256-bit FMA units (e.g., AMD EPYC / Intel Skylake).
    pub fn cpu_avx2(freq_ghz: f64, cores: usize, mem_bandwidth_gbps: f64) -> Self {
        // FP32: 2 FMA units * 8 floats * 2 (FMA = mul + add) * freq * cores
        let fp32_peak = 2.0 * 8.0 * 2.0 * freq_ghz * 1e9 * cores as f64;

        let mut peak_compute = HashMap::new();
        peak_compute.insert(Precision::Fp32, fp32_peak);

        let mut peak_bandwidth = HashMap::new();
        peak_bandwidth.insert(MemoryLevel::Dram, mem_bandwidth_gbps * 1e9);

        RooflineModel {
            target: format!("CPU AVX2+FMA ({cores} cores @ {freq_ghz} GHz)"),
            peak_compute,
            peak_bandwidth,
        }
    }

    /// Create a CPU AVX-512 roofline model.
    /// AVX-512: 2 FMA units * 16 floats * 2 (FMA) * freq * cores.
    pub fn cpu_avx512(freq_ghz: f64, cores: usize, mem_bandwidth_gbps: f64) -> Self {
        let fp32_peak = 2.0 * 16.0 * 2.0 * freq_ghz * 1e9 * cores as f64;

        let mut peak_compute = HashMap::new();
        peak_compute.insert(Precision::Fp32, fp32_peak);

        let mut peak_bandwidth = HashMap::new();
        peak_bandwidth.insert(MemoryLevel::Dram, mem_bandwidth_gbps * 1e9);

        RooflineModel {
            target: format!("CPU AVX-512+FMA ({cores} cores @ {freq_ghz} GHz)"),
            peak_compute,
            peak_bandwidth,
        }
    }

    /// Create an ARM NEON roofline model.
    /// NEON: 2 FMA units * 4 floats * 2 (FMA) * freq * cores (typical A76/A78).
    pub fn cpu_neon(freq_ghz: f64, cores: usize, mem_bandwidth_gbps: f64) -> Self {
        let fp32_peak = 2.0 * 4.0 * 2.0 * freq_ghz * 1e9 * cores as f64;

        let mut peak_compute = HashMap::new();
        peak_compute.insert(Precision::Fp32, fp32_peak);

        let mut peak_bandwidth = HashMap::new();
        peak_bandwidth.insert(MemoryLevel::Dram, mem_bandwidth_gbps * 1e9);

        RooflineModel {
            target: format!("CPU NEON ({cores} cores @ {freq_ghz} GHz)"),
            peak_compute,
            peak_bandwidth,
        }
    }
}

/// A kernel's position on the roofline chart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelRooflinePoint {
    /// FLOPs per byte transferred
    pub arithmetic_intensity: f64,
    /// Achieved throughput (FLOP/s)
    pub achieved_throughput: f64,
    /// Roofline ceiling throughput (FLOP/s)
    pub peak_throughput: f64,
    /// Achieved / peak percentage
    pub efficiency: f64,
    /// Compute or memory bound classification
    pub bound: Bound,
    /// Ridge point / arithmetic_intensity (>1 = memory-bound)
    pub distance_to_ridge: f64,
}

/// Run the `cgp roofline` command.
pub fn run_roofline(
    target: &str,
    _kernels: Option<&str>,
    export: Option<&str>,
    _empirical: bool,
    json: bool,
) -> Result<()> {
    let model = match target {
        "cuda" => RooflineModel::rtx_4090(),
        "avx2" => {
            let cores = num_cpus::get_physical();
            RooflineModel::cpu_avx2(3.5, cores, 204.8)
        }
        "avx512" => {
            let cores = num_cpus::get_physical();
            RooflineModel::cpu_avx512(3.5, cores, 204.8)
        }
        "neon" => {
            let cores = num_cpus::get_physical();
            RooflineModel::cpu_neon(3.0, cores, 51.2)
        }
        "wgpu" => RooflineModel::rtx_4090(), // wgpu uses same GPU hardware
        other => anyhow::bail!(
            "Unknown roofline target: {other}. Supported: cuda, avx2, avx512, neon, wgpu"
        ),
    };

    if json {
        let json_str = serde_json::to_string_pretty(&model)?;
        println!("{json_str}");
        return Ok(());
    }

    println!("\n=== cgp Roofline: {} ===\n", model.target);

    // Print peak compute per precision
    println!("  Peak Compute:");
    let mut precisions: Vec<_> = model.peak_compute.iter().collect();
    precisions.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (prec, peak) in &precisions {
        println!("    {prec:15}: {:8.1} TFLOP/s", *peak / 1e12);
    }

    println!("\n  Peak Bandwidth:");
    let mut levels: Vec<_> = model.peak_bandwidth.iter().collect();
    levels.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (level, bw) in &levels {
        if **bw >= 1e12 {
            println!("    {level:15}: {:8.1} TB/s", *bw / 1e12);
        } else {
            println!("    {level:15}: {:8.1} GB/s", *bw / 1e9);
        }
    }

    println!("\n  Ridge Points (vs DRAM):");
    for (prec, _) in &precisions {
        if let Some(ridge) = model.ridge_point(**prec, MemoryLevel::Dram) {
            println!("    {prec:15}: {:8.1} FLOP/byte", ridge);
        }
    }

    if let Some(path) = export {
        let json_str = serde_json::to_string_pretty(&model)?;
        std::fs::write(path, json_str)?;
        println!("\n  Exported to: {path}");
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-CGP-021: Ridge point must be correctly computed.
    /// Given: peak_compute = 330 TFLOP/s, peak_bandwidth = 1008 GB/s
    /// Then: ridge_point = 330000 / 1008 = 327.4 FLOP/byte (within 1%)
    #[test]
    fn test_ridge_point_rtx4090_fp16() {
        let model = RooflineModel::rtx_4090();
        let ridge = model.ridge_point(Precision::Fp16, MemoryLevel::Dram).unwrap();
        let expected = 330_000.0 / 1008.0; // 327.38...
        assert!(
            (ridge - expected).abs() < 0.5,
            "Ridge point {ridge:.1} not within 0.5 of expected {expected:.1}"
        );
    }

    /// FALSIFY-CGP-021: All precision ridge points match manual calculation.
    #[test]
    fn test_ridge_points_all_precisions() {
        let model = RooflineModel::rtx_4090();
        let dram_bw = 1008.0e9;

        let cases = [
            (Precision::Fp32, 82.6e12),
            (Precision::Fp16, 330.0e12),
            (Precision::Tf32, 165.0e12),
            (Precision::Int8, 660.0e12),
        ];

        for (prec, peak) in cases {
            let ridge = model.ridge_point(prec, MemoryLevel::Dram).unwrap();
            let expected = peak / dram_bw;
            assert!(
                (ridge - expected).abs() / expected < 0.001,
                "{prec}: ridge {ridge:.2} != expected {expected:.2}"
            );
        }
    }

    /// FALSIFY-ROOF-002: Memory-bound kernel classified correctly.
    #[test]
    fn test_memory_bound_classification() {
        let model = RooflineModel::rtx_4090();
        // AI = 8.0 FLOP/byte, well below ridge of 327.4
        let point = model.classify(8.0, 5e12, Precision::Fp16, MemoryLevel::Dram).unwrap();
        assert!(matches!(point.bound, Bound::Memory { .. }));
        assert!(point.distance_to_ridge > 1.0);
    }

    /// FALSIFY-ROOF-003: Compute-bound kernel classified correctly.
    #[test]
    fn test_compute_bound_classification() {
        let model = RooflineModel::rtx_4090();
        // AI = 500.0 FLOP/byte, above ridge of 327.4
        let point = model.classify(500.0, 300e12, Precision::Fp16, MemoryLevel::Dram).unwrap();
        assert!(matches!(point.bound, Bound::Compute { .. }));
        assert!(point.distance_to_ridge < 1.0);
    }

    /// Theoretical peak follows min(compute, AI*bandwidth).
    #[test]
    fn test_theoretical_peak() {
        let model = RooflineModel::rtx_4090();
        // Memory-bound region: peak = AI * bandwidth
        let low_ai = model.theoretical_peak(8.0, Precision::Fp16, MemoryLevel::Dram).unwrap();
        assert!((low_ai - 8.0 * 1008.0e9).abs() / low_ai < 0.001);

        // Compute-bound region: peak = compute peak
        let high_ai = model.theoretical_peak(500.0, Precision::Fp16, MemoryLevel::Dram).unwrap();
        assert!((high_ai - 330.0e12).abs() / high_ai < 0.001);
    }

    /// CPU AVX2 model: peak = 2 FMA units * 8 floats * 2 ops * freq * cores.
    #[test]
    fn test_cpu_avx2_peak() {
        let model = RooflineModel::cpu_avx2(3.5, 8, 51.2);
        let fp32_peak = *model.peak_compute.get(&Precision::Fp32).unwrap();
        let expected = 2.0 * 8.0 * 2.0 * 3.5e9 * 8.0; // 896 GFLOP/s
        assert!(
            (fp32_peak - expected).abs() / expected < 0.001,
            "FP32 peak {:.1} GFLOP/s != expected {:.1} GFLOP/s",
            fp32_peak / 1e9,
            expected / 1e9
        );
    }

    /// RTX 4090 bandwidth spec: 384-bit * 21 Gbps = 1008 GB/s.
    #[test]
    fn test_rtx4090_bandwidth_spec() {
        let model = RooflineModel::rtx_4090();
        let dram = *model.peak_bandwidth.get(&MemoryLevel::Dram).unwrap();
        assert!(
            (dram - 1008.0e9).abs() < 1e6,
            "DRAM bandwidth {:.1} GB/s != 1008.0 GB/s",
            dram / 1e9
        );
    }
}
