//! Hardware Capability Detection (PMAT-447)
//!
//! Detects CPU SIMD capabilities, GPU presence, and calculates
//! theoretical peak performance for roofline analysis.
//!
//! Integrates with `pmat brick-score` for hardware-aware profiling.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// SIMD instruction set width
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimdWidth {
    /// No SIMD (scalar)
    Scalar,
    /// ARM NEON (128-bit, 4×f32)
    Neon128,
    /// SSE2 (128-bit, 4×f32)
    Sse2,
    /// AVX2 (256-bit, 8×f32)
    Avx2,
    /// AVX-512 (512-bit, 16×f32)
    Avx512,
    /// WebAssembly SIMD (128-bit, 4×f32)
    WasmSimd128,
}

impl SimdWidth {
    /// Number of f32 lanes
    pub fn lanes(&self) -> usize {
        match self {
            SimdWidth::Scalar => 1,
            SimdWidth::Neon128 | SimdWidth::Sse2 | SimdWidth::WasmSimd128 => 4,
            SimdWidth::Avx2 => 8,
            SimdWidth::Avx512 => 16,
        }
    }

    /// Bit width
    pub fn bits(&self) -> usize {
        self.lanes() * 32
    }

    /// Typical speedup factor for compute-bound operations
    pub fn compute_speedup(&self) -> f64 {
        match self {
            SimdWidth::Scalar => 1.0,
            SimdWidth::Neon128 | SimdWidth::Sse2 | SimdWidth::WasmSimd128 => 4.0,
            SimdWidth::Avx2 => 10.0,  // 8-12x measured in trueno-zram
            SimdWidth::Avx512 => 12.0, // 8-13x measured
        }
    }
}

/// GPU compute backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuBackend {
    /// No GPU available
    None,
    /// NVIDIA CUDA
    Cuda,
    /// WebGPU (cross-platform)
    Wgpu,
    /// Apple Metal
    Metal,
    /// Vulkan compute
    Vulkan,
}

/// CPU capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuCapability {
    /// CPU vendor (Intel, AMD, Apple, etc.)
    pub vendor: String,
    /// CPU model name
    pub model: String,
    /// Number of physical cores
    pub cores: usize,
    /// Number of logical threads
    pub threads: usize,
    /// Best available SIMD width
    pub simd: SimdWidth,
    /// Base frequency in GHz
    pub base_freq_ghz: f64,
    /// Theoretical peak GFLOP/s (FMA)
    pub peak_gflops: f64,
    /// Memory bandwidth in GB/s (estimated)
    pub memory_bw_gbps: f64,
}

/// GPU capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapability {
    /// GPU vendor
    pub vendor: String,
    /// GPU model name
    pub model: String,
    /// Compute backend
    pub backend: GpuBackend,
    /// CUDA compute capability (e.g., "8.9" for RTX 4090)
    pub compute_capability: Option<String>,
    /// Peak FP32 TFLOP/s
    pub peak_tflops_fp32: f64,
    /// Peak Tensor Core TFLOP/s (NVIDIA only)
    pub peak_tflops_tensor: Option<f64>,
    /// Memory bandwidth in GB/s
    pub memory_bw_gbps: f64,
    /// VRAM in GB
    pub vram_gb: f64,
}

/// Complete hardware capability profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapability {
    /// Detection timestamp
    pub timestamp: String,
    /// Hostname
    pub hostname: String,
    /// CPU capabilities
    pub cpu: CpuCapability,
    /// GPU capabilities (if present)
    pub gpu: Option<GpuCapability>,
    /// Roofline model parameters
    pub roofline: RooflineParams,
    /// PMAT-452: Byte budget configuration for compression/I/O workloads
    #[serde(default)]
    pub byte_budget: Option<crate::brick::ByteBudget>,
}

/// Roofline model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RooflineParams {
    /// CPU arithmetic intensity threshold (GFLOP/s ÷ GB/s)
    pub cpu_arithmetic_intensity: f64,
    /// GPU arithmetic intensity threshold
    pub gpu_arithmetic_intensity: Option<f64>,
}

impl HardwareCapability {
    /// Detect hardware capabilities at runtime
    pub fn detect() -> Self {
        let cpu = detect_cpu();
        let gpu = detect_gpu();

        let cpu_ai = cpu.peak_gflops / cpu.memory_bw_gbps;
        let gpu_ai = gpu.as_ref().map(|g| g.peak_tflops_fp32 * 1000.0 / g.memory_bw_gbps);
        // PMAT-452: Extract memory bandwidth before moving cpu
        let byte_budget_throughput = cpu.memory_bw_gbps.min(25.0);

        HardwareCapability {
            timestamp: chrono::Utc::now().to_rfc3339(),
            hostname: hostname::get()
                .map(|h| h.to_string_lossy().to_string())
                .unwrap_or_else(|_| "unknown".to_string()),
            cpu,
            gpu,
            roofline: RooflineParams {
                cpu_arithmetic_intensity: cpu_ai,
                gpu_arithmetic_intensity: gpu_ai,
            },
            // PMAT-452: Default byte budget based on memory bandwidth
            byte_budget: Some(crate::brick::ByteBudget::from_throughput(byte_budget_throughput)),
        }
    }

    /// Load from TOML file or detect if missing
    pub fn load_or_detect(path: &Path) -> Self {
        if path.exists() {
            if let Ok(content) = fs::read_to_string(path) {
                if let Ok(cap) = toml::from_str(&content) {
                    return cap;
                }
            }
        }
        let cap = Self::detect();
        // Try to cache it
        let _ = cap.save(path);
        cap
    }

    /// Save to TOML file
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(path, content)
    }

    /// Get the best available backend for a workload
    pub fn best_backend(&self) -> GpuBackend {
        self.gpu
            .as_ref()
            .map(|g| g.backend)
            .unwrap_or(GpuBackend::None)
    }

    /// Calculate expected throughput for a brick given its arithmetic intensity
    pub fn expected_throughput_gflops(&self, arithmetic_intensity: f64, use_gpu: bool) -> f64 {
        if use_gpu {
            if let Some(gpu) = &self.gpu {
                let memory_bound = gpu.memory_bw_gbps * arithmetic_intensity;
                let compute_bound = gpu.peak_tflops_fp32 * 1000.0;
                memory_bound.min(compute_bound)
            } else {
                self.cpu_expected_throughput(arithmetic_intensity)
            }
        } else {
            self.cpu_expected_throughput(arithmetic_intensity)
        }
    }

    fn cpu_expected_throughput(&self, arithmetic_intensity: f64) -> f64 {
        let memory_bound = self.cpu.memory_bw_gbps * arithmetic_intensity;
        let compute_bound = self.cpu.peak_gflops;
        memory_bound.min(compute_bound)
    }

    /// Determine if workload is memory-bound or compute-bound
    pub fn bottleneck(&self, arithmetic_intensity: f64, use_gpu: bool) -> Bottleneck {
        let threshold = if use_gpu {
            self.roofline.gpu_arithmetic_intensity.unwrap_or(f64::MAX)
        } else {
            self.roofline.cpu_arithmetic_intensity
        };

        if arithmetic_intensity < threshold {
            Bottleneck::Memory
        } else {
            Bottleneck::Compute
        }
    }
}

/// Workload bottleneck classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Bottleneck {
    /// Limited by memory bandwidth
    Memory,
    /// Limited by compute throughput
    Compute,
}

/// Detect CPU capabilities
fn detect_cpu() -> CpuCapability {
    let simd = detect_simd();
    let cores = num_cpus::get_physical();
    let threads = num_cpus::get();

    // Estimate frequency (fallback to 3.0 GHz if unknown)
    let base_freq_ghz = 3.0;

    // Calculate peak GFLOP/s: cores × lanes × 2 (FMA) × freq
    let peak_gflops = (cores as f64) * (simd.lanes() as f64) * 2.0 * base_freq_ghz;

    // Estimate memory bandwidth (DDR5-5600 dual channel ≈ 89.6 GB/s)
    let memory_bw_gbps = 80.0; // Conservative estimate

    CpuCapability {
        vendor: "Unknown".to_string(),
        model: "Unknown".to_string(),
        cores,
        threads,
        simd,
        base_freq_ghz,
        peak_gflops,
        memory_bw_gbps,
    }
}

/// Detect best available SIMD width
fn detect_simd() -> SimdWidth {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return SimdWidth::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdWidth::Avx2;
        }
        if is_x86_feature_detected!("sse2") {
            return SimdWidth::Sse2;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return SimdWidth::Neon128;
    }

    #[cfg(target_arch = "wasm32")]
    {
        return SimdWidth::WasmSimd128;
    }

    SimdWidth::Scalar
}

/// Detect GPU capabilities
fn detect_gpu() -> Option<GpuCapability> {
    // Check for CUDA first (highest performance)
    #[cfg(feature = "cuda")]
    {
        if let Some(gpu) = detect_cuda_gpu() {
            return Some(gpu);
        }
    }

    // Fallback: no GPU detected
    None
}

#[cfg(feature = "cuda")]
fn detect_cuda_gpu() -> Option<GpuCapability> {
    // This would use cuDeviceGetAttribute in a real implementation
    // For now, return None and let the caller provide GPU info
    None
}

/// Default hardware.toml path
pub fn default_hardware_path() -> std::path::PathBuf {
    #[cfg(feature = "hardware-detect")]
    {
        dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".pmat")
            .join("hardware.toml")
    }
    #[cfg(not(feature = "hardware-detect"))]
    {
        std::path::PathBuf::from(".pmat").join("hardware.toml")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let simd = detect_simd();
        // Should detect at least scalar
        assert!(simd.lanes() >= 1);
    }

    #[test]
    fn test_simd_lanes() {
        assert_eq!(SimdWidth::Scalar.lanes(), 1);
        assert_eq!(SimdWidth::Avx2.lanes(), 8);
        assert_eq!(SimdWidth::Avx512.lanes(), 16);
    }

    #[test]
    fn test_hardware_detection() {
        let cap = HardwareCapability::detect();
        assert!(cap.cpu.cores > 0);
        assert!(cap.cpu.peak_gflops > 0.0);
    }

    #[test]
    fn test_bottleneck_classification() {
        let cap = HardwareCapability::detect();

        // Low arithmetic intensity = memory bound
        assert_eq!(cap.bottleneck(0.1, false), Bottleneck::Memory);

        // High arithmetic intensity = compute bound
        assert_eq!(cap.bottleneck(1000.0, false), Bottleneck::Compute);
    }

    #[test]
    fn test_roofline_throughput() {
        let cap = HardwareCapability::detect();

        // Memory-bound workload
        let low_ai = cap.expected_throughput_gflops(0.1, false);
        // Compute-bound workload
        let high_ai = cap.expected_throughput_gflops(1000.0, false);

        // Low AI should give lower throughput (memory limited)
        assert!(low_ai < high_ai);
    }

    #[test]
    fn test_toml_roundtrip() {
        let cap = HardwareCapability::detect();
        let toml_str = toml::to_string_pretty(&cap).unwrap();
        let parsed: HardwareCapability = toml::from_str(&toml_str).unwrap();
        assert_eq!(cap.cpu.cores, parsed.cpu.cores);
    }

    #[test]
    fn test_byte_budget_in_hardware_toml() {
        let cap = HardwareCapability::detect();

        // Should have byte_budget populated
        assert!(cap.byte_budget.is_some());
        let budget = cap.byte_budget.unwrap();
        assert!(budget.gb_per_sec > 0.0);
        assert!(budget.gb_per_sec <= 25.0); // Capped at zstd max

        // Test TOML roundtrip preserves byte_budget
        let toml_str = toml::to_string_pretty(&cap).unwrap();
        assert!(toml_str.contains("[byte_budget]"));
        assert!(toml_str.contains("gb_per_sec"));

        let parsed: HardwareCapability = toml::from_str(&toml_str).unwrap();
        assert!(parsed.byte_budget.is_some());
        let parsed_budget = parsed.byte_budget.unwrap();
        assert!((parsed_budget.gb_per_sec - budget.gb_per_sec).abs() < 0.001);
    }

    #[test]
    fn test_byte_budget_backward_compat() {
        // Test that hardware.toml without byte_budget still parses
        let toml_without_budget = r#"
timestamp = "2026-01-13T18:00:00Z"
hostname = "test"

[cpu]
vendor = "Intel"
model = "Test CPU"
cores = 4
threads = 8
simd = "Avx2"
base_freq_ghz = 3.5
peak_gflops = 112.0
memory_bw_gbps = 80.0

[roofline]
cpu_arithmetic_intensity = 1.4
"#;
        let parsed: HardwareCapability = toml::from_str(toml_without_budget).unwrap();
        // byte_budget should be None when not in TOML (backward compat)
        assert!(parsed.byte_budget.is_none());
    }

    // Additional coverage tests

    #[test]
    fn test_simd_width_bits() {
        assert_eq!(SimdWidth::Scalar.bits(), 32);
        assert_eq!(SimdWidth::Sse2.bits(), 128);
        assert_eq!(SimdWidth::Neon128.bits(), 128);
        assert_eq!(SimdWidth::WasmSimd128.bits(), 128);
        assert_eq!(SimdWidth::Avx2.bits(), 256);
        assert_eq!(SimdWidth::Avx512.bits(), 512);
    }

    #[test]
    fn test_simd_width_compute_speedup() {
        assert!((SimdWidth::Scalar.compute_speedup() - 1.0).abs() < 0.01);
        assert!((SimdWidth::Sse2.compute_speedup() - 4.0).abs() < 0.01);
        assert!((SimdWidth::Neon128.compute_speedup() - 4.0).abs() < 0.01);
        assert!((SimdWidth::WasmSimd128.compute_speedup() - 4.0).abs() < 0.01);
        assert!((SimdWidth::Avx2.compute_speedup() - 10.0).abs() < 0.01);
        assert!((SimdWidth::Avx512.compute_speedup() - 12.0).abs() < 0.01);
    }

    #[test]
    fn test_best_backend() {
        let cap = HardwareCapability::detect();
        let backend = cap.best_backend();
        // Should be a valid backend (even if None)
        assert!(matches!(
            backend,
            GpuBackend::None | GpuBackend::Cuda | GpuBackend::Metal | GpuBackend::Vulkan | GpuBackend::Wgpu
        ));
    }

    #[test]
    fn test_load_or_detect_creates_new() {
        use std::path::PathBuf;
        let tmp_path = PathBuf::from("/tmp/trueno_test_nonexistent_12345.toml");
        // Ensure it doesn't exist
        let _ = std::fs::remove_file(&tmp_path);

        let cap = HardwareCapability::load_or_detect(&tmp_path);
        assert!(cap.cpu.cores > 0);

        // Cleanup
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_save_and_load() {
        use std::path::PathBuf;
        let tmp_path = PathBuf::from("/tmp/trueno_test_save_load.toml");

        let original = HardwareCapability::detect();
        original.save(&tmp_path).expect("Failed to save");

        let loaded = HardwareCapability::load_or_detect(&tmp_path);
        assert_eq!(original.cpu.cores, loaded.cpu.cores);
        assert_eq!(original.hostname, loaded.hostname);

        // Cleanup
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_expected_throughput_with_gpu() {
        // Create a capability with GPU
        let cap = HardwareCapability::detect();

        // Test GPU branch if available
        if cap.gpu.is_some() {
            let throughput_gpu = cap.expected_throughput_gflops(10.0, true);
            let throughput_cpu = cap.expected_throughput_gflops(10.0, false);
            // Both should be positive
            assert!(throughput_gpu > 0.0);
            assert!(throughput_cpu > 0.0);
        }

        // CPU path should always work
        let cpu_throughput = cap.expected_throughput_gflops(10.0, false);
        assert!(cpu_throughput > 0.0);
    }

    #[test]
    fn test_expected_throughput_with_fake_gpu() {
        // Create a capability with a fake GPU to test GPU branch
        let mut cap = HardwareCapability::detect();
        cap.gpu = Some(GpuCapability {
            vendor: "Test".to_string(),
            model: "Fake GPU".to_string(),
            backend: GpuBackend::Cuda,
            compute_capability: Some("8.9".to_string()),
            peak_tflops_fp32: 100.0,
            peak_tflops_tensor: Some(400.0),
            memory_bw_gbps: 1000.0,
            vram_gb: 24.0,
        });

        // This should exercise the GPU branch
        let throughput_gpu = cap.expected_throughput_gflops(10.0, true);
        let throughput_cpu = cap.expected_throughput_gflops(10.0, false);

        // GPU with 1000 GB/s and AI=10: memory_bound = 10000 GFLOPS
        // GPU compute = 100 TFLOPS = 100000 GFLOPS
        // Result should be min(10000, 100000) = 10000
        assert!((throughput_gpu - 10000.0).abs() < 1.0);
        assert!(throughput_cpu > 0.0);
    }

    #[test]
    fn test_load_invalid_toml() {
        use std::path::PathBuf;
        let tmp_path = PathBuf::from("/tmp/trueno_test_invalid.toml");

        // Write invalid TOML
        std::fs::write(&tmp_path, "this is not valid toml [[[").expect("Failed to write");

        // Should fall back to detect
        let cap = HardwareCapability::load_or_detect(&tmp_path);
        assert!(cap.cpu.cores > 0);

        // Cleanup
        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_expected_throughput_no_gpu_fallback() {
        // Create a capability without GPU
        let mut cap = HardwareCapability::detect();
        cap.gpu = None;

        // Request GPU but none available - should fallback to CPU
        let throughput_gpu_request = cap.expected_throughput_gflops(10.0, true);
        let throughput_cpu = cap.expected_throughput_gflops(10.0, false);

        // Both should give the same result since there's no GPU
        assert!((throughput_gpu_request - throughput_cpu).abs() < 0.001);
        assert!(throughput_cpu > 0.0);
    }
}
