//! Hardware detection and system metrics types.

/// Hardware information detected at startup
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    /// CPU model name
    pub cpu_model: String,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// SIMD capability
    pub simd_type: &'static str,
    /// GPU name (if available)
    pub gpu_name: Option<String>,
    /// Total system memory in GB
    pub memory_gb: f64,
}

impl HardwareInfo {
    /// Detect hardware at startup
    pub fn detect() -> Self {
        let cpu_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        // Detect SIMD capability
        let simd_type = Self::detect_simd();

        // Try to get CPU model from /proc/cpuinfo
        let cpu_model = Self::read_cpu_model().unwrap_or_else(|| "Unknown CPU".to_string());

        // Try to get GPU name
        let gpu_name = Self::detect_gpu();

        // Get total memory
        let memory_gb = Self::read_memory_gb();

        Self {
            cpu_model,
            cpu_cores,
            simd_type,
            gpu_name,
            memory_gb,
        }
    }

    fn detect_simd() -> &'static str {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return "AVX-512";
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return "AVX2";
            }
            if std::arch::is_x86_feature_detected!("avx") {
                return "AVX";
            }
            if std::arch::is_x86_feature_detected!("sse4.2") {
                return "SSE4.2";
            }
            "SSE2"
        }
        #[cfg(target_arch = "aarch64")]
        {
            "NEON"
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            "Scalar"
        }
    }

    fn read_cpu_model() -> Option<String> {
        #[cfg(target_os = "linux")]
        {
            let contents = std::fs::read_to_string("/proc/cpuinfo").ok()?;
            for line in contents.lines() {
                if line.starts_with("model name") {
                    return line.split(':').nth(1).map(|s| s.trim().to_string());
                }
            }
        }
        #[cfg(target_os = "macos")]
        {
            // Use sysctl on macOS
            let output = std::process::Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
                .output()
                .ok()?;
            return String::from_utf8(output.stdout)
                .ok()
                .map(|s| s.trim().to_string());
        }
        None
    }

    fn detect_gpu() -> Option<String> {
        #[cfg(target_os = "linux")]
        {
            // Try nvidia-smi first
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=name", "--format=csv,noheader"])
                .output()
            {
                if output.status.success() {
                    return String::from_utf8(output.stdout)
                        .ok()
                        .map(|s| s.lines().next().unwrap_or("").trim().to_string())
                        .filter(|s| !s.is_empty());
                }
            }
        }
        #[cfg(target_os = "macos")]
        {
            // Use system_profiler on macOS
            if let Ok(output) = std::process::Command::new("system_profiler")
                .args(["SPDisplaysDataType"])
                .output()
            {
                if output.status.success() {
                    let text = String::from_utf8_lossy(&output.stdout);
                    for line in text.lines() {
                        if line.contains("Chipset Model:") {
                            return line.split(':').nth(1).map(|s| s.trim().to_string());
                        }
                    }
                }
            }
        }
        None
    }

    fn read_memory_gb() -> f64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
                for line in contents.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return kb as f64 / 1_048_576.0;
                            }
                        }
                    }
                }
            }
        }
        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = std::process::Command::new("sysctl")
                .args(["-n", "hw.memsize"])
                .output()
            {
                if let Ok(bytes_str) = String::from_utf8(output.stdout) {
                    if let Ok(bytes) = bytes_str.trim().parse::<u64>() {
                        return bytes as f64 / 1_073_741_824.0;
                    }
                }
            }
        }
        0.0
    }
}

/// Memory breakdown metrics (PMAT-012 UI-04)
#[derive(Debug, Clone, Default)]
pub struct MemoryBreakdown {
    /// Total RAM in KB
    pub total_kb: u64,
    /// Used RAM in KB
    pub used_kb: u64,
    /// Cached RAM in KB
    pub cached_kb: u64,
    /// Buffers in KB
    pub buffers_kb: u64,
    /// Available RAM in KB
    pub available_kb: u64,
}

impl MemoryBreakdown {
    /// Usage percentage
    pub fn usage_percent(&self) -> f64 {
        if self.total_kb > 0 {
            ((self.total_kb - self.available_kb) as f64 / self.total_kb as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Format KB as human-readable
    pub fn format_kb(kb: u64) -> String {
        if kb >= 1_048_576 {
            format!("{:.1}G", kb as f64 / 1_048_576.0)
        } else if kb >= 1024 {
            format!("{:.1}M", kb as f64 / 1024.0)
        } else {
            format!("{}K", kb)
        }
    }
}

/// Network metrics (PMAT-012 UI-07 P2)
#[derive(Debug, Clone, Default)]
pub struct NetworkMetrics {
    /// Total bytes received
    pub rx_bytes: u64,
    /// Total bytes transmitted
    pub tx_bytes: u64,
    /// Receive rate in bytes/sec
    pub rx_rate: f64,
    /// Transmit rate in bytes/sec
    pub tx_rate: f64,
}

impl NetworkMetrics {
    /// Format bytes as human-readable rate
    pub fn format_rate(bytes_per_sec: f64) -> String {
        if bytes_per_sec >= 1_073_741_824.0 {
            format!("{:.1} GB/s", bytes_per_sec / 1_073_741_824.0)
        } else if bytes_per_sec >= 1_048_576.0 {
            format!("{:.1} MB/s", bytes_per_sec / 1_048_576.0)
        } else if bytes_per_sec >= 1024.0 {
            format!("{:.1} KB/s", bytes_per_sec / 1024.0)
        } else {
            format!("{:.0} B/s", bytes_per_sec)
        }
    }
}

/// Disk metrics (PMAT-012 UI-08 P2)
#[derive(Debug, Clone, Default)]
pub struct DiskMetrics {
    /// Mount point
    pub mount: String,
    /// Total space in bytes
    pub total_bytes: u64,
    /// Used space in bytes
    pub used_bytes: u64,
    /// Usage percentage
    pub usage_percent: f64,
}

impl DiskMetrics {
    /// Format bytes as human-readable
    pub fn format_bytes(bytes: u64) -> String {
        if bytes >= 1_099_511_627_776 {
            format!("{:.1}T", bytes as f64 / 1_099_511_627_776.0)
        } else if bytes >= 1_073_741_824 {
            format!("{:.1}G", bytes as f64 / 1_073_741_824.0)
        } else if bytes >= 1_048_576 {
            format!("{:.1}M", bytes as f64 / 1_048_576.0)
        } else {
            format!("{:.1}K", bytes as f64 / 1024.0)
        }
    }
}

/// Real-time load metrics
#[derive(Debug, Clone, Default)]
pub struct LoadMetrics {
    /// Bricks executed per second
    pub bricks_per_second: f64,
    /// Total bricks executed
    pub total_bricks: u64,
    /// Average latency per brick in microseconds
    pub avg_latency_us: f64,
    /// Measured CPU usage from /proc/stat
    pub cpu_usage: f64,
    /// Per-core CPU usage (PMAT-012 UI-02)
    pub per_core_usage: Vec<f64>,
    /// Operations per second (FLOPS for GEMM)
    pub ops_per_second: f64,
    /// Bytes processed per second
    pub bytes_per_second: f64,
    /// Memory breakdown (PMAT-012 UI-04)
    pub memory: MemoryBreakdown,
    /// Network metrics (PMAT-012 UI-07 P2)
    pub network: NetworkMetrics,
    /// Disk metrics (PMAT-012 UI-08 P2)
    pub disks: Vec<DiskMetrics>,
}
