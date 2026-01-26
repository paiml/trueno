//! cbtop Application State Machine
//!
//! Manages the TUI application lifecycle using presentar-terminal.
//! Implements REAL load generation using trueno compute bricks.
//!
//! Citations:
//! - [Gregg 2020] "Systems Performance" 2nd ed. Addison-Wesley. ISBN:978-0-13-682015-4
//! - [Hennessy & Patterson 2017] "Computer Architecture" 6th ed. ISBN:978-0-12-811905-1

use std::time::{Duration, Instant};

use crossterm::event::KeyCode;
use presentar_core::{Canvas, Point, Rect, TextStyle};
use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
use presentar_terminal::{ColorMode, Theme};

use crate::bricks::generators::SimdLoadBrick;
use crate::bricks::panels::{
    ConfigPanelBrick, CpuPanelBrick, GpuPanelBrick, HelpPanelBrick, LoadControlPanelBrick,
    MemoryPanelBrick, OverviewPanelBrick, PciePanelBrick, ThermalPanelBrick,
};
use crate::config::{ComputeBackend, Config, WorkloadType};
use crate::error::CbtopError;
use crate::ring_buffer::RingBuffer;

/// Active panel in the UI
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActivePanel {
    #[default]
    Overview,
    Cpu,
    Gpu,
    Pcie,
    Memory,
    Thermal,
    Load,
    Config,
    Help,
}

impl ActivePanel {
    /// Get panel from key (1-9)
    pub fn from_key(key: char) -> Option<Self> {
        match key {
            '1' => Some(Self::Overview),
            '2' => Some(Self::Cpu),
            '3' => Some(Self::Gpu),
            '4' => Some(Self::Pcie),
            '5' => Some(Self::Memory),
            '6' => Some(Self::Thermal),
            '7' => Some(Self::Load),
            '8' => Some(Self::Config),
            '9' => Some(Self::Help),
            _ => None,
        }
    }

    /// Panel title
    pub fn title(&self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
            Self::Pcie => "PCIe",
            Self::Memory => "Memory",
            Self::Thermal => "Thermal",
            Self::Load => "Load",
            Self::Config => "Config",
            Self::Help => "Help",
        }
    }

    /// All panels for tab bar rendering (UI-10)
    pub fn all() -> &'static [Self] {
        &[
            Self::Overview,
            Self::Cpu,
            Self::Gpu,
            Self::Pcie,
            Self::Memory,
            Self::Thermal,
            Self::Load,
            Self::Config,
            Self::Help,
        ]
    }

    /// Key number for this panel (1-9)
    pub fn key_number(&self) -> char {
        match self {
            Self::Overview => '1',
            Self::Cpu => '2',
            Self::Gpu => '3',
            Self::Pcie => '4',
            Self::Memory => '5',
            Self::Thermal => '6',
            Self::Load => '7',
            Self::Config => '8',
            Self::Help => '9',
        }
    }
}

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

/// Application state
pub struct CbtopApp {
    /// Configuration
    config: Config,
    /// Active panel
    active_panel: ActivePanel,
    /// Is load generation running
    is_running: bool,
    /// Current load intensity
    intensity: f64,
    /// Current backend
    backend: ComputeBackend,
    /// Current workload
    workload: WorkloadType,
    /// Problem size
    problem_size: usize,

    /// Hardware information
    hardware: HardwareInfo,
    /// SIMD load generator (real compute)
    simd_generator: SimdLoadBrick,
    /// Real-time load metrics
    load_metrics: LoadMetrics,

    /// CPU usage history (REAL from /proc/stat)
    cpu_history: RingBuffer<f64>,
    /// Bricks/sec history
    bricks_history: RingBuffer<f64>,
    /// Frame times for FPS calculation
    frame_times: RingBuffer<f64>,
    /// Last CPU stat for delta calculation
    last_cpu_stat: Option<(u64, u64)>,
    /// Last network stat for rate calculation (rx_bytes, tx_bytes, time)
    last_network_stat: Option<(u64, u64, Instant)>,

    /// Panel Bricks
    overview_panel: OverviewPanelBrick,
    cpu_panel: CpuPanelBrick,
    gpu_panel: GpuPanelBrick,
    help_panel: HelpPanelBrick,
    memory_panel: MemoryPanelBrick,
    thermal_panel: ThermalPanelBrick,
    pcie_panel: PciePanelBrick,
    load_panel: LoadControlPanelBrick,
    config_panel: ConfigPanelBrick,

    /// Terminal buffer
    buffer: CellBuffer,
    /// Renderer
    renderer: DiffRenderer,
    /// Theme for colors
    theme: Theme,
    /// Last frame time
    last_frame: Instant,
    /// Frame count
    frame_count: u64,

    /// Should quit
    should_quit: bool,
}

impl CbtopApp {
    /// Create new application with config
    pub fn new(config: Config) -> Result<Self, CbtopError> {
        let (width, height) = crossterm::terminal::size()?;

        // Detect hardware at startup
        let hardware = HardwareInfo::detect();

        // Create SIMD load generator with configured problem size
        let mut simd_generator = SimdLoadBrick::new(config.problem_size);
        simd_generator.set_intensity(config.load_profile.intensity());

        Ok(Self {
            config: config.clone(),
            active_panel: ActivePanel::Overview,
            is_running: false,
            intensity: config.load_profile.intensity(),
            backend: config.backend,
            workload: config.workload,
            problem_size: config.problem_size,
            hardware,
            simd_generator,
            load_metrics: LoadMetrics::default(),
            cpu_history: RingBuffer::new(120),
            bricks_history: RingBuffer::new(120),
            frame_times: RingBuffer::new(60),
            last_cpu_stat: None,
            last_network_stat: None,
            overview_panel: OverviewPanelBrick::new(),
            cpu_panel: CpuPanelBrick::new(),
            gpu_panel: GpuPanelBrick::new(),
            help_panel: HelpPanelBrick::new(),
            memory_panel: MemoryPanelBrick::new(),
            thermal_panel: ThermalPanelBrick::new(),
            pcie_panel: PciePanelBrick::new(),
            load_panel: LoadControlPanelBrick::new(),
            config_panel: ConfigPanelBrick::new(),
            buffer: CellBuffer::new(width, height),
            renderer: DiffRenderer::with_color_mode(ColorMode::TrueColor),
            theme: Theme::tokyo_night(),
            last_frame: Instant::now(),
            frame_count: 0,
            should_quit: false,
        })
    }

    /// Run the application main loop
    pub fn run(&mut self) -> Result<(), CbtopError> {
        use crossterm::{
            event::{self, Event, KeyEventKind},
            terminal::{
                disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
            },
            ExecutableCommand,
        };
        use std::io::stdout;

        // Enter TUI mode
        enable_raw_mode()?;
        stdout().execute(EnterAlternateScreen)?;
        stdout().execute(crossterm::cursor::Hide)?;

        let refresh = Duration::from_millis(self.config.refresh_ms);

        while !self.should_quit {
            // Handle events
            if event::poll(refresh)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        self.handle_key(key.code);
                    }
                }
            }

            // Run REAL load generation if enabled
            if self.is_running {
                self.run_load_iteration();
            }

            // Collect REAL metrics
            self.collect_real_metrics();

            // Render
            self.render()?;
        }

        // Cleanup
        stdout().execute(crossterm::cursor::Show)?;
        stdout().execute(LeaveAlternateScreen)?;
        disable_raw_mode()?;

        Ok(())
    }

    /// Run one iteration of real compute load
    fn run_load_iteration(&mut self) {
        let start = Instant::now();

        // Execute real compute work via SIMD generator
        let duration = self.simd_generator.run_iteration();

        // Update metrics
        self.load_metrics.total_bricks += 1;

        let latency_us = duration.as_secs_f64() * 1_000_000.0;
        if latency_us > 0.0 {
            // Exponential moving average for latency
            self.load_metrics.avg_latency_us =
                0.9 * self.load_metrics.avg_latency_us + 0.1 * latency_us;
        }

        // Calculate bricks/second
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            let current_bps = 1.0 / elapsed;
            self.load_metrics.bricks_per_second =
                0.9 * self.load_metrics.bricks_per_second + 0.1 * current_bps;
        }

        // Calculate ops/second (2 * N ops for FMA: mul + add)
        let work_size = ((self.problem_size as f64) * self.intensity) as usize;
        self.load_metrics.ops_per_second =
            self.load_metrics.bricks_per_second * (work_size * 2) as f64;

        // Calculate bytes/second (3 arrays * N * 4 bytes)
        self.load_metrics.bytes_per_second =
            self.load_metrics.bricks_per_second * (work_size * 3 * 4) as f64;

        self.bricks_history
            .push(self.load_metrics.bricks_per_second);
    }

    /// Collect real system metrics
    fn collect_real_metrics(&mut self) {
        // Read real CPU usage from /proc/stat
        let cpu_usage = self.read_cpu_usage();
        self.cpu_history.push(cpu_usage);
        self.load_metrics.cpu_usage = cpu_usage;

        // Read memory breakdown from /proc/meminfo (PMAT-012 UI-04)
        self.load_metrics.memory = Self::read_memory();

        // Read network metrics from /proc/net/dev (PMAT-012 UI-07 P2)
        self.load_metrics.network = self.read_network();

        // Read disk metrics from statvfs (PMAT-012 UI-08 P2)
        self.load_metrics.disks = Self::read_disks();

        // Track frame time
        let now = Instant::now();
        let frame_ms = now.duration_since(self.last_frame).as_secs_f64() * 1000.0;
        self.frame_times.push(frame_ms);
        self.last_frame = now;
        self.frame_count += 1;
    }

    /// Read memory breakdown from /proc/meminfo (PMAT-012 UI-04)
    fn read_memory() -> MemoryBreakdown {
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
                let mut mem = MemoryBreakdown::default();
                for line in contents.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let value: u64 = parts[1].parse().unwrap_or(0);
                        match parts[0] {
                            "MemTotal:" => mem.total_kb = value,
                            "MemAvailable:" => mem.available_kb = value,
                            "Buffers:" => mem.buffers_kb = value,
                            "Cached:" => mem.cached_kb = value,
                            _ => {}
                        }
                    }
                }
                mem.used_kb = mem.total_kb.saturating_sub(mem.available_kb);
                return mem;
            }
        }
        MemoryBreakdown::default()
    }

    /// Read network metrics from /proc/net/dev (PMAT-012 UI-07 P2)
    fn read_network(&mut self) -> NetworkMetrics {
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/net/dev") {
                let mut total_rx: u64 = 0;
                let mut total_tx: u64 = 0;

                for line in contents.lines().skip(2) {
                    // Skip header lines
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 10 {
                        let iface = parts[0].trim_end_matches(':');
                        // Skip loopback
                        if iface == "lo" {
                            continue;
                        }
                        let rx: u64 = parts[1].parse().unwrap_or(0);
                        let tx: u64 = parts[9].parse().unwrap_or(0);
                        total_rx += rx;
                        total_tx += tx;
                    }
                }

                let now = Instant::now();
                let (rx_rate, tx_rate) =
                    if let Some((prev_rx, prev_tx, prev_time)) = self.last_network_stat {
                        let elapsed = now.duration_since(prev_time).as_secs_f64();
                        if elapsed > 0.0 {
                            let rx_delta = total_rx.saturating_sub(prev_rx) as f64;
                            let tx_delta = total_tx.saturating_sub(prev_tx) as f64;
                            (rx_delta / elapsed, tx_delta / elapsed)
                        } else {
                            (0.0, 0.0)
                        }
                    } else {
                        (0.0, 0.0)
                    };

                self.last_network_stat = Some((total_rx, total_tx, now));

                return NetworkMetrics {
                    rx_bytes: total_rx,
                    tx_bytes: total_tx,
                    rx_rate,
                    tx_rate,
                };
            }
        }
        NetworkMetrics::default()
    }

    /// Read disk metrics using statvfs (PMAT-012 UI-08 P2)
    fn read_disks() -> Vec<DiskMetrics> {
        let mut disks = Vec::new();

        #[cfg(target_os = "linux")]
        {
            // Read mounts from /proc/mounts and get stats for common ones
            if let Ok(contents) = std::fs::read_to_string("/proc/mounts") {
                for line in contents.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let mount = parts[1];
                        let fstype = parts.get(2).unwrap_or(&"");

                        // Only include real filesystems on common mounts
                        if !matches!(*fstype, "ext4" | "xfs" | "btrfs" | "zfs" | "ntfs" | "vfat") {
                            continue;
                        }
                        // Skip if not a standard mount
                        if !mount.starts_with("/home") && mount != "/" && !mount.starts_with("/mnt")
                        {
                            continue;
                        }

                        // Use nix or libc statvfs
                        #[cfg(unix)]
                        {
                            use std::ffi::CString;
                            use std::mem::MaybeUninit;

                            if let Ok(c_path) = CString::new(mount) {
                                let mut stat = MaybeUninit::<libc::statvfs>::uninit();
                                let result =
                                    unsafe { libc::statvfs(c_path.as_ptr(), stat.as_mut_ptr()) };
                                if result == 0 {
                                    let stat = unsafe { stat.assume_init() };
                                    let block_size = stat.f_frsize;
                                    let total = stat.f_blocks * block_size;
                                    let available = stat.f_bavail * block_size;
                                    let used = total.saturating_sub(available);
                                    let usage_pct = if total > 0 {
                                        (used as f64 / total as f64) * 100.0
                                    } else {
                                        0.0
                                    };

                                    disks.push(DiskMetrics {
                                        mount: mount.to_string(),
                                        total_bytes: total,
                                        used_bytes: used,
                                        usage_percent: usage_pct,
                                    });

                                    // Limit to 3 disks for display
                                    if disks.len() >= 3 {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        disks
    }

    /// Read real CPU usage from /proc/stat (aggregate and per-core)
    fn read_cpu_usage(&mut self) -> f64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/stat") {
                let mut aggregate_usage = 0.0;
                let mut per_core: Vec<f64> = Vec::new();

                for line in contents.lines() {
                    if line.starts_with("cpu") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 5 {
                            let user: u64 = parts[1].parse().unwrap_or(0);
                            let nice: u64 = parts[2].parse().unwrap_or(0);
                            let system: u64 = parts[3].parse().unwrap_or(0);
                            let idle: u64 = parts[4].parse().unwrap_or(0);

                            let total = user + nice + system + idle;
                            let active = user + nice + system;

                            if parts[0] == "cpu" {
                                // Aggregate CPU line
                                if let Some((prev_active, prev_total)) = self.last_cpu_stat {
                                    let delta_active = active.saturating_sub(prev_active);
                                    let delta_total = total.saturating_sub(prev_total);
                                    if delta_total > 0 {
                                        aggregate_usage =
                                            (delta_active as f64 / delta_total as f64) * 100.0;
                                    }
                                }
                                self.last_cpu_stat = Some((active, total));
                            } else if parts[0].starts_with("cpu") {
                                // Per-core CPU line (cpu0, cpu1, etc.)
                                // Calculate instantaneous usage (simplified - no delta tracking per core)
                                if total > 0 {
                                    per_core.push((active as f64 / total as f64) * 100.0);
                                }
                            }
                        }
                    }
                }

                self.load_metrics.per_core_usage = per_core;
                return aggregate_usage;
            }
        }
        // Fallback for non-Linux or on error
        0.0
    }

    /// Handle key press
    #[allow(clippy::wildcard_enum_match_arm)]
    fn handle_key(&mut self, code: KeyCode) {
        match code {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Char(' ') => {
                self.is_running = !self.is_running;
                if self.is_running {
                    self.simd_generator.start();
                } else {
                    self.simd_generator.stop();
                }
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.intensity = (self.intensity + 0.1).min(1.0);
                self.simd_generator.set_intensity(self.intensity);
            }
            KeyCode::Char('-') => {
                self.intensity = (self.intensity - 0.1).max(0.0);
                self.simd_generator.set_intensity(self.intensity);
            }
            KeyCode::Char('b') => {
                self.backend = match self.backend {
                    ComputeBackend::Simd => ComputeBackend::Wgpu,
                    ComputeBackend::Wgpu => ComputeBackend::Cuda,
                    ComputeBackend::Cuda => ComputeBackend::All,
                    ComputeBackend::All => ComputeBackend::Simd,
                };
            }
            KeyCode::Char('w') => {
                self.workload = match self.workload {
                    WorkloadType::Gemm => WorkloadType::Conv2d,
                    WorkloadType::Conv2d => WorkloadType::Attention,
                    WorkloadType::Attention => WorkloadType::Bandwidth,
                    WorkloadType::Bandwidth => WorkloadType::Elementwise,
                    WorkloadType::Elementwise => WorkloadType::Reduction,
                    WorkloadType::Reduction => WorkloadType::All,
                    WorkloadType::All => WorkloadType::Gemm,
                };
            }
            KeyCode::Char('[') => {
                self.problem_size = (self.problem_size / 2).max(1024);
                self.simd_generator = SimdLoadBrick::new(self.problem_size);
                self.simd_generator.set_intensity(self.intensity);
                if self.is_running {
                    self.simd_generator.start();
                }
            }
            KeyCode::Char(']') => {
                self.problem_size = (self.problem_size * 2).min(1_073_741_824);
                self.simd_generator = SimdLoadBrick::new(self.problem_size);
                self.simd_generator.set_intensity(self.intensity);
                if self.is_running {
                    self.simd_generator.start();
                }
            }
            KeyCode::Char('r') => {
                self.cpu_history.clear();
                self.bricks_history.clear();
                self.frame_times.clear();
                self.frame_count = 0;
                self.load_metrics = LoadMetrics::default();
            }
            KeyCode::Char(c) if c.is_ascii_digit() => {
                if let Some(panel) = ActivePanel::from_key(c) {
                    self.active_panel = panel;
                }
            }
            _ => {}
        }
    }

    /// Render the UI
    #[allow(clippy::wildcard_enum_match_arm)]
    fn render(&mut self) -> Result<(), CbtopError> {
        // Resize buffer if needed
        let (width, height) = crossterm::terminal::size()?;
        if self.buffer.width() != width || self.buffer.height() != height {
            self.buffer = CellBuffer::new(width, height);
        }

        // Extract state needed for rendering
        let active_panel = self.active_panel;
        let is_running = self.is_running;
        let intensity = self.intensity;
        let backend = self.backend;
        let show_fps = self.config.show_fps;
        let problem_size = self.problem_size;
        let _frame_count = self.frame_count;
        let cpu_data: Vec<f64> = self.cpu_history.iter().copied().collect();
        let bricks_data: Vec<f64> = self.bricks_history.iter().copied().collect();
        let cpu_avg = self.cpu_history.mean();
        let frame_avg = self.frame_times.mean();
        let hardware = self.hardware.clone();
        let metrics = self.load_metrics.clone();

        {
            let mut canvas = DirectTerminalCanvas::new(&mut self.buffer);

            // Background
            canvas.fill_rect(
                Rect::new(0.0, 0.0, width as f32, height as f32),
                self.theme.background,
            );

            // Title bar with hardware info
            Self::render_title_bar(&mut canvas, width, active_panel, &hardware, &self.theme);

            // Main content - REAL metrics display
            Self::render_main_content(
                &mut canvas,
                width,
                height,
                is_running,
                &metrics,
                &cpu_data,
                &bricks_data,
                cpu_avg,
                problem_size,
                &hardware,
                &self.theme,
            );

            // Status bar with real metrics
            Self::render_status_bar(
                &mut canvas,
                width,
                height,
                is_running,
                intensity,
                backend,
                show_fps,
                frame_avg,
                &metrics,
                &self.theme,
            );
        }

        // Flush to terminal
        let mut output = Vec::with_capacity(16384);
        self.renderer
            .flush(&mut self.buffer, &mut output)
            .map_err(|e| CbtopError::Render(e.to_string()))?;
        std::io::Write::write_all(&mut std::io::stdout(), &output)?;

        Ok(())
    }

    fn render_title_bar(
        canvas: &mut DirectTerminalCanvas,
        width: u16,
        active_panel: ActivePanel,
        hardware: &HardwareInfo,
        theme: &Theme,
    ) {
        let title_style = TextStyle {
            color: theme.foreground,
            ..Default::default()
        };

        // Title with hardware info
        let hw_info = format!(
            " cbtop │ {} ({} cores, {}) │ {:.0}GB RAM ",
            hardware.cpu_model.chars().take(30).collect::<String>(),
            hardware.cpu_cores,
            hardware.simd_type,
            hardware.memory_gb,
        );
        canvas.draw_text(&hw_info, Point::new(0.0, 0.0), &title_style);

        // GPU info if available
        if let Some(ref gpu) = hardware.gpu_name {
            let gpu_style = TextStyle {
                color: theme.cpu.sample(0.5),
                ..Default::default()
            };
            canvas.draw_text(
                &format!("│ GPU: {} ", gpu.chars().take(25).collect::<String>()),
                Point::new(hw_info.len() as f32, 0.0),
                &gpu_style,
            );
        }

        // PMAT-012 UI-10: Panel navigation tab bar
        Self::render_tab_bar(canvas, width, active_panel, theme);
    }

    /// Render panel navigation tab bar (UI-10)
    fn render_tab_bar(
        canvas: &mut DirectTerminalCanvas,
        width: u16,
        active_panel: ActivePanel,
        theme: &Theme,
    ) {
        let dim_style = TextStyle {
            color: theme.dim,
            ..Default::default()
        };
        let active_style = TextStyle {
            color: theme.foreground,
            ..Default::default()
        };
        let highlight_style = TextStyle {
            color: theme.cpu.sample(0.2),
            ..Default::default()
        };

        // Build tab bar string with highlighting
        let mut x: f32 = 1.0;
        canvas.draw_text("│", Point::new(0.0, 1.0), &dim_style);

        for panel in ActivePanel::all() {
            let key = panel.key_number();
            let title = panel.title();
            let is_active = *panel == active_panel;

            if is_active {
                // Active panel: highlighted with brackets
                canvas.draw_text("[", Point::new(x, 1.0), &highlight_style);
                x += 1.0;
                canvas.draw_text(&format!("{}", key), Point::new(x, 1.0), &highlight_style);
                x += 1.0;
                canvas.draw_text(":", Point::new(x, 1.0), &highlight_style);
                x += 1.0;
                canvas.draw_text(title, Point::new(x, 1.0), &active_style);
                x += title.len() as f32;
                canvas.draw_text("]", Point::new(x, 1.0), &highlight_style);
                x += 1.0;
            } else {
                // Inactive panel: dimmed
                canvas.draw_text(&format!(" {}:", key), Point::new(x, 1.0), &dim_style);
                x += 3.0;
                canvas.draw_text(title, Point::new(x, 1.0), &dim_style);
                x += title.len() as f32;
            }

            // Separator between panels (except last)
            if *panel != ActivePanel::Help {
                canvas.draw_text(" ", Point::new(x, 1.0), &dim_style);
                x += 1.0;
            }
        }

        // Fill remaining width with spaces and close
        let remaining = (width as f32 - x - 1.0).max(0.0) as usize;
        if remaining > 0 {
            canvas.draw_text(&" ".repeat(remaining), Point::new(x, 1.0), &dim_style);
        }
        canvas.draw_text("│", Point::new(width as f32 - 1.0, 1.0), &dim_style);
    }

    #[allow(clippy::too_many_arguments)]
    fn render_main_content(
        canvas: &mut DirectTerminalCanvas,
        width: u16,
        height: u16,
        is_running: bool,
        metrics: &LoadMetrics,
        cpu_data: &[f64],
        bricks_data: &[f64],
        cpu_avg: f64,
        problem_size: usize,
        hardware: &HardwareInfo,
        theme: &Theme,
    ) {
        let dim_style = TextStyle {
            color: theme.dim,
            ..Default::default()
        };
        let bright_style = TextStyle {
            color: theme.foreground,
            ..Default::default()
        };
        let accent_style = TextStyle {
            color: theme.cpu.sample(0.3),
            ..Default::default()
        };

        // PMAT-012 UI-01: Responsive width boxes
        let box_width = (width as usize).saturating_sub(2).max(40);
        let inner_width = box_width.saturating_sub(2);
        let bar_width = inner_width.saturating_sub(22).max(10);

        // Header line 2: Load status
        let status = if is_running {
            "● RUNNING"
        } else {
            "○ STOPPED"
        };
        let status_color = if is_running {
            theme.cpu.sample(0.0)
        } else {
            theme.dim
        };
        canvas.draw_text(
            &format!(" Load: {} ", status),
            Point::new(0.0, 2.0),
            &TextStyle {
                color: status_color,
                ..Default::default()
            },
        );

        // Metrics box with responsive width
        let box_top = format!(
            "┌─ Real-Time Metrics {}┐",
            "─".repeat(inner_width.saturating_sub(20))
        );
        canvas.draw_text(&box_top, Point::new(1.0, 3.0), &dim_style);

        // CPU Usage (REAL from /proc/stat) with color gradient
        let cpu_bar = Self::make_bar(cpu_avg, 100.0, bar_width);
        canvas.draw_text("│ CPU Usage:     ", Point::new(1.0, 4.0), &dim_style);
        canvas.draw_text(
            &cpu_bar,
            Point::new(17.0, 4.0),
            &TextStyle {
                color: theme.cpu_color(cpu_avg),
                ..Default::default()
            },
        );
        let cpu_val_x = 17.0 + bar_width as f32 + 1.0;
        canvas.draw_text(
            &format!("{:5.1}%", cpu_avg),
            Point::new(cpu_val_x, 4.0),
            &bright_style,
        );
        canvas.draw_text(" │", Point::new(box_width as f32, 4.0), &dim_style);

        // Bricks/Second with color gradient based on rate
        let bps = metrics.bricks_per_second;
        let bps_normalized = (bps / 10000.0).min(1.0) * 100.0;
        let bps_bar = Self::make_bar(bps.min(10000.0), 10000.0, bar_width);
        canvas.draw_text("│ Bricks/sec:    ", Point::new(1.0, 5.0), &dim_style);
        canvas.draw_text(
            &bps_bar,
            Point::new(17.0, 5.0),
            &TextStyle {
                color: theme.cpu_color(bps_normalized),
                ..Default::default()
            },
        );
        canvas.draw_text(
            &format!("{:>7.0}", bps),
            Point::new(cpu_val_x, 5.0),
            &bright_style,
        );
        canvas.draw_text(" │", Point::new(box_width as f32, 5.0), &dim_style);

        // Total Bricks executed
        let total_str = format!(
            "{:>width$}",
            Self::format_number(metrics.total_bricks),
            width = inner_width.saturating_sub(17)
        );
        canvas.draw_text("│ Total Bricks:  ", Point::new(1.0, 6.0), &dim_style);
        canvas.draw_text(&total_str, Point::new(17.0, 6.0), &bright_style);
        canvas.draw_text(" │", Point::new(box_width as f32, 6.0), &dim_style);

        // Avg Latency
        let latency_str = format!(
            "{:>width$.1} μs",
            metrics.avg_latency_us,
            width = inner_width.saturating_sub(20)
        );
        canvas.draw_text("│ Avg Latency:   ", Point::new(1.0, 7.0), &dim_style);
        canvas.draw_text(&latency_str, Point::new(17.0, 7.0), &bright_style);
        canvas.draw_text(" │", Point::new(box_width as f32, 7.0), &dim_style);

        // Problem size
        let size_str = format!(
            "{:>width$} elements",
            Self::format_number(problem_size as u64),
            width = inner_width.saturating_sub(26)
        );
        canvas.draw_text("│ Problem Size:  ", Point::new(1.0, 8.0), &dim_style);
        canvas.draw_text(&size_str, Point::new(17.0, 8.0), &bright_style);
        canvas.draw_text(" │", Point::new(box_width as f32, 8.0), &dim_style);

        // Throughput
        let gflops = metrics.ops_per_second / 1_000_000_000.0;
        let gbps = metrics.bytes_per_second / 1_073_741_824.0;
        canvas.draw_text("│ Throughput:    ", Point::new(1.0, 9.0), &dim_style);
        canvas.draw_text(
            &format!("{:>10.2} GFLOP/s │ {:>10.2} GB/s", gflops, gbps),
            Point::new(17.0, 9.0),
            &accent_style,
        );
        canvas.draw_text(" │", Point::new(box_width as f32, 9.0), &dim_style);

        let box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&box_bottom, Point::new(1.0, 10.0), &dim_style);

        // PMAT-012 UI-02: Per-core CPU bars
        let mut current_y = 12.0_f32;
        if !metrics.per_core_usage.is_empty() && height > 20 {
            let core_box_top = format!(
                "┌─ Per-Core CPU {}┐",
                "─".repeat(inner_width.saturating_sub(15))
            );
            canvas.draw_text(&core_box_top, Point::new(1.0, current_y), &dim_style);
            current_y += 1.0;

            // Render cores in rows of 4 for compact display
            let cores_per_row = 4.min(metrics.per_core_usage.len());
            let mini_bar_width = (inner_width / cores_per_row).saturating_sub(10).max(5);

            for (i, chunk) in metrics.per_core_usage.chunks(cores_per_row).enumerate() {
                let mut row = String::from("│ ");
                for (j, &usage) in chunk.iter().enumerate() {
                    let core_num = i * cores_per_row + j;
                    let mini_bar = Self::make_mini_bar(usage, 100.0, mini_bar_width);
                    row.push_str(&format!("C{:02}:{} ", core_num, mini_bar));
                }
                // Pad to box width
                while row.len() < box_width {
                    row.push(' ');
                }
                row.push('│');

                // Draw with per-core color gradient
                canvas.draw_text(&row[..3], Point::new(1.0, current_y), &dim_style);
                let mut x_pos = 4.0;
                for &usage in chunk.iter() {
                    let bar_str = Self::make_mini_bar(usage, 100.0, mini_bar_width);
                    canvas.draw_text(
                        &bar_str,
                        Point::new(x_pos + 4.0, current_y),
                        &TextStyle {
                            color: theme.cpu_color(usage),
                            ..Default::default()
                        },
                    );
                    x_pos += (mini_bar_width + 10) as f32;
                }
                canvas.draw_text("│", Point::new(box_width as f32, current_y), &dim_style);
                current_y += 1.0;

                if current_y > height as f32 - 10.0 {
                    break; // Don't overflow screen
                }
            }

            let core_box_bottom = format!("└{}┘", "─".repeat(inner_width));
            canvas.draw_text(&core_box_bottom, Point::new(1.0, current_y), &dim_style);
            current_y += 2.0;
        } else {
            // Hardware box (original position)
            let hw_box_top = format!(
                "┌─ Hardware {}┐",
                "─".repeat(inner_width.saturating_sub(12))
            );
            canvas.draw_text(&hw_box_top, Point::new(1.0, current_y), &dim_style);
            current_y += 1.0;

            let cpu_info = format!(
                "│ CPU:  {:width$} │",
                hardware
                    .cpu_model
                    .chars()
                    .take(inner_width.saturating_sub(10))
                    .collect::<String>(),
                width = inner_width.saturating_sub(8)
            );
            canvas.draw_text(&cpu_info, Point::new(1.0, current_y), &dim_style);
            current_y += 1.0;

            canvas.draw_text(
                &format!(
                    "│ Cores: {} │ SIMD: {} ",
                    hardware.cpu_cores, hardware.simd_type
                ),
                Point::new(1.0, current_y),
                &dim_style,
            );
            canvas.draw_text("│", Point::new(box_width as f32, current_y), &dim_style);
            current_y += 1.0;

            if let Some(ref gpu) = hardware.gpu_name {
                let gpu_str = format!(
                    "│ GPU:  {:width$} │",
                    gpu.chars()
                        .take(inner_width.saturating_sub(10))
                        .collect::<String>(),
                    width = inner_width.saturating_sub(8)
                );
                canvas.draw_text(&gpu_str, Point::new(1.0, current_y), &dim_style);
            } else {
                canvas.draw_text(
                    "│ GPU:  Not detected ",
                    Point::new(1.0, current_y),
                    &dim_style,
                );
                canvas.draw_text("│", Point::new(box_width as f32, current_y), &dim_style);
            }
            current_y += 1.0;

            canvas.draw_text(
                &format!("│ RAM:  {:.1} GB ", hardware.memory_gb),
                Point::new(1.0, current_y),
                &dim_style,
            );
            canvas.draw_text("│", Point::new(box_width as f32, current_y), &dim_style);
            current_y += 1.0;

            let hw_box_bottom = format!("└{}┘", "─".repeat(inner_width));
            canvas.draw_text(&hw_box_bottom, Point::new(1.0, current_y), &dim_style);
            current_y += 2.0;
        }

        // PMAT-012 UI-04: Memory breakdown box
        if metrics.memory.total_kb > 0 && current_y < height as f32 - 12.0 {
            let mem_box_top = format!("┌─ Memory {}┐", "─".repeat(inner_width.saturating_sub(11)));
            canvas.draw_text(&mem_box_top, Point::new(1.0, current_y), &dim_style);
            current_y += 1.0;

            // Memory usage bar with color gradient
            let mem_pct = metrics.memory.usage_percent();
            let mem_bar = Self::make_bar(mem_pct, 100.0, bar_width);
            canvas.draw_text("│ Used:    ", Point::new(1.0, current_y), &dim_style);
            canvas.draw_text(
                &mem_bar,
                Point::new(11.0, current_y),
                &TextStyle {
                    color: theme.memory_color(mem_pct),
                    ..Default::default()
                },
            );
            let mem_val_x = 11.0 + bar_width as f32 + 1.0;
            canvas.draw_text(
                &format!("{:5.1}%", mem_pct),
                Point::new(mem_val_x, current_y),
                &bright_style,
            );
            canvas.draw_text(" │", Point::new(box_width as f32, current_y), &dim_style);
            current_y += 1.0;

            // Memory breakdown values
            let total_str = MemoryBreakdown::format_kb(metrics.memory.total_kb);
            let used_str = MemoryBreakdown::format_kb(metrics.memory.used_kb);
            let cached_str = MemoryBreakdown::format_kb(metrics.memory.cached_kb);
            let buffers_str = MemoryBreakdown::format_kb(metrics.memory.buffers_kb);

            canvas.draw_text(
                &format!(
                    "│ Total: {:>8}  Used: {:>8}  Cache: {:>8}  Buf: {:>6} ",
                    total_str, used_str, cached_str, buffers_str
                ),
                Point::new(1.0, current_y),
                &dim_style,
            );
            canvas.draw_text("│", Point::new(box_width as f32, current_y), &dim_style);
            current_y += 1.0;

            let mem_box_bottom = format!("└{}┘", "─".repeat(inner_width));
            canvas.draw_text(&mem_box_bottom, Point::new(1.0, current_y), &dim_style);
            current_y += 2.0;
        }

        // PMAT-012 F405: GPU panel when NVIDIA present
        if hardware.gpu_name.is_some() && current_y < height as f32 - 10.0 {
            let gpu_box_top = format!("┌─ GPU {}┐", "─".repeat(inner_width.saturating_sub(7)));
            canvas.draw_text(&gpu_box_top, Point::new(1.0, current_y), &dim_style);
            current_y += 1.0;

            // GPU name
            if let Some(ref gpu) = hardware.gpu_name {
                let gpu_str = format!(
                    "│ {:width$} │",
                    gpu.chars()
                        .take(inner_width.saturating_sub(4))
                        .collect::<String>(),
                    width = inner_width.saturating_sub(2)
                );
                canvas.draw_text(
                    &gpu_str,
                    Point::new(1.0, current_y),
                    &TextStyle {
                        color: theme.gpu_color(50.0),
                        ..Default::default()
                    },
                );
            }
            current_y += 1.0;

            // GPU status hint
            canvas.draw_text(
                "│ Status: Ready for CUDA workloads ",
                Point::new(1.0, current_y),
                &dim_style,
            );
            canvas.draw_text("│", Point::new(box_width as f32, current_y), &dim_style);
            current_y += 1.0;

            let gpu_box_bottom = format!("└{}┘", "─".repeat(inner_width));
            canvas.draw_text(&gpu_box_bottom, Point::new(1.0, current_y), &dim_style);
            current_y += 2.0;
        }

        // PMAT-012 UI-07 P2: Network TX/RX panel
        if (metrics.network.rx_rate > 0.0 || metrics.network.tx_rate > 0.0)
            && current_y < height as f32 - 8.0
        {
            let net_box_top = format!("┌─ Network {}┐", "─".repeat(inner_width.saturating_sub(11)));
            canvas.draw_text(&net_box_top, Point::new(1.0, current_y), &dim_style);
            current_y += 1.0;

            let rx_str = NetworkMetrics::format_rate(metrics.network.rx_rate);
            let tx_str = NetworkMetrics::format_rate(metrics.network.tx_rate);
            canvas.draw_text(
                &format!("│ ↓ RX: {:>12}  ↑ TX: {:>12} ", rx_str, tx_str),
                Point::new(1.0, current_y),
                &dim_style,
            );
            canvas.draw_text("│", Point::new(box_width as f32, current_y), &dim_style);
            current_y += 1.0;

            let net_box_bottom = format!("└{}┘", "─".repeat(inner_width));
            canvas.draw_text(&net_box_bottom, Point::new(1.0, current_y), &dim_style);
            current_y += 2.0;
        }

        // PMAT-012 UI-08 P2: Disk per-mount panel
        if !metrics.disks.is_empty() && current_y < height as f32 - 6.0 {
            let disk_box_top = format!("┌─ Disks {}┐", "─".repeat(inner_width.saturating_sub(10)));
            canvas.draw_text(&disk_box_top, Point::new(1.0, current_y), &dim_style);
            current_y += 1.0;

            for disk in metrics.disks.iter().take(3) {
                let used_str = DiskMetrics::format_bytes(disk.used_bytes);
                let total_str = DiskMetrics::format_bytes(disk.total_bytes);
                let mount_short: String = disk.mount.chars().take(15).collect();
                let disk_bar = Self::make_mini_bar(disk.usage_percent, 100.0, 10);

                canvas.draw_text(
                    &format!("│ {:15} ", mount_short),
                    Point::new(1.0, current_y),
                    &dim_style,
                );
                canvas.draw_text(
                    &disk_bar,
                    Point::new(18.0, current_y),
                    &TextStyle {
                        color: theme.memory_color(disk.usage_percent),
                        ..Default::default()
                    },
                );
                canvas.draw_text(
                    &format!(
                        " {:>6}/{:>6} ({:.0}%)",
                        used_str, total_str, disk.usage_percent
                    ),
                    Point::new(29.0, current_y),
                    &dim_style,
                );
                canvas.draw_text("│", Point::new(box_width as f32, current_y), &dim_style);
                current_y += 1.0;

                if current_y > height as f32 - 5.0 {
                    break;
                }
            }

            let disk_box_bottom = format!("└{}┘", "─".repeat(inner_width));
            canvas.draw_text(&disk_box_bottom, Point::new(1.0, current_y), &dim_style);
            current_y += 2.0;
        }

        // PMAT-012 UI-09: Sparkline with corrected width (width - 6 for box chars + padding)
        let sparkline_width = (width as usize).saturating_sub(6).max(10);

        // PMAT-012 UI-05: Braille graphs for higher resolution sparklines
        if !cpu_data.is_empty() && current_y < height as f32 - 8.0 {
            let spark_box_top = format!(
                "┌─ CPU History (braille) {}┐",
                "─".repeat(inner_width.saturating_sub(23))
            );
            canvas.draw_text(&spark_box_top, Point::new(1.0, current_y), &dim_style);
            current_y += 1.0;

            // Use braille for higher resolution (2x data density)
            let braille_line = Self::make_braille_sparkline(cpu_data, sparkline_width);
            canvas.draw_text("│ ", Point::new(1.0, current_y), &dim_style);
            canvas.draw_text(
                &braille_line,
                Point::new(3.0, current_y),
                &TextStyle {
                    color: theme.cpu.sample(0.3),
                    ..Default::default()
                },
            );
            canvas.draw_text(" │", Point::new(box_width as f32, current_y), &dim_style);
            current_y += 1.0;

            let spark_box_bottom = format!("└{}┘", "─".repeat(inner_width));
            canvas.draw_text(&spark_box_bottom, Point::new(1.0, current_y), &dim_style);
            current_y += 2.0;
        }

        // Bricks/sec braille sparkline
        if !bricks_data.is_empty() && current_y < height as f32 - 5.0 {
            let brick_box_top = format!(
                "┌─ Bricks/sec (braille) {}┐",
                "─".repeat(inner_width.saturating_sub(24))
            );
            canvas.draw_text(&brick_box_top, Point::new(1.0, current_y), &dim_style);
            current_y += 1.0;

            let braille_line = Self::make_braille_sparkline(bricks_data, sparkline_width);
            canvas.draw_text("│ ", Point::new(1.0, current_y), &dim_style);
            canvas.draw_text(&braille_line, Point::new(3.0, current_y), &accent_style);
            canvas.draw_text(" │", Point::new(box_width as f32, current_y), &dim_style);
            current_y += 1.0;

            let brick_box_bottom = format!("└{}┘", "─".repeat(inner_width));
            canvas.draw_text(&brick_box_bottom, Point::new(1.0, current_y), &dim_style);
        }

        // Controls reminder
        if height > 10 {
            canvas.draw_text(
                " [Space] Toggle load  [+/-] Intensity  [b] Backend  [w] Workload  [r] Reset  [q] Quit ",
                Point::new(1.0, height as f32 - 3.0),
                &dim_style,
            );
        }
    }

    fn make_bar(value: f64, max: f64, width: usize) -> String {
        let filled = ((value / max) * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);
        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    }

    /// PMAT-012 UI-02: Mini bar for per-core CPU display (no brackets, compact)
    fn make_mini_bar(value: f64, max: f64, width: usize) -> String {
        let filled = ((value / max) * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);
        format!("{}{}", "▓".repeat(filled), "░".repeat(empty))
    }

    fn make_sparkline(data: &[f64], width: usize) -> String {
        const CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        if data.is_empty() {
            return " ".repeat(width);
        }

        let max = data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1.0);
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);
        let range = (max - min).max(1.0);

        // Sample data to fit width
        let step = data.len().max(1) as f64 / width as f64;
        let mut result = String::with_capacity(width);

        for i in 0..width {
            let idx = (i as f64 * step) as usize;
            if idx < data.len() {
                let normalized = (data[idx] - min) / range;
                let char_idx = (normalized * 7.0).round() as usize;
                result.push(CHARS[char_idx.min(7)]);
            } else {
                result.push(' ');
            }
        }

        result
    }

    /// PMAT-012 UI-05: Braille graph for higher resolution sparklines
    /// Uses Unicode braille patterns (U+2800-U+28FF) - each character encodes 2 columns x 4 rows
    fn make_braille_sparkline(data: &[f64], width: usize) -> String {
        if data.is_empty() {
            return " ".repeat(width);
        }

        let max = data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1.0);
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);
        let range = (max - min).max(1.0);

        // Braille dot positions (column 1: bits 0,1,2,6; column 2: bits 3,4,5,7)
        // Dots are numbered: 1,2,3,7 (left col), 4,5,6,8 (right col)
        // Bit mapping: dot1=0x01, dot2=0x02, dot3=0x04, dot4=0x08,
        //              dot5=0x10, dot6=0x20, dot7=0x40, dot8=0x80
        const COL1_DOTS: [u8; 4] = [0x40, 0x04, 0x02, 0x01]; // bottom to top
        const COL2_DOTS: [u8; 4] = [0x80, 0x20, 0x10, 0x08]; // bottom to top

        let mut result = String::with_capacity(width);
        let points_per_char = 2;
        let step = data.len().max(1) as f64 / (width * points_per_char) as f64;

        for i in 0..width {
            let mut pattern: u8 = 0;

            // Left column (first data point)
            let idx1 = ((i * 2) as f64 * step) as usize;
            if idx1 < data.len() {
                let normalized = (data[idx1] - min) / range;
                let dots = (normalized * 4.0).round() as usize;
                for d in 0..dots.min(4) {
                    pattern |= COL1_DOTS[d];
                }
            }

            // Right column (second data point)
            let idx2 = ((i * 2 + 1) as f64 * step) as usize;
            if idx2 < data.len() {
                let normalized = (data[idx2] - min) / range;
                let dots = (normalized * 4.0).round() as usize;
                for d in 0..dots.min(4) {
                    pattern |= COL2_DOTS[d];
                }
            }

            // Braille base is U+2800
            result.push(char::from_u32(0x2800 + pattern as u32).unwrap_or(' '));
        }

        result
    }

    fn format_number(n: u64) -> String {
        if n >= 1_000_000_000 {
            format!("{:.2}B", n as f64 / 1_000_000_000.0)
        } else if n >= 1_000_000 {
            format!("{:.2}M", n as f64 / 1_000_000.0)
        } else if n >= 1_000 {
            format!("{:.2}K", n as f64 / 1_000.0)
        } else {
            n.to_string()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_status_bar(
        canvas: &mut DirectTerminalCanvas,
        width: u16,
        height: u16,
        is_running: bool,
        intensity: f64,
        backend: ComputeBackend,
        show_fps: bool,
        frame_avg: f64,
        metrics: &LoadMetrics,
        theme: &Theme,
    ) {
        let y = height as f32 - 1.0;
        let dim_style = TextStyle {
            color: theme.dim,
            ..Default::default()
        };

        // Load status
        let status = if is_running { "●RUN" } else { "○OFF" };
        let status_color = if is_running {
            theme.cpu.sample(0.0)
        } else {
            theme.dim
        };
        canvas.draw_text(
            &format!(" {} ", status),
            Point::new(0.0, y),
            &TextStyle {
                color: status_color,
                ..Default::default()
            },
        );

        // Backend
        canvas.draw_text(&format!("│ {:?} ", backend), Point::new(6.0, y), &dim_style);

        // Intensity
        canvas.draw_text(
            &format!("│ Int:{:.0}% ", intensity * 100.0),
            Point::new(16.0, y),
            &dim_style,
        );

        // Real metrics in status bar
        canvas.draw_text(
            &format!(
                "│ {:.0} brick/s │ {:.1}μs ",
                metrics.bricks_per_second, metrics.avg_latency_us
            ),
            Point::new(28.0, y),
            &TextStyle {
                color: theme.foreground,
                ..Default::default()
            },
        );

        // PMAT-012 UI-06: GFLOP/s in status bar
        let gflops = metrics.ops_per_second / 1_000_000_000.0;
        canvas.draw_text(
            &format!("│ {:.2} GFLOP/s ", gflops),
            Point::new(55.0, y),
            &TextStyle {
                color: theme.cpu.sample(0.3),
                ..Default::default()
            },
        );

        // FPS
        if show_fps || frame_avg > 0.0 {
            let fps = if frame_avg > 0.0 {
                1000.0 / frame_avg
            } else {
                0.0
            };
            canvas.draw_text(
                &format!("│ {:.0} FPS", fps),
                Point::new(width as f32 - 10.0, y),
                &dim_style,
            );
        }
    }
}
