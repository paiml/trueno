//! cbtop Application State Machine
//!
//! Manages the TUI application lifecycle using presentar-terminal.
//! Implements REAL load generation using trueno compute bricks.
//!
//! Citations:
//! - [Gregg 2020] "Systems Performance" 2nd ed. Addison-Wesley. ISBN:978-0-13-682015-4
//! - [Hennessy & Patterson 2017] "Computer Architecture" 6th ed. ISBN:978-0-12-811905-1

mod hardware;
mod panels;
mod render;

pub use hardware::{
    DiskMetrics, HardwareInfo, LoadMetrics, MemoryBreakdown, NetworkMetrics,
};
pub use panels::ActivePanel;

use std::time::{Duration, Instant};

use crossterm::event::KeyCode;
use presentar_terminal::direct::{CellBuffer, DiffRenderer};
use presentar_terminal::{ColorMode, Theme};

use crate::bricks::generators::SimdLoadBrick;
use crate::bricks::panels::{
    ConfigPanelBrick, CpuPanelBrick, GpuPanelBrick, HelpPanelBrick, LoadControlPanelBrick,
    MemoryPanelBrick, OverviewPanelBrick, PciePanelBrick, ThermalPanelBrick,
};
use crate::config::{ComputeBackend, Config, WorkloadType};
use crate::error::CbtopError;
use crate::ring_buffer::RingBuffer;

/// Application state
pub struct CbtopApp {
    /// Configuration
    pub(crate) config: Config,
    /// Active panel
    pub(crate) active_panel: ActivePanel,
    /// Is load generation running
    pub(crate) is_running: bool,
    /// Current load intensity
    pub(crate) intensity: f64,
    /// Current backend
    pub(crate) backend: ComputeBackend,
    /// Current workload
    pub(crate) workload: WorkloadType,
    /// Problem size
    pub(crate) problem_size: usize,

    /// Hardware information
    pub(crate) hardware: HardwareInfo,
    /// SIMD load generator (real compute)
    pub(crate) simd_generator: SimdLoadBrick,
    /// Real-time load metrics
    pub(crate) load_metrics: LoadMetrics,

    /// CPU usage history (REAL from /proc/stat)
    pub(crate) cpu_history: RingBuffer<f64>,
    /// Bricks/sec history
    pub(crate) bricks_history: RingBuffer<f64>,
    /// Frame times for FPS calculation
    pub(crate) frame_times: RingBuffer<f64>,
    /// Last CPU stat for delta calculation
    pub(crate) last_cpu_stat: Option<(u64, u64)>,
    /// Last network stat for rate calculation (rx_bytes, tx_bytes, time)
    pub(crate) last_network_stat: Option<(u64, u64, Instant)>,

    /// Panel Bricks
    pub(crate) overview_panel: OverviewPanelBrick,
    pub(crate) cpu_panel: CpuPanelBrick,
    pub(crate) gpu_panel: GpuPanelBrick,
    pub(crate) help_panel: HelpPanelBrick,
    pub(crate) memory_panel: MemoryPanelBrick,
    pub(crate) thermal_panel: ThermalPanelBrick,
    pub(crate) pcie_panel: PciePanelBrick,
    pub(crate) load_panel: LoadControlPanelBrick,
    pub(crate) config_panel: ConfigPanelBrick,

    /// Terminal buffer
    pub(crate) buffer: CellBuffer,
    /// Renderer
    pub(crate) renderer: DiffRenderer,
    /// Theme for colors
    pub(crate) theme: Theme,
    /// Last frame time
    pub(crate) last_frame: Instant,
    /// Frame count
    pub(crate) frame_count: u64,

    /// Should quit
    pub(crate) should_quit: bool,
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
}
