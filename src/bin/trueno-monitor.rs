//! Trueno Compute Monitor TUI (TRUENO-SPEC-020)
//!
//! Real-time terminal UI for monitoring compute flow, memory utilization,
//! and data movement across heterogeneous hardware.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin trueno-monitor --features tui-monitor
//! cargo run --bin trueno-monitor --features tui-monitor,cuda -- --stress-test
//! RUST_LOG=debug trueno-monitor  # Enable verbose logging
//! ```
//!
//! # Logging
//!
//! Logs are written to `~/.trueno/monitor.log` by default.
//! Set `RUST_LOG=debug` for verbose output.

use std::io::stdout;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline, Tabs},
    Frame, Terminal,
};

use trueno_gpu::monitor::{
    cuda_monitoring_available, CudaDeviceInfo, CudaMemoryInfo,
    CpuDevice, ComputeDevice, MemoryMetrics, PressureLevel,
    StressTestConfig, StressTarget, ChaosPreset,
};
use tracing::{info, warn, debug};
use tracing_subscriber::{fmt, EnvFilter, prelude::*};
use tracing_appender::rolling::{RollingFileAppender, Rotation};

// Trueno compute primitives for real stress testing
use trueno::Matrix;

#[cfg(feature = "cuda")]
use trueno_gpu::driver::CudaContext;

/// Stress test worker handle
struct StressWorker {
    running: Arc<AtomicBool>,
    ops_count: Arc<AtomicU64>,
    thread: Option<thread::JoinHandle<()>>,
}

/// Stress test verdict (TRUENO-SPEC-025)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StressTestVerdict {
    /// All metrics within acceptable range
    Pass,
    /// Minor throttling, acceptable
    PassWithNotes,
    /// Errors or severe throttling
    Fail,
}

impl std::fmt::Display for StressTestVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::PassWithNotes => write!(f, "PASS (with notes)"),
            Self::Fail => write!(f, "FAIL"),
        }
    }
}

/// Stress test report (TRUENO-SPEC-025 Section 7.4)
#[derive(Debug, Clone)]
struct StressTestReport {
    /// Test duration
    duration_secs: u64,
    /// Peak CPU ops/sec
    peak_cpu_ops: u64,
    /// Peak memory ops/sec
    peak_mem_ops: u64,
    /// Peak GPU ops/sec
    peak_gpu_ops: u64,
    /// Peak CPU utilization
    peak_cpu_util: f64,
    /// Peak RAM utilization
    peak_ram_util: f64,
    /// Peak GPU VRAM utilization (max across all GPUs)
    peak_vram_util: f64,
    /// Number of CPU workers
    cpu_workers: usize,
    /// Number of GPU workers
    gpu_workers: usize,
    /// Test verdict
    verdict: StressTestVerdict,
    /// Recommendations
    recommendations: Vec<String>,
}

/// GPU state from real hardware
struct GpuState {
    info: CudaDeviceInfo,
    #[cfg(feature = "cuda")]
    ctx: CudaContext,
    vram_used_gb: f64,
    vram_total_gb: f64,
    vram_percent: f64,
}

/// Application state
struct App {
    /// CPU device monitor
    cpu: CpuDevice,
    /// Memory metrics
    memory: MemoryMetrics,
    /// CPU usage history (60 points for sparkline)
    cpu_history: Vec<u64>,
    /// Memory usage history
    mem_history: Vec<u64>,
    /// Currently selected tab
    selected_tab: usize,
    /// Is stress test running
    stress_running: bool,
    /// Stress test config
    stress_config: Option<StressTestConfig>,
    /// Show help overlay
    show_help: bool,
    /// Tick count for animations
    tick: u64,
    /// Real GPU states (from CUDA hardware)
    gpus: Vec<GpuState>,
    /// GPU VRAM history per device
    gpu_vram_history: Vec<Vec<u64>>,
    /// CPU stress workers (one per core)
    cpu_workers: Vec<StressWorker>,
    /// Memory stress worker
    mem_worker: Option<StressWorker>,
    /// GPU stress workers (one per GPU)
    gpu_workers: Vec<StressWorker>,
    /// Total CPU ops/sec
    cpu_ops_per_sec: u64,
    /// CPU ops history for sparkline
    cpu_ops_history: Vec<u64>,
    /// Memory ops/sec
    mem_ops_per_sec: u64,
    /// Memory ops history
    mem_ops_history: Vec<u64>,
    /// GPU ops/sec (FLOPS)
    gpu_ops_per_sec: u64,
    /// GPU ops history
    gpu_ops_history: Vec<u64>,
    /// Stress test start time
    stress_start: Option<Instant>,
    /// Peak CPU ops/sec
    peak_cpu_ops: u64,
    /// Peak memory ops/sec
    peak_mem_ops: u64,
    /// Peak GPU ops/sec
    peak_gpu_ops: u64,
    /// Peak CPU utilization during stress
    peak_cpu_util: f64,
    /// Peak RAM utilization during stress
    peak_ram_util: f64,
    /// Peak VRAM utilization during stress
    peak_vram_util: f64,
    /// Last stress test report (via renacer)
    stress_report: Option<StressTestReport>,
}

impl App {
    fn new() -> Self {
        let mut cpu = CpuDevice::new();
        let _ = cpu.refresh();

        info!(
            cpu_name = cpu.device_name(),
            cores = num_cpus::get(),
            "CPU detected"
        );

        // Try to enumerate real CUDA GPUs
        let cuda_available = cuda_monitoring_available();
        let mut gpus = Vec::new();
        let mut gpu_vram_history = Vec::new();

        debug!(cuda_available, "CUDA monitoring check");

        if cuda_available {
            #[cfg(feature = "cuda")]
            {
                if let Ok(devices) = CudaDeviceInfo::enumerate() {
                    info!(gpu_count = devices.len(), "CUDA GPUs enumerated");
                    for info in devices {
                        if let Ok(ctx) = CudaContext::new(info.index as i32) {
                            let (vram_used_gb, vram_total_gb, vram_percent) =
                                if let Ok(mem) = CudaMemoryInfo::query(&ctx) {
                                    (
                                        mem.used() as f64 / (1024.0 * 1024.0 * 1024.0),
                                        mem.total as f64 / (1024.0 * 1024.0 * 1024.0),
                                        mem.usage_percent(),
                                    )
                                } else {
                                    (0.0, info.total_memory_gb(), 0.0)
                                };

                            info!(
                                gpu_index = info.index,
                                gpu_name = %info.name,
                                vram_total_gb,
                                vram_used_gb,
                                "GPU initialized"
                            );

                            gpus.push(GpuState {
                                info,
                                ctx,
                                vram_used_gb,
                                vram_total_gb,
                                vram_percent,
                            });
                            gpu_vram_history.push(vec![0; 60]);
                        }
                    }
                }
            }
        } else {
            warn!("CUDA monitoring not available");
        }

        Self {
            cpu,
            memory: MemoryMetrics::default(),
            cpu_history: vec![0; 60],
            mem_history: vec![0; 60],
            selected_tab: 0,
            stress_running: false,
            stress_config: None,
            show_help: false,
            tick: 0,
            gpus,
            gpu_vram_history,
            cpu_workers: Vec::new(),
            mem_worker: None,
            gpu_workers: Vec::new(),
            cpu_ops_per_sec: 0,
            cpu_ops_history: vec![0; 60],
            mem_ops_per_sec: 0,
            mem_ops_history: vec![0; 60],
            gpu_ops_per_sec: 0,
            gpu_ops_history: vec![0; 60],
            stress_start: None,
            peak_cpu_ops: 0,
            peak_mem_ops: 0,
            peak_gpu_ops: 0,
            peak_cpu_util: 0.0,
            peak_ram_util: 0.0,
            peak_vram_util: 0.0,
            stress_report: None,
        }
    }

    fn on_tick(&mut self) {
        self.tick += 1;

        // Refresh CPU metrics
        let _ = self.cpu.refresh();

        // Update CPU history
        let cpu_pct = self.cpu.compute_utilization().unwrap_or(0.0) as u64;
        self.cpu_history.remove(0);
        self.cpu_history.push(cpu_pct);

        // Update memory metrics
        self.memory.refresh();

        // Update memory history
        let mem_pct = self.memory.ram_usage_percent() as u64;
        self.mem_history.remove(0);
        self.mem_history.push(mem_pct);

        // Update GPU metrics from real hardware
        #[cfg(feature = "cuda")]
        for (i, gpu) in self.gpus.iter_mut().enumerate() {
            if let Ok(mem) = CudaMemoryInfo::query(&gpu.ctx) {
                gpu.vram_used_gb = mem.used() as f64 / (1024.0 * 1024.0 * 1024.0);
                gpu.vram_total_gb = mem.total as f64 / (1024.0 * 1024.0 * 1024.0);
                gpu.vram_percent = mem.usage_percent();

                if i < self.gpu_vram_history.len() {
                    self.gpu_vram_history[i].remove(0);
                    self.gpu_vram_history[i].push(gpu.vram_percent as u64);
                }
            }
        }

        // Update stress test metrics
        if self.stress_running {
            // Sum CPU ops from all workers and reset counters
            let mut total_cpu_ops: u64 = 0;
            for worker in &self.cpu_workers {
                let ops = worker.ops_count.swap(0, Ordering::Relaxed);
                total_cpu_ops += ops;
            }
            // Convert to ops/sec (tick is 100ms, so multiply by 10)
            self.cpu_ops_per_sec = total_cpu_ops * 10;
            if self.cpu_ops_per_sec > self.peak_cpu_ops {
                self.peak_cpu_ops = self.cpu_ops_per_sec;
            }

            // Memory ops
            if let Some(ref worker) = self.mem_worker {
                let ops = worker.ops_count.swap(0, Ordering::Relaxed);
                self.mem_ops_per_sec = ops * 10;
                if self.mem_ops_per_sec > self.peak_mem_ops {
                    self.peak_mem_ops = self.mem_ops_per_sec;
                }
            }

            // GPU ops
            let mut total_gpu_ops: u64 = 0;
            for worker in &self.gpu_workers {
                let ops = worker.ops_count.swap(0, Ordering::Relaxed);
                total_gpu_ops += ops;
            }
            self.gpu_ops_per_sec = total_gpu_ops * 10;
            if self.gpu_ops_per_sec > self.peak_gpu_ops {
                self.peak_gpu_ops = self.gpu_ops_per_sec;
            }

            // Update histories
            self.cpu_ops_history.remove(0);
            self.cpu_ops_history.push(self.cpu_ops_per_sec / 1_000_000); // M ops/sec
            self.mem_ops_history.remove(0);
            self.mem_ops_history.push(self.mem_ops_per_sec / 1_000_000);
            self.gpu_ops_history.remove(0);
            self.gpu_ops_history.push(self.gpu_ops_per_sec / 1_000_000_000); // G FLOPS

            // Track peak utilization for stress report
            let cpu_util = self.cpu.compute_utilization().unwrap_or(0.0);
            if cpu_util > self.peak_cpu_util {
                self.peak_cpu_util = cpu_util;
            }
            let ram_util = self.memory.ram_usage_percent();
            if ram_util > self.peak_ram_util {
                self.peak_ram_util = ram_util;
            }
            let vram_util = self.gpus.iter().map(|g| g.vram_percent).fold(0.0_f64, f64::max);
            if vram_util > self.peak_vram_util {
                self.peak_vram_util = vram_util;
            }
        }
    }

    fn next_tab(&mut self) {
        self.selected_tab = (self.selected_tab + 1) % 4;
    }

    fn prev_tab(&mut self) {
        if self.selected_tab > 0 {
            self.selected_tab -= 1;
        } else {
            self.selected_tab = 3;
        }
    }

    fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }

    fn toggle_stress(&mut self) {
        if self.stress_running {
            // Stop stress test
            self.stop_stress();
        } else {
            // Start stress test
            self.start_stress();
        }
    }

    fn start_stress(&mut self) {
        let num_workers = (num_cpus::get() / 4).max(1);
        info!(
            cpu_workers = num_workers,
            matrix_size = "512x512",
            flops_per_matmul = 512 * 512 * 512 * 2,
            gpu_count = self.gpus.len(),
            gpu_buffer_size = "256MB x 4",
            "Starting stress test with trueno SIMD/CUDA"
        );

        self.stress_running = true;
        self.stress_start = Some(Instant::now());
        self.stress_config = Some(StressTestConfig {
            target: StressTarget::All,
            duration: Duration::from_secs(60),
            intensity: 0.8,
            ramp_up: Duration::from_secs(5),
            chaos_preset: Some(ChaosPreset::Gentle),
            collect_metrics: true,
            export_report: false,
        });

        // Reset peaks for new stress test
        self.peak_cpu_ops = 0;
        self.peak_mem_ops = 0;
        self.peak_gpu_ops = 0;
        self.peak_cpu_util = 0.0;
        self.peak_ram_util = 0.0;
        self.peak_vram_util = 0.0;
        self.cpu_ops_history = vec![0; 60];
        self.mem_ops_history = vec![0; 60];
        self.gpu_ops_history = vec![0; 60];
        self.stress_report = None; // Clear previous report

        // Spawn CPU stress workers using trueno SIMD (AVX-512)
        // Use fewer workers with larger matrices for better cache utilization
        let num_workers = (num_cpus::get() / 4).max(1); // 1 worker per 4 cores
        let matrix_size = 512; // 512x512 = 262K elements, fits in L3

        for worker_id in 0..num_workers {
            let running = Arc::new(AtomicBool::new(true));
            let ops_count = Arc::new(AtomicU64::new(0));

            let r = running.clone();
            let o = ops_count.clone();

            let thread = thread::spawn(move || {
                // Create matrices for SIMD matmul stress
                // 512x512 matmul = 512^3 = 134M FLOPs per operation
                let n = matrix_size;
                let data_a: Vec<f32> = (0..n*n).map(|i| ((i + worker_id) % 1000) as f32 * 0.001).collect();
                let data_b: Vec<f32> = (0..n*n).map(|i| ((i * 7 + worker_id) % 1000) as f32 * 0.001).collect();

                let a = Matrix::from_vec(n, n, data_a).unwrap();
                let b = Matrix::from_vec(n, n, data_b).unwrap();

                // Stress loop: continuous matmul using AVX-512
                while r.load(Ordering::Relaxed) {
                    // Matrix multiply uses SIMD backend (AVX-512 on Threadripper)
                    let _c = a.matmul(&b);
                    // 512^3 * 2 FLOPs (mul + add) = 268M FLOPs per matmul
                    o.fetch_add((n * n * n * 2) as u64, Ordering::Relaxed);
                }
            });

            self.cpu_workers.push(StressWorker {
                running,
                ops_count,
                thread: Some(thread),
            });
        }

        // Spawn memory stress worker
        {
            let running = Arc::new(AtomicBool::new(true));
            let ops_count = Arc::new(AtomicU64::new(0));

            let r = running.clone();
            let o = ops_count.clone();

            let thread = thread::spawn(move || {
                // Memory stress: allocate and touch memory
                let mut buffers: Vec<Vec<u8>> = Vec::new();
                let chunk_size = 64 * 1024 * 1024; // 64MB chunks

                while r.load(Ordering::Relaxed) {
                    // Allocate
                    if buffers.len() < 8 {
                        let mut buf = vec![0u8; chunk_size];
                        // Touch every page
                        for i in (0..buf.len()).step_by(4096) {
                            buf[i] = (i & 0xFF) as u8;
                        }
                        buffers.push(buf);
                        o.fetch_add(chunk_size as u64 / 4096, Ordering::Relaxed);
                    } else {
                        // Read/write existing buffers
                        for buf in &mut buffers {
                            for i in (0..buf.len()).step_by(4096) {
                                buf[i] = buf[i].wrapping_add(1);
                            }
                            o.fetch_add(buf.len() as u64 / 4096, Ordering::Relaxed);
                        }
                        // Occasionally free and reallocate
                        if buffers.len() > 4 {
                            buffers.pop();
                        }
                    }
                    thread::sleep(Duration::from_millis(10));
                }
            });

            self.mem_worker = Some(StressWorker {
                running,
                ops_count,
                thread: Some(thread),
            });
        }

        // Spawn GPU stress workers (one per GPU)
        // Uses large buffers to saturate PCIe and VRAM
        #[cfg(feature = "cuda")]
        {
            use trueno_gpu::driver::{CudaContext, GpuBuffer};

            let num_gpus = self.gpus.len();
            for gpu_idx in 0..num_gpus {
                let running = Arc::new(AtomicBool::new(true));
                let ops_count = Arc::new(AtomicU64::new(0));

                let r = running.clone();
                let o = ops_count.clone();

                let thread = thread::spawn(move || {
                    // Create CUDA context for this GPU
                    if let Ok(ctx) = CudaContext::new(gpu_idx as i32) {
                        // Allocate large GPU buffers for stress test
                        // 64M elements * 4 bytes = 256MB per buffer
                        // 4 buffers = 1GB VRAM usage (4% of 24GB RTX 4090)
                        let n: usize = 64 * 1024 * 1024; // 64M elements = 256MB
                        let data: Vec<f32> = (0..n).map(|i| (i % 10000) as f32 * 0.0001).collect();

                        // Allocate multiple buffers to use more VRAM
                        let mut buffers: Vec<GpuBuffer<f32>> = Vec::new();
                        for _ in 0..4 {
                            if let Ok(buf) = GpuBuffer::<f32>::new(&ctx, n) {
                                buffers.push(buf);
                            }
                        }

                        if buffers.len() >= 2 {
                            let mut result = vec![0.0f32; n];

                            // GPU stress loop: saturate PCIe bandwidth
                            while r.load(Ordering::Relaxed) {
                                // H2D transfers to all buffers
                                for buf in &mut buffers {
                                    let _ = buf.copy_from_host(&data);
                                }

                                // D2H transfer from first buffer
                                let _ = buffers[0].copy_to_host(&mut result);

                                // Count bytes transferred:
                                // H2D: 4 buffers * 256MB = 1GB
                                // D2H: 1 buffer * 256MB = 256MB
                                // Total: 1.25GB per iteration
                                let bytes_transferred = (buffers.len() + 1) * n * 4;
                                o.fetch_add(bytes_transferred as u64, Ordering::Relaxed);
                            }
                        }
                    }
                });

                self.gpu_workers.push(StressWorker {
                    running,
                    ops_count,
                    thread: Some(thread),
                });
            }
        }
    }

    fn stop_stress(&mut self) {
        self.stress_running = false;

        // Calculate duration before clearing
        let duration_secs = self.stress_start.map(|s| s.elapsed().as_secs()).unwrap_or(0);
        let cpu_worker_count = self.cpu_workers.len();
        let gpu_worker_count = self.gpu_workers.len();

        info!(
            duration_secs,
            cpu_workers = cpu_worker_count,
            gpu_workers = gpu_worker_count,
            "Stopping stress test"
        );

        // Signal all workers to stop
        for worker in &self.cpu_workers {
            worker.running.store(false, Ordering::Relaxed);
        }
        if let Some(ref worker) = self.mem_worker {
            worker.running.store(false, Ordering::Relaxed);
        }
        for worker in &self.gpu_workers {
            worker.running.store(false, Ordering::Relaxed);
        }

        // Wait for threads to finish
        for worker in &mut self.cpu_workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
        if let Some(ref mut worker) = self.mem_worker {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
        for worker in &mut self.gpu_workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }

        // Generate stress test report (renacer integration)
        let verdict = if self.peak_cpu_util > 95.0 && (gpu_worker_count == 0 || self.peak_vram_util > 10.0) {
            StressTestVerdict::Pass
        } else if self.peak_cpu_util > 70.0 {
            StressTestVerdict::PassWithNotes
        } else {
            StressTestVerdict::Fail
        };

        let mut recommendations = Vec::new();
        if self.peak_cpu_util < 90.0 {
            recommendations.push("CPU saturation below 90% - consider more compute-intensive workloads".to_string());
        }
        if gpu_worker_count > 0 && self.peak_vram_util < 20.0 {
            recommendations.push("GPU VRAM usage low - consider larger buffer sizes".to_string());
        }
        if self.peak_ram_util > 90.0 {
            recommendations.push("High RAM pressure detected - monitor for OOM conditions".to_string());
        }

        self.stress_report = Some(StressTestReport {
            duration_secs,
            peak_cpu_ops: self.peak_cpu_ops,
            peak_mem_ops: self.peak_mem_ops,
            peak_gpu_ops: self.peak_gpu_ops,
            peak_cpu_util: self.peak_cpu_util,
            peak_ram_util: self.peak_ram_util,
            peak_vram_util: self.peak_vram_util,
            cpu_workers: cpu_worker_count,
            gpu_workers: gpu_worker_count,
            verdict,
            recommendations: recommendations.clone(),
        });

        // Log stress test report
        info!(
            duration_secs,
            verdict = %verdict,
            peak_cpu_ops = self.peak_cpu_ops,
            peak_mem_ops = self.peak_mem_ops,
            peak_gpu_ops = self.peak_gpu_ops,
            peak_cpu_util = format!("{:.1}%", self.peak_cpu_util),
            peak_ram_util = format!("{:.1}%", self.peak_ram_util),
            peak_vram_util = format!("{:.1}%", self.peak_vram_util),
            "Stress test completed"
        );
        for rec in &recommendations {
            warn!(recommendation = %rec, "Stress test recommendation");
        }

        // Clear workers
        self.cpu_workers.clear();
        self.mem_worker = None;
        self.gpu_workers.clear();
        self.stress_config = None;
    }
}

/// Initialize logging to file
fn init_logging() -> tracing_appender::non_blocking::WorkerGuard {
    // Create log directory
    let log_dir = dirs_log_path();
    std::fs::create_dir_all(&log_dir).ok();

    // Setup file appender with daily rotation
    let file_appender = RollingFileAppender::new(Rotation::DAILY, &log_dir, "monitor.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    // Build subscriber with env filter (default: info)
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer()
            .with_writer(non_blocking)
            .with_ansi(false)
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true))
        .init();

    guard
}

/// Get log directory path (~/.trueno)
fn dirs_log_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".trueno")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging (keep guard alive for duration of program)
    let _log_guard = init_logging();

    // Parse command line args
    let args: Vec<String> = std::env::args().collect();
    let start_stress = args.iter().any(|a| a == "--stress-test");

    info!(
        version = env!("CARGO_PKG_VERSION"),
        stress_mode = start_stress,
        "Trueno Monitor starting"
    );

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = App::new();
    if start_stress {
        app.toggle_stress();
    }

    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    // Main loop
    loop {
        terminal.draw(|f| ui(f, &app))?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Char('?') => app.toggle_help(),
                        KeyCode::Char('s') => app.toggle_stress(),
                        KeyCode::Tab => app.next_tab(),
                        KeyCode::BackTab => app.prev_tab(),
                        KeyCode::Left => app.prev_tab(),
                        KeyCode::Right => app.next_tab(),
                        _ => {}
                    }
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            app.on_tick();
            last_tick = Instant::now();
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    info!("Trueno Monitor shutdown complete");

    Ok(())
}

fn ui(f: &mut Frame, app: &App) {
    let size = f.area();

    // Main layout: header, content, footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Min(10),    // Content
            Constraint::Length(3),  // Footer
        ])
        .split(size);

    // Header with tabs
    let gpu_count = app.gpus.len();
    let title = if gpu_count > 0 {
        format!(" TRUENO Monitor v0.10.1 | {} GPU(s) ", gpu_count)
    } else {
        " TRUENO Monitor v0.10.1 | No CUDA GPU ".to_string()
    };

    let titles = vec!["Compute", "Memory", "Data Flow", "Stress Test"];
    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title(title))
        .select(app.selected_tab)
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    f.render_widget(tabs, chunks[0]);

    // Content based on selected tab
    match app.selected_tab {
        0 => render_compute_tab(f, app, chunks[1]),
        1 => render_memory_tab(f, app, chunks[1]),
        2 => render_dataflow_tab(f, app, chunks[1]),
        3 => render_stress_tab(f, app, chunks[1]),
        _ => {}
    }

    // Footer
    let help_text = if app.stress_running {
        " q:Quit  Tab:Switch  s:Stop Stress  ?:Help  |  STRESS TEST RUNNING "
    } else {
        " q:Quit  Tab:Switch  s:Start Stress  ?:Help  |  Refresh: 100ms "
    };
    let footer = Paragraph::new(help_text)
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, chunks[2]);

    // Help overlay
    if app.show_help {
        render_help_overlay(f, size);
    }
}

fn render_compute_tab(f: &mut Frame, app: &App, area: Rect) {
    // Calculate constraints based on number of GPUs
    let mut constraints = vec![];

    // Add stress banner if running
    if app.stress_running {
        constraints.push(Constraint::Length(1));  // Stress banner
    }

    constraints.push(Constraint::Length(3));  // CPU gauge
    constraints.push(Constraint::Length(5));  // CPU sparkline

    // Add constraints for each GPU
    for _ in &app.gpus {
        constraints.push(Constraint::Length(3)); // GPU VRAM gauge
    }

    constraints.push(Constraint::Min(3)); // Device info

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .margin(1)
        .split(area);

    let mut idx = 0;

    // Stress banner
    if app.stress_running {
        let elapsed = app.stress_start.map(|s| s.elapsed().as_secs()).unwrap_or(0);
        let banner = Paragraph::new(format!(
            " STRESS TEST ACTIVE | {}s | CPU: {:.1}M ops/s | MEM: {:.1}M pg/s | GPU: {:.2}G xf/s ",
            elapsed,
            app.cpu_ops_per_sec as f64 / 1_000_000.0,
            app.mem_ops_per_sec as f64 / 1_000_000.0,
            app.gpu_ops_per_sec as f64 / 1_000_000_000.0
        ))
        .style(Style::default().fg(Color::Black).bg(Color::Yellow).add_modifier(Modifier::BOLD));
        f.render_widget(banner, chunks[idx]);
        idx += 1;
    }

    // CPU utilization gauge
    let cpu_pct = app.cpu.compute_utilization().unwrap_or(0.0);
    let cpu_color = if cpu_pct > 90.0 {
        Color::Red
    } else if cpu_pct > 70.0 {
        Color::Yellow
    } else {
        Color::Green
    };

    let cpu_title = if app.stress_running {
        format!(" CPU Utilization [STRESS: {} workers] ", app.cpu_workers.len())
    } else {
        " CPU Utilization ".to_string()
    };

    let cpu_gauge = Gauge::default()
        .block(Block::default().title(cpu_title).borders(Borders::ALL))
        .gauge_style(Style::default().fg(cpu_color))
        .percent(cpu_pct as u16)
        .label(format!("{:.1}%", cpu_pct));
    f.render_widget(cpu_gauge, chunks[idx]);
    idx += 1;

    // CPU sparkline (60-second history)
    let sparkline = Sparkline::default()
        .block(Block::default().title(" CPU History (60s) ").borders(Borders::ALL))
        .data(&app.cpu_history)
        .max(100)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(sparkline, chunks[idx]);
    idx += 1;

    // GPU VRAM gauges (real hardware)
    for (i, gpu) in app.gpus.iter().enumerate() {
        let gpu_color = if gpu.vram_percent > 90.0 {
            Color::Red
        } else if gpu.vram_percent > 70.0 {
            Color::Yellow
        } else {
            Color::Green
        };

        let title = if app.stress_running && !app.gpu_workers.is_empty() {
            format!(" GPU {} [STRESS] {} ", i, gpu.info.name)
        } else {
            format!(" GPU {} VRAM: {} ", i, gpu.info.name)
        };
        let label = format!(
            "{:.1} / {:.1} GB ({:.1}%)",
            gpu.vram_used_gb, gpu.vram_total_gb, gpu.vram_percent
        );

        let gpu_gauge = Gauge::default()
            .block(Block::default().title(title).borders(Borders::ALL))
            .gauge_style(Style::default().fg(gpu_color))
            .percent(gpu.vram_percent as u16)
            .label(label);
        f.render_widget(gpu_gauge, chunks[idx]);
        idx += 1;
    }

    // Device info
    let clock = app.cpu.compute_clock_mhz().unwrap_or(0);
    let temp = app.cpu.compute_temperature_c().unwrap_or(0.0);
    let cores = app.cpu.compute_unit_count();

    let mut info_lines = vec![
        Line::from(vec![
            Span::styled("CPU: ", Style::default().fg(Color::DarkGray)),
            Span::styled(app.cpu.device_name(), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Cores: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", cores), Style::default().fg(Color::Cyan)),
            Span::raw("  "),
            Span::styled("Clock: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{} MHz", clock), Style::default().fg(Color::Cyan)),
            Span::raw("  "),
            Span::styled("Temp: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}C", temp),
                Style::default().fg(if temp > 80.0 { Color::Red } else { Color::Green }),
            ),
        ]),
    ];

    // Add GPU info lines
    for (i, gpu) in app.gpus.iter().enumerate() {
        info_lines.push(Line::from(vec![
            Span::styled(format!("GPU{}: ", i), Style::default().fg(Color::DarkGray)),
            Span::styled(&gpu.info.name, Style::default().fg(Color::Magenta)),
            Span::styled(
                format!(" ({:.1} GB)", gpu.info.total_memory_gb()),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }

    let info = Paragraph::new(info_lines)
        .block(Block::default().title(" Device Info ").borders(Borders::ALL));
    f.render_widget(info, chunks[idx]);
}

fn render_memory_tab(f: &mut Frame, app: &App, area: Rect) {
    // Calculate constraints based on number of GPUs
    let mut constraints = vec![];

    if app.stress_running {
        constraints.push(Constraint::Length(1));  // Stress banner
    }

    constraints.push(Constraint::Length(3));  // RAM gauge
    constraints.push(Constraint::Length(3));  // SWAP gauge

    // Add VRAM gauge for each GPU
    for _ in &app.gpus {
        constraints.push(Constraint::Length(3));
    }

    constraints.push(Constraint::Length(5)); // Memory sparkline
    constraints.push(Constraint::Min(3));    // Pressure info

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .margin(1)
        .split(area);

    let mut idx = 0;

    // Stress banner
    if app.stress_running {
        let banner = Paragraph::new(format!(
            " STRESS: Allocating 512MB + {} GPU buffers | RAM pressure: {} ",
            app.gpu_workers.len() * 8, // 8MB per GPU (2 x 4MB buffers)
            app.memory.pressure_level
        ))
        .style(Style::default().fg(Color::Black).bg(Color::Yellow).add_modifier(Modifier::BOLD));
        f.render_widget(banner, chunks[idx]);
        idx += 1;
    }

    // RAM gauge
    let ram_pct = app.memory.ram_usage_percent();
    let ram_color = match app.memory.pressure_level {
        PressureLevel::Ok => Color::Green,
        PressureLevel::Elevated => Color::Yellow,
        PressureLevel::Warning => Color::Rgb(255, 165, 0), // Orange
        PressureLevel::Critical => Color::Red,
    };

    let ram_used_gb = app.memory.ram_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let ram_total_gb = app.memory.ram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    let ram_title = if app.stress_running {
        " RAM [STRESS ACTIVE] ".to_string()
    } else {
        " RAM ".to_string()
    };

    let ram_gauge = Gauge::default()
        .block(Block::default().title(ram_title).borders(Borders::ALL))
        .gauge_style(Style::default().fg(ram_color))
        .percent(ram_pct as u16)
        .label(format!(
            "{:.1} / {:.1} GB ({:.1}%)",
            ram_used_gb, ram_total_gb, ram_pct
        ));
    f.render_widget(ram_gauge, chunks[idx]);
    idx += 1;

    // SWAP gauge
    let swap_pct = app.memory.swap_usage_percent();
    let swap_used_gb = app.memory.swap_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let swap_total_gb = app.memory.swap_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    let swap_gauge = Gauge::default()
        .block(Block::default().title(" SWAP ").borders(Borders::ALL))
        .gauge_style(Style::default().fg(if swap_pct > 50.0 {
            Color::Yellow
        } else {
            Color::Blue
        }))
        .percent(swap_pct as u16)
        .label(format!(
            "{:.1} / {:.1} GB ({:.1}%)",
            swap_used_gb, swap_total_gb, swap_pct
        ));
    f.render_widget(swap_gauge, chunks[idx]);
    idx += 1;

    // GPU VRAM gauges (real hardware)
    for (i, gpu) in app.gpus.iter().enumerate() {
        let vram_color = if gpu.vram_percent > 90.0 {
            Color::Red
        } else if gpu.vram_percent > 70.0 {
            Color::Yellow
        } else {
            Color::Magenta
        };

        let title = if app.stress_running && !app.gpu_workers.is_empty() {
            format!(" VRAM {} [STRESS] ", i)
        } else {
            format!(" VRAM {} [{}] ", i, gpu.info.name)
        };
        let vram_gauge = Gauge::default()
            .block(Block::default().title(title).borders(Borders::ALL))
            .gauge_style(Style::default().fg(vram_color))
            .percent(gpu.vram_percent as u16)
            .label(format!(
                "{:.1} / {:.1} GB ({:.1}%)",
                gpu.vram_used_gb, gpu.vram_total_gb, gpu.vram_percent
            ));
        f.render_widget(vram_gauge, chunks[idx]);
        idx += 1;
    }

    // Memory sparkline
    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .title(" Memory History (60s) ")
                .borders(Borders::ALL),
        )
        .data(&app.mem_history)
        .max(100)
        .style(Style::default().fg(Color::Magenta));
    f.render_widget(sparkline, chunks[idx]);
    idx += 1;

    // Pressure info
    let pressure_str = match app.memory.pressure_level {
        PressureLevel::Ok => ("OK", Color::Green, ">= 50% available"),
        PressureLevel::Elevated => ("ELEVATED", Color::Yellow, "30-50% available"),
        PressureLevel::Warning => ("WARNING", Color::Rgb(255, 165, 0), "15-30% available"),
        PressureLevel::Critical => ("CRITICAL", Color::Red, "< 15% available"),
    };

    let pressure_text = vec![
        Line::from(vec![
            Span::styled("Pressure Level: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                pressure_str.0,
                Style::default()
                    .fg(pressure_str.1)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" ({})", pressure_str.2),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("Safe Parallel Jobs: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", app.memory.safe_parallel_jobs),
                Style::default().fg(Color::Cyan),
            ),
        ]),
    ];
    let pressure = Paragraph::new(pressure_text).block(
        Block::default()
            .title(" Memory Pressure (LAMBDA-0002) ")
            .borders(Borders::ALL),
    );
    f.render_widget(pressure, chunks[idx]);
}

fn render_dataflow_tab(f: &mut Frame, app: &App, area: Rect) {
    let title = if app.stress_running && !app.gpu_workers.is_empty() {
        " Data Flow Monitor [STRESS: H2D/D2H ACTIVE] "
    } else {
        " Data Flow Monitor "
    };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL);

    if app.gpus.is_empty() {
        // No GPU - show message
        let text = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No CUDA GPU Detected",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  To enable GPU monitoring:",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(""),
            Line::from("    1. Install NVIDIA CUDA drivers"),
            Line::from("    2. Build with: cargo build --features tui-monitor,cuda"),
            Line::from("    3. Run: cargo run --bin trueno-monitor --features tui-monitor,cuda"),
            Line::from(""),
            Line::from(Span::styled(
                "  PCIe bandwidth and transfer tracking require CUDA hardware.",
                Style::default().fg(Color::DarkGray),
            )),
        ];
        let paragraph = Paragraph::new(text).block(block);
        f.render_widget(paragraph, area);
    } else {
        // Real GPU data
        let mut lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  PCIe Data Flow",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];

        for (i, gpu) in app.gpus.iter().enumerate() {
            lines.push(Line::from(vec![
                Span::styled(format!("  GPU {}: ", i), Style::default().fg(Color::DarkGray)),
                Span::styled(&gpu.info.name, Style::default().fg(Color::Magenta)),
            ]));

            lines.push(Line::from(vec![
                Span::raw("    VRAM: "),
                Span::styled(
                    format!("{:.1} GB", gpu.vram_used_gb),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(" / "),
                Span::styled(
                    format!("{:.1} GB", gpu.vram_total_gb),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" ("),
                Span::styled(
                    format!("{:.1}%", gpu.vram_percent),
                    Style::default().fg(if gpu.vram_percent > 80.0 {
                        Color::Red
                    } else {
                        Color::Green
                    }),
                ),
                Span::raw(")"),
            ]));

            // PCIe info
            if app.stress_running && !app.gpu_workers.is_empty() {
                let bandwidth_gbps = (app.gpu_ops_per_sec as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0);
                lines.push(Line::from(vec![
                    Span::raw("    PCIe: "),
                    Span::styled("STRESS ACTIVE", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                    Span::raw(" - "),
                    Span::styled(format!("{:.2} GB/s", bandwidth_gbps), Style::default().fg(Color::Cyan)),
                    Span::raw(" actual"),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::raw("    PCIe: "),
                    Span::styled("Gen4 x16", Style::default().fg(Color::Green)),
                    Span::raw(" ("),
                    Span::styled("31.5 GB/s", Style::default().fg(Color::Cyan)),
                    Span::raw(" theoretical)"),
                ]));
            }

            lines.push(Line::from(""));
        }

        lines.push(Line::from(Span::styled(
            "  Active Transfers",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));

        if app.stress_running && !app.gpu_workers.is_empty() {
            lines.push(Line::from(vec![
                Span::raw("    "),
                Span::styled("H2D: ", Style::default().fg(Color::Green)),
                Span::raw("1M x f32 (4MB) per iteration"),
            ]));
            lines.push(Line::from(vec![
                Span::raw("    "),
                Span::styled("D2H: ", Style::default().fg(Color::Yellow)),
                Span::raw("1M x f32 (4MB) per iteration"),
            ]));
            lines.push(Line::from(vec![
                Span::raw("    "),
                Span::styled(
                    format!("Throughput: {:.2} G elements/sec", app.gpu_ops_per_sec as f64 / 1_000_000_000.0),
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                ),
            ]));
        } else {
            lines.push(Line::from(Span::styled(
                "    No active kernel transfers",
                Style::default().fg(Color::DarkGray),
            )));
        }

        let paragraph = Paragraph::new(lines).block(block);
        f.render_widget(paragraph, area);
    }
}

fn render_stress_tab(f: &mut Frame, app: &App, area: Rect) {
    if app.stress_running {
        // Show real-time stress visualization
        render_stress_running(f, app, area);
    } else {
        // Show start screen
        render_stress_idle(f, app, area);
    }
}

fn render_stress_idle(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .title(" Stress Test Mode (TRUENO-SPEC-025) ")
        .borders(Borders::ALL);

    let mut text = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Status: "),
            Span::styled("IDLE", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Hardware Detected:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::raw("    CPU: "),
            Span::styled(app.cpu.device_name(), Style::default().fg(Color::White)),
            Span::styled(
                format!(" ({} cores)", num_cpus::get()),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ];

    for (i, gpu) in app.gpus.iter().enumerate() {
        text.push(Line::from(vec![
            Span::raw(format!("    GPU{}: ", i)),
            Span::styled(&gpu.info.name, Style::default().fg(Color::Magenta)),
        ]));
    }

    text.extend(vec![
        Line::from(""),
        Line::from(Span::styled(
            "  Stress Test Will:",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::raw("    "),
            Span::styled("CPU:", Style::default().fg(Color::Yellow)),
            Span::raw(format!(" {} threads doing FP math (sin/cos/sqrt)", num_cpus::get())),
        ]),
        Line::from(vec![
            Span::raw("    "),
            Span::styled("MEM:", Style::default().fg(Color::Yellow)),
            Span::raw(" Allocate 512MB, touch every page"),
        ]),
    ]);

    for (i, gpu) in app.gpus.iter().enumerate() {
        text.push(Line::from(vec![
            Span::raw("    "),
            Span::styled(format!("GPU{}:", i), Style::default().fg(Color::Yellow)),
            Span::raw(format!(" {} - H2D/D2H transfers (1M x f32)", gpu.info.name)),
        ]));
    }

    // Show stress test report if available (renacer integration)
    if let Some(ref report) = app.stress_report {
        let verdict_color = match report.verdict {
            StressTestVerdict::Pass => Color::Green,
            StressTestVerdict::PassWithNotes => Color::Yellow,
            StressTestVerdict::Fail => Color::Red,
        };
        text.extend(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  Stress Test Report (renacer):",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::raw("    Verdict: "),
                Span::styled(
                    format!("{}", report.verdict),
                    Style::default().fg(verdict_color).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::raw("    Duration: "),
                Span::styled(
                    format!("{}s", report.duration_secs),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" | Workers: "),
                Span::styled(
                    format!("{} CPU + {} GPU", report.cpu_workers, report.gpu_workers),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::raw("    Peak CPU: "),
                Span::styled(
                    format!("{:.2} M ops/sec ({:.1}%)", report.peak_cpu_ops as f64 / 1_000_000.0, report.peak_cpu_util),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::raw("    Peak MEM: "),
                Span::styled(
                    format!("{:.2} M pages/sec ({:.1}%)", report.peak_mem_ops as f64 / 1_000_000.0, report.peak_ram_util),
                    Style::default().fg(Color::Magenta),
                ),
            ]),
        ]);
        if report.peak_gpu_ops > 0 {
            text.push(Line::from(vec![
                Span::raw("    Peak GPU: "),
                Span::styled(
                    format!("{:.2} G xfers/sec ({:.1}% VRAM)", report.peak_gpu_ops as f64 / 1_000_000_000.0, report.peak_vram_util),
                    Style::default().fg(Color::Yellow),
                ),
            ]));
        }
        // Show recommendations if any
        if !report.recommendations.is_empty() {
            text.push(Line::from(""));
            text.push(Line::from(Span::styled(
                "  Recommendations:",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            )));
            for rec in &report.recommendations {
                text.push(Line::from(vec![
                    Span::raw("    • "),
                    Span::styled(rec, Style::default().fg(Color::DarkGray)),
                ]));
            }
        }
    }

    text.extend(vec![
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled(
            "  >>> Press 's' to START stress test <<<",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )),
    ]);

    let paragraph = Paragraph::new(text).block(block);
    f.render_widget(paragraph, area);
}

fn render_stress_running(f: &mut Frame, app: &App, area: Rect) {
    let has_gpu = !app.gpu_workers.is_empty();

    let mut constraints = vec![
        Constraint::Length(3),  // Status bar
        Constraint::Length(3),  // CPU ops gauge
        Constraint::Length(4),  // CPU ops sparkline
        Constraint::Length(3),  // Memory ops gauge
        Constraint::Length(4),  // Memory ops sparkline
    ];

    if has_gpu {
        constraints.push(Constraint::Length(3));  // GPU ops gauge
        constraints.push(Constraint::Length(4));  // GPU ops sparkline
    }

    constraints.push(Constraint::Min(3));  // Stats

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .margin(1)
        .split(area);

    // Status bar with elapsed time
    let elapsed = app.stress_start.map(|s| s.elapsed().as_secs()).unwrap_or(0);
    let status_text = format!(
        " STRESS TEST RUNNING | {}s | {} CPU + {} GPU workers | 's' to STOP ",
        elapsed,
        app.cpu_workers.len(),
        app.gpu_workers.len()
    );
    let status = Paragraph::new(status_text)
        .style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(status, chunks[0]);

    // CPU ops gauge
    let cpu_mops = app.cpu_ops_per_sec as f64 / 1_000_000.0;
    let cpu_pct = ((cpu_mops / 100.0) * 100.0).min(100.0) as u16;
    let cpu_gauge = Gauge::default()
        .block(
            Block::default()
                .title(format!(
                    " CPU: {:.1} M ops/sec (peak: {:.1}) ",
                    cpu_mops,
                    app.peak_cpu_ops as f64 / 1_000_000.0
                ))
                .borders(Borders::ALL),
        )
        .gauge_style(Style::default().fg(Color::Cyan))
        .percent(cpu_pct)
        .label(format!("{:.2} M ops/sec", cpu_mops));
    f.render_widget(cpu_gauge, chunks[1]);

    // CPU ops sparkline
    let cpu_sparkline = Sparkline::default()
        .block(Block::default().title(" CPU History ").borders(Borders::ALL))
        .data(&app.cpu_ops_history)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(cpu_sparkline, chunks[2]);

    // Memory ops gauge
    let mem_mops = app.mem_ops_per_sec as f64 / 1_000_000.0;
    let mem_pct = ((mem_mops / 10.0) * 100.0).min(100.0) as u16;
    let mem_gauge = Gauge::default()
        .block(
            Block::default()
                .title(format!(
                    " MEM: {:.1} M pages/sec (peak: {:.1}) ",
                    mem_mops,
                    app.peak_mem_ops as f64 / 1_000_000.0
                ))
                .borders(Borders::ALL),
        )
        .gauge_style(Style::default().fg(Color::Magenta))
        .percent(mem_pct)
        .label(format!("{:.2} M pages/sec", mem_mops));
    f.render_widget(mem_gauge, chunks[3]);

    // Memory ops sparkline
    let mem_sparkline = Sparkline::default()
        .block(Block::default().title(" Memory History ").borders(Borders::ALL))
        .data(&app.mem_ops_history)
        .style(Style::default().fg(Color::Magenta));
    f.render_widget(mem_sparkline, chunks[4]);

    let stats_idx = if has_gpu {
        // GPU ops gauge
        let gpu_gops = app.gpu_ops_per_sec as f64 / 1_000_000_000.0;
        let gpu_pct = ((gpu_gops / 10.0) * 100.0).min(100.0) as u16; // Scale: 10 G = 100%
        let gpu_gauge = Gauge::default()
            .block(
                Block::default()
                    .title(format!(
                        " GPU: {:.2} G transfers/sec (peak: {:.2}) ",
                        gpu_gops,
                        app.peak_gpu_ops as f64 / 1_000_000_000.0
                    ))
                    .borders(Borders::ALL),
            )
            .gauge_style(Style::default().fg(Color::Yellow))
            .percent(gpu_pct)
            .label(format!("{:.2} G xfers/sec", gpu_gops));
        f.render_widget(gpu_gauge, chunks[5]);

        // GPU ops sparkline
        let gpu_sparkline = Sparkline::default()
            .block(Block::default().title(" GPU History ").borders(Borders::ALL))
            .data(&app.gpu_ops_history)
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(gpu_sparkline, chunks[6]);

        7
    } else {
        5
    };

    // Live stats
    let cpu_util = app.cpu.compute_utilization().unwrap_or(0.0);
    let mem_pct_used = app.memory.ram_usage_percent();

    let mut stats = vec![
        Line::from(vec![
            Span::styled("System: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("CPU {:.0}%", cpu_util),
                Style::default().fg(if cpu_util > 90.0 { Color::Red } else { Color::Green }),
            ),
            Span::raw(" | "),
            Span::styled(
                format!("RAM {:.0}%", mem_pct_used),
                Style::default().fg(if mem_pct_used > 80.0 { Color::Red } else { Color::Green }),
            ),
            Span::raw(" | Pressure: "),
            Span::styled(
                format!("{}", app.memory.pressure_level),
                Style::default().fg(match app.memory.pressure_level {
                    PressureLevel::Ok => Color::Green,
                    PressureLevel::Elevated => Color::Yellow,
                    PressureLevel::Warning => Color::Rgb(255, 165, 0),
                    PressureLevel::Critical => Color::Red,
                }),
            ),
        ]),
    ];

    // Show GPU VRAM if available
    for (i, gpu) in app.gpus.iter().enumerate() {
        stats.push(Line::from(vec![
            Span::styled(format!("GPU{} VRAM: ", i), Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1}%", gpu.vram_percent),
                Style::default().fg(if gpu.vram_percent > 90.0 { Color::Red } else { Color::Green }),
            ),
            Span::styled(
                format!(" ({:.1}/{:.1} GB)", gpu.vram_used_gb, gpu.vram_total_gb),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }

    let stats_block = Paragraph::new(stats)
        .block(Block::default().title(" System Impact ").borders(Borders::ALL));
    f.render_widget(stats_block, chunks[stats_idx]);
}

fn render_help_overlay(f: &mut Frame, size: Rect) {
    let block = Block::default()
        .title(" Help ")
        .borders(Borders::ALL)
        .style(Style::default().bg(Color::DarkGray));

    let area = centered_rect(50, 60, size);
    f.render_widget(ratatui::widgets::Clear, area);

    let help_text = vec![
        Line::from(""),
        Line::from(Span::styled(
            "Keyboard Controls",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  q        Quit"),
        Line::from("  Tab      Next tab"),
        Line::from("  Shift+Tab Previous tab"),
        Line::from("  s        Toggle stress test"),
        Line::from("  ?        Toggle this help"),
        Line::from(""),
        Line::from(Span::styled(
            "Tabs",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  Compute    CPU/GPU utilization"),
        Line::from("  Memory     RAM/SWAP/VRAM usage"),
        Line::from("  Data Flow  PCIe bandwidth"),
        Line::from("  Stress     Stress test controls"),
        Line::from(""),
        Line::from(Span::styled(
            "Press ? to close",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let paragraph = Paragraph::new(help_text).block(block);
    f.render_widget(paragraph, area);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
