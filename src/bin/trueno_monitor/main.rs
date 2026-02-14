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

mod render;
mod render_stress;

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
use ratatui::{backend::CrosstermBackend, Terminal};

use tracing::{debug, info, warn};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use trueno_gpu::monitor::{
    cuda_monitoring_available, ChaosPreset, ComputeDevice, CpuDevice, CudaDeviceInfo,
    CudaMemoryInfo, MemoryMetrics, StressTarget, StressTestConfig,
};

// Trueno compute primitives for real stress testing
use trueno::Matrix;

#[cfg(feature = "cuda")]
use trueno_gpu::driver::CudaContext;

/// Stress test worker handle
pub(crate) struct StressWorker {
    running: Arc<AtomicBool>,
    ops_count: Arc<AtomicU64>,
    thread: Option<thread::JoinHandle<()>>,
}

/// Stress test verdict (TRUENO-SPEC-025)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StressTestVerdict {
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
pub(crate) struct StressTestReport {
    /// Test duration
    pub(crate) duration_secs: u64,
    /// Peak CPU ops/sec
    pub(crate) peak_cpu_ops: u64,
    /// Peak memory ops/sec
    pub(crate) peak_mem_ops: u64,
    /// Peak GPU ops/sec
    pub(crate) peak_gpu_ops: u64,
    /// Peak CPU utilization
    pub(crate) peak_cpu_util: f64,
    /// Peak RAM utilization
    pub(crate) peak_ram_util: f64,
    /// Peak GPU VRAM utilization (max across all GPUs)
    pub(crate) peak_vram_util: f64,
    /// Number of CPU workers
    pub(crate) cpu_workers: usize,
    /// Number of GPU workers
    pub(crate) gpu_workers: usize,
    /// Test verdict
    pub(crate) verdict: StressTestVerdict,
    /// Recommendations
    pub(crate) recommendations: Vec<String>,
}

/// GPU state from real hardware
pub(crate) struct GpuState {
    pub(crate) info: CudaDeviceInfo,
    #[cfg(feature = "cuda")]
    pub(crate) ctx: CudaContext,
    pub(crate) vram_used_gb: f64,
    pub(crate) vram_total_gb: f64,
    pub(crate) vram_percent: f64,
}

/// Application state
pub(crate) struct App {
    /// CPU device monitor
    pub(crate) cpu: CpuDevice,
    /// Memory metrics
    pub(crate) memory: MemoryMetrics,
    /// CPU usage history (60 points for sparkline)
    pub(crate) cpu_history: Vec<u64>,
    /// Memory usage history
    pub(crate) mem_history: Vec<u64>,
    /// Currently selected tab
    pub(crate) selected_tab: usize,
    /// Is stress test running
    pub(crate) stress_running: bool,
    /// Stress test config
    pub(crate) stress_config: Option<StressTestConfig>,
    /// Show help overlay
    pub(crate) show_help: bool,
    /// Tick count for animations
    pub(crate) tick: u64,
    /// Real GPU states (from CUDA hardware)
    pub(crate) gpus: Vec<GpuState>,
    /// GPU VRAM history per device
    pub(crate) gpu_vram_history: Vec<Vec<u64>>,
    /// CPU stress workers (one per core)
    pub(crate) cpu_workers: Vec<StressWorker>,
    /// Memory stress worker
    pub(crate) mem_worker: Option<StressWorker>,
    /// GPU stress workers (one per GPU)
    pub(crate) gpu_workers: Vec<StressWorker>,
    /// Total CPU ops/sec
    pub(crate) cpu_ops_per_sec: u64,
    /// CPU ops history for sparkline
    pub(crate) cpu_ops_history: Vec<u64>,
    /// Memory ops/sec
    pub(crate) mem_ops_per_sec: u64,
    /// Memory ops history
    pub(crate) mem_ops_history: Vec<u64>,
    /// GPU ops/sec (FLOPS)
    pub(crate) gpu_ops_per_sec: u64,
    /// GPU ops history
    pub(crate) gpu_ops_history: Vec<u64>,
    /// Stress test start time
    pub(crate) stress_start: Option<Instant>,
    /// Peak CPU ops/sec
    pub(crate) peak_cpu_ops: u64,
    /// Peak memory ops/sec
    pub(crate) peak_mem_ops: u64,
    /// Peak GPU ops/sec
    pub(crate) peak_gpu_ops: u64,
    /// Peak CPU utilization during stress
    pub(crate) peak_cpu_util: f64,
    /// Peak RAM utilization during stress
    pub(crate) peak_ram_util: f64,
    /// Peak VRAM utilization during stress
    pub(crate) peak_vram_util: f64,
    /// Last stress test report (via renacer)
    pub(crate) stress_report: Option<StressTestReport>,
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
            self.gpu_ops_history
                .push(self.gpu_ops_per_sec / 1_000_000_000); // G FLOPS

            // Track peak utilization for stress report
            let cpu_util = self.cpu.compute_utilization().unwrap_or(0.0);
            if cpu_util > self.peak_cpu_util {
                self.peak_cpu_util = cpu_util;
            }
            let ram_util = self.memory.ram_usage_percent();
            if ram_util > self.peak_ram_util {
                self.peak_ram_util = ram_util;
            }
            let vram_util = self
                .gpus
                .iter()
                .map(|g| g.vram_percent)
                .fold(0.0_f64, f64::max);
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
                let data_a: Vec<f32> = (0..n * n)
                    .map(|i| ((i + worker_id) % 1000) as f32 * 0.001)
                    .collect();
                let data_b: Vec<f32> = (0..n * n)
                    .map(|i| ((i * 7 + worker_id) % 1000) as f32 * 0.001)
                    .collect();

                let a = Matrix::from_vec(n, n, data_a).expect("stress test matrix A creation");
                let b = Matrix::from_vec(n, n, data_b).expect("stress test matrix B creation");

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
        let duration_secs = self
            .stress_start
            .map(|s| s.elapsed().as_secs())
            .unwrap_or(0);
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
        let verdict =
            if self.peak_cpu_util > 95.0 && (gpu_worker_count == 0 || self.peak_vram_util > 10.0) {
                StressTestVerdict::Pass
            } else if self.peak_cpu_util > 70.0 {
                StressTestVerdict::PassWithNotes
            } else {
                StressTestVerdict::Fail
            };

        let mut recommendations = Vec::new();
        if self.peak_cpu_util < 90.0 {
            recommendations.push(
                "CPU saturation below 90% - consider more compute-intensive workloads".to_string(),
            );
        }
        if gpu_worker_count > 0 && self.peak_vram_util < 20.0 {
            recommendations.push("GPU VRAM usage low - consider larger buffer sizes".to_string());
        }
        if self.peak_ram_util > 90.0 {
            recommendations
                .push("High RAM pressure detected - monitor for OOM conditions".to_string());
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
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_err| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(filter)
        .with(
            fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false)
                .with_target(true)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true),
        )
        .init();

    guard
}

/// Get log directory path (~/.trueno)
fn dirs_log_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".trueno")
}

/// Returns true if the app should quit
fn handle_key_event(app: &mut App, key: crossterm::event::KeyEvent) -> bool {
    if key.kind != KeyEventKind::Press {
        return false;
    }
    match key.code {
        KeyCode::Char('q') => return true,
        KeyCode::Char('?') => app.toggle_help(),
        KeyCode::Char('s') => app.toggle_stress(),
        KeyCode::Tab => app.next_tab(),
        KeyCode::BackTab => app.prev_tab(),
        KeyCode::Left => app.prev_tab(),
        KeyCode::Right => app.next_tab(),
        _ => {}
    }
    false
}

fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
) -> Result<(), Box<dyn std::error::Error>> {
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| render::ui(f, &app))?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if handle_key_event(app, key) {
                    break;
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            app.on_tick();
            last_tick = Instant::now();
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _log_guard = init_logging();

    let args: Vec<String> = std::env::args().collect();
    let start_stress = args.iter().any(|a| a == "--stress-test");

    info!(
        version = env!("CARGO_PKG_VERSION"),
        stress_mode = start_stress,
        "Trueno Monitor starting"
    );

    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new();
    if start_stress {
        app.toggle_stress();
    }

    run_event_loop(&mut terminal, &mut app)?;

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
