//! cbtop - Compute Block Top
//!
//! Real-time load testing and hardware monitoring TUI.
//!
//! Run with: cargo run -p cbtop

use clap::Parser;
use cbtop::{Config, CbtopApp, CbtopError};
use cbtop::config::{ComputeBackend, LoadProfile, WorkloadType};

/// Compute Block Top - Real-time load testing and hardware monitoring TUI
#[derive(Parser, Debug)]
#[command(name = "cbtop")]
#[command(author = "Trueno Engineering")]
#[command(version)]
#[command(about = "Real-time load testing and hardware monitoring TUI", long_about = None)]
struct Cli {
    /// Refresh rate in milliseconds
    #[arg(short, long, default_value = "100")]
    refresh: u64,

    /// GPU device index
    #[arg(short, long, default_value = "0")]
    device: u32,

    /// Compute backend: simd, wgpu, cuda, all
    #[arg(short, long, default_value = "all")]
    backend: String,

    /// Load profile: idle, light, medium, heavy, stress
    #[arg(short, long, default_value = "idle")]
    load: String,

    /// Workload type: gemm, conv, attention, bandwidth, elementwise, reduction, all
    #[arg(short, long, default_value = "gemm")]
    workload: String,

    /// Problem size in elements
    #[arg(short, long, default_value = "1048576")]
    size: usize,

    /// Thread count for SIMD
    #[arg(short, long)]
    threads: Option<usize>,

    /// Enable deterministic mode for testing
    #[arg(long)]
    deterministic: bool,

    /// Show frame timing statistics
    #[arg(long)]
    show_fps: bool,

    /// Config file path
    #[arg(short, long)]
    config: Option<std::path::PathBuf>,
}

fn parse_backend(s: &str) -> ComputeBackend {
    match s.to_lowercase().as_str() {
        "simd" => ComputeBackend::Simd,
        "wgpu" => ComputeBackend::Wgpu,
        "cuda" => ComputeBackend::Cuda,
        _ => ComputeBackend::All,
    }
}

fn parse_load_profile(s: &str) -> LoadProfile {
    match s.to_lowercase().as_str() {
        "light" => LoadProfile::Light,
        "medium" => LoadProfile::Medium,
        "heavy" => LoadProfile::Heavy,
        "stress" => LoadProfile::Stress,
        _ => LoadProfile::Idle,
    }
}

fn parse_workload(s: &str) -> WorkloadType {
    match s.to_lowercase().as_str() {
        "conv" | "conv2d" => WorkloadType::Conv2d,
        "attention" => WorkloadType::Attention,
        "bandwidth" => WorkloadType::Bandwidth,
        "elementwise" => WorkloadType::Elementwise,
        "reduction" => WorkloadType::Reduction,
        "all" => WorkloadType::All,
        _ => WorkloadType::Gemm,
    }
}

fn main() -> Result<(), CbtopError> {
    let cli = Cli::parse();

    let config = Config {
        refresh_ms: cli.refresh,
        device_index: cli.device,
        backend: parse_backend(&cli.backend),
        load_profile: parse_load_profile(&cli.load),
        workload: parse_workload(&cli.workload),
        problem_size: cli.size,
        threads: cli.threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        }),
        deterministic: cli.deterministic,
        show_fps: cli.show_fps,
        config_path: cli.config,
    };

    let mut app = CbtopApp::new(config)?;
    app.run()
}
