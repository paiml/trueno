//! cbtop - Compute Block Top
//!
//! Real-time load testing and hardware monitoring TUI.
//!
//! Run with: cargo run -p cbtop
//!
//! Headless mode for CI/CD and AI agents:
//!   cbtop --headless --format json --duration 5

use clap::{Parser, Subcommand};
use cbtop::{Config, CbtopApp, CbtopError};
use cbtop::config::{ComputeBackend, LoadProfile, WorkloadType};
use cbtop::headless::{HeadlessBenchmark, BenchmarkResult, OutputFormat};

/// Compute Block Top - Real-time load testing and hardware monitoring TUI
#[derive(Parser, Debug)]
#[command(name = "cbtop")]
#[command(author = "Trueno Engineering")]
#[command(version)]
#[command(about = "Real-time load testing and hardware monitoring TUI", long_about = None)]
struct Cli {
    /// Subcommand (bench for headless benchmarking)
    #[command(subcommand)]
    command: Option<Commands>,

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

    /// Run in headless mode (no TUI, for CI/CD and AI agents)
    #[arg(long)]
    headless: bool,

    /// Output format for headless mode: json, text
    #[arg(long, default_value = "text")]
    format: String,

    /// Benchmark duration in seconds (headless mode)
    #[arg(long, default_value = "5")]
    duration: u64,

    /// Output file path (headless mode)
    #[arg(short, long)]
    output: Option<std::path::PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run benchmark in headless mode
    Bench {
        /// Compute backend: simd, wgpu, cuda, all
        #[arg(short, long, default_value = "simd")]
        backend: String,

        /// Workload type: gemm, dot, elementwise, reduction
        #[arg(short, long, default_value = "gemm")]
        workload: String,

        /// Problem size in elements
        #[arg(short, long, default_value = "1048576")]
        size: usize,

        /// Benchmark duration in seconds
        #[arg(short, long, default_value = "5")]
        duration: u64,

        /// Output format: json, text
        #[arg(short, long, default_value = "json")]
        format: String,

        /// Output file path
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,

        /// Baseline file for regression comparison
        #[arg(long)]
        baseline: Option<std::path::PathBuf>,

        /// Fail if regression exceeds this percentage
        #[arg(long, default_value = "5.0")]
        fail_on_regression: f64,

        /// Compare multiple backends (comma-separated)
        #[arg(long)]
        compare: Option<String>,
    },
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

fn parse_output_format(s: &str) -> OutputFormat {
    match s.to_lowercase().as_str() {
        "json" => OutputFormat::Json,
        _ => OutputFormat::Text,
    }
}

fn main() -> Result<(), CbtopError> {
    let cli = Cli::parse();

    // Handle bench subcommand
    if let Some(Commands::Bench {
        backend,
        workload,
        size,
        duration,
        format,
        output,
        baseline,
        fail_on_regression,
        compare,
    }) = cli.command
    {
        return run_bench(
            &backend,
            &workload,
            size,
            duration,
            &format,
            output,
            baseline,
            fail_on_regression,
            compare,
        );
    }

    // Handle headless mode
    if cli.headless {
        return run_headless(
            &cli.backend,
            &cli.workload,
            cli.size,
            cli.duration,
            &cli.format,
            cli.output,
        );
    }

    // Normal TUI mode
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

fn run_headless(
    backend: &str,
    workload: &str,
    size: usize,
    duration: u64,
    format: &str,
    output: Option<std::path::PathBuf>,
) -> Result<(), CbtopError> {
    let benchmark = HeadlessBenchmark::new(
        parse_backend(backend),
        parse_workload(workload),
        size,
        std::time::Duration::from_secs(duration),
    );

    let result = benchmark.run()?;
    let output_format = parse_output_format(format);
    let output_str = result.format(output_format);

    if let Some(path) = output {
        std::fs::write(&path, &output_str)
            .map_err(|e| CbtopError::Io(e.to_string()))?;
        eprintln!("Results written to: {}", path.display());
    } else {
        println!("{}", output_str);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_bench(
    backend: &str,
    workload: &str,
    size: usize,
    duration: u64,
    format: &str,
    output: Option<std::path::PathBuf>,
    baseline: Option<std::path::PathBuf>,
    fail_on_regression: f64,
    compare: Option<String>,
) -> Result<(), CbtopError> {
    let output_format = parse_output_format(format);

    // Handle comparison mode
    if let Some(backends_str) = compare {
        let backends: Vec<&str> = backends_str.split(',').collect();
        let mut results = Vec::new();

        for b in backends {
            let benchmark = HeadlessBenchmark::new(
                parse_backend(b.trim()),
                parse_workload(workload),
                size,
                std::time::Duration::from_secs(duration),
            );
            let result = benchmark.run()?;
            results.push((b.trim().to_string(), result));
        }

        // Output comparison
        let comparison = BenchmarkResult::compare(&results);
        let output_str = comparison.format(output_format);

        if let Some(path) = output {
            std::fs::write(&path, &output_str)
                .map_err(|e| CbtopError::Io(e.to_string()))?;
        } else {
            println!("{}", output_str);
        }

        return Ok(());
    }

    // Single benchmark
    let benchmark = HeadlessBenchmark::new(
        parse_backend(backend),
        parse_workload(workload),
        size,
        std::time::Duration::from_secs(duration),
    );

    let result = benchmark.run()?;

    // Check for regression if baseline provided
    if let Some(baseline_path) = baseline {
        let baseline_str = std::fs::read_to_string(&baseline_path)
            .map_err(|e| CbtopError::Io(e.to_string()))?;
        let baseline_result: BenchmarkResult = serde_json::from_str(&baseline_str)
            .map_err(|e| CbtopError::Config(e.to_string()))?;

        let regression = result.check_regression(&baseline_result, fail_on_regression);

        let output_str = regression.format(output_format);
        if let Some(path) = output {
            std::fs::write(&path, &output_str)
                .map_err(|e| CbtopError::Io(e.to_string()))?;
        } else {
            println!("{}", output_str);
        }

        if regression.is_regression {
            std::process::exit(1);
        }
        return Ok(());
    }

    // Output result
    let output_str = result.format(output_format);
    if let Some(path) = output {
        std::fs::write(&path, &output_str)
            .map_err(|e| CbtopError::Io(e.to_string()))?;
        eprintln!("Results written to: {}", path.display());
    } else {
        println!("{}", output_str);
    }

    Ok(())
}
