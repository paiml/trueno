//! cbtop - Compute Block Top
//!
//! Real-time load testing and hardware monitoring TUI.
//!
//! Run with: cargo run -p cbtop
//!
//! Headless mode for CI/CD and AI agents:
//!   cbtop --headless --format json --duration 5

use cbtop::config::{ComputeBackend, LoadProfile, WorkloadType};
use cbtop::headless::{BenchmarkResult, HeadlessBenchmark, OutputFormat};
use cbtop::optimize::{BaselineReport, OptimizationSuite, RegressionDetector};
use cbtop::{CbtopApp, CbtopError, Config};
use clap::{Parser, Subcommand};

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

    /// Optimization identification and regression detection
    Optimize {
        #[command(subcommand)]
        action: OptimizeAction,
    },
}

#[derive(Subcommand, Debug)]
enum OptimizeAction {
    /// Collect baseline measurements for all configurations
    Baseline {
        /// Output file for baseline JSON
        #[arg(short, long, default_value = "benchmarks/baseline.json")]
        output: std::path::PathBuf,

        /// Use quick mode (fewer configurations, shorter duration)
        #[arg(long)]
        quick: bool,

        /// Duration per benchmark in seconds
        #[arg(short, long, default_value = "3")]
        duration: u64,
    },

    /// Analyze baseline for performance bottlenecks
    Analyze {
        /// Baseline file to analyze
        #[arg(short, long, default_value = "benchmarks/baseline.json")]
        baseline: std::path::PathBuf,

        /// Output format: text, json
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Output file (stdout if not specified)
        #[arg(short, long)]
        output: Option<std::path::PathBuf>,
    },

    /// Check for performance regressions against baseline
    Check {
        /// Baseline file to compare against
        #[arg(short, long, default_value = "benchmarks/baseline.json")]
        baseline: std::path::PathBuf,

        /// Regression threshold percentage
        #[arg(short, long, default_value = "5.0")]
        threshold: f64,

        /// Use quick mode for current measurements
        #[arg(long)]
        quick: bool,

        /// Output format: text, json
        #[arg(short, long, default_value = "text")]
        format: String,
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

    // Handle subcommands
    match cli.command {
        Some(Commands::Bench {
            backend,
            workload,
            size,
            duration,
            format,
            output,
            baseline,
            fail_on_regression,
            compare,
        }) => {
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
        Some(Commands::Optimize { action }) => {
            return run_optimize(action);
        }
        None => {}
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
        std::fs::write(&path, &output_str).map_err(|e| CbtopError::Io(e.to_string()))?;
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
            std::fs::write(&path, &output_str).map_err(|e| CbtopError::Io(e.to_string()))?;
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
        let baseline_str =
            std::fs::read_to_string(&baseline_path).map_err(|e| CbtopError::Io(e.to_string()))?;
        let baseline_result: BenchmarkResult =
            serde_json::from_str(&baseline_str).map_err(|e| CbtopError::Config(e.to_string()))?;

        let regression = result.check_regression(&baseline_result, fail_on_regression);

        let output_str = regression.format(output_format);
        if let Some(path) = output {
            std::fs::write(&path, &output_str).map_err(|e| CbtopError::Io(e.to_string()))?;
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
        std::fs::write(&path, &output_str).map_err(|e| CbtopError::Io(e.to_string()))?;
        eprintln!("Results written to: {}", path.display());
    } else {
        println!("{}", output_str);
    }

    Ok(())
}

/// Run optimization subcommands (OPT-005)
fn run_optimize(action: OptimizeAction) -> Result<(), CbtopError> {
    match action {
        OptimizeAction::Baseline {
            output,
            quick,
            duration,
        } => run_optimize_baseline(output, quick, duration),
        OptimizeAction::Analyze {
            baseline,
            format,
            output,
        } => run_optimize_analyze(baseline, &format, output),
        OptimizeAction::Check {
            baseline,
            threshold,
            quick,
            format,
        } => run_optimize_check(baseline, threshold, quick, &format),
    }
}

fn run_optimize_baseline(
    output: std::path::PathBuf,
    quick: bool,
    duration: u64,
) -> Result<(), CbtopError> {
    eprintln!("Collecting baseline measurements...");

    let mut suite = if quick {
        OptimizationSuite::quick()
    } else {
        OptimizationSuite::standard()
    };
    suite.duration = std::time::Duration::from_secs(duration);

    let total_configs = suite.workloads.len() * suite.sizes.len() * suite.backends.len();
    eprintln!(
        "Running {} configurations ({} workloads x {} sizes x {} backends)",
        total_configs,
        suite.workloads.len(),
        suite.sizes.len(),
        suite.backends.len()
    );

    let baseline = suite.collect_baseline()?;

    // Ensure output directory exists
    if let Some(parent) = output.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CbtopError::Io(format!("Failed to create directory: {}", e)))?;
        }
    }

    baseline.save(&output)?;
    eprintln!("Baseline saved to: {}", output.display());

    // Print summary
    eprintln!("\nBaseline Summary:");
    eprintln!("  Entries: {}", baseline.entries.len());

    let avg_gflops: f64 =
        baseline.entries.iter().map(|e| e.gflops).sum::<f64>() / baseline.entries.len() as f64;
    eprintln!("  Average GFLOP/s: {:.2}", avg_gflops);

    let avg_efficiency: f64 =
        baseline.entries.iter().map(|e| e.efficiency).sum::<f64>() / baseline.entries.len() as f64;
    eprintln!("  Average Efficiency: {:.1}%", avg_efficiency * 100.0);

    Ok(())
}

fn run_optimize_analyze(
    baseline_path: std::path::PathBuf,
    format: &str,
    output: Option<std::path::PathBuf>,
) -> Result<(), CbtopError> {
    let baseline = BaselineReport::load(&baseline_path)?;
    let suite = OptimizationSuite::standard();
    let analysis = suite.analyze_bottlenecks(&baseline);

    let report = if format == "json" {
        serde_json::to_string_pretty(&analysis)
            .map_err(|e| CbtopError::Config(format!("JSON serialization failed: {}", e)))?
    } else {
        analysis.format_report()
    };

    if let Some(path) = output {
        std::fs::write(&path, &report).map_err(|e| CbtopError::Io(e.to_string()))?;
        eprintln!("Analysis saved to: {}", path.display());
    } else {
        println!("{}", report);
    }

    // Print summary to stderr
    eprintln!("\nAnalysis Summary:");
    eprintln!("  Critical: {}", analysis.summary.critical_count);
    eprintln!("  Severe: {}", analysis.summary.severe_count);
    eprintln!("  Moderate: {}", analysis.summary.moderate_count);
    eprintln!("  Unstable: {}", analysis.summary.unstable_count);

    Ok(())
}

fn run_optimize_check(
    baseline_path: std::path::PathBuf,
    threshold: f64,
    quick: bool,
    format: &str,
) -> Result<(), CbtopError> {
    eprintln!("Checking for regressions (threshold: {}%)...", threshold);

    // Load baseline
    let baseline = BaselineReport::load(&baseline_path)?;

    // Collect current measurements
    let suite = if quick {
        OptimizationSuite::quick()
    } else {
        OptimizationSuite::standard()
    };
    let current = suite.collect_baseline()?;

    // Check for regressions
    let detector = RegressionDetector::new(baseline, threshold);
    let report = detector.check(&current);

    let output = if format == "json" {
        serde_json::to_string_pretty(&report)
            .map_err(|e| CbtopError::Config(format!("JSON serialization failed: {}", e)))?
    } else {
        report.format_report()
    };

    println!("{}", output);

    // Exit with appropriate code
    std::process::exit(report.exit_code());
}
