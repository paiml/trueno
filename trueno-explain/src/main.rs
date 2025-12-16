//! trueno-explain CLI
//!
//! PTX/SIMD/wgpu Visualization and Tracing Tool
//!
//! Implements Toyota Way principle of Genchi Genbutsu (Go and See)

use clap::{Parser, Subcommand};
use std::process::ExitCode;
use trueno_explain::{
    compare_reports, format_diff_json, format_diff_text, output, run_tui, Analyzer, DiffThresholds,
    OutputFormat, PtxAnalyzer, SimdAnalyzer, SimdArch,
};
use trueno_gpu::kernels::{GemmKernel, Kernel, Q5KKernel, Q6KKernel, QuantizeKernel, SoftmaxKernel};

#[derive(Parser)]
#[command(name = "trueno-explain")]
#[command(author, version, about = "PTX/SIMD/wgpu Visualization and Tracing CLI")]
#[command(long_about = "
Implements the Toyota Way principle of Genchi Genbutsu (Go and See)
by making invisible compiler transformations visible.

Detects Muda (waste):
  - Transport: Register spills
  - Waiting: Uncoalesced memory access
  - Overprocessing: Excessive precision/redundant ops
")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze PTX code generation
    Ptx {
        /// Kernel to analyze (see --help for list)
        #[arg(short, long, value_name = "NAME")]
        kernel: String,

        /// Matrix M dimension (rows)
        #[arg(short = 'm', long, default_value = "1024")]
        rows: u32,

        /// Matrix N dimension (columns)
        #[arg(short = 'n', long, default_value = "1024")]
        cols: u32,

        /// Matrix K dimension (inner)
        #[arg(short = 'k', long, default_value = "1024")]
        inner: u32,

        /// Show register pressure details
        #[arg(long)]
        registers: bool,

        /// Show memory access pattern
        #[arg(long)]
        memory_pattern: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Interactive TUI mode (Genchi Genbutsu)
    Tui {
        /// Kernel to explore
        #[arg(short, long, value_name = "NAME")]
        kernel: String,

        /// Matrix M dimension (rows)
        #[arg(short = 'm', long, default_value = "64")]
        rows: u32,

        /// Matrix N dimension (columns)
        #[arg(short = 'n', long, default_value = "64")]
        cols: u32,

        /// Matrix K dimension (inner)
        #[arg(short = 'k', long, default_value = "256")]
        inner: u32,
    },

    /// Analyze SIMD vectorization (coming soon)
    Simd {
        /// Function to analyze
        #[arg(short, long)]
        function: String,

        /// Target architecture (sse2, avx2, avx512, neon)
        #[arg(short, long, default_value = "avx2")]
        arch: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Analyze wgpu/WGSL shaders (coming soon)
    Wgpu {
        /// Shader file to analyze
        #[arg(short, long)]
        shader: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Compare backends
    Compare {
        /// Kernel to compare
        #[arg(short, long)]
        kernel: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Compare two analyses (git diff integration)
    Diff {
        /// Kernel to analyze for comparison
        #[arg(short, long)]
        kernel: String,

        /// Baseline analysis JSON file
        #[arg(long)]
        baseline: String,

        /// Fail with exit code 1 if regression detected
        #[arg(long)]
        fail_on_regression: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Ptx {
            kernel,
            rows,
            cols,
            inner,
            json,
            ..
        } => {
            let ptx = generate_kernel_ptx(&kernel, rows, cols, inner)?;
            let analyzer = PtxAnalyzer::new();
            let report = analyzer.analyze(&ptx)?;

            let format = if json {
                OutputFormat::Json
            } else {
                OutputFormat::Text
            };
            output::write_report(&report, format)?;
        }

        Commands::Tui {
            kernel,
            rows,
            cols,
            inner,
        } => {
            let ptx = generate_kernel_ptx(&kernel, rows, cols, inner)?;
            let analyzer = PtxAnalyzer::new();
            let report = analyzer.analyze(&ptx)?;
            run_tui(ptx, report)?;
        }

        Commands::Simd { function, arch, json } => {
            let simd_arch = match arch.to_lowercase().as_str() {
                "sse2" => SimdArch::Sse2,
                "avx" | "avx2" => SimdArch::Avx2,
                "avx512" | "avx-512" => SimdArch::Avx512,
                "neon" => SimdArch::Neon,
                _ => {
                    return Err(format!(
                        "Unknown SIMD architecture: {}. Available: sse2, avx2, avx512, neon",
                        arch
                    )
                    .into());
                }
            };

            // For now, analyze sample assembly or placeholder
            // In future, will integrate with objdump/llvm-objdump for real binaries
            let sample_asm = format!(
                "; Sample x86-64 assembly for function: {}\n\
                 ; Target architecture: {:?}\n\
                 ; Use --asm-file to analyze actual disassembly\n",
                function, simd_arch
            );

            let analyzer = SimdAnalyzer::new(simd_arch);
            let report = analyzer.analyze(&sample_asm)?;

            let format = if json {
                OutputFormat::Json
            } else {
                OutputFormat::Text
            };
            output::write_report(&report, format)?;
        }

        Commands::Wgpu { shader, json } => {
            eprintln!("wgpu/WGSL analysis for {} - coming in Sprint 3", shader);
            if json {
                println!("{{\"status\": \"not_implemented\", \"shader\": \"{}\"}}", shader);
            }
        }

        Commands::Compare { kernel, json } => {
            eprintln!("Backend comparison for {} - coming soon", kernel);
            if json {
                println!("{{\"status\": \"not_implemented\", \"kernel\": \"{}\"}}", kernel);
            }
        }

        Commands::Diff {
            kernel,
            baseline,
            fail_on_regression,
            json,
        } => {
            // Load baseline from JSON file
            let baseline_json = std::fs::read_to_string(&baseline)
                .map_err(|e| format!("Failed to read baseline file '{}': {}", baseline, e))?;
            let baseline_report: trueno_explain::AnalysisReport =
                serde_json::from_str(&baseline_json)
                    .map_err(|e| format!("Failed to parse baseline JSON: {}", e))?;

            // Analyze current kernel
            let ptx = generate_kernel_ptx(&kernel, 1024, 1024, 1024)?;
            let analyzer = PtxAnalyzer::new();
            let current_report = analyzer.analyze(&ptx)?;

            // Compare reports
            let thresholds = DiffThresholds::default();
            let diff_report = compare_reports(&baseline_report, &current_report, &thresholds);

            // Output results
            if json {
                println!("{}", format_diff_json(&diff_report));
            } else {
                print!("{}", format_diff_text(&diff_report));
            }

            // Exit with error if regression detected and --fail-on-regression is set
            if fail_on_regression && diff_report.has_regression {
                return Err("Regression detected".into());
            }
        }
    }

    Ok(())
}

fn generate_kernel_ptx(
    kernel: &str,
    m: u32,
    n: u32,
    k: u32,
) -> Result<String, Box<dyn std::error::Error>> {
    let ptx = match kernel.to_lowercase().as_str() {
        "vector_add" => {
            // Simple vector add PTX
            include_str!("../data/vector_add.ptx").to_string()
        }
        "gemm_naive" => {
            let kernel = GemmKernel::naive(m, n, k);
            kernel.emit_ptx()
        }
        "gemm_tiled" => {
            let kernel = GemmKernel::tiled(m, n, k, 32);
            kernel.emit_ptx()
        }
        "softmax" => {
            let kernel = SoftmaxKernel::new(m);
            kernel.emit_ptx()
        }
        "q4k_gemm" | "q4k" => {
            // k must be divisible by 256
            let k_aligned = (k / 256) * 256;
            let k_aligned = k_aligned.max(256);
            let kernel = QuantizeKernel::ggml(m, n, k_aligned);
            kernel.emit_ptx()
        }
        "q5k_gemm" | "q5k" => {
            let k_aligned = (k / 256) * 256;
            let k_aligned = k_aligned.max(256);
            let kernel = Q5KKernel::new(m, n, k_aligned);
            kernel.emit_ptx()
        }
        "q6k_gemm" | "q6k" => {
            let k_aligned = (k / 256) * 256;
            let k_aligned = k_aligned.max(256);
            let kernel = Q6KKernel::new(m, n, k_aligned);
            kernel.emit_ptx()
        }
        _ => {
            return Err(format!(
                "Unknown kernel: {}. Available: vector_add, gemm_naive, gemm_tiled, softmax, q4k_gemm, q5k_gemm, q6k_gemm",
                kernel
            )
            .into());
        }
    };

    Ok(ptx)
}
