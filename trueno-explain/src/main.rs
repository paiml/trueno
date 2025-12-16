//! trueno-explain CLI
//!
//! PTX/SIMD/wgpu Visualization and Tracing Tool
//!
//! Implements Toyota Way principle of Genchi Genbutsu (Go and See)

use clap::{Parser, Subcommand};
use std::process::ExitCode;
use trueno_explain::{output, Analyzer, OutputFormat, PtxAnalyzer};
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
        /// Baseline analysis file or git ref
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

        Commands::Simd { function, arch, json } => {
            eprintln!(
                "SIMD analysis for {} ({}) - coming in Sprint 2",
                function, arch
            );
            if json {
                println!("{{\"status\": \"not_implemented\", \"function\": \"{}\", \"arch\": \"{}\"}}", function, arch);
            }
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
            baseline,
            fail_on_regression,
            json,
        } => {
            eprintln!(
                "Diff analysis against {} - coming soon (fail_on_regression: {})",
                baseline, fail_on_regression
            );
            if json {
                println!("{{\"status\": \"not_implemented\", \"baseline\": \"{}\"}}", baseline);
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
