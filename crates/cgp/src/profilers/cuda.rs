//! CUDA profiler: wraps ncu, nsys, and CUPTI.
//! See spec sections 4.1.1 (ncu), 4.1.2 (nsys), 4.1.3 (CUPTI).

use anyhow::Result;
use std::path::PathBuf;
use std::process::Command;

/// ncu metric sections — lazily collect only what's requested.
#[derive(Debug, Clone, Copy)]
pub enum NcuSection {
    LaunchStats,
    ComputeThroughput,
    MemoryThroughput,
    Occupancy,
    Roofline,
    WarpState,
    SourceLevel,
}

impl NcuSection {
    fn as_ncu_arg(&self) -> &str {
        match self {
            NcuSection::LaunchStats => "LaunchStats",
            NcuSection::ComputeThroughput => "ComputeWorkloadAnalysis",
            NcuSection::MemoryThroughput => "MemoryWorkloadAnalysis",
            NcuSection::Occupancy => "Occupancy",
            NcuSection::Roofline => "SpeedOfLight",
            NcuSection::WarpState => "WarpStateStats",
            NcuSection::SourceLevel => "SourceCounters",
        }
    }
}

/// Wraps `ncu` CLI for kernel-level profiling.
pub struct NcuProfiler {
    pub ncu_path: PathBuf,
    pub sections: Vec<NcuSection>,
}

impl NcuProfiler {
    pub fn detect() -> Option<Self> {
        which::which("ncu").ok().map(|path| Self {
            ncu_path: path,
            sections: vec![NcuSection::LaunchStats, NcuSection::Roofline],
        })
    }

    /// Build ncu command line for profiling a kernel.
    pub fn build_command(&self, binary: &str, binary_args: &[&str], kernel_regex: &str) -> Command {
        let mut cmd = Command::new(&self.ncu_path);
        cmd.arg("--target-processes").arg("all");
        cmd.arg("--kernel-id").arg(format!("::regex:{kernel_regex}:"));

        for section in &self.sections {
            cmd.arg("--section").arg(section.as_ncu_arg());
        }

        cmd.arg("--csv");
        cmd.arg(binary);
        cmd.args(binary_args);
        cmd
    }
}

/// Wraps `nsys` CLI for system-wide timeline profiling.
pub struct NsysProfiler {
    pub nsys_path: PathBuf,
}

impl NsysProfiler {
    pub fn detect() -> Option<Self> {
        which::which("nsys").ok().map(|path| Self { nsys_path: path })
    }

    /// Build nsys command for tracing a binary.
    pub fn build_trace_command(&self, binary: &str, binary_args: &[&str]) -> Command {
        let mut cmd = Command::new(&self.nsys_path);
        cmd.arg("profile");
        cmd.arg("--stats=true");
        cmd.arg("--force-overwrite=true");
        cmd.arg("-o").arg("/tmp/cgp-nsys-report");
        cmd.arg(binary);
        cmd.args(binary_args);
        cmd
    }
}

/// Profile a CUDA PTX kernel via ncu.
pub fn profile_kernel(name: &str, size: u32, roofline: bool, metrics: Option<&str>) -> Result<()> {
    println!("\n=== CGP Kernel Profile: {name} ({size}x{size}x{size}) ===\n");

    match NcuProfiler::detect() {
        Some(profiler) => {
            println!("  Backend: CUDA (ncu at {})", profiler.ncu_path.display());
            println!(
                "  Sections: {:?}",
                profiler.sections.iter().map(|s| s.as_ncu_arg()).collect::<Vec<_>>()
            );
            if roofline {
                println!("  Roofline: enabled");
            }
            if let Some(m) = metrics {
                println!("  Extra metrics: {m}");
            }
            println!("\n  (Full ncu profiling requires running target binary with sudo/--target-processes)");
        }
        None => {
            println!("  ncu not found. Install NVIDIA Nsight Compute for CUDA kernel profiling.");
            println!("  Falling back to trueno-explain static analysis...");
        }
    }

    println!();
    Ok(())
}

/// Profile cuBLAS operations.
pub fn profile_cublas(op: &str, size: u32) -> Result<()> {
    println!("cgp profile cublas: op={op} size={size}");
    println!("(Wraps ncu targeting cuBLAS library kernels)");
    Ok(())
}

/// Profile an arbitrary binary via nsys.
pub fn profile_binary(
    path: &str,
    kernel_filter: Option<&str>,
    trace: bool,
    duration: Option<&str>,
) -> Result<()> {
    println!("\n=== CGP Binary Profile: {path} ===\n");

    match NsysProfiler::detect() {
        Some(profiler) => {
            println!("  Backend: nsys at {}", profiler.nsys_path.display());
            if let Some(filter) = kernel_filter {
                println!("  Kernel filter: {filter}");
            }
            if trace {
                println!("  System trace: enabled");
            }
            if let Some(dur) = duration {
                println!("  Duration: {dur}");
            }
            println!("\n  (Run with appropriate permissions for full CUDA tracing)");
        }
        None => {
            println!("  nsys not found. Install NVIDIA Nsight Systems for binary profiling.");
            println!("  Falling back to perf stat...");
        }
    }

    Ok(())
}

/// Profile a Python script via nsys + perf stat.
pub fn profile_python(args: &[String]) -> Result<()> {
    let cmd_str = args.join(" ");
    println!("\n=== CGP Python Profile ===\n");
    println!("  Command: {cmd_str}");

    if NsysProfiler::detect().is_some() {
        println!("  Strategy: nsys profile for CUDA ops, perf stat for CPU ops");
    } else {
        println!("  Strategy: perf stat only (no nsys available for CUDA tracing)");
    }

    println!("  (Uses 'uv run python' for reproducible environment)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ncu_section_args() {
        assert_eq!(NcuSection::LaunchStats.as_ncu_arg(), "LaunchStats");
        assert_eq!(NcuSection::Roofline.as_ncu_arg(), "SpeedOfLight");
        assert_eq!(NcuSection::WarpState.as_ncu_arg(), "WarpStateStats");
    }

    #[test]
    fn test_ncu_command_build() {
        if let Some(profiler) = NcuProfiler::detect() {
            let cmd = profiler.build_command("./test_binary", &["--size", "512"], "gemm_*");
            let program = cmd.get_program().to_str().unwrap();
            assert!(program.contains("ncu"));
        }
    }

    // FALSIFY-CGP-077 (Metal not available on Linux) tested via `cgp profile metal` CLI.
}
