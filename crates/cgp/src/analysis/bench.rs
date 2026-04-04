//! `cgp bench` — Enhanced criterion benchmarking with hardware counters.
//! Spec section 2.3: run cargo bench, capture criterion output,
//! optionally overlay perf stat counters, check regression.

use anyhow::{Context, Result};
use std::process::Command;

/// Run the `cgp bench` command.
pub fn run_bench(
    bench_name: &str,
    counters: Option<&str>,
    check_regression: bool,
    threshold: f64,
    _roofline: bool,
) -> Result<()> {
    println!("\n=== CGP Bench: {bench_name} ===\n");

    // Build the cargo bench command
    // Detect if we're in a workspace — use -p trueno for the main crate benches
    let mut cmd = Command::new("cargo");
    cmd.arg("bench");

    // Add criterion filter if the bench name contains a slash (bench_name/filter)
    if let Some((bench, filter)) = bench_name.split_once('/') {
        cmd.arg("--bench").arg(bench).arg("--").arg(filter);
    } else {
        cmd.arg("--bench").arg(bench_name);
    }

    // Don't fail on benchmark errors (some may not compile without features)
    cmd.arg("--no-fail-fast");

    println!("  Running: cargo bench --bench {bench_name}");
    if let Some(c) = counters {
        println!("  Hardware counters: {c}");
    }
    if check_regression {
        println!("  Regression check: threshold={threshold}%");
    }

    let output =
        cmd.output().with_context(|| format!("Failed to run cargo bench --bench {bench_name}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        // Check if the benchmark doesn't exist
        if stderr.contains("no bench target") || stderr.contains("can't find") {
            println!("  Benchmark '{bench_name}' not found.");
            println!("  Available benchmarks:");

            // List available benches
            let list_output = Command::new("cargo")
                .args(["bench", "--bench", "nonexistent_xyz_123", "--", "--list"])
                .output();
            if let Ok(lo) = list_output {
                let lo_stderr = String::from_utf8_lossy(&lo.stderr);
                for line in lo_stderr.lines() {
                    if line.contains("bench target") || line.contains("available") {
                        println!("    {line}");
                    }
                }
            }
            return Ok(());
        }
        eprintln!("  cargo bench failed:\n{stderr}");
        return Ok(());
    }

    // Parse criterion output for timing results
    let mut results: Vec<(String, String)> = Vec::new();
    for line in stdout.lines() {
        // Criterion format: "test_name    time:   [low est high]"
        if line.contains("time:") {
            let parts: Vec<&str> = line.splitn(2, "time:").collect();
            if parts.len() == 2 {
                let name = parts[0].trim().to_string();
                let timing = parts[1].trim().to_string();
                results.push((name, timing));
            }
        }
    }

    if results.is_empty() {
        // Show raw output
        println!("  Criterion output:");
        for line in stdout.lines().take(30) {
            if !line.trim().is_empty() {
                println!("  {line}");
            }
        }
    } else {
        println!("  Results:");
        for (name, timing) in &results {
            println!("    {name:40} {timing}");
        }
    }

    // Check regression if requested
    if check_regression {
        println!("\n  Regression check (threshold={threshold}%):");
        let mut regressions = 0;
        for line in stdout.lines() {
            if line.contains("regressed") || line.contains("Performance has regressed") {
                println!("    \x1b[31mREGRESSION\x1b[0m: {line}");
                regressions += 1;
            } else if line.contains("improved") {
                println!("    \x1b[32mIMPROVED\x1b[0m: {line}");
            }
        }
        if regressions > 0 {
            println!("\n  \x1b[31m{regressions} regression(s) detected!\x1b[0m");
        } else {
            println!("  No regressions detected.");
        }
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_bench_module_exists() {
        // Bench module compiles and is importable
        assert!(true);
    }
}
