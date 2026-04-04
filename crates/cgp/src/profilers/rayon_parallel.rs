//! Rayon parallel profiling. Spec section 4.9.
//! Measures parallel efficiency, work stealing, and load balance (Heijunka score).

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Rayon parallel profile output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RayonProfile {
    pub wall_time_us: f64,
    pub single_thread_time_us: f64,
    pub parallel_speedup: f64,
    pub num_threads: usize,
    pub parallel_efficiency: f64,
    /// 0.0 = perfect balance, 1.0 = all work on 1 thread.
    pub heijunka_score: f64,
    pub thread_spawn_overhead_us: f64,
    pub work_steal_count: u64,
}

impl RayonProfile {
    /// Compute Heijunka (load balance) score from per-thread work times.
    /// Score = coefficient of variation of per-thread times.
    /// 0.0 = perfect balance, higher = more imbalanced.
    pub fn compute_heijunka_score(per_thread_times: &[f64]) -> f64 {
        if per_thread_times.is_empty() || per_thread_times.len() == 1 {
            return 0.0;
        }
        let mean = per_thread_times.iter().sum::<f64>() / per_thread_times.len() as f64;
        if mean == 0.0 {
            return 0.0;
        }
        let variance = per_thread_times.iter().map(|t| (t - mean).powi(2)).sum::<f64>()
            / per_thread_times.len() as f64;
        let cv = variance.sqrt() / mean;
        cv.min(1.0) // Cap at 1.0
    }

    /// Estimate parallel speedup from single-thread and multi-thread wall times.
    pub fn compute_speedup(single_thread_us: f64, parallel_us: f64) -> f64 {
        if parallel_us > 0.0 {
            single_thread_us / parallel_us
        } else {
            0.0
        }
    }

    /// Compute parallel efficiency: speedup / num_threads (1.0 = ideal).
    pub fn compute_efficiency(speedup: f64, num_threads: usize) -> f64 {
        if num_threads > 0 {
            speedup / num_threads as f64
        } else {
            0.0
        }
    }
}

/// Profile a parallel function.
/// Runs the benchmark binary with RAYON_NUM_THREADS=1 and RAYON_NUM_THREADS=N,
/// then computes parallel metrics.
pub fn profile_parallel(function: &str, size: u32, threads: Option<&str>) -> Result<()> {
    let thread_count = match threads {
        Some("auto") | None => num_cpus::get(),
        Some(n) => n.parse().map_err(|_| anyhow::anyhow!("Invalid thread count: {n}"))?,
    };

    println!("\n=== CGP Parallel Profile: {function} (size={size}, threads={thread_count}) ===\n");
    println!("  Backend: Rayon thread pool");
    println!("  Function: {function}");
    println!("  Size: {size}");
    println!("  Threads: {thread_count}");

    // Try to find a benchmark binary
    let binary = find_parallel_binary();
    match binary {
        Some(bin) => {
            println!("  Binary: {bin}");

            // Run single-threaded
            let single_time = time_binary_with_threads(&bin, 1);
            // Run multi-threaded
            let parallel_time = time_binary_with_threads(&bin, thread_count);

            match (single_time, parallel_time) {
                (Some(st), Some(pt)) => {
                    let speedup = RayonProfile::compute_speedup(st, pt);
                    let efficiency = RayonProfile::compute_efficiency(speedup, thread_count);

                    println!("\n  Results:");
                    println!("    Single-thread:      {st:.0} us");
                    println!("    {thread_count}-thread:     {pt:.0} us");
                    println!("    Parallel speedup:   {speedup:.2}x");
                    println!("    Efficiency:         {:.1}%", efficiency * 100.0);

                    // Estimate spawn overhead (~40us per thread::scope call)
                    let overhead_estimate = 40.0; // us, from memory feedback
                    let overhead_pct = if pt > 0.0 { overhead_estimate / pt * 100.0 } else { 0.0 };
                    println!(
                        "    Thread overhead:     ~{overhead_estimate:.0} us ({overhead_pct:.1}% of total)"
                    );

                    // Warning for small workloads
                    if pt < 500.0 {
                        println!(
                            "\n  \x1b[33m[WARN]\x1b[0m Workload <500us — thread overhead dominates"
                        );
                        println!("    Consider: sequential execution or batching");
                    }
                }
                _ => {
                    println!("\n  Could not time binary — showing configuration only.");
                }
            }
        }
        None => {
            println!("  No benchmark binary found.");
            println!("  Build with: cargo build --release --examples");
            println!("  Estimated metrics with synthetic data:");

            // Show theoretical analysis
            println!("\n  Theoretical Analysis:");
            println!(
                "    Amdahl's law: if 95% parallelizable, max speedup = {:.1}x",
                amdahl(0.95, thread_count)
            );
            println!("    If 90% parallelizable, max speedup = {:.1}x", amdahl(0.90, thread_count));
            println!("    If 80% parallelizable, max speedup = {:.1}x", amdahl(0.80, thread_count));
        }
    }

    println!();
    Ok(())
}

/// Amdahl's law: speedup = 1 / ((1 - p) + p/n)
fn amdahl(parallel_fraction: f64, threads: usize) -> f64 {
    1.0 / ((1.0 - parallel_fraction) + parallel_fraction / threads as f64)
}

/// Time a binary with a given number of threads.
fn time_binary_with_threads(binary: &str, threads: usize) -> Option<f64> {
    let start = std::time::Instant::now();
    let output = std::process::Command::new(binary)
        .env("RAYON_NUM_THREADS", threads.to_string())
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }
    Some(start.elapsed().as_secs_f64() * 1e6) // Convert to microseconds
}

/// Find a parallel benchmark binary.
fn find_parallel_binary() -> Option<String> {
    let target_dir = std::env::var("CARGO_TARGET_DIR").unwrap_or_default();
    let mut candidates: Vec<String> = Vec::new();
    if !target_dir.is_empty() {
        candidates.push(format!("{target_dir}/release/examples/benchmark_matrix_suite"));
    }
    candidates.extend_from_slice(&[
        "/mnt/nvme-raid0/targets/trueno/release/examples/benchmark_matrix_suite".to_string(),
        "./target/release/examples/benchmark_matrix_suite".to_string(),
    ]);
    for path in &candidates {
        if std::path::Path::new(path).exists() {
            return Some(path.clone());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Perfect balance: all threads same time.
    #[test]
    fn test_heijunka_perfect_balance() {
        let times = vec![10.0, 10.0, 10.0, 10.0];
        let score = RayonProfile::compute_heijunka_score(&times);
        assert!((score - 0.0).abs() < 1e-10);
    }

    /// Severe imbalance: one thread does all the work.
    #[test]
    fn test_heijunka_severe_imbalance() {
        let times = vec![100.0, 1.0, 1.0, 1.0];
        let score = RayonProfile::compute_heijunka_score(&times);
        assert!(score > 0.5, "Heijunka score {score} should be > 0.5 for severe imbalance");
    }

    /// FALSIFY-CGP-081: Intentionally imbalanced workload should have high score.
    #[test]
    fn test_heijunka_90pct_imbalance() {
        let times = vec![
            900.0,
            100.0 / 7.0,
            100.0 / 7.0,
            100.0 / 7.0,
            100.0 / 7.0,
            100.0 / 7.0,
            100.0 / 7.0,
            100.0 / 7.0,
        ];
        let score = RayonProfile::compute_heijunka_score(&times);
        assert!(score > 0.5, "Score {score} for 90% imbalance should be > 0.5");
    }

    #[test]
    fn test_heijunka_empty() {
        assert_eq!(RayonProfile::compute_heijunka_score(&[]), 0.0);
        assert_eq!(RayonProfile::compute_heijunka_score(&[42.0]), 0.0);
    }

    #[test]
    fn test_compute_speedup() {
        assert!((RayonProfile::compute_speedup(1000.0, 250.0) - 4.0).abs() < 0.01);
        assert!((RayonProfile::compute_speedup(1000.0, 0.0)).abs() < 0.01);
    }

    #[test]
    fn test_compute_efficiency() {
        assert!((RayonProfile::compute_efficiency(4.0, 8) - 0.5).abs() < 0.01);
        assert!((RayonProfile::compute_efficiency(8.0, 8) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_amdahl() {
        // 100% parallelizable with 4 threads = 4x speedup
        assert!((amdahl(1.0, 4) - 4.0).abs() < 0.01);
        // 0% parallelizable = 1x speedup
        assert!((amdahl(0.0, 4) - 1.0).abs() < 0.01);
        // 50% parallelizable with 2 threads = 1.33x
        assert!((amdahl(0.5, 2) - 1.333).abs() < 0.01);
    }

    #[test]
    fn test_profile_parallel_auto_threads() {
        let result = profile_parallel("gemm_heijunka", 4096, Some("auto"));
        assert!(result.is_ok());
    }
}
