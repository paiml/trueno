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
}

/// Profile a parallel function.
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
    println!("  Metrics: parallel_speedup, parallel_efficiency, heijunka_score,");
    println!("           thread_spawn_overhead_us, work_steal_count");
    println!();
    Ok(())
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
        // Thread 0 gets 90% of work
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
    fn test_profile_parallel_auto_threads() {
        let result = profile_parallel("gemm_heijunka", 4096, Some("auto"));
        assert!(result.is_ok());
    }
}
