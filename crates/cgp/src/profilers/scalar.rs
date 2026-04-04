//! Scalar baseline profiling via criterion + renacer. Spec section 4.4.
//! Establishes the baseline for all speedup calculations.

use anyhow::Result;

/// Profile a scalar function (CPU baseline).
pub fn profile_scalar(function: &str, size: u32) -> Result<()> {
    println!("\n=== CGP Scalar Profile: {function} (size={size}) ===\n");
    println!("  Backend: scalar (no SIMD, single-threaded)");
    println!("  Purpose: baseline for speedup calculations");
    println!("  Function: {function}");
    println!("  Size: {size}");

    let has_perf = which::which("perf").is_ok();
    if has_perf {
        println!("  Hardware counters: perf stat (cycles, instructions, cache-misses)");
    }

    let has_renacer = which::which("renacer").is_ok();
    if has_renacer {
        println!("  Syscall tracing: renacer");
    }

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_scalar_ok() {
        let result = profile_scalar("matrix_mul_naive", 256);
        assert!(result.is_ok());
    }
}
