//! CPU SIMD profiling via perf stat + renacer + trueno-explain.
//! Spec section 4.2.

use anyhow::Result;

/// perf stat hardware counters for SIMD analysis.
pub const SIMD_PERF_EVENTS: &[&str] = &[
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "L1-dcache-load-misses",
    "LLC-loads",
    "branches",
    "branch-misses",
];

/// Architecture-specific perf events for SIMD utilization.
pub const AVX2_EVENTS: &[&str] = &[
    "fp_arith_inst_retired.scalar_single",
    "fp_arith_inst_retired.128b_packed_single",
    "fp_arith_inst_retired.256b_packed_single",
];

pub const AVX512_EVENTS: &[&str] = &[
    "fp_arith_inst_retired.scalar_single",
    "fp_arith_inst_retired.256b_packed_single",
    "fp_arith_inst_retired.512b_packed_single",
];

/// Profile a SIMD function.
pub fn profile_simd(function: &str, size: u32, arch: &str) -> Result<()> {
    println!("\n=== CGP SIMD Profile: {function} (size={size}, arch={arch}) ===\n");

    // Validate architecture
    let events = match arch {
        "avx2" => {
            #[cfg(target_arch = "x86_64")]
            {
                if !std::arch::is_x86_feature_detected!("avx2") {
                    println!("  Warning: AVX2 not available on this CPU. Results may use scalar fallback.");
                }
            }
            AVX2_EVENTS
        }
        "avx512" => {
            #[cfg(target_arch = "x86_64")]
            {
                if !std::arch::is_x86_feature_detected!("avx512f") {
                    println!("  Warning: AVX-512 not available on this CPU.");
                }
            }
            AVX512_EVENTS
        }
        "neon" => {
            #[cfg(not(target_arch = "aarch64"))]
            {
                println!("  NEON not available -- use --cross-profile for QEMU-based analysis");
                return Ok(());
            }
            #[cfg(target_arch = "aarch64")]
            {
                &["INST_RETIRED", "CPU_CYCLES", "ASE_SPEC"][..]
            }
        }
        "sse2" => {
            &["fp_arith_inst_retired.scalar_single", "fp_arith_inst_retired.128b_packed_single"][..]
        }
        _ => {
            anyhow::bail!("Unknown SIMD architecture: {arch}. Supported: avx2, avx512, neon, sse2")
        }
    };

    // Check for perf
    let has_perf = which::which("perf").is_ok();
    if has_perf {
        println!("  Backend: perf stat");
        println!("  Base counters: {}", SIMD_PERF_EVENTS.join(", "));
        println!("  SIMD counters: {}", events.join(", "));
    } else {
        println!("  perf not found. Using wall-clock timing only.");
    }

    // Check for renacer
    let has_renacer = which::which("renacer").is_ok();
    if has_renacer {
        println!("  Syscall tracing: renacer");
    }

    println!("  Function: {function}");
    println!("  Size: {size}");
    println!("  Architecture: {arch}");
    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_events_defined() {
        assert!(!SIMD_PERF_EVENTS.is_empty());
        assert!(!AVX2_EVENTS.is_empty());
        assert!(!AVX512_EVENTS.is_empty());
    }

    #[test]
    fn test_invalid_arch_rejected() {
        let result = profile_simd("test_fn", 1024, "invalid_arch");
        assert!(result.is_err());
    }
}
