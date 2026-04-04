//! ARM NEON SIMD profiling. Spec section 4.5.
//! Uses perf stat with ARM PMU counters on aarch64 hosts.
//! On x86 hosts, reports graceful error per FALSIFY-CGP-071.

/// ARM PMU counters for NEON profiling.
pub const ARM_PMU_EVENTS: &[&str] = &["INST_RETIRED", "CPU_CYCLES", "ASE_SPEC"];

/// Whether we're running natively on ARM.
pub fn is_native_arm() -> bool {
    cfg!(target_arch = "aarch64")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-CGP-071: On x86 host, NEON should not be marked as native.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_neon_not_native_on_x86() {
        assert!(!is_native_arm());
    }

    #[test]
    fn test_arm_events_defined() {
        assert_eq!(ARM_PMU_EVENTS.len(), 3);
    }
}
