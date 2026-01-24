use super::types::*;
use super::analyzer::*;
use super::coverage::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_mem_u64_detection() {
        let ptx = r#"
.visible .entry test() {
    .reg .u64 %rd<5>;
    .reg .f32 %f<2>;
    .shared .b8 smem[4096];
    st.shared.f32 [%rd0], %f0;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::SharedMemU64Addressing));
    }

    #[test]
    fn test_shared_mem_u32_valid() {
        let ptx = r#"
.visible .entry test() {
    .reg .u32 %r<5>;
    .reg .f32 %f<2>;
    .shared .b8 smem[4096];
    st.shared.f32 [%r0], %f0;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::SharedMemU64Addressing));
    }

    #[test]
    fn test_missing_barrier_sync_strict() {
        let ptx = r#"
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%r0], %f0;
    ld.shared.f32 %f1, [%r1];
    ret;
}
"#;
        // Non-strict mode: no warning
        let normal_result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(!normal_result.has_bug(&PtxBugClass::MissingBarrierSync));

        // Strict mode: warning
        let strict_result = PtxBugAnalyzer::strict().analyze(ptx);
        assert!(strict_result.has_bug(&PtxBugClass::MissingBarrierSync));
    }

    #[test]
    fn test_barrier_present_valid() {
        let ptx = r#"
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%r0], %f0;
    bar.sync 0;
    ld.shared.f32 %f1, [%r1];
    ret;
}
"#;
        let result = PtxBugAnalyzer::strict().analyze(ptx);
        // Should not have the broad "no bar.sync" warning
        let missing_barrier_bugs: Vec<_> = result.bugs_of_class(&PtxBugClass::MissingBarrierSync);
        // The specific st/ld pattern should not trigger since bar.sync is present
        assert!(missing_barrier_bugs
            .iter()
            .all(|b| !b.message.contains("ld.shared follows st.shared")));
    }

    #[test]
    fn test_loop_branch_to_end_detection() {
        let ptx = r#"
.visible .entry test() {
main_loop:
    // loop body
    bra main_loop_end;
main_loop_end:
    ret;
}
"#;
        let result = PtxBugAnalyzer::strict().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::LoopBranchToEnd));
    }

    #[test]
    fn test_conditional_branch_not_flagged() {
        let ptx = r#"
.visible .entry test() {
loop_start:
    @%p0 bra loop_end;
    bra loop_start;
loop_end:
    ret;
}
"#;
        let result = PtxBugAnalyzer::strict().analyze(ptx);
        // Conditional branch should NOT be flagged
        assert!(!result.has_bug(&PtxBugClass::LoopBranchToEnd));
    }

    #[test]
    fn test_register_spills_detection() {
        let ptx = r#"
.visible .entry test() {
    .local .align 4 .b8 __local_depot[32];
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::RegisterSpills));
    }

    #[test]
    fn test_missing_entry_point_detection() {
        let ptx = r#"
.version 8.0
.target sm_70
.reg .f32 %f<4>;
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::MissingEntryPoint));
    }

    #[test]
    fn test_valid_kernel_no_bugs() {
        let ptx = r#"
.version 8.0
.target sm_70
.visible .entry valid_kernel() {
    .reg .f32 %f<4>;
    .reg .u32 %r<4>;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.is_valid());
        assert!(!result.has_bugs());
    }

    #[test]
    fn test_bug_severity_classification() {
        assert_eq!(
            PtxBugClass::MissingBarrierSync.severity(),
            BugSeverity::Critical
        );
        assert_eq!(
            PtxBugClass::SharedMemU64Addressing.severity(),
            BugSeverity::Critical
        );
        assert_eq!(PtxBugClass::RegisterSpills.severity(), BugSeverity::High);
        assert_eq!(
            PtxBugClass::MissingEntryPoint.severity(),
            BugSeverity::FalsePositive
        );
    }

    #[test]
    fn test_bug_report_format() {
        let ptx = r#"
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%rd0], %f0;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        let report = result.format_report();

        assert!(report.contains("PTX BUG HUNTING REPORT"));
        assert!(report.contains("P0 CRITICAL BUGS:"));
        assert!(report.contains("SUMMARY"));
    }

    #[test]
    fn test_kernel_name_extraction() {
        let ptx = r#"
.visible .entry gemm_tiled() {
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert_eq!(result.kernel_name, Some("gemm_tiled".to_string()));
    }

    #[test]
    fn test_count_by_severity() {
        let report = PtxBugReport {
            kernel_name: Some("test".to_string()),
            bugs: vec![
                PtxBug {
                    class: PtxBugClass::MissingBarrierSync,
                    line: 1,
                    instruction: "test".to_string(),
                    message: "test".to_string(),
                    fix: None,
                },
                PtxBug {
                    class: PtxBugClass::RegisterSpills,
                    line: 2,
                    instruction: "test".to_string(),
                    message: "test".to_string(),
                    fix: None,
                },
            ],
            lines_analyzed: 10,
            strict_mode: true,
        };

        assert_eq!(report.count_by_severity(BugSeverity::Critical), 1);
        assert_eq!(report.count_by_severity(BugSeverity::High), 1);
        assert_eq!(report.count_by_severity(BugSeverity::Medium), 0);
    }

    /// F101: Detect `st.shared [%rd0]`
    #[test]
    fn f101_detect_shared_u64_addressing() {
        let ptx = "st.shared.f32 [%rd5], %f0;";
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::SharedMemU64Addressing));
    }

    /// F102: Detect missing `bar.sync`
    #[test]
    fn f102_detect_missing_barrier() {
        let ptx = r#"
.visible .entry test() {
    .shared .b8 smem[1024];
    st.shared.f32 [%r0], %f0;
    ld.shared.f32 %f1, [%r1];
    ret;
}
"#;
        let result = PtxBugAnalyzer::strict().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::MissingBarrierSync));
    }

    /// F103: Detect `bra loop_end` in loop
    #[test]
    fn f103_detect_loop_branch_end() {
        let ptx = r#"
.entry test() {
test_loop:
    bra test_loop_end;
test_loop_end:
    ret;
}
"#;
        let result = PtxBugAnalyzer::strict().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::LoopBranchToEnd));
    }

    /// F104: Valid PTX passes
    #[test]
    fn f104_valid_ptx_passes() {
        let ptx = r#"
.version 8.0
.target sm_70
.visible .entry valid() {
    .reg .f32 %f<4>;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.is_valid());
    }

    /// F106: Missing `.entry` detected
    #[test]
    fn f106_missing_entry_detected() {
        let ptx = ".version 8.0
.target sm_70
.reg .f32 %f<4>;";
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::MissingEntryPoint));
    }

    /// Test RedundantMoves detection - consecutive mov chain
    #[test]
    fn test_redundant_moves_chain() {
        let ptx = r#"
.visible .entry test() {
    mov.u32 %r1, %r0;
    mov.u32 %r2, %r1;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::RedundantMoves));
    }

    /// Test RedundantMoves - no chain (valid)
    #[test]
    fn test_redundant_moves_no_chain() {
        let ptx = r#"
.visible .entry test() {
    mov.u32 %r1, %r0;
    add.u32 %r2, %r1, 1;
    mov.u32 %r3, %r2;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::RedundantMoves));
    }

    /// Test UnoptimizedMemoryPattern - multiple single loads
    #[test]
    fn test_unoptimized_memory_single_loads() {
        let ptx = r#"
.visible .entry test() {
    ld.global.f32 %f0, [%rd0];
    ld.global.f32 %f1, [%rd1];
    ld.global.f32 %f2, [%rd2];
    ld.global.f32 %f3, [%rd3];
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::UnoptimizedMemoryPattern));
    }

    /// Test UnoptimizedMemoryPattern - vector loads (valid)
    #[test]
    fn test_unoptimized_memory_vector_loads() {
        let ptx = r#"
.visible .entry test() {
    ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd0];
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::UnoptimizedMemoryPattern));
    }

    /// Test UnoptimizedMemoryPattern - few single loads (acceptable)
    #[test]
    fn test_unoptimized_memory_few_loads() {
        let ptx = r#"
.visible .entry test() {
    ld.global.f32 %f0, [%rd0];
    ld.global.f32 %f1, [%rd1];
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        // Only 2 single loads - below threshold of 4, should not flag
        assert!(!result.has_bug(&PtxBugClass::UnoptimizedMemoryPattern));
    }

    /// Test suspicious stride detection in strict mode
    #[test]
    fn test_unoptimized_memory_suspicious_stride() {
        let ptx = r#"
.visible .entry test() {
    mul.wide.u32 %rd0, %r0, 17;
    ld.global.f32 %f0, [%rd0];
    ret;
}
"#;
        let result = PtxBugAnalyzer::strict().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::UnoptimizedMemoryPattern));
    }

    /// Test normal strides are not flagged
    #[test]
    fn test_unoptimized_memory_normal_stride() {
        let ptx = r#"
.visible .entry test() {
    mul.wide.u32 %rd0, %r0, 4;
    ld.global.f32 %f0, [%rd0];
    ret;
}
"#;
        let result = PtxBugAnalyzer::strict().analyze(ptx);
        // Stride 4 is normal for f32
        assert!(!result.has_bug(&PtxBugClass::UnoptimizedMemoryPattern));
    }

    /// Test high register pressure detection
    #[test]
    fn test_high_register_pressure() {
        let ptx = r#"
.visible .entry test() {
    .reg .b32 %r<64>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<32>;
    .reg .pred %p<4>;
    ret;
}
"#;
        // 64 + 16 + 32 + 4 = 116 registers > 64 threshold
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::HighRegisterPressure));
    }

    /// Test acceptable register pressure (no bug)
    #[test]
    fn test_normal_register_pressure() {
        let ptx = r#"
.visible .entry test() {
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .pred %p<4>;
    ret;
}
"#;
        // 16 + 8 + 8 + 4 = 36 registers < 64 threshold
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::HighRegisterPressure));
    }

    /// Test predicate overflow detection
    #[test]
    fn test_predicate_overflow() {
        let ptx = r#"
.visible .entry test() {
    .reg .pred %p<12>;
    .reg .b32 %r<4>;
    ret;
}
"#;
        // 12 predicates > 8 limit
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::PredicateOverflow));
    }

    /// Test acceptable predicate count (no bug)
    #[test]
    fn test_normal_predicate_count() {
        let ptx = r#"
.visible .entry test() {
    .reg .pred %p<8>;
    .reg .b32 %r<4>;
    ret;
}
"#;
        // 8 predicates = limit, should not flag
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::PredicateOverflow));
    }

    /// Test placeholder code detection - "omitted"
    #[test]
    fn test_placeholder_code_omitted() {
        let ptx = r#"
.visible .entry test() {
    // ... loading logic omitted for brevity
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::PlaceholderCode));
    }

    /// Test placeholder code detection - "simplified"
    #[test]
    fn test_placeholder_code_simplified() {
        let ptx = r#"
.visible .entry test() {
    // Simplified: only first element
    st.global.f32 [%rd0], %f0;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::PlaceholderCode));
    }

    /// Test placeholder code detection - "placeholder"
    #[test]
    fn test_placeholder_code_explicit() {
        let ptx = r#"
.visible .entry test() {
    // This is placeholder code for now
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::PlaceholderCode));
    }

    /// Test no placeholder code (clean kernel)
    #[test]
    fn test_no_placeholder_code() {
        let ptx = r#"
.visible .entry test() {
    // Load input
    ld.global.f32 %f0, [%rd0];
    // Compute result
    mul.f32 %f1, %f0, %f0;
    // Store output
    st.global.f32 [%rd1], %f1;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::PlaceholderCode));
    }

    /// Test new bug class severities
    #[test]
    fn test_new_bug_severities() {
        assert_eq!(
            PtxBugClass::HighRegisterPressure.severity(),
            BugSeverity::High
        );
        assert_eq!(PtxBugClass::PredicateOverflow.severity(), BugSeverity::High);
        assert_eq!(PtxBugClass::PlaceholderCode.severity(), BugSeverity::High);
    }

    /// Test new bug class codes
    #[test]
    fn test_new_bug_codes() {
        assert_eq!(
            PtxBugClass::HighRegisterPressure.code(),
            "HIGH_REG_PRESSURE"
        );
        assert_eq!(PtxBugClass::PredicateOverflow.code(), "PRED_OVERFLOW");
        assert_eq!(PtxBugClass::PlaceholderCode.code(), "PLACEHOLDER_CODE");
    }

    // ========================================================================
    // WHITELIST TESTS
    // ========================================================================

    /// Test whitelist suppresses matching bug
    #[test]
    fn test_whitelist_suppresses_bug() {
        let ptx = r#"
.visible .entry q4k_gemm_ggml() {
    .reg .b32 %r<64>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<32>;
    ret;
}
"#;
        // Without whitelist: should flag high register pressure
        let result_no_whitelist = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result_no_whitelist.has_bug(&PtxBugClass::HighRegisterPressure));

        // With quantized whitelist: q4k* should be suppressed
        let result_with_whitelist = PtxBugAnalyzer::with_quantized_whitelist().analyze(ptx);
        assert!(!result_with_whitelist.has_bug(&PtxBugClass::HighRegisterPressure));
    }

    /// Test whitelist with exact kernel name match
    #[test]
    fn test_whitelist_exact_match() {
        let ptx = r#"
.visible .entry special_kernel() {
    .reg .b32 %r<64>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<32>;
    ret;
}
"#;
        // With exact match whitelist
        let analyzer = PtxBugAnalyzer::new().with_whitelist(
            "special_kernel",
            PtxBugClass::HighRegisterPressure,
            "Expected high regs",
        );
        let result = analyzer.analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::HighRegisterPressure));
    }

    /// Test whitelist doesn't suppress non-matching kernels
    #[test]
    fn test_whitelist_no_match() {
        let ptx = r#"
.visible .entry other_kernel() {
    .reg .b32 %r<64>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<32>;
    ret;
}
"#;
        // q4k* whitelist should not match "other_kernel"
        let result = PtxBugAnalyzer::with_quantized_whitelist().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::HighRegisterPressure));
    }

    /// Test performance whitelist covers Tensor Core kernels
    #[test]
    fn test_performance_whitelist_tensor_core() {
        let ptx = r#"
.visible .entry gemm_tensor_core() {
    .reg .b32 %r<64>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<64>;
    .reg .pred %p<12>;
    ret;
}
"#;
        // Without whitelist: should flag both issues
        let result_no_whitelist = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result_no_whitelist.has_bug(&PtxBugClass::HighRegisterPressure));
        assert!(result_no_whitelist.has_bug(&PtxBugClass::PredicateOverflow));

        // With performance whitelist: both should be suppressed
        let result_with_whitelist = PtxBugAnalyzer::with_performance_whitelist().analyze(ptx);
        assert!(!result_with_whitelist.has_bug(&PtxBugClass::HighRegisterPressure));
        assert!(!result_with_whitelist.has_bug(&PtxBugClass::PredicateOverflow));
    }

    /// Test performance whitelist covers attention kernels
    #[test]
    fn test_performance_whitelist_attention() {
        let ptx = r#"
.visible .entry flash_attention_tensor_core() {
    .reg .b32 %r<64>;
    .reg .b64 %rd<32>;
    .reg .f32 %f<48>;
    ret;
}
"#;
        // With performance whitelist: register pressure should be suppressed
        let result = PtxBugAnalyzer::with_performance_whitelist().analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::HighRegisterPressure));
    }

    // ========================================================================
    // EMPTY LOOP BODY TESTS
    // ========================================================================

    /// Test empty loop body detection
    #[test]
    fn test_empty_loop_body_detected() {
        let ptx = r#"
.visible .entry test() {
empty_loop:
    // Just comments here
    bra empty_loop;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::EmptyLoopBody));
    }

    /// Test valid loop body not flagged
    #[test]
    fn test_valid_loop_body_not_flagged() {
        let ptx = r#"
.visible .entry test() {
    .reg .f32 %f<4>;
    .reg .u32 %r<4>;
compute_loop:
    add.f32 %f0, %f0, %f1;
    add.u32 %r0, %r0, 1;
    setp.lt.u32 %p0, %r0, %r1;
    @%p0 bra compute_loop;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::EmptyLoopBody));
    }

    /// Test loop with only conditional branch not flagged
    #[test]
    fn test_loop_with_exit_condition_not_flagged() {
        let ptx = r#"
.visible .entry test() {
    .reg .u32 %r<4>;
    .reg .pred %p<2>;
check_loop:
    setp.lt.u32 %p0, %r0, %r1;
    @%p0 bra check_loop;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        // Has setp which is computation
        assert!(!result.has_bug(&PtxBugClass::EmptyLoopBody));
    }

    // ========================================================================
    // MISSING BOUNDS CHECK TESTS
    // ========================================================================

    /// Test missing bounds check detection
    #[test]
    fn test_missing_bounds_check() {
        let ptx = r#"
.visible .entry test() {
    .reg .u64 %rd<4>;
    .reg .f32 %f<4>;
    mov.u32 %r0, %tid.x;
    ld.global.f32 %f0, [%rd0];
    st.global.f32 [%rd1], %f0;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::MissingBoundsCheck));
    }

    /// Test proper bounds check not flagged
    #[test]
    fn test_proper_bounds_check_not_flagged() {
        let ptx = r#"
.visible .entry test() {
    .reg .u64 %rd<4>;
    .reg .f32 %f<4>;
    .reg .u32 %r<4>;
    .reg .pred %p<2>;
    mov.u32 %r0, %tid.x;
    setp.lt.u32 %p0, %r0, %r1;
    @%p0 bra do_work;
    bra done;
do_work:
    ld.global.f32 %f0, [%rd0];
    st.global.f32 [%rd1], %f0;
done:
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::MissingBoundsCheck));
    }

    /// Test kernel without global memory not flagged
    #[test]
    fn test_no_global_mem_no_bounds_check_needed() {
        let ptx = r#"
.visible .entry test() {
    .reg .u32 %r<4>;
    mov.u32 %r0, %tid.x;
    add.u32 %r1, %r0, 1;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        // No global memory, so no bounds check needed
        assert!(!result.has_bug(&PtxBugClass::MissingBoundsCheck));
    }

    // ========================================================================
    // DEAD CODE TESTS
    // ========================================================================

    /// Test dead code after ret
    #[test]
    fn test_dead_code_after_ret() {
        let ptx = r#"
.visible .entry test() {
    .reg .f32 %f<4>;
    add.f32 %f0, %f1, %f2;
    ret;
    mul.f32 %f3, %f0, %f1;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::DeadCode));
    }

    /// Test dead code after unconditional branch
    #[test]
    fn test_dead_code_after_branch() {
        let ptx = r#"
.visible .entry test() {
    .reg .f32 %f<4>;
    bra skip;
    add.f32 %f0, %f1, %f2;
skip:
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::DeadCode));
    }

    /// Test reachable code not flagged (label after branch)
    #[test]
    fn test_reachable_code_not_flagged() {
        let ptx = r#"
.visible .entry test() {
    .reg .f32 %f<4>;
    .reg .pred %p<2>;
    @%p0 bra skip;
    add.f32 %f0, %f1, %f2;
skip:
    mul.f32 %f3, %f0, %f1;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        // Conditional branch, code after is reachable
        assert!(!result.has_bug(&PtxBugClass::DeadCode));
    }

    /// Test code after label is reachable
    #[test]
    fn test_code_after_label_reachable() {
        let ptx = r#"
.visible .entry test() {
    .reg .f32 %f<4>;
    bra middle;
middle:
    add.f32 %f0, %f1, %f2;
    ret;
}
"#;
        let result = PtxBugAnalyzer::new().analyze(ptx);
        // The add after middle: label is reachable via the branch
        assert!(!result.has_bug(&PtxBugClass::DeadCode));
    }

    // ========================================================================
    // NEW BUG CLASS SEVERITY/CODE TESTS
    // ========================================================================

    /// Test new extended bug class severities
    #[test]
    fn test_extended_bug_severities() {
        assert_eq!(PtxBugClass::EmptyLoopBody.severity(), BugSeverity::High);
        assert_eq!(
            PtxBugClass::MissingBoundsCheck.severity(),
            BugSeverity::High
        );
        assert_eq!(PtxBugClass::DeadCode.severity(), BugSeverity::Medium);
    }

    /// Test new extended bug class codes
    #[test]
    fn test_extended_bug_codes() {
        assert_eq!(PtxBugClass::EmptyLoopBody.code(), "EMPTY_LOOP");
        assert_eq!(PtxBugClass::MissingBoundsCheck.code(), "NO_BOUNDS_CHECK");
        assert_eq!(PtxBugClass::DeadCode.code(), "DEAD_CODE");
    }

    // ========================================================================
    // PARITY-114: EARLY EXIT BEFORE BARRIER TESTS
    // ========================================================================

    /// PARITY-114: Detect conditional early exit before barrier
    #[test]
    fn test_parity114_conditional_exit_before_barrier() {
        let ptx = r#"
.visible .entry kernel() {
    mov.u32 %r0, %tid.x;
    setp.lt.u32 %p0, %r0, 32;

loop_start:
    @!%p0 bra exit;
    ld.shared.f32 %f0, [%r0];
    bar.sync 0;
    st.shared.f32 [%r0], %f0;
    bra loop_start;

loop_start_end:
done:
    ret;
}
"#;
        let result = PtxBugAnalyzer::strict().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::EarlyExitBeforeBarrier));
        // Verify it's P0 Critical
        assert_eq!(
            PtxBugClass::EarlyExitBeforeBarrier.severity(),
            BugSeverity::Critical
        );
    }

    /// PARITY-114: Detect unconditional early exit before barrier
    #[test]
    fn test_parity114_unconditional_exit_before_barrier() {
        let ptx = r#"
.visible .entry kernel() {
loop_start:
    bra exit;
    bar.sync 0;
    bra loop_start;

loop_start_end:
done:
    ret;
}
"#;
        let result = PtxBugAnalyzer::strict().analyze(ptx);
        assert!(result.has_bug(&PtxBugClass::EarlyExitBeforeBarrier));
    }

    /// PARITY-114: Safe kernel with barrier before any possible exit
    #[test]
    fn test_parity114_safe_barrier_first() {
        let ptx = r#"
.visible .entry kernel() {
    mov.u32 %r0, %tid.x;
    setp.lt.u32 %p0, %r0, 32;

loop_start:
    ld.shared.f32 %f0, [%r0];
    bar.sync 0;
    st.shared.f32 [%r0], %f0;
    bra loop_start;

loop_start_end:
    @!%p0 bra exit;
    st.global.f32 [%r1], %f0;
exit:
    ret;
}
"#;
        let result = PtxBugAnalyzer::strict().analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::EarlyExitBeforeBarrier));
    }

    /// PARITY-114: Exit after loop end is safe
    #[test]
    fn test_parity114_exit_after_loop_is_safe() {
        let ptx = r#"
.visible .entry kernel() {
k_tile_loop:
    bar.sync 0;
    ld.shared.f32 %f0, [%r0];
    bra k_tile_loop;

k_tile_end:
    @!%p0 bra exit;
    st.global.f32 [%r1], %f0;
done:
    ret;
}
"#;
        let result = PtxBugAnalyzer::strict().analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::EarlyExitBeforeBarrier));
    }

    /// PARITY-114: Non-strict mode does not flag barrier issues
    #[test]
    fn test_parity114_non_strict_mode() {
        let ptx = r#"
.visible .entry kernel() {
loop_start:
    @!%p0 bra exit;
    bar.sync 0;
    bra loop_start;

loop_start_end:
done:
    ret;
}
"#;
        // Non-strict mode should NOT flag this
        let result = PtxBugAnalyzer::new().analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::EarlyExitBeforeBarrier));

        // Strict mode SHOULD flag this
        let strict_result = PtxBugAnalyzer::strict().analyze(ptx);
        assert!(strict_result.has_bug(&PtxBugClass::EarlyExitBeforeBarrier));
    }

    /// PARITY-114: Bug class properties
    #[test]
    fn test_parity114_bug_class_properties() {
        assert_eq!(
            PtxBugClass::EarlyExitBeforeBarrier.code(),
            "EARLY_EXIT_BARRIER"
        );
        assert_eq!(
            PtxBugClass::EarlyExitBeforeBarrier.severity(),
            BugSeverity::Critical
        );
    }

    /// PARITY-114: kv_loop pattern (attention kernels) - safe after fix
    #[test]
    fn test_parity114_attention_kv_loop_safe() {
        let ptx = r#"
.visible .entry flash_attention() {
kv_loop:
    bar.sync 0;
    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f16.f16.f32;
    bra kv_loop;

kv_loop_end:
    @!%p_valid bra exit;
    st.global.f32 [%out], %f0;
done:
    ret;
}
"#;
        let result = PtxBugAnalyzer::strict().analyze(ptx);
        assert!(!result.has_bug(&PtxBugClass::EarlyExitBeforeBarrier));
    }
}
