//! PTX Parser and Analyzer
//!
//! Implements the Analyzer trait for NVIDIA PTX assembly.

use crate::analyzer::{
    AnalysisReport, Analyzer, MemoryPattern, MudaType, MudaWarning, RegisterUsage, RooflineMetric,
};
use crate::error::Result;
use regex::Regex;

/// PTX code analyzer
pub struct PtxAnalyzer {
    /// Warn if register count exceeds this threshold
    pub register_warning_threshold: u32,
    /// Warn if coalescing ratio falls below this threshold
    pub coalescing_warning_threshold: f32,
}

impl Default for PtxAnalyzer {
    fn default() -> Self {
        Self {
            register_warning_threshold: 128,
            coalescing_warning_threshold: 0.8,
        }
    }
}

impl PtxAnalyzer {
    /// Create a new PTX analyzer with default thresholds
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse register declarations from PTX
    fn parse_registers(&self, ptx: &str) -> RegisterUsage {
        let mut usage = RegisterUsage::default();

        // Match patterns like: .reg .f32 %f<24>;
        let reg_pattern = Regex::new(r"\.reg\s+\.(\w+)\s+%\w+<(\d+)>").unwrap();

        for cap in reg_pattern.captures_iter(ptx) {
            let reg_type = &cap[1];
            let count: u32 = cap[2].parse().unwrap_or(0);

            match reg_type {
                "f32" => usage.f32_regs += count,
                "f64" => usage.f64_regs += count,
                "b32" | "u32" | "s32" => usage.b32_regs += count,
                "b64" | "u64" | "s64" => usage.b64_regs += count,
                "pred" => usage.pred_regs += count,
                _ => {}
            }
        }

        usage
    }

    /// Parse memory operations from PTX
    fn parse_memory_ops(&self, ptx: &str) -> MemoryPattern {
        let mut pattern = MemoryPattern::default();

        // Count global loads
        let global_load = Regex::new(r"ld\.global").unwrap();
        pattern.global_loads = global_load.find_iter(ptx).count() as u32;

        // Count global stores
        let global_store = Regex::new(r"st\.global").unwrap();
        pattern.global_stores = global_store.find_iter(ptx).count() as u32;

        // Count shared loads
        let shared_load = Regex::new(r"ld\.shared").unwrap();
        pattern.shared_loads = shared_load.find_iter(ptx).count() as u32;

        // Count shared stores
        let shared_store = Regex::new(r"st\.shared").unwrap();
        pattern.shared_stores = shared_store.find_iter(ptx).count() as u32;

        // Estimate coalescing based on access patterns
        // Coalesced access indicators:
        // 1. tid/ctaid references (thread and block IDs - used for index computation)
        // 2. mad.lo with tid (computing linear index from thread/block IDs)
        // 3. mul.wide with small constant (stride-1 access)
        // 4. shfl instructions (warp shuffle - implicit coalescing)
        // Note: Include both x and y dimensions since 2D kernels use both
        let tid_pattern = Regex::new(r"%tid\.[xy]|%ntid\.[xy]|%ctaid\.[xy]").unwrap();
        let tid_refs = tid_pattern.find_iter(ptx).count();

        // mad.lo often computes coalesced indices: mad.lo %r, %ctaid, %ntid, %tid
        let mad_pattern = Regex::new(r"mad\.lo").unwrap();
        let mad_refs = mad_pattern.find_iter(ptx).count();

        // mul.lo also used for index computation
        let mul_lo_pattern = Regex::new(r"mul\.lo").unwrap();
        let mul_lo_refs = mul_lo_pattern.find_iter(ptx).count();

        // mul.wide with small constants indicates stride-based access
        let stride_pattern = Regex::new(r"mul\.wide\.[us]32").unwrap();
        let stride_refs = stride_pattern.find_iter(ptx).count();

        // Warp shuffles indicate warp-level data sharing (inherently coalesced)
        let shfl_pattern = Regex::new(r"shfl\.(down|up|bfly|idx)").unwrap();
        let shfl_refs = shfl_pattern.find_iter(ptx).count();

        // rem/div operations often used for lane computation in coalesced patterns
        let lane_pattern = Regex::new(r"rem\.u32|div\.u32").unwrap();
        let lane_refs = lane_pattern.find_iter(ptx).count();

        let total_accesses = pattern.global_loads + pattern.global_stores;
        if total_accesses > 0 {
            // Improved heuristic: weight different indicators
            // Each indicator suggests thread-based indexing which implies coalescing potential
            let coalescing_score = tid_refs as f32
                + (mad_refs as f32 * 0.6)  // mad.lo strongly indicates index computation
                + (mul_lo_refs as f32 * 0.4) // mul.lo also used for indices
                + (stride_refs as f32 * 0.3) // stride patterns
                + (shfl_refs as f32 * 0.3)  // warp shuffles
                + (lane_refs as f32 * 0.2); // lane computation
            pattern.coalesced_ratio = (coalescing_score / total_accesses as f32).min(1.0);
        } else {
            pattern.coalesced_ratio = 1.0;
        }

        pattern
    }

    /// Count total instructions
    fn count_instructions(&self, ptx: &str) -> u32 {
        // Count lines that look like instructions (not directives or labels)
        let instruction_pattern = Regex::new(r"^\s+(add|sub|mul|div|mad|fma|ld|st|mov|setp|bra|ret|cvt|and|or|xor|shl|shr|min|max|abs|neg|sqrt|rsqrt|sin|cos|ex2|lg2|rcp|selp|set|bar)").unwrap();

        ptx.lines()
            .filter(|line| instruction_pattern.is_match(line))
            .count() as u32
    }

    /// Extract kernel name from PTX
    fn extract_kernel_name(&self, ptx: &str) -> String {
        let entry_pattern = Regex::new(r"\.entry\s+(\w+)").unwrap();
        entry_pattern
            .captures(ptx)
            .map(|c| c[1].to_string())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Detect spills (Muda of Transport)
    fn detect_spills(&self, ptx: &str) -> Option<MudaWarning> {
        // Spills manifest as .local memory usage
        let local_pattern = Regex::new(r"\.local").unwrap();
        let spill_count = local_pattern.find_iter(ptx).count();

        if spill_count > 0 {
            Some(MudaWarning {
                muda_type: MudaType::Transport,
                description: format!("{} potential register spills detected", spill_count),
                impact: "High latency local memory access".to_string(),
                line: None,
                suggestion: Some(
                    "Reduce live variables or increase register allocation".to_string(),
                ),
            })
        } else {
            None
        }
    }

    /// Detect uncoalesced access (Muda of Waiting)
    fn detect_uncoalesced(&self, memory: &MemoryPattern) -> Option<MudaWarning> {
        if memory.coalesced_ratio < self.coalescing_warning_threshold {
            Some(MudaWarning {
                muda_type: MudaType::Waiting,
                description: format!(
                    "Memory coalescing ratio {:.1}% below threshold {:.1}%",
                    memory.coalesced_ratio * 100.0,
                    self.coalescing_warning_threshold * 100.0
                ),
                impact: "Serialized memory transactions, reduced bandwidth".to_string(),
                line: None,
                suggestion: Some(
                    "Ensure adjacent threads access adjacent memory addresses".to_string(),
                ),
            })
        } else {
            None
        }
    }

    /// Detect excessive register usage
    fn detect_register_pressure(&self, registers: &RegisterUsage) -> Option<MudaWarning> {
        let total = registers.total();
        if total > self.register_warning_threshold {
            Some(MudaWarning {
                muda_type: MudaType::Overprocessing,
                description: format!(
                    "High register usage: {} registers (threshold: {})",
                    total, self.register_warning_threshold
                ),
                impact: "Reduced occupancy, fewer concurrent warps".to_string(),
                line: None,
                suggestion: Some(
                    "Consider loop tiling or reducing intermediate values".to_string(),
                ),
            })
        } else {
            None
        }
    }
}

impl Analyzer for PtxAnalyzer {
    fn target_name(&self) -> &str {
        "PTX"
    }

    fn analyze(&self, ptx: &str) -> Result<AnalysisReport> {
        let registers = self.parse_registers(ptx);
        let memory = self.parse_memory_ops(ptx);
        let instruction_count = self.count_instructions(ptx);
        let name = self.extract_kernel_name(ptx);
        let warnings = self.detect_muda(ptx);
        let estimated_occupancy = registers.estimated_occupancy();

        let mut report = AnalysisReport {
            name,
            target: self.target_name().to_string(),
            registers,
            memory,
            warnings,
            instruction_count,
            estimated_occupancy,
            ..Default::default()
        };

        report.roofline = self.estimate_roofline(&report);
        Ok(report)
    }

    fn detect_muda(&self, ptx: &str) -> Vec<MudaWarning> {
        let mut warnings = Vec::new();

        if let Some(w) = self.detect_spills(ptx) {
            warnings.push(w);
        }

        let memory = self.parse_memory_ops(ptx);
        if let Some(w) = self.detect_uncoalesced(&memory) {
            warnings.push(w);
        }

        let registers = self.parse_registers(ptx);
        if let Some(w) = self.detect_register_pressure(&registers) {
            warnings.push(w);
        }

        warnings
    }

    fn estimate_roofline(&self, analysis: &AnalysisReport) -> RooflineMetric {
        // Simplified roofline model
        // Arithmetic intensity = FLOPs / Bytes transferred
        let mem_ops = analysis.memory.global_loads + analysis.memory.global_stores;
        let bytes = mem_ops * 4; // Assume f32

        let flops = analysis.instruction_count; // Rough approximation

        let arithmetic_intensity = if bytes > 0 {
            flops as f32 / bytes as f32
        } else {
            0.0
        };

        // SM 7.0 theoretical peak: ~15 TFLOPS (varies by GPU)
        let theoretical_peak_gflops = 15000.0;

        // Memory bound if AI < ridge point (typically ~10 for modern GPUs)
        let memory_bound = arithmetic_intensity < 10.0;

        RooflineMetric {
            arithmetic_intensity,
            theoretical_peak_gflops,
            memory_bound,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_PTX: &str = r#"
.version 8.0
.target sm_70
.address_size 64

.entry vector_add(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_n
)
{
    .reg .f32 %f<24>;
    .reg .b32 %r<18>;
    .reg .b64 %rd<12>;
    .reg .pred %p<4>;

    ld.param.u64 %rd1, [param_a];
    ld.param.u64 %rd2, [param_b];
    ld.param.u64 %rd3, [param_c];
    ld.param.u32 %r1, [param_n];

    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %ctaid.x;
    mad.lo.s32 %r5, %r4, %r3, %r2;

    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra exit;

    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    add.u64 %rd7, %rd3, %rd4;

    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;

exit:
    ret;
}
"#;

    #[test]
    fn test_parse_registers() {
        let analyzer = PtxAnalyzer::new();
        let usage = analyzer.parse_registers(SAMPLE_PTX);

        assert_eq!(usage.f32_regs, 24);
        assert_eq!(usage.b32_regs, 18);
        assert_eq!(usage.b64_regs, 12);
        assert_eq!(usage.pred_regs, 4);
    }

    #[test]
    fn test_parse_memory_ops() {
        let analyzer = PtxAnalyzer::new();
        let memory = analyzer.parse_memory_ops(SAMPLE_PTX);

        assert_eq!(memory.global_loads, 2);
        assert_eq!(memory.global_stores, 1);
        assert_eq!(memory.shared_loads, 0);
        assert_eq!(memory.shared_stores, 0);
    }

    #[test]
    fn test_count_instructions() {
        let analyzer = PtxAnalyzer::new();
        let count = analyzer.count_instructions(SAMPLE_PTX);

        // Should count: ld.param (4) + mov (3) + mad + setp + mul + add (4) + ld.global (2) + st.global + ret
        assert!(count >= 15, "Expected >= 15 instructions, got {}", count);
    }

    #[test]
    fn test_extract_kernel_name() {
        let analyzer = PtxAnalyzer::new();
        let name = analyzer.extract_kernel_name(SAMPLE_PTX);
        assert_eq!(name, "vector_add");
    }

    #[test]
    fn test_extract_kernel_name_missing() {
        let analyzer = PtxAnalyzer::new();
        let name = analyzer.extract_kernel_name("// no kernel here");
        assert_eq!(name, "unknown");
    }

    #[test]
    fn test_analyze_full_report() {
        let analyzer = PtxAnalyzer::new();
        let report = analyzer.analyze(SAMPLE_PTX).unwrap();

        assert_eq!(report.name, "vector_add");
        assert_eq!(report.target, "PTX");
        assert_eq!(report.registers.f32_regs, 24);
        assert_eq!(report.memory.global_loads, 2);
        assert!(report.estimated_occupancy > 0.0);
    }

    #[test]
    fn test_detect_spills() {
        let analyzer = PtxAnalyzer::new();

        // No spills in sample PTX
        let warnings = analyzer.detect_muda(SAMPLE_PTX);
        let spill_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| matches!(w.muda_type, MudaType::Transport))
            .collect();
        assert!(spill_warnings.is_empty());

        // PTX with spills
        let ptx_with_spills = r#"
            .local .align 4 .b8 __local_depot[32];
            .reg .f32 %f<4>;
        "#;
        let spill_warning = analyzer.detect_spills(ptx_with_spills);
        assert!(spill_warning.is_some());
    }

    #[test]
    fn test_detect_high_register_pressure() {
        let analyzer = PtxAnalyzer::new();

        let high_reg_ptx = r#"
            .entry big_kernel()
            {
                .reg .f32 %f<200>;
                ret;
            }
        "#;

        let warnings = analyzer.detect_muda(high_reg_ptx);
        let reg_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| matches!(w.muda_type, MudaType::Overprocessing))
            .collect();
        assert!(!reg_warnings.is_empty());
    }

    #[test]
    fn test_json_output() {
        let analyzer = PtxAnalyzer::new();
        let report = analyzer.analyze(SAMPLE_PTX).unwrap();

        let json = serde_json::to_string_pretty(&report).unwrap();
        assert!(json.contains("vector_add"));
        assert!(json.contains("PTX"));
        assert!(json.contains("f32_regs"));
    }

    #[test]
    fn test_occupancy_estimation() {
        let analyzer = PtxAnalyzer::new();
        let report = analyzer.analyze(SAMPLE_PTX).unwrap();

        // 58 total registers should give good occupancy
        assert!(
            report.estimated_occupancy > 0.5,
            "Expected > 50% occupancy, got {}",
            report.estimated_occupancy
        );
    }

    #[test]
    fn test_roofline_estimation() {
        let analyzer = PtxAnalyzer::new();
        let report = analyzer.analyze(SAMPLE_PTX).unwrap();

        // Vector add is memory-bound
        assert!(
            report.roofline.memory_bound,
            "Vector add should be memory-bound"
        );
    }

    /// F030 (Memory): Identifies coalesced pattern (tid*4 detected)
    #[test]
    fn f030_memory_identifies_coalesced_pattern() {
        let analyzer = PtxAnalyzer::new();

        // PTX with tid-based indexing (coalesced pattern)
        let coalesced_ptx = r#"
            .entry coalesced_kernel()
            {
                .reg .f32 %f<4>;
                .reg .b32 %r<4>;
                .reg .b64 %rd<4>;
                // tid.x-based indexing indicates coalesced access
                mov.u32 %r0, %tid.x;
                mul.wide.u32 %rd0, %r0, 4;
                ld.global.f32 %f0, [%rd0];
                st.global.f32 [%rd0], %f0;
                ret;
            }
        "#;

        let memory = analyzer.parse_memory_ops(coalesced_ptx);

        // Should detect tid references indicating coalesced access
        assert!(
            memory.coalesced_ratio > 0.0,
            "Should detect tid-based coalesced pattern"
        );
        assert!(memory.global_loads > 0, "Should detect global loads");
        assert!(memory.global_stores > 0, "Should detect global stores");
    }

    /// F034: Warns on <80% coalescing ratio
    #[test]
    fn f034_warn_low_coalescing() {
        let analyzer = PtxAnalyzer::new();

        // PTX with many global loads but no tid references (uncoalesced pattern)
        let uncoalesced_ptx = r#"
            .entry uncoalesced_kernel()
            {
                .reg .f32 %f<4>;
                .reg .b64 %rd<4>;
                // Many loads without tid-based indexing
                ld.global.f32 %f0, [%rd0];
                ld.global.f32 %f1, [%rd1];
                ld.global.f32 %f2, [%rd2];
                ld.global.f32 %f3, [%rd3];
                st.global.f32 [%rd0], %f0;
                st.global.f32 [%rd1], %f1;
                st.global.f32 [%rd2], %f2;
                st.global.f32 [%rd3], %f3;
                ret;
            }
        "#;

        let warnings = analyzer.detect_muda(uncoalesced_ptx);
        let coalescing_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| matches!(w.muda_type, MudaType::Waiting))
            .filter(|w| w.description.contains("coalescing"))
            .collect();

        assert!(
            !coalescing_warnings.is_empty(),
            "Should warn on <80% coalescing ratio"
        );
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_register_count_non_negative(
            f32_count in 0u32..100,
            b32_count in 0u32..100,
        ) {
            let ptx = format!(
                ".entry test() {{ .reg .f32 %f<{}>; .reg .b32 %r<{}>; ret; }}",
                f32_count, b32_count
            );
            let analyzer = PtxAnalyzer::new();
            let usage = analyzer.parse_registers(&ptx);

            prop_assert_eq!(usage.f32_regs, f32_count);
            prop_assert_eq!(usage.b32_regs, b32_count);
            prop_assert!(usage.total() >= f32_count + b32_count);
        }

        #[test]
        fn prop_occupancy_bounded(regs in 1u32..256) {
            let usage = RegisterUsage {
                f32_regs: regs,
                ..Default::default()
            };
            let occ = usage.estimated_occupancy();
            prop_assert!(occ >= 0.0 && occ <= 1.0);
        }

        #[test]
        fn prop_memory_counts_non_negative(
            global_ld in 0usize..50,
            global_st in 0usize..50,
        ) {
            let mut ptx = String::from(".entry test() {\n");
            for _ in 0..global_ld {
                ptx.push_str("    ld.global.f32 %f1, [%rd1];\n");
            }
            for _ in 0..global_st {
                ptx.push_str("    st.global.f32 [%rd1], %f1;\n");
            }
            ptx.push_str("    ret;\n}");

            let analyzer = PtxAnalyzer::new();
            let memory = analyzer.parse_memory_ops(&ptx);

            prop_assert_eq!(memory.global_loads, global_ld as u32);
            prop_assert_eq!(memory.global_stores, global_st as u32);
        }
    }
}
