//! PTX Bug Detection - Static Analysis for GPU Kernel Bugs
//!
//! Inspired by bashrs parser bug hunting methodology that found 25 bugs.
//! PTX is "scary" - invisible bugs cause silent correctness failures.
//!
//! Bug Classes (from probar `gpu_pixels`):
//! - P0 Critical: `SharedMemU64Addressing`, `LoopBranchToEnd`, `MissingBarrierSync`
//! - P1 High: `NonInPlaceLoopAccumulator`, `RegisterSpills`
//! - P2 Medium: `RedundantMoves`, `UnoptimizedMemoryPattern`
//! - False Positive: `InvalidSyntaxAccepted`, `MissingEntryPoint`

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

// Import barrier safety analyzer from trueno-gpu for PARITY-114 detection
use trueno_gpu::ptx::optimize::barrier_safety;

/// PTX bug severity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BugSeverity {
    /// P0: Correctness bugs - silent wrong results
    Critical,
    /// P1: Major performance bugs - 10-100x slowdown
    High,
    /// P2: Minor performance bugs - 2-10x slowdown
    Medium,
    /// False positive detection - error handling issues
    FalsePositive,
}

impl fmt::Display for BugSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "P0-CRITICAL"),
            Self::High => write!(f, "P1-HIGH"),
            Self::Medium => write!(f, "P2-MEDIUM"),
            Self::FalsePositive => write!(f, "FALSE-POSITIVE"),
        }
    }
}

/// PTX bug classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PtxBugClass {
    /// P0: Shared memory accessed with 64-bit register (should be 32-bit)
    SharedMemU64Addressing,
    /// P0: Loop branches to END label instead of START
    LoopBranchToEnd,
    /// P0: Missing barrier sync between shared memory write and read
    MissingBarrierSync,
    /// P0: Early thread exit before barrier in loop - causes CUDA error 700 (PARITY-114)
    EarlyExitBeforeBarrier,
    /// P1: Accumulator not updated in-place in loop
    NonInPlaceLoopAccumulator,
    /// P1: Register spills to local memory
    RegisterSpills,
    /// P1: High register pressure (>64 regs) reduces occupancy
    HighRegisterPressure,
    /// P1: Predicate register overflow (>8 predicates)
    PredicateOverflow,
    /// P1: Placeholder/incomplete code detected
    PlaceholderCode,
    /// P1: Empty loop body (no actual computation)
    EmptyLoopBody,
    /// P1: Missing thread bounds check
    MissingBoundsCheck,
    /// P2: Redundant register moves
    RedundantMoves,
    /// P2: Unoptimized memory access pattern
    UnoptimizedMemoryPattern,
    /// P2: Dead code after unconditional branch or ret
    DeadCode,
    /// False Positive: Invalid PTX syntax accepted
    InvalidSyntaxAccepted,
    /// False Positive: Missing kernel entry point
    MissingEntryPoint,
}

impl PtxBugClass {
    /// Get the severity of this bug class
    #[must_use]
    pub fn severity(&self) -> BugSeverity {
        match self {
            Self::SharedMemU64Addressing
            | Self::LoopBranchToEnd
            | Self::MissingBarrierSync
            | Self::EarlyExitBeforeBarrier => BugSeverity::Critical,

            Self::NonInPlaceLoopAccumulator
            | Self::RegisterSpills
            | Self::HighRegisterPressure
            | Self::PredicateOverflow
            | Self::PlaceholderCode
            | Self::EmptyLoopBody
            | Self::MissingBoundsCheck => BugSeverity::High,

            Self::RedundantMoves | Self::UnoptimizedMemoryPattern | Self::DeadCode => BugSeverity::Medium,

            Self::InvalidSyntaxAccepted | Self::MissingEntryPoint => BugSeverity::FalsePositive,
        }
    }

    /// Get a short code for this bug class
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::SharedMemU64Addressing => "SHARED_U64",
            Self::LoopBranchToEnd => "LOOP_BRANCH_END",
            Self::MissingBarrierSync => "MISSING_BARRIER",
            Self::EarlyExitBeforeBarrier => "EARLY_EXIT_BARRIER",
            Self::NonInPlaceLoopAccumulator => "NON_INPLACE_ACCUM",
            Self::RegisterSpills => "REG_SPILLS",
            Self::HighRegisterPressure => "HIGH_REG_PRESSURE",
            Self::PredicateOverflow => "PRED_OVERFLOW",
            Self::PlaceholderCode => "PLACEHOLDER_CODE",
            Self::EmptyLoopBody => "EMPTY_LOOP",
            Self::MissingBoundsCheck => "NO_BOUNDS_CHECK",
            Self::RedundantMoves => "REDUNDANT_MOV",
            Self::UnoptimizedMemoryPattern => "UNOPT_MEM",
            Self::DeadCode => "DEAD_CODE",
            Self::InvalidSyntaxAccepted => "INVALID_SYNTAX",
            Self::MissingEntryPoint => "NO_ENTRY",
        }
    }
}

impl fmt::Display for PtxBugClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

/// A detected PTX bug with location and fix suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtxBug {
    /// Bug classification
    pub class: PtxBugClass,
    /// Line number (1-indexed, 0 if unknown)
    pub line: usize,
    /// The offending PTX instruction
    pub instruction: String,
    /// Human-readable explanation
    pub message: String,
    /// Suggested fix
    pub fix: Option<String>,
}

impl PtxBug {
    /// Get the severity of this bug
    #[must_use]
    pub fn severity(&self) -> BugSeverity {
        self.class.severity()
    }
}

/// Result of PTX bug hunting analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtxBugReport {
    /// Kernel name if detected
    pub kernel_name: Option<String>,
    /// List of detected bugs
    pub bugs: Vec<PtxBug>,
    /// Total lines analyzed
    pub lines_analyzed: usize,
    /// Analysis mode (strict or normal)
    pub strict_mode: bool,
}

impl PtxBugReport {
    /// Check if PTX passed all validations (no critical bugs)
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.bugs.iter().any(|b| b.severity() == BugSeverity::Critical)
    }

    /// Check if there are any bugs at all
    #[must_use]
    pub fn has_bugs(&self) -> bool {
        !self.bugs.is_empty()
    }

    /// Count bugs by severity
    #[must_use]
    pub fn count_by_severity(&self, severity: BugSeverity) -> usize {
        self.bugs.iter().filter(|b| b.severity() == severity).count()
    }

    /// Check for specific bug class
    #[must_use]
    pub fn has_bug(&self, class: &PtxBugClass) -> bool {
        self.bugs.iter().any(|b| &b.class == class)
    }

    /// Get bugs by class
    #[must_use]
    pub fn bugs_of_class(&self, class: &PtxBugClass) -> Vec<&PtxBug> {
        self.bugs.iter().filter(|b| &b.class == class).collect()
    }

    /// Format as bug report (bashrs style)
    #[must_use]
    pub fn format_report(&self) -> String {
        let mut output = String::new();

        output.push_str("╔══════════════════════════════════════════════════════════════════════════════╗\n");
        output.push_str("║                         PTX BUG HUNTING REPORT                                ║\n");
        output.push_str("╚══════════════════════════════════════════════════════════════════════════════╝\n\n");

        if let Some(name) = &self.kernel_name {
            output.push_str(&format!("Kernel: {}\n", name));
        }
        output.push_str(&format!("PTX Lines Analyzed: {}\n\n", self.lines_analyzed));

        let critical = self.count_by_severity(BugSeverity::Critical);
        let high = self.count_by_severity(BugSeverity::High);
        let medium = self.count_by_severity(BugSeverity::Medium);
        let false_pos = self.count_by_severity(BugSeverity::FalsePositive);

        // P0 Critical
        output.push_str(&format!("P0 CRITICAL BUGS: {}\n", critical));
        if critical > 0 {
            output.push_str("──────────────────\n");
            for (i, bug) in self.bugs.iter().filter(|b| b.severity() == BugSeverity::Critical).enumerate() {
                output.push_str(&format!("  BUG-{:03}: {}\n", i + 1, bug.class));
                if bug.line > 0 {
                    output.push_str(&format!("    Line {}: {}\n", bug.line, bug.instruction));
                }
                output.push_str(&format!("    Impact: {}\n", bug.message));
                if let Some(fix) = &bug.fix {
                    output.push_str(&format!("    Fix: {}\n", fix));
                }
                output.push('\n');
            }
        }

        // P1 High
        output.push_str(&format!("\nP1 HIGH BUGS: {}\n", high));
        if high > 0 {
            output.push_str("─────────────────\n");
            for bug in self.bugs.iter().filter(|b| b.severity() == BugSeverity::High) {
                output.push_str(&format!("  {}: {}\n", bug.class, bug.message));
            }
        }

        // P2 Medium
        output.push_str(&format!("\nP2 MEDIUM BUGS: {}\n", medium));
        if medium > 0 {
            output.push_str("─────────────────\n");
            for bug in self.bugs.iter().filter(|b| b.severity() == BugSeverity::Medium) {
                output.push_str(&format!("  {}: {}\n", bug.class, bug.message));
            }
        }

        // False positives
        output.push_str(&format!("\nFALSE POSITIVES DETECTED: {}\n", false_pos));

        // Summary
        output.push_str("\nSUMMARY\n═══════\n");
        output.push_str(&format!("  Total Bugs: {}\n", self.bugs.len()));
        output.push_str(&format!("  P0 Critical: {}", critical));
        if critical > 0 {
            output.push_str(" ← BLOCKS RELEASE");
        }
        output.push('\n');
        output.push_str(&format!("  P1 High: {}\n", high));
        output.push_str(&format!("  P2 Medium: {}\n", medium));

        output
    }
}

/// Whitelist entry for suppressing known acceptable warnings
#[derive(Debug, Clone)]
pub struct WhitelistEntry {
    /// Kernel name pattern (supports prefix matching with *)
    pub kernel_pattern: String,
    /// Bug class to suppress
    pub bug_class: PtxBugClass,
    /// Reason for whitelisting
    pub reason: String,
}

/// PTX bug hunting analyzer (inspired by probar `gpu_pixels`)
#[derive(Debug, Default, Clone)]
pub struct PtxBugAnalyzer {
    /// Enable strict mode (more warnings, catches PARITY-114 pattern)
    pub strict: bool,
    /// Whitelist for suppressing known acceptable warnings
    pub whitelist: Vec<WhitelistEntry>,
}

impl PtxBugAnalyzer {
    /// Create analyzer with default (non-strict) mode
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create analyzer with strict mode enabled
    #[must_use]
    pub fn strict() -> Self {
        Self { strict: true, whitelist: Vec::new() }
    }

    /// Add a whitelist entry to suppress warnings
    #[must_use]
    pub fn with_whitelist(mut self, kernel_pattern: &str, bug_class: PtxBugClass, reason: &str) -> Self {
        self.whitelist.push(WhitelistEntry {
            kernel_pattern: kernel_pattern.to_string(),
            bug_class,
            reason: reason.to_string(),
        });
        self
    }

    /// Create analyzer with default whitelist for quantized kernels
    #[must_use]
    pub fn with_quantized_whitelist() -> Self {
        Self::new()
            .with_whitelist("q4k*", PtxBugClass::HighRegisterPressure,
                "Quantized kernels require high registers for dequantization")
            .with_whitelist("q5k*", PtxBugClass::HighRegisterPressure,
                "Quantized kernels require high registers for dequantization")
            .with_whitelist("q6k*", PtxBugClass::HighRegisterPressure,
                "Quantized kernels require high registers for dequantization")
            .with_whitelist("q8k*", PtxBugClass::HighRegisterPressure,
                "Quantized kernels require high registers for dequantization")
    }

    /// Create analyzer with comprehensive whitelist for all high-performance kernels
    ///
    /// This whitelist covers expected register pressure and predicate usage in:
    /// - Tensor Core kernels (WMMA requires many registers for matrix fragments)
    /// - Attention kernels (`FlashAttention` needs registers for tiling state)
    /// - Quantized kernels (dequantization requires intermediate values)
    ///
    /// These are documented performance tradeoffs, not bugs.
    #[must_use]
    pub fn with_performance_whitelist() -> Self {
        Self::new()
            // Tensor Core / WMMA kernels - high register usage is expected
            // WMMA m16n16k16 requires 8 registers per fragment × 3 fragments = 24+ registers
            // Plus accumulator, addresses, loop counters, etc.
            .with_whitelist("gemm_tensor_core*", PtxBugClass::HighRegisterPressure,
                "Tensor Core WMMA requires many registers for matrix fragments")
            .with_whitelist("gemm_tensor_core*", PtxBugClass::PredicateOverflow,
                "Tensor Core kernels use predicates for bounds checking and masking")
            .with_whitelist("gemm_wmma*", PtxBugClass::HighRegisterPressure,
                "WMMA FP16 requires registers for A/B/C/D matrix fragments")
            .with_whitelist("gemm_wmma*", PtxBugClass::PredicateOverflow,
                "WMMA kernels use predicates for tile boundary handling")
            // Attention kernels - FlashAttention tiling requires state
            .with_whitelist("flash_attention*", PtxBugClass::HighRegisterPressure,
                "FlashAttention tiling requires registers for Q/K/V/O tiles and softmax state")
            .with_whitelist("attention*", PtxBugClass::HighRegisterPressure,
                "Attention kernels require registers for Q/K/V tiles and reduction")
            // Quantized kernels - dequantization math
            .with_whitelist("q4k*", PtxBugClass::HighRegisterPressure,
                "Q4_K dequantization requires registers for scale/min extraction")
            .with_whitelist("q5k*", PtxBugClass::HighRegisterPressure,
                "Q5_K dequantization requires registers for 5-bit value reconstruction")
            .with_whitelist("q6k*", PtxBugClass::HighRegisterPressure,
                "Q6_K dequantization requires registers for 6-bit value reconstruction")
            .with_whitelist("q8k*", PtxBugClass::HighRegisterPressure,
                "Q8_K dequantization requires registers for scale application")
    }

    /// Check if a bug should be suppressed by whitelist
    fn is_whitelisted(&self, kernel_name: Option<&String>, bug_class: &PtxBugClass) -> bool {
        let Some(kernel) = kernel_name else {
            return false;
        };

        for entry in &self.whitelist {
            if &entry.bug_class != bug_class {
                continue;
            }
            // Pattern matching: "q4k*" matches "q4k_gemm_ggml"
            if entry.kernel_pattern.ends_with('*') {
                let prefix = &entry.kernel_pattern[..entry.kernel_pattern.len() - 1];
                if kernel.starts_with(prefix) {
                    return true;
                }
            } else if &entry.kernel_pattern == kernel {
                return true;
            }
        }
        false
    }

    /// Analyze PTX for bugs
    #[must_use]
    pub fn analyze(&self, ptx: &str) -> PtxBugReport {
        let mut bugs = Vec::new();
        let lines: Vec<&str> = ptx.lines().collect();

        // Extract kernel name
        let kernel_name = self.extract_kernel_name(ptx);

        // Run all bug detectors
        bugs.extend(self.detect_shared_mem_u64(ptx, &lines));
        bugs.extend(self.detect_loop_branch_to_end(ptx, &lines));
        bugs.extend(self.detect_missing_barrier_sync(ptx, &lines));
        bugs.extend(self.detect_early_exit_before_barrier(ptx));
        bugs.extend(self.detect_register_spills(ptx, &lines));
        bugs.extend(self.detect_missing_entry_point(ptx, &lines));
        bugs.extend(self.detect_redundant_moves(ptx, &lines));
        bugs.extend(self.detect_unoptimized_memory(ptx, &lines));
        bugs.extend(self.detect_high_register_pressure(ptx, &lines));
        bugs.extend(self.detect_predicate_overflow(ptx, &lines));
        bugs.extend(self.detect_placeholder_code(ptx, &lines));
        // New extended detectors
        bugs.extend(self.detect_empty_loop_body(ptx, &lines));
        bugs.extend(self.detect_missing_bounds_check(ptx, &lines));
        bugs.extend(self.detect_dead_code(ptx, &lines));

        // Filter out whitelisted bugs
        bugs.retain(|bug| !self.is_whitelisted(kernel_name.as_ref(), &bug.class));

        PtxBugReport {
            kernel_name,
            bugs,
            lines_analyzed: lines.len(),
            strict_mode: self.strict,
        }
    }

    /// Extract kernel name from PTX
    fn extract_kernel_name(&self, ptx: &str) -> Option<String> {
        let entry_pattern = Regex::new(r"\.(?:visible\s+)?\.entry\s+(\w+)").unwrap();
        entry_pattern
            .captures(ptx)
            .map(|c| c.get(1).unwrap().as_str().to_string())
    }

    /// Detect shared memory accessed with 64-bit register
    fn detect_shared_mem_u64(&self, _ptx: &str, lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();
        // Pattern: st.shared.* [%rd*] or ld.shared.* [%rd*]
        let pattern = Regex::new(r"(?:st|ld)\.shared\.[^\[]+\[%rd\d+").unwrap();

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if pattern.is_match(trimmed) {
                bugs.push(PtxBug {
                    class: PtxBugClass::SharedMemU64Addressing,
                    line: line_num + 1,
                    instruction: trimmed.to_string(),
                    message: "Shared memory accessed with 64-bit register. Use 32-bit addressing.".to_string(),
                    fix: Some("Replace %rd* with %r* for shared memory addressing".to_string()),
                });
            }
        }

        bugs
    }

    /// Detect loop branches to END instead of START
    fn detect_loop_branch_to_end(&self, _ptx: &str, lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        if !self.strict {
            return bugs;
        }

        // Collect loop labels
        let loop_label = Regex::new(r"^(\w+(?:_loop|loop_)\w*):").unwrap();
        let branch_instr = Regex::new(r"^\s*bra\s+(\w+);").unwrap();

        let mut loop_start_labels: HashSet<String> = HashSet::new();
        let mut loop_end_labels: HashSet<String> = HashSet::new();

        // First pass: collect labels
        for line in lines {
            let trimmed = line.trim();
            if let Some(caps) = loop_label.captures(trimmed) {
                let label = caps.get(1).unwrap().as_str();
                if label.contains("_start") || label.ends_with("_loop") || label.starts_with("loop_") {
                    loop_start_labels.insert(label.to_string());
                } else if label.contains("_end") {
                    loop_end_labels.insert(label.to_string());
                }
            }
        }

        // Second pass: detect unconditional branches to end labels
        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if let Some(caps) = branch_instr.captures(trimmed) {
                let target = caps.get(1).unwrap().as_str();
                // Unconditional branch (not @%p prefixed) to _end label
                if loop_end_labels.contains(target) && !trimmed.starts_with('@') {
                    bugs.push(PtxBug {
                        class: PtxBugClass::LoopBranchToEnd,
                        line: line_num + 1,
                        instruction: trimmed.to_string(),
                        message: format!("Unconditional branch to loop end '{}'. Should branch to start?", target),
                        fix: Some(format!("Change target from {} to corresponding _start label", target)),
                    });
                }
            }
        }

        bugs
    }

    /// Detect missing barrier sync between shared memory operations (PARITY-114)
    fn detect_missing_barrier_sync(&self, ptx: &str, lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        if !self.strict {
            return bugs;
        }

        // Check if shared memory is ACTUALLY used (st.shared or ld.shared operations)
        // Note: We don't flag just `.shared` declarations - only actual load/store operations
        // This prevents false positives for kernels that declare shared memory but use warp shuffles
        let has_st_shared = ptx.contains("st.shared");
        let has_ld_shared = ptx.contains("ld.shared");
        let uses_shared_ops = has_st_shared || has_ld_shared;
        let has_barrier = ptx.contains("bar.sync");

        if uses_shared_ops && !has_barrier {
            bugs.push(PtxBug {
                class: PtxBugClass::MissingBarrierSync,
                line: 0,
                instruction: String::new(),
                message: "Shared memory used but no bar.sync found. Race condition possible.".to_string(),
                fix: Some("Add bar.sync 0; between st.shared and ld.shared operations".to_string()),
            });
        }

        // More precise detection: find st.shared followed by ld.shared without bar.sync
        let st_shared = Regex::new(r"st\.shared").unwrap();
        let ld_shared = Regex::new(r"ld\.shared").unwrap();
        let bar_sync = Regex::new(r"bar\.sync").unwrap();

        let mut last_st_shared_line: Option<usize> = None;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if st_shared.is_match(trimmed) {
                last_st_shared_line = Some(line_num);
            } else if bar_sync.is_match(trimmed) {
                last_st_shared_line = None; // Reset after barrier
            } else if ld_shared.is_match(trimmed) {
                if let Some(st_line) = last_st_shared_line {
                    // ld.shared after st.shared without bar.sync
                    bugs.push(PtxBug {
                        class: PtxBugClass::MissingBarrierSync,
                        line: line_num + 1,
                        instruction: format!("st.shared at line {}, ld.shared at line {}", st_line + 1, line_num + 1),
                        message: "ld.shared follows st.shared without barrier synchronization".to_string(),
                        fix: Some(format!("Add bar.sync 0; between lines {} and {}", st_line + 1, line_num + 1)),
                    });
                }
            }
        }

        bugs
    }

    /// Detect early thread exit before barrier in loop (PARITY-114)
    ///
    /// This is the root cause of CUDA error 700 (illegal instruction) when
    /// some threads in a warp exit early via `bra exit` before reaching a
    /// `bar.sync` instruction. The remaining threads hang waiting at the barrier.
    ///
    /// Uses trueno-gpu's `barrier_safety` analyzer for consistent detection.
    fn detect_early_exit_before_barrier(&self, ptx: &str) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Only check in strict mode (matches MissingBarrierSync behavior)
        if !self.strict {
            return bugs;
        }

        // Use the authoritative barrier_safety analyzer from trueno-gpu
        let result = barrier_safety::analyze(ptx);

        for violation in result.violations {
            let kind = match violation.kind {
                barrier_safety::ViolationKind::EarlyExitBeforeBarrier => {
                    "Unconditional early exit before barrier"
                }
                barrier_safety::ViolationKind::ConditionalExitBeforeBarrier => {
                    "Conditional early exit may cause thread divergence at barrier"
                }
                barrier_safety::ViolationKind::MissingBarrierAfterSharedAccess => {
                    continue; // Already handled by detect_missing_barrier_sync
                }
            };

            bugs.push(PtxBug {
                class: PtxBugClass::EarlyExitBeforeBarrier,
                line: violation.line,
                instruction: violation.instruction,
                message: format!(
                    "PARITY-114: {} - causes CUDA error 700. {}",
                    kind, violation.context
                ),
                fix: Some(
                    "Move bounds check AFTER loop body. Use predicated loads (store 0 first) \
                     so all threads participate in bar.sync regardless of bounds."
                        .to_string(),
                ),
            });
        }

        bugs
    }

    /// Detect register spills to local memory
    fn detect_register_spills(&self, ptx: &str, lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Spills manifest as .local memory usage
        let local_pattern = Regex::new(r"\.local").unwrap();
        let spill_count = local_pattern.find_iter(ptx).count();

        if spill_count > 0 {
            // Find the first .local declaration
            let mut first_local_line = 0;
            for (line_num, line) in lines.iter().enumerate() {
                if local_pattern.is_match(line) {
                    first_local_line = line_num + 1;
                    break;
                }
            }

            bugs.push(PtxBug {
                class: PtxBugClass::RegisterSpills,
                line: first_local_line,
                instruction: format!("{} .local declarations", spill_count),
                message: format!("{} potential register spills detected. High latency local memory access.", spill_count),
                fix: Some("Reduce live variables or increase register allocation".to_string()),
            });
        }

        bugs
    }

    /// Detect missing kernel entry point
    fn detect_missing_entry_point(&self, ptx: &str, _lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        let entry_pattern = Regex::new(r"\.entry\s+\w+").unwrap();
        let has_entry = entry_pattern.is_match(ptx);

        // Only flag if PTX has some content but no entry point
        if !ptx.trim().is_empty() && !has_entry {
            bugs.push(PtxBug {
                class: PtxBugClass::MissingEntryPoint,
                line: 0,
                instruction: String::new(),
                message: "No kernel entry point (.entry) found".to_string(),
                fix: Some("Add .entry <kernel_name>(...) declaration".to_string()),
            });
        }

        bugs
    }

    /// Detect redundant register moves (P2)
    /// Pattern: mov %rx, %ry followed by mov %rz, %rx (could use %ry directly)
    fn detect_redundant_moves(&self, _ptx: &str, lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Look for mov chains: mov %a, %b; mov %c, %a; → should be mov %c, %b
        let mov_pattern = Regex::new(r"^\s*mov\.\w+\s+(%\w+),\s*(%\w+)").unwrap();

        let mut last_mov: Option<(usize, String, String)> = None; // (line, dest, src)

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if let Some(caps) = mov_pattern.captures(trimmed) {
                let dest = caps.get(1).unwrap().as_str().to_string();
                let src = caps.get(2).unwrap().as_str().to_string();

                // Check if src matches previous dest (redundant chain)
                if let Some((prev_line, prev_dest, _prev_src)) = &last_mov {
                    if &src == prev_dest {
                        bugs.push(PtxBug {
                            class: PtxBugClass::RedundantMoves,
                            line: line_num + 1,
                            instruction: format!("mov chain at lines {} and {}", prev_line + 1, line_num + 1),
                            message: format!("Redundant move: {} copied to {} then to another register", prev_dest, dest),
                            fix: Some("Combine mov chain into single mov".to_string()),
                        });
                    }
                }

                last_mov = Some((line_num, dest, src));
            } else {
                // Reset on non-mov instruction
                last_mov = None;
            }
        }

        bugs
    }

    /// Detect unoptimized memory access patterns (P2)
    /// Patterns: strided access, non-vectorized loads, etc.
    fn detect_unoptimized_memory(&self, ptx: &str, lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Pattern 1: Multiple single-element loads that could be vectorized
        // ld.global.f32 x4 in sequence could be ld.global.v4.f32
        let single_load = Regex::new(r"ld\.global\.f32").unwrap();
        let vector_load = Regex::new(r"ld\.global\.v[24]\.f32").unwrap();

        let single_loads = single_load.find_iter(ptx).count();
        let vector_loads = vector_load.find_iter(ptx).count();

        // If there are many single loads but no vector loads, suggest vectorization
        if single_loads >= 4 && vector_loads == 0 {
            bugs.push(PtxBug {
                class: PtxBugClass::UnoptimizedMemoryPattern,
                line: 0,
                instruction: format!("{} single f32 loads, 0 vector loads", single_loads),
                message: "Multiple single-element loads could potentially be vectorized".to_string(),
                fix: Some("Consider using ld.global.v2.f32 or ld.global.v4.f32 for consecutive addresses".to_string()),
            });
        }

        // Pattern 2: Look for non-coalesced access hints
        // Strided access: base + i * stride where stride != sizeof(element)
        let strided_pattern = Regex::new(r"mul\.wide\.[us]32\s+%\w+,\s*%\w+,\s*(\d+)").unwrap();
        let mut suspicious_strides = Vec::new();

        // Known quantization block strides (not bugs - legitimate data layouts)
        // Q4_K: 144 bytes, Q5_K: 176 bytes, Q6_K: 210 bytes, Q8_K: 256 bytes
        let quantization_strides: HashSet<u32> = [144, 176, 210, 256, 512].into_iter().collect();

        for (line_num, line) in lines.iter().enumerate() {
            if let Some(caps) = strided_pattern.captures(line) {
                if let Ok(stride) = caps.get(1).unwrap().as_str().parse::<u32>() {
                    // Suspicious if stride is not standard and not a known quantization block size
                    // Standard: 4 (f32), 8 (f64), 2 (f16), 1 (byte), or multiple of 4
                    if stride > 8 && stride % 4 != 0 && !quantization_strides.contains(&stride) {
                        suspicious_strides.push((line_num + 1, stride));
                    }
                }
            }
        }

        if !suspicious_strides.is_empty() && self.strict {
            bugs.push(PtxBug {
                class: PtxBugClass::UnoptimizedMemoryPattern,
                line: suspicious_strides[0].0,
                instruction: format!("Stride {} detected", suspicious_strides[0].1),
                message: "Non-standard stride may indicate strided (non-coalesced) access".to_string(),
                fix: Some("Consider restructuring data layout for coalesced access".to_string()),
            });
        }

        bugs
    }

    /// Detect high register pressure (P1)
    /// >64 registers per thread reduces occupancy and may cause spills
    fn detect_high_register_pressure(&self, ptx: &str, _lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Count register declarations: .reg .type %name<count>
        let reg_pattern = Regex::new(r"\.reg\s+\.\w+\s+%\w+<(\d+)>").unwrap();
        let total_regs: usize = reg_pattern
            .captures_iter(ptx)
            .filter_map(|c| c.get(1).and_then(|m| m.as_str().parse::<usize>().ok()))
            .sum();

        // Threshold: >64 registers is problematic for occupancy
        // SM_89 has 65536 regs/SM, 64 regs/thread allows 32 warps (100% occupancy)
        if total_regs > 64 {
            let occupancy = 65536 / (total_regs * 32);
            let occupancy_pct = (occupancy as f32 / 32.0 * 100.0).min(100.0);
            bugs.push(PtxBug {
                class: PtxBugClass::HighRegisterPressure,
                line: 0,
                instruction: format!("{} register banks declared", total_regs),
                message: format!(
                    "High register pressure: {} registers limits occupancy to {:.0}%",
                    total_regs, occupancy_pct
                ),
                fix: Some("Reduce live variables or split into multiple kernels".to_string()),
            });
        }

        bugs
    }

    /// Detect predicate register overflow (P1)
    /// PTX has 8 predicate registers (p0-p7), exceeding this causes spills
    fn detect_predicate_overflow(&self, ptx: &str, _lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Pattern: .reg .pred %p<count>
        let pred_pattern = Regex::new(r"\.reg\s+\.pred\s+%p<(\d+)>").unwrap();
        if let Some(caps) = pred_pattern.captures(ptx) {
            if let Ok(pred_count) = caps.get(1).unwrap().as_str().parse::<usize>() {
                if pred_count > 8 {
                    bugs.push(PtxBug {
                        class: PtxBugClass::PredicateOverflow,
                        line: 0,
                        instruction: format!(".reg .pred %p<{}>", pred_count),
                        message: format!(
                            "Predicate overflow: {} predicates declared (max 8 hardware registers)",
                            pred_count
                        ),
                        fix: Some("Reduce predicate usage by combining conditions or using branches".to_string()),
                    });
                }
            }
        }

        bugs
    }

    /// Detect placeholder/incomplete code (P1)
    /// Comments like "omitted", "simplified", "placeholder" indicate incomplete kernels
    fn detect_placeholder_code(&self, _ptx: &str, lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Patterns indicating incomplete code
        let placeholder_patterns = [
            "omitted",
            "simplified",
            "placeholder",
            "todo",
            "fixme",
            "not implemented",
            "for now",
            "for brevity",
        ];

        for (line_num, line) in lines.iter().enumerate() {
            let lower = line.to_lowercase();
            // Only check comments
            if lower.contains("//") {
                for pattern in &placeholder_patterns {
                    if lower.contains(pattern) {
                        bugs.push(PtxBug {
                            class: PtxBugClass::PlaceholderCode,
                            line: line_num + 1,
                            instruction: line.trim().to_string(),
                            message: format!("Placeholder code detected: contains '{}'", pattern),
                            fix: Some("Implement complete kernel or use trueno-gpu generation".to_string()),
                        });
                        break; // Only report once per line
                    }
                }
            }
        }

        bugs
    }

    /// Detect empty loop body (P1)
    /// A loop that branches back without doing any computation
    fn detect_empty_loop_body(&self, _ptx: &str, lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Find loop patterns: label followed by branch back to same label
        let label_pattern = Regex::new(r"^(\w+):$").unwrap();
        let branch_pattern = Regex::new(r"^\s*(?:@%\w+\s+)?bra\s+(\w+);").unwrap();

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();

            // Check if this is a loop label
            if let Some(label_caps) = label_pattern.captures(line) {
                let label = label_caps.get(1).unwrap().as_str();

                // Look for the loop body and back-edge
                let mut j = i + 1;
                let mut has_computation = false;
                let mut loop_end = None;

                while j < lines.len() && j < i + 20 {
                    // Limit search to 20 lines
                    let inner = lines[j].trim();

                    // Skip comments and empty lines
                    if inner.is_empty() || inner.starts_with("//") {
                        j += 1;
                        continue;
                    }

                    // Check if this line does computation
                    let compute_ops = [
                        "add.", "sub.", "mul.", "div.", "fma.", "mad.", "ld.", "st.", "cvt.", "mov.",
                        "setp.", "and.", "or.", "xor.", "shl.", "shr.", "min.", "max.", "abs.",
                        "neg.", "rcp.", "sqrt.", "rsqrt.", "sin.", "cos.", "ex2.", "lg2.",
                    ];
                    for op in &compute_ops {
                        if inner.contains(op) {
                            has_computation = true;
                            break;
                        }
                    }

                    // Check for branch back to loop label
                    if let Some(br_caps) = branch_pattern.captures(inner) {
                        let target = br_caps.get(1).unwrap().as_str();
                        if target == label {
                            loop_end = Some(j);
                            break;
                        }
                    }

                    // Check for end label (loop_end, _end suffix)
                    if inner.ends_with(':') && (inner.contains("_end") || inner.contains("END")) {
                        break;
                    }

                    j += 1;
                }

                // If we found a loop back-edge but no computation
                if loop_end.is_some() && !has_computation {
                    bugs.push(PtxBug {
                        class: PtxBugClass::EmptyLoopBody,
                        line: i + 1,
                        instruction: format!("Loop '{}' at line {}", label, i + 1),
                        message: "Loop body contains no computation - may be placeholder code".to_string(),
                        fix: Some("Implement loop body or remove empty loop".to_string()),
                    });
                }
            }
            i += 1;
        }

        bugs
    }

    /// Detect missing thread bounds check (P1)
    /// Kernels should check tid < size before accessing memory
    fn detect_missing_bounds_check(&self, ptx: &str, _lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        // Only check if there are memory operations
        let has_global_mem = ptx.contains("ld.global") || ptx.contains("st.global");
        if !has_global_mem {
            return bugs;
        }

        // Check for common bounds check patterns
        let has_tid = ptx.contains("%tid.") || ptx.contains("%ntid.");
        let has_setp_lt = ptx.contains("setp.lt") || ptx.contains("setp.ge");
        let has_predicated_branch = Regex::new(r"@%p\d+\s+bra").unwrap().is_match(ptx);

        // If kernel uses tid and global memory but has no bounds check
        if has_tid && !has_setp_lt && !has_predicated_branch {
            bugs.push(PtxBug {
                class: PtxBugClass::MissingBoundsCheck,
                line: 0,
                instruction: "No setp.lt/ge with predicated branch found".to_string(),
                message: "Kernel accesses global memory but may lack thread bounds checking".to_string(),
                fix: Some("Add: setp.lt.u32 %p0, %tid, %size; @%p0 bra do_work;".to_string()),
            });
        }

        bugs
    }

    /// Detect dead code (P2)
    /// Code after unconditional ret or bra that can never execute
    fn detect_dead_code(&self, _ptx: &str, lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        let unconditional_ret = Regex::new(r"^\s*ret;").unwrap();
        let unconditional_bra = Regex::new(r"^\s*bra\s+\w+;").unwrap(); // No @%p prefix
        let label_pattern = Regex::new(r"^\w+:$").unwrap();

        let mut after_unconditional = false;
        let mut unconditional_line = 0;

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            // Skip empty lines and comments
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }

            // Check if this is a label (reachable code)
            if label_pattern.is_match(trimmed) {
                after_unconditional = false;
                continue;
            }

            // Check if this is closing brace
            if trimmed == "}" {
                after_unconditional = false;
                continue;
            }

            // Check if we're after an unconditional jump
            if after_unconditional {
                bugs.push(PtxBug {
                    class: PtxBugClass::DeadCode,
                    line: line_num + 1,
                    instruction: trimmed.to_string(),
                    message: format!(
                        "Dead code: unreachable after unconditional jump at line {}",
                        unconditional_line + 1
                    ),
                    fix: Some("Remove unreachable code or add label".to_string()),
                });
                // Only report once per dead code block
                after_unconditional = false;
                continue;
            }

            // Check for unconditional ret
            if unconditional_ret.is_match(trimmed) {
                after_unconditional = true;
                unconditional_line = line_num;
            }

            // Check for unconditional bra (not predicated)
            if unconditional_bra.is_match(trimmed) && !trimmed.starts_with('@') {
                after_unconditional = true;
                unconditional_line = line_num;
            }
        }

        bugs
    }
}

// ============================================================================
// PTX COVERAGE TRACKER (F107 - Probar-style coverage tracking)
// ============================================================================

/// PTX feature coverage tracking (inspired by bashrs `gui_coverage!` macro)
///
/// Tracks which PTX features are covered by test cases to ensure comprehensive testing.
#[derive(Debug, Clone)]
pub struct PtxCoverageTracker {
    features: Vec<PtxFeature>,
}

/// A PTX feature that can be tracked for coverage
#[derive(Debug, Clone)]
pub struct PtxFeature {
    /// Feature name
    pub name: String,
    /// Whether this feature has been covered
    pub covered: bool,
    /// How many times this feature was seen
    pub hit_count: usize,
}

/// Coverage report summary
#[derive(Debug, Clone)]
pub struct PtxCoverageReport {
    /// Total features tracked
    pub total_features: usize,
    /// Features that were covered
    pub covered_features: usize,
    /// Coverage percentage (0.0 - 1.0)
    pub coverage: f64,
    /// Details per feature
    pub features: Vec<PtxFeature>,
}

impl PtxCoverageTracker {
    /// Create a new coverage tracker builder
    #[must_use]
    pub fn builder() -> PtxCoverageTrackerBuilder {
        PtxCoverageTrackerBuilder {
            features: Vec::new(),
        }
    }

    /// Analyze PTX code and update coverage
    pub fn analyze(&mut self, ptx: &str) {
        for feature in &mut self.features {
            let covered = match feature.name.as_str() {
                "barrier_sync" => ptx.contains("bar.sync"),
                "shared_memory" => ptx.contains(".shared") || ptx.contains("st.shared") || ptx.contains("ld.shared"),
                "global_memory" => ptx.contains("ld.global") || ptx.contains("st.global"),
                "register_allocation" => ptx.contains(".reg"),
                "loop_patterns" => ptx.contains("bra") && (ptx.contains("_loop") || ptx.contains("loop_")),
                "control_flow" => ptx.contains("@%p") || ptx.contains("bra") || ptx.contains("setp"),
                "local_memory" => ptx.contains(".local"),
                "entry_point" => ptx.contains(".entry"),
                "predicates" => ptx.contains(".pred") || ptx.contains("@%p"),
                "fma_ops" => ptx.contains("fma.") || ptx.contains("mad."),
                _ => false,
            };

            if covered {
                feature.covered = true;
                feature.hit_count += 1;
            }
        }
    }

    /// Generate coverage report
    #[must_use]
    pub fn generate_report(&self) -> PtxCoverageReport {
        let total = self.features.len();
        let covered = self.features.iter().filter(|f| f.covered).count();
        let coverage = if total > 0 {
            covered as f64 / total as f64
        } else {
            1.0
        };

        PtxCoverageReport {
            total_features: total,
            covered_features: covered,
            coverage,
            features: self.features.clone(),
        }
    }
}

impl Default for PtxCoverageTracker {
    fn default() -> Self {
        PtxCoverageTrackerBuilder::new()
            .feature("barrier_sync")
            .feature("shared_memory")
            .feature("global_memory")
            .feature("register_allocation")
            .feature("loop_patterns")
            .feature("control_flow")
            .build()
    }
}

/// Builder for `PtxCoverageTracker`
#[derive(Debug)]
pub struct PtxCoverageTrackerBuilder {
    features: Vec<PtxFeature>,
}

impl PtxCoverageTrackerBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self { features: Vec::new() }
    }

    /// Add a feature to track
    #[must_use]
    pub fn feature(mut self, name: &str) -> Self {
        self.features.push(PtxFeature {
            name: name.to_string(),
            covered: false,
            hit_count: 0,
        });
        self
    }

    /// Build the coverage tracker
    #[must_use]
    pub fn build(self) -> PtxCoverageTracker {
        PtxCoverageTracker {
            features: self.features,
        }
    }
}

impl Default for PtxCoverageTrackerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

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
        assert!(missing_barrier_bugs.iter().all(|b| !b.message.contains("ld.shared follows st.shared")));
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
        assert_eq!(PtxBugClass::MissingBarrierSync.severity(), BugSeverity::Critical);
        assert_eq!(PtxBugClass::SharedMemU64Addressing.severity(), BugSeverity::Critical);
        assert_eq!(PtxBugClass::RegisterSpills.severity(), BugSeverity::High);
        assert_eq!(PtxBugClass::MissingEntryPoint.severity(), BugSeverity::FalsePositive);
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
        let ptx = ".version 8.0\n.target sm_70\n.reg .f32 %f<4>;";
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
        assert_eq!(PtxBugClass::HighRegisterPressure.severity(), BugSeverity::High);
        assert_eq!(PtxBugClass::PredicateOverflow.severity(), BugSeverity::High);
        assert_eq!(PtxBugClass::PlaceholderCode.severity(), BugSeverity::High);
    }

    /// Test new bug class codes
    #[test]
    fn test_new_bug_codes() {
        assert_eq!(PtxBugClass::HighRegisterPressure.code(), "HIGH_REG_PRESSURE");
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
        let analyzer = PtxBugAnalyzer::new()
            .with_whitelist("special_kernel", PtxBugClass::HighRegisterPressure, "Expected high regs");
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
        assert_eq!(PtxBugClass::MissingBoundsCheck.severity(), BugSeverity::High);
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
        assert_eq!(PtxBugClass::EarlyExitBeforeBarrier.severity(), BugSeverity::Critical);
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
        assert_eq!(PtxBugClass::EarlyExitBeforeBarrier.code(), "EARLY_EXIT_BARRIER");
        assert_eq!(PtxBugClass::EarlyExitBeforeBarrier.severity(), BugSeverity::Critical);
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
