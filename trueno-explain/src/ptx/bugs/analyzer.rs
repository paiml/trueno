use regex::Regex;
use std::collections::HashSet;

use super::types::{PtxBug, PtxBugClass, PtxBugReport};
use trueno_gpu::ptx::optimize::barrier_safety;

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
        Self {
            strict: true,
            whitelist: Vec::new(),
        }
    }

    /// Add a whitelist entry to suppress warnings
    #[must_use]
    pub fn with_whitelist(
        mut self,
        kernel_pattern: &str,
        bug_class: PtxBugClass,
        reason: &str,
    ) -> Self {
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
            .with_whitelist(
                "q4k*",
                PtxBugClass::HighRegisterPressure,
                "Quantized kernels require high registers for dequantization",
            )
            .with_whitelist(
                "q5k*",
                PtxBugClass::HighRegisterPressure,
                "Quantized kernels require high registers for dequantization",
            )
            .with_whitelist(
                "q6k*",
                PtxBugClass::HighRegisterPressure,
                "Quantized kernels require high registers for dequantization",
            )
            .with_whitelist(
                "q8k*",
                PtxBugClass::HighRegisterPressure,
                "Quantized kernels require high registers for dequantization",
            )
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
            .with_whitelist(
                "gemm_tensor_core*",
                PtxBugClass::HighRegisterPressure,
                "Tensor Core WMMA requires many registers for matrix fragments",
            )
            .with_whitelist(
                "gemm_tensor_core*",
                PtxBugClass::PredicateOverflow,
                "Tensor Core kernels use predicates for bounds checking and masking",
            )
            .with_whitelist(
                "gemm_wmma*",
                PtxBugClass::HighRegisterPressure,
                "WMMA FP16 requires registers for A/B/C/D matrix fragments",
            )
            .with_whitelist(
                "gemm_wmma*",
                PtxBugClass::PredicateOverflow,
                "WMMA kernels use predicates for tile boundary handling",
            )
            // Attention kernels - FlashAttention tiling requires state
            .with_whitelist(
                "flash_attention*",
                PtxBugClass::HighRegisterPressure,
                "FlashAttention tiling requires registers for Q/K/V/O tiles and softmax state",
            )
            .with_whitelist(
                "attention*",
                PtxBugClass::HighRegisterPressure,
                "Attention kernels require registers for Q/K/V tiles and reduction",
            )
            // Quantized kernels - dequantization math
            .with_whitelist(
                "q4k*",
                PtxBugClass::HighRegisterPressure,
                "Q4_K dequantization requires registers for scale/min extraction",
            )
            .with_whitelist(
                "q5k*",
                PtxBugClass::HighRegisterPressure,
                "Q5_K dequantization requires registers for 5-bit value reconstruction",
            )
            .with_whitelist(
                "q6k*",
                PtxBugClass::HighRegisterPressure,
                "Q6_K dequantization requires registers for 6-bit value reconstruction",
            )
            .with_whitelist(
                "q8k*",
                PtxBugClass::HighRegisterPressure,
                "Q8_K dequantization requires registers for scale application",
            )
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

        // Execute all pattern detectors
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
        let entry_pattern = Regex::new(r"\.(?:visible\s+)?\.entry\s+(\w+)")
            .expect("invariant: regex pattern is valid");
        entry_pattern.captures(ptx).map(|c| {
            c.get(1)
                .expect("invariant: capture group 1 exists")
                .as_str()
                .to_string()
        })
    }

    /// Detect shared memory accessed with 64-bit register
    fn detect_shared_mem_u64(&self, _ptx: &str, lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();
        // Pattern: st.shared.* [%rd*] or ld.shared.* [%rd*]
        let pattern = Regex::new(r"(?:st|ld)\.shared\.[^\[]+\[%rd\d+")
            .expect("invariant: regex pattern is valid");

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if pattern.is_match(trimmed) {
                bugs.push(PtxBug {
                    class: PtxBugClass::SharedMemU64Addressing,
                    line: line_num + 1,
                    instruction: trimmed.to_string(),
                    message: "Shared memory accessed with 64-bit register. Use 32-bit addressing."
                        .to_string(),
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
        let loop_label =
            Regex::new(r"^(\w+(?:_loop|loop_)\w*):").expect("invariant: regex pattern is valid");
        let branch_instr =
            Regex::new(r"^\s*bra\s+(\w+);").expect("invariant: regex pattern is valid");

        let mut loop_start_labels: HashSet<String> = HashSet::new();
        let mut loop_end_labels: HashSet<String> = HashSet::new();

        // First pass: collect labels
        for line in lines {
            let trimmed = line.trim();
            if let Some(caps) = loop_label.captures(trimmed) {
                let label = caps
                    .get(1)
                    .expect("invariant: capture group exists")
                    .as_str();
                if label.contains("_start")
                    || label.ends_with("_loop")
                    || label.starts_with("loop_")
                {
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
                let target = caps
                    .get(1)
                    .expect("invariant: capture group exists")
                    .as_str();
                // Unconditional branch (not @%p prefixed) to _end label
                if loop_end_labels.contains(target) && !trimmed.starts_with('@') {
                    bugs.push(PtxBug {
                        class: PtxBugClass::LoopBranchToEnd,
                        line: line_num + 1,
                        instruction: trimmed.to_string(),
                        message: format!(
                            "Unconditional branch to loop end '{}'. Should branch to start?",
                            target
                        ),
                        fix: Some(format!(
                            "Change target from {} to corresponding _start label",
                            target
                        )),
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
                message: "Shared memory used but no bar.sync found. Race condition possible."
                    .to_string(),
                fix: Some("Add bar.sync 0; between st.shared and ld.shared operations".to_string()),
            });
        }

        // More precise detection: find st.shared followed by ld.shared without bar.sync
        let st_shared = Regex::new(r"st\.shared").expect("invariant: regex pattern is valid");
        let ld_shared = Regex::new(r"ld\.shared").expect("invariant: regex pattern is valid");
        let bar_sync = Regex::new(r"bar\.sync").expect("invariant: regex pattern is valid");

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
                        instruction: format!(
                            "st.shared at line {}, ld.shared at line {}",
                            st_line + 1,
                            line_num + 1
                        ),
                        message: "ld.shared follows st.shared without barrier synchronization"
                            .to_string(),
                        fix: Some(format!(
                            "Add bar.sync 0; between lines {} and {}",
                            st_line + 1,
                            line_num + 1
                        )),
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
        let local_pattern = Regex::new(r"\.local").expect("invariant: regex pattern is valid");
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
                message: format!(
                    "{} potential register spills detected. High latency local memory access.",
                    spill_count
                ),
                fix: Some("Reduce live variables or increase register allocation".to_string()),
            });
        }

        bugs
    }

    /// Detect missing kernel entry point
    fn detect_missing_entry_point(&self, ptx: &str, _lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        let entry_pattern =
            Regex::new(r"\.entry\s+\w+").expect("invariant: regex pattern is valid");
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
        let mov_pattern = Regex::new(r"^\s*mov\.\w+\s+(%\w+),\s*(%\w+)")
            .expect("invariant: regex pattern is valid");

        let mut last_mov: Option<(usize, String, String)> = None; // (line, dest, src)

        for (line_num, line) in lines.iter().enumerate() {
            let trimmed = line.trim();

            if let Some(caps) = mov_pattern.captures(trimmed) {
                let dest = caps
                    .get(1)
                    .expect("invariant: capture group exists")
                    .as_str()
                    .to_string();
                let src = caps
                    .get(2)
                    .expect("invariant: capture group exists")
                    .as_str()
                    .to_string();

                // Check if src matches previous dest (redundant chain)
                if let Some((prev_line, prev_dest, _prev_src)) = &last_mov {
                    if &src == prev_dest {
                        bugs.push(PtxBug {
                            class: PtxBugClass::RedundantMoves,
                            line: line_num + 1,
                            instruction: format!(
                                "mov chain at lines {} and {}",
                                prev_line + 1,
                                line_num + 1
                            ),
                            message: format!(
                                "Redundant move: {} copied to {} then to another register",
                                prev_dest, dest
                            ),
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
        let single_load =
            Regex::new(r"ld\.global\.f32").expect("invariant: regex pattern is valid");
        let vector_load =
            Regex::new(r"ld\.global\.v[24]\.f32").expect("invariant: regex pattern is valid");

        let single_loads = single_load.find_iter(ptx).count();
        let vector_loads = vector_load.find_iter(ptx).count();

        // If there are many single loads but no vector loads, suggest vectorization
        if single_loads >= 4 && vector_loads == 0 {
            bugs.push(PtxBug {
                class: PtxBugClass::UnoptimizedMemoryPattern,
                line: 0,
                instruction: format!("{} single f32 loads, 0 vector loads", single_loads),
                message: "Multiple single-element loads could potentially be vectorized"
                    .to_string(),
                fix: Some(
                    "Consider using ld.global.v2.f32 or ld.global.v4.f32 for consecutive addresses"
                        .to_string(),
                ),
            });
        }

        // Pattern 2: Look for non-coalesced access hints
        // Strided access: base + i * stride where stride != sizeof(element)
        let strided_pattern = Regex::new(r"mul\.wide\.[us]32\s+%\w+,\s*%\w+,\s*(\d+)")
            .expect("invariant: regex pattern is valid");
        let mut suspicious_strides = Vec::new();

        // Known quantization block strides (not bugs - legitimate data layouts)
        // Q4_K: 144 bytes, Q5_K: 176 bytes, Q6_K: 210 bytes, Q8_K: 256 bytes
        let quantization_strides: HashSet<u32> = [144, 176, 210, 256, 512].into_iter().collect();

        for (line_num, line) in lines.iter().enumerate() {
            if let Some(caps) = strided_pattern.captures(line) {
                if let Ok(stride) = caps
                    .get(1)
                    .expect("invariant: capture group exists")
                    .as_str()
                    .parse::<u32>()
                {
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
                message: "Non-standard stride may indicate strided (non-coalesced) access"
                    .to_string(),
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
        let reg_pattern =
            Regex::new(r"\.reg\s+\.\w+\s+%\w+<(\d+)>").expect("invariant: regex pattern is valid");
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
        let pred_pattern =
            Regex::new(r"\.reg\s+\.pred\s+%p<(\d+)>").expect("invariant: regex pattern is valid");
        if let Some(caps) = pred_pattern.captures(ptx) {
            if let Ok(pred_count) = caps
                .get(1)
                .expect("invariant: capture group exists")
                .as_str()
                .parse::<usize>()
            {
                if pred_count > 8 {
                    bugs.push(PtxBug {
                        class: PtxBugClass::PredicateOverflow,
                        line: 0,
                        instruction: format!(".reg .pred %p<{}>", pred_count),
                        message: format!(
                            "Predicate overflow: {} predicates declared (max 8 hardware registers)",
                            pred_count
                        ),
                        fix: Some(
                            "Reduce predicate usage by combining conditions or using branches"
                                .to_string(),
                        ),
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
                            fix: Some(
                                "Implement complete kernel or use trueno-gpu generation"
                                    .to_string(),
                            ),
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
        let label_pattern = Regex::new(r"^(\w+):$").expect("invariant: regex pattern is valid");
        let branch_pattern = Regex::new(r"^\s*(?:@%\w+\s+)?bra\s+(\w+);")
            .expect("invariant: regex pattern is valid");

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();

            // Check if this is a loop label
            if let Some(label_caps) = label_pattern.captures(line) {
                let label = label_caps
                    .get(1)
                    .expect("invariant: capture group exists")
                    .as_str();

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
                        "add.", "sub.", "mul.", "div.", "fma.", "mad.", "ld.", "st.", "cvt.",
                        "mov.", "setp.", "and.", "or.", "xor.", "shl.", "shr.", "min.", "max.",
                        "abs.", "neg.", "rcp.", "sqrt.", "rsqrt.", "sin.", "cos.", "ex2.", "lg2.",
                    ];
                    for op in &compute_ops {
                        if inner.contains(op) {
                            has_computation = true;
                            break;
                        }
                    }

                    // Check for branch back to loop label
                    if let Some(br_caps) = branch_pattern.captures(inner) {
                        let target = br_caps
                            .get(1)
                            .expect("invariant: capture group exists")
                            .as_str();
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
                        message: "Loop body contains no computation - may be placeholder code"
                            .to_string(),
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
        let has_predicated_branch = Regex::new(r"@%p\d+\s+bra")
            .expect("invariant: regex pattern is valid")
            .is_match(ptx);

        // If kernel uses tid and global memory but has no bounds check
        if has_tid && !has_setp_lt && !has_predicated_branch {
            bugs.push(PtxBug {
                class: PtxBugClass::MissingBoundsCheck,
                line: 0,
                instruction: "No setp.lt/ge with predicated branch found".to_string(),
                message: "Kernel accesses global memory but may lack thread bounds checking"
                    .to_string(),
                fix: Some("Add: setp.lt.u32 %p0, %tid, %size; @%p0 bra do_work;".to_string()),
            });
        }

        bugs
    }

    /// Detect dead code (P2)
    /// Code after unconditional ret or bra that can never execute
    fn detect_dead_code(&self, _ptx: &str, lines: &[&str]) -> Vec<PtxBug> {
        let mut bugs = Vec::new();

        let unconditional_ret = Regex::new(r"^\s*ret;").expect("invariant: regex pattern is valid");
        let unconditional_bra =
            Regex::new(r"^\s*bra\s+\w+;").expect("invariant: regex pattern is valid"); // No @%p prefix
        let label_pattern = Regex::new(r"^\w+:$").expect("invariant: regex pattern is valid");

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
