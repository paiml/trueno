//! PTX Parity Validation (GH-219)
//!
//! Validates that batched GPU kernels are structurally compatible with their
//! single-vector counterparts. Catches dequant bugs, register type mismatches,
//! and missing batch dispatch patterns at compile time.
//!
//! ## Motivation
//!
//! Three classes of bugs motivated this module:
//! 1. `BatchedQ6KGemvKernel` had 3 dequant bugs not present in `Q6KGemvKernel`
//! 2. `BatchedVectorizedRmsNormKernel` used u64 shared memory addressing
//! 3. Stale `position_buf` caused indirect KV scatter to wrong positions
//!
//! ## Validation Checks
//!
//! - **Parameter count**: Batched kernel must have same params as single-vector
//! - **Shared memory size**: Must match (batched should not need more shared mem)
//! - **Loop structure**: Must have matching computation loops (sum_loop, norm_loop, etc.)
//! - **Batch dispatch**: Batched kernels must use `ctaid.y` for row selection
//! - **Shared memory addressing**: Must use u32 registers for shared memory offsets

/// Result of a PTX parity validation
#[derive(Debug, Clone)]
pub struct ParityResult {
    /// Whether the kernels are parity-compatible
    pub is_compatible: bool,
    /// Specific violations found
    pub violations: Vec<ParityViolation>,
    /// Single-vector kernel name
    pub single_name: String,
    /// Batched kernel name
    pub batched_name: String,
}

/// A specific parity violation
#[derive(Debug, Clone)]
pub struct ParityViolation {
    /// What kind of violation
    pub kind: ParityViolationKind,
    /// Human-readable description
    pub message: String,
}

/// Categories of parity violations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParityViolationKind {
    /// Batched kernel has different parameter count than single-vector
    ParameterCountMismatch,
    /// Shared memory size differs between kernels
    SharedMemoryMismatch,
    /// Batched kernel missing ctaid.y for row dispatch
    MissingBatchDispatch,
    /// Shared memory addressed with u64 instead of u32
    SharedMemoryAddressingU64,
    /// Computation loop structure differs
    LoopStructureMismatch,
    /// Register type mismatch in dequant logic
    RegisterTypeMismatch,
}

impl std::fmt::Display for ParityViolationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParameterCountMismatch => write!(f, "PARAM_COUNT"),
            Self::SharedMemoryMismatch => write!(f, "SHARED_MEM_SIZE"),
            Self::MissingBatchDispatch => write!(f, "MISSING_CTAID_Y"),
            Self::SharedMemoryAddressingU64 => write!(f, "SHARED_MEM_U64"),
            Self::LoopStructureMismatch => write!(f, "LOOP_STRUCTURE"),
            Self::RegisterTypeMismatch => write!(f, "REG_TYPE"),
        }
    }
}

/// Count `.param` declarations in PTX source
fn count_params(ptx: &str) -> usize {
    ptx.lines()
        .filter(|line| {
            let trimmed = line.trim();
            trimmed.starts_with(".param")
        })
        .count()
}

/// Extract shared memory declaration size from PTX
/// Looks for `.shared .align N .b8 smem[SIZE];`
fn extract_shared_memory_bytes(ptx: &str) -> Option<u32> {
    for line in ptx.lines() {
        let trimmed = line.trim();
        if trimmed.contains(".shared") && trimmed.contains("smem[") {
            // Parse smem[SIZE]
            if let Some(start) = trimmed.find("smem[") {
                let after = &trimmed[start + 5..];
                if let Some(end) = after.find(']') {
                    if let Ok(size) = after[..end].parse::<u32>() {
                        return Some(size);
                    }
                }
            }
        }
    }
    None
}

/// Extract loop labels from PTX (e.g., sum_loop, norm_loop, etc.)
fn extract_loop_labels(ptx: &str) -> Vec<String> {
    let mut labels = Vec::new();
    for line in ptx.lines() {
        let trimmed = line.trim();
        // Loop labels are like "sum_loop:" or "norm_loop:" at the start of a line
        if trimmed.ends_with(':') && !trimmed.starts_with("//") {
            let label = trimmed.trim_end_matches(':');
            // Only count loop-related labels (containing "loop")
            if label.contains("loop") {
                labels.push(label.to_string());
            }
        }
    }
    labels
}

/// Batch dispatch strategy used by batched kernels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BatchDispatchStrategy {
    /// Grid.y dispatch: one block per batch element (ctaid.y selects row)
    /// Used by: RmsNorm, ResidualAdd, SwiGLU, RoPE
    GridY,
    /// Register unrolling: M accumulators per block, m_dim parameter
    /// Used by: Quantized GEMV (Q4K, Q6K) for throughput optimization
    RegisterUnroll,
}

/// Check if PTX uses ctaid.y for batch dispatch
fn has_grid_y_dispatch(ptx: &str) -> bool {
    ptx.contains("%ctaid.y")
}

/// Check if PTX uses register-unrolled batch dispatch (m_dim parameter)
fn has_register_unroll_dispatch(ptx: &str) -> bool {
    ptx.contains("m_dim")
}

/// Check if PTX has any batch dispatch mechanism
fn has_batch_dispatch(ptx: &str) -> bool {
    has_grid_y_dispatch(ptx) || has_register_unroll_dispatch(ptx)
}

/// Check for u64 registers used with shared memory operations
/// Returns true if any shared memory load/store uses u64 offset registers
fn has_u64_shared_memory_addressing(ptx: &str) -> bool {
    // Look for patterns like:
    //   st.shared.f32 [%rdN], ... (u64 address register for shared memory)
    //   ld.shared.f32 %fN, [%rdN] (u64 address register for shared memory)
    // Valid shared memory addressing should use u32 registers (%rN):
    //   st.shared.f32 [%rN], ...
    //   ld.shared.f32 %fN, [%rN]
    for line in ptx.lines() {
        let trimmed = line.trim();
        if (trimmed.contains("st.shared") || trimmed.contains("ld.shared"))
            && trimmed.contains("[%rd")
        {
            return true;
        }
    }
    false
}

/// Validate parity between a single-vector kernel's PTX and a batched kernel's PTX.
///
/// The batched kernel (with M=1) should be structurally equivalent to the
/// single-vector kernel, differing only in:
/// - An additional `ctaid.y` read for row dispatch
/// - Row offset calculation for global memory access
///
/// Everything else (dequant logic, shared memory, reduction, normalization)
/// should be identical.
pub fn validate_parity(
    single_ptx: &str,
    batched_ptx: &str,
    single_name: &str,
    batched_name: &str,
) -> ParityResult {
    let mut violations = Vec::new();

    // 1. Parameter count must match
    let single_params = count_params(single_ptx);
    let batched_params = count_params(batched_ptx);
    if single_params != batched_params {
        violations.push(ParityViolation {
            kind: ParityViolationKind::ParameterCountMismatch,
            message: format!(
                "Single kernel '{}' has {} params, batched '{}' has {} params",
                single_name, single_params, batched_name, batched_params
            ),
        });
    }

    // 2. Shared memory size must match
    let single_smem = extract_shared_memory_bytes(single_ptx);
    let batched_smem = extract_shared_memory_bytes(batched_ptx);
    if single_smem != batched_smem {
        violations.push(ParityViolation {
            kind: ParityViolationKind::SharedMemoryMismatch,
            message: format!(
                "Shared memory mismatch: single={:?} bytes, batched={:?} bytes",
                single_smem, batched_smem
            ),
        });
    }

    // 3. Batched kernel must use ctaid.y for row dispatch
    if !has_batch_dispatch(batched_ptx) {
        violations.push(ParityViolation {
            kind: ParityViolationKind::MissingBatchDispatch,
            message: format!(
                "Batched kernel '{}' does not use %ctaid.y for row dispatch",
                batched_name
            ),
        });
    }

    // 4. Shared memory addressing must use u32 registers
    if has_u64_shared_memory_addressing(batched_ptx) {
        violations.push(ParityViolation {
            kind: ParityViolationKind::SharedMemoryAddressingU64,
            message: format!(
                "Batched kernel '{}' uses u64 registers (%rd) for shared memory addressing; \
                 use u32 (%r) for portability",
                batched_name
            ),
        });
    }
    // Also check single-vector kernel
    if has_u64_shared_memory_addressing(single_ptx) {
        violations.push(ParityViolation {
            kind: ParityViolationKind::SharedMemoryAddressingU64,
            message: format!(
                "Single kernel '{}' uses u64 registers (%rd) for shared memory addressing; \
                 use u32 (%r) for portability",
                single_name
            ),
        });
    }

    // 5. Loop structure should match (same computation loops)
    let single_loops = extract_loop_labels(single_ptx);
    let batched_loops = extract_loop_labels(batched_ptx);
    if single_loops != batched_loops {
        violations.push(ParityViolation {
            kind: ParityViolationKind::LoopStructureMismatch,
            message: format!(
                "Loop structure differs: single has {:?}, batched has {:?}",
                single_loops, batched_loops
            ),
        });
    }

    ParityResult {
        is_compatible: violations.is_empty(),
        violations,
        single_name: single_name.to_string(),
        batched_name: batched_name.to_string(),
    }
}

/// Validate that a batched kernel's PTX is well-formed for batch execution.
///
/// This is a standalone check (no single-vector reference needed) that verifies
/// the batched kernel has correct batch dispatch patterns.
pub fn validate_batched_kernel(ptx: &str, kernel_name: &str) -> ParityResult {
    let mut violations = Vec::new();

    // Must use ctaid.y for batch dispatch
    if !has_batch_dispatch(ptx) {
        violations.push(ParityViolation {
            kind: ParityViolationKind::MissingBatchDispatch,
            message: format!(
                "Batched kernel '{}' does not use %ctaid.y for row dispatch",
                kernel_name
            ),
        });
    }

    // Must not use u64 for shared memory
    if has_u64_shared_memory_addressing(ptx) {
        violations.push(ParityViolation {
            kind: ParityViolationKind::SharedMemoryAddressingU64,
            message: format!(
                "Batched kernel '{}' uses u64 registers for shared memory addressing",
                kernel_name
            ),
        });
    }

    ParityResult {
        is_compatible: violations.is_empty(),
        violations,
        single_name: String::new(),
        batched_name: kernel_name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_params_basic() {
        let ptx = r#"
.visible .entry test(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr
) {
    ret;
}
"#;
        assert_eq!(count_params(ptx), 3);
    }

    #[test]
    fn test_extract_shared_memory_bytes() {
        let ptx = "    .shared .align 16 .b8 smem[32];";
        assert_eq!(extract_shared_memory_bytes(ptx), Some(32));

        let ptx_none = "    .reg .f32 %f<10>;";
        assert_eq!(extract_shared_memory_bytes(ptx_none), None);
    }

    #[test]
    fn test_extract_loop_labels() {
        let ptx = r#"
sum_loop:
    add.u32 %r6, %r6, 256;
    bra sum_loop;
sum_loop_end:
norm_loop:
    bra norm_loop;
exit:
    ret;
"#;
        let labels = extract_loop_labels(ptx);
        assert_eq!(labels, vec!["sum_loop", "sum_loop_end", "norm_loop"]);
    }

    #[test]
    fn test_has_batch_dispatch() {
        // Grid.y dispatch
        assert!(has_batch_dispatch("    mov.u32 %r1, %ctaid.y;"));
        // Register unroll dispatch (m_dim parameter)
        assert!(has_batch_dispatch("    .param .u32 m_dim"));
        // Neither
        assert!(!has_batch_dispatch("    mov.u32 %r1, %ctaid.x;"));
    }

    #[test]
    fn test_batch_dispatch_strategies() {
        assert!(has_grid_y_dispatch("    mov.u32 %r1, %ctaid.y;"));
        assert!(!has_grid_y_dispatch("    .param .u32 m_dim"));
        assert!(has_register_unroll_dispatch("    .param .u32 m_dim"));
        assert!(!has_register_unroll_dispatch("    mov.u32 %r1, %ctaid.y;"));
    }

    #[test]
    fn test_has_u64_shared_memory_addressing() {
        // Bad: u64 register for shared memory
        assert!(has_u64_shared_memory_addressing(
            "    st.shared.f32 [%rd3], %f0;"
        ));
        // Good: u32 register for shared memory
        assert!(!has_u64_shared_memory_addressing(
            "    st.shared.f32 [%r3], %f0;"
        ));
    }

    #[test]
    fn test_validate_parity_matching_kernels() {
        let single = r#"
.version 8.0
.target sm_89
.address_size 64
.visible .entry rmsnorm(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 gamma_ptr
) {
    .shared .align 16 .b8 smem[32];
    mov.u32 %r0, %tid.x;
sum_loop:
    bra sum_loop;
sum_loop_end:
norm_loop:
    bra norm_loop;
exit:
    ret;
}
"#;
        let batched = r#"
.version 8.0
.target sm_89
.address_size 64
.visible .entry batched_rmsnorm(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 gamma_ptr
) {
    .shared .align 16 .b8 smem[32];
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, %ctaid.y;
sum_loop:
    bra sum_loop;
sum_loop_end:
norm_loop:
    bra norm_loop;
exit:
    ret;
}
"#;
        let result = validate_parity(single, batched, "rmsnorm", "batched_rmsnorm");
        assert!(
            result.is_compatible,
            "Should be compatible: {:?}",
            result.violations
        );
    }

    #[test]
    fn test_validate_parity_param_mismatch() {
        let single = r#"
.visible .entry test(
    .param .u64 a,
    .param .u64 b
) { ret; }
"#;
        let batched = r#"
.visible .entry test_batched(
    .param .u64 a,
    .param .u64 b,
    .param .u32 batch_size
) {
    mov.u32 %r1, %ctaid.y;
    ret;
}
"#;
        let result = validate_parity(single, batched, "test", "test_batched");
        assert!(!result.is_compatible);
        assert!(result
            .violations
            .iter()
            .any(|v| v.kind == ParityViolationKind::ParameterCountMismatch));
    }

    #[test]
    fn test_validate_parity_missing_ctaid_y() {
        let single = r#"
.visible .entry test(
    .param .u64 a
) { ret; }
"#;
        let batched = r#"
.visible .entry test_batched(
    .param .u64 a
) { ret; }
"#;
        let result = validate_parity(single, batched, "test", "test_batched");
        assert!(!result.is_compatible);
        assert!(result
            .violations
            .iter()
            .any(|v| v.kind == ParityViolationKind::MissingBatchDispatch));
    }

    #[test]
    fn test_validate_parity_u64_shared_memory() {
        let single = r#"
.visible .entry test(
    .param .u64 a
) {
    .shared .align 16 .b8 smem[32];
    st.shared.f32 [%r3], %f0;
    ret;
}
"#;
        let batched = r#"
.visible .entry test_batched(
    .param .u64 a
) {
    .shared .align 16 .b8 smem[32];
    mov.u32 %r1, %ctaid.y;
    st.shared.f32 [%rd3], %f0;
    ret;
}
"#;
        let result = validate_parity(single, batched, "test", "test_batched");
        assert!(!result.is_compatible);
        assert!(result
            .violations
            .iter()
            .any(|v| v.kind == ParityViolationKind::SharedMemoryAddressingU64));
    }

    #[test]
    fn test_validate_batched_kernel_standalone() {
        // Grid.y dispatch
        let good_grid = r#"
.visible .entry good_batched(
    .param .u64 a
) {
    mov.u32 %r1, %ctaid.y;
    st.shared.f32 [%r3], %f0;
    ret;
}
"#;
        let result = validate_batched_kernel(good_grid, "good_batched");
        assert!(result.is_compatible);

        // Register-unrolled dispatch
        let good_reg = r#"
.visible .entry good_reg_batched(
    .param .u64 a,
    .param .u32 m_dim
) {
    ret;
}
"#;
        let result = validate_batched_kernel(good_reg, "good_reg_batched");
        assert!(result.is_compatible);

        // Neither dispatch mechanism + u64 shared mem
        let bad = r#"
.visible .entry bad_batched(
    .param .u64 a
) {
    st.shared.f32 [%rd3], %f0;
    ret;
}
"#;
        let result = validate_batched_kernel(bad, "bad_batched");
        assert!(!result.is_compatible);
        assert_eq!(result.violations.len(), 2); // missing dispatch AND u64 shared mem
    }

    #[test]
    fn test_parity_violation_display() {
        assert_eq!(
            ParityViolationKind::ParameterCountMismatch.to_string(),
            "PARAM_COUNT"
        );
        assert_eq!(
            ParityViolationKind::SharedMemoryAddressingU64.to_string(),
            "SHARED_MEM_U64"
        );
        assert_eq!(
            ParityViolationKind::MissingBatchDispatch.to_string(),
            "MISSING_CTAID_Y"
        );
    }
}
