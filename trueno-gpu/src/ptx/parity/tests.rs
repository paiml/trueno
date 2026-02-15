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
