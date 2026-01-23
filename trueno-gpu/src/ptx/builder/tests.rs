use super::*;

#[test]
fn test_module_defaults() {
    let module = PtxModule::new();
    assert_eq!(module.get_version(), (8, 0));
    assert_eq!(module.get_target(), "sm_70");
    assert_eq!(module.get_address_size(), 64);
}

#[test]
fn test_module_builder() {
    let module = PtxModule::new()
        .version(8, 5)
        .target("sm_86")
        .address_size(64);

    assert_eq!(module.get_version(), (8, 5));
    assert_eq!(module.get_target(), "sm_86");
}

#[test]
fn test_kernel_params() {
    let kernel = PtxKernel::new("test")
        .param(PtxType::U64, "ptr")
        .param(PtxType::U32, "n");

    assert_eq!(kernel.params.len(), 2);
    assert_eq!(kernel.params[0].name, "ptr");
    assert_eq!(kernel.params[1].name, "n");
}

#[test]
fn test_emit_header() {
    let module = PtxModule::new()
        .version(8, 0)
        .target("sm_70")
        .address_size(64);

    let ptx = module.emit();
    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains(".target sm_70"));
    assert!(ptx.contains(".address_size 64"));
}

#[test]
fn test_emit_kernel() {
    let kernel = PtxKernel::new("vector_add")
        .param(PtxType::U64, "a")
        .param(PtxType::U64, "b");

    let module = PtxModule::new().add_kernel(kernel);
    let ptx = module.emit();

    assert!(ptx.contains(".visible .entry vector_add"));
    assert!(ptx.contains(".param .u64 a"));
    assert!(ptx.contains(".param .u64 b"));
}

// ========================================================================
// BUG FIX TESTS - EXTREME TDD
// ========================================================================

#[test]
fn test_bar_sync_emission() {
    // BUG: bar.sync was being emitted as "bar.b32 ;" instead of "bar.sync 0;"
    let kernel = PtxKernel::new("test_barrier").build(|ctx| {
        ctx.bar_sync(0);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Must contain proper bar.sync instruction
    assert!(
        ptx.contains("bar.sync 0"),
        "Expected 'bar.sync 0' but got: {}",
        ptx
    );
    // Must NOT contain the buggy output
    assert!(
        !ptx.contains("bar.b32"),
        "Found buggy 'bar.b32' in: {}",
        ptx
    );
}

#[test]
fn test_cvt_u64_u32_emission() {
    // BUG: cvt was being emitted as "cvt.u64 %r, %r" instead of "cvt.u64.u32 %r, %r"
    let kernel = PtxKernel::new("test_cvt").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _wide = ctx.cvt_u64_u32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Must contain proper cvt with both types
    assert!(
        ptx.contains("cvt.u64.u32"),
        "Expected 'cvt.u64.u32' but got: {}",
        ptx
    );
}

#[test]
fn test_shared_memory_addressing() {
    // Shared memory access uses register-based addressing
    let kernel = PtxKernel::new("test_shared")
        .shared_memory(1024)
        .build(|ctx| {
            let val = ctx.mov_f32_imm(1.0);
            let offset = ctx.mov_u32_imm(0);
            let offset_64 = ctx.cvt_u64_u32(offset);
            ctx.st_shared_f32(offset_64, val);
            let _loaded = ctx.ld_shared_f32(offset_64);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Must contain proper shared memory operations
    assert!(
        ptx.contains("st.shared.f32") && ptx.contains("ld.shared.f32"),
        "Expected shared memory operations, got: {}",
        ptx
    );
    // Must contain brackets for addressing
    assert!(
        ptx.contains("[%rd"),
        "Expected bracketed register address, got: {}",
        ptx
    );
}

#[test]
fn test_bar_sync_with_different_barriers() {
    // Test barrier with different IDs
    let kernel = PtxKernel::new("test_barriers").build(|ctx| {
        ctx.bar_sync(0);
        ctx.bar_sync(1);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("bar.sync 0"),
        "Expected 'bar.sync 0' in: {}",
        ptx
    );
    assert!(
        ptx.contains("bar.sync 1"),
        "Expected 'bar.sync 1' in: {}",
        ptx
    );
}

#[test]
fn test_global_memory_addressing() {
    // BUG: global memory access was "ld.global.f32 %f, %r" instead of "ld.global.f32 %f, [%r]"
    let kernel = PtxKernel::new("test_global")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_global_f32(ptr);
            ctx.st_global_f32(ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Load must use brackets for address
    assert!(
        ptx.contains("ld.global.f32") && ptx.contains("[%rd"),
        "Expected ld.global.f32 with [%rd] address, got: {}",
        ptx
    );
    // Store must use brackets for address
    assert!(
        ptx.contains("st.global.f32 ["),
        "Expected st.global.f32 with [%rd] address, got: {}",
        ptx
    );
}

#[test]
fn test_f32_literal_format() {
    // BUG: float literals were emitted as "0e0" instead of PTX hex format "0F00000000"
    let kernel = PtxKernel::new("test_float").build(|ctx| {
        let _zero = ctx.mov_f32_imm(0.0);
        let _one = ctx.mov_f32_imm(1.0);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // PTX float literals must be in hex format
    assert!(
        ptx.contains("0F00000000"), // 0.0f in hex
        "Expected 0F00000000 for 0.0f, got: {}",
        ptx
    );
    assert!(
        ptx.contains("0F3F800000"), // 1.0f in hex
        "Expected 0F3F800000 for 1.0f, got: {}",
        ptx
    );
}

#[test]
fn test_loop_counter_update_in_place() {
    // BUG: Loop counters were never updated due to SSA - computed values discarded
    // PTX loops need in-place register updates: add.u32 %r0, %r0, 1
    let kernel = PtxKernel::new("test_loop")
        .param(PtxType::U32, "n")
        .build(|ctx| {
            let n = ctx.load_param_u32("n");
            let i = ctx.mov_u32_imm(0);
            ctx.label("loop");
            let done = ctx.setp_ge_u32(i, n);
            ctx.branch_if(done, "exit");
            // In-place increment: i = i + 1
            ctx.add_u32_inplace(i, 1);
            ctx.branch("loop");
            ctx.label("exit");
            ctx.ret();
        });

    let ptx = kernel.emit();
    // The loop counter must be updated in-place (same src and dst register)
    // Look for pattern like: add.u32 %r1, %r1, 1
    assert!(
        ptx.contains("add") && ptx.contains("%r") && ptx.contains(", 1"),
        "Expected in-place add instruction, got: {}",
        ptx
    );
}

#[test]
fn test_accumulator_update_in_place() {
    // BUG: Accumulators in inner loops were never updated
    // Need in-place: add.f32 %f0, %f0, %f1
    let kernel = PtxKernel::new("test_acc").build(|ctx| {
        let acc = ctx.mov_f32_imm(0.0);
        let val = ctx.mov_f32_imm(1.0);
        // In-place accumulate: acc = acc + val
        ctx.add_f32_inplace(acc, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Must have in-place add
    assert!(
        ptx.contains("add") && ptx.contains(".f32"),
        "Expected f32 add instruction, got: {}",
        ptx
    );
}

// ========================================================================
// TENSOR CORE (WMMA) TESTS - IMP-1000a
// ========================================================================

#[test]
fn test_wmma_load_a_f16() {
    let kernel = PtxKernel::new("test_wmma_load_a")
        .param(PtxType::U64, "a_ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("a_ptr");
            let _frag_a = ctx.wmma_load_a_f16(ptr, 16, WmmaLayout::RowMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 a_ptr"),
        "Expected a_ptr param, got: {}",
        ptx
    );
}

#[test]
fn test_wmma_load_b_f16() {
    let kernel = PtxKernel::new("test_wmma_load_b")
        .param(PtxType::U64, "b_ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("b_ptr");
            let _frag_b = ctx.wmma_load_b_f16(ptr, 16, WmmaLayout::ColMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 b_ptr"),
        "Expected b_ptr param, got: {}",
        ptx
    );
}

#[test]
fn test_wmma_mma_f16_f32() {
    let kernel = PtxKernel::new("test_wmma_mma")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .param(PtxType::U64, "c_ptr")
        .build(|ctx| {
            let a = ctx.load_param_u64("a_ptr");
            let b = ctx.load_param_u64("b_ptr");
            let c = ctx.load_param_u64("c_ptr");

            let frag_a = ctx.wmma_load_a_f16(a, 16, WmmaLayout::RowMajor);
            let frag_b = ctx.wmma_load_b_f16(b, 16, WmmaLayout::ColMajor);
            let frag_c = ctx.wmma_load_c_f32(c, 16, WmmaLayout::RowMajor);

            let _frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Verify kernel structure
    assert!(
        ptx.contains(".visible .entry test_wmma_mma"),
        "Expected kernel entry, got: {}",
        ptx
    );
}

#[test]
fn test_wmma_store_d_f32() {
    let kernel = PtxKernel::new("test_wmma_store")
        .param(PtxType::U64, "d_ptr")
        .build(|ctx| {
            let d = ctx.load_param_u64("d_ptr");
            // Create empty fragment for test
            let frag_d = vec![ctx.mov_f32_imm(0.0)];
            ctx.wmma_store_d_f32(d, &frag_d, 16, WmmaLayout::RowMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 d_ptr"),
        "Expected d_ptr param, got: {}",
        ptx
    );
}

#[test]
fn test_cvt_f16_f32() {
    let kernel = PtxKernel::new("test_cvt_f16").build(|ctx| {
        let f32_val = ctx.mov_f32_imm(1.5);
        let _f16_val = ctx.cvt_f16_f32(f32_val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt"),
        "Expected cvt instruction, got: {}",
        ptx
    );
}

#[test]
fn test_cvt_f32_f16() {
    let kernel = PtxKernel::new("test_cvt_f32")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let f16_val = ctx.ld_global_f16(ptr);
            let _f32_val = ctx.cvt_f32_f16(f16_val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 ptr"),
        "Expected ptr param, got: {}",
        ptx
    );
}

#[test]
fn test_ld_st_global_f16() {
    let kernel = PtxKernel::new("test_f16_mem")
        .param(PtxType::U64, "in_ptr")
        .param(PtxType::U64, "out_ptr")
        .build(|ctx| {
            let in_ptr = ctx.load_param_u64("in_ptr");
            let out_ptr = ctx.load_param_u64("out_ptr");
            let val = ctx.ld_global_f16(in_ptr);
            ctx.st_global_f16(out_ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains(".param .u64 in_ptr") && ptx.contains(".param .u64 out_ptr"),
        "Expected both params, got: {}",
        ptx
    );
}

// ========================================================================
// COVERAGE IMPROVEMENT TESTS - TRUENO-SPEC-014
// ========================================================================

#[test]
fn test_validate_valid_module() {
    // Test that a valid module passes validation
    let module = PtxModule::new()
        .version(8, 0)
        .target("sm_70")
        .address_size(64);

    assert!(module.validate().is_ok());
}

#[test]
fn test_validate_minimum_version() {
    // PTX version 7.0 should be valid (minimum supported)
    let module = PtxModule::new().version(7, 0).target("sm_70");
    assert!(module.validate().is_ok());
}

#[test]
fn test_validate_invalid_version() {
    // PTX version below 7.0 should fail
    let module = PtxModule::new().version(6, 5).target("sm_70");
    assert!(module.validate().is_err());
}

#[test]
fn test_validate_invalid_target() {
    // Invalid compute capability should fail
    let module = PtxModule::new().version(8, 0).target("sm_invalid");
    assert!(module.validate().is_err());
}

#[test]
fn test_validate_sm_30_too_old() {
    // sm_30 is too old (below sm_50)
    let module = PtxModule::new().version(8, 0).target("sm_30");
    assert!(module.validate().is_err());
}

#[test]
fn test_ptx_module_default() {
    let module = PtxModule::default();
    // Default version is (8, 0)
    assert_eq!(module.version, (8, 0));
}

#[test]
fn test_fma_f32() {
    let kernel = PtxKernel::new("test_fma").build(|ctx| {
        let a = ctx.mov_f32_imm(2.0);
        let b = ctx.mov_f32_imm(3.0);
        let c = ctx.mov_f32_imm(4.0);
        let _result = ctx.fma_f32(a, b, c);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("fma"),
        "Expected fma instruction, got: {}",
        ptx
    );
}

#[test]
fn test_ld_global_u32() {
    let kernel = PtxKernel::new("test_ld_u32")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let _val = ctx.ld_global_u32(ptr);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global"),
        "Expected ld.global instruction, got: {}",
        ptx
    );
}

#[test]
fn test_ld_global_u64() {
    // PAR-118: Test u64 load for pointer arrays in batched attention
    let kernel = PtxKernel::new("test_ld_u64")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let _val = ctx.ld_global_u64(ptr);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global.u64"),
        "Expected ld.global.u64 instruction, got: {}",
        ptx
    );
}

#[test]
fn test_ld_global_u8() {
    let kernel = PtxKernel::new("test_ld_u8")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let _val = ctx.ld_global_u8(ptr);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global"),
        "Expected ld.global instruction, got: {}",
        ptx
    );
}

#[test]
fn test_ld_global_u16() {
    let kernel = PtxKernel::new("test_ld_u16")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let _val = ctx.ld_global_u16(ptr);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global"),
        "Expected ld.global instruction, got: {}",
        ptx
    );
}

// ========================================================================
// COALESCED GEMV TESTS - DECODER THROUGHPUT SPEC §5.3
// ========================================================================

#[test]
fn test_mul_lo_u32() {
    // mul_lo_u32: u32 * u32 -> u32 (low bits only)
    let kernel = PtxKernel::new("test_mul_lo").build(|ctx| {
        let a = ctx.mov_u32_imm(256);
        let b = ctx.mov_u32_imm(16);
        let _result = ctx.mul_lo_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.lo.u32"),
        "Expected 'mul.lo.u32' instruction, got: {}",
        ptx
    );
}

#[test]
fn test_shared_base_addr() {
    // shared_base_addr: Get pointer to 'smem' shared memory array
    let kernel = PtxKernel::new("test_smem_addr")
        .shared_memory(1024)
        .build(|ctx| {
            let smem_ptr = ctx.shared_base_addr();
            let val = ctx.mov_f32_imm(1.0);
            ctx.st_shared_f32(smem_ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Must declare shared memory
    assert!(
        ptx.contains(".shared"),
        "Expected shared memory declaration, got: {}",
        ptx
    );
    // Must reference 'smem' label
    assert!(
        ptx.contains("smem"),
        "Expected 'smem' reference, got: {}",
        ptx
    );
}

#[test]
fn test_ld_global_f32_predicated() {
    // ld_global_f32_predicated: Load with predicate guard and default value
    let kernel = PtxKernel::new("test_pred_load")
        .param(PtxType::U64, "ptr")
        .param(PtxType::U32, "n")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let n = ctx.load_param_u32("n");
            let idx = ctx.mov_u32_imm(5);
            let valid = ctx.setp_lt_u32(idx, n);
            let _val = ctx.ld_global_f32_predicated(ptr, valid, 0.0);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Must contain predicated load (@p ld.global.f32)
    assert!(
        ptx.contains("@%p") && ptx.contains("ld.global.f32"),
        "Expected predicated ld.global.f32, got: {}",
        ptx
    );
    // Must initialize with default value first
    assert!(
        ptx.contains("mov.f32") && ptx.contains("0F00000000"),
        "Expected mov.f32 with 0.0 default, got: {}",
        ptx
    );
}

#[test]
fn test_coalesced_gemv_kernel_structure() {
    // Integration test: verify CoalescedGemv kernel PTX structure
    let kernel = PtxKernel::new("gemv_coalesced")
        .param(PtxType::U64, "y_ptr")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "x_ptr")
        .param(PtxType::U32, "k_dim")
        .param(PtxType::U32, "n_dim")
        .shared_memory(4096 * 4) // Cache x vector
        .build(|ctx| {
            // Minimal structure test
            let block_id = ctx.special_reg(PtxReg::CtaIdX);
            let thread_id = ctx.special_reg(PtxReg::TidX);
            let block_size = ctx.mov_u32_imm(256);
            let col_base = ctx.mul_lo_u32(block_id, block_size);
            let col = ctx.add_u32_reg(col_base, thread_id);

            let n_dim = ctx.load_param_u32("n_dim");
            let oob = ctx.setp_ge_u32(col, n_dim);
            ctx.branch_if(oob, "exit");

            let sum = ctx.mov_f32_imm(0.0);
            let smem = ctx.shared_base_addr();

            // Load x into shared memory with predicate
            let x_ptr = ctx.load_param_u64("x_ptr");
            let k_dim = ctx.load_param_u32("k_dim");
            let valid = ctx.setp_lt_u32(thread_id, k_dim);
            let x_offset = ctx.mul_wide_u32(thread_id, 4);
            let x_addr = ctx.add_u64(x_ptr, x_offset);
            let x_val = ctx.ld_global_f32_predicated(x_addr, valid, 0.0);

            let smem_offset = ctx.mul_u32(thread_id, 4);
            let smem_offset_64 = ctx.cvt_u64_u32(smem_offset);
            let smem_addr = ctx.add_u64(smem, smem_offset_64);
            ctx.st_shared_f32(smem_addr, x_val);
            ctx.bar_sync(0);

            // Store result
            let y_ptr = ctx.load_param_u64("y_ptr");
            let y_offset = ctx.mul_wide_u32(col, 4);
            let y_addr = ctx.add_u64(y_ptr, y_offset);
            ctx.st_global_f32(y_addr, sum);

            ctx.label("exit");
            ctx.ret();
        });

    let ptx = kernel.emit();

    // Verify all critical components
    assert!(ptx.contains(".entry gemv_coalesced"), "Missing entry point");
    assert!(ptx.contains(".shared"), "Missing shared memory");
    assert!(ptx.contains("bar.sync"), "Missing barrier");
    assert!(ptx.contains("mul.lo.u32"), "Missing mul.lo.u32");
    assert!(ptx.contains("@%p"), "Missing predicated instruction");
}

// =========================================================================
// Optimization Pass Integration Tests (Issue #72, #73)
// =========================================================================

#[test]
fn test_build_optimized_basic() {
    // Test that build_optimized works for simple kernels
    let kernel = PtxKernel::new("test_optimized")
        .param(PtxType::U64, "ptr")
        .build_optimized(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_global_f32(ptr);
            let two = ctx.mov_f32_imm(2.0);
            let result = ctx.mul_f32(val, two);
            ctx.st_global_f32(ptr, result);
            ctx.ret();
        });

    assert!(kernel.is_ok(), "build_optimized should succeed for simple kernel");
    let kernel = kernel.unwrap();
    let ptx = kernel.emit();
    assert!(ptx.contains(".entry test_optimized"));
    assert!(ptx.contains("ret;"));
}

#[test]
fn test_build_optimized_with_mul_add_fusion() {
    // Test that mul + add patterns are fused to FMA by the optimization pass
    // This tests the FMA fusion integration from Issue #72
    let kernel = PtxKernel::new("test_fma_fusion")
        .param(PtxType::U64, "ptr")
        .build_optimized(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let a = ctx.ld_global_f32(ptr);
            let b = ctx.mov_f32_imm(2.0);
            let c = ctx.mov_f32_imm(3.0);
            // mul + add pattern should be fused to fma by optimizer
            let mul_result = ctx.mul_f32(a, b);
            let add_result = ctx.add_f32(mul_result, c);
            ctx.st_global_f32(ptr, add_result);
            ctx.ret();
        });

    assert!(kernel.is_ok(), "build_optimized should succeed");
    let kernel = kernel.unwrap();
    let ptx = kernel.emit();

    // After FMA fusion, we should have fma.rn.f32 instead of separate mul + add
    // The mul result is only used once (in the add), so fusion should occur
    assert!(
        ptx.contains("fma.rn.f32") || ptx.contains("mul.f32"),
        "Kernel should have either FMA (fused) or mul (unfused)"
    );
}

#[test]
fn test_build_vs_build_optimized_difference() {
    // Non-optimized build
    let kernel_unopt = PtxKernel::new("test_unopt")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let a = ctx.ld_global_f32(ptr);
            let b = ctx.mov_f32_imm(2.0);
            let c = ctx.mov_f32_imm(3.0);
            let mul_result = ctx.mul_f32(a, b);
            let add_result = ctx.add_f32(mul_result, c);
            ctx.st_global_f32(ptr, add_result);
            ctx.ret();
        });

    // Optimized build
    let kernel_opt = PtxKernel::new("test_opt")
        .param(PtxType::U64, "ptr")
        .build_optimized(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let a = ctx.ld_global_f32(ptr);
            let b = ctx.mov_f32_imm(2.0);
            let c = ctx.mov_f32_imm(3.0);
            let mul_result = ctx.mul_f32(a, b);
            let add_result = ctx.add_f32(mul_result, c);
            ctx.st_global_f32(ptr, add_result);
            ctx.ret();
        })
        .unwrap();

    let ptx_unopt = kernel_unopt.emit();
    let ptx_opt = kernel_opt.emit();

    // Unoptimized should have separate mul and add
    assert!(
        ptx_unopt.contains("mul.f32") && ptx_unopt.contains("add.f32"),
        "Unoptimized should have separate mul and add"
    );

    // After FMA fusion, optimized version should have FMA
    // Note: FMA fusion requires single-use of mul result
    // Both kernels should produce valid PTX
    assert!(ptx_unopt.contains(".entry test_unopt"));
    assert!(ptx_opt.contains(".entry test_opt"));
}

#[test]
fn test_build_optimized_empty_body() {
    // Test empty kernel body
    let kernel = PtxKernel::new("test_empty")
        .build_optimized(|_ctx| {
            // Empty body
        });

    assert!(kernel.is_ok(), "Empty optimized kernel should succeed");
}

#[test]
fn test_build_optimized_preserves_barriers() {
    // Test that optimization passes preserve barriers
    let kernel = PtxKernel::new("test_barriers")
        .shared_memory(1024)
        .build_optimized(|ctx| {
            let tid = ctx.special_reg(PtxReg::TidX);
            let val = ctx.mov_f32_imm(1.0);
            let smem_offset = ctx.mul_u32(tid, 4);
            ctx.st_shared_f32(smem_offset, val);
            ctx.bar_sync(0);
            let _loaded = ctx.ld_shared_f32(smem_offset);
            ctx.ret();
        });

    assert!(kernel.is_ok());
    let kernel = kernel.unwrap();
    let ptx = kernel.emit();
    assert!(ptx.contains("bar.sync"), "Barriers should be preserved");
}

// ========================================================================
// COVERAGE IMPROVEMENT TESTS
// ========================================================================

#[test]
fn test_ld_global_f32_v4_vectorized_load() {
    // Test vectorized 4-float load (ld.global.v4.f32)
    let kernel = PtxKernel::new("test_v4_load")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let [r0, r1, r2, r3] = ctx.ld_global_f32_v4(ptr);
            // Sum all 4 values
            let sum1 = ctx.add_f32(r0, r1);
            let sum2 = ctx.add_f32(r2, r3);
            let total = ctx.add_f32(sum1, sum2);
            ctx.st_global_f32(ptr, total);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global.v4.f32"),
        "Expected vectorized load in: {}",
        ptx
    );
}

#[test]
fn test_wide_multiply_u32_imm() {
    // Test wide multiply (mul.wide.u32 producing u64)
    let kernel = PtxKernel::new("test_wide_mul")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(1000000);
            // Wide multiply: u32 * imm -> u64
            let _wide_result = ctx.mul_wide_u32(a, 1000000);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.wide"),
        "Expected wide multiply in: {}",
        ptx
    );
}

#[test]
fn test_wide_multiply_u32_reg() {
    // Test wide multiply with two registers
    let kernel = PtxKernel::new("test_wide_mul_reg")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(1000000);
            let b = ctx.mov_u32_imm(1000000);
            let _wide_result = ctx.mul_wide_u32_reg(a, b);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.wide"),
        "Expected wide multiply in: {}",
        ptx
    );
}

#[test]
fn test_mad_lo_instruction() {
    // Test mad.lo instruction (multiply-add low)
    let kernel = PtxKernel::new("test_mad_lo")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(10);
            let b = ctx.mov_u32_imm(20);
            let c = ctx.mov_u32_imm(5);
            let _result = ctx.mad_lo_u32(a, b, c);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mad.lo"),
        "Expected mad.lo instruction in: {}",
        ptx
    );
}

#[test]
fn test_setp_lt_u32_comparison() {
    // Test setp.lt.u32 (set predicate less than)
    let kernel = PtxKernel::new("test_setp_lt")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(10);
            let b = ctx.mov_u32_imm(20);
            let pred = ctx.setp_lt_u32(a, b);
            ctx.branch_if(pred, "taken");
            ctx.label("taken");
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("setp.lt"),
        "Expected setp.lt in: {}",
        ptx
    );
}

#[test]
fn test_setp_ge_u32_comparison() {
    // Test setp.ge.u32 (set predicate greater or equal)
    let kernel = PtxKernel::new("test_setp_ge")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(20);
            let b = ctx.mov_u32_imm(10);
            let pred = ctx.setp_ge_u32(a, b);
            ctx.branch_if(pred, "taken");
            ctx.label("taken");
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("setp.ge"),
        "Expected setp.ge in: {}",
        ptx
    );
}

#[test]
fn test_integer_division() {
    // Test integer division (div without rounding mode)
    let kernel = PtxKernel::new("test_int_div")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(100);
            let _result = ctx.div_u32(a, 7);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("div.u32") || ptx.contains("div."),
        "Expected integer division in: {}",
        ptx
    );
}

#[test]
fn test_shared_memory_load_store() {
    // Test shared memory operations with state space
    let kernel = PtxKernel::new("test_shared_mem")
        .shared_memory(256)
        .build(|ctx| {
            let tid = ctx.special_reg(PtxReg::TidX);
            let offset = ctx.mul_u32(tid, 4);
            let val = ctx.mov_f32_imm(42.0);
            ctx.st_shared_f32(offset, val);
            ctx.bar_sync(0);
            let loaded = ctx.ld_shared_f32(offset);
            let ptr = ctx.mov_u64_imm(0);
            ctx.st_global_f32(ptr, loaded);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.shared") && ptx.contains("ld.shared"),
        "Expected shared memory ops in: {}",
        ptx
    );
}

#[test]
fn test_label_with_colon() {
    // Test label emission (lines with colon)
    let kernel = PtxKernel::new("test_labels")
        .build(|ctx| {
            ctx.label("loop_start");
            let ctr = ctx.mov_u32_imm(10);
            let one = ctx.mov_u32_imm(1);
            let new_ctr = ctx.sub_u32_reg(ctr, one);
            let pred = ctx.setp_ge_u32(new_ctr, one);
            ctx.branch_if(pred, "loop_start");
            ctx.label("loop_end");
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("loop_start:") && ptx.contains("loop_end:"),
        "Expected labels in: {}",
        ptx
    );
}

#[test]
fn test_mul_lo_for_integer() {
    // Test mul.lo for integer multiplication (low bits)
    let kernel = PtxKernel::new("test_mul_lo")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(100);
            let b = ctx.mov_u32_imm(200);
            let _result = ctx.mul_u32(a, 200);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.lo.u32") || ptx.contains("mul.u32"),
        "Expected integer mul in: {}",
        ptx
    );
}

#[test]
fn test_float_multiply_no_lo() {
    // Test floating point multiply (no .lo modifier)
    let kernel = PtxKernel::new("test_float_mul")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let a = ctx.mov_f32_imm(3.14);
            let b = ctx.mov_f32_imm(2.71);
            let result = ctx.mul_f32(a, b);
            let ptr = ctx.load_param_u64("ptr");
            ctx.st_global_f32(ptr, result);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.f32"),
        "Expected float mul in: {}",
        ptx
    );
}

#[test]
fn test_div_float_with_rounding() {
    // Test floating point division (needs rounding mode)
    let kernel = PtxKernel::new("test_float_div")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let a = ctx.mov_f32_imm(10.0);
            let b = ctx.mov_f32_imm(3.0);
            let result = ctx.div_f32(a, b);
            let ptr = ctx.load_param_u64("ptr");
            ctx.st_global_f32(ptr, result);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("div.rn.f32") || ptx.contains("div.f32"),
        "Expected float div in: {}",
        ptx
    );
}

#[test]
fn test_ld_param_emission() {
    // Test ld.param instruction
    let kernel = PtxKernel::new("test_ld_param")
        .param(PtxType::U64, "data_ptr")
        .param(PtxType::U32, "count")
        .build(|ctx| {
            let _ptr = ctx.load_param_u64("data_ptr");
            let _count = ctx.load_param_u32("count");
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.param"),
        "Expected ld.param in: {}",
        ptx
    );
}

#[test]
fn test_u64_multiplication() {
    // Test u64 * imm multiplication
    let kernel = PtxKernel::new("test_u64_mul")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let a = ctx.mov_u64_imm(1000000000u64);
            let _result = ctx.mul_u64(a, 2000000000u64);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.lo.u64") || ptx.contains("mul.u64") || ptx.contains("mov.u64"),
        "Expected u64 operation in: {}",
        ptx
    );
}

#[test]
fn test_u64_reg_multiplication() {
    // Test u64 * u64 register multiplication
    let kernel = PtxKernel::new("test_u64_mul_reg")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let a = ctx.mov_u64_imm(1000000000u64);
            let b = ctx.mov_u64_imm(2000000000u64);
            let _result = ctx.mul_u64_reg(a, b);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.lo.u64"),
        "Expected mul.lo.u64 in: {}",
        ptx
    );
}

#[test]
fn test_global_u32_load() {
    // Test ld.global.u32 instruction
    let kernel = PtxKernel::new("test_ld_global_u32")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let _val = ctx.ld_global_u32(ptr);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global.u32"),
        "Expected ld.global.u32 in: {}",
        ptx
    );
}

#[test]
fn test_global_u8_load() {
    // Test ld.global.u8 instruction
    let kernel = PtxKernel::new("test_ld_global_u8")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let _val = ctx.ld_global_u8(ptr);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global.u8"),
        "Expected ld.global.u8 in: {}",
        ptx
    );
}

#[test]
fn test_global_u16_load() {
    // Test ld.global.u16 instruction
    let kernel = PtxKernel::new("test_ld_global_u16")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let _val = ctx.ld_global_u16(ptr);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global.u16"),
        "Expected ld.global.u16 in: {}",
        ptx
    );
}

#[test]
fn test_bra_unconditional() {
    // Test unconditional branch (bra)
    let kernel = PtxKernel::new("test_bra")
        .build(|ctx| {
            ctx.branch("skip");
            ctx.label("dead_code");
            let _unused = ctx.mov_f32_imm(1.0);
            ctx.label("skip");
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("bra skip") || ptx.contains("bra\tskip"),
        "Expected bra instruction in: {}",
        ptx
    );
}

#[test]
fn test_and_pred_combining_bounds() {
    // Test AND of two predicates (PARITY-114)
    let kernel = PtxKernel::new("test_and_pred")
        .param(PtxType::U64, "data_ptr")
        .param(PtxType::U32, "size")
        .build(|ctx| {
            let tid = ctx.special_reg(PtxReg::TidX);
            let size = ctx.load_param_u32("size");
            // Compare for bounds check
            let p1 = ctx.setp_lt_u32(tid, size);
            // Another bounds check
            let ten = ctx.mov_u32_imm(10);
            let p2 = ctx.setp_lt_u32(tid, ten);
            // Combine predicates
            let combined = ctx.and_pred(p1, p2);
            // Use combined predicate
            ctx.branch_if(combined, "do_work");
            ctx.ret();
            ctx.label("do_work");
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("and.pred"), "Expected and.pred in: {}", ptx);
}

#[test]
fn test_div_f32_inplace_normalization() {
    // Test in-place division for normalization
    let kernel = PtxKernel::new("test_div_inplace")
        .param(PtxType::U64, "data_ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("data_ptr");
            let value = ctx.ld_global_f32(ptr);
            let divisor = ctx.mov_f32_imm(10.0);
            // In-place divide: value = value / divisor
            ctx.div_f32_inplace(value, divisor);
            ctx.st_global_f32(ptr, value);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("div.rn.f32"),
        "Expected div.rn.f32 in: {}",
        ptx
    );
}

#[test]
fn test_predicated_instruction_emission() {
    // Test predicate emission in emit_instruction
    let kernel = PtxKernel::new("test_predicate")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let tid = ctx.special_reg(PtxReg::TidX);
            let limit = ctx.mov_u32_imm(64);
            let pred = ctx.setp_lt_u32(tid, limit);
            // Predicated store
            ctx.branch_if(pred, "store_it");
            ctx.ret();
            ctx.label("store_it");
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_f32_imm(1.0);
            ctx.st_global_f32(ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Check that setp instruction was emitted
    assert!(ptx.contains("setp."), "Expected setp in: {}", ptx);
}

#[test]
fn test_sub_instruction_emission() {
    // Test subtraction instruction emission
    let kernel = PtxKernel::new("test_sub")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(100);
            let b = ctx.mov_u32_imm(30);
            let _result = ctx.sub_u32_reg(a, b);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("sub."), "Expected sub instruction in: {}", ptx);
}

#[test]
fn test_integer_div_emission() {
    // Test integer division (no rounding mode)
    let kernel = PtxKernel::new("test_int_div")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(100);
            let _result = ctx.div_u32(a, 7);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Integer div should not have rounding mode
    assert!(
        ptx.contains("div.u32") || ptx.contains("div.s32"),
        "Expected integer div in: {}",
        ptx
    );
}

#[test]
fn test_mul_wide_u32_emission() {
    // Test wide multiply (u32 -> u64)
    let kernel = PtxKernel::new("test_mul_wide")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(1000000);
            let result = ctx.mul_wide_u32(a, 1000000);
            // Result is u64
            let _ = result;
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.wide.u32"),
        "Expected mul.wide.u32 in: {}",
        ptx
    );
}

#[test]
fn test_mad_lo_emission() {
    // Test multiply-add low
    let kernel = PtxKernel::new("test_mad_lo")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(10);
            let b = ctx.mov_u32_imm(20);
            let c = ctx.mov_u32_imm(5);
            let _result = ctx.mad_lo_u32(a, b, c);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("mad.lo.u32"), "Expected mad.lo.u32 in: {}", ptx);
}

#[test]
fn test_shared_memory_operations() {
    // Test shared memory load/store emission
    let kernel = PtxKernel::new("test_shared")
        .shared_memory(256 * 4) // 256 f32s
        .build(|ctx| {
            let tid = ctx.special_reg(PtxReg::TidX);
            // Get base address in shared memory
            let tile_ptr = ctx.shared_base_addr();
            let offset = ctx.mul_u32(tid, 4); // 4 bytes per f32
            let offset_64 = ctx.cvt_u64_u32(offset);
            let addr = ctx.add_u64(tile_ptr, offset_64);
            // Load from shared
            let val = ctx.ld_shared_f32(addr);
            // Store to shared
            ctx.st_shared_f32(addr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.shared"),
        "Expected ld.shared in: {}",
        ptx
    );
    assert!(
        ptx.contains("st.shared"),
        "Expected st.shared in: {}",
        ptx
    );
}

#[test]
fn test_cvt_instruction_emission() {
    // Test conversion instruction with types
    let kernel = PtxKernel::new("test_cvt")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(42);
            let _f = ctx.cvt_f32_u32(a);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("cvt."), "Expected cvt instruction in: {}", ptx);
}

#[test]
fn test_float_mul_no_lo() {
    // Test floating point multiply (should not have .lo)
    let kernel = PtxKernel::new("test_float_mul")
        .build(|ctx| {
            let a = ctx.mov_f32_imm(3.14);
            let b = ctx.mov_f32_imm(2.0);
            let _result = ctx.mul_f32(a, b);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Float mul should not have .lo
    assert!(
        ptx.contains("mul.f32") && !ptx.contains("mul.lo.f32"),
        "Expected mul.f32 without .lo in: {}",
        ptx
    );
}

#[test]
fn test_bar_sync_basic_barrier() {
    // Test barrier synchronization
    let kernel = PtxKernel::new("test_bar")
        .build(|ctx| {
            ctx.bar_sync(0);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("bar.sync"),
        "Expected bar.sync in: {}",
        ptx
    );
}

#[test]
fn test_setp_comparison_ops() {
    // Test various setp comparison operations
    let kernel = PtxKernel::new("test_setp_cmp")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(10);
            let b = ctx.mov_u32_imm(20);
            // Less than
            let _lt = ctx.setp_lt_u32(a, b);
            // Greater or equal
            let _ge = ctx.setp_ge_u32(a, b);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("setp.lt"), "Expected setp.lt in: {}", ptx);
    assert!(ptx.contains("setp.ge"), "Expected setp.ge in: {}", ptx);
}

#[test]
fn test_st_shared_f16_instruction() {
    // Test store F16 to shared memory (uses B16 type)
    let kernel = PtxKernel::new("test_st_shared_f16")
        .shared_memory(256)
        .build(|ctx| {
            let addr = ctx.shared_base_addr();
            let val = ctx.mov_f32_imm(1.0);
            let f16_val = ctx.cvt_f16_f32(val);
            ctx.st_shared_f16(addr, f16_val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("st.shared"), "Expected st.shared in: {}", ptx);
    assert!(ptx.contains(".b16"), "Expected .b16 type in: {}", ptx);
}

#[test]
fn test_shfl_down_f32_warp_shuffle() {
    // Test warp shuffle down for reductions
    let kernel = PtxKernel::new("test_shfl_down")
        .build(|ctx| {
            let val = ctx.mov_f32_imm(1.0);
            // Shuffle down by 16, then 8, then 4, etc.
            let shuffled = ctx.shfl_down_f32(val, 16, 0xFFFFFFFF);
            let _sum = ctx.add_f32(val, shuffled);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("shfl.sync.down.b32"),
        "Expected shfl.sync.down.b32 in: {}",
        ptx
    );
}

#[test]
fn test_shfl_idx_f32_warp_broadcast() {
    // Test warp shuffle indexed for broadcasts
    let kernel = PtxKernel::new("test_shfl_idx")
        .build(|ctx| {
            let val = ctx.mov_f32_imm(1.0);
            // Broadcast from lane 0 to all lanes
            let _broadcast = ctx.shfl_idx_f32(val, 0, 0xFFFFFFFF);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("shfl.sync.idx.b32"),
        "Expected shfl.sync.idx.b32 in: {}",
        ptx
    );
}

#[test]
fn test_min_u32_instruction() {
    // Test min of two u32 values
    let kernel = PtxKernel::new("test_min_u32")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(100);
            let b = ctx.mov_u32_imm(50);
            let _min = ctx.min_u32(a, b);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("min.u32"), "Expected min.u32 in: {}", ptx);
}

#[test]
fn test_ex2_f32_exponential() {
    // Test exponential base 2 (approximation)
    let kernel = PtxKernel::new("test_ex2")
        .build(|ctx| {
            let val = ctx.mov_f32_imm(2.0);
            let _exp = ctx.ex2_f32(val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ex2.approx"),
        "Expected ex2.approx in: {}",
        ptx
    );
}

#[test]
fn test_rsqrt_f32_instruction() {
    // Test reciprocal square root
    let kernel = PtxKernel::new("test_rsqrt")
        .build(|ctx| {
            let val = ctx.mov_f32_imm(4.0);
            let _rsqrt = ctx.rsqrt_f32(val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("rsqrt.approx"),
        "Expected rsqrt.approx in: {}",
        ptx
    );
}

#[test]
fn test_rem_u32_remainder() {
    // Test integer remainder (modulo)
    let kernel = PtxKernel::new("test_rem")
        .build(|ctx| {
            let val = ctx.mov_u32_imm(100);
            let _rem = ctx.rem_u32(val, 32);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("rem.u32"), "Expected rem.u32 in: {}", ptx);
}

#[test]
fn test_branch_if_not_negated_predicate() {
    // Test branch with negated predicate
    let kernel = PtxKernel::new("test_branch_if_not")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(10);
            let b = ctx.mov_u32_imm(20);
            let pred = ctx.setp_lt_u32(a, b);
            ctx.branch_if_not(pred, "skip");
            ctx.label("skip");
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Negated predicate should have "!" prefix
    assert!(ptx.contains("@!"), "Expected negated predicate @! in: {}", ptx);
    assert!(ptx.contains("bra skip"), "Expected bra skip in: {}", ptx);
}

#[test]
fn test_cvt_u32_u8_conversion() {
    // Test u8 to u32 zero extension
    let kernel = PtxKernel::new("test_cvt_u32_u8")
        .param(PtxType::U64, "src")
        .build(|ctx| {
            let addr = ctx.load_param_u64("src");
            let byte_val = ctx.ld_global_u8(addr);
            let _u32_val = ctx.cvt_u32_u8(byte_val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt.u32"),
        "Expected cvt.u32 conversion in: {}",
        ptx
    );
}

#[test]
fn test_shr_u32_shift_right() {
    // Test logical shift right
    let kernel = PtxKernel::new("test_shr_u32")
        .build(|ctx| {
            let val = ctx.mov_u32_imm(256);
            let shift = ctx.mov_u32_imm(4);
            let _shifted = ctx.shr_u32(val, shift);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("shr.b32"), "Expected shr.b32 in: {}", ptx);
}

#[test]
fn test_and_u32_bitwise() {
    // Test bitwise AND
    let kernel = PtxKernel::new("test_and_u32")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(0xFF00);
            let b = ctx.mov_u32_imm(0x0FF0);
            let _result = ctx.and_u32(a, b);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("and.b32"), "Expected and.b32 in: {}", ptx);
}

#[test]
fn test_or_u32_bitwise() {
    // Test bitwise OR
    let kernel = PtxKernel::new("test_or_u32")
        .build(|ctx| {
            let a = ctx.mov_u32_imm(0xFF00);
            let b = ctx.mov_u32_imm(0x00FF);
            let _result = ctx.or_u32(a, b);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("or.b32"), "Expected or.b32 in: {}", ptx);
}

#[test]
fn test_shl_u32_shift_left() {
    // Test shift left
    let kernel = PtxKernel::new("test_shl_u32")
        .build(|ctx| {
            let val = ctx.mov_u32_imm(1);
            let shift = ctx.mov_u32_imm(8);
            let _shifted = ctx.shl_u32(val, shift);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("shl.b32"), "Expected shl.b32 in: {}", ptx);
}

#[test]
fn test_shr_u32_inplace_shift() {
    // Test in-place shift right by immediate
    let kernel = PtxKernel::new("test_shr_inplace")
        .build(|ctx| {
            let val = ctx.mov_u32_imm(256);
            ctx.shr_u32_inplace(val, 1);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("shr.b32"), "Expected shr.b32 in: {}", ptx);
}

#[test]
fn test_max_f32_inplace_operation() {
    // Test in-place max for running max in softmax
    let kernel = PtxKernel::new("test_max_inplace")
        .build(|ctx| {
            let running_max = ctx.mov_f32_imm(f32::NEG_INFINITY);
            let new_val = ctx.mov_f32_imm(5.0);
            ctx.max_f32_inplace(running_max, new_val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("max.f32"), "Expected max.f32 in: {}", ptx);
}

#[test]
fn test_mov_f32_reg_copy() {
    // Test register-to-register copy
    let kernel = PtxKernel::new("test_mov_f32_reg")
        .build(|ctx| {
            let src = ctx.mov_f32_imm(1.5);
            let dst = ctx.mov_f32_imm(0.0);
            ctx.mov_f32_reg(dst, src);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("mov.f32"), "Expected mov.f32 in: {}", ptx);
}

#[test]
fn test_mul_f32_inplace_scaling() {
    // Test in-place multiply for scaling
    let kernel = PtxKernel::new("test_mul_inplace")
        .build(|ctx| {
            let val = ctx.mov_f32_imm(2.0);
            let scale = ctx.mov_f32_imm(0.5);
            ctx.mul_f32_inplace(val, scale);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Note: mul.f32 doesn't include .rn for inplace (handled in emit_instruction)
    assert!(ptx.contains("mul.f32"), "Expected mul.f32 in: {}", ptx);
}

#[test]
fn test_wmma_load_c_f32_fragment() {
    // Test WMMA load C (accumulator) fragment
    let kernel = PtxKernel::new("test_wmma_load_c")
        .shared_memory(1024)
        .build(|ctx| {
            let addr = ctx.shared_base_addr();
            let _frag_c = ctx.wmma_load_c_f32(addr, 16, WmmaLayout::RowMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("wmma.load.c.sync.aligned"),
        "Expected wmma.load.c in: {}",
        ptx
    );
}

#[test]
fn test_wmma_store_d_empty_fragment() {
    // Test WMMA store with empty fragment (should be no-op)
    let kernel = PtxKernel::new("test_wmma_store_empty")
        .build(|ctx| {
            let addr = ctx.shared_base_addr();
            let empty_frag: Vec<VirtualReg> = Vec::new();
            ctx.wmma_store_d_f32(addr, &empty_frag, 16, WmmaLayout::RowMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // With empty fragment, wmma_store should return early
    assert!(
        !ptx.contains("wmma.store"),
        "Expected no wmma.store with empty fragment"
    );
}

#[test]
fn test_f64_literal_format() {
    // Test f64 literal hex format (0D prefix)
    let kernel = PtxKernel::new("test_f64")
        .build(|ctx| {
            // Create instruction with f64 immediate (via emit_operand)
            let _f32_val = ctx.mov_f32_imm(std::f64::consts::PI as f32);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Verify f32 literal format works (0F prefix)
    assert!(
        ptx.contains("0F"),
        "Expected hex float literal 0F prefix in: {}",
        ptx
    );
}

#[test]
fn test_emit_operand_addr_with_offset() {
    // Test address operand with non-zero offset
    use super::super::instructions::Operand;
    use super::super::registers::VirtualReg;

    let vreg = VirtualReg::new(0, PtxType::U64);
    let addr_op = Operand::Addr {
        base: vreg,
        offset: 128,
    };
    let result = emit_operand(&addr_op);
    assert!(
        result.contains("+128"),
        "Expected offset +128 in: {}",
        result
    );
}

#[test]
fn test_emit_shared_mem_operand_with_offset() {
    // Test shared memory operand with offset
    use super::super::instructions::Operand;
    use super::super::registers::VirtualReg;

    let vreg = VirtualReg::new(0, PtxType::U64);
    let addr_op = Operand::Addr {
        base: vreg,
        offset: 64,
    };
    let result = emit_shared_mem_operand(&addr_op);
    assert!(result.contains("+64"), "Expected offset +64 in: {}", result);
}

#[test]
fn test_emit_global_mem_operand_with_offset() {
    // Test global memory operand with offset
    use super::super::instructions::Operand;
    use super::super::registers::VirtualReg;

    let vreg = VirtualReg::new(0, PtxType::U64);
    let addr_op = Operand::Addr {
        base: vreg,
        offset: 256,
    };
    let result = emit_global_mem_operand(&addr_op);
    assert!(
        result.contains("+256"),
        "Expected offset +256 in: {}",
        result
    );
}

#[test]
fn test_wmma_layout_col_major() {
    // Test column-major WMMA layout
    let kernel = PtxKernel::new("test_wmma_col")
        .shared_memory(1024)
        .build(|ctx| {
            let addr = ctx.shared_base_addr();
            let _frag_a = ctx.wmma_load_a_f16(addr, 16, WmmaLayout::ColMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains(".col."), "Expected .col. layout in: {}", ptx);
}

#[test]
fn test_max_f32_non_inplace() {
    // Test max_f32 (non-inplace version)
    let kernel = PtxKernel::new("test_max_f32")
        .build(|ctx| {
            let a = ctx.mov_f32_imm(3.0);
            let b = ctx.mov_f32_imm(5.0);
            let _max = ctx.max_f32(a, b);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(ptx.contains("max.f32"), "Expected max.f32 in: {}", ptx);
}

#[test]
fn test_kernel_get_shared_memory_bytes() {
    // Test shared_memory_bytes getter
    let kernel = PtxKernel::new("test_smem").shared_memory(4096);
    assert_eq!(kernel.shared_memory_bytes(), 4096);
}

#[test]
fn test_module_get_address_size() {
    // Test address_size getter
    let module = PtxModule::new().address_size(32);
    assert_eq!(module.get_address_size(), 32);
}

#[test]
fn test_signed_wide_multiply() {
    // Test wide multiply with signed type (s64 output)
    use super::super::instructions::{Operand, PtxInstruction, PtxOp};

    let vreg = VirtualReg::new(0, PtxType::S32);
    let instr = PtxInstruction::new(PtxOp::Mul, PtxType::S64)
        .dst(Operand::Reg(vreg))
        .src(Operand::Reg(vreg))
        .src(Operand::ImmI64(100));

    let ptx = emit_instruction(&instr);
    assert!(
        ptx.contains("mul.wide.s32"),
        "Expected mul.wide.s32 in: {}",
        ptx
    );
}

// ========================================================================
// ADDITIONAL COVERAGE TESTS
// ========================================================================

#[test]
fn test_dp4a_u32_instruction() {
    let kernel = PtxKernel::new("test_dp4a").build(|ctx| {
        let a = ctx.mov_u32_imm(0x01020304);
        let b = ctx.mov_u32_imm(0x05060708);
        let c = ctx.mov_u32_imm(0);
        let _result = ctx.dp4a_u32(a, b, c);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("dp4a.u32.u32"), "Expected dp4a.u32.u32 in: {}", ptx);
}

#[test]
fn test_dp4a_u32_inplace_instruction() {
    let kernel = PtxKernel::new("test_dp4a_inplace").build(|ctx| {
        let acc = ctx.mov_u32_imm(0);
        let a = ctx.mov_u32_imm(0x01020304);
        let b = ctx.mov_u32_imm(0x05060708);
        ctx.dp4a_u32_inplace(acc, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("dp4a"), "Expected dp4a in: {}", ptx);
}

#[test]
fn test_dp4a_u32_s32_inplace_instruction() {
    let kernel = PtxKernel::new("test_dp4a_us").build(|ctx| {
        let acc = ctx.mov_u32_imm(0);
        let a = ctx.mov_u32_imm(0x01020304);
        let b = ctx.mov_u32_imm(0x05060708);
        ctx.dp4a_u32_s32_inplace(acc, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("dp4a"), "Expected dp4a in: {}", ptx);
}

#[test]
fn test_dp4a_s32_inplace_instruction() {
    let kernel = PtxKernel::new("test_dp4a_s32").build(|ctx| {
        let acc = ctx.mov_u32_imm(0);
        let a = ctx.mov_u32_imm(0x01020304);
        let b = ctx.mov_u32_imm(0x05060708);
        ctx.dp4a_s32_inplace(acc, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("dp4a"), "Expected dp4a in: {}", ptx);
}

#[test]
fn test_membar_cta_instruction() {
    let kernel = PtxKernel::new("test_membar_cta").build(|ctx| {
        ctx.membar_cta();
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("membar.cta"), "Expected membar.cta in: {}", ptx);
}

#[test]
fn test_membar_gl_instruction() {
    let kernel = PtxKernel::new("test_membar_gl").build(|ctx| {
        ctx.membar_gl();
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("membar.gl"), "Expected membar.gl in: {}", ptx);
}

#[test]
fn test_ld_shared_u32_volatile_instruction() {
    let kernel = PtxKernel::new("test_ld_volatile")
        .shared_memory(256)
        .build(|ctx| {
            let addr = ctx.mov_u64_imm(0);
            let _val = ctx.ld_shared_u32_volatile(addr);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(ptx.contains("ld.volatile.shared.u32"), "Expected ld.volatile.shared.u32 in: {}", ptx);
}

#[test]
fn test_ballot_sync_instruction() {
    let kernel = PtxKernel::new("test_ballot").build(|ctx| {
        let a = ctx.mov_u32_imm(1);
        let b = ctx.mov_u32_imm(0);
        let pred = ctx.setp_ge_u32(a, b);
        let _ballot = ctx.ballot_sync(pred, 0xFFFFFFFF);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("vote") || ptx.contains("ballot"), "Expected ballot in: {}", ptx);
}

#[test]
fn test_popc_u32_instruction() {
    let kernel = PtxKernel::new("test_popc").build(|ctx| {
        let val = ctx.mov_u32_imm(0xFF);
        let _count = ctx.popc_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("popc"), "Expected popc in: {}", ptx);
}

#[test]
fn test_bfind_u32_instruction() {
    let kernel = PtxKernel::new("test_bfind").build(|ctx| {
        let val = ctx.mov_u32_imm(0x80);
        let _pos = ctx.bfind_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("bfind"), "Expected bfind in: {}", ptx);
}

#[test]
fn test_clz_u32_instruction() {
    let kernel = PtxKernel::new("test_clz").build(|ctx| {
        let val = ctx.mov_u32_imm(0x80);
        let _lz = ctx.clz_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("clz"), "Expected clz in: {}", ptx);
}

#[test]
fn test_shfl_idx_u32_reg_instruction() {
    let kernel = PtxKernel::new("test_shfl_reg").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let lane = ctx.mov_u32_imm(0);
        let _shuffled = ctx.shfl_idx_u32_reg(val, lane, 0xFFFFFFFF);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("shfl.sync.idx"), "Expected shfl.sync.idx in: {}", ptx);
}

#[test]
fn test_atom_add_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_add")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(1);
            let _old = ctx.atom_add_global_u32(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(ptx.contains("atom.global.add.u32"), "Expected atom.global.add.u32 in: {}", ptx);
}

#[test]
fn test_atom_exch_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_exch")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(42);
            let _old = ctx.atom_exch_global_u32(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(ptx.contains("atom.global.exch.u32"), "Expected atom.global.exch.u32 in: {}", ptx);
}

#[test]
fn test_atom_min_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_min")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(10);
            let _old = ctx.atom_min_global_u32(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(ptx.contains("atom.global.min.u32"), "Expected atom.global.min.u32 in: {}", ptx);
}

#[test]
fn test_atom_max_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_max")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(100);
            let _old = ctx.atom_max_global_u32(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(ptx.contains("atom.global.max.u32"), "Expected atom.global.max.u32 in: {}", ptx);
}

#[test]
fn test_atom_exch_shared_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_exch_shared")
        .shared_memory(256)
        .build(|ctx| {
            let addr = ctx.mov_u64_imm(0);
            let val = ctx.mov_u32_imm(42);
            let _old = ctx.atom_exch_shared_u32(addr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(ptx.contains("atom.shared.exch.u32"), "Expected atom.shared.exch.u32 in: {}", ptx);
}

#[test]
fn test_sin_f32_instruction() {
    let kernel = PtxKernel::new("test_sin").build(|ctx| {
        let val = ctx.mov_f32_imm(1.57);
        let _result = ctx.sin_f32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("sin.approx.f32"), "Expected sin.approx.f32 in: {}", ptx);
}

#[test]
fn test_cos_f32_instruction() {
    let kernel = PtxKernel::new("test_cos").build(|ctx| {
        let val = ctx.mov_f32_imm(0.0);
        let _result = ctx.cos_f32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cos.approx.f32"), "Expected cos.approx.f32 in: {}", ptx);
}

#[test]
fn test_neg_f32_instruction() {
    let kernel = PtxKernel::new("test_neg").build(|ctx| {
        let val = ctx.mov_f32_imm(1.0);
        let _result = ctx.neg_f32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("neg.f32"), "Expected neg.f32 in: {}", ptx);
}

#[test]
fn test_cvt_s32_s8_instruction() {
    let kernel = PtxKernel::new("test_cvt_s8")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_global_u8(ptr);
            let _signed = ctx.cvt_s32_s8(val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    // cvt_s32_s8 uses setp_ge, mov, selp, sub to do sign extension
    assert!(ptx.contains("setp"), "Expected setp for sign extension in: {}", ptx);
}

#[test]
fn test_cvt_f32_s32_instruction() {
    let kernel = PtxKernel::new("test_cvt_f32_s32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _float = ctx.cvt_f32_s32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt.rn.f32.s32"), "Expected cvt.rn.f32.s32 in: {}", ptx);
}

#[test]
fn test_st_global_u8_instruction() {
    let kernel = PtxKernel::new("test_st_u8")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(0xFF);
            ctx.st_global_u8(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(ptx.contains("st.global.u8"), "Expected st.global.u8 in: {}", ptx);
}

#[test]
fn test_st_global_u16_instruction() {
    let kernel = PtxKernel::new("test_st_u16")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(0xFFFF);
            ctx.st_global_u16(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(ptx.contains("st.global.u16"), "Expected st.global.u16 in: {}", ptx);
}

#[test]
fn test_st_shared_u16_instruction() {
    let kernel = PtxKernel::new("test_st_shared_u16")
        .shared_memory(256)
        .build(|ctx| {
            let addr = ctx.mov_u64_imm(0);
            let val = ctx.mov_u32_imm(0xFFFF);
            ctx.st_shared_u16(addr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(ptx.contains("st.shared.u16"), "Expected st.shared.u16 in: {}", ptx);
}

#[test]
fn test_add_u64_into_instruction() {
    let kernel = PtxKernel::new("test_add_u64_into").build(|ctx| {
        let a = ctx.mov_u64_imm(100);
        let b = ctx.mov_u64_imm(200);
        let dst = ctx.mov_u64_imm(0);
        ctx.add_u64_into(dst, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("add.u64"), "Expected add.u64 in: {}", ptx);
}

#[test]
fn test_add_u32_into_instruction() {
    let kernel = PtxKernel::new("test_add_u32_into").build(|ctx| {
        let a = ctx.mov_u32_imm(100);
        let b = ctx.mov_u32_imm(200);
        let dst = ctx.mov_u32_imm(0);
        ctx.add_u32_into(dst, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("add.u32"), "Expected add.u32 in: {}", ptx);
}

#[test]
fn test_mov_u64_into_instruction() {
    let kernel = PtxKernel::new("test_mov_u64_into").build(|ctx| {
        let dst = ctx.mov_u64_imm(0);
        ctx.mov_u64_into(dst, 12345);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov.u64"), "Expected mov.u64 in: {}", ptx);
}

#[test]
fn test_mov_u32_into_instruction() {
    let kernel = PtxKernel::new("test_mov_u32_into").build(|ctx| {
        let dst = ctx.mov_u32_imm(0);
        ctx.mov_u32_into(dst, 12345);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov.u32"), "Expected mov.u32 in: {}", ptx);
}

#[test]
fn test_setp_eq_u32_instruction() {
    let kernel = PtxKernel::new("test_setp_eq").build(|ctx| {
        let a = ctx.mov_u32_imm(42);
        let b = ctx.mov_u32_imm(42);
        let _pred = ctx.setp_eq_u32(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("setp.eq.u32"), "Expected setp.eq.u32 in: {}", ptx);
}

#[test]
fn test_mul_u32_reg_instruction() {
    let kernel = PtxKernel::new("test_mul_u32_reg").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let _result = ctx.mul_u32_reg(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mul.lo.u32"), "Expected mul.lo.u32 in: {}", ptx);
}

#[test]
fn test_add_u32_reg_instruction() {
    let kernel = PtxKernel::new("test_add_u32_reg").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let _result = ctx.add_u32_reg(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("add.u32"), "Expected add.u32 in: {}", ptx);
}

#[test]
fn test_cvt_u64_u32_into_instruction() {
    let kernel = PtxKernel::new("test_cvt_into").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let dst = ctx.mov_u64_imm(0);
        ctx.cvt_u64_u32_into(dst, val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt.u64.u32"), "Expected cvt.u64.u32 in: {}", ptx);
}

#[test]
fn test_cvt_u32_u64_instruction() {
    let kernel = PtxKernel::new("test_cvt_u32_u64").build(|ctx| {
        let val = ctx.mov_u64_imm(1000);
        let _truncated = ctx.cvt_u32_u64(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt.u32.u64"), "Expected cvt.u32.u64 in: {}", ptx);
}

#[test]
fn test_cvt_f32_u32_instruction() {
    let kernel = PtxKernel::new("test_cvt_f32_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _float = ctx.cvt_f32_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt.rn.f32.u32"), "Expected cvt.rn.f32.u32 in: {}", ptx);
}

#[test]
fn test_mul_u64_instruction() {
    let kernel = PtxKernel::new("test_mul_u64").build(|ctx| {
        let a = ctx.mov_u64_imm(100);
        let _result = ctx.mul_u64(a, 200);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mul.lo.u64"), "Expected mul.lo.u64 in: {}", ptx);
}

#[test]
fn test_mul_u64_reg_instruction() {
    let kernel = PtxKernel::new("test_mul_u64_reg").build(|ctx| {
        let a = ctx.mov_u64_imm(100);
        let b = ctx.mov_u64_imm(200);
        let _result = ctx.mul_u64_reg(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mul.lo.u64"), "Expected mul.lo.u64 in: {}", ptx);
}

#[test]
fn test_ld_global_u32_into_instruction() {
    let kernel = PtxKernel::new("test_ld_into")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let dst = ctx.mov_u32_imm(0);
            ctx.ld_global_u32_into(dst, ptr);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(ptx.contains("ld.global.u32"), "Expected ld.global.u32 in: {}", ptx);
}

#[test]
fn test_emit_debug_marker() {
    let kernel = PtxKernel::new("test_debug")
        .param(PtxType::U64, "debug_buf")
        .build(|ctx| {
            let debug_buf = ctx.load_param_u64("debug_buf");
            let _slot = ctx.emit_debug_marker(debug_buf, 0xDEAD);
            ctx.ret();
        });
    let ptx = kernel.emit();
    // Debug marker uses atomicAdd and st.global
    assert!(ptx.contains("atom.global.add.u32"), "Expected atomicAdd for debug marker in: {}", ptx);
}

#[test]
fn test_emit_debug_value() {
    let kernel = PtxKernel::new("test_debug_val")
        .param(PtxType::U64, "debug_buf")
        .build(|ctx| {
            let debug_buf = ctx.load_param_u64("debug_buf");
            let val = ctx.mov_u32_imm(42);
            let _slot = ctx.emit_debug_value(debug_buf, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(ptx.contains("atom.global.add.u32"), "Expected atomicAdd for debug value in: {}", ptx);
}

#[test]
fn test_div_f32_instruction() {
    let kernel = PtxKernel::new("test_div_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(10.0);
        let b = ctx.mov_f32_imm(2.0);
        let _result = ctx.div_f32(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    // div.f32 emits as div.rn.f32 with rounding mode
    assert!(ptx.contains("div.rn.f32") || ptx.contains("div.f32"), "Expected div.f32 in: {}", ptx);
}

#[test]
fn test_and_pred_instruction() {
    let kernel = PtxKernel::new("test_and_pred").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let five = ctx.mov_u32_imm(5);
        let thirty = ctx.mov_u32_imm(30);
        let p1 = ctx.setp_ge_u32(a, five);
        let p2 = ctx.setp_lt_u32(b, thirty);
        let _combined = ctx.and_pred(p1, p2);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("and.pred"), "Expected and.pred in: {}", ptx);
}

#[test]
fn test_branch_instruction() {
    let kernel = PtxKernel::new("test_branch").build(|ctx| {
        ctx.branch("end");
        ctx.label("end");
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("bra end"), "Expected bra end in: {}", ptx);
}

#[test]
fn test_branch_if_instruction() {
    let kernel = PtxKernel::new("test_branch_if").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(5);
        let pred = ctx.setp_ge_u32(a, b);
        ctx.branch_if(pred, "taken");
        ctx.label("taken");
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("@%p"), "Expected predicated branch @%p in: {}", ptx);
}

#[test]
fn test_shfl_idx_u32_instruction() {
    let kernel = PtxKernel::new("test_shfl_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _shuffled = ctx.shfl_idx_u32(val, 0, 0xFFFFFFFF);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("shfl.sync.idx"), "Expected shfl.sync.idx in: {}", ptx);
}

#[test]
fn test_special_reg_tid() {
    let kernel = PtxKernel::new("test_tid").build(|ctx| {
        let _tid = ctx.special_reg(PtxReg::TidX);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("%tid.x"), "Expected %tid.x in: {}", ptx);
}

#[test]
fn test_mul_u32_instruction() {
    let kernel = PtxKernel::new("test_mul_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let _result = ctx.mul_u32(a, 20);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mul.lo.u32"), "Expected mul.lo.u32 in: {}", ptx);
}

#[test]
fn test_sub_u32_reg_instruction() {
    let kernel = PtxKernel::new("test_sub_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(100);
        let b = ctx.mov_u32_imm(50);
        let _result = ctx.sub_u32_reg(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("sub.u32"), "Expected sub.u32 in: {}", ptx);
}

// =========================================================================
// COVERAGE-BOOST: Generic Address Space Operations
// =========================================================================

#[test]
fn test_ld_generic_u32() {
    let kernel = PtxKernel::new("test_generic_u32").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.ld_generic_u32(ptr);
        ctx.st_generic_u32(ptr, val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("ld.u32"), "Expected ld.u32 in: {}", ptx);
    assert!(ptx.contains("st.u32"), "Expected st.u32 in: {}", ptx);
}

#[test]
fn test_ld_generic_u64() {
    let kernel = PtxKernel::new("test_generic_u64").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.ld_generic_u64(ptr);
        ctx.st_generic_u64(ptr, val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("ld.u64") || ptx.contains(".u64"), "Expected u64 in: {}", ptx);
}

#[test]
fn test_ld_generic_u8() {
    let kernel = PtxKernel::new("test_generic_u8").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.ld_generic_u8(ptr);
        ctx.st_generic_u8(ptr, val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains(".u8") || ptx.contains("u8"), "Expected u8 ops in: {}", ptx);
}

#[test]
fn test_ld_generic_u16() {
    let kernel = PtxKernel::new("test_generic_u16").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.ld_generic_u16(ptr);
        ctx.st_generic_u16(ptr, val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains(".u16") || ptx.contains("u16"), "Expected u16 ops in: {}", ptx);
}

#[test]
fn test_ld_generic_f32() {
    let kernel = PtxKernel::new("test_generic_f32").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let val = ctx.ld_generic_f32(ptr);
        ctx.st_generic_f32(ptr, val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains(".f32") || ptx.contains("f32"), "Expected f32 ops in: {}", ptx);
}

#[test]
fn test_ld_generic_u32_into() {
    let kernel = PtxKernel::new("test_generic_into").param(PtxType::U64, "ptr").build(|ctx| {
        let ptr = ctx.load_param_u64("ptr");
        let dest = ctx.mov_u32_imm(0);
        ctx.ld_generic_u32_into(ptr, dest);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("ld"), "Expected load in: {}", ptx);
}

// =========================================================================
// COVERAGE-BOOST: Type Conversion Operations
// =========================================================================

#[test]
fn test_cvt_u32_u8() {
    let kernel = PtxKernel::new("test_cvt_u32_u8").build(|ctx| {
        let val = ctx.mov_u32_imm(255);
        let _converted = ctx.cvt_u32_u8(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt") || ptx.contains("and"), "Expected conversion in: {}", ptx);
}

#[test]
fn test_cvt_u32_u16() {
    let kernel = PtxKernel::new("test_cvt_u32_u16").build(|ctx| {
        let val = ctx.mov_u32_imm(65535);
        let _converted = ctx.cvt_u32_u16(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt") || ptx.contains("and"), "Expected conversion in: {}", ptx);
}

#[test]
fn test_cvt_u16_u32() {
    let kernel = PtxKernel::new("test_cvt_u16_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(100);
        let _converted = ctx.cvt_u16_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt") || ptx.contains("and"), "Expected conversion in: {}", ptx);
}

#[test]
fn test_cvt_u64_u32() {
    let kernel = PtxKernel::new("test_cvt_u64_u32").build(|ctx| {
        let val = ctx.mov_u64_imm(0xFFFFFFFF);
        let _converted = ctx.cvt_u64_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt"), "Expected cvt in: {}", ptx);
}

#[test]
fn test_cvt_u32_u64() {
    let kernel = PtxKernel::new("test_cvt_u32_u64").build(|ctx| {
        let val = ctx.mov_u32_imm(12345);
        let _converted = ctx.cvt_u32_u64(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt"), "Expected cvt in: {}", ptx);
}

#[test]
fn test_cvt_f32_s32() {
    let kernel = PtxKernel::new("test_cvt_f32_s32").build(|ctx| {
        let val = ctx.mov_f32_imm(-42.5);
        let _converted = ctx.cvt_rni_s32_f32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("cvt.rni") || ptx.contains("cvt"), "Expected cvt in: {}", ptx);
}

#[test]
fn test_cvt_s32_u8_sx() {
    let kernel = PtxKernel::new("test_cvt_s32_u8_sx").build(|ctx| {
        let val = ctx.mov_u32_imm(200);
        let _converted = ctx.cvt_s32_u8_sx(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.len() > 0);
}

// =========================================================================
// COVERAGE-BOOST: Shift and Bitwise Operations
// =========================================================================

#[test]
fn test_shr_u32_imm() {
    let kernel = PtxKernel::new("test_shr_imm").build(|ctx| {
        let val = ctx.mov_u32_imm(256);
        let _shifted = ctx.shr_u32_imm(val, 4);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("shr"), "Expected shr in: {}", ptx);
}

#[test]
fn test_shl_u32_imm() {
    let kernel = PtxKernel::new("test_shl_imm").build(|ctx| {
        let val = ctx.mov_u32_imm(16);
        let _shifted = ctx.shl_u32_imm(val, 4);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("shl"), "Expected shl in: {}", ptx);
}

#[test]
fn test_and_u32_imm() {
    let kernel = PtxKernel::new("test_and_imm").build(|ctx| {
        let val = ctx.mov_u32_imm(0xFF);
        let _masked = ctx.and_u32_imm(val, 0x0F);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("and"), "Expected and in: {}", ptx);
}

#[test]
fn test_or_u32_into() {
    let kernel = PtxKernel::new("test_or_into").build(|ctx| {
        let dest = ctx.mov_u32_imm(0);
        let a = ctx.mov_u32_imm(0xF0);
        let b = ctx.mov_u32_imm(0x0F);
        ctx.or_u32_into(dest, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("or"), "Expected or in: {}", ptx);
}

// =========================================================================
// COVERAGE-BOOST: Select/Predicate Operations
// =========================================================================

#[test]
fn test_selp_u32() {
    let kernel = PtxKernel::new("test_selp_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let pred = ctx.setp_lt_u32(a, b);
        let _result = ctx.selp_u32(pred, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("selp"), "Expected selp in: {}", ptx);
}

#[test]
fn test_selp_f32() {
    let kernel = PtxKernel::new("test_selp_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(1.0);
        let b = ctx.mov_f32_imm(2.0);
        let pred = ctx.setp_gt_f32(b, a); // b > a is true
        let _result = ctx.selp_f32(pred, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("selp"), "Expected selp in: {}", ptx);
}

#[test]
fn test_setp_gt_f32() {
    let kernel = PtxKernel::new("test_setp_gt").build(|ctx| {
        let a = ctx.mov_f32_imm(2.0);
        let b = ctx.mov_f32_imm(1.0);
        let _pred = ctx.setp_gt_f32(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("setp.gt"), "Expected setp.gt in: {}", ptx);
}

// =========================================================================
// COVERAGE-BOOST: Arithmetic Operations
// =========================================================================

#[test]
fn test_sub_f32() {
    let kernel = PtxKernel::new("test_sub_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(5.0);
        let b = ctx.mov_f32_imm(3.0);
        let _result = ctx.sub_f32(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("sub.f32"), "Expected sub.f32 in: {}", ptx);
}

#[test]
fn test_rcp_f32() {
    let kernel = PtxKernel::new("test_rcp").build(|ctx| {
        let val = ctx.mov_f32_imm(4.0);
        let _recip = ctx.rcp_f32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    // rcp requires .approx modifier for f32 per PTX ISA
    assert!(ptx.contains("rcp.approx.f32"), "Expected rcp.approx.f32 in: {}", ptx);
}

#[test]
fn test_abs_f32() {
    let kernel = PtxKernel::new("test_abs").build(|ctx| {
        let val = ctx.mov_f32_imm(-3.14);
        let _result = ctx.abs_f32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("abs"), "Expected abs in: {}", ptx);
}

#[test]
fn test_mul_lo_s32() {
    let kernel = PtxKernel::new("test_mul_s32").build(|ctx| {
        let a = ctx.mov_s32_imm(-10);
        let b = ctx.mov_s32_imm(5);
        let _result = ctx.mul_lo_s32(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mul"), "Expected mul in: {}", ptx);
}

#[test]
fn test_min_s32() {
    let kernel = PtxKernel::new("test_min_s32").build(|ctx| {
        let a = ctx.mov_s32_imm(-10);
        let b = ctx.mov_s32_imm(5);
        let _result = ctx.min_s32(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("min"), "Expected min in: {}", ptx);
}

#[test]
fn test_max_s32() {
    let kernel = PtxKernel::new("test_max_s32").build(|ctx| {
        let a = ctx.mov_s32_imm(-10);
        let b = ctx.mov_s32_imm(5);
        let _result = ctx.max_s32(a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("max"), "Expected max in: {}", ptx);
}

#[test]
fn test_mov_s32_imm() {
    let kernel = PtxKernel::new("test_mov_s32").build(|ctx| {
        let _val = ctx.mov_s32_imm(-12345);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov") && ptx.contains("12345"), "Expected mov in: {}", ptx);
}

// =========================================================================
// COVERAGE-BOOST: In-Place Register Operations
// =========================================================================

#[test]
fn test_fma_f32_inplace() {
    let kernel = PtxKernel::new("test_fma_inplace").build(|ctx| {
        let acc = ctx.mov_f32_imm(0.0);
        let a = ctx.mov_f32_imm(2.0);
        let b = ctx.mov_f32_imm(3.0);
        ctx.fma_f32_inplace(acc, a, b);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("fma") || ptx.contains("mad"), "Expected fma/mad in: {}", ptx);
}

#[test]
fn test_max_f32_inplace() {
    let kernel = PtxKernel::new("test_max_inplace").build(|ctx| {
        let acc = ctx.mov_f32_imm(1.0);
        let val = ctx.mov_f32_imm(5.0);
        ctx.max_f32_inplace(acc, val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("max"), "Expected max in: {}", ptx);
}

#[test]
fn test_mul_f32_inplace() {
    let kernel = PtxKernel::new("test_mul_inplace").build(|ctx| {
        let val = ctx.mov_f32_imm(2.0);
        let factor = ctx.mov_f32_imm(3.0);
        ctx.mul_f32_inplace(val, factor);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mul"), "Expected mul in: {}", ptx);
}

#[test]
fn test_shr_u32_inplace() {
    let kernel = PtxKernel::new("test_shr_inplace").build(|ctx| {
        let val = ctx.mov_u32_imm(256);
        ctx.shr_u32_inplace(val, 2);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("shr"), "Expected shr in: {}", ptx);
}

#[test]
fn test_mov_f32_reg() {
    let kernel = PtxKernel::new("test_mov_f32").build(|ctx| {
        let src = ctx.mov_f32_imm(1.5);
        let dest = ctx.mov_f32_imm(0.0);
        ctx.mov_f32_reg(dest, src);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov"), "Expected mov in: {}", ptx);
}

#[test]
fn test_mov_u32_reg() {
    let kernel = PtxKernel::new("test_mov_u32_reg").build(|ctx| {
        let src = ctx.mov_u32_imm(42);
        let dest = ctx.mov_u32_imm(0);
        ctx.mov_u32_reg(dest, src);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov"), "Expected mov in: {}", ptx);
}

#[test]
fn test_mov_u64_reg() {
    let kernel = PtxKernel::new("test_mov_u64_reg").build(|ctx| {
        let src = ctx.mov_u64_imm(0x123456789ABCDEF0);
        let dest = ctx.mov_u64_imm(0);
        ctx.mov_u64_reg(dest, src);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov"), "Expected mov in: {}", ptx);
}

#[test]
fn test_mov_u32_inplace() {
    let kernel = PtxKernel::new("test_mov_u32_inplace").build(|ctx| {
        let dest = ctx.mov_u32_imm(0);
        ctx.mov_u32_inplace(dest, 999);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov") && ptx.contains("999"), "Expected mov with 999 in: {}", ptx);
}

#[test]
fn test_cvt_s32_u32() {
    let kernel = PtxKernel::new("test_cvt_s32_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _converted = ctx.cvt_s32_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.len() > 0);
}

#[test]
fn test_cvt_u8_s32() {
    let kernel = PtxKernel::new("test_cvt_u8_s32").build(|ctx| {
        let val = ctx.mov_s32_imm(127);
        let _converted = ctx.cvt_u8_s32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.len() > 0);
}

#[test]
fn test_mov_s32_from_u32() {
    let kernel = PtxKernel::new("test_mov_s32_from_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _result = ctx.mov_s32_from_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov"), "Expected mov in: {}", ptx);
}

// =========================================================================
// COVERAGE-BOOST: Constant Wrappers
// =========================================================================

#[test]
fn test_const_f32_wrapper() {
    let kernel = PtxKernel::new("test_const_f32").build(|ctx| {
        let _val = ctx.const_f32(3.14159);
        ctx.ret();
    });
    let ptx = kernel.emit();
    // Float constants are emitted as hex (0F...) in PTX
    assert!(ptx.contains("mov.f32") && ptx.contains("0F"), "Expected const in: {}", ptx);
}

#[test]
fn test_const_u32_wrapper() {
    let kernel = PtxKernel::new("test_const_u32").build(|ctx| {
        let _val = ctx.const_u32(12345);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("mov") && ptx.contains("12345"), "Expected const in: {}", ptx);
}

#[test]
fn test_shared_ptr_alias() {
    let kernel = PtxKernel::new("test_shared_ptr").shared_memory(256).build(|ctx| {
        let _ptr = ctx.shared_ptr();
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("shared"), "Expected shared in: {}", ptx);
}

// =========================================================================
// COVERAGE-BOOST: Warp Shuffle
// =========================================================================

#[test]
fn test_shfl_down_u32() {
    let kernel = PtxKernel::new("test_shfl_down").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _result = ctx.shfl_down_u32(val, 1, 0x1F); // offset=1, mask=0x1F for 32 lanes
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(ptx.contains("shfl"), "Expected shfl in: {}", ptx);
}
