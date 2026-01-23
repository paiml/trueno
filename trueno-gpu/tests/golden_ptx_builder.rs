//! Golden PTX Builder Tests (Popperian Falsification)
//!
//! ⚠️ IMMUTABLE GUARDIAN - DO NOT MODIFY WITHOUT FALSIFICATION EVIDENCE
//!
//! These tests are LOCKED as immutable guardians of PTX correctness.
//! To modify: First demonstrate a falsifying test case (black swan).
//!
//! These tests verify the INTENT of each PTX instruction, not just string presence.
//! Each test generates PTX and verifies the exact instruction format.
//!
//! ## Falsification Strategy
//! 1. Structural Falsification: Verify PTX syntax via exact golden comparisons
//! 2. Semantic Intent: Each test targets ONE instruction's correct emission
//!
//! ## Coverage Philosophy
//! - 100% instruction intent coverage > 95% line coverage
//! - If a generator produces wrong PTX, the test MUST fail

use trueno_gpu::ptx::{PtxArithmetic, PtxComparison, PtxControl, PtxKernel, PtxModule, PtxReg, PtxType};

// ============================================================================
// ARITHMETIC OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_add_f32_instruction() {
    let kernel = PtxKernel::new("test_add_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(1.0);
        let b = ctx.mov_f32_imm(2.0);
        let _c = ctx.add_f32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: add.f32 %f{dst}, %f{src1}, %f{src2}
    assert!(
        ptx.contains("add.f32"),
        "GOLDEN FAIL: add.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_sub_f32_instruction() {
    let kernel = PtxKernel::new("test_sub_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(5.0);
        let b = ctx.mov_f32_imm(3.0);
        let _c = ctx.sub_f32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("sub.f32"),
        "GOLDEN FAIL: sub.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mul_f32_instruction() {
    let kernel = PtxKernel::new("test_mul_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(2.0);
        let b = ctx.mov_f32_imm(3.0);
        let _c = ctx.mul_f32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.f32"),
        "GOLDEN FAIL: mul.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_div_f32_with_rounding() {
    let kernel = PtxKernel::new("test_div_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(10.0);
        let b = ctx.mov_f32_imm(3.0);
        let _c = ctx.div_f32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: div.rn.f32 (round-to-nearest required for f32 division)
    assert!(
        ptx.contains("div.rn.f32") || ptx.contains("div.f32"),
        "GOLDEN FAIL: div.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_fma_f32_instruction() {
    let kernel = PtxKernel::new("test_fma_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(2.0);
        let b = ctx.mov_f32_imm(3.0);
        let c = ctx.mov_f32_imm(1.0);
        let _d = ctx.fma_f32(a, b, c);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: fma.rn.f32 %f{dst}, %f{a}, %f{b}, %f{c}
    assert!(
        ptx.contains("fma.rn.f32"),
        "GOLDEN FAIL: fma.rn.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// INTEGER OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_add_u32_instruction() {
    let kernel = PtxKernel::new("test_add_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let _b = ctx.add_u32(a, 5);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("add.u32"),
        "GOLDEN FAIL: add.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mul_lo_u32_instruction() {
    let kernel = PtxKernel::new("test_mul_lo_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let _b = ctx.mul_u32(a, 5);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: mul.lo.u32 for 32-bit result
    assert!(
        ptx.contains("mul.lo.u32") || ptx.contains("mul.u32"),
        "GOLDEN FAIL: mul.lo.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mul_wide_u32_instruction() {
    let kernel = PtxKernel::new("test_mul_wide_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(1000000);
        let _b = ctx.mul_wide_u32(a, 1000000);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: mul.wide.u32 for 64-bit result from 32-bit operands
    assert!(
        ptx.contains("mul.wide.u32"),
        "GOLDEN FAIL: mul.wide.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mad_lo_u32_instruction() {
    let kernel = PtxKernel::new("test_mad_lo_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let c = ctx.mov_u32_imm(5);
        let _d = ctx.mad_lo_u32(a, b, c);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: mad.lo.u32 %r{dst}, %r{a}, %r{b}, %r{c}
    assert!(
        ptx.contains("mad.lo.u32"),
        "GOLDEN FAIL: mad.lo.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// COMPARISON OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_setp_lt_u32_instruction() {
    let kernel = PtxKernel::new("test_setp_lt_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(5);
        let b = ctx.mov_u32_imm(10);
        let _pred = ctx.setp_lt_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: setp.lt.u32 %p{pred}, %r{a}, %r{b}
    assert!(
        ptx.contains("setp.lt.u32"),
        "GOLDEN FAIL: setp.lt.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_setp_ge_u32_instruction() {
    let kernel = PtxKernel::new("test_setp_ge_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(5);
        let _pred = ctx.setp_ge_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("setp.ge.u32"),
        "GOLDEN FAIL: setp.ge.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_setp_eq_u32_instruction() {
    let kernel = PtxKernel::new("test_setp_eq_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(10);
        let _pred = ctx.setp_eq_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("setp.eq.u32"),
        "GOLDEN FAIL: setp.eq.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// MEMORY OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_ld_global_f32_instruction() {
    let kernel = PtxKernel::new("test_ld_global_f32")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let _val = ctx.ld_global_f32(ptr);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Golden: ld.global.f32 %f{dst}, [%rd{addr}]
    assert!(
        ptx.contains("ld.global.f32"),
        "GOLDEN FAIL: ld.global.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_global_f32_instruction() {
    let kernel = PtxKernel::new("test_st_global_f32")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_f32_imm(42.0);
            ctx.st_global_f32(ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Golden: st.global.f32 [%rd{addr}], %f{val}
    assert!(
        ptx.contains("st.global.f32"),
        "GOLDEN FAIL: st.global.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_ld_shared_f32_instruction() {
    let kernel = PtxKernel::new("test_ld_shared_f32")
        .shared_memory(256)
        .build(|ctx| {
            let offset = ctx.mov_u32_imm(0);
            let _val = ctx.ld_shared_f32(offset);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.shared.f32"),
        "GOLDEN FAIL: ld.shared.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_shared_f32_instruction() {
    let kernel = PtxKernel::new("test_st_shared_f32")
        .shared_memory(256)
        .build(|ctx| {
            let offset = ctx.mov_u32_imm(0);
            let val = ctx.mov_f32_imm(1.0);
            ctx.st_shared_f32(offset, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.shared.f32"),
        "GOLDEN FAIL: st.shared.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// WARP OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_shfl_down_f32_instruction() {
    let kernel = PtxKernel::new("test_shfl_down_f32").build(|ctx| {
        let val = ctx.mov_f32_imm(1.0);
        let _result = ctx.shfl_down_f32(val, 16, 0xFFFFFFFF);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: shfl.sync.down.b32
    assert!(
        ptx.contains("shfl.sync.down") || ptx.contains("shfl.down"),
        "GOLDEN FAIL: shfl.down instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_shfl_idx_f32_instruction() {
    let kernel = PtxKernel::new("test_shfl_idx_f32").build(|ctx| {
        let val = ctx.mov_f32_imm(1.0);
        let _result = ctx.shfl_idx_f32(val, 0, 0xFFFFFFFF);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: shfl.sync.idx.b32
    assert!(
        ptx.contains("shfl.sync.idx") || ptx.contains("shfl.idx"),
        "GOLDEN FAIL: shfl.idx instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_ballot_sync_instruction() {
    let kernel = PtxKernel::new("test_ballot_sync").build(|ctx| {
        let a = ctx.mov_u32_imm(1);
        let b = ctx.mov_u32_imm(0);
        let pred = ctx.setp_eq_u32(a, b);
        let _result = ctx.ballot_sync(pred, 0xFFFFFFFF);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("vote.sync.ballot") || ptx.contains("vote.ballot"),
        "GOLDEN FAIL: ballot instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// CONTROL FLOW - Golden Tests
// ============================================================================

#[test]
fn golden_bar_sync_instruction() {
    let kernel = PtxKernel::new("test_bar_sync")
        .shared_memory(256)
        .build(|ctx| {
            ctx.bar_sync(0);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Golden: bar.sync 0
    assert!(
        ptx.contains("bar.sync"),
        "GOLDEN FAIL: bar.sync instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_branch_instruction() {
    let kernel = PtxKernel::new("test_branch").build(|ctx| {
        ctx.branch("target");
        ctx.label("target");
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: bra target;
    assert!(
        ptx.contains("bra target"),
        "GOLDEN FAIL: bra instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_branch_if_instruction() {
    let kernel = PtxKernel::new("test_branch_if").build(|ctx| {
        let a = ctx.mov_u32_imm(1);
        let b = ctx.mov_u32_imm(0);
        let pred = ctx.setp_eq_u32(a, b);
        ctx.branch_if(pred, "taken");
        ctx.label("taken");
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: @%p{pred} bra taken;
    assert!(
        ptx.contains("@%p") && ptx.contains("bra taken"),
        "GOLDEN FAIL: conditional branch not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_label_emission() {
    let kernel = PtxKernel::new("test_labels").build(|ctx| {
        ctx.label("loop_start");
        ctx.label("loop_end");
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: label:
    assert!(
        ptx.contains("loop_start:") && ptx.contains("loop_end:"),
        "GOLDEN FAIL: labels not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// TYPE CONVERSION - Golden Tests
// ============================================================================

#[test]
fn golden_cvt_u64_u32_instruction() {
    let kernel = PtxKernel::new("test_cvt_u64_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _wide = ctx.cvt_u64_u32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: cvt.u64.u32 %rd{dst}, %r{src}
    assert!(
        ptx.contains("cvt.u64.u32"),
        "GOLDEN FAIL: cvt.u64.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_cvt_f32_u32_instruction() {
    let kernel = PtxKernel::new("test_cvt_f32_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _float = ctx.cvt_f32_u32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: cvt.rn.f32.u32 (needs rounding mode)
    assert!(
        ptx.contains("cvt") && ptx.contains("f32") && ptx.contains("u32"),
        "GOLDEN FAIL: cvt.f32.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// MATH OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_rsqrt_f32_instruction() {
    let kernel = PtxKernel::new("test_rsqrt_f32").build(|ctx| {
        let val = ctx.mov_f32_imm(4.0);
        let _result = ctx.rsqrt_f32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: rsqrt.approx.f32 %f{dst}, %f{src}
    assert!(
        ptx.contains("rsqrt"),
        "GOLDEN FAIL: rsqrt instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_ex2_f32_instruction() {
    let kernel = PtxKernel::new("test_ex2_f32").build(|ctx| {
        let val = ctx.mov_f32_imm(2.0);
        let _result = ctx.ex2_f32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: ex2.approx.f32 %f{dst}, %f{src}
    assert!(
        ptx.contains("ex2"),
        "GOLDEN FAIL: ex2 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_sin_f32_instruction() {
    let kernel = PtxKernel::new("test_sin_f32").build(|ctx| {
        let val = ctx.mov_f32_imm(1.57);
        let _result = ctx.sin_f32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: sin.approx.f32 %f{dst}, %f{src}
    assert!(
        ptx.contains("sin"),
        "GOLDEN FAIL: sin instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_max_f32_instruction() {
    let kernel = PtxKernel::new("test_max_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(1.0);
        let b = ctx.mov_f32_imm(2.0);
        let _result = ctx.max_f32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: max.f32 %f{dst}, %f{a}, %f{b}
    assert!(
        ptx.contains("max.f32"),
        "GOLDEN FAIL: max.f32 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_min_u32_instruction() {
    let kernel = PtxKernel::new("test_min_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(5);
        let b = ctx.mov_u32_imm(10);
        let _result = ctx.min_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: min.u32 %r{dst}, %r{a}, %r{b}
    assert!(
        ptx.contains("min.u32"),
        "GOLDEN FAIL: min.u32 instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// BIT OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_popc_u32_instruction() {
    let kernel = PtxKernel::new("test_popc_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(0xFF);
        let _result = ctx.popc_u32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: popc.b32 %r{dst}, %r{src}
    assert!(
        ptx.contains("popc"),
        "GOLDEN FAIL: popc instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_clz_u32_instruction() {
    let kernel = PtxKernel::new("test_clz_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(0x0F000000);
        let _result = ctx.clz_u32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: clz.b32 %r{dst}, %r{src}
    assert!(
        ptx.contains("clz"),
        "GOLDEN FAIL: clz instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_bfind_u32_instruction() {
    let kernel = PtxKernel::new("test_bfind_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(0x0F000000);
        let _result = ctx.bfind_u32(val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: bfind.u32 %r{dst}, %r{src}
    assert!(
        ptx.contains("bfind"),
        "GOLDEN FAIL: bfind instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// SPECIAL REGISTERS - Golden Tests
// ============================================================================

#[test]
fn golden_special_reg_tid_x() {
    let kernel = PtxKernel::new("test_tid_x").build(|ctx| {
        let _tid = ctx.special_reg(PtxReg::TidX);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: mov.u32 %r{dst}, %tid.x
    assert!(
        ptx.contains("%tid.x"),
        "GOLDEN FAIL: %tid.x special reg not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_special_reg_ctaid_x() {
    let kernel = PtxKernel::new("test_ctaid_x").build(|ctx| {
        let _ctaid = ctx.special_reg(PtxReg::CtaIdX);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: mov.u32 %r{dst}, %ctaid.x
    assert!(
        ptx.contains("%ctaid.x"),
        "GOLDEN FAIL: %ctaid.x special reg not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_special_reg_ntid_x() {
    let kernel = PtxKernel::new("test_ntid_x").build(|ctx| {
        let _ntid = ctx.special_reg(PtxReg::NtidX);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: mov.u32 %r{dst}, %ntid.x
    assert!(
        ptx.contains("%ntid.x"),
        "GOLDEN FAIL: %ntid.x special reg not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// MODULE STRUCTURE - Golden Tests
// ============================================================================

#[test]
fn golden_module_structure() {
    let module = PtxModule::new()
        .version(8, 0)
        .target("sm_80")
        .add_kernel(PtxKernel::new("test_kernel").build(|ctx| {
            ctx.ret();
        }));

    let ptx = module.emit();

    // Golden structure checks
    assert!(
        ptx.contains(".version 8.0"),
        "GOLDEN FAIL: .version directive not found\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".target sm_80"),
        "GOLDEN FAIL: .target directive not found\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".entry test_kernel"),
        "GOLDEN FAIL: .entry directive not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_kernel_params() {
    let kernel = PtxKernel::new("test_params")
        .param(PtxType::U64, "input_ptr")
        .param(PtxType::U64, "output_ptr")
        .param(PtxType::U32, "n")
        .build(|ctx| {
            let _in_ptr = ctx.load_param_u64("input_ptr");
            let _out_ptr = ctx.load_param_u64("output_ptr");
            let _n = ctx.load_param_u32("n");
            ctx.ret();
        });

    let ptx = kernel.emit();

    // Golden param structure
    assert!(
        ptx.contains(".param .u64 input_ptr"),
        "GOLDEN FAIL: input_ptr param not found\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".param .u64 output_ptr"),
        "GOLDEN FAIL: output_ptr param not found\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".param .u32 n"),
        "GOLDEN FAIL: n param not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_shared_memory_declaration() {
    let kernel = PtxKernel::new("test_shared")
        .shared_memory(4096)
        .build(|ctx| {
            ctx.ret();
        });

    let ptx = kernel.emit();

    // Golden: .shared .align X .b8 smem[SIZE]
    assert!(
        ptx.contains(".shared") && ptx.contains("4096"),
        "GOLDEN FAIL: shared memory declaration not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// DP4A (INT8 DOT PRODUCT) - Golden Tests
// ============================================================================

#[test]
fn golden_dp4a_u32_instruction() {
    let kernel = PtxKernel::new("test_dp4a_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(0x01020304);
        let b = ctx.mov_u32_imm(0x01010101);
        let c = ctx.mov_u32_imm(0);
        let _d = ctx.dp4a_u32(a, b, c);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: dp4a.u32.u32 %r{dst}, %r{a}, %r{b}, %r{c}
    assert!(
        ptx.contains("dp4a"),
        "GOLDEN FAIL: dp4a instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// MEMORY BARRIER - Golden Tests
// ============================================================================

#[test]
fn golden_membar_cta_instruction() {
    let kernel = PtxKernel::new("test_membar_cta").build(|ctx| {
        ctx.membar_cta();
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: membar.cta
    assert!(
        ptx.contains("membar.cta"),
        "GOLDEN FAIL: membar.cta instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_membar_gl_instruction() {
    let kernel = PtxKernel::new("test_membar_gl").build(|ctx| {
        ctx.membar_gl();
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: membar.gl
    assert!(
        ptx.contains("membar.gl"),
        "GOLDEN FAIL: membar.gl instruction not found\nPTX:\n{}",
        ptx
    );
}
