//! Advanced PTX Builder Golden Tests
//!
//! ⚠️ IMMUTABLE GUARDIAN - DO NOT MODIFY WITHOUT FALSIFICATION EVIDENCE
//!
//! These tests are LOCKED as immutable guardians of PTX correctness.
//! To modify: First demonstrate a falsifying test case (black swan).
//!
//! Tests for less commonly used but important PTX operations.

use trueno_gpu::ptx::{PtxArithmetic, PtxComparison, PtxControl, PtxKernel, PtxReg, PtxType};

// ============================================================================
// VECTOR LOAD/STORE OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_ld_global_f32_v4_instruction() {
    let kernel = PtxKernel::new("test_ld_v4")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let _vals = ctx.ld_global_f32_v4(ptr);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // Golden: ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd0]
    assert!(
        ptx.contains("ld.global") && ptx.contains("v4"),
        "GOLDEN FAIL: v4 load not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// PREDICATE OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_and_pred_instruction() {
    let kernel = PtxKernel::new("test_and_pred").build(|ctx| {
        let a = ctx.mov_u32_imm(1);
        let b = ctx.mov_u32_imm(1);
        let c = ctx.mov_u32_imm(0);
        let p1 = ctx.setp_eq_u32(a, b);
        let p2 = ctx.setp_eq_u32(a, c);
        let _p3 = ctx.and_pred(p1, p2);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: and.pred %p{dst}, %p{src1}, %p{src2}
    assert!(
        ptx.contains("and.pred"),
        "GOLDEN FAIL: and.pred instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_branch_if_not_instruction() {
    let kernel = PtxKernel::new("test_branch_if_not").build(|ctx| {
        let a = ctx.mov_u32_imm(1);
        let b = ctx.mov_u32_imm(0);
        let pred = ctx.setp_eq_u32(a, b);
        ctx.branch_if_not(pred, "not_taken");
        ctx.label("not_taken");
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: @!%p{pred} bra label
    assert!(
        ptx.contains("@!%p") && ptx.contains("bra"),
        "GOLDEN FAIL: negated predicate branch not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// REGISTER MOVE OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_mov_u64_imm_instruction() {
    let kernel = PtxKernel::new("test_mov_u64_imm").build(|ctx| {
        let _val = ctx.mov_u64_imm(0xDEADBEEFCAFEBABE);
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Golden: mov.u64 %rd{dst}, {immediate}
    assert!(
        ptx.contains("mov.u64"),
        "GOLDEN FAIL: mov.u64 instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mov_u64_into_instruction() {
    let kernel = PtxKernel::new("test_mov_u64_into").build(|ctx| {
        let dst = ctx.mov_u64_imm(0);
        ctx.mov_u64_into(dst, 42);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mov.u64"),
        "GOLDEN FAIL: mov.u64 into instruction not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mov_u32_into_instruction() {
    let kernel = PtxKernel::new("test_mov_u32_into").build(|ctx| {
        let dst = ctx.mov_u32_imm(0);
        ctx.mov_u32_into(dst, 42);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mov.u32"),
        "GOLDEN FAIL: mov.u32 into instruction not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// INPLACE OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_add_f32_inplace_instruction() {
    let kernel = PtxKernel::new("test_add_f32_inplace").build(|ctx| {
        let acc = ctx.mov_f32_imm(1.0);
        let val = ctx.mov_f32_imm(2.0);
        ctx.add_f32_inplace(acc, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("add.f32"),
        "GOLDEN FAIL: add.f32 inplace not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mul_f32_inplace_instruction() {
    let kernel = PtxKernel::new("test_mul_f32_inplace").build(|ctx| {
        let acc = ctx.mov_f32_imm(2.0);
        let val = ctx.mov_f32_imm(3.0);
        ctx.mul_f32_inplace(acc, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul.f32"),
        "GOLDEN FAIL: mul.f32 inplace not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_add_u32_inplace_instruction() {
    let kernel = PtxKernel::new("test_add_u32_inplace").build(|ctx| {
        let acc = ctx.mov_u32_imm(10);
        ctx.add_u32_inplace(acc, 5);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("add.u32"),
        "GOLDEN FAIL: add.u32 inplace not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_fma_f32_inplace_instruction() {
    let kernel = PtxKernel::new("test_fma_f32_inplace").build(|ctx| {
        let acc = ctx.mov_f32_imm(1.0);
        let a = ctx.mov_f32_imm(2.0);
        let b = ctx.mov_f32_imm(3.0);
        ctx.fma_f32_inplace(acc, a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("fma"),
        "GOLDEN FAIL: fma inplace not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// TYPE CONVERSION OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_cvt_u32_u64_instruction() {
    let kernel = PtxKernel::new("test_cvt_u32_u64").build(|ctx| {
        let wide = ctx.mov_u64_imm(42);
        let _narrow = ctx.cvt_u32_u64(wide);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt.u32.u64"),
        "GOLDEN FAIL: cvt.u32.u64 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_cvt_f32_s32_instruction() {
    let kernel = PtxKernel::new("test_cvt_f32_s32").build(|ctx| {
        let int_val = ctx.mov_u32_imm(42);
        let _float_val = ctx.cvt_f32_s32(int_val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("f32") && ptx.contains("s32"),
        "GOLDEN FAIL: cvt.f32.s32 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// SHARED MEMORY INTEGER OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_ld_shared_u32_instruction() {
    let kernel = PtxKernel::new("test_ld_shared_u32")
        .shared_memory(256)
        .build(|ctx| {
            let offset = ctx.mov_u32_imm(0);
            let _val = ctx.ld_shared_u32(offset);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.shared") && ptx.contains("u32"),
        "GOLDEN FAIL: ld.shared.u32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_shared_u32_instruction() {
    let kernel = PtxKernel::new("test_st_shared_u32")
        .shared_memory(256)
        .build(|ctx| {
            let offset = ctx.mov_u32_imm(0);
            let val = ctx.mov_u32_imm(42);
            ctx.st_shared_u32(offset, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.shared") && ptx.contains("u32"),
        "GOLDEN FAIL: st.shared.u32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_shared_u16_instruction() {
    let kernel = PtxKernel::new("test_st_shared_u16")
        .shared_memory(256)
        .build(|ctx| {
            let offset = ctx.mov_u32_imm(0);
            let val = ctx.mov_u32_imm(42);
            ctx.st_shared_u16(offset, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.shared") && ptx.contains("u16"),
        "GOLDEN FAIL: st.shared.u16 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// SPECIAL REGISTERS - Golden Tests (Y, Z dimensions)
// ============================================================================

#[test]
fn golden_special_reg_tid_y() {
    let kernel = PtxKernel::new("test_tid_y").build(|ctx| {
        let _tid = ctx.special_reg(PtxReg::TidY);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("%tid.y"),
        "GOLDEN FAIL: %tid.y not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_special_reg_tid_z() {
    let kernel = PtxKernel::new("test_tid_z").build(|ctx| {
        let _tid = ctx.special_reg(PtxReg::TidZ);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("%tid.z"),
        "GOLDEN FAIL: %tid.z not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_special_reg_ctaid_y() {
    let kernel = PtxKernel::new("test_ctaid_y").build(|ctx| {
        let _ctaid = ctx.special_reg(PtxReg::CtaIdY);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("%ctaid.y"),
        "GOLDEN FAIL: %ctaid.y not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_special_reg_ctaid_z() {
    let kernel = PtxKernel::new("test_ctaid_z").build(|ctx| {
        let _ctaid = ctx.special_reg(PtxReg::CtaIdZ);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("%ctaid.z"),
        "GOLDEN FAIL: %ctaid.z not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_special_reg_laneid() {
    let kernel = PtxKernel::new("test_laneid").build(|ctx| {
        let _lane = ctx.special_reg(PtxReg::LaneId);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("%laneid"),
        "GOLDEN FAIL: %laneid not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_special_reg_warpid() {
    let kernel = PtxKernel::new("test_warpid").build(|ctx| {
        let _warp = ctx.special_reg(PtxReg::WarpId);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("%warpid"),
        "GOLDEN FAIL: %warpid not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// INTEGER DIVISION/REMAINDER - Golden Tests
// ============================================================================

#[test]
fn golden_div_u32_instruction() {
    let kernel = PtxKernel::new("test_div_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(100);
        let _result = ctx.div_u32(a, 7);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("div.u32"),
        "GOLDEN FAIL: div.u32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_rem_u32_instruction() {
    let kernel = PtxKernel::new("test_rem_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(100);
        let _result = ctx.rem_u32(a, 7);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("rem.u32"),
        "GOLDEN FAIL: rem.u32 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// NEGATION - Golden Tests
// ============================================================================

#[test]
fn golden_neg_f32_instruction() {
    let kernel = PtxKernel::new("test_neg_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(42.0);
        let _result = ctx.neg_f32(a);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("neg.f32"),
        "GOLDEN FAIL: neg.f32 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// ABSOLUTE VALUE - Golden Tests
// ============================================================================

#[test]
fn golden_abs_f32_instruction() {
    let kernel = PtxKernel::new("test_abs_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(-42.0);
        let _result = ctx.abs_f32(a);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("abs.f32"),
        "GOLDEN FAIL: abs.f32 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// FLOOR/CEILING - Golden Tests
// ============================================================================

#[test]
fn golden_floor_f32_instruction() {
    let kernel = PtxKernel::new("test_floor_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(3.7);
        let _result = ctx.floor_f32(a);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("rmi"),
        "GOLDEN FAIL: floor (cvt.rmi) not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// RECIPROCAL - Golden Tests
// ============================================================================

#[test]
fn golden_rcp_f32_instruction() {
    let kernel = PtxKernel::new("test_rcp_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(4.0);
        let _result = ctx.rcp_f32(a);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("rcp"),
        "GOLDEN FAIL: rcp not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// DP4A VARIANTS - Golden Tests
// ============================================================================

#[test]
fn golden_dp4a_s32_inplace_instruction() {
    let kernel = PtxKernel::new("test_dp4a_s32_inplace").build(|ctx| {
        let acc = ctx.mov_u32_imm(0);
        let a = ctx.mov_u32_imm(0x01020304);
        let b = ctx.mov_u32_imm(0x01010101);
        ctx.dp4a_s32_inplace(acc, a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("dp4a"),
        "GOLDEN FAIL: dp4a.s32 inplace not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// VOLATILE OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_ld_shared_u32_volatile_instruction() {
    let kernel = PtxKernel::new("test_ld_shared_volatile")
        .shared_memory(256)
        .build(|ctx| {
            let offset = ctx.mov_u32_imm(0);
            let _val = ctx.ld_shared_u32_volatile(offset);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.volatile.shared") || ptx.contains("ld.shared"),
        "GOLDEN FAIL: volatile shared load not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// SHUFFLE WITH REGISTER SOURCE - Golden Tests
// ============================================================================

#[test]
fn golden_shfl_idx_u32_reg_instruction() {
    let kernel = PtxKernel::new("test_shfl_idx_reg").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let src_lane = ctx.mov_u32_imm(0);
        let _result = ctx.shfl_idx_u32_reg(val, src_lane, 0xFFFFFFFF);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("shfl"),
        "GOLDEN FAIL: shfl with reg source not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// ATOMIC OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_atom_add_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_add")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(1);
            let _old = ctx.atom_add_global_u32(ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom") && ptx.contains("add") && ptx.contains("global"),
        "GOLDEN FAIL: atom.global.add not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_atom_exch_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_exch")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(42);
            let _old = ctx.atom_exch_global_u32(ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom") && ptx.contains("exch"),
        "GOLDEN FAIL: atom.global.exch not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_atom_min_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_min")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(10);
            let _old = ctx.atom_min_global_u32(ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom") && ptx.contains("min"),
        "GOLDEN FAIL: atom.global.min not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_atom_max_global_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_max")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(100);
            let _old = ctx.atom_max_global_u32(ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom") && ptx.contains("max"),
        "GOLDEN FAIL: atom.global.max not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_atom_exch_shared_u32_instruction() {
    let kernel = PtxKernel::new("test_atom_exch_shared")
        .shared_memory(256)
        .build(|ctx| {
            let addr = ctx.shared_base_addr();
            let val = ctx.mov_u32_imm(42);
            let _old = ctx.atom_exch_shared_u32(addr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("atom") && ptx.contains("exch") && ptx.contains("shared"),
        "GOLDEN FAIL: atom.shared.exch not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// GLOBAL MEMORY OPERATIONS - Various Types
// ============================================================================

#[test]
fn golden_st_global_u32_instruction() {
    let kernel = PtxKernel::new("test_st_global_u32")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(42);
            ctx.st_global_u32(ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.global") && ptx.contains("u32"),
        "GOLDEN FAIL: st.global.u32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_ld_global_u64_instruction() {
    let kernel = PtxKernel::new("test_ld_global_u64")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let _val = ctx.ld_global_u64(ptr);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global") && ptx.contains("u64"),
        "GOLDEN FAIL: ld.global.u64 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_global_u64_instruction() {
    let kernel = PtxKernel::new("test_st_global_u64")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u64_imm(0xDEADBEEF);
            ctx.st_global_u64(ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.global") && ptx.contains("u64"),
        "GOLDEN FAIL: st.global.u64 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_ld_global_u8_instruction() {
    let kernel = PtxKernel::new("test_ld_global_u8")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let _val = ctx.ld_global_u8(ptr);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global") && ptx.contains("u8"),
        "GOLDEN FAIL: ld.global.u8 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_global_u8_instruction() {
    let kernel = PtxKernel::new("test_st_global_u8")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(255);
            ctx.st_global_u8(ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.global") && ptx.contains("u8"),
        "GOLDEN FAIL: st.global.u8 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_ld_global_u16_instruction() {
    let kernel = PtxKernel::new("test_ld_global_u16")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let _val = ctx.ld_global_u16(ptr);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.global") && ptx.contains("u16"),
        "GOLDEN FAIL: ld.global.u16 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_global_u16_instruction() {
    let kernel = PtxKernel::new("test_st_global_u16")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.mov_u32_imm(65535);
            ctx.st_global_u16(ptr, val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.global") && ptx.contains("u16"),
        "GOLDEN FAIL: st.global.u16 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// TYPE CONVERSIONS - Various Width
// ============================================================================

#[test]
fn golden_cvt_u32_u8_instruction() {
    let kernel = PtxKernel::new("test_cvt_u32_u8")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let byte = ctx.ld_global_u8(ptr);
            let _wide = ctx.cvt_u32_u8(byte);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("u32"),
        "GOLDEN FAIL: cvt.u32.u8 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_cvt_u32_u16_instruction() {
    let kernel = PtxKernel::new("test_cvt_u32_u16")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let half = ctx.ld_global_u16(ptr);
            let _wide = ctx.cvt_u32_u16(half);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("u32"),
        "GOLDEN FAIL: cvt.u32.u16 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_cvt_u16_u32_instruction() {
    let kernel = PtxKernel::new("test_cvt_u16_u32").build(|ctx| {
        let wide = ctx.mov_u32_imm(65535);
        let _narrow = ctx.cvt_u16_u32(wide);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("u16"),
        "GOLDEN FAIL: cvt.u16.u32 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// SHIFT OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_shr_u32_instruction() {
    let kernel = PtxKernel::new("test_shr_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(0xFF00);
        let shift = ctx.mov_u32_imm(8);
        let _result = ctx.shr_u32(val, shift);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("shr.b32"),
        "GOLDEN FAIL: shr.b32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_shr_u32_imm_instruction() {
    let kernel = PtxKernel::new("test_shr_u32_imm").build(|ctx| {
        let val = ctx.mov_u32_imm(0xFF00);
        let _result = ctx.shr_u32_imm(val, 8);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("shr.b32"),
        "GOLDEN FAIL: shr.b32 imm not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_shl_u32_instruction() {
    let kernel = PtxKernel::new("test_shl_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(0xFF);
        let shift = ctx.mov_u32_imm(8);
        let _result = ctx.shl_u32(val, shift);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("shl.b32"),
        "GOLDEN FAIL: shl.b32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_shl_u32_imm_instruction() {
    let kernel = PtxKernel::new("test_shl_u32_imm").build(|ctx| {
        let val = ctx.mov_u32_imm(0xFF);
        let _result = ctx.shl_u32_imm(val, 8);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("shl.b32"),
        "GOLDEN FAIL: shl.b32 imm not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// BITWISE OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_or_u32_instruction() {
    let kernel = PtxKernel::new("test_or_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(0x0F);
        let b = ctx.mov_u32_imm(0xF0);
        let _result = ctx.or_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("or.b32"),
        "GOLDEN FAIL: or.b32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_or_u32_into_instruction() {
    let kernel = PtxKernel::new("test_or_u32_into").build(|ctx| {
        let dst = ctx.mov_u32_imm(0x0F);
        let a = ctx.mov_u32_imm(0x0F);
        let b = ctx.mov_u32_imm(0xF0);
        ctx.or_u32_into(dst, a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("or.b32"),
        "GOLDEN FAIL: or.b32 into not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_and_u32_imm_instruction() {
    let kernel = PtxKernel::new("test_and_u32_imm").build(|ctx| {
        let tid = ctx.special_reg(PtxReg::TidX);
        let _lane_id = ctx.and_u32_imm(tid, 31);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("and.b32"),
        "GOLDEN FAIL: and.b32 imm not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// SELECT OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_selp_u32_instruction() {
    let kernel = PtxKernel::new("test_selp_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(5);
        let pred = ctx.setp_ge_u32(a, b); // a >= b
        let t = ctx.mov_u32_imm(1);
        let f = ctx.mov_u32_imm(0);
        let _result = ctx.selp_u32(pred, t, f);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("selp.u32"),
        "GOLDEN FAIL: selp.u32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_selp_f32_instruction() {
    let kernel = PtxKernel::new("test_selp_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(10.0);
        let b = ctx.mov_f32_imm(5.0);
        let pred = ctx.setp_gt_f32(a, b);
        let t = ctx.mov_f32_imm(1.0);
        let f = ctx.mov_f32_imm(0.0);
        let _result = ctx.selp_f32(pred, t, f);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("selp.f32"),
        "GOLDEN FAIL: selp.f32 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// INPLACE OPERATIONS - More Coverage
// ============================================================================

#[test]
fn golden_shr_u32_inplace_instruction() {
    let kernel = PtxKernel::new("test_shr_u32_inplace").build(|ctx| {
        let val = ctx.mov_u32_imm(256);
        ctx.shr_u32_inplace(val, 1);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("shr.b32"),
        "GOLDEN FAIL: shr.b32 inplace not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_max_f32_inplace_instruction() {
    let kernel = PtxKernel::new("test_max_f32_inplace").build(|ctx| {
        let acc = ctx.mov_f32_imm(f32::NEG_INFINITY);
        let val = ctx.mov_f32_imm(10.0);
        ctx.max_f32_inplace(acc, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("max.f32"),
        "GOLDEN FAIL: max.f32 inplace not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_div_f32_inplace_instruction() {
    let kernel = PtxKernel::new("test_div_f32_inplace").build(|ctx| {
        let acc = ctx.mov_f32_imm(100.0);
        let divisor = ctx.mov_f32_imm(10.0);
        ctx.div_f32_inplace(acc, divisor);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("div") && ptx.contains("f32"),
        "GOLDEN FAIL: div.f32 inplace not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_add_u32_reg_inplace_instruction() {
    let kernel = PtxKernel::new("test_add_u32_reg_inplace").build(|ctx| {
        let acc = ctx.mov_u32_imm(10);
        let val = ctx.mov_u32_imm(5);
        ctx.add_u32_reg_inplace(acc, val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("add.u32"),
        "GOLDEN FAIL: add.u32 reg inplace not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// REGISTER MOVE OPERATIONS - More Coverage
// ============================================================================

#[test]
fn golden_mov_f32_reg_instruction() {
    let kernel = PtxKernel::new("test_mov_f32_reg").build(|ctx| {
        let src = ctx.mov_f32_imm(3.14);
        let dst = ctx.mov_f32_imm(0.0);
        ctx.mov_f32_reg(dst, src);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mov.f32"),
        "GOLDEN FAIL: mov.f32 reg not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mov_u32_reg_instruction() {
    let kernel = PtxKernel::new("test_mov_u32_reg").build(|ctx| {
        let src = ctx.mov_u32_imm(42);
        let dst = ctx.mov_u32_imm(0);
        ctx.mov_u32_reg(dst, src);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mov.u32"),
        "GOLDEN FAIL: mov.u32 reg not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mov_u64_reg_instruction() {
    let kernel = PtxKernel::new("test_mov_u64_reg").build(|ctx| {
        let src = ctx.mov_u64_imm(0xDEADBEEF);
        let dst = ctx.mov_u64_imm(0);
        ctx.mov_u64_reg(dst, src);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mov.u64"),
        "GOLDEN FAIL: mov.u64 reg not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// F32 COMPARISON - Golden Tests
// ============================================================================

#[test]
fn golden_setp_gt_f32_instruction() {
    let kernel = PtxKernel::new("test_setp_gt_f32").build(|ctx| {
        let a = ctx.mov_f32_imm(10.0);
        let b = ctx.mov_f32_imm(5.0);
        let _pred = ctx.setp_gt_f32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("setp") && ptx.contains("gt") && ptx.contains("f32"),
        "GOLDEN FAIL: setp.gt.f32 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// WARP SHUFFLE - More Variants
// ============================================================================

#[test]
fn golden_shfl_down_u32_instruction() {
    let kernel = PtxKernel::new("test_shfl_down_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _result = ctx.shfl_down_u32(val, 1, 0xFFFFFFFF);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("shfl") && ptx.contains("down"),
        "GOLDEN FAIL: shfl.down not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// MULTIPLY VARIANTS - Golden Tests
// ============================================================================

#[test]
fn golden_mul_lo_u32_instruction() {
    let kernel = PtxKernel::new("test_mul_lo_u32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let _result = ctx.mul_lo_u32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul") && ptx.contains("u32"),
        "GOLDEN FAIL: mul.lo.u32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_mul_lo_s32_instruction() {
    let kernel = PtxKernel::new("test_mul_lo_s32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let _result = ctx.mul_lo_s32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mul") && ptx.contains("s32"),
        "GOLDEN FAIL: mul.lo.s32 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// MIN/MAX S32 - Golden Tests
// ============================================================================

#[test]
fn golden_min_s32_instruction() {
    let kernel = PtxKernel::new("test_min_s32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let _result = ctx.min_s32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("min.s32"),
        "GOLDEN FAIL: min.s32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_max_s32_instruction() {
    let kernel = PtxKernel::new("test_max_s32").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let _result = ctx.max_s32(a, b);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("max.s32"),
        "GOLDEN FAIL: max.s32 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// CONST HELPERS - Golden Tests
// ============================================================================

#[test]
fn golden_const_f32_instruction() {
    let kernel = PtxKernel::new("test_const_f32").build(|ctx| {
        let _val = ctx.const_f32(f32::NEG_INFINITY);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mov.f32"),
        "GOLDEN FAIL: const_f32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_const_u32_instruction() {
    let kernel = PtxKernel::new("test_const_u32").build(|ctx| {
        let _val = ctx.const_u32(0xFFFFFFFF);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mov.u32"),
        "GOLDEN FAIL: const_u32 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// SHARED POINTER - Golden Tests
// ============================================================================

#[test]
fn golden_shared_ptr_instruction() {
    let kernel = PtxKernel::new("test_shared_ptr")
        .shared_memory(256)
        .build(|ctx| {
            let _ptr = ctx.shared_ptr();
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvta.shared.u64") || ptx.contains(".shared"),
        "GOLDEN FAIL: shared_ptr not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// F16 OPERATIONS - Golden Tests
// ============================================================================

#[test]
fn golden_cvt_f16_f32_instruction() {
    let kernel = PtxKernel::new("test_cvt_f16_f32").build(|ctx| {
        let f32_val = ctx.mov_f32_imm(3.14);
        let _f16_val = ctx.cvt_f16_f32(f32_val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("f16"),
        "GOLDEN FAIL: cvt.f16.f32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_cvt_f32_f16_instruction() {
    let kernel = PtxKernel::new("test_cvt_f32_f16")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let f16_val = ctx.ld_global_f16(ptr);
            let _f32_val = ctx.cvt_f32_f16(f16_val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("f32"),
        "GOLDEN FAIL: cvt.f32.f16 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_ld_global_f16_instruction() {
    let kernel = PtxKernel::new("test_ld_global_f16")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let _val = ctx.ld_global_f16(ptr);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // PTX uses .b16 for f16 loads
    assert!(
        ptx.contains("ld.global") && ptx.contains("b16"),
        "GOLDEN FAIL: ld.global.b16 (f16) not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_st_global_f16_instruction() {
    let kernel = PtxKernel::new("test_st_global_f16")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let f32_val = ctx.mov_f32_imm(3.14);
            let f16_val = ctx.cvt_f16_f32(f32_val);
            ctx.st_global_f16(ptr, f16_val);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("st.global") && ptx.contains("f16"),
        "GOLDEN FAIL: st.global.f16 not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// WMMA TENSOR CORE OPERATIONS - Golden Tests
// ============================================================================

use trueno_gpu::ptx::WmmaLayout;

#[test]
fn golden_wmma_load_a_f16_instruction() {
    let kernel = PtxKernel::new("test_wmma_load_a")
        .param(PtxType::U64, "a_ptr")
        .build(|ctx| {
            let a_ptr = ctx.load_param_u64("a_ptr");
            let _frag = ctx.wmma_load_a_f16(a_ptr, 16, WmmaLayout::RowMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("wmma") && ptx.contains("load") && ptx.contains("f16"),
        "GOLDEN FAIL: wmma.load.a.f16 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_wmma_load_b_f16_instruction() {
    let kernel = PtxKernel::new("test_wmma_load_b")
        .param(PtxType::U64, "b_ptr")
        .build(|ctx| {
            let b_ptr = ctx.load_param_u64("b_ptr");
            let _frag = ctx.wmma_load_b_f16(b_ptr, 16, WmmaLayout::ColMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("wmma") && ptx.contains("load") && ptx.contains("f16"),
        "GOLDEN FAIL: wmma.load.b.f16 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_wmma_load_c_f32_instruction() {
    let kernel = PtxKernel::new("test_wmma_load_c")
        .param(PtxType::U64, "c_ptr")
        .build(|ctx| {
            let c_ptr = ctx.load_param_u64("c_ptr");
            let _frag = ctx.wmma_load_c_f32(c_ptr, 16, WmmaLayout::RowMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("wmma") && ptx.contains("load") && ptx.contains("f32"),
        "GOLDEN FAIL: wmma.load.c.f32 not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_wmma_init_c_zero_instruction() {
    let kernel = PtxKernel::new("test_wmma_init_c_zero").build(|ctx| {
        let _frag = ctx.wmma_init_c_zero();
        ctx.ret();
    });

    let ptx = kernel.emit();
    // wmma_init_c_zero just moves 0.0 into registers
    assert!(
        ptx.contains("mov.f32"),
        "GOLDEN FAIL: wmma_init_c_zero (mov.f32 0.0) not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_wmma_mma_f16_f32_instruction() {
    let kernel = PtxKernel::new("test_wmma_mma")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .build(|ctx| {
            let a_ptr = ctx.load_param_u64("a_ptr");
            let b_ptr = ctx.load_param_u64("b_ptr");
            let frag_a = ctx.wmma_load_a_f16(a_ptr, 16, WmmaLayout::RowMajor);
            let frag_b = ctx.wmma_load_b_f16(b_ptr, 16, WmmaLayout::ColMajor);
            let frag_c = ctx.wmma_init_c_zero();
            let _frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("wmma") && ptx.contains("mma"),
        "GOLDEN FAIL: wmma.mma not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_wmma_store_d_f32_instruction() {
    let kernel = PtxKernel::new("test_wmma_store_d")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .param(PtxType::U64, "d_ptr")
        .build(|ctx| {
            let a_ptr = ctx.load_param_u64("a_ptr");
            let b_ptr = ctx.load_param_u64("b_ptr");
            let d_ptr = ctx.load_param_u64("d_ptr");
            let frag_a = ctx.wmma_load_a_f16(a_ptr, 16, WmmaLayout::RowMajor);
            let frag_b = ctx.wmma_load_b_f16(b_ptr, 16, WmmaLayout::ColMajor);
            let frag_c = ctx.wmma_init_c_zero();
            let frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);
            ctx.wmma_store_d_f32(d_ptr, &frag_d, 16, WmmaLayout::RowMajor);
            ctx.ret();
        });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("wmma") && ptx.contains("store"),
        "GOLDEN FAIL: wmma.store.d not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// DEBUG HELPERS - Golden Tests
// ============================================================================

#[test]
fn golden_emit_debug_marker_instruction() {
    let kernel = PtxKernel::new("test_debug_marker")
        .param(PtxType::U64, "debug_ptr")
        .build(|ctx| {
            let debug_ptr = ctx.load_param_u64("debug_ptr");
            let _slot = ctx.emit_debug_marker(debug_ptr, 0xDEAD);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // emit_debug_marker uses atom.add, st.global
    assert!(
        ptx.contains("atom") && ptx.contains("add") && ptx.contains("st.global"),
        "GOLDEN FAIL: emit_debug_marker (atom.add + st.global) not found\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_emit_debug_value_instruction() {
    let kernel = PtxKernel::new("test_debug_value")
        .param(PtxType::U64, "debug_ptr")
        .build(|ctx| {
            let debug_ptr = ctx.load_param_u64("debug_ptr");
            let value = ctx.special_reg(PtxReg::TidX);
            let _slot = ctx.emit_debug_value(debug_ptr, value);
            ctx.ret();
        });

    let ptx = kernel.emit();
    // emit_debug_value uses atom.add, st.global
    assert!(
        ptx.contains("atom") && ptx.contains("st.global"),
        "GOLDEN FAIL: emit_debug_value (atom + st.global) not found\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// CONVERSION WITH ROUNDING - Golden Tests
// ============================================================================

#[test]
fn golden_cvt_rni_s32_f32_instruction() {
    let kernel = PtxKernel::new("test_cvt_rni_s32_f32").build(|ctx| {
        let f_val = ctx.mov_f32_imm(3.7);
        let _i_val = ctx.cvt_rni_s32_f32(f_val);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") && ptx.contains("s32"),
        "GOLDEN FAIL: cvt.rni.s32.f32 not found\nPTX:\n{}",
        ptx
    );
}
