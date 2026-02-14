use super::super::super::*;

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
    assert!(
        ptx.contains("dp4a.u32.u32"),
        "Expected dp4a.u32.u32 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("membar.cta"),
        "Expected membar.cta in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("ld.volatile.shared.u32"),
        "Expected ld.volatile.shared.u32 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("vote") || ptx.contains("ballot"),
        "Expected ballot in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("shfl.sync.idx"),
        "Expected shfl.sync.idx in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("atom.global.add.u32"),
        "Expected atom.global.add.u32 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("atom.global.exch.u32"),
        "Expected atom.global.exch.u32 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("atom.global.min.u32"),
        "Expected atom.global.min.u32 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("atom.global.max.u32"),
        "Expected atom.global.max.u32 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("atom.shared.exch.u32"),
        "Expected atom.shared.exch.u32 in: {}",
        ptx
    );
}

#[test]
fn test_sin_f32_instruction() {
    let kernel = PtxKernel::new("test_sin").build(|ctx| {
        let val = ctx.mov_f32_imm(1.57);
        let _result = ctx.sin_f32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("sin.approx.f32"),
        "Expected sin.approx.f32 in: {}",
        ptx
    );
}

#[test]
fn test_cos_f32_instruction() {
    let kernel = PtxKernel::new("test_cos").build(|ctx| {
        let val = ctx.mov_f32_imm(0.0);
        let _result = ctx.cos_f32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("cos.approx.f32"),
        "Expected cos.approx.f32 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("setp"),
        "Expected setp for sign extension in: {}",
        ptx
    );
}

#[test]
fn test_cvt_f32_s32_instruction() {
    let kernel = PtxKernel::new("test_cvt_f32_s32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _float = ctx.cvt_f32_s32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt.rn.f32.s32"),
        "Expected cvt.rn.f32.s32 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("st.global.u8"),
        "Expected st.global.u8 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("st.global.u16"),
        "Expected st.global.u16 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("st.shared.u16"),
        "Expected st.shared.u16 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("setp.eq.u32"),
        "Expected setp.eq.u32 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("mul.lo.u32"),
        "Expected mul.lo.u32 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("cvt.u64.u32"),
        "Expected cvt.u64.u32 in: {}",
        ptx
    );
}

#[test]
fn test_cvt_u32_u64_instruction() {
    let kernel = PtxKernel::new("test_cvt_u32_u64").build(|ctx| {
        let val = ctx.mov_u64_imm(1000);
        let _truncated = ctx.cvt_u32_u64(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt.u32.u64"),
        "Expected cvt.u32.u64 in: {}",
        ptx
    );
}

#[test]
fn test_cvt_f32_u32_instruction() {
    let kernel = PtxKernel::new("test_cvt_f32_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _float = ctx.cvt_f32_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt.rn.f32.u32"),
        "Expected cvt.rn.f32.u32 in: {}",
        ptx
    );
}

#[test]
fn test_mul_u64_instruction() {
    let kernel = PtxKernel::new("test_mul_u64").build(|ctx| {
        let a = ctx.mov_u64_imm(100);
        let _result = ctx.mul_u64(a, 200);
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
fn test_mul_u64_reg_instruction() {
    let kernel = PtxKernel::new("test_mul_u64_reg").build(|ctx| {
        let a = ctx.mov_u64_imm(100);
        let b = ctx.mov_u64_imm(200);
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
    assert!(
        ptx.contains("ld.global.u32"),
        "Expected ld.global.u32 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("atom.global.add.u32"),
        "Expected atomicAdd for debug marker in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("atom.global.add.u32"),
        "Expected atomicAdd for debug value in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("div.rn.f32") || ptx.contains("div.f32"),
        "Expected div.f32 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("@%p"),
        "Expected predicated branch @%p in: {}",
        ptx
    );
}

#[test]
fn test_shfl_idx_u32_instruction() {
    let kernel = PtxKernel::new("test_shfl_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(42);
        let _shuffled = ctx.shfl_idx_u32(val, 0, 0xFFFFFFFF);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("shfl.sync.idx"),
        "Expected shfl.sync.idx in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("mul.lo.u32"),
        "Expected mul.lo.u32 in: {}",
        ptx
    );
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

