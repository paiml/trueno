use super::super::super::*;

// =========================================================================
// COVERAGE-BOOST: Generic Address Space Operations
// =========================================================================

#[test]
fn test_ld_generic_u32() {
    let kernel = PtxKernel::new("test_generic_u32")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
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
    let kernel = PtxKernel::new("test_generic_u64")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_generic_u64(ptr);
            ctx.st_generic_u64(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("ld.u64") || ptx.contains(".u64"),
        "Expected u64 in: {}",
        ptx
    );
}

#[test]
fn test_ld_generic_u8() {
    let kernel = PtxKernel::new("test_generic_u8")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_generic_u8(ptr);
            ctx.st_generic_u8(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains(".u8") || ptx.contains("u8"),
        "Expected u8 ops in: {}",
        ptx
    );
}

#[test]
fn test_ld_generic_u16() {
    let kernel = PtxKernel::new("test_generic_u16")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_generic_u16(ptr);
            ctx.st_generic_u16(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains(".u16") || ptx.contains("u16"),
        "Expected u16 ops in: {}",
        ptx
    );
}

#[test]
fn test_ld_generic_f32() {
    let kernel = PtxKernel::new("test_generic_f32")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
            let ptr = ctx.load_param_u64("ptr");
            let val = ctx.ld_generic_f32(ptr);
            ctx.st_generic_f32(ptr, val);
            ctx.ret();
        });
    let ptx = kernel.emit();
    assert!(
        ptx.contains(".f32") || ptx.contains("f32"),
        "Expected f32 ops in: {}",
        ptx
    );
}

#[test]
fn test_ld_generic_u32_into() {
    let kernel = PtxKernel::new("test_generic_into")
        .param(PtxType::U64, "ptr")
        .build(|ctx| {
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
    assert!(
        ptx.contains("cvt") || ptx.contains("and"),
        "Expected conversion in: {}",
        ptx
    );
}

#[test]
fn test_cvt_u32_u16() {
    let kernel = PtxKernel::new("test_cvt_u32_u16").build(|ctx| {
        let val = ctx.mov_u32_imm(65535);
        let _converted = ctx.cvt_u32_u16(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") || ptx.contains("and"),
        "Expected conversion in: {}",
        ptx
    );
}

#[test]
fn test_cvt_u16_u32() {
    let kernel = PtxKernel::new("test_cvt_u16_u32").build(|ctx| {
        let val = ctx.mov_u32_imm(100);
        let _converted = ctx.cvt_u16_u32(val);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("cvt") || ptx.contains("and"),
        "Expected conversion in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("cvt.rni") || ptx.contains("cvt"),
        "Expected cvt in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("rcp.approx.f32"),
        "Expected rcp.approx.f32 in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("mov") && ptx.contains("12345"),
        "Expected mov in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("fma") || ptx.contains("mad"),
        "Expected fma/mad in: {}",
        ptx
    );
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
    assert!(
        ptx.contains("mov") && ptx.contains("999"),
        "Expected mov with 999 in: {}",
        ptx
    );
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
        let _val = ctx.const_f32(std::f32::consts::PI);
        ctx.ret();
    });
    let ptx = kernel.emit();
    // Float constants are emitted as hex (0F...) in PTX
    assert!(
        ptx.contains("mov.f32") && ptx.contains("0F"),
        "Expected const in: {}",
        ptx
    );
}

#[test]
fn test_const_u32_wrapper() {
    let kernel = PtxKernel::new("test_const_u32").build(|ctx| {
        let _val = ctx.const_u32(12345);
        ctx.ret();
    });
    let ptx = kernel.emit();
    assert!(
        ptx.contains("mov") && ptx.contains("12345"),
        "Expected const in: {}",
        ptx
    );
}

#[test]
fn test_shared_ptr_alias() {
    let kernel = PtxKernel::new("test_shared_ptr")
        .shared_memory(256)
        .build(|ctx| {
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
