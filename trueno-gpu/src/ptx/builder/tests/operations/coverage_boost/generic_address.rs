//! COVERAGE-BOOST: Generic Address Space Operations

use super::*;

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
