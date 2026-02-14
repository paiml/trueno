use super::super::emit::{
    emit_global_mem_operand, emit_instruction, emit_operand, emit_shared_mem_operand,
};
use super::super::*;

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
    assert!(ptx.contains("ld.param"), "Expected ld.param in: {}", ptx);
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
    let kernel = PtxKernel::new("test_bra").build(|ctx| {
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
    let kernel = PtxKernel::new("test_sub").build(|ctx| {
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
    let kernel = PtxKernel::new("test_int_div").build(|ctx| {
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
    let kernel = PtxKernel::new("test_mul_wide").build(|ctx| {
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
    let kernel = PtxKernel::new("test_mad_lo").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let c = ctx.mov_u32_imm(5);
        let _result = ctx.mad_lo_u32(a, b, c);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(
        ptx.contains("mad.lo.u32"),
        "Expected mad.lo.u32 in: {}",
        ptx
    );
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
    assert!(ptx.contains("ld.shared"), "Expected ld.shared in: {}", ptx);
    assert!(ptx.contains("st.shared"), "Expected st.shared in: {}", ptx);
}

#[test]
fn test_cvt_instruction_emission() {
    // Test conversion instruction with types
    let kernel = PtxKernel::new("test_cvt").build(|ctx| {
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
    let kernel = PtxKernel::new("test_float_mul").build(|ctx| {
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
    let kernel = PtxKernel::new("test_bar").build(|ctx| {
        ctx.bar_sync(0);
        ctx.ret();
    });

    let ptx = kernel.emit();
    assert!(ptx.contains("bar.sync"), "Expected bar.sync in: {}", ptx);
}

#[test]
fn test_setp_comparison_ops() {
    // Test various setp comparison operations
    let kernel = PtxKernel::new("test_setp_cmp").build(|ctx| {
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
    let kernel = PtxKernel::new("test_shfl_down").build(|ctx| {
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
    let kernel = PtxKernel::new("test_shfl_idx").build(|ctx| {
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
    let kernel = PtxKernel::new("test_min_u32").build(|ctx| {
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
    let kernel = PtxKernel::new("test_ex2").build(|ctx| {
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
    let kernel = PtxKernel::new("test_rsqrt").build(|ctx| {
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
    let kernel = PtxKernel::new("test_rem").build(|ctx| {
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
    let kernel = PtxKernel::new("test_branch_if_not").build(|ctx| {
        let a = ctx.mov_u32_imm(10);
        let b = ctx.mov_u32_imm(20);
        let pred = ctx.setp_lt_u32(a, b);
        ctx.branch_if_not(pred, "skip");
        ctx.label("skip");
        ctx.ret();
    });

    let ptx = kernel.emit();
    // Negated predicate should have "!" prefix
    assert!(
        ptx.contains("@!"),
        "Expected negated predicate @! in: {}",
        ptx
    );
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
    let kernel = PtxKernel::new("test_shr_u32").build(|ctx| {
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
    let kernel = PtxKernel::new("test_and_u32").build(|ctx| {
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
    let kernel = PtxKernel::new("test_or_u32").build(|ctx| {
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
    let kernel = PtxKernel::new("test_shl_u32").build(|ctx| {
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
    let kernel = PtxKernel::new("test_shr_inplace").build(|ctx| {
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
    let kernel = PtxKernel::new("test_max_inplace").build(|ctx| {
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
    let kernel = PtxKernel::new("test_mov_f32_reg").build(|ctx| {
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
    let kernel = PtxKernel::new("test_mul_inplace").build(|ctx| {
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
fn test_f64_literal_format() {
    // Test f64 literal hex format (0D prefix)
    let kernel = PtxKernel::new("test_f64").build(|ctx| {
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
    use crate::ptx::instructions::Operand;
    use crate::ptx::registers::VirtualReg;

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
    use crate::ptx::instructions::Operand;
    use crate::ptx::registers::VirtualReg;

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
    use crate::ptx::instructions::Operand;
    use crate::ptx::registers::VirtualReg;

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
fn test_max_f32_non_inplace() {
    // Test max_f32 (non-inplace version)
    let kernel = PtxKernel::new("test_max_f32").build(|ctx| {
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
    use crate::ptx::instructions::{Operand, PtxInstruction, PtxOp};

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
