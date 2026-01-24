//! Arithmetic operation emission
//!
//! Handles: Mov, Add, Sub, Mul, MadLo, Div, Fma, Neg, Ex2, Rsqrt, Rcp, Sqrt, Sin, Cos, Dp4a variants

use crate::ptx::instructions::{Operand, PtxInstruction, PtxOp};
use crate::ptx::types::PtxType;

/// Emit arithmetic opcode to the output string (allocating version)
pub(crate) fn emit_arithmetic_opcode(instr: &PtxInstruction, s: &mut String) {
    match instr.op {
        PtxOp::Mov => s.push_str("mov"),
        PtxOp::Add => s.push_str("add"),
        PtxOp::Sub => s.push_str("sub"),
        PtxOp::Mul => emit_mul_opcode(instr, s),
        PtxOp::MadLo => s.push_str("mad.lo"),
        PtxOp::Div => {
            if instr.ty.is_float() {
                s.push_str("div.rn");
            } else {
                s.push_str("div");
            }
        }
        PtxOp::Fma => {
            let round = instr
                .rounding
                .as_ref()
                .map_or(".rn", |r| r.to_ptx_string());
            s.push_str("fma");
            s.push_str(round);
        }
        PtxOp::Neg => s.push_str("neg"),
        PtxOp::Ex2 => s.push_str("ex2.approx"),
        PtxOp::Rsqrt => s.push_str("rsqrt.approx"),
        PtxOp::Rcp => s.push_str("rcp.approx"),
        PtxOp::Sqrt => {
            let round = instr
                .rounding
                .as_ref()
                .map_or(".rn", |r| r.to_ptx_string());
            s.push_str("sqrt");
            s.push_str(round);
        }
        PtxOp::Sin => s.push_str("sin.approx"),
        PtxOp::Cos => s.push_str("cos.approx"),
        PtxOp::Dp4a => s.push_str("dp4a.u32.u32"),
        PtxOp::Dp4aUS => s.push_str("dp4a.u32.s32"),
        PtxOp::Dp4aS32 => s.push_str("dp4a.s32.s32"),
        _ => {}
    }
}

/// Handle complex mul opcode emission
fn emit_mul_opcode(instr: &PtxInstruction, s: &mut String) {
    let is_wide_output = instr.ty == PtxType::U64 || instr.ty == PtxType::S64;
    let has_u64_source = instr.srcs.first().is_some_and(|src| {
        matches!(src, Operand::Reg(vreg) if vreg.ty() == PtxType::U64 || vreg.ty() == PtxType::S64)
    });

    if is_wide_output && !has_u64_source {
        let src_ty = if instr.ty == PtxType::U64 {
            ".u32"
        } else {
            ".s32"
        };
        s.push_str("mul.wide");
        s.push_str(src_ty);
    } else if is_wide_output && has_u64_source {
        s.push_str("mul.lo");
    } else if instr.ty.is_float() {
        s.push_str("mul");
    } else {
        s.push_str("mul.lo");
    }
}

/// Check if this is an arithmetic operation
pub(crate) fn is_arithmetic_op(op: &PtxOp) -> bool {
    matches!(
        op,
        PtxOp::Mov
            | PtxOp::Add
            | PtxOp::Sub
            | PtxOp::Mul
            | PtxOp::MadLo
            | PtxOp::Div
            | PtxOp::Fma
            | PtxOp::Neg
            | PtxOp::Ex2
            | PtxOp::Rsqrt
            | PtxOp::Rcp
            | PtxOp::Sqrt
            | PtxOp::Sin
            | PtxOp::Cos
            | PtxOp::Dp4a
            | PtxOp::Dp4aUS
            | PtxOp::Dp4aS32
    )
}
