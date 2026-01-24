//! Memory operation emission
//!
//! Handles: Ld, LdVolatile, LdParam, St, Cvt, Cvta, Atom* variants

use crate::ptx::instructions::{Operand, PtxInstruction, PtxOp};
use crate::ptx::types::PtxType;
use std::fmt::Write;

/// Emit memory opcode to the output string
pub(crate) fn emit_memory_opcode(instr: &PtxInstruction, s: &mut String) {
    match instr.op {
        PtxOp::Ld => {
            if let Some(ss) = instr.state_space {
                s.push_str("ld");
                s.push_str(ss.to_ptx_string());
            } else {
                s.push_str("ld");
            }
        }
        PtxOp::LdVolatile => {
            if let Some(ss) = instr.state_space {
                s.push_str("ld.volatile");
                s.push_str(ss.to_ptx_string());
            } else {
                s.push_str("ld.volatile");
            }
        }
        PtxOp::LdParam => s.push_str("ld.param"),
        PtxOp::St => {
            if let Some(ss) = instr.state_space {
                s.push_str("st");
                s.push_str(ss.to_ptx_string());
            } else {
                s.push_str("st");
            }
        }
        PtxOp::Cvt => emit_cvt_opcode(instr, s),
        PtxOp::Cvta => {
            let space = instr
                .state_space
                .map(|ss| ss.to_ptx_string())
                .unwrap_or(".shared");
            let ty = instr.ty.to_ptx_string();
            s.push_str("cvta");
            s.push_str(space);
            s.push_str(ty);
        }
        PtxOp::AtomAdd => emit_atomic_opcode(instr, s, "add"),
        PtxOp::AtomMin => emit_atomic_opcode(instr, s, "min"),
        PtxOp::AtomMax => emit_atomic_opcode(instr, s, "max"),
        PtxOp::AtomExch => emit_atomic_opcode(instr, s, "exch"),
        PtxOp::AtomCas => emit_atomic_opcode(instr, s, "cas"),
        _ => {}
    }
}

/// Emit cvt opcode with proper type conversion handling
fn emit_cvt_opcode(instr: &PtxInstruction, s: &mut String) {
    let dst_ty = instr.ty.to_ptx_string();
    let src_ty = if let Some(st) = instr.src_type {
        st.to_ptx_string()
    } else if let Some(Operand::Reg(vreg)) = instr.srcs.first() {
        vreg.ty().to_ptx_string()
    } else {
        ".u32"
    };

    let actual_src_type = instr.src_type.unwrap_or_else(|| {
        instr
            .srcs
            .first()
            .and_then(|src| {
                if let Operand::Reg(vreg) = src {
                    Some(vreg.ty())
                } else {
                    None
                }
            })
            .unwrap_or(PtxType::U32)
    });

    let src_is_f16 = actual_src_type == PtxType::F16;
    let dst_is_f32 = instr.ty == PtxType::F32;
    let is_f16_to_f32 = src_is_f16 && dst_is_f32;

    let needs_rounding = !is_f16_to_f32
        && (instr.ty.is_float()
            || instr
                .srcs
                .first()
                .is_some_and(|src| matches!(src, Operand::Reg(vreg) if vreg.ty().is_float())));

    let round = if needs_rounding {
        instr
            .rounding
            .as_ref()
            .map_or(".rn", |r| r.to_ptx_string())
    } else {
        ""
    };

    s.push_str("cvt");
    s.push_str(round);
    s.push_str(dst_ty);
    s.push_str(src_ty);
}

/// Emit atomic operation opcode
fn emit_atomic_opcode(instr: &PtxInstruction, s: &mut String, op: &str) {
    let space = instr
        .state_space
        .map(|ss| ss.to_ptx_string())
        .unwrap_or(".global");
    let _ = write!(s, "atom{}.{}", space, op);
}

/// Check if this is a memory operation
pub(crate) fn is_memory_op(op: &PtxOp) -> bool {
    matches!(
        op,
        PtxOp::Ld
            | PtxOp::LdVolatile
            | PtxOp::LdParam
            | PtxOp::St
            | PtxOp::Cvt
            | PtxOp::Cvta
            | PtxOp::AtomAdd
            | PtxOp::AtomMin
            | PtxOp::AtomMax
            | PtxOp::AtomExch
            | PtxOp::AtomCas
    )
}

/// Check if this op requires skipping the type suffix
pub(crate) fn skip_type_for_memory_op(op: &PtxOp) -> bool {
    matches!(op, PtxOp::Cvt | PtxOp::Cvta)
}
