//! Control flow operation emission
//!
//! Handles: Bra, Ret, Bar, MemBar, Setp, Exit

use crate::ptx::instructions::{PtxInstruction, PtxOp};
use std::fmt::Write;

/// Emit control flow opcode - returns Some(full_instruction) for early-return ops
pub(crate) fn emit_control_opcode(
    instr: &PtxInstruction,
    prefix: &str,
) -> Option<String> {
    match instr.op {
        PtxOp::Bra => {
            if let Some(label) = &instr.label {
                return Some(format!("{}bra {};\n", prefix, label));
            }
            None
        }
        PtxOp::Ret => Some(format!("{}ret;\n", prefix)),
        PtxOp::Bar => {
            let barrier_id = instr.label.as_deref().unwrap_or("sync 0");
            Some(format!("{}bar.{};\n", prefix, barrier_id))
        }
        PtxOp::MemBar => {
            let scope = instr.label.as_deref().unwrap_or("cta");
            Some(format!("{}membar.{};\n", prefix, scope))
        }
        _ => None,
    }
}

/// Emit setp opcode with comparison operator
pub(crate) fn emit_setp_opcode(instr: &PtxInstruction, s: &mut String) {
    let cmp = instr.label.as_deref().unwrap_or("eq");
    let _ = write!(s, "setp.{}", cmp);
}

/// Check if this is a control flow operation
pub(crate) fn is_control_op(op: &PtxOp) -> bool {
    matches!(op, PtxOp::Bra | PtxOp::Ret | PtxOp::Bar | PtxOp::MemBar | PtxOp::Setp)
}

/// Check if this is an early-return control op
pub(crate) fn is_early_return_op(op: &PtxOp) -> bool {
    matches!(op, PtxOp::Bra | PtxOp::Ret | PtxOp::Bar | PtxOp::MemBar)
}
