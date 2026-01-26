//! PTX Instruction Emission
//!
//! This module contains all PTX code generation functions that convert
//! PtxInstruction structs into PTX assembly text.
//!
//! ## Architecture (PMAT-018 Shatter)
//!
//! The emission logic is split by operation category:
//! - `arithmetic.rs` - add, sub, mul, div, fma, transcendental ops
//! - `memory.rs` - ld, st, cvt, cvta, atomic ops
//! - `control.rs` - bra, ret, bar, membar, setp
//! - `warp.rs` - shfl, vote, bit manipulation
//! - `wmma.rs` - tensor core operations
//! - `operand.rs` - shared operand formatting
//!
//! ## Functions
//!
//! - `emit_instruction()` - Allocating version for single instruction
//! - `write_instruction()` - Zero-allocation version for bulk output

mod arithmetic;
mod control;
mod memory;
mod operand;
mod warp;
mod wmma;

use std::fmt::Write;

use crate::ptx::instructions::{Operand, PtxInstruction, PtxOp};
use crate::ptx::types::PtxStateSpace;

// Re-export operand functions for external use
pub(crate) use operand::{
    emit_global_mem_operand, emit_operand, emit_shared_mem_operand, write_mem_operand,
    write_operand,
};

/// Emit a single instruction as PTX (allocating version)
pub(crate) fn emit_instruction(instr: &PtxInstruction) -> String {
    let mut s = String::new();

    // Handle labels
    if let Some(label) = &instr.label {
        if label.ends_with(':') {
            return format!("{}:\n", &label[..label.len() - 1]);
        }
    }

    // Build predicate prefix
    let prefix = if let Some(pred) = &instr.predicate {
        let neg = if pred.negated { "!" } else { "" };
        format!("    @{}{} ", neg, pred.reg.to_ptx_string())
    } else {
        "    ".to_string()
    };
    s.push_str(&prefix);

    // Handle early-return control ops (bra, ret, bar, membar)
    if control::is_early_return_op(&instr.op) {
        if let Some(result) = control::emit_control_opcode(instr, &prefix) {
            return result;
        }
    }

    // Handle WMMA ops (complex formatting, early return)
    if wmma::is_wmma_op(&instr.op) {
        return match instr.op {
            PtxOp::WmmaLoadA => wmma::emit_wmma_load(s, instr, "a"),
            PtxOp::WmmaLoadB => wmma::emit_wmma_load(s, instr, "b"),
            PtxOp::WmmaLoadC => wmma::emit_wmma_load(s, instr, "c"),
            PtxOp::WmmaMma => wmma::emit_wmma_mma(s, instr),
            PtxOp::WmmaStoreD => wmma::emit_wmma_store(s, instr),
            _ => s,
        };
    }

    // Emit opcode based on category
    if arithmetic::is_arithmetic_op(&instr.op) {
        arithmetic::emit_arithmetic_opcode(instr, &mut s);
    } else if memory::is_memory_op(&instr.op) {
        memory::emit_memory_opcode(instr, &mut s);
    } else if instr.op == PtxOp::Setp {
        control::emit_setp_opcode(instr, &mut s);
    } else if warp::is_warp_op(&instr.op) {
        warp::emit_warp_opcode(&instr.op, &mut s);
    } else {
        // Fallback for unhandled ops
        s.push_str(&format!("{:?}", instr.op).to_lowercase());
    }

    // Type suffix (skip for certain ops)
    if !should_skip_type_suffix(instr) {
        s.push_str(instr.ty.to_ptx_string());
    }

    s.push(' ');

    // Emit destinations and sources
    emit_destinations(instr, &mut s);
    emit_sources(instr, &mut s);

    s.push_str(";\n");
    s
}

/// Check if type suffix should be skipped
fn should_skip_type_suffix(instr: &PtxInstruction) -> bool {
    // Wide mul from u32 sources
    let is_wide_mul_from_u32 = instr.op == PtxOp::Mul
        && (instr.ty == crate::ptx::types::PtxType::U64
            || instr.ty == crate::ptx::types::PtxType::S64)
        && !instr.srcs.first().is_some_and(|src| {
            matches!(src, Operand::Reg(vreg)
                if vreg.ty() == crate::ptx::types::PtxType::U64
                || vreg.ty() == crate::ptx::types::PtxType::S64)
        });

    memory::skip_type_for_memory_op(&instr.op)
        || warp::skip_type_for_warp_op(&instr.op)
        || wmma::is_wmma_op(&instr.op)
        || is_wide_mul_from_u32
}

/// Emit destination operands
fn emit_destinations(instr: &PtxInstruction, s: &mut String) {
    if !instr.dsts.is_empty() {
        s.push('{');
        for (i, dst) in instr.dsts.iter().enumerate() {
            s.push_str(&emit_operand(dst));
            if i < instr.dsts.len() - 1 {
                s.push_str(", ");
            }
        }
        s.push('}');
        if !instr.srcs.is_empty() {
            s.push_str(", ");
        }
    } else if let Some(dst) = &instr.dst {
        s.push_str(&emit_operand(dst));
        if !instr.srcs.is_empty() {
            s.push_str(", ");
        }
    }
}

/// Emit source operands with proper memory addressing
fn emit_sources(instr: &PtxInstruction, s: &mut String) {
    let is_memory_op = matches!(instr.op, PtxOp::Ld | PtxOp::LdVolatile | PtxOp::St);
    let is_atomic_op = matches!(
        instr.op,
        PtxOp::AtomAdd | PtxOp::AtomMin | PtxOp::AtomMax | PtxOp::AtomExch | PtxOp::AtomCas
    );
    let is_shared_mem = instr.state_space == Some(PtxStateSpace::Shared);
    let is_global_mem = instr.state_space == Some(PtxStateSpace::Global)
        || (is_memory_op && instr.state_space.is_none());

    for (i, src) in instr.srcs.iter().enumerate() {
        if i == 0 && (is_memory_op || is_atomic_op) {
            if is_shared_mem {
                s.push_str(&emit_shared_mem_operand(src));
            } else if is_global_mem || is_atomic_op {
                s.push_str(&emit_global_mem_operand(src));
            } else {
                s.push_str(&emit_operand(src));
            }
        } else {
            s.push_str(&emit_operand(src));
        }
        if i < instr.srcs.len() - 1 {
            s.push_str(", ");
        }
    }
}

/// Write a single instruction directly to a String buffer (zero intermediate allocations)
pub(super) fn write_instruction(instr: &PtxInstruction, out: &mut String) {
    // Handle labels
    if let Some(label) = &instr.label {
        if label.ends_with(':') {
            let _ = writeln!(out, "{}:", &label[..label.len() - 1]);
            return;
        }
    }

    // Predicate
    if let Some(pred) = &instr.predicate {
        let neg = if pred.negated { "!" } else { "" };
        let _ = write!(out, "    @{}{} ", neg, pred.reg);
    } else {
        out.push_str("    ");
    }

    // Handle early-return control ops
    if control::is_early_return_op(&instr.op) {
        let prefix = if let Some(pred) = &instr.predicate {
            let neg = if pred.negated { "!" } else { "" };
            format!("    @{}{} ", neg, pred.reg)
        } else {
            "    ".to_string()
        };
        if let Some(result) = control::emit_control_opcode(instr, &prefix) {
            out.push_str(&result[prefix.len()..]); // Skip prefix, already written
            return;
        }
    }

    // Handle WMMA ops (complex formatting)
    if wmma::is_wmma_op(&instr.op) {
        out.push_str(&emit_instruction(instr));
        return;
    }

    // Emit opcode based on category
    if arithmetic::is_arithmetic_op(&instr.op) {
        arithmetic::emit_arithmetic_opcode(instr, out);
    } else if memory::is_memory_op(&instr.op) {
        memory::emit_memory_opcode(instr, out);
    } else if instr.op == PtxOp::Setp {
        control::emit_setp_opcode(instr, out);
    } else if warp::is_warp_op(&instr.op) {
        warp::emit_warp_opcode(&instr.op, out);
    } else {
        let op_str = format!("{:?}", instr.op).to_lowercase();
        out.push_str(&op_str);
    }

    // Type suffix
    if !should_skip_type_suffix(instr) {
        out.push_str(instr.ty.to_ptx_string());
    }

    out.push(' ');

    // Destinations
    if !instr.dsts.is_empty() {
        out.push('{');
        for (i, dst) in instr.dsts.iter().enumerate() {
            write_operand(dst, out);
            if i < instr.dsts.len() - 1 {
                out.push_str(", ");
            }
        }
        out.push('}');
        if !instr.srcs.is_empty() {
            out.push_str(", ");
        }
    } else if let Some(dst) = &instr.dst {
        write_operand(dst, out);
        if !instr.srcs.is_empty() {
            out.push_str(", ");
        }
    }

    // Sources
    let is_memory_op = matches!(instr.op, PtxOp::Ld | PtxOp::LdVolatile | PtxOp::St);
    let is_atomic_op = matches!(
        instr.op,
        PtxOp::AtomAdd | PtxOp::AtomMin | PtxOp::AtomMax | PtxOp::AtomExch | PtxOp::AtomCas
    );
    let is_shared_mem = instr.state_space == Some(PtxStateSpace::Shared);
    let is_global_mem = instr.state_space == Some(PtxStateSpace::Global)
        || (is_memory_op && instr.state_space.is_none());

    for (i, src) in instr.srcs.iter().enumerate() {
        if i == 0 && (is_memory_op || is_atomic_op) {
            if is_shared_mem || is_global_mem || is_atomic_op {
                write_mem_operand(src, out);
            } else {
                write_operand(src, out);
            }
        } else {
            write_operand(src, out);
        }
        if i < instr.srcs.len() - 1 {
            out.push_str(", ");
        }
    }

    out.push_str(";\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::instructions::{Operand, Predicate};
    use crate::ptx::registers::VirtualReg;
    use crate::ptx::types::PtxType;

    fn make_instr(op: PtxOp) -> PtxInstruction {
        PtxInstruction {
            op,
            ty: PtxType::F32,
            src_type: None,
            dst: None,
            dsts: vec![],
            srcs: vec![],
            label: None,
            predicate: None,
            state_space: None,
            rounding: None,
        }
    }

    fn vreg(id: u32, ty: PtxType) -> VirtualReg {
        VirtualReg::new(id, ty)
    }

    // === emit_instruction tests ===

    #[test]
    fn test_emit_label() {
        let mut instr = make_instr(PtxOp::Add);
        instr.label = Some("loop_start:".to_string());
        let result = emit_instruction(&instr);
        assert_eq!(result, "loop_start:\n");
    }

    #[test]
    fn test_emit_with_predicate() {
        let mut instr = make_instr(PtxOp::Add);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
        instr.srcs = vec![
            Operand::Reg(vreg(1, PtxType::F32)),
            Operand::Reg(vreg(2, PtxType::F32)),
        ];
        instr.predicate = Some(Predicate {
            reg: vreg(10, PtxType::Pred),
            negated: false,
        });
        let result = emit_instruction(&instr);
        assert!(result.contains("@"));
    }

    #[test]
    fn test_emit_with_negated_predicate() {
        let mut instr = make_instr(PtxOp::Add);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
        instr.srcs = vec![
            Operand::Reg(vreg(1, PtxType::F32)),
            Operand::Reg(vreg(2, PtxType::F32)),
        ];
        instr.predicate = Some(Predicate {
            reg: vreg(10, PtxType::Pred),
            negated: true,
        });
        let result = emit_instruction(&instr);
        assert!(result.contains("@!") || result.contains("@%!"));
    }

    #[test]
    fn test_emit_add_instruction() {
        let mut instr = make_instr(PtxOp::Add);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
        instr.srcs = vec![
            Operand::Reg(vreg(1, PtxType::F32)),
            Operand::Reg(vreg(2, PtxType::F32)),
        ];
        let result = emit_instruction(&instr);
        assert!(result.contains("add"));
        assert!(result.contains(".f32"));
    }

    #[test]
    fn test_emit_mul_instruction() {
        let mut instr = make_instr(PtxOp::Mul);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
        instr.srcs = vec![
            Operand::Reg(vreg(1, PtxType::F32)),
            Operand::Reg(vreg(2, PtxType::F32)),
        ];
        let result = emit_instruction(&instr);
        assert!(result.contains("mul"));
    }

    #[test]
    fn test_emit_ld_global() {
        let mut instr = make_instr(PtxOp::Ld);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
        instr.srcs = vec![Operand::Reg(vreg(1, PtxType::U64))];
        instr.state_space = Some(PtxStateSpace::Global);
        let result = emit_instruction(&instr);
        assert!(result.contains("ld.global"));
    }

    #[test]
    fn test_emit_ld_shared() {
        let mut instr = make_instr(PtxOp::Ld);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
        instr.srcs = vec![Operand::Reg(vreg(1, PtxType::U64))];
        instr.state_space = Some(PtxStateSpace::Shared);
        let result = emit_instruction(&instr);
        assert!(result.contains("ld.shared"));
    }

    #[test]
    fn test_emit_st_instruction() {
        let mut instr = make_instr(PtxOp::St);
        instr.srcs = vec![
            Operand::Reg(vreg(0, PtxType::U64)),
            Operand::Reg(vreg(1, PtxType::F32)),
        ];
        instr.state_space = Some(PtxStateSpace::Global);
        let result = emit_instruction(&instr);
        assert!(result.contains("st.global"));
    }

    #[test]
    fn test_emit_setp_instruction() {
        let mut instr = make_instr(PtxOp::Setp);
        instr.ty = PtxType::F32;
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::Pred)));
        instr.srcs = vec![
            Operand::Reg(vreg(1, PtxType::F32)),
            Operand::Reg(vreg(2, PtxType::F32)),
        ];
        instr.label = Some("lt".to_string());
        let result = emit_instruction(&instr);
        assert!(result.contains("setp"));
    }

    #[test]
    fn test_emit_shfl_down() {
        let mut instr = make_instr(PtxOp::ShflDown);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
        instr.srcs = vec![
            Operand::ImmU64(0xFFFFFFFF),
            Operand::Reg(vreg(1, PtxType::F32)),
            Operand::ImmU64(1),
            Operand::ImmU64(0x1F),
        ];
        let result = emit_instruction(&instr);
        assert!(result.contains("shfl"));
    }

    #[test]
    fn test_emit_atom_add() {
        let mut instr = make_instr(PtxOp::AtomAdd);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::U32)));
        instr.srcs = vec![Operand::Reg(vreg(1, PtxType::U64)), Operand::ImmU64(1)];
        instr.state_space = Some(PtxStateSpace::Global);
        let result = emit_instruction(&instr);
        assert!(result.contains("atom"));
    }

    #[test]
    fn test_emit_fallback_op() {
        // Use an op that isn't handled by any category
        let instr = make_instr(PtxOp::Exit);
        let result = emit_instruction(&instr);
        assert!(result.contains("exit"));
    }

    #[test]
    fn test_emit_multiple_dsts() {
        let mut instr = make_instr(PtxOp::Ld);
        instr.dsts = vec![
            Operand::Reg(vreg(0, PtxType::F32)),
            Operand::Reg(vreg(1, PtxType::F32)),
        ];
        instr.srcs = vec![Operand::Reg(vreg(2, PtxType::U64))];
        let result = emit_instruction(&instr);
        assert!(result.contains("{"));
        assert!(result.contains("}"));
    }

    #[test]
    fn test_emit_multiple_srcs() {
        let mut instr = make_instr(PtxOp::Fma);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
        instr.srcs = vec![
            Operand::Reg(vreg(1, PtxType::F32)),
            Operand::Reg(vreg(2, PtxType::F32)),
            Operand::Reg(vreg(3, PtxType::F32)),
        ];
        let result = emit_instruction(&instr);
        assert!(result.contains("fma"));
        assert!(result.matches(",").count() >= 2);
    }

    // === write_instruction tests ===

    #[test]
    fn test_write_label() {
        let mut instr = make_instr(PtxOp::Add);
        instr.label = Some("done:".to_string());
        let mut out = String::new();
        write_instruction(&instr, &mut out);
        assert_eq!(out, "done:\n");
    }

    #[test]
    fn test_write_with_predicate() {
        let mut instr = make_instr(PtxOp::Add);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
        instr.srcs = vec![
            Operand::Reg(vreg(1, PtxType::F32)),
            Operand::Reg(vreg(2, PtxType::F32)),
        ];
        instr.predicate = Some(Predicate {
            reg: vreg(10, PtxType::Pred),
            negated: false,
        });
        let mut out = String::new();
        write_instruction(&instr, &mut out);
        assert!(out.contains("@"));
    }

    #[test]
    fn test_write_add_instruction() {
        let mut instr = make_instr(PtxOp::Add);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
        instr.srcs = vec![
            Operand::Reg(vreg(1, PtxType::F32)),
            Operand::Reg(vreg(2, PtxType::F32)),
        ];
        let mut out = String::new();
        write_instruction(&instr, &mut out);
        assert!(out.contains("add"));
        assert!(out.contains(".f32"));
    }

    #[test]
    fn test_write_ld_global() {
        let mut instr = make_instr(PtxOp::Ld);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
        instr.srcs = vec![Operand::Reg(vreg(1, PtxType::U64))];
        instr.state_space = Some(PtxStateSpace::Global);
        let mut out = String::new();
        write_instruction(&instr, &mut out);
        assert!(out.contains("ld.global"));
    }

    #[test]
    fn test_write_ld_shared() {
        let mut instr = make_instr(PtxOp::Ld);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
        instr.srcs = vec![Operand::Reg(vreg(1, PtxType::U64))];
        instr.state_space = Some(PtxStateSpace::Shared);
        let mut out = String::new();
        write_instruction(&instr, &mut out);
        assert!(out.contains("ld.shared"));
    }

    #[test]
    fn test_write_setp() {
        let mut instr = make_instr(PtxOp::Setp);
        instr.ty = PtxType::F32;
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::Pred)));
        instr.srcs = vec![
            Operand::Reg(vreg(1, PtxType::F32)),
            Operand::Reg(vreg(2, PtxType::F32)),
        ];
        let mut out = String::new();
        write_instruction(&instr, &mut out);
        assert!(out.contains("setp"));
    }

    #[test]
    fn test_write_shfl() {
        let mut instr = make_instr(PtxOp::ShflDown);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::F32)));
        instr.srcs = vec![
            Operand::ImmU64(0xFFFFFFFF),
            Operand::Reg(vreg(1, PtxType::F32)),
            Operand::ImmU64(1),
            Operand::ImmU64(0x1F),
        ];
        let mut out = String::new();
        write_instruction(&instr, &mut out);
        assert!(out.contains("shfl"));
    }

    #[test]
    fn test_write_fallback_op() {
        let instr = make_instr(PtxOp::Exit);
        let mut out = String::new();
        write_instruction(&instr, &mut out);
        assert!(out.contains("exit"));
    }

    #[test]
    fn test_write_multiple_dsts() {
        let mut instr = make_instr(PtxOp::Ld);
        instr.dsts = vec![
            Operand::Reg(vreg(0, PtxType::F32)),
            Operand::Reg(vreg(1, PtxType::F32)),
        ];
        instr.srcs = vec![Operand::Reg(vreg(2, PtxType::U64))];
        let mut out = String::new();
        write_instruction(&instr, &mut out);
        assert!(out.contains("{"));
        assert!(out.contains("}"));
    }

    #[test]
    fn test_write_atom_global() {
        let mut instr = make_instr(PtxOp::AtomAdd);
        instr.dst = Some(Operand::Reg(vreg(0, PtxType::U32)));
        instr.srcs = vec![Operand::Reg(vreg(1, PtxType::U64)), Operand::ImmU64(1)];
        instr.state_space = Some(PtxStateSpace::Global);
        let mut out = String::new();
        write_instruction(&instr, &mut out);
        assert!(out.contains("atom"));
    }

    // === should_skip_type_suffix tests ===

    #[test]
    fn test_skip_type_for_shfl() {
        let instr = make_instr(PtxOp::ShflDown);
        assert!(should_skip_type_suffix(&instr));
    }

    #[test]
    fn test_skip_type_for_cvta() {
        let instr = make_instr(PtxOp::Cvta);
        assert!(should_skip_type_suffix(&instr));
    }

    #[test]
    fn test_no_skip_type_for_add() {
        let instr = make_instr(PtxOp::Add);
        assert!(!should_skip_type_suffix(&instr));
    }

    #[test]
    fn test_wide_mul_skips_type() {
        let mut instr = make_instr(PtxOp::Mul);
        instr.ty = PtxType::U64;
        instr.srcs = vec![
            Operand::Reg(vreg(0, PtxType::U32)),
            Operand::Reg(vreg(1, PtxType::U32)),
        ];
        assert!(should_skip_type_suffix(&instr));
    }

    #[test]
    fn test_regular_mul_no_skip() {
        let mut instr = make_instr(PtxOp::Mul);
        instr.ty = PtxType::U64;
        instr.srcs = vec![
            Operand::Reg(vreg(0, PtxType::U64)),
            Operand::Reg(vreg(1, PtxType::U64)),
        ];
        assert!(!should_skip_type_suffix(&instr));
    }
}
