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

// Re-export operand functions for external use (wmma.rs, instruction_emission tests)
#[allow(unused_imports)]
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

    emit_opcode(instr, &mut s);

    // Type suffix (skip for certain ops)
    if !should_skip_type_suffix(instr) {
        s.push_str(instr.ty.to_ptx_string());
    }

    s.push(' ');

    write_destinations(instr, &mut s);
    write_sources(instr, &mut s);

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

    // DP4A opcodes already include their type qualifiers (e.g., "dp4a.u32.s32")
    let is_dp4a = matches!(instr.op, PtxOp::Dp4a | PtxOp::Dp4aUS | PtxOp::Dp4aS32);

    memory::skip_type_for_memory_op(&instr.op)
        || warp::skip_type_for_warp_op(&instr.op)
        || wmma::is_wmma_op(&instr.op)
        || is_wide_mul_from_u32
        || is_dp4a
}

/// Emit the opcode portion of an instruction (shared by both emit and write paths)
fn emit_opcode(instr: &PtxInstruction, out: &mut String) {
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
}

/// Write destination operands directly to buffer
fn write_destinations(instr: &PtxInstruction, out: &mut String) {
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
}

/// Write source operands with proper memory addressing
fn write_sources(instr: &PtxInstruction, out: &mut String) {
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
}

/// Write the predicate prefix and return whether a prefix was written
fn write_predicate_prefix(instr: &PtxInstruction, out: &mut String) {
    if let Some(pred) = &instr.predicate {
        let neg = if pred.negated { "!" } else { "" };
        let _ = write!(out, "    @{}{} ", neg, pred.reg);
    } else {
        out.push_str("    ");
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

    write_predicate_prefix(instr, out);

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

    emit_opcode(instr, out);

    // Type suffix
    if !should_skip_type_suffix(instr) {
        out.push_str(instr.ty.to_ptx_string());
    }

    out.push(' ');
    write_destinations(instr, out);
    write_sources(instr, out);
    out.push_str(";\n");
}


#[cfg(test)]
mod tests;
