//! Operand emission utilities
//!
//! Shared operand formatting for all emission modules

use crate::ptx::instructions::Operand;
use std::fmt::Write;

/// Emit an operand as PTX string (allocating version)
pub(crate) fn emit_operand(op: &Operand) -> String {
    match op {
        Operand::Reg(vreg) => vreg.to_ptx_string(),
        Operand::SpecialReg(sreg) => sreg.to_ptx_string().to_string(),
        Operand::ImmI64(v) => v.to_string(),
        Operand::ImmU64(v) => v.to_string(),
        Operand::ImmF32(v) => format!("0F{:08X}", v.to_bits()),
        Operand::ImmF64(v) => format!("0D{:016X}", v.to_bits()),
        Operand::Param(name) => format!("[{}]", name),
        Operand::Addr { base, offset } => {
            if *offset == 0 {
                format!("[{}]", base.to_ptx_string())
            } else {
                format!("[{}+{}]", base.to_ptx_string(), offset)
            }
        }
        Operand::Label(name) => name.clone(),
    }
}

/// Emit shared memory operand with proper addressing syntax
pub(crate) fn emit_shared_mem_operand(op: &Operand) -> String {
    match op {
        Operand::Reg(vreg) => format!("[{}]", vreg.to_ptx_string()),
        Operand::Addr { base, offset } => {
            if *offset == 0 {
                format!("[{}]", base.to_ptx_string())
            } else {
                format!("[{}+{}]", base.to_ptx_string(), offset)
            }
        }
        _ => emit_operand(op),
    }
}

/// Emit global memory operand with proper [addr] syntax
pub(crate) fn emit_global_mem_operand(op: &Operand) -> String {
    match op {
        Operand::Reg(vreg) => format!("[{}]", vreg.to_ptx_string()),
        Operand::Addr { base, offset } => {
            if *offset == 0 {
                format!("[{}]", base.to_ptx_string())
            } else {
                format!("[{}+{}]", base.to_ptx_string(), offset)
            }
        }
        _ => emit_operand(op),
    }
}

/// Write operand directly to buffer (zero allocation)
#[inline]
pub(crate) fn write_operand(op: &Operand, out: &mut String) {
    match op {
        Operand::Reg(vreg) => {
            let _ = write!(out, "{}", vreg);
        }
        Operand::SpecialReg(sreg) => out.push_str(sreg.to_ptx_string()),
        Operand::ImmI64(v) => {
            let _ = write!(out, "{}", v);
        }
        Operand::ImmU64(v) => {
            let _ = write!(out, "{}", v);
        }
        Operand::ImmF32(v) => {
            let _ = write!(out, "0F{:08X}", v.to_bits());
        }
        Operand::ImmF64(v) => {
            let _ = write!(out, "0D{:016X}", v.to_bits());
        }
        Operand::Param(name) => {
            let _ = write!(out, "[{}]", name);
        }
        Operand::Addr { base, offset } => {
            if *offset == 0 {
                let _ = write!(out, "[{}]", base);
            } else {
                let _ = write!(out, "[{}+{}]", base, offset);
            }
        }
        Operand::Label(name) => out.push_str(name),
    }
}

/// Write memory operand with bracket syntax directly to buffer
#[inline]
pub(crate) fn write_mem_operand(op: &Operand, out: &mut String) {
    match op {
        Operand::Reg(vreg) => {
            let _ = write!(out, "[{}]", vreg);
        }
        Operand::Addr { base, offset } => {
            if *offset == 0 {
                let _ = write!(out, "[{}]", base);
            } else {
                let _ = write!(out, "[{}+{}]", base, offset);
            }
        }
        _ => write_operand(op, out),
    }
}
