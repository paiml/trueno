//! PTX Instruction Emission
//!
//! This module contains all PTX code generation functions that convert
//! PtxInstruction structs into PTX assembly text. Extracted from mod.rs
//! for PMAT-018 domain separation.
//!
//! ## Functions
//!
//! - `emit_instruction()` - Allocating version for single instruction
//! - `write_instruction()` - Zero-allocation version for bulk output
//! - `emit_operand()` / `write_operand()` - Operand formatting
//! - `emit_wmma_*()` - WMMA tensor core instruction formatting

use std::fmt::Write;

use super::super::instructions::{Operand, PtxInstruction, PtxOp};
use super::super::types::{PtxStateSpace, PtxType};

/// Emit a single instruction as PTX (allocating version)
#[allow(clippy::too_many_lines)]
pub(crate) fn emit_instruction(instr: &PtxInstruction) -> String {
    let mut s = String::new();

    // Handle labels
    if let Some(label) = &instr.label {
        if label.ends_with(':') {
            return format!("{}:\n", &label[..label.len() - 1]);
        }
    }

    // Predicate
    if let Some(pred) = &instr.predicate {
        let neg = if pred.negated { "!" } else { "" };
        s.push_str(&format!("    @{}{} ", neg, pred.reg.to_ptx_string()));
    } else {
        s.push_str("    ");
    }

    // Opcode
    match instr.op {
        PtxOp::Mov => s.push_str("mov"),
        PtxOp::Add => s.push_str("add"),
        PtxOp::Sub => s.push_str("sub"),
        PtxOp::Mul => {
            // Check for wide multiply (dest is 64-bit from 32-bit sources)
            // mul.wide.u32 produces u64 from u32 * u32
            // BUT if source operands are already u64, use mul.lo.u64 instead
            let is_wide_output = instr.ty == PtxType::U64 || instr.ty == PtxType::S64;
            let has_u64_source = instr.srcs.first().is_some_and(|src| {
                matches!(src, Operand::Reg(vreg) if vreg.ty() == PtxType::U64 || vreg.ty() == PtxType::S64)
            });

            if is_wide_output && !has_u64_source {
                // Wide multiply uses source type, not dest type
                let src_ty = if instr.ty == PtxType::U64 {
                    ".u32"
                } else {
                    ".s32"
                };
                s.push_str("mul.wide");
                s.push_str(src_ty);
            } else if is_wide_output && has_u64_source {
                // u64 * u64 -> u64: use mul.lo.u64
                s.push_str("mul.lo");
            } else if instr.ty.is_float() {
                // Floating point multiply (no .lo needed)
                s.push_str("mul");
            } else {
                // Integer multiply needs .lo to get low bits
                s.push_str("mul.lo");
            }
        }
        PtxOp::MadLo => s.push_str("mad.lo"),
        PtxOp::Div => {
            // Float div requires rounding mode, integer div doesn't
            if instr.ty.is_float() {
                s.push_str("div.rn");
            } else {
                s.push_str("div");
            }
        }
        PtxOp::Setp => {
            // Include comparison op from label
            let cmp = instr.label.as_deref().unwrap_or("eq");
            s.push_str(&format!("setp.{}", cmp));
        }
        PtxOp::Ld => {
            // No state space = generic addressing (for cvta-derived pointers)
            // With state space = specific space (.shared, .global, etc.)
            if let Some(ss) = instr.state_space {
                s.push_str(&format!("ld{}", ss.to_ptx_string()));
            } else {
                s.push_str("ld");
            }
        }
        PtxOp::LdVolatile => {
            // Volatile load - prevents compiler optimization of dependent loads
            // Used for F082 fix to break "Computed Address From Loaded Value" SASS optimization
            if let Some(ss) = instr.state_space {
                s.push_str(&format!("ld.volatile{}", ss.to_ptx_string()));
            } else {
                s.push_str("ld.volatile");
            }
        }
        PtxOp::LdParam => s.push_str("ld.param"),
        PtxOp::St => {
            // No state space = generic addressing (for cvta-derived pointers)
            // With state space = specific space (.shared, .global, etc.)
            if let Some(ss) = instr.state_space {
                s.push_str(&format!("st{}", ss.to_ptx_string()));
            } else {
                s.push_str("st");
            }
        }
        PtxOp::Bra => {
            if let Some(label) = &instr.label {
                return format!("{}bra {};\n", s, label);
            }
            s.push_str("bra");
        }
        PtxOp::Ret => return format!("{}ret;\n", s),
        PtxOp::Bar => {
            // bar.sync instruction needs barrier ID from label
            let barrier_id = instr.label.as_deref().unwrap_or("sync 0");
            return format!("{}bar.{};\n", s, barrier_id);
        }
        PtxOp::Cvt => {
            // cvt needs both destination and source types
            // Use explicit src_type if provided, otherwise infer from source operand
            let dst_ty = instr.ty.to_ptx_string();
            let src_ty = if let Some(st) = instr.src_type {
                st.to_ptx_string()
            } else if let Some(Operand::Reg(vreg)) = instr.srcs.first() {
                vreg.ty().to_ptx_string()
            } else {
                ".u32" // Default fallback
            };
            // Add rounding mode for float conversions (required by PTX ISA)
            // EXCEPTION: f16→f32 is exact and does NOT require rounding
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
                    || instr.srcs.first().is_some_and(
                        |src| matches!(src, Operand::Reg(vreg) if vreg.ty().is_float()),
                    ));
            let round = if needs_rounding {
                instr
                    .rounding
                    .as_ref()
                    .map_or(".rn", |r| r.to_ptx_string())
            } else {
                ""
            };
            s.push_str(&format!("cvt{}{}{}", round, dst_ty, src_ty));
        }
        PtxOp::Cvta => {
            // cvta.{space}.{size} d, a - convert state-space address a TO generic d
            // Format: cvta.shared.u64 %rd, smem;  (smem is a shared memory label)
            // PTX ISA: cvta.space converts space→generic, cvta.to.space converts generic→space
            let space = instr
                .state_space
                .map(|ss| ss.to_ptx_string())
                .unwrap_or(".shared");
            let ty = instr.ty.to_ptx_string();
            s.push_str(&format!("cvta{}{}", space, ty));
        }
        PtxOp::Fma => {
            // FMA requires rounding mode: fma.rn.f32
            let round = instr
                .rounding
                .as_ref()
                .map_or(".rn", |r| r.to_ptx_string());
            s.push_str(&format!("fma{}", round));
        }
        PtxOp::Dp4a => {
            // PAR-063: Dot product of 4-element byte vectors with accumulate
            // Format: dp4a.atype.btype d, a, b, c;
            // Uses U32 for unsigned quantized values (Q4K weights are unsigned)
            s.push_str("dp4a.u32.u32");
        }
        PtxOp::Dp4aUS => {
            // PAR-063-V3: Mixed unsigned/signed DP4A
            // For Q4K weights (unsigned 0-15) × quantized activations (signed)
            s.push_str("dp4a.u32.s32");
        }
        PtxOp::Dp4aS32 => {
            // PAR-063-V3: Fully signed DP4A
            s.push_str("dp4a.s32.s32");
        }
        PtxOp::ShflDown => {
            // sm_70+ requires shfl.sync.down with b32 type
            // Format: shfl.sync.down.b32 dst, src, delta, clamp, membermask;
            s.push_str("shfl.sync.down.b32");
        }
        PtxOp::ShflIdx => {
            // sm_70+ requires shfl.sync.idx with b32 type
            // Format: shfl.sync.idx.b32 dst, src, srcLane, width, membermask;
            // NOTE: width must be power of 2 (1, 2, 4, 8, 16, or 32)
            s.push_str("shfl.sync.idx.b32");
        }
        PtxOp::Ex2 => {
            // ex2 requires .approx modifier for f32
            s.push_str("ex2.approx");
        }
        PtxOp::Rsqrt => {
            // rsqrt requires .approx modifier for f32
            // PTX format: rsqrt.approx.f32 dst, src
            s.push_str("rsqrt.approx");
        }
        PtxOp::Rcp => {
            // rcp requires .approx modifier for f32
            // PTX format: rcp.approx.f32 dst, src
            s.push_str("rcp.approx");
        }
        PtxOp::WmmaLoadA => {
            // WMMA load A fragment: wmma.load.a.sync.aligned.{shape}.{layout}.{type} {dst...}, [ptr], stride
            // Label contains: "m16n16k16.{layout}.f16.stride.{stride}"
            // We need to extract shape/layout/type from label and format properly
            return emit_wmma_load(s, instr, "a");
        }
        PtxOp::WmmaLoadB => {
            // WMMA load B fragment
            return emit_wmma_load(s, instr, "b");
        }
        PtxOp::WmmaLoadC => {
            // WMMA load C fragment
            return emit_wmma_load(s, instr, "c");
        }
        PtxOp::WmmaMma => {
            // WMMA MMA: wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32 {d...}, {a...}, {b...}, {c...}
            return emit_wmma_mma(s, instr);
        }
        PtxOp::WmmaStoreD => {
            // WMMA store D: wmma.store.d.sync.aligned.m16n16k16.row.f32 [ptr], {src...}, stride
            return emit_wmma_store(s, instr);
        }
        PtxOp::AtomAdd => {
            // Atomic add: atom.global.add.u32 dst, [addr], val
            let space = instr
                .state_space
                .map(|ss| ss.to_ptx_string())
                .unwrap_or(".global");
            s.push_str(&format!("atom{}.add", space));
        }
        PtxOp::AtomMin => {
            let space = instr
                .state_space
                .map(|ss| ss.to_ptx_string())
                .unwrap_or(".global");
            s.push_str(&format!("atom{}.min", space));
        }
        PtxOp::AtomMax => {
            let space = instr
                .state_space
                .map(|ss| ss.to_ptx_string())
                .unwrap_or(".global");
            s.push_str(&format!("atom{}.max", space));
        }
        PtxOp::AtomExch => {
            let space = instr
                .state_space
                .map(|ss| ss.to_ptx_string())
                .unwrap_or(".global");
            s.push_str(&format!("atom{}.exch", space));
        }
        PtxOp::AtomCas => {
            let space = instr
                .state_space
                .map(|ss| ss.to_ptx_string())
                .unwrap_or(".global");
            s.push_str(&format!("atom{}.cas", space));
        }
        _ => s.push_str(&format!("{:?}", instr.op).to_lowercase()),
    }

    // Type suffix (skip for Cvt, wide Mul, Fma, ShflDown which handle types specially)
    // NOTE: mul.lo.u64 still needs .u64 suffix, only mul.wide.u32 skips it
    let is_wide_mul_from_u32 = instr.op == PtxOp::Mul
        && (instr.ty == PtxType::U64 || instr.ty == PtxType::S64)
        && !instr.srcs.first().is_some_and(|src| {
            matches!(src, Operand::Reg(vreg) if vreg.ty() == PtxType::U64 || vreg.ty() == PtxType::S64)
        });
    let skip_type_suffix = instr.op == PtxOp::Cvt
        || instr.op == PtxOp::Cvta
        || is_wide_mul_from_u32
        || instr.op == PtxOp::ShflDown
        || instr.op == PtxOp::ShflIdx
        || matches!(
            instr.op,
            PtxOp::WmmaLoadA
                | PtxOp::WmmaLoadB
                | PtxOp::WmmaLoadC
                | PtxOp::WmmaMma
                | PtxOp::WmmaStoreD
        );
    if !skip_type_suffix {
        s.push_str(instr.ty.to_ptx_string());
    }

    s.push(' ');

    // Destination(s) - handle vector destinations specially
    if !instr.dsts.is_empty() {
        // Vector destination: {%f1, %f2, %f3, %f4}
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

    // Sources - handle memory addressing specially for Ld/St and Atomic ops
    let is_memory_op = matches!(instr.op, PtxOp::Ld | PtxOp::LdVolatile | PtxOp::St);
    let is_atomic_op = matches!(
        instr.op,
        PtxOp::AtomAdd | PtxOp::AtomMin | PtxOp::AtomMax | PtxOp::AtomExch | PtxOp::AtomCas
    );
    let is_shared_mem = instr.state_space == Some(PtxStateSpace::Shared);
    let is_global_mem = instr.state_space == Some(PtxStateSpace::Global)
        || (is_memory_op && instr.state_space.is_none());

    for (i, src) in instr.srcs.iter().enumerate() {
        // For memory ops, first source (address) needs bracket format
        // For atomic ops, first source (address) also needs bracket format
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

    s.push_str(";\n");
    s
}

/// Emit shared memory operand with proper addressing syntax
/// For shared memory, we use direct address register (caller computes smem base + offset)
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

/// Emit WMMA load instruction with proper register list format
/// Format: wmma.load.{a|b|c}.sync.aligned.m16n16k16.{layout}.{type} {regs}, [ptr], stride
fn emit_wmma_load(mut s: String, instr: &PtxInstruction, matrix: &str) -> String {
    // Parse label to get layout, type, stride
    // Label format: "m16n16k16.{layout}.{type}.stride.{stride}"
    let label = instr
        .label
        .as_deref()
        .unwrap_or("m16n16k16.row.f16.stride.16");
    let parts: Vec<&str> = label.split('.').collect();

    // Build instruction opcode
    s.push_str(&format!("wmma.load.{}.sync.aligned", matrix));

    // Add shape, layout, type from label (e.g., "m16n16k16.row.f16")
    if parts.len() >= 3 {
        s.push('.');
        s.push_str(parts[0]); // m16n16k16
        s.push('.');
        s.push_str(parts[1]); // row/col
        s.push('.');
        s.push_str(parts[2]); // f16/f32
    } else {
        s.push_str(".m16n16k16.row.f16");
    }

    s.push(' ');

    // Destination registers: {%r0, %r1, ..., %r7}
    s.push('{');
    for (i, dst) in instr.dsts.iter().enumerate() {
        s.push_str(&emit_operand(dst));
        if i < instr.dsts.len() - 1 {
            s.push_str(", ");
        }
    }
    s.push_str("}, ");

    // Source: [ptr]
    if let Some(src) = instr.srcs.first() {
        s.push('[');
        s.push_str(&emit_operand(src));
        s.push_str("], ");
    }

    // Stride
    if let Some(stride) = instr.srcs.get(1) {
        s.push_str(&emit_operand(stride));
    } else {
        // Extract stride from label (last part after "stride.")
        if let Some(stride_pos) = label.find("stride.") {
            s.push_str(&label[stride_pos + 7..]);
        } else {
            s.push_str("16");
        }
    }

    s.push_str(";\n");
    s
}

/// Emit WMMA MMA instruction with proper register list format
/// Format: wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32 {d}, {a}, {b}, {c}
fn emit_wmma_mma(mut s: String, instr: &PtxInstruction) -> String {
    // Label format: "m16n16k16.row.col.f32.f32"
    let label = instr
        .label
        .as_deref()
        .unwrap_or("m16n16k16.row.col.f32.f32");

    s.push_str("wmma.mma.sync.aligned.");
    s.push_str(label);
    s.push(' ');

    // D registers (first 8 of dsts)
    s.push('{');
    for (i, dst) in instr.dsts.iter().enumerate() {
        s.push_str(&emit_operand(dst));
        if i < instr.dsts.len() - 1 {
            s.push_str(", ");
        }
    }
    s.push_str("}, ");

    // A, B, C registers (each 8 registers from srcs)
    // Total srcs = 24 (8 A + 8 B + 8 C)
    let groups = [
        (0, 8),   // A
        (8, 16),  // B
        (16, 24), // C
    ];

    for (start, end) in groups {
        s.push('{');
        for i in start..end.min(instr.srcs.len()) {
            s.push_str(&emit_operand(&instr.srcs[i]));
            if i < end.min(instr.srcs.len()) - 1 {
                s.push_str(", ");
            }
        }
        s.push('}');
        if end < 24 && end <= instr.srcs.len() {
            s.push_str(", ");
        }
    }

    s.push_str(";\n");
    s
}

/// Emit WMMA store instruction with proper format
/// Format: wmma.store.d.sync.aligned.m16n16k16.{layout}.{type} [ptr], {regs}, stride
fn emit_wmma_store(mut s: String, instr: &PtxInstruction) -> String {
    // Label format: "m16n16k16.{layout}.{type}.stride.{stride}"
    let label = instr
        .label
        .as_deref()
        .unwrap_or("m16n16k16.row.f32.stride.16");
    let parts: Vec<&str> = label.split('.').collect();

    s.push_str("wmma.store.d.sync.aligned");

    // Add shape, layout, type from label
    if parts.len() >= 3 {
        s.push('.');
        s.push_str(parts[0]); // m16n16k16
        s.push('.');
        s.push_str(parts[1]); // row
        s.push('.');
        s.push_str(parts[2]); // f32
    } else {
        s.push_str(".m16n16k16.row.f32");
    }

    s.push(' ');

    // [ptr]
    if let Some(src) = instr.srcs.first() {
        s.push('[');
        s.push_str(&emit_operand(src));
        s.push_str("], ");
    }

    // {regs} - D fragment (srcs 1-8)
    s.push('{');
    let frag_end = instr.srcs.len().saturating_sub(1).min(9);
    for i in 1..frag_end {
        s.push_str(&emit_operand(&instr.srcs[i]));
        if i < frag_end - 1 {
            s.push_str(", ");
        }
    }
    s.push_str("}, ");

    // Stride (last src)
    if let Some(stride) = instr.srcs.last() {
        s.push_str(&emit_operand(stride));
    } else if let Some(stride_pos) = label.find("stride.") {
        s.push_str(&label[stride_pos + 7..]);
    } else {
        s.push_str("16");
    }

    s.push_str(";\n");
    s
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

/// Emit an operand (allocating version)
pub(crate) fn emit_operand(op: &Operand) -> String {
    match op {
        Operand::Reg(vreg) => vreg.to_ptx_string(),
        Operand::SpecialReg(sreg) => sreg.to_ptx_string().to_string(),
        Operand::ImmI64(v) => v.to_string(),
        Operand::ImmU64(v) => v.to_string(),
        Operand::ImmF32(v) => emit_f32_literal(*v),
        Operand::ImmF64(v) => emit_f64_literal(*v),
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

/// Emit f32 literal in PTX hex format (0Fxxxxxxxx)
fn emit_f32_literal(v: f32) -> String {
    let bits = v.to_bits();
    format!("0F{:08X}", bits)
}

/// Emit f64 literal in PTX hex format (0Dxxxxxxxxxxxxxxxx)
fn emit_f64_literal(v: f64) -> String {
    let bits = v.to_bits();
    format!("0D{:016X}", bits)
}

/// Write a single instruction directly to a String buffer (zero intermediate allocations)
///
/// This is more efficient than `emit_instruction()` for building large PTX output
/// as it avoids allocating a new String for each instruction.
#[allow(clippy::too_many_lines)]
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

    // Opcode
    match instr.op {
        PtxOp::Mov => out.push_str("mov"),
        PtxOp::Add => out.push_str("add"),
        PtxOp::Sub => out.push_str("sub"),
        PtxOp::Mul => {
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
                out.push_str("mul.wide");
                out.push_str(src_ty);
            } else if is_wide_output && has_u64_source {
                out.push_str("mul.lo");
            } else if instr.ty.is_float() {
                out.push_str("mul");
            } else {
                out.push_str("mul.lo");
            }
        }
        PtxOp::MadLo => out.push_str("mad.lo"),
        PtxOp::Div => {
            if instr.ty.is_float() {
                out.push_str("div.rn");
            } else {
                out.push_str("div");
            }
        }
        PtxOp::Setp => {
            let cmp = instr.label.as_deref().unwrap_or("eq");
            let _ = write!(out, "setp.{}", cmp);
        }
        PtxOp::Ld => {
            // No state space = generic addressing (for cvta-derived pointers)
            if let Some(ss) = instr.state_space {
                out.push_str("ld");
                out.push_str(ss.to_ptx_string());
            } else {
                out.push_str("ld");
            }
        }
        PtxOp::LdVolatile => {
            // Volatile load - prevents compiler optimization of dependent loads
            if let Some(ss) = instr.state_space {
                out.push_str("ld.volatile");
                out.push_str(ss.to_ptx_string());
            } else {
                out.push_str("ld.volatile");
            }
        }
        PtxOp::LdParam => out.push_str("ld.param"),
        PtxOp::St => {
            // No state space = generic addressing (for cvta-derived pointers)
            if let Some(ss) = instr.state_space {
                out.push_str("st");
                out.push_str(ss.to_ptx_string());
            } else {
                out.push_str("st");
            }
        }
        PtxOp::Bra => {
            if let Some(label) = &instr.label {
                let _ = writeln!(out, "bra {};", label);
                return;
            }
            out.push_str("bra");
        }
        PtxOp::Ret => {
            out.push_str("ret;\n");
            return;
        }
        PtxOp::Bar => {
            let barrier_id = instr.label.as_deref().unwrap_or("sync 0");
            let _ = writeln!(out, "bar.{};", barrier_id);
            return;
        }
        PtxOp::MemBar => {
            // Memory fence: membar.{scope}; where scope is cta, gl, or sys
            let scope = instr.label.as_deref().unwrap_or("cta");
            let _ = writeln!(out, "membar.{};", scope);
            return;
        }
        PtxOp::Cvt => {
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
                    || instr.srcs.first().is_some_and(
                        |src| matches!(src, Operand::Reg(vreg) if vreg.ty().is_float()),
                    ));
            let round = if needs_rounding {
                instr
                    .rounding
                    .as_ref()
                    .map_or(".rn", |r| r.to_ptx_string())
            } else {
                ""
            };
            out.push_str("cvt");
            out.push_str(round);
            out.push_str(dst_ty);
            out.push_str(src_ty);
        }
        PtxOp::Cvta => {
            // cvta.{space}.{size} d, a - convert state-space address a TO generic d
            // PTX ISA: cvta.space converts space→generic, cvta.to.space converts generic→space
            let space = instr
                .state_space
                .map(|ss| ss.to_ptx_string())
                .unwrap_or(".shared");
            let ty = instr.ty.to_ptx_string();
            out.push_str("cvta");
            out.push_str(space);
            out.push_str(ty);
        }
        PtxOp::Fma => {
            let round = instr
                .rounding
                .as_ref()
                .map_or(".rn", |r| r.to_ptx_string());
            out.push_str("fma");
            out.push_str(round);
        }
        PtxOp::Dp4a => out.push_str("dp4a.u32.u32"),
        PtxOp::Dp4aUS => out.push_str("dp4a.u32.s32"),
        PtxOp::Dp4aS32 => out.push_str("dp4a.s32.s32"),
        PtxOp::ShflDown => out.push_str("shfl.sync.down.b32"),
        PtxOp::ShflIdx => out.push_str("shfl.sync.idx.b32"),
        // KF-002: Warp vote and bit manipulation
        PtxOp::VoteBallot | PtxOp::Vote => out.push_str("vote.sync.ballot.b32"),
        PtxOp::Popc => out.push_str("popc"),
        PtxOp::Bfind => out.push_str("bfind"),
        PtxOp::Clz => out.push_str("clz"),
        PtxOp::Bfe => out.push_str("bfe"),
        PtxOp::Bfi => out.push_str("bfi"),
        PtxOp::Ex2 => out.push_str("ex2.approx"),
        PtxOp::Rsqrt => out.push_str("rsqrt.approx"),
        PtxOp::Rcp => out.push_str("rcp.approx"),
        // PAR-060: Sin/Cos for RoPE kernel
        PtxOp::Sin => out.push_str("sin.approx"),
        PtxOp::Cos => out.push_str("cos.approx"),
        PtxOp::Neg => out.push_str("neg"),
        // Atomic operations
        PtxOp::AtomAdd => {
            let space = instr
                .state_space
                .map(|ss| ss.to_ptx_string())
                .unwrap_or(".global");
            let _ = write!(out, "atom{}.add", space);
        }
        PtxOp::AtomMin => {
            let space = instr
                .state_space
                .map(|ss| ss.to_ptx_string())
                .unwrap_or(".global");
            let _ = write!(out, "atom{}.min", space);
        }
        PtxOp::AtomMax => {
            let space = instr
                .state_space
                .map(|ss| ss.to_ptx_string())
                .unwrap_or(".global");
            let _ = write!(out, "atom{}.max", space);
        }
        PtxOp::AtomExch => {
            let space = instr
                .state_space
                .map(|ss| ss.to_ptx_string())
                .unwrap_or(".global");
            let _ = write!(out, "atom{}.exch", space);
        }
        PtxOp::AtomCas => {
            let space = instr
                .state_space
                .map(|ss| ss.to_ptx_string())
                .unwrap_or(".global");
            let _ = write!(out, "atom{}.cas", space);
        }
        // WMMA ops use the existing emit functions for now (complex formatting)
        PtxOp::WmmaLoadA
        | PtxOp::WmmaLoadB
        | PtxOp::WmmaLoadC
        | PtxOp::WmmaMma
        | PtxOp::WmmaStoreD => {
            // Fall back to emit_instruction for complex WMMA ops
            out.push_str(&emit_instruction(instr));
            return;
        }
        _ => {
            let _ = write!(out, "{:?}", instr.op);
            // Convert to lowercase in-place would require unsafe, just use format
            let op_str = format!("{:?}", instr.op).to_lowercase();
            out.truncate(out.len() - format!("{:?}", instr.op).len());
            out.push_str(&op_str);
        }
    }

    // Type suffix
    let is_wide_mul_from_u32 = instr.op == PtxOp::Mul
        && (instr.ty == PtxType::U64 || instr.ty == PtxType::S64)
        && !instr.srcs.first().is_some_and(|src| {
            matches!(src, Operand::Reg(vreg) if vreg.ty() == PtxType::U64 || vreg.ty() == PtxType::S64)
        });
    let skip_type_suffix = instr.op == PtxOp::Cvt
        || instr.op == PtxOp::Cvta
        || is_wide_mul_from_u32
        || instr.op == PtxOp::ShflDown
        || instr.op == PtxOp::ShflIdx
        || instr.op == PtxOp::Vote // vote.sync.ballot has .b32 built-in
        || instr.op == PtxOp::VoteBallot
        || matches!(
            instr.op,
            PtxOp::WmmaLoadA
                | PtxOp::WmmaLoadB
                | PtxOp::WmmaLoadC
                | PtxOp::WmmaMma
                | PtxOp::WmmaStoreD
        );
    if !skip_type_suffix {
        out.push_str(instr.ty.to_ptx_string());
    }

    out.push(' ');

    // Destination(s)
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
        // For memory and atomic ops, first source (address) needs bracket format
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

/// Write operand directly to buffer (zero allocation)
#[inline]
pub(super) fn write_operand(op: &Operand, out: &mut String) {
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
pub(super) fn write_mem_operand(op: &Operand, out: &mut String) {
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
