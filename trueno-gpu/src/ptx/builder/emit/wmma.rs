//! WMMA (Tensor Core) operation emission
//!
//! Handles: WmmaLoadA, WmmaLoadB, WmmaLoadC, WmmaMma, WmmaStoreD

use crate::ptx::instructions::{PtxInstruction, PtxOp};
use super::operand::emit_operand;

/// Emit WMMA load instruction with proper register list format
/// Format: wmma.load.{a|b|c}.sync.aligned.m16n16k16.{layout}.{type} {regs}, [ptr], stride
pub(crate) fn emit_wmma_load(prefix: String, instr: &PtxInstruction, matrix: &str) -> String {
    let mut s = prefix;

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
pub(crate) fn emit_wmma_mma(prefix: String, instr: &PtxInstruction) -> String {
    let mut s = prefix;

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
pub(crate) fn emit_wmma_store(prefix: String, instr: &PtxInstruction) -> String {
    let mut s = prefix;

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

/// Check if this is a WMMA operation
pub(crate) fn is_wmma_op(op: &PtxOp) -> bool {
    matches!(
        op,
        PtxOp::WmmaLoadA
            | PtxOp::WmmaLoadB
            | PtxOp::WmmaLoadC
            | PtxOp::WmmaMma
            | PtxOp::WmmaStoreD
    )
}
