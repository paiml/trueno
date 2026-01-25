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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::instructions::Operand;
    use crate::ptx::registers::VirtualReg;
    use crate::ptx::types::PtxType;

    fn make_instr(op: PtxOp, ty: PtxType) -> PtxInstruction {
        PtxInstruction {
            op,
            ty,
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

    // === is_wmma_op tests ===

    #[test]
    fn test_is_wmma_op_all_variants() {
        assert!(is_wmma_op(&PtxOp::WmmaLoadA));
        assert!(is_wmma_op(&PtxOp::WmmaLoadB));
        assert!(is_wmma_op(&PtxOp::WmmaLoadC));
        assert!(is_wmma_op(&PtxOp::WmmaMma));
        assert!(is_wmma_op(&PtxOp::WmmaStoreD));
    }

    #[test]
    fn test_is_wmma_op_non_wmma() {
        assert!(!is_wmma_op(&PtxOp::Add));
        assert!(!is_wmma_op(&PtxOp::Ld));
        assert!(!is_wmma_op(&PtxOp::Mul));
        assert!(!is_wmma_op(&PtxOp::Bra));
    }

    // === emit_wmma_load tests ===

    #[test]
    fn test_emit_wmma_load_a_default() {
        let mut instr = make_instr(PtxOp::WmmaLoadA, PtxType::F16);
        instr.dsts = (0..8).map(|i| Operand::Reg(vreg(i, PtxType::F16))).collect();
        instr.srcs = vec![
            Operand::Reg(vreg(100, PtxType::U64)),
            Operand::ImmU64(16),
        ];
        let result = emit_wmma_load("    ".to_string(), &instr, "a");
        assert!(result.contains("wmma.load.a.sync.aligned"));
        assert!(result.contains(".m16n16k16.row.f16"));
    }

    #[test]
    fn test_emit_wmma_load_b_with_label() {
        let mut instr = make_instr(PtxOp::WmmaLoadB, PtxType::F16);
        instr.label = Some("m16n16k16.col.f16.stride.32".to_string());
        instr.dsts = (0..8).map(|i| Operand::Reg(vreg(i, PtxType::F16))).collect();
        instr.srcs = vec![Operand::Reg(vreg(100, PtxType::U64))];
        let result = emit_wmma_load("    ".to_string(), &instr, "b");
        assert!(result.contains("wmma.load.b.sync.aligned"));
        assert!(result.contains(".m16n16k16.col.f16"));
        assert!(result.contains("32"));
    }

    #[test]
    fn test_emit_wmma_load_c() {
        let mut instr = make_instr(PtxOp::WmmaLoadC, PtxType::F32);
        instr.label = Some("m16n16k16.row.f32".to_string());
        instr.dsts = (0..8).map(|i| Operand::Reg(vreg(i, PtxType::F32))).collect();
        instr.srcs = vec![
            Operand::Reg(vreg(100, PtxType::U64)),
            Operand::ImmU64(16),
        ];
        let result = emit_wmma_load("    ".to_string(), &instr, "c");
        assert!(result.contains("wmma.load.c.sync.aligned"));
        assert!(result.contains(".m16n16k16.row.f32"));
    }

    #[test]
    fn test_emit_wmma_load_partial_label() {
        // Label with fewer than 3 parts
        let mut instr = make_instr(PtxOp::WmmaLoadA, PtxType::F16);
        instr.label = Some("m16n16k16".to_string()); // Only 1 part
        instr.dsts = (0..8).map(|i| Operand::Reg(vreg(i, PtxType::F16))).collect();
        instr.srcs = vec![Operand::Reg(vreg(100, PtxType::U64))];
        let result = emit_wmma_load("    ".to_string(), &instr, "a");
        // Should fall back to default .m16n16k16.row.f16
        assert!(result.contains(".m16n16k16.row.f16"));
    }

    #[test]
    fn test_emit_wmma_load_no_srcs() {
        let mut instr = make_instr(PtxOp::WmmaLoadA, PtxType::F16);
        instr.dsts = (0..8).map(|i| Operand::Reg(vreg(i, PtxType::F16))).collect();
        // No sources
        let result = emit_wmma_load("    ".to_string(), &instr, "a");
        // Should still produce valid output with default stride
        assert!(result.contains("wmma.load.a"));
        assert!(result.contains("16"));
    }

    // === emit_wmma_mma tests ===

    #[test]
    fn test_emit_wmma_mma_default() {
        let mut instr = make_instr(PtxOp::WmmaMma, PtxType::F32);
        instr.dsts = (0..8).map(|i| Operand::Reg(vreg(i, PtxType::F32))).collect();
        // A (8) + B (8) + C (8) = 24 source registers
        instr.srcs = (0..24).map(|i| Operand::Reg(vreg(100 + i, PtxType::F16))).collect();
        let result = emit_wmma_mma("    ".to_string(), &instr);
        assert!(result.contains("wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32"));
    }

    #[test]
    fn test_emit_wmma_mma_with_label() {
        let mut instr = make_instr(PtxOp::WmmaMma, PtxType::F16);
        instr.label = Some("m8n8k4.row.row.f16.f16".to_string());
        instr.dsts = (0..4).map(|i| Operand::Reg(vreg(i, PtxType::F16))).collect();
        instr.srcs = (0..12).map(|i| Operand::Reg(vreg(100 + i, PtxType::F16))).collect();
        let result = emit_wmma_mma("    ".to_string(), &instr);
        assert!(result.contains("wmma.mma.sync.aligned.m8n8k4.row.row.f16.f16"));
    }

    #[test]
    fn test_emit_wmma_mma_partial_srcs() {
        // Fewer than 24 sources
        let mut instr = make_instr(PtxOp::WmmaMma, PtxType::F32);
        instr.dsts = (0..8).map(|i| Operand::Reg(vreg(i, PtxType::F32))).collect();
        instr.srcs = (0..16).map(|i| Operand::Reg(vreg(100 + i, PtxType::F16))).collect();
        let result = emit_wmma_mma("    ".to_string(), &instr);
        // Should handle fewer sources gracefully
        assert!(result.contains("wmma.mma.sync.aligned"));
    }

    // === emit_wmma_store tests ===

    #[test]
    fn test_emit_wmma_store_default() {
        let mut instr = make_instr(PtxOp::WmmaStoreD, PtxType::F32);
        // [ptr], {8 regs}, stride
        instr.srcs = std::iter::once(Operand::Reg(vreg(0, PtxType::U64)))
            .chain((1..9).map(|i| Operand::Reg(vreg(i, PtxType::F32))))
            .chain(std::iter::once(Operand::ImmU64(16)))
            .collect();
        let result = emit_wmma_store("    ".to_string(), &instr);
        assert!(result.contains("wmma.store.d.sync.aligned"));
        assert!(result.contains(".m16n16k16.row.f32"));
    }

    #[test]
    fn test_emit_wmma_store_with_label() {
        let mut instr = make_instr(PtxOp::WmmaStoreD, PtxType::F32);
        instr.label = Some("m16n16k16.col.f32.stride.32".to_string());
        instr.srcs = std::iter::once(Operand::Reg(vreg(0, PtxType::U64)))
            .chain((1..9).map(|i| Operand::Reg(vreg(i, PtxType::F32))))
            .chain(std::iter::once(Operand::ImmU64(32)))
            .collect();
        let result = emit_wmma_store("    ".to_string(), &instr);
        assert!(result.contains(".m16n16k16.col.f32"));
    }

    #[test]
    fn test_emit_wmma_store_partial_label() {
        // Label with fewer than 3 parts
        let mut instr = make_instr(PtxOp::WmmaStoreD, PtxType::F32);
        instr.label = Some("m8n8k4".to_string());
        instr.srcs = std::iter::once(Operand::Reg(vreg(0, PtxType::U64)))
            .chain((1..5).map(|i| Operand::Reg(vreg(i, PtxType::F32))))
            .chain(std::iter::once(Operand::ImmU64(16)))
            .collect();
        let result = emit_wmma_store("    ".to_string(), &instr);
        // Should fall back to default .m16n16k16.row.f32
        assert!(result.contains(".m16n16k16.row.f32"));
    }

    #[test]
    fn test_emit_wmma_store_stride_from_label() {
        // No sources at all, should get stride from label
        let mut instr = make_instr(PtxOp::WmmaStoreD, PtxType::F32);
        instr.label = Some("m16n16k16.row.f32.stride.64".to_string());
        instr.srcs = vec![];
        let result = emit_wmma_store("    ".to_string(), &instr);
        assert!(result.contains("64"));
    }

    #[test]
    fn test_emit_wmma_store_empty_srcs() {
        // Edge case: empty sources
        let mut instr = make_instr(PtxOp::WmmaStoreD, PtxType::F32);
        instr.srcs = vec![];
        let result = emit_wmma_store("    ".to_string(), &instr);
        // Should produce something, even if minimal
        assert!(result.contains("wmma.store.d"));
    }
}
