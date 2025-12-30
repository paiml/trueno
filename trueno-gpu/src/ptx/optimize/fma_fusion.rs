//! FMA Fusion Optimization Pass
//!
//! Detects `mul` + `add` patterns and fuses them to single `fma` instructions.
//!
//! ## Pattern Detection
//!
//! ```text
//! mul.f32 %f1, %a, %b    ; %f1 = a * b
//! add.f32 %f2, %f1, %c   ; %f2 = %f1 + c = a * b + c
//! ```
//!
//! Becomes:
//!
//! ```text
//! fma.rn.f32 %f2, %a, %b, %c  ; %f2 = a * b + c (single instruction)
//! ```
//!
//! ## Requirements for Fusion
//!
//! 1. `mul` result must have exactly one use (in the `add`)
//! 2. `mul` and `add` must have compatible rounding modes
//! 3. Both instructions must be f32 or f64 type
//!
//! ## Academic Foundation
//!
//! Based on Click & Paleczny (1995) SSA pattern matching for peephole optimization.
//! cuda-tile-behavior.md: Section 3.5, Falsification tests #16-30

use std::collections::HashMap;

use super::super::instructions::{Operand, PtxInstruction, PtxOp, RoundingMode};
use super::super::registers::VirtualReg;
use super::super::types::PtxType;

/// Apply FMA fusion pass to instruction sequence.
///
/// # Arguments
///
/// * `instructions` - Input instruction sequence
///
/// # Returns
///
/// Optimized instruction sequence with mul+add fused to fma
///
/// # cuda-tile-behavior.md References
///
/// - Falsification test #16: FMA reduces instruction count by ~33%
/// - Falsification test #17: FMA improves numerical accuracy
/// - Falsification test #18: Single-use detection prevents incorrect fusion
#[must_use]
pub fn pass(instructions: Vec<PtxInstruction>) -> Vec<PtxInstruction> {
    if instructions.is_empty() {
        return instructions;
    }

    // Build use-def chains: for each virtual register, track which instruction defines it
    // and how many times it's used
    let use_counts = count_register_uses(&instructions);
    let definitions = build_def_map(&instructions);

    // Phase 1: Find all fusion pairs (mul_idx -> fma replacement for add_idx)
    let mut fused_muls: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut fma_replacements: HashMap<usize, PtxInstruction> = HashMap::new();

    for (add_idx, _) in instructions.iter().enumerate() {
        if let Some((fma, mul_idx)) =
            try_fuse_mul_add(add_idx, &instructions, &use_counts, &definitions)
        {
            fused_muls.insert(mul_idx);
            fma_replacements.insert(add_idx, fma);
        }
    }

    // Phase 2: Emit optimized instruction sequence
    let mut result = Vec::with_capacity(instructions.len() - fused_muls.len());

    for (i, instr) in instructions.iter().enumerate() {
        // Skip mul instructions that were fused into FMAs
        if fused_muls.contains(&i) {
            continue;
        }

        // Replace add instructions with their FMA equivalents
        if let Some(fma) = fma_replacements.get(&i) {
            result.push(fma.clone());
        } else {
            result.push(instr.clone());
        }
    }

    result
}

/// Count how many times each virtual register is used as a source operand.
fn count_register_uses(instructions: &[PtxInstruction]) -> HashMap<VirtualReg, usize> {
    let mut counts = HashMap::new();

    for instr in instructions {
        for src in &instr.srcs {
            if let Operand::Reg(reg) = src {
                *counts.entry(*reg).or_insert(0) += 1;
            }
        }
        if let Some(Operand::Reg(reg)) = &instr.predicate.as_ref().map(|p| Operand::Reg(p.reg)) {
            *counts.entry(*reg).or_insert(0) += 1;
        }
    }

    counts
}

/// Build a map from virtual register to the instruction index that defines it.
fn build_def_map(instructions: &[PtxInstruction]) -> HashMap<VirtualReg, usize> {
    let mut defs = HashMap::new();

    for (i, instr) in instructions.iter().enumerate() {
        if let Some(Operand::Reg(reg)) = &instr.dst {
            defs.insert(*reg, i);
        }
    }

    defs
}

/// Try to fuse an add instruction with its defining mul.
///
/// Returns the fused FMA instruction and the index of the mul definition if fusion is possible.
fn try_fuse_mul_add(
    add_idx: usize,
    instructions: &[PtxInstruction],
    use_counts: &HashMap<VirtualReg, usize>,
    definitions: &HashMap<VirtualReg, usize>,
) -> Option<(PtxInstruction, usize)> {
    let add_instr = &instructions[add_idx];

    // Only fuse add operations
    if !matches!(add_instr.op, PtxOp::Add) {
        return None;
    }

    // Only fuse floating-point types
    if !matches!(add_instr.ty, PtxType::F32 | PtxType::F64) {
        return None;
    }

    // Check each source operand of the add to see if it's a mul result
    for (src_idx, src) in add_instr.srcs.iter().enumerate() {
        if let Operand::Reg(mul_result) = src {
            // Check if this register has exactly one use
            if use_counts.get(mul_result) != Some(&1) {
                continue;
            }

            // Find the defining instruction
            if let Some(&def_idx) = definitions.get(mul_result) {
                let mul_instr = &instructions[def_idx];

                // Must be a mul instruction
                if !matches!(mul_instr.op, PtxOp::Mul) {
                    continue;
                }

                // Must have same type
                if mul_instr.ty != add_instr.ty {
                    continue;
                }

                // Check rounding mode compatibility
                if !rounding_modes_compatible(
                    mul_instr.rounding.as_ref(),
                    add_instr.rounding.as_ref(),
                ) {
                    continue;
                }

                // Get the other operand of the add (the 'c' in a*b+c)
                let other_src = if src_idx == 0 {
                    add_instr.srcs.get(1)?
                } else {
                    add_instr.srcs.first()?
                };

                // Get mul operands (a and b)
                if mul_instr.srcs.len() < 2 {
                    continue;
                }
                let a = mul_instr.srcs.first()?;
                let b = mul_instr.srcs.get(1)?;

                // Create FMA instruction: dst = a * b + c
                let fma = PtxInstruction::new(PtxOp::Fma, add_instr.ty.clone())
                    .dst(add_instr.dst.clone()?)
                    .src(a.clone())
                    .src(b.clone())
                    .src(other_src.clone())
                    .rounding(mul_instr.rounding.unwrap_or(RoundingMode::Rn));

                return Some((fma, def_idx));
            }
        }
    }

    None
}

/// Check if two rounding modes are compatible for fusion.
///
/// Fusion is allowed if:
/// - Both are None (default rounding)
/// - Both have the same explicit mode
/// - One is None and the other is Rn (default)
fn rounding_modes_compatible(a: Option<&RoundingMode>, b: Option<&RoundingMode>) -> bool {
    match (a, b) {
        (None | Some(RoundingMode::Rn), None) | (None, Some(RoundingMode::Rn)) => true,
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a simple mul instruction
    fn make_mul(dst: VirtualReg, a: VirtualReg, b: VirtualReg) -> PtxInstruction {
        PtxInstruction::new(PtxOp::Mul, PtxType::F32)
            .dst(Operand::Reg(dst))
            .src(Operand::Reg(a))
            .src(Operand::Reg(b))
    }

    // Helper to create a simple add instruction
    fn make_add(dst: VirtualReg, a: VirtualReg, b: VirtualReg) -> PtxInstruction {
        PtxInstruction::new(PtxOp::Add, PtxType::F32)
            .dst(Operand::Reg(dst))
            .src(Operand::Reg(a))
            .src(Operand::Reg(b))
    }

    fn make_vreg(id: u32, ty: PtxType) -> VirtualReg {
        VirtualReg::new(id, ty)
    }

    // cuda-tile-behavior.md: Falsification test #16
    #[test]
    fn test_fma_reduces_instruction_count() {
        let r0 = make_vreg(0, PtxType::F32);
        let r1 = make_vreg(1, PtxType::F32);
        let r2 = make_vreg(2, PtxType::F32);
        let r3 = make_vreg(3, PtxType::F32);
        let r4 = make_vreg(4, PtxType::F32);

        // mul %r2, %r0, %r1  ; temp = a * b
        // add %r4, %r2, %r3  ; result = temp + c
        let instructions = vec![make_mul(r2, r0, r1), make_add(r4, r2, r3)];

        let result = pass(instructions);

        // Should be fused to single FMA
        assert_eq!(
            result.len(),
            1,
            "FMA fusion should reduce 2 instructions to 1"
        );
        assert!(
            matches!(result[0].op, PtxOp::Fma),
            "Result should be FMA instruction"
        );
    }

    // cuda-tile-behavior.md: Falsification test #18
    #[test]
    fn test_single_use_detection_prevents_incorrect_fusion() {
        let r0 = make_vreg(0, PtxType::F32);
        let r1 = make_vreg(1, PtxType::F32);
        let r2 = make_vreg(2, PtxType::F32);
        let r3 = make_vreg(3, PtxType::F32);
        let r4 = make_vreg(4, PtxType::F32);
        let r5 = make_vreg(5, PtxType::F32);

        // mul %r2, %r0, %r1  ; temp = a * b
        // add %r4, %r2, %r3  ; result1 = temp + c
        // add %r5, %r2, %r3  ; result2 = temp + c (uses temp again!)
        let instructions = vec![
            make_mul(r2, r0, r1),
            make_add(r4, r2, r3),
            make_add(r5, r2, r3),
        ];

        let result = pass(instructions);

        // Should NOT fuse because r2 is used twice
        assert_eq!(
            result.len(),
            3,
            "Should not fuse when mul result has multiple uses"
        );
        assert!(
            !matches!(result[0].op, PtxOp::Fma),
            "First instruction should remain mul"
        );
    }

    // cuda-tile-behavior.md: Falsification test #25
    #[test]
    fn test_fma_fusion_is_idempotent() {
        let r0 = make_vreg(0, PtxType::F32);
        let r1 = make_vreg(1, PtxType::F32);
        let r2 = make_vreg(2, PtxType::F32);
        let r3 = make_vreg(3, PtxType::F32);
        let r4 = make_vreg(4, PtxType::F32);

        let instructions = vec![make_mul(r2, r0, r1), make_add(r4, r2, r3)];

        let first_pass = pass(instructions);
        let second_pass = pass(first_pass.clone());

        assert_eq!(
            first_pass.len(),
            second_pass.len(),
            "FMA fusion should be idempotent"
        );
    }

    // cuda-tile-behavior.md: Falsification test #30
    #[test]
    fn test_fma_pass_linear_complexity() {
        // Test with 1000 non-fusible instructions to verify O(n) complexity
        let mut instructions = Vec::with_capacity(1000);
        for i in 0..1000 {
            let r = make_vreg(i, PtxType::F32);
            instructions.push(PtxInstruction::new(PtxOp::Mov, PtxType::F32).dst(Operand::Reg(r)));
        }

        let start = std::time::Instant::now();
        let _result = pass(instructions);
        let elapsed = start.elapsed();

        // Should complete quickly (< 100ms for 1000 instructions)
        assert!(
            elapsed.as_millis() < 100,
            "FMA pass should have O(n) complexity, took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_fma_preserves_non_fusible() {
        // Instructions that can't be fused should be preserved
        let r0 = make_vreg(0, PtxType::F32);
        let r1 = make_vreg(1, PtxType::F32);
        let r2 = make_vreg(2, PtxType::F32);

        let instructions = vec![
            PtxInstruction::new(PtxOp::Mov, PtxType::F32)
                .dst(Operand::Reg(r0))
                .src(Operand::ImmF32(1.0)),
            make_add(r2, r0, r1), // No preceding mul
        ];

        let result = pass(instructions);
        assert_eq!(result.len(), 2, "Non-fusible instructions should be preserved");
    }

    #[test]
    fn test_empty_input() {
        let result = pass(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_integer_ops_not_fused() {
        let r0 = make_vreg(0, PtxType::U32);
        let r1 = make_vreg(1, PtxType::U32);
        let r2 = make_vreg(2, PtxType::U32);
        let r3 = make_vreg(3, PtxType::U32);
        let r4 = make_vreg(4, PtxType::U32);

        let instructions = vec![
            PtxInstruction::new(PtxOp::Mul, PtxType::U32)
                .dst(Operand::Reg(r2))
                .src(Operand::Reg(r0))
                .src(Operand::Reg(r1)),
            PtxInstruction::new(PtxOp::Add, PtxType::U32)
                .dst(Operand::Reg(r4))
                .src(Operand::Reg(r2))
                .src(Operand::Reg(r3)),
        ];

        let result = pass(instructions);
        assert_eq!(
            result.len(),
            2,
            "Integer ops should not be fused (no integer FMA)"
        );
    }
}
