//! PTX Arithmetic Operations Extension Trait.
//!
//! Provides arithmetic operations: add, sub, mul, fma, mad, dp4a, etc.

use super::super::instructions::{Operand, PtxInstruction, PtxOp, RoundingMode};
use super::super::registers::VirtualReg;
use super::super::types::PtxType;
use super::core::KernelBuilderCore;

/// Extension trait for PTX arithmetic operations.
///
/// # Example
///
/// ```ignore
/// use trueno_gpu::ptx::builder::{KernelBuilder, PtxArithmetic};
///
/// fn build_kernel(kb: &mut KernelBuilder) {
///     let a = kb.load_param_f32("a");
///     let b = kb.load_param_f32("b");
///     let sum = kb.add_f32(a, b);  // From PtxArithmetic trait
/// }
/// ```
pub trait PtxArithmetic: KernelBuilderCore {
    // ===== Integer Arithmetic =====

    /// Multiply-add low: dst = a * b + c
    fn mad_lo_u32(&mut self, a: VirtualReg, b: VirtualReg, c: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::MadLo, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(c)),
        );
        dst
    }

    /// Multiply wide (u32 * u32 -> u64)
    fn mul_wide_u32(&mut self, a: VirtualReg, b: u32) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U64);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Mul, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::ImmU64(b as u64)),
        );
        dst
    }

    /// Multiply wide (u32 * u32 -> u64) with register operands
    fn mul_wide_u32_reg(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U64);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Mul, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Add u64
    fn add_u64(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U64);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Add, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Add u64 into existing register (register reuse)
    fn add_u64_into(&mut self, dst: VirtualReg, a: VirtualReg, b: VirtualReg) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Add, PtxType::U64)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
    }

    /// Add u32
    fn add_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Add, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Add u32 into existing register (register reuse)
    fn add_u32_into(&mut self, dst: VirtualReg, a: VirtualReg, b: VirtualReg) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Add, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
    }

    /// Subtract u32
    fn sub_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Sub, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Multiply u32
    fn mul_u32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Mul, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    // ===== Floating Point Arithmetic =====

    /// Add f32
    fn add_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Add, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Subtract f32
    fn sub_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Sub, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Multiply f32
    fn mul_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Mul, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Divide f32
    fn div_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Div, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Fused multiply-add: dst = a * b + c
    fn fma_f32(&mut self, a: VirtualReg, b: VirtualReg, c: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Fma, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(c))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Fused multiply-add into existing register
    fn fma_f32_into(&mut self, dst: VirtualReg, a: VirtualReg, b: VirtualReg, c: VirtualReg) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Fma, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(c))
                .rounding(RoundingMode::Rn),
        );
    }

    /// Negate f32
    fn neg_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Neg, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Absolute value f32
    fn abs_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Abs, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Square root f32
    fn sqrt_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Sqrt, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Reciprocal square root (approximate)
    fn rsqrt_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Rsqrt, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Reciprocal (approximate)
    fn rcp_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Rcp, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Exponential base 2 (approximate)
    fn ex2_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Ex2, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Logarithm base 2 (approximate)
    fn lg2_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Lg2, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Sine (approximate)
    fn sin_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Sin, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Cosine (approximate)
    fn cos_f32(&mut self, a: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Cos, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a)),
        );
        dst
    }

    /// Minimum f32
    fn min_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Min, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Maximum f32
    fn max_f32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Max, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    // ===== DP4A - Dot Product of 4 8-bit integers =====

    /// DP4A: signed dot product of 4 8-bit integers, accumulated into 32-bit
    /// dst = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + c
    fn dp4a_s32(&mut self, a: VirtualReg, b: VirtualReg, c: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::S32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Dp4a, PtxType::S32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(c)),
        );
        dst
    }

    /// DP4A: unsigned dot product of 4 8-bit integers
    fn dp4a_u32(&mut self, a: VirtualReg, b: VirtualReg, c: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Dp4a, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(c)),
        );
        dst
    }

    /// DP4A: signed dot product, accumulating into existing register
    fn dp4a_s32_into(&mut self, dst: VirtualReg, a: VirtualReg, b: VirtualReg, c: VirtualReg) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Dp4a, PtxType::S32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b))
                .src(Operand::Reg(c)),
        );
    }
}

// Blanket implementation - any type implementing KernelBuilderCore gets PtxArithmetic
impl<T: KernelBuilderCore> PtxArithmetic for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::registers::RegisterAllocator;

    /// Mock builder for testing traits in isolation
    struct MockBuilder {
        registers: RegisterAllocator,
        instructions: Vec<PtxInstruction>,
        labels: Vec<String>,
    }

    impl MockBuilder {
        fn new() -> Self {
            Self {
                registers: RegisterAllocator::new(),
                instructions: Vec::new(),
                labels: Vec::new(),
            }
        }
    }

    impl KernelBuilderCore for MockBuilder {
        fn registers_mut(&mut self) -> &mut RegisterAllocator {
            &mut self.registers
        }
        fn instructions_mut(&mut self) -> &mut Vec<PtxInstruction> {
            &mut self.instructions
        }
        fn labels_mut(&mut self) -> &mut Vec<String> {
            &mut self.labels
        }
    }

    #[test]
    fn test_add_f32() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::F32);
        let b = builder.registers.allocate_virtual(PtxType::F32);

        let result = builder.add_f32(a, b);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::Add);
        assert!(result.id() > 0);
    }

    #[test]
    fn test_fma_f32() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::F32);
        let b = builder.registers.allocate_virtual(PtxType::F32);
        let c = builder.registers.allocate_virtual(PtxType::F32);

        let result = builder.fma_f32(a, b, c);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::Fma);
        assert!(result.id() > 0);
    }

    #[test]
    fn test_dp4a_s32() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::U32);
        let b = builder.registers.allocate_virtual(PtxType::U32);
        let c = builder.registers.allocate_virtual(PtxType::S32);

        let result = builder.dp4a_s32(a, b, c);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::Dp4a);
        assert!(result.id() > 0);
    }

    #[test]
    fn test_transcendentals() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::F32);

        let _sin = builder.sin_f32(a);
        let _cos = builder.cos_f32(a);
        let _sqrt = builder.sqrt_f32(a);
        let _rsqrt = builder.rsqrt_f32(a);
        let _rcp = builder.rcp_f32(a);
        let _ex2 = builder.ex2_f32(a);
        let _lg2 = builder.lg2_f32(a);

        assert_eq!(builder.instructions.len(), 7);
    }

    #[test]
    fn test_sub_mul_div_f32() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::F32);
        let b = builder.registers.allocate_virtual(PtxType::F32);

        let _sub = builder.sub_f32(a, b);
        let _mul = builder.mul_f32(a, b);
        let _div = builder.div_f32(a, b);

        assert_eq!(builder.instructions.len(), 3);
        assert_eq!(builder.instructions[0].op, PtxOp::Sub);
        assert_eq!(builder.instructions[1].op, PtxOp::Mul);
        assert_eq!(builder.instructions[2].op, PtxOp::Div);
    }

    #[test]
    fn test_neg_abs_f32() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::F32);

        let _neg = builder.neg_f32(a);
        let _abs = builder.abs_f32(a);

        assert_eq!(builder.instructions.len(), 2);
        assert_eq!(builder.instructions[0].op, PtxOp::Neg);
        assert_eq!(builder.instructions[1].op, PtxOp::Abs);
    }

    #[test]
    fn test_min_max_f32() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::F32);
        let b = builder.registers.allocate_virtual(PtxType::F32);

        let _min = builder.min_f32(a, b);
        let _max = builder.max_f32(a, b);

        assert_eq!(builder.instructions.len(), 2);
        assert_eq!(builder.instructions[0].op, PtxOp::Min);
        assert_eq!(builder.instructions[1].op, PtxOp::Max);
    }

    #[test]
    fn test_u32_operations() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::U32);
        let b = builder.registers.allocate_virtual(PtxType::U32);

        let _add = builder.add_u32(a, b);
        let _sub = builder.sub_u32(a, b);
        let _mul = builder.mul_u32(a, b);

        assert_eq!(builder.instructions.len(), 3);
        assert_eq!(builder.instructions[0].ty, PtxType::U32);
    }

    #[test]
    fn test_u64_operations() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::U64);
        let b = builder.registers.allocate_virtual(PtxType::U64);

        let _add = builder.add_u64(a, b);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].ty, PtxType::U64);
    }

    #[test]
    fn test_mul_wide_u32() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::U32);

        let _wide = builder.mul_wide_u32(a, 4);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].ty, PtxType::U64);
    }

    #[test]
    fn test_mul_wide_u32_reg() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::U32);
        let b = builder.registers.allocate_virtual(PtxType::U32);

        let _wide = builder.mul_wide_u32_reg(a, b);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].ty, PtxType::U64);
    }

    #[test]
    fn test_mad_lo_u32() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::U32);
        let b = builder.registers.allocate_virtual(PtxType::U32);
        let c = builder.registers.allocate_virtual(PtxType::U32);

        let _mad = builder.mad_lo_u32(a, b, c);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::MadLo);
    }

    #[test]
    fn test_into_operations() {
        let mut builder = MockBuilder::new();
        let dst_u32 = builder.registers.allocate_virtual(PtxType::U32);
        let dst_u64 = builder.registers.allocate_virtual(PtxType::U64);
        let dst_f32 = builder.registers.allocate_virtual(PtxType::F32);
        let a = builder.registers.allocate_virtual(PtxType::U32);
        let b = builder.registers.allocate_virtual(PtxType::U32);
        let a64 = builder.registers.allocate_virtual(PtxType::U64);
        let b64 = builder.registers.allocate_virtual(PtxType::U64);
        let af = builder.registers.allocate_virtual(PtxType::F32);
        let bf = builder.registers.allocate_virtual(PtxType::F32);
        let cf = builder.registers.allocate_virtual(PtxType::F32);

        builder.add_u32_into(dst_u32, a, b);
        builder.add_u64_into(dst_u64, a64, b64);
        builder.fma_f32_into(dst_f32, af, bf, cf);

        assert_eq!(builder.instructions.len(), 3);
    }

    #[test]
    fn test_dp4a_variants() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::U32);
        let b = builder.registers.allocate_virtual(PtxType::U32);
        let c_s = builder.registers.allocate_virtual(PtxType::S32);
        let c_u = builder.registers.allocate_virtual(PtxType::U32);

        let _dp4a_s = builder.dp4a_s32(a, b, c_s);
        let _dp4a_u = builder.dp4a_u32(a, b, c_u);

        let dst = builder.registers.allocate_virtual(PtxType::S32);
        builder.dp4a_s32_into(dst, a, b, c_s);

        assert_eq!(builder.instructions.len(), 3);
        assert_eq!(builder.instructions[0].op, PtxOp::Dp4a);
        assert_eq!(builder.instructions[1].op, PtxOp::Dp4a);
        assert_eq!(builder.instructions[2].op, PtxOp::Dp4a);
    }
}
