//! PTX Synchronization Operations Extension Trait.
//!
//! Provides barriers, memory fences, and warp shuffle operations.

use super::super::instructions::{Operand, PtxInstruction, PtxOp};
use super::super::registers::VirtualReg;
use super::super::types::PtxType;
use super::core::KernelBuilderCore;

/// Extension trait for PTX synchronization operations.
///
/// # Example
///
/// ```ignore
/// use trueno_gpu::ptx::builder::{KernelBuilder, PtxSync};
///
/// fn build_kernel(kb: &mut KernelBuilder) {
///     // Barrier for all threads in block
///     kb.bar_sync(0);
///
///     // Warp shuffle for reduction
///     let val = kb.load_param_f32("val");
///     let shuffled = kb.shfl_down_f32(val, 16);
/// }
/// ```
pub trait PtxSync: KernelBuilderCore {
    // ===== Barriers =====

    /// Synchronize all threads in the block
    fn bar_sync(&mut self, id: u32) {
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Bar, PtxType::U32).src(Operand::ImmI64(id as i64)),
        );
    }

    /// Memory fence (CTA scope)
    fn membar_cta(&mut self) {
        self.instructions_mut()
            .push(PtxInstruction::new(PtxOp::MemBar, PtxType::Pred));
    }

    /// Memory fence (global scope)
    fn membar_gl(&mut self) {
        self.instructions_mut()
            .push(PtxInstruction::new(PtxOp::MemBar, PtxType::Pred));
    }

    // ===== Warp Shuffle =====

    /// Shuffle down (for warp reduction): get value from lane + delta
    fn shfl_down_f32(&mut self, val: VirtualReg, delta: u32) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::ShflDown, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::ImmI64(delta as i64))
                .src(Operand::ImmI64(31)), // mask for 32 threads
        );
        dst
    }

    /// Shuffle down (u32 version)
    fn shfl_down_u32(&mut self, val: VirtualReg, delta: u32) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::ShflDown, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::ImmI64(delta as i64))
                .src(Operand::ImmI64(31)),
        );
        dst
    }

    /// Shuffle XOR (butterfly reduction pattern)
    fn shfl_xor_f32(&mut self, val: VirtualReg, mask: u32) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::ShflBfly, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::ImmI64(mask as i64))
                .src(Operand::ImmI64(31)),
        );
        dst
    }

    /// Shuffle broadcast (get value from specific lane)
    fn shfl_idx_f32(&mut self, val: VirtualReg, lane: u32) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::F32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::ShflIdx, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::ImmI64(lane as i64))
                .src(Operand::ImmI64(31)),
        );
        dst
    }

    // ===== Warp Vote =====

    /// ballot - count threads where predicate is true
    fn vote_ballot(&mut self, pred: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::VoteBallot, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(pred)),
        );
        dst
    }

    /// all - true if all threads have predicate true
    fn vote_all(&mut self, pred: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::Pred);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::VoteAll, PtxType::Pred)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(pred)),
        );
        dst
    }

    /// any - true if any thread has predicate true
    fn vote_any(&mut self, pred: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::Pred);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::VoteAny, PtxType::Pred)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(pred)),
        );
        dst
    }

    // ===== Bit Manipulation =====

    /// Population count (number of set bits)
    fn popc_b32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Popc, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Count leading zeros
    fn clz_b32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Clz, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Bit field extract
    fn bfe_u32(&mut self, val: VirtualReg, start: u32, len: u32) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::U32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Bfe, PtxType::U32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::ImmI64(start as i64))
                .src(Operand::ImmI64(len as i64)),
        );
        dst
    }

    /// Bitwise AND
    fn and_b32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::B32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::And, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Bitwise OR
    fn or_b32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::B32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Or, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Bitwise XOR
    fn xor_b32(&mut self, a: VirtualReg, b: VirtualReg) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::B32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Xor, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(a))
                .src(Operand::Reg(b)),
        );
        dst
    }

    /// Shift left
    fn shl_b32(&mut self, val: VirtualReg, shift: u32) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::B32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Shl, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::ImmI64(shift as i64)),
        );
        dst
    }

    /// Shift right (logical)
    fn shr_b32(&mut self, val: VirtualReg, shift: u32) -> VirtualReg {
        let dst = self.registers_mut().allocate_virtual(PtxType::B32);
        self.instructions_mut().push(
            PtxInstruction::new(PtxOp::Shr, PtxType::B32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .src(Operand::ImmI64(shift as i64)),
        );
        dst
    }
}

// Blanket implementation
impl<T: KernelBuilderCore> PtxSync for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::registers::RegisterAllocator;

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
    fn test_bar_sync() {
        let mut builder = MockBuilder::new();

        builder.bar_sync(0);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::Bar);
    }

    #[test]
    fn test_membar() {
        let mut builder = MockBuilder::new();

        builder.membar_cta();
        builder.membar_gl();

        assert_eq!(builder.instructions.len(), 2);
        assert_eq!(builder.instructions[0].op, PtxOp::MemBar);
        assert_eq!(builder.instructions[1].op, PtxOp::MemBar);
    }

    #[test]
    fn test_shfl_down() {
        let mut builder = MockBuilder::new();
        let val = builder.registers.allocate_virtual(PtxType::F32);

        let result = builder.shfl_down_f32(val, 16);

        assert_eq!(builder.instructions.len(), 1);
        assert_eq!(builder.instructions[0].op, PtxOp::ShflDown);
        assert!(result.id() > 0);
    }

    #[test]
    fn test_warp_vote() {
        let mut builder = MockBuilder::new();
        let pred = builder.registers.allocate_virtual(PtxType::Pred);

        let _ballot = builder.vote_ballot(pred);
        let _all = builder.vote_all(pred);
        let _any = builder.vote_any(pred);

        assert_eq!(builder.instructions.len(), 3);
    }

    #[test]
    fn test_bit_manipulation() {
        let mut builder = MockBuilder::new();
        let a = builder.registers.allocate_virtual(PtxType::U32);
        let b = builder.registers.allocate_virtual(PtxType::U32);

        let _popc = builder.popc_b32(a);
        let _clz = builder.clz_b32(a);
        let _and = builder.and_b32(a, b);
        let _or = builder.or_b32(a, b);
        let _xor = builder.xor_b32(a, b);
        let _shl = builder.shl_b32(a, 4);
        let _shr = builder.shr_b32(a, 4);

        assert_eq!(builder.instructions.len(), 7);
    }
}
