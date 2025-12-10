//! PTX Instructions
//!
//! Defines the PTX instruction set and operations.

use super::registers::VirtualReg;
use super::types::{PtxStateSpace, PtxType};

/// PTX operation codes
#[derive(Debug, Clone, PartialEq)]
pub enum PtxOp {
    // ===== Arithmetic =====
    /// Add two values
    Add,
    /// Subtract
    Sub,
    /// Multiply
    Mul,
    /// Multiply-add (fused)
    Mad,
    /// Multiply-add low bits
    MadLo,
    /// Divide
    Div,
    /// Remainder
    Rem,
    /// Absolute value
    Abs,
    /// Negate
    Neg,
    /// Minimum
    Min,
    /// Maximum
    Max,

    // ===== Floating Point Special =====
    /// Reciprocal
    Rcp,
    /// Reciprocal square root
    Rsqrt,
    /// Square root
    Sqrt,
    /// Sine (approximate)
    Sin,
    /// Cosine (approximate)
    Cos,
    /// Exponential base 2 (approximate)
    Ex2,
    /// Logarithm base 2 (approximate)
    Lg2,
    /// Fused multiply-add
    Fma,

    // ===== Comparison =====
    /// Set predicate (comparison)
    Setp,

    // ===== Logical/Bitwise =====
    /// Bitwise AND
    And,
    /// Bitwise OR
    Or,
    /// Bitwise XOR
    Xor,
    /// Bitwise NOT
    Not,
    /// Shift left
    Shl,
    /// Shift right (logical)
    Shr,

    // ===== Data Movement =====
    /// Move/copy
    Mov,
    /// Load from memory
    Ld,
    /// Store to memory
    St,
    /// Load parameter
    LdParam,
    /// Convert type
    Cvt,
    /// Select based on predicate
    Selp,

    // ===== Warp-Level =====
    /// Warp shuffle down
    ShflDown,
    /// Warp shuffle up
    ShflUp,
    /// Warp shuffle XOR
    ShflBfly,
    /// Warp shuffle indexed
    ShflIdx,
    /// Warp vote all
    VoteAll,
    /// Warp vote any
    VoteAny,
    /// Warp vote ballot
    VoteBallot,

    // ===== Control Flow =====
    /// Branch
    Bra,
    /// Call
    Call,
    /// Return
    Ret,
    /// Exit kernel
    Exit,
    /// Barrier synchronization
    Bar,
    /// Memory fence
    MemBar,

    // ===== Texture/Surface =====
    /// Texture load
    Tex,
    /// Surface load
    Suld,
    /// Surface store
    Sust,

    // ===== Atomic =====
    /// Atomic add
    AtomAdd,
    /// Atomic min
    AtomMin,
    /// Atomic max
    AtomMax,
    /// Atomic exchange
    AtomExch,
    /// Atomic compare-and-swap
    AtomCas,
}

/// Comparison operators for setp
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Less than
    Lt,
    /// Less than or equal
    Le,
    /// Greater than
    Gt,
    /// Greater than or equal
    Ge,
    /// Less than (unsigned)
    Lo,
    /// Less than or equal (unsigned)
    Ls,
    /// Greater than (unsigned)
    Hi,
    /// Greater than or equal (unsigned)
    Hs,
}

impl CmpOp {
    /// Convert to PTX string
    #[must_use]
    pub const fn to_ptx_string(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::Lo => "lo",
            Self::Ls => "ls",
            Self::Hi => "hi",
            Self::Hs => "hs",
        }
    }
}

/// Rounding modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoundingMode {
    /// Round to nearest, ties to even (default)
    #[default]
    Rn,
    /// Round toward zero
    Rz,
    /// Round toward positive infinity
    Rp,
    /// Round toward negative infinity
    Rm,
}

impl RoundingMode {
    /// Convert to PTX string
    #[must_use]
    pub const fn to_ptx_string(self) -> &'static str {
        match self {
            Self::Rn => ".rn",
            Self::Rz => ".rz",
            Self::Rp => ".rp",
            Self::Rm => ".rm",
        }
    }
}

/// A single PTX instruction
#[derive(Debug, Clone)]
pub struct PtxInstruction {
    /// Operation
    pub op: PtxOp,
    /// Data type
    pub ty: PtxType,
    /// Destination register (if any)
    pub dst: Option<Operand>,
    /// Source operands
    pub srcs: Vec<Operand>,
    /// Predicate guard (optional)
    pub predicate: Option<Predicate>,
    /// State space (for memory ops)
    pub state_space: Option<PtxStateSpace>,
    /// Rounding mode (for FP ops)
    pub rounding: Option<RoundingMode>,
    /// Label (for branch targets)
    pub label: Option<String>,
}

/// Instruction operand
#[derive(Debug, Clone)]
pub enum Operand {
    /// Virtual register
    Reg(VirtualReg),
    /// Special register
    SpecialReg(super::registers::PtxReg),
    /// Immediate integer
    ImmI64(i64),
    /// Immediate unsigned
    ImmU64(u64),
    /// Immediate float
    ImmF32(f32),
    /// Immediate double
    ImmF64(f64),
    /// Parameter name
    Param(String),
    /// Memory address (base + offset)
    Addr {
        /// Base register
        base: VirtualReg,
        /// Offset in bytes
        offset: i32,
    },
    /// Label reference
    Label(String),
}

/// Predicate for conditional execution
#[derive(Debug, Clone)]
pub struct Predicate {
    /// Predicate register
    pub reg: VirtualReg,
    /// Negated?
    pub negated: bool,
}

impl PtxInstruction {
    /// Create a new instruction
    #[must_use]
    pub fn new(op: PtxOp, ty: PtxType) -> Self {
        Self {
            op,
            ty,
            dst: None,
            srcs: Vec::new(),
            predicate: None,
            state_space: None,
            rounding: None,
            label: None,
        }
    }

    /// Set destination
    #[must_use]
    pub fn dst(mut self, dst: Operand) -> Self {
        self.dst = Some(dst);
        self
    }

    /// Add source operand
    #[must_use]
    pub fn src(mut self, src: Operand) -> Self {
        self.srcs.push(src);
        self
    }

    /// Set predicate guard
    #[must_use]
    pub fn predicated(mut self, pred: Predicate) -> Self {
        self.predicate = Some(pred);
        self
    }

    /// Set state space
    #[must_use]
    pub fn space(mut self, space: PtxStateSpace) -> Self {
        self.state_space = Some(space);
        self
    }

    /// Set rounding mode
    #[must_use]
    pub fn rounding(mut self, mode: RoundingMode) -> Self {
        self.rounding = Some(mode);
        self
    }

    /// Set label
    #[must_use]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cmp_op_strings() {
        assert_eq!(CmpOp::Eq.to_ptx_string(), "eq");
        assert_eq!(CmpOp::Lt.to_ptx_string(), "lt");
        assert_eq!(CmpOp::Ge.to_ptx_string(), "ge");
    }

    #[test]
    fn test_rounding_mode_strings() {
        assert_eq!(RoundingMode::Rn.to_ptx_string(), ".rn");
        assert_eq!(RoundingMode::Rz.to_ptx_string(), ".rz");
    }

    #[test]
    fn test_instruction_builder() {
        let instr = PtxInstruction::new(PtxOp::Add, PtxType::F32)
            .dst(Operand::ImmF32(0.0))
            .src(Operand::ImmF32(1.0))
            .src(Operand::ImmF32(2.0));

        assert_eq!(instr.op, PtxOp::Add);
        assert_eq!(instr.ty, PtxType::F32);
        assert!(instr.dst.is_some());
        assert_eq!(instr.srcs.len(), 2);
    }

    #[test]
    fn test_instruction_predicated() {
        let pred_reg = VirtualReg::new(0, PtxType::Pred);
        let pred = Predicate {
            reg: pred_reg,
            negated: false,
        };

        let instr = PtxInstruction::new(PtxOp::Bra, PtxType::B32)
            .predicated(pred)
            .label("exit");

        assert!(instr.predicate.is_some());
        assert!(instr.label.is_some());
    }

    #[test]
    fn test_instruction_memory() {
        let instr = PtxInstruction::new(PtxOp::Ld, PtxType::F32)
            .space(PtxStateSpace::Global);

        assert_eq!(instr.state_space, Some(PtxStateSpace::Global));
    }
}
