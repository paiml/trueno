//! PTX Type System
//!
//! Defines PTX data types and state spaces.

use std::fmt;

/// PTX data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxType {
    /// Predicate (1-bit boolean)
    Pred,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// 8-bit signed integer
    S8,
    /// 16-bit signed integer
    S16,
    /// 32-bit signed integer
    S32,
    /// 64-bit signed integer
    S64,
    /// 16-bit floating point (half precision)
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 32-bit floating point (single precision)
    F32,
    /// 64-bit floating point (double precision)
    F64,
    /// 8-bit untyped (for byte operations)
    B8,
    /// 16-bit untyped
    B16,
    /// 32-bit untyped
    B32,
    /// 64-bit untyped
    B64,
}

impl PtxType {
    /// Get size in bytes
    #[must_use]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Pred | Self::U8 | Self::S8 | Self::B8 => 1,
            Self::U16 | Self::S16 | Self::F16 | Self::BF16 | Self::B16 => 2,
            Self::U32 | Self::S32 | Self::F32 | Self::B32 => 4,
            Self::U64 | Self::S64 | Self::F64 | Self::B64 => 8,
        }
    }

    /// Get size in bits
    #[must_use]
    pub const fn size_bits(self) -> usize {
        self.size_bytes() * 8
    }

    /// Convert to PTX string representation
    #[must_use]
    pub const fn to_ptx_string(self) -> &'static str {
        match self {
            Self::Pred => ".pred",
            Self::U8 => ".u8",
            Self::U16 => ".u16",
            Self::U32 => ".u32",
            Self::U64 => ".u64",
            Self::S8 => ".s8",
            Self::S16 => ".s16",
            Self::S32 => ".s32",
            Self::S64 => ".s64",
            Self::F16 => ".f16",
            Self::BF16 => ".bf16",
            Self::F32 => ".f32",
            Self::F64 => ".f64",
            Self::B8 => ".b8",
            Self::B16 => ".b16",
            Self::B32 => ".b32",
            Self::B64 => ".b64",
        }
    }

    /// Check if this is a floating point type
    #[must_use]
    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }

    /// Check if this is a signed integer type
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(self, Self::S8 | Self::S16 | Self::S32 | Self::S64)
    }

    /// Check if this is an unsigned integer type
    #[must_use]
    pub const fn is_unsigned(self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    /// Get the register prefix for this type
    #[must_use]
    pub const fn register_prefix(self) -> &'static str {
        match self {
            Self::Pred => "%p",
            Self::F16 | Self::BF16 => "%h",
            Self::F32 => "%f",
            Self::F64 => "%fd",
            _ => "%r",
        }
    }
}

impl fmt::Display for PtxType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_ptx_string())
    }
}

/// PTX state spaces (memory hierarchy)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxStateSpace {
    /// Register (fastest, per-thread)
    Reg,
    /// Shared memory (fast, per-block, 48KB-164KB)
    Shared,
    /// Global memory (slow, device-wide, GBs)
    Global,
    /// Local memory (slow, per-thread spill)
    Local,
    /// Constant memory (cached, read-only, 64KB)
    Const,
    /// Texture memory (cached, read-only, spatial locality)
    Tex,
    /// Parameter space (kernel arguments)
    Param,
}

impl PtxStateSpace {
    /// Convert to PTX string
    #[must_use]
    pub const fn to_ptx_string(self) -> &'static str {
        match self {
            Self::Reg => ".reg",
            Self::Shared => ".shared",
            Self::Global => ".global",
            Self::Local => ".local",
            Self::Const => ".const",
            Self::Tex => ".tex",
            Self::Param => ".param",
        }
    }

    /// Check if this state space is cached
    #[must_use]
    pub const fn is_cached(self) -> bool {
        matches!(self, Self::Const | Self::Tex)
    }

    /// Check if this state space is per-thread
    #[must_use]
    pub const fn is_per_thread(self) -> bool {
        matches!(self, Self::Reg | Self::Local)
    }
}

impl fmt::Display for PtxStateSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_ptx_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_sizes() {
        assert_eq!(PtxType::Pred.size_bytes(), 1);
        assert_eq!(PtxType::U8.size_bytes(), 1);
        assert_eq!(PtxType::U16.size_bytes(), 2);
        assert_eq!(PtxType::U32.size_bytes(), 4);
        assert_eq!(PtxType::U64.size_bytes(), 8);
        assert_eq!(PtxType::F16.size_bytes(), 2);
        assert_eq!(PtxType::F32.size_bytes(), 4);
        assert_eq!(PtxType::F64.size_bytes(), 8);
    }

    #[test]
    fn test_type_bits() {
        assert_eq!(PtxType::U8.size_bits(), 8);
        assert_eq!(PtxType::U32.size_bits(), 32);
        assert_eq!(PtxType::U64.size_bits(), 64);
    }

    #[test]
    fn test_float_detection() {
        assert!(PtxType::F16.is_float());
        assert!(PtxType::F32.is_float());
        assert!(PtxType::F64.is_float());
        assert!(PtxType::BF16.is_float());
        assert!(!PtxType::U32.is_float());
        assert!(!PtxType::S32.is_float());
    }

    #[test]
    fn test_signed_detection() {
        assert!(PtxType::S8.is_signed());
        assert!(PtxType::S32.is_signed());
        assert!(!PtxType::U32.is_signed());
        assert!(!PtxType::F32.is_signed());
    }

    #[test]
    fn test_state_space_strings() {
        assert_eq!(PtxStateSpace::Global.to_ptx_string(), ".global");
        assert_eq!(PtxStateSpace::Shared.to_ptx_string(), ".shared");
        assert_eq!(PtxStateSpace::Reg.to_ptx_string(), ".reg");
    }

    #[test]
    fn test_display_impl() {
        assert_eq!(format!("{}", PtxType::F32), ".f32");
        assert_eq!(format!("{}", PtxStateSpace::Global), ".global");
    }
}
