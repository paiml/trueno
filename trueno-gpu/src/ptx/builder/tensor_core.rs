//! Tensor Core (WMMA) operations for KernelBuilder.
//!
//! Provides WMMA load/store/mma operations, F16/F32 conversions,
//! and F16 global memory access for tensor core pipelines.

use super::super::instructions::{Operand, PtxInstruction, PtxOp, RoundingMode, WmmaLayout};
use super::super::registers::VirtualReg;
use super::super::types::{PtxStateSpace, PtxType};
use super::KernelBuilder;

impl<'a> KernelBuilder<'a> {
    // ===== Tensor Core (WMMA) Operations =====
    // These require sm_70+ and generate WMMA PTX intrinsics

    /// Load F16 matrix fragment A for WMMA (16x16x16 tile)
    /// Returns fragment registers for use in wmma_mma
    pub fn wmma_load_a_f16(
        &mut self,
        addr: VirtualReg,
        stride: u32,
        layout: WmmaLayout,
    ) -> Vec<VirtualReg> {
        // WMMA 16x16x16 F16 requires 8 F16x2 registers (16 half values)
        let mut frag = Vec::with_capacity(8);
        for _ in 0..8 {
            frag.push(self.registers.allocate_virtual(PtxType::B32));
        }
        // Build instruction with all 8 destination registers
        let mut instr = PtxInstruction::new(PtxOp::WmmaLoadA, PtxType::F16).label(format!(
            "m16n16k16.{}.f16.stride.{}",
            layout.to_ptx_string(),
            stride
        ));
        // Add all fragment registers as destinations (use push_dst for vector dests)
        for reg in &frag {
            instr = instr.push_dst(Operand::Reg(*reg));
        }
        // Source is address and stride immediate
        instr = instr.src(Operand::Reg(addr));
        instr = instr.src(Operand::ImmI64(i64::from(stride)));
        self.instructions.push(instr);
        frag
    }

    /// Load F16 matrix fragment B for WMMA (16x16x16 tile)
    pub fn wmma_load_b_f16(
        &mut self,
        addr: VirtualReg,
        stride: u32,
        layout: WmmaLayout,
    ) -> Vec<VirtualReg> {
        let mut frag = Vec::with_capacity(8);
        for _ in 0..8 {
            frag.push(self.registers.allocate_virtual(PtxType::B32));
        }
        // Build instruction with all 8 destination registers
        let mut instr = PtxInstruction::new(PtxOp::WmmaLoadB, PtxType::F16).label(format!(
            "m16n16k16.{}.f16.stride.{}",
            layout.to_ptx_string(),
            stride
        ));
        for reg in &frag {
            instr = instr.push_dst(Operand::Reg(*reg));
        }
        instr = instr.src(Operand::Reg(addr));
        instr = instr.src(Operand::ImmI64(i64::from(stride)));
        self.instructions.push(instr);
        frag
    }

    /// Load F32 accumulator fragment C for WMMA (16x16x16 tile)
    pub fn wmma_load_c_f32(
        &mut self,
        addr: VirtualReg,
        stride: u32,
        layout: WmmaLayout,
    ) -> Vec<VirtualReg> {
        // Accumulator is 8 F32 values
        let mut frag = Vec::with_capacity(8);
        for _ in 0..8 {
            frag.push(self.registers.allocate_virtual(PtxType::F32));
        }
        // Build instruction with all 8 destination registers
        let mut instr = PtxInstruction::new(PtxOp::WmmaLoadC, PtxType::F32).label(format!(
            "m16n16k16.{}.f32.stride.{}",
            layout.to_ptx_string(),
            stride
        ));
        for reg in &frag {
            instr = instr.push_dst(Operand::Reg(*reg));
        }
        instr = instr.src(Operand::Reg(addr));
        instr = instr.src(Operand::ImmI64(i64::from(stride)));
        self.instructions.push(instr);
        frag
    }

    /// Initialize F32 accumulator fragment C to zero (WAPR-PERF-010)
    /// This avoids loading from memory address 0 which is invalid
    pub fn wmma_init_c_zero(&mut self) -> Vec<VirtualReg> {
        // Accumulator is 8 F32 values, initialize all to 0.0
        let mut frag = Vec::with_capacity(8);
        for _ in 0..8 {
            let reg = self.registers.allocate_virtual(PtxType::F32);
            self.instructions.push(
                PtxInstruction::new(PtxOp::Mov, PtxType::F32)
                    .dst(Operand::Reg(reg))
                    .src(Operand::ImmF32(0.0)),
            );
            frag.push(reg);
        }
        frag
    }

    /// WMMA matrix multiply-accumulate: D = A * B + C
    /// Takes A, B, C fragment registers and returns D fragment registers
    #[allow(clippy::similar_names)]
    pub fn wmma_mma_f16_f32(
        &mut self,
        frag_a: &[VirtualReg],
        frag_b: &[VirtualReg],
        frag_c: &[VirtualReg],
    ) -> Vec<VirtualReg> {
        // Output accumulator D (8 F32 values)
        let mut frag_d = Vec::with_capacity(8);
        for _ in 0..8 {
            frag_d.push(self.registers.allocate_virtual(PtxType::F32));
        }

        // MMA instruction with all fragment registers
        // Format: wmma.mma.sync.aligned.m16n16k16.row.col.f32.f32 {d0-d7}, {a0-a7}, {b0-b7}, {c0-c7}
        let mut instr =
            PtxInstruction::new(PtxOp::WmmaMma, PtxType::F32).label("m16n16k16.row.col.f32.f32");

        // Add all D registers as destinations (use push_dst for vector dests)
        for reg in &frag_d {
            instr = instr.push_dst(Operand::Reg(*reg));
        }

        // Add all A, B, C fragment registers as sources (in order)
        for reg in frag_a {
            instr = instr.src(Operand::Reg(*reg));
        }
        for reg in frag_b {
            instr = instr.src(Operand::Reg(*reg));
        }
        for reg in frag_c {
            instr = instr.src(Operand::Reg(*reg));
        }

        self.instructions.push(instr);
        frag_d
    }

    /// Store F32 accumulator fragment D to memory
    pub fn wmma_store_d_f32(
        &mut self,
        addr: VirtualReg,
        frag_d: &[VirtualReg],
        stride: u32,
        layout: WmmaLayout,
    ) {
        if frag_d.is_empty() {
            return;
        }
        // Format: wmma.store.d.sync.aligned.m16n16k16.row.f32 [addr], {d0-d7}, stride
        let mut instr = PtxInstruction::new(PtxOp::WmmaStoreD, PtxType::F32).label(format!(
            "m16n16k16.{}.f32.stride.{}",
            layout.to_ptx_string(),
            stride
        ));
        // Address is first source
        instr = instr.src(Operand::Reg(addr));
        // All fragment registers
        for reg in frag_d {
            instr = instr.src(Operand::Reg(*reg));
        }
        // Stride
        instr = instr.src(Operand::ImmI64(i64::from(stride)));
        self.instructions.push(instr);
    }

    /// Convert F32 values to F16 (for feeding tensor cores)
    pub fn cvt_f16_f32(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F16);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::F16)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val))
                .rounding(RoundingMode::Rn),
        );
        dst
    }

    /// Convert F16 value to F32 (for accumulation)
    pub fn cvt_f32_f16(&mut self, val: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F32);
        self.instructions.push(
            PtxInstruction::new(PtxOp::Cvt, PtxType::F32)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(val)),
        );
        dst
    }

    /// Load F16 from global memory
    ///
    /// NOTE: PTX uses `.b16` (binary 16-bit) for half-precision loads,
    /// not `.f16`. The loaded value is still interpreted as f16 for
    /// subsequent operations.
    pub fn ld_global_f16(&mut self, addr: VirtualReg) -> VirtualReg {
        let dst = self.registers.allocate_virtual(PtxType::F16);
        self.instructions.push(
            // PTX requires ld.global.b16, not ld.global.f16
            PtxInstruction::new(PtxOp::Ld, PtxType::B16)
                .dst(Operand::Reg(dst))
                .src(Operand::Reg(addr))
                .space(PtxStateSpace::Global),
        );
        dst
    }

    /// Store F16 to global memory
    ///
    /// PTX uses `.b16` (binary 16-bit) for half-precision stores, not `.f16`.
    /// The PTX ISA does not support `st.global.f16` — only `.b16` for 16-bit stores.
    /// This matches `ld_global_f16` which already uses `PtxType::B16`.
    pub fn st_global_f16(&mut self, addr: VirtualReg, val: VirtualReg) {
        self.instructions.push(
            PtxInstruction::new(PtxOp::St, PtxType::B16)
                .src(Operand::Reg(addr))
                .src(Operand::Reg(val))
                .space(PtxStateSpace::Global),
        );
    }
}
