//! CTA-level WMMA GEMM — 4 warps cooperatively compute 32×32 output tiles
//!
//! Architecture: 2×2 warp grid, each warp owns a 16×16 WMMA subtile.
//! All 4 warps share 32×K A tile and K×32 B tile in shared memory.
//! K-dimension tiled in chunks of 16 (WMMA tile depth).
//!
//! Key improvements over basic WMMA:
//! - 4× more warps per CTA → better occupancy
//! - Shared memory reuse: each A/B element loaded once, used by 2 warps
//! - FP16 input (no FP32→FP16 conversion overhead)
//!
//! Launch: grid_2d((N+31)/32, (M+31)/32), block(128,1,1), shared_mem=2KB
//! Input: FP16 (u16), Output: FP32

use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, WmmaLayout};

/// Build a CTA-tiled WMMA GEMM kernel for FP16 input → FP32 output.
pub fn build_cta_wmma_fp16(m: u32, n: u32, k: u32) -> PtxKernel {
    let smem_bytes = 32 * 16 * 2 + 16 * 32 * 2; // A[32×16] + B[16×32] in FP16
    let n_k_tiles = (k + 15) / 16;

    PtxKernel::new("gemm_cta_wmma_fp16")
        .max_regs(64)
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .param(PtxType::U64, "c_ptr")
        .param(PtxType::U32, "m_param")
        .param(PtxType::U32, "n_param")
        .param(PtxType::U32, "k_param")
        .shared_memory(smem_bytes)
        .build(move |ctx| {
            let tid = ctx.special_reg(PtxReg::TidX);
            let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
            let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

            // Constants
            let c_1 = ctx.mov_u32_imm(1);
            let c_2 = ctx.mov_u32_imm(2);
            let c_4 = ctx.mov_u32_imm(4);
            let c_5 = ctx.mov_u32_imm(5);
            let c_15 = ctx.mov_u32_imm(15);
            let c_16 = ctx.mov_u32_imm(16);
            let c_31 = ctx.mov_u32_imm(31);
            let c_32 = ctx.mov_u32_imm(32);
            let c_512 = ctx.mov_u32_imm(512);
            let c_1024 = ctx.mov_u32_imm(1024);
            let k_const = ctx.mov_u32_imm(k);
            let n_const = ctx.mov_u32_imm(n);
            let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);
            let zero_f32 = ctx.mov_f32_imm(0.0);
            let zero_f16 = ctx.cvt_f16_f32(zero_f32);

            // CTA tile position
            let cta_row = ctx.mul_u32_reg(ctaid_y, c_32);
            let cta_col = ctx.mul_u32_reg(ctaid_x, c_32);

            let m_param = ctx.load_param_u32("m_param");
            let n_param = ctx.load_param_u32("n_param");
            let k_param = ctx.load_param_u32("k_param");

            let cta_oob = ctx.setp_ge_u32(cta_row, m_param);
            ctx.branch_if(cta_oob, "exit");
            let cta_oob2 = ctx.setp_ge_u32(cta_col, n_param);
            ctx.branch_if(cta_oob2, "exit");

            let a_ptr = ctx.load_param_u64("a_ptr");
            let b_ptr = ctx.load_param_u64("b_ptr");
            let c_ptr = ctx.load_param_u64("c_ptr");

            // Warp layout: 2×2 grid (warp_id = tid/32, 0-3)
            let warp_id = ctx.shr_u32(tid, c_5);
            let warp_row = ctx.shr_u32(warp_id, c_1);
            let warp_col = ctx.and_u32(warp_id, c_1);
            let warp_m_off = ctx.mul_u32_reg(warp_row, c_16);
            let warp_n_off = ctx.mul_u32_reg(warp_col, c_16);

            let smem_a_base = ctx.mov_u32_imm(0);

            // Init WMMA accumulator
            let frag_c = ctx.wmma_init_c_zero();

            // K-tile loop
            let k_tile = ctx.mov_u32_imm(0);
            ctx.label("k_loop");
            let k_done = ctx.setp_ge_u32(k_tile, n_k_tiles_reg);
            ctx.branch_if(k_done, "k_end");

            let k_off = ctx.mul_u32_reg(k_tile, c_16);

            // === Load A[32×16] + B[16×32] cooperatively ===
            // 1024 FP16 elements total, 128 threads, 8 elements each
            let c_8 = ctx.mov_u32_imm(8);
            let my_base = ctx.mul_u32_reg(tid, c_8);
            let load_i = ctx.mov_u32_imm(0);
            ctx.label("coop_load");
            let ld_done = ctx.setp_ge_u32(load_i, c_8);
            ctx.branch_if(ld_done, "coop_load_end");

            let idx = ctx.add_u32_reg(my_base, load_i);

            // idx < 512 → A tile, else B tile
            let is_a = ctx.setp_lt_u32(idx, c_512);
            ctx.branch_if_not(is_a, "ld_b");

            // A: idx/16 = row, idx%16 = col
            let a_r = ctx.shr_u32(idx, c_4);
            let a_c = ctx.and_u32(idx, c_15);
            let a_gr = ctx.add_u32_reg(cta_row, a_r);
            let a_gc = ctx.add_u32_reg(k_off, a_c);

            let smem_off = ctx.mul_u32_reg(idx, c_2);
            let smem_addr = ctx.add_u32_reg(smem_a_base, smem_off);
            ctx.st_shared_f16(smem_addr, zero_f16);

            let ar_ok = ctx.setp_lt_u32(a_gr, m_param);
            ctx.branch_if_not(ar_ok, "ld_next");
            let ac_ok = ctx.setp_lt_u32(a_gc, k_param);
            ctx.branch_if_not(ac_ok, "ld_next");

            let a_flat = ctx.mad_lo_u32(a_gr, k_const, a_gc);
            let a_boff = ctx.mul_wide_u32(a_flat, 2);
            let a_addr = ctx.add_u64(a_ptr, a_boff);
            let a_val = ctx.ld_global_f16(a_addr);
            ctx.st_shared_f16(smem_addr, a_val);
            ctx.branch("ld_next");

            // B: (idx-512)/32 = row, (idx-512)%32 = col
            ctx.label("ld_b");
            let bi = ctx.sub_u32_reg(idx, c_512);
            let b_r = ctx.shr_u32(bi, c_5);
            let b_c = ctx.and_u32(bi, c_31);
            let b_gr = ctx.add_u32_reg(k_off, b_r);
            let b_gc = ctx.add_u32_reg(cta_col, b_c);

            let bsmem_off = ctx.mul_u32_reg(bi, c_2);
            let bsmem_addr = ctx.add_u32_reg(c_1024, bsmem_off);
            ctx.st_shared_f16(bsmem_addr, zero_f16);

            let br_ok = ctx.setp_lt_u32(b_gr, k_param);
            ctx.branch_if_not(br_ok, "ld_next");
            let bc_ok = ctx.setp_lt_u32(b_gc, n_param);
            ctx.branch_if_not(bc_ok, "ld_next");

            let b_flat = ctx.mad_lo_u32(b_gr, n_const, b_gc);
            let b_boff = ctx.mul_wide_u32(b_flat, 2);
            let b_addr = ctx.add_u64(b_ptr, b_boff);
            let b_val = ctx.ld_global_f16(b_addr);
            ctx.st_shared_f16(bsmem_addr, b_val);

            ctx.label("ld_next");
            ctx.add_u32_inplace(load_i, 1);
            ctx.branch("coop_load");
            ctx.label("coop_load_end");

            ctx.bar_sync(0);

            // === WMMA per warp ===
            let smem_base = ctx.shared_base_addr();

            // A subtile: rows [warp_m_off..+16], stride=16
            let a_sub_bytes = ctx.mul_u32_reg(warp_m_off, c_32); // 16 cols * 2 bytes
            let a_sub_off = ctx.cvt_u64_u32(a_sub_bytes);
            let a_sub_ptr = ctx.add_u64(smem_base, a_sub_off);
            let frag_a = ctx.wmma_load_a_f16(a_sub_ptr, 16, WmmaLayout::RowMajor);

            // B subtile: cols [warp_n_off..+16], stride=32
            let b_sub_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
            let b_offset = ctx.add_u32_reg(c_1024, b_sub_bytes);
            let b_off64 = ctx.cvt_u64_u32(b_offset);
            let b_sub_ptr = ctx.add_u64(smem_base, b_off64);
            let frag_b = ctx.wmma_load_b_f16(b_sub_ptr, 32, WmmaLayout::RowMajor);

            let frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);
            for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                ctx.mov_f32_reg(*c_reg, *d_reg);
            }

            ctx.bar_sync(1);
            ctx.add_u32_inplace(k_tile, 1);
            ctx.branch("k_loop");
            ctx.label("k_end");

            // === Store C ===
            let c_row = ctx.add_u32_reg(cta_row, warp_m_off);
            let c_col = ctx.add_u32_reg(cta_col, warp_n_off);
            let cr_ok = ctx.setp_lt_u32(c_row, m_param);
            ctx.branch_if_not(cr_ok, "exit");
            let cc_ok = ctx.setp_lt_u32(c_col, n_param);
            ctx.branch_if_not(cc_ok, "exit");

            let c_row_off = ctx.mul_wide_u32_reg(c_row, n_param);
            let c_col_off = ctx.cvt_u64_u32(c_col);
            let c_base = ctx.add_u64(c_row_off, c_col_off);
            let c_base = ctx.mul_u64(c_base, 4);
            let c_addr = ctx.add_u64(c_ptr, c_base);

            ctx.wmma_store_d_f32(c_addr, &frag_c, n, WmmaLayout::RowMajor);

            ctx.label("exit");
            ctx.ret();
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx::PtxModule;

    #[test]
    fn test_cta_wmma_generates_valid_ptx() {
        let kernel = build_cta_wmma_fp16(128, 128, 128);
        let module = PtxModule::new().add_kernel(kernel);
        let ptx = module.emit();
        assert!(ptx.contains(".entry gemm_cta_wmma_fp16"));
        assert!(ptx.contains("bar.sync"));
    }

    #[test]
    fn test_cta_wmma_has_shared_memory() {
        let kernel = build_cta_wmma_fp16(64, 64, 64);
        let module = PtxModule::new().add_kernel(kernel);
        let ptx = module.emit();
        assert!(ptx.contains(".shared"));
    }
}
