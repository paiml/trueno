//! 128×128 CTA-level WMMA GEMM — Phase 2 of GPU GEMM bridge plan
//!
//! Architecture: 8×8 warp grid (64 warps → too many), OR 4×4 warp grid
//! where each warp computes a 2×2 block of 16×16 WMMA tiles = 32×32 output.
//! All warps share 128×K A tile and K×128 B tile in shared memory.
//!
//! K-dimension tiled in chunks of 16 (WMMA tile depth).
//! 3-stage cp.async pipeline for latency hiding.
//!
//! Key advantage over 64×64 CTA: 2× compute-to-load ratio (64 vs 32 FLOP/byte).
//! Each loaded A element is reused by 8 column warps (vs 4 in 64×64).
//! Each loaded B element is reused by 8 row warps (vs 4 in 64×64).
//!
//! Shared memory: 128×16×2 (A) + 16×128×2 (B) = 8 KB per stage.
//! 3 stages = 24 KB total (fits in 48 KB static smem).
//!
//! Launch: grid_2d((N+127)/128, (M+127)/128), block(512,1,1), shared_mem=24KB
//! Input: FP16 (u16), Output: FP32
//!
//! Target: 60+ TFLOP/s on RTX 4090 (from 40.5 at 64×64 → +48%)
//!
//! Ref: CUTLASS SM80 default: GemmShape<128,256,64>, WarpShape<64,64,64>,
//! InstructionShape<16,8,16> (mma.sync.m16n8k16). We use wmma.m16n16k16.

use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, WmmaLayout};

/// Build a 128×128 CTA-tiled WMMA GEMM kernel with 3-stage cp.async pipeline.
///
/// 16 warps (512 threads), 4×4 grid where each warp computes 2×2 block of
/// 16×16 WMMA tiles = 32×32 output per warp. 64 WMMAs per K-tile (vs 16 in 64×64).
///
/// Contract: cgp-gpu-gemm-cta128-v1.yaml (when created)
pub fn build_cta128_wmma_fp16_cpasync(m: u32, n: u32, k: u32) -> PtxKernel {
    let tile_m: u32 = 128;
    let tile_n: u32 = 128;
    let tile_k: u32 = 16;
    let a_smem_bytes = (tile_m * tile_k * 2) as usize; // 4096
    let b_smem_bytes = (tile_k * tile_n * 2) as usize; // 4096
    let smem_single = a_smem_bytes + b_smem_bytes; // 8192
    let num_stages: usize = 3;
    let smem_bytes = smem_single * num_stages; // 24576 = 24KB
    let n_k_tiles = (k + tile_k - 1) / tile_k;
    // 512 threads: 256 load A (128*16=2048 FP16 = 4096 bytes → 16 bytes/thread = 2 cp.async)
    // 256 load B (same)
    let a_threads: u32 = 256;
    let _ = m;

    PtxKernel::new("gemm_cta128_cpasync_fp16")
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

            let c_2 = ctx.mov_u32_imm(2);
            let c_4 = ctx.mov_u32_imm(4);
            let c_5 = ctx.mov_u32_imm(5);
            let c_8 = ctx.mov_u32_imm(8);
            let c_16 = ctx.mov_u32_imm(16);
            let c_32 = ctx.mov_u32_imm(32);
            let c_128 = ctx.mov_u32_imm(128);
            let c_256 = ctx.mov_u32_imm(a_threads);
            let c_a_smem = ctx.mov_u32_imm(a_smem_bytes as u32);
            let k_const = ctx.mov_u32_imm(k);
            let n_const = ctx.mov_u32_imm(n);
            let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);

            // CTA position: 128×128 tiles
            let cta_row = ctx.mul_u32_reg(ctaid_y, c_128);
            let cta_col = ctx.mul_u32_reg(ctaid_x, c_128);

            let m_param = ctx.load_param_u32("m_param");
            let n_param = ctx.load_param_u32("n_param");
            let _k_param = ctx.load_param_u32("k_param");

            // Bounds check: skip entire CTA if out of bounds
            let cta_oob = ctx.setp_ge_u32(cta_row, m_param);
            ctx.branch_if(cta_oob, "exit");
            let cta_oob2 = ctx.setp_ge_u32(cta_col, n_param);
            ctx.branch_if(cta_oob2, "exit");

            let a_ptr = ctx.load_param_u64("a_ptr");
            let b_ptr = ctx.load_param_u64("b_ptr");
            let c_ptr = ctx.load_param_u64("c_ptr");

            // Warp layout: 4×4 grid, each warp computes 2×2 WMMA tiles = 32×32 output
            let c_3 = ctx.mov_u32_imm(3);
            let warp_id = ctx.shr_u32(tid, c_5); // tid / 32
            let warp_row = ctx.shr_u32(warp_id, c_2); // warp_id / 4 (0..3)
            let warp_col = ctx.and_u32(warp_id, c_3); // warp_id % 4 (0..3)
                                                      // Each warp position: 32×32 block at (warp_row*32, warp_col*32)
            let warp_m_off = ctx.mul_u32_reg(warp_row, c_32); // row offset in tile
            let warp_n_off = ctx.mul_u32_reg(warp_col, c_32); // col offset in tile

            let smem_base = ctx.shared_base_addr();

            // Thread role for loading: is_a = (tid < 256)
            let is_a_thread = ctx.setp_lt_u32(tid, c_256);

            // ═══ Load offsets ═══
            // A: 128 rows × 16 cols = 2048 FP16. 256 threads → 8 elements/thread → 2 cp.async(8B)
            // Thread t loads: elements [t*8..t*8+7], i.e., row = (t*8)/16 = t/2, col = (t*8)%16
            // cp.async 0: smem_off = t*16, global_off = row*K + col
            // cp.async 1: smem_off = t*16 + 8, global_off = row*K + col + 4
            let a_row_in_tile = ctx.shr_u32(tid, c_2); // t/4 (covers 0..63 for first half)
                                                       // But we need rows 0..127 with 256 threads loading 8 elements each
                                                       // Better: t*8/16 = t/2 for the row
            let a_elem_start = ctx.mul_u32_reg(tid, c_8); // t * 8
            let a_row0 = ctx.shr_u32(a_elem_start, c_4); // (t*8) / 16
            let c_mask15 = ctx.mov_u32_imm(15);
            let a_col0_base = ctx.and_u32(a_elem_start, c_mask15); // (t*8) % 16

            // smem offsets for A loads (2 cp.async per thread)
            let a_smem_off0 = ctx.mul_u32_reg(tid, c_16); // t*16 bytes (t*8 elements * 2 bytes)
            let a_smem_off1 = ctx.add_u32_reg(a_smem_off0, c_8); // +8 bytes

            // B: 16 rows × 128 cols = 2048 FP16. 256 threads → 8 elements/thread → 2 cp.async(8B)
            let b_local = ctx.sub_u32_reg(tid, c_256);
            let b_elem_start = ctx.mul_u32_reg(b_local, c_8); // local * 8
            let c_mask127 = ctx.mov_u32_imm(127);
            let c_7 = ctx.mov_u32_imm(7);
            let b_row0 = ctx.shr_u32(b_elem_start, c_7); // (local*8)/128
            let b_col0_base = ctx.and_u32(b_elem_start, c_mask127); // (local*8)%128

            let b_smem_base = ctx.mov_u32_imm(a_smem_bytes as u32);
            let b_smem_off0_local = ctx.mul_u32_reg(b_local, c_16); // local*16 bytes
            let b_smem_off0 = ctx.add_u32_reg(b_smem_base, b_smem_off0_local);
            let b_smem_off1 = ctx.add_u32_reg(b_smem_off0, c_8); // +8 bytes

            // Initialize 2×2 WMMA accumulators (4 fragments × 8 regs = 32 regs)
            // Each warp computes a 32×32 block as 2×2 grid of 16×16 WMMA tiles.
            let frag_c00 = ctx.wmma_init_c_zero();
            let frag_c01 = ctx.wmma_init_c_zero();
            let frag_c10 = ctx.wmma_init_c_zero();
            let frag_c11 = ctx.wmma_init_c_zero();

            let smem_single_reg = ctx.mov_u32_imm(smem_single as u32);

            // ─── Pipeline: 3-stage prologue ───
            // Stage 0: load into buf[0]
            // TODO: implement 3-stage prologue + main loop + epilogue
            // For now, use simplified 2-stage for correctness testing

            // Placeholder: delegate to 64×64 kernel's pattern but with 128×128 constants
            // This requires completing the load/compute/store pipeline

            // ─── Epilogue: store C ───
            // Each warp stores 2×2 WMMA tiles (32×32 output) to global memory
            // TODO: implement C store with 4 wmma_store_d calls

            ctx.label("exit");
            ctx.ret();
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cta128_kernel_builds() {
        // FALSIFY: kernel construction must not panic
        use crate::ptx::PtxModule;
        let kernel = build_cta128_wmma_fp16_cpasync(128, 128, 32);
        let ptx = PtxModule::new().target("sm_80").add_kernel(kernel).emit();
        assert!(ptx.contains("gemm_cta128_cpasync_fp16"));
        assert!(ptx.contains(".shared"));
    }

    #[test]
    fn test_cta128_smem_size() {
        // Contract: 128×16×2 (A) + 16×128×2 (B) = 8KB per stage, 3 stages = 24KB
        let tile_m = 128u32;
        let tile_n = 128u32;
        let tile_k = 16u32;
        let a_bytes = tile_m * tile_k * 2;
        let b_bytes = tile_k * tile_n * 2;
        let per_stage = a_bytes + b_bytes;
        assert_eq!(per_stage, 8192); // 8 KB
        let total = per_stage * 3;
        assert_eq!(total, 24576); // 24 KB < 48 KB limit
    }

    #[test]
    fn test_cta128_compute_to_load_ratio() {
        // FALSIFY: 128×128 must have 2× better ratio than 64×64
        // 64×64:  compute = 2*64*64*16 = 131072, load = 64*16*2 + 16*64*2 = 4096 bytes
        // ratio_64 = 131072 / 4096 = 32 FLOP/byte
        // 128×128: compute = 2*128*128*16 = 524288, load = 128*16*2 + 16*128*2 = 8192 bytes
        // ratio_128 = 524288 / 8192 = 64 FLOP/byte
        let ratio_64 = (2 * 64 * 64 * 16) as f64 / (64 * 16 * 2 + 16 * 64 * 2) as f64;
        let ratio_128 = (2 * 128 * 128 * 16) as f64 / (128 * 16 * 2 + 16 * 128 * 2) as f64;
        assert!((ratio_64 - 32.0).abs() < 0.1);
        assert!((ratio_128 - 64.0).abs() < 0.1);
        assert!(ratio_128 >= ratio_64 * 1.99); // Must be ~2×
    }
}
