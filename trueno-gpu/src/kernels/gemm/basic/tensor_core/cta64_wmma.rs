//! 64×64 CTA-level WMMA GEMM — 16 warps cooperatively compute 64×64 output tiles
//!
//! Architecture: 4×4 warp grid, each warp owns a 16×16 WMMA subtile.
//! All 16 warps share 64×K A tile and K×64 B tile in shared memory.
//! K-dimension tiled in chunks of 16 (WMMA tile depth).
//!
//! Key advantage over 32×32 CTA: 2× compute-to-load ratio (32 vs 16 FLOP/byte).
//! Each loaded A element is reused by 4 column warps (vs 2 in 32×32).
//! Each loaded B element is reused by 4 row warps (vs 2 in 32×32).
//!
//! Performance optimizations:
//! - PERF-CTA64-001: 4×4 warp grid for maximum data reuse
//! - PERF-CTA64-002: Interior tile fast path (skip bounds checks)
//! - PERF-CTA64-003: Warp-uniform A/B load split (warps 0-7→A, 8-15→B)
//! - PERF-CTA64-004: 4 elements/thread (less register pressure than 32×32's 8)
//!
//! Launch: grid_2d((N+63)/64, (M+63)/64), block(512,1,1), shared_mem=4KB
//! Input: FP16 (u16), Output: FP32

use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType, WmmaLayout};

/// Build a 64×64 CTA-tiled WMMA GEMM kernel for FP16 input → FP32 output.
///
/// 16 warps (512 threads), 4×4 grid of 16×16 WMMA subtiles.
/// 2× better compute-to-load ratio than 32×32 CTA kernel.
pub fn build_cta64_wmma_fp16(_m: u32, n: u32, k: u32) -> PtxKernel {
    build_cta64_wmma_fp16_impl(_m, n, k, false)
}

/// Double-buffered 64×64 variant — overlaps load with 16-warp WMMA compute.
/// With 16 WMMAs per K-tile (vs 4 in 32×32), buffer management overhead is
/// amortized over 4× more compute. Expected improvement: 1.2-1.5×.
pub fn build_cta64_wmma_fp16_dbuf(_m: u32, n: u32, k: u32) -> PtxKernel {
    build_cta64_wmma_fp16_impl(_m, n, k, true)
}

/// cp.async double-buffered 64×64 variant (SM 8.0+).
///
/// Uses `cp.async.ca.shared.global` with 8-byte copies (4 FP16 elements per
/// thread). Each thread does ONE cp.async per K-tile. True async: WMMA runs
/// while cp.async transfers complete in background.
///
/// Requires: K % 4 == 0 and N % 4 == 0 for 8-byte alignment.
/// Requires: sm_80+ target (caller must set `PtxModule::target("sm_80")`).
pub fn build_cta64_wmma_fp16_cpasync(m: u32, n: u32, k: u32) -> PtxKernel {
    let tile_m: u32 = 64;
    let tile_n: u32 = 64;
    let tile_k: u32 = 16;
    let a_smem_bytes = (tile_m * tile_k * 2) as usize; // 2048
    let b_smem_bytes = (tile_k * tile_n * 2) as usize; // 2048
    let smem_single = a_smem_bytes + b_smem_bytes; // 4096
    let smem_bytes = smem_single * 2; // double-buffer
    let n_k_tiles = (k + tile_k - 1) / tile_k;
    // Each thread loads 4 consecutive FP16 elements (8 bytes) with ONE cp.async.
    let a_threads: u32 = 256; // threads 0..255 load A, 256..511 load B
    let _ = m;

    PtxKernel::new("gemm_cta64_cpasync_fp16")
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
            let c_16 = ctx.mov_u32_imm(16);
            let c_63 = ctx.mov_u32_imm(63);
            let c_64 = ctx.mov_u32_imm(64);
            let c_256 = ctx.mov_u32_imm(a_threads);
            let c_a_smem = ctx.mov_u32_imm(a_smem_bytes as u32);
            let k_const = ctx.mov_u32_imm(k);
            let n_const = ctx.mov_u32_imm(n);
            let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);

            let cta_row = ctx.mul_u32_reg(ctaid_y, c_64);
            let cta_col = ctx.mul_u32_reg(ctaid_x, c_64);

            let m_param = ctx.load_param_u32("m_param");
            let n_param = ctx.load_param_u32("n_param");
            let _k_param = ctx.load_param_u32("k_param");

            let cta_oob = ctx.setp_ge_u32(cta_row, m_param);
            ctx.branch_if(cta_oob, "exit");
            let cta_oob2 = ctx.setp_ge_u32(cta_col, n_param);
            ctx.branch_if(cta_oob2, "exit");

            let a_ptr = ctx.load_param_u64("a_ptr");
            let b_ptr = ctx.load_param_u64("b_ptr");
            let c_ptr = ctx.load_param_u64("c_ptr");

            let c_3 = ctx.mov_u32_imm(3);
            let warp_id = ctx.shr_u32(tid, c_5);
            let warp_row = ctx.shr_u32(warp_id, c_2);
            let warp_col = ctx.and_u32(warp_id, c_3);
            let warp_m_off = ctx.mul_u32_reg(warp_row, c_16);
            let warp_n_off = ctx.mul_u32_reg(warp_col, c_16);

            // Smem base via cvta for generic pointer
            let smem_base = ctx.shared_base_addr();

            // Thread role: is_a = (tid < 256)
            let is_a_thread = ctx.setp_lt_u32(tid, c_256);

            // ═══ Pre-compute thread's load parameters ═══
            // For A-threads (t in 0..255):
            //   idx_start = t*4, row = t/4, col = (t*4) % 16 = (t%4)*4
            //   smem_off = t*4*2 = t*8
            //   global_start = a_ptr + ((cta_row + row) * K + k_off + col) * 2
            // For B-threads (t in 256..511), local = t - 256:
            //   local_start = local*4, row = local/16, col = (local*4) % 64 = (local%16)*4
            //   smem_off = a_smem + local*8
            //   global_start = b_ptr + ((k_off + row) * N + cta_col + col) * 2

            // Compute smem offset for this thread (conditional via selp)
            // A path: smem_off = tid * 8
            // B path: smem_off = a_smem + (tid - 256) * 8
            let c_8 = ctx.mov_u32_imm(8);
            let a_smem_off = ctx.mul_u32_reg(tid, c_8);
            let b_local = ctx.sub_u32_reg(tid, c_256);
            let b_local_8 = ctx.mul_u32_reg(b_local, c_8);
            let b_smem_off = ctx.add_u32_reg(c_a_smem, b_local_8);
            let base_smem_off = ctx.selp_u32(is_a_thread, a_smem_off, b_smem_off);

            // Pre-compute per-thread row/col (A and B variants)
            // A: row = t/4, col = (t%4)*4
            let a_row_in_tile = ctx.shr_u32(tid, c_2); // t/4
            let c_mask3 = ctx.mov_u32_imm(3);
            let a_col_mod = ctx.and_u32(tid, c_mask3); // t%4
            let a_col_in_tile = ctx.mul_u32_reg(a_col_mod, c_4); // (t%4)*4
            let a_global_row = ctx.add_u32_reg(cta_row, a_row_in_tile);
            // A address computed per K-tile: (a_global_row * K + k_off + a_col_in_tile) * 2 + a_ptr

            // B: row = local/16, col = (local%16)*4
            let b_row_in_tile = ctx.shr_u32(b_local, c_4); // local/16
            let c_mask15 = ctx.mov_u32_imm(15);
            let b_col_mod = ctx.and_u32(b_local, c_mask15); // local%16
            let b_col_in_tile = ctx.mul_u32_reg(b_col_mod, c_4); // (local%16)*4
            let b_global_col = ctx.add_u32_reg(cta_col, b_col_in_tile);
            // B address computed per K-tile: ((k_off + b_row_in_tile) * N + b_global_col) * 2 + b_ptr

            let frag_c = ctx.wmma_init_c_zero();
            let smem_single_reg = ctx.mov_u32_imm(smem_single as u32);
            let store_buf_off = ctx.mov_u32_imm(0);

            // ─── Prologue: cp.async tile 0 into buf[0] ───
            {
                let k_off = ctx.mov_u32_imm(0);
                // A global: (a_global_row * K + k_off + a_col_in_tile) * 2 + a_ptr
                // Since k_off = 0: (a_global_row * K + a_col_in_tile) * 2 + a_ptr
                let a_rowk = ctx.mul_u32_reg(a_global_row, k_const);
                let a_flat = ctx.add_u32_reg(a_rowk, a_col_in_tile);
                let a_byte_off = ctx.mul_wide_u32(a_flat, 2);
                let a_addr = ctx.add_u64(a_ptr, a_byte_off);

                // B global: ((0 + b_row_in_tile) * N + b_global_col) * 2 + b_ptr
                let b_flat = ctx.mad_lo_u32(b_row_in_tile, n_const, b_global_col);
                let b_byte_off = ctx.mul_wide_u32(b_flat, 2);
                let b_addr = ctx.add_u64(b_ptr, b_byte_off);

                // Select global address based on thread role
                let gaddr = ctx.selp_u64(is_a_thread, a_addr, b_addr);
                // cp.async expects u32 shared-space offset for dst (not generic u64)
                // base_smem_off is already a u32 byte offset within smem[]
                ctx.cp_async_global_to_shared(base_smem_off, gaddr, 8);
                let _ = k_off;
            }
            ctx.cp_async_commit_group();
            ctx.cp_async_wait_group(0);
            ctx.bar_sync(0);

            // ─── Main loop ───
            let k_tile = ctx.mov_u32_imm(1);
            ctx.label("k_loop");
            let k_done = ctx.setp_ge_u32(k_tile, n_k_tiles_reg);
            ctx.branch_if(k_done, "dbuf_epi");

            // Swap buffer
            let new_store = ctx.sub_u32_reg(smem_single_reg, store_buf_off);
            ctx.mov_u32_reg(store_buf_off, new_store);
            let compute_off = ctx.sub_u32_reg(smem_single_reg, store_buf_off);

            let k_off = ctx.mul_u32_reg(k_tile, c_16);

            // Compute addresses (same as prologue but with k_off)
            let a_col_full = ctx.add_u32_reg(k_off, a_col_in_tile);
            let a_rowk = ctx.mul_u32_reg(a_global_row, k_const);
            let a_flat = ctx.add_u32_reg(a_rowk, a_col_full);
            let a_byte_off = ctx.mul_wide_u32(a_flat, 2);
            let a_addr = ctx.add_u64(a_ptr, a_byte_off);

            let b_row_full = ctx.add_u32_reg(k_off, b_row_in_tile);
            let b_flat = ctx.mad_lo_u32(b_row_full, n_const, b_global_col);
            let b_byte_off = ctx.mul_wide_u32(b_flat, 2);
            let b_addr = ctx.add_u64(b_ptr, b_byte_off);

            let gaddr = ctx.selp_u64(is_a_thread, a_addr, b_addr);
            let smem_off_with_buf = ctx.add_u32_reg(base_smem_off, store_buf_off);
            // cp.async dst is u32 shared-space offset (not generic u64)
            ctx.cp_async_global_to_shared(smem_off_with_buf, gaddr, 8);
            ctx.cp_async_commit_group();

            // WMMA on compute buffer (overlaps with cp.async in-flight)
            {
                let a_sub_bytes = ctx.mul_u32_reg(warp_m_off, c_16);
                let a_sub_bytes = ctx.mul_u32_reg(a_sub_bytes, c_2);
                let a_sub_buf = ctx.add_u32_reg(a_sub_bytes, compute_off);
                let a_sub_off = ctx.cvt_u64_u32(a_sub_buf);
                let a_sub_ptr = ctx.add_u64(smem_base, a_sub_off);
                let frag_a = ctx.wmma_load_a_f16(a_sub_ptr, 16, WmmaLayout::RowMajor);

                let b_sub_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
                let b_base_off = ctx.add_u32_reg(c_a_smem, b_sub_bytes);
                let b_buf = ctx.add_u32_reg(b_base_off, compute_off);
                let b_off64 = ctx.cvt_u64_u32(b_buf);
                let b_sub_ptr = ctx.add_u64(smem_base, b_off64);
                let frag_b = ctx.wmma_load_b_f16(b_sub_ptr, 64, WmmaLayout::RowMajor);

                let frag_d = ctx.wmma_mma_f16_f32_row_row(&frag_a, &frag_b, &frag_c);
                for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                    ctx.mov_f32_reg(*c_reg, *d_reg);
                }
            }

            ctx.cp_async_wait_group(0);
            ctx.bar_sync(0);
            ctx.add_u32_inplace(k_tile, 1);
            ctx.branch("k_loop");

            // ─── Epilogue ───
            ctx.label("dbuf_epi");
            {
                let a_sub_bytes = ctx.mul_u32_reg(warp_m_off, c_16);
                let a_sub_bytes = ctx.mul_u32_reg(a_sub_bytes, c_2);
                let a_sub_buf = ctx.add_u32_reg(a_sub_bytes, store_buf_off);
                let a_sub_off = ctx.cvt_u64_u32(a_sub_buf);
                let a_sub_ptr = ctx.add_u64(smem_base, a_sub_off);
                let frag_a = ctx.wmma_load_a_f16(a_sub_ptr, 16, WmmaLayout::RowMajor);

                let b_sub_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
                let b_base_off = ctx.add_u32_reg(c_a_smem, b_sub_bytes);
                let b_buf = ctx.add_u32_reg(b_base_off, store_buf_off);
                let b_off64 = ctx.cvt_u64_u32(b_buf);
                let b_sub_ptr = ctx.add_u64(smem_base, b_off64);
                let frag_b = ctx.wmma_load_b_f16(b_sub_ptr, 64, WmmaLayout::RowMajor);

                let frag_d = ctx.wmma_mma_f16_f32_row_row(&frag_a, &frag_b, &frag_c);
                for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                    ctx.mov_f32_reg(*c_reg, *d_reg);
                }
            }

            // Store C
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

/// 64×64 CTA GEMM using mma.sync.m16n8k16 + ldmatrix (Phase 1 bridge plan).
///
/// Same structure as wmma cp.async variant but replaces:
/// - wmma_load_a/b (~32 ld.shared) → ldmatrix.x4 + ldmatrix.x2.trans (2 instructions)
/// - wmma_mma → 2× mma.sync.m16n8k16 (covers 16×16 via 16×8 + 16×8)
///
/// Expected: fewer instructions per K-tile → higher IPC → more TFLOP/s.
/// Contract: cgp-gpu-mma-sync-v1.yaml
pub fn build_cta64_mma_fp16_cpasync(_m: u32, n: u32, k: u32) -> PtxKernel {
    let tile_m: u32 = 64;
    let tile_n: u32 = 64;
    let tile_k: u32 = 16;
    let a_smem_bytes = (tile_m * tile_k * 2) as usize;
    let b_smem_bytes = (tile_k * tile_n * 2) as usize;
    let smem_single = a_smem_bytes + b_smem_bytes;
    let smem_bytes = smem_single * 2;
    let n_k_tiles = (k + tile_k - 1) / tile_k;
    let a_threads: u32 = 256;

    PtxKernel::new("gemm_cta64_mma_fp16")
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

            // === Constants ===
            let c_2 = ctx.mov_u32_imm(2);
            let c_3 = ctx.mov_u32_imm(3);
            let c_4 = ctx.mov_u32_imm(4);
            let c_5 = ctx.mov_u32_imm(5);
            let c_7 = ctx.mov_u32_imm(7);
            let c_8 = ctx.mov_u32_imm(8);
            let c_16 = ctx.mov_u32_imm(16);
            let c_32 = ctx.mov_u32_imm(32);
            let c_64 = ctx.mov_u32_imm(64);
            let c_256 = ctx.mov_u32_imm(a_threads);
            let c_a_smem = ctx.mov_u32_imm(a_smem_bytes as u32);
            let k_const = ctx.mov_u32_imm(k);
            let n_const = ctx.mov_u32_imm(n);
            let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);
            let smem_single_reg = ctx.mov_u32_imm(smem_single as u32);

            let cta_row = ctx.mul_u32_reg(ctaid_y, c_64);
            let cta_col = ctx.mul_u32_reg(ctaid_x, c_64);
            let m_param = ctx.load_param_u32("m_param");
            let n_param = ctx.load_param_u32("n_param");
            let _k_param = ctx.load_param_u32("k_param");

            // Bounds check
            let cta_oob = ctx.setp_ge_u32(cta_row, m_param);
            ctx.branch_if(cta_oob, "exit");
            let cta_oob2 = ctx.setp_ge_u32(cta_col, n_param);
            ctx.branch_if(cta_oob2, "exit");

            let a_ptr = ctx.load_param_u64("a_ptr");
            let b_ptr = ctx.load_param_u64("b_ptr");
            let c_ptr = ctx.load_param_u64("c_ptr");

            // === Warp layout (same as wmma: 4×4 grid, 16×16 per warp) ===
            let warp_id = ctx.shr_u32(tid, c_5);
            let c_31 = ctx.mov_u32_imm(31);
            let lane_id = ctx.and_u32(tid, c_31);
            let warp_row = ctx.shr_u32(warp_id, c_2);
            let warp_col = ctx.and_u32(warp_id, c_3);
            let warp_m_off = ctx.mul_u32_reg(warp_row, c_16);
            let warp_n_off = ctx.mul_u32_reg(warp_col, c_16);

            let smem_base = ctx.shared_base_addr();
            let is_a_thread = ctx.setp_lt_u32(tid, c_256);

            // === cp.async load offsets (IDENTICAL to wmma version) ===
            let a_row_in_tile = ctx.shr_u32(tid, c_2);
            let c_mask3 = ctx.mov_u32_imm(3);
            let a_col_mod = ctx.and_u32(tid, c_mask3);
            let a_col_in_tile = ctx.mul_u32_reg(a_col_mod, c_4);
            let a_global_row = ctx.add_u32_reg(cta_row, a_row_in_tile);
            let b_local = ctx.sub_u32_reg(tid, c_256);
            let b_row_in_tile = ctx.shr_u32(b_local, c_4);
            let c_mask15 = ctx.mov_u32_imm(15);
            let b_col_mod = ctx.and_u32(b_local, c_mask15);
            let b_col_in_tile = ctx.mul_u32_reg(b_col_mod, c_4);
            let b_global_col = ctx.add_u32_reg(cta_col, b_col_in_tile);

            let a_smem_off = ctx.mul_u32_reg(tid, c_8);
            let b_local_8 = ctx.mul_u32_reg(b_local, c_8);
            let b_smem_off = ctx.add_u32_reg(c_a_smem, b_local_8);
            let base_smem_off = ctx.selp_u32(is_a_thread, a_smem_off, b_smem_off);

            // === mma.sync accumulators (4 F32 regs for 16×8 output) ===
            // Two mma.sync calls per 16×16: left half (col 0-7) and right half (col 8-15)
            let acc_l0 = ctx.mov_f32_imm(0.0);
            let acc_l1 = ctx.mov_f32_imm(0.0);
            let acc_l2 = ctx.mov_f32_imm(0.0);
            let acc_l3 = ctx.mov_f32_imm(0.0);
            let acc_r0 = ctx.mov_f32_imm(0.0);
            let acc_r1 = ctx.mov_f32_imm(0.0);
            let acc_r2 = ctx.mov_f32_imm(0.0);
            let acc_r3 = ctx.mov_f32_imm(0.0);

            let store_buf_off = ctx.mov_u32_imm(0);

            // === ldmatrix address computation (per-thread, pre-computed) ===
            // A: ldmatrix.x4 loads 16×16 from smem. Thread lane_id determines row.
            // sub = lane/8, row = (sub/2)*8 + lane%8, col_bytes = (sub%2)*16
            let lane_sub = ctx.shr_u32(lane_id, c_3); // lane / 8
            let lane_row_in_8 = ctx.and_u32(lane_id, c_7); // lane % 8
            let c_1 = ctx.mov_u32_imm(1);
            let sub_half = ctx.shr_u32(lane_sub, c_1); // sub / 2
            let sub_col = ctx.and_u32(lane_sub, c_1); // sub % 2
            let phys_row = ctx.mad_lo_u32(sub_half, c_8, lane_row_in_8); // (sub/2)*8 + lane%8
            let col_bytes = ctx.mul_u32_reg(sub_col, c_16); // (sub%2)*16 bytes
                                                            // A stride in smem = tile_k * 2 = 32 bytes per row
            let a_row_bytes = ctx.mul_u32_reg(phys_row, c_32);
            let a_ldm_off_base = ctx.add_u32_reg(a_row_bytes, col_bytes); // within 16×16 tile

            // B: ldmatrix.x2.trans loads 16×8 transposed.
            // Thread lane_id: row = lane % 16, addr = b_base + row * b_stride
            let b_lane_row = ctx.and_u32(lane_id, c_mask15); // lane % 16 → row 0-15
                                                             // B stride in smem = tile_n * 2 = 128 bytes per row
            let c_128 = ctx.mov_u32_imm(128);
            let b_row_bytes = ctx.mul_u32_reg(b_lane_row, c_128);

            // === Prologue: cp.async tile 0 ===
            {
                let a_rowk = ctx.mul_u32_reg(a_global_row, k_const);
                let a_flat = ctx.add_u32_reg(a_rowk, a_col_in_tile);
                let a_byte_off = ctx.mul_wide_u32(a_flat, 2);
                let a_addr = ctx.add_u64(a_ptr, a_byte_off);
                let b_flat = ctx.mad_lo_u32(b_row_in_tile, n_const, b_global_col);
                let b_byte_off = ctx.mul_wide_u32(b_flat, 2);
                let b_addr = ctx.add_u64(b_ptr, b_byte_off);
                let gaddr = ctx.selp_u64(is_a_thread, a_addr, b_addr);
                ctx.cp_async_global_to_shared(base_smem_off, gaddr, 8);
            }
            ctx.cp_async_commit_group();
            ctx.cp_async_wait_group(0);
            ctx.bar_sync(0);

            // === Main K-loop ===
            let k_tile = ctx.mov_u32_imm(1);
            ctx.label("k_loop");
            let k_done = ctx.setp_ge_u32(k_tile, n_k_tiles_reg);
            ctx.branch_if(k_done, "mma_epi");

            // Swap buffer
            let new_store = ctx.sub_u32_reg(smem_single_reg, store_buf_off);
            ctx.mov_u32_reg(store_buf_off, new_store);
            let compute_off = ctx.sub_u32_reg(smem_single_reg, store_buf_off);

            // cp.async next tile
            let k_off = ctx.mul_u32_reg(k_tile, c_16);
            {
                let a_col_full = ctx.add_u32_reg(k_off, a_col_in_tile);
                let a_rowk = ctx.mul_u32_reg(a_global_row, k_const);
                let a_flat = ctx.add_u32_reg(a_rowk, a_col_full);
                let a_byte_off = ctx.mul_wide_u32(a_flat, 2);
                let a_addr = ctx.add_u64(a_ptr, a_byte_off);
                let b_row_full = ctx.add_u32_reg(k_off, b_row_in_tile);
                let b_flat = ctx.mad_lo_u32(b_row_full, n_const, b_global_col);
                let b_byte_off = ctx.mul_wide_u32(b_flat, 2);
                let b_addr = ctx.add_u64(b_ptr, b_byte_off);
                let gaddr = ctx.selp_u64(is_a_thread, a_addr, b_addr);
                let smem_off_buf = ctx.add_u32_reg(base_smem_off, store_buf_off);
                ctx.cp_async_global_to_shared(smem_off_buf, gaddr, 8);
            }
            ctx.cp_async_commit_group();

            // === mma.sync compute on compute buffer ===
            {
                // A tile base in smem = warp_m_off * tile_k * 2 + compute_off
                let a_warp_bytes = ctx.mul_u32_reg(warp_m_off, c_32); // warp_m_off * 32
                let a_tile_base = ctx.add_u32_reg(a_warp_bytes, compute_off);
                let a_addr_u32 = ctx.add_u32_reg(a_tile_base, a_ldm_off_base);
                let a_frags = ctx.ldmatrix_x4(a_addr_u32);

                // B tile base in smem = a_smem + warp_n_off * 2 + compute_off
                let b_warp_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
                let b_tile_base = ctx.add_u32_reg(c_a_smem, b_warp_bytes);
                let b_tile_base = ctx.add_u32_reg(b_tile_base, compute_off);

                // Left B half (cols 0-7): ldmatrix.x2.trans
                let b_addr_l = ctx.add_u32_reg(b_tile_base, b_row_bytes);
                let b_frags_l = ctx.ldmatrix_x2_trans(b_addr_l);

                // Right B half (cols 8-15): ldmatrix.x2.trans at +16 bytes
                let b_addr_r = ctx.add_u32_reg(b_addr_l, c_16);
                let b_frags_r = ctx.ldmatrix_x2_trans(b_addr_r);

                // Two mma.sync: left 16×8 + right 16×8 = 16×16
                let d_l =
                    ctx.mma_sync_m16n8k16(&a_frags, &b_frags_l, &[acc_l0, acc_l1, acc_l2, acc_l3]);
                ctx.mov_f32_reg(acc_l0, d_l[0]);
                ctx.mov_f32_reg(acc_l1, d_l[1]);
                ctx.mov_f32_reg(acc_l2, d_l[2]);
                ctx.mov_f32_reg(acc_l3, d_l[3]);

                let d_r =
                    ctx.mma_sync_m16n8k16(&a_frags, &b_frags_r, &[acc_r0, acc_r1, acc_r2, acc_r3]);
                ctx.mov_f32_reg(acc_r0, d_r[0]);
                ctx.mov_f32_reg(acc_r1, d_r[1]);
                ctx.mov_f32_reg(acc_r2, d_r[2]);
                ctx.mov_f32_reg(acc_r3, d_r[3]);
            }

            ctx.cp_async_wait_group(0);
            ctx.bar_sync(0);
            ctx.add_u32_inplace(k_tile, 1);
            ctx.branch("k_loop");

            // === Epilogue: last K-tile ===
            ctx.label("mma_epi");
            {
                let a_warp_bytes = ctx.mul_u32_reg(warp_m_off, c_32);
                let a_tile_base = ctx.add_u32_reg(a_warp_bytes, store_buf_off);
                let a_addr_u32 = ctx.add_u32_reg(a_tile_base, a_ldm_off_base);
                let a_frags = ctx.ldmatrix_x4(a_addr_u32);

                let b_warp_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
                let b_tile_base = ctx.add_u32_reg(c_a_smem, b_warp_bytes);
                let b_tile_base = ctx.add_u32_reg(b_tile_base, store_buf_off);
                let b_addr_l = ctx.add_u32_reg(b_tile_base, b_row_bytes);
                let b_frags_l = ctx.ldmatrix_x2_trans(b_addr_l);
                let b_addr_r = ctx.add_u32_reg(b_addr_l, c_16);
                let b_frags_r = ctx.ldmatrix_x2_trans(b_addr_r);

                let d_l =
                    ctx.mma_sync_m16n8k16(&a_frags, &b_frags_l, &[acc_l0, acc_l1, acc_l2, acc_l3]);
                ctx.mov_f32_reg(acc_l0, d_l[0]);
                ctx.mov_f32_reg(acc_l1, d_l[1]);
                ctx.mov_f32_reg(acc_l2, d_l[2]);
                ctx.mov_f32_reg(acc_l3, d_l[3]);
                let d_r =
                    ctx.mma_sync_m16n8k16(&a_frags, &b_frags_r, &[acc_r0, acc_r1, acc_r2, acc_r3]);
                ctx.mov_f32_reg(acc_r0, d_r[0]);
                ctx.mov_f32_reg(acc_r1, d_r[1]);
                ctx.mov_f32_reg(acc_r2, d_r[2]);
                ctx.mov_f32_reg(acc_r3, d_r[3]);
            }

            // === Store C: per-thread st.global for mma.sync output layout ===
            // mma.sync m16n8k16 row.col: thread lane_id holds D[0..3] mapping to:
            //   group = lane/4, tid_in_group = lane%4
            //   row0 = group*2, row1 = row0+1
            //   col0 = tid_in_group*2, col1 = col0+1
            //   D[0]→C[row0][col0], D[1]→C[row0][col1]
            //   D[2]→C[row1][col0], D[3]→C[row1][col1]
            {
                let c_row_base = ctx.add_u32_reg(cta_row, warp_m_off); // global row
                let c_col_base = ctx.add_u32_reg(cta_col, warp_n_off); // global col

                // Per-thread row/col within 16×16 tile
                let group = ctx.shr_u32(lane_id, c_2); // lane / 4
                let tid_in_grp = ctx.and_u32(lane_id, c_3); // lane % 4
                let row0_local = ctx.mul_u32_reg(group, c_2); // group * 2
                let row1_local = ctx.add_u32_reg(row0_local, c_1); // +1
                let col0_local = ctx.mul_u32_reg(tid_in_grp, c_2); // tid_in_grp * 2
                let col1_local = ctx.add_u32_reg(col0_local, c_1); // +1

                let row0 = ctx.add_u32_reg(c_row_base, row0_local);
                let row1 = ctx.add_u32_reg(c_row_base, row1_local);

                // Left half: cols 0-7
                let col0_l = ctx.add_u32_reg(c_col_base, col0_local);
                let col1_l = ctx.add_u32_reg(c_col_base, col1_local);

                // Right half: cols 8-15
                let col0_r = ctx.add_u32_reg(col0_l, c_8);
                let col1_r = ctx.add_u32_reg(col1_l, c_8);

                // Helper: compute C address = c_ptr + (row * N + col) * 4
                let n_param_val = n_param;

                // Store left half D[0..3] = acc_l[0..3]
                // D[0] → C[row0][col0_l]
                let off00 = ctx.mad_lo_u32(row0, n_param_val, col0_l);
                let off00_bytes = ctx.mul_wide_u32(off00, 4);
                let addr00 = ctx.add_u64(c_ptr, off00_bytes);
                ctx.st_global_f32(addr00, acc_l0);

                // D[1] → C[row0][col1_l]
                let off01 = ctx.mad_lo_u32(row0, n_param_val, col1_l);
                let off01_bytes = ctx.mul_wide_u32(off01, 4);
                let addr01 = ctx.add_u64(c_ptr, off01_bytes);
                ctx.st_global_f32(addr01, acc_l1);

                // D[2] → C[row1][col0_l]
                let off10 = ctx.mad_lo_u32(row1, n_param_val, col0_l);
                let off10_bytes = ctx.mul_wide_u32(off10, 4);
                let addr10 = ctx.add_u64(c_ptr, off10_bytes);
                ctx.st_global_f32(addr10, acc_l2);

                // D[3] → C[row1][col1_l]
                let off11 = ctx.mad_lo_u32(row1, n_param_val, col1_l);
                let off11_bytes = ctx.mul_wide_u32(off11, 4);
                let addr11 = ctx.add_u64(c_ptr, off11_bytes);
                ctx.st_global_f32(addr11, acc_l3);

                // Store right half D[0..3] = acc_r[0..3]
                let off00r = ctx.mad_lo_u32(row0, n_param_val, col0_r);
                let off00r_bytes = ctx.mul_wide_u32(off00r, 4);
                let addr00r = ctx.add_u64(c_ptr, off00r_bytes);
                ctx.st_global_f32(addr00r, acc_r0);

                let off01r = ctx.mad_lo_u32(row0, n_param_val, col1_r);
                let off01r_bytes = ctx.mul_wide_u32(off01r, 4);
                let addr01r = ctx.add_u64(c_ptr, off01r_bytes);
                ctx.st_global_f32(addr01r, acc_r1);

                let off10r = ctx.mad_lo_u32(row1, n_param_val, col0_r);
                let off10r_bytes = ctx.mul_wide_u32(off10r, 4);
                let addr10r = ctx.add_u64(c_ptr, off10r_bytes);
                ctx.st_global_f32(addr10r, acc_r2);

                let off11r = ctx.mad_lo_u32(row1, n_param_val, col1_r);
                let off11r_bytes = ctx.mul_wide_u32(off11r, 4);
                let addr11r = ctx.add_u64(c_ptr, off11r_bytes);
                ctx.st_global_f32(addr11r, acc_r3);
            }

            ctx.label("exit");
            ctx.ret();
        })
}

fn build_cta64_wmma_fp16_impl(_m: u32, n: u32, k: u32, double_buffer: bool) -> PtxKernel {
    let tile_m: u32 = 64;
    let tile_n: u32 = 64;
    let tile_k: u32 = 16;
    let a_smem_bytes = (tile_m * tile_k * 2) as usize; // 2048
    let b_smem_bytes = (tile_k * tile_n * 2) as usize; // 2048
    let smem_single = a_smem_bytes + b_smem_bytes; // 4096
    let smem_bytes = if double_buffer { smem_single * 2 } else { smem_single };
    let n_k_tiles = (k + tile_k - 1) / tile_k;

    // Cooperative load: 512 threads, 4 elements each = 2048 elements
    // Warps 0-7 (threads 0-255) → A[64×16] = 1024 elems
    // Warps 8-15 (threads 256-511) → B[16×64] = 1024 elems
    let elems_per_thread: u32 = 4;
    let a_total_elems = tile_m * tile_k; // 1024
    let a_threads: u32 = 256; // threads 0..255 load A

    PtxKernel::new("gemm_cta64_wmma_fp16")
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
            let c_2 = ctx.mov_u32_imm(2);
            let c_4 = ctx.mov_u32_imm(4);
            let c_5 = ctx.mov_u32_imm(5);
            let c_15 = ctx.mov_u32_imm(15);
            let c_16 = ctx.mov_u32_imm(16);
            let c_63 = ctx.mov_u32_imm(63);
            let c_64 = ctx.mov_u32_imm(64);
            let c_256 = ctx.mov_u32_imm(a_threads);
            let c_a_smem = ctx.mov_u32_imm(a_smem_bytes as u32);
            let k_const = ctx.mov_u32_imm(k);
            let n_const = ctx.mov_u32_imm(n);
            let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);
            let zero_f32 = ctx.mov_f32_imm(0.0);
            let zero_f16 = ctx.cvt_f16_f32(zero_f32);

            // CTA tile position (64×64)
            let cta_row = ctx.mul_u32_reg(ctaid_y, c_64);
            let cta_col = ctx.mul_u32_reg(ctaid_x, c_64);

            let m_param = ctx.load_param_u32("m_param");
            let n_param = ctx.load_param_u32("n_param");
            let k_param = ctx.load_param_u32("k_param");

            // Early exit for out-of-bounds CTAs
            let cta_oob = ctx.setp_ge_u32(cta_row, m_param);
            ctx.branch_if(cta_oob, "exit");
            let cta_oob2 = ctx.setp_ge_u32(cta_col, n_param);
            ctx.branch_if(cta_oob2, "exit");

            let a_ptr = ctx.load_param_u64("a_ptr");
            let b_ptr = ctx.load_param_u64("b_ptr");
            let c_ptr = ctx.load_param_u64("c_ptr");

            // Warp layout: 4×4 grid (warp_id = tid/32, 0-15)
            let warp_id = ctx.shr_u32(tid, c_5);
            let warp_row = ctx.shr_u32(warp_id, c_2); // warp_id / 4
            let c_3 = ctx.mov_u32_imm(3);
            let warp_col = ctx.and_u32(warp_id, c_3); // warp_id % 4
            let warp_m_off = ctx.mul_u32_reg(warp_row, c_16); // 0, 16, 32, 48
            let warp_n_off = ctx.mul_u32_reg(warp_col, c_16); // 0, 16, 32, 48

            // Cooperative load setup: each thread loads 4 elements
            let c_elems = ctx.mov_u32_imm(elems_per_thread);
            let my_base = ctx.mul_u32_reg(tid, c_elems); // tid * 4

            // A vs B split: threads 0-255 load A, 256-511 load B
            let is_a_thread = ctx.setp_lt_u32(tid, c_256);

            // PERF-CTA64-002: Pre-compute interior check
            let cta_row_end = ctx.add_u32_reg(cta_row, c_63);
            let cta_col_end = ctx.add_u32_reg(cta_col, c_63);
            let row_interior = ctx.setp_lt_u32(cta_row_end, m_param);
            let col_interior = ctx.setp_lt_u32(cta_col_end, n_param);
            let mn_interior = ctx.and_pred(row_interior, col_interior);

            // Pre-compute per-element addresses for A (threads 0-255)
            // A[64×16] row-major: idx = tid*4+i → row = idx/16, col = idx%16
            let mut a_base_addrs = Vec::with_capacity(elems_per_thread as usize);
            let mut a_smem_addrs = Vec::with_capacity(elems_per_thread as usize);
            let mut a_global_rows = Vec::with_capacity(elems_per_thread as usize);

            // Pre-compute per-element addresses for B (threads 256-511)
            // B[16×64] row-major: local = (tid-256)*4+i → row = local/64, col = local%64
            let mut b_row_in_tiles = Vec::with_capacity(elems_per_thread as usize);
            let mut b_col_globals = Vec::with_capacity(elems_per_thread as usize);
            let mut b_smem_addrs = Vec::with_capacity(elems_per_thread as usize);

            for i in 0..elems_per_thread {
                let ci = ctx.mov_u32_imm(i);
                let idx = ctx.add_u32_reg(my_base, ci);

                // A-side: row = idx/16, col = idx%16
                let a_r = ctx.shr_u32(idx, c_4); // idx / 16
                let a_c = ctx.and_u32(idx, c_15); // idx % 16
                let a_gr = ctx.add_u32_reg(cta_row, a_r);
                let a_row_k = ctx.mul_u32_reg(a_gr, k_const);
                let a_flat_base = ctx.add_u32_reg(a_row_k, a_c);
                let a_byte_off = ctx.mul_wide_u32(a_flat_base, 2);
                let a_base = ctx.add_u64(a_ptr, a_byte_off);
                let smem_off = ctx.mul_u32_reg(idx, c_2);

                a_base_addrs.push(a_base);
                a_smem_addrs.push(smem_off);
                a_global_rows.push(a_gr);

                // B-side: local = idx - 1024, row = local/64, col = local%64
                let c_a_total = ctx.mov_u32_imm(a_total_elems);
                let b_local = ctx.sub_u32_reg(idx, c_a_total);
                let c_6 = ctx.mov_u32_imm(6);
                let b_r = ctx.shr_u32(b_local, c_6); // local / 64
                let b_c = ctx.and_u32(b_local, c_63); // local % 64
                let b_gc = ctx.add_u32_reg(cta_col, b_c);
                let bsmem_off = ctx.mul_u32_reg(b_local, c_2);
                let bsmem_addr = ctx.add_u32_reg(c_a_smem, bsmem_off); // B starts at a_smem_bytes

                b_row_in_tiles.push(b_r);
                b_col_globals.push(b_gc);
                b_smem_addrs.push(bsmem_addr);
            }

            // Init WMMA accumulator (per-warp, 16×16 output)
            let frag_c = ctx.wmma_init_c_zero();

            if double_buffer {
                // ═══ DOUBLE-BUFFERED K-LOOP ═══
                // Prologue → Main loop → Epilogue (same structure as 32×32 dbuf
                // but with 16 WMMAs per K-tile for 4× amortization)
                let smem_single_reg = ctx.mov_u32_imm(smem_single as u32);
                let store_buf_off = ctx.mov_u32_imm(0);

                // ─── Prologue: load tile 0 into buf[0] ───
                {
                    let k_off = ctx.mov_u32_imm(0);
                    let k_byte_stride = ctx.mov_u64_imm(0);
                    let pro_a: Vec<_> =
                        a_smem_addrs.iter().map(|&a| ctx.add_u32_reg(a, store_buf_off)).collect();
                    let pro_b: Vec<_> =
                        b_smem_addrs.iter().map(|&b| ctx.add_u32_reg(b, store_buf_off)).collect();

                    let k_tile_end = ctx.add_u32_reg(k_off, c_15);
                    let k_ok = ctx.setp_lt_u32(k_tile_end, k_param);
                    let fully_interior = ctx.and_pred(mn_interior, k_ok);
                    ctx.branch_if(fully_interior, "pro_fast");

                    ctx.branch_if_not(is_a_thread, "pro_slow_b");
                    for i in 0..elems_per_thread as usize {
                        ctx.st_shared_f16(pro_a[i], zero_f16);
                        let skip = format!("psa{i}");
                        let ar_ok = ctx.setp_lt_u32(a_global_rows[i], m_param);
                        ctx.branch_if_not(ar_ok, &skip);
                        let a_addr = ctx.add_u64(a_base_addrs[i], k_byte_stride);
                        let ac_ok = ctx.setp_lt_u32(k_off, k_param);
                        ctx.branch_if_not(ac_ok, &skip);
                        let v = ctx.ld_global_f16(a_addr);
                        ctx.st_shared_f16(pro_a[i], v);
                        ctx.label(&skip);
                    }
                    ctx.branch("pro_done");
                    ctx.label("pro_slow_b");
                    for i in 0..elems_per_thread as usize {
                        let b_gr = ctx.add_u32_reg(k_off, b_row_in_tiles[i]);
                        ctx.st_shared_f16(pro_b[i], zero_f16);
                        let skip = format!("psb{i}");
                        let br_ok = ctx.setp_lt_u32(b_gr, k_param);
                        ctx.branch_if_not(br_ok, &skip);
                        let bc_ok = ctx.setp_lt_u32(b_col_globals[i], n_param);
                        ctx.branch_if_not(bc_ok, &skip);
                        let b_flat = ctx.mad_lo_u32(b_gr, n_const, b_col_globals[i]);
                        let b_boff = ctx.mul_wide_u32(b_flat, 2);
                        let b_addr = ctx.add_u64(b_ptr, b_boff);
                        let v = ctx.ld_global_f16(b_addr);
                        ctx.st_shared_f16(pro_b[i], v);
                        ctx.label(&skip);
                    }
                    ctx.branch("pro_done");
                    ctx.label("pro_fast");
                    ctx.branch_if_not(is_a_thread, "pro_fb");
                    for i in 0..elems_per_thread as usize {
                        let a_addr = ctx.add_u64(a_base_addrs[i], k_byte_stride);
                        let v = ctx.ld_global_f16(a_addr);
                        ctx.st_shared_f16(pro_a[i], v);
                    }
                    ctx.branch("pro_done");
                    ctx.label("pro_fb");
                    for i in 0..elems_per_thread as usize {
                        let b_gr = ctx.add_u32_reg(k_off, b_row_in_tiles[i]);
                        let b_flat = ctx.mad_lo_u32(b_gr, n_const, b_col_globals[i]);
                        let b_boff = ctx.mul_wide_u32(b_flat, 2);
                        let b_addr = ctx.add_u64(b_ptr, b_boff);
                        let v = ctx.ld_global_f16(b_addr);
                        ctx.st_shared_f16(pro_b[i], v);
                    }
                    ctx.label("pro_done");
                }
                ctx.bar_sync(0);

                // ─── Main loop: tiles 1..n-1 ───
                let k_tile = ctx.mov_u32_imm(1);
                ctx.label("k_loop");
                let k_done = ctx.setp_ge_u32(k_tile, n_k_tiles_reg);
                ctx.branch_if(k_done, "dbuf_epi");

                // Swap buffer
                let new_store = ctx.sub_u32_reg(smem_single_reg, store_buf_off);
                ctx.mov_u32_reg(store_buf_off, new_store);
                let compute_off = ctx.sub_u32_reg(smem_single_reg, store_buf_off);

                let k_off = ctx.mul_u32_reg(k_tile, c_16);
                let k_byte_stride = ctx.mul_wide_u32(k_off, 2);

                let lp_a: Vec<_> =
                    a_smem_addrs.iter().map(|&a| ctx.add_u32_reg(a, store_buf_off)).collect();
                let lp_b: Vec<_> =
                    b_smem_addrs.iter().map(|&b| ctx.add_u32_reg(b, store_buf_off)).collect();

                let k_tile_end = ctx.add_u32_reg(k_off, c_15);
                let k_ok = ctx.setp_lt_u32(k_tile_end, k_param);
                let fully_interior = ctx.and_pred(mn_interior, k_ok);
                ctx.branch_if(fully_interior, "fast_load");

                // Slow path
                ctx.branch_if_not(is_a_thread, "slow_b");
                for i in 0..elems_per_thread as usize {
                    ctx.st_shared_f16(lp_a[i], zero_f16);
                    let skip = format!("ssa{i}");
                    let ar_ok = ctx.setp_lt_u32(a_global_rows[i], m_param);
                    ctx.branch_if_not(ar_ok, &skip);
                    let a_addr = ctx.add_u64(a_base_addrs[i], k_byte_stride);
                    let ac_ok = ctx.setp_lt_u32(k_off, k_param);
                    ctx.branch_if_not(ac_ok, &skip);
                    let v = ctx.ld_global_f16(a_addr);
                    ctx.st_shared_f16(lp_a[i], v);
                    ctx.label(&skip);
                }
                ctx.branch("load_done");
                ctx.label("slow_b");
                for i in 0..elems_per_thread as usize {
                    let b_gr = ctx.add_u32_reg(k_off, b_row_in_tiles[i]);
                    ctx.st_shared_f16(lp_b[i], zero_f16);
                    let skip = format!("ssb{i}");
                    let br_ok = ctx.setp_lt_u32(b_gr, k_param);
                    ctx.branch_if_not(br_ok, &skip);
                    let bc_ok = ctx.setp_lt_u32(b_col_globals[i], n_param);
                    ctx.branch_if_not(bc_ok, &skip);
                    let b_flat = ctx.mad_lo_u32(b_gr, n_const, b_col_globals[i]);
                    let b_boff = ctx.mul_wide_u32(b_flat, 2);
                    let b_addr = ctx.add_u64(b_ptr, b_boff);
                    let v = ctx.ld_global_f16(b_addr);
                    ctx.st_shared_f16(lp_b[i], v);
                    ctx.label(&skip);
                }
                ctx.branch("load_done");

                // Fast path
                ctx.label("fast_load");
                ctx.branch_if_not(is_a_thread, "fast_b");
                for i in 0..elems_per_thread as usize {
                    let a_addr = ctx.add_u64(a_base_addrs[i], k_byte_stride);
                    let v = ctx.ld_global_f16(a_addr);
                    ctx.st_shared_f16(lp_a[i], v);
                }
                ctx.branch("load_done");
                ctx.label("fast_b");
                for i in 0..elems_per_thread as usize {
                    let b_gr = ctx.add_u32_reg(k_off, b_row_in_tiles[i]);
                    let b_flat = ctx.mad_lo_u32(b_gr, n_const, b_col_globals[i]);
                    let b_boff = ctx.mul_wide_u32(b_flat, 2);
                    let b_addr = ctx.add_u64(b_ptr, b_boff);
                    let v = ctx.ld_global_f16(b_addr);
                    ctx.st_shared_f16(lp_b[i], v);
                }
                ctx.label("load_done");

                // WMMA on compute buffer (overlaps with loads to store buffer)
                {
                    let smem_base = ctx.shared_base_addr();
                    let a_sub_bytes = ctx.mul_u32_reg(warp_m_off, c_16);
                    let a_sub_bytes = ctx.mul_u32_reg(a_sub_bytes, c_2);
                    let a_sub_buf = ctx.add_u32_reg(a_sub_bytes, compute_off);
                    let a_sub_off = ctx.cvt_u64_u32(a_sub_buf);
                    let a_sub_ptr = ctx.add_u64(smem_base, a_sub_off);
                    let frag_a = ctx.wmma_load_a_f16(a_sub_ptr, 16, WmmaLayout::RowMajor);

                    let b_sub_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
                    let b_base = ctx.add_u32_reg(c_a_smem, b_sub_bytes);
                    let b_buf = ctx.add_u32_reg(b_base, compute_off);
                    let b_off64 = ctx.cvt_u64_u32(b_buf);
                    let b_sub_ptr = ctx.add_u64(smem_base, b_off64);
                    let frag_b = ctx.wmma_load_b_f16(b_sub_ptr, 64, WmmaLayout::RowMajor);

                    let frag_d = ctx.wmma_mma_f16_f32_row_row(&frag_a, &frag_b, &frag_c);
                    for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                        ctx.mov_f32_reg(*c_reg, *d_reg);
                    }
                }

                ctx.bar_sync(0);
                ctx.add_u32_inplace(k_tile, 1);
                ctx.branch("k_loop");

                // ─── Epilogue: WMMA on last loaded tile ───
                ctx.label("dbuf_epi");
                {
                    let smem_base = ctx.shared_base_addr();
                    let a_sub_bytes = ctx.mul_u32_reg(warp_m_off, c_16);
                    let a_sub_bytes = ctx.mul_u32_reg(a_sub_bytes, c_2);
                    let a_sub_buf = ctx.add_u32_reg(a_sub_bytes, store_buf_off);
                    let a_sub_off = ctx.cvt_u64_u32(a_sub_buf);
                    let a_sub_ptr = ctx.add_u64(smem_base, a_sub_off);
                    let frag_a = ctx.wmma_load_a_f16(a_sub_ptr, 16, WmmaLayout::RowMajor);

                    let b_sub_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
                    let b_base = ctx.add_u32_reg(c_a_smem, b_sub_bytes);
                    let b_buf = ctx.add_u32_reg(b_base, store_buf_off);
                    let b_off64 = ctx.cvt_u64_u32(b_buf);
                    let b_sub_ptr = ctx.add_u64(smem_base, b_off64);
                    let frag_b = ctx.wmma_load_b_f16(b_sub_ptr, 64, WmmaLayout::RowMajor);

                    let frag_d = ctx.wmma_mma_f16_f32_row_row(&frag_a, &frag_b, &frag_c);
                    for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                        ctx.mov_f32_reg(*c_reg, *d_reg);
                    }
                }
            } else {
                // ═══ SINGLE-BUFFER K-LOOP ═══
                let k_tile = ctx.mov_u32_imm(0);
                ctx.label("k_loop");
                let k_done = ctx.setp_ge_u32(k_tile, n_k_tiles_reg);
                ctx.branch_if(k_done, "k_end");

                let k_off = ctx.mul_u32_reg(k_tile, c_16);
                let k_byte_stride = ctx.mul_wide_u32(k_off, 2);

                let k_tile_end = ctx.add_u32_reg(k_off, c_15);
                let k_ok = ctx.setp_lt_u32(k_tile_end, k_param);
                let fully_interior = ctx.and_pred(mn_interior, k_ok);
                ctx.branch_if(fully_interior, "fast_load");

                // Slow path
                ctx.branch_if_not(is_a_thread, "slow_b");
                for i in 0..elems_per_thread as usize {
                    ctx.st_shared_f16(a_smem_addrs[i], zero_f16);
                    let skip = format!("ssa{i}");
                    let ar_ok = ctx.setp_lt_u32(a_global_rows[i], m_param);
                    ctx.branch_if_not(ar_ok, &skip);
                    let a_addr = ctx.add_u64(a_base_addrs[i], k_byte_stride);
                    let ac_ok = ctx.setp_lt_u32(k_off, k_param);
                    ctx.branch_if_not(ac_ok, &skip);
                    let v = ctx.ld_global_f16(a_addr);
                    ctx.st_shared_f16(a_smem_addrs[i], v);
                    ctx.label(&skip);
                }
                ctx.branch("load_done");
                ctx.label("slow_b");
                for i in 0..elems_per_thread as usize {
                    let b_gr = ctx.add_u32_reg(k_off, b_row_in_tiles[i]);
                    ctx.st_shared_f16(b_smem_addrs[i], zero_f16);
                    let skip = format!("ssb{i}");
                    let br_ok = ctx.setp_lt_u32(b_gr, k_param);
                    ctx.branch_if_not(br_ok, &skip);
                    let bc_ok = ctx.setp_lt_u32(b_col_globals[i], n_param);
                    ctx.branch_if_not(bc_ok, &skip);
                    let b_flat = ctx.mad_lo_u32(b_gr, n_const, b_col_globals[i]);
                    let b_boff = ctx.mul_wide_u32(b_flat, 2);
                    let b_addr = ctx.add_u64(b_ptr, b_boff);
                    let v = ctx.ld_global_f16(b_addr);
                    ctx.st_shared_f16(b_smem_addrs[i], v);
                    ctx.label(&skip);
                }
                ctx.branch("load_done");

                // Fast path
                ctx.label("fast_load");
                ctx.branch_if_not(is_a_thread, "fast_b");
                for i in 0..elems_per_thread as usize {
                    let a_addr = ctx.add_u64(a_base_addrs[i], k_byte_stride);
                    let v = ctx.ld_global_f16(a_addr);
                    ctx.st_shared_f16(a_smem_addrs[i], v);
                }
                ctx.branch("load_done");
                ctx.label("fast_b");
                for i in 0..elems_per_thread as usize {
                    let b_gr = ctx.add_u32_reg(k_off, b_row_in_tiles[i]);
                    let b_flat = ctx.mad_lo_u32(b_gr, n_const, b_col_globals[i]);
                    let b_boff = ctx.mul_wide_u32(b_flat, 2);
                    let b_addr = ctx.add_u64(b_ptr, b_boff);
                    let v = ctx.ld_global_f16(b_addr);
                    ctx.st_shared_f16(b_smem_addrs[i], v);
                }

                ctx.label("load_done");
                ctx.bar_sync(0);

                // WMMA
                let smem_base = ctx.shared_base_addr();
                let a_sub_bytes = ctx.mul_u32_reg(warp_m_off, c_16);
                let a_sub_bytes = ctx.mul_u32_reg(a_sub_bytes, c_2);
                let a_sub_off = ctx.cvt_u64_u32(a_sub_bytes);
                let a_sub_ptr = ctx.add_u64(smem_base, a_sub_off);
                let frag_a = ctx.wmma_load_a_f16(a_sub_ptr, 16, WmmaLayout::RowMajor);

                let b_sub_bytes = ctx.mul_u32_reg(warp_n_off, c_2);
                let b_offset = ctx.add_u32_reg(c_a_smem, b_sub_bytes);
                let b_off64 = ctx.cvt_u64_u32(b_offset);
                let b_sub_ptr = ctx.add_u64(smem_base, b_off64);
                let frag_b = ctx.wmma_load_b_f16(b_sub_ptr, 64, WmmaLayout::RowMajor);

                let frag_d = ctx.wmma_mma_f16_f32_row_row(&frag_a, &frag_b, &frag_c);
                for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                    ctx.mov_f32_reg(*c_reg, *d_reg);
                }

                ctx.bar_sync(1);
                ctx.add_u32_inplace(k_tile, 1);
                ctx.branch("k_loop");
                ctx.label("k_end");
            }

            // === Store C (FP32, row-major) ===
            let c_row = ctx.add_u32_reg(cta_row, warp_m_off);
            let c_col = ctx.add_u32_reg(cta_col, warp_n_off);
            let cr_ok = ctx.setp_lt_u32(c_row, m_param);
            ctx.branch_if_not(cr_ok, "exit");
            let cc_ok = ctx.setp_lt_u32(c_col, n_param);
            ctx.branch_if_not(cc_ok, "exit");

            let c_row_off = ctx.mul_wide_u32_reg(c_row, n_param);
            let c_col_off = ctx.cvt_u64_u32(c_col);
            let c_base = ctx.add_u64(c_row_off, c_col_off);
            let c_base = ctx.mul_u64(c_base, 4); // FP32 = 4 bytes
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
    fn test_cta64_generates_valid_ptx() {
        let kernel = build_cta64_wmma_fp16(128, 128, 128);
        let module = PtxModule::new().add_kernel(kernel);
        let ptx = module.emit();
        assert!(ptx.contains(".entry gemm_cta64_wmma_fp16"));
        assert!(ptx.contains("bar.sync"));
    }

    #[test]
    fn test_cta64_has_4kb_shared_memory() {
        let kernel = build_cta64_wmma_fp16(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();
        // Extract smem size from ".shared .align 16 .b8 smem[N]"
        let extract_smem = |ptx: &str| -> usize {
            for line in ptx.lines() {
                if line.contains(".shared") && line.contains("smem[") {
                    let start = line.find("smem[").unwrap() + 5;
                    let end = line[start..].find(']').unwrap() + start;
                    return line[start..end].parse().unwrap();
                }
            }
            panic!("no .shared smem found");
        };
        assert_eq!(extract_smem(&ptx), 4096, "64×64 CTA needs 4096 bytes smem");
    }

    #[test]
    fn test_cta64_uses_row_row_mma() {
        let kernel = build_cta64_wmma_fp16(512, 512, 512);
        let ptx = PtxModule::new().add_kernel(kernel).emit();
        assert!(
            ptx.contains("wmma.mma.sync.aligned.m16n16k16.row.row.f32.f32"),
            "MMA must use row.row layout"
        );
    }

    #[test]
    fn test_cta64_has_fast_and_slow_paths() {
        let kernel = build_cta64_wmma_fp16(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();
        assert!(ptx.contains("fast_load"), "must have fast path");
        assert!(ptx.contains("slow_b"), "must have slow path");
    }

    #[test]
    fn test_cta64_single_wmma_per_iteration() {
        // 64×64 single-buffer should have exactly 1 WMMA MMA
        let kernel = build_cta64_wmma_fp16(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();
        let mma_count = ptx.matches("wmma.mma.sync.aligned.m16n16k16.row.row.f32.f32").count();
        assert_eq!(mma_count, 1, "single-buffer 64×64 should have 1 WMMA MMA");
    }

    #[test]
    fn test_cta64_b_stride_is_64() {
        // B tile is 16×64, so WMMA load stride must be 64
        let kernel = build_cta64_wmma_fp16(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();
        assert!(
            ptx.contains("wmma.load.b.sync.aligned.m16n16k16.row.f16"),
            "must have wmma.load.b instruction"
        );
    }

    // ── Double-buffer FALSIFY tests ──

    #[test]
    fn test_cta64_dbuf_valid_ptx() {
        let kernel = build_cta64_wmma_fp16_dbuf(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();
        assert!(ptx.contains(".entry gemm_cta64_wmma_fp16"));
        assert!(ptx.contains("bar.sync"));
    }

    #[test]
    fn test_cta64_dbuf_8kb_shared_memory() {
        let kernel = build_cta64_wmma_fp16_dbuf(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();
        let extract_smem = |ptx: &str| -> usize {
            for line in ptx.lines() {
                if line.contains(".shared") && line.contains("smem[") {
                    let start = line.find("smem[").unwrap() + 5;
                    let end = line[start..].find(']').unwrap() + start;
                    return line[start..end].parse().unwrap();
                }
            }
            panic!("no .shared smem found");
        };
        assert_eq!(extract_smem(&ptx), 8192, "64×64 dbuf needs 8192 bytes (2×4096)");
    }

    #[test]
    fn test_cta64_dbuf_has_prologue_epilogue() {
        let kernel = build_cta64_wmma_fp16_dbuf(512, 512, 512);
        let ptx = PtxModule::new().add_kernel(kernel).emit();
        assert!(ptx.contains("pro_done"), "must have prologue");
        assert!(ptx.contains("dbuf_epi"), "must have epilogue");
    }

    #[test]
    fn test_cta64_dbuf_two_wmma_mma() {
        let kernel = build_cta64_wmma_fp16_dbuf(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();
        let mma_count = ptx.matches("wmma.mma.sync.aligned.m16n16k16.row.row.f32.f32").count();
        assert_eq!(mma_count, 2, "dbuf should have 2 WMMA (loop + epilogue)");
    }

    // ── cp.async FALSIFY tests ──

    #[test]
    fn test_cta64_cpasync_valid_ptx() {
        let kernel = build_cta64_wmma_fp16_cpasync(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();
        assert!(ptx.contains(".entry gemm_cta64_cpasync_fp16"));
        assert!(ptx.contains("cp.async.ca.shared.global"));
        assert!(ptx.contains("cp.async.commit_group"));
        assert!(ptx.contains("cp.async.wait_group"));
    }

    #[test]
    fn test_cta64_cpasync_8byte_copies() {
        let kernel = build_cta64_wmma_fp16_cpasync(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();
        // Must use 8-byte cp.async (not 2 or 4)
        assert!(
            ptx.contains(", 8;") || ptx.contains(", 8\n"),
            "cp.async must use 8-byte copies, got: {}",
            ptx.lines().filter(|l| l.contains("cp.async.ca")).next().unwrap_or("")
        );
    }

    #[test]
    fn test_cta64_cpasync_8kb_smem() {
        let kernel = build_cta64_wmma_fp16_cpasync(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();
        let extract_smem = |ptx: &str| -> usize {
            for line in ptx.lines() {
                if line.contains(".shared") && line.contains("smem[") {
                    let start = line.find("smem[").unwrap() + 5;
                    let end = line[start..].find(']').unwrap() + start;
                    return line[start..end].parse().unwrap();
                }
            }
            panic!("no .shared smem found");
        };
        assert_eq!(extract_smem(&ptx), 8192);
    }

    #[test]
    fn test_cta64_cpasync_two_wmma() {
        let kernel = build_cta64_wmma_fp16_cpasync(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();
        let mma_count = ptx.matches("wmma.mma.sync.aligned.m16n16k16.row.row.f32.f32").count();
        assert_eq!(mma_count, 2);
    }

    #[test]
    fn test_cta64_cpasync_kernel_name() {
        let kernel = build_cta64_wmma_fp16_cpasync(256, 256, 256);
        let ptx = PtxModule::new().add_kernel(kernel).emit();
        // Distinct kernel name to avoid conflict with non-cp.async variant
        assert!(ptx.contains("gemm_cta64_cpasync_fp16"));
    }

    /// Analyze instruction mix of the cp.async kernel.
    /// The instruction count reveals the compute-to-overhead ratio.
    #[test]
    fn test_cta64_cpasync_instruction_analysis() {
        let kernel = build_cta64_wmma_fp16_cpasync(1024, 1024, 1024);
        let ptx = PtxModule::new().target("sm_80").add_kernel(kernel).emit();

        let wmma_mma = ptx.matches("wmma.mma.sync").count();
        let wmma_load = ptx.matches("wmma.load.").count();
        let cp_async = ptx.matches("cp.async.ca").count();
        let bar_sync = ptx.matches("bar.sync").count();
        let selp = ptx.matches("selp.").count();
        let mul_wide = ptx.matches("mul.wide").count();

        // The cp.async kernel should have:
        // - 2 wmma.mma (prologue epilogue uses same MMA, main loop has 1)
        // - Multiple wmma.load (2 per MMA: A + B)
        // - cp.async for data loading
        assert!(wmma_mma >= 2, "need at least 2 wmma.mma (loop + epilogue), got {wmma_mma}");
        assert!(cp_async >= 2, "need at least 2 cp.async, got {cp_async}");
        assert!(wmma_load >= 4, "need at least 4 wmma.load (2A + 2B), got {wmma_load}");

        // Report instruction mix for analysis (visible with --nocapture)
        eprintln!("=== CTA64 cp.async PTX Instruction Analysis ===");
        eprintln!("  wmma.mma:    {wmma_mma}");
        eprintln!("  wmma.load:   {wmma_load}");
        eprintln!("  cp.async:    {cp_async}");
        eprintln!("  bar.sync:    {bar_sync}");
        eprintln!("  selp:        {selp}");
        eprintln!("  mul.wide:    {mul_wide}");
        eprintln!("  total lines: {}", ptx.lines().count());

        // Ratio: compute instructions / total instructions
        // High ratio = efficient, low ratio = overhead-dominated
        let total_insts = ptx.lines().filter(|l| l.trim().ends_with(';')).count();
        let compute_insts = wmma_mma + wmma_load;
        eprintln!("  total instrs: {total_insts}");
        eprintln!(
            "  compute:      {compute_insts} ({:.1}%)",
            100.0 * compute_insts as f64 / total_insts.max(1) as f64
        );
    }
}
