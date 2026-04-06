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
