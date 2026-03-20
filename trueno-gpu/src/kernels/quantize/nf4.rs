//! NF4 Dequantization-Fused GEMM Kernel (trueno#108).
//!
//! Implements fused dequantization with matrix multiplication for NF4 (4-bit NormalFloat)
//! quantized weights, enabling QLoRA training with 8x memory compression.
//!
//! # NF4 Block Layout
//!
//! Scales and packed data are stored in separate GPU buffers (SoA layout) for coalescing:
//! - `b_scales`: `[f32; num_blocks]` — one scale per 64-value block, column-major block order
//! - `b_nf4`:    `[u8; num_blocks * 32]` — packed 4-bit indices (2 per byte), same order
//!
//! # Dequantization
//!
//! ```text
//! val = scale × NF4_LUT[nibble]
//! ```
//!
//! Where `NF4_LUT` is a fixed 16-entry codebook from normal distribution quantiles.
//!
//! # Contract: C-NF4-003 (GEMM Numerical Parity)
//!
//! `nf4_gemm(A, Q) ≈ naive_gemm(A, dequantize(Q))` within 1e-3 per-element.

#![allow(clippy::similar_names)]

use super::nf4_cpu::{NF4_BLOCK_SIZE, NF4_LUT};
use super::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxReg, PtxType};

/// NF4 block size as u32 for PTX constants.
const NF4_BLOCK_SIZE_U32: u32 = NF4_BLOCK_SIZE as u32;

/// NF4 quantized GEMM kernel configuration.
///
/// Computes `C[M×N] = A[M×K] @ dequant(B_nf4[K×N])` where B is stored in NF4 format.
/// The kernel fuses dequantization with matmul to avoid materializing fp32 weights.
///
/// # Memory Layout (separate scale/data buffers)
///
/// - `A`: row-major f32 `[M × K]`
/// - `b_nf4`: packed nibbles `[N * (K/64) * 32]` bytes, column-major block order
/// - `b_scales`: `[N * (K/64)]` f32 values, column-major block order
/// - `C`: row-major f32 `[M × N]`
#[derive(Debug, Clone)]
pub struct Nf4GemmKernel {
    /// Output rows (M)
    pub m: u32,
    /// Output columns (N)
    pub n: u32,
    /// Inner dimension (K) — must be divisible by 64
    pub k: u32,
    /// Tile size for output (default: 32)
    pub tile_size: u32,
}

impl Nf4GemmKernel {
    /// Create a new NF4 quantized GEMM kernel.
    ///
    /// # Contract: C-NF4-002
    ///
    /// `k` must be divisible by 64 (NF4 block size).
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k, tile_size: 32 }
    }

    /// Set output tile size.
    #[must_use]
    pub const fn with_tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Get number of NF4 blocks per weight column (K / 64).
    #[must_use]
    pub const fn num_blocks_per_col(&self) -> u32 {
        self.k / NF4_BLOCK_SIZE_U32
    }
}

/// Tile dimensions for the shared-memory tiled NF4 GEMM kernel.
///
/// TILE_M × TILE_N = 32 × 32 = 1024 threads per block.
/// TILE_K = 64 matches the NF4 block size, so each K-iteration loads exactly
/// one NF4 block per column.
const TILE_M: u32 = 32;
const TILE_N: u32 = 32;
const TILE_K: u32 = NF4_BLOCK_SIZE_U32; // 64

/// Shared memory layout offsets (in bytes):
/// - LUT:  [0, 64)       — 16 × f32 = 64 bytes (NF4 codebook)
/// - s_A:  [64, 8256)    — TILE_M × TILE_K × 4 = 32 × 64 × 4 = 8192 bytes
/// - s_B:  [8256, 16448) — TILE_K × TILE_N × 4 = 64 × 32 × 4 = 8192 bytes
/// Total: 16448 bytes (~16 KB, well within 48 KB limit)
const SMEM_LUT_OFFSET: u32 = 0;
const SMEM_LUT_BYTES: u32 = 16 * 4; // 64
const SMEM_A_OFFSET: u32 = SMEM_LUT_BYTES; // 64
const SMEM_A_BYTES: u32 = TILE_M * TILE_K * 4; // 8192
const SMEM_B_OFFSET: u32 = SMEM_A_OFFSET + SMEM_A_BYTES; // 8256
const SMEM_B_BYTES: u32 = TILE_K * TILE_N * 4; // 8192
const SMEM_TOTAL: usize = (SMEM_B_OFFSET + SMEM_B_BYTES) as usize; // 16448

impl Kernel for Nf4GemmKernel {
    fn name(&self) -> &str {
        "nf4_gemm_fused"
    }

    /// Build a shared-memory tiled NF4 GEMM kernel.
    ///
    /// # Algorithm
    ///
    /// For each K-block (stride TILE_K=64):
    ///   1. ALL threads cooperatively load an A tile [TILE_M × TILE_K] from global → shared
    ///   2. ALL threads cooperatively load a B_NF4 tile, dequantize on-the-fly → shared as f32
    ///      [TILE_K × TILE_N]
    ///   3. `bar.sync 0` — ensure tiles are fully loaded
    ///   4. Each thread accumulates: for k in 0..TILE_K: acc += s_A[row][k] * s_B[k][col]
    ///   5. `bar.sync 0` — ensure all threads done before next tile overwrites shared memory
    /// Write C[row,col] = acc
    ///
    /// # Performance (vs. original serial kernel)
    ///
    /// - **Global memory bandwidth**: A tile loaded once, shared by 32 columns.
    ///   B tile loaded once, shared by 32 rows. ~32x reduction in redundant loads.
    /// - **Shared memory throughput**: ~2 TB/s vs. ~900 GB/s for global memory.
    /// - **Latency hiding**: Cooperative loading uses all 1024 threads; inner loop
    ///   has short-latency shared memory reads.
    fn build_ptx(&self) -> PtxKernel {
        let tile_size = self.tile_size;

        PtxKernel::new("nf4_gemm_fused")
            .param(PtxType::U64, "a_ptr") // Activations [M × K], f32
            .param(PtxType::U64, "b_nf4_ptr") // Packed nibbles
            .param(PtxType::U64, "b_scales_ptr") // Per-block scales, f32
            .param(PtxType::U64, "c_ptr") // Output [M × N], f32
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(SMEM_TOTAL)
            .build(|ctx| {
                // Thread and block indices
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

                // Load parameters
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_nf4_ptr = ctx.load_param_u64("b_nf4_ptr");
                let b_scales_ptr = ctx.load_param_u64("b_scales_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // =========================================================
                // Phase 0: Load NF4 codebook LUT into shared memory
                // =========================================================
                // First 16 threads each store one LUT entry at s_lut[tid]
                for (i, &val) in NF4_LUT.iter().enumerate() {
                    let imm_i = ctx.mov_u32_imm(i as u32);
                    let is_i = ctx.setp_eq_u32(tid, imm_i);
                    ctx.branch_if_not(is_i, &format!("skip_lut_{i}"));

                    let val_reg = ctx.mov_f32_imm(val);
                    let lut_smem_off = ctx.mov_u32_imm(SMEM_LUT_OFFSET + (i as u32) * 4);
                    ctx.st_shared_f32(lut_smem_off, val_reg);

                    ctx.label(&format!("skip_lut_{i}"));
                }

                ctx.bar_sync(0);

                // =========================================================
                // Calculate output position
                // =========================================================
                // Grid: (ceil(N/TILE_N), ceil(M/TILE_M)), block: (1024,1,1)
                // Each thread computes C[global_row, global_col]
                let tile_size_reg = ctx.mov_u32_imm(tile_size);
                let out_row = ctx.mul_u32_reg(ctaid_y, tile_size_reg);
                let out_col = ctx.mul_u32_reg(ctaid_x, tile_size_reg);

                // Thread (local_row, local_col) within the 32×32 tile
                let local_row = ctx.div_u32(tid, tile_size);
                let local_col = ctx.rem_u32(tid, tile_size);

                let global_row = ctx.add_u32_reg(out_row, local_row);
                let global_col = ctx.add_u32_reg(out_col, local_col);

                // Bounds check predicates (store only for valid threads)
                let row_oob = ctx.setp_ge_u32(global_row, m_param);
                let col_oob = ctx.setp_ge_u32(global_col, n_param);

                // Clamp bounds (used by cooperative tile loading)
                let one = ctx.mov_u32_imm(1);
                let m_minus_1 = ctx.sub_u32_reg(m_param, one);
                let n_minus_1 = ctx.sub_u32_reg(n_param, one);
                let k_minus_1 = ctx.sub_u32_reg(k_param, one);

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Number of NF4 blocks along K (K / 64)
                let num_k_blocks = ctx.div_u32(k_param, NF4_BLOCK_SIZE_U32);

                // =========================================================
                // Outer loop: iterate over K in tiles of TILE_K=64
                // =========================================================
                let block_idx = ctx.mov_u32_imm(0);

                ctx.label("tile_loop");
                let tile_done = ctx.setp_ge_u32(block_idx, num_k_blocks);
                ctx.branch_if(tile_done, "tile_loop_done");

                // Current K offset for this tile
                let tile_k_imm = ctx.mov_u32_imm(TILE_K);
                let k_base = ctx.mul_u32_reg(block_idx, tile_k_imm);

                // =====================================================
                // Phase 1: Cooperative load of A tile into shared memory
                // =====================================================
                // s_A[TILE_M][TILE_K] = A[out_row..out_row+TILE_M, k_base..k_base+TILE_K]
                // 1024 threads load 32×64 = 2048 elements → 2 loads per thread
                //
                // Thread tid loads elements at positions tid and tid+1024.
                // For element i: row = i / TILE_K, col = i % TILE_K
                // Global: A[out_row + row, k_base + col]
                let tile_m_times_k = ctx.mov_u32_imm(TILE_M * TILE_K); // 2048
                let block_threads = ctx.mov_u32_imm(TILE_M * TILE_N); // 1024

                // Load pass 0: element at tid
                {
                    let a_tile_row = ctx.div_u32(tid, TILE_K);
                    let a_tile_col = ctx.rem_u32(tid, TILE_K);

                    // Global row/col for A
                    let a_g_row = ctx.add_u32_reg(out_row, a_tile_row);
                    let a_g_col = ctx.add_u32_reg(k_base, a_tile_col);

                    // Clamp for safety
                    let a_g_row_c = ctx.min_u32(a_g_row, m_minus_1);
                    let a_g_col_c = ctx.min_u32(a_g_col, k_minus_1);

                    // Load from global: A[a_g_row_c, a_g_col_c]
                    let a_row_off = ctx.mul_wide_u32_reg(a_g_row_c, k_param);
                    let a_col_off = ctx.cvt_u64_u32(a_g_col_c);
                    let a_elem_off = ctx.add_u64(a_row_off, a_col_off);
                    let a_byte_off = ctx.mul_u64(a_elem_off, 4);
                    let a_addr = ctx.add_u64(a_ptr, a_byte_off);
                    let a_val = ctx.ld_global_f32(a_addr);

                    // Store to shared: s_A[tid] (tid is the linear index into the tile)
                    let sa_byte = ctx.mul_u32(tid, 4);
                    let sa_smem = ctx.add_u32(sa_byte, SMEM_A_OFFSET);
                    ctx.st_shared_f32(sa_smem, a_val);
                }

                // Load pass 1: element at tid + 1024 (if < 2048)
                {
                    let pass1_idx = ctx.add_u32_reg(tid, block_threads);
                    let pass1_valid = ctx.setp_lt_u32(pass1_idx, tile_m_times_k);
                    ctx.branch_if_not(pass1_valid, "skip_a_pass1");

                    let a_tile_row = ctx.div_u32(pass1_idx, TILE_K);
                    let a_tile_col = ctx.rem_u32(pass1_idx, TILE_K);

                    let a_g_row = ctx.add_u32_reg(out_row, a_tile_row);
                    let a_g_col = ctx.add_u32_reg(k_base, a_tile_col);

                    let a_g_row_c = ctx.min_u32(a_g_row, m_minus_1);
                    let a_g_col_c = ctx.min_u32(a_g_col, k_minus_1);

                    let a_row_off = ctx.mul_wide_u32_reg(a_g_row_c, k_param);
                    let a_col_off = ctx.cvt_u64_u32(a_g_col_c);
                    let a_elem_off = ctx.add_u64(a_row_off, a_col_off);
                    let a_byte_off = ctx.mul_u64(a_elem_off, 4);
                    let a_addr = ctx.add_u64(a_ptr, a_byte_off);
                    let a_val = ctx.ld_global_f32(a_addr);

                    let sa_byte = ctx.mul_u32(pass1_idx, 4);
                    let sa_smem = ctx.add_u32(sa_byte, SMEM_A_OFFSET);
                    ctx.st_shared_f32(sa_smem, a_val);

                    ctx.label("skip_a_pass1");
                }

                // =====================================================
                // Phase 2: Cooperative load of B tile with on-the-fly
                //           NF4 dequantization into shared memory
                // =====================================================
                // s_B[TILE_K][TILE_N] = dequant(B_nf4) for columns
                //   out_col..out_col+TILE_N, block block_idx
                // TILE_K × TILE_N = 64 × 32 = 2048 elements → 2 loads per thread
                //
                // For element i: k_in_block = i / TILE_N, col_in_tile = i % TILE_N
                // B column = out_col + col_in_tile
                // NF4 block index = col * num_k_blocks + block_idx
                let tile_k_times_n = ctx.mov_u32_imm(TILE_K * TILE_N); // 2048

                // Load pass 0: element at tid
                {
                    let b_k_local = ctx.div_u32(tid, TILE_N);
                    let b_col_local = ctx.rem_u32(tid, TILE_N);

                    let b_g_col = ctx.add_u32_reg(out_col, b_col_local);
                    let b_g_col_c = ctx.min_u32(b_g_col, n_minus_1);

                    // NF4 block index: col * num_k_blocks + block_idx
                    let col_block_off = ctx.mul_u32_reg(b_g_col_c, num_k_blocks);
                    let scale_idx = ctx.add_u32_reg(col_block_off, block_idx);

                    // Load scale
                    let scale_byte_off = ctx.mul_wide_u32(scale_idx, 4);
                    let scale_addr = ctx.add_u64(b_scales_ptr, scale_byte_off);
                    let scale = ctx.ld_global_f32(scale_addr);

                    // Load packed NF4 data
                    let data_block_off = ctx.mul_wide_u32(scale_idx, 32);
                    let data_block_addr = ctx.add_u64(b_nf4_ptr, data_block_off);

                    // NF4 byte = b_k_local / 2, nibble = b_k_local % 2
                    let byte_in_blk = ctx.div_u32(b_k_local, 2);
                    let nibble_sel = ctx.rem_u32(b_k_local, 2);

                    let byte_off_64 = ctx.cvt_u64_u32(byte_in_blk);
                    let nib_addr = ctx.add_u64(data_block_addr, byte_off_64);
                    let packed = ctx.ld_global_u8(nib_addr);
                    let packed_u32 = ctx.cvt_u32_u8(packed);

                    let four = ctx.mov_u32_imm(4);
                    let shift = ctx.mul_u32_reg(nibble_sel, four);
                    let shifted = ctx.shr_u32(packed_u32, shift);
                    let mask = ctx.mov_u32_imm(0xF);
                    let nf4_idx = ctx.and_u32(shifted, mask);

                    // LUT lookup from shared memory
                    let lut_off = ctx.mul_u32(nf4_idx, 4);
                    let lut_smem = ctx.add_u32(lut_off, SMEM_LUT_OFFSET);
                    let codebook_val = ctx.ld_shared_f32(lut_smem);

                    // Dequantize: val = scale * codebook_val
                    let dequant = ctx.mul_f32(scale, codebook_val);

                    // Store to s_B[b_k_local * TILE_N + b_col_local]
                    let sb_byte = ctx.mul_u32(tid, 4);
                    let sb_smem = ctx.add_u32(sb_byte, SMEM_B_OFFSET);
                    ctx.st_shared_f32(sb_smem, dequant);
                }

                // Load pass 1: element at tid + 1024 (if < 2048)
                {
                    let pass1_idx = ctx.add_u32_reg(tid, block_threads);
                    let pass1_valid = ctx.setp_lt_u32(pass1_idx, tile_k_times_n);
                    ctx.branch_if_not(pass1_valid, "skip_b_pass1");

                    let b_k_local = ctx.div_u32(pass1_idx, TILE_N);
                    let b_col_local = ctx.rem_u32(pass1_idx, TILE_N);

                    let b_g_col = ctx.add_u32_reg(out_col, b_col_local);
                    let b_g_col_c = ctx.min_u32(b_g_col, n_minus_1);

                    let col_block_off = ctx.mul_u32_reg(b_g_col_c, num_k_blocks);
                    let scale_idx = ctx.add_u32_reg(col_block_off, block_idx);

                    let scale_byte_off = ctx.mul_wide_u32(scale_idx, 4);
                    let scale_addr = ctx.add_u64(b_scales_ptr, scale_byte_off);
                    let scale = ctx.ld_global_f32(scale_addr);

                    let data_block_off = ctx.mul_wide_u32(scale_idx, 32);
                    let data_block_addr = ctx.add_u64(b_nf4_ptr, data_block_off);

                    let byte_in_blk = ctx.div_u32(b_k_local, 2);
                    let nibble_sel = ctx.rem_u32(b_k_local, 2);

                    let byte_off_64 = ctx.cvt_u64_u32(byte_in_blk);
                    let nib_addr = ctx.add_u64(data_block_addr, byte_off_64);
                    let packed = ctx.ld_global_u8(nib_addr);
                    let packed_u32 = ctx.cvt_u32_u8(packed);

                    let four = ctx.mov_u32_imm(4);
                    let shift = ctx.mul_u32_reg(nibble_sel, four);
                    let shifted = ctx.shr_u32(packed_u32, shift);
                    let mask = ctx.mov_u32_imm(0xF);
                    let nf4_idx = ctx.and_u32(shifted, mask);

                    let lut_off = ctx.mul_u32(nf4_idx, 4);
                    let lut_smem = ctx.add_u32(lut_off, SMEM_LUT_OFFSET);
                    let codebook_val = ctx.ld_shared_f32(lut_smem);

                    let dequant = ctx.mul_f32(scale, codebook_val);

                    let sb_byte = ctx.mul_u32(pass1_idx, 4);
                    let sb_smem = ctx.add_u32(sb_byte, SMEM_B_OFFSET);
                    ctx.st_shared_f32(sb_smem, dequant);

                    ctx.label("skip_b_pass1");
                }

                // =====================================================
                // Phase 3: Barrier — tiles are loaded
                // =====================================================
                ctx.bar_sync(0);

                // =====================================================
                // Phase 4: Compute — each thread accumulates its dot product
                //   from shared memory tiles
                // =====================================================
                // acc += sum_{k=0..TILE_K-1} s_A[local_row][k] * s_B[k][local_col]
                //
                // s_A layout: row-major [TILE_M][TILE_K] at SMEM_A_OFFSET
                //   s_A[r][k] at byte offset = SMEM_A_OFFSET + (r * TILE_K + k) * 4
                //
                // s_B layout: row-major [TILE_K][TILE_N] at SMEM_B_OFFSET
                //   s_B[k][c] at byte offset = SMEM_B_OFFSET + (k * TILE_N + c) * 4
                let four = ctx.mov_u32_imm(4);
                let tile_k_reg = ctx.mov_u32_imm(TILE_K);
                let tile_n_reg = ctx.mov_u32_imm(TILE_N);

                // Precompute row base for s_A: SMEM_A_OFFSET + local_row * TILE_K * 4
                let sa_row_elems = ctx.mul_u32_reg(local_row, tile_k_reg);
                let sa_row_bytes = ctx.mul_u32_reg(sa_row_elems, four);
                let sa_row_base = ctx.add_u32(sa_row_bytes, SMEM_A_OFFSET);

                // Precompute col offset for s_B: local_col * 4
                let sb_col_bytes = ctx.mul_u32_reg(local_col, four);

                // Inner loop: k = 0..TILE_K
                let k_idx = ctx.mov_u32_imm(0);

                ctx.label("k_loop");
                let k_done = ctx.setp_ge_u32(k_idx, tile_k_reg);
                ctx.branch_if(k_done, "k_loop_done");

                // s_A[local_row][k_idx]: offset = sa_row_base + k_idx * 4
                let sa_k_bytes = ctx.mul_u32_reg(k_idx, four);
                let sa_addr = ctx.add_u32_reg(sa_row_base, sa_k_bytes);
                let a_val = ctx.ld_shared_f32(sa_addr);

                // s_B[k_idx][local_col]: offset = SMEM_B_OFFSET + (k_idx * TILE_N + local_col) * 4
                let sb_k_row = ctx.mul_u32_reg(k_idx, tile_n_reg);
                let sb_k_row_bytes = ctx.mul_u32_reg(sb_k_row, four);
                let sb_base = ctx.add_u32(sb_k_row_bytes, SMEM_B_OFFSET);
                let sb_addr = ctx.add_u32_reg(sb_base, sb_col_bytes);
                let b_val = ctx.ld_shared_f32(sb_addr);

                // acc += a_val * b_val
                ctx.fma_f32_inplace(acc, a_val, b_val);

                ctx.add_u32_inplace(k_idx, 1);
                ctx.branch("k_loop");

                ctx.label("k_loop_done");

                // =====================================================
                // Phase 5: Barrier — wait for all threads before next tile
                // =====================================================
                ctx.bar_sync(0);

                ctx.add_u32_inplace(block_idx, 1);
                ctx.branch("tile_loop");

                ctx.label("tile_loop_done");

                // =========================================================
                // Store result (only for valid threads)
                // =========================================================
                ctx.branch_if(row_oob, "exit");
                ctx.branch_if(col_oob, "exit");

                let c_row_offset = ctx.mul_wide_u32_reg(global_row, n_param);
                let global_col_64 = ctx.cvt_u64_u32(global_col);
                let c_elem_offset = ctx.add_u64(c_row_offset, global_col_64);
                let c_elem_bytes = ctx.mul_u64(c_elem_offset, 4);
                let c_addr = ctx.add_u64(c_ptr, c_elem_bytes);

                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

/// NF4 transposed GEMM kernel for backward pass (QLoRA gradient propagation).
///
/// Computes `C[M×K] = A[M×N] @ dequant(B_nf4[K×N])^T` where B is stored in NF4 format.
/// Equivalently: `C[i,j] = sum_n A[i,n] * B[j,n]` — reduces over B's column dimension.
///
/// Used in backward pass to propagate gradients through frozen NF4 projections:
/// `grad_input = grad_output @ W^T` where W is the NF4-quantized weight.
///
/// # Memory Layout
///
/// - `A`: row-major f32 `[M × N]` (grad_output)
/// - `b_nf4`: packed nibbles, column-major block order (same as forward)
/// - `b_scales`: per-block scales, column-major block order (same as forward)
/// - `C`: row-major f32 `[M × K]` (grad_input)
#[derive(Debug, Clone)]
pub struct Nf4GemmTransposeKernel {
    /// Output rows (M) — same as grad_output rows
    pub m: u32,
    /// Reduction dimension (N) — columns of B / cols of grad_output
    pub n: u32,
    /// Output columns (K) — rows of B / input hidden size
    pub k: u32,
    /// Tile size for output (default: 16, smaller than forward due to irregular access)
    pub tile_size: u32,
}

impl Nf4GemmTransposeKernel {
    /// Create a new NF4 transposed GEMM kernel.
    #[must_use]
    pub fn new(m: u32, n: u32, k: u32) -> Self {
        Self { m, n, k, tile_size: 16 }
    }

    /// Number of NF4 blocks per column of B (K / 64).
    #[must_use]
    pub const fn num_blocks_per_col(&self) -> u32 {
        self.k / NF4_BLOCK_SIZE_U32
    }
}

impl Kernel for Nf4GemmTransposeKernel {
    fn name(&self) -> &str {
        "nf4_gemm_transpose"
    }

    fn build_ptx(&self) -> PtxKernel {
        let tile_size = self.tile_size;
        let smem_size = 16 * 4; // NF4 codebook LUT

        PtxKernel::new("nf4_gemm_transpose")
            .param(PtxType::U64, "a_ptr") // grad_output [M × N], f32
            .param(PtxType::U64, "b_nf4_ptr") // NF4 weight data
            .param(PtxType::U64, "b_scales_ptr") // NF4 weight scales
            .param(PtxType::U64, "c_ptr") // grad_input [M × K], f32
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size)
            .build(|ctx| {
                let tid = ctx.special_reg(PtxReg::TidX);
                let ctaid_x = ctx.special_reg(PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(PtxReg::CtaIdY);

                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_nf4_ptr = ctx.load_param_u64("b_nf4_ptr");
                let b_scales_ptr = ctx.load_param_u64("b_scales_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Load NF4 codebook into shared memory
                let smem_base = ctx.shared_base_addr();
                for (i, &val) in NF4_LUT.iter().enumerate() {
                    let imm_i = ctx.mov_u32_imm(i as u32);
                    let is_i = ctx.setp_eq_u32(tid, imm_i);
                    ctx.branch_if_not(is_i, &format!("skip_lut_{i}"));
                    let val_reg = ctx.mov_f32_imm(val);
                    let offset = ctx.mov_u64_imm((i * 4) as u64);
                    let addr = ctx.add_u64(smem_base, offset);
                    ctx.st_generic_f32(addr, val_reg);
                    ctx.label(&format!("skip_lut_{i}"));
                }
                ctx.bar_sync(0);

                // Output position: C[global_row, global_col] where col is in K-dimension
                let tile_size_reg = ctx.mov_u32_imm(tile_size);
                let out_row = ctx.mul_u32_reg(ctaid_y, tile_size_reg);
                let out_col = ctx.mul_u32_reg(ctaid_x, tile_size_reg);

                let local_row = ctx.div_u32(tid, tile_size);
                let local_col = ctx.rem_u32(tid, tile_size);

                let global_row = ctx.add_u32_reg(out_row, local_row);
                let global_col = ctx.add_u32_reg(out_col, local_col);

                let row_oob = ctx.setp_ge_u32(global_row, m_param);
                let col_oob = ctx.setp_ge_u32(global_col, k_param);

                let one = ctx.mov_u32_imm(1);
                let m_minus_1 = ctx.sub_u32_reg(m_param, one);
                let k_minus_1 = ctx.sub_u32_reg(k_param, one);
                let clamped_row = ctx.min_u32(global_row, m_minus_1);
                let clamped_col = ctx.min_u32(global_col, k_minus_1); // col in K-dim

                let acc = ctx.mov_f32_imm(0.0);

                // Number of K-blocks per column of B
                let num_k_blocks = ctx.div_u32(k_param, NF4_BLOCK_SIZE_U32);

                // clamped_col is in K-dimension. Find which block and position within block.
                let col_block_idx = ctx.div_u32(clamped_col, NF4_BLOCK_SIZE_U32);
                let col_elem_in_block = ctx.rem_u32(clamped_col, NF4_BLOCK_SIZE_U32);

                // NF4 byte and nibble for this K-position (fixed for all N iterations)
                let byte_in_block = ctx.div_u32(col_elem_in_block, 2);
                let nibble_idx = ctx.rem_u32(col_elem_in_block, 2);
                let four = ctx.mov_u32_imm(4);
                let nibble_shift = ctx.mul_u32_reg(nibble_idx, four);
                let mask_4bit = ctx.mov_u32_imm(0xF);

                // A row base: A[clamped_row, :] starts at a_ptr + clamped_row * N * 4
                let a_row_offset = ctx.mul_wide_u32_reg(clamped_row, n_param);

                // Loop over N (reduction dimension — columns of B)
                let n_idx = ctx.mov_u32_imm(0);

                ctx.label("n_loop");
                let n_done = ctx.setp_ge_u32(n_idx, n_param);
                ctx.branch_if(n_done, "n_loop_done");

                // B[clamped_col, n_idx]:
                // Block index = n_idx * num_k_blocks + col_block_idx
                let n_block_base = ctx.mul_u32_reg(n_idx, num_k_blocks);
                let block_idx = ctx.add_u32_reg(n_block_base, col_block_idx);

                // Load scale
                let scale_byte_off = ctx.mul_wide_u32(block_idx, 4);
                let scale_addr = ctx.add_u64(b_scales_ptr, scale_byte_off);
                let scale = ctx.ld_global_f32(scale_addr);

                // Load packed byte from NF4 data
                let data_block_off = ctx.mul_wide_u32(block_idx, 32);
                let data_block_addr = ctx.add_u64(b_nf4_ptr, data_block_off);
                let byte_off_64 = ctx.cvt_u64_u32(byte_in_block);
                let nibble_addr = ctx.add_u64(data_block_addr, byte_off_64);
                let packed_byte = ctx.ld_global_u8(nibble_addr);
                let packed_u32 = ctx.cvt_u32_u8(packed_byte);

                // Extract nibble
                let shifted = ctx.shr_u32(packed_u32, nibble_shift);
                let nf4_idx = ctx.and_u32(shifted, mask_4bit);

                // Codebook lookup
                let nf4_idx_64 = ctx.cvt_u64_u32(nf4_idx);
                let lut_byte_off = ctx.mul_u64(nf4_idx_64, 4);
                let lut_addr = ctx.add_u64(smem_base, lut_byte_off);
                let normalized_val = ctx.ld_generic_f32(lut_addr);

                // Dequantize
                let dequant = ctx.mul_f32(scale, normalized_val);

                // Load A[clamped_row, n_idx]
                let n_idx_64 = ctx.cvt_u64_u32(n_idx);
                let a_elem_off = ctx.add_u64(a_row_offset, n_idx_64);
                let a_elem_bytes = ctx.mul_u64(a_elem_off, 4);
                let a_addr = ctx.add_u64(a_ptr, a_elem_bytes);
                let a_val = ctx.ld_global_f32(a_addr);

                // acc += a_val * dequant
                ctx.fma_f32_inplace(acc, a_val, dequant);

                ctx.add_u32_inplace(n_idx, 1);
                ctx.branch("n_loop");

                ctx.label("n_loop_done");

                // Store result
                ctx.branch_if(row_oob, "exit");
                ctx.branch_if(col_oob, "exit");

                let c_row_off = ctx.mul_wide_u32_reg(global_row, k_param);
                let global_col_64 = ctx.cvt_u64_u32(global_col);
                let c_elem_off = ctx.add_u64(c_row_off, global_col_64);
                let c_elem_bytes = ctx.mul_u64(c_elem_off, 4);
                let c_addr = ctx.add_u64(c_ptr, c_elem_bytes);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_nf4_gemm_kernel_name() {
        let kernel = Nf4GemmKernel::new(128, 896, 896);
        assert_eq!(kernel.name(), "nf4_gemm_fused");
    }

    #[test]
    fn test_nf4_gemm_num_blocks_per_col() {
        let kernel = Nf4GemmKernel::new(128, 896, 896);
        assert_eq!(kernel.num_blocks_per_col(), 896 / 64);
    }

    #[test]
    fn test_nf4_gemm_ptx_emits() {
        let kernel = Nf4GemmKernel::new(128, 896, 896);
        let ptx = kernel.emit_ptx();

        // Verify kernel name appears
        assert!(ptx.contains("nf4_gemm_fused"), "PTX missing kernel name");

        // Verify parameters declared
        assert!(ptx.contains("a_ptr"), "PTX missing a_ptr param");
        assert!(ptx.contains("b_nf4_ptr"), "PTX missing b_nf4_ptr param");
        assert!(ptx.contains("b_scales_ptr"), "PTX missing b_scales_ptr param");
        assert!(ptx.contains("c_ptr"), "PTX missing c_ptr param");

        // Verify shared memory usage (LUT + A tile + B tile = 16448 bytes)
        assert!(ptx.contains(".shared"), "PTX missing shared memory");
        assert!(ptx.contains("smem[16448]"), "PTX missing tiled shared memory (16448 bytes)");

        // Verify FMA instruction present (tiled accumulation from shared memory)
        assert!(ptx.contains("fma"), "PTX missing fma instruction");

        // Verify barrier synchronization present (cooperative tiling requires barriers)
        assert!(ptx.contains("bar.sync"), "PTX missing bar.sync for tile synchronization");

        // Verify shared memory load/store (tiled kernel uses ld.shared / st.shared)
        assert!(ptx.contains("ld.shared"), "PTX missing ld.shared for tile reads");
        assert!(ptx.contains("st.shared"), "PTX missing st.shared for tile writes");
    }

    #[test]
    fn test_nf4_gemm_ptx_targets() {
        let kernel = Nf4GemmKernel::new(64, 64, 64);

        let ptx_70 = kernel.emit_ptx_for_target("sm_70");
        assert!(ptx_70.contains("sm_70"));

        let ptx_89 = kernel.emit_ptx_for_target("sm_89");
        assert!(ptx_89.contains("sm_89"));
    }

    #[test]
    fn test_nf4_gemm_with_tile_size() {
        let kernel = Nf4GemmKernel::new(128, 128, 128).with_tile_size(16);
        assert_eq!(kernel.tile_size, 16);

        // Should still emit valid PTX
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("nf4_gemm_fused"));
    }

    #[test]
    fn test_nf4_gemm_transpose_kernel_name() {
        let kernel = Nf4GemmTransposeKernel::new(128, 896, 896);
        assert_eq!(kernel.name(), "nf4_gemm_transpose");
    }

    #[test]
    fn test_nf4_gemm_transpose_ptx_emits() {
        let kernel = Nf4GemmTransposeKernel::new(128, 896, 896);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains("nf4_gemm_transpose"), "PTX missing kernel name");
        assert!(ptx.contains("a_ptr"), "PTX missing a_ptr param");
        assert!(ptx.contains("b_nf4_ptr"), "PTX missing b_nf4_ptr param");
        assert!(ptx.contains("c_ptr"), "PTX missing c_ptr param");
        assert!(ptx.contains("fma"), "PTX missing fma instruction");
    }

    #[test]
    fn test_nf4_gemm_transpose_num_blocks() {
        let kernel = Nf4GemmTransposeKernel::new(128, 2560, 2560);
        assert_eq!(kernel.num_blocks_per_col(), 40); // 2560/64
    }

    #[test]
    fn test_nf4_gemm_tiled_smem_layout() {
        // Validate shared memory layout constants
        assert_eq!(SMEM_LUT_OFFSET, 0);
        assert_eq!(SMEM_LUT_BYTES, 64); // 16 × f32
        assert_eq!(SMEM_A_OFFSET, 64);
        assert_eq!(SMEM_A_BYTES, 8192); // 32 × 64 × 4
        assert_eq!(SMEM_B_OFFSET, 8256); // 64 + 8192
        assert_eq!(SMEM_B_BYTES, 8192); // 64 × 32 × 4
        assert_eq!(SMEM_TOTAL, 16448); // 64 + 8192 + 8192

        // Validate tile dimensions
        assert_eq!(TILE_M, 32);
        assert_eq!(TILE_N, 32);
        assert_eq!(TILE_K, NF4_BLOCK_SIZE_U32); // 64

        // 1024 threads per block = TILE_M × TILE_N
        assert_eq!(TILE_M * TILE_N, 1024);

        // 2048 elements per A tile / B tile, 2 passes of 1024 threads
        assert_eq!(TILE_M * TILE_K, 2048);
        assert_eq!(TILE_K * TILE_N, 2048);

        // Total shared memory fits within 48KB hardware limit
        assert!(SMEM_TOTAL <= 48 * 1024, "Shared memory exceeds 48KB limit");
    }

    #[test]
    fn test_nf4_gemm_tiled_barrier_safety() {
        // The tiled kernel has bar.sync calls inside a loop, which requires
        // all threads to reach the barrier (no early exit before barrier).
        // Verify that the OOB check is AFTER the tile loop, not before.
        let kernel = Nf4GemmKernel::new(128, 128, 128);
        let ptx = kernel.emit_ptx();

        // Count barriers: should have 3 total (1 for LUT load, 2 per tile iteration in loop)
        let bar_count = ptx.matches("bar.sync").count();
        assert!(bar_count >= 3, "Expected at least 3 bar.sync (LUT + tile loop), got {bar_count}");

        // Verify exit label comes AFTER tile_loop_done
        let tile_done_pos = ptx.find("tile_loop_done").expect("missing tile_loop_done label");
        let exit_pos = ptx.find("exit:").expect("missing exit label");
        assert!(exit_pos > tile_done_pos, "exit label must come after tile_loop_done");
    }

    #[test]
    fn test_nf4_gemm_qwen3_4b_dimensions() {
        // Qwen3-4B: hidden=2560, intermediate=6912, heads=32, kv_heads=8, head_dim=80
        // seq_len=128 for training

        // Q/O projection: (128, 2560, 2560)
        let q_proj = Nf4GemmKernel::new(128, 2560, 2560);
        assert_eq!(q_proj.num_blocks_per_col(), 40); // 2560/64

        // K/V projection: (128, 640, 2560)  (kv_hidden = 8 * 80 = 640)
        let kv_proj = Nf4GemmKernel::new(128, 640, 2560);
        assert_eq!(kv_proj.num_blocks_per_col(), 40);

        // Gate/Up projection: (128, 6912, 2560)
        let gate_proj = Nf4GemmKernel::new(128, 6912, 2560);
        assert_eq!(gate_proj.num_blocks_per_col(), 40);

        // Down projection: (128, 2560, 6912)
        let down_proj = Nf4GemmKernel::new(128, 2560, 6912);
        assert_eq!(down_proj.num_blocks_per_col(), 108); // 6912/64
    }
}
