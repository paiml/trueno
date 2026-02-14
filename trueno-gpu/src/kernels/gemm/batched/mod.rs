//! Batched GEMM Kernel (3D batched matrix multiplication)
//!
//! Implements C[b] = A[b] @ B[b] for batch_size independent matrix multiplications.

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxType};

/// Batched GEMM configuration
#[derive(Debug, Clone)]
pub struct BatchedGemmConfig {
    /// Batch size (number of independent matrix multiplications)
    pub batch: u32,
    /// M dimension (rows of A and C)
    pub m: u32,
    /// N dimension (cols of B and C)
    pub n: u32,
    /// K dimension (cols of A, rows of B)
    pub k: u32,
    /// Tile size for shared memory
    pub tile_size: u32,
}

impl Default for BatchedGemmConfig {
    fn default() -> Self {
        Self {
            batch: 1,
            m: 1024,
            n: 1024,
            k: 1024,
            tile_size: 16,
        }
    }
}

/// Batched GEMM kernel for 3D tensor matmul
/// Each batch is processed by a separate thread block in the z-dimension
#[derive(Debug, Clone)]
pub struct BatchedGemmKernel {
    /// Kernel configuration
    pub config: BatchedGemmConfig,
    variant: BatchedGemmVariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchedGemmVariant {
    Naive,
    Tiled,
    /// Tiled with 4x unrolled inner loop (WAPR-PERF-009)
    TiledUnrolled,
    /// WMMA FP16 using Tensor Core PTX intrinsics (WAPR-PERF-011)
    /// Requires sm_70+ (Volta or later). Dimensions must be multiples of 16.
    WmmaFp16,
}

impl BatchedGemmKernel {
    /// Create naive batched GEMM kernel (for correctness testing)
    /// Pattern: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
    #[must_use]
    pub fn naive(batch: u32, m: u32, n: u32, k: u32) -> Self {
        Self {
            config: BatchedGemmConfig {
                batch,
                m,
                n,
                k,
                ..Default::default()
            },
            variant: BatchedGemmVariant::Naive,
        }
    }

    /// Create tiled batched GEMM kernel (for performance)
    /// Pattern: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
    #[must_use]
    pub fn tiled(batch: u32, m: u32, n: u32, k: u32, tile_size: u32) -> Self {
        Self {
            config: BatchedGemmConfig {
                batch,
                m,
                n,
                k,
                tile_size,
            },
            variant: BatchedGemmVariant::Tiled,
        }
    }

    /// Create tiled batched GEMM kernel with 4x unrolled inner loop (WAPR-PERF-009)
    /// Reduces loop overhead from 12:1 to ~3:1 instructions per FMA
    /// Pattern: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
    #[must_use]
    pub fn tiled_unrolled(batch: u32, m: u32, n: u32, k: u32, tile_size: u32) -> Self {
        Self {
            config: BatchedGemmConfig {
                batch,
                m,
                n,
                k,
                tile_size,
            },
            variant: BatchedGemmVariant::TiledUnrolled,
        }
    }

    /// Create WMMA FP16 batched GEMM kernel using Tensor Core PTX intrinsics (WAPR-PERF-011)
    /// Requires sm_70+ (Volta or later). Input is FP32, converted to FP16 internally.
    /// Pattern: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
    /// Dimensions m, n must be multiples of 16 for optimal performance.
    #[must_use]
    pub fn wmma_fp16(batch: u32, m: u32, n: u32, k: u32) -> Self {
        Self {
            config: BatchedGemmConfig {
                batch,
                m,
                n,
                k,
                tile_size: 16, // WMMA uses 16x16x16 tiles
            },
            variant: BatchedGemmVariant::WmmaFp16,
        }
    }

    fn build_naive(&self) -> PtxKernel {
        // Naive Batched GEMM: each thread computes one element of C[batch, row, col]
        // Grid: (n, m, batch) - z-dimension indexes batch
        let m_val = self.config.m;
        let n_val = self.config.n;
        let k_val = self.config.k;

        PtxKernel::new("batched_gemm_naive")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "batch")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .build(|ctx| {
                // Get batch index from ctaid.z
                let batch_idx = ctx.special_reg(crate::ptx::PtxReg::CtaIdZ);

                // Calculate row and column from thread/block IDs
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);
                let ntid_y = ctx.special_reg(crate::ptx::PtxReg::NtidY);
                let tid_y = ctx.special_reg(crate::ptx::PtxReg::TidY);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ntid_x = ctx.special_reg(crate::ptx::PtxReg::NtidX);
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);

                let row = ctx.mad_lo_u32(ctaid_y, ntid_y, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);

                // Bounds check
                let batch_param = ctx.load_param_u32("batch");
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                let pred_batch = ctx.setp_ge_u32(batch_idx, batch_param);
                ctx.branch_if(pred_batch, "exit");
                let pred_m = ctx.setp_ge_u32(row, m_param);
                ctx.branch_if(pred_m, "exit");
                let pred_n = ctx.setp_ge_u32(col, n_param);
                ctx.branch_if(pred_n, "exit");

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Calculate batch offsets using immediate values
                // A batch offset = batch_idx * m * k * 4
                // B batch offset = batch_idx * k * n * 4
                // C batch offset = batch_idx * m * n * 4
                let a_batch_offset = ctx.mul_wide_u32(batch_idx, m_val * k_val * 4);
                let b_batch_offset = ctx.mul_wide_u32(batch_idx, k_val * n_val * 4);
                let c_batch_offset = ctx.mul_wide_u32(batch_idx, m_val * n_val * 4);

                let a_batch_ptr = ctx.add_u64(a_ptr, a_batch_offset);
                let b_batch_ptr = ctx.add_u64(b_ptr, b_batch_offset);
                let c_batch_ptr = ctx.add_u64(c_ptr, c_batch_offset);

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Calculate base offset for A[row, 0]
                let row_offset = ctx.mul_wide_u32(row, k_val * 4);
                let a_row_ptr = ctx.add_u64(a_batch_ptr, row_offset);

                // Calculate base offset for B[0, col]
                let col_offset = ctx.mul_wide_u32(col, 4);
                let b_col_base = ctx.add_u64(b_batch_ptr, col_offset);

                // Loop over K dimension
                let i = ctx.mov_u32_imm(0);

                ctx.label("loop_k");

                let pred_k = ctx.setp_ge_u32(i, k_param);
                ctx.branch_if(pred_k, "loop_end");

                // Load A[row, i]
                let i_offset = ctx.mul_wide_u32(i, 4);
                let a_addr = ctx.add_u64(a_row_ptr, i_offset);
                let a_val = ctx.ld_global_f32(a_addr);

                // Load B[i, col]
                let b_row_offset = ctx.mul_wide_u32(i, n_val * 4);
                let b_addr = ctx.add_u64(b_col_base, b_row_offset);
                let b_val = ctx.ld_global_f32(b_addr);

                // acc += a_val * b_val
                ctx.fma_f32_inplace(acc, a_val, b_val);

                ctx.add_u32_inplace(i, 1);
                ctx.branch("loop_k");

                ctx.label("loop_end");

                // Store result: C[batch, row, col]
                let c_row_offset = ctx.mul_wide_u32(row, n_val * 4);
                let c_row_ptr = ctx.add_u64(c_batch_ptr, c_row_offset);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_addr = ctx.add_u64(c_row_ptr, c_col_offset);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }

    fn build_tiled(&self) -> PtxKernel {
        let tile_size = self.config.tile_size;
        let smem_size = tile_size * tile_size * 4 * 2; // A and B tiles
        let n_tiles = (self.config.k + tile_size - 1) / tile_size;
        let m_val = self.config.m;
        let n_val = self.config.n;
        let k_val = self.config.k;

        PtxKernel::new("batched_gemm_tiled")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "batch")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Get batch index from ctaid.z
                let batch_idx = ctx.special_reg(crate::ptx::PtxReg::CtaIdZ);

                // Thread and block indices
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let tid_y = ctx.special_reg(crate::ptx::PtxReg::TidY);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);

                let tile_size_reg = ctx.mov_u32_imm(tile_size);

                // Global row and column
                let row = ctx.mad_lo_u32(ctaid_y, tile_size_reg, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, tile_size_reg, tid_x);

                // Load parameters - DON'T exit early (PARITY-114)
                let batch_param = ctx.load_param_u32("batch");
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                // Compute predicates for valid output
                let batch_valid = ctx.setp_lt_u32(batch_idx, batch_param);
                let row_valid = ctx.setp_lt_u32(row, m_param);
                let col_valid = ctx.setp_lt_u32(col, n_param);

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Calculate batch offsets using immediate values
                let a_batch_offset = ctx.mul_wide_u32(batch_idx, m_val * k_val * 4);
                let b_batch_offset = ctx.mul_wide_u32(batch_idx, k_val * n_val * 4);
                let c_batch_offset = ctx.mul_wide_u32(batch_idx, m_val * n_val * 4);

                let a_batch_ptr = ctx.add_u64(a_ptr, a_batch_offset);
                let b_batch_ptr = ctx.add_u64(b_ptr, b_batch_offset);
                let c_batch_ptr = ctx.add_u64(c_ptr, c_batch_offset);

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Tile loop counter
                let tile_idx = ctx.mov_u32_imm(0);
                let n_tiles_reg = ctx.mov_u32_imm(n_tiles);

                ctx.label("tile_loop");

                let tile_done = ctx.setp_ge_u32(tile_idx, n_tiles_reg);
                ctx.branch_if(tile_done, "tile_loop_end");

                // Shared memory offsets
                let smem_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, tid_x);
                let smem_a_offset = ctx.mul_u32(smem_idx, 4);
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 4);
                let smem_b_offset = ctx.add_u32_reg(smem_b_base, smem_a_offset);

                // Load A tile
                let tile_k_offset = ctx.mul_u32(tile_idx, tile_size);
                let a_col = ctx.add_u32_reg(tile_k_offset, tid_x);
                let a_col_valid = ctx.setp_lt_u32(a_col, k_param);

                let zero_a = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_a_offset, zero_a);

                ctx.branch_if_not(batch_valid, "skip_a_load");
                ctx.branch_if_not(row_valid, "skip_a_load");
                ctx.branch_if_not(a_col_valid, "skip_a_load");

                let row_offset_a = ctx.mul_wide_u32(row, k_val * 4);
                let col_offset_a = ctx.mul_wide_u32(a_col, 4);
                let a_row_base = ctx.add_u64(a_batch_ptr, row_offset_a);
                let a_addr = ctx.add_u64(a_row_base, col_offset_a);
                let a_val = ctx.ld_global_f32(a_addr);
                ctx.st_shared_f32(smem_a_offset, a_val);

                ctx.label("skip_a_load");

                // Load B tile
                let b_row = ctx.add_u32_reg(tile_k_offset, tid_y);
                let b_row_valid = ctx.setp_lt_u32(b_row, k_param);

                let zero_b = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_b_offset, zero_b);

                ctx.branch_if_not(batch_valid, "skip_b_load");
                ctx.branch_if_not(b_row_valid, "skip_b_load");
                ctx.branch_if_not(col_valid, "skip_b_load");

                let row_offset_b = ctx.mul_wide_u32(b_row, n_val * 4);
                let col_offset_b = ctx.mul_wide_u32(col, 4);
                let b_row_base = ctx.add_u64(b_batch_ptr, row_offset_b);
                let b_addr = ctx.add_u64(b_row_base, col_offset_b);
                let b_val = ctx.ld_global_f32(b_addr);
                ctx.st_shared_f32(smem_b_offset, b_val);

                ctx.label("skip_b_load");

                ctx.bar_sync(0);

                // Inner loop: accumulate from shared memory
                let inner_k = ctx.mov_u32_imm(0);

                ctx.label("inner_k_loop");

                let inner_done = ctx.setp_ge_u32(inner_k, tile_size_reg);
                ctx.branch_if(inner_done, "inner_k_end");

                let as_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, inner_k);
                let as_addr = ctx.mul_u32(as_idx, 4);
                let a_shared = ctx.ld_shared_f32(as_addr);

                let bs_idx = ctx.mad_lo_u32(inner_k, tile_size_reg, tid_x);
                let bs_idx_bytes = ctx.mul_u32(bs_idx, 4);
                let bs_addr = ctx.add_u32_reg(smem_b_base, bs_idx_bytes);
                let b_shared = ctx.ld_shared_f32(bs_addr);

                ctx.fma_f32_inplace(acc, a_shared, b_shared);

                ctx.add_u32_inplace(inner_k, 1);
                ctx.branch("inner_k_loop");

                ctx.label("inner_k_end");

                ctx.bar_sync(1);

                ctx.add_u32_inplace(tile_idx, 1);
                ctx.branch("tile_loop");

                ctx.label("tile_loop_end");

                // PARITY-114: Bounds check after tile loop
                ctx.branch_if_not(batch_valid, "exit");
                ctx.branch_if_not(row_valid, "exit");
                ctx.branch_if_not(col_valid, "exit");

                // Store result
                let c_row_offset = ctx.mul_wide_u32(row, n_val * 4);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_row_base = ctx.add_u64(c_batch_ptr, c_row_offset);
                let c_addr = ctx.add_u64(c_row_base, c_col_offset);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }

    /// Batched tiled GEMM with 4x unrolled inner loop (WAPR-PERF-009)
    #[allow(clippy::too_many_lines)]
    fn build_tiled_unrolled(&self) -> PtxKernel {
        let tile_size = self.config.tile_size;
        let smem_size = tile_size * tile_size * 4 * 2;
        let n_tiles = (self.config.k + tile_size - 1) / tile_size;
        let m_val = self.config.m;
        let n_val = self.config.n;
        let k_val = self.config.k;
        let batch_stride_a = m_val * k_val;
        let batch_stride_b = k_val * n_val;
        let batch_stride_c = m_val * n_val;

        // Unroll factor
        let unroll_factor = 4u32;
        let unrolled_iters = tile_size / unroll_factor;

        PtxKernel::new("batched_gemm_tiled_unrolled")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "batch")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Get batch index from ctaid.z
                let batch_idx = ctx.special_reg(crate::ptx::PtxReg::CtaIdZ);

                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let tid_y = ctx.special_reg(crate::ptx::PtxReg::TidY);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);

                let tile_size_reg = ctx.mov_u32_imm(tile_size);

                let row = ctx.mad_lo_u32(ctaid_y, tile_size_reg, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, tile_size_reg, tid_x);

                let batch_param = ctx.load_param_u32("batch");
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                let batch_valid = ctx.setp_lt_u32(batch_idx, batch_param);
                let row_valid = ctx.setp_lt_u32(row, m_param);
                let col_valid = ctx.setp_lt_u32(col, n_param);

                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Compute batch offsets
                let batch_offset_a = ctx.mul_wide_u32(batch_idx, batch_stride_a * 4);
                let batch_offset_b = ctx.mul_wide_u32(batch_idx, batch_stride_b * 4);
                let batch_offset_c = ctx.mul_wide_u32(batch_idx, batch_stride_c * 4);
                let a_batch_ptr = ctx.add_u64(a_ptr, batch_offset_a);
                let b_batch_ptr = ctx.add_u64(b_ptr, batch_offset_b);
                let c_batch_ptr = ctx.add_u64(c_ptr, batch_offset_c);

                let acc = ctx.mov_f32_imm(0.0);

                let tile_idx = ctx.mov_u32_imm(0);
                let n_tiles_reg = ctx.mov_u32_imm(n_tiles);

                ctx.label("tile_loop");

                let tile_done = ctx.setp_ge_u32(tile_idx, n_tiles_reg);
                ctx.branch_if(tile_done, "tile_loop_end");

                let smem_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, tid_x);
                let smem_a_offset = ctx.mul_u32(smem_idx, 4);
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 4);
                let smem_b_offset = ctx.add_u32_reg(smem_b_base, smem_a_offset);

                // Load A tile
                let tile_k_offset = ctx.mul_u32(tile_idx, tile_size);
                let a_col = ctx.add_u32_reg(tile_k_offset, tid_x);
                let a_col_valid = ctx.setp_lt_u32(a_col, k_param);

                let zero_a = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_a_offset, zero_a);

                ctx.branch_if_not(batch_valid, "skip_a_load");
                ctx.branch_if_not(row_valid, "skip_a_load");
                ctx.branch_if_not(a_col_valid, "skip_a_load");

                let row_offset_a = ctx.mul_wide_u32(row, k_val * 4);
                let col_offset_a = ctx.mul_wide_u32(a_col, 4);
                let a_row_base = ctx.add_u64(a_batch_ptr, row_offset_a);
                let a_addr = ctx.add_u64(a_row_base, col_offset_a);
                let a_val = ctx.ld_global_f32(a_addr);
                ctx.st_shared_f32(smem_a_offset, a_val);

                ctx.label("skip_a_load");

                // Load B tile
                let b_row = ctx.add_u32_reg(tile_k_offset, tid_y);
                let b_row_valid = ctx.setp_lt_u32(b_row, k_param);

                let zero_b = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_b_offset, zero_b);

                ctx.branch_if_not(batch_valid, "skip_b_load");
                ctx.branch_if_not(b_row_valid, "skip_b_load");
                ctx.branch_if_not(col_valid, "skip_b_load");

                let row_offset_b = ctx.mul_wide_u32(b_row, n_val * 4);
                let col_offset_b = ctx.mul_wide_u32(col, 4);
                let b_row_base = ctx.add_u64(b_batch_ptr, row_offset_b);
                let b_addr = ctx.add_u64(b_row_base, col_offset_b);
                let b_val = ctx.ld_global_f32(b_addr);
                ctx.st_shared_f32(smem_b_offset, b_val);

                ctx.label("skip_b_load");

                ctx.bar_sync(0);

                // ========================================
                // 4x UNROLLED INNER LOOP (WAPR-PERF-009)
                // ========================================
                let inner_k = ctx.mov_u32_imm(0);
                let unrolled_iters_reg = ctx.mov_u32_imm(unrolled_iters);

                ctx.label("inner_k_loop");

                let inner_done = ctx.setp_ge_u32(inner_k, unrolled_iters_reg);
                ctx.branch_if(inner_done, "inner_k_end");

                let k_base = ctx.mul_u32(inner_k, unroll_factor);

                // === Iteration 0 ===
                let k0 = k_base;
                let as_idx0 = ctx.mad_lo_u32(tid_y, tile_size_reg, k0);
                let as_addr0 = ctx.mul_u32(as_idx0, 4);
                let a_shared0 = ctx.ld_shared_f32(as_addr0);

                let bs_idx0 = ctx.mad_lo_u32(k0, tile_size_reg, tid_x);
                let bs_idx_bytes0 = ctx.mul_u32(bs_idx0, 4);
                let bs_addr0 = ctx.add_u32_reg(smem_b_base, bs_idx_bytes0);
                let b_shared0 = ctx.ld_shared_f32(bs_addr0);

                ctx.fma_f32_inplace(acc, a_shared0, b_shared0);

                // === Iteration 1 ===
                let k1 = ctx.add_u32(k_base, 1);
                let as_idx1 = ctx.mad_lo_u32(tid_y, tile_size_reg, k1);
                let as_addr1 = ctx.mul_u32(as_idx1, 4);
                let a_shared1 = ctx.ld_shared_f32(as_addr1);

                let bs_idx1 = ctx.mad_lo_u32(k1, tile_size_reg, tid_x);
                let bs_idx_bytes1 = ctx.mul_u32(bs_idx1, 4);
                let bs_addr1 = ctx.add_u32_reg(smem_b_base, bs_idx_bytes1);
                let b_shared1 = ctx.ld_shared_f32(bs_addr1);

                ctx.fma_f32_inplace(acc, a_shared1, b_shared1);

                // === Iteration 2 ===
                let k2 = ctx.add_u32(k_base, 2);
                let as_idx2 = ctx.mad_lo_u32(tid_y, tile_size_reg, k2);
                let as_addr2 = ctx.mul_u32(as_idx2, 4);
                let a_shared2 = ctx.ld_shared_f32(as_addr2);

                let bs_idx2 = ctx.mad_lo_u32(k2, tile_size_reg, tid_x);
                let bs_idx_bytes2 = ctx.mul_u32(bs_idx2, 4);
                let bs_addr2 = ctx.add_u32_reg(smem_b_base, bs_idx_bytes2);
                let b_shared2 = ctx.ld_shared_f32(bs_addr2);

                ctx.fma_f32_inplace(acc, a_shared2, b_shared2);

                // === Iteration 3 ===
                let k3 = ctx.add_u32(k_base, 3);
                let as_idx3 = ctx.mad_lo_u32(tid_y, tile_size_reg, k3);
                let as_addr3 = ctx.mul_u32(as_idx3, 4);
                let a_shared3 = ctx.ld_shared_f32(as_addr3);

                let bs_idx3 = ctx.mad_lo_u32(k3, tile_size_reg, tid_x);
                let bs_idx_bytes3 = ctx.mul_u32(bs_idx3, 4);
                let bs_addr3 = ctx.add_u32_reg(smem_b_base, bs_idx_bytes3);
                let b_shared3 = ctx.ld_shared_f32(bs_addr3);

                ctx.fma_f32_inplace(acc, a_shared3, b_shared3);

                ctx.add_u32_inplace(inner_k, 1);
                ctx.branch("inner_k_loop");

                ctx.label("inner_k_end");

                ctx.bar_sync(1);

                ctx.add_u32_inplace(tile_idx, 1);
                ctx.branch("tile_loop");

                ctx.label("tile_loop_end");

                ctx.branch_if_not(batch_valid, "exit");
                ctx.branch_if_not(row_valid, "exit");
                ctx.branch_if_not(col_valid, "exit");

                let c_row_offset = ctx.mul_wide_u32(row, n_val * 4);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_row_base = ctx.add_u64(c_batch_ptr, c_row_offset);
                let c_addr = ctx.add_u64(c_row_base, c_col_offset);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }

    /// Build WMMA FP16 batched GEMM kernel using Tensor Core PTX intrinsics (WAPR-PERF-011)
    /// Each batch is processed by a separate grid slice in the z-dimension.
    /// Uses cvta.shared.u64 pattern from WAPR-PERF-010 for correct WMMA loads.
    /// Launch config: grid_3d((m+15)/16, (n+15)/16, batch, 32, 1, 1)
    #[allow(clippy::too_many_lines)]
    fn build_wmma_fp16(&self) -> PtxKernel {
        use crate::ptx::WmmaLayout;

        let tile_size = 16_u32;
        let smem_size = tile_size * tile_size * 2 * 2; // Two FP16 tiles (A and B)
        let n_k_tiles = (self.config.k + tile_size - 1) / tile_size;
        let m_val = self.config.m;
        let n_val = self.config.n;
        let k_val = self.config.k;

        PtxKernel::new("batched_gemm_wmma_fp16")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "batch")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // WAPR-PERF-011: Batched WMMA for multi-head attention
                // Grid z-dimension indexes batch, x/y index 16x16 output tiles
                // One warp (32 threads) processes one output tile per batch

                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);
                let batch_idx = ctx.special_reg(crate::ptx::PtxReg::CtaIdZ);

                // Calculate output tile position
                let tile_size_reg = ctx.mov_u32_imm(tile_size);
                let tile_row = ctx.mul_u32(ctaid_y, tile_size);
                let tile_col = ctx.mul_u32(ctaid_x, tile_size);

                // Load parameters
                let batch_param = ctx.load_param_u32("batch");
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                // Compute predicates for valid tile
                let batch_valid = ctx.setp_lt_u32(batch_idx, batch_param);
                let tile_row_valid = ctx.setp_lt_u32(tile_row, m_param);
                let tile_col_valid = ctx.setp_lt_u32(tile_col, n_param);

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Calculate batch offsets
                // A batch offset = batch_idx * m * k * 4
                // B batch offset = batch_idx * k * n * 4
                // C batch offset = batch_idx * m * n * 4
                let a_batch_offset = ctx.mul_wide_u32(batch_idx, m_val * k_val * 4);
                let b_batch_offset = ctx.mul_wide_u32(batch_idx, k_val * n_val * 4);
                let c_batch_offset = ctx.mul_wide_u32(batch_idx, m_val * n_val * 4);

                let a_batch_ptr = ctx.add_u64(a_ptr, a_batch_offset);
                let b_batch_ptr = ctx.add_u64(b_ptr, b_batch_offset);
                let c_batch_ptr = ctx.add_u64(c_ptr, c_batch_offset);

                // Shared memory base addresses
                let smem_a_base = ctx.mov_u32_imm(0);
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 2); // After A tile (FP16)

                // Initialize accumulator fragments
                let frag_c = ctx.wmma_init_c_zero();

                // Loop counter for K tiles
                let k_tile_idx = ctx.mov_u32_imm(0);
                let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);

                ctx.label("k_tile_loop");
                let k_done = ctx.setp_ge_u32(k_tile_idx, n_k_tiles_reg);
                ctx.branch_if(k_done, "k_tile_end");

                // K offset for this tile
                let k_offset = ctx.mul_u32_reg(k_tile_idx, tile_size_reg);

                // === Load A tile to shared memory (FP32 global → FP16 shared) ===
                let elements_per_thread = ctx.mov_u32_imm(8);
                let my_start = ctx.mul_u32_reg(tid_x, elements_per_thread);

                let load_idx = ctx.mov_u32_imm(0);
                ctx.label("load_a_loop_batched");
                let load_done = ctx.setp_ge_u32(load_idx, elements_per_thread);
                ctx.branch_if(load_done, "load_a_end_batched");

                let elem_idx = ctx.add_u32_reg(my_start, load_idx);
                let row_in_tile = ctx.div_u32(elem_idx, 16);
                let col_in_tile = ctx.rem_u32(elem_idx, 16);

                // Store 0 first (default for out-of-bounds)
                let smem_a_offset = ctx.mul_u32(elem_idx, 2);
                let smem_a_addr = ctx.add_u32_reg(smem_a_base, smem_a_offset);
                let zero_f32 = ctx.mov_f32_imm(0.0);
                let zero_f16 = ctx.cvt_f16_f32(zero_f32);
                ctx.st_shared_f16(smem_a_addr, zero_f16);

                // Check bounds: a_row < m AND a_col < k
                let a_row = ctx.add_u32_reg(tile_row, row_in_tile);
                let a_col = ctx.add_u32_reg(k_offset, col_in_tile);
                let a_row_valid = ctx.setp_lt_u32(a_row, m_param);
                let a_col_valid = ctx.setp_lt_u32(a_col, k_param);

                ctx.branch_if_not(a_row_valid, "skip_a_load_batched");
                ctx.branch_if_not(a_col_valid, "skip_a_load_batched");
                ctx.branch_if_not(batch_valid, "skip_a_load_batched");

                // Global A address: A[batch, tile_row + row_in_tile, k_offset + col_in_tile]
                let k_reg = ctx.mov_u32_imm(k_val);
                let a_idx = ctx.mad_lo_u32(a_row, k_reg, a_col);
                let a_byte_offset = ctx.mul_wide_u32(a_idx, 4);
                let a_addr = ctx.add_u64(a_batch_ptr, a_byte_offset);

                let a_val_f32 = ctx.ld_global_f32(a_addr);
                let a_val_f16 = ctx.cvt_f16_f32(a_val_f32);
                ctx.st_shared_f16(smem_a_addr, a_val_f16);

                ctx.label("skip_a_load_batched");
                ctx.add_u32_inplace(load_idx, 1);
                ctx.branch("load_a_loop_batched");
                ctx.label("load_a_end_batched");

                // === Load B tile to shared memory ===
                let load_idx_b = ctx.mov_u32_imm(0);
                ctx.label("load_b_loop_batched");
                let load_b_done = ctx.setp_ge_u32(load_idx_b, elements_per_thread);
                ctx.branch_if(load_b_done, "load_b_end_batched");

                let elem_idx_b = ctx.add_u32_reg(my_start, load_idx_b);
                let row_in_tile_b = ctx.div_u32(elem_idx_b, 16);
                let col_in_tile_b = ctx.rem_u32(elem_idx_b, 16);

                let smem_b_offset = ctx.mul_u32(elem_idx_b, 2);
                let smem_b_addr = ctx.add_u32_reg(smem_b_base, smem_b_offset);
                let zero_b_f32 = ctx.mov_f32_imm(0.0);
                let zero_b_f16 = ctx.cvt_f16_f32(zero_b_f32);
                ctx.st_shared_f16(smem_b_addr, zero_b_f16);

                // Check bounds: b_row < k AND b_col < n
                let b_row = ctx.add_u32_reg(k_offset, row_in_tile_b);
                let b_col = ctx.add_u32_reg(tile_col, col_in_tile_b);
                let b_row_valid = ctx.setp_lt_u32(b_row, k_param);
                let b_col_valid = ctx.setp_lt_u32(b_col, n_param);

                ctx.branch_if_not(b_row_valid, "skip_b_load_batched");
                ctx.branch_if_not(b_col_valid, "skip_b_load_batched");
                ctx.branch_if_not(batch_valid, "skip_b_load_batched");

                // Global B address: B[batch, k_offset + row_in_tile, tile_col + col_in_tile]
                let n_reg = ctx.mov_u32_imm(n_val);
                let b_idx = ctx.mad_lo_u32(b_row, n_reg, b_col);
                let b_byte_offset = ctx.mul_wide_u32(b_idx, 4);
                let b_addr = ctx.add_u64(b_batch_ptr, b_byte_offset);

                let b_val_f32 = ctx.ld_global_f32(b_addr);
                let b_val_f16 = ctx.cvt_f16_f32(b_val_f32);
                ctx.st_shared_f16(smem_b_addr, b_val_f16);

                ctx.label("skip_b_load_batched");
                ctx.add_u32_inplace(load_idx_b, 1);
                ctx.branch("load_b_loop_batched");
                ctx.label("load_b_end_batched");

                // Synchronize before WMMA
                ctx.bar_sync(0);

                // === WMMA matrix multiply ===
                // WAPR-PERF-010 FIX: Use cvta.shared.u64 to get generic pointer
                let smem_generic_base = ctx.shared_base_addr();

                // Load A fragment from shared memory
                let frag_a = ctx.wmma_load_a_f16(smem_generic_base, 16, WmmaLayout::RowMajor);

                // Load B fragment from shared memory
                // WAPR-PERF-014 FIX: B is stored row-major in shared memory, so use RowMajor
                let smem_b_offset_u64 = ctx.cvt_u64_u32(smem_b_base);
                let smem_b_ptr = ctx.add_u64(smem_generic_base, smem_b_offset_u64);
                let frag_b = ctx.wmma_load_b_f16(smem_b_ptr, 16, WmmaLayout::RowMajor);

                // Matrix multiply-accumulate: D = A * B + C
                let frag_d = ctx.wmma_mma_f16_f32(&frag_a, &frag_b, &frag_c);

                // WAPR-PERF-010 FIX: Copy D → C for accumulation across K tiles
                // The MMA instruction outputs to new registers, so we must copy
                // the result back to the accumulator for the next iteration
                for (c_reg, d_reg) in frag_c.iter().zip(frag_d.iter()) {
                    ctx.mov_f32_reg(*c_reg, *d_reg);
                }

                // Synchronize after WMMA (before next tile load)
                ctx.bar_sync(1);

                ctx.add_u32_inplace(k_tile_idx, 1);
                ctx.branch("k_tile_loop");

                ctx.label("k_tile_end");

                // Store result to global memory (only valid threads)
                ctx.branch_if_not(batch_valid, "exit_batched");
                ctx.branch_if_not(tile_row_valid, "exit_batched");
                ctx.branch_if_not(tile_col_valid, "exit_batched");

                // C output address with batch offset
                let c_tile_row_offset = ctx.mul_wide_u32(tile_row, n_val * 4);
                let c_tile_col_offset = ctx.mul_wide_u32(tile_col, 4);
                let c_tile_base = ctx.add_u64(c_batch_ptr, c_tile_row_offset);
                let c_tile_addr = ctx.add_u64(c_tile_base, c_tile_col_offset);

                ctx.wmma_store_d_f32(c_tile_addr, &frag_c, n_val, WmmaLayout::RowMajor);

                ctx.label("exit_batched");
                ctx.ret();
            })
    }
}

impl Kernel for BatchedGemmKernel {
    fn name(&self) -> &str {
        match self.variant {
            BatchedGemmVariant::Naive => "batched_gemm_naive",
            BatchedGemmVariant::Tiled => "batched_gemm_tiled",
            BatchedGemmVariant::TiledUnrolled => "batched_gemm_tiled_unrolled",
            BatchedGemmVariant::WmmaFp16 => "batched_gemm_wmma_fp16",
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        match self.variant {
            BatchedGemmVariant::Naive => self.build_naive(),
            BatchedGemmVariant::Tiled => self.build_tiled(),
            BatchedGemmVariant::TiledUnrolled => self.build_tiled_unrolled(),
            BatchedGemmVariant::WmmaFp16 => self.build_wmma_fp16(),
        }
    }
}

#[cfg(test)]
mod tests;
