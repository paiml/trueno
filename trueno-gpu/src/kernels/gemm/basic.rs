//! Basic GEMM Kernel (naive, tiled, tensor core variants)
//!
//! Implements C = alpha * A @ B + beta * C for standard 2D matrix multiplication.

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxType};

/// GEMM kernel configuration
#[derive(Debug, Clone)]
pub struct GemmConfig {
    /// M dimension (rows of A and C)
    pub m: u32,
    /// N dimension (cols of B and C)
    pub n: u32,
    /// K dimension (cols of A, rows of B)
    pub k: u32,
    /// Tile size for shared memory
    pub tile_size: u32,
    /// Use Tensor Cores (requires FP16 and SM >= 70)
    pub use_tensor_cores: bool,
}

impl Default for GemmConfig {
    fn default() -> Self {
        Self {
            m: 1024,
            n: 1024,
            k: 1024,
            tile_size: 32,
            use_tensor_cores: false,
        }
    }
}

/// GEMM kernel
#[derive(Debug, Clone)]
pub struct GemmKernel {
    /// Kernel configuration
    pub config: GemmConfig,
    variant: GemmVariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GemmVariant {
    Naive,
    Tiled,
    /// Tiled with 4x unrolled inner loop (WAPR-PERF-009)
    TiledUnrolled,
    TensorCore,
    /// True WMMA using Tensor Core PTX intrinsics (sm_70+)
    WmmaFp16,
}

impl GemmKernel {
    /// Create naive GEMM kernel (for correctness testing)
    #[must_use]
    pub fn naive(m: u32, n: u32, k: u32) -> Self {
        Self {
            config: GemmConfig {
                m,
                n,
                k,
                ..Default::default()
            },
            variant: GemmVariant::Naive,
        }
    }

    /// Create tiled GEMM kernel (for performance)
    #[must_use]
    pub fn tiled(m: u32, n: u32, k: u32, tile_size: u32) -> Self {
        Self {
            config: GemmConfig {
                m,
                n,
                k,
                tile_size,
                ..Default::default()
            },
            variant: GemmVariant::Tiled,
        }
    }

    /// Create tiled GEMM kernel with 4x unrolled inner loop (WAPR-PERF-009)
    /// Reduces loop overhead from 12:1 to ~4:1 instructions per FMA
    #[must_use]
    pub fn tiled_unrolled(m: u32, n: u32, k: u32, tile_size: u32) -> Self {
        Self {
            config: GemmConfig {
                m,
                n,
                k,
                tile_size,
                ..Default::default()
            },
            variant: GemmVariant::TiledUnrolled,
        }
    }

    /// Create Tensor Core GEMM kernel (highest performance)
    #[must_use]
    pub fn tensor_core(m: u32, n: u32, k: u32) -> Self {
        Self {
            config: GemmConfig {
                m,
                n,
                k,
                use_tensor_cores: true,
                ..Default::default()
            },
            variant: GemmVariant::TensorCore,
        }
    }

    /// Create WMMA FP16 GEMM kernel using true Tensor Core PTX intrinsics
    /// Requires sm_70+ (Volta or later). Input is FP32, converted to FP16 internally.
    /// Output is FP32. Dimensions must be multiples of 16.
    #[must_use]
    pub fn wmma_fp16(m: u32, n: u32, k: u32) -> Self {
        Self {
            config: GemmConfig {
                m,
                n,
                k,
                tile_size: 16, // WMMA uses 16x16x16 tiles
                use_tensor_cores: true,
            },
            variant: GemmVariant::WmmaFp16,
        }
    }
}

impl Kernel for GemmKernel {
    fn name(&self) -> &str {
        match self.variant {
            GemmVariant::Naive => "gemm_naive",
            GemmVariant::Tiled => "gemm_tiled",
            GemmVariant::TiledUnrolled => "gemm_tiled_unrolled",
            GemmVariant::TensorCore => "gemm_tensor_core",
            GemmVariant::WmmaFp16 => "gemm_wmma_fp16",
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        match self.variant {
            GemmVariant::Naive => self.build_naive(),
            GemmVariant::Tiled => self.build_tiled(),
            GemmVariant::TiledUnrolled => self.build_tiled_unrolled(),
            GemmVariant::TensorCore => self.build_tensor_core(),
            GemmVariant::WmmaFp16 => self.build_wmma_fp16(),
        }
    }
}

impl GemmKernel {
    fn build_naive(&self) -> PtxKernel {
        // Naive GEMM: each thread computes one element of C
        // C[row, col] = sum(A[row, i] * B[i, col] for i in 0..K)
        let k_val = self.config.k;

        PtxKernel::new("gemm_naive")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .build(|ctx| {
                // Calculate row and column from thread/block IDs
                // row = ctaid.y * ntid.y + tid.y
                // col = ctaid.x * ntid.x + tid.x
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);
                let ntid_y = ctx.special_reg(crate::ptx::PtxReg::NtidY);
                let tid_y = ctx.special_reg(crate::ptx::PtxReg::TidY);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ntid_x = ctx.special_reg(crate::ptx::PtxReg::NtidX);
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);

                let row = ctx.mad_lo_u32(ctaid_y, ntid_y, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);

                // Bounds check: if (row >= m || col >= n) return
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                let pred_m = ctx.setp_ge_u32(row, m_param);
                ctx.branch_if(pred_m, "exit");
                let pred_n = ctx.setp_ge_u32(col, n_param);
                ctx.branch_if(pred_n, "exit");

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Calculate base offset for A[row, 0] = a_ptr + row * K * 4
                let row_offset = ctx.mul_wide_u32(row, k_val * 4);
                let a_row_ptr = ctx.add_u64(a_ptr, row_offset);

                // Calculate base offset for B[0, col] = b_ptr + col * 4
                let col_offset = ctx.mul_wide_u32(col, 4);
                let b_col_base = ctx.add_u64(b_ptr, col_offset);

                // Loop over K dimension
                // For simplicity, unroll by 1 (production would unroll more)
                let i = ctx.mov_u32_imm(0);

                ctx.label("loop_k");

                // Check loop condition: if (i >= k) goto loop_end
                let pred_k = ctx.setp_ge_u32(i, k_param);
                ctx.branch_if(pred_k, "loop_end");

                // Load A[row, i] = a_row_ptr + i * 4
                let i_offset = ctx.mul_wide_u32(i, 4);
                let a_addr = ctx.add_u64(a_row_ptr, i_offset);
                let a_val = ctx.ld_global_f32(a_addr);

                // Load B[i, col] = b_col_base + i * N * 4
                let b_row_offset = ctx.mul_wide_u32(i, self.config.n * 4);
                let b_addr = ctx.add_u64(b_col_base, b_row_offset);
                let b_val = ctx.ld_global_f32(b_addr);

                // acc += a_val * b_val (FMA) - IN-PLACE UPDATE
                ctx.fma_f32_inplace(acc, a_val, b_val);

                // i++ - IN-PLACE UPDATE
                ctx.add_u32_inplace(i, 1);

                // Branch back to loop
                ctx.branch("loop_k");

                ctx.label("loop_end");

                // Store result: C[row, col] = c_ptr + (row * N + col) * 4
                let c_row_offset = ctx.mul_wide_u32(row, self.config.n * 4);
                let c_row_ptr = ctx.add_u64(c_ptr, c_row_offset);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_addr = ctx.add_u64(c_row_ptr, c_col_offset);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }

    fn build_tiled(&self) -> PtxKernel {
        let tile_size = self.config.tile_size;
        let smem_size = tile_size * tile_size * 4 * 2; // A and B tiles, f32
        let n_tiles = (self.config.k + tile_size - 1) / tile_size;

        PtxKernel::new("gemm_tiled")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Tiled GEMM: Uses shared memory to reduce global memory traffic
                //
                // Algorithm:
                // 1. Each thread block computes a TILE_SIZE x TILE_SIZE tile of C
                // 2. Loop over tiles along K dimension
                // 3. Load A tile and B tile into shared memory
                // 4. Synchronize threads
                // 5. Each thread computes partial results from shared memory
                // 6. Store accumulated result to C

                // Thread and block indices
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let tid_y = ctx.special_reg(crate::ptx::PtxReg::TidY);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);

                // Tile size as a register (needed throughout)
                let tile_size_reg = ctx.mov_u32_imm(tile_size);

                // Global row and column
                let row = ctx.mad_lo_u32(ctaid_y, tile_size_reg, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, tile_size_reg, tid_x);

                // Load parameters (but DON'T exit early - all threads must participate in barriers)
                // PARITY-114 FIX: Bounds check moved to after tile_loop_end
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                // Compute predicates for valid row/col (used for predicated loads)
                let row_valid = ctx.setp_lt_u32(row, m_param);
                let col_valid = ctx.setp_lt_u32(col, n_param);

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Initialize accumulator to 0.0
                let acc = ctx.mov_f32_imm(0.0);

                // Tile loop counter
                let tile_idx = ctx.mov_u32_imm(0);
                let n_tiles_reg = ctx.mov_u32_imm(n_tiles);

                ctx.label("tile_loop");

                // Check if done with all tiles
                let tile_done = ctx.setp_ge_u32(tile_idx, n_tiles_reg);
                ctx.branch_if(tile_done, "tile_loop_end");

                // Calculate shared memory address for this thread's position
                // As[tid_y][tid_x] and Bs[tid_y][tid_x]
                // Note: Shared memory uses 32-bit addressing, not 64-bit!
                let smem_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, tid_x);
                let smem_a_offset = ctx.mul_u32(smem_idx, 4); // u32 for shared memory
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 4);
                let smem_b_offset = ctx.add_u32_reg(smem_b_base, smem_a_offset); // u32 addition

                // PARITY-114 FIX: All threads must load something to shared memory
                // Strategy: Store 0.0 first, then conditionally overwrite with real value

                // Load A[row, tile_idx * TILE + tid_x] into shared memory As[tid_y][tid_x]
                let tile_k_offset = ctx.mul_u32(tile_idx, tile_size);
                let a_col = ctx.add_u32_reg(tile_k_offset, tid_x);

                // Check if A load is in bounds: row < m AND a_col < k
                // Use two branches instead of and_pred to reduce predicate pressure
                let a_col_valid = ctx.setp_lt_u32(a_col, k_param);

                // Store 0.0 to shared memory first (default for out-of-bounds)
                let zero_a = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_a_offset, zero_a);

                // If out of bounds, skip load (check both conditions with separate branches)
                ctx.branch_if_not(row_valid, "skip_a_load");
                ctx.branch_if_not(a_col_valid, "skip_a_load");
                let row_offset_a = ctx.mul_wide_u32(row, self.config.k * 4);
                let col_offset_a = ctx.mul_wide_u32(a_col, 4);
                let a_row_base = ctx.add_u64(a_ptr, row_offset_a);
                let a_addr = ctx.add_u64(a_row_base, col_offset_a);
                let a_val = ctx.ld_global_f32(a_addr);
                ctx.st_shared_f32(smem_a_offset, a_val);
                ctx.label("skip_a_load");

                // Load B[tile_idx * TILE + tid_y, col] into shared memory Bs[tid_y][tid_x]
                let b_row = ctx.add_u32_reg(tile_k_offset, tid_y);

                // Check if B load is in bounds: b_row < k AND col < n
                // Use two branches instead of and_pred to reduce predicate pressure
                let b_row_valid = ctx.setp_lt_u32(b_row, k_param);

                // Store 0.0 to shared memory first (default for out-of-bounds)
                let zero_b = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_b_offset, zero_b);

                // If out of bounds, skip load (check both conditions with separate branches)
                ctx.branch_if_not(b_row_valid, "skip_b_load");
                ctx.branch_if_not(col_valid, "skip_b_load");
                let row_offset_b = ctx.mul_wide_u32(b_row, self.config.n * 4);
                let col_offset_b = ctx.mul_wide_u32(col, 4);
                let b_row_base = ctx.add_u64(b_ptr, row_offset_b);
                let b_addr = ctx.add_u64(b_row_base, col_offset_b);
                let b_val = ctx.ld_global_f32(b_addr);
                ctx.st_shared_f32(smem_b_offset, b_val);
                ctx.label("skip_b_load");

                // Synchronize threads after loading tile
                ctx.bar_sync(0);

                // Inner loop: accumulate products from shared memory tile
                let inner_k = ctx.mov_u32_imm(0);

                ctx.label("inner_k_loop");

                let inner_done = ctx.setp_ge_u32(inner_k, tile_size_reg);
                ctx.branch_if(inner_done, "inner_k_end");

                // Load As[tid_y][inner_k] = smem[tid_y * TILE + inner_k]
                // Shared memory uses 32-bit addressing
                let as_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, inner_k);
                let as_addr = ctx.mul_u32(as_idx, 4); // u32 for shared memory
                let a_shared = ctx.ld_shared_f32(as_addr);

                // Load Bs[inner_k][tid_x] = smem[TILE*TILE + inner_k * TILE + tid_x]
                let bs_idx = ctx.mad_lo_u32(inner_k, tile_size_reg, tid_x);
                let bs_idx_bytes = ctx.mul_u32(bs_idx, 4); // u32 for shared memory
                let bs_addr = ctx.add_u32_reg(smem_b_base, bs_idx_bytes); // u32 addition
                let b_shared = ctx.ld_shared_f32(bs_addr);

                // acc += a_shared * b_shared - IN-PLACE UPDATE
                ctx.fma_f32_inplace(acc, a_shared, b_shared);

                // inner_k++ - IN-PLACE UPDATE
                ctx.add_u32_inplace(inner_k, 1);
                ctx.branch("inner_k_loop");

                ctx.label("inner_k_end");

                // Synchronize before loading next tile
                ctx.bar_sync(1);

                // tile_idx++ - IN-PLACE UPDATE
                ctx.add_u32_inplace(tile_idx, 1);
                ctx.branch("tile_loop");

                ctx.label("tile_loop_end");

                // PARITY-114 FIX: Bounds check HERE (after all threads finished tile loop)
                // Only threads with valid output coordinates store to C
                // Note: Use two branches instead of and_pred to reduce predicate register pressure
                // (PTX only has 8 predicate registers p0-p7)
                ctx.branch_if_not(row_valid, "exit");
                ctx.branch_if_not(col_valid, "exit");

                // Store result: C[row, col] = c_ptr + row * N + col
                let c_row_offset = ctx.mul_wide_u32(row, self.config.n * 4);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_row_base = ctx.add_u64(c_ptr, c_row_offset);
                let c_addr = ctx.add_u64(c_row_base, c_col_offset);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }

    /// Tiled GEMM with 4x unrolled inner loop (WAPR-PERF-009)
    ///
    /// Reduces loop overhead from 12:1 to ~3:1 instructions per FMA.
    /// The inner K loop processes 4 elements at a time, reducing the
    /// number of branch/compare instructions by 4x.
    #[allow(clippy::too_many_lines)]
    fn build_tiled_unrolled(&self) -> PtxKernel {
        let tile_size = self.config.tile_size;
        let smem_size = tile_size * tile_size * 4 * 2; // A and B tiles, f32
        let n_tiles = (self.config.k + tile_size - 1) / tile_size;

        // Unroll factor: process 4 elements per inner loop iteration
        let unroll_factor = 4u32;
        // Number of unrolled iterations (tile_size must be divisible by 4)
        let unrolled_iters = tile_size / unroll_factor;

        PtxKernel::new("gemm_tiled_unrolled")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Tiled GEMM with 4x unrolled inner loop
                //
                // Algorithm:
                // 1. Each thread block computes a TILE_SIZE x TILE_SIZE tile of C
                // 2. Loop over tiles along K dimension
                // 3. Load A tile and B tile into shared memory
                // 4. Synchronize threads
                // 5. Inner loop: process 4 K elements per iteration (unrolled)
                // 6. Store accumulated result to C

                // Thread and block indices
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let tid_y = ctx.special_reg(crate::ptx::PtxReg::TidY);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);

                // Tile size as a register (needed throughout)
                let tile_size_reg = ctx.mov_u32_imm(tile_size);

                // Global row and column
                let row = ctx.mad_lo_u32(ctaid_y, tile_size_reg, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, tile_size_reg, tid_x);

                // Load parameters (DON'T exit early - all threads must participate in barriers)
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                // Compute predicates for valid row/col (used for predicated loads)
                let row_valid = ctx.setp_lt_u32(row, m_param);
                let col_valid = ctx.setp_lt_u32(col, n_param);

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Initialize accumulator to 0.0
                let acc = ctx.mov_f32_imm(0.0);

                // Tile loop counter
                let tile_idx = ctx.mov_u32_imm(0);
                let n_tiles_reg = ctx.mov_u32_imm(n_tiles);

                ctx.label("tile_loop");

                // Check if done with all tiles
                let tile_done = ctx.setp_ge_u32(tile_idx, n_tiles_reg);
                ctx.branch_if(tile_done, "tile_loop_end");

                // Calculate shared memory address for this thread's position
                let smem_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, tid_x);
                let smem_a_offset = ctx.mul_u32(smem_idx, 4);
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 4);
                let smem_b_offset = ctx.add_u32_reg(smem_b_base, smem_a_offset);

                // Load A tile: A[row, tile_idx * TILE + tid_x] into As[tid_y][tid_x]
                let tile_k_offset = ctx.mul_u32(tile_idx, tile_size);
                let a_col = ctx.add_u32_reg(tile_k_offset, tid_x);
                let a_col_valid = ctx.setp_lt_u32(a_col, k_param);

                // Store 0.0 first (default for out-of-bounds)
                let zero_a = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_a_offset, zero_a);

                ctx.branch_if_not(row_valid, "skip_a_load");
                ctx.branch_if_not(a_col_valid, "skip_a_load");
                let row_offset_a = ctx.mul_wide_u32(row, self.config.k * 4);
                let col_offset_a = ctx.mul_wide_u32(a_col, 4);
                let a_row_base = ctx.add_u64(a_ptr, row_offset_a);
                let a_addr = ctx.add_u64(a_row_base, col_offset_a);
                let a_val = ctx.ld_global_f32(a_addr);
                ctx.st_shared_f32(smem_a_offset, a_val);
                ctx.label("skip_a_load");

                // Load B tile: B[tile_idx * TILE + tid_y, col] into Bs[tid_y][tid_x]
                let b_row = ctx.add_u32_reg(tile_k_offset, tid_y);
                let b_row_valid = ctx.setp_lt_u32(b_row, k_param);

                let zero_b = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_b_offset, zero_b);

                ctx.branch_if_not(b_row_valid, "skip_b_load");
                ctx.branch_if_not(col_valid, "skip_b_load");
                let row_offset_b = ctx.mul_wide_u32(b_row, self.config.n * 4);
                let col_offset_b = ctx.mul_wide_u32(col, 4);
                let b_row_base = ctx.add_u64(b_ptr, row_offset_b);
                let b_addr = ctx.add_u64(b_row_base, col_offset_b);
                let b_val = ctx.ld_global_f32(b_addr);
                ctx.st_shared_f32(smem_b_offset, b_val);
                ctx.label("skip_b_load");

                // Synchronize threads after loading tile
                ctx.bar_sync(0);

                // ========================================
                // 4x UNROLLED INNER LOOP (WAPR-PERF-009)
                // ========================================
                // Process 4 K elements per iteration to reduce loop overhead
                // from 12:1 to ~3:1 instructions per FMA

                let inner_k = ctx.mov_u32_imm(0);
                let unrolled_iters_reg = ctx.mov_u32_imm(unrolled_iters);

                ctx.label("inner_k_loop");

                let inner_done = ctx.setp_ge_u32(inner_k, unrolled_iters_reg);
                ctx.branch_if(inner_done, "inner_k_end");

                // Compute base k index: k_base = inner_k * 4
                let k_base = ctx.mul_u32(inner_k, unroll_factor);

                // === Iteration 0: k = k_base + 0 ===
                let k0 = k_base;
                let as_idx0 = ctx.mad_lo_u32(tid_y, tile_size_reg, k0);
                let as_addr0 = ctx.mul_u32(as_idx0, 4);
                let a_shared0 = ctx.ld_shared_f32(as_addr0);

                let bs_idx0 = ctx.mad_lo_u32(k0, tile_size_reg, tid_x);
                let bs_idx_bytes0 = ctx.mul_u32(bs_idx0, 4);
                let bs_addr0 = ctx.add_u32_reg(smem_b_base, bs_idx_bytes0);
                let b_shared0 = ctx.ld_shared_f32(bs_addr0);

                ctx.fma_f32_inplace(acc, a_shared0, b_shared0);

                // === Iteration 1: k = k_base + 1 ===
                let k1 = ctx.add_u32(k_base, 1);
                let as_idx1 = ctx.mad_lo_u32(tid_y, tile_size_reg, k1);
                let as_addr1 = ctx.mul_u32(as_idx1, 4);
                let a_shared1 = ctx.ld_shared_f32(as_addr1);

                let bs_idx1 = ctx.mad_lo_u32(k1, tile_size_reg, tid_x);
                let bs_idx_bytes1 = ctx.mul_u32(bs_idx1, 4);
                let bs_addr1 = ctx.add_u32_reg(smem_b_base, bs_idx_bytes1);
                let b_shared1 = ctx.ld_shared_f32(bs_addr1);

                ctx.fma_f32_inplace(acc, a_shared1, b_shared1);

                // === Iteration 2: k = k_base + 2 ===
                let k2 = ctx.add_u32(k_base, 2);
                let as_idx2 = ctx.mad_lo_u32(tid_y, tile_size_reg, k2);
                let as_addr2 = ctx.mul_u32(as_idx2, 4);
                let a_shared2 = ctx.ld_shared_f32(as_addr2);

                let bs_idx2 = ctx.mad_lo_u32(k2, tile_size_reg, tid_x);
                let bs_idx_bytes2 = ctx.mul_u32(bs_idx2, 4);
                let bs_addr2 = ctx.add_u32_reg(smem_b_base, bs_idx_bytes2);
                let b_shared2 = ctx.ld_shared_f32(bs_addr2);

                ctx.fma_f32_inplace(acc, a_shared2, b_shared2);

                // === Iteration 3: k = k_base + 3 ===
                let k3 = ctx.add_u32(k_base, 3);
                let as_idx3 = ctx.mad_lo_u32(tid_y, tile_size_reg, k3);
                let as_addr3 = ctx.mul_u32(as_idx3, 4);
                let a_shared3 = ctx.ld_shared_f32(as_addr3);

                let bs_idx3 = ctx.mad_lo_u32(k3, tile_size_reg, tid_x);
                let bs_idx_bytes3 = ctx.mul_u32(bs_idx3, 4);
                let bs_addr3 = ctx.add_u32_reg(smem_b_base, bs_idx_bytes3);
                let b_shared3 = ctx.ld_shared_f32(bs_addr3);

                ctx.fma_f32_inplace(acc, a_shared3, b_shared3);

                // inner_k++ (increments by 1, actual k increments by 4)
                ctx.add_u32_inplace(inner_k, 1);
                ctx.branch("inner_k_loop");

                ctx.label("inner_k_end");

                // Synchronize before loading next tile
                ctx.bar_sync(1);

                // tile_idx++
                ctx.add_u32_inplace(tile_idx, 1);
                ctx.branch("tile_loop");

                ctx.label("tile_loop_end");

                // Bounds check: only valid threads store to C
                ctx.branch_if_not(row_valid, "exit");
                ctx.branch_if_not(col_valid, "exit");

                // Store result: C[row, col] = c_ptr + row * N + col
                let c_row_offset = ctx.mul_wide_u32(row, self.config.n * 4);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_row_base = ctx.add_u64(c_ptr, c_row_offset);
                let c_addr = ctx.add_u64(c_row_base, c_col_offset);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }

    #[allow(clippy::too_many_lines)]
    fn build_tensor_core(&self) -> PtxKernel {
        // Tensor Core GEMM using 16x16 tiles
        // This kernel uses 16 threads per block (one thread per output row)
        // Each thread computes one row of the 16x16 output tile
        //
        // Launch config: grid_2d((m+15)/16, (n+15)/16, 16, 1)

        // Shared memory for two 16x16 tiles (A and B) in fp32
        // A: 16 * 16 * 4 bytes = 1024 bytes
        // B: 16 * 16 * 4 bytes = 1024 bytes
        // Total: 2048 bytes
        let tile_size = 16_u32;
        let smem_size = tile_size * tile_size * 4 * 2; // Two fp32 tiles
        let n_k_tiles = (self.config.k + tile_size - 1) / tile_size;

        PtxKernel::new("gemm_tensor_core")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Algorithm:
                // 1. Each block handles one 16x16 output tile of C
                // 2. Each thread handles one row (16 outputs)
                // 3. Loop over K dimension in steps of 16
                // 4. Load A and B tiles into shared memory
                // 5. Compute partial products and accumulate
                // 6. Store result to global memory

                // Thread and block IDs
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);

                // Calculate block's output tile position
                let tile_size_reg = ctx.mov_u32_imm(tile_size);
                let tile_row = ctx.mul_u32(ctaid_y, tile_size);
                let tile_col = ctx.mul_u32(ctaid_x, tile_size);

                // PARITY-114 FIX: Load parameters but DON'T exit early
                // All threads must participate in barriers
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                // Compute predicates for valid output (used for predicated stores)
                // Note: Per-thread row validity is computed below with my_row_valid
                let tile_col_valid = ctx.setp_lt_u32(tile_col, n_param);

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Calculate my output row (within tile)
                // Thread tid_x handles row tid_x of the output tile
                let my_row = ctx.add_u32_reg(tile_row, tid_x);

                // PARITY-114 FIX: Compute predicate but don't exit
                let my_row_valid = ctx.setp_lt_u32(my_row, m_param);

                // Initialize 16 accumulators (one per output column)
                let acc0 = ctx.mov_f32_imm(0.0);
                let acc1 = ctx.mov_f32_imm(0.0);
                let acc2 = ctx.mov_f32_imm(0.0);
                let acc3 = ctx.mov_f32_imm(0.0);
                let acc4 = ctx.mov_f32_imm(0.0);
                let acc5 = ctx.mov_f32_imm(0.0);
                let acc6 = ctx.mov_f32_imm(0.0);
                let acc7 = ctx.mov_f32_imm(0.0);
                let acc8 = ctx.mov_f32_imm(0.0);
                let acc9 = ctx.mov_f32_imm(0.0);
                let acc10 = ctx.mov_f32_imm(0.0);
                let acc11 = ctx.mov_f32_imm(0.0);
                let acc12 = ctx.mov_f32_imm(0.0);
                let acc13 = ctx.mov_f32_imm(0.0);
                let acc14 = ctx.mov_f32_imm(0.0);
                let acc15 = ctx.mov_f32_imm(0.0);

                // Loop over K tiles
                let k_tile_idx = ctx.mov_u32_imm(0);
                let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 4);

                ctx.label("k_tile_loop");

                let k_done = ctx.setp_ge_u32(k_tile_idx, n_k_tiles_reg);
                ctx.branch_if(k_done, "k_tile_end");

                // Calculate K offset for this tile
                let k_offset = ctx.mul_u32(k_tile_idx, tile_size);

                // === Load A tile row (this thread's row, 16 elements) ===
                // A[my_row, k_offset:k_offset+16] -> shared[tid_x, 0:16]
                // PARITY-114 FIX: All threads load, but invalid threads load 0.0
                let a_row_offset = ctx.mul_wide_u32(my_row, self.config.k * 4);
                let a_base = ctx.add_u64(a_ptr, a_row_offset);

                // Load 16 elements from A row into shared memory
                // Each thread loads its row's 16 elements
                let inner_k = ctx.mov_u32_imm(0);

                ctx.label("load_a_loop");
                let a_load_done = ctx.setp_ge_u32(inner_k, tile_size_reg);
                ctx.branch_if(a_load_done, "load_a_end");

                // PARITY-114 FIX: Store 0.0 first (default for out-of-bounds)
                let a_smem_idx = ctx.mad_lo_u32(tid_x, tile_size_reg, inner_k);
                let a_smem_offset = ctx.mul_u32(a_smem_idx, 4);
                let zero_a = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(a_smem_offset, zero_a);

                // Check bounds: my_row < m AND k_idx < k
                let k_idx = ctx.add_u32_reg(k_offset, inner_k);
                let k_valid = ctx.setp_lt_u32(k_idx, k_param);

                // If out of bounds, skip load
                ctx.branch_if_not(my_row_valid, "skip_a_load");
                ctx.branch_if_not(k_valid, "skip_a_load");

                let a_elem_offset = ctx.mul_wide_u32(k_idx, 4);
                let a_addr = ctx.add_u64(a_base, a_elem_offset);
                let a_val = ctx.ld_global_f32(a_addr);
                ctx.st_shared_f32(a_smem_offset, a_val);

                ctx.label("skip_a_load");
                ctx.add_u32_inplace(inner_k, 1);
                ctx.branch("load_a_loop");
                ctx.label("load_a_end");

                // === Load B tile column (use cooperative loading) ===
                // Thread tid_x loads column tid_x of B tile
                // B[k_offset:k_offset+16, tile_col + tid_x] -> shared[B_base + 0:16, tid_x]
                // PARITY-114 FIX: All threads load, but invalid threads load 0.0
                let b_col = ctx.add_u32_reg(tile_col, tid_x);
                let b_col_valid = ctx.setp_lt_u32(b_col, n_param);
                let inner_k2 = ctx.mov_u32_imm(0);

                ctx.label("load_b_loop");
                let b_load_done = ctx.setp_ge_u32(inner_k2, tile_size_reg);
                ctx.branch_if(b_load_done, "load_b_end");

                // PARITY-114 FIX: Store 0.0 first (default for out-of-bounds)
                let b_smem_idx = ctx.mad_lo_u32(inner_k2, tile_size_reg, tid_x);
                let b_smem_offset = ctx.mul_u32(b_smem_idx, 4);
                let b_smem_addr = ctx.add_u32_reg(smem_b_base, b_smem_offset);
                let zero_b = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(b_smem_addr, zero_b);

                // Check bounds: k_idx2 < k AND b_col < n
                let k_idx2 = ctx.add_u32_reg(k_offset, inner_k2);
                let k_valid2 = ctx.setp_lt_u32(k_idx2, k_param);

                // If out of bounds, skip load
                ctx.branch_if_not(k_valid2, "skip_b_load");
                ctx.branch_if_not(b_col_valid, "skip_b_load");

                let b_row_offset = ctx.mul_wide_u32(k_idx2, self.config.n * 4);
                let b_col_offset = ctx.mul_wide_u32(b_col, 4);
                let b_base = ctx.add_u64(b_ptr, b_row_offset);
                let b_addr = ctx.add_u64(b_base, b_col_offset);
                let b_val = ctx.ld_global_f32(b_addr);
                ctx.st_shared_f32(b_smem_addr, b_val);

                ctx.label("skip_b_load");
                ctx.add_u32_inplace(inner_k2, 1);
                ctx.branch("load_b_loop");
                ctx.label("load_b_end");

                // Synchronize after loading
                ctx.bar_sync(0);

                // === Compute: for each k in 0..16, acc[j] += A_shared[tid_x,k] * B_shared[k,j] ===
                let compute_k = ctx.mov_u32_imm(0);

                ctx.label("compute_loop");
                let compute_done = ctx.setp_ge_u32(compute_k, tile_size_reg);
                ctx.branch_if(compute_done, "compute_end");

                // Load A_shared[tid_x, compute_k]
                let a_compute_idx = ctx.mad_lo_u32(tid_x, tile_size_reg, compute_k);
                let a_compute_offset = ctx.mul_u32(a_compute_idx, 4);
                let a_compute_val = ctx.ld_shared_f32(a_compute_offset);

                // Load B_shared[compute_k, 0..15] and accumulate
                // Unrolled for all 16 columns
                // B is stored row-major: B[compute_k, col] = smem_b_base + (compute_k * 16 + col) * 4
                let b0_idx = ctx.mul_u32_reg(compute_k, tile_size_reg);
                let b0_offset = ctx.mul_u32(b0_idx, 4);
                let b0_addr = ctx.add_u32_reg(smem_b_base, b0_offset);
                let b0_val = ctx.ld_shared_f32(b0_addr);
                ctx.fma_f32_inplace(acc0, a_compute_val, b0_val);

                let b1_idx = ctx.add_u32(b0_idx, 1);
                let b1_offset = ctx.mul_u32(b1_idx, 4);
                let b1_addr = ctx.add_u32_reg(smem_b_base, b1_offset);
                let b1_val = ctx.ld_shared_f32(b1_addr);
                ctx.fma_f32_inplace(acc1, a_compute_val, b1_val);

                let b2_idx = ctx.add_u32(b0_idx, 2);
                let b2_offset = ctx.mul_u32(b2_idx, 4);
                let b2_addr = ctx.add_u32_reg(smem_b_base, b2_offset);
                let b2_val = ctx.ld_shared_f32(b2_addr);
                ctx.fma_f32_inplace(acc2, a_compute_val, b2_val);

                let b3_idx = ctx.add_u32(b0_idx, 3);
                let b3_offset = ctx.mul_u32(b3_idx, 4);
                let b3_addr = ctx.add_u32_reg(smem_b_base, b3_offset);
                let b3_val = ctx.ld_shared_f32(b3_addr);
                ctx.fma_f32_inplace(acc3, a_compute_val, b3_val);

                let b4_idx = ctx.add_u32(b0_idx, 4);
                let b4_offset = ctx.mul_u32(b4_idx, 4);
                let b4_addr = ctx.add_u32_reg(smem_b_base, b4_offset);
                let b4_val = ctx.ld_shared_f32(b4_addr);
                ctx.fma_f32_inplace(acc4, a_compute_val, b4_val);

                let b5_idx = ctx.add_u32(b0_idx, 5);
                let b5_offset = ctx.mul_u32(b5_idx, 4);
                let b5_addr = ctx.add_u32_reg(smem_b_base, b5_offset);
                let b5_val = ctx.ld_shared_f32(b5_addr);
                ctx.fma_f32_inplace(acc5, a_compute_val, b5_val);

                let b6_idx = ctx.add_u32(b0_idx, 6);
                let b6_offset = ctx.mul_u32(b6_idx, 4);
                let b6_addr = ctx.add_u32_reg(smem_b_base, b6_offset);
                let b6_val = ctx.ld_shared_f32(b6_addr);
                ctx.fma_f32_inplace(acc6, a_compute_val, b6_val);

                let b7_idx = ctx.add_u32(b0_idx, 7);
                let b7_offset = ctx.mul_u32(b7_idx, 4);
                let b7_addr = ctx.add_u32_reg(smem_b_base, b7_offset);
                let b7_val = ctx.ld_shared_f32(b7_addr);
                ctx.fma_f32_inplace(acc7, a_compute_val, b7_val);

                let b8_idx = ctx.add_u32(b0_idx, 8);
                let b8_offset = ctx.mul_u32(b8_idx, 4);
                let b8_addr = ctx.add_u32_reg(smem_b_base, b8_offset);
                let b8_val = ctx.ld_shared_f32(b8_addr);
                ctx.fma_f32_inplace(acc8, a_compute_val, b8_val);

                let b9_idx = ctx.add_u32(b0_idx, 9);
                let b9_offset = ctx.mul_u32(b9_idx, 4);
                let b9_addr = ctx.add_u32_reg(smem_b_base, b9_offset);
                let b9_val = ctx.ld_shared_f32(b9_addr);
                ctx.fma_f32_inplace(acc9, a_compute_val, b9_val);

                let b10_idx = ctx.add_u32(b0_idx, 10);
                let b10_offset = ctx.mul_u32(b10_idx, 4);
                let b10_addr = ctx.add_u32_reg(smem_b_base, b10_offset);
                let b10_val = ctx.ld_shared_f32(b10_addr);
                ctx.fma_f32_inplace(acc10, a_compute_val, b10_val);

                let b11_idx = ctx.add_u32(b0_idx, 11);
                let b11_offset = ctx.mul_u32(b11_idx, 4);
                let b11_addr = ctx.add_u32_reg(smem_b_base, b11_offset);
                let b11_val = ctx.ld_shared_f32(b11_addr);
                ctx.fma_f32_inplace(acc11, a_compute_val, b11_val);

                let b12_idx = ctx.add_u32(b0_idx, 12);
                let b12_offset = ctx.mul_u32(b12_idx, 4);
                let b12_addr = ctx.add_u32_reg(smem_b_base, b12_offset);
                let b12_val = ctx.ld_shared_f32(b12_addr);
                ctx.fma_f32_inplace(acc12, a_compute_val, b12_val);

                let b13_idx = ctx.add_u32(b0_idx, 13);
                let b13_offset = ctx.mul_u32(b13_idx, 4);
                let b13_addr = ctx.add_u32_reg(smem_b_base, b13_offset);
                let b13_val = ctx.ld_shared_f32(b13_addr);
                ctx.fma_f32_inplace(acc13, a_compute_val, b13_val);

                let b14_idx = ctx.add_u32(b0_idx, 14);
                let b14_offset = ctx.mul_u32(b14_idx, 4);
                let b14_addr = ctx.add_u32_reg(smem_b_base, b14_offset);
                let b14_val = ctx.ld_shared_f32(b14_addr);
                ctx.fma_f32_inplace(acc14, a_compute_val, b14_val);

                let b15_idx = ctx.add_u32(b0_idx, 15);
                let b15_offset = ctx.mul_u32(b15_idx, 4);
                let b15_addr = ctx.add_u32_reg(smem_b_base, b15_offset);
                let b15_val = ctx.ld_shared_f32(b15_addr);
                ctx.fma_f32_inplace(acc15, a_compute_val, b15_val);

                ctx.add_u32_inplace(compute_k, 1);
                ctx.branch("compute_loop");
                ctx.label("compute_end");

                // Synchronize before next K tile
                ctx.bar_sync(1);

                ctx.add_u32_inplace(k_tile_idx, 1);
                ctx.branch("k_tile_loop");
                ctx.label("k_tile_end");

                // PARITY-114 FIX: Bounds check HERE (after all threads finished barriers)
                // Only threads with valid output coordinates store to C
                ctx.branch_if_not(my_row_valid, "exit");
                ctx.branch_if_not(tile_col_valid, "exit");

                // === Store results: C[my_row, tile_col + 0..15] ===
                let c_row_offset = ctx.mul_wide_u32(my_row, self.config.n * 4);
                let c_base = ctx.add_u64(c_ptr, c_row_offset);

                // Store all 16 accumulators
                // C[my_row, tile_col + i] = acc_i
                let c0_col = ctx.add_u32(tile_col, 0);
                let c0_offset = ctx.mul_wide_u32(c0_col, 4);
                let c0_addr = ctx.add_u64(c_base, c0_offset);
                ctx.st_global_f32(c0_addr, acc0);

                let c1_col = ctx.add_u32(tile_col, 1);
                let c1_offset = ctx.mul_wide_u32(c1_col, 4);
                let c1_addr = ctx.add_u64(c_base, c1_offset);
                ctx.st_global_f32(c1_addr, acc1);

                let c2_col = ctx.add_u32(tile_col, 2);
                let c2_offset = ctx.mul_wide_u32(c2_col, 4);
                let c2_addr = ctx.add_u64(c_base, c2_offset);
                ctx.st_global_f32(c2_addr, acc2);

                let c3_col = ctx.add_u32(tile_col, 3);
                let c3_offset = ctx.mul_wide_u32(c3_col, 4);
                let c3_addr = ctx.add_u64(c_base, c3_offset);
                ctx.st_global_f32(c3_addr, acc3);

                let c4_col = ctx.add_u32(tile_col, 4);
                let c4_offset = ctx.mul_wide_u32(c4_col, 4);
                let c4_addr = ctx.add_u64(c_base, c4_offset);
                ctx.st_global_f32(c4_addr, acc4);

                let c5_col = ctx.add_u32(tile_col, 5);
                let c5_offset = ctx.mul_wide_u32(c5_col, 4);
                let c5_addr = ctx.add_u64(c_base, c5_offset);
                ctx.st_global_f32(c5_addr, acc5);

                let c6_col = ctx.add_u32(tile_col, 6);
                let c6_offset = ctx.mul_wide_u32(c6_col, 4);
                let c6_addr = ctx.add_u64(c_base, c6_offset);
                ctx.st_global_f32(c6_addr, acc6);

                let c7_col = ctx.add_u32(tile_col, 7);
                let c7_offset = ctx.mul_wide_u32(c7_col, 4);
                let c7_addr = ctx.add_u64(c_base, c7_offset);
                ctx.st_global_f32(c7_addr, acc7);

                let c8_col = ctx.add_u32(tile_col, 8);
                let c8_offset = ctx.mul_wide_u32(c8_col, 4);
                let c8_addr = ctx.add_u64(c_base, c8_offset);
                ctx.st_global_f32(c8_addr, acc8);

                let c9_col = ctx.add_u32(tile_col, 9);
                let c9_offset = ctx.mul_wide_u32(c9_col, 4);
                let c9_addr = ctx.add_u64(c_base, c9_offset);
                ctx.st_global_f32(c9_addr, acc9);

                let c10_col = ctx.add_u32(tile_col, 10);
                let c10_offset = ctx.mul_wide_u32(c10_col, 4);
                let c10_addr = ctx.add_u64(c_base, c10_offset);
                ctx.st_global_f32(c10_addr, acc10);

                let c11_col = ctx.add_u32(tile_col, 11);
                let c11_offset = ctx.mul_wide_u32(c11_col, 4);
                let c11_addr = ctx.add_u64(c_base, c11_offset);
                ctx.st_global_f32(c11_addr, acc11);

                let c12_col = ctx.add_u32(tile_col, 12);
                let c12_offset = ctx.mul_wide_u32(c12_col, 4);
                let c12_addr = ctx.add_u64(c_base, c12_offset);
                ctx.st_global_f32(c12_addr, acc12);

                let c13_col = ctx.add_u32(tile_col, 13);
                let c13_offset = ctx.mul_wide_u32(c13_col, 4);
                let c13_addr = ctx.add_u64(c_base, c13_offset);
                ctx.st_global_f32(c13_addr, acc13);

                let c14_col = ctx.add_u32(tile_col, 14);
                let c14_offset = ctx.mul_wide_u32(c14_col, 4);
                let c14_addr = ctx.add_u64(c_base, c14_offset);
                ctx.st_global_f32(c14_addr, acc14);

                let c15_col = ctx.add_u32(tile_col, 15);
                let c15_offset = ctx.mul_wide_u32(c15_col, 4);
                let c15_addr = ctx.add_u64(c_base, c15_offset);
                ctx.st_global_f32(c15_addr, acc15);

                ctx.label("exit");
                ctx.ret();
            })
    }

    /// Build WMMA FP16 GEMM kernel using true Tensor Core PTX intrinsics
    /// This kernel uses wmma.load, wmma.mma, wmma.store for hardware Tensor Core acceleration
    /// Launch config: grid_2d((m+15)/16, (n+15)/16, 32, 1) - one warp per 16x16 output tile
    #[allow(clippy::too_many_lines)]
    fn build_wmma_fp16(&self) -> PtxKernel {
        use crate::ptx::WmmaLayout;

        // WMMA 16x16x16 tile configuration
        // Shared memory for A and B tiles in FP16 format
        // A tile: 16 * 16 * 2 bytes = 512 bytes (FP16)
        // B tile: 16 * 16 * 2 bytes = 512 bytes (FP16)
        // Total: 1024 bytes
        let tile_size = 16_u32;
        let smem_size = tile_size * tile_size * 2 * 2; // Two FP16 tiles
        let n_k_tiles = (self.config.k + tile_size - 1) / tile_size;

        PtxKernel::new("gemm_wmma_fp16")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // WMMA operates at warp level (32 threads cooperatively)
                // Each warp handles one 16x16 output tile
                //
                // Thread organization:
                // - ctaid.x, ctaid.y: which 16x16 output tile
                // - tid.x (0-31): lane within warp
                //
                // Algorithm:
                // 1. Each warp processes one output tile C[tile_row:+16, tile_col:+16]
                // 2. Loop over K in steps of 16
                // 3. Load A and B tiles to shared memory (cooperative, convert FP32→FP16)
                // 4. Use WMMA intrinsics to compute 16x16x16 matrix multiply
                // 5. Accumulate in FP32
                // 6. Store result to global memory

                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);

                // Calculate output tile position
                let tile_size_reg = ctx.mov_u32_imm(tile_size);
                let tile_row = ctx.mul_u32(ctaid_y, tile_size);
                let tile_col = ctx.mul_u32(ctaid_x, tile_size);

                // PARITY-114 FIX: Load parameters but DON'T exit early
                // All threads must participate in barriers (WMMA requires full warp)
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                // Compute predicates for valid tile (used for predicated stores)
                let tile_row_valid = ctx.setp_lt_u32(tile_row, m_param);
                let tile_col_valid = ctx.setp_lt_u32(tile_col, n_param);

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Shared memory base addresses
                let smem_a_base = ctx.mov_u32_imm(0);
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 2); // After A tile (FP16)

                // Initialize accumulator fragments (8 FP32 registers per thread for 16x16 output)
                // WAPR-PERF-010: Use wmma_init_c_zero to avoid loading from invalid address 0
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
                // Each of 32 threads loads multiple elements
                // Total elements: 16 * 16 = 256, each thread loads 8 elements
                // Thread i loads elements i*8 to i*8+7
                let elements_per_thread = ctx.mov_u32_imm(8);
                let my_start = ctx.mul_u32_reg(tid_x, elements_per_thread);

                // Load 8 elements from A
                // PARITY-114 FIX: All threads load, but invalid threads load 0.0
                let load_idx = ctx.mov_u32_imm(0);
                ctx.label("load_a_loop");
                let load_done = ctx.setp_ge_u32(load_idx, elements_per_thread);
                ctx.branch_if(load_done, "load_a_end");

                let elem_idx = ctx.add_u32_reg(my_start, load_idx);
                // elem_idx = row_in_tile * 16 + col_in_tile
                let row_in_tile = ctx.div_u32(elem_idx, 16);
                let col_in_tile = ctx.rem_u32(elem_idx, 16);

                // PARITY-114 FIX: Store 0 first (default for out-of-bounds)
                let smem_a_offset = ctx.mul_u32(elem_idx, 2); // FP16 is 2 bytes
                let smem_a_addr = ctx.add_u32_reg(smem_a_base, smem_a_offset);
                let zero_f32 = ctx.mov_f32_imm(0.0);
                let zero_f16 = ctx.cvt_f16_f32(zero_f32);
                ctx.st_shared_f16(smem_a_addr, zero_f16);

                // Check bounds: a_row < m AND a_col < k
                let a_row = ctx.add_u32_reg(tile_row, row_in_tile);
                let a_col = ctx.add_u32_reg(k_offset, col_in_tile);
                let a_row_valid = ctx.setp_lt_u32(a_row, m_param);
                let a_col_valid = ctx.setp_lt_u32(a_col, k_param);

                // If out of bounds, skip load
                ctx.branch_if_not(a_row_valid, "skip_wmma_a_load");
                ctx.branch_if_not(a_col_valid, "skip_wmma_a_load");

                // Global A address: A[tile_row + row_in_tile, k_offset + col_in_tile]
                let k_reg = ctx.mov_u32_imm(self.config.k);
                let a_idx = ctx.mad_lo_u32(a_row, k_reg, a_col);
                let a_byte_offset = ctx.mul_wide_u32(a_idx, 4); // FP32 is 4 bytes
                let a_addr = ctx.add_u64(a_ptr, a_byte_offset);

                // Load FP32, convert to FP16, store to shared
                let a_val_f32 = ctx.ld_global_f32(a_addr);
                let a_val_f16 = ctx.cvt_f16_f32(a_val_f32);
                ctx.st_shared_f16(smem_a_addr, a_val_f16);

                ctx.label("skip_wmma_a_load");
                ctx.add_u32_inplace(load_idx, 1);
                ctx.branch("load_a_loop");
                ctx.label("load_a_end");

                // === Load B tile to shared memory ===
                // PARITY-114 FIX: All threads load, but invalid threads load 0.0
                let load_idx_b = ctx.mov_u32_imm(0);
                ctx.label("load_b_loop");
                let load_b_done = ctx.setp_ge_u32(load_idx_b, elements_per_thread);
                ctx.branch_if(load_b_done, "load_b_end");

                let elem_idx_b = ctx.add_u32_reg(my_start, load_idx_b);
                let row_in_tile_b = ctx.div_u32(elem_idx_b, 16);
                let col_in_tile_b = ctx.rem_u32(elem_idx_b, 16);

                // PARITY-114 FIX: Store 0 first (default for out-of-bounds)
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

                // If out of bounds, skip load
                ctx.branch_if_not(b_row_valid, "skip_wmma_b_load");
                ctx.branch_if_not(b_col_valid, "skip_wmma_b_load");

                // Global B address: B[k_offset + row_in_tile, tile_col + col_in_tile]
                let n_reg = ctx.mov_u32_imm(self.config.n);
                let b_idx = ctx.mad_lo_u32(b_row, n_reg, b_col);
                let b_byte_offset = ctx.mul_wide_u32(b_idx, 4);
                let b_addr = ctx.add_u64(b_ptr, b_byte_offset);

                let b_val_f32 = ctx.ld_global_f32(b_addr);
                let b_val_f16 = ctx.cvt_f16_f32(b_val_f32);
                ctx.st_shared_f16(smem_b_addr, b_val_f16);

                ctx.label("skip_wmma_b_load");
                ctx.add_u32_inplace(load_idx_b, 1);
                ctx.branch("load_b_loop");
                ctx.label("load_b_end");

                // Synchronize before WMMA
                ctx.bar_sync(0);

                // === WMMA matrix multiply ===
                // WAPR-PERF-010 FIX: Use cvta.shared.u64 to get generic pointer
                // WMMA loads require generic addresses, not raw offsets
                let smem_generic_base = ctx.shared_base_addr();

                // Load A fragment from shared memory (A is at offset 0)
                let frag_a = ctx.wmma_load_a_f16(smem_generic_base, 16, WmmaLayout::RowMajor);

                // Load B fragment from shared memory (B is at offset smem_b_base)
                // WAPR-PERF-014 FIX: B is stored row-major in shared memory, so use RowMajor
                // B_shared[k, n] is at offset k*16 + n (row_in_tile * 16 + col_in_tile)
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

                // Synchronize before next K tile
                ctx.bar_sync(1);

                ctx.add_u32_inplace(k_tile_idx, 1);
                ctx.branch("k_tile_loop");
                ctx.label("k_tile_end");

                // PARITY-114 FIX: Bounds check HERE (after all threads finished barriers)
                // Only warps with valid output tiles store to C
                ctx.branch_if_not(tile_row_valid, "exit");
                ctx.branch_if_not(tile_col_valid, "exit");

                // === Store result to global memory ===
                // C[tile_row:+16, tile_col:+16]
                let c_row_offset = ctx.mul_wide_u32(tile_row, self.config.n * 4);
                let c_col_offset = ctx.mul_wide_u32(tile_col, 4);
                let c_tile_base = ctx.add_u64(c_ptr, c_row_offset);
                let c_addr = ctx.add_u64(c_tile_base, c_col_offset);

                ctx.wmma_store_d_f32(c_addr, &frag_c, self.config.n, WmmaLayout::RowMajor);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

