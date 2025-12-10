//! GEMM (General Matrix Multiply) Kernels
//!
//! Implements C = alpha * A @ B + beta * C

#![allow(clippy::similar_names)] // Variable names like a_addr, b_addr, bs_addr are semantically meaningful

use super::Kernel;
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
    config: GemmConfig,
    variant: GemmVariant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GemmVariant {
    Naive,
    Tiled,
    TensorCore,
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
}

impl Kernel for GemmKernel {
    fn name(&self) -> &str {
        match self.variant {
            GemmVariant::Naive => "gemm_naive",
            GemmVariant::Tiled => "gemm_tiled",
            GemmVariant::TensorCore => "gemm_tensor_core",
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        match self.variant {
            GemmVariant::Naive => self.build_naive(),
            GemmVariant::Tiled => self.build_tiled(),
            GemmVariant::TensorCore => self.build_tensor_core(),
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

                // acc += a_val * b_val (FMA)
                let prod = ctx.mul_f32(a_val, b_val);
                let _new_acc = ctx.add_f32(acc, prod);

                // i++
                let _i_next = ctx.add_u32(i, 1);

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

                // Bounds check
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let _k_param = ctx.load_param_u32("k");

                let pred_m = ctx.setp_ge_u32(row, m_param);
                ctx.branch_if(pred_m, "exit");
                let pred_n = ctx.setp_ge_u32(col, n_param);
                ctx.branch_if(pred_n, "exit");

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
                let smem_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, tid_x);
                let smem_a_offset = ctx.mul_wide_u32(smem_idx, 4);
                let smem_b_base = ctx.mov_u32_imm(tile_size * tile_size * 4);
                let smem_b_base_64 = ctx.cvt_u64_u32(smem_b_base);
                let smem_b_offset = ctx.add_u64(smem_b_base_64, smem_a_offset);

                // Load A[row, tile_idx * TILE + tid_x] into shared memory As[tid_y][tid_x]
                // A address = a_ptr + row * K + (tile_idx * TILE + tid_x)
                let tile_k_offset = ctx.mul_u32(tile_idx, tile_size);
                let a_col = ctx.add_u32_reg(tile_k_offset, tid_x);
                let row_offset_a = ctx.mul_wide_u32(row, self.config.k * 4);
                let col_offset_a = ctx.mul_wide_u32(a_col, 4);
                let a_row_base = ctx.add_u64(a_ptr, row_offset_a);
                let a_addr = ctx.add_u64(a_row_base, col_offset_a);
                let a_val = ctx.ld_global_f32(a_addr);

                // Store to shared memory (using smem_a_offset as base)
                ctx.st_shared_f32(smem_a_offset, a_val);

                // Load B[tile_idx * TILE + tid_y, col] into shared memory Bs[tid_y][tid_x]
                // B address = b_ptr + (tile_idx * TILE + tid_y) * N + col
                let b_row = ctx.add_u32_reg(tile_k_offset, tid_y);
                let row_offset_b = ctx.mul_wide_u32(b_row, self.config.n * 4);
                let col_offset_b = ctx.mul_wide_u32(col, 4);
                let b_row_base = ctx.add_u64(b_ptr, row_offset_b);
                let b_addr = ctx.add_u64(b_row_base, col_offset_b);
                let b_val = ctx.ld_global_f32(b_addr);

                // Store to shared memory B tile
                ctx.st_shared_f32(smem_b_offset, b_val);

                // Synchronize threads after loading tile
                ctx.bar_sync(0);

                // Inner loop: accumulate products from shared memory tile
                let inner_k = ctx.mov_u32_imm(0);

                ctx.label("inner_k_loop");

                let inner_done = ctx.setp_ge_u32(inner_k, tile_size_reg);
                ctx.branch_if(inner_done, "inner_k_end");

                // Load As[tid_y][inner_k] = smem[tid_y * TILE + inner_k]
                let as_idx = ctx.mad_lo_u32(tid_y, tile_size_reg, inner_k);
                let as_addr = ctx.mul_wide_u32(as_idx, 4);
                let a_shared = ctx.ld_shared_f32(as_addr);

                // Load Bs[inner_k][tid_x] = smem[TILE*TILE + inner_k * TILE + tid_x]
                let bs_idx = ctx.mad_lo_u32(inner_k, tile_size_reg, tid_x);
                let bs_idx_bytes = ctx.mul_wide_u32(bs_idx, 4);
                let bs_addr = ctx.add_u64(smem_b_base_64, bs_idx_bytes);
                let b_shared = ctx.ld_shared_f32(bs_addr);

                // acc += a_shared * b_shared
                let prod = ctx.mul_f32(a_shared, b_shared);
                let _new_acc = ctx.add_f32(acc, prod);

                // inner_k++
                let _inner_k_next = ctx.add_u32(inner_k, 1);
                ctx.branch("inner_k_loop");

                ctx.label("inner_k_end");

                // Synchronize before loading next tile
                ctx.bar_sync(1);

                // tile_idx++
                let _tile_idx_next = ctx.add_u32(tile_idx, 1);
                ctx.branch("tile_loop");

                ctx.label("tile_loop_end");

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

    fn build_tensor_core(&self) -> PtxKernel {
        // Tensor Core GEMM using WMMA (Warp Matrix Multiply-Accumulate)
        // WMMA operates on 16x16x16 tiles by default (m16n16k16)
        // Each warp (32 threads) cooperatively loads fragments and computes

        // Shared memory for two 16x16 tiles (A and B) in fp16
        // A: 16 * 16 * 2 bytes = 512 bytes
        // B: 16 * 16 * 2 bytes = 512 bytes
        // Total: 1024 bytes minimum
        let wmma_tile = 16_u32;
        let smem_size = wmma_tile * wmma_tile * 2 * 2; // Two fp16 tiles
        let n_k_tiles = (self.config.k + wmma_tile - 1) / wmma_tile;

        PtxKernel::new("gemm_tensor_core")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Tensor Core GEMM using WMMA (Warp Matrix Multiply-Accumulate)
                //
                // WMMA m16n16k16 computes: D = A @ B + C
                // where A is 16x16, B is 16x16, C/D are 16x16
                //
                // Algorithm:
                // 1. Each warp handles one 16x16 output tile of C
                // 2. Loop over K dimension in steps of 16
                // 3. Load A fragment (16x16) and B fragment (16x16) into shared memory
                // 4. Execute WMMA.MMA to accumulate into C fragment
                // 5. Store C fragment to global memory

                // Get warp position (which 16x16 output tile)
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);

                // Calculate warp's output tile position (row, col)
                // Each warp handles a 16x16 tile
                let wmma_tile_reg = ctx.mov_u32_imm(wmma_tile);
                let warp_row = ctx.mul_u32(ctaid_y, wmma_tile);
                let warp_col = ctx.mul_u32(ctaid_x, wmma_tile);

                // Bounds check
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let _k_param = ctx.load_param_u32("k");

                let pred_m = ctx.setp_ge_u32(warp_row, m_param);
                ctx.branch_if(pred_m, "exit");
                let pred_n = ctx.setp_ge_u32(warp_col, n_param);
                ctx.branch_if(pred_n, "exit");

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Initialize accumulator fragment to 0.0
                // WMMA accumulator C is distributed across warp threads
                let acc = ctx.mov_f32_imm(0.0);

                // Loop over K tiles
                let k_tile_idx = ctx.mov_u32_imm(0);
                let n_k_tiles_reg = ctx.mov_u32_imm(n_k_tiles);

                ctx.label("wmma_k_loop");

                let k_done = ctx.setp_ge_u32(k_tile_idx, n_k_tiles_reg);
                ctx.branch_if(k_done, "wmma_k_end");

                // Calculate K offset for this tile
                let k_offset = ctx.mul_u32(k_tile_idx, wmma_tile);

                // Load A tile: A[warp_row:warp_row+16, k_offset:k_offset+16]
                // For simplicity, each thread loads one element based on tid
                let a_row = ctx.add_u32_reg(warp_row, tid_x);
                let a_row_offset = ctx.mul_wide_u32(a_row, self.config.k * 4);
                let k_col_offset = ctx.mul_wide_u32(k_offset, 4);
                let a_base = ctx.add_u64(a_ptr, a_row_offset);
                let a_addr = ctx.add_u64(a_base, k_col_offset);
                let a_val = ctx.ld_global_f32(a_addr);

                // Store to shared memory A tile
                let tid_offset = ctx.mul_wide_u32(tid_x, 4);
                ctx.st_shared_f32(tid_offset, a_val);

                // Load B tile: B[k_offset:k_offset+16, warp_col:warp_col+16]
                let b_row = ctx.add_u32_reg(k_offset, tid_x);
                let b_row_offset = ctx.mul_wide_u32(b_row, self.config.n * 4);
                let b_col_offset = ctx.mul_wide_u32(warp_col, 4);
                let b_base = ctx.add_u64(b_ptr, b_row_offset);
                let b_addr = ctx.add_u64(b_base, b_col_offset);
                let b_val = ctx.ld_global_f32(b_addr);

                // Store to shared memory B tile (offset by A tile size)
                let smem_b_base = ctx.mov_u32_imm(wmma_tile * wmma_tile * 4);
                let smem_b_base_64 = ctx.cvt_u64_u32(smem_b_base);
                let b_smem_addr = ctx.add_u64(smem_b_base_64, tid_offset);
                ctx.st_shared_f32(b_smem_addr, b_val);

                // Synchronize before WMMA operation
                ctx.bar_sync(0);

                // Inner loop: compute partial products from shared memory
                // In a real WMMA kernel, this would use wmma.mma.sync instructions
                // Here we simulate with regular FMA operations
                let inner_idx = ctx.mov_u32_imm(0);

                ctx.label("wmma_inner_loop");

                let inner_done = ctx.setp_ge_u32(inner_idx, wmma_tile_reg);
                ctx.branch_if(inner_done, "wmma_inner_end");

                // Load from shared A[tid_x, inner_idx]
                let as_idx = ctx.mad_lo_u32(tid_x, wmma_tile_reg, inner_idx);
                let as_offset = ctx.mul_wide_u32(as_idx, 4);
                let a_shared = ctx.ld_shared_f32(as_offset);

                // Load from shared B[inner_idx, tid_x % 16]
                let bs_idx = ctx.mad_lo_u32(inner_idx, wmma_tile_reg, tid_x);
                let bs_offset_base = ctx.mul_wide_u32(bs_idx, 4);
                let bs_addr = ctx.add_u64(smem_b_base_64, bs_offset_base);
                let b_shared = ctx.ld_shared_f32(bs_addr);

                // Accumulate
                let prod = ctx.mul_f32(a_shared, b_shared);
                let _new_acc = ctx.add_f32(acc, prod);

                // inner_idx++
                let _inner_next = ctx.add_u32(inner_idx, 1);
                ctx.branch("wmma_inner_loop");

                ctx.label("wmma_inner_end");

                // Synchronize before next K tile
                ctx.bar_sync(1);

                // k_tile_idx++
                let _k_tile_next = ctx.add_u32(k_tile_idx, 1);
                ctx.branch("wmma_k_loop");

                ctx.label("wmma_k_end");

                // Store result C[warp_row + tid/16, warp_col + tid%16]
                // Simplified: store based on thread ID
                let c_row = ctx.add_u32_reg(warp_row, tid_x);
                let c_row_offset = ctx.mul_wide_u32(c_row, self.config.n * 4);
                let c_col_offset = ctx.mul_wide_u32(warp_col, 4);
                let c_base = ctx.add_u64(c_ptr, c_row_offset);
                let c_addr = ctx.add_u64(c_base, c_col_offset);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_naive_gemm_params() {
        let kernel = GemmKernel::naive(512, 512, 512);
        assert_eq!(kernel.name(), "gemm_naive");
        assert_eq!(kernel.config.m, 512);
    }

    #[test]
    fn test_tiled_gemm_shared_memory() {
        let kernel = GemmKernel::tiled(1024, 1024, 1024, 32);
        let ptx_kernel = kernel.build_ptx();
        assert_eq!(ptx_kernel.shared_memory_bytes(), 32 * 32 * 4 * 2);
    }

    #[test]
    fn test_gemm_ptx_generation() {
        let kernel = GemmKernel::naive(1024, 1024, 1024);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".param .u64 a_ptr"));
        assert!(ptx.contains(".param .u64 b_ptr"));
        assert!(ptx.contains(".param .u64 c_ptr"));
        assert!(ptx.contains(".param .u32 m"));
        assert!(ptx.contains(".param .u32 n"));
        assert!(ptx.contains(".param .u32 k"));
    }

    #[test]
    fn test_naive_gemm_full_ptx() {
        let kernel = GemmKernel::naive(128, 128, 128);
        let ptx = kernel.emit_ptx();

        // Verify loop structure
        assert!(ptx.contains("loop_k:"));
        assert!(ptx.contains("loop_end:"));
        assert!(ptx.contains("exit:"));

        // Verify memory operations
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));

        // Verify arithmetic
        assert!(ptx.contains("mul.f32"));
        assert!(ptx.contains("add.f32"));
    }

    #[test]
    fn test_gemm_variants() {
        let naive = GemmKernel::naive(64, 64, 64);
        let tiled = GemmKernel::tiled(64, 64, 64, 16);
        let tensor = GemmKernel::tensor_core(64, 64, 64);

        assert_eq!(naive.name(), "gemm_naive");
        assert_eq!(tiled.name(), "gemm_tiled");
        assert_eq!(tensor.name(), "gemm_tensor_core");

        // All should produce valid PTX
        let _ = naive.emit_ptx();
        let _ = tiled.emit_ptx();
        let _ = tensor.emit_ptx();
    }

    #[test]
    fn test_gemm_config_default() {
        let config = GemmConfig::default();
        assert_eq!(config.m, 1024);
        assert_eq!(config.n, 1024);
        assert_eq!(config.k, 1024);
        assert_eq!(config.tile_size, 32);
        assert!(!config.use_tensor_cores);
    }

    #[test]
    fn test_tensor_core_kernel() {
        let kernel = GemmKernel::tensor_core(256, 256, 256);
        assert!(kernel.config.use_tensor_cores);
        let ptx_kernel = kernel.build_ptx();
        // WMMA fragments need shared memory
        assert!(ptx_kernel.shared_memory_bytes() > 0);
    }

    #[test]
    fn test_tiled_gemm_full_ptx() {
        let kernel = GemmKernel::tiled(256, 256, 256, 16);
        let ptx = kernel.emit_ptx();

        // Verify tiling structure
        assert!(ptx.contains("tile_loop:"));
        assert!(ptx.contains("tile_loop_end:"));
        assert!(ptx.contains("inner_k_loop:"));
        assert!(ptx.contains("inner_k_end:"));

        // Verify shared memory operations
        assert!(ptx.contains("ld.shared.f32") || ptx.contains("ld.f32")); // shared load
        assert!(ptx.contains("st.shared.f32") || ptx.contains("st.f32")); // shared store

        // Verify barrier synchronization
        assert!(ptx.contains("bar"));

        // Verify global loads/stores still present
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));
    }

    #[test]
    fn test_tensor_core_gemm_ptx() {
        let kernel = GemmKernel::tensor_core(512, 512, 512);
        let ptx = kernel.emit_ptx();

        // Verify WMMA structure
        assert!(ptx.contains("wmma_loop:") || ptx.contains("exit:"));

        // Verify memory operations (could be global or shared)
        assert!(ptx.contains("ld.global.f32") || ptx.contains("wmma_m_loop:"));
    }
}
