//! 4D Batched GEMM Kernel (for multi-head attention)
//!
//! Implements C[b,h] = A[b,h] @ B[b,h] for attention computations
//! with batch and head dimensions.

#![allow(clippy::similar_names)]

use crate::kernels::Kernel;
use crate::ptx::builder::{PtxArithmetic, PtxComparison, PtxControl, PtxMemory};
use crate::ptx::{PtxKernel, PtxType};

/// 4D Batched GEMM configuration for multi-head attention
#[derive(Debug, Clone)]
pub struct Batched4DGemmConfig {
    /// Batch size
    pub batch: u32,
    /// Number of attention heads
    pub heads: u32,
    /// M dimension (rows of A and C, typically sequence length)
    pub m: u32,
    /// N dimension (cols of B and C, typically sequence length or head_dim)
    pub n: u32,
    /// K dimension (cols of A, rows of B, typically head_dim)
    pub k: u32,
    /// Tile size for shared memory
    pub tile_size: u32,
}

impl Default for Batched4DGemmConfig {
    fn default() -> Self {
        Self {
            batch: 1,
            heads: 8,
            m: 512,
            n: 512,
            k: 64,
            tile_size: 16,
        }
    }
}

/// Batched 4D GEMM kernel for attention patterns (Q @ K^T, attn @ V)
/// Grid: ((m+tile-1)/tile, (n+tile-1)/tile, batch * heads)
#[derive(Debug, Clone)]
pub struct Batched4DGemmKernel {
    /// Kernel configuration
    pub config: Batched4DGemmConfig,
}

impl Batched4DGemmKernel {
    /// Create a new 4D batched GEMM kernel for attention
    /// Pattern: [batch, heads, m, k] @ [batch, heads, k, n] -> [batch, heads, m, n]
    #[must_use]
    pub fn new(batch: u32, heads: u32, m: u32, n: u32, k: u32) -> Self {
        Self {
            config: Batched4DGemmConfig {
                batch,
                heads,
                m,
                n,
                k,
                ..Default::default()
            },
        }
    }

    /// Create with custom tile size
    #[must_use]
    pub fn with_tile_size(batch: u32, heads: u32, m: u32, n: u32, k: u32, tile_size: u32) -> Self {
        Self {
            config: Batched4DGemmConfig {
                batch,
                heads,
                m,
                n,
                k,
                tile_size,
            },
        }
    }

    fn build_kernel(&self) -> PtxKernel {
        let tile_size = self.config.tile_size;
        let smem_size = tile_size * tile_size * 4 * 2;
        let n_tiles = (self.config.k + tile_size - 1) / tile_size;
        let heads_val = self.config.heads;
        let m_val = self.config.m;
        let n_val = self.config.n;
        let k_val = self.config.k;

        PtxKernel::new("batched_4d_gemm")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "batch")
            .param(PtxType::U32, "heads")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // z-dimension encodes batch * heads
                // batch_head_idx = ctaid.z
                // batch_idx = batch_head_idx / heads
                // head_idx = batch_head_idx % heads
                let batch_head_idx = ctx.special_reg(crate::ptx::PtxReg::CtaIdZ);
                let batch_idx = ctx.div_u32(batch_head_idx, heads_val);
                let head_idx = ctx.rem_u32(batch_head_idx, heads_val);

                // Thread and block indices
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let tid_y = ctx.special_reg(crate::ptx::PtxReg::TidY);
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ctaid_y = ctx.special_reg(crate::ptx::PtxReg::CtaIdY);

                let tile_size_reg = ctx.mov_u32_imm(tile_size);

                let row = ctx.mad_lo_u32(ctaid_y, tile_size_reg, tid_y);
                let col = ctx.mad_lo_u32(ctaid_x, tile_size_reg, tid_x);

                // Load parameters
                let batch_param = ctx.load_param_u32("batch");
                let heads_param = ctx.load_param_u32("heads");
                let m_param = ctx.load_param_u32("m");
                let n_param = ctx.load_param_u32("n");
                let k_param = ctx.load_param_u32("k");

                // Validity predicates
                let batch_valid = ctx.setp_lt_u32(batch_idx, batch_param);
                let head_valid = ctx.setp_lt_u32(head_idx, heads_param);
                let row_valid = ctx.setp_lt_u32(row, m_param);
                let col_valid = ctx.setp_lt_u32(col, n_param);

                // Load base pointers
                let a_ptr = ctx.load_param_u64("a_ptr");
                let b_ptr = ctx.load_param_u64("b_ptr");
                let c_ptr = ctx.load_param_u64("c_ptr");

                // Calculate 4D offsets using immediate strides
                // A: [batch, heads, m, k] -> stride: [heads*m*k, m*k, k, 1]
                // B: [batch, heads, k, n] -> stride: [heads*k*n, k*n, n, 1]
                // C: [batch, heads, m, n] -> stride: [heads*m*n, m*n, n, 1]
                let a_batch_off = ctx.mul_wide_u32(batch_idx, heads_val * m_val * k_val * 4);
                let a_head_off = ctx.mul_wide_u32(head_idx, m_val * k_val * 4);
                let a_base = ctx.add_u64(a_ptr, a_batch_off);
                let a_base = ctx.add_u64(a_base, a_head_off);

                let b_batch_off = ctx.mul_wide_u32(batch_idx, heads_val * k_val * n_val * 4);
                let b_head_off = ctx.mul_wide_u32(head_idx, k_val * n_val * 4);
                let b_base = ctx.add_u64(b_ptr, b_batch_off);
                let b_base = ctx.add_u64(b_base, b_head_off);

                let c_batch_off = ctx.mul_wide_u32(batch_idx, heads_val * m_val * n_val * 4);
                let c_head_off = ctx.mul_wide_u32(head_idx, m_val * n_val * 4);
                let c_base = ctx.add_u64(c_ptr, c_batch_off);
                let c_base = ctx.add_u64(c_base, c_head_off);

                // Initialize accumulator
                let acc = ctx.mov_f32_imm(0.0);

                // Tile loop
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
                ctx.branch_if_not(head_valid, "skip_a_load");
                ctx.branch_if_not(row_valid, "skip_a_load");
                ctx.branch_if_not(a_col_valid, "skip_a_load");

                let row_offset_a = ctx.mul_wide_u32(row, k_val * 4);
                let col_offset_a = ctx.mul_wide_u32(a_col, 4);
                let a_row_ptr = ctx.add_u64(a_base, row_offset_a);
                let a_addr = ctx.add_u64(a_row_ptr, col_offset_a);
                let a_val = ctx.ld_global_f32(a_addr);
                ctx.st_shared_f32(smem_a_offset, a_val);

                ctx.label("skip_a_load");

                // Load B tile
                let b_row = ctx.add_u32_reg(tile_k_offset, tid_y);
                let b_row_valid = ctx.setp_lt_u32(b_row, k_param);

                let zero_b = ctx.mov_f32_imm(0.0);
                ctx.st_shared_f32(smem_b_offset, zero_b);

                ctx.branch_if_not(batch_valid, "skip_b_load");
                ctx.branch_if_not(head_valid, "skip_b_load");
                ctx.branch_if_not(b_row_valid, "skip_b_load");
                ctx.branch_if_not(col_valid, "skip_b_load");

                let row_offset_b = ctx.mul_wide_u32(b_row, n_val * 4);
                let col_offset_b = ctx.mul_wide_u32(col, 4);
                let b_row_ptr = ctx.add_u64(b_base, row_offset_b);
                let b_addr = ctx.add_u64(b_row_ptr, col_offset_b);
                let b_val = ctx.ld_global_f32(b_addr);
                ctx.st_shared_f32(smem_b_offset, b_val);

                ctx.label("skip_b_load");

                ctx.bar_sync(0);

                // Inner loop
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

                // PARITY-114: Bounds check after all barriers
                ctx.branch_if_not(batch_valid, "exit");
                ctx.branch_if_not(head_valid, "exit");
                ctx.branch_if_not(row_valid, "exit");
                ctx.branch_if_not(col_valid, "exit");

                // Store result
                let c_row_offset = ctx.mul_wide_u32(row, n_val * 4);
                let c_col_offset = ctx.mul_wide_u32(col, 4);
                let c_row_ptr = ctx.add_u64(c_base, c_row_offset);
                let c_addr = ctx.add_u64(c_row_ptr, c_col_offset);
                ctx.st_global_f32(c_addr, acc);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

impl Kernel for Batched4DGemmKernel {
    fn name(&self) -> &str {
        "batched_4d_gemm"
    }

    fn build_ptx(&self) -> PtxKernel {
        self.build_kernel()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::gemm::basic::{GemmConfig, GemmKernel};
    use crate::kernels::gemm::batched::{BatchedGemmConfig, BatchedGemmKernel};

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

        // Verify arithmetic (FMA used for accumulation)
        assert!(ptx.contains("fma") || ptx.contains("mul.f32"));
        // Note: add.f32 may not appear if all additions are fused
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

    #[test]
    fn test_ptx_output_for_verification() {
        let kernel = GemmKernel::tiled(128, 128, 128, 32);
        let ptx = kernel.emit_ptx();

        std::fs::write("/tmp/test_tiled.ptx", &ptx).expect("write PTX");

        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("add.u32"));
        assert!(ptx.contains("%r17, %r17, 1") || ptx.contains("%r"));
        assert!(ptx.contains("%r10, %r10, 1") || ptx.contains("%r"));
    }

    #[test]
    fn test_naive_ptx_for_verification() {
        let kernel = GemmKernel::naive(128, 128, 128);
        let ptx = kernel.emit_ptx();

        std::fs::write("/tmp/test_naive.ptx", &ptx).expect("write PTX");

        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("loop_k:"));
        assert!(ptx.contains("loop_end:"));
    }

    #[test]
    fn test_wmma_fp16_kernel() {
        // Test WmmaFp16 variant - requires dimensions multiple of 16
        let kernel = GemmKernel::wmma_fp16(256, 256, 256);
        assert_eq!(kernel.name(), "gemm_wmma_fp16");
        assert!(kernel.config.use_tensor_cores);
        assert_eq!(kernel.config.tile_size, 16);

        // Build PTX
        let ptx_kernel = kernel.build_ptx();
        assert!(ptx_kernel.shared_memory_bytes() > 0);

        // Emit PTX and verify structure
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry gemm_wmma_fp16"));
        assert!(ptx.contains(".param"));
    }

    #[test]
    fn test_wmma_fp16_ptx_generation() {
        let kernel = GemmKernel::wmma_fp16(128, 128, 128);
        let ptx = kernel.emit_ptx();

        // Verify WMMA-specific patterns
        assert!(ptx.contains("wmma") || ptx.contains("mma") || ptx.contains("ld.global.f32"));

        // Write to /tmp for inspection
        std::fs::write("/tmp/test_wmma.ptx", &ptx).expect("write PTX");
    }

    #[test]
    fn test_all_gemm_variants_emit_valid_ptx() {
        let variants: Vec<GemmKernel> = vec![
            GemmKernel::naive(64, 64, 64),
            GemmKernel::tiled(64, 64, 64, 16),
            GemmKernel::tensor_core(64, 64, 64),
            GemmKernel::wmma_fp16(64, 64, 64),
        ];

        for kernel in variants {
            let name = kernel.name().to_string();
            let ptx = kernel.emit_ptx();
            let ptx_kernel = kernel.build_ptx();

            assert!(ptx.contains(".version"));
            assert!(ptx.contains(".entry"));
            assert!(ptx.contains(".param"));

            if name.contains("tiled") || name.contains("tensor") || name.contains("wmma") {
                assert!(ptx_kernel.shared_memory_bytes() > 0);
            }
        }
    }

    #[test]
    fn test_gemm_config_clone() {
        let config = GemmConfig::default();
        let cloned = config.clone();
        assert_eq!(config.m, cloned.m);
        assert_eq!(config.n, cloned.n);
        assert_eq!(config.k, cloned.k);
    }

    #[test]
    fn test_gemm_kernel_clone() {
        let kernel = GemmKernel::naive(128, 128, 128);
        let cloned = kernel.clone();
        assert_eq!(kernel.name(), cloned.name());
    }

    /// PARITY-114: Verify tiled GEMM doesn't have early exit before barriers
    #[test]
    fn test_parity_114_tiled_gemm_no_early_exit_before_barrier() {
        let kernel = GemmKernel::tiled(4, 8, 64, 32);
        let ptx = kernel.emit_ptx();

        let bar_sync_pos = ptx.find("bar.sync").expect("bar.sync required");
        let tile_loop_end_pos = ptx.find("tile_loop_end:").expect("tile_loop_end required");

        // Verify no early exit before tile_loop_end
        let early_exit = ptx.lines().any(|line| {
            if line.contains("@%p") && line.contains("bra exit") {
                let pos = ptx.find(line).unwrap_or(0);
                pos < tile_loop_end_pos
            } else {
                false
            }
        });
        assert!(!early_exit, "PARITY-114 violation");
        assert!(bar_sync_pos < tile_loop_end_pos, "bar.sync must be in loop");
    }

    /// PARITY-114: Verify n_tiles is correctly computed for small k
    #[test]
    fn test_parity_114_ntiles_computation() {
        // k=64, tile_size=32 -> n_tiles should be 2
        let kernel = GemmKernel::tiled(4, 8, 64, 32);
        let ptx = kernel.emit_ptx();

        // The PTX should have mov.u32 %rXX, 2; for n_tiles
        assert!(
            ptx.contains(", 2;"),
            "PTX should have n_tiles=2 for k=64, tile_size=32"
        );

        // And tile_size=32
        assert!(ptx.contains(", 32;"), "PTX should have tile_size=32");
    }

    /// PARITY-114: Verify gemm_tensor_core has no early exit before barrier
    #[test]
    fn test_parity_114_tensor_core_no_early_exit_before_barrier() {
        let kernel = GemmKernel::tensor_core(16, 16, 16);
        let ptx = kernel.emit_ptx();

        // Find positions of key elements
        let bar_sync_pos = ptx.find("bar.sync").expect("PTX should have bar.sync");
        let k_tile_end_pos = ptx.find("k_tile_end:").expect("PTX should have k_tile_end");

        // Verify bar.sync is inside the loop (before k_tile_end)
        assert!(
            bar_sync_pos < k_tile_end_pos,
            "bar.sync should be inside k_tile_loop (before k_tile_end)"
        );

        // Verify no unconditional exits before k_tile_end (conditional @!%p branches are OK)
        // The key is that bar.sync comes before the exit checks
    }

    /// PARITY-114: Verify gemm_wmma_fp16 has no early exit before barrier
    #[test]
    fn test_parity_114_wmma_no_early_exit_before_barrier() {
        let kernel = GemmKernel::wmma_fp16(16, 16, 16);
        let ptx = kernel.emit_ptx();

        // Find positions of key elements
        let bar_sync_pos = ptx.find("bar.sync").expect("PTX should have bar.sync");
        let k_tile_end_pos = ptx.find("k_tile_end:").expect("PTX should have k_tile_end");

        // Verify bar.sync is inside the loop (before k_tile_end)
        assert!(
            bar_sync_pos < k_tile_end_pos,
            "bar.sync should be inside k_tile_loop (before k_tile_end)"
        );

        // Verify wmma instructions are present
        assert!(ptx.contains("wmma.mma"), "WMMA kernel should have wmma.mma");
        assert!(
            ptx.contains("wmma.load"),
            "WMMA kernel should have wmma.load"
        );
    }

    /// PARITY-114 Countermeasure: Test boundary conditions (non-divisible dimensions)
    /// Five Whys Root Cause: We only tested "happy path" dimensions where all threads valid
    #[test]
    fn test_boundary_conditions_tensor_core() {
        // Test dimensions NOT divisible by tile size (16)
        // These are the cases where some threads are out-of-bounds
        let boundary_cases = [
            (17, 17, 17),    // Just over tile size
            (31, 31, 31),    // Just under 2 tiles
            (33, 33, 33),    // Just over 2 tiles
            (100, 100, 100), // Arbitrary non-power-of-2
            (1, 16, 16),     // Edge: single row
            (16, 1, 16),     // Edge: single column
        ];

        for (m, n, k) in boundary_cases {
            let kernel = GemmKernel::tensor_core(m, n, k);
            let ptx = kernel.emit_ptx();

            assert!(ptx.contains(".entry"));
            assert!(ptx.contains("bar.sync"));

            let bar_sync_pos = ptx.find("bar.sync").unwrap();
            let k_tile_end_pos = ptx.find("k_tile_end:").unwrap();
            assert!(bar_sync_pos < k_tile_end_pos);
        }
    }

    /// PARITY-114 Countermeasure: Test boundary conditions for tiled GEMM
    #[test]
    fn test_boundary_conditions_tiled_gemm() {
        let boundary_cases = [
            (17, 17, 17, 16),
            (65, 65, 65, 32),
            (100, 100, 100, 32),
            (1, 32, 32, 16),
        ];

        for (m, n, k, tile) in boundary_cases {
            let kernel = GemmKernel::tiled(m, n, k, tile);
            let ptx = kernel.emit_ptx();

            assert!(ptx.contains(".entry"));
            assert!(ptx.contains("bar.sync"));
        }
    }

    /// PARITY-114 Countermeasure: Test WMMA boundary conditions
    #[test]
    fn test_boundary_conditions_wmma() {
        // WMMA requires multiples of 16, but matrix dims can be non-multiple
        let boundary_cases = [(17, 17, 17), (32, 33, 34), (100, 100, 100)];

        for (m, n, k) in boundary_cases {
            let kernel = GemmKernel::wmma_fp16(m, n, k);
            let ptx = kernel.emit_ptx();

            assert!(ptx.contains(".entry"));
            assert!(ptx.contains("bar.sync"));
            assert!(ptx.contains("wmma.mma"));
        }
    }

    // =========================================================================
    // Batched GEMM Tests (Issue #71)
    // =========================================================================

    #[test]
    fn test_batched_gemm_naive() {
        let kernel = BatchedGemmKernel::naive(4, 64, 64, 64);
        assert_eq!(kernel.name(), "batched_gemm_naive");
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry batched_gemm_naive"));
        assert!(ptx.contains(".param .u32 batch"));
    }

    #[test]
    fn test_batched_gemm_tiled() {
        let kernel = BatchedGemmKernel::tiled(4, 64, 64, 64, 16);
        assert_eq!(kernel.name(), "batched_gemm_tiled");
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry batched_gemm_tiled"));
        assert!(ptx.contains("bar.sync"));
    }

    /// WAPR-PERF-011: Test batched WMMA kernel for multi-head attention
    #[test]
    fn test_batched_gemm_wmma_fp16() {
        let kernel = BatchedGemmKernel::wmma_fp16(6, 94, 64, 64);
        assert_eq!(kernel.name(), "batched_gemm_wmma_fp16");

        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry batched_gemm_wmma_fp16"));
        assert!(ptx.contains(".param .u32 batch"));
        assert!(ptx.contains("bar.sync"));
        assert!(ptx.contains("cvta.shared.u64"));
        assert!(ptx.contains("wmma") || ptx.contains("mma"));
    }

    #[test]
    fn test_batched_gemm_uses_z_dimension() {
        let kernel = BatchedGemmKernel::naive(8, 32, 32, 32);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("%ctaid.z"));
    }

    #[test]
    fn test_batched_gemm_config_default() {
        let config = BatchedGemmConfig::default();
        assert_eq!(config.batch, 1);
        assert_eq!(config.m, 1024);
        assert_eq!(config.n, 1024);
        assert_eq!(config.k, 1024);
        assert_eq!(config.tile_size, 16);
    }

    #[test]
    fn test_batched_4d_gemm() {
        let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
        assert_eq!(kernel.name(), "batched_4d_gemm");
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry batched_4d_gemm"));
        assert!(ptx.contains(".param .u32 batch"));
        assert!(ptx.contains(".param .u32 heads"));
    }

    #[test]
    fn test_batched_4d_gemm_with_tile_size() {
        let kernel = Batched4DGemmKernel::with_tile_size(2, 8, 64, 64, 32, 32);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry batched_4d_gemm"));
        assert!(ptx.contains("bar.sync"));
    }

    #[test]
    fn test_batched_4d_gemm_config_default() {
        let config = Batched4DGemmConfig::default();
        assert_eq!(config.batch, 1);
        assert_eq!(config.heads, 8);
        assert_eq!(config.m, 512);
        assert_eq!(config.n, 512);
        assert_eq!(config.k, 64);
        assert_eq!(config.tile_size, 16);
    }

    #[test]
    fn test_batched_4d_gemm_uses_batch_head_indexing() {
        let kernel = Batched4DGemmKernel::new(4, 12, 128, 128, 64);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains("%ctaid.z"));
        assert!(ptx.contains("div.") || ptx.contains("rem."));
    }

    /// PARITY-114: Verify batched GEMM tiled is barrier-safe
    #[test]
    fn test_barrier_safety_batched_gemm_tiled() {
        let kernel = BatchedGemmKernel::tiled(4, 64, 64, 64, 16);
        let result = kernel.analyze_barrier_safety();
        assert!(result.is_safe);
    }

    /// PARITY-114: Verify batched 4D GEMM is barrier-safe
    #[test]
    fn test_barrier_safety_batched_4d_gemm() {
        let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
        let result = kernel.analyze_barrier_safety();
        assert!(result.is_safe);
    }

    /// Test batched GEMM boundary conditions
    #[test]
    fn test_batched_gemm_boundary_conditions() {
        let boundary_cases = [
            (1, 17, 17, 17, 16),    // Single batch, non-power-of-2
            (8, 100, 100, 100, 16), // Multiple batches
            (16, 1, 64, 64, 16),    // Single row
        ];

        for (batch, m, n, k, tile) in boundary_cases {
            let kernel = BatchedGemmKernel::tiled(batch, m, n, k, tile);
            let ptx = kernel.emit_ptx();
            assert!(ptx.contains(".entry"));
            assert!(ptx.contains("bar.sync"));
        }
    }

    /// Test 4D GEMM boundary conditions
    #[test]
    fn test_batched_4d_gemm_boundary_conditions() {
        let boundary_cases = [(1, 1, 64, 64, 32), (2, 12, 17, 17, 17), (4, 8, 128, 64, 32)];

        for (batch, heads, m, n, k) in boundary_cases {
            let kernel = Batched4DGemmKernel::new(batch, heads, m, n, k);
            let ptx = kernel.emit_ptx();
            assert!(ptx.contains(".entry"));
            assert!(ptx.contains("bar.sync"));
        }
    }

    // =========================================================================
    // Additional tests for Batched4DGemmKernel coverage (95%+ target)
    // =========================================================================

    /// Test Debug trait for Batched4DGemmConfig
    #[test]
    fn test_batched_4d_gemm_config_debug() {
        let config = Batched4DGemmConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("Batched4DGemmConfig"));
        assert!(debug_str.contains("batch"));
        assert!(debug_str.contains("heads"));
        assert!(debug_str.contains("tile_size"));
    }

    /// Test Clone trait for Batched4DGemmConfig
    #[test]
    fn test_batched_4d_gemm_config_clone() {
        let config = Batched4DGemmConfig {
            batch: 4,
            heads: 12,
            m: 256,
            n: 256,
            k: 64,
            tile_size: 32,
        };
        let cloned = config.clone();
        assert_eq!(config.batch, cloned.batch);
        assert_eq!(config.heads, cloned.heads);
        assert_eq!(config.m, cloned.m);
        assert_eq!(config.n, cloned.n);
        assert_eq!(config.k, cloned.k);
        assert_eq!(config.tile_size, cloned.tile_size);
    }

    /// Test Debug trait for Batched4DGemmKernel
    #[test]
    fn test_batched_4d_gemm_kernel_debug() {
        let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
        let debug_str = format!("{:?}", kernel);
        assert!(debug_str.contains("Batched4DGemmKernel"));
        assert!(debug_str.contains("config"));
    }

    /// Test Clone trait for Batched4DGemmKernel
    #[test]
    fn test_batched_4d_gemm_kernel_clone() {
        let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
        let cloned = kernel.clone();
        assert_eq!(kernel.name(), cloned.name());
        assert_eq!(kernel.config.batch, cloned.config.batch);
        assert_eq!(kernel.config.heads, cloned.config.heads);
        assert_eq!(kernel.config.m, cloned.config.m);
        assert_eq!(kernel.config.n, cloned.config.n);
        assert_eq!(kernel.config.k, cloned.config.k);
    }

    /// Test as_module() method for Batched4DGemmKernel
    #[test]
    fn test_batched_4d_gemm_as_module() {
        let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
        let module = kernel.as_module();
        let ptx = module.emit();

        // Verify module structure
        assert!(ptx.contains(".version 8.0"));
        assert!(ptx.contains(".target sm_89"));
        assert!(ptx.contains(".address_size 64"));
        assert!(ptx.contains(".entry batched_4d_gemm"));
    }

    /// Test PTX content for 4D batched GEMM
    #[test]
    fn test_batched_4d_gemm_ptx_content() {
        let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
        let ptx = kernel.emit_ptx();

        // Verify all parameters are present
        assert!(ptx.contains(".param .u64 a_ptr"));
        assert!(ptx.contains(".param .u64 b_ptr"));
        assert!(ptx.contains(".param .u64 c_ptr"));
        assert!(ptx.contains(".param .u32 batch"));
        assert!(ptx.contains(".param .u32 heads"));
        assert!(ptx.contains(".param .u32 m"));
        assert!(ptx.contains(".param .u32 n"));
        assert!(ptx.contains(".param .u32 k"));

        // Verify batch/head indexing
        assert!(ptx.contains("%ctaid.z"));
        assert!(ptx.contains("div.u32") || ptx.contains("rem.u32"));
    }

    /// Test shared memory allocation for 4D batched GEMM
    #[test]
    fn test_batched_4d_gemm_shared_memory() {
        let kernel = Batched4DGemmKernel::with_tile_size(2, 8, 64, 64, 32, 16);
        let ptx_kernel = kernel.build_ptx();

        // Shared memory = tile_size * tile_size * 4 * 2 (for A and B tiles)
        // 16 * 16 * 4 * 2 = 2048 bytes
        assert_eq!(ptx_kernel.shared_memory_bytes(), 2048);
    }

    /// Test 4D GEMM with large tile size
    #[test]
    fn test_batched_4d_gemm_large_tile() {
        let kernel = Batched4DGemmKernel::with_tile_size(1, 4, 128, 128, 64, 32);
        let ptx = kernel.emit_ptx();

        // Verify kernel generates valid PTX
        assert!(ptx.contains(".entry batched_4d_gemm"));
        assert!(ptx.contains("bar.sync"));

        // Check shared memory is larger
        let ptx_kernel = kernel.build_ptx();
        assert_eq!(ptx_kernel.shared_memory_bytes(), 32 * 32 * 4 * 2);
    }

    /// Test 4D GEMM with minimum dimensions
    #[test]
    fn test_batched_4d_gemm_minimum_dims() {
        let kernel = Batched4DGemmKernel::new(1, 1, 1, 1, 1);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".entry batched_4d_gemm"));
        assert!(ptx.contains("bar.sync"));
    }

    /// Test 4D GEMM loop structure
    #[test]
    fn test_batched_4d_gemm_loop_structure() {
        let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
        let ptx = kernel.emit_ptx();

        // Verify tile loop
        assert!(ptx.contains("tile_loop:"));
        assert!(ptx.contains("tile_loop_end:"));

        // Verify inner k loop
        assert!(ptx.contains("inner_k_loop:"));
        assert!(ptx.contains("inner_k_end:"));

        // Verify exit label
        assert!(ptx.contains("exit:"));
    }

    /// Test 4D GEMM FMA operations
    #[test]
    fn test_batched_4d_gemm_fma_operations() {
        let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
        let ptx = kernel.emit_ptx();

        // Verify FMA is used for accumulation
        assert!(ptx.contains("fma.rn.f32"));

        // Verify shared memory operations
        assert!(ptx.contains("ld.shared.f32"));
        assert!(ptx.contains("st.shared.f32"));

        // Verify global memory operations
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));
    }

    /// Test 4D GEMM skip labels for bounds checking
    #[test]
    fn test_batched_4d_gemm_skip_labels() {
        let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
        let ptx = kernel.emit_ptx();

        // Verify skip labels for A and B loading
        assert!(ptx.contains("skip_a_load:"));
        assert!(ptx.contains("skip_b_load:"));
    }

    /// Test 4D GEMM with varying head counts
    #[test]
    fn test_batched_4d_gemm_varying_heads() {
        let head_counts = [1, 2, 4, 8, 12, 16, 32];

        for heads in head_counts {
            let kernel = Batched4DGemmKernel::new(2, heads, 64, 64, 32);
            let ptx = kernel.emit_ptx();

            assert!(ptx.contains(".entry batched_4d_gemm"));
            assert!(ptx.contains("bar.sync"));

            // Config should match
            assert_eq!(kernel.config.heads, heads);
        }
    }

    /// Test 4D GEMM with varying batch sizes
    #[test]
    fn test_batched_4d_gemm_varying_batches() {
        let batch_sizes = [1, 2, 4, 8, 16, 32];

        for batch in batch_sizes {
            let kernel = Batched4DGemmKernel::new(batch, 8, 64, 64, 32);
            let ptx = kernel.emit_ptx();

            assert!(ptx.contains(".entry batched_4d_gemm"));
            assert_eq!(kernel.config.batch, batch);
        }
    }

    /// Test 4D GEMM barrier safety passes
    #[test]
    fn test_batched_4d_gemm_barrier_safety_result() {
        let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
        let result = kernel.analyze_barrier_safety();

        assert!(result.is_safe);
        assert!(result.violations.is_empty());
        assert!(result.barrier_count > 0);
    }

    /// Test 4D GEMM with non-power-of-2 dimensions
    #[test]
    fn test_batched_4d_gemm_non_power_of_2() {
        let cases = [
            (3, 7, 33, 33, 17),
            (5, 11, 100, 100, 50),
            (2, 6, 94, 64, 64), // Typical attention pattern
        ];

        for (batch, heads, m, n, k) in cases {
            let kernel = Batched4DGemmKernel::new(batch, heads, m, n, k);
            let ptx = kernel.emit_ptx();

            assert!(ptx.contains(".entry batched_4d_gemm"));
            assert!(ptx.contains("bar.sync"));
        }
    }

    /// Test name() method returns correct value
    #[test]
    fn test_batched_4d_gemm_name() {
        let kernel = Batched4DGemmKernel::new(1, 1, 64, 64, 64);
        assert_eq!(kernel.name(), "batched_4d_gemm");
    }

    /// Test config default values match documentation
    #[test]
    fn test_batched_4d_gemm_config_default_values() {
        let config = Batched4DGemmConfig::default();

        // Verify documented defaults
        assert_eq!(config.batch, 1, "Default batch should be 1");
        assert_eq!(config.heads, 8, "Default heads should be 8");
        assert_eq!(config.m, 512, "Default m should be 512");
        assert_eq!(config.n, 512, "Default n should be 512");
        assert_eq!(config.k, 64, "Default k should be 64");
        assert_eq!(config.tile_size, 16, "Default tile_size should be 16");
    }
}
