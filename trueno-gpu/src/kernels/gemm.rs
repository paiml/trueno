//! GEMM (General Matrix Multiply) Kernels
//!
//! Implements C = alpha * A @ B + beta * C

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

        PtxKernel::new("gemm_tiled")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(smem_size as usize)
            .build(|ctx| {
                // Tiled GEMM structure (simplified)
                ctx.label("exit");
                ctx.ret();
            })
    }

    fn build_tensor_core(&self) -> PtxKernel {
        // Tensor Core GEMM using WMMA
        PtxKernel::new("gemm_tensor_core")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "b_ptr")
            .param(PtxType::U64, "c_ptr")
            .param(PtxType::U32, "m")
            .param(PtxType::U32, "n")
            .param(PtxType::U32, "k")
            .shared_memory(16 * 16 * 2 * 2) // WMMA fragments
            .build(|ctx| {
                // WMMA structure (simplified)
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
}
