//! Hand-Optimized GPU Kernels
//!
//! Pre-built kernels for common operations with optimal memory access patterns.
//!
//! ## Available Kernels
//!
//! - **GEMM**: Matrix multiplication (naive, tiled, Tensor Core)
//! - **Softmax**: Numerically stable softmax with warp shuffle
//! - **LayerNorm**: Fused layer normalization
//! - **Attention**: FlashAttention (SRAM tiling) + Paged (VRAM block management)
//! - **Quantize**: Q4_K/Q5_K/Q6_K dequantization fused with matmul (PARITY-115/116/117)
//! - **BiasActivation**: Fused bias + activation epilogue (ReLU, GELU)
//! - **GEMV**: Matrix-vector multiply for M=1 decode throughput (CoalescedGemvKernel)
//! - **Backward**: Training backward passes (GEMM, Softmax, LayerNorm, RMSNorm, Attention, Activations)
//! - **Optimizer**: Fused weight updates (AdamW, Adam, gradient clipping)
//!
//! ## Barrier Safety (PARITY-114)
//!
//! All kernels are validated for barrier safety to prevent thread divergence bugs.
//! Use `emit_ptx_validated()` for production to ensure no early-exit-before-barrier patterns.

mod argmax;
mod attention;
pub mod backward;
mod bias_activation;
mod conv1d;
mod elementwise;
mod fused;
mod gemm;
mod gemv;
mod layernorm;
pub mod lz4;
#[cfg(test)]
mod lz4_hash_store_test;
mod megakernel;
pub mod optimizer;
mod persistent;
mod quantize;
mod softmax;

pub use argmax::{ArgMaxFinalKernel, ArgMaxKernel};
pub use attention::{
    AttentionKernel, BatchedIncrementalAttentionKernel, FlashDecodingChunkKernel,
    FlashDecodingReduceKernel, IncrementalAttentionKernel, MultiWarpIncrementalAttentionKernel,
    FLASH_DECODE_CHUNK_SIZE,
};
pub use bias_activation::{Activation, BiasActivationKernel};
pub use conv1d::{Conv1dKernel, TiledConv1dKernel};
pub use elementwise::{
    BatchedResidualAddKernel,
    BatchedRopeKernel,
    BatchedScaleKernel,
    BatchedSoftmaxKernel,
    BatchedSwigluKernel,
    BatchedToInterleavedKernel,
    BatchedTransposeKernel,
    CopySingleHeadKernel,
    ElementwiseMulKernel,
    ExtractSingleHeadKernel,
    FusedResidualRmsNormKernel,
    FusedSwigluKernel,
    GeluKernel,
    InterleavedToBatchedKernel, // WAPR-PERF-004: Multi-head attention
    KvCacheScatterIndirectKernel,
    KvCacheScatterKernel,
    PreciseRopeIndirectKernel,
    PreciseRopeKernel, // CORRECTNESS-013
    ReluKernel,        // Issue #88: Forward ReLU kernel
    ResidualAddKernel,
    RopeIndirectKernel,
    RopeKernel,
    RopeNeoxIndirectKernel,
    RopeNeoxKernel,
    ScaleKernel,
    SiluKernel,
    TransposeKernel, // WAPR-PERF-004
};
pub use fused::{FusedGateUpKernel, FusedGemmBiasGeluKernel, FusedQKVKernel};
pub use gemm::{
    Batched4DGemmConfig, Batched4DGemmKernel, BatchedGemmConfig, BatchedGemmKernel, GemmConfig,
    GemmKernel,
};
pub use gemv::{CoalescedGemvKernel, GemvKernel};
pub use layernorm::{
    BatchedVectorizedRmsNormKernel, LayerNormKernel, PreciseRmsNormKernel, RmsNormKernel,
    VectorizedRmsNormKernel,
};
pub use lz4::{Lz4WarpCompressKernel, Lz4WarpDecompressKernel};
pub use megakernel::TransformerBlockMegakernel;
pub use optimizer::{AdamStepKernel, AdamWStepKernel, GradientClipKernel};
pub use persistent::PersistentDecoderKernel;
pub use quantize::{
    BatchedQ4KGemvKernel, BatchedQ6KGemvKernel, ChunkedTiledQ4KGemvKernel, CoalescedQ4KGemvKernel,
    CoalescedQ6KGemvKernel, Dp4aQ4KGemvKernel, Fp16Q4KGemvKernel, FusedGateUpQ4KGemvKernel,
    FusedRmsNormQ4KGemvKernel, PackedDp4aQ4KQ8Kernel, Q4KGemvKernel, Q4KQ8DotKernel,
    Q4_0GemvKernel, Q4_1GemvKernel, Q5KGemvKernel, Q5KKernel, Q5_0GemvKernel, Q6KGemvKernel,
    Q6KKernel, Q8QuantizeKernel, Q8_0GemvKernel, QuantizeKernel, TensorCoreQ4KGemmKernel,
    TiledQ4KGemvKernel, TrueDp4aQ4KGemvKernel, VectorizedQ4KGemvKernel,
};
pub use softmax::{LongRowSoftmaxKernel, SoftmaxKernel};

use crate::ptx::optimize::barrier_safety::{self, BarrierSafetyResult};
use crate::ptx::{PtxKernel, PtxModule};

/// Kernel trait for GPU kernels
pub trait Kernel {
    /// Get kernel name
    fn name(&self) -> &str;

    /// Build PTX kernel
    fn build_ptx(&self) -> PtxKernel;

    /// Get PTX module containing this kernel
    /// Uses sm_89 for RTX 4090 (Ada Lovelace) compatibility
    fn as_module(&self) -> PtxModule {
        PtxModule::new()
            .version(8, 0)
            .target("sm_89")
            .address_size(64)
            .add_kernel(self.build_ptx())
    }

    /// Emit PTX source
    fn emit_ptx(&self) -> String {
        self.as_module().emit()
    }

    /// Analyze PTX for barrier safety (PARITY-114 prevention)
    ///
    /// Returns detailed analysis of barrier safety, including any violations found.
    fn analyze_barrier_safety(&self) -> BarrierSafetyResult {
        let ptx = self.emit_ptx();
        barrier_safety::analyze(&ptx)
    }

    /// Validate PTX is barrier-safe (PARITY-114 prevention)
    ///
    /// Returns `Ok(())` if safe, `Err` with violation details if not.
    fn validate_barrier_safety(&self) -> Result<(), String> {
        let ptx = self.emit_ptx();
        barrier_safety::validate(&ptx)
    }

    /// Emit PTX with barrier safety validation (recommended for production)
    ///
    /// # Panics
    ///
    /// Panics if the PTX contains barrier safety violations (PARITY-114).
    /// Use this in production to catch bugs at compile time rather than runtime.
    fn emit_ptx_validated(&self) -> String {
        let ptx = self.emit_ptx();
        if let Err(e) = barrier_safety::validate(&ptx) {
            panic!(
                "PARITY-114: Barrier safety violation in kernel '{}': {}",
                self.name(),
                e
            );
        }
        ptx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_kernel_builds() {
        let kernel = GemmKernel::naive(1024, 1024, 1024);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".visible .entry"));
        assert!(ptx.contains("gemm"));
    }

    #[test]
    fn test_softmax_kernel_builds() {
        let kernel = SoftmaxKernel::new(4096);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".visible .entry"));
        assert!(ptx.contains("softmax"));
    }

    // =========================================================================
    // PARITY-114 Barrier Safety Tests - All Kernels
    // =========================================================================

    /// PARITY-114: GEMM naive kernel is barrier-safe
    #[test]
    fn test_barrier_safety_gemm_naive() {
        let kernel = GemmKernel::naive(64, 64, 64);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "GEMM naive should be barrier-safe: {:?}",
            result.violations
        );
    }

    /// PARITY-114: GEMM tiled kernel is barrier-safe
    #[test]
    fn test_barrier_safety_gemm_tiled() {
        let kernel = GemmKernel::tiled(64, 64, 64, 16);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "GEMM tiled should be barrier-safe: {:?}",
            result.violations
        );
    }

    /// PARITY-114: GEMM tensor core kernel is barrier-safe
    #[test]
    fn test_barrier_safety_gemm_tensor_core() {
        let kernel = GemmKernel::tensor_core(64, 64, 64);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "GEMM tensor core should be barrier-safe: {:?}",
            result.violations
        );
    }

    /// PARITY-114: GEMM WMMA FP16 kernel is barrier-safe
    #[test]
    fn test_barrier_safety_gemm_wmma() {
        let kernel = GemmKernel::wmma_fp16(64, 64, 64);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "GEMM WMMA should be barrier-safe: {:?}",
            result.violations
        );
    }

    /// PARITY-114: Attention kernel is barrier-safe
    #[test]
    fn test_barrier_safety_attention() {
        let kernel = AttentionKernel::new(64, 32);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "Attention should be barrier-safe: {:?}",
            result.violations
        );
    }

    /// PARITY-114: Tensor Core attention kernel is barrier-safe
    #[test]
    fn test_barrier_safety_attention_tensor_core() {
        let kernel = AttentionKernel::tensor_core(64, 32);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "TC Attention should be barrier-safe: {:?}",
            result.violations
        );
    }

    /// PARITY-114: Softmax kernel is barrier-safe
    #[test]
    fn test_barrier_safety_softmax() {
        let kernel = SoftmaxKernel::new(1024);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "Softmax should be barrier-safe: {:?}",
            result.violations
        );
    }

    /// PARITY-114: LayerNorm kernel is barrier-safe
    #[test]
    fn test_barrier_safety_layernorm() {
        let kernel = LayerNormKernel::new(512);
        let result = kernel.analyze_barrier_safety();
        assert!(
            result.is_safe,
            "LayerNorm should be barrier-safe: {:?}",
            result.violations
        );
    }

    /// PARITY-114: validate_barrier_safety returns Ok for safe kernels
    #[test]
    fn test_validate_barrier_safety_ok() {
        let kernel = GemmKernel::naive(32, 32, 32);
        assert!(
            kernel.validate_barrier_safety().is_ok(),
            "Safe kernel should pass validation"
        );
    }

    /// PARITY-114: emit_ptx_validated works for safe kernels
    #[test]
    fn test_emit_ptx_validated_works() {
        let kernel = GemmKernel::naive(32, 32, 32);
        let ptx = kernel.emit_ptx_validated(); // Should not panic
        assert!(ptx.contains(".entry"));
    }

    /// PARITY-114: Boundary condition - non-divisible dimensions are barrier-safe
    #[test]
    fn test_barrier_safety_boundary_conditions() {
        // Test dimensions not divisible by tile size
        let test_cases = [
            GemmKernel::tensor_core(17, 17, 17),
            GemmKernel::tensor_core(33, 33, 33),
            GemmKernel::tensor_core(100, 100, 100),
        ];

        for kernel in test_cases {
            let result = kernel.analyze_barrier_safety();
            assert!(
                result.is_safe,
                "Boundary case {} should be barrier-safe: {:?}",
                kernel.name(),
                result.violations
            );
        }
    }

    // =========================================================================
    // Additional Coverage Tests - More Kernel Types
    // =========================================================================

    /// Mock kernel that produces barrier-unsafe PTX for testing
    struct UnsafeMockKernel;

    impl Kernel for UnsafeMockKernel {
        fn name(&self) -> &str {
            "unsafe_mock_kernel"
        }

        fn build_ptx(&self) -> PtxKernel {
            PtxKernel::new("unsafe_mock_kernel")
        }

        fn emit_ptx(&self) -> String {
            r#".version 8.0
.target sm_89
.address_size 64

.visible .entry unsafe_mock_kernel() {
loop_start:
    bra exit;
    bar.sync 0;
    bra loop_start;

loop_start_end:
exit:
    ret;
}
"#
            .to_string()
        }
    }

    /// Test UnsafeMockKernel::build_ptx() coverage
    #[test]
    fn test_unsafe_mock_kernel_build_ptx() {
        let kernel = UnsafeMockKernel;
        let ptx_kernel = kernel.build_ptx();
        assert!(ptx_kernel.shared_memory_bytes() == 0);
    }

    /// Test UnsafeMockKernel::as_module()
    #[test]
    fn test_unsafe_mock_kernel_as_module() {
        let kernel = UnsafeMockKernel;
        let module = kernel.as_module();
        let ptx = module.emit();
        assert!(ptx.contains(".version 8.0"));
    }

    /// Test UnsafeMockKernel::analyze_barrier_safety() detects violations
    #[test]
    fn test_unsafe_mock_kernel_analyze() {
        let kernel = UnsafeMockKernel;
        let result = kernel.analyze_barrier_safety();
        assert!(!result.is_safe);
        assert!(!result.violations.is_empty());
    }

    /// Test UnsafeMockKernel::validate_barrier_safety() returns Err
    #[test]
    fn test_unsafe_mock_kernel_validate() {
        let kernel = UnsafeMockKernel;
        let result = kernel.validate_barrier_safety();
        assert!(result.is_err());
    }

    /// Test emit_ptx_validated panics for unsafe PTX
    #[test]
    #[should_panic(expected = "PARITY-114")]
    fn test_emit_ptx_validated_panics() {
        let kernel = UnsafeMockKernel;
        let _ = kernel.emit_ptx_validated();
    }

    /// Test CoalescedGemvKernel barrier safety
    #[test]
    fn test_barrier_safety_coalesced_gemv() {
        let kernel = CoalescedGemvKernel::new(1024, 4096);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test TransposeKernel barrier safety
    #[test]
    fn test_barrier_safety_transpose() {
        let kernel = TransposeKernel::new(1024, 1024);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test ResidualAddKernel barrier safety
    #[test]
    fn test_barrier_safety_residual_add() {
        let kernel = ResidualAddKernel::new(4096);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test FusedSwigluKernel barrier safety
    #[test]
    fn test_barrier_safety_fused_swiglu() {
        let kernel = FusedSwigluKernel::new(4096);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test RopeKernel barrier safety
    #[test]
    fn test_barrier_safety_rope() {
        let kernel = RopeKernel::new(8, 64, 10000.0);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test ElementwiseMulKernel barrier safety
    #[test]
    fn test_barrier_safety_elementwise_mul() {
        let kernel = ElementwiseMulKernel::new(4096);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test PreciseRmsNormKernel emits valid PTX
    #[test]
    fn test_precise_rmsnorm_ptx() {
        let kernel = PreciseRmsNormKernel::new(512);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".entry"));
    }

    /// Test RmsNormKernel barrier safety
    #[test]
    fn test_barrier_safety_rmsnorm() {
        let kernel = RmsNormKernel::new(512);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test VectorizedRmsNormKernel emits valid PTX
    #[test]
    fn test_vectorized_rmsnorm_ptx() {
        let kernel = VectorizedRmsNormKernel::new(512);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".entry"));
    }

    /// Test GemvKernel barrier safety
    #[test]
    fn test_barrier_safety_gemv() {
        let kernel = GemvKernel::new(1024, 4096);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test BiasActivationKernel barrier safety
    #[test]
    fn test_barrier_safety_bias_activation() {
        let kernel = BiasActivationKernel::new(1024, 64).with_relu();
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test IncrementalAttentionKernel barrier safety
    #[test]
    fn test_barrier_safety_incremental_attention() {
        let kernel = IncrementalAttentionKernel::new(2048, 64, 8);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test BatchedGemmKernel barrier safety
    #[test]
    fn test_barrier_safety_batched_gemm() {
        let kernel = BatchedGemmKernel::naive(4, 64, 64, 64);
        assert!(kernel.analyze_barrier_safety().is_safe);

        let kernel_tiled = BatchedGemmKernel::tiled(4, 64, 64, 64, 16);
        assert!(kernel_tiled.analyze_barrier_safety().is_safe);
    }

    /// Test Batched4DGemmKernel barrier safety
    #[test]
    fn test_barrier_safety_batched_4d_gemm() {
        let kernel = Batched4DGemmKernel::new(2, 8, 64, 64, 32);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test LongRowSoftmaxKernel barrier safety
    #[test]
    fn test_barrier_safety_long_row_softmax() {
        let kernel = LongRowSoftmaxKernel::new(8192);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test elementwise kernels are barrier-safe
    #[test]
    fn test_barrier_safety_elementwise_kernels() {
        assert!(ReluKernel::new(1024).analyze_barrier_safety().is_safe);
        assert!(GeluKernel::new(1024).analyze_barrier_safety().is_safe);
        assert!(SiluKernel::new(1024).analyze_barrier_safety().is_safe);
        assert!(ScaleKernel::new(1024).analyze_barrier_safety().is_safe);
    }

    /// Test validate_barrier_safety returns Ok for safe kernels
    #[test]
    fn test_validate_barrier_safety_all_pass() {
        let kernels: Vec<Box<dyn Kernel>> = vec![
            Box::new(GemmKernel::naive(32, 32, 32)),
            Box::new(SoftmaxKernel::new(128)),
            Box::new(LayerNormKernel::new(64)),
        ];
        for kernel in kernels {
            assert!(kernel.validate_barrier_safety().is_ok());
        }
    }

    /// Test as_module returns correct structure
    #[test]
    fn test_as_module_structure() {
        let kernel = GemmKernel::naive(32, 32, 32);
        let module = kernel.as_module();
        let ptx = module.emit();
        assert!(ptx.contains(".version 8.0"));
        assert!(ptx.contains(".target sm_89"));
        assert!(ptx.contains(".address_size 64"));
    }

    /// Test build_ptx for tiled kernel uses shared memory
    #[test]
    fn test_build_ptx_shared_memory() {
        let kernel = GemmKernel::tiled(64, 64, 64, 16);
        let ptx_kernel = kernel.build_ptx();
        assert!(ptx_kernel.shared_memory_bytes() > 0);
    }

    /// Test analyze_barrier_safety returns counts
    #[test]
    fn test_analyze_barrier_safety_counts() {
        let kernel = GemmKernel::tiled(64, 64, 64, 16);
        let result = kernel.analyze_barrier_safety();
        assert!(result.barrier_count > 0);
        assert!(result.exit_count > 0);
        assert!(result.is_safe);
        assert!(result.violations.is_empty());
    }

    /// Test emit_ptx_validated for multiple kernels
    #[test]
    fn test_emit_ptx_validated_multiple() {
        let kernels: Vec<Box<dyn Kernel>> = vec![
            Box::new(GemmKernel::naive(32, 32, 32)),
            Box::new(SoftmaxKernel::new(128)),
        ];
        for kernel in kernels {
            let ptx = kernel.emit_ptx_validated();
            assert!(!ptx.is_empty());
        }
    }

    /// Test kernel name consistency
    #[test]
    fn test_kernel_name_consistency() {
        let kernel = GemmKernel::naive(128, 128, 128);
        let n1 = kernel.name();
        let n2 = kernel.name();
        assert_eq!(n1, n2);
        assert!(!n1.is_empty());
    }

    /// Test emit_ptx produces valid output each time
    #[test]
    fn test_emit_ptx_validity() {
        let kernel = GemmKernel::naive(64, 64, 64);
        let ptx1 = kernel.emit_ptx();
        let ptx2 = kernel.emit_ptx();
        // Both outputs should be valid PTX (may differ in register ordering)
        assert!(ptx1.contains(".version 8.0"));
        assert!(ptx2.contains(".version 8.0"));
        assert!(ptx1.contains(".entry"));
        assert!(ptx2.contains(".entry"));
    }

    /// Test validate error message format
    #[test]
    fn test_validate_error_format() {
        let unsafe_ptx = r#"
.version 8.0
.target sm_89
.address_size 64

.visible .entry test() {
loop_start:
    bra exit;
    bar.sync 0;
    bra loop_start;

loop_start_end:
exit:
    ret;
}
"#;
        let result = barrier_safety::validate(unsafe_ptx);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("PARITY-114"));
    }

    /// Test ArgMaxKernel barrier safety
    #[test]
    fn test_barrier_safety_argmax() {
        let kernel = ArgMaxKernel::new(4096);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test ArgMaxFinalKernel barrier safety
    #[test]
    fn test_barrier_safety_argmax_final() {
        let kernel = ArgMaxFinalKernel::new(128);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test Conv1dKernel barrier safety
    #[test]
    fn test_barrier_safety_conv1d() {
        let kernel = Conv1dKernel::new(3, 128, 256, 3, 1);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test TiledConv1dKernel barrier safety
    #[test]
    fn test_barrier_safety_tiled_conv1d() {
        let kernel = TiledConv1dKernel::whisper_conv1();
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test optimizer kernels barrier safety
    #[test]
    fn test_barrier_safety_optimizer() {
        assert!(AdamWStepKernel::new(4096).analyze_barrier_safety().is_safe);
        assert!(AdamStepKernel::new(4096).analyze_barrier_safety().is_safe);
        assert!(
            GradientClipKernel::new(4096)
                .analyze_barrier_safety()
                .is_safe
        );
    }

    /// Test quantize kernels barrier safety
    #[test]
    fn test_barrier_safety_quantize() {
        assert!(
            Q4KGemvKernel::new(4096, 4096)
                .analyze_barrier_safety()
                .is_safe
        );
        assert!(
            Q6KGemvKernel::new(4096, 4096)
                .analyze_barrier_safety()
                .is_safe
        );
        assert!(Q8QuantizeKernel::new(4096).analyze_barrier_safety().is_safe);
    }

    /// Test FusedQKVKernel barrier safety
    #[test]
    fn test_barrier_safety_fused_qkv() {
        let kernel = FusedQKVKernel::new(512, 64);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test FusedGateUpKernel barrier safety
    #[test]
    fn test_barrier_safety_fused_gate_up() {
        let kernel = FusedGateUpKernel::new(512, 2048);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test FusedGemmBiasGeluKernel barrier safety
    #[test]
    fn test_barrier_safety_fused_gemm_bias_gelu() {
        let kernel = FusedGemmBiasGeluKernel::new(512, 2048, 512);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test TransformerBlockMegakernel barrier safety
    #[test]
    fn test_barrier_safety_megakernel() {
        let kernel = TransformerBlockMegakernel::new(512, 2048, 8);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test PersistentDecoderKernel emits valid PTX
    #[test]
    fn test_persistent_ptx() {
        let kernel = PersistentDecoderKernel::new(512, 12, 2048);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".entry"));
    }

    /// Test Lz4 kernels barrier safety
    #[test]
    fn test_barrier_safety_lz4() {
        assert!(
            Lz4WarpCompressKernel::new(4)
                .analyze_barrier_safety()
                .is_safe
        );
        assert!(
            Lz4WarpDecompressKernel::new(4)
                .analyze_barrier_safety()
                .is_safe
        );
    }

    /// Test batched elementwise kernels
    #[test]
    fn test_barrier_safety_batched_elementwise() {
        assert!(
            BatchedSoftmaxKernel::new(4, 1024)
                .analyze_barrier_safety()
                .is_safe
        );
        assert!(
            BatchedScaleKernel::new(1024)
                .analyze_barrier_safety()
                .is_safe
        );
    }

    /// Test FusedResidualRmsNormKernel barrier safety
    #[test]
    fn test_barrier_safety_fused_residual_rmsnorm() {
        let kernel = FusedResidualRmsNormKernel::new(512);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test KvCacheScatterKernel barrier safety
    #[test]
    fn test_barrier_safety_kv_cache_scatter() {
        let kernel = KvCacheScatterKernel::new(32, 64, 2048);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }

    /// Test KvCacheScatterIndirectKernel barrier safety
    #[test]
    fn test_barrier_safety_kv_cache_scatter_indirect() {
        let kernel = KvCacheScatterIndirectKernel::new(32, 64, 2048);
        assert!(kernel.analyze_barrier_safety().is_safe);
    }
}

/// Property tests for kernel builders (TRUENO-SPEC-014 TASK-011)
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// All GEMM naive kernels produce valid PTX regardless of dimensions
        #[test]
        fn gemm_naive_always_valid(m in 16u32..512, n in 16u32..512, k in 16u32..512) {
            let kernel = GemmKernel::naive(m, n, k);
            let ptx = kernel.emit_ptx();

            // Must have PTX header
            prop_assert!(ptx.contains(".version"), "Missing PTX version");
            prop_assert!(ptx.contains(".target"), "Missing target");
            prop_assert!(ptx.contains(".entry"), "Missing entry point");

            // Must have kernel parameters
            prop_assert!(ptx.contains(".param"), "Missing parameters");
            prop_assert!(ptx.contains("a_ptr"), "Missing A matrix pointer");
            prop_assert!(ptx.contains("b_ptr"), "Missing B matrix pointer");
            prop_assert!(ptx.contains("c_ptr"), "Missing C matrix pointer");
        }

        /// All GEMM tiled kernels produce valid PTX with shared memory
        #[test]
        fn gemm_tiled_uses_shared_memory(m in 32u32..256, n in 32u32..256, k in 32u32..256, tile in 8u32..32) {
            let kernel = GemmKernel::tiled(m, n, k, tile);
            let ptx_kernel = kernel.build_ptx();

            // Tiled GEMM must use shared memory
            prop_assert!(ptx_kernel.shared_memory_bytes() > 0, "Tiled GEMM should use shared memory");
        }

        /// All Softmax kernels produce valid PTX
        #[test]
        fn softmax_always_valid(seq_len in 64u32..8192) {
            let kernel = SoftmaxKernel::new(seq_len);
            let ptx = kernel.emit_ptx();

            // Must have PTX header
            prop_assert!(ptx.contains(".version"), "Missing PTX version");
            prop_assert!(ptx.contains(".entry"), "Missing entry point");
            prop_assert!(ptx.contains("softmax"), "Missing softmax kernel name");
        }

        /// All LayerNorm kernels produce valid PTX
        #[test]
        fn layernorm_always_valid(hidden_size in 64u32..4096) {
            let kernel = LayerNormKernel::new(hidden_size);
            let ptx = kernel.emit_ptx();

            // Must have PTX header
            prop_assert!(ptx.contains(".version"), "Missing PTX version");
            prop_assert!(ptx.contains(".entry"), "Missing entry point");
        }

        /// All Attention kernels produce valid PTX
        #[test]
        fn attention_always_valid(
            seq_len in 64u32..2048,
            head_dim in 32u32..128,
        ) {
            let kernel = AttentionKernel::new(seq_len, head_dim);
            let ptx = kernel.emit_ptx();

            // Must have PTX header
            prop_assert!(ptx.contains(".version"), "Missing PTX version");
            prop_assert!(ptx.contains(".entry"), "Missing entry point");
        }

        /// Kernel names are deterministic
        #[test]
        fn kernel_names_deterministic(m in 16u32..512, n in 16u32..512, k in 16u32..512) {
            let kernel1 = GemmKernel::naive(m, n, k);
            let kernel2 = GemmKernel::naive(m, n, k);

            prop_assert_eq!(kernel1.name(), kernel2.name(), "Kernel names should be deterministic");
        }

        /// PTX emission produces consistent structure
        #[test]
        fn ptx_emission_consistent_structure(m in 16u32..256, n in 16u32..256, k in 16u32..256) {
            let kernel = GemmKernel::naive(m, n, k);
            let ptx = kernel.emit_ptx();

            // Verify consistent structure regardless of dimensions
            prop_assert!(ptx.contains(".version 8.0"), "Must have version 8.0");
            prop_assert!(ptx.contains(".target sm_89"), "Must target sm_89 for RTX 4090");
            prop_assert!(ptx.contains(".address_size 64"), "Must use 64-bit addresses");
            prop_assert!(ptx.contains("ret;"), "Must have return statement");
        }
    }

    /// Edge case tests (not random)
    #[test]
    fn test_minimum_dimensions() {
        // Test smallest valid dimensions
        let kernel = GemmKernel::naive(1, 1, 1);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry"), "Should handle 1x1x1");
    }

    #[test]
    fn test_large_dimensions() {
        // Test large but reasonable dimensions
        let kernel = GemmKernel::naive(4096, 4096, 4096);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry"), "Should handle 4096x4096");
    }

    #[test]
    fn test_non_power_of_two() {
        // Test non-power-of-two dimensions
        let kernel = GemmKernel::naive(127, 255, 63);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry"), "Should handle non-power-of-two");
    }

    // =========================================================================
    // Batched GEMM Property Tests (Issue #71)
    // =========================================================================

    proptest! {
        /// All batched GEMM naive kernels produce valid PTX regardless of dimensions
        #[test]
        fn batched_gemm_naive_always_valid(
            batch in 1u32..16,
            m in 16u32..256,
            n in 16u32..256,
            k in 16u32..256
        ) {
            let kernel = BatchedGemmKernel::naive(batch, m, n, k);
            let ptx = kernel.emit_ptx();

            prop_assert!(ptx.contains(".version"), "Missing PTX version");
            prop_assert!(ptx.contains(".entry"), "Missing entry point");
            prop_assert!(ptx.contains(".param .u32 batch"), "Missing batch parameter");
            prop_assert!(ptx.contains("%ctaid.z"), "Missing batch indexing via ctaid.z");
        }

        /// All batched GEMM tiled kernels produce valid PTX with shared memory
        #[test]
        fn batched_gemm_tiled_always_valid(
            batch in 1u32..8,
            m in 32u32..128,
            n in 32u32..128,
            k in 32u32..128,
            tile in 8u32..17
        ) {
            let kernel = BatchedGemmKernel::tiled(batch, m, n, k, tile);
            let ptx = kernel.emit_ptx();
            let ptx_kernel = kernel.build_ptx();

            prop_assert!(ptx.contains(".entry"), "Missing entry point");
            prop_assert!(ptx.contains("bar.sync"), "Missing barrier synchronization");
            prop_assert!(ptx_kernel.shared_memory_bytes() > 0, "Should use shared memory");
        }

        /// All 4D batched GEMM kernels produce valid PTX
        #[test]
        fn batched_4d_gemm_always_valid(
            batch in 1u32..8,
            heads in 1u32..16,
            m in 32u32..128,
            n in 32u32..128,
            k in 16u32..64
        ) {
            let kernel = Batched4DGemmKernel::new(batch, heads, m, n, k);
            let ptx = kernel.emit_ptx();

            prop_assert!(ptx.contains(".version"), "Missing PTX version");
            prop_assert!(ptx.contains(".entry"), "Missing entry point");
            prop_assert!(ptx.contains(".param .u32 batch"), "Missing batch parameter");
            prop_assert!(ptx.contains(".param .u32 heads"), "Missing heads parameter");
            prop_assert!(ptx.contains("%ctaid.z"), "Missing batch*heads indexing");
        }
    }

    #[test]
    fn test_batched_gemm_minimum_batch() {
        let kernel = BatchedGemmKernel::naive(1, 32, 32, 32);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry"), "Should handle batch=1");
    }

    #[test]
    fn test_batched_4d_gemm_attention_pattern() {
        // Typical transformer attention: batch=2, heads=8, seq_len=512, head_dim=64
        let kernel = Batched4DGemmKernel::new(2, 8, 512, 512, 64);
        let ptx = kernel.emit_ptx();
        assert!(ptx.contains(".entry"), "Should handle attention pattern");
        assert!(
            ptx.contains("bar.sync"),
            "Should have barriers for tiled compute"
        );
    }
}
