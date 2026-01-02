//! Hand-Optimized GPU Kernels
//!
//! Pre-built kernels for common operations with optimal memory access patterns.
//!
//! ## Available Kernels
//!
//! - **GEMM**: Matrix multiplication (naive, tiled, Tensor Core)
//! - **Softmax**: Numerically stable softmax with warp shuffle
//! - **LayerNorm**: Fused layer normalization
//! - **Attention**: FlashAttention-style tiled attention + incremental (PAR-020)
//! - **Quantize**: Q4_K/Q5_K/Q6_K dequantization fused with matmul (PARITY-115/116/117)
//! - **BiasActivation**: Fused bias + activation epilogue (ReLU, GELU)
//! - **GEMV**: Matrix-vector multiply for M=1 decode throughput (CoalescedGemvKernel)
//!
//! ## Barrier Safety (PARITY-114)
//!
//! All kernels are validated for barrier safety to prevent thread divergence bugs.
//! Use `emit_ptx_validated()` for production to ensure no early-exit-before-barrier patterns.

mod attention;
mod bias_activation;
mod elementwise;
mod gemm;
mod gemv;
mod layernorm;
mod quantize;
mod softmax;

pub use attention::{AttentionKernel, IncrementalAttentionKernel};
pub use bias_activation::{Activation, BiasActivationKernel};
pub use elementwise::{
    ElementwiseMulKernel, FusedResidualRmsNormKernel, FusedSwigluKernel, GeluKernel,
    ResidualAddKernel, SiluKernel,
};
pub use gemm::{GemmConfig, GemmKernel};
pub use gemv::{CoalescedGemvKernel, GemvKernel};
pub use layernorm::{LayerNormKernel, RmsNormKernel};
pub use quantize::{
    Q4KGemvKernel, Q5KGemvKernel, Q5KKernel, Q6KGemvKernel, Q6KKernel, QuantizeKernel,
};
pub use softmax::SoftmaxKernel;

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
}
