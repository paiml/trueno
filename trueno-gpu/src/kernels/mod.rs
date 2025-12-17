//! Hand-Optimized GPU Kernels
//!
//! Pre-built kernels for common operations with optimal memory access patterns.
//!
//! ## Available Kernels
//!
//! - **GEMM**: Matrix multiplication (naive, tiled, Tensor Core)
//! - **Softmax**: Numerically stable softmax with warp shuffle
//! - **LayerNorm**: Fused layer normalization
//! - **Attention**: FlashAttention-style tiled attention
//! - **Quantize**: Q4_K/Q5_K/Q6_K dequantization fused with matmul (PARITY-115/116/117)
//! - **BiasActivation**: Fused bias + activation epilogue (ReLU, GELU)
//! - **GEMV**: Matrix-vector multiply for M=1 decode throughput (CoalescedGemvKernel)

mod attention;
mod bias_activation;
mod gemm;
mod gemv;
mod layernorm;
mod quantize;
mod softmax;

pub use attention::AttentionKernel;
pub use bias_activation::{Activation, BiasActivationKernel};
pub use gemm::{GemmConfig, GemmKernel};
pub use gemv::{CoalescedGemvKernel, GemvKernel};
pub use layernorm::LayerNormKernel;
pub use quantize::{Q5KKernel, Q6KKernel, QuantizeKernel};
pub use softmax::SoftmaxKernel;

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
