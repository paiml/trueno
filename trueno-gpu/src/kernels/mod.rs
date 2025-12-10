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
//! - **Quantize**: Q4_K dequantization fused with matmul

mod attention;
mod gemm;
mod layernorm;
mod quantize;
mod softmax;

pub use attention::AttentionKernel;
pub use gemm::{GemmConfig, GemmKernel};
pub use layernorm::LayerNormKernel;
pub use quantize::QuantizeKernel;
pub use softmax::SoftmaxKernel;

use crate::ptx::{PtxKernel, PtxModule};

/// Kernel trait for GPU kernels
pub trait Kernel {
    /// Get kernel name
    fn name(&self) -> &str;

    /// Build PTX kernel
    fn build_ptx(&self) -> PtxKernel;

    /// Get PTX module containing this kernel
    fn as_module(&self) -> PtxModule {
        PtxModule::new()
            .version(8, 0)
            .target("sm_70")
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
