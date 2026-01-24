//! Backward Pass Kernels for Training
//!
//! This module provides GPU kernels for computing gradients during backpropagation.
//! These kernels are essential for training neural networks with CUDA acceleration.
//!
//! ## Available Backward Kernels
//!
//! - **Activations**: ReLU, GELU, SiLU backward passes
//! - **Softmax**: Warp-parallel softmax gradient with shuffle reductions
//! - **RMSNorm**: RMS normalization gradients (input, gamma)
//! - **LayerNorm**: Layer normalization gradients (input, gamma, beta)
//! - **GEMM**: Matrix multiplication gradients for A and B
//!
//! ## Planned Kernels (Issue #85)
//!
//! - **FlashAttention**: Efficient attention backward with LSE reuse
//!
//! ## Usage
//!
//! ```rust,ignore
//! use trueno_gpu::kernels::backward::{ReluBackwardKernel, GeluBackwardKernel};
//! use trueno_gpu::kernels::Kernel;
//!
//! // Create and emit backward kernels
//! let relu_bwd = ReluBackwardKernel::new(4096);
//! let ptx = relu_bwd.emit_ptx();
//!
//! let gelu_bwd = GeluBackwardKernel::new(4096);
//! let ptx = gelu_bwd.emit_ptx();
//! ```
//!
//! ## Mathematical Guarantees
//!
//! All backward kernels satisfy:
//! 1. **Correctness**: Analytical gradient matches finite-difference within ε < 1e-4
//! 2. **Numerical Stability**: No NaN/Inf for valid inputs
//! 3. **Barrier Safety**: Pass PARITY-114 validation
//! 4. **Determinism**: Same inputs produce same outputs
//!
//! ## Integration with entrenar
//!
//! These kernels enable speedup for fine-tuning in the `entrenar` training library
//! by moving backward passes from CPU (ndarray) to GPU (CUDA).

mod activations;
mod gemm;
mod layer_norm;
mod rms_norm;
mod softmax;

// Re-export backward kernels
pub use activations::{GeluBackwardKernel, ReluBackwardKernel, SiluBackwardKernel};
pub use gemm::{GemmBackwardAKernel, GemmBackwardBKernel};
pub use layer_norm::LayerNormBackwardKernel;
pub use rms_norm::RmsNormBackwardKernel;
pub use softmax::SoftmaxBackwardKernel;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Kernel;

    #[test]
    fn test_all_backward_kernels_compile() {
        // Activation backward
        let relu = ReluBackwardKernel::new(1024);
        let gelu = GeluBackwardKernel::new(1024);
        let silu = SiluBackwardKernel::new(1024);

        assert!(relu.emit_ptx().contains(".entry"));
        assert!(gelu.emit_ptx().contains(".entry"));
        assert!(silu.emit_ptx().contains(".entry"));

        // Softmax backward
        let softmax = SoftmaxBackwardKernel::new(64, 32);
        assert!(softmax.emit_ptx().contains(".entry"));

        // RMSNorm backward
        let rms_norm = RmsNormBackwardKernel::new(64, 32, 1e-6);
        assert!(rms_norm.emit_ptx().contains(".entry"));

        // LayerNorm backward
        let layer_norm = LayerNormBackwardKernel::new(64, 32);
        assert!(layer_norm.emit_ptx().contains(".entry"));

        // GEMM backward
        let gemm_a = GemmBackwardAKernel::new(64, 64, 64);
        assert!(gemm_a.emit_ptx().contains(".entry"));
        let gemm_b = GemmBackwardBKernel::new(64, 64, 64);
        assert!(gemm_b.emit_ptx().contains(".entry"));
    }

    #[test]
    fn test_all_backward_kernels_barrier_safe() {
        let kernels: Vec<Box<dyn Kernel>> = vec![
            Box::new(ReluBackwardKernel::new(1024)),
            Box::new(GeluBackwardKernel::new(1024)),
            Box::new(SiluBackwardKernel::new(1024)),
            Box::new(SoftmaxBackwardKernel::new(64, 32)),
            Box::new(RmsNormBackwardKernel::new(64, 32, 1e-6)),
            Box::new(LayerNormBackwardKernel::new(64, 32)),
            Box::new(GemmBackwardAKernel::new(64, 64, 64)),
            Box::new(GemmBackwardBKernel::new(64, 64, 64)),
        ];

        for kernel in kernels {
            let result = kernel.analyze_barrier_safety();
            assert!(
                result.is_safe,
                "Kernel {} should be barrier-safe: {:?}",
                kernel.name(),
                result.violations
            );
        }
    }
}
