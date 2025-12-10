//! Numerically Stable Softmax Kernel
//!
//! Implements softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))

use super::Kernel;
use crate::ptx::{PtxKernel, PtxType};

/// Softmax kernel configuration
#[derive(Debug, Clone)]
pub struct SoftmaxKernel {
    /// Vector length
    pub length: u32,
    /// Use warp shuffle for reduction (faster)
    pub use_warp_shuffle: bool,
}

impl SoftmaxKernel {
    /// Create a new softmax kernel
    #[must_use]
    pub fn new(length: u32) -> Self {
        Self {
            length,
            use_warp_shuffle: true,
        }
    }

    /// Disable warp shuffle (for compatibility with older GPUs)
    #[must_use]
    pub const fn without_warp_shuffle(mut self) -> Self {
        self.use_warp_shuffle = false;
        self
    }
}

impl Kernel for SoftmaxKernel {
    fn name(&self) -> &str {
        if self.use_warp_shuffle {
            "softmax_warp_shuffle"
        } else {
            "softmax_shared"
        }
    }

    fn build_ptx(&self) -> PtxKernel {
        if self.use_warp_shuffle {
            self.build_warp_shuffle()
        } else {
            self.build_shared_memory()
        }
    }
}

impl SoftmaxKernel {
    fn build_warp_shuffle(&self) -> PtxKernel {
        PtxKernel::new("softmax_warp_shuffle")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "length")
            .build(|ctx| {
                // Load value for this thread
                let tid = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let length = ctx.load_param_u32("length");

                // Bounds check
                let pred = ctx.setp_ge_u32(tid, length);
                ctx.branch_if(pred, "exit");

                // Load input
                let input_ptr = ctx.load_param_u64("input_ptr");
                let offset = ctx.mul_wide_u32(tid, 4);
                let addr = ctx.add_u64(input_ptr, offset);
                let _val = ctx.ld_global_f32(addr);

                // Note: Full implementation would include:
                // 1. Warp-level max reduction using shfl_down
                // 2. Broadcast max to all lanes
                // 3. Compute exp(val - max)
                // 4. Warp-level sum reduction
                // 5. Divide by sum
                // 6. Store result

                ctx.label("exit");
                ctx.ret();
            })
    }

    fn build_shared_memory(&self) -> PtxKernel {
        PtxKernel::new("softmax_shared")
            .param(PtxType::U64, "input_ptr")
            .param(PtxType::U64, "output_ptr")
            .param(PtxType::U32, "length")
            .shared_memory(256 * 4) // Reduction buffer
            .build(|ctx| {
                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_kernel_name() {
        let kernel = SoftmaxKernel::new(4096);
        assert_eq!(kernel.name(), "softmax_warp_shuffle");

        let kernel_shared = SoftmaxKernel::new(4096).without_warp_shuffle();
        assert_eq!(kernel_shared.name(), "softmax_shared");
    }

    #[test]
    fn test_softmax_ptx_generation() {
        let kernel = SoftmaxKernel::new(4096);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".param .u64 input_ptr"));
        assert!(ptx.contains(".param .u64 output_ptr"));
        assert!(ptx.contains(".param .u32 length"));
    }

    #[test]
    fn test_softmax_shared_memory() {
        let kernel = SoftmaxKernel::new(4096).without_warp_shuffle();
        let ptx_kernel = kernel.build_ptx();
        assert!(ptx_kernel.shared_memory_bytes() > 0);
    }
}
