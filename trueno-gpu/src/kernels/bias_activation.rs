//! Bias + Activation Epilogue Kernel
//!
//! Element-wise kernel for applying bias and activation functions to GEMM output.
//! Used as an epilogue after matrix multiplication.
//!
//! # Operations
//! - Add bias: `output[i] += bias[i % bias_size]`
//! - ReLU: `output[i] = max(0, output[i])`
//! - GELU: `output[i] = x * sigmoid(1.702 * x)`
//!
//! # Example
//! ```
//! use trueno_gpu::kernels::{BiasActivationKernel, Kernel};
//!
//! let kernel = BiasActivationKernel::new(1024, 64)
//!     .with_relu();
//! let ptx = kernel.emit_ptx();
//! assert!(ptx.contains("bias_activation"));
//! ```

use crate::ptx::{PtxKernel, PtxType};

/// Activation function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Activation {
    /// No activation (identity)
    #[default]
    None,
    /// ReLU: max(0, x)
    ReLU,
    /// GELU approximation: x * sigmoid(1.702 * x)
    GELU,
}

/// Bias + Activation kernel configuration
#[derive(Debug, Clone)]
pub struct BiasActivationKernel {
    /// Total number of elements
    n: u32,
    /// Size of bias vector (output is bias[i % bias_size])
    bias_size: u32,
    /// Activation function
    activation: Activation,
}

impl BiasActivationKernel {
    /// Create a new bias + activation kernel
    ///
    /// # Arguments
    /// * `n` - Total number of output elements
    /// * `bias_size` - Size of bias vector (typically output dimension)
    #[must_use]
    pub fn new(n: u32, bias_size: u32) -> Self {
        Self {
            n,
            bias_size,
            activation: Activation::None,
        }
    }

    /// Add ReLU activation
    #[must_use]
    pub fn with_relu(mut self) -> Self {
        self.activation = Activation::ReLU;
        self
    }

    /// Add GELU activation
    #[must_use]
    pub fn with_gelu(mut self) -> Self {
        self.activation = Activation::GELU;
        self
    }

    /// Set activation function
    #[must_use]
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }
}

impl super::Kernel for BiasActivationKernel {
    fn name(&self) -> &str {
        "bias_activation"
    }

    fn build_ptx(&self) -> PtxKernel {
        let activation = self.activation;
        let bias_size = self.bias_size;

        PtxKernel::new("bias_activation")
            .param(PtxType::U64, "output")
            .param(PtxType::U64, "bias")
            .param(PtxType::U32, "n")
            .build(|ctx| {
                // Thread index calculation: global_id = blockIdx.x * blockDim.x + threadIdx.x
                let ctaid_x = ctx.special_reg(crate::ptx::PtxReg::CtaIdX);
                let ntid_x = ctx.special_reg(crate::ptx::PtxReg::NtidX);
                let tid_x = ctx.special_reg(crate::ptx::PtxReg::TidX);
                let global_id = ctx.mad_lo_u32(ctaid_x, ntid_x, tid_x);

                // Bounds check: if (global_id >= n) return
                let n_param = ctx.load_param_u32("n");
                let out_of_bounds = ctx.setp_ge_u32(global_id, n_param);
                ctx.branch_if(out_of_bounds, "exit");

                // Load output[global_id]
                let output_ptr = ctx.load_param_u64("output");
                let offset = ctx.mul_wide_u32(global_id, 4); // sizeof(f32)
                let addr = ctx.add_u64(output_ptr, offset);
                let value = ctx.ld_global_f32(addr);

                // Load bias[global_id % bias_size] - bias_size is baked into kernel
                let bias_ptr = ctx.load_param_u64("bias");
                let bias_idx = ctx.rem_u32(global_id, bias_size);
                let bias_offset = ctx.mul_wide_u32(bias_idx, 4);
                let bias_addr = ctx.add_u64(bias_ptr, bias_offset);
                let bias_val = ctx.ld_global_f32(bias_addr);

                // Add bias: result = value + bias
                let result = ctx.add_f32(value, bias_val);

                // Apply activation function
                let activated = match activation {
                    Activation::None => result,
                    Activation::ReLU => {
                        // ReLU: max(0, x)
                        let zero = ctx.mov_f32_imm(0.0);
                        ctx.max_f32(result, zero)
                    }
                    Activation::GELU => {
                        // GELU approximation: x * sigmoid(1.702 * x)
                        // sigmoid(y) = 1 / (1 + exp(-y))
                        // exp(-y) ≈ 2^(-y * log2(e))
                        let coeff = ctx.mov_f32_imm(1.702);
                        let scaled = ctx.mul_f32(result, coeff);

                        // Compute exp(-scaled) via ex2: negate by subtracting from 0
                        let zero = ctx.mov_f32_imm(0.0);
                        let neg_scaled = ctx.sub_f32(zero, scaled);
                        let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);
                        let scaled_for_ex2 = ctx.mul_f32(neg_scaled, log2_e);
                        let exp_val = ctx.ex2_f32(scaled_for_ex2);

                        // sigmoid = 1 / (1 + exp(-scaled))
                        let one = ctx.mov_f32_imm(1.0);
                        let denom = ctx.add_f32(one, exp_val);
                        let sigmoid = ctx.div_f32(one, denom);

                        // GELU = x * sigmoid(1.702 * x)
                        ctx.mul_f32(result, sigmoid)
                    }
                };

                // Store result back to output[global_id]
                ctx.st_global_f32(addr, activated);

                ctx.label("exit");
                ctx.ret();
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Kernel;

    #[test]
    fn test_bias_activation_default_config() {
        let kernel = BiasActivationKernel::new(1024, 64);
        assert_eq!(kernel.n, 1024);
        assert_eq!(kernel.bias_size, 64);
        assert_eq!(kernel.activation, Activation::None);
    }

    #[test]
    fn test_bias_activation_with_relu() {
        let kernel = BiasActivationKernel::new(1024, 64).with_relu();
        assert_eq!(kernel.activation, Activation::ReLU);
    }

    #[test]
    fn test_bias_activation_with_gelu() {
        let kernel = BiasActivationKernel::new(1024, 64).with_gelu();
        assert_eq!(kernel.activation, Activation::GELU);
    }

    #[test]
    fn test_bias_activation_kernel_name() {
        let kernel = BiasActivationKernel::new(1024, 64);
        assert_eq!(kernel.name(), "bias_activation");
    }

    #[test]
    fn test_bias_activation_ptx_generation() {
        let kernel = BiasActivationKernel::new(1024, 64);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains(".version 8.0"), "Missing PTX version");
        assert!(ptx.contains(".target sm_89"), "Missing target");
        assert!(
            ptx.contains(".visible .entry bias_activation"),
            "Missing entry point"
        );
        assert!(ptx.contains(".param .u64 output"), "Missing output param");
        assert!(ptx.contains(".param .u64 bias"), "Missing bias param");
        assert!(ptx.contains(".param .u32 n"), "Missing n param");
        // bias_size is baked into kernel at generation time (for efficiency)
    }

    #[test]
    fn test_bias_activation_relu_ptx() {
        let kernel = BiasActivationKernel::new(1024, 64).with_relu();
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains("max.f32"), "ReLU should use max.f32");
    }

    #[test]
    fn test_bias_activation_gelu_ptx() {
        let kernel = BiasActivationKernel::new(1024, 64).with_gelu();
        let ptx = kernel.emit_ptx();

        assert!(
            ptx.contains("ex2.approx") || ptx.contains("ex2.f32"),
            "GELU should use ex2 for exp"
        );
        assert!(
            ptx.contains("div.rn.f32") || ptx.contains("div.f32"),
            "GELU should use div for sigmoid reciprocal"
        );
    }

    #[test]
    fn test_bias_activation_contains_bias_addition() {
        let kernel = BiasActivationKernel::new(1024, 64);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains("add.f32"), "Should contain bias addition");
        assert!(
            ptx.contains("rem.u32"),
            "Should contain modulo for bias indexing"
        );
    }

    #[test]
    fn test_bias_activation_bounds_check() {
        let kernel = BiasActivationKernel::new(1024, 64);
        let ptx = kernel.emit_ptx();

        assert!(ptx.contains("setp.ge.u32"), "Should have bounds check");
    }

}

#[cfg(test)]
mod property_tests {
    use super::*;
    use crate::kernels::Kernel;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn bias_activation_always_valid(n in 64u32..8192, bias_size in 16u32..512) {
            let kernel = BiasActivationKernel::new(n, bias_size);
            let ptx = kernel.emit_ptx();

            prop_assert!(ptx.contains(".version"), "Missing PTX version");
            prop_assert!(ptx.contains(".entry"), "Missing entry point");
            prop_assert!(ptx.contains("bias_activation"), "Missing kernel name");
        }

        #[test]
        fn activation_variants_produce_valid_ptx(n in 64u32..4096, bias_size in 16u32..256) {
            for activation in [Activation::None, Activation::ReLU, Activation::GELU] {
                let kernel = BiasActivationKernel::new(n, bias_size)
                    .with_activation(activation);
                let ptx = kernel.emit_ptx();

                prop_assert!(ptx.contains(".version"), "Missing PTX version for {:?}", activation);
                prop_assert!(ptx.contains("bias_activation"), "Missing kernel name for {:?}", activation);
            }
        }
    }
}
