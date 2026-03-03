//! Operation dispatch for GPU batch execution
//!
//! Contains `execute_operation()` which routes each `GpuOp` variant to the
//! appropriate shader via `execute_unary_op()` or `execute_binary_op()`.

use super::super::{GpuCommandBatch, GpuOp};

impl GpuCommandBatch {
    /// Execute a single GPU operation
    pub(crate) async fn execute_operation(&self, op: &GpuOp) -> Result<(), String> {
        use crate::backends::gpu::shaders;

        match op {
            GpuOp::Relu { input, output } => {
                let input_info = self.buffers.get(input).ok_or("Invalid input buffer ID")?;
                let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

                let input_buffer =
                    input_info.gpu_buffer.as_ref().ok_or("Input buffer not created")?;
                let output_buffer =
                    output_info.gpu_buffer.as_ref().ok_or("Output buffer not created")?;

                self.execute_unary_op::<()>(
                    shaders::RELU_SHADER,
                    "ReLU",
                    input_buffer,
                    output_buffer,
                    input_info.size,
                    None,
                )
                .await?;
            }

            GpuOp::Scale { input, output, scalar } => {
                let input_info = self.buffers.get(input).ok_or("Invalid input buffer ID")?;
                let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

                let input_buffer =
                    input_info.gpu_buffer.as_ref().ok_or("Input buffer not created")?;
                let output_buffer =
                    output_info.gpu_buffer.as_ref().ok_or("Output buffer not created")?;

                // Create uniform buffer for scalar parameter
                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct ScaleParams {
                    scalar: f32,
                    _padding: [f32; 3], // Uniform buffer alignment
                }

                let params = ScaleParams { scalar: *scalar, _padding: [0.0; 3] };

                self.execute_unary_op(
                    shaders::SCALE_SHADER,
                    "Scale",
                    input_buffer,
                    output_buffer,
                    input_info.size,
                    Some(&params),
                )
                .await?;
            }

            GpuOp::Add { a, b, output } => {
                self.execute_binary_op_for(shaders::VEC_ADD_SHADER, "Add", a, b, output).await?;
            }

            GpuOp::Mul { a, b, output } => {
                self.execute_binary_op_for(shaders::VEC_MUL_SHADER, "Mul", a, b, output).await?;
            }

            GpuOp::Dot { a, b, output } => {
                self.execute_binary_op_for(shaders::DOT_PRODUCT_SHADER, "Dot", a, b, output)
                    .await?;
            }

            GpuOp::Sigmoid { input, output } => {
                self.execute_unary_op_for(shaders::SIGMOID_SHADER, "Sigmoid", input, output)
                    .await?;
            }

            GpuOp::Tanh { input, output } => {
                self.execute_unary_op_for(shaders::TANH_SHADER, "Tanh", input, output).await?;
            }

            GpuOp::Swish { input, output } => {
                self.execute_unary_op_for(shaders::SWISH_SHADER, "Swish", input, output).await?;
            }

            GpuOp::Gelu { input, output } => {
                self.execute_unary_op_for(shaders::GELU_SHADER, "GELU", input, output).await?;
            }

            GpuOp::Sub { a, b, output } => {
                self.execute_binary_op_for(shaders::VEC_SUB_SHADER, "Sub", a, b, output).await?;
            }

            GpuOp::Matmul { a, b, output, m, k, n } => {
                self.execute_matmul_op(shaders::MATMUL_SHADER, "Matmul", a, b, output, *m, *k, *n)
                    .await?;
            }
        }

        Ok(())
    }

    /// Helper to extract buffers and dispatch a unary operation (no params)
    async fn execute_unary_op_for(
        &self,
        shader_source: &str,
        label: &str,
        input: &super::super::BufferId,
        output: &super::super::BufferId,
    ) -> Result<(), String> {
        let input_info = self.buffers.get(input).ok_or("Invalid input buffer ID")?;
        let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

        let input_buffer = input_info.gpu_buffer.as_ref().ok_or("Input buffer not created")?;
        let output_buffer = output_info.gpu_buffer.as_ref().ok_or("Output buffer not created")?;

        self.execute_unary_op::<()>(
            shader_source,
            label,
            input_buffer,
            output_buffer,
            input_info.size,
            None,
        )
        .await
    }

    /// Helper to extract buffers and dispatch a binary operation
    async fn execute_binary_op_for(
        &self,
        shader_source: &str,
        label: &str,
        a: &super::super::BufferId,
        b: &super::super::BufferId,
        output: &super::super::BufferId,
    ) -> Result<(), String> {
        let a_info = self.buffers.get(a).ok_or("Invalid buffer A ID")?;
        let b_info = self.buffers.get(b).ok_or("Invalid buffer B ID")?;
        let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

        let a_buffer = a_info.gpu_buffer.as_ref().ok_or("Buffer A not created")?;
        let b_buffer = b_info.gpu_buffer.as_ref().ok_or("Buffer B not created")?;
        let output_buffer = output_info.gpu_buffer.as_ref().ok_or("Output buffer not created")?;

        self.execute_binary_op(shader_source, label, a_buffer, b_buffer, output_buffer, a_info.size)
            .await
    }
}
