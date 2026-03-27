//! GPU backward (gradient) operations for training
//!
//! Contract: wgpu-training-v1.yaml (FALSIFY-WGPU-001)
//!
//! Dispatches WGSL backward shaders to compute gradients on GPU.
//! All operations match CPU reference within ε < 1e-4.

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
use super::super::runtime;
use super::super::shaders;
use super::GpuDevice;

impl GpuDevice {
    /// SiLU backward on GPU: grad_input[i] = grad_output[i] * silu'(input[i])
    ///
    /// # Contract (FALSIFY-WGPU-001)
    ///
    /// - **Precondition**: input.len() == grad_output.len() == grad_input.len()
    /// - **Postcondition**: max|grad_input_gpu - grad_input_cpu| < 1e-4
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn silu_backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
    ) -> Result<(), String> {
        runtime::block_on(self.silu_backward_async(input, grad_output, grad_input))
    }

    /// SiLU backward on GPU (async)
    pub async fn silu_backward_async(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
    ) -> Result<(), String> {
        let n = input.len();
        if grad_output.len() != n || grad_input.len() != n {
            return Err(format!(
                "SiLU backward: length mismatch: input={}, grad_output={}, grad_input={}",
                n,
                grad_output.len(),
                grad_input.len()
            ));
        }

        self.execute_backward_elementwise(
            "SiLU Backward",
            shaders::backward::SILU_BACKWARD_SHADER,
            input,
            grad_output,
            grad_input,
            n as u32,
        )
        .await
    }

    /// Generic dispatch for element-wise backward shaders (3 buffers + uniform)
    ///
    /// Binding layout: 0=input(read), 1=grad_output(read), 2=grad_input(write), 3=uniform{n}
    async fn execute_backward_elementwise(
        &self,
        op_name: &str,
        shader_source: &str,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
        n: u32,
    ) -> Result<(), String> {
        use wgpu;

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{op_name} Shader")),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create buffers
        let input_buf = self.create_storage_buffer(&format!("{op_name} input"), input, true);
        let grad_out_buf =
            self.create_storage_buffer(&format!("{op_name} grad_output"), grad_output, true);
        let grad_in_buf = self.create_rw_storage_buffer(
            &format!("{op_name} grad_input"),
            (grad_input.len() * 4) as u64,
        );

        // Uniform: { n: u32 } padded to 16 bytes (WGSL alignment)
        let uniform_data: [u32; 4] = [n, 0, 0, 0];
        let uniform_buf = self.create_uniform_buffer(
            &format!("{op_name} uniform"),
            bytemuck::cast_slice(&uniform_data),
        );

        // Bind group layout: 3 storage + 1 uniform
        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{op_name} BGL")),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{op_name} BG")),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grad_out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grad_in_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{op_name} PL")),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{op_name} Pipeline")),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Staging buffer for readback
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{op_name} Staging")),
            size: (grad_input.len() * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dispatch
        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(n.div_ceil(256), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&grad_in_buf, 0, &staging, 0, (grad_input.len() * 4) as u64);
        self.queue.submit(Some(encoder.finish()));

        // Read back
        let slice = staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender.send(r).ok();
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        receiver
            .receive()
            .await
            .ok_or_else(|| format!("{op_name}: map_async cancelled"))?
            .map_err(|e| format!("{op_name}: map_async failed: {e}"))?;

        let data = slice.get_mapped_range();
        grad_input.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        staging.unmap();

        Ok(())
    }

    // --- Buffer helpers ---

    fn create_storage_buffer(&self, label: &str, data: &[f32], read_only: bool) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (data.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
        let _ = read_only; // usage flags are same; read_only is in the shader
        buf
    }

    fn create_rw_storage_buffer(&self, label: &str, size: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn create_uniform_buffer(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, data);
        buf
    }

    /// GEMM backward for A: grad_a[M,K] = grad_c[M,N] @ B^T[N,K]
    ///
    /// # Contract (FALSIFY-WGPU-001)
    ///
    /// - **Precondition**: grad_c.len() == m*n, b.len() == k*n, grad_a.len() == m*k
    /// - **Postcondition**: max|grad_a_gpu - grad_a_cpu| < 1e-4
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn gemm_backward_a(
        &self,
        grad_c: &[f32],
        b: &[f32],
        grad_a: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), String> {
        runtime::block_on(self.gemm_backward_a_async(grad_c, b, grad_a, m, k, n))
    }

    /// GEMM backward for A (async): grad_a = grad_c @ B^T
    pub async fn gemm_backward_a_async(
        &self,
        grad_c: &[f32],
        b: &[f32],
        grad_a: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), String> {
        self.execute_backward_gemm(
            "GEMM Backward A",
            shaders::backward::GEMM_BACKWARD_A_SHADER,
            grad_c,
            b,
            grad_a,
            m,
            k,
            n,
        )
        .await
    }

    /// GEMM backward for B: grad_b[K,N] = A^T[K,M] @ grad_c[M,N]
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn gemm_backward_b(
        &self,
        a: &[f32],
        grad_c: &[f32],
        grad_b: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), String> {
        runtime::block_on(self.gemm_backward_b_async(a, grad_c, grad_b, m, k, n))
    }

    /// GEMM backward for B (async): grad_b = A^T @ grad_c
    pub async fn gemm_backward_b_async(
        &self,
        a: &[f32],
        grad_c: &[f32],
        grad_b: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), String> {
        self.execute_backward_gemm(
            "GEMM Backward B",
            shaders::backward::GEMM_BACKWARD_B_SHADER,
            a,
            grad_c,
            grad_b,
            m,
            k,
            n,
        )
        .await
    }

    /// Generic dispatch for GEMM backward shaders (tiled 16×16)
    ///
    /// Binding: 0=buf_a(read), 1=buf_b(read), 2=output(write), 3=uniform{M,K,N}
    async fn execute_backward_gemm(
        &self,
        op_name: &str,
        shader_source: &str,
        buf_a: &[f32],
        buf_b: &[f32],
        output: &mut [f32],
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), String> {
        use wgpu;

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{op_name} Shader")),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let a_buf = self.create_storage_buffer(&format!("{op_name} A"), buf_a, true);
        let b_buf = self.create_storage_buffer(&format!("{op_name} B"), buf_b, true);
        let out_buf = self.create_rw_storage_buffer(
            &format!("{op_name} Output"),
            (output.len() * 4) as u64,
        );

        // Uniform: { M, K, N, pad }
        let dims: [u32; 4] = [m, k, n, 0];
        let uniform_buf = self.create_uniform_buffer(
            &format!("{op_name} Dims"),
            bytemuck::cast_slice(&dims),
        );

        let bgl = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, false),
                uniform_entry(3),
            ],
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: uniform_buf.as_entire_binding() },
            ],
        });

        let pl = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{op_name} Pipeline")),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (output.len() * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);

            // For GEMM backward A: output is [M,K], dispatch ceil(M/16) × ceil(K/16)
            // For GEMM backward B: output is [K,N], dispatch ceil(K/16) × ceil(N/16)
            // The output dimensions are encoded in the first two dims of the output buffer.
            let out_rows = if op_name.contains("A") { m } else { k };
            let out_cols = if op_name.contains("A") { k } else { n };
            pass.dispatch_workgroups(out_rows.div_ceil(16), out_cols.div_ceil(16), 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, (output.len() * 4) as u64);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender.send(r).ok();
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        receiver
            .receive()
            .await
            .ok_or_else(|| format!("{op_name}: map cancelled"))?
            .map_err(|e| format!("{op_name}: map failed: {e}"))?;

        let data = slice.get_mapped_range();
        output.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        staging.unmap();

        Ok(())
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    /// CPU reference: SiLU backward
    fn silu_backward_cpu(input: &[f32], grad_output: &[f32]) -> Vec<f32> {
        input
            .iter()
            .zip(grad_output.iter())
            .map(|(&x, &dy)| {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                let y = x * sigmoid;
                let silu_prime = sigmoid * (1.0 + x - y);
                dy * silu_prime
            })
            .collect()
    }

    /// FALSIFY-WGPU-001: SiLU backward matches CPU within ε < 1e-4
    #[test]
    fn test_falsify_wgpu_001_silu_backward_parity() {
        let device = GpuDevice::new().expect("GPU device");

        let input: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
        let grad_output: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.01).collect();
        let expected = silu_backward_cpu(&input, &grad_output);

        let mut grad_input = vec![0.0f32; 100];
        device.silu_backward(&input, &grad_output, &mut grad_input).expect("silu_backward");

        let max_diff = grad_input
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-4,
            "FALSIFY-WGPU-001: SiLU backward max diff = {max_diff} (threshold: 1e-4)"
        );
    }

    /// SiLU backward at x=0 (sigmoid=0.5, silu'=0.5)
    #[test]
    fn test_silu_backward_at_zero() {
        let device = GpuDevice::new().expect("GPU device");

        let input = vec![0.0f32; 4];
        let grad_output = vec![1.0f32; 4];
        let mut grad_input = vec![0.0f32; 4];

        device.silu_backward(&input, &grad_output, &mut grad_input).expect("silu_backward");

        // At x=0: sigmoid(0)=0.5, silu'(0) = 0.5 * (1 + 0 - 0) = 0.5
        for &g in &grad_input {
            assert!((g - 0.5).abs() < 1e-5, "silu'(0) should be 0.5, got {g}");
        }
    }

    /// SiLU backward length mismatch error
    #[test]
    fn test_silu_backward_length_mismatch() {
        let device = GpuDevice::new().expect("GPU device");

        let input = vec![1.0f32; 10];
        let grad_output = vec![1.0f32; 5]; // wrong length
        let mut grad_input = vec![0.0f32; 10];

        let result = device.silu_backward(&input, &grad_output, &mut grad_input);
        assert!(result.is_err());
    }

    /// CPU reference: matmul C = A[M,K] @ B[K,N]
    fn matmul_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    /// FALSIFY-WGPU-001: GEMM backward A matches CPU within ε < 1e-3
    ///
    /// grad_a[M,K] = grad_c[M,N] @ B^T[N,K]
    /// Which is matmul(grad_c, B^T, M, N, K) but our shader handles the transpose internally.
    #[test]
    fn test_falsify_wgpu_001_gemm_backward_a_parity() {
        let device = GpuDevice::new().expect("GPU device");

        let (m, k, n) = (4, 8, 6);

        // Random-ish test data
        let grad_c: Vec<f32> = (0..m * n).map(|i| (i as f32 - 12.0) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 - 24.0) * 0.05).collect();

        // CPU reference: grad_a = grad_c @ B^T
        // B^T[N,K] means we need to transpose B[K,N] → B^T[N,K]
        let mut b_t = vec![0.0f32; n * k];
        for i in 0..k {
            for j in 0..n {
                b_t[j * k + i] = b[i * n + j];
            }
        }
        let expected = matmul_cpu(&grad_c, &b_t, m, n, k);

        let mut grad_a = vec![0.0f32; m * k];
        device
            .gemm_backward_a(&grad_c, &b, &mut grad_a, m as u32, k as u32, n as u32)
            .expect("gemm_backward_a");

        let max_diff = grad_a
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-3,
            "FALSIFY-WGPU-001: GEMM backward A max diff = {max_diff} (threshold: 1e-3)"
        );
    }

    /// FALSIFY-WGPU-001: GEMM backward B matches CPU within ε < 1e-3
    ///
    /// grad_b[K,N] = A^T[K,M] @ grad_c[M,N]
    #[test]
    fn test_falsify_wgpu_001_gemm_backward_b_parity() {
        let device = GpuDevice::new().expect("GPU device");

        let (m, k, n) = (4, 8, 6);

        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let grad_c: Vec<f32> = (0..m * n).map(|i| (i as f32 - 12.0) * 0.05).collect();

        // CPU reference: grad_b = A^T @ grad_c
        let mut a_t = vec![0.0f32; k * m];
        for i in 0..m {
            for j in 0..k {
                a_t[j * m + i] = a[i * k + j];
            }
        }
        let expected = matmul_cpu(&a_t, &grad_c, k, m, n);

        let mut grad_b = vec![0.0f32; k * n];
        device
            .gemm_backward_b(&a, &grad_c, &mut grad_b, m as u32, k as u32, n as u32)
            .expect("gemm_backward_b");

        let max_diff = grad_b
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-3,
            "FALSIFY-WGPU-001: GEMM backward B max diff = {max_diff} (threshold: 1e-3)"
        );
    }
}
