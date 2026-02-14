//! GPU execution engine for batched operations
//!
//! Contains the `execute()`, `execute_operation()`, `execute_unary_op()`,
//! `execute_binary_op()`, and `read()` methods that perform actual GPU dispatch.

use super::{BufferId, GpuCommandBatch, GpuOp};

impl GpuCommandBatch {
    /// Execute all queued operations on GPU
    ///
    /// This performs all GPU operations in a single batch:
    /// 1. Upload all input buffers once
    /// 2. Execute all operations sequentially on GPU
    /// 3. Results stay on GPU until `read()` is called
    pub async fn execute(&mut self) -> Result<(), String> {
        // Step 1: Create GPU buffers for all BufferIds
        for (buffer_id, buffer_info) in &mut self.buffers {
            let size_bytes = (buffer_info.size * std::mem::size_of::<f32>()) as u64;

            let gpu_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Buffer {:?}", buffer_id)),
                size: size_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            buffer_info.gpu_buffer = Some(gpu_buffer);
        }

        // Step 2: Upload initial data to buffers that have it
        for buffer_info in self.buffers.values() {
            if let Some(data) = &buffer_info.data {
                if let Some(gpu_buffer) = &buffer_info.gpu_buffer {
                    self.device
                        .queue
                        .write_buffer(gpu_buffer, 0, bytemuck::cast_slice(data));
                }
            }
        }

        // Step 3: Execute each operation
        for op in &self.operations {
            self.execute_operation(op).await?;
        }

        Ok(())
    }

    /// Execute a single GPU operation
    async fn execute_operation(&self, op: &GpuOp) -> Result<(), String> {
        use crate::backends::gpu::shaders;

        match op {
            GpuOp::Relu { input, output } => {
                let input_info = self.buffers.get(input).ok_or("Invalid input buffer ID")?;
                let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

                let input_buffer = input_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Input buffer not created")?;
                let output_buffer = output_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Output buffer not created")?;

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

            GpuOp::Scale {
                input,
                output,
                scalar,
            } => {
                let input_info = self.buffers.get(input).ok_or("Invalid input buffer ID")?;
                let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

                let input_buffer = input_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Input buffer not created")?;
                let output_buffer = output_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Output buffer not created")?;

                // Create uniform buffer for scalar parameter
                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct ScaleParams {
                    scalar: f32,
                    _padding: [f32; 3], // Uniform buffer alignment
                }

                let params = ScaleParams {
                    scalar: *scalar,
                    _padding: [0.0; 3],
                };

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
                let a_info = self.buffers.get(a).ok_or("Invalid buffer A ID")?;
                let b_info = self.buffers.get(b).ok_or("Invalid buffer B ID")?;
                let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

                let a_buffer = a_info.gpu_buffer.as_ref().ok_or("Buffer A not created")?;
                let b_buffer = b_info.gpu_buffer.as_ref().ok_or("Buffer B not created")?;
                let output_buffer = output_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Output buffer not created")?;

                self.execute_binary_op(
                    shaders::VEC_ADD_SHADER,
                    "Add",
                    a_buffer,
                    b_buffer,
                    output_buffer,
                    a_info.size,
                )
                .await?;
            }

            GpuOp::Mul { a, b, output } => {
                let a_info = self.buffers.get(a).ok_or("Invalid buffer A ID")?;
                let b_info = self.buffers.get(b).ok_or("Invalid buffer B ID")?;
                let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

                let a_buffer = a_info.gpu_buffer.as_ref().ok_or("Buffer A not created")?;
                let b_buffer = b_info.gpu_buffer.as_ref().ok_or("Buffer B not created")?;
                let output_buffer = output_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Output buffer not created")?;

                self.execute_binary_op(
                    shaders::VEC_MUL_SHADER,
                    "Mul",
                    a_buffer,
                    b_buffer,
                    output_buffer,
                    a_info.size,
                )
                .await?;
            }

            GpuOp::Dot { a, b, output } => {
                let a_info = self.buffers.get(a).ok_or("Invalid buffer A ID")?;
                let b_info = self.buffers.get(b).ok_or("Invalid buffer B ID")?;
                let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

                let a_buffer = a_info.gpu_buffer.as_ref().ok_or("Buffer A not created")?;
                let b_buffer = b_info.gpu_buffer.as_ref().ok_or("Buffer B not created")?;
                let output_buffer = output_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Output buffer not created")?;

                self.execute_binary_op(
                    shaders::DOT_PRODUCT_SHADER,
                    "Dot",
                    a_buffer,
                    b_buffer,
                    output_buffer,
                    a_info.size,
                )
                .await?;
            }

            GpuOp::Sigmoid { input, output } => {
                let input_info = self.buffers.get(input).ok_or("Invalid input buffer ID")?;
                let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

                let input_buffer = input_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Input buffer not created")?;
                let output_buffer = output_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Output buffer not created")?;

                self.execute_unary_op::<()>(
                    shaders::SIGMOID_SHADER,
                    "Sigmoid",
                    input_buffer,
                    output_buffer,
                    input_info.size,
                    None,
                )
                .await?;
            }

            GpuOp::Tanh { input, output } => {
                let input_info = self.buffers.get(input).ok_or("Invalid input buffer ID")?;
                let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

                let input_buffer = input_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Input buffer not created")?;
                let output_buffer = output_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Output buffer not created")?;

                self.execute_unary_op::<()>(
                    shaders::TANH_SHADER,
                    "Tanh",
                    input_buffer,
                    output_buffer,
                    input_info.size,
                    None,
                )
                .await?;
            }

            GpuOp::Swish { input, output } => {
                let input_info = self.buffers.get(input).ok_or("Invalid input buffer ID")?;
                let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

                let input_buffer = input_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Input buffer not created")?;
                let output_buffer = output_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Output buffer not created")?;

                self.execute_unary_op::<()>(
                    shaders::SWISH_SHADER,
                    "Swish",
                    input_buffer,
                    output_buffer,
                    input_info.size,
                    None,
                )
                .await?;
            }

            GpuOp::Gelu { input, output } => {
                let input_info = self.buffers.get(input).ok_or("Invalid input buffer ID")?;
                let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

                let input_buffer = input_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Input buffer not created")?;
                let output_buffer = output_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Output buffer not created")?;

                self.execute_unary_op::<()>(
                    shaders::GELU_SHADER,
                    "GELU",
                    input_buffer,
                    output_buffer,
                    input_info.size,
                    None,
                )
                .await?;
            }

            GpuOp::Sub { a, b, output } => {
                let a_info = self.buffers.get(a).ok_or("Invalid buffer A ID")?;
                let b_info = self.buffers.get(b).ok_or("Invalid buffer B ID")?;
                let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

                let a_buffer = a_info.gpu_buffer.as_ref().ok_or("Buffer A not created")?;
                let b_buffer = b_info.gpu_buffer.as_ref().ok_or("Buffer B not created")?;
                let output_buffer = output_info
                    .gpu_buffer
                    .as_ref()
                    .ok_or("Output buffer not created")?;

                self.execute_binary_op(
                    shaders::VEC_SUB_SHADER,
                    "Sub",
                    a_buffer,
                    b_buffer,
                    output_buffer,
                    a_info.size,
                )
                .await?;
            }
        }

        Ok(())
    }

    /// Execute a unary operation (one input, one output)
    async fn execute_unary_op<T: bytemuck::Pod>(
        &self,
        shader_source: &str,
        label: &str,
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        size: usize,
        params: Option<&T>,
    ) -> Result<(), String> {
        // Create shader module
        let shader = self
            .device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{} Shader", label)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        // Create bind group layout entries
        let mut layout_entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        // Add uniform binding if params provided
        if params.is_some() {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        let bind_group_layout =
            self.device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{} Bind Group Layout", label)),
                    entries: &layout_entries,
                });

        // Create uniform buffer if params provided (needs to live through bind group creation)
        let params_buffer = if let Some(params_data) = params {
            let buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Params", label)),
                size: std::mem::size_of::<T>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.device
                .queue
                .write_buffer(&buffer, 0, bytemuck::bytes_of(params_data));

            Some(buffer)
        } else {
            None
        };

        // Create bind group entries
        let mut bind_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ];

        // Add params binding if provided
        if let Some(ref buffer) = params_buffer {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: buffer.as_entire_binding(),
            });
        }

        let bind_group = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Bind Group", label)),
                layout: &bind_group_layout,
                entries: &bind_entries,
            });

        // Create pipeline
        let pipeline_layout =
            self.device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{} Pipeline Layout", label)),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline =
            self.device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{} Pipeline", label)),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Execute
        let mut encoder =
            self.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("{} Encoder", label)),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Pass", label)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (256 threads per workgroup)
            let workgroup_size = 256;
            let num_workgroups = (size as u32).div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        self.device.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    /// Execute a binary operation (two inputs, one output)
    async fn execute_binary_op(
        &self,
        shader_source: &str,
        label: &str,
        a_buffer: &wgpu::Buffer,
        b_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        size: usize,
    ) -> Result<(), String> {
        // Create shader module
        let shader = self
            .device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{} Shader", label)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        // Create bind group layout
        let bind_group_layout =
            self.device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{} Bind Group Layout", label)),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Bind Group", label)),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

        // Create pipeline
        let pipeline_layout =
            self.device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{} Pipeline Layout", label)),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline =
            self.device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("{} Pipeline", label)),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Execute
        let mut encoder =
            self.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("{} Encoder", label)),
                });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Pass", label)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (256 threads per workgroup)
            let workgroup_size = 256;
            let num_workgroups = (size as u32).div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        self.device.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    /// Read buffer data back from GPU
    ///
    /// Must call `execute()` first.
    pub async fn read(&self, buffer_id: BufferId) -> Result<Vec<f32>, String> {
        let buffer_info = self.buffers.get(&buffer_id).ok_or("Invalid buffer ID")?;

        let gpu_buffer = buffer_info
            .gpu_buffer
            .as_ref()
            .ok_or("Buffer not executed yet - call execute() first")?;

        let size_bytes = (buffer_info.size * std::mem::size_of::<f32>()) as u64;

        // Create staging buffer for reading
        let staging_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU buffer to staging buffer
        let mut encoder =
            self.device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Read Encoder"),
                });

        encoder.copy_buffer_to_buffer(gpu_buffer, 0, &staging_buffer, 0, size_bytes);

        self.device.queue.submit(Some(encoder.finish()));

        // Map the staging buffer for reading
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        // Wait for mapping to complete
        receiver
            .receive()
            .await
            .ok_or("Failed to receive mapping result")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        // Read data from mapped buffer
        let data = {
            let mapped_range = buffer_slice.get_mapped_range();
            let float_data: &[f32] = bytemuck::cast_slice(&mapped_range);
            float_data.to_vec()
        };

        staging_buffer.unmap();

        Ok(data)
    }
}
