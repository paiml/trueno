//! Shader dispatch infrastructure for GPU compute operations
//!
//! Contains `execute_unary_op()` and `execute_binary_op()` which handle
//! the wgpu pipeline setup, bind group creation, and workgroup dispatch
//! for unary (one input) and binary (two input) operations.

use super::super::GpuCommandBatch;

/// Workgroup size for compute shaders (threads per workgroup)
const WORKGROUP_SIZE: u32 = 256;

impl GpuCommandBatch {
    /// Execute a unary operation (one input, one output)
    pub(crate) async fn execute_unary_op<T: bytemuck::Pod>(
        &self,
        shader_source: &str,
        label: &str,
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        size: usize,
        params: Option<&T>,
    ) -> Result<(), String> {
        // Create shader module
        let shader = self.device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
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
            self.device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

            self.device.queue.write_buffer(&buffer, 0, bytemuck::bytes_of(params_data));

            Some(buffer)
        } else {
            None
        };

        // Create bind group entries
        let mut bind_entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
        ];

        // Add params binding if provided
        if let Some(ref buffer) = params_buffer {
            bind_entries
                .push(wgpu::BindGroupEntry { binding: 2, resource: buffer.as_entire_binding() });
        }

        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", label)),
            layout: &bind_group_layout,
            entries: &bind_entries,
        });

        // Create pipeline
        let pipeline_layout =
            self.device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{} Pipeline Layout", label)),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline =
            self.device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{} Pipeline", label)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Execute
        let mut encoder =
            self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{} Encoder", label)),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Pass", label)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups
            let num_workgroups = (size as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        self.device.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    /// Execute a matrix multiplication operation
    ///
    /// Uses a 4-binding layout: A (storage, read), B (storage, read),
    /// C (storage, read_write), dims (uniform). Dispatches 16×16 workgroups.
    pub(crate) async fn execute_matmul_op(
        &self,
        shader_source: &str,
        label: &str,
        a: &super::super::BufferId,
        b: &super::super::BufferId,
        output: &super::super::BufferId,
        m: u32,
        k: u32,
        n: u32,
    ) -> Result<(), String> {
        let a_info = self.buffers.get(a).ok_or("Invalid buffer A ID")?;
        let b_info = self.buffers.get(b).ok_or("Invalid buffer B ID")?;
        let output_info = self.buffers.get(output).ok_or("Invalid output buffer ID")?;

        let a_buffer = a_info.gpu_buffer.as_ref().ok_or("Buffer A not created")?;
        let b_buffer = b_info.gpu_buffer.as_ref().ok_or("Buffer B not created")?;
        let output_buffer = output_info.gpu_buffer.as_ref().ok_or("Output buffer not created")?;

        // Create shader module
        let shader = self.device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} Shader", label)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Matmul uses 4 bindings: A (read), B (read), C (read_write), dims (uniform)
        let bind_group_layout =
            self.device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create dimensions uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MatmulDims {
            m: u32,
            k: u32,
            n: u32,
            _pad: u32, // Uniform buffer alignment (16-byte)
        }

        let dims = MatmulDims { m, k, n, _pad: 0 };
        let dims_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Dims", label)),
            size: std::mem::size_of::<MatmulDims>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device.queue.write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));

        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", label)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buffer.as_entire_binding() },
            ],
        });

        // Create pipeline
        let pipeline_layout =
            self.device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{} Pipeline Layout", label)),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline =
            self.device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{} Pipeline", label)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Execute with 16×16 workgroups (matching MATMUL_SHADER)
        let mut encoder =
            self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{} Encoder", label)),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Pass", label)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // 16×16 workgroup size in shader → dispatch ceil(M/16) × ceil(N/16) workgroups
            let wg_x = m.div_ceil(16);
            let wg_y = n.div_ceil(16);
            compute_pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        self.device.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    /// Execute a binary operation (two inputs, one output)
    pub(crate) async fn execute_binary_op(
        &self,
        shader_source: &str,
        label: &str,
        a_buffer: &wgpu::Buffer,
        b_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        size: usize,
    ) -> Result<(), String> {
        // Create shader module
        let shader = self.device.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} Shader", label)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout =
            self.device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let bind_group = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", label)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
            ],
        });

        // Create pipeline
        let pipeline_layout =
            self.device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{} Pipeline Layout", label)),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline =
            self.device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{} Pipeline", label)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Execute
        let mut encoder =
            self.device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{} Encoder", label)),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Pass", label)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups
            let num_workgroups = (size as u32).div_ceil(WORKGROUP_SIZE);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        self.device.queue.submit(Some(encoder.finish()));

        Ok(())
    }
}
