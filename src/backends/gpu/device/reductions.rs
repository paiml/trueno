//! GPU reduction operations
//!
//! Parallel max/sum reductions and 2D tiled reductions (sum/max/min).

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
use super::super::runtime;
use super::super::shaders;
use super::GpuDevice;

impl GpuDevice {
    /// Helper: Parallel max reduction
    pub(super) async fn reduce_max(&self, input: &[f32]) -> Result<f32, String> {
        let len = input.len();
        let workgroup_size = 256;
        let num_workgroups = (len as u32).div_ceil(workgroup_size);

        // Create shader module
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Max Reduction Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::MAX_REDUCTION_SHADER.into()),
            });

        // Create input buffer
        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Max Reduction Input"),
            size: std::mem::size_of_val(input) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Result buffer for partial maxes
        let partial_results = vec![f32::NEG_INFINITY; num_workgroups as usize];
        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Max Partial Results"),
            size: std::mem::size_of_val(partial_results.as_slice()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.queue
            .write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));

        // Create bind group layout
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Max Reduction Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Max Reduction Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: result_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Max Reduction Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Max Reduction Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Max Reduction Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Max Reduction Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Create staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Max Staging Buffer"),
            size: std::mem::size_of_val(partial_results.as_slice()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &result_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of_val(partial_results.as_slice()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        // Poll device to ensure GPU work completes and callbacks are invoked
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        receiver
            .receive()
            .await
            .ok_or("Channel receive failed")?
            .map_err(|e| format!("Buffer map failed: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        // Final reduction on CPU
        Ok(result.iter().copied().fold(f32::NEG_INFINITY, f32::max))
    }

    /// Helper: Parallel sum reduction
    pub(super) async fn reduce_sum(&self, input: &[f32]) -> Result<f32, String> {
        let len = input.len();
        let workgroup_size = 256;
        let num_workgroups = (len as u32).div_ceil(workgroup_size);

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Sum Reduction Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::SUM_REDUCTION_SHADER.into()),
            });

        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sum Reduction Input"),
            size: std::mem::size_of_val(input) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let partial_results = vec![0.0f32; num_workgroups as usize];
        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sum Partial Results"),
            size: std::mem::size_of_val(partial_results.as_slice()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.queue
            .write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Sum Reduction Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sum Reduction Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: result_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sum Reduction Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sum Reduction Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sum Reduction Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sum Reduction Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sum Staging Buffer"),
            size: std::mem::size_of_val(partial_results.as_slice()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &result_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of_val(partial_results.as_slice()) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        // Poll device to ensure GPU work completes and callbacks are invoked
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        receiver
            .receive()
            .await
            .ok_or("Channel receive failed")?
            .map_err(|e| format!("Buffer map failed: {:?}", e))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        // Final reduction on CPU
        Ok(result.iter().sum())
    }

    /// 2D Tiled Sum Reduction on GPU (sync, native only)
    ///
    /// Uses 16x16 workgroups for efficient parallel reduction with
    /// optimal memory coalescing. GPU version of `tiled_sum_2d`.
    ///
    /// # Arguments
    ///
    /// * `data` - Input 2D data in row-major order
    /// * `width` - Number of columns
    /// * `height` - Number of rows
    ///
    /// # Returns
    ///
    /// Sum of all elements
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn tiled_sum_2d(&self, data: &[f32], width: usize, height: usize) -> Result<f32, String> {
        runtime::block_on(self.tiled_sum_2d_async(data, width, height))
    }

    /// 2D Tiled Sum Reduction on GPU (async, works on all platforms)
    pub async fn tiled_sum_2d_async(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Result<f32, String> {
        self.tiled_reduce_2d_async(
            data,
            width,
            height,
            shaders::TILED_SUM_REDUCTION_SHADER,
            "TiledSum",
            0.0, // identity for sum
            |partials| partials.iter().sum(),
        )
        .await
    }

    /// 2D Tiled Max Reduction on GPU (sync, native only)
    ///
    /// Uses 16x16 workgroups for efficient parallel max reduction.
    /// GPU version of `tiled_max_2d`.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn tiled_max_2d(&self, data: &[f32], width: usize, height: usize) -> Result<f32, String> {
        runtime::block_on(self.tiled_max_2d_async(data, width, height))
    }

    /// 2D Tiled Max Reduction on GPU (async, works on all platforms)
    pub async fn tiled_max_2d_async(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Result<f32, String> {
        self.tiled_reduce_2d_async(
            data,
            width,
            height,
            shaders::TILED_MAX_REDUCTION_SHADER,
            "TiledMax",
            f32::NEG_INFINITY, // identity for max
            |partials| partials.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        )
        .await
    }

    /// 2D Tiled Min Reduction on GPU (sync, native only)
    ///
    /// Uses 16x16 workgroups for efficient parallel min reduction.
    /// GPU version of `tiled_min_2d`.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn tiled_min_2d(&self, data: &[f32], width: usize, height: usize) -> Result<f32, String> {
        runtime::block_on(self.tiled_min_2d_async(data, width, height))
    }

    /// 2D Tiled Min Reduction on GPU (async, works on all platforms)
    pub async fn tiled_min_2d_async(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
    ) -> Result<f32, String> {
        self.tiled_reduce_2d_async(
            data,
            width,
            height,
            shaders::TILED_MIN_REDUCTION_SHADER,
            "TiledMin",
            f32::INFINITY, // identity for min
            |partials| partials.iter().copied().fold(f32::INFINITY, f32::min),
        )
        .await
    }

    /// Generic 2D tiled reduction helper
    #[allow(clippy::too_many_arguments)]
    async fn tiled_reduce_2d_async<F>(
        &self,
        data: &[f32],
        width: usize,
        height: usize,
        shader_source: &str,
        op_name: &str,
        identity: f32,
        combine: F,
    ) -> Result<f32, String>
    where
        F: Fn(&[f32]) -> f32,
    {
        if data.is_empty() || width == 0 || height == 0 {
            return Ok(identity);
        }

        // Calculate workgroup dimensions (16x16 tiles)
        let workgroup_size_x: u32 = 16;
        let workgroup_size_y: u32 = 16;
        let num_workgroups_x = (width as u32).div_ceil(workgroup_size_x);
        let num_workgroups_y = (height as u32).div_ceil(workgroup_size_y);
        let total_workgroups = (num_workgroups_x * num_workgroups_y) as usize;

        // Create shader module
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&format!("{} Shader", op_name)),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        // Create input buffer
        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Input", op_name)),
            size: std::mem::size_of_val(data) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create partial results buffer
        let partial_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Partial Results", op_name)),
            size: (total_workgroups * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dimensions uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Dimensions {
            width: u32,
            height: u32,
        }

        let dims = Dimensions {
            width: width as u32,
            height: height as u32,
        };

        let dims_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Dimensions", op_name)),
            size: std::mem::size_of::<Dimensions>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data
        self.queue
            .write_buffer(&input_buffer, 0, bytemuck::cast_slice(data));
        self.queue
            .write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));

        // Create bind group layout
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{} Bind Group Layout", op_name)),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", op_name)),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: partial_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{} Pipeline Layout", op_name)),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{} Pipeline", op_name)),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Staging", op_name)),
            size: (total_workgroups * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("{} Encoder", op_name)),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{} Pass", op_name)),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Copy result to staging buffer
        encoder.copy_buffer_to_buffer(
            &partial_buffer,
            0,
            &staging_buffer,
            0,
            (total_workgroups * std::mem::size_of::<f32>()) as u64,
        );

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        // Poll device
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();

        receiver
            .receive()
            .await
            .ok_or("Failed to receive mapping result")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        let final_result = {
            let data = buffer_slice.get_mapped_range();
            let partials: &[f32] = bytemuck::cast_slice(&data);
            combine(partials)
        };

        staging_buffer.unmap();

        Ok(final_result)
    }
}
