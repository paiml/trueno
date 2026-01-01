//! GPU device initialization and management
//!
//! This module provides cross-platform GPU compute via wgpu (WebGPU).
//!
//! # Platform differences
//!
//! - **Native**: Sync wrappers available using `pollster::block_on`
//! - **WASM**: Sync wrappers unavailable (can't block main thread); use `*_async` methods
//!
//! Use `runtime::sync_available()` to check at runtime.

#[cfg(any(feature = "gpu", feature = "gpu-wasm"))]
use super::runtime;
use super::shaders;

/// GPU device manager
#[derive(Clone)]
pub struct GpuDevice {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuDevice {
    /// Initialize GPU device (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn new() -> Result<Self, String> {
        runtime::block_on(async { Self::new_async().await })
    }

    /// Initialize GPU device (async, works on all platforms)
    pub async fn new_async() -> Result<Self, String> {
        // Create instance
        let instance = wgpu::Instance::default();

        // Request adapter (GPU)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("Failed to find GPU adapter: {}", e))?;

        // Request device and queue
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Trueno GPU Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: Default::default(),
                trace: Default::default(),
            })
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        Ok(Self { device, queue })
    }

    /// Check if GPU is available (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn is_available() -> bool {
        runtime::block_on(Self::is_available_async())
    }

    /// Check if GPU is available (async, works on all platforms)
    pub async fn is_available_async() -> bool {
        let instance = wgpu::Instance::default();
        instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .is_ok()
    }

    /// Execute matrix multiplication on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), String> {
        runtime::block_on(async { self.matmul_async(a, b, result, m, k, n).await })
    }

    /// Execute matrix multiplication on GPU (async, works on all platforms)
    pub async fn matmul_async(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), String> {
        // Create shader module
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Matmul Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::MATMUL_SHADER.into()),
            });

        // Create buffers
        let a_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix A"),
            size: std::mem::size_of_val(a) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix B"),
            size: std::mem::size_of_val(b) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix C"),
            size: std::mem::size_of_val(result) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dimensions uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Dimensions {
            m: u32,
            k: u32,
            n: u32,
            _padding: u32,
        }

        let dims = Dimensions {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _padding: 0,
        };

        let dims_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dimensions"),
            size: std::mem::size_of::<Dimensions>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data to buffers
        self.queue
            .write_buffer(&a_buffer, 0, bytemuck::cast_slice(a));
        self.queue
            .write_buffer(&b_buffer, 0, bytemuck::cast_slice(b));
        self.queue
            .write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));

        // Create bind group layout
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Matmul Bind Group Layout"),
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

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul Bind Group"),
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
                    resource: c_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Matmul Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Matmul Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: std::mem::size_of_val(result) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Matmul Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matmul Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (16×16 threads per workgroup)
            let workgroup_size_x = 16;
            let workgroup_size_y = 16;
            let num_workgroups_x = (m as u32).div_ceil(workgroup_size_x);
            let num_workgroups_y = (n as u32).div_ceil(workgroup_size_y);

            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Copy result to staging buffer
        encoder.copy_buffer_to_buffer(
            &c_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of_val(result) as u64,
        );

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read back results
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
            .ok_or("Failed to receive mapping result")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        {
            let data = buffer_slice.get_mapped_range();
            result.copy_from_slice(bytemuck::cast_slice(&data));
        }

        staging_buffer.unmap();

        Ok(())
    }

    /// Execute vector addition on GPU: c = a + b (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn vec_add(&self, a: &[f32], b: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(async { self.vec_add_async(a, b, result).await })
    }

    /// Execute vector addition on GPU: c = a + b (async, works on all platforms)
    pub async fn vec_add_async(
        &self,
        a: &[f32],
        b: &[f32],
        result: &mut [f32],
    ) -> Result<(), String> {
        let len = a.len();

        // Create shader module
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Vec Add Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::VEC_ADD_SHADER.into()),
            });

        // Create buffers
        let a_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vector A"),
            size: std::mem::size_of_val(a) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vector B"),
            size: std::mem::size_of_val(b) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vector C"),
            size: std::mem::size_of_val(result) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data to buffers
        self.queue
            .write_buffer(&a_buffer, 0, bytemuck::cast_slice(a));
        self.queue
            .write_buffer(&b_buffer, 0, bytemuck::cast_slice(b));

        // Create bind group layout
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Vec Add Bind Group Layout"),
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

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vec Add Bind Group"),
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
                    resource: c_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Vec Add Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Vec Add Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: std::mem::size_of_val(result) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Vec Add Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Vec Add Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (256 threads per workgroup)
            let workgroup_size = 256;
            let num_workgroups = (len as u32).div_ceil(workgroup_size);

            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Copy result to staging buffer
        encoder.copy_buffer_to_buffer(
            &c_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of_val(result) as u64,
        );

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read back results
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
            .ok_or("Failed to receive mapping result")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        {
            let data = buffer_slice.get_mapped_range();
            result.copy_from_slice(bytemuck::cast_slice(&data));
        }

        staging_buffer.unmap();

        Ok(())
    }

    /// Generic helper for element-wise GPU operations
    ///
    /// This helper eliminates code duplication between element-wise operations
    /// (relu, clip, sigmoid, tanh, etc.) by abstracting the common GPU compute pattern.
    ///
    /// # Arguments
    ///
    /// * `op_name` - Operation name for labels (e.g., "ReLU", "Clip")
    /// * `shader_source` - WGSL shader source code
    /// * `input` - Input data
    /// * `result` - Output buffer
    /// * `uniform_data` - Optional uniform buffer data (e.g., clip parameters)
    async fn execute_element_wise_op(
        &self,
        op_name: &str,
        shader_source: &str,
        input: &[f32],
        result: &mut [f32],
        uniform_data: Option<&[u8]>,
    ) -> Result<(), String> {
        let len = input.len();

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
            size: std::mem::size_of_val(input) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create output buffer
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Output", op_name)),
            size: std::mem::size_of_val(result) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write input data
        self.queue
            .write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));

        // Create optional uniform buffer
        let uniform_buffer = uniform_data.map(|data| {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Uniform", op_name)),
                size: data.len() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buffer, 0, data);
            buffer
        });

        // Create bind group layout entries (input + output + optional uniform)
        let mut bind_group_entries = vec![
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

        // Add uniform buffer binding if present
        if uniform_buffer.is_some() {
            bind_group_entries.push(wgpu::BindGroupLayoutEntry {
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

        // Create bind group layout
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{} Bind Group Layout", op_name)),
                    entries: &bind_group_entries,
                });

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

        // Add uniform buffer binding if present
        if let Some(ref uniform_buf) = uniform_buffer {
            bind_entries.push(wgpu::BindGroupEntry {
                binding: 2,
                resource: uniform_buf.as_entire_binding(),
            });
        }

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Bind Group", op_name)),
            layout: &bind_group_layout,
            entries: &bind_entries,
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

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{} Staging Buffer", op_name)),
            size: std::mem::size_of_val(result) as u64,
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

            // Dispatch workgroups (256 threads per workgroup)
            let workgroup_size = 256;
            let num_workgroups = (len as u32).div_ceil(workgroup_size);

            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Copy result to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of_val(result) as u64,
        );

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read back results
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
            .ok_or("Failed to receive mapping result")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        {
            let data = buffer_slice.get_mapped_range();
            result.copy_from_slice(bytemuck::cast_slice(&data));
        }

        staging_buffer.unmap();

        Ok(())
    }

    /// Execute ReLU activation on GPU: result[i] = max(0, input[i]) (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn relu(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(async {
            self.execute_element_wise_op("ReLU", shaders::RELU_SHADER, input, result, None)
                .await
        })
    }

    /// Execute ReLU activation on GPU (async, works on all platforms)
    pub async fn relu_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        self.execute_element_wise_op("ReLU", shaders::RELU_SHADER, input, result, None)
            .await
    }

    /// Execute leaky ReLU activation on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn leaky_relu(
        &self,
        input: &[f32],
        result: &mut [f32],
        negative_slope: f32,
    ) -> Result<(), String> {
        runtime::block_on(self.leaky_relu_async(input, result, negative_slope))
    }

    /// Execute leaky ReLU activation on GPU (async, works on all platforms)
    pub async fn leaky_relu_async(
        &self,
        input: &[f32],
        result: &mut [f32],
        negative_slope: f32,
    ) -> Result<(), String> {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct LeakyReluParams {
            negative_slope: f32,
        }

        let params = LeakyReluParams { negative_slope };
        let uniform_data = bytemuck::bytes_of(&params);

        self.execute_element_wise_op(
            "LeakyReLU",
            shaders::LEAKY_RELU_SHADER,
            input,
            result,
            Some(uniform_data),
        )
        .await
    }

    /// Execute ELU activation on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn elu(&self, input: &[f32], result: &mut [f32], alpha: f32) -> Result<(), String> {
        runtime::block_on(self.elu_async(input, result, alpha))
    }

    /// Execute ELU activation on GPU (async, works on all platforms)
    pub async fn elu_async(
        &self,
        input: &[f32],
        result: &mut [f32],
        alpha: f32,
    ) -> Result<(), String> {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct EluParams {
            alpha: f32,
        }

        let params = EluParams { alpha };
        let uniform_data = bytemuck::bytes_of(&params);

        self.execute_element_wise_op(
            "ELU",
            shaders::ELU_SHADER,
            input,
            result,
            Some(uniform_data),
        )
        .await
    }

    /// Execute sigmoid activation on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn sigmoid(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(self.sigmoid_async(input, result))
    }

    /// Execute sigmoid activation on GPU (async, works on all platforms)
    pub async fn sigmoid_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        self.execute_element_wise_op("Sigmoid", shaders::SIGMOID_SHADER, input, result, None)
            .await
    }

    /// Execute tanh activation on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn tanh(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(self.tanh_async(input, result))
    }

    /// Execute tanh activation on GPU (async, works on all platforms)
    pub async fn tanh_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        self.execute_element_wise_op("Tanh", shaders::TANH_SHADER, input, result, None)
            .await
    }

    /// Execute swish activation on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn swish(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(self.swish_async(input, result))
    }

    /// Execute swish activation on GPU (async, works on all platforms)
    pub async fn swish_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        self.execute_element_wise_op("Swish", shaders::SWISH_SHADER, input, result, None)
            .await
    }

    /// Execute GELU activation on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn gelu(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(self.gelu_async(input, result))
    }

    /// Execute GELU activation on GPU (async, works on all platforms)
    pub async fn gelu_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        self.execute_element_wise_op("GELU", shaders::GELU_SHADER, input, result, None)
            .await
    }

    /// Execute clip (clamp) operation on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn clip(
        &self,
        input: &[f32],
        result: &mut [f32],
        min_val: f32,
        max_val: f32,
    ) -> Result<(), String> {
        runtime::block_on(self.clip_async(input, result, min_val, max_val))
    }

    /// Execute clip (clamp) operation on GPU (async, works on all platforms)
    pub async fn clip_async(
        &self,
        input: &[f32],
        result: &mut [f32],
        min_val: f32,
        max_val: f32,
    ) -> Result<(), String> {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ClipParams {
            min_val: f32,
            max_val: f32,
        }

        let params = ClipParams { min_val, max_val };
        let uniform_data = bytemuck::bytes_of(&params);

        self.execute_element_wise_op(
            "Clip",
            shaders::CLIP_SHADER,
            input,
            result,
            Some(uniform_data),
        )
        .await
    }

    /// Execute softmax on GPU (sync, native only)
    ///
    /// Multi-pass implementation:
    /// 1. Find max value (parallel reduction)
    /// 2. Compute exp(x - max) (element-wise)
    /// 3. Sum exp values (parallel reduction)
    /// 4. Normalize by sum (element-wise)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn softmax(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(async { self.softmax_async(input, result).await })
    }

    /// Execute softmax on GPU (async, works on all platforms)
    pub async fn softmax_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        // Pass 1: Find max value
        let max_val = self.reduce_max(input).await?;

        // Pass 2: Compute exp(x - max)
        let exp_vals = self.compute_exp_subtract(input, max_val).await?;

        // Pass 3: Sum exp values
        let sum_exp = self.reduce_sum(&exp_vals).await?;

        // Pass 4: Normalize by sum
        self.normalize_by_sum(&exp_vals, result, sum_exp).await?;

        Ok(())
    }

    /// Execute log_softmax on GPU (sync, native only)
    ///
    /// Multi-pass implementation:
    /// 1. Find max value (parallel reduction)
    /// 2. Compute exp(x - max) (element-wise)
    /// 3. Sum exp values (parallel reduction)
    /// 4. Compute log_softmax (element-wise)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn log_softmax(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        runtime::block_on(async { self.log_softmax_async(input, result).await })
    }

    /// Execute log_softmax on GPU (async, works on all platforms)
    pub async fn log_softmax_async(&self, input: &[f32], result: &mut [f32]) -> Result<(), String> {
        // Pass 1: Find max value
        let max_val = self.reduce_max(input).await?;

        // Pass 2: Compute exp(x - max)
        let exp_vals = self.compute_exp_subtract(input, max_val).await?;

        // Pass 3: Sum exp values
        let sum_exp = self.reduce_sum(&exp_vals).await?;

        // Pass 4: Compute log_softmax = x - max - log(sum_exp)
        let log_sum_exp = sum_exp.ln();

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct LogSoftmaxParams {
            max_val: f32,
            log_sum_exp: f32,
        }

        let params = LogSoftmaxParams {
            max_val,
            log_sum_exp,
        };
        let uniform_data = bytemuck::bytes_of(&params);

        self.execute_element_wise_op(
            "LogSoftmax",
            shaders::LOG_SOFTMAX_SHADER,
            input,
            result,
            Some(uniform_data),
        )
        .await?;

        Ok(())
    }

    /// Helper: Parallel max reduction
    async fn reduce_max(&self, input: &[f32]) -> Result<f32, String> {
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
    async fn reduce_sum(&self, input: &[f32]) -> Result<f32, String> {
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

    /// Helper: Compute exp(input[i] - max_val)
    async fn compute_exp_subtract(&self, input: &[f32], max_val: f32) -> Result<Vec<f32>, String> {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct MaxValue {
            max_val: f32,
        }

        let params = MaxValue { max_val };
        let uniform_data = bytemuck::bytes_of(&params);

        let mut result = vec![0.0f32; input.len()];
        self.execute_element_wise_op(
            "SoftmaxExp",
            shaders::SOFTMAX_EXP_SHADER,
            input,
            &mut result,
            Some(uniform_data),
        )
        .await?;

        Ok(result)
    }

    /// Helper: Normalize by sum
    async fn normalize_by_sum(
        &self,
        input: &[f32],
        result: &mut [f32],
        sum_val: f32,
    ) -> Result<(), String> {
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct SumValue {
            sum_val: f32,
        }

        let params = SumValue { sum_val };
        let uniform_data = bytemuck::bytes_of(&params);

        self.execute_element_wise_op(
            "SoftmaxNormalize",
            shaders::SOFTMAX_NORMALIZE_SHADER,
            input,
            result,
            Some(uniform_data),
        )
        .await?;

        Ok(())
    }

    /// Execute dot product on GPU (sync, native only)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn dot(&self, a: &[f32], b: &[f32]) -> Result<f32, String> {
        runtime::block_on(async { self.dot_async(a, b).await })
    }

    /// Execute dot product on GPU (async, works on all platforms)
    pub async fn dot_async(&self, a: &[f32], b: &[f32]) -> Result<f32, String> {
        let len = a.len();
        let workgroup_size = 256;
        let num_workgroups = (len as u32).div_ceil(workgroup_size);

        // Create shader module
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Dot Product Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::DOT_PRODUCT_SHADER.into()),
            });

        // Create buffers
        let a_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vector A"),
            size: std::mem::size_of_val(a) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let b_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vector B"),
            size: std::mem::size_of_val(b) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Result buffer for partial sums (one per workgroup)
        let partial_results = vec![0.0f32; num_workgroups as usize];
        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Partial Results"),
            size: std::mem::size_of_val(partial_results.as_slice()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data to buffers
        self.queue
            .write_buffer(&a_buffer, 0, bytemuck::cast_slice(a));
        self.queue
            .write_buffer(&b_buffer, 0, bytemuck::cast_slice(b));

        // Create bind group layout
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Dot Product Bind Group Layout"),
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

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dot Product Bind Group"),
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
                    resource: result_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Dot Product Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Dot Product Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create staging buffer for reading results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: std::mem::size_of_val(partial_results.as_slice()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Dot Product Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dot Product Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Copy result to staging buffer
        encoder.copy_buffer_to_buffer(
            &result_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of_val(partial_results.as_slice()) as u64,
        );

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read back results
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
            .ok_or("Failed to receive mapping result")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        let final_result = {
            let data = buffer_slice.get_mapped_range();
            let partial_sums: &[f32] = bytemuck::cast_slice(&data);

            // Sum the partial results from each workgroup on CPU
            partial_sums.iter().sum()
        };

        staging_buffer.unmap();

        Ok(final_result)
    }

    /// Perform 2D convolution on GPU (sync, native only)
    ///
    /// # Arguments
    ///
    /// * `input` - Input image (row-major)
    /// * `kernel` - Convolution kernel (row-major)
    /// * `result` - Output buffer (row-major)
    /// * `input_rows` - Number of rows in input
    /// * `input_cols` - Number of columns in input
    /// * `kernel_rows` - Number of rows in kernel
    /// * `kernel_cols` - Number of columns in kernel
    ///
    /// Output dimensions: (input_rows - kernel_rows + 1) × (input_cols - kernel_cols + 1)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    #[allow(clippy::too_many_arguments)]
    pub fn convolve2d(
        &self,
        input: &[f32],
        kernel: &[f32],
        result: &mut [f32],
        input_rows: usize,
        input_cols: usize,
        kernel_rows: usize,
        kernel_cols: usize,
    ) -> Result<(), String> {
        runtime::block_on(async {
            self.convolve2d_async(
                input,
                kernel,
                result,
                input_rows,
                input_cols,
                kernel_rows,
                kernel_cols,
            )
            .await
        })
    }

    /// Perform 2D convolution on GPU (async, works on all platforms)
    #[allow(clippy::too_many_arguments)]
    pub async fn convolve2d_async(
        &self,
        input: &[f32],
        kernel: &[f32],
        result: &mut [f32],
        input_rows: usize,
        input_cols: usize,
        kernel_rows: usize,
        kernel_cols: usize,
    ) -> Result<(), String> {
        let output_rows = input_rows - kernel_rows + 1;
        let output_cols = input_cols - kernel_cols + 1;

        // Create shader module
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Convolve2D Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::CONVOLVE2D_SHADER.into()),
            });

        // Create buffers
        let input_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Input Image"),
            size: std::mem::size_of_val(input) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let kernel_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Kernel"),
            size: std::mem::size_of_val(kernel) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output"),
            size: std::mem::size_of_val(result) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dimensions uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ConvDimensions {
            input_rows: u32,
            input_cols: u32,
            kernel_rows: u32,
            kernel_cols: u32,
            output_rows: u32,
            output_cols: u32,
        }

        let dims = ConvDimensions {
            input_rows: input_rows as u32,
            input_cols: input_cols as u32,
            kernel_rows: kernel_rows as u32,
            kernel_cols: kernel_cols as u32,
            output_rows: output_rows as u32,
            output_cols: output_cols as u32,
        };

        let dims_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conv Dimensions"),
            size: std::mem::size_of::<ConvDimensions>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write data to buffers
        self.queue
            .write_buffer(&input_buffer, 0, bytemuck::cast_slice(input));
        self.queue
            .write_buffer(&kernel_buffer, 0, bytemuck::cast_slice(kernel));
        self.queue
            .write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));

        // Create bind group layout
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Convolve2D Bind Group Layout"),
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

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Convolve2D Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kernel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dims_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Convolve2D Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        // Create compute pipeline
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Convolve2D Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Convolve2D Encoder"),
            });

        // Compute pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Convolve2D Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups: 16×16 threads per workgroup
            let workgroup_size_x = 16;
            let workgroup_size_y = 16;
            let num_workgroups_x = (output_rows as u32).div_ceil(workgroup_size_x);
            let num_workgroups_y = (output_cols as u32).div_ceil(workgroup_size_y);
            compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
        }

        // Create staging buffer for result readback
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: std::mem::size_of_val(result) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy output to staging buffer
        encoder.copy_buffer_to_buffer(
            &output_buffer,
            0,
            &staging_buffer,
            0,
            std::mem::size_of_val(result) as u64,
        );

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read result back
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
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
            .ok_or("Failed to receive mapping result")?
            .map_err(|e| format!("Buffer mapping failed: {:?}", e))?;

        {
            let data = buffer_slice.get_mapped_range();
            let output_data: &[f32] = bytemuck::cast_slice(&data);
            result.copy_from_slice(output_data);
        }

        staging_buffer.unmap();

        Ok(())
    }

    /// Execute symmetric eigendecomposition on GPU (sync, native only)
    ///
    /// Computes eigenvalues and eigenvectors using Jacobi algorithm with GPU-accelerated
    /// Givens rotations. Returns (eigenvalues, eigenvector_data) where eigenvector_data
    /// is in row-major format.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn symmetric_eigen(
        &self,
        matrix: &[f32],
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        runtime::block_on(async { self.symmetric_eigen_async(matrix, n).await })
    }

    /// Execute symmetric eigendecomposition on GPU (async, works on all platforms)
    ///
    /// Computes eigenvalues and eigenvectors using Jacobi algorithm with GPU-accelerated
    /// Givens rotations.
    pub async fn symmetric_eigen_async(
        &self,
        matrix: &[f32],
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if matrix.len() != n * n {
            return Err(format!(
                "Matrix size mismatch: expected {} elements for {}x{} matrix, got {}",
                n * n,
                n,
                n,
                matrix.len()
            ));
        }

        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        // For small matrices, use CPU (GPU overhead not worth it)
        if n < 64 {
            return self.symmetric_eigen_cpu(matrix, n);
        }

        // Create shader module for Jacobi rotation
        let rotation_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Jacobi Rotation Shader"),
                source: wgpu::ShaderSource::Wgsl(shaders::JACOBI_ROTATION_SHADER.into()),
            });

        // Create buffers
        let matrix_size = (n * n * std::mem::size_of::<f32>()) as u64;

        let matrix_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Matrix Buffer"),
            size: matrix_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let eigenvectors_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Eigenvectors Buffer"),
            size: matrix_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // JacobiParams uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct JacobiParams {
            n: u32,
            p: u32,
            q: u32,
            c: f32,
            s: f32,
            _padding: [u32; 3],
        }

        let params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Jacobi Params"),
            size: std::mem::size_of::<JacobiParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize eigenvectors to identity matrix
        let mut eigenvectors = vec![0.0f32; n * n];
        for i in 0..n {
            eigenvectors[i * n + i] = 1.0;
        }

        // Write initial data
        self.queue
            .write_buffer(&matrix_buffer, 0, bytemuck::cast_slice(matrix));
        self.queue
            .write_buffer(&eigenvectors_buffer, 0, bytemuck::cast_slice(&eigenvectors));

        // Create bind group layout
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Jacobi Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
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
            label: Some("Jacobi Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matrix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: eigenvectors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Jacobi Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let rotation_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Jacobi Rotation Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &rotation_shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        // Jacobi iteration
        let max_sweeps = 50;
        let tolerance = 1e-7 * (matrix.iter().map(|x| x * x).sum::<f32>().sqrt()).max(1.0);

        // Working copy of matrix for CPU-side pivot selection
        let mut a = matrix.to_vec();

        for _sweep in 0..max_sweeps {
            let mut converged = true;

            // Cyclic Jacobi: process all pairs (i, j) where i < j
            for i in 0..n {
                for j in (i + 1)..n {
                    let aij = a[i * n + j];

                    if aij.abs() < tolerance {
                        continue;
                    }

                    converged = false;

                    // Compute rotation parameters
                    let aii = a[i * n + i];
                    let ajj = a[j * n + j];

                    let tau = (ajj - aii) / (2.0 * aij);
                    let t = if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    };

                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;

                    // Update params and dispatch GPU
                    let params = JacobiParams {
                        n: n as u32,
                        p: i as u32,
                        q: j as u32,
                        c,
                        s,
                        _padding: [0; 3],
                    };

                    self.queue
                        .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

                    // Create command encoder and dispatch
                    let mut encoder =
                        self.device
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Jacobi Rotation Encoder"),
                            });

                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("Jacobi Rotation Pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&rotation_pipeline);
                        pass.set_bind_group(0, &bind_group, &[]);
                        pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
                    }

                    self.queue.submit(Some(encoder.finish()));

                    // Update local copy of diagonal and off-diagonal
                    a[i * n + i] = aii - t * aij;
                    a[j * n + j] = ajj + t * aij;
                    a[i * n + j] = 0.0;
                    a[j * n + i] = 0.0;

                    // Update off-diagonal elements in rows/columns i and j
                    for k in 0..n {
                        if k != i && k != j {
                            let aki = a[k * n + i];
                            let akj = a[k * n + j];
                            a[k * n + i] = c * aki - s * akj;
                            a[i * n + k] = a[k * n + i];
                            a[k * n + j] = s * aki + c * akj;
                            a[j * n + k] = a[k * n + j];
                        }
                    }
                }
            }

            if converged {
                break;
            }
        }

        // Read back results
        let staging_matrix = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Matrix"),
            size: matrix_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_eigenvectors = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Eigenvectors"),
            size: matrix_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Encoder"),
            });

        encoder.copy_buffer_to_buffer(&matrix_buffer, 0, &staging_matrix, 0, matrix_size);
        encoder.copy_buffer_to_buffer(
            &eigenvectors_buffer,
            0,
            &staging_eigenvectors,
            0,
            matrix_size,
        );

        self.queue.submit(Some(encoder.finish()));

        // Map and read eigenvectors
        let eigenvector_slice = staging_eigenvectors.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        eigenvector_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

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

        let mut result_eigenvectors = vec![0.0f32; n * n];
        {
            let data = eigenvector_slice.get_mapped_range();
            let output_data: &[f32] = bytemuck::cast_slice(&data);
            result_eigenvectors.copy_from_slice(output_data);
        }
        staging_eigenvectors.unmap();

        // Extract eigenvalues from diagonal of working matrix
        let eigenvalues: Vec<f32> = (0..n).map(|i| a[i * n + i]).collect();

        Ok((eigenvalues, result_eigenvectors))
    }

    /// CPU fallback for small matrices (GPU overhead not worthwhile)
    fn symmetric_eigen_cpu(
        &self,
        matrix: &[f32],
        n: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let max_sweeps = 50;
        let tolerance = 1e-7 * (matrix.iter().map(|x| x * x).sum::<f32>().sqrt()).max(1.0);

        let mut a = matrix.to_vec();
        let mut v = vec![0.0f32; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }

        for _sweep in 0..max_sweeps {
            let mut converged = true;

            for i in 0..n {
                for j in (i + 1)..n {
                    let aij = a[i * n + j];

                    if aij.abs() < tolerance {
                        continue;
                    }

                    converged = false;

                    let aii = a[i * n + i];
                    let ajj = a[j * n + j];

                    let tau = (ajj - aii) / (2.0 * aij);
                    let t = if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    };

                    let c = 1.0 / (1.0 + t * t).sqrt();
                    let s = t * c;

                    // Update diagonal
                    a[i * n + i] = aii - t * aij;
                    a[j * n + j] = ajj + t * aij;
                    a[i * n + j] = 0.0;
                    a[j * n + i] = 0.0;

                    // Update off-diagonal
                    for k in 0..n {
                        if k != i && k != j {
                            let aki = a[k * n + i];
                            let akj = a[k * n + j];
                            a[k * n + i] = c * aki - s * akj;
                            a[i * n + k] = a[k * n + i];
                            a[k * n + j] = s * aki + c * akj;
                            a[j * n + k] = a[k * n + j];
                        }
                    }

                    // Update eigenvectors
                    for k in 0..n {
                        let vki = v[k * n + i];
                        let vkj = v[k * n + j];
                        v[k * n + i] = c * vki - s * vkj;
                        v[k * n + j] = s * vki + c * vkj;
                    }
                }
            }

            if converged {
                break;
            }
        }

        let eigenvalues: Vec<f32> = (0..n).map(|i| a[i * n + i]).collect();
        Ok((eigenvalues, v))
    }

    /// 2D Tiled Sum Reduction on GPU (sync, native only)
    ///
    /// Uses 16×16 workgroups for efficient parallel reduction with
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
    /// Uses 16×16 workgroups for efficient parallel max reduction.
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
    /// Uses 16×16 workgroups for efficient parallel min reduction.
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

        // Calculate workgroup dimensions (16×16 tiles)
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

#[cfg(all(test, feature = "gpu", not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn test_is_available_consistency() {
        // EXTREME TDD: Kill mutant that replaces is_available() with hardcoded false
        // Test that is_available() is consistent with GpuDevice::new()
        let available = GpuDevice::is_available();
        let device_result = GpuDevice::new();

        if available {
            // If is_available() returns true, device creation should succeed
            assert!(
                device_result.is_ok(),
                "is_available() returned true, but GpuDevice::new() failed"
            );
        } else {
            // If is_available() returns false, we can't make assertions about new()
            // (it might still succeed in some edge cases, but typically should fail)
            // The key test is: mutant always returns false, so on GPU systems this fails
            eprintln!(
                "GPU not available (is_available=false), device creation result: {:?}",
                device_result.is_err()
            );
        }
    }

    #[test]
    fn test_reduce_sum_not_hardcoded() {
        // EXTREME TDD: Kill mutant that replaces reduce_sum with Ok(-1.0)
        if !GpuDevice::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let device = GpuDevice::new().expect("Failed to create GPU device");
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // sum = 15.0

        // reduce_sum is async, so we use runtime::block_on
        let result = runtime::block_on(device.reduce_sum(&input)).expect("reduce_sum failed");

        // Kill mutant: verify result is NOT -1.0
        assert_ne!(
            result, -1.0,
            "reduce_sum returned hardcoded -1.0 (mutant not killed)"
        );

        // Verify correct computation
        let expected: f32 = input.iter().sum();
        assert!(
            (result - expected).abs() < 1e-4,
            "reduce_sum({:?}) = {} (expected {})",
            input,
            result,
            expected
        );
    }
}
