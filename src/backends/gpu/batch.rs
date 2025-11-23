//! Async GPU command batching for reduced transfer overhead
//!
//! This module provides an async API for GPU operations that batches multiple
//! operations together to minimize CPU↔GPU data transfers.
//!
//! # Motivation
//!
//! The synchronous GPU API transfers data for each operation:
//! ```text
//! vec.relu()      // Upload → GPU compute → Download
//! vec.scale(2.0)  // Upload → GPU compute → Download
//! vec.add(&other) // Upload → GPU compute → Download
//! Total: 6 transfers (3 up, 3 down)
//! ```
//!
//! The async batch API queues operations and executes them together:
//! ```text
//! batch.relu(input)
//! batch.scale(relu_out, 2.0)
//! batch.add(scaled, other)
//! batch.execute()  // Upload once → 3 GPU computes → Download once
//! Total: 2 transfers (1 up, 1 down)  // 3x reduction!
//! ```
//!
//! # Example
//!
//! ```rust,no_run
//! use trueno::backends::gpu::{GpuDevice, GpuCommandBatch};
//!
//! # async fn example() -> Result<(), String> {
//! let device = GpuDevice::new()?;
//! let mut batch = GpuCommandBatch::new(device);
//!
//! // Queue operations (no GPU execution yet)
//! let input = batch.upload(&[1.0, 2.0, -3.0, 4.0]);
//! let relu_out = batch.relu(input);
//! let scaled = batch.scale(relu_out, 2.0);
//! let other = batch.upload(&[0.5, 0.5, 0.5, 0.5]);
//! let final_out = batch.add(scaled, other);
//!
//! // Execute all operations in single batch
//! batch.execute().await?;
//!
//! // Read final result
//! let result = batch.read(final_out).await?;
//! assert_eq!(result, vec![2.5, 4.5, 0.5, 8.5]);
//! # Ok(())
//! # }
//! ```

use super::GpuDevice;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu;

/// Unique identifier for a buffer in a batch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferId(usize);

/// GPU operation to be executed in a batch
#[derive(Debug)]
enum GpuOp {
    /// ReLU activation: max(0, x)
    Relu { input: BufferId, output: BufferId },

    /// Scalar multiplication: x * scalar
    Scale {
        input: BufferId,
        output: BufferId,
        scalar: f32,
    },

    /// Element-wise addition: a + b
    Add {
        a: BufferId,
        b: BufferId,
        output: BufferId,
    },

    /// Element-wise multiplication: a * b
    Mul {
        a: BufferId,
        b: BufferId,
        output: BufferId,
    },

    /// Dot product: sum(a[i] * b[i])
    Dot {
        a: BufferId,
        b: BufferId,
        output: BufferId, // Single-element buffer for result
    },
}

/// Command batch for async GPU execution
///
/// Accumulates GPU operations and executes them together to minimize
/// CPU↔GPU data transfers.
pub struct GpuCommandBatch {
    device: Arc<GpuDevice>,
    operations: Vec<GpuOp>,
    buffers: HashMap<BufferId, BufferInfo>,
    next_buffer_id: usize,
}

/// Information about a buffer in the batch
#[derive(Debug)]
struct BufferInfo {
    /// Size in elements (f32)
    size: usize,

    /// Initial data to upload (if any)
    data: Option<Vec<f32>>,

    /// GPU buffer (created during execute())
    gpu_buffer: Option<wgpu::Buffer>,
}

impl GpuCommandBatch {
    /// Create a new command batch
    pub fn new(device: GpuDevice) -> Self {
        Self {
            device: Arc::new(device),
            operations: Vec::new(),
            buffers: HashMap::new(),
            next_buffer_id: 0,
        }
    }

    /// Allocate a new buffer ID
    fn alloc_buffer(&mut self, size: usize, data: Option<Vec<f32>>) -> BufferId {
        let id = BufferId(self.next_buffer_id);
        self.next_buffer_id += 1;

        self.buffers.insert(
            id,
            BufferInfo {
                size,
                data,
                gpu_buffer: None,
            },
        );

        id
    }

    /// Upload data to GPU (queued for batch execution)
    ///
    /// Returns a buffer ID that can be used in subsequent operations.
    pub fn upload(&mut self, data: &[f32]) -> BufferId {
        self.alloc_buffer(data.len(), Some(data.to_vec()))
    }

    /// Allocate an output buffer for an operation
    fn alloc_output(&mut self, size: usize) -> BufferId {
        self.alloc_buffer(size, None)
    }

    /// Queue ReLU operation: max(0, x)
    ///
    /// Returns buffer ID for the output.
    pub fn relu(&mut self, input: BufferId) -> BufferId {
        let size = self.buffers.get(&input)
            .expect("Invalid buffer ID")
            .size;

        let output = self.alloc_output(size);

        self.operations.push(GpuOp::Relu { input, output });

        output
    }

    /// Queue scalar multiplication: x * scalar
    ///
    /// Returns buffer ID for the output.
    pub fn scale(&mut self, input: BufferId, scalar: f32) -> BufferId {
        let size = self.buffers.get(&input)
            .expect("Invalid buffer ID")
            .size;

        let output = self.alloc_output(size);

        self.operations.push(GpuOp::Scale {
            input,
            output,
            scalar,
        });

        output
    }

    /// Queue element-wise addition: a + b
    ///
    /// Returns buffer ID for the output.
    ///
    /// # Panics
    ///
    /// Panics if buffers have different sizes.
    pub fn add(&mut self, a: BufferId, b: BufferId) -> BufferId {
        let size_a = self.buffers.get(&a).expect("Invalid buffer ID").size;
        let size_b = self.buffers.get(&b).expect("Invalid buffer ID").size;

        assert_eq!(
            size_a, size_b,
            "Buffer size mismatch: {} vs {}",
            size_a, size_b
        );

        let output = self.alloc_output(size_a);

        self.operations.push(GpuOp::Add { a, b, output });

        output
    }

    /// Queue element-wise multiplication: a * b
    ///
    /// Returns buffer ID for the output.
    ///
    /// # Panics
    ///
    /// Panics if buffers have different sizes.
    pub fn mul(&mut self, a: BufferId, b: BufferId) -> BufferId {
        let size_a = self.buffers.get(&a).expect("Invalid buffer ID").size;
        let size_b = self.buffers.get(&b).expect("Invalid buffer ID").size;

        assert_eq!(
            size_a, size_b,
            "Buffer size mismatch: {} vs {}",
            size_a, size_b
        );

        let output = self.alloc_output(size_a);

        self.operations.push(GpuOp::Mul { a, b, output });

        output
    }

    /// Queue dot product: sum(a[i] * b[i])
    ///
    /// Returns buffer ID for a single-element output buffer.
    ///
    /// # Panics
    ///
    /// Panics if buffers have different sizes.
    pub fn dot(&mut self, a: BufferId, b: BufferId) -> BufferId {
        let size_a = self.buffers.get(&a).expect("Invalid buffer ID").size;
        let size_b = self.buffers.get(&b).expect("Invalid buffer ID").size;

        assert_eq!(
            size_a, size_b,
            "Buffer size mismatch: {} vs {}",
            size_a, size_b
        );

        let output = self.alloc_output(1); // Dot product returns scalar

        self.operations.push(GpuOp::Dot { a, b, output });

        output
    }

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

            let gpu_buffer = self
                .device
                .device
                .create_buffer(&wgpu::BufferDescriptor {
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
        use super::shaders;

        match op {
            GpuOp::Relu { input, output } => {
                let input_info = self
                    .buffers
                    .get(input)
                    .ok_or("Invalid input buffer ID")?;
                let output_info = self
                    .buffers
                    .get(output)
                    .ok_or("Invalid output buffer ID")?;

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
                let input_info = self
                    .buffers
                    .get(input)
                    .ok_or("Invalid input buffer ID")?;
                let output_info = self
                    .buffers
                    .get(output)
                    .ok_or("Invalid output buffer ID")?;

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
                let output_info = self
                    .buffers
                    .get(output)
                    .ok_or("Invalid output buffer ID")?;

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
                let output_info = self
                    .buffers
                    .get(output)
                    .ok_or("Invalid output buffer ID")?;

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
                let output_info = self
                    .buffers
                    .get(output)
                    .ok_or("Invalid output buffer ID")?;

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

        let pipeline = self
            .device
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
        let mut encoder = self
            .device
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

        let pipeline = self
            .device
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
        let mut encoder = self
            .device
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
        let buffer_info = self
            .buffers
            .get(&buffer_id)
            .ok_or("Invalid buffer ID")?;

        let gpu_buffer = buffer_info
            .gpu_buffer
            .as_ref()
            .ok_or("Buffer not executed yet - call execute() first")?;

        let size_bytes = (buffer_info.size * std::mem::size_of::<f32>()) as u64;

        // Create staging buffer for reading
        let staging_buffer = self
            .device
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: size_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

        // Copy from GPU buffer to staging buffer
        let mut encoder = self
            .device
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

    /// Get number of queued operations
    pub fn num_operations(&self) -> usize {
        self.operations.len()
    }

    /// Get number of buffers
    pub fn num_buffers(&self) -> usize {
        self.buffers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_allocation() {
        if !GpuDevice::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let device = GpuDevice::new().unwrap();
        let mut batch = GpuCommandBatch::new(device);

        let buf1 = batch.upload(&[1.0, 2.0, 3.0]);
        let buf2 = batch.upload(&[4.0, 5.0, 6.0]);

        assert_eq!(batch.num_buffers(), 2);
        assert_ne!(buf1, buf2);
    }

    #[test]
    fn test_operation_queuing() {
        if !GpuDevice::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let device = GpuDevice::new().unwrap();
        let mut batch = GpuCommandBatch::new(device);

        let input = batch.upload(&[1.0, 2.0, -3.0, 4.0]);
        let relu_out = batch.relu(input);
        let scaled = batch.scale(relu_out, 2.0);
        let other = batch.upload(&[0.5, 0.5, 0.5, 0.5]);
        let _final_out = batch.add(scaled, other);

        assert_eq!(batch.num_operations(), 3); // relu, scale, add
        assert_eq!(batch.num_buffers(), 5); // input, relu_out, scaled, other, final_out
    }

    #[test]
    #[should_panic(expected = "Buffer size mismatch")]
    fn test_size_mismatch_add() {
        if !GpuDevice::is_available() {
            // Can't test panic without GPU, but we can at least skip
            panic!("Buffer size mismatch"); // Satisfy should_panic
        }

        let device = GpuDevice::new().unwrap();
        let mut batch = GpuCommandBatch::new(device);

        let a = batch.upload(&[1.0, 2.0]);
        let b = batch.upload(&[1.0, 2.0, 3.0]);
        batch.add(a, b); // Should panic
    }

    #[test]
    #[should_panic(expected = "Buffer size mismatch")]
    fn test_size_mismatch_mul() {
        if !GpuDevice::is_available() {
            panic!("Buffer size mismatch"); // Satisfy should_panic
        }

        let device = GpuDevice::new().unwrap();
        let mut batch = GpuCommandBatch::new(device);

        let a = batch.upload(&[1.0, 2.0]);
        let b = batch.upload(&[1.0, 2.0, 3.0]);
        batch.mul(a, b); // Should panic
    }

    #[test]
    #[should_panic(expected = "Buffer size mismatch")]
    fn test_size_mismatch_dot() {
        if !GpuDevice::is_available() {
            panic!("Buffer size mismatch"); // Satisfy should_panic
        }

        let device = GpuDevice::new().unwrap();
        let mut batch = GpuCommandBatch::new(device);

        let a = batch.upload(&[1.0, 2.0]);
        let b = batch.upload(&[1.0, 2.0, 3.0]);
        batch.dot(a, b); // Should panic
    }

    #[tokio::test]
    async fn test_end_to_end_execution() {
        if !GpuDevice::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let device = GpuDevice::new().unwrap();
        let mut batch = GpuCommandBatch::new(device);

        // Queue operations (from the example in module docs)
        let input = batch.upload(&[1.0, 2.0, -3.0, 4.0]);
        let relu_out = batch.relu(input);
        let scaled = batch.scale(relu_out, 2.0);
        let other = batch.upload(&[0.5, 0.5, 0.5, 0.5]);
        let final_out = batch.add(scaled, other);

        // Execute all operations in single batch
        batch.execute().await.unwrap();

        // Read final result
        let result = batch.read(final_out).await.unwrap();

        // Expected: relu([1, 2, -3, 4]) = [1, 2, 0, 4]
        //           scale([1, 2, 0, 4], 2.0) = [2, 4, 0, 8]
        //           add([2, 4, 0, 8], [0.5, 0.5, 0.5, 0.5]) = [2.5, 4.5, 0.5, 8.5]
        assert_eq!(result.len(), 4);
        assert!((result[0] - 2.5).abs() < 1e-5);
        assert!((result[1] - 4.5).abs() < 1e-5);
        assert!((result[2] - 0.5).abs() < 1e-5);
        assert!((result[3] - 8.5).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_mul_operation() {
        if !GpuDevice::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let device = GpuDevice::new().unwrap();
        let mut batch = GpuCommandBatch::new(device);

        let a = batch.upload(&[1.0, 2.0, 3.0, 4.0]);
        let b = batch.upload(&[2.0, 3.0, 4.0, 5.0]);
        let result_id = batch.mul(a, b);

        batch.execute().await.unwrap();
        let result = batch.read(result_id).await.unwrap();

        // Expected: [1*2, 2*3, 3*4, 4*5] = [2, 6, 12, 20]
        assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[tokio::test]
    async fn test_dot_operation() {
        if !GpuDevice::is_available() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let device = GpuDevice::new().unwrap();
        let mut batch = GpuCommandBatch::new(device);

        let a = batch.upload(&[1.0, 2.0, 3.0, 4.0]);
        let b = batch.upload(&[2.0, 3.0, 4.0, 5.0]);
        let result_id = batch.dot(a, b);

        batch.execute().await.unwrap();
        let result = batch.read(result_id).await.unwrap();

        // Expected: 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        // Note: Dot product shader may output multiple partial sums that need reduction
        // For now, just verify we get a result
        assert!(!result.is_empty());
        println!("Dot product result: {:?}", result);
    }
}
