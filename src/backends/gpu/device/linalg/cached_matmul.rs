//! PMAT-322: Cached GPU matmul with persistent weight buffers.
//!
//! The default `matmul_async` creates all GPU objects per call (~8ms overhead).
//! This module pre-uploads weight matrices and caches the pipeline, reducing
//! per-call overhead to: upload input + dispatch + download output (~0.1ms).

use std::collections::HashMap;

/// Cached matmul state: pipeline + pre-uploaded weight buffers.
pub struct GpuMatmulCache {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// Pre-uploaded weight buffers keyed by name
    weight_buffers: HashMap<String, WeightEntry>,
    /// Reusable staging buffer (grows as needed)
    staging_size: u64,
    staging_buffer: Option<wgpu::Buffer>,
}

struct WeightEntry {
    buffer: wgpu::Buffer,
    rows: usize,
    cols: usize,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Dimensions {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}

impl GpuMatmulCache {
    /// Create a new cached matmul context from an existing GpuDevice.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CachedMatmul Shader"),
            source: wgpu::ShaderSource::Wgsl(
                crate::backends::gpu::shaders::MATMUL_SHADER.into(),
            ),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("CachedMatmul BGL"),
                entries: &[
                    bgl_entry(0, true),  // A (input, read-only)
                    bgl_entry(1, true),  // B (weight, read-only)
                    bgl_entry(2, false), // C (output, read-write)
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CachedMatmul PL"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CachedMatmul Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            weight_buffers: HashMap::new(),
            staging_size: 0,
            staging_buffer: None,
        }
    }

    /// Pre-upload a weight matrix (call once at model init).
    /// Weight is stored in row-major f32: shape [rows, cols].
    pub fn upload_weight(&mut self, name: &str, data: &[f32], rows: usize, cols: usize) {
        assert_eq!(data.len(), rows * cols, "weight size mismatch");
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: (data.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        self.weight_buffers.insert(
            name.to_string(),
            WeightEntry { buffer, rows, cols },
        );
    }

    /// Number of pre-uploaded weights.
    pub fn weight_count(&self) -> usize {
        self.weight_buffers.len()
    }

    /// Total VRAM used by weight buffers (bytes).
    pub fn weight_bytes(&self) -> usize {
        self.weight_buffers
            .values()
            .map(|w| w.rows * w.cols * 4)
            .sum()
    }

    /// Matmul: result = input × weight^T (for GEMV: M=1, result = [1, rows]).
    /// `weight_name` must have been uploaded via `upload_weight`.
    /// Input shape: [m, cols]. Output shape: [m, rows].
    pub fn matmul_cached(
        &mut self,
        weight_name: &str,
        input: &[f32],
        output: &mut [f32],
        m: usize,
    ) -> Result<(), String> {
        let entry = self
            .weight_buffers
            .get(weight_name)
            .ok_or_else(|| format!("Weight '{}' not uploaded", weight_name))?;

        let k = entry.cols;
        let n = entry.rows;

        if input.len() < m * k {
            return Err(format!("input too small: need {}, have {}", m * k, input.len()));
        }
        if output.len() < m * n {
            return Err(format!("output too small: need {}, have {}", m * n, output.len()));
        }

        // Create input buffer (per-call — small, M×K floats)
        let a_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input"),
            size: (m * k * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&a_buffer, 0, bytemuck::cast_slice(&input[..m * k]));

        // Create output buffer (per-call — small, M×N floats)
        let output_size = (m * n * 4) as u64;
        let c_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dims uniform
        let dims = Dimensions {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            _padding: 0,
        };
        let dims_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dims"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&dims_buffer, 0, bytemuck::bytes_of(&dims));

        // Bind group (per-call — references cached weight buffer)
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_cached BG"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: entry.buffer.as_entire_binding(),
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

        // Ensure staging buffer is large enough
        if self.staging_size < output_size {
            self.staging_buffer = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: output_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.staging_size = output_size;
        }
        let staging = self.staging_buffer.as_ref().unwrap();

        // Encode + dispatch
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_cached"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                (m as u32).div_ceil(16),
                (n as u32).div_ceil(16),
                1,
            );
        }

        encoder.copy_buffer_to_buffer(&c_buffer, 0, staging, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        // Readback
        let slice = staging.slice(..output_size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        rx.recv()
            .map_err(|e| format!("recv: {e}"))?
            .map_err(|e| format!("map: {e:?}"))?;

        {
            let data = slice.get_mapped_range();
            output[..m * n].copy_from_slice(bytemuck::cast_slice(&data));
        }
        staging.unmap();

        Ok(())
    }
}

fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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
