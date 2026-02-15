//! GPU execution engine for batched operations
//!
//! Contains the `execute()` and `read()` public entry points, plus sub-modules
//! for operation dispatch and shader pipeline infrastructure.
//!
//! - [`dispatch`]: Unary/binary shader dispatch (`execute_unary_op`, `execute_binary_op`)
//! - [`operations`]: Per-operation routing (`execute_operation`)

mod dispatch;
mod operations;

use super::{BufferId, GpuCommandBatch};

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
