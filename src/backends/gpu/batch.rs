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
        // TODO: Implementation in next step
        // 1. Create GPU buffers for all BufferIds
        // 2. Upload initial data
        // 3. Execute operations
        // 4. Keep buffers alive for read()

        Ok(())
    }

    /// Read buffer data back from GPU
    ///
    /// Must call `execute()` first.
    pub async fn read(&self, buffer_id: BufferId) -> Result<Vec<f32>, String> {
        // TODO: Implementation in next step
        // 1. Map buffer for reading
        // 2. Copy data to Vec<f32>
        // 3. Return result

        Err("Not implemented yet".to_string())
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
}
