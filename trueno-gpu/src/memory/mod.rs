//! GPU Memory Management
//!
//! Provides memory pooling and transfer utilities for efficient GPU memory usage.
//!
//! ## Features
//!
//! - **Pool allocator**: Reduces allocation overhead
//! - **Transfer utilities**: Efficient H2D/D2H transfers
//! - **Fragmentation tracking**: Per PagedAttention [12]

mod pool;

pub use pool::{MemoryPool, PoolConfig, AllocationInfo};

use crate::driver::DevicePtr;
use crate::error::Result;

/// GPU buffer wrapper
#[derive(Debug)]
pub struct GpuBuffer<T> {
    ptr: DevicePtr<T>,
    len: usize,
    capacity: usize,
}

impl<T> GpuBuffer<T> {
    /// Create a new uninitialized buffer
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self {
            ptr: DevicePtr::null(),
            len,
            capacity: len,
        }
    }

    /// Get buffer length
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get device pointer
    #[must_use]
    pub const fn as_ptr(&self) -> DevicePtr<T> {
        self.ptr
    }

    /// Get size in bytes
    #[must_use]
    pub const fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

/// Copy data from host to device
pub fn copy_h2d<T: Copy>(_dst: &mut GpuBuffer<T>, _src: &[T]) -> Result<()> {
    // TODO: Implement with CUDA feature
    Ok(())
}

/// Copy data from device to host
pub fn copy_d2h<T: Copy>(_src: &GpuBuffer<T>, _dst: &mut [T]) -> Result<()> {
    // TODO: Implement with CUDA feature
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_buffer_creation() {
        let buffer: GpuBuffer<f32> = GpuBuffer::new(1024);
        assert_eq!(buffer.len(), 1024);
        assert_eq!(buffer.size_bytes(), 1024 * 4);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_gpu_buffer_empty() {
        let buffer: GpuBuffer<f32> = GpuBuffer::new(0);
        assert!(buffer.is_empty());
    }
}
