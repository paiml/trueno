//! CUDA Driver Types
//!
//! Type definitions for CUDA driver API.

use std::marker::PhantomData;

/// CUDA device ordinal
pub type DeviceOrdinal = i32;

/// Device pointer (GPU memory address)
#[derive(Debug, PartialEq, Eq)]
pub struct DevicePtr<T> {
    ptr: u64,
    _marker: PhantomData<T>,
}

// Manual Copy/Clone implementations to not require T: Copy
impl<T> Copy for DevicePtr<T> {}

impl<T> Clone for DevicePtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> DevicePtr<T> {
    /// Create a null device pointer
    #[must_use]
    pub const fn null() -> Self {
        Self {
            ptr: 0,
            _marker: PhantomData,
        }
    }

    /// Create from raw address
    ///
    /// # Safety
    /// The address must be a valid device pointer.
    #[must_use]
    pub const unsafe fn from_raw(ptr: u64) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Get raw address
    #[must_use]
    pub const fn as_raw(self) -> u64 {
        self.ptr
    }

    /// Check if null
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.ptr == 0
    }

    /// Offset by bytes
    #[must_use]
    pub const fn byte_offset(self, bytes: u64) -> Self {
        Self {
            ptr: self.ptr + bytes,
            _marker: PhantomData,
        }
    }
}

/// GPU stream states (Poka-Yoke typestate pattern)
pub mod states {
    /// Stream is idle, ready to record commands
    #[derive(Debug, Clone, Copy)]
    pub struct Idle;

    /// Stream is recording commands
    #[derive(Debug, Clone, Copy)]
    pub struct Recording;

    /// Stream has submitted commands, awaiting completion
    #[derive(Debug, Clone, Copy)]
    pub struct Submitted;
}

/// Launch configuration
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    /// Grid dimensions (blocks)
    pub grid: (u32, u32, u32),
    /// Block dimensions (threads)
    pub block: (u32, u32, u32),
    /// Shared memory per block (bytes)
    pub shared_mem: u32,
}

impl LaunchConfig {
    /// Create a 1D launch configuration
    #[must_use]
    pub const fn linear(num_elements: u32, block_size: u32) -> Self {
        let grid_x = (num_elements + block_size - 1) / block_size;
        Self {
            grid: (grid_x, 1, 1),
            block: (block_size, 1, 1),
            shared_mem: 0,
        }
    }

    /// Create a 2D launch configuration
    #[must_use]
    pub const fn grid_2d(
        grid_x: u32,
        grid_y: u32,
        block_x: u32,
        block_y: u32,
    ) -> Self {
        Self {
            grid: (grid_x, grid_y, 1),
            block: (block_x, block_y, 1),
            shared_mem: 0,
        }
    }

    /// Set shared memory size
    #[must_use]
    pub const fn with_shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem = bytes;
        self
    }

    /// Total threads
    #[must_use]
    pub const fn total_threads(&self) -> u64 {
        let grid_total = self.grid.0 as u64 * self.grid.1 as u64 * self.grid.2 as u64;
        let block_total = self.block.0 as u64 * self.block.1 as u64 * self.block.2 as u64;
        grid_total * block_total
    }
}

impl Default for LaunchConfig {
    fn default() -> Self {
        Self {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_ptr_null() {
        let ptr: DevicePtr<f32> = DevicePtr::null();
        assert!(ptr.is_null());
        assert_eq!(ptr.as_raw(), 0);
    }

    #[test]
    fn test_device_ptr_offset() {
        let ptr: DevicePtr<f32> = unsafe { DevicePtr::from_raw(1000) };
        let offset_ptr = ptr.byte_offset(100);
        assert_eq!(offset_ptr.as_raw(), 1100);
    }

    #[test]
    fn test_launch_config_linear() {
        let config = LaunchConfig::linear(1000, 256);
        assert_eq!(config.grid.0, 4); // ceil(1000/256) = 4
        assert_eq!(config.block.0, 256);
    }

    #[test]
    fn test_launch_config_total_threads() {
        let config = LaunchConfig::linear(1024, 256);
        assert_eq!(config.total_threads(), 1024);
    }

    #[test]
    fn test_launch_config_2d() {
        let config = LaunchConfig::grid_2d(16, 16, 16, 16);
        assert_eq!(config.total_threads(), 16 * 16 * 16 * 16);
    }

    #[test]
    fn test_device_ptr_clone() {
        let ptr: DevicePtr<f32> = unsafe { DevicePtr::from_raw(0x1000) };
        let cloned = ptr.clone();
        assert_eq!(ptr.as_raw(), cloned.as_raw());
    }

    #[test]
    fn test_device_ptr_not_null() {
        let ptr: DevicePtr<f32> = unsafe { DevicePtr::from_raw(0x1000) };
        assert!(!ptr.is_null());
    }

    #[test]
    fn test_launch_config_with_shared_mem() {
        let config = LaunchConfig::linear(1024, 256).with_shared_mem(4096);
        assert_eq!(config.shared_mem, 4096);
        assert_eq!(config.grid.0, 4);
    }

    #[test]
    fn test_launch_config_default() {
        let config = LaunchConfig::default();
        assert_eq!(config.grid, (1, 1, 1));
        assert_eq!(config.block, (256, 1, 1));
        assert_eq!(config.shared_mem, 0);
    }

    #[test]
    fn test_device_ptr_multiple_offsets() {
        let ptr: DevicePtr<f32> = DevicePtr::null();
        let offset1 = ptr.byte_offset(100);
        let offset2 = offset1.byte_offset(200);
        assert_eq!(offset2.as_raw(), 300);
    }
}
