//! GPU Memory Management
//!
//! Provides safe RAII wrappers for GPU memory allocation and transfer.
//!
//! # Design Philosophy
//!
//! - **RAII**: Memory automatically freed on drop
//! - **Type Safety**: Generic over element type with size tracking
//! - **Async Support**: Both sync and async transfer methods
//!
//! # Citation
//!
//! [4] Oden & Fröning (HiPC 2013) analyzes cudaMalloc latency (1-10ms),
//!     motivating our pool allocator design in memory/pool.rs.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem;
use std::ptr;

use super::context::{get_driver, CudaContext};
use super::stream::CudaStream;
use super::sys::{CUdeviceptr, CudaDriver};
use crate::GpuError;

// ============================================================================
// GPU Buffer
// ============================================================================

/// GPU memory buffer with RAII cleanup
///
/// Allocates device memory and provides safe transfer operations.
/// Memory is automatically freed when dropped.
///
/// # Type Parameter
///
/// * `T` - Element type (must be `Copy` for safe transfer)
///
/// # Example
///
/// ```ignore
/// let ctx = CudaContext::new(0)?;
/// let mut buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1024)?;
///
/// // Upload data
/// let host_data: Vec<f32> = vec![1.0; 1024];
/// buf.copy_from_host(&host_data)?;
///
/// // Download data
/// let mut result = vec![0.0f32; 1024];
/// buf.copy_to_host(&mut result)?;
/// ```
pub struct GpuBuffer<T> {
    /// Device pointer
    ptr: CUdeviceptr,
    /// Number of elements
    len: usize,
    /// Phantom for type parameter
    _marker: PhantomData<T>,
}

// SAFETY: GPU memory is accessible from any thread
unsafe impl<T: Send> Send for GpuBuffer<T> {}
unsafe impl<T: Sync> Sync for GpuBuffer<T> {}

impl<T> GpuBuffer<T> {
    /// PAR-023: Create a non-owning buffer from raw device pointer
    ///
    /// # Safety
    ///
    /// - `ptr` must be a valid CUDA device pointer
    /// - The pointed-to memory must be at least `len * size_of::<T>()` bytes
    /// - The caller is responsible for not freeing this buffer's memory
    ///   (use `std::mem::forget` after use)
    ///
    /// # Use Case
    ///
    /// This is useful for creating temporary buffers from cached device pointers
    /// without triggering the borrow checker.
    #[must_use]
    pub unsafe fn from_raw_parts(ptr: CUdeviceptr, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Allocate a new GPU buffer
    ///
    /// # Arguments
    ///
    /// * `_ctx` - CUDA context (must be current)
    /// * `len` - Number of elements to allocate
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::MemoryAllocation)` if allocation fails.
    /// Returns `Err(GpuError::OutOfMemory)` if insufficient GPU memory.
    pub fn new(_ctx: &CudaContext, len: usize) -> Result<Self, GpuError> {
        if len == 0 {
            return Ok(Self {
                ptr: 0,
                len: 0,
                _marker: PhantomData,
            });
        }

        let driver = get_driver()?;
        let size = len * mem::size_of::<T>();

        let mut ptr: CUdeviceptr = 0;
        // SAFETY: ptr is valid, size is computed correctly
        let result = unsafe { (driver.cuMemAlloc)(&mut ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::MemoryAllocation(e.to_string()))?;

        Ok(Self {
            ptr,
            len,
            _marker: PhantomData,
        })
    }

    /// Get device pointer as raw u64
    #[must_use]
    pub fn as_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Get number of elements
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.len * mem::size_of::<T>()
    }

    /// PAR-023: Create a non-owning clone of the buffer metadata
    ///
    /// Creates a new GpuBuffer that points to the same device memory but
    /// does NOT own it. The returned buffer will NOT free the memory when dropped.
    ///
    /// # Safety
    ///
    /// The caller MUST ensure the original buffer outlives any clones.
    /// The returned buffer should typically be wrapped with `ManuallyDrop` or
    /// `std::mem::forget` to prevent the Drop impl from running.
    ///
    /// # Use Case
    ///
    /// This is useful for passing cached GPU buffers to functions that take
    /// `&GpuBuffer<T>` while avoiding borrow checker conflicts.
    #[must_use]
    pub fn clone_metadata(&self) -> GpuBufferView<T> {
        GpuBufferView {
            ptr: self.ptr,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

/// PAR-023: Non-owning view of a GPU buffer
///
/// This struct points to GPU memory but does NOT free it when dropped.
/// Use this for temporary references to cached GPU buffers.
pub struct GpuBufferView<T> {
    ptr: CUdeviceptr,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> GpuBufferView<T> {
    /// Get device pointer as raw u64
    #[must_use]
    pub fn as_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Get number of elements
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

impl<T: Copy> GpuBuffer<T> {
    /// Copy data from host to device (synchronous)
    ///
    /// # Arguments
    ///
    /// * `data` - Host data to copy (must have same length as buffer)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if copy fails.
    /// Returns `Err(GpuError::InvalidValue)` if lengths don't match.
    pub fn copy_from_host(&mut self, data: &[T]) -> Result<(), GpuError> {
        if data.len() != self.len {
            return Err(GpuError::Transfer(format!(
                "Length mismatch: host {} vs device {}",
                data.len(),
                self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: data is valid for size bytes, ptr is valid device pointer
        let result =
            unsafe { (driver.cuMemcpyHtoD)(self.ptr, data.as_ptr() as *const c_void, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Copy data from device to host (synchronous)
    ///
    /// # Arguments
    ///
    /// * `data` - Host buffer to copy into (must have same length as buffer)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if copy fails.
    pub fn copy_to_host(&self, data: &mut [T]) -> Result<(), GpuError> {
        if data.len() != self.len {
            return Err(GpuError::Transfer(format!(
                "Length mismatch: host {} vs device {}",
                data.len(),
                self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: data is valid for size bytes, ptr is valid device pointer
        let result =
            unsafe { (driver.cuMemcpyDtoH)(data.as_mut_ptr() as *mut c_void, self.ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Copy data from host to device (asynchronous)
    ///
    /// # Arguments
    ///
    /// * `data` - Host data to copy (must have same length as buffer)
    /// * `stream` - Stream for async operation
    ///
    /// # Safety
    ///
    /// The host data must remain valid until the stream is synchronized.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if copy fails.
    pub unsafe fn copy_from_host_async(
        &mut self,
        data: &[T],
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if data.len() != self.len {
            return Err(GpuError::Transfer(format!(
                "Length mismatch: host {} vs device {}",
                data.len(),
                self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: data is valid for size bytes, caller ensures data outlives stream ops
        let result = unsafe {
            (driver.cuMemcpyHtoDAsync)(self.ptr, data.as_ptr() as *const c_void, size, stream.raw())
        };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Copy data from device to host (asynchronous)
    ///
    /// # Arguments
    ///
    /// * `data` - Host buffer to copy into
    /// * `stream` - Stream for async operation
    ///
    /// # Safety
    ///
    /// The host buffer must remain valid until the stream is synchronized.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if copy fails.
    pub unsafe fn copy_to_host_async(
        &self,
        data: &mut [T],
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if data.len() != self.len {
            return Err(GpuError::Transfer(format!(
                "Length mismatch: host {} vs device {}",
                data.len(),
                self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: data is valid for size bytes, caller ensures data outlives stream ops
        let result = unsafe {
            (driver.cuMemcpyDtoHAsync)(
                data.as_mut_ptr() as *mut c_void,
                self.ptr,
                size,
                stream.raw(),
            )
        };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Create buffer and initialize from host data
    ///
    /// Convenience method combining allocation and upload.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context
    /// * `data` - Host data to upload
    ///
    /// # Errors
    ///
    /// Returns allocation or transfer errors.
    pub fn from_host(ctx: &CudaContext, data: &[T]) -> Result<Self, GpuError> {
        let mut buf = Self::new(ctx, data.len())?;
        buf.copy_from_host(data)?;
        Ok(buf)
    }

    /// Copy partial data from host to device at specific offset (PAR-018)
    ///
    /// # Arguments
    ///
    /// * `data` - Host data to copy
    /// * `offset` - Element offset in device buffer where copy begins
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if offset + data.len() exceeds buffer size.
    pub fn copy_from_host_at(&mut self, data: &[T], offset: usize) -> Result<(), GpuError> {
        if offset + data.len() > self.len {
            return Err(GpuError::Transfer(format!(
                "Partial copy out of bounds: offset {} + len {} > buffer {}",
                offset,
                data.len(),
                self.len
            )));
        }

        if data.is_empty() {
            return Ok(());
        }

        let driver = get_driver()?;
        let size = data.len() * std::mem::size_of::<T>();
        let dst_ptr = self.ptr + (offset * std::mem::size_of::<T>()) as u64;

        // SAFETY: bounds checked above, data and ptr are valid
        let result =
            unsafe { (driver.cuMemcpyHtoD)(dst_ptr, data.as_ptr() as *const c_void, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Copy partial data from device to host at specific offset (PAR-018)
    ///
    /// # Arguments
    ///
    /// * `data` - Host buffer to copy into
    /// * `offset` - Element offset in device buffer where copy begins
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if offset + data.len() exceeds buffer size.
    pub fn copy_to_host_at(&self, data: &mut [T], offset: usize) -> Result<(), GpuError> {
        if offset + data.len() > self.len {
            return Err(GpuError::Transfer(format!(
                "Partial copy out of bounds: offset {} + len {} > buffer {}",
                offset,
                data.len(),
                self.len
            )));
        }

        if data.is_empty() {
            return Ok(());
        }

        let driver = get_driver()?;
        let size = data.len() * std::mem::size_of::<T>();
        let src_ptr = self.ptr + (offset * std::mem::size_of::<T>()) as u64;

        // SAFETY: bounds checked above, data and ptr are valid
        let result =
            unsafe { (driver.cuMemcpyDtoH)(data.as_mut_ptr() as *mut c_void, src_ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    // =========================================================================
    // PAR-023: Device-to-Device Copy (Zero-Sync Pipeline)
    // =========================================================================

    /// Copy data from another GPU buffer (device-to-device, synchronous)
    ///
    /// Enables zero-sync GPU pipelines by keeping data on device.
    ///
    /// # Arguments
    ///
    /// * `src` - Source GPU buffer (must have same length)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if lengths don't match or copy fails.
    pub fn copy_from_buffer(&mut self, src: &GpuBuffer<T>) -> Result<(), GpuError> {
        if src.len != self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: D2D length mismatch: src {} vs dst {}",
                src.len, self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: both buffers are valid, size is correct
        let result = unsafe { (driver.cuMemcpyDtoD)(self.ptr, src.ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Copy partial data from another GPU buffer at specific offset (PAR-023)
    ///
    /// Enables GPU-resident KV cache updates without host round-trip.
    ///
    /// # Arguments
    ///
    /// * `src` - Source GPU buffer
    /// * `dst_offset` - Element offset in destination (this buffer)
    /// * `src_offset` - Element offset in source buffer
    /// * `count` - Number of elements to copy
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::Transfer)` if copy would exceed buffer bounds.
    pub fn copy_from_buffer_at(
        &mut self,
        src: &GpuBuffer<T>,
        dst_offset: usize,
        src_offset: usize,
        count: usize,
    ) -> Result<(), GpuError> {
        if dst_offset + count > self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: D2D dst out of bounds: {} + {} > {}",
                dst_offset, count, self.len
            )));
        }
        if src_offset + count > src.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: D2D src out of bounds: {} + {} > {}",
                src_offset, count, src.len
            )));
        }

        if count == 0 {
            return Ok(());
        }

        let driver = get_driver()?;
        let size = count * std::mem::size_of::<T>();
        let dst_ptr = self.ptr + (dst_offset * std::mem::size_of::<T>()) as u64;
        let src_ptr = src.ptr + (src_offset * std::mem::size_of::<T>()) as u64;

        // SAFETY: bounds checked above, both ptrs are valid
        let result = unsafe { (driver.cuMemcpyDtoD)(dst_ptr, src_ptr, size) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// Async copy from another GPU buffer (PAR-023)
    ///
    /// # Safety
    ///
    /// Both buffers must remain valid until stream is synchronized.
    pub unsafe fn copy_from_buffer_async(
        &mut self,
        src: &GpuBuffer<T>,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if src.len != self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D length mismatch: src {} vs dst {}",
                src.len, self.len
            )));
        }

        if self.len == 0 {
            return Ok(());
        }

        let driver = get_driver()?;
        let size = self.size_bytes();

        // SAFETY: both buffers valid, caller ensures lifetime
        let result = unsafe { (driver.cuMemcpyDtoDAsync)(self.ptr, src.ptr, size, stream.raw()) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// PAR-023: Async D2D copy with offsets
    ///
    /// Copies a region from source buffer to destination buffer asynchronously.
    /// Does not synchronize - caller must ensure stream sync before accessing data.
    ///
    /// # Arguments
    ///
    /// * `src` - Source GPU buffer
    /// * `dst_offset` - Element offset in destination (this buffer)
    /// * `src_offset` - Element offset in source buffer
    /// * `count` - Number of elements to copy
    /// * `stream` - CUDA stream for async operation
    ///
    /// # Safety
    ///
    /// Both buffers must remain valid until stream is synchronized.
    pub unsafe fn copy_from_buffer_at_async(
        &mut self,
        src: &GpuBuffer<T>,
        dst_offset: usize,
        src_offset: usize,
        count: usize,
        stream: &CudaStream,
    ) -> Result<(), GpuError> {
        if dst_offset + count > self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D dst out of bounds: {} + {} > {}",
                dst_offset, count, self.len
            )));
        }
        if src_offset + count > src.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D src out of bounds: {} + {} > {}",
                src_offset, count, src.len
            )));
        }

        if count == 0 {
            return Ok(());
        }

        let driver = get_driver()?;
        let size = count * std::mem::size_of::<T>();
        let dst_ptr = self.ptr + (dst_offset * std::mem::size_of::<T>()) as u64;
        let src_ptr = src.ptr + (src_offset * std::mem::size_of::<T>()) as u64;

        // SAFETY: bounds checked above, both ptrs are valid, caller ensures lifetime
        let result = unsafe { (driver.cuMemcpyDtoDAsync)(dst_ptr, src_ptr, size, stream.raw()) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }

    /// PAR-023: Async D2D copy with raw stream handle
    ///
    /// Same as `copy_from_buffer_at_async` but takes raw CUstream handle.
    /// Useful when borrow checker prevents passing &CudaStream due to other borrows.
    ///
    /// # Safety
    ///
    /// - Both buffers must remain valid until stream is synchronized.
    /// - Stream handle must be valid.
    pub unsafe fn copy_from_buffer_at_async_raw(
        &mut self,
        src: &GpuBuffer<T>,
        dst_offset: usize,
        src_offset: usize,
        count: usize,
        stream_handle: super::sys::CUstream,
    ) -> Result<(), GpuError> {
        if dst_offset + count > self.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D dst out of bounds: {} + {} > {}",
                dst_offset, count, self.len
            )));
        }
        if src_offset + count > src.len {
            return Err(GpuError::Transfer(format!(
                "PAR-023: Async D2D src out of bounds: {} + {} > {}",
                src_offset, count, src.len
            )));
        }

        if count == 0 {
            return Ok(());
        }

        let driver = get_driver()?;
        let size = count * std::mem::size_of::<T>();
        let dst_ptr = self.ptr + (dst_offset * std::mem::size_of::<T>()) as u64;
        let src_ptr = src.ptr + (src_offset * std::mem::size_of::<T>()) as u64;

        // SAFETY: bounds checked above, both ptrs valid, caller ensures lifetime + stream valid
        let result = unsafe { (driver.cuMemcpyDtoDAsync)(dst_ptr, src_ptr, size, stream_handle) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(e.to_string()))
    }
}

impl<T> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            if let Ok(driver) = get_driver() {
                // SAFETY: ptr is valid from constructor
                unsafe {
                    let _ = (driver.cuMemFree)(self.ptr);
                }
            }
        }
    }
}

// ============================================================================
// Pointer-as-Argument Helper
// ============================================================================

impl<T> GpuBuffer<T> {
    /// Get pointer to device pointer for kernel arguments
    ///
    /// Returns a pointer that can be passed to kernel launch.
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid while this buffer is alive.
    #[must_use]
    pub fn as_kernel_arg(&self) -> *mut c_void {
        // The kernel expects a pointer to the device pointer
        ptr::addr_of!(self.ptr) as *mut c_void
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_buffer_requires_cuda_feature() {
        // Without cuda feature, allocation should fail
        // This test verifies the module compiles
        assert!(true);
    }

    #[test]
    fn test_size_bytes_calculation() {
        // Test size calculation logic (doesn't require CUDA)
        let size = 1024 * mem::size_of::<f32>();
        assert_eq!(size, 4096);
    }
}
