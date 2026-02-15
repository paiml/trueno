//! GPU buffer types and core operations
//!
//! Defines `GpuBuffer<T>` (owning) and `GpuBufferView<T>` (non-owning)
//! with allocation, deallocation, and metadata access.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem;
use std::ptr;

use crate::driver::context::{get_driver, CudaContext};
use crate::driver::sys::{CUdeviceptr, CudaDriver};
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
    pub(super) ptr: CUdeviceptr,
    /// Number of elements
    pub(super) len: usize,
    /// Phantom for type parameter
    pub(super) _marker: PhantomData<T>,
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

// ============================================================================
// GPU Buffer View (non-owning)
// ============================================================================

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

// ============================================================================
// Drop + Kernel Arg
// ============================================================================

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
