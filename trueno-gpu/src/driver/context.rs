//! CUDA Context Management
//!
//! Provides safe RAII wrapper for CUDA contexts using the Primary Context API.
//!
//! # Design Philosophy
//!
//! Uses Primary Context API (cuDevicePrimaryCtxRetain) instead of cuCtxCreate:
//! - Shared across all modules in the process
//! - Reference counted by CUDA driver
//! - More efficient for multi-module applications
//!
//! # Citation
//!
//! [5] NVIDIA CUDA C++ Programming Guide v12.3, Section 3.2 "CUDA Contexts"
//!     recommends Primary Context API for applications using multiple modules.

use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

use super::sys::{CUcontext, CUdevice, CudaDriver, CUDA_SUCCESS};
use crate::GpuError;

// ============================================================================
// Global Initialization State
// ============================================================================

/// Track whether cuInit has been called
static CUDA_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Get the CUDA driver, initializing if needed
///
/// # Errors
///
/// Returns `Err(GpuError::CudaNotAvailable)` if CUDA driver is not installed.
/// Returns `Err(GpuError::DeviceInit)` if cuInit fails.
pub fn get_driver() -> Result<&'static CudaDriver, GpuError> {
    let driver = CudaDriver::load()
        .ok_or_else(|| GpuError::CudaNotAvailable("CUDA driver not found".to_string()))?;

    // Initialize CUDA if not already done
    if !CUDA_INITIALIZED.swap(true, Ordering::SeqCst) {
        // SAFETY: cuInit is safe to call multiple times, we just avoid redundant calls
        let result = unsafe { (driver.cuInit)(0) };
        if result != CUDA_SUCCESS {
            CUDA_INITIALIZED.store(false, Ordering::SeqCst);
            return Err(GpuError::DeviceInit(format!(
                "cuInit failed with code {}",
                result
            )));
        }
    }

    Ok(driver)
}

// ============================================================================
// CUDA Context
// ============================================================================

/// CUDA context with RAII cleanup
///
/// Uses Primary Context API for efficient multi-module sharing.
/// Automatically releases context when dropped.
///
/// # Example
///
/// ```ignore
/// let ctx = CudaContext::new(0)?; // Device 0
/// let (free, total) = ctx.memory_info()?;
/// println!("GPU memory: {} / {} bytes free", free, total);
/// ```
pub struct CudaContext {
    /// Device ordinal
    device: CUdevice,
    /// Primary context handle
    context: CUcontext,
}

// SAFETY: CUcontext handles are thread-safe when using Primary Context API
unsafe impl Send for CudaContext {}
unsafe impl Sync for CudaContext {}

impl CudaContext {
    /// Create a new CUDA context for the specified device
    ///
    /// Uses Primary Context API (cuDevicePrimaryCtxRetain) which shares
    /// context across all users in the process.
    ///
    /// # Arguments
    ///
    /// * `device_ordinal` - Device index (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::DeviceNotFound)` if device doesn't exist.
    /// Returns `Err(GpuError::DeviceInit)` if context creation fails.
    pub fn new(device_ordinal: i32) -> Result<Self, GpuError> {
        let driver = get_driver()?;

        // Get device count
        let mut count: i32 = 0;
        // SAFETY: count is a valid pointer
        let result = unsafe { (driver.cuDeviceGetCount)(&mut count) };
        CudaDriver::check(result)?;

        if device_ordinal < 0 || device_ordinal >= count {
            return Err(GpuError::DeviceNotFound(device_ordinal, count as usize));
        }

        // Get device handle
        let mut device: CUdevice = 0;
        // SAFETY: device_ordinal is validated above
        let result = unsafe { (driver.cuDeviceGet)(&mut device, device_ordinal) };
        CudaDriver::check(result)?;

        // Retain primary context
        let mut context: CUcontext = ptr::null_mut();
        // SAFETY: device is a valid handle from cuDeviceGet
        let result = unsafe { (driver.cuDevicePrimaryCtxRetain)(&mut context, device) };
        CudaDriver::check(result)?;

        // Set as current context
        // SAFETY: context is valid from cuDevicePrimaryCtxRetain
        let result = unsafe { (driver.cuCtxSetCurrent)(context) };
        if result != CUDA_SUCCESS {
            // Release context on failure
            unsafe { (driver.cuDevicePrimaryCtxRelease)(device) };
            return Err(GpuError::DeviceInit(format!(
                "cuCtxSetCurrent failed with code {}",
                result
            )));
        }

        Ok(Self { device, context })
    }

    /// Get device ordinal
    #[must_use]
    pub fn device(&self) -> i32 {
        self.device
    }

    /// Get raw context handle
    ///
    /// # Safety
    ///
    /// The returned handle is only valid while this `CudaContext` is alive.
    #[must_use]
    pub fn raw(&self) -> CUcontext {
        self.context
    }

    /// Query free and total device memory
    ///
    /// # Returns
    ///
    /// Tuple of (free_bytes, total_bytes)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::CudaDriver)` if query fails.
    pub fn memory_info(&self) -> Result<(usize, usize), GpuError> {
        let driver = get_driver()?;

        let mut free: usize = 0;
        let mut total: usize = 0;

        // SAFETY: pointers are valid
        let result = unsafe { (driver.cuMemGetInfo)(&mut free, &mut total) };
        CudaDriver::check(result)?;

        Ok((free, total))
    }

    /// Synchronize all work on this context
    ///
    /// Blocks until all preceding commands have completed.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::StreamSync)` if synchronization fails.
    pub fn synchronize(&self) -> Result<(), GpuError> {
        let driver = get_driver()?;

        // SAFETY: context is current (set in constructor)
        let result = unsafe { (driver.cuCtxSynchronize)() };
        CudaDriver::check(result).map_err(|e| GpuError::StreamSync(e.to_string()))
    }

    /// Get device name
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::CudaDriver)` if query fails.
    pub fn device_name(&self) -> Result<String, GpuError> {
        let driver = get_driver()?;

        let mut name = [0i8; 256];
        // SAFETY: buffer is valid and large enough
        let result = unsafe { (driver.cuDeviceGetName)(name.as_mut_ptr(), 256, self.device) };
        CudaDriver::check(result)?;

        // Convert to Rust string
        let name_str = unsafe {
            std::ffi::CStr::from_ptr(name.as_ptr())
                .to_string_lossy()
                .into_owned()
        };

        Ok(name_str)
    }

    /// Get total device memory in bytes
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::CudaDriver)` if query fails.
    pub fn total_memory(&self) -> Result<usize, GpuError> {
        let driver = get_driver()?;

        let mut bytes: usize = 0;
        // SAFETY: pointer is valid, device is valid
        let result = unsafe { (driver.cuDeviceTotalMem)(&mut bytes, self.device) };
        CudaDriver::check(result)?;

        Ok(bytes)
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if let Ok(driver) = get_driver() {
            // SAFETY: device is valid from constructor
            unsafe {
                let _ = (driver.cuDevicePrimaryCtxRelease)(self.device);
            }
        }
    }
}

// ============================================================================
// Device Enumeration
// ============================================================================

/// Get the number of CUDA devices
///
/// # Errors
///
/// Returns `Err(GpuError::CudaNotAvailable)` if CUDA is not available.
pub fn device_count() -> Result<usize, GpuError> {
    let driver = get_driver()?;

    let mut count: i32 = 0;
    // SAFETY: count is a valid pointer
    let result = unsafe { (driver.cuDeviceGetCount)(&mut count) };
    CudaDriver::check(result)?;

    Ok(count as usize)
}

/// Check if CUDA is available
///
/// Returns `true` if CUDA driver is installed and at least one device exists.
#[must_use]
pub fn cuda_available() -> bool {
    device_count().map(|c| c > 0).unwrap_or(false)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_get_driver_without_feature() {
        use super::get_driver;
        let result = get_driver();
        assert!(result.is_err());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_cuda_available_without_feature() {
        use super::cuda_available;
        assert!(!cuda_available());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_device_count_without_feature() {
        use super::device_count;
        let result = device_count();
        assert!(result.is_err());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_context_new_without_feature() {
        use super::CudaContext;
        let result = CudaContext::new(0);
        assert!(result.is_err());
    }
}
