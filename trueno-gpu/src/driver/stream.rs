//! CUDA Stream Management
//!
//! Provides async execution streams for overlapping computation with data transfer.
//!
//! # Design Philosophy
//!
//! Streams enable:
//! - Overlapping H2D copy with kernel execution
//! - Overlapping kernel execution with D2H copy
//! - Parallel kernel execution on different streams
//!
//! # Citation
//!
//! [2] Sourouri et al. (ICPADS 2014) demonstrates that overlapping computation
//!     with communication via CUDA streams is essential for hiding PCIe latency.

use std::ffi::c_void;
use std::ptr;

use super::context::{get_driver, CudaContext};
use super::graph::{CaptureMode, CudaGraph, CudaGraphExec};
use super::module::CudaModule;
use super::sys::{CUfunction, CUstream, CudaDriver, CU_STREAM_NON_BLOCKING};
use super::types::LaunchConfig;
use crate::GpuError;

// ============================================================================
// CUDA Stream
// ============================================================================

/// CUDA execution stream
///
/// Commands submitted to a stream execute in order.
/// Commands on different streams may execute concurrently.
///
/// # RAII
///
/// Stream is automatically destroyed when dropped.
pub struct CudaStream {
    /// Stream handle
    stream: CUstream,
}

// SAFETY: CUstream handles are thread-safe
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Create a new CUDA stream
    ///
    /// Creates a non-blocking stream that doesn't synchronize with stream 0.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::StreamCreate)` if stream creation fails.
    pub fn new(_ctx: &CudaContext) -> Result<Self, GpuError> {
        let driver = get_driver()?;

        let mut stream: CUstream = ptr::null_mut();
        // SAFETY: stream pointer is valid
        let result = unsafe { (driver.cuStreamCreate)(&mut stream, CU_STREAM_NON_BLOCKING) };
        CudaDriver::check(result).map_err(|e| GpuError::StreamCreate(e.to_string()))?;

        Ok(Self { stream })
    }

    /// Get raw stream handle
    ///
    /// # Safety
    ///
    /// The returned handle is only valid while this `CudaStream` is alive.
    #[must_use]
    pub fn raw(&self) -> CUstream {
        self.stream
    }

    /// Synchronize this stream
    ///
    /// Blocks until all commands in this stream have completed.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::StreamSync)` if synchronization fails.
    pub fn synchronize(&self) -> Result<(), GpuError> {
        let driver = get_driver()?;

        // SAFETY: stream is valid from constructor
        let result = unsafe { (driver.cuStreamSynchronize)(self.stream) };
        CudaDriver::check(result).map_err(|e| GpuError::StreamSync(e.to_string()))
    }

    /// PMAT-044: Synchronous device-to-device memory copy.
    ///
    /// Copies `size_bytes` from `src_ptr` to `dst_ptr` on the device.
    /// Both pointers must be valid device pointers with sufficient allocated memory.
    ///
    /// # Safety
    ///
    /// The caller must ensure both device pointers are valid and the copy
    /// does not exceed allocated memory bounds.
    pub fn memcpy_dtod_sync(
        &self,
        dst_ptr: u64,
        src_ptr: u64,
        size_bytes: usize,
    ) -> Result<(), GpuError> {
        if size_bytes == 0 {
            return Ok(());
        }
        let driver = get_driver()?;
        let result = unsafe { (driver.cuMemcpyDtoD)(dst_ptr, src_ptr, size_bytes) };
        CudaDriver::check(result).map_err(|e| GpuError::Transfer(format!("D2D copy failed: {e}")))
    }

    /// Launch a kernel on this stream
    ///
    /// # Arguments
    ///
    /// * `module` - Module containing the kernel
    /// * `func_name` - Name of the kernel function
    /// * `config` - Launch configuration (grid, block, shared memory)
    /// * `args` - Kernel arguments as raw pointers
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `args` contains valid pointers to kernel arguments
    /// - Arguments match the kernel signature
    /// - Device pointers in args are valid
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::KernelLaunch)` if launch fails.
    pub unsafe fn launch_kernel(
        &self,
        module: &mut CudaModule,
        func_name: &str,
        config: &LaunchConfig,
        args: &mut [*mut c_void],
    ) -> Result<(), GpuError> {
        let driver = get_driver()?;
        let func = module.get_function(func_name)?;

        // SAFETY: Caller guarantees args are valid pointers matching kernel signature
        unsafe { self.launch_function(driver, func, config, args) }
    }

    /// Launch a kernel function directly
    ///
    /// # Safety
    ///
    /// Same safety requirements as `launch_kernel`.
    pub unsafe fn launch_function(
        &self,
        driver: &CudaDriver,
        func: CUfunction,
        config: &LaunchConfig,
        args: &mut [*mut c_void],
    ) -> Result<(), GpuError> {
        // SAFETY: func is valid, args contains valid pointers (caller's responsibility)
        let result = unsafe {
            (driver.cuLaunchKernel)(
                func,
                config.grid.0,
                config.grid.1,
                config.grid.2,
                config.block.0,
                config.block.1,
                config.block.2,
                config.shared_mem,
                self.stream,
                args.as_mut_ptr(),
                ptr::null_mut(), // extra (not used)
            )
        };

        CudaDriver::check(result).map_err(|e| GpuError::KernelLaunch(e.to_string()))?;

        Ok(())
    }

    // ========================================================================
    // PAR-037: CUDA Graph Capture
    // ========================================================================

    /// Begin stream capture (PAR-037)
    ///
    /// All subsequent operations on this stream will be recorded into a graph.
    /// Call `end_capture()` to get the captured graph.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::GraphCapture)` if capture cannot be started.
    pub fn begin_capture(&self, mode: CaptureMode) -> Result<(), GpuError> {
        let driver = get_driver()?;
        // SAFETY: stream is valid from constructor
        let result = unsafe { (driver.cuStreamBeginCapture)(self.stream, mode.to_cuda_mode()) };
        CudaDriver::check(result).map_err(|e| GpuError::GraphCapture(e.to_string()))
    }

    /// End stream capture and return the captured graph (PAR-037)
    ///
    /// Returns the graph containing all operations recorded since `begin_capture()`.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::GraphCapture)` if capture cannot be ended.
    pub fn end_capture(&self) -> Result<CudaGraph, GpuError> {
        let driver = get_driver()?;
        let mut graph = ptr::null_mut();
        // SAFETY: stream is valid from constructor
        let result = unsafe { (driver.cuStreamEndCapture)(self.stream, &mut graph) };
        CudaDriver::check(result).map_err(|e| GpuError::GraphCapture(e.to_string()))?;
        Ok(CudaGraph::from_raw(graph))
    }

    /// Launch a captured graph on this stream (PAR-037)
    ///
    /// Replays all operations in the graph with minimal launch overhead.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::GraphLaunch)` if launch fails.
    pub fn launch_graph(&self, exec: &CudaGraphExec) -> Result<(), GpuError> {
        exec.launch(self.stream)
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if let Ok(driver) = get_driver() {
            // SAFETY: stream is valid from constructor
            unsafe {
                let _ = (driver.cuStreamDestroy)(self.stream);
            }
        }
    }
}

// ============================================================================
// Default Stream
// ============================================================================

/// Null stream handle (default stream)
///
/// Operations on the default stream synchronize with all other streams.
/// Use `CudaStream::new()` for non-blocking streams.
pub const DEFAULT_STREAM: CUstream = ptr::null_mut();

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_stream_is_null() {
        assert!(DEFAULT_STREAM.is_null());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_stream_requires_cuda_feature() {
        // This test verifies the module compiles without cuda feature
        assert!(true);
    }
}
