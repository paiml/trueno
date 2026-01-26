//! PAR-037: CUDA Graph Capture and Execution
//!
//! Provides CUDA graph capture for pre-recording kernel sequences.
//!
//! ## Benefits
//!
//! - ~3-10µs graph launch vs ~20-50µs per kernel launch
//! - Pre-validated kernel parameters
//! - Reduced CPU overhead for repetitive sequences
//!
//! ## Usage
//!
//! ```text
//! 1. stream.begin_capture()        - Start capture mode
//! 2. Execute kernel sequence       - Record operations
//! 3. stream.end_capture()          - Get captured graph
//! 4. graph.instantiate()           - Create executable
//! 5. stream.launch_graph(&exec)    - Replay sequence
//! ```

use std::ptr;

use super::context::get_driver;
use super::sys::{CUgraph, CUgraphExec, CUstream, CudaDriver};
use crate::GpuError;

// ============================================================================
// CUDA Graph
// ============================================================================

/// Captured CUDA graph
///
/// Represents a dependency graph of operations captured from stream execution.
/// Must be instantiated before execution.
pub struct CudaGraph {
    /// Graph handle
    graph: CUgraph,
}

// SAFETY: CUgraph handles are thread-safe
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl CudaGraph {
    /// Create a new empty CUDA graph (for manual construction)
    pub fn new() -> Result<Self, GpuError> {
        let driver = get_driver()?;

        let mut graph: CUgraph = ptr::null_mut();
        // SAFETY: graph pointer is valid
        let result = unsafe { (driver.cuGraphCreate)(&mut graph, 0) };
        CudaDriver::check(result).map_err(|e| GpuError::GraphCreate(e.to_string()))?;

        Ok(Self { graph })
    }

    /// Create from a captured stream
    ///
    /// This is called internally by `CudaStream::end_capture`.
    pub(crate) fn from_raw(graph: CUgraph) -> Self {
        Self { graph }
    }

    /// Get raw graph handle
    #[must_use]
    pub fn raw(&self) -> CUgraph {
        self.graph
    }

    /// Instantiate the graph for execution
    ///
    /// Creates an executable graph instance that can be launched on streams.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::GraphInstantiate)` if instantiation fails.
    pub fn instantiate(&self) -> Result<CudaGraphExec, GpuError> {
        let driver = get_driver()?;

        let mut graph_exec: CUgraphExec = ptr::null_mut();
        // SAFETY: graph and graph_exec pointers are valid
        let result =
            unsafe { (driver.cuGraphInstantiateWithFlags)(&mut graph_exec, self.graph, 0) };
        CudaDriver::check(result).map_err(|e| GpuError::GraphInstantiate(e.to_string()))?;

        Ok(CudaGraphExec::from_raw(graph_exec))
    }
}

impl Default for CudaGraph {
    fn default() -> Self {
        Self::new().expect("Failed to create CUDA graph")
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.graph.is_null() {
            if let Ok(driver) = get_driver() {
                // SAFETY: graph is valid from constructor
                unsafe { (driver.cuGraphDestroy)(self.graph) };
            }
        }
    }
}

// ============================================================================
// CUDA Graph Executable
// ============================================================================

/// Executable CUDA graph instance
///
/// Instantiated from a `CudaGraph`, this can be launched on streams
/// with minimal overhead (~3-10µs vs ~20-50µs per kernel).
pub struct CudaGraphExec {
    /// Graph executable handle
    exec: CUgraphExec,
}

// SAFETY: CUgraphExec handles are thread-safe
unsafe impl Send for CudaGraphExec {}
unsafe impl Sync for CudaGraphExec {}

impl CudaGraphExec {
    /// Create from raw handle (internal)
    pub(crate) fn from_raw(exec: CUgraphExec) -> Self {
        Self { exec }
    }

    /// Get raw executable handle
    #[must_use]
    pub fn raw(&self) -> CUgraphExec {
        self.exec
    }

    /// Launch this graph on a stream
    ///
    /// Replays all captured operations with minimal overhead.
    ///
    /// # Arguments
    ///
    /// * `stream` - Stream to launch on (use raw handle)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::GraphLaunch)` if launch fails.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `stream` is a valid CUDA stream handle.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn launch(&self, stream: CUstream) -> Result<(), GpuError> {
        let driver = get_driver()?;

        // SAFETY: exec and stream handles are valid
        let result = unsafe { (driver.cuGraphLaunch)(self.exec, stream) };
        CudaDriver::check(result).map_err(|e| GpuError::GraphLaunch(e.to_string()))
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            if let Ok(driver) = get_driver() {
                // SAFETY: exec is valid from constructor
                unsafe { (driver.cuGraphExecDestroy)(self.exec) };
            }
        }
    }
}

// ============================================================================
// Stream Capture Mode Extension
// ============================================================================

/// Capture mode for CUDA graphs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaptureMode {
    /// Global mode - all operations on any stream are captured
    #[default]
    Global,
    /// Thread-local mode - only operations from capturing thread are captured
    ThreadLocal,
    /// Relaxed mode - allows dependencies from other streams
    Relaxed,
}

impl CaptureMode {
    /// Convert to CUDA capture mode constant
    #[must_use]
    pub fn to_cuda_mode(self) -> u32 {
        match self {
            CaptureMode::Global => 0,      // CU_STREAM_CAPTURE_MODE_GLOBAL
            CaptureMode::ThreadLocal => 1, // CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
            CaptureMode::Relaxed => 2,     // CU_STREAM_CAPTURE_MODE_RELAXED
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capture_mode_values() {
        assert_eq!(CaptureMode::Global.to_cuda_mode(), 0);
        assert_eq!(CaptureMode::ThreadLocal.to_cuda_mode(), 1);
        assert_eq!(CaptureMode::Relaxed.to_cuda_mode(), 2);
    }

    #[test]
    fn test_capture_mode_default() {
        assert_eq!(CaptureMode::default(), CaptureMode::Global);
    }
}
