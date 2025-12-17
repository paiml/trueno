//! PTX Module Loading and JIT Compilation
//!
//! Loads PTX source into GPU-executable modules.
//! Uses OUR OWN FFI from driver/sys.rs - no external dependencies.
//!
//! # Design Philosophy
//!
//! PTX is JIT-compiled to SASS (device assembly) at load time.
//! This incurs one-time cost but enables runtime architecture targeting.
//!
//! # Citation
//!
//! [5] NVIDIA CUDA C++ Programming Guide v12.3, Section 3.3 "Modules"

use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;

use super::context::{get_driver, CudaContext};
use super::sys::{CUfunction, CUmodule, CudaDriver};
use crate::GpuError;

// ============================================================================
// CUDA Module
// ============================================================================

/// Compiled CUDA module containing kernels
///
/// Loads PTX source and JIT compiles to device-specific SASS.
/// Caches function handles for efficient lookup.
///
/// # RAII
///
/// Module is automatically unloaded when dropped.
pub struct CudaModule {
    /// Module handle
    module: CUmodule,
    /// Cached function handles
    functions: HashMap<String, CUfunction>,
}

// SAFETY: CUmodule handles are thread-safe for read-only operations
unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

impl CudaModule {
    /// Load PTX source and JIT compile to device code
    ///
    /// # Arguments
    ///
    /// * `_ctx` - CUDA context (must be current)
    /// * `ptx` - PTX assembly source code
    ///
    /// # JIT Compilation
    ///
    /// The PTX is compiled to SASS (device assembly) at load time.
    /// This incurs one-time cost but enables runtime architecture targeting.
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::ModuleLoad)` if PTX is invalid or compilation fails.
    pub fn from_ptx(_ctx: &CudaContext, ptx: &str) -> Result<Self, GpuError> {
        let driver = get_driver()?;

        // Ensure PTX is null-terminated
        let ptx_cstring = CString::new(ptx)
            .map_err(|_| GpuError::ModuleLoad("PTX contains null bytes".to_string()))?;

        // SAFETY: ptx_cstring is valid null-terminated string
        let mut module: CUmodule = ptr::null_mut();
        let result =
            unsafe { (driver.cuModuleLoadData)(&mut module, ptx_cstring.as_ptr() as *const _) };
        CudaDriver::check(result).map_err(|e| GpuError::ModuleLoad(e.to_string()))?;

        Ok(Self {
            module,
            functions: HashMap::new(),
        })
    }

    /// Get kernel function handle by name
    ///
    /// Function handles are cached for efficient repeated lookup.
    ///
    /// # Arguments
    ///
    /// * `name` - Kernel function name (must match PTX .entry name)
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::FunctionNotFound)` if function doesn't exist.
    pub fn get_function(&mut self, name: &str) -> Result<CUfunction, GpuError> {
        // Check cache first
        if let Some(&func) = self.functions.get(name) {
            return Ok(func);
        }

        let driver = get_driver()?;
        let name_cstring =
            CString::new(name).map_err(|_| GpuError::FunctionNotFound(name.to_string()))?;

        // SAFETY: module is valid, name_cstring is null-terminated
        let mut func: CUfunction = ptr::null_mut();
        let result =
            unsafe { (driver.cuModuleGetFunction)(&mut func, self.module, name_cstring.as_ptr()) };
        CudaDriver::check(result).map_err(|_| GpuError::FunctionNotFound(name.to_string()))?;

        // Cache for future lookups
        self.functions.insert(name.to_string(), func);
        Ok(func)
    }

    /// Get raw module handle
    ///
    /// # Safety
    ///
    /// The returned handle is only valid while this `CudaModule` is alive.
    #[must_use]
    pub fn raw(&self) -> CUmodule {
        self.module
    }

    /// Check if a function exists in the module
    pub fn has_function(&mut self, name: &str) -> bool {
        self.get_function(name).is_ok()
    }

    /// Get list of cached function names
    #[must_use]
    pub fn cached_functions(&self) -> Vec<&str> {
        self.functions.keys().map(String::as_str).collect()
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if let Ok(driver) = get_driver() {
            // SAFETY: module is valid from constructor
            unsafe {
                let _ = (driver.cuModuleUnload)(self.module);
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_module_requires_cuda_feature() {
        // Without cuda feature, we can't create modules
        // This test just verifies the module compiles
        assert!(true);
    }
}
