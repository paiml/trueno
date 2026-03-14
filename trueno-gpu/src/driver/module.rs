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

use std::ffi::c_void;
use std::os::raw::c_uint;

use super::context::{get_driver, CudaContext};
use super::ptx_patch::patch_backward_branches_sm121;
use super::sys::{
    CUfunction, CUmodule, CudaDriver, CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_TARGET,
};
use crate::GpuError;

/// CU_JIT_INFO_LOG_BUFFER - Pointer to buffer for info log
const CU_JIT_INFO_LOG_BUFFER: c_uint = 3;
/// CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES - Size of info log buffer
const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: c_uint = 4;

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
    /// Uses `cuModuleLoadDataEx` with explicit JIT target architecture
    /// derived from the device's compute capability. This ensures the JIT
    /// compiler knows exactly which SASS to generate.
    ///
    /// # Contract: F-PTX-002 (Context Currency)
    ///
    /// Ensures the CUDA context is current on the calling thread before
    /// JIT compilation. CUDA contexts are thread-local.
    ///
    /// # Arguments
    ///
    /// * `ctx` - CUDA context (will be made current)
    /// * `ptx` - PTX assembly source code
    ///
    /// # Errors
    ///
    /// Returns `Err(GpuError::ModuleLoad)` if PTX is invalid or compilation fails.
    pub fn from_ptx(ctx: &CudaContext, ptx: &str) -> Result<Self, GpuError> {
        let driver = get_driver()?;

        // F-PTX-002: Ensure context is current on this thread before JIT compilation.
        ctx.make_current()?;

        // Detect device compute capability for JIT target.
        // GH-480: CU_JIT_TARGET must match the PTX `.target` directive.
        // PTX 8.0 only supports up to sm_90, so both PTX source (via
        // CudaContext::sm_target()) and JIT target must clamp to sm_90.
        // The sm_90 SASS runs on Blackwell (sm_121) via forward compatibility.
        // Passing CU_JIT_TARGET=121 with `.target sm_90` PTX causes the driver
        // to generate incorrect native sm_121 SASS — garbage inference results.
        let (major, minor) = ctx.compute_capability()?;
        let (jit_major, jit_minor) =
            if major > 9 || (major == 9 && minor > 0) { (9, 0) } else { (major, minor) };
        let jit_target: c_uint = (jit_major * 10 + jit_minor) as c_uint;

        // GH-480: Patch unconditional backward branches for Blackwell (sm_121+).
        // The CUDA 13.0 JIT miscompiles while-loops (unconditional backward
        // branches) on sm_121, silently dropping loop iterations. Converting
        // them to conditional branches avoids the buggy JIT optimizer path.
        let ptx_patched = if major >= 12 { patch_backward_branches_sm121(ptx) } else { None };
        let ptx: &str = ptx_patched.as_deref().unwrap_or(ptx);

        // Ensure PTX is null-terminated
        let ptx_cstring = CString::new(ptx.as_bytes().to_vec())
            .map_err(|_| GpuError::ModuleLoad("PTX contains null bytes".to_string()))?;

        // Try 1: cuModuleLoadDataEx with explicit JIT target + log buffers
        let mut info_log = vec![0u8; 4096];
        let mut error_log = vec![0u8; 4096];
        let info_log_size: usize = info_log.len();
        let error_log_size: usize = error_log.len();

        let mut options: [c_uint; 5] = [
            CU_JIT_TARGET,
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        ];
        let mut option_values: [*mut c_void; 5] = [
            jit_target as *mut c_void,
            info_log.as_mut_ptr() as *mut c_void,
            info_log_size as *mut c_void,
            error_log.as_mut_ptr() as *mut c_void,
            error_log_size as *mut c_void,
        ];

        // SAFETY: ptx_cstring is valid null-terminated string, options arrays
        // are valid for the lifetime of this call, context is current.
        let mut module: CUmodule = ptr::null_mut();
        let result = unsafe {
            (driver.cuModuleLoadDataEx)(
                &mut module,
                ptx_cstring.as_ptr() as *const _,
                5,
                options.as_mut_ptr(),
                option_values.as_mut_ptr(),
            )
        };

        if CudaDriver::check(result).is_ok() {
            return Ok(Self { module, functions: HashMap::new() });
        }

        // Try 1 failed — capture diagnostics
        let kernel_name =
            ptx.lines().find(|l| l.contains(".entry")).map(|l| l.trim()).unwrap_or("<unknown>");
        let jit_info = String::from_utf8_lossy(&info_log).trim_end_matches('\0').to_string();
        let jit_err = String::from_utf8_lossy(&error_log).trim_end_matches('\0').to_string();

        eprintln!(
            "[PTX-JIT] Try 1 failed: {kernel_name}, target: sm_{major}{minor}, \
             PTX: {} bytes, result: {result}",
            ptx.len()
        );
        if !jit_info.is_empty() {
            eprintln!("[PTX-JIT] Info log: {jit_info}");
        }
        if !jit_err.is_empty() {
            eprintln!("[PTX-JIT] Error log: {jit_err}");
        }

        // Dump PTX to /tmp for offline diagnosis (#127)
        let dump_path = format!(
            "/tmp/failed-ptx-sm_{major}{minor}-{}.ptx",
            kernel_name.replace(|c: char| !c.is_alphanumeric() && c != '_', "_")
        );
        if let Ok(()) = std::fs::write(&dump_path, ptx) {
            eprintln!("[PTX-JIT] PTX dumped to {dump_path}");
        }

        // Try 2: cuModuleLoadData without explicit JIT target (let driver auto-detect)
        eprintln!("[PTX-JIT] Retrying with cuModuleLoadData (no explicit target)...");
        let mut module2: CUmodule = ptr::null_mut();
        let result2 =
            unsafe { (driver.cuModuleLoadData)(&mut module2, ptx_cstring.as_ptr() as *const _) };

        if CudaDriver::check(result2).is_ok() {
            eprintln!("[PTX-JIT] Fallback succeeded for {kernel_name}");
            return Ok(Self { module: module2, functions: HashMap::new() });
        }

        // Both attempts failed
        eprintln!("[PTX-JIT] Both attempts failed for {kernel_name}");
        Err(GpuError::ModuleLoad(format!(
            "CUDA module loading failed: try1={result} try2={result2} (JIT target: sm_{major}{minor})"
        )))
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
