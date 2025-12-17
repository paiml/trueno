//! Minimal CUDA Driver API FFI Bindings
//!
//! Hand-written FFI for the ~20 CUDA driver functions we actually need.
//! No external dependencies. Dynamic loading via libcuda.so/nvcuda.dll.
//!
//! # Design Philosophy
//!
//! **OWN THE STACK**: We built 5,500 lines of PTX generation from scratch.
//! We can build 400 lines of CUDA FFI. Total control, no third-party surprises.
//!
//! # Safety
//!
//! All functions in this module are unsafe. Safe wrappers are provided in
//! sibling modules (context.rs, module.rs, stream.rs, memory.rs).
//!
//! # Clippy Allows
//!
//! This module uses FFI-specific patterns that trigger clippy lints:
//! - `borrow_as_ptr`: FFI requires `&mut T` -> `*mut T` conversion
//! - `ptr_as_ptr`: FFI pointer casts are intentional
//! - `cast_sign_loss`: CUDA uses i32 for counts, we use usize
//!
//! # Citation
//!
//! [1] RustBelt (Jung et al., POPL 2017) proves Rust's type system safely
//!     encapsulates unsafe FFI operations via ownership and borrowing.

use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint};

use crate::GpuError;

// ============================================================================
// CUDA Type Definitions (from cuda.h)
// ============================================================================

/// CUDA error code
pub type CUresult = c_int;

/// CUDA device handle
pub type CUdevice = c_int;

/// CUDA context handle (opaque pointer)
pub type CUcontext = *mut c_void;

/// CUDA module handle (opaque pointer)
pub type CUmodule = *mut c_void;

/// CUDA function handle (opaque pointer)
pub type CUfunction = *mut c_void;

/// CUDA stream handle (opaque pointer)
pub type CUstream = *mut c_void;

/// CUDA device pointer (GPU memory address)
pub type CUdeviceptr = u64;

// ============================================================================
// CUDA Error Codes (subset we handle)
// ============================================================================

/// CUDA success
pub const CUDA_SUCCESS: CUresult = 0;
/// Invalid value passed
pub const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
/// Out of memory
pub const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
/// CUDA not initialized
pub const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
/// CUDA deinitialized
pub const CUDA_ERROR_DEINITIALIZED: CUresult = 4;
/// No CUDA device
pub const CUDA_ERROR_NO_DEVICE: CUresult = 100;
/// Invalid device
pub const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;
/// Invalid PTX
pub const CUDA_ERROR_INVALID_PTX: CUresult = 218;
/// Function not found
pub const CUDA_ERROR_NOT_FOUND: CUresult = 500;

// ============================================================================
// CUDA Stream Flags
// ============================================================================

/// Default stream creation flag
pub const CU_STREAM_DEFAULT: c_uint = 0;
/// Non-blocking stream (doesn't synchronize with stream 0)
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;

// ============================================================================
// CUDA Driver Function Pointers
// ============================================================================

/// Dynamically loaded CUDA driver functions
///
/// All function pointers are loaded at runtime from libcuda.so (Linux)
/// or nvcuda.dll (Windows). This avoids link-time dependency on CUDA.
#[allow(non_snake_case)]
pub struct CudaDriver {
    // Initialization
    /// cuInit - Initialize the CUDA driver
    pub cuInit: unsafe extern "C" fn(flags: c_uint) -> CUresult,

    // Device Management
    /// cuDeviceGetCount - Get number of CUDA devices
    pub cuDeviceGetCount: unsafe extern "C" fn(count: *mut c_int) -> CUresult,
    /// cuDeviceGet - Get device handle by ordinal
    pub cuDeviceGet: unsafe extern "C" fn(device: *mut CUdevice, ordinal: c_int) -> CUresult,
    /// cuDeviceGetName - Get device name
    pub cuDeviceGetName:
        unsafe extern "C" fn(name: *mut c_char, len: c_int, device: CUdevice) -> CUresult,
    /// cuDeviceTotalMem - Get total device memory
    pub cuDeviceTotalMem: unsafe extern "C" fn(bytes: *mut usize, device: CUdevice) -> CUresult,

    // Context Management (Primary Context API - preferred)
    /// cuDevicePrimaryCtxRetain - Retain primary context
    pub cuDevicePrimaryCtxRetain:
        unsafe extern "C" fn(ctx: *mut CUcontext, device: CUdevice) -> CUresult,
    /// cuDevicePrimaryCtxRelease - Release primary context
    pub cuDevicePrimaryCtxRelease: unsafe extern "C" fn(device: CUdevice) -> CUresult,
    /// cuCtxSetCurrent - Set current context
    pub cuCtxSetCurrent: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    /// cuCtxSynchronize - Synchronize current context
    pub cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,

    // Module Management
    /// cuModuleLoadData - Load module from PTX/cubin data
    pub cuModuleLoadData:
        unsafe extern "C" fn(module: *mut CUmodule, image: *const c_void) -> CUresult,
    /// cuModuleUnload - Unload module
    pub cuModuleUnload: unsafe extern "C" fn(module: CUmodule) -> CUresult,
    /// cuModuleGetFunction - Get function from module
    pub cuModuleGetFunction: unsafe extern "C" fn(
        func: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult,

    // Memory Management
    /// cuMemAlloc - Allocate device memory
    pub cuMemAlloc: unsafe extern "C" fn(ptr: *mut CUdeviceptr, size: usize) -> CUresult,
    /// cuMemFree - Free device memory
    pub cuMemFree: unsafe extern "C" fn(ptr: CUdeviceptr) -> CUresult,
    /// cuMemcpyHtoD - Copy from host to device
    pub cuMemcpyHtoD:
        unsafe extern "C" fn(dst: CUdeviceptr, src: *const c_void, size: usize) -> CUresult,
    /// cuMemcpyDtoH - Copy from device to host
    pub cuMemcpyDtoH:
        unsafe extern "C" fn(dst: *mut c_void, src: CUdeviceptr, size: usize) -> CUresult,
    /// cuMemcpyHtoDAsync - Async copy from host to device
    pub cuMemcpyHtoDAsync: unsafe extern "C" fn(
        dst: CUdeviceptr,
        src: *const c_void,
        size: usize,
        stream: CUstream,
    ) -> CUresult,
    /// cuMemcpyDtoHAsync - Async copy from device to host
    pub cuMemcpyDtoHAsync: unsafe extern "C" fn(
        dst: *mut c_void,
        src: CUdeviceptr,
        size: usize,
        stream: CUstream,
    ) -> CUresult,
    /// cuMemGetInfo - Get free and total memory
    pub cuMemGetInfo: unsafe extern "C" fn(free: *mut usize, total: *mut usize) -> CUresult,

    // Stream Management
    /// cuStreamCreate - Create a stream
    pub cuStreamCreate: unsafe extern "C" fn(stream: *mut CUstream, flags: c_uint) -> CUresult,
    /// cuStreamDestroy - Destroy a stream
    pub cuStreamDestroy: unsafe extern "C" fn(stream: CUstream) -> CUresult,
    /// cuStreamSynchronize - Synchronize a stream
    pub cuStreamSynchronize: unsafe extern "C" fn(stream: CUstream) -> CUresult,

    // Kernel Launch
    /// cuLaunchKernel - Launch a kernel
    #[allow(clippy::type_complexity)]
    pub cuLaunchKernel: unsafe extern "C" fn(
        func: CUfunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult,
}

// ============================================================================
// Dynamic Loading
// ============================================================================

#[cfg(feature = "cuda")]
mod loading {
    use super::*;
    use libloading::{Library, Symbol};
    use std::sync::OnceLock;

    /// Global driver instance (loaded once)
    static DRIVER: OnceLock<Option<CudaDriver>> = OnceLock::new();

    /// Library handle (must outlive function pointers)
    static LIBRARY: OnceLock<Option<Library>> = OnceLock::new();

    impl CudaDriver {
        /// Load CUDA driver dynamically
        ///
        /// Returns `None` if CUDA is not available (no driver installed).
        /// This is NOT an error - it's expected on systems without NVIDIA GPUs.
        ///
        /// # Safety
        ///
        /// This function loads a shared library and extracts function pointers.
        /// The library must remain loaded for the lifetime of the returned driver.
        #[must_use]
        pub fn load() -> Option<&'static Self> {
            // Initialize library first
            let _ = LIBRARY.get_or_init(|| {
                // Try platform-specific library names
                #[cfg(target_os = "linux")]
                let lib_names = ["libcuda.so.1", "libcuda.so"];
                #[cfg(target_os = "windows")]
                let lib_names = ["nvcuda.dll"];
                #[cfg(target_os = "macos")]
                let lib_names: [&str; 0] = []; // No CUDA on macOS

                for name in lib_names {
                    // SAFETY: We're loading a well-known system library
                    if let Ok(lib) = unsafe { Library::new(name) } {
                        return Some(lib);
                    }
                }
                None
            });

            // Then initialize driver
            DRIVER
                .get_or_init(|| {
                    let lib = LIBRARY.get()?.as_ref()?;
                    Self::load_from_library(lib)
                })
                .as_ref()
        }

        /// Load function pointers from library
        fn load_from_library(lib: &Library) -> Option<Self> {
            // SAFETY: All symbols are standard CUDA driver API functions
            unsafe {
                // Helper macro to load symbols with explicit type
                macro_rules! load_sym {
                    ($name:ident, $ty:ty) => {{
                        let sym: Symbol<'_, $ty> = lib.get(stringify!($name).as_bytes()).ok()?;
                        *sym
                    }};
                }

                type FnInit = unsafe extern "C" fn(c_uint) -> CUresult;
                type FnDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CUresult;
                type FnDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
                type FnDeviceGetName =
                    unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
                type FnDeviceTotalMem = unsafe extern "C" fn(*mut usize, CUdevice) -> CUresult;
                type FnPrimaryCtxRetain =
                    unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult;
                type FnPrimaryCtxRelease = unsafe extern "C" fn(CUdevice) -> CUresult;
                type FnCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
                type FnCtxSync = unsafe extern "C" fn() -> CUresult;
                type FnModuleLoadData =
                    unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
                type FnModuleUnload = unsafe extern "C" fn(CUmodule) -> CUresult;
                type FnModuleGetFunction =
                    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
                type FnMemAlloc = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
                type FnMemFree = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
                type FnMemcpyHtoD =
                    unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
                type FnMemcpyDtoH =
                    unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
                type FnMemcpyHtoDAsync =
                    unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
                type FnMemcpyDtoHAsync =
                    unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;
                type FnMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult;
                type FnStreamCreate = unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult;
                type FnStreamDestroy = unsafe extern "C" fn(CUstream) -> CUresult;
                type FnStreamSync = unsafe extern "C" fn(CUstream) -> CUresult;
                type FnLaunchKernel = unsafe extern "C" fn(
                    CUfunction,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    c_uint,
                    CUstream,
                    *mut *mut c_void,
                    *mut *mut c_void,
                ) -> CUresult;

                Some(CudaDriver {
                    cuInit: load_sym!(cuInit, FnInit),
                    cuDeviceGetCount: load_sym!(cuDeviceGetCount, FnDeviceGetCount),
                    cuDeviceGet: load_sym!(cuDeviceGet, FnDeviceGet),
                    cuDeviceGetName: load_sym!(cuDeviceGetName, FnDeviceGetName),
                    cuDeviceTotalMem: load_sym!(cuDeviceTotalMem_v2, FnDeviceTotalMem),
                    cuDevicePrimaryCtxRetain: load_sym!(
                        cuDevicePrimaryCtxRetain,
                        FnPrimaryCtxRetain
                    ),
                    cuDevicePrimaryCtxRelease: load_sym!(
                        cuDevicePrimaryCtxRelease_v2,
                        FnPrimaryCtxRelease
                    ),
                    cuCtxSetCurrent: load_sym!(cuCtxSetCurrent, FnCtxSetCurrent),
                    cuCtxSynchronize: load_sym!(cuCtxSynchronize, FnCtxSync),
                    cuModuleLoadData: load_sym!(cuModuleLoadData, FnModuleLoadData),
                    cuModuleUnload: load_sym!(cuModuleUnload, FnModuleUnload),
                    cuModuleGetFunction: load_sym!(cuModuleGetFunction, FnModuleGetFunction),
                    cuMemAlloc: load_sym!(cuMemAlloc_v2, FnMemAlloc),
                    cuMemFree: load_sym!(cuMemFree_v2, FnMemFree),
                    cuMemcpyHtoD: load_sym!(cuMemcpyHtoD_v2, FnMemcpyHtoD),
                    cuMemcpyDtoH: load_sym!(cuMemcpyDtoH_v2, FnMemcpyDtoH),
                    cuMemcpyHtoDAsync: load_sym!(cuMemcpyHtoDAsync_v2, FnMemcpyHtoDAsync),
                    cuMemcpyDtoHAsync: load_sym!(cuMemcpyDtoHAsync_v2, FnMemcpyDtoHAsync),
                    cuMemGetInfo: load_sym!(cuMemGetInfo_v2, FnMemGetInfo),
                    cuStreamCreate: load_sym!(cuStreamCreate, FnStreamCreate),
                    cuStreamDestroy: load_sym!(cuStreamDestroy_v2, FnStreamDestroy),
                    cuStreamSynchronize: load_sym!(cuStreamSynchronize, FnStreamSync),
                    cuLaunchKernel: load_sym!(cuLaunchKernel, FnLaunchKernel),
                })
            }
        }

        /// Check CUDA result and convert to GpuError
        ///
        /// # Errors
        ///
        /// Returns `Err(GpuError::CudaDriver)` if result is not CUDA_SUCCESS.
        pub fn check(result: CUresult) -> Result<(), GpuError> {
            if result == CUDA_SUCCESS {
                Ok(())
            } else {
                Err(GpuError::CudaDriver(
                    cuda_error_string(result).to_string(),
                    result,
                ))
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod loading {
    use super::*;

    impl CudaDriver {
        /// CUDA not available without feature
        #[must_use]
        pub fn load() -> Option<&'static Self> {
            None
        }

        /// Check is a no-op without CUDA
        pub fn check(_result: CUresult) -> Result<(), GpuError> {
            Err(GpuError::CudaNotAvailable(
                "cuda feature not enabled".to_string(),
            ))
        }
    }
}

// ============================================================================
// Error String Conversion
// ============================================================================

/// Convert CUDA error code to human-readable string
#[must_use]
pub fn cuda_error_string(code: CUresult) -> &'static str {
    match code {
        CUDA_SUCCESS => "CUDA_SUCCESS",
        CUDA_ERROR_INVALID_VALUE => "CUDA_ERROR_INVALID_VALUE",
        CUDA_ERROR_OUT_OF_MEMORY => "CUDA_ERROR_OUT_OF_MEMORY",
        CUDA_ERROR_NOT_INITIALIZED => "CUDA_ERROR_NOT_INITIALIZED",
        CUDA_ERROR_DEINITIALIZED => "CUDA_ERROR_DEINITIALIZED",
        CUDA_ERROR_NO_DEVICE => "CUDA_ERROR_NO_DEVICE",
        CUDA_ERROR_INVALID_DEVICE => "CUDA_ERROR_INVALID_DEVICE",
        CUDA_ERROR_INVALID_PTX => "CUDA_ERROR_INVALID_PTX",
        CUDA_ERROR_NOT_FOUND => "CUDA_ERROR_NOT_FOUND",
        _ => "CUDA_ERROR_UNKNOWN",
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_error_string_success() {
        assert_eq!(cuda_error_string(CUDA_SUCCESS), "CUDA_SUCCESS");
    }

    #[test]
    fn test_cuda_error_string_oom() {
        assert_eq!(
            cuda_error_string(CUDA_ERROR_OUT_OF_MEMORY),
            "CUDA_ERROR_OUT_OF_MEMORY"
        );
    }

    #[test]
    fn test_cuda_error_string_unknown() {
        assert_eq!(cuda_error_string(99999), "CUDA_ERROR_UNKNOWN");
    }

    #[test]
    fn test_cuda_constants() {
        // Verify constants match CUDA header
        assert_eq!(CUDA_SUCCESS, 0);
        assert_eq!(CUDA_ERROR_NO_DEVICE, 100);
        assert_eq!(CUDA_ERROR_INVALID_PTX, 218);
    }

    #[test]
    fn test_custream_flags() {
        assert_eq!(CU_STREAM_DEFAULT, 0);
        assert_eq!(CU_STREAM_NON_BLOCKING, 1);
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_driver_load_without_feature() {
        assert!(CudaDriver::load().is_none());
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_check_without_feature() {
        let result = CudaDriver::check(CUDA_SUCCESS);
        assert!(result.is_err());
    }

    #[test]
    fn test_all_error_strings() {
        // Test all known error codes have proper strings
        assert_eq!(
            cuda_error_string(CUDA_ERROR_INVALID_VALUE),
            "CUDA_ERROR_INVALID_VALUE"
        );
        assert_eq!(
            cuda_error_string(CUDA_ERROR_NOT_INITIALIZED),
            "CUDA_ERROR_NOT_INITIALIZED"
        );
        assert_eq!(
            cuda_error_string(CUDA_ERROR_DEINITIALIZED),
            "CUDA_ERROR_DEINITIALIZED"
        );
        assert_eq!(
            cuda_error_string(CUDA_ERROR_INVALID_DEVICE),
            "CUDA_ERROR_INVALID_DEVICE"
        );
        assert_eq!(
            cuda_error_string(CUDA_ERROR_NOT_FOUND),
            "CUDA_ERROR_NOT_FOUND"
        );
    }

    #[test]
    fn test_error_codes_are_distinct() {
        // All error codes should be distinct
        let codes = [
            CUDA_SUCCESS,
            CUDA_ERROR_INVALID_VALUE,
            CUDA_ERROR_OUT_OF_MEMORY,
            CUDA_ERROR_NOT_INITIALIZED,
            CUDA_ERROR_DEINITIALIZED,
            CUDA_ERROR_NO_DEVICE,
            CUDA_ERROR_INVALID_DEVICE,
            CUDA_ERROR_INVALID_PTX,
            CUDA_ERROR_NOT_FOUND,
        ];
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                assert_ne!(
                    codes[i], codes[j],
                    "Error codes at {} and {} are equal",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_type_sizes() {
        // Verify FFI types have expected sizes
        assert_eq!(std::mem::size_of::<CUresult>(), std::mem::size_of::<i32>());
        assert_eq!(std::mem::size_of::<CUdevice>(), std::mem::size_of::<i32>());
        assert_eq!(
            std::mem::size_of::<CUdeviceptr>(),
            std::mem::size_of::<u64>()
        );
        // Opaque pointers are pointer-sized
        assert_eq!(
            std::mem::size_of::<CUcontext>(),
            std::mem::size_of::<*mut ()>()
        );
        assert_eq!(
            std::mem::size_of::<CUmodule>(),
            std::mem::size_of::<*mut ()>()
        );
        assert_eq!(
            std::mem::size_of::<CUfunction>(),
            std::mem::size_of::<*mut ()>()
        );
        assert_eq!(
            std::mem::size_of::<CUstream>(),
            std::mem::size_of::<*mut ()>()
        );
    }

    #[test]
    fn test_null_pointers() {
        use std::ptr;
        // Null pointers are valid for CUDA types
        let ctx: CUcontext = ptr::null_mut();
        let module: CUmodule = ptr::null_mut();
        let func: CUfunction = ptr::null_mut();
        let stream: CUstream = ptr::null_mut();

        assert!(ctx.is_null());
        assert!(module.is_null());
        assert!(func.is_null());
        assert!(stream.is_null());
    }
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// cuda_error_string never panics for any i32
        #[test]
        fn prop_error_string_never_panics(code in any::<i32>()) {
            let _ = cuda_error_string(code);
        }

        /// cuda_error_string returns valid string for all inputs
        #[test]
        fn prop_error_string_valid(code in any::<i32>()) {
            let result = cuda_error_string(code);
            prop_assert!(!result.is_empty());
            prop_assert!(result.starts_with("CUDA_"));
        }

        /// Known error codes return their specific string
        #[test]
        fn prop_known_errors_have_specific_string(
            code in prop_oneof![
                Just(CUDA_SUCCESS),
                Just(CUDA_ERROR_INVALID_VALUE),
                Just(CUDA_ERROR_OUT_OF_MEMORY),
                Just(CUDA_ERROR_NO_DEVICE),
                Just(CUDA_ERROR_INVALID_PTX),
            ]
        ) {
            let result = cuda_error_string(code);
            prop_assert_ne!(result, "CUDA_ERROR_UNKNOWN");
        }
    }
}
