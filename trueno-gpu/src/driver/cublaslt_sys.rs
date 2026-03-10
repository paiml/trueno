//! cuBLASLt Runtime API FFI Bindings
//!
//! Hand-written FFI for cuBLASLt FP8 tensor core GEMM operations.
//! cuBLASLt is required for FP8 E4M3 — cublasGemmEx does NOT support FP8.
//!
//! # Design Philosophy
//!
//! Same as cublas_sys.rs: own the stack, no external dependencies.
//!
//! # Contract
//!
//! PMAT-053: FP8 E4M3 GEMM for prefill (2x BW savings vs HGEMM FP16)

use std::ffi::c_void;
use std::os::raw::c_int;

use crate::GpuError;

// ============================================================================
// cuBLASLt Type Definitions
// ============================================================================

/// cuBLASLt handle (opaque pointer)
pub type CublasLtHandle = *mut c_void;

/// Matmul descriptor (opaque pointer)
pub type CublasLtMatmulDesc = *mut c_void;

/// Matrix layout descriptor (opaque pointer)
pub type CublasLtMatrixLayout = *mut c_void;

/// Matmul preference (opaque pointer)
pub type CublasLtMatmulPreference = *mut c_void;

/// cuBLASLt status (same codes as cuBLAS)
pub type CublasLtStatus = c_int;

pub const CUBLASLT_STATUS_SUCCESS: CublasLtStatus = 0;

// ============================================================================
// cuBLASLt Matmul Descriptor Attributes
// ============================================================================

pub type CublasLtMatmulDescAttribute = u32;

/// cublasOperation_t for A (attribute index 3 per cublasLt.h)
pub const CUBLASLT_MATMUL_DESC_TRANSA: CublasLtMatmulDescAttribute = 3;
/// cublasOperation_t for B (attribute index 4 per cublasLt.h)
pub const CUBLASLT_MATMUL_DESC_TRANSB: CublasLtMatmulDescAttribute = 4;

// ============================================================================
// cuBLASLt Matrix Layout Attributes
// ============================================================================

pub type CublasLtMatrixLayoutAttribute = u32;

/// Leading dimension
pub const CUBLASLT_MATRIX_LAYOUT_LD: CublasLtMatrixLayoutAttribute = 0;
/// Number of rows
pub const CUBLASLT_MATRIX_LAYOUT_ROWS: CublasLtMatrixLayoutAttribute = 1;
/// Number of columns
pub const CUBLASLT_MATRIX_LAYOUT_COLS: CublasLtMatrixLayoutAttribute = 2;

// ============================================================================
// cuBLASLt Matmul Preference Attributes
// ============================================================================

pub type CublasLtMatmulPreferenceAttribute = u32;

/// Maximum workspace size in bytes
pub const CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: CublasLtMatmulPreferenceAttribute = 1;

// ============================================================================
// cuBLASLt Compute Type (reuse from cublas_sys for compute descriptor)
// ============================================================================

use super::cublas_sys::{CublasComputeType, CublasOperation, CudaDataType};

// ============================================================================
// cuBLASLt Heuristic Result
// ============================================================================

/// cublasLtMatmulAlgo_t — algorithm descriptor (opaque, 64 bytes)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CublasLtMatmulAlgo {
    pub data: [u64; 8],
}

/// cublasLtMatmulHeuristicResult_t
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CublasLtMatmulHeuristicResult {
    pub algo: CublasLtMatmulAlgo,
    pub workspace_size: usize,
    pub state: CublasLtStatus,
    pub waves_count: f32,
    pub reserved: [c_int; 4],
}

// ============================================================================
// cuBLASLt Function Pointers (dynamically loaded)
// ============================================================================

#[allow(non_snake_case)]
pub struct CublasLtDriver {
    pub cublasLtCreate: unsafe extern "C" fn(handle: *mut CublasLtHandle) -> CublasLtStatus,

    pub cublasLtDestroy: unsafe extern "C" fn(handle: CublasLtHandle) -> CublasLtStatus,

    #[allow(clippy::type_complexity)]
    pub cublasLtMatmulDescCreate: unsafe extern "C" fn(
        desc: *mut CublasLtMatmulDesc,
        compute_type: CublasComputeType,
        scale_type: CudaDataType,
    ) -> CublasLtStatus,

    pub cublasLtMatmulDescDestroy: unsafe extern "C" fn(desc: CublasLtMatmulDesc) -> CublasLtStatus,

    #[allow(clippy::type_complexity)]
    pub cublasLtMatmulDescSetAttribute: unsafe extern "C" fn(
        desc: CublasLtMatmulDesc,
        attr: CublasLtMatmulDescAttribute,
        buf: *const c_void,
        size: usize,
    ) -> CublasLtStatus,

    #[allow(clippy::type_complexity)]
    pub cublasLtMatrixLayoutCreate: unsafe extern "C" fn(
        layout: *mut CublasLtMatrixLayout,
        data_type: CudaDataType,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> CublasLtStatus,

    pub cublasLtMatrixLayoutDestroy:
        unsafe extern "C" fn(layout: CublasLtMatrixLayout) -> CublasLtStatus,

    pub cublasLtMatmulPreferenceCreate:
        unsafe extern "C" fn(pref: *mut CublasLtMatmulPreference) -> CublasLtStatus,

    pub cublasLtMatmulPreferenceDestroy:
        unsafe extern "C" fn(pref: CublasLtMatmulPreference) -> CublasLtStatus,

    #[allow(clippy::type_complexity)]
    pub cublasLtMatmulPreferenceSetAttribute: unsafe extern "C" fn(
        pref: CublasLtMatmulPreference,
        attr: CublasLtMatmulPreferenceAttribute,
        buf: *const c_void,
        size: usize,
    ) -> CublasLtStatus,

    #[allow(clippy::type_complexity)]
    pub cublasLtMatmulAlgoGetHeuristic: unsafe extern "C" fn(
        handle: CublasLtHandle,
        desc: CublasLtMatmulDesc,
        a_layout: CublasLtMatrixLayout,
        b_layout: CublasLtMatrixLayout,
        c_layout: CublasLtMatrixLayout,
        d_layout: CublasLtMatrixLayout,
        pref: CublasLtMatmulPreference,
        requested_algo_count: c_int,
        results: *mut CublasLtMatmulHeuristicResult,
        returned_algo_count: *mut c_int,
    ) -> CublasLtStatus,

    #[allow(clippy::type_complexity)]
    pub cublasLtMatmul: unsafe extern "C" fn(
        handle: CublasLtHandle,
        desc: CublasLtMatmulDesc,
        alpha: *const c_void,
        a: *const c_void,
        a_layout: CublasLtMatrixLayout,
        b: *const c_void,
        b_layout: CublasLtMatrixLayout,
        beta: *const c_void,
        c: *const c_void,
        c_layout: CublasLtMatrixLayout,
        d: *mut c_void,
        d_layout: CublasLtMatrixLayout,
        algo: *const CublasLtMatmulAlgo,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> CublasLtStatus,
}

// ============================================================================
// Dynamic Loading
// ============================================================================

#[cfg(feature = "cuda")]
mod loading {
    use super::*;
    use libloading::{Library, Symbol};
    use std::sync::OnceLock;

    static CUBLASLT_DRIVER: OnceLock<Option<CublasLtDriver>> = OnceLock::new();
    static CUBLASLT_LIBRARY: OnceLock<Option<Library>> = OnceLock::new();

    impl CublasLtDriver {
        #[must_use]
        pub fn load() -> Option<&'static Self> {
            let _ = CUBLASLT_LIBRARY.get_or_init(|| {
                let lib_names = ["libcublasLt.so.12", "libcublasLt.so"];
                for name in lib_names {
                    if let Ok(lib) = unsafe { Library::new(name) } {
                        return Some(lib);
                    }
                }
                None
            });

            CUBLASLT_DRIVER
                .get_or_init(|| {
                    let lib = CUBLASLT_LIBRARY.get()?.as_ref()?;
                    Self::load_from_library(lib)
                })
                .as_ref()
        }

        fn load_from_library(lib: &Library) -> Option<Self> {
            unsafe {
                macro_rules! load_sym {
                    ($name:ident, $ty:ty) => {{
                        let sym: Symbol<'_, $ty> = lib.get(stringify!($name).as_bytes()).ok()?;
                        *sym
                    }};
                }

                type FnCreate = unsafe extern "C" fn(*mut CublasLtHandle) -> CublasLtStatus;
                type FnDestroy = unsafe extern "C" fn(CublasLtHandle) -> CublasLtStatus;
                type FnMatmulDescCreate = unsafe extern "C" fn(
                    *mut CublasLtMatmulDesc,
                    CublasComputeType,
                    CudaDataType,
                ) -> CublasLtStatus;
                type FnMatmulDescDestroy =
                    unsafe extern "C" fn(CublasLtMatmulDesc) -> CublasLtStatus;
                type FnMatmulDescSetAttribute = unsafe extern "C" fn(
                    CublasLtMatmulDesc,
                    CublasLtMatmulDescAttribute,
                    *const c_void,
                    usize,
                )
                    -> CublasLtStatus;
                type FnMatrixLayoutCreate = unsafe extern "C" fn(
                    *mut CublasLtMatrixLayout,
                    CudaDataType,
                    u64,
                    u64,
                    i64,
                ) -> CublasLtStatus;
                type FnMatrixLayoutDestroy =
                    unsafe extern "C" fn(CublasLtMatrixLayout) -> CublasLtStatus;
                type FnMatmulPrefCreate =
                    unsafe extern "C" fn(*mut CublasLtMatmulPreference) -> CublasLtStatus;
                type FnMatmulPrefDestroy =
                    unsafe extern "C" fn(CublasLtMatmulPreference) -> CublasLtStatus;
                type FnMatmulPrefSetAttribute = unsafe extern "C" fn(
                    CublasLtMatmulPreference,
                    CublasLtMatmulPreferenceAttribute,
                    *const c_void,
                    usize,
                )
                    -> CublasLtStatus;
                type FnAlgoGetHeuristic = unsafe extern "C" fn(
                    CublasLtHandle,
                    CublasLtMatmulDesc,
                    CublasLtMatrixLayout,
                    CublasLtMatrixLayout,
                    CublasLtMatrixLayout,
                    CublasLtMatrixLayout,
                    CublasLtMatmulPreference,
                    c_int,
                    *mut CublasLtMatmulHeuristicResult,
                    *mut c_int,
                ) -> CublasLtStatus;
                type FnMatmul = unsafe extern "C" fn(
                    CublasLtHandle,
                    CublasLtMatmulDesc,
                    *const c_void,
                    *const c_void,
                    CublasLtMatrixLayout,
                    *const c_void,
                    CublasLtMatrixLayout,
                    *const c_void,
                    *const c_void,
                    CublasLtMatrixLayout,
                    *mut c_void,
                    CublasLtMatrixLayout,
                    *const CublasLtMatmulAlgo,
                    *mut c_void,
                    usize,
                    *mut c_void,
                ) -> CublasLtStatus;

                Some(CublasLtDriver {
                    cublasLtCreate: load_sym!(cublasLtCreate, FnCreate),
                    cublasLtDestroy: load_sym!(cublasLtDestroy, FnDestroy),
                    cublasLtMatmulDescCreate: load_sym!(
                        cublasLtMatmulDescCreate,
                        FnMatmulDescCreate
                    ),
                    cublasLtMatmulDescDestroy: load_sym!(
                        cublasLtMatmulDescDestroy,
                        FnMatmulDescDestroy
                    ),
                    cublasLtMatmulDescSetAttribute: load_sym!(
                        cublasLtMatmulDescSetAttribute,
                        FnMatmulDescSetAttribute
                    ),
                    cublasLtMatrixLayoutCreate: load_sym!(
                        cublasLtMatrixLayoutCreate,
                        FnMatrixLayoutCreate
                    ),
                    cublasLtMatrixLayoutDestroy: load_sym!(
                        cublasLtMatrixLayoutDestroy,
                        FnMatrixLayoutDestroy
                    ),
                    cublasLtMatmulPreferenceCreate: load_sym!(
                        cublasLtMatmulPreferenceCreate,
                        FnMatmulPrefCreate
                    ),
                    cublasLtMatmulPreferenceDestroy: load_sym!(
                        cublasLtMatmulPreferenceDestroy,
                        FnMatmulPrefDestroy
                    ),
                    cublasLtMatmulPreferenceSetAttribute: load_sym!(
                        cublasLtMatmulPreferenceSetAttribute,
                        FnMatmulPrefSetAttribute
                    ),
                    cublasLtMatmulAlgoGetHeuristic: load_sym!(
                        cublasLtMatmulAlgoGetHeuristic,
                        FnAlgoGetHeuristic
                    ),
                    cublasLtMatmul: load_sym!(cublasLtMatmul, FnMatmul),
                })
            }
        }

        pub fn check(result: CublasLtStatus) -> Result<(), GpuError> {
            if result == CUBLASLT_STATUS_SUCCESS {
                Ok(())
            } else {
                Err(GpuError::CudaDriver(format!("cuBLASLt error (code {result})"), result))
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod loading {
    use super::*;

    impl CublasLtDriver {
        #[must_use]
        pub fn load() -> Option<&'static Self> {
            None
        }

        pub fn check(_result: CublasLtStatus) -> Result<(), GpuError> {
            Err(GpuError::CudaNotAvailable("cuda feature not enabled".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_result_size() {
        // Ensure struct layout matches CUDA headers
        assert!(std::mem::size_of::<CublasLtMatmulHeuristicResult>() >= 64 + 8 + 4 + 4 + 16);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn test_cublaslt_not_available_without_feature() {
        assert!(CublasLtDriver::load().is_none());
    }
}
