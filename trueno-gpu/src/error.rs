//! Error types for trueno-gpu operations
//!
//! Provides comprehensive error handling for PTX generation, CUDA driver operations,
//! and memory management.

use thiserror::Error;

/// Result type alias for trueno-gpu operations
pub type Result<T> = std::result::Result<T, GpuError>;

/// Errors that can occur during GPU operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum GpuError {
    /// PTX generation error
    #[error("PTX generation error: {0}")]
    PtxGeneration(String),

    /// Invalid PTX version
    #[error("Invalid PTX version: {major}.{minor} (requires >= 7.0)")]
    InvalidPtxVersion {
        /// Major version
        major: u32,
        /// Minor version
        minor: u32,
    },

    /// Invalid compute capability target
    #[error("Invalid compute capability: {0} (requires sm_70+)")]
    InvalidTarget(String),

    /// CUDA driver error
    #[error("CUDA driver error: {0} (code: {1})")]
    CudaDriver(String, i32),

    /// Memory allocation error
    #[error("GPU memory allocation failed: {0}")]
    MemoryAllocation(String),

    /// Kernel launch error
    #[error("Kernel launch failed: {0}")]
    KernelLaunch(String),

    /// Invalid kernel configuration
    #[error("Invalid launch config: {0}")]
    InvalidLaunchConfig(String),

    /// Register allocation error
    #[error("Register allocation failed: {0}")]
    RegisterAllocation(String),

    /// Bank conflict detected (debugging)
    #[error("Bank conflict detected in shared memory access")]
    BankConflict,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_generation_error() {
        let err = GpuError::PtxGeneration("invalid instruction".to_string());
        assert!(err.to_string().contains("PTX generation error"));
        assert!(err.to_string().contains("invalid instruction"));
    }

    #[test]
    fn test_invalid_ptx_version() {
        let err = GpuError::InvalidPtxVersion { major: 6, minor: 5 };
        assert!(err.to_string().contains("6.5"));
        assert!(err.to_string().contains("requires >= 7.0"));
    }

    #[test]
    fn test_invalid_target() {
        let err = GpuError::InvalidTarget("sm_50".to_string());
        assert!(err.to_string().contains("sm_50"));
        assert!(err.to_string().contains("requires sm_70+"));
    }

    #[test]
    fn test_cuda_driver_error() {
        let err = GpuError::CudaDriver("out of memory".to_string(), 2);
        assert!(err.to_string().contains("out of memory"));
        assert!(err.to_string().contains("code: 2"));
    }

    #[test]
    fn test_memory_allocation_error() {
        let err = GpuError::MemoryAllocation("insufficient device memory".to_string());
        assert!(err.to_string().contains("allocation failed"));
    }

    #[test]
    fn test_kernel_launch_error() {
        let err = GpuError::KernelLaunch("invalid grid dimensions".to_string());
        assert!(err.to_string().contains("launch failed"));
    }

    #[test]
    fn test_error_equality() {
        let err1 = GpuError::BankConflict;
        let err2 = GpuError::BankConflict;
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_error_clone() {
        let err = GpuError::PtxGeneration("test".to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }
}
