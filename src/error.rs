//! Error types for Trueno operations

use thiserror::Error;

use crate::Backend;

/// Result type for Trueno operations
pub type Result<T> = std::result::Result<T, TruenoError>;

/// Errors that can occur during Trueno operations
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TruenoError {
    /// Backend not supported on this platform
    #[error("Backend not supported on this platform: {0:?}")]
    UnsupportedBackend(Backend),

    /// Size mismatch between operands
    #[error("Size mismatch: expected {expected}, got {actual}")]
    SizeMismatch {
        /// Expected size
        expected: usize,
        /// Actual size
        actual: usize,
    },

    /// GPU error
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Division by zero (e.g., normalizing zero vector)
    #[error("Division by zero")]
    DivisionByZero,

    /// Empty vector (e.g., computing mean of empty vector)
    #[error("Empty vector")]
    EmptyVector,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsupported_backend_error() {
        let err = TruenoError::UnsupportedBackend(Backend::AVX512);
        assert_eq!(
            err.to_string(),
            "Backend not supported on this platform: AVX512"
        );
    }

    #[test]
    fn test_size_mismatch_error() {
        let err = TruenoError::SizeMismatch {
            expected: 10,
            actual: 5,
        };
        assert_eq!(err.to_string(), "Size mismatch: expected 10, got 5");
    }

    #[test]
    fn test_gpu_error() {
        let err = TruenoError::GpuError("Device not found".to_string());
        assert_eq!(err.to_string(), "GPU error: Device not found");
    }

    #[test]
    fn test_invalid_input_error() {
        let err = TruenoError::InvalidInput("Empty vector".to_string());
        assert_eq!(err.to_string(), "Invalid input: Empty vector");
    }

    #[test]
    fn test_error_equality() {
        let err1 = TruenoError::SizeMismatch {
            expected: 10,
            actual: 5,
        };
        let err2 = TruenoError::SizeMismatch {
            expected: 10,
            actual: 5,
        };
        assert_eq!(err1, err2);
    }

    #[test]
    fn test_division_by_zero_error() {
        let err = TruenoError::DivisionByZero;
        assert_eq!(err.to_string(), "Division by zero");
    }

    #[test]
    fn test_empty_vector_error() {
        let err = TruenoError::EmptyVector;
        assert_eq!(err.to_string(), "Empty vector");
    }
}
