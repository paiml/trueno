//! CUDA GPU Monitoring (TRUENO-SPEC-010)
//!
//! Provides native CUDA device monitoring via the CUDA Driver API.
//! This module enables accurate device information and real-time memory metrics.
//!
//! # Design Philosophy
//!
//! **Native CUDA**: Direct access via cuDeviceGetName, cuMemGetInfo provides
//! accurate information (e.g., "NVIDIA GeForce RTX 4090") compared to wgpu's
//! generic backend queries.
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno_gpu::monitor::{CudaDeviceInfo, CudaMemoryInfo};
//!
//! // Query device info
//! let info = CudaDeviceInfo::query(0)?;
//! println!("GPU: {} ({} GB)", info.name, info.total_memory_gb());
//!
//! // Query memory usage
//! let mem = CudaMemoryInfo::query()?;
//! println!("Free: {} / {} MB", mem.free_mb(), mem.total_mb());
//! ```
//!
//! # References
//!
//! - NVIDIA CUDA Driver API: cuDeviceGetName, cuDeviceTotalMem, cuMemGetInfo
//! - TRUENO-SPEC-010: GPU Monitoring, Tracing, and Visualization

#[cfg(feature = "cuda")]
use crate::driver::{cuda_available, device_count, CudaContext};
use crate::GpuError;

// ============================================================================
// CUDA Device Information (TRUENO-SPEC-010 Section 3.1)
// ============================================================================

/// CUDA device information from native driver API
///
/// Provides accurate device information including:
/// - Device name (e.g., "NVIDIA GeForce RTX 4090")
/// - Total VRAM in bytes
/// - Device ordinal
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    /// Device ordinal (0-based index)
    pub index: u32,
    /// Device name from cuDeviceGetName
    pub name: String,
    /// Total VRAM in bytes from cuDeviceTotalMem
    pub total_memory: u64,
}

impl CudaDeviceInfo {
    /// Query device information for the specified device index
    ///
    /// # Arguments
    ///
    /// * `device_index` - Device ordinal (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if device is not found or query fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let info = CudaDeviceInfo::query(0)?;
    /// println!("GPU: {}", info.name);
    /// ```
    #[cfg(feature = "cuda")]
    #[allow(clippy::cast_possible_wrap)]
    pub fn query(device_index: u32) -> Result<Self, GpuError> {
        let ctx = CudaContext::new(device_index as i32)?;
        let name = ctx.device_name()?;
        let total_memory = ctx.total_memory()? as u64;

        Ok(Self {
            index: device_index,
            name,
            total_memory,
        })
    }

    /// Query device information (non-CUDA stub)
    #[cfg(not(feature = "cuda"))]
    pub fn query(_device_index: u32) -> Result<Self, GpuError> {
        Err(GpuError::CudaNotAvailable(
            "cuda feature not enabled".to_string(),
        ))
    }

    /// Enumerate all available CUDA devices
    ///
    /// # Errors
    ///
    /// Returns error if enumeration fails.
    #[cfg(feature = "cuda")]
    pub fn enumerate() -> Result<Vec<Self>, GpuError> {
        let count = device_count()?;
        let mut devices = Vec::with_capacity(count);

        for i in 0..count {
            devices.push(Self::query(i as u32)?);
        }

        Ok(devices)
    }

    /// Enumerate devices (non-CUDA stub)
    #[cfg(not(feature = "cuda"))]
    pub fn enumerate() -> Result<Vec<Self>, GpuError> {
        Err(GpuError::CudaNotAvailable(
            "cuda feature not enabled".to_string(),
        ))
    }

    /// Get total memory in megabytes
    #[must_use]
    pub fn total_memory_mb(&self) -> u64 {
        self.total_memory / (1024 * 1024)
    }

    /// Get total memory in gigabytes
    #[must_use]
    pub fn total_memory_gb(&self) -> f64 {
        self.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

impl std::fmt::Display for CudaDeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} ({:.1} GB)",
            self.index,
            self.name,
            self.total_memory_gb()
        )
    }
}

// ============================================================================
// CUDA Memory Information (TRUENO-SPEC-010 Section 4.1.2)
// ============================================================================

/// Real-time CUDA memory information from cuMemGetInfo
///
/// Provides current memory usage on the active CUDA context.
#[derive(Debug, Clone, Copy)]
pub struct CudaMemoryInfo {
    /// Free memory in bytes
    pub free: u64,
    /// Total memory in bytes
    pub total: u64,
}

impl CudaMemoryInfo {
    /// Query current memory information
    ///
    /// Requires an active CUDA context.
    ///
    /// # Errors
    ///
    /// Returns error if no context is active or query fails.
    #[cfg(feature = "cuda")]
    pub fn query(ctx: &CudaContext) -> Result<Self, GpuError> {
        let (free, total) = ctx.memory_info()?;
        Ok(Self {
            free: free as u64,
            total: total as u64,
        })
    }

    /// Get used memory in bytes
    #[must_use]
    pub fn used(&self) -> u64 {
        self.total.saturating_sub(self.free)
    }

    /// Get free memory in megabytes
    #[must_use]
    pub fn free_mb(&self) -> u64 {
        self.free / (1024 * 1024)
    }

    /// Get total memory in megabytes
    #[must_use]
    pub fn total_mb(&self) -> u64 {
        self.total / (1024 * 1024)
    }

    /// Get used memory in megabytes
    #[must_use]
    pub fn used_mb(&self) -> u64 {
        self.used() / (1024 * 1024)
    }

    /// Get memory usage percentage (0.0 - 100.0)
    #[must_use]
    pub fn usage_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.used() as f64 / self.total as f64) * 100.0
        }
    }
}

impl std::fmt::Display for CudaMemoryInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} / {} MB ({:.1}% used)",
            self.used_mb(),
            self.total_mb(),
            self.usage_percent()
        )
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Check if CUDA monitoring is available
///
/// Returns `true` if CUDA driver is installed and at least one device exists.
#[must_use]
pub fn cuda_monitoring_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get the number of CUDA devices
///
/// # Errors
///
/// Returns error if CUDA is not available.
pub fn cuda_device_count() -> Result<usize, GpuError> {
    #[cfg(feature = "cuda")]
    {
        device_count()
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(GpuError::CudaNotAvailable(
            "cuda feature not enabled".to_string(),
        ))
    }
}

// ============================================================================
// Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // H₀-CUDA-MON-01: CudaDeviceInfo unit tests
    // =========================================================================

    #[test]
    fn h0_cuda_mon_01_device_info_display() {
        let info = CudaDeviceInfo {
            index: 0,
            name: "Test GPU".to_string(),
            total_memory: 24 * 1024 * 1024 * 1024, // 24 GB
        };

        let display = format!("{}", info);
        assert!(display.contains("Test GPU"));
        assert!(display.contains("24.0"));
    }

    #[test]
    fn h0_cuda_mon_02_device_info_memory_helpers() {
        let info = CudaDeviceInfo {
            index: 0,
            name: "Test".to_string(),
            total_memory: 24 * 1024 * 1024 * 1024, // 24 GB
        };

        assert_eq!(info.total_memory_mb(), 24 * 1024);
        assert!((info.total_memory_gb() - 24.0).abs() < 0.01);
    }

    // =========================================================================
    // H₀-CUDA-MON-10: CudaMemoryInfo unit tests
    // =========================================================================

    #[test]
    fn h0_cuda_mon_10_memory_info_used() {
        let mem = CudaMemoryInfo {
            free: 16 * 1024 * 1024 * 1024,  // 16 GB free
            total: 24 * 1024 * 1024 * 1024, // 24 GB total
        };

        assert_eq!(mem.used(), 8 * 1024 * 1024 * 1024); // 8 GB used
    }

    #[test]
    fn h0_cuda_mon_11_memory_info_mb_helpers() {
        let mem = CudaMemoryInfo {
            free: 16 * 1024 * 1024 * 1024,
            total: 24 * 1024 * 1024 * 1024,
        };

        assert_eq!(mem.free_mb(), 16 * 1024);
        assert_eq!(mem.total_mb(), 24 * 1024);
        assert_eq!(mem.used_mb(), 8 * 1024);
    }

    #[test]
    fn h0_cuda_mon_12_memory_info_usage_percent() {
        let mem = CudaMemoryInfo {
            free: 12 * 1024 * 1024 * 1024,  // 12 GB free
            total: 24 * 1024 * 1024 * 1024, // 24 GB total
        };

        // 50% used
        assert!((mem.usage_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn h0_cuda_mon_13_memory_info_usage_percent_zero_total() {
        let mem = CudaMemoryInfo { free: 0, total: 0 };

        assert!((mem.usage_percent() - 0.0).abs() < 0.01);
    }

    #[test]
    fn h0_cuda_mon_14_memory_info_display() {
        let mem = CudaMemoryInfo {
            free: 16 * 1024 * 1024 * 1024,
            total: 24 * 1024 * 1024 * 1024,
        };

        let display = format!("{}", mem);
        assert!(display.contains("8192")); // 8 GB used
        assert!(display.contains("24576")); // 24 GB total
        assert!(display.contains("33.3")); // ~33% used
    }

    // =========================================================================
    // H₀-CUDA-MON-20: Integration tests (require CUDA feature)
    // =========================================================================

    #[test]
    #[cfg(feature = "cuda")]
    fn h0_cuda_mon_20_query_device_info() {
        match CudaDeviceInfo::query(0) {
            Ok(info) => {
                assert!(!info.name.is_empty());
                assert!(info.total_memory > 0);
                println!("CUDA Device: {}", info);
            }
            Err(e) => {
                // No CUDA device is OK for CI
                println!("No CUDA device (expected in CI): {}", e);
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn h0_cuda_mon_21_enumerate_devices() {
        match CudaDeviceInfo::enumerate() {
            Ok(devices) => {
                for dev in &devices {
                    println!("Found: {}", dev);
                }
            }
            Err(e) => {
                println!("CUDA enumeration failed (expected in CI): {}", e);
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn h0_cuda_mon_22_query_memory_info() {
        use crate::driver::CudaContext;

        match CudaDeviceInfo::query(0) {
            Ok(_) => {
                // Context was created by query, but we need a fresh one for memory_info
                if let Ok(ctx) = CudaContext::new(0) {
                    match CudaMemoryInfo::query(&ctx) {
                        Ok(mem) => {
                            assert!(mem.total > 0);
                            assert!(mem.free <= mem.total);
                            println!("CUDA Memory: {}", mem);
                        }
                        Err(e) => {
                            println!("Memory query failed: {}", e);
                        }
                    }
                }
            }
            Err(e) => {
                println!("No CUDA device (expected in CI): {}", e);
            }
        }
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn h0_cuda_mon_30_no_cuda_feature() {
        // Without cuda feature, queries should return error
        assert!(CudaDeviceInfo::query(0).is_err());
        assert!(CudaDeviceInfo::enumerate().is_err());
    }

    // =========================================================================
    // H0-CUDA-MON-40: Convenience function tests
    // =========================================================================

    #[test]
    fn h0_cuda_mon_40_cuda_monitoring_available() {
        // Test the convenience function - it should return a boolean
        let available = cuda_monitoring_available();
        // Without the cuda feature, this should be false
        #[cfg(not(feature = "cuda"))]
        assert!(!available, "Without CUDA feature, monitoring should not be available");
        // With cuda feature but possibly no hardware, it might be true or false
        #[cfg(feature = "cuda")]
        {
            // Just verify it returns a boolean (no panic)
            let _ = available;
        }
    }

    #[test]
    fn h0_cuda_mon_41_cuda_device_count() {
        // Test the device count function
        let result = cuda_device_count();
        #[cfg(not(feature = "cuda"))]
        {
            assert!(result.is_err(), "Without CUDA feature, device count should error");
            if let Err(e) = result {
                let err_msg = format!("{:?}", e);
                assert!(
                    err_msg.contains("cuda") || err_msg.contains("Cuda"),
                    "Error should mention CUDA"
                );
            }
        }
        #[cfg(feature = "cuda")]
        {
            // With cuda feature, it should return Ok or Err based on hardware
            match result {
                Ok(count) => {
                    // Count should be a reasonable number
                    assert!(count <= 16, "Device count should be reasonable");
                }
                Err(_) => {
                    // No CUDA hardware is OK for CI
                }
            }
        }
    }

    // =========================================================================
    // H0-CUDA-MON-50: Derive trait tests (Clone, Debug, Copy)
    // =========================================================================

    #[test]
    fn h0_cuda_mon_50_device_info_clone() {
        let info = CudaDeviceInfo {
            index: 1,
            name: "Cloneable GPU".to_string(),
            total_memory: 8 * 1024 * 1024 * 1024,
        };

        let cloned = info.clone();
        assert_eq!(cloned.index, info.index);
        assert_eq!(cloned.name, info.name);
        assert_eq!(cloned.total_memory, info.total_memory);
    }

    #[test]
    fn h0_cuda_mon_51_device_info_debug() {
        let info = CudaDeviceInfo {
            index: 2,
            name: "Debug GPU".to_string(),
            total_memory: 16 * 1024 * 1024 * 1024,
        };

        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("CudaDeviceInfo"));
        assert!(debug_str.contains("Debug GPU"));
        // 16 GB = 17179869184 bytes - check for index
        assert!(debug_str.contains("2"), "Should contain the index value");
    }

    #[test]
    fn h0_cuda_mon_52_memory_info_clone() {
        let mem = CudaMemoryInfo {
            free: 4 * 1024 * 1024 * 1024,
            total: 8 * 1024 * 1024 * 1024,
        };

        let cloned = mem.clone();
        assert_eq!(cloned.free, mem.free);
        assert_eq!(cloned.total, mem.total);
    }

    #[test]
    fn h0_cuda_mon_53_memory_info_copy() {
        let mem = CudaMemoryInfo {
            free: 2 * 1024 * 1024 * 1024,
            total: 4 * 1024 * 1024 * 1024,
        };

        // Test Copy trait by assigning to another variable
        let copied = mem;
        assert_eq!(copied.free, 2 * 1024 * 1024 * 1024);
        assert_eq!(copied.total, 4 * 1024 * 1024 * 1024);

        // Original should still be usable (Copy semantics)
        assert_eq!(mem.free, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn h0_cuda_mon_54_memory_info_debug() {
        let mem = CudaMemoryInfo {
            free: 1024 * 1024 * 1024,
            total: 2 * 1024 * 1024 * 1024,
        };

        let debug_str = format!("{:?}", mem);
        assert!(debug_str.contains("CudaMemoryInfo"));
        assert!(debug_str.contains("free"));
        assert!(debug_str.contains("total"));
    }

    // =========================================================================
    // H0-CUDA-MON-60: Edge case tests
    // =========================================================================

    #[test]
    fn h0_cuda_mon_60_memory_info_used_saturating() {
        // Test saturating_sub edge case where free > total (shouldn't happen but be safe)
        let mem = CudaMemoryInfo {
            free: 10 * 1024 * 1024 * 1024,  // 10 GB free (impossible: more than total)
            total: 8 * 1024 * 1024 * 1024,  // 8 GB total
        };

        // saturating_sub should return 0, not underflow
        assert_eq!(mem.used(), 0);
        assert_eq!(mem.used_mb(), 0);
    }

    #[test]
    fn h0_cuda_mon_61_device_info_display_index_format() {
        // Test Display formatting with different index values
        let info = CudaDeviceInfo {
            index: 7,
            name: "GPU Seven".to_string(),
            total_memory: 12 * 1024 * 1024 * 1024,
        };

        let display = format!("{}", info);
        assert!(display.contains("[7]"), "Display should show index in brackets");
        assert!(display.contains("GPU Seven"));
        assert!(display.contains("12.0"));
    }

    #[test]
    fn h0_cuda_mon_62_memory_info_display_formatting() {
        // Test Display with specific values to verify format
        let mem = CudaMemoryInfo {
            free: 0, // 0 bytes free
            total: 1024 * 1024 * 1024, // 1 GB total
        };

        let display = format!("{}", mem);
        assert!(display.contains("1024")); // 1024 MB total
        assert!(display.contains("100")); // 100% used
    }

    #[test]
    fn h0_cuda_mon_63_small_memory_values() {
        // Test with small memory values (less than 1 MB)
        let mem = CudaMemoryInfo {
            free: 512 * 1024,   // 512 KB
            total: 1024 * 1024, // 1 MB
        };

        assert_eq!(mem.free_mb(), 0); // Less than 1 MB
        assert_eq!(mem.total_mb(), 1);
        assert_eq!(mem.used_mb(), 0); // 512 KB used
        assert!((mem.usage_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn h0_cuda_mon_64_device_info_empty_name() {
        // Test with empty device name
        let info = CudaDeviceInfo {
            index: 0,
            name: String::new(),
            total_memory: 1024,
        };

        let display = format!("{}", info);
        assert!(display.contains("[0]"));
        // Should handle empty name gracefully
    }

    #[test]
    fn h0_cuda_mon_65_device_info_zero_memory() {
        // Test with zero total memory
        let info = CudaDeviceInfo {
            index: 0,
            name: "Zero Memory GPU".to_string(),
            total_memory: 0,
        };

        assert_eq!(info.total_memory_mb(), 0);
        assert!((info.total_memory_gb() - 0.0).abs() < 0.001);

        let display = format!("{}", info);
        assert!(display.contains("0.0 GB"));
    }

    #[test]
    fn h0_cuda_mon_66_memory_info_full_usage() {
        // Test with 100% memory usage
        let mem = CudaMemoryInfo {
            free: 0,
            total: 16 * 1024 * 1024 * 1024,
        };

        assert_eq!(mem.used(), 16 * 1024 * 1024 * 1024);
        assert!((mem.usage_percent() - 100.0).abs() < 0.001);
    }

    #[test]
    fn h0_cuda_mon_67_memory_info_no_usage() {
        // Test with 0% memory usage (all free)
        let mem = CudaMemoryInfo {
            free: 8 * 1024 * 1024 * 1024,
            total: 8 * 1024 * 1024 * 1024,
        };

        assert_eq!(mem.used(), 0);
        assert!((mem.usage_percent() - 0.0).abs() < 0.001);
    }

    // =========================================================================
    // H0-CUDA-MON-70: Error message validation (non-CUDA feature)
    // =========================================================================

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn h0_cuda_mon_70_query_error_message() {
        let result = CudaDeviceInfo::query(0);
        assert!(result.is_err());

        if let Err(e) = result {
            let err_str = format!("{:?}", e);
            assert!(
                err_str.contains("cuda") || err_str.contains("Cuda") || err_str.contains("CUDA"),
                "Error should mention CUDA: got {}",
                err_str
            );
        }
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn h0_cuda_mon_71_enumerate_error_message() {
        let result = CudaDeviceInfo::enumerate();
        assert!(result.is_err());

        if let Err(e) = result {
            let err_str = format!("{:?}", e);
            assert!(
                err_str.contains("cuda") || err_str.contains("Cuda") || err_str.contains("CUDA"),
                "Error should mention CUDA: got {}",
                err_str
            );
        }
    }
}
