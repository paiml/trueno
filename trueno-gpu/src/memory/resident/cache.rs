//! Kernel Cache (WAPR-PERF-004)
//!
//! Global kernel cache to eliminate PTX recompilation overhead.

#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(feature = "cuda")]
use crate::driver::{CudaContext, CudaModule};
#[cfg(feature = "cuda")]
use crate::error::Result;

/// Global kernel cache to eliminate PTX recompilation overhead.
///
/// Each unique kernel configuration (name + parameters) is compiled once
/// and cached for reuse. This eliminates the 24x recompilation per inference
/// that was previously observed.
///
/// ## Keying Strategy
///
/// Keys are strings of format: `"{kernel_name}:{config}"` where config
/// encodes all parameters that affect the PTX output.
///
/// ## Thread Safety
///
/// The cache uses double-locking:
/// - Outer Mutex guards the HashMap
/// - Inner Arc<Mutex<CudaModule>> allows concurrent kernel launches
///
/// ## Example Keys
///
/// - `"softmax:32"` - SoftmaxKernel for row_size=32
/// - `"softmax_long_row:1500"` - LongRowSoftmaxKernel for row_size=1500
/// - `"residual_add:1024"` - ResidualAddKernel for n=1024
#[cfg(feature = "cuda")]
static KERNEL_CACHE: OnceLock<Mutex<HashMap<String, Arc<Mutex<CudaModule>>>>> = OnceLock::new();

/// Statistics for kernel cache performance
#[cfg(feature = "cuda")]
static KERNEL_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "cuda")]
static KERNEL_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);

/// Get the global kernel cache, initializing if needed
#[cfg(feature = "cuda")]
pub(crate) fn get_kernel_cache() -> &'static Mutex<HashMap<String, Arc<Mutex<CudaModule>>>> {
    KERNEL_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get a cached kernel module, compiling if not present.
///
/// # Arguments
///
/// * `ctx` - CUDA context for compilation
/// * `key` - Cache key (kernel_name:config)
/// * `ptx` - PTX source to compile on cache miss
///
/// # Returns
///
/// Arc to the cached module, wrapped in Mutex for mutable access.
#[cfg(feature = "cuda")]
pub(crate) fn get_or_compile_kernel(
    ctx: &CudaContext,
    key: &str,
    ptx: &str,
) -> Result<Arc<Mutex<CudaModule>>> {
    let cache = get_kernel_cache();

    // Fast path: check if already cached
    {
        let cache_guard = cache.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Cache lock poisoned: {}", e))
        })?;
        if let Some(module) = cache_guard.get(key) {
            KERNEL_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            return Ok(Arc::clone(module));
        }
    }

    // Slow path: compile and cache
    KERNEL_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
    eprintln!("[KERNEL-CACHE] Compiling: {}", key);

    let module = CudaModule::from_ptx(ctx, ptx)?;
    let module_arc = Arc::new(Mutex::new(module));

    // Insert into cache
    {
        let mut cache_guard = cache.lock().map_err(|e| {
            crate::GpuError::KernelLaunch(format!("Cache lock poisoned: {}", e))
        })?;
        cache_guard.insert(key.to_string(), Arc::clone(&module_arc));
    }

    Ok(module_arc)
}

/// Get kernel cache hit count
#[cfg(feature = "cuda")]
#[must_use]
pub fn kernel_cache_hits() -> u64 {
    KERNEL_CACHE_HITS.load(Ordering::Relaxed)
}

/// Get kernel cache miss count
#[cfg(feature = "cuda")]
#[must_use]
pub fn kernel_cache_misses() -> u64 {
    KERNEL_CACHE_MISSES.load(Ordering::Relaxed)
}

/// Reset kernel cache statistics
#[cfg(feature = "cuda")]
pub fn reset_kernel_cache_stats() {
    KERNEL_CACHE_HITS.store(0, Ordering::Relaxed);
    KERNEL_CACHE_MISSES.store(0, Ordering::Relaxed);
}

/// Clear the kernel cache (useful for testing)
#[cfg(feature = "cuda")]
pub fn clear_kernel_cache() {
    if let Some(cache) = KERNEL_CACHE.get() {
        if let Ok(mut guard) = cache.lock() {
            guard.clear();
        }
    }
    reset_kernel_cache_stats();
}

// Non-CUDA stubs for compilation without cuda feature

/// Get kernel cache hit count (stub when CUDA disabled)
#[cfg(not(feature = "cuda"))]
#[must_use]
pub fn kernel_cache_hits() -> u64 {
    0
}

/// Get kernel cache miss count (stub when CUDA disabled)
#[cfg(not(feature = "cuda"))]
#[must_use]
pub fn kernel_cache_misses() -> u64 {
    0
}

/// Reset kernel cache statistics (stub when CUDA disabled)
#[cfg(not(feature = "cuda"))]
pub fn reset_kernel_cache_stats() {}

/// Clear the kernel cache (stub when CUDA disabled)
#[cfg(not(feature = "cuda"))]
pub fn clear_kernel_cache() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_cache_stats_initial() {
        reset_kernel_cache_stats();
        assert_eq!(kernel_cache_hits(), 0);
        assert_eq!(kernel_cache_misses(), 0);
    }

    #[test]
    fn test_clear_kernel_cache() {
        // Just verify it doesn't panic
        clear_kernel_cache();
        assert_eq!(kernel_cache_hits(), 0);
        assert_eq!(kernel_cache_misses(), 0);
    }
}
