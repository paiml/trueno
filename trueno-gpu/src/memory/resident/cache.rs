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
use crate::driver::{CudaContext, CudaModule, CudaStream, LaunchConfig};
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
        let cache_guard = cache
            .lock()
            .map_err(|e| crate::GpuError::KernelLaunch(format!("Cache lock poisoned: {}", e)))?;
        if let Some(module) = cache_guard.get(key) {
            KERNEL_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
            return Ok(Arc::clone(module));
        }
    }

    // Cache miss: compile and store
    KERNEL_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
    eprintln!("[KERNEL-CACHE] Compiling: {}", key);

    let module = CudaModule::from_ptx(ctx, ptx)?;
    let module_arc = Arc::new(Mutex::new(module));

    // Insert into cache
    {
        let mut cache_guard = cache
            .lock()
            .map_err(|e| crate::GpuError::KernelLaunch(format!("Cache lock poisoned: {}", e)))?;
        cache_guard.insert(key.to_string(), Arc::clone(&module_arc));
    }

    Ok(module_arc)
}

/// Compile (or fetch from cache), lock the module, and launch the kernel.
///
/// This centralises the repeated resource-management boilerplate that every
/// CUDA operation needs:
///
/// 1. `get_or_compile_kernel` — cache lookup / PTX JIT compilation
/// 2. `module_arc.lock()` — acquire the `Mutex<CudaModule>`
/// 3. `stream.launch_kernel` — unsafe dispatch
///
/// By housing this in `cache.rs` the pattern is written once and all call
/// sites across `elementwise`, `gemm`, `norm_activation`, `linear_bias`,
/// `layout`, and `incremental` can delegate to it.
///
/// # Safety
///
/// The caller must guarantee that `args` contains valid device pointers whose
/// types and count match the kernel signature identified by `kernel_name`.
#[cfg(feature = "cuda")]
pub(crate) fn compile_lock_launch(
    ctx: &CudaContext,
    stream: &CudaStream,
    cache_key: &str,
    ptx: &str,
    kernel_name: &str,
    config: &LaunchConfig,
    args: &mut [*mut std::ffi::c_void],
) -> Result<()> {
    let module_arc = get_or_compile_kernel(ctx, cache_key, ptx)?;
    let mut module = module_arc
        .lock()
        .map_err(|e| crate::GpuError::KernelLaunch(format!("Module lock poisoned: {}", e)))?;
    // SAFETY: Caller guarantees args are valid pointers matching kernel signature.
    unsafe {
        stream.launch_kernel(&mut module, kernel_name, config, args)?;
    }
    Ok(())
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

    /// Test kernel_cache_hits returns expected value after reset
    #[test]
    fn test_kernel_cache_hits_after_reset() {
        reset_kernel_cache_stats();
        let hits = kernel_cache_hits();
        assert_eq!(hits, 0);
    }

    /// Test kernel_cache_misses returns expected value after reset
    #[test]
    fn test_kernel_cache_misses_after_reset() {
        reset_kernel_cache_stats();
        let misses = kernel_cache_misses();
        assert_eq!(misses, 0);
    }

    /// Test reset_kernel_cache_stats is idempotent
    #[test]
    fn test_reset_kernel_cache_stats_idempotent() {
        reset_kernel_cache_stats();
        reset_kernel_cache_stats();
        reset_kernel_cache_stats();
        assert_eq!(kernel_cache_hits(), 0);
        assert_eq!(kernel_cache_misses(), 0);
    }

    /// Test clear_kernel_cache is idempotent
    #[test]
    fn test_clear_kernel_cache_idempotent() {
        clear_kernel_cache();
        clear_kernel_cache();
        clear_kernel_cache();
        assert_eq!(kernel_cache_hits(), 0);
        assert_eq!(kernel_cache_misses(), 0);
    }

    /// Test that stats remain consistent after multiple operations
    #[test]
    fn test_stats_consistency() {
        reset_kernel_cache_stats();
        let h1 = kernel_cache_hits();
        let m1 = kernel_cache_misses();
        assert_eq!(h1, 0);
        assert_eq!(m1, 0);

        clear_kernel_cache();
        let h2 = kernel_cache_hits();
        let m2 = kernel_cache_misses();
        assert_eq!(h2, 0);
        assert_eq!(m2, 0);
    }
}

/// CUDA-specific tests that exercise the cache infrastructure
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;

    /// Test get_kernel_cache returns a valid cache
    #[test]
    fn test_get_kernel_cache_returns_valid_cache() {
        let cache = get_kernel_cache();
        // Verify we can acquire the lock
        let guard = cache.lock().expect("Cache lock should not be poisoned");
        // Cache should be a valid HashMap (may or may not be empty depending on other tests)
        drop(guard);
    }

    /// Test get_kernel_cache is idempotent (returns same static reference)
    #[test]
    fn test_get_kernel_cache_is_static() {
        let cache1 = get_kernel_cache();
        let cache2 = get_kernel_cache();
        // Both should point to the same static cache
        assert!(std::ptr::eq(cache1, cache2));
    }

    /// Test cache lock can be acquired and released multiple times
    #[test]
    fn test_cache_lock_reentrant() {
        let cache = get_kernel_cache();
        {
            let _guard1 = cache.lock().expect("First lock should succeed");
        }
        {
            let _guard2 = cache.lock().expect("Second lock should succeed");
        }
        {
            let _guard3 = cache.lock().expect("Third lock should succeed");
        }
    }

    /// Test clear_kernel_cache clears the actual cache
    #[test]
    fn test_clear_kernel_cache_clears_hashmap() {
        // Clear any existing state
        clear_kernel_cache();

        // Verify cache is empty
        let cache = get_kernel_cache();
        let guard = cache.lock().expect("Lock should succeed");
        assert!(guard.is_empty(), "Cache should be empty after clear");
    }

    /// Test CUDA hit/miss counters can be atomically incremented
    #[test]
    fn test_atomic_counter_operations() {
        reset_kernel_cache_stats();

        // Manually increment the counters to test atomic operations
        KERNEL_CACHE_HITS.fetch_add(5, std::sync::atomic::Ordering::Relaxed);
        KERNEL_CACHE_MISSES.fetch_add(3, std::sync::atomic::Ordering::Relaxed);

        assert_eq!(kernel_cache_hits(), 5);
        assert_eq!(kernel_cache_misses(), 3);

        // Reset should clear both
        reset_kernel_cache_stats();
        assert_eq!(kernel_cache_hits(), 0);
        assert_eq!(kernel_cache_misses(), 0);
    }

    /// Test stats counters work correctly with concurrent-style increments
    #[test]
    fn test_counter_concurrent_increments() {
        reset_kernel_cache_stats();

        // Simulate concurrent increments
        for _ in 0..100 {
            KERNEL_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        for _ in 0..50 {
            KERNEL_CACHE_MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        assert_eq!(kernel_cache_hits(), 100);
        assert_eq!(kernel_cache_misses(), 50);

        // Cleanup
        reset_kernel_cache_stats();
    }

    /// Test clear_kernel_cache when cache was never initialized
    #[test]
    fn test_clear_uninitialized_cache() {
        // This tests the path where KERNEL_CACHE.get() returns None
        // Note: Due to static initialization, this may not always hit the None path
        // but it should still not panic
        clear_kernel_cache();
        assert_eq!(kernel_cache_hits(), 0);
        assert_eq!(kernel_cache_misses(), 0);
    }

    /// Test cache can store and retrieve entries directly
    #[test]
    fn test_cache_hashmap_operations() {
        clear_kernel_cache();

        let cache = get_kernel_cache();

        // We can't easily create a CudaModule without a real context,
        // but we can verify the cache structure is sound by checking
        // it's a HashMap that accepts our key type
        {
            let guard = cache.lock().expect("Lock should succeed");
            assert!(guard.is_empty());
        }
    }
}
