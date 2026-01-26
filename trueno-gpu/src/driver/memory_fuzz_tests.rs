//! Memory Fuzz Tests (PMAT-018)
//!
//! Stress testing for GPU memory management.
//!
//! # Falsification Strategy
//! - **Scarcity**: Force OOM conditions
//! - **Degeneracy**: Zero-sized buffers, unaligned copies
//! - **Concurrency**: Stream overlap (simulated)

#![cfg(all(test, feature = "cuda"))]

use super::context::CudaContext;
use super::memory::GpuBuffer;
use crate::GpuError;
use proptest::prelude::*;

#[test]
fn test_zero_sized_buffer() {
    let ctx = CudaContext::new(0).expect("Context");

    // 0-sized allocation should either succeed (ptr=null/special) or fail gracefully
    // It should NOT panic or crash CUDA
    let buf_result = GpuBuffer::<f32>::new(&ctx, 0);

    // We accept either behavior, but it must be robust
    if let Ok(mut buf) = buf_result {
        assert_eq!(buf.len(), 0);
        // Copying 0 bytes should be a no-op
        let src: Vec<f32> = vec![];
        buf.copy_from_host(&src)
            .expect("Zero-byte copy should succeed");
    }
}

#[test]
fn test_unaligned_byte_copy() {
    let ctx = CudaContext::new(0).expect("Context");
    let len = 1024;
    let mut buf = GpuBuffer::<u8>::new(&ctx, len).expect("Alloc");

    let data: Vec<u8> = (0..len).map(|i| (i % 255) as u8).collect();
    buf.copy_from_host(&data).expect("Copy");

    let mut out = vec![0u8; len];
    buf.copy_to_host(&mut out).expect("Download");

    assert_eq!(data, out);
}

#[test]
fn test_oom_resilience() {
    let ctx = CudaContext::new(0).expect("Context");
    let (free_start, _) = ctx.memory_info().expect("Mem info");

    // Allocate 1GB chunks until failure
    let mut allocations = Vec::new();
    let chunk_size = 1024 * 1024 * 1024 / 4; // 1GB of f32 (256M elements)

    // RTX 4090 has 24GB. 30 chunks * 1GB = 30GB -> Must OOM.
    // Limit to 20 to avoid freezing system if driver is aggressive
    let mut hit_oom = false;

    for i in 0..30 {
        match GpuBuffer::<f32>::new(&ctx, chunk_size) {
            Ok(buf) => allocations.push(buf),
            Err(GpuError::OutOfMemory { .. }) => {
                hit_oom = true;
                println!("Hit OOM at chunk {}", i);
                break;
            }
            Err(GpuError::MemoryAllocation(msg)) if msg.contains("OUT_OF_MEMORY") => {
                // Also acceptable - OOM wrapped in MemoryAllocation
                hit_oom = true;
                println!("Hit OOM (MemoryAllocation) at chunk {}", i);
                break;
            }
            Err(e) => panic!("Unexpected error during OOM stress: {:?}", e),
        }
    }

    // If we didn't hit OOM, we either have >30GB RAM or something is wrong
    // But we don't assert(hit_oom) to avoid flaky fails on 80GB A100s if running elsewhere

    // Drop all allocations
    drop(allocations);

    // Verify memory is returned
    let (free_end, _) = ctx.memory_info().expect("Mem info");
    // Allow some small driver overhead variance, but major blocks should be free
    // Diff should be small
    let diff = if free_start > free_end {
        free_start - free_end
    } else {
        0
    };
    // 100MB tolerance
    assert!(
        diff < 100 * 1024 * 1024,
        "Memory leak detected! {} bytes missing",
        diff
    );
}

proptest! {
    #[test]
    fn test_buffer_roundtrip_fuzz(
        len in 1usize..100_000usize,
        val in any::<f32>()
    ) {
        // Setup context locally per test (expensive but safe for proptest)
        // Note: In real life, use a lazy_static context or run this test single-threaded
        // For now, we assume single-threaded execution via Makefile
        if let Ok(ctx) = CudaContext::new(0) {
             let mut buf = GpuBuffer::<f32>::new(&ctx, len).unwrap();

             let data = vec![val; len];
             buf.copy_from_host(&data).unwrap();

             let mut out = vec![0.0; len];
             buf.copy_to_host(&mut out).unwrap();

             // Check first, middle, last
             prop_assert_eq!(data[0], out[0]);
             prop_assert_eq!(data[len/2], out[len/2]);
             prop_assert_eq!(data[len-1], out[len-1]);
        }
    }
}

// =============================================================================
// ADVERSARIAL TESTS (Dr. Popper's Falsification Protocol)
// =============================================================================
// These tests try to BREAK the driver, not validate happy paths.

/// Falsification Test 1: Oversize Allocation
/// Attempt to allocate 100GB - must return OOM, not panic or hang
#[test]
fn test_alloc_oversize_100gb() {
    let ctx = CudaContext::new(0).expect("Context");

    // 100GB of f32 = 25 billion elements
    let oversize = 25_000_000_000usize;

    let result = GpuBuffer::<f32>::new(&ctx, oversize);

    match result {
        Err(GpuError::OutOfMemory { .. }) => {
            // Expected - driver correctly reported OOM
        }
        Err(GpuError::MemoryAllocation(_)) => {
            // Also acceptable - allocation failed
        }
        Err(e) => {
            // Any other error is acceptable as long as it doesn't panic
            println!("Oversize alloc returned: {:?}", e);
        }
        Ok(_) => {
            panic!("CRITICAL: 100GB allocation succeeded - this should be impossible on RTX 4090!");
        }
    }
}

/// Falsification Test 2: Copy from host with size mismatch (too small host)
#[test]
fn test_copy_from_host_too_small() {
    let ctx = CudaContext::new(0).expect("Context");
    let mut buf = GpuBuffer::<f32>::new(&ctx, 1000).expect("Alloc");

    // Try to copy from a smaller host buffer
    let small_data = vec![1.0f32; 500];
    let result = buf.copy_from_host(&small_data);

    assert!(
        result.is_err(),
        "copy_from_host should fail when host buffer is smaller"
    );
    if let Err(e) = result {
        assert!(
            format!("{:?}", e).contains("mismatch") || format!("{:?}", e).contains("Transfer"),
            "Error should mention size mismatch: {:?}",
            e
        );
    }
}

/// Falsification Test 3: Copy to host with size mismatch (too large host)
#[test]
fn test_copy_to_host_too_large() {
    let ctx = CudaContext::new(0).expect("Context");
    let buf = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc");

    // Try to copy to a larger host buffer
    let mut large_data = vec![0.0f32; 500];
    let result = buf.copy_to_host(&mut large_data);

    assert!(
        result.is_err(),
        "copy_to_host should fail when host buffer size doesn't match"
    );
}

/// Falsification Test 4: Partial copy out of bounds (offset too large)
#[test]
fn test_copy_from_host_at_out_of_bounds() {
    let ctx = CudaContext::new(0).expect("Context");
    let mut buf = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc");

    let data = vec![1.0f32; 50];

    // Offset 60 + len 50 = 110 > 100 buffer size
    let result = buf.copy_from_host_at(&data, 60);
    assert!(
        result.is_err(),
        "copy_from_host_at should fail when offset+len > buffer size"
    );
}

/// Falsification Test 5: Partial copy to host out of bounds
#[test]
fn test_copy_to_host_at_out_of_bounds() {
    let ctx = CudaContext::new(0).expect("Context");
    let data = vec![1.0f32; 100];
    let buf = GpuBuffer::from_host(&ctx, &data).expect("Alloc");

    let mut result = vec![0.0f32; 50];

    // Offset 60 + len 50 = 110 > 100 buffer size
    let copy_result = buf.copy_to_host_at(&mut result, 60);
    assert!(
        copy_result.is_err(),
        "copy_to_host_at should fail when offset+len > buffer size"
    );
}

/// Falsification Test 6: D2D copy size mismatch
#[test]
fn test_d2d_copy_size_mismatch() {
    let ctx = CudaContext::new(0).expect("Context");

    let src = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc src");
    let mut dst = GpuBuffer::<f32>::new(&ctx, 200).expect("Alloc dst");

    let result = dst.copy_from_buffer(&src);
    assert!(
        result.is_err(),
        "D2D copy should fail when buffer sizes don't match"
    );
}

/// Falsification Test 7: D2D partial copy out of bounds (dst)
#[test]
fn test_d2d_copy_at_dst_out_of_bounds() {
    let ctx = CudaContext::new(0).expect("Context");

    let src = GpuBuffer::<f32>::new(&ctx, 50).expect("Alloc src");
    let mut dst = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc dst");

    // dst_offset 60 + count 50 = 110 > dst.len 100
    let result = dst.copy_from_buffer_at(&src, 60, 0, 50);
    assert!(
        result.is_err(),
        "D2D copy_at should fail when dst_offset+count > dst.len"
    );
}

/// Falsification Test 8: D2D partial copy out of bounds (src)
#[test]
fn test_d2d_copy_at_src_out_of_bounds() {
    let ctx = CudaContext::new(0).expect("Context");

    let src = GpuBuffer::<f32>::new(&ctx, 50).expect("Alloc src");
    let mut dst = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc dst");

    // src_offset 30 + count 50 = 80 > src.len 50
    let result = dst.copy_from_buffer_at(&src, 0, 30, 50);
    assert!(
        result.is_err(),
        "D2D copy_at should fail when src_offset+count > src.len"
    );
}

/// Falsification Test 9: RAII cleanup verification
/// Allocate, drop, verify memory returns to the pool
#[test]
fn test_raii_cleanup_single_buffer() {
    let ctx = CudaContext::new(0).expect("Context");

    let (free_before, _) = ctx.memory_info().expect("Memory info");

    // Allocate 100MB
    let size = 25_000_000; // 100MB of f32
    {
        let _buf = GpuBuffer::<f32>::new(&ctx, size).expect("Alloc");
        let (free_during, _) = ctx.memory_info().expect("Memory info");

        // Memory should be allocated (less free memory)
        assert!(
            free_during < free_before,
            "Memory should decrease after allocation: before={}, during={}",
            free_before,
            free_during
        );
    }
    // Buffer dropped here

    let (free_after, _) = ctx.memory_info().expect("Memory info");

    // Memory should be returned (within 10MB tolerance for driver overhead)
    let tolerance = 10 * 1024 * 1024;
    assert!(
        free_after >= free_before - tolerance,
        "Memory leak detected! before={}, after={}, diff={}",
        free_before,
        free_after,
        free_before.saturating_sub(free_after)
    );
}

/// Falsification Test 10: Async copy size mismatch
#[test]
fn test_async_d2d_copy_size_mismatch() {
    use super::stream::CudaStream;

    let ctx = CudaContext::new(0).expect("Context");
    let stream = CudaStream::new(&ctx).expect("Stream");

    let src = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc src");
    let mut dst = GpuBuffer::<f32>::new(&ctx, 200).expect("Alloc dst");

    let result = unsafe { dst.copy_from_buffer_async(&src, &stream) };
    assert!(
        result.is_err(),
        "Async D2D copy should fail when buffer sizes don't match"
    );
}

/// Falsification Test 11: Async partial copy out of bounds
#[test]
fn test_async_d2d_copy_at_out_of_bounds() {
    use super::stream::CudaStream;

    let ctx = CudaContext::new(0).expect("Context");
    let stream = CudaStream::new(&ctx).expect("Stream");

    let src = GpuBuffer::<f32>::new(&ctx, 50).expect("Alloc src");
    let mut dst = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc dst");

    // dst out of bounds
    let result = unsafe { dst.copy_from_buffer_at_async(&src, 60, 0, 50, &stream) };
    assert!(
        result.is_err(),
        "Async D2D copy_at should fail when dst out of bounds"
    );

    // src out of bounds
    let result = unsafe { dst.copy_from_buffer_at_async(&src, 0, 30, 50, &stream) };
    assert!(
        result.is_err(),
        "Async D2D copy_at should fail when src out of bounds"
    );
}

/// Falsification Test 12: Async H2D copy size mismatch
#[test]
fn test_async_h2d_copy_size_mismatch() {
    use super::stream::CudaStream;

    let ctx = CudaContext::new(0).expect("Context");
    let stream = CudaStream::new(&ctx).expect("Stream");

    let mut buf = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc");
    let small_data = vec![1.0f32; 50];

    let result = unsafe { buf.copy_from_host_async(&small_data, &stream) };
    assert!(
        result.is_err(),
        "Async H2D copy should fail when host buffer size doesn't match"
    );
}

/// Falsification Test 13: Async D2H copy size mismatch
#[test]
fn test_async_d2h_copy_size_mismatch() {
    use super::stream::CudaStream;

    let ctx = CudaContext::new(0).expect("Context");
    let stream = CudaStream::new(&ctx).expect("Stream");

    let buf = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc");
    let mut large_data = vec![0.0f32; 200];

    let result = unsafe { buf.copy_to_host_async(&mut large_data, &stream) };
    assert!(
        result.is_err(),
        "Async D2H copy should fail when host buffer size doesn't match"
    );
}

/// Falsification Test 14: Empty buffer operations
#[test]
fn test_empty_buffer_operations() {
    let ctx = CudaContext::new(0).expect("Context");

    let mut empty_buf = GpuBuffer::<f32>::new(&ctx, 0).expect("Alloc empty");
    assert!(empty_buf.is_empty());
    assert_eq!(empty_buf.len(), 0);
    assert_eq!(empty_buf.size_bytes(), 0);

    // All these should succeed as no-ops
    let empty_data: Vec<f32> = vec![];
    empty_buf
        .copy_from_host(&empty_data)
        .expect("Empty H2D should succeed");

    let mut empty_out: Vec<f32> = vec![];
    empty_buf
        .copy_to_host(&mut empty_out)
        .expect("Empty D2H should succeed");

    // D2D with empty buffers
    let mut empty_dst = GpuBuffer::<f32>::new(&ctx, 0).expect("Alloc empty dst");
    empty_dst
        .copy_from_buffer(&empty_buf)
        .expect("Empty D2D should succeed");
}

/// Falsification Test 15: Partial copy with zero count
#[test]
fn test_partial_copy_zero_count() {
    let ctx = CudaContext::new(0).expect("Context");

    let src = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc src");
    let mut dst = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc dst");

    // Zero count should be a no-op, regardless of offsets
    dst.copy_from_buffer_at(&src, 0, 0, 0)
        .expect("Zero count D2D should succeed");
    dst.copy_from_buffer_at(&src, 50, 50, 0)
        .expect("Zero count D2D with offsets should succeed");
}

/// Falsification Test 16: Async raw copy bounds check
#[test]
fn test_async_raw_copy_bounds_check() {
    use super::stream::CudaStream;

    let ctx = CudaContext::new(0).expect("Context");
    let stream = CudaStream::new(&ctx).expect("Stream");

    let src = GpuBuffer::<f32>::new(&ctx, 50).expect("Alloc src");
    let mut dst = GpuBuffer::<f32>::new(&ctx, 100).expect("Alloc dst");

    // Use raw stream handle
    let stream_handle = stream.raw();

    // dst out of bounds
    let result = unsafe { dst.copy_from_buffer_at_async_raw(&src, 60, 0, 50, stream_handle) };
    assert!(
        result.is_err(),
        "Async raw D2D should fail when dst out of bounds"
    );

    // src out of bounds
    let result = unsafe { dst.copy_from_buffer_at_async_raw(&src, 0, 30, 50, stream_handle) };
    assert!(
        result.is_err(),
        "Async raw D2D should fail when src out of bounds"
    );

    // Zero count should succeed
    unsafe {
        dst.copy_from_buffer_at_async_raw(&src, 0, 0, 0, stream_handle)
            .expect("Zero count should succeed");
    }
}

/// Falsification Test 17: Buffer view properties
#[test]
fn test_buffer_view_properties() {
    let ctx = CudaContext::new(0).expect("Context");
    let buf = GpuBuffer::<f32>::new(&ctx, 256).expect("Alloc");

    let view = buf.clone_metadata();

    assert_eq!(view.as_ptr(), buf.as_ptr());
    assert_eq!(view.len(), buf.len());
    assert_eq!(view.is_empty(), buf.is_empty());
    assert_eq!(view.size_bytes(), buf.size_bytes());

    // View should NOT free memory when dropped (non-owning)
    drop(view);

    // Original buffer should still be valid
    assert_eq!(buf.len(), 256);
}

/// Falsification Test 18: Stress multiple allocations and drops
#[test]
fn test_stress_alloc_dealloc_cycle() {
    let ctx = CudaContext::new(0).expect("Context");
    let (free_start, _) = ctx.memory_info().expect("Memory info");

    // Allocate and drop 100 buffers
    for i in 0..100 {
        let size = (i + 1) * 1000; // 1K to 100K elements
        let _buf = GpuBuffer::<f32>::new(&ctx, size).expect("Alloc");
        // Buffer dropped at end of iteration
    }

    let (free_end, _) = ctx.memory_info().expect("Memory info");

    // Should be back to roughly same free memory
    // CUDA driver has internal fragmentation and caching, so allow 50MB tolerance
    let tolerance = 50 * 1024 * 1024; // 50MB tolerance for driver overhead
    assert!(
        free_end >= free_start - tolerance,
        "Memory leak after 100 alloc/dealloc cycles! start={}, end={}, leaked={}",
        free_start,
        free_end,
        free_start.saturating_sub(free_end)
    );
}
