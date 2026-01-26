//! Falsification Tests for Batched Multi-Head Attention (WAPR-PERF-008)
//!
//! Karl Popper's Mandate: Test each stage of the batched attention pipeline.
//!
//! ## The "Catley" Bug
//!
//! The batched attention pipeline produces garbage output, causing transcription
//! to hallucinate "[Catley]" instead of "The birds can use".
//!
//! ## Pipeline Stages to Test
//!
//! 1. InterleavedToBatched: [seq_len, d_model] -> [n_heads, seq_len, head_dim]
//! 2. BatchedTranspose: K from [n_heads, seq_len, head_dim] -> [n_heads, head_dim, seq_len]
//! 3. BatchedGemm: Q @ K^T -> scores [n_heads, seq_len, seq_len]
//! 4. BatchedScale: scores / sqrt(head_dim)
//! 5. BatchedSoftmax: attention weights (VERIFIED WORKING)
//! 6. BatchedGemm: attn @ V -> output
//! 7. BatchedToInterleaved: [n_heads, seq_len, head_dim] -> [seq_len, d_model]

#![allow(unused_imports)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
#[cfg(feature = "cuda")]
use trueno_gpu::kernels::{
    BatchedGemmKernel, BatchedToInterleavedKernel, BatchedTransposeKernel,
    InterleavedToBatchedKernel, Kernel,
};

// ============================================================================
// CPU Reference Implementations
// ============================================================================

/// CPU reference: interleaved to batched layout conversion
fn cpu_interleaved_to_batched(
    input: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let d_model = n_heads * head_dim;
    let mut output = vec![0.0f32; seq_len * d_model];

    for s in 0..seq_len {
        for h in 0..n_heads {
            for d in 0..head_dim {
                let in_idx = s * d_model + h * head_dim + d;
                let out_idx = h * seq_len * head_dim + s * head_dim + d;
                output[out_idx] = input[in_idx];
            }
        }
    }
    output
}

/// CPU reference: batched transpose
fn cpu_batched_transpose(input: &[f32], batch: usize, rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * rows * cols];

    for b in 0..batch {
        for r in 0..rows {
            for c in 0..cols {
                let in_idx = b * rows * cols + r * cols + c;
                let out_idx = b * rows * cols + c * rows + r;
                output[out_idx] = input[in_idx];
            }
        }
    }
    output
}

/// CPU reference: batched GEMM
/// A: [batch, m, k], B: [batch, k, n] -> C: [batch, m, n]
fn cpu_batched_gemm(a: &[f32], b: &[f32], batch: usize, m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; batch * m * n];

    for ba in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    let a_idx = ba * m * k + i * k + kk;
                    let b_idx = ba * k * n + kk * n + j;
                    sum += a[a_idx] * b[b_idx];
                }
                let c_idx = ba * m * n + i * n + j;
                c[c_idx] = sum;
            }
        }
    }
    c
}

/// CPU reference: batched to interleaved layout conversion
fn cpu_batched_to_interleaved(
    input: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let d_model = n_heads * head_dim;
    let mut output = vec![0.0f32; seq_len * d_model];

    for h in 0..n_heads {
        for s in 0..seq_len {
            for d in 0..head_dim {
                let in_idx = h * seq_len * head_dim + s * head_dim + d;
                let out_idx = s * d_model + h * head_dim + d;
                output[out_idx] = input[in_idx];
            }
        }
    }
    output
}

// ============================================================================
// Tests
// ============================================================================

/// Test 1: InterleavedToBatched layout conversion
#[test]
#[cfg(feature = "cuda")]
fn test_interleaved_to_batched_small() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let seq_len = 4u32;
    let n_heads = 2u32;
    let head_dim = 3u32;
    let d_model = n_heads * head_dim;
    let total = seq_len * d_model;

    // Create sequential input for easy verification
    let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let expected = cpu_interleaved_to_batched(
        &input,
        seq_len as usize,
        n_heads as usize,
        head_dim as usize,
    );

    let input_buf = GpuBuffer::from_host(&ctx, &input).expect("Upload failed");
    let output_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total as usize).expect("Alloc failed");

    let kernel = InterleavedToBatchedKernel::new(seq_len, n_heads, head_dim);
    let ptx = kernel.emit_ptx();
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile failed");
    let stream = CudaStream::new(&ctx).expect("Stream failed");

    let threads = 256u32;
    let blocks = (total + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input_buf.as_ptr();
    let output_ptr = output_buf.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .expect("Launch failed");
    }
    stream.synchronize().expect("Sync failed");

    let mut output = vec![0.0f32; total as usize];
    output_buf
        .copy_to_host(&mut output)
        .expect("Download failed");

    for i in 0..total as usize {
        assert!(
            (output[i] - expected[i]).abs() < 1e-6,
            "InterleavedToBatched [{}]: GPU={} vs CPU={}",
            i,
            output[i],
            expected[i]
        );
    }

    eprintln!("✓ InterleavedToBatched small test PASSED");
}

/// Test 2: BatchedTranspose
#[test]
#[cfg(feature = "cuda")]
fn test_batched_transpose_small() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let batch = 2u32;
    let rows = 3u32;
    let cols = 4u32;
    let total = batch * rows * cols;

    // Create sequential input
    let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let expected = cpu_batched_transpose(&input, batch as usize, rows as usize, cols as usize);

    let input_buf = GpuBuffer::from_host(&ctx, &input).expect("Upload failed");
    let output_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total as usize).expect("Alloc failed");

    let kernel = BatchedTransposeKernel::new(batch, rows, cols);
    let ptx = kernel.emit_ptx();
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile failed");
    let stream = CudaStream::new(&ctx).expect("Stream failed");

    let threads = 256u32;
    let elems_per_batch = rows * cols;
    let blocks_x = (elems_per_batch + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks_x, 1, batch),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input_buf.as_ptr();
    let output_ptr = output_buf.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(batch) as *mut _,
        std::ptr::addr_of!(rows) as *mut _,
        std::ptr::addr_of!(cols) as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .expect("Launch failed");
    }
    stream.synchronize().expect("Sync failed");

    let mut output = vec![0.0f32; total as usize];
    output_buf
        .copy_to_host(&mut output)
        .expect("Download failed");

    for i in 0..total as usize {
        assert!(
            (output[i] - expected[i]).abs() < 1e-6,
            "BatchedTranspose [{}]: GPU={} vs CPU={}",
            i,
            output[i],
            expected[i]
        );
    }

    eprintln!("✓ BatchedTranspose small test PASSED");
}

/// Test 3: BatchedGemm
#[test]
#[cfg(feature = "cuda")]
fn test_batched_gemm_small() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let batch = 2u32;
    let m = 3u32;
    let n = 4u32;
    let k = 2u32;

    // Create input matrices
    let a: Vec<f32> = (0..batch * m * k).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..batch * k * n).map(|i| i as f32 * 0.1).collect();
    let expected = cpu_batched_gemm(&a, &b, batch as usize, m as usize, n as usize, k as usize);

    let a_buf = GpuBuffer::from_host(&ctx, &a).expect("Upload A failed");
    let b_buf = GpuBuffer::from_host(&ctx, &b).expect("Upload B failed");
    let c_buf: GpuBuffer<f32> =
        GpuBuffer::new(&ctx, (batch * m * n) as usize).expect("Alloc C failed");

    let kernel = BatchedGemmKernel::naive(batch, m, n, k);
    let ptx = kernel.emit_ptx();
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile failed");
    let stream = CudaStream::new(&ctx).expect("Stream failed");

    let threads_per_block = 16u32;
    let blocks_x = (n + threads_per_block - 1) / threads_per_block;
    let blocks_y = (m + threads_per_block - 1) / threads_per_block;
    let config = LaunchConfig {
        grid: (blocks_x, blocks_y, batch),
        block: (threads_per_block, threads_per_block, 1),
        shared_mem: 0,
    };

    let a_ptr = a_buf.as_ptr();
    let b_ptr = b_buf.as_ptr();
    let c_ptr = c_buf.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(a_ptr) as *mut _,
        std::ptr::addr_of!(b_ptr) as *mut _,
        std::ptr::addr_of!(c_ptr) as *mut _,
        std::ptr::addr_of!(batch) as *mut _,
        std::ptr::addr_of!(m) as *mut _,
        std::ptr::addr_of!(n) as *mut _,
        std::ptr::addr_of!(k) as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .expect("Launch failed");
    }
    stream.synchronize().expect("Sync failed");

    let mut output = vec![0.0f32; (batch * m * n) as usize];
    c_buf.copy_to_host(&mut output).expect("Download failed");

    for i in 0..(batch * m * n) as usize {
        let delta = (output[i] - expected[i]).abs();
        assert!(
            delta < 1e-4,
            "BatchedGemm [{}]: GPU={} vs CPU={}, delta={}",
            i,
            output[i],
            expected[i],
            delta
        );
    }

    eprintln!("✓ BatchedGemm small test PASSED");
}

/// Test 4: BatchedToInterleaved layout conversion
#[test]
#[cfg(feature = "cuda")]
fn test_batched_to_interleaved_small() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let seq_len = 4u32;
    let n_heads = 2u32;
    let head_dim = 3u32;
    let d_model = n_heads * head_dim;
    let total = seq_len * d_model;

    // Create sequential input in batched layout
    let input: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let expected = cpu_batched_to_interleaved(
        &input,
        seq_len as usize,
        n_heads as usize,
        head_dim as usize,
    );

    let input_buf = GpuBuffer::from_host(&ctx, &input).expect("Upload failed");
    let output_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total as usize).expect("Alloc failed");

    let kernel = BatchedToInterleavedKernel::new(seq_len, n_heads, head_dim);
    let ptx = kernel.emit_ptx();
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile failed");
    let stream = CudaStream::new(&ctx).expect("Stream failed");

    let threads = 256u32;
    let blocks = (total + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input_buf.as_ptr();
    let output_ptr = output_buf.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .expect("Launch failed");
    }
    stream.synchronize().expect("Sync failed");

    let mut output = vec![0.0f32; total as usize];
    output_buf
        .copy_to_host(&mut output)
        .expect("Download failed");

    for i in 0..total as usize {
        assert!(
            (output[i] - expected[i]).abs() < 1e-6,
            "BatchedToInterleaved [{}]: GPU={} vs CPU={}",
            i,
            output[i],
            expected[i]
        );
    }

    eprintln!("✓ BatchedToInterleaved small test PASSED");
}

/// Test 5: Round-trip InterleavedToBatched -> BatchedToInterleaved should be identity
#[test]
#[cfg(feature = "cuda")]
fn test_layout_roundtrip() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let seq_len = 4u32;
    let n_heads = 2u32;
    let head_dim = 3u32;
    let d_model = n_heads * head_dim;
    let total = seq_len * d_model;

    // Original interleaved data
    let original: Vec<f32> = (0..total).map(|i| i as f32).collect();

    let orig_buf = GpuBuffer::from_host(&ctx, &original).expect("Upload failed");
    let batched_buf: GpuBuffer<f32> =
        GpuBuffer::new(&ctx, total as usize).expect("Alloc batched failed");
    let result_buf: GpuBuffer<f32> =
        GpuBuffer::new(&ctx, total as usize).expect("Alloc result failed");

    let stream = CudaStream::new(&ctx).expect("Stream failed");

    // Step 1: Interleaved -> Batched
    {
        let kernel = InterleavedToBatchedKernel::new(seq_len, n_heads, head_dim);
        let ptx = kernel.emit_ptx();
        let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile i2b failed");

        let threads = 256u32;
        let blocks = (total + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        let input_ptr = orig_buf.as_ptr();
        let output_ptr = batched_buf.as_ptr();

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
        ];

        unsafe {
            stream
                .launch_kernel(&mut module, kernel.name(), &config, &mut args)
                .expect("Launch i2b failed");
        }
    }

    // Step 2: Batched -> Interleaved
    {
        let kernel = BatchedToInterleavedKernel::new(seq_len, n_heads, head_dim);
        let ptx = kernel.emit_ptx();
        let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile b2i failed");

        let threads = 256u32;
        let blocks = (total + threads - 1) / threads;
        let config = LaunchConfig {
            grid: (blocks, 1, 1),
            block: (threads, 1, 1),
            shared_mem: 0,
        };

        let input_ptr = batched_buf.as_ptr();
        let output_ptr = result_buf.as_ptr();

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            std::ptr::addr_of!(input_ptr) as *mut _,
            std::ptr::addr_of!(output_ptr) as *mut _,
        ];

        unsafe {
            stream
                .launch_kernel(&mut module, kernel.name(), &config, &mut args)
                .expect("Launch b2i failed");
        }
    }

    stream.synchronize().expect("Sync failed");

    let mut result = vec![0.0f32; total as usize];
    result_buf
        .copy_to_host(&mut result)
        .expect("Download failed");

    for i in 0..total as usize {
        assert!(
            (result[i] - original[i]).abs() < 1e-6,
            "Roundtrip [{}]: result={} vs original={}",
            i,
            result[i],
            original[i]
        );
    }

    eprintln!("✓ Layout roundtrip test PASSED");
}

/// Test 6: Full attention pipeline on small input
#[test]
#[cfg(feature = "cuda")]
fn test_full_attention_pipeline_small() {
    // This test will help identify which stage is broken
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let seq_len = 4usize;
    let n_heads = 2usize;
    let head_dim = 3usize;
    let d_model = n_heads * head_dim;

    // Create simple Q, K, V inputs
    let q: Vec<f32> = (0..seq_len * d_model).map(|i| (i as f32) * 0.1).collect();
    let k: Vec<f32> = (0..seq_len * d_model)
        .map(|i| (i as f32) * 0.1 + 0.5)
        .collect();
    let v: Vec<f32> = (0..seq_len * d_model)
        .map(|i| (i as f32) * 0.1 - 0.3)
        .collect();

    // CPU reference for full attention
    let q_batched = cpu_interleaved_to_batched(&q, seq_len, n_heads, head_dim);
    let k_batched = cpu_interleaved_to_batched(&k, seq_len, n_heads, head_dim);
    let v_batched = cpu_interleaved_to_batched(&v, seq_len, n_heads, head_dim);

    eprintln!(
        "CPU Q batched (head 0, first row): {:?}",
        &q_batched[..head_dim]
    );
    eprintln!(
        "CPU K batched (head 0, first row): {:?}",
        &k_batched[..head_dim]
    );

    let k_transposed = cpu_batched_transpose(&k_batched, n_heads, seq_len, head_dim);
    eprintln!(
        "CPU K transposed (head 0, first row): {:?}",
        &k_transposed[..seq_len]
    );

    // Q @ K^T -> scores
    // After transpose, K is [n_heads, head_dim, seq_len]
    // Q is [n_heads, seq_len, head_dim]
    // scores = Q @ K^T = [n_heads, seq_len, head_dim] @ [n_heads, head_dim, seq_len] = [n_heads, seq_len, seq_len]
    let scores = cpu_batched_gemm(
        &q_batched,
        &k_transposed,
        n_heads,
        seq_len,
        seq_len,
        head_dim,
    );
    eprintln!("CPU scores (head 0, first row): {:?}", &scores[..seq_len]);

    eprintln!("✓ Full attention pipeline reference computed");
}

/// Test 7: InterleavedToBatched at Whisper scale (seq_len=1500, n_heads=6, head_dim=64)
#[test]
#[cfg(feature = "cuda")]
fn test_interleaved_to_batched_whisper_scale() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let seq_len = 1500u32;
    let n_heads = 6u32;
    let head_dim = 64u32;
    let d_model = n_heads * head_dim;
    let total = seq_len * d_model;

    // Create input with pattern we can verify
    let input: Vec<f32> = (0..total).map(|i| (i % 1000) as f32 * 0.001).collect();
    let expected = cpu_interleaved_to_batched(
        &input,
        seq_len as usize,
        n_heads as usize,
        head_dim as usize,
    );

    let input_buf = GpuBuffer::from_host(&ctx, &input).expect("Upload failed");
    let output_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total as usize).expect("Alloc failed");

    let kernel = InterleavedToBatchedKernel::new(seq_len, n_heads, head_dim);
    let ptx = kernel.emit_ptx();
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile failed");
    let stream = CudaStream::new(&ctx).expect("Stream failed");

    let threads = 256u32;
    let blocks = (total + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input_buf.as_ptr();
    let output_ptr = output_buf.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .expect("Launch failed");
    }
    stream.synchronize().expect("Sync failed");

    let mut output = vec![0.0f32; total as usize];
    output_buf
        .copy_to_host(&mut output)
        .expect("Download failed");

    // Check samples throughout the output
    let mut mismatches = 0;
    let check_indices = [0, 1000, 10000, 100000, 500000, total as usize - 1];
    for &i in &check_indices {
        if i < total as usize {
            let delta = (output[i] - expected[i]).abs();
            if delta > 1e-5 {
                eprintln!(
                    "Mismatch at {}: GPU={} vs CPU={}",
                    i, output[i], expected[i]
                );
                mismatches += 1;
            }
        }
    }

    assert_eq!(
        mismatches, 0,
        "InterleavedToBatched at Whisper scale has {} mismatches",
        mismatches
    );
    eprintln!("✓ InterleavedToBatched Whisper scale test PASSED");
}

/// Test 8: BatchedTranspose at Whisper scale
#[test]
#[cfg(feature = "cuda")]
fn test_batched_transpose_whisper_scale() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    // K matrix: [n_heads, seq_len, head_dim] -> [n_heads, head_dim, seq_len]
    let batch = 6u32;
    let rows = 1500u32; // seq_len
    let cols = 64u32; // head_dim
    let total = batch * rows * cols;

    let input: Vec<f32> = (0..total).map(|i| (i % 1000) as f32 * 0.001).collect();
    let expected = cpu_batched_transpose(&input, batch as usize, rows as usize, cols as usize);

    let input_buf = GpuBuffer::from_host(&ctx, &input).expect("Upload failed");
    let output_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total as usize).expect("Alloc failed");

    let kernel = BatchedTransposeKernel::new(batch, rows, cols);
    let ptx = kernel.emit_ptx();
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile failed");
    let stream = CudaStream::new(&ctx).expect("Stream failed");

    let threads = 256u32;
    let elems_per_batch = rows * cols;
    let blocks_x = (elems_per_batch + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks_x, 1, batch),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let input_ptr = input_buf.as_ptr();
    let output_ptr = output_buf.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(batch) as *mut _,
        std::ptr::addr_of!(rows) as *mut _,
        std::ptr::addr_of!(cols) as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .expect("Launch failed");
    }
    stream.synchronize().expect("Sync failed");

    let mut output = vec![0.0f32; total as usize];
    output_buf
        .copy_to_host(&mut output)
        .expect("Download failed");

    // Check samples
    let mut mismatches = 0;
    let check_indices = [0, 1000, 10000, 100000, 500000, total as usize - 1];
    for &i in &check_indices {
        if i < total as usize {
            let delta = (output[i] - expected[i]).abs();
            if delta > 1e-5 {
                eprintln!(
                    "Mismatch at {}: GPU={} vs CPU={}",
                    i, output[i], expected[i]
                );
                mismatches += 1;
            }
        }
    }

    assert_eq!(
        mismatches, 0,
        "BatchedTranspose at Whisper scale has {} mismatches",
        mismatches
    );
    eprintln!("✓ BatchedTranspose Whisper scale test PASSED");
}

/// Test 9: BatchedGemm Q @ K^T at Whisper scale
#[test]
#[cfg(feature = "cuda")]
fn test_batched_gemm_qkt_whisper_scale() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    // Q @ K^T: [6, 1500, 64] @ [6, 64, 1500] -> [6, 1500, 1500]
    let batch = 6u32;
    let m = 1500u32; // seq_len
    let n = 1500u32; // seq_len
    let k = 64u32; // head_dim

    // Create small-valued inputs to avoid overflow
    let a: Vec<f32> = (0..batch * m * k)
        .map(|i| (i % 100) as f32 * 0.01 - 0.5)
        .collect();
    let b: Vec<f32> = (0..batch * k * n)
        .map(|i| (i % 100) as f32 * 0.01 - 0.5)
        .collect();

    eprintln!("Computing CPU reference for GEMM [6, 1500, 64] @ [6, 64, 1500]...");
    let expected = cpu_batched_gemm(&a, &b, batch as usize, m as usize, n as usize, k as usize);
    eprintln!(
        "CPU reference computed. Sample values: {:?}",
        &expected[..5]
    );

    let a_buf = GpuBuffer::from_host(&ctx, &a).expect("Upload A failed");
    let b_buf = GpuBuffer::from_host(&ctx, &b).expect("Upload B failed");
    let c_buf: GpuBuffer<f32> =
        GpuBuffer::new(&ctx, (batch * m * n) as usize).expect("Alloc C failed");

    let kernel = BatchedGemmKernel::naive(batch, m, n, k);
    let ptx = kernel.emit_ptx();
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Compile failed");
    let stream = CudaStream::new(&ctx).expect("Stream failed");

    let threads_per_block = 16u32;
    let blocks_x = (n + threads_per_block - 1) / threads_per_block;
    let blocks_y = (m + threads_per_block - 1) / threads_per_block;
    let config = LaunchConfig {
        grid: (blocks_x, blocks_y, batch),
        block: (threads_per_block, threads_per_block, 1),
        shared_mem: 0,
    };

    let a_ptr = a_buf.as_ptr();
    let b_ptr = b_buf.as_ptr();
    let c_ptr = c_buf.as_ptr();

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(a_ptr) as *mut _,
        std::ptr::addr_of!(b_ptr) as *mut _,
        std::ptr::addr_of!(c_ptr) as *mut _,
        std::ptr::addr_of!(batch) as *mut _,
        std::ptr::addr_of!(m) as *mut _,
        std::ptr::addr_of!(n) as *mut _,
        std::ptr::addr_of!(k) as *mut _,
    ];

    unsafe {
        stream
            .launch_kernel(&mut module, kernel.name(), &config, &mut args)
            .expect("Launch failed");
    }
    stream.synchronize().expect("Sync failed");

    let mut output = vec![0.0f32; (batch * m * n) as usize];
    c_buf.copy_to_host(&mut output).expect("Download failed");

    eprintln!("GPU output sample: {:?}", &output[..5]);

    // Check samples
    let mut mismatches = 0;
    let total = (batch * m * n) as usize;
    let check_indices = [0, 1000, 10000, 100000, 1000000, total - 1];
    for &i in &check_indices {
        if i < total {
            let delta = (output[i] - expected[i]).abs();
            let rel_delta = if expected[i].abs() > 1e-6 {
                delta / expected[i].abs()
            } else {
                delta
            };
            if rel_delta > 0.01 && delta > 1e-4 {
                eprintln!(
                    "Mismatch at {}: GPU={} vs CPU={}, delta={}, rel={}",
                    i, output[i], expected[i], delta, rel_delta
                );
                mismatches += 1;
            }
        }
    }

    assert_eq!(
        mismatches, 0,
        "BatchedGemm Q@K^T at Whisper scale has {} mismatches",
        mismatches
    );
    eprintln!("✓ BatchedGemm Q@K^T Whisper scale test PASSED");
}

#[cfg(not(feature = "cuda"))]
#[test]
fn test_interleaved_to_batched_whisper_scale() {}
#[cfg(not(feature = "cuda"))]
#[test]
fn test_batched_transpose_whisper_scale() {}
#[cfg(not(feature = "cuda"))]
#[test]
fn test_batched_gemm_qkt_whisper_scale() {}

// Stub tests for non-CUDA builds
#[cfg(not(feature = "cuda"))]
#[test]
fn test_interleaved_to_batched_small() {}
#[cfg(not(feature = "cuda"))]
#[test]
fn test_batched_transpose_small() {}
#[cfg(not(feature = "cuda"))]
#[test]
fn test_batched_gemm_small() {}
#[cfg(not(feature = "cuda"))]
#[test]
fn test_batched_to_interleaved_small() {}
#[cfg(not(feature = "cuda"))]
#[test]
fn test_layout_roundtrip() {}
#[cfg(not(feature = "cuda"))]
#[test]
fn test_full_attention_pipeline_small() {}
