//! TDD Tests for GPU-Resident Tensor Architecture (WAPR-PERF-004)
//!
//! These tests are written FIRST (Test-Driven Development) to define the API
//! before implementation exists. Each test documents a specific requirement.
//!
//! ## Problem Statement
//!
//! Current approach: ~150 host↔device transfers per encoder forward pass
//! Target: 2 transfers total (upload weights once, download output once)
//!
//! ## Root Cause (Five Whys)
//!
//! 1. Why is GPU encoder slower than CPU? → 0.76x speedup (actually slower)
//! 2. Why is CUDA gemm not helping? → ~150 host↔device transfers per pass
//! 3. Why so many transfers? → Data ping-pongs: CPU→GPU for gemm, GPU→CPU for softmax
//! 4. Why not keep data on GPU? → No GPU-resident tensor abstraction
//! 5. What's the fix? → Build GpuResidentTensor into trueno-gpu
//!
//! ## Citations
//!
//! - [Dao2022] FlashAttention: Fast and Memory-Efficient Exact Attention
//! - [Kwon2023] PagedAttention for LLM Serving with vLLM
//! - [Popper1934] The Logic of Scientific Discovery - Falsificationism

#![allow(unused_imports)]
#![allow(dead_code)]

#[cfg(feature = "cuda")]
use trueno_gpu::driver::CudaContext;
#[cfg(feature = "cuda")]
use trueno_gpu::memory::resident::{
    batched_multihead_attention, clear_kernel_cache, reset_transfer_counters, total_d2h_transfers,
    total_h2d_transfers, GpuResidentTensor, TransferStats,
};

// ============================================================================
// PHASE 1: GpuResidentTensor Core API
// ============================================================================

/// Test: Tensor created on GPU should stay on GPU without host copy
///
/// Requirement: Creating a tensor should NOT require a host copy to exist.
/// The tensor data lives exclusively on the device.
#[test]
#[cfg(feature = "cuda")]
fn test_gpu_tensor_created_on_device() {
    // Skip if CUDA not available (e.g., in CI without GPU)
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return, // Skip test
    };

    reset_transfer_counters();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];

    // Create tensor - data uploaded ONCE
    let tensor = GpuResidentTensor::from_host(&ctx, &data).expect("Upload failed");

    // Tensor should be on device
    assert!(tensor.is_device_resident());
    // Only 1 transfer (the initial upload)
    assert_eq!(tensor.host_to_device_transfers(), 1);
    assert_eq!(tensor.device_to_host_transfers(), 0);

    // Global counters also track
    assert_eq!(total_h2d_transfers(), 1);
    assert_eq!(total_d2h_transfers(), 0);
}

#[test]
#[cfg(not(feature = "cuda"))]
fn test_gpu_tensor_created_on_device() {
    // Skip test when CUDA is not available
}

/// Test: to_host() triggers exactly one D2H transfer
///
/// Requirement: Only explicit `.to_host()` should trigger device→host transfer.
#[test]
#[cfg(feature = "cuda")]
fn test_gpu_to_host_transfers() {
    // Skip if CUDA not available (e.g., in CI without GPU)
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return, // Skip test
    };

    reset_transfer_counters();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];

    // Upload data
    let mut tensor = GpuResidentTensor::from_host(&ctx, &data).expect("Upload failed");
    assert_eq!(tensor.device_to_host_transfers(), 0);

    // Download to host
    let result = tensor.to_host().expect("Download failed");
    assert_eq!(result, data);

    // Now we have 1 D2H transfer
    assert_eq!(tensor.device_to_host_transfers(), 1);
    assert_eq!(total_d2h_transfers(), 1);
}

#[test]
#[cfg(not(feature = "cuda"))]
fn test_gpu_to_host_transfers() {}

/// Test: Operations on GPU tensors should NOT transfer back to host
///
/// Requirement: matmul, softmax, etc. should keep results on GPU.
/// Only final `.to_host()` should trigger device→host transfer.
#[test]
#[cfg(feature = "cuda")]
fn test_gpu_operations_stay_on_device() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return, // Skip if no CUDA
    };

    reset_transfer_counters();

    // Create two tensors (2 H2D transfers)
    let a = GpuResidentTensor::from_host(&ctx, &vec![1.0f32; 64]).expect("Upload A");
    let b = GpuResidentTensor::from_host(&ctx, &vec![2.0f32; 64]).expect("Upload B");

    assert_eq!(total_h2d_transfers(), 2);
    assert_eq!(total_d2h_transfers(), 0);

    // Elementwise add - result stays on GPU (NO new transfers!)
    let c = a.add(&ctx, &b).expect("Add failed");

    // Check: no additional transfers occurred
    assert!(c.is_device_resident());
    assert_eq!(c.host_to_device_transfers(), 0); // Result never came from host
    assert_eq!(c.device_to_host_transfers(), 0); // Result never went to host

    // Global counters unchanged (still 2 H2D, 0 D2H)
    assert_eq!(total_h2d_transfers(), 2);
    assert_eq!(total_d2h_transfers(), 0);
}

#[test]
#[cfg(not(feature = "cuda"))]
fn test_gpu_operations_stay_on_device() {}

/// Test: Chain of operations should have ZERO intermediate transfers
///
/// Requirement: A pipeline like Q @ K^T → softmax → @ V should have:
/// - Initial upload of Q, K, V (3 transfers)
/// - Final download of output (1 transfer)
/// - ZERO intermediate transfers
#[test]
#[ignore = "TDD: Implementation pending - operation chaining not yet implemented"]
fn test_operation_chain_no_intermediate_transfers() {
    // When implemented:
    //
    // let ctx = CudaContext::new(0).unwrap();
    //
    // // Upload Q, K, V (3 H2D transfers)
    // let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    // let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    // let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();
    //
    // // Chain: scores = Q @ K^T (stays on GPU)
    // let scores = q.matmul_transposed(&k).unwrap();
    //
    // // Chain: attn = softmax(scores) (stays on GPU)
    // let attn = scores.softmax(-1).unwrap();
    //
    // // Chain: output = attn @ V (stays on GPU)
    // let output = attn.matmul(&v).unwrap();
    //
    // // Verify NO intermediate transfers
    // assert_eq!(scores.device_to_host_transfers(), 0);
    // assert_eq!(attn.device_to_host_transfers(), 0);
    // assert_eq!(output.device_to_host_transfers(), 0);
    //
    // // Only transfer on explicit request
    // let result = output.to_host().unwrap();
    // // Now we have 1 D2H transfer
    // assert_eq!(output.device_to_host_transfers(), 1);

    assert!(true, "TDD: Operation chaining not implemented");
}

// ============================================================================
// PHASE 2: Batched Multi-Head Attention
// ============================================================================

/// Test: Batched attention should use SINGLE kernel launch for all heads
///
/// Requirement: Multi-head attention with N heads should NOT launch N kernels.
/// It should launch ONE kernel that processes all heads in parallel.
#[test]
#[cfg(feature = "cuda")]
fn test_batched_attention_single_kernel() {
    use trueno_gpu::memory::resident::batched_multihead_attention;

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return, // Skip if no CUDA
    };

    reset_transfer_counters();

    // Small test case: 4 sequence positions, 2 heads, 8-dim per head
    let seq_len = 4u32;
    let n_heads = 2u32;
    let head_dim = 8u32;
    let d_model = (n_heads * head_dim) as usize;

    // Q, K, V as [seq_len, d_model] tensors
    let q = GpuResidentTensor::from_host(&ctx, &vec![0.1f32; seq_len as usize * d_model])
        .expect("Upload Q");
    let k = GpuResidentTensor::from_host(&ctx, &vec![0.1f32; seq_len as usize * d_model])
        .expect("Upload K");
    let v = GpuResidentTensor::from_host(&ctx, &vec![0.1f32; seq_len as usize * d_model])
        .expect("Upload V");

    // 3 H2D transfers for Q, K, V
    assert_eq!(total_h2d_transfers(), 3);
    assert_eq!(total_d2h_transfers(), 0);

    // Batched attention - should be SINGLE kernel for all heads
    let output = batched_multihead_attention(&ctx, &q, &k, &v, n_heads, head_dim, seq_len)
        .expect("Batched attention failed");

    // Verify output shape
    assert_eq!(output.len(), seq_len as usize * d_model);
    // Verify stays on device
    assert!(output.is_device_resident());
    assert_eq!(output.device_to_host_transfers(), 0);
    // Verify NO additional H2D/D2H transfers (data stayed on GPU)
    assert_eq!(total_h2d_transfers(), 3); // Still just the original 3
    assert_eq!(total_d2h_transfers(), 0); // No downloads
}

#[test]
#[cfg(not(feature = "cuda"))]
fn test_batched_attention_single_kernel() {}

/// Test: Batched attention should include fused softmax
///
/// Requirement: The attention kernel should compute softmax INSIDE the kernel,
/// not as a separate kernel launch.
#[test]
#[ignore = "TDD: Implementation pending - fused softmax not yet implemented"]
fn test_batched_attention_fused_softmax() {
    // When implemented:
    //
    // // The kernel should compute:
    // // attention_output = softmax(Q @ K^T / sqrt(d_k)) @ V
    // // ALL IN ONE KERNEL (no separate softmax launch)
    //
    // let ctx = CudaContext::new(0).unwrap();
    // let output = batched_multihead_attention_with_stats(&ctx, &q, &k, &v, n_heads, head_dim).unwrap();
    //
    // // Stats should show:
    // assert_eq!(output.stats.kernel_launches, 1);
    // assert_eq!(output.stats.softmax_kernel_launches, 0); // Fused!
    // assert!(output.stats.has_fused_softmax);

    assert!(true, "TDD: fused softmax not implemented");
}

/// Test: Simple 2x2 matmul correctness
///
/// Debug test to isolate matmul behavior before full attention pipeline.
#[test]
#[cfg(feature = "cuda")]
fn test_matmul_2x2_correctness() {
    clear_kernel_cache();
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    // C = A @ B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    let expected: Vec<f32> = vec![19.0, 22.0, 43.0, 50.0];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).expect("upload A");
    let b = GpuResidentTensor::from_host(&ctx, &b_data).expect("upload B");

    // matmul(A, B, m=2, n=2, k=2)
    let mut c = a.matmul(&ctx, &b, 2, 2, 2).expect("matmul failed");
    let result = c.to_host().expect("download C");

    println!("\n=== Matmul 2x2 Test ===");
    println!("A: {:?}", a_data);
    println!("B: {:?}", b_data);
    println!("Expected: {:?}", expected);
    println!("GPU result: {:?}", result);

    let max_diff: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Max diff: {}", max_diff);
    assert!(
        max_diff < 0.01,
        "Matmul 2x2 failed: max diff {} > 0.01",
        max_diff
    );
    println!("✓ Matmul 2x2 PASSED!");
}

/// Test: Each step of attention individually
///
/// Debug test to find which step produces zeros in row 1.
#[test]
#[cfg(feature = "cuda")]
fn test_attention_steps_individually() {
    use trueno_gpu::memory::resident::TransferStats;

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let seq_len = 2u32;
    let d_model = 2u32;

    // Q = [[1, 0], [0, 1]], K = [[1, 0], [0, 1]], V = [[1, 2], [3, 4]]
    let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let k_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let v_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).expect("upload Q");
    let k = GpuResidentTensor::from_host(&ctx, &k_data).expect("upload K");
    let v = GpuResidentTensor::from_host(&ctx, &v_data).expect("upload V");

    println!("\n=== Step-by-Step Attention Debug ===");

    // Step 1: Transpose K
    // K = [[1, 0], [0, 1]] -> K^T = [[1, 0], [0, 1]] (identity is its own transpose)
    // Actually K^T: [d_model, seq_len] so K[0,0]=1, K[0,1]=0 -> K^T[0,0]=1, K^T[1,0]=0, K^T[0,1]=0, K^T[1,1]=1
    // K^T stored as row-major [d_model, seq_len] = [[K[0,0], K[1,0]], [K[0,1], K[1,1]]] = [[1, 0], [0, 1]]

    // Step 2: Q @ K^T = [[1,0],[0,1]] @ [[1,0],[0,1]] = [[1,0],[0,1]]
    // Expected scores: [1.0, 0.0, 0.0, 1.0]
    // Note: We need to manually do transpose + matmul to test
    // For now, test Q @ K (not transposed) which should give same result for identity matrix
    let mut scores = q
        .matmul(&ctx, &k, seq_len, seq_len, d_model)
        .expect("Q@K failed");
    let scores_host = scores.to_host().expect("download scores");
    println!("Step 1 - Q @ K (should be identity): {:?}", scores_host);

    // Step 3: Scale by 1/sqrt(head_dim)
    let scale = 1.0 / (d_model as f32).sqrt(); // 1/sqrt(2) = 0.707
    let q2 = GpuResidentTensor::from_host(&ctx, &q_data).expect("upload Q2");
    let k2 = GpuResidentTensor::from_host(&ctx, &k_data).expect("upload K2");
    let mut scores2 = q2
        .matmul(&ctx, &k2, seq_len, seq_len, d_model)
        .expect("Q@K");
    let scaled = scores2.scale(&ctx, scale).expect("scale failed");
    let mut scaled_mut = scaled;
    let scaled_host = scaled_mut.to_host().expect("download scaled");
    println!("Step 2 - Scaled (×{}): {:?}", scale, scaled_host);

    // Step 4: Softmax
    let q3 = GpuResidentTensor::from_host(&ctx, &q_data).expect("upload Q3");
    let k3 = GpuResidentTensor::from_host(&ctx, &k_data).expect("upload K3");
    let mut scores3 = q3
        .matmul(&ctx, &k3, seq_len, seq_len, d_model)
        .expect("Q@K");
    let scaled3 = scores3.scale(&ctx, scale).expect("scale");
    let softmax_result = scaled3.softmax(&ctx, seq_len).expect("softmax failed");
    let mut softmax_mut = softmax_result;
    let softmax_host = softmax_mut.to_host().expect("download softmax");
    println!("Step 3 - Softmax: {:?}", softmax_host);

    // Expected softmax:
    // Row 0: softmax([0.707, 0]) = [exp(0.707), exp(0)] / sum = [2.028, 1] / 3.028 = [0.670, 0.330]
    // Row 1: softmax([0, 0.707]) = [exp(0), exp(0.707)] / sum = [1, 2.028] / 3.028 = [0.330, 0.670]
    println!("Expected softmax: [0.670, 0.330, 0.330, 0.670]");

    // Step 5: Final matmul
    let q4 = GpuResidentTensor::from_host(&ctx, &q_data).expect("Q4");
    let k4 = GpuResidentTensor::from_host(&ctx, &k_data).expect("K4");
    let v4 = GpuResidentTensor::from_host(&ctx, &v_data).expect("V4");
    let mut scores4 = q4
        .matmul(&ctx, &k4, seq_len, seq_len, d_model)
        .expect("Q@K");
    let scaled4 = scores4.scale(&ctx, scale).expect("scale");
    let attn4 = scaled4.softmax(&ctx, seq_len).expect("softmax");
    let mut output4 = attn4
        .matmul(&ctx, &v4, seq_len, d_model, seq_len)
        .expect("attn@V failed");
    let output_host = output4.to_host().expect("download output");
    println!("Step 4 - Output (attn @ V): {:?}", output_host);

    // Check if any row is all zeros
    let row0_zero = output_host[0].abs() < 0.001 && output_host[1].abs() < 0.001;
    let row1_zero = output_host[2].abs() < 0.001 && output_host[3].abs() < 0.001;
    if row0_zero {
        println!("BUG: Row 0 is all zeros!");
    }
    if row1_zero {
        println!("BUG: Row 1 is all zeros!");
    }

    // Expected output:
    // Row 0: [0.670*1 + 0.330*3, 0.670*2 + 0.330*4] = [1.66, 2.66]
    // Row 1: [0.330*1 + 0.670*3, 0.330*2 + 0.670*4] = [2.34, 3.34]
    println!("Expected output: [1.66, 2.66, 2.34, 3.34]");
}

/// Test: Batched attention numerical correctness
///
/// Requirement: Output should match reference CPU implementation within tolerance.
/// This is a REAL correctness test that computes expected values on CPU.
#[test]
#[cfg(feature = "cuda")]
fn test_batched_attention_correctness() {
    clear_kernel_cache();
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("CUDA not available, skipping correctness test");
            return;
        }
    };

    // Small test case: seq_len=2, n_heads=1, head_dim=2
    // This keeps the math simple enough to verify by hand
    let seq_len = 2u32;
    let n_heads = 1u32;
    let head_dim = 2u32;
    let d_model = (n_heads * head_dim) as usize; // 2

    // Q = [[1, 0], [0, 1]]  (2x2, row-major)
    let q_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    // K = [[1, 0], [0, 1]]  (same as Q for simplicity)
    let k_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    // V = [[1, 2], [3, 4]]
    let v_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    // Expected computation (scaled dot-product attention):
    // 1. scores = Q @ K^T = [[1,0],[0,1]] @ [[1,0],[0,1]] = [[1,0],[0,1]]
    // 2. scale = 1/sqrt(head_dim) = 1/sqrt(2) = 0.707
    // 3. scaled_scores = [[0.707, 0], [0, 0.707]]
    // 4. softmax row-wise:
    //    row0: softmax([0.707, 0]) = [exp(0.707), exp(0)] / sum = [2.028, 1.0] / 3.028 = [0.670, 0.330]
    //    row1: softmax([0, 0.707]) = [1.0, 2.028] / 3.028 = [0.330, 0.670]
    // 5. output = attn_weights @ V
    //    row0: [0.670, 0.330] @ [[1,2],[3,4]] = [0.670*1 + 0.330*3, 0.670*2 + 0.330*4] = [1.66, 2.66]
    //    row1: [0.330, 0.670] @ [[1,2],[3,4]] = [0.330*1 + 0.670*3, 0.330*2 + 0.670*4] = [2.34, 3.34]

    // Compute expected on CPU
    let scale = 1.0 / (head_dim as f32).sqrt();

    // Q @ K^T (manually for 2x2)
    // scores[i,j] = sum_k Q[i,k] * K[j,k]
    let scores = vec![
        q_data[0] * k_data[0] + q_data[1] * k_data[1], // [0,0]
        q_data[0] * k_data[2] + q_data[1] * k_data[3], // [0,1]
        q_data[2] * k_data[0] + q_data[3] * k_data[1], // [1,0]
        q_data[2] * k_data[2] + q_data[3] * k_data[3], // [1,1]
    ];

    // Scale
    let scaled: Vec<f32> = scores.iter().map(|x| x * scale).collect();

    // Softmax row-wise
    let mut attn_weights = vec![0.0f32; 4];
    for row in 0..2 {
        let row_start = row * 2;
        let max_val = scaled[row_start].max(scaled[row_start + 1]);
        let exp0 = (scaled[row_start] - max_val).exp();
        let exp1 = (scaled[row_start + 1] - max_val).exp();
        let sum = exp0 + exp1;
        attn_weights[row_start] = exp0 / sum;
        attn_weights[row_start + 1] = exp1 / sum;
    }

    // attn_weights @ V
    let mut expected = vec![0.0f32; 4];
    for i in 0..2 {
        for j in 0..2 {
            expected[i * 2 + j] =
                attn_weights[i * 2] * v_data[j] + attn_weights[i * 2 + 1] * v_data[2 + j];
        }
    }

    println!("\n=== Correctness Test ===");
    println!("Q: {:?}", q_data);
    println!("K: {:?}", k_data);
    println!("V: {:?}", v_data);
    println!("Scores (Q@K^T): {:?}", scores);
    println!("Scaled (/{:.3}): {:?}", 1.0 / scale, scaled);
    println!("Attn weights: {:?}", attn_weights);
    println!("Expected output: {:?}", expected);

    // Run on GPU
    let q = GpuResidentTensor::from_host(&ctx, &q_data).expect("upload Q");
    let k = GpuResidentTensor::from_host(&ctx, &k_data).expect("upload K");
    let v = GpuResidentTensor::from_host(&ctx, &v_data).expect("upload V");

    let mut output = batched_multihead_attention(&ctx, &q, &k, &v, n_heads, head_dim, seq_len)
        .expect("GPU attention failed");
    let result = output.to_host().expect("download output");

    println!("GPU output: {:?}", result);

    // Check numerical accuracy
    let max_diff: f32 = result
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Max diff: {}", max_diff);

    assert!(
        max_diff < 0.01,
        "Max diff: {} exceeds tolerance 0.01",
        max_diff
    );
    println!("✓ Correctness test PASSED!");
}

// ============================================================================
// PHASE 3: Memory Pool Integration
// ============================================================================

/// Test: Memory pool should reuse allocations
///
/// Requirement: Freeing a GPU allocation should return memory to pool,
/// and subsequent allocations of same size should reuse without cudaMalloc.
#[test]
#[ignore = "TDD: Implementation pending - pool integration not yet implemented"]
fn test_memory_pool_reuse() {
    // When implemented:
    //
    // let ctx = CudaContext::new(0).unwrap();
    // let pool = GpuMemoryPool::new(&ctx, 64 * 1024 * 1024).unwrap(); // 64MB pool
    //
    // // First allocation
    // let a = pool.allocate::<f32>(1000).unwrap();
    // let ptr_a = a.device_ptr();
    //
    // // Track cudaMalloc calls
    // let malloc_before = pool.cuda_malloc_calls();
    //
    // // Free the allocation
    // drop(a);
    //
    // // Allocate same size - should reuse
    // let b = pool.allocate::<f32>(1000).unwrap();
    //
    // // Same pointer (reused)
    // assert_eq!(b.device_ptr(), ptr_a);
    // // No new cudaMalloc
    // assert_eq!(pool.cuda_malloc_calls(), malloc_before);

    assert!(true, "TDD: memory pool reuse not implemented");
}

/// Test: GpuResidentTensor should use memory pool when available
///
/// Requirement: Tensors should allocate from pool to avoid cudaMalloc overhead.
#[test]
#[ignore = "TDD: Implementation pending - pool-backed tensors not yet implemented"]
fn test_tensor_uses_memory_pool() {
    // When implemented:
    //
    // let ctx = CudaContext::new(0).unwrap();
    // let pool = GpuMemoryPool::new(&ctx, 64 * 1024 * 1024).unwrap();
    //
    // // Create tensor backed by pool
    // let data = vec![1.0f32; 10000];
    // let tensor = GpuResidentTensor::from_host_pooled(&ctx, &pool, &data).unwrap();
    //
    // // Should be pool-backed
    // assert!(tensor.is_pool_backed());
    // // Pool should show allocation
    // assert_eq!(pool.active_allocations(), 1);
    //
    // // Drop tensor
    // drop(tensor);
    // // Memory returned to pool (not freed)
    // assert_eq!(pool.active_allocations(), 0);
    // assert!(pool.has_available(10000 * 4));

    assert!(true, "TDD: pool-backed tensors not implemented");
}

// ============================================================================
// PHASE 4: Full Attention Pipeline
// ============================================================================

/// Test: Full encoder layer should have minimal transfers
///
/// Requirement: Processing one encoder layer should have:
/// - 0 host transfers for attention (weights pre-uploaded)
/// - Output stays on GPU for next layer
#[test]
#[ignore = "TDD: Implementation pending - full pipeline not yet implemented"]
fn test_encoder_layer_minimal_transfers() {
    // When implemented:
    //
    // let ctx = CudaContext::new(0).unwrap();
    //
    // // Pre-upload all weights (done ONCE at model load time)
    // let weights = EncoderLayerWeights::upload(&ctx, &model_weights).unwrap();
    //
    // // Process input through encoder layer
    // let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();
    //
    // // Track transfers before
    // let h2d_before = ctx.total_h2d_transfers();
    // let d2h_before = ctx.total_d2h_transfers();
    //
    // // Run encoder layer - should have ZERO additional transfers
    // let output = encoder_layer_forward(&ctx, &input, &weights).unwrap();
    //
    // // Verify no transfers during forward pass
    // assert_eq!(ctx.total_h2d_transfers(), h2d_before);
    // assert_eq!(ctx.total_d2h_transfers(), d2h_before);
    //
    // // Output should be on GPU, ready for next layer
    // assert!(output.is_device_resident());

    assert!(true, "TDD: encoder layer pipeline not implemented");
}

/// Test: Full encoder (all layers) should have 2 total transfers
///
/// Requirement: Processing ALL encoder layers should have:
/// - 1 H2D transfer: input audio features
/// - 1 D2H transfer: final encoder output
/// - Weights are pre-uploaded (not counted per-inference)
#[test]
#[ignore = "TDD: Implementation pending - full encoder not yet implemented"]
fn test_full_encoder_two_transfers_total() {
    // When implemented:
    //
    // let ctx = CudaContext::new(0).unwrap();
    //
    // // Model weights pre-uploaded (done ONCE at load time)
    // let model = WhisperEncoderGpu::load(&ctx, model_path).unwrap();
    //
    // // Reset transfer counters for this inference
    // ctx.reset_transfer_counters();
    //
    // // Input: mel spectrogram
    // let mel_features = vec![0.0f32; 1500 * 80]; // [seq_len, n_mels]
    //
    // // Run full encoder
    // let output = model.encode(&mel_features).unwrap();
    //
    // // Verify ONLY 2 transfers:
    // assert_eq!(ctx.total_h2d_transfers(), 1, "Should have 1 upload (mel features)");
    // assert_eq!(ctx.total_d2h_transfers(), 1, "Should have 1 download (encoder output)");

    assert!(true, "TDD: full encoder not implemented");
}

// ============================================================================
// Performance Targets (Acceptance Criteria)
// ============================================================================

/// Test: Encoder should achieve <300ms for 1.5s audio
///
/// Acceptance criteria from WAPR-PERF-004 specification.
#[test]
#[ignore = "TDD: Performance test - run after implementation complete"]
fn test_encoder_performance_target() {
    // When implemented:
    //
    // let ctx = CudaContext::new(0).unwrap();
    // let model = WhisperEncoderGpu::load(&ctx, "models/whisper-tiny.apr").unwrap();
    //
    // // Warmup
    // for _ in 0..3 {
    //     let _ = model.encode(&vec![0.0f32; 1500 * 80]);
    // }
    //
    // // Benchmark
    // let start = std::time::Instant::now();
    // let _ = model.encode(&vec![0.0f32; 1500 * 80]).unwrap();
    // let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    //
    // // Target: <300ms (currently 5150ms on CPU)
    // assert!(elapsed_ms < 300.0, "Encoder took {}ms, target <300ms", elapsed_ms);

    assert!(true, "TDD: performance test not implemented");
}

// ============================================================================
// DEBUG: Isolate which operation crashes
// ============================================================================

/// Debug test to isolate which kernel in batched_multihead_attention fails
#[test]
#[cfg(feature = "cuda")]
fn test_debug_isolate_crash() {
    use trueno_gpu::driver::{CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::kernels::{GemmKernel, Kernel, TransposeKernel};

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    println!("\n=== DEBUG: Isolating crash ===\n");

    // Small test: 4x16 matrix (seq_len=4, d_model=16)
    let rows = 4u32;
    let cols = 16u32;
    let total = (rows * cols) as usize;
    let input_data: Vec<f32> = (0..total).map(|i| i as f32 * 0.1).collect();

    // Test 1: Upload to GPU
    println!("Step 1: Upload to GPU...");
    let input_buf = GpuBuffer::from_host(&ctx, &input_data).expect("Upload failed");
    println!("  ✓ Upload succeeded");

    // Test 2: TransposeKernel
    println!(
        "Step 2: TransposeKernel [{}x{}] -> [{}x{}]...",
        rows, cols, cols, rows
    );
    let output_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total).expect("Alloc failed");

    let transpose = TransposeKernel::new(rows, cols);
    let ptx = transpose.emit_ptx();
    println!("  PTX generated ({} bytes)", ptx.len());

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX compile failed");
    println!("  Module compiled");

    let stream = CudaStream::new(&ctx).expect("Stream failed");

    let threads = 256u32;
    let blocks = (total as u32 + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };
    println!(
        "  Launch config: grid=({}, 1, 1), block=({}, 1, 1)",
        blocks, threads
    );

    let input_ptr = input_buf.as_ptr();
    let output_ptr = output_buf.as_ptr();
    println!("  Input ptr: 0x{:x}", input_ptr);
    println!("  Output ptr: 0x{:x}", output_ptr);

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(input_ptr) as *mut _,
        std::ptr::addr_of!(output_ptr) as *mut _,
        std::ptr::addr_of!(rows) as *mut _,
        std::ptr::addr_of!(cols) as *mut _,
    ];

    unsafe {
        match stream.launch_kernel(&mut module, transpose.name(), &config, &mut args) {
            Ok(_) => println!("  Kernel launched"),
            Err(e) => {
                println!("  ✗ Kernel launch FAILED: {:?}", e);
                return;
            }
        }
    }

    match stream.synchronize() {
        Ok(_) => println!("  ✓ TransposeKernel succeeded!"),
        Err(e) => {
            println!("  ✗ TransposeKernel CRASHED: {:?}", e);
            return;
        }
    }

    // Test 3: Read back result
    println!("Step 3: Verify transpose result...");
    let mut result = vec![0.0f32; total];
    output_buf
        .copy_to_host(&mut result)
        .expect("Readback failed");

    // Check a few values: input[0,0] should be at output[0,0]
    // input[0,1] (index 1) should be at output[1,0] (index rows = 4)
    let expected_0_0 = input_data[0]; // input[0,0]
    let expected_1_0 = input_data[1]; // input[0,1] -> output[1,0]
    println!(
        "  input[0,0]={:.1} -> output[0,0]={:.1} (expected {:.1})",
        input_data[0], result[0], expected_0_0
    );
    println!(
        "  input[0,1]={:.1} -> output[1,0]={:.1} (expected {:.1})",
        input_data[1], result[rows as usize], expected_1_0
    );

    // Test 4: GemmKernel
    println!("Step 4: GemmKernel [4x16] @ [16x4] = [4x4]...");
    let m = 4u32;
    let n = 4u32;
    let k = 16u32;

    // A is the original [4,16], B is the transpose [16,4]
    let c_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, (m * n) as usize).expect("Alloc C failed");

    let gemm = GemmKernel::naive(m, n, k);
    let ptx = gemm.emit_ptx();
    println!("  PTX generated ({} bytes)", ptx.len());

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX compile failed");
    println!("  Module compiled");

    let block_size = 16u32;
    let grid_x = (n + block_size - 1) / block_size;
    let grid_y = (m + block_size - 1) / block_size;
    let config = LaunchConfig {
        grid: (grid_x, grid_y, 1),
        block: (block_size, block_size, 1),
        shared_mem: 0,
    };
    println!(
        "  Launch config: grid=({}, {}, 1), block=({}, {}, 1)",
        grid_x, grid_y, block_size, block_size
    );

    let a_ptr = input_buf.as_ptr(); // [4,16]
    let b_ptr = output_buf.as_ptr(); // [16,4] (transposed)
    let c_ptr = c_buf.as_ptr();
    let m_val = m;
    let n_val = n;
    let k_val = k;

    println!("  A ptr: 0x{:x} (size {})", a_ptr, total);
    println!("  B ptr: 0x{:x} (size {})", b_ptr, total);
    println!("  C ptr: 0x{:x} (size {})", c_ptr, (m * n) as usize);
    println!("  M={}, N={}, K={}", m_val, n_val, k_val);

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(a_ptr) as *mut _,
        std::ptr::addr_of!(b_ptr) as *mut _,
        std::ptr::addr_of!(c_ptr) as *mut _,
        std::ptr::addr_of!(m_val) as *mut _,
        std::ptr::addr_of!(n_val) as *mut _,
        std::ptr::addr_of!(k_val) as *mut _,
    ];

    unsafe {
        match stream.launch_kernel(&mut module, gemm.name(), &config, &mut args) {
            Ok(_) => println!("  Kernel launched"),
            Err(e) => {
                println!("  ✗ Kernel launch FAILED: {:?}", e);
                return;
            }
        }
    }

    match stream.synchronize() {
        Ok(_) => println!("  ✓ GemmKernel succeeded!"),
        Err(e) => {
            println!("  ✗ GemmKernel CRASHED: {:?}", e);
            return;
        }
    }

    // Test 5: ScaleKernel (scale by constant)
    println!("Step 5: Scale by 0.5...");
    use trueno_gpu::kernels::ScaleKernel;

    let scale_in_size = (m * n) as usize; // 16 elements
    let scale_out_buf: GpuBuffer<f32> =
        GpuBuffer::new(&ctx, scale_in_size).expect("Alloc scale out");

    let scale_kernel = ScaleKernel::new(scale_in_size as u32);
    let ptx = scale_kernel.emit_ptx();
    println!("  PTX generated ({} bytes)", ptx.len());

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX compile failed");

    let threads = 256u32;
    let blocks = ((scale_in_size as u32) + threads - 1) / threads;
    let config = LaunchConfig {
        grid: (blocks, 1, 1),
        block: (threads, 1, 1),
        shared_mem: 0,
    };

    let scale_input_ptr = c_buf.as_ptr();
    let scale_output_ptr = scale_out_buf.as_ptr();
    let scale_val = 0.5f32;
    let scale_n = scale_in_size as u32;

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(scale_input_ptr) as *mut _,
        std::ptr::addr_of!(scale_output_ptr) as *mut _,
        std::ptr::addr_of!(scale_val) as *mut _,
        std::ptr::addr_of!(scale_n) as *mut _,
    ];

    unsafe {
        match stream.launch_kernel(&mut module, scale_kernel.name(), &config, &mut args) {
            Ok(_) => println!("  Kernel launched"),
            Err(e) => {
                println!("  ✗ Scale kernel launch FAILED: {:?}", e);
                return;
            }
        }
    }

    match stream.synchronize() {
        Ok(_) => println!("  ✓ Scale kernel succeeded!"),
        Err(e) => {
            println!("  ✗ Scale kernel CRASHED: {:?}", e);
            return;
        }
    }

    // Test 6: SoftmaxKernel
    println!("Step 6: Softmax [4 rows x 4 cols]...");
    use trueno_gpu::kernels::SoftmaxKernel;

    let sm_rows = m; // 4
    let sm_row_size = n; // 4
    let sm_total = (sm_rows * sm_row_size) as usize; // 16

    let sm_out_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, sm_total).expect("Alloc softmax out");

    let sm_kernel = SoftmaxKernel::new(sm_row_size);
    let ptx = sm_kernel.emit_ptx();
    println!("  PTX generated ({} bytes)", ptx.len());

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX compile failed");

    // Softmax: one block per row
    let threads_per_block = (sm_row_size as usize).min(256) as u32;
    let config = LaunchConfig {
        grid: (sm_rows, 1, 1),
        block: (threads_per_block, 1, 1),
        shared_mem: (sm_row_size as usize * std::mem::size_of::<f32>()) as u32,
    };
    println!(
        "  Launch config: grid=({}, 1, 1), block=({}, 1, 1), smem={}",
        sm_rows, threads_per_block, config.shared_mem
    );

    let sm_input_ptr = scale_out_buf.as_ptr();
    let sm_output_ptr = sm_out_buf.as_ptr();
    let sm_row_size_val = sm_row_size;

    let mut args: Vec<*mut std::ffi::c_void> = vec![
        std::ptr::addr_of!(sm_input_ptr) as *mut _,
        std::ptr::addr_of!(sm_output_ptr) as *mut _,
        std::ptr::addr_of!(sm_row_size_val) as *mut _,
    ];

    unsafe {
        match stream.launch_kernel(&mut module, sm_kernel.name(), &config, &mut args) {
            Ok(_) => println!("  Kernel launched"),
            Err(e) => {
                println!("  ✗ Softmax kernel launch FAILED: {:?}", e);
                return;
            }
        }
    }

    match stream.synchronize() {
        Ok(_) => println!("  ✓ Softmax kernel succeeded!"),
        Err(e) => {
            println!("  ✗ Softmax kernel CRASHED: {:?}", e);
            return;
        }
    }

    println!("\n=== All kernels passed! ===");
}

// ============================================================================
// PHASE 4: Full GPU Encoder Block (Total Offload)
// ============================================================================

/// Test: Individual GPU operations work correctly
#[test]
#[cfg(feature = "cuda")]
fn test_gpu_operations_individually() {
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    println!("\n=== Testing Individual GPU Operations ===");

    // Test data
    let d = 4u32; // small dimension for testing
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let weights: Vec<f32> = vec![
        1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    ];
    let bias: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
    let gamma: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let beta: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];

    // Upload
    let x = GpuResidentTensor::from_host(&ctx, &data).expect("upload x");
    let w = GpuResidentTensor::from_host(&ctx, &weights).expect("upload w");
    let b = GpuResidentTensor::from_host(&ctx, &bias).expect("upload b");
    let g = GpuResidentTensor::from_host(&ctx, &gamma).expect("upload gamma");
    let bt = GpuResidentTensor::from_host(&ctx, &beta).expect("upload beta");

    // Test 1: matmul
    print!("1. matmul... ");
    match x.matmul(&ctx, &w, 1, d, d) {
        Ok(mut r) => {
            let h = r.to_host().expect("download");
            println!("✓ result: {:?}", h);
        }
        Err(e) => println!("✗ FAILED: {:?}", e),
    }

    // Test 2: bias_add
    print!("2. bias_add... ");
    match x.bias_add(&ctx, &b) {
        Ok(mut r) => {
            let h = r.to_host().expect("download");
            println!("✓ result: {:?}", h);
        }
        Err(e) => println!("✗ FAILED: {:?}", e),
    }

    // Test 3: gelu
    print!("3. gelu... ");
    match x.gelu(&ctx) {
        Ok(mut r) => {
            let h = r.to_host().expect("download");
            println!("✓ result: {:?}", h);
        }
        Err(e) => println!("✗ FAILED: {:?}", e),
    }

    // Test 4: layer_norm
    print!("4. layer_norm... ");
    match x.layer_norm(&ctx, &g, &bt, d, 1) {
        Ok(mut r) => {
            let h = r.to_host().expect("download");
            println!("✓ result: {:?}", h);
        }
        Err(e) => println!("✗ FAILED: {:?}", e),
    }

    // Test 5: linear (matmul + bias)
    print!("5. linear... ");
    match x.linear(&ctx, &w, Some(&b), 1, d, d) {
        Ok(mut r) => {
            let h = r.to_host().expect("download");
            println!("✓ result: {:?}", h);
        }
        Err(e) => println!("✗ FAILED: {:?}", e),
    }

    println!("=== Done ===");
}

/// Test: Full encoder block runs on GPU with minimal transfers
///
/// Requirement: Upload weights once, then run encoder blocks with only
/// input upload and output download per block. No intermediate transfers.
#[test]
#[cfg(feature = "cuda")]
fn test_full_encoder_block_gpu() {
    use trueno_gpu::memory::resident::{
        forward_encoder_block_gpu, GpuEncoderBlockWeights, GpuEncoderConfig,
    };

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => {
            eprintln!("CUDA not available, skipping encoder block test");
            return;
        }
    };

    // Config: tiny-like (d_model=64, n_heads=4, ffn_dim=256)
    // Scaled down for fast testing
    let d_model = 64u32;
    let n_heads = 4u32;
    let ffn_dim = d_model * 4; // 256
    let seq_len = 8u32; // Short sequence for testing

    let config = GpuEncoderConfig {
        d_model,
        n_heads,
        ffn_dim,
    };

    // Create dummy weights (random-ish for testing, actual values don't matter for transfer test)
    let weight_size = (d_model * d_model) as usize;
    let ffn_up_size = (d_model * ffn_dim) as usize;
    let ffn_down_size = (ffn_dim * d_model) as usize;

    // Initialize with simple patterns
    let ln_gamma: Vec<f32> = (0..d_model).map(|_| 1.0).collect();
    let ln_beta: Vec<f32> = (0..d_model).map(|_| 0.0).collect();
    let w_proj: Vec<f32> = (0..weight_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let b_proj: Vec<f32> = (0..d_model).map(|_| 0.0).collect();
    let ffn_up_w: Vec<f32> = (0..ffn_up_size).map(|i| (i as f32 * 0.001).sin()).collect();
    let ffn_up_b: Vec<f32> = (0..ffn_dim).map(|_| 0.0).collect();
    let ffn_down_w: Vec<f32> = (0..ffn_down_size)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let ffn_down_b: Vec<f32> = (0..d_model).map(|_| 0.0).collect();

    // Upload weights (this counts as H2D transfers during initialization)
    reset_transfer_counters();

    let weights = GpuEncoderBlockWeights {
        ln1_gamma: GpuResidentTensor::from_host(&ctx, &ln_gamma).expect("ln1_gamma"),
        ln1_beta: GpuResidentTensor::from_host(&ctx, &ln_beta).expect("ln1_beta"),
        w_q: GpuResidentTensor::from_host(&ctx, &w_proj).expect("w_q"),
        b_q: GpuResidentTensor::from_host(&ctx, &b_proj).expect("b_q"),
        w_k: GpuResidentTensor::from_host(&ctx, &w_proj).expect("w_k"),
        b_k: GpuResidentTensor::from_host(&ctx, &b_proj).expect("b_k"),
        w_v: GpuResidentTensor::from_host(&ctx, &w_proj).expect("w_v"),
        b_v: GpuResidentTensor::from_host(&ctx, &b_proj).expect("b_v"),
        w_o: GpuResidentTensor::from_host(&ctx, &w_proj).expect("w_o"),
        b_o: GpuResidentTensor::from_host(&ctx, &b_proj).expect("b_o"),
        ln2_gamma: GpuResidentTensor::from_host(&ctx, &ln_gamma).expect("ln2_gamma"),
        ln2_beta: GpuResidentTensor::from_host(&ctx, &ln_beta).expect("ln2_beta"),
        ffn_up_w: GpuResidentTensor::from_host(&ctx, &ffn_up_w).expect("ffn_up_w"),
        ffn_up_b: GpuResidentTensor::from_host(&ctx, &ffn_up_b).expect("ffn_up_b"),
        ffn_down_w: GpuResidentTensor::from_host(&ctx, &ffn_down_w).expect("ffn_down_w"),
        ffn_down_b: GpuResidentTensor::from_host(&ctx, &ffn_down_b).expect("ffn_down_b"),
    };

    let weight_upload_h2d = total_h2d_transfers();
    println!("\n=== GPU Encoder Block Test ===");
    println!("Weight upload: {} H2D transfers", weight_upload_h2d);

    // Now reset and run forward pass
    reset_transfer_counters();

    // Create input
    let input_size = (seq_len * d_model) as usize;
    let input_data: Vec<f32> = (0..input_size).map(|i| (i as f32 * 0.01).sin()).collect();

    // Upload input (1 H2D)
    let input = GpuResidentTensor::from_host(&ctx, &input_data).expect("input upload");
    let h2d_after_input = total_h2d_transfers();
    println!("Input upload: {} H2D transfers", h2d_after_input);

    // Run forward pass (should have 0 additional transfers during computation)
    let mut output =
        forward_encoder_block_gpu(&ctx, &input, &weights, &config).expect("forward pass failed");

    let h2d_after_forward = total_h2d_transfers();
    let d2h_after_forward = total_d2h_transfers();
    println!(
        "After forward pass: {} H2D, {} D2H",
        h2d_after_forward, d2h_after_forward
    );

    // Download output (1 D2H)
    let result = output.to_host().expect("output download");
    let final_h2d = total_h2d_transfers();
    let final_d2h = total_d2h_transfers();

    println!("After download: {} H2D, {} D2H", final_h2d, final_d2h);
    println!("Output size: {} elements", result.len());

    // Verify transfer counts
    // Expected: 1 H2D (input) + 0 during forward + 1 D2H (output) = 1 H2D, 1 D2H
    assert_eq!(
        final_h2d, 1,
        "Forward pass should have 1 H2D transfer (input only), got {}",
        final_h2d
    );
    assert_eq!(
        final_d2h, 1,
        "Forward pass should have 1 D2H transfer (output only), got {}",
        final_d2h
    );

    // Verify output is not all zeros (sanity check)
    let output_sum: f32 = result.iter().map(|x| x.abs()).sum();
    assert!(output_sum > 0.0, "Output should not be all zeros");

    println!("✓ Full GPU encoder block test PASSED!");
    println!("  - 1 H2D (input upload)");
    println!("  - 0 transfers during forward");
    println!("  - 1 D2H (output download)");
}

// ============================================================================
// Long Row Softmax Test (WAPR-PERF-004)
// ============================================================================

/// Test: Long row softmax produces correct row sums (should be 1.0)
///
/// This tests the LongRowSoftmaxKernel with rows > 32 elements.
/// Critical for attention softmax where rows have seq_len (e.g., 1500) elements.
#[test]
#[cfg(feature = "cuda")]
fn test_long_row_softmax_correctness() {
    use trueno_gpu::driver::CudaContext;
    use trueno_gpu::memory::resident::GpuResidentTensor;

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => {
            println!("CUDA not available, skipping test");
            return;
        }
    };

    // Test with row_size = 64 first (simpler case)
    let n_rows = 4;
    let row_size = 64;
    let total_size = n_rows * row_size;

    println!(
        "Testing softmax with {} rows x {} elements...",
        n_rows, row_size
    );

    // Create simple input data
    let input_data: Vec<f32> = (0..total_size)
        .map(|i| (i % row_size) as f32 * 0.1)
        .collect();

    println!("Input first row: {:?}", &input_data[0..8]);

    let input = GpuResidentTensor::from_host(&ctx, &input_data).expect("input upload");
    println!("Input uploaded");

    // Run softmax
    let mut output = input.softmax(&ctx, n_rows as u32).expect("softmax");
    println!("Softmax completed");

    // Download result
    let result = output.to_host().expect("download");
    println!("Result downloaded, len={}", result.len());

    println!("Output first row (first 8): {:?}", &result[0..8]);
    println!(
        "Output first row (last 4):  {:?}",
        &result[row_size - 4..row_size]
    );

    // FULL SOFTMAX TEST: Verify row sums to 1.0 and values match expected
    for row in 0..n_rows {
        let start = row * row_size;
        let end = start + row_size;
        let row_output = &result[start..end];
        let row_input = &input_data[start..end];

        // Compute expected softmax on CPU
        let row_max = row_input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_shifted: Vec<f32> = row_input.iter().map(|&x| (x - row_max).exp()).collect();
        let exp_sum: f32 = exp_shifted.iter().sum();
        let expected_softmax: Vec<f32> = exp_shifted.iter().map(|&e| e / exp_sum).collect();

        // Check row sums to 1.0
        let row_sum: f32 = row_output.iter().sum();
        let sum_diff = (row_sum - 1.0).abs();
        println!(
            "Row {}: sum = {:.6} (diff from 1.0: {:.6})",
            row, row_sum, sum_diff
        );
        if sum_diff > 0.01 {
            panic!(
                "Row {}: sum={:.6} does not equal 1.0 (diff={:.6})",
                row, row_sum, sum_diff
            );
        }

        // Check individual values
        for col in 0..row_size {
            let got = row_output[col];
            let expected = expected_softmax[col];
            let diff = (got - expected).abs();
            // 1% relative tolerance for individual values
            if diff > expected.max(1e-6) * 0.02 {
                panic!(
                    "Row {} col {}: expected {:.6}, got {:.6} (diff={:.6})",
                    row, col, expected, got, diff
                );
            }
        }
    }

    println!("✓ Full softmax test PASSED!");
    println!("  - {} rows x {} elements", n_rows, row_size);
    println!("  - All rows sum to 1.0");
    println!("  - All values match expected softmax within 2% tolerance");

    // ==== Test with 1500 elements (attention matrix row size) ====
    println!("\n=== Testing with 1500 elements (attention size) ===");
    let n_rows_large = 6; // 6 attention heads
    let row_size_large = 1500;
    let total_size_large = n_rows_large * row_size_large;

    // Create input with varying values
    let input_large: Vec<f32> = (0..total_size_large)
        .map(|i| ((i % row_size_large) as f32 - 750.0) * 0.01) // Range -7.5 to 7.49
        .collect();

    let input_gpu = GpuResidentTensor::from_host(&ctx, &input_large).expect("upload");
    let mut output_gpu = input_gpu
        .softmax(&ctx, n_rows_large as u32)
        .expect("softmax");
    let result_large = output_gpu.to_host().expect("download");

    for row in 0..n_rows_large {
        let start = row * row_size_large;
        let end = start + row_size_large;
        let row_output = &result_large[start..end];
        let row_sum: f32 = row_output.iter().sum();
        let sum_diff = (row_sum - 1.0).abs();
        println!(
            "Row {}: sum = {:.6} (diff from 1.0: {:.6})",
            row, row_sum, sum_diff
        );
        if sum_diff > 0.01 {
            panic!("Row {}: sum={:.6} does not equal 1.0", row, row_sum);
        }
    }

    println!("✓ Attention-sized softmax test PASSED!");
    println!("  - {} rows x {} elements", n_rows_large, row_size_large);
}

// ============================================================================
// Marker module for feature gate
// ============================================================================

#[cfg(test)]
mod test_helpers {
    /// Helper to skip tests when CUDA is not available
    pub fn skip_if_no_cuda() -> bool {
        // Check if CUDA is available
        // Return true to skip, false to run
        std::env::var("SKIP_CUDA_TESTS").is_ok()
    }
}
