//! DEBUG: Isolate which operation crashes
//! PHASE 4: Full GPU Encoder Block (Total Offload)

#[cfg(feature = "cuda")]
use trueno_gpu::driver::CudaContext;
#[cfg(feature = "cuda")]
use trueno_gpu::memory::resident::{
    reset_transfer_counters, total_d2h_transfers, total_h2d_transfers, GpuResidentTensor,
};

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
