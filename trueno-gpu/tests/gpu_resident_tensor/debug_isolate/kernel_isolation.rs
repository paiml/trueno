//! Debug test to isolate which kernel in batched_multihead_attention fails

/// Debug test to isolate which kernel in batched_multihead_attention fails
#[test]
#[cfg(feature = "cuda")]
fn test_debug_isolate_crash() {
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
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
    println!("  Upload succeeded");

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
                println!("  Kernel launch FAILED: {:?}", e);
                return;
            }
        }
    }

    match stream.synchronize() {
        Ok(_) => println!("  TransposeKernel succeeded!"),
        Err(e) => {
            println!("  TransposeKernel CRASHED: {:?}", e);
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
                println!("  Kernel launch FAILED: {:?}", e);
                return;
            }
        }
    }

    match stream.synchronize() {
        Ok(_) => println!("  GemmKernel succeeded!"),
        Err(e) => {
            println!("  GemmKernel CRASHED: {:?}", e);
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
                println!("  Scale kernel launch FAILED: {:?}", e);
                return;
            }
        }
    }

    match stream.synchronize() {
        Ok(_) => println!("  Scale kernel succeeded!"),
        Err(e) => {
            println!("  Scale kernel CRASHED: {:?}", e);
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
                println!("  Softmax kernel launch FAILED: {:?}", e);
                return;
            }
        }
    }

    match stream.synchronize() {
        Ok(_) => println!("  Softmax kernel succeeded!"),
        Err(e) => {
            println!("  Softmax kernel CRASHED: {:?}", e);
            return;
        }
    }

    println!("\n=== All kernels passed! ===");
}
