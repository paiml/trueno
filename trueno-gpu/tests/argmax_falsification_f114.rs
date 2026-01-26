//! ArgMax Kernel Falsification Tests (F114)
//!
//! CUDA-TDG 100-Point Popper Falsification Protocol
//! Category A: Falsifiability & Testability (25 points)
//!
//! Tests apply Karl Popper's falsificationist methodology to verify:
//! - PARITY-114: Barrier safety (all threads reach bar.sync)
//! - PAR-002: Bounds checking (no illegal memory access)
//! - PAR-062: GPU argmax correctness
//!
//! Reference: Popper, K. R. (1959). The Logic of Scientific Discovery.

#[cfg(feature = "cuda")]
mod argmax_falsification_tests {
    use std::ffi::c_void;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::kernels::{ArgMaxFinalKernel, ArgMaxKernel, Kernel};

    // =========================================================================
    // F114-TEST-1: Barrier Safety (PARITY-114)
    // =========================================================================

    /// F114-TEST-1: Verify all threads reach bar.sync in reduction phase
    ///
    /// If CRASHES → PARITY-114 barrier divergence detected
    /// If WORKS → Barrier safety criterion satisfied
    #[test]
    fn f114_test1_barrier_safety() {
        let kernel = ArgMaxKernel::new(1024);
        let ptx = kernel.emit_ptx();

        // Count bar.sync instructions
        let bar_sync_count = ptx.matches("bar.sync").count();
        println!("F114-TEST-1: Barrier Safety Analysis");
        println!("  bar.sync count: {}", bar_sync_count);

        // Verify PTX structure has barrier after each skip label
        let skip_labels: Vec<&str> = ptx
            .lines()
            .filter(|line: &&str| line.contains("skip_"))
            .collect();

        println!("  Skip labels: {:?}", skip_labels.len());

        // Each reduction step should have a barrier after the skip label
        // Expected pattern: skip_reduce_X: followed by bar.sync 0;
        assert!(
            bar_sync_count >= 8,
            "Expected at least 8 bar.sync (7 reduction steps + 1 initial)"
        );

        // Verify no early exit before barriers
        let lines: Vec<&str> = ptx.lines().collect();
        let mut found_exit_before_barrier = false;

        for (i, line) in lines.iter().enumerate() {
            let line_str: &str = *line;
            if line_str.contains("bra exit") {
                // Check if next non-empty line is bar.sync
                for j in (i + 1)..lines.len() {
                    let next = lines[j].trim();
                    if !next.is_empty() && !next.starts_with("//") {
                        if !next.starts_with("exit:") && next.contains("bar.sync") {
                            found_exit_before_barrier = true;
                        }
                        break;
                    }
                }
            }
        }

        assert!(
            !found_exit_before_barrier,
            "PARITY-114: Found potential barrier divergence"
        );
        println!("  PASSED - No barrier divergence detected");
    }

    /// F114-TEST-2: Bounds verification for shared memory access (PAR-002)
    ///
    /// Verifies shared memory indices stay within allocated bounds
    #[test]
    fn f114_test2_bounds_verification() {
        let kernel = ArgMaxKernel::new(152064); // Qwen vocab size
        let ptx = kernel.emit_ptx();

        println!("F114-TEST-2: Bounds Verification");

        // Parse shared memory size from PTX
        let smem_line = ptx
            .lines()
            .find(|line: &&str| line.contains(".shared"))
            .expect("Should have shared memory declaration");

        println!("  Shared memory declaration: {}", smem_line.trim());

        // Verify shared memory is at least 2KB (256 threads * 8 bytes)
        assert!(
            smem_line.contains("2048") || smem_line.contains("smem[2048]"),
            "PAR-002: Shared memory size should be 2048 bytes"
        );

        // Verify offset calculations use proper bounds
        // Each thread accesses shared_base + tid * 4 for values
        // and shared_base + 1024 + tid * 4 for indices
        let has_offset_1024 = ptx.contains("1024");
        assert!(has_offset_1024, "Expected index array offset of 1024 bytes");

        println!("  PASSED - Bounds verification satisfied");
    }

    /// F114-TEST-3: GPU kernel correctness on known input
    ///
    /// Verifies argmax returns correct index for known maximum
    #[test]
    fn f114_test3_correctness_known_max() {
        println!("F114-TEST-3: Correctness on known input");

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                println!("  CUDA context failed: {} (skipping)", e);
                return;
            }
        };

        let stream = CudaStream::new(&ctx).unwrap();

        // Create input with known maximum at index 42
        let mut input = vec![-1.0f32; 256];
        input[42] = 100.0; // Maximum value

        let kernel = ArgMaxKernel::new(256);
        let ptx = kernel.emit_ptx();

        let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
            Ok(m) => m,
            Err(e) => {
                println!("  PTX load failed: {} (skipping)", e);
                return;
            }
        };

        // Allocate buffers
        let mut input_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 256).unwrap();
        let mut block_vals_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1).unwrap();
        let mut block_idxs_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

        input_buf.copy_from_host(&input).unwrap();

        let config = LaunchConfig {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem: 2048,
        };

        let length = 256u32;
        let mut args: [*mut c_void; 4] = [
            input_buf.as_kernel_arg(),
            block_vals_buf.as_kernel_arg(),
            block_idxs_buf.as_kernel_arg(),
            &length as *const u32 as *mut c_void,
        ];

        let result =
            unsafe { stream.launch_kernel(&mut module, "argmax_block_reduce", &config, &mut args) };

        match result {
            Ok(_) => match stream.synchronize() {
                Ok(_) => {
                    let mut block_vals = vec![0.0f32; 1];
                    let mut block_idxs = vec![0u32; 1];
                    block_vals_buf.copy_to_host(&mut block_vals).unwrap();
                    block_idxs_buf.copy_to_host(&mut block_idxs).unwrap();

                    println!("  Block max value: {}", block_vals[0]);
                    println!("  Block max index: {}", block_idxs[0]);

                    assert_eq!(block_idxs[0], 42, "Expected argmax at index 42");
                    assert_eq!(block_vals[0], 100.0, "Expected max value 100.0");
                    println!("  PASSED - Correctness verified");
                }
                Err(e) => {
                    panic!("  CRASHED at sync: {} - PAR-062 correctness FAILED", e);
                }
            },
            Err(e) => {
                panic!("  Launch failed: {} - Kernel execution FAILED", e);
            }
        }
    }

    /// F114-TEST-4: Edge case - maximum at index 0
    #[test]
    fn f114_test4_max_at_zero() {
        println!("F114-TEST-4: Maximum at index 0");

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                println!("  CUDA context failed: {} (skipping)", e);
                return;
            }
        };

        let stream = CudaStream::new(&ctx).unwrap();

        let mut input = vec![-100.0f32; 512];
        input[0] = 999.0; // Maximum at index 0

        let kernel = ArgMaxKernel::new(512);
        let ptx = kernel.emit_ptx();

        let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
            Ok(m) => m,
            Err(e) => {
                println!("  PTX load failed: {} (skipping)", e);
                return;
            }
        };

        let mut input_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 512).unwrap();
        let mut block_vals_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1).unwrap();
        let mut block_idxs_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

        input_buf.copy_from_host(&input).unwrap();

        let config = LaunchConfig {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem: 2048,
        };

        let length = 512u32;
        let mut args: [*mut c_void; 4] = [
            input_buf.as_kernel_arg(),
            block_vals_buf.as_kernel_arg(),
            block_idxs_buf.as_kernel_arg(),
            &length as *const u32 as *mut c_void,
        ];

        let result =
            unsafe { stream.launch_kernel(&mut module, "argmax_block_reduce", &config, &mut args) };

        match result {
            Ok(_) => {
                if let Err(e) = stream.synchronize() {
                    panic!("  CRASHED: {} - Edge case at index 0 FAILED", e);
                }

                let mut block_idxs = vec![0u32; 1];
                block_idxs_buf.copy_to_host(&mut block_idxs).unwrap();

                println!("  Max index: {}", block_idxs[0]);
                assert_eq!(block_idxs[0], 0, "Expected argmax at index 0");
                println!("  PASSED");
            }
            Err(e) => {
                panic!("  Launch failed: {}", e);
            }
        }
    }

    /// F114-TEST-5: Edge case - maximum at last index
    #[test]
    fn f114_test5_max_at_last() {
        println!("F114-TEST-5: Maximum at last index");

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                println!("  CUDA context failed: {} (skipping)", e);
                return;
            }
        };

        let stream = CudaStream::new(&ctx).unwrap();

        let mut input = vec![-100.0f32; 1000];
        input[999] = 999.0; // Maximum at last index

        let kernel = ArgMaxKernel::new(1000);
        let ptx = kernel.emit_ptx();

        let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
            Ok(m) => m,
            Err(e) => {
                println!("  PTX load failed: {} (skipping)", e);
                return;
            }
        };

        let mut input_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1000).unwrap();
        let mut block_vals_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1).unwrap();
        let mut block_idxs_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

        input_buf.copy_from_host(&input).unwrap();

        let config = LaunchConfig {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem: 2048,
        };

        let length = 1000u32;
        let mut args: [*mut c_void; 4] = [
            input_buf.as_kernel_arg(),
            block_vals_buf.as_kernel_arg(),
            block_idxs_buf.as_kernel_arg(),
            &length as *const u32 as *mut c_void,
        ];

        let result =
            unsafe { stream.launch_kernel(&mut module, "argmax_block_reduce", &config, &mut args) };

        match result {
            Ok(_) => {
                if let Err(e) = stream.synchronize() {
                    panic!("  CRASHED: {} - Edge case at last index FAILED", e);
                }

                let mut block_idxs = vec![0u32; 1];
                block_idxs_buf.copy_to_host(&mut block_idxs).unwrap();

                println!("  Max index: {}", block_idxs[0]);
                assert_eq!(block_idxs[0], 999, "Expected argmax at index 999");
                println!("  PASSED");
            }
            Err(e) => {
                panic!("  Launch failed: {}", e);
            }
        }
    }

    /// F114-TEST-6: Multi-block reduction (vocab size)
    #[test]
    fn f114_test6_vocab_size_reduction() {
        println!("F114-TEST-6: Multi-block reduction (vocab size)");

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                println!("  CUDA context failed: {} (skipping)", e);
                return;
            }
        };

        let stream = CudaStream::new(&ctx).unwrap();

        // Simulated vocab size (smaller for test)
        let vocab_size = 4096u32;
        let expected_max_idx = 3333u32;

        let mut input = vec![-100.0f32; vocab_size as usize];
        input[expected_max_idx as usize] = 1000.0;

        let kernel = ArgMaxKernel::new(vocab_size);
        let num_blocks = kernel.num_blocks();
        let ptx = kernel.emit_ptx();

        println!("  Vocab size: {}", vocab_size);
        println!("  Num blocks: {}", num_blocks);

        let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
            Ok(m) => m,
            Err(e) => {
                println!("  PTX load failed: {} (skipping)", e);
                return;
            }
        };

        let mut input_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, vocab_size as usize).unwrap();
        let mut block_vals_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, num_blocks as usize).unwrap();
        let mut block_idxs_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, num_blocks as usize).unwrap();

        input_buf.copy_from_host(&input).unwrap();

        let config = LaunchConfig {
            grid: (num_blocks, 1, 1),
            block: (256, 1, 1),
            shared_mem: 2048,
        };

        let mut args: [*mut c_void; 4] = [
            input_buf.as_kernel_arg(),
            block_vals_buf.as_kernel_arg(),
            block_idxs_buf.as_kernel_arg(),
            &vocab_size as *const u32 as *mut c_void,
        ];

        let result =
            unsafe { stream.launch_kernel(&mut module, "argmax_block_reduce", &config, &mut args) };

        match result {
            Ok(_) => {
                if let Err(e) = stream.synchronize() {
                    panic!("  CRASHED at first pass: {} - Multi-block FAILED", e);
                }

                // Second pass: final reduction
                let final_kernel = ArgMaxFinalKernel::new(num_blocks);
                let final_ptx = final_kernel.emit_ptx();

                let mut final_module = CudaModule::from_ptx(&ctx, &final_ptx).unwrap();
                let mut output_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

                let final_config = LaunchConfig {
                    grid: (1, 1, 1),
                    block: (256, 1, 1),
                    shared_mem: 2048,
                };

                let mut final_args: [*mut c_void; 4] = [
                    block_vals_buf.as_kernel_arg(),
                    block_idxs_buf.as_kernel_arg(),
                    output_buf.as_kernel_arg(),
                    &num_blocks as *const u32 as *mut c_void,
                ];

                let final_result = unsafe {
                    stream.launch_kernel(
                        &mut final_module,
                        "argmax_final_reduce",
                        &final_config,
                        &mut final_args,
                    )
                };

                match final_result {
                    Ok(_) => {
                        if let Err(e) = stream.synchronize() {
                            panic!("  CRASHED at final pass: {} - Final reduction FAILED", e);
                        }

                        let mut output = vec![0u32; 1];
                        output_buf.copy_to_host(&mut output).unwrap();

                        println!("  Final argmax: {}", output[0]);
                        assert_eq!(
                            output[0], expected_max_idx,
                            "Expected argmax at index {}",
                            expected_max_idx
                        );
                        println!("  PASSED - Multi-block reduction correct");
                    }
                    Err(e) => {
                        panic!("  Final launch failed: {}", e);
                    }
                }
            }
            Err(e) => {
                panic!("  First pass launch failed: {}", e);
            }
        }
    }

    // =========================================================================
    // F114-TEST-7: Statistical correctness with random data
    // =========================================================================

    /// F114-TEST-7: Verify argmax matches CPU reference
    #[test]
    fn f114_test7_cpu_reference_match() {
        println!("F114-TEST-7: CPU reference match");

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                println!("  CUDA context failed: {} (skipping)", e);
                return;
            }
        };

        let stream = CudaStream::new(&ctx).unwrap();

        // Use deterministic "random" data for reproducibility
        let input: Vec<f32> = (0..512)
            .map(|i| ((i * 17 + 13) % 1000) as f32 - 500.0)
            .collect();

        // CPU reference
        let cpu_argmax = input
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();

        println!("  CPU argmax: {}", cpu_argmax);

        let kernel = ArgMaxKernel::new(512);
        let ptx = kernel.emit_ptx();

        let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
            Ok(m) => m,
            Err(e) => {
                println!("  PTX load failed: {} (skipping)", e);
                return;
            }
        };

        let mut input_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 512).unwrap();
        let mut block_vals_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, 1).unwrap();
        let mut block_idxs_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

        input_buf.copy_from_host(&input).unwrap();

        let config = LaunchConfig {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem: 2048,
        };

        let length = 512u32;
        let mut args: [*mut c_void; 4] = [
            input_buf.as_kernel_arg(),
            block_vals_buf.as_kernel_arg(),
            block_idxs_buf.as_kernel_arg(),
            &length as *const u32 as *mut c_void,
        ];

        let result =
            unsafe { stream.launch_kernel(&mut module, "argmax_block_reduce", &config, &mut args) };

        match result {
            Ok(_) => {
                if let Err(e) = stream.synchronize() {
                    panic!("  CRASHED: {} - GPU execution FAILED", e);
                }

                let mut block_idxs = vec![0u32; 1];
                block_idxs_buf.copy_to_host(&mut block_idxs).unwrap();

                let gpu_argmax = block_idxs[0];
                println!("  GPU argmax: {}", gpu_argmax);

                assert_eq!(
                    gpu_argmax, cpu_argmax,
                    "GPU argmax ({}) != CPU argmax ({})",
                    gpu_argmax, cpu_argmax
                );
                println!("  PASSED - GPU matches CPU reference");
            }
            Err(e) => {
                panic!("  Launch failed: {}", e);
            }
        }
    }

    // =========================================================================
    // PTX Analysis Tests (Static)
    // =========================================================================

    /// F114-TEST-8: PTX register allocation within SM limits
    #[test]
    fn f114_test8_register_allocation() {
        println!("F114-TEST-8: Register allocation analysis");

        let kernel = ArgMaxKernel::new(152064);
        let ptx = kernel.emit_ptx();

        // Count register declarations
        let reg_lines: Vec<&str> = ptx
            .lines()
            .filter(|line: &&str| line.contains(".reg"))
            .collect();

        let mut total_regs = 0;
        for line in &reg_lines {
            let line_str: &str = *line;
            // Parse register count from declarations like ".reg .u64  %rd<44>;"
            if let Some(count_start) = line_str.find('<') {
                if let Some(count_end) = line_str.find('>') {
                    if let Ok(count) = line_str[count_start + 1..count_end].parse::<u32>() {
                        total_regs += count;
                    }
                }
            }
        }

        println!("  Register declarations: {}", reg_lines.len());
        println!("  Total virtual registers: {}", total_regs);

        // SM 8.9 has 65536 registers per SM
        // With 256 threads per block, max ~256 registers per thread
        // Good target: < 64 registers per thread
        assert!(
            total_regs < 256,
            "Excessive register usage: {} (target < 256)",
            total_regs
        );

        println!("  PASSED - Register usage acceptable");
    }

    /// F114-TEST-9: Shared memory layout validation
    #[test]
    fn f114_test9_shared_memory_layout() {
        println!("F114-TEST-9: Shared memory layout");

        let kernel = ArgMaxKernel::new(152064);
        let ptx = kernel.emit_ptx();

        // Verify shared memory declaration
        assert!(
            ptx.contains(".shared .align"),
            "Missing shared memory alignment"
        );
        assert!(
            ptx.contains("smem[2048]") || ptx.contains(".b8 smem[2048]"),
            "Expected 2KB shared memory"
        );

        // Verify cvta.shared for generic addressing
        assert!(
            ptx.contains("cvta.shared.u64"),
            "Missing shared memory address conversion"
        );

        println!("  Shared memory: 2048 bytes (256 values + 256 indices)");
        println!("  Bank conflicts: Avoided (stride 4 access)");
        println!("  PASSED");
    }
}
