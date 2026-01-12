//! FKR-009: Enhanced Sanitizer Debug for LZ4 Compress Loop
//!
//! Uses trueno-gpu's sanitizer wrapper to get semantic error output
//! when debugging the LZ4 kernel crash.

#[cfg(feature = "cuda")]
mod fkr_009_tests {
    use std::ffi::c_void;
    use trueno_gpu::driver::sanitizer::{AddressRegistry, PtxSourceMap, SanitizerParser};
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::kernels::lz4::PAGE_SIZE;
    use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

    fn cuda_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    /// FKR-009a: Run LZ4 kernel with registered buffers and capture enhanced errors
    ///
    /// NOTE: This test is marked `#[ignore]` because it tests the known-buggy
    /// Lz4WarpCompressKernel which has the F082 computed-address bug.
    /// It will always crash with CUDA_ERROR_UNKNOWN (716).
    /// Use Lz4WarpShuffleKernel instead for working LZ4 compression.
    #[test]
    #[ignore = "Uses buggy Lz4WarpCompressKernel - F082 confirmed"]
    fn fkr_009a_lz4_with_semantic_addresses() {
        if !cuda_available() {
            eprintln!("FKR-009a SKIPPED: No CUDA device available");
            return;
        }
        let ctx = CudaContext::new(0).expect("CUDA context");
        let stream = CudaStream::new(&ctx).expect("CUDA stream");

        const NUM_PAGES: u32 = 1;

        // Create test data with pattern that triggers compression
        let mut input: Vec<u8> = vec![0u8; PAGE_SIZE as usize];
        for i in 0..PAGE_SIZE as usize {
            input[i] = (i % 256) as u8;
        }

        // Allocate and REGISTER buffers
        let input_buf: GpuBuffer<u8> = GpuBuffer::from_host(&ctx, &input).unwrap();
        let output_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, 4352).unwrap();
        let sizes_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

        // Register with meaningful names
        input_buf.register_name("lz4_input_page");
        output_buf.register_name("lz4_output_buf");
        sizes_buf.register_name("lz4_sizes");

        // Get the kernel and PTX
        let kernel = Lz4WarpCompressKernel::new(NUM_PAGES);
        let ptx = kernel.emit_ptx();

        // Create PTX source map for context
        let ptx_map = PtxSourceMap::new(&ptx);

        // Print labels in compress loop area
        println!("=== PTX Labels in Compress Loop ===");
        for label in [
            "L_compress_loop",
            "L_check_match",
            "L_no_match",
            "L_emit_remaining",
            "L_done",
        ] {
            if let Some(context) = ptx_map.context_around_label(label, 3) {
                println!("\n--- {} ---\n{}", label, context);
            }
        }

        // Try to load module and launch
        let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
            Ok(m) => m,
            Err(e) => {
                panic!("PTX compilation failed: {:?}", e);
            }
        };

        let config = LaunchConfig {
            grid: kernel.grid_dim(),
            block: kernel.block_dim(),
            shared_mem: 0,
        };

        let num_pages = NUM_PAGES;
        let mut args: [*mut c_void; 4] = [
            input_buf.as_kernel_arg(),
            output_buf.as_kernel_arg(),
            sizes_buf.as_kernel_arg(),
            &num_pages as *const u32 as *mut c_void,
        ];

        println!("\n=== Launching LZ4 kernel ===");
        println!("Grid: {:?}, Block: {:?}", kernel.grid_dim(), kernel.block_dim());

        unsafe {
            stream
                .launch_kernel(&mut module, "lz4_compress_warp", &config, &mut args)
                .expect("Kernel launch");
        }

        match stream.synchronize() {
            Ok(()) => {
                let mut sizes = vec![0u32; 1];
                sizes_buf.copy_to_host(&mut sizes).unwrap();
                println!("\n✅ Kernel succeeded! Compressed size: {} bytes", sizes[0]);
            }
            Err(e) => {
                println!("\n🛑 Kernel failed: {:?}", e);

                // Show registry for context
                if let Ok(registry) = AddressRegistry::global().lock() {
                    println!("\n=== Registered Buffers ===");
                    // We can't iterate directly, but we can check specific addresses
                    println!(
                        "  input_buf @ 0x{:X} ({} bytes)",
                        input_buf.as_ptr(),
                        input_buf.size_bytes()
                    );
                    println!(
                        "  output_buf @ 0x{:X} ({} bytes)",
                        output_buf.as_ptr(),
                        output_buf.size_bytes()
                    );
                    println!(
                        "  sizes_buf @ 0x{:X} ({} bytes)",
                        sizes_buf.as_ptr(),
                        sizes_buf.size_bytes()
                    );

                    // Check if address 0x1 maps to anything (it shouldn't)
                    println!("\n=== Address 0x1 lookup ===");
                    println!("  {}", registry.format_address(0x1));
                }

                panic!("FKR-009a: FALSIFIED - kernel crashed");
            }
        }
    }

    /// FKR-009b: Analyze PTX around the crash point (SASS offset 0x2160)
    #[test]
    fn fkr_009b_analyze_crash_location() {
        let kernel = Lz4WarpCompressKernel::new(1);
        let ptx = kernel.emit_ptx();

        let ptx_map = PtxSourceMap::new(&ptx);

        // The crash is at SASS offset 0x2160, which is in the compress loop
        // Let's analyze all ld.u32 instructions in the compress loop

        println!("=== PTX Analysis: Potential Crash Points ===\n");

        let lines: Vec<&str> = ptx.lines().collect();
        let mut in_compress_loop = false;
        let mut load_count = 0;

        for (i, line) in lines.iter().enumerate() {
            if line.contains("L_compress_loop:") {
                in_compress_loop = true;
                println!(">>> COMPRESS LOOP START at line {}", i + 1);
            }

            if in_compress_loop {
                // Track shared memory loads
                if line.contains("ld.u32") || line.contains("ld.u64") {
                    load_count += 1;
                    println!("  L{}: {} (load #{})", i + 1, line.trim(), load_count);
                }

                // Look for address calculations involving smem_base
                if line.contains("add.u64") && (line.contains("rd11") || line.contains("rd167")) {
                    println!("  L{}: {} [ADDR CALC]", i + 1, line.trim());
                }
            }

            if line.contains("L_emit_remaining:") || line.contains("L_done:") {
                if in_compress_loop {
                    println!(">>> COMPRESS LOOP END at line {}", i + 1);
                    break;
                }
            }
        }

        println!("\nTotal loads in compress loop: {}", load_count);

        // Show the specific PTX around compress_loop label
        if let Some(context) = ptx_map.context_around_label("L_compress_loop", 20) {
            println!("\n=== Detailed Context (L_compress_loop ±20 lines) ===\n{}", context);
        }
    }

    /// FKR-009c: Test sanitizer parser with actual error output
    #[test]
    fn fkr_009c_parse_actual_sanitizer_output() {
        // This is the actual output we get from compute-sanitizer
        let output = r#"
========= COMPUTE-SANITIZER
========= Invalid __shared__ read of size 4 bytes
=========     at lz4_compress_warp+0x2160
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x1 is misaligned
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: [0x381377] in libcuda.so.1
========= ERROR SUMMARY: 1 error
"#;

        let violations = SanitizerParser::parse(output);
        assert_eq!(violations.len(), 1, "Should parse one violation");

        let v = &violations[0];

        println!("=== Parsed Violation ===");
        println!("  Kernel: {}", v.kernel_name);
        println!("  SASS offset: 0x{:X}", v.sass_offset);
        println!("  Thread: {:?}", v.thread);
        println!("  Block: {:?}", v.block);
        println!("  Address: 0x{:X}", v.address);
        println!("  Type: {:?}", v.violation_type);

        // Key insight: Address 0x1 means smem_base is 0 and in_pos is 1
        // smem_base + in_pos = 0 + 1 = 0x1
        assert_eq!(v.address, 0x1, "Address should be 0x1 (smem_base=0 + in_pos=1)");
        assert_eq!(v.kernel_name, "lz4_compress_warp");
        assert_eq!(v.sass_offset, 0x2160);

        println!("\n=== ROOT CAUSE ANALYSIS ===");
        println!("The crash at Address 0x1 indicates:");
        println!("  - smem_base = 0x0 (INVALID - should be generic shared memory address)");
        println!("  - in_pos = 1 (first iteration after in_pos=0)");
        println!("  - Computed address = smem_base + in_pos = 0 + 1 = 0x1");
        println!("");
        println!("This proves smem_base register (%rd11 or %rd167) is being ZEROED");
        println!("somewhere between initialization and first use in compress loop.");
    }

    /// FKR-009d: Dump critical register assignments to verify smem_base flow
    #[test]
    fn fkr_009d_trace_smem_base_register() {
        let kernel = Lz4WarpCompressKernel::new(1);
        let ptx = kernel.emit_ptx();

        println!("=== smem_base Register Flow Analysis ===\n");

        let lines: Vec<&str> = ptx.lines().collect();

        // Find the register that holds smem_base
        println!("1. Looking for smem_base initialization (cvta.shared.u64):");
        for (i, line) in lines.iter().enumerate() {
            if line.contains("cvta.shared.u64") {
                println!("   L{}: {}", i + 1, line.trim());
            }
        }

        println!("\n2. Looking for smem_base storage (before compress loop):");
        for (i, line) in lines.iter().enumerate() {
            if line.contains("L_compress_loop") {
                // Look at the 10 lines before compress_loop
                let start = i.saturating_sub(15);
                for j in start..i {
                    if lines[j].contains("st.") {
                        println!("   L{}: {}", j + 1, lines[j].trim());
                    }
                }
                break;
            }
        }

        println!("\n3. Looking for smem_base reload (at compress loop start):");
        for (i, line) in lines.iter().enumerate() {
            if line.contains("L_compress_loop:") {
                // Show the first 10 lines of compress loop
                for j in i..(i + 15).min(lines.len()) {
                    println!("   L{}: {}", j + 1, lines[j].trim());
                }
                break;
            }
        }

        println!("\n4. Looking for ld.u32 instructions using smem_base computed address:");
        let mut in_loop = false;
        for (i, line) in lines.iter().enumerate() {
            if line.contains("L_compress_loop:") {
                in_loop = true;
            }
            if in_loop && line.contains("ld.u32") && !line.contains("ld.param") {
                println!("   L{}: {}", i + 1, line.trim());
                // Show previous 2 lines for context (the address calculation)
                for j in (i.saturating_sub(3))..i {
                    println!("      <- L{}: {}", j + 1, lines[j].trim());
                }
            }
            if in_loop && line.contains("L_emit_remaining") {
                break;
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
mod fkr_009_tests {
    #[test]
    fn fkr_009_skip_no_cuda() {
        println!("FKR-009: Skipped - CUDA feature not enabled");
    }
}
