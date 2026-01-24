//! Hardware Canary Tests
//!
//! These tests verify that the hardware is correctly detected and available.
//! If these tests fail, it indicates a configuration problem (e.g., RUSTFLAGS not set).
//!
//! ## Purpose
//!
//! These are "canary in the coal mine" tests that will fail loudly if:
//! 1. AVX-512 is not detected on a Threadripper (SIMD Canary)
//! 2. CUDA is not available on an NVIDIA GPU (GPU Canary)
//!
//! ## Running
//!
//! ```bash
//! # With native CPU features (required for AVX-512)
//! RUSTFLAGS="-C target-cpu=native" cargo test --test hardware_canary --all-features
//!
//! # This is what `make coverage` does - always use make coverage
//! ```

// ============================================================================
// SIMD CANARY TESTS
// ============================================================================

/// Canary Test: Verify AVX-512 is detected when using native RUSTFLAGS
///
/// This test panics if AVX-512 is NOT detected, proving that RUSTFLAGS
/// are correctly enabling CPU features.
///
/// **If this test fails:**
/// - Your RUSTFLAGS may not include `-C target-cpu=native`
/// - Your CPU may not support AVX-512 (unlikely on Threadripper)
/// - Run: `RUSTFLAGS="-C target-cpu=native" cargo test`
#[test]
#[cfg(target_arch = "x86_64")]
fn canary_avx512_detected() {
    use std::arch::is_x86_feature_detected;

    // Check AVX-512 Foundation (minimum for AVX-512)
    let avx512f = is_x86_feature_detected!("avx512f");

    if !avx512f {
        // Collect diagnostic information
        let avx2 = is_x86_feature_detected!("avx2");
        let avx = is_x86_feature_detected!("avx");
        let fma = is_x86_feature_detected!("fma");

        panic!(
            "\n\
            ╔══════════════════════════════════════════════════════════════════════════════╗\n\
            ║  SIMD CANARY FAILED: AVX-512 NOT DETECTED!                                   ║\n\
            ╠══════════════════════════════════════════════════════════════════════════════╣\n\
            ║  This Lambda Labs box has a Threadripper with AVX-512 support.               ║\n\
            ║  If this test fails, RUSTFLAGS are not set correctly.                        ║\n\
            ║                                                                              ║\n\
            ║  FIX: Use `make coverage` or set RUSTFLAGS='-C target-cpu=native'            ║\n\
            ║                                                                              ║\n\
            ║  Detected features:                                                          ║\n\
            ║    AVX-512F: {} (MISSING - this is the problem!)                            ║\n\
            ║    AVX2:     {}                                                             ║\n\
            ║    AVX:      {}                                                             ║\n\
            ║    FMA:      {}                                                             ║\n\
            ╚══════════════════════════════════════════════════════════════════════════════╝\n",
            avx512f, avx2, avx, fma
        );
    }

    // Also verify we're using the AVX-512 backend
    let backend = trueno::Backend::select_best();
    assert!(
        matches!(backend, trueno::Backend::AVX512 | trueno::Backend::AVX2),
        "Expected AVX512 or AVX2 backend, got {:?}",
        backend
    );

    println!("✅ SIMD CANARY PASSED: AVX-512 detected and enabled");
}

/// Canary Test: Verify at least AVX2 is detected (fallback for non-AVX512 systems)
#[test]
#[cfg(target_arch = "x86_64")]
fn canary_avx2_minimum() {
    use std::arch::is_x86_feature_detected;

    let avx2 = is_x86_feature_detected!("avx2");
    let fma = is_x86_feature_detected!("fma");

    assert!(
        avx2 && fma,
        "SIMD CANARY FAILED: AVX2+FMA not detected! \
         This is the MINIMUM for modern SIMD. CPU: {:?}, FMA: {:?}",
        avx2,
        fma
    );

    println!("✅ SIMD CANARY PASSED: AVX2+FMA detected");
}

/// Canary Test: Backend selection returns appropriate SIMD level
#[test]
fn canary_backend_selection_not_scalar() {
    let backend = trueno::Backend::select_best();

    // On a Threadripper, we should NEVER fall back to Scalar
    #[cfg(target_arch = "x86_64")]
    {
        assert_ne!(
            backend,
            trueno::Backend::Scalar,
            "SIMD CANARY FAILED: Backend fell back to Scalar on x86_64! \
             This indicates SIMD detection is broken or disabled."
        );
    }

    println!("✅ BACKEND CANARY PASSED: Selected {:?}", backend);
}

// ============================================================================
// GPU CANARY TESTS (CUDA Integration)
// ============================================================================

/// GPU Integration "Hello World": Basic vector copy to/from GPU
///
/// This test verifies CUDA is working by:
/// 1. Uploading a vector to GPU memory
/// 2. Downloading it back to host
/// 3. Verifying the data is unchanged
///
/// **If this test fails:**
/// - CUDA drivers may not be installed
/// - GPU may not be available
/// - `--all-features` may not be set
#[test]
#[cfg(feature = "cuda")]
fn canary_gpu_vector_roundtrip() {
    use trueno_gpu::driver::{CudaContext, CudaStream, GpuBuffer};

    // Try to create CUDA context
    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(e) => {
            // Gracefully skip if CUDA is not available (CI environments, etc.)
            eprintln!(
                "⚠️  GPU CANARY SKIPPED: CUDA context creation failed: {:?}\n\
                 Note: If you expect CUDA to work, check: nvidia-smi",
                e
            );
            return;
        }
    };

    // Create stream for async operations
    let stream = CudaStream::new(&ctx).expect("CUDA stream creation failed");

    // Test data: recognizable pattern to verify integrity
    let test_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.5, -0.5, 3.14159, 2.71828];
    let n = test_data.len();

    // Allocate GPU buffer and upload
    let mut gpu_buffer: GpuBuffer<f32> = GpuBuffer::new(&ctx, n).expect("GPU buffer allocation failed");
    gpu_buffer
        .copy_from_host(&test_data)
        .expect("Host→GPU copy failed");

    // Download back to host
    let mut result = vec![0.0f32; n];
    gpu_buffer
        .copy_to_host(&mut result)
        .expect("GPU→Host copy failed");

    // Synchronize to ensure all operations complete
    stream.synchronize().expect("Stream sync failed");

    // Verify data integrity
    for (i, (&expected, &actual)) in test_data.iter().zip(result.iter()).enumerate() {
        assert!(
            (expected - actual).abs() < 1e-6,
            "GPU CANARY FAILED: Data mismatch at index {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }

    println!(
        "✅ GPU CANARY PASSED: Vector roundtrip successful ({} elements)",
        n
    );
}

/// GPU Canary: Verify CUDA device properties are accessible
#[test]
#[cfg(feature = "cuda")]
fn canary_gpu_device_info() {
    use trueno_gpu::driver::CudaContext;

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => {
            println!("⚠️  GPU CANARY SKIPPED: No CUDA device available");
            return;
        }
    };

    // If we got here, CUDA is working
    println!("✅ GPU CANARY PASSED: CUDA device {} accessible", ctx.device());
}

/// GPU Canary: Basic kernel execution test
#[test]
#[cfg(feature = "cuda")]
fn canary_gpu_kernel_execution() {
    use std::ffi::c_void;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::ptx::{PtxArithmetic, PtxControl, PtxKernel, PtxModule, PtxReg, PtxType};

    let ctx = match CudaContext::new(0) {
        Ok(ctx) => ctx,
        Err(_) => {
            println!("⚠️  GPU CANARY SKIPPED: No CUDA device available");
            return;
        }
    };

    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    // Simple kernel: write thread ID to output
    let kernel = PtxKernel::new("canary_kernel")
        .param(PtxType::U64, "output")
        .build(|ctx| {
            let out_ptr = ctx.load_param_u64("output");
            let tid = ctx.special_reg(PtxReg::TidX);

            // Compute output address: output + tid * 4
            let offset = ctx.mul_u32(tid, 4); // mul_u32 takes (VirtualReg, u32)
            let offset64 = ctx.cvt_u64_u32(offset);
            let addr = ctx.add_u64(out_ptr, offset64);

            // Write tid to output[tid]
            ctx.st_global_u32(addr, tid);
            ctx.ret();
        });

    let ptx = PtxModule::new()
        .version(8, 0)
        .target("sm_70") // Safe minimum for modern GPUs
        .address_size(64)
        .add_kernel(kernel)
        .emit();

    // Compile and load module
    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX compilation failed");

    // Allocate output buffer
    let mut output: GpuBuffer<u32> = GpuBuffer::new(&ctx, 32).expect("Buffer allocation");
    output.copy_from_host(&vec![0u32; 32]).expect("Buffer init");

    // Launch kernel with 32 threads
    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 0,
    };

    let mut args: [*mut c_void; 1] = [output.as_kernel_arg()];

    unsafe {
        stream
            .launch_kernel(&mut module, "canary_kernel", &config, &mut args)
            .expect("Kernel launch failed");
    }

    stream.synchronize().expect("Sync failed");

    // Verify output
    let mut result = vec![0u32; 32];
    output.copy_to_host(&mut result).expect("Copy to host");

    for (i, &val) in result.iter().enumerate() {
        assert_eq!(
            val, i as u32,
            "GPU CANARY FAILED: Kernel output mismatch at {}: expected {}, got {}",
            i, i, val
        );
    }

    println!("✅ GPU CANARY PASSED: PTX kernel execution successful");
}

// ============================================================================
// NON-CUDA STUBS (when CUDA feature is disabled)
// ============================================================================

#[cfg(not(feature = "cuda"))]
mod cuda_stubs {
    #[test]
    fn canary_gpu_vector_roundtrip() {
        println!("⚠️  GPU CANARY SKIPPED: CUDA feature not enabled");
        println!("   Run with: cargo test --test hardware_canary --all-features");
    }

    #[test]
    fn canary_gpu_device_info() {
        println!("⚠️  GPU CANARY SKIPPED: CUDA feature not enabled");
    }

    #[test]
    fn canary_gpu_kernel_execution() {
        println!("⚠️  GPU CANARY SKIPPED: CUDA feature not enabled");
    }
}

// ============================================================================
// HARDWARE REPORT
// ============================================================================

/// Generate a hardware capability report for debugging
#[test]
fn hardware_capability_report() {
    println!("\n═══════════════════════════════════════════════════════════");
    println!("  HARDWARE CAPABILITY REPORT");
    println!("═══════════════════════════════════════════════════════════");

    // CPU Architecture
    println!("\n📦 CPU Architecture:");
    #[cfg(target_arch = "x86_64")]
    {
        println!("   Arch: x86_64");
        use std::arch::is_x86_feature_detected;
        println!("   SSE2:    {}", is_x86_feature_detected!("sse2"));
        println!("   AVX:     {}", is_x86_feature_detected!("avx"));
        println!("   AVX2:    {}", is_x86_feature_detected!("avx2"));
        println!("   FMA:     {}", is_x86_feature_detected!("fma"));
        println!("   AVX-512F: {}", is_x86_feature_detected!("avx512f"));
    }
    #[cfg(target_arch = "aarch64")]
    println!("   Arch: aarch64 (ARM64 NEON)");
    #[cfg(target_arch = "wasm32")]
    println!("   Arch: wasm32");

    // Backend selection
    let backend = trueno::Backend::select_best();
    println!("\n🔧 Selected Backend: {:?}", backend);

    // GPU status
    println!("\n🖥️  GPU Status:");
    #[cfg(feature = "cuda")]
    {
        match trueno_gpu::driver::CudaContext::new(0) {
            Ok(ctx) => println!("   CUDA: Available (device {})", ctx.device()),
            Err(e) => println!("   CUDA: Not available ({:?})", e),
        }
    }
    #[cfg(not(feature = "cuda"))]
    println!("   CUDA: Feature not enabled");

    println!("\n═══════════════════════════════════════════════════════════\n");
}
