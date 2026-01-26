//! F082 Falsification Tests
//!
//! These tests apply Popperian falsification to the F082 hypothesis:
//! "Computed Address From Loaded Value causes crash when ld.shared value
//! is used to compute address for st.global"
//!
//! Alternative Hypothesis: The bug is in 32→64 bit conversion during
//! address computation, not the dependency chain itself.

#[cfg(feature = "cuda")]
mod f082_falsification_tests {
    use std::ffi::c_void;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};

    /// F082-TEST-1: Global→Global computed address
    ///
    /// If this CRASHES → F082 is NOT shared-memory-specific
    /// If this WORKS → Confirms cross-address-space (shared→global) is key
    #[test]
    fn f082_test1_global_to_global_computed_addr() {
        let ptx = r#".version 8.0
.target sm_89
.address_size 64

.visible .entry f082_test1(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 offset_ptr
) {
    .reg .u64 %rd<20>;
    .reg .u32 %r<10>;

    // Load params
    ld.param.u64 %rd0, [input_ptr];
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u64 %rd2, [offset_ptr];

    // Load offset from GLOBAL memory (not shared)
    ld.global.u32 %r0, [%rd2];

    // Convert to 64-bit for address computation
    cvt.u64.u32 %rd3, %r0;

    // Compute target address from loaded value
    add.u64 %rd4, %rd1, %rd3;

    // Load data from input
    ld.global.u32 %r1, [%rd0];

    // Store to COMPUTED address (F082 pattern but global->global)
    st.global.u32 [%rd4], %r1;

    ret;
}"#;

        println!("F082-TEST-1: Global→Global computed address");

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                println!("  CUDA context failed: {} (skipping)", e);
                return;
            }
        };

        let stream = CudaStream::new(&ctx).unwrap();

        // Allocate buffers
        let mut input_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();
        let mut output_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 256).unwrap();
        let mut offset_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

        // Set offset to 16 (bytes) = index 4
        offset_buf.copy_from_host(&[16u32]).unwrap();
        input_buf.copy_from_host(&[0xDEADBEEF_u32]).unwrap();

        let mut module = match CudaModule::from_ptx(&ctx, ptx) {
            Ok(m) => m,
            Err(e) => {
                println!("  PTX load failed: {} (skipping)", e);
                return;
            }
        };

        let config = LaunchConfig {
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_mem: 0,
        };

        let mut args: [*mut c_void; 3] = [
            input_buf.as_kernel_arg(),
            output_buf.as_kernel_arg(),
            offset_buf.as_kernel_arg(),
        ];

        let result = unsafe { stream.launch_kernel(&mut module, "f082_test1", &config, &mut args) };

        match result {
            Ok(_) => match stream.synchronize() {
                Ok(_) => {
                    let mut output = vec![0u32; 256];
                    output_buf.copy_to_host(&mut output).unwrap();

                    if output[4] == 0xDEADBEEF {
                        println!("  PASSED - Global→Global computed address WORKS");
                        println!("  → F082 is SHARED-MEMORY-SPECIFIC");
                    } else {
                        println!("  Data mismatch - got {:08X} at index 4", output[4]);
                    }
                }
                Err(e) => {
                    println!("  CRASHED at sync: {}", e);
                    println!("  → F082 is NOT shared-memory-specific (REFUTES hypothesis)");
                }
            },
            Err(e) => {
                println!("  Launch failed: {}", e);
            }
        }
    }

    /// F082-TEST-2: Explicit 64-bit shared load (no conversion)
    ///
    /// If this WORKS → Type conversion (cvt.u64.u32) is the bug
    /// If this CRASHES → Conversion is not the issue
    #[test]
    fn f082_test2_explicit_64bit_load() {
        let ptx = r#".version 8.0
.target sm_89
.address_size 64

.visible .entry f082_test2(
    .param .u64 output_ptr
) {
    .shared .align 8 .b8 smem[64];
    .reg .u64 %rd<20>;
    .reg .u32 %r<10>;
    .reg .pred %p<5>;

    ld.param.u64 %rd0, [output_ptr];

    // Get lane ID
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, 31;
    and.b32 %r2, %r0, %r1;

    // Only lane 0
    setp.eq.u32 %p0, %r2, 0;
    @!%p0 bra L_skip;

    // Get shared memory base
    cvta.shared.u64 %rd1, smem;

    // Store a 64-bit value (16) to shared memory
    mov.u64 %rd2, 16;
    st.shared.u64 [%rd1], %rd2;

    // Load as 64-bit directly (NO conversion needed)
    ld.shared.u64 %rd3, [%rd1];

    // Compute target address (no cvt.u64.u32!)
    add.u64 %rd4, %rd0, %rd3;

    // Store marker
    mov.u32 %r3, 0xCAFEBABE;
    st.global.u32 [%rd4], %r3;

L_skip:
    ret;
}"#;

        println!("F082-TEST-2: Explicit 64-bit load (no conversion)");

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                println!("  CUDA context failed: {} (skipping)", e);
                return;
            }
        };

        let stream = CudaStream::new(&ctx).unwrap();
        let mut output_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 256).unwrap();

        let mut module = match CudaModule::from_ptx(&ctx, ptx) {
            Ok(m) => m,
            Err(e) => {
                println!("  PTX load failed: {} (skipping)", e);
                return;
            }
        };

        let config = LaunchConfig {
            grid: (1, 1, 1),
            block: (32, 1, 1),
            shared_mem: 64,
        };

        let mut args: [*mut c_void; 1] = [output_buf.as_kernel_arg()];

        let result = unsafe { stream.launch_kernel(&mut module, "f082_test2", &config, &mut args) };

        match result {
            Ok(_) => match stream.synchronize() {
                Ok(_) => {
                    let mut output = vec![0u32; 256];
                    output_buf.copy_to_host(&mut output).unwrap();

                    if output[4] == 0xCAFEBABE {
                        println!("  PASSED - 64-bit load WORKS");
                        println!("  → cvt.u64.u32 conversion IS the bug!");
                        println!("  → FIX: Use 64-bit shared memory values for addresses");
                    } else {
                        println!("  Data mismatch - got {:08X} at index 4", output[4]);
                    }
                }
                Err(e) => {
                    println!("  CRASHED at sync: {}", e);
                    println!("  → 64-bit load doesn't help (conversion not the issue)");
                }
            },
            Err(e) => {
                println!("  Launch failed: {}", e);
            }
        }
    }

    /// F082-TEST-3: Control test - the exact F082 pattern (should CRASH)
    ///
    /// This confirms we can reproduce F082 in a minimal kernel
    #[test]
    fn f082_test3_reproduce_f082() {
        let ptx = r#".version 8.0
.target sm_89
.address_size 64

.visible .entry f082_test3(
    .param .u64 output_ptr
) {
    .shared .align 4 .b8 smem[64];
    .reg .u64 %rd<20>;
    .reg .u32 %r<10>;
    .reg .pred %p<5>;

    ld.param.u64 %rd0, [output_ptr];

    // Get lane ID
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, 31;
    and.b32 %r2, %r0, %r1;

    // Only lane 0
    setp.eq.u32 %p0, %r2, 0;
    @!%p0 bra L_skip;

    // Get shared memory base (32-bit offset)
    mov.u32 %r3, 0;

    // Store a 32-bit offset value to shared memory
    mov.u32 %r4, 16;
    st.shared.u32 [%r3], %r4;

    // THE F082 PATTERN:
    // 1. Load 32-bit value from shared memory
    ld.shared.u32 %r5, [%r3];

    // 2. Convert to 64-bit
    cvt.u64.u32 %rd1, %r5;

    // 3. Compute address from loaded value
    add.u64 %rd2, %rd0, %rd1;

    // 4. Store to computed address (THIS SHOULD CRASH)
    mov.u32 %r6, 0xF082DEAD;
    st.global.u32 [%rd2], %r6;

L_skip:
    ret;
}"#;

        println!("F082-TEST-3: Reproduce F082 pattern (expected to CRASH)");

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                println!("  CUDA context failed: {} (skipping)", e);
                return;
            }
        };

        let stream = CudaStream::new(&ctx).unwrap();
        let mut output_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 256).unwrap();

        let mut module = match CudaModule::from_ptx(&ctx, ptx) {
            Ok(m) => m,
            Err(e) => {
                println!("  PTX load failed: {} (skipping)", e);
                return;
            }
        };

        let config = LaunchConfig {
            grid: (1, 1, 1),
            block: (32, 1, 1),
            shared_mem: 64,
        };

        let mut args: [*mut c_void; 1] = [output_buf.as_kernel_arg()];

        let result = unsafe { stream.launch_kernel(&mut module, "f082_test3", &config, &mut args) };

        match result {
            Ok(_) => match stream.synchronize() {
                Ok(_) => {
                    let mut output = vec![0u32; 256];
                    output_buf.copy_to_host(&mut output).unwrap();

                    if output[4] == 0xF082DEAD {
                        println!("  UNEXPECTED: F082 pattern WORKS in minimal kernel!");
                        println!(
                            "  → F082 may be triggered by kernel complexity, not this pattern"
                        );
                    } else {
                        println!("  Data mismatch - got {:08X} at index 4", output[4]);
                    }
                }
                Err(e) => {
                    println!("  CRASHED as expected: {}", e);
                    println!("  → Confirmed: F082 pattern reproduces in minimal kernel");
                }
            },
            Err(e) => {
                println!("  Launch failed: {}", e);
            }
        }
    }

    /// F082-TEST-4: Shared→Shared computed address (no global involved)
    ///
    /// If this CRASHES → Bug is in shared memory address computation
    /// If this WORKS → Bug is specifically in shared→global crossing
    #[test]
    fn f082_test4_shared_to_shared_computed_addr() {
        let ptx = r#".version 8.0
.target sm_89
.address_size 64

.visible .entry f082_test4(
    .param .u64 output_ptr
) {
    .shared .align 4 .b8 smem[1024];
    .reg .u64 %rd<20>;
    .reg .u32 %r<20>;
    .reg .pred %p<5>;

    ld.param.u64 %rd0, [output_ptr];

    // Get lane ID
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, 31;
    and.b32 %r2, %r0, %r1;

    // Only lane 0
    setp.eq.u32 %p0, %r2, 0;
    @!%p0 bra L_skip;

    // Store offset value at smem[0]
    mov.u32 %r3, 0;
    mov.u32 %r4, 64;  // Target: smem[64]
    st.shared.u32 [%r3], %r4;

    // Store data at smem[4]
    mov.u32 %r5, 4;
    mov.u32 %r6, 0x5AAED01;
    st.shared.u32 [%r5], %r6;

    // Load offset from shared memory (32-bit)
    ld.shared.u32 %r7, [%r3];

    // Compute target address IN SHARED MEMORY (not global!)
    // Just use the loaded value directly as 32-bit offset
    // This is shared->shared, NOT shared->global

    // Load data from smem[4]
    ld.shared.u32 %r8, [%r5];

    // Store to COMPUTED shared address
    st.shared.u32 [%r7], %r8;

    // Verify by loading from target and storing to global
    ld.shared.u32 %r9, [%r7];
    st.global.u32 [%rd0], %r9;

L_skip:
    bar.sync 0;
    ret;
}"#;

        println!("F082-TEST-4: Shared→Shared computed address");

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                println!("  CUDA context failed: {} (skipping)", e);
                return;
            }
        };

        let stream = CudaStream::new(&ctx).unwrap();
        let mut output_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 1).unwrap();

        let mut module = match CudaModule::from_ptx(&ctx, ptx) {
            Ok(m) => m,
            Err(e) => {
                println!("  PTX load failed: {} (skipping)", e);
                return;
            }
        };

        let config = LaunchConfig {
            grid: (1, 1, 1),
            block: (32, 1, 1),
            shared_mem: 1024,
        };

        let mut args: [*mut c_void; 1] = [output_buf.as_kernel_arg()];

        let result = unsafe { stream.launch_kernel(&mut module, "f082_test4", &config, &mut args) };

        match result {
            Ok(_) => match stream.synchronize() {
                Ok(_) => {
                    let mut output = vec![0u32; 1];
                    output_buf.copy_to_host(&mut output).unwrap();

                    if output[0] == 0x5AAED01 {
                        println!("  PASSED - Shared→Shared computed address WORKS");
                        println!("  → Bug is in shared→GLOBAL crossing specifically");
                    } else {
                        println!("  Data mismatch - got {:08X}", output[0]);
                    }
                }
                Err(e) => {
                    println!("  CRASHED: {}", e);
                    println!("  → Bug is in shared memory address computation itself");
                }
            },
            Err(e) => {
                println!("  Launch failed: {}", e);
            }
        }
    }

    /// F082-TEST-5: membar.gl between load and store
    ///
    /// Tests if global-level memory barrier prevents the crash
    #[test]
    fn f082_test5_membar_gl() {
        let ptx = r#".version 8.0
.target sm_89
.address_size 64

.visible .entry f082_test5(
    .param .u64 output_ptr
) {
    .shared .align 4 .b8 smem[64];
    .reg .u64 %rd<20>;
    .reg .u32 %r<10>;
    .reg .pred %p<5>;

    ld.param.u64 %rd0, [output_ptr];

    // Get lane ID
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, 31;
    and.b32 %r2, %r0, %r1;

    // Only lane 0
    setp.eq.u32 %p0, %r2, 0;
    @!%p0 bra L_skip;

    // Store offset to shared
    mov.u32 %r3, 0;
    mov.u32 %r4, 16;
    st.shared.u32 [%r3], %r4;

    // CTA barrier
    membar.cta;

    // Load offset from shared
    ld.shared.u32 %r5, [%r3];

    // GLOBAL memory barrier (stronger than CTA)
    membar.gl;

    // Convert to 64-bit
    cvt.u64.u32 %rd1, %r5;

    // Another global barrier after conversion
    membar.gl;

    // Compute address
    add.u64 %rd2, %rd0, %rd1;

    // Final global barrier before store
    membar.gl;

    // Store
    mov.u32 %r6, 0xBABA6001;
    st.global.u32 [%rd2], %r6;

L_skip:
    ret;
}"#;

        println!("F082-TEST-5: membar.gl between operations");

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                println!("  CUDA context failed: {} (skipping)", e);
                return;
            }
        };

        let stream = CudaStream::new(&ctx).unwrap();
        let mut output_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 256).unwrap();

        let mut module = match CudaModule::from_ptx(&ctx, ptx) {
            Ok(m) => m,
            Err(e) => {
                println!("  PTX load failed: {} (skipping)", e);
                return;
            }
        };

        let config = LaunchConfig {
            grid: (1, 1, 1),
            block: (32, 1, 1),
            shared_mem: 64,
        };

        let mut args: [*mut c_void; 1] = [output_buf.as_kernel_arg()];

        let result = unsafe { stream.launch_kernel(&mut module, "f082_test5", &config, &mut args) };

        match result {
            Ok(_) => match stream.synchronize() {
                Ok(_) => {
                    let mut output = vec![0u32; 256];
                    output_buf.copy_to_host(&mut output).unwrap();

                    if output[4] == 0xBABA6001 {
                        println!("  PASSED - membar.gl WORKS");
                        println!("  → REFUTES 'barriers don't work' claim");
                        println!("  → FIX: Use membar.gl (not just membar.cta)");
                    } else {
                        println!("  Data mismatch - got {:08X} at index 4", output[4]);
                    }
                }
                Err(e) => {
                    println!("  CRASHED: {}", e);
                    println!("  → membar.gl doesn't help (supports original F082 hypothesis)");
                }
            },
            Err(e) => {
                println!("  Launch failed: {}", e);
            }
        }
    }
}
