//! Popperian Falsification Test: GPU Hash Store
//!
//! H0 (null): Store to computed shared memory address works
//! H1 (alt): Store crashes due to address computation bug
//!
//! Run: cargo run -p trueno-gpu --example test_hash_store --features cuda

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};

#[cfg(feature = "cuda")]
fn main() {
    use std::ffi::c_void;

    println!("=== Popperian Falsification: GPU Hash Store ===");
    println!("H0: Store to computed smem address works");
    println!("H1: Store crashes (CUDA_ERROR_UNKNOWN)");
    println!();

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = CudaStream::new(&ctx).expect("Failed to create stream");

    // Generate PTX for minimal hash store test kernel
    let ptx = generate_minimal_hash_store_ptx();
    println!("PTX generated ({} bytes)", ptx.len());

    // Print key instructions
    println!("\nKey PTX instructions:");
    for line in ptx.lines() {
        if line.contains("cvta.shared")
            || line.contains("mul.lo")
            || line.contains("cvt.u64.u32")
            || line.contains("st.u32")
        {
            println!("  {}", line.trim());
        }
    }

    // Allocate output buffer for result
    let mut output_buf: GpuBuffer<u32> =
        GpuBuffer::new(&ctx, 1).expect("Failed to allocate output buffer");

    // Initialize to sentinel value
    let init_val = [0xBAD_BADu32];
    output_buf
        .copy_from_host(&init_val)
        .expect("Failed to init output");

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Failed to load PTX");
    println!("\nModule loaded successfully");

    // Launch with 1 block, 32 threads (single warp)
    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 12288, // PAGE_SIZE + HASH_TABLE_SIZE
    };

    let mut args: [*mut c_void; 1] = [output_buf.as_kernel_arg()];

    println!("Launching kernel (1 block, 32 threads)...");
    unsafe {
        stream
            .launch_kernel(&mut module, "hash_store_test", &config, &mut args)
            .expect("Kernel launch failed");
    }

    println!("Synchronizing...");
    stream.synchronize().expect("Stream sync failed");

    // Read result
    let mut result = [0u32; 1];
    output_buf
        .copy_to_host(&mut result)
        .expect("Failed to copy result");

    println!();
    println!("=== RESULT ===");
    println!("Output value: 0x{:08X}", result[0]);

    if result[0] == 0xDEAD_0002 {
        println!("SUCCESS: H0 confirmed - store to computed address works!");
        println!("The value 0xDEAD0002 was stored and loaded back correctly.");
    } else if result[0] == 0xBAD_BAD {
        println!("FAILURE: Kernel did not execute (output unchanged from sentinel)");
    } else {
        println!("UNEXPECTED: Got 0x{:08X}, expected 0xDEAD0002", result[0]);
    }
}

#[cfg(feature = "cuda")]
fn generate_minimal_hash_store_ptx() -> String {
    // Hand-crafted minimal PTX to isolate the hash store behavior
    r#".version 8.0
.target sm_89
.address_size 64

.visible .entry hash_store_test(
    .param .u64 output
)
{
    .shared .align 16 .b8 smem[12288];
    .reg .u64 %rd<20>;
    .reg .u32 %r<20>;
    .reg .pred %p<5>;

    // Get lane ID = threadIdx.x % 32
    mov.u32 %r0, %tid.x;
    and.b32 %r1, %r0, 31;

    // Only lane 0 does the test
    setp.eq.u32 %p1, %r1, 0;
    @!%p1 bra L_done;

    // Get shared memory base (generic address)
    cvta.shared.u64 %rd1, smem;

    // === TEST 1: Store to fixed offset PAGE_SIZE (4096) ===
    mov.u64 %rd2, 4096;
    add.u64 %rd3, %rd1, %rd2;
    mov.u32 %r4, 0xDEAD0001;
    st.u32 [%rd3], %r4;

    // === TEST 2: Store to computed offset ===
    // hash_idx = 100 (arbitrary valid index)
    mov.u32 %r5, 100;
    // hash_entry_off = hash_idx * 4 = 400
    mul.lo.u32 %r6, %r5, 4;
    // Convert to u64
    cvt.u64.u32 %rd4, %r6;
    // hash_table_base = smem_base + PAGE_SIZE
    mov.u64 %rd5, 4096;
    add.u64 %rd6, %rd1, %rd5;
    // hash_entry_addr = hash_table_base + hash_entry_off
    add.u64 %rd7, %rd6, %rd4;
    // Store test value
    mov.u32 %r7, 0xDEAD0002;
    st.u32 [%rd7], %r7;

    // === TEST 3: Load back and verify ===
    ld.u32 %r8, [%rd7];

    // Write to global output
    ld.param.u64 %rd10, [output];
    st.global.u32 [%rd10], %r8;

L_done:
    ret;
}
"#
    .to_string()
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled. Run with: cargo run -p trueno-gpu --example test_hash_store --features cuda");
}
