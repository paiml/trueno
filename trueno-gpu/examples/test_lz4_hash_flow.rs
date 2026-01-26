//! Popperian Falsification: Exact LZ4 Hash Flow
//!
//! Mimics the EXACT data flow of LZ4 kernel:
//! 1. Load data from shared memory (page data region)
//! 2. Compute hash from loaded data
//! 3. Store to hash table at computed index
//!
//! Run: cargo run -p trueno-gpu --example test_lz4_hash_flow --features cuda

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};

const PAGE_SIZE: u32 = 4096;
const HASH_TABLE_SIZE: u32 = 8192;
const WARP_SMEM_SIZE: u32 = PAGE_SIZE + HASH_TABLE_SIZE + 256; // 12544 bytes

#[cfg(feature = "cuda")]
fn main() {
    use std::ffi::c_void;

    println!("=== Popperian Falsification: Exact LZ4 Hash Flow ===");
    println!();

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = CudaStream::new(&ctx).expect("Failed to create stream");

    let ptx = generate_lz4_hash_flow_ptx();
    println!("PTX generated ({} bytes)", ptx.len());

    // Print full PTX for debugging
    println!("\n=== Full PTX ===");
    println!("{}", ptx);
    println!("=== End PTX ===\n");

    // Allocate output buffer
    let mut output_buf: GpuBuffer<u32> =
        GpuBuffer::new(&ctx, 1).expect("Failed to allocate output buffer");

    let init_val = [0xBAD_BADu32];
    output_buf
        .copy_from_host(&init_val)
        .expect("Failed to init output");

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Failed to load PTX");
    println!("Module loaded successfully");

    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (32, 1, 1), // Single warp
        shared_mem: 0,     // Static shared memory
    };

    let mut args: [*mut c_void; 1] = [output_buf.as_kernel_arg()];

    println!("Launching kernel...");
    unsafe {
        stream
            .launch_kernel(&mut module, "lz4_hash_flow_test", &config, &mut args)
            .expect("Kernel launch failed");
    }

    println!("Synchronizing...");
    stream.synchronize().expect("Stream sync failed");

    let mut result = [0u32; 1];
    output_buf
        .copy_to_host(&mut result)
        .expect("Failed to copy result");

    println!();
    println!("=== RESULT ===");
    println!("Output value: 0x{:08X}", result[0]);

    if result[0] == 0x600D_600D {
        println!("SUCCESS: LZ4-like hash flow works!");
    } else if result[0] == 0xBAD_BAD {
        println!("FAILURE: Kernel did not execute");
    } else {
        println!("PARTIAL: Got 0x{:08X}", result[0]);
    }
}

#[cfg(feature = "cuda")]
fn generate_lz4_hash_flow_ptx() -> String {
    // LZ4 hash constants
    let lz4_prime: u32 = 0x9E37_79B1;
    let hash_shift: u32 = 21; // >> 21 gives 11-bit index (0-2047)
    let hash_mask: u32 = 2047;

    format!(
        r#".version 8.0
.target sm_89
.address_size 64

.visible .entry lz4_hash_flow_test(
    .param .u64 output
)
{{
    .shared .align 16 .b8 smem[{warp_smem_size}];
    .reg .u64 %rd<30>;
    .reg .u32 %r<30>;
    .reg .pred %p<5>;

    // Only lane 0 runs
    mov.u32 %r0, %tid.x;
    and.b32 %r1, %r0, 31;
    setp.eq.u32 %p1, %r1, 0;
    @!%p1 bra L_done;

    // Get shared memory base
    cvta.shared.u64 %rd1, smem;

    // === STEP 1: Initialize some page data ===
    // Write test pattern at position 0: 0x12345678
    mov.u32 %r2, 0x12345678;
    st.u32 [%rd1], %r2;

    // === STEP 2: Load from page data (like LZ4 kernel) ===
    // in_pos = 0
    mov.u32 %r3, 0;
    cvt.u64.u32 %rd2, %r3;
    add.u64 %rd3, %rd1, %rd2;      // curr_addr = smem_base + in_pos
    ld.u32 %r4, [%rd3];            // curr_val = ld[curr_addr]

    // === STEP 3: Compute hash (exactly like LZ4) ===
    // hash_tmp = curr_val * LZ4_PRIME
    mov.u32 %r5, {lz4_prime};
    mul.lo.u32 %r6, %r4, %r5;      // hash_tmp

    // hash_shifted = hash_tmp >> hash_shift
    mov.u32 %r7, {hash_shift};
    shr.b32 %r8, %r6, %r7;         // hash_shifted

    // hash_idx = hash_shifted & hash_mask
    mov.u32 %r9, {hash_mask};
    and.b32 %r10, %r8, %r9;        // hash_idx

    // === STEP 4: Compute hash table address ===
    // hash_table_base = smem_base + PAGE_SIZE
    mov.u64 %rd4, {page_size};
    add.u64 %rd5, %rd1, %rd4;      // hash_table_base

    // hash_entry_off = hash_idx * 4
    mul.lo.u32 %r11, %r10, 4;
    cvt.u64.u32 %rd6, %r11;

    // hash_entry_addr = hash_table_base + hash_entry_off
    add.u64 %rd7, %rd5, %rd6;

    // === STEP 5: Store in_pos to hash table ===
    st.u32 [%rd7], %r3;            // Store in_pos (0) to hash table

    // === STEP 6: Load back to verify ===
    ld.u32 %r12, [%rd7];

    // === STEP 7: Write success marker ===
    ld.param.u64 %rd10, [output];
    mov.u32 %r13, 0x600D600D;
    // Only write success if loaded value matches stored value
    setp.eq.u32 %p2, %r12, %r3;
    @%p2 st.global.u32 [%rd10], %r13;

L_done:
    ret;
}}
"#,
        warp_smem_size = WARP_SMEM_SIZE,
        lz4_prime = lz4_prime,
        hash_shift = hash_shift,
        hash_mask = hash_mask,
        page_size = PAGE_SIZE
    )
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled");
}
