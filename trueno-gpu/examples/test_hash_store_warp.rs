//! Popperian Falsification Test: GPU Hash Store WITH WARP OFFSET
//!
//! This test more closely matches the LZ4 kernel's address computation:
//! - Includes warp_smem_offset calculation
//! - Uses 3 warps like LZ4 kernel
//!
//! Run: cargo run -p trueno-gpu --example test_hash_store_warp --features cuda

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};

const PAGE_SIZE: u32 = 4096;
const HASH_TABLE_SIZE: u32 = 8192;
const WARP_SMEM_SIZE: u32 = PAGE_SIZE + HASH_TABLE_SIZE + 256; // 12544 bytes
const NUM_WARPS: u32 = 3;

#[cfg(feature = "cuda")]
fn main() {
    use std::ffi::c_void;

    println!("=== Popperian Falsification: GPU Hash Store WITH WARP OFFSET ===");
    println!("WARP_SMEM_SIZE = {} bytes", WARP_SMEM_SIZE);
    println!("Total shared mem = {} bytes", WARP_SMEM_SIZE * NUM_WARPS);
    println!();

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = CudaStream::new(&ctx).expect("Failed to create stream");

    let ptx = generate_warp_hash_store_ptx();
    println!("PTX generated ({} bytes)", ptx.len());

    // Print key instructions
    println!("\nKey PTX instructions:");
    for line in ptx.lines() {
        if line.contains("cvta.shared")
            || line.contains("mul.lo")
            || line.contains("cvt.u64.u32")
            || (line.contains("st.u32") && !line.contains(".global"))
            || line.contains("add.u64")
        {
            println!("  {}", line.trim());
        }
    }

    // Allocate output buffer for 3 results (one per warp)
    let mut output_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, 3)
        .expect("Failed to allocate output buffer");

    let init_val = [0xBAD_BADu32; 3];
    output_buf.copy_from_host(&init_val).expect("Failed to init output");

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Failed to load PTX");
    println!("\nModule loaded successfully");

    // Launch with 1 block, 96 threads (3 warps)
    // Note: shared_mem=0 because shared memory is statically declared in PTX
    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (96, 1, 1), // 3 warps
        shared_mem: 0,
    };

    let mut args: [*mut c_void; 1] = [
        output_buf.as_kernel_arg(),
    ];

    println!("Launching kernel (1 block, 96 threads = 3 warps)...");
    unsafe {
        stream.launch_kernel(&mut module, "hash_store_warp_test", &config, &mut args)
            .expect("Kernel launch failed");
    }

    println!("Synchronizing...");
    stream.synchronize().expect("Stream sync failed");

    // Read results
    let mut results = [0u32; 3];
    output_buf.copy_to_host(&mut results).expect("Failed to copy results");

    println!();
    println!("=== RESULTS ===");
    for (i, &result) in results.iter().enumerate() {
        let expected = 0xDEAD0000 + (i as u32 + 1) * 0x100;
        println!("Warp {}: 0x{:08X} (expected 0x{:08X}) - {}",
            i, result, expected,
            if result == expected { "OK" } else { "MISMATCH" });
    }

    let all_ok = results.iter().enumerate().all(|(i, &r)| r == 0xDEAD0000 + (i as u32 + 1) * 0x100);
    if all_ok {
        println!("\nSUCCESS: All warps stored and loaded correctly!");
    } else {
        println!("\nFAILURE: Some warps failed!");
    }
}

#[cfg(feature = "cuda")]
fn generate_warp_hash_store_ptx() -> String {
    format!(r#".version 8.0
.target sm_89
.address_size 64

.visible .entry hash_store_warp_test(
    .param .u64 output
)
{{
    .shared .align 16 .b8 smem[{total_smem}];
    .reg .u64 %rd<30>;
    .reg .u32 %r<30>;
    .reg .pred %p<5>;

    // Get thread ID and compute warp_id, lane_id
    mov.u32 %r0, %tid.x;
    shr.b32 %r1, %r0, 5;           // warp_id = tid / 32
    and.b32 %r2, %r0, 31;          // lane_id = tid % 32

    // Only lane 0 of each warp does the test
    setp.eq.u32 %p1, %r2, 0;
    @!%p1 bra L_done;

    // === Compute warp_smem_offset = warp_id * WARP_SMEM_SIZE ===
    mov.u32 %r3, {warp_smem_size};
    mul.lo.u32 %r4, %r1, %r3;      // warp_smem_offset (u32)
    cvt.u64.u32 %rd1, %r4;         // warp_smem_offset_64

    // Get raw shared memory base
    cvta.shared.u64 %rd2, smem;

    // smem_base = raw_smem_base + warp_smem_offset
    add.u64 %rd3, %rd2, %rd1;

    // === Compute hash_table_base = smem_base + PAGE_SIZE ===
    mov.u64 %rd4, {page_size};
    add.u64 %rd5, %rd3, %rd4;      // hash_table_base

    // === Store to computed offset in hash table ===
    // hash_idx = 100 + warp_id (different per warp)
    mov.u32 %r5, 100;
    add.u32 %r6, %r5, %r1;         // hash_idx = 100 + warp_id

    // hash_entry_off = hash_idx * 4
    mul.lo.u32 %r7, %r6, 4;
    cvt.u64.u32 %rd6, %r7;

    // hash_entry_addr = hash_table_base + hash_entry_off
    add.u64 %rd7, %rd5, %rd6;

    // Store unique value per warp: 0xDEAD0100, 0xDEAD0200, 0xDEAD0300
    mov.u32 %r8, 0xDEAD0000;
    add.u32 %r9, %r1, 1;           // warp_id + 1
    shl.b32 %r10, %r9, 8;          // (warp_id + 1) << 8 = 0x100, 0x200, 0x300
    add.u32 %r11, %r8, %r10;       // 0xDEAD0100, etc.

    st.u32 [%rd7], %r11;           // Store to hash table

    // Load back to verify
    ld.u32 %r12, [%rd7];

    // Write to global output[warp_id]
    ld.param.u64 %rd10, [output];
    cvt.u64.u32 %rd11, %r1;        // warp_id as u64
    shl.b64 %rd12, %rd11, 2;       // warp_id * 4 (bytes)
    add.u64 %rd13, %rd10, %rd12;   // &output[warp_id]
    st.global.u32 [%rd13], %r12;

L_done:
    ret;
}}
"#,
    total_smem = WARP_SMEM_SIZE * NUM_WARPS,
    warp_smem_size = WARP_SMEM_SIZE,
    page_size = PAGE_SIZE)
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled");
}
