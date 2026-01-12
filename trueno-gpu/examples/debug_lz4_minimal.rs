//! Minimal reproduction of LZ4 compress loop crash

#[cfg(feature = "cuda")]
fn main() {
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use std::ffi::c_void;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    // Mimics exactly what LZ4 kernel does in compress loop
    let ptx = r#".version 8.0
.target sm_89
.address_size 64

.visible .entry debug_lz4_minimal(
    .param .u64 output
) {
    // Same shared memory size as LZ4 kernel (3 warps * 12544 = 37632)
    .shared .align 16 .b8 smem[37632];
    .reg .u64 %rd<20>;
    .reg .u32 %r<20>;
    .reg .pred %p<5>;

    // Get thread and warp IDs (same as LZ4)
    mov.u32 %r0, %tid.x;
    mov.u32 %r1, 5;
    shr.b32 %r2, %r0, %r1;      // warp_id = tid.x >> 5
    mov.u32 %r3, 31;
    and.b32 %r4, %r0, %r3;      // lane_id = tid.x & 31

    // Compute smem_base for this warp (same as LZ4)
    mul.lo.u32 %r5, %r2, 12544;         // warp_offset = warp_id * 12544
    cvt.u64.u32 %rd0, %r5;
    cvta.shared.u64 %rd1, smem;         // smem generic address
    add.u64 %rd2, %rd1, %rd0;           // smem_base = smem + warp_offset

    // Load output param
    ld.param.u64 %rd10, [output];

    // Write smem_base to output[0] for verification
    setp.eq.u32 %p0, %r4, 0;
    @!%p0 bra L_not_leader;

    // Leader thread only
    st.global.u64 [%rd10], %rd2;        // output[0] = smem_base

    // Now simulate the compress loop:
    // 1. Store initial in_pos (0) to state at smem_base + 12420
    mov.u32 %r6, 12420;
    cvt.u64.u32 %rd3, %r6;
    add.u64 %rd4, %rd2, %rd3;           // state_base = smem_base + 12420
    mov.u32 %r7, 0;
    st.u32 [%rd4], %r7;                 // *state_base = 0

    // Write state_base to output[1]
    add.u64 %rd11, %rd10, 8;
    st.global.u64 [%rd11], %rd4;

    // 2. Load in_pos back (like at start of compress loop)
    ld.u32 %r8, [%rd4];                 // in_pos = *state_base

    // Write loaded in_pos to output[2]
    add.u64 %rd12, %rd10, 16;
    cvt.u64.u32 %rd5, %r8;
    st.global.u64 [%rd12], %rd5;

    // 3. Compute address: smem_base + in_pos (this is where LZ4 crashes)
    cvt.u64.u32 %rd5, %r8;
    add.u64 %rd6, %rd2, %rd5;           // curr_addr = smem_base + in_pos

    // Write curr_addr to output[3]
    add.u64 %rd13, %rd10, 24;
    st.global.u64 [%rd13], %rd6;

    // 4. Try to load from curr_addr (THE CRASHING INSTRUCTION IN LZ4)
    ld.u32 %r9, [%rd6];                 // curr_val = ld[curr_addr]

    // If we get here, write success marker
    add.u64 %rd14, %rd10, 32;
    mov.u32 %r10, 0x600D600D;
    st.global.u32 [%rd14], %r10;

L_not_leader:
    ret;
}
"#;

    let mut output_buf: GpuBuffer<u64> = GpuBuffer::new(&ctx, 8).unwrap();
    let init_val = [0xDEADBEEF_u64; 8];
    output_buf.copy_from_host(&init_val).unwrap();

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (96, 1, 1),  // 3 warps like LZ4
        shared_mem: 0,
    };

    let mut args: [*mut c_void; 1] = [
        output_buf.as_kernel_arg(),
    ];

    println!("Launching minimal LZ4 reproduction kernel...");
    unsafe {
        stream.launch_kernel(&mut module, "debug_lz4_minimal", &config, &mut args)
            .expect("Kernel launch");
    }

    println!("Synchronizing...");
    stream.synchronize().expect("Sync");

    let mut result = [0u64; 8];
    output_buf.copy_to_host(&mut result).unwrap();

    println!("smem_base     = 0x{:016X}", result[0]);
    println!("state_base    = 0x{:016X}", result[1]);
    println!("in_pos        = 0x{:016X}", result[2]);
    println!("curr_addr     = 0x{:016X}", result[3]);
    println!("success_marker= 0x{:08X}", result[4] as u32);

    if result[4] as u32 == 0x600D600D {
        println!("\nSUCCESS: Minimal LZ4 flow works!");
    } else {
        println!("\nFAILURE: Minimal LZ4 flow crashed");
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled");
}
