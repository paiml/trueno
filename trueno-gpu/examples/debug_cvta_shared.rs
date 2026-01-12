//! Debug: Check what address cvta.shared returns

#[cfg(feature = "cuda")]
fn main() {
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use std::ffi::c_void;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    let ptx = r#".version 8.0
.target sm_89
.address_size 64

.visible .entry debug_cvta(
    .param .u64 output
) {
    .shared .align 16 .b8 smem[12544];
    .reg .u64 %rd<10>;
    .reg .u32 %r<5>;
    .reg .pred %p0;

    // Get thread ID
    mov.u32 %r0, %tid.x;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra L_done;

    // Only thread 0 runs
    ld.param.u64 %rd0, [output];

    // Get shared memory generic address
    cvta.shared.u64 %rd1, smem;

    // Write it to output[0]
    st.global.u64 [%rd0], %rd1;

    // Also compute smem_base + 0 and write to output[1]
    add.u64 %rd2, %rd1, 0;
    add.u64 %rd3, %rd0, 8;
    st.global.u64 [%rd3], %rd2;

    // And smem_base + 4096 (hash table base) to output[2]
    mov.u64 %rd4, 4096;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd0, 16;
    st.global.u64 [%rd6], %rd5;

L_done:
    ret;
}
"#;

    let mut output_buf: GpuBuffer<u64> = GpuBuffer::new(&ctx, 3).unwrap();
    let init_val = [0xDEADBEEF_u64; 3];
    output_buf.copy_from_host(&init_val).unwrap();

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

    let config = LaunchConfig {
        grid: (1, 1, 1),
        block: (32, 1, 1),
        shared_mem: 0,
    };

    let mut args: [*mut c_void; 1] = [
        output_buf.as_kernel_arg(),
    ];

    unsafe {
        stream.launch_kernel(&mut module, "debug_cvta", &config, &mut args)
            .expect("Kernel launch");
    }
    stream.synchronize().expect("Sync");

    let mut result = [0u64; 3];
    output_buf.copy_to_host(&mut result).unwrap();

    println!("cvta.shared.u64 smem = 0x{:016X}", result[0]);
    println!("smem_base + 0        = 0x{:016X}", result[1]);
    println!("smem_base + 4096     = 0x{:016X}", result[2]);

    if result[0] > 0x1000 {
        println!("SUCCESS: cvta.shared returns valid address");
    } else {
        println!("FAILURE: cvta.shared returned invalid address!");
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled");
}
