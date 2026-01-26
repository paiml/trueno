//! Test LZ4 compression with a single non-zero page

#[cfg(feature = "cuda")]
fn main() {
    use std::ffi::c_void;
    use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
    use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

    const PAGE_SIZE: u32 = 4096;
    const NUM_PAGES: u32 = 1; // Just ONE page

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("CUDA stream");

    // Create non-zero sequential data (same pattern as fkr_101)
    let mut input: Vec<u8> = Vec::with_capacity((NUM_PAGES * PAGE_SIZE) as usize);
    for page_idx in 0..NUM_PAGES {
        for byte_idx in 0..PAGE_SIZE {
            input.push(((page_idx * 17 + byte_idx) % 256) as u8);
        }
    }

    println!("Input first 16 bytes: {:02x?}", &input[0..16]);

    let mut input_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, input.len()).unwrap();
    let mut output_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, (NUM_PAGES * 4352) as usize).unwrap();
    let mut sizes_buf: GpuBuffer<u32> = GpuBuffer::new(&ctx, NUM_PAGES as usize).unwrap();

    input_buf.copy_from_host(&input).unwrap();

    let kernel = Lz4WarpCompressKernel::new(NUM_PAGES);
    let ptx = kernel.emit_ptx();

    println!(
        "Grid: {:?}, Block: {:?}",
        kernel.grid_dim(),
        kernel.block_dim()
    );

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("PTX load");

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

    println!("Launching kernel...");
    unsafe {
        stream
            .launch_kernel(&mut module, "lz4_compress_warp", &config, &mut args)
            .expect("Kernel launch");
    }

    println!("Synchronizing...");
    stream.synchronize().expect("Sync should not crash");

    let mut sizes = vec![0u32; NUM_PAGES as usize];
    sizes_buf.copy_to_host(&mut sizes).unwrap();

    println!("Output sizes: {:?}", sizes);

    for (i, &size) in sizes.iter().enumerate() {
        assert!(
            size > 0 && size <= 4352,
            "Page {} should have valid size in (0, 4352], got {}",
            i,
            size
        );
    }

    println!("SUCCESS! Single page compression works.");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled");
}
