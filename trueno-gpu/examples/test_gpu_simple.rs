//! Simple GPU LZ4 test with non-zero pages

use trueno_gpu::kernels::lz4::PAGE_SIZE;
use trueno_gpu::kernels::{Kernel, Lz4WarpCompressKernel};

use std::ffi::c_void;
#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};

#[cfg(feature = "cuda")]
fn main() {
    println!("Testing GPU LZ4 kernel with non-zero pages...");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = CudaStream::new(&ctx).expect("Failed to create stream");

    const NUM_PAGES: u32 = 3; // Just one block (3 warps)

    // Non-zero pages: sequential pattern (highly compressible)
    let mut input_flat: Vec<u8> = Vec::with_capacity((NUM_PAGES * PAGE_SIZE) as usize);
    for page_idx in 0..NUM_PAGES {
        for byte_idx in 0..PAGE_SIZE {
            // Sequential bytes 0-255 repeating
            input_flat.push(((page_idx * 17 + byte_idx) % 256) as u8);
        }
    }
    println!(
        "Input: {} bytes, first 16: {:?}",
        input_flat.len(),
        &input_flat[0..16]
    );

    // Allocate GPU buffers (using OUTPUT_STRIDE = 4352 bytes per page)
    let mut input_buf: GpuBuffer<u8> =
        GpuBuffer::new(&ctx, input_flat.len()).expect("Failed to allocate input buffer");
    let mut output_buf: GpuBuffer<u8> = GpuBuffer::new(&ctx, (NUM_PAGES * 4352) as usize)
        .expect("Failed to allocate output buffer");
    let mut sizes_buf: GpuBuffer<u32> =
        GpuBuffer::new(&ctx, NUM_PAGES as usize).expect("Failed to allocate sizes buffer");

    input_buf
        .copy_from_host(&input_flat)
        .expect("Failed to copy input");

    let kernel = Lz4WarpCompressKernel::new(NUM_PAGES);
    let ptx = kernel.emit_ptx();
    println!("PTX generated ({} bytes)", ptx.len());
    // Dump relevant st.u32 instructions
    for line in ptx.lines() {
        if line.contains("st.u32") && !line.contains(".global") {
            println!("  {}", line.trim());
        }
    }

    let mut module = CudaModule::from_ptx(&ctx, &ptx).expect("Failed to load PTX");
    println!("Module loaded");

    let grid = kernel.grid_dim();
    let block = kernel.block_dim();
    println!("Grid: {:?}, Block: {:?}", grid, block);

    let config = LaunchConfig {
        grid,
        block,
        shared_mem: 0,
    };

    let num_pages_u32 = NUM_PAGES as u32;
    let mut args: [*mut c_void; 4] = [
        input_buf.as_kernel_arg(),
        output_buf.as_kernel_arg(),
        sizes_buf.as_kernel_arg(),
        &num_pages_u32 as *const u32 as *mut c_void,
    ];

    println!("Launching kernel...");
    unsafe {
        stream
            .launch_kernel(&mut module, "lz4_compress_warp", &config, &mut args)
            .expect("Kernel launch failed");
    }

    println!("Synchronizing...");
    stream.synchronize().expect("Stream sync failed");

    println!("Copying results...");
    let mut sizes_host = vec![0u32; NUM_PAGES as usize];
    sizes_buf
        .copy_to_host(&mut sizes_host)
        .expect("Failed to copy sizes");

    println!("Results:");
    for (i, size) in sizes_host.iter().enumerate() {
        println!("  Page {}: size = {}", i, size);
    }

    println!("\nSuccess!");
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("CUDA feature not enabled");
}
