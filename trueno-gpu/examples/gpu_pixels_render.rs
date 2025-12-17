//! GPU Pixel Rendering Example - ACTUALLY RUNS ON GPU
//!
//! Renders a color gradient on GPU and displays in terminal.
//!
//! Run with: `cargo run -p trueno-gpu --example gpu_pixels_render --features cuda`

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::ptx::{PtxKernel, PtxModule, PtxType};

/// Width and height of the rendered image
const WIDTH: usize = 80;
const HEIGHT: usize = 30;

/// Build the gradient kernel
fn build_gradient_kernel() -> trueno_gpu::ptx::PtxKernel {
    PtxKernel::new("gradient_pixel")
        .param(PtxType::U64, "output") // Output buffer (f32 per pixel)
        .param(PtxType::U32, "width")
        .param(PtxType::U32, "height")
        .build(|ctx| {
            // Get thread indices
            let tid_x = ctx.special_reg(trueno_gpu::ptx::PtxReg::TidX);
            let tid_y = ctx.special_reg(trueno_gpu::ptx::PtxReg::TidY);
            let bid_x = ctx.special_reg(trueno_gpu::ptx::PtxReg::CtaIdX);
            let bid_y = ctx.special_reg(trueno_gpu::ptx::PtxReg::CtaIdY);
            let bdim_x = ctx.special_reg(trueno_gpu::ptx::PtxReg::NtidX);
            let bdim_y = ctx.special_reg(trueno_gpu::ptx::PtxReg::NtidY);

            // Global pixel coordinates: px = bid_x * bdim_x + tid_x
            let px = ctx.mad_lo_u32(bid_x, bdim_x, tid_x);
            let py = ctx.mad_lo_u32(bid_y, bdim_y, tid_y);

            // Load parameters
            let width = ctx.load_param_u32("width");
            let height = ctx.load_param_u32("height");

            // Bounds check X
            let oob_x = ctx.setp_ge_u32(px, width);
            ctx.branch_if(oob_x, "exit");

            // Bounds check Y
            let oob_y = ctx.setp_ge_u32(py, height);
            ctx.branch_if(oob_y, "exit");

            // Calculate gradient value: (px + py) / (width + height)
            // This creates a diagonal gradient from black to white
            let sum_xy = ctx.add_u32_reg(px, py);
            let sum_wh = ctx.add_u32_reg(width, height);

            // Convert to float for division
            let sum_xy_f = ctx.cvt_f32_u32(sum_xy);
            let sum_wh_f = ctx.cvt_f32_u32(sum_wh);
            let intensity = ctx.div_f32(sum_xy_f, sum_wh_f);

            // Calculate output address: output + (py * width + px) * 4
            let output = ctx.load_param_u64("output");
            let row_offset = ctx.mul_u32_reg(py, width);
            let idx = ctx.add_u32_reg(row_offset, px);
            let offset = ctx.mul_wide_u32(idx, 4); // 4 bytes per f32
            let addr = ctx.add_u64(output, offset);

            // Store intensity value
            ctx.st_global_f32(addr, intensity);

            ctx.label("exit");
            ctx.ret();
        })
}

/// Generate PTX for a simple gradient kernel with proper headers for sm_89 (RTX 4090)
fn gradient_kernel_ptx() -> String {
    let kernel = build_gradient_kernel();

    // Create module with proper headers for RTX 4090 (sm_89, Ada Lovelace)
    PtxModule::new()
        .version(8, 0)
        .target("sm_89")
        .address_size(64)
        .add_kernel(kernel)
        .emit()
}

/// Render pixel buffer to terminal using Unicode block characters
fn render_to_terminal(pixels: &[f32], width: usize, height: usize) {
    // Grayscale characters from dark to light
    let chars = [' ', '░', '▒', '▓', '█'];

    // Header
    println!("\n\x1b[1;36m┌{}┐\x1b[0m", "─".repeat(width));
    println!(
        "\x1b[1;36m│\x1b[0m\x1b[1;33m{:^width$}\x1b[0m\x1b[1;36m│\x1b[0m",
        "GPU RENDERED GRADIENT",
        width = width
    );
    println!("\x1b[1;36m├{}┤\x1b[0m", "─".repeat(width));

    // Render pixels
    for y in 0..height {
        print!("\x1b[1;36m│\x1b[0m");
        for x in 0..width {
            let idx = y * width + x;
            let v = pixels[idx].clamp(0.0, 1.0);

            // Map intensity to character and color
            let char_idx = (v * (chars.len() - 1) as f32) as usize;
            let color_code = 232 + (v * 23.0) as u8; // Grayscale ANSI codes

            print!("\x1b[38;5;{}m{}\x1b[0m", color_code, chars[char_idx]);
        }
        println!("\x1b[1;36m│\x1b[0m");
    }

    println!("\x1b[1;36m└{}┘\x1b[0m", "─".repeat(width));
}

fn main() {
    println!("\n\x1b[1;35m╔════════════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[1;35m║       trueno-gpu: REAL GPU PIXEL RENDERING             ║\x1b[0m");
    println!("\x1b[1;35m╚════════════════════════════════════════════════════════╝\x1b[0m\n");

    // Step 1: Initialize CUDA
    println!("\x1b[33m[1/6]\x1b[0m Initializing CUDA...");
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("\x1b[31m✗ Failed to initialize CUDA: {}\x1b[0m", e);
            eprintln!("\x1b[31m  Make sure you have an NVIDIA GPU and CUDA installed.\x1b[0m");
            std::process::exit(1);
        }
    };

    let device_name = ctx.device_name().unwrap_or_else(|_| "Unknown".to_string());
    let (free, total) = ctx.memory_info().unwrap_or((0, 0));
    println!(
        "       \x1b[32m✓\x1b[0m GPU: \x1b[1;32m{}\x1b[0m",
        device_name
    );
    println!(
        "       Memory: {} MB free / {} MB total",
        free / 1024 / 1024,
        total / 1024 / 1024
    );

    // Step 2: Generate PTX
    println!("\x1b[33m[2/6]\x1b[0m Generating gradient kernel PTX...");
    let ptx = gradient_kernel_ptx();
    println!(
        "       PTX size: {} bytes ({} lines)",
        ptx.len(),
        ptx.lines().count()
    );

    // Step 3: Load and JIT compile module
    println!("\x1b[33m[3/6]\x1b[0m JIT compiling PTX to SASS...");
    let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("\x1b[31m✗ Failed to compile PTX: {}\x1b[0m", e);
            eprintln!("\n\x1b[33mPTX dump:\x1b[0m");
            for (i, line) in ptx.lines().enumerate() {
                eprintln!("{:4}: {}", i + 1, line);
            }
            std::process::exit(1);
        }
    };
    println!("       \x1b[32m✓\x1b[0m Compiled to device code");

    // Step 4: Allocate GPU memory
    println!("\x1b[33m[4/6]\x1b[0m Allocating GPU memory...");
    let num_pixels = WIDTH * HEIGHT;
    let output_buf: GpuBuffer<f32> =
        GpuBuffer::new(&ctx, num_pixels).expect("Failed to allocate GPU buffer");
    println!(
        "       Allocated {} bytes for {}x{} pixels",
        num_pixels * 4,
        WIDTH,
        HEIGHT
    );

    // Step 5: Launch kernel
    println!("\x1b[33m[5/6]\x1b[0m Launching kernel on GPU...");
    let stream = CudaStream::new(&ctx).expect("Failed to create stream");

    // Kernel parameters
    let mut output_ptr = output_buf.as_ptr();
    let mut width: u32 = WIDTH as u32;
    let mut height: u32 = HEIGHT as u32;

    let mut args: [*mut c_void; 3] = [
        &mut output_ptr as *mut _ as *mut c_void,
        &mut width as *mut _ as *mut c_void,
        &mut height as *mut _ as *mut c_void,
    ];

    // 16x16 thread blocks
    let block_x = 16u32;
    let block_y = 16u32;
    let grid_x = (WIDTH as u32 + block_x - 1) / block_x;
    let grid_y = (HEIGHT as u32 + block_y - 1) / block_y;

    let config = LaunchConfig {
        grid: (grid_x, grid_y, 1),
        block: (block_x, block_y, 1),
        shared_mem: 0,
    };

    println!(
        "       Grid: {}x{}, Block: {}x{}, Threads: {}",
        grid_x,
        grid_y,
        block_x,
        block_y,
        grid_x * grid_y * block_x * block_y
    );

    let start = std::time::Instant::now();
    unsafe {
        stream
            .launch_kernel(&mut module, "gradient_pixel", &config, &mut args)
            .expect("Kernel launch failed");
    }
    stream.synchronize().expect("Stream sync failed");
    let elapsed = start.elapsed();

    println!("       \x1b[32m✓\x1b[0m Kernel executed in {:?}", elapsed);

    // Step 6: Copy results back
    println!("\x1b[33m[6/6]\x1b[0m Copying results from GPU...");
    let mut host_pixels = vec![0.0f32; num_pixels];
    output_buf
        .copy_to_host(&mut host_pixels)
        .expect("D2H copy failed");
    println!("       \x1b[32m✓\x1b[0m Copied {} bytes\n", num_pixels * 4);

    // Render to terminal
    println!("\x1b[1;35m═══ GPU OUTPUT ═══\x1b[0m");
    render_to_terminal(&host_pixels, WIDTH, HEIGHT);

    // Statistics
    println!("\n\x1b[1;35m═══ STATISTICS ═══\x1b[0m");
    println!("┌────────────────────┬──────────────────────┐");
    println!("│ Pixels computed    │ {:>20} │", num_pixels);
    println!("│ GPU execution time │ {:>17?} │", elapsed);
    println!(
        "│ Throughput         │ {:>14.2} Mpx/s │",
        num_pixels as f64 / elapsed.as_secs_f64() / 1_000_000.0
    );
    println!(
        "│ Device             │ {:>20} │",
        &device_name[..device_name.len().min(20)]
    );
    println!("└────────────────────┴──────────────────────┘");
    println!("\n\x1b[32m✓ GPU pixel rendering complete!\x1b[0m\n");
}
