//! cuBLAS vs PTX GEMM Benchmark — Training Shapes (ALB-075)
//!
//! Three-tier GEMM comparison for Albor 350M training shapes:
//! - Tier 2: Rust cuBLAS wrapper (CublasHandle::gemm_f32_row_major)
//! - Tier 3: Rust PTX hand-written kernel (GemmKernel::tiled_unrolled)
//!
//! Tier 1 (raw C cuBLAS ceiling) is measured separately via gemm_cublas_raw.c.
//!
//! Run: cargo run --example bench_cublas_vs_ptx --features cuda --release
//!
//! Contract: cublas-gemm-v1.yaml (FALSIFY-CUBLAS-003, -008)

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This benchmark requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() {
    bench_cublas_vs_ptx::run();
}

#[cfg(feature = "cuda")]
mod bench_cublas_vs_ptx {
    use trueno_gpu::driver::{CublasHandle, CudaContext, CudaModule, CudaStream, GpuBuffer};
    use trueno_gpu::kernels::{GemmKernel, Kernel};

    /// Albor 350M training GEMM shapes (exact dimensions from forward+backward)
    const SHAPES: &[(u32, u32, u32, &str)] = &[
        // Attention projections (S=1024 × batch=4 = 4096 tokens)
        (4096, 1024, 1024, "attn_qo"),       // Q/O projection: [S,H] × [H,H]
        (4096, 256, 1024, "attn_kv"),         // K/V projection: [S,H] × [H,kv_dim]
        // FFN projections
        (4096, 4096, 1024, "ffn_gate_up"),    // gate/up: [S,H] × [H,I]
        (4096, 1024, 4096, "ffn_down"),       // down: [S,I] × [I,H]
        // LM head
        (4096, 32768, 1024, "lm_head"),       // [S,H] × [H,V]
        // Reference squares
        (1024, 1024, 1024, "square_1k"),
        (4096, 4096, 4096, "square_4k"),
    ];

    const WARMUP_ITERS: usize = 10;
    const TIMED_ITERS: usize = 50;

    pub fn run() {
        let ctx = CudaContext::new(0).expect("CUDA context required");
        let stream = CudaStream::new(&ctx).expect("stream required");
        let cublas = CublasHandle::new(&ctx).expect("cuBLAS handle required");
        cublas.set_stream(&stream).expect("set_stream");

        let sm_target = ctx.sm_target().unwrap_or_else(|_| "sm_70".to_string());

        println!("==========================================================================");
        println!("GEMM BENCHMARK: cuBLAS (Tier 2) vs PTX (Tier 3)  — Albor 350M shapes");
        println!("GPU: RTX 4090 (sm_89), cuBLAS TF32 tensor cores");
        println!("Warmup: {WARMUP_ITERS} iters, Timed: {TIMED_ITERS} iters");
        println!("==========================================================================");
        println!();
        println!(
            "{:<14} {:>10} {:>10} {:>10} {:>10}",
            "Shape", "cuBLAS", "PTX", "Speedup", "% Peak"
        );
        println!("{}", "-".repeat(60));

        for &(m, n, k, label) in SHAPES {
            let cublas_tflops = bench_cublas(&ctx, &stream, &cublas, m, n, k);
            let ptx_tflops = bench_ptx(&ctx, &stream, &sm_target, m, n, k);

            let speedup = if ptx_tflops > 0.0 {
                cublas_tflops / ptx_tflops
            } else {
                f64::INFINITY
            };
            // TF32 tensor core peak = 165 TFLOP/s (RTX 4090)
            let pct_peak = cublas_tflops / 165.0 * 100.0;

            println!(
                "{:<14} {:>8.1}T  {:>8.1}T  {:>8.1}x  {:>7.1}%",
                label, cublas_tflops, ptx_tflops, speedup, pct_peak
            );
        }

        println!();
        println!("cuBLAS = CublasHandle::gemm_f32_row_major (TF32 tensor cores)");
        println!("PTX    = GemmKernel::tiled_unrolled(m,n,k,16) scalar FP32 FMA");
        println!("% Peak = cuBLAS / 165 TFLOP/s (RTX 4090 TF32 tensor core peak)");
        println!();
        println!("Contract: cublas-gemm-v1.yaml (FALSIFY-CUBLAS-003)");
    }

    fn bench_cublas(
        ctx: &CudaContext,
        stream: &CudaStream,
        cublas: &CublasHandle,
        m: u32,
        n: u32,
        k: u32,
    ) -> f64 {
        let a_buf =
            GpuBuffer::<f32>::new(ctx, (m * k) as usize).expect("A alloc");
        let b_buf =
            GpuBuffer::<f32>::new(ctx, (k * n) as usize).expect("B alloc");
        let c_buf =
            GpuBuffer::<f32>::new(ctx, (m * n) as usize).expect("C alloc");

        // Warmup
        for _ in 0..WARMUP_ITERS {
            cublas
                .gemm_f32_row_major(
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    0.0,
                    c_buf.as_ptr(),
                )
                .expect("cuBLAS warmup");
        }
        stream.synchronize().expect("warmup sync");

        // Timed
        let start = std::time::Instant::now();
        for _ in 0..TIMED_ITERS {
            cublas
                .gemm_f32_row_major(
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    0.0,
                    c_buf.as_ptr(),
                )
                .expect("cuBLAS timed");
        }
        stream.synchronize().expect("timed sync");
        let elapsed = start.elapsed();

        let flops = 2.0 * m as f64 * n as f64 * k as f64 * TIMED_ITERS as f64;
        flops / elapsed.as_secs_f64() / 1e12
    }

    fn bench_ptx(
        ctx: &CudaContext,
        stream: &CudaStream,
        sm_target: &str,
        m: u32,
        n: u32,
        k: u32,
    ) -> f64 {
        let tile: u32 = 16;
        let kernel = GemmKernel::tiled_unrolled(m, n, k, tile);
        let kernel_name = kernel.name().to_string();
        let ptx = kernel.emit_ptx_for_target(sm_target);
        let mut module = CudaModule::from_ptx(ctx, &ptx)
            .expect("PTX compile");

        let a_buf =
            GpuBuffer::<f32>::new(ctx, (m * k) as usize).expect("A alloc");
        let b_buf =
            GpuBuffer::<f32>::new(ctx, (k * n) as usize).expect("B alloc");
        let c_buf =
            GpuBuffer::<f32>::new(ctx, (m * n) as usize).expect("C alloc");

        let smem = 2 * tile * tile * 4;
        let config = trueno_gpu::driver::LaunchConfig {
            grid: (n.div_ceil(tile), m.div_ceil(tile), 1),
            block: (tile, tile, 1),
            shared_mem: smem,
        };

        let a_ptr = a_buf.as_ptr();
        let b_ptr = b_buf.as_ptr();
        let c_ptr = c_buf.as_ptr();

        // Warmup
        for _ in 0..WARMUP_ITERS {
            let mut args: [*mut std::ffi::c_void; 6] = [
                &a_ptr as *const _ as *mut _,
                &b_ptr as *const _ as *mut _,
                &c_ptr as *const _ as *mut _,
                &m as *const _ as *mut _,
                &n as *const _ as *mut _,
                &k as *const _ as *mut _,
            ];
            unsafe {
                stream
                    .launch_kernel(&mut module, &kernel_name, &config, &mut args)
                    .expect("PTX warmup");
            }
        }
        stream.synchronize().expect("warmup sync");

        // Timed
        let start = std::time::Instant::now();
        for _ in 0..TIMED_ITERS {
            let mut args: [*mut std::ffi::c_void; 6] = [
                &a_ptr as *const _ as *mut _,
                &b_ptr as *const _ as *mut _,
                &c_ptr as *const _ as *mut _,
                &m as *const _ as *mut _,
                &n as *const _ as *mut _,
                &k as *const _ as *mut _,
            ];
            unsafe {
                stream
                    .launch_kernel(&mut module, &kernel_name, &config, &mut args)
                    .expect("PTX timed");
            }
        }
        stream.synchronize().expect("timed sync");
        let elapsed = start.elapsed();

        let flops = 2.0 * m as f64 * n as f64 * k as f64 * TIMED_ITERS as f64;
        flops / elapsed.as_secs_f64() / 1e12
    }
}
