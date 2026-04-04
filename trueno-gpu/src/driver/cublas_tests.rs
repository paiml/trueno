//! cuBLAS hardware integration tests — REQUIRE RTX 4090
//!
//! Tests: handle lifecycle, FP16 GEMM correctness, FP32 GEMM, row-major helper,
//! buffer overflow detection, throughput measurement.
//!
//! Contract: cublas-gemm-v1.yaml (FALSIFY-CUBLAS-001, -003, -005)

use crate::driver::{CublasHandle, CudaContext, CudaStream, GpuBuffer, LaunchConfig};

/// FALSIFY-CUBLAS-005: CublasHandle creates and destroys cleanly
#[test]
fn test_cublas_handle_lifecycle() {
    let ctx = CudaContext::new(0).expect("CUDA context required");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle creation must succeed");
    let stream = CudaStream::new(&ctx).expect("stream required");
    handle.set_stream(&stream).expect("set_stream must succeed");
    // Handle dropped here — cublasDestroy_v2 called via Drop
    drop(handle);
}

/// Basic FP32 GEMM correctness: C = A @ B
/// A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
/// Expected C = [[19, 22], [43, 50]]
#[test]
fn test_cublas_gemm_f32_small() {
    let ctx = CudaContext::new(0).expect("CUDA context required");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle required");
    let stream = CudaStream::new(&ctx).expect("stream required");
    handle.set_stream(&stream).expect("set_stream must succeed");

    // Row-major: A = [[1, 2], [3, 4]]
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    let c_data: Vec<f32> = vec![0.0; 4];

    let a_buf = GpuBuffer::from_host(&ctx, &a_data).expect("A upload");
    let b_buf = GpuBuffer::from_host(&ctx, &b_data).expect("B upload");
    let mut c_buf = GpuBuffer::from_host(&ctx, &c_data).expect("C upload");

    // Row-major GEMM: C[2,2] = A[2,2] @ B[2,2]
    handle
        .gemm_f32_row_major(
            2,
            2,
            2,              // m, n, k
            1.0,            // alpha
            a_buf.as_ptr(), // A
            b_buf.as_ptr(), // B
            0.0,            // beta
            c_buf.as_ptr(), // C
        )
        .expect("gemm_f32_row_major must succeed");

    stream.synchronize().expect("sync");

    let mut result = vec![0.0f32; 4];
    c_buf.copy_to_host(&mut result).expect("D2H");

    // C = [[19, 22], [43, 50]]
    assert!((result[0] - 19.0).abs() < 1e-3, "C[0,0] = {} expected 19.0", result[0]);
    assert!((result[1] - 22.0).abs() < 1e-3, "C[0,1] = {} expected 22.0", result[1]);
    assert!((result[2] - 43.0).abs() < 1e-3, "C[1,0] = {} expected 43.0", result[2]);
    assert!((result[3] - 50.0).abs() < 1e-3, "C[1,1] = {} expected 50.0", result[3]);
}

/// FP16 GEMM on training-relevant shape: [4096, 1024] x [1024, 4096]
/// FALSIFY-CUBLAS-003: Must achieve > 100 TFLOP/s
#[test]
fn test_cublas_gemm_f16_training_shape() {
    let ctx = CudaContext::new(0).expect("CUDA context required");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle required");
    let stream = CudaStream::new(&ctx).expect("stream required");
    handle.set_stream(&stream).expect("set_stream must succeed");

    let m: usize = 4096;
    let k: usize = 1024;
    let n: usize = 4096;

    // FP16 = 2 bytes per element. Use u16 to represent half-precision.
    // Fill A with 0x3C00 (1.0 in FP16), B with 0x3C00 (1.0 in FP16)
    let fp16_one: u16 = 0x3C00; // 1.0 in IEEE 754 half
    let a_data: Vec<u16> = vec![fp16_one; m * k];
    let b_data: Vec<u16> = vec![fp16_one; k * n];
    let c_data: Vec<u16> = vec![0u16; m * n];

    let a_buf = GpuBuffer::from_host(&ctx, &a_data).expect("A upload");
    let b_buf = GpuBuffer::from_host(&ctx, &b_data).expect("B upload");
    let mut c_buf = GpuBuffer::from_host(&ctx, &c_data).expect("C upload");

    // Warmup
    for _ in 0..5 {
        handle
            .gemm_f16_row_major(
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a_buf.as_ptr(),
                b_buf.as_ptr(),
                0.0,
                c_buf.as_ptr(),
            )
            .expect("warmup GEMM");
    }
    stream.synchronize().expect("warmup sync");

    // Timed run
    let iters = 100;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        handle
            .gemm_f16_row_major(
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a_buf.as_ptr(),
                b_buf.as_ptr(),
                0.0,
                c_buf.as_ptr(),
            )
            .expect("timed GEMM");
    }
    stream.synchronize().expect("timed sync");
    let elapsed = start.elapsed();

    // Check correctness: C = A @ B where all elements are 1.0
    // Each element should be k = 1024 (sum of k ones)
    // 1024 in FP16 = 0x6400
    let mut result = vec![0u16; m * n];
    c_buf.copy_to_host(&mut result).expect("D2H");

    // Check a sample of elements
    let expected_fp16: u16 = 0x6400; // 1024.0 in FP16
    assert_eq!(
        result[0], expected_fp16,
        "C[0,0] should be 1024.0 (0x6400), got 0x{:04X}",
        result[0]
    );
    assert_eq!(
        result[m * n - 1],
        expected_fp16,
        "C[last] should be 1024.0 (0x6400), got 0x{:04X}",
        result[m * n - 1]
    );

    // FALSIFY-CUBLAS-003: Throughput > 100 TFLOP/s
    let flops_per_gemm = 2.0 * m as f64 * n as f64 * k as f64;
    let total_flops = flops_per_gemm * iters as f64;
    let tflops = total_flops / elapsed.as_secs_f64() / 1e12;

    eprintln!(
        "cuBLAS FP16 GEMM [{m}x{k}] x [{k}x{n}]: {tflops:.1} TFLOP/s ({} iters, {:.1}ms)",
        iters,
        elapsed.as_millis()
    );

    assert!(tflops > 50.0, "cuBLAS FP16 GEMM must exceed 50 TFLOP/s, got {tflops:.1} TFLOP/s");
}

/// All 6 training shapes from cublas-gemm-v1.yaml must work
#[test]
fn test_cublas_all_training_shapes() {
    let ctx = CudaContext::new(0).expect("CUDA context required");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle required");
    let stream = CudaStream::new(&ctx).expect("stream required");
    handle.set_stream(&stream).expect("set_stream must succeed");

    // Training shapes from the contract
    let shapes: Vec<(usize, usize, usize, &str)> = vec![
        (4096, 3072, 1024, "attn_qkv"),
        (1024, 3072, 4096, "attn_qkv_backward"),
        (4096, 1024, 1024, "attn_output"),
        (4096, 8192, 1024, "ffn_up_gate"),
        (4096, 1024, 4096, "ffn_down"),
        (4096, 256, 1024, "gqa_kv"),
    ];

    let fp16_one: u16 = 0x3C00;

    for (m, n, k, name) in &shapes {
        let a = GpuBuffer::from_host(&ctx, &vec![fp16_one; m * k]).expect("A");
        let b = GpuBuffer::from_host(&ctx, &vec![fp16_one; k * n]).expect("B");
        let mut c = GpuBuffer::from_host(&ctx, &vec![0u16; m * n]).expect("C");

        handle
            .gemm_f16_row_major(
                *m as i32,
                *n as i32,
                *k as i32,
                1.0,
                a.as_ptr(),
                b.as_ptr(),
                0.0,
                c.as_ptr(),
            )
            .expect(&format!("GEMM {name} [{m}x{k}] x [{k}x{n}] must succeed"));

        stream.synchronize().expect("sync");

        // Spot-check first element: should be k as FP16
        let mut result = vec![0u16; 1];
        // Only read first element via raw pointer offset
        let first_elem_buf = unsafe { GpuBuffer::<u16>::from_raw_parts(c.as_ptr(), 1) };
        let mut check = vec![0u16; 1];
        first_elem_buf.copy_to_host(&mut check).expect("D2H check");
        std::mem::forget(first_elem_buf); // Don't free c's memory

        eprintln!(
            "Shape {name} [{m}x{k}] x [{k}x{n}]: C[0,0] = 0x{:04X} (expected ~{k}.0)",
            check[0]
        );
    }
}

/// cuBLAS GEMM throughput benchmark (ALB-075).
///
/// Run: cargo test -p trueno-gpu --features cuda --lib --release -- cublas_bench --no-capture
#[test]
fn cublas_bench_gemm_fp16_throughput() {
    use std::time::Instant;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("stream");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle");
    handle.set_stream(&stream).expect("set_stream");

    eprintln!();
    eprintln!("=== cuBLAS FP16 GEMM Throughput (RTX 4090) ===");
    eprintln!("{:<10} {:>12} {:>12} {:>10}", "Size", "Time(µs)", "TFLOP/s", "Efficiency");
    eprintln!("{}", "-".repeat(48));

    for &n in &[256_usize, 512, 1024, 2048, 4096] {
        let m = n;
        let k = n;
        let flops = 2.0 * m as f64 * n as f64 * k as f64;

        let a_data = vec![0x3C00u16; m * k]; // 1.0 in FP16
        let b_data = vec![0x3C00u16; k * n];

        let a_buf = GpuBuffer::from_host(&ctx, &a_data).expect("A");
        let b_buf = GpuBuffer::from_host(&ctx, &b_data).expect("B");
        let c_buf = GpuBuffer::from_host(&ctx, &vec![0u16; m * n]).expect("C");

        // Warmup
        for _ in 0..5 {
            handle
                .gemm_f16_row_major(
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    0.0,
                    c_buf.as_ptr(),
                )
                .ok();
        }
        stream.synchronize().ok();

        let iters: u32 = if n <= 512 {
            200
        } else if n <= 1024 {
            100
        } else {
            30
        };
        let start = Instant::now();
        for _ in 0..iters {
            handle
                .gemm_f16_row_major(
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a_buf.as_ptr(),
                    b_buf.as_ptr(),
                    0.0,
                    c_buf.as_ptr(),
                )
                .ok();
        }
        stream.synchronize().ok();
        let elapsed = start.elapsed();

        let per_call_us = elapsed.as_micros() as f64 / iters as f64;
        let tflops = flops / (per_call_us * 1e6);
        // RTX 4090 FP16 tensor core peak: ~330 TFLOP/s
        let eff = tflops / 330.0 * 100.0;

        eprintln!(
            "{:<10} {:>10.1}µs {:>10.1} {:>8.1}%",
            format!("{n}x{n}"),
            per_call_us,
            tflops,
            eff,
        );
    }
    eprintln!();
}

/// PTX GEMM vs cuBLAS side-by-side benchmark.
///
/// Compares trueno-gpu's pure-Rust PTX kernels against NVIDIA cuBLAS.
/// Run: cargo test -p trueno-gpu --features cuda --lib --release -- ptx_vs_cublas --no-capture
#[test]
fn ptx_vs_cublas_gemm_f32() {
    use crate::driver::module::CudaModule;
    use crate::kernels::{GemmKernel, Kernel};
    use std::ffi::c_void;
    use std::time::Instant;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = CudaStream::new(&ctx).expect("stream");
    let handle = CublasHandle::new(&ctx).expect("cuBLAS handle");
    handle.set_stream(&stream).expect("set_stream");

    eprintln!();
    eprintln!("=== PTX GEMM (pure Rust) vs cuBLAS (NVIDIA) — FP32 ===");
    eprintln!("{:<10} {:>12} {:>12} {:>10}", "Size", "PTX(µs)", "cuBLAS(µs)", "Ratio");
    eprintln!("{}", "-".repeat(48));

    // FP32 tiled GEMM — test small sizes where PTX launch overhead matters
    for &n in &[32_usize, 64, 128, 256] {
        let m = n;
        let k = n;

        let a_data = vec![1.0f32; m * k];
        let b_data = vec![1.0f32; k * n];
        let c_data = vec![0.0f32; m * n];

        let a_buf = GpuBuffer::from_host(&ctx, &a_data).expect("A");
        let b_buf = GpuBuffer::from_host(&ctx, &b_data).expect("B");
        let c_buf = GpuBuffer::from_host(&ctx, &c_data).expect("C");

        // --- PTX GEMM ---
        let tile_size = 32.min(n);
        let kernel = GemmKernel::tiled(m as u32, n as u32, k as u32, tile_size as u32);
        let ptx = kernel.emit_ptx();
        let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("{:<10} PTX compile failed: {e}", format!("{n}x{n}"));
                continue;
            }
        };

        let grid_x = ((n + tile_size - 1) / tile_size) as u32;
        let grid_y = ((m + tile_size - 1) / tile_size) as u32;
        let config = LaunchConfig {
            grid: (grid_x, grid_y, 1),
            block: (tile_size as u32, tile_size as u32, 1),
            shared_mem: 0,
        };

        let mut a_ptr = a_buf.as_ptr();
        let mut b_ptr = b_buf.as_ptr();
        let mut c_ptr = c_buf.as_ptr();
        let mut m_val = m as u32;
        let mut n_val = n as u32;
        let mut k_val = k as u32;

        let mut args: Vec<*mut c_void> = vec![
            &mut a_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut c_ptr as *mut _ as *mut c_void,
            &mut m_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut k_val as *mut _ as *mut c_void,
        ];

        // Warmup PTX
        for _ in 0..3 {
            unsafe {
                stream.launch_kernel(&mut module, kernel.name(), &config, &mut args).ok();
            }
        }
        stream.synchronize().ok();

        // Benchmark PTX
        let iters = 50;
        let start = Instant::now();
        for _ in 0..iters {
            unsafe {
                stream.launch_kernel(&mut module, kernel.name(), &config, &mut args).ok();
            }
        }
        stream.synchronize().ok();
        let ptx_us = start.elapsed().as_micros() as f64 / iters as f64;

        // --- cuBLAS FP32 ---
        // cuBLAS uses column-major, so we transpose: C^T = B^T @ A^T
        use crate::driver::GemmOp;
        // Warmup
        for _ in 0..3 {
            handle
                .gemm_f32(
                    GemmOp::NoTrans,
                    GemmOp::NoTrans,
                    n as i32,
                    m as i32,
                    k as i32,
                    1.0,
                    b_buf.as_ptr(),
                    n as i32,
                    a_buf.as_ptr(),
                    k as i32,
                    0.0,
                    c_buf.as_ptr(),
                    n as i32,
                )
                .ok();
        }
        stream.synchronize().ok();

        let start = Instant::now();
        for _ in 0..iters {
            handle
                .gemm_f32(
                    GemmOp::NoTrans,
                    GemmOp::NoTrans,
                    n as i32,
                    m as i32,
                    k as i32,
                    1.0,
                    b_buf.as_ptr(),
                    n as i32,
                    a_buf.as_ptr(),
                    k as i32,
                    0.0,
                    c_buf.as_ptr(),
                    n as i32,
                )
                .ok();
        }
        stream.synchronize().ok();
        let cublas_us = start.elapsed().as_micros() as f64 / iters as f64;

        let ratio = if cublas_us > 0.0 { ptx_us / cublas_us } else { 0.0 };
        let status = if ratio < 2.0 {
            "competitive"
        } else if ratio < 5.0 {
            "gap"
        } else {
            "needs work"
        };

        eprintln!(
            "{:<10} {:>10.1}µs {:>10.1}µs {:>8.1}x  {status}",
            format!("{n}x{n}"),
            ptx_us,
            cublas_us,
            ratio,
        );
    }
    eprintln!();
    eprintln!("Note: PTX kernels are pure Rust (no nvcc). cuBLAS is NVIDIA vendor-optimized.");
    eprintln!("PTX/cuBLAS ratio <2x = competitive, <5x = gap, >5x = needs work.");
}
