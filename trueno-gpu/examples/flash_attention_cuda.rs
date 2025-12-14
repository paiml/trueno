//! FlashAttention CUDA Execution Example - ACTUALLY RUNS ON GPU
//!
//! Executes the FlashAttention kernel on RTX 4090 and verifies output.
//!
//! Run with: `cargo run -p trueno-gpu --example flash_attention_cuda --features cuda`

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::kernels::{AttentionKernel, Kernel};
use trueno_gpu::ptx::PtxModule;

/// Test configuration
const SEQ_LEN: usize = 64;
const HEAD_DIM: usize = 64;
const NUM_HEADS: usize = 1;

/// Generate attention kernel PTX with proper headers for sm_89 (RTX 4090)
fn attention_kernel_ptx(causal: bool) -> String {
    let kernel = if causal {
        AttentionKernel::new(SEQ_LEN as u32, HEAD_DIM as u32).with_causal()
    } else {
        AttentionKernel::new(SEQ_LEN as u32, HEAD_DIM as u32)
    };

    // Create module with proper headers for RTX 4090 (sm_89, Ada Lovelace)
    PtxModule::new()
        .version(8, 0)
        .target("sm_89")
        .address_size(64)
        .add_kernel(kernel.build_ptx())
        .emit()
}

/// CPU reference implementation for verification
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], seq_len: usize, head_dim: usize, causal: bool) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for i in 0..seq_len {
        // Compute attention scores for row i
        let mut scores = vec![0.0f32; seq_len];
        let mut max_score = f32::NEG_INFINITY;

        for j in 0..seq_len {
            if causal && j > i {
                scores[j] = f32::NEG_INFINITY;
            } else {
                // Dot product Q[i] . K[j]
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[j] = dot * scale;
            }
            max_score = max_score.max(scores[j]);
        }

        // Softmax
        let mut sum_exp = 0.0f32;
        for j in 0..seq_len {
            scores[j] = (scores[j] - max_score).exp();
            sum_exp += scores[j];
        }
        for j in 0..seq_len {
            scores[j] /= sum_exp;
        }

        // Weighted sum of V
        for d in 0..head_dim {
            let mut val = 0.0f32;
            for j in 0..seq_len {
                val += scores[j] * v[j * head_dim + d];
            }
            output[i * head_dim + d] = val;
        }
    }

    output
}

/// Generate deterministic test data
fn generate_test_data(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let size = seq_len * head_dim;
    let mut q = vec![0.0f32; size];
    let mut k = vec![0.0f32; size];
    let mut v = vec![0.0f32; size];

    // Simple deterministic pattern
    for i in 0..size {
        q[i] = ((i % 17) as f32 - 8.0) * 0.1;
        k[i] = ((i % 13) as f32 - 6.0) * 0.1;
        v[i] = ((i % 11) as f32 - 5.0) * 0.1;
    }

    (q, k, v)
}

fn main() {
    println!("\n\x1b[1;35m╔════════════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[1;35m║       trueno-gpu: FlashAttention CUDA Execution        ║\x1b[0m");
    println!("\x1b[1;35m╚════════════════════════════════════════════════════════╝\x1b[0m\n");

    // Step 1: Initialize CUDA
    println!("\x1b[33m[1/8]\x1b[0m Initializing CUDA...");
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
    println!("       \x1b[32m✓\x1b[0m GPU: \x1b[1;32m{}\x1b[0m", device_name);
    println!(
        "       Memory: {} MB free / {} MB total",
        free / 1024 / 1024,
        total / 1024 / 1024
    );

    // Step 2: Generate PTX
    println!("\x1b[33m[2/8]\x1b[0m Generating FlashAttention PTX...");
    let ptx = attention_kernel_ptx(false);
    println!(
        "       PTX size: {} bytes ({} lines)",
        ptx.len(),
        ptx.lines().count()
    );

    // Step 3: JIT compile
    println!("\x1b[33m[3/8]\x1b[0m JIT compiling PTX to SASS...");
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

    // Step 4: Generate test data
    println!("\x1b[33m[4/8]\x1b[0m Generating test data...");
    let (q_host, k_host, v_host) = generate_test_data(SEQ_LEN, HEAD_DIM);
    let total_elements = SEQ_LEN * HEAD_DIM * NUM_HEADS;
    println!(
        "       Q/K/V: {}x{} = {} elements each",
        SEQ_LEN, HEAD_DIM, total_elements
    );

    // Step 5: Allocate GPU buffers
    println!("\x1b[33m[5/8]\x1b[0m Allocating GPU memory...");
    let mut q_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total_elements).expect("Failed to allocate Q");
    let mut k_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total_elements).expect("Failed to allocate K");
    let mut v_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total_elements).expect("Failed to allocate V");
    let o_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total_elements).expect("Failed to allocate O");
    println!(
        "       Allocated {} KB total ({} KB per buffer)",
        total_elements * 4 * 4 / 1024,
        total_elements * 4 / 1024
    );

    // Step 6: Copy data to GPU
    println!("\x1b[33m[6/8]\x1b[0m Copying data to GPU...");
    q_buf.copy_from_host(&q_host).expect("H2D copy Q failed");
    k_buf.copy_from_host(&k_host).expect("H2D copy K failed");
    v_buf.copy_from_host(&v_host).expect("H2D copy V failed");
    println!("       \x1b[32m✓\x1b[0m H2D transfer complete");

    // Step 7: Launch kernel
    println!("\x1b[33m[7/8]\x1b[0m Launching FlashAttention kernel...");
    let stream = CudaStream::new(&ctx).expect("Failed to create stream");

    // Kernel parameters
    let mut q_ptr = q_buf.as_ptr();
    let mut k_ptr = k_buf.as_ptr();
    let mut v_ptr = v_buf.as_ptr();
    let mut o_ptr = o_buf.as_ptr();
    let mut seq_len: u32 = SEQ_LEN as u32;
    let mut head_dim: u32 = HEAD_DIM as u32;
    let mut num_heads: u32 = NUM_HEADS as u32;

    let mut args: [*mut c_void; 7] = [
        &mut q_ptr as *mut _ as *mut c_void,
        &mut k_ptr as *mut _ as *mut c_void,
        &mut v_ptr as *mut _ as *mut c_void,
        &mut o_ptr as *mut _ as *mut c_void,
        &mut seq_len as *mut _ as *mut c_void,
        &mut head_dim as *mut _ as *mut c_void,
        &mut num_heads as *mut _ as *mut c_void,
    ];

    // Launch config: grid = (num_q_blocks, num_heads), block = tile_q * head_dim
    let tile_q = SEQ_LEN.min(64) as u32;
    let num_q_blocks = (SEQ_LEN as u32 + tile_q - 1) / tile_q;
    let threads_per_block = tile_q * (HEAD_DIM as u32);

    let config = LaunchConfig {
        grid: (num_q_blocks, NUM_HEADS as u32, 1),
        block: (threads_per_block.min(1024), 1, 1), // Cap at 1024 threads
        shared_mem: 0, // Shared mem declared in PTX
    };

    println!(
        "       Grid: {}x{}, Block: {}, Threads: {}",
        num_q_blocks,
        NUM_HEADS,
        threads_per_block.min(1024),
        num_q_blocks * NUM_HEADS as u32 * threads_per_block.min(1024)
    );

    let start = std::time::Instant::now();
    unsafe {
        stream
            .launch_kernel(&mut module, "flash_attention", &config, &mut args)
            .expect("Kernel launch failed");
    }
    stream.synchronize().expect("Stream sync failed");
    let elapsed = start.elapsed();

    println!("       \x1b[32m✓\x1b[0m Kernel executed in {:?}", elapsed);

    // Step 8: Copy results back and verify
    println!("\x1b[33m[8/8]\x1b[0m Verifying results...");
    let mut o_host = vec![0.0f32; total_elements];
    o_buf.copy_to_host(&mut o_host).expect("D2H copy failed");

    // Compute CPU reference
    let cpu_output = cpu_attention(&q_host, &k_host, &v_host, SEQ_LEN, HEAD_DIM, false);

    // Compare
    let mut max_diff = 0.0f32;
    let mut avg_diff = 0.0f32;
    for i in 0..total_elements {
        let diff = (o_host[i] - cpu_output[i]).abs();
        max_diff = max_diff.max(diff);
        avg_diff += diff;
    }
    avg_diff /= total_elements as f32;

    let tolerance = 1e-3; // Allow some numerical tolerance
    let passed = max_diff < tolerance;

    if passed {
        println!("       \x1b[32m✓\x1b[0m Output matches CPU within {:.0e}", tolerance);
    } else {
        println!(
            "       \x1b[31m✗\x1b[0m Output differs from CPU (max diff: {:.6})",
            max_diff
        );
    }

    // Statistics
    println!("\n\x1b[1;35m═══ STATISTICS ═══\x1b[0m");
    println!("┌────────────────────┬──────────────────────┐");
    println!("│ Sequence length    │ {:>20} │", SEQ_LEN);
    println!("│ Head dimension     │ {:>20} │", HEAD_DIM);
    println!("│ Num heads          │ {:>20} │", NUM_HEADS);
    println!("│ Total elements     │ {:>20} │", total_elements);
    println!("│ GPU execution time │ {:>17?} │", elapsed);
    println!("│ Max difference     │ {:>18.6e} │", max_diff);
    println!("│ Avg difference     │ {:>18.6e} │", avg_diff);
    println!(
        "│ Verification       │ {:>20} │",
        if passed { "✓ PASS" } else { "✗ FAIL" }
    );
    println!("│ Device             │ {:>20} │", &device_name[..device_name.len().min(20)]);
    println!("└────────────────────┴──────────────────────┘");

    if passed {
        println!("\n\x1b[32m✓ FlashAttention CUDA execution verified!\x1b[0m\n");
    } else {
        println!("\n\x1b[31m✗ FlashAttention verification FAILED\x1b[0m\n");

        // Debug: show first few values
        println!("First 5 GPU outputs: {:?}", &o_host[..5]);
        println!("First 5 CPU outputs: {:?}", &cpu_output[..5]);

        std::process::exit(1);
    }
}
