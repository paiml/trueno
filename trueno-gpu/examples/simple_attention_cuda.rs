//! Simple Attention CUDA Execution - Naive Implementation for Verification
//!
//! This uses a naive O(N²) attention implementation for correctness verification.
//! Run with: `cargo run -p trueno-gpu --example simple_attention_cuda --features cuda`

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::ptx::{PtxKernel, PtxModule, PtxReg, PtxType};

/// Test configuration - keep small for naive O(N²) kernel
const SEQ_LEN: usize = 16;
const HEAD_DIM: usize = 16;

/// Build a simple naive attention kernel
/// Each thread computes one full output row: O[row, :] = softmax(Q[row,:] @ K^T) @ V
fn build_simple_attention_kernel(seq_len: u32, head_dim: u32) -> PtxKernel {
    let scale = 1.0 / (head_dim as f32).sqrt();

    PtxKernel::new("simple_attention")
        .param(PtxType::U64, "q_ptr")
        .param(PtxType::U64, "k_ptr")
        .param(PtxType::U64, "v_ptr")
        .param(PtxType::U64, "o_ptr")
        .param(PtxType::U32, "seq_len")
        .param(PtxType::U32, "head_dim")
        .build(move |ctx| {
            // Thread index = row of output to compute
            let tid = ctx.special_reg(PtxReg::TidX);
            let bid = ctx.special_reg(PtxReg::CtaIdX);
            let bdim = ctx.special_reg(PtxReg::NtidX);
            let row = ctx.mad_lo_u32(bid, bdim, tid);

            // Load parameters
            let seq_len_p = ctx.load_param_u32("seq_len");
            let head_dim_p = ctx.load_param_u32("head_dim");
            let q_ptr = ctx.load_param_u64("q_ptr");
            let k_ptr = ctx.load_param_u64("k_ptr");
            let v_ptr = ctx.load_param_u64("v_ptr");
            let o_ptr = ctx.load_param_u64("o_ptr");

            // Bounds check
            let oob = ctx.setp_ge_u32(row, seq_len_p);
            ctx.branch_if(oob, "exit");

            // Constants
            let scale_reg = ctx.mov_f32_imm(scale);
            let log2_e = ctx.mov_f32_imm(std::f32::consts::LOG2_E);

            // Pre-compute Q row base address: q_ptr + row * head_dim * 4
            let row_stride = ctx.mul_wide_u32(row, head_dim * 4);
            let q_row_base = ctx.add_u64(q_ptr, row_stride);

            // Step 1: Compute max score for numerical stability
            let j = ctx.mov_u32_imm(0);
            let max_score = ctx.mov_f32_imm(f32::NEG_INFINITY);
            let hd_u32 = ctx.mov_u32_imm(head_dim);

            ctx.label("max_loop");
            let max_done = ctx.setp_ge_u32(j, seq_len_p);
            ctx.branch_if(max_done, "max_end");

            // Compute Q[row] . K[j]
            let dot = ctx.mov_f32_imm(0.0);
            let d = ctx.mov_u32_imm(0);

            ctx.label("dot1_loop");
            let dot1_done = ctx.setp_ge_u32(d, head_dim_p);
            ctx.branch_if(dot1_done, "dot1_end");

            // Q[row, d]
            let d_offset = ctx.mul_wide_u32(d, 4);
            let q_addr = ctx.add_u64(q_row_base, d_offset);
            let q_val = ctx.ld_global_f32(q_addr);

            // K[j, d]: k_ptr + (j * head_dim + d) * 4
            let j_stride = ctx.mul_wide_u32(j, head_dim * 4);
            let k_j_base = ctx.add_u64(k_ptr, j_stride);
            let k_addr = ctx.add_u64(k_j_base, d_offset);
            let k_val = ctx.ld_global_f32(k_addr);

            ctx.fma_f32_inplace(dot, q_val, k_val);
            ctx.add_u32_inplace(d, 1);
            ctx.branch("dot1_loop");
            ctx.label("dot1_end");

            // Apply scale and update max
            let score = ctx.mul_f32(dot, scale_reg);
            let new_max = ctx.max_f32(max_score, score);
            ctx.mov_f32_reg(max_score, new_max);

            ctx.add_u32_inplace(j, 1);
            ctx.branch("max_loop");
            ctx.label("max_end");

            // Step 2: Compute sum of exp(score - max)
            let j2 = ctx.mov_u32_imm(0);
            let sum_exp = ctx.mov_f32_imm(0.0);

            ctx.label("sum_loop");
            let sum_done = ctx.setp_ge_u32(j2, seq_len_p);
            ctx.branch_if(sum_done, "sum_end");

            // Recompute Q[row] . K[j2]
            let dot2 = ctx.mov_f32_imm(0.0);
            let d2 = ctx.mov_u32_imm(0);

            ctx.label("dot2_loop");
            let dot2_done = ctx.setp_ge_u32(d2, head_dim_p);
            ctx.branch_if(dot2_done, "dot2_end");

            let d2_offset = ctx.mul_wide_u32(d2, 4);
            let q_addr2 = ctx.add_u64(q_row_base, d2_offset);
            let q_val2 = ctx.ld_global_f32(q_addr2);

            let j2_stride = ctx.mul_wide_u32(j2, head_dim * 4);
            let k_j2_base = ctx.add_u64(k_ptr, j2_stride);
            let k_addr2 = ctx.add_u64(k_j2_base, d2_offset);
            let k_val2 = ctx.ld_global_f32(k_addr2);

            ctx.fma_f32_inplace(dot2, q_val2, k_val2);
            ctx.add_u32_inplace(d2, 1);
            ctx.branch("dot2_loop");
            ctx.label("dot2_end");

            let score2 = ctx.mul_f32(dot2, scale_reg);
            let shifted = ctx.sub_f32(score2, max_score);
            let shifted_log2 = ctx.mul_f32(shifted, log2_e);
            let exp_val = ctx.ex2_f32(shifted_log2);
            let new_sum = ctx.add_f32(sum_exp, exp_val);
            ctx.mov_f32_reg(sum_exp, new_sum);

            ctx.add_u32_inplace(j2, 1);
            ctx.branch("sum_loop");
            ctx.label("sum_end");

            // Step 3: Compute output for each dimension
            let col = ctx.mov_u32_imm(0);

            ctx.label("col_loop");
            let col_done = ctx.setp_ge_u32(col, head_dim_p);
            ctx.branch_if(col_done, "col_end");

            let output_val = ctx.mov_f32_imm(0.0);
            let j3 = ctx.mov_u32_imm(0);

            ctx.label("attn_loop");
            let attn_done = ctx.setp_ge_u32(j3, seq_len_p);
            ctx.branch_if(attn_done, "attn_end");

            // Recompute attention weight for j3
            let dot3 = ctx.mov_f32_imm(0.0);
            let d3 = ctx.mov_u32_imm(0);

            ctx.label("dot3_loop");
            let dot3_done = ctx.setp_ge_u32(d3, head_dim_p);
            ctx.branch_if(dot3_done, "dot3_end");

            let d3_offset = ctx.mul_wide_u32(d3, 4);
            let q_addr3 = ctx.add_u64(q_row_base, d3_offset);
            let q_val3 = ctx.ld_global_f32(q_addr3);

            let j3_stride = ctx.mul_wide_u32(j3, head_dim * 4);
            let k_j3_base = ctx.add_u64(k_ptr, j3_stride);
            let k_addr3 = ctx.add_u64(k_j3_base, d3_offset);
            let k_val3 = ctx.ld_global_f32(k_addr3);

            ctx.fma_f32_inplace(dot3, q_val3, k_val3);
            ctx.add_u32_inplace(d3, 1);
            ctx.branch("dot3_loop");
            ctx.label("dot3_end");

            let score3 = ctx.mul_f32(dot3, scale_reg);
            let shifted3 = ctx.sub_f32(score3, max_score);
            let shifted3_log2 = ctx.mul_f32(shifted3, log2_e);
            let exp_val3 = ctx.ex2_f32(shifted3_log2);
            let attn_weight = ctx.div_f32(exp_val3, sum_exp);

            // V[j3, col]
            let col_offset = ctx.mul_wide_u32(col, 4);
            let v_j3_base = ctx.add_u64(v_ptr, j3_stride);
            let v_addr = ctx.add_u64(v_j3_base, col_offset);
            let v_val = ctx.ld_global_f32(v_addr);

            ctx.fma_f32_inplace(output_val, attn_weight, v_val);

            ctx.add_u32_inplace(j3, 1);
            ctx.branch("attn_loop");
            ctx.label("attn_end");

            // Store O[row, col]
            let o_row_base = ctx.add_u64(o_ptr, row_stride);
            let o_addr = ctx.add_u64(o_row_base, col_offset);
            ctx.st_global_f32(o_addr, output_val);

            ctx.add_u32_inplace(col, 1);
            ctx.branch("col_loop");
            ctx.label("col_end");

            ctx.label("exit");
            ctx.ret();
        })
}

/// Generate PTX with proper headers for sm_89
fn attention_kernel_ptx() -> String {
    let kernel = build_simple_attention_kernel(SEQ_LEN as u32, HEAD_DIM as u32);

    PtxModule::new()
        .version(8, 0)
        .target("sm_89")
        .address_size(64)
        .add_kernel(kernel)
        .emit()
}

/// CPU reference implementation
fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], seq_len: usize, head_dim: usize) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for i in 0..seq_len {
        let mut scores = vec![0.0f32; seq_len];
        let mut max_score = f32::NEG_INFINITY;

        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[j] = dot * scale;
            max_score = max_score.max(scores[j]);
        }

        let mut sum_exp = 0.0f32;
        for j in 0..seq_len {
            scores[j] = (scores[j] - max_score).exp();
            sum_exp += scores[j];
        }
        for j in 0..seq_len {
            scores[j] /= sum_exp;
        }

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

fn generate_test_data(seq_len: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let size = seq_len * head_dim;
    let mut q = vec![0.0f32; size];
    let mut k = vec![0.0f32; size];
    let mut v = vec![0.0f32; size];

    for i in 0..size {
        q[i] = ((i % 17) as f32 - 8.0) * 0.1;
        k[i] = ((i % 13) as f32 - 6.0) * 0.1;
        v[i] = ((i % 11) as f32 - 5.0) * 0.1;
    }

    (q, k, v)
}

fn main() {
    println!("\n\x1b[1;35m╔════════════════════════════════════════════════════════╗\x1b[0m");
    println!("\x1b[1;35m║    trueno-gpu: Simple Attention CUDA Verification      ║\x1b[0m");
    println!("\x1b[1;35m╚════════════════════════════════════════════════════════╝\x1b[0m\n");

    println!("\x1b[33m[1/7]\x1b[0m Initializing CUDA...");
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("\x1b[31m✗ CUDA init failed: {}\x1b[0m", e);
            std::process::exit(1);
        }
    };

    let device_name = ctx.device_name().unwrap_or_else(|_| "Unknown".to_string());
    println!("       \x1b[32m✓\x1b[0m GPU: \x1b[1;32m{}\x1b[0m", device_name);

    println!("\x1b[33m[2/7]\x1b[0m Generating PTX...");
    let ptx = attention_kernel_ptx();
    println!("       PTX: {} bytes ({} lines)", ptx.len(), ptx.lines().count());

    println!("\x1b[33m[3/7]\x1b[0m JIT compiling...");
    let mut module = match CudaModule::from_ptx(&ctx, &ptx) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("\x1b[31m✗ PTX compile failed: {}\x1b[0m", e);
            eprintln!("\nPTX dump:");
            for (i, line) in ptx.lines().enumerate() {
                eprintln!("{:4}: {}", i + 1, line);
            }
            std::process::exit(1);
        }
    };
    println!("       \x1b[32m✓\x1b[0m Compiled");

    println!("\x1b[33m[4/7]\x1b[0m Generating test data...");
    let (q_host, k_host, v_host) = generate_test_data(SEQ_LEN, HEAD_DIM);
    let total_elements = SEQ_LEN * HEAD_DIM;
    println!("       Q/K/V: {}x{} = {}", SEQ_LEN, HEAD_DIM, total_elements);

    println!("\x1b[33m[5/7]\x1b[0m Allocating and transferring...");
    let mut q_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total_elements).expect("Q alloc");
    let mut k_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total_elements).expect("K alloc");
    let mut v_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total_elements).expect("V alloc");
    let o_buf: GpuBuffer<f32> = GpuBuffer::new(&ctx, total_elements).expect("O alloc");

    q_buf.copy_from_host(&q_host).expect("H2D Q");
    k_buf.copy_from_host(&k_host).expect("H2D K");
    v_buf.copy_from_host(&v_host).expect("H2D V");
    println!("       \x1b[32m✓\x1b[0m Done");

    println!("\x1b[33m[6/7]\x1b[0m Launching kernel...");
    let stream = CudaStream::new(&ctx).expect("stream");

    let mut q_ptr = q_buf.as_ptr();
    let mut k_ptr = k_buf.as_ptr();
    let mut v_ptr = v_buf.as_ptr();
    let mut o_ptr = o_buf.as_ptr();
    let mut seq_len_param: u32 = SEQ_LEN as u32;
    let mut head_dim_param: u32 = HEAD_DIM as u32;

    let mut args: [*mut c_void; 6] = [
        &mut q_ptr as *mut _ as *mut c_void,
        &mut k_ptr as *mut _ as *mut c_void,
        &mut v_ptr as *mut _ as *mut c_void,
        &mut o_ptr as *mut _ as *mut c_void,
        &mut seq_len_param as *mut _ as *mut c_void,
        &mut head_dim_param as *mut _ as *mut c_void,
    ];

    // One thread per row
    let block_size = SEQ_LEN.min(256) as u32;
    let grid_size = (SEQ_LEN as u32 + block_size - 1) / block_size;

    let config = LaunchConfig {
        grid: (grid_size, 1, 1),
        block: (block_size, 1, 1),
        shared_mem: 0,
    };

    println!("       Grid: {}, Block: {}", grid_size, block_size);

    let start = std::time::Instant::now();
    unsafe {
        stream
            .launch_kernel(&mut module, "simple_attention", &config, &mut args)
            .expect("launch");
    }
    stream.synchronize().expect("sync");
    let elapsed = start.elapsed();

    println!("       \x1b[32m✓\x1b[0m Executed in {:?}", elapsed);

    println!("\x1b[33m[7/7]\x1b[0m Verifying...");
    let mut o_host = vec![0.0f32; total_elements];
    o_buf.copy_to_host(&mut o_host).expect("D2H");

    let cpu_output = cpu_attention(&q_host, &k_host, &v_host, SEQ_LEN, HEAD_DIM);

    let mut max_diff = 0.0f32;
    for i in 0..total_elements {
        max_diff = max_diff.max((o_host[i] - cpu_output[i]).abs());
    }

    let tolerance = 1e-3;
    let passed = max_diff < tolerance;

    if passed {
        println!("       \x1b[32m✓\x1b[0m PASS (max diff: {:.6e})", max_diff);
    } else {
        println!("       \x1b[31m✗\x1b[0m FAIL (max diff: {:.6e})", max_diff);
    }

    println!("\n\x1b[1;35m═══ RESULTS ═══\x1b[0m");
    println!("┌────────────────────┬──────────────────────┐");
    println!("│ Sequence length    │ {:>20} │", SEQ_LEN);
    println!("│ Head dimension     │ {:>20} │", HEAD_DIM);
    println!("│ GPU execution      │ {:>17?} │", elapsed);
    println!("│ Max difference     │ {:>18.6e} │", max_diff);
    println!("│ Status             │ {:>20} │", if passed { "✓ PASS" } else { "✗ FAIL" });
    println!("└────────────────────┴──────────────────────┘");

    if !passed {
        println!("\nGPU first 5: {:?}", &o_host[..5]);
        println!("CPU first 5: {:?}", &cpu_output[..5]);
        std::process::exit(1);
    }

    println!("\n\x1b[32m✓ Simple Attention CUDA verified on {}!\x1b[0m\n", device_name);
}
