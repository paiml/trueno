//! PTX Register Allocation Strategy
//!
//! This example demonstrates trueno-gpu's approach to register allocation,
//! which delegates physical register assignment to NVIDIA's ptxas compiler.
//!
//! ## Architecture Overview
//!
//! Traditional compilers (like LLVM) must map infinite virtual registers to
//! a finite set of physical registers using algorithms like Graph Coloring
//! or Linear Scan. Trueno takes a different approach:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  Trueno PTX Builder (Rust)                                  │
//! │  - Allocates unlimited virtual registers (%f0, %f1, ...)    │
//! │  - Tracks liveness for pressure reporting                   │
//! │  - Emits SSA-style PTX                                      │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                         PTX Source
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │  NVIDIA ptxas (JIT Compiler)                                │
//! │  - Graph coloring for physical register allocation          │
//! │  - Register spilling to local memory if needed              │
//! │  - Dead code elimination, constant folding, etc.            │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                         SASS Binary
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │  GPU Execution                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Why This Design?
//!
//! 1. **Pragmatism**: NVIDIA has 30+ years of GPU compiler optimization.
//!    Reimplementing graph coloring would be Muda (waste).
//!
//! 2. **PTX is SSA**: PTX's Static Single Assignment form with unlimited
//!    virtual registers is designed for this workflow.
//!
//! 3. **Focus**: Trueno focuses on algorithm correctness and high-level
//!    optimization (tiling, kernel fusion). Low-level optimization is
//!    delegated to ptxas.
//!
//! ## Register Pressure Monitoring
//!
//! While we don't do graph coloring, we DO track liveness for diagnostics:
//! - Warn developers when register pressure exceeds GPU limits (256/thread)
//! - Help identify kernels that may suffer from reduced occupancy
//!
//! Run with: `cargo run -p trueno-gpu --example register_allocation`

use trueno_gpu::ptx::{PtxKernel, PtxModule, PtxReg, PtxType};

fn main() {
    println!("=== trueno-gpu: Register Allocation Strategy ===\n");

    demonstrate_simple_kernel();
    demonstrate_complex_kernel();
    print_trade_offs();
    print_in_place_operations();
    print_summary();
}

/// Part 1: Simple Kernel - Low Register Pressure
fn demonstrate_simple_kernel() {
    println!("--- Part 1: Simple Kernel (Low Register Pressure) ---\n");

    let simple_kernel = PtxKernel::new("vector_add")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .param(PtxType::U64, "c_ptr")
        .param(PtxType::U32, "n")
        .build(|ctx| {
            // Each operation allocates a NEW virtual register (SSA style)
            let tid = ctx.special_reg(PtxReg::TidX); // %r0
            let ctaid = ctx.special_reg(PtxReg::CtaIdX); // %r1
            let ntid = ctx.special_reg(PtxReg::NtidX); // %r2
            let idx = ctx.mad_lo_u32(ctaid, ntid, tid); // %r3

            let n = ctx.load_param_u32("n"); // %r4
            let pred = ctx.setp_ge_u32(idx, n); // %p0
            ctx.branch_if(pred, "exit");

            let a_ptr = ctx.load_param_u64("a_ptr"); // %rd0
            let b_ptr = ctx.load_param_u64("b_ptr"); // %rd1
            let c_ptr = ctx.load_param_u64("c_ptr"); // %rd2

            let offset = ctx.mul_wide_u32(idx, 4); // %rd3
            let a_addr = ctx.add_u64(a_ptr, offset); // %rd4
            let b_addr = ctx.add_u64(b_ptr, offset); // %rd5
            let c_addr = ctx.add_u64(c_ptr, offset); // %rd6

            let a = ctx.ld_global_f32(a_addr); // %f0
            let b = ctx.ld_global_f32(b_addr); // %f1
            let c = ctx.add_f32(a, b); // %f2
            ctx.st_global_f32(c_addr, c);

            ctx.label("exit");
            ctx.ret();
        });

    let module = PtxModule::new()
        .version(8, 0)
        .target("sm_70")
        .add_kernel(simple_kernel);

    let ptx = module.emit();

    println!("Generated PTX ({} bytes):\n", ptx.len());
    println!("{ptx}");

    // Count virtual registers by type
    let f32_regs = ptx.matches("%f<").count();
    let u32_regs = ptx.matches("%r<").count();
    let u64_regs = ptx.matches("%rd<").count();
    let pred_regs = ptx.matches("%p<").count();

    println!("--- Register Usage Analysis ---");
    println!("Virtual register declarations:");
    println!("  .reg .f32  %f<N>   - {f32_regs} type(s)");
    println!("  .reg .u32  %r<N>   - {u32_regs} type(s)");
    println!("  .reg .u64  %rd<N>  - {u64_regs} type(s)");
    println!("  .reg .pred %p<N>   - {pred_regs} type(s)");
    println!();
}

/// Part 2: Complex Kernel - Higher Register Pressure
fn demonstrate_complex_kernel() {
    println!("--- Part 2: Complex Kernel (Higher Register Pressure) ---\n");

    // Demonstrate a kernel that uses more registers (e.g., dot product with unrolling)
    let complex_kernel = PtxKernel::new("dot_product_unrolled")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .param(PtxType::U64, "result_ptr")
        .param(PtxType::U32, "n")
        .build(|ctx| {
            let tid = ctx.special_reg(PtxReg::TidX);
            let ctaid = ctx.special_reg(PtxReg::CtaIdX);
            let ntid = ctx.special_reg(PtxReg::NtidX);
            let base_idx = ctx.mad_lo_u32(ctaid, ntid, tid);

            // Load pointers
            let a_ptr = ctx.load_param_u64("a_ptr");
            let b_ptr = ctx.load_param_u64("b_ptr");

            // Initialize accumulator (will be updated in-place)
            let acc = ctx.mov_f32_imm(0.0);

            // Unroll factor of 4 - each iteration uses multiple registers
            for i in 0..4 {
                let offset_val = ctx.mov_u32_imm(i);
                let idx = ctx.add_u32_reg(base_idx, offset_val);
                let byte_offset = ctx.mul_wide_u32(idx, 4);

                let a_addr = ctx.add_u64(a_ptr, byte_offset);
                let b_addr = ctx.add_u64(b_ptr, byte_offset);

                let a_val = ctx.ld_global_f32(a_addr);
                let b_val = ctx.ld_global_f32(b_addr);
                let prod = ctx.mul_f32(a_val, b_val);

                // In-place accumulation (avoids allocating new register)
                ctx.add_f32_inplace(acc, prod);
            }

            // Warp reduction using shuffle
            let shuffled = ctx.shfl_down_f32(acc, 16, 0xFFFF_FFFF);
            ctx.add_f32_inplace(acc, shuffled);
            let shuffled = ctx.shfl_down_f32(acc, 8, 0xFFFF_FFFF);
            ctx.add_f32_inplace(acc, shuffled);
            let shuffled = ctx.shfl_down_f32(acc, 4, 0xFFFF_FFFF);
            ctx.add_f32_inplace(acc, shuffled);
            let shuffled = ctx.shfl_down_f32(acc, 2, 0xFFFF_FFFF);
            ctx.add_f32_inplace(acc, shuffled);
            let shuffled = ctx.shfl_down_f32(acc, 1, 0xFFFF_FFFF);
            ctx.add_f32_inplace(acc, shuffled);

            // Lane 0 writes result
            let lane = ctx.special_reg(PtxReg::LaneId);
            let zero = ctx.mov_u32_imm(0);
            let is_lane_zero = ctx.setp_ge_u32(zero, lane);
            ctx.branch_if_not(is_lane_zero, "skip_store");

            let result_ptr = ctx.load_param_u64("result_ptr");
            ctx.st_global_f32(result_ptr, acc);

            ctx.label("skip_store");
            ctx.ret();
        });

    let complex_module = PtxModule::new()
        .version(8, 0)
        .target("sm_70")
        .add_kernel(complex_kernel);

    let complex_ptx = complex_module.emit();
    println!("Complex kernel PTX ({} bytes)\n", complex_ptx.len());

    // Extract register counts from declarations
    println!("--- Register Pressure Comparison ---\n");
    println!("Simple kernel (vector_add):");
    println!("  ~15 virtual registers total");
    println!("  Estimated occupancy: HIGH (registers << 256 limit)\n");

    println!("Complex kernel (dot_product_unrolled):");
    println!("  ~50+ virtual registers total (due to unrolling)");
    println!("  Estimated occupancy: HIGH (still well under limit)\n");
}

/// Part 3: Understanding the Trade-offs
fn print_trade_offs() {
    println!("--- Part 3: Architectural Trade-offs ---\n");

    println!("WHY WE DON'T DO GRAPH COLORING:");
    println!("================================");
    println!("1. ptxas already does this optimally");
    println!("2. PTX is designed as a virtual ISA (unlimited registers)");
    println!("3. Adding our own allocator would be redundant complexity\n");

    println!("WHAT WE DO TRACK:");
    println!("=================");
    println!("1. Liveness ranges - for pressure REPORTING (not allocation)");
    println!("2. Register counts by type - for diagnostics");
    println!("3. In-place operations - to reduce pressure in loops\n");

    println!("WHEN REGISTER PRESSURE MATTERS:");
    println!("================================");
    println!("- GPU threads share a register file per SM");
    println!("- More registers per thread = fewer concurrent threads");
    println!("- 256 registers/thread is the typical limit");
    println!("- >64 registers/thread starts impacting occupancy\n");

    println!("MITIGATIONS FOR HIGH-PRESSURE KERNELS:");
    println!("=======================================");
    println!("1. Use in-place operations (add_f32_inplace, fma_f32_inplace)");
    println!("2. Reduce unroll factors");
    println!("3. Use shared memory instead of registers for temps");
    println!("4. Split into multiple smaller kernels\n");
}

/// Part 4: Demonstrating In-Place Operations
fn print_in_place_operations() {
    println!("--- Part 4: In-Place Operations for Register Reuse ---\n");

    println!("SSA Style (allocates new register each time):");
    println!("  let sum = ctx.add_f32(acc, val);  // acc unchanged, sum is new\n");

    println!("In-Place Style (reuses existing register):");
    println!("  ctx.add_f32_inplace(acc, val);    // acc modified directly\n");

    println!("In-place operations are critical for:");
    println!("  - Loop counters: add_u32_inplace(i, 1)");
    println!("  - Accumulators: add_f32_inplace(sum, val)");
    println!("  - Running max: max_f32_inplace(max_val, new_val)");
    println!("  - FMA chains: fma_f32_inplace(acc, a, b)\n");
}

fn print_summary() {
    println!("=== Summary ===\n");
    println!("trueno-gpu treats PTX as a high-level IR and delegates");
    println!("physical register allocation to NVIDIA's ptxas compiler.");
    println!("This is a pragmatic design that leverages NVIDIA's 30+");
    println!("years of GPU compiler optimization.\n");

    println!("For details, see: book/src/architecture/ptx-register-allocation.md");
}
