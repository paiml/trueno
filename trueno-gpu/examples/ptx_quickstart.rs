//! PTX Code Generation Quickstart
//!
//! This example demonstrates how to generate PTX assembly from Rust
//! without requiring LLVM or nvcc.
//!
//! Run with: `cargo run -p trueno-gpu --example ptx_quickstart`

use trueno_gpu::ptx::{PtxKernel, PtxModule, PtxType};

fn main() {
    println!("=== trueno-gpu: Pure Rust PTX Generation ===\n");

    // Create a PTX module
    let module = PtxModule::new()
        .version(8, 0) // PTX ISA 8.0
        .target("sm_70") // Volta+
        .address_size(64); // 64-bit addressing

    // Build a vector addition kernel
    let kernel = PtxKernel::new("vector_add")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .param(PtxType::U64, "c_ptr")
        .param(PtxType::U32, "n")
        .build(|ctx| {
            // Calculate global thread ID
            let tid = ctx.special_reg(trueno_gpu::ptx::PtxReg::TidX);
            let ctaid = ctx.special_reg(trueno_gpu::ptx::PtxReg::CtaIdX);
            let ntid = ctx.special_reg(trueno_gpu::ptx::PtxReg::NtidX);
            let idx = ctx.mad_lo_u32(ctaid, ntid, tid);

            // Bounds check
            let n = ctx.load_param_u32("n");
            let pred = ctx.setp_ge_u32(idx, n);
            ctx.branch_if(pred, "exit");

            // Load pointers
            let a_ptr = ctx.load_param_u64("a_ptr");
            let b_ptr = ctx.load_param_u64("b_ptr");
            let c_ptr = ctx.load_param_u64("c_ptr");

            // Calculate addresses (idx * 4 bytes for f32)
            let offset = ctx.mul_wide_u32(idx, 4);
            let a_addr = ctx.add_u64(a_ptr, offset);
            let b_addr = ctx.add_u64(b_ptr, offset);
            let c_addr = ctx.add_u64(c_ptr, offset);

            // Load, add, store
            let a = ctx.ld_global_f32(a_addr);
            let b = ctx.ld_global_f32(b_addr);
            let c = ctx.add_f32(a, b);
            ctx.st_global_f32(c_addr, c);

            ctx.label("exit");
            ctx.ret();
        });

    // Add kernel to module and emit PTX
    let full_module = module.add_kernel(kernel);
    let ptx_source = full_module.emit();

    println!("Generated PTX source ({} bytes):\n", ptx_source.len());
    println!("{}", ptx_source);

    println!("\n=== Key Points ===");
    println!("- No LLVM required");
    println!("- No nvcc required");
    println!("- Pure Rust code generation");
    println!("- Fluent builder API");
}
