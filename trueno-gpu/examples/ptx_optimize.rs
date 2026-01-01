//! PTX Optimization Passes Demo
//!
//! Demonstrates the NVIDIA CUDA Tile IR-aligned optimization passes:
//! - FMA Fusion: mul + add → fma
//! - Loop Splitting: profitability analysis for conditional loops
//! - Token-Based Ordering (TKO): memory dependency tracking
//! - Tile Validation: power-of-two constraints for GPU efficiency
//!
//! Run with: cargo run --example ptx_optimize

use trueno_gpu::ptx::optimize::{fma_fusion, loop_split, tile_validation, tko};
use trueno_gpu::ptx::{CmpOp, Operand, PtxInstruction, PtxOp, PtxType, VirtualReg, WmmaShape};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     PTX Optimization Passes (NVIDIA CUDA Tile IR Aligned)    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    demo_fma_fusion();
    demo_loop_splitting();
    demo_tko();
    demo_tile_validation();

    println!("\n✅ All optimization demos completed successfully!");
}

/// Demonstrate FMA (Fused Multiply-Add) fusion
fn demo_fma_fusion() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("1️⃣  FMA FUSION PASS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create mul + add pattern that can be fused
    let r0 = VirtualReg::new(0, PtxType::F32);
    let r1 = VirtualReg::new(1, PtxType::F32);
    let r2 = VirtualReg::new(2, PtxType::F32);
    let r3 = VirtualReg::new(3, PtxType::F32);

    let mul = PtxInstruction::new(PtxOp::Mul, PtxType::F32)
        .dst(Operand::Reg(r2.clone()))
        .src(Operand::Reg(r0.clone()))
        .src(Operand::Reg(r1.clone()));

    let add = PtxInstruction::new(PtxOp::Add, PtxType::F32)
        .dst(Operand::Reg(r3.clone()))
        .src(Operand::Reg(r2.clone()))
        .src(Operand::ImmF32(1.0));

    let instructions = vec![mul, add];
    println!("Input:  {} instructions (mul + add)", instructions.len());
    println!("        r2 = r0 * r1");
    println!("        r3 = r2 + 1.0\n");

    let fused = fma_fusion::pass(instructions);
    println!("Output: {} instruction(s)", fused.len());
    println!("        r3 = fma(r0, r1, 1.0)  // Single FMA instruction\n");

    println!("Benefit: Reduced latency, single rounding operation");
    println!("Reference: Click & Paleczny (1995) SSA pattern matching\n");
}

/// Demonstrate loop splitting analysis
fn demo_loop_splitting() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("2️⃣  LOOP SPLITTING PASS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Test profitability with heavy vs light operations
    let heavy_op = PtxInstruction::new(PtxOp::Ld, PtxType::F32);
    let light_op = PtxInstruction::new(PtxOp::Add, PtxType::F32);

    println!("Heavy operations (Ld, St, WmmaMma, WmmaLoad*, WmmaStoreD):");
    println!(
        "  is_split_profitable([Ld], threshold=10) = {}",
        loop_split::is_split_profitable(&[heavy_op.clone()], 10)
    );

    println!("\nLight operations (Add, Mul, etc.):");
    println!(
        "  is_split_profitable([Add], threshold=10) = {}",
        loop_split::is_split_profitable(&[light_op.clone()], 10)
    );

    // Split point alignment demo
    println!("\nSplit point alignment for non-unit step sizes:");
    println!(
        "  align_split_point(5, 0, 4) = {} (aligned to step boundary)",
        loop_split::align_split_point(5, 0, 4)
    );
    println!(
        "  align_split_point(8, 0, 4) = {} (already aligned)",
        loop_split::align_split_point(8, 0, 4)
    );

    // LoopPredicate conversion
    println!("\nLoop predicate conversion:");
    println!(
        "  CmpOp::Lt -> {:?}",
        loop_split::LoopPredicate::from_cmp_op(CmpOp::Lt)
    );
    println!(
        "  CmpOp::Ge -> {:?}",
        loop_split::LoopPredicate::from_cmp_op(CmpOp::Ge)
    );

    println!("\nBenefit: Eliminates branch divergence in GPU warps");
    println!("Reference: NVIDIA LoopSplit.cpp (CUDA Toolkit 13.1)\n");
}

/// Demonstrate Token-Based Ordering (TKO)
fn demo_tko() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("3️⃣  TOKEN-BASED ORDERING (TKO)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create tokens for memory operations
    let t1 = tko::Token::new();
    let t2 = tko::Token::new();
    let t3 = tko::Token::new();

    println!("Token creation (unique IDs):");
    println!("  t1.id() = {}", t1.id());
    println!("  t2.id() = {}", t2.id());
    println!("  t3.id() = {}", t3.id());

    // Join tokens
    let joined = tko::join_tokens(&[t1, t2, t3]);
    println!("\nJoin tokens (synchronization point):");
    println!("  join_tokens([t1, t2, t3]).id() = {}", joined.id());

    // Memory ordering semantics
    println!("\nMemory ordering semantics:");
    println!(
        "  MemoryOrdering::Weak    -> \"{}\"",
        tko::MemoryOrdering::Weak.to_ptx_modifier()
    );
    println!(
        "  MemoryOrdering::Relaxed -> \"{}\"",
        tko::MemoryOrdering::Relaxed.to_ptx_modifier()
    );
    println!(
        "  MemoryOrdering::Acquire -> \"{}\"",
        tko::MemoryOrdering::Acquire.to_ptx_modifier()
    );
    println!(
        "  MemoryOrdering::Release -> \"{}\"",
        tko::MemoryOrdering::Release.to_ptx_modifier()
    );

    // Memory scopes
    println!("\nMemory scopes:");
    println!(
        "  MemoryScope::Block   -> \"{}\"",
        tko::MemoryScope::Block.to_ptx_scope()
    );
    println!(
        "  MemoryScope::Device  -> \"{}\"",
        tko::MemoryScope::Device.to_ptx_scope()
    );
    println!(
        "  MemoryScope::System  -> \"{}\"",
        tko::MemoryScope::System.to_ptx_scope()
    );

    // Token graph with cycle detection
    let mut graph = tko::TokenGraph::new();
    let ta = tko::Token::new();
    let tb = tko::Token::new();
    let tc = tko::Token::new();

    graph.create_token(ta);
    graph.create_token(tb);
    graph.create_token(tc);
    graph.add_dependency(tb, ta);
    graph.add_dependency(tc, tb);

    println!("\nToken graph (deadlock detection):");
    println!("  Linear chain: ta -> tb -> tc");
    println!("  has_cycle() = {} (safe)", graph.has_cycle());

    // Create a cycle
    graph.add_dependency(ta, tc);
    println!("\n  After adding tc -> ta:");
    println!("  has_cycle() = {} (DEADLOCK!)", graph.has_cycle());

    println!("\nBenefit: Enables compiler-driven barrier elimination");
    println!("Reference: NVIDIA memory_consistency_ops.mlir\n");
}

/// Demonstrate tile validation
fn demo_tile_validation() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("4️⃣  TILE VALIDATION");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Valid shapes
    let valid_shapes: &[&[usize]] = &[&[16, 16], &[32, 32], &[64, 64], &[128, 128]];
    println!("Valid tile shapes (power of two):");
    for shape in valid_shapes {
        match tile_validation::validate_shape(shape) {
            Ok(()) => println!("  {:?} OK", shape),
            Err(e) => println!("  {:?} ERROR: {}", shape, e),
        }
    }

    // Invalid shapes
    println!("\nInvalid tile shapes:");
    let invalid_shapes: &[&[usize]] = &[&[17, 16], &[100, 100], &[0, 32]];
    for shape in invalid_shapes {
        match tile_validation::validate_shape(shape) {
            Ok(()) => println!("  {:?} OK", shape),
            Err(e) => println!("  {:?} -> {}", shape, e),
        }
    }

    // WMMA shapes
    println!("\nWMMA (Tensor Core) shapes:");
    let wmma_16x16x16 = WmmaShape::M16N16K16;
    let wmma_8x32x16 = WmmaShape::M8N32K16;
    let wmma_invalid = WmmaShape { m: 24, n: 24, k: 16 };

    match tile_validation::validate_wmma_shape(&wmma_16x16x16) {
        Ok(()) => println!("  m16n16k16 OK (standard)"),
        Err(e) => println!("  m16n16k16 ERROR: {}", e),
    }
    match tile_validation::validate_wmma_shape(&wmma_8x32x16) {
        Ok(()) => println!("  m8n32k16  OK (alternate)"),
        Err(e) => println!("  m8n32k16  ERROR: {}", e),
    }
    match tile_validation::validate_wmma_shape(&wmma_invalid) {
        Ok(()) => println!("  m24n24k16 OK"),
        Err(e) => println!("  m24n24k16 -> {}", e),
    }

    println!("\nBenefit: Prevents register pressure issues at compile time");
    println!("Reference: Volkov & Demmel (2008) GPU optimization\n");
}
