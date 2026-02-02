//! PTX Verification and Mutation Demo
//!
//! Demonstrates PTX structural verification and mutation operators
//! for GPU kernel testing.
//!
//! Run with: cargo run --example ptx_verifier_demo

use trueno_cuda_edge::ptx_poison::{
    default_mutators, PtxMutator, PtxVerifier, MINIMAL_VALID_PTX,
};

fn main() {
    println!("=== PTX Verification and Mutation Demo ===\n");

    let verifier = PtxVerifier::new();

    // 1. Valid PTX Verification
    println!("1. Valid PTX Verification");
    println!("   ───────────────────────");
    println!("   Source:");
    for line in MINIMAL_VALID_PTX.lines() {
        println!("   │ {}", line);
    }

    match verifier.verify(MINIMAL_VALID_PTX) {
        Ok(verified) => {
            println!("   Result: ✓ VERIFIED");
            println!("   Source length: {} bytes", verified.source().len());
        }
        Err(e) => println!("   Result: ✗ FAILED - {}", e),
    }

    println!();

    // 2. Structural Error Detection
    println!("2. Structural Error Detection");
    println!("   ───────────────────────────");

    let test_cases = [
        ("Empty source", ""),
        ("Missing .version", ".target sm_80\n.address_size 64\n.entry k() { ret; }"),
        ("Missing .target", ".version 7.0\n.address_size 64\n.entry k() { ret; }"),
        ("Missing .address_size", ".version 7.0\n.target sm_80\n.entry k() { ret; }"),
        ("Missing entry point", ".version 7.0\n.target sm_80\n.address_size 64\n"),
        ("Unbalanced braces", ".version 7.0\n.target sm_80\n.address_size 64\n.entry k() {"),
    ];

    for (name, source) in test_cases {
        let errors = verifier.check_all(source);
        if errors.is_empty() {
            println!("   {} → ✓ No errors", name);
        } else {
            print!("   {} → ✗ ", name);
            for (i, e) in errors.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{}", e);
            }
            println!();
        }
    }

    println!();

    // 3. Mutation Operators
    println!("3. Mutation Operators");
    println!("   ───────────────────");

    let mutators = default_mutators();
    println!("   Available mutators ({}):", mutators.len());
    for m in &mutators {
        println!("   │ {:?}", m);
    }

    println!();

    // 4. Mutation Application
    println!("4. Mutation Application");
    println!("   ─────────────────────");

    let test_mutations = [
        (PtxMutator::FlipAddSub, "add.f32 %f1, %f2, %f3;"),
        (PtxMutator::FlipMulDiv, "mul.f32 %f1, %f2, %f3;"),
        (PtxMutator::InvertPredicate, "setp.lt.f32 %p1, %f1, %f2;"),
        (PtxMutator::WidenPrecision, "add.f32 %f1, %f2, %f3;"),
        (PtxMutator::SwapMemorySpace, "ld.shared.f32 %f1, [%r1];"),
    ];

    for (mutator, source) in test_mutations {
        println!("\n   {:?}:", mutator);
        println!("   Before: {}", source);
        match mutator.apply(source) {
            Some(mutated) => println!("   After:  {}", mutated),
            None => println!("   After:  (no match)"),
        }
    }

    println!();

    // 5. Barrier Removal
    println!("5. Barrier Removal (Synchronization Bug Injection)");
    println!("   ─────────────────────────────────────────────────");

    let kernel_with_barrier = "\
.version 7.0
.target sm_80
.address_size 64
.entry sync_kernel() {
    ld.shared.f32 %f1, [%r1];
    bar.sync 0;
    st.shared.f32 [%r2], %f1;
    ret;
}";

    println!("   Before:");
    for line in kernel_with_barrier.lines() {
        println!("   │ {}", line);
    }

    if let Some(mutated) = PtxMutator::RemoveBarrier.apply(kernel_with_barrier) {
        println!("\n   After (bar.sync removed):");
        for line in mutated.lines() {
            println!("   │ {}", line);
        }
    }

    println!("\n=== Demo Complete ===");
}
