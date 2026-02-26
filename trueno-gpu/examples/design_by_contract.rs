//! Design by Contract in Trueno-GPU
//!
//! Demonstrates PTX-level contracts: SM target validation, PTX version
//! validation, module construction, and kernel parameter type safety.
//!
//! Run with: `cargo run --example design_by_contract`

use trueno_gpu::ptx::{self, PtxKernel, PtxModule, PtxType};

/// Demonstrate SM target validation (sm_70+ required).
fn demo_sm_target_validation() {
    println!("1. SM target validation (sm_70+ required):");
    for target in &["sm_70", "sm_80", "sm_86", "sm_90"] {
        match ptx::validate_target(target) {
            Ok(()) => println!("   PASS: {target}"),
            Err(e) => println!("   FAIL: {e}"),
        }
    }
    for target in &["sm_50", "sm_61", "compute_70", ""] {
        match ptx::validate_target(target) {
            Ok(()) => println!("   UNEXPECTED PASS: {target}"),
            Err(e) => println!("   REJECTED (expected): {e}"),
        }
    }
}

/// Demonstrate PTX version validation (>= 7.0 required).
fn demo_ptx_version_validation() {
    println!("\n2. PTX version validation (>= 7.0 required):");
    for (major, minor) in &[(7, 0), (8, 0), (8, 5), (9, 0)] {
        match ptx::validate_version(*major, *minor) {
            Ok(()) => println!("   PASS: {major}.{minor}"),
            Err(e) => println!("   FAIL: {e}"),
        }
    }
    for (major, minor) in &[(6, 5), (6, 0), (5, 0)] {
        match ptx::validate_version(*major, *minor) {
            Ok(()) => println!("   UNEXPECTED PASS: {major}.{minor}"),
            Err(e) => println!("   REJECTED (expected): {e}"),
        }
    }
}

/// Demonstrate module-level validation.
fn demo_module_validation() {
    println!("\n3. Module-level validation:");
    let good_module = PtxModule::new().version(8, 0).target("sm_80").address_size(64);
    match good_module.validate() {
        Ok(()) => println!("   PASS: sm_80, PTX 8.0, 64-bit"),
        Err(e) => println!("   FAIL: {e}"),
    }

    let bad_module = PtxModule::new().version(6, 0).target("sm_50");
    match bad_module.validate() {
        Ok(()) => println!("   UNEXPECTED PASS"),
        Err(e) => println!("   REJECTED (expected): {e}"),
    }
}

/// Demonstrate PTX type system contracts.
fn demo_ptx_type_system() {
    println!("\n4. PTX type system:");
    let types = [
        PtxType::U32,
        PtxType::U64,
        PtxType::S32,
        PtxType::F16,
        PtxType::F32,
        PtxType::F64,
        PtxType::V2F32,
        PtxType::V4F32,
    ];
    for ty in &types {
        println!(
            "   {} -> {} bytes, register prefix: {}",
            ty.to_ptx_string(),
            ty.size_bytes(),
            ty.register_prefix()
        );
    }
}

/// Demonstrate full module emit with kernel and assertions.
fn demo_full_module_emit() {
    println!("\n5. Kernel shared memory contract:");
    let kernel = PtxKernel::new("gemm_tiled").shared_memory(4096);
    println!(
        "   Kernel 'gemm_tiled': {} bytes shared memory declared",
        kernel.shared_memory_bytes()
    );

    println!("\n6. Full PTX module emission:");
    let kernel = PtxKernel::new("vector_add")
        .param(PtxType::U64, "a_ptr")
        .param(PtxType::U64, "b_ptr")
        .param(PtxType::U64, "c_ptr")
        .param(PtxType::U32, "n");

    let module = PtxModule::new().version(8, 0).target("sm_80").address_size(64).add_kernel(kernel);

    let ptx_source = module.emit();
    println!("   Generated {} bytes of PTX", ptx_source.len());
    assert!(ptx_source.contains(".version 8.0"), "must contain version directive");
    assert!(ptx_source.contains(".target sm_80"), "must contain target directive");
    assert!(ptx_source.contains(".address_size 64"), "must contain address size");
    assert!(ptx_source.contains(".visible .entry vector_add"), "must contain kernel entry");
    assert!(ptx_source.contains(".param .u64 a_ptr"), "must contain typed parameters");
    println!("   All PTX structure assertions passed");
}

fn main() {
    println!("=== Trueno-GPU Design by Contract ===\n");
    demo_sm_target_validation();
    demo_ptx_version_validation();
    demo_module_validation();
    demo_ptx_type_system();
    demo_full_module_emit();
    println!("\n=== All contract demonstrations complete ===");
}
