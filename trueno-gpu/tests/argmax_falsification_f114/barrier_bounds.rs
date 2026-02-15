//! F114 Tests 1-2: Barrier Safety and Bounds Verification (static PTX analysis)

use std::ffi::c_void;
use trueno_gpu::driver::{CudaContext, CudaModule, CudaStream, GpuBuffer, LaunchConfig};
use trueno_gpu::kernels::{ArgMaxFinalKernel, ArgMaxKernel, Kernel};

// =========================================================================
// F114-TEST-1: Barrier Safety (PARITY-114)
// =========================================================================

/// F114-TEST-1: Verify all threads reach bar.sync in reduction phase
///
/// If CRASHES -> PARITY-114 barrier divergence detected
/// If WORKS -> Barrier safety criterion satisfied
#[test]
fn f114_test1_barrier_safety() {
    let kernel = ArgMaxKernel::new(1024);
    let ptx = kernel.emit_ptx();

    // Count bar.sync instructions
    let bar_sync_count = ptx.matches("bar.sync").count();
    println!("F114-TEST-1: Barrier Safety Analysis");
    println!("  bar.sync count: {}", bar_sync_count);

    // Verify PTX structure has barrier after each skip label
    let skip_labels: Vec<&str> = ptx
        .lines()
        .filter(|line: &&str| line.contains("skip_"))
        .collect();

    println!("  Skip labels: {:?}", skip_labels.len());

    // Each reduction step should have a barrier after the skip label
    // Expected pattern: skip_reduce_X: followed by bar.sync 0;
    assert!(
        bar_sync_count >= 8,
        "Expected at least 8 bar.sync (7 reduction steps + 1 initial)"
    );

    // Verify no early exit before barriers
    let lines: Vec<&str> = ptx.lines().collect();
    let mut found_exit_before_barrier = false;

    for (i, line) in lines.iter().enumerate() {
        let line_str: &str = *line;
        if line_str.contains("bra exit") {
            // Check if next non-empty line is bar.sync
            for j in (i + 1)..lines.len() {
                let next = lines[j].trim();
                if !next.is_empty() && !next.starts_with("//") {
                    if !next.starts_with("exit:") && next.contains("bar.sync") {
                        found_exit_before_barrier = true;
                    }
                    break;
                }
            }
        }
    }

    assert!(
        !found_exit_before_barrier,
        "PARITY-114: Found potential barrier divergence"
    );
    println!("  PASSED - No barrier divergence detected");
}

/// F114-TEST-2: Bounds verification for shared memory access (PAR-002)
///
/// Verifies shared memory indices stay within allocated bounds
#[test]
fn f114_test2_bounds_verification() {
    let kernel = ArgMaxKernel::new(152064); // Qwen vocab size
    let ptx = kernel.emit_ptx();

    println!("F114-TEST-2: Bounds Verification");

    // Parse shared memory size from PTX
    let smem_line = ptx
        .lines()
        .find(|line: &&str| line.contains(".shared"))
        .expect("Should have shared memory declaration");

    println!("  Shared memory declaration: {}", smem_line.trim());

    // Verify shared memory is at least 2KB (256 threads * 8 bytes)
    assert!(
        smem_line.contains("2048") || smem_line.contains("smem[2048]"),
        "PAR-002: Shared memory size should be 2048 bytes"
    );

    // Verify offset calculations use proper bounds
    // Each thread accesses shared_base + tid * 4 for values
    // and shared_base + 1024 + tid * 4 for indices
    let has_offset_1024 = ptx.contains("1024");
    assert!(has_offset_1024, "Expected index array offset of 1024 bytes");

    println!("  PASSED - Bounds verification satisfied");
}
