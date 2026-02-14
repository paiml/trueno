use super::*;
use crate::kernels::Kernel;

#[test]
fn test_f051_kernel_creation() {
    let kernel = Lz4WarpCompressKernel::new(1000);
    assert_eq!(kernel.batch_size(), 1000);
    assert_eq!(kernel.name(), "lz4_compress_warp");
}

#[test]
fn test_f051_grid_dimensions() {
    let kernel = Lz4WarpCompressKernel::new(1000);
    let (gx, gy, gz) = kernel.grid_dim();
    assert_eq!(gx, 250);
    assert_eq!(gy, 1);
    assert_eq!(gz, 1);
}

#[test]
fn test_f051_block_dimensions() {
    let kernel = Lz4WarpCompressKernel::new(1000);
    let (bx, by, bz) = kernel.block_dim();
    assert_eq!(bx, 128);
    assert_eq!(by, 1);
    assert_eq!(bz, 1);
}

#[test]
fn test_f052_shared_memory_size() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let smem = kernel.shared_memory_bytes();
    assert!(smem > 0);
    assert!(smem <= 100 * 1024);
}

#[test]
fn test_f053_ptx_generation_valid() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".version"), "Missing PTX version");
    assert!(ptx.contains(".target"), "Missing PTX target");
    assert!(ptx.contains(".entry"), "Missing entry point");
}

#[test]
fn test_f053_ptx_has_parameters() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains("input_batch"));
    assert!(ptx.contains("output_batch"));
    assert!(ptx.contains("output_sizes"));
    assert!(ptx.contains("batch_size"));
}

#[test]
fn test_f053_ptx_has_shared_memory() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains(".shared"));
}

#[test]
fn test_f054_barrier_safety() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let result = kernel.analyze_barrier_safety();
    assert!(
        result.is_safe,
        "LZ4 kernel should be barrier-safe: {:?}",
        result.violations
    );
}

#[test]
fn test_f055_kernel_name_deterministic() {
    let k1 = Lz4WarpCompressKernel::new(100);
    let k2 = Lz4WarpCompressKernel::new(100);
    assert_eq!(k1.name(), k2.name());
}

#[test]
fn test_f056_ptx_has_barrier_sync() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_f059_grid_covers_all_pages() {
    for batch_size in [1, 4, 5, 100, 1000, 18432] {
        let kernel = Lz4WarpCompressKernel::new(batch_size);
        let (gx, _, _) = kernel.grid_dim();
        let (bx, _, _) = kernel.block_dim();
        let warps_per_block = bx / 32;
        let total_warps = gx * warps_per_block;
        assert!(total_warps >= batch_size);
    }
}

#[test]
fn test_f060_module_emission() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let module = kernel.as_module();
    let ptx = module.emit();
    assert!(ptx.contains(".version 8.0"));
    assert!(ptx.contains(".target sm_89"));
}

#[test]
fn test_f061_ptx_validates_with_ptxas() {
    use std::io::Write;
    use std::process::Command;

    // Check if ptxas is available
    let ptxas_check = Command::new("which").arg("ptxas").output();
    if ptxas_check.is_err() || !ptxas_check.unwrap().status.success() {
        eprintln!("ptxas not available, skipping validation");
        return;
    }

    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Write PTX to temp file
    let mut tmpfile = std::env::temp_dir();
    tmpfile.push("lz4_compress_warp.ptx");
    let mut f = std::fs::File::create(&tmpfile).expect("Failed to create temp file");
    f.write_all(ptx.as_bytes()).expect("Failed to write PTX");

    // Validate with ptxas
    let output = Command::new("ptxas")
        .args(["-arch=sm_89", tmpfile.to_str().unwrap(), "-o", "/dev/null"])
        .output()
        .expect("Failed to run ptxas");

    // Clean up
    let _ = std::fs::remove_file(&tmpfile);

    assert!(
        output.status.success(),
        "ptxas validation failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

// =========================================================================
// WGSL Backend Tests (Dual-Backend Support)
// =========================================================================

#[test]
fn test_f062_wgsl_generation_valid() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let wgsl = kernel.emit_wgsl();
    assert!(wgsl.contains("@compute"), "Missing @compute attribute");
    assert!(wgsl.contains("@workgroup_size"), "Missing workgroup_size");
    assert!(
        wgsl.contains("workgroupBarrier"),
        "Missing workgroup barrier"
    );
}

#[test]
fn test_f062_wgsl_has_bindings() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let wgsl = kernel.emit_wgsl();
    assert!(
        wgsl.contains("@group(0) @binding(0)"),
        "Missing input binding"
    );
    assert!(
        wgsl.contains("@group(0) @binding(1)"),
        "Missing output binding"
    );
    assert!(
        wgsl.contains("@group(0) @binding(2)"),
        "Missing sizes binding"
    );
}

#[test]
fn test_f062_wgsl_has_shared_memory() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let wgsl = kernel.emit_wgsl();
    assert!(
        wgsl.contains("var<workgroup>"),
        "Missing workgroup shared memory"
    );
}

#[test]
fn test_f063_wgsl_batch_size_embedded() {
    let kernel = Lz4WarpCompressKernel::new(500);
    let wgsl = kernel.emit_wgsl();
    assert!(
        wgsl.contains("500u"),
        "Batch size should be embedded in WGSL"
    );
}

#[test]
fn test_f063_wgsl_has_entry_point() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let wgsl = kernel.emit_wgsl();
    assert!(
        wgsl.contains("fn lz4_compress_warp"),
        "Missing entry point function"
    );
}

#[test]
fn test_f064_wgsl_has_builtins() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let wgsl = kernel.emit_wgsl();
    assert!(
        wgsl.contains("@builtin(workgroup_id)"),
        "Missing workgroup_id builtin"
    );
    assert!(
        wgsl.contains("@builtin(local_invocation_id)"),
        "Missing local_invocation_id builtin"
    );
}

#[test]
fn test_f064_dual_backend_consistency() {
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();
    let wgsl = kernel.emit_wgsl();

    // Both should have the same logical structure
    assert!(
        ptx.contains("bar.sync") || ptx.contains("barrier"),
        "PTX missing barrier"
    );
    assert!(wgsl.contains("workgroupBarrier"), "WGSL missing barrier");

    // Both should have the same entry point name
    assert!(ptx.contains("lz4_compress_warp"));
    assert!(wgsl.contains("lz4_compress_warp"));
}

// =========================================================================
// GPU Kernel Integration Tests (F036-F050)
// =========================================================================

#[test]
fn test_f036_ptx_has_zero_page_detection() {
    // F036: GPU kernel detects zero pages for optimal compression
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should have OR operations for zero detection
    assert!(
        ptx.contains("or.b32"),
        "Missing OR operations for zero detection"
    );
    // Should have conditional branching for zero vs non-zero path
    assert!(
        ptx.contains("L_write_zero_size"),
        "Missing zero-size output path"
    );
    assert!(
        ptx.contains("L_after_size_write"),
        "Missing size write merge label"
    );
}

#[test]
fn test_f037_ptx_warp_reduction() {
    // F037: PTX uses warp-level reduction for zero detection
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should have multiple barrier syncs (load, reduction, store)
    let bar_count = ptx.matches("bar.sync").count();
    assert!(
        bar_count >= 3,
        "Should have at least 3 barrier syncs, found {}",
        bar_count
    );
}

#[test]
fn test_f038_zero_page_compressed_size() {
    // F038: Zero page should produce minimal output size
    // GPU kernel reports 20 bytes for zero pages (LZ4 sequence encoding)
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should have the compressed size constant (20 bytes for zero page)
    assert!(
        ptx.contains("20"),
        "Should reference compressed zero page size"
    );
}

#[test]
fn test_f039_page_id_calculation() {
    // F039: Page ID correctly calculated from block/thread indices
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should access blockIdx.x and threadIdx.x
    assert!(ptx.contains("%ctaid.x"), "Missing blockIdx.x access");
    assert!(ptx.contains("%tid.x"), "Missing threadIdx.x access");
}

#[test]
fn test_f040_lane_id_masking() {
    // F040: Lane ID correctly computed using mask
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should use AND with 31 for lane_id = threadIdx.x % 32
    assert!(ptx.contains("and.b32"), "Missing lane ID masking");
}

#[test]
fn test_f041_shared_memory_allocation() {
    // F041: Sufficient shared memory for page + hash table
    let kernel = Lz4WarpCompressKernel::new(100);
    let smem = kernel.shared_memory_bytes();

    // Need at least 4KB page + 8KB hash table per warp, times 4 warps
    let min_required = 4 * (PAGE_SIZE as usize + LZ4_HASH_SIZE as usize * 2);
    assert!(
        smem >= min_required,
        "Shared memory {} < required {}",
        smem,
        min_required
    );
}

#[test]
fn test_f042_bounds_check_present() {
    // F042: Kernel has bounds check for page_id < batch_size
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should have comparison instruction for bounds check
    // Uses setp.lt for in-bounds predicate (threads participate in barriers even when OOB)
    assert!(
        ptx.contains("setp.lt"),
        "Missing bounds check comparison (setp.lt)"
    );
    assert!(ptx.contains("L_exit"), "Missing exit label for OOB pages");
}

#[test]
fn test_f043_cooperative_load() {
    // F043: All 32 threads participate in loading 4KB page
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Each thread loads 128 bytes = 32 u32s = 8 chunks of 4 u32s
    // Should have many ld.global.u32 instructions
    let ld_count = ptx.matches("ld.global.u32").count();
    assert!(
        ld_count >= 32,
        "Should have many global loads, found {}",
        ld_count
    );
}

#[test]
fn test_f044_leader_thread_writes_size() {
    // F044: Only lane 0 (leader) writes the output size
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should have comparison for lane_id == 0
    assert!(ptx.contains("setp.eq"), "Missing leader thread check");
    assert!(
        ptx.contains("L_not_leader"),
        "Missing non-leader skip label"
    );
}

#[test]
fn test_f045_output_size_write() {
    // F045: Output size correctly written to sizes array
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should store to output_sizes array
    assert!(ptx.contains("st.global.u32"), "Missing size output store");
}

#[test]
fn test_f046_wgsl_zero_page_detection() {
    // F046: WGSL shader also has zero-page detection
    let kernel = Lz4WarpCompressKernel::new(100);
    let wgsl = kernel.emit_wgsl();

    // Should have OR operations for zero detection
    assert!(
        wgsl.contains("thread_or = thread_or |"),
        "Missing thread OR reduction"
    );
    // Should have conditional for zero page
    assert!(
        wgsl.contains("if (page_or == 0u)"),
        "Missing zero page check"
    );
    // Should output minimal size for zero pages
    assert!(wgsl.contains("20u"), "Missing compressed zero page size");
}

#[test]
fn test_f047_wgsl_reduction_barrier() {
    // F047: WGSL has proper barriers for reduction
    let kernel = Lz4WarpCompressKernel::new(100);
    let wgsl = kernel.emit_wgsl();

    // Should have multiple workgroup barriers
    let barrier_count = wgsl.matches("workgroupBarrier()").count();
    assert!(
        barrier_count >= 3,
        "Should have at least 3 barriers, found {}",
        barrier_count
    );
}

#[test]
fn test_f048_shared_memory_reduction() {
    // F048: Both PTX and WGSL use shared memory for reduction
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();
    let wgsl = kernel.emit_wgsl();

    // PTX uses generic addressing (after cvta.shared) for flexible warp offset handling
    // Check for generic store/load (st.u32/ld.u32 without state space = generic)
    assert!(
        ptx.contains("st.u32"),
        "PTX missing generic store for reduction"
    );
    assert!(
        ptx.contains("ld.u32"),
        "PTX missing generic load for reduction"
    );
    // Verify shared memory is declared and cvta is used to get generic address
    // cvta.shared converts shared->generic; cvta.to.shared converts generic->shared
    assert!(
        ptx.contains(".shared"),
        "PTX missing shared memory declaration"
    );
    assert!(
        ptx.contains("cvta.shared"),
        "PTX missing cvta for shared->generic"
    );

    // WGSL should use smem for reduction
    assert!(
        wgsl.contains("smem[reduction_idx]"),
        "WGSL missing shared memory reduction"
    );
}

#[test]
fn test_f049_page_data_integrity() {
    // F049: Page data correctly passed through shared memory
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Should have matching global loads and stores
    let global_loads = ptx.matches("ld.global.u32").count();
    let global_stores = ptx.matches("st.global.u32").count();

    // Should have balanced load/store for page data
    assert!(global_loads >= 32, "Need at least 32 global loads for 4KB");
    assert!(
        global_stores >= 32,
        "Need at least 32 global stores for 4KB"
    );
}

#[test]
fn test_f050_kernel_determinism() {
    // F050: Kernel generation is structurally deterministic
    // Note: PTX register numbers may vary between invocations due to allocator state,
    // but the WGSL (which uses names, not registers) should be exactly deterministic.
    let k1 = Lz4WarpCompressKernel::new(100);
    let k2 = Lz4WarpCompressKernel::new(100);

    // WGSL should be exactly deterministic (uses named variables)
    let wgsl1 = k1.emit_wgsl();
    let wgsl2 = k2.emit_wgsl();
    assert_eq!(wgsl1, wgsl2, "WGSL should be deterministic");

    // PTX should have same instruction count and structure
    let ptx1 = k1.emit_ptx();
    let ptx2 = k2.emit_ptx();

    // Same number of instructions
    let instr_count_1 = ptx1
        .lines()
        .filter(|l| l.trim().starts_with(|c: char| c.is_alphabetic()))
        .count();
    let instr_count_2 = ptx2
        .lines()
        .filter(|l| l.trim().starts_with(|c: char| c.is_alphabetic()))
        .count();
    assert_eq!(
        instr_count_1, instr_count_2,
        "PTX instruction count should match"
    );

    // Same labels
    assert_eq!(
        ptx1.matches("L_exit").count(),
        ptx2.matches("L_exit").count()
    );
    assert_eq!(
        ptx1.matches("L_not_leader").count(),
        ptx2.matches("L_not_leader").count()
    );
}

// =========================================================================
// GPU LZ4 FULL COMPRESSION TESTS (TDD - These define requirements)
// =========================================================================

#[test]
fn test_gpu_lz4_ptx_has_hash_table() {
    // REQ-LZ4-001: PTX kernel must have hash table for match finding
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Hash table should be in shared memory (8KB per warp = 4096 entries x 2 bytes)
    // Check for hash computation using the LZ4 hash multiplier (2654435761 = 0x9E3779B1)
    assert!(
        ptx.contains("0x9e3779b1") || ptx.contains("2654435761") || ptx.contains("hash"),
        "PTX must have LZ4 hash computation (mul by 0x9E3779B1)"
    );
}

#[test]
fn test_gpu_lz4_ptx_has_match_finding() {
    // REQ-LZ4-002: PTX kernel must find matches of >= 4 bytes
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Match finding requires labeled branches for match logic
    assert!(
        ptx.contains("match") || ptx.contains("L_found_match") || ptx.contains("L_check_match"),
        "PTX must have match finding logic with labeled branches"
    );
}

#[test]
fn test_gpu_lz4_ptx_has_sequence_encoding() {
    // REQ-LZ4-003: PTX kernel must encode LZ4 sequences (token + literals + offset + matchlen)
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Sequence encoding requires labeled logic
    assert!(
        ptx.contains("token") || ptx.contains("L_encode") || ptx.contains("L_write_sequence"),
        "PTX must have LZ4 sequence encoding logic"
    );
}

#[test]
fn test_gpu_lz4_ptx_has_output_buffer_management() {
    // REQ-LZ4-004: PTX kernel must manage output buffer correctly
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // For now, check that there's some form of dynamic size tracking
    let has_dynamic_size =
        ptx.contains("out_pos") || ptx.contains("L_compress") || ptx.contains("compressed_len");
    assert!(
        has_dynamic_size,
        "PTX must track output buffer position dynamically for compression"
    );
}

#[test]
fn test_gpu_lz4_kernel_has_compression_loop() {
    // REQ-LZ4-006: GPU kernel must have main compression loop
    let kernel = Lz4WarpCompressKernel::new(100);
    let ptx = kernel.emit_ptx();

    // Look for loop structure in PTX
    let has_compress_loop = ptx.contains("L_compress_loop")
        || ptx.contains("L_main_loop")
        || (ptx.contains("bra") && ptx.contains("L_loop"));

    assert!(
        has_compress_loop,
        "GPU kernel must have main compression loop (L_compress_loop or similar)"
    );
}
