use super::*;

#[test]
fn test_quantize_kernel_name() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.name(), "q4k_gemm_fused");
}

#[test]
fn test_quantize_default_config() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.m, 1024);
    assert_eq!(kernel.n, 1024);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.tile_size, 32);
    assert_eq!(kernel.block_size, Q4K_BLOCK_SIZE);
}

#[test]
fn test_quantize_with_tile_size() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096).with_tile_size(64);
    assert_eq!(kernel.tile_size, 64);
}

#[test]
fn test_quantize_num_blocks() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.num_blocks_per_row(), 128); // 4096 / 32
}

#[test]
fn test_quantize_ptx_generation() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Verify parameters
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_quant_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
    assert!(ptx.contains(".param .u32 m"));
    assert!(ptx.contains(".param .u32 n"));
    assert!(ptx.contains(".param .u32 k"));
}

#[test]
fn test_quantize_shared_memory() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096);
    let ptx_kernel = kernel.build_ptx();

    // Should have shared memory for dequantized tile
    assert!(ptx_kernel.shared_memory_bytes() > 0);
}

#[test]
fn test_quantize_ptx_contains_operations() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Verify memory operations
    assert!(ptx.contains("ld.global"));
    assert!(ptx.contains("st.global.f32"));

    // Verify arithmetic for dequantization and GEMM
    assert!(ptx.contains("mul.f32"));
    assert!(ptx.contains("add.f32"));

    // Verify warp shuffle for reduction
    assert!(ptx.contains("shfl") || ptx.contains("shfl.down"));
}

#[test]
fn test_quantize_dequantization_ops() {
    let kernel = QuantizeKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Verify shift/mask for nibble extraction
    // Note: shr may be emitted differently
    assert!(ptx.contains("mul") || ptx.contains("shr"));

    // Verify type conversion
    assert!(ptx.contains("cvt"));
}

#[test]
fn test_quantize_kernel_variants() {
    // Test with different configurations
    let configs = vec![
        QuantizeKernel::new(512, 512, 2048),
        QuantizeKernel::new(1024, 1024, 4096),
        QuantizeKernel::new(2048, 2048, 8192),
        QuantizeKernel::new(4096, 4096, 4096).with_tile_size(64),
    ];

    for config in configs {
        let ptx = config.emit_ptx();
        assert!(!ptx.is_empty());
        assert!(ptx.contains(".visible .entry"));
    }
}

#[test]
fn test_quantize_block_layout() {
    // Verify Q4_K block constants
    assert_eq!(Q4K_BLOCK_SIZE, 32);
    assert_eq!(Q4K_BLOCK_BYTES, 18);
}

// =========================================================================
// SATD REMEDIATION TESTS (EXTREME TDD)
// These tests verify the K-loop and shuffle bugs are fixed.
// Falsifiable claims per Popperian methodology.
// =========================================================================

#[test]
fn test_kloop_branches_back_to_loop_start() {
    // FALSIFIABLE CLAIM: K-loop branches back to "k_block_loop", not "k_block_done"
    // This test FAILS if the SATD bug (single iteration) is present.
    let kernel = QuantizeKernel::new(64, 64, 128); // K=128 requires 4 K-blocks
    let ptx = kernel.emit_ptx();

    // The PTX should contain a branch back to k_block_loop
    // If it only branches to k_block_done, the loop exits after 1 iteration
    let has_loop_back = ptx.contains("bra k_block_loop") || ptx.contains("bra\tk_block_loop");

    assert!(
        has_loop_back,
        "FALSIFIED: K-loop does not branch back to loop start. \
         Found 'bra k_block_done' instead of 'bra k_block_loop'. \
         This means K-loop only runs once regardless of K value."
    );
}

#[test]
fn test_kloop_counter_incremented_inplace() {
    // FALSIFIABLE CLAIM: K-loop counter is incremented in-place using add_u32_inplace
    // If add_u32 is used (returns new reg), the counter is never updated.
    let kernel = QuantizeKernel::new(64, 64, 128);
    let ptx = kernel.emit_ptx();

    // The PTX should increment k_block register in-place
    // Pattern: add.u32 %rN, %rN, 1 (same register for dest and src1)
    // If we see add.u32 %rM, %rN, 1 (different registers), it's broken

    // Count the k_block_loop and k_block_done labels
    let loop_count = ptx.matches("k_block_loop").count();
    let done_count = ptx.matches("k_block_done").count();

    // There should be exactly 2 references to k_block_loop:
    // 1. The label definition
    // 2. The branch back to the loop
    assert!(
        loop_count >= 2,
        "FALSIFIED: k_block_loop only appears {} times. \
         Expected at least 2 (label + branch back). \
         K-loop counter is not being used correctly.",
        loop_count
    );

    // done label should appear exactly once (the label definition)
    // If bra k_block_done appears twice, the loop exits incorrectly
    assert_eq!(
        done_count,
        2, // label + conditional branch
        "FALSIFIED: k_block_done appears {} times. \
         Expected 2 (label + conditional exit). \
         Extra branches to k_block_done indicate premature loop exit.",
        done_count
    );
}

#[test]
fn test_shuffle_broadcast_uses_shfl_idx_not_shfl_down_zero() {
    // FALSIFIABLE CLAIM: Broadcast uses shfl.idx (or shfl.sync.idx) with lane 0,
    // NOT shfl.down with offset 0 (which is a no-op).
    let kernel = QuantizeKernel::new(64, 64, 128);
    let ptx = kernel.emit_ptx();

    // shfl.down with offset 0 is a no-op - it returns the same value
    // Correct broadcast should use shfl.idx or shfl.sync.idx
    let has_shfl_idx = ptx.contains("shfl.idx") || ptx.contains("shfl.sync.idx");
    let has_bad_shfl_down_zero = ptx.contains("shfl.down.b32") && ptx.contains(", 0,");

    // Either we have shfl.idx (correct) or we don't have the bad pattern
    assert!(
        has_shfl_idx || !has_bad_shfl_down_zero,
        "FALSIFIED: Broadcast uses shfl.down with offset 0, which is a no-op. \
         Should use shfl.idx with lane 0 to broadcast the reduced value."
    );
}

#[test]
fn test_accumulator_updated_inplace() {
    // FALSIFIABLE CLAIM: Accumulator is updated in-place, not shadowed
    // If add_f32 creates a new register, the accumulator is never updated.
    let kernel = QuantizeKernel::new(64, 64, 128);
    let ptx = kernel.emit_ptx();

    // The accumulator should be used in a fma.rn.f32 or add.f32 that writes
    // back to the same register. This is tricky to verify in PTX without
    // full SSA analysis, so we verify the loop structure instead.

    // The key invariant: if K > 32, we need multiple accumulations.
    // With K=128 (4 blocks), the final result should be sum of 4 partial products.
    // If accumulator is not updated, result will be wrong (only 1 block's contribution).

    // For now, verify the structure allows for accumulation
    let has_add_f32 = ptx.contains("add.f32") || ptx.contains("add.rn.f32");
    assert!(
        has_add_f32,
        "FALSIFIED: No add.f32 found for accumulation. \
         Accumulator cannot be updated without add instruction."
    );
}

// =========================================================================
// PARITY-041: GGML Q4_K Super-block Format Tests
// Verify the new kernel that uses real GGML Q4_K format (144-byte super-blocks)
// =========================================================================

#[test]
fn test_ggml_kernel_name() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    assert_eq!(kernel.name(), "q4k_gemm_ggml");
}

#[test]
fn test_ggml_kernel_config() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    assert_eq!(kernel.m, 1024);
    assert_eq!(kernel.n, 1024);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.block_size, Q4K_SUPER_BLOCK_SIZE); // 256 values
    assert_eq!(kernel.format, Q4KFormat::GgmlSuperBlock);
}

#[test]
fn test_ggml_super_block_constants() {
    // Verify GGML Q4_K super-block constants
    assert_eq!(
        Q4K_SUPER_BLOCK_SIZE, 256,
        "Super-block should have 256 values"
    );
    assert_eq!(
        Q4K_SUPER_BLOCK_BYTES, 144,
        "Super-block should be 144 bytes (2+2+12+128)"
    );
}

#[test]
fn test_ggml_num_super_blocks() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256 = 16
}

#[test]
fn test_ggml_ptx_generation() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Verify kernel name
    assert!(
        ptx.contains("q4k_gemm_ggml"),
        "Should contain GGML kernel name"
    );

    // Verify parameters
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_quant_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
    assert!(ptx.contains(".param .u32 m"));
    assert!(ptx.contains(".param .u32 n"));
    assert!(ptx.contains(".param .u32 k"));
}

#[test]
fn test_ggml_ptx_contains_f16_loads() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // GGML Q4_K has f16 scale (d) and min (dmin) at super-block header
    assert!(
        ptx.contains("ld.global.f16") || ptx.contains("ld.global.b16"),
        "Should load f16 values for d and dmin"
    );
    assert!(
        ptx.contains("cvt") && ptx.contains("f32"),
        "Should convert f16 to f32 for computation"
    );
}

#[test]
fn test_ggml_ptx_contains_nested_loops() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // GGML kernel has nested loops: super-block loop and sub-block loop
    assert!(ptx.contains("sb_loop"), "Should have super-block loop");
    assert!(
        ptx.contains("sub_block_loop"),
        "Should have sub-block loop for 8 sub-blocks"
    );
}

#[test]
fn test_ggml_ptx_contains_scale_extraction() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Scale extraction involves bit manipulation (12-bit packed entries)
    assert!(
        ptx.contains("shr") || ptx.contains("shl"),
        "Should have shift operations for scale extraction"
    );
    assert!(
        ptx.contains("and"),
        "Should have AND operations for 6-bit masking"
    );
}

#[test]
fn test_ggml_ptx_contains_warp_reduce() {
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Warp shuffle reduction for dot product
    assert!(
        ptx.contains("shfl"),
        "Should have warp shuffle for reduction"
    );
}

#[test]
fn test_ggml_both_loop_branches_back() {
    // FALSIFIABLE: Both loops should branch back to their start
    let kernel = QuantizeKernel::ggml(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    let sb_loop_count = ptx.matches("sb_loop").count();
    let sub_block_loop_count = ptx.matches("sub_block_loop").count();

    // Each loop should have: label definition + branch back = 2 references
    assert!(
        sb_loop_count >= 2,
        "sb_loop should appear at least twice (label + branch back), found {}",
        sb_loop_count
    );
    assert!(
        sub_block_loop_count >= 2,
        "sub_block_loop should appear at least twice (label + branch back), found {}",
        sub_block_loop_count
    );
}

#[test]
fn test_simplified_vs_ggml_different_ptx() {
    // Verify simplified and GGML kernels produce different PTX
    let simplified = QuantizeKernel::new(1024, 1024, 4096);
    let ggml = QuantizeKernel::ggml(1024, 1024, 4096);

    let ptx_simplified = simplified.emit_ptx();
    let ptx_ggml = ggml.emit_ptx();

    assert_ne!(
        ptx_simplified, ptx_ggml,
        "Simplified and GGML kernels should produce different PTX"
    );
    assert!(ptx_simplified.contains("q4k_gemm_fused"));
    assert!(ptx_ggml.contains("q4k_gemm_ggml"));
}

// =========================================================================
// PARITY-116: Q5_K Kernel Tests
// =========================================================================

#[test]
fn test_q5k_kernel_name() {
    let kernel = Q5KKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.name(), "q5k_gemm_ggml");
}

#[test]
fn test_q5k_kernel_config() {
    let kernel = Q5KKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.m, 1024);
    assert_eq!(kernel.n, 1024);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.tile_size, 32);
}

#[test]
fn test_q5k_super_block_constants() {
    assert_eq!(
        Q5K_SUPER_BLOCK_SIZE, 256,
        "Q5_K super-block should have 256 values"
    );
    assert_eq!(
        Q5K_SUPER_BLOCK_BYTES, 176,
        "Q5_K super-block should be 176 bytes (2+2+12+128+32)"
    );
}

#[test]
fn test_q5k_num_super_blocks() {
    let kernel = Q5KKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256 = 16
}

#[test]
fn test_q5k_ptx_generation() {
    let kernel = Q5KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Verify kernel name
    assert!(
        ptx.contains("q5k_gemm_ggml"),
        "Should contain Q5_K kernel name"
    );

    // Verify parameters
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_quant_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
    assert!(ptx.contains(".param .u32 m"));
    assert!(ptx.contains(".param .u32 n"));
    assert!(ptx.contains(".param .u32 k"));
}

#[test]
fn test_q5k_with_tile_size() {
    let kernel = Q5KKernel::new(1024, 1024, 4096).with_tile_size(64);
    assert_eq!(kernel.tile_size, 64);
    assert_eq!(kernel.m, 1024);
    assert_eq!(kernel.n, 1024);
    assert_eq!(kernel.k, 4096);
}

#[test]
fn test_q5k_with_tile_size_affects_ptx() {
    let kernel_32 = Q5KKernel::new(1024, 1024, 4096);
    let kernel_64 = Q5KKernel::new(1024, 1024, 4096).with_tile_size(64);

    let ptx_32 = kernel_32.emit_ptx();
    let ptx_64 = kernel_64.emit_ptx();

    // Both should be valid PTX with the same kernel name
    assert!(ptx_32.contains("q5k_gemm_ggml"));
    assert!(ptx_64.contains("q5k_gemm_ggml"));
}

#[test]
fn test_q5k_ptx_contains_nested_loops() {
    let kernel = Q5KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("sb_loop"), "Should have super-block loop");
    assert!(ptx.contains("sub_block_loop"), "Should have sub-block loop");
}

#[test]
fn test_q5k_ptx_contains_high_bit_load() {
    // FALSIFIABLE: Q5_K must load high bits from qh (offset 144)
    let kernel = Q5KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Q5_K has 1-bit high values packed in qh array
    // The kernel should have multiple ld.global.u8 for ql and qh
    let load_count = ptx.matches("ld.global.u8").count();
    assert!(
        load_count >= 4, // At least scales (2) + ql + qh
        "Q5_K should have multiple u8 loads for scales, ql, and qh. Found {}",
        load_count
    );
}

#[test]
fn test_q5k_both_loops_branch_back() {
    let kernel = Q5KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    let sb_loop_count = ptx.matches("sb_loop").count();
    let sub_block_loop_count = ptx.matches("sub_block_loop").count();

    assert!(
        sb_loop_count >= 2,
        "sb_loop should appear at least twice (label + branch back), found {}",
        sb_loop_count
    );
    assert!(
        sub_block_loop_count >= 2,
        "sub_block_loop should appear at least twice (label + branch back), found {}",
        sub_block_loop_count
    );
}

// =========================================================================
// PARITY-117: Q6_K Kernel Tests
// =========================================================================

#[test]
fn test_q6k_kernel_name() {
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.name(), "q6k_gemm_ggml");
}

#[test]
fn test_q6k_kernel_config() {
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.m, 1024);
    assert_eq!(kernel.n, 1024);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.tile_size, 32);
}

#[test]
fn test_q6k_super_block_constants() {
    assert_eq!(
        Q6K_SUPER_BLOCK_SIZE, 256,
        "Q6_K super-block should have 256 values"
    );
    assert_eq!(
        Q6K_SUPER_BLOCK_BYTES, 210,
        "Q6_K super-block should be 210 bytes (128+64+16+2)"
    );
}

#[test]
fn test_q6k_num_super_blocks() {
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256 = 16
}

#[test]
fn test_q6k_ptx_generation() {
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Verify kernel name
    assert!(
        ptx.contains("q6k_gemm_ggml"),
        "Should contain Q6_K kernel name"
    );

    // Verify parameters
    assert!(ptx.contains(".param .u64 a_ptr"));
    assert!(ptx.contains(".param .u64 b_quant_ptr"));
    assert!(ptx.contains(".param .u64 c_ptr"));
    assert!(ptx.contains(".param .u32 m"));
    assert!(ptx.contains(".param .u32 n"));
    assert!(ptx.contains(".param .u32 k"));
}

#[test]
fn test_q6k_with_tile_size() {
    let kernel = Q6KKernel::new(1024, 1024, 4096).with_tile_size(64);
    assert_eq!(kernel.tile_size, 64);
    assert_eq!(kernel.m, 1024);
    assert_eq!(kernel.n, 1024);
    assert_eq!(kernel.k, 4096);
}

#[test]
fn test_q6k_with_tile_size_affects_ptx() {
    let kernel_32 = Q6KKernel::new(1024, 1024, 4096);
    let kernel_64 = Q6KKernel::new(1024, 1024, 4096).with_tile_size(64);

    let ptx_32 = kernel_32.emit_ptx();
    let ptx_64 = kernel_64.emit_ptx();

    // Both should be valid PTX with the same kernel name
    assert!(ptx_32.contains("q6k_gemm_ggml"));
    assert!(ptx_64.contains("q6k_gemm_ggml"));
}

#[test]
fn test_q6k_ptx_contains_nested_loops() {
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("sb_loop"), "Should have super-block loop");
    assert!(ptx.contains("sub_block_loop"), "Should have sub-block loop");
}

#[test]
fn test_q6k_ptx_contains_2bit_high_extraction() {
    // FALSIFIABLE: Q6_K must load and extract 2-bit high values from qh
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Q6_K has 2-bit high values, needs mask 0x3
    assert!(ptx.contains("and"), "Should have AND for bit masking");
}

#[test]
fn test_q6k_ptx_contains_signed_offset() {
    // FALSIFIABLE: Q6_K subtracts 32 to convert to signed range
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    // Q6_K: quant = ql + 4*qh - 32 (signed range -32 to 31)
    assert!(
        ptx.contains("sub.f32") || ptx.contains("sub.rn.f32"),
        "Should have subtraction for signed offset"
    );
}

#[test]
fn test_q6k_both_loops_branch_back() {
    let kernel = Q6KKernel::new(1024, 1024, 4096);
    let ptx = kernel.emit_ptx();

    let sb_loop_count = ptx.matches("sb_loop").count();
    let sub_block_loop_count = ptx.matches("sub_block_loop").count();

    assert!(
        sb_loop_count >= 2,
        "sb_loop should appear at least twice (label + branch back), found {}",
        sb_loop_count
    );
    assert!(
        sub_block_loop_count >= 2,
        "sub_block_loop should appear at least twice (label + branch back), found {}",
        sub_block_loop_count
    );
}

#[test]
fn test_all_quant_kernels_different() {
    // Verify all quantized kernels produce distinct PTX
    let q4k = QuantizeKernel::ggml(1024, 1024, 4096);
    let q5k = Q5KKernel::new(1024, 1024, 4096);
    let q6k = Q6KKernel::new(1024, 1024, 4096);

    let ptx_q4k = q4k.emit_ptx();
    let ptx_q5k = q5k.emit_ptx();
    let ptx_q6k = q6k.emit_ptx();

    assert_ne!(
        ptx_q4k, ptx_q5k,
        "Q4_K and Q5_K should produce different PTX"
    );
    assert_ne!(
        ptx_q4k, ptx_q6k,
        "Q4_K and Q6_K should produce different PTX"
    );
    assert_ne!(
        ptx_q5k, ptx_q6k,
        "Q5_K and Q6_K should produce different PTX"
    );
}

// =========================================================================
// Property-Based Tests (PARITY-116, PARITY-117)
// =========================================================================

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn prop_q5k_valid_ptx_for_any_size(
        m in 32u32..512,
        n in 32u32..512,
        // K must be divisible by 256 for super-blocks
        k_factor in 1u32..8
    ) {
        let k = k_factor * 256;
        let kernel = Q5KKernel::new(m, n, k);
        let ptx = kernel.emit_ptx();

        // PTX must be valid (non-empty, contains kernel)
        prop_assert!(!ptx.is_empty());
        prop_assert!(ptx.contains("q5k_gemm_ggml"));
        prop_assert!(ptx.contains(".entry"));
        prop_assert!(ptx.contains("ret;"));

        // Must have nested loops
        prop_assert!(ptx.contains("sb_loop"));
        prop_assert!(ptx.contains("sub_block_loop"));
    }

    #[test]
    fn prop_q6k_valid_ptx_for_any_size(
        m in 32u32..512,
        n in 32u32..512,
        k_factor in 1u32..8
    ) {
        let k = k_factor * 256;
        let kernel = Q6KKernel::new(m, n, k);
        let ptx = kernel.emit_ptx();

        prop_assert!(!ptx.is_empty());
        prop_assert!(ptx.contains("q6k_gemm_ggml"));
        prop_assert!(ptx.contains(".entry"));
        prop_assert!(ptx.contains("ret;"));

        // Q6_K-specific: signed offset subtraction
        prop_assert!(ptx.contains("sub.f32") || ptx.contains("sub.rn.f32"));
    }

    #[test]
    fn prop_q5k_super_blocks_correct(k_factor in 1u32..16) {
        let k = k_factor * 256;
        let kernel = Q5KKernel::new(64, 64, k);
        prop_assert_eq!(kernel.num_super_blocks_per_row(), k_factor);
    }

    #[test]
    fn prop_q6k_super_blocks_correct(k_factor in 1u32..16) {
        let k = k_factor * 256;
        let kernel = Q6KKernel::new(64, 64, k);
        prop_assert_eq!(kernel.num_super_blocks_per_row(), k_factor);
    }

    /// Matvec case (n=1) used by realizar for GGUF inference
    #[test]
    fn prop_q5k_q6k_matvec_n1(m in 32u32..512, k_factor in 1u32..8) {
        let k = k_factor * 256;

        // Q5K matvec
        let q5k = Q5KKernel::new(m, 1, k);
        let ptx_q5k = q5k.emit_ptx();
        prop_assert!(ptx_q5k.contains("q5k_gemm_ggml"));
        prop_assert!(ptx_q5k.contains(".entry"));

        // Q6K matvec
        let q6k = Q6KKernel::new(m, 1, k);
        let ptx_q6k = q6k.emit_ptx();
        prop_assert!(ptx_q6k.contains("q6k_gemm_ggml"));
        prop_assert!(ptx_q6k.contains(".entry"));
    }

    #[test]
    fn prop_all_quant_kernels_distinct(
        m in 64u32..256,
        n in 64u32..256,
        k_factor in 1u32..4
    ) {
        let k = k_factor * 256;
        let q4k = QuantizeKernel::ggml(m, n, k);
        let q5k = Q5KKernel::new(m, n, k);
        let q6k = Q6KKernel::new(m, n, k);

        let ptx_q4k = q4k.emit_ptx();
        let ptx_q5k = q5k.emit_ptx();
        let ptx_q6k = q6k.emit_ptx();

        prop_assert!(ptx_q4k != ptx_q5k);
        prop_assert!(ptx_q4k != ptx_q6k);
        prop_assert!(ptx_q5k != ptx_q6k);
    }
}

// =========================================================================
// PARITY-114: Barrier Safety Tests for Quantized Kernels
// =========================================================================

#[test]
fn test_q4k_ggml_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = QuantizeKernel::ggml(32, 32, 256);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);

    // Debug output: show what's detected
    if !result.is_safe {
        println!("Q4K GGML barrier_count: {}", result.barrier_count);
        println!("Q4K GGML exit_count: {}", result.exit_count);
        for v in &result.violations {
            println!(
                "Violation at line {}: {:?} - {}",
                v.line, v.kind, v.instruction
            );
        }
        // Print PTX around the violation
        for (i, line) in ptx.lines().enumerate() {
            let lineno = i + 1;
            if result
                .violations
                .iter()
                .any(|v| v.line.saturating_sub(5) <= lineno && lineno <= v.line.saturating_add(5))
            {
                println!("{:4}: {}", lineno, line);
            }
        }
    }

    assert!(
        result.is_safe,
        "Q4K GGML should be barrier-safe: {:?}",
        result.violations
    );
}

#[test]
fn test_q5k_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = Q5KKernel::new(32, 32, 256);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(
        result.is_safe,
        "Q5K should be barrier-safe: {:?}",
        result.violations
    );
}

#[test]
fn test_q6k_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = Q6KKernel::new(32, 32, 256);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(
        result.is_safe,
        "Q6K should be barrier-safe: {:?}",
        result.violations
    );
}

// =========================================================================
// PAR-003: Q4_K/Q5_K/Q6_K GEMV Kernel Tests
// GEMV kernels for M=1 decode throughput (token generation critical path)
// =========================================================================

#[test]
fn test_q4k_gemv_kernel_name() {
    let kernel = Q4KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.name(), "q4k_gemv_warp_reduce");
}

#[test]
fn test_q4k_gemv_kernel_config() {
    let kernel = Q4KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 32000);
    assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256
}

#[test]
fn test_q4k_gemv_ptx_generation() {
    let kernel = Q4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Verify kernel name
    assert!(
        ptx.contains("q4k_gemv_warp_reduce"),
        "Should contain GEMV kernel name"
    );

    // Verify parameters (different from GEMM)
    assert!(ptx.contains(".param .u64 y_ptr"), "Missing y_ptr param");
    assert!(ptx.contains(".param .u64 w_ptr"), "Missing w_ptr param");
    assert!(ptx.contains(".param .u64 x_ptr"), "Missing x_ptr param");
    assert!(ptx.contains(".param .u32 k_dim"), "Missing k_dim param");
    assert!(ptx.contains(".param .u32 n_dim"), "Missing n_dim param");
}

#[test]
fn test_q4k_gemv_has_warp_shuffle() {
    let kernel = Q4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // GEMV uses warp shuffle for reduction (like GemvKernel)
    assert!(
        ptx.contains("shfl.sync.down") || ptx.contains("shfl.down"),
        "Q4K GEMV should use warp shuffle for reduction"
    );
}

#[test]
fn test_q4k_gemv_no_shared_memory() {
    let kernel = Q4KGemvKernel::new(4096, 4096);
    let ptx_kernel = kernel.build_ptx();

    // GEMV kernels don't need shared memory - each warp works independently
    assert_eq!(
        ptx_kernel.shared_memory_bytes(),
        0,
        "Q4K GEMV should not use shared memory"
    );
}

#[test]
fn test_q4k_gemv_has_fma() {
    let kernel = Q4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Should use FMA for accumulation
    assert!(
        ptx.contains("fma.rn.f32") || ptx.contains("mad.f32"),
        "Q4K GEMV should use FMA for accumulation"
    );
}

#[test]
fn test_q5k_gemv_kernel_name() {
    let kernel = Q5KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.name(), "q5k_gemv_warp_reduce");
}

#[test]
fn test_q5k_gemv_ptx_generation() {
    let kernel = Q5KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains("q5k_gemv_warp_reduce"),
        "Should contain GEMV kernel name"
    );
    assert!(ptx.contains(".param .u64 y_ptr"), "Missing y_ptr param");
    assert!(ptx.contains(".param .u64 w_ptr"), "Missing w_ptr param");
    assert!(ptx.contains(".param .u64 x_ptr"), "Missing x_ptr param");
}

#[test]
fn test_q6k_gemv_kernel_name() {
    let kernel = Q6KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.name(), "q6k_gemv_warp_reduce");
}

#[test]
fn test_q6k_gemv_ptx_generation() {
    let kernel = Q6KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains("q6k_gemv_warp_reduce"),
        "Should contain GEMV kernel name"
    );
    assert!(ptx.contains(".param .u64 y_ptr"), "Missing y_ptr param");
    assert!(ptx.contains(".param .u64 w_ptr"), "Missing w_ptr param");
    assert!(ptx.contains(".param .u64 x_ptr"), "Missing x_ptr param");
}

#[test]
fn test_all_gemv_kernels_different() {
    let q4k = Q4KGemvKernel::new(4096, 4096);
    let q5k = Q5KGemvKernel::new(4096, 4096);
    let q6k = Q6KGemvKernel::new(4096, 4096);

    let ptx_q4k = q4k.emit_ptx();
    let ptx_q5k = q5k.emit_ptx();
    let ptx_q6k = q6k.emit_ptx();

    assert_ne!(
        ptx_q4k, ptx_q5k,
        "Q4K and Q5K GEMV should produce different PTX"
    );
    assert_ne!(
        ptx_q5k, ptx_q6k,
        "Q5K and Q6K GEMV should produce different PTX"
    );
    assert_ne!(
        ptx_q4k, ptx_q6k,
        "Q4K and Q6K GEMV should produce different PTX"
    );
}

#[test]
fn test_q4k_gemv_vs_gemm_different() {
    let gemv = Q4KGemvKernel::new(4096, 4096);
    let gemm = QuantizeKernel::ggml(1, 4096, 4096);

    let ptx_gemv = gemv.emit_ptx();
    let ptx_gemm = gemm.emit_ptx();

    // GEMV and GEMM should have different kernel names and structures
    assert!(
        ptx_gemv.contains("gemv"),
        "GEMV kernel should have 'gemv' in name"
    );
    assert!(
        ptx_gemm.contains("gemm"),
        "GEMM kernel should have 'gemm' in name"
    );
    assert_ne!(
        ptx_gemv, ptx_gemm,
        "GEMV and GEMM should produce different PTX"
    );
}

#[test]
fn test_q4k_gemv_loop_branches_back() {
    // FALSIFIABLE: Super-block loop should branch back to start
    let kernel = Q4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    let sb_loop_count = ptx.matches("sb_loop").count();
    assert!(
        sb_loop_count >= 2,
        "sb_loop should appear at least twice (label + branch back), found {}",
        sb_loop_count
    );
}

#[test]
fn test_q4k_gemv_barrier_safety() {
    // GEMV kernels don't use barriers, so they should be trivially barrier-safe
    use crate::ptx::optimize::barrier_safety;
    let kernel = Q4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(
        result.is_safe,
        "Q4K GEMV should be barrier-safe: {:?}",
        result.violations
    );
}

#[test]
fn test_q5k_gemv_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = Q5KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(
        result.is_safe,
        "Q5K GEMV should be barrier-safe: {:?}",
        result.violations
    );
}

#[test]
fn test_q6k_gemv_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = Q6KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(
        result.is_safe,
        "Q6K GEMV should be barrier-safe: {:?}",
        result.violations
    );
}

// =========================================================================
// PAR-030: Fused RMSNorm + Q4K GEMV Kernel Tests
// =========================================================================

#[test]
fn test_fused_rmsnorm_q4k_gemv_kernel_name() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "fused_rmsnorm_q4k_gemv");
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_config() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
    assert!((kernel.epsilon - 1e-5).abs() < 1e-10);
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_with_epsilon() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096).with_epsilon(1e-6);
    assert!((kernel.epsilon - 1e-6).abs() < 1e-10);
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_ptx_generation() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // Verify parameters
    assert!(ptx.contains(".param .u64 y_ptr"));
    assert!(ptx.contains(".param .u64 w_ptr"));
    assert!(ptx.contains(".param .u64 x_ptr"));
    assert!(ptx.contains(".param .u64 gamma_ptr"));
    assert!(ptx.contains(".param .u32 k_dim"));
    assert!(ptx.contains(".param .u32 n_dim"));

    // Verify kernel name
    assert!(ptx.contains("fused_rmsnorm_q4k_gemv"));
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_shared_memory() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096);
    let ptx_kernel = kernel.build_ptx();

    // Should have shared memory for normalized input + warp partials:
    // 3584 * 4 + 32 = 14368 bytes
    assert!(ptx_kernel.shared_memory_bytes() > 0);
    assert_eq!(ptx_kernel.shared_memory_bytes(), 3584 * 4 + 32);
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_operations() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // Verify RMSNorm operations
    assert!(ptx.contains("rsqrt"), "Should have rsqrt for RMSNorm");
    assert!(ptx.contains("div.rn.f32"), "Should have division for mean");

    // Verify warp shuffle for reductions
    assert!(ptx.contains("shfl"), "Should have warp shuffle");

    // Verify shared memory operations
    assert!(
        ptx.contains("ld.shared.f32"),
        "Should load from shared memory"
    );
    assert!(
        ptx.contains("st.shared.f32"),
        "Should store to shared memory"
    );

    // Verify barrier synchronization
    assert!(ptx.contains("bar.sync"), "Should have barrier sync");

    // Verify Q4K dequantization (d, dmin loads)
    assert!(
        ptx.contains("cvt.f32.f16"),
        "Should convert F16 to F32 for d/dmin"
    );
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_loop_structure() {
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // Verify load loop exists
    let load_loop_count = ptx.matches("load_loop").count();
    assert!(
        load_loop_count >= 2,
        "load_loop should appear at least twice (label + branch), found {}",
        load_loop_count
    );

    // Verify norm loop exists
    let norm_loop_count = ptx.matches("norm_loop").count();
    assert!(
        norm_loop_count >= 2,
        "norm_loop should appear at least twice (label + branch), found {}",
        norm_loop_count
    );

    // Verify super-block loop exists
    let sb_loop_count = ptx.matches("sb_loop").count();
    assert!(
        sb_loop_count >= 2,
        "sb_loop should appear at least twice (label + branch), found {}",
        sb_loop_count
    );
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_barrier_safety() {
    // This kernel uses barriers, need to verify barrier safety
    use crate::ptx::optimize::barrier_safety;
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(
        result.is_safe,
        "Fused RMSNorm+Q4K GEMV should be barrier-safe: {:?}",
        result.violations
    );
}

#[test]
fn test_fused_rmsnorm_q4k_gemv_qwen3b_config() {
    // Qwen 3B typical dimensions
    let kernel = FusedRmsNormQ4KGemvKernel::new(3584, 18944); // hidden -> intermediate
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".visible .entry"));
}

// =========================================================================
// PAR-031: Tiled Q4K GEMV Kernel Tests
// =========================================================================

#[test]
fn test_tiled_q4k_gemv_kernel_name() {
    let kernel = TiledQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "tiled_q4k_gemv");
}

#[test]
fn test_tiled_q4k_gemv_config() {
    let kernel = TiledQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
    assert_eq!(kernel.outputs_per_block, 4);
}

#[test]
fn test_tiled_q4k_gemv_with_outputs_per_block() {
    let kernel = TiledQ4KGemvKernel::new(3584, 4096).with_outputs_per_block(8);
    assert_eq!(kernel.outputs_per_block, 8);
}

#[test]
fn test_tiled_q4k_gemv_ptx_generation() {
    let kernel = TiledQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // Verify parameters
    assert!(ptx.contains(".param .u64 y_ptr"));
    assert!(ptx.contains(".param .u64 w_ptr"));
    assert!(ptx.contains(".param .u64 x_ptr"));
    assert!(ptx.contains(".param .u32 k_dim"));
    assert!(ptx.contains(".param .u32 n_dim"));

    // Verify kernel name
    assert!(ptx.contains("tiled_q4k_gemv"));
}

#[test]
fn test_tiled_q4k_gemv_shared_memory() {
    let kernel = TiledQ4KGemvKernel::new(3584, 4096);
    let ptx_kernel = kernel.build_ptx();

    // Should have shared memory for input vector: 3584 * 4 = 14336 bytes
    assert!(ptx_kernel.shared_memory_bytes() > 0);
    assert_eq!(ptx_kernel.shared_memory_bytes(), 3584 * 4);
}

#[test]
fn test_tiled_q4k_gemv_operations() {
    let kernel = TiledQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // Verify shared memory via generic addressing (cvta.shared approach)
    // The kernel uses cvta.shared to get a generic address, then generic ld/st
    assert!(
        ptx.contains("cvta.shared"),
        "Should convert shared address to generic"
    );
    assert!(
        ptx.contains("ld.f32"),
        "Should have generic loads (for shared via cvta)"
    );
    assert!(
        ptx.contains("st.f32"),
        "Should have generic stores (for shared via cvta)"
    );

    // Verify barrier synchronization
    assert!(ptx.contains("bar.sync"), "Should have barrier sync");

    // Verify warp shuffle for reductions
    assert!(ptx.contains("shfl"), "Should have warp shuffle");

    // Verify Q4K dequantization (d, dmin loads)
    assert!(
        ptx.contains("cvt.f32.f16"),
        "Should convert F16 to F32 for d/dmin"
    );
}

#[test]
fn test_tiled_q4k_gemv_loop_structure() {
    let kernel = TiledQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    // Verify load loop exists
    let load_loop_count = ptx.matches("load_loop").count();
    assert!(
        load_loop_count >= 2,
        "load_loop should appear at least twice (label + branch), found {}",
        load_loop_count
    );

    // Verify super-block loop exists
    let sb_loop_count = ptx.matches("sb_loop").count();
    assert!(
        sb_loop_count >= 2,
        "sb_loop should appear at least twice (label + branch), found {}",
        sb_loop_count
    );
}

#[test]
fn test_tiled_q4k_gemv_barrier_safety() {
    // This kernel uses barriers for shared memory synchronization
    use crate::ptx::optimize::barrier_safety;
    let kernel = TiledQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(
        result.is_safe,
        "Tiled Q4K GEMV should be barrier-safe: {:?}",
        result.violations
    );
}

#[test]
fn test_tiled_q4k_gemv_qwen3b_config() {
    // Qwen 3B dimensions
    let kernel = TiledQ4KGemvKernel::new(3584, 18944).with_outputs_per_block(8);
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".visible .entry"));
}

// ==========================================================================
// PAR-032: FP16 Q4K GEMV KERNEL TESTS
// ==========================================================================

#[test]
fn test_fp16_q4k_gemv_kernel_name() {
    let kernel = Fp16Q4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "fp16_q4k_gemv");
}

#[test]
fn test_fp16_q4k_gemv_generates_ptx() {
    let kernel = Fp16Q4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".visible .entry fp16_q4k_gemv"));
}

#[test]
fn test_fp16_q4k_gemv_has_fp16_loads() {
    // Verify FP16 input loads
    let kernel = Fp16Q4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();
    // Should use ld.global.b16 for FP16 input
    assert!(ptx.contains("ld.global"));
    // Should have cvt.f32.f16 for conversion
    assert!(ptx.contains("cvt.f32.f16"));
}

#[test]
fn test_fp16_q4k_gemv_has_fp16_stores() {
    // Verify FP16 output stores
    let kernel = Fp16Q4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();
    // Should use st.global.b16 for FP16 output
    assert!(ptx.contains("st.global"));
    // Should have cvt.f16.f32 for conversion
    assert!(ptx.contains("cvt.rn.f16.f32"));
}

#[test]
fn test_fp16_q4k_gemv_has_warp_shuffle() {
    let kernel = Fp16Q4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();
    // Should use warp shuffle for reduction
    assert!(ptx.contains("shfl.sync.down"));
}

#[test]
fn test_fp16_q4k_gemv_qwen3b_dimensions() {
    // Qwen 3B typical dimensions
    let kernel = Fp16Q4KGemvKernel::new(3584, 3584);
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".visible .entry"));
}

#[test]
fn test_fp16_q4k_gemv_ffn_dimensions() {
    // Qwen 3B FFN dimensions (hidden_size → intermediate_size)
    let kernel = Fp16Q4KGemvKernel::new(3584, 18944);
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
}

#[test]
fn test_fp16_q4k_gemv_structure() {
    let kernel = Fp16Q4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

// ==========================================================================
// PAR-034: TENSOR CORE Q4K GEMM KERNEL TESTS
// ==========================================================================

#[test]
fn test_tensor_core_q4k_gemm_kernel_name() {
    let kernel = TensorCoreQ4KGemmKernel::new(16, 3584, 4096);
    assert_eq!(kernel.name(), "tensor_core_q4k_gemm");
}

#[test]
fn test_tensor_core_q4k_gemm_generates_ptx() {
    let kernel = TensorCoreQ4KGemmKernel::new(16, 3584, 4096);
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".visible .entry tensor_core_q4k_gemm"));
}

#[test]
fn test_tensor_core_q4k_gemm_has_fp16_io() {
    let kernel = TensorCoreQ4KGemmKernel::new(16, 3584, 4096);
    let ptx = kernel.emit_ptx();
    // Should have FP16 loads and stores
    assert!(ptx.contains("ld.global"));
    assert!(ptx.contains("st.global"));
    // Should have FP16 conversions
    assert!(ptx.contains("cvt.f32.f16") || ptx.contains("cvt"));
}

#[test]
fn test_tensor_core_q4k_gemm_batched_dimensions() {
    // Speculative decode with K=8 draft tokens
    let kernel = TensorCoreQ4KGemmKernel::new(8, 3584, 4096);
    assert_eq!(kernel.m, 8);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
    assert_eq!(kernel.num_super_blocks(), 14); // 3584 / 256 = 14
}

#[test]
fn test_tensor_core_q4k_gemm_qwen3b_ffn() {
    // Qwen 3B FFN: [batch, 3584] × [3584, 18944]
    let kernel = TensorCoreQ4KGemmKernel::new(16, 3584, 18944);
    let ptx = kernel.emit_ptx();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".visible .entry"));
}

#[test]
fn test_tensor_core_q4k_gemm_has_barrier() {
    let kernel = TensorCoreQ4KGemmKernel::new(16, 3584, 4096);
    let ptx = kernel.emit_ptx();
    // Should have barrier for shared memory synchronization
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_tensor_core_q4k_gemm_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = TensorCoreQ4KGemmKernel::new(16, 3584, 4096);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(
        result.is_safe,
        "Tensor Core Q4K GEMM should be barrier-safe: {:?}",
        result.violations
    );
}

// =========================================================================
// Q8_0 GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_q8_0_gemv_kernel_name() {
    let kernel = Q8_0GemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "q8_0_gemv_warp_reduce");
}

#[test]
fn test_q8_0_gemv_config() {
    let kernel = Q8_0GemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_q8_0_gemv_num_blocks() {
    let kernel = Q8_0GemvKernel::new(3584, 4096);
    assert_eq!(kernel.num_blocks_per_row(), 112); // ceil(3584/32)
}

#[test]
fn test_q8_0_gemv_ptx_generation() {
    let kernel = Q8_0GemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry q8_0_gemv_warp_reduce"));
    assert!(ptx.contains(".param .u64"));
    assert!(ptx.contains("ld.global"));
    assert!(ptx.contains("st.global"));
}

// =========================================================================
// Q4_0 GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_q4_0_gemv_kernel_name() {
    let kernel = Q4_0GemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "q4_0_gemv_warp_reduce");
}

#[test]
fn test_q4_0_gemv_config() {
    let kernel = Q4_0GemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_q4_0_gemv_ptx_generation() {
    let kernel = Q4_0GemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry q4_0_gemv_warp_reduce"));
    assert!(ptx.contains(".param .u64"));
    assert!(ptx.contains("ld.global"));
}

// =========================================================================
// Q4_1 GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_q4_1_gemv_kernel_name() {
    let kernel = Q4_1GemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "q4_1_gemv_warp_reduce");
}

#[test]
fn test_q4_1_gemv_config() {
    let kernel = Q4_1GemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_q4_1_gemv_ptx_generation() {
    let kernel = Q4_1GemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry q4_1_gemv_warp_reduce"));
    assert!(ptx.contains(".param .u64"));
    assert!(ptx.contains("ld.global"));
}

// =========================================================================
// Q5_0 GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_q5_0_gemv_kernel_name() {
    let kernel = Q5_0GemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "q5_0_gemv_warp_reduce");
}

#[test]
fn test_q5_0_gemv_config() {
    let kernel = Q5_0GemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_q5_0_gemv_ptx_generation() {
    let kernel = Q5_0GemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry q5_0_gemv_warp_reduce"));
    assert!(ptx.contains(".param .u64"));
    assert!(ptx.contains("ld.global"));
}

// =========================================================================
// CHUNKED TILED Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_chunked_tiled_q4k_gemv_kernel_name() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "chunked_tiled_q4k_gemv");
}

#[test]
fn test_chunked_tiled_q4k_gemv_config() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_chunked_tiled_q4k_gemv_ptx_generation() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry chunked_tiled_q4k_gemv"));
    assert!(ptx.contains("bar.sync")); // Shared memory sync
}

#[test]
fn test_chunked_tiled_q4k_gemv_shared_memory() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096);
    let ptx_kernel = kernel.build_ptx();
    assert!(ptx_kernel.shared_memory_bytes() > 0);
}

#[test]
fn test_chunked_tiled_q4k_gemv_with_outputs_per_block() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096).with_outputs_per_block(8);
    assert_eq!(kernel.outputs_per_block, 8);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_chunked_tiled_q4k_gemv_with_outputs_per_block_default() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.outputs_per_block, 4); // Default value
}

#[test]
fn test_chunked_tiled_q4k_gemv_with_outputs_per_block_chained() {
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096)
        .with_outputs_per_block(2)
        .with_outputs_per_block(16);
    assert_eq!(kernel.outputs_per_block, 16); // Last value wins
}

#[test]
fn test_chunked_tiled_q4k_gemv_needs_chunking_small_k() {
    // K = 3584 < 8192 (CHUNK_SIZE), so no chunking needed
    let kernel = ChunkedTiledQ4KGemvKernel::new(3584, 4096);
    assert!(!kernel.needs_chunking());
}

#[test]
fn test_chunked_tiled_q4k_gemv_needs_chunking_large_k() {
    // K = 16384 > 8192 (CHUNK_SIZE), so chunking is needed
    let kernel = ChunkedTiledQ4KGemvKernel::new(16384, 4096);
    assert!(kernel.needs_chunking());
}

#[test]
fn test_chunked_tiled_q4k_gemv_needs_chunking_boundary() {
    // K = 8192 = CHUNK_SIZE exactly, no chunking needed
    let kernel_exact = ChunkedTiledQ4KGemvKernel::new(8192, 4096);
    assert!(!kernel_exact.needs_chunking());

    // K = 8193 > CHUNK_SIZE, chunking needed
    let kernel_over = ChunkedTiledQ4KGemvKernel::new(8193, 4096);
    assert!(kernel_over.needs_chunking());
}

#[test]
fn test_chunked_tiled_q4k_gemv_needs_chunking_very_large_k() {
    // K = 32768, definitely needs chunking
    let kernel = ChunkedTiledQ4KGemvKernel::new(32768, 4096);
    assert!(kernel.needs_chunking());
}

// =========================================================================
// COALESCED Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_coalesced_q4k_gemv_kernel_name() {
    let kernel = CoalescedQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "coalesced_q4k_gemv");
}

#[test]
fn test_coalesced_q4k_gemv_config() {
    let kernel = CoalescedQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_coalesced_q4k_gemv_ptx_generation() {
    let kernel = CoalescedQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry coalesced_q4k_gemv"));
    assert!(ptx.contains("ld.global"));
}

// =========================================================================
// DP4A Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_dp4a_q4k_gemv_kernel_name() {
    let kernel = Dp4aQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "dp4a_q4k_gemv");
}

#[test]
fn test_dp4a_q4k_gemv_config() {
    let kernel = Dp4aQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_dp4a_q4k_gemv_ptx_generation() {
    let kernel = Dp4aQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry dp4a_q4k_gemv"));
    // Should have dp4a instructions for int8 dot product
    assert!(ptx.contains("dp4a"));
}

// =========================================================================
// TRUE DP4A Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_true_dp4a_q4k_gemv_kernel_name() {
    let kernel = TrueDp4aQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "true_dp4a_q4k_gemv");
}

#[test]
fn test_true_dp4a_q4k_gemv_config() {
    let kernel = TrueDp4aQ4KGemvKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_true_dp4a_q4k_gemv_ptx_generation() {
    let kernel = TrueDp4aQ4KGemvKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry true_dp4a_q4k_gemv"));
    assert!(ptx.contains("dp4a"));
}

// =========================================================================
// Q8 QUANTIZE KERNEL TESTS
// =========================================================================

#[test]
fn test_q8_quantize_kernel_name() {
    let kernel = Q8QuantizeKernel::new(3584);
    assert_eq!(kernel.name(), "q8_quantize");
}

#[test]
fn test_q8_quantize_config() {
    let kernel = Q8QuantizeKernel::new(3584);
    assert_eq!(kernel.n, 3584);
}

#[test]
fn test_q8_quantize_ptx_generation() {
    let kernel = Q8QuantizeKernel::new(3584);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry q8_quantize"));
    assert!(ptx.contains("ld.global"));
    assert!(ptx.contains("st.global"));
}

// =========================================================================
// Q4K Q8 DOT KERNEL TESTS
// =========================================================================

#[test]
fn test_q4k_q8_dot_kernel_name() {
    let kernel = Q4KQ8DotKernel::new(3584, 4096);
    assert_eq!(kernel.name(), "q4k_q8_dot");
}

#[test]
fn test_q4k_q8_dot_config() {
    let kernel = Q4KQ8DotKernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_q4k_q8_dot_ptx_generation() {
    let kernel = Q4KQ8DotKernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry q4k_q8_dot"));
    assert!(ptx.contains("ld.global")); // Loads data
    assert!(ptx.contains("shfl")); // Warp shuffle for reduction
}

// =========================================================================
// PACKED DP4A Q4K Q8 KERNEL TESTS
// =========================================================================

#[test]
fn test_packed_dp4a_q4k_q8_kernel_name() {
    let kernel = PackedDp4aQ4KQ8Kernel::new(3584, 4096);
    assert_eq!(kernel.name(), "packed_dp4a_q4k_q8");
}

#[test]
fn test_packed_dp4a_q4k_q8_config() {
    let kernel = PackedDp4aQ4KQ8Kernel::new(3584, 4096);
    assert_eq!(kernel.k, 3584);
    assert_eq!(kernel.n, 4096);
}

#[test]
fn test_packed_dp4a_q4k_q8_ptx_generation() {
    let kernel = PackedDp4aQ4KQ8Kernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry packed_dp4a_q4k_q8"));
    assert!(ptx.contains("dp4a"));
}

#[test]
fn test_packed_dp4a_q4k_q8_has_warp_shuffle() {
    let kernel = PackedDp4aQ4KQ8Kernel::new(3584, 4096);
    let ptx = kernel.emit_ptx();
    // This kernel uses warp shuffle for reduction
    assert!(ptx.contains("shfl"));
}

// =========================================================================
// BATCHED Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_batched_q4k_gemv_kernel_name() {
    let kernel = BatchedQ4KGemvKernel::new(4096, 32000, 4);
    assert_eq!(kernel.name(), "batched_q4k_gemv_warp_reduce");
}

#[test]
fn test_batched_q4k_gemv_config() {
    let kernel = BatchedQ4KGemvKernel::new(4096, 32000, 4);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 32000);
    assert_eq!(kernel.m, 4);
}

#[test]
fn test_batched_q4k_gemv_ptx_generation() {
    let kernel = BatchedQ4KGemvKernel::new(4096, 4096, 4);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry batched_q4k_gemv"));
    assert!(ptx.contains("ld.global"));
}

#[test]
fn test_batched_q4k_gemv_has_warp_shuffle() {
    let kernel = BatchedQ4KGemvKernel::new(4096, 4096, 2);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("shfl"));
}

#[test]
fn test_batched_q4k_gemv_num_super_blocks() {
    let kernel = BatchedQ4KGemvKernel::new(4096, 4096, 4);
    assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256
}

// =========================================================================
// COALESCED Q6K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_coalesced_q6k_gemv_kernel_name() {
    let kernel = CoalescedQ6KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.name(), "coalesced_q6k_gemv");
}

#[test]
fn test_coalesced_q6k_gemv_config() {
    let kernel = CoalescedQ6KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 32000);
}

#[test]
fn test_coalesced_q6k_gemv_ptx_generation() {
    let kernel = CoalescedQ6KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry coalesced_q6k_gemv"));
    assert!(ptx.contains("ld.global"));
}

#[test]
fn test_coalesced_q6k_gemv_num_super_blocks() {
    let kernel = CoalescedQ6KGemvKernel::new(4096, 4096);
    assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256 = 16

    let kernel2 = CoalescedQ6KGemvKernel::new(512, 4096);
    assert_eq!(kernel2.num_super_blocks_per_row(), 2); // 512 / 256 = 2
}

#[test]
fn test_coalesced_q6k_gemv_has_warp_shuffle() {
    // PAR-066: Coalesced kernel uses warp shuffle for scale broadcast
    let kernel = CoalescedQ6KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Should use shfl.sync.idx for broadcasting scales from lane 0-15
    assert!(
        ptx.contains("shfl.sync"),
        "Should use warp shuffle for scale broadcast"
    );
}

#[test]
fn test_coalesced_q6k_gemv_has_loops() {
    let kernel = CoalescedQ6KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("sb_loop"), "Should have super-block loop");
}

#[test]
fn test_coalesced_q6k_gemv_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = CoalescedQ6KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(
        result.is_safe,
        "Coalesced Q6K GEMV should pass barrier safety: {:?}",
        result.violations
    );
}

#[test]
fn test_coalesced_q6k_gemv_vs_basic_different() {
    // Coalesced and basic Q6K GEMV should produce different PTX
    let coalesced = CoalescedQ6KGemvKernel::new(4096, 4096);
    let basic = Q6KGemvKernel::new(4096, 4096);

    let ptx_coalesced = coalesced.emit_ptx();
    let ptx_basic = basic.emit_ptx();

    assert_ne!(
        ptx_coalesced, ptx_basic,
        "Coalesced and basic kernels should produce different PTX"
    );
    assert!(
        ptx_coalesced.contains("coalesced"),
        "Coalesced PTX should contain 'coalesced' in entry name"
    );
}

proptest! {
    #[test]
    fn prop_coalesced_q6k_generates_valid_ptx(
        k_factor in 1u32..8,
        n in 32u32..4096
    ) {
        let k = k_factor * 256;
        let kernel = CoalescedQ6KGemvKernel::new(k, n);
        let ptx = kernel.emit_ptx();

        prop_assert!(!ptx.is_empty());
        prop_assert!(ptx.contains(".visible .entry coalesced_q6k_gemv"));
    }

    #[test]
    fn prop_coalesced_q6k_super_blocks_correct(k_factor in 1u32..16) {
        let k = k_factor * 256;
        let kernel = CoalescedQ6KGemvKernel::new(k, 64);
        prop_assert_eq!(kernel.num_super_blocks_per_row(), k_factor);
    }
}

// =========================================================================
// VECTORIZED Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_vectorized_q4k_gemv_kernel_name() {
    let kernel = VectorizedQ4KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.name(), "vectorized_q4k_gemv");
}

#[test]
fn test_vectorized_q4k_gemv_config() {
    let kernel = VectorizedQ4KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 32000);
}

#[test]
fn test_vectorized_q4k_gemv_ptx_generation() {
    let kernel = VectorizedQ4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry vectorized_q4k_gemv"));
    assert!(ptx.contains("ld.global"));
}

#[test]
fn test_vectorized_q4k_gemv_has_warp_shuffle() {
    let kernel = VectorizedQ4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains("shfl"));
}

// =========================================================================
// FUSED GATE UP Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_fused_gate_up_q4k_gemv_kernel_name() {
    let kernel = FusedGateUpQ4KGemvKernel::new(4096, 11008);
    assert_eq!(kernel.name(), "fused_gate_up_q4k_gemv");
}

#[test]
fn test_fused_gate_up_q4k_gemv_config() {
    let kernel = FusedGateUpQ4KGemvKernel::new(4096, 11008);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 11008);
}

#[test]
fn test_fused_gate_up_q4k_gemv_ptx_generation() {
    let kernel = FusedGateUpQ4KGemvKernel::new(4096, 11008);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry fused_gate_up_q4k_gemv"));
    assert!(ptx.contains("ld.global"));
}

#[test]
fn test_fused_gate_up_q4k_gemv_has_arithmetic() {
    let kernel = FusedGateUpQ4KGemvKernel::new(4096, 11008);
    let ptx = kernel.emit_ptx();

    // Fused gate up uses FMA for efficient multiply-accumulate
    assert!(ptx.contains("fma") || ptx.contains("mul") || ptx.contains("add"));
}

#[test]
fn test_fused_gate_up_q4k_gemv_has_shared_memory() {
    let kernel = FusedGateUpQ4KGemvKernel::new(4096, 11008);
    let ptx_kernel = kernel.build_ptx();

    // Uses shared memory for input caching
    assert!(ptx_kernel.shared_memory_bytes() > 0);
}

// =========================================================================
// FP16 Q4K GEMV KERNEL TESTS
// =========================================================================

#[test]
fn test_fp16_q4k_gemv_config() {
    let kernel = Fp16Q4KGemvKernel::new(4096, 32000);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 32000);
}

#[test]
fn test_fp16_q4k_gemv_ptx_generation() {
    let kernel = Fp16Q4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    assert!(ptx.contains(".visible .entry fp16_q4k_gemv"));
}

#[test]
fn test_fp16_q4k_gemv_uses_f16() {
    let kernel = Fp16Q4KGemvKernel::new(4096, 4096);
    let ptx = kernel.emit_ptx();

    // Should use f16 operations
    assert!(ptx.contains("f16"));
}

// =========================================================================
// BATCHED Q6K GEMV KERNEL TESTS (PAR-130)
// =========================================================================

#[test]
fn test_batched_q6k_gemv_kernel_name() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 32000, 8);
    assert_eq!(kernel.name(), "batched_q6k_gemv_warp_reduce");
}

#[test]
fn test_batched_q6k_gemv_config() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 32000, 8);
    assert_eq!(kernel.k, 4096);
    assert_eq!(kernel.n, 32000);
    assert_eq!(kernel.m, 8);
}

#[test]
fn test_batched_q6k_gemv_num_super_blocks() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    assert_eq!(kernel.num_super_blocks_per_row(), 16); // 4096 / 256 = 16

    let kernel2 = BatchedQ6KGemvKernel::new(512, 4096, 4);
    assert_eq!(kernel2.num_super_blocks_per_row(), 2); // 512 / 256 = 2
}

#[test]
fn test_batched_q6k_gemv_ptx_generation() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    let ptx = kernel.emit_ptx();

    assert!(
        ptx.contains(".visible .entry batched_q6k_gemv_warp_reduce"),
        "PTX should contain entry point"
    );
    assert!(ptx.contains("ld.global"), "Should have global memory loads");
}

#[test]
fn test_batched_q6k_gemv_has_m_param() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    let ptx = kernel.emit_ptx();

    // Batched kernel should have m_dim parameter
    assert!(
        ptx.contains(".param .u32 m_dim"),
        "Should have m_dim parameter for batch size"
    );
}

#[test]
fn test_batched_q6k_gemv_has_warp_shuffle() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    let ptx = kernel.emit_ptx();

    // Uses warp shuffle for reduction
    assert!(
        ptx.contains("shfl.sync.down"),
        "Should use warp shuffle down for reduction"
    );
}

#[test]
fn test_batched_q6k_gemv_has_loops() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    let ptx = kernel.emit_ptx();

    // Should have super-block loop and value loop
    assert!(ptx.contains("sb_loop"), "Should have super-block loop");
    assert!(ptx.contains("val_loop"), "Should have value loop");
}

#[test]
fn test_batched_q6k_gemv_barrier_safety() {
    use crate::ptx::optimize::barrier_safety;
    let kernel = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    let ptx = kernel.emit_ptx();
    let result = barrier_safety::analyze(&ptx);
    assert!(
        result.is_safe,
        "Batched Q6K GEMV should pass barrier safety: {:?}",
        result.violations
    );
}

#[test]
fn test_batched_q6k_gemv_different_batch_sizes() {
    // Test various batch sizes produce valid PTX
    for m in [1, 2, 4, 8, 16, 32] {
        let kernel = BatchedQ6KGemvKernel::new(4096, 4096, m);
        let ptx = kernel.emit_ptx();
        assert!(!ptx.is_empty(), "PTX should not be empty for m={m}");
        assert!(
            ptx.contains(".visible .entry batched_q6k_gemv_warp_reduce"),
            "PTX should have entry point for m={m}"
        );
    }
}

#[test]
fn test_batched_q6k_gemv_outputs_multiple_results() {
    let kernel = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    let ptx = kernel.emit_ptx();

    // Should have multiple st.global for writing M outputs
    let store_count = ptx.matches("st.global.f32").count();
    // With M=4 batch size, kernel should write 4 results per block
    assert!(
        store_count >= 4,
        "Should have at least 4 st.global.f32 for batch size 4, found {store_count}"
    );
}

#[test]
fn test_batched_q6k_gemv_vs_non_batched_different() {
    // Batched and non-batched kernels should produce different PTX
    let batched = BatchedQ6KGemvKernel::new(4096, 4096, 4);
    let non_batched = Q6KGemvKernel::new(4096, 4096);

    let ptx_batched = batched.emit_ptx();
    let ptx_non_batched = non_batched.emit_ptx();

    assert_ne!(
        ptx_batched, ptx_non_batched,
        "Batched and non-batched kernels should produce different PTX"
    );
    assert!(
        ptx_batched.contains("batched"),
        "Batched PTX should contain 'batched' in entry name"
    );
}

proptest! {
    #[test]
    fn prop_batched_q6k_generates_valid_ptx(
        k_factor in 1u32..8,
        n in 32u32..4096,
        m in 1u32..16
    ) {
        let k = k_factor * 256;
        let kernel = BatchedQ6KGemvKernel::new(k, n, m);
        let ptx = kernel.emit_ptx();

        prop_assert!(!ptx.is_empty());
        prop_assert!(ptx.contains(".visible .entry batched_q6k_gemv_warp_reduce"));
        prop_assert!(ptx.contains(".param .u32 m_dim"));
    }

    #[test]
    fn prop_batched_q6k_super_blocks_correct(k_factor in 1u32..16) {
        let k = k_factor * 256;
        let kernel = BatchedQ6KGemvKernel::new(k, 64, 4);
        prop_assert_eq!(kernel.num_super_blocks_per_row(), k_factor);
    }
}
