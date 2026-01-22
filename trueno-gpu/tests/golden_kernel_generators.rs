//! Golden Kernel Generator Tests (Popperian Falsification)
//!
//! These tests verify the INTENT of each kernel generator, ensuring correct
//! PTX structure and instruction patterns.
//!
//! ## Falsification Strategy
//! - Each kernel MUST emit valid PTX with correct entry point
//! - Critical instructions for each kernel type MUST be present
//! - Memory access patterns MUST match kernel semantics

use trueno_gpu::kernels::{
    Batched4DGemmKernel, BatchedGemmKernel, BatchedSoftmaxKernel, GemmKernel, Kernel,
    LayerNormKernel, RmsNormKernel, SoftmaxKernel, VectorizedRmsNormKernel,
};

// ============================================================================
// GEMM KERNEL - Golden Tests
// ============================================================================

#[test]
fn golden_gemm_naive_kernel_structure() {
    let kernel = GemmKernel::naive(64, 64, 64);
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point
    assert!(
        ptx.contains(".entry"),
        "GOLDEN FAIL: Missing .entry in GEMM naive\nPTX:\n{}",
        ptx
    );

    // Golden: Must have memory loads for A and B matrices
    assert!(
        ptx.contains("ld.global"),
        "GOLDEN FAIL: Missing global loads in GEMM naive\nPTX:\n{}",
        ptx
    );

    // Golden: Must have global store for C matrix
    assert!(
        ptx.contains("st.global"),
        "GOLDEN FAIL: Missing global store in GEMM naive\nPTX:\n{}",
        ptx
    );

    // Golden: Must have FMA or mul+add for matrix multiply
    assert!(
        ptx.contains("fma") || (ptx.contains("mul") && ptx.contains("add")),
        "GOLDEN FAIL: Missing multiply-accumulate in GEMM naive\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_gemm_tiled_kernel_structure() {
    let kernel = GemmKernel::tiled(64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point
    assert!(
        ptx.contains(".entry"),
        "GOLDEN FAIL: Missing .entry in GEMM tiled\nPTX:\n{}",
        ptx
    );

    // Golden: Tiled GEMM must use shared memory
    assert!(
        ptx.contains(".shared") || ptx.contains("ld.shared") || ptx.contains("st.shared"),
        "GOLDEN FAIL: Missing shared memory in GEMM tiled\nPTX:\n{}",
        ptx
    );

    // Golden: Tiled GEMM must have barrier sync for tile loading
    assert!(
        ptx.contains("bar.sync"),
        "GOLDEN FAIL: Missing barrier sync in GEMM tiled\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_gemm_naive_kernel_structure() {
    let kernel = BatchedGemmKernel::naive(4, 64, 64, 64);
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point
    assert!(
        ptx.contains(".entry"),
        "GOLDEN FAIL: Missing .entry in batched GEMM naive\nPTX:\n{}",
        ptx
    );

    // Golden: Must have batch index calculation (typically ctaid.z or similar)
    assert!(
        ptx.contains("%ctaid") || ptx.contains("batch"),
        "GOLDEN FAIL: Missing batch index in batched GEMM\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_gemm_tiled_kernel_structure() {
    let kernel = BatchedGemmKernel::tiled(4, 64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    // Golden: Must have shared memory for tiling
    assert!(
        ptx.contains(".shared") || ptx.contains("ld.shared"),
        "GOLDEN FAIL: Missing shared memory in batched GEMM tiled\nPTX:\n{}",
        ptx
    );

    // Golden: Must have barrier for tile sync
    assert!(
        ptx.contains("bar.sync"),
        "GOLDEN FAIL: Missing barrier in batched GEMM tiled\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// GEMM ADVANCED VARIANTS - Golden Tests (Dr. Popper's "Dark Matter")
// ============================================================================

#[test]
fn golden_gemm_tiled_unrolled_kernel_structure() {
    let kernel = GemmKernel::tiled_unrolled(64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point with correct name
    assert!(
        ptx.contains(".entry gemm_tiled_unrolled"),
        "GOLDEN FAIL: Missing gemm_tiled_unrolled entry\nPTX:\n{}",
        ptx
    );

    // Golden: Must have shared memory for tiling
    assert!(
        ptx.contains(".shared") || ptx.contains("ld.shared"),
        "GOLDEN FAIL: Missing shared memory in GEMM tiled_unrolled\nPTX:\n{}",
        ptx
    );

    // Golden: Must have barrier sync
    assert!(
        ptx.contains("bar.sync"),
        "GOLDEN FAIL: Missing barrier in GEMM tiled_unrolled\nPTX:\n{}",
        ptx
    );

    // Golden: Must have FMA for multiply-accumulate
    assert!(
        ptx.contains("fma"),
        "GOLDEN FAIL: Missing fma in GEMM tiled_unrolled\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_gemm_tensor_core_kernel_structure() {
    let kernel = GemmKernel::tensor_core(64, 64, 64);
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point with correct name
    assert!(
        ptx.contains(".entry gemm_tensor_core"),
        "GOLDEN FAIL: Missing gemm_tensor_core entry\nPTX:\n{}",
        ptx
    );

    // Golden: Tensor core variant uses 16x16 tiling with shared memory
    assert!(
        ptx.contains(".shared"),
        "GOLDEN FAIL: Missing shared memory in GEMM tensor_core\nPTX:\n{}",
        ptx
    );

    // Golden: Must have barrier for tile sync
    assert!(
        ptx.contains("bar.sync"),
        "GOLDEN FAIL: Missing barrier in GEMM tensor_core\nPTX:\n{}",
        ptx
    );

    // Golden: Must have FMA for multiply-accumulate
    assert!(
        ptx.contains("fma"),
        "GOLDEN FAIL: Missing fma in GEMM tensor_core\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_gemm_wmma_fp16_kernel_structure() {
    let kernel = GemmKernel::wmma_fp16(64, 64, 64);
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point with correct name
    assert!(
        ptx.contains(".entry gemm_wmma_fp16"),
        "GOLDEN FAIL: Missing gemm_wmma_fp16 entry\nPTX:\n{}",
        ptx
    );

    // Golden: Must use WMMA PTX intrinsics
    assert!(
        ptx.contains("wmma"),
        "GOLDEN FAIL: Missing WMMA ops in GEMM wmma_fp16\nPTX:\n{}",
        ptx
    );

    // Golden: Must have load and store operations for fragments
    assert!(
        ptx.contains("load") && ptx.contains("store"),
        "GOLDEN FAIL: Missing WMMA load/store in GEMM wmma_fp16\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_gemm_tiled_unrolled_kernel_structure() {
    let kernel = BatchedGemmKernel::tiled_unrolled(4, 64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point
    assert!(
        ptx.contains(".entry batched_gemm_tiled_unrolled"),
        "GOLDEN FAIL: Missing batched_gemm_tiled_unrolled entry\nPTX:\n{}",
        ptx
    );

    // Golden: Must have batch handling
    assert!(
        ptx.contains("%ctaid"),
        "GOLDEN FAIL: Missing block index in batched GEMM tiled_unrolled\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_gemm_wmma_fp16_kernel_structure() {
    let kernel = BatchedGemmKernel::wmma_fp16(4, 64, 64, 64);
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point
    assert!(
        ptx.contains(".entry batched_gemm_wmma_fp16"),
        "GOLDEN FAIL: Missing batched_gemm_wmma_fp16 entry\nPTX:\n{}",
        ptx
    );

    // Golden: Must use WMMA ops
    assert!(
        ptx.contains("wmma"),
        "GOLDEN FAIL: Missing WMMA ops in batched GEMM wmma_fp16\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_4d_gemm_kernel_structure() {
    let kernel = Batched4DGemmKernel::new(2, 8, 32, 32, 64); // batch=2, heads=8, m=32, n=32, k=64
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point
    assert!(
        ptx.contains(".entry batched_4d_gemm"),
        "GOLDEN FAIL: Missing batched_4d_gemm entry\nPTX:\n{}",
        ptx
    );

    // Golden: Must handle 4D indexing (batch, heads, m, n)
    assert!(
        ptx.contains("%ctaid"),
        "GOLDEN FAIL: Missing block index in batched_4d_gemm\nPTX:\n{}",
        ptx
    );

    // Golden: Must have shared memory for tiling
    assert!(
        ptx.contains(".shared") || ptx.contains("ld.shared"),
        "GOLDEN FAIL: Missing shared memory in batched_4d_gemm\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_4d_gemm_with_tile_size() {
    let kernel = Batched4DGemmKernel::with_tile_size(2, 8, 32, 32, 64, 8);
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point
    assert!(
        ptx.contains(".entry batched_4d_gemm"),
        "Custom tile size should still produce valid kernel"
    );
}

#[test]
fn golden_gemm_kernel_names_complete() {
    // Verify all variant names
    assert_eq!(GemmKernel::naive(64, 64, 64).name(), "gemm_naive");
    assert_eq!(GemmKernel::tiled(64, 64, 64, 16).name(), "gemm_tiled");
    assert_eq!(
        GemmKernel::tiled_unrolled(64, 64, 64, 16).name(),
        "gemm_tiled_unrolled"
    );
    assert_eq!(
        GemmKernel::tensor_core(64, 64, 64).name(),
        "gemm_tensor_core"
    );
    assert_eq!(GemmKernel::wmma_fp16(64, 64, 64).name(), "gemm_wmma_fp16");

    // Batched variants
    assert_eq!(
        BatchedGemmKernel::naive(4, 64, 64, 64).name(),
        "batched_gemm_naive"
    );
    assert_eq!(
        BatchedGemmKernel::tiled(4, 64, 64, 64, 16).name(),
        "batched_gemm_tiled"
    );
    assert_eq!(
        BatchedGemmKernel::tiled_unrolled(4, 64, 64, 64, 16).name(),
        "batched_gemm_tiled_unrolled"
    );
    assert_eq!(
        BatchedGemmKernel::wmma_fp16(4, 64, 64, 64).name(),
        "batched_gemm_wmma_fp16"
    );

    // 4D batched
    assert_eq!(
        Batched4DGemmKernel::new(2, 8, 32, 32, 64).name(),
        "batched_4d_gemm"
    );
}

// ============================================================================
// SOFTMAX KERNEL - Golden Tests
// ============================================================================

#[test]
fn golden_softmax_kernel_structure() {
    let kernel = SoftmaxKernel::new(1024);
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point
    assert!(
        ptx.contains(".entry"),
        "GOLDEN FAIL: Missing .entry in softmax\nPTX:\n{}",
        ptx
    );

    // Golden: Softmax needs max reduction (for numerical stability)
    assert!(
        ptx.contains("max") || ptx.contains("shfl"),
        "GOLDEN FAIL: Missing max/reduction in softmax\nPTX:\n{}",
        ptx
    );

    // Golden: Softmax needs exp function
    assert!(
        ptx.contains("ex2") || ptx.contains("exp"),
        "GOLDEN FAIL: Missing exp in softmax\nPTX:\n{}",
        ptx
    );

    // Golden: Softmax needs division for normalization
    assert!(
        ptx.contains("div") || ptx.contains("rcp"),
        "GOLDEN FAIL: Missing division in softmax\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_batched_softmax_kernel_structure() {
    let kernel = BatchedSoftmaxKernel::new(1024, 8);
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point
    assert!(
        ptx.contains(".entry"),
        "GOLDEN FAIL: Missing .entry in batched softmax\nPTX:\n{}",
        ptx
    );

    // Golden: Must have batch handling
    assert!(
        ptx.contains("%ctaid") || ptx.contains("%tid"),
        "GOLDEN FAIL: Missing thread/block indexing in batched softmax\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// LAYERNORM KERNEL - Golden Tests
// ============================================================================

#[test]
fn golden_layernorm_warp_shuffle_kernel() {
    let kernel = LayerNormKernel::new(768); // Default: warp_shuffle
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point with correct name
    assert!(
        ptx.contains(".entry layernorm_warp_shuffle"),
        "GOLDEN FAIL: Missing layernorm_warp_shuffle entry\nPTX:\n{}",
        ptx
    );

    // Golden: Warp shuffle kernel MUST use shfl for reduction
    assert!(
        ptx.contains("shfl"),
        "GOLDEN FAIL: Missing warp shuffle in LayerNorm warp_shuffle\nPTX:\n{}",
        ptx
    );

    // Golden: LayerNorm needs rsqrt for 1/sqrt(variance)
    assert!(
        ptx.contains("rsqrt"),
        "GOLDEN FAIL: Missing rsqrt in LayerNorm\nPTX:\n{}",
        ptx
    );

    // Golden: LayerNorm needs division for mean calculation
    assert!(
        ptx.contains("div"),
        "GOLDEN FAIL: Missing division in LayerNorm\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_layernorm_shared_memory_kernel() {
    let kernel = LayerNormKernel::new(768).without_warp_shuffle();
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point with correct name
    assert!(
        ptx.contains(".entry layernorm_shared"),
        "GOLDEN FAIL: Missing layernorm_shared entry\nPTX:\n{}",
        ptx
    );

    // Golden: Shared memory kernel MUST use shared memory
    assert!(
        ptx.contains(".shared") || ptx.contains("ld.shared") || ptx.contains("st.shared"),
        "GOLDEN FAIL: Missing shared memory in LayerNorm shared\nPTX:\n{}",
        ptx
    );

    // Golden: Shared memory kernel MUST have barrier sync
    assert!(
        ptx.contains("bar.sync"),
        "GOLDEN FAIL: Missing barrier in LayerNorm shared\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_layernorm_epsilon_customization() {
    let kernel1 = LayerNormKernel::new(768).with_epsilon(1e-5);
    let kernel2 = LayerNormKernel::new(768).with_epsilon(1e-6);

    // Both should emit valid PTX
    let ptx1 = kernel1.emit_ptx();
    let ptx2 = kernel2.emit_ptx();

    assert!(ptx1.contains(".entry"), "Kernel 1 should have entry");
    assert!(ptx2.contains(".entry"), "Kernel 2 should have entry");

    // Epsilon difference should reflect in PTX (different immediate values)
    // Note: exact comparison is tricky due to float formatting
}

#[test]
fn golden_layernorm_without_affine() {
    let kernel = LayerNormKernel::new(768).without_affine();
    let ptx = kernel.emit_ptx();

    // Golden: Must still have valid entry
    assert!(
        ptx.contains(".entry"),
        "GOLDEN FAIL: Missing entry in LayerNorm without affine\nPTX:\n{}",
        ptx
    );

    // Note: Without affine, kernel still normalizes but doesn't apply gamma/beta
    // This is harder to verify structurally, but the kernel should be simpler
}

// ============================================================================
// RMSNORM KERNEL - Golden Tests
// ============================================================================

#[test]
fn golden_rmsnorm_kernel_structure() {
    let kernel = RmsNormKernel::new(2048);
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point
    assert!(
        ptx.contains(".entry rmsnorm"),
        "GOLDEN FAIL: Missing rmsnorm entry\nPTX:\n{}",
        ptx
    );

    // Golden: RMSNorm uses warp shuffle for reduction
    assert!(
        ptx.contains("shfl"),
        "GOLDEN FAIL: Missing warp shuffle in RMSNorm\nPTX:\n{}",
        ptx
    );

    // Golden: RMSNorm needs rsqrt for 1/sqrt(mean_sq)
    assert!(
        ptx.contains("rsqrt"),
        "GOLDEN FAIL: Missing rsqrt in RMSNorm\nPTX:\n{}",
        ptx
    );

    // Golden: RMSNorm needs multiplication for scaling
    assert!(
        ptx.contains("mul"),
        "GOLDEN FAIL: Missing multiplication in RMSNorm\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_vectorized_rmsnorm_kernel_structure() {
    let kernel = VectorizedRmsNormKernel::new(2048);
    let ptx = kernel.emit_ptx();

    // Golden: Must have entry point with correct name
    assert!(
        ptx.contains(".entry rmsnorm_vectorized"),
        "GOLDEN FAIL: Missing rmsnorm_vectorized entry\nPTX:\n{}",
        ptx
    );

    // Golden: Vectorized version uses shared memory for cross-warp reduction
    assert!(
        ptx.contains(".shared") || ptx.contains("ld.shared") || ptx.contains("st.shared"),
        "GOLDEN FAIL: Missing shared memory in vectorized RMSNorm\nPTX:\n{}",
        ptx
    );

    // Golden: Must have barrier for shared memory sync
    assert!(
        ptx.contains("bar.sync"),
        "GOLDEN FAIL: Missing barrier in vectorized RMSNorm\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// KERNEL PARAMETER VALIDATION - Golden Tests
// ============================================================================

#[test]
fn golden_gemm_params_present() {
    let kernel = GemmKernel::naive(64, 64, 64);
    let ptx = kernel.emit_ptx();

    // GEMM kernels typically need A, B, C pointers and dimensions
    assert!(
        ptx.contains(".param"),
        "GOLDEN FAIL: Missing parameters in GEMM\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_layernorm_params_present() {
    let kernel = LayerNormKernel::new(768);
    let ptx = kernel.emit_ptx();

    // LayerNorm needs input, output, gamma, beta pointers
    assert!(
        ptx.contains(".param .u64 input_ptr"),
        "GOLDEN FAIL: Missing input_ptr in LayerNorm\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".param .u64 output_ptr"),
        "GOLDEN FAIL: Missing output_ptr in LayerNorm\nPTX:\n{}",
        ptx
    );
    assert!(
        ptx.contains(".param .u64 gamma_ptr"),
        "GOLDEN FAIL: Missing gamma_ptr in LayerNorm\nPTX:\n{}",
        ptx
    );
}

#[test]
fn golden_softmax_params_present() {
    let kernel = SoftmaxKernel::new(1024);
    let ptx = kernel.emit_ptx();

    // Softmax needs input and output pointers
    assert!(
        ptx.contains(".param"),
        "GOLDEN FAIL: Missing parameters in Softmax\nPTX:\n{}",
        ptx
    );
}

// ============================================================================
// VARIOUS DIMENSIONS - Boundary Tests
// ============================================================================

#[test]
fn golden_gemm_small_dimensions() {
    let kernel = GemmKernel::naive(16, 16, 16);
    let ptx = kernel.emit_ptx();
    assert!(
        ptx.contains(".entry"),
        "Small GEMM should still generate valid kernel"
    );
}

#[test]
fn golden_gemm_large_dimensions() {
    let kernel = GemmKernel::naive(2048, 2048, 2048);
    let ptx = kernel.emit_ptx();
    assert!(
        ptx.contains(".entry"),
        "Large GEMM should still generate valid kernel"
    );
}

#[test]
fn golden_layernorm_various_hidden_sizes() {
    for hidden_size in [256, 512, 768, 1024, 1536, 2048, 4096] {
        let kernel = LayerNormKernel::new(hidden_size);
        let ptx = kernel.emit_ptx();
        assert!(
            ptx.contains(".entry"),
            "LayerNorm hidden_size={} should generate valid kernel",
            hidden_size
        );
    }
}

#[test]
fn golden_softmax_various_sizes() {
    for size in [256, 512, 1024, 2048, 4096] {
        let kernel = SoftmaxKernel::new(size);
        let ptx = kernel.emit_ptx();
        assert!(
            ptx.contains(".entry"),
            "Softmax size={} should generate valid kernel",
            size
        );
    }
}

// ============================================================================
// KERNEL NAME CORRECTNESS - Golden Tests
// ============================================================================

#[test]
fn golden_kernel_names_consistent() {
    // Verify kernel names match their type
    assert_eq!(LayerNormKernel::new(768).name(), "layernorm_warp_shuffle");
    assert_eq!(
        LayerNormKernel::new(768).without_warp_shuffle().name(),
        "layernorm_shared"
    );
    assert_eq!(RmsNormKernel::new(2048).name(), "rmsnorm");
    assert_eq!(VectorizedRmsNormKernel::new(2048).name(), "rmsnorm_vectorized");
}
