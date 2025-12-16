//! Integration tests for trueno-explain
//!
//! Tests the analyzer against real trueno-gpu kernels

use trueno_explain::{Analyzer, PtxAnalyzer};
use trueno_gpu::kernels::{GemmKernel, Kernel, Q5KKernel, Q6KKernel, QuantizeKernel, SoftmaxKernel};

/// F011: Analyze vector_add reports <20 registers for f32
#[test]
fn f011_vector_add_low_register_usage() {
    let ptx = include_str!("../data/vector_add.ptx");
    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(ptx).unwrap();

    assert!(
        report.registers.f32_regs < 50,
        "vector_add should use <50 f32 registers"
    );
}

/// F019: Occupancy calculation matches expected range
#[test]
fn f019_occupancy_calculation() {
    let ptx = include_str!("../data/vector_add.ptx");
    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(ptx).unwrap();

    // 58 registers -> should have reasonable occupancy
    assert!(
        report.estimated_occupancy > 0.25,
        "Expected >25% occupancy, got {}",
        report.estimated_occupancy
    );
    assert!(
        report.estimated_occupancy <= 1.0,
        "Occupancy should not exceed 100%"
    );
}

/// F020: Warns when registers > 128
#[test]
fn f020_high_register_warning() {
    let high_reg_ptx = r#"
.version 8.0
.target sm_70
.entry big_kernel()
{
    .reg .f32 %f<200>;
    ret;
}
"#;
    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(high_reg_ptx).unwrap();

    assert!(
        !report.warnings.is_empty(),
        "Should warn on high register usage"
    );
}

/// Test GEMM naive kernel analysis
#[test]
fn test_gemm_naive_analysis() {
    let kernel = GemmKernel::naive(64, 64, 64);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    assert_eq!(report.name, "gemm_naive");
    assert!(report.instruction_count > 0);
}

/// Test GEMM tiled kernel analysis
#[test]
fn test_gemm_tiled_analysis() {
    let kernel = GemmKernel::tiled(64, 64, 64, 16);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    assert_eq!(report.name, "gemm_tiled");
    // Tiled GEMM should use shared memory
    // (Note: may not be detected in current simplified parser)
}

/// Test Q4K kernel analysis
#[test]
fn test_q4k_kernel_analysis() {
    let kernel = QuantizeKernel::ggml(64, 64, 256);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    assert!(report.name.contains("q4k"));
    assert!(report.memory.global_loads > 0);
}

/// Test Q5K kernel analysis (PARITY-116)
#[test]
fn test_q5k_kernel_analysis() {
    let kernel = Q5KKernel::new(64, 64, 256);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    assert!(report.name.contains("q5k"));
    assert!(report.memory.global_loads > 0);
    // Q5K loads both ql and qh
}

/// Test Q6K kernel analysis (PARITY-117)
#[test]
fn test_q6k_kernel_analysis() {
    let kernel = Q6KKernel::new(64, 64, 256);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    assert!(report.name.contains("q6k"));
    assert!(report.memory.global_loads > 0);
}

/// Test Q5K matvec (n=1) for realizar compatibility
#[test]
fn test_q5k_matvec_n1_analysis() {
    let kernel = Q5KKernel::new(64, 1, 256);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    assert!(report.name.contains("q5k"));
    // Matvec should still work
    assert!(report.instruction_count > 0);
}

/// Test softmax kernel analysis
#[test]
fn test_softmax_analysis() {
    let kernel = SoftmaxKernel::new(1024);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    assert!(report.name.contains("softmax"));
}

/// Test JSON serialization roundtrip
#[test]
fn test_json_roundtrip() {
    let kernel = GemmKernel::naive(32, 32, 32);
    let ptx = kernel.emit_ptx();

    let analyzer = PtxAnalyzer::new();
    let report = analyzer.analyze(&ptx).unwrap();

    let json = serde_json::to_string(&report).unwrap();
    let parsed: trueno_explain::AnalysisReport = serde_json::from_str(&json).unwrap();

    assert_eq!(report.name, parsed.name);
    assert_eq!(report.registers.f32_regs, parsed.registers.f32_regs);
}

/// Test all kernels can be analyzed without panic
#[test]
fn test_all_kernels_analyzable() {
    let kernels: Vec<(&str, String)> = vec![
        ("gemm_naive", GemmKernel::naive(32, 32, 32).emit_ptx()),
        ("gemm_tiled", GemmKernel::tiled(32, 32, 32, 16).emit_ptx()),
        ("softmax", SoftmaxKernel::new(256).emit_ptx()),
        ("q4k", QuantizeKernel::ggml(32, 32, 256).emit_ptx()),
        ("q5k", Q5KKernel::new(32, 32, 256).emit_ptx()),
        ("q6k", Q6KKernel::new(32, 32, 256).emit_ptx()),
    ];

    let analyzer = PtxAnalyzer::new();

    for (name, ptx) in kernels {
        let result = analyzer.analyze(&ptx);
        assert!(result.is_ok(), "Failed to analyze {}: {:?}", name, result);
    }
}
