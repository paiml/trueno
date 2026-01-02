//! Deep PTX bug hunt - find ALL potential issues
//!
//! Run: `cargo run -p trueno-explain --example deep_bug_hunt`

#![allow(clippy::too_many_lines)]

use std::collections::HashMap;
use trueno_explain::{BugSeverity, PtxBugAnalyzer};
use trueno_gpu::kernels::{
    AttentionKernel, BiasActivationKernel, GemmKernel, Kernel, LayerNormKernel, Q5KKernel,
    Q6KKernel, QuantizeKernel, SoftmaxKernel,
};

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    DEEP PTX BUG HUNT (STRICT MODE, NO WHITELIST)              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // Generate ALL kernel variants
    let kernels: Vec<(&str, String)> = vec![
        // GEMM variants
        ("gemm_naive_32", GemmKernel::naive(32, 32, 32).emit_ptx()),
        ("gemm_naive_64", GemmKernel::naive(64, 64, 64).emit_ptx()),
        (
            "gemm_naive_128",
            GemmKernel::naive(128, 128, 128).emit_ptx(),
        ),
        (
            "gemm_naive_256",
            GemmKernel::naive(256, 256, 256).emit_ptx(),
        ),
        ("gemm_tiled_32", GemmKernel::tiled(32, 32, 32, 8).emit_ptx()),
        (
            "gemm_tiled_64",
            GemmKernel::tiled(64, 64, 64, 16).emit_ptx(),
        ),
        (
            "gemm_tiled_128",
            GemmKernel::tiled(128, 128, 128, 32).emit_ptx(),
        ),
        (
            "gemm_tensor_core",
            GemmKernel::tensor_core(64, 64, 64).emit_ptx(),
        ),
        (
            "gemm_wmma_fp16",
            GemmKernel::wmma_fp16(64, 64, 64).emit_ptx(),
        ),
        // Softmax variants
        ("softmax_256", SoftmaxKernel::new(256).emit_ptx()),
        ("softmax_1024", SoftmaxKernel::new(1024).emit_ptx()),
        ("softmax_4096", SoftmaxKernel::new(4096).emit_ptx()),
        // LayerNorm variants
        ("layernorm_128", LayerNormKernel::new(128).emit_ptx()),
        ("layernorm_256", LayerNormKernel::new(256).emit_ptx()),
        ("layernorm_512", LayerNormKernel::new(512).emit_ptx()),
        ("layernorm_1024", LayerNormKernel::new(1024).emit_ptx()),
        // Attention variants
        ("attention_32_32", AttentionKernel::new(32, 32).emit_ptx()),
        ("attention_64_32", AttentionKernel::new(64, 32).emit_ptx()),
        ("attention_64_64", AttentionKernel::new(64, 64).emit_ptx()),
        ("attention_128_64", AttentionKernel::new(128, 64).emit_ptx()),
        ("attention_256_64", AttentionKernel::new(256, 64).emit_ptx()),
        // Quantized kernels (ALL variants)
        ("q4k_32", QuantizeKernel::ggml(32, 32, 256).emit_ptx()),
        ("q4k_64", QuantizeKernel::ggml(64, 64, 256).emit_ptx()),
        ("q4k_128", QuantizeKernel::ggml(128, 128, 256).emit_ptx()),
        ("q5k_32", Q5KKernel::new(32, 32, 256).emit_ptx()),
        ("q5k_64", Q5KKernel::new(64, 64, 256).emit_ptx()),
        ("q5k_128", Q5KKernel::new(128, 128, 256).emit_ptx()),
        ("q6k_32", Q6KKernel::new(32, 32, 256).emit_ptx()),
        ("q6k_64", Q6KKernel::new(64, 64, 256).emit_ptx()),
        ("q6k_128", Q6KKernel::new(128, 128, 256).emit_ptx()),
        // BiasActivation variants (epilogue kernels)
        (
            "bias_activation_none_1024",
            BiasActivationKernel::new(1024, 64).emit_ptx(),
        ),
        (
            "bias_activation_relu_1024",
            BiasActivationKernel::new(1024, 64).with_relu().emit_ptx(),
        ),
        (
            "bias_activation_gelu_1024",
            BiasActivationKernel::new(1024, 64).with_gelu().emit_ptx(),
        ),
        (
            "bias_activation_none_4096",
            BiasActivationKernel::new(4096, 256).emit_ptx(),
        ),
        (
            "bias_activation_relu_4096",
            BiasActivationKernel::new(4096, 256).with_relu().emit_ptx(),
        ),
        (
            "bias_activation_gelu_4096",
            BiasActivationKernel::new(4096, 256).with_gelu().emit_ptx(),
        ),
    ];

    let mut total_bugs = 0;
    let mut p0_bugs = 0;
    let mut p1_bugs = 0;
    let mut p2_bugs = 0;
    let mut bugs_by_class: HashMap<String, usize> = HashMap::new();

    // STRICT MODE - NO WHITELIST - catch EVERYTHING
    let analyzer = PtxBugAnalyzer::strict();

    for (name, ptx) in &kernels {
        let result = analyzer.analyze(ptx);

        let p0 = result.count_by_severity(BugSeverity::Critical);
        let p1 = result.count_by_severity(BugSeverity::High);
        let p2 = result.count_by_severity(BugSeverity::Medium);

        total_bugs += result.bugs.len();
        p0_bugs += p0;
        p1_bugs += p1;
        p2_bugs += p2;

        for bug in &result.bugs {
            *bugs_by_class
                .entry(bug.class.code().to_string())
                .or_insert(0) += 1;
        }

        if result.has_bugs() {
            let icon = if p0 > 0 {
                "🔴"
            } else if p1 > 0 {
                "🟡"
            } else {
                "🟠"
            };
            println!(
                "{} {} - {} bugs ({} P0, {} P1, {} P2)",
                icon,
                name,
                result.bugs.len(),
                p0,
                p1,
                p2
            );
            for bug in &result.bugs {
                println!(
                    "   └─ [{}] {}: {}",
                    bug.class.severity(),
                    bug.class.code(),
                    bug.message
                );
                if let Some(fix) = &bug.fix {
                    println!("      Fix: {}", fix);
                }
            }
        } else {
            println!("✅ {} - CLEAN", name);
        }
    }

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("SUMMARY: {} kernels analyzed", kernels.len());
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("  Total bugs: {}", total_bugs);
    println!("  🔴 P0 Critical: {}", p0_bugs);
    println!("  🟡 P1 High: {}", p1_bugs);
    println!("  🟠 P2 Medium: {}", p2_bugs);

    println!("\nBUGS BY CLASS:");
    let mut sorted_bugs: Vec<_> = bugs_by_class.iter().collect();
    sorted_bugs.sort_by(|a, b| b.1.cmp(a.1));
    for (class, count) in sorted_bugs {
        println!("  {:25} : {}", class, count);
    }

    if p0_bugs > 0 {
        println!(
            "\n⚠️  CRITICAL: {} P0 bugs found - these need immediate attention!",
            p0_bugs
        );
    }

    // =========================================================================
    // PRODUCTION MODE - With Performance Whitelist
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    PRODUCTION MODE (WITH PERFORMANCE WHITELIST)              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let prod_analyzer = PtxBugAnalyzer::strict()
        .with_whitelist(
            "gemm_tensor_core*",
            trueno_explain::PtxBugClass::HighRegisterPressure,
            "Tensor Core WMMA requires many registers for matrix fragments",
        )
        .with_whitelist(
            "gemm_tensor_core*",
            trueno_explain::PtxBugClass::PredicateOverflow,
            "Tensor Core kernels use predicates for bounds checking",
        )
        .with_whitelist(
            "gemm_wmma*",
            trueno_explain::PtxBugClass::HighRegisterPressure,
            "WMMA FP16 requires registers for matrix fragments",
        )
        .with_whitelist(
            "gemm_wmma*",
            trueno_explain::PtxBugClass::PredicateOverflow,
            "WMMA kernels use predicates for tile handling",
        )
        .with_whitelist(
            "flash_attention*",
            trueno_explain::PtxBugClass::HighRegisterPressure,
            "FlashAttention tiling requires registers for Q/K/V/O",
        )
        .with_whitelist(
            "attention*",
            trueno_explain::PtxBugClass::HighRegisterPressure,
            "Attention kernels require registers for tiling",
        )
        .with_whitelist(
            "q4k*",
            trueno_explain::PtxBugClass::HighRegisterPressure,
            "Q4_K dequantization requires registers",
        )
        .with_whitelist(
            "q5k*",
            trueno_explain::PtxBugClass::HighRegisterPressure,
            "Q5_K dequantization requires registers",
        )
        .with_whitelist(
            "q6k*",
            trueno_explain::PtxBugClass::HighRegisterPressure,
            "Q6_K dequantization requires registers",
        );

    let mut prod_bugs = 0;
    let mut prod_p0 = 0;

    for (name, ptx) in &kernels {
        let result = prod_analyzer.analyze(ptx);
        let p0 = result.count_by_severity(BugSeverity::Critical);
        prod_bugs += result.bugs.len();
        prod_p0 += p0;

        if result.has_bugs() {
            let icon = if p0 > 0 { "🔴" } else { "🟡" };
            println!("{} {} - {} bugs remaining", icon, name, result.bugs.len());
        } else {
            println!("✅ {} - CLEAN", name);
        }
    }

    println!("\n══════════════════════════════════════════════════════════════════════════════");
    println!("PRODUCTION SUMMARY");
    println!("══════════════════════════════════════════════════════════════════════════════");
    println!("  Bugs after whitelist: {}", prod_bugs);
    println!("  🔴 P0 Critical: {}", prod_p0);

    if prod_p0 == 0 && prod_bugs == 0 {
        println!("\n✅ ALL KERNELS PASS PRODUCTION QUALITY GATE");
    } else if prod_p0 == 0 {
        println!(
            "\n✅ No critical bugs - {} advisory warnings remain",
            prod_bugs
        );
    }
}
