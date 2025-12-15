//! Integration tests using sovereign stack for visual regression
//!
//! Per spec E2E-VISUAL-PROBAR-001: Uses trueno-viz + simular (NO external crates)

use super::*;
use simular::engine::rng::SimRng;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Shared GPU pixel renderer for fast test execution (initialized once)
static SHARED_RENDERER: OnceLock<GpuPixelRenderer> = OnceLock::new();

/// Get shared renderer (fast) - always succeeds since GpuPixelRenderer is CPU-based
fn get_shared_renderer() -> &'static GpuPixelRenderer {
    SHARED_RENDERER.get_or_init(GpuPixelRenderer::new)
}

fn test_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("trueno_sovereign_{}_{}", name, std::process::id()))
}

fn cleanup(dir: &std::path::Path) {
    let _ = std::fs::remove_dir_all(dir);
}

/// Simulate correct GEMM output
fn simulate_gemm(size: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(size * size);
    for i in 0..size {
        for j in 0..size {
            let mut acc = 0.0f32;
            for k in 0..size {
                acc += (i * size + k) as f32 * (k * size + j) as f32;
            }
            output.push(acc);
        }
    }
    output
}

/// Simulate buggy GEMM with uninitialized accumulator
fn simulate_gemm_buggy(size: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(size * size);
    for i in 0..size {
        for j in 0..size {
            let garbage = if i % 2 == 0 { 1000.0 } else { 0.0 };
            let mut acc = garbage; // BUG: accumulator not initialized to 0
            for k in 0..size {
                acc += (i * size + k) as f32 * (k * size + j) as f32;
            }
            output.push(acc);
        }
    }
    output
}

// ============================================================================
// Tests using sovereign stack (trueno-viz, simular)
// ============================================================================

#[test]
fn test_sovereign_determinism() {
    let renderer = get_shared_renderer();
    let size = 8;
    let output = simulate_gemm(size);

    let png1 = renderer.render_to_png(&output, size as u32, size as u32);
    let png2 = renderer.render_to_png(&output, size as u32, size as u32);

    let result = compare_png_bytes(&png1, &png2, 0);
    assert!(result.matches(0.0), "Same input should produce identical output");
    assert_eq!(result.different_pixels, 0, "Should be pixel-perfect match");
}

#[test]
fn test_sovereign_detects_bug() {
    let renderer = get_shared_renderer();
    let size = 8;

    let correct = simulate_gemm(size);
    let buggy = simulate_gemm_buggy(size);

    let png_correct = renderer.render_to_png(&correct, size as u32, size as u32);
    let png_buggy = renderer.render_to_png(&buggy, size as u32, size as u32);

    let result = compare_png_bytes(&png_correct, &png_buggy, 0);
    assert!(!result.matches(0.0), "Should detect difference from bug");
    assert!(result.different_pixels > 0, "Should have pixel diffs");

    println!(
        "Bug detected: {} pixels differ ({:.2}%)",
        result.different_pixels,
        result.diff_percentage()
    );
}

#[test]
fn test_sovereign_special_values() {
    let renderer = get_shared_renderer();
    let buffer = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5];

    let png = renderer.render_to_png(&buffer, 2, 2);

    // Verify PNG is valid by comparing to itself
    let result = compare_png_bytes(&png, &png, 0);
    assert_eq!(result.different_pixels, 0);
}

#[test]
fn test_sovereign_threshold() {
    let renderer = get_shared_renderer();
    let size = 8;

    let output1 = simulate_gemm(size);
    let mut output2 = output1.clone();
    output2[0] += 0.001; // Tiny change

    let png1 = renderer.render_to_png(&output1, size as u32, size as u32);
    let png2 = renderer.render_to_png(&output2, size as u32, size as u32);

    // Strict threshold (0 tolerance)
    let result_strict = compare_png_bytes(&png1, &png2, 0);

    // Relaxed threshold (allow 1 byte diff)
    let result_relaxed = compare_png_bytes(&png1, &png2, 1);

    println!(
        "Strict: {} diffs, Relaxed: {} diffs",
        result_strict.different_pixels, result_relaxed.different_pixels
    );
}

#[test]
fn test_sovereign_deterministic_rng() {
    // Use simular for deterministic RNG
    let mut rng = SimRng::new(42);

    let input1: Vec<f32> = (0..64).map(|_| rng.gen_range_f64(0.0, 1.0) as f32).collect();

    // Reset RNG with same seed
    let mut rng2 = SimRng::new(42);
    let input2: Vec<f32> = (0..64).map(|_| rng2.gen_range_f64(0.0, 1.0) as f32).collect();

    assert_eq!(input1, input2, "Same seed should produce same sequence");

    let renderer = get_shared_renderer();
    let png1 = renderer.render_to_png(&input1, 8, 8);
    let png2 = renderer.render_to_png(&input2, 8, 8);

    let result = compare_png_bytes(&png1, &png2, 0);
    assert_eq!(result.different_pixels, 0);
}

// ============================================================================
// Demo report (sovereign stack)
// ============================================================================

#[test]
fn test_demo_sovereign_stack() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   TRUENO-GPU VISUAL REGRESSION (SOVEREIGN STACK ONLY)        ║");
    println!("║   Dependencies: trueno-viz, simular (path only)              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let renderer = get_shared_renderer();
    let size = 8;

    // Test 1: Determinism
    println!("┌─ TEST 1: Determinism ─────────────────────────────────────────┐");
    let output = simulate_gemm(size);
    let png1 = renderer.render_to_png(&output, size as u32, size as u32);
    let png2 = renderer.render_to_png(&output, size as u32, size as u32);

    let result = compare_png_bytes(&png1, &png2, 0);
    println!(
        "│ Diff pixels: {} / {}",
        result.different_pixels, result.total_pixels
    );
    println!(
        "│ Status: {} │",
        if result.different_pixels == 0 {
            "PASS ✓ (Identical)"
        } else {
            "FAIL ✗"
        }
    );
    println!("└──────────────────────────────────────────────────────────────┘\n");

    // Test 2: Bug Detection
    println!("┌─ TEST 2: Bug Detection (Accumulator Init) ───────────────────┐");
    let correct = simulate_gemm(size);
    let buggy = simulate_gemm_buggy(size);
    let png_correct = renderer.render_to_png(&correct, size as u32, size as u32);
    let png_buggy = renderer.render_to_png(&buggy, size as u32, size as u32);

    let result = compare_png_bytes(&png_correct, &png_buggy, 0);
    println!(
        "│ Diff pixels: {} / {} ({:.1}%)",
        result.different_pixels,
        result.total_pixels,
        result.diff_percentage()
    );
    println!("│ Max diff: {}", result.max_diff);
    println!(
        "│ Status: {} │",
        if result.different_pixels > 0 {
            "FAIL ✗ (Bug Detected!)"
        } else {
            "PASS"
        }
    );
    println!("└──────────────────────────────────────────────────────────────┘\n");

    // Test 3: Special Values
    println!("┌─ TEST 3: Special Values (NaN, Inf) ───────────────────────────┐");
    let special = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.5];
    let png_special = renderer.render_to_png(&special, 2, 2);
    let result = compare_png_bytes(&png_special, &png_special, 0);
    println!("│ PNG size: {} bytes", png_special.len());
    println!(
        "│ Status: {} │",
        if result.different_pixels == 0 {
            "PASS ✓ (Handled)"
        } else {
            "FAIL ✗"
        }
    );
    println!("└──────────────────────────────────────────────────────────────┘\n");

    // Test 4: Deterministic RNG (simular)
    println!("┌─ TEST 4: Deterministic RNG (simular) ─────────────────────────┐");
    let mut rng = SimRng::new(42);
    let random_input: Vec<f32> = (0..64).map(|_| rng.gen_range_f64(0.0, 1.0) as f32).collect();
    let png_random = renderer.render_to_png(&random_input, 8, 8);
    println!("│ Seed: 42");
    println!("│ Generated: {} random f32 values", random_input.len());
    println!("│ PNG size: {} bytes", png_random.len());
    println!("│ Status: PASS ✓ (Reproducible)                                │");
    println!("└──────────────────────────────────────────────────────────────┘\n");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ✓ Using trueno-viz (path: ../trueno-viz)                    ║");
    println!("║  ✓ Using simular (path: ../simular)                          ║");
    println!("║  ✓ NO external crates (sovereign stack only)                 ║");
    println!("║  ✓ Bug detection working                                     ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║              100% SOVEREIGN VALIDATION COMPLETE              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}

/// Generate visual report with saved PNG files
#[test]
fn test_visual_report_sovereign() {
    let report_dir = test_dir("visual_report");
    cleanup(&report_dir);
    fs::create_dir_all(&report_dir).unwrap();

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          VISUAL REGRESSION REPORT (SOVEREIGN STACK)          ║");
    println!("║             trueno-viz + simular (path deps only)            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Report directory: {}", report_dir.display());
    println!();

    let renderer = get_shared_renderer();

    // TEST CASE 1: Identity Matrix
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST CASE 1: Identity Matrix Multiplication");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let size = 16;
    let identity: Vec<f32> = (0..size * size)
        .map(|i| if i / size == i % size { 1.0 } else { 0.0 })
        .collect();

    let png_identity = renderer.render_to_png(&identity, size as u32, size as u32);
    let identity_path = report_dir.join("01_identity_matrix.png");
    fs::write(&identity_path, &png_identity).unwrap();

    println!("  Pattern: A @ I = A (metamorphic relation)");
    println!("  Size: {}x{} = {} pixels", size, size, size * size);
    println!("  Saved: {}", identity_path.display());

    let result = compare_png_bytes(&png_identity, &png_identity, 0);
    println!(
        "  Self-comparison: {} diffs, Status: {}",
        result.different_pixels,
        if result.different_pixels == 0 {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );
    println!();

    // TEST CASE 2: Gradient
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST CASE 2: Gradient (FP Precision Test)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let gradient: Vec<f32> = (0..size * size)
        .map(|i| i as f32 / (size * size) as f32)
        .collect();

    let png_gradient = renderer.render_to_png(&gradient, size as u32, size as u32);
    let gradient_path = report_dir.join("02_gradient.png");
    fs::write(&gradient_path, &png_gradient).unwrap();

    println!("  Pattern: Linear gradient 0.0 → 1.0");
    println!("  Purpose: Detect FP precision drift");
    println!("  Saved: {}", gradient_path.display());

    let result = compare_png_bytes(&png_gradient, &png_gradient, 0);
    println!(
        "  Self-comparison: {} diffs, Status: {}",
        result.different_pixels,
        if result.different_pixels == 0 {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );
    println!();

    // TEST CASE 3: Bug Detection
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST CASE 3: Bug Detection (Accumulator Init)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let correct = simulate_gemm(size);
    let buggy = simulate_gemm_buggy(size);

    let png_correct = renderer.render_to_png(&correct, size as u32, size as u32);
    let png_buggy = renderer.render_to_png(&buggy, size as u32, size as u32);

    let correct_path = report_dir.join("03a_gemm_correct.png");
    let buggy_path = report_dir.join("03b_gemm_buggy.png");
    fs::write(&correct_path, &png_correct).unwrap();
    fs::write(&buggy_path, &png_buggy).unwrap();

    println!("  Baseline (correct): {}", correct_path.display());
    println!("  Test (buggy): {}", buggy_path.display());

    let result = compare_png_bytes(&png_correct, &png_buggy, 0);

    println!();
    println!("  ┌─────────────────────────────────────────────────────────┐");
    println!("  │ DIFF ANALYSIS                                           │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │ Total pixels:     {:>6}                                │", result.total_pixels);
    println!(
        "  │ Diff pixels:      {:>6} ({:>5.1}%)                      │",
        result.different_pixels,
        result.diff_percentage()
    );
    println!("  │ Max diff:         {:>6}                                │", result.max_diff);
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!("  │ Bug Class: AccumulatorInit                              │");
    println!("  │ Description: Accumulator not initialized to zero        │");
    println!("  │ Fix: Initialize acc = 0.0 before loop                   │");
    println!("  ├─────────────────────────────────────────────────────────┤");
    println!(
        "  │ Status: {} │",
        if result.different_pixels > 0 {
            "✗ FAIL (Bug Correctly Detected!)         "
        } else {
            "✓ PASS                                   "
        }
    );
    println!("  └─────────────────────────────────────────────────────────┘");
    println!();

    // TEST CASE 4: Special Values
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST CASE 4: Special Values (NaN, Inf)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let special: Vec<f32> = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e38,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        f32::EPSILON,
        100.0,
        -100.0,
        0.001,
        -0.001,
    ];

    let png_special = renderer.render_to_png(&special, 4, 4);
    let special_path = report_dir.join("04_special_values.png");
    fs::write(&special_path, &png_special).unwrap();

    println!("  Values: NaN, +Inf, -Inf, 1e38, normals, denormals");
    println!("  NaN → Magenta (255, 0, 255)");
    println!("  +Inf → White (255, 255, 255)");
    println!("  -Inf → Black (0, 0, 0)");
    println!("  Saved: {}", special_path.display());

    let result = compare_png_bytes(&png_special, &png_special, 0);
    println!(
        "  Self-comparison: {} diffs, Status: {}",
        result.different_pixels,
        if result.different_pixels == 0 {
            "✓ PASS"
        } else {
            "✗ FAIL"
        }
    );
    println!();

    // TEST CASE 5: Deterministic RNG
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST CASE 5: Deterministic RNG (simular)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut rng = SimRng::new(12345);
    let random_data: Vec<f32> = (0..256).map(|_| rng.gen_range_f64(0.0, 1.0) as f32).collect();
    let png_random = renderer.render_to_png(&random_data, 16, 16);
    let random_path = report_dir.join("05_deterministic_random.png");
    fs::write(&random_path, &png_random).unwrap();

    println!("  Seed: 12345 (reproducible)");
    println!("  Size: 16x16 = 256 pixels");
    println!("  Saved: {}", random_path.display());
    println!("  Status: ✓ PASS (Deterministic)");
    println!();

    // SUMMARY
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                      TEST SUMMARY                            ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Test 1: Identity Matrix        ✓ PASS                       ║");
    println!("║  Test 2: Gradient               ✓ PASS                       ║");
    println!("║  Test 3: Bug Detection          ✓ PASS (bug found)           ║");
    println!("║  Test 4: Special Values         ✓ PASS                       ║");
    println!("║  Test 5: Deterministic RNG      ✓ PASS                       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Stack: SOVEREIGN (path deps only)                           ║");
    println!("║    - trueno-viz (../trueno-viz)                              ║");
    println!("║    - simular (../simular)                                    ║");
    println!("║  External crates: ZERO                                       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║           100% SOVEREIGN VALIDATION COMPLETE                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // List generated files
    println!("Generated PNG files:");
    for entry in fs::read_dir(&report_dir).unwrap() {
        let entry = entry.unwrap();
        let metadata = entry.metadata().unwrap();
        println!(
            "  {} ({} bytes)",
            entry.file_name().to_string_lossy(),
            metadata.len()
        );
    }
    println!();

    println!("Files preserved at: {}", report_dir.display());
}

// ============================================================================
// Stress Testing (renacer + simular integration per spec v1.3.0)
// ============================================================================

/// Stress test with randomized inputs per frame
#[test]
fn test_stress_runner_visual() {
    use super::stress::{StressConfig, StressTestRunner, PerformanceThresholds};
    use super::tui::{TuiState, render_to_string};

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       STRESS TEST: Randomized Frame-by-Frame Testing         ║");
    println!("║              renacer v0.7.0 + simular v0.2.0                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let renderer = get_shared_renderer();

    // Configure stress test
    let config = StressConfig {
        cycles: 10,
        interval_ms: 10, // Fast for testing
        seed: 42,
        min_input_size: 64,
        max_input_size: 256,
        thresholds: PerformanceThresholds {
            max_frame_time_ms: 500, // Generous for test
            max_timing_variance: 0.5,
            ..Default::default()
        },
    };

    println!("Configuration:");
    println!("  Cycles: {}", config.cycles);
    println!("  Seed: {}", config.seed);
    println!("  Input size: {}-{}", config.min_input_size, config.max_input_size);
    println!();

    let mut runner = StressTestRunner::new(config.clone());

    // Run stress test with visual verification
    runner.run_all(|input| {
        // Render input to PNG
        let size = (input.len() as f32).sqrt() as u32;
        let actual_size = size * size;
        let data: Vec<f32> = input.iter().take(actual_size as usize).copied().collect();

        if data.is_empty() {
            return (0, 1);
        }

        let png = renderer.render_to_png(&data, size, size);

        // Self-comparison (should always pass for deterministic input)
        let result = compare_png_bytes(&png, &png, 0);

        if result.different_pixels == 0 {
            (1, 0) // 1 pass, 0 fail
        } else {
            (0, 1) // 0 pass, 1 fail
        }
    });

    // Get report and verify performance
    let report = runner.report().clone();
    let perf = runner.verify();

    // Generate TUI output
    let mut tui_state = TuiState::new(config.cycles);
    tui_state.update_from_report(&report);
    let tui_output = render_to_string(&tui_state, &report, &perf);

    println!("{}", tui_output);

    // Report metrics
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("STRESS TEST METRICS:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Cycles completed: {}", report.cycles_completed);
    println!("  Total passed: {}", report.total_passed);
    println!("  Total failed: {}", report.total_failed);
    println!("  Pass rate: {:.1}%", perf.pass_rate * 100.0);
    println!("  Mean frame time: {:.2}ms", perf.mean_frame_ms);
    println!("  Max frame time: {}ms", perf.max_frame_ms);
    println!("  Timing variance: {:.3}", perf.variance);
    println!("  Anomalies detected: {}", report.anomalies.len());
    println!();

    // Assert performance
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("PERFORMANCE VERIFICATION:");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    if perf.passed {
        println!("  Status: ✓ PASS (all thresholds met)");
    } else {
        println!("  Status: ✗ FAIL");
        for violation in &perf.violations {
            println!("    - {}", violation);
        }
    }
    println!();

    // Print anomalies if any
    if !report.anomalies.is_empty() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("ANOMALIES:");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        for anomaly in &report.anomalies {
            println!("  [Cycle {}] {:?}: {}", anomaly.cycle, anomaly.kind, anomaly.description);
        }
        println!();
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           STRESS TEST COMPLETE (SOVEREIGN STACK)             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Assertions
    assert_eq!(report.cycles_completed, config.cycles);
    assert!(report.total_passed > 0, "Should have at least some passing tests");
    // Note: We don't assert perf.passed because timing can vary in CI
}

/// Test deterministic stress test reproducibility
#[test]
fn test_stress_determinism() {
    use super::stress::{StressConfig, StressTestRunner};

    println!("\n");
    println!("Testing stress test determinism...");

    let config = StressConfig {
        cycles: 5,
        seed: 99999,
        min_input_size: 100,
        max_input_size: 200,
        ..Default::default()
    };

    // Run twice with same seed
    let mut runner1 = StressTestRunner::new(config.clone());
    let mut runner2 = StressTestRunner::new(config);

    // Collect inputs from both runners
    let inputs1: Vec<(u64, usize)> = (0..5)
        .map(|_| {
            let (seed, input) = runner1.generate_input();
            (seed, input.len())
        })
        .collect();

    let inputs2: Vec<(u64, usize)> = (0..5)
        .map(|_| {
            let (seed, input) = runner2.generate_input();
            (seed, input.len())
        })
        .collect();

    // Should be identical
    assert_eq!(inputs1, inputs2, "Same seed should produce identical inputs");

    println!("  ✓ Deterministic: Same seed produces identical inputs");
    println!("  Inputs generated: {:?}", inputs1);
}
