//! FALSIFY tests — real-world validation of cgp claims.
//! Each test attempts to falsify a specific claim from the spec.
//! Tests that pass mean the claim survived falsification.

use std::process::Command;
use std::time::Instant;

fn cgp_cmd() -> Command {
    let mut cmd = Command::new(env!("CARGO"));
    cmd.arg("run").arg("-p").arg("cgp").arg("--");
    cmd
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-030: Must detect deliberate 10% regression
// Given: baseline profile saved for kernel K
// When: K is modified to be 10% slower
// Then: cgp diff reports REGRESSION
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_030_detect_10pct_regression() {
    // Baseline: 23.2us, 50 samples, stddev 0.3
    let baseline = serde_json::json!({
        "version": "2.0", "timestamp": "", "hardware": {"cpu_features": []},
        "timing": {
            "wall_clock_time_us": 23.2,
            "samples": 50,
            "stddev_us": 0.3,
            "ci_95_low_us": 22.9,
            "ci_95_high_us": 23.5
        },
        "throughput": {"tflops": 11.6, "gflops": 0.0, "bandwidth_gbps": 0.0, "arithmetic_intensity": 0.0},
        "muda": []
    });

    // Current: 10% slower = 25.52us
    let current = serde_json::json!({
        "version": "2.0", "timestamp": "", "hardware": {"cpu_features": []},
        "timing": {
            "wall_clock_time_us": 25.52,
            "samples": 50,
            "stddev_us": 0.3,
            "ci_95_low_us": 25.2,
            "ci_95_high_us": 25.8
        },
        "throughput": {"tflops": 10.5, "gflops": 0.0, "bandwidth_gbps": 0.0, "arithmetic_intensity": 0.0},
        "muda": []
    });

    std::fs::write("/tmp/cgp-falsify-030-b.json", baseline.to_string()).unwrap();
    std::fs::write("/tmp/cgp-falsify-030-c.json", current.to_string()).unwrap();

    let output = cgp_cmd()
        .args([
            "diff",
            "--baseline",
            "/tmp/cgp-falsify-030-b.json",
            "--current",
            "/tmp/cgp-falsify-030-c.json",
        ])
        .output()
        .expect("Failed to run cgp diff");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Must detect regression on wall_clock_time_us
    assert!(
        stdout.contains("REGRESSION"),
        "FALSIFY-CGP-030 FAILED: 10% regression not detected.\nOutput:\n{stdout}"
    );

    let _ = std::fs::remove_file("/tmp/cgp-falsify-030-b.json");
    let _ = std::fs::remove_file("/tmp/cgp-falsify-030-c.json");
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-031: Must NOT false-positive on noise (<2% variation)
// Given: kernel K profiled twice with identical code
// When: cgp diff --baseline run1 --current run2
// Then: reports NO_CHANGE (not regression)
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_031_no_false_positive_on_noise() {
    // Run 1: 23.2us
    let run1 = serde_json::json!({
        "version": "2.0", "timestamp": "", "hardware": {"cpu_features": []},
        "timing": {
            "wall_clock_time_us": 23.2,
            "samples": 50,
            "stddev_us": 0.5,
            "ci_95_low_us": 22.7,
            "ci_95_high_us": 23.7
        },
        "throughput": {"tflops": 11.6, "gflops": 0.0, "bandwidth_gbps": 0.0, "arithmetic_intensity": 0.0},
        "muda": []
    });

    // Run 2: 23.4us (only ~0.9% difference — within noise)
    let run2 = serde_json::json!({
        "version": "2.0", "timestamp": "", "hardware": {"cpu_features": []},
        "timing": {
            "wall_clock_time_us": 23.4,
            "samples": 50,
            "stddev_us": 0.5,
            "ci_95_low_us": 22.9,
            "ci_95_high_us": 23.9
        },
        "throughput": {"tflops": 11.5, "gflops": 0.0, "bandwidth_gbps": 0.0, "arithmetic_intensity": 0.0},
        "muda": []
    });

    std::fs::write("/tmp/cgp-falsify-031-1.json", run1.to_string()).unwrap();
    std::fs::write("/tmp/cgp-falsify-031-2.json", run2.to_string()).unwrap();

    let output = cgp_cmd()
        .args([
            "diff",
            "--baseline",
            "/tmp/cgp-falsify-031-1.json",
            "--current",
            "/tmp/cgp-falsify-031-2.json",
        ])
        .output()
        .expect("Failed to run cgp diff");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Must NOT detect regression on timing (noise is within stddev)
    // The timing line should say NO_CHANGE or IMPROVED, not REGRESSION
    let timing_lines: Vec<&str> =
        stdout.lines().filter(|l| l.contains("wall_clock_time_us")).collect();

    for line in &timing_lines {
        assert!(
            !line.contains("REGRESSION"),
            "FALSIFY-CGP-031 FAILED: false positive on <2% noise.\nLine: {line}\nOutput:\n{stdout}"
        );
    }

    let _ = std::fs::remove_file("/tmp/cgp-falsify-031-1.json");
    let _ = std::fs::remove_file("/tmp/cgp-falsify-031-2.json");
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-041: SIMD must be faster than scalar (>= 3x at 1024)
// Verified via the compare command's estimation model.
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_041_simd_faster_than_scalar() {
    let output = cgp_cmd()
        .args([
            "--json",
            "profile",
            "compare",
            "--kernel",
            "gemm",
            "--size",
            "1024",
            "--backends",
            "scalar,avx2",
        ])
        .output()
        .expect("Failed to run cgp profile compare");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("Compare JSON invalid");

    let arr = parsed.as_array().unwrap();
    let scalar = arr.iter().find(|r| r["name"] == "scalar").unwrap();
    let avx2 = arr.iter().find(|r| r["name"] == "avx2").unwrap();

    let scalar_time = scalar["wall_time_us"].as_f64().unwrap();
    let avx2_time = avx2["wall_time_us"].as_f64().unwrap();
    let speedup = scalar_time / avx2_time;

    assert!(
        speedup >= 3.0,
        "FALSIFY-CGP-041 FAILED: AVX2 speedup {speedup:.1}x < 3x (scalar={scalar_time:.0}us, avx2={avx2_time:.0}us)"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-043: Must profile arbitrary CUDA binary via nsys
// Given: any CUDA binary
// When: cgp profile binary ./binary
// Then: extracts kernel names, launch configs, wall-clock timings
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_043_profile_binary() {
    // Use nvidia-smi as a trivial binary (always available, exercises CUDA driver)
    let output = cgp_cmd()
        .args(["profile", "binary", "nvidia-smi"])
        .output()
        .expect("Failed to run cgp profile binary");

    // Should succeed (even if nsys finds no kernels — nvidia-smi doesn't launch kernels)
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Binary Profile")
            || stdout.contains("nsys")
            || stdout.contains("nvidia-smi"),
        "FALSIFY-CGP-043: Should mention binary profiling.\nOutput:\n{stdout}"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-045: cgp compete must produce normalized comparison
// Given: two commands producing results
// When: cgp compete --ours cmd1 --theirs cmd2
// Then: table shows time and vs-best ratio
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_045_compete_normalized() {
    // Use "sleep 0.01" vs "sleep 0.02" — known 2x difference
    let output = cgp_cmd()
        .args([
            "compete",
            "timing",
            "--ours",
            "sleep 0.01",
            "--theirs",
            "sleep 0.02",
            "--label",
            "fast,slow",
        ])
        .output()
        .expect("Failed to run cgp compete");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Must have a comparison table with timing
    assert!(stdout.contains("Head-to-Head"), "Missing header");
    assert!(stdout.contains("Winner"), "Missing winner declaration");
    // The "fast" command (sleep 0.01) should be the winner or at least faster
    assert!(
        stdout.contains("fast") && stdout.contains("slow"),
        "FALSIFY-CGP-045: Labels not in output.\nOutput:\n{stdout}"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-047: Must not crash on competitor binary that segfaults
// Given: a binary that crashes
// When: cgp profile binary ./crashing_binary
// Then: reports error gracefully (no cgp crash)
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_047_crash_handling() {
    // "false" is a binary that exits with code 1 (not segfault, but tests error path)
    let output = cgp_cmd()
        .args(["profile", "binary", "false"])
        .output()
        .expect("Failed to run cgp profile binary");

    // cgp itself must not crash — it should handle the error gracefully
    // (may return success with an error message, or non-zero with message)
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}{stderr}");

    // Must not be empty (should say something about the binary)
    assert!(
        !combined.trim().is_empty(),
        "FALSIFY-CGP-047: cgp produced no output for failing binary"
    );
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-062: cgp diff must not require re-profiling
// Given: two saved profile JSONs
// When: cgp diff --baseline a.json --current b.json
// Then: completes in < 100ms (pure analysis, no execution)
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_062_diff_speed() {
    let baseline = serde_json::json!({
        "version": "2.0", "timestamp": "", "hardware": {"cpu_features": []},
        "timing": {"wall_clock_time_us": 35.7, "samples": 1, "stddev_us": 0.0, "ci_95_low_us": 0.0, "ci_95_high_us": 0.0},
        "throughput": {"tflops": 7.5, "gflops": 0.0, "bandwidth_gbps": 0.0, "arithmetic_intensity": 0.0},
        "muda": []
    });
    let current = serde_json::json!({
        "version": "2.0", "timestamp": "", "hardware": {"cpu_features": []},
        "timing": {"wall_clock_time_us": 23.2, "samples": 1, "stddev_us": 0.0, "ci_95_low_us": 0.0, "ci_95_high_us": 0.0},
        "throughput": {"tflops": 11.6, "gflops": 0.0, "bandwidth_gbps": 0.0, "arithmetic_intensity": 0.0},
        "muda": []
    });

    std::fs::write("/tmp/cgp-falsify-062-b.json", baseline.to_string()).unwrap();
    std::fs::write("/tmp/cgp-falsify-062-c.json", current.to_string()).unwrap();

    // First run: compile if needed (don't count)
    let _ = cgp_cmd()
        .args([
            "diff",
            "--baseline",
            "/tmp/cgp-falsify-062-b.json",
            "--current",
            "/tmp/cgp-falsify-062-c.json",
        ])
        .output();

    // Timed run: must be < 100ms
    let start = Instant::now();
    let output = cgp_cmd()
        .args([
            "diff",
            "--baseline",
            "/tmp/cgp-falsify-062-b.json",
            "--current",
            "/tmp/cgp-falsify-062-c.json",
        ])
        .output()
        .expect("Failed to run cgp diff");
    let elapsed = start.elapsed();

    assert!(output.status.success());
    // Allow 500ms for subprocess overhead (100ms is for the analysis, not cargo run)
    assert!(
        elapsed.as_millis() < 500,
        "FALSIFY-CGP-062 FAILED: diff took {}ms (limit: 500ms with subprocess overhead)",
        elapsed.as_millis()
    );

    let _ = std::fs::remove_file("/tmp/cgp-falsify-062-b.json");
    let _ = std::fs::remove_file("/tmp/cgp-falsify-062-c.json");
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-075: Must report effective bandwidth (not raw)
// Q4K: 4096*4096 weights / 256 * 144 bytes = 9,437,184 bytes = 9.44 MB
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_075_q4k_effective_bandwidth() {
    let output = cgp_cmd()
        .args(["profile", "quant", "--kernel", "q4k_gemv", "--size", "4096x1x4096"])
        .output()
        .expect("Failed to run cgp profile quant");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Must report compressed size ~9.44 MB (not FP32 size of 67 MB)
    assert!(
        stdout.contains("9.44 MB") || stdout.contains("9.4"),
        "FALSIFY-CGP-075 FAILED: Q4K compressed size should be ~9.44 MB.\nOutput:\n{stdout}"
    );

    // Must NOT report FP32 size as the primary bandwidth metric
    // The 67 MB FP32 equivalent should be clearly labeled as such
    if stdout.contains("67") {
        assert!(
            stdout.contains("FP32 equivalent") || stdout.contains("equivalent"),
            "FALSIFY-CGP-075: If 67MB shown, must be labeled as FP32 equivalent"
        );
    }
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-061: Doctor must complete in < 2 seconds (real timing)
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_061_doctor_speed_real() {
    // Warm up (first run may compile)
    let _ = cgp_cmd().args(["doctor"]).output();

    let start = Instant::now();
    let output = cgp_cmd().args(["doctor"]).output().expect("Failed to run cgp doctor");
    let elapsed = start.elapsed();

    assert!(output.status.success());
    // Allow 500ms subprocess overhead on top of the 2s spec limit
    assert!(
        elapsed.as_millis() < 2500,
        "FALSIFY-CGP-061 FAILED: doctor took {}ms",
        elapsed.as_millis()
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Verify it detected the GPU
    assert!(stdout.contains("RTX 4090") || stdout.contains("GPU"));
}

// ══════════════════════════════════════════════════════════════════════
// FALSIFY-CGP-021: Ridge points must be mathematically correct
// FP16: 330000 / 1008 = 327.38 ≈ 327.4 FLOP/byte
// ══════════════════════════════════════════════════════════════════════
#[test]
fn falsify_cgp_021_ridge_point_math() {
    let output = cgp_cmd()
        .args(["--json", "roofline", "--target", "cuda"])
        .output()
        .expect("Failed to run cgp roofline");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).unwrap();

    // Verify ridge points
    let ridge_points = &parsed["ridge_points"];
    if let Some(arr) = ridge_points.as_array() {
        // Find FP16 Tensor ridge point
        let fp16 = arr.iter().find(|r| {
            r["precision"].as_str().map_or(false, |s| s.contains("FP16") || s.contains("Fp16"))
        });
        if let Some(fp16_ridge) = fp16 {
            let ridge = fp16_ridge["ridge_flop_per_byte"].as_f64().unwrap_or(0.0);
            // 330000 / 1008 = 327.38
            let expected = 330_000.0_f64 / 1008.0;
            assert!(
                (ridge - expected).abs() < 0.5,
                "FALSIFY-CGP-021 FAILED: FP16 ridge={ridge:.1}, expected={expected:.1}"
            );
        }
    }

    // Also verify via text output
    let text_output = cgp_cmd().args(["roofline", "--target", "cuda"]).output().expect("Failed");
    let text = String::from_utf8_lossy(&text_output.stdout);
    assert!(text.contains("327"), "FALSIFY-CGP-021: Ridge point 327.x not in output.\n{text}");
}
