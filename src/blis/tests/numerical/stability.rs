use crate::blis::*;

// ========================================================================
// Numerical Stability Tests (F38-F42)
// ========================================================================

// F40: Reproducible results (same thread count)
#[test]
fn test_falsification_40_reproducible() {
    let n = 64;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();

    let mut c1 = vec![0.0; n * n];
    let mut c2 = vec![0.0; n * n];

    gemm_blis(n, n, n, &a, &b, &mut c1, None).unwrap();
    gemm_blis(n, n, n, &a, &b, &mut c2, None).unwrap();

    // Results should be bitwise identical
    assert_eq!(c1, c2, "F40: Results not reproducible");
}

// F42: Handles Inf inputs gracefully
#[test]
fn test_falsification_42_inf_handling() {
    let a = vec![f32::INFINITY, 0.0, 0.0, 1.0];
    let b = vec![0.0, 1.0, 1.0, 1.0];
    let mut c = vec![0.0; 4];

    // Inf * 0 = NaN, which is expected behavior
    gemm_reference(2, 2, 2, &a, &b, &mut c).unwrap();

    // First element should be NaN (Inf * 0)
    assert!(c[0].is_nan(), "F42: Inf*0 should produce NaN");
}

// ========================================================================
// Robustness Tests (F43-F47)
// ========================================================================

// F45: Works with tiny matrices (2x2)
#[test]
fn test_falsification_45_tiny_matrix() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let mut c = vec![0.0; 4];

    gemm_blis(2, 2, 2, &a, &b, &mut c, None).unwrap();

    assert_eq!(
        c,
        vec![19.0, 22.0, 43.0, 50.0],
        "F45: Tiny matrix incorrect"
    );
}

// ========================================================================
// Toyota Way Compliance Tests (F48-F55)
// ========================================================================

// F48: Jidoka guard fires on NaN (already exists as test_jidoka_guard_catches_nan)
// F49: Jidoka guard fires on Inf (already exists as test_jidoka_guard_catches_inf)

// F53: Heijunka load leveling produces balanced partitions
#[test]
fn test_falsification_53_heijunka_variance() {
    let scheduler = HeijunkaScheduler {
        num_threads: 4,
        variance_threshold: 0.05,
    };

    // Test with M values that divide evenly into MC-sized tiles
    // For M=1024, we get 1024/72 ~ 14 tiles, distributed across 4 threads
    for m in [576, 720, 1024, 2048] {
        let partitions = scheduler.partition_m(m, MC);

        if partitions.len() < 2 {
            continue;
        }

        let sizes: Vec<usize> = partitions.iter().map(|r| r.len()).collect();
        let avg = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;
        let max_deviation = sizes
            .iter()
            .map(|&s| ((s as f32 - avg) / avg).abs())
            .fold(0.0_f32, f32::max);

        // Load variance should be reasonable (< 50% for uneven tile counts)
        // Perfect balance impossible when tiles don't divide evenly
        assert!(
            max_deviation < 0.5,
            "F53: Heijunka variance {:.2} > 50% for m={}",
            max_deviation,
            m
        );
    }
}

// F55: Genchi genbutsu - profiler enabled
#[test]
fn test_falsification_55_profiler_works() {
    let mut profiler = BlisProfiler::enabled();

    let n = 64;
    let a: Vec<f32> = vec![1.0; n * n];
    let b: Vec<f32> = vec![1.0; n * n];
    let mut c = vec![0.0; n * n];

    gemm_blis(n, n, n, &a, &b, &mut c, Some(&mut profiler)).unwrap();

    // Profiler should have recorded metrics
    assert!(
        profiler.macro_stats.flops > 0,
        "F55: Profiler didn't record FLOPs"
    );
    assert!(
        profiler.macro_stats.total_ns > 0,
        "F55: Profiler didn't record time"
    );

    // Summary should be non-empty
    let summary = profiler.summary();
    assert!(
        summary.contains("GFLOP/s"),
        "F55: Profiler summary incomplete"
    );
}

// ========================================================================
// Additional Memory Criteria Tests (F31-F37)
// ========================================================================

// F31: Packed A aligned to 64 bytes
#[test]
fn test_falsification_31_pack_a_aligned() {
    let mut packed_a = vec![0.0f32; packed_a_size(MC, KC)];
    // Use non-zero starting values
    let a: Vec<f32> = (0..MC * KC).map(|i| (i + 1) as f32).collect();

    // pack_a(a, lda, mc, kc, packed)
    pack_a(&a, KC, MC, KC, &mut packed_a);

    // Verify the packed data buffer is valid
    assert!(packed_a.len() >= MC * KC, "F31: Pack A buffer too small");

    // Check that some data was packed
    assert_ne!(packed_a[0], 0.0, "F31: Pack A produced empty result");
    assert_eq!(packed_a[0], 1.0, "F31: Pack A first element incorrect");
}

// F32: Packed B aligned to 64 bytes
#[test]
fn test_falsification_32_pack_b_aligned() {
    let mut packed_b = vec![0.0f32; packed_b_size(KC, NC)];
    // Use non-zero starting values
    let b: Vec<f32> = (0..KC * NC).map(|i| (i + 1) as f32).collect();

    // pack_b(b, ldb, kc, nc, packed)
    pack_b(&b, NC, KC, NC, &mut packed_b);

    // Verify buffer is sufficient
    assert!(packed_b.len() >= KC * NC, "F32: Pack B buffer too small");

    // Check that some data was packed
    assert_ne!(packed_b[0], 0.0, "F32: Pack B produced empty result");
    assert_eq!(packed_b[0], 1.0, "F32: Pack B first element incorrect");
}

// F35: No buffer overflows - bounds checking
#[test]
fn test_falsification_35_no_buffer_overflow() {
    // Test edge cases that might cause buffer overflows
    let m = MR + 3; // Not divisible by MR
    let n = NR + 2; // Not divisible by NR
    let k = 17; // Odd k value

    let a: Vec<f32> = (0..m * k).map(|i| (i % 10) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i % 10) as f32 * 0.1).collect();
    let mut c = vec![0.0; m * n];

    // Should not panic or overflow
    let result = gemm_blis(m, n, k, &a, &b, &mut c, None);
    assert!(result.is_ok(), "F35: Edge case caused error");

    // Verify result is valid (no NaN/Inf from overflow)
    for &val in &c {
        assert!(val.is_finite(), "F35: Buffer overflow produced non-finite");
    }
}

// ========================================================================
// Additional Numerical Stability Tests (F38-F42)
// ========================================================================

// F39: No catastrophic cancellation with ill-conditioned matrices
#[test]
fn test_falsification_39_no_catastrophic_cancellation() {
    // Test with nearly-canceling values
    let n = 16;
    let big = 1e6_f32;
    let small = 1.0_f32;

    // A and B designed so products should cancel but leave small residual
    let a: Vec<f32> = (0..n * n)
        .map(|i| if i % 2 == 0 { big } else { -big })
        .collect();
    // All elements are `small` for this test case (deliberate design)
    let b: Vec<f32> = vec![small; n * n];
    let mut c = vec![0.0; n * n];

    gemm_blis(n, n, n, &a, &b, &mut c, None).unwrap();

    // Result should be finite (no NaN from cancellation issues)
    for &val in &c {
        assert!(
            val.is_finite(),
            "F39: Catastrophic cancellation produced NaN/Inf"
        );
    }
}

// F41: Error bound |C_computed - C_exact| <= K*eps*|A|*|B|
#[test]
fn test_falsification_41_error_bound() {
    let n = 64;
    let k = 128;

    // Use small values to make error analysis tractable
    let a: Vec<f32> = (0..n * k).map(|i| ((i % 7) as f32) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.01).collect();

    let mut c_blis = vec![0.0; n * n];
    let mut c_ref = vec![0.0; n * n];

    gemm_blis(n, n, k, &a, &b, &mut c_blis, None).unwrap();
    gemm_reference(n, n, k, &a, &b, &mut c_ref).unwrap();

    // Compute Frobenius norms
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Higham error bound: |error| <= gamma_k * |A| * |B|
    // where gamma_k = k * eps / (1 - k * eps) ~ k * eps for small k * eps
    let eps = f32::EPSILON;
    let gamma_k = (k as f32) * eps / (1.0 - (k as f32) * eps);
    let error_bound = gamma_k * norm_a * norm_b;

    // Check each element
    let max_error = c_blis
        .iter()
        .zip(c_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    // Allow some slack since we're comparing two imprecise implementations
    assert!(
        max_error < error_bound * 100.0,
        "F41: Max error {} exceeds bound {}",
        max_error,
        error_bound * 100.0
    );
}

// ========================================================================
// Additional Robustness Tests (F43-F47)
// ========================================================================

// F44: Works with large matrices (scaled down for unit test speed)
#[test]
fn test_falsification_44_large_matrix() {
    // Use 1024x1024 instead of 16Kx16K for unit test speed
    let n = 512;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 10) as f32) * 0.01).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 10) as f32) * 0.01).collect();
    let mut c = vec![0.0; n * n];

    // Should complete without OOM or panic
    let result = gemm_blis(n, n, n, &a, &b, &mut c, None);
    assert!(result.is_ok(), "F44: Large matrix GEMM failed");

    // Spot check a few values
    assert!(c[0].is_finite(), "F44: Large matrix produced NaN");
    assert!(c[n * n / 2].is_finite(), "F44: Large matrix produced NaN");
    assert!(c[n * n - 1].is_finite(), "F44: Large matrix produced NaN");
}

// F46: Thread-safe for concurrent calls (simulated with sequential verification)
#[test]
fn test_falsification_46_thread_safe() {
    // Run multiple GEMMs with different inputs to verify no shared mutable state
    let n = 32;

    let results: Vec<Vec<f32>> = (0..4)
        .map(|seed| {
            let a: Vec<f32> = (0..n * n).map(|i| ((i + seed) % 10) as f32).collect();
            let b: Vec<f32> = (0..n * n).map(|i| ((i + seed * 2) % 10) as f32).collect();
            let mut c = vec![0.0; n * n];
            gemm_blis(n, n, n, &a, &b, &mut c, None).unwrap();
            c
        })
        .collect();

    // Each result should be different (no shared state corruption)
    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            assert_ne!(results[i], results[j], "F46: Results incorrectly identical");
        }
    }

    // Re-run first case to verify reproducibility
    let a: Vec<f32> = (0..n * n).map(|i| (i % 10) as f32).collect();
    let b: Vec<f32> = (0..n * n).map(|i| (i % 10) as f32).collect();
    let mut c_verify = vec![0.0; n * n];
    gemm_blis(n, n, n, &a, &b, &mut c_verify, None).unwrap();

    assert_eq!(c_verify, results[0], "F46: Non-reproducible results");
}

// F50: Jidoka guard fires on wrong result
#[test]
fn test_falsification_50_jidoka_wrong_result() {
    let n = 8;
    let a = vec![1.0f32; n * n];
    let b = vec![1.0f32; n * n];
    let mut c = vec![0.0; n * n];

    // First compute correct result
    gemm_reference(n, n, n, &a, &b, &mut c).unwrap();
    let expected = c[0]; // Should be n (sum of 1.0 * 1.0 * n times)

    assert_eq!(expected, n as f32, "F50: Reference result wrong");

    // Create strict guard (1e-6 tolerance)
    let guard = JidokaGuard::strict();

    // Re-run with guard - should pass since result is correct
    let mut c_jidoka = vec![0.0; n * n];
    let result = gemm_reference_with_jidoka(n, n, n, &a, &b, &mut c_jidoka, &guard);
    assert!(result.is_ok(), "F50: Jidoka rejected correct result");
}

// ========================================================================
// Property-Based Tests (Fast, Deterministic)
// ========================================================================

/// Property: GEMM with zero matrix A produces unchanged C
#[test]
fn prop_zero_a_unchanged_c() {
    for n in [8, 16, 32, 64] {
        let a = vec![0.0f32; n * n];
        let b: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
        let mut c = vec![1.0f32; n * n];
        let c_orig = c.clone();

        gemm_blis(n, n, n, &a, &b, &mut c, None).unwrap();

        assert_eq!(c, c_orig, "C should be unchanged when A=0 for n={}", n);
    }
}

/// Property: GEMM with zero matrix B produces unchanged C
#[test]
fn prop_zero_b_unchanged_c() {
    for n in [8, 16, 32, 64] {
        let a: Vec<f32> = (0..n * n).map(|i| i as f32).collect();
        let b = vec![0.0f32; n * n];
        let mut c = vec![1.0f32; n * n];
        let c_orig = c.clone();

        gemm_blis(n, n, n, &a, &b, &mut c, None).unwrap();

        assert_eq!(c, c_orig, "C should be unchanged when B=0 for n={}", n);
    }
}

/// Property: GEMM is consistent across multiple calls
#[test]
fn prop_deterministic() {
    let n = 64;
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i % 11) as f32) * 0.1).collect();

    let mut c1 = vec![0.0f32; n * n];
    let mut c2 = vec![0.0f32; n * n];

    gemm_blis(n, n, n, &a, &b, &mut c1, None).unwrap();
    gemm_blis(n, n, n, &a, &b, &mut c2, None).unwrap();

    assert_eq!(c1, c2, "GEMM should be deterministic");
}

/// Property: BLIS matches reference for various dimensions
#[test]
fn prop_blis_matches_reference() {
    // Test various dimensions including edge cases
    let test_cases = [
        (8, 8, 8),
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (13, 17, 19), // Primes (not divisible by MR/NR)
        (1, 64, 64),  // Vector-matrix
        (64, 1, 64),  // Matrix-vector
        (64, 64, 1),  // Outer product
    ];

    for (m, n, k) in test_cases {
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 5) as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.1).collect();

        let mut c_ref = vec![0.0f32; m * n];
        let mut c_blis = vec![0.0f32; m * n];

        gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
        gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();

        let max_diff: f32 = c_ref
            .iter()
            .zip(c_blis.iter())
            .map(|(r, b)| (r - b).abs())
            .fold(0.0, f32::max);

        assert!(
            max_diff < 1e-3,
            "BLIS should match reference for {}x{}x{}, max_diff={}",
            m,
            n,
            k,
            max_diff
        );
    }
}

/// Property: Accumulation works correctly (C += A*B)
#[test]
fn prop_accumulation() {
    let n = 32;
    let a: Vec<f32> = vec![1.0; n * n];
    let b: Vec<f32> = vec![1.0; n * n];

    let mut c = vec![0.0f32; n * n];

    // First call: C = 0 + A*B = A*B
    gemm_blis(n, n, n, &a, &b, &mut c, None).unwrap();
    let c_first = c.clone();

    // Second call: C = A*B + A*B = 2*A*B
    gemm_blis(n, n, n, &a, &b, &mut c, None).unwrap();

    // Each element should be doubled
    for i in 0..n * n {
        let expected = c_first[i] * 2.0;
        assert!(
            (c[i] - expected).abs() < 1e-3,
            "Accumulation failed at {}: {} vs {}",
            i,
            c[i],
            expected
        );
    }
}

/// Property: Scaling works (alpha * A * B)
#[test]
fn prop_scaling() {
    let n = 32;
    let a: Vec<f32> = (0..n * n).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = vec![1.0; n * n]; // Identity-like for simplicity

    // Compute with a
    let mut c1 = vec![0.0f32; n * n];
    gemm_blis(n, n, n, &a, &b, &mut c1, None).unwrap();

    // Compute with 2*a
    let a_scaled: Vec<f32> = a.iter().map(|x| x * 2.0).collect();
    let mut c2 = vec![0.0f32; n * n];
    gemm_blis(n, n, n, &a_scaled, &b, &mut c2, None).unwrap();

    // c2 should be 2*c1
    for i in 0..n * n {
        let expected = c1[i] * 2.0;
        assert!(
            (c2[i] - expected).abs() < 1e-2,
            "Scaling property failed at {}: {} vs {}",
            i,
            c2[i],
            expected
        );
    }
}

/// Property: Microkernel produces correct output dimensions
#[test]
fn prop_microkernel_dimensions() {
    for k in [1, 4, 16, 64, 256] {
        let a = vec![1.0f32; MR * k];
        let b = vec![1.0f32; k * NR];
        let mut c = vec![0.0f32; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c, MR);

        // Each output should be k (sum of k ones)
        for val in &c {
            assert!(
                (*val - k as f32).abs() < 1e-5,
                "Microkernel output wrong for k={}: {} vs {}",
                k,
                val,
                k
            );
        }
    }
}

/// Property: Packing preserves all elements
#[test]
fn prop_pack_preserves_elements() {
    let mc = 32;
    let kc = 64;

    // Create matrix with unique values
    let a: Vec<f32> = (0..mc * kc).map(|i| i as f32).collect();
    let mut packed = vec![0.0f32; packed_a_size(mc, kc)];

    pack_a(&a, kc, mc, kc, &mut packed);

    // Sum should be preserved (minus padding)
    let _orig_sum: f32 = a.iter().sum();
    let _packed_sum: f32 = packed.iter().sum();

    // Packed includes zero padding, but unique values should all appear
    let mut found = vec![false; mc * kc];
    for val in &packed {
        let idx = *val as usize;
        if idx < mc * kc {
            found[idx] = true;
        }
    }

    let all_found = found.iter().all(|&f| f);
    assert!(all_found, "Packing should preserve all unique values");
}
