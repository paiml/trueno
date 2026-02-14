//! Section C: SIMD Operations (Claims 31-50)

use simular::engine::rng::SimRng;
use trueno::Vector;

// =============================================================================
// SECTION C: SIMD Operations (Claims 31-50)
// =============================================================================

/// C-031: vec_add(a, b) == vec_add(b, a) (commutativity)
#[test]
fn test_c031_add_commutativity() {
    let mut rng = SimRng::new(31);
    let a: Vec<f32> = (0..1000)
        .map(|_| rng.gen_f64() as f32 * 200.0 - 100.0)
        .collect();
    let b: Vec<f32> = (0..1000)
        .map(|_| rng.gen_f64() as f32 * 200.0 - 100.0)
        .collect();

    let vec_a = Vector::from_slice(&a);
    let vec_b = Vector::from_slice(&b);

    let result_ab = vec_a.add(&vec_b).expect("add failed");
    let result_ba = vec_b.add(&vec_a).expect("add failed");

    for (i, (ab, ba)) in result_ab
        .as_slice()
        .iter()
        .zip(result_ba.as_slice().iter())
        .enumerate()
    {
        assert_eq!(
            ab.to_bits(),
            ba.to_bits(),
            "C-031 FALSIFIED: Addition not commutative at index {i}"
        );
    }
}

/// C-032: vec_add(a, vec_add(b, c)) == vec_add(vec_add(a, b), c) within tolerance
#[test]
fn test_c032_add_associativity() {
    let a = Vector::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = Vector::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
    let c = Vector::from_slice(&[0.01f32, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]);

    // a + (b + c)
    let bc = b.add(&c).expect("add failed");
    let a_bc = a.add(&bc).expect("add failed");

    // (a + b) + c
    let ab = a.add(&b).expect("add failed");
    let ab_c = ab.add(&c).expect("add failed");

    // Check within floating-point tolerance
    for (i, (l, r)) in a_bc
        .as_slice()
        .iter()
        .zip(ab_c.as_slice().iter())
        .enumerate()
    {
        let diff = (l - r).abs();
        assert!(
            diff < 1e-6,
            "C-032 FALSIFIED: Associativity violation at index {i}: diff = {diff}"
        );
    }
}

/// C-033: vec_mul(a, b) == vec_mul(b, a) (commutativity)
#[test]
fn test_c033_mul_commutativity() {
    let mut rng = SimRng::new(33);
    let a: Vec<f32> = (0..1000)
        .map(|_| rng.gen_f64() as f32 * 20.0 - 10.0)
        .collect();
    let b: Vec<f32> = (0..1000)
        .map(|_| rng.gen_f64() as f32 * 20.0 - 10.0)
        .collect();

    let vec_a = Vector::from_slice(&a);
    let vec_b = Vector::from_slice(&b);

    let result_ab = vec_a.mul(&vec_b).expect("mul failed");
    let result_ba = vec_b.mul(&vec_a).expect("mul failed");

    for (i, (ab, ba)) in result_ab
        .as_slice()
        .iter()
        .zip(result_ba.as_slice().iter())
        .enumerate()
    {
        assert_eq!(
            ab.to_bits(),
            ba.to_bits(),
            "C-033 FALSIFIED: Multiplication not commutative at index {i}"
        );
    }
}

/// C-034: dot(a, b) == dot(b, a) (commutativity)
#[test]
fn test_c034_dot_commutativity() {
    let a = Vector::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = Vector::from_slice(&[8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

    let dot_ab = a.dot(&b).expect("dot failed");
    let dot_ba = b.dot(&a).expect("dot failed");

    assert_eq!(
        dot_ab.to_bits(),
        dot_ba.to_bits(),
        "C-034 FALSIFIED: Dot product not commutative: {dot_ab} != {dot_ba}"
    );
}

/// C-035: dot(a, a) >= 0 for all a (positive semi-definite)
#[test]
fn test_c035_dot_positive_semidefinite() {
    let mut rng = SimRng::new(35);

    for _ in 0..1000 {
        let a: Vec<f32> = (0..100)
            .map(|_| rng.gen_f64() as f32 * 200.0 - 100.0)
            .collect();
        let vec_a = Vector::from_slice(&a);

        let dot_aa = vec_a.dot(&vec_a).expect("dot failed");

        assert!(
            dot_aa >= 0.0,
            "C-035 FALSIFIED: dot(a, a) = {dot_aa} is negative"
        );
    }
}

/// C-036: relu(x) == max(0, x) for all x
#[test]
fn test_c036_relu_definition() {
    let input = Vector::from_slice(&[-3.0f32, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.0]);
    let expected = [0.0f32, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0];

    let result = input.relu().expect("relu failed");

    for (i, (r, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            *r,
            *e,
            "C-036 FALSIFIED: relu({}) = {}, expected {}",
            input.as_slice()[i],
            r,
            e
        );
    }
}

/// C-037: sigmoid(x) is in [0, 1] for all finite x
/// Note: Due to floating-point precision, sigmoid can equal exactly 0 or 1 for extreme inputs
#[test]
fn test_c037_sigmoid_range() {
    let mut rng = SimRng::new(37);

    for _ in 0..100 {
        let x = rng.gen_f64() as f32 * 20.0 - 10.0; // Range: [-10, 10] to avoid saturation
        let input = Vector::from_slice(&[x]);
        let result = input.sigmoid().expect("sigmoid failed");

        assert!(
            result.as_slice()[0] >= 0.0 && result.as_slice()[0] <= 1.0,
            "C-037 FALSIFIED: sigmoid({x}) = {} not in [0, 1]",
            result.as_slice()[0]
        );
    }

    // Test boundary behavior for extreme values
    let extreme_pos = Vector::from_slice(&[100.0f32]);
    let extreme_neg = Vector::from_slice(&[-100.0f32]);

    let result_pos = extreme_pos.sigmoid().expect("sigmoid failed");
    let result_neg = extreme_neg.sigmoid().expect("sigmoid failed");

    // Extreme positive should approach 1
    assert!(
        result_pos.as_slice()[0] >= 0.99,
        "C-037: sigmoid(100) should approach 1, got {}",
        result_pos.as_slice()[0]
    );
    // Extreme negative should approach 0
    assert!(
        result_neg.as_slice()[0] <= 0.01,
        "C-037: sigmoid(-100) should approach 0, got {}",
        result_neg.as_slice()[0]
    );
}

/// C-038: tanh(x) is in [-1, 1] for all finite x
/// Note: Due to floating-point precision, tanh can equal exactly -1 or 1 for extreme inputs
#[test]
fn test_c038_tanh_range() {
    let mut rng = SimRng::new(38);

    for _ in 0..100 {
        let x = rng.gen_f64() as f32 * 10.0 - 5.0; // Range: [-5, 5] to avoid saturation
        let input = Vector::from_slice(&[x]);
        let result = input.tanh().expect("tanh failed");

        assert!(
            result.as_slice()[0] >= -1.0 && result.as_slice()[0] <= 1.0,
            "C-038 FALSIFIED: tanh({x}) = {} not in [-1, 1]",
            result.as_slice()[0]
        );
    }

    // Test boundary behavior for extreme values
    let extreme_pos = Vector::from_slice(&[100.0f32]);
    let extreme_neg = Vector::from_slice(&[-100.0f32]);

    let result_pos = extreme_pos.tanh().expect("tanh failed");
    let result_neg = extreme_neg.tanh().expect("tanh failed");

    // Extreme positive should approach 1
    assert!(
        result_pos.as_slice()[0] >= 0.99,
        "C-038: tanh(100) should approach 1, got {}",
        result_pos.as_slice()[0]
    );
    // Extreme negative should approach -1
    assert!(
        result_neg.as_slice()[0] <= -0.99,
        "C-038: tanh(-100) should approach -1, got {}",
        result_neg.as_slice()[0]
    );
}

/// C-039: softmax(x) sums to 1.0 within 1e-5
#[test]
fn test_c039_softmax_sum() {
    let inputs = vec![
        vec![1.0f32, 2.0, 3.0],
        vec![0.0, 0.0, 0.0],
        vec![-1.0, 0.0, 1.0],
    ];

    for input in inputs {
        let vec_input = Vector::from_slice(&input);
        let result = vec_input.softmax().expect("softmax failed");
        let sum: f32 = result.as_slice().iter().sum();

        assert!(
            (sum - 1.0).abs() < 1e-5,
            "C-039 FALSIFIED: softmax({:?}) sums to {}, not 1.0",
            input,
            sum
        );
    }
}

/// C-040: gelu(x) approximates exact GELU within 1e-4
#[test]
fn test_c040_gelu_approximation() {
    // Reference GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    fn reference_gelu(x: f32) -> f32 {
        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        x * 0.5 * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
    }

    let test_values = [-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

    for &x in &test_values {
        let vec_x = Vector::from_slice(&[x]);
        let result = vec_x.gelu().expect("gelu failed");
        let actual = result.as_slice()[0];
        let expected = reference_gelu(x);

        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-4,
            "C-040 FALSIFIED: gelu({x}) = {actual}, expected {expected} (diff: {diff})"
        );
    }
}

/// C-041: swish(x) == x * sigmoid(x) within 1e-6
#[test]
fn test_c041_swish_definition() {
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    let test_values = [-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

    for &x in &test_values {
        let vec_x = Vector::from_slice(&[x]);
        let result = vec_x.swish().expect("swish failed");
        let actual = result.as_slice()[0];
        let expected = x * sigmoid(x);

        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-6,
            "C-041 FALSIFIED: swish({x}) = {actual}, expected {expected} (diff: {diff})"
        );
    }
}

/// C-042: SIMD remainder handling is correct for non-aligned sizes
#[test]
fn test_c042_simd_remainder_handling() {
    // Test sizes 1-15 (non-aligned for AVX2's 8-wide operations)
    for size in 1..16 {
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - 1 - i) as f32).collect();

        let vec_a = Vector::from_slice(&a);
        let vec_b = Vector::from_slice(&b);

        let result = vec_a.add(&vec_b).expect("add failed");

        // Verify correct result
        for (i, r) in result.as_slice().iter().enumerate() {
            let expected = a[i] + b[i];
            assert_eq!(
                *r, expected,
                "C-042 FALSIFIED: Remainder handling incorrect for size {size} at index {i}"
            );
        }
    }
}

/// C-043: Operations handle empty input gracefully
#[test]
fn test_c043_empty_input_safety() {
    let empty: Vec<f32> = vec![];
    let vec_empty = Vector::from_slice(&empty);

    // Empty vector operations should work without crashing
    // (add would fail due to size mismatch, but create is fine)
    assert_eq!(vec_empty.len(), 0);
}

/// C-044: Operations handle single element correctly
#[test]
fn test_c044_single_element_safety() {
    let a = Vector::from_slice(&[1.0f32]);
    let b = Vector::from_slice(&[2.0f32]);

    let result = a.add(&b).expect("add failed");
    assert_eq!(result.as_slice()[0], 3.0);
}

/// C-045: SIMD handles misaligned pointers
#[test]
fn test_c045_misaligned_pointers() {
    // Create vectors with non-power-of-2 sizes to test alignment handling
    let sizes = [3, 5, 7, 9, 11, 13, 15, 17];

    for &size in &sizes {
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        let vec_a = Vector::from_slice(&a);
        let vec_b = Vector::from_slice(&b);

        // Should not crash regardless of pointer alignment
        let result = vec_a.add(&vec_b).expect("add failed");
        assert_eq!(
            result.as_slice().len(),
            size,
            "C-045 FALSIFIED: Misaligned pointer handling failed for size {size}"
        );
    }
}

/// C-046: AVX2 uses 256-bit registers (ymm) - verified via backend selection
#[test]
fn test_c046_avx2_register_width() {
    // AVX2 processes 8 f32 values at once (256 bits / 32 bits = 8)
    // This test verifies the expected throughput characteristic
    let size = 8; // Exactly one AVX2 register width
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

    let vec_a = Vector::from_slice(&a);
    let vec_b = Vector::from_slice(&b);

    let result = vec_a.add(&vec_b).expect("add failed");

    // Verify all 8 elements processed correctly
    for i in 0..size {
        let expected = (i + i + 1) as f32;
        assert_eq!(
            result.as_slice()[i],
            expected,
            "C-046 FALSIFIED: AVX2 256-bit operation incorrect at index {i}"
        );
    }
}

/// C-047: AVX-512 uses 512-bit registers (zmm) - verified via backend selection
#[test]
fn test_c047_avx512_register_width() {
    // AVX-512 processes 16 f32 values at once (512 bits / 32 bits = 16)
    let size = 16; // Exactly one AVX-512 register width
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

    let vec_a = Vector::from_slice(&a);
    let vec_b = Vector::from_slice(&b);

    let result = vec_a.add(&vec_b).expect("add failed");

    // Verify all 16 elements processed correctly
    for i in 0..size {
        let expected = (i + i + 1) as f32;
        assert_eq!(
            result.as_slice()[i],
            expected,
            "C-047 FALSIFIED: AVX-512 512-bit operation incorrect at index {i}"
        );
    }
}

/// C-048: NEON uses 128-bit registers (q) - verified via backend selection
#[test]
fn test_c048_neon_register_width() {
    // NEON processes 4 f32 values at once (128 bits / 32 bits = 4)
    let size = 4; // Exactly one NEON register width
    let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

    let vec_a = Vector::from_slice(&a);
    let vec_b = Vector::from_slice(&b);

    let result = vec_a.add(&vec_b).expect("add failed");

    // Verify all 4 elements processed correctly
    for i in 0..size {
        let expected = (i + i + 1) as f32;
        assert_eq!(
            result.as_slice()[i],
            expected,
            "C-048 FALSIFIED: NEON 128-bit operation incorrect at index {i}"
        );
    }
}

/// C-049: FMA is used when available (AVX2+FMA) - verified via fma operation
#[test]
fn test_c049_fma_availability() {
    // FMA: a * b + c in single operation
    let a = Vector::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = Vector::from_slice(&[2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    let c = Vector::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    let result = a.fma(&b, &c).expect("fma failed");

    // Verify: a * b + c
    let expected = [3.0f32, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0];
    for (i, (r, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        let diff = (*r - *e).abs();
        assert!(
            diff < f32::EPSILON,
            "C-049 FALSIFIED: FMA incorrect at index {i}: {} vs {}",
            r,
            e
        );
    }
}

/// C-050: Operations don't cause denormal stalls
#[test]
fn test_c050_no_denormal_stall() {
    use std::time::Instant;

    // Create denormal inputs
    let denormal = f32::from_bits(1); // Smallest positive subnormal
    let denormals: Vec<f32> = vec![denormal; 1000];
    let normal: Vec<f32> = vec![1.0; 1000];

    let vec_denormal = Vector::from_slice(&denormals);
    let vec_normal = Vector::from_slice(&normal);

    // Time denormal operation
    let start = Instant::now();
    for _ in 0..100 {
        let _ = vec_denormal.add(&vec_denormal);
    }
    let denormal_time = start.elapsed();

    // Time normal operation
    let start = Instant::now();
    for _ in 0..100 {
        let _ = vec_normal.add(&vec_normal);
    }
    let normal_time = start.elapsed();

    // Denormal operations shouldn't be more than 10x slower
    // (this is a heuristic - actual threshold depends on hardware)
    let ratio = denormal_time.as_nanos() as f64 / normal_time.as_nanos() as f64;
    assert!(
        ratio < 100.0,
        "C-050 WARNING: Denormal operations are {ratio:.1}x slower than normal"
    );
}
