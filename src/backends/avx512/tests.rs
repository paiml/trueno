use super::*;
use crate::backends::scalar::ScalarBackend;

fn avx512_test<F>(test_fn: F)
where
    F: FnOnce(),
{
    if is_x86_feature_detected!("avx512f") {
        test_fn();
    } else {
        println!("Skipping AVX-512 test (CPU does not support avx512f)");
    }
}

/// Helper: test a binary element-wise op (`a OP b => expected` for all elements).
fn assert_binary_op(
    a_val: f32,
    b_val: f32,
    expected: f32,
    op: unsafe fn(&[f32], &[f32], &mut [f32]),
) {
    let a = vec![a_val; 32];
    let b = vec![b_val; 32];
    let mut result = vec![0.0; 32];
    // SAFETY: test-only; vectors are identically sized, backend selected by caller
    unsafe { op(&a, &b, &mut result) };
    assert!(
        result.iter().all(|&x| (x - expected).abs() < 1e-6),
        "expected all {expected}, got {:?}",
        &result[..4]
    );
}

/// Helper: test a unary transform against per-element expected values.
fn assert_unary_transform(
    input: &[f32],
    expected: &[f32],
    tol: f32,
    op: unsafe fn(&[f32], &mut [f32]),
) {
    let mut result = vec![0.0; input.len()];
    // SAFETY: test-only; result matches input length
    unsafe { op(input, &mut result) };
    for (i, (&val, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (val - exp).abs() < tol,
            "mismatch at {i}: got {val}, expected {exp}"
        );
    }
}

/// Helper: test a unary transform on a large array against a scalar reference function.
/// `input_gen` produces the input, `reference_fn` computes the expected scalar result.
fn assert_unary_large(
    input: Vec<f32>,
    tol: f32,
    op: unsafe fn(&[f32], &mut [f32]),
    reference_fn: fn(f32) -> f32,
    label: &str,
) {
    let mut result = vec![0.0; input.len()];
    // SAFETY: test-only; result matches input length
    unsafe { op(&input, &mut result) };
    for (i, &val) in result.iter().enumerate() {
        let expected = reference_fn(input[i]);
        assert!(
            (val - expected).abs() < tol,
            "{label} large mismatch at {i}: {val} vs {expected}"
        );
    }
}

/// Helper: test a unary transform on a large array using relative error tolerance.
fn assert_unary_large_relative(
    input: Vec<f32>,
    rel_tol: f32,
    op: unsafe fn(&[f32], &mut [f32]),
    reference_fn: fn(f32) -> f32,
    label: &str,
) {
    let mut result = vec![0.0; input.len()];
    // SAFETY: test-only; result matches input length
    unsafe { op(&input, &mut result) };
    for (i, &val) in result.iter().enumerate() {
        let expected = reference_fn(input[i]);
        assert!(
            (val - expected).abs() / expected.max(1e-6) < rel_tol,
            "{label} mismatch at {i}: {val} vs {expected}"
        );
    }
}

/// Helper: test a scalar reduction on sequential 1..=32 input.
fn assert_reduction_f32(expected: f32, tol: f32, op: unsafe fn(&[f32]) -> f32) {
    let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
    let result = unsafe { op(&a) };
    assert!(
        (result - expected).abs() < tol,
        "expected {expected}, got {result}"
    );
}

/// Helper: test an index-returning reduction on sequential 1..=32 input.
fn assert_reduction_usize(expected: usize, op: unsafe fn(&[f32]) -> usize) {
    let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
    let result = unsafe { op(&a) };
    assert_eq!(result, expected);
}

/// Helper: test a large activation function where f(0) == 0 at the midpoint.
fn assert_activation_zero_at_origin(op: unsafe fn(&[f32], &mut [f32]), label: &str) {
    let a: Vec<f32> = (-16..16).map(|i| i as f32 * 0.3).collect();
    let mut result = vec![0.0; 32];
    // SAFETY: test-only; result matches input length
    unsafe { op(&a, &mut result) };
    assert!(
        (result[16]).abs() < 1e-4,
        "{label}(0) should be 0, got {}",
        result[16]
    );
}

// ========== Binary Element-Wise Operations ==========

#[test]
fn test_avx512_add() {
    avx512_test(|| assert_binary_op(1.0, 2.0, 3.0, Avx512Backend::add));
}

#[test]
fn test_avx512_sub() {
    avx512_test(|| assert_binary_op(5.0, 2.0, 3.0, Avx512Backend::sub));
}

#[test]
fn test_avx512_mul() {
    avx512_test(|| assert_binary_op(2.0, 3.0, 6.0, Avx512Backend::mul));
}

#[test]
fn test_avx512_div() {
    avx512_test(|| assert_binary_op(6.0, 2.0, 3.0, Avx512Backend::div));
}

// ========== Scalar Reductions ==========

#[test]
fn test_avx512_dot() {
    avx512_test(|| {
        let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let result = unsafe { Avx512Backend::dot(&a, &b) };
        let expected: f32 = (1..=32).map(|i| (i * i) as f32).sum();
        assert!((result - expected).abs() < 1e-3);
    });
}

#[test]
fn test_avx512_sum() {
    avx512_test(|| assert_reduction_f32(528.0, 1e-3, Avx512Backend::sum));
}

#[test]
fn test_avx512_max() {
    avx512_test(|| assert_reduction_f32(32.0, 1e-6, Avx512Backend::max));
}

#[test]
fn test_avx512_min() {
    avx512_test(|| assert_reduction_f32(1.0, 1e-6, Avx512Backend::min));
}

#[test]
fn test_avx512_argmax() {
    avx512_test(|| assert_reduction_usize(31, Avx512Backend::argmax));
}

#[test]
fn test_avx512_argmin() {
    avx512_test(|| assert_reduction_usize(0, Avx512Backend::argmin));
}

#[test]
fn test_avx512_sum_kahan() {
    avx512_test(|| assert_reduction_f32(528.0, 1e-3, Avx512Backend::sum_kahan));
}

// ========== Norms ==========

#[test]
fn test_avx512_norm_l2() {
    avx512_test(|| {
        let a = vec![3.0, 4.0];
        let result = unsafe { Avx512Backend::norm_l2(&a) };
        assert!((result - 5.0).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_norm_l1() {
    avx512_test(|| {
        let a = vec![-1.0, 2.0, -3.0, 4.0];
        let result = unsafe { Avx512Backend::norm_l1(&a) };
        assert!((result - 10.0).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_norm_linf() {
    avx512_test(|| {
        let a = vec![-5.0, 2.0, -3.0, 4.0];
        let result = unsafe { Avx512Backend::norm_linf(&a) };
        assert!((result - 5.0).abs() < 1e-5);
    });
}

// ========== Unary Transforms (small) ==========

#[test]
fn test_avx512_scale() {
    avx512_test(|| {
        let a = vec![1.0; 32];
        let mut result = vec![0.0; 32];
        unsafe {
            Avx512Backend::scale(&a, 3.0, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    });
}

#[test]
fn test_avx512_abs() {
    avx512_test(|| {
        assert_unary_transform(
            &[-1.0, 2.0, -3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
            1e-6,
            Avx512Backend::abs,
        );
    });
}

#[test]
fn test_avx512_clamp() {
    avx512_test(|| {
        let a = vec![0.0, 5.0, 10.0, 15.0];
        let mut result = vec![0.0; 4];
        unsafe {
            Avx512Backend::clamp(&a, 2.0, 12.0, &mut result);
        }
        assert_eq!(result, vec![2.0, 5.0, 10.0, 12.0]);
    });
}

#[test]
fn test_avx512_lerp() {
    avx512_test(|| {
        let a = vec![0.0; 32];
        let b = vec![10.0; 32];
        let mut result = vec![0.0; 32];
        unsafe {
            Avx512Backend::lerp(&a, &b, 0.5, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 5.0).abs() < 1e-5));
    });
}

#[test]
fn test_avx512_fma() {
    avx512_test(|| {
        let a = vec![2.0; 32];
        let b = vec![3.0; 32];
        let c = vec![1.0; 32];
        let mut result = vec![0.0; 32];
        unsafe {
            Avx512Backend::fma(&a, &b, &c, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 7.0).abs() < 1e-5));
    });
}

#[test]
fn test_avx512_relu() {
    avx512_test(|| {
        assert_unary_transform(
            &[-1.0, 0.0, 1.0, 2.0],
            &[0.0, 0.0, 1.0, 2.0],
            1e-6,
            Avx512Backend::relu,
        );
    });
}

#[test]
fn test_avx512_exp() {
    avx512_test(|| {
        assert_unary_transform(
            &[0.0, 1.0],
            &[1.0, std::f32::consts::E],
            1e-3,
            Avx512Backend::exp,
        );
    });
}

#[test]
fn test_avx512_sigmoid() {
    avx512_test(|| {
        assert_unary_transform(&[0.0], &[0.5], 1e-5, Avx512Backend::sigmoid);
    });
}

#[test]
fn test_avx512_gelu() {
    avx512_test(|| {
        assert_unary_transform(
            &[0.0, 1.0],
            &[0.0, 0.841_192],
            1e-3,
            Avx512Backend::gelu,
        );
    });
}

#[test]
fn test_avx512_swish() {
    avx512_test(|| {
        assert_unary_transform(
            &[0.0, 1.0],
            &[0.0, 0.731_059],
            1e-3,
            Avx512Backend::swish,
        );
    });
}

#[test]
fn test_avx512_tanh() {
    avx512_test(|| {
        assert_unary_transform(
            &[0.0, 1.0],
            &[0.0, 0.761_594_2],
            1e-3,
            Avx512Backend::tanh,
        );
    });
}

#[test]
fn test_avx512_sqrt() {
    avx512_test(|| {
        assert_unary_transform(
            &[4.0, 9.0, 16.0],
            &[2.0, 3.0, 4.0],
            1e-5,
            Avx512Backend::sqrt,
        );
    });
}

#[test]
fn test_avx512_recip() {
    avx512_test(|| {
        assert_unary_transform(
            &[2.0, 4.0, 5.0],
            &[0.5, 0.25, 0.2],
            1e-5,
            Avx512Backend::recip,
        );
    });
}

#[test]
fn test_avx512_transcendental() {
    avx512_test(|| {
        let a = vec![1.0, std::f32::consts::E, 10.0];
        let mut ln_result = vec![0.0; 3];
        let mut log2_result = vec![0.0; 3];
        let mut log10_result = vec![0.0; 3];
        unsafe {
            Avx512Backend::ln(&a, &mut ln_result);
            Avx512Backend::log2(&a, &mut log2_result);
            Avx512Backend::log10(&a, &mut log10_result);
        }
        assert!((ln_result[0]).abs() < 1e-5);
        assert!((ln_result[1] - 1.0).abs() < 1e-4);
        assert!((log10_result[2] - 1.0).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_trig() {
    avx512_test(|| {
        let a = vec![0.0, std::f32::consts::FRAC_PI_2];
        let mut sin_result = vec![0.0; 2];
        let mut cos_result = vec![0.0; 2];
        let mut tan_result = vec![0.0; 2];
        unsafe {
            Avx512Backend::sin(&a, &mut sin_result);
            Avx512Backend::cos(&a, &mut cos_result);
            Avx512Backend::tan(&a, &mut tan_result);
        }
        assert!((sin_result[0]).abs() < 1e-5);
        assert!((sin_result[1] - 1.0).abs() < 1e-5);
        assert!((cos_result[0] - 1.0).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_rounding() {
    avx512_test(|| {
        let a = vec![1.3, 1.5, 1.7, -1.3, -1.5, -1.7];
        let mut floor_result = vec![0.0; 6];
        let mut ceil_result = vec![0.0; 6];
        let mut round_result = vec![0.0; 6];
        unsafe {
            Avx512Backend::floor(&a, &mut floor_result);
            Avx512Backend::ceil(&a, &mut ceil_result);
            Avx512Backend::round(&a, &mut round_result);
        }
        assert_eq!(floor_result, vec![1.0, 1.0, 1.0, -2.0, -2.0, -2.0]);
        assert_eq!(ceil_result, vec![2.0, 2.0, 2.0, -1.0, -1.0, -1.0]);
    });
}

// ========== Large-Array Tests (exercise main SIMD loop, not just remainder) ==========

#[test]
fn test_avx512_exp_large() {
    avx512_test(|| {
        // 48 elements: 3 full iterations (16 elements each)
        let input: Vec<f32> = (0..48).map(|i| i as f32 * 0.1).collect();
        assert_unary_large_relative(input, 0.05, Avx512Backend::exp, f32::exp, "exp");
    });
}

#[test]
fn test_avx512_exp_non_aligned() {
    avx512_test(|| {
        // Non-aligned sizes: 1 full SIMD iteration + remainder
        for size in [17, 19, 23, 31, 33] {
            let input: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
            assert_unary_large_relative(
                input,
                0.05,
                Avx512Backend::exp,
                f32::exp,
                &format!("exp non-aligned size={size}"),
            );
        }
    });
}

#[test]
fn test_avx512_relu_large() {
    avx512_test(|| {
        let input: Vec<f32> = (-24..24).map(|i| i as f32).collect();
        assert_unary_large(input, 1e-6, Avx512Backend::relu, |x| x.max(0.0), "relu");
    });
}

#[test]
fn test_avx512_tanh_large() {
    avx512_test(|| {
        let input: Vec<f32> = (-24..24).map(|i| i as f32 * 0.2).collect();
        assert_unary_large(input, 1e-3, Avx512Backend::tanh, f32::tanh, "tanh");
    });
}

#[test]
fn test_avx512_sigmoid_large() {
    avx512_test(|| {
        let a: Vec<f32> = (-16..16).map(|i| i as f32 * 0.5).collect();
        let mut result = vec![0.0; 32];
        unsafe {
            Avx512Backend::sigmoid(&a, &mut result);
        }
        for (i, &val) in result.iter().enumerate() {
            assert!(val >= 0.0 && val <= 1.0, "sigmoid out of range at {}: {}", i, val);
            let expected = 1.0 / (1.0 + (-a[i]).exp());
            assert!((val - expected).abs() < 1e-4, "sigmoid large mismatch at {}: {} vs {}", i, val, expected);
        }
    });
}

#[test]
fn test_avx512_gelu_large() {
    avx512_test(|| assert_activation_zero_at_origin(Avx512Backend::gelu, "gelu"));
}

#[test]
fn test_avx512_swish_large() {
    avx512_test(|| assert_activation_zero_at_origin(Avx512Backend::swish, "swish"));
}

#[test]
fn test_avx512_backend_equivalence() {
    avx512_test(|| {
        let a: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..100).map(|i| (100 - i) as f32 * 0.1).collect();
        let mut avx512_add = vec![0.0; 100];
        let mut scalar_add = vec![0.0; 100];
        unsafe {
            Avx512Backend::add(&a, &b, &mut avx512_add);
            ScalarBackend::add(&a, &b, &mut scalar_add);
        }
        for i in 0..100 {
            assert!(
                (avx512_add[i] - scalar_add[i]).abs() < 1e-5,
                "add mismatch at {}",
                i
            );
        }
    });
}
