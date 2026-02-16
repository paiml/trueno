//! Tests for activation functions (softmax, log_softmax, relu, sigmoid,
//! leaky_relu, elu, gelu, swish, hardswish, mish, selu).

use crate::vector::Vector;
use crate::{Backend, TruenoError};

// ========== Test Helpers ==========

type ActFn = fn(&Vector<f32>) -> Result<Vector<f32>, TruenoError>;

/// Assert that an activation produces expected element-wise results on a given backend.
fn assert_activation_elementwise(
    data: &[f32],
    backend: Backend,
    activation_fn: ActFn,
    expected_fn: fn(f32) -> f32,
    tolerance: f32,
    label: &str,
) {
    let v = Vector::from_slice_with_backend(data, backend);
    let result = activation_fn(&v).unwrap();
    for (i, &val) in result.as_slice().iter().enumerate() {
        let exp = expected_fn(data[i]);
        assert!(
            (val - exp).abs() < tolerance,
            "{label} {backend:?} mismatch at index {i}: got {val} expected {exp}",
        );
    }
}

/// Assert that an activation on a given backend produces values within a range.
fn assert_activation_in_range(
    data: &[f32],
    backend: Backend,
    activation_fn: ActFn,
    lo: f32,
    hi: f32,
    label: &str,
) {
    let v = Vector::from_slice_with_backend(data, backend);
    let result = activation_fn(&v).unwrap();
    for &val in result.as_slice() {
        assert!(val >= lo && val <= hi, "{label} {backend:?} out of range [{lo}, {hi}]: {val}");
    }
}

/// Assert that an activation at a specific index equals an expected value.
fn assert_activation_at(
    data: &[f32],
    backend: Backend,
    activation_fn: ActFn,
    index: usize,
    expected: f32,
    tolerance: f32,
    label: &str,
) {
    let v = Vector::from_slice_with_backend(data, backend);
    let result = activation_fn(&v).unwrap();
    let got = result.as_slice()[index];
    assert!(
        (got - expected).abs() < tolerance,
        "{label} {backend:?} at index {index}: got {got} expected {expected}",
    );
}

/// Assert backend equivalence: compare Scalar result with SSE2 and AVX2.
#[cfg(target_arch = "x86_64")]
fn assert_backend_equivalence(
    data: &[f32],
    activation_fn: ActFn,
    tolerance: f32,
    label: &str,
) {
    let scalar = activation_fn(&Vector::from_slice_with_backend(data, Backend::Scalar)).unwrap();
    for &backend in &[Backend::SSE2] {
        let other = activation_fn(&Vector::from_slice_with_backend(data, backend)).unwrap();
        for (i, (&s, &x)) in scalar.as_slice().iter().zip(other.as_slice().iter()).enumerate() {
            assert!((s - x).abs() < tolerance, "Scalar vs {backend:?} {label} mismatch at {i}: {s} vs {x}");
        }
    }
    if is_x86_feature_detected!("avx2") {
        let avx2 = activation_fn(&Vector::from_slice_with_backend(data, Backend::AVX2)).unwrap();
        for (i, (&s, &x)) in scalar.as_slice().iter().zip(avx2.as_slice().iter()).enumerate() {
            assert!((s - x).abs() < tolerance, "Scalar vs AVX2 {label} mismatch at {i}: {s} vs {x}");
        }
    }
}

// Activation adapters: wrap method calls as fn pointers for helpers.
fn act_relu(v: &Vector<f32>) -> Result<Vector<f32>, TruenoError> { v.relu() }
fn act_sigmoid(v: &Vector<f32>) -> Result<Vector<f32>, TruenoError> { v.sigmoid() }
fn act_gelu(v: &Vector<f32>) -> Result<Vector<f32>, TruenoError> { v.gelu() }
fn act_swish(v: &Vector<f32>) -> Result<Vector<f32>, TruenoError> { v.swish() }

/// Activation spec for parametric tests: (adapter, label, zero_output).
fn activation_specs() -> [(ActFn, &'static str, f32); 4] {
    [
        (act_relu, "relu", 0.0),
        (act_sigmoid, "sigmoid", 0.5),
        (act_gelu, "gelu", 0.0),
        (act_swish, "swish", 0.0),
    ]
}

// ========== Softmax ==========

#[test]
fn test_softmax_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.softmax().unwrap();
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    for &val in result.as_slice() {
        assert!((0.0..=1.0).contains(&val));
    }
    assert!(result.as_slice()[2] > result.as_slice()[1]);
    assert!(result.as_slice()[1] > result.as_slice()[0]);
}

#[test]
fn test_softmax_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.softmax(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_softmax_single() {
    let v = Vector::from_slice(&[5.0]);
    let result = v.softmax().unwrap();
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_uniform() {
    let v = Vector::from_slice(&[1.0, 1.0, 1.0, 1.0]);
    let result = v.softmax().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.25).abs() < 1e-6);
    }
}

#[test]
fn test_softmax_large_values() {
    let v = Vector::from_slice(&[1000.0, 1001.0, 1002.0]);
    let result = v.softmax().unwrap();
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_very_negative_values() {
    let v = Vector::from_slice(&[-1000.0, -999.0, -998.0]);
    let result = v.softmax().unwrap();
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
}

#[test]
fn test_softmax_large() {
    let v = Vector::from_slice(&[1.0; 100]);
    let result = v.softmax().unwrap();
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
    for &val in result.as_slice() {
        assert!((val - 0.01).abs() < 1e-4);
    }
}

#[test]
fn test_softmax_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::Scalar);
    let result = v.softmax().unwrap();
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ========== Log Softmax ==========

#[test]
fn test_log_softmax_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.log_softmax().unwrap();
    for &val in result.as_slice() {
        assert!(val <= 0.0);
    }
}

#[test]
fn test_log_softmax_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.log_softmax(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_log_softmax_single() {
    let v = Vector::from_slice(&[5.0]);
    let result = v.log_softmax().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_log_softmax_consistency_with_softmax() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let softmax = v.softmax().unwrap();
    let log_softmax = v.log_softmax().unwrap();
    for (i, &ls) in log_softmax.as_slice().iter().enumerate() {
        let from_log = ls.exp();
        assert!(
            (from_log - softmax.as_slice()[i]).abs() < 1e-5,
            "Mismatch at {i}: exp(log_softmax)={from_log}, softmax={}",
            softmax.as_slice()[i]
        );
    }
}

#[test]
fn test_log_softmax_uniform() {
    let v = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);
    let result = v.log_softmax().unwrap();
    let expected = -(4.0_f32).ln();
    for &val in result.as_slice() {
        assert!((val - expected).abs() < 1e-5);
    }
}

#[test]
fn test_log_softmax_large_values() {
    let v = Vector::from_slice(&[100.0, 101.0, 102.0]);
    let result = v.log_softmax().unwrap();
    for &val in result.as_slice() {
        assert!(val <= 0.0, "log_softmax value should be <= 0, got {val}");
    }
}

#[test]
fn test_log_softmax_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::Scalar);
    let result = v.log_softmax().unwrap();
    for &val in result.as_slice() {
        assert!(val <= 0.0);
    }
}

// ========== ReLU ==========

#[test]
fn test_relu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.relu().unwrap();
    let expected = [0.0, 0.0, 0.0, 1.0, 2.0];
    for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "relu[{i}]: {got} != {exp}");
    }
}

#[test]
fn test_relu_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.relu(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_relu_all_negative() {
    let v = Vector::from_slice(&[-5.0, -3.0, -1.0]);
    let result = v.relu().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-6);
    }
}

#[test]
fn test_relu_all_positive() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.relu().unwrap();
    let expected = [1.0, 2.0, 3.0];
    for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "relu[{i}]: {got} != {exp}");
    }
}

#[test]
fn test_relu_large() {
    let v = Vector::from_slice(&[-1.0; 1000]);
    let result = v.relu().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-6);
    }
}

// ========== Sigmoid ==========

#[test]
fn test_sigmoid_basic() {
    let v = Vector::from_slice(&[-10.0, 0.0, 10.0]);
    let result = v.sigmoid().unwrap();
    assert!(result.as_slice()[0] < 0.001);
    assert!((result.as_slice()[1] - 0.5).abs() < 1e-6);
    assert!(result.as_slice()[2] > 0.999);
}

#[test]
fn test_sigmoid_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.sigmoid(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_sigmoid_range() {
    let v = Vector::from_slice(&[-100.0, -1.0, 0.0, 1.0, 100.0]);
    let result = v.sigmoid().unwrap();
    for &val in result.as_slice() {
        assert!((0.0..=1.0).contains(&val));
    }
}

#[test]
fn test_sigmoid_large() {
    let v = Vector::from_slice(&[0.0; 1000]);
    let result = v.sigmoid().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.5).abs() < 1e-6);
    }
}

// ========== Leaky ReLU ==========

#[test]
fn test_leaky_relu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.leaky_relu(0.01).unwrap();
    let expected = [-0.02, -0.01, 0.0, 1.0, 2.0];
    for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "leaky_relu[{i}]: {got} != {exp}");
    }
}

#[test]
fn test_leaky_relu_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.leaky_relu(0.01), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_leaky_relu_different_slopes() {
    let v = Vector::from_slice(&[-1.0]);
    let r1 = v.leaky_relu(0.1).unwrap();
    assert!((r1.as_slice()[0] - (-0.1)).abs() < 1e-6);
    let r2 = v.leaky_relu(0.2).unwrap();
    assert!((r2.as_slice()[0] - (-0.2)).abs() < 1e-6);
}

#[test]
fn test_leaky_relu_slope_zero_acts_like_relu() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.leaky_relu(0.0).unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[3] - 1.0).abs() < 1e-6);
}

#[test]
fn test_leaky_relu_invalid_slopes() {
    let v = Vector::from_slice(&[1.0, 2.0]);
    assert!(matches!(v.leaky_relu(-0.1), Err(TruenoError::InvalidInput(_))));
    assert!(matches!(v.leaky_relu(1.0), Err(TruenoError::InvalidInput(_))));
    assert!(matches!(v.leaky_relu(1.5), Err(TruenoError::InvalidInput(_))));
}

// ========== ELU ==========

#[test]
fn test_elu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.elu(1.0).unwrap();
    assert!((result.as_slice()[3] - 1.0).abs() < 1e-6);
    assert!((result.as_slice()[4] - 2.0).abs() < 1e-6);
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
    assert!(result.as_slice()[0] < 0.0);
    assert!(result.as_slice()[1] < 0.0);
}

#[test]
fn test_elu_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.elu(1.0), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_elu_invalid_alpha() {
    let v = Vector::from_slice(&[1.0, 2.0]);
    assert!(matches!(v.elu(0.0), Err(TruenoError::InvalidInput(_))));
    assert!(matches!(v.elu(-1.0), Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_elu_different_alpha_values() {
    let v = Vector::from_slice(&[-1.0]);
    let r1 = v.elu(0.5).unwrap();
    let r2 = v.elu(2.0).unwrap();
    assert!(r2.as_slice()[0] < r1.as_slice()[0]);
    assert!((r1.as_slice()[0] - 0.5 * ((-1.0_f32).exp() - 1.0)).abs() < 1e-5);
    assert!((r2.as_slice()[0] - 2.0 * ((-1.0_f32).exp() - 1.0)).abs() < 1e-5);
}

// ========== GELU ==========

#[test]
fn test_gelu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.gelu().unwrap();
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
    assert!(result.as_slice()[3] > 0.5);
    assert!(result.as_slice()[4] > 1.5);
    assert!(result.as_slice()[0].abs() < 0.1);
}

#[test]
fn test_gelu_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.gelu(), Err(TruenoError::EmptyVector)));
}

// ========== Swish ==========

#[test]
fn test_swish_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.swish().unwrap();
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[3] - 0.731).abs() < 0.01);
}

#[test]
fn test_swish_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.swish(), Err(TruenoError::EmptyVector)));
}

// ========== Hardswish ==========

#[test]
fn test_hardswish_basic() {
    let v = Vector::from_slice(&[-4.0, -3.0, 0.0, 3.0, 4.0]);
    let result = v.hardswish().unwrap();
    let expected = [0.0, 0.0, 0.0, 3.0, 4.0];
    for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "hardswish[{i}]: {got} != {exp}");
    }
}

#[test]
fn test_hardswish_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.hardswish(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_hardswish_mid_range() {
    let inputs = [-2.0, -1.0, 1.0, 2.0];
    let v = Vector::from_slice(&inputs);
    let result = v.hardswish().unwrap();
    for (i, &val) in result.as_slice().iter().enumerate() {
        let x = inputs[i];
        let expected = x * (x + 3.0) / 6.0;
        assert!((val - expected).abs() < 1e-5, "hardswish({x}) = {val}, expected {expected}");
    }
}

#[test]
fn test_hardswish_continuity_at_boundaries() {
    let v = Vector::from_slice(&[-3.001, -3.0, -2.999]);
    let result = v.hardswish().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-3);
    assert!((result.as_slice()[1] - 0.0).abs() < 1e-6);
    let v2 = Vector::from_slice(&[2.999, 3.0, 3.001]);
    let result2 = v2.hardswish().unwrap();
    assert!((result2.as_slice()[1] - 3.0).abs() < 1e-5);
    assert!((result2.as_slice()[2] - 3.001).abs() < 1e-5);
}

// ========== Mish ==========

#[test]
fn test_mish_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.mish().unwrap();
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
    assert!(result.as_slice()[0] < 0.0);
}

#[test]
fn test_mish_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.mish(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_mish_extreme_values() {
    // Very negative → 0
    let v = Vector::from_slice(&[-25.0, -30.0, -50.0]);
    let result = v.mish().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-6, "mish of very negative should be 0, got {val}");
    }
    // Very positive → x
    let v2 = Vector::from_slice(&[25.0, 30.0, 50.0]);
    let result2 = v2.mish().unwrap();
    for (i, &val) in result2.as_slice().iter().enumerate() {
        let input = [25.0, 30.0, 50.0][i];
        assert!((val - input).abs() < 1e-4, "mish of very positive should be x, got {val} for {input}");
    }
    // Boundary values (no panics, finite)
    let v3 = Vector::from_slice(&[-20.0, -19.9, 19.9, 20.0]);
    let result3 = v3.mish().unwrap();
    for &val in result3.as_slice() {
        assert!(val.is_finite(), "mish should produce finite results at boundaries");
    }
}

// ========== SELU ==========

#[test]
fn test_selu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.selu().unwrap();
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
    assert!((result.as_slice()[3] - 1.0507).abs() < 0.001);
    assert!((result.as_slice()[4] - 2.1014).abs() < 0.001);
    assert!(result.as_slice()[0] < 0.0);
    assert!(result.as_slice()[1] < 0.0);
}

#[test]
fn test_selu_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.selu(), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_selu_specific_values() {
    // lambda * 1.0 = 1.0507
    let v1 = Vector::from_slice(&[1.0]);
    assert!((v1.selu().unwrap().as_slice()[0] - 1.0507).abs() < 0.001);
    // Negative: lambda * alpha * (exp(-1) - 1) ≈ -1.1113
    let v2 = Vector::from_slice(&[-1.0]);
    assert!((v2.selu().unwrap().as_slice()[0] - (-1.1113)).abs() < 0.01);
    // Very negative: ≈ -lambda * alpha ≈ -1.7581
    let v3 = Vector::from_slice(&[-50.0]);
    assert!((v3.selu().unwrap().as_slice()[0] - (-1.7581)).abs() < 0.01);
}

// ========== Combined Backend Tests (all activations at once) ==========

#[test]
fn test_all_activations_scalar_backend() {
    for (act_fn, label, zero_out) in activation_specs() {
        assert_activation_at(&[0.0], Backend::Scalar, act_fn, 0, zero_out, 1e-5, label);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_all_activations_sse2_backend() {
    for (act_fn, label, zero_out) in activation_specs() {
        assert_activation_at(&[-1.0, 0.0, 1.0, 2.0], Backend::SSE2, act_fn, 1, zero_out, 1e-5, label);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_all_activations_avx2_backend() {
    if !is_x86_feature_detected!("avx2") { return; }
    let data: Vec<f32> = (-8..8).map(|i| i as f32).collect();
    for (act_fn, label, zero_out) in activation_specs() {
        assert_activation_at(&data, Backend::AVX2, act_fn, 8, zero_out, 1e-5, label);
    }
}

#[test]
fn test_all_activations_fallback_backends() {
    for (act_fn, label, zero_out) in activation_specs() {
        for backend in [Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto] {
            assert_activation_at(&[0.0, 1.0], backend, act_fn, 0, zero_out, 1e-5, label);
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_all_activations_avx512_backend() {
    if !is_x86_feature_detected!("avx512f") { return; }
    let data: Vec<f32> = (-8..8).map(|i| i as f32 * 0.5).collect();
    for (act_fn, label, zero_out) in activation_specs() {
        assert_activation_at(&data, Backend::AVX512, act_fn, 8, zero_out, 1e-5, label);
    }
}

// ========== Combined Backend Equivalence (all activations) ==========

#[test]
fn test_all_activations_backend_equivalence() {
    let tolerances = [1e-5, 1e-2, 1e-2, 1e-2];
    for ((act_fn, label, _), &tol) in activation_specs().iter().zip(tolerances.iter()) {
        let data: Vec<f32> = (-20..20).map(|i| i as f32 * 0.5).collect();
        #[cfg(target_arch = "x86_64")]
        assert_backend_equivalence(&data, *act_fn, tol, label);
        #[cfg(not(target_arch = "x86_64"))]
        let _ = (data, act_fn, tol, label);
    }
}

// ========== Combined Non-Aligned Size Tests ==========

#[test]
fn test_all_activations_non_aligned_sizes() {
    let sizes = [1, 3, 5, 7, 9, 13, 15, 17, 31, 33];
    for (act_fn, label, _) in activation_specs() {
        for &size in &sizes {
            let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
            let v = Vector::from_slice(&data);
            let result = act_fn(&v).unwrap();
            assert_eq!(result.as_slice().len(), size, "{label} non-aligned size={size}");
        }
    }
}

// ========== ReLU-specific SIMD verification ==========

#[test]
fn test_relu_elementwise_avx2() {
    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx2") { return; }
        let data: Vec<f32> = (-16..16).map(|i| i as f32).collect();
        assert_activation_elementwise(&data, Backend::AVX2, act_relu, |x| x.max(0.0), 1e-6, "relu");
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_avx512_non_aligned() {
    if !is_x86_feature_detected!("avx512f") { return; }
    for size in [17, 19, 23, 31, 33] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
        assert_activation_elementwise(&data, Backend::AVX512, act_relu, |x| x.max(0.0), 1e-6, "relu AVX512");
    }
}

// ========== Sigmoid-specific range tests on SIMD backends ==========

#[test]
#[cfg(target_arch = "x86_64")]
fn test_sigmoid_avx2_range() {
    if !is_x86_feature_detected!("avx2") { return; }
    let data: Vec<f32> = (-8..8).map(|i| i as f32).collect();
    assert_activation_in_range(&data, Backend::AVX2, act_sigmoid, 0.0, 1.0, "sigmoid");
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_sigmoid_avx512_range() {
    if !is_x86_feature_detected!("avx512f") { return; }
    let data: Vec<f32> = (-16..16).map(|i| i as f32 * 0.5).collect();
    assert_activation_in_range(&data, Backend::AVX512, act_sigmoid, 0.0, 1.0, "sigmoid");
}

// ========== ReLU: Comprehensive Backend Dispatch Coverage ==========

#[test]
fn test_relu_scalar_backend_comprehensive() {
    // Exercise Backend::Scalar arm in dispatch_unary_op for relu
    let data = [-5.0, -1.0, -0.001, 0.0, 0.001, 1.0, 5.0];
    assert_activation_elementwise(
        &data, Backend::Scalar, act_relu, |x| x.max(0.0), 1e-6, "relu Scalar",
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_sse2_backend_comprehensive() {
    // Exercise Backend::SSE2 arm: Sse2Backend::relu
    let data: Vec<f32> = (-20..20).map(|i| i as f32 * 0.5).collect();
    assert_activation_elementwise(
        &data, Backend::SSE2, act_relu, |x| x.max(0.0), 1e-6, "relu SSE2",
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_avx_backend_comprehensive() {
    // Exercise Backend::AVX arm (dispatches to Sse2Backend)
    let data: Vec<f32> = (-20..20).map(|i| i as f32 * 0.5).collect();
    assert_activation_elementwise(
        &data, Backend::AVX, act_relu, |x| x.max(0.0), 1e-6, "relu AVX",
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_avx2_backend_comprehensive() {
    if !is_x86_feature_detected!("avx2") { return; }
    // Exercise Backend::AVX2 arm: Avx2Backend::relu
    // Use 33 elements to test non-aligned remainder (33 % 8 = 1)
    let data: Vec<f32> = (-16..17).map(|i| i as f32 * 0.3).collect();
    assert_activation_elementwise(
        &data, Backend::AVX2, act_relu, |x| x.max(0.0), 1e-6, "relu AVX2",
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_avx512_backend_comprehensive() {
    if !is_x86_feature_detected!("avx512f") { return; }
    // Exercise Backend::AVX512 arm: Avx2Backend::relu
    // Use 35 elements (35 % 16 = 3) to test remainder handling
    let data: Vec<f32> = (-17..18).map(|i| i as f32 * 0.2).collect();
    assert_activation_elementwise(
        &data, Backend::AVX512, act_relu, |x| x.max(0.0), 1e-6, "relu AVX512",
    );
}

#[test]
fn test_relu_neon_fallback_backend() {
    // On non-ARM, NEON falls back to ScalarBackend
    let data = [-3.0, -1.0, 0.0, 1.0, 3.0];
    assert_activation_elementwise(
        &data, Backend::NEON, act_relu, |x| x.max(0.0), 1e-6, "relu NEON",
    );
}

#[test]
fn test_relu_wasm_fallback_backend() {
    // On non-WASM, WasmSIMD falls back to ScalarBackend
    let data = [-3.0, -1.0, 0.0, 1.0, 3.0];
    assert_activation_elementwise(
        &data, Backend::WasmSIMD, act_relu, |x| x.max(0.0), 1e-6, "relu WasmSIMD",
    );
}

#[test]
fn test_relu_gpu_backend() {
    // GPU falls back to ScalarBackend in dispatch_unary_op
    let data = [-3.0, -1.0, 0.0, 1.0, 3.0];
    assert_activation_elementwise(
        &data, Backend::GPU, act_relu, |x| x.max(0.0), 1e-6, "relu GPU",
    );
}

#[test]
fn test_relu_auto_backend() {
    // Auto falls back to ScalarBackend in dispatch_unary_op
    let data = [-3.0, -1.0, 0.0, 1.0, 3.0];
    assert_activation_elementwise(
        &data, Backend::Auto, act_relu, |x| x.max(0.0), 1e-6, "relu Auto",
    );
}

// ========== ReLU: Backend Equivalence Across All Backends ==========

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_backend_equivalence_comprehensive() {
    // Large data set with mixed positive/negative to exercise all SIMD lanes
    let data: Vec<f32> = (-50..50).map(|i| i as f32 * 0.37).collect();
    let scalar_v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let expected = scalar_v.relu().unwrap();

    // SSE2 and AVX (both dispatch to Sse2Backend)
    for &backend in &[Backend::SSE2, Backend::AVX] {
        let v = Vector::from_slice_with_backend(&data, backend);
        let result = v.relu().unwrap();
        for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.as_slice().iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "relu Scalar vs {backend:?} at [{i}]: {got} vs {exp}",
            );
        }
    }

    // AVX2 and AVX512 (both dispatch to Avx2Backend)
    if is_x86_feature_detected!("avx2") {
        for &backend in &[Backend::AVX2, Backend::AVX512] {
            let v = Vector::from_slice_with_backend(&data, backend);
            let result = v.relu().unwrap();
            for (i, (&got, &exp)) in
                result.as_slice().iter().zip(expected.as_slice().iter()).enumerate()
            {
                assert!(
                    (got - exp).abs() < 1e-6,
                    "relu Scalar vs {backend:?} at [{i}]: {got} vs {exp}",
                );
            }
        }
    }

    // Fallback backends (NEON, WasmSIMD, GPU, Auto) all use ScalarBackend on x86_64
    for &backend in &[Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto] {
        let v = Vector::from_slice_with_backend(&data, backend);
        let result = v.relu().unwrap();
        for (i, (&got, &exp)) in result.as_slice().iter().zip(expected.as_slice().iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "relu Scalar vs {backend:?} at [{i}]: {got} vs {exp}",
            );
        }
    }
}

// ========== ReLU: Non-Aligned Size Tests Per Backend ==========

#[test]
fn test_relu_non_aligned_sizes_per_backend() {
    // Test with sizes that don't align to SSE2(4), AVX2(8), or AVX-512(16) lanes
    let sizes = [1, 2, 3, 5, 7, 9, 11, 15, 17, 31, 33, 63, 65];
    let backends = [Backend::Scalar, Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto];

    for &size in &sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
        for &backend in &backends {
            let v = Vector::from_slice_with_backend(&data, backend);
            let result = v.relu().unwrap();
            assert_eq!(result.as_slice().len(), size, "relu {backend:?} size={size}");
            for (i, &val) in result.as_slice().iter().enumerate() {
                let exp = data[i].max(0.0);
                assert!(
                    (val - exp).abs() < 1e-6,
                    "relu {backend:?} size={size} [{i}]: {val} vs {exp}",
                );
            }
        }
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_non_aligned_sizes_simd_backends() {
    let sizes = [1, 3, 5, 7, 9, 11, 15, 17, 31, 33, 63, 65];
    for &size in &sizes {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();

        // SSE2 / AVX (both use Sse2Backend, 4-wide)
        for &backend in &[Backend::SSE2, Backend::AVX] {
            let v = Vector::from_slice_with_backend(&data, backend);
            let result = v.relu().unwrap();
            for (i, &val) in result.as_slice().iter().enumerate() {
                let exp = data[i].max(0.0);
                assert!(
                    (val - exp).abs() < 1e-6,
                    "relu {backend:?} size={size} [{i}]: {val} vs {exp}",
                );
            }
        }

        // AVX2 / AVX512 (use Avx2Backend, 8-wide / 16-wide)
        if is_x86_feature_detected!("avx2") {
            for &backend in &[Backend::AVX2, Backend::AVX512] {
                let v = Vector::from_slice_with_backend(&data, backend);
                let result = v.relu().unwrap();
                for (i, &val) in result.as_slice().iter().enumerate() {
                    let exp = data[i].max(0.0);
                    assert!(
                        (val - exp).abs() < 1e-6,
                        "relu {backend:?} size={size} [{i}]: {val} vs {exp}",
                    );
                }
            }
        }
    }
}

// ========== ReLU: Edge Cases ==========

#[test]
fn test_relu_single_element_backends() {
    for &backend in &[Backend::Scalar, Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto] {
        let v_neg = Vector::from_slice_with_backend(&[-1.0], backend);
        assert!((v_neg.relu().unwrap().as_slice()[0] - 0.0).abs() < 1e-6, "relu {backend:?} single neg");

        let v_pos = Vector::from_slice_with_backend(&[1.0], backend);
        assert!((v_pos.relu().unwrap().as_slice()[0] - 1.0).abs() < 1e-6, "relu {backend:?} single pos");

        let v_zero = Vector::from_slice_with_backend(&[0.0], backend);
        assert!((v_zero.relu().unwrap().as_slice()[0] - 0.0).abs() < 1e-6, "relu {backend:?} single zero");
    }
}

#[test]
fn test_relu_special_float_values() {
    let data = [f32::INFINITY, f32::NEG_INFINITY, f32::MIN, f32::MAX, f32::EPSILON, -f32::EPSILON];
    let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result = v.relu().unwrap();
    assert_eq!(result.as_slice()[0], f32::INFINITY);       // relu(+inf) = +inf
    assert_eq!(result.as_slice()[1], 0.0);                  // relu(-inf) = 0
    assert_eq!(result.as_slice()[2], 0.0);                  // relu(f32::MIN) = 0 (MIN is negative)
    assert_eq!(result.as_slice()[3], f32::MAX);             // relu(f32::MAX) = MAX
    assert_eq!(result.as_slice()[4], f32::EPSILON);         // relu(EPSILON) = EPSILON
    assert_eq!(result.as_slice()[5], 0.0);                  // relu(-EPSILON) = 0
}

#[test]
fn test_relu_nan_handling() {
    let data = [f32::NAN, -1.0, 1.0, f32::NAN];
    let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result = v.relu().unwrap();
    // NaN behavior: max(0, NaN) is NaN in IEEE 754 (implementation-defined in SIMD)
    // Just verify no panic and correct non-NaN values
    assert_eq!(result.as_slice()[1], 0.0);
    assert_eq!(result.as_slice()[2], 1.0);
}

// ========== ReLU: Parallel Path (Refs CB-130) ==========

#[test]
#[cfg(feature = "parallel")]
fn test_relu_parallel_large_vector() {
    // Trigger the parallel path: len >= 500_000
    let size = 500_000;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result = v.relu().unwrap();
    assert_eq!(result.as_slice().len(), size);

    // Verify correctness: first half should be 0, second half should be positive
    for (i, &val) in result.as_slice().iter().enumerate() {
        let exp = data[i].max(0.0);
        assert!(
            (val - exp).abs() < 1e-6,
            "relu parallel [{i}]: got {val} expected {exp}",
        );
    }
}

#[test]
#[cfg(feature = "parallel")]
fn test_relu_parallel_boundary() {
    // Just below threshold (499_999) - sequential path
    let size = 499_999;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result = v.relu().unwrap();
    assert_eq!(result.as_slice().len(), size);

    // Exactly at threshold (500_000) - parallel path
    let size = 500_000;
    let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result = v.relu().unwrap();
    assert_eq!(result.as_slice().len(), size);
}

#[test]
#[cfg(all(feature = "parallel", target_arch = "x86_64"))]
fn test_relu_parallel_with_simd_backends() {
    // Parallel + SIMD dispatch combined
    let size = 600_000;
    let data: Vec<f32> = (0..size).map(|i| ((i % 100) as f32) - 50.0).collect();

    let v_scalar = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let expected = v_scalar.relu().unwrap();

    // SSE2 parallel path
    let v_sse = Vector::from_slice_with_backend(&data, Backend::SSE2);
    let result_sse = v_sse.relu().unwrap();
    for (i, (&got, &exp)) in result_sse
        .as_slice()
        .iter()
        .zip(expected.as_slice().iter())
        .enumerate()
        .take(100)  // Spot-check first 100
    {
        assert!(
            (got - exp).abs() < 1e-6,
            "relu parallel Scalar vs SSE2 [{i}]: {got} vs {exp}",
        );
    }

    if is_x86_feature_detected!("avx2") {
        let v_avx2 = Vector::from_slice_with_backend(&data, Backend::AVX2);
        let result_avx2 = v_avx2.relu().unwrap();
        for (i, (&got, &exp)) in result_avx2
            .as_slice()
            .iter()
            .zip(expected.as_slice().iter())
            .enumerate()
            .take(100)  // Spot-check first 100
        {
            assert!(
                (got - exp).abs() < 1e-6,
                "relu parallel Scalar vs AVX2 [{i}]: {got} vs {exp}",
            );
        }
    }
}

#[test]
#[cfg(feature = "parallel")]
fn test_relu_parallel_all_negative() {
    // All negative: entire output should be zeros
    let size = 500_000;
    let data = vec![-1.0; size];
    let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result = v.relu().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-6);
    }
}

#[test]
#[cfg(feature = "parallel")]
fn test_relu_parallel_all_positive() {
    // All positive: output should equal input
    let size = 500_000;
    let data = vec![42.0; size];
    let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
    let result = v.relu().unwrap();
    for &val in result.as_slice() {
        assert!((val - 42.0).abs() < 1e-6);
    }
}
