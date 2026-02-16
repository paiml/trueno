//! Tests for activation functions (softmax, log_softmax, relu, sigmoid,
//! leaky_relu, elu, gelu, swish, hardswish, mish, selu).

use crate::vector::Vector;
use crate::{Backend, TruenoError};

// ========== Softmax ==========

#[test]
fn test_softmax_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.softmax().unwrap();
    // Check sum = 1
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
    // Check all values in [0, 1]
    for &val in result.as_slice() {
        assert!((0.0..=1.0).contains(&val));
    }
    // Check highest input has highest probability
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
    // All equal inputs should give equal outputs
    for &val in result.as_slice() {
        assert!((val - 0.25).abs() < 1e-6);
    }
}

#[test]
fn test_softmax_large_values() {
    // Test numerical stability with large values
    let v = Vector::from_slice(&[1000.0, 1001.0, 1002.0]);
    let result = v.softmax().unwrap();
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

// ========== Log Softmax ==========

#[test]
fn test_log_softmax_basic() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.log_softmax().unwrap();
    // All log probabilities should be <= 0
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
    // log(1) = 0
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
}

// ========== ReLU ==========

#[test]
fn test_relu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.relu().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[3] - 1.0).abs() < 1e-6);
    assert!((result.as_slice()[4] - 2.0).abs() < 1e-6);
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
    assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 2.0).abs() < 1e-6);
    assert!((result.as_slice()[2] - 3.0).abs() < 1e-6);
}

// ========== Sigmoid ==========

#[test]
fn test_sigmoid_basic() {
    let v = Vector::from_slice(&[-10.0, 0.0, 10.0]);
    let result = v.sigmoid().unwrap();
    // sigmoid(-10) ≈ 0
    assert!(result.as_slice()[0] < 0.001);
    // sigmoid(0) = 0.5
    assert!((result.as_slice()[1] - 0.5).abs() < 1e-6);
    // sigmoid(10) ≈ 1
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

// ========== Leaky ReLU ==========

#[test]
fn test_leaky_relu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.leaky_relu(0.01).unwrap();
    assert!((result.as_slice()[0] - (-0.02)).abs() < 1e-6);
    assert!((result.as_slice()[1] - (-0.01)).abs() < 1e-6);
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[3] - 1.0).abs() < 1e-6);
    assert!((result.as_slice()[4] - 2.0).abs() < 1e-6);
}

#[test]
fn test_leaky_relu_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.leaky_relu(0.01), Err(TruenoError::EmptyVector)));
}

#[test]
fn test_leaky_relu_different_slopes() {
    let v = Vector::from_slice(&[-1.0]);
    // slope 0.1
    let result = v.leaky_relu(0.1).unwrap();
    assert!((result.as_slice()[0] - (-0.1)).abs() < 1e-6);
    // slope 0.2
    let result = v.leaky_relu(0.2).unwrap();
    assert!((result.as_slice()[0] - (-0.2)).abs() < 1e-6);
}

// ========== ELU ==========

#[test]
fn test_elu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.elu(1.0).unwrap();
    // Positive values unchanged
    assert!((result.as_slice()[3] - 1.0).abs() < 1e-6);
    assert!((result.as_slice()[4] - 2.0).abs() < 1e-6);
    // Zero stays zero
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
    // Negative values: alpha * (exp(x) - 1)
    assert!(result.as_slice()[0] < 0.0);
    assert!(result.as_slice()[1] < 0.0);
}

#[test]
fn test_elu_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.elu(1.0), Err(TruenoError::EmptyVector)));
}

// ========== GELU ==========

#[test]
fn test_gelu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.gelu().unwrap();
    // GELU(0) = 0
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
    // GELU is approximately linear for positive values
    assert!(result.as_slice()[3] > 0.5);
    assert!(result.as_slice()[4] > 1.5);
    // Negative values are small but not zero
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
    // Swish(0) = 0
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
    // Swish(x) = x * sigmoid(x)
    // Swish(1) ≈ 0.731
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
    // x <= -3: hardswish(x) = 0
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 0.0).abs() < 1e-6);
    // x >= 3: hardswish(x) = x
    assert!((result.as_slice()[3] - 3.0).abs() < 1e-6);
    assert!((result.as_slice()[4] - 4.0).abs() < 1e-6);
    // x = 0: hardswish(0) = 0
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
}

#[test]
fn test_hardswish_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.hardswish(), Err(TruenoError::EmptyVector)));
}

// ========== Mish ==========

#[test]
fn test_mish_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.mish().unwrap();
    // Mish(0) = 0
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-6);
    // Mish is smooth and non-monotonic for negative values
    assert!(result.as_slice()[0] < 0.0);
}

#[test]
fn test_mish_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.mish(), Err(TruenoError::EmptyVector)));
}

// ========== SELU ==========

#[test]
fn test_selu_basic() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.selu().unwrap();
    // SELU(0) = 0
    assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
    // Positive values scaled by λ ≈ 1.0507
    assert!((result.as_slice()[3] - 1.0507).abs() < 0.001);
    assert!((result.as_slice()[4] - 2.1014).abs() < 0.001);
    // Negative values use ELU-like formula
    assert!(result.as_slice()[0] < 0.0);
    assert!(result.as_slice()[1] < 0.0);
}

#[test]
fn test_selu_empty() {
    let v = Vector::<f32>::from_slice(&[]);
    assert!(matches!(v.selu(), Err(TruenoError::EmptyVector)));
}

// ========== Backend Tests ==========

#[test]
fn test_relu_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[-1.0, 0.0, 1.0], Backend::Scalar);
    let result = v.relu().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[2] - 1.0).abs() < 1e-6);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_sse2_backend() {
    let v = Vector::from_slice_with_backend(&[-1.0, 0.0, 1.0, 2.0], Backend::SSE2);
    let result = v.relu().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[2] - 1.0).abs() < 1e-6);
}

#[test]
fn test_sigmoid_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[0.0], Backend::Scalar);
    let result = v.sigmoid().unwrap();
    assert!((result.as_slice()[0] - 0.5).abs() < 1e-6);
}

// ========== Large Array Tests ==========

#[test]
fn test_relu_large() {
    let v = Vector::from_slice(&[-1.0; 1000]);
    let result = v.relu().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-6);
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

#[test]
fn test_softmax_large() {
    let v = Vector::from_slice(&[1.0; 100]);
    let result = v.softmax().unwrap();
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
    // All equal inputs should give equal probabilities
    for &val in result.as_slice() {
        assert!((val - 0.01).abs() < 1e-4);
    }
}

// ========== Backend Dispatch Coverage ==========
// Tests to exercise all backend dispatch paths through the dispatch_unary_op macro

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_avx2_backend() {
    if !is_x86_feature_detected!("avx2") {
        return; // Skip if AVX2 not available
    }
    let data: Vec<f32> = (-16..16).map(|i| i as f32).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
    let result = v.relu().unwrap();
    for (i, &val) in result.as_slice().iter().enumerate() {
        let expected = if data[i] > 0.0 { data[i] } else { 0.0 };
        assert!((val - expected).abs() < 1e-6, "relu AVX2 mismatch at index {}", i);
    }
}

#[test]
fn test_relu_neon_backend_fallback() {
    // NEON backend falls back to scalar on non-ARM
    let v = Vector::from_slice_with_backend(&[-1.0, 0.0, 1.0, 2.0], Backend::NEON);
    let result = v.relu().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[2] - 1.0).abs() < 1e-6);
    assert!((result.as_slice()[3] - 2.0).abs() < 1e-6);
}

#[test]
fn test_relu_wasm_backend_fallback() {
    // WASM backend falls back to scalar on non-WASM
    let v = Vector::from_slice_with_backend(&[-3.0, -1.0, 0.0, 5.0], Backend::WasmSIMD);
    let result = v.relu().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[3] - 5.0).abs() < 1e-6);
}

#[test]
fn test_relu_gpu_backend_fallback() {
    // GPU backend falls back to scalar
    let v = Vector::from_slice_with_backend(&[-2.0, 0.0, 3.0], Backend::GPU);
    let result = v.relu().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[2] - 3.0).abs() < 1e-6);
}

#[test]
fn test_relu_auto_backend_fallback() {
    // Auto backend falls back to scalar
    let v = Vector::from_slice_with_backend(&[-1.0, 2.0], Backend::Auto);
    let result = v.relu().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 2.0).abs() < 1e-6);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_sigmoid_avx2_backend() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let data: Vec<f32> = (-8..8).map(|i| i as f32).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
    let result = v.sigmoid().unwrap();
    for &val in result.as_slice() {
        assert!(val >= 0.0 && val <= 1.0, "sigmoid AVX2 out of range: {}", val);
    }
    // sigmoid(0) should be at index 8 (value 0.0)
    assert!((result.as_slice()[8] - 0.5).abs() < 1e-5);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_sigmoid_sse2_backend() {
    let v = Vector::from_slice_with_backend(&[-5.0, 0.0, 5.0, 10.0], Backend::SSE2);
    let result = v.sigmoid().unwrap();
    assert!(result.as_slice()[0] < 0.01);
    assert!((result.as_slice()[1] - 0.5).abs() < 1e-5);
    assert!(result.as_slice()[2] > 0.99);
}

#[test]
fn test_sigmoid_neon_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0, 1.0], Backend::NEON);
    let result = v.sigmoid().unwrap();
    assert!((result.as_slice()[0] - 0.5).abs() < 1e-5);
    assert!(result.as_slice()[1] > 0.5);
}

#[test]
fn test_sigmoid_wasm_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[-1.0, 0.0, 1.0], Backend::WasmSIMD);
    let result = v.sigmoid().unwrap();
    assert!((result.as_slice()[1] - 0.5).abs() < 1e-5);
}

#[test]
fn test_sigmoid_gpu_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0], Backend::GPU);
    let result = v.sigmoid().unwrap();
    assert!((result.as_slice()[0] - 0.5).abs() < 1e-5);
}

#[test]
fn test_sigmoid_auto_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0], Backend::Auto);
    let result = v.sigmoid().unwrap();
    assert!((result.as_slice()[0] - 0.5).abs() < 1e-5);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_gelu_sse2_backend() {
    let v = Vector::from_slice_with_backend(&[-1.0, 0.0, 1.0, 2.0], Backend::SSE2);
    let result = v.gelu().unwrap();
    assert!((result.as_slice()[1] - 0.0).abs() < 1e-5);
    assert!(result.as_slice()[2] > 0.5);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_gelu_avx2_backend() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let data: Vec<f32> = (-8..8).map(|i| i as f32).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
    let result = v.gelu().unwrap();
    assert!((result.as_slice()[8] - 0.0).abs() < 1e-5); // gelu(0) = 0
}

#[test]
fn test_gelu_neon_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0, 1.0, -1.0], Backend::NEON);
    let result = v.gelu().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_gelu_wasm_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0, 2.0], Backend::WasmSIMD);
    let result = v.gelu().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_gelu_gpu_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0, 1.0], Backend::GPU);
    let result = v.gelu().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_gelu_auto_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0], Backend::Auto);
    let result = v.gelu().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_swish_sse2_backend() {
    let v = Vector::from_slice_with_backend(&[-1.0, 0.0, 1.0, 2.0], Backend::SSE2);
    let result = v.swish().unwrap();
    assert!((result.as_slice()[1] - 0.0).abs() < 1e-5);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_swish_avx2_backend() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }
    let data: Vec<f32> = (-8..8).map(|i| i as f32).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
    let result = v.swish().unwrap();
    assert!((result.as_slice()[8] - 0.0).abs() < 1e-5); // swish(0) = 0
}

#[test]
fn test_swish_neon_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0, 1.0], Backend::NEON);
    let result = v.swish().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_swish_wasm_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0], Backend::WasmSIMD);
    let result = v.swish().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_swish_gpu_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0, 1.0], Backend::GPU);
    let result = v.swish().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_swish_auto_backend_fallback() {
    let v = Vector::from_slice_with_backend(&[0.0], Backend::Auto);
    let result = v.swish().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
}

// ========== Softmax Backend Tests ==========

#[test]
fn test_softmax_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::Scalar);
    let result = v.softmax().unwrap();
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_log_softmax_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::Scalar);
    let result = v.log_softmax().unwrap();
    for &val in result.as_slice() {
        assert!(val <= 0.0);
    }
}

// ========== Parameter Validation Tests ==========

#[test]
fn test_leaky_relu_invalid_negative_slope_negative() {
    let v = Vector::from_slice(&[1.0, 2.0]);
    let result = v.leaky_relu(-0.1);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_leaky_relu_invalid_negative_slope_too_large() {
    let v = Vector::from_slice(&[1.0, 2.0]);
    let result = v.leaky_relu(1.0);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_leaky_relu_invalid_negative_slope_above_one() {
    let v = Vector::from_slice(&[1.0, 2.0]);
    let result = v.leaky_relu(1.5);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_leaky_relu_slope_zero_acts_like_relu() {
    let v = Vector::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
    let result = v.leaky_relu(0.0).unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[1] - 0.0).abs() < 1e-6);
    assert!((result.as_slice()[3] - 1.0).abs() < 1e-6);
}

#[test]
fn test_elu_invalid_alpha_zero() {
    let v = Vector::from_slice(&[1.0, 2.0]);
    let result = v.elu(0.0);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_elu_invalid_alpha_negative() {
    let v = Vector::from_slice(&[1.0, 2.0]);
    let result = v.elu(-1.0);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_elu_different_alpha_values() {
    let v = Vector::from_slice(&[-1.0]);
    let result1 = v.elu(0.5).unwrap();
    let result2 = v.elu(2.0).unwrap();
    // Larger alpha gives more negative output for negative inputs
    assert!(result2.as_slice()[0] < result1.as_slice()[0]);
    // Both should be: alpha * (exp(-1) - 1) ≈ alpha * (-0.632)
    assert!((result1.as_slice()[0] - 0.5 * ((-1.0_f32).exp() - 1.0)).abs() < 1e-5);
    assert!((result2.as_slice()[0] - 2.0 * ((-1.0_f32).exp() - 1.0)).abs() < 1e-5);
}

// ========== Softmax and Log-Softmax Numerical Edge Cases ==========

#[test]
fn test_softmax_very_negative_values() {
    let v = Vector::from_slice(&[-1000.0, -999.0, -998.0]);
    let result = v.softmax().unwrap();
    let sum: f32 = result.as_slice().iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
}

#[test]
fn test_log_softmax_consistency_with_softmax() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let softmax = v.softmax().unwrap();
    let log_softmax = v.log_softmax().unwrap();
    // exp(log_softmax) should equal softmax
    for (i, &ls) in log_softmax.as_slice().iter().enumerate() {
        let from_log = ls.exp();
        assert!(
            (from_log - softmax.as_slice()[i]).abs() < 1e-5,
            "Mismatch at index {}: exp(log_softmax)={}, softmax={}",
            i,
            from_log,
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
        assert!(val <= 0.0, "log_softmax value should be <= 0, got {}", val);
    }
}

// ========== Mish Edge Cases ==========

#[test]
fn test_mish_very_negative() {
    // For x < -20, mish(x) should return 0
    let v = Vector::from_slice(&[-25.0, -30.0, -50.0]);
    let result = v.mish().unwrap();
    for &val in result.as_slice() {
        assert!((val - 0.0).abs() < 1e-6, "mish of very negative should be 0, got {}", val);
    }
}

#[test]
fn test_mish_very_positive() {
    // For x > 20, mish(x) should return x
    let v = Vector::from_slice(&[25.0, 30.0, 50.0]);
    let result = v.mish().unwrap();
    for (i, &val) in result.as_slice().iter().enumerate() {
        let input = [25.0, 30.0, 50.0][i];
        assert!((val - input).abs() < 1e-4, "mish of very positive should be x, got {} for input {}", val, input);
    }
}

#[test]
fn test_mish_boundary_values() {
    // Test around the boundary at x = -20 and x = 20
    let v = Vector::from_slice(&[-20.0, -19.9, 19.9, 20.0]);
    let result = v.mish().unwrap();
    // Just verifying no panics and values are finite
    for &val in result.as_slice() {
        assert!(val.is_finite(), "mish should produce finite results at boundaries");
    }
}

// ========== Hardswish Edge Cases ==========

#[test]
fn test_hardswish_mid_range() {
    // Test the middle region -3 < x < 3 where hardswish(x) = x * (x + 3) / 6
    let v = Vector::from_slice(&[-2.0, -1.0, 1.0, 2.0]);
    let result = v.hardswish().unwrap();
    for (i, &val) in result.as_slice().iter().enumerate() {
        let x = [-2.0, -1.0, 1.0, 2.0][i];
        let expected = x * (x + 3.0) / 6.0;
        assert!((val - expected).abs() < 1e-5, "hardswish({}) = {}, expected {}", x, val, expected);
    }
}

#[test]
fn test_hardswish_continuity_at_boundaries() {
    // At x = -3: limit from right should be 0
    let v = Vector::from_slice(&[-3.001, -3.0, -2.999]);
    let result = v.hardswish().unwrap();
    assert!((result.as_slice()[0] - 0.0).abs() < 1e-3);
    assert!((result.as_slice()[1] - 0.0).abs() < 1e-6);
    // At x = 3: limit from left should be 3
    let v2 = Vector::from_slice(&[2.999, 3.0, 3.001]);
    let result2 = v2.hardswish().unwrap();
    assert!((result2.as_slice()[1] - 3.0).abs() < 1e-5);
    assert!((result2.as_slice()[2] - 3.001).abs() < 1e-5);
}

// ========== SELU Edge Cases ==========

#[test]
fn test_selu_single_positive() {
    let v = Vector::from_slice(&[1.0]);
    let result = v.selu().unwrap();
    // lambda * 1.0 = 1.0507
    assert!((result.as_slice()[0] - 1.0507).abs() < 0.001);
}

#[test]
fn test_selu_single_negative() {
    let v = Vector::from_slice(&[-1.0]);
    let result = v.selu().unwrap();
    // lambda * alpha * (exp(-1) - 1) ≈ 1.0507 * 1.6733 * (0.3679 - 1) ≈ -1.1113
    assert!(result.as_slice()[0] < 0.0);
    assert!((result.as_slice()[0] - (-1.1113)).abs() < 0.01);
}

#[test]
fn test_selu_large_negative() {
    // For very negative x, selu(x) ≈ -lambda * alpha ≈ -1.7581
    let v = Vector::from_slice(&[-50.0]);
    let result = v.selu().unwrap();
    assert!(result.as_slice()[0] < 0.0);
    assert!((result.as_slice()[0] - (-1.7581)).abs() < 0.01);
}

// ========== Backend Equivalence Tests ==========

#[test]
fn test_relu_backend_equivalence() {
    let data: Vec<f32> = (-50..50).map(|i| i as f32 * 0.1).collect();
    let scalar = Vector::from_slice_with_backend(&data, Backend::Scalar).relu().unwrap();

    #[cfg(target_arch = "x86_64")]
    {
        let sse2 = Vector::from_slice_with_backend(&data, Backend::SSE2).relu().unwrap();
        for (i, (&s, &x)) in scalar.as_slice().iter().zip(sse2.as_slice().iter()).enumerate() {
            assert!((s - x).abs() < 1e-5, "Scalar vs SSE2 mismatch at index {}", i);
        }

        if is_x86_feature_detected!("avx2") {
            let avx2 = Vector::from_slice_with_backend(&data, Backend::AVX2).relu().unwrap();
            for (i, (&s, &x)) in scalar.as_slice().iter().zip(avx2.as_slice().iter()).enumerate() {
                assert!((s - x).abs() < 1e-5, "Scalar vs AVX2 mismatch at index {}", i);
            }
        }
    }
}

#[test]
fn test_sigmoid_backend_equivalence() {
    let data: Vec<f32> = (-20..20).map(|i| i as f32 * 0.5).collect();
    let scalar = Vector::from_slice_with_backend(&data, Backend::Scalar).sigmoid().unwrap();

    #[cfg(target_arch = "x86_64")]
    {
        let sse2 = Vector::from_slice_with_backend(&data, Backend::SSE2).sigmoid().unwrap();
        for (i, (&s, &x)) in scalar.as_slice().iter().zip(sse2.as_slice().iter()).enumerate() {
            assert!((s - x).abs() < 1e-2, "Scalar vs SSE2 sigmoid mismatch at index {}: {} vs {}", i, s, x);
        }

        if is_x86_feature_detected!("avx2") {
            let avx2 = Vector::from_slice_with_backend(&data, Backend::AVX2).sigmoid().unwrap();
            for (i, (&s, &x)) in scalar.as_slice().iter().zip(avx2.as_slice().iter()).enumerate() {
                assert!((s - x).abs() < 1e-2, "Scalar vs AVX2 sigmoid mismatch at index {}: {} vs {}", i, s, x);
            }
        }
    }
}

#[test]
fn test_gelu_backend_equivalence() {
    let data: Vec<f32> = (-10..10).map(|i| i as f32 * 0.5).collect();
    let scalar = Vector::from_slice_with_backend(&data, Backend::Scalar).gelu().unwrap();

    #[cfg(target_arch = "x86_64")]
    {
        let sse2 = Vector::from_slice_with_backend(&data, Backend::SSE2).gelu().unwrap();
        for (i, (&s, &x)) in scalar.as_slice().iter().zip(sse2.as_slice().iter()).enumerate() {
            assert!((s - x).abs() < 1e-2, "Scalar vs SSE2 gelu mismatch at index {}: {} vs {}", i, s, x);
        }

        if is_x86_feature_detected!("avx2") {
            let avx2 = Vector::from_slice_with_backend(&data, Backend::AVX2).gelu().unwrap();
            for (i, (&s, &x)) in scalar.as_slice().iter().zip(avx2.as_slice().iter()).enumerate() {
                assert!((s - x).abs() < 1e-2, "Scalar vs AVX2 gelu mismatch at index {}: {} vs {}", i, s, x);
            }
        }
    }
}

#[test]
fn test_swish_backend_equivalence() {
    let data: Vec<f32> = (-10..10).map(|i| i as f32 * 0.5).collect();
    let scalar = Vector::from_slice_with_backend(&data, Backend::Scalar).swish().unwrap();

    #[cfg(target_arch = "x86_64")]
    {
        let sse2 = Vector::from_slice_with_backend(&data, Backend::SSE2).swish().unwrap();
        for (i, (&s, &x)) in scalar.as_slice().iter().zip(sse2.as_slice().iter()).enumerate() {
            assert!((s - x).abs() < 1e-2, "Scalar vs SSE2 swish mismatch at index {}: {} vs {}", i, s, x);
        }

        if is_x86_feature_detected!("avx2") {
            let avx2 = Vector::from_slice_with_backend(&data, Backend::AVX2).swish().unwrap();
            for (i, (&s, &x)) in scalar.as_slice().iter().zip(avx2.as_slice().iter()).enumerate() {
                assert!((s - x).abs() < 1e-2, "Scalar vs AVX2 swish mismatch at index {}: {} vs {}", i, s, x);
            }
        }
    }
}

// ========== Non-Aligned Size Tests (remainder handling in SIMD) ==========

#[test]
fn test_relu_non_aligned_sizes() {
    // Test various non-aligned sizes to exercise SIMD remainder handling
    for size in [1, 3, 5, 7, 9, 13, 15, 17, 31, 33, 63, 65] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
        let v = Vector::from_slice(&data);
        let result = v.relu().unwrap();
        for (i, &val) in result.as_slice().iter().enumerate() {
            let expected = data[i].max(0.0);
            assert!((val - expected).abs() < 1e-6, "relu non-aligned size={} index={}", size, i);
        }
    }
}

#[test]
fn test_sigmoid_non_aligned_sizes() {
    for size in [1, 3, 5, 7, 9, 15, 17] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
        let v = Vector::from_slice(&data);
        let result = v.sigmoid().unwrap();
        for &val in result.as_slice() {
            assert!(val >= 0.0 && val <= 1.0, "sigmoid out of range for size {}", size);
        }
    }
}

#[test]
fn test_gelu_non_aligned_sizes() {
    for size in [1, 3, 5, 7, 9, 15] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
        let v = Vector::from_slice(&data);
        let result = v.gelu().unwrap();
        assert_eq!(result.as_slice().len(), size);
    }
}

#[test]
fn test_swish_non_aligned_sizes() {
    for size in [1, 3, 5, 7, 9, 15] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
        let v = Vector::from_slice(&data);
        let result = v.swish().unwrap();
        assert_eq!(result.as_slice().len(), size);
    }
}

// ========== AVX-512 Backend Dispatch for Activations ==========

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_avx512_backend() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    // 32 elements = 2 full AVX-512 iterations (16 f32s each)
    let data: Vec<f32> = (-16..16).map(|i| i as f32).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::AVX512);
    let result = v.relu().unwrap();
    for (i, &val) in result.as_slice().iter().enumerate() {
        let expected = data[i].max(0.0);
        assert!((val - expected).abs() < 1e-6, "relu AVX512 mismatch at {}: {} vs {}", i, val, expected);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_sigmoid_avx512_backend() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let data: Vec<f32> = (-16..16).map(|i| i as f32 * 0.5).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::AVX512);
    let result = v.sigmoid().unwrap();
    for &val in result.as_slice() {
        assert!(val >= 0.0 && val <= 1.0, "sigmoid AVX512 out of range: {}", val);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_gelu_avx512_backend() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let data: Vec<f32> = (-8..8).map(|i| i as f32 * 0.5).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::AVX512);
    let result = v.gelu().unwrap();
    assert!((result.as_slice()[8] - 0.0).abs() < 1e-5);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_swish_avx512_backend() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    let data: Vec<f32> = (-8..8).map(|i| i as f32 * 0.5).collect();
    let v = Vector::from_slice_with_backend(&data, Backend::AVX512);
    let result = v.swish().unwrap();
    assert!((result.as_slice()[8] - 0.0).abs() < 1e-5);
}

// ========== AVX-512 Non-Aligned Sizes ==========

#[test]
#[cfg(target_arch = "x86_64")]
fn test_relu_avx512_non_aligned() {
    if !is_x86_feature_detected!("avx512f") {
        return;
    }
    for size in [17, 19, 23, 31, 33] {
        let data: Vec<f32> = (0..size).map(|i| (i as f32) - (size as f32 / 2.0)).collect();
        let v = Vector::from_slice_with_backend(&data, Backend::AVX512);
        let result = v.relu().unwrap();
        for (i, &val) in result.as_slice().iter().enumerate() {
            let expected = data[i].max(0.0);
            assert!((val - expected).abs() < 1e-6, "relu AVX512 size={} mismatch at {}", size, i);
        }
    }
}
