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
