//! Contract-trait enforcement for trueno.
//!
//! Proves that trueno's kernel implementations satisfy the provable-contracts
//! trait signatures at compile time. Each `impl` block is a compile-time proof:
//! missing method = compile error, wrong signature = compile error.
//!
//! See: provable-contracts docs/specifications/sub/contract-trait-enforcement.md (Section 23)

use provable_contracts::traits::{
    ActivationKernelV1, LayernormKernelV1, RmsnormKernelV1, SiluKernelV1, SoftmaxKernelV1,
};

// ============================================================================
// Wrapper struct that delegates to trueno's scalar/BLIS implementations
// ============================================================================

/// Wrapper providing contract-trait compliance over trueno's free functions.
struct TruenoKernels;

// ----------------------------------------------------------------------------
// ActivationKernelV1: gelu, relu, silu
// ----------------------------------------------------------------------------

impl ActivationKernelV1 for TruenoKernels {
    fn gelu(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| trueno::gelu_scalar(x)).collect()
    }

    fn relu(&self, input: &[f32]) -> Vec<f32> {
        trueno::blis::elementwise::relu_alloc(input)
    }

    fn silu(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| trueno::silu_scalar(x)).collect()
    }
}

// ----------------------------------------------------------------------------
// SoftmaxKernelV1: softmax
// ----------------------------------------------------------------------------

impl SoftmaxKernelV1 for TruenoKernels {
    fn softmax(&self, input: &[f32]) -> Vec<f32> {
        trueno::blis::softmax::softmax_1d_alloc(input)
    }
}

// ----------------------------------------------------------------------------
// SiluKernelV1: sigmoid, silu
// ----------------------------------------------------------------------------

impl SiluKernelV1 for TruenoKernels {
    fn sigmoid(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| trueno::sigmoid_scalar(x)).collect()
    }

    fn silu(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| trueno::silu_scalar(x)).collect()
    }
}

// ----------------------------------------------------------------------------
// RmsnormKernelV1: rmsnorm (uses unit gamma, eps=1e-5 defaults)
// ----------------------------------------------------------------------------

impl RmsnormKernelV1 for TruenoKernels {
    fn rmsnorm(&self, input: &[f32]) -> Vec<f32> {
        let gamma: Vec<f32> = vec![1.0; input.len()];
        let eps = 1e-5_f32;
        trueno::blis::norms::rms_norm_alloc(input, &gamma, eps)
    }
}

// ----------------------------------------------------------------------------
// LayernormKernelV1: layernorm, statistics
// ----------------------------------------------------------------------------

impl LayernormKernelV1 for TruenoKernels {
    fn layernorm(&self, input: &[f32]) -> Vec<f32> {
        let gamma: Vec<f32> = vec![1.0; input.len()];
        let beta: Vec<f32> = vec![0.0; input.len()];
        let eps = 1e-5_f32;
        trueno::blis::norms::layer_norm_alloc(input, &gamma, &beta, eps)
    }

    fn statistics(&self, input: &[f32]) -> Vec<f32> {
        let n = input.len() as f32;
        if n == 0.0 {
            return vec![0.0, 0.0];
        }
        let mu = input.iter().sum::<f32>() / n;
        let var = input.iter().map(|&x| (x - mu) * (x - mu)).sum::<f32>() / n;
        vec![mu, var]
    }
}

// ============================================================================
// Runtime validation tests
// ============================================================================

#[test]
fn activation_gelu_zero_preserving() {
    let k = TruenoKernels;
    let out = ActivationKernelV1::gelu(&k, &[0.0]);
    assert!((out[0]).abs() < 1e-6, "GELU(0) must be 0, got {}", out[0]);
}

#[test]
fn activation_relu_non_negative() {
    let k = TruenoKernels;
    let input = [-3.0, -1.0, 0.0, 1.0, 5.0];
    let out = ActivationKernelV1::relu(&k, &input);
    for (i, &v) in out.iter().enumerate() {
        assert!(v >= 0.0, "ReLU output[{}] = {} must be >= 0", i, v);
    }
    assert!((out[2]).abs() < 1e-6, "ReLU(0) must be 0");
    assert!((out[3] - 1.0).abs() < 1e-6, "ReLU(1) must be 1");
}

#[test]
fn activation_silu_zero_preserving() {
    let k = TruenoKernels;
    let out = ActivationKernelV1::silu(&k, &[0.0]);
    assert!((out[0]).abs() < 1e-6, "SiLU(0) must be 0, got {}", out[0]);
}

#[test]
fn softmax_sums_to_one() {
    let k = TruenoKernels;
    let input = [1.0, 2.0, 3.0, 4.0];
    let out = SoftmaxKernelV1::softmax(&k, &input);
    let sum: f32 = out.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax must sum to 1.0, got {}",
        sum
    );
    for (i, &v) in out.iter().enumerate() {
        assert!(v > 0.0, "softmax[{}] = {} must be > 0", i, v);
    }
}

#[test]
fn softmax_order_preservation() {
    let k = TruenoKernels;
    let input = [1.0, 3.0, 2.0];
    let out = SoftmaxKernelV1::softmax(&k, &input);
    // argmax(input) = 1, argmax(output) should also be 1
    let argmax_in = input
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let argmax_out = out
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(argmax_in, argmax_out, "softmax must preserve argmax");
}

#[test]
fn silu_sigmoid_at_zero() {
    let k = TruenoKernels;
    let sig = SiluKernelV1::sigmoid(&k, &[0.0]);
    assert!(
        (sig[0] - 0.5).abs() < 1e-6,
        "sigmoid(0) must be 0.5, got {}",
        sig[0]
    );
    let silu = SiluKernelV1::silu(&k, &[0.0]);
    assert!(
        (silu[0]).abs() < 1e-6,
        "SiLU(0) must be 0, got {}",
        silu[0]
    );
}

#[test]
fn silu_sigmoid_symmetry() {
    let k = TruenoKernels;
    let pos = SiluKernelV1::sigmoid(&k, &[2.0])[0];
    let neg = SiluKernelV1::sigmoid(&k, &[-2.0])[0];
    assert!(
        (pos + neg - 1.0).abs() < 1e-5,
        "sigmoid(-x) + sigmoid(x) must equal 1.0, got {} + {} = {}",
        neg,
        pos,
        neg + pos
    );
}

#[test]
fn rmsnorm_unit_gamma() {
    let k = TruenoKernels;
    let input = [1.0, 2.0, 3.0, 4.0];
    let out = RmsnormKernelV1::rmsnorm(&k, &input);
    assert_eq!(out.len(), input.len());
    // With unit gamma, the output RMS should be approximately 1.0
    let rms: f32 = (out.iter().map(|x| x * x).sum::<f32>() / out.len() as f32).sqrt();
    assert!(
        (rms - 1.0).abs() < 0.01,
        "RMSNorm with unit gamma should have output RMS ~ 1.0, got {}",
        rms
    );
}

#[test]
fn layernorm_standardization() {
    let k = TruenoKernels;
    let input = [1.0, 2.0, 3.0, 4.0, 5.0];
    let out = LayernormKernelV1::layernorm(&k, &input);
    assert_eq!(out.len(), input.len());
    // With gamma=1, beta=0: mean should be ~0, variance should be ~1
    let n = out.len() as f32;
    let mean = out.iter().sum::<f32>() / n;
    let var = out.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    assert!(
        mean.abs() < 1e-4,
        "LayerNorm mean should be ~0, got {}",
        mean
    );
    assert!(
        (var - 1.0).abs() < 0.05,
        "LayerNorm variance should be ~1, got {}",
        var
    );
}

#[test]
fn layernorm_statistics_basic() {
    let k = TruenoKernels;
    let input = [2.0, 4.0, 6.0];
    let stats = LayernormKernelV1::statistics(&k, &input);
    assert_eq!(stats.len(), 2, "statistics must return [mu, var]");
    let mu = stats[0];
    let var = stats[1];
    // mean of [2,4,6] = 4
    assert!(
        (mu - 4.0).abs() < 1e-6,
        "mean of [2,4,6] should be 4.0, got {}",
        mu
    );
    // variance = ((2-4)^2 + (4-4)^2 + (6-4)^2) / 3 = 8/3 ≈ 2.6667
    let expected_var = 8.0 / 3.0;
    assert!(
        (var - expected_var).abs() < 1e-4,
        "variance of [2,4,6] should be {}, got {}",
        expected_var,
        var
    );
}

#[test]
fn layernorm_statistics_constant_input() {
    let k = TruenoKernels;
    let input = [5.0, 5.0, 5.0, 5.0];
    let stats = LayernormKernelV1::statistics(&k, &input);
    let var = stats[1];
    assert!(
        var.abs() < 1e-6,
        "variance of constant input must be 0, got {}",
        var
    );
}
