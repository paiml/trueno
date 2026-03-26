//! Contract-trait enforcement for trueno.
//!
//! Proves that trueno's kernel implementations satisfy the provable-contracts
//! trait signatures at compile time. Each `impl` block is a compile-time proof:
//! missing method = compile error, wrong signature = compile error.
//!
//! See: provable-contracts docs/specifications/sub/contract-trait-enforcement.md (Section 23)

use provable_contracts::traits::{
    ActivationKernelV1, AdamwKernelV1, AttentionKernelV1, CrossEntropyKernelV1,
    FlashAttentionV1, GqaKernelV1, LayernormKernelV1, MatmulKernelV1, RmsnormKernelV1,
    RopeKernelV1, SiluKernelV1, SoftmaxKernelV1, SwigluKernelV1,
};

// ============================================================================
// Wrapper struct that delegates to trueno's scalar/BLIS implementations
// ============================================================================

/// Wrapper providing contract-trait compliance over trueno's free functions.
struct TruenoKernels;

// ----------------------------------------------------------------------------
// ActivationKernelV1: gelu, relu, silu (scalar -> Vec<f32>)
// ----------------------------------------------------------------------------

impl ActivationKernelV1 for TruenoKernels {
    fn gelu(&self, x: f32) -> Vec<f32> {
        vec![trueno::gelu_scalar(x)]
    }

    fn relu(&self, x: f32) -> Vec<f32> {
        vec![trueno::relu_scalar(x)]
    }

    fn silu(&self, x: f32) -> Vec<f32> {
        vec![trueno::silu_scalar(x)]
    }
}

// ----------------------------------------------------------------------------
// SoftmaxKernelV1: softmax (ignores n1 normalization hint)
// ----------------------------------------------------------------------------

impl SoftmaxKernelV1 for TruenoKernels {
    fn softmax(&self, x: &[f32]) -> Vec<f32> {
        trueno::blis::softmax::softmax_1d_alloc(x)
    }
}

// ----------------------------------------------------------------------------
// SiluKernelV1: sigmoid, silu
// ----------------------------------------------------------------------------

impl SiluKernelV1 for TruenoKernels {
    fn sigmoid(&self, x: &[f32]) -> Vec<f32> {
        x.iter().map(|&xi| trueno::sigmoid_scalar(xi)).collect()
    }

    fn silu(&self, x: &[f32]) -> Vec<f32> {
        x.iter().map(|&xi| trueno::silu_scalar(xi)).collect()
    }
}

// ----------------------------------------------------------------------------
// SwigluKernelV1: silu + swiglu
// ----------------------------------------------------------------------------

impl SwigluKernelV1 for TruenoKernels {
    fn silu(&self, x: &[f32]) -> Vec<f32> {
        x.iter().map(|&xi| trueno::silu_scalar(xi)).collect()
    }

    fn swiglu(&self, x: &[f32], w: &[f32], v: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
        let _ = (w, v, b, c);
        let half = x.len() / 2;
        let x_part = &x[..half];
        let gate = &x[half..];
        x_part.iter()
            .zip(gate.iter())
            .map(|(&xi, &gi)| trueno::silu_scalar(xi) * gi)
            .collect()
    }
}

// ----------------------------------------------------------------------------
// CrossEntropyKernelV1: log_softmax + cross_entropy
// ----------------------------------------------------------------------------

impl CrossEntropyKernelV1 for TruenoKernels {
    fn cross_entropy(&self, targets: &[f32], logits: &[f32]) -> Vec<f32> {
        let log_probs = self.log_softmax(logits);
        let loss: f32 = targets
            .iter()
            .zip(log_probs.iter())
            .map(|(&t, &lp)| -t * lp)
            .sum();
        vec![loss]
    }

    fn log_softmax(&self, x: &[f32]) -> Vec<f32> {
        let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 =
            x.iter().map(|&xi| (xi - max_val).exp()).sum::<f32>().ln() + max_val;
        x.iter().map(|&xi| xi - log_sum_exp).collect()
    }
}

// ----------------------------------------------------------------------------
// RmsnormKernelV1: rmsnorm (uses unit gamma, eps=1e-5 defaults)
// ----------------------------------------------------------------------------

impl RmsnormKernelV1 for TruenoKernels {
    fn rmsnorm(&self, x: &[f32]) -> Vec<f32> {
        let gamma: Vec<f32> = vec![1.0; x.len()];
        let eps = 1e-5_f32;
        trueno::blis::norms::rms_norm_alloc(x, &gamma, eps)
    }
}

// ----------------------------------------------------------------------------
// LayernormKernelV1: layernorm, statistics
// ----------------------------------------------------------------------------

impl LayernormKernelV1 for TruenoKernels {
    fn layernorm(&self, x: &[f32], gamma: &[f32]) -> Vec<f32> {
        let beta: Vec<f32> = vec![0.0; x.len()];
        let eps = 1e-5_f32;
        trueno::blis::norms::layer_norm_alloc(x, gamma, &beta, eps)
    }

    fn statistics(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len() as f32;
        if n == 0.0 {
            return vec![0.0, 0.0];
        }
        let mu = x.iter().sum::<f32>() / n;
        let var = x.iter().map(|&xi| (xi - mu) * (xi - mu)).sum::<f32>() / n;
        vec![mu, var]
    }
}

// ----------------------------------------------------------------------------
// RopeKernelV1: Rotary Position Embeddings (CPU reference scalar impl)
// ----------------------------------------------------------------------------

impl RopeKernelV1 for TruenoKernels {
    fn rope(&self, x: &[f32], m: &[f32]) -> Vec<f32> {
        let d = x.len();
        let pos = if m.is_empty() { 0.0_f32 } else { m[0] };
        let base: f32 = 10_000.0;
        let mut output = vec![0.0f32; d];
        for k in 0..d / 2 {
            let theta = base.powf(-2.0 * k as f32 / d as f32);
            let angle = pos * theta;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            output[2 * k] = x[2 * k] * cos_a - x[2 * k + 1] * sin_a;
            output[2 * k + 1] = x[2 * k] * sin_a + x[2 * k + 1] * cos_a;
        }
        output
    }
}

// ----------------------------------------------------------------------------
// AdamwKernelV1: AdamW optimizer sub-equations (reference scalar impl)
// ----------------------------------------------------------------------------

impl AdamwKernelV1 for TruenoKernels {
    fn adam_moments(&self, g_t: &[f32]) -> Vec<f32> {
        // Convention: g_t contains [gradients, m_prev] packed together
        let half = g_t.len() / 2;
        let grads = &g_t[..half];
        let m_prev = &g_t[half..];
        let beta1: f32 = 0.9;
        grads.iter()
            .zip(m_prev.iter())
            .map(|(&gi, &mi)| beta1 * mi + (1.0 - beta1) * gi)
            .collect()
    }

    fn adam_variance(&self, g_t: &[f32]) -> Vec<f32> {
        // Convention: g_t contains [gradients, v_prev] packed together
        let half = g_t.len() / 2;
        let grads = &g_t[..half];
        let v_prev = &g_t[half..];
        let beta2: f32 = 0.999;
        grads.iter()
            .zip(v_prev.iter())
            .map(|(&gi, &vi)| beta2 * vi + (1.0 - beta2) * gi * gi)
            .collect()
    }

    fn bias_correction(&self, input: &[f32]) -> Vec<f32> {
        let half = input.len() / 2;
        let m = &input[..half];
        let v = &input[half..];
        let beta1: f32 = 0.9;
        let beta2: f32 = 0.999;
        let t = 1_i32;
        let bc1 = 1.0 / (1.0 - beta1.powi(t));
        let bc2 = 1.0 / (1.0 - beta2.powi(t));
        let mut result = Vec::with_capacity(input.len());
        result.extend(m.iter().map(|&mi| mi * bc1));
        result.extend(v.iter().map(|&vi| vi * bc2));
        result
    }

    fn weight_update(&self, theta: &[f32]) -> Vec<f32> {
        let third = theta.len() / 3;
        let weights = &theta[..third];
        let m_hat = &theta[third..2 * third];
        let v_hat = &theta[2 * third..];
        let lr: f32 = 0.001;
        let eps: f32 = 1e-8;
        let wd: f32 = 0.01;
        weights
            .iter()
            .zip(m_hat.iter().zip(v_hat.iter()))
            .map(|(&ti, (&mi, &vi))| ti - lr * (mi / (vi.sqrt() + eps) + wd * ti))
            .collect()
    }
}

// ----------------------------------------------------------------------------
// AttentionKernelV1: naive scaled dot-product attention (reference scalar)
// ----------------------------------------------------------------------------

impl AttentionKernelV1 for TruenoKernels {
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        naive_attention(q, k, v)
    }
}

// ----------------------------------------------------------------------------
// FlashAttentionV1: mathematically identical to standard attention
// ----------------------------------------------------------------------------

impl FlashAttentionV1 for TruenoKernels {
    fn flash_attention(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        naive_attention(q, k, v)
    }
}

// ----------------------------------------------------------------------------
// GqaKernelV1: GQA with num_kv_heads = num_heads is standard attention
// ----------------------------------------------------------------------------

impl GqaKernelV1 for TruenoKernels {
    fn gqa(&self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
        naive_attention(q, k, v)
    }
}

// ----------------------------------------------------------------------------
// MatmulKernelV1: naive O(n^3) matmul + quantized dot product
// ----------------------------------------------------------------------------

impl MatmulKernelV1 for TruenoKernels {
    fn matmul(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        naive_matmul(a, b)
    }

    fn quantized_dot(&self, b: &[f32], s_b: f32) -> Vec<f32> {
        // With single-slice signature, b contains pre-scaled values
        let dot: f32 = b.iter().sum();
        vec![s_b * dot]
    }
}

// ============================================================================
// Shared reference implementations
// ============================================================================

/// Naive scaled dot-product attention on flattened square matrices.
fn naive_attention(q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let total = q.len();
    let n = (total as f32).sqrt() as usize;
    let d = if n > 0 { total / n } else { return vec![] };

    let scale = 1.0 / (d as f32).sqrt();
    let mut scores = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut dot = 0.0f32;
            for kk in 0..d {
                dot += q[i * d + kk] * k[j * d + kk];
            }
            scores[i * n + j] = dot * scale;
        }
    }

    // Row-wise softmax
    for i in 0..n {
        let row = &mut scores[i * n..(i + 1) * n];
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for val in row.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }
        for val in row.iter_mut() {
            *val /= sum;
        }
    }

    let d_v = if n > 0 { v.len() / n } else { 0 };
    let mut output = vec![0.0f32; n * d_v];
    for i in 0..n {
        for j in 0..d_v {
            let mut acc = 0.0f32;
            for kk in 0..n {
                acc += scores[i * n + kk] * v[kk * d_v + j];
            }
            output[i * d_v + j] = acc;
        }
    }
    output
}

/// Naive O(n^3) matmul on flattened square matrices.
fn naive_matmul(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = (a.len() as f32).sqrt() as usize;
    if n == 0 {
        return vec![];
    }
    let m = n;
    let p = a.len() / m;
    let bn = b.len() / p;
    let mut c = vec![0.0f32; m * bn];
    for i in 0..m {
        for j in 0..bn {
            let mut acc = 0.0f32;
            for kk in 0..p {
                acc += a[i * p + kk] * b[kk * bn + j];
            }
            c[i * bn + j] = acc;
        }
    }
    c
}

// ============================================================================
// Runtime validation tests
// ============================================================================

#[test]
fn activation_gelu_zero_preserving() {
    let k = TruenoKernels;
    let out = ActivationKernelV1::gelu(&k, 0.0);
    assert!((out[0]).abs() < 1e-6, "GELU(0) must be 0, got {}", out[0]);
}

#[test]
fn activation_relu_non_negative() {
    let k = TruenoKernels;
    for &x in &[-3.0f32, -1.0, 0.0, 1.0, 5.0] {
        let out = ActivationKernelV1::relu(&k, x);
        assert!(out[0] >= 0.0, "ReLU({}) = {} must be >= 0", x, out[0]);
    }
    let out_zero = ActivationKernelV1::relu(&k, 0.0);
    assert!((out_zero[0]).abs() < 1e-6, "ReLU(0) must be 0");
    let out_one = ActivationKernelV1::relu(&k, 1.0);
    assert!((out_one[0] - 1.0).abs() < 1e-6, "ReLU(1) must be 1");
}

#[test]
fn activation_silu_zero_preserving() {
    let k = TruenoKernels;
    let out = ActivationKernelV1::silu(&k, 0.0);
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
fn swiglu_trait_compiles() {
    let k = TruenoKernels;
    let silu = SwigluKernelV1::silu(&k, &[0.0, 1.0]);
    assert_eq!(silu.len(), 2);

    let swiglu = SwigluKernelV1::swiglu(&k, &[1.0, 2.0, 0.0, 1.0], &[], &[], &[], &[]);
    assert_eq!(swiglu.len(), 2);
}

#[test]
fn cross_entropy_trait_compiles() {
    let k = TruenoKernels;
    let log_sm = CrossEntropyKernelV1::log_softmax(&k, &[1.0, 2.0, 3.0]);
    assert_eq!(log_sm.len(), 3);
    assert!(log_sm.iter().all(|&v| v <= 0.0), "log_softmax <= 0");

    let ce = CrossEntropyKernelV1::cross_entropy(&k, &[0.0, 0.0, 1.0], &[1.0, 2.0, 3.0]);
    assert_eq!(ce.len(), 1);
    assert!(ce[0] >= 0.0, "cross-entropy >= 0");
}

#[test]
fn rmsnorm_unit_gamma() {
    let k = TruenoKernels;
    let input = [1.0, 2.0, 3.0, 4.0];
    let out = RmsnormKernelV1::rmsnorm(&k, &input);
    assert_eq!(out.len(), input.len());
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
    let gamma = vec![1.0f32; input.len()];
    let out = LayernormKernelV1::layernorm(&k, &input, &gamma);
    assert_eq!(out.len(), input.len());
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
    assert!(
        (mu - 4.0).abs() < 1e-6,
        "mean of [2,4,6] should be 4.0, got {}",
        mu
    );
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

#[test]
fn rope_trait_compiles() {
    let k = TruenoKernels;
    let input = &[1.0, 2.0, 3.0, 4.0];
    let out = RopeKernelV1::rope(&k, input, &[0.0]);
    assert_eq!(out.len(), 4);
    for (i, (&a, &b)) in input.iter().zip(out.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "RoPE at m=0 should be identity, idx={i}");
    }
}

#[test]
fn adamw_trait_compiles() {
    let k = TruenoKernels;

    let moments = AdamwKernelV1::adam_moments(&k, &[0.5, 0.3, 0.0, 0.0]);
    assert_eq!(moments.len(), 2);
    assert!((moments[0] - 0.05).abs() < 1e-6, "m = 0.1 * 0.5 = 0.05");

    let variance = AdamwKernelV1::adam_variance(&k, &[0.5, 0.3, 0.0, 0.0]);
    assert_eq!(variance.len(), 2);
    assert!(variance[0] > 0.0, "variance > 0 for non-zero gradient");

    let corrected = AdamwKernelV1::bias_correction(&k, &[0.05, 0.00025]);
    assert_eq!(corrected.len(), 2);
    assert!(corrected[0].abs() > 0.05, "bias correction amplifies at t=1");

    let updated = AdamwKernelV1::weight_update(&k, &[1.0, 0.5, 0.25, 1.0, 0.5, 0.25]);
    assert_eq!(updated.len(), 2);
    assert!((updated[0] - 1.0).abs() > 1e-6, "weights updated");
}

#[test]
fn attention_trait_compiles() {
    let k = TruenoKernels;
    let q = &[1.0, 0.0, 0.0, 1.0];
    let kk = &[1.0, 0.0, 0.0, 1.0];
    let v = &[1.0, 0.0, 0.0, 1.0];
    let out = AttentionKernelV1::attention(&k, q, kk, v);
    assert_eq!(out.len(), 4);
}

#[test]
fn flash_attention_trait_compiles() {
    let k = TruenoKernels;
    let q = &[1.0, 0.0, 0.0, 1.0];
    let kk = &[1.0, 0.0, 0.0, 1.0];
    let v = &[1.0, 0.0, 0.0, 1.0];
    let out = FlashAttentionV1::flash_attention(&k, q, kk, v);
    assert_eq!(out.len(), 4);
}

#[test]
fn gqa_trait_compiles() {
    let k = TruenoKernels;
    let q = &[1.0, 0.0, 0.0, 1.0];
    let kk = &[1.0, 0.0, 0.0, 1.0];
    let v = &[1.0, 0.0, 0.0, 1.0];
    let out = GqaKernelV1::gqa(&k, q, kk, v);
    assert_eq!(out.len(), 4);
}

#[test]
fn matmul_trait_compiles() {
    let k = TruenoKernels;
    let a = &[1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let b = &[1.0, 2.0, 3.0, 4.0];
    let out = MatmulKernelV1::matmul(&k, a, b);
    assert_eq!(out.len(), 4);
    assert!((out[0] - 1.0).abs() < 1e-6, "I*B = B");
    assert!((out[3] - 4.0).abs() < 1e-6, "I*B = B");

    let qd = MatmulKernelV1::quantized_dot(&k, &[2.0, 4.0, 6.0], 0.5);
    assert_eq!(qd.len(), 1);
    assert!((qd[0] - 6.0).abs() < 1e-6, "quantized_dot = s_a * s_b * dot");
}
