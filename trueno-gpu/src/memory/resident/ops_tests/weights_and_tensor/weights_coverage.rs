//! PMAT-018: Additional weights.rs coverage tests
//!
//! Tests for KV cache field access, encoder output shape verification,
//! different n_heads configurations, decoder all-fields validation,
//! conv frontend tensor sizes, config Copy traits, varied input values,
//! and complete field size verification for encoder/decoder weights.

use crate::driver::CudaContext;
use crate::memory::resident::GpuResidentTensor;

/// Helper to create CUDA context, skipping test if unavailable
macro_rules! cuda_ctx {
    () => {
        match CudaContext::new(0) {
            Ok(ctx) => ctx,
            Err(e) => {
                eprintln!("Skipping CUDA test: {:?}", e);
                return;
            }
        }
    };
}

// ============================================================================
// KV Cache Extended Tests
// ============================================================================

#[test]
fn test_gpu_kv_cache_key_and_value_access() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::GpuKvCache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let max_seq_len = 32usize;
    let d_model = 16usize;

    let cache = GpuKvCache::new(&ctx, max_seq_len, d_model).unwrap();

    // Verify key tensor is correctly allocated
    assert_eq!(cache.key.len(), max_seq_len * d_model);
    assert!(cache.key.is_device_resident());

    // Verify value tensor is correctly allocated
    assert_eq!(cache.value.len(), max_seq_len * d_model);
    assert!(cache.value.is_device_resident());

    // Verify fields are accessible
    assert_eq!(cache.max_seq_len, max_seq_len);
    assert_eq!(cache.d_model, d_model);
}

#[test]
fn test_gpu_kv_cache_len_changes() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::GpuKvCache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let mut cache = GpuKvCache::new(&ctx, 64, 32).unwrap();

    // Start empty
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    // Simulate adding tokens
    cache.seq_len = 5;
    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 5);

    // Add more tokens
    cache.seq_len = 20;
    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 20);

    // Reset
    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

// ============================================================================
// Forward Encoder Block Output Shape Tests
// ============================================================================

#[test]
fn test_forward_encoder_block_gpu_verifies_output_shape() {
    use crate::memory::resident::{
        clear_kernel_cache, forward_encoder_block_gpu, GpuEncoderBlockWeights, GpuEncoderConfig,
    };
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let d_model = 32usize;
    let n_heads = 2u32;
    let ffn_dim = 64usize;
    let seq_len = 8usize;

    let config = GpuEncoderConfig {
        d_model: d_model as u32,
        n_heads,
        ffn_dim: ffn_dim as u32,
    };

    let weights = GpuEncoderBlockWeights {
        ln1_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln1_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_q: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        b_q: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_k: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        b_k: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_v: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        b_v: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_o: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        b_o: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ln2_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln2_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ffn_up_w: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * ffn_dim]).unwrap(),
        ffn_up_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; ffn_dim]).unwrap(),
        ffn_down_w: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; ffn_dim * d_model]).unwrap(),
        ffn_down_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
    };

    // Input with different sequence length
    let input_data: Vec<f32> = (0..(seq_len * d_model))
        .map(|i| (i as f32) * 0.01)
        .collect();
    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();

    let output = forward_encoder_block_gpu(&ctx, &input, &weights, &config).unwrap();

    // Verify output shape is correct
    assert_eq!(output.len(), seq_len * d_model);
    assert!(output.is_device_resident());

    // Verify output contains valid values
    let mut output_copy = output;
    let host_output = output_copy.to_host().unwrap();
    assert!(host_output.iter().all(|v| v.is_finite()));
}

#[test]
fn test_forward_encoder_block_gpu_with_different_n_heads() {
    use crate::memory::resident::{
        clear_kernel_cache, forward_encoder_block_gpu, GpuEncoderBlockWeights, GpuEncoderConfig,
    };
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Use 4 heads instead of 2 to exercise different head_dim calculation
    let d_model = 32usize;
    let n_heads = 4u32;
    let ffn_dim = 128usize;
    let seq_len = 4usize;

    let config = GpuEncoderConfig {
        d_model: d_model as u32,
        n_heads,
        ffn_dim: ffn_dim as u32,
    };

    let weights = GpuEncoderBlockWeights {
        ln1_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln1_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_q: GpuResidentTensor::from_host(&ctx, &vec![0.02f32; d_model * d_model]).unwrap(),
        b_q: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_k: GpuResidentTensor::from_host(&ctx, &vec![0.02f32; d_model * d_model]).unwrap(),
        b_k: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_v: GpuResidentTensor::from_host(&ctx, &vec![0.02f32; d_model * d_model]).unwrap(),
        b_v: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_o: GpuResidentTensor::from_host(&ctx, &vec![0.02f32; d_model * d_model]).unwrap(),
        b_o: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ln2_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln2_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ffn_up_w: GpuResidentTensor::from_host(&ctx, &vec![0.02f32; d_model * ffn_dim]).unwrap(),
        ffn_up_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; ffn_dim]).unwrap(),
        ffn_down_w: GpuResidentTensor::from_host(&ctx, &vec![0.02f32; ffn_dim * d_model]).unwrap(),
        ffn_down_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
    };

    let input_data: Vec<f32> = (0..(seq_len * d_model))
        .map(|i| (i as f32) * 0.01)
        .collect();
    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();

    let output = forward_encoder_block_gpu(&ctx, &input, &weights, &config).unwrap();
    assert_eq!(output.len(), seq_len * d_model);
}

// ============================================================================
// Decoder Block Weights All-Fields Validation
// ============================================================================

#[test]
fn test_gpu_decoder_block_weights_all_fields() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::GpuDecoderBlockWeights;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let d_model = 16usize;
    let ffn_dim = 64usize;

    let weights = GpuDecoderBlockWeights {
        // Self-attention
        ln1_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln1_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        self_w_q: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        self_b_q: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        self_w_k: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        self_b_k: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        self_w_v: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        self_b_v: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        self_w_o: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        self_b_o: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        // Cross-attention
        ln2_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln2_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        cross_w_q: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        cross_b_q: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        cross_w_k: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        cross_b_k: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        cross_w_v: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        cross_b_v: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        cross_w_o: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        cross_b_o: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        // FFN
        ln3_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln3_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ffn_up_w: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * ffn_dim]).unwrap(),
        ffn_up_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; ffn_dim]).unwrap(),
        ffn_down_w: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; ffn_dim * d_model]).unwrap(),
        ffn_down_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
    };

    // Verify all self-attention weights
    assert!(weights.self_w_q.is_device_resident());
    assert!(weights.self_b_q.is_device_resident());
    assert!(weights.self_w_k.is_device_resident());
    assert!(weights.self_b_k.is_device_resident());
    assert!(weights.self_w_v.is_device_resident());
    assert!(weights.self_b_v.is_device_resident());
    assert!(weights.self_w_o.is_device_resident());
    assert!(weights.self_b_o.is_device_resident());

    // Verify all cross-attention weights
    assert!(weights.cross_w_q.is_device_resident());
    assert!(weights.cross_b_q.is_device_resident());
    assert!(weights.cross_w_k.is_device_resident());
    assert!(weights.cross_b_k.is_device_resident());
    assert!(weights.cross_w_v.is_device_resident());
    assert!(weights.cross_b_v.is_device_resident());
    assert!(weights.cross_w_o.is_device_resident());
    assert!(weights.cross_b_o.is_device_resident());

    // Verify FFN weights
    assert!(weights.ln3_gamma.is_device_resident());
    assert!(weights.ln3_beta.is_device_resident());
}

// ============================================================================
// Conv Frontend Tensor Size Tests
// ============================================================================

#[test]
fn test_gpu_conv_frontend_weights_tensor_sizes() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::GpuConvFrontendWeights;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Whisper-like configuration: conv1 [384, 80, 3], conv2 [384, 384, 3]
    let in_channels = 80usize;
    let hidden = 384usize;
    let kernel_size = 3usize;

    let weights = GpuConvFrontendWeights {
        conv1_weight: GpuResidentTensor::from_host(
            &ctx,
            &vec![0.01f32; hidden * in_channels * kernel_size],
        )
        .unwrap(),
        conv1_bias: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; hidden]).unwrap(),
        conv2_weight: GpuResidentTensor::from_host(
            &ctx,
            &vec![0.01f32; hidden * hidden * kernel_size],
        )
        .unwrap(),
        conv2_bias: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; hidden]).unwrap(),
    };

    // Verify tensor sizes match expected dimensions
    assert_eq!(
        weights.conv1_weight.len(),
        hidden * in_channels * kernel_size
    );
    assert_eq!(weights.conv1_bias.len(), hidden);
    assert_eq!(weights.conv2_weight.len(), hidden * hidden * kernel_size);
    assert_eq!(weights.conv2_bias.len(), hidden);
}

// ============================================================================
// Config Copy Trait Tests
// ============================================================================

#[test]
fn test_gpu_encoder_config_copy_trait() {
    use crate::memory::resident::GpuEncoderConfig;

    let config1 = GpuEncoderConfig {
        d_model: 256,
        n_heads: 4,
        ffn_dim: 1024,
    };

    // Test Copy trait - should work without move
    let config2 = config1;
    let config3 = config1;

    assert_eq!(config2.d_model, config3.d_model);
    assert_eq!(config2.n_heads, config3.n_heads);
    assert_eq!(config2.ffn_dim, config3.ffn_dim);

    // Original is still usable due to Copy
    assert_eq!(config1.d_model, 256);
}

#[test]
fn test_gpu_decoder_config_copy_trait() {
    use crate::memory::resident::GpuDecoderConfig;

    let config1 = GpuDecoderConfig {
        d_model: 512,
        n_heads: 8,
        ffn_dim: 2048,
        max_seq_len: 1024,
        n_layers: 6,
    };

    // Test Copy trait - should work without move
    let config2 = config1;
    let config3 = config1;

    assert_eq!(config2.d_model, config3.d_model);
    assert_eq!(config2.n_heads, config3.n_heads);
    assert_eq!(config2.ffn_dim, config3.ffn_dim);
    assert_eq!(config2.max_seq_len, config3.max_seq_len);
    assert_eq!(config2.n_layers, config3.n_layers);

    // Original is still usable due to Copy
    assert_eq!(config1.n_layers, 6);
}

// ============================================================================
// Forward Encoder Block with Varied Input
// ============================================================================

#[test]
fn test_forward_encoder_block_with_varied_input_values() {
    use crate::memory::resident::{
        clear_kernel_cache, forward_encoder_block_gpu, GpuEncoderBlockWeights, GpuEncoderConfig,
    };
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let d_model = 16usize;
    let n_heads = 2u32;
    let ffn_dim = 32usize;
    let seq_len = 4usize;

    let config = GpuEncoderConfig {
        d_model: d_model as u32,
        n_heads,
        ffn_dim: ffn_dim as u32,
    };

    // Create weights with varying values (not all uniform)
    let weights = GpuEncoderBlockWeights {
        ln1_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln1_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_q: GpuResidentTensor::from_host(
            &ctx,
            &(0..d_model * d_model)
                .map(|i| (i as f32) * 0.001)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        b_q: GpuResidentTensor::from_host(&ctx, &vec![0.1f32; d_model]).unwrap(),
        w_k: GpuResidentTensor::from_host(
            &ctx,
            &(0..d_model * d_model)
                .map(|i| (i as f32) * 0.001)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        b_k: GpuResidentTensor::from_host(&ctx, &vec![0.1f32; d_model]).unwrap(),
        w_v: GpuResidentTensor::from_host(
            &ctx,
            &(0..d_model * d_model)
                .map(|i| (i as f32) * 0.001)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        b_v: GpuResidentTensor::from_host(&ctx, &vec![0.1f32; d_model]).unwrap(),
        w_o: GpuResidentTensor::from_host(
            &ctx,
            &(0..d_model * d_model)
                .map(|i| (i as f32) * 0.001)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        b_o: GpuResidentTensor::from_host(&ctx, &vec![0.1f32; d_model]).unwrap(),
        ln2_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln2_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ffn_up_w: GpuResidentTensor::from_host(
            &ctx,
            &(0..d_model * ffn_dim)
                .map(|i| (i as f32) * 0.001)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        ffn_up_b: GpuResidentTensor::from_host(&ctx, &vec![0.1f32; ffn_dim]).unwrap(),
        ffn_down_w: GpuResidentTensor::from_host(
            &ctx,
            &(0..ffn_dim * d_model)
                .map(|i| (i as f32) * 0.001)
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        ffn_down_b: GpuResidentTensor::from_host(&ctx, &vec![0.1f32; d_model]).unwrap(),
    };

    // Input with varied values
    let input_data: Vec<f32> = (0..(seq_len * d_model))
        .map(|i| ((i % 10) as f32) * 0.1 - 0.5)
        .collect();
    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();

    let output = forward_encoder_block_gpu(&ctx, &input, &weights, &config).unwrap();
    assert_eq!(output.len(), seq_len * d_model);
}

// ============================================================================
// Encoder Block Weights Field Size Verification
// ============================================================================

#[test]
fn test_gpu_encoder_block_weights_field_sizes() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::GpuEncoderBlockWeights;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let d_model = 32usize;
    let ffn_dim = 64usize;

    let weights = GpuEncoderBlockWeights {
        ln1_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln1_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_q: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        b_q: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_k: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        b_k: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_v: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        b_v: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_o: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        b_o: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ln2_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln2_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ffn_up_w: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * ffn_dim]).unwrap(),
        ffn_up_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; ffn_dim]).unwrap(),
        ffn_down_w: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; ffn_dim * d_model]).unwrap(),
        ffn_down_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
    };

    // Verify all field sizes match expected dimensions
    assert_eq!(weights.ln1_gamma.len(), d_model);
    assert_eq!(weights.ln1_beta.len(), d_model);
    assert_eq!(weights.w_q.len(), d_model * d_model);
    assert_eq!(weights.b_q.len(), d_model);
    assert_eq!(weights.w_k.len(), d_model * d_model);
    assert_eq!(weights.b_k.len(), d_model);
    assert_eq!(weights.w_v.len(), d_model * d_model);
    assert_eq!(weights.b_v.len(), d_model);
    assert_eq!(weights.w_o.len(), d_model * d_model);
    assert_eq!(weights.b_o.len(), d_model);
    assert_eq!(weights.ln2_gamma.len(), d_model);
    assert_eq!(weights.ln2_beta.len(), d_model);
    assert_eq!(weights.ffn_up_w.len(), d_model * ffn_dim);
    assert_eq!(weights.ffn_up_b.len(), ffn_dim);
    assert_eq!(weights.ffn_down_w.len(), ffn_dim * d_model);
    assert_eq!(weights.ffn_down_b.len(), d_model);
}

// ============================================================================
// Decoder Block Weights Field Size Verification
// ============================================================================

#[test]
fn test_gpu_decoder_block_weights_field_sizes() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::GpuDecoderBlockWeights;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let d_model = 16usize;
    let ffn_dim = 32usize;

    let weights = GpuDecoderBlockWeights {
        ln1_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln1_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        self_w_q: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        self_b_q: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        self_w_k: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        self_b_k: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        self_w_v: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        self_b_v: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        self_w_o: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        self_b_o: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ln2_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln2_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        cross_w_q: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        cross_b_q: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        cross_w_k: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        cross_b_k: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        cross_w_v: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        cross_b_v: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        cross_w_o: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * d_model]).unwrap(),
        cross_b_o: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ln3_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln3_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ffn_up_w: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; d_model * ffn_dim]).unwrap(),
        ffn_up_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; ffn_dim]).unwrap(),
        ffn_down_w: GpuResidentTensor::from_host(&ctx, &vec![0.01f32; ffn_dim * d_model]).unwrap(),
        ffn_down_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
    };

    // Verify self-attention field sizes
    assert_eq!(weights.self_w_q.len(), d_model * d_model);
    assert_eq!(weights.self_b_q.len(), d_model);
    assert_eq!(weights.self_w_k.len(), d_model * d_model);
    assert_eq!(weights.self_b_k.len(), d_model);
    assert_eq!(weights.self_w_v.len(), d_model * d_model);
    assert_eq!(weights.self_b_v.len(), d_model);
    assert_eq!(weights.self_w_o.len(), d_model * d_model);
    assert_eq!(weights.self_b_o.len(), d_model);

    // Verify cross-attention field sizes
    assert_eq!(weights.cross_w_q.len(), d_model * d_model);
    assert_eq!(weights.cross_b_q.len(), d_model);
    assert_eq!(weights.cross_w_k.len(), d_model * d_model);
    assert_eq!(weights.cross_b_k.len(), d_model);
    assert_eq!(weights.cross_w_v.len(), d_model * d_model);
    assert_eq!(weights.cross_b_v.len(), d_model);
    assert_eq!(weights.cross_w_o.len(), d_model * d_model);
    assert_eq!(weights.cross_b_o.len(), d_model);

    // Verify FFN field sizes
    assert_eq!(weights.ln3_gamma.len(), d_model);
    assert_eq!(weights.ln3_beta.len(), d_model);
    assert_eq!(weights.ffn_up_w.len(), d_model * ffn_dim);
    assert_eq!(weights.ffn_up_b.len(), ffn_dim);
    assert_eq!(weights.ffn_down_w.len(), ffn_dim * d_model);
    assert_eq!(weights.ffn_down_b.len(), d_model);
}

// ============================================================================
// KV Cache Field Access Test
// ============================================================================

#[test]
fn test_gpu_kv_cache_field_access() {
    use crate::memory::resident::clear_kernel_cache;
    use crate::memory::resident::GpuKvCache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let max_seq_len = 64usize;
    let d_model = 32usize;
    let total_size = max_seq_len * d_model;

    let mut cache = GpuKvCache::new(&ctx, max_seq_len, d_model).unwrap();

    // Verify key and value tensors can be accessed
    assert_eq!(cache.key.len(), total_size);
    assert_eq!(cache.value.len(), total_size);

    // Verify peek_host works on both caches
    let key_data = cache.key.peek_host().unwrap();
    let value_data = cache.value.peek_host().unwrap();
    assert_eq!(key_data.len(), total_size);
    assert_eq!(value_data.len(), total_size);

    // Verify all zeros initially
    assert!(key_data.iter().all(|&v| v == 0.0));
    assert!(value_data.iter().all(|&v| v == 0.0));

    // Test sequence length management
    cache.seq_len = 10;
    assert_eq!(cache.len(), 10);
    assert!(!cache.is_empty());

    cache.reset();
    assert_eq!(cache.len(), 0);
    assert!(cache.is_empty());
}
