//! Forward Encoder Block tests: output shape, n_heads, varied input, field sizes

use super::*;

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
