//! Basic batched multihead attention and optimized variant tests.

use crate::driver::{CudaContext, CudaStream};
use crate::memory::resident::{reset_transfer_counters, GpuResidentTensor};


#[test]
fn test_batched_multihead_attention_basic() {
    use crate::memory::resident::batched_multihead_attention;
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    reset_transfer_counters();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 3u32;
    let d_model = (n_heads * head_dim) as usize; // 8

    // Q, K, V: [seq_len * d_model] = [3 * 8] = 24 elements
    let q_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 + 0.01).collect();
    let v_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1 + 0.02).collect();

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let output = batched_multihead_attention(&ctx, &q, &k, &v, n_heads, head_dim, seq_len).unwrap();
    assert_eq!(output.len(), seq_len as usize * d_model);
    assert!(output.is_device_resident());
}

#[test]
fn test_batched_multihead_attention_dimension_error() {
    use crate::memory::resident::batched_multihead_attention;
    let ctx = cuda_ctx!();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 3u32;

    // Wrong size Q (should be 24, giving 12)
    let q_data: Vec<f32> = vec![1.0; 12];
    let k_data: Vec<f32> = vec![1.0; 24];
    let v_data: Vec<f32> = vec![1.0; 24];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let result = batched_multihead_attention(&ctx, &q, &k, &v, n_heads, head_dim, seq_len);
    assert!(result.is_err());
}

#[test]
fn test_batched_multihead_attention_optimized() {
    use crate::memory::resident::batched_multihead_attention_optimized;
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    reset_transfer_counters();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 4u32;
    let d_model = (n_heads * head_dim) as usize;

    let q_data: Vec<f32> = (0..(seq_len as usize * d_model))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..(seq_len as usize * d_model))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..(seq_len as usize * d_model))
        .map(|i| (i as f32) * 0.1)
        .collect();

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let output =
        batched_multihead_attention_optimized(&ctx, &q, &k, &v, n_heads, head_dim, seq_len)
            .unwrap();
    assert_eq!(output.len(), seq_len as usize * d_model);
}

#[test]
fn test_batched_multihead_attention_with_debug() {
    use crate::memory::resident::batched_multihead_attention;
    use crate::memory::resident::clear_kernel_cache;
    // Clear cache because kernel modules are tied to CUDA contexts
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Enable debug mode for attention
    std::env::set_var("WHISPER_DEBUG_ATTN", "1");

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 3u32;
    let d_model = (n_heads * head_dim) as usize;

    let q_data: Vec<f32> = (0..(seq_len as usize * d_model))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..(seq_len as usize * d_model))
        .map(|i| (i as f32) * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..(seq_len as usize * d_model))
        .map(|i| (i as f32) * 0.1)
        .collect();

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let output = batched_multihead_attention(&ctx, &q, &k, &v, n_heads, head_dim, seq_len).unwrap();
    assert_eq!(output.len(), seq_len as usize * d_model);

    std::env::remove_var("WHISPER_DEBUG_ATTN");
}

#[test]
fn test_batched_multihead_attention_k_v_mismatch() {
    use crate::memory::resident::batched_multihead_attention;
    let ctx = cuda_ctx!();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 3u32;
    let d_model = (n_heads * head_dim) as usize;

    let q_data: Vec<f32> = vec![1.0; seq_len as usize * d_model];
    // K has wrong size
    let k_data: Vec<f32> = vec![1.0; 12]; // Wrong!
    let v_data: Vec<f32> = vec![1.0; seq_len as usize * d_model];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let result = batched_multihead_attention(&ctx, &q, &k, &v, n_heads, head_dim, seq_len);
    assert!(result.is_err());
}

#[test]
fn test_batched_multihead_attention_optimized_size_error() {
    use crate::memory::resident::batched_multihead_attention_optimized;
    let ctx = cuda_ctx!();

    let n_heads = 2u32;
    let head_dim = 4u32;
    let seq_len = 4u32;
    let d_model = (n_heads * head_dim) as usize;

    // Q has wrong size
    let q_data: Vec<f32> = vec![1.0; 10]; // Wrong!
    let k_data: Vec<f32> = vec![1.0; seq_len as usize * d_model];
    let v_data: Vec<f32> = vec![1.0; seq_len as usize * d_model];

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let result =
        batched_multihead_attention_optimized(&ctx, &q, &k, &v, n_heads, head_dim, seq_len);
    assert!(result.is_err());
}

#[test]
fn test_batched_multihead_attention_larger_heads() {
    use crate::memory::resident::batched_multihead_attention;
    let ctx = cuda_ctx!();

    // Test with more heads to exercise more loop iterations
    let n_heads = 4u32;
    let head_dim = 8u32;
    let seq_len = 4u32;
    let d_model = (n_heads * head_dim) as usize;

    let q_data: Vec<f32> = (0..(seq_len as usize * d_model))
        .map(|i| ((i % 10) as f32) * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..(seq_len as usize * d_model))
        .map(|i| ((i % 7) as f32) * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..(seq_len as usize * d_model))
        .map(|i| ((i % 5) as f32) * 0.1)
        .collect();

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let output = batched_multihead_attention(&ctx, &q, &k, &v, n_heads, head_dim, seq_len).unwrap();
    assert_eq!(output.len(), seq_len as usize * d_model);
}

#[test]
fn test_batched_multihead_attention_optimized_larger() {
    use crate::memory::resident::batched_multihead_attention_optimized;
    let ctx = cuda_ctx!();

    let n_heads = 4u32;
    let head_dim = 16u32;
    let seq_len = 8u32;
    let d_model = (n_heads * head_dim) as usize;

    let q_data: Vec<f32> = (0..(seq_len as usize * d_model))
        .map(|i| ((i % 10) as f32) * 0.01)
        .collect();
    let k_data: Vec<f32> = (0..(seq_len as usize * d_model))
        .map(|i| ((i % 10) as f32) * 0.01)
        .collect();
    let v_data: Vec<f32> = (0..(seq_len as usize * d_model))
        .map(|i| ((i % 10) as f32) * 0.01)
        .collect();

    let q = GpuResidentTensor::from_host(&ctx, &q_data).unwrap();
    let k = GpuResidentTensor::from_host(&ctx, &k_data).unwrap();
    let v = GpuResidentTensor::from_host(&ctx, &v_data).unwrap();

    let output =
        batched_multihead_attention_optimized(&ctx, &q, &k, &v, n_heads, head_dim, seq_len)
            .unwrap();
    assert_eq!(output.len(), seq_len as usize * d_model);
}
