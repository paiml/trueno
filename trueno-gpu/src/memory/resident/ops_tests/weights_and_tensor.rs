//! GPU config, weights structures, encoder block, KV cache, GpuResidentTensor methods,
//! TransferStats, and kernel cache stats tests

use crate::driver::{CudaContext, CudaStream};
use crate::memory::resident::{reset_transfer_counters, GpuResidentTensor};

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
// CUDA Weights Tests (PMAT-018: weights.rs coverage)
// ============================================================================

#[test]
fn test_gpu_encoder_config_creation() {
    use crate::memory::resident::GpuEncoderConfig;

    let config = GpuEncoderConfig {
        d_model: 256,
        n_heads: 4,
        ffn_dim: 1024,
    };
    assert_eq!(config.d_model, 256);
    assert_eq!(config.n_heads, 4);
    assert_eq!(config.ffn_dim, 1024);
}

#[test]
fn test_gpu_decoder_config_creation() {
    use crate::memory::resident::GpuDecoderConfig;

    let config = GpuDecoderConfig {
        d_model: 512,
        n_heads: 8,
        ffn_dim: 2048,
        max_seq_len: 1024,
        n_layers: 6,
    };
    assert_eq!(config.d_model, 512);
    assert_eq!(config.n_heads, 8);
    assert_eq!(config.ffn_dim, 2048);
    assert_eq!(config.max_seq_len, 1024);
    assert_eq!(config.n_layers, 6);
}

#[test]
fn test_gpu_kv_cache_creation() {
    use crate::memory::resident::GpuKvCache;
    let ctx = cuda_ctx!();

    let d_model = 256usize;
    let max_seq_len = 512usize;
    let cache_size = max_seq_len * d_model;

    let key = GpuResidentTensor::from_host(&ctx, &vec![0.0f32; cache_size]).unwrap();
    let value = GpuResidentTensor::from_host(&ctx, &vec![0.0f32; cache_size]).unwrap();

    let kv_cache = GpuKvCache {
        key,
        value,
        seq_len: 0,
        max_seq_len,
        d_model,
    };
    assert_eq!(kv_cache.seq_len, 0);
    assert_eq!(kv_cache.max_seq_len, max_seq_len);
    assert_eq!(kv_cache.d_model, d_model);
}

#[test]
fn test_gpu_encoder_block_weights_structure() {
    use crate::memory::resident::GpuEncoderBlockWeights;
    let ctx = cuda_ctx!();

    let d_model = 64usize;
    let ffn_dim = 256usize;

    // Create minimal encoder block weights
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

    // Verify all tensors are device resident
    assert!(weights.ln1_gamma.is_device_resident());
    assert!(weights.w_q.is_device_resident());
    assert!(weights.ffn_up_w.is_device_resident());
}

#[test]
fn test_forward_encoder_block_gpu() {
    use crate::memory::resident::{
        clear_kernel_cache, forward_encoder_block_gpu, GpuEncoderBlockWeights, GpuEncoderConfig,
    };
    // Clear cache because kernel modules are tied to CUDA contexts
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let d_model = 32usize; // Small for testing
    let n_heads = 2u32;
    let ffn_dim = 128usize;
    let seq_len = 4usize;

    let config = GpuEncoderConfig {
        d_model: d_model as u32,
        n_heads,
        ffn_dim: ffn_dim as u32,
    };

    // Create weights
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

    // Input: [seq_len * d_model]
    let input_data: Vec<f32> = (0..(seq_len * d_model))
        .map(|i| (i as f32) * 0.01)
        .collect();
    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();

    let output = forward_encoder_block_gpu(&ctx, &input, &weights, &config).unwrap();
    assert_eq!(output.len(), seq_len * d_model);
    assert!(output.is_device_resident());
}

#[test]
fn test_gpu_kv_cache_new_and_methods() {
    use crate::memory::resident::GpuKvCache;
    let ctx = cuda_ctx!();

    let max_seq_len = 64usize;
    let d_model = 32usize;

    // Test GpuKvCache::new
    let mut cache = GpuKvCache::new(&ctx, max_seq_len, d_model).unwrap();

    // Test is_empty and len on new cache
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);

    // Manually set seq_len to simulate token addition
    cache.seq_len = 10;
    assert!(!cache.is_empty());
    assert_eq!(cache.len(), 10);

    // Test reset
    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_gpu_conv_frontend_weights_structure() {
    use crate::memory::resident::GpuConvFrontendWeights;
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

    assert!(weights.conv1_weight.is_device_resident());
    assert!(weights.conv1_bias.is_device_resident());
    assert!(weights.conv2_weight.is_device_resident());
    assert!(weights.conv2_bias.is_device_resident());
    assert_eq!(
        weights.conv1_weight.len(),
        hidden * in_channels * kernel_size
    );
    assert_eq!(weights.conv2_weight.len(), hidden * hidden * kernel_size);
}

#[test]
fn test_gpu_decoder_block_weights_structure() {
    use crate::memory::resident::GpuDecoderBlockWeights;
    let ctx = cuda_ctx!();

    let d_model = 32usize;
    let ffn_dim = 128usize;

    // Create decoder block weights (self-attention + cross-attention + FFN)
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

    // Verify all tensors are device resident
    assert!(weights.ln1_gamma.is_device_resident());
    assert!(weights.self_w_q.is_device_resident());
    assert!(weights.cross_w_q.is_device_resident());
    assert!(weights.ffn_up_w.is_device_resident());
}

#[test]
fn test_forward_encoder_block_with_debug() {
    use crate::memory::resident::{
        clear_kernel_cache, forward_encoder_block_gpu, GpuEncoderBlockWeights, GpuEncoderConfig,
    };
    // Clear cache because kernel modules are tied to CUDA contexts
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Enable debug mode
    std::env::set_var("WHISPER_DEBUG_GPU_INTERNALS", "1");

    let d_model = 16usize;
    let n_heads = 2u32;
    let ffn_dim = 64usize;
    let seq_len = 2usize;

    let config = GpuEncoderConfig {
        d_model: d_model as u32,
        n_heads,
        ffn_dim: ffn_dim as u32,
    };

    // Create weights
    let weights = GpuEncoderBlockWeights {
        ln1_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln1_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_q: GpuResidentTensor::from_host(&ctx, &vec![0.05f32; d_model * d_model]).unwrap(),
        b_q: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_k: GpuResidentTensor::from_host(&ctx, &vec![0.05f32; d_model * d_model]).unwrap(),
        b_k: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_v: GpuResidentTensor::from_host(&ctx, &vec![0.05f32; d_model * d_model]).unwrap(),
        b_v: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        w_o: GpuResidentTensor::from_host(&ctx, &vec![0.05f32; d_model * d_model]).unwrap(),
        b_o: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ln2_gamma: GpuResidentTensor::from_host(&ctx, &vec![1.0f32; d_model]).unwrap(),
        ln2_beta: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
        ffn_up_w: GpuResidentTensor::from_host(&ctx, &vec![0.05f32; d_model * ffn_dim]).unwrap(),
        ffn_up_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; ffn_dim]).unwrap(),
        ffn_down_w: GpuResidentTensor::from_host(&ctx, &vec![0.05f32; ffn_dim * d_model]).unwrap(),
        ffn_down_b: GpuResidentTensor::from_host(&ctx, &vec![0.0f32; d_model]).unwrap(),
    };

    // Input
    let input_data: Vec<f32> = (0..(seq_len * d_model)).map(|i| (i as f32) * 0.1).collect();
    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();

    let output = forward_encoder_block_gpu(&ctx, &input, &weights, &config).unwrap();
    assert_eq!(output.len(), seq_len * d_model);

    // Clean up env var
    std::env::remove_var("WHISPER_DEBUG_GPU_INTERNALS");
}

#[test]
fn test_gpu_encoder_config_clone_and_debug() {
    use crate::memory::resident::GpuEncoderConfig;

    let config = GpuEncoderConfig {
        d_model: 512,
        n_heads: 8,
        ffn_dim: 2048,
    };

    // Test Clone
    let cloned = config;
    assert_eq!(cloned.d_model, 512);
    assert_eq!(cloned.n_heads, 8);
    assert_eq!(cloned.ffn_dim, 2048);

    // Test Debug
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("GpuEncoderConfig"));
    assert!(debug_str.contains("512"));
}

#[test]
fn test_gpu_decoder_config_clone_and_debug() {
    use crate::memory::resident::GpuDecoderConfig;

    let config = GpuDecoderConfig {
        d_model: 768,
        n_heads: 12,
        ffn_dim: 3072,
        max_seq_len: 1024,
        n_layers: 12,
    };

    // Test Clone (Copy trait)
    let cloned = config;
    assert_eq!(cloned.d_model, 768);
    assert_eq!(cloned.n_heads, 12);
    assert_eq!(cloned.ffn_dim, 3072);
    assert_eq!(cloned.max_seq_len, 1024);
    assert_eq!(cloned.n_layers, 12);

    // Test Debug
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("GpuDecoderConfig"));
    assert!(debug_str.contains("768"));
}

// ============================================================================
// PMAT-018: mod.rs GpuResidentTensor method coverage tests
// ============================================================================

#[test]
fn test_gpu_resident_tensor_transfer_aliases() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    reset_transfer_counters();

    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    // Test alias methods: host_to_device_transfers() == h2d_transfers()
    assert_eq!(tensor.host_to_device_transfers(), tensor.h2d_transfers());
    assert_eq!(tensor.host_to_device_transfers(), 1);

    // Initially no D2H transfers
    assert_eq!(tensor.device_to_host_transfers(), tensor.d2h_transfers());
    assert_eq!(tensor.device_to_host_transfers(), 0);

    // After to_host(), D2H counter increments
    let _ = tensor.to_host().unwrap();
    assert_eq!(tensor.device_to_host_transfers(), 1);
    assert_eq!(tensor.device_to_host_transfers(), tensor.d2h_transfers());
}

#[test]
fn test_gpu_resident_tensor_record_kernel_launch() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    // Initial kernel launches should be 0
    assert_eq!(tensor.kernel_launches(), 0);

    // Call record_kernel_launch() multiple times
    tensor.record_kernel_launch();
    assert_eq!(tensor.kernel_launches(), 1);

    tensor.record_kernel_launch();
    tensor.record_kernel_launch();
    assert_eq!(tensor.kernel_launches(), 3);
}

#[test]
fn test_gpu_resident_tensor_is_empty() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Non-empty tensor
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    assert!(!tensor.is_empty());
    assert_eq!(tensor.len(), 4);

    // Empty tensor via new_uninit with 0 length
    let empty_tensor: GpuResidentTensor<f32> = GpuResidentTensor::new_uninit(&ctx, 0).unwrap();
    assert!(empty_tensor.is_empty());
    assert_eq!(empty_tensor.len(), 0);
}

#[test]
fn test_gpu_resident_tensor_size_bytes() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let data = vec![1.0f32; 100];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    // 100 f32s = 100 * 4 = 400 bytes
    assert_eq!(tensor.size_bytes(), 400);
}

#[test]
fn test_gpu_resident_tensor_as_ptr() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let data = vec![1.0f32; 16];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    // as_ptr() should return non-zero device pointer
    let ptr = tensor.as_ptr();
    assert!(ptr != 0, "Device pointer should be non-zero");
}

#[test]
fn test_gpu_resident_tensor_is_device_resident() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let data = vec![1.0f32; 8];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    // Tensor should be device resident
    assert!(tensor.is_device_resident());
}

#[test]
fn test_gpu_resident_tensor_buffer_and_buffer_mut() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    // Test buffer() immutable reference
    {
        let buf = tensor.buffer();
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.size_bytes(), 16);
    }

    // Test buffer_mut() mutable reference
    {
        let buf_mut = tensor.buffer_mut();
        assert_eq!(buf_mut.len(), 4);
    }
}

#[test]
fn test_gpu_resident_tensor_from_buffer_internal() {
    use crate::driver::GpuBuffer;
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Create a buffer directly
    let buf = GpuBuffer::<f32>::new(&ctx, 32).unwrap();

    // Create tensor from buffer (internal API used by operations)
    let tensor = GpuResidentTensor::from_buffer_internal(buf, 5);

    // Verify initial state
    assert_eq!(tensor.len(), 32);
    assert_eq!(tensor.h2d_transfers(), 0); // No H2D since created from buffer
    assert_eq!(tensor.d2h_transfers(), 0);
    assert_eq!(tensor.kernel_launches(), 5);
    assert!(tensor.is_device_resident());
}

#[test]
fn test_gpu_resident_tensor_peek_vs_to_host() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();
    reset_transfer_counters();

    let data = vec![42.0f32; 16];
    let mut tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    // peek_host() should NOT increment counters
    let before_d2h = tensor.d2h_transfers();
    let peeked = tensor.peek_host().unwrap();
    assert_eq!(peeked, data);
    assert_eq!(tensor.d2h_transfers(), before_d2h); // Counter unchanged

    // to_host() SHOULD increment counters
    let result = tensor.to_host().unwrap();
    assert_eq!(result, data);
    assert_eq!(tensor.d2h_transfers(), before_d2h + 1); // Counter incremented
}

#[test]
fn test_gpu_resident_tensor_new_uninit_various_sizes() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Test various sizes
    for size in [0, 1, 16, 256, 1024, 4096] {
        let tensor: GpuResidentTensor<f32> = GpuResidentTensor::new_uninit(&ctx, size).unwrap();
        assert_eq!(tensor.len(), size);
        assert_eq!(tensor.h2d_transfers(), 0); // No transfer for uninit
        assert_eq!(tensor.d2h_transfers(), 0);
        assert!(tensor.is_device_resident());
    }
}

#[test]
fn test_transfer_stats_default() {
    use crate::memory::resident::TransferStats;

    let stats = TransferStats::default();
    assert_eq!(stats.h2d_transfers, 0);
    assert_eq!(stats.d2h_transfers, 0);
    assert_eq!(stats.h2d_bytes, 0);
    assert_eq!(stats.d2h_bytes, 0);
    assert_eq!(stats.total_transfers(), 0);
    assert_eq!(stats.total_bytes(), 0);
}

#[test]
fn test_transfer_stats_clone() {
    use crate::memory::resident::TransferStats;

    let stats = TransferStats {
        h2d_transfers: 10,
        d2h_transfers: 5,
        h2d_bytes: 1000,
        d2h_bytes: 500,
    };

    let cloned = stats.clone();
    assert_eq!(cloned.h2d_transfers, 10);
    assert_eq!(cloned.d2h_transfers, 5);
    assert_eq!(cloned.h2d_bytes, 1000);
    assert_eq!(cloned.d2h_bytes, 500);
}

#[test]
fn test_transfer_stats_debug() {
    use crate::memory::resident::TransferStats;

    let stats = TransferStats {
        h2d_transfers: 100,
        d2h_transfers: 50,
        h2d_bytes: 10240,
        d2h_bytes: 5120,
    };

    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("TransferStats"));
    assert!(debug_str.contains("100"));
    assert!(debug_str.contains("50"));
}

#[test]
fn test_kernel_cache_stats_after_operations() {
    use crate::memory::resident::{
        clear_kernel_cache, kernel_cache_hits, kernel_cache_misses, reset_kernel_cache_stats,
    };
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    reset_kernel_cache_stats();

    // First operation should be a cache miss
    let data = vec![1.0f32; 16];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    let _ = tensor.gelu(&ctx).unwrap();

    let first_misses = kernel_cache_misses();
    assert!(first_misses >= 1, "Should have at least 1 cache miss");

    // Same operation again should be a cache hit
    let tensor2 = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    let _ = tensor2.gelu(&ctx).unwrap();

    let hits = kernel_cache_hits();
    assert!(
        hits >= 1,
        "Should have at least 1 cache hit on repeated operation"
    );
}

// ============================================================================
// PMAT-018: Additional weights.rs coverage tests
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
