//! Elementwise GPU-Resident Tensor tests: softmax, add, scale, gelu, bias_add, layer_norm

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
// Softmax Tests
// ============================================================================

#[test]
fn test_ops_softmax_warp() {
    let ctx = cuda_ctx!();

    // Small row size (<=32) uses warp shuffle softmax
    let seq_len = 8u32;
    let row_size = 16u32;
    let data: Vec<f32> = (0..(seq_len * row_size))
        .map(|i| (i % row_size) as f32 * 0.1)
        .collect();

    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    let mut result = tensor.softmax(&ctx, seq_len).unwrap();

    assert_eq!(result.len(), (seq_len * row_size) as usize);

    // Verify softmax output - values should be positive and <= 1
    let host_result = result.to_host().unwrap();
    for val in &host_result {
        assert!(*val >= 0.0 && *val <= 1.0 + 1e-5);
    }
}

#[test]
fn test_ops_softmax_long_row() {
    let ctx = cuda_ctx!();

    // Large row size (>32) uses long row softmax
    let seq_len = 4u32;
    let row_size = 128u32;
    let data: Vec<f32> = (0..(seq_len * row_size))
        .map(|i| (i % row_size) as f32 * 0.01)
        .collect();

    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    let result = tensor.softmax(&ctx, seq_len).unwrap();

    assert_eq!(result.len(), (seq_len * row_size) as usize);
}

#[test]
fn test_ops_softmax_dimension_error() {
    let ctx = cuda_ctx!();

    // Tensor size not divisible by seq_len
    let data: Vec<f32> = vec![1.0; 10];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    let result = tensor.softmax(&ctx, 3); // 10 not divisible by 3
    assert!(result.is_err());
}

#[test]
fn test_ops_softmax_with_stream() {
    let ctx = cuda_ctx!();

    let seq_len = 4u32;
    let row_size = 64u32;
    let data: Vec<f32> = vec![1.0; (seq_len * row_size) as usize];

    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    let stream = CudaStream::new(&ctx).unwrap();

    let result = tensor.softmax_with_stream(&ctx, seq_len, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(result.len(), (seq_len * row_size) as usize);
}

// ============================================================================
// Add Tests
// ============================================================================

#[test]
fn test_ops_add() {
    let ctx = cuda_ctx!();

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![0.5f32, 0.5, 0.5, 0.5];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let mut c = a.add(&ctx, &b).unwrap();
    let result = c.to_host().unwrap();

    assert_eq!(result, vec![1.5, 2.5, 3.5, 4.5]);
}

#[test]
fn test_ops_add_size_mismatch() {
    let ctx = cuda_ctx!();

    let a_data = vec![1.0f32; 10];
    let b_data = vec![1.0f32; 5];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let result = a.add(&ctx, &b);
    assert!(result.is_err());
}

#[test]
fn test_ops_add_with_stream() {
    let ctx = cuda_ctx!();

    let a_data = vec![1.0f32; 256];
    let b_data = vec![2.0f32; 256];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let mut c = a.add_with_stream(&ctx, &b, &stream).unwrap();
    stream.synchronize().unwrap();

    let result = c.to_host().unwrap();
    assert!(result.iter().all(|&v| (v - 3.0).abs() < 1e-5));
}

// ============================================================================
// Scale Tests
// ============================================================================

#[test]
fn test_ops_scale() {
    let ctx = cuda_ctx!();

    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    let mut scaled = tensor.scale(&ctx, 2.0).unwrap();
    let result = scaled.to_host().unwrap();

    assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
}

// ============================================================================
// Layer Norm Tests
// ============================================================================

#[test]
fn test_ops_layer_norm() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let hidden_size = 16u32;
    let batch_size = 4u32;

    // Input data
    let input_data: Vec<f32> = (0..(hidden_size * batch_size))
        .map(|i| i as f32 * 0.1)
        .collect();

    // Gamma (scale) = all ones
    let gamma_data = vec![1.0f32; hidden_size as usize];
    // Beta (shift) = all zeros
    let beta_data = vec![0.0f32; hidden_size as usize];

    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();
    let gamma = GpuResidentTensor::from_host(&ctx, &gamma_data).unwrap();
    let beta = GpuResidentTensor::from_host(&ctx, &beta_data).unwrap();

    let mut output = input
        .layer_norm(&ctx, &gamma, &beta, hidden_size, batch_size)
        .unwrap();
    assert_eq!(output.len(), (hidden_size * batch_size) as usize);

    // With gamma=1, beta=0, output should be normalized (mean~0, std~1)
    let host_output = output.to_host().unwrap();
    // Check first row mean is approximately 0
    let first_row: Vec<f32> = host_output[0..hidden_size as usize].to_vec();
    let mean: f32 = first_row.iter().sum::<f32>() / hidden_size as f32;
    assert!(
        (mean).abs() < 0.1,
        "LayerNorm output mean should be ~0, got {}",
        mean
    );
}

#[test]
fn test_ops_layer_norm_with_stream() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let hidden_size = 32u32;
    let batch_size = 2u32;

    let input_data = vec![1.0f32; (hidden_size * batch_size) as usize];
    let gamma_data = vec![1.0f32; hidden_size as usize];
    let beta_data = vec![0.0f32; hidden_size as usize];

    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();
    let gamma = GpuResidentTensor::from_host(&ctx, &gamma_data).unwrap();
    let beta = GpuResidentTensor::from_host(&ctx, &beta_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let output = input
        .layer_norm_with_stream(&ctx, &gamma, &beta, hidden_size, batch_size, &stream)
        .unwrap();
    stream.synchronize().unwrap();

    assert_eq!(output.len(), (hidden_size * batch_size) as usize);
}

// ============================================================================
// GELU Tests
// ============================================================================

#[test]
fn test_ops_gelu() {
    let ctx = cuda_ctx!();

    // Test with known values
    // GELU(0) ~= 0, GELU(x) for large x ~= x
    let data = vec![0.0f32, 1.0, 2.0, -1.0, -2.0];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    let mut output = tensor.gelu(&ctx).unwrap();
    let result = output.to_host().unwrap();

    // GELU(0) should be 0
    assert!((result[0]).abs() < 1e-5, "GELU(0) should be ~0");

    // GELU(1) ~= 0.841 (approximate)
    assert!((result[1] - 0.841).abs() < 0.1, "GELU(1) should be ~0.841");

    // GELU(-1) ~= -0.159 (approximate)
    assert!(
        (result[3] - (-0.159)).abs() < 0.1,
        "GELU(-1) should be ~-0.159"
    );
}

#[test]
fn test_ops_gelu_with_stream() {
    let ctx = cuda_ctx!();

    let data = vec![1.0f32; 512];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let output = tensor.gelu_with_stream(&ctx, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(output.len(), 512);
}

// ============================================================================
// Bias Add Tests
// ============================================================================

#[test]
fn test_ops_bias_add() {
    let ctx = cuda_ctx!();

    // 4 rows of 3 elements each
    let input_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let bias_data = vec![0.1f32, 0.2, 0.3];

    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();
    let bias = GpuResidentTensor::from_host(&ctx, &bias_data).unwrap();

    let mut output = input.bias_add(&ctx, &bias).unwrap();
    let result = output.to_host().unwrap();

    // Check first row
    assert!((result[0] - 1.1).abs() < 1e-5);
    assert!((result[1] - 2.2).abs() < 1e-5);
    assert!((result[2] - 3.3).abs() < 1e-5);
}

#[test]
fn test_ops_bias_add_with_stream() {
    let ctx = cuda_ctx!();

    let input_data = vec![1.0f32; 256];
    let bias_data = vec![0.5f32; 64];

    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();
    let bias = GpuResidentTensor::from_host(&ctx, &bias_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let mut output = input.bias_add_with_stream(&ctx, &bias, &stream).unwrap();
    stream.synchronize().unwrap();

    let result = output.to_host().unwrap();
    assert!((result[0] - 1.5).abs() < 1e-5);
}

// ============================================================================
// PMAT-018: ops.rs elementwise coverage tests for 95%+ target
// ============================================================================

#[test]
fn test_ops_softmax_with_stream_warp_path() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // row_size <= 32 uses warp shuffle path
    let seq_len = 4u32;
    let row_size = 16u32;
    let data: Vec<f32> = (0..(seq_len * row_size))
        .map(|i| (i % 10) as f32 * 0.1)
        .collect();

    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    let stream = CudaStream::new(&ctx).unwrap();

    let result = tensor.softmax_with_stream(&ctx, seq_len, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(result.len(), (seq_len * row_size) as usize);
}

#[test]
fn test_ops_softmax_with_stream_long_row_path() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // row_size > 32 uses long row softmax path
    let seq_len = 4u32;
    let row_size = 64u32;
    let data: Vec<f32> = (0..(seq_len * row_size))
        .map(|i| (i % 10) as f32 * 0.1)
        .collect();

    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    let stream = CudaStream::new(&ctx).unwrap();

    let result = tensor.softmax_with_stream(&ctx, seq_len, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(result.len(), (seq_len * row_size) as usize);
}

#[test]
fn test_ops_softmax_with_stream_dimension_error() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Size not divisible by seq_len
    let data: Vec<f32> = vec![1.0; 10];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    let stream = CudaStream::new(&ctx).unwrap();

    let result = tensor.softmax_with_stream(&ctx, 3, &stream); // 10 not divisible by 3
    assert!(result.is_err());
}

#[test]
fn test_ops_add_with_stream_dimension_error() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let a_data = vec![1.0f32; 10];
    let b_data = vec![1.0f32; 5];

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let result = a.add_with_stream(&ctx, &b, &stream);
    assert!(result.is_err());
}

#[test]
fn test_ops_scale_larger_tensor() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Test with larger tensor to exercise multi-block launch path
    let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect();
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    let mut scaled = tensor.scale(&ctx, 0.5).unwrap();
    let result = scaled.to_host().unwrap();

    // Check a few values
    assert!((result[0] - 0.0).abs() < 1e-5);
    assert!((result[100] - 0.5).abs() < 1e-5);
    assert!((result[1000] - 5.0).abs() < 1e-5);
}

#[test]
fn test_ops_gelu_larger_tensor() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Test with larger tensor to exercise multi-block launch path
    let data: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    let output = tensor.gelu(&ctx).unwrap();
    assert_eq!(output.len(), 1024);
}

#[test]
fn test_ops_layer_norm_larger_batch() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let hidden_size = 32u32;
    let batch_size = 16u32;

    let input_data: Vec<f32> = (0..(hidden_size * batch_size))
        .map(|i| (i as f32) * 0.01)
        .collect();
    let gamma_data = vec![1.0f32; hidden_size as usize];
    let beta_data = vec![0.0f32; hidden_size as usize];

    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();
    let gamma = GpuResidentTensor::from_host(&ctx, &gamma_data).unwrap();
    let beta = GpuResidentTensor::from_host(&ctx, &beta_data).unwrap();

    let output = input
        .layer_norm(&ctx, &gamma, &beta, hidden_size, batch_size)
        .unwrap();
    assert_eq!(output.len(), (hidden_size * batch_size) as usize);
}

#[test]
fn test_ops_bias_add_larger_tensor() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // 256 rows of 64 elements each
    let n_rows = 256usize;
    let bias_size = 64usize;
    let input_data = vec![1.0f32; n_rows * bias_size];
    let bias_data: Vec<f32> = (0..bias_size).map(|i| i as f32 * 0.1).collect();

    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();
    let bias = GpuResidentTensor::from_host(&ctx, &bias_data).unwrap();

    let mut output = input.bias_add(&ctx, &bias).unwrap();
    let result = output.to_host().unwrap();

    // Check that bias was added correctly
    assert!((result[0] - 1.0).abs() < 1e-5); // 1.0 + 0.0
    assert!((result[1] - 1.1).abs() < 1e-5); // 1.0 + 0.1
    assert!((result[64] - 1.0).abs() < 1e-5); // Next row, first element
}

#[test]
fn test_ops_add_larger_tensor() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Test with tensor larger than 256 (one block)
    let size = 1024usize;
    let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let mut c = a.add(&ctx, &b).unwrap();
    let result = c.to_host().unwrap();

    // All values should sum to 1024
    assert!(result.iter().all(|&v| (v - 1024.0).abs() < 1e-3));
}

#[test]
fn test_ops_gelu_with_stream_larger() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    // Large tensor to exercise multi-block path
    let data: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.001).collect();
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let output = tensor.gelu_with_stream(&ctx, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(output.len(), 2048);
}

#[test]
fn test_ops_layer_norm_with_stream_larger() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let hidden_size = 64u32;
    let batch_size = 8u32;

    let input_data: Vec<f32> = (0..(hidden_size * batch_size))
        .map(|i| (i as f32) * 0.01)
        .collect();
    let gamma_data = vec![1.0f32; hidden_size as usize];
    let beta_data = vec![0.5f32; hidden_size as usize];

    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();
    let gamma = GpuResidentTensor::from_host(&ctx, &gamma_data).unwrap();
    let beta = GpuResidentTensor::from_host(&ctx, &beta_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let output = input
        .layer_norm_with_stream(&ctx, &gamma, &beta, hidden_size, batch_size, &stream)
        .unwrap();
    stream.synchronize().unwrap();

    assert_eq!(output.len(), (hidden_size * batch_size) as usize);
}

#[test]
fn test_ops_bias_add_with_stream_larger() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let n_rows = 128usize;
    let bias_size = 32usize;
    let input_data = vec![0.5f32; n_rows * bias_size];
    let bias_data: Vec<f32> = (0..bias_size).map(|i| i as f32 * 0.05).collect();

    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();
    let bias = GpuResidentTensor::from_host(&ctx, &bias_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let output = input.bias_add_with_stream(&ctx, &bias, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(output.len(), n_rows * bias_size);
}

#[test]
fn test_ops_add_with_stream_larger() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let size = 2048usize;
    let a_data: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.001).collect();

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let stream = CudaStream::new(&ctx).unwrap();
    let c = a.add_with_stream(&ctx, &b, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(c.len(), size);
}
