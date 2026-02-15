//! PMAT-018: ops.rs elementwise coverage tests for 95%+ target

use super::*;

#[test]
fn test_ops_softmax_with_stream_warp_path() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

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

    let data: Vec<f32> = vec![1.0; 10];
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();
    let stream = CudaStream::new(&ctx).unwrap();

    let result = tensor.softmax_with_stream(&ctx, 3, &stream);
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

    let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect();
    let tensor = GpuResidentTensor::from_host(&ctx, &data).unwrap();

    let mut scaled = tensor.scale(&ctx, 0.5).unwrap();
    let result = scaled.to_host().unwrap();

    assert!((result[0] - 0.0).abs() < 1e-5);
    assert!((result[100] - 0.5).abs() < 1e-5);
    assert!((result[1000] - 5.0).abs() < 1e-5);
}

#[test]
fn test_ops_gelu_larger_tensor() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

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

    let n_rows = 256usize;
    let bias_size = 64usize;
    let input_data = vec![1.0f32; n_rows * bias_size];
    let bias_data: Vec<f32> = (0..bias_size).map(|i| i as f32 * 0.1).collect();

    let input = GpuResidentTensor::from_host(&ctx, &input_data).unwrap();
    let bias = GpuResidentTensor::from_host(&ctx, &bias_data).unwrap();

    let mut output = input.bias_add(&ctx, &bias).unwrap();
    let result = output.to_host().unwrap();

    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - 1.1).abs() < 1e-5);
    assert!((result[64] - 1.0).abs() < 1e-5);
}

#[test]
fn test_ops_add_larger_tensor() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

    let size = 1024usize;
    let a_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..size).map(|i| (size - i) as f32).collect();

    let a = GpuResidentTensor::from_host(&ctx, &a_data).unwrap();
    let b = GpuResidentTensor::from_host(&ctx, &b_data).unwrap();

    let mut c = a.add(&ctx, &b).unwrap();
    let result = c.to_host().unwrap();

    assert!(result.iter().all(|&v| (v - 1024.0).abs() < 1e-3));
}

#[test]
fn test_ops_gelu_with_stream_larger() {
    use crate::memory::resident::clear_kernel_cache;
    clear_kernel_cache();
    let ctx = cuda_ctx!();

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
