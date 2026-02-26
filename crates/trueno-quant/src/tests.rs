use super::*;

/// Standard test data: 256 floats centered around zero.
fn test_data_256() -> Vec<f32> {
    (0..256).map(|i| (i as f32 - 128.0) / 10.0).collect()
}

/// Compute max absolute error between original and dequantized data.
fn max_abs_error(original: &[f32], dequantized: &[f32]) -> f32 {
    original.iter().zip(dequantized.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
}

/// Compute data range (max - min).
fn data_range(data: &[f32]) -> f32 {
    data.iter().fold(0.0f32, |a, &b| a.max(b)) - data.iter().fold(0.0f32, |a, &b| a.min(b))
}

/// Assert roundtrip error is within a fraction of data range.
fn assert_roundtrip_within_range(
    original: &[f32],
    dequantized: &[f32],
    fraction: f32,
    label: &str,
) {
    let error = max_abs_error(original, dequantized);
    let threshold = data_range(original) * fraction;
    assert!(error < threshold, "{label} roundtrip error {error} exceeds threshold {threshold}");
}

#[test]
fn test_q4k_roundtrip() {
    let data = test_data_256();
    let quantized = quantize_q4_k(&data);
    assert_eq!(quantized.len(), 144);
    let dequantized = dequantize_q4_k_to_f32(&quantized, 256);
    assert_roundtrip_within_range(&data, &dequantized, 0.5, "Q4K");
}

#[test]
fn test_q5k_roundtrip() {
    let data = test_data_256();
    let quantized = quantize_q5_k(&data);
    assert_eq!(quantized.len(), 176);
    let dequantized = dequantize_q5_k_to_f32(&quantized, 256);
    assert_roundtrip_within_range(&data, &dequantized, 0.4, "Q5K");
}

#[test]
fn test_q6k_roundtrip() {
    let data = test_data_256();
    let quantized = quantize_q6_k(&data);
    assert_eq!(quantized.len(), 210);
    let dequantized = dequantize_q6_k_to_f32(&quantized, 256);
    assert!(max_abs_error(&data, &dequantized) < 1.0, "Q6K roundtrip error too high");
}

#[test]
fn test_q4k_matrix() {
    let data: Vec<f32> = (0..512).map(|i| i as f32 / 100.0).collect();
    let shape = vec![2, 256];
    let quantized = quantize_q4_k_matrix(&data, &shape);
    assert_eq!(quantized.len(), 2 * 144);
}

#[test]
fn test_transpose_q4k() {
    let cols = 256;
    let rows = 2;
    let data: Vec<f32> = (0..(rows * cols)).map(|i| i as f32 / 10.0).collect();
    let quantized = quantize_q4_k(&data);
    let shape = vec![cols, rows];
    let (transposed_data, new_shape) = transpose_q4k_for_matmul(&quantized, &shape);
    assert_eq!(new_shape, vec![rows, cols]);
    assert!(!transposed_data.is_empty());
}

#[test]
fn test_f16_min_normal() {
    let f16_val = half::f16::from_f32(F16_MIN_NORMAL);
    let roundtrip = f16_val.to_f32();
    assert!(roundtrip > 0.0, "F16_MIN_NORMAL should be positive after f16 roundtrip");
    assert!(roundtrip < 1e-4, "F16_MIN_NORMAL should be small");
}

#[test]
fn test_constants() {
    assert_eq!(Q4_K_BLOCK_SIZE, 256);
    assert_eq!(Q4_K_BLOCK_BYTES, 144);
    assert_eq!(Q5_K_BLOCK_SIZE, 256);
    assert_eq!(Q5_K_BLOCK_BYTES, 176);
    assert_eq!(Q6_K_BLOCK_SIZE, 256);
    assert_eq!(Q6_K_BLOCK_BYTES, 210);
}
