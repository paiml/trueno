use super::*;

#[test]
fn test_abs_basic() {
    let v = Vector::from_slice(&[3.0, -4.0, 5.0, -2.0]);
    let result = v.abs().unwrap();
    assert_eq!(result.as_slice(), &[3.0, 4.0, 5.0, 2.0]);
}

#[test]
fn test_abs_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.abs().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_clip_basic() {
    let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
    let clipped = v.clip(0.0, 10.0).unwrap();
    assert_eq!(clipped.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
}

#[test]
fn test_clip_invalid_range() {
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let result = v.clip(10.0, 5.0);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_clamp_basic() {
    let v = Vector::from_slice(&[-5.0, 0.0, 5.0, 10.0, 15.0]);
    let result = v.clamp(0.0, 10.0).unwrap();
    assert_eq!(result.as_slice(), &[0.0, 0.0, 5.0, 10.0, 10.0]);
}

#[test]
fn test_clamp_negative_range() {
    let v = Vector::from_slice(&[-10.0, -5.0, 0.0, 5.0]);
    let result = v.clamp(-8.0, -2.0).unwrap();
    assert_eq!(result.as_slice(), &[-8.0, -5.0, -2.0, -2.0]);
}

#[test]
fn test_lerp_midpoint() {
    let a = Vector::from_slice(&[0.0, 10.0, 20.0]);
    let b = Vector::from_slice(&[100.0, 110.0, 120.0]);
    let result = a.lerp(&b, 0.5).unwrap();
    assert_eq!(result.as_slice(), &[50.0, 60.0, 70.0]);
}

#[test]
fn test_lerp_extrapolation() {
    let a = Vector::from_slice(&[0.0, 10.0]);
    let b = Vector::from_slice(&[10.0, 20.0]);
    let result = a.lerp(&b, 2.0).unwrap();
    assert_eq!(result.as_slice(), &[20.0, 30.0]);
}

#[test]
fn test_lerp_size_mismatch() {
    let a = Vector::from_slice(&[0.0, 10.0]);
    let b = Vector::from_slice(&[10.0, 20.0, 30.0]);
    let result = a.lerp(&b, 0.5);
    assert!(matches!(result, Err(TruenoError::SizeMismatch { .. })));
}

#[test]
fn test_sqrt_basic() {
    let a = Vector::from_slice(&[4.0, 9.0, 16.0, 25.0]);
    let result = a.sqrt().unwrap();
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_sqrt_negative() {
    let a = Vector::from_slice(&[-1.0, 4.0]);
    let result = a.sqrt().unwrap();
    assert!(result.as_slice()[0].is_nan());
    assert_eq!(result.as_slice()[1], 2.0);
}

#[test]
fn test_recip_basic() {
    let a = Vector::from_slice(&[2.0, 4.0, 5.0, 10.0]);
    let result = a.recip().unwrap();
    assert_eq!(result.as_slice(), &[0.5, 0.25, 0.2, 0.1]);
}

#[test]
fn test_recip_zero() {
    let a = Vector::from_slice(&[0.0, 2.0]);
    let result = a.recip().unwrap();
    assert!(result.as_slice()[0].is_infinite());
    assert_eq!(result.as_slice()[1], 0.5);
}

#[test]
fn test_pow_squared() {
    let v = Vector::from_slice(&[2.0, 3.0, 4.0]);
    let squared = v.pow(2.0).unwrap();
    assert_eq!(squared.as_slice(), &[4.0, 9.0, 16.0]);
}

#[test]
fn test_pow_square_root() {
    let v = Vector::from_slice(&[4.0, 9.0, 16.0]);
    let sqrt = v.pow(0.5).unwrap();
    assert!((sqrt.as_slice()[0] - 2.0).abs() < 1e-5);
    assert!((sqrt.as_slice()[1] - 3.0).abs() < 1e-5);
    assert!((sqrt.as_slice()[2] - 4.0).abs() < 1e-5);
}

// =====================================================================
// Coverage: GPU and Auto backend error paths
// =====================================================================

#[test]
fn test_abs_gpu_backend_returns_error() {
    let v = Vector::from_slice_with_backend(&[1.0, -2.0, 3.0], Backend::GPU);
    let result = v.abs();
    assert!(matches!(
        result,
        Err(TruenoError::UnsupportedBackend(Backend::GPU))
    ));
}

#[test]
fn test_abs_auto_backend_returns_error() {
    // Auto is resolved at construction, but we can test the error path
    // by using from_slice_with_backend which resolves Auto to best available.
    // For the GPU path we already tested above. Let's ensure Scalar path works.
    let v = Vector::from_slice_with_backend(&[3.0, -4.0, 5.0], Backend::Scalar);
    let result = v.abs().unwrap();
    assert_eq!(result.as_slice(), &[3.0, 4.0, 5.0]);
}

#[test]
fn test_clamp_gpu_backend_returns_error() {
    let v = Vector::from_slice_with_backend(&[1.0, 2.0, 3.0], Backend::GPU);
    let result = v.clamp(0.0, 2.0);
    assert!(matches!(
        result,
        Err(TruenoError::UnsupportedBackend(Backend::GPU))
    ));
}

#[test]
fn test_clamp_scalar_backend() {
    let v = Vector::from_slice_with_backend(&[-5.0, 0.0, 5.0, 10.0], Backend::Scalar);
    let result = v.clamp(0.0, 8.0).unwrap();
    assert_eq!(result.as_slice(), &[0.0, 0.0, 5.0, 8.0]);
}

#[test]
fn test_clamp_invalid_range() {
    let v = Vector::from_slice(&[1.0, 2.0]);
    let result = v.clamp(10.0, 5.0);
    assert!(matches!(result, Err(TruenoError::InvalidInput(_))));
}

#[test]
fn test_clamp_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.clamp(0.0, 1.0).unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_lerp_gpu_backend_returns_error() {
    let a = Vector::from_slice_with_backend(&[1.0, 2.0], Backend::GPU);
    let b = Vector::from_slice_with_backend(&[3.0, 4.0], Backend::GPU);
    let result = a.lerp(&b, 0.5);
    assert!(matches!(
        result,
        Err(TruenoError::UnsupportedBackend(Backend::GPU))
    ));
}

#[test]
fn test_lerp_scalar_backend() {
    let a = Vector::from_slice_with_backend(&[0.0, 10.0], Backend::Scalar);
    let b = Vector::from_slice_with_backend(&[10.0, 20.0], Backend::Scalar);
    let result = a.lerp(&b, 0.5).unwrap();
    assert_eq!(result.as_slice(), &[5.0, 15.0]);
}

#[test]
fn test_lerp_empty() {
    let a: Vector<f32> = Vector::from_slice(&[]);
    let b: Vector<f32> = Vector::from_slice(&[]);
    let result = a.lerp(&b, 0.5).unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_sqrt_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.sqrt().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_recip_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.recip().unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_pow_zero_exponent() {
    let v = Vector::from_slice(&[2.0, 0.0, -3.0]);
    let result = v.pow(0.0).unwrap();
    // x^0 = 1.0 for all x
    assert_eq!(result.as_slice(), &[1.0, 1.0, 1.0]);
}

#[test]
fn test_pow_empty() {
    let v: Vector<f32> = Vector::from_slice(&[]);
    let result = v.pow(2.0).unwrap();
    assert_eq!(result.len(), 0);
}
