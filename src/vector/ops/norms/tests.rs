use super::*;
use crate::Backend;

type NormMethod = fn(&Vector<f32>) -> Result<f32>;

fn norm_l1(v: &Vector<f32>) -> Result<f32> { v.norm_l1() }
fn norm_l2(v: &Vector<f32>) -> Result<f32> { v.norm_l2() }
fn norm_linf(v: &Vector<f32>) -> Result<f32> { v.norm_linf() }

fn assert_norm_backend(
    norm_fn: NormMethod, data: &[f32], expected: f32, tol: f32, backend: Backend,
) {
    let result = norm_fn(&Vector::from_slice_with_backend(data, backend)).unwrap();
    assert!((result - expected).abs() <= tol, "expected {expected} got {result} ({backend:?})");
}

/// Unified edge-case + large-vector table for all three norms.
/// Each entry: (norm_fn, label, input_data, expected_result).
fn edge_cases() -> Vec<(NormMethod, &'static str, Vec<f32>, f32)> {
    let l2_large: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01).collect();
    let l2_large_exp = l2_large.iter().map(|x| x * x).sum::<f32>().sqrt();

    vec![
        // --- L2 ---
        (norm_l2, "l2-pythagorean", vec![3.0, 4.0], 5.0),
        (norm_l2, "l2-empty", vec![], 0.0),
        (norm_l2, "l2-unit", vec![1.0, 0.0, 0.0], 1.0),
        (norm_l2, "l2-single", vec![7.0], 7.0),
        (norm_l2, "l2-neg", vec![-5.0], 5.0),
        (norm_l2, "l2-zeros", vec![0.0, 0.0, 0.0, 0.0], 0.0),
        (norm_l2, "l2-mixed", vec![3.0, -4.0, 0.0], 5.0),
        (norm_l2, "l2-5elem", vec![1.0, 2.0, 3.0, 4.0, 5.0], (1.0 + 4.0 + 9.0 + 16.0 + 25.0_f32).sqrt()),
        (norm_l2, "l2-large", l2_large, l2_large_exp),
        // --- L1 ---
        (norm_l1, "l1-basic", vec![3.0, -4.0, 5.0], 12.0),
        (norm_l1, "l1-empty", vec![], 0.0),
        (norm_l1, "l1-single-neg", vec![-7.0], 7.0),
        (norm_l1, "l1-zeros", vec![0.0, 0.0, 0.0], 0.0),
        (norm_l1, "l1-positive", vec![1.0, 2.0, 3.0, 4.0], 10.0),
        (norm_l1, "l1-all-neg", vec![-1.0, -2.0, -3.0], 6.0),
        (norm_l1, "l1-non-aligned", vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0], 28.0),
        (norm_l1, "l1-large", (0..512).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect(), 512.0),
        // --- Linf ---
        (norm_linf, "linf-basic", vec![3.0, -7.0, 5.0, -2.0], 7.0),
        (norm_linf, "linf-empty", vec![], 0.0),
        (norm_linf, "linf-all-neg", vec![-1.0, -5.0, -3.0], 5.0),
        (norm_linf, "linf-single", vec![-42.0], 42.0),
        (norm_linf, "linf-zeros", vec![0.0, 0.0, 0.0], 0.0),
        (norm_linf, "linf-end", vec![1.0, 2.0, 3.0, 100.0], 100.0),
        (norm_linf, "linf-begin", vec![-100.0, 2.0, 3.0, 4.0], 100.0),
        (norm_linf, "linf-equal", vec![5.0, 5.0, 5.0, 5.0], 5.0),
        (norm_linf, "linf-non-aligned", vec![1.0, -9.0, 3.0, -4.0, 5.0], 9.0),
        (norm_linf, "linf-large", {
            let mut d: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
            d[200] = -99.9;
            d
        }, 99.9),
    ]
}

#[test]
fn test_all_norm_edge_cases() {
    for (method, label, data, expected) in edge_cases() {
        let result = method(&Vector::from_slice(&data)).unwrap();
        let tol = if data.len() > 256 { 1e-2 } else { 1e-5 };
        assert!(
            (result - expected).abs() <= tol,
            "{label}: expected {expected}, got {result}"
        );
    }
}

#[test]
fn test_norm_l2_very_small_values() {
    let v = Vector::from_slice(&[1e-20, 1e-20, 1e-20]);
    let norm = v.norm_l2().unwrap();
    assert!(norm > 0.0 && norm < 1e-10);
}

// Cross-norm property: L-inf <= L2 <= L1
#[test]
fn test_norm_ordering_property() {
    for data in [
        vec![3.0, -4.0, 5.0, -2.0, 1.0],
        (0..100).map(|i| ((i as f32) * 0.37).sin()).collect::<Vec<_>>(),
    ] {
        let v = Vector::from_slice(&data);
        let l1 = v.norm_l1().unwrap();
        let l2 = v.norm_l2().unwrap();
        let linf = v.norm_linf().unwrap();
        assert!(linf <= l2 + 1e-4, "L-inf ({linf}) should be <= L2 ({l2})");
        assert!(l2 <= l1 + 1e-4, "L2 ({l2}) should be <= L1 ({l1})");
    }
}

// =========================================================================
// Backend dispatch and equivalence
// =========================================================================

fn norm_specs() -> [(NormMethod, &'static str, &'static [f32], f32); 3] {
    [
        (norm_l1, "l1", &[3.0, -4.0, 5.0], 12.0),
        (norm_l2, "l2", &[3.0, 4.0, 0.0, 0.0], 5.0),
        (norm_linf, "linf", &[3.0, -7.0, 5.0, -2.0], 7.0),
    ]
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_all_norms_sse2_backend() {
    for (method, _name, data, expected) in norm_specs() {
        assert_norm_backend(method, data, expected, 1e-5, Backend::SSE2);
    }
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_all_norms_avx2_backend() {
    if !is_x86_feature_detected!("avx2") { return; }
    for (method, _name, data, expected) in norm_specs() {
        assert_norm_backend(method, data, expected, 1e-3, Backend::AVX2);
    }
}

#[test]
fn test_all_norms_fallback_backends() {
    for (method, _name, data, expected) in norm_specs() {
        for &b in &[Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto, Backend::Scalar] {
            assert_norm_backend(method, data, expected, 1e-5, b);
        }
    }
}

#[test]
fn test_all_norms_backend_equivalence() {
    let data: Vec<f32> = (0..100).map(|i| ((i as f32) * 0.13).sin()).collect();
    for (method, _name, _, _) in norm_specs() {
        let scalar = method(&Vector::from_slice_with_backend(&data, Backend::Scalar)).unwrap();
        for &backend in &[Backend::NEON, Backend::WasmSIMD, Backend::GPU, Backend::Auto] {
            let val = method(&Vector::from_slice_with_backend(&data, backend)).unwrap();
            assert!((scalar - val).abs() < 1e-3, "Scalar vs {backend:?}: {scalar} vs {val}");
        }
        #[cfg(target_arch = "x86_64")]
        {
            let sse2 = method(&Vector::from_slice_with_backend(&data, Backend::SSE2)).unwrap();
            assert!((scalar - sse2).abs() < 1e-3, "Scalar vs SSE2: {scalar} vs {sse2}");
            if is_x86_feature_detected!("avx2") {
                let avx2 = method(&Vector::from_slice_with_backend(&data, Backend::AVX2)).unwrap();
                assert!((scalar - avx2).abs() < 1e-3, "Scalar vs AVX2: {scalar} vs {avx2}");
            }
        }
    }
}

#[test]
fn test_all_norms_non_aligned_sizes() {
    let norms: [(NormMethod, fn(usize) -> Vec<f32>, fn(&[f32]) -> f32); 3] = [
        (norm_l2, |sz| (0..sz).map(|i| (i as f32 + 1.0) * 0.1).collect(), |d| d.iter().map(|x| x * x).sum::<f32>().sqrt()),
        (norm_l1, |sz| (0..sz).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect(), |d| d.len() as f32),
        (norm_linf, |sz| (0..sz).map(|i| i as f32 + 1.0).collect(), |d| d.len() as f32),
    ];
    for (method, make_data, make_expected) in norms {
        for size in [1, 2, 3, 5, 7, 9, 13, 15, 17, 31, 33] {
            let data = make_data(size);
            let result = method(&Vector::from_slice(&data)).unwrap();
            let expected = make_expected(&data);
            assert!((result - expected).abs() < 1e-3, "size {size}: {result} vs {expected}");
        }
    }
}
