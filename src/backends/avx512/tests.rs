use super::*;
use crate::backends::scalar::ScalarBackend;

fn avx512_test<F>(test_fn: F)
where
    F: FnOnce(),
{
    if is_x86_feature_detected!("avx512f") {
        test_fn();
    } else {
        println!("Skipping AVX-512 test (CPU does not support avx512f)");
    }
}

#[test]
fn test_avx512_add() {
    avx512_test(|| {
        let a = vec![1.0; 32];
        let b = vec![2.0; 32];
        let mut result = vec![0.0; 32];
        unsafe {
            Avx512Backend::add(&a, &b, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    });
}

#[test]
fn test_avx512_sub() {
    avx512_test(|| {
        let a = vec![5.0; 32];
        let b = vec![2.0; 32];
        let mut result = vec![0.0; 32];
        unsafe {
            Avx512Backend::sub(&a, &b, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    });
}

#[test]
fn test_avx512_mul() {
    avx512_test(|| {
        let a = vec![2.0; 32];
        let b = vec![3.0; 32];
        let mut result = vec![0.0; 32];
        unsafe {
            Avx512Backend::mul(&a, &b, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 6.0).abs() < 1e-6));
    });
}

#[test]
fn test_avx512_div() {
    avx512_test(|| {
        let a = vec![6.0; 32];
        let b = vec![2.0; 32];
        let mut result = vec![0.0; 32];
        unsafe {
            Avx512Backend::div(&a, &b, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    });
}

#[test]
fn test_avx512_dot() {
    avx512_test(|| {
        let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let result = unsafe { Avx512Backend::dot(&a, &b) };
        let expected: f32 = (1..=32).map(|i| (i * i) as f32).sum();
        assert!((result - expected).abs() < 1e-3);
    });
}

#[test]
fn test_avx512_sum() {
    avx512_test(|| {
        let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let result = unsafe { Avx512Backend::sum(&a) };
        assert!((result - 528.0).abs() < 1e-3);
    });
}

#[test]
fn test_avx512_max() {
    avx512_test(|| {
        let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let result = unsafe { Avx512Backend::max(&a) };
        assert!((result - 32.0).abs() < 1e-6);
    });
}

#[test]
fn test_avx512_min() {
    avx512_test(|| {
        let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let result = unsafe { Avx512Backend::min(&a) };
        assert!((result - 1.0).abs() < 1e-6);
    });
}

#[test]
fn test_avx512_argmax() {
    avx512_test(|| {
        let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let result = unsafe { Avx512Backend::argmax(&a) };
        assert_eq!(result, 31);
    });
}

#[test]
fn test_avx512_argmin() {
    avx512_test(|| {
        let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let result = unsafe { Avx512Backend::argmin(&a) };
        assert_eq!(result, 0);
    });
}

#[test]
fn test_avx512_sum_kahan() {
    avx512_test(|| {
        let a: Vec<f32> = (1..=32).map(|i| i as f32).collect();
        let result = unsafe { Avx512Backend::sum_kahan(&a) };
        assert!((result - 528.0).abs() < 1e-3);
    });
}

#[test]
fn test_avx512_norm_l2() {
    avx512_test(|| {
        let a = vec![3.0, 4.0];
        let result = unsafe { Avx512Backend::norm_l2(&a) };
        assert!((result - 5.0).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_norm_l1() {
    avx512_test(|| {
        let a = vec![-1.0, 2.0, -3.0, 4.0];
        let result = unsafe { Avx512Backend::norm_l1(&a) };
        assert!((result - 10.0).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_norm_linf() {
    avx512_test(|| {
        let a = vec![-5.0, 2.0, -3.0, 4.0];
        let result = unsafe { Avx512Backend::norm_linf(&a) };
        assert!((result - 5.0).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_scale() {
    avx512_test(|| {
        let a = vec![1.0; 32];
        let mut result = vec![0.0; 32];
        unsafe {
            Avx512Backend::scale(&a, 3.0, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-6));
    });
}

#[test]
fn test_avx512_abs() {
    avx512_test(|| {
        let a = vec![-1.0, 2.0, -3.0, 4.0];
        let mut result = vec![0.0; 4];
        unsafe {
            Avx512Backend::abs(&a, &mut result);
        }
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
    });
}

#[test]
fn test_avx512_clamp() {
    avx512_test(|| {
        let a = vec![0.0, 5.0, 10.0, 15.0];
        let mut result = vec![0.0; 4];
        unsafe {
            Avx512Backend::clamp(&a, 2.0, 12.0, &mut result);
        }
        assert_eq!(result, vec![2.0, 5.0, 10.0, 12.0]);
    });
}

#[test]
fn test_avx512_lerp() {
    avx512_test(|| {
        let a = vec![0.0; 32];
        let b = vec![10.0; 32];
        let mut result = vec![0.0; 32];
        unsafe {
            Avx512Backend::lerp(&a, &b, 0.5, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 5.0).abs() < 1e-5));
    });
}

#[test]
fn test_avx512_fma() {
    avx512_test(|| {
        let a = vec![2.0; 32];
        let b = vec![3.0; 32];
        let c = vec![1.0; 32];
        let mut result = vec![0.0; 32];
        unsafe {
            Avx512Backend::fma(&a, &b, &c, &mut result);
        }
        assert!(result.iter().all(|&x| (x - 7.0).abs() < 1e-5));
    });
}

#[test]
fn test_avx512_relu() {
    avx512_test(|| {
        let a = vec![-1.0, 0.0, 1.0, 2.0];
        let mut result = vec![0.0; 4];
        unsafe {
            Avx512Backend::relu(&a, &mut result);
        }
        assert_eq!(result, vec![0.0, 0.0, 1.0, 2.0]);
    });
}

#[test]
fn test_avx512_exp() {
    avx512_test(|| {
        let a = vec![0.0, 1.0];
        let mut result = vec![0.0; 2];
        unsafe {
            Avx512Backend::exp(&a, &mut result);
        }
        assert!((result[0] - 1.0).abs() < 1e-4);
        assert!((result[1] - std::f32::consts::E).abs() < 1e-3);
    });
}

#[test]
fn test_avx512_sigmoid() {
    avx512_test(|| {
        let a = vec![0.0];
        let mut result = vec![0.0; 1];
        unsafe {
            Avx512Backend::sigmoid(&a, &mut result);
        }
        assert!((result[0] - 0.5).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_gelu() {
    avx512_test(|| {
        let a = vec![0.0, 1.0];
        let mut result = vec![0.0; 2];
        unsafe {
            Avx512Backend::gelu(&a, &mut result);
        }
        assert!((result[0]).abs() < 1e-5);
        assert!((result[1] - 0.841_192).abs() < 1e-3);
    });
}

#[test]
fn test_avx512_swish() {
    avx512_test(|| {
        let a = vec![0.0, 1.0];
        let mut result = vec![0.0; 2];
        unsafe {
            Avx512Backend::swish(&a, &mut result);
        }
        assert!((result[0]).abs() < 1e-5);
        assert!((result[1] - 0.731_059).abs() < 1e-3);
    });
}

#[test]
fn test_avx512_tanh() {
    avx512_test(|| {
        let a = vec![0.0, 1.0];
        let mut result = vec![0.0; 2];
        unsafe {
            Avx512Backend::tanh(&a, &mut result);
        }
        assert!((result[0]).abs() < 1e-5);
        assert!((result[1] - 0.761_594_2).abs() < 1e-3);
    });
}

#[test]
fn test_avx512_sqrt() {
    avx512_test(|| {
        let a = vec![4.0, 9.0, 16.0];
        let mut result = vec![0.0; 3];
        unsafe {
            Avx512Backend::sqrt(&a, &mut result);
        }
        assert!((result[0] - 2.0).abs() < 1e-5);
        assert!((result[1] - 3.0).abs() < 1e-5);
        assert!((result[2] - 4.0).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_recip() {
    avx512_test(|| {
        let a = vec![2.0, 4.0, 5.0];
        let mut result = vec![0.0; 3];
        unsafe {
            Avx512Backend::recip(&a, &mut result);
        }
        assert!((result[0] - 0.5).abs() < 1e-5);
        assert!((result[1] - 0.25).abs() < 1e-5);
        assert!((result[2] - 0.2).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_transcendental() {
    avx512_test(|| {
        let a = vec![1.0, std::f32::consts::E, 10.0];
        let mut ln_result = vec![0.0; 3];
        let mut log2_result = vec![0.0; 3];
        let mut log10_result = vec![0.0; 3];
        unsafe {
            Avx512Backend::ln(&a, &mut ln_result);
            Avx512Backend::log2(&a, &mut log2_result);
            Avx512Backend::log10(&a, &mut log10_result);
        }
        assert!((ln_result[0]).abs() < 1e-5);
        assert!((ln_result[1] - 1.0).abs() < 1e-4);
        assert!((log10_result[2] - 1.0).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_trig() {
    avx512_test(|| {
        let a = vec![0.0, std::f32::consts::FRAC_PI_2];
        let mut sin_result = vec![0.0; 2];
        let mut cos_result = vec![0.0; 2];
        let mut tan_result = vec![0.0; 2];
        unsafe {
            Avx512Backend::sin(&a, &mut sin_result);
            Avx512Backend::cos(&a, &mut cos_result);
            Avx512Backend::tan(&a, &mut tan_result);
        }
        assert!((sin_result[0]).abs() < 1e-5);
        assert!((sin_result[1] - 1.0).abs() < 1e-5);
        assert!((cos_result[0] - 1.0).abs() < 1e-5);
    });
}

#[test]
fn test_avx512_rounding() {
    avx512_test(|| {
        let a = vec![1.3, 1.5, 1.7, -1.3, -1.5, -1.7];
        let mut floor_result = vec![0.0; 6];
        let mut ceil_result = vec![0.0; 6];
        let mut round_result = vec![0.0; 6];
        unsafe {
            Avx512Backend::floor(&a, &mut floor_result);
            Avx512Backend::ceil(&a, &mut ceil_result);
            Avx512Backend::round(&a, &mut round_result);
        }
        assert_eq!(floor_result, vec![1.0, 1.0, 1.0, -2.0, -2.0, -2.0]);
        assert_eq!(ceil_result, vec![2.0, 2.0, 2.0, -1.0, -1.0, -1.0]);
    });
}

#[test]
fn test_avx512_backend_equivalence() {
    avx512_test(|| {
        let a: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..100).map(|i| (100 - i) as f32 * 0.1).collect();
        let mut avx512_add = vec![0.0; 100];
        let mut scalar_add = vec![0.0; 100];
        unsafe {
            Avx512Backend::add(&a, &b, &mut avx512_add);
            ScalarBackend::add(&a, &b, &mut scalar_add);
        }
        for i in 0..100 {
            assert!(
                (avx512_add[i] - scalar_add[i]).abs() < 1e-5,
                "add mismatch at {}",
                i
            );
        }
    });
}
