//! AVX2 backend tests
//!
//! Tests verifying AVX2 SIMD operations produce correct results and match scalar baseline.

#[cfg(test)]
mod tests {
    use super::super::avx2::Avx2Backend;
    use super::super::scalar::ScalarBackend;
    use super::super::VectorBackend;

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_add() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2");
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = vec![0.0; 9];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::add(&a, &b, &mut result);
        }

        assert_eq!(
            result,
            vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_mul() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2");
            return;
        }

        let a = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut result = vec![0.0; 9];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::mul(&a, &b, &mut result);
        }

        assert_eq!(
            result,
            vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0, 90.0]
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_dot() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        // SAFETY: Test code calling backend trait methods marked unsafe
        let result = unsafe { Avx2Backend::dot(&a, &b) };

        // 1*9 + 2*8 + 3*7 + 4*6 + 5*5 + 6*4 + 7*3 + 8*2 + 9*1
        // = 9 + 16 + 21 + 24 + 25 + 24 + 21 + 16 + 9 = 165
        assert!((result - 165.0).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_sum() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2");
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // SAFETY: Test code calling backend trait methods marked unsafe
        let result = unsafe { Avx2Backend::sum(&a) };

        assert!((result - 45.0).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_max() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2");
            return;
        }

        let a = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];

        // SAFETY: Test code calling backend trait methods marked unsafe
        let result = unsafe { Avx2Backend::max(&a) };

        assert_eq!(result, 9.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_min() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2");
            return;
        }

        let a = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0];

        // SAFETY: Test code calling backend trait methods marked unsafe
        let result = unsafe { Avx2Backend::min(&a) };

        assert_eq!(result, 1.0);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];
        let b = vec![10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5];

        // Test add
        let mut avx2_result = vec![0.0; 10];
        let mut scalar_result = vec![0.0; 10];
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::add(&a, &b, &mut avx2_result);
            ScalarBackend::add(&a, &b, &mut scalar_result);
        }
        for (avx2, scalar) in avx2_result.iter().zip(&scalar_result) {
            assert!((avx2 - scalar).abs() < 1e-5);
        }

        // Test dot
        let (avx2_dot, scalar_dot) =
            // SAFETY: Calling backend methods with verified safety invariants
            unsafe { (Avx2Backend::dot(&a, &b), ScalarBackend::dot(&a, &b)) };
        assert!((avx2_dot - scalar_dot).abs() < 1e-3); // Relaxed tolerance for FMA

        // Test sum
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        let (avx2_sum, scalar_sum) = unsafe { (Avx2Backend::sum(&a), ScalarBackend::sum(&a)) };
        assert!((avx2_sum - scalar_sum).abs() < 1e-3);

        // Test max
        // SAFETY: Calling backend methods with verified safety invariants
        let (avx2_max, scalar_max) = unsafe { (Avx2Backend::max(&a), ScalarBackend::max(&a)) };
        assert_eq!(avx2_max, scalar_max);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_relu() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }

        // Test with 16 elements (2 AVX2 registers of 8 f32s)
        let a = [
            -3.0, -1.0, 0.0, 1.0, 3.0, -2.0, 2.0, -0.5, -4.0, 4.0, -5.0, 5.0, 0.0, -0.1, 0.1, 10.0,
        ];
        let mut result = [0.0; 16];
        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::relu(&a, &mut result);
        }
        let expected = [
            0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 5.0, 0.0, 0.0, 0.1, 10.0,
        ];
        assert_eq!(result, expected);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_relu_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, -2.0, 2.0, -4.0, 4.0];
        let mut avx2_result = [0.0; 11];
        let mut scalar_result = [0.0; 11];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::relu(&a, &mut avx2_result);
            ScalarBackend::relu(&a, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_sigmoid_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [-10.0, -1.0, 0.0, 1.0, 10.0];
        let mut avx2_result = [0.0; 5];
        let mut scalar_result = [0.0; 5];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::sigmoid(&a, &mut avx2_result);
            ScalarBackend::sigmoid(&a, &mut scalar_result);
        }

        for (avx2, scalar) in avx2_result.iter().zip(scalar_result.iter()) {
            assert!(
                (avx2 - scalar).abs() < 1e-6,
                "sigmoid mismatch: avx2={}, scalar={}",
                avx2,
                scalar
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_exp_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        // Test various ranges: negative, zero, positive, large values
        let test_values = vec![
            -10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, -50.0, 87.0,
            -87.0, // near overflow/underflow limits
        ];
        let mut avx2_result = vec![0.0; test_values.len()];
        let mut scalar_result = vec![0.0; test_values.len()];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::exp(&test_values, &mut avx2_result);
            ScalarBackend::exp(&test_values, &mut scalar_result);
        }

        for (i, (avx2, scalar)) in avx2_result.iter().zip(scalar_result.iter()).enumerate() {
            let rel_error = if scalar.abs() > 1e-10 {
                (avx2 - scalar).abs() / scalar.abs()
            } else {
                (avx2 - scalar).abs()
            };
            assert!(
                rel_error < 1e-5,
                "exp({}) mismatch: avx2={}, scalar={}, rel_error={}",
                test_values[i],
                avx2,
                scalar,
                rel_error
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_gelu_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut avx2_result = [0.0; 5];
        let mut scalar_result = [0.0; 5];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::gelu(&a, &mut avx2_result);
            ScalarBackend::gelu(&a, &mut scalar_result);
        }

        for (avx2, scalar) in avx2_result.iter().zip(scalar_result.iter()) {
            assert!(
                (avx2 - scalar).abs() < 1e-5,
                "gelu mismatch: avx2={}, scalar={}",
                avx2,
                scalar
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_swish_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [-10.0, -1.0, 0.0, 1.0, 10.0];
        let mut avx2_result = [0.0; 5];
        let mut scalar_result = [0.0; 5];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::swish(&a, &mut avx2_result);
            ScalarBackend::swish(&a, &mut scalar_result);
        }

        for (avx2, scalar) in avx2_result.iter().zip(scalar_result.iter()) {
            assert!(
                (avx2 - scalar).abs() < 1e-5,
                "swish mismatch: avx2={}, scalar={}",
                avx2,
                scalar
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_sub_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::sub(&a, &b, &mut avx2_result);
            ScalarBackend::sub(&a, &b, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_div_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let b = [2.0, 4.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::div(&a, &b, &mut avx2_result);
            ScalarBackend::div(&a, &b, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_scale_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let scalar = 2.5;
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::scale(&a, scalar, &mut avx2_result);
            ScalarBackend::scale(&a, scalar, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_clamp_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0];
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::clamp(&a, 5.0, 30.0, &mut avx2_result);
            ScalarBackend::clamp(&a, 5.0, 30.0, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_fma_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let c = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0];
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::fma(&a, &b, &c, &mut avx2_result);
            ScalarBackend::fma(&a, &b, &c, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_lerp_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let b = [
            100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0,
        ];
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::lerp(&a, &b, 0.25, &mut avx2_result);
            ScalarBackend::lerp(&a, &b, 0.25, &mut scalar_result);
        }

        for (avx2, scalar) in avx2_result.iter().zip(scalar_result.iter()) {
            assert!(
                (avx2 - scalar).abs() < 1e-5,
                "lerp mismatch: avx2={}, scalar={}",
                avx2,
                scalar
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_argmax_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [1.0, 5.0, 3.0, 10.0, 2.0, 8.0, 4.0, 9.0, 6.0];

        // SAFETY: Test code calling backend trait methods marked unsafe
        let avx2_result = unsafe { Avx2Backend::argmax(&a) };
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        let scalar_result = unsafe { ScalarBackend::argmax(&a) };

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_argmin_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [5.0, 1.0, 3.0, 10.0, 2.0, 8.0, 4.0, 9.0, 6.0];

        // SAFETY: Test code calling backend trait methods marked unsafe
        let avx2_result = unsafe { Avx2Backend::argmin(&a) };
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        let scalar_result = unsafe { ScalarBackend::argmin(&a) };

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_sum_kahan_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // SAFETY: Test code calling backend trait methods marked unsafe
        let avx2_result = unsafe { Avx2Backend::sum_kahan(&a) };
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        let scalar_result = unsafe { ScalarBackend::sum_kahan(&a) };

        assert!((avx2_result - scalar_result).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_norm_l1_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0];

        // SAFETY: Test code calling backend trait methods marked unsafe
        let avx2_result = unsafe { Avx2Backend::norm_l1(&a) };
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        let scalar_result = unsafe { ScalarBackend::norm_l1(&a) };

        assert!((avx2_result - scalar_result).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_norm_l2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [3.0, 4.0, 0.0, 0.0, 5.0, 12.0, 0.0, 8.0, 15.0];

        // SAFETY: Test code calling backend trait methods marked unsafe
        let avx2_result = unsafe { Avx2Backend::norm_l2(&a) };
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        let scalar_result = unsafe { ScalarBackend::norm_l2(&a) };

        assert!((avx2_result - scalar_result).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_dot_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        // SAFETY: Test code calling backend trait methods marked unsafe
        let avx2_result = unsafe { Avx2Backend::dot(&a, &b) };
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        let scalar_result = unsafe { ScalarBackend::dot(&a, &b) };

        assert!((avx2_result - scalar_result).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_mul_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
        let b = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::mul(&a, &b, &mut avx2_result);
            ScalarBackend::mul(&a, &b, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_add_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
        let b = [8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5];
        let mut avx2_result = [0.0; 9];
        let mut scalar_result = [0.0; 9];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::add(&a, &b, &mut avx2_result);
            ScalarBackend::add(&a, &b, &mut scalar_result);
        }

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_sum_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // SAFETY: Test code calling backend trait methods marked unsafe
        let avx2_result = unsafe { Avx2Backend::sum(&a) };
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        let scalar_result = unsafe { ScalarBackend::sum(&a) };

        assert!((avx2_result - scalar_result).abs() < 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_max_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [1.0, 5.0, 3.0, 10.0, 2.0, 8.0, 4.0, 9.0, 6.0];

        // SAFETY: Test code calling backend trait methods marked unsafe
        let avx2_result = unsafe { Avx2Backend::max(&a) };
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        let scalar_result = unsafe { ScalarBackend::max(&a) };

        assert_eq!(avx2_result, scalar_result);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_min_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2+FMA");
            return;
        }


        let a = [5.0, 1.0, 3.0, 10.0, 2.0, 8.0, 4.0, 9.0, 6.0];

        // SAFETY: Test code calling backend trait methods marked unsafe
        let avx2_result = unsafe { Avx2Backend::min(&a) };
        // SAFETY: CPU feature verified at runtime, slices bounds-checked
        let scalar_result = unsafe { ScalarBackend::min(&a) };

        assert_eq!(avx2_result, scalar_result);
    }

    // Tests for mathematical operations
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_sqrt_matches_scalar() {

        let a = [4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 144.0];
        let mut avx2_result = vec![0.0; a.len()];
        let mut scalar_result = vec![0.0; a.len()];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::sqrt(&a, &mut avx2_result);
            ScalarBackend::sqrt(&a, &mut scalar_result);
        }

        for i in 0..a.len() {
            assert!(
                (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
                "sqrt({}) mismatch: avx2={}, scalar={}",
                a[i],
                avx2_result[i],
                scalar_result[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_recip_matches_scalar() {

        let a = [1.0, 2.0, 4.0, 5.0, 8.0, 10.0, 16.0, 20.0, 25.0, 32.0];
        let mut avx2_result = vec![0.0; a.len()];
        let mut scalar_result = vec![0.0; a.len()];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::recip(&a, &mut avx2_result);
            ScalarBackend::recip(&a, &mut scalar_result);
        }

        for i in 0..a.len() {
            assert!(
                (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
                "recip({}) mismatch: avx2={}, scalar={}",
                a[i],
                avx2_result[i],
                scalar_result[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_ln_matches_scalar() {

        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
        let mut avx2_result = vec![0.0; a.len()];
        let mut scalar_result = vec![0.0; a.len()];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::ln(&a, &mut avx2_result);
            ScalarBackend::ln(&a, &mut scalar_result);
        }

        for i in 0..a.len() {
            assert!(
                (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
                "ln({}) mismatch: avx2={}, scalar={}",
                a[i],
                avx2_result[i],
                scalar_result[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_log2_matches_scalar() {

        let a = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0];
        let mut avx2_result = vec![0.0; a.len()];
        let mut scalar_result = vec![0.0; a.len()];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::log2(&a, &mut avx2_result);
            ScalarBackend::log2(&a, &mut scalar_result);
        }

        for i in 0..a.len() {
            assert!(
                (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
                "log2({}) mismatch: avx2={}, scalar={}",
                a[i],
                avx2_result[i],
                scalar_result[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_log10_matches_scalar() {

        let a = [1.0, 10.0, 100.0, 1000.0, 2.0, 20.0, 200.0, 5.0, 50.0, 500.0];
        let mut avx2_result = vec![0.0; a.len()];
        let mut scalar_result = vec![0.0; a.len()];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::log10(&a, &mut avx2_result);
            ScalarBackend::log10(&a, &mut scalar_result);
        }

        for i in 0..a.len() {
            assert!(
                (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
                "log10({}) mismatch: avx2={}, scalar={}",
                a[i],
                avx2_result[i],
                scalar_result[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_sin_matches_scalar() {

        use std::f32::consts::PI;
        let a = [
            0.0,
            PI / 6.0,
            PI / 4.0,
            PI / 3.0,
            PI / 2.0,
            PI,
            1.5 * PI,
            2.0 * PI,
            -PI / 4.0,
            -PI / 2.0,
        ];
        let mut avx2_result = vec![0.0; a.len()];
        let mut scalar_result = vec![0.0; a.len()];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::sin(&a, &mut avx2_result);
            ScalarBackend::sin(&a, &mut scalar_result);
        }

        for i in 0..a.len() {
            assert!(
                (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
                "sin({}) mismatch: avx2={}, scalar={}",
                a[i],
                avx2_result[i],
                scalar_result[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_cos_matches_scalar() {

        use std::f32::consts::PI;
        let a = [
            0.0,
            PI / 6.0,
            PI / 4.0,
            PI / 3.0,
            PI / 2.0,
            PI,
            1.5 * PI,
            2.0 * PI,
            -PI / 4.0,
            -PI / 2.0,
        ];
        let mut avx2_result = vec![0.0; a.len()];
        let mut scalar_result = vec![0.0; a.len()];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::cos(&a, &mut avx2_result);
            ScalarBackend::cos(&a, &mut scalar_result);
        }

        for i in 0..a.len() {
            assert!(
                (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
                "cos({}) mismatch: avx2={}, scalar={}",
                a[i],
                avx2_result[i],
                scalar_result[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_tan_matches_scalar() {

        use std::f32::consts::PI;
        let a = [
            0.0,
            PI / 6.0,
            PI / 4.0,
            PI / 3.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
            -2.0,
        ];
        let mut avx2_result = vec![0.0; a.len()];
        let mut scalar_result = vec![0.0; a.len()];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::tan(&a, &mut avx2_result);
            ScalarBackend::tan(&a, &mut scalar_result);
        }

        for i in 0..a.len() {
            assert!(
                (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
                "tan({}) mismatch: avx2={}, scalar={}",
                a[i],
                avx2_result[i],
                scalar_result[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_floor_matches_scalar() {

        let a = [1.1, 2.5, 3.9, -1.1, -2.5, -3.9, 0.1, 0.9, -0.1, -0.9];
        let mut avx2_result = vec![0.0; a.len()];
        let mut scalar_result = vec![0.0; a.len()];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::floor(&a, &mut avx2_result);
            ScalarBackend::floor(&a, &mut scalar_result);
        }

        for i in 0..a.len() {
            assert_eq!(
                avx2_result[i], scalar_result[i],
                "floor({}) mismatch: avx2={}, scalar={}",
                a[i], avx2_result[i], scalar_result[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_ceil_matches_scalar() {

        let a = [1.1, 2.5, 3.9, -1.1, -2.5, -3.9, 0.1, 0.9, -0.1, -0.9];
        let mut avx2_result = vec![0.0; a.len()];
        let mut scalar_result = vec![0.0; a.len()];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::ceil(&a, &mut avx2_result);
            ScalarBackend::ceil(&a, &mut scalar_result);
        }

        for i in 0..a.len() {
            assert_eq!(
                avx2_result[i], scalar_result[i],
                "ceil({}) mismatch: avx2={}, scalar={}",
                a[i], avx2_result[i], scalar_result[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_round_matches_scalar() {

        let a = [1.1, 2.5, 3.9, -1.1, -2.5, -3.9, 0.1, 0.9, -0.1, -0.9];
        let mut avx2_result = vec![0.0; a.len()];
        let mut scalar_result = vec![0.0; a.len()];

        // SAFETY: Test code calling backend trait methods marked unsafe
        unsafe {
            Avx2Backend::round(&a, &mut avx2_result);
            ScalarBackend::round(&a, &mut scalar_result);
        }

        for i in 0..a.len() {
            assert!(
                (avx2_result[i] - scalar_result[i]).abs() < 1e-5,
                "round({}) mismatch: avx2={}, scalar={}",
                a[i],
                avx2_result[i],
                scalar_result[i]
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_norm_linf_matches_scalar() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("Skipping AVX2 test: CPU does not support AVX2");
            return;
        }


        // Test various input sizes
        let test_cases = vec![
            vec![],                                                // empty
            vec![5.0],                                             // single element
            vec![-3.0, 1.0, -4.0, 1.0, 5.0],                       // small vector
            vec![-10.0, 5.0, 3.0, 7.0, -2.0, 8.0, 4.0],            // 7 elements
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],          // 8 elements (aligned)
            vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0], // 9 elements (remainder)
        ];

        for test_vec in test_cases {
            // SAFETY: Test code calling backend trait methods marked unsafe
            let scalar_result = unsafe { ScalarBackend::norm_linf(&test_vec) };
            // SAFETY: CPU feature verified at runtime, slices bounds-checked
            let avx2_result = unsafe { Avx2Backend::norm_linf(&test_vec) };

            assert!(
                (scalar_result - avx2_result).abs() < 1e-5,
                "norm_linf mismatch for {:?}: scalar={}, avx2={}",
                test_vec,
                scalar_result,
                avx2_result
            );
        }
    }

    // =========================================================================
    // Golden Parity Test (PMAT-018): AVX2 vs Scalar Exhaustive Comparison
    // =========================================================================

    /// PMAT-018 Golden Parity: Run 1,000 operations on random vectors
    /// and assert AVX2 and Scalar produce identical results (within FP tolerance).
    ///
    /// This test locks in the numerical behavior of AVX2 optimizations.
    /// If they differ by more than 1e-5 for f32 operations, the optimization
    /// is a "Precision-Sacrificing Conjecture" that must be documented.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_vs_scalar_golden_parity() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("Skipping AVX2 golden parity test: CPU does not support AVX2+FMA");
            return;
        }


        // Simple deterministic pseudo-random generator (xorshift32)
        // Using deterministic seed for reproducibility
        let mut rng_state: u32 = 12345;
        let mut next_rand = || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            // Map to range [-100.0, 100.0]
            (rng_state as f32 / u32::MAX as f32) * 200.0 - 100.0
        };

        const NUM_ITERATIONS: usize = 1000;
        const VECTOR_SIZE: usize = 127; // Non-power-of-2 to stress remainder handling
        const FP_TOLERANCE: f32 = 1e-5;

        let mut total_ops = 0;
        let mut max_diff: f32 = 0.0;

        for iteration in 0..NUM_ITERATIONS {
            // Generate random vectors
            let a: Vec<f32> = (0..VECTOR_SIZE).map(|_| next_rand()).collect();
            let b: Vec<f32> = (0..VECTOR_SIZE).map(|_| next_rand()).collect();
            let mut avx2_result = vec![0.0f32; VECTOR_SIZE];
            let mut scalar_result = vec![0.0f32; VECTOR_SIZE];

            // Test add
            unsafe {
                Avx2Backend::add(&a, &b, &mut avx2_result);
                ScalarBackend::add(&a, &b, &mut scalar_result);
            }
            for (i, (&av, &sc)) in avx2_result.iter().zip(scalar_result.iter()).enumerate() {
                let diff = (av - sc).abs();
                max_diff = max_diff.max(diff);
                assert!(
                    diff < FP_TOLERANCE,
                    "ADD parity fail iter={} idx={}: avx2={} scalar={} diff={}",
                    iteration, i, av, sc, diff
                );
            }
            total_ops += 1;

            // Test mul
            unsafe {
                Avx2Backend::mul(&a, &b, &mut avx2_result);
                ScalarBackend::mul(&a, &b, &mut scalar_result);
            }
            for (i, (&av, &sc)) in avx2_result.iter().zip(scalar_result.iter()).enumerate() {
                let diff = (av - sc).abs();
                max_diff = max_diff.max(diff);
                assert!(
                    diff < FP_TOLERANCE,
                    "MUL parity fail iter={} idx={}: avx2={} scalar={} diff={}",
                    iteration, i, av, sc, diff
                );
            }
            total_ops += 1;

            // Test dot product
            let avx2_dot = unsafe { Avx2Backend::dot(&a, &b) };
            let scalar_dot = unsafe { ScalarBackend::dot(&a, &b) };
            // Dot products accumulate, so tolerance scales with vector size
            let dot_tolerance = FP_TOLERANCE * VECTOR_SIZE as f32 * 100.0;
            let dot_diff = (avx2_dot - scalar_dot).abs();
            max_diff = max_diff.max(dot_diff / (VECTOR_SIZE as f32 * 100.0));
            assert!(
                dot_diff < dot_tolerance,
                "DOT parity fail iter={}: avx2={} scalar={} diff={}",
                iteration, avx2_dot, scalar_dot, dot_diff
            );
            total_ops += 1;

            // Test sum
            let avx2_sum = unsafe { Avx2Backend::sum(&a) };
            let scalar_sum = unsafe { ScalarBackend::sum(&a) };
            let sum_tolerance = FP_TOLERANCE * VECTOR_SIZE as f32 * 100.0;
            let sum_diff = (avx2_sum - scalar_sum).abs();
            assert!(
                sum_diff < sum_tolerance,
                "SUM parity fail iter={}: avx2={} scalar={} diff={}",
                iteration, avx2_sum, scalar_sum, sum_diff
            );
            total_ops += 1;
        }

        eprintln!(
            "Golden Parity PASSED: {} operations, max element diff = {:.2e}",
            total_ops, max_diff
        );
    }
}
