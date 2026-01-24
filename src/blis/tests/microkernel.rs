use super::super::*;

    // ========================================================================
    // Phase 2: Microkernel Tests
    // ========================================================================

    #[test]
    fn test_microkernel_scalar_single_k() {
        // MR=8, NR=6, K=1
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 8x1
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 1x6
        let mut c = vec![0.0; MR * NR]; // 8x6 column-major

        microkernel_scalar(1, &a, &b, &mut c, MR);

        // c[j,i] = a[i] * b[j]
        for j in 0..NR {
            for i in 0..MR {
                let expected = a[i] * b[j];
                assert!(
                    (c[j * MR + i] - expected).abs() < 1e-6,
                    "Mismatch at ({}, {}): {} vs {}",
                    i,
                    j,
                    c[j * MR + i],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_microkernel_scalar_accumulation() {
        let a = vec![1.0; MR * 4]; // 8x4
        let b = vec![1.0; 4 * NR]; // 4x6
        let mut c = vec![0.0; MR * NR];

        microkernel_scalar(4, &a, &b, &mut c, MR);

        // Each output should be 4.0 (sum of 4 ones)
        for val in &c {
            assert!((val - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_microkernel_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }

        let k = 64;
        let a: Vec<f32> = (0..MR * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.01).collect();

        let mut c_scalar = vec![0.0; MR * NR];
        let mut c_avx2 = vec![0.0; MR * NR];

        microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

        unsafe {
            microkernel_8x6_avx2(k, a.as_ptr(), b.as_ptr(), c_avx2.as_mut_ptr(), MR);
        }

        for i in 0..MR * NR {
            let diff = (c_scalar[i] - c_avx2[i]).abs();
            let rel_diff = diff / c_scalar[i].abs().max(1e-10);
            assert!(
                rel_diff < 1e-5,
                "Mismatch at {}: scalar={}, avx2={}, rel_diff={}",
                i,
                c_scalar[i],
                c_avx2[i],
                rel_diff
            );
        }
    }
