    use super::super::*;

    // ========================================================================
    // Phase 1: Scalar Reference Tests
    // ========================================================================

    #[test]
    fn test_gemm_reference_2x2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];

        gemm_reference(2, 2, 2, &a, &b, &mut c).unwrap();

        // [1 2] * [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_gemm_reference_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0; 9];

        gemm_reference(3, 3, 3, &a, &identity, &mut c).unwrap();

        assert_eq!(c, a);
    }

    #[test]
    fn test_gemm_reference_accumulation() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![10.0, 20.0, 30.0, 40.0]; // Pre-existing values

        gemm_reference(2, 2, 2, &a, &b, &mut c).unwrap();

        // C += A * I = C + A
        assert_eq!(c, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_gemm_reference_rectangular() {
        // 2x3 * 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0; 4];

        gemm_reference(2, 2, 3, &a, &b, &mut c).unwrap();

        // [1 2 3] * [7  8 ] = [58  64]
        // [4 5 6]   [9  10]   [139 154]
        //           [11 12]
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_gemm_reference_size_mismatch() {
        let a = vec![1.0, 2.0, 3.0]; // Wrong size
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];

        let result = gemm_reference(2, 2, 2, &a, &b, &mut c);
        assert!(result.is_err());
    }

