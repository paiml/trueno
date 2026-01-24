use super::super::*;

    // ========================================================================
    // Profiler Tests
    // ========================================================================

    #[test]
    fn test_profiler_records_timing() {
        let mut profiler = BlisProfiler::enabled();

        let n = 128;
        let a: Vec<f32> = vec![1.0; n * n];
        let b: Vec<f32> = vec![1.0; n * n];
        let mut c = vec![0.0; n * n];

        gemm_blis(n, n, n, &a, &b, &mut c, Some(&mut profiler)).unwrap();

        assert!(profiler.macro_stats.count > 0);
        assert!(profiler.macro_stats.flops > 0);
        assert!(profiler.micro_stats.count > 0);
    }

    #[test]
    fn test_kaizen_metrics() {
        let mut metrics = KaizenMetrics::default();

        metrics.record(100, 100, 100, std::time::Duration::from_micros(100));

        assert_eq!(metrics.flops, 2_000_000); // 2 * 100^3
        assert!(metrics.gflops() > 0.0);
    }

    // ========================================================================
    // Heijunka Tests
    // ========================================================================

    #[test]
    fn test_heijunka_balanced_partition() {
        let scheduler = HeijunkaScheduler {
            num_threads: 4,
            variance_threshold: 0.05,
        };

        // Use m=288 which divides evenly into 4 blocks of MC=72
        let partitions = scheduler.partition_m(288, MC);

        // Should have 4 partitions
        assert_eq!(partitions.len(), 4);

        // Each partition should be exactly equal (72 rows each)
        let sizes: Vec<usize> = partitions.iter().map(|r| r.len()).collect();
        let avg = sizes.iter().sum::<usize>() as f32 / sizes.len() as f32;

        for size in &sizes {
            let variance = ((*size as f32 - avg) / avg).abs();
            assert!(variance < 0.01, "Partition variance too high: {}", variance);
        }

        // Also test uneven case - should still work
        let partitions_uneven = scheduler.partition_m(256, MC);
        assert_eq!(partitions_uneven.len(), 4);
        let total: usize = partitions_uneven.iter().map(|r| r.len()).sum();
        assert_eq!(total, 256); // All rows covered
    }
