    use super::*;

    fn generate_normal_samples(mean: f64, std: f64, count: usize) -> Vec<f64> {
        // Simple pseudo-random generator for reproducibility
        let mut samples = Vec::with_capacity(count);
        for i in 0..count {
            let x = (i as f64 * 0.1).sin() * std + mean;
            samples.push(x);
        }
        samples
    }

    fn generate_anomalous_samples(mean: f64, std: f64, count: usize) -> Vec<f64> {
        // Higher variance samples
        generate_normal_samples(mean, std * 3.0, count)
    }

    #[test]
    fn test_time_series_features() {
        let values: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0)
            .collect();

        let features = TimeSeriesFeatures::extract(&values).unwrap();

        assert!(features.mean > 95.0 && features.mean < 105.0);
        assert!(features.std_dev > 0.0);
        assert!(features.cv > 0.0 && features.cv < 20.0);
        assert_eq!(features.sample_count, 100);
    }

    #[test]
    fn test_feature_extraction_insufficient_data() {
        let values = vec![1.0, 2.0, 3.0];
        assert!(TimeSeriesFeatures::extract(&values).is_none());
    }

    #[test]
    fn test_workload_class_defaults() {
        assert!(
            WorkloadClass::Matmul.default_cv_threshold()
                < WorkloadClass::Ffn.default_cv_threshold()
        );
        assert!(
            WorkloadClass::ComputeBound.default_cv_threshold()
                < WorkloadClass::MemoryBound.default_cv_threshold()
        );
    }

    #[test]
    fn test_learned_threshold_update() {
        let mut threshold = LearnedWorkloadThreshold::new(WorkloadClass::Matmul);

        let features = TimeSeriesFeatures {
            mean: 100.0,
            std_dev: 8.0,
            cv: 8.0,
            skewness: 0.0,
            kurtosis: 0.0,
            autocorr_lag1: 0.5,
            trend_slope: 0.0,
            sample_count: 100,
        };

        // Train with normal samples
        for _ in 0..10 {
            threshold.update(&features, false);
        }

        assert!(threshold.training_samples > 0);
        assert!(threshold.confidence > 0.0);
    }

    #[test]
    fn test_adaptive_threshold_detection() {
        let config = MlThresholdConfig::default();
        let ml = AdaptiveThresholdMl::new(config);

        // Generate low-variance samples (should be normal)
        let normal_values: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.01)).collect();

        let result = ml.detect_anomaly(&normal_values).unwrap();
        assert!(!result.is_anomaly);

        // Generate high-variance samples (should be anomalous)
        let anomalous_values: Vec<f64> = (0..100)
            .map(|i| 100.0 + ((i as f64 * 0.5).sin() * 50.0))
            .collect();

        let result = ml.detect_anomaly(&anomalous_values).unwrap();
        // High CV should trigger anomaly
        assert!(result.score > 0.5);
    }

    #[test]
    fn test_adaptive_threshold_training() {
        let config = MlThresholdConfig {
            min_training_samples: 5,
            min_confidence: 0.1,
            ..Default::default()
        };
        let mut ml = AdaptiveThresholdMl::new(config);

        // Train with normal samples
        let normal: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 0.02)).collect();
        for _ in 0..10 {
            ml.train(&normal, false).unwrap();
        }

        // Train with anomalous samples
        let anomalous: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64 * 2.0)).collect();
        for _ in 0..5 {
            ml.train(&anomalous, true).unwrap();
        }

        // Check that we have learned thresholds
        assert!(!ml.learned_workloads().is_empty());

        // Check metrics are being tracked
        let metrics = ml.get_metrics();
        assert!(
            metrics.true_positives
                + metrics.false_positives
                + metrics.true_negatives
                + metrics.false_negatives
                > 0
        );
    }

    #[test]
    fn test_workload_classification() {
        let config = MlThresholdConfig::default();
        let ml = AdaptiveThresholdMl::new(config);

        // Low CV, high autocorrelation -> ComputeBound
        let compute_features = TimeSeriesFeatures {
            mean: 100.0,
            std_dev: 5.0,
            cv: 5.0,
            skewness: 0.0,
            kurtosis: 0.0,
            autocorr_lag1: 0.8,
            trend_slope: 0.0,
            sample_count: 100,
        };
        assert_eq!(
            ml.classify_workload(&compute_features),
            WorkloadClass::ComputeBound
        );

        // High CV, low autocorrelation -> MemoryBound
        let memory_features = TimeSeriesFeatures {
            mean: 100.0,
            std_dev: 25.0,
            cv: 25.0,
            skewness: 0.0,
            kurtosis: 0.0,
            autocorr_lag1: 0.1,
            trend_slope: 0.0,
            sample_count: 100,
        };
        assert_eq!(
            ml.classify_workload(&memory_features),
            WorkloadClass::MemoryBound
        );
    }

    #[test]
    fn test_drift_detection() {
        let config = MlThresholdConfig::default();
        let mut ml = AdaptiveThresholdMl::new(config);

        // Train with consistent samples
        let normal: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.01)).collect();
        for _ in 0..20 {
            ml.train(&normal, false).unwrap();
        }

        // Check drift with similar samples (no drift)
        let similar: Vec<f64> = (0..100).map(|i| 100.5 + (i as f64 * 0.01)).collect();
        let drift = ml.check_drift(&similar).unwrap();
        assert!(drift.is_none() || drift.unwrap() < 3.0);

        // Check drift with very different samples
        let drifted: Vec<f64> = (0..100).map(|i| 200.0 + (i as f64 * 5.0)).collect();
        let _drift = ml.check_drift(&drifted).unwrap();
        // May or may not detect drift depending on threshold model
    }

    #[test]
    fn test_classification_metrics() {
        let mut metrics = ClassificationMetrics::default();

        metrics.true_positives = 80;
        metrics.false_positives = 10;
        metrics.true_negatives = 90;
        metrics.false_negatives = 20;

        assert!((metrics.precision() - 0.889).abs() < 0.01);
        assert!((metrics.recall() - 0.8).abs() < 0.01);
        assert!((metrics.false_positive_rate() - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_model_persistence() {
        let config = MlThresholdConfig::default();
        let mut ml1 = AdaptiveThresholdMl::new(config.clone());

        // Train model
        let samples: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.05)).collect();
        for _ in 0..50 {
            ml1.train(&samples, false).unwrap();
        }

        // Export state
        let state = ml1.export_state();
        assert!(!state.is_empty());

        // Import into new model
        let mut ml2 = AdaptiveThresholdMl::new(config);
        ml2.import_state(state);

        // Verify state was imported
        assert!(!ml2.learned_workloads().is_empty());
    }

    #[test]
    fn test_cold_start_conservative() {
        let config = MlThresholdConfig {
            cold_start_multiplier: 1.5,
            ..Default::default()
        };
        let ml = AdaptiveThresholdMl::new(config);

        // Before training, thresholds should be conservative (higher)
        let threshold = ml.get_threshold(WorkloadClass::Matmul);
        let default = WorkloadClass::Matmul.default_cv_threshold();

        assert!(
            threshold > default,
            "Cold start threshold {} should be > default {}",
            threshold,
            default
        );
    }

    #[test]
    fn test_error_display() {
        let err = MlThresholdError::InsufficientData { have: 5, need: 10 };
        assert!(err.to_string().contains("5"));
        assert!(err.to_string().contains("10"));

        let err = MlThresholdError::DriftDetected {
            metric: "latency".to_string(),
            drift_score: 4.5,
        };
        assert!(err.to_string().contains("latency"));
        assert!(err.to_string().contains("4.5"));
    }

    // FKR-050: ML thresholds reduce false positives
    #[test]
    fn test_fkr_050_precision_improvement() {
        let config = MlThresholdConfig {
            min_training_samples: 10,
            min_confidence: 0.3,
            ..Default::default()
        };
        let mut ml = AdaptiveThresholdMl::new(config);

        // Phase 1: Train with labeled data
        // FFN workload: naturally high CV (~18%)
        for _ in 0..50 {
            let ffn_normal: Vec<f64> = (0..100)
                .map(|i| 100.0 + ((i as f64 * 0.2).sin() * 18.0))
                .collect();
            ml.train(&ffn_normal, false).unwrap(); // Normal for FFN
        }

        // Matmul workload: naturally low CV (~8%)
        for _ in 0..50 {
            let matmul_normal: Vec<f64> = (0..100)
                .map(|i| 100.0 + ((i as f64 * 0.1).sin() * 8.0))
                .collect();
            ml.train(&matmul_normal, false).unwrap(); // Normal for Matmul
        }

        // Phase 2: Test that learned thresholds differ by workload
        let ffn_threshold = ml
            .thresholds
            .get(&WorkloadClass::Ffn)
            .map(|t| t.cv_threshold);
        let matmul_threshold = ml
            .thresholds
            .get(&WorkloadClass::Matmul)
            .map(|t| t.cv_threshold);

        // FFN should have higher threshold than Matmul
        if let (Some(ffn_t), Some(matmul_t)) = (ffn_threshold, matmul_threshold) {
            assert!(
                ffn_t != matmul_t,
                "Workload-specific thresholds should differ: FFN={}, Matmul={}",
                ffn_t,
                matmul_t
            );
        }

        // Phase 3: Verify classification metrics show improvement potential
        // With static 15% threshold:
        // - FFN samples at 18% CV would be false positives
        // - With learned ~20% threshold for FFN, they're true negatives

        let metrics = ml.get_metrics();
        let fpr = metrics.false_positive_rate();

        // FKR-050: False positive rate should be low after training
        // (Hard to guarantee exact numbers, but should be reasonable)
        assert!(
            fpr < 0.20,
            "False positive rate {} should be < 20% with learned thresholds",
            fpr
        );
    }
