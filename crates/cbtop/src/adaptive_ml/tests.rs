    use super::*;

    /// Generate sinusoidal test samples: `base + sin(i * freq) * amplitude`.
    fn sinusoidal_samples(count: usize, base: f64, freq: f64, amplitude: f64) -> Vec<f64> {
        (0..count)
            .map(|i| base + (i as f64 * freq).sin() * amplitude)
            .collect()
    }

    /// Generate linear test samples: `base + i * step`.
    fn linear_samples(count: usize, base: f64, step: f64) -> Vec<f64> {
        (0..count).map(|i| base + i as f64 * step).collect()
    }

    /// Build a `TimeSeriesFeatures` with given CV and autocorrelation; other fields defaulted.
    fn features_with(cv: f64, autocorr: f64) -> TimeSeriesFeatures {
        TimeSeriesFeatures {
            mean: 100.0,
            std_dev: cv,
            cv,
            skewness: 0.0,
            kurtosis: 0.0,
            autocorr_lag1: autocorr,
            trend_slope: 0.0,
            sample_count: 100,
        }
    }

    /// Train the model `n` times with the given samples.
    fn train_n(ml: &mut AdaptiveThresholdMl, samples: &[f64], is_anomaly: bool, n: usize) {
        for _ in 0..n {
            ml.train(samples, is_anomaly).unwrap();
        }
    }

    #[test]
    fn test_time_series_features() {
        let values = sinusoidal_samples(100, 100.0, 0.1, 10.0);
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
        let features = features_with(8.0, 0.5);

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

        // Low-variance samples (should be normal)
        let normal_values = linear_samples(100, 100.0, 0.01);
        let result = ml.detect_anomaly(&normal_values).unwrap();
        assert!(!result.is_anomaly);

        // High-variance samples (should be anomalous)
        let anomalous_values = sinusoidal_samples(100, 100.0, 0.5, 50.0);
        let result = ml.detect_anomaly(&anomalous_values).unwrap();
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

        let normal = linear_samples(50, 100.0, 0.02);
        train_n(&mut ml, &normal, false, 10);

        let anomalous = linear_samples(50, 100.0, 2.0);
        train_n(&mut ml, &anomalous, true, 5);

        assert!(!ml.learned_workloads().is_empty());

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
        assert_eq!(
            ml.classify_workload(&features_with(5.0, 0.8)),
            WorkloadClass::ComputeBound
        );

        // High CV, low autocorrelation -> MemoryBound
        assert_eq!(
            ml.classify_workload(&features_with(25.0, 0.1)),
            WorkloadClass::MemoryBound
        );
    }

    #[test]
    fn test_drift_detection() {
        let config = MlThresholdConfig::default();
        let mut ml = AdaptiveThresholdMl::new(config);

        let normal = linear_samples(100, 100.0, 0.01);
        train_n(&mut ml, &normal, false, 20);

        // Similar samples (no drift)
        let similar = linear_samples(100, 100.5, 0.01);
        let drift = ml.check_drift(&similar).unwrap();
        assert!(drift.is_none() || drift.unwrap() < 3.0);

        // Very different samples
        let drifted = linear_samples(100, 200.0, 5.0);
        let _drift = ml.check_drift(&drifted).unwrap();
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

        let samples = linear_samples(100, 100.0, 0.05);
        train_n(&mut ml1, &samples, false, 50);

        let state = ml1.export_state();
        assert!(!state.is_empty());

        let mut ml2 = AdaptiveThresholdMl::new(config);
        ml2.import_state(state);
        assert!(!ml2.learned_workloads().is_empty());
    }

    #[test]
    fn test_cold_start_conservative() {
        let config = MlThresholdConfig {
            cold_start_multiplier: 1.5,
            ..Default::default()
        };
        let ml = AdaptiveThresholdMl::new(config);

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
        let ffn_normal = sinusoidal_samples(100, 100.0, 0.2, 18.0);
        train_n(&mut ml, &ffn_normal, false, 50);

        let matmul_normal = sinusoidal_samples(100, 100.0, 0.1, 8.0);
        train_n(&mut ml, &matmul_normal, false, 50);

        // Phase 2: Test that learned thresholds differ by workload
        let ffn_threshold = ml
            .thresholds
            .get(&WorkloadClass::Ffn)
            .map(|t| t.cv_threshold);
        let matmul_threshold = ml
            .thresholds
            .get(&WorkloadClass::Matmul)
            .map(|t| t.cv_threshold);

        if let (Some(ffn_t), Some(matmul_t)) = (ffn_threshold, matmul_threshold) {
            assert!(
                ffn_t != matmul_t,
                "Workload-specific thresholds should differ: FFN={}, Matmul={}",
                ffn_t,
                matmul_t
            );
        }

        // Phase 3: Verify false positive rate is low after training
        let metrics = ml.get_metrics();
        let fpr = metrics.false_positive_rate();

        assert!(
            fpr < 0.20,
            "False positive rate {} should be < 20% with learned thresholds",
            fpr
        );
    }
