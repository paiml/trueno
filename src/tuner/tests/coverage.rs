//! Additional coverage tests for tuner module.

use super::super::*;

#[test]
fn test_tuner_features_validate() {
    let features = TunerFeatures::builder().build();
    assert!(features.validate().is_ok());

    // Test with NaN
    let mut bad_features = features.clone();
    bad_features.model_params_b = f32::NAN;
    assert!(bad_features.validate().is_err());
}

#[test]
fn test_tuner_error_display() {
    let err = TunerError::InvalidFeature("test".to_string());
    assert!(format!("{}", err).contains("Invalid feature"));

    let err = TunerError::InsufficientData(5);
    assert!(format!("{}", err).contains("Insufficient"));

    let err = TunerError::Serialization("test".to_string());
    assert!(format!("{}", err).contains("Serialization"));

    let err = TunerError::ModelNotFound;
    assert!(format!("{}", err).contains("not found"));

    let err = TunerError::PredictionFailed("test".to_string());
    assert!(format!("{}", err).contains("Prediction failed"));
}

#[test]
fn test_throughput_regressor_predict_raw() {
    let regressor = ThroughputRegressor::new();
    let features = TunerFeatures::builder().batch_size(4).build();
    let vec = features.to_vector();
    let raw = regressor.predict_raw(&vec);
    assert!(raw > 0.0);
}

#[test]
fn test_brick_tuner_recommend() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .batch_size(4)
        .build();
    let rec = tuner.recommend(&features);

    assert!(rec.throughput.predicted_tps > 0.0);
    assert!(!rec.suggested_experiments.is_empty());
}

#[test]
fn test_experiment_suggestion_display() {
    let exp = ExperimentSuggestion::IncreaseBatchSize { from: 1, to: 4 };
    assert!(format!("{}", exp).contains("Increase batch size"));

    let exp = ExperimentSuggestion::EnableCudaGraphs;
    assert!(format!("{}", exp).contains("CUDA graphs"));

    let exp = ExperimentSuggestion::TryKernel {
        kernel: KernelType::BatchedQ4K,
    };
    assert!(format!("{}", exp).contains("kernel"));

    let exp = ExperimentSuggestion::ReduceSequenceLength { factor: 0.5 };
    assert!(format!("{}", exp).contains("sequence"));

    let exp = ExperimentSuggestion::EnableMultiKvCache { count: 4 };
    assert!(format!("{}", exp).contains("KV"));
}

#[test]
fn test_tuner_data_collector() {
    let collector = TunerDataCollector::new();
    assert!(collector.is_empty());
    assert_eq!(collector.len(), 0);
    assert!(collector.samples().is_empty());
}

#[test]
fn test_feature_extractor_default() {
    let extractor = FeatureExtractor::new();
    assert!(extractor.hardware.is_none());
}

#[test]
fn test_feature_extractor_debug() {
    let extractor = FeatureExtractor::new();
    let debug_str = format!("{:?}", extractor);
    assert!(debug_str.contains("FeatureExtractor"));
}

#[test]
fn test_chrono_lite_now() {
    let timestamp = super::super::chrono_lite_now();
    let parsed: u64 = timestamp.parse().expect("Should be a number");
    assert!(parsed > 0);
}

#[test]
fn test_pad_right() {
    assert_eq!(super::super::pad_right("test", 10), "test      ");
    assert_eq!(super::super::pad_right("longstring", 5), "longs");
}

#[test]
fn test_validation_infinite_features() {
    let features = TunerFeatures {
        model_params_b: f32::INFINITY,
        ..Default::default()
    };
    let result = features.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Infinite"));
}

#[test]
fn test_validation_out_of_range() {
    let features = TunerFeatures {
        batch_size_norm: 2.0, // Out of [0, 1]
        ..Default::default()
    };
    let result = features.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("outside [0, 1]"));
}

#[test]
fn test_validation_bad_quant_onehot() {
    let features = TunerFeatures {
        quant_type_onehot: [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Sums to 1 but invalid one-hot
        ..Default::default()
    };
    // This should actually pass since sum is 1.0
    assert!(features.validate().is_ok());

    // Now test with sum != 1
    let features2 = TunerFeatures {
        quant_type_onehot: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Sums to 0.5
        ..Default::default()
    };
    let result = features2.validate();
    assert!(result.is_err());
}

#[test]
fn test_validation_bad_kernel_onehot() {
    let mut features = TunerFeatures {
        kernel_type_onehot: [0.0; 16], // All zeros, sum = 0
        ..Default::default()
    };
    // Zero sum is allowed (unspecified kernel)
    assert!(features.validate().is_ok());

    // Sum != 0 and != 1 should fail
    features.kernel_type_onehot[0] = 0.5;
    let result = features.validate();
    assert!(result.is_err());
}

#[test]
fn test_builder_gpu_l2_cache_mb() {
    let features = TunerFeatures::builder()
        .gpu_l2_cache_mb(96.0) // 96MB L2 cache
        .build();
    // Normalized: 96 / 128 = 0.75
    assert!((features.gpu_l2_cache_norm - 0.75).abs() < 0.01);
}

#[test]
fn test_builder_is_zero_copy() {
    let features_enabled = TunerFeatures::builder().is_zero_copy(true).build();
    assert_eq!(features_enabled.is_zero_copy, 1.0);

    let features_disabled = TunerFeatures::builder().is_zero_copy(false).build();
    assert_eq!(features_disabled.is_zero_copy, 0.0);
}

#[test]
fn test_builder_hardware() {
    use crate::hardware::{GpuBackend, GpuCapability};

    let gpu = GpuCapability {
        vendor: "NVIDIA".to_string(),
        model: "Test GPU".to_string(),
        backend: GpuBackend::Cuda,
        compute_capability: Some("8.9".to_string()),
        peak_tflops_fp32: 100.0,
        peak_tflops_tensor: Some(400.0),
        memory_bw_gbps: 1000.0,
        vram_gb: 24.0,
    };

    // Directly test the normalization without HardwareCapability
    let features = TunerFeatures::builder()
        .gpu_mem_bw_gbs(gpu.memory_bw_gbps as f32)
        .gpu_compute_tflops(gpu.peak_tflops_fp32 as f32)
        .build();

    // Memory BW: 1000 / 3000 approx 0.333
    assert!((features.gpu_mem_bw_norm - (1000.0 / 3000.0)).abs() < 0.01);
    // Compute: 100 / 500 = 0.2
    assert!((features.gpu_compute_norm - 0.2).abs() < 0.01);
}

#[test]
fn test_brick_tuner_train() {
    let mut tuner = BrickTuner::new();

    // Create minimal training data
    let data: Vec<(TunerFeatures, f32)> = (0..15)
        .map(|i| {
            let features = TunerFeatures::builder()
                .batch_size((i % 4) as u32 + 1)
                .model_params_b(1.5 + (i as f32) * 0.1)
                .build();
            (features, 100.0 + (i as f32) * 10.0)
        })
        .collect();

    let result = tuner.train(&data);
    assert!(result.is_ok());
    assert_eq!(tuner.sample_count, 15);
}

#[test]
fn test_brick_tuner_train_insufficient_data() {
    let mut tuner = BrickTuner::new();

    // Too few samples
    let data: Vec<(TunerFeatures, f32)> = (0..5)
        .map(|i| {
            let features = TunerFeatures::builder().batch_size(i as u32 + 1).build();
            (features, 100.0)
        })
        .collect();

    let result = tuner.train(&data);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        TunerError::InsufficientData(5)
    ));
}

#[test]
fn test_collector_training_progress() {
    let mut collector = TunerDataCollector::new();

    let (current, required) = collector.training_progress();
    assert_eq!(current, 0);
    assert_eq!(required, TunerDataCollector::MIN_SAMPLES_FOR_TRAINING);

    // Add 5 samples
    for i in 0..5 {
        let features = TunerFeatures::builder()
            .model_params_b(7.0)
            .batch_size((i as u32) + 1)
            .build();
        collector.samples.push(TrainingSample {
            features,
            throughput_tps: 100.0,
            best_kernel: KernelType::TiledQ4K,
            bottleneck: BottleneckClass::MemoryBound,
            timestamp: format!("{}", i),
            hardware_id: "test".to_string(),
        });
    }

    let (current, required) = collector.training_progress();
    assert_eq!(current, 5);
    assert_eq!(required, TunerDataCollector::MIN_SAMPLES_FOR_TRAINING);
}

#[test]
fn test_collector_train_if_ready_not_ready() {
    let collector = TunerDataCollector::new();
    assert!(collector.train_if_ready().is_none());
}

#[test]
fn test_collector_train_if_ready_success() {
    let mut collector = TunerDataCollector::new();

    // Add MIN_SAMPLES_FOR_TRAINING samples (1000) to trigger training
    for i in 0..TunerDataCollector::MIN_SAMPLES_FOR_TRAINING {
        let features = TunerFeatures::builder()
            .model_params_b(1.0 + (i as f32) % 20.0)
            .hidden_dim(2048 + (i as u32) % 4096)
            .batch_size((i as u32) % 16 + 1)
            .quant_type(if i % 2 == 0 {
                QuantType::Q4K
            } else {
                QuantType::Q8_0
            })
            .build();
        collector.samples.push(TrainingSample {
            features,
            throughput_tps: 50.0 + (i as f32) % 200.0,
            best_kernel: KernelType::TiledQ4K,
            bottleneck: BottleneckClass::MemoryBound,
            timestamp: format!("{}", i),
            hardware_id: "test-gpu".to_string(),
        });
    }

    let result = collector.train_if_ready();
    assert!(result.is_some());

    let tuner = result.unwrap();
    // Tuner should have been trained
    assert!(tuner.throughput_sample_count() > 0);
}

#[test]
fn test_collector_bootstrap_from_five_whys() {
    let collector = TunerDataCollector::bootstrap_from_five_whys();

    // Bootstrap returns empty collector for now (TODO: load actual data)
    // But it should still be a valid collector
    // Collector is either empty or has samples - this is always true but exercises the API
    let _ = collector.samples();
    assert!(!collector.is_online_learning_enabled());
}

#[test]
fn test_collector_auto_retrain_not_ready() {
    let mut collector = TunerDataCollector::new();
    let mut tuner = BrickTuner::new();

    // Should return false when not ready to retrain
    assert!(!collector.auto_retrain(&mut tuner));
}

#[test]
fn test_collector_auto_retrain_success() {
    let mut collector = TunerDataCollector::with_online_learning();
    let mut tuner = BrickTuner::new();

    // Add enough samples to trigger retrain
    for i in 0..150 {
        let features = TunerFeatures::builder()
            .model_params_b(1.0 + (i as f32) * 0.1)
            .hidden_dim(2048)
            .batch_size((i as u32) % 16 + 1)
            .quant_type(if i % 2 == 0 {
                QuantType::Q4K
            } else {
                QuantType::Q8_0
            })
            .build();
        collector.samples.push(TrainingSample {
            features,
            throughput_tps: 30.0 + (i as f32) * 2.0,
            best_kernel: KernelType::TiledQ4K,
            bottleneck: BottleneckClass::MemoryBound,
            timestamp: format!("{}", i),
            hardware_id: "auto-retrain-test".to_string(),
        });
    }

    // Force should_retrain to return true by exceeding threshold
    collector.samples_at_last_train = 0;
    collector.retrain_threshold = 50;

    let result = collector.auto_retrain(&mut tuner);
    assert!(result);
}

#[test]
fn test_collector_from_json_valid() {
    // Create a collector with samples, serialize it, then deserialize
    let mut original = TunerDataCollector::new();
    let features = TunerFeatures::builder()
        .model_params_b(7.0)
        .hidden_dim(4096)
        .batch_size(1)
        .build();
    original.samples.push(TrainingSample {
        features,
        throughput_tps: 150.0,
        best_kernel: KernelType::TiledQ4K,
        bottleneck: BottleneckClass::MemoryBound,
        timestamp: "1704067200".to_string(),
        hardware_id: "RTX4090".to_string(),
    });

    // Round-trip through JSON
    let json = original.to_json().unwrap();
    let result = TunerDataCollector::from_json(&json);
    assert!(result.is_ok());

    let collector = result.unwrap();
    assert_eq!(collector.samples().len(), 1);
    assert_eq!(collector.samples()[0].throughput_tps, 150.0);
}

#[test]
fn test_collector_from_json_invalid() {
    let json = "not valid json";
    let result = TunerDataCollector::from_json(json);
    assert!(result.is_err());
}

#[test]
fn test_concept_drift_status_fields() {
    let status = ConceptDriftStatus {
        drift_detected: true,
        staleness_score: 0.75,
        samples_since_training: 100,
        recommend_retrain: true,
        explanation: "High error rate detected".to_string(),
    };

    assert!(status.drift_detected);
    assert_eq!(status.staleness_score, 0.75);
    assert_eq!(status.samples_since_training, 100);
    assert!(status.recommend_retrain);
    assert!(status.explanation.contains("error"));
}

#[test]
fn test_training_stats_all_fields() {
    let stats = TrainingStats {
        total_samples: 500,
        samples_since_training: 50,
        accepted_count: 200,
        rejected_count: 50,
        alternative_count: 100,
        staleness_score: 0.3,
        drift_detected: false,
        online_learning_enabled: true,
    };

    assert_eq!(stats.total_samples, 500);
    assert_eq!(stats.samples_since_training, 50);
    assert_eq!(stats.accepted_count, 200);
    assert_eq!(stats.rejected_count, 50);
    assert_eq!(stats.alternative_count, 100);
    assert!(!stats.drift_detected);
    assert!(stats.online_learning_enabled);
}
