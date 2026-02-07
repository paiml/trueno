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

// =========================================================================
// BrickProfiler::tokens_per_sec + tuner integration (tuner/mod.rs coverage)
// =========================================================================

#[test]
fn test_brick_profiler_tokens_per_sec_no_data() {
    use crate::brick::BrickProfiler;
    let profiler = BrickProfiler::new();
    // No tokens processed, no time elapsed -> None
    assert!(profiler.tokens_per_sec().is_none());
}

#[test]
fn test_brick_profiler_tokens_per_sec_with_data() {
    use crate::brick::BrickProfiler;
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    let timer = profiler.start("test_brick");
    std::thread::sleep(std::time::Duration::from_millis(1));
    profiler.stop(timer, 100);

    let tps = profiler.tokens_per_sec();
    // If total_ns > 0 and total_tokens > 0, should return Some
    if profiler.total_ns() > 0 && profiler.total_tokens() > 0 {
        assert!(tps.is_some());
        assert!(tps.unwrap() > 0.0);
    }
}

#[test]
fn test_brick_profiler_get_tuner_recommendations_disabled() {
    use crate::brick::BrickProfiler;
    let profiler = BrickProfiler::new();
    // Profiler is disabled by default
    assert!(!profiler.is_enabled());
    let config = RunConfig::default();
    let result = profiler.get_tuner_recommendations(&config);
    assert!(result.is_none(), "Disabled profiler should return None");
}

#[test]
fn test_brick_profiler_get_tuner_recommendations_enabled() {
    use crate::brick::BrickProfiler;
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    let config = RunConfig::default();
    let result = profiler.get_tuner_recommendations(&config);
    assert!(result.is_some(), "Enabled profiler should return Some");
    let rec = result.unwrap();
    assert!(rec.throughput.predicted_tps > 0.0);
    assert!(rec.confidence_overall > 0.0);
}

#[test]
fn test_brick_profiler_print_tuner_recommendations_disabled() {
    use crate::brick::BrickProfiler;
    let profiler = BrickProfiler::new();
    let config = RunConfig::default();
    // Should not panic, just print "not available"
    profiler.print_tuner_recommendations(&config);
}

#[test]
fn test_brick_profiler_print_tuner_recommendations_enabled() {
    use crate::brick::BrickProfiler;
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    let config = RunConfig::default();
    // Should print recommendation without panic
    profiler.print_tuner_recommendations(&config);
}

// =========================================================================
// BrickTuner: suggest_experiments coverage (various bottleneck types)
// =========================================================================

#[test]
fn test_suggest_experiments_memory_bound_small_batch() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(1)
        .build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::MemoryBound,
        confidence: 0.85,
        explanation: "Memory bound".to_string(),
        recommended_action: "Increase batch size".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck);
    // Should suggest increasing batch size since batch_size < 8
    assert!(
        suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::IncreaseBatchSize { .. })),
        "Should suggest increasing batch size for memory-bound with small batch"
    );
    // Should suggest trying BatchedQ4K kernel
    assert!(
        suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::TryKernel { .. })),
        "Should suggest trying a kernel"
    );
}

#[test]
fn test_suggest_experiments_memory_bound_large_batch() {
    let tuner = BrickTuner::new();
    // batch_size_norm = 8/64 = 0.125, so batch_size = round(0.125*64) = 8
    let features = TunerFeatures::builder()
        .batch_size(8)
        .build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::MemoryBound,
        confidence: 0.85,
        explanation: "Memory bound".to_string(),
        recommended_action: "Increase batch size".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck);
    // batch_size >= 8, so no batch size increase suggestion
    assert!(
        !suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::IncreaseBatchSize { .. })),
        "Should NOT suggest increasing batch size when already >= 8"
    );
    // But should suggest EnableMultiKvCache since batch_size > 1
    assert!(
        suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::EnableMultiKvCache { .. })),
        "Should suggest multi-KV cache for batch_size > 1"
    );
}

#[test]
fn test_suggest_experiments_launch_bound_no_cuda_graphs() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(1)
        .cuda_graphs(false)
        .build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::LaunchBound,
        confidence: 0.75,
        explanation: "Launch bound".to_string(),
        recommended_action: "Enable CUDA graphs".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck);
    assert!(
        suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::EnableCudaGraphs)),
        "Should suggest enabling CUDA graphs"
    );
    assert!(
        suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::TryKernel { kernel: KernelType::FusedRmsNormQ4K })),
        "Should suggest fused kernel"
    );
}

#[test]
fn test_suggest_experiments_launch_bound_with_cuda_graphs() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(1)
        .cuda_graphs(true)
        .build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::LaunchBound,
        confidence: 0.75,
        explanation: "Launch bound".to_string(),
        recommended_action: "Enable CUDA graphs".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck);
    // Should NOT suggest CUDA graphs since already enabled
    assert!(
        !suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::EnableCudaGraphs)),
        "Should NOT suggest CUDA graphs when already enabled"
    );
}

#[test]
fn test_suggest_experiments_attention_bound() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(1)
        .seq_len(1024)
        .build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::AttentionBound,
        confidence: 0.80,
        explanation: "Attention bound".to_string(),
        recommended_action: "Use Flash Decoding".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck);
    assert!(
        suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::TryKernel { kernel: KernelType::BatchedAttention })),
        "Should suggest batched attention kernel"
    );
    assert!(
        suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::ReduceSequenceLength { .. })),
        "Should suggest reducing sequence length"
    );
}

#[test]
fn test_suggest_experiments_compute_bound() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(1)
        .build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::ComputeBound,
        confidence: 0.70,
        explanation: "Compute bound".to_string(),
        recommended_action: "Check for redundant computation".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck);
    // ComputeBound falls into the default arm with batch_size < 4
    assert!(
        suggestions.iter().any(|s| matches!(s, ExperimentSuggestion::IncreaseBatchSize { .. })),
        "Default arm should suggest increasing batch size when < 4"
    );
}

#[test]
fn test_suggest_experiments_unknown_large_batch() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(4)
        .build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::Unknown,
        confidence: 0.50,
        explanation: "Unknown".to_string(),
        recommended_action: "Run profiling".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck);
    // batch_size >= 4, so default arm should NOT suggest increasing
    assert!(
        suggestions.is_empty(),
        "No suggestions when unknown bottleneck and batch_size >= 4"
    );
}

// =========================================================================
// BrickTuner: render_panel, render_compact, render_comparison
// =========================================================================

#[test]
fn test_brick_tuner_render_panel() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(2)
        .model_params_b(1.5)
        .build();
    let rec = tuner.recommend(&features);
    let lines = tuner.render_panel(&rec);
    assert!(!lines.is_empty(), "Panel should have lines");
    // Should contain version info
    assert!(lines[0].contains("BrickTuner"), "First line should mention BrickTuner");
}

#[test]
fn test_brick_tuner_render_panel_few_suggestions() {
    let tuner = BrickTuner::new();
    let _features = TunerFeatures::builder()
        .batch_size(4)
        .build();

    let bottleneck = BottleneckPrediction {
        class: BottleneckClass::Unknown,
        confidence: 0.50,
        explanation: "Unknown".to_string(),
        recommended_action: "Run profiling".to_string(),
    };

    // Create a recommendation with zero suggestions
    let rec = TunerRecommendation {
        throughput: ThroughputPrediction {
            predicted_tps: 100.0,
            confidence: 0.85,
            top_features: vec![],
        },
        kernel: KernelRecommendation {
            top_kernel: KernelType::TiledQ4K,
            confidence: 0.90,
            alternatives: vec![],
        },
        bottleneck,
        model_version: "1.0.0".to_string(),
        confidence_overall: 0.75,
        suggested_experiments: vec![], // No suggestions
    };

    let lines = tuner.render_panel(&rec);
    // Should pad to 3 empty suggestion lines
    assert!(lines.len() >= 10, "Panel should have padding for missing suggestions");
}

#[test]
fn test_brick_tuner_render_compact() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(2)
        .model_params_b(1.5)
        .build();
    let rec = tuner.recommend(&features);
    let compact = tuner.render_compact(&rec);
    assert!(compact.contains("Tuner:"), "Compact should start with 'Tuner:'");
    assert!(compact.contains("tok/s"), "Compact should mention tok/s");
}

#[test]
fn test_brick_tuner_render_comparison_excellent() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(2)
        .model_params_b(1.5)
        .build();
    let rec = tuner.recommend(&features);
    // Give actual_tps close to predicted for "Excellent"
    let actual_tps = rec.throughput.predicted_tps * 0.98;
    let comparison = tuner.render_comparison(&rec, actual_tps);
    assert_eq!(comparison.len(), 2);
    assert!(comparison[0].contains("Predicted"));
    assert!(comparison[0].contains("Actual"));
}

#[test]
fn test_brick_tuner_render_comparison_poor() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(2)
        .model_params_b(1.5)
        .build();
    let rec = tuner.recommend(&features);
    // Give very different actual_tps for "Poor"
    let actual_tps = rec.throughput.predicted_tps * 0.5;
    let comparison = tuner.render_comparison(&rec, actual_tps);
    assert_eq!(comparison.len(), 2);
    assert!(comparison[1].contains("Poor"));
}

#[test]
fn test_brick_tuner_render_comparison_zero_actual() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(2)
        .build();
    let rec = tuner.recommend(&features);
    let comparison = tuner.render_comparison(&rec, 0.0);
    assert_eq!(comparison.len(), 2);
    // With actual=0, error_pct=0.0, so "Excellent"
    assert!(comparison[1].contains("Excellent"));
}

// =========================================================================
// BrickTuner: JSON serialization round-trip
// =========================================================================

#[test]
fn test_brick_tuner_json_roundtrip() {
    let tuner = BrickTuner::new();
    let json = tuner.to_json().expect("Serialization should succeed");
    let loaded = BrickTuner::from_json(&json).expect("Deserialization should succeed");
    assert_eq!(tuner.version(), loaded.version());
    assert_eq!(tuner.throughput_mape(), loaded.throughput_mape());
}

#[test]
fn test_brick_tuner_from_json_invalid() {
    let result = BrickTuner::from_json("not valid json");
    assert!(result.is_err());
}

// =========================================================================
// BrickTuner: APR save/load round-trip
// =========================================================================

#[test]
fn test_brick_tuner_apr_roundtrip() {
    let tuner = BrickTuner::new();
    let dir = std::env::temp_dir().join("trueno_test_apr_roundtrip");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_tuner.apr");

    tuner.save_apr(&path).expect("Save should succeed");
    let loaded = BrickTuner::load_apr(&path).expect("Load should succeed");
    assert_eq!(tuner.version(), loaded.version());
    assert_eq!(tuner.sample_count, loaded.sample_count);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_brick_tuner_load_apr_bad_magic() {
    let dir = std::env::temp_dir().join("trueno_test_bad_magic");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("bad_magic.apr");

    // Write file with wrong magic bytes
    std::fs::write(&path, b"BAD1xxxxxxxxxxxx").expect("write");
    let result = BrickTuner::load_apr(&path);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("Invalid") || err_msg.contains("magic"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_brick_tuner_load_apr_crc_mismatch() {
    let tuner = BrickTuner::new();
    let dir = std::env::temp_dir().join("trueno_test_crc_mismatch");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("bad_crc.apr");

    tuner.save_apr(&path).expect("Save should succeed");

    // Corrupt one byte in the file (after the magic and length)
    let mut data = std::fs::read(&path).expect("read");
    if data.len() > 12 {
        data[12] ^= 0xFF; // Flip bits in JSON payload
    }
    std::fs::write(&path, &data).expect("write");

    let result = BrickTuner::load_apr(&path);
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================================
// TunerDataCollector: APR save/load round-trip
// =========================================================================

#[test]
fn test_data_collector_apr_roundtrip() {
    let mut collector = TunerDataCollector::new();
    let features = TunerFeatures::builder()
        .model_params_b(7.0)
        .batch_size(4)
        .build();
    collector.samples.push(TrainingSample {
        features,
        throughput_tps: 200.0,
        best_kernel: KernelType::VectorizedQ4K,
        bottleneck: BottleneckClass::MemoryBound,
        timestamp: "12345".to_string(),
        hardware_id: "test-hw".to_string(),
    });

    let dir = std::env::temp_dir().join("trueno_test_collector_apr");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("test_collector.apr");

    collector.save_apr(&path).expect("Save should succeed");
    let loaded = TunerDataCollector::load_apr(&path).expect("Load should succeed");
    assert_eq!(loaded.samples().len(), 1);
    assert_eq!(loaded.samples()[0].throughput_tps, 200.0);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_data_collector_load_apr_bad_magic() {
    let dir = std::env::temp_dir().join("trueno_test_collector_bad_magic");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("bad.apr");
    std::fs::write(&path, b"XXXXABCD").expect("write");

    let result = TunerDataCollector::load_apr(&path);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("APR2"));

    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================================
// TunerDataCollector: online learning and concept drift
// =========================================================================

#[test]
fn test_collector_record_feedback() {
    let mut collector = TunerDataCollector::new();
    collector.record_feedback(0, UserFeedback::Accepted);
    collector.record_feedback(2, UserFeedback::Rejected);

    assert_eq!(collector.get_feedback(0), UserFeedback::Accepted);
    assert_eq!(collector.get_feedback(1), UserFeedback::None); // Filled with None
    assert_eq!(collector.get_feedback(2), UserFeedback::Rejected);
    assert_eq!(collector.get_feedback(100), UserFeedback::None); // Out of range
}

#[test]
fn test_collector_record_prediction_error_disabled() {
    let mut collector = TunerDataCollector::new();
    // Online learning disabled by default
    collector.record_prediction_error(100.0, 80.0);
    // Error window should remain empty
    assert!(collector.error_window.is_empty());
}

#[test]
fn test_collector_record_prediction_error_enabled() {
    let mut collector = TunerDataCollector::with_online_learning();
    collector.record_prediction_error(100.0, 80.0);
    assert_eq!(collector.error_window.len(), 1);
    // Error = |100-80|/80 = 0.25
    assert!((collector.error_window[0] - 0.25).abs() < 0.01);
}

#[test]
fn test_collector_record_prediction_error_zero_actual() {
    let mut collector = TunerDataCollector::with_online_learning();
    collector.record_prediction_error(100.0, 0.0);
    assert_eq!(collector.error_window.len(), 1);
    // When actual=0, error should be 1.0
    assert_eq!(collector.error_window[0], 1.0);
}

#[test]
fn test_collector_record_prediction_error_window_trimming() {
    let mut collector = TunerDataCollector::with_online_learning();
    // Default window size is 50
    for i in 0..60 {
        collector.record_prediction_error(100.0 + i as f32, 100.0);
    }
    assert_eq!(collector.error_window.len(), 50, "Window should be trimmed to max size");
}

#[test]
fn test_collector_detect_concept_drift_insufficient_data() {
    let collector = TunerDataCollector::new();
    let status = collector.detect_concept_drift();
    assert!(!status.drift_detected);
    assert!(!status.recommend_retrain);
    assert!(status.explanation.contains("Insufficient"));
}

#[test]
fn test_collector_detect_concept_drift_with_high_error() {
    let mut collector = TunerDataCollector::with_online_learning();
    // Add enough errors that exceed threshold (0.15)
    for _ in 0..20 {
        collector.error_window.push(0.30); // 30% error > 15% threshold
    }
    let status = collector.detect_concept_drift();
    assert!(status.drift_detected);
    assert!(status.recommend_retrain);
}

#[test]
fn test_collector_detect_concept_drift_fresh_model() {
    let mut collector = TunerDataCollector::with_online_learning();
    // Add low errors
    for _ in 0..20 {
        collector.error_window.push(0.05); // 5% error < 15% threshold
    }
    collector.samples_at_last_train = collector.samples.len(); // Just trained
    let status = collector.detect_concept_drift();
    assert!(!status.drift_detected);
    assert!(!status.recommend_retrain);
    assert!(status.explanation.contains("fresh"));
}

#[test]
fn test_collector_detect_concept_drift_stale() {
    let mut collector = TunerDataCollector::with_online_learning();
    // Low errors but high staleness
    for _ in 0..20 {
        collector.error_window.push(0.05);
    }
    // Add many samples since last training
    for i in 0..120 {
        let features = TunerFeatures::builder()
            .model_params_b(1.0 + (i as f32) * 0.1)
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
    collector.samples_at_last_train = 0; // Trained at the beginning

    let status = collector.detect_concept_drift();
    assert!(status.staleness_score > 0.8);
    assert!(status.recommend_retrain);
    assert!(status.explanation.contains("stale") || status.explanation.contains("Model stale"));
}

#[test]
fn test_collector_should_retrain_disabled() {
    let collector = TunerDataCollector::new();
    assert!(!collector.should_retrain(), "Should not retrain when online learning disabled");
}

#[test]
fn test_collector_mark_trained() {
    let mut collector = TunerDataCollector::with_online_learning();
    for _ in 0..20 {
        collector.error_window.push(0.20);
    }
    collector.mark_trained();
    assert!(collector.error_window.is_empty(), "Error window should be cleared after training");
    assert_eq!(collector.samples_at_last_train, collector.samples.len());
}

#[test]
fn test_collector_merge() {
    let mut collector1 = TunerDataCollector::new();
    let mut collector2 = TunerDataCollector::new();

    let features = TunerFeatures::builder().build();
    collector1.samples.push(TrainingSample {
        features: features.clone(),
        throughput_tps: 100.0,
        best_kernel: KernelType::TiledQ4K,
        bottleneck: BottleneckClass::MemoryBound,
        timestamp: "1".to_string(),
        hardware_id: "hw1".to_string(),
    });
    collector2.samples.push(TrainingSample {
        features,
        throughput_tps: 200.0,
        best_kernel: KernelType::VectorizedQ4K,
        bottleneck: BottleneckClass::ComputeBound,
        timestamp: "2".to_string(),
        hardware_id: "hw2".to_string(),
    });

    collector1.merge(&collector2);
    assert_eq!(collector1.samples.len(), 2);
}

#[test]
fn test_collector_training_stats() {
    let mut collector = TunerDataCollector::with_online_learning();
    collector.record_feedback(0, UserFeedback::Accepted);
    collector.record_feedback(1, UserFeedback::Rejected);
    collector.record_feedback(2, UserFeedback::Alternative);

    let stats = collector.training_stats();
    assert_eq!(stats.accepted_count, 1);
    assert_eq!(stats.rejected_count, 1);
    assert_eq!(stats.alternative_count, 1);
    assert!(stats.online_learning_enabled);
}

#[test]
fn test_collector_enable_disable_online_learning() {
    let mut collector = TunerDataCollector::new();
    assert!(!collector.is_online_learning_enabled());
    collector.enable_online_learning();
    assert!(collector.is_online_learning_enabled());
    collector.disable_online_learning();
    assert!(!collector.is_online_learning_enabled());
}

// =========================================================================
// TunerError variants coverage
// =========================================================================

#[test]
fn test_tuner_error_training_failed() {
    let err = TunerError::TrainingFailed("gradient explosion".to_string());
    assert!(format!("{}", err).contains("Training failed"));
    assert!(format!("{}", err).contains("gradient explosion"));
}

#[test]
fn test_tuner_error_io() {
    let err = TunerError::Io("file not found".to_string());
    assert!(format!("{}", err).contains("I/O error"));
}

#[test]
fn test_tuner_error_invalid_format() {
    let err = TunerError::InvalidFormat("bad magic".to_string());
    assert!(format!("{}", err).contains("Invalid format"));
}

#[test]
fn test_tuner_error_is_std_error() {
    let err = TunerError::ModelNotFound;
    // Verify it implements std::error::Error
    let _: &dyn std::error::Error = &err;
}

// =========================================================================
// Types coverage: QuantType, KernelType, BottleneckClass
// =========================================================================

#[test]
fn test_quant_type_to_index_all_variants() {
    assert_eq!(QuantType::Q4_0.to_index(), 0);
    assert_eq!(QuantType::Q4_1.to_index(), 1);
    assert_eq!(QuantType::Q4K.to_index(), 2);
    assert_eq!(QuantType::Q5K.to_index(), 3);
    assert_eq!(QuantType::Q6K.to_index(), 4);
    assert_eq!(QuantType::Q8_0.to_index(), 5);
    assert_eq!(QuantType::F16.to_index(), 6);
    assert_eq!(QuantType::F32.to_index(), 7);
}

#[test]
fn test_quant_type_bytes_per_param() {
    assert!((QuantType::Q4_0.bytes_per_param() - 0.5625).abs() < 0.001);
    assert!((QuantType::Q4_1.bytes_per_param() - 0.5625).abs() < 0.001);
    assert!((QuantType::Q4K.bytes_per_param() - 0.5625).abs() < 0.001);
    assert!((QuantType::Q5K.bytes_per_param() - 0.6875).abs() < 0.001);
    assert!((QuantType::Q6K.bytes_per_param() - 0.8125).abs() < 0.001);
    assert!((QuantType::Q8_0.bytes_per_param() - 1.0).abs() < 0.001);
    assert!((QuantType::F16.bytes_per_param() - 2.0).abs() < 0.001);
    assert!((QuantType::F32.bytes_per_param() - 4.0).abs() < 0.001);
}

#[test]
fn test_kernel_type_from_index_roundtrip() {
    for idx in 0..15 {
        let kt = KernelType::from_index(idx);
        assert_eq!(kt.to_index(), idx, "Round-trip failed for index {}", idx);
    }
    // Out of range should return Unknown
    assert_eq!(KernelType::from_index(100), KernelType::Unknown);
    assert_eq!(KernelType::from_index(15), KernelType::Unknown);
}

#[test]
fn test_bottleneck_class_from_brick_bottleneck() {
    use crate::brick::BrickBottleneck;
    assert_eq!(
        BottleneckClass::from_brick_bottleneck(BrickBottleneck::Memory),
        BottleneckClass::MemoryBound
    );
    assert_eq!(
        BottleneckClass::from_brick_bottleneck(BrickBottleneck::Compute),
        BottleneckClass::ComputeBound
    );
    assert_eq!(
        BottleneckClass::from_brick_bottleneck(BrickBottleneck::Unknown),
        BottleneckClass::Unknown
    );
}

#[test]
fn test_bottleneck_class_recommended_action() {
    let action = BottleneckClass::MemoryBound.recommended_action();
    assert!(action.contains("batch size"));
    let action = BottleneckClass::ComputeBound.recommended_action();
    assert!(action.contains("tensor cores") || action.contains("Rare"));
    let action = BottleneckClass::LaunchBound.recommended_action();
    assert!(action.contains("CUDA graphs"));
    let action = BottleneckClass::AttentionBound.recommended_action();
    assert!(action.contains("Flash") || action.contains("sequence"));
    let action = BottleneckClass::Unknown.recommended_action();
    assert!(action.contains("profiling"));
}

#[test]
fn test_bottleneck_class_to_index() {
    assert_eq!(BottleneckClass::Unknown.to_index(), 0);
    assert_eq!(BottleneckClass::MemoryBound.to_index(), 1);
    assert_eq!(BottleneckClass::ComputeBound.to_index(), 2);
    assert_eq!(BottleneckClass::LaunchBound.to_index(), 3);
    assert_eq!(BottleneckClass::AttentionBound.to_index(), 4);
}

#[test]
fn test_bottleneck_class_display() {
    assert_eq!(format!("{}", BottleneckClass::Unknown), "Unknown");
    assert_eq!(format!("{}", BottleneckClass::MemoryBound), "MemoryBound");
    assert_eq!(format!("{}", BottleneckClass::ComputeBound), "ComputeBound");
    assert_eq!(format!("{}", BottleneckClass::LaunchBound), "LaunchBound");
    assert_eq!(format!("{}", BottleneckClass::AttentionBound), "AttentionBound");
}

// =========================================================================
// Throughput Regressor: roofline clamping
// =========================================================================

#[test]
fn test_compute_roofline_bound() {
    let features = TunerFeatures::builder()
        .model_params_b(7.0)
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(1000.0)
        .batch_size(1)
        .build();
    let bound = ThroughputRegressor::compute_roofline_bound(&features);
    assert!(bound > 0.0, "Roofline bound should be positive");
    assert!(bound <= 10000.0, "Roofline bound should be <= 10000");
}

#[test]
fn test_bytes_per_param_from_onehot() {
    // Test each quant type
    for idx in 0..8 {
        let mut onehot = [0.0f32; 8];
        onehot[idx] = 1.0;
        let bpp = ThroughputRegressor::bytes_per_param_from_onehot(&onehot);
        assert!(bpp > 0.0, "Bytes per param should be positive for index {}", idx);
    }

    // All zeros: max_by returns last of equal elements (index 7) -> F32 -> 4.0
    let onehot_zeros = [0.0f32; 8];
    let bpp = ThroughputRegressor::bytes_per_param_from_onehot(&onehot_zeros);
    assert!((bpp - 4.0).abs() < 0.001, "All-zero onehot should map to last index (F32=4.0), got {}", bpp);
}

// =========================================================================
// BottleneckClassifier with profiler bottleneck class preset
// =========================================================================

#[test]
fn test_bottleneck_classifier_with_preset_class() {
    let classifier = BottleneckClassifier::new();
    let mut features = TunerFeatures::builder().build();
    features.bottleneck_class = Some(BottleneckClass::AttentionBound);

    let prediction = classifier.predict(&features);
    assert_eq!(prediction.class, BottleneckClass::AttentionBound);
    assert_eq!(prediction.confidence, 0.95);
}

#[test]
fn test_bottleneck_classifier_heuristic_launch_bound() {
    let classifier = BottleneckClassifier::new();
    // batch_size=1 and no CUDA graphs -> launch bound
    let features = TunerFeatures::builder()
        .batch_size(1)
        .cuda_graphs(false)
        .build();
    let prediction = classifier.predict(&features);
    assert_eq!(prediction.class, BottleneckClass::LaunchBound);
}

#[test]
fn test_bottleneck_classifier_heuristic_attention_bound() {
    let classifier = BottleneckClassifier::new();
    // Long sequence -> attention bound
    let features = TunerFeatures::builder()
        .batch_size(4)
        .seq_len(2048)
        .build();
    let prediction = classifier.predict(&features);
    assert_eq!(prediction.class, BottleneckClass::AttentionBound);
}

#[test]
fn test_bottleneck_classifier_heuristic_memory_bound() {
    let classifier = BottleneckClassifier::new();
    // Normal batch size, short sequence -> memory bound
    let features = TunerFeatures::builder()
        .batch_size(4)
        .seq_len(64)
        .build();
    let prediction = classifier.predict(&features);
    assert_eq!(prediction.class, BottleneckClass::MemoryBound);
}

// =========================================================================
// KernelClassifier various batch sizes
// =========================================================================

#[test]
fn test_kernel_classifier_large_batch() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder()
        .batch_size(8)
        .build();
    let rec = classifier.predict(&features);
    assert_eq!(rec.top_kernel, KernelType::BatchedQ4K);
}

#[test]
fn test_kernel_classifier_medium_batch() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder()
        .batch_size(2)
        .build();
    let rec = classifier.predict(&features);
    assert_eq!(rec.top_kernel, KernelType::VectorizedQ4K);
}

#[test]
fn test_kernel_classifier_single_no_cuda_graphs() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder()
        .batch_size(1)
        .cuda_graphs(false)
        .build();
    let rec = classifier.predict(&features);
    assert_eq!(rec.top_kernel, KernelType::CoalescedQ4K);
}

#[test]
fn test_kernel_classifier_single_with_cuda_graphs() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder()
        .batch_size(1)
        .cuda_graphs(true)
        .build();
    let rec = classifier.predict(&features);
    assert_eq!(rec.top_kernel, KernelType::VectorizedQ4K);
}

// =========================================================================
// TunerFeatures builder all options
// =========================================================================

#[test]
fn test_builder_all_options() {
    let features = TunerFeatures::builder()
        .model_params_b(7.0)
        .hidden_dim(4096)
        .num_layers(32)
        .num_heads(32)
        .head_dim(128)
        .vocab_size(32000)
        .batch_size(4)
        .seq_len(2048)
        .cuda_graphs(true)
        .kv_caches(4)
        .is_prefill(true)
        .quant_type(QuantType::Q6K)
        .kernel_type(KernelType::CoalescedQ6K)
        .gpu_mem_bw_gbs(1000.0)
        .gpu_compute_tflops(100.0)
        .gpu_sm_count(128)
        .gpu_l2_cache_mb(64.0)
        .is_zero_copy(true)
        .measured_tps(150.0)
        .build();

    assert!(features.validate().is_ok());
    assert_eq!(features.cuda_graphs, 1.0);
    assert_eq!(features.is_prefill, 1.0);
    assert_eq!(features.is_zero_copy, 1.0);
    assert!(features.measured_tps.is_some());
    assert_eq!(features.measured_tps.unwrap(), 150.0);
    assert_eq!(features.quant_type_onehot[4], 1.0); // Q6K index
    assert_eq!(features.kernel_type_onehot[6], 1.0); // CoalescedQ6K index

    let vec = features.to_vector();
    assert_eq!(vec.len(), TunerFeatures::DIM);
}

// =========================================================================
// Coverage: BrickTuner::default() (lines 109-111)
// =========================================================================

#[test]
fn test_brick_tuner_default_trait() {
    let tuner: BrickTuner = Default::default();
    assert_eq!(tuner.version(), BrickTuner::VERSION);
    assert_eq!(tuner.sample_count, 0);
}

// =========================================================================
// Coverage: render_comparison "Good" and "Fair" accuracy indicators
// =========================================================================

#[test]
fn test_brick_tuner_render_comparison_good() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(2)
        .model_params_b(1.5)
        .build();
    let rec = tuner.recommend(&features);
    // ~7% error -> "Good" branch (5% <= error < 10%)
    let actual_tps = rec.throughput.predicted_tps * 0.93;
    let comparison = tuner.render_comparison(&rec, actual_tps);
    assert_eq!(comparison.len(), 2);
    assert!(comparison[1].contains("Good"), "Expected 'Good' indicator, got: {}", comparison[1]);
}

#[test]
fn test_brick_tuner_render_comparison_fair() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(2)
        .model_params_b(1.5)
        .build();
    let rec = tuner.recommend(&features);
    // ~15% error -> "Fair" branch (10% <= error < 20%)
    let actual_tps = rec.throughput.predicted_tps * 0.85;
    let comparison = tuner.render_comparison(&rec, actual_tps);
    assert_eq!(comparison.len(), 2);
    assert!(comparison[1].contains("Fair"), "Expected 'Fair' indicator, got: {}", comparison[1]);
}

// =========================================================================
// Coverage: TunerFeaturesBuilder::hardware() with GPU (lines 351-359)
// =========================================================================

#[test]
fn test_builder_hardware_with_gpu() {
    use crate::hardware::{CpuCapability, GpuBackend, GpuCapability, HardwareCapability, RooflineParams, SimdWidth};

    let hw = HardwareCapability {
        timestamp: "test".to_string(),
        hostname: "test-host".to_string(),
        cpu: CpuCapability {
            vendor: "Intel".to_string(),
            model: "Test CPU".to_string(),
            cores: 8,
            threads: 16,
            simd: SimdWidth::Avx2,
            base_freq_ghz: 3.5,
            peak_gflops: 100.0,
            memory_bw_gbps: 50.0,
        },
        gpu: Some(GpuCapability {
            vendor: "NVIDIA".to_string(),
            model: "RTX 4090".to_string(),
            backend: GpuBackend::Cuda,
            compute_capability: Some("8.9".to_string()),
            peak_tflops_fp32: 82.6,
            peak_tflops_tensor: Some(330.0),
            memory_bw_gbps: 1008.0,
            vram_gb: 24.0,
        }),
        roofline: RooflineParams {
            cpu_arithmetic_intensity: 10.0,
            gpu_arithmetic_intensity: Some(50.0),
        },
        byte_budget: None,
    };

    let features = TunerFeatures::builder()
        .hardware(&hw)
        .build();

    // Memory BW: 1008 / 3000 ~ 0.336
    assert!((features.gpu_mem_bw_norm - (1008.0 / 3000.0)).abs() < 0.01);
    // Compute: 82.6 / 500 ~ 0.1652
    assert!((features.gpu_compute_norm - (82.6 / 500.0)).abs() < 0.01);
}

#[test]
fn test_builder_hardware_without_gpu() {
    use crate::hardware::{CpuCapability, HardwareCapability, RooflineParams, SimdWidth};

    let hw = HardwareCapability {
        timestamp: "test".to_string(),
        hostname: "test-host".to_string(),
        cpu: CpuCapability {
            vendor: "Intel".to_string(),
            model: "Test CPU".to_string(),
            cores: 8,
            threads: 16,
            simd: SimdWidth::Avx2,
            base_freq_ghz: 3.5,
            peak_gflops: 100.0,
            memory_bw_gbps: 50.0,
        },
        gpu: None,
        roofline: RooflineParams {
            cpu_arithmetic_intensity: 10.0,
            gpu_arithmetic_intensity: None,
        },
        byte_budget: None,
    };

    let features = TunerFeatures::builder()
        .hardware(&hw)
        .build();

    // No GPU: should use defaults
    // Default gpu_mem_bw_gbs = 1000.0 / 3000.0
    assert!((features.gpu_mem_bw_norm - (1000.0 / 3000.0)).abs() < 0.01);
}

// =========================================================================
// Coverage: FeatureExtractor::with_hardware() + extract() + calculate_efficiency()
// =========================================================================

#[test]
fn test_feature_extractor_with_hardware_and_extract() {
    use crate::hardware::{CpuCapability, GpuBackend, GpuCapability, HardwareCapability, RooflineParams, SimdWidth};
    use crate::brick::BrickProfiler;

    let hw = HardwareCapability {
        timestamp: "test".to_string(),
        hostname: "test-host".to_string(),
        cpu: CpuCapability {
            vendor: "Intel".to_string(),
            model: "Test CPU".to_string(),
            cores: 8,
            threads: 16,
            simd: SimdWidth::Avx2,
            base_freq_ghz: 3.5,
            peak_gflops: 100.0,
            memory_bw_gbps: 50.0,
        },
        gpu: Some(GpuCapability {
            vendor: "NVIDIA".to_string(),
            model: "RTX 4090".to_string(),
            backend: GpuBackend::Cuda,
            compute_capability: Some("8.9".to_string()),
            peak_tflops_fp32: 82.6,
            peak_tflops_tensor: Some(330.0),
            memory_bw_gbps: 1008.0,
            vram_gb: 24.0,
        }),
        roofline: RooflineParams {
            cpu_arithmetic_intensity: 10.0,
            gpu_arithmetic_intensity: Some(50.0),
        },
        byte_budget: None,
    };

    let extractor = FeatureExtractor::with_hardware(hw);
    assert!(extractor.hardware.is_some());

    // Create a profiler with data so tokens_per_sec returns Some
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    let elapsed = std::time::Duration::from_millis(10);
    profiler.record_elapsed("RmsNorm", elapsed, 1000);

    let config = RunConfig::default();
    let features = extractor.extract(&profiler, &config);

    // Should have measured_tps set
    assert!(features.measured_tps.is_some());
    // Should have theoretical_efficiency set
    assert!(features.theoretical_efficiency >= 0.0);
    assert!(features.theoretical_efficiency <= 1.0);
    // Should have bottleneck_class set
    assert!(features.bottleneck_class.is_some());
}

#[test]
fn test_calculate_efficiency_with_hardware() {
    use crate::hardware::{CpuCapability, GpuBackend, GpuCapability, HardwareCapability, RooflineParams, SimdWidth};
    use crate::brick::BrickProfiler;

    let hw = HardwareCapability {
        timestamp: "test".to_string(),
        hostname: "test-host".to_string(),
        cpu: CpuCapability {
            vendor: "Intel".to_string(),
            model: "Test CPU".to_string(),
            cores: 8,
            threads: 16,
            simd: SimdWidth::Avx2,
            base_freq_ghz: 3.5,
            peak_gflops: 100.0,
            memory_bw_gbps: 50.0,
        },
        gpu: Some(GpuCapability {
            vendor: "NVIDIA".to_string(),
            model: "RTX 4090".to_string(),
            backend: GpuBackend::Cuda,
            compute_capability: Some("8.9".to_string()),
            peak_tflops_fp32: 82.6,
            peak_tflops_tensor: Some(330.0),
            memory_bw_gbps: 1008.0,
            vram_gb: 24.0,
        }),
        roofline: RooflineParams {
            cpu_arithmetic_intensity: 10.0,
            gpu_arithmetic_intensity: Some(50.0),
        },
        byte_budget: None,
    };

    let extractor = FeatureExtractor::with_hardware(hw);

    let mut profiler = BrickProfiler::new();
    profiler.enable();
    let elapsed = std::time::Duration::from_millis(10);
    profiler.record_elapsed("RmsNorm", elapsed, 1000);

    let config = RunConfig::default();
    let efficiency = extractor.calculate_efficiency(&profiler, &config);
    assert!(efficiency.is_some());
    let eff = efficiency.unwrap();
    assert!(eff >= 0.0 && eff <= 1.0);
}

#[test]
fn test_calculate_efficiency_no_hardware() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    let elapsed = std::time::Duration::from_millis(10);
    profiler.record_elapsed("RmsNorm", elapsed, 1000);

    let config = RunConfig::default();
    let efficiency = extractor.calculate_efficiency(&profiler, &config);
    assert!(efficiency.is_none(), "No hardware -> no efficiency calculation");
}

// =========================================================================
// Coverage: classify_bottleneck with profiler data (lines 527-553)
// =========================================================================

#[test]
fn test_classify_bottleneck_attention_dominant() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Record attention bricks with 50% of time
    let attn_elapsed = std::time::Duration::from_millis(50);
    profiler.record_elapsed("QkvProjection", attn_elapsed, 100);
    profiler.record_elapsed("AttentionScore", attn_elapsed, 100);

    // Record FFN with 20% of time
    let ffn_elapsed = std::time::Duration::from_millis(20);
    profiler.record_elapsed("GateProjection", ffn_elapsed, 100);

    // Record norm with 5% of time
    let norm_elapsed = std::time::Duration::from_millis(5);
    profiler.record_elapsed("RmsNorm", norm_elapsed, 100);

    let bottleneck = extractor.classify_bottleneck(&profiler);
    assert_eq!(bottleneck, BottleneckClass::AttentionBound);
}

#[test]
fn test_classify_bottleneck_ffn_dominant() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Record FFN bricks with 60% of time
    let ffn_elapsed = std::time::Duration::from_millis(60);
    profiler.record_elapsed("GateProjection", ffn_elapsed, 100);
    profiler.record_elapsed("UpProjection", ffn_elapsed, 100);
    profiler.record_elapsed("DownProjection", ffn_elapsed, 100);

    // Record attention with 10% of time
    let attn_elapsed = std::time::Duration::from_millis(10);
    profiler.record_elapsed("QkvProjection", attn_elapsed, 100);

    // Record norm with 5% of time
    let norm_elapsed = std::time::Duration::from_millis(5);
    profiler.record_elapsed("RmsNorm", norm_elapsed, 100);

    let bottleneck = extractor.classify_bottleneck(&profiler);
    assert_eq!(bottleneck, BottleneckClass::MemoryBound);
}

#[test]
fn test_classify_bottleneck_norm_dominant() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Record norm bricks with 30% of time
    let norm_elapsed = std::time::Duration::from_millis(30);
    profiler.record_elapsed("RmsNorm", norm_elapsed, 100);

    // Record attention with 25% of time
    let attn_elapsed = std::time::Duration::from_millis(25);
    profiler.record_elapsed("QkvProjection", attn_elapsed, 100);

    // Record FFN with 30% of time
    let ffn_elapsed = std::time::Duration::from_millis(15);
    profiler.record_elapsed("GateProjection", ffn_elapsed, 100);
    profiler.record_elapsed("DownProjection", ffn_elapsed, 100);

    let bottleneck = extractor.classify_bottleneck(&profiler);
    assert_eq!(bottleneck, BottleneckClass::LaunchBound);
}

#[test]
fn test_classify_bottleneck_default_memory_bound() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Record mixed low-percentage bricks (no dominant category)
    // Attention < 35%, FFN < 50%, Norm < 20%
    let elapsed = std::time::Duration::from_millis(10);
    profiler.record_elapsed("QkvProjection", elapsed, 100);  // ~25% attention
    profiler.record_elapsed("GateProjection", elapsed, 100); // ~25% FFN
    profiler.record_elapsed("RmsNorm", elapsed, 100);        // ~25% norm... need to adjust

    // Actually: 3 equal parts means each is ~33%. Attention=33% < 35%, FFN=33% < 50%, Norm=33% > 20%
    // So this would hit LaunchBound. Let me adjust.
    // Need: attn<35%, ffn<50%, norm<20%
    // Use a dynamic "other" brick to dilute
    profiler.record_elapsed("Embedding", std::time::Duration::from_millis(30), 100);

    let bottleneck = extractor.classify_bottleneck(&profiler);
    // Embedding is "Other" category, so attn=10/60=16.7%, ffn=10/60=16.7%, norm=10/60=16.7%
    // All below thresholds -> default MemoryBound
    assert_eq!(bottleneck, BottleneckClass::MemoryBound);
}

#[test]
fn test_classify_bottleneck_empty_profiler() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let profiler = BrickProfiler::new();

    let bottleneck = extractor.classify_bottleneck(&profiler);
    assert_eq!(bottleneck, BottleneckClass::Unknown);
}
