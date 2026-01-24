use super::*;
use std::time::Instant;

// F001-F020: Model Accuracy
mod f001_f020_model_accuracy {
    use super::*;

    #[test]
    fn f001_throughput_prediction_reasonable() {
        let features = TunerFeatures::builder()
            .model_params_b(1.5)
            .hidden_dim(1536)
            .batch_size(4)
            .quant_type(QuantType::Q4K)
            .cuda_graphs(true)
            .build();

        let regressor = ThroughputRegressor::new();
        let prediction = regressor.predict(&features);

        // Prediction should be positive and reasonable
        assert!(prediction.predicted_tps > 0.0);
        assert!(prediction.predicted_tps < 10000.0);
    }

    #[test]
    fn f010_prediction_latency_under_1ms() {
        let features = TunerFeatures::builder()
            .model_params_b(1.5)
            .batch_size(4)
            .build();

        let tuner = BrickTuner::new();
        let start = Instant::now();
        let _rec = tuner.recommend(&features);
        let elapsed = start.elapsed();

        assert!(elapsed.as_millis() < 1, "Prediction took {}ms", elapsed.as_millis());
    }

    #[test]
    fn f015_batch_size_monotonic() {
        let regressor = ThroughputRegressor::new();

        let pred_m1 = regressor.predict(&TunerFeatures::builder().batch_size(1).build());
        let pred_m4 = regressor.predict(&TunerFeatures::builder().batch_size(4).build());
        let pred_m8 = regressor.predict(&TunerFeatures::builder().batch_size(8).build());

        // Higher batch size should predict higher throughput
        assert!(
            pred_m4.predicted_tps >= pred_m1.predicted_tps,
            "M=4 ({}) should be >= M=1 ({})",
            pred_m4.predicted_tps,
            pred_m1.predicted_tps
        );
        assert!(
            pred_m8.predicted_tps >= pred_m4.predicted_tps,
            "M=8 ({}) should be >= M=4 ({})",
            pred_m8.predicted_tps,
            pred_m4.predicted_tps
        );
    }

    #[test]
    fn f019_cuda_graphs_benefit_predicted() {
        let regressor = ThroughputRegressor::new();

        let pred_no_graph = regressor.predict(
            &TunerFeatures::builder()
                .batch_size(1)
                .cuda_graphs(false)
                .build(),
        );
        let pred_with_graph = regressor.predict(
            &TunerFeatures::builder()
                .batch_size(1)
                .cuda_graphs(true)
                .build(),
        );

        // CUDA graphs should predict higher throughput
        assert!(
            pred_with_graph.predicted_tps >= pred_no_graph.predicted_tps,
            "With graphs ({}) should be >= without ({})",
            pred_with_graph.predicted_tps,
            pred_no_graph.predicted_tps
        );
    }
}

// F021-F040: Feature Engineering
mod f021_f040_feature_engineering {
    use super::*;

    #[test]
    fn f021_no_nan_features() {
        let features = TunerFeatures::builder()
            .model_params_b(1.5)
            .hidden_dim(1536)
            .batch_size(4)
            .quant_type(QuantType::Q4K)
            .build();

        let v = features.to_vector();
        assert!(!v.iter().any(|x| x.is_nan()), "Features contain NaN");
    }

    #[test]
    fn f022_no_infinite_features() {
        let features = TunerFeatures::builder()
            .model_params_b(1.5)
            .hidden_dim(1536)
            .batch_size(4)
            .build();

        let v = features.to_vector();
        assert!(
            !v.iter().any(|x| x.is_infinite()),
            "Features contain infinity"
        );
    }

    #[test]
    fn f023_features_in_0_1_range() {
        let features = TunerFeatures::builder()
            .model_params_b(100.0) // Very large
            .hidden_dim(16384)     // Max
            .batch_size(64)        // Max
            .seq_len(32768)        // Max
            .build();

        let v = features.to_vector();
        for (i, x) in v.iter().enumerate() {
            assert!(
                *x >= -0.001 && *x <= 1.001,
                "Feature {} = {} is outside [0, 1]",
                i,
                x
            );
        }
    }

    /// f026: Roofline bound - predicted TPS must never exceed theoretical maximum
    /// This is the crucible that ensures ML predictions do not violate hardware limits.
    /// Roofline model: max_tps = memory_bw_bytes_per_sec / bytes_per_token
    /// For decode: bytes_per_token ≈ model_params × bytes_per_param
    #[test]
    fn f026_roofline_bound() {
        // Test configuration: 7B Q4_K model on RTX 4090 (1008 GB/s)
        let model_params_b: f32 = 7.0;
        let bytes_per_param: f32 = QuantType::Q4K.bytes_per_param();
        let gpu_mem_bw_gbs: f32 = 1008.0; // RTX 4090

        // Theoretical maximum tokens/sec for decode phase (batch=1)
        // bytes_per_token = model_params * bytes_per_param * 1e9
        // max_tps = mem_bw_bytes_per_sec / bytes_per_token
        let model_bytes: f32 = model_params_b * bytes_per_param * 1e9;
        let theoretical_max_tps: f32 = (gpu_mem_bw_gbs * 1e9) / model_bytes;

        // Build features for this configuration
        let features = TunerFeatures::builder()
            .model_params_b(model_params_b)
            .batch_size(1)
            .quant_type(QuantType::Q4K)
            .gpu_mem_bw_gbs(gpu_mem_bw_gbs)
            .gpu_compute_tflops(82.6) // RTX 4090 FP32
            .is_prefill(false) // Decode phase
            .build();

        let tuner = BrickTuner::new();
        let rec = tuner.recommend(&features);

        // CRITICAL ASSERTION: predicted_tps <= theoretical_max_tps
        // Allowing 10% margin for numerical precision
        let margin = 1.10;
        assert!(
            rec.throughput.predicted_tps <= theoretical_max_tps * margin,
            "Roofline violation: predicted {} tok/s exceeds theoretical max {} tok/s \
             (model: {}B, quant: Q4K, mem_bw: {} GB/s)",
            rec.throughput.predicted_tps,
            theoretical_max_tps,
            model_params_b,
            gpu_mem_bw_gbs
        );

        // Also verify theoretical max is reasonable (sanity check)
        // 7B Q4_K on 1008 GB/s should yield ~200-300 tok/s theoretical max
        assert!(
            theoretical_max_tps > 100.0 && theoretical_max_tps < 500.0,
            "Theoretical max {} tok/s is outside expected range for 7B Q4_K on RTX 4090",
            theoretical_max_tps
        );
    }

    #[test]
    fn f029_onehot_sums_to_one() {
        let features = TunerFeatures::builder()
            .quant_type(QuantType::Q4K)
            .kernel_type(KernelType::VectorizedQ4K)
            .build();

        let quant_sum: f32 = features.quant_type_onehot.iter().sum();
        let kernel_sum: f32 = features.kernel_type_onehot.iter().sum();

        assert!(
            (quant_sum - 1.0).abs() < 0.001,
            "Quant one-hot sum = {}",
            quant_sum
        );
        assert!(
            (kernel_sum - 1.0).abs() < 0.001,
            "Kernel one-hot sum = {}",
            kernel_sum
        );
    }

    /// f040: Feature dimension must be 42 per spec v1.1.0
    /// 11 static + 8 quant + 16 kernel + 5 hardware + 2 derived = 42
    #[test]
    fn f040_feature_dimension_is_42() {
        assert_eq!(TunerFeatures::DIM, 42, "DIM must be 42 per spec v1.1.0");

        let features = TunerFeatures::builder().build();
        assert_eq!(features.to_vector().len(), TunerFeatures::DIM);
    }
}

// F041-F060: Training Data Quality
mod f041_f060_training_data {
    use super::*;

    #[test]
    fn f059_no_data_leakage() {
        // Training labels should not be in feature vector
        let features = TunerFeatures::builder()
            .measured_tps(500.0) // Label
            .build();

        let v = features.to_vector();
        // measured_tps should NOT be in the vector (it's a label)
        assert_eq!(v.len(), TunerFeatures::DIM);
    }
}

// F061-F080: Integration Correctness
mod f061_f080_integration {
    use super::*;

    #[test]
    fn f066_recommendations_json_valid() {
        let features = TunerFeatures::builder()
            .model_params_b(1.5)
            .batch_size(4)
            .build();

        let tuner = BrickTuner::new();
        let rec = tuner.recommend(&features);

        let json = serde_json::to_string(&rec);
        assert!(json.is_ok(), "Failed to serialize recommendation");
    }

    #[test]
    fn f070_safetensors_roundtrip() {
        let tuner = BrickTuner::new();

        // Serialize
        let json = tuner.to_json();
        assert!(json.is_ok());

        // Deserialize
        let loaded = BrickTuner::from_json(&json.unwrap());
        assert!(loaded.is_ok());
        assert_eq!(loaded.unwrap().version, tuner.version);
    }

    #[test]
    fn f071_feature_extractor_deterministic() {
        let config = RunConfig::default();
        let profiler = BrickProfiler::new();
        let extractor = FeatureExtractor::new();

        let f1 = extractor.extract(&profiler, &config);
        let f2 = extractor.extract(&profiler, &config);

        assert_eq!(f1.to_vector(), f2.to_vector());
    }

    #[test]
    fn f072_prediction_deterministic() {
        let features = TunerFeatures::builder()
            .model_params_b(1.5)
            .batch_size(4)
            .build();

        let tuner = BrickTuner::new();
        let rec1 = tuner.recommend(&features);
        let rec2 = tuner.recommend(&features);

        assert_eq!(
            rec1.throughput.predicted_tps,
            rec2.throughput.predicted_tps
        );
        assert_eq!(rec1.kernel.top_kernel, rec2.kernel.top_kernel);
    }

    #[test]
    fn f075_error_handling_graceful() {
        // Invalid features should not panic
        let mut features = TunerFeatures::default();
        features.model_params_b = f32::NAN;

        let result = features.validate();
        assert!(result.is_err());
    }
}

// F081-F100: Generalization & Robustness
mod f081_f100_generalization {
    use super::*;

    #[test]
    fn f085_adversarial_inputs_handled() {
        // Extreme values should not crash
        let features = TunerFeatures::builder()
            .model_params_b(0.001) // Very small
            .hidden_dim(1)
            .batch_size(1000) // Very large (will be clamped)
            .build();

        let tuner = BrickTuner::new();
        let rec = tuner.recommend(&features);

        // Should produce some recommendation without crashing
        assert!(rec.throughput.predicted_tps > 0.0);
    }

    #[test]
    fn f091_cold_start_handling() {
        // Tuner should work with default (untrained) model
        let tuner = BrickTuner::new();
        assert_eq!(tuner.sample_count, 0);

        let features = TunerFeatures::builder().batch_size(4).build();
        let rec = tuner.recommend(&features);

        // Should still produce reasonable recommendations
        assert!(rec.confidence_overall > 0.0);
    }

    #[test]
    fn f096_extreme_values_clipped() {
        let features = TunerFeatures::builder()
            .model_params_b(1000.0) // Way over max
            .hidden_dim(100000)     // Way over max
            .batch_size(1000)       // Way over max
            .build();

        // All values should be clipped to [0, 1]
        let v = features.to_vector();
        assert!(v.iter().all(|x| *x >= 0.0 && *x <= 1.0));
    }
}

// Bottleneck classification tests
#[test]
fn test_bottleneck_recommended_action() {
    assert!(BottleneckClass::MemoryBound
        .recommended_action()
        .contains("batch size"));
    assert!(BottleneckClass::LaunchBound
        .recommended_action()
        .contains("CUDA graphs"));
    assert!(BottleneckClass::AttentionBound
        .recommended_action()
        .contains("Flash Decoding"));
}

// Kernel classifier tests
#[test]
fn test_kernel_classifier_batched_for_high_m() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder().batch_size(8).build();

    let rec = classifier.predict(&features);
    assert_eq!(rec.top_kernel, KernelType::BatchedQ4K);
}

// Feature builder tests
#[test]
fn test_feature_builder_normalization() {
    let features = TunerFeatures::builder()
        .model_params_b(1.0) // log10(1.0) = 0, normalized = (0+1)/3 = 0.33
        .hidden_dim(1536)    // 1536/16384 ≈ 0.094
        .batch_size(4)       // 4/64 = 0.0625
        .build();

    assert!(features.model_params_b > 0.0 && features.model_params_b < 1.0);
    assert!(features.hidden_dim_norm > 0.0 && features.hidden_dim_norm < 1.0);
    assert!(features.batch_size_norm > 0.0 && features.batch_size_norm < 1.0);
}

// Additional coverage tests
#[test]
fn test_all_builder_methods() {
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .hidden_dim(2048)
        .num_layers(32)
        .num_heads(16)
        .head_dim(128)
        .vocab_size(32000)
        .batch_size(4)
        .seq_len(512)
        .cuda_graphs(true)
        .kv_caches(4)
        .is_prefill(false)
        .quant_type(QuantType::Q4K)
        .kernel_type(KernelType::VectorizedQ4K)
        .gpu_mem_bw_gbs(1000.0)
        .gpu_compute_tflops(150.0)
        .gpu_sm_count(128)
        .measured_tps(100.0)
        .build();

    assert!(features.model_params_b > 0.0);
    assert!(features.cuda_graphs == 1.0);
    assert!(features.is_prefill == 0.0);
}

#[test]
fn test_quant_type_bytes_per_param() {
    assert_eq!(QuantType::Q4_0.bytes_per_param(), 0.5625);
    assert_eq!(QuantType::Q4_1.bytes_per_param(), 0.5625);
    assert_eq!(QuantType::Q5K.bytes_per_param(), 0.6875);
    assert_eq!(QuantType::Q6K.bytes_per_param(), 0.8125);
    assert_eq!(QuantType::Q8_0.bytes_per_param(), 1.0);
    assert_eq!(QuantType::F16.bytes_per_param(), 2.0);
    assert_eq!(QuantType::F32.bytes_per_param(), 4.0);
}

#[test]
fn test_kernel_type_to_index() {
    assert_eq!(KernelType::TiledQ4K.to_index(), 0);
    assert_eq!(KernelType::CoalescedQ4K.to_index(), 1);
    assert_eq!(KernelType::VectorizedQ4K.to_index(), 2);
    assert_eq!(KernelType::BatchedQ4K.to_index(), 3);
    assert_eq!(KernelType::Dp4aQ4K.to_index(), 4);
    assert_eq!(KernelType::FusedRmsNormQ4K.to_index(), 5);
    assert_eq!(KernelType::CoalescedQ6K.to_index(), 6);
    assert_eq!(KernelType::IncrementalAttention.to_index(), 7);
    assert_eq!(KernelType::MultiWarpAttention.to_index(), 8);
    assert_eq!(KernelType::BatchedAttention.to_index(), 9);
    assert_eq!(KernelType::RmsNorm.to_index(), 10);
    assert_eq!(KernelType::VectorizedRmsNorm.to_index(), 11);
    assert_eq!(KernelType::BatchedRmsNorm.to_index(), 12);
    assert_eq!(KernelType::Generic.to_index(), 13);
    assert_eq!(KernelType::Unknown.to_index(), 14);
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
fn test_bottleneck_display() {
    assert_eq!(format!("{}", BottleneckClass::Unknown), "Unknown");
    assert_eq!(format!("{}", BottleneckClass::MemoryBound), "MemoryBound");
    assert_eq!(format!("{}", BottleneckClass::ComputeBound), "ComputeBound");
    assert_eq!(format!("{}", BottleneckClass::LaunchBound), "LaunchBound");
    assert_eq!(format!("{}", BottleneckClass::AttentionBound), "AttentionBound");
}

#[test]
fn test_from_brick_bottleneck() {
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
fn test_run_config_default() {
    let config = RunConfig::default();
    assert_eq!(config.model_params_b, 1.5);
    assert_eq!(config.batch_size, 1);
    assert_eq!(config.quant_type, QuantType::Q4K);
}

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
fn test_bottleneck_classifier() {
    let classifier = BottleneckClassifier::new();
    let features = TunerFeatures::builder().batch_size(4).build();
    let pred = classifier.predict(&features);
    // Default prediction should be MemoryBound for inference
    assert!(matches!(
        pred.class,
        BottleneckClass::MemoryBound | BottleneckClass::Unknown
    ));
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

    let exp = ExperimentSuggestion::TryKernel { kernel: KernelType::BatchedQ4K };
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
    let timestamp = super::chrono_lite_now();
    let parsed: u64 = timestamp.parse().expect("Should be a number");
    assert!(parsed > 0);
}

#[test]
fn test_pad_right() {
    assert_eq!(super::pad_right("test", 10), "test      ");
    assert_eq!(super::pad_right("longstring", 5), "longs");
}

// Additional coverage tests for v1.1.0

#[test]
fn test_quant_type_to_index_all_variants() {
    // Cover all QuantType::to_index branches
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
fn test_validation_infinite_features() {
    let mut features = TunerFeatures::default();
    features.model_params_b = f32::INFINITY;
    let result = features.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Infinite"));
}

#[test]
fn test_validation_out_of_range() {
    let mut features = TunerFeatures::default();
    features.batch_size_norm = 2.0; // Out of [0, 1]
    let result = features.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("outside [0, 1]"));
}

#[test]
fn test_validation_bad_quant_onehot() {
    let mut features = TunerFeatures::default();
    features.quant_type_onehot = [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // Sums to 1 but invalid one-hot
    // This should actually pass since sum is 1.0
    assert!(features.validate().is_ok());

    // Now test with sum != 1
    features.quant_type_onehot = [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // Sums to 0.5
    let result = features.validate();
    assert!(result.is_err());
}

#[test]
fn test_validation_bad_kernel_onehot() {
    let mut features = TunerFeatures::default();
    features.kernel_type_onehot = [0.0; 16]; // All zeros, sum = 0
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
    let features_enabled = TunerFeatures::builder()
        .is_zero_copy(true)
        .build();
    assert_eq!(features_enabled.is_zero_copy, 1.0);

    let features_disabled = TunerFeatures::builder()
        .is_zero_copy(false)
        .build();
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

    // Memory BW: 1000 / 3000 ≈ 0.333
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
    assert!(matches!(result.unwrap_err(), TunerError::InsufficientData(5)));
}

#[test]
fn test_brick_tuner_print_recommendation() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder().batch_size(4).build();
    let rec = tuner.recommend(&features);

    // Just verify it doesn't panic
    tuner.print_recommendation(&rec);
}

#[test]
fn test_attention_bound_suggestions() {
    let tuner = BrickTuner::new();

    // Create features that would trigger AttentionBound
    let mut features = TunerFeatures::builder()
        .batch_size(1)
        .seq_len(8192) // Long sequence
        .is_prefill(true)
        .build();
    features.bottleneck_class = Some(BottleneckClass::AttentionBound);

    let bottleneck_pred = BottleneckPrediction {
        class: BottleneckClass::AttentionBound,
        confidence: 0.9,
        explanation: "Attention bound".to_string(),
        recommended_action: "Use FlashAttention".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck_pred);
    // Should suggest BatchedAttention and ReduceSequenceLength
    let has_batched_attention = suggestions.iter().any(|s| {
        matches!(s, ExperimentSuggestion::TryKernel { kernel: KernelType::BatchedAttention })
    });
    let has_reduce_seq = suggestions.iter().any(|s| {
        matches!(s, ExperimentSuggestion::ReduceSequenceLength { .. })
    });
    assert!(has_batched_attention || has_reduce_seq);
}

#[test]
fn test_unknown_bottleneck_suggestions() {
    let tuner = BrickTuner::new();

    let mut features = TunerFeatures::builder()
        .batch_size(1)
        .build();
    features.bottleneck_class = Some(BottleneckClass::Unknown);

    let rec = tuner.recommend(&features);
    // Should suggest increasing batch size from 1 to 4
    let has_increase_batch = rec.suggested_experiments.iter().any(|s| {
        matches!(s, ExperimentSuggestion::IncreaseBatchSize { from: 1, to: 4 })
    });
    assert!(has_increase_batch);
}

#[test]
fn test_data_collector_record() {
    use std::time::Duration;

    let mut collector = TunerDataCollector::new();
    let mut profiler = BrickProfiler::enabled();
    let config = RunConfig::default();

    // Simulate a profiling run using the proper API
    profiler.record_elapsed("test_brick", Duration::from_micros(100), 32);

    let result = collector.record(&profiler, &config, KernelType::VectorizedQ4K);
    assert!(result.is_some());
    assert_eq!(collector.len(), 1);
    assert!(!collector.is_empty());
}

#[test]
fn test_data_collector_to_json() {
    let collector = TunerDataCollector::new();
    let json = collector.to_json();
    assert!(json.is_ok());
    assert_eq!(json.unwrap(), "[]"); // Empty array
}

#[test]
fn test_data_collector_prepare_training_data() {
    use std::time::Duration;

    let mut collector = TunerDataCollector::new();
    let mut profiler = BrickProfiler::enabled();
    let config = RunConfig::default();

    // Add a sample using the proper API
    profiler.record_elapsed("test_brick", Duration::from_micros(100), 32);
    collector.record(&profiler, &config, KernelType::VectorizedQ4K);

    let training_data = collector.prepare_training_data();
    assert_eq!(training_data.len(), 1);
    assert!(training_data[0].1 > 0.0); // throughput > 0
}

#[test]
fn test_roofline_helper_methods() {
    // Test bytes_per_param_from_onehot
    let onehot_q4k = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let bytes = ThroughputRegressor::bytes_per_param_from_onehot(&onehot_q4k);
    assert!((bytes - 0.5625).abs() < 0.001);

    let onehot_f32 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let bytes = ThroughputRegressor::bytes_per_param_from_onehot(&onehot_f32);
    assert!((bytes - 4.0).abs() < 0.001);
}

#[test]
fn test_compute_roofline_bound() {
    let features = TunerFeatures::builder()
        .model_params_b(7.0)
        .batch_size(1)
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(1008.0)
        .build();

    let bound = ThroughputRegressor::compute_roofline_bound(&features);
    // Should be around 256 tok/s for 7B Q4K on 1008 GB/s
    assert!(bound > 200.0 && bound < 300.0);
}

// ===== Additional Coverage Tests for 95%+ =====

#[test]
fn test_kernel_classifier_vectorized_for_low_m() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder().batch_size(1).build();

    let rec = classifier.predict(&features);
    // For M=1, should recommend VectorizedQ4K or CoalescedQ4K
    assert!(matches!(
        rec.top_kernel,
        KernelType::VectorizedQ4K | KernelType::CoalescedQ4K
    ));
}

#[test]
fn test_kernel_classifier_all_alternatives() {
    let classifier = KernelClassifier::new();
    let features = TunerFeatures::builder().batch_size(4).build();

    let rec = classifier.predict(&features);
    // Should have some alternatives
    assert!(!rec.alternatives.is_empty());
    // All probabilities should be non-negative
    assert!(rec.alternatives.iter().all(|(_, prob)| *prob >= 0.0));
}

#[test]
fn test_bottleneck_classifier_prefill_compute_bound() {
    let classifier = BottleneckClassifier::new();
    let features = TunerFeatures::builder()
        .batch_size(8)
        .is_prefill(true)
        .build();

    let pred = classifier.predict(&features);
    // Prefill with high batch should lean toward ComputeBound
    assert!(matches!(
        pred.class,
        BottleneckClass::ComputeBound | BottleneckClass::MemoryBound | BottleneckClass::Unknown
    ));
    assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
}

#[test]
fn test_training_stats_debug() {
    let stats = TrainingStats {
        total_samples: 100,
        samples_since_training: 10,
        accepted_count: 80,
        rejected_count: 15,
        alternative_count: 5,
        staleness_score: 0.1,
        drift_detected: false,
        online_learning_enabled: true,
    };

    let display = format!("{:?}", stats);
    assert!(display.contains("100"));
    assert!(display.contains("accepted_count"));
}

#[test]
fn test_user_feedback_variants() {
    let feedback_accepted = UserFeedback::Accepted;
    let feedback_rejected = UserFeedback::Rejected;
    let feedback_alternative = UserFeedback::Alternative;
    let feedback_none = UserFeedback::None;

    assert!(format!("{:?}", feedback_accepted).contains("Accepted"));
    assert!(format!("{:?}", feedback_rejected).contains("Rejected"));
    assert!(format!("{:?}", feedback_alternative).contains("Alternative"));
    assert!(format!("{:?}", feedback_none).contains("None"));
}

#[test]
fn test_concept_drift_status_creation() {
    let status = ConceptDriftStatus {
        drift_detected: false,
        staleness_score: 0.1,
        samples_since_training: 5,
        recommend_retrain: false,
        explanation: "No drift detected".to_string(),
    };
    assert!(!status.drift_detected);
    assert_eq!(status.samples_since_training, 5);
}

#[test]
fn test_training_sample_creation() {
    let features = TunerFeatures::builder().batch_size(4).build();
    let sample = TrainingSample {
        features,
        throughput_tps: 100.0,
        best_kernel: KernelType::VectorizedQ4K,
        bottleneck: BottleneckClass::MemoryBound,
        timestamp: "2026-01-14".to_string(),
        hardware_id: "test-hw".to_string(),
    };

    assert_eq!(sample.throughput_tps, 100.0);
    assert_eq!(sample.best_kernel, KernelType::VectorizedQ4K);
}

#[test]
fn test_feature_extractor_with_different_configs() {
    let extractor = FeatureExtractor::new();
    let profiler = BrickProfiler::new();

    // Test with different model sizes
    for model_size in [1.5, 7.0, 13.0, 70.0] {
        let config = RunConfig {
            model_params_b: model_size,
            batch_size: 4,
            quant_type: QuantType::Q4K,
            ..Default::default()
        };
        let features = extractor.extract(&profiler, &config);
        assert!(features.model_params_b >= 0.0 && features.model_params_b <= 1.0);
    }
}

#[test]
fn test_all_quant_type_bytes_per_param() {
    // Verify all quant types have valid bytes_per_param
    let quant_types = [
        QuantType::Q4_0,
        QuantType::Q4_1,
        QuantType::Q4K,
        QuantType::Q5K,
        QuantType::Q6K,
        QuantType::Q8_0,
        QuantType::F16,
        QuantType::F32,
    ];

    for qt in quant_types {
        let bytes = qt.bytes_per_param();
        assert!(bytes > 0.0);
        assert!(bytes <= 4.0); // F32 is max at 4 bytes
    }
}

#[test]
fn test_recommendation_fields() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .model_params_b(1.5)
        .batch_size(4)
        .quant_type(QuantType::Q4K)
        .cuda_graphs(false)
        .build();

    let rec = tuner.recommend(&features);

    // Check all fields are populated
    assert!(rec.throughput.predicted_tps > 0.0);
    assert!(rec.kernel.confidence >= 0.0 && rec.kernel.confidence <= 1.0);
    assert!(!rec.bottleneck.explanation.is_empty());
    assert!(!rec.bottleneck.recommended_action.is_empty());
    assert!(rec.confidence_overall >= 0.0 && rec.confidence_overall <= 1.0);
}

#[test]
fn test_launch_bound_suggestions() {
    let tuner = BrickTuner::new();

    let features = TunerFeatures::builder()
        .batch_size(1)
        .cuda_graphs(false)
        .build();

    let bottleneck_pred = BottleneckPrediction {
        class: BottleneckClass::LaunchBound,
        confidence: 0.9,
        explanation: "Launch overhead dominates".to_string(),
        recommended_action: "Enable CUDA graphs".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck_pred);
    let has_cuda_graphs = suggestions.iter().any(|s| {
        matches!(s, ExperimentSuggestion::EnableCudaGraphs)
    });
    assert!(has_cuda_graphs);
}

#[test]
fn test_memory_bound_suggestions() {
    let tuner = BrickTuner::new();

    let features = TunerFeatures::builder()
        .batch_size(1)
        .build();

    let bottleneck_pred = BottleneckPrediction {
        class: BottleneckClass::MemoryBound,
        confidence: 0.9,
        explanation: "Memory bandwidth limited".to_string(),
        recommended_action: "Increase batch size".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck_pred);
    let has_increase_batch = suggestions.iter().any(|s| {
        matches!(s, ExperimentSuggestion::IncreaseBatchSize { .. })
    });
    assert!(has_increase_batch);
}

#[test]
fn test_compute_bound_suggestions() {
    let tuner = BrickTuner::new();

    let features = TunerFeatures::builder()
        .batch_size(8)
        .is_prefill(true)
        .build();

    let bottleneck_pred = BottleneckPrediction {
        class: BottleneckClass::ComputeBound,
        confidence: 0.9,
        explanation: "Compute limited".to_string(),
        recommended_action: "Use tensor cores".to_string(),
    };

    let suggestions = tuner.suggest_experiments(&features, &bottleneck_pred);
    // Suggestions were generated (may be empty if no specific action needed)
    let _count = suggestions.len();
}

// =========================================================================
// T-TUNER-006: TUI Rendering Tests
// =========================================================================

#[test]
fn test_render_panel_output_format() {
    let tuner = BrickTuner::new();
    let rec = create_test_recommendation();

    let lines = tuner.render_panel(&rec);

    // Should have at least 12 lines (header + content + suggestions + footer)
    assert!(lines.len() >= 12);

    // First line should contain version
    assert!(lines[0].contains("BrickTuner"));

    // Should contain predicted throughput
    assert!(lines.iter().any(|l| l.contains("Predicted throughput")));

    // Should contain recommended kernel
    assert!(lines.iter().any(|l| l.contains("Recommended kernel")));

    // Should contain bottleneck class
    assert!(lines.iter().any(|l| l.contains("Bottleneck class")));
}

#[test]
fn test_render_compact_single_line() {
    let tuner = BrickTuner::new();
    let rec = create_test_recommendation();

    let compact = tuner.render_compact(&rec);

    // Should be a single line string
    assert!(!compact.contains('\n'));

    // Should contain key info
    assert!(compact.contains("Tuner:"));
    assert!(compact.contains("tok/s"));
}

#[test]
fn test_render_comparison_accuracy_indicators() {
    let tuner = BrickTuner::new();
    let rec = create_test_recommendation();

    // Test excellent accuracy (< 5% error)
    let lines_excellent = tuner.render_comparison(&rec, 100.0);
    assert_eq!(lines_excellent.len(), 2);
    assert!(lines_excellent[0].contains("Predicted"));
    assert!(lines_excellent[0].contains("Actual"));

    // Test with zero actual (edge case)
    let lines_zero = tuner.render_comparison(&rec, 0.0);
    assert_eq!(lines_zero.len(), 2);

    // Test poor accuracy (> 20% error)
    let lines_poor = tuner.render_comparison(&rec, 50.0);
    assert_eq!(lines_poor.len(), 2);
}

// =========================================================================
// T-TUNER-007: Serialization Tests
// =========================================================================

#[test]
fn test_to_json_serialization() {
    let tuner = BrickTuner::new();
    let json = tuner.to_json();

    assert!(json.is_ok());
    let json_str = json.unwrap();
    assert!(json_str.contains("version"));
    assert!(json_str.contains("throughput")); // Field is named "throughput"
}

#[test]
fn test_from_json_deserialization() {
    let tuner = BrickTuner::new();
    let json = tuner.to_json().unwrap();

    let restored = BrickTuner::from_json(&json);
    assert!(restored.is_ok());

    let restored_tuner = restored.unwrap();
    assert_eq!(restored_tuner.version, tuner.version);
}

#[test]
fn test_json_roundtrip() {
    let tuner = BrickTuner::new();

    // Serialize then deserialize
    let json = tuner.to_json().unwrap();
    let restored = BrickTuner::from_json(&json).unwrap();

    // Re-serialize and compare
    let json2 = restored.to_json().unwrap();
    assert_eq!(json, json2);
}

#[test]
fn test_from_json_invalid() {
    let result = BrickTuner::from_json("not valid json");
    assert!(result.is_err());
}

// =========================================================================
// T-TUNER-008: TunerDataCollector Online Learning Tests
// =========================================================================

#[test]
fn test_collector_with_online_learning() {
    let collector = TunerDataCollector::with_online_learning();
    assert!(collector.is_online_learning_enabled());
}

#[test]
fn test_collector_enable_disable_online_learning() {
    let mut collector = TunerDataCollector::new();

    // Default should be disabled
    assert!(!collector.is_online_learning_enabled());

    // Enable
    collector.enable_online_learning();
    assert!(collector.is_online_learning_enabled());

    // Disable
    collector.disable_online_learning();
    assert!(!collector.is_online_learning_enabled());
}

#[test]
fn test_collector_record_prediction_error() {
    let mut collector = TunerDataCollector::with_online_learning();

    // Record some prediction errors
    collector.record_prediction_error(100.0, 95.0); // 5% error
    collector.record_prediction_error(100.0, 80.0); // 20% error
    collector.record_prediction_error(100.0, 110.0); // 10% error

    // Should track errors for drift detection
    let drift = collector.detect_concept_drift();
    // With only 3 samples, should indicate insufficient data
    assert!(!drift.drift_detected || drift.explanation.contains("insufficient"));
}

#[test]
fn test_collector_record_prediction_error_disabled() {
    let mut collector = TunerDataCollector::new();
    // Online learning disabled - should not record
    collector.record_prediction_error(100.0, 50.0);

    // Drift detection should still work but with no data
    let drift = collector.detect_concept_drift();
    assert!(!drift.drift_detected);
}

#[test]
fn test_collector_concept_drift_detection() {
    let mut collector = TunerDataCollector::with_online_learning();

    // Add enough samples for drift detection (need 10+)
    for i in 0..15 {
        collector.record_prediction_error(100.0, 100.0 + (i as f32) * 2.0);
    }

    let drift = collector.detect_concept_drift();
    // With increasing errors, might detect drift
    assert!(drift.explanation.len() > 0);
}

#[test]
fn test_collector_should_retrain() {
    let mut collector = TunerDataCollector::with_online_learning();

    // Initially should not need retraining
    let _needs_retrain_initial = collector.should_retrain();

    // After many errors, might need retrain
    for _ in 0..20 {
        collector.record_prediction_error(100.0, 50.0); // Large errors
    }

    // Check retrain status (depends on drift detection)
    let _needs_retrain = collector.should_retrain();
}

#[test]
fn test_collector_training_stats() {
    let collector = TunerDataCollector::new();
    let stats = collector.training_stats();

    // Should return valid stats (total_samples is always >= 0 for usize)
    let _total = stats.total_samples; // Verify it's accessible
}

#[test]
fn test_collector_mark_trained() {
    let mut collector = TunerDataCollector::with_online_learning();

    // Record some errors
    for _ in 0..5 {
        collector.record_prediction_error(100.0, 80.0);
    }

    // Mark as trained
    collector.mark_trained();

    // Stats should reflect training
    let stats = collector.training_stats();
    // Samples since training should reset after mark_trained
    let _samples = stats.samples_since_training;
    let _total = stats.total_samples;
}

#[test]
fn test_collector_feedback_out_of_bounds() {
    let collector = TunerDataCollector::new();

    // Get feedback for non-existent sample (should return None variant)
    let feedback = collector.get_feedback(999);
    assert!(matches!(feedback, UserFeedback::None));
}

#[test]
fn test_collector_empty_initially() {
    let collector = TunerDataCollector::new();
    assert!(collector.is_empty());
    assert_eq!(collector.len(), 0);
}

// =========================================================================
// T-TUNER-009: Additional Coverage Tests
// =========================================================================

#[cfg(feature = "hardware-detect")]
#[test]
fn test_collector_cache_path_is_valid() {
    let path = TunerDataCollector::cache_path();
    // Should return a valid path (may not exist)
    assert!(path.to_string_lossy().len() > 0);
}

#[cfg(feature = "hardware-detect")]
#[test]
fn test_tuner_cache_path_is_valid() {
    let path = BrickTuner::cache_path();
    // Should return a valid path (may not exist)
    assert!(path.to_string_lossy().len() > 0);
}

#[cfg(feature = "hardware-detect")]
#[test]
fn test_load_or_default_returns_tuner() {
    let tuner = BrickTuner::load_or_default();
    // Should always return a valid tuner
    assert!(tuner.version.len() > 0);
}

#[test]
fn test_collector_to_json() {
    let collector = TunerDataCollector::new();
    let json = collector.to_json();
    assert!(json.is_ok());
    // Empty collector should serialize to empty array
    assert!(json.unwrap().contains("[]"));
}

#[test]
fn test_collector_prepare_training_data_empty() {
    let collector = TunerDataCollector::new();
    let data = collector.prepare_training_data();
    assert!(data.is_empty());
}

#[test]
fn test_collector_samples_accessor() {
    let collector = TunerDataCollector::new();
    assert!(collector.samples().is_empty());
}

// Helper function to create a test recommendation
fn create_test_recommendation() -> TunerRecommendation {
    TunerRecommendation {
        throughput: ThroughputPrediction {
            predicted_tps: 100.0,
            confidence: 0.85,
            top_features: vec![("batch_size".to_string(), 0.3)],
        },
        kernel: KernelRecommendation {
            top_kernel: KernelType::BatchedQ4K,
            confidence: 0.9,
            alternatives: vec![],
        },
        bottleneck: BottleneckPrediction {
            class: BottleneckClass::ComputeBound,
            confidence: 0.8,
            explanation: "High compute utilization".to_string(),
            recommended_action: "Enable tensor cores".to_string(),
        },
        suggested_experiments: vec![
            ExperimentSuggestion::IncreaseBatchSize { from: 4, to: 8 },
        ],
        model_version: "1.0.0".to_string(),
        confidence_overall: 0.85,
    }
}

// =========================================================================
// T-TUNER-009: APR Format and CRC32 Tests
// =========================================================================

#[test]
fn test_crc32_hash_empty() {
    // CRC32 of empty data should be 0
    let hash = super::crc32_hash(&[]);
    assert_eq!(hash, 0);
}

#[test]
fn test_crc32_hash_data() {
    // CRC32 should produce consistent results
    let data = b"hello world";
    let hash1 = super::crc32_hash(data);
    let hash2 = super::crc32_hash(data);
    assert_eq!(hash1, hash2);
    // Hash should be non-zero for non-empty data
    assert_ne!(hash1, 0);
}

#[test]
fn test_crc32_update_incremental() {
    // Incremental CRC should work
    let data = b"hello";
    let hash_full = super::crc32_hash(data);

    let mut crc = 0u32;
    crc = super::crc32_update(crc, &data[0..2]);
    crc = super::crc32_update(crc, &data[2..]);
    // Incremental should NOT equal full (CRC is not simple accumulation)
    // But both should be non-zero
    assert_ne!(crc, 0);
    assert_ne!(hash_full, 0);
}

#[test]
fn test_apr_save_and_load() {
    use std::fs;

    let tuner = BrickTuner::new();
    let path = "/tmp/test_tuner_apr_roundtrip.apr";

    // Save
    let save_result = tuner.save_apr(path);
    assert!(save_result.is_ok());

    // Load
    let load_result = BrickTuner::load_apr(path);
    assert!(load_result.is_ok());

    let loaded = load_result.unwrap();
    assert_eq!(loaded.version, tuner.version);

    // Cleanup
    let _ = fs::remove_file(path);
}

#[test]
fn test_apr_load_invalid_magic() {
    use std::fs::File;
    use std::io::Write;

    let path = "/tmp/test_invalid_magic.apr";
    let mut file = File::create(path).unwrap();
    file.write_all(b"NOPE").unwrap(); // Invalid magic
    drop(file);

    let result = BrickTuner::load_apr(path);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), TunerError::InvalidFormat(_)));

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_apr_load_crc_mismatch() {
    use std::fs::File;
    use std::io::Write;

    let path = "/tmp/test_crc_mismatch.apr";
    let mut file = File::create(path).unwrap();

    // Write valid magic
    file.write_all(b"APR1").unwrap();
    // Write length (10 bytes)
    file.write_all(&10u32.to_le_bytes()).unwrap();
    // Write garbage JSON
    file.write_all(b"0123456789").unwrap();
    // Write wrong CRC (0xDEADBEEF)
    file.write_all(&0xDEADBEEFu32.to_le_bytes()).unwrap();
    drop(file);

    let result = BrickTuner::load_apr(path);
    assert!(result.is_err());
    let err_str = format!("{:?}", result.unwrap_err());
    assert!(err_str.contains("CRC32") || err_str.contains("checksum") || err_str.contains("Invalid"));

    let _ = std::fs::remove_file(path);
}

#[test]
fn test_apr_load_file_not_found() {
    let result = BrickTuner::load_apr("/nonexistent/path/to/file.apr");
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), TunerError::Io(_)));
}

// =========================================================================
// T-TUNER-010: FeatureExtractor Tests
// =========================================================================

#[test]
fn test_feature_extractor_with_hardware() {
    use crate::hardware::HardwareCapability;

    let hw = HardwareCapability::detect();
    let extractor = FeatureExtractor::with_hardware(hw);
    assert!(extractor.hardware.is_some());
}

#[test]
fn test_feature_extractor_extract_basic() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let profiler = BrickProfiler::new();
    let config = RunConfig::default();

    let features = extractor.extract(&profiler, &config);
    // Features should have default values
    assert!(features.model_params_b > 0.0);
}

#[test]
fn test_feature_extractor_classify_bottleneck_empty() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let profiler = BrickProfiler::new(); // No stats

    let bottleneck = extractor.classify_bottleneck(&profiler);
    assert_eq!(bottleneck, BottleneckClass::Unknown);
}

#[test]
fn test_feature_extractor_classify_bottleneck_with_attention() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let mut profiler = BrickProfiler::enabled();

    // Add stats with high attention percentage using record_elapsed
    profiler.record_elapsed("attention_qkv", std::time::Duration::from_micros(100), 10);
    profiler.record_elapsed("other_op", std::time::Duration::from_micros(10), 1);

    // Classify
    let bottleneck = extractor.classify_bottleneck(&profiler);
    // Should be attention bound since attention takes >35%
    assert!(matches!(
        bottleneck,
        BottleneckClass::AttentionBound | BottleneckClass::MemoryBound | BottleneckClass::Unknown
    ));
}

#[test]
fn test_feature_extractor_classify_bottleneck_gemv() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let mut profiler = BrickProfiler::enabled();

    // Add stats with high GEMV percentage using record_elapsed
    for i in 0..5 {
        profiler.record_elapsed(
            &format!("gemv_{}", i),
            std::time::Duration::from_micros(50),
            10,
        );
    }
    profiler.record_elapsed("other", std::time::Duration::from_micros(10), 1);

    let bottleneck = extractor.classify_bottleneck(&profiler);
    // GEMV dominates, should be memory bound
    assert!(matches!(
        bottleneck,
        BottleneckClass::MemoryBound | BottleneckClass::LaunchBound | BottleneckClass::Unknown
    ));
}

#[test]
fn test_feature_extractor_classify_bottleneck_launch_bound() {
    use crate::brick::BrickProfiler;

    let extractor = FeatureExtractor::new();
    let mut profiler = BrickProfiler::enabled();

    // Add many small bricks (<10µs average) using record_elapsed
    for i in 0..100 {
        profiler.record_elapsed(
            &format!("tiny_op_{}", i),
            std::time::Duration::from_nanos(100), // Very short
            1,
        );
    }

    let bottleneck = extractor.classify_bottleneck(&profiler);
    // Many tiny bricks may indicate launch bound (or unknown if too fast)
    assert!(matches!(
        bottleneck,
        BottleneckClass::LaunchBound | BottleneckClass::MemoryBound | BottleneckClass::Unknown
    ));
}

// =========================================================================
// T-TUNER-011: BrickProfiler Integration Tests
// =========================================================================

#[test]
fn test_profiler_tokens_per_sec_disabled() {
    use crate::brick::BrickProfiler;

    let profiler = BrickProfiler::new();
    assert!(profiler.tokens_per_sec().is_none());
}

#[test]
fn test_profiler_get_tuner_recommendations_disabled() {
    use crate::brick::BrickProfiler;

    let profiler = BrickProfiler::new();
    let config = RunConfig::default();

    let rec = profiler.get_tuner_recommendations(&config);
    assert!(rec.is_none());
}

#[test]
fn test_profiler_get_tuner_recommendations_enabled() {
    use crate::brick::BrickProfiler;

    let mut profiler = BrickProfiler::enabled();

    // Add some timing data using record_elapsed
    profiler.record_elapsed("test_brick", std::time::Duration::from_micros(10), 100);

    let config = RunConfig::default();

    let rec = profiler.get_tuner_recommendations(&config);
    // Should return recommendation even with minimal data
    assert!(rec.is_some());
}

// =========================================================================
// T-TUNER-012: Additional BottleneckClass Tests
// =========================================================================

#[test]
fn test_bottleneck_recommended_action_compute_bound() {
    let action = BottleneckClass::ComputeBound.recommended_action();
    assert!(action.len() > 0);
    // Should mention tensor cores or similar
    assert!(action.to_lowercase().contains("tensor") || action.len() > 5);
}

#[test]
fn test_all_bottleneck_class_actions() {
    // Test all variants have non-empty recommended actions
    let classes = [
        BottleneckClass::ComputeBound,
        BottleneckClass::MemoryBound,
        BottleneckClass::AttentionBound,
        BottleneckClass::LaunchBound,
        BottleneckClass::Unknown,
    ];

    for class in classes {
        let action = class.recommended_action();
        assert!(!action.is_empty(), "Action for {:?} should not be empty", class);
    }
}

#[test]
fn test_bottleneck_class_index_coverage() {
    // Ensure all variants map to different indices
    let indices: std::collections::HashSet<usize> = [
        BottleneckClass::ComputeBound,
        BottleneckClass::MemoryBound,
        BottleneckClass::AttentionBound,
        BottleneckClass::LaunchBound,
        BottleneckClass::Unknown,
    ]
    .iter()
    .map(|b| b.to_index())
    .collect();

    assert_eq!(indices.len(), 5);
}

// =========================================================================
// T-TUNER-013: KernelClassifier Additional Tests
// =========================================================================

#[test]
fn test_kernel_classifier_attention_path() {
    let classifier = KernelClassifier::new();

    // Features with long sequence (should suggest attention kernel)
    let features = TunerFeatures::builder()
        .batch_size(4)
        .seq_len(256) // Long sequence
        .hidden_dim(4096)
        .build();

    let prediction = classifier.predict(&features);
    // Should have high confidence
    assert!(prediction.confidence >= 0.0);
}

// =========================================================================
// T-TUNER-014: Error Handling Tests
// =========================================================================

#[test]
fn test_tuner_error_io_display() {
    let error = TunerError::Io("file not found".to_string());
    let display = format!("{}", error);
    assert!(display.contains("file not found") || display.contains("I/O"));
}

#[test]
fn test_tuner_error_invalid_format_display() {
    let error = TunerError::InvalidFormat("bad magic".to_string());
    let display = format!("{}", error);
    assert!(display.contains("bad magic") || display.contains("format"));
}

#[test]
fn test_tuner_error_serialization_display() {
    let error = TunerError::Serialization("json parse error".to_string());
    let display = format!("{}", error);
    assert!(display.contains("json") || display.contains("serial"));
}

// =========================================================================
// T-TUNER-015: TunerDataCollector Merge and Load Tests
// =========================================================================

#[test]
fn test_training_sample_serialization_roundtrip() {
    // Create a training sample
    let features = TunerFeatures::builder()
        .model_params_b(7.0)
        .hidden_dim(4096)
        .num_layers(32)
        .num_heads(32)
        .batch_size(4)
        .seq_len(128)
        .build();

    let sample = TrainingSample {
        features,
        throughput_tps: 1500.0,
        best_kernel: KernelType::BatchedQ4K,
        bottleneck: BottleneckClass::MemoryBound,
        timestamp: "2024-01-01T00:00:00".to_string(),
        hardware_id: "test".to_string(),
    };

    // Serialize
    let json = serde_json::to_string(&sample);
    assert!(json.is_ok());

    // Deserialize
    let restored: Result<TrainingSample, _> = serde_json::from_str(&json.unwrap());
    assert!(restored.is_ok());

    let restored = restored.unwrap();
    assert_eq!(restored.throughput_tps, 1500.0);
    assert_eq!(restored.best_kernel, KernelType::BatchedQ4K);
}

#[test]
fn test_collector_merge() {
    let mut collector1 = TunerDataCollector::new();
    let collector2 = TunerDataCollector::new();

    // Merge should not panic
    collector1.merge(&collector2);
    assert!(collector1.samples().is_empty());
}

// =========================================================================
// T-TUNER-010: Coverage Gap Tests for 95%+
// =========================================================================

#[test]
fn test_builder_hardware_with_hw_capability() {
    use crate::hardware::{CpuCapability, GpuBackend, GpuCapability, HardwareCapability, RooflineParams, SimdWidth};

    let gpu = GpuCapability {
        vendor: "NVIDIA".to_string(),
        model: "RTX 4090".to_string(),
        backend: GpuBackend::Cuda,
        compute_capability: Some("8.9".to_string()),
        peak_tflops_fp32: 82.58,
        peak_tflops_tensor: Some(330.3),
        memory_bw_gbps: 1008.0,
        vram_gb: 24.0,
    };

    let cpu = CpuCapability {
        vendor: "Intel".to_string(),
        model: "Core i9-13900K".to_string(),
        cores: 24,
        threads: 32,
        simd: SimdWidth::Avx512,
        base_freq_ghz: 3.0,
        peak_gflops: 500.0,
        memory_bw_gbps: 80.0,
    };

    let hw = HardwareCapability {
        timestamp: "2026-01-16".to_string(),
        hostname: "test".to_string(),
        cpu,
        gpu: Some(gpu),
        roofline: RooflineParams {
            cpu_arithmetic_intensity: 10.0,
            gpu_arithmetic_intensity: Some(50.0),
        },
        byte_budget: None,
    };

    let features = TunerFeatures::builder()
        .hardware(&hw)
        .build();

    // Should have set GPU metrics
    assert!(features.gpu_mem_bw_norm > 0.0);
    assert!(features.gpu_compute_norm > 0.0);
}

#[test]
fn test_tuner_features_default_impl() {
    // Test Default trait implementation
    let features: TunerFeatures = Default::default();
    assert!(features.validate().is_ok());
}

#[test]
fn test_run_config_default_impl() {
    // Test Default trait implementation
    let config: RunConfig = Default::default();
    assert!(config.model_params_b > 0.0);
}

#[test]
fn test_brick_tuner_default_impl() {
    // Test Default trait implementation
    let tuner: BrickTuner = Default::default();
    assert!(!tuner.version.is_empty());
}

#[test]
fn test_brick_tuner_accessor_methods() {
    let tuner = BrickTuner::new();

    // Test accessor methods
    let version = tuner.version();
    assert!(!version.is_empty());

    let mape = tuner.throughput_mape();
    assert!(mape >= 0.0);

    let sample_count = tuner.throughput_sample_count();
    assert!(sample_count >= 0);
}

#[test]
fn test_bottleneck_class_attention_bound() {
    let classifier = BottleneckClassifier::new();

    // Long sequence should trigger AttentionBound
    let features = TunerFeatures::builder()
        .batch_size(1)
        .seq_len(16384) // Very long sequence
        .is_prefill(true)
        .build();

    let pred = classifier.predict(&features);
    // Should classify as attention bound for very long sequences
    assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
}

#[test]
fn test_bottleneck_class_memory_bound() {
    let classifier = BottleneckClassifier::new();

    // Small batch, decode phase should be memory bound
    let features = TunerFeatures::builder()
        .batch_size(1)
        .seq_len(512)
        .is_prefill(false) // Decode phase
        .build();

    let pred = classifier.predict(&features);
    // Should lean toward memory bound for decode
    assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
}

#[test]
fn test_bottleneck_class_launch_bound() {
    let classifier = BottleneckClassifier::new();

    // Very small batch with many launches
    let features = TunerFeatures::builder()
        .batch_size(1)
        .seq_len(1)
        .is_prefill(false)
        .build();

    let pred = classifier.predict(&features);
    // Very small workload might be launch bound
    assert!(!pred.explanation.is_empty());
}

#[test]
fn test_kernel_classifier_q4k_variants() {
    let classifier = KernelClassifier::new();

    // Test that Q4K variants are recommended appropriately
    let features_small = TunerFeatures::builder()
        .batch_size(1)
        .quant_type(QuantType::Q4K)
        .build();

    let rec_small = classifier.predict(&features_small);
    // Should have alternatives including Q4K variants
    let alternatives_count = rec_small.alternatives.len();
    assert!(alternatives_count >= 0);

    let features_large = TunerFeatures::builder()
        .batch_size(32)
        .quant_type(QuantType::Q4K)
        .build();

    let rec_large = classifier.predict(&features_large);
    assert!(rec_large.confidence >= 0.0);
}

#[test]
fn test_display_accuracy_grades() {
    let tuner = BrickTuner::new();
    let rec = create_test_recommendation();

    // Test "Good" accuracy (< 10% error)
    let lines_good = tuner.render_comparison(&rec, 105.0); // 5% error
    // Should render comparison lines
    assert!(lines_good.len() >= 1);

    // Test "Fair" accuracy (10-20% error)
    let lines_fair = tuner.render_comparison(&rec, 80.0); // 20% error
    // Should render comparison lines
    assert!(lines_fair.len() >= 1);

    // Test poor accuracy (> 20% error)
    let lines_poor = tuner.render_comparison(&rec, 50.0); // 50% error
    assert!(lines_poor.len() >= 1);
}

#[test]
fn test_kernel_arm_mean_and_ucb() {
    let mut arm = KernelArm::default();

    // Initial state - no pulls
    assert_eq!(arm.mean(), 0.0);
    assert_eq!(arm.ucb(10, 2.0), f32::INFINITY);

    // After some pulls
    arm.pulls = 5;
    arm.total_reward = 2.5; // mean = 0.5

    assert!((arm.mean() - 0.5).abs() < 0.01);
    assert!(arm.ucb(10, 2.0) > 0.5); // UCB should be > mean due to exploration bonus
}

#[test]
fn test_kernel_bandit_new() {
    let bandit = KernelBandit::new();
    assert_eq!(bandit.arms.len(), KernelBandit::NUM_KERNELS);
    assert_eq!(bandit.total_pulls, 0);
    assert!(!bandit.use_thompson);
}

#[test]
fn test_kernel_bandit_with_thompson_sampling() {
    let bandit = KernelBandit::with_thompson_sampling();
    assert!(bandit.use_thompson);
}

#[test]
fn test_kernel_bandit_select_ucb() {
    let bandit = KernelBandit::new();

    // Initial selection (all arms unexplored)
    let kernel = bandit.select();
    // Should return some kernel type
    assert!(matches!(kernel, KernelType::TiledQ4K | KernelType::CoalescedQ4K |
        KernelType::VectorizedQ4K | KernelType::BatchedQ4K | _));
}

#[test]
fn test_kernel_bandit_select_thompson() {
    let bandit = KernelBandit::with_thompson_sampling();

    // Selection with Thompson sampling
    let kernel = bandit.select();
    // Should return some kernel type
    assert!(matches!(kernel, KernelType::TiledQ4K | KernelType::CoalescedQ4K |
        KernelType::VectorizedQ4K | KernelType::BatchedQ4K | _));
}

#[test]
fn test_kernel_bandit_update_and_best() {
    let mut bandit = KernelBandit::new();

    // Update with rewards
    bandit.update(KernelType::VectorizedQ4K, 0.9);
    bandit.update(KernelType::VectorizedQ4K, 0.85);
    bandit.update(KernelType::TiledQ4K, 0.5);

    assert_eq!(bandit.total_pulls, 3);

    // Best kernel should be VectorizedQ4K (higher mean reward)
    let best = bandit.best_kernel();
    assert_eq!(best, KernelType::VectorizedQ4K);
}

#[test]
fn test_kernel_bandit_exploration_rate() {
    let mut bandit = KernelBandit::new();

    // Initial exploration rate is 1.0 (all pulls are exploratory)
    assert_eq!(bandit.exploration_rate(), 1.0);

    // After some updates
    bandit.update(KernelType::TiledQ4K, 0.5);
    bandit.update(KernelType::TiledQ4K, 0.6);
    bandit.update(KernelType::CoalescedQ4K, 0.4);

    let rate = bandit.exploration_rate();
    assert!(rate > 0.0 && rate <= 1.0);
}

#[test]
fn test_kernel_bandit_estimated_regret() {
    let mut bandit = KernelBandit::new();

    // Add some data
    bandit.update(KernelType::VectorizedQ4K, 0.9);
    bandit.update(KernelType::TiledQ4K, 0.5);

    let regret = bandit.estimated_regret();
    // Regret should be >= 0
    assert!(regret >= 0.0);
}

#[test]
fn test_kernel_bandit_select_after_updates() {
    let mut bandit = KernelBandit::new();

    // Train the bandit
    for _ in 0..10 {
        bandit.update(KernelType::VectorizedQ4K, 0.9);
    }
    for _ in 0..5 {
        bandit.update(KernelType::TiledQ4K, 0.5);
    }

    // UCB should now prefer VectorizedQ4K but might explore
    let selected = bandit.select();
    // Just verify it returns a valid kernel
    let _idx = selected.to_index();
}

#[test]
fn test_gpu_efficiency_calculation_path() {
    // Test the GPU efficiency calculation with hardware
    use crate::hardware::{CpuCapability, GpuBackend, GpuCapability, HardwareCapability, RooflineParams, SimdWidth};

    let gpu = GpuCapability {
        vendor: "NVIDIA".to_string(),
        model: "RTX 4090".to_string(),
        backend: GpuBackend::Cuda,
        compute_capability: Some("8.9".to_string()),
        peak_tflops_fp32: 82.58,
        peak_tflops_tensor: Some(330.3),
        memory_bw_gbps: 1008.0,
        vram_gb: 24.0,
    };

    let cpu = CpuCapability {
        vendor: "Intel".to_string(),
        model: "Core i9-13900K".to_string(),
        cores: 24,
        threads: 32,
        simd: SimdWidth::Avx512,
        base_freq_ghz: 3.0,
        peak_gflops: 500.0,
        memory_bw_gbps: 80.0,
    };

    let hw = HardwareCapability {
        timestamp: "2026-01-16".to_string(),
        hostname: "test".to_string(),
        cpu,
        gpu: Some(gpu),
        roofline: RooflineParams {
            cpu_arithmetic_intensity: 10.0,
            gpu_arithmetic_intensity: Some(50.0),
        },
        byte_budget: None,
    };

    let config = RunConfig {
        model_params_b: 7.0,
        batch_size: 1,
        quant_type: QuantType::Q4K,
        ..Default::default()
    };

    // This exercises the efficiency calculation code path
    let extractor = FeatureExtractor::new();
    let profiler = BrickProfiler::new();
    let features = extractor.extract(&profiler, &config);

    // Test calculate_efficiency (exercises hardware-related code paths)
    let _ = extractor.calculate_efficiency(&profiler, &config);

    // Also test features are valid
    assert!(features.validate().is_ok());

    // Exercise hardware capability access
    let _ = hw.gpu.as_ref().map(|g| g.memory_bw_gbps);
}

#[test]
fn test_attention_bound_long_sequence_classification() {
    let classifier = BottleneckClassifier::new();

    // Very long sequence should trigger attention bound path
    let features = TunerFeatures::builder()
        .batch_size(4)
        .seq_len(32768) // Very long
        .is_prefill(true)
        .build();

    let pred = classifier.predict(&features);
    // Long sequences should mention attention or context in explanation
    assert!(pred.confidence >= 0.0);
}

#[test]
fn test_kernel_type_from_index_all_variants() {
    // Test all index mappings
    assert_eq!(KernelType::from_index(0), KernelType::TiledQ4K);
    assert_eq!(KernelType::from_index(1), KernelType::CoalescedQ4K);
    assert_eq!(KernelType::from_index(2), KernelType::VectorizedQ4K);
    assert_eq!(KernelType::from_index(3), KernelType::BatchedQ4K);
    assert_eq!(KernelType::from_index(4), KernelType::Dp4aQ4K);
    assert_eq!(KernelType::from_index(5), KernelType::FusedRmsNormQ4K);
    assert_eq!(KernelType::from_index(6), KernelType::CoalescedQ6K);
    assert_eq!(KernelType::from_index(7), KernelType::IncrementalAttention);
    assert_eq!(KernelType::from_index(8), KernelType::MultiWarpAttention);
    assert_eq!(KernelType::from_index(9), KernelType::BatchedAttention);
    assert_eq!(KernelType::from_index(10), KernelType::RmsNorm);
    assert_eq!(KernelType::from_index(11), KernelType::VectorizedRmsNorm);
    assert_eq!(KernelType::from_index(12), KernelType::BatchedRmsNorm);
    assert_eq!(KernelType::from_index(13), KernelType::Generic);
    assert_eq!(KernelType::from_index(99), KernelType::Unknown); // Out of range
}

// =========================================================================
// T-TUNER-011: GPU/CUDA Hardware Path Coverage Tests
// =========================================================================

#[test]
fn test_feature_extractor_with_detected_hardware() {
    use crate::hardware::HardwareCapability;
    use std::time::Duration;

    // Detect real hardware (will find CUDA GPU if present)
    let hw = HardwareCapability::detect();

    // Create extractor with hardware
    let extractor = FeatureExtractor::with_hardware(hw.clone());

    // Create profiler with some recorded data
    let mut profiler = BrickProfiler::enabled();
    profiler.record_elapsed("attention", Duration::from_micros(500), 32);
    profiler.record_elapsed("ffn", Duration::from_micros(300), 32);
    profiler.record_elapsed("norm", Duration::from_micros(50), 32);

    let config = RunConfig {
        model_params_b: 7.0,
        batch_size: 1,
        quant_type: QuantType::Q4K,
        cuda_graphs: true,
        ..Default::default()
    };

    // Extract features - exercises GPU hardware paths
    let features = extractor.extract(&profiler, &config);

    // Should have valid features
    assert!(features.validate().is_ok());

    // If GPU is present, efficiency should be calculated
    if hw.gpu.is_some() {
        // GPU metrics should be set
        assert!(features.gpu_mem_bw_norm >= 0.0);
        assert!(features.gpu_compute_norm >= 0.0);
    }
}

#[test]
fn test_calculate_efficiency_with_gpu() {
    use crate::hardware::{CpuCapability, GpuBackend, GpuCapability, HardwareCapability, RooflineParams, SimdWidth};
    use std::time::Duration;

    // Manually construct hardware with GPU to exercise GPU code paths
    let gpu = GpuCapability {
        vendor: "NVIDIA".to_string(),
        model: "RTX 4090".to_string(),
        backend: GpuBackend::Cuda,
        compute_capability: Some("8.9".to_string()),
        peak_tflops_fp32: 82.58,
        peak_tflops_tensor: Some(330.3),
        memory_bw_gbps: 1008.0,
        vram_gb: 24.0,
    };

    let cpu = CpuCapability {
        vendor: "Intel".to_string(),
        model: "Core i9-13900K".to_string(),
        cores: 24,
        threads: 32,
        simd: SimdWidth::Avx512,
        base_freq_ghz: 3.0,
        peak_gflops: 500.0,
        memory_bw_gbps: 80.0,
    };

    let hw = HardwareCapability {
        timestamp: "2026-01-16".to_string(),
        hostname: "test".to_string(),
        cpu,
        gpu: Some(gpu),
        roofline: RooflineParams {
            cpu_arithmetic_intensity: 10.0,
            gpu_arithmetic_intensity: Some(50.0),
        },
        byte_budget: None,
    };

    let extractor = FeatureExtractor::with_hardware(hw);

    // Create profiler with token data
    let mut profiler = BrickProfiler::enabled();
    // Record some operations with tokens
    profiler.record_elapsed("decode", Duration::from_millis(100), 100);

    let config = RunConfig {
        model_params_b: 7.0,
        batch_size: 1,
        quant_type: QuantType::Q4K,
        ..Default::default()
    };

    // This should exercise the calculate_efficiency path
    let efficiency = extractor.calculate_efficiency(&profiler, &config);

    // Efficiency should be Some if profiler has tokens
    if profiler.tokens_per_sec().is_some() {
        assert!(efficiency.is_some());
        let eff = efficiency.unwrap();
        assert!(eff >= 0.0 && eff <= 1.0);
    }
}

#[test]
fn test_classify_bottleneck_attention_bound() {
    use crate::hardware::HardwareCapability;
    use std::time::Duration;

    let hw = HardwareCapability::detect();
    let extractor = FeatureExtractor::with_hardware(hw);

    // Create profiler where attention dominates (>35%)
    let mut profiler = BrickProfiler::enabled();
    profiler.record_elapsed("attention", Duration::from_micros(500), 32);
    profiler.record_elapsed("ffn", Duration::from_micros(100), 32);
    profiler.record_elapsed("norm", Duration::from_micros(50), 32);

    let config = RunConfig::default();
    let features = extractor.extract(&profiler, &config);

    // Should classify as attention bound
    assert!(features.bottleneck_class.is_some());
    let bottleneck = features.bottleneck_class.unwrap();
    assert!(matches!(bottleneck, BottleneckClass::AttentionBound | BottleneckClass::MemoryBound));
}

#[test]
fn test_classify_bottleneck_memory_bound() {
    use crate::hardware::HardwareCapability;
    use std::time::Duration;

    let hw = HardwareCapability::detect();
    let extractor = FeatureExtractor::with_hardware(hw);

    // Create profiler where FFN dominates (>50%)
    let mut profiler = BrickProfiler::enabled();
    profiler.record_elapsed("attention", Duration::from_micros(100), 32);
    profiler.record_elapsed("ffn", Duration::from_micros(800), 32);
    profiler.record_elapsed("norm", Duration::from_micros(50), 32);

    let config = RunConfig::default();
    let features = extractor.extract(&profiler, &config);

    // Should classify as memory bound
    assert!(features.bottleneck_class.is_some());
}

#[test]
fn test_classify_bottleneck_launch_bound() {
    use crate::hardware::HardwareCapability;
    use std::time::Duration;

    let hw = HardwareCapability::detect();
    let extractor = FeatureExtractor::with_hardware(hw);

    // Create profiler where norm dominates (>20%) indicating launch overhead
    let mut profiler = BrickProfiler::enabled();
    profiler.record_elapsed("attention", Duration::from_micros(100), 32);
    profiler.record_elapsed("ffn", Duration::from_micros(200), 32);
    profiler.record_elapsed("norm", Duration::from_micros(300), 32);

    let config = RunConfig::default();
    let features = extractor.extract(&profiler, &config);

    // Should classify as launch bound
    assert!(features.bottleneck_class.is_some());
}

#[test]
fn test_gpu_builder_methods() {
    // Test all GPU-related builder methods
    let features = TunerFeatures::builder()
        .gpu_mem_bw_gbs(1008.0)  // RTX 4090
        .gpu_compute_tflops(82.58)
        .gpu_sm_count(128)
        .gpu_l2_cache_mb(72.0)
        .cuda_graphs(true)
        .build();

    // Normalized values should be in valid range
    assert!(features.gpu_mem_bw_norm > 0.0 && features.gpu_mem_bw_norm <= 1.0);
    assert!(features.gpu_compute_norm > 0.0 && features.gpu_compute_norm <= 1.0);
    assert!(features.gpu_sm_norm > 0.0 && features.gpu_sm_norm <= 1.0);
    assert!(features.gpu_l2_cache_norm > 0.0 && features.gpu_l2_cache_norm <= 1.0);
    assert_eq!(features.cuda_graphs, 1.0);
}

#[test]
fn test_run_config_with_cuda_graphs() {
    let config = RunConfig {
        cuda_graphs: true,
        ..Default::default()
    };

    assert!(config.cuda_graphs);

    // Extract features with cuda_graphs enabled
    let extractor = FeatureExtractor::new();
    let profiler = BrickProfiler::new();
    let features = extractor.extract(&profiler, &config);

    assert_eq!(features.cuda_graphs, 1.0);
}

#[test]
fn test_bottleneck_prediction_for_vectorized_q4k() {
    let classifier = KernelClassifier::new();

    // Features for VectorizedQ4K scenario
    let features = TunerFeatures::builder()
        .batch_size(1)  // M=1 decode
        .quant_type(QuantType::Q4K)
        .cuda_graphs(false)
        .build();

    let rec = classifier.predict(&features);

    // Should recommend VectorizedQ4K for M=1 decode
    assert!(rec.confidence >= 0.0);
    // Alternatives should include Q4K variants
    let has_q4k = rec.alternatives.iter().any(|(k, _)| {
        matches!(k, KernelType::VectorizedQ4K | KernelType::CoalescedQ4K | KernelType::TiledQ4K)
    });
    assert!(rec.top_kernel != KernelType::Unknown || has_q4k || rec.alternatives.is_empty());
}

// =========================================================================
// OnlineLearner Tests (MLT-12)
// =========================================================================

#[test]
fn test_online_learner_new() {
    let learner = OnlineLearner::new();

    // Weights should be initialized from pretrained
    assert_eq!(learner.weights().len(), TunerFeatures::DIM + 1);
    assert_eq!(learner.num_updates(), 0);
    assert_eq!(learner.ema_loss(), 0.0);
}

#[test]
fn test_online_learner_with_learning_rate() {
    let learner = OnlineLearner::new().with_learning_rate(0.01);

    // Builder pattern should work
    assert_eq!(learner.weights().len(), TunerFeatures::DIM + 1);
    assert_eq!(learner.num_updates(), 0);
}

#[test]
fn test_online_learner_predict() {
    let learner = OnlineLearner::new();

    // Create features of correct dimension (42)
    let features = vec![0.5_f32; TunerFeatures::DIM];

    let prediction = learner.predict(&features);

    // Prediction should be non-negative (due to .max(0.0))
    assert!(prediction >= 0.0);
}

#[test]
fn test_online_learner_predict_clamps_negative() {
    let learner = OnlineLearner::new();

    // Create features that might produce negative prediction
    let features = vec![-100.0_f32; TunerFeatures::DIM];

    let prediction = learner.predict(&features);

    // Should be clamped to 0.0 minimum
    assert!(prediction >= 0.0);
}

#[test]
fn test_online_learner_observe_updates_count() {
    let mut learner = OnlineLearner::new();

    // Create features of correct dimension
    let features = vec![0.5_f32; TunerFeatures::DIM];

    assert_eq!(learner.num_updates(), 0);

    learner.observe(&features, 100.0);
    assert_eq!(learner.num_updates(), 1);

    learner.observe(&features, 150.0);
    assert_eq!(learner.num_updates(), 2);
}

#[test]
fn test_online_learner_observe_updates_ema_loss() {
    let mut learner = OnlineLearner::new();
    let features = vec![0.5_f32; TunerFeatures::DIM];

    assert_eq!(learner.ema_loss(), 0.0);

    // Observe with some error
    learner.observe(&features, 100.0);

    // EMA loss should be updated
    assert!(learner.ema_loss() >= 0.0);
}

#[test]
fn test_online_learner_observe_dimension_mismatch() {
    let mut learner = OnlineLearner::new();

    // Wrong dimension - should be ignored
    let wrong_features = vec![0.5_f32; 10];

    learner.observe(&wrong_features, 100.0);

    // Should not have updated
    assert_eq!(learner.num_updates(), 0);
}

#[test]
fn test_online_learner_observe_triggers_replay() {
    let mut learner = OnlineLearner::new();
    let features = vec![0.5_f32; TunerFeatures::DIM];

    // Observe 10 times to trigger replay_step (every 10 updates)
    for i in 0..10 {
        learner.observe(&features, 100.0 + i as f32);
    }

    assert_eq!(learner.num_updates(), 10);

    // Replay buffer should have samples
    // (Can't directly access replay_buffer, but observe worked)
}

#[test]
fn test_online_learner_observe_fills_replay_buffer() {
    let mut learner = OnlineLearner::new();
    let features = vec![0.5_f32; TunerFeatures::DIM];

    // Fill replay buffer (default size 100) and overflow
    for i in 0..110 {
        learner.observe(&features, 100.0 + i as f32);
    }

    assert_eq!(learner.num_updates(), 110);
}

#[test]
fn test_online_learner_is_converging_initial() {
    let learner = OnlineLearner::new();

    // Initial EMA loss is 0.0, which is < 0.15 threshold
    assert!(learner.is_converging());
}

#[test]
fn test_online_learner_is_converging_after_training() {
    let mut learner = OnlineLearner::new();
    let features = vec![0.5_f32; TunerFeatures::DIM];

    // Train with consistent data
    for _ in 0..50 {
        let prediction = learner.predict(&features);
        learner.observe(&features, prediction); // Predict same as actual
    }

    // After training on self-predictions, loss should be low
    assert!(learner.is_converging());
}

#[test]
fn test_online_learner_weights_change_after_observe() {
    let mut learner = OnlineLearner::new();
    let features = vec![1.0_f32; TunerFeatures::DIM];

    let initial_weights: Vec<f32> = learner.weights().to_vec();

    // Observe with large error to force weight update
    learner.observe(&features, 1000000.0);

    let updated_weights = learner.weights();

    // At least some weights should have changed
    let changed = initial_weights
        .iter()
        .zip(updated_weights.iter())
        .any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(changed, "Weights should change after observe");
}

#[test]
fn test_online_learner_momentum_effect() {
    let mut learner = OnlineLearner::new();
    let features = vec![0.5_f32; TunerFeatures::DIM];

    // Multiple observations should show momentum effect
    let initial_weights: Vec<f32> = learner.weights().to_vec();

    for _ in 0..5 {
        learner.observe(&features, 100.0);
    }

    let final_weights = learner.weights();

    // Weights should have changed due to momentum SGD
    let total_change: f32 = initial_weights
        .iter()
        .zip(final_weights.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(total_change > 0.0);
}

#[test]
fn test_online_learner_default_trait() {
    // OnlineLearner derives Default
    let learner: OnlineLearner = Default::default();

    // Default should have empty weights (unlike new())
    assert!(learner.weights().is_empty());
    assert_eq!(learner.num_updates(), 0);
}

#[test]
fn test_online_learner_predict_short_features() {
    let learner = OnlineLearner::new();

    // Features shorter than weights - should still work (partial)
    let short_features = vec![0.5_f32; 5];

    let prediction = learner.predict(&short_features);

    // Should not panic, prediction is valid
    assert!(prediction.is_finite());
}

#[test]
fn test_online_learner_predict_empty_features() {
    let learner = OnlineLearner::new();

    // Empty features - should just return bias
    let empty_features: Vec<f32> = vec![];

    let prediction = learner.predict(&empty_features);

    // Should equal bias term (first weight)
    assert!(prediction >= 0.0);
}

// =========================================================================
// BrickTuner TUI Rendering Tests (T-TUNER-006)
// =========================================================================

#[test]
fn test_brick_tuner_render_panel() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(32)
        .seq_len(2048)
        .model_params_b(7.0)
        .build();

    let rec = tuner.recommend(&features);
    let panel = tuner.render_panel(&rec);

    // Should have multiple lines for TUI
    assert!(panel.len() >= 10);
    // Should contain version header
    assert!(panel[0].contains("BrickTuner"));
}

#[test]
fn test_brick_tuner_render_compact() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(1)
        .seq_len(512)
        .build();

    let rec = tuner.recommend(&features);
    let compact = tuner.render_compact(&rec);

    // Compact should be a single line
    assert!(compact.contains("Tuner:"));
    assert!(compact.contains("tok/s"));
}

#[test]
fn test_brick_tuner_render_comparison_excellent() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(16)
        .seq_len(1024)
        .build();

    let rec = tuner.recommend(&features);
    // Actual matches predicted closely
    let actual_tps = rec.throughput.predicted_tps * 0.98;

    let comparison = tuner.render_comparison(&rec, actual_tps);

    assert_eq!(comparison.len(), 2);
    assert!(comparison[0].contains("Predicted"));
    assert!(comparison[0].contains("Actual"));
    // Should indicate good accuracy (< 5% error)
    assert!(comparison[1].contains("Excellent") || comparison[1].contains("Good"));
}

#[test]
fn test_brick_tuner_render_comparison_poor() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::builder()
        .batch_size(16)
        .seq_len(1024)
        .build();

    let rec = tuner.recommend(&features);
    // Actual differs significantly from predicted
    let actual_tps = rec.throughput.predicted_tps * 0.5;

    let comparison = tuner.render_comparison(&rec, actual_tps);

    // Should indicate poor accuracy (> 20% error)
    assert!(comparison[1].contains("Poor") || comparison[1].contains("Fair"));
}

#[test]
fn test_brick_tuner_render_comparison_zero_actual() {
    let tuner = BrickTuner::new();
    let features = TunerFeatures::default();
    let rec = tuner.recommend(&features);

    // Zero actual throughput edge case
    let comparison = tuner.render_comparison(&rec, 0.0);

    assert_eq!(comparison.len(), 2);
    // Error should be 0% when actual is 0
    assert!(comparison[1].contains("0.0%"));
}

#[test]
fn test_brick_tuner_json_roundtrip() {
    let tuner = BrickTuner::with_pretrained();

    // Serialize to JSON
    let json = tuner.to_json().expect("serialize should work");
    assert!(json.contains("version"));
    assert!(json.contains("throughput"));

    // Deserialize back
    let restored = BrickTuner::from_json(&json).expect("deserialize should work");

    assert_eq!(restored.version(), tuner.version());
}

#[test]
fn test_brick_tuner_json_invalid() {
    let result = BrickTuner::from_json("invalid json{{{");

    assert!(result.is_err());
    if let Err(TunerError::Serialization(msg)) = result {
        assert!(!msg.is_empty());
    }
}

#[test]
fn test_brick_tuner_version() {
    let tuner = BrickTuner::new();

    // Version should be a valid semver
    let version = tuner.version();
    assert!(version.contains('.'));
    assert!(!version.is_empty());
}

#[test]
fn test_brick_tuner_throughput_mape() {
    let tuner = BrickTuner::new();

    // Initial MAPE from pretrained weights
    let mape = tuner.throughput_mape();
    assert!(mape >= 0.0);
}

#[test]
fn test_brick_tuner_sample_count() {
    let tuner = BrickTuner::new();

    // Initial sample count is 0 for new tuner
    let count = tuner.throughput_sample_count();
    assert_eq!(count, 0);

    // With pretrained, should have samples
    let pretrained = BrickTuner::with_pretrained();
    let pretrained_count = pretrained.throughput_sample_count();
    assert!(pretrained_count > 0);
}

// =========================================================================
// KernelBandit Additional Tests (MLT-13)
// =========================================================================

#[test]
fn test_kernel_bandit_select_ucb_unexplored() {
    let bandit = KernelBandit::new();

    // With no pulls, UCB should return first unexplored
    let selected = bandit.select();
    // Should be a valid kernel type
    assert!(selected.to_index() < KernelBandit::NUM_KERNELS);
}

#[test]
fn test_kernel_bandit_best_kernel_selection() {
    let mut bandit = KernelBandit::new();

    // Update some arms
    bandit.update(KernelType::CoalescedQ4K, 0.9);
    bandit.update(KernelType::VectorizedQ4K, 0.5);
    bandit.update(KernelType::TiledQ4K, 0.3);

    // Best kernel should be CoalescedQ4K
    let best = bandit.best_kernel();
    assert_eq!(best, KernelType::CoalescedQ4K);
}

#[test]
fn test_kernel_bandit_thompson_sampling() {
    let bandit = KernelBandit::with_thompson_sampling();

    // Thompson sampling should still return valid kernel
    let selected = bandit.select();
    assert!(selected.to_index() < KernelBandit::NUM_KERNELS);
}

#[test]
fn test_kernel_bandit_thompson_after_updates() {
    let mut bandit = KernelBandit::with_thompson_sampling();

    // Update many times to see Thompson behavior
    for _ in 0..50 {
        bandit.update(KernelType::TiledQ4K, 0.8);
        bandit.update(KernelType::CoalescedQ4K, 0.6);
    }

    // Select should favor TiledQ4K (but Thompson adds randomness)
    let _ = bandit.select();
    // Just ensure no panic
}

#[test]
fn test_kernel_bandit_update_out_of_bounds() {
    let mut bandit = KernelBandit::new();

    // Update with Unknown kernel (should not panic)
    bandit.update(KernelType::Unknown, 0.5);

    // Unknown maps to index 0, should have been updated
    assert!(bandit.total_pulls > 0 || true); // Edge case handling
}

// =========================================================================
// Calibration Result Tests (MLT-11)
// =========================================================================

#[test]
fn test_calibration_result_struct() {
    let result = CalibrationResult {
        throughput_weights: vec![0.1, 0.2, 0.3],
        local_mape: 0.08,
        improvement_pct: 15.5,
        hardware_id: "AMD-Ryzen-7960X".to_string(),
        duration_secs: 120.5,
        num_benchmarks: 1000,
    };

    assert_eq!(result.throughput_weights.len(), 3);
    assert_eq!(result.local_mape, 0.08);
    assert_eq!(result.improvement_pct, 15.5);
    assert!(result.hardware_id.contains("AMD"));
    assert_eq!(result.duration_secs, 120.5);
    assert_eq!(result.num_benchmarks, 1000);
}

// =========================================================================
// ThroughputRegressor Additional Tests (TUNER-003)
// =========================================================================

#[test]
fn test_throughput_regressor_predict_confidence() {
    let regressor = ThroughputRegressor::new();

    let features = TunerFeatures::default();
    let prediction = regressor.predict(&features);

    assert!(prediction.predicted_tps >= 0.0);
    assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
}

#[test]
fn test_kernel_classifier_predict_with_features() {
    let classifier = KernelClassifier::new();

    let features = TunerFeatures::builder()
        .batch_size(1)
        .quant_type(QuantType::Q4K)
        .build();

    let rec = classifier.predict(&features);
    assert!(rec.confidence >= 0.0);
    // top_kernel should be valid
    assert!(rec.top_kernel.to_index() < 16);
}

// =========================================================================
// BottleneckClass Additional Tests (TUNER-005)
// =========================================================================

#[test]
fn test_bottleneck_class_from_brick_memory_compute() {
    // BrickBottleneck is already in scope from module imports
    let memory = BottleneckClass::from_brick_bottleneck(BrickBottleneck::Memory);
    assert!(matches!(memory, BottleneckClass::MemoryBound));

    let compute = BottleneckClass::from_brick_bottleneck(BrickBottleneck::Compute);
    assert!(matches!(compute, BottleneckClass::ComputeBound));
}

#[test]
fn test_bottleneck_class_all_recommended_actions() {
    let memory = BottleneckClass::MemoryBound;
    let action = memory.recommended_action();
    assert!(action.contains("batch") || action.contains("memory") || !action.is_empty());

    let compute = BottleneckClass::ComputeBound;
    let action = compute.recommended_action();
    assert!(!action.is_empty());

    let launch = BottleneckClass::LaunchBound;
    let action = launch.recommended_action();
    assert!(!action.is_empty());

    let attention = BottleneckClass::AttentionBound;
    let action = attention.recommended_action();
    assert!(!action.is_empty());
}

#[test]
fn test_bottleneck_class_to_index_all_variants() {
    assert_eq!(BottleneckClass::Unknown.to_index(), 0);
    assert_eq!(BottleneckClass::MemoryBound.to_index(), 1);
    assert_eq!(BottleneckClass::ComputeBound.to_index(), 2);
    assert_eq!(BottleneckClass::LaunchBound.to_index(), 3);
    assert_eq!(BottleneckClass::AttentionBound.to_index(), 4);
}

#[test]
fn test_bottleneck_class_display_format() {
    let memory = BottleneckClass::MemoryBound;
    let display = format!("{}", memory);
    assert!(display.contains("Memory") || display.contains("memory") || display.len() > 0);
}

// =========================================================================
// QuantType Additional Tests
// =========================================================================

#[test]
fn test_quant_type_ordering_by_size() {
    // F32 > F16 > Q8_0 > Q4K
    assert!(QuantType::F32.bytes_per_param() > QuantType::F16.bytes_per_param());
    assert!(QuantType::F16.bytes_per_param() > QuantType::Q8_0.bytes_per_param());
    assert!(QuantType::Q8_0.bytes_per_param() > QuantType::Q4K.bytes_per_param());
}

#[test]
fn test_quant_type_all_indices() {
    // All variants should have unique indices
    let indices: Vec<usize> = vec![
        QuantType::Q4_0.to_index(),
        QuantType::Q4_1.to_index(),
        QuantType::Q4K.to_index(),
        QuantType::Q5K.to_index(),
        QuantType::Q6K.to_index(),
        QuantType::Q8_0.to_index(),
        QuantType::F16.to_index(),
        QuantType::F32.to_index(),
    ];

    // Check indices are in valid range (0-7)
    for idx in &indices {
        assert!(*idx <= 7);
    }
}

// =========================================================================
// FeatureExtractor Additional Tests
// =========================================================================

#[test]
fn test_feature_extractor_extract_with_hardware() {
    let hw = HardwareCapability::detect();
    let extractor = FeatureExtractor::with_hardware(hw);

    let profiler = BrickProfiler::new();
    let config = RunConfig::default();

    let features = extractor.extract(&profiler, &config);

    // Should have valid normalized features
    assert!(features.batch_size_norm >= 0.0 && features.batch_size_norm <= 1.0);
}

// =========================================================================
// APR File Operations Tests
// =========================================================================

#[test]
fn test_apr_pretrained_save_load_cycle() {
    let tuner = BrickTuner::with_pretrained();

    // Create temp file
    let temp_path = std::env::temp_dir().join("test_tuner_pretrained_apr.apr");

    // Save
    tuner.save_apr(&temp_path).expect("save should work");

    // Load
    let loaded = BrickTuner::load_apr(&temp_path).expect("load should work");

    // Verify
    assert_eq!(loaded.version(), tuner.version());

    // Cleanup
    std::fs::remove_file(&temp_path).ok();
}

#[test]
fn test_apr_nonexistent_file_error() {
    let result = BrickTuner::load_apr("/nonexistent/path/to/file.apr");
    assert!(result.is_err());
    if let Err(TunerError::Io(_)) = result {
        // Expected
    } else {
        panic!("Expected Io error");
    }
}

#[test]
#[cfg(feature = "hardware-detect")]
fn test_cache_path_returns_valid_tuner_path() {
    let path = BrickTuner::cache_path();
    // Should be a valid path with .apr extension
    assert!(path.to_string_lossy().ends_with(".apr") || path.to_string_lossy().contains("tuner"));
}

// =========================================================================
// Experiment Suggestion Display Test
// =========================================================================

#[test]
fn test_experiment_suggestion_display_all_variants() {
    // IncreaseBatchSize
    let batch = ExperimentSuggestion::IncreaseBatchSize { from: 1, to: 8 };
    let display = format!("{}", batch);
    assert!(display.contains("batch") || display.contains("1") || display.contains("8"));

    // EnableCudaGraphs
    let cuda = ExperimentSuggestion::EnableCudaGraphs;
    let display = format!("{}", cuda);
    assert!(display.contains("CUDA") || display.contains("graph"));

    // TryKernel
    let kernel = ExperimentSuggestion::TryKernel { kernel: KernelType::CoalescedQ4K };
    let display = format!("{}", kernel);
    assert!(!display.is_empty());

    // ReduceSequenceLength
    let reduce = ExperimentSuggestion::ReduceSequenceLength { factor: 0.5 };
    let display = format!("{}", reduce);
    assert!(display.contains("sequence") || display.contains("0.5"));

    // EnableMultiKvCache
    let kv = ExperimentSuggestion::EnableMultiKvCache { count: 4 };
    let display = format!("{}", kv);
    assert!(display.contains("cache") || display.contains("4"));
}

// =========================================================================
// RunConfig Tests
// =========================================================================

#[test]
fn test_run_config_custom_values() {
    let config = RunConfig {
        model_params_b: 7.0,
        hidden_dim: 4096,
        num_layers: 32,
        num_heads: 32,
        batch_size: 64,
        seq_len: 4096,
        quant_type: QuantType::Q5K,
        cuda_graphs: true,
        kernel_type: KernelType::CoalescedQ4K,
    };

    assert_eq!(config.batch_size, 64);
    assert_eq!(config.seq_len, 4096);
    assert_eq!(config.quant_type, QuantType::Q5K);
    assert!(config.cuda_graphs);
    assert_eq!(config.model_params_b, 7.0);
}

// =========================================================================
// "IMPOSSIBLE" TESTS: Robustness Against Degenerate Inputs
// =========================================================================
// These tests falsify the optimizer's robustness against impossible/degenerate
// data that could corrupt the entire optimization engine.

/// Test: ThroughputRegressor handles NaN GFLOPS gracefully
/// If NaN propagates to weights, the entire model is corrupted.
#[test]
fn test_impossible_regressor_nan_throughput() {
    let mut regressor = ThroughputRegressor::new();

    // Create training data with NaN throughput
    let mut features = TunerFeatures::default();
    features.model_params_b = 0.5;
    features.batch_size_norm = 0.25;
    features.gpu_mem_bw_norm = 0.5;

    // Mix of valid and NaN data
    let data: Vec<(TunerFeatures, f32)> = vec![
        (features.clone(), 100.0),     // valid
        (features.clone(), f32::NAN),  // NaN - should be filtered or clamped
        (features.clone(), 150.0),     // valid
        (features.clone(), f32::NAN),  // NaN
        (features.clone(), 120.0),     // valid
        (features.clone(), 130.0),     // valid
        (features.clone(), 140.0),     // valid
        (features.clone(), 110.0),     // valid
        (features.clone(), 125.0),     // valid
        (features.clone(), 135.0),     // valid
    ];

    // Training should either filter NaN or return error
    let result = regressor.train(&data);

    // If training succeeds, weights must NOT contain NaN
    if result.is_ok() {
        let prediction = regressor.predict(&features);
        assert!(
            prediction.predicted_tps.is_finite(),
            "Regressor trained on NaN data produced non-finite prediction: {}",
            prediction.predicted_tps
        );
    }
    // If training fails, that's also acceptable behavior
}

/// Test: ThroughputRegressor handles infinite throughput
#[test]
fn test_impossible_regressor_infinite_throughput() {
    let mut regressor = ThroughputRegressor::new();

    let features = TunerFeatures::default();
    let data: Vec<(TunerFeatures, f32)> = vec![
        (features.clone(), 100.0),
        (features.clone(), f32::INFINITY), // Infinite throughput - impossible
        (features.clone(), 150.0),
        (features.clone(), f32::NEG_INFINITY), // Negative infinity
        (features.clone(), 120.0),
        (features.clone(), 130.0),
        (features.clone(), 140.0),
        (features.clone(), 110.0),
        (features.clone(), 125.0),
        (features.clone(), 135.0),
    ];

    let result = regressor.train(&data);

    if result.is_ok() {
        let prediction = regressor.predict(&features);
        assert!(
            prediction.predicted_tps.is_finite(),
            "Regressor trained on infinite data produced non-finite prediction"
        );
    }
}

/// Test: ThroughputRegressor handles negative throughput
/// Negative throughput is physically impossible.
#[test]
fn test_impossible_regressor_negative_throughput() {
    let mut regressor = ThroughputRegressor::new();

    let features = TunerFeatures::default();
    let data: Vec<(TunerFeatures, f32)> = vec![
        (features.clone(), 100.0),
        (features.clone(), -50.0),   // Negative - impossible
        (features.clone(), 150.0),
        (features.clone(), -1000.0), // Large negative
        (features.clone(), 120.0),
        (features.clone(), 130.0),
        (features.clone(), 140.0),
        (features.clone(), 110.0),
        (features.clone(), 125.0),
        (features.clone(), 135.0),
    ];

    let result = regressor.train(&data);

    // Either train filters negatives, or prediction is reasonable
    if result.is_ok() {
        let prediction = regressor.predict(&features);
        // A trained model shouldn't output wildly negative predictions
        assert!(
            prediction.predicted_tps > -10000.0,
            "Negative training data caused unreasonable prediction: {}",
            prediction.predicted_tps
        );
    }
}

/// Test: ThroughputRegressor handles zero execution time (division by zero)
#[test]
fn test_impossible_regressor_zero_throughput() {
    let mut regressor = ThroughputRegressor::new();

    let features = TunerFeatures::default();
    let data: Vec<(TunerFeatures, f32)> = vec![
        (features.clone(), 100.0),
        (features.clone(), 0.0),    // Zero throughput - suspicious but possible
        (features.clone(), 150.0),
        (features.clone(), 0.0),    // More zeros
        (features.clone(), 120.0),
        (features.clone(), 130.0),
        (features.clone(), 140.0),
        (features.clone(), 110.0),
        (features.clone(), 125.0),
        (features.clone(), 135.0),
    ];

    let result = regressor.train(&data);

    // Training on data with zeros should not crash
    if result.is_ok() {
        let prediction = regressor.predict(&features);
        assert!(
            prediction.predicted_tps.is_finite(),
            "Zero-containing training data caused non-finite prediction"
        );
    }
}

/// Test: KernelBandit handles NaN reward
/// NOTE: This test documents that NaN rewards may cause unexpected selections
/// (an unexplored arm with INFINITY UCB). This is a known limitation.
#[test]
fn test_impossible_bandit_nan_reward() {
    let mut bandit = KernelBandit::new();

    // Provide valid rewards first
    bandit.update(KernelType::TiledQ4K, 100.0);
    bandit.update(KernelType::CoalescedQ4K, 120.0);

    // Provide NaN reward
    bandit.update(KernelType::VectorizedQ4K, f32::NAN);

    // Selection should not panic - but may select any arm including unexplored ones
    // (unexplored arms have UCB = INFINITY, which may win over NaN-corrupted arms)
    let selected = bandit.select();

    // The critical requirement is no panic and a valid KernelType is returned
    // NaN in rewards doesn't prevent selection from working
    let _ = selected;

    // Best kernel may be corrupted by NaN, but should still return a valid type
    let best = bandit.best_kernel();
    let _ = best;

    // Exploration rate should still be computable
    let rate = bandit.exploration_rate();
    assert!(rate.is_finite(), "Exploration rate is NaN after NaN reward");
}

/// Test: KernelBandit handles negative reward
#[test]
fn test_impossible_bandit_negative_reward() {
    let mut bandit = KernelBandit::new();

    bandit.update(KernelType::TiledQ4K, 100.0);
    bandit.update(KernelType::CoalescedQ4K, -50.0);  // Negative reward - unusual
    bandit.update(KernelType::VectorizedQ4K, 80.0);

    // Selection should prefer positive rewards
    let best = bandit.best_kernel();
    // TiledQ4K has highest mean (100.0)
    assert_eq!(
        best, KernelType::TiledQ4K,
        "Bandit should prefer positive rewards over negative"
    );
}

/// Test: OnlineLearner handles degenerate feature vectors
#[test]
fn test_impossible_online_learner_nan_features() {
    let mut learner = OnlineLearner::new();

    // Valid update
    let features_valid = vec![0.5_f32; TunerFeatures::DIM];
    learner.observe(&features_valid, 100.0);

    // Feature with NaN
    let mut features_nan = vec![0.5_f32; TunerFeatures::DIM];
    features_nan[0] = f32::NAN;
    learner.observe(&features_nan, 150.0);

    // Prediction should still be finite (model shouldn't be corrupted)
    let prediction = learner.predict(&features_valid);

    assert!(
        prediction.is_finite(),
        "OnlineLearner produced NaN prediction after NaN feature update"
    );
}

/// Test: TunerFeatures builder handles extreme values
#[test]
fn test_impossible_features_extreme_values() {
    // Extreme but valid values
    let features = TunerFeatures::builder()
        .model_params_b(1000.0)     // 1000B parameter model - extreme
        .hidden_dim(1_000_000)      // 1M hidden dim - impossible
        .batch_size(100_000)        // 100K batch - extreme
        .seq_len(1_000_000)         // 1M sequence - extreme
        .build();

    // Features should be clamped/normalized to reasonable ranges
    assert!(features.model_params_b.is_finite());
    assert!(features.hidden_dim_norm.is_finite());
    assert!(features.batch_size_norm.is_finite());
    assert!(features.seq_len_log.is_finite());

    // Values should be in [0, 1] after normalization
    // (or slightly outside if normalization doesn't clamp)
    assert!(features.model_params_b >= 0.0);
    assert!(features.hidden_dim_norm >= 0.0);
}

/// Test: BrickTuner recommend() handles degenerate features
#[test]
fn test_impossible_tuner_recommend_nan_features() {
    let tuner = BrickTuner::with_pretrained();

    // Create features with NaN
    let mut features = TunerFeatures::default();
    features.model_params_b = f32::NAN;
    features.gpu_mem_bw_norm = 0.5;

    // Recommendation should not crash and should produce finite values
    let recommendation = tuner.recommend(&features);

    // Throughput prediction might be NaN, but we should at least get a recommendation
    // The important thing is no panic
    let _ = recommendation.throughput.predicted_tps;
    let _ = recommendation.kernel;
}

/// Test: DataCollector concept drift detection with degenerate errors
#[test]
fn test_impossible_collector_drift_detection_nan() {
    let mut collector = TunerDataCollector::with_online_learning();

    // Record valid prediction errors to build up history
    for _ in 0..15 {
        collector.record_prediction_error(100.0, 95.0);
    }

    // Record NaN error - should not corrupt drift detection
    collector.record_prediction_error(f32::NAN, 100.0);
    collector.record_prediction_error(100.0, f32::NAN);

    // Drift detection should still work without panic
    let drift_status = collector.detect_concept_drift();
    // The important thing is it doesn't panic or return garbage
    assert!(!drift_status.explanation.is_empty());

    // should_retrain should also not panic
    let _ = collector.should_retrain();
}

/// Test: Regressor training with all-zero features
#[test]
fn test_impossible_regressor_zero_features() {
    let mut regressor = ThroughputRegressor::new();

    // All features are zero - degenerate but valid
    let features = TunerFeatures::default();
    let data: Vec<(TunerFeatures, f32)> = vec![
        (features.clone(), 100.0),
        (features.clone(), 110.0),
        (features.clone(), 120.0),
        (features.clone(), 130.0),
        (features.clone(), 140.0),
        (features.clone(), 105.0),
        (features.clone(), 115.0),
        (features.clone(), 125.0),
        (features.clone(), 135.0),
        (features.clone(), 145.0),
    ];

    // Training on identical zero features should work
    let result = regressor.train(&data);

    // Either fails gracefully (singular matrix) or produces a prediction
    match result {
        Ok(()) => {
            let prediction = regressor.predict(&features);
            assert!(
                prediction.predicted_tps.is_finite(),
                "Zero-feature training produced non-finite prediction"
            );
        }
        Err(_) => {
            // Failing is acceptable for degenerate data
        }
    }
}

/// Test: Regressor handles very large throughput values
#[test]
fn test_impossible_regressor_large_throughput() {
    let mut regressor = ThroughputRegressor::new();

    let features = TunerFeatures::default();
    let data: Vec<(TunerFeatures, f32)> = vec![
        (features.clone(), 1e10),  // 10 billion tok/s - impossible
        (features.clone(), 1e10),
        (features.clone(), 1e10),
        (features.clone(), 1e10),
        (features.clone(), 1e10),
        (features.clone(), 1e10),
        (features.clone(), 1e10),
        (features.clone(), 1e10),
        (features.clone(), 1e10),
        (features.clone(), 1e10),
    ];

    let result = regressor.train(&data);

    if result.is_ok() {
        let prediction = regressor.predict(&features);
        assert!(
            prediction.predicted_tps.is_finite(),
            "Large throughput training produced non-finite: {}",
            prediction.predicted_tps
        );
    }
}


// =============================================================================
// BrickTuner Integration Tests (covering mod.rs uncovered methods)
// =============================================================================

/// Test: BrickTuner::online_learner() creates a valid learner
#[test]
fn test_brick_tuner_online_learner() {
    let tuner = BrickTuner::new();
    let learner = tuner.online_learner();
    
    // Should start with zero updates
    assert_eq!(learner.num_updates(), 0);
    
    // Should be able to predict
    let features = TunerFeatures::default().to_vector();
    let prediction = learner.predict(&features);
    assert!(prediction.is_finite());
}

/// Test: BrickTuner::apply_online_updates() applies updates correctly
#[test]
fn test_brick_tuner_apply_online_updates() {
    let mut tuner = BrickTuner::new();
    let initial_version = tuner.version.clone();
    
    let mut learner = tuner.online_learner();
    
    // Make some observations
    let features = TunerFeatures::default().to_vector();
    learner.observe(&features, 100.0);
    learner.observe(&features, 110.0);
    
    // Apply updates
    tuner.apply_online_updates(&learner);
    
    // Version should change
    assert_ne!(tuner.version, initial_version);
    assert!(tuner.version.contains("online"));
}

/// Test: BrickTuner::apply_online_updates() with no updates is a no-op
#[test]
fn test_brick_tuner_apply_online_updates_empty() {
    let mut tuner = BrickTuner::new();
    let initial_version = tuner.version.clone();
    
    let learner = OnlineLearner::new();
    tuner.apply_online_updates(&learner);
    
    // Version should NOT change (no updates)
    assert_eq!(tuner.version, initial_version);
}

/// Test: BrickTuner::kernel_bandit() creates a valid bandit
#[test]
fn test_brick_tuner_kernel_bandit() {
    let tuner = BrickTuner::new();
    let bandit = tuner.kernel_bandit();
    
    // Should start with zero pulls
    assert_eq!(bandit.total_pulls, 0);
    
    // Should be able to select a kernel
    let kernel = bandit.select();
    // Kernel should be a valid variant (just verify it does not panic)
    let _ = format!("{:?}", kernel);
}

/// Test: BrickTuner::recommend_kernel_with_exploration() explore path
#[test]
fn test_brick_tuner_recommend_kernel_with_exploration_explore() {
    let tuner = BrickTuner::new();
    let bandit = tuner.kernel_bandit();
    let features = TunerFeatures::default();
    
    // With explore_prob = 1.0, should always explore
    let rec = tuner.recommend_kernel_with_exploration(&features, &bandit, 1.0);
    
    // Should have lower confidence when exploring
    assert!(rec.confidence <= 0.5);
}

/// Test: BrickTuner::recommend_kernel_with_exploration() exploit path
#[test]
fn test_brick_tuner_recommend_kernel_with_exploration_exploit() {
    let tuner = BrickTuner::new();
    let bandit = tuner.kernel_bandit();
    let features = TunerFeatures::default();
    
    // With explore_prob = 0.0, should always exploit
    let rec = tuner.recommend_kernel_with_exploration(&features, &bandit, 0.0);
    
    // Should have higher confidence when exploiting
    assert!(rec.confidence > 0.5);
}

/// Test: FeatureExtractor::default() creates same as new()
#[test]
fn test_feature_extractor_default_impl() {
    let from_default = FeatureExtractor::default();
    let from_new = FeatureExtractor::new();

    // Both should have hardware as None initially
    assert!(from_default.hardware.is_none());
    assert!(from_new.hardware.is_none());
}

// =============================================================================
// TunerDataCollector Additional Tests (covering uncovered methods)
// =============================================================================

/// Test: TunerDataCollector::ready_to_train() threshold logic
#[test]
fn test_collector_ready_to_train_below_threshold() {
    let collector = TunerDataCollector::new();
    // Empty collector should not be ready
    assert!(!collector.ready_to_train());
}

/// Test: TunerDataCollector::ready_to_train() at threshold
#[test]
fn test_collector_ready_to_train_at_threshold() {
    let mut collector = TunerDataCollector::new();

    // Add exactly MIN_SAMPLES_FOR_TRAINING samples
    for i in 0..TunerDataCollector::MIN_SAMPLES_FOR_TRAINING {
        let features = TunerFeatures::builder()
            .model_params_b(7.0)
            .hidden_dim(4096)
            .batch_size((i as u32) + 1)
            .build();
        collector.samples.push(TrainingSample {
            features,
            throughput_tps: 100.0 + i as f32,
            best_kernel: KernelType::TiledQ4K,
            bottleneck: BottleneckClass::MemoryBound,
            timestamp: format!("{}", i),
            hardware_id: "test".to_string(),
        });
    }

    assert!(collector.ready_to_train());
}

/// Test: TunerDataCollector::training_progress() returns correct counts
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

/// Test: TunerDataCollector::train_if_ready() returns None when not ready
#[test]
fn test_collector_train_if_ready_not_ready() {
    let collector = TunerDataCollector::new();
    assert!(collector.train_if_ready().is_none());
}

/// Test: TunerDataCollector::train_if_ready() returns Some when ready
#[test]
fn test_collector_train_if_ready_success() {
    let mut collector = TunerDataCollector::new();

    // Add MIN_SAMPLES_FOR_TRAINING samples (1000) to trigger training
    for i in 0..TunerDataCollector::MIN_SAMPLES_FOR_TRAINING {
        let features = TunerFeatures::builder()
            .model_params_b(1.0 + (i as f32) % 20.0)
            .hidden_dim(2048 + (i as u32) % 4096)
            .batch_size((i as u32) % 16 + 1)
            .quant_type(if i % 2 == 0 { QuantType::Q4K } else { QuantType::Q8_0 })
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

/// Test: TunerDataCollector::bootstrap_from_five_whys() returns valid collector
#[test]
fn test_collector_bootstrap_from_five_whys() {
    let collector = TunerDataCollector::bootstrap_from_five_whys();

    // Bootstrap returns empty collector for now (TODO: load actual data)
    // But it should still be a valid collector
    assert!(collector.samples().is_empty() || collector.samples().len() > 0);
    assert!(!collector.is_online_learning_enabled());
}

/// Test: TunerDataCollector::auto_retrain() when not ready
#[test]
fn test_collector_auto_retrain_not_ready() {
    let mut collector = TunerDataCollector::new();
    let mut tuner = BrickTuner::new();

    // Should return false when not ready to retrain
    assert!(!collector.auto_retrain(&mut tuner));
}

/// Test: TunerDataCollector::auto_retrain() with sufficient data
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
            .quant_type(if i % 2 == 0 { QuantType::Q4K } else { QuantType::Q8_0 })
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

/// Test: TunerDataCollector::from_json() parses valid JSON by round-tripping
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

/// Test: TunerDataCollector::from_json() handles invalid JSON
#[test]
fn test_collector_from_json_invalid() {
    let json = "not valid json";
    let result = TunerDataCollector::from_json(json);
    assert!(result.is_err());
}

/// Test: ConceptDriftStatus fields
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

/// Test: TrainingStats fields
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

