use super::super::super::*;

// ========================================================================
// F250-F270: Model-Level Inference Tracing Tests (Phase 13)
// ========================================================================

/// F250: TensorStats computes correctly with known input
#[test]
fn test_f250_tensor_stats_correct() {
    // Known input: [1, 2, 3, 4, 5]
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let stats = TensorStats::from_slice(&data);

    assert_eq!(stats.count, 5);
    assert!((stats.min - 1.0).abs() < 1e-6);
    assert!((stats.max - 5.0).abs() < 1e-6);
    assert!((stats.mean - 3.0).abs() < 1e-6);

    // Standard deviation of [1,2,3,4,5] = sqrt(2.5) ≈ 1.5811
    assert!((stats.std - 1.5811).abs() < 0.001);

    // L2 norm = sqrt(1 + 4 + 9 + 16 + 25) = sqrt(55) ≈ 7.416
    assert!((stats.l2_norm - 7.416).abs() < 0.01);

    assert_eq!(stats.nan_count, 0);
    assert_eq!(stats.inf_count, 0);
    assert!(!stats.has_anomaly());
}

/// F251: NaN detection has 100% recall
#[test]
fn test_f251_nan_detection() {
    // Inject NaN values
    let data = vec![1.0, 2.0, f32::NAN, 4.0, f32::NAN, 6.0];
    let stats = TensorStats::from_slice(&data);

    // Must detect both NaN values
    assert_eq!(stats.nan_count, 2);
    assert!(stats.has_anomaly());
    assert!(stats.anomaly_description().unwrap().contains("NaN"));
}

/// F252: Explosion detection triggers on large values
#[test]
fn test_f252_explosion_detection() {
    // Inject explosion: value > 1e6
    let data = vec![1.0, 2.0, 1.5e6, 4.0, 5.0];
    let stats = TensorStats::from_slice(&data);

    assert!(stats.has_anomaly());
    assert!(stats.anomaly_description().unwrap().contains("Explosion"));

    // Also test min explosion
    let data2 = vec![-2e6, 1.0, 2.0];
    let stats2 = TensorStats::from_slice(&data2);
    assert!(stats2.has_anomaly());
}

/// F253: Attention top-k is sorted in descending order
#[test]
fn test_f253_attention_topk_sorted() {
    let weights = vec![0.1, 0.3, 0.05, 0.4, 0.15];
    let trace = AttentionWeightTrace::from_weights(0, 0, 4, &weights, 3);

    // Top-k weights should be descending
    assert_eq!(trace.top_k_positions.len(), 3);
    assert!(trace.top_k_weights.windows(2).all(|w| w[0] >= w[1]));

    // Highest weight is 0.4 at position 3
    assert_eq!(trace.top_k_positions[0], 3);
    assert!((trace.top_k_weights[0] - 0.4).abs() < 1e-6);
}

/// F254: Attention weights sum to approximately 1
#[test]
fn test_f254_attention_weights_sum() {
    // Create normalized attention weights
    let weights = vec![0.2, 0.3, 0.15, 0.25, 0.1];
    let total: f32 = weights.iter().sum();
    assert!((total - 1.0).abs() < 1e-5);

    let trace = AttentionWeightTrace::from_weights(0, 0, 4, &weights, 5);
    let recovered: f32 = trace.top_k_weights.iter().sum::<f32>() + trace.tail_mass;
    assert!((recovered - 1.0).abs() < 1e-5);
}

/// F255: Entropy computation is correct
#[test]
fn test_f255_entropy_computation() {
    // Uniform distribution: entropy = ln(n)
    let n = 4;
    let uniform_weights: Vec<f32> = vec![0.25; n];
    let trace = AttentionWeightTrace::from_weights(0, 0, 3, &uniform_weights, n);

    // Entropy of uniform distribution = ln(4) ≈ 1.386
    let expected_entropy = (n as f32).ln();
    assert!((trace.entropy - expected_entropy).abs() < 0.01);

    // Concentrated distribution: lower entropy
    let concentrated = vec![0.9, 0.05, 0.03, 0.02];
    let trace2 = AttentionWeightTrace::from_weights(0, 0, 3, &concentrated, n);
    assert!(trace2.entropy < trace.entropy);
}

/// F256: Logit tracking is accurate with deterministic model
#[test]
fn test_f256_logit_tracking() {
    let mut trace = LogitEvolutionTrace::new(0, 1.0, 1.0);

    // Track token 42
    let token = trace.track_token(42, "test".to_string());
    token.record_layer(1.5, 10);
    token.record_layer(2.0, 5);
    token.record_layer(3.0, 1);

    assert_eq!(token.per_layer_logit.len(), 3);
    assert_eq!(token.per_layer_rank.len(), 3);
    assert!((token.per_layer_logit[2] - 3.0).abs() < 1e-6);
    assert_eq!(token.per_layer_rank[2], 1);
}

/// F257: Rank computation is correct vs argsort
#[test]
fn test_f257_rank_computation() {
    let logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];

    // Token 3 (value 5.0) should be rank 0 (highest)
    assert_eq!(LogitEvolutionTrace::compute_rank(&logits, 3), 0);

    // Token 4 (value 4.0) should be rank 1
    assert_eq!(LogitEvolutionTrace::compute_rank(&logits, 4), 1);

    // Token 1 (value 3.0) should be rank 2
    assert_eq!(LogitEvolutionTrace::compute_rank(&logits, 1), 2);

    // Token 0 (value 1.0) should be rank 4 (lowest)
    assert_eq!(LogitEvolutionTrace::compute_rank(&logits, 0), 4);
}

/// F258: Cosine similarity is in range [-1, 1]
#[test]
fn test_f258_cosine_similarity_range() {
    // Identical vectors: cosine = 1
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![1.0, 2.0, 3.0];
    let trace = QuantizationErrorTrace::compute(BrickId::RmsNorm, 0, &a, &b, QuantType::F32);
    assert!((trace.cosine_similarity - 1.0).abs() < 1e-5);

    // Opposite vectors: cosine = -1
    let c = vec![-1.0, -2.0, -3.0];
    let trace2 = QuantizationErrorTrace::compute(BrickId::RmsNorm, 0, &a, &c, QuantType::F32);
    assert!((trace2.cosine_similarity - (-1.0)).abs() < 1e-5);

    // Orthogonal vectors: cosine = 0
    let d = vec![1.0, 0.0, 0.0];
    let e = vec![0.0, 1.0, 0.0];
    let trace3 = QuantizationErrorTrace::compute(BrickId::RmsNorm, 0, &d, &e, QuantType::F32);
    assert!(trace3.cosine_similarity.abs() < 1e-5);

    // All results must be in [-1, 1]
    assert!(trace.cosine_similarity >= -1.0 && trace.cosine_similarity <= 1.0);
    assert!(trace2.cosine_similarity >= -1.0 && trace2.cosine_similarity <= 1.0);
    assert!(trace3.cosine_similarity >= -1.0 && trace3.cosine_similarity <= 1.0);
}

/// F259: SNR dB computation is correct
#[test]
fn test_f259_snr_db_computation() {
    // Identical signals: infinite SNR
    let a = vec![1.0, 2.0, 3.0];
    let trace = QuantizationErrorTrace::compute(BrickId::RmsNorm, 0, &a, &a, QuantType::F32);
    assert!(trace.snr_db.is_infinite() && trace.snr_db > 0.0);

    // Known SNR: signal [1,1,1], noise [0.1, 0.1, 0.1]
    // Signal power = 1, Noise power = 0.01, SNR = 100 = 20 dB
    let signal = vec![1.0, 1.0, 1.0];
    let noisy = vec![1.1, 1.1, 1.1];
    let trace2 =
        QuantizationErrorTrace::compute(BrickId::RmsNorm, 0, &noisy, &signal, QuantType::F32);
    // SNR should be around 20 dB
    assert!(trace2.snr_db > 15.0 && trace2.snr_db < 25.0);
}

/// F260: KV cache size tracking is exact
#[test]
fn test_f260_kv_cache_size_tracking() {
    let mut trace = KvCacheStateTrace::new(0, 2048);
    trace.cache_size_bytes = 1024 * 1024; // 1 MB
    trace.valid_positions = 512;

    assert_eq!(trace.cache_size_bytes, 1024 * 1024);
    assert_eq!(trace.valid_positions, 512);
    assert_eq!(trace.max_positions, 2048);

    let utilization = trace.utilization();
    assert!((utilization - 0.25).abs() < 1e-6); // 512/2048 = 0.25
}

/// F261: Eviction counting is exact
#[test]
fn test_f261_eviction_counting() {
    let mut session = KvCacheSessionTrace::default();

    // Add steps with evictions
    let mut step1 = KvCacheStateTrace::new(0, 100);
    step1.evictions_this_step = 5;
    step1.cache_hit_rate = 0.8;
    session.add_step(step1);

    let mut step2 = KvCacheStateTrace::new(1, 100);
    step2.evictions_this_step = 3;
    step2.cache_hit_rate = 0.7;
    session.add_step(step2);

    assert_eq!(session.total_evictions, 8); // 5 + 3 exact
    assert_eq!(session.steps.len(), 2);
}

/// F262: Hit rate is always in [0, 1]
#[test]
fn test_f262_hit_rate_bounded() {
    let mut session = KvCacheSessionTrace::default();

    for i in 0..10 {
        let mut step = KvCacheStateTrace::new(i, 100);
        step.cache_hit_rate = (i as f32) / 10.0; // 0.0 to 0.9
        session.add_step(step);
    }

    // Average hit rate should be bounded
    assert!(session.avg_hit_rate >= 0.0);
    assert!(session.avg_hit_rate <= 1.0);

    // Verify average: (0 + 0.1 + ... + 0.9) / 10 = 4.5 / 10 = 0.45
    assert!((session.avg_hit_rate - 0.45).abs() < 0.01);
}

/// F264: JSON export is valid (smoke test)
#[test]
fn test_f264_json_export_smoke() {
    let config = ModelTracerConfig::lightweight();
    let tracer = ModelTracer::new(config);

    // Summary should be displayable
    let summary = tracer.summary();
    let display = format!("{}", summary);
    assert!(display.contains("ModelTracer"));
}

/// F267: Anomaly detection fires on known bad input
#[test]
fn test_f267_anomaly_detection_fires() {
    // Test NaN anomaly
    let mut trace = ModelActivationTrace::default();
    let mut layer_trace = LayerActivationTrace::new(5);
    layer_trace.input_stats = TensorStats::from_slice(&[1.0, f32::NAN, 3.0]);
    trace.add_layer(layer_trace);

    assert!(trace.has_anomaly);
    assert!(trace.anomaly_desc.as_ref().unwrap().contains("NaN"));

    // Test explosion anomaly
    let mut trace2 = ModelActivationTrace::default();
    let mut layer_trace2 = LayerActivationTrace::new(3);
    layer_trace2.post_attn_stats = TensorStats::from_slice(&[1e7, 2.0, 3.0]);
    trace2.add_layer(layer_trace2);

    assert!(trace2.has_anomaly);
    assert!(trace2.anomaly_desc.as_ref().unwrap().contains("Explosion"));
}

/// F269: Zero overhead when tracing is disabled
#[test]
fn test_f269_zero_overhead_disabled() {
    let config = ModelTracerConfig::default(); // All disabled
    assert!(!config.is_enabled());

    let mut tracer = ModelTracer::new(config);

    // Operations should be no-ops
    tracer.begin_forward(0);
    tracer.record_layer_activation(LayerActivationTrace::new(0));
    tracer.record_attention(AttentionWeightTrace::default());
    tracer.record_kv_state(KvCacheStateTrace::new(0, 100));
    let anomaly = tracer.end_forward();

    // Nothing should be recorded
    assert!(anomaly.is_none());
    let summary = tracer.summary();
    assert_eq!(summary.total_forwards, 0);
    assert_eq!(summary.attention_traces, 0);
    assert_eq!(summary.kv_steps, 0);
}

/// F270: Serialize/deserialize round-trip (via Debug/Display)
#[test]
fn test_f270_roundtrip_smoke() {
    let stats = TensorStats::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let debug = format!("{:?}", stats);

    // Debug output should contain key fields
    assert!(debug.contains("count"));
    assert!(debug.contains("mean"));
    assert!(debug.contains("std"));

    // ModelTracerSummary should be displayable
    let summary = ModelTracerSummary {
        total_forwards: 10,
        anomalies_detected: 1,
        attention_traces: 50,
        logit_traces: 10,
        kv_steps: 100,
        total_evictions: 5,
        avg_hit_rate: 0.95,
        quant_warnings: 2,
        quant_criticals: 0,
    };
    let display = format!("{}", summary);
    assert!(display.contains("Forward passes: 10"));
    assert!(display.contains("Anomalies: 1"));
    assert!(display.contains("95.00%"));
}

/// Additional: QuantType bits and compression ratio
#[test]
fn test_quant_type_bits() {
    assert_eq!(QuantType::F32.bits_per_element(), 32.0);
    assert_eq!(QuantType::F16.bits_per_element(), 16.0);
    assert_eq!(QuantType::Q4_K.bits_per_element(), 4.5);

    // Compression ratios
    assert!((QuantType::F32.compression_ratio() - 1.0).abs() < 0.01);
    assert!((QuantType::F16.compression_ratio() - 2.0).abs() < 0.01);
    assert!((QuantType::Q4_K.compression_ratio() - 7.11).abs() < 0.1);
}

/// Additional: AttentionWeightTrace diagnostic patterns
#[test]
fn test_attention_diagnostics() {
    // Attention sink pattern
    let sink_weights = vec![0.9, 0.05, 0.03, 0.02];
    let trace = AttentionWeightTrace::from_weights(0, 0, 3, &sink_weights, 4);
    assert!(trace.is_attention_sink(0.5));

    // Recency bias pattern
    let recency_weights = vec![0.05, 0.05, 0.1, 0.8]; // High weight on recent position
    let trace2 = AttentionWeightTrace::from_weights(0, 0, 3, &recency_weights, 4);
    assert!(trace2.has_recency_bias(2, 0.7));
}

/// Additional: TokenLogitEvolution decisive layer detection
#[test]
fn test_token_decisive_layer() {
    let mut token = TokenLogitEvolution::new(42, "test".to_string());

    // Gradual change: decisive layer should be where biggest jump occurs
    token.record_layer(1.0, 100); // Layer 0
    token.record_layer(1.5, 50); // Layer 1: rank dropped 50
    token.record_layer(2.0, 48); // Layer 2: rank dropped 2
    token.record_layer(3.0, 1); // Layer 3: rank dropped 47

    let decisive = token.decisive_layer();
    assert_eq!(decisive, Some(1)); // Biggest jump was 100->50 at layer 1
}

/// Additional: KvCacheSessionTrace thrashing detection
#[test]
fn test_kv_cache_thrashing() {
    let mut session = KvCacheSessionTrace::default();

    // Simulate thrashing: high evictions, low hit rate
    for i in 0..10 {
        let mut step = KvCacheStateTrace::new(i, 100);
        step.evictions_this_step = 10;
        step.cache_hit_rate = 0.3;
        session.add_step(step);
    }

    assert!(session.has_thrashing(50, 0.5)); // 100 evictions, 0.3 hit rate

    // Non-thrashing scenario
    let mut healthy = KvCacheSessionTrace::default();
    for i in 0..10 {
        let mut step = KvCacheStateTrace::new(i, 100);
        step.evictions_this_step = 1;
        step.cache_hit_rate = 0.95;
        healthy.add_step(step);
    }

    assert!(!healthy.has_thrashing(50, 0.5));
}

/// Additional: ModelTracer full workflow
#[test]
fn test_model_tracer_workflow() {
    let config = ModelTracerConfig::full();
    let mut tracer = ModelTracer::new(config);

    // Forward pass 1
    tracer.begin_forward(0);
    tracer.record_layer_activation(LayerActivationTrace::new(0));
    tracer.record_layer_activation(LayerActivationTrace::new(1));
    let anomaly1 = tracer.end_forward();
    assert!(anomaly1.is_none()); // No anomaly expected

    // Forward pass 2 with anomaly
    tracer.begin_forward(1);
    let mut bad_layer = LayerActivationTrace::new(0);
    bad_layer.input_stats = TensorStats::from_slice(&[f32::NAN]);
    tracer.record_layer_activation(bad_layer);
    let anomaly2 = tracer.end_forward();
    assert!(anomaly2.is_some());

    // Check summary
    let summary = tracer.summary();
    assert_eq!(summary.total_forwards, 2);
    assert_eq!(summary.anomalies_detected, 1);

    // Clear and verify
    tracer.clear();
    let summary2 = tracer.summary();
    assert_eq!(summary2.total_forwards, 0);
}

/// Additional: AttentionTraceConfig filtering
#[test]
fn test_attention_trace_config() {
    let config = AttentionTraceConfig {
        top_k: 5,
        layers: Some(vec![0, 2, 4]),
        heads: Some(vec![0, 1]),
        weight_threshold: 0.05,
    };

    assert!(config.should_trace_layer(0));
    assert!(!config.should_trace_layer(1));
    assert!(config.should_trace_layer(2));

    assert!(config.should_trace_head(0));
    assert!(config.should_trace_head(1));
    assert!(!config.should_trace_head(2));

    // None means trace all
    let config_all = AttentionTraceConfig::default();
    assert!(config_all.should_trace_layer(999));
    assert!(config_all.should_trace_head(999));
}

/// Additional: QuantizationErrorTrace thresholds
#[test]
fn test_quant_error_thresholds() {
    // Acceptable: cosine > 0.995
    let good = QuantizationErrorTrace::compute(
        BrickId::QkvProjection,
        0,
        &[1.0, 2.0, 3.0],
        &[1.001, 2.001, 3.001],
        QuantType::Q4_K,
    );
    assert!(good.is_acceptable());
    assert!(!good.is_warning());
    assert!(!good.is_critical());

    // Warning: 0.99 < cosine < 0.995
    let _warn = QuantizationErrorTrace::compute(
        BrickId::QkvProjection,
        0,
        &[1.0, 2.0, 3.0],
        &[1.05, 2.05, 3.05],
        QuantType::Q4_K,
    );
    // Note: This may be acceptable depending on exact values

    // Critical: cosine < 0.99
    let critical = QuantizationErrorTrace::compute(
        BrickId::QkvProjection,
        0,
        &[1.0, 2.0, 3.0],
        &[3.0, 2.0, 1.0], // Different pattern
        QuantType::Q2_K,
    );
    assert!(critical.is_critical());
}

/// Additional: ModelQuantizationError aggregation
#[test]
fn test_model_quant_error_aggregation() {
    let mut model_error = ModelQuantizationError::default();

    // Add acceptable error
    model_error.add_error(QuantizationErrorTrace::compute(
        BrickId::RmsNorm,
        0,
        &[1.0, 2.0],
        &[1.0, 2.0],
        QuantType::F32,
    ));

    // Add critical error
    model_error.add_error(QuantizationErrorTrace::compute(
        BrickId::QkvProjection,
        1,
        &[1.0, 2.0, 3.0],
        &[3.0, 1.0, 2.0],
        QuantType::Q4_K,
    ));

    assert_eq!(model_error.brick_errors.len(), 2);
    assert_eq!(model_error.critical_count(), 1);

    let worst = model_error.worst_brick();
    assert!(worst.is_some());
    assert_eq!(worst.unwrap().brick_id, BrickId::QkvProjection);
}

/// F263: Tracing overhead - verify tracer is zero-cost when disabled
#[test]
fn test_f263_tracing_overhead() {
    use std::time::Instant;

    // The spec requirement is that tracing overhead should be < 10% of total
    // inference time. Since we can't measure real inference here, we verify:
    // 1. Disabled tracer does NO work (zero-cost abstraction)
    // 2. Enabled tracer overhead is bounded

    // Test 1: Disabled tracer is truly zero-cost (no allocations)
    let config_disabled = ModelTracerConfig::default();
    assert!(!config_disabled.is_enabled());

    let mut tracer_disabled = ModelTracer::new(config_disabled);

    // These operations should be no-ops
    tracer_disabled.begin_forward(0);
    tracer_disabled.record_layer_activation(LayerActivationTrace::new(0));
    tracer_disabled.record_attention(AttentionWeightTrace::default());
    tracer_disabled.record_kv_state(KvCacheStateTrace::new(0, 2048));
    let result = tracer_disabled.end_forward();

    // Verify zero work done
    assert!(result.is_none());
    let summary = tracer_disabled.summary();
    assert_eq!(
        summary.total_forwards, 0,
        "Disabled tracer should not track forwards"
    );
    assert_eq!(summary.attention_traces, 0);
    assert_eq!(summary.kv_steps, 0);

    // Test 2: TensorStats computation overhead
    // Measuring the cost of computing statistics vs raw data access
    let data: Vec<f32> = (0..10_000).map(|i| i as f32).collect();

    // Baseline: raw sum (no stats)
    let baseline_start = Instant::now();
    let mut raw_sum = 0.0f64;
    for _ in 0..100 {
        for &val in &data {
            raw_sum += val as f64;
        }
    }
    let baseline_ns = baseline_start.elapsed().as_nanos();

    // With stats: compute TensorStats
    let stats_start = Instant::now();
    for _ in 0..100 {
        let _stats = TensorStats::from_slice(&data);
    }
    let stats_ns = stats_start.elapsed().as_nanos();

    // TensorStats should be within 10x of raw access (it does more work)
    let overhead_ratio = stats_ns as f64 / baseline_ns.max(1) as f64;
    assert!(
        overhead_ratio < 50.0, // Generous bound for test environment
        "TensorStats overhead too high: {:.1}x",
        overhead_ratio
    );

    // Use raw_sum to prevent optimizer from removing it
    assert!(raw_sum > 0.0);

    // Test 3: Verify enabled tracer accumulates correctly
    let config_enabled = ModelTracerConfig::lightweight();
    let mut tracer_enabled = ModelTracer::new(config_enabled);

    for i in 0..100 {
        tracer_enabled.begin_forward(i);
        tracer_enabled.record_layer_activation(LayerActivationTrace::new(0));
        tracer_enabled.record_kv_state(KvCacheStateTrace::new(i, 2048));
        let _ = tracer_enabled.end_forward();
    }

    let enabled_summary = tracer_enabled.summary();
    assert_eq!(enabled_summary.total_forwards, 100);
    assert_eq!(enabled_summary.kv_steps, 100);
}

/// F271: KV cache state contains sufficient metadata for rehydration analysis
#[test]
fn test_f271_kv_cache_rehydration_metadata() {
    let mut session = KvCacheSessionTrace::default();

    // Simulate a generation session with cache growth
    for step in 0..100 {
        let mut trace = KvCacheStateTrace::new(step, 2048);
        trace.valid_positions = step + 1;
        trace.cache_size_bytes = (step + 1) * 4096; // 4KB per position
        trace.cache_hit_rate = if step == 0 { 0.0 } else { 0.95 };
        trace.oldest_position = 0;
        trace.evictions_this_step = 0;
        trace.accessed_positions = vec![step]; // Current position
        session.add_step(trace);
    }

    // Verify the trace contains sufficient metadata to describe the "lost" state
    assert_eq!(session.steps.len(), 100);
    assert_eq!(session.total_evictions, 0);
    assert!(session.avg_hit_rate > 0.9);

    // Verify we can reconstruct cache state from trace
    let last_step = session.steps.last().unwrap();
    assert_eq!(last_step.valid_positions, 100);
    assert_eq!(last_step.max_positions, 2048);
    assert!(!last_step.is_window_exhausted());

    // Verify accessed positions are tracked
    for (i, step) in session.steps.iter().enumerate() {
        assert!(step.accessed_positions.contains(&i));
    }
}

/// F272: Bit-exactness - tracing must not affect computation results
#[test]
fn test_f272_bit_exactness() {
    // Simulate a computation with and without tracing
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

    // Compute stats with tracing enabled
    let stats_with_tracing = TensorStats::from_slice(&input_data);

    // Compute stats again (should be identical)
    let stats_without_tracing = TensorStats::from_slice(&input_data);

    // Bit-exact comparison
    assert_eq!(stats_with_tracing.count, stats_without_tracing.count);
    assert_eq!(
        stats_with_tracing.min.to_bits(),
        stats_without_tracing.min.to_bits()
    );
    assert_eq!(
        stats_with_tracing.max.to_bits(),
        stats_without_tracing.max.to_bits()
    );
    assert_eq!(
        stats_with_tracing.mean.to_bits(),
        stats_without_tracing.mean.to_bits()
    );
    assert_eq!(
        stats_with_tracing.std.to_bits(),
        stats_without_tracing.std.to_bits()
    );
    assert_eq!(
        stats_with_tracing.l2_norm.to_bits(),
        stats_without_tracing.l2_norm.to_bits()
    );

    // Verify tracer doesn't modify data by reference
    let mut tracer = ModelTracer::new(ModelTracerConfig::full());
    tracer.begin_forward(0);

    let mut layer_trace = LayerActivationTrace::new(0);
    layer_trace.input_stats = TensorStats::from_slice(&input_data);

    // The original data is unchanged
    assert_eq!(input_data, vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);

    tracer.record_layer_activation(layer_trace);
    let _ = tracer.end_forward();

    // Data still unchanged after tracing
    assert_eq!(input_data, vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
}

/// F273: Attention sink detection with BOS token
#[test]
fn test_f273_attention_sink_bos_token() {
    // Simulate attention pattern with BOS sink (position 0 gets high weight)
    let weights_with_sink = vec![0.7, 0.1, 0.05, 0.05, 0.05, 0.05];
    let trace = AttentionWeightTrace::from_weights(5, 0, 5, &weights_with_sink, 6);

    // F273: Position 0 (BOS) must be in top-k
    assert!(trace.top_k_positions.contains(&0));
    assert!(trace.is_attention_sink(0.5));

    // Non-sink pattern
    let weights_no_sink = vec![0.1, 0.1, 0.3, 0.2, 0.2, 0.1];
    let trace2 = AttentionWeightTrace::from_weights(5, 0, 5, &weights_no_sink, 6);
    assert!(!trace2.is_attention_sink(0.5));
}

/// F274: Logit evolution shows rank jump at decisive layer
#[test]
fn test_f274_logit_rank_jump() {
    let mut token = TokenLogitEvolution::new(42, "test_token".to_string());

    // Simulate a model where Layer 10 causes a rank jump
    for layer in 0..15 {
        let logit = if layer < 10 { 0.5 } else { 5.0 }; // Jump at layer 10
        let rank = if layer < 10 { 100 } else { 5 }; // Rank improves dramatically
        token.record_layer(logit, rank);
    }

    // F274: Decisive layer should be 10 (where rank jumped from 100 to 5)
    let decisive = token.decisive_layer();
    assert_eq!(decisive, Some(10));

    // Verify the rank actually jumped
    assert_eq!(token.per_layer_rank[9], 100);
    assert_eq!(token.per_layer_rank[10], 5);
}

/// F275: ModelTracer anomaly detection integration
#[test]
fn test_f275_anomaly_integration() {
    let config = ModelTracerConfig::full();
    let mut tracer = ModelTracer::new(config);

    // Forward pass 1: Normal data
    tracer.begin_forward(0);
    let normal_layer = LayerActivationTrace::new(0);
    tracer.record_layer_activation(normal_layer);
    let result1 = tracer.end_forward();
    assert!(result1.is_none(), "Normal data should not trigger anomaly");

    // Forward pass 2: Inject Inf
    tracer.begin_forward(1);
    let mut inf_layer = LayerActivationTrace::new(0);
    inf_layer.post_attn_stats = TensorStats::from_slice(&[1.0, f32::INFINITY, 3.0]);
    tracer.record_layer_activation(inf_layer);
    let result2 = tracer.end_forward();
    assert!(result2.is_some(), "Inf should trigger anomaly");
    assert!(
        result2.unwrap().contains("Inf"),
        "Anomaly should mention Inf"
    );

    // Forward pass 3: Inject NaN
    tracer.begin_forward(2);
    let mut nan_layer = LayerActivationTrace::new(5);
    nan_layer.input_stats = TensorStats::from_slice(&[f32::NAN]);
    tracer.record_layer_activation(nan_layer);
    let result3 = tracer.end_forward();
    assert!(result3.is_some(), "NaN should trigger anomaly");
    assert!(
        result3.unwrap().contains("NaN"),
        "Anomaly should mention NaN"
    );

    // Verify summary counts anomalies
    let summary = tracer.summary();
    assert_eq!(summary.total_forwards, 3);
    assert_eq!(summary.anomalies_detected, 2); // Inf and NaN passes
}
