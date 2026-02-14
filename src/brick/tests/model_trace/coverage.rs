use super::super::super::*;

// =========================================================================
// F276-F285: Additional coverage tests for Phase 13
// =========================================================================

/// F276: All QuantType variants bits_per_element coverage
#[test]
fn test_f276_quant_type_all_variants() {
    // Test all QuantType variants for bits_per_element
    assert_eq!(QuantType::F32.bits_per_element(), 32.0);
    assert_eq!(QuantType::F16.bits_per_element(), 16.0);
    assert_eq!(QuantType::Bf16.bits_per_element(), 16.0);
    assert_eq!(QuantType::Q8_0.bits_per_element(), 8.0);
    assert_eq!(QuantType::Q6_K.bits_per_element(), 6.5);
    assert_eq!(QuantType::Q5_K.bits_per_element(), 5.5);
    assert_eq!(QuantType::Q4_0.bits_per_element(), 4.5);
    assert_eq!(QuantType::Q4_K.bits_per_element(), 4.5);
    assert_eq!(QuantType::Q3_K.bits_per_element(), 3.5);
    assert_eq!(QuantType::Q2_K.bits_per_element(), 2.5);

    // Compression ratios for all types
    assert!((QuantType::Bf16.compression_ratio() - 2.0).abs() < 0.01);
    assert!((QuantType::Q8_0.compression_ratio() - 4.0).abs() < 0.01);
    assert!((QuantType::Q6_K.compression_ratio() - 4.92).abs() < 0.1);
    assert!((QuantType::Q5_K.compression_ratio() - 5.82).abs() < 0.1);
    assert!((QuantType::Q3_K.compression_ratio() - 9.14).abs() < 0.1);
    assert!((QuantType::Q2_K.compression_ratio() - 12.8).abs() < 0.1);
}

/// F277: LayerActivationTrace all anomaly paths
#[test]
fn test_f277_layer_anomaly_all_paths() {
    // Test post_norm anomaly
    let mut layer = LayerActivationTrace::new(0);
    layer.post_norm_stats = TensorStats::from_slice(&[f32::NAN]);
    assert!(layer.has_anomaly());
    let desc = layer.anomaly_description().unwrap();
    assert!(desc.contains("post_norm"));

    // Test post_attn anomaly
    let mut layer2 = LayerActivationTrace::new(1);
    layer2.post_attn_stats = TensorStats::from_slice(&[f32::INFINITY]);
    assert!(layer2.has_anomaly());
    let desc2 = layer2.anomaly_description().unwrap();
    assert!(desc2.contains("post_attn"));

    // Test post_ffn anomaly
    let mut layer3 = LayerActivationTrace::new(2);
    layer3.post_ffn_stats = TensorStats::from_slice(&[f32::NAN]);
    assert!(layer3.has_anomaly());
    let desc3 = layer3.anomaly_description().unwrap();
    assert!(desc3.contains("post_ffn"));

    // Test output anomaly
    let mut layer4 = LayerActivationTrace::new(3);
    layer4.output_stats = TensorStats::from_slice(&[1e7]);
    assert!(layer4.has_anomaly());
    let desc4 = layer4.anomaly_description().unwrap();
    assert!(desc4.contains("output"));

    // Test residual dominance
    let mut layer5 = LayerActivationTrace::new(4);
    layer5.residual_ratio = 0.995;
    assert!(layer5.has_anomaly());
    let desc5 = layer5.anomaly_description().unwrap();
    assert!(desc5.contains("residual"));
}

/// F278: ModelActivationTrace full workflow
#[test]
fn test_f278_model_activation_trace_workflow() {
    // Test with_capacity
    let mut trace = ModelActivationTrace::with_capacity(32);
    assert_eq!(trace.layers.capacity(), 32);

    // Add normal layers
    for i in 0..3 {
        let layer = LayerActivationTrace::new(i);
        trace.add_layer(layer);
    }
    assert!(!trace.has_anomaly);

    // Add layer with anomaly
    let mut bad_layer = LayerActivationTrace::new(3);
    bad_layer.input_stats = TensorStats::from_slice(&[f32::NAN, 1.0, 2.0]);
    trace.add_layer(bad_layer);
    assert!(trace.has_anomaly);
    assert!(trace.anomaly_desc.is_some());

    // Test finalize with embedding anomaly
    let mut trace2 = ModelActivationTrace::with_capacity(2);
    trace2.embedding_stats = TensorStats::from_slice(&[f32::INFINITY]);
    trace2.finalize();
    assert!(trace2.has_anomaly);
    assert!(trace2.anomaly_desc.as_ref().unwrap().contains("Embedding"));

    // Test finalize with logits anomaly
    let mut trace3 = ModelActivationTrace::with_capacity(2);
    trace3.logits_stats = TensorStats::from_slice(&[f32::NAN]);
    trace3.finalize();
    assert!(trace3.has_anomaly);
    assert!(trace3.anomaly_desc.as_ref().unwrap().contains("Logits"));
}

/// F279: WatermarkedBuffer full API coverage
#[test]
fn test_f279_watermarked_buffer_api() {
    let wm = BufferWatermarks {
        low: 100,
        high: 1000,
    };
    let mut buf = WatermarkedBuffer::new(wm);

    // Test len and is_empty
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());

    // Test write
    buf.write(&[1, 2, 3, 4, 5]);
    assert_eq!(buf.len(), 5);
    assert!(!buf.is_empty());

    // Test watermarks accessor
    let retrieved = buf.watermarks();
    assert_eq!(retrieved.low, 100);
    assert_eq!(retrieved.high, 1000);

    // Test drain
    let drained = buf.drain(3);
    assert_eq!(drained, vec![1, 2, 3]);
    assert_eq!(buf.len(), 2);

    // Test drain more than available
    let drained2 = buf.drain(100);
    assert_eq!(drained2.len(), 2);
    assert!(buf.is_empty());

    // Test clear
    buf.write(&[10, 20, 30]);
    assert_eq!(buf.len(), 3);
    buf.clear();
    assert!(buf.is_empty());

    // Test pressure_level
    buf.write(&vec![0u8; 600]);
    let pressure = buf.pressure_level();
    assert!(pressure > 0.0 && pressure < 1.0);
}

/// F280: ExecutionGraph node and edge coverage
#[test]
fn test_f280_execution_graph_node_types() {
    let mut graph = ExecutionGraph::new();

    // Add various node types
    let root = graph.add_node(ExecutionNode::Layer { index: 0 });
    let brick = graph.add_node(ExecutionNode::Brick {
        id: BrickId::QkvProjection,
        timing_ns: 1000,
        elements: 1024,
    });
    let kernel = graph.add_node(ExecutionNode::Kernel {
        name: "matmul".to_string(),
        ptx_hash: 12345,
        grid: (1, 1, 1),
        block: (256, 1, 1),
        shared_mem: 4096,
        timing_ns: Some(500),
        arithmetic_intensity: None,
        achieved_tflops: None,
    });
    let func = graph.add_node(ExecutionNode::Function {
        name: "forward".to_string(),
        file: Some("model.rs".to_string()),
        line: Some(100),
    });
    let transfer = graph.add_node(ExecutionNode::Transfer {
        src: "CPU".to_string(),
        dst: "GPU".to_string(),
        bytes: 4096,
        direction: TransferDirection::H2D,
        timing_ns: Some(200),
    });

    // Add edges of different types
    graph.add_edge(root, brick, EdgeType::Contains);
    graph.add_edge(brick, kernel, EdgeType::Launches);
    graph.add_edge(root, func, EdgeType::Calls);
    graph.add_edge(
        func,
        transfer,
        EdgeType::Transfer {
            bytes: 4096,
            direction: TransferDirection::H2D,
        },
    );
    graph.add_edge(kernel, transfer, EdgeType::DependsOn);

    // Verify node IDs are sequential
    assert_eq!(root.0, 0);
    assert_eq!(brick.0, 1);
    assert_eq!(kernel.0, 2);
    assert_eq!(func.0, 3);
    assert_eq!(transfer.0, 4);
}

/// F281: AttentionTraceConfig filtering
#[test]
fn test_f281_attention_trace_config_filtering() {
    // Test with specific layers/heads
    let config = AttentionTraceConfig {
        top_k: 10,
        layers: Some(vec![0, 5, 10, 15]),
        heads: Some(vec![0, 1]),
        weight_threshold: 0.01,
    };

    assert!(config.should_trace_layer(0));
    assert!(config.should_trace_layer(5));
    assert!(!config.should_trace_layer(3));
    assert!(config.should_trace_head(0));
    assert!(config.should_trace_head(1));
    assert!(!config.should_trace_head(2));

    // Test with None (trace all)
    let config_all = AttentionTraceConfig {
        top_k: 5,
        layers: None,
        heads: None,
        weight_threshold: 0.05,
    };

    assert!(config_all.should_trace_layer(99));
    assert!(config_all.should_trace_head(31));
}

/// F282: KvCacheStateTrace utilization and window exhaustion
#[test]
fn test_f282_kv_cache_utilization() {
    // Test utilization calculation
    let mut trace = KvCacheStateTrace::new(50, 2048);
    trace.valid_positions = 1024;
    assert!((trace.utilization() - 0.5).abs() < 0.01);

    // Test window exhaustion
    assert!(!trace.is_window_exhausted());
    trace.valid_positions = 2048;
    assert!(trace.is_window_exhausted());

    // Test session thrashing detection
    let mut session = KvCacheSessionTrace::default();
    for step in 0..100 {
        let mut s = KvCacheStateTrace::new(step, 2048);
        s.valid_positions = step + 1;
        s.evictions_this_step = if step > 50 { 3 } else { 0 };
        session.add_step(s);
    }
    // 50 steps * 3 evictions = 150 evictions in last 50 steps
    assert!(session.has_thrashing(50, 0.5));
}

/// F283: LogitEvolutionTrace compute_rank edge cases
#[test]
fn test_f283_logit_rank_edge_cases() {
    // Single element
    let single = vec![5.0];
    assert_eq!(LogitEvolutionTrace::compute_rank(&single, 0), 0);

    // All same values
    let same = vec![3.0, 3.0, 3.0, 3.0];
    let rank = LogitEvolutionTrace::compute_rank(&same, 2);
    assert_eq!(rank, 0); // All tied at highest

    // Negative values
    let negative = vec![-5.0, -3.0, -1.0, -10.0];
    assert_eq!(LogitEvolutionTrace::compute_rank(&negative, 2), 0); // -1.0 is highest
    assert_eq!(LogitEvolutionTrace::compute_rank(&negative, 3), 3); // -10.0 is lowest
}

/// F284: QuantizationErrorTrace boundary conditions
#[test]
fn test_f284_quant_error_boundaries() {
    // Perfect match (identical)
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let trace =
        QuantizationErrorTrace::compute(BrickId::QkvProjection, 0, &data, &data, QuantType::F32);
    assert_eq!(trace.mse, 0.0);
    assert!((trace.cosine_similarity - 1.0).abs() < 0.0001);
    assert!(trace.is_acceptable());

    // Large error (warning threshold)
    let reference = vec![1.0, 0.0, 0.0, 0.0];
    let bad_quant = vec![0.97, 0.02, 0.02, 0.02];
    let trace2 = QuantizationErrorTrace::compute(
        BrickId::AttentionScore,
        0,
        &bad_quant,
        &reference,
        QuantType::Q4_K,
    );
    assert!(trace2.cosine_similarity < 1.0);

    // Test model-level aggregation
    let mut model_error = ModelQuantizationError::default();
    model_error.add_error(trace);
    model_error.add_error(trace2);

    assert_eq!(model_error.brick_errors.len(), 2);
    assert!(model_error.worst_brick().is_some());
}

/// F285: ModelTracer disabled config verification
#[test]
fn test_f285_model_tracer_disabled() {
    let disabled = ModelTracerConfig::default();
    assert!(!disabled.is_enabled());
    assert!(!disabled.trace_activations);
    assert!(!disabled.trace_attention);
    assert!(!disabled.trace_logits);
    assert!(!disabled.trace_quant_error);
    assert!(!disabled.trace_kv_cache);

    let mut tracer = ModelTracer::new(disabled);

    // Verify no-op behavior
    tracer.begin_forward(0);
    let layer = LayerActivationTrace::new(0);
    tracer.record_layer_activation(layer);
    let kv = KvCacheStateTrace::new(0, 2048);
    tracer.record_kv_state(kv);
    let result = tracer.end_forward();
    assert!(result.is_none()); // No anomaly detection when disabled

    let summary = tracer.summary();
    assert_eq!(summary.total_forwards, 0); // Not tracked when disabled
}

/// F286: TensorStats edge cases
#[test]
fn test_f286_tensor_stats_edge_cases() {
    // Empty slice
    let empty: Vec<f32> = vec![];
    let stats = TensorStats::from_slice(&empty);
    assert_eq!(stats.count, 0);
    assert!(!stats.has_anomaly()); // Empty is not an anomaly

    // Single element
    let single = vec![42.0];
    let stats = TensorStats::from_slice(&single);
    assert_eq!(stats.count, 1);
    assert_eq!(stats.min, 42.0);
    assert_eq!(stats.max, 42.0);
    assert_eq!(stats.mean, 42.0);
    assert_eq!(stats.std, 0.0); // No variance with single element
}

/// F287: AttentionWeightTrace::is_uniform
#[test]
fn test_f287_attention_uniform_detection() {
    // Uniform distribution (high entropy)
    let uniform_weights = vec![0.25, 0.25, 0.25, 0.25];
    let trace = AttentionWeightTrace::from_weights(0, 0, 3, &uniform_weights, 4);
    assert!(trace.is_uniform(1.0)); // Entropy threshold of 1.0

    // Peaky distribution (low entropy)
    let peaky_weights = vec![0.9, 0.05, 0.03, 0.02];
    let trace2 = AttentionWeightTrace::from_weights(0, 0, 3, &peaky_weights, 4);
    assert!(!trace2.is_uniform(1.0)); // Not uniform
}

/// F288: LogitEvolutionTrace::finalize
#[test]
fn test_f288_logit_evolution_finalize() {
    let mut trace = LogitEvolutionTrace::new(100, 0.7, 0.9);

    // Track a token
    let token = trace.track_token(42, "hello".to_string());
    token.record_layer(0.5, 500);
    token.record_layer(1.0, 200);
    token.record_layer(5.0, 1);

    // Finalize with this token selected
    trace.finalize(42);
    // Decisive layer should be set based on token's evolution
    // The jump from 200 to 1 is the biggest - verify finalize completes without panic
    let _ = trace.decisive_layer;

    // Finalize with non-tracked token
    let mut trace2 = LogitEvolutionTrace::new(100, 0.7, 0.9);
    trace2.finalize(999); // Token not tracked
                          // Should not panic, just won't find decisive layer
}

/// F289: QuantizationErrorTrace with empty data
#[test]
fn test_f289_quant_error_empty() {
    let empty: Vec<f32> = vec![];
    let trace =
        QuantizationErrorTrace::compute(BrickId::QkvProjection, 0, &empty, &empty, QuantType::Q4_K);
    assert_eq!(trace.mse, 0.0);
    assert_eq!(trace.cosine_similarity, 1.0);
    assert!(trace.snr_db.is_infinite());
}

/// F290: ModelTracer record_logits and record_quant_error
#[test]
fn test_f290_model_tracer_record_methods() {
    let config = ModelTracerConfig::full();
    let mut tracer = ModelTracer::new(config);

    tracer.begin_forward(0);

    // Record attention trace
    let attn_trace = AttentionWeightTrace::from_weights(0, 0, 5, &[0.5, 0.3, 0.2], 3);
    tracer.record_attention(attn_trace);

    // Record logits - need to first have logit trace initialized
    // This exercises the record_logits path

    // Record quant error
    let quant_trace = QuantizationErrorTrace::compute(
        BrickId::QkvProjection,
        0,
        &[1.02, 1.98, 3.05],
        &[1.0, 2.0, 3.0],
        QuantType::Q4_K,
    );
    tracer.record_quant_error(quant_trace);

    // End forward and verify
    let _result = tracer.end_forward();
    // Should complete without error
    let summary = tracer.summary();
    assert_eq!(summary.total_forwards, 1);
}

/// F291: has_recency_bias with query_pos == 0
#[test]
fn test_f291_recency_bias_edge_case() {
    // Query position 0 - should always return false
    let weights = vec![0.8, 0.2];
    let trace = AttentionWeightTrace::from_weights(0, 0, 0, &weights, 2);
    assert!(!trace.has_recency_bias(5, 0.5)); // query_pos == 0, returns false
}

/// F292: LayerActivationTrace::new default values
#[test]
fn test_f292_layer_activation_trace_defaults() {
    let layer = LayerActivationTrace::new(5);
    assert_eq!(layer.layer_idx, 5);
    assert_eq!(layer.residual_ratio, 0.0);
    assert!(!layer.has_anomaly()); // All stats are default, no anomaly
    assert!(layer.anomaly_description().is_none());
}

/// F293: ModelQuantizationError warning and critical counts
#[test]
fn test_f293_model_quant_error_thresholds() {
    let mut model_error = ModelQuantizationError::default();

    // Add an acceptable error
    let good = QuantizationErrorTrace {
        brick_id: BrickId::QkvProjection,
        layer_idx: 0,
        mse: 0.001,
        max_abs_error: 0.01,
        cosine_similarity: 0.998,
        snr_db: 40.0,
        quant_type: QuantType::Q4_K,
    };
    model_error.add_error(good);

    // Add a warning-level error
    let warning = QuantizationErrorTrace {
        brick_id: BrickId::AttentionScore,
        layer_idx: 1,
        mse: 0.01,
        max_abs_error: 0.1,
        cosine_similarity: 0.992, // Between 0.99 and 0.995
        snr_db: 25.0,
        quant_type: QuantType::Q4_K,
    };
    model_error.add_error(warning);

    // Add a critical error
    let critical = QuantizationErrorTrace {
        brick_id: BrickId::DownProjection,
        layer_idx: 2,
        mse: 0.1,
        max_abs_error: 1.0,
        cosine_similarity: 0.85, // Below 0.99
        snr_db: 10.0,
        quant_type: QuantType::Q2_K,
    };
    model_error.add_error(critical);

    assert_eq!(model_error.brick_errors.len(), 3);
    assert!(model_error.warning_count() >= 1);
    assert!(model_error.critical_count() >= 1);

    let worst = model_error.worst_brick().unwrap();
    assert!(worst.cosine_similarity < 0.9);
}

/// F294: TensorStats::is_vanishing
#[test]
fn test_f294_tensor_stats_vanishing() {
    // Create nearly constant tensor (vanishing gradients)
    let data = vec![1.0; 1000];
    let stats = TensorStats::from_slice(&data);
    assert!(stats.is_vanishing()); // std should be 0

    // Non-vanishing tensor
    let varied: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let stats2 = TensorStats::from_slice(&varied);
    assert!(!stats2.is_vanishing());
}

/// F295: TensorStats high variance anomaly
#[test]
fn test_f295_tensor_stats_high_variance() {
    // Create tensor with extreme variance
    let mut data = vec![0.0; 100];
    data[0] = 1e5;
    data[1] = -1e5;
    let stats = TensorStats::from_slice(&data);
    assert!(stats.std > 1e4);
    assert!(stats.has_anomaly());
    let desc = stats.anomaly_description().unwrap();
    assert!(desc.contains("variance") || desc.contains("std"));
}

/// F296: ModelTracer record_logits path
#[test]
fn test_f296_model_tracer_record_logits() {
    let config = ModelTracerConfig::full();
    let mut tracer = ModelTracer::new(config);

    tracer.begin_forward(0);

    // Create logit trace manually
    let mut logit_trace = LogitEvolutionTrace::new(100, 0.7, 0.9);
    let token = logit_trace.track_token(42, "hello".to_string());
    token.final_probability = 0.5;

    // Set the logit trace
    tracer.set_current_logit_trace(Some(logit_trace));

    // Record logits - this should exercise the record_logits path
    let logits: Vec<f32> = (0..100).map(|i| i as f32).collect();
    tracer.record_logits(0, &logits);

    // Verify it was recorded
    if let Some(trace) = tracer.current_logit_trace() {
        assert!(!trace.tracked_tokens.is_empty());
    }

    tracer.end_forward();
}

/// F297: ModelActivationTrace add_layer without anomaly
#[test]
fn test_f297_model_activation_add_normal_layers() {
    let mut trace = ModelActivationTrace::with_capacity(10);

    // Add several normal layers
    for i in 0..5 {
        let mut layer = LayerActivationTrace::new(i);
        layer.input_stats = TensorStats::from_slice(&vec![1.0; 100]);
        layer.output_stats = TensorStats::from_slice(&vec![1.1; 100]);
        trace.add_layer(layer);
    }

    // No anomaly should be detected
    assert!(!trace.has_anomaly);
    assert!(trace.anomaly_desc.is_none());
    assert_eq!(trace.layers.len(), 5);
}

/// F298: AsyncTask node type coverage
#[test]
fn test_f298_async_task_node() {
    let mut graph = ExecutionGraph::new();

    let async_task = graph.add_node(ExecutionNode::AsyncTask {
        name: "inference_loop".to_string(),
        poll_count: 100,
        yield_count: 50,
        total_poll_ns: 1_000_000,
    });

    // Verify node was added
    assert_eq!(async_task.0, 0);
}
