//! Tests for the brick profiler module.

use super::*;

// ========================================================================
// TileStats Tests
// ========================================================================

#[test]
fn test_tile_stats_new() {
    let stats = TileStats::new(TileLevel::Macro);
    assert_eq!(stats.level, TileLevel::Macro);
    assert_eq!(stats.count, 0);
    assert_eq!(stats.total_ns, 0);
    assert_eq!(stats.min_ns, u64::MAX);
    assert_eq!(stats.max_ns, 0);
}

#[test]
fn test_tile_stats_add_sample() {
    let mut stats = TileStats::new(TileLevel::Midi);
    stats.add_sample(1000, 100, 200);
    assert_eq!(stats.count, 1);
    assert_eq!(stats.total_ns, 1000);
    assert_eq!(stats.min_ns, 1000);
    assert_eq!(stats.max_ns, 1000);
    assert_eq!(stats.total_elements, 100);
    assert_eq!(stats.total_flops, 200);

    stats.add_sample(2000, 150, 300);
    assert_eq!(stats.count, 2);
    assert_eq!(stats.total_ns, 3000);
    assert_eq!(stats.min_ns, 1000);
    assert_eq!(stats.max_ns, 2000);
}

#[test]
fn test_tile_stats_avg_us() {
    let mut stats = TileStats::new(TileLevel::Micro);
    assert_eq!(stats.avg_us(), 0.0);

    stats.add_sample(2000_000, 100, 200); // 2000 µs
    assert!((stats.avg_us() - 2000.0).abs() < 0.01);

    stats.add_sample(4000_000, 100, 200); // 4000 µs
    assert!((stats.avg_us() - 3000.0).abs() < 0.01); // (2000 + 4000) / 2
}

#[test]
fn test_tile_stats_throughput() {
    let mut stats = TileStats::new(TileLevel::Macro);
    assert_eq!(stats.throughput(), 0.0);

    // 1 billion ns = 1 second, 1000 elements = 1000 elem/s
    stats.add_sample(1_000_000_000, 1000, 0);
    assert!((stats.throughput() - 1000.0).abs() < 1.0);
}

#[test]
fn test_tile_stats_gflops() {
    let mut stats = TileStats::new(TileLevel::Macro);
    assert_eq!(stats.gflops(), 0.0);

    // 1 second, 1 billion FLOPs = 1 GFLOP/s
    stats.add_sample(1_000_000_000, 100, 1_000_000_000);
    assert!((stats.gflops() - 1.0).abs() < 0.01);
}

#[test]
fn test_tile_stats_arithmetic_intensity() {
    let mut stats = TileStats::new(TileLevel::Midi);
    assert_eq!(stats.arithmetic_intensity(), 0.0);

    // 1000 elements * 4 bytes = 4000 bytes, 4000 flops = 1.0 AI
    stats.add_sample(1000, 1000, 4000);
    assert!((stats.arithmetic_intensity() - 1.0).abs() < 0.01);
}

#[test]
fn test_tile_level_name() {
    assert_eq!(TileLevel::Macro.name(), "macro");
    assert_eq!(TileLevel::Midi.name(), "midi");
    assert_eq!(TileLevel::Micro.name(), "micro");
}

// ========================================================================
// BrickProfiler Tests
// ========================================================================

#[test]
fn test_brick_profiler_new() {
    let profiler = BrickProfiler::new();
    assert!(!profiler.is_enabled());
    assert_eq!(profiler.total_tokens(), 0);
    assert_eq!(profiler.total_ns(), 0);
}

#[test]
fn test_brick_profiler_enabled() {
    let profiler = BrickProfiler::enabled();
    assert!(profiler.is_enabled());
}

#[test]
fn test_brick_profiler_enable_disable() {
    let mut profiler = BrickProfiler::new();
    assert!(!profiler.is_enabled());

    profiler.enable();
    assert!(profiler.is_enabled());

    profiler.disable();
    assert!(!profiler.is_enabled());
}

#[test]
fn test_brick_profiler_sync_mode() {
    let mut profiler = BrickProfiler::new();
    assert_eq!(profiler.sync_mode(), SyncMode::Deferred);

    profiler.set_sync_mode(SyncMode::Immediate);
    assert_eq!(profiler.sync_mode(), SyncMode::Immediate);
}

#[test]
fn test_brick_profiler_start_stop() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    let timer = profiler.start("TestBrick");
    std::thread::sleep(std::time::Duration::from_micros(100));
    profiler.stop(timer, 10);

    // Should have recorded something
    assert!(profiler.total_tokens() >= 10);
    assert!(profiler.total_ns() > 0);
}

#[test]
fn test_brick_profiler_start_stop_disabled() {
    let mut profiler = BrickProfiler::new();
    // profiler is disabled by default

    let timer = profiler.start("TestBrick");
    profiler.stop(timer, 10);

    // Should NOT have recorded anything
    assert_eq!(profiler.total_tokens(), 0);
    assert_eq!(profiler.total_ns(), 0);
}

#[test]
fn test_brick_profiler_brick_id_api() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    let timer = profiler.start_brick(BrickId::RmsNorm);
    std::thread::sleep(std::time::Duration::from_micros(50));
    profiler.stop_brick(timer, 5);

    let stats = profiler.brick_stats(BrickId::RmsNorm);
    assert_eq!(stats.count, 1);
    assert!(stats.total_ns > 0);
}

#[test]
fn test_brick_profiler_deferred_api() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    let start_ns = profiler.elapsed_ns();
    std::thread::sleep(std::time::Duration::from_micros(100));
    profiler.record_deferred(BrickId::QkvProjection, start_ns, 100);

    assert!(profiler.has_pending());
    assert_eq!(profiler.pending_count(), 1);

    let end_ns = profiler.elapsed_ns();
    profiler.finalize(end_ns);

    assert!(!profiler.has_pending());
    assert_eq!(profiler.pending_count(), 0);

    let stats = profiler.brick_stats(BrickId::QkvProjection);
    assert_eq!(stats.count, 1);
}

#[test]
fn test_brick_profiler_reset() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    let timer = profiler.start_brick(BrickId::AttentionSoftmax);
    profiler.stop_brick(timer, 10);

    assert!(profiler.total_tokens() > 0);

    profiler.reset();

    assert_eq!(profiler.total_tokens(), 0);
    assert_eq!(profiler.total_ns(), 0);
}

#[test]
fn test_brick_profiler_tile_profiling() {
    let mut profiler = BrickProfiler::new();
    assert!(!profiler.is_tile_profiling_enabled());

    profiler.enable_tile_profiling();
    assert!(profiler.is_tile_profiling_enabled());

    let timer = profiler.start_tile(TileLevel::Macro, 0, 0);
    std::thread::sleep(std::time::Duration::from_micros(50));
    profiler.stop_tile(timer, 1024, 2048);

    let stats = profiler.tile_stats(TileLevel::Macro);
    assert_eq!(stats.count, 1);
    assert!(stats.total_ns > 0);
}

#[test]
fn test_brick_profiler_tile_profiling_disabled() {
    let mut profiler = BrickProfiler::new();
    // Tile profiling disabled by default

    let timer = profiler.start_tile(TileLevel::Midi, 1, 1);
    profiler.stop_tile(timer, 512, 1024);

    let stats = profiler.tile_stats(TileLevel::Midi);
    assert_eq!(stats.count, 0);
}

// ========================================================================
// Checksum Tests (CORRECTNESS-011)
// ========================================================================

#[test]
fn test_fnv1a_checksum_empty() {
    let data: [f32; 0] = [];
    let checksum = fnv1a_f32_checksum(&data);
    // Should return offset basis for empty input
    assert_eq!(checksum, 0xcbf29ce484222325);
}

#[test]
fn test_fnv1a_checksum_deterministic() {
    let data = [1.0f32, 2.0, 3.0, 4.0];
    let c1 = fnv1a_f32_checksum(&data);
    let c2 = fnv1a_f32_checksum(&data);
    assert_eq!(c1, c2);
}

#[test]
fn test_fnv1a_checksum_different_inputs() {
    let data1 = [1.0f32, 2.0, 3.0];
    let data2 = [1.0f32, 2.0, 4.0];
    let c1 = fnv1a_f32_checksum(&data1);
    let c2 = fnv1a_f32_checksum(&data2);
    assert_ne!(c1, c2);
}

#[test]
fn test_fnv1a_checksum_truncates_at_64() {
    let data_short: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let data_long: Vec<f32> = (0..100).map(|i| i as f32).collect();

    let c1 = fnv1a_f32_checksum(&data_short);
    let c2 = fnv1a_f32_checksum(&data_long);
    // Both should hash only first 64 elements
    assert_eq!(c1, c2);
}

#[test]
fn test_brick_profiler_divergence_detection() {
    let mut cpu_profiler = BrickProfiler::new();
    let mut gpu_profiler = BrickProfiler::new();
    cpu_profiler.enable();
    gpu_profiler.enable();

    // Record same checksum on both
    let data = [1.0f32, 2.0, 3.0, 4.0];
    cpu_profiler.record_checksum("TestKernel", 0, 0, &data);
    gpu_profiler.record_checksum("TestKernel", 0, 0, &data);

    // No divergence
    assert!(gpu_profiler.find_divergence(&cpu_profiler).is_none());

    // Now record different checksum
    let different_data = [1.0f32, 2.0, 3.0, 5.0]; // Changed last element
    gpu_profiler.reset_checksums();
    gpu_profiler.record_checksum("TestKernel", 0, 0, &different_data);

    // Should find divergence
    let div = gpu_profiler.find_divergence(&cpu_profiler);
    assert!(div.is_some());
    let div = div.unwrap();
    assert_eq!(div.kernel_name, "TestKernel");
    assert_eq!(div.layer_idx, 0);
    assert_eq!(div.position, 0);
}

// ========================================================================
// Falsification Tests
// ========================================================================

/// FALSIFICATION TEST: TileStats min/max monotonicity
///
/// After any sequence of add_sample calls, min_ns <= max_ns must hold.
#[test]
fn test_falsify_tile_stats_min_max_monotonicity() {
    let mut stats = TileStats::new(TileLevel::Macro);

    // Add samples with varying elapsed times
    for ns in [1000, 500, 2000, 100, 5000, 50] {
        stats.add_sample(ns, 10, 20);
        assert!(
            stats.min_ns <= stats.max_ns,
            "FALSIFICATION FAILED: min {} > max {} after adding {}",
            stats.min_ns,
            stats.max_ns,
            ns
        );
    }

    assert_eq!(stats.min_ns, 50);
    assert_eq!(stats.max_ns, 5000);
}

/// FALSIFICATION TEST: BrickProfiler total_tokens accumulation
///
/// total_tokens must equal sum of all elements passed to stop/stop_brick.
#[test]
fn test_falsify_total_tokens_accumulation() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    let mut expected_total = 0u64;
    let element_counts = [10, 20, 30, 15, 25];

    for &count in &element_counts {
        let timer = profiler.start_brick(BrickId::RmsNorm);
        profiler.stop_brick(timer, count);
        expected_total += count;
    }

    assert_eq!(
        profiler.total_tokens(),
        expected_total,
        "FALSIFICATION FAILED: total_tokens {} != expected {}",
        profiler.total_tokens(),
        expected_total
    );
}

/// FALSIFICATION TEST: Checksum collision resistance
///
/// Different float patterns should produce different checksums.
#[test]
fn test_falsify_checksum_collision_resistance() {
    // Generate various patterns that might collide in weak hashes
    let patterns: Vec<Vec<f32>> = vec![
        vec![0.0; 10],
        vec![1.0; 10],
        vec![-1.0; 10],
        (0..10).map(|i| i as f32).collect(),
        (0..10).map(|i| -(i as f32)).collect(),
        vec![f32::MIN_POSITIVE; 10],
        vec![f32::MAX; 10],
    ];

    let checksums: Vec<u64> = patterns.iter().map(|p| fnv1a_f32_checksum(p)).collect();

    // Check all pairs for uniqueness
    for i in 0..checksums.len() {
        for j in (i + 1)..checksums.len() {
            assert_ne!(
                checksums[i], checksums[j],
                "FALSIFICATION FAILED: patterns {} and {} collide with checksum {:016X}",
                i, j, checksums[i]
            );
        }
    }
}

// ========================================================================
// Additional Coverage Tests
// ========================================================================

#[test]
fn test_tile_stats_cache_efficiency() {
    let mut stats = TileStats::new(TileLevel::Macro);

    // Zero peak_gflops should return 0.0
    assert_eq!(stats.cache_efficiency(0.0), 0.0);
    assert_eq!(stats.cache_efficiency(-1.0), 0.0);

    // 1 second, 1 billion FLOPs = 1 GFLOP/s
    stats.add_sample(1_000_000_000, 100, 1_000_000_000);
    let efficiency = stats.cache_efficiency(10.0); // 10 GFLOP/s peak
    assert!((efficiency - 0.1).abs() < 0.01);

    // Efficiency capped at 1.0
    let capped = stats.cache_efficiency(0.5); // Lower peak than actual
    assert!((capped - 1.0).abs() < 0.001);
}

#[test]
fn test_brick_profiler_record_elapsed() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    let duration = std::time::Duration::from_micros(1000);
    profiler.record_elapsed("MyBrick", duration, 100);

    assert!(profiler.total_tokens() >= 100);
    assert!(profiler.total_ns() > 0);
}

#[test]
fn test_brick_profiler_record_elapsed_disabled() {
    let mut profiler = BrickProfiler::new();
    // profiler is disabled

    let duration = std::time::Duration::from_micros(1000);
    profiler.record_elapsed("MyBrick", duration, 100);

    // Should not record when disabled
    assert_eq!(profiler.total_tokens(), 0);
}

#[test]
fn test_brick_profiler_record_elapsed_with_bytes() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    let duration = std::time::Duration::from_micros(500);
    profiler.record_elapsed_with_bytes("ByteBrick", duration, 200, 4096, 2048);

    assert!(profiler.total_tokens() >= 200);
}

#[test]
fn test_brick_profiler_set_brick_bottleneck() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Record something first
    let timer = profiler.start("TestBrick");
    profiler.stop(timer, 10);

    profiler.set_brick_bottleneck("TestBrick", BrickBottleneck::Memory);
    // Should not panic
}

#[test]
fn test_brick_profiler_category_stats() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Record normalization brick
    let timer = profiler.start_brick(BrickId::RmsNorm);
    std::thread::sleep(std::time::Duration::from_micros(10));
    profiler.stop_brick(timer, 100);

    let category_stats = profiler.category_stats();
    // Should have accumulated in Norm category
    let norm_stats = &category_stats[BrickCategory::Norm as usize];
    assert!(norm_stats.count >= 1);
}

#[test]
fn test_brick_profiler_graph_operations() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Graph disabled by default
    assert!(!profiler.is_graph_enabled());

    profiler.enable_graph();
    assert!(profiler.is_graph_enabled());

    // Push a scope using Layer variant
    let node = ExecutionNode::Layer { index: 0 };
    let scope_id = profiler.graph_push_scope(node);
    assert!(scope_id.is_some());

    // Record a brick
    profiler.graph_record_brick(BrickId::RmsNorm, 1000, 100);

    // Pop the scope
    let popped_id = profiler.graph_pop_scope();
    assert!(popped_id.is_some());

    assert!(profiler.graph_is_scope_balanced());

    profiler.disable_graph();
    assert!(!profiler.is_graph_enabled());
}

#[test]
fn test_brick_profiler_graph_to_dot() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    profiler.enable_graph();

    let node = ExecutionNode::Layer { index: 1 };
    profiler.graph_push_scope(node);
    profiler.graph_record_brick(BrickId::RmsNorm, 500, 50);
    profiler.graph_pop_scope();

    let dot = profiler.graph_to_dot();
    assert!(dot.contains("digraph"));
}

#[test]
fn test_brick_profiler_graph_clear() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    profiler.enable_graph();

    let node = ExecutionNode::Layer { index: 2 };
    profiler.graph_push_scope(node);
    profiler.graph_pop_scope();

    profiler.graph_clear();
    // Should be balanced after clear
    assert!(profiler.graph_is_scope_balanced());
}

#[test]
fn test_brick_profiler_l2_cache_hit_rate() {
    let mut profiler = BrickProfiler::new();

    // Default is None
    assert!(profiler.l2_cache_hit_rate().is_none());

    profiler.set_l2_cache_hit_rate(0.95);
    assert_eq!(profiler.l2_cache_hit_rate(), Some(0.95));
}

#[test]
fn test_brick_profiler_zero_copy() {
    let mut profiler = BrickProfiler::new();

    // Default is false
    assert!(!profiler.is_zero_copy());

    profiler.set_zero_copy(true);
    assert!(profiler.is_zero_copy());

    profiler.set_zero_copy(false);
    assert!(!profiler.is_zero_copy());
}

#[test]
fn test_brick_profiler_reset_epoch() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Let some time pass
    std::thread::sleep(std::time::Duration::from_micros(100));
    let ns1 = profiler.elapsed_ns();
    assert!(ns1 > 0);

    profiler.reset_epoch();
    let ns2 = profiler.elapsed_ns();
    // After reset, elapsed should be close to zero
    assert!(ns2 < ns1);
}

#[test]
fn test_brick_profiler_brick_stats_mut() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    // Get mutable stats and modify
    let stats = profiler.brick_stats_mut(BrickId::RmsNorm);
    stats.count = 42;

    // Verify modification persisted
    let stats = profiler.brick_stats(BrickId::RmsNorm);
    assert_eq!(stats.count, 42);
}

#[test]
fn test_brick_profiler_record_deferred_dynamic() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();

    let start_ns = profiler.elapsed_ns();
    std::thread::sleep(std::time::Duration::from_micros(50));
    profiler.record_deferred_dynamic("DynamicBrick", start_ns, 75);

    assert!(profiler.has_pending());
    assert_eq!(profiler.pending_count(), 1);

    let end_ns = profiler.elapsed_ns();
    profiler.finalize(end_ns);

    assert!(!profiler.has_pending());
}

#[test]
fn test_tile_stats_default() {
    let stats = TileStats::default();
    assert_eq!(stats.level, TileLevel::Macro); // Default is Macro
    assert_eq!(stats.count, 0);
}

#[test]
fn test_tile_level_default() {
    let level = TileLevel::default();
    assert_eq!(level, TileLevel::Macro);
}

#[test]
fn test_brick_profiler_execution_graph_accessors() {
    let mut profiler = BrickProfiler::new();

    // Read-only access
    let _graph = profiler.execution_graph();

    // Mutable access
    let _graph_mut = profiler.execution_graph_mut();
}

#[test]
fn test_brick_profiler_graph_record_kernel() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    profiler.enable_graph();

    let node = ExecutionNode::Layer { index: 3 };
    profiler.graph_push_scope(node);
    // graph_record_kernel(name, ptx_hash, grid, block, shared_mem)
    profiler.graph_record_kernel("my_kernel", 0x12345678, (1, 1, 1), (256, 1, 1), 4096);
    profiler.graph_pop_scope();

    // Should be able to convert to DOT
    let dot = profiler.graph_to_dot();
    assert!(!dot.is_empty());
}

#[test]
fn test_brick_profiler_graph_disabled_operations() {
    let mut profiler = BrickProfiler::new();
    profiler.enable();
    // Graph is NOT enabled

    // Operations should be no-ops when graph disabled
    let node = ExecutionNode::Layer { index: 4 };
    let scope_id = profiler.graph_push_scope(node);
    assert!(scope_id.is_none()); // Returns None when disabled

    let popped = profiler.graph_pop_scope();
    assert!(popped.is_none());
}
