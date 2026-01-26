//! Falsification Tests for PMAT-025: Cache Efficiency Analysis
//!
//! F1121-F1130: Cache analysis falsification tests
//!
//! These tests verify the cache efficiency analysis module for:
//! - Cache level classification
//! - Working set estimation
//! - Tiling recommendations
//! - Bandwidth prediction

use cbtop::{
    elementwise_working_set, matrix_working_set, optimal_matmul_tile, AccessPattern,
    BandwidthPrediction, CacheConfig, CacheLevel, WorkingSetAnalysis,
};

// =============================================================================
// F1121: Cache Level Classification Tests
// =============================================================================

/// F1121.1: L1 cache classification for small working sets
#[test]
fn f1121_l1_classification() {
    let config = CacheConfig::default();

    // 1KB should fit in L1 (32KB default)
    assert_eq!(config.classify(1024), CacheLevel::L1);

    // 20KB should fit in L1 (< 32KB * 0.75 = 24KB)
    assert_eq!(config.classify(20 * 1024), CacheLevel::L1);
}

/// F1121.2: L2 cache classification
#[test]
fn f1121_l2_classification() {
    let config = CacheConfig::default();

    // 100KB should fit in L2 (512KB default)
    assert_eq!(config.classify(100 * 1024), CacheLevel::L2);

    // 300KB should fit in L2
    assert_eq!(config.classify(300 * 1024), CacheLevel::L2);
}

/// F1121.3: L3 cache classification
#[test]
fn f1121_l3_classification() {
    let config = CacheConfig::default();

    // 10MB should fit in L3 (32MB default)
    assert_eq!(config.classify(10 * 1024 * 1024), CacheLevel::L3);

    // 20MB should fit in L3
    assert_eq!(config.classify(20 * 1024 * 1024), CacheLevel::L3);
}

/// F1121.4: RAM classification for large working sets
#[test]
fn f1121_ram_classification() {
    let config = CacheConfig::default();

    // 100MB should overflow to RAM
    assert_eq!(config.classify(100 * 1024 * 1024), CacheLevel::Ram);

    // 1GB should be in RAM
    assert_eq!(config.classify(1024 * 1024 * 1024), CacheLevel::Ram);
}

/// F1121.5: Cache level names
#[test]
fn f1121_cache_level_names() {
    assert_eq!(CacheLevel::L1.name(), "L1 cache");
    assert_eq!(CacheLevel::L2.name(), "L2 cache");
    assert_eq!(CacheLevel::L3.name(), "L3 cache");
    assert_eq!(CacheLevel::Ram.name(), "main memory");
}

/// F1121.6: Cache level latency ordering
#[test]
fn f1121_latency_ordering() {
    let l1_lat = CacheLevel::L1.typical_latency_cycles();
    let l2_lat = CacheLevel::L2.typical_latency_cycles();
    let l3_lat = CacheLevel::L3.typical_latency_cycles();
    let ram_lat = CacheLevel::Ram.typical_latency_cycles();

    assert!(l1_lat < l2_lat);
    assert!(l2_lat < l3_lat);
    assert!(l3_lat < ram_lat);
}

/// F1121.7: Relative bandwidth ordering
#[test]
fn f1121_bandwidth_ordering() {
    let l1_bw = CacheLevel::L1.relative_bandwidth();
    let l2_bw = CacheLevel::L2.relative_bandwidth();
    let l3_bw = CacheLevel::L3.relative_bandwidth();
    let ram_bw = CacheLevel::Ram.relative_bandwidth();

    assert!(l1_bw >= l2_bw);
    assert!(l2_bw >= l3_bw);
    assert!(l3_bw >= ram_bw);
    assert_eq!(l1_bw, 1.0); // L1 is baseline
}

// =============================================================================
// F1122: Cache Config Presets Tests
// =============================================================================

/// F1122.1: Default config has reasonable values
#[test]
fn f1122_default_config() {
    let config = CacheConfig::default();

    assert_eq!(config.l1_size, 32 * 1024);
    assert_eq!(config.l2_size, 512 * 1024);
    assert_eq!(config.l3_size, 32 * 1024 * 1024);
    assert_eq!(config.line_size, 64);
}

/// F1122.2: Zen4 preset
#[test]
fn f1122_zen4_config() {
    let config = CacheConfig::zen4();

    assert_eq!(config.l1_size, 32 * 1024);
    assert_eq!(config.l2_size, 1024 * 1024);
    assert_eq!(config.l3_size, 32 * 1024 * 1024);
    assert_eq!(config.l3_sharing, 8);
}

/// F1122.3: Sapphire Rapids preset
#[test]
fn f1122_sapphire_rapids_config() {
    let config = CacheConfig::sapphire_rapids();

    assert_eq!(config.l1_size, 48 * 1024);
    assert_eq!(config.l2_size, 2 * 1024 * 1024);
    assert_eq!(config.l3_size, 60 * 1024 * 1024);
    assert_eq!(config.l3_sharing, 16);
}

/// F1122.4: Apple M2 preset (no L3)
#[test]
fn f1122_apple_m2_config() {
    let config = CacheConfig::apple_m2();

    assert_eq!(config.l1_size, 128 * 1024);
    assert_eq!(config.l2_size, 16 * 1024 * 1024);
    assert_eq!(config.l3_size, 0); // No L3
    assert_eq!(config.line_size, 128); // Larger cache lines
}

/// F1122.5: Apple M2 classification skips L3
#[test]
fn f1122_m2_skips_l3() {
    let config = CacheConfig::apple_m2();

    // Large working set should go directly to RAM (no L3)
    assert_eq!(config.classify(20 * 1024 * 1024), CacheLevel::Ram);
}

/// F1122.6: L3 per core calculation
#[test]
fn f1122_l3_per_core() {
    let config = CacheConfig::default();

    // 32MB / 8 cores = 4MB per core
    assert_eq!(config.l3_per_core(), 4 * 1024 * 1024);
}

// =============================================================================
// F1123: Working Set Analysis Tests
// =============================================================================

/// F1123.1: Small working set fits in L1
#[test]
fn f1123_small_working_set() {
    let config = CacheConfig::default();
    let analysis = WorkingSetAnalysis::analyze(1000, 4, 1.0, &config);

    // 1000 elements * 4 bytes * 1.0 factor = 4KB
    assert_eq!(analysis.working_set_bytes, 4000);
    assert_eq!(analysis.cache_level, CacheLevel::L1);
    assert!(!analysis.tiling_recommended);
    assert!(analysis.recommended_tile_bytes.is_none());
}

/// F1123.2: Large working set triggers tiling
#[test]
fn f1123_large_working_set_tiling() {
    let config = CacheConfig::default();
    // 1M elements * 4 bytes * 2.0 factor = 8MB > L2
    let analysis = WorkingSetAnalysis::analyze(1_000_000, 4, 2.0, &config);

    assert!(analysis.tiling_recommended);
    assert!(analysis.recommended_tile_bytes.is_some());

    // Recommended tile should be 75% of L2
    let expected_tile = config.l2_size * 3 / 4;
    assert_eq!(analysis.recommended_tile_bytes.unwrap(), expected_tile);
}

/// F1123.3: Access factor multiplier
#[test]
fn f1123_access_factor() {
    let config = CacheConfig::default();

    let analysis1 = WorkingSetAnalysis::analyze(1000, 4, 1.0, &config);
    let analysis2 = WorkingSetAnalysis::analyze(1000, 4, 3.0, &config);

    // 3x access factor should triple working set
    assert_eq!(analysis1.working_set_bytes * 3, analysis2.working_set_bytes);
}

/// F1123.4: Utilization percentage
#[test]
fn f1123_utilization_percent() {
    let config = CacheConfig::default();

    // 16KB / 32KB L1 = 50% utilization
    let analysis = WorkingSetAnalysis::analyze(4096, 4, 1.0, &config);

    assert!(analysis.utilization_percent > 45.0);
    assert!(analysis.utilization_percent < 55.0);
}

/// F1123.5: Expected efficiency from cache level
#[test]
fn f1123_expected_efficiency() {
    let config = CacheConfig::default();

    let l1_analysis = WorkingSetAnalysis::analyze(1000, 4, 1.0, &config);
    let ram_analysis = WorkingSetAnalysis::analyze(100_000_000, 4, 1.0, &config);

    // L1 should have higher efficiency than RAM
    assert!(l1_analysis.expected_efficiency > ram_analysis.expected_efficiency);
}

/// F1123.6: Recommendation message contains size
#[test]
fn f1123_recommendation_message() {
    let config = CacheConfig::default();
    let analysis = WorkingSetAnalysis::analyze(1000, 4, 1.0, &config);

    let rec = analysis.recommendation();
    assert!(rec.contains("4000"));
    assert!(rec.contains("fits"));
}

// =============================================================================
// F1124: Matrix Working Set Tests
// =============================================================================

/// F1124.1: Square matrix working set
#[test]
fn f1124_square_matrix() {
    // C = A × B: 1024×1024 matrices
    let ws = matrix_working_set(1024, 1024, 1024, 4);

    // 3 matrices × 1024² × 4 bytes = 12MB
    assert_eq!(ws, 3 * 1024 * 1024 * 4);
}

/// F1124.2: Rectangular matrix working set
#[test]
fn f1124_rectangular_matrix() {
    // C (100×50) = A (100×200) × B (200×50)
    let ws = matrix_working_set(100, 50, 200, 4);

    // A: 100×200×4 = 80,000 bytes
    // B: 200×50×4 = 40,000 bytes
    // C: 100×50×4 = 20,000 bytes
    // Total: 140,000 bytes
    assert_eq!(ws, 140_000);
}

/// F1124.3: Double precision matrix
#[test]
fn f1124_double_precision() {
    let ws_f32 = matrix_working_set(512, 512, 512, 4);
    let ws_f64 = matrix_working_set(512, 512, 512, 8);

    // f64 should be exactly 2x f32
    assert_eq!(ws_f64, ws_f32 * 2);
}

// =============================================================================
// F1125: Optimal Tile Size Tests
// =============================================================================

/// F1125.1: Tile size is positive
#[test]
fn f1125_tile_positive() {
    let config = CacheConfig::default();
    let tile = optimal_matmul_tile(&config, 4);

    assert!(tile > 0);
}

/// F1125.2: Tile size is cache-aligned
#[test]
fn f1125_tile_aligned() {
    let config = CacheConfig::default();
    let tile = optimal_matmul_tile(&config, 4);

    // Tile should be multiple of elements per cache line
    let elements_per_line = config.line_size / 4;
    assert_eq!(tile % elements_per_line, 0);
}

/// F1125.3: Tile fits in L2
#[test]
fn f1125_tile_fits_l2() {
    let config = CacheConfig::default();
    let tile = optimal_matmul_tile(&config, 4);

    // 3 tiles × tile² × 4 bytes should fit in 75% of L2
    let tile_bytes = 3 * tile * tile * 4;
    let target = config.l2_size * 3 / 4;

    assert!(tile_bytes <= target);
}

/// F1125.4: Different element sizes
#[test]
fn f1125_different_element_sizes() {
    let config = CacheConfig::default();

    let tile_f32 = optimal_matmul_tile(&config, 4);
    let tile_f64 = optimal_matmul_tile(&config, 8);

    // Larger elements should give smaller tile size
    assert!(tile_f32 >= tile_f64);
}

// =============================================================================
// F1126: Elementwise Working Set Tests
// =============================================================================

/// F1126.1: Single input single output
#[test]
fn f1126_single_io() {
    // sqrt: 1 input, 1 output
    let ws = elementwise_working_set(1000, 1, 1, 4);

    // 1000 × (1 + 1) × 4 = 8000 bytes
    assert_eq!(ws, 8000);
}

/// F1126.2: Multiple inputs
#[test]
fn f1126_multiple_inputs() {
    // add: 2 inputs, 1 output
    let ws = elementwise_working_set(1000, 2, 1, 4);

    // 1000 × (2 + 1) × 4 = 12000 bytes
    assert_eq!(ws, 12000);
}

/// F1126.3: Fused operation
#[test]
fn f1126_fused_op() {
    // fma: 3 inputs, 1 output
    let ws = elementwise_working_set(10000, 3, 1, 4);

    // 10000 × (3 + 1) × 4 = 160000 bytes
    assert_eq!(ws, 160000);
}

// =============================================================================
// F1127: Access Pattern Tests
// =============================================================================

/// F1127.1: Single iteration is streaming
#[test]
fn f1127_streaming_pattern() {
    let config = CacheConfig::default();
    let pattern = AccessPattern::estimate(1024, 1, &config);

    assert_eq!(pattern, AccessPattern::Streaming);
}

/// F1127.2: Small with reuse
#[test]
fn f1127_reuse_pattern() {
    let config = CacheConfig::default();
    // Small working set with multiple iterations
    let pattern = AccessPattern::estimate(1024, 10, &config);

    assert_eq!(pattern, AccessPattern::Reuse);
}

/// F1127.3: Large with iterations is random
#[test]
fn f1127_random_pattern() {
    let config = CacheConfig::default();
    // Large working set with multiple iterations
    let pattern = AccessPattern::estimate(100 * 1024 * 1024, 10, &config);

    assert_eq!(pattern, AccessPattern::Random);
}

/// F1127.4: Pattern names
#[test]
fn f1127_pattern_names() {
    assert_eq!(AccessPattern::Streaming.name(), "streaming");
    assert_eq!(AccessPattern::Reuse.name(), "reuse");
    assert_eq!(AccessPattern::Random.name(), "random");
}

/// F1127.5: Efficiency factors
#[test]
fn f1127_efficiency_factors() {
    // Reuse should be best, random worst
    assert!(
        AccessPattern::Reuse.efficiency_factor() > AccessPattern::Streaming.efficiency_factor()
    );
    assert!(
        AccessPattern::Streaming.efficiency_factor() > AccessPattern::Random.efficiency_factor()
    );
}

// =============================================================================
// F1128: Bandwidth Prediction Tests
// =============================================================================

/// F1128.1: L1 reuse near peak
#[test]
fn f1128_l1_reuse_prediction() {
    let config = CacheConfig::default();
    let prediction = BandwidthPrediction::predict(
        100.0, // 100 GB/s peak
        1024,  // 1KB working set (L1)
        AccessPattern::Reuse,
        &config,
    );

    // L1 with reuse should be near 100%
    assert!(prediction.efficiency_percent > 90.0);
    assert!(prediction.predicted_bandwidth_gbps > 90.0);
}

/// F1128.2: RAM random is slow
#[test]
fn f1128_ram_random_prediction() {
    let config = CacheConfig::default();
    let prediction = BandwidthPrediction::predict(
        100.0,
        100 * 1024 * 1024, // 100MB (RAM)
        AccessPattern::Random,
        &config,
    );

    // RAM with random should be very low
    assert!(prediction.efficiency_percent < 20.0);
}

/// F1128.3: Limiting factor identification
#[test]
fn f1128_limiting_factor() {
    let config = CacheConfig::default();

    // Random pattern should be the limiting factor for L1
    let prediction = BandwidthPrediction::predict(100.0, 1024, AccessPattern::Random, &config);

    assert!(prediction.limiting_factor.contains("random"));
}

/// F1128.4: Peak bandwidth preserved
#[test]
fn f1128_peak_preserved() {
    let config = CacheConfig::default();
    let prediction = BandwidthPrediction::predict(200.0, 1024, AccessPattern::Reuse, &config);

    assert_eq!(prediction.peak_bandwidth_gbps, 200.0);
}

// =============================================================================
// F1129: Edge Cases Tests
// =============================================================================

/// F1129.1: Zero elements
#[test]
fn f1129_zero_elements() {
    let config = CacheConfig::default();
    let analysis = WorkingSetAnalysis::analyze(0, 4, 1.0, &config);

    assert_eq!(analysis.working_set_bytes, 0);
    assert_eq!(analysis.cache_level, CacheLevel::L1);
}

/// F1129.2: Huge working set
#[test]
fn f1129_huge_working_set() {
    let config = CacheConfig::default();
    let analysis = WorkingSetAnalysis::analyze(1_000_000_000, 8, 1.0, &config);

    assert_eq!(analysis.cache_level, CacheLevel::Ram);
    assert!(analysis.tiling_recommended);
}

/// F1129.3: Very small access factor
#[test]
fn f1129_small_access_factor() {
    let config = CacheConfig::default();
    let analysis = WorkingSetAnalysis::analyze(1_000_000, 4, 0.001, &config);

    // Should still produce valid result
    assert!(analysis.working_set_bytes > 0);
}

// =============================================================================
// F1130: Integration Tests
// =============================================================================

/// F1130.1: §31.2 memory bandwidth cliff at 4M elements
#[test]
fn f1130_bandwidth_cliff() {
    let config = CacheConfig::default();

    // 4M elements × 8 bytes = 32MB (exactly L3 size)
    let analysis = WorkingSetAnalysis::analyze(4_000_000, 8, 1.0, &config);

    // Should trigger L3/RAM boundary
    assert!(matches!(
        analysis.cache_level,
        CacheLevel::L3 | CacheLevel::Ram
    ));
    assert!(analysis.tiling_recommended);
}

/// F1130.2: Optimal tile maintains L2 residency
#[test]
fn f1130_tile_l2_residency() {
    let config = CacheConfig::default();
    let tile = optimal_matmul_tile(&config, 4);

    // Use tile for matrix working set
    let ws = matrix_working_set(tile, tile, tile, 4);

    // Should fit in L2
    assert!(config.classify(ws) == CacheLevel::L2 || config.classify(ws) == CacheLevel::L1);
}

/// F1130.3: Real-world LLM attention pattern
#[test]
fn f1130_llm_attention() {
    let config = CacheConfig::zen4();

    // Attention: Q, K, V matrices for batch=32, seq=2048, dim=4096
    let batch = 32;
    let seq = 2048;
    let dim = 4096;
    let element_size = 2; // fp16

    // Q, K, V each are batch × seq × dim
    let _qkv_size = 3 * batch * seq * dim * element_size;

    let analysis = WorkingSetAnalysis::analyze(
        batch * seq * dim * 3, // total elements
        element_size,
        1.0,
        &config,
    );

    // This should overflow L3 and require tiling
    assert_eq!(analysis.cache_level, CacheLevel::Ram);
    assert!(analysis.tiling_recommended);
}

/// F1130.4: Streaming workload prediction
#[test]
fn f1130_streaming_workload() {
    let config = CacheConfig::default();

    // Large streaming workload (video processing)
    let prediction = BandwidthPrediction::predict(
        50.0,               // 50 GB/s memory bandwidth
        1024 * 1024 * 1024, // 1GB stream
        AccessPattern::Streaming,
        &config,
    );

    // Streaming should achieve ~50% of peak (prefetching helps)
    assert!(prediction.efficiency_percent > 4.0);
    assert!(prediction.efficiency_percent < 10.0); // RAM × streaming = 0.1 × 0.5 = 5%
}
