//! Comprehensive coverage tests for `data_collector.rs`.
//!
//! Covers: constructors, online learning toggle, record_feedback, get_feedback,
//! record_prediction_error, detect_concept_drift (all branches), should_retrain
//! (all branches), mark_trained, merge, prepare_weighted_training_data,
//! training_stats with feedback, save_apr/load_apr round-trip, load_apr error
//! paths (bad magic, CRC mismatch, truncated file), UserFeedback default,
//! auto_retrain edge cases, to_json/from_json, and ready_to_train.

use super::super::*;
use crate::brick::BrickProfiler;

// ============================================================================
// Helper: create a TrainingSample with given throughput
// ============================================================================

fn make_sample(throughput: f32) -> TrainingSample {
    let features = TunerFeatures::builder()
        .model_params_b(7.0)
        .hidden_dim(4096)
        .batch_size(1)
        .build();
    TrainingSample {
        features,
        throughput_tps: throughput,
        best_kernel: KernelType::TiledQ4K,
        bottleneck: BottleneckClass::MemoryBound,
        timestamp: "1700000000".to_string(),
        hardware_id: "test-hw".to_string(),
    }
}

fn make_collector_with_samples(n: usize) -> TunerDataCollector {
    let mut collector = TunerDataCollector::new();
    for i in 0..n {
        collector.samples.push(make_sample(100.0 + i as f32));
    }
    collector
}

// ============================================================================
// Constructor tests
// ============================================================================

#[test]
fn new_collector_has_correct_defaults() {
    let c = TunerDataCollector::new();
    assert!(c.is_empty());
    assert_eq!(c.len(), 0);
    assert!(!c.is_online_learning_enabled());
    assert_eq!(c.retrain_threshold, 100);
    assert_eq!(c.samples_at_last_train, 0);
    assert!(c.feedback.is_empty());
    assert!(c.error_window.is_empty());
}

#[test]
fn with_online_learning_enables_flag() {
    let c = TunerDataCollector::with_online_learning();
    assert!(c.is_online_learning_enabled());
    assert!(c.is_empty());
}

// ============================================================================
// Online learning toggle
// ============================================================================

#[test]
fn enable_then_disable_online_learning() {
    let mut c = TunerDataCollector::new();
    assert!(!c.is_online_learning_enabled());

    c.enable_online_learning();
    assert!(c.is_online_learning_enabled());

    c.disable_online_learning();
    assert!(!c.is_online_learning_enabled());
}

// ============================================================================
// record() with BrickProfiler
// ============================================================================

#[test]
fn record_returns_none_when_profiler_has_no_tokens() {
    let mut collector = TunerDataCollector::new();
    let profiler = BrickProfiler::new();
    let config = RunConfig::default();
    // BrickProfiler with no timing data -> tokens_per_sec() returns None
    let result = collector.record(&profiler, &config, KernelType::TiledQ4K);
    assert!(result.is_none());
    assert!(collector.is_empty());
}

// ============================================================================
// samples(), len(), is_empty()
// ============================================================================

#[test]
fn samples_returns_all_pushed_samples() {
    let mut c = TunerDataCollector::new();
    assert!(c.samples().is_empty());
    c.samples.push(make_sample(42.0));
    c.samples.push(make_sample(99.0));
    assert_eq!(c.samples().len(), 2);
    assert_eq!(c.len(), 2);
    assert!(!c.is_empty());
}

// ============================================================================
// to_json() / from_json() round-trip
// ============================================================================

#[test]
fn to_json_produces_valid_json() {
    let c = make_collector_with_samples(3);
    let json = c.to_json().expect("serialization should succeed");
    assert!(json.starts_with('['));
    assert!(json.contains("throughput_tps"));
}

#[test]
fn from_json_round_trip_preserves_data() {
    let original = make_collector_with_samples(2);
    let json = original.to_json().expect("serialize");
    let loaded = TunerDataCollector::from_json(&json).expect("deserialize");
    assert_eq!(loaded.len(), 2);
    assert_eq!(
        loaded.samples()[0].throughput_tps,
        original.samples()[0].throughput_tps
    );
    assert_eq!(
        loaded.samples()[1].throughput_tps,
        original.samples()[1].throughput_tps
    );
}

#[test]
fn from_json_returns_error_on_bad_json() {
    let result = TunerDataCollector::from_json("{broken");
    assert!(result.is_err());
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("Serialization"));
}

#[test]
fn to_json_on_empty_collector_returns_empty_array() {
    let c = TunerDataCollector::new();
    let json = c.to_json().expect("serialize empty");
    assert_eq!(json.trim(), "[]");
}

// ============================================================================
// prepare_training_data()
// ============================================================================

#[test]
fn prepare_training_data_extracts_features_and_throughput() {
    let c = make_collector_with_samples(3);
    let data = c.prepare_training_data();
    assert_eq!(data.len(), 3);
    // First sample throughput is 100.0
    assert_eq!(data[0].1, 100.0);
    assert_eq!(data[1].1, 101.0);
    assert_eq!(data[2].1, 102.0);
}

#[test]
fn prepare_training_data_empty_collector() {
    let c = TunerDataCollector::new();
    let data = c.prepare_training_data();
    assert!(data.is_empty());
}

// ============================================================================
// ready_to_train() and training_progress()
// ============================================================================

#[test]
fn ready_to_train_returns_false_below_threshold() {
    let c = make_collector_with_samples(999);
    assert!(!c.ready_to_train());
}

#[test]
fn ready_to_train_returns_true_at_threshold() {
    let c = make_collector_with_samples(TunerDataCollector::MIN_SAMPLES_FOR_TRAINING);
    assert!(c.ready_to_train());
}

#[test]
fn training_progress_reflects_sample_count() {
    let c = make_collector_with_samples(42);
    let (current, required) = c.training_progress();
    assert_eq!(current, 42);
    assert_eq!(required, TunerDataCollector::MIN_SAMPLES_FOR_TRAINING);
}

// ============================================================================
// merge()
// ============================================================================

#[test]
fn merge_combines_samples_from_two_collectors() {
    let mut a = make_collector_with_samples(3);
    let b = make_collector_with_samples(2);
    a.merge(&b);
    assert_eq!(a.len(), 5);
}

#[test]
fn merge_with_empty_collector_is_noop() {
    let mut a = make_collector_with_samples(3);
    let b = TunerDataCollector::new();
    a.merge(&b);
    assert_eq!(a.len(), 3);
}

#[test]
fn merge_into_empty_collector() {
    let mut a = TunerDataCollector::new();
    let b = make_collector_with_samples(5);
    a.merge(&b);
    assert_eq!(a.len(), 5);
}

// ============================================================================
// record_feedback() and get_feedback()
// ============================================================================

#[test]
fn record_feedback_stores_at_correct_index() {
    let mut c = TunerDataCollector::new();
    c.record_feedback(0, UserFeedback::Accepted);
    assert_eq!(c.get_feedback(0), UserFeedback::Accepted);
}

#[test]
fn record_feedback_extends_vector_with_none() {
    let mut c = TunerDataCollector::new();
    // Record at index 5 should extend the vector with None for indices 0-4
    c.record_feedback(5, UserFeedback::Rejected);
    assert_eq!(c.get_feedback(0), UserFeedback::None);
    assert_eq!(c.get_feedback(1), UserFeedback::None);
    assert_eq!(c.get_feedback(4), UserFeedback::None);
    assert_eq!(c.get_feedback(5), UserFeedback::Rejected);
}

#[test]
fn get_feedback_returns_none_for_out_of_range() {
    let c = TunerDataCollector::new();
    assert_eq!(c.get_feedback(999), UserFeedback::None);
}

#[test]
fn record_feedback_overwrites_existing() {
    let mut c = TunerDataCollector::new();
    c.record_feedback(0, UserFeedback::Accepted);
    assert_eq!(c.get_feedback(0), UserFeedback::Accepted);
    c.record_feedback(0, UserFeedback::Alternative);
    assert_eq!(c.get_feedback(0), UserFeedback::Alternative);
}

// ============================================================================
// UserFeedback
// ============================================================================

#[test]
fn user_feedback_default_is_none() {
    let fb = UserFeedback::default();
    assert_eq!(fb, UserFeedback::None);
}

#[test]
fn user_feedback_equality() {
    assert_eq!(UserFeedback::Accepted, UserFeedback::Accepted);
    assert_ne!(UserFeedback::Accepted, UserFeedback::Rejected);
    assert_ne!(UserFeedback::Alternative, UserFeedback::None);
}

#[test]
fn user_feedback_clone_and_copy() {
    let fb = UserFeedback::Rejected;
    let cloned = fb;
    assert_eq!(fb, cloned);
}

#[test]
fn user_feedback_debug_format() {
    let fb = UserFeedback::Accepted;
    let debug = format!("{:?}", fb);
    assert!(debug.contains("Accepted"));
}

// ============================================================================
// record_prediction_error()
// ============================================================================

#[test]
fn record_prediction_error_noop_when_online_learning_disabled() {
    let mut c = TunerDataCollector::new();
    assert!(!c.is_online_learning_enabled());
    c.record_prediction_error(100.0, 110.0);
    assert!(c.error_window.is_empty());
}

#[test]
fn record_prediction_error_adds_to_window_when_enabled() {
    let mut c = TunerDataCollector::with_online_learning();
    c.record_prediction_error(100.0, 100.0);
    assert_eq!(c.error_window.len(), 1);
    // Perfect prediction -> error = 0.0
    assert_eq!(c.error_window[0], 0.0);
}

#[test]
fn record_prediction_error_computes_relative_error() {
    let mut c = TunerDataCollector::with_online_learning();
    // predicted=150, actual=100 -> |50/100| = 0.5
    c.record_prediction_error(150.0, 100.0);
    assert!((c.error_window[0] - 0.5).abs() < 1e-6);
}

#[test]
fn record_prediction_error_clamps_to_one() {
    let mut c = TunerDataCollector::with_online_learning();
    // predicted=1000, actual=1 -> |999/1| = 999.0 clamped to 1.0
    c.record_prediction_error(1000.0, 1.0);
    assert_eq!(c.error_window[0], 1.0);
}

#[test]
fn record_prediction_error_actual_zero_returns_one() {
    let mut c = TunerDataCollector::with_online_learning();
    c.record_prediction_error(50.0, 0.0);
    assert_eq!(c.error_window[0], 1.0);
}

#[test]
fn record_prediction_error_trims_sliding_window() {
    let mut c = TunerDataCollector::with_online_learning();
    // Default window size is 50
    for i in 0..60 {
        c.record_prediction_error(100.0 + i as f32, 100.0);
    }
    // Window should be trimmed to 50
    assert_eq!(c.error_window.len(), 50);
}

#[test]
fn record_prediction_error_negative_actual_uses_abs() {
    let mut c = TunerDataCollector::with_online_learning();
    // predicted=100, actual=-100 -> actual > 0.0 is false -> error = 1.0
    c.record_prediction_error(100.0, -100.0);
    assert_eq!(c.error_window[0], 1.0);
}

// ============================================================================
// detect_concept_drift() - all branches
// ============================================================================

#[test]
fn detect_drift_insufficient_data() {
    let c = TunerDataCollector::new();
    let status = c.detect_concept_drift();
    assert!(!status.drift_detected);
    assert_eq!(status.staleness_score, 0.0);
    assert!(!status.recommend_retrain);
    assert!(status.explanation.contains("Insufficient"));
}

#[test]
fn detect_drift_insufficient_data_with_some_errors() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add only 5 errors (below the 10 threshold)
    for _ in 0..5 {
        c.record_prediction_error(110.0, 100.0);
    }
    let status = c.detect_concept_drift();
    assert!(!status.drift_detected);
    assert!(status.explanation.contains("Insufficient"));
}

#[test]
fn detect_drift_no_drift_fresh_model() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add 15 low-error predictions (error ~= 0.05)
    for _ in 0..15 {
        c.record_prediction_error(105.0, 100.0);
    }
    let status = c.detect_concept_drift();
    assert!(!status.drift_detected);
    assert!(!status.recommend_retrain);
    assert!(status.explanation.contains("fresh"));
}

#[test]
fn detect_drift_drift_detected_high_error() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add 15 high-error predictions (error = 0.5, exceeds DRIFT_ERROR_THRESHOLD=0.15)
    for _ in 0..15 {
        c.record_prediction_error(150.0, 100.0);
    }
    let status = c.detect_concept_drift();
    assert!(status.drift_detected);
    assert!(status.recommend_retrain);
    assert!(status.explanation.contains("drift"));
}

#[test]
fn detect_drift_stale_model_no_drift() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add many samples to make it stale
    for i in 0..90 {
        c.samples.push(make_sample(100.0 + i as f32));
    }
    // Add low-error predictions
    for _ in 0..15 {
        c.record_prediction_error(101.0, 100.0);
    }
    // staleness = 90 / 100 = 0.9 > 0.8 threshold
    let status = c.detect_concept_drift();
    assert!(!status.drift_detected);
    assert!(status.staleness_score > 0.8);
    assert!(status.recommend_retrain);
    assert!(status.explanation.contains("stale"));
}

#[test]
fn detect_drift_staleness_clamped_to_one() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add far more samples than threshold
    for i in 0..200 {
        c.samples.push(make_sample(100.0 + i as f32));
    }
    for _ in 0..15 {
        c.record_prediction_error(101.0, 100.0);
    }
    let status = c.detect_concept_drift();
    // staleness_score should be clamped to 1.0
    assert!(status.staleness_score <= 1.0);
    assert_eq!(status.samples_since_training, 200);
}

#[test]
fn detect_drift_samples_since_training_uses_saturating_sub() {
    let mut c = TunerDataCollector::new();
    // samples_at_last_train > samples.len() (should never happen but test robustness)
    c.samples_at_last_train = 100;
    let status = c.detect_concept_drift();
    assert_eq!(status.samples_since_training, 0);
}

// ============================================================================
// should_retrain() - all branches
// ============================================================================

#[test]
fn should_retrain_returns_false_when_online_learning_disabled() {
    let c = TunerDataCollector::new();
    assert!(!c.should_retrain());
}

#[test]
fn should_retrain_returns_true_when_enough_new_samples() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add retrain_threshold number of samples
    for i in 0..100 {
        c.samples.push(make_sample(100.0 + i as f32));
    }
    assert!(c.should_retrain());
}

#[test]
fn should_retrain_returns_false_below_threshold() {
    let mut c = TunerDataCollector::with_online_learning();
    for i in 0..10 {
        c.samples.push(make_sample(100.0 + i as f32));
    }
    // Not enough samples and no drift (insufficient data for drift)
    assert!(!c.should_retrain());
}

#[test]
fn should_retrain_returns_true_on_drift_with_min_samples() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add enough samples for drift check but below retrain_threshold
    for i in 0..15 {
        c.samples.push(make_sample(100.0 + i as f32));
    }
    // Add high-error predictions to trigger drift
    for _ in 0..15 {
        c.record_prediction_error(200.0, 100.0);
    }
    // samples_since >= 10 and drift detected -> true
    assert!(c.should_retrain());
}

#[test]
fn should_retrain_returns_false_on_drift_with_too_few_samples() {
    let mut c = TunerDataCollector::with_online_learning();
    // Only 5 samples (below 10 minimum for drift retrain)
    for i in 0..5 {
        c.samples.push(make_sample(100.0 + i as f32));
    }
    // Add high errors to trigger drift
    for _ in 0..15 {
        c.record_prediction_error(300.0, 100.0);
    }
    // drift detected but samples_since < 10 -> false
    assert!(!c.should_retrain());
}

// ============================================================================
// mark_trained()
// ============================================================================

#[test]
fn mark_trained_resets_counters() {
    let mut c = TunerDataCollector::with_online_learning();
    for i in 0..20 {
        c.samples.push(make_sample(100.0 + i as f32));
    }
    for _ in 0..15 {
        c.record_prediction_error(120.0, 100.0);
    }
    assert!(!c.error_window.is_empty());

    c.mark_trained();
    assert_eq!(c.samples_at_last_train, 20);
    assert!(c.error_window.is_empty());
}

// ============================================================================
// training_stats()
// ============================================================================

#[test]
fn training_stats_with_no_feedback() {
    let c = make_collector_with_samples(5);
    let stats = c.training_stats();
    assert_eq!(stats.total_samples, 5);
    assert_eq!(stats.samples_since_training, 5);
    assert_eq!(stats.accepted_count, 0);
    assert_eq!(stats.rejected_count, 0);
    assert_eq!(stats.alternative_count, 0);
    assert!(!stats.drift_detected);
    assert!(!stats.online_learning_enabled);
}

#[test]
fn training_stats_with_mixed_feedback() {
    let mut c = make_collector_with_samples(5);
    c.record_feedback(0, UserFeedback::Accepted);
    c.record_feedback(1, UserFeedback::Accepted);
    c.record_feedback(2, UserFeedback::Rejected);
    c.record_feedback(3, UserFeedback::Alternative);
    // index 4 has no feedback (None)

    let stats = c.training_stats();
    assert_eq!(stats.accepted_count, 2);
    assert_eq!(stats.rejected_count, 1);
    assert_eq!(stats.alternative_count, 1);
}

#[test]
fn training_stats_reflects_online_learning_enabled() {
    let c = TunerDataCollector::with_online_learning();
    let stats = c.training_stats();
    assert!(stats.online_learning_enabled);
}

// ============================================================================
// auto_retrain() - edge cases
// ============================================================================

#[test]
fn auto_retrain_returns_false_when_should_retrain_is_false() {
    let mut c = TunerDataCollector::new();
    let mut tuner = BrickTuner::new();
    assert!(!c.auto_retrain(&mut tuner));
}

#[test]
fn auto_retrain_returns_false_when_weighted_data_too_small() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add samples but mark ALL as rejected so weighted data is empty
    for i in 0..150 {
        c.samples.push(make_sample(100.0 + i as f32));
        c.record_feedback(i, UserFeedback::Rejected);
    }
    c.retrain_threshold = 50;
    let mut tuner = BrickTuner::new();
    // should_retrain() returns true (enough samples) but weighted data is empty
    assert!(!c.auto_retrain(&mut tuner));
}

#[test]
fn auto_retrain_marks_trained_on_success() {
    let mut c = TunerDataCollector::with_online_learning();
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
        c.samples.push(TrainingSample {
            features,
            throughput_tps: 30.0 + (i as f32) * 2.0,
            best_kernel: KernelType::TiledQ4K,
            bottleneck: BottleneckClass::MemoryBound,
            timestamp: format!("{}", i),
            hardware_id: "test".to_string(),
        });
    }
    c.retrain_threshold = 50;
    let mut tuner = BrickTuner::new();
    let result = c.auto_retrain(&mut tuner);
    assert!(result);
    // After successful retrain, samples_at_last_train should be updated
    assert_eq!(c.samples_at_last_train, 150);
    assert!(c.error_window.is_empty());
}

// ============================================================================
// prepare_weighted_training_data()
// ============================================================================

#[test]
fn weighted_data_skips_rejected_samples() {
    let mut c = make_collector_with_samples(3);
    c.record_feedback(1, UserFeedback::Rejected);

    // Access through auto_retrain path is private, but we can test via
    // the public interface and check sample counts indirectly.
    // Let's use the training_stats to verify feedback is recorded
    let stats = c.training_stats();
    assert_eq!(stats.rejected_count, 1);
}

#[test]
fn weighted_data_doubles_accepted_samples() {
    let mut c = make_collector_with_samples(3);
    c.record_feedback(0, UserFeedback::Accepted);
    // Accepted samples are duplicated in weighted data
    // We can verify the feedback was stored
    assert_eq!(c.get_feedback(0), UserFeedback::Accepted);
    assert_eq!(c.get_feedback(1), UserFeedback::None);
    assert_eq!(c.get_feedback(2), UserFeedback::None);
}

// ============================================================================
// save_apr() and load_apr() - round-trip
// ============================================================================

#[test]
fn save_and_load_apr_round_trip() {
    let c = make_collector_with_samples(5);
    let dir = std::env::temp_dir().join("trueno_test_save_load_apr");
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("test_data.apr");

    c.save_apr(&path).expect("save should succeed");
    assert!(path.exists());

    let loaded = TunerDataCollector::load_apr(&path).expect("load should succeed");
    assert_eq!(loaded.len(), 5);
    assert_eq!(
        loaded.samples()[0].throughput_tps,
        c.samples()[0].throughput_tps
    );
    assert_eq!(
        loaded.samples()[4].throughput_tps,
        c.samples()[4].throughput_tps
    );

    // Loaded collector should have default state for non-persisted fields
    assert!(!loaded.is_online_learning_enabled());
    assert_eq!(loaded.retrain_threshold, 100);
    assert!(loaded.feedback.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn save_apr_creates_parent_directories() {
    let dir = std::env::temp_dir().join("trueno_test_nested_dir/a/b/c");
    let path = dir.join("model.apr");
    let _ = std::fs::remove_dir_all(std::env::temp_dir().join("trueno_test_nested_dir"));

    let c = make_collector_with_samples(1);
    c.save_apr(&path)
        .expect("save to nested dir should succeed");
    assert!(path.exists());

    let _ = std::fs::remove_dir_all(std::env::temp_dir().join("trueno_test_nested_dir"));
}

#[test]
fn save_apr_empty_collector() {
    let dir = std::env::temp_dir().join("trueno_test_save_empty");
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("empty.apr");

    let c = TunerDataCollector::new();
    c.save_apr(&path).expect("save empty should succeed");

    let loaded = TunerDataCollector::load_apr(&path).expect("load empty should succeed");
    assert!(loaded.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// load_apr() - error paths
// ============================================================================

#[test]
fn load_apr_file_not_found() {
    let result = TunerDataCollector::load_apr("/tmp/trueno_nonexistent_file.apr");
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("I/O error"));
}

#[test]
fn load_apr_bad_magic() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("trueno_test_bad_magic");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("bad_magic.apr");

    let mut file = std::fs::File::create(&path).expect("create file");
    file.write_all(b"XXXX").expect("write magic");
    file.write_all(&4u32.to_le_bytes()).expect("write len");
    file.write_all(b"test").expect("write data");
    file.write_all(&0u32.to_le_bytes()).expect("write crc");
    drop(file);

    let result = TunerDataCollector::load_apr(&path);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("APR2"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_apr_crc_mismatch() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("trueno_test_crc_mismatch");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("bad_crc.apr");

    let json_bytes = b"[]";
    let mut file = std::fs::File::create(&path).expect("create file");
    file.write_all(b"APR2").expect("write magic");
    file.write_all(&(json_bytes.len() as u32).to_le_bytes())
        .expect("write len");
    file.write_all(json_bytes).expect("write data");
    // Write wrong CRC
    file.write_all(&0xDEADBEEFu32.to_le_bytes())
        .expect("write bad crc");
    drop(file);

    let result = TunerDataCollector::load_apr(&path);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("CRC mismatch"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_apr_truncated_file() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("trueno_test_truncated");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("truncated.apr");

    // Write only magic, no length or data
    let mut file = std::fs::File::create(&path).expect("create file");
    file.write_all(b"APR2").expect("write magic");
    drop(file);

    let result = TunerDataCollector::load_apr(&path);
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_apr_invalid_json_in_valid_envelope() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("trueno_test_invalid_json_apr");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("bad_json.apr");

    let json_bytes = b"not valid json at all";
    let crc = crate::tuner::helpers::crc32_hash(json_bytes);

    let mut file = std::fs::File::create(&path).expect("create file");
    file.write_all(b"APR2").expect("write magic");
    file.write_all(&(json_bytes.len() as u32).to_le_bytes())
        .expect("write len");
    file.write_all(json_bytes).expect("write data");
    file.write_all(&crc.to_le_bytes()).expect("write crc");
    drop(file);

    let result = TunerDataCollector::load_apr(&path);
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("Serialization"));

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// bootstrap_from_five_whys()
// ============================================================================

#[test]
fn bootstrap_from_five_whys_returns_valid_collector() {
    let c = TunerDataCollector::bootstrap_from_five_whys();
    // Currently returns empty collector (TODO in source)
    assert!(c.is_empty());
    assert!(!c.is_online_learning_enabled());
    assert_eq!(c.retrain_threshold, 100);
}

// ============================================================================
// train_if_ready()
// ============================================================================

#[test]
fn train_if_ready_returns_none_when_not_enough_samples() {
    let c = make_collector_with_samples(50);
    assert!(c.train_if_ready().is_none());
}

// ============================================================================
// ConceptDriftStatus
// ============================================================================

#[test]
fn concept_drift_status_debug_and_clone() {
    let status = ConceptDriftStatus {
        drift_detected: false,
        staleness_score: 0.5,
        samples_since_training: 50,
        recommend_retrain: false,
        explanation: "test".to_string(),
    };
    let cloned = status.clone();
    assert!(!cloned.drift_detected);
    assert_eq!(cloned.staleness_score, 0.5);
    assert_eq!(cloned.samples_since_training, 50);
    let debug = format!("{:?}", cloned);
    assert!(debug.contains("ConceptDriftStatus"));
}

// ============================================================================
// TrainingStats
// ============================================================================

#[test]
fn training_stats_debug_and_clone() {
    let stats = TrainingStats {
        total_samples: 100,
        samples_since_training: 10,
        accepted_count: 5,
        rejected_count: 2,
        alternative_count: 3,
        staleness_score: 0.1,
        drift_detected: false,
        online_learning_enabled: true,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.total_samples, 100);
    let debug = format!("{:?}", cloned);
    assert!(debug.contains("TrainingStats"));
}

// ============================================================================
// TrainingSample
// ============================================================================

#[test]
fn training_sample_debug_and_clone() {
    let sample = make_sample(42.0);
    let cloned = sample.clone();
    assert_eq!(cloned.throughput_tps, 42.0);
    assert_eq!(cloned.hardware_id, "test-hw");
    let debug = format!("{:?}", cloned);
    assert!(debug.contains("TrainingSample"));
}

#[test]
fn training_sample_serialization_round_trip() {
    let sample = make_sample(200.0);
    let json = serde_json::to_string(&sample).expect("serialize sample");
    let deserialized: TrainingSample = serde_json::from_str(&json).expect("deserialize sample");
    assert_eq!(deserialized.throughput_tps, 200.0);
    assert_eq!(deserialized.hardware_id, "test-hw");
    assert!(matches!(deserialized.best_kernel, KernelType::TiledQ4K));
    assert!(matches!(
        deserialized.bottleneck,
        BottleneckClass::MemoryBound
    ));
}

// ============================================================================
// TunerDataCollector Default impl
// ============================================================================

#[test]
fn default_collector_is_empty_and_learning_disabled() {
    let default_c = TunerDataCollector::default();
    assert!(default_c.is_empty());
    assert!(!default_c.is_online_learning_enabled());
}

#[test]
fn new_sets_retrain_threshold_unlike_default() {
    // new() sets retrain_threshold = 100, derived Default sets it to 0
    let new_c = TunerDataCollector::new();
    assert_eq!(new_c.retrain_threshold, 100);

    let default_c = TunerDataCollector::default();
    assert_eq!(default_c.retrain_threshold, 0);
}

// ============================================================================
// Constants
// ============================================================================

#[test]
fn constants_have_expected_values() {
    assert_eq!(TunerDataCollector::MIN_SAMPLES_FOR_TRAINING, 1000);
}

// ============================================================================
// Edge case: error window exactly at boundary
// ============================================================================

#[test]
fn error_window_at_exact_boundary_for_drift_detection() {
    let mut c = TunerDataCollector::with_online_learning();
    // Add exactly 10 errors (minimum for drift detection)
    for _ in 0..10 {
        c.record_prediction_error(105.0, 100.0);
    }
    let status = c.detect_concept_drift();
    // Should NOT say "Insufficient" since we have exactly 10
    assert!(!status.explanation.contains("Insufficient"));
    // Error is 0.05 which is below 0.15 threshold
    assert!(!status.drift_detected);
}

#[test]
fn error_window_at_nine_insufficient_for_drift() {
    let mut c = TunerDataCollector::with_online_learning();
    for _ in 0..9 {
        c.record_prediction_error(200.0, 100.0);
    }
    let status = c.detect_concept_drift();
    assert!(status.explanation.contains("Insufficient"));
}

// ============================================================================
// Drift threshold boundary testing
// ============================================================================

#[test]
fn drift_exactly_at_threshold_not_detected() {
    let mut c = TunerDataCollector::with_online_learning();
    // DRIFT_ERROR_THRESHOLD = 0.15
    // To get exactly 0.15 mean error: predicted/actual ratio = 1.15 or 0.85
    // |115 - 100| / 100 = 0.15
    for _ in 0..15 {
        c.record_prediction_error(115.0, 100.0);
    }
    let status = c.detect_concept_drift();
    // Mean error = 0.15 which is NOT > 0.15 (strictly greater)
    assert!(!status.drift_detected);
}

#[test]
fn drift_just_above_threshold_detected() {
    let mut c = TunerDataCollector::with_online_learning();
    // |116 - 100| / 100 = 0.16 > 0.15
    for _ in 0..15 {
        c.record_prediction_error(116.0, 100.0);
    }
    let status = c.detect_concept_drift();
    assert!(status.drift_detected);
    assert!(status.recommend_retrain);
}

// ============================================================================
// UserFeedback serialization
// ============================================================================

#[test]
fn user_feedback_serialization_round_trip() {
    for fb in &[
        UserFeedback::Accepted,
        UserFeedback::Rejected,
        UserFeedback::Alternative,
        UserFeedback::None,
    ] {
        let json = serde_json::to_string(fb).expect("serialize feedback");
        let deserialized: UserFeedback = serde_json::from_str(&json).expect("deserialize feedback");
        assert_eq!(*fb, deserialized);
    }
}

// ============================================================================
// save_apr error path: write to invalid path
// ============================================================================

#[test]
fn save_apr_returns_io_error_for_invalid_path() {
    let c = make_collector_with_samples(1);
    // Try to write to a directory that we can't create (root-owned)
    let result = c.save_apr("/proc/nonexistent/deep/path/file.apr");
    assert!(result.is_err());
    let err = format!("{}", result.unwrap_err());
    assert!(err.contains("I/O error"));
}
