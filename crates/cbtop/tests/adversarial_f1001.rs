//! Adversarial Falsification Tests (F1001-F1020)
//!
//! Popperian falsification criteria for adversarial testing per §36.
//! Instead of "proving it works," we attempt to break the system.

use cbtop::{
    AdversarialError, AdversarialTactic, AdversarialTestSummary, BitFlipInjector,
    CancellationToken, CheckedArithmetic, ConfigValidator, InputValidator, MonotonicClock,
    RecoveryHandler, ResourceLimiter,
};
use std::time::Duration;

// ============================================================================
// F1001: Bit-flip in tensor maintains safety
// ============================================================================

#[test]
fn f1001_bit_flip_detected_by_checksum() {
    let data = b"test data for checksum validation";
    let original_checksum = InputValidator::compute_checksum(data);

    // Inject bit flips
    let injector = BitFlipInjector::new(42, 3);
    let corrupted = injector.inject(data);

    // Checksum should detect corruption
    let corrupted_checksum = InputValidator::compute_checksum(&corrupted);
    assert_ne!(original_checksum, corrupted_checksum);
}

#[test]
fn f1001_bit_flip_produces_different_output() {
    let injector = BitFlipInjector::new(12345, 5);
    let data = vec![0xAAu8; 1000];
    let corrupted = injector.inject(&data);

    // Should be different but same length
    assert_eq!(data.len(), corrupted.len());
    assert_ne!(data, corrupted);
}

// ============================================================================
// F1002: Arbitrary bit-flips detected
// ============================================================================

#[test]
fn f1002_checksum_verification_fails_on_corruption() {
    let validator = InputValidator::new();
    let data = b"important data";
    let checksum = InputValidator::compute_checksum(data);

    // Corrupt the data
    let mut corrupted = data.to_vec();
    corrupted[0] ^= 0xFF;

    // Verification should fail
    let result = validator.verify_checksum(&corrupted, checksum);
    assert!(matches!(
        result,
        Err(AdversarialError::CorruptedInput { .. })
    ));
}

#[test]
fn f1002_checksum_passes_on_valid_data() {
    let validator = InputValidator::new();
    let data = b"valid data";
    let checksum = InputValidator::compute_checksum(data);

    let result = validator.verify_checksum(data, checksum);
    assert!(result.is_ok());
}

// ============================================================================
// F1003: Memory pressure handled gracefully
// ============================================================================

#[test]
fn f1003_memory_request_within_limit_succeeds() {
    let mut limiter = ResourceLimiter::new().with_max_memory(1024 * 1024);
    assert!(limiter.request_memory(1024).is_ok());
}

#[test]
fn f1003_memory_request_exceeding_limit_fails() {
    let mut limiter = ResourceLimiter::new().with_max_memory(1024);
    let result = limiter.request_memory(2048);
    assert!(matches!(
        result,
        Err(AdversarialError::ResourceExhausted { .. })
    ));
}

#[test]
fn f1003_memory_release_works() {
    let mut limiter = ResourceLimiter::new().with_max_memory(1024);
    limiter.request_memory(512).unwrap();
    limiter.release_memory(512);
    // Should be able to request again
    assert!(limiter.request_memory(1024).is_ok());
}

// ============================================================================
// F1004: Zero-size inputs handled
// ============================================================================

#[test]
fn f1004_zero_size_bytes_rejected() {
    let validator = InputValidator::new();
    let result = validator.validate_bytes(&[]);
    assert!(matches!(result, Err(AdversarialError::ZeroSizeInput)));
}

#[test]
fn f1004_zero_size_floats_rejected() {
    let validator = InputValidator::new();
    let result = validator.validate_floats(&[]);
    assert!(matches!(result, Err(AdversarialError::ZeroSizeInput)));
}

// ============================================================================
// F1005: Maximum-size inputs handled
// ============================================================================

#[test]
fn f1005_max_size_bytes_rejected() {
    let validator = InputValidator::new().with_max_size(100);
    let data = vec![0u8; 200];
    let result = validator.validate_bytes(&data);
    assert!(matches!(
        result,
        Err(AdversarialError::MaxSizeExceeded { .. })
    ));
}

#[test]
fn f1005_within_max_size_accepted() {
    let validator = InputValidator::new().with_max_size(1000);
    let data = vec![0u8; 500];
    assert!(validator.validate_bytes(&data).is_ok());
}

// ============================================================================
// F1006: Clock skew doesn't corrupt state
// ============================================================================

#[test]
fn f1006_monotonic_clock_maintains_order() {
    let mut clock = MonotonicClock::new();

    let t1 = clock.tick().unwrap();
    std::thread::sleep(Duration::from_millis(1));
    let t2 = clock.tick().unwrap();
    std::thread::sleep(Duration::from_millis(1));
    let t3 = clock.tick().unwrap();

    assert!(t2 >= t1);
    assert!(t3 >= t2);
}

#[test]
fn f1006_clock_reset_works() {
    let mut clock = MonotonicClock::new();
    clock.tick().unwrap();
    assert!(clock.elapsed().is_some());

    clock.reset();
    assert!(clock.elapsed().is_none());
}

// ============================================================================
// F1007: Concurrent access is safe
// ============================================================================

#[test]
fn f1007_cancellation_token_thread_safe() {
    let token = CancellationToken::new();
    let token_clone = token.clone_token();

    // Spawn thread that cancels
    let handle = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(10));
        token_clone.cancel();
    });

    // Wait for cancellation
    while !token.is_cancelled() {
        std::thread::sleep(Duration::from_millis(1));
    }

    handle.join().unwrap();
    assert!(token.is_cancelled());
}

// ============================================================================
// F1008: Config corruption detected
// ============================================================================

#[test]
fn f1008_unclosed_brackets_rejected() {
    let validator = ConfigValidator::new();
    let result = validator.validate_toml_string("[section");
    assert!(matches!(
        result,
        Err(AdversarialError::ConfigParseError { .. })
    ));
}

#[test]
fn f1008_unclosed_quotes_rejected() {
    let validator = ConfigValidator::new();
    let result = validator.validate_toml_string(r#"key = "unclosed"#);
    assert!(matches!(
        result,
        Err(AdversarialError::ConfigParseError { .. })
    ));
}

#[test]
fn f1008_empty_config_rejected() {
    let validator = ConfigValidator::new();
    let result = validator.validate_toml_string("");
    assert!(matches!(
        result,
        Err(AdversarialError::ConfigParseError { .. })
    ));
}

#[test]
fn f1008_valid_toml_accepted() {
    let validator = ConfigValidator::new();
    let result = validator.validate_toml_string(
        r#"
[section]
key = "value"
number = 42
"#,
    );
    assert!(result.is_ok());
}

// ============================================================================
// F1009: Pathological configs bounded
// ============================================================================

#[test]
fn f1009_value_below_min_rejected() {
    let validator = ConfigValidator::new().with_bound("temperature", 0.0, 2.0);

    let result = validator.validate_numeric("temperature", -1.0);
    assert!(matches!(
        result,
        Err(AdversarialError::ConfigOutOfBounds { .. })
    ));
}

#[test]
fn f1009_value_above_max_rejected() {
    let validator = ConfigValidator::new().with_bound("temperature", 0.0, 2.0);

    let result = validator.validate_numeric("temperature", 5.0);
    assert!(matches!(
        result,
        Err(AdversarialError::ConfigOutOfBounds { .. })
    ));
}

#[test]
fn f1009_value_within_bounds_accepted() {
    let validator = ConfigValidator::new().with_bound("temperature", 0.0, 2.0);

    let result = validator.validate_numeric("temperature", 1.0);
    assert!(result.is_ok());
}

// ============================================================================
// F1010: Double-free prevented (Rust guarantees this)
// F1011: Use-after-free prevented (Rust guarantees this)
// ============================================================================

#[test]
fn f1010_f1011_rust_memory_safety() {
    // Rust's ownership system prevents double-free and use-after-free at compile time.
    // This test documents that we rely on Rust's guarantees.

    let data = vec![1, 2, 3, 4, 5];
    let sum: i32 = data.iter().sum();
    assert_eq!(sum, 15);
    // `data` is dropped here, and Rust guarantees no double-free or use-after-free
}

// ============================================================================
// F1012: Integer overflow handled
// ============================================================================

#[test]
fn f1012_i64_add_overflow_detected() {
    let result = CheckedArithmetic::checked_add_i64(i64::MAX, 1);
    assert!(matches!(
        result,
        Err(AdversarialError::IntegerOverflow { .. })
    ));
}

#[test]
fn f1012_i64_mul_overflow_detected() {
    let result = CheckedArithmetic::checked_mul_i64(i64::MAX, 2);
    assert!(matches!(
        result,
        Err(AdversarialError::IntegerOverflow { .. })
    ));
}

#[test]
fn f1012_usize_add_overflow_detected() {
    let result = CheckedArithmetic::checked_add_usize(usize::MAX, 1);
    assert!(matches!(
        result,
        Err(AdversarialError::IntegerOverflow { .. })
    ));
}

#[test]
fn f1012_valid_arithmetic_succeeds() {
    assert_eq!(CheckedArithmetic::checked_add_i64(10, 20).unwrap(), 30);
    assert_eq!(CheckedArithmetic::checked_mul_i64(10, 20).unwrap(), 200);
}

// ============================================================================
// F1013: Division by zero handled
// ============================================================================

#[test]
fn f1013_float_div_zero_detected() {
    let result = CheckedArithmetic::checked_div_f64(10.0, 0.0);
    assert!(matches!(
        result,
        Err(AdversarialError::DivisionByZero { .. })
    ));
}

#[test]
fn f1013_int_div_zero_detected() {
    let result = CheckedArithmetic::checked_div_i64(10, 0);
    assert!(matches!(
        result,
        Err(AdversarialError::DivisionByZero { .. })
    ));
}

#[test]
fn f1013_valid_division_succeeds() {
    assert!((CheckedArithmetic::checked_div_f64(10.0, 2.0).unwrap() - 5.0).abs() < 1e-10);
    assert_eq!(CheckedArithmetic::checked_div_i64(10, 2).unwrap(), 5);
}

// ============================================================================
// F1014: NaN propagation controlled
// ============================================================================

#[test]
fn f1014_nan_detected_in_floats() {
    let validator = InputValidator::new();
    let data = vec![1.0, 2.0, f32::NAN, 4.0];
    let result = validator.validate_floats(&data);
    assert!(matches!(
        result,
        Err(AdversarialError::NaNDetected { index: 2 })
    ));
}

#[test]
fn f1014_config_nan_rejected() {
    let validator = ConfigValidator::new();
    let result = validator.validate_numeric("value", f64::NAN);
    assert!(matches!(
        result,
        Err(AdversarialError::ConfigParseError { .. })
    ));
}

// ============================================================================
// F1015: Inf propagation controlled
// ============================================================================

#[test]
fn f1015_positive_inf_detected() {
    let validator = InputValidator::new();
    let data = vec![1.0, f32::INFINITY, 3.0];
    let result = validator.validate_floats(&data);
    assert!(matches!(
        result,
        Err(AdversarialError::InfinityDetected {
            index: 1,
            positive: true
        })
    ));
}

#[test]
fn f1015_negative_inf_detected() {
    let validator = InputValidator::new();
    let data = vec![1.0, f32::NEG_INFINITY, 3.0];
    let result = validator.validate_floats(&data);
    assert!(matches!(
        result,
        Err(AdversarialError::InfinityDetected {
            index: 1,
            positive: false
        })
    ));
}

// ============================================================================
// F1016: Stack overflow prevented
// ============================================================================

#[test]
fn f1016_stack_depth_limit_enforced() {
    let mut limiter = ResourceLimiter::new().with_max_depth(10);

    // Should succeed 10 times
    for _ in 0..10 {
        assert!(limiter.enter_recursion().is_ok());
    }

    // 11th should fail
    let result = limiter.enter_recursion();
    assert!(matches!(
        result,
        Err(AdversarialError::StackOverflow { .. })
    ));
}

#[test]
fn f1016_recursion_exit_decrements_depth() {
    let mut limiter = ResourceLimiter::new().with_max_depth(3);

    limiter.enter_recursion().unwrap();
    limiter.enter_recursion().unwrap();
    limiter.enter_recursion().unwrap();
    // At max

    limiter.exit_recursion();
    // Should be able to enter again
    assert!(limiter.enter_recursion().is_ok());
}

// ============================================================================
// F1017: Resource exhaustion graceful
// ============================================================================

#[test]
fn f1017_cumulative_memory_tracked() {
    let mut limiter = ResourceLimiter::new().with_max_memory(1000);

    limiter.request_memory(400).unwrap();
    limiter.request_memory(400).unwrap();
    // Third request would exceed
    let result = limiter.request_memory(400);
    assert!(matches!(
        result,
        Err(AdversarialError::ResourceExhausted { .. })
    ));
}

// ============================================================================
// F1018: Timeout enforcement correct
// ============================================================================

#[test]
fn f1018_timeout_not_reached() {
    let mut limiter = ResourceLimiter::new().with_timeout(Duration::from_secs(60));
    limiter.start_operation();

    // Immediately check - should not timeout
    assert!(limiter.check_timeout("test_op").is_ok());
}

#[test]
fn f1018_timeout_enforced() {
    let mut limiter = ResourceLimiter::new().with_timeout(Duration::from_millis(10));
    limiter.start_operation();

    // Wait past timeout
    std::thread::sleep(Duration::from_millis(20));

    let result = limiter.check_timeout("test_op");
    assert!(matches!(result, Err(AdversarialError::Timeout { .. })));
}

// ============================================================================
// F1019: Cancellation safe
// ============================================================================

#[test]
fn f1019_uncancelled_check_succeeds() {
    let token = CancellationToken::new();
    assert!(token.check("operation").is_ok());
}

#[test]
fn f1019_cancelled_check_fails() {
    let token = CancellationToken::new();
    token.cancel();
    let result = token.check("operation");
    assert!(matches!(result, Err(AdversarialError::Cancelled { .. })));
}

// ============================================================================
// F1020: Recovery after failure
// ============================================================================

#[test]
fn f1020_recovery_without_checkpoint_fails() {
    let handler: RecoveryHandler<i32> = RecoveryHandler::new();
    let result = handler.recover();
    assert!(matches!(
        result,
        Err(AdversarialError::RecoveryFailed { .. })
    ));
}

#[test]
fn f1020_recovery_with_checkpoint_succeeds() {
    let mut handler: RecoveryHandler<String> = RecoveryHandler::new();
    handler.checkpoint("saved_state".to_string());

    let recovered = handler.recover().unwrap();
    assert_eq!(recovered, "saved_state");
}

#[test]
fn f1020_has_checkpoint_accurate() {
    let mut handler: RecoveryHandler<u32> = RecoveryHandler::new();
    assert!(!handler.has_checkpoint());

    handler.checkpoint(42);
    assert!(handler.has_checkpoint());
}

// ============================================================================
// Tactic Coverage Tests
// ============================================================================

#[test]
fn tactic_all_tactics_defined() {
    let tactics = AdversarialTactic::all();
    assert_eq!(tactics.len(), 5);
}

#[test]
fn tactic_names_non_empty() {
    for tactic in AdversarialTactic::all() {
        assert!(!tactic.name().is_empty());
        assert!(!tactic.tool().is_empty());
    }
}

// ============================================================================
// Summary Tests
// ============================================================================

#[test]
fn summary_tracks_passes() {
    let mut summary = AdversarialTestSummary::new();
    summary.record_pass(AdversarialTactic::BitFlipInjection);
    summary.record_pass(AdversarialTactic::ConfigFuzzing);

    assert_eq!(summary.total_tests, 2);
    assert_eq!(summary.passed, 2);
    assert_eq!(summary.failed, 0);
    assert!(summary.all_passed());
    assert!((summary.pass_rate() - 100.0).abs() < 1e-10);
}

#[test]
fn summary_tracks_failures() {
    let mut summary = AdversarialTestSummary::new();
    summary.record_pass(AdversarialTactic::BitFlipInjection);
    summary.record_fail(AdversarialTactic::ConfigFuzzing, "config failed");

    assert_eq!(summary.total_tests, 2);
    assert_eq!(summary.passed, 1);
    assert_eq!(summary.failed, 1);
    assert!(!summary.all_passed());
    assert!((summary.pass_rate() - 50.0).abs() < 1e-10);
}

#[test]
fn summary_empty_pass_rate_zero() {
    let summary = AdversarialTestSummary::new();
    assert!((summary.pass_rate() - 0.0).abs() < 1e-10);
}
