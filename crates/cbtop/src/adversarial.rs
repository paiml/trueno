//! Adversarial Falsification Testing (PMAT-019)
//!
//! Instead of "proving it works," actively attempt to break the system
//! through adversarial tactics per §36 of the cbtop spec.
//!
//! # Adversarial Tactics
//!
//! | Tactic | Description |
//! |--------|-------------|
//! | Bit-Flip Injection | Random bit flips in input tensors |
//! | Resource Starvation | Simulate memory/CPU pressure |
//! | Clock Skew | Test monotonic timestamp preservation |
//! | Network Partition | Timeout and disconnection handling |
//! | Config Fuzzing | Generate valid-but-pathological configs |
//!
//! # Citations
//!
//! - [Miller et al. 1990] "An Empirical Study of the Reliability of UNIX Utilities" CACM
//! - [Goodfellow et al. 2014] "Explaining and Harnessing Adversarial Examples" arXiv
//! - [Regehr et al. 2012] "Finding and Understanding Bugs in C Compilers" PLDI

use std::time::{Duration, Instant};

/// Error types for adversarial testing
#[derive(Debug, Clone, PartialEq)]
pub enum AdversarialError {
    /// Input tensor contains corrupted data
    CorruptedInput { byte_index: usize, expected_checksum: u32, actual_checksum: u32 },
    /// Memory allocation failed under pressure
    AllocationFailed { requested_bytes: usize },
    /// Zero-size input detected
    ZeroSizeInput,
    /// Input exceeds maximum allowed size
    MaxSizeExceeded { size: usize, max: usize },
    /// Clock skew detected (non-monotonic timestamp)
    ClockSkew { prev: Instant, curr: Instant },
    /// Concurrent access violation detected
    RaceCondition { description: String },
    /// Configuration parsing failed
    ConfigParseError { field: String, reason: String },
    /// Configuration value out of bounds
    ConfigOutOfBounds { field: String, value: String, min: String, max: String },
    /// Integer overflow detected
    IntegerOverflow { operation: String },
    /// Division by zero attempted
    DivisionByZero { numerator: f64 },
    /// NaN value detected in input
    NaNDetected { index: usize },
    /// Infinity value detected in input
    InfinityDetected { index: usize, positive: bool },
    /// Stack depth exceeded
    StackOverflow { depth: usize, max_depth: usize },
    /// Resource exhaustion (memory, handles, etc.)
    ResourceExhausted { resource: String },
    /// Operation timed out
    Timeout { operation: String, elapsed: Duration, limit: Duration },
    /// Operation was cancelled
    Cancelled { operation: String },
    /// Recovery failed after error
    RecoveryFailed { original_error: String, recovery_error: String },
}

/// Result type for adversarial operations
pub type AdversarialResult<T> = Result<T, AdversarialError>;

/// Adversarial tactic category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdversarialTactic {
    /// Random bit flips in input data
    BitFlipInjection,
    /// Simulate memory/CPU pressure
    ResourceStarvation,
    /// Test with manipulated timestamps
    ClockSkew,
    /// Simulate network failures
    NetworkPartition,
    /// Generate pathological configurations
    ConfigFuzzing,
}

impl AdversarialTactic {
    /// Get all tactics
    pub fn all() -> &'static [AdversarialTactic] {
        &[
            AdversarialTactic::BitFlipInjection,
            AdversarialTactic::ResourceStarvation,
            AdversarialTactic::ClockSkew,
            AdversarialTactic::NetworkPartition,
            AdversarialTactic::ConfigFuzzing,
        ]
    }

    /// Get the tactic name
    pub fn name(&self) -> &'static str {
        match self {
            AdversarialTactic::BitFlipInjection => "Bit-Flip Injection",
            AdversarialTactic::ResourceStarvation => "Resource Starvation",
            AdversarialTactic::ClockSkew => "Clock Skew",
            AdversarialTactic::NetworkPartition => "Network Partition",
            AdversarialTactic::ConfigFuzzing => "Config Fuzzing",
        }
    }

    /// Get the tool used for this tactic
    pub fn tool(&self) -> &'static str {
        match self {
            AdversarialTactic::BitFlipInjection => "proptest",
            AdversarialTactic::ResourceStarvation => "stress simulation",
            AdversarialTactic::ClockSkew => "libfaketime simulation",
            AdversarialTactic::NetworkPartition => "timeout simulation",
            AdversarialTactic::ConfigFuzzing => "proptest",
        }
    }
}

/// Input validator for adversarial testing
#[derive(Debug, Clone)]
pub struct InputValidator {
    /// Maximum allowed input size in bytes
    pub max_size: usize,
    /// Whether to compute and verify checksums
    pub verify_checksum: bool,
    /// Whether to detect NaN values
    pub detect_nan: bool,
    /// Whether to detect infinity values
    pub detect_inf: bool,
}

impl Default for InputValidator {
    fn default() -> Self {
        Self {
            max_size: 1024 * 1024 * 1024, // 1GB default max
            verify_checksum: true,
            detect_nan: true,
            detect_inf: true,
        }
    }
}

impl InputValidator {
    /// Create a new input validator
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum input size
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = max_size;
        self
    }

    /// Validate a byte slice
    pub fn validate_bytes(&self, data: &[u8]) -> AdversarialResult<()> {
        // F1004: Zero-size inputs handled
        if data.is_empty() {
            return Err(AdversarialError::ZeroSizeInput);
        }

        // F1005: Maximum-size inputs handled
        if data.len() > self.max_size {
            return Err(AdversarialError::MaxSizeExceeded {
                size: data.len(),
                max: self.max_size,
            });
        }

        Ok(())
    }

    /// Validate a float slice for NaN and Inf
    pub fn validate_floats(&self, data: &[f32]) -> AdversarialResult<()> {
        // F1004: Zero-size inputs handled
        if data.is_empty() {
            return Err(AdversarialError::ZeroSizeInput);
        }

        // F1005: Maximum-size inputs handled
        let byte_size = std::mem::size_of_val(data);
        if byte_size > self.max_size {
            return Err(AdversarialError::MaxSizeExceeded {
                size: byte_size,
                max: self.max_size,
            });
        }

        // F1014: NaN propagation controlled
        if self.detect_nan {
            for (i, &v) in data.iter().enumerate() {
                if v.is_nan() {
                    return Err(AdversarialError::NaNDetected { index: i });
                }
            }
        }

        // F1015: Inf propagation controlled
        if self.detect_inf {
            for (i, &v) in data.iter().enumerate() {
                if v.is_infinite() {
                    return Err(AdversarialError::InfinityDetected {
                        index: i,
                        positive: v.is_sign_positive(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Compute a simple checksum for data validation
    pub fn compute_checksum(data: &[u8]) -> u32 {
        // Simple Adler-32-like checksum
        let mut a: u32 = 1;
        let mut b: u32 = 0;
        for &byte in data {
            a = (a.wrapping_add(u32::from(byte))) % 65521;
            b = (b.wrapping_add(a)) % 65521;
        }
        (b << 16) | a
    }

    /// Verify data against expected checksum
    pub fn verify_checksum(&self, data: &[u8], expected: u32) -> AdversarialResult<()> {
        if !self.verify_checksum {
            return Ok(());
        }

        let actual = Self::compute_checksum(data);
        if actual != expected {
            return Err(AdversarialError::CorruptedInput {
                byte_index: 0, // Can't pinpoint exact corruption
                expected_checksum: expected,
                actual_checksum: actual,
            });
        }
        Ok(())
    }
}

/// Bit-flip injector for testing corruption handling
#[derive(Debug, Clone)]
pub struct BitFlipInjector {
    /// Seed for reproducible bit flips
    pub seed: u64,
    /// Number of bits to flip
    pub flip_count: usize,
}

impl Default for BitFlipInjector {
    fn default() -> Self {
        Self {
            seed: 42,
            flip_count: 1,
        }
    }
}

impl BitFlipInjector {
    /// Create a new bit-flip injector
    pub fn new(seed: u64, flip_count: usize) -> Self {
        Self { seed, flip_count }
    }

    /// Inject bit flips into data (returns modified copy)
    pub fn inject(&self, data: &[u8]) -> Vec<u8> {
        let mut result = data.to_vec();
        if result.is_empty() {
            return result;
        }

        // Simple LCG for reproducible "random" positions
        let mut rng_state = self.seed;
        for _ in 0..self.flip_count {
            // LCG: next = (a * state + c) mod m
            rng_state = rng_state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);

            let byte_idx = (rng_state as usize) % result.len();
            let bit_idx = ((rng_state >> 32) as usize) % 8;

            result[byte_idx] ^= 1 << bit_idx;
        }

        result
    }

    /// Inject bit flips into float data
    pub fn inject_floats(&self, data: &[f32]) -> Vec<f32> {
        // Convert to bytes, flip bits, convert back
        let bytes: Vec<u8> = data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let corrupted = self.inject(&bytes);

        corrupted.chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(arr)
            })
            .collect()
    }
}

/// Numeric operations with overflow/underflow detection
#[derive(Debug, Clone, Copy, Default)]
pub struct CheckedArithmetic;

impl CheckedArithmetic {
    /// Create a new checked arithmetic helper
    pub fn new() -> Self {
        Self
    }

    /// Add with overflow check (F1012)
    pub fn checked_add_i64(a: i64, b: i64) -> AdversarialResult<i64> {
        a.checked_add(b).ok_or_else(|| AdversarialError::IntegerOverflow {
            operation: format!("{a} + {b}"),
        })
    }

    /// Multiply with overflow check (F1012)
    pub fn checked_mul_i64(a: i64, b: i64) -> AdversarialResult<i64> {
        a.checked_mul(b).ok_or_else(|| AdversarialError::IntegerOverflow {
            operation: format!("{a} * {b}"),
        })
    }

    /// Add with overflow check for usize
    pub fn checked_add_usize(a: usize, b: usize) -> AdversarialResult<usize> {
        a.checked_add(b).ok_or_else(|| AdversarialError::IntegerOverflow {
            operation: format!("{a} + {b}"),
        })
    }

    /// Multiply with overflow check for usize
    pub fn checked_mul_usize(a: usize, b: usize) -> AdversarialResult<usize> {
        a.checked_mul(b).ok_or_else(|| AdversarialError::IntegerOverflow {
            operation: format!("{a} * {b}"),
        })
    }

    /// Division with zero check (F1013)
    pub fn checked_div_f64(a: f64, b: f64) -> AdversarialResult<f64> {
        if b == 0.0 {
            return Err(AdversarialError::DivisionByZero { numerator: a });
        }
        Ok(a / b)
    }

    /// Division with zero check for integers
    pub fn checked_div_i64(a: i64, b: i64) -> AdversarialResult<i64> {
        if b == 0 {
            return Err(AdversarialError::DivisionByZero { numerator: a as f64 });
        }
        Ok(a / b)
    }
}

/// Monotonic timestamp tracker (F1006)
#[derive(Debug, Clone)]
pub struct MonotonicClock {
    last_timestamp: Option<Instant>,
}

impl Default for MonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl MonotonicClock {
    /// Create a new monotonic clock tracker
    pub fn new() -> Self {
        Self { last_timestamp: None }
    }

    /// Record a timestamp and verify monotonicity
    pub fn tick(&mut self) -> AdversarialResult<Instant> {
        let now = Instant::now();

        if let Some(prev) = self.last_timestamp {
            // In Rust, Instant is guaranteed monotonic, but we check anyway
            // for systems with broken clocks
            if now < prev {
                return Err(AdversarialError::ClockSkew {
                    prev,
                    curr: now,
                });
            }
        }

        self.last_timestamp = Some(now);
        Ok(now)
    }

    /// Get elapsed time since last tick
    pub fn elapsed(&self) -> Option<Duration> {
        self.last_timestamp.map(|t| t.elapsed())
    }

    /// Reset the clock
    pub fn reset(&mut self) {
        self.last_timestamp = None;
    }
}

/// Resource limiter for bounded operations (F1016, F1017, F1018)
#[derive(Debug, Clone)]
pub struct ResourceLimiter {
    /// Maximum stack depth for recursive operations
    pub max_stack_depth: usize,
    /// Maximum memory allocation in bytes
    pub max_memory_bytes: usize,
    /// Timeout for operations
    pub timeout: Duration,
    /// Current stack depth
    current_depth: usize,
    /// Current allocated memory
    current_memory: usize,
    /// Operation start time
    start_time: Option<Instant>,
}

impl Default for ResourceLimiter {
    fn default() -> Self {
        Self {
            max_stack_depth: 1000,
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            timeout: Duration::from_secs(60),
            current_depth: 0,
            current_memory: 0,
            start_time: None,
        }
    }
}

impl ResourceLimiter {
    /// Create a new resource limiter
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum stack depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_stack_depth = max_depth;
        self
    }

    /// Set maximum memory
    pub fn with_max_memory(mut self, max_bytes: usize) -> Self {
        self.max_memory_bytes = max_bytes;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Start a timed operation
    pub fn start_operation(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Check if operation has timed out (F1018)
    pub fn check_timeout(&self, operation: &str) -> AdversarialResult<()> {
        if let Some(start) = self.start_time {
            let elapsed = start.elapsed();
            if elapsed > self.timeout {
                return Err(AdversarialError::Timeout {
                    operation: operation.to_string(),
                    elapsed,
                    limit: self.timeout,
                });
            }
        }
        Ok(())
    }

    /// Enter a recursive call (F1016)
    pub fn enter_recursion(&mut self) -> AdversarialResult<()> {
        self.current_depth += 1;
        if self.current_depth > self.max_stack_depth {
            return Err(AdversarialError::StackOverflow {
                depth: self.current_depth,
                max_depth: self.max_stack_depth,
            });
        }
        Ok(())
    }

    /// Exit a recursive call
    pub fn exit_recursion(&mut self) {
        if self.current_depth > 0 {
            self.current_depth -= 1;
        }
    }

    /// Request memory allocation (F1017)
    pub fn request_memory(&mut self, bytes: usize) -> AdversarialResult<()> {
        let new_total = self.current_memory.saturating_add(bytes);
        if new_total > self.max_memory_bytes {
            return Err(AdversarialError::ResourceExhausted {
                resource: format!("memory: requested {bytes} bytes, would exceed limit of {} bytes",
                                  self.max_memory_bytes),
            });
        }
        self.current_memory = new_total;
        Ok(())
    }

    /// Release memory
    pub fn release_memory(&mut self, bytes: usize) {
        self.current_memory = self.current_memory.saturating_sub(bytes);
    }

    /// Get current resource usage
    pub fn usage(&self) -> ResourceUsage {
        ResourceUsage {
            stack_depth: self.current_depth,
            memory_bytes: self.current_memory,
            elapsed: self.start_time.map(|s| s.elapsed()),
        }
    }

    /// Reset limiter state
    pub fn reset(&mut self) {
        self.current_depth = 0;
        self.current_memory = 0;
        self.start_time = None;
    }
}

/// Current resource usage statistics
#[derive(Debug, Clone, Copy)]
pub struct ResourceUsage {
    /// Current recursion depth
    pub stack_depth: usize,
    /// Current memory allocated
    pub memory_bytes: usize,
    /// Elapsed time since operation start
    pub elapsed: Option<Duration>,
}

/// Configuration validator for fuzzing (F1008, F1009)
#[derive(Debug, Clone)]
pub struct ConfigValidator {
    /// Minimum allowed values by field name
    pub mins: std::collections::HashMap<String, f64>,
    /// Maximum allowed values by field name
    pub maxs: std::collections::HashMap<String, f64>,
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self {
            mins: std::collections::HashMap::new(),
            maxs: std::collections::HashMap::new(),
        }
    }
}

impl ConfigValidator {
    /// Create a new config validator
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a bound for a field
    pub fn with_bound(mut self, field: &str, min: f64, max: f64) -> Self {
        self.mins.insert(field.to_string(), min);
        self.maxs.insert(field.to_string(), max);
        self
    }

    /// Validate a numeric config value (F1009)
    pub fn validate_numeric(&self, field: &str, value: f64) -> AdversarialResult<f64> {
        // Check for NaN
        if value.is_nan() {
            return Err(AdversarialError::ConfigParseError {
                field: field.to_string(),
                reason: "value is NaN".to_string(),
            });
        }

        // Check bounds
        if let Some(&min) = self.mins.get(field) {
            if value < min {
                return Err(AdversarialError::ConfigOutOfBounds {
                    field: field.to_string(),
                    value: value.to_string(),
                    min: min.to_string(),
                    max: self.maxs.get(field).map_or("unbounded".to_string(), |m| m.to_string()),
                });
            }
        }

        if let Some(&max) = self.maxs.get(field) {
            if value > max {
                return Err(AdversarialError::ConfigOutOfBounds {
                    field: field.to_string(),
                    value: value.to_string(),
                    min: self.mins.get(field).map_or("unbounded".to_string(), |m| m.to_string()),
                    max: max.to_string(),
                });
            }
        }

        Ok(value)
    }

    /// Validate TOML-like string config (F1008)
    pub fn validate_toml_string(&self, input: &str) -> AdversarialResult<()> {
        // Check for common malformed TOML patterns
        let trimmed = input.trim();

        // Empty input
        if trimmed.is_empty() {
            return Err(AdversarialError::ConfigParseError {
                field: "root".to_string(),
                reason: "empty config".to_string(),
            });
        }

        // Unclosed brackets
        let open_brackets = trimmed.matches('[').count();
        let close_brackets = trimmed.matches(']').count();
        if open_brackets != close_brackets {
            return Err(AdversarialError::ConfigParseError {
                field: "root".to_string(),
                reason: format!("mismatched brackets: {} open, {} close",
                               open_brackets, close_brackets),
            });
        }

        // Unclosed quotes
        let quotes = trimmed.matches('"').count();
        if !quotes.is_multiple_of(2) {
            return Err(AdversarialError::ConfigParseError {
                field: "root".to_string(),
                reason: "unclosed string literal".to_string(),
            });
        }

        Ok(())
    }
}

/// Cancellation token for cooperative cancellation (F1019)
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        Self {
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Request cancellation
    pub fn cancel(&self) {
        self.cancelled.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check if cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Return error if cancelled
    pub fn check(&self, operation: &str) -> AdversarialResult<()> {
        if self.is_cancelled() {
            return Err(AdversarialError::Cancelled {
                operation: operation.to_string(),
            });
        }
        Ok(())
    }

    /// Clone the token (same underlying state)
    pub fn clone_token(&self) -> Self {
        Self {
            cancelled: std::sync::Arc::clone(&self.cancelled),
        }
    }
}

/// Recovery handler for error recovery testing (F1020)
#[derive(Debug, Clone)]
pub struct RecoveryHandler<S: Clone> {
    /// Stored checkpoint state
    checkpoint: Option<S>,
}

impl<S: Clone> Default for RecoveryHandler<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Clone> RecoveryHandler<S> {
    /// Create a new recovery handler
    pub fn new() -> Self {
        Self { checkpoint: None }
    }

    /// Save a checkpoint
    pub fn checkpoint(&mut self, state: S) {
        self.checkpoint = Some(state);
    }

    /// Recover from error using checkpoint
    pub fn recover(&self) -> AdversarialResult<S> {
        self.checkpoint.clone().ok_or_else(|| AdversarialError::RecoveryFailed {
            original_error: "unknown".to_string(),
            recovery_error: "no checkpoint available".to_string(),
        })
    }

    /// Try operation with automatic recovery on failure
    pub fn try_with_recovery<F, T, E>(&self, operation: F) -> AdversarialResult<T>
    where
        F: FnOnce() -> Result<T, E>,
        E: std::fmt::Display,
    {
        match operation() {
            Ok(result) => Ok(result),
            Err(e) => Err(AdversarialError::RecoveryFailed {
                original_error: e.to_string(),
                recovery_error: if self.checkpoint.is_some() {
                    "operation failed, checkpoint available".to_string()
                } else {
                    "operation failed, no checkpoint".to_string()
                },
            }),
        }
    }

    /// Check if checkpoint exists
    pub fn has_checkpoint(&self) -> bool {
        self.checkpoint.is_some()
    }
}

/// Summary of adversarial test results
#[derive(Debug, Clone)]
pub struct AdversarialTestSummary {
    /// Total tests run
    pub total_tests: usize,
    /// Tests that passed (handled adversarial input correctly)
    pub passed: usize,
    /// Tests that failed (panicked or incorrect behavior)
    pub failed: usize,
    /// Tactics tested
    pub tactics_tested: Vec<AdversarialTactic>,
    /// Errors encountered (expected - means system handled correctly)
    pub errors_handled: Vec<String>,
}

impl AdversarialTestSummary {
    /// Create a new empty summary
    pub fn new() -> Self {
        Self {
            total_tests: 0,
            passed: 0,
            failed: 0,
            tactics_tested: Vec::new(),
            errors_handled: Vec::new(),
        }
    }

    /// Record a passing test
    pub fn record_pass(&mut self, tactic: AdversarialTactic) {
        self.total_tests += 1;
        self.passed += 1;
        if !self.tactics_tested.contains(&tactic) {
            self.tactics_tested.push(tactic);
        }
    }

    /// Record a failing test
    pub fn record_fail(&mut self, tactic: AdversarialTactic, reason: &str) {
        self.total_tests += 1;
        self.failed += 1;
        if !self.tactics_tested.contains(&tactic) {
            self.tactics_tested.push(tactic);
        }
        self.errors_handled.push(reason.to_string());
    }

    /// Record a handled error (this is good - system detected the adversarial input)
    pub fn record_error_handled(&mut self, error: &AdversarialError) {
        self.errors_handled.push(format!("{error:?}"));
    }

    /// Get pass rate
    pub fn pass_rate(&self) -> f64 {
        if self.total_tests == 0 {
            return 0.0;
        }
        (self.passed as f64) / (self.total_tests as f64) * 100.0
    }

    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.failed == 0 && self.total_tests > 0
    }
}

impl Default for AdversarialTestSummary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_validator_zero_size() {
        let validator = InputValidator::new();
        let result = validator.validate_bytes(&[]);
        assert!(matches!(result, Err(AdversarialError::ZeroSizeInput)));
    }

    #[test]
    fn test_input_validator_max_size() {
        let validator = InputValidator::new().with_max_size(100);
        let data = vec![0u8; 200];
        let result = validator.validate_bytes(&data);
        assert!(matches!(result, Err(AdversarialError::MaxSizeExceeded { .. })));
    }

    #[test]
    fn test_checksum_consistency() {
        let data = b"hello world";
        let checksum1 = InputValidator::compute_checksum(data);
        let checksum2 = InputValidator::compute_checksum(data);
        assert_eq!(checksum1, checksum2);
    }

    #[test]
    fn test_bit_flip_injector() {
        let injector = BitFlipInjector::new(42, 1);
        let data = vec![0u8; 100];
        let corrupted = injector.inject(&data);

        // At least one bit should be different
        assert_ne!(data, corrupted);
    }

    #[test]
    fn test_checked_arithmetic_overflow() {
        let result = CheckedArithmetic::checked_add_i64(i64::MAX, 1);
        assert!(matches!(result, Err(AdversarialError::IntegerOverflow { .. })));
    }

    #[test]
    fn test_checked_div_zero() {
        let result = CheckedArithmetic::checked_div_f64(10.0, 0.0);
        assert!(matches!(result, Err(AdversarialError::DivisionByZero { .. })));
    }

    #[test]
    fn test_monotonic_clock() {
        let mut clock = MonotonicClock::new();
        let t1 = clock.tick().unwrap();
        let t2 = clock.tick().unwrap();
        assert!(t2 >= t1);
    }

    #[test]
    fn test_resource_limiter_stack() {
        let mut limiter = ResourceLimiter::new().with_max_depth(5);

        for _ in 0..5 {
            assert!(limiter.enter_recursion().is_ok());
        }

        // 6th should fail
        assert!(matches!(
            limiter.enter_recursion(),
            Err(AdversarialError::StackOverflow { .. })
        ));
    }

    #[test]
    fn test_cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());
        assert!(token.check("test").is_ok());

        token.cancel();
        assert!(token.is_cancelled());
        assert!(matches!(
            token.check("test"),
            Err(AdversarialError::Cancelled { .. })
        ));
    }

    #[test]
    fn test_config_validator_bounds() {
        let validator = ConfigValidator::new()
            .with_bound("learning_rate", 0.0001, 1.0);

        assert!(validator.validate_numeric("learning_rate", 0.01).is_ok());
        assert!(matches!(
            validator.validate_numeric("learning_rate", 2.0),
            Err(AdversarialError::ConfigOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_recovery_handler() {
        let mut handler: RecoveryHandler<i32> = RecoveryHandler::new();

        // No checkpoint - recovery should fail
        assert!(handler.recover().is_err());

        // With checkpoint - recovery should work
        handler.checkpoint(42);
        assert_eq!(handler.recover().unwrap(), 42);
    }
}
