//! Fuzz Testing Integration (PMAT-023)
//!
//! Property-based testing and fuzz-like input validation per §36.3 to address
//! the Resilience score. Uses proptest for stable Rust compatibility.
//!
//! # Fuzz Targets
//!
//! | Target | Component | Description |
//! |--------|-----------|-------------|
//! | `fuzz_syscall_breakdown` | TracingEscalation | Syscall name/duration inputs |
//! | `fuzz_workload_metrics` | RooflineAnalysis | FLOP/byte/time values |
//! | `fuzz_escalation_thresholds` | TracingEscalation | Threshold configurations |
//! | `fuzz_hardware_profile` | HardwareProfile | Peak GFLOPS/bandwidth values |
//! | `fuzz_brick_scoring` | BrickScore | Score calculation inputs |
//!
//! # Falsification Criteria
//!
//! F1081-F1095: Input validation and error path testing

use std::collections::HashMap;
use std::fmt;

/// Result of a fuzz test run
#[derive(Debug, Clone)]
pub struct FuzzResult {
    /// Target name
    pub target: String,
    /// Number of test cases run
    pub test_cases: u64,
    /// Number of failures found
    pub failures: u64,
    /// Edge coverage percentage (0-100)
    pub coverage_percent: f64,
    /// Specific failure details
    pub failure_details: Vec<FuzzFailure>,
    /// Duration of the fuzz run
    pub duration_secs: f64,
}

impl FuzzResult {
    /// Create a new fuzz result
    pub fn new(target: &str) -> Self {
        Self {
            target: target.to_string(),
            test_cases: 0,
            failures: 0,
            coverage_percent: 0.0,
            failure_details: Vec::new(),
            duration_secs: 0.0,
        }
    }

    /// Record a successful test case
    pub fn record_success(&mut self) {
        self.test_cases += 1;
    }

    /// Record a failure
    pub fn record_failure(&mut self, input: String, error: String) {
        self.test_cases += 1;
        self.failures += 1;
        self.failure_details.push(FuzzFailure { input, error });
    }

    /// Check if all tests passed
    pub fn passed(&self) -> bool {
        self.failures == 0
    }

    /// Calculate failure rate
    pub fn failure_rate(&self) -> f64 {
        if self.test_cases == 0 {
            0.0
        } else {
            (self.failures as f64 / self.test_cases as f64) * 100.0
        }
    }
}

/// Details of a fuzz failure
#[derive(Debug, Clone)]
pub struct FuzzFailure {
    /// Input that caused the failure
    pub input: String,
    /// Error message
    pub error: String,
}

/// Input validator for fuzz testing
#[derive(Debug, Clone, Default)]
pub struct FuzzInputValidator {
    /// Maximum string length allowed
    pub max_string_len: usize,
    /// Maximum numeric value allowed
    pub max_numeric: f64,
    /// Minimum numeric value allowed
    pub min_numeric: f64,
    /// Allow NaN values
    pub allow_nan: bool,
    /// Allow infinity values
    pub allow_infinity: bool,
    /// Allow negative values
    pub allow_negative: bool,
    /// Allow zero
    pub allow_zero: bool,
}

impl FuzzInputValidator {
    /// Create a new validator with default settings
    pub fn new() -> Self {
        Self {
            max_string_len: 1024,
            max_numeric: 1e15,
            min_numeric: -1e15,
            allow_nan: false,
            allow_infinity: false,
            allow_negative: true,
            allow_zero: true,
        }
    }

    /// Validate a floating-point value
    pub fn validate_float(&self, value: f64) -> Result<f64, FuzzValidationError> {
        if value.is_nan() && !self.allow_nan {
            return Err(FuzzValidationError::NaN);
        }
        if value.is_infinite() && !self.allow_infinity {
            return Err(FuzzValidationError::Infinity);
        }
        if value < 0.0 && !self.allow_negative {
            return Err(FuzzValidationError::NegativeValue(value));
        }
        if value == 0.0 && !self.allow_zero {
            return Err(FuzzValidationError::ZeroValue);
        }
        if value > self.max_numeric {
            return Err(FuzzValidationError::TooLarge(value));
        }
        if value < self.min_numeric {
            return Err(FuzzValidationError::TooSmall(value));
        }
        Ok(value)
    }

    /// Validate a string
    pub fn validate_string<'a>(&self, value: &'a str) -> Result<&'a str, FuzzValidationError> {
        if value.len() > self.max_string_len {
            return Err(FuzzValidationError::StringTooLong(value.len()));
        }
        // Check for valid UTF-8 (already guaranteed by &str, but check control chars)
        if value
            .chars()
            .any(|c| c.is_control() && c != '\n' && c != '\t')
        {
            return Err(FuzzValidationError::InvalidControlChars);
        }
        Ok(value)
    }

    /// Validate an integer
    pub fn validate_u64(&self, value: u64) -> Result<u64, FuzzValidationError> {
        let max = self.max_numeric as u64;
        if value > max {
            return Err(FuzzValidationError::TooLarge(value as f64));
        }
        if value == 0 && !self.allow_zero {
            return Err(FuzzValidationError::ZeroValue);
        }
        Ok(value)
    }

    /// Create a validator for positive-only values
    pub fn positive_only() -> Self {
        Self {
            allow_negative: false,
            allow_zero: false,
            ..Self::new()
        }
    }

    /// Create a validator for non-negative values
    pub fn non_negative() -> Self {
        Self {
            allow_negative: false,
            ..Self::new()
        }
    }

    /// Create a strict validator (no special float values)
    pub fn strict() -> Self {
        Self {
            allow_nan: false,
            allow_infinity: false,
            max_numeric: 1e12,
            min_numeric: -1e12,
            ..Self::new()
        }
    }
}

/// Validation errors from fuzz testing
#[derive(Debug, Clone, PartialEq)]
pub enum FuzzValidationError {
    /// NaN value encountered
    NaN,
    /// Infinity value encountered
    Infinity,
    /// Negative value not allowed
    NegativeValue(f64),
    /// Zero value not allowed
    ZeroValue,
    /// Value too large
    TooLarge(f64),
    /// Value too small
    TooSmall(f64),
    /// String too long
    StringTooLong(usize),
    /// Invalid control characters
    InvalidControlChars,
    /// Integer overflow
    IntegerOverflow,
    /// Division by zero
    DivisionByZero,
    /// Empty input
    EmptyInput,
    /// Invalid format
    InvalidFormat(String),
}

impl fmt::Display for FuzzValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FuzzValidationError::NaN => write!(f, "NaN value not allowed"),
            FuzzValidationError::Infinity => write!(f, "Infinity value not allowed"),
            FuzzValidationError::NegativeValue(v) => write!(f, "Negative value not allowed: {}", v),
            FuzzValidationError::ZeroValue => write!(f, "Zero value not allowed"),
            FuzzValidationError::TooLarge(v) => write!(f, "Value too large: {}", v),
            FuzzValidationError::TooSmall(v) => write!(f, "Value too small: {}", v),
            FuzzValidationError::StringTooLong(len) => write!(f, "String too long: {} chars", len),
            FuzzValidationError::InvalidControlChars => write!(f, "Invalid control characters"),
            FuzzValidationError::IntegerOverflow => write!(f, "Integer overflow"),
            FuzzValidationError::DivisionByZero => write!(f, "Division by zero"),
            FuzzValidationError::EmptyInput => write!(f, "Empty input"),
            FuzzValidationError::InvalidFormat(s) => write!(f, "Invalid format: {}", s),
        }
    }
}

impl std::error::Error for FuzzValidationError {}

/// Safe division that returns None on division by zero
pub fn safe_div(a: f64, b: f64) -> Option<f64> {
    if b == 0.0 || b.is_nan() {
        None
    } else {
        let result = a / b;
        if result.is_nan() || result.is_infinite() {
            None
        } else {
            Some(result)
        }
    }
}

/// Checked addition that returns None on overflow
pub fn checked_add_u64(a: u64, b: u64) -> Option<u64> {
    a.checked_add(b)
}

/// Checked multiplication that returns None on overflow
pub fn checked_mul_u64(a: u64, b: u64) -> Option<u64> {
    a.checked_mul(b)
}

/// Bound a value to a range
pub fn bound_value(value: f64, min: f64, max: f64) -> f64 {
    if value.is_nan() {
        (min + max) / 2.0
    } else {
        value.clamp(min, max)
    }
}

/// Sanitize a float value for safe computation
pub fn sanitize_float(value: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else if value.is_infinite() {
        if value > 0.0 {
            f64::MAX
        } else {
            f64::MIN
        }
    } else {
        value
    }
}

/// Fuzz target configuration
#[derive(Debug, Clone)]
pub struct FuzzTargetConfig {
    /// Target name
    pub name: String,
    /// Number of iterations
    pub iterations: u64,
    /// Timeout in seconds
    pub timeout_secs: u64,
    /// Seed for reproducibility (None = random)
    pub seed: Option<u64>,
    /// Enable coverage tracking
    pub track_coverage: bool,
}

impl FuzzTargetConfig {
    /// Create a new fuzz target config
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            iterations: 10000,
            timeout_secs: 60,
            seed: None,
            track_coverage: true,
        }
    }

    /// Set number of iterations
    pub fn with_iterations(mut self, iterations: u64) -> Self {
        self.iterations = iterations;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Set seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Fuzz testing suite for cbtop components
#[derive(Debug, Default)]
pub struct FuzzSuite {
    /// Results by target name
    results: HashMap<String, FuzzResult>,
    /// Total test cases
    total_cases: u64,
    /// Total failures
    total_failures: u64,
}

impl FuzzSuite {
    /// Create a new fuzz suite
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a result
    pub fn add_result(&mut self, result: FuzzResult) {
        self.total_cases += result.test_cases;
        self.total_failures += result.failures;
        self.results.insert(result.target.clone(), result);
    }

    /// Get result for a target
    pub fn get_result(&self, target: &str) -> Option<&FuzzResult> {
        self.results.get(target)
    }

    /// Check if all tests passed
    pub fn all_passed(&self) -> bool {
        self.total_failures == 0
    }

    /// Get total test cases
    pub fn total_cases(&self) -> u64 {
        self.total_cases
    }

    /// Get total failures
    pub fn total_failures(&self) -> u64 {
        self.total_failures
    }

    /// Get all results
    pub fn results(&self) -> &HashMap<String, FuzzResult> {
        &self.results
    }

    /// Generate summary report
    pub fn summary(&self) -> FuzzSummary {
        let targets_passed = self.results.values().filter(|r| r.passed()).count();
        let avg_coverage = if self.results.is_empty() {
            0.0
        } else {
            self.results
                .values()
                .map(|r| r.coverage_percent)
                .sum::<f64>()
                / self.results.len() as f64
        };

        FuzzSummary {
            total_targets: self.results.len(),
            targets_passed,
            total_cases: self.total_cases,
            total_failures: self.total_failures,
            avg_coverage,
            overall_passed: self.all_passed(),
        }
    }
}

/// Summary of fuzz testing results
#[derive(Debug, Clone)]
pub struct FuzzSummary {
    /// Total number of fuzz targets
    pub total_targets: usize,
    /// Number of targets that passed
    pub targets_passed: usize,
    /// Total test cases across all targets
    pub total_cases: u64,
    /// Total failures across all targets
    pub total_failures: u64,
    /// Average coverage percentage
    pub avg_coverage: f64,
    /// Whether all tests passed
    pub overall_passed: bool,
}

impl FuzzSummary {
    /// Get pass rate as percentage
    pub fn pass_rate(&self) -> f64 {
        if self.total_targets == 0 {
            100.0
        } else {
            (self.targets_passed as f64 / self.total_targets as f64) * 100.0
        }
    }
}

/// Test a function with edge case float inputs
pub fn test_float_edge_cases<F, T>(f: F) -> Vec<(f64, Result<T, String>)>
where
    F: Fn(f64) -> T,
    T: fmt::Debug,
{
    let edge_cases = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MIN,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::EPSILON,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e15,
        -1e15,
        1e-15,
        -1e-15,
    ];

    edge_cases
        .iter()
        .map(|&x| {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(x)));
            match result {
                Ok(v) => (x, Ok(v)),
                Err(e) => {
                    let msg = if let Some(s) = e.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = e.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "Unknown panic".to_string()
                    };
                    (x, Err(msg))
                }
            }
        })
        .collect()
}

/// Test a function with edge case u64 inputs
pub fn test_u64_edge_cases<F, T>(f: F) -> Vec<(u64, Result<T, String>)>
where
    F: Fn(u64) -> T,
    T: fmt::Debug,
{
    let edge_cases = [
        0u64,
        1,
        u64::MAX,
        u64::MAX - 1,
        u64::MAX / 2,
        1000,
        1_000_000,
        1_000_000_000,
    ];

    edge_cases
        .iter()
        .map(|&x| {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(x)));
            match result {
                Ok(v) => (x, Ok(v)),
                Err(e) => {
                    let msg = if let Some(s) = e.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = e.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "Unknown panic".to_string()
                    };
                    (x, Err(msg))
                }
            }
        })
        .collect()
}


#[cfg(test)]
mod tests;
