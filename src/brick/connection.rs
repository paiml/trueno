//! Connection TTL and Health Check
//!
//! AWP-06: Managed connections with TTL, idle timeout, and health tracking.

use std::time::{Duration, Instant};

// ----------------------------------------------------------------------------
// AWP-06: Connection TTL + Health Check
// ----------------------------------------------------------------------------

/// Connection with TTL and health tracking.
#[derive(Debug)]
pub struct ManagedConnection<T> {
    /// The underlying connection
    inner: T,
    /// When the connection was created
    created_at: Instant,
    /// When the connection was last used
    last_used: Instant,
    /// Maximum lifetime (TTL)
    max_lifetime: Duration,
    /// Maximum idle time
    max_idle: Duration,
    /// Health check failures
    health_failures: usize,
}

impl<T> ManagedConnection<T> {
    /// Create a new managed connection.
    pub fn new(inner: T, max_lifetime: Duration, max_idle: Duration) -> Self {
        let now = Instant::now();
        Self {
            inner,
            created_at: now,
            last_used: now,
            max_lifetime,
            max_idle,
            health_failures: 0,
        }
    }

    /// Check if the connection is still valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let now = Instant::now();
        let not_expired = now.duration_since(self.created_at) < self.max_lifetime;
        let not_idle = now.duration_since(self.last_used) < self.max_idle;
        let healthy = self.health_failures < 3;
        not_expired && not_idle && healthy
    }

    /// Check if the connection has expired (TTL exceeded).
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() >= self.max_lifetime
    }

    /// Check if the connection is idle.
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.last_used.elapsed() >= self.max_idle
    }

    /// Mark the connection as used.
    pub fn touch(&mut self) {
        self.last_used = Instant::now();
    }

    /// Record a health check failure.
    pub fn record_health_failure(&mut self) {
        self.health_failures += 1;
    }

    /// Reset health failure count.
    pub fn reset_health(&mut self) {
        self.health_failures = 0;
    }

    /// Get the underlying connection.
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get mutable access to the underlying connection.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Consume and return the underlying connection.
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Get connection age.
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get idle time.
    #[must_use]
    pub fn idle_time(&self) -> Duration {
        self.last_used.elapsed()
    }

    /// Get health failure count (for testing/diagnostics).
    #[must_use]
    pub fn health_failures(&self) -> usize {
        self.health_failures
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_managed_connection_new() {
        let conn = ManagedConnection::new(
            "test_connection",
            Duration::from_secs(60),
            Duration::from_secs(10),
        );

        assert_eq!(conn.inner(), &"test_connection");
        assert!(conn.is_valid());
        assert!(!conn.is_expired());
        assert!(!conn.is_idle());
    }

    #[test]
    fn test_managed_connection_inner_mut() {
        let mut conn = ManagedConnection::new(
            vec![1, 2, 3],
            Duration::from_secs(60),
            Duration::from_secs(10),
        );

        conn.inner_mut().push(4);
        assert_eq!(conn.inner(), &vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_managed_connection_into_inner() {
        let conn = ManagedConnection::new(42u32, Duration::from_secs(60), Duration::from_secs(10));

        let value = conn.into_inner();
        assert_eq!(value, 42);
    }

    #[test]
    fn test_managed_connection_touch() {
        let mut conn = ManagedConnection::new(
            "test",
            Duration::from_secs(60),
            Duration::from_millis(10),
        );

        std::thread::sleep(Duration::from_millis(5));
        let idle_before = conn.idle_time();

        conn.touch();
        let idle_after = conn.idle_time();

        assert!(idle_after < idle_before);
    }

    #[test]
    fn test_managed_connection_health_failures() {
        let mut conn =
            ManagedConnection::new("test", Duration::from_secs(60), Duration::from_secs(10));

        assert!(conn.is_valid());

        conn.record_health_failure();
        conn.record_health_failure();
        assert!(conn.is_valid()); // 2 failures, still valid

        conn.record_health_failure();
        assert!(!conn.is_valid()); // 3 failures, now invalid

        conn.reset_health();
        assert!(conn.is_valid()); // Reset, valid again
    }

    #[test]
    fn test_managed_connection_age() {
        let conn =
            ManagedConnection::new("test", Duration::from_secs(60), Duration::from_secs(10));

        let age = conn.age();
        assert!(age < Duration::from_millis(100));
    }

    #[test]
    fn test_managed_connection_ttl_expiry() {
        let conn =
            ManagedConnection::new("test", Duration::from_millis(10), Duration::from_secs(60));

        assert!(!conn.is_expired());

        std::thread::sleep(Duration::from_millis(15));

        assert!(conn.is_expired());
        assert!(!conn.is_valid());
    }

    #[test]
    fn test_managed_connection_idle_expiry() {
        let conn =
            ManagedConnection::new("test", Duration::from_secs(60), Duration::from_millis(10));

        assert!(!conn.is_idle());

        std::thread::sleep(Duration::from_millis(15));

        assert!(conn.is_idle());
        assert!(!conn.is_valid());
    }

    /// FALSIFICATION TEST: Verify all three conditions must pass for validity
    ///
    /// A connection is valid only if: not_expired AND not_idle AND healthy.
    /// If any one fails, the connection should be invalid.
    #[test]
    fn test_falsify_validity_requires_all_conditions() {
        // Test 1: Healthy and not idle, but expired
        let expired = ManagedConnection::new(
            "test",
            Duration::from_millis(5),
            Duration::from_secs(60),
        );
        std::thread::sleep(Duration::from_millis(10));
        assert!(
            !expired.is_valid(),
            "FALSIFICATION FAILED: Expired connection should be invalid"
        );

        // Test 2: Healthy and not expired, but idle
        let idle = ManagedConnection::new(
            "test",
            Duration::from_secs(60),
            Duration::from_millis(5),
        );
        std::thread::sleep(Duration::from_millis(10));
        assert!(
            !idle.is_valid(),
            "FALSIFICATION FAILED: Idle connection should be invalid"
        );

        // Test 3: Not expired and not idle, but unhealthy
        let mut unhealthy =
            ManagedConnection::new("test", Duration::from_secs(60), Duration::from_secs(60));
        unhealthy.record_health_failure();
        unhealthy.record_health_failure();
        unhealthy.record_health_failure();
        assert!(
            !unhealthy.is_valid(),
            "FALSIFICATION FAILED: Unhealthy connection should be invalid"
        );

        // Test 4: All conditions pass
        let valid =
            ManagedConnection::new("test", Duration::from_secs(60), Duration::from_secs(60));
        assert!(
            valid.is_valid(),
            "FALSIFICATION FAILED: Fresh connection should be valid"
        );
    }

    /// FALSIFICATION TEST: Touch must reset idle timer
    #[test]
    fn test_falsify_touch_resets_idle() {
        let mut conn =
            ManagedConnection::new("test", Duration::from_secs(60), Duration::from_millis(20));

        // Wait until almost idle
        std::thread::sleep(Duration::from_millis(15));
        assert!(
            !conn.is_idle(),
            "Should not be idle yet at 15ms with 20ms timeout"
        );

        // Touch to reset
        conn.touch();

        // Wait another 15ms (would be 30ms total without touch, but only 15ms since touch)
        std::thread::sleep(Duration::from_millis(15));
        assert!(
            !conn.is_idle(),
            "FALSIFICATION FAILED: Touch should have reset idle timer"
        );

        // Now wait until actually idle
        std::thread::sleep(Duration::from_millis(10));
        assert!(conn.is_idle(), "Should be idle now (25ms since touch)");
    }
}
