//! Remote SSH/Headless Agent Integration (PMAT-044)
//!
//! Distributed benchmark collection from remote hosts via SSH.
//!
//! # Design
//!
//! - SSH-based command execution with connection pooling
//! - Multi-host result aggregation with statistical merging
//! - Host health monitoring and automatic failover
//! - Secure credential handling (no plaintext storage)
//!
//! # Falsification (FKR-045)
//!
//! H₀: Remote agent cannot collect metrics from heterogeneous hosts
//! Test: Connect to 3+ hosts with different architectures, verify metric aggregation

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Result type for remote agent operations
pub type RemoteResult<T> = Result<T, RemoteError>;

/// Errors that can occur during remote operations
#[derive(Debug, Clone, PartialEq)]
pub enum RemoteError {
    /// SSH connection failed
    ConnectionFailed { host: String, reason: String },
    /// Authentication failed
    AuthenticationFailed { host: String },
    /// Command execution failed
    CommandFailed { host: String, exit_code: i32, stderr: String },
    /// Timeout waiting for response
    Timeout { host: String, timeout_ms: u64 },
    /// Host not found in pool
    HostNotFound { host: String },
    /// All hosts failed
    AllHostsFailed { failures: Vec<String> },
    /// Invalid configuration
    InvalidConfig { reason: String },
    /// Result aggregation failed
    AggregationFailed { reason: String },
}

impl std::fmt::Display for RemoteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed { host, reason } => {
                write!(f, "Connection to {} failed: {}", host, reason)
            }
            Self::AuthenticationFailed { host } => {
                write!(f, "Authentication failed for {}", host)
            }
            Self::CommandFailed { host, exit_code, stderr } => {
                write!(f, "Command failed on {} (exit {}): {}", host, exit_code, stderr)
            }
            Self::Timeout { host, timeout_ms } => {
                write!(f, "Timeout after {}ms waiting for {}", timeout_ms, host)
            }
            Self::HostNotFound { host } => {
                write!(f, "Host {} not found in pool", host)
            }
            Self::AllHostsFailed { failures } => {
                write!(f, "All hosts failed: {:?}", failures)
            }
            Self::InvalidConfig { reason } => {
                write!(f, "Invalid configuration: {}", reason)
            }
            Self::AggregationFailed { reason } => {
                write!(f, "Result aggregation failed: {}", reason)
            }
        }
    }
}

impl std::error::Error for RemoteError {}

/// SSH authentication method
#[derive(Debug, Clone)]
pub enum AuthMethod {
    /// SSH key-based authentication
    Key {
        /// Path to private key file
        key_path: String,
        /// Optional passphrase (from environment or keyring, never stored)
        passphrase_env: Option<String>,
    },
    /// SSH agent forwarding
    Agent,
    /// Password from environment variable (never stored in config)
    PasswordEnv {
        /// Environment variable name containing password
        env_var: String,
    },
}

impl Default for AuthMethod {
    fn default() -> Self {
        Self::Agent
    }
}

/// Remote host configuration
#[derive(Debug, Clone)]
pub struct HostConfig {
    /// Hostname or IP address
    pub host: String,
    /// SSH port (default: 22)
    pub port: u16,
    /// Username for SSH connection
    pub username: String,
    /// Authentication method
    pub auth: AuthMethod,
    /// Connection timeout in milliseconds
    pub connect_timeout_ms: u64,
    /// Command execution timeout in milliseconds
    pub command_timeout_ms: u64,
    /// Host architecture (x86_64, aarch64, etc.)
    pub architecture: Option<String>,
    /// Host labels for grouping
    pub labels: HashMap<String, String>,
}

impl HostConfig {
    /// Create a new host configuration
    pub fn new(host: impl Into<String>, username: impl Into<String>) -> Self {
        Self {
            host: host.into(),
            port: 22,
            username: username.into(),
            auth: AuthMethod::default(),
            connect_timeout_ms: 10_000,
            command_timeout_ms: 60_000,
            architecture: None,
            labels: HashMap::new(),
        }
    }

    /// Set SSH port
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set authentication method
    pub fn with_auth(mut self, auth: AuthMethod) -> Self {
        self.auth = auth;
        self
    }

    /// Set connection timeout
    pub fn with_connect_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.connect_timeout_ms = timeout_ms;
        self
    }

    /// Set command timeout
    pub fn with_command_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.command_timeout_ms = timeout_ms;
        self
    }

    /// Set architecture
    pub fn with_architecture(mut self, arch: impl Into<String>) -> Self {
        self.architecture = Some(arch.into());
        self
    }

    /// Add a label
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }
}

/// Health status of a remote host
#[derive(Debug, Clone, PartialEq)]
pub enum HostHealth {
    /// Host is healthy and responding
    Healthy,
    /// Host is degraded (slow responses)
    Degraded { latency_ms: u64 },
    /// Host is unreachable
    Unreachable { last_error: String },
    /// Host status unknown (never checked)
    Unknown,
}

impl Default for HostHealth {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Remote host state
#[derive(Debug, Clone)]
pub struct HostState {
    /// Host configuration
    pub config: HostConfig,
    /// Current health status
    pub health: HostHealth,
    /// Last successful connection time
    pub last_success: Option<Instant>,
    /// Consecutive failure count
    pub failure_count: u32,
    /// Total commands executed
    pub commands_executed: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
}

impl HostState {
    /// Create new host state from configuration
    pub fn new(config: HostConfig) -> Self {
        Self {
            config,
            health: HostHealth::Unknown,
            last_success: None,
            failure_count: 0,
            commands_executed: 0,
            avg_latency_ms: 0.0,
        }
    }

    /// Check if host is available for commands
    pub fn is_available(&self) -> bool {
        matches!(self.health, HostHealth::Healthy | HostHealth::Degraded { .. } | HostHealth::Unknown)
    }

    /// Record a successful command execution
    pub fn record_success(&mut self, latency_ms: u64) {
        self.last_success = Some(Instant::now());
        self.failure_count = 0;
        self.commands_executed += 1;

        // Update running average
        let n = self.commands_executed as f64;
        self.avg_latency_ms = self.avg_latency_ms * ((n - 1.0) / n) + (latency_ms as f64) / n;

        // Update health based on latency
        self.health = if latency_ms > 5000 {
            HostHealth::Degraded { latency_ms }
        } else {
            HostHealth::Healthy
        };
    }

    /// Record a failed command
    pub fn record_failure(&mut self, error: &str) {
        self.failure_count += 1;
        if self.failure_count >= 3 {
            self.health = HostHealth::Unreachable {
                last_error: error.to_string(),
            };
        }
    }
}

/// Result from a single remote command
#[derive(Debug, Clone)]
pub struct CommandResult {
    /// Host that executed the command
    pub host: String,
    /// Exit code (0 = success)
    pub exit_code: i32,
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Execution time in milliseconds
    pub duration_ms: u64,
}

impl CommandResult {
    /// Check if command succeeded
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }
}

/// Aggregated benchmark result from multiple hosts
#[derive(Debug, Clone)]
pub struct AggregatedResult {
    /// Individual results from each host
    pub host_results: Vec<HostBenchmark>,
    /// Aggregated throughput (ops/sec, geometric mean)
    pub throughput_geomean: f64,
    /// Aggregated latency p50 (arithmetic mean)
    pub latency_p50_mean_us: f64,
    /// Aggregated latency p99 (max across hosts)
    pub latency_p99_max_us: f64,
    /// Number of successful hosts
    pub hosts_succeeded: usize,
    /// Number of failed hosts
    pub hosts_failed: usize,
    /// Total collection time
    pub collection_time_ms: u64,
}

impl AggregatedResult {
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        let total = self.hosts_succeeded + self.hosts_failed;
        if total == 0 {
            0.0
        } else {
            self.hosts_succeeded as f64 / total as f64
        }
    }
}

/// Benchmark result from a single host
#[derive(Debug, Clone)]
pub struct HostBenchmark {
    /// Host identifier
    pub host: String,
    /// Host architecture
    pub architecture: String,
    /// Throughput in operations per second
    pub throughput_ops: f64,
    /// Latency p50 in microseconds
    pub latency_p50_us: f64,
    /// Latency p99 in microseconds
    pub latency_p99_us: f64,
    /// Memory usage in bytes
    pub memory_bytes: u64,
    /// GPU utilization (0.0-1.0)
    pub gpu_utilization: Option<f64>,
    /// Collection timestamp
    pub timestamp_ns: u64,
}

/// Strategy for aggregating results from multiple hosts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Use geometric mean for throughput, arithmetic for latency
    GeometricMean,
    /// Use median values
    Median,
    /// Use minimum values (pessimistic)
    Minimum,
    /// Use maximum values (optimistic)
    Maximum,
}

impl Default for AggregationStrategy {
    fn default() -> Self {
        Self::GeometricMean
    }
}

/// Remote agent configuration
#[derive(Debug, Clone)]
pub struct RemoteAgentConfig {
    /// Maximum concurrent connections
    pub max_concurrent: usize,
    /// Retry count for failed commands
    pub retry_count: u32,
    /// Retry delay in milliseconds
    pub retry_delay_ms: u64,
    /// Health check interval in seconds
    pub health_check_interval_sec: u64,
    /// Aggregation strategy
    pub aggregation: AggregationStrategy,
    /// Path to cbtop binary on remote hosts
    pub remote_binary_path: String,
}

impl Default for RemoteAgentConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 10,
            retry_count: 3,
            retry_delay_ms: 1000,
            health_check_interval_sec: 60,
            aggregation: AggregationStrategy::default(),
            remote_binary_path: "/usr/local/bin/cbtop".to_string(),
        }
    }
}

/// Remote agent for distributed benchmark collection
#[derive(Debug)]
pub struct RemoteAgent {
    /// Agent configuration
    config: RemoteAgentConfig,
    /// Registered hosts
    hosts: HashMap<String, HostState>,
    /// Command execution history
    history: Vec<CommandResult>,
    /// Maximum history size
    max_history: usize,
}

impl RemoteAgent {
    /// Create a new remote agent
    pub fn new(config: RemoteAgentConfig) -> Self {
        Self {
            config,
            hosts: HashMap::new(),
            history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Add a host to the agent pool
    pub fn add_host(&mut self, config: HostConfig) {
        let key = format!("{}:{}", config.host, config.port);
        self.hosts.insert(key, HostState::new(config));
    }

    /// Remove a host from the pool
    pub fn remove_host(&mut self, host: &str) -> Option<HostState> {
        // Try with default port if not specified
        self.hosts.remove(host)
            .or_else(|| self.hosts.remove(&format!("{}:22", host)))
    }

    /// Get all registered hosts
    pub fn hosts(&self) -> impl Iterator<Item = &HostState> {
        self.hosts.values()
    }

    /// Get available (healthy) hosts
    pub fn available_hosts(&self) -> impl Iterator<Item = &HostState> {
        self.hosts.values().filter(|h| h.is_available())
    }

    /// Get host count
    pub fn host_count(&self) -> usize {
        self.hosts.len()
    }

    /// Get available host count
    pub fn available_count(&self) -> usize {
        self.available_hosts().count()
    }

    /// Execute a command on a specific host (simulated for now)
    pub fn execute_on_host(&mut self, host_key: &str, command: &str) -> RemoteResult<CommandResult> {
        // Clone config to avoid borrow conflict
        let config = {
            let state = self.hosts.get(host_key)
                .ok_or_else(|| RemoteError::HostNotFound { host: host_key.to_string() })?;
            state.config.clone()
        };

        // Simulate command execution
        let start = Instant::now();

        // In a real implementation, this would use SSH
        // For now, we simulate based on command content
        let (exit_code, stdout, stderr) = self.simulate_command(command, &config);

        let duration_ms = start.elapsed().as_millis() as u64;

        let result = CommandResult {
            host: host_key.to_string(),
            exit_code,
            stdout,
            stderr,
            duration_ms,
        };

        // Update host state
        if let Some(state) = self.hosts.get_mut(host_key) {
            if result.success() {
                state.record_success(duration_ms);
            } else {
                state.record_failure(&result.stderr);
            }
        }

        // Add to history
        self.add_to_history(result.clone());

        if result.success() {
            Ok(result)
        } else {
            Err(RemoteError::CommandFailed {
                host: host_key.to_string(),
                exit_code: result.exit_code,
                stderr: result.stderr,
            })
        }
    }

    /// Simulate command execution (placeholder for real SSH)
    fn simulate_command(&self, command: &str, config: &HostConfig) -> (i32, String, String) {
        // Check for health check command
        if command.contains("echo") && command.contains("health") {
            return (0, "OK".to_string(), String::new());
        }

        // Check for cbtop benchmark command
        if command.contains("cbtop") && command.contains("--json") {
            let arch = config.architecture.as_deref().unwrap_or("x86_64");
            let json = format!(
                r#"{{"host":"{}","arch":"{}","throughput":1000000,"latency_p50":50,"latency_p99":200,"memory":1073741824}}"#,
                config.host, arch
            );
            return (0, json, String::new());
        }

        // Unknown command - simulate failure
        (1, String::new(), "Unknown command".to_string())
    }

    /// Execute a command on all available hosts
    pub fn execute_on_all(&mut self, command: &str) -> Vec<Result<CommandResult, RemoteError>> {
        let host_keys: Vec<String> = self.available_hosts()
            .map(|h| format!("{}:{}", h.config.host, h.config.port))
            .collect();

        let mut results = Vec::new();
        for key in host_keys {
            results.push(self.execute_on_host(&key, command));
        }
        results
    }

    /// Run a health check on all hosts
    pub fn health_check(&mut self) -> HashMap<String, HostHealth> {
        let host_keys: Vec<String> = self.hosts.keys().cloned().collect();
        let mut results = HashMap::new();

        for key in host_keys {
            let _ = self.execute_on_host(&key, "echo health");
            if let Some(state) = self.hosts.get(&key) {
                results.insert(key, state.health.clone());
            }
        }

        results
    }

    /// Collect benchmarks from all available hosts
    pub fn collect_benchmarks(&mut self) -> RemoteResult<AggregatedResult> {
        let start = Instant::now();
        let command = format!("{} benchmark --json", self.config.remote_binary_path);

        let results = self.execute_on_all(&command);

        let mut host_benchmarks = Vec::new();
        let mut failures = Vec::new();

        for result in results {
            match result {
                Ok(cmd_result) => {
                    if let Some(benchmark) = self.parse_benchmark_json(&cmd_result) {
                        host_benchmarks.push(benchmark);
                    }
                }
                Err(e) => {
                    failures.push(e.to_string());
                }
            }
        }

        if host_benchmarks.is_empty() {
            return Err(RemoteError::AllHostsFailed { failures });
        }

        // Aggregate results
        let aggregated = self.aggregate_results(&host_benchmarks, failures.len());

        Ok(AggregatedResult {
            host_results: host_benchmarks,
            throughput_geomean: aggregated.0,
            latency_p50_mean_us: aggregated.1,
            latency_p99_max_us: aggregated.2,
            hosts_succeeded: aggregated.3,
            hosts_failed: aggregated.4,
            collection_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    /// Parse benchmark JSON output
    fn parse_benchmark_json(&self, result: &CommandResult) -> Option<HostBenchmark> {
        // Simple JSON parsing (in production, use serde_json)
        let stdout = &result.stdout;

        let host = self.extract_json_string(stdout, "host")?;
        let arch = self.extract_json_string(stdout, "arch").unwrap_or_else(|| "unknown".to_string());
        let throughput = self.extract_json_number(stdout, "throughput")?;
        let latency_p50 = self.extract_json_number(stdout, "latency_p50")?;
        let latency_p99 = self.extract_json_number(stdout, "latency_p99")?;
        let memory = self.extract_json_number(stdout, "memory")? as u64;

        Some(HostBenchmark {
            host,
            architecture: arch,
            throughput_ops: throughput,
            latency_p50_us: latency_p50,
            latency_p99_us: latency_p99,
            memory_bytes: memory,
            gpu_utilization: None,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
        })
    }

    /// Extract string value from simple JSON
    fn extract_json_string(&self, json: &str, key: &str) -> Option<String> {
        let pattern = format!(r#""{}":"#, key);
        let start = json.find(&pattern)? + pattern.len();
        let rest = &json[start..];

        if rest.starts_with('"') {
            let end = rest[1..].find('"')? + 1;
            Some(rest[1..end].to_string())
        } else {
            None
        }
    }

    /// Extract number value from simple JSON
    fn extract_json_number(&self, json: &str, key: &str) -> Option<f64> {
        let pattern = format!(r#""{}":"#, key);
        let start = json.find(&pattern)? + pattern.len();
        let rest = &json[start..];

        let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
        rest[..end].trim().parse().ok()
    }

    /// Aggregate benchmark results based on strategy
    fn aggregate_results(&self, benchmarks: &[HostBenchmark], failure_count: usize) -> (f64, f64, f64, usize, usize) {
        if benchmarks.is_empty() {
            return (0.0, 0.0, 0.0, 0, failure_count);
        }

        match self.config.aggregation {
            AggregationStrategy::GeometricMean => {
                // Geometric mean for throughput
                let log_sum: f64 = benchmarks.iter()
                    .map(|b| b.throughput_ops.ln())
                    .sum();
                let throughput_geomean = (log_sum / benchmarks.len() as f64).exp();

                // Arithmetic mean for latency p50
                let latency_p50_mean = benchmarks.iter()
                    .map(|b| b.latency_p50_us)
                    .sum::<f64>() / benchmarks.len() as f64;

                // Max for latency p99 (worst case)
                let latency_p99_max = benchmarks.iter()
                    .map(|b| b.latency_p99_us)
                    .fold(0.0_f64, |a, b| a.max(b));

                (throughput_geomean, latency_p50_mean, latency_p99_max, benchmarks.len(), failure_count)
            }
            AggregationStrategy::Median => {
                let mut throughputs: Vec<f64> = benchmarks.iter().map(|b| b.throughput_ops).collect();
                let mut latencies_p50: Vec<f64> = benchmarks.iter().map(|b| b.latency_p50_us).collect();
                let mut latencies_p99: Vec<f64> = benchmarks.iter().map(|b| b.latency_p99_us).collect();

                throughputs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                latencies_p50.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                latencies_p99.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let mid = benchmarks.len() / 2;
                (throughputs[mid], latencies_p50[mid], latencies_p99[mid], benchmarks.len(), failure_count)
            }
            AggregationStrategy::Minimum => {
                let throughput = benchmarks.iter().map(|b| b.throughput_ops).fold(f64::INFINITY, f64::min);
                let latency_p50 = benchmarks.iter().map(|b| b.latency_p50_us).fold(f64::INFINITY, f64::min);
                let latency_p99 = benchmarks.iter().map(|b| b.latency_p99_us).fold(f64::INFINITY, f64::min);
                (throughput, latency_p50, latency_p99, benchmarks.len(), failure_count)
            }
            AggregationStrategy::Maximum => {
                let throughput = benchmarks.iter().map(|b| b.throughput_ops).fold(0.0_f64, f64::max);
                let latency_p50 = benchmarks.iter().map(|b| b.latency_p50_us).fold(0.0_f64, f64::max);
                let latency_p99 = benchmarks.iter().map(|b| b.latency_p99_us).fold(0.0_f64, f64::max);
                (throughput, latency_p50, latency_p99, benchmarks.len(), failure_count)
            }
        }
    }

    /// Add result to history with size limit
    fn add_to_history(&mut self, result: CommandResult) {
        self.history.push(result);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get command history
    pub fn history(&self) -> &[CommandResult] {
        &self.history
    }

    /// Get configuration
    pub fn config(&self) -> &RemoteAgentConfig {
        &self.config
    }

    /// Filter hosts by label
    pub fn hosts_with_label(&self, key: &str, value: &str) -> Vec<&HostState> {
        self.hosts.values()
            .filter(|h| h.config.labels.get(key).map(|v| v == value).unwrap_or(false))
            .collect()
    }

    /// Filter hosts by architecture
    pub fn hosts_with_arch(&self, arch: &str) -> Vec<&HostState> {
        self.hosts.values()
            .filter(|h| h.config.architecture.as_deref() == Some(arch))
            .collect()
    }
}

/// Default retry delay in milliseconds
pub const DEFAULT_RETRY_DELAY_MS: u64 = 1000;

/// Default maximum concurrent connections
pub const DEFAULT_MAX_CONCURRENT: usize = 10;

/// Default health check interval in seconds
pub const DEFAULT_HEALTH_CHECK_INTERVAL_SEC: u64 = 60;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_config_builder() {
        let config = HostConfig::new("server1.example.com", "admin")
            .with_port(2222)
            .with_connect_timeout_ms(5000)
            .with_architecture("x86_64")
            .with_label("env", "production");

        assert_eq!(config.host, "server1.example.com");
        assert_eq!(config.username, "admin");
        assert_eq!(config.port, 2222);
        assert_eq!(config.connect_timeout_ms, 5000);
        assert_eq!(config.architecture, Some("x86_64".to_string()));
        assert_eq!(config.labels.get("env"), Some(&"production".to_string()));
    }

    #[test]
    fn test_host_state_availability() {
        let config = HostConfig::new("host1", "user");
        let mut state = HostState::new(config);

        // Initially unknown, should be available
        assert!(state.is_available());

        // After success, should be healthy
        state.record_success(100);
        assert!(state.is_available());
        assert!(matches!(state.health, HostHealth::Healthy));

        // After degraded response, still available
        state.record_success(6000);
        assert!(state.is_available());
        assert!(matches!(state.health, HostHealth::Degraded { .. }));

        // After 3 failures, unreachable
        state.record_failure("error 1");
        state.record_failure("error 2");
        state.record_failure("error 3");
        assert!(!state.is_available());
        assert!(matches!(state.health, HostHealth::Unreachable { .. }));
    }

    #[test]
    fn test_host_state_latency_tracking() {
        let config = HostConfig::new("host1", "user");
        let mut state = HostState::new(config);

        state.record_success(100);
        assert_eq!(state.avg_latency_ms, 100.0);

        state.record_success(200);
        assert_eq!(state.avg_latency_ms, 150.0);

        state.record_success(300);
        assert!((state.avg_latency_ms - 200.0).abs() < 0.01);
    }

    #[test]
    fn test_remote_agent_add_hosts() {
        let config = RemoteAgentConfig::default();
        let mut agent = RemoteAgent::new(config);

        agent.add_host(HostConfig::new("host1", "user1"));
        agent.add_host(HostConfig::new("host2", "user2"));

        assert_eq!(agent.host_count(), 2);
        assert_eq!(agent.available_count(), 2);
    }

    #[test]
    fn test_remote_agent_remove_host() {
        let config = RemoteAgentConfig::default();
        let mut agent = RemoteAgent::new(config);

        agent.add_host(HostConfig::new("host1", "user1"));
        assert_eq!(agent.host_count(), 1);

        let removed = agent.remove_host("host1");
        assert!(removed.is_some());
        assert_eq!(agent.host_count(), 0);
    }

    #[test]
    fn test_remote_agent_execute_health_check() {
        let config = RemoteAgentConfig::default();
        let mut agent = RemoteAgent::new(config);

        agent.add_host(HostConfig::new("host1", "user1"));

        let result = agent.execute_on_host("host1:22", "echo health");
        assert!(result.is_ok());

        let cmd_result = result.unwrap();
        assert!(cmd_result.success());
        assert_eq!(cmd_result.stdout, "OK");
    }

    #[test]
    fn test_remote_agent_health_check_all() {
        let config = RemoteAgentConfig::default();
        let mut agent = RemoteAgent::new(config);

        agent.add_host(HostConfig::new("host1", "user1"));
        agent.add_host(HostConfig::new("host2", "user2"));

        let health = agent.health_check();

        assert_eq!(health.len(), 2);
        assert!(matches!(health.get("host1:22"), Some(HostHealth::Healthy)));
        assert!(matches!(health.get("host2:22"), Some(HostHealth::Healthy)));
    }

    #[test]
    fn test_remote_agent_collect_benchmarks() {
        let config = RemoteAgentConfig::default();
        let mut agent = RemoteAgent::new(config);

        agent.add_host(HostConfig::new("host1", "user1").with_architecture("x86_64"));
        agent.add_host(HostConfig::new("host2", "user2").with_architecture("aarch64"));

        let result = agent.collect_benchmarks();
        assert!(result.is_ok());

        let aggregated = result.unwrap();
        assert_eq!(aggregated.hosts_succeeded, 2);
        assert_eq!(aggregated.hosts_failed, 0);
        assert!(aggregated.throughput_geomean > 0.0);
    }

    #[test]
    fn test_aggregation_strategy_geometric_mean() {
        let config = RemoteAgentConfig {
            aggregation: AggregationStrategy::GeometricMean,
            ..Default::default()
        };
        let agent = RemoteAgent::new(config);

        let benchmarks = vec![
            HostBenchmark {
                host: "h1".to_string(),
                architecture: "x86_64".to_string(),
                throughput_ops: 100.0,
                latency_p50_us: 50.0,
                latency_p99_us: 100.0,
                memory_bytes: 1024,
                gpu_utilization: None,
                timestamp_ns: 0,
            },
            HostBenchmark {
                host: "h2".to_string(),
                architecture: "x86_64".to_string(),
                throughput_ops: 400.0,
                latency_p50_us: 100.0,
                latency_p99_us: 200.0,
                memory_bytes: 2048,
                gpu_utilization: None,
                timestamp_ns: 0,
            },
        ];

        let (throughput, latency_p50, latency_p99, succeeded, failed) =
            agent.aggregate_results(&benchmarks, 0);

        // Geometric mean of 100 and 400 is 200
        assert!((throughput - 200.0).abs() < 0.01);
        // Arithmetic mean of 50 and 100 is 75
        assert!((latency_p50 - 75.0).abs() < 0.01);
        // Max of 100 and 200 is 200
        assert_eq!(latency_p99, 200.0);
        assert_eq!(succeeded, 2);
        assert_eq!(failed, 0);
    }

    #[test]
    fn test_aggregation_strategy_median() {
        let config = RemoteAgentConfig {
            aggregation: AggregationStrategy::Median,
            ..Default::default()
        };
        let agent = RemoteAgent::new(config);

        let benchmarks = vec![
            HostBenchmark {
                host: "h1".to_string(),
                architecture: "x86_64".to_string(),
                throughput_ops: 100.0,
                latency_p50_us: 50.0,
                latency_p99_us: 100.0,
                memory_bytes: 1024,
                gpu_utilization: None,
                timestamp_ns: 0,
            },
            HostBenchmark {
                host: "h2".to_string(),
                architecture: "x86_64".to_string(),
                throughput_ops: 200.0,
                latency_p50_us: 75.0,
                latency_p99_us: 150.0,
                memory_bytes: 2048,
                gpu_utilization: None,
                timestamp_ns: 0,
            },
            HostBenchmark {
                host: "h3".to_string(),
                architecture: "x86_64".to_string(),
                throughput_ops: 300.0,
                latency_p50_us: 100.0,
                latency_p99_us: 200.0,
                memory_bytes: 3072,
                gpu_utilization: None,
                timestamp_ns: 0,
            },
        ];

        let (throughput, latency_p50, latency_p99, _, _) =
            agent.aggregate_results(&benchmarks, 0);

        // Median of [100, 200, 300] is 200
        assert_eq!(throughput, 200.0);
        assert_eq!(latency_p50, 75.0);
        assert_eq!(latency_p99, 150.0);
    }

    #[test]
    fn test_aggregation_strategy_minimum() {
        let config = RemoteAgentConfig {
            aggregation: AggregationStrategy::Minimum,
            ..Default::default()
        };
        let agent = RemoteAgent::new(config);

        let benchmarks = vec![
            HostBenchmark {
                host: "h1".to_string(),
                architecture: "x86_64".to_string(),
                throughput_ops: 100.0,
                latency_p50_us: 50.0,
                latency_p99_us: 100.0,
                memory_bytes: 1024,
                gpu_utilization: None,
                timestamp_ns: 0,
            },
            HostBenchmark {
                host: "h2".to_string(),
                architecture: "x86_64".to_string(),
                throughput_ops: 200.0,
                latency_p50_us: 75.0,
                latency_p99_us: 150.0,
                memory_bytes: 2048,
                gpu_utilization: None,
                timestamp_ns: 0,
            },
        ];

        let (throughput, latency_p50, latency_p99, _, _) =
            agent.aggregate_results(&benchmarks, 0);

        assert_eq!(throughput, 100.0);
        assert_eq!(latency_p50, 50.0);
        assert_eq!(latency_p99, 100.0);
    }

    #[test]
    fn test_hosts_with_label() {
        let config = RemoteAgentConfig::default();
        let mut agent = RemoteAgent::new(config);

        agent.add_host(HostConfig::new("host1", "user").with_label("env", "prod"));
        agent.add_host(HostConfig::new("host2", "user").with_label("env", "staging"));
        agent.add_host(HostConfig::new("host3", "user").with_label("env", "prod"));

        let prod_hosts = agent.hosts_with_label("env", "prod");
        assert_eq!(prod_hosts.len(), 2);
    }

    #[test]
    fn test_hosts_with_arch() {
        let config = RemoteAgentConfig::default();
        let mut agent = RemoteAgent::new(config);

        agent.add_host(HostConfig::new("host1", "user").with_architecture("x86_64"));
        agent.add_host(HostConfig::new("host2", "user").with_architecture("aarch64"));
        agent.add_host(HostConfig::new("host3", "user").with_architecture("x86_64"));

        let x86_hosts = agent.hosts_with_arch("x86_64");
        assert_eq!(x86_hosts.len(), 2);

        let arm_hosts = agent.hosts_with_arch("aarch64");
        assert_eq!(arm_hosts.len(), 1);
    }

    #[test]
    fn test_command_result_success() {
        let result = CommandResult {
            host: "host1".to_string(),
            exit_code: 0,
            stdout: "output".to_string(),
            stderr: String::new(),
            duration_ms: 100,
        };
        assert!(result.success());

        let failed = CommandResult {
            host: "host1".to_string(),
            exit_code: 1,
            stdout: String::new(),
            stderr: "error".to_string(),
            duration_ms: 50,
        };
        assert!(!failed.success());
    }

    #[test]
    fn test_aggregated_result_success_rate() {
        let result = AggregatedResult {
            host_results: vec![],
            throughput_geomean: 100.0,
            latency_p50_mean_us: 50.0,
            latency_p99_max_us: 200.0,
            hosts_succeeded: 3,
            hosts_failed: 1,
            collection_time_ms: 1000,
        };

        assert!((result.success_rate() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_remote_error_display() {
        let err = RemoteError::ConnectionFailed {
            host: "host1".to_string(),
            reason: "timeout".to_string(),
        };
        assert!(err.to_string().contains("host1"));
        assert!(err.to_string().contains("timeout"));

        let err = RemoteError::AllHostsFailed {
            failures: vec!["error1".to_string(), "error2".to_string()],
        };
        assert!(err.to_string().contains("error1"));
    }

    #[test]
    fn test_json_parsing() {
        let config = RemoteAgentConfig::default();
        let agent = RemoteAgent::new(config);

        let json = r#"{"host":"server1","arch":"x86_64","throughput":1000000,"latency_p50":50,"latency_p99":200,"memory":1073741824}"#;

        // Test string extraction
        assert_eq!(agent.extract_json_string(json, "host"), Some("server1".to_string()));
        assert_eq!(agent.extract_json_string(json, "arch"), Some("x86_64".to_string()));

        // Test number extraction
        assert_eq!(agent.extract_json_number(json, "throughput"), Some(1000000.0));
        assert_eq!(agent.extract_json_number(json, "latency_p50"), Some(50.0));
        assert_eq!(agent.extract_json_number(json, "memory"), Some(1073741824.0));
    }

    #[test]
    fn test_history_limit() {
        let config = RemoteAgentConfig::default();
        let mut agent = RemoteAgent::new(config);
        agent.max_history = 3;

        agent.add_host(HostConfig::new("host1", "user"));

        // Execute more commands than history limit
        for _ in 0..5 {
            let _ = agent.execute_on_host("host1:22", "echo health");
        }

        assert_eq!(agent.history().len(), 3);
    }

    // FKR-045: Multi-host heterogeneous collection
    #[test]
    fn test_fkr_045_heterogeneous_collection() {
        let config = RemoteAgentConfig::default();
        let mut agent = RemoteAgent::new(config);

        // Add hosts with different architectures
        agent.add_host(HostConfig::new("x86-host", "user").with_architecture("x86_64"));
        agent.add_host(HostConfig::new("arm-host", "user").with_architecture("aarch64"));
        agent.add_host(HostConfig::new("riscv-host", "user").with_architecture("riscv64"));

        let result = agent.collect_benchmarks();
        assert!(result.is_ok());

        let aggregated = result.unwrap();

        // Verify all 3 hosts contributed results
        assert_eq!(aggregated.hosts_succeeded, 3);
        assert_eq!(aggregated.host_results.len(), 3);

        // Verify different architectures are represented
        let archs: Vec<&str> = aggregated.host_results.iter()
            .map(|r| r.architecture.as_str())
            .collect();
        assert!(archs.contains(&"x86_64"));
        assert!(archs.contains(&"aarch64"));
        assert!(archs.contains(&"riscv64"));
    }
}
