//! Federated Metrics Aggregation (PMAT-048)
//!
//! Multi-host metrics aggregation with CRDT-based merging for distributed profiling.
//!
//! # Design
//!
//! - CRDT (Conflict-free Replicated Data Types) for partition-tolerant merging
//! - Adaptive sampling based on network bandwidth
//! - Automatic topology detection and health-based routing
//! - Skew detection for identifying degraded nodes
//!
//! # Falsification (FKR-049)
//!
//! H₀: Federated aggregation cannot maintain correctness across network partitions
//! Test: Simulate 3-host cluster with partition, verify CRDT convergence

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Result type for federated operations
pub type FederatedResult<T> = Result<T, FederatedError>;

/// Errors in federated operations
#[derive(Debug, Clone, PartialEq)]
pub enum FederatedError {
    /// Host not found in federation
    HostNotFound { host_id: String },
    /// Network partition detected
    PartitionDetected { affected_hosts: Vec<String> },
    /// Merge conflict that couldn't be resolved
    MergeConflict { reason: String },
    /// Clock drift too large
    ClockDriftExceeded { drift_ms: i64, max_ms: i64 },
    /// Memory limit exceeded
    MemoryLimitExceeded {
        used_bytes: usize,
        limit_bytes: usize,
    },
    /// Invalid configuration
    InvalidConfig { reason: String },
}

impl std::fmt::Display for FederatedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HostNotFound { host_id } => write!(f, "Host not found: {}", host_id),
            Self::PartitionDetected { affected_hosts } => {
                write!(f, "Partition detected affecting: {:?}", affected_hosts)
            }
            Self::MergeConflict { reason } => write!(f, "Merge conflict: {}", reason),
            Self::ClockDriftExceeded { drift_ms, max_ms } => {
                write!(f, "Clock drift {}ms exceeds max {}ms", drift_ms, max_ms)
            }
            Self::MemoryLimitExceeded {
                used_bytes,
                limit_bytes,
            } => {
                write!(f, "Memory {} exceeds limit {}", used_bytes, limit_bytes)
            }
            Self::InvalidConfig { reason } => write!(f, "Invalid config: {}", reason),
        }
    }
}

impl std::error::Error for FederatedError {}

/// Unique identifier for a metric sample
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SampleId {
    /// Host that generated the sample
    pub host_id: String,
    /// Logical timestamp (Lamport clock)
    pub logical_time: u64,
    /// Unique sequence number within host
    pub sequence: u64,
}

impl SampleId {
    /// Create a new sample ID
    pub fn new(host_id: impl Into<String>, logical_time: u64, sequence: u64) -> Self {
        Self {
            host_id: host_id.into(),
            logical_time,
            sequence,
        }
    }
}

/// A single metric sample with vector clock
#[derive(Debug, Clone)]
pub struct MetricSample {
    /// Unique identifier
    pub id: SampleId,
    /// Metric name
    pub metric_name: String,
    /// Metric value
    pub value: f64,
    /// Wall clock timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Vector clock for causal ordering
    pub vector_clock: HashMap<String, u64>,
}

impl MetricSample {
    /// Create a new metric sample
    pub fn new(
        host_id: impl Into<String>,
        logical_time: u64,
        sequence: u64,
        metric_name: impl Into<String>,
        value: f64,
    ) -> Self {
        let host = host_id.into();
        let mut vector_clock = HashMap::new();
        vector_clock.insert(host.clone(), logical_time);

        Self {
            id: SampleId::new(host, logical_time, sequence),
            metric_name: metric_name.into(),
            value,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            vector_clock,
        }
    }
}

/// G-Counter CRDT for monotonic counters
#[derive(Debug, Clone, Default)]
pub struct GCounter {
    /// Per-host counts
    counts: HashMap<String, u64>,
}

impl GCounter {
    /// Create a new G-Counter
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment counter for a host
    pub fn increment(&mut self, host_id: &str, amount: u64) {
        *self.counts.entry(host_id.to_string()).or_insert(0) += amount;
    }

    /// Get total count across all hosts
    pub fn value(&self) -> u64 {
        self.counts.values().sum()
    }

    /// Merge with another G-Counter (take max per host)
    pub fn merge(&mut self, other: &GCounter) {
        for (host, count) in &other.counts {
            let entry = self.counts.entry(host.clone()).or_insert(0);
            *entry = (*entry).max(*count);
        }
    }

    /// Get count for a specific host
    pub fn host_count(&self, host_id: &str) -> u64 {
        self.counts.get(host_id).copied().unwrap_or(0)
    }
}

/// LWW-Register CRDT for last-writer-wins values
#[derive(Debug, Clone)]
pub struct LwwRegister<T: Clone> {
    /// Current value
    value: T,
    /// Timestamp of last write
    timestamp: u64,
    /// Host that performed last write
    writer: String,
}

impl<T: Clone + Default> Default for LwwRegister<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            timestamp: 0,
            writer: String::new(),
        }
    }
}

impl<T: Clone> LwwRegister<T> {
    /// Create a new register with initial value
    pub fn new(value: T, timestamp: u64, writer: impl Into<String>) -> Self {
        Self {
            value,
            timestamp,
            writer: writer.into(),
        }
    }

    /// Update value if timestamp is newer
    pub fn update(&mut self, value: T, timestamp: u64, writer: impl Into<String>) {
        if timestamp > self.timestamp {
            self.value = value;
            self.timestamp = timestamp;
            self.writer = writer.into();
        }
    }

    /// Get current value
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Get timestamp
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Merge with another register (keep newer)
    pub fn merge(&mut self, other: &LwwRegister<T>) {
        if other.timestamp > self.timestamp {
            self.value = other.value.clone();
            self.timestamp = other.timestamp;
            self.writer = other.writer.clone();
        }
    }
}

/// OR-Set CRDT for add/remove sets
#[derive(Debug, Clone)]
pub struct OrSet<T: Clone + Eq + std::hash::Hash> {
    /// Elements with their unique tags
    elements: HashMap<T, HashSet<String>>,
    /// Tombstones for removed elements
    tombstones: HashMap<T, HashSet<String>>,
}

impl<T: Clone + Eq + std::hash::Hash> Default for OrSet<T> {
    fn default() -> Self {
        Self {
            elements: HashMap::new(),
            tombstones: HashMap::new(),
        }
    }
}

impl<T: Clone + Eq + std::hash::Hash> OrSet<T> {
    /// Create a new OR-Set
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an element with a unique tag
    pub fn add(&mut self, element: T, tag: String) {
        self.elements.entry(element).or_default().insert(tag);
    }

    /// Remove an element (tombstone all tags)
    pub fn remove(&mut self, element: &T) {
        if let Some(tags) = self.elements.get(element) {
            let tombstone_entry = self.tombstones.entry(element.clone()).or_default();
            for tag in tags {
                tombstone_entry.insert(tag.clone());
            }
        }
    }

    /// Check if element is in set
    pub fn contains(&self, element: &T) -> bool {
        if let Some(tags) = self.elements.get(element) {
            let tombstones = self.tombstones.get(element);
            tags.iter()
                .any(|tag| tombstones.map_or(true, |ts| !ts.contains(tag)))
        } else {
            false
        }
    }

    /// Get all active elements
    pub fn elements(&self) -> Vec<&T> {
        self.elements.keys().filter(|e| self.contains(e)).collect()
    }

    /// Merge with another OR-Set
    pub fn merge(&mut self, other: &OrSet<T>) {
        // Merge elements
        for (elem, tags) in &other.elements {
            let entry = self.elements.entry(elem.clone()).or_default();
            entry.extend(tags.iter().cloned());
        }
        // Merge tombstones
        for (elem, tags) in &other.tombstones {
            let entry = self.tombstones.entry(elem.clone()).or_default();
            entry.extend(tags.iter().cloned());
        }
    }
}

/// Host state in the federation
#[derive(Debug, Clone)]
pub struct FederatedHost {
    /// Host identifier
    pub host_id: String,
    /// Last seen timestamp
    pub last_seen: Instant,
    /// Sample count from this host
    pub sample_count: GCounter,
    /// Health status (0.0 = dead, 1.0 = healthy)
    pub health: f64,
    /// Current sampling rate (samples per second)
    pub sampling_rate: f64,
    /// Network latency to this host (milliseconds)
    pub latency_ms: f64,
    /// Logical clock value
    pub logical_clock: u64,
}

impl FederatedHost {
    /// Create a new federated host
    pub fn new(host_id: impl Into<String>) -> Self {
        Self {
            host_id: host_id.into(),
            last_seen: Instant::now(),
            sample_count: GCounter::new(),
            health: 1.0,
            sampling_rate: 100.0, // Default 100 Hz
            latency_ms: 0.0,
            logical_clock: 0,
        }
    }

    /// Update last seen time
    pub fn touch(&mut self) {
        self.last_seen = Instant::now();
    }

    /// Check if host is stale (not seen recently)
    pub fn is_stale(&self, timeout: Duration) -> bool {
        self.last_seen.elapsed() > timeout
    }

    /// Increment logical clock
    pub fn tick(&mut self) -> u64 {
        self.logical_clock += 1;
        self.logical_clock
    }

    /// Update logical clock from received message
    pub fn sync_clock(&mut self, received_time: u64) {
        self.logical_clock = self.logical_clock.max(received_time) + 1;
    }
}

/// Aggregated metrics across federation
#[derive(Debug, Clone, Default)]
pub struct AggregatedMetrics {
    /// Metric name
    pub metric_name: String,
    /// All sample values
    pub values: Vec<f64>,
    /// Per-host sample counts
    pub host_counts: HashMap<String, usize>,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Sum for mean calculation
    pub sum: f64,
}

impl AggregatedMetrics {
    /// Create new aggregated metrics
    pub fn new(metric_name: impl Into<String>) -> Self {
        Self {
            metric_name: metric_name.into(),
            values: Vec::new(),
            host_counts: HashMap::new(),
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            sum: 0.0,
        }
    }

    /// Add a sample
    pub fn add_sample(&mut self, host_id: &str, value: f64) {
        self.values.push(value);
        *self.host_counts.entry(host_id.to_string()).or_insert(0) += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value;
    }

    /// Get mean value
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }

    /// Get percentile value
    pub fn percentile(&self, p: f64) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }

        let mut sorted = self.values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Get p50
    pub fn p50(&self) -> f64 {
        self.percentile(50.0)
    }

    /// Get p95
    pub fn p95(&self) -> f64 {
        self.percentile(95.0)
    }

    /// Get p99
    pub fn p99(&self) -> f64 {
        self.percentile(99.0)
    }

    /// Detect skewed host (significantly slower)
    pub fn detect_skew(&self, threshold_percent: f64) -> Vec<String> {
        let mean = self.mean();
        if mean == 0.0 {
            return Vec::new();
        }

        let mut skewed = Vec::new();
        for (host_id, count) in &self.host_counts {
            // Calculate host-specific mean (simplified: use count as proxy)
            let expected_count = self.values.len() / self.host_counts.len().max(1);
            let deviation =
                ((*count as f64 - expected_count as f64) / expected_count as f64).abs() * 100.0;

            if deviation > threshold_percent {
                skewed.push(host_id.clone());
            }
        }
        skewed
    }
}

/// Configuration for federation
#[derive(Debug, Clone)]
pub struct FederationConfig {
    /// Maximum clock drift tolerance (milliseconds)
    pub max_clock_drift_ms: i64,
    /// Host timeout before considered stale
    pub host_timeout: Duration,
    /// Memory limit per federation (bytes)
    pub memory_limit_bytes: usize,
    /// Default sampling rate (Hz)
    pub default_sampling_rate: f64,
    /// Skew detection threshold (percent)
    pub skew_threshold_percent: f64,
    /// Partition recovery timeout
    pub partition_recovery_timeout: Duration,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            max_clock_drift_ms: 100,
            host_timeout: Duration::from_secs(30),
            memory_limit_bytes: 100 * 1024 * 1024, // 100MB
            default_sampling_rate: 100.0,
            skew_threshold_percent: 40.0,
            partition_recovery_timeout: Duration::from_secs(30),
        }
    }
}

/// Federated metrics aggregator
#[derive(Debug)]
pub struct MetricsFederation {
    /// Federation configuration
    config: FederationConfig,
    /// Local host ID
    local_host_id: String,
    /// Known hosts in federation
    hosts: HashMap<String, FederatedHost>,
    /// Active hosts (OR-Set for partition tolerance)
    active_hosts: OrSet<String>,
    /// Collected samples (keyed by sample ID to prevent duplicates)
    samples: HashMap<SampleId, MetricSample>,
    /// Total sample counter
    total_samples: GCounter,
    /// Aggregated metrics by name
    aggregated: HashMap<String, AggregatedMetrics>,
    /// Current memory usage estimate
    memory_usage: usize,
    /// Logical clock for this node
    logical_clock: u64,
    /// Sample sequence counter
    sequence: u64,
}

impl MetricsFederation {
    /// Create a new federation
    pub fn new(local_host_id: impl Into<String>, config: FederationConfig) -> Self {
        let host_id = local_host_id.into();
        let mut hosts = HashMap::new();
        hosts.insert(host_id.clone(), FederatedHost::new(&host_id));

        let mut active_hosts = OrSet::new();
        active_hosts.add(host_id.clone(), format!("{}-0", host_id));

        Self {
            config,
            local_host_id: host_id,
            hosts,
            active_hosts,
            samples: HashMap::new(),
            total_samples: GCounter::new(),
            aggregated: HashMap::new(),
            memory_usage: 0,
            logical_clock: 0,
            sequence: 0,
        }
    }

    /// Add a host to the federation
    pub fn add_host(&mut self, host_id: impl Into<String>) {
        let id = host_id.into();
        if !self.hosts.contains_key(&id) {
            self.hosts.insert(id.clone(), FederatedHost::new(&id));
            let tick = self.tick();
            let local_id = self.local_host_id.clone();
            let tag = format!("{}-{}", local_id, tick);
            self.active_hosts.add(id, tag);
        }
    }

    /// Remove a host from the federation
    pub fn remove_host(&mut self, host_id: &str) {
        self.active_hosts.remove(&host_id.to_string());
    }

    /// Get active host count
    pub fn active_host_count(&self) -> usize {
        self.active_hosts.elements().len()
    }

    /// Increment logical clock
    fn tick(&mut self) -> u64 {
        self.logical_clock += 1;
        self.logical_clock
    }

    /// Record a local metric sample
    pub fn record(
        &mut self,
        metric_name: impl Into<String>,
        value: f64,
    ) -> FederatedResult<SampleId> {
        let time = self.tick();
        self.sequence += 1;

        let sample =
            MetricSample::new(&self.local_host_id, time, self.sequence, metric_name, value);

        self.add_sample(sample)
    }

    /// Add a sample (local or remote)
    pub fn add_sample(&mut self, sample: MetricSample) -> FederatedResult<SampleId> {
        // Check for duplicates (idempotent)
        if self.samples.contains_key(&sample.id) {
            return Ok(sample.id);
        }

        // Check memory limit
        let sample_size = std::mem::size_of::<MetricSample>() + sample.metric_name.len();
        if self.memory_usage + sample_size > self.config.memory_limit_bytes {
            return Err(FederatedError::MemoryLimitExceeded {
                used_bytes: self.memory_usage,
                limit_bytes: self.config.memory_limit_bytes,
            });
        }

        // Update host state
        if let Some(host) = self.hosts.get_mut(&sample.id.host_id) {
            host.touch();
            host.sample_count.increment(&sample.id.host_id, 1);
            host.sync_clock(sample.id.logical_time);
        }

        // Update aggregation
        let agg = self
            .aggregated
            .entry(sample.metric_name.clone())
            .or_insert_with(|| AggregatedMetrics::new(&sample.metric_name));
        agg.add_sample(&sample.id.host_id, sample.value);

        // Update totals
        self.total_samples.increment(&sample.id.host_id, 1);
        self.memory_usage += sample_size;

        let id = sample.id.clone();
        self.samples.insert(sample.id.clone(), sample);

        Ok(id)
    }

    /// Merge samples from another federation node
    pub fn merge(&mut self, other: &MetricsFederation) -> FederatedResult<usize> {
        let mut merged_count = 0;

        // Merge active hosts
        self.active_hosts.merge(&other.active_hosts);

        // Merge samples
        for (id, sample) in &other.samples {
            if !self.samples.contains_key(id) {
                self.add_sample(sample.clone())?;
                merged_count += 1;
            }
        }

        // Merge counters
        self.total_samples.merge(&other.total_samples);

        // Sync logical clock
        self.logical_clock = self.logical_clock.max(other.logical_clock) + 1;

        Ok(merged_count)
    }

    /// Get aggregated metrics for a metric name
    pub fn get_aggregated(&self, metric_name: &str) -> Option<&AggregatedMetrics> {
        self.aggregated.get(metric_name)
    }

    /// Get all metric names
    pub fn metric_names(&self) -> Vec<&String> {
        self.aggregated.keys().collect()
    }

    /// Detect hosts with performance skew
    pub fn detect_skewed_hosts(&self) -> Vec<String> {
        let mut skewed = HashSet::new();

        for agg in self.aggregated.values() {
            for host in agg.detect_skew(self.config.skew_threshold_percent) {
                skewed.insert(host);
            }
        }

        skewed.into_iter().collect()
    }

    /// Get hosts that are stale (not seen recently)
    pub fn get_stale_hosts(&self) -> Vec<String> {
        self.hosts
            .iter()
            .filter(|(_, host)| host.is_stale(self.config.host_timeout))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Update host health based on sample count
    pub fn update_health(&mut self) {
        let total = self.total_samples.value() as f64;
        if total == 0.0 {
            return;
        }

        let host_count = self.hosts.len() as f64;
        let expected_per_host = total / host_count;

        for (host_id, host) in &mut self.hosts {
            let count = self.total_samples.host_count(host_id) as f64;
            if expected_per_host > 0.0 {
                // Health is ratio of actual to expected samples
                host.health = (count / expected_per_host).min(1.0);
            }
        }
    }

    /// Adapt sampling rates based on network conditions
    pub fn adapt_sampling_rates(&mut self, bandwidth_mbps: f64) {
        // Lower sampling rate for higher latency hosts
        let base_rate = self.config.default_sampling_rate;

        for host in self.hosts.values_mut() {
            // Reduce rate proportionally to latency
            let latency_factor = 1.0 / (1.0 + host.latency_ms / 100.0);
            // Reduce rate based on bandwidth
            let bandwidth_factor = (bandwidth_mbps / 100.0).min(1.0);

            host.sampling_rate = base_rate * latency_factor * bandwidth_factor;
        }
    }

    /// Get total sample count
    pub fn total_samples(&self) -> u64 {
        self.total_samples.value()
    }

    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        self.memory_usage
    }

    /// Get configuration
    pub fn config(&self) -> &FederationConfig {
        &self.config
    }

    /// Get host by ID
    pub fn get_host(&self, host_id: &str) -> Option<&FederatedHost> {
        self.hosts.get(host_id)
    }
}

/// Default maximum clock drift (milliseconds)
pub const DEFAULT_MAX_CLOCK_DRIFT_MS: i64 = 100;

/// Default memory limit (bytes)
pub const DEFAULT_MEMORY_LIMIT_BYTES: usize = 100 * 1024 * 1024;

/// Default skew threshold (percent)
pub const DEFAULT_SKEW_THRESHOLD_PERCENT: f64 = 40.0;


#[cfg(test)]
mod tests;
