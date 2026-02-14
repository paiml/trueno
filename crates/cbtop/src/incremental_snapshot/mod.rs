//! Incremental Profile Snapshots (PMAT-050)
//!
//! Time-series profile compression with differential storage and streaming decompression.
//!
//! # Design
//!
//! - Delta compression using XOR encoding for consecutive snapshots
//! - Tiered retention: raw (1 day) → compressed (30 days) → archive (1 year)
//! - Streaming decompression with bounded memory
//! - Index by timestamp and workload fingerprint for fast queries
//!
//! # Falsification (FKR-051)
//!
//! H₀: Incremental snapshots cannot achieve <5% compression ratio
//! Test: Store 100 sequential snapshots, verify disk usage <5% of raw

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Result type for snapshot operations
pub type SnapshotResult<T> = Result<T, SnapshotError>;

/// Errors in snapshot operations
#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotError {
    /// IO error
    IoError { reason: String },
    /// Snapshot not found
    NotFound { index: usize },
    /// Corrupt snapshot
    Corrupt { reason: String },
    /// Checksum mismatch
    ChecksumMismatch { expected: u32, actual: u32 },
    /// Index out of bounds
    IndexOutOfBounds { index: usize, max: usize },
    /// Memory limit exceeded
    MemoryExceeded { limit_bytes: usize },
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError { reason } => write!(f, "IO error: {}", reason),
            Self::NotFound { index } => write!(f, "Snapshot {} not found", index),
            Self::Corrupt { reason } => write!(f, "Corrupt snapshot: {}", reason),
            Self::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "Checksum mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::IndexOutOfBounds { index, max } => {
                write!(f, "Index {} out of bounds (max {})", index, max)
            }
            Self::MemoryExceeded { limit_bytes } => {
                write!(f, "Memory limit {} exceeded", limit_bytes)
            }
        }
    }
}

impl std::error::Error for SnapshotError {}

/// Retention tier for snapshots
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionTier {
    /// Raw snapshots (full data)
    Raw,
    /// Compressed snapshots (delta encoded)
    Compressed,
    /// Archived snapshots (highly compressed, read-only)
    Archive,
}

impl RetentionTier {
    /// Get maximum age for this tier
    pub fn max_age(&self) -> Duration {
        match self {
            Self::Raw => Duration::from_secs(24 * 60 * 60), // 1 day
            Self::Compressed => Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            Self::Archive => Duration::from_secs(365 * 24 * 60 * 60), // 1 year
        }
    }
}

/// Profile metric data
#[derive(Debug, Clone)]
pub struct MetricData {
    /// Metric name
    pub name: String,
    /// Metric values (time series)
    pub values: Vec<f64>,
    /// Timestamps (nanoseconds since epoch)
    pub timestamps: Vec<u64>,
}

impl MetricData {
    /// Create new metric data
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            values: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    /// Add a sample
    pub fn add(&mut self, value: f64, timestamp: u64) {
        self.values.push(value);
        self.timestamps.push(timestamp);
    }

    /// Get serialized size estimate
    pub fn size_bytes(&self) -> usize {
        self.name.len() + self.values.len() * 8 + self.timestamps.len() * 8
    }

    /// Compute delta from another metric
    pub fn delta_from(&self, other: &MetricData) -> DeltaMetric {
        let mut changed_indices = Vec::new();
        let mut changed_values = Vec::new();

        // Find changed values
        let max_len = self.values.len().max(other.values.len());
        for i in 0..max_len {
            let self_val = self.values.get(i).copied().unwrap_or(0.0);
            let other_val = other.values.get(i).copied().unwrap_or(0.0);

            if (self_val - other_val).abs() > 1e-10 {
                changed_indices.push(i);
                changed_values.push(self_val);
            }
        }

        DeltaMetric {
            name: self.name.clone(),
            base_len: other.values.len(),
            new_len: self.values.len(),
            changed_indices,
            changed_values,
        }
    }

    /// Apply delta to reconstruct
    pub fn apply_delta(&self, delta: &DeltaMetric) -> MetricData {
        let mut result = MetricData::new(&delta.name);

        // Start with base values
        result.values = self.values.clone();
        result.timestamps = self.timestamps.clone();

        // Resize if needed
        result.values.resize(delta.new_len, 0.0);
        result.timestamps.resize(delta.new_len, 0);

        // Apply changes
        for (i, &idx) in delta.changed_indices.iter().enumerate() {
            if idx < result.values.len() {
                result.values[idx] = delta.changed_values[i];
            }
        }

        result
    }
}

/// Delta-encoded metric
#[derive(Debug, Clone)]
pub struct DeltaMetric {
    /// Metric name
    pub name: String,
    /// Base array length
    pub base_len: usize,
    /// New array length
    pub new_len: usize,
    /// Indices of changed values
    pub changed_indices: Vec<usize>,
    /// Changed values
    pub changed_values: Vec<f64>,
}

impl DeltaMetric {
    /// Get compressed size estimate
    pub fn size_bytes(&self) -> usize {
        self.name.len() + 16 + self.changed_indices.len() * 8 + self.changed_values.len() * 8
    }

    /// Get compression ratio (compressed/original)
    pub fn compression_ratio(&self, original_size: usize) -> f64 {
        if original_size == 0 {
            1.0
        } else {
            self.size_bytes() as f64 / original_size as f64
        }
    }
}

/// A profile snapshot
#[derive(Debug, Clone)]
pub struct ProfileSnapshot {
    /// Snapshot index
    pub index: usize,
    /// Timestamp when snapshot was taken
    pub timestamp_ns: u64,
    /// Workload fingerprint for identification
    pub workload_fingerprint: String,
    /// Metric data
    pub metrics: HashMap<String, MetricData>,
    /// Snapshot metadata
    pub metadata: HashMap<String, String>,
    /// CRC32 checksum
    pub checksum: u32,
}

impl ProfileSnapshot {
    /// Create a new snapshot
    pub fn new(index: usize) -> Self {
        Self {
            index,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            workload_fingerprint: String::new(),
            metrics: HashMap::new(),
            metadata: HashMap::new(),
            checksum: 0,
        }
    }

    /// Add metric data
    pub fn add_metric(&mut self, metric: MetricData) {
        self.metrics.insert(metric.name.clone(), metric);
    }

    /// Get metric by name
    pub fn get_metric(&self, name: &str) -> Option<&MetricData> {
        self.metrics.get(name)
    }

    /// Set workload fingerprint
    pub fn set_fingerprint(&mut self, fingerprint: impl Into<String>) {
        self.workload_fingerprint = fingerprint.into();
    }

    /// Get total size in bytes
    pub fn size_bytes(&self) -> usize {
        let metrics_size: usize = self.metrics.values().map(|m| m.size_bytes()).sum();
        let metadata_size: usize = self.metadata.iter().map(|(k, v)| k.len() + v.len()).sum();

        24 + self.workload_fingerprint.len() + metrics_size + metadata_size
    }

    /// Compute checksum
    pub fn compute_checksum(&mut self) {
        // Simple CRC32-like checksum
        let mut hash: u32 = 0;

        hash = hash.wrapping_add(self.index as u32);
        hash = hash.wrapping_mul(31);
        hash = hash.wrapping_add((self.timestamp_ns & 0xFFFFFFFF) as u32);

        // Sort keys for deterministic iteration
        let mut keys: Vec<_> = self.metrics.keys().collect();
        keys.sort();

        for name in keys {
            if let Some(metric) = self.metrics.get(name) {
                for c in name.bytes() {
                    hash = hash.wrapping_mul(31).wrapping_add(c as u32);
                }
                for &val in &metric.values {
                    // Include both halves of f64 bits
                    let bits = val.to_bits();
                    hash = hash.wrapping_mul(31).wrapping_add((bits >> 32) as u32);
                    hash = hash
                        .wrapping_mul(31)
                        .wrapping_add((bits & 0xFFFFFFFF) as u32);
                }
            }
        }

        self.checksum = hash;
    }

    /// Verify checksum
    pub fn verify_checksum(&self) -> bool {
        let mut copy = self.clone();
        copy.compute_checksum();
        copy.checksum == self.checksum
    }
}

/// Delta-encoded snapshot
#[derive(Debug, Clone)]
pub struct DeltaSnapshot {
    /// Snapshot index
    pub index: usize,
    /// Base snapshot index
    pub base_index: usize,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Workload fingerprint
    pub workload_fingerprint: String,
    /// Delta metrics
    pub deltas: HashMap<String, DeltaMetric>,
    /// New metrics (not in base)
    pub new_metrics: HashMap<String, MetricData>,
    /// Removed metric names
    pub removed_metrics: Vec<String>,
    /// Checksum
    pub checksum: u32,
}

impl DeltaSnapshot {
    /// Create from two snapshots
    pub fn from_diff(base: &ProfileSnapshot, current: &ProfileSnapshot) -> Self {
        let mut deltas = HashMap::new();
        let mut new_metrics = HashMap::new();
        let mut removed_metrics = Vec::new();

        // Find deltas and new metrics
        for (name, metric) in &current.metrics {
            if let Some(base_metric) = base.metrics.get(name) {
                let delta = metric.delta_from(base_metric);
                // Only store delta if it's smaller than full metric
                if delta.size_bytes() < metric.size_bytes() {
                    deltas.insert(name.clone(), delta);
                } else {
                    new_metrics.insert(name.clone(), metric.clone());
                }
            } else {
                new_metrics.insert(name.clone(), metric.clone());
            }
        }

        // Find removed metrics
        for name in base.metrics.keys() {
            if !current.metrics.contains_key(name) {
                removed_metrics.push(name.clone());
            }
        }

        Self {
            index: current.index,
            base_index: base.index,
            timestamp_ns: current.timestamp_ns,
            workload_fingerprint: current.workload_fingerprint.clone(),
            deltas,
            new_metrics,
            removed_metrics,
            checksum: 0,
        }
    }

    /// Get compressed size
    pub fn size_bytes(&self) -> usize {
        let delta_size: usize = self.deltas.values().map(|d| d.size_bytes()).sum();
        let new_size: usize = self.new_metrics.values().map(|m| m.size_bytes()).sum();
        let removed_size: usize = self.removed_metrics.iter().map(|s| s.len()).sum();

        24 + self.workload_fingerprint.len() + delta_size + new_size + removed_size
    }

    /// Apply delta to base snapshot
    pub fn apply_to(&self, base: &ProfileSnapshot) -> ProfileSnapshot {
        let mut result = ProfileSnapshot::new(self.index);
        result.timestamp_ns = self.timestamp_ns;
        result.workload_fingerprint = self.workload_fingerprint.clone();

        // Copy base metrics and apply deltas
        for (name, metric) in &base.metrics {
            if self.removed_metrics.contains(name) {
                continue;
            }

            if let Some(delta) = self.deltas.get(name) {
                let reconstructed = metric.apply_delta(delta);
                result.metrics.insert(name.clone(), reconstructed);
            } else {
                result.metrics.insert(name.clone(), metric.clone());
            }
        }

        // Add new metrics
        for (name, metric) in &self.new_metrics {
            result.metrics.insert(name.clone(), metric.clone());
        }

        result.compute_checksum();
        result
    }
}

/// Index entry for fast lookup
#[derive(Debug, Clone)]
pub struct SnapshotIndex {
    /// Snapshot index
    pub index: usize,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Workload fingerprint
    pub fingerprint: String,
    /// Retention tier
    pub tier: RetentionTier,
    /// Offset in storage
    pub offset: usize,
    /// Size in bytes
    pub size_bytes: usize,
    /// Is delta encoded
    pub is_delta: bool,
    /// Base index (if delta)
    pub base_index: Option<usize>,
}

/// Query filter for snapshots
#[derive(Debug, Clone, Default)]
pub struct SnapshotQuery {
    /// Start timestamp (inclusive)
    pub start_ns: Option<u64>,
    /// End timestamp (inclusive)
    pub end_ns: Option<u64>,
    /// Workload fingerprint filter
    pub fingerprint: Option<String>,
    /// Metric name filter
    pub metric_name: Option<String>,
    /// Maximum results
    pub limit: Option<usize>,
}

impl SnapshotQuery {
    /// Create new query
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by time range
    pub fn time_range(mut self, start_ns: u64, end_ns: u64) -> Self {
        self.start_ns = Some(start_ns);
        self.end_ns = Some(end_ns);
        self
    }

    /// Filter by fingerprint
    pub fn fingerprint(mut self, fp: impl Into<String>) -> Self {
        self.fingerprint = Some(fp.into());
        self
    }

    /// Filter by metric name
    pub fn metric(mut self, name: impl Into<String>) -> Self {
        self.metric_name = Some(name.into());
        self
    }

    /// Limit results
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Check if index matches query
    pub fn matches(&self, idx: &SnapshotIndex) -> bool {
        if let Some(start) = self.start_ns {
            if idx.timestamp_ns < start {
                return false;
            }
        }
        if let Some(end) = self.end_ns {
            if idx.timestamp_ns > end {
                return false;
            }
        }
        if let Some(ref fp) = self.fingerprint {
            if &idx.fingerprint != fp {
                return false;
            }
        }
        true
    }
}

/// Configuration for snapshot storage
#[derive(Debug, Clone)]
pub struct SnapshotConfig {
    /// Maximum memory for query operations
    pub max_query_memory_bytes: usize,
    /// Keyframe interval (full snapshot every N deltas)
    pub keyframe_interval: usize,
    /// Enable checksum verification
    pub verify_checksums: bool,
    /// Raw tier max age
    pub raw_max_age: Duration,
    /// Compressed tier max age
    pub compressed_max_age: Duration,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            max_query_memory_bytes: 50 * 1024 * 1024, // 50MB
            keyframe_interval: 10,
            verify_checksums: true,
            raw_max_age: Duration::from_secs(24 * 60 * 60),
            compressed_max_age: Duration::from_secs(30 * 24 * 60 * 60),
        }
    }
}

/// Incremental snapshot store
#[derive(Debug)]
pub struct IncrementalSnapshotStore {
    /// Configuration
    config: SnapshotConfig,
    /// Snapshot index
    index: Vec<SnapshotIndex>,
    /// Keyframe snapshots (full snapshots)
    keyframes: HashMap<usize, ProfileSnapshot>,
    /// Delta snapshots
    deltas: HashMap<usize, DeltaSnapshot>,
    /// Total raw size (uncompressed)
    total_raw_size: usize,
    /// Total compressed size
    total_compressed_size: usize,
    /// Next snapshot index
    next_index: usize,
}

impl IncrementalSnapshotStore {
    /// Create a new snapshot store
    pub fn new(config: SnapshotConfig) -> Self {
        Self {
            config,
            index: Vec::new(),
            keyframes: HashMap::new(),
            deltas: HashMap::new(),
            total_raw_size: 0,
            total_compressed_size: 0,
            next_index: 0,
        }
    }

    /// Append a new snapshot
    pub fn append(&mut self, mut snapshot: ProfileSnapshot) -> SnapshotResult<usize> {
        snapshot.index = self.next_index;
        snapshot.compute_checksum();

        let raw_size = snapshot.size_bytes();
        self.total_raw_size += raw_size;

        // Decide if this should be a keyframe
        let is_keyframe = self
            .next_index
            .is_multiple_of(self.config.keyframe_interval)
            || self.keyframes.is_empty();

        let (compressed_size, is_delta, base_index) = if is_keyframe {
            // Store as keyframe
            let size = snapshot.size_bytes();
            self.keyframes.insert(self.next_index, snapshot.clone());
            (size, false, None)
        } else {
            // Find nearest keyframe
            let keyframe_idx =
                (self.next_index / self.config.keyframe_interval) * self.config.keyframe_interval;

            if let Some(base) = self.keyframes.get(&keyframe_idx) {
                // Create delta from keyframe
                let delta = DeltaSnapshot::from_diff(base, &snapshot);
                let size = delta.size_bytes();
                self.deltas.insert(self.next_index, delta);
                (size, true, Some(keyframe_idx))
            } else {
                // No keyframe found, store as keyframe
                let size = snapshot.size_bytes();
                self.keyframes.insert(self.next_index, snapshot.clone());
                (size, false, None)
            }
        };

        self.total_compressed_size += compressed_size;

        // Add index entry
        self.index.push(SnapshotIndex {
            index: self.next_index,
            timestamp_ns: snapshot.timestamp_ns,
            fingerprint: snapshot.workload_fingerprint.clone(),
            tier: RetentionTier::Raw,
            offset: 0, // In-memory, no offset
            size_bytes: compressed_size,
            is_delta,
            base_index,
        });

        let idx = self.next_index;
        self.next_index += 1;

        Ok(idx)
    }

    /// Get snapshot by index
    pub fn get(&self, index: usize) -> SnapshotResult<ProfileSnapshot> {
        if index >= self.next_index {
            return Err(SnapshotError::IndexOutOfBounds {
                index,
                max: self.next_index.saturating_sub(1),
            });
        }

        // Check if it's a keyframe
        if let Some(snapshot) = self.keyframes.get(&index) {
            return Ok(snapshot.clone());
        }

        // It's a delta, need to reconstruct
        if let Some(delta) = self.deltas.get(&index) {
            if let Some(base) = self.keyframes.get(&delta.base_index) {
                let reconstructed = delta.apply_to(base);

                if self.config.verify_checksums && !reconstructed.verify_checksum() {
                    return Err(SnapshotError::Corrupt {
                        reason: "Checksum verification failed".to_string(),
                    });
                }

                return Ok(reconstructed);
            }
        }

        Err(SnapshotError::NotFound { index })
    }

    /// Query snapshots
    pub fn query(&self, query: &SnapshotQuery) -> SnapshotResult<Vec<ProfileSnapshot>> {
        let mut results = Vec::new();
        let mut memory_used = 0;

        for idx_entry in &self.index {
            if !query.matches(idx_entry) {
                continue;
            }

            // Check memory limit
            if memory_used + idx_entry.size_bytes > self.config.max_query_memory_bytes {
                break;
            }

            let snapshot = self.get(idx_entry.index)?;

            // Check metric filter
            if let Some(ref metric_name) = query.metric_name {
                if !snapshot.metrics.contains_key(metric_name) {
                    continue;
                }
            }

            memory_used += idx_entry.size_bytes;
            results.push(snapshot);

            // Check limit
            if let Some(limit) = query.limit {
                if results.len() >= limit {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.total_raw_size == 0 {
            1.0
        } else {
            self.total_compressed_size as f64 / self.total_raw_size as f64
        }
    }

    /// Get snapshot count
    pub fn count(&self) -> usize {
        self.next_index
    }

    /// Get total raw size
    pub fn total_raw_size(&self) -> usize {
        self.total_raw_size
    }

    /// Get total compressed size
    pub fn total_compressed_size(&self) -> usize {
        self.total_compressed_size
    }

    /// Clean up old snapshots based on retention policy
    pub fn cleanup(&mut self, _now: Instant, reference_time: u64) {
        let _raw_cutoff = reference_time.saturating_sub(self.config.raw_max_age.as_nanos() as u64);
        let compressed_cutoff =
            reference_time.saturating_sub(self.config.compressed_max_age.as_nanos() as u64);

        // Update tiers and remove expired
        self.index.retain(|idx| {
            if idx.timestamp_ns < compressed_cutoff {
                // Remove from storage
                self.keyframes.remove(&idx.index);
                self.deltas.remove(&idx.index);
                return false;
            }
            true
        });
    }

    /// Get configuration
    pub fn config(&self) -> &SnapshotConfig {
        &self.config
    }

    /// Get index entries
    pub fn index(&self) -> &[SnapshotIndex] {
        &self.index
    }
}

/// Default keyframe interval
pub const DEFAULT_KEYFRAME_INTERVAL: usize = 10;

/// Default max query memory (bytes)
pub const DEFAULT_MAX_QUERY_MEMORY: usize = 50 * 1024 * 1024;


#[cfg(test)]
mod tests;
