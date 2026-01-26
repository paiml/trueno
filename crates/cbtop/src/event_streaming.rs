//! Structured Event Streaming (PMAT-043)
//!
//! Stream metrics to time-series databases and event systems.
//!
//! # Features
//!
//! - InfluxDB Line Protocol export
//! - JSON Lines file sink
//! - Batch buffering and compression
//! - Correlation ID tracking
//!
//! # Falsification Criteria (F1351-F1360)
//!
//! See `tests/event_streaming_f1351.rs` for falsification tests.

use std::collections::HashMap;

/// Schema version for event format
pub const SCHEMA_VERSION: u32 = 1;

/// Default batch size
pub const DEFAULT_BATCH_SIZE: usize = 100;

/// Event sink type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SinkType {
    /// InfluxDB Line Protocol
    InfluxDb,
    /// JSON Lines file
    JsonLines,
    /// Kafka (simulated)
    Kafka,
    /// Console (for testing)
    Console,
}

impl SinkType {
    /// Get sink name
    pub fn name(&self) -> &'static str {
        match self {
            Self::InfluxDb => "influxdb",
            Self::JsonLines => "jsonlines",
            Self::Kafka => "kafka",
            Self::Console => "console",
        }
    }
}

/// Metric event
#[derive(Debug, Clone)]
pub struct MetricEvent {
    /// Measurement name
    pub measurement: String,
    /// Tags (indexed fields)
    pub tags: HashMap<String, String>,
    /// Fields (values)
    pub fields: HashMap<String, f64>,
    /// Timestamp (nanoseconds)
    pub timestamp_ns: u64,
    /// Correlation ID for tracing
    pub correlation_id: Option<String>,
    /// Schema version
    pub schema_version: u32,
}

impl MetricEvent {
    /// Create new event
    pub fn new(measurement: &str) -> Self {
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        Self {
            measurement: measurement.to_string(),
            tags: HashMap::new(),
            fields: HashMap::new(),
            timestamp_ns,
            correlation_id: None,
            schema_version: SCHEMA_VERSION,
        }
    }

    /// Add tag
    pub fn with_tag(mut self, key: &str, value: &str) -> Self {
        self.tags.insert(key.to_string(), value.to_string());
        self
    }

    /// Add field
    pub fn with_field(mut self, key: &str, value: f64) -> Self {
        self.fields.insert(key.to_string(), value);
        self
    }

    /// Set correlation ID
    pub fn with_correlation_id(mut self, id: &str) -> Self {
        self.correlation_id = Some(id.to_string());
        self
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp_ns: u64) -> Self {
        self.timestamp_ns = timestamp_ns;
        self
    }

    /// Format as InfluxDB Line Protocol
    pub fn to_influx_line(&self) -> String {
        // measurement,tag1=val1,tag2=val2 field1=val1,field2=val2 timestamp
        let mut line = self.measurement.clone();

        // Add tags (sorted for consistency)
        let mut tag_pairs: Vec<_> = self.tags.iter().collect();
        tag_pairs.sort_by_key(|(k, _)| *k);
        for (key, value) in tag_pairs {
            line.push_str(&format!(",{}={}", escape_influx(key), escape_influx(value)));
        }

        // Add fields
        line.push(' ');
        let mut field_pairs: Vec<_> = self.fields.iter().collect();
        field_pairs.sort_by_key(|(k, _)| *k);
        let field_str: Vec<String> = field_pairs
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        line.push_str(&field_str.join(","));

        // Add timestamp
        line.push_str(&format!(" {}", self.timestamp_ns));

        line
    }

    /// Format as JSON
    pub fn to_json(&self) -> String {
        let tags_json: Vec<String> = self
            .tags
            .iter()
            .map(|(k, v)| format!("\"{}\":\"{}\"", k, v))
            .collect();

        let fields_json: Vec<String> = self
            .fields
            .iter()
            .map(|(k, v)| format!("\"{}\":{}", k, v))
            .collect();

        let correlation = self
            .correlation_id
            .as_ref()
            .map(|id| format!(",\"correlation_id\":\"{}\"", id))
            .unwrap_or_default();

        format!(
            r#"{{"measurement":"{}","tags":{{{}}},"fields":{{{}}},"timestamp_ns":{},"schema_version":{}{}}}"#,
            self.measurement,
            tags_json.join(","),
            fields_json.join(","),
            self.timestamp_ns,
            self.schema_version,
            correlation
        )
    }
}

/// Escape string for InfluxDB Line Protocol
fn escape_influx(s: &str) -> String {
    s.replace(' ', "\\ ")
        .replace(',', "\\,")
        .replace('=', "\\=")
}

/// Event batch
#[derive(Debug, Clone)]
pub struct EventBatch {
    /// Events in batch
    pub events: Vec<MetricEvent>,
    /// Batch ID
    pub batch_id: u64,
    /// Created timestamp
    pub created_ns: u64,
}

impl EventBatch {
    /// Create new batch
    pub fn new(batch_id: u64) -> Self {
        let created_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        Self {
            events: Vec::new(),
            batch_id,
            created_ns,
        }
    }

    /// Add event
    pub fn add(&mut self, event: MetricEvent) {
        self.events.push(event);
    }

    /// Get batch size
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Format as InfluxDB batch
    pub fn to_influx_batch(&self) -> String {
        self.events
            .iter()
            .map(|e| e.to_influx_line())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Format as JSON Lines
    pub fn to_json_lines(&self) -> String {
        self.events
            .iter()
            .map(|e| e.to_json())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Sink health status
#[derive(Debug, Clone)]
pub struct SinkHealth {
    /// Is connected
    pub connected: bool,
    /// Last successful write timestamp
    pub last_write_ns: Option<u64>,
    /// Events written
    pub events_written: u64,
    /// Write errors
    pub write_errors: u64,
}

impl Default for SinkHealth {
    fn default() -> Self {
        Self {
            connected: true,
            last_write_ns: None,
            events_written: 0,
            write_errors: 0,
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Max retries
    pub max_retries: u32,
    /// Initial delay (ms)
    pub initial_delay_ms: u64,
    /// Max delay (ms)
    pub max_delay_ms: u64,
    /// Backoff multiplier
    pub multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 10000,
            multiplier: 2.0,
        }
    }
}

impl RetryConfig {
    /// Calculate delay for attempt
    pub fn delay_for_attempt(&self, attempt: u32) -> u64 {
        let delay = self.initial_delay_ms as f64 * self.multiplier.powi(attempt as i32);
        (delay as u64).min(self.max_delay_ms)
    }
}

/// Event streamer
#[derive(Debug)]
pub struct EventStreamer {
    /// Sink type
    sink_type: SinkType,
    /// Batch size
    batch_size: usize,
    /// Current batch
    current_batch: EventBatch,
    /// Batch counter
    batch_counter: u64,
    /// Retry config
    retry_config: RetryConfig,
    /// Health status
    health: SinkHealth,
    /// Enable compression
    compression: bool,
    /// Correlation ID generator counter
    correlation_counter: u64,
    /// Output buffer (for file sink)
    output_buffer: Vec<String>,
}

impl Default for EventStreamer {
    fn default() -> Self {
        Self::new(SinkType::Console)
    }
}

impl EventStreamer {
    /// Create new streamer
    pub fn new(sink_type: SinkType) -> Self {
        Self {
            sink_type,
            batch_size: DEFAULT_BATCH_SIZE,
            current_batch: EventBatch::new(0),
            batch_counter: 0,
            retry_config: RetryConfig::default(),
            health: SinkHealth::default(),
            compression: false,
            correlation_counter: 0,
            output_buffer: Vec::new(),
        }
    }

    /// Set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size.max(1);
        self
    }

    /// Enable compression
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression = enabled;
        self
    }

    /// Set retry config
    pub fn with_retry(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    /// Generate correlation ID
    pub fn generate_correlation_id(&mut self) -> String {
        self.correlation_counter += 1;
        format!("cbtop-{}-{}", std::process::id(), self.correlation_counter)
    }

    /// Send event
    pub fn send(&mut self, event: MetricEvent) -> bool {
        self.current_batch.add(event);

        if self.current_batch.len() >= self.batch_size {
            self.flush()
        } else {
            true
        }
    }

    /// Flush current batch
    pub fn flush(&mut self) -> bool {
        if self.current_batch.is_empty() {
            return true;
        }

        let result = self.write_batch(&self.current_batch.clone());

        if result {
            self.health.events_written += self.current_batch.len() as u64;
            self.health.last_write_ns = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0),
            );

            // Start new batch
            self.batch_counter += 1;
            self.current_batch = EventBatch::new(self.batch_counter);
            true
        } else {
            self.health.write_errors += 1;
            false
        }
    }

    /// Write batch to sink
    fn write_batch(&mut self, batch: &EventBatch) -> bool {
        match self.sink_type {
            SinkType::Console => {
                for event in &batch.events {
                    println!("{}", event.to_json());
                }
                true
            }
            SinkType::InfluxDb => {
                // In production, this would make HTTP request
                // For now, store in buffer
                self.output_buffer.push(batch.to_influx_batch());
                true
            }
            SinkType::JsonLines => {
                self.output_buffer.push(batch.to_json_lines());
                true
            }
            SinkType::Kafka => {
                // Simulated Kafka produce
                self.output_buffer.push(batch.to_json_lines());
                true
            }
        }
    }

    /// Get health status
    pub fn health(&self) -> &SinkHealth {
        &self.health
    }

    /// Check if healthy
    pub fn is_healthy(&self) -> bool {
        self.health.connected && self.health.write_errors == 0
    }

    /// Get output buffer (for testing)
    pub fn output_buffer(&self) -> &[String] {
        &self.output_buffer
    }

    /// Clear output buffer
    pub fn clear_buffer(&mut self) {
        self.output_buffer.clear();
    }

    /// Get events written count
    pub fn events_written(&self) -> u64 {
        self.health.events_written
    }

    /// Get pending events count
    pub fn pending_count(&self) -> usize {
        self.current_batch.len()
    }

    /// Graceful shutdown - flush remaining events
    pub fn shutdown(&mut self) -> bool {
        self.flush()
    }
}

/// Compress data using simple run-length encoding (placeholder)
pub fn compress_data(data: &[u8]) -> Vec<u8> {
    // In production, use flate2 or lz4
    // For now, just return as-is
    data.to_vec()
}

/// Create event from performance sample
pub fn event_from_sample(metric: &str, value: f64, tags: &[(&str, &str)]) -> MetricEvent {
    let mut event = MetricEvent::new(metric).with_field("value", value);

    for (key, val) in tags {
        event = event.with_tag(key, val);
    }

    event
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sink_type_names() {
        assert_eq!(SinkType::InfluxDb.name(), "influxdb");
        assert_eq!(SinkType::JsonLines.name(), "jsonlines");
        assert_eq!(SinkType::Kafka.name(), "kafka");
    }

    #[test]
    fn test_metric_event_creation() {
        let event = MetricEvent::new("cpu_usage")
            .with_tag("host", "server1")
            .with_field("value", 85.5);

        assert_eq!(event.measurement, "cpu_usage");
        assert_eq!(event.tags.get("host"), Some(&"server1".to_string()));
        assert_eq!(event.fields.get("value"), Some(&85.5));
    }

    #[test]
    fn test_influx_line_protocol() {
        let event = MetricEvent::new("cpu")
            .with_tag("host", "server1")
            .with_field("usage", 85.5)
            .with_timestamp(1234567890000000000);

        let line = event.to_influx_line();
        assert!(line.starts_with("cpu,host=server1"));
        assert!(line.contains("usage=85.5"));
        assert!(line.ends_with("1234567890000000000"));
    }

    #[test]
    fn test_event_json() {
        let event = MetricEvent::new("test")
            .with_tag("env", "prod")
            .with_field("value", 42.0);

        let json = event.to_json();
        assert!(json.contains("\"measurement\":\"test\""));
        assert!(json.contains("\"env\":\"prod\""));
        assert!(json.contains("\"value\":42"));
    }

    #[test]
    fn test_correlation_id() {
        let event = MetricEvent::new("test").with_correlation_id("trace-123");

        assert_eq!(event.correlation_id, Some("trace-123".to_string()));
        assert!(event.to_json().contains("correlation_id"));
    }

    #[test]
    fn test_event_batch() {
        let mut batch = EventBatch::new(1);
        batch.add(MetricEvent::new("test1"));
        batch.add(MetricEvent::new("test2"));

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_influx_format() {
        let mut batch = EventBatch::new(1);
        batch.add(MetricEvent::new("cpu").with_field("value", 50.0));
        batch.add(MetricEvent::new("mem").with_field("value", 70.0));

        let output = batch.to_influx_batch();
        assert!(output.contains("cpu"));
        assert!(output.contains("mem"));
    }

    #[test]
    fn test_streamer_send() {
        let mut streamer = EventStreamer::new(SinkType::JsonLines).with_batch_size(5);

        for i in 0..3 {
            let event = MetricEvent::new("test").with_field("i", i as f64);
            streamer.send(event);
        }

        assert_eq!(streamer.pending_count(), 3);
    }

    #[test]
    fn test_streamer_flush() {
        let mut streamer = EventStreamer::new(SinkType::JsonLines).with_batch_size(2);

        streamer.send(MetricEvent::new("test"));
        streamer.send(MetricEvent::new("test")); // Triggers flush

        assert!(streamer.events_written() >= 2);
    }

    #[test]
    fn test_streamer_health() {
        let streamer = EventStreamer::new(SinkType::Console);

        assert!(streamer.is_healthy());
        assert!(streamer.health().connected);
    }

    #[test]
    fn test_retry_config() {
        let config = RetryConfig::default();

        assert_eq!(config.delay_for_attempt(0), 100);
        assert_eq!(config.delay_for_attempt(1), 200);
        assert_eq!(config.delay_for_attempt(2), 400);
    }

    #[test]
    fn test_correlation_id_generation() {
        let mut streamer = EventStreamer::new(SinkType::Console);

        let id1 = streamer.generate_correlation_id();
        let id2 = streamer.generate_correlation_id();

        assert_ne!(id1, id2);
        assert!(id1.contains("cbtop"));
    }

    #[test]
    fn test_graceful_shutdown() {
        let mut streamer = EventStreamer::new(SinkType::JsonLines);

        streamer.send(MetricEvent::new("test"));
        let result = streamer.shutdown();

        assert!(result);
        assert_eq!(streamer.pending_count(), 0);
    }

    #[test]
    fn test_event_from_sample() {
        let event = event_from_sample(
            "latency_ms",
            42.5,
            &[("host", "server1"), ("region", "us-west")],
        );

        assert_eq!(event.measurement, "latency_ms");
        assert_eq!(event.fields.get("value"), Some(&42.5));
        assert_eq!(event.tags.get("host"), Some(&"server1".to_string()));
    }

    #[test]
    fn test_escape_influx() {
        assert_eq!(escape_influx("hello world"), "hello\\ world");
        assert_eq!(escape_influx("key=value"), "key\\=value");
        assert_eq!(escape_influx("a,b,c"), "a\\,b\\,c");
    }
}
