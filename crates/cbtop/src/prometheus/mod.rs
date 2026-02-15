//! Prometheus Metrics Exporter (PMAT-041)
//!
//! Native Prometheus `/metrics` endpoint for monitoring integration.
//!
//! # Features
//!
//! - Gauge, Counter, and Histogram metric types
//! - Label support with cardinality limits
//! - Prometheus text format (v0.0.4) compliance
//! - HTTP endpoint export
//!
//! # Falsification Criteria (F1331-F1340)
//!
//! See `tests/prometheus_f1331.rs` for falsification tests.

mod registry;
mod types;

pub use registry::MetricsRegistry;
pub use types::{
    CounterValue, GaugeValue, HistogramBuckets, HistogramValue, Labels, MetricDef, MetricType,
    escape_label_value, validate_metric_name, DEFAULT_BUCKETS,
    DEFAULT_MAX_LABELS,
};


#[cfg(test)]
mod tests;
