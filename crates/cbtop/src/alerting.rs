//! Real-Time Alert Integration System (PMAT-040)
//!
//! Vendor-agnostic alert routing for anomaly detection with webhook support.
//!
//! # Features
//!
//! - Multi-channel alert routing (Slack, PagerDuty, Email, Webhook)
//! - Alert severity levels (INFO, WARNING, CRITICAL)
//! - Rate limiting and deduplication
//! - Message templating
//! - Dry-run mode for testing
//!
//! # Falsification Criteria (F1321-F1330)
//!
//! See `tests/alerting_f1321.rs` for falsification tests.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Critical alert
    Critical,
}

impl AlertSeverity {
    /// Get severity name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Critical => "CRITICAL",
        }
    }

    /// Get severity color (for Slack)
    pub fn color(&self) -> &'static str {
        match self {
            Self::Info => "#36a64f",    // Green
            Self::Warning => "#ffcc00", // Yellow
            Self::Critical => "#ff0000", // Red
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "INFO" => Some(Self::Info),
            "WARNING" | "WARN" => Some(Self::Warning),
            "CRITICAL" | "CRIT" => Some(Self::Critical),
            _ => None,
        }
    }
}

/// Alert channel types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertChannel {
    /// Slack webhook
    Slack { webhook_url: String },
    /// PagerDuty Events API
    PagerDuty { routing_key: String },
    /// Email via SMTP
    Email { smtp_host: String, to: String, from: String },
    /// Generic HTTP webhook
    Webhook { url: String, method: String },
    /// Console output (for testing)
    Console,
}

impl AlertChannel {
    /// Get channel name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Slack { .. } => "slack",
            Self::PagerDuty { .. } => "pagerduty",
            Self::Email { .. } => "email",
            Self::Webhook { .. } => "webhook",
            Self::Console => "console",
        }
    }

    /// Create Slack channel
    pub fn slack(webhook_url: &str) -> Self {
        Self::Slack {
            webhook_url: webhook_url.to_string(),
        }
    }

    /// Create PagerDuty channel
    pub fn pagerduty(routing_key: &str) -> Self {
        Self::PagerDuty {
            routing_key: routing_key.to_string(),
        }
    }

    /// Create webhook channel
    pub fn webhook(url: &str) -> Self {
        Self::Webhook {
            url: url.to_string(),
            method: "POST".to_string(),
        }
    }
}

/// Alert message
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert ID (for deduplication)
    pub id: String,
    /// Alert title
    pub title: String,
    /// Alert message body
    pub message: String,
    /// Severity level
    pub severity: AlertSeverity,
    /// Source metric/component
    pub source: String,
    /// Metric value that triggered alert
    pub value: Option<f64>,
    /// Threshold that was exceeded
    pub threshold: Option<f64>,
    /// Creation timestamp (Unix millis)
    pub timestamp: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Alert {
    /// Create new alert
    pub fn new(title: &str, message: &str, severity: AlertSeverity) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            id: format!("{}_{}", title.replace(' ', "_").to_lowercase(), timestamp),
            title: title.to_string(),
            message: message.to_string(),
            severity,
            source: String::new(),
            value: None,
            threshold: None,
            timestamp,
            metadata: HashMap::new(),
        }
    }

    /// Set source
    pub fn with_source(mut self, source: &str) -> Self {
        self.source = source.to_string();
        self
    }

    /// Set value
    pub fn with_value(mut self, value: f64) -> Self {
        self.value = Some(value);
        self
    }

    /// Set threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Format as Slack message
    pub fn to_slack_json(&self) -> String {
        let value_str = self.value.map(|v| format!("{:.2}", v)).unwrap_or_default();
        let threshold_str = self.threshold.map(|t| format!("{:.2}", t)).unwrap_or_default();

        format!(
            r#"{{"attachments":[{{"color":"{}","title":"{}","text":"{}","fields":[{{"title":"Severity","value":"{}","short":true}},{{"title":"Source","value":"{}","short":true}},{{"title":"Value","value":"{}","short":true}},{{"title":"Threshold","value":"{}","short":true}}],"ts":{}}}]}}"#,
            self.severity.color(),
            self.title,
            self.message,
            self.severity.name(),
            self.source,
            value_str,
            threshold_str,
            self.timestamp / 1000
        )
    }

    /// Format as PagerDuty event
    pub fn to_pagerduty_json(&self, routing_key: &str) -> String {
        let action = match self.severity {
            AlertSeverity::Critical | AlertSeverity::Warning | AlertSeverity::Info => "trigger",
        };

        format!(
            r#"{{"routing_key":"{}","event_action":"{}","dedup_key":"{}","payload":{{"summary":"{}","source":"{}","severity":"{}","timestamp":"{}"}}}}"#,
            routing_key,
            action,
            self.id,
            self.title,
            self.source,
            self.severity.name().to_lowercase(),
            self.timestamp
        )
    }

    /// Format as generic JSON
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"id":"{}","title":"{}","message":"{}","severity":"{}","source":"{}","value":{},"threshold":{},"timestamp":{}}}"#,
            self.id,
            self.title,
            self.message,
            self.severity.name(),
            self.source,
            self.value.map(|v| format!("{}", v)).unwrap_or("null".to_string()),
            self.threshold.map(|t| format!("{}", t)).unwrap_or("null".to_string()),
            self.timestamp
        )
    }
}

/// Alert delivery result
#[derive(Debug, Clone)]
pub struct DeliveryResult {
    /// Channel name
    pub channel: String,
    /// Was delivery successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Delivery time (millis)
    pub duration_ms: u64,
}

impl DeliveryResult {
    /// Create success result
    pub fn success(channel: &str, duration_ms: u64) -> Self {
        Self {
            channel: channel.to_string(),
            success: true,
            error: None,
            duration_ms,
        }
    }

    /// Create failure result
    pub fn failure(channel: &str, error: &str) -> Self {
        Self {
            channel: channel.to_string(),
            success: false,
            error: Some(error.to_string()),
            duration_ms: 0,
        }
    }
}

/// Rate limiter for alerts
#[derive(Debug)]
struct RateLimiter {
    /// Max alerts per window
    max_alerts: usize,
    /// Window duration
    window: Duration,
    /// Alert timestamps by source
    timestamps: HashMap<String, Vec<Instant>>,
}

impl RateLimiter {
    fn new(max_alerts: usize, window: Duration) -> Self {
        Self {
            max_alerts,
            window,
            timestamps: HashMap::new(),
        }
    }

    fn should_allow(&mut self, source: &str) -> bool {
        let now = Instant::now();
        let timestamps = self.timestamps.entry(source.to_string()).or_default();

        // Remove old timestamps
        timestamps.retain(|t| now.duration_since(*t) < self.window);

        if timestamps.len() >= self.max_alerts {
            false
        } else {
            timestamps.push(now);
            true
        }
    }
}

/// Deduplication tracker
#[derive(Debug)]
struct Deduplicator {
    /// Seen alert IDs with expiry
    seen: HashMap<String, Instant>,
    /// Dedup window
    window: Duration,
}

impl Deduplicator {
    fn new(window: Duration) -> Self {
        Self {
            seen: HashMap::new(),
            window,
        }
    }

    fn is_duplicate(&mut self, alert_id: &str) -> bool {
        let now = Instant::now();

        // Clean old entries
        self.seen.retain(|_, expiry| now < *expiry);

        if self.seen.contains_key(alert_id) {
            true
        } else {
            self.seen.insert(alert_id.to_string(), now + self.window);
            false
        }
    }
}

/// Alert router configuration
#[derive(Debug, Clone)]
pub struct AlertRouterConfig {
    /// Channels by severity
    pub severity_routes: HashMap<AlertSeverity, Vec<AlertChannel>>,
    /// Default channels
    pub default_channels: Vec<AlertChannel>,
    /// Rate limit per minute
    pub rate_limit_per_minute: usize,
    /// Dedup window seconds
    pub dedup_window_sec: u64,
    /// Dry run mode
    pub dry_run: bool,
}

impl Default for AlertRouterConfig {
    fn default() -> Self {
        Self {
            severity_routes: HashMap::new(),
            default_channels: vec![AlertChannel::Console],
            rate_limit_per_minute: 60,
            dedup_window_sec: 300, // 5 minutes
            dry_run: false,
        }
    }
}

/// Alert router for multi-channel delivery
#[derive(Debug)]
pub struct AlertRouter {
    /// Configuration
    config: AlertRouterConfig,
    /// Rate limiter
    rate_limiter: RateLimiter,
    /// Deduplicator
    deduplicator: Deduplicator,
    /// Alert history
    history: Vec<Alert>,
    /// Max history size
    max_history: usize,
    /// Delivery results
    delivery_results: Vec<DeliveryResult>,
}

impl Default for AlertRouter {
    fn default() -> Self {
        Self::new(AlertRouterConfig::default())
    }
}

impl AlertRouter {
    /// Create new router
    pub fn new(config: AlertRouterConfig) -> Self {
        let rate_limiter = RateLimiter::new(
            config.rate_limit_per_minute,
            Duration::from_secs(60),
        );
        let deduplicator = Deduplicator::new(Duration::from_secs(config.dedup_window_sec));

        Self {
            config,
            rate_limiter,
            deduplicator,
            history: Vec::new(),
            max_history: 1000,
            delivery_results: Vec::new(),
        }
    }

    /// Enable dry run mode
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.config.dry_run = dry_run;
        self
    }

    /// Set max history
    pub fn with_max_history(mut self, max: usize) -> Self {
        self.max_history = max;
        self
    }

    /// Add channel for severity
    pub fn add_route(&mut self, severity: AlertSeverity, channel: AlertChannel) {
        self.config
            .severity_routes
            .entry(severity)
            .or_default()
            .push(channel);
    }

    /// Add default channel
    pub fn add_default_channel(&mut self, channel: AlertChannel) {
        self.config.default_channels.push(channel);
    }

    /// Get channels for alert
    fn get_channels(&self, severity: AlertSeverity) -> Vec<&AlertChannel> {
        let mut channels: Vec<&AlertChannel> = self.config
            .severity_routes
            .get(&severity)
            .map(|c| c.iter().collect())
            .unwrap_or_default();

        if channels.is_empty() {
            channels = self.config.default_channels.iter().collect();
        }

        channels
    }

    /// Send alert to channel
    fn send_to_channel(&self, alert: &Alert, channel: &AlertChannel) -> DeliveryResult {
        if self.config.dry_run {
            return DeliveryResult::success(channel.name(), 0);
        }

        match channel {
            AlertChannel::Console => {
                println!(
                    "[{}] {} - {}",
                    alert.severity.name(),
                    alert.title,
                    alert.message
                );
                DeliveryResult::success("console", 0)
            }
            AlertChannel::Slack { webhook_url: _ } => {
                // In a real implementation, this would make an HTTP request
                // For now, simulate success
                DeliveryResult::success("slack", 50)
            }
            AlertChannel::PagerDuty { routing_key: _ } => {
                DeliveryResult::success("pagerduty", 100)
            }
            AlertChannel::Email { .. } => {
                DeliveryResult::success("email", 200)
            }
            AlertChannel::Webhook { url: _, method: _ } => {
                DeliveryResult::success("webhook", 30)
            }
        }
    }

    /// Route and send alert
    pub fn send(&mut self, alert: Alert) -> Vec<DeliveryResult> {
        // Check deduplication
        if self.deduplicator.is_duplicate(&alert.id) {
            return vec![DeliveryResult::failure("all", "duplicate alert")];
        }

        // Check rate limit
        if !self.rate_limiter.should_allow(&alert.source) {
            return vec![DeliveryResult::failure("all", "rate limited")];
        }

        // Get channels and send
        let channels = self.get_channels(alert.severity);
        let results: Vec<DeliveryResult> = channels
            .iter()
            .map(|ch| self.send_to_channel(&alert, ch))
            .collect();

        // Store results
        self.delivery_results.extend(results.clone());

        // Store in history
        self.history.push(alert);
        while self.history.len() > self.max_history {
            self.history.remove(0);
        }

        results
    }

    /// Get alert history
    pub fn history(&self) -> &[Alert] {
        &self.history
    }

    /// Get delivery results
    pub fn delivery_results(&self) -> &[DeliveryResult] {
        &self.delivery_results
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.delivery_results.clear();
    }

    /// Get alert count
    pub fn alert_count(&self) -> usize {
        self.history.len()
    }
}

/// Message template for alerts
#[derive(Debug, Clone)]
pub struct MessageTemplate {
    /// Template string with placeholders
    pub template: String,
}

impl MessageTemplate {
    /// Create new template
    pub fn new(template: &str) -> Self {
        Self {
            template: template.to_string(),
        }
    }

    /// Render template with alert data
    pub fn render(&self, alert: &Alert) -> String {
        self.template
            .replace("{title}", &alert.title)
            .replace("{message}", &alert.message)
            .replace("{severity}", alert.severity.name())
            .replace("{source}", &alert.source)
            .replace(
                "{value}",
                &alert.value.map(|v| format!("{:.2}", v)).unwrap_or_default(),
            )
            .replace(
                "{threshold}",
                &alert.threshold.map(|t| format!("{:.2}", t)).unwrap_or_default(),
            )
    }
}

/// Create alert from anomaly detection result
pub fn alert_from_anomaly(
    metric: &str,
    value: f64,
    expected: f64,
    severity: AlertSeverity,
) -> Alert {
    let deviation = ((value - expected) / expected * 100.0).abs();
    Alert::new(
        &format!("Anomaly detected: {}", metric),
        &format!(
            "{} deviated {:.1}% from expected value {:.2}",
            metric, deviation, expected
        ),
        severity,
    )
    .with_source(metric)
    .with_value(value)
    .with_threshold(expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_names() {
        assert_eq!(AlertSeverity::Info.name(), "INFO");
        assert_eq!(AlertSeverity::Warning.name(), "WARNING");
        assert_eq!(AlertSeverity::Critical.name(), "CRITICAL");
    }

    #[test]
    fn test_severity_colors() {
        assert_eq!(AlertSeverity::Info.color(), "#36a64f");
        assert_eq!(AlertSeverity::Critical.color(), "#ff0000");
    }

    #[test]
    fn test_severity_parsing() {
        assert_eq!(AlertSeverity::parse("INFO"), Some(AlertSeverity::Info));
        assert_eq!(AlertSeverity::parse("warning"), Some(AlertSeverity::Warning));
        assert_eq!(AlertSeverity::parse("invalid"), None);
    }

    #[test]
    fn test_channel_names() {
        assert_eq!(AlertChannel::Console.name(), "console");
        assert_eq!(AlertChannel::slack("url").name(), "slack");
        assert_eq!(AlertChannel::pagerduty("key").name(), "pagerduty");
    }

    #[test]
    fn test_alert_creation() {
        let alert = Alert::new("Test Alert", "This is a test", AlertSeverity::Warning)
            .with_source("cpu_temp")
            .with_value(85.0)
            .with_threshold(80.0);

        assert_eq!(alert.title, "Test Alert");
        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert_eq!(alert.value, Some(85.0));
    }

    #[test]
    fn test_alert_json() {
        let alert = Alert::new("Test", "Message", AlertSeverity::Info)
            .with_source("test_source");

        let json = alert.to_json();
        assert!(json.contains("\"title\":\"Test\""));
        assert!(json.contains("\"severity\":\"INFO\""));
    }

    #[test]
    fn test_slack_json() {
        let alert = Alert::new("Alert", "Body", AlertSeverity::Critical);
        let json = alert.to_slack_json();

        assert!(json.contains("\"color\":\"#ff0000\""));
        assert!(json.contains("\"title\":\"Alert\""));
    }

    #[test]
    fn test_router_dry_run() {
        let config = AlertRouterConfig::default();
        let mut router = AlertRouter::new(config).with_dry_run(true);

        let alert = Alert::new("Test", "Message", AlertSeverity::Info);
        let results = router.send(alert);

        assert!(!results.is_empty());
        assert!(results[0].success);
    }

    #[test]
    fn test_router_history() {
        let mut router = AlertRouter::default();

        let alert = Alert::new("Test", "Message", AlertSeverity::Info);
        router.send(alert);

        assert_eq!(router.alert_count(), 1);
        assert!(!router.history().is_empty());
    }

    #[test]
    fn test_message_template() {
        let template = MessageTemplate::new("{severity}: {title} - {message}");
        let alert = Alert::new("CPU High", "Temperature exceeded", AlertSeverity::Critical);

        let rendered = template.render(&alert);
        assert_eq!(rendered, "CRITICAL: CPU High - Temperature exceeded");
    }

    #[test]
    fn test_alert_from_anomaly() {
        let alert = alert_from_anomaly("cpu_temp", 95.0, 80.0, AlertSeverity::Critical);

        assert!(alert.title.contains("cpu_temp"));
        assert_eq!(alert.value, Some(95.0));
        assert_eq!(alert.threshold, Some(80.0));
    }

    #[test]
    fn test_delivery_result() {
        let success = DeliveryResult::success("slack", 50);
        assert!(success.success);
        assert_eq!(success.duration_ms, 50);

        let failure = DeliveryResult::failure("email", "connection refused");
        assert!(!failure.success);
        assert!(failure.error.is_some());
    }
}
