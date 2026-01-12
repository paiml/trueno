//! Multi-Metric Correlation Analysis (PMAT-032)
//!
//! Correlate performance variance with system events (interrupts, I/O, processes).
//!
//! # Features
//!
//! - Correlate CV spikes with system events
//! - Detect "noisy neighbor" interference
//! - Recommend isolation strategies
//! - Capture system state snapshots
//!
//! # Falsification Criteria (F1241-F1250)
//!
//! See `tests/correlation_analysis_f1241.rs` for falsification tests.

use std::collections::VecDeque;

/// System event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    /// CPU interrupt
    Interrupt,
    /// Disk I/O
    DiskIo,
    /// Network activity
    Network,
    /// Context switch
    ContextSwitch,
    /// Page fault
    PageFault,
    /// Other process CPU usage
    ProcessCpu,
}

impl EventType {
    /// Get event type name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Interrupt => "interrupt",
            Self::DiskIo => "disk_io",
            Self::Network => "network",
            Self::ContextSwitch => "context_switch",
            Self::PageFault => "page_fault",
            Self::ProcessCpu => "process_cpu",
        }
    }
}

/// System event sample
#[derive(Debug, Clone)]
pub struct EventSample {
    /// Event type
    pub event_type: EventType,
    /// Timestamp (seconds since start)
    pub timestamp: f64,
    /// Event count or value
    pub value: f64,
}

impl EventSample {
    /// Create new sample
    pub fn new(event_type: EventType, timestamp: f64, value: f64) -> Self {
        Self {
            event_type,
            timestamp,
            value,
        }
    }
}

/// Performance sample with CV
#[derive(Debug, Clone)]
pub struct PerformanceSample {
    /// Timestamp (seconds since start)
    pub timestamp: f64,
    /// Coefficient of variation (%)
    pub cv_percent: f64,
    /// Latency (microseconds)
    pub latency_us: f64,
}

impl PerformanceSample {
    /// Create new sample
    pub fn new(timestamp: f64, cv_percent: f64, latency_us: f64) -> Self {
        Self {
            timestamp,
            cv_percent,
            latency_us,
        }
    }

    /// Check if this is a CV spike (>15%)
    pub fn is_spike(&self, threshold: f64) -> bool {
        self.cv_percent > threshold
    }
}

/// Correlation result between performance and events
#[derive(Debug, Clone)]
pub struct CorrelationResult {
    /// Event type
    pub event_type: EventType,
    /// Pearson correlation coefficient (-1 to 1)
    pub pearson_r: f64,
    /// Number of paired samples
    pub sample_count: usize,
    /// Is correlation significant?
    pub is_significant: bool,
    /// Lag (seconds) at max correlation
    pub optimal_lag: f64,
}

impl CorrelationResult {
    /// Check if there's positive correlation
    pub fn has_correlation(&self) -> bool {
        self.pearson_r.abs() > 0.3 && self.is_significant
    }

    /// Get correlation strength description
    pub fn strength(&self) -> &'static str {
        let r = self.pearson_r.abs();
        if r < 0.1 {
            "negligible"
        } else if r < 0.3 {
            "weak"
        } else if r < 0.5 {
            "moderate"
        } else if r < 0.7 {
            "strong"
        } else {
            "very strong"
        }
    }
}

/// Interference detection result
#[derive(Debug, Clone)]
pub struct InterferenceResult {
    /// Primary interference source
    pub primary_source: EventType,
    /// Correlation strength
    pub correlation: f64,
    /// Confidence (0-1)
    pub confidence: f64,
    /// Secondary sources
    pub secondary_sources: Vec<(EventType, f64)>,
}

impl InterferenceResult {
    /// Get interference category
    pub fn category(&self) -> InterferenceCategory {
        if self.confidence > 0.8 && self.correlation > 0.5 {
            InterferenceCategory::High
        } else if self.confidence > 0.5 || self.correlation > 0.3 {
            InterferenceCategory::Moderate
        } else {
            InterferenceCategory::Low
        }
    }
}

/// Interference severity category
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterferenceCategory {
    /// Low interference
    Low,
    /// Moderate interference
    Moderate,
    /// High interference
    High,
}

impl InterferenceCategory {
    /// Get category name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Moderate => "moderate",
            Self::High => "high",
        }
    }
}

/// Isolation recommendation
#[derive(Debug, Clone)]
pub struct IsolationRecommendation {
    /// Recommended action
    pub action: IsolationAction,
    /// Reason for recommendation
    pub reason: String,
    /// Expected improvement (%)
    pub expected_improvement: f64,
    /// Confidence (0-1)
    pub confidence: f64,
}

/// Isolation action types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsolationAction {
    /// Pin to specific CPU cores
    CpuPin,
    /// Use memory isolation (cgroups)
    MemoryIsolation,
    /// Disable hyperthreading on core
    DisableHyperthread,
    /// Use dedicated NUMA node
    NumaIsolation,
    /// Reduce network polling
    NetworkIsolation,
    /// Use realtime priority
    RealtimePriority,
    /// No action needed
    None,
}

impl IsolationAction {
    /// Get action name
    pub fn name(&self) -> &'static str {
        match self {
            Self::CpuPin => "cpu_pin",
            Self::MemoryIsolation => "memory_isolation",
            Self::DisableHyperthread => "disable_hyperthread",
            Self::NumaIsolation => "numa_isolation",
            Self::NetworkIsolation => "network_isolation",
            Self::RealtimePriority => "realtime_priority",
            Self::None => "none",
        }
    }

    /// Get action description
    pub fn description(&self) -> &'static str {
        match self {
            Self::CpuPin => "Pin benchmark to specific CPU cores using taskset/numactl",
            Self::MemoryIsolation => "Use cgroups to limit memory access from other processes",
            Self::DisableHyperthread => "Disable hyperthreading on benchmark cores",
            Self::NumaIsolation => "Run benchmark on dedicated NUMA node",
            Self::NetworkIsolation => "Reduce network polling frequency during benchmark",
            Self::RealtimePriority => "Use SCHED_FIFO realtime priority",
            Self::None => "No isolation needed",
        }
    }
}

/// System state snapshot
#[derive(Debug, Clone, Default)]
pub struct SystemSnapshot {
    /// Timestamp
    pub timestamp: f64,
    /// IRQ counts by type
    pub irq_counts: Vec<(String, u64)>,
    /// Disk I/O bytes/sec
    pub disk_io_bytes_per_sec: f64,
    /// Network packets/sec
    pub network_packets_per_sec: f64,
    /// Context switches/sec
    pub context_switches_per_sec: f64,
    /// Top CPU consumers (name, %)
    pub top_processes: Vec<(String, f64)>,
    /// Load average (1, 5, 15 min)
    pub load_average: (f64, f64, f64),
}

impl SystemSnapshot {
    /// Create empty snapshot
    pub fn new(timestamp: f64) -> Self {
        Self {
            timestamp,
            ..Default::default()
        }
    }

    /// Add IRQ count
    pub fn with_irq(mut self, name: &str, count: u64) -> Self {
        self.irq_counts.push((name.to_string(), count));
        self
    }

    /// Set disk I/O
    pub fn with_disk_io(mut self, bytes_per_sec: f64) -> Self {
        self.disk_io_bytes_per_sec = bytes_per_sec;
        self
    }

    /// Set network activity
    pub fn with_network(mut self, packets_per_sec: f64) -> Self {
        self.network_packets_per_sec = packets_per_sec;
        self
    }

    /// Set context switches
    pub fn with_context_switches(mut self, per_sec: f64) -> Self {
        self.context_switches_per_sec = per_sec;
        self
    }

    /// Add top process
    pub fn with_process(mut self, name: &str, cpu_percent: f64) -> Self {
        self.top_processes.push((name.to_string(), cpu_percent));
        self
    }

    /// Set load average
    pub fn with_load_average(mut self, load1: f64, load5: f64, load15: f64) -> Self {
        self.load_average = (load1, load5, load15);
        self
    }

    /// Get total IRQ count
    pub fn total_irqs(&self) -> u64 {
        self.irq_counts.iter().map(|(_, c)| c).sum()
    }
}

/// Correlation analyzer
#[derive(Debug)]
pub struct CorrelationAnalyzer {
    /// Performance samples
    perf_samples: VecDeque<PerformanceSample>,
    /// Event samples by type
    event_samples: VecDeque<EventSample>,
    /// System snapshots
    snapshots: VecDeque<SystemSnapshot>,
    /// Maximum samples to keep
    max_samples: usize,
    /// CV spike threshold (%)
    spike_threshold: f64,
    /// Correlation window (seconds)
    window_sec: f64,
}

impl Default for CorrelationAnalyzer {
    fn default() -> Self {
        Self {
            perf_samples: VecDeque::new(),
            event_samples: VecDeque::new(),
            snapshots: VecDeque::new(),
            max_samples: 1000,
            spike_threshold: 15.0,
            window_sec: 60.0,
        }
    }
}

impl CorrelationAnalyzer {
    /// Create new analyzer
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum samples
    pub fn with_max_samples(mut self, max: usize) -> Self {
        self.max_samples = max;
        self
    }

    /// Set CV spike threshold
    pub fn with_spike_threshold(mut self, percent: f64) -> Self {
        self.spike_threshold = percent;
        self
    }

    /// Set correlation window
    pub fn with_window(mut self, seconds: f64) -> Self {
        self.window_sec = seconds;
        self
    }

    /// Add performance sample
    pub fn add_perf_sample(&mut self, sample: PerformanceSample) {
        if self.perf_samples.len() >= self.max_samples {
            self.perf_samples.pop_front();
        }
        self.perf_samples.push_back(sample);
    }

    /// Add event sample
    pub fn add_event_sample(&mut self, sample: EventSample) {
        if self.event_samples.len() >= self.max_samples {
            self.event_samples.pop_front();
        }
        self.event_samples.push_back(sample);
    }

    /// Add system snapshot
    pub fn add_snapshot(&mut self, snapshot: SystemSnapshot) {
        if self.snapshots.len() >= self.max_samples / 10 {
            self.snapshots.pop_front();
        }
        self.snapshots.push_back(snapshot);
    }

    /// Get performance sample count
    pub fn perf_sample_count(&self) -> usize {
        self.perf_samples.len()
    }

    /// Get event sample count
    pub fn event_sample_count(&self) -> usize {
        self.event_samples.len()
    }

    /// Calculate correlation between CV and event type
    pub fn correlate_events(&self, event_type: EventType) -> Option<CorrelationResult> {
        let events: Vec<_> = self
            .event_samples
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect();

        if events.len() < 5 || self.perf_samples.len() < 5 {
            return None;
        }

        // Simple Pearson correlation at lag 0
        let pearson_r = self.compute_correlation(&events);

        let is_significant = pearson_r.abs() > 0.3 && events.len() >= 10;

        Some(CorrelationResult {
            event_type,
            pearson_r,
            sample_count: events.len().min(self.perf_samples.len()),
            is_significant,
            optimal_lag: 0.0,
        })
    }

    /// Compute Pearson correlation
    fn compute_correlation(&self, events: &[&EventSample]) -> f64 {
        // Match events to closest perf samples by timestamp
        let mut paired: Vec<(f64, f64)> = Vec::new();

        for event in events {
            if let Some(perf) = self.find_closest_perf(event.timestamp) {
                paired.push((event.value, perf.cv_percent));
            }
        }

        if paired.len() < 3 {
            return 0.0;
        }

        let n = paired.len() as f64;
        let sum_x: f64 = paired.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = paired.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = paired.iter().map(|(x, y)| x * y).sum();
        let sum_xx: f64 = paired.iter().map(|(x, _)| x * x).sum();
        let sum_yy: f64 = paired.iter().map(|(_, y)| y * y).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Find closest performance sample
    fn find_closest_perf(&self, timestamp: f64) -> Option<&PerformanceSample> {
        self.perf_samples
            .iter()
            .min_by(|a, b| {
                (a.timestamp - timestamp)
                    .abs()
                    .partial_cmp(&(b.timestamp - timestamp).abs())
                    .unwrap()
            })
            .filter(|p| (p.timestamp - timestamp).abs() < self.window_sec)
    }

    /// Detect interference source
    pub fn detect_interference(&self) -> Option<InterferenceResult> {
        let event_types = [
            EventType::Interrupt,
            EventType::DiskIo,
            EventType::Network,
            EventType::ContextSwitch,
            EventType::PageFault,
            EventType::ProcessCpu,
        ];

        let mut correlations: Vec<(EventType, f64)> = Vec::new();

        for event_type in event_types {
            if let Some(result) = self.correlate_events(event_type) {
                if result.pearson_r.abs() > 0.1 {
                    correlations.push((event_type, result.pearson_r));
                }
            }
        }

        if correlations.is_empty() {
            return None;
        }

        correlations.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        let (primary_source, correlation) = correlations[0];
        let secondary_sources: Vec<_> = correlations.into_iter().skip(1).collect();

        let confidence = if correlation.abs() > 0.5 && self.perf_samples.len() > 20 {
            0.9
        } else if correlation.abs() > 0.3 {
            0.7
        } else {
            0.5
        };

        Some(InterferenceResult {
            primary_source,
            correlation,
            confidence,
            secondary_sources,
        })
    }

    /// Recommend isolation strategy
    pub fn recommend_isolation(&self) -> IsolationRecommendation {
        if let Some(interference) = self.detect_interference() {
            let (action, expected_improvement) = match interference.primary_source {
                EventType::Interrupt => (IsolationAction::CpuPin, 20.0),
                EventType::ContextSwitch => (IsolationAction::RealtimePriority, 15.0),
                EventType::ProcessCpu => (IsolationAction::CpuPin, 25.0),
                EventType::DiskIo => (IsolationAction::MemoryIsolation, 10.0),
                EventType::Network => (IsolationAction::NetworkIsolation, 15.0),
                EventType::PageFault => (IsolationAction::MemoryIsolation, 20.0),
            };

            let reason = format!(
                "{} correlation ({:.2}) with {} detected",
                interference.category().name(),
                interference.correlation,
                interference.primary_source.name()
            );

            IsolationRecommendation {
                action,
                reason,
                expected_improvement: expected_improvement * interference.correlation.abs(),
                confidence: interference.confidence,
            }
        } else {
            IsolationRecommendation {
                action: IsolationAction::None,
                reason: "No significant interference detected".to_string(),
                expected_improvement: 0.0,
                confidence: 0.9,
            }
        }
    }

    /// Capture current system state
    pub fn capture_system_state(&mut self, timestamp: f64) -> SystemSnapshot {
        // In real implementation, would read from /proc
        // For now, return empty snapshot
        let snapshot = SystemSnapshot::new(timestamp);
        self.add_snapshot(snapshot.clone());
        snapshot
    }

    /// Get CV spikes
    pub fn get_spikes(&self) -> Vec<&PerformanceSample> {
        self.perf_samples
            .iter()
            .filter(|s| s.is_spike(self.spike_threshold))
            .collect()
    }

    /// Clear all samples
    pub fn clear(&mut self) {
        self.perf_samples.clear();
        self.event_samples.clear();
        self.snapshots.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type_names() {
        assert_eq!(EventType::Interrupt.name(), "interrupt");
        assert_eq!(EventType::DiskIo.name(), "disk_io");
        assert_eq!(EventType::Network.name(), "network");
    }

    #[test]
    fn test_perf_sample_spike() {
        let sample = PerformanceSample::new(0.0, 20.0, 100.0);
        assert!(sample.is_spike(15.0));
        assert!(!sample.is_spike(25.0));
    }

    #[test]
    fn test_correlation_strength() {
        let result = CorrelationResult {
            event_type: EventType::Interrupt,
            pearson_r: 0.6,
            sample_count: 100,
            is_significant: true,
            optimal_lag: 0.0,
        };

        assert_eq!(result.strength(), "strong");
        assert!(result.has_correlation());
    }

    #[test]
    fn test_isolation_action_names() {
        assert_eq!(IsolationAction::CpuPin.name(), "cpu_pin");
        assert_eq!(IsolationAction::None.name(), "none");
    }

    #[test]
    fn test_system_snapshot() {
        let snapshot = SystemSnapshot::new(1.0)
            .with_irq("timer", 1000)
            .with_irq("disk", 500)
            .with_disk_io(1_000_000.0)
            .with_network(5000.0)
            .with_load_average(1.5, 1.0, 0.8);

        assert_eq!(snapshot.total_irqs(), 1500);
        assert_eq!(snapshot.disk_io_bytes_per_sec, 1_000_000.0);
    }

    #[test]
    fn test_analyzer_add_samples() {
        let mut analyzer = CorrelationAnalyzer::new();

        analyzer.add_perf_sample(PerformanceSample::new(0.0, 5.0, 100.0));
        analyzer.add_perf_sample(PerformanceSample::new(1.0, 10.0, 120.0));

        assert_eq!(analyzer.perf_sample_count(), 2);

        analyzer.add_event_sample(EventSample::new(EventType::Interrupt, 0.0, 1000.0));
        assert_eq!(analyzer.event_sample_count(), 1);
    }

    #[test]
    fn test_analyzer_correlation() {
        let mut analyzer = CorrelationAnalyzer::new();

        // Add correlated samples
        for i in 0..20 {
            let t = i as f64;
            analyzer.add_perf_sample(PerformanceSample::new(t, 5.0 + i as f64, 100.0));
            analyzer.add_event_sample(EventSample::new(EventType::Interrupt, t, 100.0 + i as f64 * 10.0));
        }

        let result = analyzer.correlate_events(EventType::Interrupt).unwrap();
        assert!(result.pearson_r > 0.5);
    }

    #[test]
    fn test_recommend_isolation() {
        let analyzer = CorrelationAnalyzer::new();
        let rec = analyzer.recommend_isolation();

        // With no data, should recommend none
        assert_eq!(rec.action, IsolationAction::None);
    }
}
