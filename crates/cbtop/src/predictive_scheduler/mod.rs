//! PMAT-051: Predictive Scheduling Optimizer
//!
//! SLO-aware workload scheduling with cost optimization for multi-host deployments.
//! Uses PMAT-033 predictive models for SLO violation risk assessment.
//!
//! # Falsifiable Hypothesis (FKR-052)
//! "Predictive scheduling achieves >99% SLO compliance while minimizing cost"
//!
//! # Features
//! - Host performance/cost modeling
//! - SLO risk prediction using regression models
//! - Cost optimization (minimize cloud spend)
//! - Load balancing with weighted capacity
//! - Spot instance and preemption support

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for predictive scheduler
#[derive(Debug, Clone)]
pub struct PredictiveSchedulerConfig {
    /// Target SLO compliance rate (0.0-1.0)
    pub target_slo_compliance: f64,
    /// Maximum cost per operation (normalized units)
    pub max_cost_per_op: f64,
    /// Enable spot instance scheduling
    pub enable_spot_instances: bool,
    /// Preemption buffer time (avoid scheduling near preemption)
    pub preemption_buffer: Duration,
    /// Load balancing weight decay factor
    pub load_decay_factor: f64,
    /// Minimum host capacity utilization before overflow
    pub min_capacity_threshold: f64,
    /// SLO violation penalty multiplier for cost function
    pub slo_violation_penalty: f64,
    /// History window for performance tracking
    pub history_window: usize,
}

impl Default for PredictiveSchedulerConfig {
    fn default() -> Self {
        Self {
            target_slo_compliance: 0.99,
            max_cost_per_op: 1.0,
            enable_spot_instances: true,
            preemption_buffer: Duration::from_secs(300), // 5 minutes
            load_decay_factor: 0.9,
            min_capacity_threshold: 0.8,
            slo_violation_penalty: 10.0,
            history_window: 100,
        }
    }
}

/// Host instance type for cost modeling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InstanceType {
    /// On-demand instances (guaranteed availability)
    OnDemand,
    /// Spot instances (can be preempted)
    Spot,
    /// Reserved instances (committed capacity)
    Reserved,
    /// Preemptible instances (scheduled termination)
    Preemptible,
}

impl InstanceType {
    /// Cost multiplier relative to on-demand
    pub fn cost_multiplier(&self) -> f64 {
        match self {
            Self::OnDemand => 1.0,
            Self::Spot => 0.3,        // 70% discount
            Self::Reserved => 0.6,    // 40% discount
            Self::Preemptible => 0.2, // 80% discount
        }
    }

    /// Reliability score (probability of availability)
    pub fn reliability(&self) -> f64 {
        match self {
            Self::OnDemand => 0.9999,
            Self::Reserved => 0.9999,
            Self::Spot => 0.85,
            Self::Preemptible => 0.90,
        }
    }
}

/// Performance profile for a host
#[derive(Debug, Clone)]
pub struct HostProfile {
    /// Unique host identifier
    pub host_id: String,
    /// Instance type for cost modeling
    pub instance_type: InstanceType,
    /// Compute capacity (operations per second)
    pub compute_capacity: f64,
    /// Memory capacity (bytes)
    pub memory_capacity: u64,
    /// Current load (0.0-1.0)
    pub current_load: f64,
    /// Base cost per hour (normalized units)
    pub hourly_cost: f64,
    /// Network latency to coordinator (ms)
    pub network_latency_ms: f64,
    /// Historical SLO compliance rate
    pub historical_slo_compliance: f64,
    /// Preemption deadline (None = not preemptible)
    pub preemption_deadline: Option<Instant>,
    /// Performance variance (CV of operation times)
    pub performance_cv: f64,
}

impl HostProfile {
    /// Create a new host profile
    pub fn new(host_id: impl Into<String>, instance_type: InstanceType) -> Self {
        Self {
            host_id: host_id.into(),
            instance_type,
            compute_capacity: 1000.0,
            memory_capacity: 8 * 1024 * 1024 * 1024, // 8GB
            current_load: 0.0,
            hourly_cost: 1.0,
            network_latency_ms: 1.0,
            historical_slo_compliance: 0.99,
            preemption_deadline: None,
            performance_cv: 0.1,
        }
    }

    /// Effective cost per operation
    pub fn cost_per_op(&self) -> f64 {
        let base_cost = self.hourly_cost * self.instance_type.cost_multiplier();
        // Cost per op = hourly cost / ops per hour, adjusted for current load
        let effective_capacity = self.compute_capacity * (1.0 - self.current_load);
        if effective_capacity > 0.0 {
            base_cost / (effective_capacity * 3600.0)
        } else {
            f64::MAX
        }
    }

    /// Available capacity (0.0-1.0)
    pub fn available_capacity(&self) -> f64 {
        (1.0 - self.current_load).max(0.0)
    }

    /// Time until preemption (if applicable)
    pub fn time_until_preemption(&self) -> Option<Duration> {
        self.preemption_deadline.map(|deadline| {
            let now = Instant::now();
            if deadline > now {
                deadline - now
            } else {
                Duration::ZERO
            }
        })
    }

    /// Check if host is safe for scheduling (not near preemption)
    pub fn is_safe_for_scheduling(&self, buffer: Duration) -> bool {
        match self.time_until_preemption() {
            Some(time_left) => time_left > buffer,
            None => true, // Non-preemptible hosts are always safe
        }
    }
}

/// Workload characteristics for scheduling decisions
#[derive(Debug, Clone)]
pub struct WorkloadSpec {
    /// Unique workload identifier
    pub workload_id: String,
    /// Estimated operation count
    pub operation_count: u64,
    /// Memory requirement (bytes)
    pub memory_required: u64,
    /// SLO deadline
    pub slo_deadline: Duration,
    /// Priority (higher = more important)
    pub priority: u32,
    /// Whether workload can be preempted
    pub preemptible: bool,
    /// Estimated compute intensity (ops per byte)
    pub compute_intensity: f64,
}

impl WorkloadSpec {
    /// Create a new workload specification
    pub fn new(workload_id: impl Into<String>, operation_count: u64) -> Self {
        Self {
            workload_id: workload_id.into(),
            operation_count,
            memory_required: 1024 * 1024, // 1MB default
            slo_deadline: Duration::from_millis(100),
            priority: 1,
            preemptible: true,
            compute_intensity: 100.0,
        }
    }

    /// Estimated execution time on a host
    pub fn estimated_execution_time(&self, host: &HostProfile) -> Duration {
        let ops_per_sec = host.compute_capacity * host.available_capacity();
        if ops_per_sec > 0.0 {
            let seconds = self.operation_count as f64 / ops_per_sec;
            Duration::from_secs_f64(seconds)
        } else {
            Duration::MAX
        }
    }
}

/// Scheduling decision for a workload
#[derive(Debug, Clone)]
pub struct SchedulingDecision {
    /// Target host for execution
    pub host_id: String,
    /// Predicted execution time
    pub predicted_time: Duration,
    /// Predicted cost
    pub predicted_cost: f64,
    /// SLO compliance probability
    pub slo_compliance_prob: f64,
    /// Scheduling score (higher = better)
    pub score: f64,
    /// Reason for selection
    pub reason: String,
}

/// Scheduling metrics for monitoring
#[derive(Debug, Clone, Default)]
pub struct SchedulerMetrics {
    /// Total scheduling decisions made
    pub total_decisions: u64,
    /// Decisions resulting in SLO violations
    pub slo_violations: u64,
    /// Total cost incurred
    pub total_cost: f64,
    /// Average scheduling latency
    pub avg_scheduling_latency_us: f64,
    /// Host utilization map
    pub host_utilization: HashMap<String, f64>,
    /// Spot instance savings
    pub spot_savings: f64,
}

impl SchedulerMetrics {
    /// Current SLO compliance rate
    pub fn slo_compliance_rate(&self) -> f64 {
        if self.total_decisions > 0 {
            1.0 - (self.slo_violations as f64 / self.total_decisions as f64)
        } else {
            1.0
        }
    }

    /// Average cost per decision
    pub fn avg_cost_per_decision(&self) -> f64 {
        if self.total_decisions > 0 {
            self.total_cost / self.total_decisions as f64
        } else {
            0.0
        }
    }
}

/// Predictive scheduling optimizer
pub struct PredictiveScheduler {
    config: PredictiveSchedulerConfig,
    hosts: HashMap<String, HostProfile>,
    metrics: SchedulerMetrics,
    /// Historical execution times per host
    execution_history: HashMap<String, Vec<Duration>>,
    /// SLO violation history per host
    violation_history: HashMap<String, Vec<bool>>,
}

impl PredictiveScheduler {
    /// Create a new predictive scheduler
    pub fn new(config: PredictiveSchedulerConfig) -> Self {
        Self {
            config,
            hosts: HashMap::new(),
            metrics: SchedulerMetrics::default(),
            execution_history: HashMap::new(),
            violation_history: HashMap::new(),
        }
    }

    /// Register a host with the scheduler
    pub fn register_host(&mut self, profile: HostProfile) {
        let host_id = profile.host_id.clone();
        self.hosts.insert(host_id.clone(), profile);
        self.execution_history.insert(host_id.clone(), Vec::new());
        self.violation_history.insert(host_id, Vec::new());
    }

    /// Remove a host from the scheduler
    pub fn deregister_host(&mut self, host_id: &str) {
        self.hosts.remove(host_id);
        self.execution_history.remove(host_id);
        self.violation_history.remove(host_id);
    }

    /// Update host load
    pub fn update_host_load(&mut self, host_id: &str, load: f64) {
        if let Some(host) = self.hosts.get_mut(host_id) {
            host.current_load = load.clamp(0.0, 1.0);
        }
    }

    /// Update host preemption deadline
    pub fn update_preemption_deadline(&mut self, host_id: &str, deadline: Option<Instant>) {
        if let Some(host) = self.hosts.get_mut(host_id) {
            host.preemption_deadline = deadline;
        }
    }

    /// Schedule a workload to optimal host
    pub fn schedule(&mut self, workload: &WorkloadSpec) -> Option<SchedulingDecision> {
        let start = Instant::now();

        // Filter eligible hosts
        let eligible_hosts: Vec<_> = self
            .hosts
            .values()
            .filter(|h| self.is_host_eligible(h, workload))
            .collect();

        if eligible_hosts.is_empty() {
            return None;
        }

        // Score each host
        let mut best_decision: Option<SchedulingDecision> = None;
        let mut best_score = f64::NEG_INFINITY;

        for host in eligible_hosts {
            let decision = self.evaluate_host(host, workload);
            if decision.score > best_score {
                best_score = decision.score;
                best_decision = Some(decision);
            }
        }

        // Update metrics
        if let Some(ref decision) = best_decision {
            self.metrics.total_decisions += 1;
            let scheduling_time = start.elapsed().as_micros() as f64;
            let n = self.metrics.total_decisions as f64;
            self.metrics.avg_scheduling_latency_us =
                self.metrics.avg_scheduling_latency_us * (n - 1.0) / n + scheduling_time / n;

            // Track spot savings
            if let Some(host) = self.hosts.get(&decision.host_id) {
                if host.instance_type == InstanceType::Spot {
                    let on_demand_cost =
                        decision.predicted_cost / host.instance_type.cost_multiplier();
                    self.metrics.spot_savings += on_demand_cost - decision.predicted_cost;
                }
            }
        }

        best_decision
    }

    /// Check if host is eligible for workload
    fn is_host_eligible(&self, host: &HostProfile, workload: &WorkloadSpec) -> bool {
        // Check capacity
        if host.current_load >= self.config.min_capacity_threshold {
            return false;
        }

        // Check memory
        if host.memory_capacity < workload.memory_required {
            return false;
        }

        // Check preemption safety
        if !host.is_safe_for_scheduling(self.config.preemption_buffer) {
            return false;
        }

        // Check spot instance policy
        if host.instance_type == InstanceType::Spot && !self.config.enable_spot_instances {
            return false;
        }

        true
    }

    /// Evaluate a host for workload placement
    fn evaluate_host(&self, host: &HostProfile, workload: &WorkloadSpec) -> SchedulingDecision {
        let predicted_time = self.predict_execution_time(host, workload);
        let slo_compliance_prob = self.predict_slo_compliance(host, workload, predicted_time);
        let predicted_cost = self.calculate_cost(host, workload, predicted_time);

        // Multi-objective scoring
        let score = self.calculate_score(host, slo_compliance_prob, predicted_cost, workload);

        let reason = self.generate_reason(host, slo_compliance_prob, predicted_cost);

        SchedulingDecision {
            host_id: host.host_id.clone(),
            predicted_time,
            predicted_cost,
            slo_compliance_prob,
            score,
            reason,
        }
    }

    /// Predict execution time using historical data
    fn predict_execution_time(&self, host: &HostProfile, workload: &WorkloadSpec) -> Duration {
        let base_estimate = workload.estimated_execution_time(host);

        // Adjust based on historical variance
        if let Some(history) = self.execution_history.get(&host.host_id) {
            if !history.is_empty() {
                // Use exponential smoothing on historical data
                let alpha = 0.3;
                let mut smoothed = history[0].as_secs_f64();
                for duration in history.iter().skip(1) {
                    smoothed = alpha * duration.as_secs_f64() + (1.0 - alpha) * smoothed;
                }

                // Blend historical with estimate
                let blended = 0.7 * base_estimate.as_secs_f64() + 0.3 * smoothed;
                return Duration::from_secs_f64(blended);
            }
        }

        // Add safety margin based on performance CV
        let margin = 1.0 + host.performance_cv;
        Duration::from_secs_f64(base_estimate.as_secs_f64() * margin)
    }

    /// Predict SLO compliance probability
    fn predict_slo_compliance(
        &self,
        host: &HostProfile,
        workload: &WorkloadSpec,
        predicted_time: Duration,
    ) -> f64 {
        // Base compliance from time vs deadline
        let time_ratio = predicted_time.as_secs_f64() / workload.slo_deadline.as_secs_f64();

        // Sigmoid function for compliance probability
        // P(comply) = 1 / (1 + exp(k * (time_ratio - 1)))
        let k = 10.0; // Steepness
        let base_prob = 1.0 / (1.0 + (k * (time_ratio - 0.9)).exp());

        // Adjust for host reliability
        let reliability_factor = host.instance_type.reliability();

        // Adjust for historical compliance
        let historical_factor = if let Some(history) = self.violation_history.get(&host.host_id) {
            if history.len() >= 10 {
                let recent: Vec<_> = history.iter().rev().take(10).collect();
                let violations = recent.iter().filter(|&&v| *v).count();
                1.0 - (violations as f64 / 10.0)
            } else {
                host.historical_slo_compliance
            }
        } else {
            host.historical_slo_compliance
        };

        base_prob * reliability_factor * historical_factor
    }

    /// Calculate execution cost
    fn calculate_cost(
        &self,
        host: &HostProfile,
        _workload: &WorkloadSpec,
        predicted_time: Duration,
    ) -> f64 {
        let hours = predicted_time.as_secs_f64() / 3600.0;
        let base_cost = host.hourly_cost * host.instance_type.cost_multiplier() * hours;

        // Add network cost based on latency
        let network_cost = host.network_latency_ms * 0.0001; // Small factor for latency

        base_cost + network_cost
    }

    /// Calculate multi-objective score
    fn calculate_score(
        &self,
        host: &HostProfile,
        slo_compliance_prob: f64,
        predicted_cost: f64,
        workload: &WorkloadSpec,
    ) -> f64 {
        // Priority weighting
        let priority_weight = 1.0 + (workload.priority as f64 * 0.1);

        // SLO compliance score (heavily weighted)
        let slo_score = if slo_compliance_prob >= self.config.target_slo_compliance {
            slo_compliance_prob * 100.0
        } else {
            // Penalty for below-target compliance
            slo_compliance_prob * 100.0
                - self.config.slo_violation_penalty
                    * (self.config.target_slo_compliance - slo_compliance_prob)
                    * 100.0
        };

        // Cost score (inverse - lower is better)
        let max_cost = self.config.max_cost_per_op;
        let cost_score = if predicted_cost <= max_cost {
            (1.0 - predicted_cost / max_cost) * 50.0
        } else {
            -((predicted_cost / max_cost) - 1.0) * 50.0
        };

        // Load balancing score (prefer less loaded hosts)
        let load_score = (1.0 - host.current_load) * 20.0;

        // Combine scores
        (slo_score + cost_score + load_score) * priority_weight
    }

    /// Generate human-readable reason for selection
    fn generate_reason(
        &self,
        host: &HostProfile,
        slo_compliance_prob: f64,
        predicted_cost: f64,
    ) -> String {
        let mut reasons = Vec::new();

        if slo_compliance_prob >= 0.99 {
            reasons.push("excellent SLO compliance");
        } else if slo_compliance_prob >= 0.95 {
            reasons.push("good SLO compliance");
        }

        if host.instance_type == InstanceType::Spot {
            reasons.push("cost-effective spot instance");
        } else if host.instance_type == InstanceType::Reserved {
            reasons.push("reserved capacity");
        }

        if host.current_load < 0.3 {
            reasons.push("low current load");
        }

        if predicted_cost < self.config.max_cost_per_op * 0.5 {
            reasons.push("low cost");
        }

        if reasons.is_empty() {
            "best available option".to_string()
        } else {
            reasons.join(", ")
        }
    }

    /// Record execution result for learning
    pub fn record_result(
        &mut self,
        host_id: &str,
        actual_time: Duration,
        slo_violated: bool,
        actual_cost: f64,
    ) {
        // Update execution history
        if let Some(history) = self.execution_history.get_mut(host_id) {
            history.push(actual_time);
            if history.len() > self.config.history_window {
                history.remove(0);
            }
        }

        // Update violation history
        if let Some(history) = self.violation_history.get_mut(host_id) {
            history.push(slo_violated);
            if history.len() > self.config.history_window {
                history.remove(0);
            }
        }

        // Update metrics
        if slo_violated {
            self.metrics.slo_violations += 1;
        }
        self.metrics.total_cost += actual_cost;

        // Update host utilization
        if let Some(host) = self.hosts.get(host_id) {
            self.metrics
                .host_utilization
                .insert(host_id.to_string(), host.current_load);
        }

        // Update host historical compliance
        if let Some(host) = self.hosts.get_mut(host_id) {
            if let Some(history) = self.violation_history.get(host_id) {
                let recent_violations = history
                    .iter()
                    .rev()
                    .take(self.config.history_window)
                    .filter(|&&v| v)
                    .count();
                let total = history.len().min(self.config.history_window);
                if total > 0 {
                    host.historical_slo_compliance =
                        1.0 - (recent_violations as f64 / total as f64);
                }
            }
        }
    }

    /// Get current scheduler metrics
    pub fn metrics(&self) -> &SchedulerMetrics {
        &self.metrics
    }

    /// Get all registered hosts
    pub fn hosts(&self) -> impl Iterator<Item = &HostProfile> {
        self.hosts.values()
    }

    /// Get host by ID
    pub fn get_host(&self, host_id: &str) -> Option<&HostProfile> {
        self.hosts.get(host_id)
    }

    /// Rebalance workloads across hosts (returns migration suggestions)
    pub fn suggest_rebalancing(&self) -> Vec<(String, String)> {
        let mut migrations = Vec::new();

        // Find overloaded and underloaded hosts
        let mut overloaded: Vec<_> = self
            .hosts
            .values()
            .filter(|h| h.current_load > 0.8)
            .collect();
        let mut underloaded: Vec<_> = self
            .hosts
            .values()
            .filter(|h| {
                h.current_load < 0.3 && h.is_safe_for_scheduling(self.config.preemption_buffer)
            })
            .collect();

        overloaded.sort_by(|a, b| {
            b.current_load
                .partial_cmp(&a.current_load)
                .expect("values should be comparable")
        });
        underloaded.sort_by(|a, b| {
            a.current_load
                .partial_cmp(&b.current_load)
                .expect("values should be comparable")
        });

        // Suggest migrations from overloaded to underloaded
        for (over, under) in overloaded.iter().zip(underloaded.iter()) {
            migrations.push((over.host_id.clone(), under.host_id.clone()));
        }

        migrations
    }
}


#[cfg(test)]
mod tests;
