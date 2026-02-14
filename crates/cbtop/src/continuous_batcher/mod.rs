//! ContinuousBatcher Implementation (PMAT-015)
//!
//! Implements continuous batching for LLM inference per cbtop spec §19.
//!
//! # Overview
//!
//! Continuous batching processes inference requests dynamically, allowing
//! new requests to join and completed requests to leave mid-batch.
//!
//! # Citations
//!
//! - [Yu et al. 2022] "ORCA: Continuous Batching for LLM Inference" OSDI
//! - [Leviathan et al. 2023] "Fast Inference from Transformers via Speculative Decoding" ICML
//! - [Chen et al. 2023] "Accelerating LLM Decoding with Speculative Sampling" arXiv

use std::collections::VecDeque;
use std::fmt;
use std::time::Instant;

use crate::paged_kv::SeqId;

/// Token type (simplified - u32 vocabulary index).
pub type Token = u32;

/// Request priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Priority(pub u8);

impl Default for Priority {
    fn default() -> Self {
        Priority(128) // Middle priority
    }
}

/// Inference request.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Unique request ID
    pub id: SeqId,
    /// Input tokens (prompt)
    pub input_tokens: Vec<Token>,
    /// Maximum output tokens to generate
    pub max_new_tokens: usize,
    /// Request priority
    pub priority: Priority,
    /// Arrival timestamp
    pub arrival_time: Instant,
    /// Estimated total tokens (input + output)
    pub estimated_tokens: usize,
}

impl InferenceRequest {
    /// Create a new inference request.
    pub fn new(id: SeqId, input_tokens: Vec<Token>, max_new_tokens: usize) -> Self {
        let estimated_tokens = input_tokens.len() + max_new_tokens;
        Self {
            id,
            input_tokens,
            max_new_tokens,
            priority: Priority::default(),
            arrival_time: Instant::now(),
            estimated_tokens,
        }
    }

    /// Create request with priority.
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Input sequence length.
    pub fn input_len(&self) -> usize {
        self.input_tokens.len()
    }
}

/// Sequence group (request + generation state).
#[derive(Debug, Clone)]
pub struct SequenceGroup {
    /// Original request
    pub request: InferenceRequest,
    /// Generated tokens so far
    pub output_tokens: Vec<Token>,
    /// Is generation complete?
    pub is_finished: bool,
    /// Last access timestamp (for LRU)
    pub last_access: Instant,
    /// Number of decode steps
    pub num_steps: usize,
}

impl SequenceGroup {
    /// Create new sequence group from request.
    pub fn new(request: InferenceRequest) -> Self {
        Self {
            request,
            output_tokens: Vec::new(),
            is_finished: false,
            last_access: Instant::now(),
            num_steps: 0,
        }
    }

    /// Total tokens (input + output so far).
    pub fn total_tokens(&self) -> usize {
        self.request.input_tokens.len() + self.output_tokens.len()
    }

    /// Remaining tokens to generate.
    pub fn remaining_tokens(&self) -> usize {
        self.request
            .max_new_tokens
            .saturating_sub(self.output_tokens.len())
    }

    /// Add generated token.
    pub fn add_token(&mut self, token: Token) {
        self.output_tokens.push(token);
        self.last_access = Instant::now();
        self.num_steps += 1;

        // Check if finished
        if self.output_tokens.len() >= self.request.max_new_tokens {
            self.is_finished = true;
        }
    }

    /// Mark as finished.
    pub fn finish(&mut self) {
        self.is_finished = true;
    }
}

/// Scheduling policy for request prioritization.
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulingPolicy {
    /// First-come, first-served
    FCFS,
    /// Shortest job first (by estimated tokens)
    SJF,
    /// Priority-based (API tiers)
    Priority { preempt_enabled: bool },
    /// Fair share (equal GPU time per user)
    FairShare,
}

impl Default for SchedulingPolicy {
    fn default() -> Self {
        SchedulingPolicy::FCFS
    }
}

impl fmt::Display for SchedulingPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchedulingPolicy::FCFS => write!(f, "FCFS"),
            SchedulingPolicy::SJF => write!(f, "SJF"),
            SchedulingPolicy::Priority { preempt_enabled } => {
                write!(f, "Priority(preempt={})", preempt_enabled)
            }
            SchedulingPolicy::FairShare => write!(f, "FairShare"),
        }
    }
}

/// Batch schedule result.
#[derive(Debug, Clone)]
pub struct BatchSchedule {
    /// Sequence IDs in this batch
    pub sequence_ids: Vec<SeqId>,
    /// Number of sequences in batch
    pub batch_size: usize,
    /// Total tokens to process
    pub total_tokens: usize,
    /// Prefill sequences (first token)
    pub prefill_count: usize,
    /// Decode sequences (continuation)
    pub decode_count: usize,
}

impl BatchSchedule {
    /// Create empty schedule.
    pub fn empty() -> Self {
        Self {
            sequence_ids: Vec::new(),
            batch_size: 0,
            total_tokens: 0,
            prefill_count: 0,
            decode_count: 0,
        }
    }

    /// Check if schedule is empty.
    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
    }
}

/// Token output from a decode step.
#[derive(Debug, Clone)]
pub struct TokenOutput {
    /// Sequence ID
    pub seq_id: SeqId,
    /// Generated token
    pub token: Token,
    /// Is EOS token?
    pub is_eos: bool,
}

/// Batcher statistics.
#[derive(Debug, Clone, Default)]
pub struct BatcherStats {
    /// Total tokens processed
    pub total_tokens: u64,
    /// Total requests completed
    pub total_requests: u64,
    /// Total preemptions
    pub total_preemptions: u64,
    /// Total swaps (CPU<->GPU)
    pub total_swaps: u64,
    /// Processing start time
    pub start_time: Option<Instant>,
}

impl BatcherStats {
    /// Calculate throughput (tokens/sec).
    pub fn throughput(&self) -> f64 {
        if let Some(start) = self.start_time {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.0 {
                return self.total_tokens as f64 / elapsed;
            }
        }
        0.0
    }
}

/// Continuous batching scheduler for LLM inference.
///
/// Processes requests as they arrive without waiting for batch completion.
/// Based on ORCA continuous batching algorithm.
#[derive(Debug)]
pub struct ContinuousBatcher {
    /// Maximum batch size (GPU memory limited)
    max_batch_size: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Active sequences in current batch
    running: Vec<SequenceGroup>,
    /// Waiting queue (sorted by policy)
    waiting: VecDeque<SequenceGroup>,
    /// Swapped sequences (offloaded to CPU)
    swapped: Vec<SequenceGroup>,
    /// Scheduling policy
    policy: SchedulingPolicy,
    /// Statistics
    stats: BatcherStats,
    /// Memory threshold for preemption (0.0-1.0)
    memory_threshold: f64,
}

impl ContinuousBatcher {
    /// Create a new continuous batcher.
    pub fn new(max_batch_size: usize, max_seq_len: usize) -> Self {
        Self {
            max_batch_size,
            max_seq_len,
            running: Vec::new(),
            waiting: VecDeque::new(),
            swapped: Vec::new(),
            policy: SchedulingPolicy::default(),
            stats: BatcherStats {
                start_time: Some(Instant::now()),
                ..Default::default()
            },
            memory_threshold: 0.9,
        }
    }

    /// Set scheduling policy.
    pub fn with_policy(mut self, policy: SchedulingPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set memory threshold for preemption.
    pub fn with_memory_threshold(mut self, threshold: f64) -> Self {
        self.memory_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Get scheduling policy.
    pub fn policy(&self) -> &SchedulingPolicy {
        &self.policy
    }

    /// Get max batch size.
    pub fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Get max sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get number of running sequences.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Get number of waiting sequences.
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    /// Get number of swapped sequences.
    pub fn swapped_count(&self) -> usize {
        self.swapped.len()
    }

    /// Get statistics.
    pub fn stats(&self) -> &BatcherStats {
        &self.stats
    }

    /// Current throughput (tokens/sec).
    pub fn throughput(&self) -> f64 {
        self.stats.throughput()
    }

    /// Add new inference request.
    pub fn add_request(&mut self, request: InferenceRequest) {
        let seq_group = SequenceGroup::new(request);
        self.insert_waiting(seq_group);
    }

    /// Insert into waiting queue according to policy.
    fn insert_waiting(&mut self, seq_group: SequenceGroup) {
        match &self.policy {
            SchedulingPolicy::FCFS => {
                // Add to back of queue
                self.waiting.push_back(seq_group);
            }
            SchedulingPolicy::SJF => {
                // Insert sorted by estimated tokens (shortest first)
                let insert_idx = self
                    .waiting
                    .iter()
                    .position(|s| s.request.estimated_tokens > seq_group.request.estimated_tokens)
                    .unwrap_or(self.waiting.len());
                self.waiting.insert(insert_idx, seq_group);
            }
            SchedulingPolicy::Priority { .. } => {
                // Insert sorted by priority (highest first)
                let insert_idx = self
                    .waiting
                    .iter()
                    .position(|s| s.request.priority < seq_group.request.priority)
                    .unwrap_or(self.waiting.len());
                self.waiting.insert(insert_idx, seq_group);
            }
            SchedulingPolicy::FairShare => {
                // Simple round-robin for now
                self.waiting.push_back(seq_group);
            }
        }
    }

    /// Schedule next iteration batch.
    pub fn schedule(&mut self) -> BatchSchedule {
        // Remove finished sequences from running
        self.running.retain(|s| !s.is_finished);

        // Calculate available slots
        let _available = self.max_batch_size.saturating_sub(self.running.len());

        // Promote from waiting to running
        let mut prefill_count = 0;
        while !self.waiting.is_empty() && self.running.len() < self.max_batch_size {
            if let Some(seq_group) = self.waiting.pop_front() {
                // Check if sequence fits
                if seq_group.total_tokens() <= self.max_seq_len {
                    prefill_count += 1;
                    self.running.push(seq_group);
                } else {
                    // Too long, put back
                    self.waiting.push_front(seq_group);
                    break;
                }
            }
        }

        // Swap in from swapped if space available
        while !self.swapped.is_empty() && self.running.len() < self.max_batch_size {
            if let Some(seq_group) = self.swapped.pop() {
                self.running.push(seq_group);
                self.stats.total_swaps += 1;
            }
        }

        // Build schedule
        let sequence_ids: Vec<SeqId> = self.running.iter().map(|s| s.request.id).collect();
        let total_tokens: usize = self.running.iter().map(|s| s.total_tokens()).sum();
        let decode_count = self.running.len() - prefill_count;

        BatchSchedule {
            batch_size: sequence_ids.len(),
            sequence_ids,
            total_tokens,
            prefill_count,
            decode_count,
        }
    }

    /// Process outputs from a decode step.
    pub fn process_outputs(&mut self, outputs: Vec<TokenOutput>) {
        for output in outputs {
            // Find the sequence and add the token
            if let Some(seq_group) = self
                .running
                .iter_mut()
                .find(|s| s.request.id == output.seq_id)
            {
                seq_group.add_token(output.token);
                self.stats.total_tokens += 1;

                // Check for EOS
                if output.is_eos {
                    seq_group.finish();
                    self.stats.total_requests += 1;
                }
            }
        }
    }

    /// Preempt sequences under memory pressure.
    pub fn preempt(&mut self, num_to_preempt: usize) -> Vec<SeqId> {
        let mut preempted = Vec::new();

        // Preempt from running (lowest priority or longest)
        for _ in 0..num_to_preempt {
            if self.running.is_empty() {
                break;
            }

            // Find victim (longest sequence for simplicity)
            let victim_idx = self
                .running
                .iter()
                .enumerate()
                .max_by_key(|(_, s)| s.total_tokens())
                .map(|(i, _)| i);

            if let Some(idx) = victim_idx {
                let victim = self.running.remove(idx);
                preempted.push(victim.request.id);
                self.swapped.push(victim);
                self.stats.total_preemptions += 1;
            }
        }

        preempted
    }

    /// Check if preemption is needed (simulated memory pressure).
    pub fn needs_preemption(&self, current_utilization: f64) -> bool {
        current_utilization >= self.memory_threshold
            && !self.running.is_empty()
            && matches!(
                self.policy,
                SchedulingPolicy::Priority {
                    preempt_enabled: true
                }
            )
    }

    /// Get sequence by ID.
    pub fn get_sequence(&self, seq_id: SeqId) -> Option<&SequenceGroup> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.swapped.iter())
            .find(|s| s.request.id == seq_id)
    }

    /// Get all sequence IDs.
    pub fn all_sequence_ids(&self) -> Vec<SeqId> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .chain(self.swapped.iter())
            .map(|s| s.request.id)
            .collect()
    }
}

impl fmt::Display for ContinuousBatcher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ContinuousBatcher")?;
        writeln!(
            f,
            "  Policy: {} | Max Batch: {} | Max Seq: {}",
            self.policy, self.max_batch_size, self.max_seq_len
        )?;
        writeln!(
            f,
            "  Running: {} | Waiting: {} | Swapped: {}",
            self.running.len(),
            self.waiting.len(),
            self.swapped.len()
        )?;
        writeln!(f, "  Throughput: {:.1} tok/s", self.throughput())?;
        writeln!(
            f,
            "  Stats: tokens={}, requests={}, preemptions={}, swaps={}",
            self.stats.total_tokens,
            self.stats.total_requests,
            self.stats.total_preemptions,
            self.stats.total_swaps
        )?;
        Ok(())
    }
}

/// Exponential moving average for tracking metrics.
#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage {
    /// Current value
    value: f64,
    /// Smoothing factor (0-1)
    alpha: f64,
    /// Number of samples
    count: u64,
}

impl ExponentialMovingAverage {
    /// Create new EMA with smoothing factor.
    pub fn new(alpha: f64) -> Self {
        Self {
            value: 0.0,
            alpha: alpha.clamp(0.0, 1.0),
            count: 0,
        }
    }

    /// Update with new sample.
    pub fn update(&mut self, sample: f64) {
        if self.count == 0 {
            self.value = sample;
        } else {
            self.value = self.alpha * sample + (1.0 - self.alpha) * self.value;
        }
        self.count += 1;
    }

    /// Get current value.
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.count = 0;
    }
}

impl Default for ExponentialMovingAverage {
    fn default() -> Self {
        Self::new(0.1)
    }
}

/// Output from speculative decoding step.
#[derive(Debug, Clone)]
pub struct SpeculativeOutput {
    /// Accepted tokens from draft
    pub accepted: Vec<Token>,
    /// Rejection index (first rejected draft token)
    pub rejection_idx: Option<usize>,
    /// Token from target model (after rejection or all accepted)
    pub target_token: Token,
    /// Total draft tokens proposed
    pub draft_count: usize,
}

impl SpeculativeOutput {
    /// Calculate acceptance rate for this step.
    pub fn acceptance_rate(&self) -> f64 {
        if self.draft_count == 0 {
            return 0.0;
        }
        self.accepted.len() as f64 / self.draft_count as f64
    }

    /// Number of tokens produced (accepted + 1 target).
    pub fn total_tokens(&self) -> usize {
        self.accepted.len() + 1
    }
}

/// Speculative decoding coordinator.
///
/// Coordinates draft and target models for speculative decoding.
/// The draft model proposes tokens, target model verifies.
#[derive(Debug)]
pub struct SpeculativeDecoder {
    /// Speculation depth (draft tokens per step)
    k: usize,
    /// Acceptance rate tracker
    acceptance_rate: ExponentialMovingAverage,
    /// Total steps
    total_steps: u64,
    /// Total accepted tokens
    total_accepted: u64,
    /// Total draft tokens
    total_draft: u64,
}

impl SpeculativeDecoder {
    /// Create new speculative decoder.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            acceptance_rate: ExponentialMovingAverage::new(0.1),
            total_steps: 0,
            total_accepted: 0,
            total_draft: 0,
        }
    }

    /// Get speculation depth.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Set speculation depth.
    pub fn set_k(&mut self, k: usize) {
        self.k = k;
    }

    /// Simulate speculative decoding step.
    ///
    /// In a real implementation, this would:
    /// 1. Run draft model k times to get draft tokens
    /// 2. Run target model once on all draft positions
    /// 3. Compare and accept/reject
    pub fn simulate_step(
        &mut self,
        draft_tokens: &[Token],
        target_probs: &[(Token, f64)],
    ) -> SpeculativeOutput {
        let draft_count = draft_tokens.len().min(self.k);
        let mut accepted = Vec::new();
        let mut rejection_idx = None;

        // Simulate acceptance (simplified: accept if target agrees)
        for (i, &draft_token) in draft_tokens.iter().take(draft_count).enumerate() {
            if let Some((target_token, _)) = target_probs.get(i) {
                if *target_token == draft_token {
                    accepted.push(draft_token);
                } else {
                    rejection_idx = Some(i);
                    break;
                }
            } else {
                rejection_idx = Some(i);
                break;
            }
        }

        // Get target token (either after rejection or as the k+1 token)
        let target_token = if let Some(idx) = rejection_idx {
            target_probs.get(idx).map(|(t, _)| *t).unwrap_or(0)
        } else {
            target_probs.get(draft_count).map(|(t, _)| *t).unwrap_or(0)
        };

        let output = SpeculativeOutput {
            accepted: accepted.clone(),
            rejection_idx,
            target_token,
            draft_count,
        };

        // Update statistics
        self.total_steps += 1;
        self.total_accepted += accepted.len() as u64;
        self.total_draft += draft_count as u64;
        self.acceptance_rate.update(output.acceptance_rate());

        output
    }

    /// Current acceptance rate (EMA).
    pub fn acceptance_rate(&self) -> f64 {
        self.acceptance_rate.value()
    }

    /// Overall acceptance rate.
    pub fn overall_acceptance_rate(&self) -> f64 {
        if self.total_draft == 0 {
            return 0.0;
        }
        self.total_accepted as f64 / self.total_draft as f64
    }

    /// Effective speedup vs naive decoding.
    ///
    /// Speedup = (accepted + 1) / (1 + verification_cost_ratio)
    /// Simplified: assume verification cost = 1/k of naive
    pub fn speedup(&self) -> f64 {
        let rate = self.acceptance_rate();
        // Expected tokens per step = 1 + k * acceptance_rate
        // Cost = 1 (target call) + k * draft_cost (assume draft_cost << 1)
        // Simplified model: speedup ≈ 1 + k * acceptance_rate
        1.0 + (self.k as f64) * rate
    }

    /// Get statistics.
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.total_steps, self.total_accepted, self.total_draft)
    }
}

impl fmt::Display for SpeculativeDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpeculativeDecoder(k={}, acceptance={:.1}%, speedup={:.2}x)",
            self.k,
            self.acceptance_rate() * 100.0,
            self.speedup()
        )
    }
}


#[cfg(test)]
mod tests;
