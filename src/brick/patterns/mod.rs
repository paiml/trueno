//! Async and Buffer Pattern Catalog
//!
//! Utility patterns from Phase 12 (E.10) for async operations,
//! buffer management, and flow control.
//!
//! # Patterns Included
//!
//! - **LCP-12**: AsyncResult - async compute with sync fallback
//! - **AWP-11**: BoundedQueue - bounded message queue with back-pressure
//! - **AWP-13**: ReserveStrategy - buffer reservation strategies
//! - **LCP-08**: GraphReuseCounter - graph reuse tracking
//! - **AWP-03**: DualWakerState - dual-waker backpressure
//! - **AWP-04**: StreamCapacity - HTTP/2 flow control
//! - **AWP-09**: WakeSkipState - smart wake skip optimization

#[cfg(test)]
mod tests;

use std::collections::VecDeque;

// ============================================================================
// LCP-12: Async Compute with Sync Fallback
// ============================================================================

/// Result of an async operation with fallback capability.
#[derive(Debug, Clone)]
pub enum AsyncResult<T, E> {
    /// Operation completed asynchronously
    Async(T),
    /// Operation completed synchronously (fallback)
    Sync(T),
    /// Operation failed
    Error(E),
}

impl<T, E> AsyncResult<T, E> {
    /// Check if result was obtained asynchronously.
    #[must_use]
    pub fn is_async(&self) -> bool {
        matches!(self, AsyncResult::Async(_))
    }

    /// Check if result was obtained synchronously (fallback).
    #[must_use]
    pub fn is_sync(&self) -> bool {
        matches!(self, AsyncResult::Sync(_))
    }

    /// Check if operation failed.
    #[must_use]
    pub fn is_error(&self) -> bool {
        matches!(self, AsyncResult::Error(_))
    }

    /// Get the result value, regardless of async/sync.
    pub fn into_result(self) -> Result<T, E> {
        match self {
            AsyncResult::Async(v) | AsyncResult::Sync(v) => Ok(v),
            AsyncResult::Error(e) => Err(e),
        }
    }

    /// Map the success value.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> AsyncResult<U, E> {
        match self {
            AsyncResult::Async(v) => AsyncResult::Async(f(v)),
            AsyncResult::Sync(v) => AsyncResult::Sync(f(v)),
            AsyncResult::Error(e) => AsyncResult::Error(e),
        }
    }
}

// ============================================================================
// AWP-11: Bounded Message Queue
// ============================================================================

/// Bounded message queue with back-pressure.
///
/// # Example
/// ```rust
/// use trueno::brick::BoundedQueue;
///
/// let mut queue: BoundedQueue<i32> = BoundedQueue::new(3);
///
/// assert!(queue.try_push(1).is_ok());
/// assert!(queue.try_push(2).is_ok());
/// assert!(queue.try_push(3).is_ok());
/// assert!(queue.try_push(4).is_err()); // Queue full
///
/// assert_eq!(queue.pop(), Some(1));
/// assert!(queue.try_push(4).is_ok()); // Space available
/// ```
#[derive(Debug)]
pub struct BoundedQueue<T> {
    items: VecDeque<T>,
    capacity: usize,
}

impl<T> BoundedQueue<T> {
    /// Create a new bounded queue.
    pub fn new(capacity: usize) -> Self {
        Self {
            items: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Try to push an item. Returns error if queue is full.
    pub fn try_push(&mut self, item: T) -> Result<(), T> {
        if self.items.len() >= self.capacity {
            Err(item)
        } else {
            self.items.push_back(item);
            Ok(())
        }
    }

    /// Pop an item from the front.
    pub fn pop(&mut self) -> Option<T> {
        self.items.pop_front()
    }

    /// Peek at the front item.
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        self.items.front()
    }

    /// Get the number of items in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Check if the queue is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.items.len() >= self.capacity
    }

    /// Get the capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get remaining capacity.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.items.len())
    }

    /// Clear all items.
    pub fn clear(&mut self) {
        self.items.clear();
    }
}

impl<T> Default for BoundedQueue<T> {
    fn default() -> Self {
        Self::new(16)
    }
}

// ============================================================================
// AWP-13: Buffer Reserve Strategy
// ============================================================================

/// Strategy for buffer reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveStrategy {
    /// Reserve exact amount needed
    Exact,
    /// Reserve with 50% growth headroom
    Grow50,
    /// Reserve with 100% growth headroom (double)
    Double,
    /// Reserve to next power of two
    PowerOfTwo,
}

/// Reserve buffer capacity according to strategy.
///
/// # Example
/// ```rust
/// use trueno::brick::{reserve_capacity, ReserveStrategy};
///
/// assert_eq!(reserve_capacity(100, ReserveStrategy::Exact), 100);
/// assert_eq!(reserve_capacity(100, ReserveStrategy::Grow50), 150);
/// assert_eq!(reserve_capacity(100, ReserveStrategy::Double), 200);
/// assert_eq!(reserve_capacity(100, ReserveStrategy::PowerOfTwo), 128);
/// ```
#[must_use]
pub fn reserve_capacity(needed: usize, strategy: ReserveStrategy) -> usize {
    match strategy {
        ReserveStrategy::Exact => needed,
        ReserveStrategy::Grow50 => needed + needed / 2,
        ReserveStrategy::Double => needed * 2,
        ReserveStrategy::PowerOfTwo => needed.next_power_of_two(),
    }
}

/// Buffer with configurable reserve strategy.
#[derive(Debug)]
pub struct StrategicBuffer {
    data: Vec<u8>,
    strategy: ReserveStrategy,
}

impl StrategicBuffer {
    /// Create a new buffer with the given strategy.
    pub fn new(strategy: ReserveStrategy) -> Self {
        Self {
            data: Vec::new(),
            strategy,
        }
    }

    /// Create with initial capacity.
    pub fn with_capacity(capacity: usize, strategy: ReserveStrategy) -> Self {
        Self {
            data: Vec::with_capacity(reserve_capacity(capacity, strategy)),
            strategy,
        }
    }

    /// Ensure capacity for additional bytes.
    pub fn reserve(&mut self, additional: usize) {
        let needed = self.data.len() + additional;
        if needed > self.data.capacity() {
            let new_cap = reserve_capacity(needed, self.strategy);
            self.data.reserve(new_cap - self.data.capacity());
        }
    }

    /// Write bytes to the buffer.
    pub fn write(&mut self, bytes: &[u8]) {
        self.reserve(bytes.len());
        self.data.extend_from_slice(bytes);
    }

    /// Get the data.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get current length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl Default for StrategicBuffer {
    fn default() -> Self {
        Self::new(ReserveStrategy::Double)
    }
}

// ============================================================================
// LCP-08: Graph Reuse Counter
// ============================================================================

/// Counter for tracking graph reuse in inference optimization.
///
/// Tracks how many times a computation graph has been reused,
/// enabling optimization decisions like caching or recompilation.
#[derive(Debug, Clone, Default)]
pub struct GraphReuseCounter {
    /// Number of times this graph has been executed
    reuse_count: u64,
    /// Threshold for considering graph "hot"
    hot_threshold: u64,
    /// Whether to enable caching
    cache_enabled: bool,
}

impl GraphReuseCounter {
    /// Create a new counter with hot threshold.
    pub fn new(hot_threshold: u64) -> Self {
        Self {
            reuse_count: 0,
            hot_threshold,
            cache_enabled: false,
        }
    }

    /// Record a graph execution.
    pub fn record_use(&mut self) {
        self.reuse_count += 1;
        if self.reuse_count >= self.hot_threshold {
            self.cache_enabled = true;
        }
    }

    /// Check if graph is considered "hot" (heavily reused).
    #[must_use]
    pub fn is_hot(&self) -> bool {
        self.reuse_count >= self.hot_threshold
    }

    /// Check if caching should be enabled.
    #[must_use]
    pub fn should_cache(&self) -> bool {
        self.cache_enabled
    }

    /// Get the current reuse count.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.reuse_count
    }

    /// Reset the counter.
    pub fn reset(&mut self) {
        self.reuse_count = 0;
        self.cache_enabled = false;
    }
}

// ============================================================================
// AWP-03: Dual-Waker Payload Backpressure
// ============================================================================

/// Dual-waker state for async backpressure.
///
/// Tracks two wakers: one for the producer, one for the consumer.
/// Enables efficient producer/consumer coordination.
#[derive(Debug, Default)]
pub struct DualWakerState {
    /// Producer is waiting
    producer_waiting: bool,
    /// Consumer is waiting
    consumer_waiting: bool,
    /// Buffer fill level (0-100%)
    fill_percent: u8,
    /// High watermark for backpressure (%)
    high_watermark: u8,
    /// Low watermark for resume (%)
    low_watermark: u8,
}

impl DualWakerState {
    /// Create new state with watermarks.
    pub fn new(low_watermark: u8, high_watermark: u8) -> Self {
        Self {
            producer_waiting: false,
            consumer_waiting: false,
            fill_percent: 0,
            high_watermark: high_watermark.min(100),
            low_watermark: low_watermark.min(high_watermark),
        }
    }

    /// Update fill level and determine who should wake.
    pub fn update_fill(&mut self, fill_percent: u8) -> WakeDecision {
        let old_fill = self.fill_percent;
        self.fill_percent = fill_percent.min(100);

        // Crossed high watermark going up - pause producer
        if old_fill < self.high_watermark && self.fill_percent >= self.high_watermark {
            return WakeDecision::PauseProducer;
        }

        // Crossed low watermark going down - resume producer
        if old_fill > self.low_watermark && self.fill_percent <= self.low_watermark {
            return WakeDecision::WakeProducer;
        }

        // Data available - wake consumer if waiting
        if self.fill_percent > 0 && self.consumer_waiting {
            return WakeDecision::WakeConsumer;
        }

        WakeDecision::None
    }

    /// Producer is now waiting.
    pub fn producer_wait(&mut self) {
        self.producer_waiting = true;
    }

    /// Consumer is now waiting.
    pub fn consumer_wait(&mut self) {
        self.consumer_waiting = true;
    }

    /// Producer was woken.
    pub fn producer_woke(&mut self) {
        self.producer_waiting = false;
    }

    /// Consumer was woken.
    pub fn consumer_woke(&mut self) {
        self.consumer_waiting = false;
    }

    /// Check if producer should be allowed to produce.
    #[must_use]
    pub fn can_produce(&self) -> bool {
        self.fill_percent < self.high_watermark
    }

    /// Check if consumer has data to consume.
    #[must_use]
    pub fn can_consume(&self) -> bool {
        self.fill_percent > 0
    }
}

/// Decision on which waker to invoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WakeDecision {
    /// No action needed
    None,
    /// Wake the producer (buffer drained below low watermark)
    WakeProducer,
    /// Wake the consumer (data available)
    WakeConsumer,
    /// Pause the producer (buffer above high watermark)
    PauseProducer,
}

// ============================================================================
// AWP-04: HTTP/2 Stream Capacity
// ============================================================================

/// HTTP/2 flow control window state.
///
/// Tracks send and receive window sizes for stream-level flow control.
#[derive(Debug, Clone)]
pub struct StreamCapacity {
    /// Connection-level send window
    connection_send: i32,
    /// Stream-level send window
    stream_send: i32,
    /// Receive window (how much we can receive)
    receive_window: i32,
    /// Initial window size
    initial_window: i32,
    /// Whether stream is blocked on flow control
    is_blocked: bool,
}

impl StreamCapacity {
    /// Default window size (HTTP/2 spec: 65535).
    pub const DEFAULT_WINDOW: i32 = 65535;

    /// Create with default windows.
    pub fn new() -> Self {
        Self {
            connection_send: Self::DEFAULT_WINDOW,
            stream_send: Self::DEFAULT_WINDOW,
            receive_window: Self::DEFAULT_WINDOW,
            initial_window: Self::DEFAULT_WINDOW,
            is_blocked: false,
        }
    }

    /// Create with custom initial window.
    pub fn with_initial_window(initial: i32) -> Self {
        Self {
            connection_send: initial,
            stream_send: initial,
            receive_window: initial,
            initial_window: initial,
            is_blocked: false,
        }
    }

    /// Reserve capacity for sending.
    pub fn reserve_send(&mut self, bytes: i32) -> Result<(), FlowControlError> {
        if bytes < 0 {
            return Err(FlowControlError::NegativeReservation);
        }

        let available = self.available_send();
        if bytes > available {
            self.is_blocked = true;
            return Err(FlowControlError::InsufficientCapacity {
                requested: bytes,
                available,
            });
        }

        self.stream_send -= bytes;
        self.connection_send -= bytes;
        self.is_blocked = false;
        Ok(())
    }

    /// Release send capacity (after WINDOW_UPDATE).
    pub fn release_send(&mut self, bytes: i32) {
        self.stream_send += bytes;
        self.connection_send += bytes;
        if self.available_send() > 0 {
            self.is_blocked = false;
        }
    }

    /// Consume receive window (data received).
    pub fn consume_receive(&mut self, bytes: i32) {
        self.receive_window -= bytes;
    }

    /// Replenish receive window (sending WINDOW_UPDATE).
    pub fn replenish_receive(&mut self, bytes: i32) {
        self.receive_window += bytes;
    }

    /// Get available send capacity.
    #[must_use]
    pub fn available_send(&self) -> i32 {
        self.stream_send.min(self.connection_send).max(0)
    }

    /// Get available receive capacity.
    #[must_use]
    pub fn available_receive(&self) -> i32 {
        self.receive_window.max(0)
    }

    /// Check if stream is blocked on flow control.
    #[must_use]
    pub fn is_blocked(&self) -> bool {
        self.is_blocked
    }

    /// Check if receive window needs replenishment.
    #[must_use]
    pub fn needs_window_update(&self) -> bool {
        self.receive_window < self.initial_window / 2
    }
}

impl Default for StreamCapacity {
    fn default() -> Self {
        Self::new()
    }
}

/// Flow control errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlowControlError {
    /// Tried to reserve negative bytes
    NegativeReservation,
    /// Not enough capacity
    InsufficientCapacity { requested: i32, available: i32 },
}

// ============================================================================
// AWP-09: Smart Payload Wake Skip
// ============================================================================

/// Wake skip optimization state.
///
/// Tracks whether a wakeup is actually needed or can be skipped
/// to avoid unnecessary context switches.
#[derive(Debug, Default)]
pub struct WakeSkipState {
    /// Number of items pending
    pending_items: usize,
    /// Whether there's a registered waker
    has_waker: bool,
    /// Last poll had work to do
    last_poll_had_work: bool,
    /// Consecutive empty polls
    empty_poll_count: u32,
    /// Threshold for skipping wakes
    skip_threshold: u32,
}

impl WakeSkipState {
    /// Create with skip threshold.
    pub fn new(skip_threshold: u32) -> Self {
        Self {
            pending_items: 0,
            has_waker: false,
            last_poll_had_work: false,
            empty_poll_count: 0,
            skip_threshold,
        }
    }

    /// Register that a waker exists.
    pub fn register_waker(&mut self) {
        self.has_waker = true;
    }

    /// Clear waker registration.
    pub fn clear_waker(&mut self) {
        self.has_waker = false;
    }

    /// Add pending items.
    pub fn add_pending(&mut self, count: usize) {
        self.pending_items += count;
    }

    /// Remove pending items.
    pub fn remove_pending(&mut self, count: usize) {
        self.pending_items = self.pending_items.saturating_sub(count);
    }

    /// Record poll result.
    pub fn record_poll(&mut self, had_work: bool) {
        self.last_poll_had_work = had_work;
        if had_work {
            self.empty_poll_count = 0;
        } else {
            self.empty_poll_count += 1;
        }
    }

    /// Determine if wake should be skipped.
    #[must_use]
    pub fn should_skip_wake(&self) -> bool {
        // Skip if:
        // 1. No waker registered
        // 2. Already has pending items (will be polled anyway)
        // 3. Had recent empty polls (probably will be empty again)
        if !self.has_waker {
            return true;
        }
        if self.pending_items > 0 && self.last_poll_had_work {
            return true; // Already has work queued
        }
        if self.empty_poll_count >= self.skip_threshold {
            return true; // Likely to be empty again
        }
        false
    }

    /// Check if wake is needed.
    #[must_use]
    pub fn needs_wake(&self) -> bool {
        !self.should_skip_wake() && self.pending_items > 0
    }

    /// Get pending count.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.pending_items
    }

    /// Reset empty poll tracking (after successful wake).
    pub fn reset_tracking(&mut self) {
        self.empty_poll_count = 0;
    }
}
