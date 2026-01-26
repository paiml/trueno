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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // AsyncResult Tests (LCP-12)
    // ------------------------------------------------------------------------

    #[test]
    fn test_async_result_states() {
        let async_val: AsyncResult<i32, &str> = AsyncResult::Async(42);
        let sync_val: AsyncResult<i32, &str> = AsyncResult::Sync(42);
        let err: AsyncResult<i32, &str> = AsyncResult::Error("fail");

        assert!(async_val.is_async());
        assert!(!async_val.is_sync());
        assert!(!async_val.is_error());

        assert!(!sync_val.is_async());
        assert!(sync_val.is_sync());
        assert!(!sync_val.is_error());

        assert!(err.is_error());
        assert!(!err.is_async());
        assert!(!err.is_sync());

        assert_eq!(async_val.into_result(), Ok(42));
        assert_eq!(sync_val.into_result(), Ok(42));
        assert_eq!(err.into_result(), Err("fail"));
    }

    #[test]
    fn test_async_result_map() {
        let async_val: AsyncResult<i32, &str> = AsyncResult::Async(10);
        let sync_val: AsyncResult<i32, &str> = AsyncResult::Sync(10);
        let err: AsyncResult<i32, &str> = AsyncResult::Error("fail");

        let mapped_async = async_val.map(|x| x * 2);
        let mapped_sync = sync_val.map(|x| x * 2);
        let mapped_err = err.map(|x| x * 2);

        assert!(matches!(mapped_async, AsyncResult::Async(20)));
        assert!(matches!(mapped_sync, AsyncResult::Sync(20)));
        assert!(matches!(mapped_err, AsyncResult::Error("fail")));
    }

    // ------------------------------------------------------------------------
    // BoundedQueue Tests (AWP-11)
    // ------------------------------------------------------------------------

    #[test]
    fn test_bounded_queue_basic() {
        let mut queue: BoundedQueue<i32> = BoundedQueue::new(5);

        assert!(queue.is_empty());
        assert!(!queue.is_full());

        queue.try_push(1).unwrap();
        queue.try_push(2).unwrap();
        queue.try_push(3).unwrap();

        assert_eq!(queue.len(), 3);
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.pop(), Some(2));
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_bounded_queue_backpressure() {
        let mut queue: BoundedQueue<i32> = BoundedQueue::new(3);

        assert!(queue.try_push(1).is_ok());
        assert!(queue.try_push(2).is_ok());
        assert!(queue.try_push(3).is_ok());
        assert!(queue.is_full());

        assert!(queue.try_push(4).is_err());

        queue.pop();
        assert!(queue.try_push(4).is_ok());
    }

    #[test]
    fn test_bounded_queue_comprehensive() {
        let mut queue: BoundedQueue<String> = BoundedQueue::new(3);

        // Test peek
        assert!(queue.peek().is_none());
        queue.try_push("first".to_string()).unwrap();
        assert_eq!(queue.peek(), Some(&"first".to_string()));

        // Test remaining
        assert_eq!(queue.remaining(), 2);
        queue.try_push("second".to_string()).unwrap();
        assert_eq!(queue.remaining(), 1);

        // Test clear
        queue.clear();
        assert!(queue.is_empty());
        assert_eq!(queue.remaining(), 3);
    }

    // ------------------------------------------------------------------------
    // ReserveStrategy Tests (AWP-13)
    // ------------------------------------------------------------------------

    #[test]
    fn test_reserve_strategy_variants() {
        assert_eq!(reserve_capacity(100, ReserveStrategy::Exact), 100);
        assert_eq!(reserve_capacity(100, ReserveStrategy::Grow50), 150);
        assert_eq!(reserve_capacity(100, ReserveStrategy::Double), 200);
        assert_eq!(reserve_capacity(100, ReserveStrategy::PowerOfTwo), 128);
    }

    #[test]
    fn test_reserve_capacity_edge_cases() {
        assert_eq!(reserve_capacity(0, ReserveStrategy::Exact), 0);
        assert_eq!(reserve_capacity(0, ReserveStrategy::Double), 0);
        assert_eq!(reserve_capacity(1, ReserveStrategy::PowerOfTwo), 1);
        assert_eq!(reserve_capacity(3, ReserveStrategy::PowerOfTwo), 4);
    }

    #[test]
    fn test_strategic_buffer() {
        let mut buf = StrategicBuffer::with_capacity(10, ReserveStrategy::Double);
        buf.write(b"hello");
        assert_eq!(buf.len(), 5);
        assert_eq!(buf.as_slice(), b"hello");

        buf.write(b" world");
        assert_eq!(buf.len(), 11);
        assert_eq!(buf.as_slice(), b"hello world");

        buf.clear();
        assert!(buf.is_empty());
    }

    // ------------------------------------------------------------------------
    // GraphReuseCounter Tests (LCP-08)
    // ------------------------------------------------------------------------

    #[test]
    fn test_graph_reuse_counter() {
        let mut counter = GraphReuseCounter::new(3);

        assert!(!counter.is_hot());
        assert!(!counter.should_cache());

        counter.record_use();
        counter.record_use();
        assert!(!counter.is_hot());

        counter.record_use();
        assert!(counter.is_hot());
        assert!(counter.should_cache());
        assert_eq!(counter.count(), 3);

        counter.reset();
        assert!(!counter.is_hot());
        assert_eq!(counter.count(), 0);
    }

    // ------------------------------------------------------------------------
    // DualWakerState Tests (AWP-03)
    // ------------------------------------------------------------------------

    #[test]
    fn test_dual_waker_state() {
        let mut state = DualWakerState::new(20, 80);

        assert!(state.can_produce());
        assert!(!state.can_consume());

        // Fill above high watermark
        let decision = state.update_fill(85);
        assert_eq!(decision, WakeDecision::PauseProducer);
        assert!(!state.can_produce());
        assert!(state.can_consume());

        // Drain below low watermark
        let decision = state.update_fill(15);
        assert_eq!(decision, WakeDecision::WakeProducer);
        assert!(state.can_produce());
    }

    #[test]
    fn test_dual_waker_consumer_wake() {
        let mut state = DualWakerState::new(20, 80);
        state.consumer_wait();

        // Add data while consumer waiting
        let decision = state.update_fill(50);
        assert_eq!(decision, WakeDecision::WakeConsumer);
    }

    #[test]
    fn test_dual_waker_edge_cases() {
        let mut state = DualWakerState::new(20, 80);

        state.producer_wait();
        state.consumer_wait();

        state.producer_woke();
        state.consumer_woke();

        // Test clamping
        state.update_fill(200); // Should clamp to 100
        assert!(!state.can_produce());
    }

    // ------------------------------------------------------------------------
    // StreamCapacity Tests (AWP-04)
    // ------------------------------------------------------------------------

    #[test]
    fn test_stream_capacity_basic() {
        let mut cap = StreamCapacity::new();

        assert_eq!(cap.available_send(), StreamCapacity::DEFAULT_WINDOW);
        assert!(!cap.is_blocked());

        cap.reserve_send(1000).unwrap();
        assert_eq!(cap.available_send(), StreamCapacity::DEFAULT_WINDOW - 1000);
    }

    #[test]
    fn test_stream_capacity_blocking() {
        let mut cap = StreamCapacity::with_initial_window(100);

        let result = cap.reserve_send(150);
        assert!(result.is_err());
        assert!(cap.is_blocked());

        cap.release_send(100);
        assert!(!cap.is_blocked());
    }

    #[test]
    fn test_stream_capacity_window_ops() {
        let mut cap = StreamCapacity::new();
        // Default window is 65535, needs_window_update when < 32767

        // Consume more than half to trigger update
        cap.consume_receive(40000);
        assert!(cap.needs_window_update()); // 25535 < 32767

        cap.replenish_receive(20000);
        assert!(!cap.needs_window_update()); // 45535 > 32767
    }

    #[test]
    fn test_flow_control_error() {
        let err1 = FlowControlError::NegativeReservation;
        let err2 = FlowControlError::InsufficientCapacity {
            requested: 100,
            available: 50,
        };

        assert_eq!(err1, FlowControlError::NegativeReservation);
        assert!(matches!(
            err2,
            FlowControlError::InsufficientCapacity { .. }
        ));
    }

    // ------------------------------------------------------------------------
    // WakeSkipState Tests (AWP-09)
    // ------------------------------------------------------------------------

    #[test]
    fn test_wake_skip_state() {
        let mut state = WakeSkipState::new(3);

        // No waker = always skip
        assert!(state.should_skip_wake());

        state.register_waker();
        assert!(!state.should_skip_wake());

        // Accumulate empty polls
        state.record_poll(false);
        state.record_poll(false);
        state.record_poll(false);
        assert!(state.should_skip_wake());

        // Reset tracking
        state.reset_tracking();
        assert!(!state.should_skip_wake());
    }

    #[test]
    fn test_wake_skip_needs_wake() {
        let mut state = WakeSkipState::new(3);
        state.register_waker();

        // No pending items
        assert!(!state.needs_wake());

        // Add pending
        state.add_pending(5);
        assert!(state.needs_wake());
        assert_eq!(state.pending(), 5);

        // Remove some
        state.remove_pending(3);
        assert_eq!(state.pending(), 2);
    }

    #[test]
    fn test_wake_skip_tracking() {
        let mut state = WakeSkipState::new(5);
        state.register_waker();

        // Work resets empty poll count
        state.record_poll(false);
        state.record_poll(false);
        state.record_poll(true); // Had work
        state.record_poll(false);

        // Should not skip yet (only 1 empty after work)
        assert!(!state.should_skip_wake());
    }

    // ------------------------------------------------------------------------
    // Falsification Tests
    // ------------------------------------------------------------------------

    /// FALSIFICATION: BoundedQueue capacity invariant.
    /// Queue must never exceed its capacity.
    #[test]
    fn test_falsify_bounded_queue_capacity_invariant() {
        for cap in [1, 5, 10, 100] {
            let mut queue: BoundedQueue<usize> = BoundedQueue::new(cap);

            // Try to push more than capacity
            for i in 0..cap * 2 {
                let _ = queue.try_push(i);
            }

            assert!(
                queue.len() <= cap,
                "FALSIFICATION FAILED: Queue exceeded capacity {} with len {}",
                cap,
                queue.len()
            );
        }
    }

    /// FALSIFICATION: ReserveStrategy must always return >= needed.
    #[test]
    fn test_falsify_reserve_strategy_minimum() {
        for needed in [0, 1, 10, 100, 1000, 10000] {
            for strategy in [
                ReserveStrategy::Exact,
                ReserveStrategy::Grow50,
                ReserveStrategy::Double,
                ReserveStrategy::PowerOfTwo,
            ] {
                let reserved = reserve_capacity(needed, strategy);
                assert!(
                    reserved >= needed,
                    "FALSIFICATION FAILED: reserve_capacity({}, {:?}) = {} < {}",
                    needed,
                    strategy,
                    reserved,
                    needed
                );
            }
        }
    }

    /// FALSIFICATION: GraphReuseCounter hot transition.
    /// Must become hot exactly at threshold.
    #[test]
    fn test_falsify_graph_reuse_threshold() {
        for threshold in [1, 5, 10, 100] {
            let mut counter = GraphReuseCounter::new(threshold);

            // Before threshold
            for _ in 0..threshold - 1 {
                counter.record_use();
                assert!(
                    !counter.is_hot(),
                    "FALSIFICATION FAILED: Became hot before threshold {}",
                    threshold
                );
            }

            // At threshold
            counter.record_use();
            assert!(
                counter.is_hot(),
                "FALSIFICATION FAILED: Not hot at threshold {}",
                threshold
            );
        }
    }

    /// FALSIFICATION: StreamCapacity window consistency.
    #[test]
    fn test_falsify_stream_capacity_consistency() {
        let mut cap = StreamCapacity::new();
        let initial = cap.available_send();

        // Reserve and release should return to initial
        cap.reserve_send(10000).unwrap();
        cap.release_send(10000);
        assert_eq!(
            cap.available_send(),
            initial,
            "FALSIFICATION FAILED: Window not restored after reserve+release"
        );
    }
}
