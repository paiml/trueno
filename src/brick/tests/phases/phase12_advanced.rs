use super::super::super::*;
use std::time::Duration;

// ========================================================================
// Phase 12: Advanced Features (F177-F215)
// ========================================================================

/// F177: BatchSplitStrategy variants (LCP-09)
#[test]
fn test_f177_batch_split_strategy() {
    let simple = BatchSplitStrategy::Simple;
    let equal = BatchSplitStrategy::Equal;
    let seq_aware = BatchSplitStrategy::SequenceAware;

    // Verify variants exist and are distinct
    assert!(matches!(simple, BatchSplitStrategy::Simple));
    assert!(matches!(equal, BatchSplitStrategy::Equal));
    assert!(matches!(seq_aware, BatchSplitStrategy::SequenceAware));

    // Default should be Simple
    assert!(matches!(
        BatchSplitStrategy::default(),
        BatchSplitStrategy::Simple
    ));
}

/// F178: split_batch correctness (LCP-09)
#[test]
fn test_f178_split_batch() {
    // Simple strategy: 100 items into 4 workers
    let chunks = split_batch(100, 4, BatchSplitStrategy::Simple);
    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks.iter().sum::<usize>(), 100);

    // Equal (Balance211): 50 items with 2 workers - guarantees max-min <= 1
    let chunks = split_batch(50, 2, BatchSplitStrategy::Equal);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks.iter().sum::<usize>(), 50);
    // Balance211 property: max - min <= 1
    let max = *chunks.iter().max().unwrap();
    let min = *chunks.iter().min().unwrap();
    assert!(max - min <= 1);

    // SequenceAware: 1000 items with 4 workers
    let chunks = split_batch(1000, 4, BatchSplitStrategy::SequenceAware);
    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks.iter().sum::<usize>(), 1000);
}

/// F179: AsyncResult states (LCP-12)
#[test]
fn test_f179_async_result() {
    let async_val: AsyncResult<i32, &str> = AsyncResult::Async(42);
    let sync_val: AsyncResult<i32, &str> = AsyncResult::Sync(42);
    let err: AsyncResult<i32, &str> = AsyncResult::Error("fail");

    // Check async/sync detection
    assert!(async_val.is_async());
    assert!(!async_val.is_sync());
    assert!(!async_val.is_error());

    assert!(!sync_val.is_async());
    assert!(sync_val.is_sync());
    assert!(!sync_val.is_error());

    assert!(err.is_error());
    assert!(!err.is_async());
    assert!(!err.is_sync());

    // Extract values using into_result()
    assert_eq!(async_val.into_result(), Ok(42));
    assert_eq!(sync_val.into_result(), Ok(42));
    assert_eq!(err.into_result(), Err("fail"));
}

/// F180: CircuitBreaker initial state (AWP-02)
#[test]
fn test_f180_circuit_breaker_initial() {
    let mut cb = CircuitBreaker::new(3, Duration::from_secs(30));

    // Should start closed
    assert_eq!(cb.state(), CircuitState::Closed);
    assert!(cb.allow_request());
}

/// F181: CircuitBreaker state transitions (AWP-02)
#[test]
fn test_f181_circuit_breaker_transitions() {
    let mut cb = CircuitBreaker::new(3, Duration::from_millis(10));

    // Record failures to open the circuit
    cb.record_failure();
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Closed); // Still closed

    cb.record_failure(); // 3rd failure
    assert_eq!(cb.state(), CircuitState::Open); // Now open
    assert!(!cb.allow_request());

    // Wait for open duration to expire
    std::thread::sleep(Duration::from_millis(15));

    // Now should allow a probe request (half-open)
    assert!(cb.allow_request());
    assert_eq!(cb.state(), CircuitState::HalfOpen);

    // Record success to close
    cb.record_success();
    assert_eq!(cb.state(), CircuitState::Closed);
}

/// F182: ManagedConnection TTL (AWP-06)
#[test]
fn test_f182_managed_connection_ttl() {
    let conn = ManagedConnection::new(
        "test-conn",
        Duration::from_millis(50), // max lifetime
        Duration::from_millis(20), // max idle
    );

    assert!(conn.is_valid());
    assert!(!conn.is_expired());

    // Wait for expiry
    std::thread::sleep(Duration::from_millis(55));
    assert!(conn.is_expired());
    assert!(!conn.is_valid());
}

/// F183: ManagedConnection health (AWP-06)
#[test]
fn test_f183_managed_connection_health() {
    let mut conn = ManagedConnection::new(42i32, Duration::from_secs(60), Duration::from_secs(30));

    assert_eq!(conn.health_failures(), 0);
    assert!(conn.is_valid());

    // Record some failures
    conn.record_health_failure();
    conn.record_health_failure();
    conn.record_health_failure();
    assert_eq!(conn.health_failures(), 3);
    assert!(!conn.is_valid()); // 3+ failures = invalid

    // Reset health
    conn.reset_health();
    assert_eq!(conn.health_failures(), 0);
    assert!(conn.is_valid());
}

/// F184: BoundedQueue push/pop (AWP-11)
#[test]
fn test_f184_bounded_queue_basic() {
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

/// F185: BoundedQueue back-pressure (AWP-11)
#[test]
fn test_f185_bounded_queue_backpressure() {
    let mut queue: BoundedQueue<i32> = BoundedQueue::new(3);

    // Fill the queue
    assert!(queue.try_push(1).is_ok());
    assert!(queue.try_push(2).is_ok());
    assert!(queue.try_push(3).is_ok());
    assert!(queue.is_full());

    // Back-pressure: can't push more
    assert!(queue.try_push(4).is_err());

    // Pop one, now can push
    queue.pop();
    assert!(queue.try_push(4).is_ok());
}

/// F186: ReserveStrategy variants (AWP-13)
#[test]
fn test_f186_reserve_strategy_variants() {
    let exact = ReserveStrategy::Exact;
    let grow = ReserveStrategy::Grow50;
    let double = ReserveStrategy::Double;
    let power = ReserveStrategy::PowerOfTwo;

    // Verify distinct variants
    assert!(matches!(exact, ReserveStrategy::Exact));
    assert!(matches!(grow, ReserveStrategy::Grow50));
    assert!(matches!(double, ReserveStrategy::Double));
    assert!(matches!(power, ReserveStrategy::PowerOfTwo));
}

/// F187: reserve_capacity correctness (AWP-13)
#[test]
fn test_f187_reserve_capacity() {
    // Exact: returns exactly what's needed
    assert_eq!(reserve_capacity(100, ReserveStrategy::Exact), 100);

    // Grow50: adds 50%
    assert_eq!(reserve_capacity(100, ReserveStrategy::Grow50), 150);

    // Double: 2x
    assert_eq!(reserve_capacity(100, ReserveStrategy::Double), 200);

    // PowerOfTwo: next power of 2
    assert_eq!(reserve_capacity(100, ReserveStrategy::PowerOfTwo), 128);
    assert_eq!(reserve_capacity(128, ReserveStrategy::PowerOfTwo), 128);
    assert_eq!(reserve_capacity(129, ReserveStrategy::PowerOfTwo), 256);
}

/// F188: StrategicBuffer operations (AWP-13)
#[test]
fn test_f188_strategic_buffer() {
    let mut buf = StrategicBuffer::new(ReserveStrategy::Double);

    // Initially empty
    assert!(buf.is_empty());

    // Reserve using strategy
    buf.reserve(10);
    assert!(buf.capacity() >= 10); // Reserved at least 10

    // Write bytes
    buf.write(&[1, 2, 3]);
    assert_eq!(buf.len(), 3);

    // Access inner
    assert_eq!(buf.as_slice(), &[1, 2, 3]);

    // Clear and verify
    buf.clear();
    assert!(buf.is_empty());
}

/// F189: AsyncResult map transform (LCP-12)
#[test]
fn test_f189_async_result_map() {
    let async_val: AsyncResult<i32, &str> = AsyncResult::Async(10);
    let mapped = async_val.map(|x| x * 2);
    assert!(mapped.is_async());
    assert_eq!(mapped.into_result(), Ok(20));

    let sync_val: AsyncResult<i32, &str> = AsyncResult::Sync(10);
    let mapped = sync_val.map(|x| x * 2);
    assert!(mapped.is_sync());
    assert_eq!(mapped.into_result(), Ok(20));

    let err: AsyncResult<i32, &str> = AsyncResult::Error("fail");
    let mapped = err.map(|x| x * 2);
    assert!(mapped.is_error());
}

/// F190: split_batch edge cases (LCP-09)
#[test]
fn test_f190_split_batch_edge_cases() {
    // Zero items
    let chunks = split_batch(0, 4, BatchSplitStrategy::Simple);
    assert!(chunks.is_empty());

    // Zero workers
    let chunks = split_batch(100, 0, BatchSplitStrategy::Simple);
    assert!(chunks.is_empty());

    // Single worker gets all items
    let chunks = split_batch(100, 1, BatchSplitStrategy::Simple);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0], 100);

    // Exactly divisible: 64 items, 2 workers with Equal strategy
    let chunks = split_batch(64, 2, BatchSplitStrategy::Equal);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks.iter().sum::<usize>(), 64);
    // Both workers get exactly 32
    assert_eq!(chunks[0], 32);
    assert_eq!(chunks[1], 32);
}

/// F191: GraphReuseCounter hot detection (LCP-08)
#[test]
fn test_f191_graph_reuse_counter() {
    let mut counter = GraphReuseCounter::new(5);

    assert!(!counter.is_hot());
    assert!(!counter.should_cache());
    assert_eq!(counter.count(), 0);

    // Record uses until hot
    for _ in 0..4 {
        counter.record_use();
    }
    assert!(!counter.is_hot());

    counter.record_use(); // 5th use
    assert!(counter.is_hot());
    assert!(counter.should_cache());

    // Reset clears everything
    counter.reset();
    assert!(!counter.is_hot());
    assert_eq!(counter.count(), 0);
}

/// F192: KvCacheSlotInfo eviction priority (LCP-10)
#[test]
fn test_f192_kv_cache_slot_info() {
    let mut slot = KvCacheSlotInfo::new(0, 42, 0, 0);

    assert!(slot.valid);
    assert_eq!(slot.position, 0);
    assert_eq!(slot.token_id, 42);

    // Touch updates last_access
    slot.touch(10);
    assert_eq!(slot.last_access, 10);

    // Eviction priority
    assert_eq!(slot.eviction_priority(10), 0);
    assert_eq!(slot.eviction_priority(20), 10);

    // Invalidate gives max priority
    slot.invalidate();
    assert!(!slot.valid);
    assert_eq!(slot.eviction_priority(100), u64::MAX);
}

/// F193: KvCacheManager allocation and eviction (LCP-10)
#[test]
fn test_f193_kv_cache_manager() {
    let mut mgr = KvCacheManager::new(3);

    assert_eq!(mgr.capacity(), 3);
    assert_eq!(mgr.valid_count(), 0);

    // Allocate slots
    let idx0 = mgr.allocate(0, 100, 0, 0).unwrap();
    mgr.step();
    let idx1 = mgr.allocate(1, 101, 0, 0).unwrap();
    mgr.step();
    let _idx2 = mgr.allocate(2, 102, 0, 0).unwrap();

    assert_eq!(mgr.valid_count(), 3);
    assert!(mgr.allocate(3, 103, 0, 0).is_none()); // Full

    // Access slot 0 to update its last_access
    mgr.step();
    mgr.access(idx0);

    // Evict LRU (should be slot 1, oldest access)
    let evicted = mgr.evict_lru().unwrap();
    assert_eq!(evicted, idx1);
    assert_eq!(mgr.valid_count(), 2);
}

/// F194: SequentialBatchOrderer iteration (LCP-14)
#[test]
fn test_f194_sequential_batch_orderer() {
    // Sequential order
    let mut orderer = SequentialBatchOrderer::new(4);
    assert_eq!(orderer.next_batch(), Some(0));
    assert_eq!(orderer.next_batch(), Some(1));
    assert_eq!(orderer.next_batch(), Some(2));
    assert_eq!(orderer.next_batch(), Some(3));
    assert_eq!(orderer.next_batch(), None);
    assert!(orderer.is_done());

    // Reversed order
    let mut orderer = SequentialBatchOrderer::reversed(3);
    assert_eq!(orderer.next_batch(), Some(2));
    assert_eq!(orderer.next_batch(), Some(1));
    assert_eq!(orderer.next_batch(), Some(0));

    // Reset
    orderer.reset();
    assert_eq!(orderer.remaining(), 3);
}

/// F195: SequentialBatchOrderer interleaved (LCP-14)
#[test]
fn test_f195_batch_orderer_interleaved() {
    // 4 batches: interleaved is 0, 2, 1, 3
    let orderer = SequentialBatchOrderer::interleaved(4);
    let order: Vec<_> = orderer.collect();
    assert_eq!(order, vec![0, 2, 1, 3]);

    // 5 batches: interleaved is 0, 2, 1, 3, 4
    let orderer = SequentialBatchOrderer::interleaved(5);
    let order: Vec<_> = orderer.collect();
    assert_eq!(order.len(), 5);
    // All indices present
    let mut sorted = order.clone();
    sorted.sort();
    assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
}

/// F196: KeepAliveConfig parsing (AWP-10)
#[test]
fn test_f196_keep_alive_config() {
    // Default config
    let config = KeepAliveConfig::new();
    assert!(config.enabled);
    assert_eq!(config.timeout_secs, 60);
    assert_eq!(config.max_requests, 100);

    // Parse from header
    let config = KeepAliveConfig::from_header("timeout=5, max=50");
    assert_eq!(config.timeout_secs, 5);
    assert_eq!(config.max_requests, 50);

    // Disabled config
    let config = KeepAliveConfig::disabled();
    assert!(!config.enabled);
}

/// F197: KeepAliveConfig should_keep_alive (AWP-10)
#[test]
fn test_f197_keep_alive_should() {
    let config = KeepAliveConfig::new(); // max_requests = 100

    assert!(config.should_keep_alive(0));
    assert!(config.should_keep_alive(99));
    assert!(!config.should_keep_alive(100));
    assert!(!config.should_keep_alive(150));

    // Disabled never keeps alive
    let disabled = KeepAliveConfig::disabled();
    assert!(!disabled.should_keep_alive(0));
}

/// F198: ConnectionState bitflags (AWP-12)
#[test]
fn test_f198_connection_state_flags() {
    let mut state = ConnectionState::new();
    assert_eq!(state.bits(), 0);
    assert!(!state.is_healthy());

    // Set flags
    state.set(ConnectionState::OPEN);
    assert!(state.is_set(ConnectionState::OPEN));
    assert!(!state.is_set(ConnectionState::READABLE));

    state.set(ConnectionState::WRITABLE);
    assert!(state.is_healthy());
    assert!(state.can_write());

    // Clear flags
    state.set(ConnectionState::ERROR);
    assert!(!state.is_healthy());

    state.clear(ConnectionState::ERROR);
    assert!(state.is_healthy());
}

/// F199: ConnectionState open_connection (AWP-12)
#[test]
fn test_f199_connection_state_open() {
    let state = ConnectionState::open_connection();

    assert!(state.is_set(ConnectionState::OPEN));
    assert!(state.is_set(ConnectionState::WRITABLE));
    assert!(!state.is_set(ConnectionState::READABLE));
    assert!(state.is_healthy());
    assert!(state.can_write());
    assert!(!state.can_read());
}

/// F200: ConnectionState closing prevents write (AWP-12)
#[test]
fn test_f200_connection_state_closing() {
    let mut state = ConnectionState::open_connection();
    state.set(ConnectionState::READABLE);

    assert!(state.can_read());
    assert!(state.can_write());

    // Set closing
    state.set(ConnectionState::CLOSING);
    assert!(state.can_read()); // Can still read
    assert!(!state.can_write()); // Cannot write when closing
    assert!(!state.is_healthy());
}

/// F201: LazySimdConfig lazy initialization (LCP-07)
#[test]
fn test_f201_lazy_simd_config() {
    let mut config = LazySimdConfig::new();

    // Starts uninitialized
    assert_eq!(config.state(), SimdBackendState::Uninitialized);

    // First ensure_ready initializes
    let backend = config.ensure_ready().unwrap();
    assert_eq!(config.state(), SimdBackendState::Ready);

    // Second call returns immediately
    let backend2 = config.ensure_ready().unwrap();
    assert_eq!(backend, backend2);

    // Reset works
    config.reset();
    assert_eq!(config.state(), SimdBackendState::Uninitialized);
}

/// F202: UnrollFactor values (LCP-13)
#[test]
fn test_f202_unroll_factor() {
    assert_eq!(UnrollFactor::None.value(), 1);
    assert_eq!(UnrollFactor::X2.value(), 2);
    assert_eq!(UnrollFactor::X4.value(), 4);
    assert_eq!(UnrollFactor::X8.value(), 8);

    // Backend selection
    assert_eq!(
        UnrollFactor::for_backend(ComputeBackend::Avx512),
        UnrollFactor::X8
    );
    assert_eq!(
        UnrollFactor::for_backend(ComputeBackend::Avx2),
        UnrollFactor::X4
    );
    assert_eq!(
        UnrollFactor::for_backend(ComputeBackend::Scalar),
        UnrollFactor::None
    );
}

/// F203: UnrollTailIterator chunks and tail (LCP-13)
#[test]
fn test_f203_unroll_tail_iterator() {
    // 10 elements with X4 unroll: 2 full chunks + 2 tail
    let mut iter = UnrollTailIterator::new(10, UnrollFactor::X4);

    assert_eq!(iter.full_iterations(), 2);
    assert_eq!(iter.tail_size(), 2);
    assert!(iter.has_tail());

    // Get chunks
    assert_eq!(iter.next_chunk(), Some((0, 4)));
    assert_eq!(iter.next_chunk(), Some((4, 8)));
    assert_eq!(iter.next_chunk(), None);

    // Get tail
    assert_eq!(iter.tail_range(), Some((8, 10)));
}

/// F204: unroll_tail_process function (LCP-13)
#[test]
fn test_f204_unroll_tail_process() {
    let data: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    let results = unroll_tail_process(
        &data,
        UnrollFactor::X4,
        |chunk| chunk.iter().sum::<i32>(),
        |&elem| elem,
    );

    // 2 chunks: sum(1,2,3,4)=10, sum(5,6,7,8)=20
    // 2 tail elements: 9, 10
    assert_eq!(results, vec![10, 26, 9, 10]);
}

/// F205: DualWakerState watermarks (AWP-03)
#[test]
fn test_f205_dual_waker_state() {
    let mut state = DualWakerState::new(20, 80);

    assert!(state.can_produce());
    assert!(!state.can_consume());

    // Fill to 50%
    let decision = state.update_fill(50);
    assert_eq!(decision, WakeDecision::None);
    assert!(state.can_produce());
    assert!(state.can_consume());

    // Fill to 80% (high watermark)
    let decision = state.update_fill(80);
    assert_eq!(decision, WakeDecision::PauseProducer);
    assert!(!state.can_produce());

    // Drain to 20% (low watermark)
    let decision = state.update_fill(20);
    assert_eq!(decision, WakeDecision::WakeProducer);
    assert!(state.can_produce());
}

/// F206: DualWakerState consumer wake (AWP-03)
#[test]
fn test_f206_dual_waker_consumer_wake() {
    let mut state = DualWakerState::new(20, 80);

    // Consumer waiting with no data
    state.consumer_wait();
    let decision = state.update_fill(0);
    assert_eq!(decision, WakeDecision::None);

    // Data arrives - should wake consumer
    let decision = state.update_fill(10);
    assert_eq!(decision, WakeDecision::WakeConsumer);
}

/// F207: StreamCapacity flow control (AWP-04)
#[test]
fn test_f207_stream_capacity() {
    let mut cap = StreamCapacity::new();

    assert_eq!(cap.available_send(), StreamCapacity::DEFAULT_WINDOW);
    assert!(!cap.is_blocked());

    // Reserve some capacity
    cap.reserve_send(1000).unwrap();
    assert_eq!(cap.available_send(), StreamCapacity::DEFAULT_WINDOW - 1000);

    // Release capacity
    cap.release_send(1000);
    assert_eq!(cap.available_send(), StreamCapacity::DEFAULT_WINDOW);
}

/// F208: StreamCapacity blocking (AWP-04)
#[test]
fn test_f208_stream_capacity_blocking() {
    let mut cap = StreamCapacity::with_initial_window(100);

    // Try to reserve more than available
    let result = cap.reserve_send(150);
    assert!(result.is_err());
    assert!(cap.is_blocked());

    // Negative reservation should fail
    let result = cap.reserve_send(-10);
    assert!(matches!(result, Err(FlowControlError::NegativeReservation)));
}

/// F209: WakeSkipState optimization (AWP-09)
#[test]
fn test_f209_wake_skip_state() {
    let mut state = WakeSkipState::new(3);

    // No waker - should skip
    assert!(state.should_skip_wake());

    // Register waker, no pending - shouldn't skip (might get work soon)
    state.register_waker();
    assert!(!state.should_skip_wake());

    // Add pending and last poll had work - SHOULD skip (will be polled anyway)
    state.add_pending(1);
    state.record_poll(true);
    assert!(state.should_skip_wake()); // Has work queued, will be polled

    // No pending, last poll had no work - shouldn't skip
    state.remove_pending(1);
    state.record_poll(false);
    assert!(!state.should_skip_wake());

    // Multiple empty polls reach threshold
    state.record_poll(false);
    state.record_poll(false);
    assert!(state.should_skip_wake()); // 3 empty polls
}

/// F210: WakeSkipState needs_wake (AWP-09)
#[test]
fn test_f210_wake_skip_needs_wake() {
    let mut state = WakeSkipState::new(5);

    // No waker, no pending - doesn't need wake
    assert!(!state.needs_wake());

    // Has waker and pending - needs wake
    state.register_waker();
    state.add_pending(1);
    assert!(state.needs_wake());

    // Clear waker - doesn't need wake
    state.clear_waker();
    assert!(!state.needs_wake());

    // Remove pending - doesn't need wake
    state.register_waker();
    state.remove_pending(1);
    assert!(!state.needs_wake());
}

/// F211: LazySimdConfig additional methods
#[test]
fn test_f211_lazy_simd_config_methods() {
    let config = LazySimdConfig::new();

    // best_backend returns detected backend
    let backend = config.best_backend();
    assert!(!format!("{backend:?}").is_empty());

    // has_amx check
    let _amx = config.has_amx(); // Just verify it doesn't panic

    // Default trait
    let config2 = LazySimdConfig::default();
    assert_eq!(config2.state(), SimdBackendState::Uninitialized);
}

/// F212: UnrollTailIterator edge cases
#[test]
fn test_f212_unroll_tail_iterator_edge_cases() {
    // Empty data
    let iter = UnrollTailIterator::new(0, UnrollFactor::X4);
    assert_eq!(iter.full_iterations(), 0);
    assert_eq!(iter.tail_size(), 0);
    assert!(!iter.has_tail());
    assert_eq!(iter.tail_range(), None);

    // Exactly divisible
    let iter = UnrollTailIterator::new(8, UnrollFactor::X4);
    assert_eq!(iter.full_iterations(), 2);
    assert_eq!(iter.tail_size(), 0);
    assert!(!iter.has_tail());

    // No unroll factor
    let mut iter = UnrollTailIterator::new(5, UnrollFactor::None);
    assert_eq!(iter.full_iterations(), 5);
    assert_eq!(iter.tail_size(), 0);
    for i in 0..5 {
        assert_eq!(iter.next_chunk(), Some((i, i + 1)));
    }
    assert_eq!(iter.next_chunk(), None);
}

/// F213: DualWakerState edge cases
#[test]
fn test_f213_dual_waker_state_edge_cases() {
    let mut state = DualWakerState::new(20, 80);

    // Test producer/consumer wait/wake cycle
    state.producer_wait();
    state.producer_woke();
    state.consumer_wait();
    state.consumer_woke();

    // Low fill with consumer waiting should wake consumer
    state.consumer_wait();
    let decision = state.update_fill(30);
    assert_eq!(decision, WakeDecision::WakeConsumer);

    // Empty buffer - can't consume
    state.update_fill(0);
    assert!(!state.can_consume());
}

/// F214: StreamCapacity window operations
#[test]
fn test_f214_stream_capacity_window_ops() {
    let mut cap = StreamCapacity::new();

    // Initial state
    assert_eq!(cap.available_receive(), StreamCapacity::DEFAULT_WINDOW);
    assert!(!cap.needs_window_update());

    // Consume receive window
    cap.consume_receive(50000);
    assert_eq!(
        cap.available_receive(),
        StreamCapacity::DEFAULT_WINDOW - 50000
    );

    // Check if needs window update (when < 50% of initial)
    cap.consume_receive(20000);
    assert!(cap.needs_window_update()); // Below 50% threshold

    // Replenish
    cap.replenish_receive(10000);
    assert_eq!(
        cap.available_receive(),
        StreamCapacity::DEFAULT_WINDOW - 60000
    );

    // Default trait
    let cap2 = StreamCapacity::default();
    assert!(!cap2.is_blocked());
}

/// F215: WakeSkipState tracking
#[test]
fn test_f215_wake_skip_state_tracking() {
    let mut state = WakeSkipState::new(2);
    state.register_waker(); // Must register waker for should_skip_wake to work

    // Pending count
    state.add_pending(5);
    assert_eq!(state.pending(), 5);
    state.add_pending(3);
    assert_eq!(state.pending(), 8);
    state.remove_pending(4);
    assert_eq!(state.pending(), 4);

    // Reset tracking
    state.record_poll(false);
    state.record_poll(false);
    state.reset_tracking();
    // After reset, empty poll count is 0, so should not skip (waker is registered)
    assert!(!state.should_skip_wake()); // Reset clears history
}
