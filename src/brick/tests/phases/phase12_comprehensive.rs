use super::super::super::*;

// ========================================================================
// Phase 12: Comprehensive Coverage Tests (F216-F246)
// ========================================================================

/// F216: ComputeBackend Display
#[test]
fn test_f216_compute_backend_display() {
    assert_eq!(format!("{}", ComputeBackend::Scalar), "Scalar");
    assert_eq!(format!("{}", ComputeBackend::Sse2), "SSE2");
    assert_eq!(format!("{}", ComputeBackend::Avx2), "AVX2");
    assert_eq!(format!("{}", ComputeBackend::Avx512), "AVX-512");
    assert_eq!(format!("{}", ComputeBackend::Neon), "NEON");
    assert_eq!(format!("{}", ComputeBackend::Wasm), "WASM");
    assert_eq!(format!("{}", ComputeBackend::Cuda), "CUDA");
    assert_eq!(format!("{}", ComputeBackend::Wgpu), "wgpu");
    assert_eq!(format!("{}", ComputeBackend::Auto), "Auto");
}

/// F217: ByteBudget methods
#[test]
fn test_f217_byte_budget_methods() {
    // From throughput
    let budget = ByteBudget::from_throughput(10.0);
    assert!(budget.gb_per_sec > 9.9 && budget.gb_per_sec < 10.1);

    // From latency
    let budget = ByteBudget::from_latency(1.0);
    let expected_throughput = 4096.0 * 1_000_000.0 / 1e9;
    assert!((budget.gb_per_sec - expected_throughput).abs() < 0.001);

    // With page size
    let budget = ByteBudget::from_throughput(10.0).with_page_size(65536);
    assert_eq!(budget.page_size, 65536);

    // To token budget
    let token_budget = budget.to_token_budget();
    assert!(token_budget.us_per_token > 0.0);

    // Is met / utilization
    let budget = ByteBudget::from_latency(10.0);
    assert!(budget.is_met(5.0));
    assert!(!budget.is_met(15.0));
    assert!(budget.utilization(5.0) < 1.0);

    // Throughput from latency
    let throughput = ByteBudget::throughput_from_latency(1.0, 4096);
    assert!(throughput > 0.0);

    // Default
    let budget = ByteBudget::default();
    assert!(budget.gb_per_sec > 20.0); // Default is 25 GB/s
}

/// F218: TokenBudget methods
#[test]
fn test_f218_token_budget_methods() {
    // From latency
    let budget = TokenBudget::from_latency(50.0);
    assert!((budget.tokens_per_sec - 20000.0).abs() < 0.1);

    // From throughput
    let budget = TokenBudget::from_throughput(10000.0);
    assert!((budget.us_per_token - 100.0).abs() < 0.1);

    // With batch size
    let budget = TokenBudget::from_latency(50.0).with_batch_size(4);
    assert_eq!(budget.batch_size, 4);

    // Is met / utilization
    let budget = TokenBudget::from_latency(100.0);
    assert!(budget.is_met(50.0));
    assert!(!budget.is_met(150.0));
    assert!(budget.utilization(50.0) < 1.0);

    // Default
    let budget = TokenBudget::default();
    assert!((budget.us_per_token - 50.0).abs() < 0.1);
}

/// F219: UnrollFactor Debug/Clone
#[test]
fn test_f219_unroll_factor_traits() {
    let factor = UnrollFactor::X4;
    let factor_clone = factor;
    assert_eq!(factor, factor_clone);
    assert!(!format!("{factor:?}").is_empty());

    // PartialEq
    assert_eq!(UnrollFactor::X2, UnrollFactor::X2);
    assert_ne!(UnrollFactor::X2, UnrollFactor::X8);
}

/// F220: SimdBackendState Debug/PartialEq
#[test]
fn test_f220_simd_backend_state_traits() {
    assert_eq!(
        SimdBackendState::Uninitialized,
        SimdBackendState::Uninitialized
    );
    assert_ne!(SimdBackendState::Ready, SimdBackendState::Failed);
    assert!(!format!("{:?}", SimdBackendState::Configuring).is_empty());
}

/// F221: WakeDecision Debug/PartialEq
#[test]
fn test_f221_wake_decision_traits() {
    assert_eq!(WakeDecision::None, WakeDecision::None);
    assert_ne!(WakeDecision::WakeProducer, WakeDecision::WakeConsumer);
    assert!(!format!("{:?}", WakeDecision::PauseProducer).is_empty());
}

/// F222: FlowControlError Debug/Display
#[test]
fn test_f222_flow_control_error_traits() {
    let err = FlowControlError::NegativeReservation;
    assert!(!format!("{err:?}").is_empty());

    let err = FlowControlError::InsufficientCapacity {
        requested: 100,
        available: 50,
    };
    assert!(!format!("{err:?}").is_empty());
}

/// F223: unroll_tail_process with X2 and X8
#[test]
fn test_f223_unroll_tail_process_factors() {
    let data: Vec<i32> = (1..=10).collect();

    // X2 factor
    let results = unroll_tail_process(
        &data,
        UnrollFactor::X2,
        |chunk| chunk.iter().sum::<i32>(),
        |&elem| elem,
    );
    // 5 full chunks: (1+2), (3+4), (5+6), (7+8), (9+10)
    assert_eq!(results, vec![3, 7, 11, 15, 19]);

    // X8 factor
    let results = unroll_tail_process(
        &data,
        UnrollFactor::X8,
        |chunk| chunk.iter().sum::<i32>(),
        |&elem| elem,
    );
    // 1 full chunk: sum(1..=8)=36, tail: 9, 10
    assert_eq!(results, vec![36, 9, 10]);

    // None factor (no unrolling)
    let results = unroll_tail_process(
        &data,
        UnrollFactor::None,
        |chunk| chunk.iter().sum::<i32>(),
        |&elem| elem,
    );
    // 10 chunks of 1 each
    assert_eq!(results, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
}

/// F224: ConnectionState additional coverage
#[test]
fn test_f224_connection_state_all_methods() {
    let mut state = ConnectionState::new();

    // Test all flags
    state.set(ConnectionState::OPEN);
    assert!(state.is_set(ConnectionState::OPEN));

    state.set(ConnectionState::READABLE);
    assert!(state.can_read());

    state.set(ConnectionState::WRITABLE);
    assert!(state.can_write());

    // is_healthy - needs OPEN, not ERROR, not CLOSING
    assert!(state.is_healthy());

    // Clear OPEN and verify
    state.clear(ConnectionState::OPEN);
    assert!(!state.is_healthy());
    assert!(!state.can_read());

    // bits() method
    let bits = state.bits();
    assert!(bits > 0);

    // open_connection starts with OPEN + WRITABLE
    let conn_state = ConnectionState::open_connection();
    assert!(conn_state.is_healthy());
    assert!(conn_state.can_write());

    // ERROR and CLOSING affect is_healthy
    let mut state = ConnectionState::open_connection();
    state.set(ConnectionState::ERROR);
    assert!(!state.is_healthy());

    let mut state = ConnectionState::open_connection();
    state.set(ConnectionState::CLOSING);
    assert!(!state.is_healthy());

    // Test other flags
    let mut state = ConnectionState::new();
    state.set(ConnectionState::HAS_PENDING);
    assert!(state.is_set(ConnectionState::HAS_PENDING));
    state.set(ConnectionState::KEEP_ALIVE);
    assert!(state.is_set(ConnectionState::KEEP_ALIVE));
    state.set(ConnectionState::UPGRADE);
    assert!(state.is_set(ConnectionState::UPGRADE));
}

/// F225: KeepAliveConfig all branches
#[test]
fn test_f225_keep_alive_config_all_branches() {
    // Default
    let config = KeepAliveConfig::new();
    assert!(config.should_keep_alive(1));

    // Disabled
    let config = KeepAliveConfig::disabled();
    assert!(!config.should_keep_alive(1));

    // From header - with max parameter
    let config = KeepAliveConfig::from_header("max=5");
    assert_eq!(config.max_requests, 5);

    // From header - with timeout parameter
    let config = KeepAliveConfig::from_header("timeout=120");
    assert_eq!(config.timeout_secs, 120);

    // Max requests exceeded - uses < comparison
    let config = KeepAliveConfig::from_header("max=3");
    assert!(config.should_keep_alive(2));
    assert!(!config.should_keep_alive(3));

    // Default trait
    let config = KeepAliveConfig::default();
    assert!(config.enabled);
}

/// F226: AsyncResult comprehensive tests
#[test]
fn test_f226_async_result_comprehensive() {
    // Async variant
    let result: AsyncResult<i32, &str> = AsyncResult::Async(42);
    assert!(result.is_async());
    assert!(!result.is_sync());
    assert!(!result.is_error());
    assert_eq!(result.into_result().unwrap(), 42);

    // Sync variant
    let result: AsyncResult<i32, &str> = AsyncResult::Sync(24);
    assert!(!result.is_async());
    assert!(result.is_sync());
    assert!(!result.is_error());
    assert_eq!(result.into_result().unwrap(), 24);

    // Error variant
    let result: AsyncResult<i32, &str> = AsyncResult::Error("oops");
    assert!(!result.is_async());
    assert!(!result.is_sync());
    assert!(result.is_error());
    assert_eq!(result.into_result().unwrap_err(), "oops");

    // Map function - async
    let result: AsyncResult<i32, &str> = AsyncResult::Async(10);
    let mapped = result.map(|x| x * 2);
    assert!(mapped.is_async());
    assert_eq!(mapped.into_result().unwrap(), 20);

    // Map function - sync
    let result: AsyncResult<i32, &str> = AsyncResult::Sync(10);
    let mapped = result.map(|x| x * 3);
    assert!(mapped.is_sync());
    assert_eq!(mapped.into_result().unwrap(), 30);

    // Map function - error (preserves error)
    let result: AsyncResult<i32, &str> = AsyncResult::Error("error");
    let mapped = result.map(|x| x * 2);
    assert!(mapped.is_error());
    assert_eq!(mapped.into_result().unwrap_err(), "error");
}

/// F227: split_batch comprehensive tests
#[test]
fn test_f227_split_batch_comprehensive() {
    // Zero workers
    let chunks = split_batch(100, 0, BatchSplitStrategy::Simple);
    assert!(chunks.is_empty());

    // Zero total
    let chunks = split_batch(0, 4, BatchSplitStrategy::Simple);
    assert!(chunks.is_empty());

    // Simple strategy with remainder
    let chunks = split_batch(10, 3, BatchSplitStrategy::Simple);
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0], 3);
    assert_eq!(chunks[1], 3);
    assert_eq!(chunks[2], 4); // remainder
    assert_eq!(chunks.iter().sum::<usize>(), 10);

    // Equal strategy
    let chunks = split_batch(10, 3, BatchSplitStrategy::Equal);
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks.iter().sum::<usize>(), 10);

    // SequenceAware strategy (same as Equal for now)
    let chunks = split_batch(10, 3, BatchSplitStrategy::SequenceAware);
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks.iter().sum::<usize>(), 10);

    // Perfect division
    let chunks = split_batch(12, 4, BatchSplitStrategy::Simple);
    assert_eq!(chunks, vec![3, 3, 3, 3]);
}

/// F228: PerfMetrics comprehensive tests
#[test]
fn test_f228_perf_metrics_comprehensive() {
    let mut metrics = PerfMetrics::new();

    // Record load
    metrics.record_load(100);
    assert_eq!(metrics.total_ms(), 100);

    // Record prefill
    metrics.record_prefill(50, 10);
    assert_eq!(metrics.total_ms(), 150);
    assert_eq!(metrics.time_to_first_token_ms(), 150);
    assert!(metrics.prefill_tokens_per_second() > 0.0);

    // Record decode
    metrics.record_decode(20);
    assert_eq!(metrics.total_ms(), 170);
    assert!(metrics.tokens_per_second() > 0.0);
    assert!(metrics.avg_token_latency_ms() > 0.0);

    // Record decode batch
    metrics.record_decode_batch(100, 5);
    assert_eq!(metrics.total_ms(), 270);

    // Summary - format is "load: ...total: ..."
    let summary = metrics.summary();
    assert!(summary.contains("total:"));
    assert!(summary.contains("tok/s"));

    // Reset
    metrics.reset();
    assert_eq!(metrics.total_ms(), 0);

    // Default trait
    let metrics = PerfMetrics::default();
    assert_eq!(metrics.total_ms(), 0);
}

/// F229: Balance211Iter tests
#[test]
fn test_f229_balance211_iter() {
    // Basic iteration - returns Range<usize>
    let iter = Balance211Iter::new(10, 3);
    let ranges: Vec<std::ops::Range<usize>> = iter.collect();
    assert_eq!(ranges.len(), 3);

    // Sum of range lengths equals total
    let total: usize = ranges.iter().map(|r| r.len()).sum();
    assert_eq!(total, 10);

    // ExactSizeIterator
    let iter = Balance211Iter::new(10, 3);
    assert_eq!(iter.len(), 3);

    // Edge case: more threads than items
    let iter = Balance211Iter::new(2, 5);
    let ranges: Vec<_> = iter.collect();
    assert!(!ranges.is_empty());

    // balance211 function returns (offset, count) tuples
    let ranges = balance211(100, 4);
    assert_eq!(ranges.len(), 4);
    assert_eq!(ranges.iter().map(|(_, c)| c).sum::<usize>(), 100);
}

/// F230: CacheAligned tests
#[test]
fn test_f230_cache_aligned() {
    // Create
    let aligned = CacheAligned::new(42);
    assert_eq!(*aligned.get(), 42);

    // Mutable access
    let mut aligned = CacheAligned::new(10);
    *aligned.get_mut() += 5;
    assert_eq!(*aligned.get(), 15);

    // Into inner
    let aligned = CacheAligned::new(100);
    assert_eq!(aligned.into_inner(), 100);

    // Default trait
    let aligned: CacheAligned<i32> = CacheAligned::default();
    assert_eq!(*aligned.get(), 0);

    // Clone trait
    let aligned = CacheAligned::new(42);
    let cloned = aligned.clone();
    assert_eq!(*cloned.get(), 42);
}

/// F231: AlignedBuffer tests
#[test]
fn test_f231_aligned_buffer() {
    // Create aligned buffer
    let mut buffer = AlignedBuffer::new(4096).unwrap();
    assert_eq!(buffer.len(), 4096);
    assert!(!buffer.is_empty());

    // Write and read
    buffer.as_mut_slice()[0] = 0xAB;
    assert_eq!(buffer.as_slice()[0], 0xAB);

    // Pointers
    assert!(!buffer.as_ptr().is_null());
    assert!(!buffer.as_mut_ptr().is_null());

    // Alignment check
    assert!(is_direct_io_aligned(buffer.as_ptr()));
}

/// F232: BufferWatermarks tests
#[test]
fn test_f232_buffer_watermarks() {
    // Create watermarks (low=25, high=75)
    let watermarks = BufferWatermarks::new(25, 75);

    // Backpressure when current >= high
    assert!(!watermarks.should_backpressure(50));
    assert!(watermarks.should_backpressure(75));
    assert!(watermarks.should_backpressure(80));

    // can_write when current < low
    assert!(watermarks.can_write(10)); // 10 < 25
    assert!(watermarks.can_write(20)); // 20 < 25
    assert!(!watermarks.can_write(50)); // 50 >= 25

    // Pressure level
    let pressure = watermarks.pressure_level(50);
    assert!(pressure > 0.0 && pressure < 1.0);

    // Default watermarks
    let watermarks = BufferWatermarks::default();
    assert!(watermarks.can_write(0));
}

/// F233: AsyncTaskProfiler tests
#[test]
fn test_f233_async_task_profiler() {
    let mut profiler = AsyncTaskProfiler::new("test_task");

    // Initial state
    assert!(profiler.efficiency().is_nan() || profiler.efficiency() >= 0.0);

    // Simulate polls
    profiler.on_poll_start();
    profiler.on_poll_end(false); // Pending

    profiler.on_poll_start();
    profiler.on_poll_end(true); // Ready

    // Stats
    assert!(profiler.avg_poll_us() >= 0.0);
    assert!(profiler.yield_ratio() >= 0.0 && profiler.yield_ratio() <= 1.0);

    // To execution node
    let _node = profiler.to_execution_node();

    // Default trait
    let profiler = AsyncTaskProfiler::default();
    assert_eq!(profiler.poll_count, 0);
}

/// F234: InferencePhase tests
#[test]
fn test_f234_inference_phase() {
    // All variants
    assert!(!format!("{:?}", InferencePhase::Prefill).is_empty());
    assert!(!format!("{:?}", InferencePhase::Decode).is_empty());

    // PartialEq
    assert_eq!(InferencePhase::Prefill, InferencePhase::Prefill);
    assert_ne!(InferencePhase::Prefill, InferencePhase::Decode);

    // Clone
    let phase = InferencePhase::Prefill;
    let cloned = phase;
    assert_eq!(phase, cloned);

    // Default
    let phase = InferencePhase::default();
    assert_eq!(phase, InferencePhase::Prefill);
}

/// F235: CircuitBreaker comprehensive tests
#[test]
fn test_f235_circuit_breaker_comprehensive() {
    use std::time::Duration;

    let mut breaker = CircuitBreaker::new(2, Duration::from_millis(50));

    // Initial state - closed
    assert_eq!(breaker.state(), CircuitState::Closed);
    assert!(breaker.allow_request());

    // Record failures to open
    breaker.record_failure();
    assert_eq!(breaker.state(), CircuitState::Closed);
    breaker.record_failure();
    assert_eq!(breaker.state(), CircuitState::Open);
    assert!(!breaker.allow_request());

    // Wait for half-open transition
    std::thread::sleep(Duration::from_millis(60));
    // allow_request triggers the state transition
    assert!(breaker.allow_request()); // This transitions to HalfOpen
    assert_eq!(breaker.state(), CircuitState::HalfOpen);

    // Success closes it
    breaker.record_success();
    assert_eq!(breaker.state(), CircuitState::Closed);

    // Reset
    breaker.record_failure();
    breaker.record_failure();
    breaker.reset();
    assert_eq!(breaker.state(), CircuitState::Closed);

    // Default trait
    let breaker = CircuitBreaker::default();
    assert_eq!(breaker.state(), CircuitState::Closed);
}

/// F236: ManagedConnection tests
#[test]
fn test_f236_managed_connection() {
    use std::time::Duration;

    let mut conn = ManagedConnection::new("test", Duration::from_secs(60), Duration::from_secs(30));

    // Initial state
    assert!(conn.is_valid());
    assert!(!conn.is_expired());
    assert!(!conn.is_idle());

    // Access inner
    assert_eq!(*conn.inner(), "test");
    *conn.inner_mut() = "modified";
    assert_eq!(*conn.inner(), "modified");

    // Touch updates idle time
    conn.touch();

    // Health tracking
    conn.record_health_failure();
    conn.reset_health();

    // Age and idle time
    let _age = conn.age();
    let _idle = conn.idle_time();

    // Into inner
    let inner = conn.into_inner();
    assert_eq!(inner, "modified");
}

/// F237: BoundedQueue comprehensive tests
#[test]
fn test_f237_bounded_queue_comprehensive() {
    let mut queue: BoundedQueue<i32> = BoundedQueue::new(3);

    // Initial state
    assert!(queue.is_empty());
    assert!(!queue.is_full());
    assert_eq!(queue.capacity(), 3);
    assert_eq!(queue.remaining(), 3);

    // Push items
    assert!(queue.try_push(1).is_ok());
    assert!(queue.try_push(2).is_ok());
    assert_eq!(queue.len(), 2);
    assert_eq!(queue.remaining(), 1);

    // Peek
    assert_eq!(queue.peek(), Some(&1));

    // Fill queue
    assert!(queue.try_push(3).is_ok());
    assert!(queue.is_full());

    // Push to full queue fails
    assert_eq!(queue.try_push(4), Err(4));

    // Pop
    assert_eq!(queue.pop(), Some(1));
    assert!(!queue.is_full());

    // Clear
    queue.clear();
    assert!(queue.is_empty());

    // Default trait
    let queue: BoundedQueue<i32> = BoundedQueue::default();
    assert!(queue.is_empty());
}

/// F238: StrategicBuffer tests
#[test]
fn test_f238_strategic_buffer() {
    // With strategy
    let mut buffer = StrategicBuffer::new(ReserveStrategy::Double);
    buffer.write(&[1, 2, 3]);
    assert_eq!(buffer.len(), 3);
    assert!(!buffer.is_empty());
    assert_eq!(buffer.as_slice(), &[1, 2, 3]);
    assert!(buffer.capacity() >= 3);

    // Reserve
    buffer.reserve(100);
    assert!(buffer.capacity() >= 103);

    // Clear
    buffer.clear();
    assert!(buffer.is_empty());

    // With capacity
    let buffer = StrategicBuffer::with_capacity(100, ReserveStrategy::Grow50);
    assert!(buffer.capacity() >= 100);

    // Default trait
    let buffer = StrategicBuffer::default();
    assert!(buffer.is_empty());

    // Different strategies
    let _buffer = StrategicBuffer::new(ReserveStrategy::Exact);
    let _buffer = StrategicBuffer::new(ReserveStrategy::PowerOfTwo);
}

/// F239: GraphReuseCounter tests
#[test]
fn test_f239_graph_reuse_counter() {
    let mut counter = GraphReuseCounter::new(5);

    // Initial state
    assert!(!counter.is_hot());
    assert!(!counter.should_cache());
    assert_eq!(counter.count(), 0);

    // Record uses
    counter.record_use();
    counter.record_use();
    counter.record_use();
    assert!(!counter.is_hot());
    assert_eq!(counter.count(), 3);

    // Reach hot threshold
    counter.record_use();
    counter.record_use();
    assert!(counter.is_hot());
    assert!(counter.should_cache());

    // Reset
    counter.reset();
    assert!(!counter.is_hot());
    assert_eq!(counter.count(), 0);
}

/// F240: KvCacheSlot and KvCacheManager
#[test]
fn test_f240_kv_cache() {
    // Create cache manager
    let mut mgr = KvCacheManager::new(3);
    assert_eq!(mgr.capacity(), 3);
    assert_eq!(mgr.valid_count(), 0);

    // Allocate slots
    let idx0 = mgr.allocate(0, 100, 0, 0).unwrap();
    let idx1 = mgr.allocate(1, 101, 0, 0).unwrap();
    assert_eq!(mgr.valid_count(), 2);

    // Access
    let slot = mgr.access(idx0).unwrap();
    assert_eq!(slot.token_id, 100);

    // Step advances global step
    mgr.step();

    // Evict LRU
    let _evicted = mgr.evict_lru();

    // Access
    assert!(mgr.access(idx1).is_some());
}

/// F241: SequentialBatchOrderer tests
#[test]
fn test_f241_sequential_batch_orderer() {
    // Forward order
    let mut orderer = SequentialBatchOrderer::new(3);
    assert!(!orderer.is_done());
    assert_eq!(orderer.remaining(), 3);

    assert_eq!(orderer.next_batch(), Some(0));
    assert_eq!(orderer.next_batch(), Some(1));
    assert_eq!(orderer.next_batch(), Some(2));
    assert_eq!(orderer.next_batch(), None);
    assert!(orderer.is_done());

    // Reset
    orderer.reset();
    assert!(!orderer.is_done());
    assert_eq!(orderer.remaining(), 3);

    // Reversed order
    let mut orderer = SequentialBatchOrderer::reversed(3);
    assert_eq!(orderer.next_batch(), Some(2));
    assert_eq!(orderer.next_batch(), Some(1));
    assert_eq!(orderer.next_batch(), Some(0));

    // Interleaved order
    let mut orderer = SequentialBatchOrderer::interleaved(4);
    let batches: Vec<_> = orderer.by_ref().collect();
    assert_eq!(batches.len(), 4);

    // Iterator trait
    let orderer = SequentialBatchOrderer::new(3);
    let batches: Vec<_> = orderer.collect();
    assert_eq!(batches, vec![0, 1, 2]);
}

/// F242: reserve_capacity and ReserveStrategy
#[test]
fn test_f242_reserve_capacity() {
    // Exact strategy
    assert_eq!(reserve_capacity(10, ReserveStrategy::Exact), 10);

    // Grow50 strategy - adds 50% headroom
    let cap = reserve_capacity(10, ReserveStrategy::Grow50);
    assert!(cap >= 15); // 10 + 50%

    // Double strategy
    let cap = reserve_capacity(10, ReserveStrategy::Double);
    assert_eq!(cap, 20);

    // PowerOfTwo strategy
    let cap = reserve_capacity(10, ReserveStrategy::PowerOfTwo);
    assert_eq!(cap, 16); // next power of two
}

/// F243: ServeLimits tests
#[test]
fn test_f243_serve_limits() {
    // Default limits
    let limits = ServeLimits::default();
    assert!(limits.max_request_size > 0);
    assert!(limits.max_headers > 0);
    assert!(limits.max_header_size > 0);
    assert!(limits.max_pipelined > 0);
    assert!(limits.max_connections > 0);

    // Custom limits
    let limits = ServeLimits {
        max_request_size: 1024,
        max_headers: 10,
        max_header_size: 4096,
        keep_alive_timeout: std::time::Duration::from_secs(30),
        client_timeout: std::time::Duration::from_secs(60),
        max_pipelined: 5,
        max_connections: 100,
    };
    assert_eq!(limits.max_request_size, 1024);
}

/// F244: LimitError Display
#[test]
fn test_f244_limit_error_display() {
    let err = LimitError::BodyTooLarge {
        size: 2000,
        max: 1000,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("2000"));

    let err = LimitError::TooManyHeaders { count: 50, max: 10 };
    let msg = format!("{}", err);
    assert!(msg.contains("50"));

    let err = LimitError::ConnectionLimitReached {
        current: 200,
        max: 100,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("200"));

    let err = LimitError::HeaderTooLarge {
        size: 5000,
        max: 1000,
    };
    let msg = format!("{}", err);
    assert!(msg.contains("5000"));

    let err = LimitError::TooManyPipelined { count: 20, max: 10 };
    let msg = format!("{}", err);
    assert!(msg.contains("20"));
}

/// F245: GracefulShutdown tests
#[test]
fn test_f245_graceful_shutdown() {
    use std::time::Duration;

    let shutdown = GracefulShutdown::new(Duration::from_millis(100));

    // Initial state
    assert!(!shutdown.is_shutdown_requested());
    assert_eq!(shutdown.active_count(), 0);

    // Register guard
    let guard = shutdown.register();
    assert!(guard.is_some());
    assert_eq!(shutdown.active_count(), 1);
    drop(guard);
    assert_eq!(shutdown.active_count(), 0);

    // Shutdown
    let result = shutdown.shutdown();
    assert!(matches!(result, ShutdownResult::Clean));
    assert!(shutdown.is_shutdown_requested());

    // Can't register after shutdown
    let guard = shutdown.register();
    assert!(guard.is_none());

    // Reset
    shutdown.reset();
    assert!(!shutdown.is_shutdown_requested());

    // Default trait
    let shutdown = GracefulShutdown::default();
    assert!(!shutdown.is_shutdown_requested());
}

/// F246: ResourcePool tests
#[test]
fn test_f246_resource_pool() {
    let pool: ResourcePool<i32> = ResourcePool::new(3, || 42);

    // Initial state
    assert_eq!(pool.max_resources(), 3);
    assert_eq!(pool.available(), 3);

    // Acquire resource
    let resource = pool.try_acquire();
    assert!(resource.is_some());
    assert_eq!(pool.available(), 2);

    // Use resource via Deref
    let mut resource = resource.unwrap();
    assert_eq!(*resource, 42);
    *resource = 100;
    assert_eq!(*resource, 100);

    // Drop returns to pool
    drop(resource);
    assert_eq!(pool.available(), 3);

    // Debug trait
    let pool: ResourcePool<i32> = ResourcePool::new(2, || 0);
    let debug = format!("{:?}", pool);
    assert!(debug.contains("ResourcePool"));
}
