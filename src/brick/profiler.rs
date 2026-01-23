//! BrickProfiler: Token-Centric Profiling System
//!
//! TILING-SPEC-001: Tile-Level Profiling Support
//!
//! This module provides hierarchical profiling for compute bricks:
//! - Per-brick timing and throughput (PAR-073)
//! - O(1) hot path with BrickId enum (PAR-200)
//! - Tile-level profiling for cache-blocked operations (TILING-SPEC-001)
//! - Kernel checksum capture for divergence detection (CORRECTNESS-011)

use std::fmt;
use std::time::Instant;

use super::exec_graph::{
    BrickBottleneck, BrickCategory, BrickId, BrickStats, CategoryStats, ExecutionGraph,
    ExecutionNode, ExecutionNodeId, SyncMode,
};

// ============================================================================
// TILING-SPEC-001: Tile-Level Profiling Support
// ============================================================================

/// Tile-level profiling statistics.
///
/// Tracks per-tile performance metrics for hierarchical cache-blocked operations.
/// Used in conjunction with `TcbGeometry` and `TilingConfig` from the tiling module.
///
/// # Example
///
/// ```ignore
/// let mut profiler = BrickProfiler::new();
/// profiler.enable();
///
/// let tile_timer = profiler.start_tile(TileLevel::Macro, 0, 0);
/// // ... execute tile ...
/// profiler.stop_tile(tile_timer, 1024 * 1024);
/// ```
#[derive(Debug, Clone, Default)]
pub struct TileStats {
    /// Tile level (Macro/Midi/Micro)
    pub level: TileLevel,
    /// Total samples collected
    pub count: u64,
    /// Total elapsed time (nanoseconds)
    pub total_ns: u64,
    /// Min elapsed time (nanoseconds)
    pub min_ns: u64,
    /// Max elapsed time (nanoseconds)
    pub max_ns: u64,
    /// Total elements processed
    pub total_elements: u64,
    /// Total cache misses (estimated)
    pub cache_misses: u64,
    /// Total arithmetic operations
    pub total_flops: u64,
}

/// Tile hierarchy level for profiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TileLevel {
    /// Macro-tile: L3 cache / GPU global memory
    #[default]
    Macro,
    /// Midi-tile: L2 cache / GPU shared memory
    Midi,
    /// Micro-tile: Registers / SIMD lanes
    Micro,
}

impl TileLevel {
    /// Get the name of this tile level.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            TileLevel::Macro => "macro",
            TileLevel::Midi => "midi",
            TileLevel::Micro => "micro",
        }
    }
}

impl TileStats {
    /// Create new tile stats for a given level.
    pub fn new(level: TileLevel) -> Self {
        Self {
            level,
            count: 0,
            total_ns: 0,
            min_ns: u64::MAX,
            max_ns: 0,
            total_elements: 0,
            cache_misses: 0,
            total_flops: 0,
        }
    }

    /// Add a sample to statistics.
    pub fn add_sample(&mut self, elapsed_ns: u64, elements: u64, flops: u64) {
        self.count += 1;
        self.total_ns += elapsed_ns;
        self.min_ns = self.min_ns.min(elapsed_ns);
        self.max_ns = self.max_ns.max(elapsed_ns);
        self.total_elements += elements;
        self.total_flops += flops;
    }

    /// Average time in microseconds.
    #[must_use]
    pub fn avg_us(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.total_ns as f64 / self.count as f64 / 1000.0
        }
    }

    /// Throughput in elements/second.
    #[must_use]
    pub fn throughput(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.total_elements as f64 / (self.total_ns as f64 / 1_000_000_000.0)
        }
    }

    /// Compute throughput in GFLOP/s.
    #[must_use]
    pub fn gflops(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.total_flops as f64 / (self.total_ns as f64 / 1_000_000_000.0) / 1e9
        }
    }

    /// Arithmetic intensity (FLOP/byte) estimate.
    ///
    /// Assumes 4 bytes per element (f32).
    #[must_use]
    pub fn arithmetic_intensity(&self) -> f64 {
        if self.total_elements == 0 {
            0.0
        } else {
            self.total_flops as f64 / (self.total_elements as f64 * 4.0)
        }
    }

    /// Estimated cache efficiency (0.0-1.0).
    ///
    /// Based on ratio of actual throughput vs theoretical peak.
    #[must_use]
    pub fn cache_efficiency(&self, peak_gflops: f64) -> f64 {
        if peak_gflops <= 0.0 {
            0.0
        } else {
            (self.gflops() / peak_gflops).min(1.0)
        }
    }
}

/// Timer handle for tile-level profiling.
#[derive(Debug)]
pub struct TileTimer {
    /// Tile level
    level: TileLevel,
    /// Row index within parent tile (reserved for spatial analysis)
    _row: u32,
    /// Column index within parent tile (reserved for spatial analysis)
    _col: u32,
    /// Start time
    start: Instant,
}

/// Pending measurement for deferred sync mode.
#[derive(Debug, Clone)]
struct PendingMeasurement {
    /// Brick ID (if known)
    brick_id: Option<BrickId>,
    /// Brick name (for dynamic bricks)
    name: Option<String>,
    /// Start time in nanoseconds (from Instant::now())
    start_ns: u64,
    /// Number of elements processed
    elements: u64,
}

/// Per-brick profiler using pure Rust timing.
///
/// # Design (PAR-073, PAR-200)
///
/// - Uses `std::time::Instant` for timing (no CUDA event FFI)
/// - PAR-200: O(1) hot path with `BrickId` enum + array storage
/// - GPU operations require explicit sync before timing point
/// - Supports deferred sync mode for low-overhead production profiling
/// - Aggregates statistics per brick name
///
/// # Usage
///
/// ```rust,ignore
/// use trueno::brick::{BrickProfiler, BrickId, SyncMode};
///
/// let mut profiler = BrickProfiler::new();
/// profiler.enable();
///
/// // Fast path: use BrickId for known bricks (PAR-200)
/// let timer = profiler.start_brick(BrickId::RmsNorm);
/// // ... do work ...
/// // For GPU: cuda_stream.synchronize() HERE
/// profiler.stop_brick(timer, 1);
///
/// // Legacy path: string-based (slower, for unknown bricks)
/// let timer = profiler.start("CustomBrick");
/// profiler.stop(timer, 1);
///
/// // Deferred sync mode (production)
/// profiler.set_sync_mode(SyncMode::Deferred);
/// profiler.record_deferred(BrickId::RmsNorm, start_ns, 1);
/// // ... more operations ...
/// cuda_stream.synchronize();
/// profiler.finalize(end_ns);
///
/// // Get statistics
/// let stats = profiler.brick_stats(BrickId::RmsNorm);
/// println!("RmsNorm avg: {:.2}µs", stats.avg_us());
///
/// // Get category breakdown
/// let cats = profiler.category_stats();
/// println!("Attention: {:.1}%", cats[BrickCategory::Attention as usize].percentage(profiler.total_ns()));
/// ```
#[derive(Debug)]
pub struct BrickProfiler {
    // PAR-200: Fast path - pre-allocated array for known bricks
    /// Per-brick statistics for known BrickId types (O(1) lookup)
    brick_stats: [BrickStats; BrickId::COUNT],

    // Legacy path - HashMap for dynamic/unknown brick names
    /// Per-brick statistics for unknown brick names (slower, O(1) amortized)
    dynamic_stats: std::collections::HashMap<String, BrickStats>,

    // PAR-200: Deferred sync support
    /// Pending measurements awaiting GPU sync
    pending: Vec<PendingMeasurement>,
    /// Synchronization mode
    sync_mode: SyncMode,
    /// Reference instant for deferred timing
    epoch: Instant,

    /// Whether profiling is enabled
    enabled: bool,
    /// Total tokens processed
    total_tokens: u64,
    /// Total time (ns) across all bricks
    total_ns: u64,
    /// L2 cache hit rate (0.0-1.0) - v1.1.0 OBSERVE phase
    l2_cache_hit_rate: Option<f32>,
    /// Whether zero-copy memory transfers are enabled - v1.1.0 OBSERVE phase
    is_zero_copy: bool,
    /// CORRECTNESS-011: Per-kernel checksums for divergence detection
    kernel_checksums: Vec<KernelChecksum>,

    // PAR-201: Execution path graph
    /// Whether execution graph tracking is enabled
    graph_enabled: bool,
    /// Execution path graph for PTX→kernel→brick relationships
    execution_graph: ExecutionGraph,

    // TILING-SPEC-001: Tile-level profiling
    /// Per-level tile statistics (Macro, Midi, Micro)
    tile_stats: [TileStats; 3],
    /// Whether tile profiling is enabled
    tile_profiling_enabled: bool,
}

/// Timer handle returned by `start()` (legacy string-based API).
#[derive(Debug)]
pub struct BrickTimer {
    /// Brick name
    name: String,
    /// Start time
    start: Instant,
}

/// Timer handle returned by `start_brick()` (PAR-200 fast path).
#[derive(Debug)]
pub struct BrickIdTimer {
    /// Brick ID
    brick_id: BrickId,
    /// Start time
    start: Instant,
}

impl Default for BrickProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickProfiler {
    /// Create a new profiler (disabled by default for zero overhead).
    pub fn new() -> Self {
        Self {
            brick_stats: std::array::from_fn(|i| {
                // Safety: i < BrickId::COUNT by construction
                let brick_id = unsafe { std::mem::transmute::<u8, BrickId>(i as u8) };
                BrickStats::new(brick_id.name())
            }),
            dynamic_stats: std::collections::HashMap::new(),
            pending: Vec::new(),
            sync_mode: SyncMode::Deferred,
            epoch: Instant::now(),
            enabled: false,
            total_tokens: 0,
            total_ns: 0,
            l2_cache_hit_rate: None,
            is_zero_copy: false,
            kernel_checksums: Vec::new(),
            graph_enabled: false,
            execution_graph: ExecutionGraph::new(),
            tile_stats: [
                TileStats::new(TileLevel::Macro),
                TileStats::new(TileLevel::Midi),
                TileStats::new(TileLevel::Micro),
            ],
            tile_profiling_enabled: false,
        }
    }

    /// Create an enabled profiler.
    pub fn enabled() -> Self {
        let mut profiler = Self::new();
        profiler.enabled = true;
        profiler
    }

    // ========================================================================
    // PAR-200: Sync Mode Configuration
    // ========================================================================

    /// Set the synchronization mode for GPU profiling.
    ///
    /// # Modes
    /// - `Immediate`: Sync after each kernel (accurate but slow)
    /// - `PerLayer`: Sync once per transformer layer
    /// - `Deferred`: Sync once per forward pass (default, fast)
    /// - `None`: No synchronization
    pub fn set_sync_mode(&mut self, mode: SyncMode) {
        self.sync_mode = mode;
    }

    /// Get the current synchronization mode.
    #[must_use]
    pub fn sync_mode(&self) -> SyncMode {
        self.sync_mode
    }

    /// Reset the epoch for deferred timing.
    /// Call this at the start of a forward pass.
    pub fn reset_epoch(&mut self) {
        self.epoch = Instant::now();
    }

    /// Get nanoseconds elapsed since epoch.
    #[inline]
    pub fn elapsed_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    // ========================================================================
    // PAR-200: Fast Path API (BrickId-based)
    // ========================================================================

    /// Start timing a brick using BrickId (O(1) hot path).
    ///
    /// This is the preferred API for known brick types.
    /// For GPU operations, call `stream.synchronize()` before `stop_brick()`.
    #[inline]
    #[must_use]
    pub fn start_brick(&self, brick_id: BrickId) -> BrickIdTimer {
        BrickIdTimer {
            brick_id,
            start: Instant::now(),
        }
    }

    /// Stop timing and record the sample (O(1) hot path).
    #[inline]
    pub fn stop_brick(&mut self, timer: BrickIdTimer, elements: u64) {
        if !self.enabled {
            return;
        }

        let elapsed = timer.start.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;

        // O(1) array access
        let stats = &mut self.brick_stats[timer.brick_id as usize];
        stats.add_sample(elapsed_ns, elements);

        // Update totals
        self.total_tokens += elements;
        self.total_ns += elapsed_ns;
    }

    /// Get statistics for a known brick type (O(1)).
    #[inline]
    #[must_use]
    pub fn brick_stats(&self, brick_id: BrickId) -> &BrickStats {
        &self.brick_stats[brick_id as usize]
    }

    /// Get mutable statistics for a known brick type (O(1)).
    #[inline]
    pub fn brick_stats_mut(&mut self, brick_id: BrickId) -> &mut BrickStats {
        &mut self.brick_stats[brick_id as usize]
    }

    // ========================================================================
    // PAR-200: Deferred Sync API
    // ========================================================================

    /// Record a measurement without GPU sync (deferred mode).
    ///
    /// Call `finalize()` after GPU sync to apply all pending measurements.
    ///
    /// # Arguments
    /// - `brick_id`: The brick type
    /// - `start_ns`: Start time (from `elapsed_ns()` at operation start)
    /// - `elements`: Number of elements processed
    #[inline]
    pub fn record_deferred(&mut self, brick_id: BrickId, start_ns: u64, elements: u64) {
        if !self.enabled {
            return;
        }
        self.pending.push(PendingMeasurement {
            brick_id: Some(brick_id),
            name: None,
            start_ns,
            elements,
        });
    }

    /// Record a measurement for a dynamic brick (deferred mode).
    #[inline]
    pub fn record_deferred_dynamic(&mut self, name: &str, start_ns: u64, elements: u64) {
        if !self.enabled {
            return;
        }
        self.pending.push(PendingMeasurement {
            brick_id: BrickId::from_str(name),
            name: Some(name.to_string()),
            start_ns,
            elements,
        });
    }

    /// Finalize all pending measurements after GPU sync.
    ///
    /// Must be called after `stream.synchronize()` to get accurate timing.
    ///
    /// # Arguments
    /// - `end_ns`: End time (from `elapsed_ns()` after sync)
    pub fn finalize(&mut self, end_ns: u64) {
        if self.pending.is_empty() {
            return;
        }

        // Calculate elapsed time for each pending measurement
        for m in self.pending.drain(..) {
            let elapsed_ns = end_ns.saturating_sub(m.start_ns);

            if let Some(brick_id) = m.brick_id {
                // Fast path: known brick
                let stats = &mut self.brick_stats[brick_id as usize];
                stats.add_sample(elapsed_ns, m.elements);
            } else if let Some(name) = m.name {
                // Slow path: dynamic brick
                let stats = self
                    .dynamic_stats
                    .entry(name.clone())
                    .or_insert_with(|| BrickStats::new(&name));
                stats.add_sample(elapsed_ns, m.elements);
            }

            self.total_tokens += m.elements;
            self.total_ns += elapsed_ns;
        }
    }

    /// Check if there are pending measurements.
    #[inline]
    #[must_use]
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Get number of pending measurements.
    #[inline]
    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    // ========================================================================
    // PAR-200: Category Aggregation
    // ========================================================================

    /// Get aggregated statistics by category.
    ///
    /// Returns an array indexed by `BrickCategory as usize`.
    #[must_use]
    pub fn category_stats(&self) -> [CategoryStats; BrickCategory::COUNT] {
        let mut result = [CategoryStats::default(); BrickCategory::COUNT];

        for (i, stats) in self.brick_stats.iter().enumerate() {
            // Safety: i < BrickId::COUNT by construction
            let brick_id = unsafe { std::mem::transmute::<u8, BrickId>(i as u8) };
            let cat = brick_id.category() as usize;
            result[cat].total_ns += stats.total_ns;
            result[cat].total_elements += stats.total_elements;
            result[cat].count += stats.count;
        }

        // Include dynamic stats in "Other" category
        for stats in self.dynamic_stats.values() {
            let cat = BrickCategory::Other as usize;
            result[cat].total_ns += stats.total_ns;
            result[cat].total_elements += stats.total_elements;
            result[cat].count += stats.count;
        }

        result
    }

    /// Print category breakdown to console.
    pub fn print_category_stats(&self) {
        let cats = self.category_stats();
        let total = self.total_ns;

        println!("╭─────────────────────────────────────────────────────────╮");
        println!("│            Category Breakdown (PAR-200)                 │");
        println!("├─────────────────────────────────────────────────────────┤");
        for (i, cat_stats) in cats.iter().enumerate() {
            // Safety: i < BrickCategory::COUNT
            let cat = unsafe { std::mem::transmute::<u8, BrickCategory>(i as u8) };
            if cat_stats.count > 0 {
                println!(
                    "│ {:12} {:8.2}µs avg {:6.1}% [{:5} samples]        │",
                    cat.name(),
                    cat_stats.avg_us(),
                    cat_stats.percentage(total),
                    cat_stats.count
                );
            }
        }
        println!("╰─────────────────────────────────────────────────────────╯");
    }

    // ========================================================================
    // PAR-201: Execution Path Graph
    // ========================================================================

    /// Enable execution graph tracking.
    ///
    /// When enabled, the profiler records the execution hierarchy:
    /// - Layer → Brick → Kernel relationships
    /// - PTX hashes for kernel identity
    /// - Timing data per node
    pub fn enable_graph(&mut self) {
        self.graph_enabled = true;
    }

    /// Disable execution graph tracking.
    pub fn disable_graph(&mut self) {
        self.graph_enabled = false;
    }

    /// Check if execution graph tracking is enabled.
    #[must_use]
    pub fn is_graph_enabled(&self) -> bool {
        self.graph_enabled
    }

    /// Get the execution graph (immutable).
    #[must_use]
    pub fn execution_graph(&self) -> &ExecutionGraph {
        &self.execution_graph
    }

    /// Get the execution graph (mutable).
    pub fn execution_graph_mut(&mut self) -> &mut ExecutionGraph {
        &mut self.execution_graph
    }

    /// Push a scope for hierarchical graph recording.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// profiler.enable_graph();
    /// profiler.graph_push_scope(ExecutionNode::Layer { index: 0 });
    /// // ... record bricks and kernels ...
    /// profiler.graph_pop_scope();
    /// ```
    pub fn graph_push_scope(&mut self, node: ExecutionNode) -> Option<ExecutionNodeId> {
        if !self.graph_enabled {
            return None;
        }
        Some(self.execution_graph.push_scope(node))
    }

    /// Pop the current scope.
    pub fn graph_pop_scope(&mut self) -> Option<ExecutionNodeId> {
        if !self.graph_enabled {
            return None;
        }
        self.execution_graph.pop_scope()
    }

    /// Record a brick in the execution graph.
    ///
    /// This should be called after `stop_brick()` with the timing data.
    pub fn graph_record_brick(
        &mut self,
        brick_id: BrickId,
        timing_ns: u64,
        elements: u64,
    ) -> Option<ExecutionNodeId> {
        if !self.graph_enabled {
            return None;
        }
        let node = ExecutionNode::Brick {
            id: brick_id,
            timing_ns,
            elements,
        };
        Some(self.execution_graph.add_node_in_scope(node))
    }

    /// Record a kernel launch in the execution graph.
    ///
    /// # Arguments
    /// - `name`: Kernel name (e.g., "batched_q4k_gemv")
    /// - `ptx_hash`: FNV-1a hash of PTX source for identity
    /// - `grid`: Grid dimensions (blocks)
    /// - `block`: Block dimensions (threads)
    /// - `shared_mem`: Shared memory bytes
    pub fn graph_record_kernel(
        &mut self,
        name: &str,
        ptx_hash: u64,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
    ) -> Option<ExecutionNodeId> {
        if !self.graph_enabled {
            return None;
        }
        Some(
            self.execution_graph
                .record_kernel_launch(name, ptx_hash, grid, block, shared_mem),
        )
    }

    /// Export execution graph to DOT format for visualization.
    ///
    /// Use with Graphviz: `dot -Tsvg output.dot -o graph.svg`
    #[must_use]
    pub fn graph_to_dot(&self) -> String {
        self.execution_graph.to_dot()
    }

    /// Export execution graph to trueno-graph CsrGraph.
    #[cfg(feature = "execution-graph")]
    #[must_use]
    pub fn graph_to_csr(&self) -> trueno_graph::CsrGraph {
        self.execution_graph.to_csr()
    }

    /// Clear the execution graph.
    pub fn graph_clear(&mut self) {
        self.execution_graph.clear();
    }

    /// Check if the execution graph scope stack is balanced.
    #[must_use]
    pub fn graph_is_scope_balanced(&self) -> bool {
        self.execution_graph.is_scope_balanced()
    }

    /// Set L2 cache hit rate (v1.1.0 OBSERVE phase)
    pub fn set_l2_cache_hit_rate(&mut self, rate: f32) {
        self.l2_cache_hit_rate = Some(rate.clamp(0.0, 1.0));
    }

    /// Get L2 cache hit rate
    pub fn l2_cache_hit_rate(&self) -> Option<f32> {
        self.l2_cache_hit_rate
    }

    /// Set zero-copy mode (v1.1.0 OBSERVE phase)
    pub fn set_zero_copy(&mut self, enabled: bool) {
        self.is_zero_copy = enabled;
    }

    /// Check if zero-copy is enabled
    pub fn is_zero_copy(&self) -> bool {
        self.is_zero_copy
    }

    /// Enable profiling.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable profiling.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Check if profiling is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Start timing a brick. Returns timer handle.
    ///
    /// IMPORTANT: For GPU operations, call sync AFTER the operation
    /// completes but BEFORE calling stop().
    #[must_use]
    pub fn start(&self, name: &str) -> BrickTimer {
        BrickTimer {
            name: name.to_string(),
            start: Instant::now(),
        }
    }

    /// Stop timing and record the sample.
    ///
    /// # Arguments
    /// - `timer`: Timer handle from `start()`
    /// - `elements`: Number of elements (tokens) processed
    pub fn stop(&mut self, timer: BrickTimer, elements: u64) {
        if !self.enabled {
            return;
        }

        let elapsed = timer.start.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;

        // PAR-200: Try fast path first if name matches a known BrickId
        if let Some(brick_id) = BrickId::from_str(&timer.name) {
            let stats = &mut self.brick_stats[brick_id as usize];
            stats.add_sample(elapsed_ns, elements);
        } else {
            // Fall back to dynamic stats
            let name = timer.name;
            let stats = self
                .dynamic_stats
                .entry(name.clone())
                .or_insert_with(|| BrickStats::new(&name));
            stats.add_sample(elapsed_ns, elements);
        }

        // Update totals
        self.total_tokens += elements;
        self.total_ns += elapsed_ns;
    }

    /// Record a pre-measured duration for a brick.
    ///
    /// PAR-073: This method allows timing with raw `Instant` calls, avoiding
    /// borrow conflicts when profiling CUDA operations that also need `&mut self`.
    ///
    /// # Arguments
    /// - `name`: Brick name
    /// - `elapsed`: Duration of the operation (from `Instant::elapsed()`)
    /// - `elements`: Number of elements (tokens) processed
    ///
    /// # Example
    /// ```rust,ignore
    /// let start = std::time::Instant::now();
    /// cuda_stream.synchronize()?;
    /// self.some_cuda_operation()?;
    /// cuda_stream.synchronize()?;
    /// let elapsed = start.elapsed();
    /// self.profiler.record_elapsed("SomeBrick", elapsed, 1);
    /// ```
    pub fn record_elapsed(&mut self, name: &str, elapsed: std::time::Duration, elements: u64) {
        if !self.enabled {
            return;
        }

        let elapsed_ns = elapsed.as_nanos() as u64;

        // PAR-200: Try fast path first if name matches a known BrickId
        if let Some(brick_id) = BrickId::from_str(name) {
            let stats = &mut self.brick_stats[brick_id as usize];
            stats.add_sample(elapsed_ns, elements);
        } else {
            // Fall back to dynamic stats
            let stats = self
                .dynamic_stats
                .entry(name.to_string())
                .or_insert_with(|| BrickStats::new(name));
            stats.add_sample(elapsed_ns, elements);
        }

        // Update totals
        self.total_tokens += elements;
        self.total_ns += elapsed_ns;
    }

    /// PMAT-451: Record elapsed time with byte metrics for compression workloads.
    ///
    /// # Arguments
    /// - `name`: Brick name
    /// - `elapsed`: Duration of the operation
    /// - `elements`: Number of elements (pages) processed
    /// - `input_bytes`: Original uncompressed size
    /// - `output_bytes`: Compressed output size
    ///
    /// # Example
    /// ```rust,ignore
    /// let start = std::time::Instant::now();
    /// let compressed = zstd_compress(&page_data);
    /// let elapsed = start.elapsed();
    /// profiler.record_elapsed_with_bytes(
    ///     "ZstdCompress",
    ///     elapsed,
    ///     1,
    ///     page_data.len() as u64,
    ///     compressed.len() as u64,
    /// );
    /// ```
    pub fn record_elapsed_with_bytes(
        &mut self,
        name: &str,
        elapsed: std::time::Duration,
        elements: u64,
        input_bytes: u64,
        output_bytes: u64,
    ) {
        if !self.enabled {
            return;
        }

        let elapsed_ns = elapsed.as_nanos() as u64;

        // PAR-200: Try fast path first if name matches a known BrickId
        if let Some(brick_id) = BrickId::from_str(name) {
            let stats = &mut self.brick_stats[brick_id as usize];
            stats.add_sample_with_bytes(elapsed_ns, elements, input_bytes, output_bytes);
        } else {
            // Fall back to dynamic stats
            let stats = self
                .dynamic_stats
                .entry(name.to_string())
                .or_insert_with(|| BrickStats::new(name));
            stats.add_sample_with_bytes(elapsed_ns, elements, input_bytes, output_bytes);
        }

        // Update totals
        self.total_tokens += elements;
        self.total_ns += elapsed_ns;
    }

    /// PMAT-451: Set bottleneck classification for a brick.
    pub fn set_brick_bottleneck(&mut self, name: &str, bottleneck: BrickBottleneck) {
        // PAR-200: Try fast path first
        if let Some(brick_id) = BrickId::from_str(name) {
            self.brick_stats[brick_id as usize].set_bottleneck(bottleneck);
        } else if let Some(stats) = self.dynamic_stats.get_mut(name) {
            stats.set_bottleneck(bottleneck);
        }
    }

    /// Get statistics for a specific brick by name.
    ///
    /// First checks known BrickId types (O(1)), then falls back to dynamic stats.
    #[must_use]
    pub fn stats(&self, name: &str) -> Option<&BrickStats> {
        // Try fast path first
        if let Some(brick_id) = BrickId::from_str(name) {
            let stats = &self.brick_stats[brick_id as usize];
            if stats.count > 0 {
                return Some(stats);
            }
        }
        // Fall back to dynamic stats
        self.dynamic_stats.get(name)
    }

    /// Get all brick statistics (legacy API, returns dynamic stats only).
    ///
    /// For full statistics including known bricks, use `all_brick_stats()` instead.
    #[must_use]
    #[deprecated(
        since = "0.12.0",
        note = "Use all_brick_stats() for complete statistics"
    )]
    pub fn all_stats(&self) -> &std::collections::HashMap<String, BrickStats> {
        &self.dynamic_stats
    }

    /// Get all brick statistics including both known and dynamic bricks.
    pub fn all_brick_stats(&self) -> impl Iterator<Item = &BrickStats> {
        self.brick_stats
            .iter()
            .filter(|s| s.count > 0)
            .chain(self.dynamic_stats.values())
    }

    /// Get total throughput across all bricks.
    #[must_use]
    pub fn total_throughput(&self) -> f64 {
        if self.total_ns == 0 {
            0.0
        } else {
            self.total_tokens as f64 / (self.total_ns as f64 / 1_000_000_000.0)
        }
    }

    /// Get total tokens processed.
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Get total time in nanoseconds.
    #[must_use]
    pub fn total_ns(&self) -> u64 {
        self.total_ns
    }

    /// Get all brick names.
    #[must_use]
    pub fn brick_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .brick_stats
            .iter()
            .enumerate()
            .filter(|(_, s)| s.count > 0)
            .map(|(i, _)| {
                // Safety: i < BrickId::COUNT
                let brick_id = unsafe { std::mem::transmute::<u8, BrickId>(i as u8) };
                brick_id.name().to_string()
            })
            .collect();
        names.extend(self.dynamic_stats.keys().cloned());
        names
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        for stats in &mut self.brick_stats {
            stats.count = 0;
            stats.total_ns = 0;
            stats.min_ns = u64::MAX;
            stats.max_ns = 0;
            stats.total_elements = 0;
            stats.total_bytes = 0;
            stats.total_compressed_bytes = 0;
        }
        self.dynamic_stats.clear();
        self.pending.clear();
        self.total_tokens = 0;
        self.total_ns = 0;
    }

    /// Generate a summary report.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Brick Profiler Summary (PAR-200) ===\n");
        report.push_str(&format!(
            "Total: {} tokens, {:.2}µs, {:.1} tok/s\n",
            self.total_tokens,
            self.total_ns as f64 / 1000.0,
            self.total_throughput()
        ));
        report.push_str("\nPer-Brick Breakdown:\n");

        // Collect all stats (known + dynamic)
        let mut all_stats: Vec<(&str, &BrickStats)> = Vec::new();

        // Add known bricks with non-zero counts
        for (i, stats) in self.brick_stats.iter().enumerate() {
            if stats.count > 0 {
                // Safety: i < BrickId::COUNT
                let brick_id = unsafe { std::mem::transmute::<u8, BrickId>(i as u8) };
                all_stats.push((brick_id.name(), stats));
            }
        }

        // Add dynamic bricks
        for (name, stats) in &self.dynamic_stats {
            all_stats.push((name.as_str(), stats));
        }

        // Sort by total time descending
        all_stats.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));

        for (name, stats) in all_stats {
            let pct = if self.total_ns > 0 {
                100.0 * stats.total_ns as f64 / self.total_ns as f64
            } else {
                0.0
            };
            report.push_str(&format!(
                "  {:20} {:8.2}µs avg ({:5.1}%) [{} samples]\n",
                name,
                stats.avg_us(),
                pct,
                stats.count
            ));
        }

        // Add category breakdown
        report.push_str("\nCategory Breakdown:\n");
        let cats = self.category_stats();
        for (i, cat_stats) in cats.iter().enumerate() {
            if cat_stats.count > 0 {
                // Safety: i < BrickCategory::COUNT
                let cat = unsafe { std::mem::transmute::<u8, BrickCategory>(i as u8) };
                report.push_str(&format!(
                    "  {:12} {:8.2}µs avg ({:5.1}%)\n",
                    cat.name(),
                    cat_stats.avg_us(),
                    cat_stats.percentage(self.total_ns)
                ));
            }
        }

        report
    }

    /// Export profiling data as JSON for pmat metrics integration.
    ///
    /// Format compatible with `.pmat-metrics/trends/` structure:
    /// ```json
    /// {
    ///   "total_tokens": 1000,
    ///   "total_ns": 5000000,
    ///   "total_throughput": 200000.0,
    ///   "bricks": [
    ///     {
    ///       "name": "RmsNorm",
    ///       "count": 10,
    ///       "total_ns": 1000000,
    ///       "avg_us": 100.0,
    ///       "min_us": 90.0,
    ///       "max_us": 120.0,
    ///       "throughput": 10000.0,
    ///       "pct": 20.0
    ///     }
    ///   ]
    /// }
    /// ```
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut bricks = Vec::new();

        // Collect all stats (known + dynamic)
        let mut all_stats: Vec<(&str, &BrickStats)> = Vec::new();

        // Add known bricks with non-zero counts
        for (i, stats) in self.brick_stats.iter().enumerate() {
            if stats.count > 0 {
                // Safety: i < BrickId::COUNT
                let brick_id = unsafe { std::mem::transmute::<u8, BrickId>(i as u8) };
                all_stats.push((brick_id.name(), stats));
            }
        }

        // Add dynamic bricks
        for (name, stats) in &self.dynamic_stats {
            all_stats.push((name.as_str(), stats));
        }

        // Sort by total time descending
        all_stats.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));

        for (name, stats) in all_stats {
            let pct = if self.total_ns > 0 {
                100.0 * stats.total_ns as f64 / self.total_ns as f64
            } else {
                0.0
            };
            // PMAT-451: Include compression_ratio, throughput_gbps, and bottleneck
            let compression = stats.compression_ratio();
            let throughput_gbps = stats.throughput_gbps();
            let bottleneck = stats.get_bottleneck();
            bricks.push(format!(
                r#"{{"name":"{}","count":{},"total_ns":{},"avg_us":{:.2},"min_us":{:.2},"max_us":{:.2},"throughput":{:.1},"pct":{:.1},"total_bytes":{},"compression_ratio":{:.2},"throughput_gbps":{:.2},"bottleneck":"{}"}}"#,
                name,
                stats.count,
                stats.total_ns,
                stats.avg_us(),
                stats.min_us(),
                stats.max_us(),
                stats.throughput(),
                pct,
                stats.total_bytes,
                compression,
                throughput_gbps,
                bottleneck
            ));
        }

        format!(
            r#"{{"total_tokens":{},"total_ns":{},"total_throughput":{:.1},"bricks":[{}]}}"#,
            self.total_tokens,
            self.total_ns,
            self.total_throughput(),
            bricks.join(",")
        )
    }

    /// Write profiling data to a JSON file for pmat tracking.
    ///
    /// # Errors
    /// Returns error if file cannot be written.
    pub fn write_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    // =======================================================================
    // CORRECTNESS-011: Per-kernel checksum capture for divergence detection
    // =======================================================================

    /// Record a kernel trace with output checksum for divergence detection.
    ///
    /// This enables automated CPU/GPU divergence detection by capturing
    /// output checksums alongside timing data. When GPU produces wrong output,
    /// this identifies WHICH kernel diverged without hours of manual debugging.
    ///
    /// Five-Whys Root Cause: Hours of manual "let me check X in Y" debugging
    /// → No automated tool identified which kernel diverged
    /// → BrickProfiler only captured timing, not checksums
    /// → Missing feature: per-kernel checksum capture
    ///
    /// # Arguments
    /// - `name`: Brick/kernel name
    /// - `layer_idx`: Layer index (0-N for transformer layers)
    /// - `position`: Position in sequence
    /// - `output`: Output tensor data (first 64 floats checksummed)
    ///
    /// # Example
    /// ```rust,ignore
    /// // After RoPE kernel
    /// profiler.record_checksum("RopeNeox", layer_idx, position, &q_rotated);
    /// ```
    pub fn record_checksum(&mut self, name: &str, layer_idx: usize, position: u32, output: &[f32]) {
        if !self.enabled {
            return;
        }
        let checksum = fnv1a_f32_checksum(output);
        let trace = KernelChecksum {
            name: name.to_string(),
            layer_idx,
            position,
            checksum,
        };
        self.kernel_checksums.push(trace);
    }

    /// Get all kernel checksums for divergence comparison.
    #[must_use]
    pub fn get_checksums(&self) -> &[KernelChecksum] {
        &self.kernel_checksums
    }

    /// Compare checksums with a reference profiler (e.g., CPU baseline).
    ///
    /// Returns None if all checksums match, or the first divergent kernel.
    #[must_use]
    pub fn find_divergence(&self, reference: &BrickProfiler) -> Option<DivergenceInfo> {
        use std::collections::HashMap;

        // Index reference checksums by (name, layer_idx, position)
        let ref_index: HashMap<(&str, usize, u32), u64> = reference
            .kernel_checksums
            .iter()
            .map(|t| ((t.name.as_str(), t.layer_idx, t.position), t.checksum))
            .collect();

        // Check each of our checksums against reference
        for trace in &self.kernel_checksums {
            let key = (trace.name.as_str(), trace.layer_idx, trace.position);
            if let Some(&expected) = ref_index.get(&key) {
                if trace.checksum != expected {
                    return Some(DivergenceInfo {
                        kernel_name: trace.name.clone(),
                        layer_idx: trace.layer_idx,
                        position: trace.position,
                        expected_checksum: expected,
                        actual_checksum: trace.checksum,
                    });
                }
            }
        }
        None
    }

    /// Reset checksum tracking (call before new forward pass).
    pub fn reset_checksums(&mut self) {
        self.kernel_checksums.clear();
    }

    // ========================================================================
    // TILING-SPEC-001: Tile-Level Profiling (Phase 15)
    // ========================================================================

    /// Enable tile-level profiling.
    ///
    /// When enabled, `start_tile()`/`stop_tile()` record per-tile statistics
    /// for Macro/Midi/Micro tile hierarchy.
    pub fn enable_tile_profiling(&mut self) {
        self.tile_profiling_enabled = true;
    }

    /// Disable tile-level profiling.
    pub fn disable_tile_profiling(&mut self) {
        self.tile_profiling_enabled = false;
    }

    /// Check if tile profiling is enabled.
    #[must_use]
    pub fn is_tile_profiling_enabled(&self) -> bool {
        self.tile_profiling_enabled
    }

    /// Start timing a tile execution.
    ///
    /// Returns a `TileTimer` that should be passed to `stop_tile()` after
    /// the tile computation completes.
    ///
    /// # Arguments
    /// - `level`: Tile hierarchy level (Macro/Midi/Micro)
    /// - `row`: Row index within parent tile
    /// - `col`: Column index within parent tile
    ///
    /// # Example
    /// ```rust,ignore
    /// let timer = profiler.start_tile(TileLevel::Macro, 0, 0);
    /// // ... execute tile computation ...
    /// profiler.stop_tile(timer, 256 * 256, 2 * 256 * 256 * 256);
    /// ```
    #[must_use]
    pub fn start_tile(&self, level: TileLevel, row: u32, col: u32) -> TileTimer {
        TileTimer {
            level,
            _row: row,
            _col: col,
            start: Instant::now(),
        }
    }

    /// Stop timing and record tile statistics.
    ///
    /// # Arguments
    /// - `timer`: Timer handle from `start_tile()`
    /// - `elements`: Number of elements processed by this tile
    /// - `flops`: Number of floating-point operations performed
    pub fn stop_tile(&mut self, timer: TileTimer, elements: u64, flops: u64) {
        if !self.tile_profiling_enabled {
            return;
        }

        let elapsed_ns = timer.start.elapsed().as_nanos() as u64;
        let idx = timer.level as usize;
        self.tile_stats[idx].add_sample(elapsed_ns, elements, flops);
    }

    /// Get tile statistics for a given level.
    #[must_use]
    pub fn tile_stats(&self, level: TileLevel) -> &TileStats {
        &self.tile_stats[level as usize]
    }

    /// Get mutable tile statistics for a given level.
    pub fn tile_stats_mut(&mut self, level: TileLevel) -> &mut TileStats {
        &mut self.tile_stats[level as usize]
    }

    /// Get all tile statistics as a slice.
    #[must_use]
    pub fn all_tile_stats(&self) -> &[TileStats; 3] {
        &self.tile_stats
    }

    /// Reset tile statistics for all levels.
    pub fn reset_tile_stats(&mut self) {
        self.tile_stats = [
            TileStats::new(TileLevel::Macro),
            TileStats::new(TileLevel::Midi),
            TileStats::new(TileLevel::Micro),
        ];
    }

    /// Generate tile profiling summary report.
    ///
    /// # Example Output
    /// ```text
    /// === Tile Profiling Summary (TILING-SPEC-001) ===
    /// Level       Samples   Avg µs    GFLOP/s   AI      Elements
    /// Macro           128    1234.5     12.34  0.50    1048576
    /// Midi           2048      78.2     45.67  2.00      65536
    /// Micro         32768       4.9     89.12  4.00       4096
    /// ```
    #[must_use]
    pub fn tile_summary(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Tile Profiling Summary (TILING-SPEC-001) ===\n");
        report.push_str("Level       Samples   Avg µs    GFLOP/s   AI      Elements\n");

        for stats in &self.tile_stats {
            if stats.count > 0 {
                report.push_str(&format!(
                    "{:8}  {:9}  {:8.1}  {:8.2}  {:4.2}  {:10}\n",
                    stats.level.name(),
                    stats.count,
                    stats.avg_us(),
                    stats.gflops(),
                    stats.arithmetic_intensity(),
                    stats.total_elements / stats.count.max(1)
                ));
            }
        }

        report
    }

    /// Export tile statistics as JSON.
    ///
    /// Compatible with pmat metrics integration.
    #[must_use]
    pub fn tile_stats_to_json(&self) -> String {
        let tiles: Vec<String> = self
            .tile_stats
            .iter()
            .filter(|s| s.count > 0)
            .map(|s| {
                format!(
                    r#"{{"level":"{}","count":{},"total_ns":{},"avg_us":{:.2},"min_us":{:.2},"max_us":{:.2},"gflops":{:.2},"arithmetic_intensity":{:.2},"total_elements":{},"total_flops":{}}}"#,
                    s.level.name(),
                    s.count,
                    s.total_ns,
                    s.avg_us(),
                    s.min_ns as f64 / 1000.0,
                    s.max_ns as f64 / 1000.0,
                    s.gflops(),
                    s.arithmetic_intensity(),
                    s.total_elements,
                    s.total_flops
                )
            })
            .collect();

        format!(
            r#"{{"tile_profiling_enabled":{},"tiles":[{}]}}"#,
            self.tile_profiling_enabled,
            tiles.join(",")
        )
    }
}

/// Kernel checksum for divergence detection.
///
/// CORRECTNESS-011: Captures output checksum per kernel invocation.
#[derive(Debug, Clone)]
pub struct KernelChecksum {
    /// Kernel/brick name
    pub name: String,
    /// Layer index
    pub layer_idx: usize,
    /// Sequence position
    pub position: u32,
    /// FNV-1a checksum of first 64 output floats
    pub checksum: u64,
}

/// Information about a detected divergence between CPU and GPU.
#[derive(Debug, Clone)]
pub struct DivergenceInfo {
    /// Name of the divergent kernel
    pub kernel_name: String,
    /// Layer where divergence occurred
    pub layer_idx: usize,
    /// Position where divergence occurred
    pub position: u32,
    /// Expected checksum (from CPU/reference)
    pub expected_checksum: u64,
    /// Actual checksum (from GPU/test)
    pub actual_checksum: u64,
}

impl fmt::Display for DivergenceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DIVERGENCE at '{}' (layer {}, pos {}): expected 0x{:016X}, got 0x{:016X}",
            self.kernel_name,
            self.layer_idx,
            self.position,
            self.expected_checksum,
            self.actual_checksum
        )
    }
}

/// FNV-1a hash of f32 slice (first 64 elements for efficiency).
///
/// Used for quick divergence detection between CPU and GPU outputs.
#[inline]
pub fn fnv1a_f32_checksum(data: &[f32]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    let len = data.len().min(64);
    for &val in &data[..len] {
        let bytes = val.to_le_bytes();
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

/// Macro for convenient brick timing with automatic sync.
///
/// # Usage
///
/// ```rust,ignore
/// time_brick!(profiler, "RmsNorm", 1, {
///     rmsnorm_kernel.launch();
///     stream.synchronize(); // REQUIRED for GPU
/// });
/// ```
#[macro_export]
macro_rules! time_brick {
    ($profiler:expr, $name:expr, $elements:expr, $body:block) => {{
        let timer = $profiler.start($name);
        let result = $body;
        $profiler.stop(timer, $elements);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // TileStats Tests
    // ========================================================================

    #[test]
    fn test_tile_stats_new() {
        let stats = TileStats::new(TileLevel::Macro);
        assert_eq!(stats.level, TileLevel::Macro);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.total_ns, 0);
        assert_eq!(stats.min_ns, u64::MAX);
        assert_eq!(stats.max_ns, 0);
    }

    #[test]
    fn test_tile_stats_add_sample() {
        let mut stats = TileStats::new(TileLevel::Midi);
        stats.add_sample(1000, 100, 200);
        assert_eq!(stats.count, 1);
        assert_eq!(stats.total_ns, 1000);
        assert_eq!(stats.min_ns, 1000);
        assert_eq!(stats.max_ns, 1000);
        assert_eq!(stats.total_elements, 100);
        assert_eq!(stats.total_flops, 200);

        stats.add_sample(2000, 150, 300);
        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_ns, 3000);
        assert_eq!(stats.min_ns, 1000);
        assert_eq!(stats.max_ns, 2000);
    }

    #[test]
    fn test_tile_stats_avg_us() {
        let mut stats = TileStats::new(TileLevel::Micro);
        assert_eq!(stats.avg_us(), 0.0);

        stats.add_sample(2000_000, 100, 200); // 2000 µs
        assert!((stats.avg_us() - 2000.0).abs() < 0.01);

        stats.add_sample(4000_000, 100, 200); // 4000 µs
        assert!((stats.avg_us() - 3000.0).abs() < 0.01); // (2000 + 4000) / 2
    }

    #[test]
    fn test_tile_stats_throughput() {
        let mut stats = TileStats::new(TileLevel::Macro);
        assert_eq!(stats.throughput(), 0.0);

        // 1 billion ns = 1 second, 1000 elements = 1000 elem/s
        stats.add_sample(1_000_000_000, 1000, 0);
        assert!((stats.throughput() - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_tile_stats_gflops() {
        let mut stats = TileStats::new(TileLevel::Macro);
        assert_eq!(stats.gflops(), 0.0);

        // 1 second, 1 billion FLOPs = 1 GFLOP/s
        stats.add_sample(1_000_000_000, 100, 1_000_000_000);
        assert!((stats.gflops() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_tile_stats_arithmetic_intensity() {
        let mut stats = TileStats::new(TileLevel::Midi);
        assert_eq!(stats.arithmetic_intensity(), 0.0);

        // 1000 elements * 4 bytes = 4000 bytes, 4000 flops = 1.0 AI
        stats.add_sample(1000, 1000, 4000);
        assert!((stats.arithmetic_intensity() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_tile_level_name() {
        assert_eq!(TileLevel::Macro.name(), "macro");
        assert_eq!(TileLevel::Midi.name(), "midi");
        assert_eq!(TileLevel::Micro.name(), "micro");
    }

    // ========================================================================
    // BrickProfiler Tests
    // ========================================================================

    #[test]
    fn test_brick_profiler_new() {
        let profiler = BrickProfiler::new();
        assert!(!profiler.is_enabled());
        assert_eq!(profiler.total_tokens(), 0);
        assert_eq!(profiler.total_ns(), 0);
    }

    #[test]
    fn test_brick_profiler_enabled() {
        let profiler = BrickProfiler::enabled();
        assert!(profiler.is_enabled());
    }

    #[test]
    fn test_brick_profiler_enable_disable() {
        let mut profiler = BrickProfiler::new();
        assert!(!profiler.is_enabled());

        profiler.enable();
        assert!(profiler.is_enabled());

        profiler.disable();
        assert!(!profiler.is_enabled());
    }

    #[test]
    fn test_brick_profiler_sync_mode() {
        let mut profiler = BrickProfiler::new();
        assert_eq!(profiler.sync_mode(), SyncMode::Deferred);

        profiler.set_sync_mode(SyncMode::Immediate);
        assert_eq!(profiler.sync_mode(), SyncMode::Immediate);
    }

    #[test]
    fn test_brick_profiler_start_stop() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        let timer = profiler.start("TestBrick");
        std::thread::sleep(std::time::Duration::from_micros(100));
        profiler.stop(timer, 10);

        // Should have recorded something
        assert!(profiler.total_tokens() >= 10);
        assert!(profiler.total_ns() > 0);
    }

    #[test]
    fn test_brick_profiler_start_stop_disabled() {
        let mut profiler = BrickProfiler::new();
        // profiler is disabled by default

        let timer = profiler.start("TestBrick");
        profiler.stop(timer, 10);

        // Should NOT have recorded anything
        assert_eq!(profiler.total_tokens(), 0);
        assert_eq!(profiler.total_ns(), 0);
    }

    #[test]
    fn test_brick_profiler_brick_id_api() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        let timer = profiler.start_brick(BrickId::RmsNorm);
        std::thread::sleep(std::time::Duration::from_micros(50));
        profiler.stop_brick(timer, 5);

        let stats = profiler.brick_stats(BrickId::RmsNorm);
        assert_eq!(stats.count, 1);
        assert!(stats.total_ns > 0);
    }

    #[test]
    fn test_brick_profiler_deferred_api() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        let start_ns = profiler.elapsed_ns();
        std::thread::sleep(std::time::Duration::from_micros(100));
        profiler.record_deferred(BrickId::QkvProjection, start_ns, 100);

        assert!(profiler.has_pending());
        assert_eq!(profiler.pending_count(), 1);

        let end_ns = profiler.elapsed_ns();
        profiler.finalize(end_ns);

        assert!(!profiler.has_pending());
        assert_eq!(profiler.pending_count(), 0);

        let stats = profiler.brick_stats(BrickId::QkvProjection);
        assert_eq!(stats.count, 1);
    }

    #[test]
    fn test_brick_profiler_reset() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        let timer = profiler.start_brick(BrickId::AttentionSoftmax);
        profiler.stop_brick(timer, 10);

        assert!(profiler.total_tokens() > 0);

        profiler.reset();

        assert_eq!(profiler.total_tokens(), 0);
        assert_eq!(profiler.total_ns(), 0);
    }

    #[test]
    fn test_brick_profiler_tile_profiling() {
        let mut profiler = BrickProfiler::new();
        assert!(!profiler.is_tile_profiling_enabled());

        profiler.enable_tile_profiling();
        assert!(profiler.is_tile_profiling_enabled());

        let timer = profiler.start_tile(TileLevel::Macro, 0, 0);
        std::thread::sleep(std::time::Duration::from_micros(50));
        profiler.stop_tile(timer, 1024, 2048);

        let stats = profiler.tile_stats(TileLevel::Macro);
        assert_eq!(stats.count, 1);
        assert!(stats.total_ns > 0);
    }

    #[test]
    fn test_brick_profiler_tile_profiling_disabled() {
        let mut profiler = BrickProfiler::new();
        // Tile profiling disabled by default

        let timer = profiler.start_tile(TileLevel::Midi, 1, 1);
        profiler.stop_tile(timer, 512, 1024);

        let stats = profiler.tile_stats(TileLevel::Midi);
        assert_eq!(stats.count, 0);
    }

    // ========================================================================
    // Checksum Tests (CORRECTNESS-011)
    // ========================================================================

    #[test]
    fn test_fnv1a_checksum_empty() {
        let data: [f32; 0] = [];
        let checksum = fnv1a_f32_checksum(&data);
        // Should return offset basis for empty input
        assert_eq!(checksum, 0xcbf29ce484222325);
    }

    #[test]
    fn test_fnv1a_checksum_deterministic() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let c1 = fnv1a_f32_checksum(&data);
        let c2 = fnv1a_f32_checksum(&data);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_fnv1a_checksum_different_inputs() {
        let data1 = [1.0f32, 2.0, 3.0];
        let data2 = [1.0f32, 2.0, 4.0];
        let c1 = fnv1a_f32_checksum(&data1);
        let c2 = fnv1a_f32_checksum(&data2);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_fnv1a_checksum_truncates_at_64() {
        let data_short: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let data_long: Vec<f32> = (0..100).map(|i| i as f32).collect();

        let c1 = fnv1a_f32_checksum(&data_short);
        let c2 = fnv1a_f32_checksum(&data_long);
        // Both should hash only first 64 elements
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_brick_profiler_divergence_detection() {
        let mut cpu_profiler = BrickProfiler::new();
        let mut gpu_profiler = BrickProfiler::new();
        cpu_profiler.enable();
        gpu_profiler.enable();

        // Record same checksum on both
        let data = [1.0f32, 2.0, 3.0, 4.0];
        cpu_profiler.record_checksum("TestKernel", 0, 0, &data);
        gpu_profiler.record_checksum("TestKernel", 0, 0, &data);

        // No divergence
        assert!(gpu_profiler.find_divergence(&cpu_profiler).is_none());

        // Now record different checksum
        let different_data = [1.0f32, 2.0, 3.0, 5.0]; // Changed last element
        gpu_profiler.reset_checksums();
        gpu_profiler.record_checksum("TestKernel", 0, 0, &different_data);

        // Should find divergence
        let div = gpu_profiler.find_divergence(&cpu_profiler);
        assert!(div.is_some());
        let div = div.unwrap();
        assert_eq!(div.kernel_name, "TestKernel");
        assert_eq!(div.layer_idx, 0);
        assert_eq!(div.position, 0);
    }

    // ========================================================================
    // Falsification Tests
    // ========================================================================

    /// FALSIFICATION TEST: TileStats min/max monotonicity
    ///
    /// After any sequence of add_sample calls, min_ns <= max_ns must hold.
    #[test]
    fn test_falsify_tile_stats_min_max_monotonicity() {
        let mut stats = TileStats::new(TileLevel::Macro);

        // Add samples with varying elapsed times
        for ns in [1000, 500, 2000, 100, 5000, 50] {
            stats.add_sample(ns, 10, 20);
            assert!(
                stats.min_ns <= stats.max_ns,
                "FALSIFICATION FAILED: min {} > max {} after adding {}",
                stats.min_ns,
                stats.max_ns,
                ns
            );
        }

        assert_eq!(stats.min_ns, 50);
        assert_eq!(stats.max_ns, 5000);
    }

    /// FALSIFICATION TEST: BrickProfiler total_tokens accumulation
    ///
    /// total_tokens must equal sum of all elements passed to stop/stop_brick.
    #[test]
    fn test_falsify_total_tokens_accumulation() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        let mut expected_total = 0u64;
        let element_counts = [10, 20, 30, 15, 25];

        for &count in &element_counts {
            let timer = profiler.start_brick(BrickId::RmsNorm);
            profiler.stop_brick(timer, count);
            expected_total += count;
        }

        assert_eq!(
            profiler.total_tokens(),
            expected_total,
            "FALSIFICATION FAILED: total_tokens {} != expected {}",
            profiler.total_tokens(),
            expected_total
        );
    }

    /// FALSIFICATION TEST: Checksum collision resistance
    ///
    /// Different float patterns should produce different checksums.
    #[test]
    fn test_falsify_checksum_collision_resistance() {
        // Generate various patterns that might collide in weak hashes
        let patterns: Vec<Vec<f32>> = vec![
            vec![0.0; 10],
            vec![1.0; 10],
            vec![-1.0; 10],
            (0..10).map(|i| i as f32).collect(),
            (0..10).map(|i| -(i as f32)).collect(),
            vec![f32::MIN_POSITIVE; 10],
            vec![f32::MAX; 10],
        ];

        let checksums: Vec<u64> = patterns.iter().map(|p| fnv1a_f32_checksum(p)).collect();

        // Check all pairs for uniqueness
        for i in 0..checksums.len() {
            for j in (i + 1)..checksums.len() {
                assert_ne!(
                    checksums[i], checksums[j],
                    "FALSIFICATION FAILED: patterns {} and {} collide with checksum {:016X}",
                    i, j, checksums[i]
                );
            }
        }
    }

    // ========================================================================
    // Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_tile_stats_cache_efficiency() {
        let mut stats = TileStats::new(TileLevel::Macro);

        // Zero peak_gflops should return 0.0
        assert_eq!(stats.cache_efficiency(0.0), 0.0);
        assert_eq!(stats.cache_efficiency(-1.0), 0.0);

        // 1 second, 1 billion FLOPs = 1 GFLOP/s
        stats.add_sample(1_000_000_000, 100, 1_000_000_000);
        let efficiency = stats.cache_efficiency(10.0); // 10 GFLOP/s peak
        assert!((efficiency - 0.1).abs() < 0.01);

        // Efficiency capped at 1.0
        let capped = stats.cache_efficiency(0.5); // Lower peak than actual
        assert!((capped - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_brick_profiler_record_elapsed() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        let duration = std::time::Duration::from_micros(1000);
        profiler.record_elapsed("MyBrick", duration, 100);

        assert!(profiler.total_tokens() >= 100);
        assert!(profiler.total_ns() > 0);
    }

    #[test]
    fn test_brick_profiler_record_elapsed_disabled() {
        let mut profiler = BrickProfiler::new();
        // profiler is disabled

        let duration = std::time::Duration::from_micros(1000);
        profiler.record_elapsed("MyBrick", duration, 100);

        // Should not record when disabled
        assert_eq!(profiler.total_tokens(), 0);
    }

    #[test]
    fn test_brick_profiler_record_elapsed_with_bytes() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        let duration = std::time::Duration::from_micros(500);
        profiler.record_elapsed_with_bytes("ByteBrick", duration, 200, 4096, 2048);

        assert!(profiler.total_tokens() >= 200);
    }

    #[test]
    fn test_brick_profiler_set_brick_bottleneck() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        // Record something first
        let timer = profiler.start("TestBrick");
        profiler.stop(timer, 10);

        profiler.set_brick_bottleneck("TestBrick", BrickBottleneck::Memory);
        // Should not panic
    }

    #[test]
    fn test_brick_profiler_category_stats() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        // Record normalization brick
        let timer = profiler.start_brick(BrickId::RmsNorm);
        std::thread::sleep(std::time::Duration::from_micros(10));
        profiler.stop_brick(timer, 100);

        let category_stats = profiler.category_stats();
        // Should have accumulated in Norm category
        let norm_stats = &category_stats[BrickCategory::Norm as usize];
        assert!(norm_stats.count >= 1);
    }

    #[test]
    fn test_brick_profiler_graph_operations() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        // Graph disabled by default
        assert!(!profiler.is_graph_enabled());

        profiler.enable_graph();
        assert!(profiler.is_graph_enabled());

        // Push a scope using Layer variant
        let node = ExecutionNode::Layer { index: 0 };
        let scope_id = profiler.graph_push_scope(node);
        assert!(scope_id.is_some());

        // Record a brick
        profiler.graph_record_brick(BrickId::RmsNorm, 1000, 100);

        // Pop the scope
        let popped_id = profiler.graph_pop_scope();
        assert!(popped_id.is_some());

        assert!(profiler.graph_is_scope_balanced());

        profiler.disable_graph();
        assert!(!profiler.is_graph_enabled());
    }

    #[test]
    fn test_brick_profiler_graph_to_dot() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();
        profiler.enable_graph();

        let node = ExecutionNode::Layer { index: 1 };
        profiler.graph_push_scope(node);
        profiler.graph_record_brick(BrickId::RmsNorm, 500, 50);
        profiler.graph_pop_scope();

        let dot = profiler.graph_to_dot();
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_brick_profiler_graph_clear() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();
        profiler.enable_graph();

        let node = ExecutionNode::Layer { index: 2 };
        profiler.graph_push_scope(node);
        profiler.graph_pop_scope();

        profiler.graph_clear();
        // Should be balanced after clear
        assert!(profiler.graph_is_scope_balanced());
    }

    #[test]
    fn test_brick_profiler_l2_cache_hit_rate() {
        let mut profiler = BrickProfiler::new();

        // Default is None
        assert!(profiler.l2_cache_hit_rate().is_none());

        profiler.set_l2_cache_hit_rate(0.95);
        assert_eq!(profiler.l2_cache_hit_rate(), Some(0.95));
    }

    #[test]
    fn test_brick_profiler_zero_copy() {
        let mut profiler = BrickProfiler::new();

        // Default is false
        assert!(!profiler.is_zero_copy());

        profiler.set_zero_copy(true);
        assert!(profiler.is_zero_copy());

        profiler.set_zero_copy(false);
        assert!(!profiler.is_zero_copy());
    }

    #[test]
    fn test_brick_profiler_reset_epoch() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        // Let some time pass
        std::thread::sleep(std::time::Duration::from_micros(100));
        let ns1 = profiler.elapsed_ns();
        assert!(ns1 > 0);

        profiler.reset_epoch();
        let ns2 = profiler.elapsed_ns();
        // After reset, elapsed should be close to zero
        assert!(ns2 < ns1);
    }

    #[test]
    fn test_brick_profiler_brick_stats_mut() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        // Get mutable stats and modify
        let stats = profiler.brick_stats_mut(BrickId::RmsNorm);
        stats.count = 42;

        // Verify modification persisted
        let stats = profiler.brick_stats(BrickId::RmsNorm);
        assert_eq!(stats.count, 42);
    }

    #[test]
    fn test_brick_profiler_record_deferred_dynamic() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();

        let start_ns = profiler.elapsed_ns();
        std::thread::sleep(std::time::Duration::from_micros(50));
        profiler.record_deferred_dynamic("DynamicBrick", start_ns, 75);

        assert!(profiler.has_pending());
        assert_eq!(profiler.pending_count(), 1);

        let end_ns = profiler.elapsed_ns();
        profiler.finalize(end_ns);

        assert!(!profiler.has_pending());
    }

    #[test]
    fn test_tile_stats_default() {
        let stats = TileStats::default();
        assert_eq!(stats.level, TileLevel::Macro); // Default is Macro
        assert_eq!(stats.count, 0);
    }

    #[test]
    fn test_tile_level_default() {
        let level = TileLevel::default();
        assert_eq!(level, TileLevel::Macro);
    }

    #[test]
    fn test_brick_profiler_execution_graph_accessors() {
        let mut profiler = BrickProfiler::new();

        // Read-only access
        let _graph = profiler.execution_graph();

        // Mutable access
        let _graph_mut = profiler.execution_graph_mut();
    }

    #[test]
    fn test_brick_profiler_graph_record_kernel() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();
        profiler.enable_graph();

        let node = ExecutionNode::Layer { index: 3 };
        profiler.graph_push_scope(node);
        // graph_record_kernel(name, ptx_hash, grid, block, shared_mem)
        profiler.graph_record_kernel("my_kernel", 0x12345678, (1, 1, 1), (256, 1, 1), 4096);
        profiler.graph_pop_scope();

        // Should be able to convert to DOT
        let dot = profiler.graph_to_dot();
        assert!(!dot.is_empty());
    }

    #[test]
    fn test_brick_profiler_graph_disabled_operations() {
        let mut profiler = BrickProfiler::new();
        profiler.enable();
        // Graph is NOT enabled

        // Operations should be no-ops when graph disabled
        let node = ExecutionNode::Layer { index: 4 };
        let scope_id = profiler.graph_push_scope(node);
        assert!(scope_id.is_none()); // Returns None when disabled

        let popped = profiler.graph_pop_scope();
        assert!(popped.is_none());
    }
}
