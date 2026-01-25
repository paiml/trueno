//! BrickProfiler: Token-Centric Profiling System
//!
//! TILING-SPEC-001: Tile-Level Profiling Support
//!
//! This module provides hierarchical profiling for compute bricks:
//! - Per-brick timing and throughput (PAR-073)
//! - O(1) hot path with BrickId enum (PAR-200)
//! - Tile-level profiling for cache-blocked operations (TILING-SPEC-001)
//! - Kernel checksum capture for divergence detection (CORRECTNESS-011)

mod tile_stats;
mod checksum;

#[cfg(test)]
mod tests;

pub use tile_stats::{TileStats, TileLevel, TileTimer};
pub use checksum::{KernelChecksum, DivergenceInfo, fnv1a_f32_checksum};

use std::time::Instant;

use super::exec_graph::{
    BrickBottleneck, BrickCategory, BrickId, BrickStats, CategoryStats, ExecutionGraph,
    ExecutionNode, ExecutionNodeId, SyncMode,
};

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
