//! PagedKvCache Implementation (PMAT-014)
//!
//! Implements PagedAttention-style KV cache management per cbtop spec §18.
//!
//! # Overview
//!
//! PagedAttention manages KV cache memory using fixed-size blocks, enabling:
//! - Dynamic memory allocation without fragmentation
//! - Copy-on-write for beam search
//! - Efficient eviction under memory pressure
//!
//! # Citations
//!
//! - [Kwon et al. 2023] "Efficient Memory Management for LLM Serving with PagedAttention" SOSP
//! - [Xiao et al. 2023] "StreamingLLM: Efficient Streaming with Attention Sinks" arXiv
//! - [Yu et al. 2022] "ORCA: A Distributed Serving System for Transformer-Based Models" OSDI

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

/// Block identifier (index into physical_blocks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "B{}", self.0)
    }
}

/// Sequence identifier (request ID).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SeqId(pub u64);

impl fmt::Display for SeqId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S{}", self.0)
    }
}

/// KV cache eviction strategies.
#[derive(Debug, Clone, PartialEq)]
pub enum EvictionStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Evict longest sequences first
    LongestFirst,
    /// Evict by priority (preempt low-priority requests)
    Priority { levels: usize },
    /// StreamingLLM: keep sink tokens + recent window
    StreamingLLM {
        sink_tokens: usize,
        window_tokens: usize,
    },
}

impl Default for EvictionStrategy {
    fn default() -> Self {
        EvictionStrategy::LRU
    }
}

impl fmt::Display for EvictionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvictionStrategy::LRU => write!(f, "LRU"),
            EvictionStrategy::LFU => write!(f, "LFU"),
            EvictionStrategy::LongestFirst => write!(f, "LongestFirst"),
            EvictionStrategy::Priority { levels } => write!(f, "Priority({})", levels),
            EvictionStrategy::StreamingLLM {
                sink_tokens,
                window_tokens,
            } => {
                write!(f, "StreamingLLM(sink={}, window={})", sink_tokens, window_tokens)
            }
        }
    }
}

/// Single KV cache block (logical representation).
///
/// In a real implementation, this would contain GPU buffers:
/// ```ignore
/// pub keys: DeviceBuffer<f16>,   // [block_size, num_heads, head_dim]
/// pub values: DeviceBuffer<f16>, // [block_size, num_heads, head_dim]
/// ```
#[derive(Debug)]
pub struct KvBlock {
    /// Block ID
    pub id: BlockId,
    /// Number of tokens stored in this block (0 to block_size)
    pub num_tokens: usize,
    /// Reference count (for copy-on-write)
    pub ref_count: AtomicU32,
    /// Block size (max tokens)
    pub capacity: usize,
}

impl KvBlock {
    /// Create a new empty block.
    pub fn new(id: BlockId, capacity: usize) -> Self {
        Self {
            id,
            num_tokens: 0,
            ref_count: AtomicU32::new(1),
            capacity,
        }
    }

    /// Check if block is full.
    pub fn is_full(&self) -> bool {
        self.num_tokens >= self.capacity
    }

    /// Remaining capacity.
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.num_tokens)
    }

    /// Get reference count.
    pub fn refs(&self) -> u32 {
        self.ref_count.load(Ordering::Acquire)
    }

    /// Increment reference count.
    pub fn inc_ref(&self) {
        self.ref_count.fetch_add(1, Ordering::AcqRel);
    }

    /// Decrement reference count, returns true if count reached zero.
    pub fn dec_ref(&self) -> bool {
        self.ref_count.fetch_sub(1, Ordering::AcqRel) == 1
    }
}

/// Sequence metadata for tracking access patterns.
#[derive(Debug, Clone)]
pub struct SequenceInfo {
    /// Sequence ID
    pub seq_id: SeqId,
    /// Total tokens in sequence
    pub num_tokens: usize,
    /// Blocks allocated to this sequence
    pub block_ids: Vec<BlockId>,
    /// Last access timestamp
    pub last_access: Instant,
    /// Access count (for LFU)
    pub access_count: u64,
    /// Priority level (for priority-based eviction)
    pub priority: u32,
}

impl SequenceInfo {
    /// Create new sequence info.
    pub fn new(seq_id: SeqId) -> Self {
        Self {
            seq_id,
            num_tokens: 0,
            block_ids: Vec::new(),
            last_access: Instant::now(),
            access_count: 0,
            priority: 0,
        }
    }

    /// Update access timestamp.
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
        self.access_count += 1;
    }

    /// Memory usage in blocks.
    pub fn num_blocks(&self) -> usize {
        self.block_ids.len()
    }
}

/// PagedKvCache error types.
#[derive(Debug, Clone)]
pub enum PagedKvError {
    /// Out of memory (no free blocks)
    OutOfMemory { requested: usize, available: usize },
    /// Sequence not found
    SequenceNotFound(SeqId),
    /// Block not found
    BlockNotFound(BlockId),
    /// Invalid operation
    InvalidOperation(String),
}

impl fmt::Display for PagedKvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PagedKvError::OutOfMemory {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Out of memory: requested {} blocks, {} available",
                    requested, available
                )
            }
            PagedKvError::SequenceNotFound(seq_id) => {
                write!(f, "Sequence not found: {}", seq_id)
            }
            PagedKvError::BlockNotFound(block_id) => {
                write!(f, "Block not found: {}", block_id)
            }
            PagedKvError::InvalidOperation(msg) => {
                write!(f, "Invalid operation: {}", msg)
            }
        }
    }
}

impl std::error::Error for PagedKvError {}

/// Result type for PagedKvCache operations.
pub type PagedKvResult<T> = Result<T, PagedKvError>;

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total allocations
    pub total_allocations: u64,
    /// Total frees
    pub total_frees: u64,
    /// Total evictions
    pub total_evictions: u64,
    /// Total forks (copy-on-write)
    pub total_forks: u64,
    /// Peak blocks used
    pub peak_blocks_used: usize,
}

/// Paged KV cache for efficient memory management.
///
/// Based on vLLM's PagedAttention algorithm. Manages KV cache memory
/// using fixed-size blocks to prevent fragmentation and enable
/// efficient memory sharing.
#[derive(Debug)]
pub struct PagedKvCache {
    /// Block size (tokens per block)
    block_size: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Total number of physical blocks
    num_blocks: usize,
    /// Free block indices
    free_blocks: VecDeque<BlockId>,
    /// Sequence → info mapping
    sequences: HashMap<SeqId, SequenceInfo>,
    /// Block reference counts (for COW)
    block_refs: HashMap<BlockId, u32>,
    /// Eviction strategy
    eviction_strategy: EvictionStrategy,
    /// Memory threshold for eviction (0.0-1.0)
    eviction_threshold: f64,
    /// Cache statistics
    stats: CacheStats,
}

impl PagedKvCache {
    /// Create a new PagedKvCache.
    ///
    /// # Arguments
    /// - `num_blocks`: Total number of physical blocks
    /// - `block_size`: Tokens per block
    /// - `num_heads`: Number of attention heads
    /// - `head_dim`: Dimension of each head
    pub fn new(num_blocks: usize, block_size: usize, num_heads: usize, head_dim: usize) -> Self {
        // Initialize free blocks
        let free_blocks: VecDeque<BlockId> =
            (0..num_blocks as u32).map(BlockId).collect();

        Self {
            block_size,
            num_heads,
            head_dim,
            num_blocks,
            free_blocks,
            sequences: HashMap::new(),
            block_refs: HashMap::new(),
            eviction_strategy: EvictionStrategy::default(),
            eviction_threshold: 0.9,
            stats: CacheStats::default(),
        }
    }

    /// Set eviction strategy.
    pub fn with_eviction_strategy(mut self, strategy: EvictionStrategy) -> Self {
        self.eviction_strategy = strategy;
        self
    }

    /// Set eviction threshold (0.0-1.0).
    pub fn with_eviction_threshold(mut self, threshold: f64) -> Self {
        self.eviction_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Get block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Get total number of blocks.
    pub fn total_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Get number of free blocks.
    pub fn free_block_count(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get number of used blocks.
    pub fn used_block_count(&self) -> usize {
        self.num_blocks - self.free_blocks.len()
    }

    /// Memory utilization percentage (0.0-1.0).
    pub fn utilization(&self) -> f64 {
        if self.num_blocks == 0 {
            return 0.0;
        }
        self.used_block_count() as f64 / self.num_blocks as f64
    }

    /// Calculate memory for a block in bytes.
    pub fn block_memory_bytes(&self) -> usize {
        // KV cache: 2 (K+V) * block_size * num_heads * head_dim * 2 (f16)
        2 * self.block_size * self.num_heads * self.head_dim * 2
    }

    /// Total memory capacity in bytes.
    pub fn total_memory_bytes(&self) -> usize {
        self.num_blocks * self.block_memory_bytes()
    }

    /// Used memory in bytes.
    pub fn used_memory_bytes(&self) -> usize {
        self.used_block_count() * self.block_memory_bytes()
    }

    /// Check if eviction is needed.
    pub fn needs_eviction(&self) -> bool {
        self.utilization() >= self.eviction_threshold
    }

    /// Get number of active sequences.
    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Get sequence info.
    pub fn get_sequence(&self, seq_id: SeqId) -> Option<&SequenceInfo> {
        self.sequences.get(&seq_id)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get eviction strategy.
    pub fn eviction_strategy(&self) -> &EvictionStrategy {
        &self.eviction_strategy
    }

    /// Calculate blocks needed for tokens.
    fn blocks_needed(&self, num_tokens: usize) -> usize {
        num_tokens.div_ceil(self.block_size)
    }

    /// Allocate a single block.
    fn allocate_block(&mut self) -> PagedKvResult<BlockId> {
        if let Some(block_id) = self.free_blocks.pop_front() {
            self.block_refs.insert(block_id, 1);
            self.stats.total_allocations += 1;

            // Track peak usage
            let used = self.used_block_count();
            if used > self.stats.peak_blocks_used {
                self.stats.peak_blocks_used = used;
            }

            Ok(block_id)
        } else {
            Err(PagedKvError::OutOfMemory {
                requested: 1,
                available: 0,
            })
        }
    }

    /// Free a single block.
    fn free_block(&mut self, block_id: BlockId) -> PagedKvResult<()> {
        if let Some(refs) = self.block_refs.get_mut(&block_id) {
            *refs -= 1;
            if *refs == 0 {
                self.block_refs.remove(&block_id);
                self.free_blocks.push_back(block_id);
                self.stats.total_frees += 1;
            }
            Ok(())
        } else {
            Err(PagedKvError::BlockNotFound(block_id))
        }
    }

    /// Allocate blocks for a new sequence.
    pub fn allocate(&mut self, seq_id: SeqId, num_tokens: usize) -> PagedKvResult<()> {
        if self.sequences.contains_key(&seq_id) {
            return Err(PagedKvError::InvalidOperation(format!(
                "Sequence {} already exists",
                seq_id
            )));
        }

        let blocks_needed = self.blocks_needed(num_tokens);

        // Check if we have enough blocks
        if blocks_needed > self.free_blocks.len() {
            return Err(PagedKvError::OutOfMemory {
                requested: blocks_needed,
                available: self.free_blocks.len(),
            });
        }

        // Allocate blocks
        let mut block_ids = Vec::with_capacity(blocks_needed);
        for _ in 0..blocks_needed {
            block_ids.push(self.allocate_block()?);
        }

        // Create sequence info
        let mut seq_info = SequenceInfo::new(seq_id);
        seq_info.num_tokens = num_tokens;
        seq_info.block_ids = block_ids;
        seq_info.touch();

        self.sequences.insert(seq_id, seq_info);
        Ok(())
    }

    /// Append tokens to an existing sequence.
    pub fn append(&mut self, seq_id: SeqId, num_new_tokens: usize) -> PagedKvResult<()> {
        // First, calculate how many blocks we need (immutably)
        let (old_tokens, additional_blocks) = {
            let seq_info = self
                .sequences
                .get(&seq_id)
                .ok_or(PagedKvError::SequenceNotFound(seq_id))?;

            let old_tokens = seq_info.num_tokens;
            let new_tokens = old_tokens + num_new_tokens;
            let old_blocks = self.blocks_needed(old_tokens);
            let new_blocks = self.blocks_needed(new_tokens);
            let additional = new_blocks.saturating_sub(old_blocks);

            (old_tokens, additional)
        };

        // Check if we have enough blocks
        if additional_blocks > self.free_blocks.len() {
            return Err(PagedKvError::OutOfMemory {
                requested: additional_blocks,
                available: self.free_blocks.len(),
            });
        }

        // Allocate the blocks
        let mut new_block_ids = Vec::with_capacity(additional_blocks);
        for _ in 0..additional_blocks {
            new_block_ids.push(self.allocate_block()?);
        }

        // Update sequence info
        let seq_info = self
            .sequences
            .get_mut(&seq_id)
            .ok_or(PagedKvError::SequenceNotFound(seq_id))?;

        seq_info.block_ids.extend(new_block_ids);
        seq_info.num_tokens = old_tokens + num_new_tokens;
        seq_info.touch();
        Ok(())
    }

    /// Free all blocks for a sequence.
    pub fn free(&mut self, seq_id: SeqId) -> PagedKvResult<()> {
        let seq_info = self
            .sequences
            .remove(&seq_id)
            .ok_or(PagedKvError::SequenceNotFound(seq_id))?;

        for block_id in seq_info.block_ids {
            self.free_block(block_id)?;
        }

        Ok(())
    }

    /// Copy-on-write fork for beam search.
    ///
    /// Creates a new sequence that shares blocks with the source sequence.
    /// Blocks are only copied when modified (copy-on-write).
    pub fn fork(&mut self, src_seq: SeqId, dst_seq: SeqId) -> PagedKvResult<()> {
        if self.sequences.contains_key(&dst_seq) {
            return Err(PagedKvError::InvalidOperation(format!(
                "Destination sequence {} already exists",
                dst_seq
            )));
        }

        let src_info = self
            .sequences
            .get(&src_seq)
            .ok_or(PagedKvError::SequenceNotFound(src_seq))?
            .clone();

        // Increment reference counts for shared blocks
        for block_id in &src_info.block_ids {
            if let Some(refs) = self.block_refs.get_mut(block_id) {
                *refs += 1;
            }
        }

        // Create new sequence with shared blocks
        let mut dst_info = SequenceInfo::new(dst_seq);
        dst_info.num_tokens = src_info.num_tokens;
        dst_info.block_ids = src_info.block_ids.clone();
        dst_info.touch();

        self.sequences.insert(dst_seq, dst_info);
        self.stats.total_forks += 1;
        Ok(())
    }

    /// Select sequence to evict based on strategy.
    pub fn select_eviction_target(&self) -> Option<SeqId> {
        if self.sequences.is_empty() {
            return None;
        }

        match &self.eviction_strategy {
            EvictionStrategy::LRU => {
                // Evict least recently used
                self.sequences
                    .values()
                    .min_by_key(|s| s.last_access)
                    .map(|s| s.seq_id)
            }
            EvictionStrategy::LFU => {
                // Evict least frequently used
                self.sequences
                    .values()
                    .min_by_key(|s| s.access_count)
                    .map(|s| s.seq_id)
            }
            EvictionStrategy::LongestFirst => {
                // Evict longest sequence (most blocks)
                self.sequences
                    .values()
                    .max_by_key(|s| s.num_tokens)
                    .map(|s| s.seq_id)
            }
            EvictionStrategy::Priority { .. } => {
                // Evict lowest priority
                self.sequences
                    .values()
                    .min_by_key(|s| s.priority)
                    .map(|s| s.seq_id)
            }
            EvictionStrategy::StreamingLLM { .. } => {
                // StreamingLLM doesn't evict sequences, it evicts tokens
                // For simplicity, fall back to LRU for sequence eviction
                self.sequences
                    .values()
                    .min_by_key(|s| s.last_access)
                    .map(|s| s.seq_id)
            }
        }
    }

    /// Evict a sequence to free memory.
    pub fn evict(&mut self) -> PagedKvResult<SeqId> {
        let target = self
            .select_eviction_target()
            .ok_or(PagedKvError::InvalidOperation("No sequences to evict".to_string()))?;

        self.free(target)?;
        self.stats.total_evictions += 1;
        Ok(target)
    }

    /// Evict until memory utilization is below threshold.
    pub fn evict_to_threshold(&mut self, target_util: f64) -> PagedKvResult<Vec<SeqId>> {
        let mut evicted = Vec::new();
        while self.utilization() > target_util && !self.sequences.is_empty() {
            evicted.push(self.evict()?);
        }
        Ok(evicted)
    }

    /// Apply StreamingLLM eviction to a sequence.
    ///
    /// Keeps sink tokens at the beginning and a recent window at the end,
    /// evicting middle tokens.
    pub fn apply_streaming_llm(
        &mut self,
        seq_id: SeqId,
        sink_tokens: usize,
        window_tokens: usize,
    ) -> PagedKvResult<usize> {
        // Get sequence info immutably first to compute values
        let (num_tokens, blocks_to_remove) = {
            let seq_info = self
                .sequences
                .get(&seq_id)
                .ok_or(PagedKvError::SequenceNotFound(seq_id))?;

            let keep_tokens = sink_tokens + window_tokens;
            if seq_info.num_tokens <= keep_tokens {
                return Ok(0); // Nothing to evict
            }

            let old_blocks = self.blocks_needed(seq_info.num_tokens);
            let new_blocks = self.blocks_needed(keep_tokens);
            let blocks_to_free = old_blocks.saturating_sub(new_blocks);

            // Collect blocks to remove
            let blocks: Vec<BlockId> = seq_info
                .block_ids
                .iter()
                .skip(sink_tokens / self.block_size + 1)
                .take(blocks_to_free)
                .cloned()
                .collect();

            (seq_info.num_tokens, blocks)
        };

        let keep_tokens = sink_tokens + window_tokens;
        let evict_tokens = num_tokens - keep_tokens;

        // Free the blocks
        for block_id in &blocks_to_remove {
            self.free_block(*block_id)?;
        }

        // Update sequence info
        if let Some(seq_info) = self.sequences.get_mut(&seq_id) {
            for block_id in blocks_to_remove {
                seq_info.block_ids.retain(|&id| id != block_id);
            }
            seq_info.num_tokens = keep_tokens;
        }

        Ok(evict_tokens)
    }

    /// Get all sequence IDs.
    pub fn sequence_ids(&self) -> Vec<SeqId> {
        self.sequences.keys().cloned().collect()
    }
}

impl fmt::Display for PagedKvCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "PagedKvCache")?;
        writeln!(f, "  Strategy: {} (block_size={})", self.eviction_strategy, self.block_size)?;
        writeln!(
            f,
            "  Blocks: {}/{} ({:.1}% used)",
            self.used_block_count(),
            self.num_blocks,
            self.utilization() * 100.0
        )?;
        writeln!(
            f,
            "  Memory: {:.2} MB / {:.2} MB",
            self.used_memory_bytes() as f64 / 1_000_000.0,
            self.total_memory_bytes() as f64 / 1_000_000.0
        )?;
        writeln!(f, "  Sequences: {} active", self.num_sequences())?;
        writeln!(
            f,
            "  Stats: allocs={}, frees={}, evictions={}, forks={}",
            self.stats.total_allocations,
            self.stats.total_frees,
            self.stats.total_evictions,
            self.stats.total_forks
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_cache() {
        let cache = PagedKvCache::new(1024, 16, 32, 128);
        assert_eq!(cache.total_blocks(), 1024);
        assert_eq!(cache.block_size(), 16);
        assert_eq!(cache.free_block_count(), 1024);
        assert_eq!(cache.used_block_count(), 0);
        assert!((cache.utilization() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_allocate_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 32, 128);

        let seq_id = SeqId(1);
        cache.allocate(seq_id, 48).unwrap(); // 3 blocks needed (48/16)

        assert_eq!(cache.used_block_count(), 3);
        assert_eq!(cache.num_sequences(), 1);

        let seq = cache.get_sequence(seq_id).unwrap();
        assert_eq!(seq.num_tokens, 48);
        assert_eq!(seq.num_blocks(), 3);
    }

    #[test]
    fn test_append_tokens() {
        let mut cache = PagedKvCache::new(100, 16, 32, 128);

        let seq_id = SeqId(1);
        cache.allocate(seq_id, 8).unwrap(); // 1 block (8 tokens, block_size=16)
        assert_eq!(cache.used_block_count(), 1);

        cache.append(seq_id, 4).unwrap(); // Still fits in 1 block (12 tokens)
        assert_eq!(cache.used_block_count(), 1);

        cache.append(seq_id, 8).unwrap(); // Needs 2nd block (20 tokens > 16)
        assert_eq!(cache.used_block_count(), 2);
    }

    #[test]
    fn test_free_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 32, 128);

        let seq_id = SeqId(1);
        cache.allocate(seq_id, 48).unwrap();
        assert_eq!(cache.used_block_count(), 3);

        cache.free(seq_id).unwrap();
        assert_eq!(cache.used_block_count(), 0);
        assert_eq!(cache.num_sequences(), 0);
    }

    #[test]
    fn test_fork_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 32, 128);

        let src = SeqId(1);
        let dst = SeqId(2);

        cache.allocate(src, 48).unwrap();
        cache.fork(src, dst).unwrap();

        // Both sequences share blocks
        assert_eq!(cache.num_sequences(), 2);
        // No new blocks allocated (COW)
        assert_eq!(cache.used_block_count(), 3);
        assert_eq!(cache.stats().total_forks, 1);
    }

    #[test]
    fn test_out_of_memory() {
        let mut cache = PagedKvCache::new(10, 16, 32, 128);

        let result = cache.allocate(SeqId(1), 200); // Needs 13 blocks, only 10 available
        assert!(matches!(result, Err(PagedKvError::OutOfMemory { .. })));
    }

    #[test]
    fn test_sequence_not_found() {
        let mut cache = PagedKvCache::new(100, 16, 32, 128);

        let result = cache.free(SeqId(999));
        assert!(matches!(result, Err(PagedKvError::SequenceNotFound(_))));
    }

    #[test]
    fn test_utilization() {
        let mut cache = PagedKvCache::new(100, 16, 32, 128);

        cache.allocate(SeqId(1), 160).unwrap(); // 10 blocks
        let util = cache.utilization();
        assert!((util - 0.1).abs() < 0.001); // 10/100 = 10%

        cache.allocate(SeqId(2), 480).unwrap(); // 30 blocks
        let util = cache.utilization();
        assert!((util - 0.4).abs() < 0.001); // 40/100 = 40%
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = PagedKvCache::new(100, 16, 32, 128)
            .with_eviction_strategy(EvictionStrategy::LRU);

        cache.allocate(SeqId(1), 16).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.allocate(SeqId(2), 16).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        cache.allocate(SeqId(3), 16).unwrap();

        // Seq 1 should be evicted (oldest)
        let target = cache.select_eviction_target();
        assert_eq!(target, Some(SeqId(1)));
    }

    #[test]
    fn test_longest_first_eviction() {
        let mut cache = PagedKvCache::new(100, 16, 32, 128)
            .with_eviction_strategy(EvictionStrategy::LongestFirst);

        cache.allocate(SeqId(1), 16).unwrap();
        cache.allocate(SeqId(2), 64).unwrap(); // Longest
        cache.allocate(SeqId(3), 32).unwrap();

        // Seq 2 should be evicted (longest)
        let target = cache.select_eviction_target();
        assert_eq!(target, Some(SeqId(2)));
    }

    #[test]
    fn test_evict_to_threshold() {
        let mut cache = PagedKvCache::new(10, 16, 32, 128);

        // Fill to 80%
        for i in 0..8 {
            cache.allocate(SeqId(i), 16).unwrap();
        }
        assert!((cache.utilization() - 0.8).abs() < 0.01);

        // Evict to 50%
        let evicted = cache.evict_to_threshold(0.5).unwrap();
        assert!(evicted.len() >= 3); // Need to evict at least 3 to go from 8 to 5
        assert!(cache.utilization() <= 0.5);
    }

    #[test]
    fn test_memory_calculation() {
        // block_size=16, num_heads=32, head_dim=128
        // block_memory = 2 * 16 * 32 * 128 * 2 = 262144 bytes = 256KB
        let cache = PagedKvCache::new(100, 16, 32, 128);
        assert_eq!(cache.block_memory_bytes(), 262144);
        assert_eq!(cache.total_memory_bytes(), 26214400); // 25MB
    }

    #[test]
    fn test_stats_tracking() {
        let mut cache = PagedKvCache::new(100, 16, 32, 128);

        cache.allocate(SeqId(1), 32).unwrap();
        cache.allocate(SeqId(2), 16).unwrap();
        cache.fork(SeqId(1), SeqId(3)).unwrap();
        cache.free(SeqId(2)).unwrap();

        let stats = cache.stats();
        assert_eq!(stats.total_allocations, 3);
        assert_eq!(stats.total_frees, 1);
        assert_eq!(stats.total_forks, 1);
    }

    #[test]
    fn test_eviction_strategy_display() {
        assert_eq!(format!("{}", EvictionStrategy::LRU), "LRU");
        assert_eq!(format!("{}", EvictionStrategy::LFU), "LFU");
        assert_eq!(
            format!(
                "{}",
                EvictionStrategy::StreamingLLM {
                    sink_tokens: 4,
                    window_tokens: 512
                }
            ),
            "StreamingLLM(sink=4, window=512)"
        );
    }

    #[test]
    fn test_duplicate_sequence() {
        let mut cache = PagedKvCache::new(100, 16, 32, 128);

        cache.allocate(SeqId(1), 16).unwrap();
        let result = cache.allocate(SeqId(1), 16);
        assert!(matches!(result, Err(PagedKvError::InvalidOperation(_))));
    }

    #[test]
    fn test_blocks_needed_calculation() {
        let cache = PagedKvCache::new(100, 16, 32, 128);

        // blocks_needed is private, test through allocate behavior
        let mut cache2 = cache;
        cache2.allocate(SeqId(1), 1).unwrap();
        assert_eq!(cache2.used_block_count(), 1);

        let mut cache3 = PagedKvCache::new(100, 16, 32, 128);
        cache3.allocate(SeqId(1), 16).unwrap();
        assert_eq!(cache3.used_block_count(), 1);

        let mut cache4 = PagedKvCache::new(100, 16, 32, 128);
        cache4.allocate(SeqId(1), 17).unwrap();
        assert_eq!(cache4.used_block_count(), 2);
    }
}
