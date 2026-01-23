//! KV Cache Management and Batch Ordering
//!
//! LCP-10: KV Cache Slot tracking for transformer inference.
//! LCP-14: Sequential Batch Ordering for cache-friendly processing.

// ----------------------------------------------------------------------------
// LCP-10: KV Cache Slot Info
// ----------------------------------------------------------------------------

/// Metadata for a KV cache slot in transformer inference.
///
/// Tracks position, token info, and usage for cache management.
#[derive(Debug, Clone, Default)]
pub struct KvCacheSlotInfo {
    /// Sequence position this slot represents
    pub position: u32,
    /// Token ID stored in this slot
    pub token_id: u32,
    /// Layer index
    pub layer: u16,
    /// Head index
    pub head: u16,
    /// Whether this slot is valid/filled
    pub valid: bool,
    /// Last access time (in steps)
    pub last_access: u64,
}

impl KvCacheSlotInfo {
    /// Create a new slot info.
    pub fn new(position: u32, token_id: u32, layer: u16, head: u16) -> Self {
        Self {
            position,
            token_id,
            layer,
            head,
            valid: true,
            last_access: 0,
        }
    }

    /// Mark slot as accessed.
    pub fn touch(&mut self, step: u64) {
        self.last_access = step;
    }

    /// Invalidate the slot.
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Check if slot can be evicted (LRU policy).
    #[must_use]
    pub fn eviction_priority(&self, current_step: u64) -> u64 {
        if !self.valid {
            return u64::MAX; // Invalid slots have highest eviction priority
        }
        current_step.saturating_sub(self.last_access)
    }
}

/// KV cache manager with slot tracking.
#[derive(Debug)]
pub struct KvCacheManager {
    /// Slot metadata
    slots: Vec<KvCacheSlotInfo>,
    /// Current step counter
    current_step: u64,
    /// Number of valid slots
    valid_count: usize,
}

impl KvCacheManager {
    /// Create manager with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: vec![KvCacheSlotInfo::default(); capacity],
            current_step: 0,
            valid_count: 0,
        }
    }

    /// Allocate a slot.
    pub fn allocate(
        &mut self,
        position: u32,
        token_id: u32,
        layer: u16,
        head: u16,
    ) -> Option<usize> {
        // Find first invalid slot
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if !slot.valid {
                *slot = KvCacheSlotInfo::new(position, token_id, layer, head);
                slot.touch(self.current_step);
                self.valid_count += 1;
                return Some(i);
            }
        }
        None // No free slots
    }

    /// Access a slot.
    pub fn access(&mut self, index: usize) -> Option<&KvCacheSlotInfo> {
        if index < self.slots.len() {
            self.slots[index].touch(self.current_step);
            Some(&self.slots[index])
        } else {
            None
        }
    }

    /// Evict LRU slot.
    pub fn evict_lru(&mut self) -> Option<usize> {
        let mut best_idx = None;
        let mut best_priority = 0u64;

        for (i, slot) in self.slots.iter().enumerate() {
            if slot.valid {
                let priority = slot.eviction_priority(self.current_step);
                // Use >= for first found slot (best_idx.is_none()), then > for ties
                if best_idx.is_none() || priority > best_priority {
                    best_priority = priority;
                    best_idx = Some(i);
                }
            }
        }

        if let Some(idx) = best_idx {
            self.slots[idx].invalidate();
            self.valid_count -= 1;
        }
        best_idx
    }

    /// Advance step counter.
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Get number of valid slots.
    #[must_use]
    pub fn valid_count(&self) -> usize {
        self.valid_count
    }

    /// Get capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
}

// ----------------------------------------------------------------------------
// LCP-14: Sequential Batch Ordering
// ----------------------------------------------------------------------------

/// Sequential batch orderer for cache-friendly processing.
///
/// Ensures batches are processed in optimal order for memory access patterns.
#[derive(Debug, Clone)]
pub struct SequentialBatchOrderer {
    /// Batch indices in processing order
    order: Vec<usize>,
    /// Current position in order
    position: usize,
}

impl SequentialBatchOrderer {
    /// Create orderer for n batches.
    pub fn new(n_batches: usize) -> Self {
        Self {
            order: (0..n_batches).collect(),
            position: 0,
        }
    }

    /// Create orderer with reverse order (sometimes better for certain patterns).
    pub fn reversed(n_batches: usize) -> Self {
        Self {
            order: (0..n_batches).rev().collect(),
            position: 0,
        }
    }

    /// Create orderer with interleaved order (for better cache utilization).
    pub fn interleaved(n_batches: usize) -> Self {
        let mut order = Vec::with_capacity(n_batches);
        let mid = n_batches / 2;

        // Interleave: 0, mid, 1, mid+1, 2, mid+2, ...
        for i in 0..mid {
            order.push(i);
            if mid + i < n_batches {
                order.push(mid + i);
            }
        }
        // Handle odd number of batches
        if !n_batches.is_multiple_of(2) {
            order.push(n_batches - 1);
        }

        Self { order, position: 0 }
    }

    /// Get next batch index.
    pub fn next_batch(&mut self) -> Option<usize> {
        if self.position < self.order.len() {
            let idx = self.order[self.position];
            self.position += 1;
            Some(idx)
        } else {
            None
        }
    }

    /// Reset to beginning.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Check if all batches have been processed.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.position >= self.order.len()
    }

    /// Get remaining count.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.order.len().saturating_sub(self.position)
    }
}

impl Iterator for SequentialBatchOrderer {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // KvCacheSlotInfo Tests
    // =========================================================================

    #[test]
    fn test_kv_cache_slot_info_new() {
        let slot = KvCacheSlotInfo::new(5, 42, 2, 8);

        assert_eq!(slot.position, 5);
        assert_eq!(slot.token_id, 42);
        assert_eq!(slot.layer, 2);
        assert_eq!(slot.head, 8);
        assert!(slot.valid);
        assert_eq!(slot.last_access, 0);
    }

    #[test]
    fn test_kv_cache_slot_info_default() {
        let slot = KvCacheSlotInfo::default();

        assert_eq!(slot.position, 0);
        assert_eq!(slot.token_id, 0);
        assert_eq!(slot.layer, 0);
        assert_eq!(slot.head, 0);
        assert!(!slot.valid);
        assert_eq!(slot.last_access, 0);
    }

    #[test]
    fn test_kv_cache_slot_info_touch() {
        let mut slot = KvCacheSlotInfo::new(0, 0, 0, 0);

        slot.touch(10);
        assert_eq!(slot.last_access, 10);

        slot.touch(50);
        assert_eq!(slot.last_access, 50);
    }

    #[test]
    fn test_kv_cache_slot_info_invalidate() {
        let mut slot = KvCacheSlotInfo::new(0, 0, 0, 0);
        assert!(slot.valid);

        slot.invalidate();
        assert!(!slot.valid);
    }

    #[test]
    fn test_kv_cache_slot_info_eviction_priority_valid() {
        let mut slot = KvCacheSlotInfo::new(0, 0, 0, 0);
        slot.touch(10);

        // Same step as last access = priority 0
        assert_eq!(slot.eviction_priority(10), 0);

        // Steps since last access
        assert_eq!(slot.eviction_priority(20), 10);
        assert_eq!(slot.eviction_priority(100), 90);
    }

    #[test]
    fn test_kv_cache_slot_info_eviction_priority_invalid() {
        let mut slot = KvCacheSlotInfo::new(0, 0, 0, 0);
        slot.invalidate();

        // Invalid slots have MAX priority (should be evicted first)
        assert_eq!(slot.eviction_priority(0), u64::MAX);
        assert_eq!(slot.eviction_priority(100), u64::MAX);
    }

    // =========================================================================
    // KvCacheManager Tests
    // =========================================================================

    #[test]
    fn test_kv_cache_manager_new() {
        let mgr = KvCacheManager::new(10);

        assert_eq!(mgr.capacity(), 10);
        assert_eq!(mgr.valid_count(), 0);
    }

    #[test]
    fn test_kv_cache_manager_allocate() {
        let mut mgr = KvCacheManager::new(3);

        let idx0 = mgr.allocate(0, 100, 0, 0);
        assert_eq!(idx0, Some(0));
        assert_eq!(mgr.valid_count(), 1);

        let idx1 = mgr.allocate(1, 101, 0, 0);
        assert_eq!(idx1, Some(1));
        assert_eq!(mgr.valid_count(), 2);

        let idx2 = mgr.allocate(2, 102, 0, 0);
        assert_eq!(idx2, Some(2));
        assert_eq!(mgr.valid_count(), 3);

        // Full - no more slots
        let idx3 = mgr.allocate(3, 103, 0, 0);
        assert_eq!(idx3, None);
        assert_eq!(mgr.valid_count(), 3);
    }

    #[test]
    fn test_kv_cache_manager_access() {
        let mut mgr = KvCacheManager::new(3);

        let idx = mgr.allocate(0, 42, 1, 2).unwrap();

        let slot = mgr.access(idx).unwrap();
        assert_eq!(slot.token_id, 42);
        assert_eq!(slot.layer, 1);
        assert_eq!(slot.head, 2);

        // Out of bounds access
        assert!(mgr.access(100).is_none());
    }

    #[test]
    fn test_kv_cache_manager_step() {
        let mut mgr = KvCacheManager::new(2);
        let idx = mgr.allocate(0, 0, 0, 0).unwrap();

        // Step advances counter
        mgr.step();
        mgr.step();

        // Access updates last_access to current step
        let slot = mgr.access(idx).unwrap();
        assert_eq!(slot.last_access, 2);
    }

    #[test]
    fn test_kv_cache_manager_evict_lru() {
        let mut mgr = KvCacheManager::new(3);

        // Allocate with steps between to create LRU ordering
        let idx0 = mgr.allocate(0, 100, 0, 0).unwrap();
        mgr.step();
        let idx1 = mgr.allocate(1, 101, 0, 0).unwrap();
        mgr.step();
        let _idx2 = mgr.allocate(2, 102, 0, 0).unwrap();
        mgr.step();

        // Access idx0 to make it recently used
        mgr.access(idx0);

        // LRU should be idx1 (oldest access without recent touch)
        let evicted = mgr.evict_lru();
        assert_eq!(evicted, Some(idx1));
        assert_eq!(mgr.valid_count(), 2);
    }

    #[test]
    fn test_kv_cache_manager_evict_empty() {
        let mut mgr = KvCacheManager::new(3);

        // No valid slots - nothing to evict
        let evicted = mgr.evict_lru();
        assert_eq!(evicted, None);
    }

    #[test]
    fn test_kv_cache_manager_allocate_after_evict() {
        let mut mgr = KvCacheManager::new(2);

        // Fill up
        mgr.allocate(0, 100, 0, 0).unwrap();
        mgr.step();
        mgr.allocate(1, 101, 0, 0).unwrap();
        assert_eq!(mgr.valid_count(), 2);

        // Full
        assert!(mgr.allocate(2, 102, 0, 0).is_none());

        // Evict one
        mgr.evict_lru();
        assert_eq!(mgr.valid_count(), 1);

        // Now can allocate
        let idx = mgr.allocate(2, 102, 0, 0);
        assert!(idx.is_some());
        assert_eq!(mgr.valid_count(), 2);
    }

    // =========================================================================
    // SequentialBatchOrderer Tests
    // =========================================================================

    #[test]
    fn test_sequential_batch_orderer_new() {
        let orderer = SequentialBatchOrderer::new(5);

        assert_eq!(orderer.remaining(), 5);
        assert!(!orderer.is_done());
    }

    #[test]
    fn test_sequential_batch_orderer_sequential() {
        let mut orderer = SequentialBatchOrderer::new(4);

        assert_eq!(orderer.next_batch(), Some(0));
        assert_eq!(orderer.next_batch(), Some(1));
        assert_eq!(orderer.next_batch(), Some(2));
        assert_eq!(orderer.next_batch(), Some(3));
        assert_eq!(orderer.next_batch(), None);
        assert!(orderer.is_done());
    }

    #[test]
    fn test_sequential_batch_orderer_reversed() {
        let mut orderer = SequentialBatchOrderer::reversed(4);

        assert_eq!(orderer.next_batch(), Some(3));
        assert_eq!(orderer.next_batch(), Some(2));
        assert_eq!(orderer.next_batch(), Some(1));
        assert_eq!(orderer.next_batch(), Some(0));
        assert_eq!(orderer.next_batch(), None);
    }

    #[test]
    fn test_sequential_batch_orderer_interleaved_even() {
        let orderer = SequentialBatchOrderer::interleaved(4);
        let order: Vec<_> = orderer.collect();

        // 4 batches: 0, 2, 1, 3
        assert_eq!(order, vec![0, 2, 1, 3]);
    }

    #[test]
    fn test_sequential_batch_orderer_interleaved_odd() {
        let orderer = SequentialBatchOrderer::interleaved(5);
        let order: Vec<_> = orderer.collect();

        // All indices should be present
        assert_eq!(order.len(), 5);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_sequential_batch_orderer_reset() {
        let mut orderer = SequentialBatchOrderer::new(3);

        orderer.next_batch();
        orderer.next_batch();
        assert_eq!(orderer.remaining(), 1);

        orderer.reset();
        assert_eq!(orderer.remaining(), 3);
        assert!(!orderer.is_done());
    }

    #[test]
    fn test_sequential_batch_orderer_remaining() {
        let mut orderer = SequentialBatchOrderer::new(5);

        assert_eq!(orderer.remaining(), 5);
        orderer.next_batch();
        assert_eq!(orderer.remaining(), 4);
        orderer.next_batch();
        assert_eq!(orderer.remaining(), 3);
    }

    #[test]
    fn test_sequential_batch_orderer_iterator() {
        let orderer = SequentialBatchOrderer::new(3);
        let batches: Vec<_> = orderer.collect();

        assert_eq!(batches, vec![0, 1, 2]);
    }

    #[test]
    fn test_sequential_batch_orderer_empty() {
        let mut orderer = SequentialBatchOrderer::new(0);

        assert!(orderer.is_done());
        assert_eq!(orderer.remaining(), 0);
        assert_eq!(orderer.next_batch(), None);
    }

    // =========================================================================
    // FALSIFICATION TESTS
    // =========================================================================

    /// FALSIFICATION TEST: KvCacheManager must maintain valid_count invariant
    ///
    /// valid_count must always equal the number of slots with valid=true
    #[test]
    fn test_falsify_kv_cache_valid_count_invariant() {
        let mut mgr = KvCacheManager::new(5);

        // Allocate some
        mgr.allocate(0, 0, 0, 0);
        mgr.allocate(1, 0, 0, 0);
        mgr.allocate(2, 0, 0, 0);

        assert_eq!(
            mgr.valid_count(),
            3,
            "FALSIFICATION FAILED: valid_count should be 3 after 3 allocations"
        );

        // Evict some
        mgr.evict_lru();
        assert_eq!(
            mgr.valid_count(),
            2,
            "FALSIFICATION FAILED: valid_count should be 2 after 1 eviction"
        );

        mgr.evict_lru();
        mgr.evict_lru();
        assert_eq!(
            mgr.valid_count(),
            0,
            "FALSIFICATION FAILED: valid_count should be 0 after all evictions"
        );

        // Can't go negative
        mgr.evict_lru();
        assert_eq!(
            mgr.valid_count(),
            0,
            "FALSIFICATION FAILED: valid_count should not go negative"
        );
    }

    /// FALSIFICATION TEST: LRU eviction must evict oldest-accessed slot
    ///
    /// Given slots with different access times, evict_lru must always
    /// return the slot with the oldest (smallest) last_access.
    #[test]
    fn test_falsify_lru_eviction_order() {
        let mut mgr = KvCacheManager::new(4);

        // Allocate 4 slots with step advances
        let idx0 = mgr.allocate(0, 0, 0, 0).unwrap();
        mgr.step();
        let idx1 = mgr.allocate(1, 0, 0, 0).unwrap();
        mgr.step();
        let idx2 = mgr.allocate(2, 0, 0, 0).unwrap();
        mgr.step();
        let idx3 = mgr.allocate(3, 0, 0, 0).unwrap();
        mgr.step();

        // Access slots in specific order to set LRU
        // Touch idx2, then idx0 - making idx1, idx3 the oldest
        mgr.access(idx2);
        mgr.step();
        mgr.access(idx0);
        mgr.step();

        // First eviction should be idx1 (oldest not accessed)
        let evicted1 = mgr.evict_lru().unwrap();
        assert_eq!(
            evicted1, idx1,
            "FALSIFICATION FAILED: First eviction should be idx1 (step 1)"
        );

        // Second eviction should be idx3 (next oldest)
        let evicted2 = mgr.evict_lru().unwrap();
        assert_eq!(
            evicted2, idx3,
            "FALSIFICATION FAILED: Second eviction should be idx3 (step 3)"
        );
    }

    /// FALSIFICATION TEST: SequentialBatchOrderer must cover all indices
    ///
    /// After iteration completes, all indices 0..n must have been returned
    /// exactly once.
    #[test]
    fn test_falsify_batch_orderer_covers_all_indices() {
        for n_batches in [1, 2, 3, 5, 10, 100] {
            for orderer in [
                SequentialBatchOrderer::new(n_batches),
                SequentialBatchOrderer::reversed(n_batches),
                SequentialBatchOrderer::interleaved(n_batches),
            ] {
                let batches: Vec<_> = orderer.collect();

                // Must have exactly n_batches elements
                assert_eq!(
                    batches.len(),
                    n_batches,
                    "FALSIFICATION FAILED: orderer returned {} batches, expected {}",
                    batches.len(),
                    n_batches
                );

                // Sort and verify all indices present
                let mut sorted = batches.clone();
                sorted.sort();
                let expected: Vec<_> = (0..n_batches).collect();
                assert_eq!(
                    sorted, expected,
                    "FALSIFICATION FAILED: orderer did not return all indices 0..{}",
                    n_batches
                );
            }
        }
    }

    /// FALSIFICATION TEST: Eviction priority must be monotonic with age
    ///
    /// For valid slots, older access time = higher eviction priority.
    #[test]
    fn test_falsify_eviction_priority_monotonic() {
        let mut slot = KvCacheSlotInfo::new(0, 0, 0, 0);
        slot.touch(10);

        let mut prev_priority = slot.eviction_priority(10);
        for current_step in 11..=100 {
            let priority = slot.eviction_priority(current_step);
            assert!(
                priority >= prev_priority,
                "FALSIFICATION FAILED: eviction priority decreased from {} to {} at step {}",
                prev_priority,
                priority,
                current_step
            );
            prev_priority = priority;
        }
    }
}
