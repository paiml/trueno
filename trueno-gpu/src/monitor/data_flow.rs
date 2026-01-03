//! Data Flow Tracking (TRUENO-SPEC-023)
//!
//! PCIe bandwidth monitoring and memory transfer tracking for
//! host-device data movement.
//!
//! # Transfer Directions
//!
//! - H2D: Host to Device (CPU → GPU)
//! - D2H: Device to Host (GPU → CPU)
//! - D2D: Device to Device (GPU → GPU, same device)
//! - P2P: Peer to Peer (GPU → GPU, NVLink/PCIe)
//!
//! # References
//!
//! - PCIe 4.0: 31.5 GB/s theoretical (x16)
//! - PCIe 5.0: 63 GB/s theoretical (x16)

use std::collections::VecDeque;
use std::time::Instant;

use super::device::DeviceId;

// ============================================================================
// Data Flow Metrics (TRUENO-SPEC-023 Section 5.2)
// ============================================================================

/// Data flow and transfer metrics
#[derive(Debug, Clone)]
pub struct DataFlowMetrics {
    // PCIe metrics
    /// PCIe generation (4, 5, etc.)
    pub pcie_generation: u8,
    /// PCIe link width (x1, x4, x8, x16)
    pub pcie_width: u8,
    /// Theoretical PCIe bandwidth in GB/s
    pub pcie_theoretical_gbps: f64,
    /// Current TX bandwidth in GB/s
    pub pcie_tx_gbps: f64,
    /// Current RX bandwidth in GB/s
    pub pcie_rx_gbps: f64,

    // Active transfers
    /// Currently active transfers
    pub active_transfers: Vec<Transfer>,
    /// Recently completed transfers (last 100)
    pub completed_transfers: VecDeque<Transfer>,

    // Memory bus
    /// GPU memory bus utilization percentage
    pub memory_bus_utilization_pct: f64,
    /// Memory read bandwidth in GB/s
    pub memory_read_gbps: f64,
    /// Memory write bandwidth in GB/s
    pub memory_write_gbps: f64,

    // Buffer pools
    /// Pinned memory used in bytes
    pub pinned_memory_used_bytes: u64,
    /// Pinned memory total in bytes
    pub pinned_memory_total_bytes: u64,
    /// Staging buffer used in bytes
    pub staging_buffer_used_bytes: u64,

    // History (60-point sparklines)
    /// PCIe TX history
    pub pcie_tx_history: VecDeque<f64>,
    /// PCIe RX history
    pub pcie_rx_history: VecDeque<f64>,
    /// Memory bus utilization history
    pub memory_bus_history: VecDeque<f64>,
}

impl DataFlowMetrics {
    /// Maximum history points
    pub const MAX_HISTORY_POINTS: usize = 60;
    /// Maximum completed transfers to keep
    pub const MAX_COMPLETED_TRANSFERS: usize = 100;

    /// Create new data flow metrics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate PCIe theoretical bandwidth based on generation and width
    #[must_use]
    pub fn calculate_pcie_bandwidth(generation: u8, width: u8) -> f64 {
        // GT/s per lane by generation
        let gt_per_lane = match generation {
            1 => 2.5,   // PCIe 1.0
            2 => 5.0,   // PCIe 2.0
            3 => 8.0,   // PCIe 3.0
            4 => 16.0,  // PCIe 4.0
            5 => 32.0,  // PCIe 5.0
            6 => 64.0,  // PCIe 6.0
            _ => 0.0,
        };

        // 128b/130b encoding for PCIe 3.0+, 8b/10b for 1.0/2.0
        let encoding_efficiency = if generation >= 3 { 128.0 / 130.0 } else { 0.8 };

        // GB/s = GT/s * lanes * encoding / 8 bits per byte
        gt_per_lane * width as f64 * encoding_efficiency / 8.0
    }

    /// Set PCIe configuration
    pub fn set_pcie_config(&mut self, generation: u8, width: u8) {
        self.pcie_generation = generation;
        self.pcie_width = width;
        self.pcie_theoretical_gbps = Self::calculate_pcie_bandwidth(generation, width);
    }

    /// Get PCIe TX utilization percentage
    #[must_use]
    pub fn pcie_tx_utilization_pct(&self) -> f64 {
        if self.pcie_theoretical_gbps > 0.0 {
            (self.pcie_tx_gbps / self.pcie_theoretical_gbps) * 100.0
        } else {
            0.0
        }
    }

    /// Get PCIe RX utilization percentage
    #[must_use]
    pub fn pcie_rx_utilization_pct(&self) -> f64 {
        if self.pcie_theoretical_gbps > 0.0 {
            (self.pcie_rx_gbps / self.pcie_theoretical_gbps) * 100.0
        } else {
            0.0
        }
    }

    /// Start tracking a new transfer
    pub fn start_transfer(&mut self, transfer: Transfer) {
        self.active_transfers.push(transfer);
    }

    /// Complete a transfer and move to history
    pub fn complete_transfer(&mut self, transfer_id: TransferId) {
        if let Some(idx) = self.active_transfers.iter().position(|t| t.id == transfer_id) {
            let mut transfer = self.active_transfers.remove(idx);
            transfer.complete();
            self.completed_transfers.push_back(transfer);
            if self.completed_transfers.len() > Self::MAX_COMPLETED_TRANSFERS {
                self.completed_transfers.pop_front();
            }
        }
    }

    /// Update history sparklines
    pub fn update_history(&mut self) {
        self.pcie_tx_history.push_back(self.pcie_tx_gbps);
        if self.pcie_tx_history.len() > Self::MAX_HISTORY_POINTS {
            self.pcie_tx_history.pop_front();
        }

        self.pcie_rx_history.push_back(self.pcie_rx_gbps);
        if self.pcie_rx_history.len() > Self::MAX_HISTORY_POINTS {
            self.pcie_rx_history.pop_front();
        }

        self.memory_bus_history.push_back(self.memory_bus_utilization_pct);
        if self.memory_bus_history.len() > Self::MAX_HISTORY_POINTS {
            self.memory_bus_history.pop_front();
        }
    }

    /// Get total bytes currently being transferred
    #[must_use]
    pub fn bytes_in_flight(&self) -> u64 {
        self.active_transfers
            .iter()
            .map(|t| t.size_bytes.saturating_sub(t.transferred_bytes))
            .sum()
    }

    /// Get pinned memory utilization percentage
    #[must_use]
    pub fn pinned_memory_utilization_pct(&self) -> f64 {
        if self.pinned_memory_total_bytes > 0 {
            (self.pinned_memory_used_bytes as f64 / self.pinned_memory_total_bytes as f64) * 100.0
        } else {
            0.0
        }
    }
}

impl Default for DataFlowMetrics {
    fn default() -> Self {
        Self {
            pcie_generation: 4,
            pcie_width: 16,
            pcie_theoretical_gbps: Self::calculate_pcie_bandwidth(4, 16),
            pcie_tx_gbps: 0.0,
            pcie_rx_gbps: 0.0,
            active_transfers: Vec::new(),
            completed_transfers: VecDeque::with_capacity(Self::MAX_COMPLETED_TRANSFERS),
            memory_bus_utilization_pct: 0.0,
            memory_read_gbps: 0.0,
            memory_write_gbps: 0.0,
            pinned_memory_used_bytes: 0,
            pinned_memory_total_bytes: 0,
            staging_buffer_used_bytes: 0,
            pcie_tx_history: VecDeque::with_capacity(Self::MAX_HISTORY_POINTS),
            pcie_rx_history: VecDeque::with_capacity(Self::MAX_HISTORY_POINTS),
            memory_bus_history: VecDeque::with_capacity(Self::MAX_HISTORY_POINTS),
        }
    }
}

// ============================================================================
// Transfer Tracking
// ============================================================================

/// Unique transfer identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransferId(pub u64);

impl TransferId {
    /// Generate a new unique transfer ID
    #[must_use]
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for TransferId {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory transfer between host and device
#[derive(Debug, Clone)]
pub struct Transfer {
    /// Unique transfer ID
    pub id: TransferId,
    /// Transfer direction
    pub direction: TransferDirection,
    /// Source memory location
    pub source: MemoryLocation,
    /// Destination memory location
    pub destination: MemoryLocation,
    /// Total transfer size in bytes
    pub size_bytes: u64,
    /// Bytes transferred so far
    pub transferred_bytes: u64,
    /// Transfer start time
    pub start_time: Instant,
    /// Transfer end time (if completed)
    pub end_time: Option<Instant>,
    /// Transfer status
    pub status: TransferStatus,
    /// Human-readable label
    pub label: String,
}

impl Transfer {
    /// Create a new transfer
    #[must_use]
    pub fn new(
        direction: TransferDirection,
        source: MemoryLocation,
        destination: MemoryLocation,
        size_bytes: u64,
    ) -> Self {
        Self {
            id: TransferId::new(),
            direction,
            source,
            destination,
            size_bytes,
            transferred_bytes: 0,
            start_time: Instant::now(),
            end_time: None,
            status: TransferStatus::Pending,
            label: String::new(),
        }
    }

    /// Create H2D transfer
    #[must_use]
    pub fn host_to_device(size_bytes: u64, device_id: DeviceId) -> Self {
        Self::new(
            TransferDirection::HostToDevice,
            MemoryLocation::SystemRam,
            MemoryLocation::GpuVram(device_id),
            size_bytes,
        )
    }

    /// Create D2H transfer
    #[must_use]
    pub fn device_to_host(size_bytes: u64, device_id: DeviceId) -> Self {
        Self::new(
            TransferDirection::DeviceToHost,
            MemoryLocation::GpuVram(device_id),
            MemoryLocation::SystemRam,
            size_bytes,
        )
    }

    /// Set label
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Get transfer progress percentage (0.0-100.0)
    #[must_use]
    pub fn progress_pct(&self) -> f64 {
        if self.size_bytes == 0 {
            return 100.0;
        }
        (self.transferred_bytes as f64 / self.size_bytes as f64) * 100.0
    }

    /// Get elapsed time
    #[must_use]
    pub fn elapsed(&self) -> std::time::Duration {
        match self.end_time {
            Some(end) => end.duration_since(self.start_time),
            None => self.start_time.elapsed(),
        }
    }

    /// Get elapsed time in milliseconds
    #[must_use]
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed().as_secs_f64() * 1000.0
    }

    /// Get current bandwidth in GB/s
    #[must_use]
    pub fn bandwidth_gbps(&self) -> f64 {
        let elapsed_s = self.elapsed().as_secs_f64();
        if elapsed_s > 0.0 {
            self.transferred_bytes as f64 / elapsed_s / 1e9
        } else {
            0.0
        }
    }

    /// Update transfer progress
    pub fn update_progress(&mut self, bytes_transferred: u64) {
        self.transferred_bytes = bytes_transferred;
        if self.status == TransferStatus::Pending {
            self.status = TransferStatus::InProgress;
        }
    }

    /// Mark transfer as complete
    pub fn complete(&mut self) {
        self.transferred_bytes = self.size_bytes;
        self.status = TransferStatus::Completed;
        self.end_time = Some(Instant::now());
    }

    /// Mark transfer as failed
    pub fn fail(&mut self, _reason: &str) {
        self.status = TransferStatus::Failed;
        self.end_time = Some(Instant::now());
    }
}

/// Transfer direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host (CPU) to Device (GPU)
    HostToDevice,
    /// Device (GPU) to Host (CPU)
    DeviceToHost,
    /// Device to Device (same GPU)
    DeviceToDevice,
    /// Peer to Peer (GPU to GPU via NVLink/PCIe)
    PeerToPeer,
}

impl std::fmt::Display for TransferDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HostToDevice => write!(f, "H→D"),
            Self::DeviceToHost => write!(f, "D→H"),
            Self::DeviceToDevice => write!(f, "D→D"),
            Self::PeerToPeer => write!(f, "P2P"),
        }
    }
}

/// Memory location
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLocation {
    /// System RAM
    SystemRam,
    /// Pinned (page-locked) memory
    PinnedMemory,
    /// GPU VRAM
    GpuVram(DeviceId),
    /// Unified/managed memory
    UnifiedMemory,
}

impl std::fmt::Display for MemoryLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SystemRam => write!(f, "RAM"),
            Self::PinnedMemory => write!(f, "Pinned"),
            Self::GpuVram(id) => write!(f, "VRAM:{}", id),
            Self::UnifiedMemory => write!(f, "Unified"),
        }
    }
}

/// Transfer status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStatus {
    /// Transfer queued
    Pending,
    /// Transfer in progress
    InProgress,
    /// Transfer completed
    Completed,
    /// Transfer failed
    Failed,
}

// ============================================================================
// Tests (Extreme TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // H031: PCIe Bandwidth Tests
    // =========================================================================

    #[test]
    fn h031_pcie_bandwidth_gen4_x16() {
        // PCIe 4.0 x16 = ~31.5 GB/s
        let bw = DataFlowMetrics::calculate_pcie_bandwidth(4, 16);
        assert!((bw - 31.5).abs() < 0.5, "Expected ~31.5 GB/s, got {}", bw);
    }

    #[test]
    fn h031_pcie_bandwidth_gen5_x16() {
        // PCIe 5.0 x16 = ~63 GB/s
        let bw = DataFlowMetrics::calculate_pcie_bandwidth(5, 16);
        assert!((bw - 63.0).abs() < 1.0, "Expected ~63 GB/s, got {}", bw);
    }

    #[test]
    fn h031_pcie_bandwidth_gen3_x16() {
        // PCIe 3.0 x16 = ~15.75 GB/s
        let bw = DataFlowMetrics::calculate_pcie_bandwidth(3, 16);
        assert!((bw - 15.75).abs() < 0.5, "Expected ~15.75 GB/s, got {}", bw);
    }

    #[test]
    fn h031_pcie_bandwidth_gen4_x8() {
        // PCIe 4.0 x8 = ~15.75 GB/s
        let bw = DataFlowMetrics::calculate_pcie_bandwidth(4, 8);
        assert!((bw - 15.75).abs() < 0.5, "Expected ~15.75 GB/s, got {}", bw);
    }

    #[test]
    fn h031_pcie_bandwidth_unknown_gen() {
        let bw = DataFlowMetrics::calculate_pcie_bandwidth(99, 16);
        assert_eq!(bw, 0.0);
    }

    // =========================================================================
    // H032: Data Flow Metrics Tests
    // =========================================================================

    #[test]
    fn h032_data_flow_default() {
        let metrics = DataFlowMetrics::default();
        assert_eq!(metrics.pcie_generation, 4);
        assert_eq!(metrics.pcie_width, 16);
        assert!(metrics.pcie_theoretical_gbps > 30.0);
    }

    #[test]
    fn h032_data_flow_set_pcie_config() {
        let mut metrics = DataFlowMetrics::new();
        metrics.set_pcie_config(5, 16);

        assert_eq!(metrics.pcie_generation, 5);
        assert_eq!(metrics.pcie_width, 16);
        assert!(metrics.pcie_theoretical_gbps > 60.0);
    }

    #[test]
    fn h032_data_flow_tx_utilization() {
        let mut metrics = DataFlowMetrics::new();
        metrics.pcie_theoretical_gbps = 31.5;
        metrics.pcie_tx_gbps = 15.75;

        assert!((metrics.pcie_tx_utilization_pct() - 50.0).abs() < 1.0);
    }

    #[test]
    fn h032_data_flow_rx_utilization() {
        let mut metrics = DataFlowMetrics::new();
        metrics.pcie_theoretical_gbps = 31.5;
        metrics.pcie_rx_gbps = 7.875;

        assert!((metrics.pcie_rx_utilization_pct() - 25.0).abs() < 1.0);
    }

    #[test]
    fn h032_data_flow_utilization_zero_theoretical() {
        let mut metrics = DataFlowMetrics::new();
        metrics.pcie_theoretical_gbps = 0.0;

        assert_eq!(metrics.pcie_tx_utilization_pct(), 0.0);
        assert_eq!(metrics.pcie_rx_utilization_pct(), 0.0);
    }

    // =========================================================================
    // H033: Transfer Tracking Tests
    // =========================================================================

    #[test]
    fn h033_transfer_id_unique() {
        let id1 = TransferId::new();
        let id2 = TransferId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn h033_transfer_new() {
        let transfer = Transfer::new(
            TransferDirection::HostToDevice,
            MemoryLocation::SystemRam,
            MemoryLocation::GpuVram(DeviceId::nvidia(0)),
            1024 * 1024,
        );

        assert_eq!(transfer.direction, TransferDirection::HostToDevice);
        assert_eq!(transfer.size_bytes, 1024 * 1024);
        assert_eq!(transfer.transferred_bytes, 0);
        assert_eq!(transfer.status, TransferStatus::Pending);
    }

    #[test]
    fn h033_transfer_h2d_helper() {
        let transfer = Transfer::host_to_device(1024 * 1024, DeviceId::nvidia(0));
        assert_eq!(transfer.direction, TransferDirection::HostToDevice);
        assert_eq!(transfer.source, MemoryLocation::SystemRam);
    }

    #[test]
    fn h033_transfer_d2h_helper() {
        let transfer = Transfer::device_to_host(1024 * 1024, DeviceId::nvidia(0));
        assert_eq!(transfer.direction, TransferDirection::DeviceToHost);
        assert_eq!(transfer.destination, MemoryLocation::SystemRam);
    }

    #[test]
    fn h033_transfer_with_label() {
        let transfer = Transfer::host_to_device(1024, DeviceId::nvidia(0))
            .with_label("tensor_a");
        assert_eq!(transfer.label, "tensor_a");
    }

    #[test]
    fn h033_transfer_progress() {
        let mut transfer = Transfer::host_to_device(1000, DeviceId::nvidia(0));
        assert_eq!(transfer.progress_pct(), 0.0);

        transfer.update_progress(500);
        assert!((transfer.progress_pct() - 50.0).abs() < 0.01);
        assert_eq!(transfer.status, TransferStatus::InProgress);

        transfer.complete();
        assert_eq!(transfer.progress_pct(), 100.0);
        assert_eq!(transfer.status, TransferStatus::Completed);
    }

    #[test]
    fn h033_transfer_progress_zero_size() {
        let transfer = Transfer::host_to_device(0, DeviceId::nvidia(0));
        assert_eq!(transfer.progress_pct(), 100.0);
    }

    #[test]
    fn h033_transfer_elapsed() {
        let transfer = Transfer::host_to_device(1024, DeviceId::nvidia(0));
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(transfer.elapsed_ms() >= 10.0);
    }

    #[test]
    fn h033_transfer_bandwidth() {
        let mut transfer = Transfer::host_to_device(1_000_000_000, DeviceId::nvidia(0)); // 1 GB
        std::thread::sleep(std::time::Duration::from_millis(100));
        transfer.transferred_bytes = 100_000_000; // 100 MB in ~100ms = ~1 GB/s

        let bw = transfer.bandwidth_gbps();
        // Should be roughly 1 GB/s (but timing is imprecise in tests)
        assert!(bw > 0.5 && bw < 2.0, "Bandwidth {} GB/s unexpected", bw);
    }

    // =========================================================================
    // H034: Transfer Direction Display Tests
    // =========================================================================

    #[test]
    fn h034_transfer_direction_display() {
        assert_eq!(format!("{}", TransferDirection::HostToDevice), "H→D");
        assert_eq!(format!("{}", TransferDirection::DeviceToHost), "D→H");
        assert_eq!(format!("{}", TransferDirection::DeviceToDevice), "D→D");
        assert_eq!(format!("{}", TransferDirection::PeerToPeer), "P2P");
    }

    // =========================================================================
    // H035: Memory Location Display Tests
    // =========================================================================

    #[test]
    fn h035_memory_location_display() {
        assert_eq!(format!("{}", MemoryLocation::SystemRam), "RAM");
        assert_eq!(format!("{}", MemoryLocation::PinnedMemory), "Pinned");
        assert_eq!(format!("{}", MemoryLocation::UnifiedMemory), "Unified");
        assert!(format!("{}", MemoryLocation::GpuVram(DeviceId::nvidia(0))).contains("NVIDIA"));
    }

    // =========================================================================
    // H036: History Tracking Tests
    // =========================================================================

    #[test]
    fn h036_history_update() {
        let mut metrics = DataFlowMetrics::new();

        for i in 0..100 {
            metrics.pcie_tx_gbps = i as f64;
            metrics.pcie_rx_gbps = i as f64 * 0.5;
            metrics.memory_bus_utilization_pct = i as f64;
            metrics.update_history();
        }

        assert_eq!(metrics.pcie_tx_history.len(), DataFlowMetrics::MAX_HISTORY_POINTS);
        assert_eq!(metrics.pcie_rx_history.len(), DataFlowMetrics::MAX_HISTORY_POINTS);
        assert_eq!(metrics.memory_bus_history.len(), DataFlowMetrics::MAX_HISTORY_POINTS);
    }

    // =========================================================================
    // H037: Transfer Management Tests
    // =========================================================================

    #[test]
    fn h037_start_and_complete_transfer() {
        let mut metrics = DataFlowMetrics::new();
        let transfer = Transfer::host_to_device(1024, DeviceId::nvidia(0));
        let id = transfer.id;

        metrics.start_transfer(transfer);
        assert_eq!(metrics.active_transfers.len(), 1);

        metrics.complete_transfer(id);
        assert_eq!(metrics.active_transfers.len(), 0);
        assert_eq!(metrics.completed_transfers.len(), 1);
    }

    #[test]
    fn h037_completed_transfer_limit() {
        let mut metrics = DataFlowMetrics::new();

        for _ in 0..150 {
            let transfer = Transfer::host_to_device(1024, DeviceId::nvidia(0));
            let id = transfer.id;
            metrics.start_transfer(transfer);
            metrics.complete_transfer(id);
        }

        assert_eq!(metrics.completed_transfers.len(), DataFlowMetrics::MAX_COMPLETED_TRANSFERS);
    }

    #[test]
    fn h037_bytes_in_flight() {
        let mut metrics = DataFlowMetrics::new();

        let mut t1 = Transfer::host_to_device(1000, DeviceId::nvidia(0));
        t1.transferred_bytes = 400;

        let mut t2 = Transfer::host_to_device(2000, DeviceId::nvidia(0));
        t2.transferred_bytes = 500;

        metrics.start_transfer(t1);
        metrics.start_transfer(t2);

        // (1000-400) + (2000-500) = 600 + 1500 = 2100
        assert_eq!(metrics.bytes_in_flight(), 2100);
    }

    // =========================================================================
    // H038: Pinned Memory Tests
    // =========================================================================

    #[test]
    fn h038_pinned_memory_utilization() {
        let mut metrics = DataFlowMetrics::new();
        metrics.pinned_memory_used_bytes = 512 * 1024 * 1024; // 512 MB
        metrics.pinned_memory_total_bytes = 1024 * 1024 * 1024; // 1 GB

        assert!((metrics.pinned_memory_utilization_pct() - 50.0).abs() < 0.01);
    }

    #[test]
    fn h038_pinned_memory_utilization_zero_total() {
        let metrics = DataFlowMetrics::new();
        assert_eq!(metrics.pinned_memory_utilization_pct(), 0.0);
    }
}
