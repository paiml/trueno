//! Compute Flow Visualization (TRUENO-SPEC-022)
//!
//! Pipeline metrics for tracking compute throughput, kernel execution,
//! and efficiency across CPU and GPU devices.
//!
//! # Pipeline Stages
//!
//! ```text
//! INPUT → COMPUTE → REDUCE → OUTPUT
//! (H2D)   (Kernel)  (Tile)   (D2H)
//! ```
//!
//! # References
//!
//! - [Harris2007] Optimizing Parallel Reduction in CUDA
//! - [Volkov2008] Tile size optimization

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::device::DeviceId;

// ============================================================================
// Compute Metrics (TRUENO-SPEC-022 Section 4.2)
// ============================================================================

/// Compute pipeline metrics
#[derive(Debug, Clone)]
pub struct ComputeMetrics {
    /// Per-device compute metrics
    pub devices: Vec<DeviceComputeMetrics>,

    /// Active kernel executions
    pub active_kernels: Vec<KernelExecution>,

    /// Pipeline stage latencies
    /// Input stage latency (H2D transfers)
    pub input_latency_ms: f64,
    /// Compute stage latency (kernel execution)
    pub compute_latency_ms: f64,
    /// Reduce stage latency (tile reduction)
    pub reduce_latency_ms: f64,
    /// Output stage latency (D2H transfers)
    pub output_latency_ms: f64,

    /// Throughput in operations per second
    pub operations_per_second: f64,
    /// Achieved FLOPS
    pub flops_achieved: f64,
    /// Theoretical peak FLOPS
    pub flops_theoretical: f64,

    /// Compute efficiency percentage
    pub compute_efficiency_pct: f64,
    /// Memory efficiency percentage
    pub memory_efficiency_pct: f64,
}

impl ComputeMetrics {
    /// Create new compute metrics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate total pipeline latency
    #[must_use]
    pub fn total_latency_ms(&self) -> f64 {
        self.input_latency_ms
            + self.compute_latency_ms
            + self.reduce_latency_ms
            + self.output_latency_ms
    }

    /// Calculate throughput in operations per second
    #[must_use]
    pub fn throughput_ops(&self) -> f64 {
        let latency_s = self.total_latency_ms() / 1000.0;
        if latency_s > 0.0 {
            1.0 / latency_s
        } else {
            0.0
        }
    }

    /// Get compute efficiency as a percentage
    #[must_use]
    pub fn efficiency_percent(&self) -> f64 {
        if self.flops_theoretical > 0.0 {
            (self.flops_achieved / self.flops_theoretical) * 100.0
        } else {
            0.0
        }
    }

    /// Add a device's compute metrics
    pub fn add_device(&mut self, device_metrics: DeviceComputeMetrics) {
        self.devices.push(device_metrics);
    }

    /// Track a kernel execution
    pub fn track_kernel(&mut self, kernel: KernelExecution) {
        self.active_kernels.push(kernel);
    }

    /// Clear completed kernels
    pub fn clear_completed_kernels(&mut self) {
        self.active_kernels
            .retain(|k| k.status != KernelStatus::Completed);
    }
}

impl Default for ComputeMetrics {
    fn default() -> Self {
        Self {
            devices: Vec::new(),
            active_kernels: Vec::new(),
            input_latency_ms: 0.0,
            compute_latency_ms: 0.0,
            reduce_latency_ms: 0.0,
            output_latency_ms: 0.0,
            operations_per_second: 0.0,
            flops_achieved: 0.0,
            flops_theoretical: 0.0,
            compute_efficiency_pct: 0.0,
            memory_efficiency_pct: 0.0,
        }
    }
}

// ============================================================================
// Device Compute Metrics
// ============================================================================

/// Per-device compute metrics
#[derive(Debug, Clone)]
pub struct DeviceComputeMetrics {
    /// Device ID
    pub device_id: DeviceId,
    /// Compute utilization (0.0-100.0)
    pub utilization_pct: f64,
    /// Streaming multiprocessor / Compute unit active percentage
    pub sm_active_pct: f64,
    /// Active warps
    pub warps_active: u32,
    /// Maximum warps
    pub warps_max: u32,
    /// Current clock speed in MHz
    pub clock_mhz: u32,
    /// Maximum clock speed in MHz
    pub clock_max_mhz: u32,
    /// Current power in watts
    pub power_watts: f64,
    /// Power limit in watts
    pub power_limit_watts: f64,
    /// Temperature in Celsius
    pub temperature_c: f64,
    /// Throttle reason (if any)
    pub throttle_reason: Option<ThrottleReason>,
    /// Utilization history (60-point sparkline)
    pub history: VecDeque<f64>,
}

impl DeviceComputeMetrics {
    /// Maximum history points
    pub const MAX_HISTORY_POINTS: usize = 60;

    /// Create new device compute metrics
    #[must_use]
    pub fn new(device_id: DeviceId) -> Self {
        Self {
            device_id,
            utilization_pct: 0.0,
            sm_active_pct: 0.0,
            warps_active: 0,
            warps_max: 0,
            clock_mhz: 0,
            clock_max_mhz: 0,
            power_watts: 0.0,
            power_limit_watts: 0.0,
            temperature_c: 0.0,
            throttle_reason: None,
            history: VecDeque::with_capacity(Self::MAX_HISTORY_POINTS),
        }
    }

    /// Update utilization and add to history
    pub fn update_utilization(&mut self, pct: f64) {
        self.utilization_pct = pct;
        self.history.push_back(pct);
        if self.history.len() > Self::MAX_HISTORY_POINTS {
            self.history.pop_front();
        }
    }

    /// Get warp occupancy percentage
    #[must_use]
    pub fn warp_occupancy_pct(&self) -> f64 {
        if self.warps_max == 0 {
            return 0.0;
        }
        (self.warps_active as f64 / self.warps_max as f64) * 100.0
    }

    /// Get clock ratio (current/max)
    #[must_use]
    pub fn clock_ratio(&self) -> f64 {
        if self.clock_max_mhz == 0 {
            return 0.0;
        }
        self.clock_mhz as f64 / self.clock_max_mhz as f64
    }

    /// Get power ratio (current/limit)
    #[must_use]
    pub fn power_ratio(&self) -> f64 {
        if self.power_limit_watts == 0.0 {
            return 0.0;
        }
        self.power_watts / self.power_limit_watts
    }

    /// Check if device is throttling
    #[must_use]
    pub fn is_throttling(&self) -> bool {
        self.throttle_reason.is_some() && self.throttle_reason != Some(ThrottleReason::None)
    }
}

// ThrottleReason is defined in device.rs and re-exported from mod.rs
use super::device::ThrottleReason;

// ============================================================================
// Kernel Execution Tracking
// ============================================================================

/// Active kernel execution
#[derive(Debug, Clone)]
pub struct KernelExecution {
    /// Kernel name
    pub name: String,
    /// Grid dimensions (x, y, z)
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (x, y, z)
    pub block_dim: (u32, u32, u32),
    /// Shared memory per block in bytes
    pub shared_mem_bytes: usize,
    /// Registers per thread
    pub registers_per_thread: u32,
    /// Theoretical occupancy percentage
    pub occupancy_pct: f64,
    /// Elapsed time in milliseconds
    pub elapsed_ms: f64,
    /// Execution status
    pub status: KernelStatus,
    /// Device executing this kernel
    pub device_id: DeviceId,
    /// Start time
    pub start_time: Instant,
}

impl KernelExecution {
    /// Create a new kernel execution tracker
    #[must_use]
    pub fn new(name: impl Into<String>, device_id: DeviceId) -> Self {
        Self {
            name: name.into(),
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
            registers_per_thread: 0,
            occupancy_pct: 0.0,
            elapsed_ms: 0.0,
            status: KernelStatus::Pending,
            device_id,
            start_time: Instant::now(),
        }
    }

    /// Set grid and block dimensions
    #[must_use]
    pub fn with_dims(mut self, grid: (u32, u32, u32), block: (u32, u32, u32)) -> Self {
        self.grid_dim = grid;
        self.block_dim = block;
        self
    }

    /// Set shared memory usage
    #[must_use]
    pub fn with_shared_mem(mut self, bytes: usize) -> Self {
        self.shared_mem_bytes = bytes;
        self
    }

    /// Set register usage
    #[must_use]
    pub fn with_registers(mut self, regs: u32) -> Self {
        self.registers_per_thread = regs;
        self
    }

    /// Get total thread count
    #[must_use]
    pub fn total_threads(&self) -> u64 {
        let grid_total = self.grid_dim.0 as u64 * self.grid_dim.1 as u64 * self.grid_dim.2 as u64;
        let block_total =
            self.block_dim.0 as u64 * self.block_dim.1 as u64 * self.block_dim.2 as u64;
        grid_total * block_total
    }

    /// Get total blocks
    #[must_use]
    pub fn total_blocks(&self) -> u64 {
        self.grid_dim.0 as u64 * self.grid_dim.1 as u64 * self.grid_dim.2 as u64
    }

    /// Get threads per block
    #[must_use]
    pub fn threads_per_block(&self) -> u32 {
        self.block_dim.0 * self.block_dim.1 * self.block_dim.2
    }

    /// Mark kernel as running
    pub fn start(&mut self) {
        self.status = KernelStatus::Running;
        self.start_time = Instant::now();
    }

    /// Mark kernel as completed and record elapsed time
    pub fn complete(&mut self) {
        self.status = KernelStatus::Completed;
        self.elapsed_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;
    }

    /// Update elapsed time for running kernel
    pub fn update_elapsed(&mut self) {
        if self.status == KernelStatus::Running {
            self.elapsed_ms = self.start_time.elapsed().as_secs_f64() * 1000.0;
        }
    }

    /// Get progress percentage (estimated based on time if available)
    #[must_use]
    pub fn progress_pct(&self) -> f64 {
        match self.status {
            KernelStatus::Pending | KernelStatus::Failed => 0.0,
            KernelStatus::Completed => 100.0,
            KernelStatus::Running => {
                // Can't know actual progress without kernel instrumentation
                // Return a placeholder based on typical kernel times
                (self.elapsed_ms / 100.0).min(99.0)
            }
        }
    }
}

/// Kernel execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelStatus {
    /// Kernel queued but not started
    Pending,
    /// Kernel currently executing
    Running,
    /// Kernel completed successfully
    Completed,
    /// Kernel failed
    Failed,
}

// ============================================================================
// FLOPS Calculation Helpers
// ============================================================================

/// Calculate theoretical FLOPS for a GEMM operation
///
/// FLOPS = 2 * M * N * K (for FMA operations)
#[must_use]
pub fn gemm_flops(m: u64, n: u64, k: u64) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64
}

/// Calculate achieved GFLOPS from operation count and time
#[must_use]
pub fn achieved_gflops(flops: f64, duration: Duration) -> f64 {
    let seconds = duration.as_secs_f64();
    if seconds > 0.0 {
        flops / seconds / 1e9
    } else {
        0.0
    }
}

/// Calculate compute efficiency percentage
#[must_use]
pub fn compute_efficiency(achieved_gflops: f64, theoretical_gflops: f64) -> f64 {
    if theoretical_gflops > 0.0 {
        (achieved_gflops / theoretical_gflops) * 100.0
    } else {
        0.0
    }
}

// ============================================================================
// Tests (Extreme TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // H021: Compute Metrics Tests
    // =========================================================================

    #[test]
    fn h021_compute_metrics_default() {
        let metrics = ComputeMetrics::default();
        assert!(metrics.devices.is_empty());
        assert!(metrics.active_kernels.is_empty());
        assert_eq!(metrics.input_latency_ms, 0.0);
    }

    #[test]
    fn h021_compute_metrics_total_latency() {
        let mut metrics = ComputeMetrics::default();
        metrics.input_latency_ms = 1.0;
        metrics.compute_latency_ms = 5.0;
        metrics.reduce_latency_ms = 0.5;
        metrics.output_latency_ms = 1.0;

        assert!((metrics.total_latency_ms() - 7.5).abs() < 0.01);
    }

    #[test]
    fn h021_compute_metrics_throughput() {
        let mut metrics = ComputeMetrics::default();
        metrics.input_latency_ms = 10.0; // 10ms total = 100 ops/s

        assert!((metrics.throughput_ops() - 100.0).abs() < 1.0);
    }

    #[test]
    fn h021_compute_metrics_throughput_zero_latency() {
        let metrics = ComputeMetrics::default();
        assert_eq!(metrics.throughput_ops(), 0.0);
    }

    #[test]
    fn h021_compute_metrics_efficiency() {
        let mut metrics = ComputeMetrics::default();
        metrics.flops_achieved = 500.0;
        metrics.flops_theoretical = 1000.0;

        assert!((metrics.efficiency_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn h021_compute_metrics_efficiency_zero_theoretical() {
        let metrics = ComputeMetrics::default();
        assert_eq!(metrics.efficiency_percent(), 0.0);
    }

    // =========================================================================
    // H022: Device Compute Metrics Tests
    // =========================================================================

    #[test]
    fn h022_device_metrics_new() {
        let metrics = DeviceComputeMetrics::new(DeviceId::nvidia(0));
        assert_eq!(metrics.device_id, DeviceId::nvidia(0));
        assert_eq!(metrics.utilization_pct, 0.0);
    }

    #[test]
    fn h022_device_metrics_update_history() {
        let mut metrics = DeviceComputeMetrics::new(DeviceId::nvidia(0));

        for i in 0..100 {
            metrics.update_utilization(i as f64);
        }

        assert_eq!(
            metrics.history.len(),
            DeviceComputeMetrics::MAX_HISTORY_POINTS
        );
        // Last value should be 99
        assert!((metrics.history.back().unwrap() - 99.0).abs() < 0.01);
    }

    #[test]
    fn h022_device_metrics_warp_occupancy() {
        let mut metrics = DeviceComputeMetrics::new(DeviceId::nvidia(0));
        metrics.warps_active = 48;
        metrics.warps_max = 64;

        assert!((metrics.warp_occupancy_pct() - 75.0).abs() < 0.01);
    }

    #[test]
    fn h022_device_metrics_warp_occupancy_zero() {
        let metrics = DeviceComputeMetrics::new(DeviceId::nvidia(0));
        assert_eq!(metrics.warp_occupancy_pct(), 0.0);
    }

    #[test]
    fn h022_device_metrics_clock_ratio() {
        let mut metrics = DeviceComputeMetrics::new(DeviceId::nvidia(0));
        metrics.clock_mhz = 1800;
        metrics.clock_max_mhz = 2400;

        assert!((metrics.clock_ratio() - 0.75).abs() < 0.01);
    }

    #[test]
    fn h022_device_metrics_power_ratio() {
        let mut metrics = DeviceComputeMetrics::new(DeviceId::nvidia(0));
        metrics.power_watts = 300.0;
        metrics.power_limit_watts = 450.0;

        assert!((metrics.power_ratio() - 0.666).abs() < 0.01);
    }

    #[test]
    fn h022_device_metrics_throttling() {
        let mut metrics = DeviceComputeMetrics::new(DeviceId::nvidia(0));
        assert!(!metrics.is_throttling());

        metrics.throttle_reason = Some(ThrottleReason::Thermal);
        assert!(metrics.is_throttling());

        metrics.throttle_reason = Some(ThrottleReason::None);
        assert!(!metrics.is_throttling());
    }

    // =========================================================================
    // H023: Kernel Execution Tests
    // =========================================================================

    #[test]
    fn h023_kernel_execution_new() {
        let kernel = KernelExecution::new("test_kernel", DeviceId::nvidia(0));
        assert_eq!(kernel.name, "test_kernel");
        assert_eq!(kernel.status, KernelStatus::Pending);
    }

    #[test]
    fn h023_kernel_execution_builder() {
        let kernel = KernelExecution::new("gemm", DeviceId::nvidia(0))
            .with_dims((128, 128, 1), (16, 16, 1))
            .with_shared_mem(4096)
            .with_registers(32);

        assert_eq!(kernel.grid_dim, (128, 128, 1));
        assert_eq!(kernel.block_dim, (16, 16, 1));
        assert_eq!(kernel.shared_mem_bytes, 4096);
        assert_eq!(kernel.registers_per_thread, 32);
    }

    #[test]
    fn h023_kernel_execution_total_threads() {
        let kernel =
            KernelExecution::new("test", DeviceId::nvidia(0)).with_dims((128, 64, 1), (16, 16, 1));

        // 128 * 64 * 1 blocks * 16 * 16 * 1 threads/block = 2,097,152
        assert_eq!(kernel.total_threads(), 128 * 64 * 16 * 16);
    }

    #[test]
    fn h023_kernel_execution_total_blocks() {
        let kernel =
            KernelExecution::new("test", DeviceId::nvidia(0)).with_dims((128, 64, 2), (16, 16, 1));

        assert_eq!(kernel.total_blocks(), 128 * 64 * 2);
    }

    #[test]
    fn h023_kernel_execution_threads_per_block() {
        let kernel =
            KernelExecution::new("test", DeviceId::nvidia(0)).with_dims((1, 1, 1), (16, 16, 4));

        assert_eq!(kernel.threads_per_block(), 16 * 16 * 4);
    }

    #[test]
    fn h023_kernel_execution_lifecycle() {
        let mut kernel = KernelExecution::new("test", DeviceId::nvidia(0));
        assert_eq!(kernel.status, KernelStatus::Pending);
        assert_eq!(kernel.progress_pct(), 0.0);

        kernel.start();
        assert_eq!(kernel.status, KernelStatus::Running);

        std::thread::sleep(std::time::Duration::from_millis(10));
        kernel.update_elapsed();
        assert!(kernel.elapsed_ms > 0.0);

        kernel.complete();
        assert_eq!(kernel.status, KernelStatus::Completed);
        assert_eq!(kernel.progress_pct(), 100.0);
    }

    // =========================================================================
    // H024: FLOPS Calculation Tests
    // =========================================================================

    #[test]
    fn h024_gemm_flops() {
        // GEMM 1024x1024x1024 = 2 * 1024^3 = 2,147,483,648 FLOPs
        let flops = gemm_flops(1024, 1024, 1024);
        assert!((flops - 2.0 * 1024.0 * 1024.0 * 1024.0).abs() < 1.0);
    }

    #[test]
    fn h024_achieved_gflops() {
        // 2e9 FLOPs in 1 second = 2 GFLOPS
        let gflops = achieved_gflops(2e9, Duration::from_secs(1));
        assert!((gflops - 2.0).abs() < 0.01);
    }

    #[test]
    fn h024_achieved_gflops_zero_time() {
        let gflops = achieved_gflops(1e9, Duration::ZERO);
        assert_eq!(gflops, 0.0);
    }

    #[test]
    fn h024_compute_efficiency_calculation() {
        // 75 GFLOPS achieved out of 100 GFLOPS theoretical = 75%
        let eff = compute_efficiency(75.0, 100.0);
        assert!((eff - 75.0).abs() < 0.01);
    }

    #[test]
    fn h024_compute_efficiency_zero_theoretical() {
        let eff = compute_efficiency(100.0, 0.0);
        assert_eq!(eff, 0.0);
    }

    // =========================================================================
    // H025: Kernel Tracking in Metrics
    // =========================================================================

    #[test]
    fn h025_metrics_track_kernel() {
        let mut metrics = ComputeMetrics::new();
        let kernel = KernelExecution::new("test", DeviceId::nvidia(0));

        metrics.track_kernel(kernel);
        assert_eq!(metrics.active_kernels.len(), 1);
    }

    #[test]
    fn h025_metrics_clear_completed() {
        let mut metrics = ComputeMetrics::new();

        let mut k1 = KernelExecution::new("running", DeviceId::nvidia(0));
        k1.status = KernelStatus::Running;

        let mut k2 = KernelExecution::new("completed", DeviceId::nvidia(0));
        k2.status = KernelStatus::Completed;

        metrics.track_kernel(k1);
        metrics.track_kernel(k2);

        assert_eq!(metrics.active_kernels.len(), 2);

        metrics.clear_completed_kernels();
        assert_eq!(metrics.active_kernels.len(), 1);
        assert_eq!(metrics.active_kernels[0].name, "running");
    }
}
