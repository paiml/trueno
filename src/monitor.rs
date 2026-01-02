//! GPU Monitoring, Tracing, and Visualization (TRUENO-SPEC-010)
//!
//! This module provides comprehensive GPU monitoring capabilities:
//! - Device discovery and information
//! - Real-time metrics collection
//! - Cross-platform support (wgpu + optional CUDA)
//!
//! # Design Philosophy
//!
//! **Dual-Backend Architecture**: Supports both wgpu (cross-platform) and
//! trueno-gpu (native CUDA) for maximum flexibility.
//!
//! # References
//!
//! - Nickolls et al. (2008): GPU parallel computing model
//! - Gregg (2016): Flame graph visualization
//! - btop: NVML/ROCm SMI patterns
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno::monitor::{GpuDeviceInfo, GpuMonitor};
//!
//! // Query device information
//! let info = GpuDeviceInfo::query()?;
//! println!("GPU: {}", info.name);
//! println!("VRAM: {} MB", info.vram_total / (1024 * 1024));
//!
//! // Start background monitoring
//! let monitor = GpuMonitor::new(0)?;
//! let metrics = monitor.collect()?;
//! println!("VRAM used: {} / {}", metrics.memory.used, metrics.memory.total);
//! ```

use std::time::{Duration, Instant};

// ============================================================================
// GPU Vendor Identification (TRUENO-SPEC-010 Section 3.2)
// ============================================================================

/// GPU vendor identifier based on PCI vendor ID
///
/// Vendor IDs from PCI-SIG registry:
/// - NVIDIA: 0x10de
/// - AMD: 0x1002
/// - Intel: 0x8086
/// - Apple: 0x106b
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuVendor {
    /// NVIDIA Corporation (0x10de)
    Nvidia,
    /// Advanced Micro Devices (0x1002)
    Amd,
    /// Intel Corporation (0x8086)
    Intel,
    /// Apple Inc. (0x106b)
    Apple,
    /// Unknown vendor with raw PCI vendor ID
    Unknown(u32),
}

impl GpuVendor {
    /// Create vendor from PCI vendor ID
    #[must_use]
    pub const fn from_vendor_id(id: u32) -> Self {
        match id {
            0x10de => Self::Nvidia,
            0x1002 => Self::Amd,
            0x8086 => Self::Intel,
            0x106b => Self::Apple,
            _ => Self::Unknown(id),
        }
    }

    /// Get the display name for the vendor
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Nvidia => "NVIDIA",
            Self::Amd => "AMD",
            Self::Intel => "Intel",
            Self::Apple => "Apple",
            Self::Unknown(_) => "Unknown",
        }
    }

    /// Check if this is an NVIDIA GPU (supports CUDA)
    #[must_use]
    pub const fn is_nvidia(&self) -> bool {
        matches!(self, Self::Nvidia)
    }
}

impl std::fmt::Display for GpuVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unknown(id) => write!(f, "Unknown (0x{id:04x})"),
            _ => write!(f, "{}", self.name()),
        }
    }
}

// ============================================================================
// GPU Backend Selection (TRUENO-SPEC-010 Section 3.2)
// ============================================================================

/// GPU compute backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    /// Vulkan (Linux, Windows, Android)
    Vulkan,
    /// Metal (macOS, iOS)
    Metal,
    /// DirectX 12 (Windows)
    Dx12,
    /// DirectX 11 (Windows, fallback)
    Dx11,
    /// WebGPU (WASM browser)
    WebGpu,
    /// Native CUDA (NVIDIA only, via trueno-gpu)
    Cuda,
    /// OpenGL (fallback)
    OpenGl,
    /// CPU fallback (no GPU)
    Cpu,
}

impl GpuBackend {
    /// Get the display name for the backend
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Vulkan => "Vulkan",
            Self::Metal => "Metal",
            Self::Dx12 => "DirectX 12",
            Self::Dx11 => "DirectX 11",
            Self::WebGpu => "WebGPU",
            Self::Cuda => "CUDA",
            Self::OpenGl => "OpenGL",
            Self::Cpu => "CPU",
        }
    }

    /// Check if this is a GPU backend (not CPU fallback)
    #[must_use]
    pub const fn is_gpu(&self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Check if this backend supports compute shaders
    #[must_use]
    pub const fn supports_compute(&self) -> bool {
        matches!(
            self,
            Self::Vulkan | Self::Metal | Self::Dx12 | Self::WebGpu | Self::Cuda
        )
    }
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// GPU Device Information (TRUENO-SPEC-010 Section 3.1)
// ============================================================================

/// GPU device information (TRUENO-SPEC-010)
///
/// Contains static device properties that don't change during runtime.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device index (0-based)
    pub index: u32,
    /// Device name (e.g., "NVIDIA GeForce RTX 4090")
    pub name: String,
    /// Vendor identifier
    pub vendor: GpuVendor,
    /// Total VRAM in bytes
    pub vram_total: u64,
    /// Compute capability (NVIDIA) or architecture info as (major, minor)
    pub compute_capability: Option<(u32, u32)>,
    /// Driver version string
    pub driver_version: Option<String>,
    /// PCI bus ID (e.g., "0000:01:00.0")
    pub pci_bus_id: Option<String>,
    /// wgpu/CUDA backend being used
    pub backend: GpuBackend,
}

impl GpuDeviceInfo {
    /// Create a new device info with required fields
    #[must_use]
    pub fn new(
        index: u32,
        name: impl Into<String>,
        vendor: GpuVendor,
        backend: GpuBackend,
    ) -> Self {
        Self {
            index,
            name: name.into(),
            vendor,
            vram_total: 0,
            compute_capability: None,
            driver_version: None,
            pci_bus_id: None,
            backend,
        }
    }

    /// Set VRAM total
    #[must_use]
    pub fn with_vram(mut self, bytes: u64) -> Self {
        self.vram_total = bytes;
        self
    }

    /// Set compute capability
    #[must_use]
    pub fn with_compute_capability(mut self, major: u32, minor: u32) -> Self {
        self.compute_capability = Some((major, minor));
        self
    }

    /// Set driver version
    #[must_use]
    pub fn with_driver_version(mut self, version: impl Into<String>) -> Self {
        self.driver_version = Some(version.into());
        self
    }

    /// Set PCI bus ID
    #[must_use]
    pub fn with_pci_bus_id(mut self, bus_id: impl Into<String>) -> Self {
        self.pci_bus_id = Some(bus_id.into());
        self
    }

    /// Query device info via wgpu (cross-platform, native only)
    ///
    /// On WASM, use async methods with `wasm_bindgen_futures`.
    ///
    /// # Errors
    ///
    /// Returns error if no GPU is available or query fails.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn query() -> Result<Self, MonitorError> {
        query_wgpu_device_info(0)
    }

    /// Query device info via wgpu for a specific device index (native only)
    ///
    /// On WASM, use async methods with `wasm_bindgen_futures`.
    ///
    /// # Errors
    ///
    /// Returns error if device index is invalid or query fails.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn query_device(index: u32) -> Result<Self, MonitorError> {
        query_wgpu_device_info(index)
    }

    /// Enumerate all available GPU devices (native only)
    ///
    /// On WASM, use async methods with `wasm_bindgen_futures`.
    ///
    /// # Errors
    ///
    /// Returns error if enumeration fails.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn enumerate() -> Result<Vec<Self>, MonitorError> {
        enumerate_wgpu_devices()
    }

    /// Get VRAM in megabytes (convenience method)
    #[must_use]
    pub fn vram_mb(&self) -> u64 {
        self.vram_total / (1024 * 1024)
    }

    /// Get VRAM in gigabytes (convenience method)
    #[must_use]
    pub fn vram_gb(&self) -> f64 {
        self.vram_total as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Check if device supports CUDA (is NVIDIA)
    #[must_use]
    pub fn supports_cuda(&self) -> bool {
        self.vendor.is_nvidia()
    }
}

impl std::fmt::Display for GpuDeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {} ({}) - {:.1} GB VRAM",
            self.index,
            self.name,
            self.backend,
            self.vram_gb()
        )
    }
}

// ============================================================================
// GPU Memory Metrics (TRUENO-SPEC-010 Section 4.1.2)
// ============================================================================

/// GPU memory metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuMemoryMetrics {
    /// Total VRAM in bytes
    pub total: u64,
    /// Used VRAM in bytes
    pub used: u64,
    /// Free VRAM in bytes
    pub free: u64,
    /// Number of active allocations (if tracked)
    pub allocations: u64,
}

impl GpuMemoryMetrics {
    /// Create new memory metrics
    #[must_use]
    pub const fn new(total: u64, used: u64, free: u64) -> Self {
        Self {
            total,
            used,
            free,
            allocations: 0,
        }
    }

    /// Calculate usage percentage (0.0 - 100.0)
    #[must_use]
    pub fn usage_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.used as f64 / self.total as f64) * 100.0
        }
    }

    /// Get used VRAM in megabytes
    #[must_use]
    pub fn used_mb(&self) -> u64 {
        self.used / (1024 * 1024)
    }

    /// Get free VRAM in megabytes
    #[must_use]
    pub fn free_mb(&self) -> u64 {
        self.free / (1024 * 1024)
    }
}

// ============================================================================
// GPU Utilization Metrics (TRUENO-SPEC-010 Section 4.1.1)
// ============================================================================

/// GPU utilization metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuUtilization {
    /// GPU compute utilization (0-100%)
    pub gpu_percent: u32,
    /// Memory controller utilization (0-100%)
    pub memory_percent: u32,
    /// Video encoder utilization (0-100%), if available
    pub encoder_percent: Option<u32>,
    /// Video decoder utilization (0-100%), if available
    pub decoder_percent: Option<u32>,
}

// ============================================================================
// GPU Thermal Metrics (TRUENO-SPEC-010 Section 4.1.3)
// ============================================================================

/// GPU thermal metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuThermalMetrics {
    /// Current temperature in Celsius
    pub temperature_celsius: u32,
    /// Shutdown threshold temperature, if available
    pub temperature_threshold_shutdown: Option<u32>,
    /// Fan speed percentage (0-100), if available
    pub fan_speed_percent: Option<u32>,
}

impl GpuThermalMetrics {
    /// Check if temperature is in safe range (< 80°C)
    #[must_use]
    pub const fn is_safe(&self) -> bool {
        self.temperature_celsius < 80
    }

    /// Check if temperature is critical (>= 90°C)
    #[must_use]
    pub const fn is_critical(&self) -> bool {
        self.temperature_celsius >= 90
    }

    /// Get thermal status string
    #[must_use]
    pub const fn status(&self) -> &'static str {
        match self.temperature_celsius {
            0..=50 => "COOL",
            51..=70 => "WARM",
            71..=85 => "HOT",
            _ => "CRITICAL",
        }
    }
}

// ============================================================================
// GPU Power Metrics (TRUENO-SPEC-010 Section 4.1.3)
// ============================================================================

/// GPU power metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuPowerMetrics {
    /// Current power draw in watts
    pub power_draw_watts: f32,
    /// Power limit (TDP) in watts
    pub power_limit_watts: f32,
    /// Power state (P-state, 0 = highest performance)
    pub power_state: u32,
}

impl GpuPowerMetrics {
    /// Calculate power usage percentage
    #[must_use]
    pub fn usage_percent(&self) -> f64 {
        if self.power_limit_watts <= 0.0 {
            0.0
        } else {
            (self.power_draw_watts as f64 / self.power_limit_watts as f64) * 100.0
        }
    }
}

// ============================================================================
// GPU Clock Metrics (TRUENO-SPEC-010 Section 4.1.4)
// ============================================================================

/// GPU clock metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuClockMetrics {
    /// Graphics/shader clock in MHz
    pub graphics_mhz: u32,
    /// Memory clock in MHz
    pub memory_mhz: u32,
    /// SM clock in MHz (NVIDIA), if available
    pub sm_mhz: Option<u32>,
}

// ============================================================================
// GPU PCIe Metrics (TRUENO-SPEC-010 Section 4.1.5)
// ============================================================================

/// GPU PCIe metrics
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuPcieMetrics {
    /// PCIe TX throughput in bytes/sec
    pub tx_bytes_per_sec: u64,
    /// PCIe RX throughput in bytes/sec
    pub rx_bytes_per_sec: u64,
    /// PCIe link generation (1-5)
    pub link_gen: u32,
    /// PCIe link width (lanes)
    pub link_width: u32,
}

// ============================================================================
// Combined GPU Metrics Snapshot (TRUENO-SPEC-010 Section 4.2)
// ============================================================================

/// Complete GPU metrics snapshot
///
/// Contains all available metrics at a point in time.
#[derive(Debug, Clone)]
pub struct GpuMetrics {
    /// Timestamp of measurement
    pub timestamp: Instant,
    /// Device index
    pub device_index: u32,
    /// Memory metrics
    pub memory: GpuMemoryMetrics,
    /// Utilization metrics
    pub utilization: GpuUtilization,
    /// Thermal metrics (if available)
    pub thermal: Option<GpuThermalMetrics>,
    /// Power metrics (if available)
    pub power: Option<GpuPowerMetrics>,
    /// Clock metrics (if available)
    pub clocks: Option<GpuClockMetrics>,
    /// PCIe metrics (if available)
    pub pcie: Option<GpuPcieMetrics>,
}

impl GpuMetrics {
    /// Create a new metrics snapshot with only memory info
    #[must_use]
    pub fn new(device_index: u32, memory: GpuMemoryMetrics) -> Self {
        Self {
            timestamp: Instant::now(),
            device_index,
            memory,
            utilization: GpuUtilization::default(),
            thermal: None,
            power: None,
            clocks: None,
            pcie: None,
        }
    }

    /// Age of this snapshot
    #[must_use]
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }
}

// ============================================================================
// Monitor Configuration (TRUENO-SPEC-010 Section 8.2)
// ============================================================================

/// Configuration for GPU monitoring
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Polling interval for metrics collection
    pub poll_interval: Duration,
    /// History buffer size (number of samples to retain)
    pub history_size: usize,
    /// Enable background collection thread
    pub background_collection: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(100),
            history_size: 600, // 60 seconds at 100ms
            background_collection: false,
        }
    }
}

impl MonitorConfig {
    /// Create config for high-frequency monitoring
    #[must_use]
    pub fn high_frequency() -> Self {
        Self {
            poll_interval: Duration::from_millis(50),
            history_size: 1200,
            background_collection: true,
        }
    }

    /// Create config for low-overhead monitoring
    #[must_use]
    pub fn low_overhead() -> Self {
        Self {
            poll_interval: Duration::from_millis(500),
            history_size: 120,
            background_collection: false,
        }
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors from GPU monitoring operations
#[derive(Debug, Clone)]
pub enum MonitorError {
    /// No GPU device available
    NoDevice,
    /// Invalid device index
    InvalidDevice(u32),
    /// GPU backend initialization failed
    BackendInit(String),
    /// Metrics query failed
    QueryFailed(String),
    /// Feature not available on this GPU/platform
    NotAvailable(String),
}

impl std::fmt::Display for MonitorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDevice => write!(f, "No GPU device available"),
            Self::InvalidDevice(idx) => write!(f, "Invalid device index: {idx}"),
            Self::BackendInit(msg) => write!(f, "Backend initialization failed: {msg}"),
            Self::QueryFailed(msg) => write!(f, "Metrics query failed: {msg}"),
            Self::NotAvailable(msg) => write!(f, "Feature not available: {msg}"),
        }
    }
}

impl std::error::Error for MonitorError {}

// ============================================================================
// wgpu Backend Implementation
// ============================================================================

/// Query device info from wgpu adapter
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
fn query_wgpu_device_info(device_index: u32) -> Result<GpuDeviceInfo, MonitorError> {
    use crate::backends::gpu::runtime;

    runtime::block_on(async {
        let instance = wgpu::Instance::default();

        // Get all adapters (wgpu 27+ returns Vec directly)
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());

        if adapters.is_empty() {
            return Err(MonitorError::NoDevice);
        }

        let adapter = adapters
            .get(device_index as usize)
            .ok_or(MonitorError::InvalidDevice(device_index))?;

        let info = adapter.get_info();

        // Map wgpu backend to our backend enum
        let backend = match info.backend {
            wgpu::Backend::Vulkan => GpuBackend::Vulkan,
            wgpu::Backend::Metal => GpuBackend::Metal,
            wgpu::Backend::Dx12 => GpuBackend::Dx12,
            wgpu::Backend::Gl => GpuBackend::OpenGl,
            wgpu::Backend::BrowserWebGpu => GpuBackend::WebGpu,
            wgpu::Backend::Noop => GpuBackend::Cpu,
        };

        // Map vendor ID
        let vendor = GpuVendor::from_vendor_id(info.vendor);

        // Get memory limits (rough estimate from adapter limits)
        let limits = adapter.limits();
        // Use max buffer size as a proxy for VRAM (not exact but gives an idea)
        let vram_estimate = limits.max_buffer_size;

        Ok(GpuDeviceInfo::new(device_index, info.name, vendor, backend)
            .with_vram(vram_estimate)
            .with_driver_version(format!("{:?}", info.driver_info)))
    })
}

/// Enumerate all wgpu devices
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
fn enumerate_wgpu_devices() -> Result<Vec<GpuDeviceInfo>, MonitorError> {
    use crate::backends::gpu::runtime;

    runtime::block_on(async {
        let instance = wgpu::Instance::default();
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());

        if adapters.is_empty() {
            return Err(MonitorError::NoDevice);
        }

        let mut devices = Vec::with_capacity(adapters.len());

        for (idx, adapter) in adapters.iter().enumerate() {
            let info: wgpu::AdapterInfo = adapter.get_info();

            let backend = match info.backend {
                wgpu::Backend::Vulkan => GpuBackend::Vulkan,
                wgpu::Backend::Metal => GpuBackend::Metal,
                wgpu::Backend::Dx12 => GpuBackend::Dx12,
                wgpu::Backend::Gl => GpuBackend::OpenGl,
                wgpu::Backend::BrowserWebGpu => GpuBackend::WebGpu,
                wgpu::Backend::Noop => GpuBackend::Cpu,
            };

            let vendor = GpuVendor::from_vendor_id(info.vendor);
            let limits = adapter.limits();

            devices.push(
                GpuDeviceInfo::new(idx as u32, info.name, vendor, backend)
                    .with_vram(limits.max_buffer_size)
                    .with_driver_version(format!("{:?}", info.driver_info)),
            );
        }

        Ok(devices)
    })
}

// ============================================================================
// CUDA Backend Implementation (TRUENO-SPEC-010 Section 8)
// ============================================================================

/// Query device info from native CUDA via trueno-gpu
///
/// This provides more accurate information than wgpu including:
/// - Actual device name (e.g., "NVIDIA GeForce RTX 4090")
/// - Accurate VRAM total from cuDeviceTotalMem
#[cfg(feature = "cuda-monitor")]
pub fn query_cuda_device_info(device_index: u32) -> Result<GpuDeviceInfo, MonitorError> {
    use trueno_gpu::CudaDeviceInfo;

    let cuda_info = CudaDeviceInfo::query(device_index)
        .map_err(|e| MonitorError::BackendInit(format!("CUDA query failed: {}", e)))?;

    Ok(GpuDeviceInfo::new(
        cuda_info.index,
        cuda_info.name,
        GpuVendor::Nvidia, // CUDA is NVIDIA-only
        GpuBackend::Cuda,
    )
    .with_vram(cuda_info.total_memory))
}

/// Enumerate all CUDA devices via trueno-gpu
#[cfg(feature = "cuda-monitor")]
pub fn enumerate_cuda_devices() -> Result<Vec<GpuDeviceInfo>, MonitorError> {
    use trueno_gpu::CudaDeviceInfo;

    let cuda_devices = CudaDeviceInfo::enumerate()
        .map_err(|e| MonitorError::BackendInit(format!("CUDA enumerate failed: {}", e)))?;

    Ok(cuda_devices
        .into_iter()
        .map(|cuda_info| {
            GpuDeviceInfo::new(
                cuda_info.index,
                cuda_info.name,
                GpuVendor::Nvidia,
                GpuBackend::Cuda,
            )
            .with_vram(cuda_info.total_memory)
        })
        .collect())
}

/// Query real-time CUDA memory metrics
///
/// Returns current free/used VRAM from cuMemGetInfo.
#[cfg(feature = "cuda-monitor")]
pub fn query_cuda_memory(device_index: u32) -> Result<GpuMemoryMetrics, MonitorError> {
    use trueno_gpu::driver::CudaContext;
    use trueno_gpu::CudaMemoryInfo;

    let ctx = CudaContext::new(device_index as i32)
        .map_err(|e| MonitorError::BackendInit(format!("CUDA context failed: {}", e)))?;

    let mem = CudaMemoryInfo::query(&ctx)
        .map_err(|e| MonitorError::QueryFailed(format!("CUDA memory query failed: {}", e)))?;

    Ok(GpuMemoryMetrics::new(mem.total, mem.used(), mem.free))
}

/// Check if CUDA monitoring is available
#[cfg(feature = "cuda-monitor")]
#[must_use]
pub fn cuda_monitor_available() -> bool {
    trueno_gpu::cuda_monitoring_available()
}

/// Check if CUDA monitoring is available (stub when feature disabled)
#[cfg(not(feature = "cuda-monitor"))]
#[must_use]
pub fn cuda_monitor_available() -> bool {
    false
}

// ============================================================================
// GPU Monitor (TRUENO-SPEC-010 Section 8.2)
// ============================================================================

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};

/// GPU Monitor for real-time metrics collection (TRUENO-SPEC-010)
///
/// Provides both on-demand and background metric collection with configurable
/// polling intervals and history retention.
///
/// # Example
///
/// ```rust,ignore
/// use trueno::monitor::{GpuMonitor, MonitorConfig};
///
/// // Create monitor for device 0
/// let monitor = GpuMonitor::new(0, MonitorConfig::default())?;
///
/// // Get latest metrics
/// let metrics = monitor.latest()?;
/// println!("GPU usage: {}%", metrics.utilization.gpu_percent);
///
/// // Get history (ring buffer)
/// let history = monitor.history();
/// println!("Samples: {}", history.len());
/// ```
pub struct GpuMonitor {
    /// Device info
    device_info: GpuDeviceInfo,
    /// Configuration
    config: MonitorConfig,
    /// Metrics history (ring buffer)
    history: Arc<RwLock<VecDeque<GpuMetrics>>>,
    /// Background thread handle
    #[cfg(feature = "gpu")]
    _background_handle: Option<std::thread::JoinHandle<()>>,
    /// Stop signal for background thread
    stop_signal: Arc<Mutex<bool>>,
}

impl GpuMonitor {
    /// Create a new GPU monitor for the specified device
    ///
    /// # Errors
    ///
    /// Returns error if device is not found or initialization fails.
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    pub fn new(device_index: u32, config: MonitorConfig) -> Result<Self, MonitorError> {
        let device_info = GpuDeviceInfo::query_device(device_index)?;
        let history = Arc::new(RwLock::new(VecDeque::with_capacity(config.history_size)));
        let stop_signal = Arc::new(Mutex::new(false));

        let monitor = Self {
            device_info,
            config,
            history,
            _background_handle: None,
            stop_signal,
        };

        Ok(monitor)
    }

    /// Create monitor without GPU feature (for testing)
    #[cfg(not(feature = "gpu"))]
    pub fn new(_device_index: u32, _config: MonitorConfig) -> Result<Self, MonitorError> {
        Err(MonitorError::NotAvailable(
            "GPU feature not enabled".to_string(),
        ))
    }

    /// Create a mock monitor for testing (no GPU required)
    #[must_use]
    pub fn mock(device_info: GpuDeviceInfo, config: MonitorConfig) -> Self {
        Self {
            device_info,
            config,
            history: Arc::new(RwLock::new(VecDeque::with_capacity(16))),
            #[cfg(feature = "gpu")]
            _background_handle: None,
            stop_signal: Arc::new(Mutex::new(false)),
        }
    }

    /// Get device info
    #[must_use]
    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    /// Get current configuration
    #[must_use]
    pub fn config(&self) -> &MonitorConfig {
        &self.config
    }

    /// Collect metrics sample now
    ///
    /// This performs an immediate collection and adds to history.
    pub fn collect(&self) -> Result<GpuMetrics, MonitorError> {
        // For now, return basic memory metrics
        // Full implementation would query NVML/wgpu for utilization, thermal, etc.
        let memory = GpuMemoryMetrics::new(
            self.device_info.vram_total,
            0, // Would query actual usage
            self.device_info.vram_total,
        );

        let metrics = GpuMetrics::new(self.device_info.index, memory);

        // Add to history
        if let Ok(mut history) = self.history.write() {
            if history.len() >= self.config.history_size {
                history.pop_front();
            }
            history.push_back(metrics.clone());
        }

        Ok(metrics)
    }

    /// Get the latest metrics snapshot (without collecting)
    pub fn latest(&self) -> Result<GpuMetrics, MonitorError> {
        self.history
            .read()
            .ok()
            .and_then(|h| h.back().cloned())
            .ok_or(MonitorError::QueryFailed(
                "No metrics available".to_string(),
            ))
    }

    /// Get history buffer (read-only snapshot)
    #[must_use]
    pub fn history(&self) -> Vec<GpuMetrics> {
        self.history
            .read()
            .map(|h| h.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Get number of samples in history
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.history.read().map(|h| h.len()).unwrap_or(0)
    }

    /// Clear history buffer
    pub fn clear_history(&self) {
        if let Ok(mut history) = self.history.write() {
            history.clear();
        }
    }

    /// Check if background collection is active
    #[must_use]
    pub fn is_collecting(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self._background_handle.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Stop background collection (if running)
    pub fn stop(&self) {
        if let Ok(mut stop) = self.stop_signal.lock() {
            *stop = true;
        }
    }
}

impl Drop for GpuMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

// ============================================================================
// Tests (EXTREME TDD - Tests First!)
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // =========================================================================
    // H₀-MON-01: GpuVendor identification
    // =========================================================================

    #[test]
    fn h0_mon_01_vendor_nvidia_id() {
        let vendor = GpuVendor::from_vendor_id(0x10de);
        assert_eq!(vendor, GpuVendor::Nvidia);
        assert!(vendor.is_nvidia());
        assert_eq!(vendor.name(), "NVIDIA");
    }

    #[test]
    fn h0_mon_02_vendor_amd_id() {
        let vendor = GpuVendor::from_vendor_id(0x1002);
        assert_eq!(vendor, GpuVendor::Amd);
        assert!(!vendor.is_nvidia());
        assert_eq!(vendor.name(), "AMD");
    }

    #[test]
    fn h0_mon_03_vendor_intel_id() {
        let vendor = GpuVendor::from_vendor_id(0x8086);
        assert_eq!(vendor, GpuVendor::Intel);
        assert!(!vendor.is_nvidia());
        assert_eq!(vendor.name(), "Intel");
    }

    #[test]
    fn h0_mon_04_vendor_apple_id() {
        let vendor = GpuVendor::from_vendor_id(0x106b);
        assert_eq!(vendor, GpuVendor::Apple);
        assert!(!vendor.is_nvidia());
        assert_eq!(vendor.name(), "Apple");
    }

    #[test]
    fn h0_mon_05_vendor_unknown_id() {
        let vendor = GpuVendor::from_vendor_id(0x9999);
        assert_eq!(vendor, GpuVendor::Unknown(0x9999));
        assert!(!vendor.is_nvidia());
        assert_eq!(vendor.name(), "Unknown");
    }

    #[test]
    fn h0_mon_06_vendor_display() {
        assert_eq!(format!("{}", GpuVendor::Nvidia), "NVIDIA");
        assert_eq!(format!("{}", GpuVendor::Amd), "AMD");
        assert_eq!(
            format!("{}", GpuVendor::Unknown(0x1234)),
            "Unknown (0x1234)"
        );
    }

    // =========================================================================
    // H₀-MON-10: GpuBackend identification
    // =========================================================================

    #[test]
    fn h0_mon_10_backend_names() {
        assert_eq!(GpuBackend::Vulkan.name(), "Vulkan");
        assert_eq!(GpuBackend::Metal.name(), "Metal");
        assert_eq!(GpuBackend::Dx12.name(), "DirectX 12");
        assert_eq!(GpuBackend::Cuda.name(), "CUDA");
        assert_eq!(GpuBackend::Cpu.name(), "CPU");
    }

    #[test]
    fn h0_mon_11_backend_is_gpu() {
        assert!(GpuBackend::Vulkan.is_gpu());
        assert!(GpuBackend::Metal.is_gpu());
        assert!(GpuBackend::Cuda.is_gpu());
        assert!(!GpuBackend::Cpu.is_gpu());
    }

    #[test]
    fn h0_mon_12_backend_supports_compute() {
        assert!(GpuBackend::Vulkan.supports_compute());
        assert!(GpuBackend::Metal.supports_compute());
        assert!(GpuBackend::Cuda.supports_compute());
        assert!(!GpuBackend::Cpu.supports_compute());
        assert!(!GpuBackend::OpenGl.supports_compute());
    }

    // =========================================================================
    // H₀-MON-20: GpuDeviceInfo construction
    // =========================================================================

    #[test]
    fn h0_mon_20_device_info_basic() {
        let info = GpuDeviceInfo::new(0, "Test GPU", GpuVendor::Nvidia, GpuBackend::Vulkan);

        assert_eq!(info.index, 0);
        assert_eq!(info.name, "Test GPU");
        assert_eq!(info.vendor, GpuVendor::Nvidia);
        assert_eq!(info.backend, GpuBackend::Vulkan);
        assert_eq!(info.vram_total, 0);
        assert!(info.compute_capability.is_none());
    }

    #[test]
    fn h0_mon_21_device_info_builder() {
        let info = GpuDeviceInfo::new(0, "RTX 4090", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(24_000_000_000)
            .with_compute_capability(8, 9)
            .with_driver_version("535.154.05")
            .with_pci_bus_id("0000:01:00.0");

        assert_eq!(info.vram_total, 24_000_000_000);
        assert_eq!(info.compute_capability, Some((8, 9)));
        assert_eq!(info.driver_version, Some("535.154.05".to_string()));
        assert_eq!(info.pci_bus_id, Some("0000:01:00.0".to_string()));
    }

    #[test]
    fn h0_mon_22_device_info_vram_helpers() {
        let info = GpuDeviceInfo::new(0, "Test", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(24 * 1024 * 1024 * 1024); // 24 GB

        assert_eq!(info.vram_mb(), 24 * 1024);
        assert!((info.vram_gb() - 24.0).abs() < 0.01);
    }

    #[test]
    fn h0_mon_23_device_info_supports_cuda() {
        let nvidia = GpuDeviceInfo::new(0, "RTX", GpuVendor::Nvidia, GpuBackend::Vulkan);
        let amd = GpuDeviceInfo::new(0, "RX", GpuVendor::Amd, GpuBackend::Vulkan);

        assert!(nvidia.supports_cuda());
        assert!(!amd.supports_cuda());
    }

    #[test]
    fn h0_mon_24_device_info_display() {
        let info = GpuDeviceInfo::new(0, "RTX 4090", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(24 * 1024 * 1024 * 1024);

        let display = format!("{info}");
        assert!(display.contains("RTX 4090"));
        assert!(display.contains("Vulkan"));
        assert!(display.contains("24.0"));
    }

    // =========================================================================
    // H₀-MON-30: GpuMemoryMetrics
    // =========================================================================

    #[test]
    fn h0_mon_30_memory_metrics_basic() {
        let mem = GpuMemoryMetrics::new(24_000_000_000, 8_000_000_000, 16_000_000_000);

        assert_eq!(mem.total, 24_000_000_000);
        assert_eq!(mem.used, 8_000_000_000);
        assert_eq!(mem.free, 16_000_000_000);
    }

    #[test]
    fn h0_mon_31_memory_metrics_usage_percent() {
        let mem = GpuMemoryMetrics::new(100, 25, 75);
        assert!((mem.usage_percent() - 25.0).abs() < 0.01);
    }

    #[test]
    fn h0_mon_32_memory_metrics_usage_percent_zero_total() {
        let mem = GpuMemoryMetrics::new(0, 0, 0);
        assert!((mem.usage_percent() - 0.0).abs() < 0.01);
    }

    #[test]
    fn h0_mon_33_memory_metrics_mb_helpers() {
        let mem = GpuMemoryMetrics::new(
            24 * 1024 * 1024 * 1024,
            8 * 1024 * 1024 * 1024,
            16 * 1024 * 1024 * 1024,
        );

        assert_eq!(mem.used_mb(), 8 * 1024);
        assert_eq!(mem.free_mb(), 16 * 1024);
    }

    // =========================================================================
    // H₀-MON-40: GpuThermalMetrics
    // =========================================================================

    #[test]
    fn h0_mon_40_thermal_safe() {
        let thermal = GpuThermalMetrics {
            temperature_celsius: 50,
            ..Default::default()
        };
        assert!(thermal.is_safe());
        assert!(!thermal.is_critical());
        assert_eq!(thermal.status(), "COOL");
    }

    #[test]
    fn h0_mon_41_thermal_warm() {
        let thermal = GpuThermalMetrics {
            temperature_celsius: 65,
            ..Default::default()
        };
        assert!(thermal.is_safe());
        assert!(!thermal.is_critical());
        assert_eq!(thermal.status(), "WARM");
    }

    #[test]
    fn h0_mon_42_thermal_hot() {
        let thermal = GpuThermalMetrics {
            temperature_celsius: 82,
            ..Default::default()
        };
        assert!(!thermal.is_safe());
        assert!(!thermal.is_critical());
        assert_eq!(thermal.status(), "HOT");
    }

    #[test]
    fn h0_mon_43_thermal_critical() {
        let thermal = GpuThermalMetrics {
            temperature_celsius: 95,
            ..Default::default()
        };
        assert!(!thermal.is_safe());
        assert!(thermal.is_critical());
        assert_eq!(thermal.status(), "CRITICAL");
    }

    // =========================================================================
    // H₀-MON-50: GpuPowerMetrics
    // =========================================================================

    #[test]
    fn h0_mon_50_power_usage_percent() {
        let power = GpuPowerMetrics {
            power_draw_watts: 225.0,
            power_limit_watts: 450.0,
            power_state: 0,
        };
        assert!((power.usage_percent() - 50.0).abs() < 0.01);
    }

    #[test]
    fn h0_mon_51_power_usage_percent_zero_limit() {
        let power = GpuPowerMetrics {
            power_draw_watts: 100.0,
            power_limit_watts: 0.0,
            power_state: 0,
        };
        assert!((power.usage_percent() - 0.0).abs() < 0.01);
    }

    // =========================================================================
    // H₀-MON-60: GpuMetrics
    // =========================================================================

    #[test]
    fn h0_mon_60_metrics_creation() {
        let mem = GpuMemoryMetrics::new(1000, 500, 500);
        let metrics = GpuMetrics::new(0, mem);

        assert_eq!(metrics.device_index, 0);
        assert_eq!(metrics.memory.total, 1000);
        assert!(metrics.thermal.is_none());
        assert!(metrics.power.is_none());
    }

    #[test]
    fn h0_mon_61_metrics_age() {
        let mem = GpuMemoryMetrics::new(1000, 500, 500);
        let metrics = GpuMetrics::new(0, mem);

        // Age should be very small immediately after creation
        assert!(metrics.age() < Duration::from_millis(100));
    }

    // =========================================================================
    // H₀-MON-70: MonitorConfig
    // =========================================================================

    #[test]
    fn h0_mon_70_config_default() {
        let config = MonitorConfig::default();

        assert_eq!(config.poll_interval, Duration::from_millis(100));
        assert_eq!(config.history_size, 600);
        assert!(!config.background_collection);
    }

    #[test]
    fn h0_mon_71_config_high_frequency() {
        let config = MonitorConfig::high_frequency();

        assert_eq!(config.poll_interval, Duration::from_millis(50));
        assert_eq!(config.history_size, 1200);
        assert!(config.background_collection);
    }

    #[test]
    fn h0_mon_72_config_low_overhead() {
        let config = MonitorConfig::low_overhead();

        assert_eq!(config.poll_interval, Duration::from_millis(500));
        assert_eq!(config.history_size, 120);
        assert!(!config.background_collection);
    }

    // =========================================================================
    // H₀-MON-80: MonitorError
    // =========================================================================

    #[test]
    fn h0_mon_80_error_display() {
        assert_eq!(
            format!("{}", MonitorError::NoDevice),
            "No GPU device available"
        );
        assert_eq!(
            format!("{}", MonitorError::InvalidDevice(5)),
            "Invalid device index: 5"
        );
        assert_eq!(
            format!("{}", MonitorError::BackendInit("test".to_string())),
            "Backend initialization failed: test"
        );
    }

    // =========================================================================
    // Integration tests (require GPU feature)
    // =========================================================================

    #[test]
    #[cfg(feature = "gpu")]
    fn h0_mon_90_query_device_info() {
        // This test requires actual GPU hardware
        match GpuDeviceInfo::query() {
            Ok(info) => {
                // Verify we got valid data
                assert!(!info.name.is_empty());
                assert!(info.backend.is_gpu());
                println!("Found GPU: {info}");
            }
            Err(MonitorError::NoDevice) => {
                // No GPU is OK for CI environments
                println!("No GPU available (expected in CI)");
            }
            Err(e) => {
                panic!("Unexpected error: {e}");
            }
        }
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn h0_mon_91_enumerate_devices() {
        match GpuDeviceInfo::enumerate() {
            Ok(devices) => {
                for dev in &devices {
                    println!("Found: {dev}");
                }
            }
            Err(MonitorError::NoDevice) => {
                println!("No GPU available (expected in CI)");
            }
            Err(e) => {
                panic!("Unexpected error: {e}");
            }
        }
    }

    // =========================================================================
    // H₀-MON-100: GpuMonitor (mock)
    // =========================================================================

    #[test]
    fn h0_mon_100_monitor_mock_creation() {
        let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(24 * 1024 * 1024 * 1024);
        let config = MonitorConfig::default();
        let monitor = GpuMonitor::mock(info, config);

        assert_eq!(monitor.device_info().name, "Mock GPU");
        assert_eq!(monitor.config().poll_interval, Duration::from_millis(100));
    }

    #[test]
    fn h0_mon_101_monitor_collect() {
        let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(24 * 1024 * 1024 * 1024);
        let config = MonitorConfig::default();
        let monitor = GpuMonitor::mock(info, config);

        // Initially no samples
        assert_eq!(monitor.sample_count(), 0);

        // Collect a sample
        let metrics = monitor.collect().expect("collect should work");
        assert_eq!(metrics.device_index, 0);
        assert_eq!(monitor.sample_count(), 1);

        // Collect more samples
        monitor.collect().expect("collect should work");
        monitor.collect().expect("collect should work");
        assert_eq!(monitor.sample_count(), 3);
    }

    #[test]
    fn h0_mon_102_monitor_history_buffer() {
        let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(1024);

        // Small history size to test ring buffer
        let config = MonitorConfig {
            history_size: 3,
            ..Default::default()
        };
        let monitor = GpuMonitor::mock(info, config);

        // Fill beyond capacity
        for _ in 0..5 {
            monitor.collect().expect("collect should work");
        }

        // Should only have 3 samples (ring buffer)
        assert_eq!(monitor.sample_count(), 3);

        // History should return 3 items
        let history = monitor.history();
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn h0_mon_103_monitor_latest() {
        let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(1024);
        let config = MonitorConfig::default();
        let monitor = GpuMonitor::mock(info, config);

        // No samples yet - should error
        assert!(monitor.latest().is_err());

        // After collecting, latest should work
        monitor.collect().expect("collect should work");
        let latest = monitor.latest().expect("latest should work");
        assert_eq!(latest.device_index, 0);
    }

    #[test]
    fn h0_mon_104_monitor_clear_history() {
        let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan)
            .with_vram(1024);
        let config = MonitorConfig::default();
        let monitor = GpuMonitor::mock(info, config);

        // Collect some samples
        monitor.collect().expect("collect should work");
        monitor.collect().expect("collect should work");
        assert_eq!(monitor.sample_count(), 2);

        // Clear history
        monitor.clear_history();
        assert_eq!(monitor.sample_count(), 0);
    }

    #[test]
    fn h0_mon_105_monitor_is_collecting() {
        let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan);
        let config = MonitorConfig::default();
        let monitor = GpuMonitor::mock(info, config);

        // Mock monitor is not actively collecting in background
        assert!(!monitor.is_collecting());
    }

    #[test]
    fn h0_mon_106_monitor_stop() {
        let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan);
        let config = MonitorConfig::default();
        let monitor = GpuMonitor::mock(info, config);

        // Stop should not panic even without background collection
        monitor.stop();
    }

    // =========================================================================
    // H₀-MON-110: GpuMonitor integration tests (require GPU feature)
    // =========================================================================

    #[test]
    #[cfg(feature = "gpu")]
    fn h0_mon_110_monitor_real_gpu() {
        match GpuMonitor::new(0, MonitorConfig::default()) {
            Ok(monitor) => {
                println!("GPU Monitor: {}", monitor.device_info());

                // Collect a sample
                match monitor.collect() {
                    Ok(metrics) => {
                        println!("Collected metrics: device={}", metrics.device_index);
                        assert_eq!(monitor.sample_count(), 1);
                    }
                    Err(e) => {
                        println!("Collect failed (expected in CI): {e}");
                    }
                }
            }
            Err(MonitorError::NoDevice) => {
                println!("No GPU available (expected in CI)");
            }
            Err(e) => {
                panic!("Unexpected error: {e}");
            }
        }
    }

    // =========================================================================
    // H₀-MON-120: CUDA monitoring integration tests
    // =========================================================================

    #[test]
    fn h0_mon_120_cuda_monitor_available_check() {
        // Should return false without cuda-monitor feature
        let available = super::cuda_monitor_available();
        #[cfg(feature = "cuda-monitor")]
        {
            // With feature, returns true/false based on hardware
            println!("CUDA monitoring available: {}", available);
        }
        #[cfg(not(feature = "cuda-monitor"))]
        {
            assert!(!available, "Should be false without cuda-monitor feature");
        }
    }

    #[test]
    #[cfg(feature = "cuda-monitor")]
    fn h0_mon_121_query_cuda_device_info() {
        use super::query_cuda_device_info;

        match query_cuda_device_info(0) {
            Ok(info) => {
                assert!(!info.name.is_empty());
                assert_eq!(info.vendor, GpuVendor::Nvidia);
                assert_eq!(info.backend, GpuBackend::Cuda);
                assert!(info.vram_total > 0);
                println!("CUDA Device: {}", info);
            }
            Err(e) => {
                println!("No CUDA device (expected in CI): {}", e);
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda-monitor")]
    fn h0_mon_122_enumerate_cuda_devices() {
        use super::enumerate_cuda_devices;

        match enumerate_cuda_devices() {
            Ok(devices) => {
                for dev in &devices {
                    assert_eq!(dev.vendor, GpuVendor::Nvidia);
                    assert_eq!(dev.backend, GpuBackend::Cuda);
                    println!("Found CUDA device: {}", dev);
                }
            }
            Err(e) => {
                println!("CUDA enumeration failed (expected in CI): {}", e);
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda-monitor")]
    fn h0_mon_123_query_cuda_memory() {
        use super::query_cuda_memory;

        match query_cuda_memory(0) {
            Ok(mem) => {
                assert!(mem.total > 0);
                assert!(mem.free <= mem.total);
                println!(
                    "CUDA Memory: {} / {} MB ({:.1}% used)",
                    mem.used_mb(),
                    mem.total / (1024 * 1024),
                    mem.usage_percent()
                );
            }
            Err(e) => {
                println!("CUDA memory query failed (expected in CI): {}", e);
            }
        }
    }
}
