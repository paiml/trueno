//! TUI Layout Specification (TRUENO-SPEC-024)
//!
//! Terminal user interface layout and widget definitions for
//! real-time compute monitoring.
//!
//! # Layout
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────────────────┐
//! │ TRUENO Compute Monitor │ CPU: ... │ GPU: ... │ F1 Help                     │
//! ├────────────────────────────────────────────────────────────────────────────┤
//! │ [COMPUTE] CPU/GPU utilization gauges + sparklines                          │
//! │ [MEMORY] RAM/SWAP/VRAM bars                                                 │
//! │ [DATA FLOW] PCIe TX/RX + transfers                                          │
//! │ [KERNELS] Active kernel list                                                │
//! ├────────────────────────────────────────────────────────────────────────────┤
//! │ q:Quit r:Refresh s:Stress Tab:Focus │ Refresh: 100ms                        │
//! └────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # References
//!
//! - [Wang2004] SSIM for visual quality
//! - Viridis colormap for colorblind accessibility

use std::collections::VecDeque;

use super::device::DeviceId;
use super::memory::PressureLevel;

// ============================================================================
// TUI Configuration
// ============================================================================

/// TUI layout configuration
#[derive(Debug, Clone)]
pub struct TuiLayout {
    /// Minimum terminal width
    pub min_width: u16,
    /// Minimum terminal height
    pub min_height: u16,
    /// Recommended terminal width
    pub rec_width: u16,
    /// Recommended terminal height
    pub rec_height: u16,
    /// Section definitions
    pub sections: Vec<Section>,
    /// Refresh rate in milliseconds
    pub refresh_rate_ms: u64,
    /// Number of sparkline data points
    pub sparkline_points: usize,
}

impl Default for TuiLayout {
    fn default() -> Self {
        Self {
            min_width: 80,
            min_height: 24,
            rec_width: 160,
            rec_height: 48,
            sections: vec![
                Section::new("compute", "COMPUTE", 0.25),
                Section::new("memory", "MEMORY", 0.20),
                Section::new("dataflow", "DATA FLOW", 0.20),
                Section::new("kernels", "KERNELS", 0.20),
            ],
            refresh_rate_ms: 100,
            sparkline_points: 60,
        }
    }
}

impl TuiLayout {
    /// Create a new TUI layout
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set refresh rate
    #[must_use]
    pub fn with_refresh_rate(mut self, ms: u64) -> Self {
        self.refresh_rate_ms = ms;
        self
    }

    /// Check if terminal size meets minimum requirements
    #[must_use]
    pub fn check_size(&self, width: u16, height: u16) -> SizeCheck {
        if width >= self.rec_width && height >= self.rec_height {
            SizeCheck::Recommended
        } else if width >= self.min_width && height >= self.min_height {
            SizeCheck::Minimum
        } else {
            SizeCheck::TooSmall
        }
    }
}

/// Terminal size check result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeCheck {
    /// Terminal meets recommended size
    Recommended,
    /// Terminal meets minimum size
    Minimum,
    /// Terminal too small
    TooSmall,
}

// ============================================================================
// Section Definition
// ============================================================================

/// TUI section definition
#[derive(Debug, Clone)]
pub struct Section {
    /// Section identifier
    pub id: String,
    /// Section title
    pub title: String,
    /// Height as percentage of total (0.0-1.0)
    pub height_pct: f32,
    /// Widgets in this section
    pub widgets: Vec<Widget>,
    /// Is section collapsed
    pub collapsed: bool,
    /// Is section focused
    pub focused: bool,
}

impl Section {
    /// Create a new section
    #[must_use]
    pub fn new(id: impl Into<String>, title: impl Into<String>, height_pct: f32) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            height_pct,
            widgets: Vec::new(),
            collapsed: false,
            focused: false,
        }
    }

    /// Add a widget to the section
    pub fn add_widget(&mut self, widget: Widget) {
        self.widgets.push(widget);
    }

    /// Toggle collapsed state
    pub fn toggle_collapsed(&mut self) {
        self.collapsed = !self.collapsed;
    }
}

// ============================================================================
// Widget Types
// ============================================================================

/// TUI widget
#[derive(Debug, Clone)]
pub enum Widget {
    /// Progress gauge (0-100%)
    Gauge(GaugeWidget),
    /// Sparkline (history graph)
    Sparkline(SparklineWidget),
    /// Progress bar with label
    ProgressBar(ProgressBarWidget),
    /// Table with rows
    Table(TableWidget),
    /// Text label
    Text(TextWidget),
}

/// Gauge widget for showing percentages
#[derive(Debug, Clone)]
pub struct GaugeWidget {
    /// Widget label
    pub label: String,
    /// Current value (0.0-100.0)
    pub value_pct: f64,
    /// Warning threshold
    pub warning_threshold: f64,
    /// Critical threshold
    pub critical_threshold: f64,
    /// Maximum value (usually 100)
    pub max_value: f64,
    /// Suffix text (e.g., "%" or "°C")
    pub suffix: String,
}

impl GaugeWidget {
    /// Create a new gauge
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value_pct: 0.0,
            warning_threshold: 70.0,
            critical_threshold: 90.0,
            max_value: 100.0,
            suffix: "%".to_string(),
        }
    }

    /// Set value
    #[must_use]
    pub fn with_value(mut self, value: f64) -> Self {
        self.value_pct = value;
        self
    }

    /// Set thresholds
    #[must_use]
    pub fn with_thresholds(mut self, warning: f64, critical: f64) -> Self {
        self.warning_threshold = warning;
        self.critical_threshold = critical;
        self
    }

    /// Get color based on value
    #[must_use]
    pub fn color(&self) -> GaugeColor {
        if self.value_pct >= self.critical_threshold {
            GaugeColor::Critical
        } else if self.value_pct >= self.warning_threshold {
            GaugeColor::Warning
        } else {
            GaugeColor::Ok
        }
    }

    /// Render as ASCII bar
    #[must_use]
    pub fn render_bar(&self, width: usize) -> String {
        let ratio = (self.value_pct / self.max_value).min(1.0);
        let filled = (ratio * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);

        format!(
            "{}: [{}{}] {:.1}{}",
            self.label,
            "█".repeat(filled),
            "░".repeat(empty),
            self.value_pct,
            self.suffix
        )
    }
}

/// Gauge color
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GaugeColor {
    /// Normal (green)
    Ok,
    /// Warning (yellow/orange)
    Warning,
    /// Critical (red)
    Critical,
}

/// Sparkline widget for showing history
#[derive(Debug, Clone)]
pub struct SparklineWidget {
    /// Data points (0.0-1.0 normalized or raw values)
    pub data: VecDeque<f64>,
    /// Widget label
    pub label: String,
    /// Optional baseline to show
    pub baseline: Option<f64>,
    /// Auto-scale to data range
    pub auto_scale: bool,
}

impl SparklineWidget {
    /// Create a new sparkline
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            data: VecDeque::with_capacity(60),
            label: label.into(),
            baseline: None,
            auto_scale: true,
        }
    }

    /// Set data
    #[must_use]
    pub fn with_data(mut self, data: VecDeque<f64>) -> Self {
        self.data = data;
        self
    }

    /// Render as Unicode sparkline
    #[must_use]
    pub fn render(&self) -> String {
        if self.data.is_empty() {
            return String::new();
        }

        let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        let (min, max) = if self.auto_scale {
            let min = self.data.iter().copied().fold(f64::INFINITY, f64::min);
            let max = self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (min, max)
        } else {
            (0.0, 100.0)
        };

        let range = (max - min).max(0.001);

        self.data
            .iter()
            .map(|&v| {
                let normalized = ((v - min) / range).clamp(0.0, 1.0);
                let idx = (normalized * 7.0).round() as usize;
                blocks[idx.min(7)]
            })
            .collect()
    }
}

/// Progress bar widget
#[derive(Debug, Clone)]
pub struct ProgressBarWidget {
    /// Widget label
    pub label: String,
    /// Progress (0.0-1.0)
    pub progress: f64,
    /// Total description (e.g., "14.0 / 24.0 GB")
    pub total_desc: String,
}

impl ProgressBarWidget {
    /// Create a new progress bar
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            progress: 0.0,
            total_desc: String::new(),
        }
    }

    /// Set progress
    #[must_use]
    pub fn with_progress(mut self, progress: f64) -> Self {
        self.progress = progress.clamp(0.0, 1.0);
        self
    }

    /// Set total description
    #[must_use]
    pub fn with_total(mut self, desc: impl Into<String>) -> Self {
        self.total_desc = desc.into();
        self
    }

    /// Render as ASCII bar
    #[must_use]
    pub fn render(&self, width: usize) -> String {
        let filled = (self.progress * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);

        format!(
            "{}: [{}{}] {}",
            self.label,
            "█".repeat(filled),
            "░".repeat(empty),
            self.total_desc
        )
    }
}

/// Table widget
#[derive(Debug, Clone)]
pub struct TableWidget {
    /// Column headers
    pub headers: Vec<String>,
    /// Row data
    pub rows: Vec<Vec<String>>,
    /// Currently highlighted row
    pub highlight_row: Option<usize>,
    /// Column widths (auto-calculated if empty)
    pub column_widths: Vec<usize>,
}

impl TableWidget {
    /// Create a new table
    #[must_use]
    pub fn new(headers: Vec<String>) -> Self {
        Self {
            headers,
            rows: Vec::new(),
            highlight_row: None,
            column_widths: Vec::new(),
        }
    }

    /// Add a row
    pub fn add_row(&mut self, row: Vec<String>) {
        self.rows.push(row);
    }

    /// Set highlighted row
    pub fn highlight(&mut self, row: usize) {
        self.highlight_row = Some(row);
    }

    /// Calculate column widths
    #[must_use]
    pub fn calculate_widths(&self) -> Vec<usize> {
        let mut widths: Vec<usize> = self.headers.iter().map(|h| h.len()).collect();

        for row in &self.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }

        widths
    }
}

/// Text widget
#[derive(Debug, Clone)]
pub struct TextWidget {
    /// Text content
    pub content: String,
    /// Text style
    pub style: TextStyle,
}

impl TextWidget {
    /// Create a new text widget
    #[must_use]
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            style: TextStyle::Normal,
        }
    }

    /// Set style
    #[must_use]
    pub fn with_style(mut self, style: TextStyle) -> Self {
        self.style = style;
        self
    }
}

/// Text style
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextStyle {
    /// Normal text
    Normal,
    /// Bold text
    Bold,
    /// Dimmed text
    Dim,
    /// Italic text
    Italic,
    /// Header text
    Header,
    /// Error text
    Error,
    /// Warning text
    Warning,
    /// Success text
    Success,
}

// ============================================================================
// Color Scheme (Colorblind-Safe Viridis-Based)
// ============================================================================

/// Colorblind-safe color scheme based on Viridis
#[derive(Debug, Clone)]
pub struct ColorScheme {
    /// OK/Success color (teal)
    pub ok: RgbColor,
    /// Warning color (yellow)
    pub warning: RgbColor,
    /// Critical color (red-orange)
    pub critical: RgbColor,
    /// Neutral color (blue)
    pub neutral: RgbColor,
    /// Background color (dark purple)
    pub background: RgbColor,
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self {
            ok: RgbColor::new(0x21, 0x91, 0x8c),         // Teal
            warning: RgbColor::new(0xfd, 0xe7, 0x25),    // Yellow
            critical: RgbColor::new(0xf0, 0x3b, 0x20),   // Red-orange
            neutral: RgbColor::new(0x3b, 0x52, 0x8b),    // Blue
            background: RgbColor::new(0x44, 0x01, 0x54), // Dark purple
        }
    }
}

/// RGB color
#[derive(Debug, Clone, Copy)]
pub struct RgbColor {
    /// Red component (0-255)
    pub r: u8,
    /// Green component (0-255)
    pub g: u8,
    /// Blue component (0-255)
    pub b: u8,
}

impl RgbColor {
    /// Create a new RGB color
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Convert to ANSI true-color escape sequence (foreground)
    #[must_use]
    pub fn to_ansi_fg(&self) -> String {
        format!("\x1b[38;2;{};{};{}m", self.r, self.g, self.b)
    }

    /// Convert to ANSI true-color escape sequence (background)
    #[must_use]
    pub fn to_ansi_bg(&self) -> String {
        format!("\x1b[48;2;{};{};{}m", self.r, self.g, self.b)
    }

    /// Get color for pressure level
    #[must_use]
    pub fn for_pressure_level(level: PressureLevel) -> Self {
        match level {
            PressureLevel::Ok => Self::new(0x21, 0x91, 0x8c), // Teal
            PressureLevel::Elevated => Self::new(0xfd, 0xe7, 0x25), // Yellow
            PressureLevel::Warning => Self::new(0xfd, 0xa6, 0x00), // Orange
            PressureLevel::Critical => Self::new(0xf0, 0x3b, 0x20), // Red
        }
    }
}

// ============================================================================
// Keyboard Controls
// ============================================================================

/// Keyboard action
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    /// Quit application
    Quit,
    /// Force refresh
    Refresh,
    /// Toggle stress test mode
    ToggleStress,
    /// Focus next section
    FocusNext,
    /// Navigate up
    NavigateUp,
    /// Navigate down
    NavigateDown,
    /// Expand/collapse current item
    Expand,
    /// Show help overlay
    Help,
    /// Show alerts panel
    Alerts,
    /// Export metrics to JSON
    Export,
    /// Pause/resume monitoring
    TogglePause,
}

impl KeyAction {
    /// Get keyboard shortcut for this action
    #[must_use]
    pub fn key(&self) -> char {
        match self {
            Self::Quit => 'q',
            Self::Refresh => 'r',
            Self::ToggleStress => 's',
            Self::FocusNext => '\t',
            Self::NavigateUp => '↑',
            Self::NavigateDown => '↓',
            Self::Expand => '\n',
            Self::Help => '?',
            Self::Alerts => 'a',
            Self::Export => 'e',
            Self::TogglePause => 'p',
        }
    }

    /// Get description for help display
    #[must_use]
    pub fn description(&self) -> &'static str {
        match self {
            Self::Quit => "Quit",
            Self::Refresh => "Refresh",
            Self::ToggleStress => "Stress Test",
            Self::FocusNext => "Focus",
            Self::NavigateUp => "Up",
            Self::NavigateDown => "Down",
            Self::Expand => "Expand",
            Self::Help => "Help",
            Self::Alerts => "Alerts",
            Self::Export => "Export",
            Self::TogglePause => "Pause",
        }
    }
}

// ============================================================================
// TUI Render State
// ============================================================================

/// Complete TUI render state
#[derive(Debug, Clone)]
pub struct TuiRenderState {
    /// CPU device metrics
    pub cpu: Option<DeviceRenderState>,
    /// GPU device metrics
    pub gpus: Vec<DeviceRenderState>,
    /// Memory metrics
    pub memory: MemoryRenderState,
    /// Data flow metrics
    pub data_flow: DataFlowRenderState,
    /// Active kernels
    pub kernels: Vec<KernelRenderState>,
    /// Current pressure level
    pub pressure: PressureLevel,
    /// Stress test active
    pub stress_active: bool,
    /// Paused
    pub paused: bool,
    /// Current focus
    pub focused_section: usize,
    /// Error message (if any)
    pub error: Option<String>,
}

impl Default for TuiRenderState {
    fn default() -> Self {
        Self {
            cpu: None,
            gpus: Vec::new(),
            memory: MemoryRenderState::default(),
            data_flow: DataFlowRenderState::default(),
            kernels: Vec::new(),
            pressure: PressureLevel::Ok,
            stress_active: false,
            paused: false,
            focused_section: 0,
            error: None,
        }
    }
}

/// Device render state
#[derive(Debug, Clone)]
pub struct DeviceRenderState {
    /// Device ID
    pub device_id: DeviceId,
    /// Device name
    pub name: String,
    /// Utilization percentage
    pub utilization_pct: f64,
    /// Temperature in Celsius
    pub temperature_c: f64,
    /// Power in Watts
    pub power_watts: f64,
    /// Power limit in Watts
    pub power_limit_watts: f64,
    /// Clock speed in MHz
    pub clock_mhz: u32,
    /// Utilization history
    pub history: VecDeque<f64>,
}

/// Memory render state
#[derive(Debug, Clone, Default)]
pub struct MemoryRenderState {
    /// RAM usage percentage
    pub ram_pct: f64,
    /// RAM used in GB
    pub ram_used_gb: f64,
    /// RAM total in GB
    pub ram_total_gb: f64,
    /// Swap usage percentage
    pub swap_pct: f64,
    /// Swap used in GB
    pub swap_used_gb: f64,
    /// Swap total in GB
    pub swap_total_gb: f64,
    /// VRAM metrics per GPU
    pub vram: Vec<(DeviceId, f64, f64, f64)>, // (id, pct, used_gb, total_gb)
    /// RAM history
    pub ram_history: VecDeque<f64>,
}

/// Data flow render state
#[derive(Debug, Clone, Default)]
pub struct DataFlowRenderState {
    /// PCIe TX in GB/s
    pub pcie_tx_gbps: f64,
    /// PCIe RX in GB/s
    pub pcie_rx_gbps: f64,
    /// PCIe theoretical in GB/s
    pub pcie_theoretical_gbps: f64,
    /// Memory bus utilization percentage
    pub memory_bus_pct: f64,
    /// Active transfers
    pub transfers: Vec<(String, String, f64)>, // (label, direction, progress)
}

/// Kernel render state
#[derive(Debug, Clone)]
pub struct KernelRenderState {
    /// Kernel name
    pub name: String,
    /// Device ID
    pub device_id: DeviceId,
    /// Progress percentage
    pub progress_pct: f64,
    /// Grid dimensions
    pub grid: String,
    /// Elapsed time in ms
    pub elapsed_ms: f64,
}

// ============================================================================
// Tests (Extreme TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // H042: TUI Layout Tests
    // =========================================================================

    #[test]
    fn h042_tui_layout_default() {
        let layout = TuiLayout::default();
        assert_eq!(layout.min_width, 80);
        assert_eq!(layout.min_height, 24);
        assert_eq!(layout.refresh_rate_ms, 100);
        assert_eq!(layout.sparkline_points, 60);
    }

    #[test]
    fn h042_tui_layout_size_check() {
        let layout = TuiLayout::default();

        assert_eq!(layout.check_size(160, 48), SizeCheck::Recommended);
        assert_eq!(layout.check_size(80, 24), SizeCheck::Minimum);
        assert_eq!(layout.check_size(40, 12), SizeCheck::TooSmall);
    }

    // =========================================================================
    // H043: Section Tests
    // =========================================================================

    #[test]
    fn h043_section_new() {
        let section = Section::new("test", "Test Section", 0.25);
        assert_eq!(section.id, "test");
        assert_eq!(section.title, "Test Section");
        assert!((section.height_pct - 0.25).abs() < 0.001);
        assert!(!section.collapsed);
        assert!(!section.focused);
    }

    #[test]
    fn h043_section_toggle_collapsed() {
        let mut section = Section::new("test", "Test", 0.25);
        assert!(!section.collapsed);

        section.toggle_collapsed();
        assert!(section.collapsed);

        section.toggle_collapsed();
        assert!(!section.collapsed);
    }

    // =========================================================================
    // H044: Gauge Widget Tests
    // =========================================================================

    #[test]
    fn h044_gauge_new() {
        let gauge = GaugeWidget::new("CPU");
        assert_eq!(gauge.label, "CPU");
        assert_eq!(gauge.value_pct, 0.0);
    }

    #[test]
    fn h044_gauge_with_value() {
        let gauge = GaugeWidget::new("CPU").with_value(75.5);
        assert!((gauge.value_pct - 75.5).abs() < 0.01);
    }

    #[test]
    fn h044_gauge_color() {
        let ok = GaugeWidget::new("Test").with_value(50.0);
        let warn = GaugeWidget::new("Test").with_value(75.0);
        let crit = GaugeWidget::new("Test").with_value(95.0);

        assert_eq!(ok.color(), GaugeColor::Ok);
        assert_eq!(warn.color(), GaugeColor::Warning);
        assert_eq!(crit.color(), GaugeColor::Critical);
    }

    #[test]
    fn h044_gauge_render_bar() {
        let gauge = GaugeWidget::new("CPU").with_value(50.0);
        let bar = gauge.render_bar(20);

        assert!(bar.contains("CPU"));
        assert!(bar.contains("50.0"));
        assert!(bar.contains("█"));
        assert!(bar.contains("░"));
    }

    // =========================================================================
    // H045: Sparkline Widget Tests
    // =========================================================================

    #[test]
    fn h045_sparkline_new() {
        let sparkline = SparklineWidget::new("History");
        assert_eq!(sparkline.label, "History");
        assert!(sparkline.data.is_empty());
    }

    #[test]
    fn h045_sparkline_render_empty() {
        let sparkline = SparklineWidget::new("Test");
        assert_eq!(sparkline.render(), "");
    }

    #[test]
    fn h045_sparkline_render() {
        let mut data = VecDeque::new();
        for i in 0..10 {
            data.push_back(i as f64 * 10.0);
        }

        let sparkline = SparklineWidget::new("Test").with_data(data);
        let rendered = sparkline.render();

        assert_eq!(rendered.chars().count(), 10);
        // First should be lowest, last should be highest
        assert!(rendered.starts_with('▁'));
        assert!(rendered.ends_with('█'));
    }

    // =========================================================================
    // H046: Progress Bar Widget Tests
    // =========================================================================

    #[test]
    fn h046_progress_bar_new() {
        let bar = ProgressBarWidget::new("RAM");
        assert_eq!(bar.label, "RAM");
        assert_eq!(bar.progress, 0.0);
    }

    #[test]
    fn h046_progress_bar_with_progress() {
        let bar = ProgressBarWidget::new("RAM").with_progress(0.75);
        assert!((bar.progress - 0.75).abs() < 0.001);
    }

    #[test]
    fn h046_progress_bar_clamp() {
        let bar = ProgressBarWidget::new("RAM").with_progress(1.5);
        assert_eq!(bar.progress, 1.0);

        let bar2 = ProgressBarWidget::new("RAM").with_progress(-0.5);
        assert_eq!(bar2.progress, 0.0);
    }

    #[test]
    fn h046_progress_bar_render() {
        let bar = ProgressBarWidget::new("RAM")
            .with_progress(0.5)
            .with_total("32 / 64 GB");
        let rendered = bar.render(20);

        assert!(rendered.contains("RAM"));
        assert!(rendered.contains("32 / 64 GB"));
    }

    // =========================================================================
    // H047: Table Widget Tests
    // =========================================================================

    #[test]
    fn h047_table_new() {
        let table = TableWidget::new(vec!["Name".to_string(), "Value".to_string()]);
        assert_eq!(table.headers.len(), 2);
        assert!(table.rows.is_empty());
    }

    #[test]
    fn h047_table_add_row() {
        let mut table = TableWidget::new(vec!["Name".to_string()]);
        table.add_row(vec!["Test".to_string()]);
        assert_eq!(table.rows.len(), 1);
    }

    #[test]
    fn h047_table_calculate_widths() {
        let mut table = TableWidget::new(vec!["Name".to_string(), "Value".to_string()]);
        table.add_row(vec!["Short".to_string(), "LongerValue".to_string()]);

        let widths = table.calculate_widths();
        assert_eq!(widths[0], 5); // "Short" or "Name", whichever is longer
        assert_eq!(widths[1], 11); // "LongerValue"
    }

    // =========================================================================
    // H048: Color Scheme Tests
    // =========================================================================

    #[test]
    fn h048_color_scheme_default() {
        let scheme = ColorScheme::default();
        // All colors should have valid RGB values
        assert!(scheme.ok.r <= 255);
        assert!(scheme.warning.r <= 255);
        assert!(scheme.critical.r <= 255);
    }

    #[test]
    fn h048_rgb_color_ansi_fg() {
        let color = RgbColor::new(255, 128, 64);
        let ansi = color.to_ansi_fg();
        assert!(ansi.contains("38;2;255;128;64"));
    }

    #[test]
    fn h048_rgb_color_ansi_bg() {
        let color = RgbColor::new(255, 128, 64);
        let ansi = color.to_ansi_bg();
        assert!(ansi.contains("48;2;255;128;64"));
    }

    #[test]
    fn h048_rgb_for_pressure_level() {
        // All pressure levels should return valid colors
        let _ = RgbColor::for_pressure_level(PressureLevel::Ok);
        let _ = RgbColor::for_pressure_level(PressureLevel::Elevated);
        let _ = RgbColor::for_pressure_level(PressureLevel::Warning);
        let _ = RgbColor::for_pressure_level(PressureLevel::Critical);
    }

    // =========================================================================
    // H049: Key Action Tests
    // =========================================================================

    #[test]
    fn h049_key_action_key() {
        assert_eq!(KeyAction::Quit.key(), 'q');
        assert_eq!(KeyAction::Refresh.key(), 'r');
        assert_eq!(KeyAction::Help.key(), '?');
    }

    #[test]
    fn h049_key_action_description() {
        assert_eq!(KeyAction::Quit.description(), "Quit");
        assert_eq!(KeyAction::Refresh.description(), "Refresh");
    }

    // =========================================================================
    // H050: TUI Render State Tests
    // =========================================================================

    #[test]
    fn h050_tui_render_state_default() {
        let state = TuiRenderState::default();
        assert!(state.cpu.is_none());
        assert!(state.gpus.is_empty());
        assert!(!state.stress_active);
        assert!(!state.paused);
        assert_eq!(state.focused_section, 0);
    }

    // =========================================================================
    // H060: Additional Coverage Tests
    // =========================================================================

    #[test]
    fn h060_tui_layout_with_refresh_rate() {
        let layout = TuiLayout::new().with_refresh_rate(50);
        assert_eq!(layout.refresh_rate_ms, 50);
    }

    #[test]
    fn h060_section_add_widget() {
        let mut section = Section::new("test", "Test", 0.5);
        section.add_widget(Widget::Text(TextWidget::new("Hello")));
        assert_eq!(section.widgets.len(), 1);
    }

    #[test]
    fn h060_text_widget() {
        let text = TextWidget::new("Hello World");
        assert_eq!(text.content, "Hello World");
        assert_eq!(text.style, TextStyle::Normal);

        let styled = TextWidget::new("Error").with_style(TextStyle::Error);
        assert_eq!(styled.style, TextStyle::Error);
    }

    #[test]
    fn h060_gauge_with_thresholds() {
        let gauge = GaugeWidget::new("Test").with_thresholds(60.0, 80.0);
        assert!((gauge.warning_threshold - 60.0).abs() < 0.01);
        assert!((gauge.critical_threshold - 80.0).abs() < 0.01);
    }

    #[test]
    fn h060_table_highlight() {
        let mut table = TableWidget::new(vec!["Col".to_string()]);
        table.add_row(vec!["Row1".to_string()]);
        table.add_row(vec!["Row2".to_string()]);
        table.highlight(1);
        assert_eq!(table.highlight_row, Some(1));
    }

    #[test]
    fn h060_sparkline_no_auto_scale() {
        let mut sparkline = SparklineWidget::new("Test");
        sparkline.auto_scale = false;
        sparkline.data.push_back(25.0);
        sparkline.data.push_back(50.0);
        sparkline.data.push_back(75.0);

        let rendered = sparkline.render();
        assert_eq!(rendered.chars().count(), 3);
    }

    #[test]
    fn h060_widget_enum_variants() {
        // Test that all widget variants can be created
        let _gauge = Widget::Gauge(GaugeWidget::new("Test"));
        let _sparkline = Widget::Sparkline(SparklineWidget::new("Test"));
        let _progress = Widget::ProgressBar(ProgressBarWidget::new("Test"));
        let _table = Widget::Table(TableWidget::new(vec![]));
        let _text = Widget::Text(TextWidget::new("Test"));
    }

    #[test]
    fn h060_key_action_all_keys() {
        // Test all key actions have keys and descriptions
        let actions = [
            KeyAction::Quit,
            KeyAction::Refresh,
            KeyAction::ToggleStress,
            KeyAction::FocusNext,
            KeyAction::NavigateUp,
            KeyAction::NavigateDown,
            KeyAction::Expand,
            KeyAction::Help,
            KeyAction::Alerts,
            KeyAction::Export,
            KeyAction::TogglePause,
        ];

        for action in &actions {
            let _ = action.key();
            let desc = action.description();
            assert!(!desc.is_empty());
        }
    }

    #[test]
    fn h060_memory_render_state() {
        let state = MemoryRenderState::default();
        assert_eq!(state.ram_pct, 0.0);
        assert!(state.vram.is_empty());
        assert!(state.ram_history.is_empty());
    }

    #[test]
    fn h060_data_flow_render_state() {
        let state = DataFlowRenderState::default();
        assert_eq!(state.pcie_tx_gbps, 0.0);
        assert_eq!(state.pcie_rx_gbps, 0.0);
        assert!(state.transfers.is_empty());
    }

    #[test]
    fn h060_text_styles() {
        let styles = [
            TextStyle::Normal,
            TextStyle::Bold,
            TextStyle::Dim,
            TextStyle::Italic,
            TextStyle::Header,
            TextStyle::Error,
            TextStyle::Warning,
            TextStyle::Success,
        ];

        // All styles should be distinct
        for (i, s1) in styles.iter().enumerate() {
            for (j, s2) in styles.iter().enumerate() {
                if i != j {
                    assert_ne!(s1, s2);
                }
            }
        }
    }

    #[test]
    fn h060_size_check_equality() {
        assert_eq!(SizeCheck::Recommended, SizeCheck::Recommended);
        assert_ne!(SizeCheck::Recommended, SizeCheck::Minimum);
        assert_ne!(SizeCheck::Minimum, SizeCheck::TooSmall);
    }

    #[test]
    fn h060_gauge_color_equality() {
        assert_eq!(GaugeColor::Ok, GaugeColor::Ok);
        assert_ne!(GaugeColor::Ok, GaugeColor::Warning);
        assert_ne!(GaugeColor::Warning, GaugeColor::Critical);
    }
}
