//! Load control panel brick (Layer 3)
//!
//! Interactive controls for load testing - start/stop, backend selection,
//! workload type, intensity slider, and real-time status display.

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickGrade, BrickScore, BrickVerification};
use presentar_core::{Canvas, Color, Point, Rect, TextStyle, Widget};
use presentar_terminal::{Meter, Theme};
use std::any::Any;

/// Compute backend options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeBackend {
    #[default]
    Auto,
    CpuScalar,
    CpuSimd,
    GpuCuda,
    GpuWgpu,
}

impl ComputeBackend {
    /// Display name for the backend
    pub fn name(&self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::CpuScalar => "CPU (Scalar)",
            Self::CpuSimd => "CPU (SIMD)",
            Self::GpuCuda => "GPU (CUDA)",
            Self::GpuWgpu => "GPU (wgpu)",
        }
    }

    /// All available backends
    pub const ALL: [ComputeBackend; 5] = [
        Self::Auto,
        Self::CpuScalar,
        Self::CpuSimd,
        Self::GpuCuda,
        Self::GpuWgpu,
    ];
}

/// Workload type for load testing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkloadType {
    #[default]
    Gemm,
    Softmax,
    LayerNorm,
    Attention,
    Lz4Compress,
    Mixed,
}

impl WorkloadType {
    /// Display name for the workload
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gemm => "GEMM (Matrix Multiply)",
            Self::Softmax => "Softmax",
            Self::LayerNorm => "Layer Normalization",
            Self::Attention => "Attention",
            Self::Lz4Compress => "LZ4 Compression",
            Self::Mixed => "Mixed Workload",
        }
    }

    /// Short name for compact display
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::Gemm => "GEMM",
            Self::Softmax => "Softmax",
            Self::LayerNorm => "LayerNorm",
            Self::Attention => "Attention",
            Self::Lz4Compress => "LZ4",
            Self::Mixed => "Mixed",
        }
    }

    /// All available workloads
    pub const ALL: [WorkloadType; 6] = [
        Self::Gemm,
        Self::Softmax,
        Self::LayerNorm,
        Self::Attention,
        Self::Lz4Compress,
        Self::Mixed,
    ];
}

/// Load test run statistics
#[derive(Debug, Clone, Default)]
pub struct LoadStats {
    /// Iterations completed
    pub iterations: u64,
    /// Total elapsed time in milliseconds
    pub elapsed_ms: u64,
    /// Operations per second
    pub ops_per_sec: f64,
    /// Current throughput in GB/s
    pub throughput_gbs: f64,
    /// Average latency in microseconds
    pub avg_latency_us: f64,
    /// P99 latency in microseconds
    pub p99_latency_us: f64,
}

/// Load control panel for interactive load testing
pub struct LoadControlPanelBrick {
    /// Selected compute backend
    pub backend: ComputeBackend,
    /// Selected workload type
    pub workload: WorkloadType,
    /// Intensity level (0.0 to 100.0)
    pub intensity: f64,
    /// Problem size (affects memory usage)
    pub problem_size: usize,
    /// Whether load test is running
    pub is_running: bool,
    /// Current load statistics
    pub stats: LoadStats,
    /// Error message if any
    pub error: Option<String>,
    /// Selected menu item (0=backend, 1=workload, 2=intensity, 3=size, 4=start/stop)
    pub selected_item: usize,
    /// Theme for rendering
    pub theme: Theme,
    /// Current ComputeBrick quality score (§29.8)
    pub brick_score: Option<BrickScore>,
    /// Current GFLOP/s throughput
    pub gflops: f64,
}

impl LoadControlPanelBrick {
    /// Create a new load control panel
    pub fn new() -> Self {
        Self {
            backend: ComputeBackend::default(),
            workload: WorkloadType::default(),
            intensity: 50.0,
            problem_size: 1024,
            is_running: false,
            stats: LoadStats::default(),
            error: None,
            selected_item: 0,
            theme: Theme::tokyo_night(),
            brick_score: None,
            gflops: 0.0,
        }
    }

    /// Cycle to next backend
    pub fn next_backend(&mut self) {
        let idx = ComputeBackend::ALL
            .iter()
            .position(|&b| b == self.backend)
            .unwrap_or(0);
        self.backend = ComputeBackend::ALL[(idx + 1) % ComputeBackend::ALL.len()];
    }

    /// Cycle to previous backend
    pub fn prev_backend(&mut self) {
        let idx = ComputeBackend::ALL
            .iter()
            .position(|&b| b == self.backend)
            .unwrap_or(0);
        self.backend =
            ComputeBackend::ALL[(idx + ComputeBackend::ALL.len() - 1) % ComputeBackend::ALL.len()];
    }

    /// Cycle to next workload
    pub fn next_workload(&mut self) {
        let idx = WorkloadType::ALL
            .iter()
            .position(|&w| w == self.workload)
            .unwrap_or(0);
        self.workload = WorkloadType::ALL[(idx + 1) % WorkloadType::ALL.len()];
    }

    /// Cycle to previous workload
    pub fn prev_workload(&mut self) {
        let idx = WorkloadType::ALL
            .iter()
            .position(|&w| w == self.workload)
            .unwrap_or(0);
        self.workload =
            WorkloadType::ALL[(idx + WorkloadType::ALL.len() - 1) % WorkloadType::ALL.len()];
    }

    /// Increase intensity
    pub fn increase_intensity(&mut self) {
        self.intensity = (self.intensity + 5.0).min(100.0);
    }

    /// Decrease intensity
    pub fn decrease_intensity(&mut self) {
        self.intensity = (self.intensity - 5.0).max(0.0);
    }

    /// Increase problem size
    pub fn increase_size(&mut self) {
        self.problem_size = (self.problem_size * 2).min(65536);
    }

    /// Decrease problem size
    pub fn decrease_size(&mut self) {
        self.problem_size = (self.problem_size / 2).max(64);
    }

    /// Toggle running state
    pub fn toggle_running(&mut self) {
        self.is_running = !self.is_running;
        if self.is_running {
            self.stats = LoadStats::default();
            self.error = None;
        }
    }

    /// Update statistics from load generator
    pub fn update_stats(&mut self, stats: LoadStats) {
        self.stats = stats;
    }

    /// Update ComputeBrick score (§29.8)
    pub fn update_score(&mut self, score: BrickScore, gflops: f64) {
        self.brick_score = Some(score);
        self.gflops = gflops;
    }

    /// Render a score bar component
    fn render_score_bar(value: u8, max: u8, width: usize) -> String {
        BrickScore::render_bar(value, max, width)
    }

    /// Set error message
    pub fn set_error(&mut self, error: String) {
        self.error = Some(error);
        self.is_running = false;
    }

    /// Format problem size for display
    fn format_size(size: usize) -> String {
        if size >= 1024 {
            format!("{}K", size / 1024)
        } else {
            format!("{}", size)
        }
    }

    /// Paint the load control panel
    pub fn paint(&self, canvas: &mut dyn Canvas, width: f32, _height: f32) {
        let label_style = TextStyle {
            color: self.theme.foreground,
            ..Default::default()
        };
        let dim_style = TextStyle {
            color: self.theme.dim,
            ..Default::default()
        };
        let selected_style = TextStyle {
            color: Color::new(0.3, 0.8, 1.0, 1.0), // Cyan for selected
            ..Default::default()
        };
        let running_style = TextStyle {
            color: Color::new(0.3, 1.0, 0.5, 1.0), // Green for running
            ..Default::default()
        };
        let stopped_style = TextStyle {
            color: Color::new(1.0, 0.5, 0.3, 1.0), // Orange for stopped
            ..Default::default()
        };
        let error_style = TextStyle {
            color: Color::new(1.0, 0.2, 0.2, 1.0), // Red for errors
            ..Default::default()
        };

        canvas.draw_text("Load Control", Point::new(2.0, 2.0), &label_style);

        // Status indicator
        let status_text = if self.is_running {
            "RUNNING"
        } else {
            "STOPPED"
        };
        let status_style = if self.is_running {
            &running_style
        } else {
            &stopped_style
        };
        canvas.draw_text(status_text, Point::new(width - 12.0, 2.0), status_style);

        // Backend selection
        let backend_style = if self.selected_item == 0 {
            &selected_style
        } else {
            &dim_style
        };
        canvas.draw_text("Backend:", Point::new(2.0, 4.0), backend_style);
        canvas.draw_text(
            &format!("< {} >", self.backend.name()),
            Point::new(12.0, 4.0),
            &label_style,
        );

        // Workload selection
        let workload_style = if self.selected_item == 1 {
            &selected_style
        } else {
            &dim_style
        };
        canvas.draw_text("Workload:", Point::new(2.0, 5.0), workload_style);
        canvas.draw_text(
            &format!("< {} >", self.workload.short_name()),
            Point::new(12.0, 5.0),
            &label_style,
        );

        // Intensity slider
        let intensity_style = if self.selected_item == 2 {
            &selected_style
        } else {
            &dim_style
        };
        canvas.draw_text("Intensity:", Point::new(2.0, 6.0), intensity_style);
        let intensity_color = self.theme.cpu_color(self.intensity);
        let mut intensity_meter = Meter::new(self.intensity, 100.0).with_color(intensity_color);
        intensity_meter.layout(Rect::new(12.0, 6.0, width - 30.0, 1.0));
        intensity_meter.paint(canvas);
        canvas.draw_text(
            &format!("{:.0}%", self.intensity),
            Point::new(width - 16.0, 6.0),
            &label_style,
        );

        // Problem size
        let size_style = if self.selected_item == 3 {
            &selected_style
        } else {
            &dim_style
        };
        canvas.draw_text("Size:", Point::new(2.0, 7.0), size_style);
        canvas.draw_text(
            &format!("< {} >", Self::format_size(self.problem_size)),
            Point::new(12.0, 7.0),
            &label_style,
        );

        // Start/Stop button indicator
        let button_style = if self.selected_item == 4 {
            &selected_style
        } else {
            &dim_style
        };
        let button_text = if self.is_running { "[STOP]" } else { "[START]" };
        canvas.draw_text(button_text, Point::new(2.0, 9.0), button_style);

        // Error display
        if let Some(ref err) = self.error {
            canvas.draw_text("Error:", Point::new(2.0, 11.0), &error_style);
            canvas.draw_text(err, Point::new(9.0, 11.0), &error_style);
        }

        // ComputeBrick Score section (§29.8)
        if let Some(ref score) = self.brick_score {
            let grade = score.grade();
            let grade_color = match grade {
                BrickGrade::A | BrickGrade::B => Color::new(0.3, 1.0, 0.5, 1.0), // Green
                BrickGrade::C => Color::new(1.0, 0.8, 0.3, 1.0),                 // Yellow
                BrickGrade::D | BrickGrade::F => Color::new(1.0, 0.3, 0.3, 1.0), // Red
            };
            let grade_style = TextStyle {
                color: grade_color,
                ..Default::default()
            };

            canvas.draw_text("ComputeBrick Score", Point::new(2.0, 13.0), &label_style);
            canvas.draw_text(
                &format!("{}/100 ({})", score.total(), grade.letter()),
                Point::new(22.0, 13.0),
                &grade_style,
            );
            canvas.draw_text(
                &format!("{:.2} GFLOP/s", self.gflops),
                Point::new(width - 18.0, 13.0),
                &label_style,
            );

            // Score breakdown bars
            let bar_width = 20;
            canvas.draw_text("Performance:", Point::new(2.0, 15.0), &dim_style);
            canvas.draw_text(
                &Self::render_score_bar(score.performance, 40, bar_width),
                Point::new(14.0, 15.0),
                &label_style,
            );
            canvas.draw_text(
                &format!("{}/40", score.performance),
                Point::new(36.0, 15.0),
                &dim_style,
            );

            canvas.draw_text("Efficiency:", Point::new(2.0, 16.0), &dim_style);
            canvas.draw_text(
                &Self::render_score_bar(score.efficiency, 25, bar_width),
                Point::new(14.0, 16.0),
                &label_style,
            );
            canvas.draw_text(
                &format!("{}/25", score.efficiency),
                Point::new(36.0, 16.0),
                &dim_style,
            );

            canvas.draw_text("Correctness:", Point::new(2.0, 17.0), &dim_style);
            canvas.draw_text(
                &Self::render_score_bar(score.correctness, 20, bar_width),
                Point::new(14.0, 17.0),
                &label_style,
            );
            canvas.draw_text(
                &format!("{}/20", score.correctness),
                Point::new(36.0, 17.0),
                &dim_style,
            );

            canvas.draw_text("Stability:", Point::new(2.0, 18.0), &dim_style);
            canvas.draw_text(
                &Self::render_score_bar(score.stability, 15, bar_width),
                Point::new(14.0, 18.0),
                &label_style,
            );
            canvas.draw_text(
                &format!("{}/15", score.stability),
                Point::new(36.0, 18.0),
                &dim_style,
            );
        } else {
            // Statistics section (fallback when no score available)
            canvas.draw_text("Statistics", Point::new(2.0, 13.0), &label_style);

            canvas.draw_text("Iterations:", Point::new(2.0, 14.0), &dim_style);
            canvas.draw_text(
                &format!("{}", self.stats.iterations),
                Point::new(14.0, 14.0),
                &label_style,
            );

            canvas.draw_text("Ops/sec:", Point::new(2.0, 15.0), &dim_style);
            canvas.draw_text(
                &format!("{:.1}", self.stats.ops_per_sec),
                Point::new(14.0, 15.0),
                &label_style,
            );

            canvas.draw_text("Throughput:", Point::new(2.0, 16.0), &dim_style);
            canvas.draw_text(
                &format!("{:.2} GB/s", self.stats.throughput_gbs),
                Point::new(14.0, 16.0),
                &label_style,
            );

            canvas.draw_text("Avg Latency:", Point::new(2.0, 17.0), &dim_style);
            canvas.draw_text(
                &format!("{:.1} us", self.stats.avg_latency_us),
                Point::new(14.0, 17.0),
                &label_style,
            );

            canvas.draw_text("P99 Latency:", Point::new(2.0, 18.0), &dim_style);
            canvas.draw_text(
                &format!("{:.1} us", self.stats.p99_latency_us),
                Point::new(14.0, 18.0),
                &label_style,
            );
        }

        // Help text
        canvas.draw_text(
            "Use arrow keys to navigate, Enter to toggle",
            Point::new(2.0, 20.0),
            &dim_style,
        );
    }

    /// Navigate to next menu item
    pub fn next_item(&mut self) {
        self.selected_item = (self.selected_item + 1) % 5;
    }

    /// Navigate to previous menu item
    pub fn prev_item(&mut self) {
        self.selected_item = (self.selected_item + 4) % 5;
    }

    /// Handle left key based on selected item
    pub fn handle_left(&mut self) {
        match self.selected_item {
            0 => self.prev_backend(),
            1 => self.prev_workload(),
            2 => self.decrease_intensity(),
            3 => self.decrease_size(),
            4 => {} // No left action for button
            _ => {}
        }
    }

    /// Handle right key based on selected item
    pub fn handle_right(&mut self) {
        match self.selected_item {
            0 => self.next_backend(),
            1 => self.next_workload(),
            2 => self.increase_intensity(),
            3 => self.increase_size(),
            4 => {} // No right action for button
            _ => {}
        }
    }

    /// Handle enter key
    pub fn handle_enter(&mut self) {
        if self.selected_item == 4 {
            self.toggle_running();
        }
    }
}

impl Default for LoadControlPanelBrick {
    fn default() -> Self {
        Self::new()
    }
}

impl Brick for LoadControlPanelBrick {
    fn brick_name(&self) -> &'static str {
        "load_control_panel"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::MinWidth(50),
            BrickAssertion::MinHeight(20),
            BrickAssertion::max_latency_ms(8),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::FRAME_60FPS
    }

    fn verify(&self) -> BrickVerification {
        let mut v = BrickVerification::new();
        for assertion in self.assertions() {
            v.check(&assertion);
        }
        v
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}


#[cfg(test)]
mod tests;
