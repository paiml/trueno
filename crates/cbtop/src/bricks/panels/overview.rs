//! Overview panel brick (Layer 3)

use std::any::Any;
use presentar_core::{Canvas, Point, Rect, TextStyle, Widget};
use presentar_terminal::{BrailleGraph, GraphMode, Theme};
use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};

/// Overview panel brick
pub struct OverviewPanelBrick {
    pub cpu_data: Vec<f64>,
    pub gpu_data: Vec<f64>,
    pub cpu_avg: f64,
    pub gpu_avg: f64,
    pub frame_count: u64,
    pub problem_size: usize,
    pub theme: Theme,
}

impl OverviewPanelBrick {
    pub fn new() -> Self {
        Self {
            cpu_data: Vec::new(),
            gpu_data: Vec::new(),
            cpu_avg: 0.0,
            gpu_avg: 0.0,
            frame_count: 0,
            problem_size: 0,
            theme: Theme::tokyo_night(),
        }
    }

    pub fn paint(&self, canvas: &mut dyn Canvas, width: f32, _height: f32) {
        let label_style = TextStyle {
            color: self.theme.foreground,
            ..Default::default()
        };
        let dim_style = TextStyle {
            color: self.theme.dim,
            ..Default::default()
        };

        // CPU section
        canvas.draw_text("CPU Utilization", Point::new(2.0, 2.0), &label_style);
        if !self.cpu_data.is_empty() {
            let cpu_usage = self.cpu_data.last().copied().unwrap_or(0.0);
            let mut graph = BrailleGraph::new(self.cpu_data.clone())
                .with_color(self.theme.cpu_color(cpu_usage))
                .with_range(0.0, 100.0)
                .with_mode(GraphMode::Braille);
            graph.layout(Rect::new(2.0, 3.0, width / 2.0 - 4.0, 6.0));
            graph.paint(canvas);
        }

        // GPU section
        canvas.draw_text("GPU Utilization", Point::new(width / 2.0 + 2.0, 2.0), &label_style);
        if !self.gpu_data.is_empty() {
            let gpu_usage = self.gpu_data.last().copied().unwrap_or(0.0);
            let mut graph = BrailleGraph::new(self.gpu_data.clone())
                .with_color(self.theme.gpu_color(gpu_usage))
                .with_range(0.0, 100.0)
                .with_mode(GraphMode::Braille);
            graph.layout(Rect::new(width / 2.0 + 2.0, 3.0, width / 2.0 - 4.0, 6.0));
            graph.paint(canvas);
        }

        // Statistics
        canvas.draw_text("Statistics", Point::new(2.0, 10.0), &label_style);

        // Color the CPU/GPU avg values according to their usage
        let cpu_color_style = TextStyle {
            color: self.theme.cpu_color(self.cpu_avg),
            ..Default::default()
        };
        let gpu_color_style = TextStyle {
            color: self.theme.gpu_color(self.gpu_avg),
            ..Default::default()
        };

        canvas.draw_text("CPU Avg: ", Point::new(2.0, 11.0), &dim_style);
        canvas.draw_text(&format!("{:.1}%", self.cpu_avg), Point::new(12.0, 11.0), &cpu_color_style);
        canvas.draw_text("  GPU Avg: ", Point::new(18.0, 11.0), &dim_style);
        canvas.draw_text(&format!("{:.1}%", self.gpu_avg), Point::new(30.0, 11.0), &gpu_color_style);

        canvas.draw_text(
            &format!("Samples: {}  Problem Size: {}", self.frame_count, self.problem_size),
            Point::new(2.0, 12.0),
            &dim_style,
        );
    }
}

impl Brick for OverviewPanelBrick {
    fn brick_name(&self) -> &'static str {
        "overview_panel"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::MinWidth(40),
            BrickAssertion::MinHeight(15),
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