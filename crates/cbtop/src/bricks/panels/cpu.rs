//! CPU panel brick (Layer 3)

use std::any::Any;
use presentar_core::{Canvas, Point, Rect, TextStyle, Widget};
use presentar_terminal::{BrailleGraph, GraphMode, Meter, Theme};
use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};

pub struct CpuPanelBrick {
    pub cpu_data: Vec<f64>,
    pub intensity: f64,
    pub theme: Theme,
}

impl CpuPanelBrick {
    pub fn new() -> Self {
        Self {
            cpu_data: Vec::new(),
            intensity: 0.0,
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

        canvas.draw_text("CPU Monitor", Point::new(2.0, 2.0), &label_style);

        // Main graph
        if !self.cpu_data.is_empty() {
            let cpu_usage = self.cpu_data.last().copied().unwrap_or(0.0);
            let mut graph = BrailleGraph::new(self.cpu_data.clone())
                .with_color(self.theme.cpu_color(cpu_usage))
                .with_range(0.0, 100.0)
                .with_mode(GraphMode::Braille);
            graph.layout(Rect::new(2.0, 3.0, width - 4.0, 8.0));
            graph.paint(canvas);
        }

        // Simulated per-core meters
        canvas.draw_text("Per-Core Usage (simulated)", Point::new(2.0, 12.0), &label_style);
        for i in 0..8 {
            let y = 13.0 + i as f32;
            let usage = 30.0 + (i as f64 * 7.0) + (self.intensity * 50.0);
            let usage = usage.min(100.0);

            canvas.draw_text(&format!("Core {}: ", i), Point::new(2.0, y), &dim_style);

            // Use theme gradient for per-core meter color
            let mut meter = Meter::new(usage, 100.0)
                .with_color(self.theme.cpu_color(usage));
            meter.layout(Rect::new(12.0, y, 20.0, 1.0));
            meter.paint(canvas);

            // Color the percentage based on usage
            let pct_style = TextStyle {
                color: self.theme.cpu_color(usage),
                ..Default::default()
            };
            canvas.draw_text(&format!("{:5.1}%", usage), Point::new(34.0, y), &pct_style);
        }
    }
}

impl Brick for CpuPanelBrick {
    fn brick_name(&self) -> &'static str {
        "cpu_panel"
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