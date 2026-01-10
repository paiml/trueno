//! Help panel brick (Layer 3)

use std::any::Any;
use presentar_core::{Canvas, Point, TextStyle};
use presentar_terminal::Theme;
use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};

pub struct HelpPanelBrick {
    pub theme: Theme,
}

impl HelpPanelBrick {
    pub fn new() -> Self {
        Self {
            theme: Theme::tokyo_night(),
        }
    }

    pub fn paint(&self, canvas: &mut dyn Canvas, _width: f32, _height: f32) {
        let title_style = TextStyle {
            color: self.theme.foreground,
            ..Default::default()
        };
        let key_style = TextStyle {
            color: self.theme.cpu.sample(0.3), // Use gradient for accent color
            ..Default::default()
        };
        let desc_style = TextStyle {
            color: self.theme.dim,
            ..Default::default()
        };

        canvas.draw_text("Help - Keyboard Controls", Point::new(2.0, 2.0), &title_style);

        let controls = [
            ("1-9", "Switch panels"),
            ("Space", "Start/Stop load generation"),
            ("+/-", "Increase/Decrease intensity"),
            ("b", "Cycle backend (SIMD/wgpu/CUDA/All)"),
            ("w", "Cycle workload type"),
            ("[/]", "Decrease/Increase problem size"),
            ("r", "Reset statistics"),
            ("q/Esc", "Quit"),
        ];

        for (i, (key, desc)) in controls.iter().enumerate() {
            let y = 4.0 + i as f32;
            canvas.draw_text(&format!("{:>8}", key), Point::new(4.0, y), &key_style);
            canvas.draw_text(&format!("  {}", desc), Point::new(12.0, y), &desc_style);
        }
    }
}

impl Brick for HelpPanelBrick {
    fn brick_name(&self) -> &'static str {
        "help_panel"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::MinWidth(30),
            BrickAssertion::MinHeight(10),
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