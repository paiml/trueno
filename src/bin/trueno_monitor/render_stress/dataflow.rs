//! Data flow tab rendering (PCIe bandwidth monitoring).

use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

use crate::App;

pub(crate) fn render_dataflow_tab(f: &mut Frame, app: &App, area: Rect) {
    let title = if app.stress_running && !app.gpu_workers.is_empty() {
        " Data Flow Monitor [STRESS: H2D/D2H ACTIVE] "
    } else {
        " Data Flow Monitor "
    };

    let block = Block::default().title(title).borders(Borders::ALL);

    if app.gpus.is_empty() {
        // No GPU - show message
        let text = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  No CUDA GPU Detected",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  To enable GPU monitoring:",
                Style::default().fg(Color::DarkGray),
            )),
            Line::from(""),
            Line::from("    1. Install NVIDIA CUDA drivers"),
            Line::from("    2. Build with: cargo build --features tui-monitor,cuda"),
            Line::from("    3. Run: cargo run --bin trueno-monitor --features tui-monitor,cuda"),
            Line::from(""),
            Line::from(Span::styled(
                "  PCIe bandwidth and transfer tracking require CUDA hardware.",
                Style::default().fg(Color::DarkGray),
            )),
        ];
        let paragraph = Paragraph::new(text).block(block);
        f.render_widget(paragraph, area);
    } else {
        // Real GPU data
        let mut lines = vec![
            Line::from(""),
            Line::from(Span::styled(
                "  PCIe Data Flow",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
        ];

        for (i, gpu) in app.gpus.iter().enumerate() {
            lines.push(Line::from(vec![
                Span::styled(format!("  GPU {}: ", i), Style::default().fg(Color::DarkGray)),
                Span::styled(&gpu.info.name, Style::default().fg(Color::Magenta)),
            ]));

            lines.push(Line::from(vec![
                Span::raw("    VRAM: "),
                Span::styled(
                    format!("{:.1} GB", gpu.vram_used_gb),
                    Style::default().fg(Color::Cyan),
                ),
                Span::raw(" / "),
                Span::styled(
                    format!("{:.1} GB", gpu.vram_total_gb),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" ("),
                Span::styled(
                    format!("{:.1}%", gpu.vram_percent),
                    Style::default().fg(if gpu.vram_percent > 80.0 {
                        Color::Red
                    } else {
                        Color::Green
                    }),
                ),
                Span::raw(")"),
            ]));

            // PCIe info
            if app.stress_running && !app.gpu_workers.is_empty() {
                let bandwidth_gbps =
                    (app.gpu_ops_per_sec as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0);
                lines.push(Line::from(vec![
                    Span::raw("    PCIe: "),
                    Span::styled(
                        "STRESS ACTIVE",
                        Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(" - "),
                    Span::styled(
                        format!("{:.2} GB/s", bandwidth_gbps),
                        Style::default().fg(Color::Cyan),
                    ),
                    Span::raw(" actual"),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::raw("    PCIe: "),
                    Span::styled("Gen4 x16", Style::default().fg(Color::Green)),
                    Span::raw(" ("),
                    Span::styled("31.5 GB/s", Style::default().fg(Color::Cyan)),
                    Span::raw(" theoretical)"),
                ]));
            }

            lines.push(Line::from(""));
        }

        lines.push(Line::from(Span::styled(
            "  Active Transfers",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));

        if app.stress_running && !app.gpu_workers.is_empty() {
            lines.push(Line::from(vec![
                Span::raw("    "),
                Span::styled("H2D: ", Style::default().fg(Color::Green)),
                Span::raw("1M x f32 (4MB) per iteration"),
            ]));
            lines.push(Line::from(vec![
                Span::raw("    "),
                Span::styled("D2H: ", Style::default().fg(Color::Yellow)),
                Span::raw("1M x f32 (4MB) per iteration"),
            ]));
            lines.push(Line::from(vec![
                Span::raw("    "),
                Span::styled(
                    format!(
                        "Throughput: {:.2} G elements/sec",
                        app.gpu_ops_per_sec as f64 / 1_000_000_000.0
                    ),
                    Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                ),
            ]));
        } else {
            lines.push(Line::from(Span::styled(
                "    No active kernel transfers",
                Style::default().fg(Color::DarkGray),
            )));
        }

        let paragraph = Paragraph::new(lines).block(block);
        f.render_widget(paragraph, area);
    }
}
