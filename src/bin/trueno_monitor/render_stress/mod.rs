//! Stress test, data flow, and help overlay rendering.

mod dataflow;
mod stress;

pub(crate) use dataflow::render_dataflow_tab;
pub(crate) use stress::render_stress_tab;

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

pub(crate) fn render_help_overlay(f: &mut Frame, size: Rect) {
    let block = Block::default()
        .title(" Help ")
        .borders(Borders::ALL)
        .style(Style::default().bg(Color::DarkGray));

    let area = centered_rect(50, 60, size);
    f.render_widget(ratatui::widgets::Clear, area);

    let help_text = vec![
        Line::from(""),
        Line::from(Span::styled(
            "Keyboard Controls",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from("  q        Quit"),
        Line::from("  Tab      Next tab"),
        Line::from("  Shift+Tab Previous tab"),
        Line::from("  s        Toggle stress test"),
        Line::from("  ?        Toggle this help"),
        Line::from(""),
        Line::from(Span::styled("Tabs", Style::default().add_modifier(Modifier::BOLD))),
        Line::from(""),
        Line::from("  Compute    CPU/GPU utilization"),
        Line::from("  Memory     RAM/SWAP/VRAM usage"),
        Line::from("  Data Flow  PCIe bandwidth"),
        Line::from("  Stress     Stress test controls"),
        Line::from(""),
        Line::from(Span::styled("Press ? to close", Style::default().fg(Color::DarkGray))),
    ];

    let paragraph = Paragraph::new(help_text).block(block);
    f.render_widget(paragraph, area);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
