//! Sidebar widget rendering for the TUI dashboard.
//!
//! Contains render functions for registers, memory, roofline, bug hunting,
//! and muda (waste) warning panels.

use crate::analyzer::MudaType;
use crate::ptx::BugSeverity;
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem},
    Frame,
};

use super::TuiApp;

pub(super) fn render_sidebar(frame: &mut Frame<'_>, app: &TuiApp, area: Rect) {
    // Split sidebar into sections
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8), // Registers
            Constraint::Length(6), // Memory
            Constraint::Length(5), // Roofline
            Constraint::Length(6), // Bug hunting
            Constraint::Min(4),    // Warnings
        ])
        .split(area);

    // Register usage
    render_register_widget(frame, app, chunks[0]);

    // Memory patterns
    render_memory_widget(frame, app, chunks[1]);

    // Roofline
    render_roofline_widget(frame, app, chunks[2]);

    // PTX pattern analysis results
    render_bugs_widget(frame, app, chunks[3]);

    // Muda warnings
    render_warnings_widget(frame, app, chunks[4]);
}

fn render_register_widget(frame: &mut Frame<'_>, app: &TuiApp, area: Rect) {
    let regs = &app.report.registers;
    let total = regs.total();
    let occupancy = app.report.estimated_occupancy;

    let status_color = if total < 64 {
        Color::Green
    } else if total < 128 {
        Color::Yellow
    } else {
        Color::Red
    };

    let items = vec![
        ListItem::new(format!(".f32: {:3} / 255", regs.f32_regs)),
        ListItem::new(format!(".b32: {:3} / 255", regs.b32_regs)),
        ListItem::new(format!(".b64: {:3} / 255", regs.b64_regs)),
        ListItem::new(format!(".pred: {:2} / 8", regs.pred_regs)), // PTX has p0-p7
        ListItem::new(Line::from(vec![
            Span::raw(format!("Total: {} → ", total)),
            Span::styled(
                format!("{:.0}% occ", occupancy * 100.0),
                Style::default().fg(status_color).add_modifier(Modifier::BOLD),
            ),
        ])),
    ];

    let block = Block::default()
        .title(" Registers ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(status_color));

    let list = List::new(items).block(block);
    frame.render_widget(list, area);
}

fn render_memory_widget(frame: &mut Frame<'_>, app: &TuiApp, area: Rect) {
    let mem = &app.report.memory;
    let coal_pct = mem.coalesced_ratio * 100.0;

    let status_color = if coal_pct >= 90.0 {
        Color::Green
    } else if coal_pct >= 70.0 {
        Color::Yellow
    } else {
        Color::Red
    };

    let items = vec![
        ListItem::new(format!("Global ld: {}", mem.global_loads)),
        ListItem::new(format!("Global st: {}", mem.global_stores)),
        ListItem::new(Line::from(vec![
            Span::raw("Coalesced: "),
            Span::styled(
                format!("{:.1}%", coal_pct),
                Style::default().fg(status_color).add_modifier(Modifier::BOLD),
            ),
        ])),
    ];

    let block = Block::default()
        .title(" Memory ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(status_color));

    let list = List::new(items).block(block);
    frame.render_widget(list, area);
}

fn render_roofline_widget(frame: &mut Frame<'_>, app: &TuiApp, area: Rect) {
    let roof = &app.report.roofline;

    let bound_text = if roof.memory_bound {
        Span::styled("Memory-bound", Style::default().fg(Color::Yellow))
    } else {
        Span::styled("Compute-bound", Style::default().fg(Color::Green))
    };

    let items = vec![
        ListItem::new(format!("AI: {:.2} FLOP/B", roof.arithmetic_intensity)),
        ListItem::new(Line::from(vec![Span::raw("Bottleneck: "), bound_text])),
    ];

    let block = Block::default()
        .title(" Roofline ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Blue));

    let list = List::new(items).block(block);
    frame.render_widget(list, area);
}

fn render_bugs_widget(frame: &mut Frame<'_>, app: &TuiApp, area: Rect) {
    let bug_report = &app.bug_report;
    let critical = bug_report.count_by_severity(BugSeverity::Critical);
    let high = bug_report.count_by_severity(BugSeverity::High);
    let medium = bug_report.count_by_severity(BugSeverity::Medium);

    let status_color = if critical > 0 {
        Color::Red
    } else if high > 0 {
        Color::Yellow
    } else {
        Color::Green
    };

    let items = if bug_report.bugs.is_empty() {
        vec![ListItem::new(Line::from(vec![
            Span::styled("✓ ", Style::default().fg(Color::Green)),
            Span::raw("No bugs detected"),
        ]))]
    } else {
        vec![
            ListItem::new(Line::from(vec![Span::styled(
                format!("P0 Critical: {}", critical),
                Style::default().fg(if critical > 0 { Color::Red } else { Color::Green }),
            )])),
            ListItem::new(Line::from(vec![Span::styled(
                format!("P1 High: {}", high),
                Style::default().fg(if high > 0 { Color::Yellow } else { Color::Green }),
            )])),
            ListItem::new(Line::from(vec![Span::styled(
                format!("P2 Medium: {}", medium),
                Style::default().fg(if medium > 0 { Color::Blue } else { Color::Green }),
            )])),
        ]
    };

    let block = Block::default()
        .title(" Bug Hunt ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(status_color));

    let list = List::new(items).block(block);
    frame.render_widget(list, area);
}

fn render_warnings_widget(frame: &mut Frame<'_>, app: &TuiApp, area: Rect) {
    let items: Vec<ListItem<'_>> = if app.report.warnings.is_empty() {
        vec![ListItem::new(Line::from(vec![
            Span::styled("✓ ", Style::default().fg(Color::Green)),
            Span::raw("No Muda detected"),
        ]))]
    } else {
        app.report
            .warnings
            .iter()
            .map(|w| {
                let icon = match w.muda_type {
                    MudaType::Transport => ("⚠ ", Color::Yellow),
                    MudaType::Waiting => ("⏳", Color::Yellow),
                    MudaType::Overprocessing => ("🔄", Color::Yellow),
                };
                ListItem::new(Line::from(vec![
                    Span::styled(icon.0, Style::default().fg(icon.1)),
                    Span::raw(&w.description),
                ]))
            })
            .collect()
    };

    let border_color = if app.report.warnings.is_empty() { Color::Green } else { Color::Yellow };

    let block = Block::default()
        .title(" Muda (Waste) ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color));

    let list = List::new(items).block(block);
    frame.render_widget(list, area);
}
