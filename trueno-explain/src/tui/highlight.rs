//! PTX syntax highlighting for the TUI source pane.
//!
//! Classifies PTX source lines by category (comments, directives,
//! labels, memory ops, arithmetic, control flow) and applies
//! appropriate terminal colors.

use ratatui::{
    style::{Color, Style},
    text::Span,
};

/// PTX instruction prefix to color category mapping.
const PTX_MEMORY_PREFIXES: &[&str] = &["ld.", "st."];
const PTX_ARITH_PREFIXES: &[&str] = &["add", "sub", "mul", "mad", "fma"];
const PTX_CONTROL_PREFIXES: &[&str] = &["bra", "ret", "setp"];

/// Classify a PTX instruction's syntax category for color highlighting.
fn ptx_instruction_color(trimmed: &str) -> Option<Color> {
    let categories: &[(&[&str], Color)] = &[
        (PTX_MEMORY_PREFIXES, Color::Yellow),
        (PTX_ARITH_PREFIXES, Color::Green),
        (PTX_CONTROL_PREFIXES, Color::Red),
    ];
    categories
        .iter()
        .find(|(prefixes, _)| prefixes.iter().any(|p| trimmed.starts_with(p)))
        .map(|(_, color)| *color)
}

/// Apply simple syntax highlighting to a single PTX source line.
pub(super) fn highlight_ptx_line(line: &str) -> Span<'static> {
    let line = line.to_string();
    let trimmed = line.trim();

    if trimmed.starts_with("//") {
        return Span::styled(line, Style::default().fg(Color::DarkGray));
    }
    if trimmed.starts_with('.') {
        return Span::styled(line, Style::default().fg(Color::Magenta));
    }
    if trimmed.ends_with(':') && !trimmed.contains(' ') {
        return Span::styled(line, Style::default().fg(Color::Cyan));
    }
    if line.starts_with('\t') || line.starts_with("    ") {
        if let Some(color) = ptx_instruction_color(trimmed) {
            return Span::styled(line, Style::default().fg(color));
        }
    }

    Span::raw(line)
}
