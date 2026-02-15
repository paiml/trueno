//! PTX syntax highlighting for the TUI source pane.
//!
//! Classifies PTX source lines by category (comments, directives,
//! labels, memory ops, arithmetic, control flow) and applies
//! appropriate terminal colors.

use ratatui::{
    style::{Color, Style},
    text::Span,
};

/// Classify a PTX instruction's syntax category for color highlighting.
fn ptx_instruction_color(trimmed: &str) -> Option<Color> {
    // Memory operations
    if trimmed.starts_with("ld.") || trimmed.starts_with("st.") {
        return Some(Color::Yellow);
    }
    // Arithmetic
    if trimmed.starts_with("add")
        || trimmed.starts_with("sub")
        || trimmed.starts_with("mul")
        || trimmed.starts_with("mad")
        || trimmed.starts_with("fma")
    {
        return Some(Color::Green);
    }
    // Control flow
    if trimmed.starts_with("bra") || trimmed.starts_with("ret") || trimmed.starts_with("setp") {
        return Some(Color::Red);
    }
    None
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
