//! Interactive TUI mode for trueno-explain
//!
//! Implements Genchi Genbutsu (Go and See) through interactive exploration.
//!
//! Layout:
//! - Left Pane: Source/PTX code with syntax highlighting
//! - Right Pane: Analysis dashboard (registers, memory, warnings, bugs)
//! - Bottom: Status bar with keybindings

mod highlight;
mod widgets;

use crate::analyzer::AnalysisReport;
use crate::ptx::{PtxBugAnalyzer, PtxBugReport};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{
        Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
    },
    Frame, Terminal,
};
use std::io::{self, Stdout};

use highlight::highlight_ptx_line;
use widgets::render_sidebar;

/// TUI application state
pub struct TuiApp {
    /// The PTX source code to display
    pub ptx_source: String,
    /// Analysis report
    pub report: AnalysisReport,
    /// Bug hunting report (probar-style)
    pub bug_report: PtxBugReport,
    /// Current scroll position in source pane
    pub source_scroll: u16,
    /// Whether sidebar is visible
    pub sidebar_visible: bool,
    /// Should quit
    pub should_quit: bool,
    /// Total lines in source
    source_lines: usize,
}

impl TuiApp {
    /// Create a new TUI application
    #[must_use]
    pub fn new(ptx_source: String, report: AnalysisReport) -> Self {
        let source_lines = ptx_source.lines().count();
        // Perform PTX pattern analysis in strict mode for TUI
        let bug_report = PtxBugAnalyzer::strict().analyze(&ptx_source);
        Self {
            ptx_source,
            report,
            bug_report,
            source_scroll: 0,
            sidebar_visible: true,
            should_quit: false,
            source_lines,
        }
    }

    /// Handle keyboard input
    pub fn handle_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Char('s') => self.sidebar_visible = !self.sidebar_visible,
            KeyCode::Down | KeyCode::Char('j') => self.scroll_down(),
            KeyCode::Up | KeyCode::Char('k') => self.scroll_up(),
            KeyCode::PageDown => self.page_down(),
            KeyCode::PageUp => self.page_up(),
            KeyCode::Home => self.source_scroll = 0,
            KeyCode::End => self.scroll_to_end(),
            _ => {}
        }
    }

    fn scroll_down(&mut self) {
        if (self.source_scroll as usize) < self.source_lines.saturating_sub(1) {
            self.source_scroll = self.source_scroll.saturating_add(1);
        }
    }

    fn scroll_up(&mut self) {
        self.source_scroll = self.source_scroll.saturating_sub(1);
    }

    fn page_down(&mut self) {
        self.source_scroll = self
            .source_scroll
            .saturating_add(20)
            .min(self.source_lines.saturating_sub(1) as u16);
    }

    fn page_up(&mut self) {
        self.source_scroll = self.source_scroll.saturating_sub(20);
    }

    fn scroll_to_end(&mut self) {
        self.source_scroll = self.source_lines.saturating_sub(1) as u16;
    }
}

/// Run the TUI application
///
/// # Errors
///
/// Returns `io::Error` if terminal operations fail.
pub fn run_tui(ptx_source: String, report: AnalysisReport) -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = TuiApp::new(ptx_source, report);

    // Main loop
    let result = run_app(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    result
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &mut TuiApp) -> io::Result<()> {
    loop {
        terminal.draw(|f| ui(f, app))?;

        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press {
                app.handle_key(key.code);
            }
        }

        if app.should_quit {
            return Ok(());
        }
    }
}

fn ui(frame: &mut Frame<'_>, app: &TuiApp) {
    let size = frame.area();

    // Main layout: source pane + optional sidebar
    let main_chunks = if app.sidebar_visible {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(size)
    } else {
        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(100)])
            .split(size)
    };

    // Render source pane
    render_source_pane(frame, app, main_chunks[0]);

    // Render sidebar if visible
    if app.sidebar_visible && main_chunks.len() > 1 {
        render_sidebar(frame, app, main_chunks[1]);
    }
}

fn render_source_pane(frame: &mut Frame<'_>, app: &TuiApp, area: Rect) {
    // Split into source area and status bar
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(3)])
        .split(area);

    // Source code with syntax highlighting
    let lines: Vec<Line<'_>> = app
        .ptx_source
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let line_num = format!("{:4} ", i + 1);
            let highlighted = highlight_ptx_line(line);
            Line::from(vec![
                Span::styled(line_num, Style::default().fg(Color::DarkGray)),
                highlighted,
            ])
        })
        .collect();

    let source_block = Block::default()
        .title(format!(" PTX: {} ", app.report.name))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let paragraph = Paragraph::new(lines)
        .block(source_block)
        .scroll((app.source_scroll, 0));

    frame.render_widget(paragraph, chunks[0]);

    // Scrollbar
    let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
        .begin_symbol(Some("↑"))
        .end_symbol(Some("↓"));
    let mut scrollbar_state =
        ScrollbarState::new(app.source_lines).position(app.source_scroll as usize);
    frame.render_stateful_widget(scrollbar, chunks[0], &mut scrollbar_state);

    // Status bar
    let status = Line::from(vec![
        Span::styled(
            " q",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(":Quit "),
        Span::styled(
            "s",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(":Sidebar "),
        Span::styled(
            "↑↓",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(":Scroll "),
        Span::styled(
            "PgUp/Dn",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(":Page "),
    ]);

    let status_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let status_para = Paragraph::new(status).block(status_block);
    frame.render_widget(status_para, chunks[1]);
}

#[cfg(test)]
mod tests;
