//! Main UI rendering: tabs, compute tab, memory tab.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline, Tabs},
    Frame,
};
use trueno_gpu::monitor::{ComputeDevice, PressureLevel};

use super::render_stress;
use super::{App, GpuState};

/// Map a percentage to a traffic-light color (green/yellow/red).
fn pct_color(pct: f64) -> Color {
    if pct > 90.0 {
        Color::Red
    } else if pct > 70.0 {
        Color::Yellow
    } else {
        Color::Green
    }
}

/// Render a stress-mode banner. Returns the next layout index.
fn render_stress_banner(f: &mut Frame, text: String, chunks: &[Rect], idx: usize) -> usize {
    let banner = Paragraph::new(text)
        .style(Style::default().fg(Color::Black).bg(Color::Yellow).add_modifier(Modifier::BOLD));
    f.render_widget(banner, chunks[idx]);
    idx + 1
}

/// Render a single GPU VRAM gauge. Returns the next layout index.
fn render_gpu_gauge(
    f: &mut Frame,
    gpu: &GpuState,
    gpu_idx: usize,
    stress_active: bool,
    chunks: &[Rect],
    layout_idx: usize,
) -> usize {
    let gpu_color = pct_color(gpu.vram_percent);
    let title = if stress_active {
        format!(" GPU {} [STRESS] {} ", gpu_idx, gpu.info.name)
    } else {
        format!(" GPU {} VRAM: {} ", gpu_idx, gpu.info.name)
    };
    let label =
        format!("{:.1} / {:.1} GB ({:.1}%)", gpu.vram_used_gb, gpu.vram_total_gb, gpu.vram_percent);
    let gauge = Gauge::default()
        .block(Block::default().title(title).borders(Borders::ALL))
        .gauge_style(Style::default().fg(gpu_color))
        .percent(gpu.vram_percent as u16)
        .label(label);
    f.render_widget(gauge, chunks[layout_idx]);
    layout_idx + 1
}

pub(crate) fn ui(f: &mut Frame, app: &App) {
    let size = f.area();

    // Main layout: header, content, footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Content
            Constraint::Length(3), // Footer
        ])
        .split(size);

    // Header with tabs
    let gpu_count = app.gpus.len();
    let title = if gpu_count > 0 {
        format!(" TRUENO Monitor v0.10.1 | {} GPU(s) ", gpu_count)
    } else {
        " TRUENO Monitor v0.10.1 | No CUDA GPU ".to_string()
    };

    let titles = vec!["Compute", "Memory", "Data Flow", "Stress Test"];
    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title(title))
        .select(app.selected_tab)
        .style(Style::default().fg(Color::White))
        .highlight_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    f.render_widget(tabs, chunks[0]);

    // Content based on selected tab
    match app.selected_tab {
        0 => render_compute_tab(f, app, chunks[1]),
        1 => render_memory_tab(f, app, chunks[1]),
        2 => render_stress::render_dataflow_tab(f, app, chunks[1]),
        3 => render_stress::render_stress_tab(f, app, chunks[1]),
        4.. => {} // No tab at this index; ignore
    }

    // Footer
    let help_text = if app.stress_running {
        " q:Quit  Tab:Switch  s:Stop Stress  ?:Help  |  STRESS TEST RUNNING "
    } else {
        " q:Quit  Tab:Switch  s:Start Stress  ?:Help  |  Refresh: 100ms "
    };
    let footer = Paragraph::new(help_text)
        .style(Style::default().fg(Color::DarkGray))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(footer, chunks[2]);

    // Help overlay
    if app.show_help {
        render_stress::render_help_overlay(f, size);
    }
}

fn render_compute_tab(f: &mut Frame, app: &App, area: Rect) {
    let mut constraints = vec![];
    if app.stress_running {
        constraints.push(Constraint::Length(1));
    }
    constraints.push(Constraint::Length(3)); // CPU gauge
    constraints.push(Constraint::Length(5)); // CPU sparkline
    constraints.extend(app.gpus.iter().map(|_| Constraint::Length(3)));
    constraints.push(Constraint::Min(3)); // Device info

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .margin(1)
        .split(area);

    let mut idx = 0;

    if app.stress_running {
        let elapsed = app.stress_start.map(|s| s.elapsed().as_secs()).unwrap_or(0);
        idx = render_stress_banner(
            f,
            format!(
                " STRESS TEST ACTIVE | {}s | CPU: {:.1}M ops/s | MEM: {:.1}M pg/s | GPU: {:.2}G xf/s ",
                elapsed,
                app.cpu_ops_per_sec as f64 / 1_000_000.0,
                app.mem_ops_per_sec as f64 / 1_000_000.0,
                app.gpu_ops_per_sec as f64 / 1_000_000_000.0
            ),
            &chunks,
            idx,
        );
    }

    idx = render_cpu_gauge(f, app, &chunks, idx);
    idx = render_cpu_sparkline(f, app, &chunks, idx);

    let stress_active = app.stress_running && !app.gpu_workers.is_empty();
    for (i, gpu) in app.gpus.iter().enumerate() {
        idx = render_gpu_gauge(f, gpu, i, stress_active, &chunks, idx);
    }

    render_device_info(f, app, &chunks, idx);
}

/// Render the CPU utilization gauge. Returns the next layout index.
fn render_cpu_gauge(f: &mut Frame, app: &App, chunks: &[Rect], idx: usize) -> usize {
    let cpu_pct = app.cpu.compute_utilization().unwrap_or(0.0);
    let cpu_title = if app.stress_running {
        format!(" CPU Utilization [STRESS: {} workers] ", app.cpu_workers.len())
    } else {
        " CPU Utilization ".to_string()
    };
    let cpu_gauge = Gauge::default()
        .block(Block::default().title(cpu_title).borders(Borders::ALL))
        .gauge_style(Style::default().fg(pct_color(cpu_pct)))
        .percent(cpu_pct as u16)
        .label(format!("{:.1}%", cpu_pct));
    f.render_widget(cpu_gauge, chunks[idx]);
    idx + 1
}

/// Render the CPU sparkline (60-second history). Returns the next layout index.
fn render_cpu_sparkline(f: &mut Frame, app: &App, chunks: &[Rect], idx: usize) -> usize {
    let sparkline = Sparkline::default()
        .block(Block::default().title(" CPU History (60s) ").borders(Borders::ALL))
        .data(&app.cpu_history)
        .max(100)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(sparkline, chunks[idx]);
    idx + 1
}

/// Render the device info panel (CPU + GPU summary).
fn render_device_info(f: &mut Frame, app: &App, chunks: &[Rect], idx: usize) {
    let clock = app.cpu.compute_clock_mhz().unwrap_or(0);
    let temp = app.cpu.compute_temperature_c().unwrap_or(0.0);
    let cores = app.cpu.compute_unit_count();
    let temp_color = if temp > 80.0 { Color::Red } else { Color::Green };

    let mut info_lines = vec![
        Line::from(vec![
            Span::styled("CPU: ", Style::default().fg(Color::DarkGray)),
            Span::styled(app.cpu.device_name(), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("Cores: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{}", cores), Style::default().fg(Color::Cyan)),
            Span::raw("  "),
            Span::styled("Clock: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{} MHz", clock), Style::default().fg(Color::Cyan)),
            Span::raw("  "),
            Span::styled("Temp: ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.0}C", temp), Style::default().fg(temp_color)),
        ]),
    ];

    for (i, gpu) in app.gpus.iter().enumerate() {
        info_lines.push(Line::from(vec![
            Span::styled(format!("GPU{}: ", i), Style::default().fg(Color::DarkGray)),
            Span::styled(&gpu.info.name, Style::default().fg(Color::Magenta)),
            Span::styled(
                format!(" ({:.1} GB)", gpu.info.total_memory_gb()),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }

    let info = Paragraph::new(info_lines)
        .block(Block::default().title(" Device Info ").borders(Borders::ALL));
    f.render_widget(info, chunks[idx]);
}

fn render_memory_tab(f: &mut Frame, app: &App, area: Rect) {
    let mut constraints = vec![];
    if app.stress_running {
        constraints.push(Constraint::Length(1));
    }
    constraints.push(Constraint::Length(3)); // RAM gauge
    constraints.push(Constraint::Length(3)); // SWAP gauge
    constraints.extend(app.gpus.iter().map(|_| Constraint::Length(3)));
    constraints.push(Constraint::Length(5)); // Memory sparkline
    constraints.push(Constraint::Min(3)); // Pressure info

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .margin(1)
        .split(area);

    let mut idx = 0;

    if app.stress_running {
        idx = render_stress_banner(
            f,
            format!(
                " STRESS: Allocating 512MB + {} GPU buffers | RAM pressure: {} ",
                app.gpu_workers.len() * 8,
                app.memory.pressure_level
            ),
            &chunks,
            idx,
        );
    }

    idx = render_ram_gauge(f, app, &chunks, idx);
    idx = render_swap_gauge(f, app, &chunks, idx);

    let stress_active = app.stress_running && !app.gpu_workers.is_empty();
    for (i, gpu) in app.gpus.iter().enumerate() {
        idx = render_vram_gauge(f, gpu, i, stress_active, &chunks, idx);
    }

    idx = render_mem_sparkline(f, app, &chunks, idx);
    render_pressure_info(f, app, &chunks, idx);
}

/// Render the RAM usage gauge. Returns the next layout index.
fn render_ram_gauge(f: &mut Frame, app: &App, chunks: &[Rect], idx: usize) -> usize {
    let ram_pct = app.memory.ram_usage_percent();
    let ram_color = pressure_color(&app.memory.pressure_level);
    let ram_used_gb = app.memory.ram_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let ram_total_gb = app.memory.ram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let ram_title = if app.stress_running { " RAM [STRESS ACTIVE] " } else { " RAM " };
    let ram_gauge = Gauge::default()
        .block(Block::default().title(ram_title).borders(Borders::ALL))
        .gauge_style(Style::default().fg(ram_color))
        .percent(ram_pct as u16)
        .label(format!("{:.1} / {:.1} GB ({:.1}%)", ram_used_gb, ram_total_gb, ram_pct));
    f.render_widget(ram_gauge, chunks[idx]);
    idx + 1
}

/// Render the SWAP usage gauge. Returns the next layout index.
fn render_swap_gauge(f: &mut Frame, app: &App, chunks: &[Rect], idx: usize) -> usize {
    let swap_pct = app.memory.swap_usage_percent();
    let swap_used_gb = app.memory.swap_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let swap_total_gb = app.memory.swap_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let swap_color = if swap_pct > 50.0 { Color::Yellow } else { Color::Blue };
    let swap_gauge = Gauge::default()
        .block(Block::default().title(" SWAP ").borders(Borders::ALL))
        .gauge_style(Style::default().fg(swap_color))
        .percent(swap_pct as u16)
        .label(format!("{:.1} / {:.1} GB ({:.1}%)", swap_used_gb, swap_total_gb, swap_pct));
    f.render_widget(swap_gauge, chunks[idx]);
    idx + 1
}

/// Map a `PressureLevel` to a display color.
fn pressure_color(level: &PressureLevel) -> Color {
    match level {
        PressureLevel::Ok => Color::Green,
        PressureLevel::Elevated => Color::Yellow,
        PressureLevel::Warning => Color::Rgb(255, 165, 0),
        PressureLevel::Critical => Color::Red,
    }
}

/// Render a single GPU VRAM gauge in the memory tab. Returns the next layout index.
fn render_vram_gauge(
    f: &mut Frame,
    gpu: &GpuState,
    gpu_idx: usize,
    stress_active: bool,
    chunks: &[Rect],
    layout_idx: usize,
) -> usize {
    let vram_color = pct_color(gpu.vram_percent);
    let title = if stress_active {
        format!(" VRAM {} [STRESS] ", gpu_idx)
    } else {
        format!(" VRAM {} [{}] ", gpu_idx, gpu.info.name)
    };
    let vram_gauge = Gauge::default()
        .block(Block::default().title(title).borders(Borders::ALL))
        .gauge_style(Style::default().fg(vram_color))
        .percent(gpu.vram_percent as u16)
        .label(format!(
            "{:.1} / {:.1} GB ({:.1}%)",
            gpu.vram_used_gb, gpu.vram_total_gb, gpu.vram_percent
        ));
    f.render_widget(vram_gauge, chunks[layout_idx]);
    layout_idx + 1
}

/// Render the memory sparkline (60-second history). Returns the next layout index.
fn render_mem_sparkline(f: &mut Frame, app: &App, chunks: &[Rect], idx: usize) -> usize {
    let sparkline = Sparkline::default()
        .block(Block::default().title(" Memory History (60s) ").borders(Borders::ALL))
        .data(&app.mem_history)
        .max(100)
        .style(Style::default().fg(Color::Magenta));
    f.render_widget(sparkline, chunks[idx]);
    idx + 1
}

/// Render the memory pressure info panel.
fn render_pressure_info(f: &mut Frame, app: &App, chunks: &[Rect], idx: usize) {
    let (label, color, desc) = match app.memory.pressure_level {
        PressureLevel::Ok => ("OK", Color::Green, ">= 50% available"),
        PressureLevel::Elevated => ("ELEVATED", Color::Yellow, "30-50% available"),
        PressureLevel::Warning => ("WARNING", Color::Rgb(255, 165, 0), "15-30% available"),
        PressureLevel::Critical => ("CRITICAL", Color::Red, "< 15% available"),
    };

    let pressure_text = vec![
        Line::from(vec![
            Span::styled("Pressure Level: ", Style::default().fg(Color::DarkGray)),
            Span::styled(label, Style::default().fg(color).add_modifier(Modifier::BOLD)),
            Span::styled(format!(" ({})", desc), Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("Safe Parallel Jobs: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", app.memory.safe_parallel_jobs),
                Style::default().fg(Color::Cyan),
            ),
        ]),
    ];
    let pressure = Paragraph::new(pressure_text)
        .block(Block::default().title(" Memory Pressure (LAMBDA-0002) ").borders(Borders::ALL));
    f.render_widget(pressure, chunks[idx]);
}
