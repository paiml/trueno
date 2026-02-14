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
use super::App;

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
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
    f.render_widget(tabs, chunks[0]);

    // Content based on selected tab
    match app.selected_tab {
        0 => render_compute_tab(f, app, chunks[1]),
        1 => render_memory_tab(f, app, chunks[1]),
        2 => render_stress::render_dataflow_tab(f, app, chunks[1]),
        3 => render_stress::render_stress_tab(f, app, chunks[1]),
        _ => {}
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
    // Calculate constraints based on number of GPUs
    let mut constraints = vec![];

    // Add stress banner if running
    if app.stress_running {
        constraints.push(Constraint::Length(1)); // Stress banner
    }

    constraints.push(Constraint::Length(3)); // CPU gauge
    constraints.push(Constraint::Length(5)); // CPU sparkline

    // Add constraints for each GPU
    for _ in &app.gpus {
        constraints.push(Constraint::Length(3)); // GPU VRAM gauge
    }

    constraints.push(Constraint::Min(3)); // Device info

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .margin(1)
        .split(area);

    let mut idx = 0;

    // Stress banner
    if app.stress_running {
        let elapsed = app.stress_start.map(|s| s.elapsed().as_secs()).unwrap_or(0);
        let banner = Paragraph::new(format!(
            " STRESS TEST ACTIVE | {}s | CPU: {:.1}M ops/s | MEM: {:.1}M pg/s | GPU: {:.2}G xf/s ",
            elapsed,
            app.cpu_ops_per_sec as f64 / 1_000_000.0,
            app.mem_ops_per_sec as f64 / 1_000_000.0,
            app.gpu_ops_per_sec as f64 / 1_000_000_000.0
        ))
        .style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );
        f.render_widget(banner, chunks[idx]);
        idx += 1;
    }

    // CPU utilization gauge
    let cpu_pct = app.cpu.compute_utilization().unwrap_or(0.0);
    let cpu_color = if cpu_pct > 90.0 {
        Color::Red
    } else if cpu_pct > 70.0 {
        Color::Yellow
    } else {
        Color::Green
    };

    let cpu_title = if app.stress_running {
        format!(
            " CPU Utilization [STRESS: {} workers] ",
            app.cpu_workers.len()
        )
    } else {
        " CPU Utilization ".to_string()
    };

    let cpu_gauge = Gauge::default()
        .block(Block::default().title(cpu_title).borders(Borders::ALL))
        .gauge_style(Style::default().fg(cpu_color))
        .percent(cpu_pct as u16)
        .label(format!("{:.1}%", cpu_pct));
    f.render_widget(cpu_gauge, chunks[idx]);
    idx += 1;

    // CPU sparkline (60-second history)
    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .title(" CPU History (60s) ")
                .borders(Borders::ALL),
        )
        .data(&app.cpu_history)
        .max(100)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(sparkline, chunks[idx]);
    idx += 1;

    // GPU VRAM gauges (real hardware)
    for (i, gpu) in app.gpus.iter().enumerate() {
        let gpu_color = if gpu.vram_percent > 90.0 {
            Color::Red
        } else if gpu.vram_percent > 70.0 {
            Color::Yellow
        } else {
            Color::Green
        };

        let title = if app.stress_running && !app.gpu_workers.is_empty() {
            format!(" GPU {} [STRESS] {} ", i, gpu.info.name)
        } else {
            format!(" GPU {} VRAM: {} ", i, gpu.info.name)
        };
        let label = format!(
            "{:.1} / {:.1} GB ({:.1}%)",
            gpu.vram_used_gb, gpu.vram_total_gb, gpu.vram_percent
        );

        let gpu_gauge = Gauge::default()
            .block(Block::default().title(title).borders(Borders::ALL))
            .gauge_style(Style::default().fg(gpu_color))
            .percent(gpu.vram_percent as u16)
            .label(label);
        f.render_widget(gpu_gauge, chunks[idx]);
        idx += 1;
    }

    // Device info
    let clock = app.cpu.compute_clock_mhz().unwrap_or(0);
    let temp = app.cpu.compute_temperature_c().unwrap_or(0.0);
    let cores = app.cpu.compute_unit_count();

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
            Span::styled(
                format!("{:.0}C", temp),
                Style::default().fg(if temp > 80.0 {
                    Color::Red
                } else {
                    Color::Green
                }),
            ),
        ]),
    ];

    // Add GPU info lines
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

    let info = Paragraph::new(info_lines).block(
        Block::default()
            .title(" Device Info ")
            .borders(Borders::ALL),
    );
    f.render_widget(info, chunks[idx]);
}

fn render_memory_tab(f: &mut Frame, app: &App, area: Rect) {
    // Calculate constraints based on number of GPUs
    let mut constraints = vec![];

    if app.stress_running {
        constraints.push(Constraint::Length(1)); // Stress banner
    }

    constraints.push(Constraint::Length(3)); // RAM gauge
    constraints.push(Constraint::Length(3)); // SWAP gauge

    // Add VRAM gauge for each GPU
    for _ in &app.gpus {
        constraints.push(Constraint::Length(3));
    }

    constraints.push(Constraint::Length(5)); // Memory sparkline
    constraints.push(Constraint::Min(3)); // Pressure info

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .margin(1)
        .split(area);

    let mut idx = 0;

    // Stress banner
    if app.stress_running {
        let banner = Paragraph::new(format!(
            " STRESS: Allocating 512MB + {} GPU buffers | RAM pressure: {} ",
            app.gpu_workers.len() * 8, // 8MB per GPU (2 x 4MB buffers)
            app.memory.pressure_level
        ))
        .style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );
        f.render_widget(banner, chunks[idx]);
        idx += 1;
    }

    // RAM gauge
    let ram_pct = app.memory.ram_usage_percent();
    let ram_color = match app.memory.pressure_level {
        PressureLevel::Ok => Color::Green,
        PressureLevel::Elevated => Color::Yellow,
        PressureLevel::Warning => Color::Rgb(255, 165, 0), // Orange
        PressureLevel::Critical => Color::Red,
    };

    let ram_used_gb = app.memory.ram_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let ram_total_gb = app.memory.ram_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    let ram_title = if app.stress_running {
        " RAM [STRESS ACTIVE] ".to_string()
    } else {
        " RAM ".to_string()
    };

    let ram_gauge = Gauge::default()
        .block(Block::default().title(ram_title).borders(Borders::ALL))
        .gauge_style(Style::default().fg(ram_color))
        .percent(ram_pct as u16)
        .label(format!(
            "{:.1} / {:.1} GB ({:.1}%)",
            ram_used_gb, ram_total_gb, ram_pct
        ));
    f.render_widget(ram_gauge, chunks[idx]);
    idx += 1;

    // SWAP gauge
    let swap_pct = app.memory.swap_usage_percent();
    let swap_used_gb = app.memory.swap_used_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let swap_total_gb = app.memory.swap_total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    let swap_gauge = Gauge::default()
        .block(Block::default().title(" SWAP ").borders(Borders::ALL))
        .gauge_style(Style::default().fg(if swap_pct > 50.0 {
            Color::Yellow
        } else {
            Color::Blue
        }))
        .percent(swap_pct as u16)
        .label(format!(
            "{:.1} / {:.1} GB ({:.1}%)",
            swap_used_gb, swap_total_gb, swap_pct
        ));
    f.render_widget(swap_gauge, chunks[idx]);
    idx += 1;

    // GPU VRAM gauges (real hardware)
    for (i, gpu) in app.gpus.iter().enumerate() {
        let vram_color = if gpu.vram_percent > 90.0 {
            Color::Red
        } else if gpu.vram_percent > 70.0 {
            Color::Yellow
        } else {
            Color::Magenta
        };

        let title = if app.stress_running && !app.gpu_workers.is_empty() {
            format!(" VRAM {} [STRESS] ", i)
        } else {
            format!(" VRAM {} [{}] ", i, gpu.info.name)
        };
        let vram_gauge = Gauge::default()
            .block(Block::default().title(title).borders(Borders::ALL))
            .gauge_style(Style::default().fg(vram_color))
            .percent(gpu.vram_percent as u16)
            .label(format!(
                "{:.1} / {:.1} GB ({:.1}%)",
                gpu.vram_used_gb, gpu.vram_total_gb, gpu.vram_percent
            ));
        f.render_widget(vram_gauge, chunks[idx]);
        idx += 1;
    }

    // Memory sparkline
    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .title(" Memory History (60s) ")
                .borders(Borders::ALL),
        )
        .data(&app.mem_history)
        .max(100)
        .style(Style::default().fg(Color::Magenta));
    f.render_widget(sparkline, chunks[idx]);
    idx += 1;

    // Pressure info
    let pressure_str = match app.memory.pressure_level {
        PressureLevel::Ok => ("OK", Color::Green, ">= 50% available"),
        PressureLevel::Elevated => ("ELEVATED", Color::Yellow, "30-50% available"),
        PressureLevel::Warning => ("WARNING", Color::Rgb(255, 165, 0), "15-30% available"),
        PressureLevel::Critical => ("CRITICAL", Color::Red, "< 15% available"),
    };

    let pressure_text = vec![
        Line::from(vec![
            Span::styled("Pressure Level: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                pressure_str.0,
                Style::default()
                    .fg(pressure_str.1)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" ({})", pressure_str.2),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("Safe Parallel Jobs: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", app.memory.safe_parallel_jobs),
                Style::default().fg(Color::Cyan),
            ),
        ]),
    ];
    let pressure = Paragraph::new(pressure_text).block(
        Block::default()
            .title(" Memory Pressure (LAMBDA-0002) ")
            .borders(Borders::ALL),
    );
    f.render_widget(pressure, chunks[idx]);
}
