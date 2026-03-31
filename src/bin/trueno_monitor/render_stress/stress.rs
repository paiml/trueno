//! Stress test tab rendering (idle screen and running visualization).

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline},
    Frame,
};
use trueno_gpu::monitor::{ComputeDevice, PressureLevel};

use crate::{App, StressTestVerdict};

pub(crate) fn render_stress_tab(f: &mut Frame, app: &App, area: Rect) {
    if app.stress_running {
        // Show real-time stress visualization
        render_stress_running(f, app, area);
    } else {
        // Show start screen
        render_stress_idle(f, app, area);
    }
}

fn render_stress_idle(f: &mut Frame, app: &App, area: Rect) {
    let block =
        Block::default().title(" Stress Test Mode (TRUENO-SPEC-025) ").borders(Borders::ALL);

    let mut text = vec![
        Line::from(""),
        Line::from(vec![
            Span::raw("  Status: "),
            Span::styled("IDLE", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Hardware Detected:",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::raw("    CPU: "),
            Span::styled(app.cpu.device_name(), Style::default().fg(Color::White)),
            Span::styled(
                format!(" ({} cores)", num_cpus::get()),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ];

    for (i, gpu) in app.gpus.iter().enumerate() {
        text.push(Line::from(vec![
            Span::raw(format!("    GPU{}: ", i)),
            Span::styled(&gpu.info.name, Style::default().fg(Color::Magenta)),
        ]));
    }

    text.extend(vec![
        Line::from(""),
        Line::from(Span::styled(
            "  Stress Test Will:",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::raw("    "),
            Span::styled("CPU:", Style::default().fg(Color::Yellow)),
            Span::raw(format!(" {} threads doing FP math (sin/cos/sqrt)", num_cpus::get())),
        ]),
        Line::from(vec![
            Span::raw("    "),
            Span::styled("MEM:", Style::default().fg(Color::Yellow)),
            Span::raw(" Allocate 512MB, touch every page"),
        ]),
    ]);

    for (i, gpu) in app.gpus.iter().enumerate() {
        text.push(Line::from(vec![
            Span::raw("    "),
            Span::styled(format!("GPU{}:", i), Style::default().fg(Color::Yellow)),
            Span::raw(format!(" {} - H2D/D2H transfers (1M x f32)", gpu.info.name)),
        ]));
    }

    // Show stress test report if available (renacer integration)
    if let Some(ref report) = app.stress_report {
        let verdict_color = match report.verdict {
            StressTestVerdict::Pass => Color::Green,
            StressTestVerdict::PassWithNotes => Color::Yellow,
            StressTestVerdict::Fail => Color::Red,
        };
        text.extend(vec![
            Line::from(""),
            Line::from(Span::styled(
                "  Stress Test Report (renacer):",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            )),
            Line::from(""),
            Line::from(vec![
                Span::raw("    Verdict: "),
                Span::styled(
                    format!("{}", report.verdict),
                    Style::default().fg(verdict_color).add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(vec![
                Span::raw("    Duration: "),
                Span::styled(
                    format!("{}s", report.duration_secs),
                    Style::default().fg(Color::White),
                ),
                Span::raw(" | Workers: "),
                Span::styled(
                    format!("{} CPU + {} GPU", report.cpu_workers, report.gpu_workers),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(""),
            Line::from(vec![
                Span::raw("    Peak CPU: "),
                Span::styled(
                    format!(
                        "{:.2} M ops/sec ({:.1}%)",
                        report.peak_cpu_ops as f64 / 1_000_000.0,
                        report.peak_cpu_util
                    ),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::raw("    Peak MEM: "),
                Span::styled(
                    format!(
                        "{:.2} M pages/sec ({:.1}%)",
                        report.peak_mem_ops as f64 / 1_000_000.0,
                        report.peak_ram_util
                    ),
                    Style::default().fg(Color::Magenta),
                ),
            ]),
        ]);
        if report.peak_gpu_ops > 0 {
            text.push(Line::from(vec![
                Span::raw("    Peak GPU: "),
                Span::styled(
                    format!(
                        "{:.2} G xfers/sec ({:.1}% VRAM)",
                        report.peak_gpu_ops as f64 / 1_000_000_000.0,
                        report.peak_vram_util
                    ),
                    Style::default().fg(Color::Yellow),
                ),
            ]));
        }
        // Show recommendations if any
        if !report.recommendations.is_empty() {
            text.push(Line::from(""));
            text.push(Line::from(Span::styled(
                "  Recommendations:",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            )));
            for rec in &report.recommendations {
                text.push(Line::from(vec![
                    Span::raw("    \u{2022} "),
                    Span::styled(rec, Style::default().fg(Color::DarkGray)),
                ]));
            }
        }
    }

    text.extend(vec![
        Line::from(""),
        Line::from(""),
        Line::from(Span::styled(
            "  >>> Press 's' to START stress test <<<",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        )),
    ]);

    let paragraph = Paragraph::new(text).block(block);
    f.render_widget(paragraph, area);
}

fn render_stress_running(f: &mut Frame, app: &App, area: Rect) {
    let has_gpu = !app.gpu_workers.is_empty();

    let mut constraints = vec![
        Constraint::Length(3), // Status bar
        Constraint::Length(3), // CPU ops gauge
        Constraint::Length(4), // CPU ops sparkline
        Constraint::Length(3), // Memory ops gauge
        Constraint::Length(4), // Memory ops sparkline
    ];

    if has_gpu {
        constraints.push(Constraint::Length(3)); // GPU ops gauge
        constraints.push(Constraint::Length(4)); // GPU ops sparkline
                                                 // GH-194: Per-GPU VRAM history sparklines (was dead code)
        for _ in 0..app.gpu_vram_history.len() {
            constraints.push(Constraint::Length(4)); // VRAM sparkline per GPU
        }
    }

    constraints.push(Constraint::Min(3)); // Stats

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .margin(1)
        .split(area);

    // Status bar with elapsed time
    let elapsed = app.stress_start.map(|s| s.elapsed().as_secs()).unwrap_or(0);
    let status_text = format!(
        " STRESS TEST RUNNING | {}s | {} CPU + {} GPU workers | 's' to STOP ",
        elapsed,
        app.cpu_workers.len(),
        app.gpu_workers.len()
    );
    let status = Paragraph::new(status_text)
        .style(Style::default().fg(Color::Black).bg(Color::Green).add_modifier(Modifier::BOLD))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(status, chunks[0]);

    // CPU ops gauge
    let cpu_mops = app.cpu_ops_per_sec as f64 / 1_000_000.0;
    let cpu_pct = ((cpu_mops / 100.0) * 100.0).min(100.0) as u16;
    let cpu_gauge = Gauge::default()
        .block(
            Block::default()
                .title(format!(
                    " CPU: {:.1} M ops/sec (peak: {:.1}) ",
                    cpu_mops,
                    app.peak_cpu_ops as f64 / 1_000_000.0
                ))
                .borders(Borders::ALL),
        )
        .gauge_style(Style::default().fg(Color::Cyan))
        .percent(cpu_pct)
        .label(format!("{:.2} M ops/sec", cpu_mops));
    f.render_widget(cpu_gauge, chunks[1]);

    // CPU ops sparkline
    let cpu_sparkline = Sparkline::default()
        .block(Block::default().title(" CPU History ").borders(Borders::ALL))
        .data(&app.cpu_ops_history)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(cpu_sparkline, chunks[2]);

    // Memory ops gauge
    let mem_mops = app.mem_ops_per_sec as f64 / 1_000_000.0;
    let mem_pct = ((mem_mops / 10.0) * 100.0).min(100.0) as u16;
    let mem_gauge = Gauge::default()
        .block(
            Block::default()
                .title(format!(
                    " MEM: {:.1} M pages/sec (peak: {:.1}) ",
                    mem_mops,
                    app.peak_mem_ops as f64 / 1_000_000.0
                ))
                .borders(Borders::ALL),
        )
        .gauge_style(Style::default().fg(Color::Magenta))
        .percent(mem_pct)
        .label(format!("{:.2} M pages/sec", mem_mops));
    f.render_widget(mem_gauge, chunks[3]);

    // Memory ops sparkline
    let mem_sparkline = Sparkline::default()
        .block(Block::default().title(" Memory History ").borders(Borders::ALL))
        .data(&app.mem_ops_history)
        .style(Style::default().fg(Color::Magenta));
    f.render_widget(mem_sparkline, chunks[4]);

    let stats_idx = if has_gpu {
        // GPU ops gauge
        let gpu_gops = app.gpu_ops_per_sec as f64 / 1_000_000_000.0;
        let gpu_pct = ((gpu_gops / 10.0) * 100.0).min(100.0) as u16; // Scale: 10 G = 100%
        let gpu_gauge = Gauge::default()
            .block(
                Block::default()
                    .title(format!(
                        " GPU: {:.2} G transfers/sec (peak: {:.2}) ",
                        gpu_gops,
                        app.peak_gpu_ops as f64 / 1_000_000_000.0
                    ))
                    .borders(Borders::ALL),
            )
            .gauge_style(Style::default().fg(Color::Yellow))
            .percent(gpu_pct)
            .label(format!("{:.2} G xfers/sec", gpu_gops));
        f.render_widget(gpu_gauge, chunks[5]);

        // GPU ops sparkline
        let gpu_sparkline = Sparkline::default()
            .block(Block::default().title(" GPU History ").borders(Borders::ALL))
            .data(&app.gpu_ops_history)
            .style(Style::default().fg(Color::Yellow));
        f.render_widget(gpu_sparkline, chunks[6]);

        // GH-194: Per-GPU VRAM history sparklines (was dead code — now displayed)
        let mut next_chunk = 7;
        for (i, vram_history) in app.gpu_vram_history.iter().enumerate() {
            let gpu_name = app.gpus.get(i).map(|g| g.info.name.as_str()).unwrap_or("GPU");
            let vram_sparkline = Sparkline::default()
                .block(
                    Block::default()
                        .title(format!(" GPU{} VRAM % ({}) ", i, gpu_name))
                        .borders(Borders::ALL),
                )
                .data(vram_history)
                .max(100)
                .style(Style::default().fg(Color::Magenta));
            f.render_widget(vram_sparkline, chunks[next_chunk]);
            next_chunk += 1;
        }

        next_chunk
    } else {
        5
    };

    // Live stats
    let cpu_util = app.cpu.compute_utilization().unwrap_or(0.0);
    let mem_pct_used = app.memory.ram_usage_percent();

    let mut stats = vec![Line::from(vec![
        Span::styled("System: ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("CPU {:.0}%", cpu_util),
            Style::default().fg(if cpu_util > 90.0 { Color::Red } else { Color::Green }),
        ),
        Span::raw(" | "),
        Span::styled(
            format!("RAM {:.0}%", mem_pct_used),
            Style::default().fg(if mem_pct_used > 80.0 { Color::Red } else { Color::Green }),
        ),
        Span::raw(" | Pressure: "),
        Span::styled(
            format!("{}", app.memory.pressure_level),
            Style::default().fg(match app.memory.pressure_level {
                PressureLevel::Ok => Color::Green,
                PressureLevel::Elevated => Color::Yellow,
                PressureLevel::Warning => Color::Rgb(255, 165, 0),
                PressureLevel::Critical => Color::Red,
            }),
        ),
    ])];

    // Show GPU VRAM if available
    for (i, gpu) in app.gpus.iter().enumerate() {
        stats.push(Line::from(vec![
            Span::styled(format!("GPU{} VRAM: ", i), Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1}%", gpu.vram_percent),
                Style::default().fg(if gpu.vram_percent > 90.0 {
                    Color::Red
                } else {
                    Color::Green
                }),
            ),
            Span::styled(
                format!(" ({:.1}/{:.1} GB)", gpu.vram_used_gb, gpu.vram_total_gb),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }

    let stats_block = Paragraph::new(stats)
        .block(Block::default().title(" System Impact ").borders(Borders::ALL));
    f.render_widget(stats_block, chunks[stats_idx]);
}
