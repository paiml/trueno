//! Rendering methods for the cbtop TUI.

use presentar_core::{Canvas, Point, Rect, TextStyle};
use presentar_terminal::direct::DirectTerminalCanvas;
use presentar_terminal::Theme;

use super::hardware::{DiskMetrics, HardwareInfo, LoadMetrics, MemoryBreakdown, NetworkMetrics};
use super::panels::ActivePanel;
use super::CbtopApp;
use crate::config::ComputeBackend;
use crate::error::CbtopError;

impl CbtopApp {
    /// Render the UI
    #[allow(clippy::wildcard_enum_match_arm)]
    pub(super) fn render(&mut self) -> Result<(), CbtopError> {
        use presentar_terminal::direct::CellBuffer;

        // Resize buffer if needed
        let (width, height) = crossterm::terminal::size()?;
        if self.buffer.width() != width || self.buffer.height() != height {
            self.buffer = CellBuffer::new(width, height);
        }

        // Extract state needed for rendering
        let active_panel = self.active_panel;
        let is_running = self.is_running;
        let intensity = self.intensity;
        let backend = self.backend;
        let show_fps = self.config.show_fps;
        let problem_size = self.problem_size;
        let _frame_count = self.frame_count;
        let cpu_data: Vec<f64> = self.cpu_history.iter().copied().collect();
        let bricks_data: Vec<f64> = self.bricks_history.iter().copied().collect();
        let cpu_avg = self.cpu_history.mean();
        let frame_avg = self.frame_times.mean();
        let hardware = self.hardware.clone();
        let metrics = self.load_metrics.clone();

        {
            let mut canvas = DirectTerminalCanvas::new(&mut self.buffer);

            // Background
            canvas.fill_rect(
                Rect::new(0.0, 0.0, width as f32, height as f32),
                self.theme.background,
            );

            // Title bar with hardware info
            Self::render_title_bar(&mut canvas, width, active_panel, &hardware, &self.theme);

            // Main content - REAL metrics display
            Self::render_main_content(
                &mut canvas,
                width,
                height,
                is_running,
                &metrics,
                &cpu_data,
                &bricks_data,
                cpu_avg,
                problem_size,
                &hardware,
                &self.theme,
            );

            // Status bar with real metrics
            Self::render_status_bar(
                &mut canvas,
                width,
                height,
                is_running,
                intensity,
                backend,
                show_fps,
                frame_avg,
                &metrics,
                &self.theme,
            );
        }

        // Flush to terminal
        let mut output = Vec::with_capacity(16384);
        self.renderer
            .flush(&mut self.buffer, &mut output)
            .map_err(|e| CbtopError::Render(e.to_string()))?;
        std::io::Write::write_all(&mut std::io::stdout(), &output)?;

        Ok(())
    }

    pub(super) fn render_title_bar(
        canvas: &mut DirectTerminalCanvas,
        width: u16,
        active_panel: ActivePanel,
        hardware: &HardwareInfo,
        theme: &Theme,
    ) {
        let title_style = TextStyle {
            color: theme.foreground,
            ..Default::default()
        };

        // Title with hardware info
        let hw_info = format!(
            " cbtop │ {} ({} cores, {}) │ {:.0}GB RAM ",
            hardware.cpu_model.chars().take(30).collect::<String>(),
            hardware.cpu_cores,
            hardware.simd_type,
            hardware.memory_gb,
        );
        canvas.draw_text(&hw_info, Point::new(0.0, 0.0), &title_style);

        // GPU info if available
        if let Some(ref gpu) = hardware.gpu_name {
            let gpu_style = TextStyle {
                color: theme.cpu.sample(0.5),
                ..Default::default()
            };
            canvas.draw_text(
                &format!("│ GPU: {} ", gpu.chars().take(25).collect::<String>()),
                Point::new(hw_info.len() as f32, 0.0),
                &gpu_style,
            );
        }

        // PMAT-012 UI-10: Panel navigation tab bar
        Self::render_tab_bar(canvas, width, active_panel, theme);
    }

    /// Render panel navigation tab bar (UI-10)
    pub(super) fn render_tab_bar(
        canvas: &mut DirectTerminalCanvas,
        width: u16,
        active_panel: ActivePanel,
        theme: &Theme,
    ) {
        let dim_style = TextStyle {
            color: theme.dim,
            ..Default::default()
        };
        let active_style = TextStyle {
            color: theme.foreground,
            ..Default::default()
        };
        let highlight_style = TextStyle {
            color: theme.cpu.sample(0.2),
            ..Default::default()
        };

        // Build tab bar string with highlighting
        let mut x: f32 = 1.0;
        canvas.draw_text("│", Point::new(0.0, 1.0), &dim_style);

        for panel in ActivePanel::all() {
            let key = panel.key_number();
            let title = panel.title();
            let is_active = *panel == active_panel;

            if is_active {
                // Active panel: highlighted with brackets
                canvas.draw_text("[", Point::new(x, 1.0), &highlight_style);
                x += 1.0;
                canvas.draw_text(&format!("{}", key), Point::new(x, 1.0), &highlight_style);
                x += 1.0;
                canvas.draw_text(":", Point::new(x, 1.0), &highlight_style);
                x += 1.0;
                canvas.draw_text(title, Point::new(x, 1.0), &active_style);
                x += title.len() as f32;
                canvas.draw_text("]", Point::new(x, 1.0), &highlight_style);
                x += 1.0;
            } else {
                // Inactive panel: dimmed
                canvas.draw_text(&format!(" {}:", key), Point::new(x, 1.0), &dim_style);
                x += 3.0;
                canvas.draw_text(title, Point::new(x, 1.0), &dim_style);
                x += title.len() as f32;
            }

            // Separator between panels (except last)
            if *panel != ActivePanel::Help {
                canvas.draw_text(" ", Point::new(x, 1.0), &dim_style);
                x += 1.0;
            }
        }

        // Fill remaining width with spaces and close
        let remaining = (width as f32 - x - 1.0).max(0.0) as usize;
        if remaining > 0 {
            canvas.draw_text(&" ".repeat(remaining), Point::new(x, 1.0), &dim_style);
        }
        canvas.draw_text("│", Point::new(width as f32 - 1.0, 1.0), &dim_style);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_main_content(
        canvas: &mut DirectTerminalCanvas,
        width: u16,
        height: u16,
        is_running: bool,
        metrics: &LoadMetrics,
        cpu_data: &[f64],
        bricks_data: &[f64],
        cpu_avg: f64,
        problem_size: usize,
        hardware: &HardwareInfo,
        theme: &Theme,
    ) {
        let dim_style = TextStyle {
            color: theme.dim,
            ..Default::default()
        };
        let bright_style = TextStyle {
            color: theme.foreground,
            ..Default::default()
        };
        let accent_style = TextStyle {
            color: theme.cpu.sample(0.3),
            ..Default::default()
        };

        // PMAT-012 UI-01: Responsive width boxes
        let box_width = (width as usize).saturating_sub(2).max(40);
        let inner_width = box_width.saturating_sub(2);
        let bar_width = inner_width.saturating_sub(22).max(10);

        // Header line 2: Load status
        let status = if is_running {
            "● RUNNING"
        } else {
            "○ STOPPED"
        };
        let status_color = if is_running {
            theme.cpu.sample(0.0)
        } else {
            theme.dim
        };
        canvas.draw_text(
            &format!(" Load: {} ", status),
            Point::new(0.0, 2.0),
            &TextStyle {
                color: status_color,
                ..Default::default()
            },
        );

        // Metrics box with responsive width
        let box_top = format!(
            "┌─ Real-Time Metrics {}┐",
            "─".repeat(inner_width.saturating_sub(20))
        );
        canvas.draw_text(&box_top, Point::new(1.0, 3.0), &dim_style);

        // CPU Usage (REAL from /proc/stat) with color gradient
        let cpu_bar = Self::make_bar(cpu_avg, 100.0, bar_width);
        canvas.draw_text("│ CPU Usage:     ", Point::new(1.0, 4.0), &dim_style);
        canvas.draw_text(
            &cpu_bar,
            Point::new(17.0, 4.0),
            &TextStyle {
                color: theme.cpu_color(cpu_avg),
                ..Default::default()
            },
        );
        let cpu_val_x = 17.0 + bar_width as f32 + 1.0;
        canvas.draw_text(
            &format!("{:5.1}%", cpu_avg),
            Point::new(cpu_val_x, 4.0),
            &bright_style,
        );
        canvas.draw_text(" │", Point::new(box_width as f32, 4.0), &dim_style);

        // Bricks/Second with color gradient based on rate
        let bps = metrics.bricks_per_second;
        let bps_normalized = (bps / 10000.0).min(1.0) * 100.0;
        let bps_bar = Self::make_bar(bps.min(10000.0), 10000.0, bar_width);
        canvas.draw_text("│ Bricks/sec:    ", Point::new(1.0, 5.0), &dim_style);
        canvas.draw_text(
            &bps_bar,
            Point::new(17.0, 5.0),
            &TextStyle {
                color: theme.cpu_color(bps_normalized),
                ..Default::default()
            },
        );
        canvas.draw_text(
            &format!("{:>7.0}", bps),
            Point::new(cpu_val_x, 5.0),
            &bright_style,
        );
        canvas.draw_text(" │", Point::new(box_width as f32, 5.0), &dim_style);

        // Total Bricks executed
        let total_str = format!(
            "{:>width$}",
            Self::format_number(metrics.total_bricks),
            width = inner_width.saturating_sub(17)
        );
        canvas.draw_text("│ Total Bricks:  ", Point::new(1.0, 6.0), &dim_style);
        canvas.draw_text(&total_str, Point::new(17.0, 6.0), &bright_style);
        canvas.draw_text(" │", Point::new(box_width as f32, 6.0), &dim_style);

        // Avg Latency
        let latency_str = format!(
            "{:>width$.1} μs",
            metrics.avg_latency_us,
            width = inner_width.saturating_sub(20)
        );
        canvas.draw_text("│ Avg Latency:   ", Point::new(1.0, 7.0), &dim_style);
        canvas.draw_text(&latency_str, Point::new(17.0, 7.0), &bright_style);
        canvas.draw_text(" │", Point::new(box_width as f32, 7.0), &dim_style);

        // Problem size
        let size_str = format!(
            "{:>width$} elements",
            Self::format_number(problem_size as u64),
            width = inner_width.saturating_sub(26)
        );
        canvas.draw_text("│ Problem Size:  ", Point::new(1.0, 8.0), &dim_style);
        canvas.draw_text(&size_str, Point::new(17.0, 8.0), &bright_style);
        canvas.draw_text(" │", Point::new(box_width as f32, 8.0), &dim_style);

        // Throughput
        let gflops = metrics.ops_per_second / 1_000_000_000.0;
        let gbps = metrics.bytes_per_second / 1_073_741_824.0;
        canvas.draw_text("│ Throughput:    ", Point::new(1.0, 9.0), &dim_style);
        canvas.draw_text(
            &format!("{:>10.2} GFLOP/s │ {:>10.2} GB/s", gflops, gbps),
            Point::new(17.0, 9.0),
            &accent_style,
        );
        canvas.draw_text(" │", Point::new(box_width as f32, 9.0), &dim_style);

        let box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&box_bottom, Point::new(1.0, 10.0), &dim_style);

        // PMAT-012 UI-02: Per-core CPU bars
        let mut current_y = 12.0_f32;
        if !metrics.per_core_usage.is_empty() && height > 20 {
            Self::render_per_core_cpu(
                canvas,
                &mut current_y,
                height,
                metrics,
                inner_width,
                box_width,
                theme,
                &dim_style,
            );
        } else {
            Self::render_hardware_box(
                canvas,
                &mut current_y,
                hardware,
                inner_width,
                box_width,
                theme,
                &dim_style,
            );
        }

        // PMAT-012 UI-04: Memory breakdown box
        if metrics.memory.total_kb > 0 && current_y < height as f32 - 12.0 {
            Self::render_memory_box(
                canvas,
                &mut current_y,
                metrics,
                inner_width,
                box_width,
                bar_width,
                theme,
                &dim_style,
                &bright_style,
            );
        }

        // PMAT-012 F405: GPU panel when NVIDIA present
        if hardware.gpu_name.is_some() && current_y < height as f32 - 10.0 {
            Self::render_gpu_box(
                canvas,
                &mut current_y,
                hardware,
                inner_width,
                box_width,
                theme,
                &dim_style,
            );
        }

        // PMAT-012 UI-07 P2: Network TX/RX panel
        if (metrics.network.rx_rate > 0.0 || metrics.network.tx_rate > 0.0)
            && current_y < height as f32 - 8.0
        {
            Self::render_network_box(
                canvas,
                &mut current_y,
                metrics,
                inner_width,
                box_width,
                &dim_style,
            );
        }

        // PMAT-012 UI-08 P2: Disk per-mount panel
        if !metrics.disks.is_empty() && current_y < height as f32 - 6.0 {
            Self::render_disk_box(
                canvas,
                &mut current_y,
                height,
                metrics,
                inner_width,
                box_width,
                theme,
                &dim_style,
            );
        }

        // PMAT-012 UI-09: Sparklines
        let sparkline_width = (width as usize).saturating_sub(6).max(10);
        Self::render_sparklines(
            canvas,
            &mut current_y,
            height,
            cpu_data,
            bricks_data,
            inner_width,
            box_width,
            sparkline_width,
            theme,
            &dim_style,
            &accent_style,
        );

        // Controls reminder
        if height > 10 {
            canvas.draw_text(
                " [Space] Toggle load  [+/-] Intensity  [b] Backend  [w] Workload  [r] Reset  [q] Quit ",
                Point::new(1.0, height as f32 - 3.0),
                &dim_style,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn render_per_core_cpu(
        canvas: &mut DirectTerminalCanvas,
        current_y: &mut f32,
        height: u16,
        metrics: &LoadMetrics,
        inner_width: usize,
        box_width: usize,
        theme: &Theme,
        dim_style: &TextStyle,
    ) {
        let core_box_top = format!(
            "┌─ Per-Core CPU {}┐",
            "─".repeat(inner_width.saturating_sub(15))
        );
        canvas.draw_text(&core_box_top, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        // Render cores in rows of 4 for compact display
        let cores_per_row = 4.min(metrics.per_core_usage.len());
        let mini_bar_width = (inner_width / cores_per_row).saturating_sub(10).max(5);

        for (i, chunk) in metrics.per_core_usage.chunks(cores_per_row).enumerate() {
            let mut row = String::from("│ ");
            for (j, &usage) in chunk.iter().enumerate() {
                let core_num = i * cores_per_row + j;
                let mini_bar = Self::make_mini_bar(usage, 100.0, mini_bar_width);
                row.push_str(&format!("C{:02}:{} ", core_num, mini_bar));
            }
            // Pad to box width
            while row.len() < box_width {
                row.push(' ');
            }
            row.push('│');

            // Draw with per-core color gradient
            canvas.draw_text(&row[..3], Point::new(1.0, *current_y), dim_style);
            let mut x_pos = 4.0;
            for &usage in chunk.iter() {
                let bar_str = Self::make_mini_bar(usage, 100.0, mini_bar_width);
                canvas.draw_text(
                    &bar_str,
                    Point::new(x_pos + 4.0, *current_y),
                    &TextStyle {
                        color: theme.cpu_color(usage),
                        ..Default::default()
                    },
                );
                x_pos += (mini_bar_width + 10) as f32;
            }
            canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
            *current_y += 1.0;

            if *current_y > height as f32 - 10.0 {
                break; // Don't overflow screen
            }
        }

        let core_box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&core_box_bottom, Point::new(1.0, *current_y), dim_style);
        *current_y += 2.0;
    }

    #[allow(clippy::too_many_arguments)]
    fn render_hardware_box(
        canvas: &mut DirectTerminalCanvas,
        current_y: &mut f32,
        hardware: &HardwareInfo,
        inner_width: usize,
        box_width: usize,
        _theme: &Theme,
        dim_style: &TextStyle,
    ) {
        let hw_box_top = format!(
            "┌─ Hardware {}┐",
            "─".repeat(inner_width.saturating_sub(12))
        );
        canvas.draw_text(&hw_box_top, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        let cpu_info = format!(
            "│ CPU:  {:width$} │",
            hardware
                .cpu_model
                .chars()
                .take(inner_width.saturating_sub(10))
                .collect::<String>(),
            width = inner_width.saturating_sub(8)
        );
        canvas.draw_text(&cpu_info, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        canvas.draw_text(
            &format!(
                "│ Cores: {} │ SIMD: {} ",
                hardware.cpu_cores, hardware.simd_type
            ),
            Point::new(1.0, *current_y),
            dim_style,
        );
        canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
        *current_y += 1.0;

        if let Some(ref gpu) = hardware.gpu_name {
            let gpu_str = format!(
                "│ GPU:  {:width$} │",
                gpu.chars()
                    .take(inner_width.saturating_sub(10))
                    .collect::<String>(),
                width = inner_width.saturating_sub(8)
            );
            canvas.draw_text(&gpu_str, Point::new(1.0, *current_y), dim_style);
        } else {
            canvas.draw_text(
                "│ GPU:  Not detected ",
                Point::new(1.0, *current_y),
                dim_style,
            );
            canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
        }
        *current_y += 1.0;

        canvas.draw_text(
            &format!("│ RAM:  {:.1} GB ", hardware.memory_gb),
            Point::new(1.0, *current_y),
            dim_style,
        );
        canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
        *current_y += 1.0;

        let hw_box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&hw_box_bottom, Point::new(1.0, *current_y), dim_style);
        *current_y += 2.0;
    }

    #[allow(clippy::too_many_arguments)]
    fn render_memory_box(
        canvas: &mut DirectTerminalCanvas,
        current_y: &mut f32,
        metrics: &LoadMetrics,
        inner_width: usize,
        box_width: usize,
        bar_width: usize,
        theme: &Theme,
        dim_style: &TextStyle,
        bright_style: &TextStyle,
    ) {
        let mem_box_top = format!("┌─ Memory {}┐", "─".repeat(inner_width.saturating_sub(11)));
        canvas.draw_text(&mem_box_top, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        // Memory usage bar with color gradient
        let mem_pct = metrics.memory.usage_percent();
        let mem_bar = Self::make_bar(mem_pct, 100.0, bar_width);
        canvas.draw_text("│ Used:    ", Point::new(1.0, *current_y), dim_style);
        canvas.draw_text(
            &mem_bar,
            Point::new(11.0, *current_y),
            &TextStyle {
                color: theme.memory_color(mem_pct),
                ..Default::default()
            },
        );
        let mem_val_x = 11.0 + bar_width as f32 + 1.0;
        canvas.draw_text(
            &format!("{:5.1}%", mem_pct),
            Point::new(mem_val_x, *current_y),
            bright_style,
        );
        canvas.draw_text(" │", Point::new(box_width as f32, *current_y), dim_style);
        *current_y += 1.0;

        // Memory breakdown values
        let total_str = MemoryBreakdown::format_kb(metrics.memory.total_kb);
        let used_str = MemoryBreakdown::format_kb(metrics.memory.used_kb);
        let cached_str = MemoryBreakdown::format_kb(metrics.memory.cached_kb);
        let buffers_str = MemoryBreakdown::format_kb(metrics.memory.buffers_kb);

        canvas.draw_text(
            &format!(
                "│ Total: {:>8}  Used: {:>8}  Cache: {:>8}  Buf: {:>6} ",
                total_str, used_str, cached_str, buffers_str
            ),
            Point::new(1.0, *current_y),
            dim_style,
        );
        canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
        *current_y += 1.0;

        let mem_box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&mem_box_bottom, Point::new(1.0, *current_y), dim_style);
        *current_y += 2.0;
    }

    #[allow(clippy::too_many_arguments)]
    fn render_gpu_box(
        canvas: &mut DirectTerminalCanvas,
        current_y: &mut f32,
        hardware: &HardwareInfo,
        inner_width: usize,
        box_width: usize,
        theme: &Theme,
        dim_style: &TextStyle,
    ) {
        let gpu_box_top = format!("┌─ GPU {}┐", "─".repeat(inner_width.saturating_sub(7)));
        canvas.draw_text(&gpu_box_top, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        // GPU name
        if let Some(ref gpu) = hardware.gpu_name {
            let gpu_str = format!(
                "│ {:width$} │",
                gpu.chars()
                    .take(inner_width.saturating_sub(4))
                    .collect::<String>(),
                width = inner_width.saturating_sub(2)
            );
            canvas.draw_text(
                &gpu_str,
                Point::new(1.0, *current_y),
                &TextStyle {
                    color: theme.gpu_color(50.0),
                    ..Default::default()
                },
            );
        }
        *current_y += 1.0;

        // GPU status hint
        canvas.draw_text(
            "│ Status: Ready for CUDA workloads ",
            Point::new(1.0, *current_y),
            dim_style,
        );
        canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
        *current_y += 1.0;

        let gpu_box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&gpu_box_bottom, Point::new(1.0, *current_y), dim_style);
        *current_y += 2.0;
    }

    fn render_network_box(
        canvas: &mut DirectTerminalCanvas,
        current_y: &mut f32,
        metrics: &LoadMetrics,
        inner_width: usize,
        box_width: usize,
        dim_style: &TextStyle,
    ) {
        let net_box_top = format!("┌─ Network {}┐", "─".repeat(inner_width.saturating_sub(11)));
        canvas.draw_text(&net_box_top, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        let rx_str = NetworkMetrics::format_rate(metrics.network.rx_rate);
        let tx_str = NetworkMetrics::format_rate(metrics.network.tx_rate);
        canvas.draw_text(
            &format!("│ ↓ RX: {:>12}  ↑ TX: {:>12} ", rx_str, tx_str),
            Point::new(1.0, *current_y),
            dim_style,
        );
        canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
        *current_y += 1.0;

        let net_box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&net_box_bottom, Point::new(1.0, *current_y), dim_style);
        *current_y += 2.0;
    }

    #[allow(clippy::too_many_arguments)]
    fn render_disk_box(
        canvas: &mut DirectTerminalCanvas,
        current_y: &mut f32,
        height: u16,
        metrics: &LoadMetrics,
        inner_width: usize,
        box_width: usize,
        theme: &Theme,
        dim_style: &TextStyle,
    ) {
        let disk_box_top = format!("┌─ Disks {}┐", "─".repeat(inner_width.saturating_sub(10)));
        canvas.draw_text(&disk_box_top, Point::new(1.0, *current_y), dim_style);
        *current_y += 1.0;

        for disk in metrics.disks.iter().take(3) {
            let used_str = DiskMetrics::format_bytes(disk.used_bytes);
            let total_str = DiskMetrics::format_bytes(disk.total_bytes);
            let mount_short: String = disk.mount.chars().take(15).collect();
            let disk_bar = Self::make_mini_bar(disk.usage_percent, 100.0, 10);

            canvas.draw_text(
                &format!("│ {:15} ", mount_short),
                Point::new(1.0, *current_y),
                dim_style,
            );
            canvas.draw_text(
                &disk_bar,
                Point::new(18.0, *current_y),
                &TextStyle {
                    color: theme.memory_color(disk.usage_percent),
                    ..Default::default()
                },
            );
            canvas.draw_text(
                &format!(
                    " {:>6}/{:>6} ({:.0}%)",
                    used_str, total_str, disk.usage_percent
                ),
                Point::new(29.0, *current_y),
                dim_style,
            );
            canvas.draw_text("│", Point::new(box_width as f32, *current_y), dim_style);
            *current_y += 1.0;

            if *current_y > height as f32 - 5.0 {
                break;
            }
        }

        let disk_box_bottom = format!("└{}┘", "─".repeat(inner_width));
        canvas.draw_text(&disk_box_bottom, Point::new(1.0, *current_y), dim_style);
        *current_y += 2.0;
    }

    #[allow(clippy::too_many_arguments)]
    fn render_sparklines(
        canvas: &mut DirectTerminalCanvas,
        current_y: &mut f32,
        height: u16,
        cpu_data: &[f64],
        bricks_data: &[f64],
        inner_width: usize,
        box_width: usize,
        sparkline_width: usize,
        theme: &Theme,
        dim_style: &TextStyle,
        accent_style: &TextStyle,
    ) {
        // PMAT-012 UI-05: Braille graphs for higher resolution sparklines
        if !cpu_data.is_empty() && *current_y < height as f32 - 8.0 {
            let spark_box_top = format!(
                "┌─ CPU History (braille) {}┐",
                "─".repeat(inner_width.saturating_sub(23))
            );
            canvas.draw_text(&spark_box_top, Point::new(1.0, *current_y), dim_style);
            *current_y += 1.0;

            // Use braille for higher resolution (2x data density)
            let braille_line = Self::make_braille_sparkline(cpu_data, sparkline_width);
            canvas.draw_text("│ ", Point::new(1.0, *current_y), dim_style);
            canvas.draw_text(
                &braille_line,
                Point::new(3.0, *current_y),
                &TextStyle {
                    color: theme.cpu.sample(0.3),
                    ..Default::default()
                },
            );
            canvas.draw_text(" │", Point::new(box_width as f32, *current_y), dim_style);
            *current_y += 1.0;

            let spark_box_bottom = format!("└{}┘", "─".repeat(inner_width));
            canvas.draw_text(&spark_box_bottom, Point::new(1.0, *current_y), dim_style);
            *current_y += 2.0;
        }

        // Bricks/sec braille sparkline
        if !bricks_data.is_empty() && *current_y < height as f32 - 5.0 {
            let brick_box_top = format!(
                "┌─ Bricks/sec (braille) {}┐",
                "─".repeat(inner_width.saturating_sub(24))
            );
            canvas.draw_text(&brick_box_top, Point::new(1.0, *current_y), dim_style);
            *current_y += 1.0;

            let braille_line = Self::make_braille_sparkline(bricks_data, sparkline_width);
            canvas.draw_text("│ ", Point::new(1.0, *current_y), dim_style);
            canvas.draw_text(&braille_line, Point::new(3.0, *current_y), accent_style);
            canvas.draw_text(" │", Point::new(box_width as f32, *current_y), dim_style);
            *current_y += 1.0;

            let brick_box_bottom = format!("└{}┘", "─".repeat(inner_width));
            canvas.draw_text(&brick_box_bottom, Point::new(1.0, *current_y), dim_style);
        }
    }

    pub(super) fn make_bar(value: f64, max: f64, width: usize) -> String {
        let filled = ((value / max) * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);
        format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
    }

    /// PMAT-012 UI-02: Mini bar for per-core CPU display (no brackets, compact)
    pub(super) fn make_mini_bar(value: f64, max: f64, width: usize) -> String {
        let filled = ((value / max) * width as f64).round() as usize;
        let empty = width.saturating_sub(filled);
        format!("{}{}", "▓".repeat(filled), "░".repeat(empty))
    }

    pub(super) fn make_sparkline(data: &[f64], width: usize) -> String {
        const CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

        if data.is_empty() {
            return " ".repeat(width);
        }

        let max = data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1.0);
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);
        let range = (max - min).max(1.0);

        // Sample data to fit width
        let step = data.len().max(1) as f64 / width as f64;
        let mut result = String::with_capacity(width);

        for i in 0..width {
            let idx = (i as f64 * step) as usize;
            if idx < data.len() {
                let normalized = (data[idx] - min) / range;
                let char_idx = (normalized * 7.0).round() as usize;
                result.push(CHARS[char_idx.min(7)]);
            } else {
                result.push(' ');
            }
        }

        result
    }

    /// PMAT-012 UI-05: Braille graph for higher resolution sparklines
    /// Uses Unicode braille patterns (U+2800-U+28FF) - each character encodes 2 columns x 4 rows
    pub(super) fn make_braille_sparkline(data: &[f64], width: usize) -> String {
        if data.is_empty() {
            return " ".repeat(width);
        }

        let max = data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(1.0);
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min).min(0.0);
        let range = (max - min).max(1.0);

        // Braille dot positions (column 1: bits 0,1,2,6; column 2: bits 3,4,5,7)
        // Dots are numbered: 1,2,3,7 (left col), 4,5,6,8 (right col)
        // Bit mapping: dot1=0x01, dot2=0x02, dot3=0x04, dot4=0x08,
        //              dot5=0x10, dot6=0x20, dot7=0x40, dot8=0x80
        const COL1_DOTS: [u8; 4] = [0x40, 0x04, 0x02, 0x01]; // bottom to top
        const COL2_DOTS: [u8; 4] = [0x80, 0x20, 0x10, 0x08]; // bottom to top

        let mut result = String::with_capacity(width);
        let points_per_char = 2;
        let step = data.len().max(1) as f64 / (width * points_per_char) as f64;

        for i in 0..width {
            let mut pattern: u8 = 0;

            // Left column (first data point)
            let idx1 = ((i * 2) as f64 * step) as usize;
            if idx1 < data.len() {
                let normalized = (data[idx1] - min) / range;
                let dots = (normalized * 4.0).round() as usize;
                for d in 0..dots.min(4) {
                    pattern |= COL1_DOTS[d];
                }
            }

            // Right column (second data point)
            let idx2 = ((i * 2 + 1) as f64 * step) as usize;
            if idx2 < data.len() {
                let normalized = (data[idx2] - min) / range;
                let dots = (normalized * 4.0).round() as usize;
                for d in 0..dots.min(4) {
                    pattern |= COL2_DOTS[d];
                }
            }

            // Braille base is U+2800
            result.push(char::from_u32(0x2800 + pattern as u32).unwrap_or(' '));
        }

        result
    }

    pub(super) fn format_number(n: u64) -> String {
        if n >= 1_000_000_000 {
            format!("{:.2}B", n as f64 / 1_000_000_000.0)
        } else if n >= 1_000_000 {
            format!("{:.2}M", n as f64 / 1_000_000.0)
        } else if n >= 1_000 {
            format!("{:.2}K", n as f64 / 1_000.0)
        } else {
            n.to_string()
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_status_bar(
        canvas: &mut DirectTerminalCanvas,
        width: u16,
        height: u16,
        is_running: bool,
        intensity: f64,
        backend: ComputeBackend,
        show_fps: bool,
        frame_avg: f64,
        metrics: &LoadMetrics,
        theme: &Theme,
    ) {
        let y = height as f32 - 1.0;
        let dim_style = TextStyle {
            color: theme.dim,
            ..Default::default()
        };

        // Load status
        let status = if is_running { "●RUN" } else { "○OFF" };
        let status_color = if is_running {
            theme.cpu.sample(0.0)
        } else {
            theme.dim
        };
        canvas.draw_text(
            &format!(" {} ", status),
            Point::new(0.0, y),
            &TextStyle {
                color: status_color,
                ..Default::default()
            },
        );

        // Backend
        canvas.draw_text(&format!("│ {:?} ", backend), Point::new(6.0, y), &dim_style);

        // Intensity
        canvas.draw_text(
            &format!("│ Int:{:.0}% ", intensity * 100.0),
            Point::new(16.0, y),
            &dim_style,
        );

        // Real metrics in status bar
        canvas.draw_text(
            &format!(
                "│ {:.0} brick/s │ {:.1}μs ",
                metrics.bricks_per_second, metrics.avg_latency_us
            ),
            Point::new(28.0, y),
            &TextStyle {
                color: theme.foreground,
                ..Default::default()
            },
        );

        // PMAT-012 UI-06: GFLOP/s in status bar
        let gflops = metrics.ops_per_second / 1_000_000_000.0;
        canvas.draw_text(
            &format!("│ {:.2} GFLOP/s ", gflops),
            Point::new(55.0, y),
            &TextStyle {
                color: theme.cpu.sample(0.3),
                ..Default::default()
            },
        );

        // FPS
        if show_fps || frame_avg > 0.0 {
            let fps = if frame_avg > 0.0 {
                1000.0 / frame_avg
            } else {
                0.0
            };
            canvas.draw_text(
                &format!("│ {:.0} FPS", fps),
                Point::new(width as f32 - 10.0, y),
                &dim_style,
            );
        }
    }
}
