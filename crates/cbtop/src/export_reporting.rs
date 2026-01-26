//! Multi-Format Export System (PMAT-036)
//!
//! Unified export system for benchmark results and analysis reports.
//!
//! # Features
//!
//! - Multiple export formats: JSON, CSV, Markdown, HTML
//! - Report types: Benchmark, Comparison, Regression, Summary
//! - File write operations
//! - Fluent report builder API
//!
//! # Falsification Criteria (F1281-F1290)
//!
//! See `tests/export_reporting_f1281.rs` for falsification tests.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

/// Export format type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// JSON format for API/CI integration
    Json,
    /// CSV format for spreadsheet analysis
    Csv,
    /// Markdown format for documentation
    Markdown,
    /// HTML format for interactive reports
    Html,
}

impl ExportFormat {
    /// Get format name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Csv => "csv",
            Self::Markdown => "markdown",
            Self::Html => "html",
        }
    }

    /// Get file extension
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Csv => "csv",
            Self::Markdown => "md",
            Self::Html => "html",
        }
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "json" => Some(Self::Json),
            "csv" => Some(Self::Csv),
            "markdown" | "md" => Some(Self::Markdown),
            "html" => Some(Self::Html),
            _ => None,
        }
    }
}

/// Report type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportType {
    /// Benchmark results report
    Benchmark,
    /// Comparison report (baseline vs current)
    Comparison,
    /// Regression detection report
    Regression,
    /// Executive summary report
    Summary,
}

impl ReportType {
    /// Get report type name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Benchmark => "benchmark",
            Self::Comparison => "comparison",
            Self::Regression => "regression",
            Self::Summary => "summary",
        }
    }
}

/// Benchmark metric data
#[derive(Debug, Clone)]
pub struct BenchmarkMetric {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Unit of measurement
    pub unit: String,
}

impl BenchmarkMetric {
    /// Create new metric
    pub fn new(name: &str, value: f64, unit: &str) -> Self {
        Self {
            name: name.to_string(),
            value,
            unit: unit.to_string(),
        }
    }
}

/// Benchmark report data
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    /// Report title
    pub title: String,
    /// Timestamp
    pub timestamp: String,
    /// Workload name
    pub workload: String,
    /// Backend used
    pub backend: String,
    /// Problem size
    pub size: usize,
    /// Metrics
    pub metrics: Vec<BenchmarkMetric>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Default for BenchmarkReport {
    fn default() -> Self {
        Self {
            title: "Benchmark Report".to_string(),
            timestamp: "".to_string(),
            workload: "".to_string(),
            backend: "".to_string(),
            size: 0,
            metrics: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

impl BenchmarkReport {
    /// Create new benchmark report
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            ..Default::default()
        }
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, ts: &str) -> Self {
        self.timestamp = ts.to_string();
        self
    }

    /// Set workload
    pub fn with_workload(mut self, workload: &str) -> Self {
        self.workload = workload.to_string();
        self
    }

    /// Set backend
    pub fn with_backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_string();
        self
    }

    /// Set size
    pub fn with_size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }

    /// Add metric
    pub fn add_metric(&mut self, name: &str, value: f64, unit: &str) {
        self.metrics.push(BenchmarkMetric::new(name, value, unit));
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
}

/// Comparison entry (baseline vs current)
#[derive(Debug, Clone)]
pub struct ComparisonEntry {
    /// Metric name
    pub metric: String,
    /// Baseline value
    pub baseline: f64,
    /// Current value
    pub current: f64,
    /// Unit
    pub unit: String,
}

impl ComparisonEntry {
    /// Create new comparison entry
    pub fn new(metric: &str, baseline: f64, current: f64, unit: &str) -> Self {
        Self {
            metric: metric.to_string(),
            baseline,
            current,
            unit: unit.to_string(),
        }
    }

    /// Calculate percent change
    pub fn percent_change(&self) -> f64 {
        if self.baseline.abs() < 1e-10 {
            if self.current.abs() < 1e-10 {
                0.0
            } else {
                100.0
            }
        } else {
            ((self.current - self.baseline) / self.baseline) * 100.0
        }
    }

    /// Check if regressed (positive change is bad for latency, negative is bad for throughput)
    pub fn is_regression(&self, threshold_percent: f64) -> bool {
        self.percent_change().abs() > threshold_percent
    }
}

/// Comparison report data
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// Report title
    pub title: String,
    /// Baseline label
    pub baseline_label: String,
    /// Current label
    pub current_label: String,
    /// Comparison entries
    pub entries: Vec<ComparisonEntry>,
    /// Regression threshold (%)
    pub regression_threshold: f64,
}

impl Default for ComparisonReport {
    fn default() -> Self {
        Self {
            title: "Comparison Report".to_string(),
            baseline_label: "baseline".to_string(),
            current_label: "current".to_string(),
            entries: Vec::new(),
            regression_threshold: 5.0,
        }
    }
}

impl ComparisonReport {
    /// Create new comparison report
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            ..Default::default()
        }
    }

    /// Set labels
    pub fn with_labels(mut self, baseline: &str, current: &str) -> Self {
        self.baseline_label = baseline.to_string();
        self.current_label = current.to_string();
        self
    }

    /// Set regression threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.regression_threshold = threshold;
        self
    }

    /// Add comparison entry
    pub fn add_entry(&mut self, metric: &str, baseline: f64, current: f64, unit: &str) {
        self.entries
            .push(ComparisonEntry::new(metric, baseline, current, unit));
    }

    /// Get regressions
    pub fn get_regressions(&self) -> Vec<&ComparisonEntry> {
        self.entries
            .iter()
            .filter(|e| e.is_regression(self.regression_threshold))
            .collect()
    }

    /// Check if any regressions exist
    pub fn has_regressions(&self) -> bool {
        !self.get_regressions().is_empty()
    }
}

/// Report exporter
#[derive(Debug)]
pub struct ReportExporter;

impl ReportExporter {
    /// Export benchmark report to JSON
    pub fn benchmark_to_json(report: &BenchmarkReport) -> String {
        let metrics_json: Vec<String> = report
            .metrics
            .iter()
            .map(|m| {
                format!(
                    r#"{{"name":"{}","value":{},"unit":"{}"}}"#,
                    m.name, m.value, m.unit
                )
            })
            .collect();

        let metadata_json: Vec<String> = report
            .metadata
            .iter()
            .map(|(k, v)| format!(r#""{}":"{}""#, k, v))
            .collect();

        format!(
            r#"{{"title":"{}","timestamp":"{}","workload":"{}","backend":"{}","size":{},"metrics":[{}],"metadata":{{{}}}}}"#,
            report.title,
            report.timestamp,
            report.workload,
            report.backend,
            report.size,
            metrics_json.join(","),
            metadata_json.join(",")
        )
    }

    /// Export benchmark report to CSV
    pub fn benchmark_to_csv(report: &BenchmarkReport) -> String {
        let mut lines = vec!["metric,value,unit".to_string()];
        for m in &report.metrics {
            lines.push(format!("{},{},{}", m.name, m.value, m.unit));
        }
        lines.join("\n")
    }

    /// Export benchmark report to Markdown
    pub fn benchmark_to_markdown(report: &BenchmarkReport) -> String {
        let mut md = format!("# {}\n\n", report.title);
        md.push_str(&format!("**Workload**: {}  \n", report.workload));
        md.push_str(&format!("**Backend**: {}  \n", report.backend));
        md.push_str(&format!("**Size**: {}  \n\n", report.size));

        md.push_str("## Metrics\n\n");
        md.push_str("| Metric | Value | Unit |\n");
        md.push_str("|--------|-------|------|\n");
        for m in &report.metrics {
            md.push_str(&format!("| {} | {:.4} | {} |\n", m.name, m.value, m.unit));
        }

        md
    }

    /// Export benchmark report to HTML
    pub fn benchmark_to_html(report: &BenchmarkReport) -> String {
        let mut html = String::from("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!("<title>{}</title>\n", report.title));
        html.push_str("<style>table{border-collapse:collapse;}th,td{border:1px solid #ccc;padding:8px;}</style>\n");
        html.push_str("</head>\n<body>\n");
        html.push_str(&format!("<h1>{}</h1>\n", report.title));
        html.push_str(&format!(
            "<p><strong>Workload:</strong> {}</p>\n",
            report.workload
        ));
        html.push_str(&format!(
            "<p><strong>Backend:</strong> {}</p>\n",
            report.backend
        ));
        html.push_str(&format!("<p><strong>Size:</strong> {}</p>\n", report.size));

        html.push_str("<h2>Metrics</h2>\n");
        html.push_str("<table>\n<tr><th>Metric</th><th>Value</th><th>Unit</th></tr>\n");
        for m in &report.metrics {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{:.4}</td><td>{}</td></tr>\n",
                m.name, m.value, m.unit
            ));
        }
        html.push_str("</table>\n</body>\n</html>");

        html
    }

    /// Export comparison report to JSON
    pub fn comparison_to_json(report: &ComparisonReport) -> String {
        let entries_json: Vec<String> = report
            .entries
            .iter()
            .map(|e| {
                format!(
                    r#"{{"metric":"{}","baseline":{},"current":{},"change":{:.2},"unit":"{}"}}"#,
                    e.metric,
                    e.baseline,
                    e.current,
                    e.percent_change(),
                    e.unit
                )
            })
            .collect();

        format!(
            r#"{{"title":"{}","baseline":"{}","current":"{}","has_regressions":{},"entries":[{}]}}"#,
            report.title,
            report.baseline_label,
            report.current_label,
            report.has_regressions(),
            entries_json.join(",")
        )
    }

    /// Export comparison report to Markdown
    pub fn comparison_to_markdown(report: &ComparisonReport) -> String {
        let mut md = format!("# {}\n\n", report.title);
        md.push_str(&format!(
            "Comparing **{}** vs **{}**\n\n",
            report.baseline_label, report.current_label
        ));

        md.push_str("| Metric | Baseline | Current | Change |\n");
        md.push_str("|--------|----------|---------|--------|\n");
        for e in &report.entries {
            let change = e.percent_change();
            let indicator = if change.abs() > report.regression_threshold {
                "⚠️"
            } else {
                "✅"
            };
            md.push_str(&format!(
                "| {} | {:.4} {} | {:.4} {} | {:.2}% {} |\n",
                e.metric, e.baseline, e.unit, e.current, e.unit, change, indicator
            ));
        }

        if report.has_regressions() {
            md.push_str("\n## Regressions Detected\n\n");
            for e in report.get_regressions() {
                md.push_str(&format!(
                    "- **{}**: {:.2}% change\n",
                    e.metric,
                    e.percent_change()
                ));
            }
        }

        md
    }

    /// Export to specified format
    pub fn export_benchmark(report: &BenchmarkReport, format: ExportFormat) -> String {
        match format {
            ExportFormat::Json => Self::benchmark_to_json(report),
            ExportFormat::Csv => Self::benchmark_to_csv(report),
            ExportFormat::Markdown => Self::benchmark_to_markdown(report),
            ExportFormat::Html => Self::benchmark_to_html(report),
        }
    }

    /// Export comparison to specified format
    pub fn export_comparison(report: &ComparisonReport, format: ExportFormat) -> String {
        match format {
            ExportFormat::Json => Self::comparison_to_json(report),
            ExportFormat::Markdown => Self::comparison_to_markdown(report),
            ExportFormat::Csv | ExportFormat::Html => Self::comparison_to_json(report),
        }
    }

    /// Write report to file
    pub fn write_to_file<P: AsRef<Path>>(content: &str, path: P) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(content.as_bytes())?;
        Ok(())
    }
}

/// Report builder for fluent API
#[derive(Debug)]
pub struct ReportBuilder {
    /// Report type
    report_type: ReportType,
    /// Export format
    format: ExportFormat,
    /// Title
    title: String,
    /// Benchmark metrics
    metrics: Vec<BenchmarkMetric>,
    /// Comparison entries
    comparisons: Vec<ComparisonEntry>,
    /// Metadata
    metadata: HashMap<String, String>,
}

impl ReportBuilder {
    /// Create new builder
    pub fn new(report_type: ReportType) -> Self {
        Self {
            report_type,
            format: ExportFormat::Json,
            title: format!("{} Report", report_type.name()),
            metrics: Vec::new(),
            comparisons: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set format
    pub fn format(mut self, format: ExportFormat) -> Self {
        self.format = format;
        self
    }

    /// Set title
    pub fn title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    /// Add metric
    pub fn metric(mut self, name: &str, value: f64, unit: &str) -> Self {
        self.metrics.push(BenchmarkMetric::new(name, value, unit));
        self
    }

    /// Add comparison
    pub fn comparison(mut self, metric: &str, baseline: f64, current: f64, unit: &str) -> Self {
        self.comparisons
            .push(ComparisonEntry::new(metric, baseline, current, unit));
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Build and export
    pub fn build(self) -> String {
        match self.report_type {
            ReportType::Benchmark | ReportType::Summary => {
                let mut report = BenchmarkReport::new(&self.title);
                report.metrics = self.metrics;
                report.metadata = self.metadata;
                ReportExporter::export_benchmark(&report, self.format)
            }
            ReportType::Comparison | ReportType::Regression => {
                let mut report = ComparisonReport::new(&self.title);
                report.entries = self.comparisons;
                ReportExporter::export_comparison(&report, self.format)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_format_names() {
        assert_eq!(ExportFormat::Json.name(), "json");
        assert_eq!(ExportFormat::Csv.name(), "csv");
        assert_eq!(ExportFormat::Markdown.name(), "markdown");
        assert_eq!(ExportFormat::Html.name(), "html");
    }

    #[test]
    fn test_export_format_extensions() {
        assert_eq!(ExportFormat::Json.extension(), "json");
        assert_eq!(ExportFormat::Markdown.extension(), "md");
    }

    #[test]
    fn test_format_from_str() {
        assert_eq!(ExportFormat::parse("json"), Some(ExportFormat::Json));
        assert_eq!(ExportFormat::parse("md"), Some(ExportFormat::Markdown));
        assert_eq!(ExportFormat::parse("invalid"), None);
    }

    #[test]
    fn test_benchmark_to_json() {
        let mut report = BenchmarkReport::new("Test Report");
        report.add_metric("latency", 10.5, "ms");
        report.add_metric("throughput", 1000.0, "ops/s");

        let json = ReportExporter::benchmark_to_json(&report);
        assert!(json.contains("\"title\":\"Test Report\""));
        assert!(json.contains("\"latency\""));
        assert!(json.contains("10.5"));
    }

    #[test]
    fn test_benchmark_to_csv() {
        let mut report = BenchmarkReport::new("Test Report");
        report.add_metric("latency", 10.5, "ms");

        let csv = ReportExporter::benchmark_to_csv(&report);
        assert!(csv.contains("metric,value,unit"));
        assert!(csv.contains("latency,10.5,ms"));
    }

    #[test]
    fn test_benchmark_to_markdown() {
        let mut report = BenchmarkReport::new("Test Report");
        report.add_metric("latency", 10.5, "ms");

        let md = ReportExporter::benchmark_to_markdown(&report);
        assert!(md.contains("# Test Report"));
        assert!(md.contains("| Metric |"));
    }

    #[test]
    fn test_benchmark_to_html() {
        let mut report = BenchmarkReport::new("Test Report");
        report.add_metric("latency", 10.5, "ms");

        let html = ReportExporter::benchmark_to_html(&report);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<h1>Test Report</h1>"));
        assert!(html.contains("</html>"));
    }

    #[test]
    fn test_comparison_percent_change() {
        let entry = ComparisonEntry::new("latency", 100.0, 110.0, "ms");
        assert!((entry.percent_change() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_comparison_regression() {
        let entry = ComparisonEntry::new("latency", 100.0, 115.0, "ms");
        assert!(entry.is_regression(10.0));

        let ok_entry = ComparisonEntry::new("latency", 100.0, 103.0, "ms");
        assert!(!ok_entry.is_regression(10.0));
    }

    #[test]
    fn test_comparison_to_json() {
        let mut report = ComparisonReport::new("Test Comparison");
        report.add_entry("latency", 100.0, 110.0, "ms");

        let json = ReportExporter::comparison_to_json(&report);
        assert!(json.contains("\"title\":\"Test Comparison\""));
        assert!(json.contains("\"change\":10.00"));
    }

    #[test]
    fn test_report_builder() {
        let output = ReportBuilder::new(ReportType::Benchmark)
            .title("My Report")
            .metric("latency", 10.0, "ms")
            .format(ExportFormat::Json)
            .build();

        assert!(output.contains("My Report"));
        assert!(output.contains("latency"));
    }

    #[test]
    fn test_comparison_builder() {
        let output = ReportBuilder::new(ReportType::Comparison)
            .title("Compare")
            .comparison("latency", 100.0, 110.0, "ms")
            .format(ExportFormat::Markdown)
            .build();

        assert!(output.contains("# Compare"));
    }
}
