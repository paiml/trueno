//! Visual Regression Testing (Genchi Genbutsu: Go and See)
//!
//! Provides pixel-perfect validation of compute outputs through
//! heatmap rendering and golden baseline comparison.

use std::path::PathBuf;

/// RGB color for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb {
    /// Red component
    pub r: u8,
    /// Green component
    pub g: u8,
    /// Blue component
    pub b: u8,
}

impl Rgb {
    /// Create new RGB color
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Magenta for NaN values
    pub const NAN_COLOR: Self = Self::new(255, 0, 255);
    /// White for +Infinity
    pub const INF_COLOR: Self = Self::new(255, 255, 255);
    /// Black for -Infinity
    pub const NEG_INF_COLOR: Self = Self::new(0, 0, 0);
}

/// Color palette for heatmap rendering
#[derive(Debug, Clone)]
pub struct ColorPalette {
    pub(crate) colors: Vec<Rgb>,
}

impl Default for ColorPalette {
    fn default() -> Self {
        Self::viridis()
    }
}

impl ColorPalette {
    /// Viridis colorblind-friendly palette
    #[must_use]
    pub fn viridis() -> Self {
        Self {
            colors: vec![
                Rgb::new(68, 1, 84),
                Rgb::new(59, 82, 139),
                Rgb::new(33, 145, 140),
                Rgb::new(94, 201, 98),
                Rgb::new(253, 231, 37),
            ],
        }
    }

    /// Grayscale palette
    #[must_use]
    pub fn grayscale() -> Self {
        Self {
            colors: vec![
                Rgb::new(0, 0, 0),
                Rgb::new(128, 128, 128),
                Rgb::new(255, 255, 255),
            ],
        }
    }

    /// Interpolate color at position t (0.0 to 1.0)
    #[must_use]
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    pub fn interpolate(&self, t: f32) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        let n = self.colors.len() - 1;
        let idx = (t * n as f32).floor() as usize;
        let idx = idx.min(n - 1);
        let local_t = t * n as f32 - idx as f32;

        let c1 = &self.colors[idx];
        let c2 = &self.colors[idx + 1];

        Rgb {
            r: (c1.r as f32 + (c2.r as f32 - c1.r as f32) * local_t) as u8,
            g: (c1.g as f32 + (c2.g as f32 - c1.g as f32) * local_t) as u8,
            b: (c1.b as f32 + (c2.b as f32 - c1.b as f32) * local_t) as u8,
        }
    }
}

/// Visual regression test configuration (Genchi Genbutsu)
#[derive(Debug, Clone)]
pub struct VisualRegressionConfig {
    /// Golden baseline directory
    pub golden_dir: PathBuf,
    /// Output directory for test results
    pub output_dir: PathBuf,
    /// Maximum allowed different pixels (percentage)
    pub max_diff_pct: f64,
    /// Color palette for visualization
    pub palette: ColorPalette,
}

impl Default for VisualRegressionConfig {
    fn default() -> Self {
        Self {
            golden_dir: PathBuf::from("golden"),
            output_dir: PathBuf::from("test_output"),
            max_diff_pct: 0.0, // Exact match by default
            palette: ColorPalette::default(),
        }
    }
}

impl VisualRegressionConfig {
    /// Create new config with custom golden directory
    #[must_use]
    pub fn new(golden_dir: impl Into<PathBuf>) -> Self {
        Self {
            golden_dir: golden_dir.into(),
            ..Default::default()
        }
    }

    /// Set output directory
    #[must_use]
    pub fn with_output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = dir.into();
        self
    }

    /// Set maximum diff percentage
    #[must_use]
    pub const fn with_max_diff_pct(mut self, pct: f64) -> Self {
        self.max_diff_pct = pct;
        self
    }

    /// Set color palette
    #[must_use]
    pub fn with_palette(mut self, palette: ColorPalette) -> Self {
        self.palette = palette;
        self
    }
}

/// Pixel diff result for visual regression testing
#[derive(Debug, Clone)]
pub struct PixelDiffResult {
    /// Number of pixels that differ
    pub different_pixels: usize,
    /// Total number of pixels
    pub total_pixels: usize,
    /// Maximum color difference found
    pub max_diff: u32,
}

impl PixelDiffResult {
    /// Calculate percentage of different pixels
    #[must_use]
    pub fn diff_percentage(&self) -> f64 {
        if self.total_pixels == 0 {
            0.0
        } else {
            (self.different_pixels as f64 / self.total_pixels as f64) * 100.0
        }
    }

    /// Check if images match within threshold
    #[must_use]
    pub fn matches(&self, threshold_pct: f64) -> bool {
        self.diff_percentage() <= threshold_pct
    }

    /// Create a passing result (no differences)
    #[must_use]
    pub const fn pass(total_pixels: usize) -> Self {
        Self {
            different_pixels: 0,
            total_pixels,
            max_diff: 0,
        }
    }
}

/// Simple buffer renderer for SIMD output visualization
///
/// Converts f32 buffers to raw RGBA bytes for testing
#[derive(Debug, Clone)]
pub struct BufferRenderer {
    palette: ColorPalette,
    pub(crate) range: Option<(f32, f32)>,
}

impl Default for BufferRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferRenderer {
    /// Create renderer with auto-normalization
    #[must_use]
    pub fn new() -> Self {
        Self {
            palette: ColorPalette::default(),
            range: None,
        }
    }

    /// Set fixed range for normalization
    #[must_use]
    pub const fn with_range(mut self, min: f32, max: f32) -> Self {
        self.range = Some((min, max));
        self
    }

    /// Set color palette
    #[must_use]
    pub fn with_palette(mut self, palette: ColorPalette) -> Self {
        self.palette = palette;
        self
    }

    /// Render f32 buffer to raw RGBA bytes
    ///
    /// Returns Vec<u8> with RGBA pixels (4 bytes per pixel)
    #[must_use]
    pub fn render_to_rgba(&self, buffer: &[f32], width: u32, height: u32) -> Vec<u8> {
        assert_eq!(buffer.len(), (width * height) as usize);

        let (min_val, max_val) = self.range.unwrap_or_else(|| {
            let valid: Vec<f32> = buffer.iter().copied().filter(|v| v.is_finite()).collect();
            if valid.is_empty() {
                (0.0, 1.0)
            } else {
                let min = valid.iter().copied().fold(f32::INFINITY, f32::min);
                let max = valid.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                (min, max.max(min + f32::EPSILON))
            }
        });

        let mut rgba = Vec::with_capacity(buffer.len() * 4);

        for &value in buffer {
            let color = if value.is_nan() {
                Rgb::NAN_COLOR
            } else if value.is_infinite() {
                if value > 0.0 {
                    Rgb::INF_COLOR
                } else {
                    Rgb::NEG_INF_COLOR
                }
            } else {
                let t = (value - min_val) / (max_val - min_val);
                self.palette.interpolate(t)
            };

            rgba.push(color.r);
            rgba.push(color.g);
            rgba.push(color.b);
            rgba.push(255); // Alpha
        }

        rgba
    }

    /// Compare two RGBA buffers and return diff result
    #[must_use]
    pub fn compare_rgba(&self, a: &[u8], b: &[u8], tolerance: u8) -> PixelDiffResult {
        if a == b {
            return PixelDiffResult::pass(a.len() / 4);
        }

        let min_len = a.len().min(b.len());
        let mut different = 0;
        let mut max_diff: u32 = 0;

        // Compare pixels (4 bytes each: RGBA)
        for i in (0..min_len).step_by(4) {
            let mut pixel_diff = false;
            for j in 0..4 {
                if i + j < min_len {
                    let diff = (a[i + j] as i32 - b[i + j] as i32).unsigned_abs();
                    if diff > tolerance as u32 {
                        pixel_diff = true;
                        max_diff = max_diff.max(diff);
                    }
                }
            }
            if pixel_diff {
                different += 1;
            }
        }

        // Count size difference as pixel differences
        if a.len() != b.len() {
            different += a.len().abs_diff(b.len()) / 4;
        }

        PixelDiffResult {
            different_pixels: different,
            total_pixels: min_len.max(a.len()).max(b.len()) / 4,
            max_diff,
        }
    }
}

/// Golden baseline manager for visual regression testing
#[derive(Debug, Clone)]
pub struct GoldenBaseline {
    config: VisualRegressionConfig,
}

impl GoldenBaseline {
    /// Create new golden baseline manager
    #[must_use]
    pub fn new(config: VisualRegressionConfig) -> Self {
        Self { config }
    }

    /// Get path for a golden baseline file
    #[must_use]
    pub fn golden_path(&self, name: &str) -> PathBuf {
        self.config.golden_dir.join(format!("{name}.golden"))
    }

    /// Get path for an output file
    #[must_use]
    pub fn output_path(&self, name: &str) -> PathBuf {
        self.config.output_dir.join(format!("{name}.output"))
    }

    /// Get the config
    #[must_use]
    pub const fn config(&self) -> &VisualRegressionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_color_creation() {
        let color = Rgb::new(255, 128, 64);
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 128);
        assert_eq!(color.b, 64);
    }

    #[test]
    fn test_rgb_special_colors() {
        assert_eq!(Rgb::NAN_COLOR, Rgb::new(255, 0, 255));
        assert_eq!(Rgb::INF_COLOR, Rgb::new(255, 255, 255));
        assert_eq!(Rgb::NEG_INF_COLOR, Rgb::new(0, 0, 0));
    }

    #[test]
    fn test_color_palette_viridis() {
        let palette = ColorPalette::viridis();
        assert_eq!(palette.colors.len(), 5);

        // Test interpolation at boundaries
        let at_0 = palette.interpolate(0.0);
        let at_1 = palette.interpolate(1.0);

        // Viridis starts dark purple, ends yellow
        assert_eq!(at_0, Rgb::new(68, 1, 84));
        assert_eq!(at_1, Rgb::new(253, 231, 37));
    }

    #[test]
    fn test_color_palette_grayscale() {
        let palette = ColorPalette::grayscale();
        assert_eq!(palette.colors.len(), 3);

        let at_0 = palette.interpolate(0.0);
        let at_1 = palette.interpolate(1.0);

        assert_eq!(at_0, Rgb::new(0, 0, 0));
        assert_eq!(at_1, Rgb::new(255, 255, 255));
    }

    #[test]
    fn test_color_palette_interpolation_midpoint() {
        let palette = ColorPalette::grayscale();
        let at_mid = palette.interpolate(0.5);

        // Should be close to gray
        assert_eq!(at_mid, Rgb::new(128, 128, 128));
    }

    #[test]
    fn test_color_palette_clamping() {
        let palette = ColorPalette::viridis();

        // Values outside [0, 1] should be clamped
        let at_neg = palette.interpolate(-0.5);
        let at_over = palette.interpolate(1.5);

        assert_eq!(at_neg, palette.interpolate(0.0));
        assert_eq!(at_over, palette.interpolate(1.0));
    }

    #[test]
    fn test_visual_regression_config_default() {
        let config = VisualRegressionConfig::default();

        assert_eq!(config.golden_dir, PathBuf::from("golden"));
        assert_eq!(config.output_dir, PathBuf::from("test_output"));
        assert_eq!(config.max_diff_pct, 0.0);
    }

    #[test]
    fn test_visual_regression_config_builder() {
        let config = VisualRegressionConfig::new("my_golden")
            .with_output_dir("my_output")
            .with_max_diff_pct(1.5)
            .with_palette(ColorPalette::grayscale());

        assert_eq!(config.golden_dir, PathBuf::from("my_golden"));
        assert_eq!(config.output_dir, PathBuf::from("my_output"));
        assert_eq!(config.max_diff_pct, 1.5);
    }

    #[test]
    fn test_pixel_diff_result_percentage() {
        let result = PixelDiffResult {
            different_pixels: 10,
            total_pixels: 100,
            max_diff: 50,
        };

        assert_eq!(result.diff_percentage(), 10.0);
        assert!(!result.matches(5.0));
        assert!(result.matches(10.0));
        assert!(result.matches(15.0));
    }

    #[test]
    fn test_pixel_diff_result_zero_total() {
        let result = PixelDiffResult {
            different_pixels: 0,
            total_pixels: 0,
            max_diff: 0,
        };

        assert_eq!(result.diff_percentage(), 0.0);
    }

    #[test]
    fn test_pixel_diff_result_pass() {
        let result = PixelDiffResult::pass(100);

        assert_eq!(result.different_pixels, 0);
        assert_eq!(result.total_pixels, 100);
        assert_eq!(result.max_diff, 0);
        assert!(result.matches(0.0));
    }

    #[test]
    fn test_buffer_renderer_default() {
        let renderer = BufferRenderer::default();
        assert!(renderer.range.is_none());
    }

    #[test]
    fn test_buffer_renderer_with_range() {
        let renderer = BufferRenderer::new().with_range(0.0, 10.0);
        assert_eq!(renderer.range, Some((0.0, 10.0)));
    }

    #[test]
    fn test_buffer_renderer_with_palette() {
        let renderer = BufferRenderer::new().with_palette(ColorPalette::grayscale());
        assert_eq!(renderer.palette.colors.len(), 3);
    }

    #[test]
    fn test_buffer_renderer_rgba_output() {
        let renderer = BufferRenderer::new();
        let buffer: Vec<f32> = (0..4).map(|i| i as f32 / 3.0).collect();
        let rgba = renderer.render_to_rgba(&buffer, 2, 2);

        // 4 pixels * 4 bytes = 16 bytes
        assert_eq!(rgba.len(), 16);

        // Check alpha channel is always 255
        for i in (3..16).step_by(4) {
            assert_eq!(rgba[i], 255);
        }
    }

    #[test]
    fn test_buffer_renderer_nan_handling() {
        let renderer = BufferRenderer::new();
        let buffer = vec![0.0, f32::NAN, 1.0, 0.5];
        let rgba = renderer.render_to_rgba(&buffer, 2, 2);

        // Second pixel should be NAN_COLOR (magenta: 255, 0, 255)
        assert_eq!(rgba[4], 255); // R
        assert_eq!(rgba[5], 0); // G
        assert_eq!(rgba[6], 255); // B
        assert_eq!(rgba[7], 255); // A
    }

    #[test]
    fn test_buffer_renderer_inf_handling() {
        let renderer = BufferRenderer::new();
        let buffer = vec![f32::INFINITY, f32::NEG_INFINITY, 0.5, 0.5];
        let rgba = renderer.render_to_rgba(&buffer, 2, 2);

        // First pixel: +INF should be white
        assert_eq!(rgba[0], 255);
        assert_eq!(rgba[1], 255);
        assert_eq!(rgba[2], 255);

        // Second pixel: -INF should be black
        assert_eq!(rgba[4], 0);
        assert_eq!(rgba[5], 0);
        assert_eq!(rgba[6], 0);
    }

    #[test]
    fn test_buffer_renderer_compare_identical() {
        let renderer = BufferRenderer::new();
        let buffer: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let rgba = renderer.render_to_rgba(&buffer, 4, 4);

        let result = renderer.compare_rgba(&rgba, &rgba, 0);
        assert_eq!(result.different_pixels, 0);
        assert!(result.matches(0.0));
    }

    #[test]
    fn test_buffer_renderer_compare_different() {
        let renderer = BufferRenderer::new();
        let buffer_a: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let buffer_b: Vec<f32> = (0..16).map(|i| 1.0 - i as f32 / 15.0).collect();

        let rgba_a = renderer.render_to_rgba(&buffer_a, 4, 4);
        let rgba_b = renderer.render_to_rgba(&buffer_b, 4, 4);

        let result = renderer.compare_rgba(&rgba_a, &rgba_b, 0);
        assert!(result.different_pixels > 0);
    }

    #[test]
    fn test_buffer_renderer_compare_with_tolerance() {
        let renderer = BufferRenderer::new();
        let rgba_a = vec![100, 100, 100, 255];
        let rgba_b = vec![105, 102, 98, 255];

        // With tolerance 10, should match
        let result = renderer.compare_rgba(&rgba_a, &rgba_b, 10);
        assert_eq!(result.different_pixels, 0);

        // With tolerance 1, should differ
        let result_strict = renderer.compare_rgba(&rgba_a, &rgba_b, 1);
        assert!(result_strict.different_pixels > 0);
    }

    #[test]
    fn test_golden_baseline_paths() {
        let config = VisualRegressionConfig::new("/test/golden").with_output_dir("/test/output");
        let baseline = GoldenBaseline::new(config);

        assert_eq!(
            baseline.golden_path("relu_4x4"),
            PathBuf::from("/test/golden/relu_4x4.golden")
        );
        assert_eq!(
            baseline.output_path("relu_4x4"),
            PathBuf::from("/test/output/relu_4x4.output")
        );
    }

    #[test]
    fn test_golden_baseline_config_access() {
        let config = VisualRegressionConfig::new("/golden").with_max_diff_pct(2.5);
        let baseline = GoldenBaseline::new(config);

        assert_eq!(baseline.config().max_diff_pct, 2.5);
    }
}
