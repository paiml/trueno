//! GPU image processing primitives.
//!
//! # Contract: image-conv2d-v1.yaml
//!
//! Provides convolution, Gaussian blur, Sobel edge detection, and
//! Canny edge detection with provable properties.
//!
//! # Example
//!
//! ```
//! use trueno_image::{conv2d, BorderMode};
//!
//! // Identity convolution (delta kernel)
//! let image = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0_f32];
//! let delta = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0_f32];
//! let out = conv2d(&image, 3, 3, &delta, 3, 3, BorderMode::Zero).unwrap();
//! assert!((out[4] - 5.0).abs() < 1e-6); // Center pixel preserved
//! ```

mod buf;
mod color;
mod conv;
mod error;
mod histogram;
mod morphology;
mod resize;

#[cfg(test)]
mod tests;

pub use buf::{DType, ImageBuf};
pub use color::{connected_components, hsv_to_rgb, rgb_to_gray, rgb_to_hsv};
pub use conv::{
    canny, canny_rgb, conv2d, gaussian_blur, gradient_magnitude, separable_conv2d, sobel,
    BorderMode,
};
pub use error::ImageError;
pub use histogram::{cumulative_histogram, equalize, histogram};
pub use morphology::{closing, dilate, erode, opening};
pub use resize::{resize, Interpolation};

/// Image operations trait for `ImageBuf` method dispatch.
///
/// Provides a unified interface for applying image processing operations
/// to structured `ImageBuf` instances with automatic dimension handling.
pub trait ImageOps {
    /// Apply Gaussian blur with given sigma.
    fn blur(&self, sigma: f32) -> Result<ImageBuf, ImageError>;

    /// Apply Canny edge detection (converts multi-channel to grayscale).
    fn canny_edges(
        &self,
        sigma: f32,
        low: f32,
        high: f32,
    ) -> Result<ImageBuf, ImageError>;

    /// Convert to grayscale.
    fn to_gray(&self) -> Result<ImageBuf, ImageError>;
}

impl ImageOps for ImageBuf {
    fn blur(&self, sigma: f32) -> Result<ImageBuf, ImageError> {
        if self.channels() == 1 {
            let data = gaussian_blur(self.data(), self.width(), self.height(), sigma)?;
            ImageBuf::new(data, self.width(), self.height(), 1)
        } else {
            // Process each channel independently
            let npix = self.width() * self.height();
            let mut out = vec![0.0_f32; npix * self.channels()];
            for c in 0..self.channels() {
                let ch = self.channel(c)?;
                let blurred = gaussian_blur(ch.data(), self.width(), self.height(), sigma)?;
                for i in 0..npix {
                    out[i * self.channels() + c] = blurred[i];
                }
            }
            ImageBuf::new(out, self.width(), self.height(), self.channels())
        }
    }

    fn canny_edges(
        &self,
        sigma: f32,
        low: f32,
        high: f32,
    ) -> Result<ImageBuf, ImageError> {
        let edges = canny_rgb(
            self.data(),
            self.width(),
            self.height(),
            self.channels(),
            sigma,
            low,
            high,
        )?;
        ImageBuf::new(edges, self.width(), self.height(), 1)
    }

    fn to_gray(&self) -> Result<ImageBuf, ImageError> {
        if self.channels() == 1 {
            return Ok(self.clone());
        }
        let gray = rgb_to_gray(self.data(), self.width(), self.height())?;
        ImageBuf::new(gray, self.width(), self.height(), 1)
    }
}
