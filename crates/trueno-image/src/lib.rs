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

mod conv;
mod error;

#[cfg(test)]
mod tests;

pub use conv::{
    canny, conv2d, gaussian_blur, gradient_magnitude, separable_conv2d, sobel, BorderMode,
};
pub use error::ImageError;
