//! Fast Fourier Transform library with Stockham auto-sort algorithm.
//!
//! # Contract: fft-stockham-v1.yaml
//!
//! Provides 1D and 2D FFT with provable properties:
//! - Parseval energy conservation
//! - Inverse roundtrip accuracy
//! - Linearity
//!
//! # Example
//!
//! ```
//! use trueno_fft::{Complex, FftPlan};
//!
//! let plan = FftPlan::new(4).unwrap();
//! let input = [
//!     Complex::new(1.0, 0.0),
//!     Complex::new(2.0, 0.0),
//!     Complex::new(3.0, 0.0),
//!     Complex::new(4.0, 0.0),
//! ];
//! let mut output = [Complex::ZERO; 4];
//! plan.forward(&input, &mut output).unwrap();
//!
//! // Verify Parseval: Σ|x|² = (1/N)·Σ|X|²
//! let energy_time: f32 = input.iter().map(|x| x.norm_sq()).sum();
//! let energy_freq: f32 = output.iter().map(|x| x.norm_sq()).sum::<f32>() / 4.0;
//! assert!((energy_time - energy_freq).abs() < 1e-4);
//! ```

mod complex;
mod error;
mod stockham;

#[cfg(test)]
mod tests;

pub use complex::Complex;
pub use error::FftError;
pub use stockham::{fft_2d, FftPlan};
