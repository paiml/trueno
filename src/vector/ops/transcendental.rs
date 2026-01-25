//! Transcendental mathematical functions for Vector<f32>
//!
//! This module provides element-wise transcendental functions including:
//! - Exponentials: `exp`, `ln`, `log2`, `log10`
//! - Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
//! - Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

#[cfg(target_arch = "x86_64")]
use crate::backends::avx2::Avx2Backend;
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
use crate::backends::neon::NeonBackend;
use crate::backends::scalar::ScalarBackend;
#[cfg(target_arch = "x86_64")]
use crate::backends::sse2::Sse2Backend;
#[cfg(target_arch = "wasm32")]
use crate::backends::wasm::WasmBackend;
use crate::backends::VectorBackend;
use crate::vector::Vector;
use crate::{dispatch_unary_op, Backend, Result, TruenoError};

impl Vector<f32> {
    /// Element-wise exponential: result\[i\] = e^x\[i\]
    ///
    /// Computes the natural exponential (e^x) for each element.
    /// Uses Rust's optimized f32::exp() method.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, 2.0]);
    /// let result = v.exp().unwrap();
    /// // result ≈ [1.0, 2.718, 7.389]
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `exp(0.0)` returns 1.0
    /// - `exp(1.0)` returns e ≈ 2.71828
    /// - `exp(-∞)` returns 0.0
    /// - `exp(+∞)` returns +∞
    ///
    /// # Applications
    ///
    /// - Machine learning: Softmax activation, sigmoid, exponential loss
    /// - Statistics: Exponential distribution, log-normal distribution
    /// - Physics: Radioactive decay, population growth models
    /// - Signal processing: Exponential smoothing, envelope detection
    /// - Numerical methods: Solving differential equations
    pub fn exp(&self) -> Result<Vector<f32>> {
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            // Use parallel processing for large arrays
            #[cfg(feature = "parallel")]
            {
                const PARALLEL_THRESHOLD: usize = 100_000;
                const CHUNK_SIZE: usize = 65536;

                if self.len() >= PARALLEL_THRESHOLD {
                    use rayon::prelude::*;

                    self.data
                        .par_chunks(CHUNK_SIZE)
                        .zip(result_data.par_chunks_mut(CHUNK_SIZE))
                        .for_each(|(chunk_in, chunk_out)| {
                            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
                            unsafe {
                                match self.backend {
                                    Backend::Scalar => ScalarBackend::exp(chunk_in, chunk_out),
                                    #[cfg(target_arch = "x86_64")]
                                    Backend::SSE2 | Backend::AVX => {
                                        Sse2Backend::exp(chunk_in, chunk_out)
                                    }
                                    #[cfg(target_arch = "x86_64")]
                                    Backend::AVX2 | Backend::AVX512 => {
                                        Avx2Backend::exp(chunk_in, chunk_out)
                                    }
                                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                                    Backend::NEON => NeonBackend::exp(chunk_in, chunk_out),
                                    #[cfg(target_arch = "wasm32")]
                                    Backend::WasmSIMD => WasmBackend::exp(chunk_in, chunk_out),
                                    Backend::GPU => ScalarBackend::exp(chunk_in, chunk_out),
                                    Backend::Auto => ScalarBackend::exp(chunk_in, chunk_out),
                                    #[allow(unreachable_patterns)]
                                    _ => ScalarBackend::exp(chunk_in, chunk_out),
                                }
                            }
                        });

                    return Ok(Vector {
                        data: result_data,
                        backend: self.backend,
                    });
                }
            }

            // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
            unsafe {
                match self.backend {
                    Backend::Scalar => ScalarBackend::exp(&self.data, &mut result_data),
                    #[cfg(target_arch = "x86_64")]
                    Backend::SSE2 | Backend::AVX => Sse2Backend::exp(&self.data, &mut result_data),
                    #[cfg(target_arch = "x86_64")]
                    Backend::AVX2 | Backend::AVX512 => {
                        Avx2Backend::exp(&self.data, &mut result_data)
                    }
                    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                    Backend::NEON => NeonBackend::exp(&self.data, &mut result_data),
                    #[cfg(target_arch = "wasm32")]
                    Backend::WasmSIMD => WasmBackend::exp(&self.data, &mut result_data),
                    Backend::GPU => return Err(TruenoError::UnsupportedBackend(Backend::GPU)),
                    Backend::Auto => {
                        // Auto should have been resolved at creation time
                        return Err(TruenoError::UnsupportedBackend(Backend::Auto));
                    }
                    #[allow(unreachable_patterns)]
                    _ => ScalarBackend::exp(&self.data, &mut result_data),
                }
            }
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Element-wise natural logarithm: result\[i\] = ln(x\[i\])
    ///
    /// Computes the natural logarithm (base e) for each element.
    /// Uses Rust's optimized f32::ln() method.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, std::f32::consts::E, std::f32::consts::E.powi(2)]);
    /// let result = v.ln().unwrap();
    /// // result ≈ [0.0, 1.0, 2.0]
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `ln(1.0)` returns 0.0
    /// - `ln(e)` returns 1.0
    /// - `ln(x)` for x ≤ 0 returns NaN
    /// - `ln(0.0)` returns -∞
    /// - `ln(+∞)` returns +∞
    ///
    /// # Applications
    ///
    /// - Machine learning: Log loss, log-likelihood, softmax normalization
    /// - Statistics: Log-normal distribution, log transformation for skewed data
    /// - Information theory: Entropy calculation, mutual information
    /// - Economics: Log returns, elasticity calculations
    /// - Signal processing: Decibel conversion, log-frequency analysis
    pub fn ln(&self) -> Result<Vector<f32>> {
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, ln, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Element-wise base-2 logarithm: result\[i\] = log₂(x\[i\])
    ///
    /// Computes the base-2 logarithm for each element.
    /// Uses Rust's optimized f32::log2() method.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 4.0, 8.0]);
    /// let result = v.log2().unwrap();
    /// // result ≈ [0.0, 1.0, 2.0, 3.0]
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `log2(1.0)` returns 0.0
    /// - `log2(2.0)` returns 1.0
    /// - `log2(x)` for x ≤ 0 returns NaN
    /// - `log2(0.0)` returns -∞
    /// - `log2(+∞)` returns +∞
    ///
    /// # Applications
    ///
    /// - Information theory: Entropy in bits, mutual information
    /// - Computer science: Bit manipulation, binary search complexity
    /// - Audio: Octave calculations, pitch detection
    /// - Data compression: Huffman coding, arithmetic coding
    pub fn log2(&self) -> Result<Vector<f32>> {
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, log2, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Element-wise base-10 logarithm: result\[i\] = log₁₀(x\[i\])
    ///
    /// Computes the base-10 (common) logarithm for each element.
    /// Uses Rust's optimized f32::log10() method.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 10.0, 100.0, 1000.0]);
    /// let result = v.log10().unwrap();
    /// // result ≈ [0.0, 1.0, 2.0, 3.0]
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `log10(1.0)` returns 0.0
    /// - `log10(10.0)` returns 1.0
    /// - `log10(x)` for x ≤ 0 returns NaN
    /// - `log10(0.0)` returns -∞
    /// - `log10(+∞)` returns +∞
    ///
    /// # Applications
    ///
    /// - Audio: Decibel calculations (dB = 20 * log10(amplitude))
    /// - Chemistry: pH calculations (-log10(H+ concentration))
    /// - Seismology: Richter scale
    /// - Scientific notation: Order of magnitude calculations
    pub fn log10(&self) -> Result<Vector<f32>> {
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, log10, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Element-wise sine: result\[i\] = sin(x\[i\])
    ///
    /// Computes the sine for each element (input in radians).
    /// Uses Rust's optimized f32::sin() method.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let v = Vector::from_slice(&[0.0, PI / 2.0, PI]);
    /// let result = v.sin().unwrap();
    /// // result ≈ [0.0, 1.0, 0.0]
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `sin(0)` returns 0.0
    /// - `sin(π/2)` returns 1.0
    /// - `sin(π)` returns 0.0 (approximately)
    /// - `sin(-x)` returns -sin(x) (odd function)
    /// - Periodic with period 2π: sin(x + 2π) = sin(x)
    ///
    /// # Applications
    ///
    /// - Signal processing: Waveform generation, oscillators, modulation
    /// - Physics: Harmonic motion, wave propagation, pendulums
    /// - Audio: Synthesizers, tone generation, effects processing
    /// - Graphics: Animation, rotation transformations, procedural generation
    /// - Fourier analysis: Frequency decomposition, spectral analysis
    pub fn sin(&self) -> Result<Vector<f32>> {
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, sin, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Element-wise cosine: result\[i\] = cos(x\[i\])
    ///
    /// Computes the cosine for each element (input in radians).
    /// Uses Rust's optimized f32::cos() method.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let v = Vector::from_slice(&[0.0, PI / 2.0, PI]);
    /// let result = v.cos().unwrap();
    /// // result ≈ [1.0, 0.0, -1.0]
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `cos(0)` returns 1.0
    /// - `cos(π/2)` returns 0.0 (approximately)
    /// - `cos(π)` returns -1.0
    /// - `cos(-x)` returns cos(x) (even function)
    /// - Periodic with period 2π: cos(x + 2π) = cos(x)
    /// - Relation to sine: cos(x) = sin(x + π/2)
    ///
    /// # Applications
    ///
    /// - Signal processing: Phase-shifted waveforms, I/Q modulation, quadrature signals
    /// - Physics: Projectile motion, wave interference, damped oscillations
    /// - Graphics: Rotation matrices, camera transforms, circular motion
    /// - Audio: Stereo panning, spatial audio, frequency synthesis
    /// - Engineering: Control systems, frequency response, AC circuits
    pub fn cos(&self) -> Result<Vector<f32>> {
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, cos, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Computes element-wise tangent (tan) of the vector.
    ///
    /// Returns a new vector where each element is the tangent of the corresponding input element.
    /// tan(x) = sin(x) / cos(x)
    ///
    /// # Returns
    /// - `Ok(Vector<f32>)`: New vector with tan(x) for each element
    ///
    /// # Properties
    /// - Odd function: tan(-x) = -tan(x)
    /// - Period: 2π (not π, despite common misconception)
    /// - Undefined at x = π/2 + nπ (where n is any integer)
    /// - tan(x) = sin(x) / cos(x)
    /// - Range: (-∞, +∞)
    ///
    /// # Performance
    /// - Iterator map pattern for cache efficiency
    /// - Leverages Rust's optimized f32::tan()
    /// - Auto-vectorized by LLVM on supporting platforms
    ///
    /// # Examples
    /// ```
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let angles = Vector::from_slice(&[0.0, PI / 4.0, -PI / 4.0]);
    /// let result = angles.tan().unwrap();
    /// // Result: [0.0, 1.0, -1.0] (approximately)
    /// ```
    ///
    /// # Use Cases
    /// - Trigonometry: Slope calculations, angle relationships
    /// - Signal processing: Phase analysis, modulation
    /// - Physics: Projectile trajectories, optics (Snell's law angles)
    /// - Graphics: Perspective projection, field of view calculations
    /// - Engineering: Slope gradients, tangent lines to curves
    pub fn tan(&self) -> Result<Vector<f32>> {
        let mut result_data = vec![0.0; self.len()];

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, tan, &self.data, &mut result_data);
        }

        Ok(Vector {
            data: result_data,
            backend: self.backend,
        })
    }

    /// Computes element-wise arcsine (asin/sin⁻¹) of the vector.
    ///
    /// Returns a new vector where each element is the inverse sine of the corresponding input element.
    /// This is the inverse function of sin: if y = sin(x), then x = asin(y).
    ///
    /// # Returns
    /// - `Ok(Vector<f32>)`: New vector with asin(x) for each element
    ///
    /// # Properties
    /// - Domain: [-1, 1] (inputs outside this range produce NaN)
    /// - Range: [-π/2, π/2]
    /// - Odd function: asin(-x) = -asin(x)
    /// - Inverse relation: asin(sin(x)) = x for x ∈ [-π/2, π/2]
    /// - asin(0) = 0
    /// - asin(1) = π/2
    /// - asin(-1) = -π/2
    ///
    /// # Performance
    /// - Iterator map pattern for cache efficiency
    /// - Leverages Rust's optimized f32::asin()
    /// - Auto-vectorized by LLVM on supporting platforms
    ///
    /// # Examples
    /// ```
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let values = Vector::from_slice(&[0.0, 0.5, 1.0]);
    /// let result = values.asin().unwrap();
    /// // Result: [0.0, π/6, π/2] (approximately)
    /// ```
    ///
    /// # Use Cases
    /// - Physics: Calculating angles from sine values in mechanics, optics
    /// - Signal processing: Phase recovery, demodulation
    /// - Graphics: Inverse transformations, angle calculations
    /// - Navigation: GPS calculations, spherical trigonometry
    /// - Control systems: Inverse kinematics, servo positioning
    pub fn asin(&self) -> Result<Vector<f32>> {
        let asin_data: Vec<f32> = self.data.iter().map(|x| x.asin()).collect();
        Ok(Vector {
            data: asin_data,
            backend: self.backend,
        })
    }

    /// Computes element-wise arccosine (acos/cos⁻¹) of the vector.
    ///
    /// Returns a new vector where each element is the inverse cosine of the corresponding input element.
    /// This is the inverse function of cos: if y = cos(x), then x = acos(y).
    ///
    /// # Returns
    /// - `Ok(Vector<f32>)`: New vector with acos(x) for each element
    ///
    /// # Properties
    /// - Domain: [-1, 1] (inputs outside this range produce NaN)
    /// - Range: [0, π]
    /// - Symmetry: acos(-x) = π - acos(x)
    /// - Inverse relation: acos(cos(x)) = x for x ∈ [0, π]
    /// - acos(0) = π/2
    /// - acos(1) = 0
    /// - acos(-1) = π
    ///
    /// # Performance
    /// - Iterator map pattern for cache efficiency
    /// - Leverages Rust's optimized f32::acos()
    /// - Auto-vectorized by LLVM on supporting platforms
    ///
    /// # Examples
    /// ```
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let values = Vector::from_slice(&[0.0, 0.5, 1.0]);
    /// let result = values.acos().unwrap();
    /// // Result: [π/2, π/3, 0.0] (approximately)
    /// ```
    ///
    /// # Use Cases
    /// - Physics: Angle calculations in mechanics, optics, reflections
    /// - Signal processing: Phase analysis, correlation functions
    /// - Graphics: View angle calculations, lighting models
    /// - Navigation: Bearing calculations, great circle distances
    /// - Robotics: Joint angle solving, orientation calculations
    pub fn acos(&self) -> Result<Vector<f32>> {
        let acos_data: Vec<f32> = self.data.iter().map(|x| x.acos()).collect();
        Ok(Vector {
            data: acos_data,
            backend: self.backend,
        })
    }

    /// Computes element-wise arctangent (atan/tan⁻¹) of the vector.
    ///
    /// Returns a new vector where each element is the inverse tangent of the corresponding input element.
    /// This is the inverse function of tan: if y = tan(x), then x = atan(y).
    ///
    /// # Returns
    /// - `Ok(Vector<f32>)`: New vector with atan(x) for each element
    ///
    /// # Properties
    /// - Domain: All real numbers (-∞, +∞)
    /// - Range: (-π/2, π/2)
    /// - Odd function: atan(-x) = -atan(x)
    /// - Inverse relation: atan(tan(x)) = x for x ∈ (-π/2, π/2)
    /// - atan(0) = 0
    /// - atan(1) = π/4
    /// - atan(-1) = -π/4
    /// - lim(x→∞) atan(x) = π/2
    /// - lim(x→-∞) atan(x) = -π/2
    ///
    /// # Performance
    /// - Iterator map pattern for cache efficiency
    /// - Leverages Rust's optimized f32::atan()
    /// - Auto-vectorized by LLVM on supporting platforms
    ///
    /// # Examples
    /// ```
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let values = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = values.atan().unwrap();
    /// // Result: [0.0, π/4, -π/4] (approximately)
    /// ```
    ///
    /// # Use Cases
    /// - Physics: Angle calculations from slopes, velocity components
    /// - Signal processing: Phase unwrapping, FM demodulation
    /// - Graphics: Rotation calculations, camera orientation
    /// - Robotics: Inverse kinematics, steering angles
    /// - Navigation: Heading calculations from coordinates
    pub fn atan(&self) -> Result<Vector<f32>> {
        let atan_data: Vec<f32> = self.data.iter().map(|x| x.atan()).collect();
        Ok(Vector {
            data: atan_data,
            backend: self.backend,
        })
    }

    /// Computes the hyperbolic sine (sinh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// sinh(x) = (e^x - e^(-x)) / 2
    ///
    /// # Properties
    ///
    /// - Domain: (-∞, +∞)
    /// - Range: (-∞, +∞)
    /// - Odd function: sinh(-x) = -sinh(x)
    /// - sinh(0) = 0
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = v.sinh().unwrap();
    /// assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    /// ```
    pub fn sinh(&self) -> Result<Vector<f32>> {
        let sinh_data: Vec<f32> = self.data.iter().map(|x| x.sinh()).collect();
        Ok(Vector {
            data: sinh_data,
            backend: self.backend,
        })
    }

    /// Computes the hyperbolic cosine (cosh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// cosh(x) = (e^x + e^(-x)) / 2
    ///
    /// # Properties
    ///
    /// - Domain: (-∞, +∞)
    /// - Range: [1, +∞)
    /// - Even function: cosh(-x) = cosh(x)
    /// - cosh(0) = 1
    /// - Always positive: cosh(x) ≥ 1 for all x
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = v.cosh().unwrap();
    /// assert!((result.as_slice()[0] - 1.0).abs() < 1e-5);
    /// ```
    pub fn cosh(&self) -> Result<Vector<f32>> {
        let cosh_data: Vec<f32> = self.data.iter().map(|x| x.cosh()).collect();
        Ok(Vector {
            data: cosh_data,
            backend: self.backend,
        })
    }

    /// Computes the hyperbolic tangent (tanh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// tanh(x) = sinh(x) / cosh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    ///
    /// # Properties
    ///
    /// - Domain: (-∞, +∞)
    /// - Range: (-1, 1)
    /// - Odd function: tanh(-x) = -tanh(x)
    /// - tanh(0) = 0
    /// - Bounded: -1 < tanh(x) < 1 for all x
    /// - Commonly used as activation function in neural networks
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = v.tanh().unwrap();
    /// assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    /// // All values are in range (-1, 1)
    /// assert!(result.as_slice().iter().all(|&x| x > -1.0 && x < 1.0));
    /// ```
    pub fn tanh(&self) -> Result<Vector<f32>> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.tanh(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        let mut result = vec![0.0; self.len()];

        // Dispatch to appropriate SIMD backend
        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::tanh(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::tanh(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    Avx2Backend::tanh(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::tanh(&self.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::tanh(&self.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    ScalarBackend::tanh(&self.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::tanh(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    ScalarBackend::tanh(&self.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
                    // Auto should have been resolved at Vector creation
                    // GPU falls back to best available SIMD
                    #[cfg(target_arch = "x86_64")]
                    {
                        if is_x86_feature_detected!("avx2") {
                            Avx2Backend::tanh(&self.data, &mut result);
                        } else {
                            Sse2Backend::tanh(&self.data, &mut result);
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        ScalarBackend::tanh(&self.data, &mut result);
                    }
                }
            }
        }

        Ok(Vector {
            data: result,
            backend: self.backend,
        })
    }

    /// Computes the inverse hyperbolic sine (asinh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// asinh(x) = ln(x + sqrt(x² + 1))
    ///
    /// # Properties
    ///
    /// - Domain: (-∞, +∞)
    /// - Range: (-∞, +∞)
    /// - Odd function: asinh(-x) = -asinh(x)
    /// - asinh(0) = 0
    /// - Inverse of sinh: asinh(sinh(x)) = x
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = v.asinh().unwrap();
    /// assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    /// ```
    pub fn asinh(&self) -> Result<Vector<f32>> {
        let asinh_data: Vec<f32> = self.data.iter().map(|x| x.asinh()).collect();
        Ok(Vector {
            data: asinh_data,
            backend: self.backend,
        })
    }

    /// Computes the inverse hyperbolic cosine (acosh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// acosh(x) = ln(x + sqrt(x² - 1))
    ///
    /// # Properties
    ///
    /// - Domain: [1, +∞)
    /// - Range: [0, +∞)
    /// - acosh(1) = 0
    /// - Inverse of cosh: acosh(cosh(x)) = x for x >= 0
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let result = v.acosh().unwrap();
    /// assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    /// ```
    pub fn acosh(&self) -> Result<Vector<f32>> {
        let acosh_data: Vec<f32> = self.data.iter().map(|x| x.acosh()).collect();
        Ok(Vector {
            data: acosh_data,
            backend: self.backend,
        })
    }

    /// Computes the inverse hyperbolic tangent (atanh) of each element.
    ///
    /// Domain: (-1, 1)
    /// Range: (-∞, +∞)
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 0.5, -0.5]);
    /// let result = v.atanh().unwrap();
    /// // atanh(0) = 0, atanh(0.5) ≈ 0.549, atanh(-0.5) ≈ -0.549
    /// ```
    pub fn atanh(&self) -> Result<Vector<f32>> {
        let atanh_data: Vec<f32> = self.data.iter().map(|x| x.atanh()).collect();
        Ok(Vector {
            data: atanh_data,
            backend: self.backend,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Exponential Functions ==========

    #[test]
    fn test_exp_basic() {
        let v = Vector::from_slice(&[0.0, 1.0, 2.0]);
        let result = v.exp().unwrap();
        assert!((result.as_slice()[0] - 1.0).abs() < 1e-6); // e^0 = 1
        assert!((result.as_slice()[1] - std::f32::consts::E).abs() < 1e-5); // e^1 = e
        assert!((result.as_slice()[2] - std::f32::consts::E.powi(2)).abs() < 1e-4);
    }

    #[test]
    fn test_exp_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.exp().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_exp_negative() {
        let v = Vector::from_slice(&[-1.0, -2.0]);
        let result = v.exp().unwrap();
        assert!((result.as_slice()[0] - 1.0 / std::f32::consts::E).abs() < 1e-5);
    }

    #[test]
    fn test_ln_basic() {
        let v = Vector::from_slice(&[1.0, std::f32::consts::E, std::f32::consts::E.powi(2)]);
        let result = v.ln().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 1.0).abs() < 1e-5);
        assert!((result.as_slice()[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_ln_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.ln().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_log2_basic() {
        let v = Vector::from_slice(&[1.0, 2.0, 4.0, 8.0]);
        let result = v.log2().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 1.0).abs() < 1e-6);
        assert!((result.as_slice()[2] - 2.0).abs() < 1e-6);
        assert!((result.as_slice()[3] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_log2_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.log2().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_log10_basic() {
        let v = Vector::from_slice(&[1.0, 10.0, 100.0, 1000.0]);
        let result = v.log10().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 1.0).abs() < 1e-5);
        assert!((result.as_slice()[2] - 2.0).abs() < 1e-5);
        assert!((result.as_slice()[3] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_log10_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.log10().unwrap();
        assert!(result.is_empty());
    }

    // ========== Trigonometric Functions ==========

    #[test]
    fn test_sin_basic() {
        let v = Vector::from_slice(&[0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI]);
        let result = v.sin().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 1.0).abs() < 1e-6);
        assert!((result.as_slice()[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_sin_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.sin().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_cos_basic() {
        let v = Vector::from_slice(&[0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI]);
        let result = v.cos().unwrap();
        assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[2] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_cos_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.cos().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_tan_basic() {
        let v = Vector::from_slice(&[0.0, std::f32::consts::PI / 4.0]);
        let result = v.tan().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_tan_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.tan().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_asin_basic() {
        let v = Vector::from_slice(&[0.0, 0.5, 1.0]);
        let result = v.asin().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - std::f32::consts::FRAC_PI_6).abs() < 1e-3);
        assert!((result.as_slice()[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    }

    #[test]
    fn test_asin_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.asin().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_acos_basic() {
        let v = Vector::from_slice(&[1.0, 0.5, 0.0]);
        let result = v.acos().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - std::f32::consts::FRAC_PI_3).abs() < 1e-3);
        assert!((result.as_slice()[2] - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
    }

    #[test]
    fn test_acos_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.acos().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_atan_basic() {
        let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
        let result = v.atan().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - std::f32::consts::FRAC_PI_4).abs() < 1e-5);
        assert!((result.as_slice()[2] - (-std::f32::consts::FRAC_PI_4)).abs() < 1e-5);
    }

    #[test]
    fn test_atan_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.atan().unwrap();
        assert!(result.is_empty());
    }

    // ========== Hyperbolic Functions ==========

    #[test]
    fn test_sinh_basic() {
        let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
        let result = v.sinh().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 1.1752).abs() < 1e-3);
        assert!((result.as_slice()[2] - (-1.1752)).abs() < 1e-3);
    }

    #[test]
    fn test_sinh_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.sinh().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_cosh_basic() {
        let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
        let result = v.cosh().unwrap();
        assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 1.5431).abs() < 1e-3);
        assert!((result.as_slice()[2] - 1.5431).abs() < 1e-3); // cosh is even
    }

    #[test]
    fn test_cosh_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.cosh().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_tanh_basic() {
        let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
        let result = v.tanh().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 0.7616).abs() < 1e-3);
        assert!((result.as_slice()[2] - (-0.7616)).abs() < 1e-3);
    }

    #[test]
    fn test_tanh_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        // tanh returns EmptyVector error for empty input
        assert!(matches!(v.tanh(), Err(TruenoError::EmptyVector)));
    }

    #[test]
    fn test_asinh_basic() {
        let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
        let result = v.asinh().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 0.8814).abs() < 1e-3);
        assert!((result.as_slice()[2] - (-0.8814)).abs() < 1e-3);
    }

    #[test]
    fn test_asinh_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.asinh().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_acosh_basic() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let result = v.acosh().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 1.3170).abs() < 1e-3);
        assert!((result.as_slice()[2] - 1.7627).abs() < 1e-3);
    }

    #[test]
    fn test_acosh_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.acosh().unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_atanh_basic() {
        let v = Vector::from_slice(&[0.0, 0.5, -0.5]);
        let result = v.atanh().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 0.5493).abs() < 1e-3);
        assert!((result.as_slice()[2] - (-0.5493)).abs() < 1e-3);
    }

    #[test]
    fn test_atanh_empty() {
        let v = Vector::<f32>::from_slice(&[]);
        let result = v.atanh().unwrap();
        assert!(result.is_empty());
    }

    // ========== Backend-specific Tests ==========

    #[test]
    fn test_exp_scalar_backend() {
        let v = Vector::from_slice_with_backend(&[0.0, 1.0, 2.0], Backend::Scalar);
        let result = v.exp().unwrap();
        assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_exp_sse2_backend() {
        let v = Vector::from_slice_with_backend(&[0.0, 1.0, 2.0, 3.0], Backend::SSE2);
        let result = v.exp().unwrap();
        assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_exp_avx2_backend() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let v = Vector::from_slice_with_backend(&[0.0; 16], Backend::AVX2);
        let result = v.exp().unwrap();
        for val in result.as_slice() {
            assert!((val - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sin_scalar_backend() {
        let v = Vector::from_slice_with_backend(&[0.0, std::f32::consts::FRAC_PI_2], Backend::Scalar);
        let result = v.sin().unwrap();
        assert!((result.as_slice()[0] - 0.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cos_scalar_backend() {
        let v = Vector::from_slice_with_backend(&[0.0, std::f32::consts::PI], Backend::Scalar);
        let result = v.cos().unwrap();
        assert!((result.as_slice()[0] - 1.0).abs() < 1e-6);
        assert!((result.as_slice()[1] - (-1.0)).abs() < 1e-5);
    }

    // ========== Large Array Tests ==========

    #[test]
    fn test_exp_large() {
        let v = Vector::from_slice(&[1.0; 1000]);
        let result = v.exp().unwrap();
        assert_eq!(result.len(), 1000);
        for val in result.as_slice() {
            assert!((val - std::f32::consts::E).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sin_large() {
        let v = Vector::from_slice(&[0.0; 1000]);
        let result = v.sin().unwrap();
        assert_eq!(result.len(), 1000);
        for val in result.as_slice() {
            assert!((val - 0.0).abs() < 1e-6);
        }
    }

    // ========== Inverse Relationship Tests ==========

    #[test]
    fn test_exp_ln_inverse() {
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let exp_result = v.exp().unwrap();
        let roundtrip = exp_result.ln().unwrap();
        for (orig, rt) in v.as_slice().iter().zip(roundtrip.as_slice()) {
            assert!((orig - rt).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sin_asin_inverse() {
        let v = Vector::from_slice(&[0.0, 0.3, 0.5, 0.7]);
        let sin_result = v.sin().unwrap();
        let roundtrip = sin_result.asin().unwrap();
        for (orig, rt) in v.as_slice().iter().zip(roundtrip.as_slice()) {
            assert!((orig - rt).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cos_acos_inverse() {
        let v = Vector::from_slice(&[0.0, 0.3, 0.5, 0.7]);
        let cos_result = v.cos().unwrap();
        let roundtrip = cos_result.acos().unwrap();
        for (orig, rt) in v.as_slice().iter().zip(roundtrip.as_slice()) {
            assert!((orig - rt).abs() < 1e-5);
        }
    }

    #[test]
    fn test_sinh_asinh_inverse() {
        let v = Vector::from_slice(&[0.0, 1.0, 2.0, -1.0, -2.0]);
        let sinh_result = v.sinh().unwrap();
        let roundtrip = sinh_result.asinh().unwrap();
        for (orig, rt) in v.as_slice().iter().zip(roundtrip.as_slice()) {
            assert!((orig - rt).abs() < 1e-4);
        }
    }

    #[test]
    fn test_tanh_atanh_inverse() {
        let v = Vector::from_slice(&[0.0, 0.3, 0.5, -0.3, -0.5]);
        let tanh_result = v.tanh().unwrap();
        let roundtrip = tanh_result.atanh().unwrap();
        for (orig, rt) in v.as_slice().iter().zip(roundtrip.as_slice()) {
            assert!((orig - rt).abs() < 1e-4);
        }
    }
}
