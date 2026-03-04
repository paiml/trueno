# Signal Processing

The `trueno-fft` crate provides cuFFT-parity FFT with provable contracts.

## 1D FFT (Stockham)

Power-of-two sizes using the Stockham auto-sort algorithm with f64 twiddle factors:

```rust
use trueno_fft::{Complex, FftPlan};

let plan = FftPlan::new(1024)?;
let mut output = vec![Complex::ZERO; 1024];
plan.forward(&input, &mut output)?;
plan.inverse(&output, &mut recovered)?;
```

## Arbitrary-Length FFT (Bluestein)

For any positive N (not just powers of two):

```rust
use trueno_fft::bluestein_fft;

let mut output = vec![Complex::ZERO; 7];  // prime size
bluestein_fft(&input, &mut output, false)?;  // forward
bluestein_fft(&output, &mut recovered, true)?;  // inverse
```

## 2D FFT

Row-then-column decomposition:

```rust
use trueno_fft::fft_2d;

let mut output = vec![Complex::ZERO; 64 * 64];
fft_2d(&input, &mut output, 64, 64)?;
```

## 3D FFT

Row-column-depth decomposition for volumetric data:

```rust
use trueno_fft::{fft_3d, ifft_3d};

let mut freq = vec![Complex::ZERO; 32 * 32 * 32];
fft_3d(&input, &mut freq, 32, 32, 32)?;
ifft_3d(&freq, &mut recovered, 32, 32, 32)?;
```

## Batched FFT

Process multiple signals simultaneously:

```rust
use trueno_fft::fft_batched;

let n = 256;
let batch_count = 100;
let mut output = vec![Complex::ZERO; n * batch_count];
fft_batched(&input, &mut output, n, batch_count, false)?;
```

## Real-to-Complex (R2C)

Exploits Hermitian symmetry — outputs only N/2+1 complex values:

```rust
let mut r2c_output = vec![Complex::ZERO; n / 2 + 1];
plan.forward_r2c(&real_input, &mut r2c_output)?;
```

## Provable Contracts

Backed by `fft-stockham-v1.yaml`, `fft-bluestein-v1.yaml`, and `fft-3d-v1.yaml`:
- **Parseval**: Σ|x_n|² = (1/N)·Σ|X_k|²
- **Roundtrip**: IFFT(FFT(x)) = x within tolerance
- **Impulse**: FFT(δ) = 1
- **Bluestein ↔ Stockham equivalence** for power-of-two sizes
