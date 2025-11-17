# Trueno ⚡

> **Multi-Target High-Performance Compute Library**

[![CI](https://github.com/paiml/trueno/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/trueno/actions)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/paiml/trueno)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/trueno.svg)](https://crates.io/crates/trueno)

**Trueno** (Spanish: "thunder") provides unified, high-performance compute primitives across three execution targets:

1. **CPU SIMD** - x86 (SSE2/AVX/AVX2/AVX-512), ARM (NEON), WASM (SIMD128)
2. **GPU** - Vulkan/Metal/DX12/WebGPU via `wgpu`
3. **WebAssembly** - Portable SIMD128 for browser/edge deployment

## Quick Start

```rust
use trueno::{Vector, Matrix};

// Vector operations
let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);

// Auto-selects best backend (AVX2/GPU/WASM)
let result = a.add(&b).unwrap();
assert_eq!(result.as_slice(), &[6.0, 8.0, 10.0, 12.0]);

let dot_product = a.dot(&b).unwrap();  // 70.0
let sum = a.sum().unwrap();            // 10.0
let max = a.max().unwrap();            // 4.0

// Matrix operations (NEW in v0.1)
let m1 = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
let m2 = Matrix::identity(2);
let product = m1.matmul(&m2).unwrap();  // Matrix multiplication
let transposed = m1.transpose();        // Matrix transpose
```

## Performance

Trueno delivers **exceptional performance** through multi-level SIMD optimization:

### SSE2 (128-bit SIMD) vs Scalar

| Operation | Speedup | Use Case |
|-----------|---------|----------|
| **Dot Product** | **340%** faster | Machine learning, signal processing |
| **Sum Reduction** | **315%** faster | Statistics, aggregations |
| **Max Finding** | **348%** faster | Data analysis, optimization |
| Element-wise Add | 3-10% faster | Memory-bound (limited SIMD benefit) |
| Element-wise Mul | 5-6% faster | Memory-bound (limited SIMD benefit) |

### AVX2 (256-bit SIMD) vs SSE2

| Operation | Speedup | Notes |
|-----------|---------|-------|
| **Dot Product** | **182%** faster | FMA (fused multiply-add) acceleration |
| Element-wise Add | 15% faster | Memory bandwidth limited |
| Element-wise Mul | 12% faster | Memory bandwidth limited |

**Key Insights**:
- SIMD excels at compute-intensive operations (dot product, reductions)
- Element-wise operations are memory-bound, limiting SIMD gains
- AVX2's FMA provides significant acceleration for dot products

### Matrix Operations (SIMD-Optimized)

| Operation | Size | Time | Performance |
|-----------|------|------|-------------|
| **Matrix Multiply** | 64×64 | 59.9 µs | SIMD threshold |
| **Matrix Multiply** | 128×128 | 434.9 µs | ~7x faster than naive |
| **Matrix Multiply** | 256×256 | 2.67 ms | Scales O(n³) |
| **Matrix Transpose** | 256×256 | 69.1 µs | Cache-optimized |
| **Matrix-Vector** | 512×512 | 139.8 µs | SIMD dot products |

**SIMD Optimization Strategy**:
- **Threshold**: 64×64 (auto-selects SIMD vs naive)
- **Transpose**: Pre-transpose B for cache locality (row-major access)
- **Dot Products**: Uses Vector::dot() for SIMD acceleration (2-8x speedup)
- **Small Matrices**: Uses naive O(n³) to avoid SIMD overhead

**📖 See [Performance Guide](docs/PERFORMANCE_GUIDE.md) and [AVX2 Benchmarks](docs/AVX2_BENCHMARKS.md) for detailed analysis.**

## Features

- **🚀 Write Once, Optimize Everywhere**: Single algorithm, multiple backends
- **⚡ Runtime Dispatch**: Auto-select best implementation based on CPU features
- **🎮 GPU Acceleration**: Optional wgpu backend for very large matrices (>1000×1000)
- **🛡️ Zero Unsafe in Public API**: Safety via type system, `unsafe` isolated in backends
- **📊 Benchmarked Performance**: Every optimization proves ≥10% speedup
- **🧪 Extreme TDD**: >90% test coverage, mutation testing, property-based tests
- **🎯 Production Ready**: PMAT quality gates, Toyota Way principles

## Design Principles

### Write Once, Optimize Everywhere
```rust
// Same code runs optimally on x86, ARM, WASM, GPU
let result = a.add(&b).unwrap();
```

Trueno automatically selects the best backend:
- **x86_64**: AVX-512 → AVX2 → AVX → SSE2 → Scalar
- **ARM**: NEON → Scalar
- **WASM**: SIMD128 → Scalar
- **GPU** (optional): Vulkan/Metal/DX12/WebGPU (>1000×1000 matrices)

### Safety First
```rust
// Public API is 100% safe Rust
let result = vector.add(&other)?;  // Returns Result<Vector, TruenoError>

// Size mismatches caught at runtime
let a = Vector::from_slice(&[1.0, 2.0]);
let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
assert!(a.add(&b).is_err());  // SizeMismatch error
```

### Performance Targets

| Operation | Size | Target Speedup vs Scalar | Backend |
|-----------|------|-------------------------|---------|
| `add()` | 1K | 8x | AVX2 |
| `add()` | 100K | 16x | GPU |
| `dot()` | 10K | 12x | AVX2 + FMA |
| `sum()` | 1M | 20x | GPU |

*All optimizations benchmarked with Criterion.rs, minimum 10% improvement required*

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
trueno = "0.1"
```

### GPU Acceleration (Optional)

Enable GPU support for very large matrices:

```toml
[dependencies]
trueno = { version = "0.1", features = ["gpu"] }
```

**Requirements**:
- Vulkan, Metal, or DirectX 12 compatible GPU
- wgpu runtime dependencies
- GPU backend automatically activates for matrices >1000×1000

For bleeding-edge features:

```toml
[dependencies]
trueno = { git = "https://github.com/paiml/trueno", features = ["gpu"] }
```

## Usage Examples

### Basic Vector Operations

```rust
use trueno::Vector;

// Element-wise addition
let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
let sum = a.add(&b).unwrap();
assert_eq!(sum.as_slice(), &[5.0, 7.0, 9.0]);

// Element-wise multiplication
let product = a.mul(&b).unwrap();
assert_eq!(product.as_slice(), &[4.0, 10.0, 18.0]);

// Dot product
let dot = a.dot(&b).unwrap();
assert_eq!(dot, 32.0);  // 1*4 + 2*5 + 3*6

// Reductions
let total = a.sum().unwrap();  // 6.0
let maximum = a.max().unwrap();  // 3.0
```

### Backend Selection

```rust
use trueno::{Vector, Backend};

// Auto-select best backend (recommended)
let v = Vector::from_slice(&data);  // Uses Backend::Auto

// Explicit backend (for testing/benchmarking)
let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
let v = Vector::from_slice_with_backend(&data, Backend::GPU);
```

### Error Handling

```rust
use trueno::{Vector, TruenoError};

let a = Vector::from_slice(&[1.0, 2.0]);
let b = Vector::from_slice(&[1.0, 2.0, 3.0]);

match a.add(&b) {
    Ok(result) => println!("Sum: {:?}", result.as_slice()),
    Err(TruenoError::SizeMismatch { expected, actual }) => {
        eprintln!("Size mismatch: expected {}, got {}", expected, actual);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Ecosystem Integration

Trueno integrates with the Pragmatic AI Labs transpiler ecosystem:

### Ruchy
```ruby
# Ruchy syntax
let v = Vector([1.0, 2.0]) + Vector([3.0, 4.0])
# Transpiles to: trueno::Vector::add()
```

### Depyler (Python → Rust)
```python
# Python/NumPy code
import numpy as np
result = np.dot(a, b)
# Transpiles to: trueno::Vector::dot(&a, &b)
```

### Decy (C → Rust)
```c
// C SIMD intrinsics
__m256 result = _mm256_add_ps(a, b);
// Transpiles to: trueno::Vector::add() (safe!)
```

## Development

### Prerequisites

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install development tools
make install-tools
```

### Building

```bash
# Development build
make build

# Release build (optimized)
make build-release

# Run tests
make test

# Fast test run (<5 min target)
make test-fast
```

### Quality Gates

Trueno enforces **EXTREME TDD** quality standards:

```bash
# Run all quality gates (pre-commit)
make quality-gates

# Individual gates
make lint       # Zero warnings policy
make fmt-check  # Format verification
make test-fast  # All tests (<5 min)
make coverage   # >85% required (<10 min)
make mutate     # Mutation testing (>80% kill rate)
```

**Quality Metrics**:
- ✅ **Test Coverage**: 100% (target >85%)
- ✅ **PMAT TDG Score**: 96.1/100 (A+)
- ✅ **Clippy Warnings**: 0
- ✅ **Property Tests**: 10 tests × 100 cases each
- ✅ **Cyclomatic Complexity**: Median 1.0 (very low)

### PMAT Integration

```bash
# Technical Debt Grading
make pmat-tdg

# Complexity analysis
make pmat-analyze

# Repository health score
make pmat-score
```

### Profiling & Performance Analysis

Trueno integrates **Renacer** for deep performance profiling:

```bash
# Profile benchmarks to find bottlenecks
make profile

# Generate flamegraph visualization
make profile-flamegraph

# Profile specific benchmark
make profile-bench BENCH=vector_ops

# Profile test suite
make profile-test
```

**Profiling Use Cases**:
- 🔬 **SIMD Validation**: Verify optimizations show expected speedups (2-8x)
- 🎯 **Hot Path Analysis**: Identify top 10 functions consuming most time
- 💾 **Memory Bottlenecks**: Detect cache misses and memory access patterns
- 🚀 **Backend Selection**: Validate runtime dispatch overhead is minimal
- 📊 **Flamegraph Visualization**: Visual analysis of performance characteristics

**Example Output**:
```
🔬 Profiling benchmark: vector_ops
I/O Bottleneck: memcpy() - 15.2ms (42% of runtime)
Hot Functions:
  1. _mm256_add_ps - 3.4ms (9.4%)
  2. Vector::dot - 2.1ms (5.8%)
  3. backend_dispatch - 0.3ms (0.8%)
```

### Testing Philosophy

Trueno uses **multi-layered testing**:

1. **Unit Tests** (30 tests): Basic functionality, edge cases, error paths
2. **Property Tests** (10 tests × 100 cases): Mathematical properties verification
   - Commutativity: `a + b == b + a`
   - Associativity: `(a + b) + c == a + (b + c)`
   - Identity elements: `a + 0 == a`, `a * 1 == a`
   - Distributive: `a * (b + c) == a*b + a*c`
3. **Integration Tests**: Backend selection, large datasets
4. **Benchmarks**: Performance regression prevention (Criterion.rs)
5. **Mutation Tests**: Test suite effectiveness (>80% kill rate)

Run property tests with verbose output:
```bash
cargo test property_tests -- --nocapture
```

## Benchmarking

```bash
# Run all benchmarks
make bench

# Benchmark specific operation
cargo bench -- add
cargo bench -- dot
```

Benchmark results are stored in `target/criterion/` and include:
- Throughput (elements/second)
- Latency (mean, median, p95, p99)
- Backend comparison (Scalar vs SIMD vs GPU)
- Regression detection

## Examples

Trueno includes several runnable examples demonstrating real-world use cases:

```bash
# Machine Learning: Cosine similarity, L2 normalization, k-NN
cargo run --release --example ml_similarity

# Performance: Compare Scalar vs SSE2 backends
cargo run --release --example performance_demo

# Backend Detection: Runtime CPU feature detection
cargo run --release --example backend_detection
```

**ML Example Features**:
- Document similarity for recommendation systems
- Feature normalization for neural networks
- k-Nearest Neighbors classification
- Demonstrates 340% speedup for dot products

See `examples/` directory for complete code.

## Project Structure

```
trueno/
├── src/
│   ├── lib.rs          # Public API, backend enum, auto-selection
│   ├── error.rs        # Error types (TruenoError)
│   ├── vector.rs       # Vector<T> implementation
│   └── backends/       # Backend implementations (future)
│       ├── scalar.rs
│       ├── simd/
│       │   ├── avx2.rs
│       │   ├── avx512.rs
│       │   └── neon.rs
│       ├── gpu.rs
│       └── wasm.rs
├── benches/            # Criterion benchmarks (future)
├── docs/
│   └── specifications/ # Design specifications
├── Cargo.toml          # Dependencies, optimization flags
├── Makefile            # Quality gates, development commands
└── README.md           # This file
```

## Roadmap

### Phase 1: Scalar Baseline ✅ COMPLETE
- [x] Core `Vector<f32>` API (add, mul, dot, sum, max)
- [x] Error handling with `TruenoError`
- [x] 100% test coverage (40 tests)
- [x] Property-based tests (PROPTEST_CASES=100)
- [x] PMAT quality gates integration
- [x] Documentation and README

### Phase 2: x86 SIMD ✅ COMPLETE
- [x] Runtime CPU feature detection (`is_x86_feature_detected!`)
- [x] SSE2 implementation (baseline x86_64)
- [x] Benchmarks proving ≥10% speedup (66.7% of tests, avg 178.5%)
- [x] Auto-dispatch based on CPU features
- [x] Backend trait architecture
- [x] Comprehensive performance analysis

### Phase 3: AVX2 SIMD ✅ COMPLETE
- [x] AVX2 implementation with FMA support (256-bit SIMD)
- [x] Benchmarks proving exceptional speedups (1.82x for dot product)
- [x] Performance analysis and documentation
- [x] All quality gates passing (0 warnings, 78 tests)

### Phase 4: ARM SIMD ✅ COMPLETE
- [x] ARM NEON implementation (128-bit SIMD)
- [x] Runtime feature detection (ARMv7/ARMv8/AArch64)
- [x] Cross-platform compilation support
- [x] Comprehensive tests with cross-validation
- [ ] Benchmarks on ARM hardware (pending ARM access)

### Phase 5: WebAssembly ✅ COMPLETE
- [x] WASM SIMD128 implementation (128-bit SIMD)
- [x] All 5 operations with f32x4 intrinsics
- [x] Comprehensive tests with cross-validation
- [ ] Browser deployment example (future)
- [ ] Edge computing use case (future)

### Phase 6: GPU Compute 🚧 INITIAL IMPLEMENTATION
- [x] `wgpu` integration (optional `gpu` feature flag)
- [x] Compute shader kernels (WGSL) for matrix multiplication
- [x] Host-device memory transfer with async execution
- [x] GPU dispatch heuristics (>1000×1000 threshold)
- [x] Automatic fallback to SIMD/CPU if GPU unavailable
- [ ] Vector operations on GPU (add, dot product, reductions)
- [ ] Multi-GPU support
- [ ] Performance benchmarks (10-50x speedup target)

**Phase 6 Progress**: GPU-accelerated matrix multiplication implemented with wgpu (Vulkan/Metal/DX12/WebGPU). Automatic backend selection: GPU for >1000×1000 matrices, SIMD for >64×64, naive for smaller. Complete WGSL compute shader with 16×16 workgroups. Graceful fallback if GPU unavailable.

### Phase 7: Advanced Operations ✅ COMPLETE
- [x] Element-wise subtraction (sub) and division (div)
- [x] Reductions: min, max, sum, sum_kahan (Kahan summation)
- [x] Index finding: argmax, argmin
- [x] Vector norms: norm_l2 (Euclidean norm), normalize (unit vector)
- [x] Activation functions: ReLU, Leaky ReLU, ELU, Sigmoid, Softmax/Log-Softmax, GELU, Swish/SiLU
- [x] Preprocessing: zscore, minmax_normalize, clip
- [x] Statistical operations: mean, variance, stddev, covariance, correlation

### Phase 8: Matrix Operations 🚧 IN PROGRESS
- [x] Matrix<T> type with row-major storage (NumPy-compatible)
- [x] Matrix multiplication (matmul) - naive O(n³)
- [x] Matrix transpose
- [x] Matrix-vector operations (matvec, vecmat)
- [x] Comprehensive examples (matrix_operations.rs)
- [x] SIMD-optimized matmul (Vector::dot with transpose optimization)
- [x] Backend equivalence tests (naive vs SIMD)
- [ ] GPU dispatch for large matrices

**Phase 8 Progress**: SIMD-optimized matrix multiplication complete with 38 tests passing (636 lib total, 756 overall). Matmul now uses Vector::dot() for SIMD acceleration (threshold: 64×64). Added 3 backend equivalence tests verifying naive and SIMD implementations produce identical results within floating-point tolerance. Automatic backend selection based on matrix size for optimal performance.

**Phase 7 Status**: ✅ COMPLETE - Core vector operations with 587 tests passing. The library now supports:
- **Element-wise operations**: add, sub, mul, div, abs (absolute value), neg (negation/unary minus), clamp (range constraint), lerp (linear interpolation), fma (fused multiply-add), sqrt (square root), recip (reciprocal), pow (power), exp (exponential), ln (natural logarithm), sin (sine), cos (cosine), tan (tangent), asin (arcsine), acos (arccosine), atan (arctangent), sinh (hyperbolic sine), cosh (hyperbolic cosine), tanh (hyperbolic tangent), asinh (inverse hyperbolic sine), acosh (inverse hyperbolic cosine), atanh (inverse hyperbolic tangent), floor (round down), ceil (round up), round (round to nearest), trunc (truncate toward zero), fract (fractional part), signum (sign function), copysign (copy sign from one vector to another), minimum (element-wise minimum of two vectors), maximum (element-wise maximum of two vectors)
- **Scalar operations**: scale (scalar multiplication with full SIMD support)
- **Dot product**: Optimized for ML/scientific computing
- **Reductions**: sum (naive + Kahan), min, max, sum_of_squares, mean (arithmetic average), variance (population variance), stddev (standard deviation), covariance (population covariance between two vectors), correlation (Pearson correlation coefficient)
- **Activation functions**: relu (rectified linear unit - max(0, x)), leaky_relu (leaky ReLU with configurable negative slope), elu (exponential linear unit with smooth gradients), sigmoid (logistic function - 1/(1+e^-x)), softmax (convert logits to probability distribution), log_softmax (numerically stable log of softmax for cross-entropy loss), gelu (Gaussian Error Linear Unit - smooth activation used in transformers like BERT/GPT), swish/silu (Swish/Sigmoid Linear Unit - self-gated activation used in EfficientNet/MobileNet v3)
- **Preprocessing**: zscore (z-score normalization/standardization), minmax_normalize (min-max scaling to [0,1] range), clip (constrain values to [min,max] range)
- **Index operations**: argmin, argmax
- **Vector norms**: L1 (Manhattan), L2 (Euclidean), L∞ (max norm), normalization to unit vectors
- **Numerical stability**: Kahan summation for accurate floating-point accumulation
- **FMA optimization**: Hardware-accelerated fused multiply-add on AVX2 and NEON platforms
- **Mathematical functions**: Element-wise square root, reciprocal, power, exponential, logarithm, trigonometric (sine, cosine, tangent), inverse trigonometric (arcsine, arccosine, arctangent), hyperbolic functions (sinh, cosh, tanh), and inverse hyperbolic functions (asinh, acosh, atanh) for ML (neural network activations), signal processing (waveforms, oscillators, phase recovery, FM demodulation), physics simulations, graphics (perspective projection, inverse transformations, lighting models, camera orientation), navigation (GPS, spherical trigonometry, bearing calculations, heading calculations), robotics (orientation calculations, inverse kinematics, steering angles), and Fourier analysis

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Quality Gates**: All PRs must pass `make quality-gates`
   - Zero clippy warnings
   - 100% formatted code
   - All tests passing
   - Coverage >85%

2. **Testing**: Include tests for new features
   - Unit tests for basic functionality
   - Property tests for mathematical operations
   - Benchmarks for performance claims

3. **Documentation**: Update README and docs for new features

4. **Toyota Way Principles**:
   - **Jidoka** (built-in quality): Tests catch issues immediately
   - **Kaizen** (continuous improvement): Every PR makes the codebase better
   - **Genchi Genbutsu** (go and see): Benchmark claims, measure reality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Pragmatic AI Labs** - [https://github.com/paiml](https://github.com/paiml)

## Acknowledgments

- Inspired by NumPy, Eigen, and ndarray
- SIMD guidance from `std::arch` documentation
- GPU compute via `wgpu` project
- Quality standards from Toyota Production System
- PMAT quality gates by Pragmatic AI Labs

## Citation

If you use Trueno in academic work, please cite:

```bibtex
@software{trueno2025,
  title = {Trueno: Multi-Target High-Performance Compute Library},
  author = {Pragmatic AI Labs},
  year = {2025},
  url = {https://github.com/paiml/trueno}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/paiml/trueno/issues)
- **Discussions**: [GitHub Discussions](https://github.com/paiml/trueno/discussions)
- **Email**: contact@paiml.com

---

**Built with EXTREME TDD and Toyota Way principles** 🚗⚡
