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
use trueno::Vector;

let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);

// Auto-selects best backend (AVX2/GPU/WASM)
let result = a.add(&b).unwrap();
assert_eq!(result.as_slice(), &[6.0, 8.0, 10.0, 12.0]);

// Other operations
let dot_product = a.dot(&b).unwrap();  // 70.0
let sum = a.sum().unwrap();            // 10.0
let max = a.max().unwrap();            // 4.0
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

**📖 See [Performance Guide](docs/PERFORMANCE_GUIDE.md) and [AVX2 Benchmarks](docs/AVX2_BENCHMARKS.md) for detailed analysis.**

## Features

- **🚀 Write Once, Optimize Everywhere**: Single algorithm, multiple backends
- **⚡ Runtime Dispatch**: Auto-select best implementation based on CPU features
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
- **GPU**: Vulkan/Metal/DX12/WebGPU (large datasets)

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

For bleeding-edge features:

```toml
[dependencies]
trueno = { git = "https://github.com/paiml/trueno" }
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

### Phase 6: GPU Compute
- [ ] `wgpu` integration
- [ ] Compute shader kernels (WGSL)
- [ ] Host-device memory transfer optimization
- [ ] GPU dispatch heuristics (OpComplexity)
- [ ] Multi-GPU support

### Phase 7: Advanced Operations (IN PROGRESS)
- [x] Element-wise subtraction (sub) and division (div)
- [x] Reductions: min, max, sum, sum_kahan (Kahan summation)
- [x] Index finding: argmax, argmin
- [x] Vector norms: norm_l2 (Euclidean norm), normalize (unit vector)
- [ ] Matrix operations (matmul, transpose)
- [ ] Convolutions (conv2d)
- [ ] Async GPU API

**Phase 7 Progress**: Core vector operations complete with 250 tests passing. The library now supports:
- **Element-wise operations**: add, sub, mul, div, abs (absolute value), clamp (range constraint), lerp (linear interpolation), fma (fused multiply-add), sqrt (square root), recip (reciprocal), pow (power), exp (exponential), ln (natural logarithm), sin (sine), cos (cosine)
- **Scalar operations**: scale (scalar multiplication with full SIMD support)
- **Dot product**: Optimized for ML/scientific computing
- **Reductions**: sum (naive + Kahan), min, max
- **Index operations**: argmin, argmax
- **Vector norms**: L1 (Manhattan), L2 (Euclidean), L∞ (max norm), normalization to unit vectors
- **Numerical stability**: Kahan summation for accurate floating-point accumulation
- **FMA optimization**: Hardware-accelerated fused multiply-add on AVX2 and NEON platforms
- **Mathematical functions**: Element-wise square root, reciprocal, power, exponential, logarithm, and trigonometric (sine, cosine) for ML, signal processing (waveforms, oscillators), physics simulations, and Fourier analysis

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
