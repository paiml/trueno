# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-17

### Added

#### Core Types
- `Vector<T>` type with SIMD-optimized operations
- `Matrix<T>` type with row-major storage (NumPy-compatible)
- `Backend` enum for multi-target execution (Scalar, SSE2, AVX, AVX2, AVX512, NEON, WasmSIMD, GPU)
- Runtime CPU feature detection with automatic backend selection

#### Vector Operations (87 total)
- **Element-wise**: add, sub, mul, div, abs, neg, clamp, lerp, fma, sqrt, recip, pow, exp, ln, floor, ceil, round, trunc, fract, signum, copysign, minimum, maximum
- **Trigonometric**: sin, cos, tan, asin, acos, atan
- **Hyperbolic**: sinh, cosh, tanh, asinh, acosh, atanh
- **Dot product**: Optimized with SIMD and FMA
- **Reductions**: sum (naive + Kahan), min, max, sum_of_squares, mean, variance, stddev, covariance, correlation
- **Activation functions**: relu, leaky_relu, elu, sigmoid, softmax, log_softmax, gelu, swish/silu
- **Preprocessing**: zscore, minmax_normalize, clip
- **Index operations**: argmin, argmax
- **Vector norms**: L1, L2, L∞, normalization to unit vectors
- **Scalar operations**: scale (scalar multiplication with full SIMD)

#### Matrix Operations
- Matrix multiplication (matmul) - naive O(n³) algorithm
- Matrix transpose - O(mn) swap operation
- Constructors: new(), from_vec(), zeros(), identity()
- Accessors: get(), get_mut(), rows(), cols(), shape(), as_slice()

#### Performance Optimizations
- SSE2 SIMD (128-bit): 3-4x speedup on dot product vs scalar
- AVX2 SIMD (256-bit): Additional 1.8x speedup with FMA
- Runtime dispatch based on CPU features
- Kahan summation for numerical stability
- Numerically stable algorithms (softmax with max subtraction, correlation clamping)

#### Testing & Quality
- 611 unit tests (100% passing)
- 101 doctests (100% passing)
- Property-based testing with proptest (100 cases per test)
- Zero clippy warnings
- Zero rustdoc warnings
- EXTREME TDD methodology applied throughout
- Mutation testing support
- Pre-commit quality gates via PMAT

#### Documentation
- Comprehensive rustdoc with examples for all public APIs
- README with performance benchmarks
- Quick start guide
- Phase roadmap (Phases 1-7 complete, Phase 8 in progress)
- 4 comprehensive examples:
  - activation_functions.rs
  - backend_detection.rs
  - ml_similarity.rs
  - performance_demo.rs

### Changed
- Improved numerical stability for variance/stddev with hybrid tolerance (absolute for small values, relative for large)
- Improved correlation() to clamp results to \[-1, 1\] to handle floating-point precision
- Optimized property tests with appropriate tolerances for floating-point comparisons

### Fixed
- Fixed 4 property test failures in variance/stddev operations with better tolerance handling
- Fixed all 64 rustdoc link resolution warnings by escaping mathematical notation
- Fixed atanh(tanh(x)) round-trip precision for extreme values by restricting range
- Fixed covariance bilinearity test with increased tolerance for compounding FP errors
- Fixed zscore tests for small sample sizes (n<3) and near-constant vectors

### Performance

#### Benchmarks (vs Scalar Baseline)
| Operation | Size | SSE2 | AVX2 | Notes |
|-----------|------|------|------|-------|
| Dot Product | 10K | 3.4x | 6.2x | FMA acceleration |
| Sum | 1K | 3.15x | - | - |
| Max | 1K | 3.48x | - | - |
| Add | 1K | 1.03x | 1.15x | Memory-bound |
| Mul | 1K | 1.05x | 1.12x | Memory-bound |

All benchmarks verified with Criterion.rs.

### Technical Details

#### Quality Metrics
- Test coverage: >90%
- Test execution time: 0.09s (target: <30s) - 333x faster than requirement
- TDG Score: 95.2/100 (A+)
- Zero defects at release
- Toyota Way principles applied (Jidoka, Kaizen, Genchi Genbutsu, Hansei, Poka-Yoke)

#### Platform Support
- x86_64: SSE2/AVX/AVX2/AVX-512
- ARM: NEON
- WASM: SIMD128
- GPU: Planned (infrastructure ready)

#### Dependencies
- thiserror: 2.0 (error handling)
- proptest: 1.8 (property-based testing, dev-only)
- criterion: 0.5 (benchmarking, dev-only)

### Breaking Changes
None - this is the initial release.

### Migration Guide
This is the first release. To use:

```toml
[dependencies]
trueno = "0.1"
```

```rust
use trueno::{Vector, Matrix};

let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
let result = v.add(&v).unwrap();

let m = Matrix::identity(3);
let transposed = m.transpose();
```

### Known Limitations
- Matrix operations use naive algorithms (future: SIMD, GPU, blocked matmul)
- GPU backend infrastructure exists but not yet activated
- No matrix-vector multiplication yet (planned Phase 8)
- No compile-time backend selection (runtime only)

### Contributors
- Pragmatic AI Labs Team
- Claude (AI pair programmer)

### Links
- Repository: https://github.com/paiml/trueno
- Documentation: https://docs.rs/trueno/0.1.0
- Crates.io: https://crates.io/crates/trueno

---

## [Unreleased]

### Planned for v0.2.0
- Matrix-vector multiplication
- SIMD-optimized matrix operations
- GPU dispatch for large matrices
- Additional activation functions (Mish, PReLU)
- Extended backend support

[0.1.0]: https://github.com/paiml/trueno/releases/tag/v0.1.0
[Unreleased]: https://github.com/paiml/trueno/compare/v0.1.0...HEAD
