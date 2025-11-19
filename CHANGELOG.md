# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Infrastructure
- **GitHub Pages deployment**: Automated documentation deployment workflow
  - Combines mdBook guide and rustdoc API documentation
  - Deploys to GitHub Pages on push to main branch
  - API documentation available at `/api` subdirectory
  - Workflow file: `.github/workflows/deploy-docs.yml`

### Documentation
- **Fixed Intel Intrinsics Guide reference**: Updated to mirror URL
  - Original Intel URL blocked automated link validation (HTTP 403)
  - Now references automation-friendly mirror at `laruence.com/sse`
  - Passes PMAT `validate-docs` quality gate (136/136 links valid)

### Fixed
- **AVX512 FMA tolerance**: Increased tolerance for 3-way matmul associativity
  - Addresses floating-point precision differences in AVX-512 FMA operations
  - Commit 6cd7ba2

### Quality
- **Bashrs enforcement**: Shell script quality validation
  - Replaced C-grade shell validation with A-grade Rust xtask
  - Enforces bashrs validation for Makefile and all shell scripts
  - Handles missing shell scripts gracefully

---

## [0.2.2] - 2025-11-18

### Fixed
- **CRITICAL**: Missing SIMD implementation for `abs()` operation (Issue #2)
  - Blocked downstream projects (realizar)
  - Added implementations in AVX2Backend, SSE2Backend, ScalarBackend
  - Uses bitwise AND with `0x7FFFFFFF` to clear sign bit
  - All 109 tests pass, backend equivalence verified

### Performance
- **argmax/argmin SIMD optimization**: 2.8-3.1x speedup
  - Replaced scalar index scan with SIMD index tracking
  - Uses comparison masks and blend operations
  - Processes 8 elements/iteration (AVX2) or 4 elements/iteration (SSE2)

### Added
- Comprehensive performance benchmarks for 7 operations:
  - `norm_l1()` - L1 norm (4-11x SIMD speedup, compute-bound)
  - `norm_l2()` - L2 norm (4-9x SIMD speedup, compute-bound)
  - `scale()` - Scalar multiplication (~1x speedup, memory-bound)
  - `fma()` - Fused multiply-add (~1x speedup, memory-bound despite FMA hardware)
  - `sub()` - Subtraction (~1x speedup, memory-bound)
  - `div()` - Division (~1x speedup, memory-bound)
  - `abs()` - Absolute value (~1.1x speedup, memory-bound)
  - `min()` - Minimum reduction (6-10x SIMD speedup)

### Documentation
- **Performance pattern analysis documented**:
  - **Compute-bound operations** (4-12x SIMD benefit): min, argmax/argmin, norm_l1, norm_l2, dot, sum
  - **Memory-bound operations** (~1x SIMD benefit): sub, div, fma, scale, abs
  - Root cause: Memory bandwidth saturation prevents SIMD benefit for simple operations

### Testing
- All 889 tests passing (759 unit + 21 integration + 109 doc)
- Zero clippy warnings
- EXTREME TDD methodology with RED-GREEN-REFACTOR cycle applied for abs()

### Closes
- Issue #2: Missing abs trait implementation in VectorBackend

---

## [0.2.1] - 2025-11-18

### Added

#### Activation Functions
- `hardswish()` - MobileNetV3 efficient activation
- `mish()` - Modern swish alternative (x * tanh(softplus(x)))
- `selu()` - Self-normalizing exponential linear unit
- `relu()` - ReLU with EXTREME TDD

#### Math Operations
- `log2()` - Base-2 logarithm (information theory, entropy)
- `log10()` - Base-10 logarithm (decibels, pH)

#### Documentation
- Comprehensive GPU performance analysis (`docs/performance-analysis.md`)
- Performance baselines for regression detection

### Changed

#### Critical GPU Performance Optimization
- **GPU disabled for ALL element-wise operations** (2-65,000x slower than scalar!)
- **GPU enabled ONLY for matmul** (2-10x speedup at 500×500+)
- Updated OpComplexity thresholds based on empirical benchmarks
- Lowered matmul GPU threshold from 1000 to 500 (proven 2x speedup)

#### Documentation Updates
- README updated with honest GPU performance claims
- ROADMAP pivoted from GPU to SIMD optimization strategy

### Fixed
- False GPU speedup claims (advertised 10-50x, actual was 2-65,000x SLOWER)
- GPU overhead analysis: 14-55ms fixed cost per operation

### Performance

#### GPU Benchmark Results (Empirical - Genchi Genbutsu)
| Operation | Size | GPU vs Scalar | Result |
|-----------|------|---------------|--------|
| vec_add | 1M | 510x SLOWER | ❌ GPU disabled |
| dot | 1M | 93x SLOWER | ❌ GPU disabled |
| relu | 1M | 824x SLOWER | ❌ GPU disabled |
| matmul | 500×500 | **2.01x faster** | ✅ GPU enabled |
| matmul | 1000×1000 | **9.59x faster** | ✅ GPU enabled |

**Root Cause**: 14-55ms GPU overhead (buffer allocation + PCIe transfer) dominates execution time for element-wise ops.

### Testing
- 33 new tests for activations (hardswish, mish, selu)
- 14 new tests for log2/log10
- Property-based tests for all new functions
- Total: 699+ tests

### Closes
- Issue #1: Element-wise transcendental functions (log2, ln, exp)

---

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

### Planned for v0.3.0
- SIMD-optimized activation functions (AVX2/AVX-512)
- Performance regression CI integration
- Matrix-vector multiplication
- Additional backends (WASM SIMD128)

[0.2.2]: https://github.com/paiml/trueno/releases/tag/v0.2.2
[0.2.1]: https://github.com/paiml/trueno/releases/tag/v0.2.1
[0.1.0]: https://github.com/paiml/trueno/releases/tag/v0.1.0
[Unreleased]: https://github.com/paiml/trueno/compare/v0.2.2...HEAD
