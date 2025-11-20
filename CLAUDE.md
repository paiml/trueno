# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Trueno** (Spanish: "thunder") is a Rust library providing unified, high-performance compute primitives across three execution targets:

1. **CPU SIMD** - x86 (SSE2/AVX/AVX2/AVX-512), ARM (NEON), WASM (SIMD128)
2. **GPU** - Vulkan/Metal/DX12/WebGPU via `wgpu`
3. **WebAssembly** - Portable SIMD128 for browser/edge deployment

**Core Principles**:
- Write once, optimize everywhere: Single algorithm, multiple backends
- Runtime dispatch: Auto-select best implementation based on CPU features
- Zero unsafe in public API: Safety via type system, `unsafe` isolated in backends
- Benchmarked performance: Every optimization must prove ≥10% speedup
- Extreme TDD: >90% test coverage, mutation testing, property-based tests

## Development Commands

### Building
```bash
# Standard build
cargo build

# Release build (optimized)
cargo build --release

# Build with all features
cargo build --all-features

# Build for WASM
cargo build --target wasm32-unknown-unknown
```

### Testing
```bash
# Run all tests
cargo test --all-features

# Run tests for a specific module
cargo test vector

# Run tests with output
cargo test -- --nocapture

# Run property-based tests
cargo test property_tests

# Run backend equivalence tests
cargo test backend_equivalence

# Run integration tests
cargo test --test integration_tests
```

### Coverage
```bash
# Generate coverage report
cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info

# View coverage report
cargo llvm-cov report

# Coverage must be ≥90%
cargo llvm-cov --all-features --workspace --fail-under-lines 90
```

### Linting
```bash
# Run clippy (no warnings allowed)
cargo clippy --all-features -- -D warnings

# Format code
cargo fmt

# Check formatting without modifying
cargo fmt -- --check
```

### Benchmarking
```bash
# Run all benchmarks
cargo bench --no-fail-fast

# Run specific benchmark
cargo bench vector_ops

# Compare with baseline
cargo bench -- --save-baseline main
cargo bench -- --baseline main
```

### Profiling
```bash
# Install Renacer v0.5.0+ (syscall tracing and function profiling)
cargo install renacer

# Profile benchmarks to identify bottlenecks
make profile

# Generate flamegraph visualization
make profile-flamegraph

# Profile specific benchmark
make profile-bench BENCH=vector_ops

# Profile test suite to find slow tests
make profile-test

# Advanced: I/O bottleneck detection (>1ms threshold)
renacer --function-time --source -- cargo bench vector_ops

# Advanced: Generate flamegraph from profiling output
renacer --function-time --source -- cargo bench | flamegraph.pl > flame.svg
```

**Profiling Use Cases**:
- **SIMD Validation**: Verify SIMD optimizations show expected speedups
- **Backend Selection**: Identify if backend dispatch overhead is significant
- **Hot Path Analysis**: Find top 10 functions consuming most time
- **Memory Access**: Detect cache misses and memory bottlenecks
- **GPU Transfer**: Profile PCIe transfer overhead for GPU backend

### Distributed Tracing with OpenTelemetry (Renacer v0.5.0+)

**NEW:** Export syscall traces to observability backends (Jaeger, Grafana Tempo, etc.)

```bash
# Profile with Jaeger (easiest - single Docker container)
make profile-otlp-jaeger

# View traces at: http://localhost:16686
# Stop Jaeger: docker stop jaeger-trueno && docker rm jaeger-trueno

# Profile with Grafana Tempo (production-ready stack)
make profile-otlp-tempo

# View traces at: http://localhost:3000 (admin/admin)
# Stop stack: docker-compose -f docs/profiling/docker-compose-tempo.yml down
```

**OTLP Features**:
- **Span Hierarchy**: Process root span → syscall child spans
- **Rich Attributes**: syscall name, result, duration, source location (file:line)
- **Distributed Context**: Trace benchmark execution across all syscalls
- **Integration**: Works with all Renacer features (--source, -T, --function-time)
- **Backends**: Jaeger, Grafana Tempo, Elastic APM, Honeycomb, any OTLP-compatible collector

**Use Cases**:
- **End-to-End Visibility**: See entire benchmark execution timeline
- **Cross-Service Tracing**: Correlate Trueno benchmarks with production traces
- **Performance Regression Detection**: Compare trace spans across releases
- **Team Collaboration**: Share trace links for performance discussions

**OTLP Profiling Best Practices** (Institutionalized Workflow):

1. **Pre-Release Performance Validation**
   ```bash
   # Baseline current release
   make profile-otlp-jaeger
   curl "localhost:16686/api/traces?service=trueno-benchmarks" > traces-v0.4.0.json

   # After changes
   make profile-otlp-jaeger
   curl "localhost:16686/api/traces?service=trueno-benchmarks" > traces-v0.4.1.json

   # Compare syscall distributions
   python3 scripts/compare_traces.py traces-v0.4.0.json traces-v0.4.1.json
   ```

2. **Debug Performance Regression**
   - **Symptom**: Benchmark shows slowdown
   - **Action**: Profile with `make profile-otlp-jaeger`
   - **Investigate**: Check for unexpected syscalls (mmap, futex, munmap)
   - **Validate**: Zero-allocation in hot path (no mmap during compute)
   - **Fix**: Reduce syscall overhead, pre-allocate buffers
   - **Verify**: Re-profile and compare trace data

3. **Team Collaboration Protocol**
   - Share Jaeger UI link: `http://localhost:16686/trace/<trace-id>`
   - Export trace JSON for async review: `curl "localhost:16686/api/traces?..."`
   - Tag releases in Grafana Tempo for historical comparison
   - Include trace links in performance PRs

4. **CI/CD Integration**
   ```yaml
   # .github/workflows/performance.yml
   - name: Profile with OTLP
     run: make profile-otlp-export  # Exports traces to S3/GCS
   - name: Compare with baseline
     run: make profile-compare BASELINE=main
   ```

5. **Production Observability**
   - Deploy Grafana Tempo in staging/production
   - Export Trueno operation traces alongside API traces
   - Correlate slow requests with specific syscalls
   - Alert on unexpected syscall patterns (e.g., >10 mmap per request)

**Key Insights from Empirical Analysis** (Renacer 0.5.0):
- **Futex overhead**: Thread sync dominates for <1μs operations (up to 22x slowdown)
- **Test harness cost**: Cargo test adds 0.9ms overhead (1600x for 547ns operation)
- **Zero-allocation validation**: Confirmed no mmap/munmap in hot path
- **Failed syscalls**: 19 statx ENOENT errors during test discovery (expected)
- **Recommendation**: Use raw binaries for micro-benchmarks, avoid async for <10μs ops

### Quality Gates
```bash
# PMAT Technical Debt Grading (minimum: B+ / 85/100)
pmat analyze tdg --min-grade B+

# Repository health score (minimum: 90/110)
pmat repo-score . --min-score 90

# Mutation testing (minimum: 80% kill rate)
cargo mutants --timeout 120 --minimum-pass-rate 80
```

## Architecture

### Multi-Backend Design

```
┌─────────────────────────────────────────────────┐
│           Trueno Public API (Safe)              │
│  compute(), map(), reduce(), transform()        │
└─────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌────────┐   ┌─────────┐   ┌──────────┐
   │  SIMD  │   │   GPU   │   │   WASM   │
   │ Backend│   │ Backend │   │  Backend │
   └────────┘   └─────────┘   └──────────┘
        │             │             │
   ┌────┴────┐   ┌────┴────┐   ┌───┴─────┐
   │ Runtime │   │  wgpu   │   │ SIMD128 │
   │ Detect  │   │ Compute │   │ Portable│
   └─────────┘   └─────────┘   └─────────┘
   │  │  │  │
   SSE2 AVX  NEON AVX512
```

### Backend Selection Priority

1. GPU (if available + workload size > 100,000 elements)
2. AVX-512 (if CPU supports)
3. AVX2 (if CPU supports)
4. AVX (if CPU supports)
5. SSE2 (baseline x86_64)
6. NEON (ARM64)
7. SIMD128 (WASM)
8. Scalar fallback

### Project Structure

```
src/
├── lib.rs                  # Public API exports
├── error.rs                # TruenoError types
├── vector.rs               # Vector<T> type and VectorOps trait
├── backend/
│   ├── mod.rs              # Backend enum and dispatch logic
│   ├── scalar.rs           # Scalar fallback (baseline correctness)
│   ├── simd/
│   │   ├── mod.rs          # SIMD backend selection
│   │   ├── sse2.rs         # x86_64 baseline (guaranteed available)
│   │   ├── avx.rs          # 256-bit operations
│   │   ├── avx2.rs         # 256-bit with FMA
│   │   ├── avx512.rs       # 512-bit (Zen4/Sapphire Rapids+)
│   │   ├── neon.rs         # ARM64 SIMD
│   │   └── wasm.rs         # WASM SIMD128
│   └── gpu/
│       ├── mod.rs          # GPU device management
│       ├── device.rs       # wgpu integration
│       └── shaders/
│           └── vector_add.wgsl  # Compute shaders
└── utils/
    ├── mod.rs
    └── cpu_detect.rs       # Runtime CPU feature detection
```

### Key Implementation Patterns

**SIMD Backend Pattern**:
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
unsafe fn add_f32_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    // Process 8 elements at a time (256-bit / 32-bit = 8)
    let chunks = a.len() / 8;
    for i in 0..chunks {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        let result = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(out.as_mut_ptr().add(i * 8), result);
    }
    // Handle remainder with scalar fallback
    for i in (chunks * 8)..a.len() {
        out[i] = a[i] + b[i];
    }
}
```

**GPU Dispatch Pattern**:
```rust
pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

// Only use GPU for large workloads (transfer overhead)
const GPU_MIN_SIZE: usize = 100_000;

fn should_use_gpu(size: usize) -> bool {
    size >= GPU_MIN_SIZE && gpu_available()
}
```

## Testing Requirements

### Coverage Standards

| Component | Minimum Coverage | Target Coverage |
|-----------|-----------------|-----------------|
| Public API | 100% | 100% |
| SIMD backends | 90% | 95% |
| GPU backend | 85% | 90% |
| WASM backend | 90% | 95% |
| **Overall** | **90%** | **95%+** |

### Test Categories

1. **Unit Tests** - Correctness for all operations
   - Empty inputs, single element, non-aligned sizes
   - Edge cases: NaN, infinity, subnormal numbers

2. **Property-Based Tests** (using `proptest`)
   - Commutativity: `a + b == b + a`
   - Associativity: `(a + b) + c == a + (b + c)`
   - Distributivity: `a * (b + c) == (a * b) + (a * c)`

3. **Backend Equivalence Tests**
   - All backends must produce identical results
   - Compare scalar vs SSE2 vs AVX2 vs GPU vs WASM
   - Floating-point tolerance: `< 1e-5` for f32

4. **Mutation Testing**
   - Must achieve ≥80% mutation kill rate
   - Run with: `cargo mutants --timeout 120`

5. **Benchmark Tests**
   - Every optimization must prove ≥10% speedup
   - Test sizes: 100, 1K, 10K, 100K, 1M, 10M elements
   - Compare against scalar baseline

### Writing Tests

Always include all test categories when adding new operations:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_correctness() { /* ... */ }

    #[test]
    fn test_add_empty() { /* ... */ }

    #[test]
    fn test_add_non_aligned() { /* ... */ }
}

#[cfg(test)]
mod property_tests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_add_commutative(
            a in prop::collection::vec(-1000.0f32..1000.0, 1..10000),
            b in prop::collection::vec(-1000.0f32..1000.0, 1..10000)
        ) {
            // Test implementation
        }
    }
}

#[test]
fn test_backend_equivalence() {
    let a = vec![1.0f32; 10000];
    let b = vec![2.0f32; 10000];

    let scalar = add_vectors_scalar(&a, &b);
    let sse2 = unsafe { add_vectors_sse2(&a, &b) };
    let avx2 = unsafe { add_vectors_avx2(&a, &b) };

    assert_eq!(scalar, sse2);
    assert_eq!(scalar, avx2);
}
```

## Quality Standards (EXTREME TDD)

### Every Commit Must:
- ✅ Compile without warnings (`cargo clippy -- -D warnings`)
- ✅ Pass all tests (`cargo test --all-features`)
- ✅ Maintain >90% coverage (`cargo llvm-cov`)
- ✅ Pass rustfmt (`cargo fmt -- --check`)
- ✅ Pass PMAT TDG ≥B+ (`pmat analyze tdg --min-grade B+`)

### Every PR Must:
- ✅ Include tests for new functionality (all 5 categories)
- ✅ Update rustdoc documentation
- ✅ Benchmark new optimizations (prove ≥10% improvement)
- ✅ Pass mutation testing (≥80% kill rate)
- ✅ Include integration test if adding backend

### Every Release Must:
- ✅ Pass full CI pipeline
- ✅ Repository score ≥90/110 (`pmat repo-score`)
- ✅ Changelog updated (keep-a-changelog format)
- ✅ Version bumped (semver)
- ✅ Git tag created (`vX.Y.Z`)

## Safety Rules

### Unsafe Usage
- `unsafe` is ONLY allowed in backend implementations (never in public API)
- Every `unsafe` block must have safety comment explaining invariants
- SIMD intrinsics must be wrapped in `#[target_feature]` functions
- All public APIs must be safe (bounds-checked, validated inputs)

### Example Safe Wrapper:
```rust
// SAFE public API
pub fn add_f32(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        return Err(TruenoError::SizeMismatch {
            expected: a.len(),
            actual: b.len()
        });
    }

    let mut result = vec![0.0; a.len()];

    // UNSAFE internal implementation (isolated)
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { add_f32_avx2(a, b, &mut result) };
        return Ok(result);
    }

    // Safe scalar fallback
    add_f32_scalar(a, b, &mut result);
    Ok(result)
}
```

## Performance Targets

### Expected Speedups (vs Scalar Baseline)

| Operation | Size | SSE2 | AVX2 | AVX-512 | GPU | WASM SIMD |
|-----------|------|------|------|---------|-----|-----------|
| add_f32 | 1K | 2x | 4x | 8x | - | 2x |
| add_f32 | 100K | 2x | 4x | 8x | 3x | 2x |
| add_f32 | 1M | 2x | 4x | 8x | 10x | 2x |
| add_f32 | 10M | 2x | 4x | 8x | 50x | - |
| dot_product | 1K | 3x | 6x | 12x | - | 3x |
| dot_product | 1M | 3x | 6x | 12x | 20x | 3x |

### Benchmark Validation
- Minimum 100 iterations per benchmark
- Coefficient of variation (CV) must be <5%
- No regressions >5% compared to previous baseline
- Results saved to `target/criterion/` for comparison

## Ecosystem Integration

Trueno integrates with the Pragmatic AI Labs transpiler ecosystem:

1. **Ruchy** - Language-level vector operations
   - `let v = Vector([1.0, 2.0]) + Vector([3.0, 4.0])` → `trueno::Vector::add()`

2. **Depyler** (Python → Rust)
   - `np.dot(a, b)` → `trueno::Vector::dot(&a, &b)`

3. **Decy** (C → Rust)
   - `_mm256_add_ps()` → `trueno::Vector::add()` (eliminates unsafe)

4. **ruchy-lambda** - AWS Lambda optimization
   - Drop-in performance boost for data processing

5. **ruchy-docker** - Cross-language benchmarking
   - Prove transpiler-generated code matches hand-written performance

6. **paiml-mcp-agent-toolkit (PMAT)** - Quality gates
   - Pre-commit hooks enforce >90% coverage
   - TDG grading (target: A- / 92+)

7. **Renacer** - Syscall tracing and function profiling
   - Identify performance bottlenecks and hot paths
   - I/O bottleneck detection (>1ms threshold)
   - Flamegraph generation for visualization
   - Validate SIMD optimizations show expected speedups

## Documentation Standards

### Rustdoc Requirements
- 100% coverage of public API
- Every function has example code that compiles
- Document panics, errors, safety invariants
- Performance characteristics documented

### Example:
```rust
/// Add two vectors element-wise using optimal SIMD backend.
///
/// # Performance
///
/// Auto-selects the best available backend:
/// - **AVX2**: ~4x faster than scalar for 1K+ elements
/// - **GPU**: ~50x faster than scalar for 10M+ elements
///
/// # Examples
///
/// ```
/// use trueno::Vector;
///
/// let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
/// let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
/// let result = a.add(&b).unwrap();
///
/// assert_eq!(result.as_slice(), &[5.0, 7.0, 9.0]);
/// ```
///
/// # Errors
///
/// Returns [`TruenoError::SizeMismatch`] if vectors have different lengths.
pub fn add(&self, other: &Self) -> Result<Self> {
    // Implementation
}
```

## Rationale: Why Assembly/SIMD Matters

**FFmpeg Case Study** (real-world evidence):
- **390 assembly files**, ~180,000 lines (11% of codebase)
- **Speedups**: SSE2 (2-4x), AVX2 (4-8x), AVX-512 (8-16x)
- **Critical operations**: IDCT transforms, motion compensation, deblocking filters

**Why Not Hand-Written Assembly?**
- ❌ Unsafe (raw pointers, no bounds checking)
- ❌ Unmaintainable (390 files, platform-specific)
- ❌ Non-portable (separate implementations per CPU)

**Trueno's Value**:
- ✅ Safety: Zero unsafe in public API
- ✅ Portability: Single source → x86/ARM/WASM
- ✅ Performance: 85-95% of hand-tuned assembly
- ✅ Maintainability: Rust type system catches errors

## Common Pitfalls

1. **Don't forget remainder handling in SIMD loops**
   - AVX2 processes 8 f32s at a time
   - Must handle `len % 8` with scalar fallback

2. **GPU transfer overhead**
   - Only use GPU for >100K elements
   - PCIe transfer costs ~0.5ms

3. **Floating-point precision**
   - SIMD can reorder operations (different rounding)
   - Use tolerance `< 1e-5` for f32 comparisons

4. **Target feature detection**
   - Always check `is_x86_feature_detected!()` before using intrinsics
   - Wrap intrinsics in `#[target_feature]` functions

5. **WASM limitations**
   - SIMD128 only (4x f32), not 8x like AVX2
   - No GPU support in standard WASM (WebGPU is separate)

## Toyota Way & Kaizen Improvements

This project follows Toyota Production System principles:

### Jidoka (Built-in Quality)
- EXTREME TDD (>90% coverage) builds quality in, doesn't inspect it in later
- Pre-commit hooks act as "Andon cord" - stop the line on defects
- Mutation testing catches defects traditional unit tests miss

### Kaizen (Continuous Improvement)
- Every optimization must prove ≥10% speedup (data-driven)
- Backend selection optimized to resolve once at Vector creation (eliminates redundant CPU detection)
- OpComplexity explicitly defined to prevent GPU threshold mistakes

### Key Improvements Applied

1. **Backend Selection Efficiency** (v1.0.0)
   - `Backend::Auto` resolved at Vector creation, not on every operation
   - Eliminates redundant CPU feature detection
   - See: `Vector::from_slice()` implementation

2. **OpComplexity Definition** (v1.0.0)
   - Low: Simple operations (add, mul) - prefer SIMD
   - Medium: Moderate operations (dot, reduce) - GPU at 100K+
   - High: Complex operations (matmul, conv2d) - GPU at 10K+

3. **Future: Async GPU API** (planned v2.0)
   - Current synchronous API simple but inefficient for chained operations
   - Future async API will enable operation batching to reduce transfer overhead

### Academic Foundations

Key publications informing Trueno's design:
- **Halide (PLDI 2013)**: Write once, optimize everywhere philosophy
- **Rayon (PLDI 2017)**: Safe zero-cost abstractions in Rust
- **WebAssembly (PLDI 2017)**: WASM SIMD performance model
- **TVM (OSDI 2018)**: Multi-target compiler architecture

See specification section 16.3 for complete list with links.

## Trueno Analyze Tool (`trueno-analyze`)

**Purpose**: Static analysis and runtime profiling tool to identify vectorization opportunities in existing code.

### Usage

```bash
# Analyze Rust source code
trueno-analyze --source ./src --lang rust

# Profile binary to find hotspots
trueno-analyze --profile ./target/release/myapp --duration 30s

# Generate flamegraph
trueno-analyze --profile ./myapp --flamegraph --output report.svg

# Analyze for transpiler integration
trueno-analyze --source ./src --lang python --transpiler depyler --output json
```

### Analysis Modes

**Mode 1: Static Analysis**
- Detects vectorizable patterns (scalar loops, iterator chains, SIMD intrinsics)
- Identifies existing unsafe SIMD code that could be replaced with safe Trueno API
- Estimates speedup potential (2-50x depending on operation and backend)
- Suggests specific Trueno functions to use

**Mode 2: Binary Profiling** (perf + DWARF)
- Profiles runtime execution to find hotspots (>5% runtime)
- Analyzes assembly to detect missed auto-vectorization
- Correlates with source code using debug symbols
- Recommends GPU usage for large workloads

**Mode 3: Transpiler Integration**
- Guides Depyler/Decy on which operations to transpile to Trueno
- Outputs JSON for automated tooling
- Confidence scores for each suggestion

### Example Output

```
Trueno Analysis Report
======================
Project: image-processor v0.3.0

VECTORIZATION OPPORTUNITIES: 5
===============================

[1] src/filters/blur.rs:234-245
    Pattern: Scalar element-wise multiply-add
    Suggestion: trueno::Vector::mul().add()
    Est. Speedup: 4-8x (AVX2)
    LOC to change: 3 lines

[2] src/math/matmul.rs:45-67
    Pattern: Naive matrix multiplication
    Suggestion: trueno::matmul() [Phase 2]
    Est. Speedup: 10-50x (GPU for large matrices)
    GPU Eligible: Yes (matrix size > 1000x1000)

SUMMARY
=======
Total Opportunities: 5
Estimated Overall Speedup: 3.2-6.8x
Estimated Effort: 42 LOC to change
Safety Wins: 37 lines of unsafe eliminated
```

### Key Features

**Pattern Detection**:
- Element-wise operations (add, mul, sub, div)
- Dot products and reductions
- Matrix multiplication
- Existing SIMD intrinsics (AVX2, SSE2, NEON)
- NumPy operations (for Python/Depyler)

**Speedup Estimation**:
- Backend-specific models (SSE2: 2-4x, AVX2: 4-8x, GPU: 10-50x)
- Accounts for memory access patterns (sequential vs strided vs random)
- GPU transfer overhead modeling
- Conservative to optimistic range

**CI Integration**:
- GitHub Actions workflow for PR analysis
- JSON output for automated tooling
- Posts PR comments with optimization suggestions

### Development Roadmap

- **v1.1**: Static analysis (Rust AST, pattern database)
- **v1.2**: Binary profiling (perf, DWARF, flamegraphs)
- **v1.3**: Multi-language support (C, Python)
- **v1.4**: ML-based pattern detection, automated migration tool

See specification section 17 for complete details.
