# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.1] - 2025-11-24

### Added ✨

- **EXTREME PMAT Integration** - O(1) Quality Gates for automated quality enforcement
- **Golden Trace Validation** - Syscall-level performance regression detection with Renacer v0.6.2+
- **GPU Batch API Example** - Demonstration of 3x transfer reduction for chained operations

### Fixed 🐛

- Replaced `.unwrap()` with `.expect()` in examples for better error messages
- Corrected relative paths in golden-trace-validation.md documentation

### Infrastructure 🔧

- GitHub Actions workflow for automated golden trace validation
- Enhanced gitignore for benchmark logs

### Dependencies 📦

- Updated all dependencies to latest versions (wgpu 27.0.1, criterion 0.7, thiserror 2.0.17)

### Quality 🎯

- Test coverage: 90.41% (exceeds 90% requirement)
- 942 tests passing (up from 936)
- All quality gates passing
- Pre-commit hooks enforce coverage threshold

## [0.7.0] - 2025-11-22

### Performance - Phase 3: Large Matrix Optimization 🚀

**Achievement**: 18% improvement for 1024×1024 matrices via 3-level cache blocking

- **3-level cache hierarchy** (L3 → L2 → micro-kernel) for matrices ≥512×512
  - L3 blocks: 256×256 (fits in 4-16MB L3 cache)
  - L2 blocks: 64×64 (fits in 256KB L2 cache)
  - Micro-kernel: 4×1 AVX2/FMA (register blocking)
  - Smart threshold: Only activates for matrices ≥512×512

- **Zero-allocation implementation**:
  - No Vec allocations in hot path
  - Code duplication with if/else branches
  - Preserves fast 2-level path for smaller matrices

- **Performance results**:
  - 1024×1024: **47.4 ms (18% faster than v0.6.0's 57.8 ms)** ✅
  - 512×512: ~5.3 ms (8.5% improvement)
  - 256×256: No regression (uses 2-level path)
  - Target: Within 1.5× of NumPy (currently 1.64×)

- **Testing**:
  - Added `test_matmul_3level_blocking` for 512×512 matrices
  - 878 tests passing (all existing tests pass)
  - Coverage: 90.41% (improved from 90.00%)

### Quality & Testing

- **Test coverage: 90.26%** (trueno library, exceeds 90% EXTREME TDD requirement)
- Added 60+ new tests across xtask tooling and core library
- Fixed clippy warnings (needless_range_loop)
- Updated coverage policy: xtask (dev tooling) excluded from main coverage requirement
- All quality gates passing: lint, format, tests, coverage

### Documentation

- Updated Phase 2 book chapter with 3-level blocking details
- Added benchmark data for 512×512 and 1024×1024
- GitHub issue #34 tracking Phase 3 progress

## [0.6.0] - 2025-11-21

### Performance - Phase 2: NumPy Performance Parity 🎯

**Major Achievement**: Pure Rust matches NumPy/OpenBLAS performance at 256×256 matrices

- **4×1 AVX2 micro-kernel** implementation (Pure Rust, zero external dependencies)
  - Fused Multiply-Add (FMA) instructions for 3× throughput
  - Register blocking: 4 YMM accumulators stay in CPU registers
  - Eliminates memory traffic, maximizes compute utilization

- **2-level cache blocking** (outer loop: L2, inner loop: L1)
  - Outer blocks: 64×64 (fits in L2 cache)
  - Inner blocks: 4×4 (micro-kernel size, stays in registers)
  - Adaptive based on matrix size

- **Performance results**:
  - 256×256: **7.3 ms** (matches NumPy/OpenBLAS's 7.3 ms) ✅
  - 128×128: **0.9 ms** (vs NumPy 0.9 ms - parity achieved)
  - 64×64: **0.12 ms** (vs NumPy 0.12 ms - parity)
  - Validates Phase 2 goal: **pure Rust can match C/Fortran + assembly**

- **Algorithm validation**:
  - Correctness: `test_matmul_simd_equivalence_large` with 100×100 matrices
  - No regressions: All 843 tests passing
  - Coverage: 90.00% (meets EXTREME TDD requirement)

### Documentation

- Added Phase 2 book chapter documenting micro-kernel design
- Updated performance benchmark tables with Phase 2 results
- Added "Pragmatic Parity" definition to glossary

## Earlier Releases

For earlier releases, see the [CHANGELOG.md](https://github.com/paiml/trueno/blob/main/CHANGELOG.md) in the repository root.

---

**Installation:**

```bash
cargo add trueno
```

**Links:**
- [📦 crates.io](https://crates.io/crates/trueno)
- [📚 Documentation](https://docs.rs/trueno)
- [🏠 Repository](https://github.com/paiml/trueno)
