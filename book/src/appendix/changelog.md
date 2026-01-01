# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [trueno-gpu 0.4.3] - 2026-01-01

### Performance

- **PTX Emission Optimization** - 20.9% improvement in PTX code generation
  - Pre-allocated String capacity based on instruction count
  - Zero-allocation `write_instruction()` writes directly to buffer
  - Zero-allocation `write_operand()` and `write_mem_operand()` helpers
  - Added `Display` impl for `VirtualReg` enabling `write!()` formatting
  - Throughput: 68,316 kernels/sec

### Added

- **Kernel Generation Benchmark** - New example `bench_kernel_gen`
  - Benchmarks all kernel types: GEMM, Softmax, LayerNorm, Attention, Quantize
  - Measures generation time, PTX size, and throughput

- **Performance Whitelist** - `PtxBugAnalyzer::with_performance_whitelist()`
  - Documents expected register pressure in high-performance kernels
  - Whitelists Tensor Core, Attention, and Quantized kernel patterns
  - Separates "expected performance tradeoffs" from actual bugs

### Fixed

- **Barrier Safety Analyzer** - Fixed false positives in quantized kernels
  - Now recognizes `*_done` suffix labels as loop ends (not just `*_end`)
  - Added explicit patterns: `sb_loop_done`, `sub_block_done`, `k_block_done`
  - All 22 barrier safety tests pass

## [trueno-gpu 0.4.2] - 2026-01-01

### Fixed

- **PARITY-114: Barrier Safety Bug** - Fixed thread divergence causing CUDA error 700
  - Root cause: Threads exiting early before `bar.sync` barriers caused remaining threads to hang
  - Fixed 4 kernels: `gemm_tensor_core`, `gemm_wmma_fp16`, `flash_attention`, `flash_attention_tensor_core`
  - Fix pattern: Predicated loads (store 0 first), bounds check AFTER loop, all threads participate in barriers

### Added

- **Barrier Safety Analyzer** - Static PTX analysis (PARITY-114 prevention)
  - `barrier_safety.rs` - Detects early-exit-before-barrier patterns
  - `Kernel::analyze_barrier_safety()` - Analyze any kernel for violations
  - `Kernel::emit_ptx_validated()` - Production-ready PTX with safety check
  - 19 barrier safety tests (9 analyzer + 10 kernel validation)

- **Boundary Condition Tests** - Test dimensions not divisible by tile size
  - GEMM: 17×17, 33×33, 100×100, single row/column
  - Attention: seq_len=17, 33, 100
  - Prevents future PARITY-114 regressions

- **CI Target** - `make barrier-safety` for automated validation

### Changed

- Specification updated to v1.5.0 with 15 new falsification tests (§5.8)
- Overall test count: 452 tests (up from 441)

## [trueno-gpu 0.4.1] - 2026-01-01

### Added

- **PTX Optimization Passes** - NVIDIA CUDA Tile IR aligned (v1.4.0 spec)
  - `loop_split.rs` - Loop splitting with profitability analysis (99.80% coverage)
  - `tko.rs` - Token-Based Ordering for memory dependencies (94.29% coverage)
  - Exported `CmpOp` and `Operand` in public API
  - New example: `ptx_optimize` demonstrating all optimization passes

- **Book Chapter** - [PTX Optimization Passes](../architecture/ptx-optimization.md)
  - FMA Fusion, Loop Splitting, TKO, Tile Validation documentation
  - Academic references and NVIDIA CUDA Tile IR alignment

### Changed

- Overall test coverage: 94.28% (57 optimize module tests)

## [trueno-gpu 0.4.0] - 2026-01-01

### Fixed

- **WMMA Tensor Core Attention** - Fixed four PTX bugs enabling Tensor Core attention on RTX 4090
  - Register prefix conflict: B32 registers now use `%rb` prefix instead of `%r`
  - Zero initialization: Use `mov.f32` instead of loading from NULL pointer
  - FP16 shared memory store: Use B16 type for 16-bit stores
  - Address conversion: Added `cvta.shared.u64` for WMMA generic pointer requirement
  - Added `Cvta` operation to PtxOp enum for address space conversion

### Added

- **Tensor Core Validation Tests** - New kernel validation tests
  - `tensor_core_attention_ptx_structure` - Verifies WMMA instructions and cvta.shared.u64
  - `tensor_core_attention_ptx_validate_with_ptxas` - Validates PTX with NVIDIA ptxas

### Performance

- Tensor Core attention benchmarked on RTX 4090:
  - 64x64: 8.7 GFLOPS (1.01x vs FP32)
  - 256x64: 80.0 GFLOPS (1.06x vs FP32)
  - 512x64: 202.5 GFLOPS (1.03x vs FP32)

## [0.9.0] - 2025-12-31

### Added

- **CUDA Tile GPU Optimizations** - Major performance improvements for GPU kernels
- **TensorView and PartitionView** - New abstractions for tiled reduction

## [0.8.7] - 2025-12-16

### Changed

- **Dependencies**: Updated trueno-gpu to 0.2.2

## [trueno-explain 0.2.0] - 2025-12-16

### Added

- **PTX Bug Detection** - Static analysis for PTX to catch common bugs
  - 12 bug classes across 3 severity levels (P0 Critical, P1 High, P2 Medium)
  - `PtxBugAnalyzer` with default, strict, and whitelist modes
  - Detects: shared memory addressing bugs, missing barriers, register pressure, placeholder code, dead code, empty loops, missing bounds checks
  - `with_quantized_whitelist()` for Q4K/Q5K/Q6K/Q8K kernels
  - Coverage tracking with `PtxCoverageTracker`

- **Examples**
  - `deep_bug_hunt` - Analyze all trueno-gpu kernels (30 kernels)
  - `analyze_realizar` - Analyze external hand-rolled PTX
  - `ptx_inspector` - Deep dive into specific kernel PTX

### Documentation

- New chapter: [PTX Bug Detection](../development/ptx-bug-detection.md)
- 190 new tests for bug detection

## [trueno-gpu 0.2.2] - 2025-12-16

### Changed

- **Internal**: Reduced predicate pressure in tiled GEMM by using two branches instead of `and_pred`
- No API changes

## [0.7.3] - 2025-11-25

### Added ✨

- **WebGPU for WASM** (`gpu-wasm` feature)
  - Cross-platform GPU compute: native and browser support
  - Async-first API: all GPU operations have `*_async` variants
  - Runtime detection via `runtime::sync_available()`
  - Enables [trueno-viz](https://github.com/paiml/trueno-viz) browser-based visualization

- **Cross-platform GPU API**
  - `GpuDevice::new_async()` - Works on all platforms
  - All operations have async variants (`relu_async`, `matmul_async`, etc.)

### Documentation 📚

- Complete rewrite of [GPU Backend](../architecture/gpu-backend.md) chapter
- Added WebGPU/WASM section to [GPU Performance](../performance/gpu-performance.md)
- trueno-viz integration examples

### Fixed 🐛

- Type inference fixes for empty slice comparisons
- Parameter naming in `select_backend_for_operation`

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
