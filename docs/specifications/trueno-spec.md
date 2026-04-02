# Trueno Specification

**Unified high-performance compute primitives across CPU SIMD, NVIDIA CUDA, wgpu, and WebAssembly.**

Version 1.0 · April 2026 · Pragmatic AI Labs · [paiml/trueno](https://github.com/paiml/trueno)

---

> **Canonical spec.** This is the ONE specification for trueno. Detail lives in
> `docs/specifications/sub/`. Old specs in `docs/specifications/*.md` are
> superseded by this document — do not create new top-level spec files.

---

## Table of Contents

| # | Section | Sub-spec |
|---|---------|----------|
| 1 | [Philosophy](#1-philosophy) | |
| 2 | [Provable-Contract-First Design](#2-provable-contract-first-design) | [sub/contracts.md](sub/contracts.md) |
| 3 | [Multi-Backend Architecture](#3-multi-backend-architecture) | [sub/backends.md](sub/backends.md) |
| 4 | [Backend Story Policy](#4-backend-story-policy) | [sub/backends.md](sub/backends.md) |
| 5 | [CPU SIMD Backends](#5-cpu-simd-backends) | [sub/simd.md](sub/simd.md) |
| 6 | [CUDA Backend (trueno-gpu)](#6-cuda-backend-trueno-gpu) | [sub/cuda.md](sub/cuda.md) |
| 7 | [wgpu Backend](#7-wgpu-backend) | [sub/wgpu.md](sub/wgpu.md) |
| 8 | [WASM Backend](#8-wasm-backend) | [sub/simd.md](sub/simd.md) |
| 9 | [Layout Mandate (Q4K/Q6K)](#9-layout-mandate-q4kq6k) | [sub/layout.md](sub/layout.md) |
| 10 | [Crate Architecture](#10-crate-architecture) | |
| 11 | [Quality Gates](#11-quality-gates) | [sub/quality.md](sub/quality.md) |
| 12 | [Testing Requirements](#12-testing-requirements) | [sub/quality.md](sub/quality.md) |
| 13 | [Coverage](#13-coverage) | [sub/quality.md](sub/quality.md) |
| 14 | [Profiling & Tracing](#14-profiling--tracing) | [sub/profiling.md](sub/profiling.md) |
| 15 | [Blackwell Infrastructure](#15-blackwell-infrastructure) | [sub/cuda.md](sub/cuda.md) |
| 16 | [Safety Model](#16-safety-model) | |
| 17 | [Performance Contracts](#17-performance-contracts) | [sub/contracts.md](sub/contracts.md) |
| 18 | [Stack Integration](#18-stack-integration) | |
| 19 | [Development Commands](#19-development-commands) | |

---

## 1. Philosophy

Trueno exists because hand-written assembly is unsafe, unmaintainable, and non-portable. A single Rust source produces optimized code for x86, ARM, WASM, NVIDIA CUDA, and cross-platform GPU — with zero `unsafe` in the public API.

**Core invariants:**
- Write once, optimize everywhere via runtime dispatch
- Every optimization must prove ≥10% speedup via benchmarks
- >90% test coverage, mutation testing, property-based tests
- Contract-first: no kernel ships without a provable contract

---

## 2. Provable-Contract-First Design

**Every kernel implementation MUST begin with a YAML contract in `contracts/`.** The contract is the specification; the Rust code is the implementation. This is non-negotiable.

### The Contract-First Workflow

```
1. Write YAML contract       → contracts/my-kernel-v1.yaml
2. Define equations           → mathematical specification + pre/postconditions
3. Define FALSIFY tests       → how to disprove correctness
4. Define proof obligations   → formal properties (tolerance, equivalence)
5. Register binding           → ../provable-contracts/contracts/trueno/binding.yaml
6. Generate scaffold          → pv generate contracts/my-kernel-v1.yaml
7. Implement kernel           → fill in scaffold with real logic
8. Run FALSIFY tests + lint   → pv test + pv lint (4 gates)
9. Run Kani harnesses         → cargo kani (bounded model checking)
10. Merge only if all pass
```

### Escape-Proof Six-Stage Pipeline

It must be *impossible* to ship code that violates a contract. Six stages, each gates the next:

```
A. Equation (YAML)          → mathematical ground truth must exist
B. Lean 4 Proof             → theorem must have no sorry
C. YAML Validation          → pv lint Gates 1-4 must pass
D. build.rs Codegen         → sets CONTRACT_* env vars from binding.yaml
E. #[contract] Proc Macro   → checks env vars, inserts debug_assert pre/post
F. Test Execution           → cargo test runs FALSIFY tests
```

### Build-Time Enforcement (build.rs)

`build.rs` reads `../provable-contracts/contracts/trueno/binding.yaml`, sets `CONTRACT_{STEM}_{EQUATION}={status}` env vars, and enforces the **AllImplemented policy** — any `not_implemented` binding panics the build.

```
binding.yaml → parse → CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=implemented
                      → CONTRACT_GEMM_BACKWARD_TILED_V1_BACKWARD_A_GEMM=implemented
                      → if any not_implemented: panic!()
```

### Binding Stats

| Metric | Count |
|--------|-------|
| Total bindings | 38 |
| Implemented | 38 |
| Critical path equations | 8 (softmax, matmul, silu, gelu, execute_matmul, PipelineCache, GemmBackward, arithmetic_intensity) |

### Contract Inventory (28 local contracts)

| Domain | Contracts |
|--------|-----------|
| Core kernels | gemv, softmax, elementwise, transpose |
| BLAS | level3, trsm |
| FFT | stockham, bluestein, 2d, 3d |
| Image | conv2d, resize, canny, color, histogram |
| Sparse | formats, spmv, spmm, spgemm, bsr |
| Solvers | cholesky, lu, qr, svd |
| Random | philox, threefry |
| Quantization | tensor-contraction |
| GPU | dimension-independent-kernels |

### Verification Ladder

| Level | Method | Tool | Trueno status |
|-------|--------|------|--------------|
| L5 | Theorem proving | Lean 4 | 3 theorems (softmax) |
| L4 | Bounded model check | Kani | YAML-defined, not yet in CI |
| L3 | Property-based test | proptest | Active |
| L2 | Falsification test | `#[test]` | Active, all contracts |
| L1 | Type system + traits | rustc | Active, AllImplemented enforced |
| L0 | Code review + lint | pv lint, pmat comply | Active |

See [sub/contracts.md](sub/contracts.md) for binding.yaml schema, escape analysis, `#[contract]` macro, trait enforcement, `pv lint` gates, KAIZEN workflow, and contract schema reference.

---

## 3. Multi-Backend Architecture

```
Public API (safe) → Backend Dispatch → {SIMD, CUDA, wgpu, WASM, Scalar}
```

**Backend selection priority** (highest to lowest):
1. CUDA — NVIDIA GPU, workload benefits from parallelism
2. wgpu — cross-platform GPU, workload >100K elements
3. AVX-512 → AVX2 → AVX → SSE2 — x86_64, detected at runtime
4. NEON — ARM64
5. SIMD128 — WASM
6. Scalar — always available fallback

`Backend::Auto` resolves once at `Vector` creation via `is_x86_feature_detected!()`, not per-operation.

See [sub/backends.md](sub/backends.md) for dispatch logic and OpComplexity thresholds.

---

## 4. Backend Story Policy

**ZERO TOLERANCE: every operation MUST work on ALL backends.** No exceptions. If GPU acceleration is not beneficial, the GPU method falls back to CPU and documents why.

When adding a new operation:
1. Add to `VectorBackend` trait (`src/backends/mod.rs`)
2. Implement in all 7 backend modules: scalar, sse2, avx2, avx512, neon, wasm, gpu
3. Add WGSL shader if GPU-accelerable
4. Add sync + async device methods
5. Add integration test in `tests/backend_story.rs`
6. **Write contract FIRST** (`contracts/my-op-v1.yaml`)

Enforcement: pre-commit hook + `tests/backend_story.rs` + CI.

---

## 5. CPU SIMD Backends

| Backend | Width | Elements (f32) | Detection |
|---------|-------|-----------------|-----------|
| SSE2 | 128-bit | 4 | Baseline x86_64 |
| AVX | 256-bit | 8 | `is_x86_feature_detected!("avx")` |
| AVX2+FMA | 256-bit | 8 | Preferred for most ops |
| AVX-512 | 512-bit | 16 | Zen4/Sapphire Rapids+ |
| NEON | 128-bit | 4 | Baseline ARM64 |

**Critical patterns:**
- Always handle remainder: `len % lane_width` with scalar fallback
- Wrap intrinsics in `#[target_feature(enable = "...")]` functions
- Every `unsafe` block needs a `// SAFETY:` comment

See [sub/simd.md](sub/simd.md) for lane widths, FMA patterns, and horizontal reduction techniques.

---

## 6. CUDA Backend (trueno-gpu)

Pure Rust PTX generation — no nvcc, no LLVM, no external toolchains. The `trueno-gpu` crate generates PTX strings from Rust at compile-time or runtime.

**Available kernels:** GEMM (naive/tiled/tensor core), Softmax, LayerNorm, Attention (FlashAttention-style), Q4_K dequantization, 7 backward (training) kernels.

**Key APIs:** `PtxModule`, `PtxKernel`, `KernelBuilder`, `Kernel::emit_ptx()`.

**Testing without GPU:** All `build_ptx()` / `emit_ptx()` functions are pure string generators — test by checking `.version`, `.entry`, `.target` directives.

See [sub/cuda.md](sub/cuda.md) for PTX generation details, register allocation, Blackwell workarounds, and the dimension-independent kernel plan.

---

## 7. wgpu Backend

Cross-platform GPU compute via Vulkan/Metal/DX12/WebGPU. No CUDA required.

**Inference:** `WgslForwardPass` — RMSNorm, GEMV (cooperative K-reduction, vec4 loads), SiLU, RoPE. GEMV for M=1, tiled GEMM for M>1. 27.6 tok/s on Radeon Pro W5700X.

**Training:** 7 backward shaders in `src/backends/gpu/shaders/backward.rs` — silu_backward, gemm_backward_a/b, rmsnorm_backward, rope_backward, adamw_step, nf4_dequant. All FALSIFY tests pass. Enables full training loop on AMD/Intel/Apple.

**GPU threshold:** Only dispatch to GPU for >100K elements (PCIe transfer ~0.5ms).

See [sub/wgpu.md](sub/wgpu.md) for shader source locations, `GpuMatmulCache`, and provable contracts.

---

## 8. WASM Backend

Portable SIMD128 for browser/edge deployment. 4x f32 per lane (vs 8x AVX2). No GPU support in standard WASM — WebGPU is separate.

Build: `cargo build --target wasm32-unknown-unknown`

---

## 9. Layout Mandate (Q4K/Q6K)

**LAYOUT-002:** The Sovereign AI Stack uses **row-major exclusively** for APR/GGUF data. Column-major kernels exist for internal BLAS-style ops only.

Garbage inference output (`"olumbia+lsi nunca/localENTS"`) = wrong kernel layout. Aprender handles GGUF→APR transpose during import (`src/format/converter/write.rs`).

Pipeline: `GGUF (col-major) → aprender transpose → APR (row-major) → realizar → trueno row-major kernels`

See [sub/layout.md](sub/layout.md) for kernel selection guide and fused Q4K spec reference.

---

## 10. Crate Architecture

```
trueno/                  Main crate (CPU SIMD + wgpu)
├── src/backends/        scalar, sse2, avx2, avx512, neon, wasm, gpu/
├── src/vector.rs        Vector<T> + VectorOps trait
├── src/matrix.rs        matmul, transpose
└── src/error.rs         TruenoError

trueno-gpu/              CUDA sub-crate (pure Rust PTX)
├── src/ptx/             PTX builder, instructions, registers
├── src/kernels/         gemm, softmax, layernorm, attention, quantize
├── src/driver/          CUDA driver FFI
└── src/memory/          DeviceBuffer, HostBuffer, pool

crates/                  Domain sub-crates
├── trueno-fft           FFT (Stockham, Bluestein, 2D, 3D)
├── trueno-image         Image processing (conv2d, resize, canny)
├── trueno-quant         Quantization
├── trueno-rand          RNG (Philox, ThreeFry)
├── trueno-solve         Solvers (Cholesky, LU, QR, SVD)
├── trueno-sparse        Sparse (CSR, SELL, BSR, SpMV, SpGEMM)
└── trueno-tensor        Tensor contraction

contracts/               28 YAML provable contracts (source of truth)
```

---

## 11. Quality Gates

**Every commit:** clippy clean, all tests pass, ≥90% coverage, rustfmt, PMAT TDG ≥ B+.

**Every PR:** Tests for new code (all 5 categories), rustdoc, benchmarks prove ≥10% speedup, mutation testing ≥80% kill rate, contract FALSIFY tests pass.

**Every release:** CI green, repo-score ≥90/110, changelog updated, semver bump, git tag.

```bash
cargo clippy --all-features -- -D warnings
cargo test --all-features
make coverage                                    # ≥90% or commit blocked
pmat analyze tdg --min-grade B+
pmat repo-score . --min-score 90
cargo mutants --timeout 120 --minimum-pass-rate 80
```

See [sub/quality.md](sub/quality.md) for coverage enforcement, test categories, and mutation testing details.

---

## 12. Testing Requirements

Five mandatory test categories for every operation:

1. **Unit** — correctness, empty inputs, NaN/infinity/subnormal edge cases
2. **Property-based** (proptest) — commutativity, associativity, distributivity
3. **Backend equivalence** — all backends produce identical results (f32 tolerance < 1e-5)
4. **Mutation** — ≥80% kill rate (`cargo mutants`)
5. **Benchmark** — prove ≥10% speedup vs scalar baseline

---

## 13. Coverage

**≥90% line coverage is non-negotiable.** Enforced by pre-commit hook, `make coverage-check`, and CI.

- ONLY use `make coverage` — never `cargo llvm-cov` directly, never `cargo-tarpaulin`
- New code must have 100% coverage
- Pre-commit hook blocks commits below 90%
- HTML report: `target/coverage/html/index.html`

| Component | Minimum | Target |
|-----------|---------|--------|
| Public API | 100% | 100% |
| SIMD backends | 90% | 95% |
| GPU backend | 85% | 90% |
| Overall | **90%** | **95%+** |

---

## 14. Profiling & Tracing

Renacer v0.5.0+ for syscall tracing, function profiling, flamegraphs, and OTLP export.

```bash
make profile                     # benchmark profiling
make profile-flamegraph          # flamegraph
make profile-otlp-jaeger         # traces → Jaeger (localhost:16686)
```

**Golden trace validation** (`renacer.toml`): CI fails if syscall count or latency exceeds budget. Captures baseline traces for backend_detection, matrix_ops, activations, similarity.

See [sub/profiling.md](sub/profiling.md) for OTLP best practices and golden trace details.

---

## 15. Blackwell Infrastructure

**JIT Bug (trueno#200):** `cuModuleLoadDataEx` fails on sm_121 during active GPU work. Forward kernels work after pre-warming; backward kernels crash during training. Inference unaffected (cuBLAS/SIMD path).

**Fix (trueno#203):** Dimension-independent kernels (M,K,N as runtime params → ~15 types vs 50+ variants) + pre-compiled cubin pipeline: `build.rs → nvcc → include_bytes!() → zero JIT at runtime`.

Contract: `contracts/dimension-independent-kernels-v1.yaml` (6 FALSIFY tests).

---

## 16. Safety Model

- `unsafe` ONLY in backend implementations — never in public API
- Every `unsafe` block has a `// SAFETY:` comment explaining invariants
- SIMD intrinsics wrapped in `#[target_feature]` functions
- Public APIs are bounds-checked with `Result<T, TruenoError>`
- SIMD loops always handle remainder with scalar fallback

---

## 17. Performance Contracts

Every contract in `contracts/` tracks measured performance:

```yaml
performance:
  baseline: scalar
  measured:
    sse2: 2.1x
    avx2: 4.3x
    avx512: 8.1x
    cuda: 50x      # for 1M+ elements
    wgpu: 20x      # for 1M+ elements
  regression_threshold: 5%   # CI fails on >5% regression
```

Benchmark validation: ≥100 iterations, CV <5%, results saved to `target/criterion/`.

---

## 18. Stack Integration

Trueno is the compute foundation for the Sovereign AI Stack:
- **aprender** — tensor operations, format conversion
- **realizar** — fused inference kernels (uses trueno Q4K/Q6K)
- **entrenar** — training (blocked on Blackwell fix, trueno#200)
- **Depyler** — `np.dot()` → `trueno::Vector::dot()`
- **PMAT** — quality gates, pre-commit hooks, TDG grading

Stack-wide search: `batuta oracle --rag "your question"`

---

## 19. Development Commands

```bash
# Build
cargo build --all-features

# Test
cargo test --all-features

# Coverage (ONLY this command)
make coverage

# Lint
cargo clippy --all-features -- -D warnings && cargo fmt -- --check

# Bench
cargo bench --no-fail-fast

# Profile
make profile && make profile-flamegraph

# Quality
pmat analyze tdg --min-grade B+ && pmat repo-score . --min-score 90

# Code search (never grep for code discovery)
pmat query "simd kernel" --limit 10

# CUDA tests (requires GPU)
cargo test -p trueno-gpu --features cuda

# Stack search
batuta oracle --rag "your question"
```
