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
| 18 | [BLIS GEMM Engine](#18-blis-gemm-engine) | [sub/blis.md](sub/blis.md) |
| 19 | [ComputeBrick & Profiling](#19-computebrick--profiling) | [sub/brick.md](sub/brick.md) |
| 20 | [PTX Optimizer](#20-ptx-optimizer) | [sub/cuda.md](sub/cuda.md) |
| 21 | [Runtime Contracts](#21-runtime-contracts) | [sub/contracts.md](sub/contracts.md) |
| 22 | [Activation One Path Rule](#22-activation-one-path-rule) | |
| 23 | [Contract-Aware Tracing (Tier 3)](#23-contract-aware-tracing-tier-3) | [sub/deep-integration.md](sub/deep-integration.md) |
| 24 | [Stack Integration](#24-stack-integration) | |
| 25 | [Development Commands](#25-development-commands) | |

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
8. Run FALSIFY tests + lint   → pv test + pv lint (7 gates)
9. Run Kani harnesses         → cargo kani (bounded model checking)
10. Merge only if all pass
```

### Escape-Proof Six-Stage Pipeline

It must be *impossible* to ship code that violates a contract. Six stages, each gates the next:

```
A. Equation (YAML)          → mathematical ground truth must exist
B. Lean 4 Proof             → theorem must have no sorry
C. YAML Validation          → pv lint Gates 1-7 must pass
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
| L5 | Theorem proving | Lean 4 | 30 theorems, 18 domains, 0 sorry |
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

**Default backend selection** (`Backend::Auto`, resolved once at `Vector` creation via OnceLock):
1. **AVX2+FMA** — preferred x86_64 (safer than AVX-512 for memory-bound ops)
2. **AVX** — fallback x86_64
3. **SSE2** — baseline x86_64
4. **NEON** — ARM64
5. **SIMD128** — WASM
6. **Scalar** — always available

**AVX-512** is NOT auto-selected — only used for ComputeBound operations via `select_backend_for_operation()`. GPU backends (CUDA, wgpu) are dispatched separately based on workload size and OpComplexity.

See [sub/backends.md](sub/backends.md) for dispatch logic and OpComplexity thresholds.

---

## 4. Backend Story Policy

**ZERO TOLERANCE: every operation MUST work on ALL backends.** No exceptions. If GPU acceleration is not beneficial, the GPU method falls back to CPU and documents why.

When adding a new operation:
1. **Write contract FIRST** (`contracts/my-op-v1.yaml`) — equations, FALSIFY tests, proof obligations
2. Register binding in `../provable-contracts/contracts/trueno/binding.yaml`
3. Add to `VectorBackend` trait (`src/backends/mod.rs`)
4. Implement in all backend modules: `scalar/`, `sse2/`, `avx2/`, `avx512/`, `neon/`, `wasm/`, `gpu/`, `q4k/`, `q6k/`
5. Add WGSL shader if GPU-accelerable
6. Add sync + async device methods
7. Add integration test in `tests/backend_story.rs`

Enforcement: `tests/backend_story.rs` + CI.

---

## 5. CPU SIMD Backends

| Backend | Width | Elements (f32) | Detection |
|---------|-------|-----------------|-----------|
| SSE2 | 128-bit | 4 | Baseline x86_64 |
| AVX | 256-bit | 8 | `is_x86_feature_detected!("avx")` |
| AVX2+FMA | 256-bit | 8 | Preferred for most ops |
| AVX-512 | 512-bit | 16 | ComputeBound ops only (`avx512f` feature flag) |
| NEON | 128-bit | 4 | Baseline ARM64 |

**Critical patterns:**
- Always handle remainder: `len % lane_width` with scalar fallback
- Wrap intrinsics in `#[target_feature(enable = "...")]` functions
- Every `unsafe` block needs a `// SAFETY:` comment

See [sub/simd.md](sub/simd.md) for lane widths, FMA patterns, and horizontal reduction techniques.

---

## 6. CUDA Backend (trueno-gpu)

Pure Rust PTX generation — no nvcc, no LLVM, no external toolchains. The `trueno-gpu` crate generates PTX strings from Rust at compile-time or runtime.

**Available kernels:** GEMM (naive/tiled/tensor core), Softmax, LayerNorm, Attention (FlashAttention-style), Q4_K dequantization, 6 backward kernels (activations, cross_entropy, gemm, layer_norm, rms_norm, softmax).

**Key APIs:** `PtxModule`, `PtxKernel`, `KernelBuilder`, `Kernel::emit_ptx()`.

**Testing without GPU:** All `build_ptx()` / `emit_ptx()` functions are pure string generators — test by checking `.version`, `.entry`, `.target` directives.

See [sub/cuda.md](sub/cuda.md) for PTX generation details, register allocation, Blackwell workarounds, and the dimension-independent kernel plan.

---

## 7. wgpu Backend

Cross-platform GPU compute via Vulkan/Metal/DX12/WebGPU. No CUDA required.

**Inference:** `WgslForwardPass` — RMSNorm, GEMV (cooperative K-reduction, vec4 loads), SiLU, RoPE. GEMV for M=1, tiled GEMM for M>1. 27.6 tok/s on Radeon Pro W5700X.

**Training:** 9 shaders in `src/backends/gpu/shaders/backward.rs` — 6 backward (silu, gemm_a, gemm_b, rmsnorm, rope, cross_entropy), plus adamw_step optimizer, nf4_dequant, and cross_entropy_forward. All FALSIFY tests pass. Enables full training loop on AMD/Intel/Apple.

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
├── src/backends/        scalar/, sse2/, avx2/, avx512/, neon/, wasm/, gpu/, q4k/, q6k/
├── src/vector/          Vector<T> + VectorOps trait
├── src/matrix/          matmul, transpose
├── src/blis/            BLIS micro-kernel delegation
├── src/brick/           ComputeBrick, BrickProfiler, quant_ops
├── src/eigen/           Eigendecomposition
├── src/monitor/         GPU monitoring, ComputeDevice trait
├── src/tiling/          Cache-aware tiling
├── src/tuner/           ML-based backend tuner
└── src/error.rs         TruenoError

trueno-gpu/              CUDA sub-crate (pure Rust PTX)
├── src/ptx/             PTX builder, instructions, registers, optimizer
├── src/kernels/         gemm, softmax, layernorm, attention, quantize, backward, lz4
├── src/driver/          CUDA driver FFI
└── src/memory/          DeviceBuffer, HostBuffer, pool

crates/                  Domain sub-crates
├── cbtop                Compute Block Top TUI + adaptive ML
├── trueno-fft           FFT (Stockham, Bluestein, 2D, 3D)
├── trueno-image         Image processing (conv2d, resize, canny)
├── trueno-quant         Quantization (Q4K, Q5K, Q6K, NF4)
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

**≥90% line coverage is non-negotiable.** Enforced by `make coverage-check` and CI.

- ONLY use `make coverage` — never `cargo llvm-cov` directly, never `cargo-tarpaulin`
- New code must have 100% coverage
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
  measured_ratio: 1.53           # vs scalar baseline
  measured_throughput: "16.3 Gelem/s"
  regression_threshold: 5%      # CI fails on >5% regression
```

Benchmark validation: ≥100 iterations, CV <5%, results saved to `target/criterion/`.

---

## 18. BLIS GEMM Engine

`src/blis/` implements BLIS-style blocked GEMM with cache hierarchy optimization (L3→L2→L1→registers). Micro-kernels: `8x6` AVX2, `8x8` NEON.

**Block sizes:** MC=72, KC=256, NC=4096, MR=8, NR=6. Packing: `pack_a()`, `pack_b()`, `PrepackedB` for weight caching.

**Toyota Production System integration:**
- **Jidoka** — `JidokaGuard` stops on numerical error (NaN, divergence >1e-3 from reference)
- **Heijunka** — `HeijunkaScheduler` for load-balanced parallel GEMM
- **Kaizen** — `BlisProfiler` tracks per-level (L3/L2/L1/micro) timing

Backend selection via `BackendCostModel` with roofline analysis. `gemm_profiled()` returns profiling stats alongside results.

See [sub/blis.md](sub/blis.md) for micro-kernel patterns, packing layout, and cost model.

---

## 19. ComputeBrick & Profiling

`src/brick/` provides token-centric compute units — self-verifying blocks with budgets, assertions, and backends.

**Key types:**
- `ComputeBrick` — Composable compute unit with pre/postconditions
- `BrickProfiler` — O(1) hot-path profiling via `BrickId` enum (PAR-200)
- `ExecutionGraph` — Full execution path tracking with kernel checksums
- `ModelTracer` — Model-level inference tracing with tensor stats, attention weights, logit evolution

**Quantization ops:** `BlockQ5K`, `BlockQ6K`, `DotQ5KOp`, `DotQ6KOp` (llama.cpp compatible). Fused ops: `FusedQKVOp`, `FusedGateUpOp` for transformer inference.

**Integration:** `BrickTuner::get_tuner_recommendations()` in `src/tuner/` uses profiler data for kernel selection. SyncMode (Eager/Deferred) controls GPU synchronization granularity.

See [sub/brick.md](sub/brick.md) for the full brick taxonomy, profiling protocol, and tracing API.

---

## 20. PTX Optimizer

`trueno-gpu/src/ptx/optimize/` implements multi-pass PTX optimization:

| Pass | What it does | Reference |
|------|-------------|-----------|
| FMA fusion | `mul` + `add` → `fma` pattern matching | Click & Paleczny 1995 |
| Tile validation | Validate tile constraints, prevent register spill | Volkov & Demmel 2008 |
| Loop splitting | Split loops at conditional boundaries | NVIDIA CUDA Tile IR |
| TKO (Token ordering) | Memory dependency tracking, barrier elimination | NVIDIA Tile IR model |
| Barrier safety | Detect early-exit-before-barrier bugs (PARITY-114) | Five Whys 2026 |

Applied via `optimize()` in sequence. All passes are pure functions on PTX AST — no GPU required for testing.

---

## 21. Runtime Contracts

`src/contracts.rs` enforces kernel-level preconditions/postconditions at runtime. `src/generated_contracts.rs` is auto-generated from YAML via `pv codegen`.

**Three-layer contract hierarchy:**
1. **aprender** (import) — `enforce_architecture_completeness()`: validate tensor names
2. **realizar** (load) — `contract_gate::validate_model_load()`: validate architecture
3. **trueno** (kernel) — `contracts::validate_weight_buffer()`: validate bytes & layout

`STACK_LAYOUT = RowMajor` — the ONLY layout trueno kernels accept. Hard-errors on violation (no silent defaults).

Generated contracts use `debug_assert!()` — zero cost in release builds. Covers: activation kernels (gelu, relu, silu), matmul pre/postconditions, position encoding, active learning.

---

## 22. Activation One Path Rule

`src/activations.rs` defines canonical scalar activation functions per UCBD §4 (One Path Rule):

`silu_scalar()`, `gelu_scalar()`, `sigmoid_scalar()`, `relu_scalar()`, `tanh_scalar()`, plus `f16_to_f32()`/`f32_to_f16()` conversions.

**Downstream crates (aprender, realizar, entrenar, whisper-apr) MUST import from here** — re-implementing is a contract violation. SIMD-vectorized versions exist in `backends/*/ops/activations` but delegate to these canonical implementations for correctness.

---

## 23. Contract-Aware Tracing (Tier 3)

Tiers 1 (compile-time) and 2 (CI) enforce contracts statically. Tier 3 enforces at runtime via tracing integration — closing the gap between "contract says X" and "system does X in production."

### Architecture

```
Contract YAML → ContractRegistry (startup)
                    ↓
BrickProfiler ──→ budget check ──→ violation event
ModelTracer   ──→ postcondition check ──→ violation event
                    ↓
ContractTracingLayer (tracing::Layer)
                    ↓
Structured diagnostics (SARIF-compatible)
```

### Gap Closures

| Gap | Problem | Fix |
|-----|---------|-----|
| Gap 2 | ComputeBrick budget is hardcoded | `ComputeBrick::from_contract()` derives `TokenBudget` from roofline YAML |
| Gap 3 | ModelTracer observes but doesn't verify | `end_forward()` checks MLT-01..05 against contract invariants |

### ContractTracingLayer

A `tracing::Layer` that intercepts spans tagged with `contract.id` and verifies postconditions on span close. Violations emit structured `tracing::error!` events with contract ID, obligation, and measured value.

**Performance budget:** ≤130ns per check (NCCLbpf demonstrates this is achievable for GPU data paths). BrickProfiler's deferred sync mode batches checks with existing finalization — zero per-kernel overhead on hot path.

### ModelTracer Contract Hooks

`end_forward()` verifies existing trace data against contract invariants:
- **MLT-01**: no NaN/Inf in activations (activation-kernel-v1 postcondition)
- **MLT-02**: attention weights sum to 1 per row (softmax-kernel-v1 postcondition)
- **MLT-03**: logit magnitudes within bounds (model-config-algebra-v1)
- **MLT-04**: quantization error ≤ contract threshold (quantization-ordering-v1)
- **MLT-05**: KV cache utilization ≤ capacity (gpu-decode-profiling-v1)

Zero additional collection overhead — reuses existing trace data.

### Rust-Native Path (Future)

When Rust MCP-759 contracts stabilize (`#[contracts::requires]`/`#[contracts::ensures]`), YAML postcondition checks can migrate to compiler-inserted assertions with zero cost in release builds via `-Z contract-checks=off`.

**References:** ProofWright (arXiv:2511.12294), Volta (arXiv:2511.12638), NCCLbpf (arXiv:2603.11438), Rust MCP-759

See [sub/deep-integration.md](sub/deep-integration.md) for full design, code examples, and enforcement pipeline.

---

## 24. Stack Integration

Trueno is the compute foundation for the Sovereign AI Stack:
- **aprender** — tensor operations, format conversion
- **realizar** — fused inference kernels (uses trueno Q4K/Q6K)
- **entrenar** — training (blocked on Blackwell fix, trueno#200)
- **Depyler** — `np.dot()` → `trueno::Vector::dot()`
- **PMAT** — quality gates, pre-commit hooks, TDG grading

Stack-wide search: `batuta oracle --rag "your question"`

---

## 25. Development Commands

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
