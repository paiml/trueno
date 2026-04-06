# APR-MONO: Sovereign Stack Monorepo Consolidation

**Version**: 1.0
**Date**: 2026-04-06
**Status**: PROPOSAL — Ready for Review
**Priority**: P0 — Unblocks daily apr-cli releases
**Author**: PAIML Team + Claude

---

## Executive Summary

Merge **5 repositories** (trueno, aprender, entrenar, realizar, batuta) into
a **single `paiml/aprender` monorepo** with ~30 workspace crates under the
`aprender-*` namespace. This eliminates the cross-repo version sync problem
that has caused **19 broken crates.io publishes** (paiml/aprender#701) and
enables daily `apr-cli` releases from a single `cargo publish -p apr-cli`.

### Precedent

Every successful large Rust project uses this pattern:

| Project | Crates | Repo | Pattern |
|---------|--------|------|---------|
| Polars | 28 | 1 (`pola-rs/polars`) | `polars-{core,lazy,io,...}` |
| Burn (ML) | 33 | 1 (`tracel-ai/burn`) | `burn-{tensor,train,wgpu,...}` |
| Nushell | 30+ | 1 (`nushell/nushell`) | `nu-{cli,command,engine,...}` |
| DataFusion | 15 | 1 (`apache/datafusion`) | `datafusion-{common,expr,...}` |
| TiKV | 20+ | 1 (`tikv/tikv`) | `tikv-{client,server,...}` |
| **PAIML (current)** | **32+** | **5** | **4 namespaces, 19 broken publishes** |

---

## Current State (5 repos, 4752 .rs files, 32+ published crates)

### Repository Inventory

| Repo | Files | Version | Published Crates | Role |
|------|-------|---------|-----------------|------|
| trueno | 478 | 0.18.0 | 18 (`trueno-*`) | Compute: SIMD, GPU, WASM |
| aprender | 1179 | 0.27.8 | 4 (`aprender-*`) | ML format, tokenizers, model ops |
| entrenar | 1052 | 0.7.13 | 7 (`entrenar-*`) | Training loops |
| realizar | 1499 | 0.8.6 | 1 | Inference server |
| batuta | 544 | 0.7.3 | 2 | Orchestration, agents, RAG oracle |
| **Total** | **4752** | — | **32** | — |

### Satellite Crates (separate repos, stack-dependent)

These crates live in their own repos but depend on the core stack:

| Repo | Version | Files | Role | Disposition |
|------|---------|-------|------|-------------|
| presentar | 0.3.5 | 1 | TUI framework (workspace) | **MERGE** — core UI for cbtop, batuta |
| renacer | 0.10.2 | 119 | Profiling/tracing | **MERGE** — used by all 5 core crates |
| certeza | 0.1.1 | 9 | Quality validation | **MERGE** — tiny, used in CI |
| trueno-db | 0.3.17 | 27 | Embedded analytics DB | **MERGE** — already trueno-namespaced |
| trueno-graph | 0.1.18 | 23 | Graph database | **MERGE** — already trueno-namespaced |
| trueno-rag | 0.2.5 | 42 | RAG pipeline | **MERGE** — already trueno-namespaced |
| trueno-viz | 0.2.4 | 114 | Visualization | **MERGE** — already trueno-namespaced |
| trueno-zram | 0.3.1 | 3 | Compressed RAM (workspace) | **MERGE** — already trueno-namespaced |
| batuta-common | 0.1.0 | 6 | Shared batuta types | **MERGE** — folded into aprender-orchestrate |
| repartir | 2.0.4 | 23 | Distributed computing | **MERGE** — used by batuta |
| manzana | 0.1.0 | 10 | Apple hardware interfaces | KEEP SEPARATE — platform-specific |
| whisper.apr | 0.2.8 | 197 | Whisper speech model | KEEP SEPARATE — application, not framework |
| alimentar | 0.2.9 | 83 | Data loading/synthetic data | **MERGE** — core data pipeline |
| simular | 0.3.2 | 93 | Simulation framework | **MERGE** — used by training |
| verificar | 0.5.0 | 52 | Verification/testing | **MERGE** — used by CI/quality |
| probar | 1.0.3 | 1 (workspace: 4 crates) | WASM/browser test framework | **MERGE** — depends on trueno+presentar |

**Updated totals with satellites:**
- **Merge into monorepo**: 5 core + 13 satellites = 18 repos
- **Keep separate**: manzana, whisper.apr (+ pmat, which is its own product)
- **Total .rs files**: ~5500+
- **Total workspace crates**: ~45

### Dependency Graph (Current)

```
apr-cli ──→ aprender ──→ trueno 0.17, trueno-quant
       ──→ entrenar ──→ trueno 0.17, aprender 0.27, realizar(opt)
       ──→ realizar ──→ trueno 0.17, trueno-gpu, aprender(opt)
       ──→ batuta(?) ──→ trueno 0.16, aprender, entrenar, realizar
       ──→ trueno 0.17, trueno-explain, trueno-viz
```

**Problems:**
1. **Version skew**: trueno is 0.18.0 but all consumers pin 0.17 → diamond deps
2. **[patch.crates-io]** hacks required during development → leak to publishes
3. **Publishing order matters**: trueno → aprender → entrenar → realizar → apr-cli (5 sequential publishes, any can break)
4. **Circular deps**: aprender→trueno, but trueno's inference needs aprender's tokenizer
5. **19 broken publishes** documented in paiml/aprender#701

---

## Proposed Structure

```
paiml/aprender/                          # THE monorepo
├── Cargo.toml                           # workspace root
│   [workspace]
│   members = ["crates/*"]
│   [workspace.package]
│   version = "0.29.0"                   # ALL crates share one version
│
├── crates/
│   │
│   │ ── User-facing ──
│   ├── apr-cli/                         # Binary: `apr` command (DAILY releases)
│   │
│   │ ── Core ML ──
│   ├── aprender/                        # ML format (.apr), tokenizers, model ops
│   ├── aprender-train/                  # Was: entrenar (training loops)
│   ├── aprender-serve/                  # Was: realizar (inference server)
│   ├── aprender-orchestrate/            # Was: batuta (agents, RAG oracle, playbooks)
│   │
│   │ ── Compute primitives ──
│   ├── aprender-compute/                # Was: trueno (SIMD/GPU/WASM core)
│   ├── aprender-gpu/                    # Was: trueno-gpu (CUDA PTX, no nvcc)
│   ├── aprender-quant/                  # Was: trueno-quant
│   ├── aprender-gemm-codegen/           # Was: trueno-gemm-codegen
│   ├── aprender-inference/              # Was: trueno src/inference/ (GGUF, LlamaModel)
│   │
│   │ ── Data & Storage ──
│   ├── aprender-db/                     # Was: trueno-db
│   ├── aprender-rag/                    # Was: trueno-rag
│   ├── aprender-graph/                  # Was: trueno-graph
│   │
│   │ ── Visualization & Tooling ──
│   ├── aprender-viz/                    # Was: trueno-viz
│   ├── aprender-explain/                # Was: trueno-explain
│   ├── aprender-profile/               # Was: renacer (profiling/tracing)
│   ├── aprender-present/               # Was: presentar (TUI framework)
│   ├── aprender-shell/                  # REPL (already in aprender)
│   ├── aprender-verify/                # Was: certeza (quality validation)
│   │
│   │ ── Training sub-crates ──
│   ├── aprender-train-common/           # Was: entrenar-common
│   ├── aprender-train-lora/             # Was: entrenar-lora
│   │
│   │ ── Data & Simulation ──
│   ├── aprender-data/                   # Was: alimentar (data loading, synthetic data)
│   ├── aprender-simulate/              # Was: simular (simulation framework)
│   ├── aprender-distribute/            # Was: repartir (distributed computing)
│   │
│   │ ── Edge / Specialized ──
│   ├── aprender-cuda-edge/              # Was: trueno-cuda-edge
│   ├── aprender-zram/                   # Was: trueno-zram-core + trueno-zram-adaptive
│   ├── aprender-fft/                    # Was: trueno-fft
│   ├── aprender-sparse/                 # Was: trueno-sparse
│   ├── aprender-solve/                  # Was: trueno-solve
│   ├── aprender-rand/                   # Was: trueno-rand
│   ├── aprender-image/                  # Was: trueno-image
│   ├── aprender-tensor/                 # Was: trueno-tensor
│   │
│   │ ── Benchmarks & Testing ──
│   ├── aprender-bench-tokenizer/        # Already in aprender
│   ├── aprender-bench-compute/          # Already in aprender
│   └── aprender-tsp/                    # Already in aprender
│
├── contracts/                           # ALL provable contracts (merged)
├── book/                                # Unified mdbook documentation
├── cookbook/                             # apr-cookbook (merged in)
└── docs/specifications/                 # Specs (merged)
```

### Crate Count: ~42 workspace members

Comparable to Polars (28), Burn (33), Nushell (30+). Slightly larger
but includes infrastructure that Polars gets from external deps
(we own the full stack: DB, graph, profiler, TUI, distributed compute).

---

## Backward Compatibility

### Re-export shim crates

Old crate names continue to work via thin re-export crates:

```rust
// trueno/src/lib.rs (published as trueno 0.19.0)
//! trueno is now aprender-compute. This crate re-exports for backward compatibility.
pub use aprender_compute::*;
```

Same for `entrenar`, `realizar`, `batuta`, and all `trueno-*` sub-crates.
These shim crates are ~5 lines each, maintained indefinitely, never change.

### For existing users

| Current dependency | Migration |
|-------------------|-----------|
| `trueno = "0.18"` | Works forever (shim re-exports aprender-compute) |
| `aprender = "0.27"` | `aprender = "0.29"` (same crate, new version) |
| `entrenar = "0.7"` | `aprender-train = "0.29"` or keep `entrenar = "0.8"` (shim) |
| `realizar = "0.8"` | `aprender-serve = "0.29"` or keep `realizar = "0.9"` (shim) |
| `batuta = "0.7"` | `aprender-orchestrate = "0.29"` or keep `batuta = "0.8"` (shim) |

---

## Migration Plan

### Phase 1: Prepare (1 day)

1. Create `paiml/aprender` branch `monorepo-consolidation`
2. Add `crates/` directories for new workspace members
3. Set up `[workspace.package] version = "0.29.0"`
4. Set up `[workspace.dependencies]` for all shared deps (like Polars does)

### Phase 2: Move source (2 days)

For each repo (trueno, entrenar, realizar, batuta):

```bash
# Preserve git history with subtree merge
git subtree add --prefix=crates/aprender-compute \
  git@github.com:paiml/trueno.git main
```

Then:
- Rename `[package] name` in each moved Cargo.toml
- Update internal `use trueno::` → `use aprender_compute::`
- Update internal path deps to workspace-relative

### Phase 3: Wire workspace deps (1 day)

Replace all version-pinned cross-crate deps with workspace paths:

```toml
# Before (in entrenar/Cargo.toml):
trueno = { version = "0.17", features = ["parallel"] }
aprender = { version = "0.27" }

# After (in crates/aprender-train/Cargo.toml):
aprender-compute = { path = "../aprender-compute", features = ["parallel"] }
aprender = { path = "../aprender" }
```

### Phase 4: Publish & Shim (1 day)

1. Publish all `aprender-*` crates from the monorepo
2. Publish shim crates for old names (`trueno 0.19.0`, `entrenar 0.8.0`, etc.)
3. Verify `cargo install apr-cli` works from crates.io
4. Archive old repositories (read-only, link to monorepo)

### Phase 5: Daily workflow (ongoing)

```bash
# Daily apr-cli release (ONE command):
cargo publish -p apr-cli

# If a compute primitive changed too:
cargo publish -p aprender-compute && cargo publish -p apr-cli

# Workspace-wide test (catches ALL breakage):
cargo test --workspace
```

---

## What This Fixes

| Problem | Before (5 repos) | After (1 repo) |
|---------|------------------|----------------|
| Version sync | Manual, 19 failures (#701) | Automatic (workspace) |
| Daily apr-cli | 5-repo coordination | `cargo publish -p apr-cli` |
| Diamond deps | `trueno 0.17` vs `0.18` | Impossible (one version) |
| `[patch.crates-io]` | Required, leaks to publish | Eliminated |
| Circular deps | aprender↔trueno blocked | Workspace siblings |
| CI coverage | 5 separate pipelines | 1 pipeline, 1 report |
| New contributor setup | Clone 5 repos | Clone 1 repo |
| Cross-crate refactoring | 5 PRs, coordinated merge | 1 PR |

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Large git repo (4752 files) | Certain | Low | Polars has 28 crates, Burn 33 — proven at scale |
| Compile time increase | Medium | Medium | `default-members` limits what builds by default; `cargo test -p apr-cli` for focused work |
| CI time increase | Medium | Medium | Use `cargo nextest` + `--partition` for parallel CI; cache `target/` |
| Migration breaks existing users | Low | High | Shim crates provide indefinite backward compat |
| Git history loss | Low | Medium | `git subtree` preserves full history; old repos archived read-only |
| Merge conflicts during migration | Medium | Low | Do it over a weekend freeze; migrate one repo at a time |

---

## Decision Matrix

| Option | Impact | Effort | Risk | Recommendation |
|--------|--------|--------|------|---------------|
| **A: Full monorepo** (this spec, 17 repos → 1) | **Critical** | **5-7 days** | **Low** | **RECOMMENDED — matches industry standard** |
| B: Keep trueno separate | Medium | 2 days | Medium | Partial fix, version sync remains |
| C: Do nothing | — | 0 | **High** | 19 incidents → 30+ incidents |

---

## Success Criteria

1. `cargo test --workspace` passes (all 5500+ files compile together)
2. `cargo publish -p apr-cli` succeeds without `[patch.crates-io]`
3. `cargo install apr-cli` works from a clean machine
4. Old crate names (`trueno`, `entrenar`, `realizar`, `batuta`) still resolve via shims
5. Daily apr-cli releases take < 5 minutes (publish + verify)
6. Zero cross-crate version mismatch incidents for 90 days post-migration

---

## Appendix A: Crate Rename Mapping

| Old Name | New Name | Published Shim? |
|----------|----------|----------------|
| trueno | aprender-compute | Yes (trueno 0.19) |
| trueno-gpu | aprender-gpu | Yes |
| trueno-quant | aprender-quant | Yes |
| trueno-db | aprender-db | Yes |
| trueno-viz | aprender-viz | Yes |
| trueno-explain | aprender-explain | Yes |
| trueno-rag | aprender-rag | Yes |
| trueno-graph | aprender-graph | Yes |
| trueno-gemm-codegen | aprender-gemm-codegen | Yes |
| trueno-zram-core | aprender-zram | Yes |
| trueno-zram-adaptive | aprender-zram-adaptive | Yes |
| trueno-cuda-edge | aprender-cuda-edge | Yes |
| trueno-fft | aprender-fft | Yes |
| trueno-sparse | aprender-sparse | Yes |
| trueno-solve | aprender-solve | Yes |
| trueno-rand | aprender-rand | Yes |
| trueno-image | aprender-image | Yes |
| trueno-tensor | aprender-tensor | Yes |
| trueno-ptx-debug | aprender-ptx-debug | No (internal) |
| entrenar | aprender-train | Yes (entrenar 0.8) |
| entrenar-common | aprender-train-common | Yes |
| entrenar-lora | aprender-train-lora | Yes |
| realizar | aprender-serve | Yes (realizar 0.9) |
| batuta | aprender-orchestrate | Yes (batuta 0.8) |
| apr-cli | apr-cli | No rename needed |
| aprender | aprender | No rename needed |
| presentar | aprender-present | Yes (presentar 0.4) |
| renacer | aprender-profile | Yes (renacer 0.11) |
| certeza | aprender-verify | Yes (certeza 0.2) |
| batuta-common | (folded into aprender-orchestrate) | Yes |
| repartir | aprender-distribute | Yes (repartir 2.1) |
| alimentar | aprender-data | Yes (alimentar 0.3) |
| simular | aprender-simulate | Yes (simular 0.4) |
| verificar | aprender-verify-ml | Yes (verificar 0.6) |
| probar | aprender-test | Yes (probar 1.1) |
| probar-derive | aprender-test-derive | Yes |
| probar-cli | aprender-test-cli | Yes |
| probar-js-gen | aprender-test-js-gen | Yes |

### Appendix B: Kept Separate (NOT merged)

| Crate | Reason |
|-------|--------|
| pmat / paiml-mcp-agent-toolkit | Separate product, own release cycle, 3830 .rs files |
| manzana | Platform-specific (Apple only) |
| whisper.apr | Application built ON the stack, not part of it |
| ruchy | Separate language/runtime project |
| apr-cookbook | Becomes `aprender/cookbook/` (content, not a crate) |

### Appendix C: Polars Reference Architecture

```
pola-rs/polars/Cargo.toml:
  [workspace.package]
  version = "0.53.0"           ← ALL 28 crates share this version
  
  [workspace.dependencies]     ← shared dep versions, DRY
  arrow = "53"
  serde = { version = "1", features = ["derive"] }
  
polars-core/Cargo.toml:
  [package]
  name = "polars-core"
  version.workspace = true     ← inherits from workspace
  
  [dependencies]
  polars-arrow = { workspace = true }   ← resolves to local path
```

This is the target state for `paiml/aprender`.
