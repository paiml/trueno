# CUDA Library Parity

Trueno achieves feature parity with NVIDIA's CUDA library ecosystem through six dedicated crates, each backed by provable design-by-contract YAML specifications.

## Implementation Matrix

| CUDA Library | Trueno Crate | Operations | Tests | Status |
|---|---|---|---|---|
| **cuSPARSE** | `trueno-sparse` | CSR/COO/BSR/SELL, SpMV, SpMM, SpGEMM, SparseBackend trait (spmv+spmm) | 45 | Complete (CPU) |
| **cuFFT** | `trueno-fft` | Stockham 1D/2D/3D, R2C/C2R, Bluestein, Batched, Fft trait (+fft_2d) | 41 | Complete (CPU) |
| **cuSOLVER** | `trueno-solve` | LU, QR, SVD, Cholesky, TRSM, syrk/syr2k/trmm/symm, gemmEx, gemmStridedBatched, Solver trait, Epilogue fusion | 56 | Complete (CPU) |
| **NPP** | `trueno-image` | Conv2D, Gaussian, Sobel, Canny, canny_rgb, histogram, morphology, resize (4 modes), color, CC, ImageBuf, ImageOps trait (11 methods) | 69 | Complete (CPU) |
| **cuRAND** | `trueno-rand` | Philox 4×32-10, Threefry 4×64-20 (uniform, normal, stateless), Rng trait | 28 | Complete (CPU) |
| **cuTENSOR** | `trueno-tensor` | Einstein summation (TTGT), einsum_nary, matmul, outer, trace | 28 | Complete (CPU) |

## Provable Contracts

Each crate has YAML contracts in `contracts/` with formal proof obligations mapped to falsification tests:

| Contract | Crate | Key Invariants |
|---|---|---|
| `sparse-spmv-v1.yaml` | trueno-sparse | Backward error ≤ nnz·u·‖A‖·‖x‖ |
| `sparse-spmm-v1.yaml` | trueno-sparse | Dense equivalence, identity |
| `sparse-spgemm-v1.yaml` | trueno-sparse | Identity: AI = A, Associativity |
| `sparse-formats-v1.yaml` | trueno-sparse | SELL ↔ CSR equivalence |
| `fft-stockham-v1.yaml` | trueno-fft | Parseval, roundtrip, impulse |
| `fft-2d-v1.yaml` | trueno-fft | 2D impulse, Parseval |
| `fft-bluestein-v1.yaml` | trueno-fft | Stockham equivalence for 2^k |
| `fft-3d-v1.yaml` | trueno-fft | 3D impulse, roundtrip, Parseval |
| `solve-lu/qr/svd-v1.yaml` | trueno-solve | Backward error, residual bounds |
| `solve-cholesky-v1.yaml` | trueno-solve | SPD reconstruction, non-SPD rejection |
| `blas-trsm-v1.yaml` | trueno-solve | AX = B within backward error |
| `blas-level3-v1.yaml` | trueno-solve | syrk symmetry, trmm identity |
| `blas-gemmex-v1.yaml` | trueno-solve | f16 roundtrip, batch independence |
| `image-conv2d-v1.yaml` | trueno-image | Identity preservation, linearity |
| `image-canny-v1.yaml` | trueno-image | Binary output, constant → no edges |
| `image-resize-v1.yaml` | trueno-image | Constant preservation, identity resize |
| `image-color-v1.yaml` | trueno-image | HSV roundtrip, BT.601 weights |
| `rand-philox-v1.yaml` | trueno-rand | Determinism, distribution properties |
| `rand-threefry-v1.yaml` | trueno-rand | Determinism, no-multiply design |
| `tensor-contraction-v1.yaml` | trueno-tensor | matmul known values, trace identity |

## Running Examples

```bash
cargo run -p trueno-sparse --example sparse_spmv
cargo run -p trueno-fft    --example fft_demo
cargo run -p trueno-solve  --example solver_demo
cargo run -p trueno-image  --example image_demo
cargo run -p trueno-rand   --example rng_demo
cargo run -p trueno-tensor --example tensor_demo
```

## Unified Traits

Each crate exposes a trait for dynamic dispatch and pluggable backends:

| Trait | Crate | Implementors | Purpose |
|---|---|---|---|
| `Solver` | trueno-solve | LU, QR, Cholesky | Unified `solve(b)` for any factorization |
| `Rng` | trueno-rand | Philox4x32, Threefry4x64 | `fill_uniform()` / `fill_normal()` backend swap |
| `ImageOps` | trueno-image | ImageBuf | `blur()`, `canny_edges()`, `to_gray()` on structured buffers |
| `SparseBackend` | trueno-sparse | Scalar, AVX2, NEON | SIMD-pluggable SpMV kernels |
| `Fft` | trueno-fft | FftPlan | `fft_1d()`, `ifft_1d()`, `fft_r2c()`, `fft_c2r()` |

## Quality Gates

All code passes PMAT pre-commit quality gates:
- Cyclomatic complexity ≤ 30 per function
- Cognitive complexity ≤ 25 per function
- Zero SATD comments
- Zero `unwrap()`/`expect()` in library code
- clippy with `-D warnings`
