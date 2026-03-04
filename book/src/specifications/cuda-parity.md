# CUDA Library Parity

Trueno achieves feature parity with NVIDIA's CUDA library ecosystem through six dedicated crates, each backed by provable design-by-contract YAML specifications.

## Implementation Matrix

| CUDA Library | Trueno Crate | Operations | Tests | Status |
|---|---|---|---|---|
| **cuSPARSE** | `trueno-sparse` | CSR/COO/BSR/SELL, SpMV, SpMM, SpGEMM | 41 | Complete (CPU) |
| **cuFFT** | `trueno-fft` | Stockham 1D/2D/3D, R2C, Bluestein, Batched | 30 | Complete (CPU) |
| **cuSOLVER** | `trueno-solve` | LU, QR, SVD, Cholesky, TRSM, syrk/syr2k/trmm/symm | 39 | Complete (CPU) |
| **NPP** | `trueno-image` | Conv2D, Gaussian, Sobel, Canny, histogram, morphology, resize, color, CC | 40 | Complete (CPU) |
| **cuRAND** | `trueno-rand` | Philox 4×32-10 (uniform, normal, stateless) | 13 | Complete (CPU) |
| **cuTENSOR** | `trueno-tensor` | Einstein summation (TTGT), matmul, outer, trace | 22 | Complete (CPU) |

## Provable Contracts

Each crate has YAML contracts in `contracts/` with formal proof obligations mapped to falsification tests:

| Contract | Crate | Key Invariants |
|---|---|---|
| `sparse-spmv-v1.yaml` | trueno-sparse | Backward error ≤ nnz·u·‖A‖·‖x‖ |
| `sparse-spgemm-v1.yaml` | trueno-sparse | Identity: AI = A, Associativity |
| `fft-stockham-v1.yaml` | trueno-fft | Parseval, roundtrip, impulse |
| `fft-bluestein-v1.yaml` | trueno-fft | Stockham equivalence for 2^k |
| `fft-3d-v1.yaml` | trueno-fft | 3D impulse, roundtrip, Parseval |
| `solve-lu/qr/svd/cholesky-v1.yaml` | trueno-solve | Backward error, residual bounds |
| `blas-trsm-v1.yaml` | trueno-solve | AX = B within backward error |
| `blas-level3-v1.yaml` | trueno-solve | syrk symmetry, trmm identity |
| `image-conv2d-v1.yaml` | trueno-image | Identity preservation, linearity |
| `image-color-v1.yaml` | trueno-image | HSV roundtrip, BT.601 weights |
| `rand-philox-v1.yaml` | trueno-rand | Determinism, distribution properties |
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

## Quality Gates

All code passes PMAT pre-commit quality gates:
- Cyclomatic complexity ≤ 30 per function
- Cognitive complexity ≤ 25 per function
- Zero SATD comments
- Zero `unwrap()`/`expect()` in library code
- clippy with `-D warnings`
