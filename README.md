# Trueno ⚡

> **Multi-Target High-Performance Compute Library**

[![CI](https://github.com/paiml/trueno/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/trueno/actions)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/paiml/trueno)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/trueno.svg)](https://crates.io/crates/trueno)

<div align="center">
  <img src="docs/images/trueno-vs-numpy-pytorch.png" alt="Trueno vs NumPy/PyTorch Performance Comparison" width="100%">
</div>

## 🤥 The Lambda Lie: "Just Wrap Python in C++"

<div align="center">
  <img src="docs/images/python-c-lambda-lie.png" alt="The Lambda Lie: Wrapping Python in C++ makes serverless WORSE" width="100%">
</div>

**The Truth**: Each "optimization" layer makes Lambda deployment **WORSE**. Python wrapped in C++ gives you fast local compute but **HUGE** binaries (10-50MB) and **MASSIVE** cold starts (28-86ms). Trueno gives you **BOTH**: 1.6x faster locally (10.8µs dot product) **AND** 12.8x faster on Lambda (6.70ms cold start, 396KB binary). No wrapping needed.

_See [TRUENO_LAMBDA_ADVANTAGE.md](docs/TRUENO_LAMBDA_ADVANTAGE.md) for the complete story._

---

**Trueno** (Spanish: "thunder") provides unified, high-performance compute primitives across three execution targets:

1. **CPU SIMD** - x86 (SSE2/AVX/AVX2/AVX-512), ARM (NEON), WASM (SIMD128)
2. **GPU** - Vulkan/Metal/DX12/WebGPU via `wgpu`
3. **WebAssembly** - Portable SIMD128 for browser/edge deployment

## Quick Start

```rust
use trueno::{Vector, Matrix};

// Vector operations
let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);

// Auto-selects best backend (AVX2/GPU/WASM)
let result = a.add(&b).unwrap();
assert_eq!(result.as_slice(), &[6.0, 8.0, 10.0, 12.0]);

let dot_product = a.dot(&b).unwrap();  // 70.0
let sum = a.sum().unwrap();            // 10.0
let max = a.max().unwrap();            // 4.0

// Matrix operations
let m1 = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
let m2 = Matrix::identity(2);
let product = m1.matmul(&m2).unwrap();  // Matrix multiplication
let transposed = m1.transpose();        // Matrix transpose

// 2D Convolution (image processing, CNNs)
let image = Matrix::from_vec(5, 5, vec![/* 25 pixels */]).unwrap();
let kernel = Matrix::from_vec(3, 3, vec![1.0/9.0; 9]).unwrap();  // 3x3 averaging filter
let filtered = image.convolve2d(&kernel).unwrap();  // Auto-selects GPU for large images
```

## Performance vs NumPy/PyTorch

**Dot Product (1K elements):**
- Trueno AVX-512: **11.9x vs scalar** | **1.6x vs NumPy** | **2.8x vs PyTorch**

**Matrix Multiply (500×500):**
- Trueno GPU: **2-10x faster** than scalar

**Replicate:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # Install UV (one-time)
make bench-comprehensive                          # 12-17 minutes
```

_See [benchmarks/README.md](benchmarks/README.md) for methodology. Feedback welcome via [issues](https://github.com/paiml/trueno/issues)._

## Performance

⚠️ **CRITICAL: GPU Performance Validation (2025-11-17)**

Recent comprehensive benchmarking ([docs/performance-analysis.md](docs/performance-analysis.md)) revealed:

- ✅ **GPU beneficial for**: Matrix multiplication ONLY (2-10x speedup for 500×500+ matrices)
- ❌ **GPU detrimental for**: ALL element-wise operations (2-65,000x SLOWER than scalar)
- 🎯 **Recommendation**: GPU backend disabled for vector operations in v0.2.1+ (matmul only)

**Root cause**: 14-55ms GPU overhead (buffer allocation + PCIe transfer) dominates element-wise operations which complete in 0.01-10ms on CPU. GPU speedup requires O(N³) compute complexity to amortize transfer costs.

See [performance analysis](docs/performance-analysis.md) for complete empirical data.

---

Trueno delivers **exceptional performance** through multi-level SIMD optimization:

### SSE2 (128-bit SIMD) vs Scalar

| Operation | Speedup | Use Case |
|-----------|---------|----------|
| **Dot Product** | **340%** faster | Machine learning, signal processing |
| **Sum Reduction** | **315%** faster | Statistics, aggregations |
| **Max Finding** | **348%** faster | Data analysis, optimization |
| Element-wise Add | 3-10% faster | Memory-bound (limited SIMD benefit) |
| Element-wise Mul | 5-6% faster | Memory-bound (limited SIMD benefit) |

### AVX2 (256-bit SIMD) vs SSE2

| Operation | Speedup | Notes |
|-----------|---------|-------|
| **Dot Product** | **182%** faster | FMA (fused multiply-add) acceleration |
| Element-wise Add | 15% faster | Memory bandwidth limited |
| Element-wise Mul | 12% faster | Memory bandwidth limited |

**Key Insights**:
- SIMD excels at compute-intensive operations (dot product, reductions)
- Element-wise operations are memory-bound, limiting SIMD gains
- AVX2's FMA provides significant acceleration for dot products

### Matrix Operations (SIMD-Optimized)

| Operation | Size | Time | Performance |
|-----------|------|------|-------------|
| **Matrix Multiply** | 64×64 | 59.9 µs | SIMD threshold |
| **Matrix Multiply** | 128×128 | 434.9 µs | ~7x faster than naive |
| **Matrix Multiply** | 256×256 | 2.67 ms | Scales O(n³) |
| **Matrix Transpose** | 256×256 | 69.1 µs | Cache-optimized |
| **Matrix-Vector** | 512×512 | 139.8 µs | SIMD dot products |

**SIMD Optimization Strategy**:
- **Threshold**: 64×64 (auto-selects SIMD vs naive)
- **Transpose**: Pre-transpose B for cache locality (row-major access)
- **Dot Products**: Uses Vector::dot() for SIMD acceleration (2-8x speedup)
- **Small Matrices**: Uses naive O(n³) to avoid SIMD overhead

### 2D Convolution (GPU-Accelerated)

| Operation | Input Size | Kernel | Time | Backend |
|-----------|------------|--------|------|---------|
| **Convolution** | 32×32 | 3×3 | ~6.78 µs | Scalar |
| **Convolution** | 128×128 | 3×3 | ~1.2 ms | Scalar |
| **Convolution** | 256×256 | 3×3 | ~4.8 ms | Scalar/GPU threshold |
| **Convolution** | 512×512 | 3×3 | ~20 ms (scalar) | **GPU** (10-50x target) |
| **Sobel Edge Detection** | 512×512 | 3×3 | - | GPU-accelerated |

**GPU Acceleration Strategy** (OpComplexity::High):
- **GPU Threshold**: >10,000 output elements (e.g., 100×100 output)
- **Example**: 512×512 input with 3×3 kernel → 510×510 output = 260,100 elements
- **Workgroups**: 16×16 threads (256 threads per workgroup)
- **Valid Padding**: Output size = (input - kernel + 1) for each dimension
- **Use Cases**: Image processing, CNN inference, feature extraction

**Automatic Backend Selection**:
- Small images (<10K elements): Scalar baseline (~6.78 µs for 32×32)
- Large images (>10K elements): GPU compute shader (10-50x speedup target)
- Graceful fallback to scalar if GPU unavailable

**Example: Edge Detection with Sobel Operator**
```rust
use trueno::backends::gpu::GpuBackend;

// Sobel X kernel (vertical edge detection)
let sobel_x = vec![
    -1.0, 0.0, 1.0,
    -2.0, 0.0, 2.0,
    -1.0, 0.0, 1.0,
];

// 512×512 grayscale image (flattened row-major)
let image: Vec<f32> = vec![...]; // 262,144 elements

// GPU convolution for large images (>10K output elements)
let mut gpu = GpuBackend::new();
let edges = gpu.convolve2d(&image, &sobel_x, 512, 512, 3, 3).unwrap();
// Output: 510×510 = 260,100 elements (GPU-accelerated)
```

### ReLU Activation (GPU-Accelerated)

| Operation | Vector Size | Time (Scalar) | Time (GPU Target) | Speedup |
|-----------|------------|---------------|------------------|---------|
| **ReLU** | 10K | ~40 µs | - | Below threshold |
| **ReLU** | 100K | ~400 µs | ~40 µs | 10x target |
| **ReLU** | 1M | ~4 ms | ~80 µs | 50x target |

**GPU Acceleration Strategy** (OpComplexity::Low):
- **GPU Threshold**: >100,000 elements
- **Operation**: Simple element-wise max(0, x)
- **Workgroups**: 256 threads per workgroup (1D dispatch)
- **Use Cases**: Neural network inference, batch activation processing

**Automatic Backend Selection**:
- Small vectors (<100K elements): Scalar/SIMD (iterator-based)
- Large vectors (>100K elements): GPU compute shader (10-50x speedup target)
- Graceful fallback to scalar if GPU unavailable

**Example: Neural Network Inference**
```rust
use trueno::Vector;

// Process large activation batch (e.g., ResNet-50 layer)
let activations = Vector::from_slice(&vec![...]);  // 1M neurons
let output = activations.relu().unwrap();  // Auto-uses GPU for >100K elements
```

### Leaky ReLU Activation (GPU-Accelerated)

| Operation | Vector Size | Time (Scalar) | Time (GPU Target) | Speedup |
|-----------|------------|---------------|------------------|---------|
| **Leaky ReLU** | 10K | ~42 µs | - | Below threshold |
| **Leaky ReLU** | 100K | ~420 µs | ~42 µs | 10x target |
| **Leaky ReLU** | 1M | ~4.2 ms | ~85 µs | 50x target |

**GPU Acceleration Strategy** (OpComplexity::Low):
- **GPU Threshold**: >100,000 elements
- **Operation**: Element-wise leaky_relu(x, α) = x if x > 0, else αx
- **Workgroups**: 256 threads per workgroup (1D dispatch)
- **Parameters**: Runtime negative_slope (α) via uniform buffer
- **Use Cases**: GANs, deep networks (prevents "dying ReLU" problem)

**Automatic Backend Selection**:
- Small vectors (<100K elements): Scalar/SIMD (iterator-based)
- Large vectors (>100K elements): GPU compute shader (10-50x speedup target)
- Graceful fallback to scalar if GPU unavailable

**Example: GAN Generator Network**
```rust
use trueno::Vector;

// Leaky ReLU for GAN generator (prevents vanishing gradients)
let hidden = Vector::from_slice(&vec![...]);  // 512K hidden units
let activated = hidden.leaky_relu(0.01).unwrap();  // Auto-uses GPU for >100K elements
```

### ELU Activation (GPU-Accelerated)

| Operation | Vector Size | Time (Scalar) | Time (GPU Target) | Speedup |
|-----------|------------|---------------|------------------|---------|
| **ELU** | 10K | ~55 µs | - | Below threshold |
| **ELU** | 100K | ~550 µs | ~55 µs | 10x target |
| **ELU** | 1M | ~5.5 ms | ~110 µs | 50x target |

**GPU Acceleration Strategy** (OpComplexity::Low):
- **GPU Threshold**: >100,000 elements
- **Operation**: Element-wise elu(x, α) = x if x > 0, else α(e^x - 1)
- **Workgroups**: 256 threads per workgroup (1D dispatch)
- **Parameters**: Runtime alpha (α) via uniform buffer
- **Use Cases**: Deep networks, smooth gradients, improved learning dynamics

**Automatic Backend Selection**:
- Small vectors (<100K elements): Scalar/SIMD (iterator-based)
- Large vectors (>100K elements): GPU compute shader (10-50x speedup target)
- Graceful fallback to scalar if GPU unavailable

**Example: Deep Residual Network**
```rust
use trueno::Vector;

// ELU for deep ResNet (smooth gradients prevent vanishing/exploding)
let residual = Vector::from_slice(&vec![...]);  // 256K hidden units
let activated = residual.elu(1.0).unwrap();  // Auto-uses GPU for >100K elements
```

### Clip (Clamp) Operation (GPU-Accelerated)

| Operation | Vector Size | Time (Scalar) | Time (GPU Target) | Speedup |
|-----------|------------|---------------|------------------|---------|
| **Clip** | 10K | ~45 µs | - | Below threshold |
| **Clip** | 100K | ~450 µs | ~45 µs | 10x target |
| **Clip** | 1M | ~4.5 ms | ~90 µs | 50x target |

**GPU Acceleration Strategy** (OpComplexity::Low):
- **GPU Threshold**: >100,000 elements
- **Operation**: Element-wise clamp(x, min_val, max_val)
- **Workgroups**: 256 threads per workgroup (1D dispatch)
- **Parameters**: Runtime min/max bounds via uniform buffer
- **Use Cases**: Gradient clipping, value bounding, range normalization

**Automatic Backend Selection**:
- Small vectors (<100K elements): Scalar/SIMD (iterator-based)
- Large vectors (>100K elements): GPU compute shader (10-50x speedup target)
- Graceful fallback to scalar if GPU unavailable

**Example: Gradient Clipping**
```rust
use trueno::Vector;

// Clip gradients for stable training
let gradients = Vector::from_slice(&vec![...]);  // 500K parameters
let clipped = gradients.clip(-1.0, 1.0).unwrap();  // Auto-uses GPU for >100K elements
```

### Sigmoid Activation (GPU-Accelerated)

| Operation | Vector Size | Time (Scalar) | Time (GPU Target) | Speedup |
|-----------|------------|---------------|------------------|---------|
| **Sigmoid** | 10K | ~60 µs | - | Below threshold |
| **Sigmoid** | 100K | ~600 µs | ~60 µs | 10x target |
| **Sigmoid** | 1M | ~6 ms | ~120 µs | 50x target |

**GPU Acceleration Strategy** (OpComplexity::Low):
- **GPU Threshold**: >100,000 elements
- **Operation**: Element-wise σ(x) = 1 / (1 + e^(-x))
- **Workgroups**: 256 threads per workgroup (1D dispatch)
- **Numerical Stability**: Separate handling for positive/negative inputs
- **Use Cases**: Binary classification, attention mechanisms, gating functions

**Automatic Backend Selection**:
- Small vectors (<100K elements): Scalar (iterator-based with stability checks)
- Large vectors (>100K elements): GPU compute shader (10-50x speedup target)
- Graceful fallback to scalar if GPU unavailable

**Example: Neural Network Layer**
```rust
use trueno::Vector;

// Sigmoid activation for binary classification
let logits = Vector::from_slice(&vec![...]);  // 500K neurons
let activations = logits.sigmoid().unwrap();  // Auto-uses GPU for >100K elements
```

### Tanh Activation (GPU-Accelerated)

| Operation | Vector Size | Time (Scalar) | Time (GPU Target) | Speedup |
|-----------|------------|---------------|------------------|---------|
| **Tanh** | 10K | ~55 µs | - | Below threshold |
| **Tanh** | 100K | ~550 µs | ~55 µs | 10x target |
| **Tanh** | 1M | ~5.5 ms | ~110 µs | 50x target |

**GPU Acceleration Strategy** (OpComplexity::Low):
- **GPU Threshold**: >100,000 elements
- **Operation**: Element-wise tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
- **Workgroups**: 256 threads per workgroup (1D dispatch)
- **Numerical Stability**: Saturation handling for |x| > 20
- **Use Cases**: LSTM, GRU, recurrent neural networks, traditional activation

**Automatic Backend Selection**:
- Small vectors (<100K elements): Scalar (standard library tanh)
- Large vectors (>100K elements): GPU compute shader (10-50x speedup target)
- Graceful fallback to scalar if GPU unavailable

**Example: LSTM Cell**
```rust
use trueno::Vector;

// Tanh activation in LSTM forget/input gates
let cell_state = Vector::from_slice(&vec![...]);  // 250K hidden units
let activated = cell_state.tanh().unwrap();  // Auto-uses GPU for >100K elements
```

### Swish Activation (GPU-Accelerated)

| Operation | Vector Size | Time (Scalar) | Time (GPU Target) | Speedup |
|-----------|------------|---------------|------------------|---------|
| **Swish** | 10K | ~70 µs | - | Below threshold |
| **Swish** | 100K | ~700 µs | ~70 µs | 10x target |
| **Swish** | 1M | ~7 ms | ~140 µs | 50x target |

**GPU Acceleration Strategy** (OpComplexity::Low):
- **GPU Threshold**: >100,000 elements
- **Operation**: Element-wise swish(x) = x * σ(x) = x / (1 + e^(-x))
- **Workgroups**: 256 threads per workgroup (1D dispatch)
- **Numerical Stability**: Separate handling for positive/negative inputs
- **Use Cases**: Transformers (BERT, GPT, T5), modern neural networks, SiLU activation
- **Also known as**: SiLU (Sigmoid Linear Unit)

**Automatic Backend Selection**:
- Small vectors (<100K elements): Scalar (iterator-based with stability checks)
- Large vectors (>100K elements): GPU compute shader (10-50x speedup target)
- Graceful fallback to scalar if GPU unavailable

**Example: Transformer Inference**
```rust
use trueno::Vector;

// Swish activation in transformer feed-forward network
let ffn_output = Vector::from_slice(&vec![...]);  // 768K hidden units (BERT-large)
let activated = ffn_output.swish().unwrap();  // Auto-uses GPU for >100K elements
```

### GELU Activation (GPU-Accelerated)

| Operation | Vector Size | Time (Scalar) | Time (GPU Target) | Speedup |
|-----------|------------|---------------|------------------|---------|
| **GELU** | 10K | ~80 µs | - | Below threshold |
| **GELU** | 100K | ~800 µs | ~80 µs | 10x target |
| **GELU** | 1M | ~8 ms | ~160 µs | 50x target |

**GPU Acceleration Strategy** (OpComplexity::Low):
- **GPU Threshold**: >100,000 elements
- **Operation**: Element-wise GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
- **Workgroups**: 256 threads per workgroup (1D dispatch)
- **Approximation**: Tanh-based (standard in production)
- **Use Cases**: BERT, GPT-2, GPT-3, Vision Transformers, modern NLP models
- **THE activation**: Standard in transformer architectures since 2018

**Automatic Backend Selection**:
- Small vectors (<100K elements): Scalar (iterator-based tanh approximation)
- Large vectors (>100K elements): GPU compute shader (10-50x speedup target)
- Graceful fallback to scalar if GPU unavailable

**Example: BERT Inference**
```rust
use trueno::Vector;

// GELU activation in BERT transformer layer
let ffn_hidden = Vector::from_slice(&vec![...]);  // 3.07M elements (BERT-base: 768 * 4 * 1024 batch)
let activated = ffn_hidden.gelu().unwrap();  // Auto-uses GPU for >100K elements
```

**📖 See [Performance Guide](docs/PERFORMANCE_GUIDE.md) and [AVX2 Benchmarks](docs/AVX2_BENCHMARKS.md) for detailed analysis.**

### Softmax Activation (GPU-Accelerated)

| Operation | Vector Size | Time (Scalar) | Time (GPU Target) | Speedup |
|-----------|------------|---------------|------------------|---------|
| **Softmax** | 10K | ~120 µs | ~60 µs | 2x target |
| **Softmax** | 100K | ~1.2 ms | ~120 µs | 10x target |
| **Softmax** | 1M | ~12 ms | ~600 µs | 20x target |

**GPU Acceleration Strategy** (OpComplexity::Medium):
- **GPU Threshold**: >10,000 elements (multi-pass overhead higher than element-wise ops)
- **Operation**: Multi-pass softmax(x)[i] = exp(x[i] - max) / sum(exp(x - max))
- **Implementation**: 4-pass GPU reduction
  - **Pass 1**: Max reduction (parallel, numerical stability)
  - **Pass 2**: Exp-subtract (element-wise exp(x - max))
  - **Pass 3**: Sum reduction (parallel sum of exp values)
  - **Pass 4**: Normalize (element-wise division by sum)
- **Workgroups**: 256 threads per workgroup (1D dispatch)
- **Numerical Stability**: Subtracts max before exp to prevent overflow
- **Use Cases**: Classification networks, attention mechanisms, transformers

**Automatic Backend Selection**:
- Small vectors (<10K elements): Scalar (multi-pass CPU implementation)
- Large vectors (>10K elements): GPU compute shader (5-20x speedup target)
- Graceful fallback to scalar if GPU unavailable

**Example: Attention Mechanism**
```rust
use trueno::Vector;

// Softmax in multi-head attention (transformer)
let attention_scores = Vector::from_slice(&vec![...]);  // 512K scores (64 heads * 128 seq * 64 seq)
let attention_weights = attention_scores.softmax().unwrap();  // Auto-uses GPU for >10K elements
```

### Log-Softmax Activation (GPU-Accelerated)

| Operation | Vector Size | Time (Scalar) | Time (GPU Target) | Speedup |
|-----------|------------|---------------|------------------|---------|
| **Log-Softmax** | 10K | ~130 µs | ~65 µs | 2x target |
| **Log-Softmax** | 100K | ~1.3 ms | ~130 µs | 10x target |
| **Log-Softmax** | 1M | ~13 ms | ~650 µs | 20x target |

**GPU Acceleration Strategy** (OpComplexity::Medium):
- **GPU Threshold**: >10,000 elements (multi-pass overhead)
- **Operation**: Multi-pass log_softmax(x)[i] = x[i] - max - log(sum(exp(x - max)))
- **Implementation**: 4-pass GPU reduction (same as softmax but final step computes log)
  - **Pass 1**: Max reduction (parallel, numerical stability)
  - **Pass 2**: Exp-subtract (element-wise exp(x - max))
  - **Pass 3**: Sum reduction (parallel sum of exp values)
  - **Pass 4**: Log-normalize (element-wise x - max - log(sum))
- **Workgroups**: 256 threads per workgroup (1D dispatch)
- **Numerical Stability**: More stable than computing log(softmax(x))
- **Use Cases**: Cross-entropy loss, NLL loss, classification training

**Automatic Backend Selection**:
- Small vectors (<10K elements): Scalar (multi-pass CPU implementation)
- Large vectors (>10K elements): GPU compute shader (5-20x speedup target)
- Graceful fallback to scalar if GPU unavailable

**Example: Cross-Entropy Loss**
```rust
use trueno::Vector;

// Log-softmax for stable cross-entropy loss computation
let logits = Vector::from_slice(&vec![...]);  // 100K logits (1000 batch * 100 classes)
let log_probs = logits.log_softmax().unwrap();  // Auto-uses GPU for >10K elements

// Compute NLL loss: -log_probs[target_class]
// More numerically stable than log(softmax(x))
```

**📖 See [Performance Guide](docs/PERFORMANCE_GUIDE.md) and [AVX2 Benchmarks](docs/AVX2_BENCHMARKS.md) for detailed analysis.**

## Features

- **🚀 Write Once, Optimize Everywhere**: Single algorithm, multiple backends
- **⚡ Runtime Dispatch**: Auto-select best implementation based on CPU features
- **🎮 GPU Acceleration**: ⚠️ **Matmul ONLY** (2-10x for 500×500+) - GPU disabled for all vector operations after empirical benchmarking showed 2-65,000x slowdown (see [performance-analysis.md](docs/performance-analysis.md))
- **🛡️ Zero Unsafe in Public API**: Safety via type system, `unsafe` isolated in backends
- **📊 Benchmarked Performance**: Every optimization proves ≥10% speedup
- **🧪 Extreme TDD**: >90% test coverage, mutation testing, property-based tests
- **🎯 Production Ready**: PMAT quality gates, Toyota Way principles

**🎯 Strategic Positioning**: Trueno is designed as a **drop-in replacement** for NumPy (~35% complete) and PyTorch (~15% complete) in Rust applications. See [PyTorch/NumPy Replacement Specification](docs/specifications/pytorch-numpy-replacement-spec.md) for detailed roadmap to full parity.

## Design Principles

### Write Once, Optimize Everywhere
```rust
// Same code runs optimally on x86, ARM, WASM, GPU
let result = a.add(&b).unwrap();
```

Trueno automatically selects the best backend:
- **x86_64**: AVX-512 → AVX2 → AVX → SSE2 → Scalar
- **ARM**: NEON → Scalar
- **WASM**: SIMD128 → Scalar
- **GPU** (optional): Vulkan/Metal/DX12/WebGPU (>1000×1000 matrices)

### Safety First
```rust
// Public API is 100% safe Rust
let result = vector.add(&other)?;  // Returns Result<Vector, TruenoError>

// Size mismatches caught at runtime
let a = Vector::from_slice(&[1.0, 2.0]);
let b = Vector::from_slice(&[1.0, 2.0, 3.0]);
assert!(a.add(&b).is_err());  // SizeMismatch error
```

### Performance Targets

| Operation | Size | Target Speedup vs Scalar | Backend |
|-----------|------|-------------------------|---------|
| `add()` | 1K | 8x | AVX2 |
| `add()` | 100K | 16x | GPU |
| `dot()` | 10K | 12x | AVX2 + FMA |
| `sum()` | 1M | 20x | GPU |

*All optimizations benchmarked with Criterion.rs, minimum 10% improvement required*

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
trueno = "0.1"
```

### GPU Acceleration (Optional)

Enable GPU support for very large matrices:

```toml
[dependencies]
trueno = { version = "0.1", features = ["gpu"] }
```

**Requirements**:
- Vulkan, Metal, or DirectX 12 compatible GPU
- wgpu runtime dependencies
- GPU backend automatically activates for matrices >1000×1000

For bleeding-edge features:

```toml
[dependencies]
trueno = { git = "https://github.com/paiml/trueno", features = ["gpu"] }
```

## Usage Examples

### Basic Vector Operations

```rust
use trueno::Vector;

// Element-wise addition
let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
let sum = a.add(&b).unwrap();
assert_eq!(sum.as_slice(), &[5.0, 7.0, 9.0]);

// Element-wise multiplication
let product = a.mul(&b).unwrap();
assert_eq!(product.as_slice(), &[4.0, 10.0, 18.0]);

// Dot product
let dot = a.dot(&b).unwrap();
assert_eq!(dot, 32.0);  // 1*4 + 2*5 + 3*6

// Reductions
let total = a.sum().unwrap();  // 6.0
let maximum = a.max().unwrap();  // 3.0
```

### Backend Selection

```rust
use trueno::{Vector, Backend};

// Auto-select best backend (recommended)
let v = Vector::from_slice(&data);  // Uses Backend::Auto

// Explicit backend (for testing/benchmarking)
let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
let v = Vector::from_slice_with_backend(&data, Backend::GPU);
```

### 2D Convolution (Image Processing)

```rust
use trueno::Matrix;

// Create a 5×5 input image
let image = Matrix::from_vec(
    5, 5,
    vec![
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 9.0, 0.0, 0.0,  // Center pixel
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ]
).unwrap();

// 3×3 averaging filter (blur)
let kernel_val = 1.0 / 9.0;
let kernel = Matrix::from_vec(3, 3, vec![kernel_val; 9]).unwrap();

// Apply convolution (valid padding)
let filtered = image.convolve2d(&kernel).unwrap();

// Output is 3×3 (5 - 3 + 1 = 3)
assert_eq!(filtered.rows(), 3);
assert_eq!(filtered.cols(), 3);
assert!((filtered.get(1, 1).unwrap() - 1.0).abs() < 1e-5);  // Center smoothed

// Sobel edge detection (horizontal edges)
let sobel_h = Matrix::from_vec(
    3, 3,
    vec![
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0,  1.0,
    ]
).unwrap();

let edges = image.convolve2d(&sobel_h).unwrap();

// GPU acceleration for large images
let large_image = Matrix::zeros(512, 512);  // 512×512 image
let result = large_image.convolve2d(&kernel).unwrap();  // Auto-uses GPU (>10K elements)
```

### Error Handling

```rust
use trueno::{Vector, TruenoError};

let a = Vector::from_slice(&[1.0, 2.0]);
let b = Vector::from_slice(&[1.0, 2.0, 3.0]);

match a.add(&b) {
    Ok(result) => println!("Sum: {:?}", result.as_slice()),
    Err(TruenoError::SizeMismatch { expected, actual }) => {
        eprintln!("Size mismatch: expected {}, got {}", expected, actual);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Ecosystem Integration

Trueno integrates with the Pragmatic AI Labs transpiler ecosystem:

### Ruchy
```ruby
# Ruchy syntax
let v = Vector([1.0, 2.0]) + Vector([3.0, 4.0])
# Transpiles to: trueno::Vector::add()
```

### Depyler (Python → Rust)
```python
# Python/NumPy code
import numpy as np
result = np.dot(a, b)
# Transpiles to: trueno::Vector::dot(&a, &b)
```

### Decy (C → Rust)
```c
// C SIMD intrinsics
__m256 result = _mm256_add_ps(a, b);
// Transpiles to: trueno::Vector::add() (safe!)
```

## Development

### Prerequisites

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install development tools
make install-tools
```

### Building

```bash
# Development build
make build

# Release build (optimized)
make build-release

# Run tests
make test

# Fast test run (<5 min target)
make test-fast
```

### Quality Gates

Trueno enforces **EXTREME TDD** quality standards:

```bash
# Run all quality gates (pre-commit)
make quality-gates

# Individual gates
make lint       # Zero warnings policy
make fmt-check  # Format verification
make test-fast  # All tests (<5 min)
make coverage   # >85% required (<10 min)
make mutate     # Mutation testing (>80% kill rate)
```

**Quality Metrics**:
- ✅ **Test Coverage**: 100% (target >85%)
- ✅ **PMAT TDG Score**: 96.1/100 (A+)
- ✅ **Clippy Warnings**: 0
- ✅ **Property Tests**: 10 tests × 100 cases each
- ✅ **Cyclomatic Complexity**: Median 1.0 (very low)

### PMAT Integration

```bash
# Technical Debt Grading
make pmat-tdg

# Complexity analysis
make pmat-analyze

# Repository health score
make pmat-score
```

### Profiling & Performance Analysis

Trueno integrates **Renacer** for deep performance profiling:

```bash
# Profile benchmarks to find bottlenecks
make profile

# Generate flamegraph visualization
make profile-flamegraph

# Profile specific benchmark
make profile-bench BENCH=vector_ops

# Profile test suite
make profile-test
```

**Profiling Use Cases**:
- 🔬 **SIMD Validation**: Verify optimizations show expected speedups (2-8x)
- 🎯 **Hot Path Analysis**: Identify top 10 functions consuming most time
- 💾 **Memory Bottlenecks**: Detect cache misses and memory access patterns
- 🚀 **Backend Selection**: Validate runtime dispatch overhead is minimal
- 📊 **Flamegraph Visualization**: Visual analysis of performance characteristics

**Example Output**:
```
🔬 Profiling benchmark: vector_ops
I/O Bottleneck: memcpy() - 15.2ms (42% of runtime)
Hot Functions:
  1. _mm256_add_ps - 3.4ms (9.4%)
  2. Vector::dot - 2.1ms (5.8%)
  3. backend_dispatch - 0.3ms (0.8%)
```

### Testing Philosophy

Trueno uses **multi-layered testing**:

1. **Unit Tests** (30 tests): Basic functionality, edge cases, error paths
2. **Property Tests** (10 tests × 100 cases): Mathematical properties verification
   - Commutativity: `a + b == b + a`
   - Associativity: `(a + b) + c == a + (b + c)`
   - Identity elements: `a + 0 == a`, `a * 1 == a`
   - Distributive: `a * (b + c) == a*b + a*c`
3. **Integration Tests**: Backend selection, large datasets
4. **Benchmarks**: Performance regression prevention (Criterion.rs)
5. **Mutation Tests**: Test suite effectiveness (>80% kill rate)

Run property tests with verbose output:
```bash
cargo test property_tests -- --nocapture
```

## Benchmarking

### 🏆 Comprehensive Comparison vs NumPy/PyTorch

**Makefile Targets** (Recommended):

```bash
# Complete comparison suite (Rust + NumPy + PyTorch + Analysis)
make bench-comprehensive    # ~12-17 minutes, interactive confirmation

# Individual components
make bench                  # Rust benchmarks only (~10-15 min)
make bench-python          # Python benchmarks only (~2 min)
make bench-compare-frameworks  # Generate comparison report
```

**Alternative** (Direct script execution):

```bash
./benchmarks/run_all.sh    # Same as make bench-comprehensive
```

**What's Included**:
1. ✅ Trueno benchmarks (Rust/Criterion) - 10-15 minutes
2. ✅ Python benchmarks (NumPy/PyTorch) - 2-3 minutes
3. ✅ Comparative analysis and report generation - <1 minute

**Results**:
- `benchmarks/comparison_report.md` - Performance comparison report
- `benchmarks/comparison_summary.json` - Machine-readable data
- `target/criterion/` - Detailed Criterion benchmark data

**Success Criteria (v0.3.0)**: Trueno within 20% of NumPy for ≥80% of 1D operations

**📊 Performance Results** (v0.3.0 - Comprehensive Benchmarks):

✅ **Trueno dramatically outperforms NumPy and PyTorch:**
- **88.5% faster than NumPy** (54/61 comparisons)
- **90.2% faster than PyTorch** (55/61 comparisons)

**Extreme speedups on reductions** (small vectors):
- `sum`: 310.97x faster than NumPy (100 elements)
- `max`: 356.30x faster than NumPy (100 elements)
- `dot`: 123.97x faster than NumPy (100 elements)
- `norm_l2`: 178.33x faster than NumPy (100 elements)

**Consistent wins on element-wise operations**:
- `add`: 1.44-12.53x faster than NumPy
- `mul`: 1.44-17.94x faster than NumPy
- `sub`: 1.57-12.78x faster than NumPy
- `div`: 1.54-10.27x faster than NumPy

**Optimizations needed** (slower than NumPy):
- `tanh` at large sizes (5.59x slower at 100K elements)
- `relu` at 1M elements (8.32x slower - investigation needed)

**Architecture**: AVX-512 dominates reductions, AVX2 optimal for element-wise ops

See full report: [`benchmarks/comparison_report.md`](benchmarks/comparison_report.md)

See [`benchmarks/README.md`](benchmarks/README.md) for detailed documentation.

### Rust-Only Benchmarks

```bash
# Run all Rust benchmarks
make bench

# Benchmark specific operation
cargo bench -- add
cargo bench -- dot

# GPU benchmarks (if available)
make bench-gpu

# Save baseline for regression detection
make bench-save-baseline
make bench-compare
```

**Benchmark results** stored in `target/criterion/`:
- Throughput (elements/second)
- Latency (mean, median, p95, p99)
- Backend comparison (Scalar vs SSE2 vs AVX2 vs AVX-512)
- Statistical analysis (outliers, confidence intervals)

**Operations Benchmarked** (25 total):
- Element-wise: `add`, `sub`, `mul`, `div`, `scale`, `abs`, `clamp`, `lerp`, `fma`
- Reductions: `dot`, `sum`, `max`, `min`, `argmax`, `argmin`, `norm_l1`, `norm_l2`, `norm_linf`
- Activations: `relu`, `sigmoid`, `tanh`, `gelu`, `swish`, `exp`, `softmax`, `log_softmax`

### Dependencies

**Rust**: Included in `dev-dependencies` (Criterion)

**Python** (for comprehensive comparison):
```bash
# Install UV (Rust-based Python package manager)
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
bash /tmp/uv-install.sh
rm -f /tmp/uv-install.sh

# Dependencies installed automatically by make targets
```

## Examples

Trueno includes several runnable examples demonstrating real-world use cases:

```bash
# Machine Learning: Cosine similarity, L2 normalization, k-NN
cargo run --release --example ml_similarity

# Performance: Compare Scalar vs SSE2 backends
cargo run --release --example performance_demo

# Backend Detection: Runtime CPU feature detection
cargo run --release --example backend_detection
```

**ML Example Features**:
- Document similarity for recommendation systems
- Feature normalization for neural networks
- k-Nearest Neighbors classification
- Demonstrates 340% speedup for dot products

See `examples/` directory for complete code.

## Project Structure

```
trueno/
├── src/
│   ├── lib.rs          # Public API, backend enum, auto-selection
│   ├── error.rs        # Error types (TruenoError)
│   ├── vector.rs       # Vector<T> implementation
│   └── backends/       # Backend implementations (future)
│       ├── scalar.rs
│       ├── simd/
│       │   ├── avx2.rs
│       │   ├── avx512.rs
│       │   └── neon.rs
│       ├── gpu.rs
│       └── wasm.rs
├── benches/            # Criterion benchmarks (future)
├── docs/
│   └── specifications/ # Design specifications
├── Cargo.toml          # Dependencies, optimization flags
├── Makefile            # Quality gates, development commands
└── README.md           # This file
```

## Roadmap

### Phase 1: Scalar Baseline ✅ COMPLETE
- [x] Core `Vector<f32>` API (add, mul, dot, sum, max)
- [x] Error handling with `TruenoError`
- [x] 100% test coverage (40 tests)
- [x] Property-based tests (PROPTEST_CASES=100)
- [x] PMAT quality gates integration
- [x] Documentation and README

### Phase 2: x86 SIMD ✅ COMPLETE
- [x] Runtime CPU feature detection (`is_x86_feature_detected!`)
- [x] SSE2 implementation (baseline x86_64)
- [x] Benchmarks proving ≥10% speedup (66.7% of tests, avg 178.5%)
- [x] Auto-dispatch based on CPU features
- [x] Backend trait architecture
- [x] Comprehensive performance analysis

### Phase 3: AVX2 SIMD ✅ COMPLETE
- [x] AVX2 implementation with FMA support (256-bit SIMD)
- [x] Benchmarks proving exceptional speedups (1.82x for dot product)
- [x] Performance analysis and documentation
- [x] All quality gates passing (0 warnings, 78 tests)

### Phase 4: ARM SIMD ✅ COMPLETE
- [x] ARM NEON implementation (128-bit SIMD)
- [x] Runtime feature detection (ARMv7/ARMv8/AArch64)
- [x] Cross-platform compilation support
- [x] Comprehensive tests with cross-validation
- [ ] Benchmarks on ARM hardware (pending ARM access)

### Phase 5: WebAssembly ✅ COMPLETE
- [x] WASM SIMD128 implementation (128-bit SIMD)
- [x] All 5 operations with f32x4 intrinsics
- [x] Comprehensive tests with cross-validation
- [ ] Browser deployment example (future)
- [ ] Edge computing use case (future)

### Phase 6: GPU Compute ✅ COMPLETE
- [x] `wgpu` integration (optional `gpu` feature flag)
- [x] Compute shader kernels (WGSL): matmul, vec_add, dot product
- [x] Host-device memory transfer with async execution
- [x] GPU dispatch heuristics (>1000×1000 for matmul)
- [x] Automatic fallback to SIMD/CPU if GPU unavailable
- [x] Vector operations on GPU (vec_add, dot product with parallel reduction)
- [x] Performance benchmarks (GPU vs Scalar baseline validation)
- [ ] Multi-GPU support (deferred to future phase)
- [ ] GPU reductions (sum, max, min) (deferred to future phase)

**Phase 6 Status**: ⚠️ COMPLETE WITH CRITICAL FINDINGS - Full GPU compute backend with wgpu (Vulkan/Metal/DX12/WebGPU) implemented and **empirically validated**. Comprehensive benchmarks ([docs/performance-analysis.md](docs/performance-analysis.md)) show GPU is ONLY beneficial for matmul (2-10x for 500×500+). **All vector operations show massive slowdown** (2-65,000x) due to 14-55ms PCIe transfer overhead dominating memory-bound ops. GPU backend disabled for element-wise operations in v0.2.1+. 765 tests passing. WGSL shaders implemented with async execution and CPU fallback.

### Phase 7: Advanced Operations ✅ COMPLETE
- [x] Element-wise subtraction (sub) and division (div)
- [x] Reductions: min, max, sum, sum_kahan (Kahan summation)
- [x] Index finding: argmax, argmin
- [x] Vector norms: norm_l2 (Euclidean norm), normalize (unit vector)
- [x] Activation functions: ReLU, Leaky ReLU, ELU, Sigmoid, Softmax/Log-Softmax, GELU, Swish/SiLU
- [x] Preprocessing: zscore, minmax_normalize, clip
- [x] Statistical operations: mean, variance, stddev, covariance, correlation

### Phase 8: Matrix Operations ✅ COMPLETE
- [x] Matrix<T> type with row-major storage (NumPy-compatible)
- [x] Matrix multiplication (matmul) - naive O(n³)
- [x] Matrix transpose
- [x] Matrix-vector operations (matvec, vecmat)
- [x] Comprehensive examples (matrix_operations.rs)
- [x] SIMD-optimized matmul (Vector::dot with transpose optimization)
- [x] Backend equivalence tests (naive vs SIMD)
- [x] GPU dispatch for large matrices (>1000×1000 with wgpu)

**Phase 8 Status**: ✅ COMPLETE - Full matrix operations with 3-tier backend selection. 759 tests passing (637 lib + 19 integration + 103 bench). Matrix multiplication automatically selects optimal backend: **GPU for ≥500×500 matrices (empirical: 2-10x speedup)**, SIMD for >64×64 matrices (2-8x speedup), naive for smaller matrices (minimal overhead). GPU backend uses wgpu with WGSL compute shaders (16×16 workgroups), async execution via pollster, and graceful CPU fallback. See [performance-analysis.md](docs/performance-analysis.md) for complete empirical validation.

**Phase 7 Status**: ✅ COMPLETE - Core vector operations with 587 tests passing. The library now supports:
- **Element-wise operations**: add, sub, mul, div, abs (absolute value), neg (negation/unary minus), clamp (range constraint), lerp (linear interpolation), fma (fused multiply-add), sqrt (square root), recip (reciprocal), pow (power), exp (exponential), ln (natural logarithm), sin (sine), cos (cosine), tan (tangent), asin (arcsine), acos (arccosine), atan (arctangent), sinh (hyperbolic sine), cosh (hyperbolic cosine), tanh (hyperbolic tangent), asinh (inverse hyperbolic sine), acosh (inverse hyperbolic cosine), atanh (inverse hyperbolic tangent), floor (round down), ceil (round up), round (round to nearest), trunc (truncate toward zero), fract (fractional part), signum (sign function), copysign (copy sign from one vector to another), minimum (element-wise minimum of two vectors), maximum (element-wise maximum of two vectors)
- **Scalar operations**: scale (scalar multiplication with full SIMD support)
- **Dot product**: Optimized for ML/scientific computing
- **Reductions**: sum (naive + Kahan), min, max, sum_of_squares, mean (arithmetic average), variance (population variance), stddev (standard deviation), covariance (population covariance between two vectors), correlation (Pearson correlation coefficient)
- **Activation functions**: relu (rectified linear unit - max(0, x)), leaky_relu (leaky ReLU with configurable negative slope), elu (exponential linear unit with smooth gradients), sigmoid (logistic function - 1/(1+e^-x)), softmax (convert logits to probability distribution), log_softmax (numerically stable log of softmax for cross-entropy loss), gelu (Gaussian Error Linear Unit - smooth activation used in transformers like BERT/GPT), swish/silu (Swish/Sigmoid Linear Unit - self-gated activation used in EfficientNet/MobileNet v3)
- **Preprocessing**: zscore (z-score normalization/standardization), minmax_normalize (min-max scaling to [0,1] range), clip (constrain values to [min,max] range)
- **Index operations**: argmin, argmax
- **Vector norms**: L1 (Manhattan), L2 (Euclidean), L∞ (max norm), normalization to unit vectors
- **Numerical stability**: Kahan summation for accurate floating-point accumulation
- **FMA optimization**: Hardware-accelerated fused multiply-add on AVX2 and NEON platforms
- **Mathematical functions**: Element-wise square root, reciprocal, power, exponential, logarithm, trigonometric (sine, cosine, tangent), inverse trigonometric (arcsine, arccosine, arctangent), hyperbolic functions (sinh, cosh, tanh), and inverse hyperbolic functions (asinh, acosh, atanh) for ML (neural network activations), signal processing (waveforms, oscillators, phase recovery, FM demodulation), physics simulations, graphics (perspective projection, inverse transformations, lighting models, camera orientation), navigation (GPS, spherical trigonometry, bearing calculations, heading calculations), robotics (orientation calculations, inverse kinematics, steering angles), and Fourier analysis

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Quality Gates**: All PRs must pass `make quality-gates`
   - Zero clippy warnings
   - 100% formatted code
   - All tests passing
   - Coverage >85%

2. **Testing**: Include tests for new features
   - Unit tests for basic functionality
   - Property tests for mathematical operations
   - Benchmarks for performance claims

3. **Documentation**: Update README and docs for new features

4. **Toyota Way Principles**:
   - **Jidoka** (built-in quality): Tests catch issues immediately
   - **Kaizen** (continuous improvement): Every PR makes the codebase better
   - **Genchi Genbutsu** (go and see): Benchmark claims, measure reality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Pragmatic AI Labs** - [https://github.com/paiml](https://github.com/paiml)

## Acknowledgments

- Inspired by NumPy, Eigen, and ndarray
- SIMD guidance from `std::arch` documentation
- GPU compute via `wgpu` project
- Quality standards from Toyota Production System
- PMAT quality gates by Pragmatic AI Labs

## Citation

If you use Trueno in academic work, please cite:

```bibtex
@software{trueno2025,
  title = {Trueno: Multi-Target High-Performance Compute Library},
  author = {Pragmatic AI Labs},
  year = {2025},
  url = {https://github.com/paiml/trueno}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/paiml/trueno/issues)
- **Email**: contact@paiml.com

---

**Built with EXTREME TDD and Toyota Way principles** 🚗⚡
