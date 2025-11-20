# Trueno Examples

This directory contains examples demonstrating Trueno's high-performance compute capabilities and comparisons with NumPy/PyTorch.

## 📁 Examples Overview

### Rust Examples (Trueno Native)

| Example | Description | Command |
|---------|-------------|---------|
| `performance_demo.rs` | Compare Scalar vs SSE2/AVX backends | `cargo run --release --example performance_demo` |
| `matrix_operations.rs` | Matrix multiplication and transpose | `cargo run --release --example matrix_operations` |
| `activation_functions.rs` | Neural network activations (ReLU, Sigmoid, etc.) | `cargo run --release --example activation_functions` |
| `backend_detection.rs` | Auto-detection of available SIMD backends | `cargo run --release --example backend_detection` |
| `ml_similarity.rs` | Cosine similarity for ML applications | `cargo run --release --example ml_similarity` |

### Python Examples (NumPy/PyTorch Comparison)

| Example | Description | Command |
|---------|-------------|---------|
| `dot_product_comparison.py` | **⚡ Dot product benchmark** | `uv run examples/dot_product_comparison.py` |
| `matrix_multiply_comparison.py` | **🔢 Matrix multiplication benchmark** | `uv run examples/matrix_multiply_comparison.py` |
| `activation_comparison.py` | **🧠 Activation functions benchmark** | `uv run examples/activation_comparison.py` |

## 🚀 Quick Start

### Running Python Comparisons

1. Install UV (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Run any comparison example (UV handles dependencies automatically):
```bash
uv run examples/dot_product_comparison.py
uv run examples/matrix_multiply_comparison.py
uv run examples/activation_comparison.py
```

Alternatively, use the benchmarks environment:
```bash
cd benchmarks
uv run ../examples/dot_product_comparison.py
```

### Running Rust Examples

```bash
cargo run --release --example performance_demo
cargo run --release --example matrix_operations
```

## 📊 What Do These Examples Show?

### Dot Product Comparison (`dot_product_comparison.py`)

**Key Insights:**
- Demonstrates compute-intensive operations where SIMD excels
- Shows NumPy vs PyTorch performance characteristics
- Highlights Trueno's 1.6x advantage over NumPy (11.9x over scalar)

**Expected Output:**
```
Size          NumPy (μs)      PyTorch (μs)    Winner          Speedup
--------------------------------------------------------------------------------
100               0.82 ±  0.15      1.85 ±  0.22  NumPy           2.26x
1,000             3.21 ±  0.18      6.45 ±  0.31  NumPy           2.01x
10,000           25.67 ±  1.23     58.32 ±  2.45  NumPy           2.27x
```

**Trueno Context:**
- Trueno AVX-512: **11.9x faster than scalar**
- Trueno AVX-512: **1.6x faster than NumPy**
- Trueno AVX-512: **2.8x faster than PyTorch**

### Matrix Multiplication Comparison (`matrix_multiply_comparison.py`)

**Key Insights:**
- Shows O(n³) complexity scaling
- Demonstrates when GPU acceleration becomes effective
- Highlights optimized BLAS libraries in NumPy

**Expected Output:**
```
Size       NumPy Time           PyTorch Time         Winner       Speedup
------------------------------------------------------------------------------------------
64×64      59.87 μs             125.34 μs            NumPy          2.09x
128×128    434.23 μs            678.45 μs            NumPy          1.56x
256×256    2.67 ms              3.45 ms              NumPy          1.29x
512×512    19.82 ms             25.67 ms             NumPy          1.29x
```

**Trueno Context:**
- SIMD backend: **~7x faster than naive O(n³)** for 128×128
- GPU backend: **2-10x faster** than scalar for 500×500+
- Automatic backend selection based on matrix size

### Activation Functions Comparison (`activation_comparison.py`)

**Key Insights:**
- Compares common ML activation functions
- Shows relative costs (ReLU << Tanh < Sigmoid < Exp)
- Demonstrates SIMD benefits for transcendental functions

**Expected Output:**
```
Activation      NumPy (μs)      PyTorch (μs)    Winner       Speedup
------------------------------------------------------------------------------------------
ReLU                2.34 ±  0.12      5.67 ±  0.23  NumPy          2.42x
Sigmoid            15.67 ±  0.45     32.34 ±  1.12  NumPy          2.06x
Tanh                8.92 ±  0.34     18.45 ±  0.67  NumPy          2.07x
Exp                12.45 ±  0.56     28.91 ±  1.23  NumPy          2.32x
```

**Trueno Context:**
- SIMD-optimized implementations
- 2-4x speedup for compute-intensive activations
- Zero Python overhead for ML inference

## 🎯 Performance Summary

| Operation | Trueno vs Scalar | Trueno vs NumPy | Trueno vs PyTorch |
|-----------|-----------------|-----------------|-------------------|
| **Dot Product** | 11.9x faster | 1.6x faster | 2.8x faster |
| **Matrix Multiply** | 7x faster (128×128) | ~1x (competitive) | ~1.5x faster |
| **Activations** | 2-4x faster | ~1x (competitive) | ~2x faster |

## 💡 When to Use Trueno

✅ **Ideal Use Cases:**
- Real-time systems requiring predictable latency
- Embedded systems without Python runtime
- WebAssembly deployment (browser/edge)
- ML inference pipelines in Rust
- Systems programming with high-performance compute needs

⚠️ **When NumPy/PyTorch May Be Better:**
- Rapid prototyping in Python
- Large ecosystem of Python ML libraries
- Training large neural networks (PyTorch GPU)
- Interactive data exploration (Jupyter notebooks)

## 📚 More Resources

- **Comprehensive Benchmarks**: See `benchmarks/README.md`
- **Performance Analysis**: See `docs/performance-analysis.md`
- **API Documentation**: See `docs/` directory
- **Project README**: See root `README.md`

## 🤝 Contributing

To add new examples:

1. **Rust examples**: Add to this directory with `.rs` extension
2. **Python examples**: Add comparison scripts with NumPy/PyTorch
3. **Update this README**: Document the new example
4. **Follow TDD**: Ensure examples are well-tested

See `CLAUDE.md` for development guidelines.

---

**Last Updated**: 2025-11-20
**Version**: v0.4.0
**Contact**: [GitHub Issues](https://github.com/paiml/trueno/issues)
