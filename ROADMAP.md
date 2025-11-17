# Trueno Roadmap

## Current State: v0.2.0 (January 2025)

### What Trueno Is Today
High-performance 1D vector compute library with GPU acceleration for large-scale operations.

**Core Capabilities:**
- ✅ 1D `Vector<f32>` type
- ✅ CPU SIMD backends (SSE2/AVX/AVX2/NEON)
- ✅ GPU backend (wgpu: Vulkan/Metal/DX12/WebGPU)
- ✅ 14 GPU-accelerated operations
- ✅ Runtime dispatch (auto-select best backend)
- ✅ EXTREME TDD (>90% coverage, mutation testing)

**GPU Operations:**
- **OpComplexity::Low** (>100K threshold): vec_add, dot, relu, leaky_relu, elu, sigmoid, tanh, swish, GELU, clip
- **OpComplexity::Medium** (>10K threshold): softmax, log_softmax
- **OpComplexity::High** (>1K threshold): matmul, convolve2d

### "Drop-in" Replacement Analysis

#### vs NumPy: ~35% Complete
**What Works:**
- ✅ 1D element-wise ops (`+`, `-`, `*`, `/`, `exp`, `log`, etc.)
- ✅ Reductions (`sum`, `mean`, `std`, `min`, `max`)
- ✅ Linear algebra (`dot`, `matmul`, `norm`)
- ✅ GPU acceleration for large workloads

**Critical Gaps:**
- ❌ Multi-dimensional arrays (only 1D `Vector<f32>`)
- ❌ Broadcasting (`(3, 1) + (1, 5)` → `(3, 5)`)
- ❌ Advanced indexing/slicing
- ❌ Reshaping/transposing (`reshape`, `transpose`, `flatten`)
- ❌ Data types (only `f32`, no `i32`/`f64`/etc.)

**Viable NumPy Use Cases Today:**
- 1D signal processing
- Vector similarity/distance computations
- Large-scale element-wise transformations

#### vs PyTorch: ~15% Complete
**What Works:**
- ✅ GPU-accelerated activations (inference only)
- ✅ Basic tensor operations (1D only)
- ✅ Dot products, matrix multiplication

**Critical Blockers:**
- ❌ **No autograd** (can't train neural networks)
- ❌ No multi-dimensional tensors
- ❌ No layers (Linear, Conv2d, BatchNorm, etc.)
- ❌ No loss functions
- ❌ No optimizers (SGD, Adam, etc.)
- ❌ No broadcasting
- ❌ No model saving/loading

**Viable PyTorch Use Cases Today:**
- Neural network inference (forward pass only, 1D)
- Custom activation function testing
- Batch activation processing

---

## Phase 1: Complete 1D Operations (v0.2.x - v0.3.0)
**Timeline:** 2-3 months
**Goal:** Best-in-class 1D vector compute

### Immediate (v0.2.1 - Next 2 weeks)
- [x] **softmax/log_softmax GPU** (OpComplexity::Medium) - ✅ COMPLETE
  - **Implementation approach:**
    - Pass 1: Max reduction (parallel reduction shader) ✅
    - Pass 2: Exp and subtract max (element-wise shader) ✅
    - Pass 3: Sum reduction (parallel reduction shader) ✅
    - Pass 4: Normalize by sum (element-wise shader) ✅
  - **Key achievements:**
    - Multi-pass coordination (4 GPU dispatches via async/await)
    - Numerical stability (subtract max before exp)
    - Memory efficiency (staging buffers for intermediate results)
    - 5 WGSL shaders: max_reduction, sum_reduction, exp_subtract, normalize, log_softmax
    - GpuDevice methods + GpuBackend wrappers + Vector GPU dispatch
    - 18 tests pass (unit + property-based)
    - Benchmarks added (10K, 100K, 1M sizes)
    - Documentation complete (README with examples)
  - **Critical for:** Attention mechanisms, classification, transformers
  - **GPU threshold:** >10K elements (higher overhead than element-wise)
  - **Actual speedup:** 2-20x over scalar (validated by benchmarks)
- [ ] **Benchmark all GPU ops** - validate 10-50x claims
- [ ] **Performance regression suite** - prevent slowdowns

### Near-term (v0.2.2 - v0.2.5)
- [ ] **Remaining activations:**
  - [ ] hardswish (MobileNetV3)
  - [ ] mish (modern alternative to swish)
  - [ ] selu (self-normalizing networks)
- [ ] **More GPU reductions:**
  - [ ] argmax/argmin
  - [ ] sum/mean/std (GPU-accelerated)
- [ ] **GPU element-wise ops:**
  - [ ] add/sub/mul/div (binary ops)
  - [ ] exp/log/pow/sqrt (unary ops)

### v0.3.0 Goals
- [ ] **Async GPU API** - batch operations, reduce transfer overhead
- [ ] **CPU backend optimizations:**
  - [ ] AVX-512 support
  - [ ] Better auto-vectorization hints
- [ ] **WASM SIMD128** - browser deployment
- [ ] **Comprehensive benchmarks:**
  - [ ] vs NumPy (for 1D ops)
  - [ ] vs PyTorch (for activations)
  - [ ] Publish results in README

---

## Phase 2: Multi-Dimensional Tensors (v0.4.0 - v0.6.0)
**Timeline:** 6-12 months
**Goal:** NumPy-competitive for 2D/3D arrays

### v0.4.0: Tensor Type Foundation
- [ ] **`Tensor<T, const N: usize>`** - generic N-dimensional tensor
  - [ ] Row-major storage (C-contiguous)
  - [ ] Shape, strides, offset tracking
  - [ ] Views vs owned data
- [ ] **2D operations:**
  - [ ] Transpose, reshape, flatten
  - [ ] Row/column slicing
  - [ ] 2D matmul (optimized)

### v0.5.0: Broadcasting
- [ ] **Broadcasting rules** (NumPy-compatible)
  - [ ] Shape compatibility checking
  - [ ] Stride computation
  - [ ] Element-wise ops with broadcasting
- [ ] **Advanced indexing:**
  - [ ] Boolean masking
  - [ ] Integer array indexing
  - [ ] Slicing syntax (`[1:5, ::2]`)

### v0.6.0: NumPy Parity (Core Ops)
- [ ] **All NumPy dtypes:**
  - [ ] `f32`, `f64`, `i32`, `i64`, `u32`, etc.
  - [ ] Generic trait-based implementation
- [ ] **NumPy-style API:**
  - [ ] `zeros`, `ones`, `arange`, `linspace`
  - [ ] `concatenate`, `stack`, `split`
  - [ ] `where`, `argwhere`
- [ ] **Performance target:** 80-120% of NumPy speed

---

## Phase 3: Autograd & Training (v0.7.0 - v1.0.0)
**Timeline:** 12-18 months
**Goal:** PyTorch-competitive for training

### v0.7.0: Autograd Engine
- [ ] **Computational graph:**
  - [ ] Reverse-mode AD (backpropagation)
  - [ ] Dynamic graph construction
  - [ ] Gradient tape
- [ ] **Core operations with gradients:**
  - [ ] All element-wise ops
  - [ ] Matmul, conv2d
  - [ ] All activations
- [ ] **Memory optimization:**
  - [ ] Gradient checkpointing
  - [ ] In-place operations where safe

### v0.8.0: Neural Network Layers
- [ ] **nn::Module trait**
- [ ] **Core layers:**
  - [ ] Linear (fully connected)
  - [ ] Conv2d, MaxPool2d
  - [ ] BatchNorm, LayerNorm
  - [ ] Dropout
- [ ] **Loss functions:**
  - [ ] CrossEntropyLoss
  - [ ] MSELoss, BCELoss
  - [ ] Custom loss support

### v0.9.0: Optimizers
- [ ] **SGD** (with momentum, nesterov)
- [ ] **Adam** (with weight decay, AMSGrad)
- [ ] **AdamW**, **RMSprop**
- [ ] **Learning rate schedulers:**
  - [ ] StepLR, ExponentialLR
  - [ ] CosineAnnealing

### v1.0.0: Training-Ready
- [ ] **Model serialization:**
  - [ ] Save/load checkpoints
  - [ ] ONNX export
- [ ] **Distributed training:**
  - [ ] Data parallelism
  - [ ] Gradient synchronization
- [ ] **Performance target:** 60-80% of PyTorch speed
- [ ] **Full MNIST/CIFAR-10 examples**

---

## Phase 4: Production & Ecosystem (v1.x)
**Timeline:** 18-24 months
**Goal:** Production-grade deep learning library

### Ecosystem Integration
- [ ] **Ruchy transpiler:**
  - [ ] Auto-generate Trueno from Python
  - [ ] Depyler: NumPy → Trueno
  - [ ] Type-level optimization hints
- [ ] **ruchy-lambda:**
  - [ ] AWS Lambda deployment
  - [ ] Cold start optimization
- [ ] **Model zoo:**
  - [ ] ResNet, VGG, MobileNet
  - [ ] BERT, GPT-2 (transformer)
  - [ ] Pre-trained weights

### Advanced Features
- [ ] **Mixed precision training** (FP16/BF16)
- [ ] **Quantization** (INT8 inference)
- [ ] **Pruning & compression**
- [ ] **Custom CUDA kernels** (optional)
- [ ] **TensorRT integration**

---

## Non-Goals

**What Trueno Will NOT Be:**
- ❌ **Research-first library** - production performance is priority
- ❌ **Dynamic typing** - static typing for safety
- ❌ **Python-first** - Rust-native, Python bindings optional
- ❌ **100% PyTorch-compatible** - inspired by, not clone of
- ❌ **Symbolic computation** - eager execution only

---

## Success Metrics

### v0.3.0 (1D Complete)
- ✅ All common 1D operations GPU-accelerated
- ✅ 10-50x GPU speedup validated by benchmarks
- ✅ Used in production by ≥3 projects

### v0.6.0 (NumPy Parity)
- ✅ 80-120% of NumPy performance
- ✅ Pass 80%+ of NumPy test suite (for covered ops)
- ✅ ≥10 production deployments

### v1.0.0 (Training Ready)
- ✅ Train ResNet-50 on CIFAR-10 in <30 minutes (single GPU)
- ✅ 60-80% of PyTorch performance
- ✅ Used in production ML training pipelines

---

## Current Focus: v0.2.1

**Next Implementation:** GPU softmax/log_softmax

**Why softmax?**
1. **Critical for ML:** Every classification network uses softmax
2. **Attention mechanisms:** Transformers rely on softmax
3. **OpComplexity::Medium:** More challenging than element-wise (multi-pass reduction)
4. **Performance wins:** GPU can parallelize max/sum reductions

**After softmax:**
1. Benchmark all GPU ops (validate claims)
2. Remaining activations (hardswish, mish, selu)
3. GPU binary ops (add, mul, etc.)
4. Start v0.3.0 async GPU API design
